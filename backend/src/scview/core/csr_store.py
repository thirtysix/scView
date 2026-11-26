"""Column-stored (CSC) matrices become row-stored (CSR) when a file comes in.

A backed dataset whose X is CSC cannot be read by rows: anndata reads every stored value
for any block of cells, so markers, DE and every other row reader pay for the whole matrix
each time. We store every new dataset's X, layers and ``.raw`` X as CSR instead. The
values, their dtype and the cell and gene order stay as they are: only the layout changes,
and the result equals ``scipy.sparse.csc_matrix(...).tocsr()`` (stored zeros and repeated
entries kept, each cell's genes in order).

- ``in_memory(adata)``: a dataset read whole (the in-memory ingest). One matrix at a time,
  so the extra memory is one matrix's copy (``preflight.estimate_load`` counts it).
- ``convert(src, parent, name)``: one h5 group to another, in bounded memory. Pass 1 reads
  the stored row numbers and counts the values per cell (the CSR row pointers). Pass 2 reads
  the matrix once, a block of genes at a time, and appends each value to the file of its
  range of cells (temporary files beside the output, 8 bytes + the value per stored value).
  Pass 3 reads one range at a time, sorts it by cell with scipy's own kernel and appends it
  to the output. A matrix that fits one range skips the files. Peak memory is one range
  (``RANGE_BYTES``) plus one block, whatever the size of the matrix. The output is
  compressed as the source is (gzip costs 15 s against 4 s on 93M values, and every later
  read pays for it again), unless the caller names a compression.
- ``rewrite_file`` / ``copy_file``: a whole h5ad with its CSC matrices converted, everything
  else copied as stored. ``rewrite_dataset``: the same for a dataset already in the library,
  into its derived file.

A file in the anndata < 0.7 layout (``h5sparse_format``) is left as it is.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import time
import uuid
from pathlib import Path

import h5py
import numpy as np
import scipy.sparse as sp

from scview.core import h5_copy

logger = logging.getLogger(__name__)

CSC, CSR = "csc_matrix", "csr_matrix"
# Memory of one range of cells while it is sorted: its values, row and gene numbers as
# read, and the CSR arrays made from them.
RANGE_BYTES = 256 * 2**20
BLOCK_NNZ = 2_000_000  # stored values read per block of genes
GZIP = {"compression": "gzip", "compression_opts": 4}  # anndata's own default
_I32 = np.iinfo(np.int32).max


class CsrStoreError(RuntimeError):
    """A matrix that cannot be stored by rows; the message says why."""


def encoding(node) -> str:
    """The anndata encoding of an h5 node ("" for a group without one)."""
    e = node.attrs.get("encoding-type", "array" if isinstance(node, h5py.Dataset) else "")
    return e.decode() if isinstance(e, bytes) else str(e)


def is_csc(node) -> bool:
    return isinstance(node, h5py.Group) and encoding(node) == CSC and "indptr" in node


def matrices(f: h5py.Group) -> list[tuple[str, h5py.Group | h5py.Dataset]]:
    """The cells x genes matrices of an h5ad: X, each layer, ``.raw``'s X."""
    out = [("X", f["X"])] if "X" in f else []
    if isinstance(f.get("layers"), h5py.Group):
        out += [(f"layers/{k}", f["layers"][k]) for k in f["layers"].keys()]
    raw = f.get("raw")
    if isinstance(raw, h5py.Group) and "X" in raw:
        out.append(("raw/X", raw["X"]))
    return out


def csc_in_file(path: str | Path) -> list[str]:
    """Which of the file's matrices are stored CSC ([] when none, or not an HDF5 file)."""
    try:
        with h5py.File(path, "r") as f:
            return [where for where, node in matrices(f) if is_csc(node)]
    except (OSError, KeyError, ValueError):
        return []


# ---------------------------------------------------------------------------
# In memory
# ---------------------------------------------------------------------------


def _is_csc_memory(m) -> bool:
    return sp.issparse(m) and m.format == "csc"


def in_memory(adata) -> list[str]:
    """The dataset's CSC matrices (X, layers, ``.raw``) made CSR, one at a time: each copy
    replaces its matrix before the next is made. Returns which were changed."""
    import anndata as ad

    done = []
    if _is_csc_memory(adata.X):
        adata.X = adata.X.tocsr()
        done.append("X")
    for key in list(adata.layers.keys()):
        if _is_csc_memory(adata.layers[key]):
            adata.layers[key] = adata.layers[key].tocsr()
            done.append(f"layers/{key}")
    raw = adata.raw
    if raw is not None and _is_csc_memory(raw.X):
        adata.raw = ad.AnnData(X=raw.X.tocsr(), var=raw.var, varm=dict(raw.varm),
                               obs=adata.obs[[]].copy())
        done.append("raw/X")
    if done:
        logger.info("Stored by rows (CSR) instead of by columns: %s", ", ".join(done))
    return done


def csc_in_memory(adata) -> list[str]:
    """Which of a loaded dataset's matrices are CSC."""
    out = ["X"] if _is_csc_memory(adata.X) else []
    out += [f"layers/{k}" for k in adata.layers.keys() if _is_csc_memory(adata.layers[k])]
    if adata.raw is not None and _is_csc_memory(adata.raw.X):
        out.append("raw/X")
    return out


def write_h5ad(adata, path: str | Path, compression: str | None = "gzip") -> list[str]:
    """``adata.write_h5ad(path)`` for a NEW dataset made from one in memory (a subset, a
    joint embedding): a CSC matrix is stored CSR without a second copy of it in memory.
    The dataset is written as it is beside ``path`` (lzf), then ``rewrite_file`` converts
    that file a range of cells at a time. With no CSC matrix it is one plain write.
    Returns what was converted."""
    path = Path(path)
    if not csc_in_memory(adata):
        adata.write_h5ad(path, compression=compression)
        return []
    part = path.with_name(f".{path.name}.csc.part")
    try:
        adata.write_h5ad(part, compression="lzf")
        return rewrite_file(part, path, compression={"compression": compression} if compression else {})
    finally:
        part.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# On disk, one matrix
# ---------------------------------------------------------------------------


def index_dtype(nnz: int, n_vars: int) -> np.dtype:
    """As scipy picks it for a CSR in memory: int32 unless the values or genes need int64."""
    return np.dtype(np.int64 if max(nnz, n_vars) > _I32 else np.int32)


def temp_bytes(nnz: int, itemsize: int) -> int:
    """Disk ``convert`` uses for its temporary files (0 when the matrix fits one range)."""
    return 0 if nnz <= _range_values(itemsize) else nnz * (8 + itemsize)


def working_bytes(path: str | Path | None = None, n_obs: int = 0) -> int:
    """Memory ``convert`` holds at its peak: one range of cells, one block of genes as read
    and sorted to its ranges (about 96 bytes a value), and the row pointers and counts of
    the cells (``n_obs``, or read from the CSC matrices of ``path``). Measured on
    73,260 x 10,237 (93M float32 values): +0.41 GB."""
    if path is not None:
        try:
            with h5py.File(path, "r") as f:
                n_obs = max([int(node.attrs["shape"][0]) for _, node in matrices(f) if is_csc(node)] or [0])
        except (OSError, KeyError, ValueError):
            n_obs = 0
    return int(RANGE_BYTES + 96 * BLOCK_NNZ + 40 * n_obs)


def _range_values(itemsize: int, range_bytes: int | None = None) -> int:
    range_bytes = RANGE_BYTES if range_bytes is None else range_bytes
    # As read: row and gene numbers (4 + 4) and the value; sorted: gene number and value again.
    return max(1, range_bytes // (16 + 2 * itemsize))


def _blocks(indptr: np.ndarray, budget: int):
    """[start, end) ranges of the stored axis holding about ``budget`` values each."""
    n = len(indptr) - 1
    start = 0
    while start < n:
        end = int(np.searchsorted(indptr, indptr[start] + budget, side="right")) - 1
        end = min(n, max(end, start + 1))
        yield start, end
        start = end


class _Ranges:
    """The stored values of each range of cells, appended in gene order: in memory for one
    range, else in files under ``folder`` (rows, genes and values apart, 4 + 4 + itemsize
    bytes per value). Files are opened per write: a large matrix has hundreds of ranges."""

    def __init__(self, n: int, dtype: np.dtype, folder: Path | None):
        self.n, self.dtype, self.folder = n, dtype, folder
        self.kept: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    def _path(self, b: int, what: str) -> Path:
        return self.folder / f"{b}.{what}"

    def add(self, b: int, rows: np.ndarray, cols: np.ndarray, data: np.ndarray) -> None:
        if self.folder is None:
            self.kept.append((rows, cols, data))
            return
        for what, values in (("rows", rows), ("cols", cols), ("data", data)):
            with open(self._path(b, what), "ab") as fh:
                fh.write(np.ascontiguousarray(values).tobytes())

    def take(self, b: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.folder is None:
            got, self.kept = self.kept, []
            if not got:
                return np.zeros(0, np.int32), np.zeros(0, np.int32), np.zeros(0, self.dtype)
            return tuple(np.concatenate([g[i] for g in got]) for i in range(3))
        out = []
        for what, dtype in (("rows", np.int32), ("cols", np.int32), ("data", self.dtype)):
            p = self._path(b, what)
            out.append(np.fromfile(p, dtype=dtype) if p.exists() else np.zeros(0, dtype))
            p.unlink(missing_ok=True)
        return tuple(out)


def convert(src: h5py.Group, parent: h5py.Group, name: str, *, keep: np.ndarray | None = None,
            extract: list[np.ndarray] | tuple = (), workdir: str | Path | None = None,
            range_bytes: int | None = None, compression: dict | None = None) -> list[np.ndarray]:
    """Write the CSC matrix ``src`` (cells x genes) to ``parent[name]`` as CSR.

    ``keep``: a mask of the genes to write (all of them without it), numbered in order.
    ``extract``: masks of genes returned as dense float32 tables (cells x the mask's genes),
    as ``feature_split`` takes antibody and tag columns out. ``workdir``: where the temporary
    files go (default: beside the output file). ``compression``: h5py's dataset arguments
    (``GZIP``, ``{}`` for none); None: as the source's values are stored."""
    if compression is None:
        compression = _filters(src["data"])
    n_obs, n_var = (int(x) for x in src.attrs["shape"])
    indptr = src["indptr"][:].astype(np.int64)
    if len(indptr) != n_var + 1:
        raise CsrStoreError(f"{name}: {len(indptr) - 1} stored columns, {n_var} genes.")
    dtype = np.dtype(src["data"].dtype)
    remap = None
    if keep is not None:
        keep = np.asarray(keep, dtype=bool)
        remap = np.full(n_var, -1, dtype=np.int64)
        remap[keep] = np.arange(int(keep.sum()))
    n_out = n_var if keep is None else int(keep.sum())
    side = [(np.asarray(m, dtype=bool), np.zeros((n_obs, int(np.sum(m))), dtype=np.float32))
            for m in extract]
    side = [(m, _side_map(m), table) for m, table in side]

    # Pass 1: the stored values of each cell (kept genes only).
    counts = np.zeros(n_obs, dtype=np.int64)
    blocks = list(_blocks(indptr, BLOCK_NNZ))
    for start, end in blocks:
        lo, hi = int(indptr[start]), int(indptr[end])
        if keep is not None and not keep[start:end].any():
            continue
        rows = src["indices"][lo:hi].astype(np.int64, copy=False)
        if rows.size and (int(rows.min()) < 0 or int(rows.max()) >= n_obs):
            raise CsrStoreError(f"{name}: a stored row number is outside its {n_obs} cells.")
        if keep is not None:
            rows = rows[np.repeat(keep[start:end], np.diff(indptr[start:end + 1]))]
        counts += np.bincount(rows, minlength=n_obs)
    out_indptr = np.concatenate([[0], np.cumsum(counts)])
    nnz = int(out_indptr[-1])
    idx_dtype = index_dtype(nnz, n_out)

    out = parent.create_group(name)
    out.attrs["encoding-type"] = CSR
    out.attrs["encoding-version"] = "0.1.0"
    out.attrs["shape"] = np.asarray((n_obs, n_out), dtype=np.int64)
    chunk = (min(1 << 16, max(1, nnz)),)
    for key, dt in (("data", dtype), ("indices", idx_dtype)):
        out.create_dataset(key, shape=(nnz,), maxshape=(None,), dtype=dt, chunks=chunk, **compression)

    ranges = list(_blocks(out_indptr, _range_values(dtype.itemsize, range_bytes)))
    first_row = np.asarray([r0 for r0, _ in ranges], dtype=np.int64)
    folder = None
    try:
        if len(ranges) > 1:
            base = Path(workdir) if workdir is not None else Path(parent.file.filename).parent
            folder = Path(tempfile.mkdtemp(prefix=".csr-", dir=base))
        store = _Ranges(len(ranges), dtype, folder)

        # Pass 2: every stored value to its range of cells, in gene order.
        for start, end in blocks:
            lo, hi = int(indptr[start]), int(indptr[end])
            wanted = keep is None or bool(keep[start:end].any())
            taken = [s for s in side if s[0][start:end].any()]
            if hi == lo or not (wanted or taken):
                continue
            rows = src["indices"][lo:hi].astype(np.int64, copy=False)
            data = src["data"][lo:hi]
            cols = np.repeat(np.arange(start, end, dtype=np.int64), np.diff(indptr[start:end + 1]))
            for m, where, table in taken:
                sel = m[cols]
                np.add.at(table, (rows[sel], where[cols[sel]]), data[sel])
            if not wanted:
                continue
            if remap is not None:
                k = keep[cols]
                rows, data, cols = rows[k], data[k], remap[cols[k]]
            if len(ranges) == 1:
                store.add(0, rows.astype(np.int32), cols.astype(np.int32), data)
                continue
            which = (np.searchsorted(first_row, rows, side="right") - 1).astype(np.uint16 if len(ranges) <= 65535
                                                                                else np.int64)
            order = np.argsort(which, kind="stable")  # by range; gene order kept within one
            which = which[order]
            cuts = np.flatnonzero(np.diff(which)) + 1
            for a, b in zip(np.concatenate([[0], cuts]), np.concatenate([cuts, [which.size]])):
                r = int(which[a])
                sel = order[a:b]
                store.add(r, (rows[sel] - first_row[r]).astype(np.int32), cols[sel].astype(np.int32), data[sel])

        # Pass 3: each range sorted by cell (scipy's CSC to CSR kernel) and appended.
        for r, (r0, r1) in enumerate(ranges):
            rows, cols, data = store.take(r)
            lo, hi = int(out_indptr[r0]), int(out_indptr[r1])
            if rows.size != hi - lo:
                raise CsrStoreError(f"{name}: read {rows.size} values for cells {r0}-{r1}, counted {hi - lo}.")
            if hi == lo:
                continue
            col_ptr = np.concatenate([[0], np.cumsum(np.bincount(cols, minlength=n_out))])
            del cols
            piece = sp.csc_matrix((data, rows, col_ptr), shape=(r1 - r0, n_out)).tocsr()
            del rows, data
            out["data"][lo:hi] = piece.data
            out["indices"][lo:hi] = piece.indices.astype(idx_dtype, copy=False)
    finally:
        if folder is not None:
            shutil.rmtree(folder, ignore_errors=True)
    out.create_dataset("indptr", data=out_indptr.astype(idx_dtype), **compression)
    return [table for _, _, table in side]


def _filters(ds: h5py.Dataset) -> dict:
    """The compression of an h5 dataset as ``create_dataset`` arguments ({} for none)."""
    if ds.compression is None:
        return {}
    out = {"compression": ds.compression}
    if ds.compression_opts is not None:
        out["compression_opts"] = ds.compression_opts
    return out


def _side_map(mask: np.ndarray) -> np.ndarray:
    where = np.full(mask.shape[0], -1, dtype=np.int64)
    where[mask] = np.arange(int(mask.sum()))
    return where


# ---------------------------------------------------------------------------
# On disk, a group or a file
# ---------------------------------------------------------------------------


def copy_matrix(src, parent: h5py.Group, name: str, workdir: str | Path | None = None,
                compression: dict | None = None) -> bool:
    """``src`` into ``parent[name]``: converted when it is CSC, else copied as stored.
    Returns whether it was converted."""
    if is_csc(src):
        convert(src, parent, name, workdir=workdir, compression=compression)
        return True
    h5_copy.copy(src, parent, name)
    return False


def _group_like(src: h5py.Group, parent: h5py.Group, name: str) -> h5py.Group:
    g = parent.create_group(name)
    for k, v in src.attrs.items():
        g.attrs[k] = v
    return g


def copy_raw(raw: h5py.Group, parent: h5py.Group, workdir: str | Path | None = None,
             compression: dict | None = None) -> bool:
    """``.raw`` into ``parent["raw"]``, its X converted when it is CSC."""
    if not (isinstance(raw, h5py.Group) and "X" in raw and is_csc(raw["X"])):
        h5_copy.copy(raw, parent, "raw")
        return False
    g = _group_like(raw, parent, "raw")
    for key in raw.keys():
        if key == "X":
            convert(raw["X"], g, "X", workdir=workdir, compression=compression)
        else:
            h5_copy.copy(raw[key], g, key)
    return True


def rewrite_file(src: str | Path, dst: str | Path, workdir: str | Path | None = None,
                 compression: dict | None = None) -> list[str]:
    """Write ``dst``: the h5ad ``src`` with its CSC X, layers and ``.raw`` X as CSR and
    everything else copied as stored. ``src`` is only read. ``compression`` of the converted
    matrices: as each is stored in ``src`` unless given. Returns what was converted."""
    done = []
    with h5py.File(src, "r") as fi, h5py.File(dst, "w") as fo:
        for k, v in fi.attrs.items():
            fo.attrs[k] = v
        for key in fi.keys():
            node = fi[key]
            if key == "X":
                if copy_matrix(node, fo, "X", workdir, compression):
                    done.append("X")
            elif key == "layers" and isinstance(node, h5py.Group):
                g = _group_like(node, fo, "layers")
                for name in node.keys():
                    if copy_matrix(node[name], g, name, workdir, compression):
                        done.append(f"layers/{name}")
            elif key == "raw" and isinstance(node, h5py.Group):
                if copy_raw(node, fo, workdir, compression):
                    done.append("raw/X")
            else:
                h5_copy.copy(node, fo, key)
    return done


def copy_file(src: str | Path, dst: str | Path) -> list[str]:
    """A writable copy of the h5ad ``src`` at ``dst`` with no CSC matrix: a plain file copy
    when it has none, else ``rewrite_file``. Returns what was converted."""
    if not csc_in_file(src):
        shutil.copyfile(src, dst)
        Path(dst).chmod(0o644)
        return []
    return rewrite_file(src, dst)


# ---------------------------------------------------------------------------
# A dataset already in the library
# ---------------------------------------------------------------------------


def rewrite_dataset(dm, dataset_id: str) -> dict:
    """Rewrite one dataset's CSC matrices as CSR into its derived file.

    Call it inside ``obs_writes.dataset_edit`` (``edit.run``): nothing else may save the
    dataset meanwhile. The active file is read and never changed when it is an upload, an
    ingest or a conversion; the new file is written beside the derived one and swapped in
    (``os.replace``), so a kill mid-write leaves the dataset as it was. Nothing is read
    into memory whole. The caller drops the cached copy afterwards (``dm.evict``, as
    ``rewrite_dataset_held`` does), so the next open reads the new file. Returns what was converted, the seconds and the file sizes; ``converted`` is empty
    (and nothing written) for a dataset with no CSC matrix."""
    t0 = time.monotonic()
    source = dm._resolve_h5ad(dataset_id)
    if source is None:
        raise CsrStoreError(f"Dataset {dataset_id} has no h5ad file.")
    where = csc_in_file(source)
    before = os.path.getsize(source)
    if not where:
        return {"converted": [], "seconds": 0.0, "source": source, "bytes_before": before,
                "bytes_after": before}
    target = dm.derived_h5ad_path(dataset_id, source)
    partial = target.with_name(f".{target.name}.{uuid.uuid4().hex[:8]}.part")  # one per writer
    try:
        done = rewrite_file(source, partial, workdir=target.parent)
        os.replace(partial, target)
    finally:
        partial.unlink(missing_ok=True)
    out = {"converted": done, "seconds": round(time.monotonic() - t0, 1), "source": source,
           "target": str(target), "bytes_before": before, "bytes_after": os.path.getsize(target)}
    logger.info("Dataset %s stored by rows (CSR): %s in %.1f s", dataset_id, ", ".join(done), out["seconds"])
    return out


async def rewrite_dataset_held(dm, dataset_id: str) -> dict:
    """``rewrite_dataset`` in a worker thread holding the dataset's edit lock, then the
    cached copy dropped. Not called by the app yet: when to run it is a product decision."""
    from scview.core.obs_writes import dataset_edit

    async with dataset_edit(dataset_id) as held:
        out = await held.run(rewrite_dataset, dm, dataset_id)
        if out["converted"]:
            dm.evict(dataset_id)
    return out
