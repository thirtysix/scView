"""A large dataset in the viewer: a processed dataset whose matrices stay on disk.

The files are described in ``core/large_layout.py``: the h5ad holds X (log-normalised
CSR, every kept gene) beside obs, var, obsm, obsp, varm and uns; the counts and a
gene-major copy of X (the gene store) are sidecars. The adaptor opens the h5ad backed
whatever its size, so only the tables, embeddings, graphs and results are in memory,
and reads genes from the gene store (``core/anndata_adaptor.py``).

This module holds what the rest of the viewer needs for that:

- ``LargeDatasetError``: work that would read a whole matrix into memory is refused
  with a plain reason (a 422, ``main.py``), never attempted: at 1.3M cells X alone is
  20 GB.
- ``OnDiskMatrix``: the on-disk X (and counts) of a large dataset. Rows can be read,
  a bounded number at a time; a few of X's genes too, from the gene store (anndata's
  own column read of a CSR on disk loads the whole matrix); the whole matrix never.
  So an endpoint that was never taught about large datasets either works through it
  or fails with a reason, instead of taking the backend down.
- ``write_in_place``: an edit of obs or uns (labels, category order, colours, a kept
  clustering) written into the h5ad itself, without reading or copying its matrix.
- ``export_h5ad``: one self-contained h5ad (X, ``layers['counts']``, optionally
  ``.raw``) copied from the files chunk by chunk, never through memory.
"""

from __future__ import annotations

import logging
import os
import shutil
import threading
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from anndata._core.file_backing import AnnDataFileManager
from anndata._core.sparse_dataset import _CSRDataset

from scview.core import large_layout

logger = logging.getLogger(__name__)

MiB = 1 << 20

# Values (plus their column indices, 8 bytes in all) one read of an on-disk matrix of a
# large dataset may bring into memory: a row sample, a block of a chunked pass. 512 MiB
# is ~33k cells of the 1.3M-cell brain; the whole of it is 20 GB.
MAX_READ_BYTES = int(float(os.environ.get("SCVIEW_LARGE_READ_MB", "512") or 512) * MiB)

# Genes one read of a large dataset's X may ask for (served by the gene store): a few
# for an overlay, a dot plot or a check; not a whole-matrix method.
MAX_COLUMNS = int(os.environ.get("SCVIEW_LARGE_READ_GENES", "256") or 256)

# Memory a large dataset holds beyond its tables (preflight counts those): the gene
# store's open file (16 MiB chunk cache, the gene pointer), the counts sidecar's and the
# h5ad's file handles, and the row pointer cached for read checks (8 bytes per cell).
HELD_FIXED = 48 * MiB
HELD_PER_CELL = 8 + 8


class LargeDatasetError(Exception):
    """Work that is not done on a large dataset (its matrices stay on disk): a 422 with
    ``what`` is not available for large datasets yet, and why."""

    def __init__(self, what: str, why: str) -> None:
        self.what = what
        self.why = why.rstrip(".")
        super().__init__(f"{what} is not available for large datasets yet: {self.why}.")


def refuse(what: str, why: str):
    raise LargeDatasetError(what, why)


def is_large(obj: Any) -> bool:
    """Whether an adaptor, an AnnData or an uns dict is a large dataset."""
    flag = getattr(obj, "is_large", None)
    if isinstance(flag, bool):
        return flag
    return large_layout.is_large(obj)


def refusal_in(exc: BaseException | None) -> LargeDatasetError | None:
    """The ``LargeDatasetError`` an exception was raised while handling (its cause or
    context chain), if any: an endpoint that turns any failure into a 500 then still
    answers with the refusal."""
    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        if isinstance(exc, LargeDatasetError):
            return exc
        seen.add(id(exc))
        exc = exc.__cause__ or exc.__context__
    return None


def check(obj: Any, what: str, why: str) -> None:
    """Refuse ``what`` (``LargeDatasetError``) when ``obj`` is a large dataset."""
    if is_large(obj):
        refuse(what, why)


# The reasons most refusals give, so the messages read alike.
WHOLE_MATRIX = "it needs every cell's values for many genes in memory at once"
DENSE_MATRIX = "it works on a dense copy of the expression matrix"
COUNTS_MATRIX = "it needs the whole counts matrix in memory"
REPROCESS = "it reprocesses the dataset in memory; large datasets are processed in chunks by the pipeline"


def file_is_large(path: str | Path) -> bool:
    """Whether the h5ad at ``path`` is a large dataset (``uns['scview_large']``), read
    from its header alone."""
    try:
        with h5py.File(path, "r") as f:
            uns = f.get("uns")
            return isinstance(uns, h5py.Group) and isinstance(uns.get(large_layout.KEY), h5py.Group)
    except (OSError, KeyError):
        return False


# ---------------------------------------------------------------------------
# The on-disk matrices


class _Guard:
    """Shared by every ``OnDiskMatrix`` of one file: the lock that orders reads against
    an in-place write (which closes and reopens the file), and the row pointer."""

    def __init__(self, lock: threading.RLock) -> None:
        self.lock = lock
        self.indptr: np.ndarray | None = None
        self.fm: AnnDataFileManager | None = None  # None: a sidecar
        self.h5: h5py.File | None = None
        self.key = "X"
        self.label = "the expression matrix"
        # X's genes, read from the gene store: (column positions, cell positions or None)
        # -> sparse cells x genes. None: columns cannot be read (the counts sidecar).
        self.columns: Callable[[np.ndarray, np.ndarray | None], Any] | None = None
        # The counts sidecar's matrix, for helpers that look for a dataset's counts.
        self.counts: Callable[[], Any] | None = None

    def group(self) -> h5py.Group:
        f = self.fm._file if self.fm is not None else self.h5
        return f[self.key]


def _unpack(index: Any) -> tuple[Any, Any]:
    if isinstance(index, tuple):
        if len(index) == 0:
            return slice(None), slice(None)
        if len(index) == 1:
            return index[0], slice(None)
        if len(index) == 2:
            return index
        raise IndexError("A matrix takes at most two indices")
    return index, slice(None)


def _positions(idx: Any, n: int) -> np.ndarray:
    """A 1-D index (slice, int, mask, list) as integer positions."""
    if isinstance(idx, slice):
        return np.arange(*idx.indices(n), dtype=np.int64)
    if isinstance(idx, (int, np.integer)):
        return np.array([int(idx) + (n if idx < 0 else 0)], dtype=np.int64)
    arr = np.asarray(idx).ravel()
    if arr.dtype == bool:
        return np.flatnonzero(arr)
    arr = arr.astype(np.int64)
    return np.where(arr < 0, arr + n, arr)


def _all_of(idx: Any, n: int) -> bool:
    """Whether a 1-D index selects every position 0..n-1 in order."""
    if isinstance(idx, slice):
        return range(*idx.indices(n)) == range(n)
    arr = np.asarray(idx)
    if arr.dtype == bool:
        return arr.size == n and bool(arr.all())
    arr = arr.ravel()
    return arr.size == n and bool(np.array_equal(arr, np.arange(n)))


class OnDiskMatrix(_CSRDataset):
    """A large dataset's cell-major matrix on disk (``X``, or the counts sidecar's).

    Reads of whole rows work, up to ``MAX_READ_BYTES`` of values at a time: a sample
    of cells, a block of a chunked pass. Up to ``MAX_COLUMNS`` of X's genes (of any
    cells) come from the gene store. Anything more, columns of the counts, and the
    whole matrix (``to_memory``) raise ``LargeDatasetError``: whole-matrix work is
    done in chunks or not at all."""

    _guard: _Guard

    @classmethod
    def made(cls, guard: _Guard) -> OnDiskMatrix:
        m = cls(guard.group())
        m._guard = guard
        return m

    # anndata's own reads go through ``_group``; after an in-place write reopened the
    # file the old group handle is closed, so it is looked up again.
    def _fresh(self) -> None:
        if not self._group.id.valid:
            self._group = self._guard.group()

    def _row_values(self, row: Any) -> int:
        g = self._guard
        if g.indptr is None:
            g.indptr = np.asarray(self._group["indptr"][:], dtype=np.int64)
        lengths = None
        n = self.shape[0]
        if isinstance(row, slice):
            start, stop, step = row.indices(n)
            if step == 1:
                return int(g.indptr[max(stop, start)] - g.indptr[start])
            lengths = np.diff(g.indptr)[start:stop:step]
        elif isinstance(row, (int, np.integer)):
            return int(g.indptr[int(row) + 1] - g.indptr[int(row)])
        else:
            arr = np.asarray(row)
            lengths = np.diff(g.indptr)[arr.ravel()] if arr.dtype != bool else np.diff(g.indptr)[arr]
        return int(lengths.sum())

    def __getitem__(self, index: Any):
        row, col = _unpack(index)
        if isinstance(row, Iterable) and not isinstance(row, (str, bytes)) and np.ndim(row) > 1:
            row = np.asarray(row).ravel()  # np.ix_ form
        if isinstance(col, Iterable) and np.ndim(col) > 1:
            col = np.asarray(col).ravel()
        n_cells, n_genes = self.shape
        every_gene = _all_of(col, n_genes)
        with self._guard.lock:
            self._fresh()
            values = self._row_values(row)
            fits = values * 8 <= MAX_READ_BYTES
            if every_gene or (fits and not _all_of(row, n_cells)):
                if not fits:
                    raise LargeDatasetError(
                        f"Reading this many cells of {self._guard.label} at once",
                        f"about {values * 8 / 2**30:.1f} GB would be read into memory "
                        f"(at most {MAX_READ_BYTES / 2**30:.2g} GB per read)")
                if every_gene:
                    return super().__getitem__(index)
                # A sample of cells (an assessment's first rows): read them, cut the
                # genes in memory.
                block = super().__getitem__((row, slice(None)))
                return block[:, col]
        return self._columns(row, col, n_cells, n_genes)

    def _columns(self, row: Any, col: Any, n_cells: int, n_genes: int):
        """Some genes (of some cells): from the gene store, never by reading every row
        (anndata's own column read loads the whole matrix)."""
        import scipy.sparse as sp

        source = self._guard.columns
        if source is None:
            raise LargeDatasetError(
                f"Reading genes from {self._guard.label}",
                "it is stored cell by cell, so every cell would be read into memory")
        pos = _positions(col, n_genes)
        if pos.size > MAX_COLUMNS:
            raise LargeDatasetError(
                f"Reading {pos.size:,} genes at once",
                f"genes come from the gene-major copy, at most {MAX_COLUMNS} at a time; "
                "this view needs them all in memory")
        cells = None if _all_of(row, n_cells) else _positions(row, n_cells)
        block = sp.csr_matrix(source(pos, cells))
        return block[0] if isinstance(row, (int, np.integer)) else block

    def to_memory(self):
        raise LargeDatasetError(
            f"Reading {self._guard.label} into memory",
            f"it holds {self.shape[0]:,} cells x {self.shape[1]:,} genes and stays on disk")

    def rows(self, start: int, stop: int):
        """Rows ``start:stop`` as a scipy CSR (one guarded read)."""
        return self[start:stop]

    def row_blocks(self, max_values: int | None = None):
        """(first row, CSR block) over every row in order, each block holding about
        ``max_values`` values (default a quarter of ``MAX_READ_BYTES``)."""
        limit = max(1, (max_values or MAX_READ_BYTES // 32))
        with self._guard.lock:
            self._fresh()
            self._row_values(0)  # the row pointer, cached
            indptr = self._guard.indptr
        n = self.shape[0]
        start = 0
        while start < n:
            stop = int(np.searchsorted(indptr, int(indptr[start]) + limit, side="right")) - 1
            stop = min(max(stop, start + 1), n)
            yield start, self[start:stop]
            start = stop


class _GuardedFileManager(AnnDataFileManager):
    """A large dataset's backing-file manager: X comes back as an ``OnDiskMatrix``,
    and every open or lookup waits for an in-place write that holds the lock."""

    _guard: _Guard

    def __getitem__(self, key: str):
        with self._guard.lock:
            if key == "X":
                return OnDiskMatrix.made(self._guard)
            return super().__getitem__(key)

    def open(self, file_name=None, filemode=None):
        with self._guard.lock:
            if self.is_open and file_name is None and filemode is None:
                return
            super().open(file_name, filemode)


def guard_backed(adata, lock: threading.RLock) -> None:
    """Make a backed AnnData of a large dataset hand out its X as an ``OnDiskMatrix``."""
    guard = _Guard(lock)
    fm = adata.file
    fm.__class__ = _GuardedFileManager
    fm._guard = guard
    guard.fm = fm


def counts_of(adata):
    """A large dataset's counts (the sidecar's ``OnDiskMatrix``, row reads only), for the
    helpers that look for a dataset's counts; None for any other AnnData. Raises
    ``LargeDatasetError`` when the sidecar is missing or holds other cells."""
    guard = getattr(getattr(adata, "file", None), "_guard", None)
    source = getattr(guard, "counts", None)
    return source() if source is not None else None


def open_matrix(path: str | Path, key: str, lock: threading.RLock, label: str):
    """(open h5py file, ``OnDiskMatrix``) of a CSR matrix in an h5 file (a sidecar)."""
    f = h5py.File(path, "r", rdcc_nbytes=8 * MiB)
    try:
        guard = _Guard(lock)
        guard.h5, guard.key, guard.label = f, key, label
        return f, OnDiskMatrix.made(guard)
    except BaseException:
        f.close()
        raise


def held_bytes(n_cells: int) -> int:
    """What a large dataset holds beyond what preflight counts for a backed open."""
    return HELD_FIXED + HELD_PER_CELL * int(n_cells)


# ---------------------------------------------------------------------------
# Edits written in place


def _open_for_write(path: str | Path, tries: int = 20, wait: float = 0.25) -> h5py.File:
    """The h5ad opened for writing; another read handle open at the moment (a header
    read) makes HDF5 refuse, so it is tried again for a few seconds."""
    for i in range(tries):
        try:
            return h5py.File(path, "r+")
        except OSError:
            if i == tries - 1:
                raise
            time.sleep(wait)
    raise RuntimeError("unreachable")


def _replace(group: h5py.Group, key: str, value: Any) -> None:
    """``group[key] = value`` as anndata writes it: written under a temporary name and
    swapped in, so a failure leaves the old entry."""
    from anndata.io import write_elem

    tmp = f"{key}.__scview_new"
    if tmp in group:
        del group[tmp]
    write_elem(group, tmp, value)
    if key in group:
        del group[key]
    group.move(tmp, key)


def write_in_place(adata, path: str | Path, lock: threading.RLock, *, obs: bool = True,
                   uns_keys: Iterable[str] = (), uns_removed: Iterable[str] = (),
                   obsm_keys: Iterable[str] = ()) -> None:
    """Write a backed large dataset's edited obs, ``uns_keys`` (``uns_removed`` deleted)
    and ``obsm_keys`` into its h5ad, leaving X alone. Holds ``lock`` throughout: the
    backing file is closed (HDF5 opens a file for writing only once nothing in the
    process reads it), written, and opened again; reads of X wait meanwhile.

    HDF5 does not give the space of a replaced entry back, so the file grows by the
    size of obs at each edit (an export writes a compact copy)."""
    fm = adata.file
    with lock:
        fm.close()
        try:
            with _open_for_write(path) as f:
                if obs:
                    frame = adata.obs
                    if frame.index.name is not None and frame.index.name in frame.columns:
                        frame.index.name = None
                    _replace(f, "obs", frame)
                if "uns" not in f:
                    from anndata.io import write_elem

                    write_elem(f, "uns", {})
                group = f["uns"]
                for key in uns_removed:
                    if key in group and key not in adata.uns:
                        del group[key]
                for key in uns_keys:
                    if key in adata.uns:
                        _replace(group, key, adata.uns[key])
                    elif key in group:
                        del group[key]
                keys = list(obsm_keys)
                if keys and "obsm" not in f:
                    from anndata.io import write_elem

                    write_elem(f, "obsm", {})
                for key in keys:
                    if key in adata.obsm:
                        _replace(f["obsm"], key, adata.obsm[key])
                    elif key in f["obsm"]:
                        del f["obsm"][key]
        finally:
            fm.open()


def changed_uns(adata, before: dict[str, int] | None) -> tuple[list[str], list[str]]:
    """(keys set or replaced, keys removed) since ``obs_writes.uns_snapshot``."""
    if before is None:
        return list(adata.uns.keys()), []
    changed = [k for k, v in adata.uns.items() if before.get(k) != id(v)]
    removed = [k for k in before if k not in adata.uns]
    return changed, removed


# ---------------------------------------------------------------------------
# Export


def _copy_matrix(src: h5py.Group, dst: h5py.Group, name: str) -> None:
    """Copy one stored matrix group chunk by chunk (HDF5's own object copy: the
    compressed chunks are copied as they are, nothing is decompressed)."""
    src.file.copy(src, dst, name=name)
    for attr in [a for a in dst[name].attrs if str(a).startswith("scview_")]:
        del dst[name].attrs[attr]


def export_h5ad(adata, h5ad_path: str | Path, counts_path: str | Path | None, out_path: str | Path,
                *, raw: bool = False) -> Path:
    """One self-contained h5ad of a large dataset at ``out_path``: obs, var, obsm, obsp,
    varm and uns from the open dataset (``uns['scview_large']`` dropped: the copy has no
    sidecars), X copied from the h5ad, ``layers['counts']`` from the counts sidecar
    (when given) and, with ``raw``, ``.raw`` = X. Written beside ``out_path`` and moved
    into place."""
    import anndata as ad

    out_path = Path(out_path)
    tmp = out_path.with_name(f".{out_path.name}.part")
    uns = {k: v for k, v in adata.uns.items() if k != large_layout.KEY}
    meta = ad.AnnData(obs=adata.obs, var=adata.var, obsm=dict(adata.obsm), varm=dict(adata.varm),
                      obsp=dict(adata.obsp), uns=uns)
    try:
        meta.write_h5ad(tmp, compression="gzip")
        with h5py.File(tmp, "r+") as out, h5py.File(h5ad_path, "r") as src:
            if "X" in out:
                del out["X"]
            _copy_matrix(src["X"], out, "X")
            if counts_path is not None:
                with h5py.File(counts_path, "r") as cf:
                    layers = out.require_group("layers")
                    if "counts" in layers:
                        del layers["counts"]
                    _copy_matrix(cf[large_layout.COUNTS_KEY], layers, "counts")
            if raw:
                from anndata.io import write_elem

                if "raw" in out:
                    del out["raw"]
                grp = out.create_group("raw")  # as anndata writes a Raw
                grp.attrs["encoding-type"] = "raw"
                grp.attrs["encoding-version"] = "0.1.0"
                _copy_matrix(src["X"], grp, "X")
                write_elem(grp, "var", adata.var.iloc[:, :0])
                write_elem(grp, "varm", {})
        os.replace(tmp, out_path)
    finally:
        tmp.unlink(missing_ok=True)
    return out_path


def export_bytes(h5ad_path: str | Path, counts_path: str | Path | None, raw: bool) -> int:
    """About how big ``export_h5ad``'s file is: the h5ad, the counts and X again for raw."""
    size = Path(h5ad_path).stat().st_size
    if counts_path is not None and Path(counts_path).exists():
        size += Path(counts_path).stat().st_size
    if raw:
        with h5py.File(h5ad_path, "r") as f:
            x = f["X"]
            size += sum(x[k].id.get_storage_size() for k in x if isinstance(x[k], h5py.Dataset))
    return int(size)


# ---------------------------------------------------------------------------
# Sidecars left behind


def stray_sidecars(folder: str | Path) -> list[Path]:
    """Sidecar files in ``folder`` that no large dataset there names: left by a run that
    stopped before writing its h5ad, or by an h5ad since replaced by one processed in
    memory. Gene-store build folders of a process that is gone are removed on the way."""
    from scview.core import gene_store

    folder = Path(folder)
    if not folder.is_dir():
        return []
    from anndata.io import read_elem

    keep: set[str] = set()
    for h5ad in folder.glob("*.h5ad"):
        if file_is_large(h5ad):
            try:
                with h5py.File(h5ad, "r") as f:
                    info = read_elem(f["uns"][large_layout.KEY])
            except (OSError, KeyError):
                continue
            keep.update(str(info[w]) for w in ("counts", "genes") if info.get(w))
    stray = []
    stores: set[str] = set()
    for entry in folder.iterdir():
        name = entry.name
        if not entry.is_file():
            build = f"{large_layout.GENES_SUFFIX}.build-"
            if name.startswith(".") and build in name:
                stores.add(name[1:name.index(".build-")])
            continue
        if name.endswith((large_layout.COUNTS_SUFFIX, large_layout.GENES_SUFFIX)) and name not in keep:
            stray.append(entry)
        elif name.startswith(".") and name.endswith(".part"):  # an unfinished h5ad or counts write
            stray.append(entry)
    for name in stores:
        gene_store.remove_stale_builds(folder / name)
    return stray


def remove_paths(paths: Iterable[Path]) -> int:
    n = 0
    for p in paths:
        try:
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink()
            n += 1
        except OSError as e:
            logger.warning("Could not remove %s: %s", p, e)
    return n
