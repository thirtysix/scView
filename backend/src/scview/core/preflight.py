"""Memory and time estimates before a load or a pipeline run (preflight).

Before a file is loaded or a run starts, scView says how much memory it will need at
its peak and roughly how long it will take, against the memory it has free (the
container limit minus what it uses now). It warns and never refuses: the estimate
can be wrong, and the user decides.

- ``read_header(path)``: cells, genes, how X, the layers and .raw are stored (sparse
  with its stored values, or dense; dtypes), obsm, without reading the matrix: h5ad,
  10x .h5, MEX (matrix.mtx size line), loom (non-zeros sampled), zarr (v2 and v3
  metadata), CSV/TSV (rows estimated from the file size), RDS (a guess). Kang's 1.9 GB
  h5ad: 17 ms.
- ``estimate_load(header)``: what the dataset holds once open, and at the peak of
  reading it (and, for an ingest, of saving it as h5ad).
- ``estimate_run(shape | header, nnz, dense, steps, params)``: each step's memory over
  what the dataset holds when it starts, and seconds; the run's peak is the largest
  held + transient as the dataset's state changes through the steps (reset to counts
  replaces X, filtering leaves the opened copy held by the adaptor until the run ends,
  scaling makes X dense float32 and keeps the old X as .raw).
- ``estimate_step_peak(step, shape, nnz, dense, params, *, other_bytes=0,
  dtype="float32") -> int``: one step, pure, for a hook before each step;
  ``step_peak_for(adata, step, params)`` reads the same from a live AnnData.
- ``make_verdict``: fits (under 70% of the memory free), tight (70-100%), won't fit,
  with suggestions in this order: raise SCVIEW_MEMORY_GB, close the other datasets
  scView holds, skip doublet detection, fewer variable genes, subsample; each
  alternative is re-estimated and offered only when it lowers the peak.

Calibration

The coefficients come from measured runs, not guesses: one fresh container per
dataset, 2 CPUs (scview-backend image, 2026-10-21), memory = the cgroup's anon memory
(which counts the Leiden child process) sampled every 0.1 s, or the process's peak RSS
where that is higher (short reads), minus the process before loading (scanpy imported,
0.17 GiB). Runs are the "Reprocess from scratch" steps with the UI's marker columns.
GiB; err = estimate / measured - 1.

    dataset            cells   genes   X stored values      load meas/est  err   run peak meas/est  err  at
    pbmc 1k 10x .h5     1,222  33,538  2.5M sparse         0.04 / 0.04   -1%   0.63 / 0.72   +15%  (late steps)
    paul15             2,730   3,451  dense (+2 dense)    0.13 / 0.14   +5%   0.62 / 0.73   +17%  (late steps)
    Kang              13,836  14,053  dense f64 (+2 sp.)  1.76 / 1.74   -1%   3.00 / 2.99    -0%  doublets
    ovary 10k sub.    10,000  20,082  12.1M sp. + dense   0.91 / 0.88   -3%   6.17 / 6.22    +1%  doublets
    ovary 20k sub.    20,000  20,082  24.1M sp. + dense   1.75 / 1.72   -2%  11.73 / 11.27   -4%  doublets
    ovary_raw         37,104  20,082  44.7M sp. + dense   3.22 / 3.17   -2%  20.86 / 19.91   -5%  doublets

ovary_raw was killed at an 18 GiB limit (in doublet detection) and measured at 26 GiB
through PCA; its later steps are estimated at 16.5 GiB at most. Its counts layer is
dense (3 GB), and Scrublet copies the whole dataset, so doublet detection holds several
copies of it: 15 GiB over the 5.7 the dataset held. The small datasets are over by fixed
costs (library imports, caches). Ingest (read, save as gzip h5ad, reopen), peak:
pbmc 10x .h5 0.056 / 0.060 (+7%), the same as MEX 0.144 / 0.148 (+3%), a 1,222 x 5,000
CSV 0.189 / 0.203 (+7%), paul15 0.140 / 0.147 (+5%), Kang 1.79 / 1.91 (+7%). Not
measured: loom and zarr reads (8 and 0 bytes per value, reasoned), RDS (a guess), batch
correction, the uncorrected UMAP and cell-type annotation (reasoned from what they
allocate; the estimate says so). Header reads take 3-35 ms (Kang's 1.9 GB h5ad: 17 ms).

Time: the same runs' seconds per step, as a function of cells, stored values and
groups, measured at 2 CPUs and scaled by the square root of the CPU quota; shown as
half to twice the point estimate, which held every measured run (Kang 116 s within
70-279 s, ovary 20k 580 s within 264-1054 s).
"""

from __future__ import annotations

import gzip
import json
import os
import time
import zlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

GiB = 1 << 30
MiB = 1 << 20

_HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"
_GZIP_MAGIC = b"\x1f\x8b"
# Bytes of a variable-length string element (a pandas object column / index entry).
_STR_BYTES = 64


# ---------------------------------------------------------------------------
# Header: what a file holds, read without reading its matrix
# ---------------------------------------------------------------------------


@dataclass
class Matrix:
    """One cells x genes matrix (X, a layer, .raw) as it will sit in memory."""

    where: str  # "X", "layers/counts", "raw/X"
    n_cells: int
    n_genes: int
    sparse: bool
    nnz: int  # values stored: the non-zeros of a sparse matrix, every cell x gene of a dense one
    dtype: str = "float32"  # of the values, in memory
    index_dtype: str = "int32"  # sparse only
    nnz_estimated: bool = False  # sampled, not read (loom)

    @property
    def nbytes(self) -> int:
        item = np.dtype(self.dtype).itemsize
        if not self.sparse:
            return self.n_cells * self.n_genes * item
        idx = np.dtype(self.index_dtype).itemsize
        return self.nnz * (item + idx) + (self.n_cells + 1) * idx

    def to_dict(self) -> dict:
        d = asdict(self)
        d["bytes"] = self.nbytes
        return d


@dataclass
class Header:
    """A data file's shape and contents, read from its metadata only."""

    format: str  # h5ad | 10x_h5 | mex | loom | zarr | table | rds | unknown
    n_cells: int | None = None
    n_genes: int | None = None
    matrices: list[Matrix] = field(default_factory=list)  # X first
    obsm: dict[str, list[int]] = field(default_factory=dict)  # key -> shape
    obsm_bytes: int = 0
    obsp_bytes: int = 0
    varm_bytes: int = 0
    uns_bytes: int = 0
    obs_columns: int = 0
    var_columns: int = 0
    file_bytes: int = 0
    backed: bool = False  # opened from disk (files over 2 GB): X is not read into memory
    estimated: bool = False  # cells / genes estimated from a sample (text tables)
    guess: bool = False  # nothing in the file could be read (RDS): sizes are a guess
    note: str = ""  # what the numbers rest on, when not read exactly
    seconds: float = 0.0

    @property
    def x(self) -> Matrix | None:
        return self.matrices[0] if self.matrices and self.matrices[0].where == "X" else None

    def matrix(self, where: str) -> Matrix | None:
        return next((m for m in self.matrices if m.where == where), None)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["matrices"] = [m.to_dict() for m in self.matrices]
        return d


def _s(v: Any) -> str:
    return v.decode() if isinstance(v, bytes) else str(v) if v is not None else ""


def _h5_matrix(node, where: str, n_cells: int | None = None) -> Matrix | None:
    """A matrix from an h5ad node: a dense 2-D dataset or a csr/csc group."""
    import h5py

    if isinstance(node, h5py.Dataset):
        if node.ndim != 2 or node.dtype.kind not in "biuf":
            return None
        r, c = (int(x) for x in node.shape)
        return Matrix(where, r, c, False, r * c, node.dtype.name)
    enc = _s(node.attrs.get("encoding-type") or node.attrs.get("h5sparse_format"))
    if enc not in ("csr_matrix", "csc_matrix", "csr", "csc") or "data" not in node:
        return None
    shape = node.attrs.get("shape")
    if shape is None:
        shape = node.attrs.get("h5sparse_shape")
    if shape is None:
        return None
    data, indices = node["data"], node["indices"]
    return Matrix(where, int(shape[0]), int(shape[1]), True, int(data.shape[0]),
                  data.dtype.name, indices.dtype.name)


def _h5_bytes(node) -> int:
    """Bytes the datasets under an h5 node take once read (strings at _STR_BYTES each)."""
    import h5py

    if isinstance(node, h5py.Dataset):
        n = int(np.prod(node.shape)) if node.shape else 1
        return n * (_STR_BYTES if node.dtype.kind in "OSU" else node.dtype.itemsize)
    total = 0

    def add(_name, obj):
        nonlocal total
        if isinstance(obj, h5py.Dataset):
            n = int(np.prod(obj.shape)) if obj.shape else 1
            total += n * (_STR_BYTES if obj.dtype.kind in "OSU" else obj.dtype.itemsize)

    node.visititems(add)
    return total


def _n_columns(node) -> int:
    order = node.attrs.get("column-order") if node is not None else None
    if order is None:
        return 0
    return 1 if isinstance(order, (str, bytes)) else len(order)


def _read_h5ad(path: Path) -> Header:
    import h5py

    h = Header("h5ad")
    with h5py.File(path, "r") as f:
        obs = f.get("obs")
        h.obs_columns = _n_columns(obs)
        h.var_columns = _n_columns(f.get("var"))
        if "X" in f:
            x = _h5_matrix(f["X"], "X")
            if x is not None:
                h.matrices.append(x)
        if h.x is not None:
            h.n_cells, h.n_genes = h.x.n_cells, h.x.n_genes
        else:  # no X: the cells from the obs index, the genes from var's
            for grp, attr in (("obs", "n_cells"), ("var", "n_genes")):
                g = f.get(grp)
                idx = _s(g.attrs.get("_index", "_index")) if g is not None else ""
                if g is not None and idx in g:
                    setattr(h, attr, int(g[idx].shape[0]))
        for name in f.get("layers", {}):
            m = _h5_matrix(f["layers"][name], f"layers/{name}")
            if m is not None:
                h.matrices.append(m)
        if "raw" in f and "X" in f["raw"]:
            m = _h5_matrix(f["raw"]["X"], "raw/X")
            if m is not None:
                h.matrices.append(m)
        for key in f.get("obsm", {}):
            node = f["obsm"][key]
            if isinstance(node, h5py.Dataset):
                h.obsm[key] = [int(x) for x in node.shape]
            else:  # a DataFrame (protein counts) or a sparse matrix
                h.obsm[key] = [h.n_cells or 0, _n_columns(node) or 0]
            h.obsm_bytes += _h5_bytes(node)
        for key in f.get("obsp", {}):
            h.obsp_bytes += _h5_bytes(f["obsp"][key])
        for key in f.get("varm", {}):
            h.varm_bytes += _h5_bytes(f["varm"][key])
        if "uns" in f:
            h.uns_bytes = _h5_bytes(f["uns"])
    return h


def _read_10x_h5(path: Path) -> Header:
    """CellRanger HDF5: genes x cells CSC in /matrix (v3) or /<genome> (v2)."""
    import h5py

    h = Header("10x_h5")
    with h5py.File(path, "r") as f:
        g = f["matrix"] if "matrix" in f else next(
            (f[k] for k in f if isinstance(f[k], h5py.Group) and "shape" in f[k]), None)
        if g is None:
            raise ValueError("no matrix group")
        n_genes, n_cells = (int(x) for x in g["shape"][()][:2])
        nnz = int(g["data"].shape[0])
    h.n_cells, h.n_genes = n_cells, n_genes
    # scanpy reads the counts as float32 (cells x genes CSR).
    h.matrices.append(Matrix("X", n_cells, n_genes, True, nnz, "float32"))
    return h


def _mtx_header(path: Path) -> tuple[int, int, int]:
    """(rows, cols, entries) from a Matrix Market file's size line."""
    opener = gzip.open if _is_gzip(path) else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith("%"):
                continue
            rows, cols, nnz = (int(x) for x in s.split()[:3])
            return rows, cols, nnz
    raise ValueError("no size line")


def _read_mex(path: Path) -> Header:
    """10x MEX: matrix.mtx is genes x cells."""
    n_genes, n_cells, nnz = _mtx_header(path)
    h = Header("mex", n_cells, n_genes)
    h.matrices.append(Matrix("X", n_cells, n_genes, True, nnz, "float32"))
    return h


def _loom_nnz(ds, n_blocks: int = 16, rows: int = 64, cols: int = 4096) -> tuple[int, bool]:
    """Non-zeros of a loom matrix (genes x cells): counted when it is small, else
    estimated from evenly spaced row x column blocks."""
    r, c = ds.shape
    if r * c <= 4_000_000:
        return int(np.count_nonzero(ds[()])), False
    rng = np.random.default_rng(0)
    seen = nz = 0
    for i in np.linspace(0, max(0, r - rows), n_blocks).astype(int):
        j = int(rng.integers(0, max(1, c - cols)))
        block = ds[i:i + rows, j:j + cols]
        seen += block.size
        nz += int(np.count_nonzero(block))
    return int(round(nz / max(1, seen) * r * c)), True


def _read_loom(path: Path) -> Header:
    import h5py

    h = Header("loom")
    with h5py.File(path, "r") as f:
        ds = f["matrix"]
        n_genes, n_cells = (int(x) for x in ds.shape)
        nnz, est = _loom_nnz(ds)
        h.n_cells, h.n_genes = n_cells, n_genes
        # scanpy reads loom sparse (float32 CSR), layers alike.
        h.matrices.append(Matrix("X", n_cells, n_genes, True, nnz, "float32", nnz_estimated=est))
        for name in f.get("layers", {}):
            lnnz, lest = _loom_nnz(f["layers"][name])
            h.matrices.append(Matrix(f"layers/{name}", n_cells, n_genes, True, lnnz, "float32",
                                     nnz_estimated=lest))
        h.obs_columns = len(f.get("col_attrs", {}))
        h.var_columns = len(f.get("row_attrs", {}))
    if any(m.nnz_estimated for m in h.matrices):
        h.note = "Non-zero values estimated from a sample of the loom matrix."
    return h


def _zjson(d: Path, name: str) -> dict:
    p = d / name
    try:
        return json.loads(p.read_text()) if p.is_file() else {}
    except (OSError, ValueError):
        return {}


def _zarr_node(d: Path) -> tuple[dict, dict]:
    """(attributes, array metadata) of a zarr v2 or v3 node."""
    v3 = _zjson(d, "zarr.json")
    if v3:
        meta = {"shape": v3.get("shape"), "dtype": v3.get("data_type")} if v3.get(
            "node_type") == "array" else {}
        return v3.get("attributes") or {}, meta
    return _zjson(d, ".zattrs"), _zjson(d, ".zarray")


def _zarr_dtype(dtype: Any) -> str:
    try:
        return np.dtype(dtype).name
    except TypeError:
        return "float32"


def _zarr_matrix(d: Path, where: str) -> Matrix | None:
    attrs, meta = _zarr_node(d)
    if meta.get("shape") and len(meta["shape"]) == 2:
        r, c = (int(x) for x in meta["shape"])
        return Matrix(where, r, c, False, r * c, _zarr_dtype(meta.get("dtype")))
    enc = attrs.get("encoding-type", "")
    if enc in ("csr_matrix", "csc_matrix") and attrs.get("shape"):
        _, dmeta = _zarr_node(d / "data")
        _, imeta = _zarr_node(d / "indices")
        if not dmeta.get("shape"):
            return None
        r, c = (int(x) for x in attrs["shape"])
        return Matrix(where, r, c, True, int(dmeta["shape"][0]), _zarr_dtype(dmeta.get("dtype")),
                      _zarr_dtype(imeta.get("dtype", "int32")))
    return None


def _read_zarr(path: Path) -> Header:
    h = Header("zarr")
    x = _zarr_matrix(path / "X", "X")
    if x is not None:
        h.matrices.append(x)
        h.n_cells, h.n_genes = x.n_cells, x.n_genes
    layers = path / "layers"
    if layers.is_dir():
        for d in sorted(p for p in layers.iterdir() if p.is_dir()):
            m = _zarr_matrix(d, f"layers/{d.name}")
            if m is not None:
                h.matrices.append(m)
    raw = _zarr_matrix(path / "raw" / "X", "raw/X")
    if raw is not None:
        h.matrices.append(raw)
    for grp, attr in (("obs", "obs_columns"), ("var", "var_columns")):
        order = _zarr_node(path / grp)[0].get("column-order") or []
        setattr(h, attr, len(order))
    obsm = path / "obsm"
    if obsm.is_dir():
        for d in sorted(p for p in obsm.iterdir() if p.is_dir()):
            meta = _zarr_node(d)[1]
            if meta.get("shape"):
                shape = [int(x) for x in meta["shape"]]
                h.obsm[d.name] = shape
                h.obsm_bytes += int(np.prod(shape)) * np.dtype(_zarr_dtype(meta.get("dtype"))).itemsize
    return h


def _is_gzip(path: Path) -> bool:
    try:
        with open(path, "rb") as fh:
            return fh.read(2) == _GZIP_MAGIC
    except OSError:
        return False


# Text read to estimate a table's rows: enough lines to average their length.
_TABLE_SAMPLE_BYTES = 4 * MiB


def _gunzip_head(path: Path, n: int) -> tuple[bytes, int]:
    """About the first ``n`` bytes of a gzip file's text, and how many compressed
    bytes produced them (a GzipFile reads ahead, so its position would not say)."""
    d = zlib.decompressobj(16 + zlib.MAX_WBITS)
    out, got, fed = [], 0, 0
    with open(path, "rb") as fh:
        while got < n:
            chunk = fh.read(16384)
            if not chunk:
                break
            fed += len(chunk)
            data = d.decompress(chunk)
            out.append(data)
            got += len(data)
            if d.eof:
                fed -= len(d.unused_data)
                break
    return b"".join(out), fed


def _read_table(path: Path, genes_in_rows: bool = True) -> Header:
    """A delimited expression table: its columns from the first lines, its rows
    estimated from the file size and the average line length (for a gzipped
    file, also its compression ratio over the sampled text)."""
    size = path.stat().st_size
    gz = _is_gzip(path)
    if gz:
        text, consumed = _gunzip_head(path, _TABLE_SAMPLE_BYTES)
    else:
        with open(path, "rb") as fh:
            text = fh.read(_TABLE_SAMPLE_BYTES)
        consumed = len(text)
    lines = text.split(b"\n")
    complete = lines[:-1] if len(lines) > 1 else lines  # the last one may be cut off
    if len(complete) < 2:
        raise ValueError("fewer than two lines")
    probe = complete[1].decode("utf-8", "replace")
    delim = "\t" if probe.count("\t") >= probe.count(",") and "\t" in probe else ","
    n_data_cols = len(probe.split(delim)) - 1  # the first column is the row name
    body = complete[1:]
    avg = sum(len(ln) + 1 for ln in body) / len(body)
    whole = len(text) < _TABLE_SAMPLE_BYTES  # the sample was the whole file
    if whole:
        n_rows = len([ln for ln in body if ln.strip()])
    else:
        text_bytes = size * (len(text) / max(1, consumed)) if gz else size
        n_rows = int(round((text_bytes - len(complete[0]) - 1) / avg))
    n_genes, n_cells = (n_rows, n_data_cols) if genes_in_rows else (n_data_cols, n_rows)
    h = Header("table", n_cells, n_genes, estimated=not whole)
    h.matrices.append(Matrix("X", n_cells, n_genes, False, n_cells * n_genes, "float32"))
    if not whole:
        h.note = (f"Rows estimated from the file size and the length of the first "
                  f"{len(body):,} lines" + (" and their compression" if gz else "") + ".")
    return h


# A converted Seurat object in memory, per byte of .rds: a guess (not measured).
RDS_BYTES_PER_FILE_BYTE = (1.5, 3.0, 6.0)  # low, point, high


def _read_rds(path: Path) -> Header:
    h = Header("rds", guess=True)
    h.note = ("An .rds file has no header to read before the converter opens it: the "
              "memory is a guess of about 3 times the file size (from 1.5 to 6 times).")
    return h


def sniff_format(path: Path) -> str:
    """The header reader for a path: by content, then name."""
    path = Path(path)
    if path.is_dir():
        return "zarr"
    name = path.name.lower()
    if name.endswith((".rds", ".rdata", ".rda")):
        return "rds"
    try:
        with open(path, "rb") as fh:
            head = fh.read(8)
    except OSError:
        return "unknown"
    if head.startswith(_HDF5_MAGIC):
        import h5py

        try:
            with h5py.File(path, "r") as f:
                keys = set(f.keys())
                if "matrix" in keys and isinstance(f["matrix"], h5py.Group):
                    return "10x_h5"
                if {"row_attrs", "col_attrs"} <= keys:
                    return "loom"
                if {"obs", "var"} <= keys:
                    return "h5ad"
                if any(isinstance(f[k], h5py.Group) and "shape" in f[k] for k in keys):
                    return "10x_h5"  # CellRanger v2: one group per genome
        except (OSError, KeyError, RuntimeError):
            return "unknown"
        return "unknown"
    opener = gzip.open if head.startswith(_GZIP_MAGIC) else open
    try:
        with opener(path, "rt", encoding="utf-8", errors="strict") as fh:
            first = fh.read(256)
    except (OSError, UnicodeDecodeError, EOFError):
        return "unknown"
    if first.lstrip().startswith("%%MatrixMarket"):
        return "mex"
    return "table"


_READERS = {
    "h5ad": _read_h5ad, "10x_h5": _read_10x_h5, "mex": _read_mex, "loom": _read_loom,
    "zarr": _read_zarr, "rds": _read_rds,
}


def read_header(path: str | Path, fmt: str | None = None, genes_in_rows: bool = True) -> Header:
    """The shape and contents of a data file, without reading its matrix.

    ``fmt`` is one of h5ad | 10x_h5 | mex (the matrix.mtx file) | loom | zarr |
    table | rds, else sniffed. A file that cannot be read gives format "unknown"
    with the reason in ``note``; this never raises for a bad file.
    """
    path = Path(path)
    t0 = time.perf_counter()
    fmt = fmt or sniff_format(path)
    try:
        if fmt == "table":
            h = _read_table(path, genes_in_rows)
        elif fmt in _READERS:
            h = _READERS[fmt](path)
        else:
            h = Header("unknown", note="The file is not a format scView reads.")
    except Exception as e:  # noqa: BLE001 — a damaged file is reported, not raised
        h = Header("unknown", note=f"The file could not be read ({type(e).__name__}: {e}).")
    h.file_bytes = _size(path)
    # The adaptor opens files over 2 GB on disk (anndata backed mode).
    h.backed = fmt == "h5ad" and h.file_bytes > 2 * GiB
    h.seconds = round(time.perf_counter() - t0, 4)
    return h


def _size(path: Path) -> int:
    try:
        if path.is_dir():
            return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
        return path.stat().st_size
    except OSError:
        return 0


def header_from_adata(adata) -> Header:
    """The same header for an AnnData in memory (the cached dataset). Not for a
    backed one: its X would be read from disk; use ``read_header`` on its file."""
    import scipy.sparse as sp

    def mat(m, where: str) -> Matrix | None:
        if m is None or getattr(m, "ndim", 2) != 2:
            return None
        if sp.issparse(m):
            return Matrix(where, m.shape[0], m.shape[1], True, int(m.nnz), m.dtype.name,
                          m.indices.dtype.name)
        if hasattr(m, "dtype") and hasattr(m, "shape"):
            r, c = m.shape
            return Matrix(where, int(r), int(c), False, int(r) * int(c), np.dtype(m.dtype).name)
        return None

    h = Header("h5ad", adata.n_obs, adata.n_vars)
    x = mat(adata.X, "X")
    if x is not None:
        h.matrices.append(x)
    for name, m in adata.layers.items():
        mm = mat(m, f"layers/{name}")
        if mm is not None:
            h.matrices.append(mm)
    if adata.raw is not None:
        mm = mat(adata.raw.X, "raw/X")
        if mm is not None:
            h.matrices.append(mm)
    for key, v in adata.obsm.items():
        h.obsm[key] = [int(x) for x in getattr(v, "shape", (adata.n_obs, 0))]
        h.obsm_bytes += _array_bytes(v)
    for v in adata.obsp.values():
        h.obsp_bytes += _array_bytes(v)
    for v in adata.varm.values():
        h.varm_bytes += _array_bytes(v)
    h.obs_columns = adata.obs.shape[1]
    h.var_columns = adata.var.shape[1]
    return h


def _array_bytes(v) -> int:
    import scipy.sparse as sp

    if sp.issparse(v):
        return int(v.data.nbytes + v.indices.nbytes + v.indptr.nbytes)
    if hasattr(v, "to_numpy"):  # a DataFrame
        return int(v.shape[0] * v.shape[1] * 8)
    return int(getattr(v, "nbytes", 0))


# ---------------------------------------------------------------------------
# Load estimate
# ---------------------------------------------------------------------------

# What the adaptor and the dataset list add on open besides the arrays (pandas
# frames, per-cell QC caches, metadata), per cell, and once. Measured (see the top).
LOAD_PER_CELL = 600
LOAD_FIXED = 20 * MiB
# Transient while reading, per stored value, by format (the values are converted
# and the index arrays widened on the way in). Measured on the formats listed.
READ_BYTES_PER_VALUE = {
    "h5ad": 0.0, "zarr": 0.0, "10x_h5": 8.0, "mex": 46.0, "loom": 8.0, "table": 28.0,
}
# Saving an ingested dataset (gzip h5ad): transient per byte of its matrices.
WRITE_FRACTION = 0.1
# Rough throughput for the time, at 2 CPUs: bytes of file per second, except h5ad and
# zarr, read at about 400 MB/s of what they hold in memory (gzip or not).
READ_FILE_BPS = {"10x_h5": 60e6, "mex": 12e6, "loom": 30e6, "table": 5e6, "rds": 20e6}
READ_MEMORY_BPS = 400e6
WRITE_BPS = 150e6  # gzip h5ad, bytes of matrices per second


def _frames_bytes(h: Header) -> int:
    n, m = h.n_cells or 0, h.n_genes or 0
    return n * (_STR_BYTES + 8 * h.obs_columns) + m * (_STR_BYTES + 8 * h.var_columns)


def _matrices_bytes(h: Header) -> int:
    return sum(mm.nbytes for mm in h.matrices if not (h.backed and mm.where == "X"))


def resident_bytes(h: Header) -> int:
    """Memory the dataset holds once open (its arrays, frames and caches)."""
    if h.guess:
        return int(h.file_bytes * RDS_BYTES_PER_FILE_BYTE[1])
    return (_matrices_bytes(h) + h.obsm_bytes + h.obsp_bytes + h.varm_bytes + h.uns_bytes
            + _frames_bytes(h) + (h.n_cells or 0) * LOAD_PER_CELL + LOAD_FIXED)


@dataclass
class LoadEstimate:
    """Memory and time of opening a file (``ingest``: reading, saving and reopening it)."""

    resident_bytes: int
    peak_bytes: int
    peak_stage: str
    seconds: tuple[float, float]
    breakdown: list[tuple[str, int]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _range(t: float, cpus: float | None = None) -> tuple[float, float]:
    """A rough time as a range: half to twice the point estimate, scaled by the CPUs."""
    t *= _cpu_factor(cpus)
    return (round(max(0.5, t * 0.5), 1), round(max(1.0, t * 2.0), 1))


def estimate_load(h: Header, ingest: bool = True, cpus: float | None = None) -> LoadEstimate:
    """Memory once loaded and at the peak of loading, and a rough time.

    ``ingest``: the ingest commit reads the file, saves it as a gzip h5ad and opens
    that; otherwise (an existing dataset) the file is only opened."""
    res = resident_bytes(h)
    notes = [h.note] if h.note else []
    if h.guess:
        lo, mid, hi = (int(h.file_bytes * f) for f in RDS_BYTES_PER_FILE_BYTE)
        return LoadEstimate(mid, mid, "opening the converted file",
                            _range(h.file_bytes / READ_FILE_BPS["rds"], cpus),
                            [("Guess from the file size", mid)], notes)
    values = sum(mm.nnz for mm in h.matrices if not (h.backed and mm.where == "X"))
    read = int(values * READ_BYTES_PER_VALUE.get(h.format, 8.0))
    breakdown = [(f"{mm.where} ({'sparse' if mm.sparse else 'dense'}, {mm.dtype})", mm.nbytes)
                 for mm in h.matrices if not (h.backed and mm.where == "X")]
    other = res - _matrices_bytes(h)
    breakdown.append(("Cell and gene tables, embeddings, results", other))
    peak, stage = res + read, "reading the file"
    mats = _matrices_bytes(h)
    secs = (mats / READ_MEMORY_BPS if h.format in ("h5ad", "zarr")
            else h.file_bytes / READ_FILE_BPS.get(h.format, 30e6))
    if ingest:  # saved as a gzip h5ad, then opened
        write = int(mats * WRITE_FRACTION)
        if res + write > peak:
            peak, stage = res + write, "saving the dataset"
        secs += mats / WRITE_BPS + mats / READ_MEMORY_BPS
    if read:
        breakdown.append(("Reading (converting the values)", read))
    if h.backed:
        notes.append("Files over 2 GB open from disk: the matrix X is read when needed, "
                     "not held in memory.")
    return LoadEstimate(res, peak, stage, _range(secs, cpus), breakdown, notes)


# ---------------------------------------------------------------------------
# Run estimate
# ---------------------------------------------------------------------------

STEP_LABELS = {
    "reset_to_counts": "reset to counts", "qc_metrics": "QC metrics",
    "doublet_detection": "doublet detection", "filtering": "filtering",
    "normalization": "normalisation", "log_transform": "log transform",
    "highly_variable_genes": "variable genes", "scaling": "scaling", "pca": "PCA",
    "uncorrected_embedding": "uncorrected UMAP", "batch_correction": "batch correction",
    "neighbors": "neighbour graph", "clustering": "clustering", "embeddings": "UMAP",
    "marker_genes": "marker genes", "enrichment": "enrichment", "cell_cycle": "cell cycle",
    "dissociation_stress": "dissociation stress", "cell_type_annotation": "cell-type annotation",
    "save": "saving the result",
}
# Steps whose coefficients come from a measured run; the others are reasoned from
# what the step allocates and say so.
MEASURED_STEPS = {
    "reset_to_counts", "qc_metrics", "doublet_detection", "filtering", "normalization",
    "log_transform", "highly_variable_genes", "scaling", "pca", "neighbors", "clustering",
    "embeddings", "marker_genes", "enrichment", "cell_cycle", "dissociation_stress", "save",
}

# Filtering in MAD mode removed 11% of the cells on the ovary subsample; genes seen in
# fewer than 3 cells go too (22% there, none on processed data).
FILTER_CELLS_KEPT = 0.9
FILTER_GENES_KEPT_RAW = 0.8

# Scrublet (sc.pp.scrublet): it copies the dataset, keeps raw and log1p copies of X
# while it picks its variable genes, then z-scores observed + 2x simulated doublets
# (dense) and finds exact neighbours (sklearn, chunked at 1 GiB of distances).
DBL_X_COPIES = 4.0  # copies of X alive at once during its gene selection
DBL_DENSE_PER_VALUE = 7.0  # bytes per cell x gene for the z-scored observed + simulated cells
DBL_KNN = 1.0 * GiB
DBL_KNN_SMALL = 0.4  # of the full (3n)^2 distance matrix, while it is under a chunk
# Steps that import or JIT-compile libraries the first time they run in a process
# (numba, umap, pynndescent, the MSigDB cache): held afterwards.
WARMUP = {"doublet_detection": 150 * MiB, "neighbors": 100 * MiB, "marker_genes": 100 * MiB,
          "enrichment": 80 * MiB, "embeddings": 20 * MiB}
WARMUP_MODULE = "umap"  # imported by the first neighbour graph: warm-up already paid
NEIGHBORS_PER_CELL = 12_000


@dataclass
class _State:
    """What the dataset holds while a run works on it."""

    n: int
    m: int
    x: Matrix | None
    layers: dict[str, Matrix]
    raw: Matrix | None
    other: int  # obsm, obsp, varm, uns, frames
    held: int = 0  # an earlier copy the dataset's adaptor keeps until the run ends
    processed: bool = False  # an embedding or .raw on arrival: genes already filtered

    @classmethod
    def from_header(cls, h: Header) -> "_State":
        layers = {mm.where.split("/", 1)[1]: mm for mm in h.matrices if mm.where.startswith("layers/")}
        return cls(h.n_cells or 0, h.n_genes or 0, h.x, layers, h.matrix("raw/X"),
                   h.obsm_bytes + h.obsp_bytes + h.varm_bytes + h.uns_bytes + _frames_bytes(h)
                   + (h.n_cells or 0) * LOAD_PER_CELL + LOAD_FIXED,
                   processed=bool(h.obsm) or h.matrix("raw/X") is not None)

    def matrices(self) -> list[Matrix]:
        return [mm for mm in [self.x, self.raw, *self.layers.values()] if mm is not None]

    def resident(self) -> int:
        return sum(mm.nbytes for mm in self.matrices()) + self.other + self.held

    def xb(self) -> int:
        return self.x.nbytes if self.x is not None else 0

    def others(self) -> int:
        """Bytes of the layers and .raw (a copy of the dataset copies them too)."""
        return sum(mm.nbytes for mm in self.matrices()) - self.xb()

    def lognorm(self) -> Matrix | None:
        """What markers and scores read: .raw (log-normalised) when set, else X."""
        return self.raw or self.x


def _like(mm: Matrix, where: str, n: int, m: int) -> Matrix:
    """``mm`` resized to n x m, its stored values scaled with it."""
    nnz = int(mm.nnz * (n * m) / max(1, mm.n_cells * mm.n_genes)) if mm.sparse else n * m
    return Matrix(where, n, m, mm.sparse, nnz, mm.dtype, mm.index_dtype)


def _dense_f32(s: _State) -> int:
    return s.n * s.m * 4


def _p(params, name: str, default):
    return getattr(params, name, default) if params is not None else default


# Each step: (state) -> (transient bytes over the state's resident, state after, seconds at 2 CPUs).
def _step_reset(s: _State, params) -> tuple[int, _State, float]:
    counts = s.layers.get("counts")
    if counts is not None:
        x = _like(counts, "X", s.n, s.m)
        return counts.nbytes, _State(s.n, s.m, x, dict(s.layers), None, s.other // 2, s.held,
                                     s.processed), 0.05 + counts.nbytes * 1.5e-9
    if s.raw is not None:  # a new object: the adaptor keeps the old one
        x = _like(s.raw, "X", s.n, s.raw.n_genes)
        return s.raw.nbytes, _State(s.n, s.raw.n_genes, x, {}, None, s.other // 2,
                                    s.held + s.resident(), s.processed), 0.05 + s.raw.nbytes * 1.5e-9
    return 0, s, 0.1


def _step_qc(s: _State, params) -> tuple[int, _State, float]:
    if s.x is None:
        return 0, s, 0.1
    t = int((0.6 if s.x.sparse else 1.0) * s.xb())
    s.other += s.n * 6 * 8
    return t, s, 0.1 + s.x.nnz * (1.8e-7 if s.x.sparse else 1.2e-8)


def _step_doublet(s: _State, params) -> tuple[int, _State, float]:
    xb, lb = s.xb(), s.others()
    copies = int(DBL_X_COPIES * xb) + lb
    knn = min(DBL_KNN, int(DBL_KNN_SMALL * (3 * s.n) ** 2 * 8))
    dense = int(DBL_DENSE_PER_VALUE * s.n * s.m) + knn + 2 * xb + lb
    s.other += s.n * 16
    nnz = s.x.nnz if s.x is not None else 0
    sparse = s.x is not None and s.x.sparse
    return max(copies, dense), s, 2.0 + nnz * (8e-7 if sparse else 2.5e-7)


def _step_filter(s: _State, params) -> tuple[int, _State, float]:
    nnz = s.x.nnz if s.x is not None else 0
    if s.processed:  # filtered before: every cell passes, genes are subset in place
        return int(0.2 * s.xb()), s, 0.2 + 1e-8 * nnz
    n2 = int(s.n * FILTER_CELLS_KEPT)
    m2 = int(s.m * FILTER_GENES_KEPT_RAW)
    mats = sum(mm.nbytes for mm in s.matrices())
    # The cells kept are a new object; the adaptor keeps the one it opened until the run ends.
    new = _State(n2, m2, _like(s.x, "X", n2, m2) if s.x else None,
                 {k: _like(v, k, n2, m2) for k, v in s.layers.items()},
                 _like(s.raw, "raw/X", n2, s.raw.n_genes) if s.raw else None,
                 s.other, s.held + mats if s.held == 0 else s.held, s.processed)
    new_mats = sum(mm.nbytes for mm in new.matrices())
    # Copying the view of the kept cells, then filter_genes, each rebuild the arrays once.
    transient = (new.resident() - s.resident()) + 2 * new_mats
    sparse = s.x is not None and s.x.sparse
    return max(0, transient), new, 0.2 + nnz * (1e-7 if sparse else 7e-8)


def _step_normalization(s: _State, params) -> tuple[int, _State, float]:
    if s.x is None:
        return 0, s, 0.1
    if "counts" not in s.layers:
        s.layers["counts"] = _like(s.x, "layers/counts", s.n, s.m)
    t = s.xb() if s.x.sparse else int(0.05 * s.xb())
    return t, s, 0.05 + s.x.nnz * 5e-9


def _step_log(s: _State, params) -> tuple[int, _State, float]:
    return 0, s, 0.05 + (s.x.nnz if s.x else 0) * 5e-9


def _step_hvg(s: _State, params) -> tuple[int, _State, float]:
    sparse = s.x is not None and s.x.sparse
    return 2 * s.xb(), s, 0.05 + (s.x.nnz if s.x else 0) * (1e-8 if sparse else 2.2e-8)


def _step_scaling(s: _State, params) -> tuple[int, _State, float]:
    if s.x is None:
        return 0, s, 0.1
    dense = _dense_f32(s)
    cast = s.x.nnz * 8 if s.x.sparse else 0
    if s.raw is None and "counts" not in s.layers:
        s.layers["counts"] = _like(s.x, "layers/counts", s.n, s.m)  # counts kept from .raw
    raw = Matrix("raw/X", s.n, s.m, s.x.sparse, s.x.nnz, s.x.dtype, s.x.index_dtype)
    after = _State(s.n, s.m, Matrix("X", s.n, s.m, False, s.n * s.m, "float32"), s.layers, raw,
                   s.other, s.held, s.processed)
    grow = after.resident() - s.resident()
    return max(grow, 0) + dense + cast, after, 0.2 + s.n * s.m * 1.5e-8


def _step_pca(s: _State, params) -> tuple[int, _State, float]:
    n_hvg = min(s.m, int(_p(params, "n_top_genes", 2000)))
    n_comps = int(_p(params, "n_comps", 50))
    t = int(2.5 * s.n * n_hvg * 4) + s.n * n_comps * 16
    s.other += s.n * n_comps * 4 + s.m * n_comps * 8
    return t, s, 0.3 + s.n * n_hvg * 9e-8


def _step_neighbors(s: _State, params) -> tuple[int, _State, float]:
    # The graph, and what building it keeps (pynndescent's index structures): held.
    g = s.n * NEIGHBORS_PER_CELL
    s.other += g
    # Mostly numba compilation: exact neighbours under 4,096 cells, pynndescent above.
    return g, s, (10.0 + 1e-3 * s.n) if s.n <= 4096 else (28.0 + 3e-4 * s.n)


def _step_clustering(s: _State, params) -> tuple[int, _State, float]:
    # Leiden runs in a child process (core/leiden.py): its interpreter and the graph.
    return 60 * MiB + s.n * 9000, s, 0.5 + 4e-4 * s.n


def _epochs(n: int) -> int:
    return 500 if n <= 10_000 else 200  # umap-learn's default


def _step_embeddings(s: _State, params) -> tuple[int, _State, float]:
    s.other += s.n * 5 * 4
    return s.n * 200, s, 0.5 + 3.8e-6 * s.n * _epochs(s.n)


def _step_uncorrected(s: _State, params) -> tuple[int, _State, float]:
    k = int(_p(params, "n_neighbors", 15))
    s.other += s.n * 2 * 4
    return s.n * k * 48, s, 1.5e-3 * s.n + 1.9e-6 * s.n * _epochs(s.n)


def _step_batch(s: _State, params) -> tuple[int, _State, float]:
    # Harmony: cluster x cell matrices (float64, up to 100 clusters) and PC copies.
    clusters = min(100, max(1, s.n // 30))
    t = s.n * clusters * 8 * 3 + s.n * 50 * 8 * 4
    s.other += s.n * 50 * 8
    return t, s, 1.0 + 1e-3 * s.n


def _groups(params) -> int:
    """Marker / enrichment groups: the new clustering's and the columns named."""
    cols = len(_p(params, "marker_columns", []) or []) or 1
    return 15 * (cols + 1)


def _step_markers(s: _State, params) -> tuple[int, _State, float]:
    ln = s.lognorm()
    if ln is None:
        return 0, s, 0.1
    # Wilcoxon ranks dense chunks of 10M values; each group's cells are copied out.
    t = (2 * ln.nbytes if not ln.sparse else ln.nbytes) + 150 * MiB
    groups = _groups(params)
    s.other += 56 * ln.n_genes * groups
    return t, s, 1.0 + s.n * ln.n_genes * groups * (5e-9 if ln.sparse else 2e-8)


def _step_enrichment(s: _State, params) -> tuple[int, _State, float]:
    return 20 * MiB, s, 2.0 + 0.4 * _groups(params)


def _step_scores(s: _State, params) -> tuple[int, _State, float]:
    ln = s.lognorm()
    if ln is None:
        return 0, s, 0.05
    s.other += s.n * 4 * 8
    return (int(1.5 * ln.nbytes) if not ln.sparse else int(0.6 * ln.nbytes)), s, 0.05 + ln.nnz * 2e-8


def _step_annotation(s: _State, params) -> tuple[int, _State, float]:
    method = _p(params, "annotation_method", "llm")
    ln = s.lognorm()
    if method == "celltypist" and ln is not None:
        # CellTypist scales the model's genes (about 6,000) densely, in float64.
        return s.n * min(ln.n_genes, 6000) * 8 * 2, s, 5.0 + 2e-3 * s.n
    return 100 * MiB, s, 10.0 + 2e-4 * s.n  # markers per group + one LLM call or a score


_STEP_MODELS = {
    "reset_to_counts": _step_reset, "qc_metrics": _step_qc, "doublet_detection": _step_doublet,
    "filtering": _step_filter, "normalization": _step_normalization, "log_transform": _step_log,
    "highly_variable_genes": _step_hvg, "scaling": _step_scaling, "pca": _step_pca,
    "uncorrected_embedding": _step_uncorrected, "batch_correction": _step_batch,
    "neighbors": _step_neighbors, "clustering": _step_clustering, "embeddings": _step_embeddings,
    "marker_genes": _step_markers, "enrichment": _step_enrichment, "cell_cycle": _step_scores,
    "dissociation_stress": _step_scores, "cell_type_annotation": _step_annotation,
}

# After a run the app frees the run's copy (gc + malloc_trim) and opens the saved file.
SAVE_SECONDS_PER_BYTE = 1.0 / WRITE_BPS


@dataclass
class StepEstimate:
    step: str
    resident_bytes: int  # held when the step starts
    transient_bytes: int  # over that, at the step's peak
    seconds: tuple[float, float]
    measured: bool

    @property
    def peak_bytes(self) -> int:
        return self.resident_bytes + self.transient_bytes

    def to_dict(self) -> dict:
        return {"step": self.step, "label": STEP_LABELS.get(self.step, self.step),
                "resident_bytes": self.resident_bytes, "transient_bytes": self.transient_bytes,
                "peak_bytes": self.peak_bytes, "seconds": list(self.seconds),
                "measured": self.measured}


@dataclass
class RunEstimate:
    steps: list[StepEstimate]
    start_bytes: int  # what the dataset holds before the run
    peak_bytes: int
    peak_step: str
    end_bytes: int  # held once the result is saved and open again
    seconds: tuple[float, float]


def _warm(step: str, warm: bool) -> int:
    return 0 if warm else WARMUP.get(step, 0)


def process_warm() -> bool:
    """True once this process has run the libraries whose first use costs memory."""
    import sys

    return WARMUP_MODULE in sys.modules


def estimate_run(shape: tuple[int, int] | Header, nnz: int | None = None, dense: bool = False,
                 steps: list[str] | None = None, params=None, cpus: float | None = None,
                 warm: bool | None = None) -> RunEstimate:
    """Memory per step and at the run's peak, and rough seconds, for ``steps`` (any
    order; run in the pipeline's) on a dataset.

    ``shape`` is (cells, genes) of X with ``nnz`` stored values (``dense``: every
    value), or a full ``Header`` (layers and .raw count too: pass one when you have
    it). The run's peak is the largest (held at the step's start + the step's
    transient) over the steps, as the dataset's state changes through them (reset
    to counts replaces X, filtering leaves the opened copy held by the adaptor,
    scaling makes X dense), then saving and reopening the result. ``warm``: the
    process already paid its one-time imports (default: looked up)."""
    from scview.core.pipeline import ALL_STEPS

    if isinstance(shape, Header):
        h = shape
    else:
        n, m = shape
        nnz = n * m if dense or nnz is None else nnz
        h = Header("h5ad", n, m, [Matrix("X", n, m, not dense, nnz, "float32")])
    warm = process_warm() if warm is None else warm
    s = _State.from_header(h)
    start = s.resident()
    out: list[StepEstimate] = []
    total = 0.0
    warm_held = 0
    for step in [x for x in ALL_STEPS if x in set(steps or [])]:
        before = s.resident() + warm_held
        t, s, secs = _STEP_MODELS[step](s, params)
        w = _warm(step, warm)
        warm_held += w
        total += secs
        out.append(StepEstimate(step, before, int(t) + w, _range(secs, cpus), step in MEASURED_STEPS))
    end = s.resident() + warm_held
    # Saving: the final file written, the run's copy freed, the saved file opened.
    final = _State(s.n, s.m, s.x, s.layers, s.raw, s.other, 0, s.processed).resident()
    # anndata copies a dense .raw or layer once while writing it (a dense X it does not).
    dense_write = max((mm.nbytes for mm in s.matrices() if not mm.sparse and mm is not s.x),
                      default=0)
    save_t = dense_write if out else 0
    save_secs = sum(mm.nbytes for mm in s.matrices()) * SAVE_SECONDS_PER_BYTE if out else 0.0
    if out:
        total += save_secs
        out.append(StepEstimate("save", end, save_t, _range(save_secs, cpus), True))
    peak = max(out, key=lambda e: e.peak_bytes) if out else None
    return RunEstimate(out, start, peak.peak_bytes if peak else start,
                       peak.step if peak else "", final + warm_held, _range(total, cpus))


def estimate_step_peak(step: str, shape: tuple[int, int], nnz: int, dense: bool,
                       params=None, *, other_bytes: int = 0, dtype: str = "float32") -> int:
    """Memory (bytes) ``step`` needs at its peak over what the dataset holds when it
    starts. Pure: no I/O, no process state (one-time imports not counted).

    ``shape`` = (cells, genes) of X; ``nnz`` its stored values (ignored for a dense
    X); ``dense`` whether X is a dense array; ``params`` the run's ``PipelineParams``
    (None: the defaults); ``other_bytes`` the bytes of the layers and .raw, which
    doublet detection and filtering copy (0 underestimates those two); ``dtype`` of
    X's values. Filtering is taken to remove cells (a copy), as it does on data not
    filtered before. Unknown steps give 0. From a live AnnData use
    ``step_peak_for(adata, step, params)``, which reads all of this from it.
    """
    model = _STEP_MODELS.get(step)
    if model is None:
        return 0
    n, m = shape
    x = Matrix("X", n, m, not dense, n * m if dense else nnz, dtype)
    layers = {}
    if other_bytes and n:  # an opaque block of the layers' and .raw's bytes, per cell
        layers["other"] = Matrix("layers/other", n, max(1, other_bytes // n), False, 0, "uint8")
    t, _, _ = model(_State(n, m, x, layers, None, 0), params)
    return int(t)


def step_peak_for(adata, step: str, params=None) -> int:
    """``estimate_step_peak`` for a live AnnData (in memory, not backed): its X,
    layers and .raw as they are, e.g. from a hook before each pipeline step."""
    model = _STEP_MODELS.get(step)
    if model is None:
        return 0
    t, _, _ = model(_State.from_header(header_from_adata(adata)), params)
    return int(t)


# ---------------------------------------------------------------------------
# CPUs and memory available
# ---------------------------------------------------------------------------

CALIBRATION_CPUS = 2.0


def cpu_quota(cgroup: Path = Path("/sys/fs/cgroup")) -> float:
    """CPUs this container may use: the cgroup quota (``cpu.max``), else the CPUs it sees."""
    try:
        quota, period = (cgroup / "cpu.max").read_text().split()[:2]
        if quota != "max":
            return max(0.1, int(quota) / int(period))
    except (OSError, ValueError):
        pass
    try:
        return float(len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return float(os.cpu_count() or 1)


def _cpu_factor(cpus: float | None) -> float:
    """Time relative to the 2-CPU calibration: most steps use one or two cores, so
    more CPUs help little (square root, at most 2x faster) and fewer slow it down."""
    c = cpu_quota() if cpus is None else cpus
    return float(np.clip((CALIBRATION_CPUS / max(c, 0.1)) ** 0.5, 0.5, 4.0))


@dataclass
class Available:
    limit_bytes: int | None
    used_bytes: int | None
    available_bytes: int | None
    limited: bool
    source: str


def available_memory(cgroup: Path | None = None, meminfo: Path | None = None) -> Available:
    """``read_budget()`` (which leaves out file cache the kernel can drop, as
    ``docker stats`` does, so this agrees with the sidebar gauge)."""
    from scview.core.memory_budget import CGROUP, MEMINFO, read_budget

    b = read_budget(cgroup or CGROUP, meminfo or MEMINFO)
    used = b.used_bytes
    avail = (max(0, b.limit_bytes - used) if b.limit_bytes is not None and used is not None
             else None)
    return Available(b.limit_bytes, used, avail, b.limited, b.source)


# ---------------------------------------------------------------------------
# Verdict: the estimate against the memory scView has free
# ---------------------------------------------------------------------------

FITS_BELOW = 0.7  # of the memory free: fits; up to 1.0 tight; beyond, won't fit
# Largest error of the peak estimates on the calibration datasets (17%, see the top), rounded up.
ERROR_BAND = 0.2


def _gb(b: float) -> str:
    g = b / GiB
    return f"{g:.0f}" if g >= 10 else f"{g:.1f}"


def _about(b: float) -> str:
    """'about 1.2 GB', or 'under 0.1 GB' for what would read 0.0."""
    return "under 0.1 GB" if b < 0.05 * GiB else f"about {_gb(b)} GB"


def _dur(s: float) -> str:
    if round(s) < 60:
        return f"{max(1, round(s))} s"
    if round(s / 60) < 60:
        return f"{round(s / 60)} min"
    return f"{s / 3600:.1f} h"


def time_text(seconds: tuple[float, float]) -> str:
    """'roughly 20 s to 1 min', 'roughly 2 to 6 min'."""
    lo, hi = seconds
    if hi < 10:
        return "a few seconds"
    a, b = _dur(lo), _dur(hi)
    if a.split()[-1] == b.split()[-1]:
        a = a.split()[0]
    return f"roughly {a} to {b}" if a != b else f"roughly {b}"


def _level(need: int, avail: int | None) -> str:
    if avail is None:
        return "unknown"
    if need <= 0:
        return "fits"
    if avail <= 0:
        return "wont_fit"
    r = need / avail
    return "fits" if r < FITS_BELOW else "tight" if r <= 1.0 else "wont_fit"


def _budget_text(a: Available) -> str:
    if a.available_bytes is None or a.limit_bytes is None:
        return "scView could not read how much memory it has."
    if a.limited:
        return f"scView has {_gb(a.available_bytes)} GB free of its {_gb(a.limit_bytes)} GB limit."
    return (f"scView has {_gb(a.available_bytes)} GB free of the {_gb(a.limit_bytes)} GB "
            "this machine gives it (no limit is set).")


def _raise_limit(a: Available, need: int) -> str | None:
    if a.limit_bytes is None or a.available_bytes is None:
        return None
    # So that the estimate is under 70% of what is free: room for its error.
    used = a.limit_bytes - a.available_bytes
    target = int(np.ceil((used + need / FITS_BELOW) / GiB))
    if a.limited:
        return (f"Raise SCVIEW_MEMORY_GB in the .env file in the scView folder from "
                f"{_gb(a.limit_bytes)} to at least {target} GB (at most the computer's free "
                "memory minus 3 GB, the start scripts' rule), then restart scView.")
    return ("Free memory on this computer by closing other programs, or give Docker or "
            f"Podman more memory (at least {target} GB in all; on Windows and macOS in its "
            "settings, as scView gets what its virtual machine has).")


@dataclass
class Cached:
    count: int = 0
    bytes: int = 0


def cached_others(dm, dataset_id: str | None = None) -> Cached:
    """Other datasets the cache holds in memory (up to 3 stay open), and their size."""
    c = Cached()
    for did in dm.cached_ids():
        if did == dataset_id:
            continue
        adaptor = dm.get_dataset(did)
        if adaptor is None:
            continue
        try:
            a = adaptor.adata
            h = header_from_adata(a) if not getattr(a, "isbacked", False) else read_header(
                adaptor.h5ad_path)
            c.count += 1
            c.bytes += resident_bytes(h)
        except Exception:  # noqa: BLE001 — a dataset being replaced: leave it out
            continue
    return c


def make_verdict(*, need: int, peak: int, stage: str, seconds: tuple[float, float],
                 avail: Available, cached: Cached, lead: str,
                 breakdown: list[tuple[str, int]], notes: list[str],
                 alternatives: list[tuple[str, int]]) -> dict:
    """The verdict and its one sentence. ``lead`` is the sentence's start (what is
    estimated); ``alternatives`` are (suggestion, need with it) for won't fit."""
    level = _level(need, avail.available_bytes)
    sentence = f"{lead}, {time_text(seconds)}. {_budget_text(avail)}"
    if cached.count:
        sentence += (f" {cached.count} other open dataset{'s' if cached.count > 1 else ''} "
                     f"hold{'s' if cached.count == 1 else ''} {_about(cached.bytes)}.")
    frees = (avail.available_bytes is not None and need > avail.available_bytes * FITS_BELOW
             and need <= (avail.available_bytes + 0.9 * cached.bytes) * FITS_BELOW)
    suggestions: list[str] = []
    if level in ("tight", "wont_fit"):
        r = _raise_limit(avail, need)
        if r:
            suggestions.append(r)
        if cached.count:
            suggestions.append(
                f"Close the {cached.count} other open dataset{'s' if cached.count > 1 else ''} "
                f"({_about(cached.bytes)}): restart scView and open only this one"
                + ("; that would free enough." if frees else "."))
        for text, alt_need in alternatives:
            if alt_need < need * 0.95:
                suggestions.append(f"{text} ({_about(max(0, alt_need))} more at its peak).")
    return {
        "level": level, "sentence": sentence, "need_bytes": int(max(0, need)),
        "peak_bytes": int(peak), "peak_stage": stage,
        "available_bytes": avail.available_bytes, "limit_bytes": avail.limit_bytes,
        "used_bytes": avail.used_bytes, "limited": avail.limited, "source": avail.source,
        "seconds": list(seconds),
        "cached": {"count": cached.count, "bytes": cached.bytes, "frees_enough": bool(frees)},
        "suggestions": suggestions, "error_band": ERROR_BAND,
        "breakdown": [{"label": k, "bytes": int(v)} for k, v in breakdown],
        "notes": [n for n in notes if n],
    }


# ---------------------------------------------------------------------------
# The two questions the endpoints ask
# ---------------------------------------------------------------------------

_UNIT_FORMATS = {"tenx_mex": "mex", "tenx_h5": "10x_h5", "anndata": "h5ad", "loom": "loom",
                 "zarr": "zarr", "dense_table": "table", "seurat": "rds"}


def read_unit_header(unit, genes_in_rows: bool = True) -> Header:
    """The header of one ingest unit (a MEX unit's from its matrix.mtx)."""
    fmt = _UNIT_FORMATS.get(unit.format.value, None)
    if fmt is None or not unit.files:
        return Header("unknown", note="The file is not a format scView reads.")
    f = next((x for x in unit.files if x.role.value == "matrix"), None) if fmt == "mex" else unit.files[0]
    if f is None:
        return Header("unknown", note="The matrix file of this 10x sample is missing.")
    return read_header(f.path, fmt, genes_in_rows)


def _load_lead(h: Header, est: LoadEstimate) -> str:
    shape = (f"{h.n_cells:,} cells x {h.n_genes:,} genes"
             + (" (estimated)" if h.estimated else "") if h.n_cells and h.n_genes else "")
    head = f"{shape}: " if shape else ""
    if h.guess:
        return f"{head}Loading takes {_about(est.peak_bytes)} (a guess from the file size)"
    return (f"{head}Loading takes {_about(est.peak_bytes)} at its peak ({est.peak_stage}) "
            f"and {_about(est.resident_bytes)} once open")


def _half_cells(h: Header) -> Header:
    """The header with half the cells (the subsample suggestion)."""
    import copy

    h2 = copy.deepcopy(h)
    h2.n_cells = (h.n_cells or 0) // 2
    for mm in h2.matrices:
        mm.n_cells //= 2
        mm.nnz //= 2
    h2.obsm_bytes //= 2
    h2.obsp_bytes //= 2
    h2.file_bytes //= 2
    return h2


def session_preflight(mgr, sid: str, dm) -> dict:
    """Per staged unit (and the merge of several): header, load estimate, verdict."""
    from scview.core.ingestion.bundling import UnitFormat, build_bundle

    bundle = build_bundle(mgr.staged_paths(sid))
    opts = mgr.get_options(sid)
    avail = available_memory()
    cached = cached_others(dm)
    units, residents, peaks, secs = [], [], [], [0.0, 0.0]
    for u in bundle.units:
        if u.format == UnitFormat.unknown or not u.complete:
            continue
        h = read_unit_header(u, opts.genes_in_rows)
        est = estimate_load(h)
        half = estimate_load(_half_cells(h)) if not h.guess else None
        v = make_verdict(
            need=est.peak_bytes, peak=est.peak_bytes, stage=est.peak_stage, seconds=est.seconds,
            avail=avail, cached=cached, lead=_load_lead(h, est), breakdown=est.breakdown,
            notes=est.notes,
            alternatives=[("Subsample the cells before loading: half of them", half.peak_bytes)]
            if half else [])
        units.append({"label": u.label, "format": u.format.value, "header": h.to_dict(),
                      "load": {"resident_bytes": est.resident_bytes, "peak_bytes": est.peak_bytes,
                               "peak_stage": est.peak_stage, "seconds": list(est.seconds)},
                      "verdict": v})
        residents.append(est.resident_bytes)
        peaks.append(est.peak_bytes)
        secs = [secs[0] + est.seconds[0], secs[1] + est.seconds[1]]
    merge = None
    if len(units) > 1:
        # Every sample is loaded, then merged into one more copy, saved and reopened.
        peak = sum(residents) * 2
        merge = make_verdict(
            need=peak, peak=peak, stage="merging the samples", seconds=(secs[0] * 1.5, secs[1] * 1.5),
            avail=avail, cached=cached,
            lead=(f"Merging the {len(units)} samples takes {_about(peak)} at its peak "
                  f"and {_about(sum(residents))} once open"),
            breakdown=[("The samples, loaded", sum(residents)), ("The merged copy", sum(residents))],
            notes=["Merging holds every sample and the merged copy at once."],
            alternatives=[])
    return {"session_id": sid, "units": units, "merge": merge}


def _params_with(params, **changes):
    import copy

    p = copy.copy(params)
    for k, v in changes.items():
        setattr(p, k, v)
    return p


def dataset_preflight(dm, dataset_id: str, steps: list[str], overrides: dict | None = None) -> dict:
    """The run estimate for ``steps`` on a dataset, with its verdict. The shape comes
    from the cached dataset, else from the header of its file (never loaded here)."""
    from scview.core.pipeline import PipelineParams

    params = PipelineParams()
    for k, v in (overrides or {}).items():
        if hasattr(params, k):
            setattr(params, k, v)
    adaptor = dm.get_dataset(dataset_id)
    in_memory = adaptor is not None
    h = None
    # Not while a run changes it in a worker thread: its file says what it was.
    if in_memory and not dm.is_busy(dataset_id):
        a = adaptor.adata
        if not getattr(a, "isbacked", False):
            try:
                h = header_from_adata(a)
            except (RuntimeError, ValueError):  # changed under us
                h = None
    if h is None:
        files = dm.h5ad_candidates(dataset_id)
        if not files:
            raise FileNotFoundError(dataset_id)
        h = read_header(files[0], "h5ad")
    warm = process_warm()
    est = estimate_run(h, steps=steps, params=params, warm=warm)
    held = est.start_bytes if in_memory else 0
    need = est.peak_bytes - held
    avail = available_memory()
    cached = cached_others(dm, dataset_id)

    alternatives: list[tuple[str, int]] = []
    if "doublet_detection" in steps:
        alt = estimate_run(h, steps=[s for s in steps if s != "doublet_detection"],
                           params=params, warm=warm)
        alternatives.append(("Skip doublet detection", alt.peak_bytes - held))
    if set(steps) & {"highly_variable_genes", "pca"} and params.n_top_genes > 1000:
        alt = estimate_run(h, steps=steps, params=_params_with(params, n_top_genes=1000), warm=warm)
        alternatives.append((f"Use 1,000 variable genes instead of {params.n_top_genes:,}",
                             alt.peak_bytes - held))
    alt = estimate_run(_half_cells(h), steps=steps, params=params, warm=warm)
    alternatives.append(("Subsample the cells: half of them", alt.peak_bytes - held))

    stage = STEP_LABELS.get(est.peak_step, est.peak_step)
    if need > 0:
        about = _about(need)
        lead = f"{about[0].upper()}{about[1:]} more than now at its peak ({stage})"
    else:
        lead = f"No more memory than this dataset holds now at its peak ({stage})"
    breakdown = [("This dataset now" if in_memory else "This dataset once loaded", est.start_bytes),
                 (f"At the peak ({stage})", est.peak_bytes),
                 ("After the run (saved and reopened)", est.end_bytes)]
    notes = []
    unmeasured = [STEP_LABELS[e.step] for e in est.steps if not e.measured]
    if unmeasured:
        notes.append("Not measured, estimated from what the step allocates: "
                     + ", ".join(unmeasured) + ".")
    if "filtering" in steps:
        notes.append("Assumes filtering keeps every cell, as the data were processed before."
                     if _State.from_header(h).processed else
                     "Assumes filtering keeps 90% of the cells and 80% of the genes.")
    if not warm:
        notes.append("Includes about 0.5 GB the first run in a fresh scView keeps for the "
                     "libraries it loads.")
    v = make_verdict(need=need, peak=est.peak_bytes, stage=stage, seconds=est.seconds,
                     avail=avail, cached=cached, lead=lead, breakdown=breakdown, notes=notes,
                     alternatives=alternatives)
    return {"dataset_id": dataset_id, "in_memory": in_memory, "header": h.to_dict(),
            "steps": [e.to_dict() for e in est.steps], "verdict": v}
