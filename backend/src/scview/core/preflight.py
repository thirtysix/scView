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
  replaces X, filtering cuts each matrix once and the opened copy gives its matrices up,
  scaling sets .raw to X itself and PCA makes the variable genes dense for its own length,
  core/scaling.py).
- ``estimate_step_peak(step, shape, nnz, dense, params, *, other_bytes=0,
  dtype="float32", nonzero=None) -> int``: one step, pure, for a hook before each step;
  ``step_peak_for(adata, step, params)`` reads the same from a live AnnData.
- ``make_verdict``: fits (under 70% of the memory free), tight (70-100%), won't fit,
  with suggestions in this order: raise SCVIEW_MEMORY_GB, close the other datasets
  scView holds, skip doublet detection, fewer variable genes, subsample; each
  alternative is re-estimated and offered only when it lowers the peak.

Calibration

The coefficients come from measured runs, not guesses: one fresh container per
dataset, 2 CPUs (scview-backend image, 2026-10-21; re-measured 2026-10-22 once doublet
detection scored the counts alone and filtering cut each matrix once), memory = the
cgroup's anon memory (which counts the Leiden child process) sampled every 0.1 s, or the
process's peak RSS where that is higher (short reads), minus the process before loading
(scanpy imported, 0.17 GiB). Runs are the "Reprocess from scratch" steps with the UI's
marker columns.
GiB; err = estimate / measured - 1.

    dataset            cells   genes   X stored values      load meas/est  err   run peak meas/est  err  at
    pbmc 1k 10x .h5     1,222  33,538  2.5M sparse         0.04 / 0.04   -1%   0.63 / 0.70   +11%  (late steps)
    paul15             2,730   3,451  dense (+2 dense)    0.13 / 0.14   +5%   0.63 / 0.71   +13%  (late steps)
    Kang              13,836  14,053  dense f64 (+2 sp.)  1.76 / 1.74   -1%   2.65 / 2.72    +3%  doublets
    ovary 10k sub.    10,000  20,082  12.1M sp. + dense   0.91 / 0.88   -3%   4.36 / 4.06    -7%  doublets
    ovary 20k sub.    20,000  20,082  24.1M sp. + dense   1.75 / 1.72   -2%   8.07 / 7.95    -2%  doublets
    ovary_raw         37,104  20,082  44.7M sp. + dense   3.22 / 3.17   -2%  13.90 / 14.61   +5%  doublets
    hashing PBMC *    20,238  22,628  dense (+2 sp.)      2.05 / 2.03   -1%   3.36 / 3.57    +6%  filtering

* filtered again after scaling (doublets by either source, per-type % mito with the
floor; 23% of the cells go): filtering alone. Filtering's own peak is within 2% on the
ovary runs (2.38 / 2.34, 4.53 / 4.51, 8.28 / 8.23) and 0% on Kang (0.50).
Doublet detection used to score a copy of the whole dataset: ovary_raw's dense 3 GB counts
layer beside the dense X a reset makes took it to 20.86 GiB (killed at an 18 GiB limit),
Kang's to 3.00, the subsamples' to 6.17 and 11.73. Filtering used to copy every matrix
twice and keep the dataset's own: +11.15 GiB on ovary_raw (killed at 16 GiB), +4.28 on
the hashing PBMC (killed at 6 GB, with an estimate of +0.34 that took data filtered before
to keep every cell). The small datasets are over by fixed costs (library imports,
caches). Ingest (read, save as gzip h5ad, reopen), peak: pbmc 10x .h5 0.056 / 0.060
(+7%), the same as MEX 0.144 / 0.148 (+3%), a 1,222 x 5,000 CSV 0.189 / 0.203 (+7%),
paul15 0.140 / 0.147 (+5%), Kang 1.79 / 1.91 (+7%). Not measured: loom and zarr reads
(8 and 0 bytes per value, reasoned), RDS (a guess), batch correction, the uncorrected
UMAP and cell-type annotation (reasoned from what they allocate; the estimate says so).
Header reads take 3-35 ms (Kang's 1.9 GB h5ad: 17 ms).

Scaling, PCA and markers were re-measured 2026-10-23, once scaling stopped making X dense
(core/scaling.py): each step alone on the saved state before it, warm, 2 CPUs, process RSS
every 20 ms, GiB over the step's start, measured / estimated. paul15: PCA 0.048 / 0.045,
markers (dense) 0.110 / 0.113. Kang: PCA 0.240 / 0.226, markers 0.220 / 0.222. 100k
neurons (88,640 x 19,824 after filtering, 187M stored values): PCA 1.43 / 1.45, markers
2.85 / 2.86 (the old model said 1.54, as Roihu found: 1.8x under). Markers by blocks of
genes (core/rank_genes.py, 2026-10-24, the same way on the states a live run saved):
Kang 0.105 / 0.116 in 0.8 s, 100k neurons 0.329 / 0.315 in 18 s. Scaling: 0-0.08
against 0.01-0.12 (it was +14.5 GiB at 100k, +38 at 250k). In a whole run the 100k
peak is doublet detection, 15.5 GiB (scaling's 18.3 GiB before).

Time: the same runs' seconds per step, as a function of cells, stored values and
groups, measured at 2 CPUs and scaled by the square root of the CPU quota; shown as
half to twice the point estimate, which held every measured run (Kang 124 s within
73-290 s, ovary 20k 369 s within 225-898 s, ovary_raw 915 s within 394-1574 s).
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

from scview.core import scaling as _scaling

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
    # Dense only: its non-zero values, counted on a sample of rows (a live AnnData, or the
    # first rows of a file with no sparse matrix), or taken from the sparse matrices beside
    # it (``_State``); None: not known. Doublet detection sizes dense counts by it.
    nonzero: int | None = None

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

    format: str  # h5ad | 10x_h5 | mex | loom | zarr | table | rds | h5seurat | unknown
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
    # In large-dataset layout (uns["scview_large"], core/large_layout.py): opens backed
    # whatever its size and is never read into memory (core/large_run.py).
    large: bool = False
    # Variable genes (var["highly_variable"]), what PCA reads; None: no such column (every gene).
    n_hvg: int | None = None
    # An obs clustering a reset to counts keeps (``CLUSTER_COLUMNS``): the ambient RNA
    # step estimates rho on it instead of clustering the cells itself.
    has_clusters: bool = False
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


# Rows of an on-disk dense matrix whose non-zeros are counted (the first ones: one read).
NONZERO_HEAD_ROWS = 64


def _h5_head_nonzero(node, n_cells: int) -> int | None:
    """Non-zero values of a dense h5 dataset, counted on its first rows and scaled."""
    rows = min(n_cells, NONZERO_HEAD_ROWS)
    try:
        block = np.asarray(node[:rows])
    except Exception:  # noqa: BLE001 - unreadable: not known
        return None
    return int(round(np.count_nonzero(block) * n_cells / rows)) if rows else None


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
        h.has_clusters = isinstance(obs, h5py.Group) and any(c in obs for c in CLUSTER_COLUMNS)
        h.var_columns = _n_columns(f.get("var"))
        # var is a group of columns, or in older files one compound dataset (Kang's).
        var = f.get("var")
        hv = var.get("highly_variable") if isinstance(var, h5py.Group) else None
        if isinstance(hv, h5py.Dataset) and hv.ndim == 1 and hv.dtype.kind in "bui":
            h.n_hvg = int(np.count_nonzero(hv[()]))
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
        if h.matrices and not any(mm.sparse for mm in h.matrices):
            # No sparse matrix to take the share of non-zeros from (``_State``): count it
            # on the first rows of each dense one (paul15: 2 ms).
            for mm in h.matrices:
                mm.nonzero = _h5_head_nonzero(f[mm.where], mm.n_cells)
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
            from scview.core.large_layout import KEY as LARGE_KEY

            h.large = LARGE_KEY in f["uns"]
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
    # scanpy reads the counts as float32 (cells x genes CSR); scipy's indices are int64
    # once the values outnumber int32 (the 1.3M-cell brain: 2.6e9).
    idx = "int64" if max(nnz, n_genes) > np.iinfo(np.int32).max else "int32"
    h.matrices.append(Matrix("X", n_cells, n_genes, True, nnz, "float32", idx))
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


def _read_h5seurat(path: Path) -> Header:
    """SeuratDisk's .h5Seurat: X is the active assay's counts (core/ingestion/h5seurat.py),
    read as float64 compressed rows; the reductions go in obsm."""
    import h5py

    from scview.core.ingestion import h5seurat

    h = Header("h5seurat")
    s = h5seurat.summary(path)
    h.n_cells, h.n_genes = s["n_cells"], s["n_genes"]
    with h5py.File(path, "r") as f:
        node = h5seurat.matrix_node(f["assays"][s["assay"]])
        if isinstance(node, h5py.Group):
            data, idx = node["data"], node["indices"]
            h.matrices.append(Matrix("X", h.n_cells, h.n_genes, True, int(data.shape[0]),
                                     data.dtype.name, idx.dtype.name))
        else:
            h.matrices.append(Matrix("X", h.n_cells, h.n_genes, False, h.n_cells * h.n_genes,
                                     node.dtype.name))
        for name in s["reductions"]:
            emb = f["reductions"][name].get("cell.embeddings")
            if emb is not None and emb.ndim == 2:
                k = int(min(emb.shape))
                h.obsm[f"X_{name}"] = [h.n_cells, k]
                h.obsm_bytes += h.n_cells * k * 8
        h.obs_columns = len(f["meta.data"].attrs.get("colnames", []))
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
                if {"assays", "meta.data", "cell.names"} <= keys and "active.assay" in f.attrs:
                    return "h5seurat"
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
    "zarr": _read_zarr, "rds": _read_rds, "h5seurat": _read_h5seurat,
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
    from scview.core.anndata_adaptor import opens_backed

    h.backed = fmt == "h5ad" and (opens_backed(h.file_bytes) or h.large)
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
            return Matrix(where, int(r), int(c), False, int(r) * int(c), np.dtype(m.dtype).name,
                          nonzero=_dense_nonzero(m))
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
    h.has_clusters = any(c in adata.obs.columns for c in CLUSTER_COLUMNS)
    if "highly_variable" in adata.var.columns:
        h.n_hvg = int(np.count_nonzero(adata.var["highly_variable"].to_numpy()))
    return h


# Rows of a dense matrix whose non-zeros are counted (evenly spaced; a strided view, no copy).
NONZERO_SAMPLE_ROWS = 512


def _dense_nonzero(m) -> int | None:
    """Non-zero values of a dense in-memory matrix, counted on ``NONZERO_SAMPLE_ROWS``
    evenly spaced rows and scaled to all of them (ovary's 3 GB counts: ~10 ms)."""
    if not isinstance(m, np.ndarray) or m.ndim != 2 or not m.size:
        return None
    sample = m[::max(1, m.shape[0] // NONZERO_SAMPLE_ROWS)]
    return int(round(np.count_nonzero(sample) * m.shape[0] / sample.shape[0]))


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
    "h5seurat": 8.0,
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


# What a backed open leaves on disk (anndata reads X and .raw's X when used; the
# layers, obsm and the rest are read into memory).
ON_DISK_WHEN_BACKED = ("X", "raw/X")


def _in_memory(h: Header, mm: Matrix) -> bool:
    return not (h.backed and mm.where in ON_DISK_WHEN_BACKED)


def _matrices_bytes(h: Header) -> int:
    return sum(mm.nbytes for mm in h.matrices if _in_memory(h, mm))


def backed_bytes(h: Header) -> int:
    """Memory that reading a backed dataset's on-disk matrices into memory adds (0 for
    one opened in memory): what ``core/working_copy.py`` makes room for."""
    return sum(mm.nbytes for mm in h.matrices if not _in_memory(h, mm))


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
    values = sum(mm.nnz for mm in h.matrices if _in_memory(h, mm))
    read = int(values * READ_BYTES_PER_VALUE.get(h.format, 8.0))
    breakdown = [(f"{mm.where} ({'sparse' if mm.sparse else 'dense'}, {mm.dtype})", mm.nbytes)
                 for mm in h.matrices if _in_memory(h, mm)]
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
        from scview.core.anndata_adaptor import BACKED_MIN_BYTES

        notes.append(f"Files over {BACKED_MIN_BYTES / GiB:.3g} GB open from disk: the matrix X is "
                     "read when needed, not held in memory (processing reads it in first).")
    return LoadEstimate(res, peak, stage, _range(secs, cpus), breakdown, notes)


# ---------------------------------------------------------------------------
# Run estimate
# ---------------------------------------------------------------------------

STEP_LABELS = {
    "reset_to_counts": "reset to counts", "qc_metrics": "QC metrics",
    "doublet_detection": "doublet detection", "filtering": "filtering",
    "ambient_correction": "ambient RNA correction",
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
    "reset_to_counts", "qc_metrics", "doublet_detection", "filtering", "ambient_correction",
    "normalization",
    "log_transform", "highly_variable_genes", "scaling", "pca", "neighbors", "clustering",
    "embeddings", "marker_genes", "enrichment", "cell_cycle", "dissociation_stress", "save",
}

# Filtering in MAD mode removed 11% of the cells on the ovary subsample; genes seen in
# fewer than 3 cells go too (22% there, a few on processed data: 3% of the hashing PBMC's
# when its doublets and per-type mito outliers went). Data filtered before can lose cells
# again (a new doublet source, hashing, the mito floor): taken to lose as many.
FILTER_CELLS_KEPT = 0.9
FILTER_GENES_KEPT_RAW = 0.8
# Cutting the matrices once each: seconds per byte they hold (hashing PBMC 4 s for 2.0 GiB,
# ovary 10k / 20k / 37k 3.8, 8.2, 31 s for 1.5, 3.0, 5.5 GiB).
FILTER_SECONDS_PER_BYTE = 3e-9

# Scrublet (core/scrublet_lean.py: sc.pp.scrublet's steps on a sparse matrix of the counts
# alone, never a copy of the dataset), run in each capture channel alone, one after another
# (core/doublets.py): the step's peak is its largest channel's, plus the per-cell results.
# While it picks its variable genes it holds about six copies of the channel's counts
# (filtered, raw, normalised, log, expm1). Then it z-scores the observed and 2x simulated
# cells' variable genes densely in float64, and scipy makes a CSC of each fully dense
# result: 144 bytes per cell x variable gene (peak RSS over the process before, one channel
# of 750 to 40,000 cells of the mouse brain, 2,000 to 13,836 of Kang, paul15: 132-154).
# How many variable genes its cut-offs keep is the dataset's: about 1,450 of Kang's 14,053
# genes, 1,300 of the mouse brain's 27,998, 1,900 of ovary's 20,082, 750 of paul15's 3,451,
# and more in a small channel (the brain: 1,884 at 750 cells, 1,207 at 100,000). So on a
# dataset in memory the step's own selection is run on up to DBL_HVG_SAMPLE cells of the
# largest channel (``doublet_plan``); from a header alone they are taken as 10% of the
# genes, 12.8 bytes per cell x gene (Kang 11%, ovary 9.5%; the brain's 4.7% is over 2x).
DBL_SPARSE_COPIES = 6.0
DBL_PER_CELL_HVG = 144
DBL_DENSE_PER_VALUE = 12.8  # per cell x gene, when the variable genes are not counted
DBL_HVG_SAMPLE = 5000
# Per cell, beside the largest channel: the scores, calls, channel codes and their sort,
# and the obs columns written (core/doublets.py run_scrublet).
DBL_OUT_PER_CELL = 48
# Seconds per channel of c cells: 2e-5 x c^1.4 with exact neighbours (one channel of the
# brain: 0.3 s at 750 cells, 9.7 s at 10k, 68.6 s at 40k; Kang 8.2 s at 13.8k; its 133
# wells 43 s in all), 6.9e-5 x c^1.4 with pynndescent's over doublets.EXACT_KNN_MAX_CELLS
# (the 100k brain as one channel: 688 s), and 0.2 s each.
DBL_SECONDS_COEF = 2e-5
DBL_SECONDS_COEF_APPROX = 6.9e-5
DBL_SECONDS_POWER = 1.4
DBL_SECONDS_PER_CHANNEL = 0.2
# The scDblFinder-style classifier (core/doublet_classifier.py), in the same channels after
# Scrublet: its largest channel's counts as float32 CSR about twice over (the channel, its
# filtered genes, the selected genes of real and artificial cells), plus per point (cells and
# artificial doublets: as many as cells, 1,500 to 25,000) the dense selected genes, the PCs,
# neighbours, features and the co-expression score. Peak RSS over the process before
# (2 CPUs): Kang 6.6k / 13.8k cells 306 / 619 MB, hgmm 11.5k 1,380 MB, paul15 179 MB, the
# brain 2k-50k 181-3,277 MB; this model is within -7 % .. +18 % of each. Seconds: 0.24 ms
# per point (Kang 3.3 / 6.2 s, the brain at 50k 29 s: slower per point there).
DBL_CLS_CSR_COPIES = 2.2
DBL_CLS_PER_POINT = 18_000
DBL_CLS_FIXED = 50 * MiB
DBL_CLS_MIN_ARTIFICIAL = 1500  # doublet_classifier.MIN_ARTIFICIAL / MAX_ARTIFICIAL
DBL_CLS_MAX_ARTIFICIAL = 25_000
DBL_CLS_SECONDS_PER_POINT = 2.4e-4
# Steps that import or JIT-compile libraries the first time they run in a process
# (numba, umap, pynndescent, the MSigDB cache): held afterwards.
WARMUP = {"doublet_detection": 150 * MiB, "neighbors": 100 * MiB, "marker_genes": 100 * MiB,
          "enrichment": 80 * MiB, "embeddings": 20 * MiB}
WARMUP_MODULE = "umap"  # imported by the first neighbour graph: warm-up already paid
NEIGHBORS_PER_CELL = 12_000
# Scaling (core/scaling.py) holds one chunk of a sparse X's stored values at a time
# (int64 column numbers, the values and their squares in float32 and float64: ~33 bytes
# each), or a dense X's genes STAT_BLOCK at a time.
SPARSE_STATS_CHUNK = _scaling.SPARSE_CHUNK
STAT_BLOCK = _scaling.STAT_BLOCK
SCALING_SECONDS_PER_VALUE = 5e-9  # per stored value of a sparse X, per cell x gene of a dense one
# Regress-out (scaling ``regress_out``): the fit and the residuals' statistics, per cell x
# variable gene (Kang 13,836 x 2,000, two covariates: +0.25 s at 2 CPUs, 2026-10-24); at
# PCA the fit taken off again, per cell x gene x regressor (too small to measure there).
REGRESS_SECONDS_PER_VALUE = 9e-9
REGRESS_PCA_SECONDS_PER_VALUE = 1e-9
# PCA: copies of the n x variable-genes float32 matrix at its peak (the matrix and
# scikit-learn's centred copy), bytes per cell x component (ARPACK's vectors, the
# scores), and the variable genes' share of a sparse X's stored values relative to
# their share of the genes (they are the more expressed ones).
PCA_DENSE_COPIES = 2.1
PCA_BYTES_PER_COMP = 16
PCA_CUT_SHARE = 2.0
PCA_SECONDS_PER_VALUE = 4.5e-8
# Markers (Wilcoxon on the log values) by blocks of genes (core/rank_genes.py): one block
# of values in flight (matrix_genes.block_entries, ~70 bytes a value) and per gene x group
# statistics, never a copy of the matrix (a CSR whose rows are in no gene order adds a
# byte per stored value, not counted). scanpy's own took 2.85 GiB on 100k neurons, where
# this takes 0.33. Seconds per stored value (a dense matrix: per cell x gene) per column.
MARKERS_MEMORY_PER_VALUE = 70
MARKERS_SECONDS_PER_VALUE = 8e-8


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
    n_hvg: int | None = None  # the variable genes PCA reads; None: every gene
    raw_is_x: bool = False  # .raw set to X by the scaling step: one matrix until saved
    clusters: bool = False  # a clustering the ambient RNA step can estimate rho on
    bbknn_graph: bool = False  # batch correction made the graph (BBKNN): neighbours keep it

    @classmethod
    def from_header(cls, h: Header) -> "_State":
        mats = _with_nonzero(h.matrices)
        x = next((mm for mm in mats if mm.where == "X"), None)
        raw = next((mm for mm in mats if mm.where == "raw/X"), None)
        layers = {mm.where.split("/", 1)[1]: mm for mm in mats if mm.where.startswith("layers/")}
        return cls(h.n_cells or 0, h.n_genes or 0, x, layers, raw,
                   h.obsm_bytes + h.obsp_bytes + h.varm_bytes + h.uns_bytes + _frames_bytes(h)
                   + (h.n_cells or 0) * LOAD_PER_CELL + LOAD_FIXED,
                   processed=bool(h.obsm) or h.matrix("raw/X") is not None, n_hvg=h.n_hvg,
                   clusters=h.has_clusters)

    def matrices(self) -> list[Matrix]:
        return [mm for mm in [self.x, self.raw, *self.layers.values()] if mm is not None]

    def resident(self) -> int:
        shared = self.raw.nbytes if self.raw_is_x and self.raw is not None else 0
        return sum(mm.nbytes for mm in self.matrices()) - shared + self.other + self.held

    def pca_genes(self) -> int:
        """The genes PCA reads: the variable genes, else every gene."""
        return self.m if self.n_hvg is None else max(1, min(self.m, self.n_hvg))

    def xb(self) -> int:
        return self.x.nbytes if self.x is not None else 0

    def others(self) -> int:
        """Bytes of the layers and .raw (a copy of the dataset copies them too)."""
        return sum(mm.nbytes for mm in self.matrices()) - self.xb()

    def lognorm(self) -> Matrix | None:
        """What markers and scores read: .raw (log-normalised) when set, else X."""
        return self.raw or self.x


def _with_nonzero(mats: list[Matrix]) -> list[Matrix]:
    """The matrices, a dense one whose non-zeros are not known given the share of the
    sparse ones beside it (a normalised X keeps its counts' zeros); none: every value."""
    from dataclasses import replace

    shares = [mm.nnz / (mm.n_cells * mm.n_genes) for mm in mats
              if mm.sparse and mm.n_cells and mm.n_genes]
    if not shares:
        return list(mats)
    share = max(shares)
    return [replace(mm, nonzero=int(mm.nnz * share)) if not mm.sparse and mm.nonzero is None
            else mm for mm in mats]


def _like(mm: Matrix, where: str, n: int, m: int) -> Matrix:
    """``mm`` resized to n x m, its stored values scaled with it."""
    share = (n * m) / max(1, mm.n_cells * mm.n_genes)
    nnz = int(mm.nnz * share) if mm.sparse else n * m
    nonzero = None if mm.nonzero is None else int(mm.nonzero * share)
    return Matrix(where, n, m, mm.sparse, nnz, mm.dtype, mm.index_dtype, nonzero=nonzero)


def _csr_bytes(mm: Matrix) -> int:
    """Bytes of ``mm`` as CSR in its own dtype (int32 indices); a dense one by its non-zeros
    (every value when they are not known)."""
    if mm.sparse:
        return mm.nbytes
    nz = mm.nnz if mm.nonzero is None else mm.nonzero
    return nz * (np.dtype(mm.dtype).itemsize + 4) + (mm.n_cells + 1) * 4


def _p(params, name: str, default):
    return getattr(params, name, default) if params is not None else default


# Each step: (state) -> (transient bytes over the state's resident, state after, seconds at 2 CPUs).
def _step_reset(s: _State, params) -> tuple[int, _State, float]:
    layers = dict(s.layers)
    # Counts an ambient RNA correction replaced come back as the counts (core/ambient.py).
    original = layers.pop("counts_uncorrected", None)
    if original is not None:
        layers["counts"] = _like(original, "layers/counts", s.n, s.m)
    counts = layers.get("counts")
    if counts is not None:
        x = _like(counts, "X", s.n, s.m)
        return counts.nbytes, _State(s.n, s.m, x, layers, None, s.other // 2, s.held,
                                     s.processed, clusters=s.clusters), 0.05 + counts.nbytes * 1.5e-9
    if s.raw is not None:  # a new object: the adaptor keeps the old one
        x = _like(s.raw, "X", s.n, s.raw.n_genes)
        return s.raw.nbytes, _State(s.n, s.raw.n_genes, x, {}, None, s.other // 2,
                                    s.held + s.resident(), s.processed,
                                    clusters=s.clusters), 0.05 + s.raw.nbytes * 1.5e-9
    return 0, s, 0.1


def _step_qc(s: _State, params) -> tuple[int, _State, float]:
    if s.x is None:
        return 0, s, 0.1
    t = int((0.6 if s.x.sparse else 1.0) * s.xb())
    s.other += s.n * 6 * 8
    return t, s, 0.1 + s.x.nnz * (1.8e-7 if s.x.sparse else 1.2e-8)


@dataclass
class DoubletPlan:
    """What the doublet step will run on: each capture channel's cells (core/doublets.py
    ``channels``) and, when counted, Scrublet's variable genes in the largest channel."""

    channels: list[int]
    hvg: int | None = None


def _step_doublet(s: _State, params, plan: DoubletPlan | None = None) -> tuple[int, _State, float]:
    counts = s.x
    beside = s.layers.get("counts") or s.raw
    if s.processed and s.x is not None and not s.x.sparse and beside is not None:
        # A processed (scaled, dense) X: the counts beside it are scored
        # (pipeline._run_doublet_detection).
        counts = beside
    if counts is None:
        return 0, s, 0.1
    from scview.core.doublets import EXACT_KNN_MAX_CELLS
    from scview.core.scrublet_lean import MIN_CHANNEL_CELLS

    # No plan: all cells as one channel (a header alone does not say which cells share one).
    sizes = [c for c in (plan.channels if plan is not None else [s.n]) if c >= MIN_CHANNEL_CELLS]
    largest = max(sizes, default=0)
    # Only the channel's counts, as CSR (a dense matrix made so by blocks): the layers,
    # .raw and a dense X are not copied.
    copies = int(DBL_SPARSE_COPIES * _csr_bytes(counts) * largest / max(1, counts.n_cells))
    if plan is not None and plan.hvg:
        dense = int(DBL_PER_CELL_HVG * largest * plan.hvg)
    else:
        dense = int(DBL_DENSE_PER_VALUE * largest * counts.n_genes)
    method = _doublet_method(params)
    s.other += s.n * 16 * (2 if method == "both" else 1)
    peak, secs = 0, 0.0
    if method in ("scrublet", "both"):
        peak = max(copies, dense)
        secs = 3.0 + sum((DBL_SECONDS_COEF if c <= EXACT_KNN_MAX_CELLS else DBL_SECONDS_COEF_APPROX)
                         * c ** DBL_SECONDS_POWER + DBL_SECONDS_PER_CHANNEL for c in sizes)
    if method in ("scdblfinder-style", "both"):  # after Scrublet, one channel at a time
        nnz = (counts.nnz if counts.nonzero is None else counts.nonzero) * largest / max(1, counts.n_cells)
        points = largest + (min(max(largest, DBL_CLS_MIN_ARTIFICIAL), DBL_CLS_MAX_ARTIFICIAL) if largest else 0)
        peak = max(peak, int(DBL_CLS_CSR_COPIES * nnz * 8 + DBL_CLS_PER_POINT * points + DBL_CLS_FIXED))
        secs += 1.0 + sum(DBL_CLS_SECONDS_PER_POINT * (c + min(max(c, DBL_CLS_MIN_ARTIFICIAL),
                                                              DBL_CLS_MAX_ARTIFICIAL))
                          + DBL_SECONDS_PER_CHANNEL for c in sizes)
    return peak + DBL_OUT_PER_CELL * s.n, s, secs


def _doublet_method(params) -> str:
    """The doublet step's detectors (core/doublets.py ``method_of``; Scrublet for anything
    it does not know: the step itself then fails)."""
    from scview.core.doublets import DoubletError, method_of

    try:
        return method_of(_p(params, "doublet_method", "scrublet"))
    except DoubletError:
        return "scrublet"


def doublet_plan(adata, params=None) -> DoubletPlan | None:
    """The doublet step's channels on a live AnnData (in memory), and Scrublet's variable
    genes counted on up to ``DBL_HVG_SAMPLE`` evenly spaced cells of the largest channel
    (its own selection, ``scrublet_lean.n_variable_genes``). None when they can't be read."""
    from scview.core import doublets, scrublet_lean
    from scview.core.pipeline import StepSkipped, _doublet_matrix

    try:
        ch = doublets.channels(adata, _p(params, "doublet_channel_key", "") or None)
    except doublets.DoubletError:
        return None
    sizes = [int(k) for k in ch.sizes]
    plan = DoubletPlan(sizes)
    if not sizes or max(sizes) < scrublet_lean.MIN_CHANNEL_CELLS:
        return plan
    try:
        counts, _ = _doublet_matrix(adata)
        idx = np.flatnonzero(ch.codes == int(np.argmax(sizes)))
        idx = idx[np.linspace(0, idx.size - 1, min(idx.size, DBL_HVG_SAMPLE)).astype(np.int64)]
        plan.hvg = scrublet_lean.n_variable_genes(scrublet_lean.channel_rows(counts, np.unique(idx)))
    except (StepSkipped, ValueError, TypeError, IndexError):
        pass
    return plan


def _cut_peak(old: list[Matrix], new: list[Matrix]) -> tuple[int, int]:
    """(most memory over the start, memory freed at the end) while matrices are cut one at
    a time, each old one let go once its cut exists (``pipeline.subset_lean``)."""
    peak = freed = 0
    for o, nw in zip(old, new):
        peak = max(peak, nw.nbytes - freed)
        freed += o.nbytes - nw.nbytes
    return peak, freed


def _step_filter(s: _State, params) -> tuple[int, _State, float]:
    n2 = int(s.n * FILTER_CELLS_KEPT)
    # Genes seen in too few cells: a fifth of raw data's; on data filtered before, a few, so
    # the genes still take one cut each (taken at every gene).
    m2 = s.m if s.processed else int(s.m * FILTER_GENES_KEPT_RAW)
    def resize(mm: Matrix | None, n: int, m: int | None = None) -> Matrix | None:
        return None if mm is None else _like(mm, mm.where, n, mm.n_genes if m is None else m)

    # The cells kept: X, each layer, then .raw cut once each (pipeline.subset_lean); the
    # dataset's own object gives its matrices up as it goes.
    x1, raw1 = resize(s.x, n2), resize(s.raw, n2)
    layers1 = {k: resize(v, n2) for k, v in s.layers.items()}
    old = [mm for mm in [s.x, *s.layers.values(), s.raw] if mm is not None]
    cells_peak, cells_freed = _cut_peak(old, [mm for mm in [x1, *layers1.values(), raw1]
                                               if mm is not None])
    # Then the genes: scanpy compares a dense X with 0 (a bool per value), then X and the
    # layers are cut once each (.raw keeps every gene).
    x2 = resize(x1, n2, m2)
    layers2 = {k: resize(v, n2, m2) for k, v in layers1.items()}
    genes_peak, _ = _cut_peak([mm for mm in [x1, *layers1.values()] if mm is not None],
                              [mm for mm in [x2, *layers2.values()] if mm is not None])
    mask = 0 if x1 is None else (x1.nnz if x1.sparse else n2 * s.m)
    transient = max(cells_peak, max(genes_peak, mask) - cells_freed)
    new = _State(n2, m2, x2, layers2, raw1, s.other, s.held, s.processed, n_hvg=s.n_hvg,
                 clusters=s.clusters)
    return max(0, transient), new, 0.2 + FILTER_SECONDS_PER_BYTE * sum(mm.nbytes for mm in old)


# Ambient RNA correction (core/ambient.py): the counts as CSR (kept as
# layers['counts_uncorrected']), a float64 copy of their values while each cell is worked
# on, and the corrected CSR that becomes X (and layers['counts'], a copy, when there is
# one). Without rho set, each sample's estimate (marker genes counted per cluster) and,
# with no clustering, a quick one of its own (a normalised copy, PCA, graph, Leiden).
# Calibrated 2026-10-24 after filtering, one fresh 2-CPU container per run, process RSS
# every 20 ms over the step's start, measured / estimated. Kang (13,836 cells, 9.65M stored
# values, X CSR float64): rho set 1.2 / 1.3 s and 0.181 / 0.189 GiB; rho estimated on the
# author's clusters 6.4 / 6.1 s; clustering the cells itself 47.8 / 46 s and 0.654 / 0.68
# GiB. paul15 (2,730 cells, dense float32, 2.39M non-zeros): rho set 0.3 / 0.7 s and
# 0.041 / 0.063 GiB; clustering itself 19.7 / 17 s and 0.311 / 0.35 GiB (its first neighbour
# graph and Leiden child in a fresh process are most of that: a fixed cost).
CLUSTER_COLUMNS = ("leiden", "louvain", "cluster", "clusters", "seurat_clusters")  # cluster_qc's
AMBIENT_SECONDS_FIXED = 0.3
AMBIENT_SECONDS_PER_CELL = 7e-5  # the subtraction, one cell at a time
AMBIENT_ESTIMATE_SECONDS = 1.5  # the estimate: posterior over 1,001 rho values, per sample
AMBIENT_ESTIMATE_PER_VALUE = 3.5e-7  # its passes over the counts (markers, Poisson tests)
AMBIENT_WORK_PER_VALUE = 21  # float64 values + the corrected CSR's values and indices
AMBIENT_QUICK_SECONDS = 8.0  # its own clustering (log, HVG, PCA, graph, Leiden): fixed...
AMBIENT_QUICK_SECONDS_PER_CELL = 2.3e-3  # ... and per cell
AMBIENT_QUICK_BYTES = 240 * MiB
AMBIENT_QUICK_PER_CELL = 20_000  # PCA, the graph, the Leiden child's share


def _step_ambient(s: _State, params) -> tuple[int, _State, float]:
    if s.x is None:
        return 0, s, 0.1
    nz = s.x.nnz if s.x.sparse else (s.x.nonzero if s.x.nonzero is not None else s.x.nnz)
    csr = Matrix("layers/counts_uncorrected", s.n, s.m, True, nz, s.x.dtype)
    transient = nz * AMBIENT_WORK_PER_VALUE + (0 if s.x.sparse else csr.nbytes)
    secs = AMBIENT_SECONDS_FIXED + s.n * AMBIENT_SECONDS_PER_CELL
    if _p(params, "ambient_rho", None) is None:
        secs += AMBIENT_ESTIMATE_SECONDS + nz * AMBIENT_ESTIMATE_PER_VALUE
        if not s.clusters:
            secs += AMBIENT_QUICK_SECONDS + s.n * AMBIENT_QUICK_SECONDS_PER_CELL
            transient += AMBIENT_QUICK_BYTES + s.n * AMBIENT_QUICK_PER_CELL
    # The originals stay as a layer (CSR), the corrected counts become X (and the counts).
    if "counts_uncorrected" not in s.layers:
        s.layers["counts_uncorrected"] = csr
    s.x = _like(csr, "X", s.n, s.m)
    if "counts" in s.layers:
        s.layers["counts"] = _like(csr, "layers/counts", s.n, s.m)
    return int(transient), s, secs


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
    s.n_hvg = min(s.m, int(_p(params, "n_top_genes", 2000)))
    return 2 * s.xb(), s, 0.05 + (s.x.nnz if s.x else 0) * (1e-8 if sparse else 2.2e-8)


def _step_scaling(s: _State, params) -> tuple[int, _State, float]:
    """Plans the matrix PCA reads (core/scaling.py): X stays as it is and becomes .raw
    too (one matrix until saved). The variable genes' statistics come from a sparse X's
    stored values (no dense copy), or a dense X's genes a block at a time."""
    if s.x is None:
        return 0, s, 0.1
    if s.raw is not None and not s.raw_is_x and "counts" not in s.layers:
        s.layers["counts"] = _like(s.raw, "layers/counts", s.n, s.m)  # counts kept from .raw
    k = s.pca_genes()
    if s.x.sparse:
        t = SPARSE_STATS_CHUNK * 33 + s.m * 24  # one chunk's column numbers, values, squares
        secs = 0.1 + s.x.nnz * SCALING_SECONDS_PER_VALUE
    else:
        t = 3 * s.n * min(k, STAT_BLOCK) * 4  # a dense block, its square and its copy
        secs = 0.1 + s.n * k * SCALING_SECONDS_PER_VALUE
    s.raw = Matrix("raw/X", s.n, s.m, s.x.sparse, s.x.nnz, s.x.dtype, s.x.index_dtype)
    s.raw_is_x = True
    keys = _regress_keys(params)
    if keys:
        # The fit and the residuals' statistics, a block of genes dense at a time
        # (core/scaling.py), and the cell-cycle scores first when they are covariates.
        p = len(keys) + 1
        b = min(k, STAT_BLOCK)
        t = max(t, s.n * b * 4 * 2 + _scaling.REGRESS_ROWS * b * 8 + s.n * p * 8)
        secs += s.n * k * REGRESS_SECONDS_PER_VALUE
        if set(keys) & set(_scaling.CELL_CYCLE_COVARIATES):
            t2, s, secs2 = _step_scores(s, params)
            t, secs = max(t, t2), secs + secs2
    return t, s, secs


def _regress_keys(params) -> list[str]:
    try:
        return _scaling.check_regress_out(_p(params, "regress_out", []))
    except ValueError:
        return []


def _step_pca(s: _State, params) -> tuple[int, _State, float]:
    """The matrix PCA reads (core/scaling.py: the variable genes, dense float32) and
    scikit-learn's centred copy of it, ARPACK's vectors; a sparse X's columns are cut
    out first. An older scaled X is read the same way (a masked copy)."""
    k = s.pca_genes()
    n_comps = int(_p(params, "n_comps", 50))
    dense = s.n * k * 4
    cut = int(s.x.nnz * min(1.0, PCA_CUT_SHARE * k / max(1, s.m))) * 8 \
        if s.x is not None and s.x.sparse else 0
    t = max(cut + dense, int(PCA_DENSE_COPIES * dense)) + s.n * n_comps * PCA_BYTES_PER_COMP
    s.other += s.n * n_comps * 4 + s.m * n_comps * 8
    secs = 2.0 + s.n * k * PCA_SECONDS_PER_VALUE
    keys = _regress_keys(params)
    if keys:  # the fit taken off the matrix, a block of cells at a time
        t += s.n * (len(keys) + 1) * 8 + _scaling.REGRESS_ROWS * k * 8
        secs += s.n * k * (len(keys) + 1) * REGRESS_PCA_SECONDS_PER_VALUE
    return t, s, secs


def _step_neighbors(s: _State, params) -> tuple[int, _State, float]:
    if s.bbknn_graph:  # the BBKNN graph batch correction made is kept: nothing to build
        return 0, s, 0.0
    # The graph, and what building it keeps (pynndescent's index structures): held.
    g = s.n * NEIGHBORS_PER_CELL
    s.other += g
    # Mostly numba compilation: exact neighbours under 4,096 cells, pynndescent above.
    return g, s, (10.0 + 1e-3 * s.n) if s.n <= 4096 else (28.0 + 3e-4 * s.n)


# Leiden's time (core/leiden.py; one thread in its child, so more CPUs do not speed it
# up): the small-data line measured on the calibration runs, or a power law in the cells,
# whichever is larger. The power laws are fits to the same saved graph of the brain data
# clustered on Roihu at resolution 0.5 (research/out-of-core, 2026-10-23), scaled by 1.08,
# the laptop's 253 s against Roihu's 234 s at 88,640 cells:
#   full optimisation (n_iterations -1): 234 / 1,586 / 8,738 s at 88,640 / 221,828 /
#   443,946 cells -> 224 x (n / 88,640)^2.24 (fit within -5..+10 %);
#   fast Leiden (2 passes): 17 / 44 / 138 / 446 s at 88,640 / 221,828 / 443,946 /
#   1,214,593 cells -> 16.0 x (n / 88,640)^1.275 (fit within -10..+17 %).
LEIDEN_SCALE_CELLS = 88_640
LEIDEN_SECONDS = {-1: (242.0, 2.24), 2: (17.3, 1.275)}  # (seconds at LEIDEN_SCALE_CELLS, power)
LEIDEN_SMALL = {-1: (0.5, 4e-4), 2: (0.5, 2e-4)}  # (fixed, per cell): child start, the graph
# Clustering estimated over this (the full optimisation) offers fast Leiden before a run;
# SCVIEW_LEIDEN_OFFER_SECONDS in .env changes it.
LEIDEN_OFFER_SECONDS = 600.0


def leiden_seconds(n_cells: int, n_iterations: int = -1) -> float:
    """Seconds Leiden takes on ``n_cells`` (2 CPUs or more: it uses one)."""
    key = -1 if n_iterations == -1 else 2
    fixed, per_cell = LEIDEN_SMALL[key]
    s0, power = LEIDEN_SECONDS[key]
    n = max(0, int(n_cells))
    return float(max(fixed + per_cell * n, s0 * (n / LEIDEN_SCALE_CELLS) ** power))


def leiden_offer_seconds() -> float:
    """The full optimisation's estimate above which a run's note offers fast Leiden."""
    raw = os.environ.get("SCVIEW_LEIDEN_OFFER_SECONDS", "").strip()
    try:
        return float(raw) if raw else LEIDEN_OFFER_SECONDS
    except ValueError:
        return LEIDEN_OFFER_SECONDS


def _step_clustering(s: _State, params) -> tuple[int, _State, float]:
    # Leiden runs in a child process (core/leiden.py): its interpreter and the graph.
    if _p(params, "clustering_method", "leiden") == "louvain":  # igraph's multilevel, fast
        secs = LEIDEN_SMALL[-1][0] + LEIDEN_SMALL[-1][1] * s.n
    else:
        secs = leiden_seconds(s.n, leiden_iterations(params))
    return 60 * MiB + s.n * 9000, s, secs


def leiden_iterations(params) -> int:
    """-1 or 2: the run's Leiden passes (anything else is the step's error, not an estimate's)."""
    v = _p(params, "leiden_iterations", -1)
    try:
        return 2 if int(float(v)) == 2 else -1
    except (TypeError, ValueError):
        return -1


def _epochs(n: int) -> int:
    return 500 if n <= 10_000 else 200  # umap-learn's default


# UMAP (3D then 2D) on the mouse brain: 88,640 cells +0.47 GiB / 90 s at 2 CPUs; 1.16M
# cells about +10 GB / 2,941 s at 8 CPUs (Roihu e2e, 2026-10-24). Memory per cell grows
# a little with n (5.7 -> ~9 kB, ~n^1.18), time ~n^1.36 and barely with CPUs.
UMAP_BYTES_PER_CELL = 5_700  # at 88,640 cells
UMAP_MEMORY_EXPONENT = 0.18
UMAP_TIME_EXPONENT = 0.36


# The 3D UMAP's share of the step's time (umap-learn lays out 3D and 2D the same way:
# spectral start, then the same epochs over the same graph)
UMAP_3D_SHARE = 0.5


def umap_seconds(n: int, three_d: bool = True) -> float:
    """The UMAP step's time at 2 CPUs: the 2D UMAP, and the 3D one when ``three_d``."""
    size = max(1.0, n / 88_640)
    both = 5e-6 * n * _epochs(n) * size ** UMAP_TIME_EXPONENT
    return 0.5 + (both if three_d else both * (1 - UMAP_3D_SHARE))


def _three_d(params) -> bool:
    """Whether the run makes the 3D UMAP; auto is on here (the large-dataset estimate gets
    params with auto resolved, ``_large_params``)."""
    return str(_p(params, "umap_3d", "auto")).lower() not in ("off", "false")


def _step_embeddings(s: _State, params) -> tuple[int, _State, float]:
    three_d = _three_d(params)
    s.other += s.n * (5 if three_d else 2) * 4
    size = max(1.0, s.n / 88_640)
    t = s.n * UMAP_BYTES_PER_CELL * size ** UMAP_MEMORY_EXPONENT
    return int(t), s, umap_seconds(s.n, three_d)


def _large_params(params):
    """``params`` as large-dataset mode runs them: no 3D UMAP unless asked for."""
    from scview.core.pipeline import resolve_umap_3d

    return _params_with(params, umap_3d=resolve_umap_3d(_p(params, "umap_3d", "auto"), True))


def umap_3d_choice(n_cells: int | None, steps: list[str], params, chunked: bool) -> dict | None:
    """For a run with the UMAP step: whether it makes the 3D UMAP (``on``: auto is on in
    memory, off in large-dataset mode), the setting, and the time the 3D UMAP adds
    (``extra_*``, a point estimate and a range). ``offer`` when the note should say so:
    in large-dataset mode, or when the 3D UMAP alone would take over 10 minutes. None for
    a run without the UMAP step."""
    from scview.core.pipeline import check_umap_3d, resolve_umap_3d

    if "embeddings" not in (steps or []) or not n_cells:
        return None
    n = int(n_cells * FILTER_CELLS_KEPT) if "filtering" in steps else int(n_cells)
    setting = check_umap_3d(_p(params, "umap_3d", "auto"))
    extra = umap_seconds(n, True) - umap_seconds(n, False)
    return {
        "n_cells": n,
        "setting": setting,
        "on": resolve_umap_3d(setting, chunked) == "on",
        "default_on": not chunked,
        "extra_seconds": round(extra, 1),
        "extra_text": time_text(_range(extra, CALIBRATION_CPUS)),
        "offer": bool(chunked or extra > leiden_offer_seconds()),
    }


def _step_uncorrected(s: _State, params) -> tuple[int, _State, float]:
    k = int(_p(params, "n_neighbors", 15))
    s.other += s.n * 2 * 4
    return s.n * k * 48, s, 1.5e-3 * s.n + 1.9e-6 * s.n * _epochs(s.n)


# Batch correction per method (core/batch_methods.py), measured on Kang (13,836 cells,
# ``stim`` as batch, 50 PCs, 2 CPUs; and a random half, 6,918), 2026-10-24:
#   Harmony (harmonypy 2.x) 0.13 / 0.28 s;
#   Scanorama 1.93 / 4.87 s, Python peak 172 / 190 MiB: the Gaussian kernel block
#     (_KERNEL_ROWS x batch size 5,000, float64 = 164 MB) and copies of the PCs; its time
#     grows faster than the cells (every cell against every matched cell);
#   BBKNN 26.7 / 29.0 s, peak 60 / 64 MiB: mostly numba compiling UMAP's connectivities
#     (as the neighbours step's own fixed cost, which BBKNN replaces); then the graph.
SCANORAMA_SCALE_CELLS = 13_836
SCANORAMA_SECONDS = (0.5, 4.4, 1.6)  # fixed + s0 x (n / scale)^power
BBKNN_SECONDS = (26.0, 2.2e-4)  # fixed (compilation), per cell


def _step_batch(s: _State, params) -> tuple[int, _State, float]:
    method = str(_p(params, "batch_method", "harmony") or "harmony").lower()
    n_comps = int(_p(params, "n_comps", 50))
    if method == "scanorama":
        from scview.core.batch_methods import _KERNEL_ROWS, SCANORAMA_BATCH_SIZE

        t = min(s.n, _KERNEL_ROWS) * SCANORAMA_BATCH_SIZE * 8 + s.n * n_comps * 8 * 4
        s.other += s.n * n_comps * 8
        fixed, s0, power = SCANORAMA_SECONDS
        return t, s, fixed + s0 * (s.n / SCANORAMA_SCALE_CELLS) ** power
    if method == "bbknn":
        # The graph replaces the neighbours step's (same size), so it is held from here.
        g = s.n * NEIGHBORS_PER_CELL
        s.other += g
        s.bbknn_graph = True
        fixed, per_cell = BBKNN_SECONDS
        return g + s.n * n_comps * 8 * 2, s, fixed + per_cell * s.n
    # Harmony: cluster x cell matrices (float64, up to 100 clusters) and PC copies.
    clusters = min(100, max(1, s.n // 30))
    t = s.n * clusters * 8 * 3 + s.n * 50 * 8 * 4
    s.other += s.n * 50 * 8
    return t, s, 0.1 + 1.3e-5 * s.n


def _groups(params) -> int:
    """Marker / enrichment groups: the new clustering's and the columns named."""
    cols = len(_p(params, "marker_columns", []) or []) or 1
    return 15 * (cols + 1)


def _step_markers(s: _State, params) -> tuple[int, _State, float]:
    ln = s.lognorm()
    if ln is None:
        return 0, s, 0.1
    from scview.core import matrix_genes

    # One block of genes at a time, and each group's statistics (as the large-dataset
    # estimate below, with the in-memory gather's working set).
    groups = _groups(params)
    t = (matrix_genes.block_entries(ln.nnz) * MARKERS_MEMORY_PER_VALUE
         + groups * ln.n_genes * 8 * 12 + s.n * 16)
    s.other += 56 * ln.n_genes * groups
    return t, s, 1.0 + (groups // 15) * ln.nnz * MARKERS_SECONDS_PER_VALUE


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
    "filtering": _step_filter, "ambient_correction": _step_ambient,
    "normalization": _step_normalization, "log_transform": _step_log,
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


# Steps that use one CPU whatever the quota: their time is not scaled by the CPUs.
SINGLE_THREAD_STEPS = ("clustering",)


def _cpu_free(step: str, secs: float, cpus: float | None) -> float:
    """``secs`` such that ``_range`` (which scales by the CPUs) gives it back unscaled for
    a single-threaded step (Leiden); other steps' unchanged."""
    return secs / _cpu_factor(cpus) if step in SINGLE_THREAD_STEPS else secs


def process_warm() -> bool:
    """True once this process has run the libraries whose first use costs memory."""
    import sys

    return WARMUP_MODULE in sys.modules


def estimate_run(shape: tuple[int, int] | Header, nnz: int | None = None, dense: bool = False,
                 steps: list[str] | None = None, params=None, cpus: float | None = None,
                 warm: bool | None = None, doublets: DoubletPlan | None = None) -> RunEstimate:
    """Memory per step and at the run's peak, and rough seconds, for ``steps`` (any
    order; run in the pipeline's) on a dataset.

    ``shape`` is (cells, genes) of X with ``nnz`` stored values (``dense``: every
    value), or a full ``Header`` (layers and .raw count too: pass one when you have
    it). The run's peak is the largest (held at the step's start + the step's
    transient) over the steps, as the dataset's state changes through them (reset
    to counts replaces X, filtering cuts each matrix once and the opened copy gives
    its matrices up, scaling sets .raw to X itself), then saving and reopening the result. ``warm``: the
    process already paid its one-time imports (default: looked up). ``doublets``: the
    doublet step's channels (``doublet_plan``; None: all cells as one)."""
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
        model = _STEP_MODELS[step]
        t, s, secs = (_step_doublet(s, params, doublets) if model is _step_doublet
                      else model(s, params))
        secs = _cpu_free(step, secs, cpus)
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
                       params=None, *, other_bytes: int = 0, dtype: str = "float32",
                       nonzero: int | None = None, doublets: DoubletPlan | None = None) -> int:
    """Memory (bytes) ``step`` needs at its peak over what the dataset holds when it
    starts. Pure: no I/O, no process state (one-time imports not counted).

    ``shape`` = (cells, genes) of X; ``nnz`` its stored values (ignored for a dense
    X); ``dense`` whether X is a dense array; ``params`` the run's ``PipelineParams``
    (None: the defaults); ``other_bytes`` the bytes of the layers and .raw, which
    filtering copies (0 underestimates it); ``dtype`` of X's values; ``nonzero`` the
    non-zero values of a dense X (None: every value, which overestimates doublet
    detection on dense counts); ``doublets`` its channels (None: all cells as one). Filtering
    is taken to remove cells (one cut of each matrix), also on data filtered before. Unknown
    steps give 0. From a live AnnData use ``step_peak_for(adata, step, params)``, which reads
    all of this from it.
    """
    model = _STEP_MODELS.get(step)
    if model is None:
        return 0
    n, m = shape
    x = Matrix("X", n, m, not dense, n * m if dense else nnz, dtype,
               nonzero=nonzero if dense else None)
    layers = {}
    if other_bytes and n:  # an opaque block of the layers' and .raw's bytes, per cell
        layers["other"] = Matrix("layers/other", n, max(1, other_bytes // n), False, 0, "uint8")
    # PCA reads the variable genes a run picks (n_top_genes).
    hvg = min(m, int(_p(params, "n_top_genes", 2000)))
    st = _State(n, m, x, layers, None, 0, n_hvg=hvg)
    t, _, _ = _step_doublet(st, params, doublets) if model is _step_doublet else model(st, params)
    return int(t)


def step_peak_for(adata, step: str, params=None) -> int:
    """``estimate_step_peak`` for a live AnnData (in memory, not backed): its X,
    layers and .raw as they are (and for doublet detection its channels,
    ``doublet_plan``), e.g. from a hook before each pipeline step."""
    model = _STEP_MODELS.get(step)
    if model is None:
        return 0
    st = _State.from_header(header_from_adata(adata))
    if model is _step_doublet:
        t, _, _ = _step_doublet(st, params, doublet_plan(adata, params))
    else:
        t, _, _ = model(st, params)
    return int(t)


# ---------------------------------------------------------------------------
# Large-dataset (chunked) runs: core/large_run.py
# ---------------------------------------------------------------------------
#
# The matrices stay on disk: the run holds the dataset without them (obs, var, obsm,
# obsp, uns) and what its steps add, and each pass over the matrix holds a batch of row
# blocks (``chunked.step_bytes``: one block's working memory per worker). The one write
# of X, the counts and the gene store holds a batch of blocks and the gene store's staged
# rows and gene block.
#
# Calibration (2026-10-24): the live run of the UI's 15 steps on the 100k-cell brain
# (100,000 x 27,998, 201M stored values, gzip; 88,640 x 19,824 after filtering) in a
# 6 GB / 2-CPU container, auto mode, through /assessment/run-stream: 551 s, peak 2.19 GB
# (Leiden), cgroup memory (current - inactive_file) sampled every 0.5 s. Per step, GB at the
# peak, measured / estimated: QC 0.82 / 0.73 (first estimate, before calibration), write
# + variable genes 1.86 / 1.86, PCA 1.43 / 0.89, neighbours 1.33 / 1.27, Leiden 2.19 / 2.08,
# markers 1.58 / 1.52, cell cycle 1.76 / 1.65. The first estimate had the memory a process
# keeps after its first pass too low (+0.45-0.55 GB: dask's threads, numba, h5py caches,
# freed memory glibc keeps; another 0.4 GB after the write) and the neighbour graph too
# high (the in-memory model's 12 KB per cell; 2.5 KB measured, 1.2 KB of it kept); these
# are set from this run, as is what the server keeps after Leiden (0.2 GB). Calibrated:
# write + variable genes 1.81, PCA 1.62, neighbours 1.21, Leiden 2.04 (the run's peak,
# -7%), UMAP 1.33 (1.58), markers 1.57, cell cycle 1.69; QC 1.07, filtering 1.16 (over).
# Seconds: a pass 3.5e-8 per stored value (QC 6.2 s, filtering 5.3 s, scaling 5.4 s), the
# write 4e-7 (71 s), Leiden 253 s at 88,640 cells (``leiden_seconds``: about n^2.2, or
# n^1.25 with fast Leiden's two passes). Disk: X 520 MB, counts 418 MB,
# gene store 710 MB, and the store's build spills 8 bytes per value while it runs.

CHUNKED_FIXED = 500 * MiB  # kept by the process from its first pass on (see above)
# The gene store's build (core/gene_store.py): one staged chunk of rows (STAGE_ENTRIES
# values, cleaned, stacked and transposed) or one gene block (BLOCK_ENTRIES), 8 bytes each
# a few times over; 0.80 GB peak RSS building 100k cells on its own.
GENESTORE_BUILD_BYTES = 750 * MiB
CHUNKED_WRITE_KEPT = 400 * MiB  # kept by the process after the write (measured, as above)
MARKERS_STORE_PER_VALUE = 60  # sparse_markers: working set per stored value of a gene block
CELLTYPIST_LABELS = 100  # probabilities kept per cell (float32)
CHUNKED_NEIGHBORS_PER_CELL = 2_600  # building the graph; 1,300 of it kept
CHUNKED_GRAPH_PER_CELL = 1_300
CHUNKED_LEIDEN_PER_CELL = 10_500  # the Leiden child: its graph copy and partition
CHUNKED_LEIDEN_KEPT_PER_CELL = 2_500  # kept by the server after it (the edges it sent)
# Seconds per stored value: a pass that reads the matrix (gzip CSR, 2 CPUs), and the one
# write (X and counts gzip, the gene store built and filled); markers per column.
CHUNKED_PASS_SECONDS_PER_VALUE = 3.5e-8
CHUNKED_WRITE_SECONDS_PER_VALUE = 4e-7
CHUNKED_MARKERS_SECONDS_PER_VALUE = 7e-8
# Disk per stored value: X (float32 + int64 index, gzip), the counts (gzip), the gene store
# (gzip-4 + shuffle) and its build's spill files (8 bytes, removed after).
DISK_X_PER_VALUE = 2.8
DISK_COUNTS_PER_VALUE = 2.5
DISK_STORE_PER_VALUE = 4.0
DISK_STORE_TEMP_PER_VALUE = 8.0
# Steps after which the large run writes X, the counts and the gene store again (any
# step on a dataset not yet in large-dataset layout writes them too).
CHUNKED_WRITES = ("reset_to_counts", "filtering", "normalization", "log_transform")
# Steps whose first need of X makes the run write it (else the save does).
_CHUNKED_READS_X = ("highly_variable_genes", "scaling", "pca", "marker_genes", "cell_cycle",
                    "dissociation_stress", "cell_type_annotation")


@dataclass
class ChunkedShape:
    """What a large-dataset run works on: cells, genes, the counts' stored values (every
    value of a dense matrix), and the bytes of the dataset without its matrices."""

    n: int
    m: int
    nnz: int
    dense: bool = False
    other: int = 0
    n_groups: int | None = None  # the active clustering's groups (markers), when known

    @classmethod
    def from_header(cls, h: Header) -> "ChunkedShape":
        mats = h.matrices
        nnz = max((mm.nnz for mm in mats), default=0)
        dense = any(not mm.sparse for mm in mats)
        n = h.n_cells or 0
        other = (h.obsm_bytes + h.obsp_bytes + h.varm_bytes + h.uns_bytes + _frames_bytes(h)
                 + n * LOAD_PER_CELL + LOAD_FIXED)
        return cls(n, h.n_genes or 0, int(nnz), dense, int(other))


def _chunk_pass(sh: ChunkedShape, params, factor: float = 1.0) -> int:
    from scview.core import chunked

    hvg = min(sh.m, int(_p(params, "n_top_genes", 2000)))
    per_row = sh.nnz / max(1, sh.n)
    rows = chunked.rows_per_block(sh.m, nnz_per_row=per_row, dense=sh.dense, n_hvg=hvg)
    return int(factor * chunked.step_bytes(rows, sh.m, nnz_per_row=per_row, dense=sh.dense,
                                           n_hvg=hvg))


def _pass_secs(sh: ChunkedShape, passes: float = 1.0) -> float:
    return 0.5 + passes * sh.nnz * CHUNKED_PASS_SECONDS_PER_VALUE


def chunked_step(step: str, sh: ChunkedShape, params, *, write: bool = False,
                 doublets: DoubletPlan | None = None) -> tuple[int, int, float]:
    """(transient bytes over what the run holds, bytes the step adds to it, seconds at 2
    CPUs) of one step in large-dataset mode; ``write``: the step first writes X, the
    counts and the gene store."""
    n, m = sh.n, sh.m
    held = 0
    if step in ("reset_to_counts",):
        t, secs = int(0.5 * sh.other), 0.5
    elif step == "qc_metrics":
        t, secs, held = _chunk_pass(sh, params), _pass_secs(sh), n * 5 * 8 + m * 5 * 8
    elif step == "doublet_detection":
        x = Matrix("X", n, m, not sh.dense, sh.nnz if not sh.dense else n * m, "float32",
                   nonzero=sh.nnz if sh.dense else None)
        t, _, secs = _step_doublet(_State(n, m, x, {}, None, 0), params, doublets)
        held = n * 16
    elif step == "filtering":
        t, secs = _chunk_pass(sh, params) + sh.other, _pass_secs(sh)
    elif step == "normalization":
        target = _p(params, "target_sum", 1e4)
        t, secs = (_chunk_pass(sh, params), _pass_secs(sh)) if target is None else (0, 0.1)
    elif step == "log_transform":
        t, secs = 0, 0.1
    elif step == "highly_variable_genes":
        two = (_p(params, "hvg_flavor", "seurat") == "seurat_v3"
               or _p(params, "normalization_method", "log") == "pearson_residuals")
        t, secs = _chunk_pass(sh, params), _pass_secs(sh, 2 if two else 1)
    elif step == "scaling":
        t, secs = _chunk_pass(sh, params), _pass_secs(sh)
        keys = _regress_keys(params)
        if keys:  # two passes (the fit, the residuals' statistics) for the plain one
            t += n * (len(keys) + 1) * 8
            secs = _pass_secs(sh, 2)
            if set(keys) & set(_scaling.CELL_CYCLE_COVARIATES):
                t2, held, secs2 = chunked_step("cell_cycle", sh, params)
                t, secs = max(t, t2), secs + secs2
    elif step == "pca":
        k = min(m, int(_p(params, "n_top_genes", 2000)))
        c = int(_p(params, "n_comps", 50))
        t = _chunk_pass(sh, params) + k * k * 8 * 4 + n * c * 4 * 2
        secs, held = _pass_secs(sh, 2) + 2e-9 * n * k * k / 50, n * c * 4 + m * c * 8
        keys = _regress_keys(params)
        if keys:
            t += n * (len(keys) + 1) * 8
            secs += 2 * n * k * (len(keys) + 1) * REGRESS_PCA_SECONDS_PER_VALUE
    elif step == "neighbors":
        t, held = n * CHUNKED_NEIGHBORS_PER_CELL, n * CHUNKED_GRAPH_PER_CELL
        secs = _STEP_MODELS[step](_State(n, m, None, {}, None, 0), params)[2]
    elif step == "clustering":  # the in-memory step's own time (leiden_seconds)
        t, held = 60 * MiB + n * CHUNKED_LEIDEN_PER_CELL, n * CHUNKED_LEIDEN_KEPT_PER_CELL
        secs = _STEP_MODELS[step](_State(n, m, None, {}, None, 0), params)[2]
    elif step in ("embeddings", "uncorrected_embedding", "batch_correction", "enrichment"):
        s = _State(n, m, None, {}, None, 0)
        before = s.other
        t, s, secs = _STEP_MODELS[step](s, params)
        held = s.other - before
    elif step == "marker_genes":
        from scview.core import sparse_markers

        g = sh.n_groups or 15
        cols = len(_p(params, "marker_columns", []) or []) or 1
        t = sparse_markers.MAX_ENTRIES * MARKERS_STORE_PER_VALUE + g * m * 8 * 12 + n * 16
        held = (56 + 16) * m * g * cols
        secs = 1.0 + cols * sh.nnz * CHUNKED_MARKERS_SECONDS_PER_VALUE
    elif step in ("cell_cycle", "dissociation_stress"):
        from scview.core import gene_store

        t = max(_chunk_pass(sh, params, 0.5), gene_store.BLOCK_ENTRIES * 12)
        secs, held = _pass_secs(sh, 1.5), n * 4 * 8
    elif step == "cell_type_annotation":
        from scview.core import large_scores

        if _p(params, "annotation_method", "llm") == "celltypist":
            t = _chunk_pass(sh, params) + large_scores.CELLTYPIST_BLOCK_BYTES
            held = n * CELLTYPIST_LABELS * 4 + n * 3 * 8
            secs = 5.0 + _pass_secs(sh) + 2e-3 * n
        else:
            t, secs, held = 100 * MiB, 10.0 + 2e-4 * n, n * 8
    elif step == "save":
        t, secs = int(0.3 * sh.other) + 64 * MiB, 1.0 + sh.other / WRITE_BPS
    else:
        t, secs = 0, 0.1
    if write:
        t = max(t, _chunk_pass(sh, params) + GENESTORE_BUILD_BYTES)
        held += CHUNKED_WRITE_KEPT
        secs += sh.nnz * CHUNKED_WRITE_SECONDS_PER_VALUE
    return int(t), int(held), float(secs)


def _writes_at(steps: list[str], rewrite: bool) -> str | None:
    """The step at which the run writes X, the counts and the gene store (None: none)."""
    if not rewrite:
        return None
    first = next((s for s in steps if s in _CHUNKED_READS_X), None)
    return first or "save"


def estimate_chunked_run(sh: ChunkedShape | Header, steps: list[str], params=None, *,
                         large: bool = False, cpus: float | None = None,
                         doublets: DoubletPlan | None = None) -> RunEstimate:
    """Memory per step and at the peak, and rough seconds, of ``steps`` in large-dataset
    mode. ``large``: the dataset is in that layout already (its matrices are written
    again only after a step that changes them, ``CHUNKED_WRITES``)."""
    from scview.core.pipeline import ALL_STEPS

    if isinstance(sh, Header):
        sh = ChunkedShape.from_header(sh)
    ordered = [s for s in ALL_STEPS if s in set(steps or [])]
    rewrite = not large or any(s in CHUNKED_WRITES for s in ordered)
    write_at = _writes_at(ordered, rewrite)
    resident = sh.other
    start = resident
    out: list[StepEstimate] = []
    total = 0.0
    cur = sh
    fixed = 0
    for step in ordered + ["save"]:
        t, held, secs = chunked_step(step, cur, params, write=step == write_at, doublets=doublets)
        secs = _cpu_free(step, secs, cpus)
        if step not in ("reset_to_counts", "neighbors", "clustering", "embeddings",
                        "uncorrected_embedding", "batch_correction", "enrichment", "save"):
            fixed = CHUNKED_FIXED
        total += secs
        out.append(StepEstimate(step, resident + fixed, t, _range(secs, cpus),
                                step in CHUNKED_MEASURED))
        resident += held
        if step == "filtering":  # as the in-memory model takes it
            n2 = int(cur.n * FILTER_CELLS_KEPT)
            m2 = cur.m if large else int(cur.m * FILTER_GENES_KEPT_RAW)
            cur = ChunkedShape(n2, m2, int(cur.nnz * FILTER_CELLS_KEPT), cur.dense,
                               int(cur.other * FILTER_CELLS_KEPT), cur.n_groups)
    peak = max(out, key=lambda e: e.peak_bytes)
    return RunEstimate(out, start, peak.peak_bytes, peak.step, resident + fixed,
                       _range(total, cpus))


# Steps of large-dataset mode whose coefficients come from a measured run (the live
# 100k-cell run; see CHUNKED_CALIBRATION).
CHUNKED_MEASURED: set[str] = {
    "reset_to_counts", "qc_metrics", "doublet_detection", "filtering", "normalization",
    "log_transform", "highly_variable_genes", "scaling", "pca", "neighbors", "clustering",
    "embeddings", "marker_genes", "cell_cycle", "dissociation_stress", "save",
}


def chunked_step_peak(step: str, sh: ChunkedShape, params=None, *, write: bool = False,
                      doublets: DoubletPlan | None = None) -> int:
    """``estimate_step_peak`` for large-dataset mode: memory ``step`` needs at its peak
    over what the run holds (the pre-step check, core/large_run.py)."""
    t, _, _ = chunked_step(step, sh, params, write=write, doublets=doublets)
    return int(t)


def chunked_disk_bytes(sh: ChunkedShape, *, rewrite: bool = True, x_file_bytes: int = 0) -> int:
    """Disk a large-dataset run writes before it can swap the result in: X, the counts, the
    gene store and its build's temporary files (``rewrite``), else a copy of X
    (``x_file_bytes``), and the rest of the dataset."""
    if not rewrite:
        return int(x_file_bytes + sh.other)
    per_value = (DISK_X_PER_VALUE + DISK_COUNTS_PER_VALUE + DISK_STORE_PER_VALUE
                 + DISK_STORE_TEMP_PER_VALUE)
    return int(sh.nnz * per_value + sh.other)


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
                 "zarr": "zarr", "dense_table": "table", "seurat": "rds",
                 "h5seurat": "h5seurat"}


def read_unit_header(unit, genes_in_rows: bool = True) -> Header:
    """The header of one ingest unit (a MEX unit's from its matrix.mtx)."""
    fmt = _UNIT_FORMATS.get(unit.format.value, None)
    if fmt is None or not unit.files:
        return Header("unknown", note="The file is not a format scView reads.")
    f = next((x for x in unit.files if x.role.value == "matrix"), None) if fmt == "mex" else unit.files[0]
    if f is None:
        return Header("unknown", note="The matrix file of this 10x sample is missing.")
    h = read_header(f.path, fmt, genes_in_rows)
    spatial_parts = [(x.spatial_part or "", x.path) for x in getattr(unit, "side_files", [])
                     if getattr(x, "spatial_part", None)]
    if spatial_parts:  # Space Ranger's tissue images (uint8 RGB in uns) and the coordinates
        from scview.core import spatial

        h.uns_bytes += spatial.side_bytes(spatial_parts)
        h.obsm_bytes += 16 * int(h.n_cells or 0)
    return h


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

    from scview.core.ingestion import streamed

    bundle = build_bundle(mgr.staged_paths(sid))
    opts = mgr.get_options(sid)
    avail = available_memory()
    cached = cached_others(dm)
    loadable = [u for u in bundle.units if u.format != UnitFormat.unknown and u.complete]
    # Too large to read whole: the commit reads and saves the matrices in blocks.
    try:
        choice = streamed.choose(loadable, opts.streamed, opts.genes_in_rows) if loadable else None
    except streamed.StreamUnsupported:
        choice = None
    blocks = bool(choice and choice.streamed)
    units, residents, peaks, secs = [], [], [], [0.0, 0.0]
    for u in loadable:
        h = read_unit_header(u, opts.genes_in_rows)
        h.backed = False  # the ingest reads an .h5ad whole, whatever its size
        est = estimate_load(h)
        half = estimate_load(_half_cells(h)) if not h.guess else None
        if blocks:
            need = streamed.streamed_need([u], opts.genes_in_rows)
            v = make_verdict(
                need=need, peak=need, stage="reading and saving it in blocks", seconds=est.seconds,
                avail=avail, cached=cached,
                lead=(_load_lead(h, est).replace("Loading takes", "Read whole, loading would take")
                      + f"; it is read and saved in blocks instead, {_about(need)} at its peak"),
                breakdown=[("Cell and gene tables, one block", need)],
                notes=est.notes + [streamed.BLOCKS_NOTE], alternatives=[])
        else:
            v = make_verdict(
                need=est.peak_bytes, peak=est.peak_bytes, stage=est.peak_stage, seconds=est.seconds,
                avail=avail, cached=cached, lead=_load_lead(h, est), breakdown=est.breakdown,
                notes=est.notes,
                alternatives=[("Subsample the cells before loading: half of them", half.peak_bytes)]
                if half else [])
        units.append({"label": u.label, "format": u.format.value, "header": h.to_dict(),
                      "load": {"resident_bytes": est.resident_bytes, "peak_bytes": est.peak_bytes,
                               "peak_stage": est.peak_stage, "seconds": list(est.seconds)},
                      "streamed": blocks, "verdict": v})
        residents.append(est.resident_bytes)
        peaks.append(est.peak_bytes)
        secs = [secs[0] + est.seconds[0], secs[1] + est.seconds[1]]
    merge = None
    if len(units) > 1 and blocks:
        need = streamed.streamed_need(loadable, opts.genes_in_rows)
        merge = make_verdict(
            need=need, peak=need, stage="merging the samples in blocks", seconds=tuple(secs),
            avail=avail, cached=cached,
            lead=(f"Merging the {len(units)} samples in blocks takes {_about(need)} at its peak "
                  f"(read whole, {_about(sum(residents) * 2)})"),
            breakdown=[("Cell and gene tables, one block", need)],
            notes=[streamed.BLOCKS_NOTE], alternatives=[])
    elif len(units) > 1:
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


def in_memory_header(adata) -> Header:
    """The header of a live AnnData as a run sees it: in memory as it is, or, opened
    backed, from its file as read in (``core/working_copy.py`` reads it in before the
    run; ``header_from_adata`` would take its on-disk X for a dense one)."""
    if getattr(adata, "isbacked", False):
        h = read_header(adata.filename, "h5ad")
        h.backed = False
        return h
    return header_from_adata(adata)


def run_need(adata, steps: list[str], params=None) -> int:
    """Memory a run of ``steps`` needs beyond what the dataset holds in memory (a backed
    one: once read in, which the run makes room for separately)."""
    plan = doublet_plan(adata, params) if "doublet_detection" in steps else None
    est = estimate_run(in_memory_header(adata), steps=steps, params=params, warm=process_warm(),
                       doublets=plan)
    return max(0, int(est.peak_bytes - est.start_bytes))


def _file_doublet_plan(path, params=None) -> DoubletPlan | None:
    """The doublet step's channels from a file's obs alone (its matrix is not read, so
    Scrublet's variable genes are not counted). None when obs can't be read."""
    try:
        import anndata as ad
        import h5py

        from scview.core import doublets

        with h5py.File(path, "r") as f:
            obs = ad.io.read_elem(f["obs"])
        ch = doublets.channels(ad.AnnData(obs=obs), _p(params, "doublet_channel_key", "") or None)
        return DoubletPlan([int(k) for k in ch.sizes])
    except Exception:  # noqa: BLE001 — an estimate never fails a request: one channel then
        return None


def dataset_preflight(dm, dataset_id: str, steps: list[str], overrides: dict | None = None) -> dict:
    """The run estimate for ``steps`` on a dataset, with its verdict. The shape comes
    from the cached dataset, else from the header of its file (never loaded here).
    ``mode`` says whether the run would go to large-dataset mode (``params.chunked``:
    auto by the rule ``large_run.choose_mode`` applies, or on / off); in that mode the
    steps and verdict are that mode's (``estimate_chunked_run``, with the disk it writes)
    and ``in_memory`` holds the in-memory estimate it replaced."""
    from scview.core import large_layout
    from scview.core.pipeline import PipelineParams, check_chunked

    params = PipelineParams()
    for k, v in (overrides or {}).items():
        if hasattr(params, k):
            setattr(params, k, v)
    requested = check_chunked(params.chunked)
    adaptor = dm.get_dataset(dataset_id)
    in_memory = adaptor is not None
    h = None
    plan = None
    dbl = "doublet_detection" in steps
    files = dm.h5ad_candidates(dataset_id)
    # Not while a run changes it in a worker thread: its file says what it was.
    if in_memory and not dm.is_busy(dataset_id):
        a = adaptor.adata
        if not getattr(a, "isbacked", False):
            try:
                h = header_from_adata(a)
                plan = doublet_plan(a, params) if dbl else None
            except (RuntimeError, ValueError):  # changed under us
                h = None
    if h is None:
        if not files:
            raise FileNotFoundError(dataset_id)
        h = read_header(files[0], "h5ad")
        plan = _file_doublet_plan(files[0], params) if dbl else None
    large = h.large or (in_memory and large_layout.is_large(adaptor.adata))
    warm = process_warm()
    est = estimate_run(h, steps=steps, params=params, warm=warm, doublets=plan)
    # A backed dataset holds less than the run's start (its matrices are read in first).
    held = (resident_bytes(h) if h.backed else est.start_bytes) if in_memory else 0
    need = est.peak_bytes - held
    avail = available_memory()
    cached = cached_others(dm, dataset_id)

    alternatives: list[tuple[str, int]] = []
    if dbl:
        alt = estimate_run(h, steps=[s for s in steps if s != "doublet_detection"],
                           params=params, warm=warm)
        alternatives.append(("Skip doublet detection", alt.peak_bytes - held))
    if set(steps) & {"highly_variable_genes", "pca"} and params.n_top_genes > 1000:
        alt = estimate_run(h, steps=steps, params=_params_with(params, n_top_genes=1000), warm=warm,
                           doublets=plan)
        alternatives.append((f"Use 1,000 variable genes instead of {params.n_top_genes:,}",
                             alt.peak_bytes - held))
    half = None if plan is None else DoubletPlan([c // 2 for c in plan.channels], plan.hvg)
    alt = estimate_run(_half_cells(h), steps=steps, params=params, warm=warm, doublets=half)
    alternatives.append(("Subsample the cells: half of them", alt.peak_bytes - held))

    stage = STEP_LABELS.get(est.peak_step, est.peak_step)
    if need > 0:
        about = _about(need)
        lead = f"{about[0].upper()}{about[1:]} more than now at its peak ({stage})"
    else:
        lead = f"No more memory than this dataset holds now at its peak ({stage})"
    breakdown = [("This dataset now" if in_memory else "This dataset once loaded",
                  held if in_memory else est.start_bytes),
                 (f"At the peak ({stage})", est.peak_bytes),
                 ("After the run (saved and reopened)", est.end_bytes)]
    notes = []
    unmeasured = [STEP_LABELS[e.step] for e in est.steps if not e.measured]
    if unmeasured:
        notes.append("Not measured, estimated from what the step allocates: "
                     + ", ".join(unmeasured) + ".")
    if "filtering" in steps:
        notes.append("Assumes filtering keeps 90% of the cells and every gene, as the data were "
                     "processed before." if _State.from_header(h).processed else
                     "Assumes filtering keeps 90% of the cells and 80% of the genes.")
    if not warm:
        notes.append("Includes about 0.5 GB the first run in a fresh scView keeps for the "
                     "libraries it loads.")
    v = make_verdict(need=need, peak=est.peak_bytes, stage=stage, seconds=est.seconds,
                     avail=avail, cached=cached, lead=lead, breakdown=breakdown, notes=notes,
                     alternatives=alternatives)
    out = {"dataset_id": dataset_id, "in_memory": in_memory, "header": h.to_dict(),
           "steps": [e.to_dict() for e in est.steps], "verdict": v,
           "leiden": leiden_choice(h.n_cells, steps, params)}
    chunked, reason = _would_chunk(requested, need, avail, cached, large)
    out["mode"] = {"chunked": chunked, "requested": requested, "reason": reason}
    out["umap_3d"] = umap_3d_choice(h.n_cells, steps, params, chunked)
    if chunked:
        out["in_memory_estimate"] = {"steps": out["steps"], "verdict": v}
        out.update(_chunked_preflight(h, steps, _large_params(params), held, avail, cached, plan,
                                      large=large,
                                      disk_folder=Path(dm.data_dir)))
    return out


def leiden_choice(n_cells: int | None, steps: list[str], params=None) -> dict | None:
    """For a run that clusters with Leiden: its time with the full optimisation and with
    fast Leiden (point estimates, one CPU's worth; ``*_text`` as ranges), the setting the
    run has, and ``offer_fast`` when the full optimisation's estimate is over
    ``leiden_offer_seconds()`` (10 min): the pre-run note then offers fast Leiden. None
    for a run without Leiden clustering. The cells are the dataset's, less filtering's
    share when the run filters (``FILTER_CELLS_KEPT``)."""
    if "clustering" not in (steps or []) or not n_cells:
        return None
    if _p(params, "clustering_method", "leiden") != "leiden":
        return None
    n = int(n_cells * FILTER_CELLS_KEPT) if "filtering" in steps else int(n_cells)
    full, fast = leiden_seconds(n, -1), leiden_seconds(n, 2)
    limit = leiden_offer_seconds()
    return {
        "n_cells": n,
        "full_seconds": round(full, 1), "fast_seconds": round(fast, 1),
        "full_text": time_text(_range(full, CALIBRATION_CPUS)),
        "fast_text": time_text(_range(fast, CALIBRATION_CPUS)),
        "offer_seconds": limit,
        "offer_fast": bool(full > limit),
        "iterations": leiden_iterations(params),
    }


def _would_chunk(requested: str, need: int, avail: Available, cached: Cached,
                 large: bool = False) -> tuple[bool, str]:
    """Large-dataset mode or not, by the rule ``large_run.choose_mode`` applies."""
    from scview.core.dataset_manager import LOAD_MARGIN, working_reserve

    if large:
        return True, ("This dataset is in large-dataset layout, so it runs in large-dataset "
                      "mode: it is never read into memory.")
    if requested == "on":
        return True, "Large-dataset mode was chosen for this run."
    if requested == "off":
        return False, "In memory was chosen for this run."
    if avail.limit_bytes is None or avail.available_bytes is None:
        return False, "No memory limit is known: the run works in memory."
    free = avail.available_bytes + cached.bytes
    target = max(0, need) * LOAD_MARGIN + working_reserve(avail.limit_bytes)
    if target <= free:
        return False, "The run fits in memory."
    return True, (f"In memory the run needs {_about(max(0, need))} more at its peak, more "
                  f"than scView can give it with its working reserve, so it runs in "
                  "large-dataset mode: the matrices stay on disk, read in blocks.")


def _chunked_preflight(h: Header, steps: list[str], params, held: int, avail: Available,
                       cached: Cached, plan, *, large: bool, disk_folder: Path) -> dict:
    """The large-dataset estimate, its verdict (memory and disk) and the in-memory steps."""
    import shutil

    sh = ChunkedShape.from_header(h)
    est = estimate_chunked_run(sh, steps, params, large=large, doublets=plan)
    need = est.peak_bytes - held
    stage = STEP_LABELS.get(est.peak_step, est.peak_step)
    rewrite = not large or any(s in steps for s in CHUNKED_WRITES)
    x_bytes = (h.x.nbytes // 2) if (not rewrite and h.x is not None) else 0  # its gzip copy
    disk = chunked_disk_bytes(sh, rewrite=rewrite, x_file_bytes=x_bytes)
    try:
        free = shutil.disk_usage(disk_folder).free
    except OSError:
        free = None
    about = _about(max(0, need))
    lead = (f"In large-dataset mode (the matrices stay on disk, read in blocks): "
            f"{about} more than now at its peak ({stage}), and {_about(disk)} of disk")
    notes = ["Large-dataset mode writes the log-normalised matrix, the counts and a gene-major "
             "copy beside the dataset; the steps' results are the in-memory ones."]
    unmeasured = [STEP_LABELS.get(e.step, e.step) for e in est.steps if not e.measured]
    if unmeasured:
        notes.append("Not measured, estimated from what the step allocates: "
                     + ", ".join(unmeasured) + ".")
    v = make_verdict(need=need, peak=est.peak_bytes, stage=stage, seconds=est.seconds,
                     avail=avail, cached=cached, lead=lead,
                     breakdown=[("This dataset without its matrices", est.start_bytes),
                                (f"At the peak ({stage})", est.peak_bytes)],
                     notes=notes, alternatives=[])
    if free is not None and disk > free:
        v["level"] = "wont_fit"
        v["sentence"] += (f" The data folder has {_about(free)} of disk free: not enough, so "
                          "the run would stop before it starts.")
        v["suggestions"].insert(0, "Free some disk: delete datasets you no longer need.")
    return {"steps": [e.to_dict() for e in est.steps], "verdict": v,
            "disk": {"need_bytes": int(disk), "free_bytes": free}}


# ---------------------------------------------------------------------------
# Reference mapping (core/label_transfer.py): both datasets' values in memory
# ---------------------------------------------------------------------------

# Measured on Kang split by donor (reference 6.5k cells, query 7.3k, both in memory;
# VmHWM over the run): see label_transfer_need.
LT_COUNTS_COPIES = 2.5  # per byte of each dataset's counts: the recovered CSR, its log copy, the shared-gene cut
LT_NOVELTY_GENES = 1.5  # genes variable in either dataset, per variable gene of the reference
LT_DENSE_COPIES = 3.0  # per cell x gene of the joint matrix: scaled values, PCA's centred copy, its output
LT_CT_PER_VALUE = 12  # CellTypist: the reference densified (float32) and scaled (float64)
LT_FIXED = 100 * MiB


def _lt_counts_bytes(h: Header) -> int:
    """The matrix the run reads counts from (layers['counts'], else .raw, else X) as a
    float32 CSR of its non-zeros."""
    for where in ("layers/counts", "raw/X", "X"):
        mm = h.matrix(where)
        if mm is not None:
            nz = mm.nnz if mm.sparse else (mm.nonzero if mm.nonzero is not None else mm.nnz)
            return int(nz * 8 + (mm.n_cells + 1) * 4)
    return 0


def label_transfer_need(ref: Header, query: Header, *, method: str, ref_open: bool,
                        n_hvg: int = 2000) -> int:
    """Peak memory of mapping ``query`` onto ``ref`` beyond what the open datasets hold:
    the reference read from its file (unless it is open), each dataset's counts copied
    and log-normalised, the joint dense matrix of the variable genes, and CellTypist's
    dense reference when that method runs."""
    n_ref, n_query = ref.n_cells or 0, query.n_cells or 0
    need = 0 if ref_open else estimate_load(ref, ingest=False).peak_bytes
    need += LT_COUNTS_COPIES * (_lt_counts_bytes(ref) + _lt_counts_bytes(query))
    need += LT_DENSE_COPIES * (n_ref + n_query) * n_hvg * LT_NOVELTY_GENES * 4
    if method == "celltypist":
        need += LT_CT_PER_VALUE * n_ref * min(ref.n_genes or 0, query.n_genes or 0)
    return int(need + LT_FIXED)


# ---------------------------------------------------------------------------
# Copy-number inference (core/cnv.py): the cells x windows result, one dense block
# ---------------------------------------------------------------------------

CNV_BLOCK_BYTES = 14  # per cell x gene of a block: log values, the difference, masks, a float64 copy
# Measured on maynard2020_3k (3,000 cells, 23,182 genes, 2,099 windows): 411 MiB above
# the open dataset at the peak; estimated 449 MiB.
CNV_CLUSTER_BYTES = 10  # per cell x window: the sparse float64 copy PCA reads (about half non-zero)
CNV_FIXED = 100 * MiB


def cnv_need(n_cells: int, n_genes: int, n_windows: int) -> int:
    """Peak memory of a copy-number run beyond the open dataset: its result (cells x
    windows, float32) plus the larger of one block of cells (dense, every positioned
    gene) and the copy the CNV clustering reads."""
    from scview.core.cnv import CHUNK, N_PCS

    result = n_cells * n_windows * 4
    block = min(CHUNK, n_cells) * n_genes * CNV_BLOCK_BYTES
    clustering = n_cells * n_windows * CNV_CLUSTER_BYTES + n_cells * N_PCS * 8 * 3
    return int(result + max(block, clustering) + CNV_FIXED)


def memory_level(need: int, avail: int | None) -> str:
    """fits / tight / wont_fit / unknown, as the run preflight words it."""
    return _level(need, avail)
