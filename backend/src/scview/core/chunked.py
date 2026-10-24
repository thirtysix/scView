"""Chunked (out-of-core) QC..PCA: the pipeline's steps on a counts matrix that stays on disk.

For the large-dataset mode (research/out-of-core/DESIGN-large-datasets.md). The counts are
read lazily as a dask array of row blocks (scipy CSR, or numpy for a dense X on disk), by
``anndata.experimental.read_elem_lazy``. Each function here gives what scView's in-memory
step gives on the same cells and genes:

- ``qc_metrics``: the obs and var columns ``pipeline._run_qc_metrics`` writes. Per-cell
  values are the same scipy/numpy operations on the same rows, so they match bitwise; the
  per-gene sums are taken in float64 (exact for counts), equal to scanpy's float32 totals
  while those are exact (a gene's total below 2**24).
- ``gene_detection`` / ``subset``: the filtering step's per-gene cell counts and lazy row
  and column cuts (blocks with no cells kept are dropped).
- ``normalize_total`` / ``log1p``: scanpy's own functions on each row block with the
  target sum fixed, so the values are bitwise the in-memory ones.
- ``highly_variable_genes`` / ``pearson_hvg``: per-gene sums by block, fed to scanpy's own
  binning and selection code (seurat, cell_ranger, seurat_v3, Pearson residuals; with or
  without a batch key), so the same genes are chosen.
- ``pca_scaled`` / ``pca_pearson``: PCA of the variable genes. The scaled (or residual)
  block exists only inside a task; its covariance (genes x genes) is summed over blocks,
  decomposed with ``eigh`` and the cells projected block by block. The result is laid out
  as ``sc.tl.pca`` stores it, signs as scikit-learn's ``svd_flip`` sets them.
- ``write_matrix`` / ``write_h5ad``: a dask matrix streamed into an h5ad or h5 group, beside
  obs, var, obsm, ... written as anndata writes them.

Why not scanpy's own Dask support: its ``calculate_qc_metrics`` was not chunk-bounded
(+22.7 GB at 500k cells on Roihu), and its Dask column sums of a float32 CSR accumulate in
float32, so its variable genes need not equal the in-memory ones.

Work is scheduled in one place (``scheduler``): dask's threads, as many as the container's
CPU quota, with BLAS and numba at one thread inside each task. Block results are taken a
batch of ``workers`` blocks at a time and combined in block order, so memory is about one
block per worker (``block_bytes``) and the result does not depend on the worker count.
Between batches the hook set by ``on_blocks`` is called (the pipeline's large-dataset run,
``core/large_run.py``: progress events, and a cancelled run stops there).
"""

from __future__ import annotations

import contextlib
import contextvars
import functools
import os
import warnings
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

import anndata as ad
import dask
import dask.array as da
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

from scview.core import normalization
from scview.core.preflight import cpu_quota

# Working memory one task may use for its block (the block, its normalised and log copies,
# the dense variable-gene block in float32 and float64). Rows per block come from this.
CHUNK_BYTES = 256 * 2**20
MIN_ROWS, MAX_ROWS = 256, 50_000
_COUNTS_ROWS = 2_000  # rows read to judge whether a matrix holds counts (as looks_like_counts)
_GENE_BLOCK = normalization._CHUNK  # genes per dense residual block (Pearson residuals)
_I32 = np.iinfo(np.int32).max

# scanpy warns once per call when its numba kernels run serially in a worker thread, and
# normalize_total warns about cells without counts in every block that has one.
_QUIET = ("Detected unsupported threading environment", "Some cells have zero counts")


# ---------------------------------------------------------------------------
# Scheduling
# ---------------------------------------------------------------------------

# Blocks in flight may hold at most this share of the memory limit, and no more workers
# than this run at once: each worker holds a block (~CHUNK_BYTES), and a container sees
# every CPU of the machine (16 workers = 4 GB per pass, the whole of a 4 GB limit).
WORKERS_MEMORY_SHARE = 0.25
MAX_WORKERS = 8


def _memory_limit() -> int | None:
    from scview.core.memory_budget import read_budget

    return read_budget().limit_bytes


def n_workers() -> int:
    """Dask workers: the container's CPU quota (``preflight.cpu_quota``), at most
    ``MAX_WORKERS`` and as many blocks as ``WORKERS_MEMORY_SHARE`` of the memory limit
    holds; at least one."""
    w = min(int(cpu_quota()), MAX_WORKERS)
    limit = _memory_limit()
    if limit:
        w = min(w, int(limit * WORKERS_MEMORY_SHARE // CHUNK_BYTES))
    return max(1, w)


@contextlib.contextmanager
def scheduler(workers: int | None = None) -> Iterator[int]:
    """The one place chunked work is scheduled: dask's thread pool with ``workers`` threads
    (default ``n_workers()``), BLAS at one thread meanwhile (the workers are the
    parallelism). Numba is held at one thread inside each task (``_one_thread``)."""
    from threadpoolctl import threadpool_limits

    w = int(workers or n_workers())
    with dask.config.set(scheduler="threads", num_workers=w), \
            threadpool_limits(limits=1, user_api="blas"), warnings.catch_warnings():
        for msg in _QUIET:
            warnings.filterwarnings("ignore", message=msg)
        yield w


def _one_thread(fn: Callable) -> Callable:
    """Run a block function with numba at one thread (numba's count is per thread): scanpy's
    parallel kernels would otherwise start NUMBA_NUM_THREADS threads in every worker."""

    @functools.wraps(fn)
    def run(*args, **kwargs):
        import numba

        before = numba.get_num_threads()
        numba.set_num_threads(1)
        try:
            return fn(*args, **kwargs)
        finally:
            numba.set_num_threads(before)

    return run


def block_bytes(rows: int, n_vars: int, *, nnz_per_row: float = 0.0, dense: bool = False,
                n_hvg: int = 2000) -> int:
    """Working memory of one task on a block of ``rows`` cells (an estimate): the block and
    two working copies, plus the dense variable-gene block in float32 and float64 (PCA) or
    a dense residual block (Pearson residuals), whichever is larger."""
    per_row = n_vars * 4 * 3 if dense else nnz_per_row * (4 + 4) * 3
    per_row += max(n_hvg * (4 + 8), min(n_vars, _GENE_BLOCK) * 8 * 4)
    return int(rows * per_row)


def step_bytes(rows: int, n_vars: int, *, nnz_per_row: float = 0.0, dense: bool = False,
               n_hvg: int = 2000, workers: int | None = None) -> int:
    """Memory of a chunked step over the resident results: one block per worker
    (``block_bytes`` x workers)."""
    w = int(workers or n_workers())
    return w * block_bytes(rows, n_vars, nnz_per_row=nnz_per_row, dense=dense, n_hvg=n_hvg)


def rows_per_block(n_vars: int, *, nnz_per_row: float = 0.0, dense: bool = False,
                   n_hvg: int = 2000, chunk_bytes: int = CHUNK_BYTES) -> int:
    """Rows per block so that one block's working memory is about ``chunk_bytes``."""
    one = block_bytes(1, n_vars, nnz_per_row=nnz_per_row, dense=dense, n_hvg=n_hvg)
    return int(np.clip(chunk_bytes // max(one, 1), MIN_ROWS, MAX_ROWS))


# (blocks done, blocks in all) -> None, called between batches of blocks; may raise to stop.
BlockHook = Callable[[int, int], None]
_BLOCK_HOOK: contextvars.ContextVar[BlockHook | None] = contextvars.ContextVar(
    "scview_chunked_block_hook", default=None)


@contextlib.contextmanager
def on_blocks(hook: BlockHook | None) -> Iterator[None]:
    """Call ``hook(done, total)`` before each batch of blocks of every pass made in this
    thread (and once a pass ends): for progress, and to stop a pass between batches (an
    exception it raises ends the pass)."""
    token = _BLOCK_HOOK.set(hook)
    try:
        yield
    finally:
        _BLOCK_HOOK.reset(token)


def _tick(done: int, total: int) -> None:
    hook = _BLOCK_HOOK.get()
    if hook is not None:
        hook(done, total)


def _blocks(X: da.Array) -> list:
    """One Delayed per row block (the graph optimised once)."""
    if X.numblocks[1] != 1:
        raise ValueError("chunked arrays must have one column block (chunks=(rows, -1))")
    return list(X.to_delayed().ravel())


def _each_block(X: da.Array, fn: Callable, *per_block_args) -> Iterator[Any]:
    """``fn(block, *args_i)`` for every row block, in order, ``workers`` blocks at a time.

    ``per_block_args``: sequences with one entry per block. At most one batch of blocks is
    in memory, so a step's memory is about one block per worker whatever the matrix size."""
    from dask import delayed

    parts = _blocks(X)
    task = delayed(_one_thread(fn), pure=False)
    with scheduler() as w:
        for start in range(0, len(parts), w):
            _tick(start, len(parts))
            idx = range(start, min(len(parts), start + w))
            batch = [task(parts[i], *(a[i] for a in per_block_args)) for i in idx]
            yield from dask.compute(*batch)
        _tick(len(parts), len(parts))


def _row_slices(X: da.Array, values) -> list:
    """``values`` (one per cell) cut into the row blocks of ``X``."""
    bounds = np.cumsum((0,) + tuple(X.chunks[0]))
    return [values[a:b] for a, b in zip(bounds[:-1], bounds[1:])]


def _sum_dtype(dtype) -> np.dtype:
    """The dtype numpy/scipy give a sum of this dtype (int -> int64, float -> itself)."""
    return np.zeros(1, dtype=dtype).sum().dtype


def _is_sparse(X: da.Array) -> bool:
    return sp.issparse(X._meta)


# ---------------------------------------------------------------------------
# Opening a counts matrix
# ---------------------------------------------------------------------------

@dataclass
class LazyCounts:
    """A cells x genes counts matrix on disk, read by row blocks.

    ``X``: dask array (one column block), CSR or numpy blocks. ``obs`` / ``var``: the cells'
    and these genes' annotations. ``source``: where the counts are, as
    ``normalization.find_counts`` names it (``layers['counts']`` | ``.raw`` | ``X``).
    ``index_dtype``: the dtype an in-memory read gives ``n_genes_by_counts`` (the stored
    indptr's; int64 for a dense X)."""

    X: da.Array
    obs: pd.DataFrame
    var: pd.DataFrame
    source: str
    path: str
    index_dtype: np.dtype

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(int(s) for s in self.X.shape)

    @property
    def dense(self) -> bool:
        return not _is_sparse(self.X)

    @property
    def obs_names(self) -> pd.Index:
        return self.obs.index

    @property
    def var_names(self) -> pd.Index:
        return self.var.index

    def subset(self, obs=None, var=None) -> LazyCounts:
        """The cells and genes of two boolean masks (lazily), obs and var cut to match with
        unused categories removed, as anndata's subset leaves them."""
        X = subset(self.X, obs=obs, var=var)
        o = self.obs if obs is None else _drop_unused(self.obs.iloc[np.flatnonzero(obs)])
        v = self.var if var is None else _drop_unused(self.var.iloc[np.flatnonzero(var)])
        return replace(self, X=X, obs=o, var=v)


def _drop_unused(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in df.columns:
        if isinstance(df[c].dtype, pd.CategoricalDtype):
            df[c] = df[c].cat.remove_unused_categories()
    return df


def _encoding(elem) -> str:
    enc = elem.attrs.get("encoding-type", "")
    return enc.decode() if isinstance(enc, bytes) else str(enc)


def _elem_shape(elem) -> tuple[int, int]:
    if isinstance(elem, h5py.Dataset):
        return tuple(int(s) for s in elem.shape)
    return tuple(int(s) for s in elem.attrs["shape"])


def read_lazy(path: str | Path, key: str, *, chunk_rows: int) -> da.Array:
    """One stored matrix (``X``, ``layers/counts``, ``raw/X``...) as a dask array of row
    blocks. CSR and dense matrices, any HDF5 compression (gzip, lzf). Each block opens the
    file itself, so nothing here keeps it open."""
    with h5py.File(path, "r") as f:
        elem = f[key]
        enc = _encoding(elem)
        if enc not in ("csr_matrix", "array"):
            what = enc or "an old (pre-0.7) anndata encoding"
            raise ValueError(
                f"{key} in {Path(path).name} is stored as {what}: the chunked reader needs a "
                "row-major matrix (CSR or dense) written by anndata 0.7 or later. Save the "
                "dataset again with anndata to convert it."
            )
        n_vars = _elem_shape(elem)[1]
        return ad.experimental.read_elem_lazy(elem, chunks=(int(chunk_rows), n_vars))


def _looks_like_counts(X: da.Array) -> bool:
    """``pipeline.looks_like_counts`` on the first rows, as it judges an on-disk matrix."""
    from scview.core.pipeline import looks_like_counts  # the pipeline will import this module

    return looks_like_counts(X[: min(X.shape[0], _COUNTS_ROWS)].compute(scheduler="sync"))


def _take_columns(X: da.Array, idx: np.ndarray) -> da.Array:
    """Columns ``idx`` (positions) of every block, lazily."""
    idx = np.asarray(idx, dtype=np.int64)
    if idx.size == X.shape[1] and np.array_equal(idx, np.arange(X.shape[1])):
        return X

    def take(b):
        return b[:, idx] if sp.issparse(b) else np.take(b, idx, axis=1)

    return X.map_blocks(take, dtype=X.dtype, chunks=(X.chunks[0], (idx.size,)), meta=X._meta)


Rule = Literal["find_counts", "all_genes", "doublets"]


def open_counts(path: str | Path, *, rule: Rule = "find_counts", rows: int | None = None,
                chunk_bytes: int = CHUNK_BYTES, n_hvg: int = 2000) -> LazyCounts:
    """The counts of an h5ad as the in-memory pipeline picks them, read lazily.

    ``rule``:
      - ``find_counts`` (``normalization.find_counts``): ``layers['counts']``, then ``.raw``
        when it holds counts for every gene in X (in X's gene order), then X;
      - ``all_genes`` (``normalization.counts_all_genes``): ``.raw`` when it has more genes
        than X and holds counts, else as ``find_counts``;
      - ``doublets`` (``pipeline._doublet_matrix``): X when it holds counts, else
        ``layers['counts']``, else ``.raw`` (all its genes).
    "Holds counts" is ``pipeline.looks_like_counts`` on the first 2,000 rows, as it judges
    a matrix on disk. Rows per block: ``rows``, else from ``chunk_bytes``
    (``rows_per_block``). Raises ``ValueError`` when no candidate holds counts."""
    from anndata.io import read_elem

    path = str(path)
    with h5py.File(path, "r") as f:
        obs = read_elem(f["obs"])
        var = read_elem(f["var"])
        keys = [k for k in ("X", "layers/counts", "raw/X") if k in f]
        raw_var = read_elem(f["raw/var"]) if "raw/X" in keys else None
        dense = {k: isinstance(f[k], h5py.Dataset) for k in keys}
        nnz = {k: 0 if dense[k] else int(f[k]["indptr"][-1]) for k in keys}
        # What an in-memory read gives n_genes_by_counts: the stored indptr's dtype.
        index_dtype = {k: np.dtype(np.int64) if dense[k] else np.dtype(f[k]["indptr"].dtype)
                       for k in keys}
        n_obs, n_vars = _elem_shape(f["X"])
    if rows is None:
        rows = rows_per_block(n_vars, nnz_per_row=nnz["X"] / max(n_obs, 1), dense=dense["X"],
                              n_hvg=n_hvg, chunk_bytes=chunk_bytes)
    lazy: dict[str, da.Array] = {}

    def holds_counts(key: str) -> bool:
        if key not in keys:
            return False
        if key not in lazy:
            lazy[key] = read_lazy(path, key, chunk_rows=rows)
        return _looks_like_counts(lazy[key])

    def made(key: str, source: str, X=None, v=None, idx_dtype=None) -> LazyCounts:
        return LazyCounts(X=lazy[key] if X is None else X, obs=obs, var=var if v is None else v,
                          source=source, path=path,
                          index_dtype=index_dtype[key] if idx_dtype is None else idx_dtype)

    def find() -> LazyCounts:
        if holds_counts("layers/counts"):
            return made("layers/counts", "layers['counts']")
        if (raw_var is not None and not len(var.index.difference(raw_var.index))
                and holds_counts("raw/X")):
            X = _take_columns(lazy["raw/X"], raw_var.index.get_indexer(var.index))
            if X is lazy["raw/X"]:
                return made("raw/X", ".raw")
            # raw[:, var_names] in memory: scipy picks the index dtype (int32 when it fits)
            big = nnz["raw/X"] > _I32
            return made("raw/X", ".raw", X=X, idx_dtype=np.dtype(np.int64 if big else np.int32))
        if holds_counts("X"):
            return made("X", "X")
        raise ValueError(
            "No raw counts found (layers['counts'], .raw or X): the chunked steps need counts."
        )

    if rule == "find_counts":
        return find()
    if rule == "all_genes":
        if raw_var is not None and len(raw_var) > n_vars and holds_counts("raw/X"):
            return made("raw/X", ".raw", v=raw_var)
        return find()
    if rule == "doublets":
        if holds_counts("X"):
            return made("X", "X")
        if holds_counts("layers/counts"):
            return made("layers/counts", "layers['counts']")
        if holds_counts("raw/X"):
            return made("raw/X", ".raw", v=raw_var)
        raise ValueError("No raw counts found (X, layers['counts'] or .raw).")
    raise ValueError(f"Unknown counts rule '{rule}'")


# ---------------------------------------------------------------------------
# QC metrics
# ---------------------------------------------------------------------------

def mito_mask(var_names) -> np.ndarray:
    """``var['mt']`` as ``pipeline._run_qc_metrics`` sets it (MT- or mt- prefix)."""
    names = pd.Index(var_names)
    return np.asarray(names.str.startswith("MT-") | names.str.startswith("mt-"), dtype=bool)


def _qc_block(b, mt: np.ndarray, index_dtype: np.dtype, sum_dtype: np.dtype) -> dict:
    """Per-cell values exactly as ``describe_obs`` takes them, per-gene sums in float64."""
    if sp.issparse(b):
        b = sp.csr_matrix(b, copy=True)
        b.eliminate_zeros()  # calculate_qc_metrics does this to X first
        n_genes = b.getnnz(axis=1).astype(index_dtype, copy=False)
        n_cells = np.bincount(b.indices, minlength=b.shape[1]).astype(np.int64)
        col = np.bincount(b.indices, weights=b.data, minlength=b.shape[1])
    else:
        n_genes = np.count_nonzero(b, axis=1)
        n_cells = np.count_nonzero(b, axis=0).astype(np.int64)
        col = b.sum(axis=0, dtype=np.float64)
    total = np.ravel(np.sum(b, axis=1))
    total_mt = np.ravel(np.sum(b[:, mt], axis=1))
    return {"n_genes": n_genes, "total": total.astype(sum_dtype, copy=False),
            "total_mt": total_mt.astype(sum_dtype, copy=False), "n_cells": n_cells, "col": col}


def qc_metrics(m: LazyCounts) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(obs columns, var columns) of ``pipeline._run_qc_metrics``, i.e. ``var['mt']`` and
    ``sc.pp.calculate_qc_metrics(qc_vars=['mt'], percent_top=None, log1p=False)``: obs
    n_genes_by_counts, total_counts, total_counts_mt, pct_counts_mt; var mt,
    n_cells_by_counts, mean_counts, pct_dropout_by_counts, total_counts. Same names, order
    and dtypes."""
    X = m.X
    mt = mito_mask(m.var_names)
    sdt = _sum_dtype(X.dtype)
    n_genes, total, total_mt = [], [], []
    n_cells = np.zeros(X.shape[1], dtype=np.int64)
    col = np.zeros(X.shape[1], dtype=np.float64)
    nb = X.numblocks[0]
    for r in _each_block(X, _qc_block, [mt] * nb, [m.index_dtype] * nb, [sdt] * nb):
        n_genes.append(r["n_genes"])
        total.append(r["total"])
        total_mt.append(r["total_mt"])
        n_cells += r["n_cells"]
        col += r["col"]
    n = X.shape[0]
    obs = pd.DataFrame(index=m.obs_names)
    obs["n_genes_by_counts"] = _cat(n_genes, m.index_dtype if not m.dense else np.int64)
    obs["total_counts"] = _cat(total, sdt)
    obs["total_counts_mt"] = _cat(total_mt, sdt)
    obs["pct_counts_mt"] = obs["total_counts_mt"] / obs["total_counts"] * 100
    var = pd.DataFrame(index=m.var_names)
    var["mt"] = mt
    var["n_cells_by_counts"] = n_cells
    var["mean_counts"] = col / n
    var["pct_dropout_by_counts"] = (1 - var["n_cells_by_counts"] / n) * 100
    var["total_counts"] = col.astype(sdt)
    return obs, var


def _cat(parts: list, dtype) -> np.ndarray:
    return np.concatenate(parts) if parts else np.zeros(0, dtype=dtype)


# ---------------------------------------------------------------------------
# Filtering helpers
# ---------------------------------------------------------------------------

def _detected_block(b) -> np.ndarray:
    if sp.issparse(b):
        b = sp.csr_matrix(b)
        return np.bincount(b.indices[b.data > 0], minlength=b.shape[1]).astype(np.int64)
    return (b > 0).sum(axis=0).astype(np.int64)


def gene_detection(X: da.Array) -> np.ndarray:
    """Cells with a value above 0, per gene (int64): the number ``sc.pp.filter_genes``
    compares with ``min_cells`` and the filtering step stores as ``var['n_cells']``."""
    out = np.zeros(X.shape[1], dtype=np.int64)
    for r in _each_block(X, _detected_block):
        out += r
    return out


def subset(X: da.Array, obs=None, var=None) -> da.Array:
    """Rows of a boolean cell mask and columns of a boolean gene mask, lazily, with known
    block sizes (each block keeps its own kept rows; blocks left empty are dropped)."""
    if obs is not None:
        obs = np.asarray(obs, dtype=bool)
        if obs.shape != (X.shape[0],):
            raise ValueError(f"cell mask has {obs.shape[0]} values for {X.shape[0]} cells")
        local = [np.flatnonzero(s) for s in _row_slices(X, obs)]
        if not all(len(r) == c for r, c in zip(local, X.chunks[0])):

            def take(b, block_info=None):
                return b[local[block_info[0]["chunk-location"][0]]]

            X = X.map_blocks(take, dtype=X.dtype, meta=X._meta,
                             chunks=(tuple(len(r) for r in local), X.chunks[1]))
            keep = [i for i, r in enumerate(local) if len(r)]
            if len(keep) < len(local):
                if not keep:
                    raise ValueError("the cell mask keeps no cells")
                X = da.concatenate([X.blocks[i] for i in keep], axis=0)
    if var is not None:
        var = np.asarray(var, dtype=bool)
        if var.shape != (X.shape[1],):
            raise ValueError(f"gene mask has {var.shape[0]} values for {X.shape[1]} genes")
        X = _take_columns(X, np.flatnonzero(var))
    return X


# ---------------------------------------------------------------------------
# Normalisation and log
# ---------------------------------------------------------------------------

def _norm_dtype(dtype) -> np.dtype:
    """normalize_total: integer counts become float32, floats keep their dtype."""
    return np.dtype(np.float32) if np.issubdtype(dtype, np.integer) else np.dtype(dtype)


def _meta_like(X: da.Array, dtype) -> Any:
    return sp.csr_matrix((0, 0), dtype=dtype) if _is_sparse(X) else np.empty((0, 0), dtype)


def _normalized(b, target_sum: float):
    import scanpy as sc

    return sc.pp.normalize_total(ad.AnnData(X=b), target_sum=target_sum, inplace=False)["X"]


def _cell_totals_block(b) -> np.ndarray:
    """normalize_total's per-cell counts (its ``norm_factor`` with target_sum 1)."""
    import scanpy as sc

    f = sc.pp.normalize_total(ad.AnnData(X=b), target_sum=1.0, inplace=False)["norm_factor"]
    return np.asarray(f)


def normalize_target(X: da.Array) -> float:
    """The target sum ``normalize_total(target_sum=None)`` picks: the median of the cells'
    counts (a CSR matrix: over all cells; dense: over cells with counts, as scanpy does)."""
    counts = _cat(list(_each_block(X, _cell_totals_block)), _norm_dtype(X.dtype))
    return np.median(counts) if _is_sparse(X) else np.median(counts[counts > 0])


def normalize_total(X: da.Array, target_sum: float | None = 1e4) -> da.Array:
    """``sc.pp.normalize_total(target_sum=...)`` lazily (``pipeline._run_normalization``):
    scanpy's own function on each block, so the values are the in-memory ones."""
    target = normalize_target(X) if target_sum is None else target_sum
    dt = _norm_dtype(X.dtype)
    return X.map_blocks(_one_thread(_normalized), target, dtype=dt, meta=_meta_like(X, dt))


def _log1p(b):
    import scanpy as sc

    return sc.pp.log1p(b, copy=True)


def log1p(X: da.Array) -> da.Array:
    """``sc.pp.log1p`` lazily (``pipeline._run_log_transform``; natural log)."""
    dt = np.dtype(np.float64) if np.issubdtype(X.dtype, np.integer) else X.dtype
    return X.map_blocks(_one_thread(_log1p), dtype=dt, meta=_meta_like(X, dt))


def _log_normalized(b, target_sum: float):
    import scanpy as sc

    return sc.pp.log1p(_normalized(b, target_sum), copy=False)


def log_normalize(X: da.Array, target_sum: float | None = 1e4) -> da.Array:
    """``normalize_total`` then ``log1p`` in one block function (one working copy)."""
    target = normalize_target(X) if target_sum is None else target_sum
    dt = _norm_dtype(X.dtype)
    return X.map_blocks(_one_thread(_log_normalized), target, dtype=dt, meta=_meta_like(X, dt))


# ---------------------------------------------------------------------------
# Per-gene statistics by block
# ---------------------------------------------------------------------------

def _group_stats_block(b, codes: np.ndarray, n_groups: int, expm1: bool,
                       cols: np.ndarray | None) -> dict:
    """Per (group, gene): cells above 0, sum (float64) and sum of squares (each square in the
    matrix's dtype, as scanpy's ``_get_mean_var`` squares, summed in float64)."""
    if cols is not None:
        b = b[:, cols] if sp.issparse(b) else np.take(b, cols, axis=1)
    b = _float_counts(b)
    g = b.shape[1]
    if sp.issparse(b):
        b = sp.csr_matrix(b)
        v = np.expm1(b.data) if expm1 else b.data
        flat = np.repeat(codes.astype(np.int64), np.diff(b.indptr)) * g + b.indices
        size = n_groups * g
        s1 = np.bincount(flat, weights=v, minlength=size)
        s2 = np.bincount(flat, weights=v * v, minlength=size)
        s0 = np.bincount(flat[v > 0], minlength=size)
        return {"s0": s0.reshape(n_groups, g), "s1": s1.reshape(n_groups, g),
                "s2": s2.reshape(n_groups, g)}
    v = np.expm1(b) if expm1 else b
    s0 = np.zeros((n_groups, g), dtype=np.int64)
    s1 = np.zeros((n_groups, g), dtype=np.float64)
    s2 = np.zeros((n_groups, g), dtype=np.float64)
    for k in np.unique(codes):
        rows = v[codes == k]
        s0[k] = (rows > 0).sum(axis=0)
        s1[k] = rows.sum(axis=0, dtype=np.float64)
        s2[k] = (rows * rows).sum(axis=0, dtype=np.float64)
    return {"s0": s0, "s1": s1, "s2": s2}


def _float_counts(b):
    """Integer counts as float64 (exact): scanpy's numba kernels square an integer into
    int64, numpy's int32 product would overflow."""
    if np.issubdtype(b.dtype, np.floating):
        return b
    return b.astype(np.float64)


@dataclass
class _Stats:
    n: np.ndarray  # cells per group
    s0: np.ndarray  # (groups, genes)
    s1: np.ndarray
    s2: np.ndarray

    def mean_var(self, k: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """``scanpy.preprocessing._utils._get_mean_var`` of group ``k`` (None: all cells)."""
        if k is None:
            n, s1, s2 = float(self.n.sum()), self.s1.sum(axis=0), self.s2.sum(axis=0)
        else:
            n, s1, s2 = float(self.n[k]), self.s1[k], self.s2[k]
        mean = s1 / n
        var = s2 / n - mean**2
        if n != 1:
            var *= n / (n - 1)
        return mean, var


def _group_stats(X: da.Array, codes: np.ndarray | None = None, n_groups: int = 1, *,
                 expm1: bool = False, cols: np.ndarray | None = None) -> _Stats:
    codes = np.zeros(X.shape[0], dtype=np.int64) if codes is None else np.asarray(codes)
    g = X.shape[1] if cols is None else len(cols)
    out = _Stats(np.bincount(codes, minlength=n_groups), np.zeros((n_groups, g), np.int64),
                 np.zeros((n_groups, g)), np.zeros((n_groups, g)))
    nb = X.numblocks[0]
    for r in _each_block(X, _group_stats_block, _row_slices(X, codes), [n_groups] * nb,
                         [expm1] * nb, [cols] * nb):
        out.s0 += r["s0"]
        out.s1 += r["s1"]
        out.s2 += r["s2"]
    return out


# ---------------------------------------------------------------------------
# Highly variable genes
# ---------------------------------------------------------------------------

HVG_FLAVORS: tuple[str, ...] = ("seurat", "cell_ranger", "seurat_v3")


class _NVars:
    """What scanpy's ``_subset_genes`` reads of the AnnData it is given."""

    def __init__(self, n_vars: int):
        self.n_vars = n_vars


def _batch_codes(batch, n_obs: int, *, unique: bool) -> tuple[np.ndarray, list]:
    """(code per cell, batch labels). ``unique``: the labels ``np.unique`` sorts out of the
    values (seurat_v3, Pearson residuals); else the categories, as the batched dispersion
    flavours take them after ``sanitize_anndata`` (strings become sorted categories; unused
    categories dropped, as anndata's subset leaves them)."""
    if batch is None:
        return np.zeros(n_obs, dtype=np.int64), [0]
    values = batch.array if isinstance(batch, pd.Series) else batch
    if len(values) != n_obs:
        raise ValueError(f"batch has {len(values)} values for {n_obs} cells")
    if unique:
        labels, codes = np.unique(np.asarray(values), return_inverse=True)
        return codes.astype(np.int64).ravel(), list(labels)
    cat = values if isinstance(values, pd.Categorical) else pd.Categorical(values)
    cat = cat.remove_unused_categories()
    return np.asarray(cat.codes, dtype=np.int64), list(cat.categories)


def _dispersion_frame(mean, var, var_names: pd.Index, filt, flavor: str, n_top: int,
                      n_bins: int) -> pd.DataFrame:
    """``_highly_variable_genes_single_batch`` from its ``mean, var`` on, with scanpy's own
    binning (``_get_mean_bins``), bin statistics (``_get_disp_stats``) and top-n cut
    (``_subset_genes``). ``filt``: genes kept (the batched path drops undetected ones)."""
    from scanpy.preprocessing._highly_variable_genes import (
        _get_disp_stats,
        _get_mean_bins,
        _subset_genes,
    )

    n_removed = int(np.sum(~filt))
    mean = np.array(mean[filt] if n_removed else mean, dtype=np.float64)
    var = np.array(var[filt] if n_removed else var, dtype=np.float64)
    mean[mean == 0] = 1e-12
    dispersion = var / mean
    if flavor == "seurat":
        dispersion[dispersion == 0] = np.nan
        dispersion = np.log(dispersion)
        mean = np.log1p(mean)
    df = pd.DataFrame(dict(zip(["means", "dispersions"], (mean, dispersion), strict=True)))
    df["mean_bin"] = _get_mean_bins(df["means"], flavor, n_bins)
    disp_stats = _get_disp_stats(df, flavor)
    df["dispersions_norm"] = (df["dispersions"] - disp_stats["avg"]) / disp_stats["dev"]
    names = var_names[filt] if n_removed else var_names
    df["highly_variable"] = _subset_genes(
        _NVars(len(names)), mean=mean, dispersion_norm=df["dispersions_norm"].to_numpy(),
        cutoff=n_top,
    )
    df.index = names
    if n_removed > 0:
        missing_hvg = pd.DataFrame(np.zeros((n_removed, len(df.columns))), columns=df.columns)
        missing_hvg["highly_variable"] = missing_hvg["highly_variable"].astype(bool)
        missing_hvg.index = var_names[~filt]
        df = pd.concat([df, missing_hvg]).loc[var_names]
    return df


def _combine_batches(dfs: list[pd.DataFrame], var_names: pd.Index, n_top: int) -> pd.DataFrame:
    """``_highly_variable_genes_batched`` after its per-batch frames (n_top_genes mode)."""
    n_batches = len(dfs)
    df = pd.concat(dfs, axis=0)
    df["highly_variable"] = df["highly_variable"].astype(int)
    df = df.groupby(df.index, observed=True).agg(
        dict(means="mean", dispersions="mean", dispersions_norm="mean", highly_variable="sum")
    )
    df["highly_variable_nbatches"] = df["highly_variable"]
    df["highly_variable_intersection"] = df["highly_variable_nbatches"] == n_batches
    df.sort_values(["highly_variable_nbatches", "dispersions_norm"], ascending=False,
                   na_position="last", inplace=True)
    df["highly_variable"] = np.arange(df.shape[0]) < n_top
    return df.loc[var_names]


def _dispersion_hvg(X, var_names, flavor, n_top, batch, n_bins) -> pd.DataFrame:
    codes, labels = _batch_codes(batch, X.shape[0], unique=False)
    st = _group_stats(X, codes, len(labels), expm1=flavor == "seurat")
    if batch is None:
        mean, var = st.mean_var(0)
        df = _dispersion_frame(mean, var, var_names, np.ones(len(var_names), bool), flavor,
                               n_top, n_bins)
    else:
        dfs = []
        for k in range(len(labels)):
            mean, var = st.mean_var(k)
            # filter_genes(min_cells=1) on the batch: cells above 0 (before expm1)
            dfs.append(_dispersion_frame(mean, var, var_names, st.s0[k] >= 1, flavor, n_top,
                                         n_bins))
        df = _combine_batches(dfs, var_names, n_top)
    # What highly_variable_genes(inplace=True) writes to var.
    out = pd.DataFrame(index=var_names)
    out["highly_variable"] = df["highly_variable"]
    out["means"] = df["means"]
    out["dispersions"] = df["dispersions"]
    out["dispersions_norm"] = df["dispersions_norm"].astype(np.float32, copy=False)
    if batch is not None:
        out["highly_variable_nbatches"] = df["highly_variable_nbatches"]
        out["highly_variable_intersection"] = df["highly_variable_intersection"]
    return out


def _clipped_block(b, codes: np.ndarray, clip: np.ndarray) -> dict:
    """Per (batch, gene): sum and sum of squares of min(count, clip), as Seurat v3 clips
    (``_sum_and_sum_squares_clipped``; zeros add nothing)."""
    n_groups, g = clip.shape
    if sp.issparse(b):
        b = sp.csr_matrix(b)
        flat = np.repeat(codes.astype(np.int64), np.diff(b.indptr)) * g + b.indices
        e = np.minimum(b.data.astype(np.float64), clip.ravel()[flat])
        size = n_groups * g
        return {"s1": np.bincount(flat, weights=e, minlength=size).reshape(n_groups, g),
                "s2": np.bincount(flat, weights=e**2, minlength=size).reshape(n_groups, g)}
    s1 = np.zeros((n_groups, g))
    s2 = np.zeros((n_groups, g))
    for k in np.unique(codes):
        rows = b[codes == k].astype(np.float64)
        cv = np.broadcast_to(clip[k], rows.shape)
        np.putmask(rows, rows > cv, cv)
        s1[k] = rows.sum(axis=0)
        s2[k] = np.square(rows).sum(axis=0)
    return {"s1": s1, "s2": s2}


def _seurat_v3_hvg(X, var_names, n_top, batch, span: float = 0.3) -> pd.DataFrame:
    """``_highly_variable_genes_seurat_v3`` (flavor seurat_v3) with its two passes over the
    counts by block: mean/variance per batch, then the clipped sums. Returns its
    ``inplace=False`` frame, which ``pipeline._seurat_v3_hvg`` copies into var."""
    from skmisc.loess import loess

    codes, labels = _batch_codes(batch, X.shape[0], unique=True)
    st = _group_stats(X, codes, len(labels))
    df = pd.DataFrame(index=var_names)
    df["means"], df["variances"] = st.mean_var(None)
    g = X.shape[1]
    clip = np.zeros((len(labels), g))
    fits = []
    for k in range(len(labels)):
        mean, var = st.mean_var(k)
        not_const = var > 0
        estimat_var = np.zeros(g, dtype=np.float64)
        y = np.log10(var[not_const])
        x = np.log10(mean[not_const])
        model = loess(x, y, span=span, degree=2)
        model.fit()
        estimat_var[not_const] = model.outputs.fitted_values
        reg_std = np.sqrt(10**estimat_var)
        n_obs = int(st.n[k])
        vmax = np.sqrt(n_obs)
        clip[k] = reg_std * vmax + mean
        fits.append((mean, reg_std, n_obs))
    s1 = np.zeros_like(clip)
    s2 = np.zeros_like(clip)
    for r in _each_block(X, _clipped_block, _row_slices(X, codes), [clip] * X.numblocks[0]):
        s1 += r["s1"]
        s2 += r["s2"]
    norm_gene_vars = []
    for k, (mean, reg_std, n_obs) in enumerate(fits):
        norm_gene_var = (1 / ((n_obs - 1) * np.square(reg_std))) * (
            (n_obs * np.square(mean)) + s2[k] - 2 * s1[k] * mean
        )
        norm_gene_vars.append(norm_gene_var.reshape(1, -1))
    norm_gene_vars = np.concatenate(norm_gene_vars, axis=0)
    ranked_norm_gene_vars = np.argsort(np.argsort(-norm_gene_vars, axis=1), axis=1)
    ranked_norm_gene_vars = ranked_norm_gene_vars.astype(np.float32)
    num_batches_high_var = np.sum((ranked_norm_gene_vars < n_top).astype(int), axis=0)
    ranked_norm_gene_vars[ranked_norm_gene_vars >= n_top] = np.nan
    ma_ranked = np.ma.masked_invalid(ranked_norm_gene_vars)
    median_ranked = np.ma.median(ma_ranked, axis=0).filled(np.nan)
    df["gene_name"] = df.index
    df["highly_variable_nbatches"] = num_batches_high_var
    df["highly_variable_rank"] = median_ranked
    df["variances_norm"] = np.mean(norm_gene_vars, axis=0)
    sort_cols = ["highly_variable_rank", "highly_variable_nbatches"]
    sorted_index = (df[sort_cols].sort_values(sort_cols, ascending=[True, False],
                                              na_position="last").index)
    df["highly_variable"] = False
    df.loc[sorted_index[: int(n_top)], "highly_variable"] = True
    if batch is None:
        df = df.drop(["highly_variable_nbatches"], axis=1)
    return df


def highly_variable_genes(X: da.Array, var_names, *, flavor: str = "seurat",
                          n_top_genes: int = 2000, batch=None, n_bins: int = 20) -> pd.DataFrame:
    """Variable genes as ``pipeline._run_hvg`` picks them (n_top_genes mode).

    ``X``: the log-normalised matrix for ``seurat`` / ``cell_ranger``, the counts for
    ``seurat_v3``. ``batch``: one label per cell (the ``hvg_batch_key`` column) or None.
    Returns the var columns the in-memory step writes: for the dispersion flavours those of
    ``sc.pp.highly_variable_genes(inplace=True)`` (highly_variable, means, dispersions,
    dispersions_norm, + highly_variable_nbatches/intersection with a batch; scanpy also
    sets ``uns['hvg'] = {'flavor': flavor}``), for seurat_v3 scanpy's ``inplace=False``
    frame, which ``pipeline._seurat_v3_hvg`` copies (pass ``n_top_genes`` as it does:
    ``min(n_top_genes, n_vars)``)."""
    var_names = pd.Index(var_names)
    if flavor == "seurat_v3":
        return _seurat_v3_hvg(X, var_names, int(n_top_genes), batch)
    if flavor not in HVG_FLAVORS:
        raise ValueError(f"Unsupported HVG flavor '{flavor}'; choose from {', '.join(HVG_FLAVORS)}.")
    return _dispersion_hvg(X, var_names, flavor, int(n_top_genes), batch, n_bins)


# --- Pearson residuals ------------------------------------------------------

def _gene_cell_sums_block(b, codes: np.ndarray, n_groups: int) -> dict:
    """Per (batch, gene) sums and sums of squares (float64), and each cell's sum over every
    gene (float64; the Pearson-residual HVG sums a batch in its own dtype, exact for
    counts)."""
    b = _float_counts(b)
    g = b.shape[1]
    if sp.issparse(b):
        b = sp.csr_matrix(b)
        per_row = np.diff(b.indptr)
        flat = np.repeat(codes.astype(np.int64), per_row) * g + b.indices
        size = n_groups * g
        s1 = np.bincount(flat, weights=b.data, minlength=size).reshape(n_groups, g)
        s2 = np.bincount(flat, weights=b.data * b.data, minlength=size).reshape(n_groups, g)
        row = np.repeat(np.arange(b.shape[0]), per_row)
        cells = np.bincount(row, weights=b.data, minlength=b.shape[0])
        return {"s1": s1, "s2": s2, "cells": cells}
    s1 = np.zeros((n_groups, g))
    s2 = np.zeros((n_groups, g))
    for k in np.unique(codes):
        rows = b[codes == k]
        s1[k] = rows.sum(axis=0, dtype=np.float64)
        s2[k] = (rows * rows).sum(axis=0, dtype=np.float64)
    return {"s1": s1, "s2": s2, "cells": b.sum(axis=1, dtype=np.float64)}


def _residual_sums_block(b, codes: np.ndarray, cells: np.ndarray, genes: np.ndarray,
                         totals: np.ndarray, nonzero: np.ndarray, theta: float,
                         clip: float) -> dict:
    """Per (batch, gene): sum and sum of squares of the clipped residuals over these cells,
    every cell counted (zeros too). mu = gene sum x cell sum (in their dtype: float32 for a
    float32 CSR, as scanpy's sparse kernel multiplies them) / the batch total (float64)."""
    n_groups, g = genes.shape
    s1 = np.zeros((n_groups, g))
    s2 = np.zeros((n_groups, g))
    for k in np.unique(codes):
        rows = np.flatnonzero(codes == k)
        sub = b[rows]
        for j0 in range(0, g, _GENE_BLOCK):
            j1 = min(g, j0 + _GENE_BLOCK)
            C = sub[:, j0:j1]
            C = (C.toarray() if sp.issparse(C) else np.asarray(C)).astype(np.float64)
            mu = np.outer(cells[rows], genes[k, j0:j1]).astype(np.float64) / totals[k]
            with np.errstate(divide="ignore", invalid="ignore"):
                R = (C - mu) / np.sqrt(mu + mu * mu / theta)
            np.clip(R, -clip, clip, out=R)
            R[:, ~nonzero[k, j0:j1]] = 0.0  # genes without counts in the batch: left out
            s1[k, j0:j1] = R.sum(axis=0)
            s2[k, j0:j1] = (R * R).sum(axis=0)
    return {"s1": s1, "s2": s2}


def pearson_hvg(counts: da.Array, var_names, *, n_top_genes: int, batch=None,
                theta: float = normalization.THETA, clip: float | None = None) -> pd.DataFrame:
    """``sc.experimental.pp.highly_variable_genes(flavor='pearson_residuals', inplace=False)``
    by blocks (``pipeline._pearson_hvg``; pass ``n_top_genes`` as it does:
    ``min(n_top_genes, n_vars)``). Residuals are formed for every cell and gene of each
    block (dense, a block of genes at a time); their variance per gene comes from the sum
    and sum of squares (scanpy: two passes), equal to ~1e-15 relative. As in scanpy,
    ``clip=None`` takes sqrt(cells of the first batch) for every batch."""
    var_names = pd.Index(var_names)
    n_top = int(n_top_genes)
    if theta <= 0:
        raise ValueError("Pearson residuals require theta > 0")
    codes, labels = _batch_codes(batch, counts.shape[0], unique=True)
    n_groups = len(labels)
    nb = counts.numblocks[0]
    s1 = np.zeros((n_groups, counts.shape[1]))
    s2 = np.zeros_like(s1)
    cells = []
    for r in _each_block(counts, _gene_cell_sums_block, _row_slices(counts, codes),
                         [n_groups] * nb):
        s1 += r["s1"]
        s2 += r["s2"]
        cells.append(r["cells"])
    n_per = np.bincount(codes, minlength=n_groups)
    nonzero = s1 != 0
    # scanpy sums a float32 CSR batch in float32 (exact for counts below 2**24) and forms mu
    # from those float32 sums; a dense batch it converts to float64 first.
    dt = np.float32 if _is_sparse(counts) and counts.dtype == np.float32 else np.float64
    genes = s1.astype(dt)
    cells = _cat(cells, np.float64).astype(dt)
    totals = np.array([np.float64(np.sum(genes[k][nonzero[k]])) for k in range(n_groups)])
    if clip is None:
        clip = float(np.sqrt(n_per[0]))
    if clip < 0:
        raise ValueError("Pearson residuals require `clip>=0` or `clip=None`.")
    r1 = np.zeros_like(s1)
    r2 = np.zeros_like(s1)
    for r in _each_block(counts, _residual_sums_block, _row_slices(counts, codes),
                         _row_slices(counts, cells), [genes] * nb, [totals] * nb,
                         [nonzero] * nb, [float(theta)] * nb, [float(clip)] * nb):
        r1 += r["s1"]
        r2 += r["s2"]
    n = n_per[:, None].astype(np.float64)
    residual_gene_vars = np.where(nonzero, r2 / n - (r1 / n) ** 2, 0.0)
    ranks_residual_var = np.argsort(np.argsort(-residual_gene_vars, axis=1), axis=1)
    ranks_residual_var = ranks_residual_var.astype(np.float32)
    highly_variable_nbatches = np.sum((ranks_residual_var < n_top).astype(int), axis=0)
    ranks_residual_var[ranks_residual_var >= n_top] = np.nan
    ranks_masked_array = np.ma.masked_invalid(ranks_residual_var)
    medianrank_residual_var = np.ma.median(ranks_masked_array, axis=0).filled(np.nan)
    means, variances = _Stats(n_per, nonzero, s1, s2).mean_var(None)
    df = pd.DataFrame.from_dict(dict(
        means=means,
        variances=variances,
        residual_variances=np.mean(residual_gene_vars, axis=0),
        highly_variable_rank=medianrank_residual_var,
        highly_variable_nbatches=highly_variable_nbatches.astype(np.int64),
        highly_variable_intersection=highly_variable_nbatches == n_groups,
    ))
    df = df.set_index(var_names)
    df.sort_values(["highly_variable_nbatches", "highly_variable_rank"], ascending=[False, True],
                   na_position="last", inplace=True)
    high_var = np.zeros(df.shape[0], dtype=bool)
    high_var[:n_top] = True
    df["highly_variable"] = high_var
    df = df.loc[var_names, :]
    if batch is None:
        df = df.drop(["highly_variable_nbatches", "highly_variable_intersection"], axis=1)
    return df


# ---------------------------------------------------------------------------
# Scaling + PCA of the variable genes
# ---------------------------------------------------------------------------

PCA_PARAMS = {"zero_center": True, "use_highly_variable": True, "mask_var": "highly_variable"}


def _as_float32(b):
    return b.astype(np.float32)


def _dense_cols(b, cols: np.ndarray) -> np.ndarray:
    """The block's columns ``cols`` as a dense float32 array (row-major)."""
    d = b[:, cols].toarray() if sp.issparse(b) else np.take(b, cols, axis=1)
    return d.astype(np.float32, copy=False)


def _scaled_block(b, cols, mean, std, max_value) -> np.ndarray:
    """``sc.pp.scale(max_value=...)`` of these cells: the same in-place float32 operations
    (minus the float64 mean, over the float64 sd, clipped at +-max_value)."""
    Z = _dense_cols(b, cols)
    Z -= mean
    np.true_divide(Z, std, out=Z)
    if max_value is not None:
        np.clip(Z, -max_value, max_value, out=Z)
    return Z


def _residual_block(b, cols, genes, total, theta, clip) -> np.ndarray:
    """``normalization.pearson_residuals(columns=cols)`` of these cells (float32): cell sums
    over every gene of the block, the given gene sums and total."""
    cells = np.asarray(b.sum(axis=1), dtype=np.float64).ravel()
    C = b[:, cols] if sp.issparse(b) else np.take(b, cols, axis=1)
    out = np.empty((b.shape[0], len(cols)), dtype=np.float32)
    for j0 in range(0, len(cols), _GENE_BLOCK):
        j1 = min(len(cols), j0 + _GENE_BLOCK)
        out[:, j0:j1] = normalization._block(C, j0, j1, cells, genes, total, theta, clip)
    return out


def _cov_block(b, transform: Callable, args: tuple) -> dict:
    Z = transform(b, *args).astype(np.float64)
    return {"zz": Z.T @ Z, "z": Z.sum(axis=0)}


def _proj_block(b, transform: Callable, args: tuple, mean: np.ndarray, V: np.ndarray):
    Z = transform(b, *args).astype(np.float64)
    return ((Z - mean) @ V).astype(np.float32)


def _pca_blocks(X: da.Array, transform: Callable, args: tuple, hvg: np.ndarray,
                n_comps: int) -> dict:
    """PCA of the transformed variable-gene blocks: covariance by blocks, ``eigh``,
    projection by blocks; laid out as ``sc.tl.pca`` stores it with a variable-gene mask."""
    n = X.shape[0]
    k = int(hvg.sum())
    n_comps = int(min(n_comps, n - 1, k - 1))
    if n_comps < 1:
        raise ValueError(f"PCA needs at least 2 cells and 2 variable genes ({n} x {k})")
    nb = X.numblocks[0]
    zz = np.zeros((k, k))
    z = np.zeros(k)
    for r in _each_block(X, _cov_block, [transform] * nb, [args] * nb):
        zz += r["zz"]
        z += r["z"]
    mean = z / n
    # sklearn centres the transformed matrix on its own mean (clipping moves it off 0).
    cov = (zz - n * np.outer(mean, mean)) / (n - 1)
    w, vecs = np.linalg.eigh(cov)
    order = np.argsort(w)[::-1][:n_comps]
    V = vecs[:, order]
    # svd_flip(u_based_decision=False): each component's largest loading positive.
    signs = np.sign(V[np.argmax(np.abs(V), axis=0), np.arange(n_comps)])
    signs[signs == 0] = 1
    V = V * signs
    X_pca = np.concatenate(list(_each_block(X, _proj_block, [transform] * nb, [args] * nb,
                                            [mean] * nb, [V] * nb)))
    pcs = np.zeros((len(hvg), n_comps))
    pcs[hvg] = V.astype(np.float32)
    variance = w[order]
    total_var = float(np.trace(cov))
    return {
        "X_pca": X_pca,
        "PCs": pcs,
        "variance": variance.astype(np.float32),
        "variance_ratio": (variance / total_var).astype(np.float32),
        "params": dict(PCA_PARAMS),
        "n_comps": n_comps,
    }


def scaled_stats(X: da.Array, cols) -> tuple[np.ndarray, np.ndarray]:
    """(mean, sd) of columns ``cols`` of X as ``sc.pp.scale`` takes them on the float32 copy
    (float64 sums of the float32 values and squares, sd with ddof 1, 0 -> 1); one pass."""
    if X.dtype != np.float32:
        X = X.map_blocks(_as_float32, dtype=np.float32, meta=_meta_like(X, np.float32))
    mean, var = _group_stats(X, cols=np.asarray(cols, dtype=np.int64)).mean_var(0)
    std = np.sqrt(var)
    std[std == 0] = 1
    return mean, std


def pca_scaled(X: da.Array, hvg, *, n_comps: int = 50, max_value: float | None = 10,
               stats: tuple[np.ndarray, np.ndarray] | None = None) -> dict:
    """PCA of the scaled variable genes, as "copy of the HVG columns (float32) ->
    ``sc.pp.scale(max_value=10)`` -> ``sc.tl.pca(svd_solver='arpack')``" (the same values
    as scaling every gene and masking: scaling is per gene).

    ``X``: the log-normalised matrix (lazy), ``hvg``: boolean mask over its genes. The
    genes' mean and sd come from one pass (scanpy's ``_get_mean_var``: float64 sums of the
    float32 values and squares, sd with ddof 1, 0 -> 1); the scaled block exists only inside
    a task. Returns X_pca (float32, cells x comps), PCs (float64, genes x comps, zero rows
    for other genes, float32 values), variance and variance_ratio (float32), params (as
    ``uns['pca']['params']``), n_comps = min(n_comps, cells - 1, variable genes - 1).
    ``stats``: the genes' (mean, sd) already worked out (``scaled_stats``, the scaling step's
    plan); mean 0 and sd 1 with ``max_value=None`` is PCA of the unscaled values."""
    hvg = np.asarray(hvg, dtype=bool)
    cols = np.flatnonzero(hvg)
    if X.dtype != np.float32:  # the copy is float32 before its mean and sd are taken
        X = X.map_blocks(_as_float32, dtype=np.float32, meta=_meta_like(X, np.float32))
    mean, std = scaled_stats(X, cols) if stats is None else stats
    return _pca_blocks(X, _scaled_block, (cols, mean, std, max_value), hvg, n_comps)


def pearson_gene_sums(counts: da.Array) -> tuple[np.ndarray, float]:
    """(gene sums, total) as ``normalization._sums`` takes them: column sums in the
    matrix's sum dtype (float32 for float32 counts: exact below 2**24), as float64."""
    st = _group_stats(counts)
    genes = st.s1[0].astype(_sum_dtype(counts.dtype)).astype(np.float64)
    return genes, float(genes.sum())


def pca_pearson(counts: da.Array, hvg, *, n_comps: int = 50, theta: float = normalization.THETA,
                clip: float | None = None) -> dict:
    """PCA of the Pearson residuals of the variable genes (``normalization.pearson_residuals``
    with clip sqrt(cells) by default, as ``pipeline._run_scaling`` writes them, then
    ``sc.tl.pca``). ``counts``: the filtered counts (lazy). Same output as ``pca_scaled``."""
    hvg = np.asarray(hvg, dtype=bool)
    cols = np.flatnonzero(hvg)
    clip = normalization.default_clip(counts.shape[0]) if clip is None else float(clip)
    genes, total = pearson_gene_sums(counts)
    return _pca_blocks(counts, _residual_block, (cols, genes[cols], total, float(theta), clip),
                       hvg, n_comps)


# ---------------------------------------------------------------------------
# Streaming writers
# ---------------------------------------------------------------------------

def _as_written(b):
    return sp.csr_matrix(b) if sp.issparse(b) else np.asarray(b)


class CsrAppender:
    """A CSR matrix written into ``group[key]`` as anndata encodes it, rows appended a block
    at a time (int64 indices, as anndata writes a dask CSR): the first block written, the
    rest appended. ``finish`` writes an empty matrix of the right width when no block came."""

    def __init__(self, group: h5py.Group, key: str, *, compression: str | None = "gzip"):
        self.group, self.key = group, key
        self.kw = {"compression": compression} if compression else {}
        self._disk = None
        self.n_rows = 0

    def add(self, b) -> None:
        from anndata.io import sparse_dataset, write_elem

        b = sp.csr_matrix(b)
        if self._disk is None:
            b.indptr = b.indptr.astype(np.int64, copy=False)
            b.indices = b.indices.astype(np.int64, copy=False)
            write_elem(self.group, self.key, b, dataset_kwargs=self.kw)
            self._disk = sparse_dataset(self.group[self.key])
        elif b.shape[0]:
            self._disk.append(b)
        self.n_rows += b.shape[0]

    def finish(self, n_vars: int, dtype) -> None:
        from anndata.io import write_elem

        if self._disk is None:
            write_elem(self.group, self.key, sp.csr_matrix((0, n_vars), dtype=dtype),
                       dataset_kwargs=self.kw)


def write_matrix(group: h5py.Group, key: str, X: da.Array, *,
                 compression: str | None = "gzip") -> None:
    """Write a dask matrix into ``group[key]`` as anndata encodes it, one batch of blocks in
    memory at a time: a CSR matrix block by block (``CsrAppender``), a dense one by
    ``write_elem`` (dask's store). ``compression``: gzip | lzf | None."""
    from anndata.io import write_elem

    if not _is_sparse(X):
        kw = {"compression": compression} if compression else {}
        with scheduler():
            write_elem(group, key, X, dataset_kwargs=kw)
        return
    out = CsrAppender(group, key, compression=compression)
    for b in _each_block(X, _as_written):
        out.add(b)
    out.finish(X.shape[1], X.dtype)


def write_h5ad(path: str | Path, X: da.Array, *, obs: pd.DataFrame, var: pd.DataFrame,
               obsm: dict | None = None, varm: dict | None = None, obsp: dict | None = None,
               varp: dict | None = None, uns: dict | None = None,
               compression: str | None = "gzip") -> None:
    """An h5ad with X streamed from a dask matrix and everything else written by anndata.

    anndata writes the rest (``AnnData(obs=..., var=..., obsm=..., uns=...)`` with no X,
    strings to categoricals, the same compression), then X is streamed in
    (``write_matrix``). No layers and no .raw (a large dataset keeps its counts in a
    sidecar). Written to a temporary name first, so a failed write leaves no half file.
    The result opens with ``ad.read_h5ad(path, backed="r")``."""
    path = Path(path)
    if X.shape != (len(obs), len(var)):
        raise ValueError(f"X is {X.shape}, obs x var is {(len(obs), len(var))}")
    tmp = path.with_name(f".{path.name}.part")
    meta = ad.AnnData(obs=obs, var=var, obsm=obsm, varm=varm, obsp=obsp, varp=varp,
                      uns=uns or {})
    try:
        meta.write_h5ad(tmp, compression=compression)
        with h5py.File(tmp, "r+") as f:
            if "X" in f:
                del f["X"]
            write_matrix(f, "X", X, compression=compression)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def write_counts(path: str | Path, X: da.Array, *, key: str = "counts",
                 compression: str | None = "gzip") -> None:
    """A counts sidecar (``<id>.counts.h5``): one matrix in an h5 file, readable with
    ``read_lazy(path, key, ...)`` or ``anndata.io.read_elem``."""
    path = Path(path)
    tmp = path.with_name(f".{path.name}.part")
    try:
        with h5py.File(tmp, "w") as f:
            write_matrix(f, key, X, compression=compression)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
