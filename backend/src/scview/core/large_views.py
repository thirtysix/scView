"""Views of a large dataset computed from its files on disk (core/large_mode.py), each
equal to what the in-memory code gives on the same matrix.

- ``sum_rows_by_code``: per-group sums of the counts (pseudobulk profiles) and genes
  detected per cell, in one pass over the counts sidecar by row blocks.
- ``column_and_totals``: one gene's counts per cell and every cell's total counts, in
  one pass over the counts sidecar (the counts are cell-major: a gene needs every row).
- ``gene_means`` / ``score_genes``: ``sc.tl.score_genes`` from the gene store: the
  genes' means over all cells (one pass over the store), the control genes drawn as
  scanpy draws them, and the cells' means over the set and the controls. Sums are
  taken as scipy takes them for scanpy (dtype, order), so the scores are scanpy's.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp


def _rows_in(rows: np.ndarray, start: int, stop: int) -> tuple[int, int]:
    """The slice of the sorted ``rows`` that falls in [start, stop)."""
    return int(np.searchsorted(rows, start)), int(np.searchsorted(rows, stop))


def sum_rows_by_code(M, rows: np.ndarray, codes: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """(genes detected per cell of ``rows``, k x genes sums of their rows by ``codes``) of
    an on-disk matrix read by row blocks (``OnDiskMatrix.row_blocks``). ``rows`` sorted
    ascending, ``codes`` aligned with them."""
    rows = np.asarray(rows, dtype=np.int64)
    order = np.argsort(rows, kind="stable")
    rows, codes = rows[order], np.asarray(codes)[order]
    genes = np.zeros(rows.size, dtype=np.float64)
    P = np.zeros((k, M.shape[1]), dtype=np.float64)
    for start, block in M.row_blocks():
        a, b = _rows_in(rows, start, start + block.shape[0])
        if a == b:
            continue
        sub = sp.csr_matrix(block)[rows[a:b] - start]
        genes[a:b] = np.asarray((sub > 0).sum(axis=1), dtype=np.float64).ravel()
        S = sp.csr_matrix((np.ones(b - a), (codes[a:b], np.arange(b - a))), shape=(k, b - a))
        P += np.asarray((S @ sub).toarray(), dtype=np.float64)
    out = np.empty_like(genes)
    out[order] = genes
    return out, P


def column_and_totals(M, j: int) -> tuple[np.ndarray, np.ndarray]:
    """(column ``j`` per cell, each row's total) of an on-disk matrix, by row blocks."""
    col = np.zeros(M.shape[0], dtype=np.float64)
    totals = np.zeros(M.shape[0], dtype=np.float64)
    for start, block in M.row_blocks():
        block = sp.csr_matrix(block)
        stop = start + block.shape[0]
        totals[start:stop] = np.asarray(block.sum(axis=1), dtype=np.float64).ravel()
        col[start:stop] = np.asarray(block[:, j].toarray(), dtype=np.float64).ravel()
    return col, totals


def gene_means(store) -> np.ndarray:
    """Each gene's mean over all cells, as scanpy's score_genes takes it from a CSR
    matrix: scipy sums a column as a float32 matrix-vector product, its values added
    in cell order (the same product here, on the store's gene-major blocks), then the
    sum is divided in float64. One pass over the store."""
    sums = np.zeros(store.n_genes, dtype=np.float64)
    ones = np.ones(store.n_cells, dtype=store.dtype)
    for g0, g1, block in store.blocks():
        sums[g0:g1] = np.asarray(sp.csr_matrix(block.T) @ ones, dtype=np.float64)
    return sums / store.n_cells


def _cell_means(store, positions: np.ndarray, rows: int = 200_000) -> np.ndarray:
    """Each cell's mean over the genes at ``positions``, as scanpy takes it from
    ``X[:, positions]``: scipy sums each CSR row (its values in the matrix's gene order)
    in the matrix's dtype, then divides in float64. Rows read in blocks of ``rows``."""
    positions = np.sort(np.asarray(positions, dtype=np.int64))
    out = np.zeros(store.n_cells, dtype=np.float64)
    for r0 in range(0, store.n_cells, rows):
        cells = np.arange(r0, min(r0 + rows, store.n_cells))
        block = sp.csr_matrix(store.genes(positions, cells=cells))
        block.sort_indices()
        out[cells] = np.asarray(block.sum(axis=1, dtype=np.float64)).ravel()
    return out / max(len(positions), 1)


def score_genes(store, var_names, gene_list, *, means: np.ndarray | None = None,
                ctrl_size: int = 50, n_bins: int = 25, random_state: int | None = 0,
                ctrl_as_ref: bool = True) -> tuple[np.ndarray, pd.Index, pd.Index]:
    """(score per cell, the set's genes used, the control genes): scanpy 1.11's
    ``score_genes`` (``tools/_score_genes.py``) on the matrix the store holds, with the
    genes' means given (``gene_means``) or computed. Raises ValueError as scanpy does."""
    var_names = pd.Index(var_names)
    if random_state is not None:
        np.random.seed(random_state)
    gene_list = pd.Index([gene_list] if isinstance(gene_list, str) else gene_list)
    gene_list = gene_list.intersection(var_names)
    if len(gene_list) == 0:
        raise ValueError("No valid genes were passed for scoring.")
    gene_pool = var_names.astype("string")
    if means is None:
        means = gene_means(store)
    obs_avg = pd.Series(means, index=gene_pool)
    obs_avg = obs_avg[np.isfinite(obs_avg)]
    n_items = int(np.round(len(obs_avg) / (n_bins - 1)))
    obs_cut = obs_avg.rank(method="min") // n_items
    keep_ctrl_in_obs_cut = False if ctrl_as_ref else obs_cut.index.isin(gene_list)
    control = pd.Index([], dtype="string")
    for cut in np.unique(obs_cut.loc[gene_list]):
        r_genes = obs_cut[(obs_cut == cut) & ~keep_ctrl_in_obs_cut].index
        if ctrl_size < len(r_genes):
            r_genes = r_genes.to_series().sample(ctrl_size).index
        if ctrl_as_ref:
            r_genes = r_genes.difference(gene_list)
        control = control.union(r_genes)
    if len(control) == 0:
        raise RuntimeError("No control genes found in any cut.")
    score = (_cell_means(store, var_names.get_indexer(gene_list))
             - _cell_means(store, var_names.get_indexer(control)))
    return score, gene_list, control
