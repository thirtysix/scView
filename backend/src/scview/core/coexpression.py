"""Genes co-expressed with a query gene across cells (Pearson, or Spearman).

Correlation over cells of the log-normalised expression the overlays show
(``.raw`` when the dataset has it, else X; counts are log-normalised first, as
for DE), within all cells or a selection / group. Neither the whole matrix nor
the selected rows are ever made dense:

- Pearson: sum_i (x_i - mean_x)(q_i - mean_q) = sum_i x_i (q_i - mean_q), as the
  centred query sums to 0, so one sparse matrix-vector product gives every
  gene's covariance with the query, and the variances come from each gene's sum
  and sum of squares. A dense (or on-disk) matrix is read in gene chunks.
- Spearman: Pearson of ranks. For non-negative sparse values every zero of a
  gene ties at the same average rank, so shifting each gene's ranks by that
  rank leaves the zeros at 0 and the matrix as sparse as before; Pearson does
  not see the shift.

A gene needs at least ``max(MIN_CELLS, MIN_FRACTION of the cells)`` expressing
(> 0) cells among the chosen ones, or its correlation rests on a handful of
cells; a gene with zero variance there has none at all. Both are skipped and
counted, never returned as NaN. Scaled values (z-scores, residuals) have no
zero for "not detected", so the expressing-cell minimum is not applied to them.

Kang 2018 PBMCs, ISG15 over all 13,836 cells (.raw, 14,053 genes; 7,380 pass
the 139-cell minimum), Pearson: IFIT3 0.85, IFIT1 0.84, IFI6 0.82, LY6E 0.82,
ISG20 0.81, MX1 0.78, the ribosomal genes most negative (RPL7 -0.45); within
CD14 Mono alone the same interferon genes at 0.88-0.93, IL8 -0.66 and CD14
-0.57 most negative. 1.8 s on 2 cores. paul15 (dense, 2,730 cells): Gata1 with
Ermap 0.55, Klf1 0.54, Rhd 0.52; Mpo -0.46.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import scipy.sparse as sp
from scipy import stats

Method = Literal["pearson", "spearman"]

MIN_CELLS = 10
MIN_FRACTION = 0.01
TOP_N = 25
MAX_TOP_N = 200
GENE_CHUNK = 2000  # genes per block of a dense or on-disk matrix
# Cells one correlation uses: more chosen cells are sampled to this many (fixed
# seed, so a request gives the same answer each time; ``capped`` says so). The
# work copies the chosen rows of the whole matrix several times (float64, the
# squares, the log-normalised counts), so memory grows with the cells: ovary
# (37,104 cells, raw counts, 2 cores) took +1.49 GB and 10.9 s over all cells,
# +1.03 GB and 8.0 s over 20,000 (top r within 0.003), and a 100k-cell dataset
# would take ~4 GB. r's standard error at 20,000 cells is under 0.01. Kang
# (13,836 cells) is below the cap.
MAX_CELLS = 20_000
SAMPLE_SEED = 0


class CoexpressionError(ValueError):
    """A request this cannot answer (unknown gene, too few cells): a 4xx."""


@dataclass
class _Source:
    M: object  # cells x genes
    names: list[str]
    label: str  # "raw" or "X"
    from_counts: bool  # counts, log-normalised here


def _source(adata) -> _Source:
    from scview.core.pipeline import looks_like_counts

    if adata.raw is not None:
        M, names, label = adata.raw.X, adata.raw.var_names, "raw"
    else:
        M, names, label = adata.X, adata.var_names, "X"
    sample = M[: min(200, M.shape[0])] if not sp.issparse(M) else M
    return _Source(M, [str(n) for n in names], label, bool(looks_like_counts(sample)))


def _rows_block(M, rows: np.ndarray | None, start: int, stop: int):
    """Columns start:stop of the chosen rows, sparse kept sparse, as float64."""
    if rows is not None and isinstance(M, np.ndarray):
        block = M[rows, start:stop]
    else:  # sparse or on disk (h5py / backed): the columns first, then the rows
        block = M[:, start:stop]
        if rows is not None:
            block = block[rows]
    if sp.issparse(block):
        return sp.csr_matrix(block, dtype=np.float64)
    return np.asarray(block, dtype=np.float64)


def _blocks(M, rows: np.ndarray | None):
    """(start, block) over the genes: one block for an in-memory sparse matrix
    (its rows picked once), gene chunks otherwise."""
    n_genes = M.shape[1]
    if sp.issparse(M):
        sub = M if rows is None else M[rows]
        yield 0, sp.csr_matrix(sub, dtype=np.float64)
        return
    for start in range(0, n_genes, GENE_CHUNK):
        yield start, _rows_block(M, rows, start, min(start + GENE_CHUNK, n_genes))


def _row_totals(M, rows: np.ndarray | None) -> np.ndarray:
    total = None
    for _, b in _blocks(M, rows):
        s = np.asarray(b.sum(axis=1), dtype=np.float64).ravel()
        total = s if total is None else total + s
    return total


def _log_normalise(block, scale: np.ndarray | None):
    """log1p(counts per 10k) of a block, given each row's 1e4 / total."""
    if scale is None:
        return block
    if sp.issparse(block):
        out = sp.csr_matrix(block.multiply(scale[:, None]))
        out.data = np.log1p(out.data)
        return out
    return np.log1p(block * scale[:, None])


def _ranks_sparse(block: sp.csr_matrix) -> sp.csr_matrix:
    """Each column's ranks over the rows, shifted so the tied zeros sit at 0
    (values must be >= 0; explicit zeros are dropped first)."""
    C = sp.csc_matrix(block)
    C.eliminate_zeros()
    n = C.shape[0]
    data = C.data.copy()
    for j in range(C.shape[1]):
        a, b = C.indptr[j], C.indptr[j + 1]
        nz = b - a
        if nz == 0:
            continue
        n_zero = n - nz
        zero_rank = (n_zero + 1) / 2.0
        data[a:b] = n_zero + stats.rankdata(C.data[a:b]) - zero_rank
    return sp.csr_matrix(sp.csc_matrix((data, C.indices, C.indptr), shape=C.shape))


def _col_stats(block):
    """Per gene: sum, sum of squares, and cells above zero."""
    if sp.issparse(block):
        s = np.asarray(block.sum(axis=0)).ravel()
        sq = block.copy()
        sq.data **= 2
        ss = np.asarray(sq.sum(axis=0)).ravel()
        pos = np.asarray((block > 0).sum(axis=0)).ravel()
    else:
        s = block.sum(axis=0)
        ss = (block * block).sum(axis=0)
        pos = (block > 0).sum(axis=0)
    return s, ss, pos.astype(np.int64)


def coexpression(
    adata,
    gene: str,
    *,
    rows: np.ndarray | None = None,
    method: Method = "pearson",
    top_n: int = TOP_N,
) -> dict:
    """The genes most positively and most negatively correlated with ``gene``
    over the ``rows`` cells (every cell when None)."""
    src = _source(adata)
    try:
        q_idx = src.names.index(gene)
    except ValueError:
        raise CoexpressionError(f"Gene '{gene}' not found.") from None
    n_total = int(src.M.shape[0])
    if rows is not None:
        rows = np.unique(np.asarray(rows, dtype=np.int64))
        rows = rows[(rows >= 0) & (rows < n_total)]
        if rows.size == n_total:
            rows = None
    n_chosen = n_total if rows is None else int(rows.size)
    capped = n_chosen > MAX_CELLS
    if capped:
        pool = np.arange(n_total) if rows is None else rows
        rows = np.sort(np.random.default_rng(SAMPLE_SEED).choice(pool, MAX_CELLS, replace=False))
    n = n_total if rows is None else int(rows.size)
    if hasattr(src.M, "row_blocks"):
        # A large dataset's X on disk (core/large_mode.py), which gives rows, not genes:
        # the chosen cells (at most MAX_CELLS) are read once and correlated in memory.
        src.M = sp.csr_matrix(src.M[np.arange(n_total) if rows is None else rows])
        rows = None
    min_expressing = max(MIN_CELLS, math.ceil(MIN_FRACTION * n))
    if n < max(MIN_CELLS, 3):
        raise CoexpressionError(f"Pick at least {MIN_CELLS} cells (got {n}).")

    scale = None
    if src.from_counts:
        totals = _row_totals(src.M, rows)
        scale = np.divide(1e4, totals, out=np.zeros_like(totals), where=totals > 0)

    # The query's values over the chosen cells.
    q = _log_normalise(_rows_block(src.M, rows, q_idx, q_idx + 1), scale)
    q = (q.toarray() if sp.issparse(q) else q).ravel()
    scaled = bool(np.nanmin(q) < 0)
    q_expr = int((q > 0).sum())
    # The centred query, on the scale the genes are correlated on (ranks for
    # Spearman; the shift of the sparse ranks does not change a correlation).
    qq = stats.rankdata(q) if method == "spearman" else q
    qc = qq - qq.mean()
    q_ss = float(qc @ qc)

    n_genes = len(src.names)
    r = np.full(n_genes, np.nan)
    n_pos = np.zeros(n_genes, dtype=np.int64)
    var_ok = np.zeros(n_genes, dtype=bool)
    for start, block in _blocks(src.M, rows):
        block = _log_normalise(block, scale)
        stop = start + block.shape[1]
        _, _, pos = _col_stats(block)  # expressing cells: on values, not ranks
        n_pos[start:stop] = pos
        if sp.issparse(block):
            has_neg = bool(block.data.size) and bool(block.data.min() < 0)
        else:
            has_neg = bool(block.size) and bool(block.min() < 0)
        scaled = scaled or has_neg
        if method == "spearman":
            if sp.issparse(block) and not has_neg:
                block = _ranks_sparse(block)
            else:  # a dense chunk (or signed sparse values): rank it as it is
                block = stats.rankdata(block.toarray() if sp.issparse(block) else block, axis=0)
        s, ss, _ = _col_stats(block)
        var = ss - s * s / n  # n x the variance
        var_ok[start:stop] = var > 1e-12 * np.maximum(ss, 1.0)
        num = np.asarray(block.T @ qc).ravel()  # n x the covariance
        denom = np.sqrt(np.maximum(var, 0.0) * q_ss)
        with np.errstate(invalid="ignore", divide="ignore"):
            r[start:stop] = np.where(denom > 0, num / denom, np.nan)

    if not scaled and q_expr < min_expressing:
        raise CoexpressionError(
            f"{gene} is expressed in {q_expr} of these {n} cells; it needs at least "
            f"{min_expressing} for a correlation."
        )
    if not var_ok[q_idx]:
        raise CoexpressionError(f"{gene} does not vary across these {n} cells.")

    rare = np.zeros(n_genes, dtype=bool) if scaled else n_pos < min_expressing
    constant = ~var_ok & ~rare
    keep = ~rare & ~constant & np.isfinite(r)
    keep[q_idx] = False
    r = np.clip(r, -1.0, 1.0)

    idx = np.flatnonzero(keep)
    names = np.asarray(src.names, dtype=object)

    def rows_of(order: np.ndarray) -> list[dict]:
        return [{"gene": str(names[i]), "r": round(float(r[i]), 4),
                 "n_expressing": int(n_pos[i]), "pct_expressing": round(100.0 * n_pos[i] / n, 2)}
                for i in order]

    by_desc = idx[np.lexsort((names[idx], -r[idx]))]
    by_asc = idx[np.lexsort((names[idx], r[idx]))]
    positive = by_desc[r[by_desc] > 0][:top_n]
    negative = by_asc[r[by_asc] < 0][:top_n]
    return {
        "gene": gene,
        "method": method,
        "n_cells": n,
        "n_cells_total": n_total,
        # The cells asked for; more than MAX_CELLS are sampled (capped).
        "n_cells_chosen": n_chosen,
        "capped": capped,
        "query_n_expressing": q_expr,
        "min_expressing": None if scaled else min_expressing,
        "n_genes_tested": int(keep.sum()),
        "n_skipped_rare": int(rare.sum() - (1 if rare[q_idx] else 0)),
        "n_skipped_constant": int(constant.sum() - (1 if constant[q_idx] else 0)),
        "layer": src.label,
        "values": "scaled" if scaled else ("log-normalised from counts" if src.from_counts else "log-normalised"),
        "positive": rows_of(positive),
        "negative": rows_of(negative),
    }
