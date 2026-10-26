"""ssGSEA: single-sample gene set enrichment (Barbie et al. 2009, Nature, PMID 19847166).

Within one cell the genes are ranked by expression. Walking down that ranking from the
highest-expressed gene, a running sum rises at each gene of the set, by the gene's rank
raised to ``alpha`` (0.25) as a share of the set's total, and falls at every other gene
by ``1 / (genes - set size)``. The cell's score is the sum of that running difference
over all positions: high when the set's genes sit near the top of the cell's ranking.

Computed as R's GSVA package does by default (``gsva(ssgseaParam(X, sets))``, GSVA 2.6;
agreement on Kang in the C097 report), implemented from the paper and GSVA's
documentation:

- every gene of the matrix is ranked, also genes no cell detects (they tie at zero);
- ranks run upwards (the highest-expressed gene has rank = number of genes), tied genes
  take their average rank, truncated to a whole number;
- the walk visits genes by that rank, highest first, tied genes in gene order;
- the score is divided by the range (max - min) of the scores, which GSVA takes over the
  whole result: for one set, over the cells (``normalize=TRUE``, the default).

The sum of the running difference has a closed form, ``sum_i w_i (n - pos_i + 1) /
sum_i w_i - (n (n + 1) / 2 - sum_i (n - pos_i + 1)) / (n - k)`` over the set's k genes at
positions pos_i of n (Barbie's walk summed by parts), so a cell costs one sort of its
detected genes: undetected genes all tie, and their ranks and positions follow from
the cell's detected genes. Cells are scored in blocks of rows of a sparse, dense or
backed matrix, which is never densified or read whole.

Only the order of genes within a cell counts, so ``rank_matrix`` is AUCell's: counts,
CPM or log values give the same scores.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp

ALPHA = 0.25  # Barbie et al. 2009; GSVA's ssgseaParam default
_ROWS = 2048  # cells per block

# Non-zero values one ssGSEA or GSVA request may score: about 150,000 cells at 2,000
# genes each. Both cost a sort of every cell's detected genes per set (about 1.7 s per
# 10 million values on 2 cores), and a score is one plain response, which the proxy
# closes after 300 s of silence: this keeps one set near a minute.
MAX_VALUES = 300_000_000


def csr_block(M, start: int, stop: int) -> sp.csr_matrix:
    """Rows start:stop as a canonical CSR copy (sorted indices, no stored zeros), from a
    sparse, dense or backed matrix."""
    B = M[start:stop]
    B = sp.csr_matrix(B, dtype=np.float64) if sp.issparse(B) else sp.csr_matrix(np.asarray(B, dtype=np.float64))
    B.eliminate_zeros()
    B.sort_indices()
    return B


def _row_ids(B: sp.csr_matrix) -> np.ndarray:
    return np.repeat(np.arange(B.shape[0], dtype=np.int64), np.diff(B.indptr))


def entry_ranks(B: sp.csr_matrix) -> tuple[np.ndarray, np.ndarray]:
    """For each stored entry of a canonical CSR block (non-negative values): its rank
    in its row over all ``B.shape[1]`` genes, upwards, ties averaged and truncated, and
    its position in the row's walk (1 = highest; ties in gene order)."""
    p = B.shape[1]
    nnz = B.nnz
    rows = _row_ids(B)
    v = B.data
    order = np.lexsort((B.indices, -v, rows))
    rs, vs = rows[order], v[order]
    pos = np.arange(nnz, dtype=np.int64) - B.indptr[rs] + 1  # zero genes come after
    start = np.ones(nnz, dtype=bool)
    if nnz:
        start[1:] = (rs[1:] != rs[:-1]) | (vs[1:] != vs[:-1])
    first = np.maximum.accumulate(np.where(start, np.arange(nnz), 0))
    end = np.ones(nnz, dtype=bool)
    if nnz:
        end[:-1] = start[1:]
    last = np.minimum.accumulate(np.where(end, np.arange(nnz), nnz)[::-1])[::-1]
    # Upward rank of the entry at walk position q is p - q + 1; a tie group spanning
    # positions qf..ql takes the average, truncated as GSVA stores ranks as integers.
    qf, ql = pos[first], pos[last]
    rank_sorted = np.floor(p + 1 - (qf + ql) / 2.0)
    rank = np.empty(nnz, dtype=np.float64)
    walk = np.empty(nnz, dtype=np.int64)
    rank[order] = rank_sorted
    walk[order] = pos
    return rank, walk


def _set_index(gene_idx, n_genes: int) -> np.ndarray:
    idx = np.unique(np.asarray(gene_idx, dtype=np.int64))
    if idx.size == 0:
        raise ValueError("ssGSEA needs at least one of the set's genes in the dataset.")
    if idx.size >= n_genes:
        raise ValueError("ssGSEA needs genes outside the set: the set holds every gene.")
    return idx


def raw_scores_block(B: sp.csr_matrix, S: np.ndarray, alpha: float = ALPHA) -> np.ndarray:
    """Un-normalised ssGSEA scores of the rows of a canonical CSR block for the set
    with sorted, unique gene indices ``S``."""
    n_rows, p = B.shape
    k = S.size
    rows = _row_ids(B)
    g = B.indices.astype(np.int64)
    rank, walk = entry_ranks(B)
    d = np.diff(B.indptr).astype(np.float64)  # detected genes per cell

    in_set = np.zeros(p, dtype=bool)
    in_set[S] = True
    m = in_set[g]
    rm = rows[m]
    w = rank[m] ** alpha
    c = (p - walk[m] + 1).astype(np.float64)  # n - pos + 1
    w_nz = np.bincount(rm, weights=w, minlength=n_rows)
    a_nz = np.bincount(rm, weights=w * c, minlength=n_rows)
    c_nz = np.bincount(rm, weights=c, minlength=n_rows)
    k_nz = np.bincount(rm, minlength=n_rows).astype(np.float64)
    g_nz = np.bincount(rm, weights=g[m].astype(np.float64), minlength=n_rows)

    # The set's undetected genes: they all tie, at the average of ranks 1..z (z genes
    # undetected in the cell), and walk after the detected genes in gene order, so the
    # undetected gene g sits at d + 1 + g - #(detected genes before g).
    offset = (np.arange(B.nnz, dtype=np.int64) - B.indptr[rows]).astype(np.float64)
    before_nz = np.bincount(rm, weights=offset[m], minlength=n_rows)  # detected set genes
    set_after = (k - np.searchsorted(S, g, side="right")).astype(np.float64)
    pairs = np.bincount(rows, weights=set_after, minlength=n_rows)
    detected_before = pairs - before_nz  # summed over the set's undetected genes
    k_z = k - k_nz
    g_z = float(S.sum()) - g_nz
    pos_z = k_z * (d + 1) + g_z - detected_before
    c_z = k_z * (p + 1) - pos_z
    z = p - d
    w0 = np.floor((z + 1) / 2.0) ** alpha

    walk_in = (a_nz + w0 * c_z) / (w_nz + w0 * k_z)
    walk_out = (p * (p + 1) / 2.0 - (c_nz + c_z)) / (p - k)
    return walk_in - walk_out


def normalise(es: np.ndarray) -> np.ndarray:
    """Divide by the range of the scores, as GSVA's ``normalize=TRUE`` does."""
    if es.size == 0:
        return es
    span = float(np.max(es) - np.min(es))
    return es / span if span > 0 else es


def score(M, gene_idx, alpha: float = ALPHA, normalize: bool = True,
          block_rows: int = _ROWS) -> np.ndarray:
    """Every cell's ssGSEA score for the set with these gene indices into the columns
    of ``M`` (cells x genes, non-negative; sparse, dense or backed)."""
    n_cells, n_genes = M.shape
    S = _set_index(gene_idx, n_genes)
    out = np.zeros(n_cells, dtype=np.float64)
    for start in range(0, n_cells, block_rows):
        stop = min(start + block_rows, n_cells)
        out[start:stop] = raw_scores_block(csr_block(M, start, stop), S, alpha)
    return normalise(out) if normalize else out


# ---------------------------------------------------------------------------
# One set on a dataset
# ---------------------------------------------------------------------------


def stored_values(M, block_rows: int = 4096) -> int:
    """Non-zero values of a sparse, dense or backed matrix (dense: counted in blocks)."""
    if sp.issparse(M):
        return int(M.nnz)
    n = 0
    for start in range(0, M.shape[0], block_rows):
        B = M[start:start + block_rows]
        n += int(B.nnz) if sp.issparse(B) else int(np.count_nonzero(np.asarray(B)))
    return n


def check_size(M, method: str) -> None:
    """Refuse (ValueError) a matrix above ``MAX_VALUES``."""
    n = stored_values(M)
    if n > MAX_VALUES:
        raise ValueError(
            f"{method} scores up to {MAX_VALUES // 1_000_000:,} million non-zero values in one "
            f"request (about 150,000 cells at 2,000 genes each); this dataset has "
            f"{n / 1e6:,.0f} million. Score with AUCell or score_genes instead."
        )


def match(genes, gene_set: list[str], usable: np.ndarray | None = None):
    """The set's genes as column indices (``aucell.SetMatch``): each name exactly, else
    ignoring case. ``usable``: per column, whether the method can rank it (else the gene
    goes to ``undetected``)."""
    from scview.core.aucell import SetMatch

    names = [str(g) for g in genes]
    exact = {g: i for i, g in enumerate(names)}
    upper: dict[str, int] = {}
    for i, g in enumerate(names):
        upper.setdefault(g.upper(), i)
    out = SetMatch([], [], [], [])
    seen: set[int] = set()
    for g in gene_set:
        i = exact.get(g)
        if i is None:
            i = upper.get(str(g).upper())
        if i is None:
            out.missing.append(g)
        elif i not in seen:
            seen.add(i)
            if usable is None or usable[i]:
                out.idx.append(i)
                out.found.append(names[i])
            else:
                out.undetected.append(names[i])
    return out


@dataclass
class SetScores:
    """A set scored on every cell, with what was used and how."""

    scores: np.ndarray
    match: object  # aucell.SetMatch
    source: str
    note: str


def score_set(adata, gene_set: list[str]) -> SetScores:
    """ssGSEA for one set on a loaded dataset (AUCell's ``rank_matrix``: counts or log
    values rank alike). Every gene of the matrix is ranked, as GSVA does. Call in a
    worker thread; ValueError when nothing can be scored."""
    from scview.core.aucell import rank_matrix

    M, genes, source = rank_matrix(adata)
    check_size(M, "ssGSEA")
    m = match(pd.Index(genes), gene_set)
    if not m.idx:
        raise ValueError(f"None of the {len(gene_set)} provided genes were found in the dataset.")
    scores = score(M, m.idx)
    note = (f"ssGSEA over all {M.shape[1]:,} genes of the {source}, alpha {ALPHA}, "
            "divided by the range of the scores")
    return SetScores(scores=scores, match=m, source=source, note=note)
