"""GSVA: gene set variation analysis (Hänzelmann et al. 2013, BMC Bioinformatics, PMID 23323831).

Where ssGSEA and AUCell rank genes by their level in the cell, GSVA ranks them by how
high the cell sits in each gene's own distribution over all cells, so a gene that is
always high does not dominate every cell. Per gene, each cell's value is turned into
the gene's cumulative distribution at that value (F, 0 to 1). Within each cell, genes
are then ranked by F; from the ranks r (1..p, upwards) each gene gets the weight
|p/2 - r|, and a running sum is walked down the ranking from the top: up at the set's
genes (their weight as a share of the set's total), down by 1/(p - k) at every other
gene. The score is the largest positive plus the largest negative deviation of that
walk (``maxDiff=TRUE``): about +1 when the set's genes are at the top of the cell's
ranking, about -1 at the bottom.

Computed as R's GSVA package does by default on a matrix of log expression
(``gsva(gsvaParam(X, sets))``, GSVA 2.6, its classical algorithm; agreement on Kang in
the C097 report), implemented from the paper and GSVA's documentation:

- genes with one value in every cell (never detected, or the same value everywhere)
  are left out, as ``filterRows=TRUE`` does on a dense matrix;
- F: below 200 cells a Gaussian kernel estimate, bandwidth sd/4 (the paper's), else the
  empirical CDF directly, the fraction of cells at or below the value (``kcdf="auto"``
  with ``kcdfNoneMinSampleSize=200``). The kernel costs cells^2 per gene, the
  empirical CDF a sort;
- genes tied on F within a cell rank in reverse gene order (``ties.method="last"``).

Given a sparse dgCMatrix, R's GSVA runs differently: by default a sparse variant
(``sparse=TRUE``: CDFs of the non-zero values only, zeros kept at the bottom), and
with either setting it picks the kernel by non-zeros per gene rather than by cells.
Its scores are then not these (Spearman median 0.72 over the Hallmark sets on Kang,
0.66 with ``sparse=FALSE``). Pass ``as.matrix(x)`` in R to compare.

Large data: a gene's F is its share of cells at zero (counted over all cells, exact)
plus its share of detected cells times the empirical CDF of its non-zero values, which
come from a reference of at most ``REF_CELLS`` cells (all of them up to that; above, a
fixed random sample). So the result is R's exactly up to 20,000 cells; above, only the
order among a cell's detected genes is estimated (by the Dvoretzky-Kiefer-Wolfowitz
inequality each gene's non-zero CDF is within sqrt(ln(40) / (2 m)) of the full one at
95% confidence, m its detected reference cells), and ``test_gsva`` measures the
agreement with the full reference. Cells are scored in row blocks of a sparse, dense
or backed matrix, never densified or read whole. In a cell, an undetected gene sits at
its gene's share of zeros, so each cell needs only a sort of its detected genes.

GSVA compares a gene across cells, so it needs depth-normalised values: the
log-normalised ``.raw`` or X, else counts normalised to 10,000 per cell and log1p,
block by block (``gsva_matrix``).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import special

from scview.core.object_cache import ObjectCache
from scview.core.ssgsea import csr_block

KCDF_MIN_CELLS = 200  # GSVA's kcdfNoneMinSampleSize: the kernel below, the ECDF from here
REF_CELLS = 20_000  # cells in the ECDF's reference: exact up to this many cells
_STEPS = 1024  # sampled reference: levels per cell step, so estimated levels rarely tie
TAU = 1.0  # the walk's weight exponent (the paper's)
TARGET_SUM = 1e4  # counts are normalised to this per cell, then log1p
_ROWS = 2048  # cells per block
_GENE_BLOCK = 256  # genes per block of the kernel estimate


# ---------------------------------------------------------------------------
# What to score
# ---------------------------------------------------------------------------


class _LogNormalised:
    """Counts read a block of rows at a time as log1p(counts / total * 1e4), so a
    counts-only dataset is never normalised (or copied) whole."""

    def __init__(self, counts):
        self.counts = counts
        self.shape = counts.shape

    def __getitem__(self, rows: slice):
        B = self.counts[rows]
        B = sp.csr_matrix(B, dtype=np.float64) if sp.issparse(B) else sp.csr_matrix(np.asarray(B, dtype=np.float64))
        totals = np.asarray(B.sum(axis=1)).ravel()
        scale = np.divide(TARGET_SUM, totals, out=np.zeros_like(totals), where=totals > 0)
        B = sp.csr_matrix(sp.diags(scale) @ B)
        B.data = np.log1p(B.data)
        return B


def gsva_matrix(adata):
    """(matrix, gene names, where from): log-normalised ``.raw``, else log-normalised
    X, else the counts normalised block by block. Never z-scores or residuals."""
    from scview.core.aucell import _block, _non_negative
    from scview.core.normalization import find_counts
    from scview.core.pipeline import looks_like_counts

    def lognorm(M) -> bool:
        return _non_negative(M) and not looks_like_counts(_block(M, 0, min(1000, M.shape[0])))

    if adata.raw is not None and lognorm(adata.raw.X):
        return adata.raw.X, adata.raw.var_names, "log-normalised .raw"
    if lognorm(adata.X):
        return adata.X, adata.var_names, "log-normalised X"
    try:
        counts, source = find_counts(adata)
    except ValueError:
        raise ValueError(
            "GSVA needs log-normalised expression or raw counts, and this dataset holds "
            "only scaled values."
        ) from None
    return _LogNormalised(counts), adata.var_names, f"counts in {source}, log-normalised (10,000 per cell)"


# ---------------------------------------------------------------------------
# Per-gene distributions
# ---------------------------------------------------------------------------


@dataclass
class Prepared:
    """What a dataset's GSVA needs before any set: kept genes and each gene's
    distribution (its reference values)."""

    n_cells: int
    keep: np.ndarray  # bool per column of the matrix: genes GSVA ranks
    source: str
    kernel: bool  # Gaussian kernel (fewer than KCDF_MIN_CELLS cells), else ECDF
    n_ref: int  # cells in the ECDF's reference
    ref_keys: np.ndarray | None = None  # sorted gene * stride + value, reference non-zeros
    ref_ptr: np.ndarray | None = None  # per column: first index of its keys
    nnz: np.ndarray | None = None  # per column: cells with a non-zero value (all cells)
    nnz_ref: np.ndarray | None = None  # per column: the same in the reference
    stride: float = 1.0
    F: np.ndarray | None = None  # kernel: cells x kept genes, the CDF values

    @property
    def p(self) -> int:
        return int(self.keep.sum())

    @property
    def sampled(self) -> bool:
        return self.n_ref < self.n_cells

    @property
    def steps(self) -> int:
        """Levels per cell: 1 with the full reference (levels are cell counts, ties as
        in R), finer with a sampled one."""
        return _STEPS if self.sampled else 1

    def levels(self, genes: np.ndarray, below: np.ndarray) -> np.ndarray:
        """Integer levels (cells at or below, times ``steps``) of detected values with
        ``below`` reference non-zeros at or under them."""
        zeros = (self.n_cells - self.nnz[genes]) * self.steps
        if not self.sampled:
            return zeros + below
        nnz, ref = self.nnz[genes], self.nnz_ref[genes]
        est = np.where(ref > 0, (nnz * below * _STEPS) // np.maximum(ref, 1), (nnz * _STEPS) // 2)
        return zeros + est

    def zero_levels(self) -> np.ndarray:
        """Per kept gene: the level of an undetected value."""
        return (self.n_cells - self.nnz[self.keep]) * self.steps


def gene_stats(M, block_rows: int = _ROWS):
    """Per gene: cells with a non-zero value, the smallest and largest of those values,
    and whether any value is negative (in row blocks)."""
    n_genes = M.shape[1]
    nnz = np.zeros(n_genes, dtype=np.int64)
    lo = np.full(n_genes, np.inf)
    hi = np.full(n_genes, -np.inf)
    negative = False
    for start in range(0, M.shape[0], block_rows):
        B = csr_block(M, start, min(start + block_rows, M.shape[0]))
        if B.nnz == 0:
            continue
        negative = negative or bool(B.data.min() < 0)
        nnz += np.bincount(B.indices, minlength=n_genes)
        np.minimum.at(lo, B.indices, B.data)
        np.maximum.at(hi, B.indices, B.data)
    return nnz, lo, hi, negative


def reference_cells(n_cells: int, ref_cells: int = REF_CELLS) -> np.ndarray:
    """All cells up to ``ref_cells``, else a fixed random sample of that many (seed 0)."""
    if n_cells <= ref_cells:
        return np.arange(n_cells)
    return np.sort(np.random.default_rng(0).choice(n_cells, ref_cells, replace=False))


def _rows(M, idx: np.ndarray, block_rows: int = _ROWS) -> sp.csr_matrix:
    """The rows ``idx`` (sorted) as one canonical CSR, read in blocks."""
    parts = []
    for start in range(0, M.shape[0], block_rows):
        stop = min(start + block_rows, M.shape[0])
        sel = idx[(idx >= start) & (idx < stop)] - start
        if sel.size:
            parts.append(csr_block(M, start, stop)[sel])
    return sp.vstack(parts, format="csr") if parts else sp.csr_matrix((0, M.shape[1]))


def _search(arrays: list[np.ndarray], queries: np.ndarray, side: str = "left") -> list[np.ndarray]:
    """``np.searchsorted`` of the queries in each sorted array, the queries sorted
    once first: the same results, several times faster on a large array (neighbouring
    queries share cache lines)."""
    flat = queries.ravel()
    order = np.argsort(flat)
    ordered = flat[order]
    out = []
    for arr in arrays:
        found = np.empty(flat.size, dtype=np.int64)
        found[order] = np.searchsorted(arr, ordered, side=side)
        out.append(found.reshape(queries.shape))
    return out


def _keys(genes: np.ndarray, values: np.ndarray, stride: float) -> np.ndarray:
    # Equal (gene, value) pairs give equal keys, and keys keep the order of values
    # within a gene: one sorted array serves every gene's searchsorted.
    return genes.astype(np.float64) * stride + values


def kernel_cdf(X: np.ndarray, gene_block: int = _GENE_BLOCK) -> np.ndarray:
    """Gaussian kernel CDF per gene (column): F[c, g] = mean_j Phi((x_cg - x_jg) / h_g),
    h_g = sd_g / 4 (GSVA's kernel, ``kcdf="Gaussian"``). Dense: for few cells."""
    X = np.asarray(X, dtype=np.float64)
    n, p = X.shape
    F = np.empty((n, p))
    h = X.std(axis=0, ddof=1) / 4.0 if n > 1 else np.ones(p)
    h[~(h > 0)] = 1.0  # constant genes are left out anyway
    for a in range(0, p, gene_block):
        b = min(a + gene_block, p)
        D = (X[:, None, a:b] - X[None, :, a:b]) / h[None, None, a:b]
        F[:, a:b] = special.ndtr(D).mean(axis=1)
    return F


def prepare(M, source: str = "", ref_cells: int = REF_CELLS,
            kcdf_min_cells: int = KCDF_MIN_CELLS) -> Prepared:
    """Kept genes and each gene's distribution, from one pass over ``M``."""
    n_cells, n_genes = M.shape
    if n_cells < 2:
        raise ValueError("GSVA needs at least 2 cells: it compares each gene across cells.")
    nnz, lo, hi, negative = gene_stats(M)
    # filterRows: genes with one value in every cell (all zero, or one value everywhere).
    keep = (nnz > 0) & ~((nnz == n_cells) & (hi == lo))
    if keep.sum() < 2:
        raise ValueError("GSVA needs at least 2 genes that vary between cells.")
    if n_cells < kcdf_min_cells:
        dense = csr_block(M, 0, n_cells).toarray()[:, keep]
        return Prepared(n_cells=n_cells, keep=keep, source=source, kernel=True,
                        n_ref=n_cells, F=kernel_cdf(dense))
    if negative:
        raise ValueError("GSVA here needs non-negative log expression (or counts).")
    ref = reference_cells(n_cells, ref_cells)
    R = _rows(M, ref)
    stride = float(np.max(hi[np.isfinite(hi)], initial=0.0)) + 1.0
    g = R.indices.astype(np.int64)
    ref_keys = np.sort(_keys(g, R.data, stride))
    counts = np.bincount(g, minlength=n_genes).astype(np.int64)
    ref_ptr = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)
    return Prepared(n_cells=n_cells, keep=keep, source=source, kernel=False, n_ref=int(ref.size),
                    ref_keys=ref_keys, ref_ptr=ref_ptr, nnz=nnz, nnz_ref=counts, stride=stride)


# ---------------------------------------------------------------------------
# Scores
# ---------------------------------------------------------------------------


def walk_scores(pos: np.ndarray, weight: np.ndarray, p: int, max_diff: bool = True) -> np.ndarray:
    """GSVA's random-walk statistic per row: the set's genes at walk positions ``pos``
    (1 = top, rows x k) with ``weight`` (|p/2 - r|^tau). maxDiff: largest positive plus
    largest negative deviation; else the deviation furthest from 0. A row whose set
    weights are all 0 (one gene at the middle rank) has no walk: NaN, as in GSVA."""
    k = pos.shape[1]
    order = np.argsort(pos, axis=1, kind="stable")
    q = np.take_along_axis(pos, order, axis=1).astype(np.float64)
    w = np.take_along_axis(weight, order, axis=1)
    cw = np.cumsum(w, axis=1)
    total = cw[:, -1:]
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = cw / total
        prev = np.concatenate([np.zeros((pos.shape[0], 1)), frac[:, :-1]], axis=1)
    out_steps = (q - np.arange(1, k + 1)) / (p - k)
    top = np.maximum((frac - out_steps).max(axis=1), 0.0)
    bottom = np.minimum((prev - out_steps).min(axis=1), 0.0)
    es = top + bottom if max_diff else np.where(top > -bottom, top, bottom)
    es[(total[:, 0] <= 0) | ~np.isfinite(total[:, 0])] = np.nan
    return es


def _set_columns(prep: Prepared, gene_idx) -> np.ndarray:
    """The set's genes as indices into the kept genes (sorted)."""
    kept_pos = np.full(prep.keep.size, -1, dtype=np.int64)
    kept_pos[prep.keep] = np.arange(prep.p)
    s = kept_pos[np.unique(np.asarray(gene_idx, dtype=np.int64))]
    s = s[s >= 0]
    if s.size == 0:
        raise ValueError("None of the set's genes varies between cells, so GSVA cannot rank them.")
    if s.size >= prep.p:
        raise ValueError("GSVA needs genes outside the set: the set holds every ranked gene.")
    return s


def _ranks_kernel(F: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Upward ranks (1..p) of the set's genes in each row of F, ties in reverse gene
    order (``ties.method="last"``)."""
    p = F.shape[1]
    rev = F[:, ::-1]
    order = np.argsort(rev, axis=1, kind="stable")  # ties: the later gene first
    ranks = np.empty_like(order)
    np.put_along_axis(ranks, order, np.arange(1, p + 1)[None, :].repeat(F.shape[0], 0), axis=1)
    ranks = ranks[:, ::-1]
    return ranks[:, S]


def _ranks_ecdf_block(prep: Prepared, B: sp.csr_matrix, S: np.ndarray) -> np.ndarray:
    """Upward ranks of the set's (kept) genes in each row of a block, by the ECDF.

    A gene's level in a cell is the count of cells at or below its value (``levels``):
    its zeros for an undetected gene. Genes are ordered by (level, reverse gene index) through one integer key, unique per gene, so a rank is the number of keys
    below plus one: keys of the cell's undetected genes = all genes' zero-level keys
    minus those of its detected genes."""
    keep = prep.keep
    p = prep.p
    kept_pos = np.full(keep.size, -1, dtype=np.int64)
    kept_pos[keep] = np.arange(p)
    n_rows = B.shape[0]
    rows = np.repeat(np.arange(n_rows, dtype=np.int64), np.diff(B.indptr))
    g = B.indices.astype(np.int64)
    m = keep[g]
    rows, g, v = rows[m], g[m], B.data[m]
    kg = kept_pos[g]
    below = _search([prep.ref_keys], _keys(g, v, prep.stride), side="right")[0] - prep.ref_ptr[g]
    level = prep.levels(g, below)
    zero_level = prep.zero_levels()
    span = np.int64(prep.n_cells * prep.steps + 1) * p  # keys run 0 .. span - 1
    tie = p - 1 - np.arange(p, dtype=np.int64)  # reverse gene order breaks ties
    key0_all = np.sort(zero_level * p + tie)
    key0 = rows * span + zero_level[kg] * p + tie[kg]
    key1 = rows * span + level * p + tie[kg]
    key0.sort()
    key1.sort()

    # The set's genes in each row: their level (detected) or zero level.
    q = np.repeat((zero_level[S] * p + tie[S])[None, :], n_rows, axis=0)
    in_set = np.full(p, -1, dtype=np.int64)
    in_set[S] = np.arange(S.size)
    hit = in_set[kg] >= 0
    q[rows[hit], in_set[kg[hit]]] = level[hit] * p + tie[kg[hit]]
    row_start = (np.arange(n_rows, dtype=np.int64) * span)[:, None]
    shifted = q + row_start
    # keys below: all genes at their zero level, less the cell's detected genes at their
    # zero level, plus the detected genes at their level (within the row).
    in0, in1 = _search([key0, key1], shifted)
    r = (_search([key0_all], q)[0]
         - in0 + np.searchsorted(key0, row_start)
         + in1 - np.searchsorted(key1, row_start))
    return r + 1


def score(prep: Prepared, M, gene_idx, tau: float = TAU, max_diff: bool = True,
          block_rows: int = _ROWS) -> np.ndarray:
    """Every cell's GSVA score for the set with these gene indices (columns of ``M``;
    genes left out by the filter are dropped). NaN where the walk is undefined."""
    S = _set_columns(prep, gene_idx)
    p = prep.p
    out = np.empty(prep.n_cells, dtype=np.float64)
    for start in range(0, prep.n_cells, block_rows):
        stop = min(start + block_rows, prep.n_cells)
        if prep.kernel:
            r = _ranks_kernel(prep.F[start:stop], S)
        else:
            r = _ranks_ecdf_block(prep, csr_block(M, start, stop), S)
        pos = p - r + 1
        weight = np.abs(p / 2.0 - r) ** tau
        out[start:stop] = walk_scores(pos, weight, p, max_diff)
    return out


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

# loaded dataset object -> (matrix, genes, Prepared); a pipeline run drops the object.
_CACHE = ObjectCache()
_BUILD = threading.Lock()


def prepared(adata) -> tuple[object, pd.Index, Prepared]:
    """This loaded dataset's matrix, gene names and GSVA preparation, built on first
    use. Call in a worker thread."""
    with _BUILD:
        entry = _CACHE.get(adata)
        if entry is None:
            M, genes, source = gsva_matrix(adata)
            entry = (M, pd.Index(np.asarray(genes, dtype=str)), prepare(M, source))
            _CACHE.put(adata, entry)
        return entry


def score_set(adata, gene_set: list[str]):
    """GSVA for one set on a loaded dataset (``ssgsea.SetScores``). Genes left out by
    the filter are reported as undetected; a cell whose walk is undefined scores 0.
    Call in a worker thread; ValueError when nothing can be scored."""
    from scview.core import ssgsea

    entry = _CACHE.get(adata)  # the size check comes before a preparation is built
    ssgsea.check_size(entry[0] if entry is not None else gsva_matrix(adata)[0], "GSVA")
    M, genes, prep = prepared(adata)
    m = ssgsea.match(genes, gene_set, usable=prep.keep)
    if not m.idx:
        if m.undetected:
            raise ValueError(
                f"None of the set's genes varies between cells ({len(m.undetected)} found but "
                "never detected, or the same in every cell), so GSVA cannot rank them."
            )
        raise ValueError(f"None of the {len(gene_set)} provided genes were found in the dataset.")
    scores = score(prep, M, m.idx)
    undefined = int(np.isnan(scores).sum())
    scores = np.nan_to_num(scores, nan=0.0)
    if prep.kernel:
        how = f"Gaussian kernel CDF over the {prep.n_cells:,} cells (fewer than {KCDF_MIN_CELLS})"
    elif prep.sampled:
        how = (f"empirical CDF: share of zeros over all {prep.n_cells:,} cells, non-zero values "
               f"from a fixed sample of {prep.n_ref:,}")
    else:
        how = f"empirical CDF over all {prep.n_cells:,} cells"
    note = f"GSVA on the {prep.source}, {prep.p:,} genes that vary; {how}"
    if undefined:
        note += f"; {undefined:,} cells with no defined walk score 0"
    return ssgsea.SetScores(scores=scores, match=m, source=prep.source, note=note)
