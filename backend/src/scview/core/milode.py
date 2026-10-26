"""Cluster-free differential expression in neighbourhoods of the cell graph,
after miloDE (Missarova et al. 2024, Genome Biology 25:189, PMID 39026254).
Implemented from the paper and the package's documented behaviour (miloDE is
MIT); no code is copied.

A condition effect confined to part of a cell type, or one that crosses
cluster borders, is averaged away by DE per cluster. miloDE tests DE between
two conditions inside each of many small, overlapping neighbourhoods instead,
with the samples (donors) as replicates:

1. **Neighbourhoods.** A k-nearest-neighbour graph (k = 25, miloDE's default)
   is built on the first 30 dimensions of the representation the dataset's
   own graph used (PCA or a corrected PCA). A share of the cells (``prop``,
   0.2 as in miloDE) is sampled and refined as Milo does
   (``milo.make_nhoods``). A neighbourhood is every cell within two steps of
   its index cell (miloDE's second-order neighbourhoods, a few hundred cells),
   so each holds enough cells per sample to sum. A greedy set cover then keeps
   only as many as it takes to cover every cell they hold (miloDE's
   ``filtering``), which removes most of the overlap.
2. **Pseudobulk.** Raw counts are summed per sample in each neighbourhood.
   Samples with fewer than 3 cells there are left out (miloDE's
   ``min_n_cells_per_sample``); a neighbourhood is tested only when each
   condition keeps 2 or more samples.
3. **Test** (``test_nhood``): edgeR's ``filterByExpr`` as miloDE calls it
   (min count 3, min total 4, each sample its own group), TMM factors, then
   limma-voom: log-CPM with precision weights from the mean-variance trend,
   weighted least squares per gene, empirical Bayes moderated t of the
   condition. It matches edgeR 4.10 + limma 3.68 to 1e-9 (log2FC and log10 p). miloDE
   itself fits edgeR's quasi-likelihood model; voom is the faster of the two
   standard count models and is vectorised across genes here. Library sizes
   are each sample's counts over all genes, also when only the highly
   variable genes are tested. Donors in both conditions are paired
   (``~ donor + condition``), as in Milo; a neighbourhood where no donor has
   both conditions falls back to ``~ condition``.
4. **Multiple testing**, as miloDE: per gene across the neighbourhoods where
   it was tested, a weighted BH whose weight is 1 / (cells the neighbourhood
   shares with all of them), so overlapping neighbourhoods count less each
   (miloDE's ``spatial_pval_adjustment``); and per neighbourhood, BH across
   its genes.

Against miloDE 0.1.0 (edgeR 4.10.7) on Kang (13,836 PBMCs, Harmony over the
condition, 2,000 HVGs, ``~ Donor + stim``), given the same 49 neighbourhoods
and HVG library sizes: the same 42 neighbourhoods and 34,515 (neighbourhood,
gene) pairs are tested. On the 34 where both conditions mix, log2FC r = 0.92
(voom's are about 0.7 of edgeR's: its 0.5 pseudo-count shrinks low counts more),
-log10 p Spearman 0.86, and of the pairs DE at spatial FDR < 0.1 (4,626 here,
4,061 there) 3,559 are shared (Jaccard 0.69), all in the same direction; ISG15,
CXCL10 and IFITM3 are DE in 42 of 42 in both. 0.6 s here, 89 s there. The 8
others hold under 10% of one condition's cells (CD14 monocytes the embedding
did not integrate): there the two models disagree by up to 5 log2 units, so
such neighbourhoods are flagged (``ONE_SIDED``).

Summaries: per gene, in how many of the neighbourhoods where it was tested it
is DE (miloDE's "how widespread is the DE"); per neighbourhood, how many genes
are DE there.
"""

from __future__ import annotations

import logging
import threading
import time
import warnings
import weakref
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats

from scview.core import milo, results_store
from scview.core.composition import CompositionError, fit_f_dist, fit_f_dist_robust

logger = logging.getLogger(__name__)

K = 25  # miloDE's default k for the neighbour graph
MIN_K, MAX_K = 10, 50
ORDER = 2  # cells within two steps of the index cell
PROP = 0.2  # miloDE's default share of cells sampled as index cells
MIN_PROP, MAX_PROP = 0.02, 0.3
DIMS = 30  # miloDE's d: the representation's first 30 dimensions
MIN_CELLS_PER_SAMPLE = 3  # miloDE's min_n_cells_per_sample
MIN_SAMPLES = 2  # per condition, in a neighbourhood
MIN_COUNT = 3  # filterByExpr's min.count as miloDE sets it
MIN_TOTAL_COUNT = 4  # miloDE: round(1.5 * min_count), R rounding half to even
MIN_EXPRESSED_TOTAL = 15  # "expressed genes" for the gene choice
MIN_GENES_TESTED = 10  # fewer pass the filter: the neighbourhood is not tested
ROBUST = False  # eBayes robust = TRUE (limma); see test_nhood
ONE_SIDED = 0.1  # a neighbourhood with under 10% of its cells from one condition
ONE_SIDED_WARN = 0.25  # share of tested neighbourhoods that are one-sided before warning
ALPHA = 0.1  # spatial FDR (and within-neighbourhood BH) threshold, as miloDE's summaries
MAX_CELLS = 200_000
MAX_NHOODS = 5_000
# Sampled centres before the set cover: more only adds overlap (and memory: each
# second-order neighbourhood holds hundreds of cells), so prop is lowered above it.
MAX_SAMPLED = 6_000
MAX_GENES = 15_000  # "expressed genes": the most counted, at most this many
HVG = "hvg"
EXPRESSED = "expressed"
CHUNK = 20  # neighbourhoods per progress step (Cancel is checked between them)
NHOOD_GENES_MAX = 300  # rows a neighbourhood's gene table returns
# Seconds per (tested neighbourhood x gene) for the test, and per (cell^2 x
# dimension) for the exact kNN; measured on Kang (one laptop core).
SECONDS_PER_NHOOD_GENE = 1e-5
SECONDS_PER_KNN = 2e-11
SECONDS_BASE = 1.0


class MiloDEError(ValueError):
    """A request miloDE cannot serve (no counts, too few samples): a 400."""


# ---------------------------------------------------------------------------
# Neighbourhoods
# ---------------------------------------------------------------------------


@dataclass
class Nhoods:
    index: np.ndarray  # index cell of each neighbourhood
    membership: sp.csc_matrix  # cells x neighbourhoods, 1 = member
    n_sampled: int  # neighbourhoods before the redundant ones were dropped

    @property
    def sizes(self) -> np.ndarray:
        return np.diff(self.membership.indptr)


def knn_graph(X: np.ndarray, k: int) -> sp.csr_matrix:
    """Symmetric binary k-nearest-neighbour graph (exact, self left out), the
    undirected graph miloR's buildGraph makes."""
    from sklearn.neighbors import NearestNeighbors

    n = X.shape[0]
    k = min(k, n - 1)
    if k < 1:
        raise MiloDEError("Too few cells for a neighbour graph.")
    nn = NearestNeighbors(n_neighbors=k + 1, n_jobs=1).fit(X)
    idx = nn.kneighbors(X, return_distance=False)
    rows = np.repeat(np.arange(n), k)
    # The first neighbour is the cell itself, except among exact duplicates.
    cols = np.where(idx[:, :1] == np.arange(n)[:, None], idx[:, 1:], idx[:, :k]).ravel()
    A = sp.csr_matrix((np.ones(rows.size, dtype=np.float32), (rows, cols)), shape=(n, n))
    A = ((A + A.T) > 0).astype(np.float32)
    A.setdiag(0)
    A.eliminate_zeros()
    return sp.csr_matrix(A)


def second_order(A: sp.csr_matrix, index: np.ndarray) -> sp.csc_matrix:
    """Cells within two steps of each index cell (the cell itself included)."""
    n = A.shape[0]
    first = sp.csc_matrix(A[:, index])
    eye = sp.csc_matrix((np.ones(len(index)), (index, np.arange(len(index)))),
                        shape=(n, len(index)))
    M = sp.csc_matrix(A @ first) + first + eye
    M.data[:] = 1
    M = sp.csc_matrix(M, dtype=np.int32)
    M.eliminate_zeros()
    M.sort_indices()
    return M


def drop_redundant(M: sp.csc_matrix) -> np.ndarray:
    """Neighbourhoods (columns) to keep, sorted: a greedy set cover of the
    cells they hold (miloDE's ``filtering``), taking the neighbourhood that
    covers the most still-uncovered cells each time (lazily re-scored)."""
    import heapq

    covered = np.zeros(M.shape[0], dtype=bool)
    heap = [(-int(M.indptr[j + 1] - M.indptr[j]), j) for j in range(M.shape[1])]
    heapq.heapify(heap)
    keep: list[int] = []
    while heap:
        neg, j = heapq.heappop(heap)
        if neg == 0:
            break
        gain = int((~covered[M.indices[M.indptr[j]:M.indptr[j + 1]]]).sum())
        if gain == 0:
            continue
        if heap and gain < -heap[0][0]:
            heapq.heappush(heap, (-gain, j))  # stale score: try again later
            continue
        keep.append(j)
        covered[M.indices[M.indptr[j]:M.indptr[j + 1]]] = True
    return np.sort(np.array(keep, dtype=np.int64))


def make_nhoods(X: np.ndarray, k: int = K, prop: float = PROP, seed: int = milo.SEED,
                filtering: bool = True) -> Nhoods:
    """Second-order neighbourhoods on a k-NN graph of ``X``, index cells by
    Milo's refined sampling, redundant ones dropped."""
    X = np.asarray(X, dtype=np.float32)
    A = knn_graph(X, k)
    first = milo.make_nhoods(A, X, None, prop, seed)
    M = second_order(A, first.index)
    n_sampled = M.shape[1]
    cols = drop_redundant(M) if filtering else np.arange(n_sampled)
    M = sp.csc_matrix(M[:, cols])
    index = first.index[cols]
    return Nhoods(index=index, membership=M, n_sampled=n_sampled)


def representation(adata) -> tuple[str, np.ndarray]:
    """The dataset graph's representation, its first ``DIMS`` dimensions."""
    rep, X = milo.representation(adata)
    return rep, np.asarray(X)[:, :DIMS]


def overlap_weights(membership: sp.csc_matrix, tested: np.ndarray) -> np.ndarray:
    """miloDE's spatial weights, per gene: neighbourhoods x genes, each tested
    neighbourhood's 1 / (cells it shares with every neighbourhood where that
    gene was tested, itself included), NaN where the gene was not tested."""
    overlap = np.asarray((membership.T @ membership).todense(), dtype=np.float64)
    shared = overlap @ tested.astype(np.float64)
    with np.errstate(divide="ignore"):
        return np.where(tested, 1.0 / shared, np.nan)


def spatial_fdr(pval: np.ndarray, membership: sp.csc_matrix) -> np.ndarray:
    """Per gene, a weighted Benjamini-Hochberg across the neighbourhoods where
    it was tested, weighted by :func:`overlap_weights` (miloDE's
    ``spatial_pval_adjustment``): heavily overlapping neighbourhoods count
    less each. Neighbourhoods x genes, NaN where not tested."""
    p = np.asarray(pval, dtype=np.float64)
    tested = np.isfinite(p)
    W = overlap_weights(membership, tested)
    out = np.full(p.shape, np.nan, dtype=np.float32)
    for g in range(p.shape[1]):
        rows = np.flatnonzero(tested[:, g])
        if rows.size == 1:
            out[rows, g] = p[rows, g]  # miloDE: a single p is left as it is
        elif rows.size > 1:
            # milo.spatial_fdr weights by 1 / its second argument.
            out[rows, g] = milo.spatial_fdr(p[rows, g], 1.0 / W[rows, g])
    return out


_nhood_cache: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_nhood_lock = threading.Lock()


def nhoods_for(adata, k: int, prop: float) -> tuple[str, Nhoods]:
    """The neighbourhoods of a dataset for (k, prop), cached on the AnnData:
    the plan builds them and the run reuses them."""
    rep, X = representation(adata)
    key = (rep, X.shape, int(k), round(float(prop), 4), adata.n_obs)
    try:
        with _nhood_lock:
            hit = _nhood_cache.get(adata)
        if hit is not None and hit[0] == key:
            return rep, hit[1]
    except TypeError:
        hit = None
    nh = make_nhoods(X, k, prop)
    try:
        with _nhood_lock:
            _nhood_cache[adata] = (key, nh)
    except TypeError:
        pass
    return rep, nh


# ---------------------------------------------------------------------------
# The test in one neighbourhood (edgeR filter + TMM, limma-voom, eBayes)
# ---------------------------------------------------------------------------


def filter_by_expr(counts: np.ndarray, lib: np.ndarray, min_count: float = MIN_COUNT,
                   min_total: float = MIN_TOTAL_COUNT) -> np.ndarray:
    """edgeR's filterByExpr as miloDE calls it (each sample its own group, so
    one sample suffices): genes (columns of ``counts``, samples x genes) with
    a CPM of at least min_count / median library size in some sample and
    ``min_total`` counts in all."""
    cutoff = min_count / float(np.median(lib)) * 1e6
    cpm = counts / lib[:, None] * 1e6
    tol = 1e-14
    return ((cpm >= cutoff).any(axis=0) & (counts.sum(axis=0) >= min_total - tol))


def tmm_factors(counts: np.ndarray, lib: np.ndarray, trim_m: float = 0.3,
                trim_a: float = 0.05) -> np.ndarray:
    """edgeR's TMM normalisation factors (samples x genes counts), scaled to a
    geometric mean of 1."""
    n = counts.shape[0]
    q75 = np.array([np.quantile(counts[i] / lib[i], 0.75) for i in range(n)])
    if np.median(q75) < 1e-20:
        ref = int(np.argmax(np.sqrt(counts).sum(axis=1)))
    else:
        ref = int(np.argmin(np.abs(q75 - q75.mean())))
    r, nr = counts[ref].astype(np.float64), float(lib[ref])
    f = np.ones(n)
    for i in range(n):
        if i == ref:
            continue
        o, no = counts[i].astype(np.float64), float(lib[i])
        with np.errstate(divide="ignore", invalid="ignore"):
            log_r = np.log2((o / no) / (r / nr))
            abs_e = (np.log2(o / no) + np.log2(r / nr)) / 2
            v = (no - o) / no / o + (nr - r) / nr / r
        ok = np.isfinite(log_r) & np.isfinite(abs_e)
        log_r, abs_e, v = log_r[ok], abs_e[ok], v[ok]
        if log_r.size == 0 or np.abs(log_r).max() < 1e-6:
            continue
        m = log_r.size
        lo_l, lo_s = np.floor(m * trim_m) + 1, np.floor(m * trim_a) + 1
        rank_r, rank_e = stats.rankdata(log_r), stats.rankdata(abs_e)
        keep = ((rank_r >= lo_l) & (rank_r <= m + 1 - lo_l)
                & (rank_e >= lo_s) & (rank_e <= m + 1 - lo_s))
        if keep.any():
            f[i] = 2 ** (float((log_r[keep] / v[keep]).sum()) / float((1 / v[keep]).sum()))
    return f / np.exp(np.log(f).mean())


def lowess_span(n_genes: int) -> float:
    """voom's adaptive span (limma >= 3.62, ``chooseLowessSpan``): wider with
    fewer genes, 0.3 at very many."""
    return float(min(0.3 + 0.7 * (50 / max(n_genes, 1)) ** (1 / 3), 1.0))


def _lowess_fun(x: np.ndarray, y: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """R's lowess(x, y, f = span) (3 robustness iterations, delta 1% of the
    range) as a function, linear between points and flat beyond them."""
    from statsmodels.nonparametric.smoothers_lowess import lowess

    delta = 0.01 * float(x.max() - x.min())
    fit = lowess(y, x, frac=lowess_span(len(x)), it=3, delta=delta, return_sorted=True)
    xs, inv = np.unique(fit[:, 0], return_inverse=True)
    ys = np.bincount(inv, weights=fit[:, 1]) / np.bincount(inv)
    return lambda q: np.interp(q, xs, ys)


@dataclass
class NhoodTest:
    log_fc: np.ndarray  # per gene (NaN = filtered out)
    pval: np.ndarray
    n_genes: int  # genes tested


def test_nhood(counts: np.ndarray, lib: np.ndarray, X: np.ndarray,
               min_count: float = MIN_COUNT, robust: bool = ROBUST) -> NhoodTest | None:
    """(log2FC, p) per gene of the last column of design ``X`` on pseudobulk
    ``counts`` (samples x genes; ``lib`` = each sample's total counts over all
    genes): filterByExpr, TMM, voom, weighted least squares, eBayes. None when
    too few genes pass the filter. ``robust``: eBayes(robust = TRUE), outlying
    variances get less shrinkage. One function so the engine can be swapped."""
    counts = np.asarray(counts, dtype=np.float64)
    lib = np.asarray(lib, dtype=np.float64)
    n, p = X.shape
    G = counts.shape[1]
    df = n - p
    out_fc = np.full(G, np.nan)
    out_p = np.full(G, np.nan)
    keep = filter_by_expr(counts, lib, min_count)
    if keep.sum() < MIN_GENES_TESTED or df < 1:
        return None
    y = counts[:, keep]
    eff = lib * tmm_factors(y, lib)

    # voom: log-CPM, a first unweighted fit, the sqrt-sd trend, the weights.
    logcpm = np.log2((y + 0.5) / (eff[:, None] + 1) * 1e6)
    pinv = np.linalg.pinv(X)
    beta = pinv @ logcpm
    fitted = X @ beta
    sigma = np.sqrt(((logcpm - fitted) ** 2).sum(axis=0) / df)
    sx = logcpm.mean(axis=0) + np.log2(eff + 1).mean() - np.log2(1e6)
    trend = _lowess_fun(sx, np.sqrt(sigma))
    fitted_logcount = fitted + np.log2(eff + 1)[:, None] - np.log2(1e6)
    w = 1.0 / np.maximum(trend(fitted_logcount), 1e-3) ** 4

    # Weighted least squares per gene, then limma's eBayes on the residual variances.
    XtWX = np.einsum("sg,si,sj->gij", w, X, X)
    XtWy = np.einsum("sg,si,sg->gi", w, X, logcpm)
    try:
        V = np.linalg.inv(XtWX)
    except np.linalg.LinAlgError:
        return None
    b = np.einsum("gij,gj->gi", V, XtWy)
    resid = logcpm - X @ b.T
    s2 = (w * resid ** 2).sum(axis=0) / df
    g = int(keep.sum())
    s0, d0 = float("nan"), np.zeros(g)
    if g >= 3:
        try:
            if robust:
                s0, d0 = fit_f_dist_robust(s2, float(df))
            else:
                s0, d = fit_f_dist(s2, float(df))
                d0 = np.full(g, d)
        except CompositionError:
            s0 = float("nan")
    if not np.isfinite(s0):
        s_post, d0 = s2, np.zeros(g)
    else:
        with np.errstate(invalid="ignore"):
            s_post = np.where(np.isfinite(d0), (df * s2 + d0 * s0) / (df + d0), s0)
    df_total = np.minimum(df + d0, df * g)
    j = p - 1
    with np.errstate(divide="ignore", invalid="ignore"):
        t = b[:, j] / np.sqrt(s_post * V[:, j, j])
    pv = 2 * stats.t.sf(np.abs(t), df_total)
    out_fc[keep] = b[:, j]
    out_p[keep] = pv
    return NhoodTest(log_fc=out_fc, pval=out_p, n_genes=g)


# ---------------------------------------------------------------------------
# Plan and run
# ---------------------------------------------------------------------------


@dataclass
class Plan:
    base: milo.Plan  # samples, conditions, design (Milo's own checks)
    k: int
    prop: float
    genes_mode: str
    gene_idx: np.ndarray  # columns of the counts matrix tested
    genes: list[str]
    counts_source: str
    nhoods: Nhoods
    n_samples_nhood: np.ndarray  # per neighbourhood: samples with 3+ cells
    testable: np.ndarray  # per neighbourhood: 2+ such samples in each condition
    estimate_seconds: float
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        b = self.base
        sizes = self.nhoods.sizes
        return {
            "condition_key": b.condition_key, "sample_key": b.sample_key,
            "conditions": b.conditions, "reference": b.conditions[0], "design": b.design,
            "covariates": b.covariates, "formula": b.formula, "n_cells": b.n_cells,
            "n_samples": b.n_samples, "representation": b.representation,
            "k": self.k, "order": ORDER, "prop": self.prop, "genes_mode": self.genes_mode,
            "n_genes": len(self.genes), "counts_source": self.counts_source,
            "n_nhoods": int(len(self.nhoods.index)), "n_nhoods_sampled": self.nhoods.n_sampled,
            "n_testable": int(self.testable.sum()),
            "median_nhood_size": float(np.median(sizes)) if len(sizes) else 0.0,
            "median_samples_per_nhood": float(np.median(self.n_samples_nhood))
            if len(self.n_samples_nhood) else 0.0,
            "min_cells_per_sample": MIN_CELLS_PER_SAMPLE,
            "estimate_seconds": round(self.estimate_seconds, 1),
            "warnings": b.warnings, "notes": b.notes + self.notes,
            "dropped_samples": b.dropped_samples,
        }


def _counts(adata):
    from scview.core.pseudobulk import PseudobulkError, raw_counts

    try:
        return raw_counts(adata)
    except PseudobulkError as exc:
        raise MiloDEError(str(exc).replace("Pseudobulk DE", "Neighbourhood DE")) from exc


def choose_genes(adata, M, gene_names, mode: str) -> tuple[np.ndarray, str, list[str]]:
    """Columns of the counts matrix to test: the highly variable genes, or the
    expressed genes (15+ counts in all), the most counted first, capped."""
    notes: list[str] = []
    names = pd.Index([str(g) for g in gene_names])
    if mode == HVG:
        hv = adata.var.get("highly_variable")
        if hv is not None and bool(np.asarray(hv, dtype=bool).any()):
            want = adata.var_names[np.asarray(hv, dtype=bool)]
            idx = names.get_indexer([str(g) for g in want])
            idx = np.sort(idx[idx >= 0])
            if idx.size >= MIN_GENES_TESTED:
                return idx, HVG, notes
        notes.append("This dataset has no highly variable genes flagged: the expressed genes "
                     "are tested instead.")
        mode = EXPRESSED
    if mode != EXPRESSED:
        raise MiloDEError(f"Genes must be '{HVG}' or '{EXPRESSED}'.")
    totals = np.asarray(M.sum(axis=0)).ravel()
    idx = np.flatnonzero(totals >= MIN_EXPRESSED_TOTAL)
    if idx.size > MAX_GENES:
        notes.append(f"{idx.size:,} genes are expressed; the {MAX_GENES:,} with the most counts "
                     "are tested.")
        idx = np.sort(idx[np.argsort(-totals[idx], kind="stable")[:MAX_GENES]])
    if idx.size < MIN_GENES_TESTED:
        raise MiloDEError("Too few expressed genes to test.")
    return idx, EXPRESSED, notes


def _cell_samples(adata, base: milo.Plan) -> np.ndarray:
    """Per cell: its sample code, -1 when it is not counted."""
    s = -np.ones(adata.n_obs, dtype=np.int64)
    s[np.flatnonzero(base.keep)] = base.sample_codes
    return s


def _samples_per_nhood(M: sp.csc_matrix, cell_sample: np.ndarray, n_samples: int) -> np.ndarray:
    """Cells per (neighbourhood, sample)."""
    ok = cell_sample >= 0
    onehot = sp.csr_matrix((np.ones(int(ok.sum()), dtype=np.int32),
                            (np.flatnonzero(ok), cell_sample[ok])),
                           shape=(len(cell_sample), n_samples))
    return np.asarray((M.T @ onehot).todense(), dtype=np.int64)


def estimate_seconds(n_cells: int, dims: int, n_testable: int, n_genes: int) -> float:
    return (SECONDS_BASE + SECONDS_PER_NHOOD_GENE * n_testable * n_genes
            + SECONDS_PER_KNN * float(n_cells) ** 2 * dims)


def plan(adata, condition_key: str, sample_key: str | None, *,
         conditions: list[str] | None = None, covariates: list[str] | None = None,
         paired: bool | None = None, k: int = K, prop: float = PROP, genes: str = HVG,
         batch_key: str | None = None) -> Plan:
    """Check a request, build (or reuse) the neighbourhoods and count the
    samples in each: what a run would test and about how long it would take."""
    from scview.core import large_mode

    large_mode.check(adata, "Neighbourhood DE (miloDE)", large_mode.COUNTS_MATRIX)
    if not MIN_K <= k <= MAX_K:
        raise MiloDEError(f"k must be between {MIN_K} and {MAX_K}.")
    if not MIN_PROP <= prop <= MAX_PROP:
        raise MiloDEError(f"The sampled share must be between {MIN_PROP} and {MAX_PROP}.")
    if adata.n_obs > MAX_CELLS:
        raise MiloDEError(f"Neighbourhood DE runs on at most {MAX_CELLS:,} cells; this dataset "
                          f"has {adata.n_obs:,}.")
    try:
        base = milo.plan(adata, condition_key, sample_key, conditions=conditions,
                         covariates=covariates, paired=paired, prop=prop, batch_key=batch_key)
    except milo.MiloError as exc:
        raise MiloDEError(str(exc).replace("Milo's neighbourhoods", "the neighbourhoods")
                          .replace("Milo ", "Neighbourhood DE ")) from exc
    M, gene_names, source = _counts(adata)
    gene_idx, mode, notes = choose_genes(adata, M, gene_names, genes)
    if adata.n_obs * prop > MAX_SAMPLED:
        prop = round(MAX_SAMPLED / adata.n_obs, 4)
        notes.append(f"{prop:.1%} of the cells ({MAX_SAMPLED:,}) are sampled as neighbourhood "
                     "centres: the set cover keeps few of them anyway.")
    rep, nh = nhoods_for(adata, k, prop)
    if len(nh.index) > MAX_NHOODS:
        raise MiloDEError(f"{len(nh.index):,} neighbourhoods is too many (at most "
                          f"{MAX_NHOODS:,}); sample a smaller share of the cells.")
    per = _samples_per_nhood(nh.membership, _cell_samples(adata, base), len(base.sample_cond))
    ok = per >= MIN_CELLS_PER_SAMPLE
    n_ok = ok.sum(axis=1)
    testable = np.all([(ok[:, base.sample_cond == c]).sum(axis=1) >= MIN_SAMPLES
                       for c in base.conditions], axis=0)
    if not testable.any():
        notes.append(f"No neighbourhood has {MIN_SAMPLES}+ samples of each condition with "
                     f"{MIN_CELLS_PER_SAMPLE}+ cells: nothing can be tested. A larger k gives "
                     "bigger neighbourhoods.")
    dims = representation(adata)[1].shape[1]
    est = estimate_seconds(adata.n_obs, dims, int(testable.sum()), int(gene_idx.size))
    return Plan(base=base, k=int(k), prop=float(prop), genes_mode=mode, gene_idx=gene_idx,
                genes=[str(gene_names[i]) for i in gene_idx], counts_source=source, nhoods=nh,
                n_samples_nhood=n_ok, testable=testable, estimate_seconds=est, notes=notes)


@dataclass
class MiloDEResult:
    condition_key: str
    sample_key: str
    label_key: str | None
    conditions: list[str]  # [reference, tested]
    design: str
    covariates: list[str]
    formula: str
    representation: str
    k: int
    prop: float
    genes_mode: str
    counts_source: str
    n_cells: int
    n_samples: dict[str, int]
    genes: list[str]
    index_cells: list[str]
    index: np.ndarray
    sizes: np.ndarray
    n_samples_nhood: np.ndarray
    tested: np.ndarray  # per neighbourhood
    nhood_design: list[str]  # per neighbourhood: "paired" | "unpaired" | "" (not tested)
    log_fc: np.ndarray  # neighbourhoods x genes, float32, NaN = not tested
    pval: np.ndarray
    spatial_fdr: np.ndarray  # per gene across neighbourhoods
    labels: list[str] | None = None
    label_fraction: np.ndarray | None = None
    membership: sp.csc_matrix | None = None
    # per neighbourhood: cells of each condition, and the smaller condition's
    # share after scaling both conditions to the same size
    cells_by_condition: dict[str, np.ndarray] = field(default_factory=dict)
    minority_share: np.ndarray | None = None
    alpha: float = ALPHA
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    dropped_samples: list[str] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)

    # -- summaries ------------------------------------------------------------

    @property
    def significant(self) -> np.ndarray:
        return np.nan_to_num(self.spatial_fdr, nan=1.0) < self.alpha

    def within_fdr(self) -> np.ndarray:
        """Per neighbourhood: BH across its tested genes."""
        out = np.full(self.pval.shape, np.nan, dtype=np.float32)
        for i in range(self.pval.shape[0]):
            p = self.pval[i].astype(np.float64)
            ok = np.isfinite(p)
            if ok.any():
                out[i, ok] = stats.false_discovery_control(p[ok], method="bh")
        return out

    def gene_table(self) -> list[dict[str, Any]]:
        """Per gene: neighbourhoods tested, DE (spatial FDR < alpha), up, down,
        the DE share of the tested ones and the median log2FC, most widespread first."""
        fc = self.log_fc.astype(np.float64)
        tested = np.isfinite(fc)
        sig = self.significant
        n_t = tested.sum(axis=0)
        n_up = (sig & (fc > 0)).sum(axis=0)
        n_dn = (sig & (fc < 0)).sum(axis=0)
        with warnings.catch_warnings():  # all-NaN columns: genes never tested
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(np.where(tested, fc, np.nan), axis=0)
            min_fdr = np.nanmin(np.where(tested, self.spatial_fdr, np.nan), axis=0)
        rows = []
        for g, name in enumerate(self.genes):
            if not n_t[g]:
                continue
            n_de = int(n_up[g] + n_dn[g])
            rows.append({"gene": name, "n_tested": int(n_t[g]), "n_de": n_de,
                         "n_up": int(n_up[g]), "n_down": int(n_dn[g]),
                         "share": n_de / int(n_t[g]),
                         "median_logfc": _f(med[g]), "min_spatial_fdr": _f(min_fdr[g], sig=True)})
        rows.sort(key=lambda r: (-r["n_de"], -r["share"],
                                 1.0 if r["min_spatial_fdr"] is None else r["min_spatial_fdr"],
                                 r["gene"]))
        return rows

    def nhood_counts(self) -> tuple[np.ndarray, np.ndarray]:
        """Per neighbourhood: genes DE by spatial FDR, and by BH within it."""
        n_de = self.significant.sum(axis=1)
        within = (np.nan_to_num(self.within_fdr(), nan=1.0) < self.alpha).sum(axis=1)
        return n_de.astype(int), within.astype(int)

    def gene_detail(self, gene: str) -> dict[str, Any]:
        """One gene in every neighbourhood: log2FC, p, spatial FDR."""
        try:
            g = self.genes.index(gene)
        except ValueError:
            low = {x.lower(): i for i, x in enumerate(self.genes)}
            if gene.lower() not in low:
                raise MiloDEError(f"'{gene}' was not among the genes tested.") from None
            g = low[gene.lower()]
        fdr = self.spatial_fdr[:, g]
        return {"gene": self.genes[g], "alpha": self.alpha,
                "log_fc": _floats(self.log_fc[:, g]), "pval": _floats(self.pval[:, g], sig=True),
                "spatial_fdr": _floats(fdr, sig=True),
                "significant": [bool(x) for x in np.nan_to_num(fdr, nan=1.0) < self.alpha]}

    def nhood_detail(self, i: int, limit: int = NHOOD_GENES_MAX) -> dict[str, Any]:
        """One neighbourhood's genes, smallest p first."""
        if not 0 <= i < len(self.index):
            raise MiloDEError(f"No neighbourhood {i}.")
        p = self.pval[i].astype(np.float64)
        ok = np.flatnonzero(np.isfinite(p))
        within = np.full(len(p), np.nan)
        if ok.size:
            within[ok] = stats.false_discovery_control(p[ok], method="bh")
        order = ok[np.argsort(p[ok], kind="stable")]
        rows = [{"gene": self.genes[g], "log_fc": _f(self.log_fc[i, g]),
                 "pval": _f(p[g], sig=True), "fdr": _f(within[g], sig=True),
                 "spatial_fdr": _f(self.spatial_fdr[i, g], sig=True)}
                for g in order[:limit]]
        return {"nhood": i, "index_cell": self.index_cells[i], "size": int(self.sizes[i]),
                "n_samples": int(self.n_samples_nhood[i]), "tested": bool(self.tested[i]),
                "design": self.nhood_design[i],
                "label": self.labels[i] if self.labels else None,
                "n_genes_tested": int(ok.size),
                "n_de": int((np.nan_to_num(self.spatial_fdr[i], nan=1.0) < self.alpha).sum()),
                "n_de_within": int((np.nan_to_num(within, nan=1.0) < self.alpha).sum()),
                "alpha": self.alpha, "genes": rows, "n_rows": int(ok.size)}

    def to_dict(self, layouts: dict[str, np.ndarray] | None = None) -> dict:
        n_de, n_within = self.nhood_counts()
        genes = self.gene_table()
        return {
            "condition_key": self.condition_key, "sample_key": self.sample_key,
            "label_key": self.label_key, "conditions": self.conditions,
            "reference": self.conditions[0], "design": self.design,
            "covariates": self.covariates, "formula": self.formula,
            "representation": self.representation, "k": self.k, "order": ORDER,
            "prop": self.prop, "genes_mode": self.genes_mode, "counts_source": self.counts_source,
            "n_cells": self.n_cells, "n_samples": self.n_samples,
            "n_nhoods": int(len(self.index)), "n_tested": int(self.tested.sum()),
            "n_genes": len(self.genes), "n_genes_de": sum(1 for r in genes if r["n_de"]),
            "alpha": self.alpha, "min_cells_per_sample": MIN_CELLS_PER_SAMPLE,
            "one_sided": ONE_SIDED,
            "nhoods": {
                "index_cell": self.index_cells, "cell": self.index.astype(int).tolist(),
                "size": self.sizes.astype(int).tolist(),
                "n_samples": self.n_samples_nhood.astype(int).tolist(),
                "cells": {k: v.astype(int).tolist() for k, v in self.cells_by_condition.items()},
                "minority_share": _floats(self.minority_share),
                "tested": [bool(x) for x in self.tested],
                "design": self.nhood_design,
                "n_de": n_de.tolist(), "n_de_within": n_within.tolist(),
                "label": self.labels, "label_fraction": _floats(self.label_fraction),
            },
            "genes": genes,
            "layouts": {k: _floats2(v) for k, v in (layouts or {}).items()},
            "warnings": self.warnings, "notes": self.notes,
            "dropped_samples": self.dropped_samples,
            "timings": {k: round(v, 2) for k, v in self.timings.items()},
        }


def _f(v, sig: bool = False) -> float | None:
    v = float(v)
    if not np.isfinite(v):
        return None
    return float(f"{v:.4g}") if sig else round(v, 4)


def _floats(a: np.ndarray | None, sig: bool = False) -> list | None:
    if a is None:
        return None
    return [_f(v, sig) for v in np.asarray(a, dtype=np.float64)]


def _floats2(a: np.ndarray) -> list:
    return [[_f(v) for v in row] for row in np.asarray(a, dtype=np.float64)]


def _design(p: milo.Plan, s: np.ndarray) -> tuple[np.ndarray, str] | None:
    """The design over samples ``s`` of one neighbourhood: paired when the run
    is and a donor there has both conditions; None when nothing is estimable."""
    cond = p.sample_cond[s]
    covs = [p.sample_cov[c][s] for c in p.covariates]
    tries = [(p.sample_donor[s], "paired"), (None, "unpaired")] if p.design == "paired" else [
        (None, "unpaired")]
    for donors, kind in tries:
        X = milo.design_matrix(cond, p.conditions, donors, covs)
        if np.linalg.matrix_rank(X) == X.shape[1] and X.shape[0] - X.shape[1] >= 1:
            return X, kind
    return None


def iter_milode(adata, plan_: Plan, *, label_key: str | None = None,
                stop: Callable[[], bool] | None = None) -> Iterator[dict[str, Any]]:
    """Run neighbourhood DE, yielding progress events (``{"step", "message"}``,
    test steps with ``done`` / ``total``) and, last, ``{"result": MiloDEResult}``.
    ``adata`` is not changed; ``stop`` is checked between chunks."""
    stop = stop or (lambda: False)
    t0 = time.perf_counter()
    timings: dict[str, float] = {}
    b = plan_.base
    if label_key is not None and label_key not in adata.obs.columns:
        raise MiloDEError(f"Column '{label_key}' not found.")
    nh = plan_.nhoods
    n_nh, n_s = len(nh.index), len(b.sample_cond)
    yield {"step": "counts", "message": f"Reading counts of {len(plan_.genes):,} genes"}
    M, _names, _src = _counts(adata)
    M = sp.csr_matrix(M) if sp.issparse(M) else sp.csr_matrix(np.asarray(M))
    cell_sample = _cell_samples(adata, b)
    lib_cell = np.asarray(M.sum(axis=1)).ravel().astype(np.float64)
    Mg = sp.csr_matrix(M[:, plan_.gene_idx], dtype=np.float64)
    del M
    timings["counts"] = time.perf_counter() - t0
    if stop():
        return

    G = len(plan_.genes)
    log_fc = np.full((n_nh, G), np.nan, dtype=np.float32)
    pval = np.full((n_nh, G), np.nan, dtype=np.float32)
    tested = np.zeros(n_nh, dtype=bool)
    nhood_design = [""] * n_nh
    todo = np.flatnonzero(plan_.testable)
    t = time.perf_counter()
    total = len(todo)
    yield {"step": "test", "message": f"Testing {total:,} neighbourhoods "
           f"(about {plan_.estimate_seconds:.0f} s)", "done": 0, "total": total}
    memb = nh.membership
    for start in range(0, total, CHUNK):
        for i in todo[start:start + CHUNK]:
            cells = memb.indices[memb.indptr[i]:memb.indptr[i + 1]]
            cs = cell_sample[cells]
            cells, cs = cells[cs >= 0], cs[cs >= 0]
            n_per = np.bincount(cs, minlength=n_s)
            s = np.flatnonzero(n_per >= MIN_CELLS_PER_SAMPLE)
            if any((b.sample_cond[s] == c).sum() < MIN_SAMPLES for c in b.conditions):
                continue
            remap = -np.ones(n_s, dtype=np.int64)
            remap[s] = np.arange(len(s))
            r = remap[cs]
            use = r >= 0
            S = sp.csr_matrix((np.ones(int(use.sum())), (r[use], np.flatnonzero(use))),
                              shape=(len(s), len(cells)))
            counts = np.asarray((S @ Mg[cells]).todense())
            lib = S @ lib_cell[cells]
            d = _design(b, s)
            if d is None:
                continue
            res = test_nhood(counts, lib, d[0])
            if res is None:
                continue
            log_fc[i], pval[i] = res.log_fc, np.maximum(res.pval, np.finfo(np.float32).tiny)
            tested[i] = True
            nhood_design[i] = d[1]
        if stop():
            return
        done = min(start + CHUNK, total)
        yield {"step": "test", "message": f"Tested {done:,} of {total:,} neighbourhoods",
               "done": done, "total": total}
    timings["test"] = time.perf_counter() - t

    yield {"step": "fdr", "message": "Spatial FDR across neighbourhoods, per gene"}
    t = time.perf_counter()
    sfdr = spatial_fdr(pval, nh.membership)
    labels = frac = None
    if label_key is not None:
        labels, frac = milo.annotate(adata.obs[label_key], nh.membership)
    timings["fdr"] = time.perf_counter() - t
    timings["total"] = time.perf_counter() - t0

    per = _samples_per_nhood(nh.membership, cell_sample, n_s)
    by_cond = {c: per[:, b.sample_cond == c].sum(axis=1) for c in b.conditions}
    sizes_c = {c: max(int(np.isin(cell_sample, np.flatnonzero(b.sample_cond == c)).sum()), 1)
               for c in b.conditions}
    x, y = (by_cond[c] / sizes_c[c] for c in b.conditions)
    with np.errstate(invalid="ignore", divide="ignore"):
        minority = np.minimum(x, y) / (x + y)
    warns = list(b.warnings)
    one_sided = tested & (np.nan_to_num(minority, nan=0.0) < ONE_SIDED)
    if tested.any() and one_sided.sum() / tested.sum() >= ONE_SIDED_WARN:
        warns.append(
            f"{int(one_sided.sum())} of {int(tested.sum())} tested neighbourhoods hold under "
            f"{ONE_SIDED:.0%} of their cells from one condition (adjusted for condition size): the "
            "conditions separate on this embedding there, so those neighbourhoods compare a few "
            "cells per sample of one condition with many of the other, and their log2FC are "
            "unreliable. miloDE expects an embedding where matching cells of both conditions mix "
            "(one integrated across the conditions).")
    notes = list(plan_.notes) + list(b.notes)
    untested = int(n_nh - tested.sum())
    if untested:
        notes.append(f"{untested} of {n_nh} neighbourhoods were not tested: fewer than "
                     f"{MIN_SAMPLES} samples of a condition had {MIN_CELLS_PER_SAMPLE}+ cells "
                     f"there, or fewer than {MIN_GENES_TESTED} genes passed the expression filter.")
    if b.design == "paired" and any(x == "unpaired" for x in nhood_design):
        n_u = sum(1 for x in nhood_design if x == "unpaired")
        notes.append(f"{n_u} neighbourhoods had no donor in both conditions and were tested "
                     "unpaired.")
    if labels is not None:
        notes.append(f"Neighbourhoods whose commonest '{label_key}' label holds under "
                     f"{milo.MIXED_FRACTION:.0%} of their cells are labelled {milo.MIXED}.")
    names = adata.obs_names
    yield {"result": MiloDEResult(
        condition_key=b.condition_key, sample_key=b.sample_key, label_key=label_key,
        conditions=b.conditions, design=b.design, covariates=b.covariates, formula=b.formula,
        representation=b.representation, k=plan_.k, prop=plan_.prop, genes_mode=plan_.genes_mode,
        counts_source=plan_.counts_source, n_cells=b.n_cells, n_samples=b.n_samples,
        genes=plan_.genes, index_cells=[str(names[i]) for i in nh.index], index=nh.index,
        sizes=nh.sizes, n_samples_nhood=plan_.n_samples_nhood,
        tested=tested, nhood_design=nhood_design, log_fc=log_fc, pval=pval, spatial_fdr=sfdr,
        labels=labels, label_fraction=frac, membership=nh.membership,
        cells_by_condition=by_cond, minority_share=minority,
        warnings=warns, notes=notes, dropped_samples=b.dropped_samples,
        timings=timings,
    )}


def run(adata, plan_: Plan, *, label_key: str | None = None) -> MiloDEResult:
    """:func:`iter_milode` to the end."""
    for item in iter_milode(adata, plan_, label_key=label_key):
        if "result" in item:
            return item["result"]
    raise MiloDEError("Neighbourhood DE stopped before it finished.")


# ---------------------------------------------------------------------------
# The last run per loaded dataset: co-pilot, gene/neighbourhood reads, cells
# ---------------------------------------------------------------------------

_last: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_last_lock = threading.Lock()


def remember(adaptor, result: MiloDEResult) -> None:
    try:
        with _last_lock:
            _last[adaptor] = result
    except TypeError:  # not weak-referenceable (a test double)
        return
    results_store.keep(adaptor, "milode", result,
                       key=f"{result.condition_key}|{result.sample_key}",
                       label=f"Neighbourhood DE: {' vs '.join(reversed(result.conditions))}")


def last_result(adaptor) -> MiloDEResult | None:
    r = results_store.remembered(_last, _last_lock, adaptor, "milode")
    if r is None:
        return None
    obs = adaptor.adata.obs if hasattr(adaptor, "adata") else None
    if obs is not None:
        if r.membership is not None and r.membership.shape[0] != len(obs):
            return None  # cells filtered since
        if any(c and c not in obs.columns for c in (r.condition_key, r.sample_key, r.label_key)):
            return None
    return r


def nhood_cells(result: MiloDEResult, nhoods: list[int]) -> np.ndarray:
    return milo.nhood_cells(result, nhoods)  # type: ignore[arg-type]


def grounding_lines(adaptor) -> tuple[list[str], str | None]:
    """For the co-pilot: the last run's summary and its most widespread genes."""
    r = last_result(adaptor)
    if r is None:
        return [], None
    ref, test = r.conditions
    genes = r.gene_table()
    n_t = int(r.tested.sum())
    reps = ", ".join(f"{k} {n}" for k, n in r.n_samples.items())
    n_de_genes = sum(1 for g in genes if g["n_de"])
    detail = (f"{test} vs {ref}, samples from '{r.sample_key}' ({reps}; {r.design}), "
              f"{n_t} of {len(r.index)} neighbourhoods tested (second-order, k = {r.k}), "
              f"{len(r.genes)} genes: {n_de_genes} DE in at least one neighbourhood "
              f"(spatial FDR < {r.alpha})")
    lines = ["\n## Neighbourhood DE (miloDE-style, last run; samples as replicates)",
             f"- {detail} [result:milode]."]
    for g in [x for x in genes if x["n_de"]][:8]:
        lines.append(f"- {g['gene']}: DE in {g['n_de']} of {g['n_tested']} neighbourhoods "
                     f"({g['share']:.0%}; {g['n_up']} up, {g['n_down']} down; median log2FC "
                     f"{(g['median_logfc'] or 0.0):+.2f})")
    if r.labels is not None:
        n_de, _ = r.nhood_counts()
        by = pd.DataFrame({"label": r.labels, "n_de": n_de, "t": r.tested})
        by = by[by["t"]].groupby("label")["n_de"].median().sort_values(ascending=False)
        top = [f"{lab} {v:.0f}" for lab, v in by.head(4).items() if v > 0]
        if top:
            lines.append(f"- Median DE genes per neighbourhood by '{r.label_key}': "
                         + ", ".join(top))
    for w in r.warnings[:2]:
        lines.append(f"- Caution: {w}")
    return lines, detail
