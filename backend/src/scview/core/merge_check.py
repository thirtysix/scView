"""Should any clusters merge? Neighbouring clusters tested on held-out counts.

Testing differential expression between clusters found in the same data uses
the data twice ("double dipping"): clustering puts cells with different
expression in different clusters, so any two clusters, even of one cell type,
show significant genes. This check uses count splitting (Neufeld et al. 2022,
Biostatistics, PMID 35166794) to keep the test honest:

1. The counts are split into two independent halves, a training half and a
   test half, each holding about half of every cell's molecules.
2. The pipeline's own steps (normalisation, the recorded variable genes, scaling
   or Pearson residuals, PCA, Harmony when the graph was corrected, the
   neighbour graph and the clustering with the recorded method) run on the
   training half, as the variable-gene check re-runs them
   (``core/hvg_sensitivity.py``). Half the molecules give fewer clusters at the
   same resolution (Kang, 1.0: 10 instead of 15), so the resolution is then
   moved until the training half has as many clusters as the clustering being
   checked, as Neufeld et al. chose theirs to match the original analysis's
   count (``match_count``; it reads the training half only).
3. Training clusters that touch in the training graph (the share of their edge
   weight joining them) are compared on the test half's normalised counts with
   a Wilcoxon test (scanpy's, as the DE tab), on at most ``MAX_CELLS_PER_SIDE``
   cells a side so pairs of big and small clusters are judged alike. A gene is
   distinguishing at adjusted p < 0.05 and a two-fold change, detected in 10% of
   either cluster.
4. Training clusters the held-out half cannot tell apart (fewer than
   ``MERGE_MAX_GENES`` distinguishing genes) are joined into groups, fewest genes
   first, never joining a pair tested as distinct (``supported_groups``). Each
   checked cluster belongs to the group holding most of its cells
   (``HOST_SHARE``). Two neighbouring checked clusters in one group are a merge
   candidate, reported with the most distinct tested pair between their parts;
   in different groups they are kept apart. Two whose cells sit mostly in the
   same training cluster were never split on the training half, so nothing tested
   them: "together", not a candidate (at half depth close types fall together
   too: Kang's CD8 T and NK).

Negative binomial, not Poisson, splitting: scRNA-seq counts are overdispersed,
and Poisson splitting of overdispersed counts leaves the two halves positively
correlated (Neufeld et al. 2023, "Negative binomial count splitting for
single-cell RNA sequencing data", arXiv 2307.12985, Theorem 3), which lets a
split found on the training half look real again on the test half: the double
dipping this check exists to avoid. Each count is split with a beta-binomial
draw (the two-fold Dirichlet-multinomial of their Algorithm 2) at its gene's
overdispersion, estimated as their analysis does with sctransform: a per-gene
negative binomial fit with the cell's total count as offset, on a sample of
genes and cells, smoothed against the gene's mean (Gaussian kernel) and read off
for every gene (``estimate_overdispersion``).

Caveats, also shown in the UI: the halves are independent only as far as the
overdispersion is right (their Theorem 4: a wrong value either way correlates
them); the clusters tested are the training half's, found on half the
molecules, and stand for the checked ones only approximately (the training ARI
says how closely); one split with one seed, so a borderline pair can change
with another split; few distinguishing genes supports a merge but does not
prove one cell type (a small state difference can show in fewer than ten genes
at half depth); and only neighbouring pairs are tested.
"""

from __future__ import annotations

import dataclasses
import hashlib
import logging
import threading
import time
import weakref
from collections import OrderedDict
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp

from scview.core import hvg_sensitivity as hs
from scview.core import leiden, provenance
from scview.core import resolution_sweep as rs
from scview.core.obs_writes import forget_column_results

logger = logging.getLogger(__name__)

EPSILON = 0.5  # training share of each count, as Neufeld et al. use
SEED = 0
THETA_GENES = 2000  # genes and cells the overdispersion is fitted on (sctransform's defaults)
THETA_CELLS = 5000
THETA_MIN, THETA_MAX = 1e-3, 1e6  # THETA_MAX is Poisson in practice
MIN_GENE_CELLS = 5  # a gene fitted needs this many cells with a count
BANDWIDTH_ADJUST = 3.0  # sctransform's bw_adjust, here on Silverman's bandwidth

MAX_NEIGHBOURS = 3  # pairs tested: each cluster with its most connected neighbours
MIN_ADJACENCY = 0.02  # share of the smaller cluster's edge weight joining the two
MIN_TEST_CELLS = 10  # a side with fewer cells is not tested
MAX_CELLS_PER_SIDE = 500
DE_PADJ = 0.05
DE_LOG2FC = 1.0
DE_MIN_PCT = 0.1
MERGE_MAX_GENES = 10
HOST_SHARE = 0.5  # a checked cluster belongs to the training group holding more than this share of it
PART_SHARE = 0.2  # a training cluster holding this share of a checked cluster is one of its parts
MAX_PAIRS = 200
MAX_GROUPS = 100
RESOLUTION_TRIES = 6  # quick counts of the training graph's clusters to match the cluster count
PROBE_ITERATIONS = 2  # Leiden passes for a quick count

# Time (Kang copy, 13,836 cells, 2 cores, measured 2026-10-21, server warm): the
# fit and the split 7-8 s, the training run 18-45 s (Leiden is slower at high
# resolution), matching the cluster count 20-45 s, the pairs 0.15 s each: 83 s at
# resolution 1.0, 100 s at 2.5. The first run after a server start adds ~70 s.
EXTRA_SECONDS_PER_1K_CELLS = 4.5  # on top of one variable-gene setting's estimate
SECONDS_PER_PAIR = 0.15
FIRST_RUN_SECONDS = 70.0  # numba compiling the neighbour search and more, after a server start


class MergeCheckError(ValueError):
    """A check that cannot run on this dataset, or a bad request: a 400."""


@dataclass
class Plan:
    """What is checked: the clustering's codes and names, and how it was made."""

    reference: str  # a column, or the column a sweep resolution would be kept as
    exists: bool  # the reference is a column of the dataset
    codes: np.ndarray
    names: list[str]
    rec: hs.Recorded
    notes: list[str] = field(default_factory=list)
    match_count: bool = True  # re-cluster the training half to the reference's number of clusters


def check(adata) -> None:
    hs.check(adata)
    if hs._counts(adata)[0] is None:
        raise MergeCheckError(
            "The merge check splits the raw counts, which this dataset does not keep "
            "(layers['counts'], .raw or X)."
        )


def plan(adata, reference: str | None = None, resolution: float | None = None,
         method: str | None = None, match_count: bool = True) -> Plan:
    """The clustering to check: a column (default the active clustering), or the
    dataset's graph clustered at ``resolution`` as the sweep shows it."""
    p = _plan(adata, reference, resolution, method)
    p.match_count = bool(match_count)
    return p


def _plan(adata, reference: str | None, resolution: float | None, method: str | None) -> Plan:
    try:
        check(adata)
        if resolution is not None:
            method = method or rs.dataset_method(adata)
            if method not in rs.METHODS:
                raise MergeCheckError(f"Unknown clustering method '{method}'.")
            resolution = rs.clean_resolutions([resolution])[0]
            existing = rs.existing_column(adata, method, resolution)
            if existing is None:
                rs.check(adata)
                labels = rs.cluster(adata, method, resolution)
                base = hs.recorded(adata, None)
                rec = dataclasses.replace(base, reference=rs.column_name(method, resolution),
                                          method=method, resolution=resolution)
                return Plan(rec.reference, False, np.asarray(labels.codes, dtype=np.int64),
                            [str(c) for c in labels.categories], rec, list(rec.notes))
            reference = existing
        rec = hs.recorded(adata, reference)
    except (hs.SensitivityError, rs.SweepError) as exc:
        raise MergeCheckError(str(exc)) from None
    codes, names = hs._reference_codes(adata, rec.reference)
    if len(names) > MAX_GROUPS:
        raise MergeCheckError(f"'{rec.reference}' has {len(names)} groups; the check takes at most {MAX_GROUPS}.")
    return Plan(rec.reference, True, codes, names, rec, list(rec.notes))


def estimate_seconds(n_cells: int, rec: hs.Recorded, n_groups: int) -> float:
    pairs = min(MAX_PAIRS, n_groups * MAX_NEIGHBOURS)
    return round(hs.estimate_seconds(n_cells, 1, rec) + EXTRA_SECONDS_PER_1K_CELLS * n_cells / 1000
                 + SECONDS_PER_PAIR * pairs, 1)


# ---------------------------------------------------------------------------
# Overdispersion and the split
# ---------------------------------------------------------------------------


TOTAL_BINS = 256  # cells grouped by total count for the sums over all cells


class _Fit:
    """What the theta score reads: the counts (sparse), each count's mean, and the
    cells' totals in TOTAL_BINS log-spaced bins (bin weight, mean total)."""

    def __init__(self, X: sp.csr_matrix, totals: np.ndarray):
        self.n_cells, self.n_genes = X.shape
        self.p = np.asarray(X.sum(axis=0)).ravel() / totals.sum()
        coo = X.tocoo()
        self.cols, self.x = coo.col, coo.data.astype(np.float64)
        self.mu_nz = totals[coo.row] * self.p[coo.col]
        edges = np.unique(np.geomspace(totals.min(), totals.max() * (1 + 1e-9), TOTAL_BINS + 1))
        b = np.clip(np.searchsorted(edges, totals, side="right") - 1, 0, edges.size - 2)
        self.w = np.bincount(b, minlength=edges.size - 1).astype(np.float64)
        self.t = np.bincount(b, weights=totals, minlength=edges.size - 1) / np.maximum(self.w, 1)
        self.w, self.t = self.w[self.w > 0], self.t[self.w > 0]


def _theta_score(log_theta: np.ndarray, f: _Fit) -> np.ndarray:
    """d loglik / d theta per gene of a negative binomial with means total_i * p_j,
    at ``exp(log_theta)``. Terms of the zeros are summed over the binned totals
    (the cell's mean only enters through its total)."""
    from scipy.special import digamma

    theta = np.exp(log_theta)
    th = theta[f.cols]
    # Over the counts: psi(x + theta) - psi(theta) - x / (theta + mu).
    nz = np.bincount(f.cols, weights=digamma(f.x + th) - digamma(th) - f.x / (th + f.mu_nz),
                     minlength=f.n_genes)
    tm = theta[None, :] + f.t[:, None] * f.p[None, :]  # bins x genes
    over_all = (f.w[:, None] * (np.log(tm) + theta[None, :] / tm)).sum(axis=0)
    return nz + f.n_cells * (np.log(theta) + 1) - over_all


def _fit_theta(X, totals: np.ndarray) -> np.ndarray:
    """Per-gene maximum-likelihood theta with means ``totals * p_j`` (the cell's
    total count as offset), by bisection on log theta; THETA_MAX when the gene is
    not overdispersed."""
    f = _Fit(sp.csr_matrix(X), np.asarray(totals, dtype=np.float64))
    lo = np.full(f.n_genes, np.log(THETA_MIN))
    hi = np.full(f.n_genes, np.log(THETA_MAX))
    poisson = _theta_score(hi, f) > 0  # still rising at THETA_MAX
    for _ in range(40):  # the bracket shrinks to 2e-11 of its log width
        mid = (lo + hi) / 2
        up = _theta_score(mid, f) > 0
        lo = np.where(up, mid, lo)
        hi = np.where(up, hi, mid)
    theta = np.exp((lo + hi) / 2)
    theta[poisson] = THETA_MAX
    return theta


def _smooth(x: np.ndarray, y: np.ndarray, at: np.ndarray) -> np.ndarray:
    """Nadaraya-Watson Gaussian kernel regression of y on x, evaluated at ``at``
    (clamped to the range of x)."""
    sd = np.std(x)
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    h = 0.9 * min(sd, iqr / 1.34 if iqr > 0 else sd) * x.size ** -0.2 * BANDWIDTH_ADJUST
    h = max(h, 1e-3)
    at = np.clip(at, x.min(), x.max())
    out = np.empty(at.size)
    for i in range(0, at.size, 2048):
        w = np.exp(-0.5 * ((at[i:i + 2048, None] - x[None, :]) / h) ** 2)
        out[i:i + 2048] = (w @ y) / w.sum(axis=1)
    return out


def estimate_overdispersion(counts, rng: np.random.Generator) -> tuple[np.ndarray, dict[str, Any]]:
    """Each gene's negative binomial theta (variance mu + mu^2 / theta), fitted as
    sctransform does: per gene on a sample of genes and cells, then smoothed
    against log10 mean. Genes never seen get THETA_MAX (no count to split)."""
    counts = sp.csr_matrix(counts)
    n, g = counts.shape
    totals_all = np.asarray(counts.sum(axis=1)).ravel()
    mean_all = np.asarray(counts.sum(axis=0)).ravel() / n
    cells = np.sort(rng.choice(n, size=min(n, THETA_CELLS), replace=False))
    sub = counts[cells]
    detected = np.asarray((sub > 0).sum(axis=0)).ravel()
    usable = np.flatnonzero(detected >= MIN_GENE_CELLS)
    theta = np.full(g, THETA_MAX)
    info = {"n_genes_fit": 0, "n_cells_fit": int(cells.size), "median_theta": None, "poisson_share": None}
    if usable.size < 10:
        return theta, info
    # Genes sampled evenly over log mean (inverse density), as sctransform does.
    lm = np.log10(mean_all[usable])
    hist, edges = np.histogram(lm, bins=50)
    dens = hist[np.clip(np.searchsorted(edges, lm, side="right") - 1, 0, hist.size - 1)]
    prob = 1.0 / dens
    genes = usable if usable.size <= THETA_GENES else np.sort(
        rng.choice(usable, size=THETA_GENES, replace=False, p=prob / prob.sum()))
    fit = _fit_theta(sub[:, genes], np.maximum(totals_all[cells], 1.0))
    ok = fit < THETA_MAX
    x_fit, y_fit = np.log10(mean_all[genes]), np.log10(fit)
    seen = mean_all > 0
    if ok.sum() >= 10:
        theta[seen] = 10 ** _smooth(x_fit, y_fit, np.log10(mean_all[seen]))
    info.update(n_genes_fit=int(genes.size), median_theta=round(float(np.median(theta[seen])), 3),
                poisson_share=round(float(1 - ok.mean()), 3))
    return theta, info


def split_counts(counts, theta: np.ndarray, rng: np.random.Generator, epsilon: float = EPSILON
                 ) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    """Negative binomial count splitting (Neufeld et al. 2024, Algorithm 2, two
    folds): each count X goes to the training half as Binomial(X, p), p drawn from
    Beta(epsilon * theta, (1 - epsilon) * theta) for its gene; the rest is the
    test half. theta = THETA_MAX is Poisson splitting (p = epsilon)."""
    M = sp.csr_matrix(counts, dtype=np.float64, copy=True)
    M.sum_duplicates()
    x = np.rint(M.data).astype(np.int64)
    t = theta[M.indices]
    p = np.full(x.size, epsilon)
    nb = t < THETA_MAX
    p[nb] = rng.beta(epsilon * t[nb], (1 - epsilon) * t[nb])
    train = rng.binomial(x, p).astype(np.float32)
    test = (x - train).astype(np.float32)
    shape = M.shape
    a = sp.csr_matrix((train, M.indices.copy(), M.indptr.copy()), shape=shape)
    b = sp.csr_matrix((test, M.indices, M.indptr), shape=shape)
    a.eliminate_zeros()
    b.eliminate_zeros()
    return a, b


# ---------------------------------------------------------------------------
# Pairs and the held-out test
# ---------------------------------------------------------------------------


def adjacency(graph, codes: np.ndarray, k: int) -> np.ndarray:
    """k x k share of edge weight joining two clusters, over the smaller of their
    total edge weights (each cluster's edges to itself included)."""
    keep = codes >= 0
    Z = sp.csr_matrix((np.ones(int(keep.sum())), (np.flatnonzero(keep), codes[keep])),
                      shape=(codes.size, k))
    W = (Z.T @ sp.csr_matrix(graph) @ Z).toarray()
    deg = W.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        A = W / np.minimum(deg[:, None], deg[None, :])
    A = np.nan_to_num(A)
    np.fill_diagonal(A, 0)
    return A


def neighbour_pairs(A: np.ndarray) -> list[tuple[int, int]]:
    """Each cluster with up to MAX_NEIGHBOURS most connected others (at least
    MIN_ADJACENCY), as unordered pairs, most connected first."""
    pairs: set[tuple[int, int]] = set()
    for i in range(A.shape[0]):
        order = np.argsort(-A[i])[:MAX_NEIGHBOURS]
        pairs.update((min(i, int(j)), max(i, int(j))) for j in order if A[i, j] >= MIN_ADJACENCY)
    return sorted(pairs, key=lambda p: -A[p])[:MAX_PAIRS]


def _normalise(counts, target_sum: float) -> sp.csr_matrix:
    import scanpy as sc
    import anndata as ad

    tmp = ad.AnnData(X=sp.csr_matrix(counts, dtype=np.float32))
    sc.pp.normalize_total(tmp, target_sum=target_sum)
    sc.pp.log1p(tmp)
    return tmp.X


def test_pair(log, var_names: pd.Index, in_a: np.ndarray, in_b: np.ndarray) -> dict[str, Any]:
    """Wilcoxon (scanpy's, BH-adjusted) of the cells in_a vs in_b on ``log``,
    genes detected in DE_MIN_PCT of either side; the distinguishing genes."""
    import anndata as ad
    import scanpy as sc

    rows = np.concatenate([in_a, in_b])
    X = log[rows]
    pct_a = np.asarray((X[: in_a.size] > 0).mean(axis=0)).ravel()
    pct_b = np.asarray((X[in_a.size:] > 0).mean(axis=0)).ravel()
    genes = np.flatnonzero(np.maximum(pct_a, pct_b) >= DE_MIN_PCT)
    out = {"n_tested_genes": int(genes.size), "n_genes": 0, "up": [], "down": []}
    if genes.size == 0:
        return out
    tmp = ad.AnnData(X=X[:, genes], obs=pd.DataFrame(
        {"g": pd.Categorical(["a"] * in_a.size + ["b"] * in_b.size)}, index=[str(i) for i in range(rows.size)]))
    tmp.var_names = var_names[genes]
    tmp.uns["log1p"] = {"base": None}
    sc.tl.rank_genes_groups(tmp, "g", groups=["a"], reference="b", method="wilcoxon", n_genes=genes.size)
    r = tmp.uns["rank_genes_groups"]
    names = np.asarray(r["names"]["a"]).astype(str)
    padj = np.asarray(r["pvals_adj"]["a"], dtype=np.float64)
    lfc = np.asarray(r["logfoldchanges"]["a"], dtype=np.float64)
    sig = (padj < DE_PADJ) & (np.abs(lfc) >= DE_LOG2FC)
    up = [str(g) for g in names[sig & (lfc > 0)][:5]]  # names are ranked by score
    down = [str(g) for g in names[sig & (lfc < 0)][::-1][:5]]
    out.update(n_genes=int(sig.sum()), up=up, down=down)
    return out


def _cluster_graph(graph, method: str, resolution: float) -> pd.Categorical:
    import anndata as ad

    tmp = ad.AnnData(obs=pd.DataFrame(index=[str(i) for i in range(graph.shape[0])]))
    tmp.obsp["connectivities"] = graph
    tmp.uns["neighbors"] = {"connectivities_key": "connectivities"}
    return rs.cluster(tmp, method, resolution)


def _count(graph, method: str, resolution: float) -> int:
    """Clusters at ``resolution``: Leiden with PROBE_ITERATIONS passes (its full
    run, until no pass improves, took 6-20 s a time on Kang), Louvain in full."""
    if method == "leiden":
        return int(np.unique(leiden.membership(graph, resolution, n_iterations=PROBE_ITERATIONS)).size)
    return len(_cluster_graph(graph, method, resolution).categories)


def resolution_for_count(graph, method: str, resolution: float, labels: pd.Categorical, target: int,
                         stop: Callable[[], bool] | None = None) -> tuple[float, pd.Categorical]:
    """The resolution (and its labels) whose clustering of ``graph`` has the
    number of clusters closest to ``target``, starting from ``labels`` (the
    clustering at ``resolution``): a bisection on log resolution up to four
    times, or down to a quarter of, ``resolution``, with at most RESOLUTION_TRIES
    quick counts; of equally close ones, the nearest to ``resolution``. The
    chosen resolution is then clustered in full, as the pipeline does."""
    k = len(labels.categories)
    best: tuple[int, float, float] = (abs(k - target), 0.0, resolution)
    lo, hi = (np.log(resolution), np.log(resolution * 4)) if k < target else (np.log(resolution / 4),
                                                                               np.log(resolution))
    for _ in range(RESOLUTION_TRIES):
        if best[0] <= 1:  # a cluster either way is close enough to compare
            break
        hs._check_stop(stop)
        r = round(float(np.exp((lo + hi) / 2)), 3)
        k = _count(graph, method, r)
        key = (abs(k - target), abs(float(np.log(r / resolution))))
        if key < best[:2]:
            best = (*key, r)
        lo, hi = (np.log(r), hi) if k < target else (lo, np.log(r))
    r = best[2]
    hs._check_stop(stop)
    return r, labels if r == resolution else _cluster_graph(graph, method, r)


def _side(rows: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if rows.size <= MAX_CELLS_PER_SIDE:
        return rows
    return np.sort(rng.choice(rows, size=MAX_CELLS_PER_SIDE, replace=False))


def _common(ref_codes: np.ndarray, n_ref: int, train_codes: np.ndarray, k: int) -> np.ndarray:
    """Cells in common: k training x n_ref checked clusters."""
    keep = (ref_codes >= 0) & (train_codes >= 0)
    return np.bincount(train_codes[keep] * n_ref + ref_codes[keep], minlength=k * n_ref).reshape(k, n_ref)


def supported_groups(k: int, tested: list[dict[str, Any]], max_genes: int) -> np.ndarray:
    """Training clusters joined into the groups the held-out half cannot tell
    apart: pairs with fewer than ``max_genes`` distinguishing genes are joined,
    fewest genes first, unless the join would put a tested distinct pair (at
    least ``max_genes``) in one group. A group id per training cluster."""
    parent = list(range(k))
    members = {i: {i} for i in range(k)}
    distinct = {frozenset((r["i"], r["j"])) for r in tested if r["tested"] and r["n_genes"] >= max_genes}

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for r in sorted((r for r in tested if r["tested"] and r["n_genes"] < max_genes), key=lambda r: r["n_genes"]):
        gi, gj = find(r["i"]), find(r["j"])
        if gi == gj or any(frozenset((a, b)) in distinct for a in members[gi] for b in members[gj]):
            continue
        parent[gj] = gi
        members[gi] |= members.pop(gj)
    return np.array([find(i) for i in range(k)])


@dataclass
class _Stage:
    """The expensive part of a run, kept so another threshold is a re-read."""

    train_codes: np.ndarray
    train_names: list[str]
    train_resolution: float
    comparison: dict[str, Any]
    A_ref: np.ndarray
    tested: list[dict[str, Any]]
    theta_info: dict[str, Any]
    seconds: dict[str, float]


def summarise(p: Plan, s: _Stage, max_genes: int = MERGE_MAX_GENES) -> dict[str, Any]:
    """The checked clustering's pairs from the training pairs' held-out tests.

    Each checked cluster belongs to the group of training clusters (see
    :func:`supported_groups`) holding more than HOST_SHARE of its cells. Two
    neighbouring checked clusters in one group are a merge candidate when their
    main training clusters differ (a held-out test joins them); with the same
    main training cluster they were never split on the training half and are
    reported as "together", untested. In different groups they are kept apart,
    with the most distinct tested training pair between the groups.
    """
    n_ref, k = len(p.names), len(s.train_names)
    common = _common(p.codes, n_ref, s.train_codes, k)
    group = supported_groups(k, s.tested, max_genes)
    n_groups = int(np.unique(group).size)
    by_group = np.zeros((n_groups, n_ref), dtype=np.int64)
    gid = {g: i for i, g in enumerate(np.unique(group))}
    gix = np.array([gid[g] for g in group])
    np.add.at(by_group, gix, common)
    ref_size = np.maximum(common.sum(axis=0), 1)
    g_of = np.where(by_group.max(axis=0) > HOST_SHARE * ref_size, by_group.argmax(axis=0), -1)
    main = common.argmax(axis=0)  # each checked cluster's main training cluster

    # The training clusters holding a real part (PART_SHARE) of each checked cluster.
    parts = common >= PART_SHARE * ref_size[None, :]

    def evidence(a: int, b: int, ga: int, gb: int) -> list[dict[str, Any]]:
        """The tested training pairs read for checked clusters a and b: those
        joining a part of a to a part of b, else any between (or within) their
        groups."""
        in_groups = [r for r in s.tested if r["tested"] and {gix[r["i"]], gix[r["j"]]} <= {ga, gb}
                     and (ga == gb or gix[r["i"]] != gix[r["j"]])]
        direct = [r for r in in_groups if (parts[r["i"], a] and parts[r["j"], b])
                  or (parts[r["i"], b] and parts[r["j"], a])]
        return direct or in_groups

    pairs = []
    for a, b in neighbour_pairs(s.A_ref):
        out: dict[str, Any] = {"a": p.names[a], "b": p.names[b], "adjacency": round(float(s.A_ref[a, b]), 4),
                               "n_cells_a": int(ref_size[a]), "n_cells_b": int(ref_size[b]),
                               "n_genes": None, "up": [], "down": [], "train_pairs": [], "candidate": False}
        ga, gb = int(g_of[a]), int(g_of[b])
        rows: list[dict[str, Any]] = []
        if ga < 0 or gb < 0:
            out["evidence"] = "untested"
        elif ga == gb and main[a] == main[b]:
            out["evidence"] = "together"
        else:
            rows = evidence(a, b, ga, gb)
            out["evidence"] = "held_out" if rows else "untested"
            out["candidate"] = ga == gb and bool(rows)
        if rows:
            worst = max(rows, key=lambda r: r["n_genes"])  # the most distinct of them
            out.update(n_genes=worst["n_genes"], up=worst["up"], down=worst["down"],
                       train_pairs=[f"{r['a']}+{r['b']}" for r in rows])
        pairs.append(out)
    order = {"held_out": 0, "together": 1, "untested": 2}
    pairs.sort(key=lambda r: (not r["candidate"], order[r["evidence"]],
                              r["n_genes"] if r["n_genes"] is not None else 0, -r["adjacency"]))
    return {
        "reference": p.reference,
        "reference_exists": p.exists,
        "method": p.rec.method,
        "resolution": p.rec.resolution,
        "train_resolution": s.train_resolution,
        "match_count": p.match_count,
        "n_reference_clusters": int(np.unique(p.codes[p.codes >= 0]).size),
        "epsilon": EPSILON,
        "seed": SEED,
        "max_genes": int(max_genes),
        "overdispersion": s.theta_info,
        "train": {
            "n_clusters": k, "ari": s.comparison["ari"], "ami": s.comparison["ami"],
            "sizes": s.comparison["sizes"], "n_groups": n_groups,
            "clusters": [{"cluster": s.train_names[i], "n_cells": int(common[i].sum()),
                          "group": int(gix[i])} for i in range(k)],
        },
        "groups": [[p.names[r] for r in np.flatnonzero(g_of == g)] for g in range(n_groups)],
        "pairs": pairs,
        "train_pairs": [{k_: v for k_, v in r.items() if k_ not in ("i", "j")} for r in s.tested],
        "n_candidates": int(sum(r["candidate"] for r in pairs)),
        "thresholds": {"padj": DE_PADJ, "log2fc": DE_LOG2FC, "min_pct": DE_MIN_PCT,
                       "merge_max_genes": int(max_genes), "max_cells_per_side": MAX_CELLS_PER_SIDE,
                       "min_adjacency": MIN_ADJACENCY, "host_share": HOST_SHARE},
        "seconds": s.seconds,
    }


# Stages kept for re-reading at another threshold, across datasets.
_stages: OrderedDict[str, _Stage] = OrderedDict()
_STAGE_CACHE = 4


def _stage_key(adata, p: Plan) -> str:
    h = hashlib.sha1(hs.fingerprint(adata, p.rec).encode())
    h.update(repr((p.reference, p.rec.method, p.rec.resolution, p.match_count, SEED, EPSILON)).encode())
    h.update(np.ascontiguousarray(p.codes).tobytes())
    h.update("\x00".join(p.names).encode())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


def iter_merge_check(adata, p: Plan, stop: Callable[[], bool] | None = None,
                     max_genes: int = MERGE_MAX_GENES) -> Iterator[dict[str, Any]]:
    """Split, re-cluster the training half, test neighbouring pairs on the test half.

    Yields a header, ``{"step": ...}`` progress items, then the result (with
    ``"pairs"``). ``adata`` is not changed. ``stop`` is checked between steps.
    The same check again (another ``max_genes``) re-reads the kept tests.
    """
    rec = p.rec
    key = _stage_key(adata, p)
    cached = _stages.get(key)
    yield {
        "n_cells": int(adata.n_obs),
        "reference": p.reference,
        "reference_exists": p.exists,
        "recorded": rec.to_dict(),
        "estimate_seconds": 0.0 if cached else estimate_seconds(adata.n_obs, rec, len(p.names)),
        "cached": cached is not None,
    }
    if cached is not None:
        _stages.move_to_end(key)
        yield summarise(p, cached, max_genes)
        return
    t0 = time.perf_counter()
    rng = np.random.default_rng(SEED)
    counts, _ = hs._counts(adata)
    yield {"step": "overdispersion", "label": "Estimating each gene's overdispersion"}
    theta, theta_info = estimate_overdispersion(counts, rng)
    hs._check_stop(stop)
    yield {"step": "split", "label": "Splitting the counts into two halves"}
    train, test = split_counts(counts, theta, rng)
    del counts
    hs._check_stop(stop)
    yield {"step": "train", "label": "Re-running the pipeline on the training half"}
    t_train = time.perf_counter()
    inp = hs.inputs_from_counts(adata, rec, train)
    _, labels, graph = hs.run_setting_graph(inp, rec.current, rec, stop)
    del inp, train
    train_resolution = rec.resolution
    n_ref = len(p.names)
    n_ref_used = int(np.unique(p.codes[p.codes >= 0]).size)
    if p.match_count and len(labels.categories) != n_ref_used:
        yield {"step": "resolution", "label": f"Finding the resolution that gives {n_ref_used} clusters on the "
                                              "training half"}
        train_resolution, labels = resolution_for_count(graph, rec.method, rec.resolution, labels, n_ref_used,
                                                         stop)
    train_seconds = time.perf_counter() - t_train
    hs._check_stop(stop)

    k = len(labels.categories)
    train_codes = np.asarray(labels.codes, dtype=np.int64)
    train_names = [str(c) for c in labels.categories]
    A_train = adjacency(graph, train_codes, k)
    train_pairs = neighbour_pairs(A_train)
    yield {"step": "test", "label": f"Testing {len(train_pairs)} neighbouring pairs on the held-out half"}
    t_test = time.perf_counter()
    log = _normalise(test, rec.target_sum)
    del test
    tested: list[dict[str, Any]] = []
    for i, j in train_pairs:
        hs._check_stop(stop)
        rows_i, rows_j = np.flatnonzero(train_codes == i), np.flatnonzero(train_codes == j)
        row: dict[str, Any] = {
            "i": i, "j": j, "a": train_names[i], "b": train_names[j],
            "adjacency": round(float(A_train[i, j]), 4),
            "n_cells_a": int(rows_i.size), "n_cells_b": int(rows_j.size),
        }
        if min(rows_i.size, rows_j.size) < MIN_TEST_CELLS:
            row.update(tested=False, n_genes=None, up=[], down=[], n_tested_genes=0)
        else:
            row.update(tested=True, **test_pair(log, adata.var_names, _side(rows_i, rng), _side(rows_j, rng)))
        tested.append(row)
    del log
    stage = _Stage(
        train_codes=train_codes, train_names=train_names, train_resolution=train_resolution,
        comparison=hs.compare(p.codes, p.names, labels), A_ref=adjacency(graph, p.codes, n_ref),
        tested=tested, theta_info=theta_info,
        seconds={"train": round(train_seconds, 1), "test": round(time.perf_counter() - t_test, 1),
                 "total": round(time.perf_counter() - t0, 1)},
    )
    _stages[key] = stage
    while len(_stages) > _STAGE_CACHE:
        _stages.popitem(last=False)
    yield summarise(p, stage, max_genes)


def run(adata, p: Plan, max_genes: int = MERGE_MAX_GENES) -> dict[str, Any]:
    """:func:`iter_merge_check`'s result."""
    *_, result = iter_merge_check(adata, p, max_genes=max_genes)
    return result


# ---------------------------------------------------------------------------
# Saving a merged clustering
# ---------------------------------------------------------------------------


def target_column(reference: str) -> str:
    return f"{reference}_merged"


def merged_labels(values: pd.Series, merges: list[tuple[str, str]]) -> pd.Categorical:
    """``values`` with each merged set of groups relabelled ``a+b`` (members in the
    column's order); groups keep their order, a set at its first member's place."""
    cat = values if isinstance(values.dtype, pd.CategoricalDtype) else values.astype(str).astype("category")
    order = [str(c) for c in cat.cat.categories]
    parent = {g: g for g in order}

    def find(g: str) -> str:
        while parent[g] != g:
            parent[g] = parent[parent[g]]
            g = parent[g]
        return g

    for a, b in merges:
        ra, rb = find(a), find(b)
        if ra != rb:
            first, second = sorted((ra, rb), key=order.index)
            parent[second] = first
    members: dict[str, list[str]] = {}
    for g in order:
        members.setdefault(find(g), []).append(g)
    name = {g: "+".join(members[find(g)]) for g in order}
    new_order = list(dict.fromkeys(name[g] for g in order))
    mapped = cat.astype(str).map(name).where(cat.notna())
    return pd.Categorical(mapped, categories=new_order)


def clean_merges(names: list[str], pairs: list[list[str]] | None) -> list[tuple[str, str]]:
    if not pairs:
        raise MergeCheckError("Pick at least one pair of clusters to merge.")
    out = []
    for pr in pairs:
        if len(pr) != 2 or pr[0] == pr[1]:
            raise MergeCheckError("Each merge is two different clusters.")
        a, b = str(pr[0]), str(pr[1])
        for g in (a, b):
            if g not in names:
                raise MergeCheckError(f"No cluster '{g}' in this clustering.")
        out.append((a, b))
    return out


def save(adata, p: Plan, pairs: list[list[str]] | None) -> dict[str, Any]:
    """Write ``<reference>_merged``: the reference with the pairs merged. A sweep
    resolution not yet saved is kept first (as "Keep this one" does). The active
    clustering does not change."""
    merges = clean_merges(p.names, pairs)
    if not p.exists:
        rs.keep(adata, p.rec.method, p.rec.resolution)
    column = target_column(p.reference)
    labels = merged_labels(adata.obs[p.reference], merges)
    old = adata.obs[p.reference].astype(str).where(adata.obs[p.reference].notna())
    mapping = {str(k): str(v) for k, v in zip(old, labels) if isinstance(k, str) and isinstance(v, str)}
    adata.obs[column] = labels
    forget_column_results(adata, column)  # left by an earlier merge of that name
    try:
        provenance.record_step(
            adata, step="merge_clusters", tool="scView merge check (count splitting)",
            params={"column": p.reference, "target": column, "merges": [list(m) for m in merges],
                    "mapping": mapping, "categories": [str(c) for c in labels.categories],
                    "from": "merge suggestions (count splitting)"},
        )
    except Exception as e:  # provenance must never block the result
        logger.warning("Could not record merge_clusters provenance: %s", e)
    return {"column": column, "n_clusters": int(adata.obs[column].nunique()),
            "groups": [str(c) for c in adata.obs[column].cat.categories]}


# ---------------------------------------------------------------------------
# Co-pilot grounding: the last check run on each loaded dataset
# ---------------------------------------------------------------------------

_last: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_last_lock = threading.Lock()


def remember(adaptor, result: dict[str, Any]) -> None:
    """Keep the last result for the co-pilot, as long as the loaded dataset is."""
    try:
        with _last_lock:
            _last[adaptor] = result
    except TypeError:  # not weak-referenceable (a test double)
        pass


def grounding_lines(adaptor, obs: pd.DataFrame) -> tuple[list[str], str | None]:
    """For the co-pilot: the last merge check's summary and its candidate pairs."""
    try:
        with _last_lock:
            r = _last.get(adaptor)
    except TypeError:
        return [], None
    if r is None or (r.get("reference_exists") and r["reference"] not in obs.columns):
        return [], None
    t = r["thresholds"]
    cands = [x for x in r["pairs"] if x["candidate"]]
    held = [x for x in r["pairs"] if x["evidence"] == "held_out"]
    detail = (f"'{r['reference']}' ({r['method']} at {r['resolution']}): training half re-clustered into "
              f"{r['train']['n_clusters']} clusters at resolution {r['train_resolution']} (ARI "
              f"{r['train']['ari']:.2f} with '{r['reference']}'); {len(held)} neighbouring pairs tested on the "
              f"held-out half; {len(cands)} merge candidates (fewer than {t['merge_max_genes']} distinguishing "
              "genes)")
    lines = ["\n## Merge check (last run; count splitting, held-out Wilcoxon)",
             f"- {detail} [result:merge_check]."]
    for x in cands[:8]:
        genes = f"{x['n_genes']} gene{'' if x['n_genes'] == 1 else 's'}"
        lines.append(f"- merge candidate: {x['a']} + {x['b']}: {genes} at adj. p < {t['padj']} "
                     "and >= 2-fold on held-out counts")
    apart = sorted((x for x in held if not x["candidate"]), key=lambda x: -x["adjacency"])[:3]
    for x in apart:
        lines.append(f"- kept apart: {x['a']} vs {x['b']}: {x['n_genes']} distinguishing genes "
                     f"(e.g. {', '.join((x['up'] + x['down'])[:4])})")
    lines.append("- Caveat: few held-out genes supports a merge but does not prove one cell type; one split, "
                 "and the tested clusters are the training half's, standing for these.")
    return lines, detail
