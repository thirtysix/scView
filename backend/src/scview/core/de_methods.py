"""Cell-level DE tests side by side: Wilcoxon, Welch's t-test, logistic regression.

The cell-level DE (selection vs rest, and between groups) runs one of three of
scanpy's ``rank_genes_groups`` tests on the same log-normalised matrix and the
same detection prefilter (core/de_prefilter.py):

- ``wilcoxon`` (default): the rank-sum test; statistic = its z-score.
- ``t-test``: scanpy's ``t-test``, which is Welch's (unequal variances, each
  group's own n). Not ``t-test_overestim_var``, which scanpy uses for markers
  and which inflates the group's variance by using the rest's n.
- ``logreg``: a logistic regression of group membership on every tested gene at
  once (scanpy's ``logreg``: scikit-learn's ``LogisticRegression`` with its
  defaults, L2 penalty, C = 1). It gives one coefficient per gene and no
  p-value: the coefficients are joint, penalised and depend on each gene's
  scale, so there is no per-gene null to test them against. Results carry
  ``pval = None``; nothing here makes one up.

MAST (a hurdle model) is R-only and not offered.

``agreement`` says how far the tests agree on one comparison: the overlap
(Jaccard) of their top-N genes on each side, the Spearman correlation of their
per-gene statistics, and the genes one test ranks in its top N and no other
does. On Kang 2018 PBMCs (1% prefilter, top 50): CD14 Mono vs the rest,
Wilcoxon vs Welch Jaccard 0.96 up / 0.67 down, Spearman 0.98, but either vs
logreg 0.18 / 0.02, Spearman 0.13 (logreg's strongest "down" genes are PPBP,
HBB, PF4: the platelets and red cells that set the rest apart); STIM vs CTRL
within CD14 Mono, Wilcoxon vs Welch 0.89 / 0.69, 0.97, vs logreg 0.54 / 0.33,
0.42. The two univariate tests agree; logreg answers a different question
(which genes jointly tell the groups apart).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
import scipy.sparse as sp

logger = logging.getLogger(__name__)

METHODS = ("wilcoxon", "t-test", "logreg")
LABELS = {
    "wilcoxon": "Wilcoxon rank-sum",
    "t-test": "Welch t-test",
    "logreg": "Logistic regression",
}
STAT_LABELS = {"wilcoxon": "z-score", "t-test": "t statistic", "logreg": "coefficient"}
HAS_PVALUES = {"wilcoxon": True, "t-test": True, "logreg": False}
# Methods asked for that no backend here can run, with the reason.
UNAVAILABLE = {"MAST": "MAST is an R package (a hurdle model); scView's backend runs Python only."}

TOP_N = 50
MAX_TOP_N = 500
UNIQUE_PER_METHOD = 10  # genes per method and side in the "only this test" table
# scikit-learn's default, as scanpy: Kang CD14 Mono vs the rest (13,836 cells
# x 8,086 genes) converges in 43 iterations, 0.9 s on 2 cores.
LOGREG_MAX_ITER = 100


class MethodError(ValueError):
    """A test that could not run on these cells."""


@dataclass
class MethodResult:
    method: str
    genes: np.ndarray  # tested genes, in the matrix's order
    scores: np.ndarray  # the statistic per gene (NaN when undefined)
    logfoldchanges: np.ndarray
    pvals: np.ndarray | None  # None for logreg
    pvals_adj: np.ndarray | None
    seconds: float


def _expm1(uns: dict):
    base = (uns.get("log1p") or {}).get("base")
    if base is None:
        return np.expm1
    return lambda x: np.expm1(x * np.log(base))


def _column_means(M, rows: np.ndarray) -> np.ndarray:
    sub = M[rows]
    return np.asarray(sub.mean(axis=0), dtype=np.float64).ravel()


def logfoldchanges(M, in_a: np.ndarray, uns: dict) -> np.ndarray:
    """log2 fold change of the ``in_a`` cells over the others from mean
    log-normalised values, exactly as scanpy's rank_genes_groups takes it."""
    expm1 = _expm1(uns)
    in_a = np.asarray(in_a, dtype=bool)
    mean_a, mean_b = _column_means(M, in_a), _column_means(M, ~in_a)
    return np.log2((expm1(mean_a) + 1e-9) / (expm1(mean_b) + 1e-9))


def _ranked_test(test, method: str) -> MethodResult:
    """Wilcoxon or Welch t-test of ``test`` (obs ``_de_group`` = selection / rest):
    scanpy's, by blocks of genes (``core/rank_genes.py``, bit for bit). ``test`` holds
    only the two sides' cells, so between groups the ranks are within A and B, as
    scanpy's ``reference=`` mode takes them."""
    from scview.core import rank_genes

    t0 = time.perf_counter()
    rank_genes.rank_genes_groups(
        test, "_de_group", groups=["selection"], reference="rest",
        method=method, use_raw=False,
        n_genes=test.n_vars,  # every tested gene, not the top-scoring ones
    )
    rgg = test.uns.pop("rank_genes_groups")
    names = np.asarray(rgg["names"]["selection"]).astype(str)
    # Back into the matrix's gene order (scanpy sorts by score), so results of
    # different tests line up gene for gene.
    genes = np.asarray(test.var_names).astype(str)
    idx = pd.Index(names).get_indexer(genes)

    def col(field: str) -> np.ndarray:
        return np.asarray(rgg[field]["selection"], dtype=np.float64)[idx]

    return MethodResult(
        method=method, genes=genes, scores=col("scores"), logfoldchanges=col("logfoldchanges"),
        pvals=col("pvals"), pvals_adj=col("pvals_adj"), seconds=time.perf_counter() - t0,
    )


def _logreg(test) -> MethodResult:
    """One logistic regression of selection-vs-rest on every tested gene; the
    coefficient is positive when the gene is higher in the selection."""
    import warnings

    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression

    t0 = time.perf_counter()
    in_a = (test.obs["_de_group"] == "selection").to_numpy()
    X = test.X
    if sp.issparse(X):
        X = sp.csr_matrix(X)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)  # logged below instead
        clf = LogisticRegression(max_iter=LOGREG_MAX_ITER)
        clf.fit(X, in_a.astype(np.int8))
    if clf.n_iter_[0] >= LOGREG_MAX_ITER:
        logger.warning("logreg DE stopped at %d iterations before converging", LOGREG_MAX_ITER)
    return MethodResult(
        method="logreg", genes=np.asarray(test.var_names).astype(str),
        scores=np.asarray(clf.coef_[0], dtype=np.float64),
        logfoldchanges=logfoldchanges(test.X, in_a, test.uns),
        pvals=None, pvals_adj=None, seconds=time.perf_counter() - t0,
    )


def run(test, method: str) -> MethodResult:
    """``method`` on ``test``: an AnnData of the tested genes with obs
    ``_de_group`` (categories rest, selection) and the log1p uns of the data."""
    if method not in METHODS:
        raise MethodError(f"Unknown DE method '{method}'.")
    if method == "logreg":
        return _logreg(test)
    return _ranked_test(test, method)


# ---------------------------------------------------------------------------
# Agreement between tests
# ---------------------------------------------------------------------------


def _ranked(genes: np.ndarray, scores: np.ndarray, descending: bool) -> list[str]:
    """Genes with a finite score, strongest first (ties by name: repeatable)."""
    ok = np.isfinite(scores)
    g, s = genes[ok], scores[ok]
    order = np.lexsort((g, -s if descending else s))
    return g[order].tolist()


def _jaccard(a: set, b: set) -> float | None:
    union = a | b
    return round(len(a & b) / len(union), 4) if union else None


def _spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    from scipy import stats

    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.ptp(x[ok]) == 0 or np.ptp(y[ok]) == 0:
        return None
    rho = stats.spearmanr(x[ok], y[ok]).statistic
    return round(float(rho), 4) if np.isfinite(rho) else None


def agreement(results: list[MethodResult], top_n: int = TOP_N) -> dict:
    """How far ``results`` (one comparison, tested on the same genes) agree.

    Per pair: the Jaccard index of the top ``top_n`` genes higher in the first
    group (by statistic, highest first) and of the top ``top_n`` higher in the
    second, and the Spearman correlation of the statistics over every gene.
    ``unique``: genes in one test's top ``top_n`` of a side that no other test
    has there, with each test's rank of the gene on that side (1 = top).
    """
    ranks: dict[str, dict[str, dict[str, int]]] = {}
    tops: dict[str, dict[str, set[str]]] = {}
    for r in results:
        ranks[r.method] = {}
        tops[r.method] = {}
        for side, desc in (("up", True), ("down", False)):
            order = _ranked(r.genes, r.scores, desc)
            ranks[r.method][side] = {g: i + 1 for i, g in enumerate(order)}
            tops[r.method][side] = set(order[:top_n])

    pairs = []
    for a, b in combinations(results, 2):
        if not np.array_equal(a.genes, b.genes):
            raise MethodError("The tests did not run on the same genes.")
        up_a, up_b = tops[a.method]["up"], tops[b.method]["up"]
        dn_a, dn_b = tops[a.method]["down"], tops[b.method]["down"]
        pairs.append({
            "a": a.method, "b": b.method,
            "jaccard_up": _jaccard(up_a, up_b), "shared_up": len(up_a & up_b),
            "jaccard_down": _jaccard(dn_a, dn_b), "shared_down": len(dn_a & dn_b),
            "spearman": _spearman(a.scores, b.scores),
        })

    unique = []
    for r in results:
        for side in ("up", "down"):
            others = set().union(*(tops[o.method][side] for o in results if o is not r))
            only = sorted(tops[r.method][side] - others, key=lambda g: ranks[r.method][side][g])
            for g in only[:UNIQUE_PER_METHOD]:
                unique.append({
                    "gene": g, "method": r.method, "side": side,
                    "ranks": {o.method: ranks[o.method][side].get(g) for o in results},
                })
    return {"top_n": top_n, "pairs": pairs, "unique": unique}
