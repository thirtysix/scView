"""Genes whose expression changes along a pseudotime.

Each gene's log-normalised expression (``activity.lognorm_matrix``: ``.raw``
after the pipeline) is fitted with a natural cubic spline of pseudotime and
compared with a constant by an F-test, every gene at once: the spline basis is
orthonormalised (QR), so a gene's fitted sum of squares is the squared norm of
``Q.T @ x`` and the whole test is one sparse product (paul15, 2,730 cells x
3,451 genes: 0.1 s).

The spline runs along the cells' pseudotime *order* (ranks scaled to 0..1), not
the raw values. DPT has long tails (paul15 from its MEPs: 95% of cells below
0.3, the 30 dendritic cells near 0.9), and on raw values the last spline piece
fitted that small far-out group on its own, so its markers (H2-Eb1, Cd74) led
the list. Along the order each cell counts once. The knots sit at quantiles of
the order, as in tradeSeq, but the fit is least squares on log values, not
tradeSeq's negative-binomial GAM: a screening statistic. Cells are not
independent replicates and p-values come out tiny on thousands of cells, so
genes are ranked by F and the p-values are indicative only.

Trends for the top genes are binned by equal cell counts along pseudotime, so a
sparse tail gets as many cells per bin as the dense start.

Branch-specific genes (``rank_by="branch_specific"``, ``branch_genes``). Ranked within one
branch, every branch leads with the same genes: the shared early switch (pancreas: Spp1, Cpe,
Aplp1 on all four hormone branches; Gcg not in the alpha branch's top 200). Instead each gene
is fitted on the cells of the whole PAGA tree along their shared pseudotime order, once with
one curve for every cell (null) and once with that curve plus a second spline that only the
branch's own cells take (its groups not shared with another root-to-end path), and the two are
compared by an F-test. The idea is tradeSeq's patternTest (Van den Berge 2020): do the smoothers
of two lineages differ anywhere along pseudotime. It differs from it: least squares on log
values (not a negative-binomial GAM), this branch against the other branches pooled (not every
pair of lineages), cells of shared groups fitted once by the common curve (not assigned to
lineages by weights), and genes ranked by F among those HIGHER on the branch (the mean of the
branch's own spline over its cells: ``branch_diff``), so a list names the branch's markers, not
the other branches' (which a two-sided test puts at the top of every list).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats

N_KNOTS = 5  # boundary + 3 interior: 5 basis functions, 4 df beyond the mean
MIN_CELLS = 30
MIN_DETECTED = 10  # cells a gene must be detected in to be tested ...
MIN_DETECTED_FRACTION = 0.01  # ... and this share of the cells used
MIN_CELLS_PER_BIN = 10
DEFAULT_TOP = 50
MAX_TOP = 200
DEFAULT_BINS = 40
MAX_BINS = 100
EDGE = 0.1  # a peak within this share of either end: "rising" / "falling"
_CHUNK = 2048  # genes per block when the matrix is dense

METHOD = (
    f"natural cubic spline of pseudotime order ({N_KNOTS} knots), "
    "F-test against a constant, on log-normalised expression"
)
RANK_BY = ("trend", "branch_specific")
BRANCH_METHOD = (
    f"branch against the other branches of the PAGA tree: one spline of the shared pseudotime order for all "
    f"cells ({N_KNOTS} knots) against that spline plus a spline only the branch's own cells take, F-test "
    "(after tradeSeq's patternTest), on log-normalised expression; genes higher on the branch, by F"
)


class TrendError(ValueError):
    """A request that cannot be served (too few cells, unknown group): a 400."""


def natural_spline_basis(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Natural cubic spline basis at ``x``: n x K for K knots (1, x and K-2
    truncated-power terms), cubic between the knots and linear beyond the
    boundary ones (Hastie, Tibshirani & Friedman, ESL eqs. 5.4-5.5)."""
    x = np.asarray(x, dtype=np.float64)
    last = knots[-1]

    def d(k: int) -> np.ndarray:
        return (np.clip(x - knots[k], 0, None) ** 3 - np.clip(x - last, 0, None) ** 3) / (
            last - knots[k]
        )

    tail = d(len(knots) - 2)
    return np.column_stack([np.ones_like(x), x] + [d(k) - tail for k in range(len(knots) - 2)])


def _moments(X, Q: np.ndarray):
    """Per gene: sum, sum of squares, ``Q.T @ x`` (p x genes) and the number of
    cells it is detected in."""
    if sp.issparse(X):
        X = sp.csr_matrix(X, dtype=np.float64)
        detected = np.bincount(X.indices[X.data != 0], minlength=X.shape[1])
        return (
            np.asarray(X.sum(axis=0)).ravel(),
            np.asarray(X.multiply(X).sum(axis=0)).ravel(),
            np.asarray((X.T @ Q).T),
            detected,
        )
    n_genes = X.shape[1]
    s1, s2 = np.empty(n_genes), np.empty(n_genes)
    qtx, detected = np.empty((Q.shape[1], n_genes)), np.empty(n_genes, dtype=np.int64)
    for a in range(0, n_genes, _CHUNK):  # float64 copies of a block, not the whole matrix
        block = np.asarray(X[:, a:a + _CHUNK], dtype=np.float64)
        s1[a:a + _CHUNK] = block.sum(axis=0)
        s2[a:a + _CHUNK] = (block * block).sum(axis=0)
        qtx[:, a:a + _CHUNK] = Q.T @ block
        detected[a:a + _CHUNK] = np.count_nonzero(block, axis=0)
    return s1, s2, qtx, detected


def _columns(X, cols: np.ndarray) -> np.ndarray:
    sub = X[:, cols]
    return np.asarray(sub.toarray() if sp.issparse(sub) else sub, dtype=np.float64)


def _num(v: float, digits: int = 4) -> float | None:
    return round(float(v), digits) if math.isfinite(v) else None


def _sig(v: float) -> float | None:
    """A p-value to 4 significant figures (they span hundreds of decades)."""
    return float(f"{v:.4g}") if math.isfinite(v) else None


BINNING = "equal cell counts along pseudotime (quantile bins); each bin at its median pseudotime"


def equal_count_bins(pt: np.ndarray, n_bins: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bins of (nearly) equal cell counts along pseudotime, at most ``n_bins`` and at
    least ``MIN_CELLS_PER_BIN`` cells each (2 bins at the least): each cell's bin, each
    bin's median pseudotime, each bin's cell count. Bins by value put 9,797 of 9,815 cells
    in the first of 20 when a few cells sat far out (Pijuan-Sala's erythroid cells)."""
    pt = np.asarray(pt, dtype=np.float64)
    n = len(pt)
    nb = int(min(n_bins, max(2, n // MIN_CELLS_PER_BIN), n))
    bin_of = np.empty(n, dtype=np.int64)
    bin_of[np.argsort(pt, kind="stable")] = np.arange(n) * nb // n
    counts = np.bincount(bin_of, minlength=nb)
    order = np.argsort(bin_of, kind="stable")
    bounds = np.r_[0, np.cumsum(counts)]
    by_bin = pt[order]
    bin_pt = np.array([np.median(by_bin[bounds[b]:bounds[b + 1]]) for b in range(nb)])
    return bin_of, bin_pt, counts


def cells_used(adata, column: str, groups_column: str | None = None,
               groups: list[str] | None = None) -> tuple[np.ndarray, dict[str, Any] | None]:
    """Indices of the cells with a finite pseudotime (in ``groups`` of
    ``groups_column`` when given), and the restriction as recorded."""
    pt = pd.to_numeric(adata.obs[column], errors="coerce").to_numpy(dtype=np.float64)
    use = np.isfinite(pt)
    restrict = None
    if groups_column is not None:
        if groups_column not in adata.obs.columns:
            raise TrendError(f"Column '{groups_column}' not found.")
        col = adata.obs[groups_column]
        labels = col.astype(str).where(col.notna())
        wanted = list(dict.fromkeys(groups or []))
        if not wanted:
            raise TrendError("Pick at least one group to restrict the cells to.")
        present = set(labels.dropna())
        missing = [g for g in wanted if g not in present]
        if missing:
            raise TrendError(f"No cells have {groups_column} = {', '.join(map(repr, missing))}.")
        use &= labels.isin(wanted).to_numpy()
        restrict = {"column": groups_column, "groups": wanted}
    return np.flatnonzero(use), restrict


def trend_genes(
    adata,
    column: str,
    *,
    n_top: int = DEFAULT_TOP,
    n_bins: int = DEFAULT_BINS,
    groups_column: str | None = None,
    groups: list[str] | None = None,
    rank_by: str = "trend",
) -> dict[str, Any]:
    """The ``n_top`` genes that change most along the pseudotime ``column``.

    Cells without a finite pseudotime are left out, and with ``groups_column``
    so are cells outside ``groups`` (a branch: the groups on a PAGA path). Each
    gene comes with its test (``f_stat``, ``pval``, BH ``pval_adj``, ``r2``),
    binned ``mean`` expression and the spline's ``fitted`` mean in the same
    bins, the fitted ``amplitude`` (max - min, log units), the bin
    pseudotime of the fitted peak and a ``pattern``: rising, falling (peak at the
    start) or transient. ``rank_by="branch_specific"``: the branch's genes against the
    other branches instead (``branch_genes``).
    """
    from scview.core.activity import lognorm_source

    if rank_by not in RANK_BY:
        raise TrendError(f"rank_by is one of {', '.join(RANK_BY)}.")
    if rank_by == "branch_specific":
        return branch_genes(adata, column, n_top=n_top, n_bins=n_bins, groups_column=groups_column, groups=groups)
    idx, restrict = cells_used(adata, column, groups_column, groups)
    n = len(idx)
    where = " in those groups" if restrict else ""
    if n < MIN_CELLS:
        raise TrendError(
            f"Only {n} cells have a pseudotime{where}; at least {MIN_CELLS} are needed."
        )
    pt = pd.to_numeric(adata.obs[column], errors="coerce").to_numpy(dtype=np.float64)[idx]
    order = (stats.rankdata(pt) - 1) / (n - 1)
    knots = np.unique(np.quantile(order, np.linspace(0, 1, N_KNOTS)))
    if len(knots) < 3:
        raise TrendError(f"Too few distinct pseudotime values{where} to fit a trend.")
    Q, _ = np.linalg.qr(natural_spline_basis(order, knots))
    p = Q.shape[1]

    try:
        choice = lognorm_source(adata)
    except ValueError:
        raise TrendError(
            "No log-normalised expression or raw counts to test: run normalization and "
            "log transform first."
        ) from None
    names, source = np.asarray(choice.names, dtype=object), choice.source
    X = choice.rows(idx)  # a backed dataset's: these cells read (the endpoint holds the room)
    s1, s2, qtx, detected = _moments(X, Q)

    tss = s2 - s1 ** 2 / n
    ess = np.maximum((qtx ** 2).sum(axis=0) - s1 ** 2 / n, 0.0)  # Q spans the constant
    rss = np.maximum(tss - ess, 0.0)
    min_detected = max(MIN_DETECTED, math.ceil(MIN_DETECTED_FRACTION * n))
    tested = np.flatnonzero((detected >= min_detected) & (tss > 1e-12 * np.maximum(s2, 1)))
    if not len(tested):
        raise TrendError(f"No gene is detected in {min_detected} or more of these cells.")
    num = ess[tested] / (p - 1)
    den = rss[tested] / (n - p)
    f_stat = np.divide(num, den, out=np.full(len(tested), np.inf), where=den > 0)
    pval = stats.f.sf(f_stat, p - 1, n - p)
    pval_adj = stats.false_discovery_control(pval, method="bh")
    r2 = ess[tested] / tss[tested]

    rank = np.lexsort((names[tested].astype(str), -f_stat))[: max(1, n_top)]
    top = tested[rank]

    bin_of, bin_pt, counts = equal_count_bins(pt, n_bins)
    nb = len(bin_pt)
    B = sp.csr_matrix((np.ones(n), (np.arange(n), bin_of)), shape=(n, nb))
    counts = counts.astype(np.float64)
    mean = (B.T @ _columns(X, top)) / counts[:, None]  # bins x genes
    fitted = ((B.T @ Q) / counts[:, None]) @ qtx[:, top]

    genes = []
    for k, g in enumerate(top):
        curve = fitted[:, k]
        peak = int(np.argmax(curve))
        rel = peak / (nb - 1)
        genes.append({
            "gene": str(names[g]),
            "f_stat": _num(f_stat[rank[k]], 2),
            "pval": _sig(pval[rank[k]]),
            "pval_adj": _sig(pval_adj[rank[k]]),
            "r2": _num(r2[rank[k]]),
            "amplitude": _num(curve.max() - curve.min()),
            "peak_pseudotime": _num(bin_pt[peak]),
            "pattern": "rising" if rel >= 1 - EDGE else "falling" if rel <= EDGE else "transient",
            "n_detected": int(detected[g]),
            "mean": np.round(mean[:, k], 4).tolist(),
            "fitted": np.round(curve, 4).tolist(),
        })
    return {
        "column": column,
        "n_cells": n,
        "n_genes_tested": int(len(tested)),
        "min_detected": min_detected,
        "expression": source,
        "method": METHOD,
        "restrict": restrict,
        "rank_by": "trend",
        "bins": {"pseudotime": np.round(bin_pt, 4).tolist(), "n_cells": counts.astype(int).tolist(),
                 "binning": BINNING},
        "genes": genes,
    }


def _tree_path(adj: dict[int, set[int]], a: int, b: int) -> list[int]:
    parent = {a: a}
    stack = [a]
    while stack:
        x = stack.pop()
        for y in adj[x]:
            if y not in parent:
                parent[y] = x
                stack.append(y)
    if b not in parent:
        return []
    path = [b]
    while path[-1] != a:
        path.append(parent[path[-1]])
    return path[::-1]


def branch_split(adata, column: str, groups_column: str | None, groups: list[str] | None) -> dict[str, Any]:
    """A branch (``groups``: the groups on a path of the PAGA tree of ``groups_column``) split
    into its own groups and the trunk it shares with the paths from its start to the tree's other
    ends; every other tree group is "others". The start is the path end with the earlier median
    pseudotime."""
    from scview.core import trajectory as tj

    if not groups_column or not groups:
        raise TrendError("Branch-specific genes need a branch: groups_column and the groups on it.")
    record = tj.read_paga(adata)
    if not record or record.get("groupby") != groups_column:
        raise TrendError(f"Branch-specific genes need the PAGA graph of '{groups_column}': compute PAGA on it "
                         "(Trajectory), then pick a branch.")
    why = tj.paga_stale(adata, record)
    if why:
        raise TrendError(f"The PAGA graph of '{groups_column}' is out of date: {why}")
    names = record["groups"]
    wanted = list(dict.fromkeys(groups))
    missing = [g for g in wanted if g not in names]
    if missing:
        raise TrendError(f"Not in the PAGA graph of '{groups_column}': {', '.join(map(repr, missing))}.")
    adj = tj._tree_adjacency(record)
    on = {names.index(g) for g in wanted}
    inner = {i: adj[i] & on for i in on}
    ends = [i for i in on if len(inner[i]) <= 1]
    if len(on) < 2 or any(len(v) > 2 for v in inner.values()) or len(ends) != 2 or \
            len(_tree_path(adj, ends[0], ends[1])) != len(on):
        raise TrendError("The groups must be one path of the PAGA tree with at least two groups, from the start "
                         "to a branch end.")
    pt = pd.to_numeric(adata.obs[column], errors="coerce")
    labels = adata.obs[groups_column].astype(str).where(adata.obs[groups_column].notna())

    def median(i: int) -> float:
        v = pt[(labels == names[i]).to_numpy()]
        v = v[np.isfinite(v)]
        return float(np.median(v)) if len(v) else np.inf

    start, end = sorted(ends, key=median)
    shared: set[int] = set()
    for e in (names.index(g) for g in tj.tree_ends(record)):
        if e not in on:
            shared |= set(_tree_path(adj, start, e)) & on
    path = _tree_path(adj, start, end)
    own = [names[i] for i in path if i not in shared]
    if not own:
        raise TrendError(f"Every group on the branch to '{names[end]}' lies on another branch too: pick the branch "
                         "to another end.")
    return {"column": groups_column, "path": [names[i] for i in path], "start": names[start], "end": names[end],
            "trunk": [names[i] for i in path if i in shared], "own": own,
            "others": [g for i, g in enumerate(names) if i not in on]}


def branch_genes(
    adata,
    column: str,
    *,
    n_top: int = DEFAULT_TOP,
    n_bins: int = DEFAULT_BINS,
    groups_column: str | None = None,
    groups: list[str] | None = None,
) -> dict[str, Any]:
    """The genes higher on one branch than on the other branches of the PAGA tree, along
    the shared pseudotime (module docstring: ``BRANCH_METHOD``). Cells: those with a
    finite pseudotime in the tree's groups. Each gene has the test (``f_stat``, ``pval``,
    BH ``pval_adj`` over all genes tested, ``r2`` of the full fit), ``branch_diff`` (log
    units: how far the branch's own cells sit above the common curve, on average),
    ``mean_branch`` / ``mean_others`` (mean expression of the branch's own cells and of
    the other branches' cells), and, along the branch's cells (start to end), binned
    ``mean``, ``fitted``, ``amplitude``, peak and ``pattern`` as ``trend_genes`` gives them."""
    from scview.core.activity import lognorm_source

    split = branch_split(adata, column, groups_column, groups)
    pt_all = pd.to_numeric(adata.obs[column], errors="coerce").to_numpy(dtype=np.float64)
    col = adata.obs[split["column"]]
    labels = col.astype(str).where(col.notna()).to_numpy()
    tree_groups = set(split["path"]) | set(split["others"])
    use = np.isfinite(pt_all) & np.isin(labels, list(tree_groups))
    idx = np.flatnonzero(use)
    n = len(idx)
    lab = labels[idx]
    own = np.isin(lab, split["own"])
    on_path = np.isin(lab, split["path"])
    others = np.isin(lab, split["others"])
    for what, m in (("the branch's own groups", own), ("the other branches", others)):
        if m.sum() < MIN_CELLS:
            raise TrendError(f"Only {int(m.sum())} cells with a pseudotime in {what}; at least {MIN_CELLS} are needed.")
    pt = pt_all[idx]
    order = (stats.rankdata(pt) - 1) / (n - 1)
    knots = np.unique(np.quantile(order, np.linspace(0, 1, N_KNOTS)))
    knots_own = np.unique(np.quantile(order[own], np.linspace(0, 1, N_KNOTS)))
    if len(knots) < 3 or len(knots_own) < 2:
        raise TrendError("Too few distinct pseudotime values to fit a trend.")
    D0 = natural_spline_basis(order, knots)
    D_own = (natural_spline_basis(order, knots_own) if len(knots_own) >= 3
             else np.column_stack([np.ones(n), order])) * own[:, None]
    Q0, _ = np.linalg.qr(D0)
    D1 = np.hstack([D0, D_own])
    U, S, Vt = np.linalg.svd(D1, full_matrices=False)
    keep = S > S[0] * 1e-10
    Q1 = U[:, keep]
    p0, p1 = Q0.shape[1], Q1.shape[1]
    if p1 <= p0:
        raise TrendError("The branch's own cells add nothing to fit: pick another branch.")
    # The own spline's mean over the branch's own cells, per gene: c @ (Q1.T x) (pseudo-inverse).
    coef = (Vt[keep].T / S[keep])[p0:]  # own-spline coefficients from Q1.T x
    c = D_own[own].mean(axis=0) @ coef

    try:
        choice = lognorm_source(adata)
    except ValueError:
        raise TrendError(
            "No log-normalised expression or raw counts to test: run normalization and "
            "log transform first."
        ) from None
    names, source = np.asarray(choice.names, dtype=object), choice.source
    X = choice.rows(idx)
    s1, s2, qtx, detected = _moments(X, np.hstack([Q0, Q1]))
    qtx0, qtx1 = qtx[:p0], qtx[p0:]
    tss = s2 - s1 ** 2 / n
    ess0 = (qtx0 ** 2).sum(axis=0)
    ess1 = (qtx1 ** 2).sum(axis=0)
    rss1 = np.maximum(s2 - ess1, 0.0)
    min_detected = max(MIN_DETECTED, math.ceil(MIN_DETECTED_FRACTION * n))
    tested = np.flatnonzero((detected >= min_detected) & (tss > 1e-12 * np.maximum(s2, 1)))
    if not len(tested):
        raise TrendError(f"No gene is detected in {min_detected} or more of these cells.")
    num = np.maximum(ess1[tested] - ess0[tested], 0.0) / (p1 - p0)
    den = rss1[tested] / (n - p1)
    f_stat = np.divide(num, den, out=np.full(len(tested), np.inf), where=den > 0)
    pval = stats.f.sf(f_stat, p1 - p0, n - p1)
    pval_adj = stats.false_discovery_control(pval, method="bh")
    r2 = np.clip(1 - rss1[tested] / tss[tested], 0, 1)
    diff = c @ qtx1[:, tested]
    higher = diff > 0
    pick = np.flatnonzero(higher)
    rank = pick[np.lexsort((names[tested[pick]].astype(str), -f_stat[pick]))][: max(1, n_top)]
    top = tested[rank]

    cols = _columns(X, top)
    mean_own = cols[own].mean(axis=0)
    mean_oth = cols[others].mean(axis=0)
    path_rows = np.flatnonzero(on_path)
    bin_of, bin_pt, counts = equal_count_bins(pt[path_rows], n_bins)
    nb = len(bin_pt)
    B = sp.csr_matrix((np.ones(len(path_rows)), (np.arange(len(path_rows)), bin_of)), shape=(len(path_rows), nb))
    counts = counts.astype(np.float64)
    mean = (B.T @ cols[path_rows]) / counts[:, None]
    fitted = ((B.T @ Q1[path_rows]) / counts[:, None]) @ qtx1[:, top]

    genes = []
    for k, g in enumerate(top):
        curve = fitted[:, k]
        peak = int(np.argmax(curve))
        rel = peak / (nb - 1)
        genes.append({
            "gene": str(names[g]),
            "f_stat": _num(f_stat[rank[k]], 2),
            "pval": _sig(pval[rank[k]]),
            "pval_adj": _sig(pval_adj[rank[k]]),
            "r2": _num(r2[rank[k]]),
            "branch_diff": _num(diff[rank[k]]),
            "mean_branch": _num(mean_own[k]),
            "mean_others": _num(mean_oth[k]),
            "amplitude": _num(curve.max() - curve.min()),
            "peak_pseudotime": _num(bin_pt[peak]),
            "pattern": "rising" if rel >= 1 - EDGE else "falling" if rel <= EDGE else "transient",
            "n_detected": int(detected[g]),
            "mean": np.round(mean[:, k], 4).tolist(),
            "fitted": np.round(curve, 4).tolist(),
        })
    return {
        "column": column,
        "n_cells": n,
        "n_genes_tested": int(len(tested)),
        "n_higher": int(higher.sum()),
        "min_detected": min_detected,
        "expression": source,
        "method": BRANCH_METHOD,
        "restrict": {"column": split["column"], "groups": split["path"]},
        "rank_by": "branch_specific",
        "branch": {**split, "n_own_cells": int(own.sum()), "n_trunk_cells": int((on_path & ~own).sum()),
                   "n_other_cells": int(others.sum())},
        "bins": {"pseudotime": np.round(bin_pt, 4).tolist(), "n_cells": counts.astype(int).tolist(),
                 "binning": BINNING + "; the branch's cells, start to end"},
        "genes": genes,
    }
