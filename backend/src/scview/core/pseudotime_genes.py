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
) -> dict[str, Any]:
    """The ``n_top`` genes that change most along the pseudotime ``column``.

    Cells without a finite pseudotime are left out, and with ``groups_column``
    so are cells outside ``groups`` (a branch: the groups on a PAGA path). Each
    gene comes with its test (``f_stat``, ``pval``, BH ``pval_adj``, ``r2``),
    binned ``mean`` expression and the spline's ``fitted`` mean in the same
    bins, the fitted ``amplitude`` (max - min, log units), the bin
    pseudotime of the fitted peak and a ``pattern``: rising, falling (peak at the
    start) or transient.
    """
    from scview.core.activity import lognorm_matrix

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
        M, names, source = lognorm_matrix(adata)
    except ValueError:
        raise TrendError(
            "No log-normalised expression or raw counts to test: run normalization and "
            "log transform first."
        ) from None
    names = np.asarray(names, dtype=object)
    X = M[idx]
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

    # Bins of (nearly) equal cell counts along pseudotime.
    nb = int(min(n_bins, max(2, n // MIN_CELLS_PER_BIN)))
    bin_of = np.empty(n, dtype=np.int64)
    bin_of[np.argsort(pt, kind="stable")] = np.arange(n) * nb // n
    B = sp.csr_matrix((np.ones(n), (np.arange(n), bin_of)), shape=(n, nb))
    counts = np.bincount(bin_of, minlength=nb).astype(np.float64)
    bin_pt = np.array([np.median(pt[bin_of == b]) for b in range(nb)])
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
        "bins": {"pseudotime": np.round(bin_pt, 4).tolist(), "n_cells": counts.astype(int).tolist()},
        "genes": genes,
    }
