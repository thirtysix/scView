"""limma-voom on per-sample counts: TMM factors, voom precision weights (one
mean-variance trend, or one per group as voomByGroup), weighted least squares
and empirical Bayes moderated t-tests.

Written from the published methods, not from limma/edgeR code:
- TMM: Robinson & Oshlack 2010, Genome Biology 11:R25 (edgeR calcNormFactors).
- voom: Law et al. 2014, Genome Biology 15:R29.
- voomByGroup: You et al. 2023, Genome Biology 24:93 (PMID 37147723): a trend per
  group, so a group whose samples vary more does not lend its variance to the other.
- lmFit + eBayes: Smyth 2004; the prior comes from core/composition.py fit_f_dist.

Every matrix is samples x genes (rows = samples), like an AnnData and like
``design`` (samples x coefficients); limma's are genes x samples. The API is
general, for any per-sample count model (pseudobulk DE, neighbourhood DE)::

    E, w = voom(counts, design)                     # one trend
    E, w = voom(counts, design, group=condition)    # voomByGroup
    fit = lm_fit_weighted(E, design, w)
    res = ebayes(fit, coef=design.shape[1] - 1)     # res.logfc, res.t, res.p

``voom_fit`` returns the same plus each group's trend and residual variances.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats

# The trend's lowess span: limma (>= 3.60) picks it from the number of genes by
# default (``adaptive.span = TRUE``), between MIN_SPAN and 1.
MIN_SPAN = 0.3
SMALL_N = 50
# edgeR's TMM trims: 30% of log ratios (M) and 5% of mean abundances (A) each side.
TMM_LOGRATIO_TRIM = 0.3
TMM_SUM_TRIM = 0.05


class VoomError(ValueError):
    """Input voom cannot fit (no residual degrees of freedom, a singular design)."""


# ---------------------------------------------------------------------------
# Normalisation factors
# ---------------------------------------------------------------------------


def _quantile_factors(counts: np.ndarray, lib_size: np.ndarray, p: float) -> np.ndarray:
    """Each sample's ``p`` quantile of counts / library size (R's type-7 quantile)."""
    return np.quantile(counts / lib_size[:, None], p, axis=1)


def _tmm_one(obs: np.ndarray, ref: np.ndarray, n_obs: float, n_ref: float) -> float:
    """TMM factor of one sample against the reference: the precision-weighted mean
    of the log ratios left after trimming extreme ratios and abundances."""
    with np.errstate(divide="ignore", invalid="ignore"):
        log_r = np.log2((obs / n_obs) / (ref / n_ref))
        abs_e = (np.log2(obs / n_obs) + np.log2(ref / n_ref)) / 2
        v = (n_obs - obs) / n_obs / obs + (n_ref - ref) / n_ref / ref
    fin = np.isfinite(log_r) & np.isfinite(abs_e)
    log_r, abs_e, v = log_r[fin], abs_e[fin], v[fin]
    if len(log_r) == 0 or np.max(np.abs(log_r)) < 1e-6:
        return 1.0
    n = len(log_r)
    lo_l = np.floor(n * TMM_LOGRATIO_TRIM) + 1
    hi_l = n + 1 - lo_l
    lo_s = np.floor(n * TMM_SUM_TRIM) + 1
    hi_s = n + 1 - lo_s
    rl, ra = stats.rankdata(log_r), stats.rankdata(abs_e)
    keep = (rl >= lo_l) & (rl <= hi_l) & (ra >= lo_s) & (ra <= hi_s)
    if not keep.any():
        return 1.0
    f = float(np.sum(log_r[keep] / v[keep]) / np.sum(1 / v[keep]))
    return float(2**f) if np.isfinite(f) else 1.0


def norm_factors(counts: np.ndarray, method: str = "TMM", lib_size: np.ndarray | None = None) -> np.ndarray:
    """Per-sample normalisation factors (geometric mean 1), as edgeR's
    calcNormFactors: ``TMM`` (against the sample whose upper quartile is closest
    to the mean), ``upperquartile`` or ``none``. The effective library size is
    ``lib_size * factors``."""
    x = np.asarray(counts, dtype=np.float64)
    n = x.shape[0]
    if method == "none":
        return np.ones(n)
    lib = x.sum(axis=1) if lib_size is None else np.asarray(lib_size, dtype=np.float64)
    x = x[:, (x > 0).any(axis=0)]  # genes with a count somewhere
    if method == "upperquartile":
        f = _quantile_factors(x, lib, 0.75)
    elif method == "TMM":
        f75 = _quantile_factors(x, lib, 0.75)
        if np.median(f75) < 1e-20:
            ref = int(np.argmax(np.sqrt(x).sum(axis=1)))
        else:
            ref = int(np.argmin(np.abs(f75 - f75.mean())))
        f = np.array([_tmm_one(x[i], x[ref], lib[i], lib[ref]) for i in range(n)])
    else:
        raise VoomError(f"Unknown normalisation '{method}' (TMM, upperquartile or none).")
    if not np.all(np.isfinite(f) & (f > 0)):
        raise VoomError("A sample's normalisation factor is zero; it has too few counts.")
    return f / np.exp(np.mean(np.log(f)))


def log_cpm(counts: np.ndarray, lib_size: np.ndarray) -> np.ndarray:
    """voom's log2 counts per million: log2((count + 0.5) / (lib.size + 1) x 1e6)."""
    lib = np.asarray(lib_size, dtype=np.float64)
    return np.log2((np.asarray(counts, dtype=np.float64) + 0.5) / (lib[:, None] + 1) * 1e6)


# ---------------------------------------------------------------------------
# Least squares
# ---------------------------------------------------------------------------


def full_rank_columns(X: np.ndarray, tol: float = 1e-7) -> list[int]:
    """The columns of ``X`` that are not combinations of earlier ones (in order),
    as R's ``qr`` pivots dependent columns to the end."""
    keep: list[int] = []
    Q = np.zeros((X.shape[0], 0))
    for j in range(X.shape[1]):
        col = X[:, j].astype(np.float64)
        norm = np.linalg.norm(col)
        if norm == 0:
            continue
        r = col - Q @ (Q.T @ col)
        if np.linalg.norm(r) > tol * norm:
            keep.append(j)
            Q = np.column_stack([Q, r / np.linalg.norm(r)])
    return keep


@dataclass
class LmFit:
    coef: np.ndarray  # coefficients x genes
    stdev_unscaled: np.ndarray  # coefficients x genes
    s2: np.ndarray  # residual variance per gene
    df_residual: int
    amean: np.ndarray  # mean of E per gene


def lm_fit_weighted(E: np.ndarray, design: np.ndarray, weights: np.ndarray | None = None) -> LmFit:
    """Per gene (column of ``E``), least squares on ``design`` with that gene's
    per-sample ``weights`` (limma lmFit). The design must have full column rank."""
    E = np.asarray(E, dtype=np.float64)
    X = np.asarray(design, dtype=np.float64)
    n, p = X.shape
    if len(full_rank_columns(X)) < p:
        raise VoomError("The design is not of full rank: a covariate repeats another.")
    df = n - p
    if weights is None:
        V = np.linalg.inv(X.T @ X)
        coef = V @ X.T @ E
        resid = E - X @ coef
        s2 = (resid**2).sum(axis=0) / df if df > 0 else np.full(E.shape[1], np.nan)
        su = np.sqrt(np.repeat(np.diag(V)[:, None], E.shape[1], axis=1))
        return LmFit(coef=coef, stdev_unscaled=su, s2=s2, df_residual=df, amean=E.mean(axis=0))
    W = np.asarray(weights, dtype=np.float64)
    XtWX = np.einsum("ni,ng,nj->gij", X, W, X)
    XtWy = np.einsum("ni,ng,ng->gi", X, W, E)
    V = np.linalg.inv(XtWX)  # genes x p x p
    coef = np.einsum("gij,gj->gi", V, XtWy).T
    resid = E - X @ coef
    s2 = (W * resid**2).sum(axis=0) / df if df > 0 else np.full(E.shape[1], np.nan)
    su = np.sqrt(np.einsum("gii->gi", V)).T
    return LmFit(coef=coef, stdev_unscaled=su, s2=s2, df_residual=df, amean=E.mean(axis=0))


@dataclass
class EBayes:
    logfc: np.ndarray  # the coefficient
    t: np.ndarray  # moderated t
    p: np.ndarray  # two-sided
    s2_post: np.ndarray
    s2_prior: float
    df_prior: float  # inf: the variances vary no more than chance allows
    df_total: np.ndarray


def ebayes(fit: LmFit, coef: int, robust: bool = False) -> EBayes:
    """Moderated t-test of one coefficient: each gene's residual variance is
    squeezed toward a prior fitted to them all (limma eBayes, ``robust`` as its
    argument), and the t-test takes the prior's degrees of freedom too."""
    from scview.core.composition import fit_f_dist, fit_f_dist_robust

    df = fit.df_residual
    if df < 1:
        raise VoomError("No residual degrees of freedom: too few samples for the design.")
    s2 = fit.s2
    G = len(s2)
    if G < 3:
        s0, prior = float("nan"), np.zeros(G)
    elif robust:
        s0, prior = fit_f_dist_robust(s2, df)
    else:
        s0, d = fit_f_dist(s2, df)
        prior = np.full(G, d)
    if not np.isfinite(s0):
        s_post, prior = s2.copy(), np.zeros(G)
    else:
        with np.errstate(invalid="ignore"):
            s_post = np.where(np.isfinite(prior), (df * s2 + prior * s0) / (df + prior), s0)
    df_total = np.minimum(df + prior, df * G)  # limma caps at the pooled residual df
    b = fit.coef[coef]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = b / (fit.stdev_unscaled[coef] * np.sqrt(s_post))
    p = 2 * stats.t.sf(np.abs(t), df_total)
    return EBayes(logfc=b, t=t, p=p, s2_post=s_post, s2_prior=float(s0),
                  df_prior=float(np.max(prior)) if G else 0.0, df_total=df_total)


# ---------------------------------------------------------------------------
# voom
# ---------------------------------------------------------------------------


@dataclass
class Trend:
    """voom's mean-variance trend: sqrt(residual SD) against mean log2 count,
    a lowess curve read by linear interpolation (flat beyond its ends)."""

    x: np.ndarray
    y: np.ndarray

    def __call__(self, v: np.ndarray) -> np.ndarray:
        return np.interp(v, self.x, self.y)

    def points(self, n: int = 60) -> list[list[float]]:
        """At most ``n`` points along the curve, for a plot."""
        idx = np.unique(np.linspace(0, len(self.x) - 1, min(n, len(self.x))).round().astype(int))
        return [[float(self.x[i]), float(self.y[i])] for i in idx]


def choose_span(n_genes: int) -> float:
    """limma's chooseLowessSpan: wider for fewer genes, 0.3 in the limit."""
    return float(min(MIN_SPAN + (1 - MIN_SPAN) * (SMALL_N / max(n_genes, 1)) ** (1 / 3), 1.0))


def _lowess_trend(sx: np.ndarray, sy: np.ndarray, span: float) -> Trend:
    from statsmodels.nonparametric.smoothers_lowess import lowess

    # R's lowess: 3 robustness iterations, interpolation within 1% of the range.
    delta = 0.01 * float(np.ptp(sx))
    fitted = lowess(sy, sx, frac=span, it=3, delta=delta, return_sorted=True)
    x, y = fitted[:, 0], fitted[:, 1]
    ux, inv = np.unique(x, return_inverse=True)  # ties: their mean, as approxfun
    uy = np.bincount(inv, weights=y) / np.bincount(inv)
    return Trend(x=ux, y=uy)


@dataclass
class VoomFit:
    E: np.ndarray  # log2 CPM, samples x genes
    weights: np.ndarray  # precision weights, samples x genes
    lib_size: np.ndarray  # effective library sizes used
    # One trend per group (key "all" without groups); None where a group had no
    # residual degrees of freedom (its weights are then 1).
    trends: dict[str, Trend | None] = field(default_factory=dict)
    # Each group's per-gene residual variance of log2 CPM in its own fit.
    s2: dict[str, np.ndarray] = field(default_factory=dict)


def _voom_one(counts: np.ndarray, design: np.ndarray, lib: np.ndarray, span: float | None
              ) -> tuple[np.ndarray, np.ndarray, Trend | None, np.ndarray]:
    """voom on one set of samples: (E, weights, trend, residual variances)."""
    E = log_cpm(counts, lib)
    fit = lm_fit_weighted(E, design)
    if fit.df_residual < 1:  # no replication: limma sets every weight to 1
        return E, np.ones_like(E), None, np.full(E.shape[1], np.nan)
    sx = fit.amean + np.mean(np.log2(lib + 1)) - np.log2(1e6)
    sy = np.sqrt(np.sqrt(fit.s2))
    expressed = counts.sum(axis=0) > 0
    span = choose_span(counts.shape[1]) if span is None else span
    trend = _lowess_trend(sx[expressed], sy[expressed], span)
    fitted_cpm = 2.0 ** (design @ fit.coef)
    fitted_logcount = np.log2(1e-6 * fitted_cpm * (lib[:, None] + 1))
    w = 1.0 / trend(fitted_logcount) ** 4
    return E, w, trend, fit.s2


def _group_design(design: np.ndarray) -> np.ndarray:
    """voomByGroup's design for one group's samples: its estimable columns, or the
    intercept alone when those leave no residual degrees of freedom."""
    keep = full_rank_columns(design)
    d = design[:, keep]
    if d.shape[1] >= d.shape[0] or d.shape[1] == 0:
        return np.ones((design.shape[0], 1))
    return d


def voom_fit(
    counts: np.ndarray,
    design: np.ndarray,
    group: np.ndarray | None = None,
    lib_size: np.ndarray | None = None,
    span: float | None = None,
) -> VoomFit:
    """voom (``group`` None) or voomByGroup (a trend per value of ``group``).

    ``counts`` is samples x genes; ``lib_size`` the effective library sizes
    (default: each sample's total; pass ``total * norm_factors(...)`` for TMM).
    ``span``: the trend's lowess span; None picks it from the number of genes, as
    limma does by default."""
    C = np.asarray(counts, dtype=np.float64)
    X = np.asarray(design, dtype=np.float64)
    if C.shape[0] != X.shape[0]:
        raise VoomError("counts and design need one row per sample.")
    if C.shape[1] < 2:
        raise VoomError("voom needs at least two genes to fit a mean-variance trend.")
    lib = C.sum(axis=1) if lib_size is None else np.asarray(lib_size, dtype=np.float64)
    if group is None:
        E, w, trend, s2 = _voom_one(C, X, lib, span)
        return VoomFit(E=E, weights=w, lib_size=lib, trends={"all": trend}, s2={"all": s2})
    g = np.asarray(group).astype(str)
    E = np.empty_like(C)
    w = np.empty_like(C)
    trends: dict[str, Trend | None] = {}
    s2s: dict[str, np.ndarray] = {}
    for lev in sorted(set(g.tolist())):
        i = g == lev
        Ei, wi, ti, si = _voom_one(C[i], _group_design(X[i]), lib[i], span)
        E[i], w[i] = Ei, wi
        trends[lev], s2s[lev] = ti, si
    return VoomFit(E=E, weights=w, lib_size=lib, trends=trends, s2=s2s)


def voom(
    counts: np.ndarray,
    design: np.ndarray,
    group: np.ndarray | None = None,
    lib_size: np.ndarray | None = None,
    span: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """(log2 CPM, precision weights), both samples x genes: voom, or voomByGroup
    when ``group`` is given (see ``voom_fit``)."""
    f = voom_fit(counts, design, group=group, lib_size=lib_size, span=span)
    return f.E, f.weights
