"""DESeq2's Wald test on pseudobulk counts, vectorised over genes.

The steps and defaults are pydeseq2 0.5.4's (``DeseqDataSet.deseq2()`` then
``DeseqStats.summary()``; pydeseq2 is MIT, Muzellec et al. 2023,
Bioinformatics 39:btad547), which follows Love, Huber & Anders 2014 (Genome
Biology 15:550). pydeseq2 runs one scipy optimiser per gene for each
dispersion fit; here every gene moves at once:

1. Median-of-ratios size factors over the genes with no zero; when every gene
   has a zero, pydeseq2's iterative size factors (Powell on the trimmed NB
   likelihood, intercept-only dispersions, mean trend).
2. Start dispersions: min of the rough (linear model) and moments estimates.
3. Means for the dispersion fit: the linear model when the design has as many
   distinct rows as columns, else NB IRLS at the start dispersions.
4. Gene-wise dispersions: Cox-Reid adjusted profile likelihood maximised in
   log(alpha) on [1e-8, max(10, n)] from the start (``nb_glm.maximise_1d``,
   analytic derivatives; a 100-point grid only for genes it cannot handle).
5. Trend a0 + a1 / mean by pydeseq2's gamma GLM (its L-BFGS-B fit, refitted
   without genes off the curve), else the mean trend.
6. Prior variance: squared MAD of the log residuals (genes with dispersion
   >= 100 * min_disp) minus trigamma(df / 2), at least 0.25.
7. MAP dispersions (same likelihood plus the log-normal prior around the
   trend); gene-wise kept where it lies more than 2 MADs above the trend.
8. Coefficients by IRLS (``nb_glm.irls``), Cook's distances (robust moments
   dispersion), the outlier refit: counts with Cook's above F(0.99; p, n - p)
   in samples whose design cell has >= ``min_replicates`` samples replaced by
   the 20% trimmed mean and the gene refitted on the same size factors,
   trend and prior.
9. Wald test of the contrast, Cook's filtering of p values (cells with >= 3
   samples), independent filtering on the mean (pydeseq2's lowess rule) for
   the adjusted p.

Deliberate differences: the dispersion maxima are found to 1e-8 in
log(alpha), where pydeseq2 stops at L-BFGS-B's tolerance, so pydeseq2's
grid fallback after an L-BFGS-B failure is not reproduced; the grid fallback
here keeps the prior for MAP dispersions (pydeseq2 drops it).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import optimize, special, stats

from scview.core import nb_glm

MIN_DISP = 1e-8
MAX_DISP = 10.0
_CHUNK_ELEMENTS = 2_000_000  # genes x samples per chunk of the dispersion and GLM fits


@dataclass
class DeseqResult:
    base_mean: np.ndarray
    log2fc: np.ndarray
    lfc_se: np.ndarray
    stat: np.ndarray
    pvalue: np.ndarray
    padj: np.ndarray
    beta: np.ndarray  # genes x p, natural log
    size_factors: np.ndarray
    dispersions: np.ndarray
    genewise_dispersions: np.ndarray
    fitted_dispersions: np.ndarray
    map_dispersions: np.ndarray
    cooks_outlier: np.ndarray
    replaced: np.ndarray
    refitted: np.ndarray
    trend: str  # "parametric" | "mean"
    trend_coeffs: np.ndarray | None
    prior_disp_var: float
    squared_logres: float
    genewise_converged: np.ndarray
    map_converged: np.ndarray
    lfc_converged: np.ndarray


# ---------------------------------------------------------------------------
# Small pieces
# ---------------------------------------------------------------------------

def _cells(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Design cells: per sample its cell code, and per cell its sample count."""
    _, inv, cnt = np.unique(X, axis=0, return_inverse=True, return_counts=True)
    return inv.reshape(-1), cnt


def n_or_more_replicates(X: np.ndarray, k: int) -> np.ndarray:
    inv, cnt = _cells(X)
    return cnt[inv] >= k


def trimmed_mean(x: np.ndarray, trim: float, axis: int = -1) -> np.ndarray:
    s = np.sort(x, axis=axis)
    n = x.shape[axis]
    t = math.floor(n * trim)
    return np.take(s, np.arange(t, n - t), axis=axis).mean(axis=axis)


def _mad(x: np.ndarray) -> float:
    return float(np.median(np.abs(x - np.median(x))) / stats.norm.ppf(0.75))


def _chunks(G: int, n: int):
    step = max(1, _CHUNK_ELEMENTS // max(n, 1))
    return [slice(i, min(i + step, G)) for i in range(0, G, step)]


def size_factors(counts: np.ndarray) -> np.ndarray:
    """Median-of-ratios size factors (samples x genes) over the genes with no zero."""
    with np.errstate(divide="ignore"):
        logc = np.log(counts)
    logmeans = logc.mean(axis=0)
    ok = np.isfinite(logmeans)
    return np.exp(np.median(logc[:, ok] - logmeans[ok], axis=1))


def _rough_dispersions(norm: np.ndarray, X: np.ndarray) -> np.ndarray:
    n, p = X.shape
    coef = np.linalg.lstsq(X, norm.T, rcond=None)[0]
    yhat = np.maximum((X @ coef).T, 1)
    return np.maximum((((norm - yhat) ** 2 - yhat) / ((n - p) * yhat ** 2)).sum(axis=1), 0)


def _moments_dispersions(norm: np.ndarray, sf: np.ndarray) -> np.ndarray:
    m = norm.mean(axis=1)
    v = norm.var(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.nan_to_num((v - (1 / sf).mean() * m) / m ** 2)


def _robust_mom_dispersions(norm: np.ndarray, X: np.ndarray) -> np.ndarray:
    """pydeseq2's ``robust_method_of_moments_disp`` (for Cook's distances)."""
    three = n_or_more_replicates(X, 3)
    if three.any():
        inv, _ = _cells(X[three])
        sub = norm[:, three]
        v = np.full(len(norm), -np.inf)
        for c in np.unique(inv):
            cols = inv == c
            k = int(cols.sum())
            b = 2 if k >= 23.5 else 1 if k >= 3.5 else 0
            trim = (1 / 3, 1 / 4, 1 / 8)[b]
            x = sub[:, cols]
            sq = (x - trimmed_mean(x, trim)[:, None]) ** 2
            v = np.maximum(v, (2.04, 1.86, 1.51)[b] * trimmed_mean(sq, trim))
    else:
        sq = (norm - trimmed_mean(norm, 0.125)[:, None]) ** 2
        v = 1.51 * trimmed_mean(sq, 0.125)
    m = norm.mean(axis=1)
    return np.maximum((v - m) / m ** 2, 0.04)


# ---------------------------------------------------------------------------
# Dispersions
# ---------------------------------------------------------------------------

class _Fitter:
    """Size factors, design and bounds shared by the steps of one fit."""

    def __init__(self, X, sf, min_mu, min_disp, max_disp, beta_tol):
        self.X = X
        self.n, self.p = X.shape
        self.XX = nb_glm.outer_design(X)
        self.sf = sf
        self.logsf = np.log(sf)
        self.min_mu, self.min_disp, self.max_disp = min_mu, min_disp, max_disp
        self.beta_tol = beta_tol
        self.saturated = len(_cells(X)[1]) == self.p

    def irls(self, Y, alpha) -> nb_glm.IrlsResult:
        parts = [nb_glm.irls(Y[s], self.logsf, self.X, alpha[s], min_mu=self.min_mu,
                             beta_tol=self.beta_tol) for s in _chunks(*Y.shape)]
        if len(parts) == 1:
            return parts[0]
        return nb_glm.IrlsResult(*(np.concatenate([getattr(r, f) for r in parts])
                                   for f in ("beta", "mu", "hat", "converged", "fallback")))

    def start_and_mu(self, Y):
        """Start dispersions and the means the dispersions are fitted at."""
        norm = Y / self.sf
        mom = np.clip(np.minimum(_rough_dispersions(norm, self.X),
                                 _moments_dispersions(norm, self.sf)),
                      self.min_disp, self.max_disp)
        if self.saturated:
            coef = np.linalg.lstsq(self.X, norm.T, rcond=None)[0]
            mu = np.maximum(self.sf * (self.X @ coef).T, self.min_mu)
        else:
            mu = self.irls(Y, mom).mu
        return mom, mu

    def log_disp(self, Y, mu, start, prior=None):
        """Maximise the Cox-Reid adjusted likelihood (+ prior) per gene."""
        G = len(Y)
        lo = np.full(G, math.log(self.min_disp))
        hi = np.full(G, math.log(self.max_disp))
        out = np.empty(G)
        ok = np.empty(G, dtype=bool)
        for s in _chunks(G, self.n):
            Ys, ms = Y[s], mu[s]
            pr = None if prior is None else (prior[0][s], prior[1])

            def derivs(idx, th, Ys=Ys, ms=ms, pr=pr):
                a = np.exp(th)
                g1, h1 = nb_glm.nb_loglik_derivs(Ys[idx], ms[idx], a)
                g2, h2 = nb_glm.cox_reid_derivs(ms[idx], a, self.XX, self.p)
                g, h = g1 + g2, h1 + h2
                if pr is not None:
                    g = g - (th - pr[0][idx]) / pr[1]
                    h = h - 1 / pr[1]
                return g, h

            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                out[s], ok[s] = nb_glm.maximise_1d(derivs, np.log(start[s]), lo[s], hi[s])
        if prior is None:
            # A gene-wise start on min_disp stays there: the slope in log(alpha)
            # vanishes with alpha, so pydeseq2's L-BFGS-B stops at once (R DESeq2
            # keeps every such start too, measured on Kang).
            at_min = start <= self.min_disp
            out[at_min], ok[at_min] = math.log(self.min_disp), True
        bad = np.flatnonzero(~ok)
        if bad.size:
            out[bad] = self._grid_log_disp(Y[bad], mu[bad], None if prior is None
                                           else (prior[0][bad], prior[1]))
        return np.clip(np.exp(out), self.min_disp, self.max_disp), ok

    def _grid_log_disp(self, Y, mu, prior, n_grid: int = 100):
        """pydeseq2's ``grid_fit_alpha``: 100 points, then 100 around the best."""
        lo, hi = math.log(self.min_disp), math.log(self.max_disp)

        def f(grid):  # genes x K
            out = np.empty(grid.shape)
            for k in range(grid.shape[1]):
                a = np.exp(grid[:, k])
                out[:, k] = (-nb_glm.nb_nll(Y, mu, a)
                             + nb_glm.cox_reid(mu, a, self.XX, self.p))
                if prior is not None:
                    out[:, k] -= (grid[:, k] - prior[0]) ** 2 / (2 * prior[1])
            return out

        grid = np.broadcast_to(np.linspace(lo, hi, n_grid), (len(Y), n_grid))
        best = grid[np.arange(len(Y)), np.nanargmax(f(grid), axis=1)]
        d = (hi - lo) / (n_grid - 1)
        fine = best[:, None] + np.linspace(-d, d, n_grid)[None]
        return fine[np.arange(len(Y)), np.nanargmax(f(fine), axis=1)]


def _gamma_glm(x: np.ndarray, y: np.ndarray):
    """pydeseq2's ``dispersion_trend_gamma_glm``: y ~ a0 + a1 x, gamma family."""
    Xc = np.column_stack([np.ones(len(x)), x])

    def loss(c):
        m = Xc @ c
        return np.nanmean(y / m + np.log(m), axis=0)

    def grad(c):
        m = Xc @ c
        return -np.nanmean(((y / m - 1)[:, None] * Xc) / m[:, None], axis=0)

    res = optimize.minimize(loss, x0=np.array([1.0, 1.0]), jac=grad, method="L-BFGS-B",
                            bounds=[(1e-12, np.inf)] * 2)
    return res.x, Xc @ res.x, res.success


def _parametric_trend(mean: np.ndarray, disp: np.ndarray) -> np.ndarray | None:
    """Trend coefficients (a0, a1), or None for pydeseq2's fall back to the mean."""
    x, y = 1 / mean, disp
    keep = np.isfinite(x)
    x, y = x[keep], y[keep]
    old, coeffs = np.array([0.1, 0.1]), np.array([1.0, 1.0])
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        while (coeffs > 1e-10).all() and (np.log(np.abs(coeffs / old)) ** 2).sum() >= 1e-6:
            old = coeffs
            if not len(x):
                return None
            coeffs, pred, ok = _gamma_glm(x, y)
            if not ok or (coeffs <= 1e-10).any():
                return None
            ratio = y / pred
            keep = ~((ratio < 1e-4) | (ratio >= 15))
            x, y = x[keep], y[keep]
    return coeffs


def _mean_trend(disp: np.ndarray, min_disp: float) -> float:
    return float(stats.trim_mean(disp[disp > 10 * min_disp], proportiontocut=0.001))


def _prior(gw, fitted, n, p, min_disp) -> tuple[float, float]:
    """(squared log-residual MAD, prior variance of log dispersions)."""
    resid = np.log(gw) - np.log(fitted)
    above = gw >= 100 * min_disp
    slr = _mad(resid[above]) ** 2 if above.any() else 0.0
    return slr, max(slr - float(special.polygamma(1, (n - p) / 2)), 0.25)


def _map_and_lfc(F: _Fitter, Y, mu, gw, fitted, slr, prior_var):
    lf = np.log(fitted)
    mp, mconv = F.log_disp(Y, mu, fitted, prior=(lf, prior_var))
    disp = np.where(np.log(gw) > lf + 2 * math.sqrt(slr), gw, mp)
    return mp, mconv, disp, F.irls(Y, disp)


# ---------------------------------------------------------------------------
# Iterative size factors (every gene has a zero)
# ---------------------------------------------------------------------------

def _iterative_size_factors(C: np.ndarray, min_mu, min_disp, max_disp, beta_tol,
                            niter: int = 10, quant: float = 0.95) -> np.ndarray:
    """pydeseq2's ``_fit_iterate_size_factors``: its normalised counts stay
    the raw counts inside the loop, as there."""
    n = C.shape[0]
    X1 = np.ones((n, 1))
    sf = np.ones(n)
    Y = C.T[~(C == 0).all(axis=0)].astype(np.float64)
    for i in range(niter):
        F = _Fitter(X1, sf, min_mu, min_disp, max_disp, beta_tol)
        mom = np.clip(np.minimum(_rough_dispersions(Y, X1), _moments_dispersions(Y, sf)),
                      min_disp, max_disp)
        mu = np.maximum(sf * (Y / sf).mean(axis=1, keepdims=True), min_mu)
        gw, _ = F.log_disp(Y, mu, mom)
        if not (gw > 10 * min_disp).any():
            break
        fitted = np.full(len(Y), _mean_trend(gw, min_disp))
        slr, prior_var = _prior(gw, fitted, n, 1, min_disp)
        mp, _ = F.log_disp(Y, mu, fitted, prior=(np.log(fitted), prior_var))
        disp = np.where(np.log(gw) > np.log(fitted) + 2 * math.sqrt(slr), gw, mp)
        old = sf.copy()
        base = mu / old

        def objective(lp, base=base, disp=disp):
            s = np.exp(lp - lp.mean())
            nll = nb_glm.nb_nll(Y, base * s, disp)
            return float(nll[nll < np.quantile(nll, quant)].sum())

        res = optimize.minimize(objective, np.log(old), method="Powell")
        sf = np.exp(res.x - res.x.mean())
        if not res.success:
            break
        if i > 1 and np.sum((np.log(old) - np.log(sf)) ** 2) < 1e-4:
            break
    return sf


# ---------------------------------------------------------------------------
# Independent filtering
# ---------------------------------------------------------------------------

def _lowess(x: np.ndarray, y: np.ndarray, frac: float, it: int = 3) -> np.ndarray:
    """pydeseq2's ``lowess`` (robust locally weighted linear fit)."""
    n = len(x)
    r = int(math.ceil(frac * n))
    h = np.maximum(np.sort(np.abs(x[None, :] - x[:, None]), axis=1)[:, r], 1e-12)
    w = np.clip(np.abs(np.nan_to_num((x[:, None] - x[None, :]) / h)), 0.0, 1.0)
    w = (1 - w ** 3) ** 3
    yest = np.zeros(n)
    delta = np.ones(n)
    for _ in range(it):
        for i in range(n):
            wt = delta * w[:, i]
            b = np.array([np.sum(wt * y), np.sum(wt * y * x)])
            A = np.array([[np.sum(wt), np.sum(wt * x)], [np.sum(wt * x), np.sum(wt * x * x)]])
            beta = np.linalg.lstsq(A, b, rcond=None)[0]
            yest[i] = beta[0] + beta[1] * x[i]
        res = y - yest
        s = np.median(np.abs(res))
        delta = (np.abs(res) > 0).astype(float) if s == 0 else np.clip(res / (6.0 * s), -1, 1)
        delta = (1 - delta ** 2) ** 2
    return yest


def _bh(p: np.ndarray) -> np.ndarray:
    out = np.full(len(p), np.nan)
    ok = ~np.isnan(p)
    if ok.any():
        out[ok] = stats.false_discovery_control(p[ok], method="bh")
    return out


def filter_by_mean(base_mean: np.ndarray, p: np.ndarray, alpha: float) -> np.ndarray:
    """pydeseq2's ``_independent_filtering``: BH among the genes above the
    mean cutoff that maximises rejections (lowess rule)."""
    lower = float(np.mean(base_mean == 0))
    upper = 0.95 if lower < 0.95 else 1.0
    theta = np.linspace(lower, upper, 50)
    cutoffs = np.quantile(base_mean, theta)
    result = np.full((len(p), len(theta)), np.nan)
    for i, c in enumerate(cutoffs):
        use = (base_mean >= c) & ~np.isnan(p)
        if use.any():
            result[use, i] = stats.false_discovery_control(p[use], method="bh")
    num_rej = (result < alpha).sum(axis=0)
    fit = _lowess(theta, num_rej.astype(float), frac=1 / 5)
    j = 0
    if num_rej.max() > 10:
        pos = num_rej > 0
        thresh = fit.max() - np.sqrt(np.mean((num_rej[pos] - fit[pos]) ** 2))
        if np.any(num_rej > thresh):
            j = int(np.flatnonzero(num_rej > thresh)[0])
    return result[:, j]


# ---------------------------------------------------------------------------
# The test
# ---------------------------------------------------------------------------

def deseq2_wald(counts: np.ndarray, X: np.ndarray, coef: int | np.ndarray = -1, *,
                refit_cooks: bool = True, cooks_filter: bool = True,
                independent_filter: bool = True, alpha: float = 0.05,
                min_replicates: int = 7, fit_type: str = "parametric",
                min_mu: float = nb_glm.MIN_MU, min_disp: float = MIN_DISP,
                max_disp: float = MAX_DISP, beta_tol: float = nb_glm.BETA_TOL) -> DeseqResult:
    """DESeq2's Wald test of design column ``coef`` (or a contrast vector) on
    ``counts`` (samples x genes, non-negative integers) with design ``X``
    (samples x p, full column rank, intercept first). NaN as pydeseq2 sets
    them: all-zero genes get no test; Cook's outliers lose their p value;
    genes below the independent-filtering cutoff lose their adjusted p."""
    C = np.asarray(counts)
    if C.ndim != 2 or not np.all(np.isfinite(C)) or (C < 0).any() or np.any(C != np.round(C)):
        raise ValueError("Counts must be a samples x genes matrix of non-negative integers.")
    C = C.astype(np.int64)
    X = np.asarray(X, dtype=np.float64)
    n, p = X.shape
    if C.shape[0] != n:
        raise ValueError("Counts and design have different numbers of samples.")
    if np.linalg.matrix_rank(X) < p:
        raise ValueError("The design matrix is not full rank.")
    if n <= p:
        raise ValueError("No residual degrees of freedom to estimate the dispersions.")
    if fit_type not in ("parametric", "mean"):
        raise ValueError(f"Unknown fit_type {fit_type!r}.")
    c = np.zeros(p)
    if np.ndim(coef) == 0:
        c[int(coef)] = 1.0
    else:
        c = np.asarray(coef, dtype=np.float64)
        if c.shape != (p,):
            raise ValueError("The contrast vector must have one entry per design column.")
    max_disp = max(max_disp, float(n))
    G = C.shape[1]

    if (C == 0).any(axis=0).all():
        sf = _iterative_size_factors(C, min_mu, min_disp, max_disp, beta_tol)
    else:
        sf = size_factors(C)
    F = _Fitter(X, sf, min_mu, min_disp, max_disp, beta_tol)
    nz = ~(C == 0).all(axis=0)
    Yall = C.T.astype(np.float64)
    Y = Yall[nz]
    base_mean = (Yall / sf).mean(axis=1)

    # Gene-wise dispersions, trend, prior, MAP, coefficients.
    mom, mu_hat = F.start_and_mu(Y)
    gw, gconv = F.log_disp(Y, mu_hat, mom)
    coeffs = _parametric_trend(base_mean[nz], gw) if fit_type == "parametric" else None
    if coeffs is not None:
        trend = "parametric"

        def trend_fn(m):
            return coeffs[0] + coeffs[1] / m
    else:
        trend, md = "mean", _mean_trend(gw, min_disp)

        def trend_fn(m):
            return np.full(len(m), md)
    fitted = trend_fn(base_mean[nz])
    slr, prior_var = _prior(gw, fitted, n, p, min_disp)
    mp, mconv, disp, fit = _map_and_lfc(F, Y, mu_hat, gw, fitted, slr, prior_var)
    beta = fit.beta.copy()

    # Cook's distances.
    rmm = _robust_mom_dispersions(Y / sf, X)
    V = fit.mu + rmm[:, None] * fit.mu ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        cooks = (Y - fit.mu) ** 2 / V / p * (fit.hat / (1 - fit.hat) ** 2)
    cutoff = float(stats.f.ppf(0.99, p, n - p))

    # Outlier replacement and refit.
    Gz = len(Y)
    replaced = np.zeros(Gz, dtype=bool)
    refitted = np.zeros(Gz, dtype=bool)
    new_zero = np.zeros(Gz, dtype=bool)
    bm = base_mean[nz].copy()
    replace_cooks = None
    replaceable = n_or_more_replicates(X, min_replicates)
    if refit_cooks and replaceable.any():
        idx = cooks > cutoff
        replaced = idx.any(axis=1)
        if replaced.any():
            rows = np.flatnonzero(replaced)
            Yr = Y[rows].copy()
            repl = (trimmed_mean(Yr / sf, 0.2)[:, None] * sf[None, :]).astype(np.int64)
            mask = replaceable[None, :] & idx[rows]
            Yr[mask] = repl[mask]
            zr = (Yr == 0).all(axis=1)
            new_zero[rows[zr]] = True
            bm[rows[zr]] = 0.0
            beta[rows[zr]] = 0.0
            sub = rows[~zr]
            refitted[sub] = True
            if sub.size:
                Ys = Yr[~zr]
                bms = (Ys / sf).mean(axis=1)
                mom2, mu2 = F.start_and_mu(Ys)
                gw2, gconv[sub] = F.log_disp(Ys, mu2, mom2)
                fit2_fitted = trend_fn(bms)
                mp2, mconv[sub], disp2, fit2 = _map_and_lfc(F, Ys, mu2, gw2, fit2_fitted, slr,
                                                            prior_var)
                bm[sub], beta[sub], gw[sub], fitted[sub] = bms, fit2.beta, gw2, fit2_fitted
                disp[sub], mp[sub] = disp2, mp2
                fit.converged[sub] = fit2.converged
                replace_cooks = cooks.copy()
                replace_cooks[np.ix_(sub, np.flatnonzero(replaceable))] = 0.0

    # Cook's outliers (cells with >= 3 samples).
    use_max = n_or_more_replicates(X, 3)
    layer = replace_cooks if (refit_cooks and refitted.any()) else cooks
    outl = (layer[:, use_max] > cutoff).any(axis=1)
    if outl.any():
        o = np.flatnonzero(outl)
        pos = np.argmax(cooks[o], axis=1)
        Yo = Y[o]
        outl[o] = (Yo > Yo[np.arange(len(o)), pos][:, None]).sum(axis=1) < 3

    # Wald test.
    mu = np.exp(beta @ X.T) * sf
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        se, st = nb_glm.wald(beta, mu, disp, X, c)
        pv = 2 * stats.norm.sf(np.abs(st))
    if refit_cooks and replaced.any():
        se[new_zero], st[new_zero], pv[new_zero] = 0.0, 0.0, 1.0
    if cooks_filter:
        pv = np.where(outl, np.nan, pv)

    def full(v, fill=np.nan, dtype=np.float64):
        out = np.full((G,) + v.shape[1:], fill, dtype=dtype)
        out[nz] = v
        return out

    base_mean[nz] = bm
    pvalue = full(pv)
    padj = filter_by_mean(base_mean, pvalue, alpha) if independent_filter else _bh(pvalue)
    B = full(beta)
    return DeseqResult(
        base_mean=base_mean, log2fc=(B @ c) / math.log(2), lfc_se=full(se) / math.log(2),
        stat=full(st), pvalue=pvalue, padj=padj, beta=B, size_factors=sf,
        dispersions=full(disp), genewise_dispersions=full(gw), fitted_dispersions=full(fitted),
        map_dispersions=full(mp), cooks_outlier=full(outl, False, bool),
        replaced=full(replaced, False, bool), refitted=full(refitted, False, bool),
        trend=trend, trend_coeffs=coeffs, prior_disp_var=float(prior_var),
        squared_logres=float(slr), genewise_converged=full(gconv, False, bool),
        map_converged=full(mconv, False, bool), lfc_converged=full(fit.converged, False, bool))
