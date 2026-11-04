"""edgeR v4's quasi-likelihood pipeline (glmQLFit with legacy=FALSE, then glmQLFTest) on
per-sample counts, vectorised over genes. Research build: not wired into pseudobulk DE yet.

    res = glm_ql(counts, design, coef=-1)          # counts: samples x genes
    res.logFC, res.F, res.pvalue, res.padj          # + dispersion, df_prior, s2_prior, s2_post, ...

The steps, in edgeR v4's order:

1. TMM factors (Robinson & Oshlack 2010) and AveLogCPM (log2 of an intercept-only NB mean with a
   library-scaled prior count of 2, dispersion 0.05).
2. One NB dispersion psi for every gene: the Cox-Reid common dispersion (McCarthy et al 2012) of
   the ``top_proportion`` most abundant genes, top_proportion = chooseLowessSpan(G sqrt(n - p),
   small.n 20, min.span 0.02) (the documented default), maximised over psi^(1/4) by Brent's method.
3. The QL-NB variance sigma^2 mu + psi mu^2 (Chen et al 2025): an average quasi-dispersion
   ``avg`` (two passes of the 90% quantile of a robust lowess trend of s2^(1/4) on AveLogCPM),
   then the GLM refitted at phi = psi / avg.
4. Adjusted deviances: each unit deviance is a scaled chi-square on fractional df matching its
   first two moments; df = (1 - h) k(mu/avg, psi), deviance = d_phi(y, mu) / s(mu/avg, psi), the
   exact NB moments read from a table (``UnitMoments``).
5. Empirical Bayes with unequal df (limma's fitFDistUnequalDF1): s2 ~ s0^2 F(d, d0) with
   log-scale moments for the scale (weights 1/trigamma(d/2), a weighted lowess trend on
   AveLogCPM) and profile likelihood for d0 (Smyth 2004, Phipson et al 2016).
6. Quasi F-test: LR (null model refitted at phi) / (df_test s2_post) on F(df_test, d0 + df),
   the denominator df capped at the pooled residual df G (n - p).

Provenance: written from the papers (Chen, Chen, Lun, Baldoni & Smyth, NAR 2025 gkaf018; Lund
et al 2012; Chen, Lun & Smyth 2016; Robinson & Oshlack 2010; McCarthy et al 2012; Smyth 2004;
Phipson et al 2016; Cleveland 1979/1981 for lowess), the Bioconductor help pages (argument
documentation) and black-box comparison with R edgeR 4.6.3 / limma 3.64.3: their outputs, and
the arguments and results of their internal calls, observed by instrumenting the call site,
never their source. No edgeR, limma, pyedger or inmoose code was read or copied. Each rule found
that way is marked "black-box" below. Agreement with R and the parts that differ: see
``tests/test_edger_ql.py`` and the module's report (median |d log10 p| ~2e-5 on Kang pseudobulk).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import special

# ---------------------------------------------------------------------------
# Small numerical helpers
# ---------------------------------------------------------------------------

_GOLD = 0.5 * (3.0 - math.sqrt(5.0))
_EPS = math.sqrt(np.finfo(float).eps)


def brent_min(f, a: float, b: float, tol: float) -> float:
    """Brent's (1973) golden-section / parabolic minimiser on [a, b] with absolute tolerance
    ``tol`` (the published algorithm, as used by R's ``optimize``)."""
    v = w = x = a + _GOLD * (b - a)
    fv = fw = fx = f(x)
    d = e = 0.0
    tol3 = tol / 3.0
    while True:
        xm = 0.5 * (a + b)
        tol1 = _EPS * abs(x) + tol3
        t2 = 2.0 * tol1
        if abs(x - xm) <= t2 - 0.5 * (b - a):
            return x
        p = q = r = 0.0
        if abs(e) > tol1:
            r = (x - w) * (fx - fv)
            q = (x - v) * (fx - fw)
            p = (x - v) * q - (x - w) * r
            q = 2.0 * (q - r)
            if q > 0.0:
                p = -p
            else:
                q = -q
            r = e
            e = d
        if abs(p) >= abs(0.5 * q * r) or p <= q * (a - x) or p >= q * (b - x):
            e = (b - x) if x < xm else (a - x)
            d = _GOLD * e
        else:
            d = p / q
            u = x + d
            if u - a < t2 or b - u < t2:
                d = tol1 if x < xm else -tol1
        u = x + d if abs(d) >= tol1 else (x + tol1 if d > 0.0 else x - tol1)
        fu = f(u)
        if fu <= fx:
            if u < x:
                b = x
            else:
                a = x
            v, fv, w, fw, x, fx = w, fw, x, fx, u, fu
        else:
            if u < x:
                a = u
            else:
                b = u
            if fu <= fw or w == x:
                v, fv, w, fw = w, fw, u, fu
            elif fu <= fv or v == x or v == w:
                v, fv = u, fu


def choose_lowess_span(n: int, small_n: float = 50, min_span: float = 0.3, power: float = 1 / 3) -> float:
    """limma's chooseLowessSpan (documented formula)."""
    if n <= small_n:
        return 1.0
    return float(min(min_span + (1 - min_span) * (small_n / n) ** power, 1.0))


# ---------------------------------------------------------------------------
# Negative binomial pieces
# ---------------------------------------------------------------------------


def unit_deviance(y: np.ndarray, mu: np.ndarray, phi) -> np.ndarray:
    """NB unit deviance 2[y log(y/mu) - (y + 1/phi) log((1 + phi y)/(1 + phi mu))]; 0 where mu = 0."""
    y = np.asarray(y, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    pos = mu > 0
    m = np.where(pos, mu, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = np.where(y > 0, y * np.log(np.where(y > 0, y, 1.0) / m), 0.0)
        t2 = (y + 1.0 / phi) * np.log1p(phi * (y - m) / (1.0 + phi * m))
        # y = 0: (2/phi) log(1 + phi mu) in plain floating point, which is exactly 0 once phi mu is
        # below machine precision (as edgeR's: matters only for fits with a diverging coefficient)
        d0 = (2.0 / phi) * np.log(1.0 + phi * m)
    return np.where(pos, np.where(y > 0, np.maximum(2.0 * (t1 - t2), 0.0), d0), 0.0)


def nb_loglik(y: np.ndarray, mu: np.ndarray, phi: float) -> np.ndarray:
    r = 1.0 / phi
    pos = mu > 0
    m = np.where(pos, mu, 1.0)
    ll = (special.gammaln(y + r) - special.gammaln(r) - special.gammaln(y + 1.0)
          - r * np.log1p(phi * m) + np.where(y > 0, y * np.log(phi * m / (1.0 + phi * m)), 0.0))
    return np.where(pos, ll, np.where(y > 0, -np.inf, 0.0))


# ---------------------------------------------------------------------------
# Moments of the NB unit deviance
# ---------------------------------------------------------------------------

_SUM_SD = 15.0  # window: mean +- 15 SD, and at least 40 decay lengths of the right tail
_N_INT = 400  # counts summed one by one; beyond them the sum is an integral (Simpson in log y)


def deviance_moments(m: np.ndarray, phi: float) -> tuple[np.ndarray, np.ndarray]:
    """Mean and variance of the unit deviance d(y, m, phi) for y ~ NB(mean m, dispersion phi).

    The pmf (gamma-function form) times d and d^2 is summed count by count over the first
    ``_N_INT`` counts of the window; the rest of a wider window is the integral from half a count
    before its start (midpoint rule, error ~ f''/24, negligible where f varies on a scale of tens
    of counts), by Simpson's rule on a log-spaced grid. Relative error < 1e-5 against brute-force
    sums for phi 0.005..3 and m 1e-6..2e4 (tests)."""
    m = np.atleast_1d(np.asarray(m, dtype=np.float64))
    r = 1.0 / phi
    sd = np.sqrt(m * (1.0 + phi * m))
    lo = np.maximum(0.0, np.floor(m - _SUM_SD * sd))
    hi = np.ceil(m + np.maximum(_SUM_SD * sd, 40.0 * (1.0 + phi * m)) + 15.0)
    k = np.arange(_N_INT, dtype=np.float64)
    ya = lo[:, None] + k[None, :]
    wa = (ya <= hi[:, None]).astype(np.float64)
    a = lo + _N_INT - 0.5  # integral part over [a, hi + 0.5] where needed
    need = hi > lo + _N_INT - 1
    b = np.where(need, hi + 0.5, a + 1.0)
    nb = _N_INT  # even number of Simpson intervals
    u = np.linspace(0.0, 1.0, nb + 1)
    la, lb = np.log(a), np.log(b)
    yb = np.exp(la[:, None] + (lb - la)[:, None] * u[None, :])
    simpson = np.ones(nb + 1)
    simpson[1:-1:2] = 4.0
    simpson[2:-1:2] = 2.0
    wb = np.where(need[:, None], yb * ((lb - la) / (3.0 * nb))[:, None] * simpson[None, :], 0.0)
    y = np.concatenate([ya, yb], axis=1)
    wt = np.concatenate([wa, wb], axis=1)
    mm = m[:, None]
    lp = (special.gammaln(y + r) - special.gammaln(r) - special.gammaln(y + 1.0)
          + r * np.log(r / (r + mm)) + y * np.log(mm / (r + mm)))
    p = wt * np.exp(lp)
    d = unit_deviance(y, mm, phi)
    tot = p.sum(axis=1)
    E = (p * d).sum(axis=1) / tot
    V = (p * (d - E[:, None]) ** 2).sum(axis=1) / tot
    return E, V


# ---------------------------------------------------------------------------
# Smoothers
# ---------------------------------------------------------------------------


def _anchor_index(xs: np.ndarray, delta: float) -> np.ndarray:
    """lowess anchors on sorted x: the first point, then each point more than ``delta`` beyond the
    previous anchor, and the last point."""
    n = len(xs)
    if delta <= 0:
        idx = np.flatnonzero(np.r_[True, np.diff(xs) > 0])
    else:
        idx = [0]
        last = xs[0]
        nxt = np.searchsorted(xs, last + delta, side="right")
        while nxt < n - 1:
            idx.append(nxt)
            last = xs[nxt]
            nxt = np.searchsorted(xs, last + delta, side="right")
        idx = np.asarray(idx)
    if idx[-1] != n - 1:
        idx = np.r_[idx, n - 1]
    return idx


def weighted_lowess(x: np.ndarray, y: np.ndarray, w: np.ndarray, span: float, delta: float) -> np.ndarray:
    """Local linear regression with prior weights (limma's weightedLowess, one iteration, as its
    help page describes it): each anchor's window is the smallest set of nearest neighbours whose
    prior weights reach ``span`` of the total; tricube distance weights times prior weights;
    anchors as in lowess (points more than ``delta`` beyond the last anchor); linear interpolation
    between anchors. Vectorised over anchors (black-box identical to limma to ~1e-14)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    n = len(x)
    o = np.argsort(x, kind="stable")
    xs, ys, ws = x[o], y[o], w[o]
    if n == 1:
        return y.copy()
    target = span * ws.sum()
    a = _anchor_index(xs, delta)
    x0 = xs[a]
    k = _kernels().get("wlowess")
    if k is not None:
        out = np.empty(n)
        out[o] = np.interp(xs, x0, k(xs, ys, ws, float(target), a.astype(np.int64)))
        return out
    cw = np.r_[0.0, np.cumsum(ws)]
    # window radius: the smallest r with weight(|x - x0| <= r) >= target, by bisection
    lo_r = np.zeros(len(a))
    hi_r = np.full(len(a), xs[-1] - xs[0] + 1.0)
    for _ in range(200):
        mid = 0.5 * (lo_r + hi_r)
        wt = cw[np.searchsorted(xs, x0 + mid, side="right")] - cw[np.searchsorted(xs, x0 - mid, side="left")]
        ok = wt >= target * (1 - 1e-15)
        hi_r = np.where(ok, mid, hi_r)
        lo_r = np.where(ok, lo_r, mid)
        if np.all(hi_r - lo_r <= 1e-15 * np.maximum(1.0, np.abs(x0))):
            break
    li = np.searchsorted(xs, x0 - hi_r, side="left")
    ri = np.searchsorted(xs, x0 + hi_r, side="right") - 1
    h = np.maximum(x0 - xs[li], xs[ri] - x0)
    D = np.abs(xs[None, :] - x0[:, None])
    with np.errstate(divide="ignore", invalid="ignore"):
        u = np.where(h[:, None] > 0, D / h[:, None], np.where(D == 0, 0.0, 2.0))
    tw = np.clip(1.0 - u**3, 0.0, None) ** 3
    ww = tw * ws[None, :]
    sw = ww.sum(axis=1)
    mx = (ww @ xs) / sw
    my = (ww @ ys) / sw
    dx = xs[None, :] - mx[:, None]
    vx = (ww * dx * dx).sum(axis=1)
    cxy = (ww * dx) @ ys
    lin = vx > 1e-12 * np.maximum(1.0, mx * mx) * sw
    fit = np.where(lin, my + np.where(lin, cxy / np.where(lin, vx, 1.0), 0.0) * (x0 - mx), my)
    out = np.empty(n)
    out[o] = np.interp(xs, x0, fit)
    return out


def lowess_delta(x: np.ndarray, npts: int = 200) -> float:
    """weightedLowess's anchor spacing for ``npts`` anchors: the d with
    ``npts * d = sum(min(gap, d))`` over the gaps between sorted distinct x (0 when there are no
    more than ``npts`` distinct values). Matches limma's delta (black-box, 8 cases)."""
    g = np.sort(np.diff(np.unique(np.asarray(x, dtype=np.float64))))[::-1]  # largest first
    m = len(g)
    if m + 1 <= npts:
        return 0.0
    tail = np.concatenate([np.cumsum(g[::-1])[::-1], [0.0]])  # tail[k] = sum(g[k:])
    for k in range(min(m, npts)):  # exactly the k largest gaps exceed d
        d = tail[k] / (npts - k)
        if (k == 0 or g[k - 1] > d) and d >= g[k]:
            return float(d)
    return float(tail[0] / npts)


def r_lowess(x: np.ndarray, y: np.ndarray, f: float, iterations: int = 3) -> np.ndarray:
    """R's lowess(x, y, f, iter) fitted values in the input order (statsmodels' implementation of
    the same Cleveland algorithm; delta = 1% of the range as R's default)."""
    from statsmodels.nonparametric.smoothers_lowess import lowess

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    k = _kernels().get("clowess")
    if k is not None:
        o = np.argsort(x, kind="stable")
        out = np.empty(len(x))
        out[o] = k(x[o], y[o], float(f), int(iterations), 0.01 * float(np.ptp(x)))
        return out
    fit = lowess(y, x, frac=f, it=iterations, delta=0.01 * float(np.ptp(x)), return_sorted=False)
    return np.asarray(fit, dtype=np.float64)


# ---------------------------------------------------------------------------
# NB GLM fitting (log link, offsets, a dispersion per gene or constant)
# ---------------------------------------------------------------------------


def design_groups(X: np.ndarray) -> np.ndarray | None:
    """Group label per sample when the design is a one-way layout (as many distinct rows as
    columns, full rank), else None."""
    _, inv = np.unique(np.round(X, 12), axis=0, return_inverse=True)
    inv = np.ravel(inv)
    if inv.max() + 1 == X.shape[1]:
        return inv
    return None


def _fit_one_group(y: np.ndarray, off: np.ndarray, phi, tol: float = 1e-11, maxit: int = 60) -> np.ndarray:
    """Intercept-only NB MLE per gene (columns of y: samples x genes). Returns beta (-inf if all 0)."""
    tot = y.sum(axis=0)
    eo = np.exp(off)
    b = np.log(np.maximum(tot, 1e-300) / eo.sum(axis=0))
    pos = tot > 0
    k = _kernels().get("one_group")
    if k is not None:
        ph = np.broadcast_to(np.asarray(phi, dtype=np.float64), b.shape).copy()
        return k(np.ascontiguousarray(y, dtype=np.float64), np.ascontiguousarray(off, dtype=np.float64), ph, b, tol, maxit)
    for _ in range(maxit):
        mu = np.exp(b + off)
        den = 1.0 + phi * mu
        score = ((y - mu) / den).sum(axis=0)
        info = (mu / den).sum(axis=0)
        step = np.where(pos, score / np.where(info > 0, info, 1.0), 0.0)
        b = b + step
        if np.max(np.abs(step)) < tol:
            break
    return np.where(pos, b, -np.inf)


def glm_fit(counts: np.ndarray, X: np.ndarray, offset: np.ndarray, phi, groups: np.ndarray | None = None,
            maxit: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """NB GLM fits for every gene (columns of ``counts``, samples x genes): (coefficients p x genes,
    fitted means samples x genes). ``offset`` is samples or samples x genes; ``phi`` scalar or per gene."""
    y = np.asarray(counts, dtype=np.float64)
    n, G = y.shape
    off = np.broadcast_to(np.asarray(offset, dtype=np.float64).reshape(n, -1), (n, G))
    phi = np.broadcast_to(np.asarray(phi, dtype=np.float64), (G,))
    if groups is None:
        groups = design_groups(X)
    if groups is not None:
        k = int(groups.max()) + 1
        gb = np.empty((k, G))
        mu = np.zeros((n, G))
        for j in range(k):
            idx = groups == j
            gb[j] = _fit_one_group(y[idx], off[idx], phi)
            mu[idx] = np.where(np.isfinite(gb[j]), np.exp(gb[j] + off[idx]), 0.0)
        # design coefficients from the group means: X_unique @ beta = group beta
        first = np.array([np.flatnonzero(groups == j)[0] for j in range(k)])
        Xu = X[first]
        with np.errstate(invalid="ignore"):
            coef = np.linalg.solve(Xu, np.where(np.isfinite(gb), gb, -1e8))
        return coef, mu
    return _glm_irls(y, X, off, phi, maxit)


def _glm_irls(y, X, off, phi, maxit, tol: float = 1e-6):
    """Fisher scoring vectorised over genes (general designs), from the null model of equal
    expression, halving a step that raises the deviance, stopping once the deviance falls by less
    than ``tol`` (absolute). That rule, as edgeR's mglmLevenberg (tol 1e-6), decides how far a
    coefficient that diverges (a level with all-zero counts) has travelled when the fit stops."""
    n, G = y.shape
    p = X.shape[1]
    beta = np.zeros((p, G))
    tot = y.sum(axis=0)
    beta[0] = np.log(np.maximum(tot, 1e-300) / np.exp(off).sum(axis=0))
    if not np.allclose(X[:, 0], 1.0):  # no intercept column: least squares on the null fit
        beta = np.linalg.lstsq(X, np.broadcast_to(beta[0], (n, G)), rcond=None)[0]
    mu = np.exp(X @ beta + off)
    dev = unit_deviance(y, mu, phi[None, :]).sum(axis=0)
    active = tot > 0
    mu[:, ~active] = 0.0
    idx = np.arange(p)
    for _ in range(maxit):
        if not active.any():
            break
        a = np.flatnonzero(active)
        ma, ya, oa, pa = mu[:, a], y[:, a], off[:, a], phi[a]
        W = ma / (1.0 + pa[None, :] * ma)
        z = (X @ beta[:, a]) + (ya - ma) / np.maximum(ma, 1e-300)
        XtWX = np.einsum("ni,ng,nj->gij", X, W, X)
        XtWz = np.einsum("ni,ng,ng->gi", X, W, z)
        XtWX[:, idx, idx] *= 1.0 + 1e-10
        XtWX[:, idx, idx] += 1e-300
        step = np.linalg.solve(XtWX, XtWz[..., None])[..., 0].T - beta[:, a]
        b_old, d_old = beta[:, a], dev[a]
        lam = np.ones(len(a))
        for _h in range(30):
            cand = b_old + lam[None, :] * step
            mu_c = np.exp(np.clip(X @ cand + oa, -700.0, 700.0))
            d_c = unit_deviance(ya, mu_c, pa[None, :]).sum(axis=0)
            bad = d_c > d_old
            if not bad.any():
                break
            lam = np.where(bad, lam * 0.5, lam)
        keep = d_c <= d_old
        cand = np.where(keep[None, :], cand, b_old)
        d_c = np.where(keep, d_c, d_old)
        beta[:, a] = cand
        mu[:, a] = np.exp(np.clip(X @ cand + oa, -700.0, 700.0))
        dev[a] = d_c
        active[a[(d_old - d_c) < tol]] = False
    return beta, mu


def leverage(mu: np.ndarray, X: np.ndarray, phi, groups: np.ndarray | None = None) -> np.ndarray:
    """Hat values of the working weighted least squares, W = mu / (1 + phi mu)."""
    phi = np.asarray(phi, dtype=np.float64)
    W = mu / (1.0 + phi * mu)
    if groups is not None:
        h = np.zeros_like(W)
        for j in range(int(groups.max()) + 1):
            idx = groups == j
            s = W[idx].sum(axis=0)
            h[idx] = np.where(s > 0, W[idx] / np.where(s > 0, s, 1.0), 0.0)
        return h
    A = np.einsum("ni,ng,nj->gij", X, W, X)
    Ai = np.linalg.pinv(A)
    return np.clip(W * np.einsum("ni,gij,nj->ng", X, Ai, X), 0.0, 1.0)


def _rank_window(v: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """v's elements whose rank (ties averaged, as R's rank) lies in [lo, hi]. Ranks rise with the
    value, so this is a value window, found on the sorted values."""
    sv = np.sort(v)
    starts = np.flatnonzero(np.r_[True, sv[1:] != sv[:-1]])
    ends = np.r_[starts[1:], len(sv)]
    rk = 0.5 * (starts + ends - 1) + 1.0  # rank of each distinct value
    ok = (rk >= lo) & (rk <= hi)
    if not ok.any():
        return np.zeros(len(v), dtype=bool)
    vals = sv[starts]
    return (v >= vals[ok][0]) & (v <= vals[ok][-1])


def tmm_factors(counts: np.ndarray, lib: np.ndarray) -> np.ndarray:
    """TMM normalisation factors (Robinson & Oshlack 2010; edgeR calcNormFactors defaults: 30% of
    log ratios and 5% of abundances trimmed each side, precision weights, reference = the sample
    whose upper-quartile count fraction is nearest the mean). Same numbers as
    ``voom.norm_factors(counts, "TMM", lib)``, vectorised over samples."""
    x = np.asarray(counts, dtype=np.float64)
    x = x[:, (x > 0).any(axis=0)]
    n = x.shape[0]
    f75 = np.quantile(x / lib[:, None], 0.75, axis=1)
    if np.median(f75) < 1e-20:
        ref = int(np.argmax(np.sqrt(x).sum(axis=1)))
    else:
        ref = int(np.argmin(np.abs(f75 - f75.mean())))
    with np.errstate(divide="ignore", invalid="ignore"):
        fr = x / lib[:, None]
        lo_ = np.log2(fr)
        log_r = np.log2(fr / fr[ref])  # this exact form: ties between equal ratios must stay ties
        abs_e = 0.5 * (lo_ + lo_[ref])
        v = (lib[:, None] - x) / lib[:, None] / x + ((lib[ref] - x[ref]) / lib[ref] / x[ref])[None, :]
    fin = np.isfinite(log_r) & np.isfinite(abs_e)
    f = np.ones(n)
    for i in range(n):
        lr, ae, vv = log_r[i, fin[i]], abs_e[i, fin[i]], v[i, fin[i]]
        m = len(lr)
        if m == 0 or np.max(np.abs(lr)) < 1e-6:
            continue
        lo_l = np.floor(m * 0.3) + 1
        lo_s = np.floor(m * 0.05) + 1
        keep = _rank_window(lr, lo_l, m + 1 - lo_l) & _rank_window(ae, lo_s, m + 1 - lo_s)
        if keep.any():
            val = float(np.sum(lr[keep] / vv[keep]) / np.sum(1.0 / vv[keep]))
            f[i] = 2.0**val if np.isfinite(val) else 1.0
    return f / np.exp(np.mean(np.log(f)))


def nb_deviance(y: np.ndarray, mu: np.ndarray, phi) -> np.ndarray:
    return unit_deviance(y, mu, np.asarray(phi, dtype=np.float64)).sum(axis=0)


def add_prior_count(y: np.ndarray, lib: np.ndarray, prior_count: float) -> tuple[np.ndarray, np.ndarray]:
    """edgeR's addPriorCount (as documented): prior scaled by library size, library sizes
    increased by twice the scaled prior. Returns (y + prior, log offsets)."""
    pc = prior_count * lib / lib.mean()
    return y + pc[:, None], np.log(lib + 2.0 * pc)


def ave_log_cpm(counts: np.ndarray, lib: np.ndarray, prior_count: float = 2.0, dispersion: float = 0.05) -> np.ndarray:
    """edgeR's aveLogCPM: log2 of the NB intercept-only mean per million after adding a prior count."""
    y, off = add_prior_count(np.asarray(counts, dtype=np.float64), lib, prior_count)
    b = _fit_one_group(y, np.broadcast_to(off[:, None], y.shape), dispersion, tol=1e-13)
    return (b + math.log(1e6)) / math.log(2.0)


def _cox_reid_apl(y, X, off, phi, groups):
    _, mu = glm_fit(y, X, off, phi, groups)
    ll = nb_loglik(y, mu, phi).sum(axis=0)
    W = mu / (1.0 + phi * mu)
    A = np.einsum("ni,ng,nj->gij", X, W, X)
    return float(np.sum(ll - 0.5 * _chol_logdet(A)))


_PIVOT_FLOOR = 1e-20  # Cholesky pivots below this are set to it (edgeR's APL, black-box)
_PIVOT_REL = 1e-12  # ... and pivots below this share of their diagonal (rounding noise)


def _chol_logdet(A: np.ndarray) -> np.ndarray:
    """log det of each p x p matrix in A (G x p x p) by Cholesky, flooring pivots at 1e-20, so a
    level with no expression (zero weights) adds a constant instead of -inf."""
    G, p, _ = A.shape
    L = np.zeros_like(A)
    out = np.zeros(G)
    for j in range(p):
        d = A[:, j, j] - np.sum(L[:, j, :j] ** 2, axis=1)
        # a pivot that is rounding noise (a level with no weight) is set to the floor too
        d = np.where(d < np.maximum(_PIVOT_FLOOR, _PIVOT_REL * A[:, j, j]), _PIVOT_FLOOR, d)
        out += np.log(d)
        L[:, j, j] = np.sqrt(d)
        for i in range(j + 1, p):
            L[:, i, j] = (A[:, i, j] - np.sum(L[:, i, :j] * L[:, j, :j], axis=1)) / L[:, j, j]
    return out


def common_dispersion(counts: np.ndarray, X: np.ndarray, offset: np.ndarray, groups=None,
                      interval: tuple[float, float] = (0.0, 4.0), tol: float = 1e-5, min_row_sum: float = 5) -> float:
    """Cox-Reid common dispersion (McCarthy et al 2012): maximise the summed adjusted profile
    likelihood over phi^(1/4) in interval^(1/4) with Brent's method (as edgeR's dispCoxReid)."""
    y = np.asarray(counts, dtype=np.float64)
    y = y[:, y.sum(axis=0) >= min_row_sum]
    off = np.broadcast_to(np.asarray(offset, dtype=np.float64)[:, None], y.shape)
    lo, hi = interval[0] ** 0.25, interval[1] ** 0.25
    kern = _kernels().get("apl_oneway")
    if kern is not None and groups is not None:
        yc, oc = np.ascontiguousarray(y), np.ascontiguousarray(off)
        gr = np.asarray(groups, dtype=np.int64)
        k = int(gr.max()) + 1
        gb = np.full((k, y.shape[1]), np.nan)
        Xc = np.ascontiguousarray(X, dtype=np.float64)

        def f(p):
            return -kern(yc, oc, gr, k, Xc, max(p, 1e-300) ** 4, gb, 1e-11, 60, _PIVOT_FLOOR)
    else:
        def f(p):
            return -_cox_reid_apl(y, X, off, max(p, 1e-300) ** 4, groups)
    par = brent_min(f, lo, hi, tol)
    return float(par**4)


def top_proportion(n_genes: int, df_residual: int) -> float:
    return choose_lowess_span(n_genes * math.sqrt(df_residual), small_n=20, min_span=0.02)


# ---------------------------------------------------------------------------
# Adjusted deviances and residual df (edgeR v4)
# ---------------------------------------------------------------------------

_MIN_DF = 1e-8
_TINY_MU = 1e-6
_GRID_STEP = 0.25  # spacing of the moment table in log(mean)
_M_FLOOR = 1e-10


class UnitMoments:
    """Scaled chi-square approximation of the NB unit deviance at dispersion ``psi``: for mean m,
    df k(m) = 2E^2/V and scale s(m) = V/(2E) from the exact moments (``deviance_moments``) on a
    log-spaced table, read by a cubic spline in log m on log E and log V. The table grows to cover
    the means asked for; below ``_M_FLOOR`` k is taken proportional to m (its limit) and s fixed."""

    def __init__(self, psi: float, step: float = _GRID_STEP):
        self.psi = float(psi)
        self.step = step
        self.t = None

    def _build(self, lo: float, hi: float) -> None:
        from scipy.interpolate import CubicSpline

        k = max(int(math.ceil((hi - lo) / self.step)), 3)
        self.t = np.linspace(lo, hi, k + 1)
        self.dt = self.t[1] - self.t[0]
        E, V = deviance_moments(np.exp(self.t), self.psi)
        self._cE = CubicSpline(self.t, np.log(E)).c
        self._cV = CubicSpline(self.t, np.log(V)).c

    def _eval(self, c: np.ndarray, lt: np.ndarray, idx: np.ndarray, dx: np.ndarray) -> np.ndarray:
        return ((c[0, idx] * dx + c[1, idx]) * dx + c[2, idx]) * dx + c[3, idx]

    def ensure(self, lo: float, hi: float) -> None:
        """Make the table cover log means lo..hi."""
        if self.t is None or lo < self.t[0] or hi > self.t[-1]:
            a = lo - self.step if self.t is None else min(lo - self.step, self.t[0])
            b = hi + self.step if self.t is None else max(hi + self.step, self.t[-1])
            self._build(a, b)

    def __call__(self, m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        m = np.asarray(m, dtype=np.float64)
        lt = np.log(np.maximum(m, _M_FLOOR))
        self.ensure(float(lt.min()), float(lt.max()))
        idx = np.clip(((lt - self.t[0]) / self.dt).astype(np.int64), 0, len(self.t) - 2)
        dx = lt - self.t[idx]
        E = np.exp(self._eval(self._cE, lt, idx, dx))
        V = np.exp(self._eval(self._cV, lt, idx, dx))
        k = 2.0 * E * E / V
        k = np.where(m < _M_FLOOR, k * m / _M_FLOOR, k)
        return k, V / (2.0 * E)


def adjusted_deviance(y: np.ndarray, mu: np.ndarray, X: np.ndarray, psi: float, avg: float,
                      groups: np.ndarray | None = None, um: UnitMoments | None = None
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """edgeR v4 adjusted unit deviances and df (black-box verified against glmQLFit keep.unit.mat):
    the GLM is at phi = psi/avg; m = mu/avg; unit df = (1 - h) k(m, psi), unit deviance
    d_phi(y, mu) / s(m, psi). Returns (deviance.adj, df.residual.adj, unit dev adj, unit df adj)."""
    phi = psi / avg
    pos = mu > 0
    um = UnitMoments(psi) if um is None else um
    h = leverage(mu, X, phi, groups)
    kern = _kernels().get("adjust")
    if kern is not None and pos.any():
        mp = mu[pos]
        um.ensure(math.log(max(float(mp.min()) / avg, _M_FLOOR)), math.log(max(float(mp.max()) / avg, _M_FLOOR)))
        uda, udf = kern(np.ascontiguousarray(y, dtype=np.float64), np.ascontiguousarray(mu), np.ascontiguousarray(h),
                        float(phi), float(avg), float(um.t[0]), float(um.dt), um._cE, um._cV, _M_FLOOR)
        return uda.sum(axis=0), udf.sum(axis=0), uda, udf
    m = np.where(pos, mu / avg, 1.0)
    k, s = um(m)
    udf = np.where(pos, (1.0 - h) * k, 0.0)
    uda = np.where(pos, unit_deviance(y, mu, phi) / s, 0.0)
    return uda.sum(axis=0), udf.sum(axis=0), uda, udf


def average_ql_dispersion(y, mu, X, psi, ave_log_cpm_, groups=None, iterations: int = 2,
                          um: UnitMoments | None = None) -> float:
    """edgeR v4's average quasi-dispersion (black-box rule, matches .cxx_compute_ave_qd exactly when
    fed R's adjusted deviances): start at 1; each iteration takes the 90% quantile of a robust lowess
    trend (span 0.5, 3 robustness iterations) of the adjusted s2^(1/4) against AveLogCPM, to the 4th."""
    a = 1.0
    um = UnitMoments(psi) if um is None else um
    tiny = mu < _TINY_MU
    for _ in range(iterations):
        _D, F, uda, _f = adjusted_deviance(y, mu, X, psi, a, groups, um)
        # units fitted at (almost) zero by a diverging coefficient add df but no deviance here, so a
        # gene made only of perfect fits and such units has s2 = 0 (as edgeR's, black-box; such
        # genes arise only in designs that are not one-way)
        D = np.where(tiny, 0.0, uda).sum(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            s2 = D / F
        ok = F > _MIN_DF  # genes with (almost) no residual df are left out of the trend
        tr = r_lowess(ave_log_cpm_[ok], s2[ok] ** 0.25, 0.5, 3)
        a = max(float(np.quantile(tr, 0.9)) ** 4, 1.0)
    return a


# ---------------------------------------------------------------------------
# Empirical Bayes with unequal df (limma's fitFDistUnequalDF1, black-box)
# ---------------------------------------------------------------------------

_U_RANGE = (0.5, 0.9998)  # u = d0 / (d0 + 2)
_U_TOL = 2.0**-13
_EB_MIN_DF = 0.01


def fit_f_dist_unequal_df(x: np.ndarray, df1: np.ndarray, covariate: np.ndarray | None = None,
                          span: float | None = None) -> tuple[np.ndarray, float]:
    """(scale per gene, df2) of x_g ~ scale_g F(df1_g, df2): log-scale weighted moments for the
    scale (weights 1/trigamma(df1/2), a weighted lowess trend in the covariate) and profile
    maximum likelihood for df2."""
    x = np.asarray(x, dtype=np.float64)
    d = np.asarray(df1, dtype=np.float64)
    use = np.isfinite(x) & (d >= _EB_MIN_DF)  # below that df a gene gets weight 0 (black-box)
    x = np.where(use, x, 1.0)
    pos = x[use] > 0
    if not pos.all():  # zero variances: floored at 1e-12 x median (black-box, approximately)
        x = np.where(use & (x <= 0), 1e-12 * np.median(x[use]), x)
    dd = np.where(use, d, 1.0)
    z = np.log(x) - special.digamma(dd / 2) + np.log(dd / 2)
    w = np.where(use, 1.0 / special.polygamma(1, dd / 2), 0.0)
    if covariate is None:
        trend = np.full_like(z, np.sum(w * z) / np.sum(w))
    else:
        cov = np.asarray(covariate, dtype=np.float64)
        sp = choose_lowess_span(len(x), small_n=500) if span is None else span
        trend = weighted_lowess(cov, z, w, sp, lowess_delta(cov, 200))
    xu, du, tu = x[use], d[use], trend[use]

    def nll(u):
        d0 = 2 * u / (1 - u)
        ls = tu + special.digamma(d0 / 2) - math.log(d0 / 2)
        return -np.sum(_f_logpdf(xu * np.exp(-ls), du, d0) - ls)

    u = brent_min(nll, _U_RANGE[0], _U_RANGE[1], _U_TOL)
    d0 = 2 * u / (1 - u)
    scale = np.exp(trend + special.digamma(d0 / 2) - math.log(d0 / 2))
    return scale, float(d0)


def _f_logpdf(q, d1, d2):
    return (0.5 * d1 * np.log(d1 / d2) + (0.5 * d1 - 1) * np.log(q) - 0.5 * (d1 + d2) * np.log1p(d1 * q / d2)
            - special.betaln(0.5 * d1, 0.5 * d2))


# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------


@dataclass
class QLResult:
    """Per gene (arrays of length G) unless noted."""

    logFC: np.ndarray  # log2 fold change from the fit with prior count 0.125 (as topTags)
    F: np.ndarray  # quasi-likelihood F statistic
    pvalue: np.ndarray
    padj: np.ndarray  # Benjamini-Hochberg
    AveLogCPM: np.ndarray
    dispersion: float  # the NB dispersion psi (fit$dispersion)
    average_ql_dispersion: float
    df_prior: float | np.ndarray  # a vector when robust
    s2_prior: np.ndarray
    s2_post: np.ndarray
    df_resid: np.ndarray  # adjusted residual df (fit$df.residual.adj)
    deviance_adj: np.ndarray
    norm_factors: np.ndarray  # per sample
    fitted: np.ndarray  # samples x genes, at phi = psi / avg
    coefficients: np.ndarray  # p x genes, natural log scale, unshrunk


def glm_ql(counts: np.ndarray, X: np.ndarray, coef: int = -1, *, robust: bool = False,
           lib_size: np.ndarray | None = None, norm_factors: np.ndarray | None = None) -> QLResult:
    """edgeR v4 quasi-likelihood test of one coefficient: ``glmQLFit(y, X)`` (legacy=FALSE, NB
    dispersion from the most abundant genes, abundance trend) then ``glmQLFTest(fit, coef)``.

    ``counts`` is samples x genes (raw counts), ``X`` samples x coefficients of full column rank
    with at least two columns; ``coef`` indexes X's columns (Python order, -1 = last).
    ``lib_size`` defaults to the column sums and ``norm_factors`` to TMM, as
    ``calcNormFactors(DGEList(t(counts)))``."""
    from scipy import stats

    C = np.asarray(counts, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    if C.ndim != 2 or X.ndim != 2 or C.shape[0] != X.shape[0]:
        raise ValueError("counts (samples x genes) and X (samples x coefficients) need one row per sample.")
    n, G = C.shape
    p = X.shape[1]
    if p < 2 or np.linalg.matrix_rank(X) < p or n <= p:
        raise ValueError("The design needs >= 2 columns, full column rank and more samples than columns.")
    if robust:
        raise NotImplementedError("robust=True is not implemented yet.")
    coef = coef % p
    lib0 = C.sum(axis=1) if lib_size is None else np.asarray(lib_size, dtype=np.float64)
    nf = tmm_factors(C, lib0) if norm_factors is None else np.asarray(norm_factors, dtype=np.float64)
    lib = lib0 * nf
    off = np.log(lib)
    alc = ave_log_cpm(C, lib)
    groups = design_groups(X)
    ntop = int(math.ceil(top_proportion(G, n - p) * G))
    top = np.argsort(-alc, kind="stable")[:ntop]
    psi = common_dispersion(C[:, top], X, off, groups)
    _b1, mu1 = glm_fit(C, X, off, psi, groups)
    um = UnitMoments(psi)
    avg = average_ql_dispersion(C, mu1, X, psi, alc, groups, um=um)
    phi = psi / avg
    beta, mu = glm_fit(C, X, off, phi, groups)
    D, F, _uda, _udf = adjusted_deviance(C, mu, X, psi, avg, groups, um)
    with np.errstate(divide="ignore", invalid="ignore"):
        s2 = D / F
    s2_prior, d0 = fit_f_dist_unequal_df(s2, F, alc)
    s2_post = np.where(F > 0, (F * np.nan_to_num(s2) + d0 * s2_prior) / (F + d0), s2_prior)
    keep = [j for j in range(p) if j != coef]
    X0 = X[:, keep]
    _b0, mu0 = glm_fit(C, X0, off, phi, design_groups(X0))
    lr = nb_deviance(C, mu0, phi) - nb_deviance(C, mu, phi)
    Fstat = np.maximum(lr, 0.0) / s2_post
    # capped at the pooled residual df of all genes (black-box: glmQLFTest's df.total)
    df_total = np.minimum(d0 + F, float(G * (n - p)))
    pval = stats.f.sf(Fstat, 1, df_total)
    yp, offp = add_prior_count(C, lib, 0.125)
    bs, _ = glm_fit(yp, X, offp, phi, groups)
    padj = np.full(G, np.nan)
    fin = np.isfinite(pval)
    padj[fin] = stats.false_discovery_control(pval[fin], method="bh")
    return QLResult(logFC=bs[coef] / math.log(2.0), F=Fstat, pvalue=pval, padj=padj, AveLogCPM=alc,
                    dispersion=psi, average_ql_dispersion=avg, df_prior=d0, s2_prior=s2_prior, s2_post=s2_post,
                    df_resid=F, deviance_adj=D, norm_factors=nf, fitted=mu, coefficients=beta)


# ---------------------------------------------------------------------------
# Compiled kernels (numba, which comes with scanpy); the numpy paths remain when it is missing
# ---------------------------------------------------------------------------

_KERNELS: dict | None = None


def _wlowess_loop(xs, ys, ws, target, anchors):
    """weightedLowess fitted values at the anchors (inputs sorted by x)."""
    n = xs.shape[0]
    out = np.empty(anchors.shape[0])
    for k in range(anchors.shape[0]):
        i = anchors[k]
        x0 = xs[i]
        lo = i
        hi = i
        tot = ws[i]
        # grow the window from the anchor, nearer side first, until it holds the target weight
        while tot < target and (lo > 0 or hi < n - 1):
            dl = x0 - xs[lo - 1] if lo > 0 else np.inf
            dr = xs[hi + 1] - x0 if hi < n - 1 else np.inf
            if dl <= dr:
                lo -= 1
                tot += ws[lo]
            else:
                hi += 1
                tot += ws[hi]
        h = max(x0 - xs[lo], xs[hi] - x0)
        sw = 0.0
        sx = 0.0
        sxx = 0.0
        sy = 0.0
        sxy = 0.0
        for j in range(lo, hi + 1):
            tw = 1.0
            if h > 0:
                u = abs(xs[j] - x0) / h
                t = 1.0 - u * u * u
                tw = t * t * t if t > 0 else 0.0
            wj = tw * ws[j]
            xj = xs[j]
            sw += wj
            sx += wj * xj
            sxx += wj * xj * xj
            sy += wj * ys[j]
            sxy += wj * xj * ys[j]
        mx = sx / sw
        my = sy / sw
        vx = sxx - sw * mx * mx
        if vx > 1e-12 * max(1.0, mx * mx) * sw:
            out[k] = my + (sxy - sw * mx * my) / vx * (x0 - mx)
        else:
            out[k] = my
    return out


def _one_group_loop(y, off, phi, b, tol, maxit):
    """Intercept-only NB MLE per gene (columns of y): Newton's method on the score, which is
    decreasing in b, kept inside a bracket of the root (bisection when a step leaves it);
    -inf where all counts are zero."""
    n, G = y.shape
    for g in range(G):
        tot = 0.0
        for i in range(n):
            tot += y[i, g]
        if tot <= 0:
            b[g] = -np.inf
            continue
        bg = b[g]
        lo = -np.inf
        hi = np.inf
        for _ in range(maxit):
            sc = 0.0
            dsc = 0.0
            for i in range(n):
                mu = math.exp(bg + off[i, g])
                den = 1.0 + phi[g] * mu
                sc += (y[i, g] - mu) / den
                dsc += mu * (1.0 + phi[g] * y[i, g]) / (den * den)
            if sc > 0:
                lo = bg
            else:
                hi = bg
            step = sc / dsc
            if abs(step) < tol:
                bg += step
                break
            new = bg + step
            if not (lo < new < hi):
                if np.isfinite(lo) and np.isfinite(hi):
                    new = 0.5 * (lo + hi)
                else:
                    new = bg + (1.0 if sc > 0 else -1.0)
            if abs(new - bg) < tol:
                bg = new
                break
            bg = new
        b[g] = bg
    return b


def _clowess_loop(x, y, f, nsteps, delta):
    """Cleveland's (1979, 1981) LOWESS on sorted x: local linear fits with tricube weights over the
    f*n nearest points, ``nsteps`` robustness iterations with bisquare weights (6 x median absolute
    residual), fits only at points more than ``delta`` apart, linear interpolation between."""
    n = x.shape[0]
    ys = np.empty(n)
    rw = np.ones(n)
    res = np.empty(n)
    ns = max(2, min(n, int(f * n + 1e-7)))
    xrange = x[n - 1] - x[0]
    for it in range(nsteps + 1):
        nleft = 0
        nright = ns - 1
        last = -1
        i = 0
        while True:
            if nright < n - 1:
                d1 = x[i] - x[nleft]
                d2 = x[nright + 1] - x[i]
                if d1 > d2:
                    nleft += 1
                    nright += 1
                    continue
            # local fit at x[i]: one pass for the weighted moments of the window
            xs = x[i]
            h = max(xs - x[nleft], x[nright] - xs)
            h9 = 0.999 * h
            h1 = 0.001 * h
            sw = 0.0
            swx = 0.0
            swxx = 0.0
            swy = 0.0
            swxy = 0.0
            j = nleft
            while j < n:
                r = abs(x[j] - xs)
                if r <= h9:
                    if r <= h1:
                        wj = 1.0
                    else:
                        q = r / h
                        q = 1.0 - q * q * q
                        wj = q * q * q
                    if it > 0:
                        wj *= rw[j]
                    xj = x[j]
                    sw += wj
                    swx += wj * xj
                    swxx += wj * xj * xj
                    swy += wj * y[j]
                    swxy += wj * xj * y[j]
                elif x[j] > xs:
                    break
                j += 1
            if sw <= 0.0:
                ys[i] = y[i]
            else:
                a = swx / sw
                ybar = swy / sw
                fit = ybar
                if h > 0.0:
                    c = swxx / sw - a * a
                    if c > 0.0 and math.sqrt(c) > 0.001 * xrange:
                        fit = ybar + (xs - a) / c * (swxy / sw - a * ybar)
                ys[i] = fit
            if last < i - 1:
                denom = x[i] - x[last]
                for j in range(last + 1, i):
                    alpha = (x[j] - x[last]) / denom
                    ys[j] = alpha * ys[i] + (1.0 - alpha) * ys[last]
            last = i
            cut = x[last] + delta
            i = last + 1
            while i < n:
                if x[i] > cut:
                    break
                if x[i] == x[last]:
                    ys[i] = ys[last]
                    last = i
                i += 1
            i = max(last + 1, i - 1)
            if last >= n - 1:
                break
        for i in range(n):
            res[i] = y[i] - ys[i]
        if it >= nsteps:
            break
        sc = 0.0
        for i in range(n):
            sc += abs(res[i])
        sc /= n
        ar = np.abs(res)
        srt = np.sort(ar)
        m1 = n // 2
        if n % 2 == 0:
            cmad = 3.0 * (srt[m1 - 1] + srt[m1])
        else:
            cmad = 6.0 * srt[m1]
        if cmad < 1e-7 * sc:
            break
        c9 = 0.999 * cmad
        c1 = 0.001 * cmad
        for i in range(n):
            r = ar[i]
            if r <= c1:
                rw[i] = 1.0
            elif r <= c9:
                q = r / cmad
                q = 1.0 - q * q
                rw[i] = q * q
            else:
                rw[i] = 0.0
    return ys


def _adjust_loop(y, mu, h, phi, avg, t0, dt, cE, cV, mfloor):
    """Per unit: df (1 - h) k(m) and deviance d_phi(y, mu) / s(m), m = mu / avg, k and s from the
    moment table's cubic pieces (log E, log V on an even grid in log m)."""
    n, G = y.shape
    uda = np.zeros((n, G))
    udf = np.zeros((n, G))
    last = cE.shape[1] - 1
    for i in range(n):
        for g in range(G):
            m_ = mu[i, g]
            if m_ <= 0.0:
                continue
            mm = m_ / avg
            lt = math.log(max(mm, mfloor))
            j = int((lt - t0) / dt)
            if j < 0:
                j = 0
            elif j > last:
                j = last
            dx = lt - (t0 + j * dt)
            lE = ((cE[0, j] * dx + cE[1, j]) * dx + cE[2, j]) * dx + cE[3, j]
            lV = ((cV[0, j] * dx + cV[1, j]) * dx + cV[2, j]) * dx + cV[3, j]
            E = math.exp(lE)
            V = math.exp(lV)
            k = 2.0 * E * E / V
            if mm < mfloor:
                k *= mm / mfloor
            sc = V / (2.0 * E)
            yy = y[i, g]
            if yy > 0:
                t1 = yy * math.log(yy / m_)
                t2 = (yy + 1.0 / phi) * math.log1p(phi * (yy - m_) / (1.0 + phi * m_))
                d = 2.0 * (t1 - t2)
                if d < 0.0:
                    d = 0.0
            else:
                d = (2.0 / phi) * math.log(1.0 + phi * m_)
            uda[i, g] = d / sc
            udf[i, g] = (1.0 - h[i, g]) * k
    return uda, udf


def _apl_oneway_loop(y, off, grp, ngrp, X, phi, gb, tol, maxit, floor):
    """Summed Cox-Reid adjusted profile log-likelihood at dispersion phi for a one-way design:
    each group's mean by Newton (from gb, updated in place), NB log-likelihood, and
    -0.5 log det(X'WX) by Cholesky with pivots floored at ``floor``."""
    n, G = y.shape
    p = X.shape[1]
    r = 1.0 / phi
    lgr = math.lgamma(r)
    total = 0.0
    mu = np.empty(n)
    A = np.empty((p, p))
    L = np.empty((p, p))
    for g in range(G):
        for j in range(ngrp):
            tot = 0.0
            for i in range(n):
                if grp[i] == j:
                    tot += y[i, g]
            if tot <= 0:
                gb[j, g] = -np.inf
                continue
            b = gb[j, g]
            if not np.isfinite(b):
                s0 = 0.0
                for i in range(n):
                    if grp[i] == j:
                        s0 += math.exp(off[i, g])
                b = math.log(tot / s0)
            for _ in range(maxit):
                sc = 0.0
                dsc = 0.0
                for i in range(n):
                    if grp[i] == j:
                        m_ = math.exp(b + off[i, g])
                        den = 1.0 + phi * m_
                        sc += (y[i, g] - m_) / den
                        dsc += m_ * (1.0 + phi * y[i, g]) / (den * den)
                step = sc / dsc
                if step > 5.0:
                    step = 5.0
                elif step < -5.0:
                    step = -5.0
                b += step
                if abs(step) < tol:
                    break
            gb[j, g] = b
        ll = 0.0
        for i in range(n):
            bj = gb[grp[i], g]
            yy = y[i, g]
            if np.isfinite(bj):
                m_ = math.exp(bj + off[i, g])
                ll += (math.lgamma(yy + r) - lgr - math.lgamma(yy + 1.0) - r * math.log1p(phi * m_))
                if yy > 0:
                    ll += yy * math.log(phi * m_ / (1.0 + phi * m_))
                mu[i] = m_
            else:
                mu[i] = 0.0
        for a in range(p):
            for c in range(p):
                A[a, c] = 0.0
        for i in range(n):
            w = mu[i] / (1.0 + phi * mu[i])
            for a in range(p):
                for c in range(p):
                    A[a, c] += w * X[i, a] * X[i, c]
        logdet = 0.0
        for c in range(p):
            d = A[c, c]
            for q in range(c):
                d -= L[c, q] * L[c, q]
            if d < floor or d < 1e-12 * A[c, c]:
                d = floor
            logdet += math.log(d)
            L[c, c] = math.sqrt(d)
            for a in range(c + 1, p):
                v = A[a, c]
                for q in range(c):
                    v -= L[a, q] * L[c, q]
                L[a, c] = v / L[c, c]
        total += ll - 0.5 * logdet
    return total


def _kernels() -> dict:
    global _KERNELS
    if _KERNELS is None:
        try:
            import numba

            jit = numba.njit(nogil=True, cache=True)
            _KERNELS = {"wlowess": jit(_wlowess_loop), "one_group": jit(_one_group_loop),
                        "clowess": jit(_clowess_loop), "adjust": jit(_adjust_loop),
                        "apl_oneway": jit(_apl_oneway_loop)}
        except Exception:  # noqa: BLE001 - numba missing or failing: the numpy paths
            _KERNELS = {}
    return _KERNELS
