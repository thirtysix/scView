"""Negative binomial GLM pieces vectorised over genes: the shared core of the
pseudobulk tests (core/deseq2_vec.py; an edgeR-style quasi-likelihood test is
meant to sit on the same functions).

Layout: counts ``Y`` are genes x samples (one gene per row), the design ``X``
samples x p, offsets (log size factors) per sample ``(n,)`` or per gene and
sample ``(G, n)``, dispersions ``alpha`` per gene, with
``Var(y) = mu + alpha * mu^2`` and ``log mu = offset + X @ beta``.

* ``irls``: Fisher scoring for beta at fixed dispersions, as pydeseq2 0.5.4's
  ``irls_solver`` runs it (MIT; QR start on log(y / s + 0.1), ridge 1e-6,
  fitted means floored at ``min_mu`` while fitting, stop on the relative
  change of -2 log-likelihood), with its per-gene L-BFGS-B fallback for the
  genes that diverge or run out of iterations.
* ``nb_nll`` / ``nb_deviance``: negative log-likelihood and deviance per gene.
* ``cox_reid`` / ``cox_reid_derivs``: the Cox-Reid adjustment -1/2 log det(X'WX)
  and its first two derivatives in log(alpha); ``nb_loglik_derivs`` the same
  for the log-likelihood. Together they give the adjusted profile likelihood
  of the dispersion (Cox & Reid 1987; McCarthy et al. 2012; Love et al. 2014).
* ``maximise_1d``: a bracketed, safeguarded Newton search for one maximum per
  gene from derivatives only, in place of one scalar optimiser per gene.
* ``wald``: Wald standard errors and statistics of a contrast.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy import optimize, special

MIN_MU = 0.5
RIDGE = 1e-6
BETA_TOL = 1e-8
MAX_ITER = 250
MAX_BETA = 30.0


def outer_design(X: np.ndarray) -> np.ndarray:
    """Per sample, the flattened outer product x x' (n x p*p): ``W @ outer``
    gives every gene's X'WX at once."""
    n, p = X.shape
    return (X[:, :, None] * X[:, None, :]).reshape(n, p * p)


def info_matrix(W: np.ndarray, XX: np.ndarray, p: int) -> np.ndarray:
    """X' diag(w) X per gene (genes x p x p) from weights ``W`` (genes x n)."""
    return (W @ XX).reshape(-1, p, p)


def _offsets(offset: np.ndarray, G: int, n: int) -> np.ndarray:
    off = np.asarray(offset, dtype=np.float64)
    return np.broadcast_to(off, (G, n)) if off.ndim == 1 else off


def nb_nll(Y: np.ndarray, mu: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Negative log-likelihood per gene (pydeseq2's ``nb_nll`` form)."""
    n = Y.shape[1]
    a = np.asarray(alpha, dtype=np.float64)
    r = (1 / a)[:, None]
    logbinom = special.gammaln(Y + r) - special.gammaln(Y + 1) - special.gammaln(r)
    return n * (1 / a) * np.log(a) + (
        -logbinom + (Y + r) * np.log(r + mu) - Y * np.log(mu)).sum(axis=1)


def nb_deviance(Y: np.ndarray, mu: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Unit deviances summed per gene: 2 * sum(y log(y/mu) - (y + 1/a) log((1 + a y)/(1 + a mu)))."""
    a = np.asarray(alpha, dtype=np.float64)[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = np.where(Y > 0, Y * np.log(Y / mu), 0.0)
    t2 = (Y + 1 / a) * (np.log1p(a * Y) - np.log1p(a * mu))
    return 2 * (t1 - t2).sum(axis=1)


def hat_diagonals(W: np.ndarray, XX: np.ndarray, p: int, ridge: float = RIDGE) -> np.ndarray:
    """Diagonal of W^1/2 X (X'WX + ridge)^-1 X' W^1/2 per gene (genes x n)."""
    Hinv = np.linalg.inv(info_matrix(W, XX, p) + ridge * np.eye(p))
    return (Hinv.reshape(len(W), p * p) @ XX.T) * W


@dataclass
class IrlsResult:
    beta: np.ndarray  # genes x p (natural log)
    mu: np.ndarray  # genes x n, not floored
    hat: np.ndarray  # genes x n, from the floored means
    converged: np.ndarray  # genes; False only where the fallback optimiser failed
    fallback: np.ndarray  # genes that left Fisher scoring for L-BFGS-B


def irls(Y: np.ndarray, offset: np.ndarray, X: np.ndarray, alpha: np.ndarray, *,
         min_mu: float = MIN_MU, beta_tol: float = BETA_TOL, maxiter: int = MAX_ITER,
         max_beta: float = MAX_BETA, ridge: float = RIDGE) -> IrlsResult:
    """Negative binomial GLM fit per gene at fixed dispersions ``alpha``."""
    Y = np.asarray(Y, dtype=np.float64)
    G, n = Y.shape
    p = X.shape[1]
    off = _offsets(offset, G, n)
    alpha = np.asarray(alpha, dtype=np.float64)
    XX = outer_design(X)
    R = ridge * np.eye(p)
    Q, Rq = np.linalg.qr(X)
    start = np.linalg.solve(Rq, Q.T @ np.log(Y / np.exp(off) + 0.1).T).T  # G x p
    beta = start.copy()
    mu = np.maximum(np.exp(off + beta @ X.T), min_mu)
    dev = np.full(G, 1000.0)
    fallback = np.zeros(G, dtype=bool)
    active = np.arange(G)
    it = 0
    while active.size:
        it += 1
        y, m, d, o = Y[active], mu[active], alpha[active, None], off[active]
        W = m / (1 + m * d)
        z = np.log(m) - o + (y - m) / m
        H = info_matrix(W, XX, p) + R
        bh = np.linalg.solve(H, ((W * z) @ X)[..., None])[..., 0]
        bad = (np.abs(bh) > max_beta).any(axis=1) | (it >= maxiter)
        fallback[active[bad]] = True
        act, bh = active[~bad], bh[~bad]
        if not act.size:
            break
        beta[act] = bh
        mu_new = np.maximum(np.exp(off[act] + bh @ X.T), min_mu)
        mu[act] = mu_new
        new = -2 * nb_nll(Y[act], mu_new, alpha[act])
        ratio = np.abs(new - dev[act]) / (np.abs(new) + 0.1)
        dev[act] = new
        active = act[ratio > beta_tol]
    converged = np.ones(G, dtype=bool)
    for g in np.flatnonzero(fallback):
        beta[g], converged[g] = _fallback_fit(Y[g], off[g], X, alpha[g], start[g], min_mu,
                                              max_beta, ridge)
        mu[g] = np.maximum(np.exp(off[g] + X @ beta[g]), min_mu)
    W = mu / (1 + mu * alpha[:, None])
    hat = hat_diagonals(W, XX, p, ridge)
    return IrlsResult(beta, np.exp(off + beta @ X.T), hat, converged, fallback)


def _fallback_fit(y, off, X, alpha, start, min_mu, max_beta, ridge):
    """One gene by L-BFGS-B on the ridge-penalised likelihood (pydeseq2's
    fallback), then a 2-D grid when that fails and p <= 2."""
    p = X.shape[1]
    r = 1 / alpha

    def f(b):
        m = np.maximum(np.exp(off + X @ b), min_mu)
        return float(nb_nll(y[None], m[None], np.array([alpha]))[0] + 0.5 * ridge * (b ** 2).sum())

    def df(b):
        m = np.maximum(np.exp(off + X @ b), min_mu)
        return -X.T @ y + ((r + y) * m / (r + m)) @ X + ridge * b

    res = optimize.minimize(f, start, jac=df, method="L-BFGS-B",
                            bounds=[(-max_beta, max_beta)] * p)
    if res.success or p > 2:
        return res.x, bool(res.success)
    return _grid_beta(y, off, X, alpha, min_mu, max_beta, ridge), False


def _grid_beta(y, off, X, alpha, min_mu, max_beta, ridge, n_grid: int = 60):
    """pydeseq2's ``grid_fit_beta``: coarse then fine 60 x 60 grid (p <= 2)."""
    p = X.shape[1]

    def loss(B):  # K x p
        m = np.maximum(np.exp(off[None] + B @ X.T), min_mu)
        a = np.full(len(B), alpha)
        return nb_nll(np.broadcast_to(y, m.shape), m, a) + 0.5 * ridge * (B ** 2).sum(axis=1)

    axes = [np.linspace(-max_beta, max_beta, n_grid)] * p
    delta = axes[0][1] - axes[0][0]
    for _ in range(2):
        grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, p)
        best = grid[np.argmin(loss(grid))]
        axes = [np.linspace(b - delta, b + delta, n_grid) for b in best]
    return best


def trigamma(x: np.ndarray) -> np.ndarray:
    """psi'(x) for x > 0: recurrence up to 10, then the asymptotic series
    (relative error < 1e-11; scipy's polygamma(1, x) is ~20x slower). Only
    Newton's curvature uses it, so the maxima do not depend on its last digits."""
    x = np.asarray(x, dtype=np.float64)
    acc = np.zeros_like(x)
    small = x < 10
    if small.any():
        xs = x[small].copy()
        add = np.zeros_like(xs)
        for _ in range(10):
            m = xs < 10
            if not m.any():
                break
            add[m] += 1 / xs[m] ** 2
            xs[m] += 1
        x = x.copy()
        x[small] = xs
        acc[small] = add
    z = 1 / (x * x)
    series = (1 / x) * (1 + 0.5 / x + z * (1 / 6 + z * (-1 / 30 + z * (1 / 42 + z * (-1 / 30
                                                                                + z * 5 / 66)))))
    return acc + series


def nb_loglik_derivs(Y: np.ndarray, mu: np.ndarray, alpha: np.ndarray
                     ) -> tuple[np.ndarray, np.ndarray]:
    """First and second derivatives of the NB log-likelihood in log(alpha), mu fixed."""
    a = np.asarray(alpha, dtype=np.float64)[:, None]
    r = 1 / a
    am = a * mu
    opa = 1 + am
    lr = (special.digamma(Y + r) - special.digamma(r) - np.log1p(am)
          + a * (mu - Y) / opa).sum(axis=1)
    lrr = (trigamma(Y + r) - trigamma(r)
           + a * a * mu / opa + a * a * (Y - mu) / opa ** 2).sum(axis=1)
    rr = r[:, 0]
    return -rr * lr, rr * rr * lrr + rr * lr


def cox_reid(mu: np.ndarray, alpha: np.ndarray, XX: np.ndarray, p: int) -> np.ndarray:
    """The Cox-Reid adjustment -1/2 log det(X'WX), W = mu / (1 + alpha mu)."""
    W = mu / (1 + np.asarray(alpha)[:, None] * mu)
    return -0.5 * np.linalg.slogdet(info_matrix(W, XX, p))[1]


def cox_reid_derivs(mu: np.ndarray, alpha: np.ndarray, XX: np.ndarray, p: int
                    ) -> tuple[np.ndarray, np.ndarray]:
    """First and second derivatives of ``cox_reid`` in log(alpha):
    d log det M = tr(M^-1 M'), d2 = tr(M^-1 M'') - tr((M^-1 M')^2)."""
    a = np.asarray(alpha, dtype=np.float64)[:, None]
    W = mu / (1 + a * mu)
    W1 = -a * W * W  # dW / dlog(alpha)
    W2 = W1 + 2 * a * a * W ** 3
    Minv = np.linalg.inv(info_matrix(W, XX, p))
    h = Minv.reshape(len(W), p * p) @ XX.T  # x_i' M^-1 x_i
    A = Minv @ info_matrix(W1, XX, p)
    d1 = (W1 * h).sum(axis=1)
    d2 = (W2 * h).sum(axis=1) - np.einsum("gij,gji->g", A, A)
    return -0.5 * d1, -0.5 * d2


def maximise_1d(derivs: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]],
                x0: np.ndarray, lo: np.ndarray, hi: np.ndarray, *, tol: float = 1e-8,
                step: float = 0.5, maxiter: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Per gene, the first maximum of a smooth function on [lo, hi] uphill
    from ``x0``, from derivatives only: ``derivs(idx, x)`` returns the first
    and second derivatives for genes ``idx`` at ``x``. Uphill steps of
    ``step``, doubling, until the slope turns (a bracket) or a bound is
    reached (the answer); inside a bracket Newton steps, bisection when a step
    would leave it or not shrink it fast enough (rtsafe). Returns (x, ok): ok
    False for genes that did not converge or had non-finite derivatives."""
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    x = np.clip(np.asarray(x0, dtype=np.float64), lo, hi)
    G = len(x)
    g, h = derivs(np.arange(G), x)
    ok = np.isfinite(g)
    conv = ~ok | (g == 0)
    sign = np.sign(np.where(ok, g, 0.0))
    bound = np.where(sign > 0, hi, lo)
    at = ~conv & (x == bound)
    conv |= at
    L, R = x.copy(), x.copy()
    # Expanding uphill search for a bracket [L, R] with g(L) > 0 > g(R).
    act = np.flatnonzero(~conv)
    last, glast, hlast = x[act], g[act], h[act]
    s = np.full(act.size, step)
    for _ in range(64):
        if not act.size:
            break
        d = sign[act]
        trial = np.clip(last + d * s, lo[act], hi[act])
        gt, ht = derivs(act, trial)
        bad = ~np.isfinite(gt)
        turned = ~bad & (gt * d < 0)
        edge = ~bad & ~turned & (trial == bound[act])
        ok[act[bad]] = False
        conv[act[bad]] = True
        x[act[edge]] = trial[edge]
        conv[act[edge]] = True
        b = act[turned]
        up = d[turned] > 0
        L[b] = np.where(up, last[turned], trial[turned])
        R[b] = np.where(up, trial[turned], last[turned])
        # Newton starts from the turned point's uphill side.
        x[b], g[b], h[b] = last[turned], glast[turned], hlast[turned]
        more = ~(bad | turned | edge)
        act, last, glast, hlast, s = act[more], trial[more], gt[more], ht[more], s[more] * 2
    act = np.flatnonzero(~conv)
    gx, hx = g[act], h[act]
    dxold = R[act] - L[act]
    for _ in range(maxiter):
        if not act.size:
            break
        xa, La, Ra = x[act], L[act], R[act]
        with np.errstate(divide="ignore", invalid="ignore"):
            xn = xa - gx / hx
        newton = ((hx < 0) & (xn > La) & (xn < Ra) & np.isfinite(xn)
                  & (np.abs(2 * gx) <= np.abs(dxold * hx)))
        xn = np.where(newton, xn, 0.5 * (La + Ra))
        dx = xn - xa
        x[act] = xn
        fin = (np.abs(dx) < tol) | (Ra - La < tol)
        conv[act[fin]] = True
        keep = ~fin
        act, xn, dx = act[keep], xn[keep], dx[keep]
        if not act.size:
            break
        gx, hx = derivs(act, xn)
        bad = ~np.isfinite(gx)
        ok[act[bad]] = False
        up = gx > 0
        L[act[up]] = xn[up]
        R[act[~up & ~bad]] = xn[~up & ~bad]
        zero = gx == 0
        conv[act[zero | bad]] = True
        keep = ~(zero | bad)
        act, gx, hx, dxold = act[keep], gx[keep], hx[keep], dx[keep]
    return x, ok & conv


def wald(beta: np.ndarray, mu: np.ndarray, alpha: np.ndarray, X: np.ndarray,
         contrast: np.ndarray, ridge: float = RIDGE) -> tuple[np.ndarray, np.ndarray]:
    """Wald standard error and statistic of ``contrast @ beta`` per gene, with
    the sandwich form pydeseq2 uses: se^2 = c' H M H c, H = (M + ridge)^-1."""
    p = X.shape[1]
    W = mu / (1 + mu * np.asarray(alpha)[:, None])
    M = info_matrix(W, outer_design(X), p)
    Hc = np.linalg.inv(M + ridge * np.eye(p)) @ contrast  # G x p
    se = np.sqrt(np.einsum("gi,gij,gj->g", Hc, M, Hc))
    return se, (beta @ contrast) / se
