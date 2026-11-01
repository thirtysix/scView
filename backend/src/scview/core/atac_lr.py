"""Logistic regression with a depth covariate for differential peaks (Seurat's LRDETest).

Seurat 5.6.0 ``LRDETest``, what Signac's vignette runs as ``FindMarkers(test.use = "LR",
latent.vars = "nCount_peaks")``: for each peak, ``glm(group ~ GENE + nCount_peaks, binomial)``
against ``glm(group ~ nCount_peaks, binomial)``, a likelihood ratio chi-square on 1 df. GENE is
the data slot, ``log1p(TF x 10,000 x IDF)``; the latent is the raw total, neither logged nor
scaled. Here it is centred and scaled for the numbers' sake: the same model space, the same
likelihoods, the same p.

Every peak of a group shares one null fit and x is sparse, so the score test of all of them is one
pass of sparse products against fixed vectors (``score_products``). The true likelihood ratio is then
a vectorised IRLS (``lrt_fit``) in blocks.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.special import expit, ndtr

LR_MAX_ITER = 25  # glm.control(maxit)
LR_TOL = 1e-8  # glm.control(epsilon): |dev - dev_old| / (|dev| + 0.1)
LR_SEPARATED_ETA = 25.0  # a fitted probability within 1e-11 of 0 or 1: the estimate is on its way to infinity

FIT_OK = "ok"
FIT_SEPARATED = "separated"
FIT_NOT_CONVERGED = "not_converged"


def deviance_and_mu(eta, y0):
    """The Bernoulli deviance per column and the fitted probabilities: with e = exp(-eta), each
    cell adds 2 log(1 + e), and 2 eta more when y = 0 (``y0``). One exp and one log per cell, as
    SIMD ufuncs; a term below 1e-16 is lost from a sum of thousands, which glm's test never sees."""
    with np.errstate(over="ignore", invalid="ignore"):
        e = np.exp(-eta)
        dev = 2.0 * (np.log(1.0 + e).sum(axis=0) + y0 @ eta)
        return dev, 1.0 / (1.0 + e)


def logistic_null(y, z, max_iter=LR_MAX_ITER, tol=LR_TOL):
    """IRLS for ``y ~ 1 + z``, the null model of every peak in a group: (b0, b1), the deviance,
    and whether it converged to finite estimates (a covariate that alone separates the group
    leaves nothing for a peak to add)."""
    b = np.array([np.log((y.sum() + 0.5) / (y.size - y.sum() + 0.5)), 0.0])
    design = np.column_stack([np.ones_like(z), z])
    y0 = 1.0 - y
    dev, mu = deviance_and_mu(design @ b, y0)
    for _ in range(max_iter):
        w = mu * (1.0 - mu)
        try:
            step = np.linalg.solve((design * w[:, None]).T @ design, design.T @ (y - mu))
        except np.linalg.LinAlgError:
            return b, float(dev), False
        b = b + step
        new, mu = deviance_and_mu(design @ b, y0)
        done = abs(new - dev) / (abs(new) + 0.1) < tol
        dev = new
        if done:
            break
    else:
        return b, float(dev), False
    return b, float(dev), bool(np.abs(design @ b).max() < LR_SEPARATED_ETA)


def null_vectors(y, z, b):
    """What the score test multiplies each peak by, per cell: (y - mu), w, w z; and the inverse
    of the null's 2 x 2 information (for its Schur complement)."""
    mu = expit(b[0] + b[1] * z)
    w = mu * (1.0 - mu)
    s00, s01, s11 = w.sum(), (w * z).sum(), (w * z * z).sum()
    det = s00 * s11 - s01 * s01
    inverse = np.array([s11 / det, -s01 / det, s00 / det]) if det > 0 else np.array([np.nan, np.nan, np.nan])
    return y - mu, w, w * z, inverse


def score_products(values, residual, w, wz):
    """One block of cells x peaks (``values`` CSR, cells in the order of the vectors' rows; the
    vectors' columns are the groups): the sparse products a score test needs, peaks x groups."""
    t = values.T  # CSC: a sparse-dense product costs one operation per value per column
    squares = sp.csc_matrix((t.data * t.data, t.indices, t.indptr), shape=t.shape)
    return np.asarray(t @ residual), np.asarray(squares @ w), np.asarray(t @ w), np.asarray(t @ wz)


def score_statistic(u, a, b, c, inverse):
    """(score chi-square, the variance): U^2 / (x'Wx - x'WZ (Z'WZ)^-1 Z'Wx), NaN where the peak
    has no variance left once the null's covariates are out."""
    v = a - (b * b * inverse[0] + 2.0 * b * c * inverse[1] + c * c * inverse[2])
    ok = v > 1e-10 * np.maximum(a, 1e-300)
    return np.divide(u * u, v, out=np.full_like(u, np.nan), where=ok), v


def signed_root_p(stat, sign):
    """A 1-df chi-square as a signed z and its p; NaN stays NaN."""
    z = np.sqrt(np.maximum(stat, 0.0))
    return z * np.where(sign < 0, -1.0, 1.0), 2.0 * ndtr(-z)


def lrt_fit(x, y, z, null_b, null_dev, max_iter=LR_MAX_ITER, tol=LR_TOL):
    """IRLS for ``y ~ 1 + z + x`` on each column of ``x`` (cells x peaks, dense), started from
    the null fit. Returns (LR statistic, gamma, status) per peak. A peak leaves the block when it
    has converged, as glm's test decides. Deviance that rises halves the step (glm does so only when
    it goes infinite). Status ``separated``: the fitted probabilities reached 0 or 1, so gamma is on
    its way to infinity and the statistic is the deviance's limit, the number glm stops near."""
    n_peaks = x.shape[1]
    stat = np.full(n_peaks, np.nan)
    gamma = np.zeros(n_peaks)
    status = np.full(n_peaks, FIT_NOT_CONVERGED, dtype=object)
    active = np.arange(n_peaks)
    cur = x
    b = np.tile(np.array([null_b[0], null_b[1], 0.0]), (n_peaks, 1))  # rows: intercept, z, x
    y0 = 1.0 - y
    z2 = z * z
    eta = b[:, 0] + z[:, None] * b[:, 1]
    dev, mu = deviance_and_mu(eta, y0)
    for _ in range(max_iter):
        if active.size == 0:
            break
        w = mu * (1.0 - mu)
        r = y[:, None] - mu
        wx = w * cur
        s00, s01, s11 = w.sum(axis=0), z @ w, z2 @ w
        s02, s12, s22 = wx.sum(axis=0), z @ wx, np.einsum("ij,ij->j", wx, cur)
        g0, g1, g2 = r.sum(axis=0), z @ r, np.einsum("ij,ij->j", r, cur)
        del wx, r
        # The 3 x 3 by cofactors, per peak; a det that is not positive cannot be stepped from.
        c00, c01, c02 = s11 * s22 - s12 * s12, s02 * s12 - s01 * s22, s01 * s12 - s02 * s11
        c11, c12, c22 = s00 * s22 - s02 * s02, s01 * s02 - s00 * s12, s00 * s11 - s01 * s01
        det = s00 * c00 + s01 * c01 + s02 * c02
        good = np.isfinite(det) & (det > 0)
        safe = np.where(good, det, 1.0)
        step = np.column_stack([(c00 * g0 + c01 * g1 + c02 * g2) / safe, (c01 * g0 + c11 * g1 + c12 * g2) / safe,
                                (c02 * g0 + c12 * g1 + c22 * g2) / safe])
        step[~good] = 0.0
        new_b = b + step
        eta = new_b[:, 0] + z[:, None] * new_b[:, 1] + cur * new_b[:, 2]
        new_dev, mu = deviance_and_mu(eta, y0)
        rising = np.flatnonzero(~(new_dev <= dev + 1e-9 * (np.abs(dev) + 0.1)) & good)
        for _halve in range(8):
            if rising.size == 0:
                break
            step[rising] /= 2.0
            new_b[rising] = b[rising] + step[rising]
            eta[:, rising] = new_b[rising, 0] + z[:, None] * new_b[rising, 1] + cur[:, rising] * new_b[rising, 2]
            new_dev[rising], mu[:, rising] = deviance_and_mu(eta[:, rising], y0)
            rising = rising[~(new_dev[rising] <= dev[rising] + 1e-9 * (np.abs(dev[rising]) + 0.1))]
        if rising.size:  # no step helps: this is where the numbers stop, and the test below ends it
            new_b[rising] = b[rising]
            eta[:, rising] = b[rising, 0] + z[:, None] * b[rising, 1] + cur[:, rising] * b[rising, 2]
            new_dev[rising], mu[:, rising] = deviance_and_mu(eta[:, rising], y0)
        done = (np.abs(new_dev - dev) / (np.abs(new_dev) + 0.1) < tol) & good
        b, dev = new_b, new_dev
        leaving = done | ~good
        if leaving.any():
            ids = active[leaving]
            extreme = np.abs(eta[:, leaving]).max(axis=0) >= LR_SEPARATED_ETA
            stat[ids] = null_dev - dev[leaving]
            gamma[ids] = b[leaving, 2]
            status[ids] = np.where(~good[leaving], FIT_NOT_CONVERGED, np.where(extreme, FIT_SEPARATED, FIT_OK))
            keep = ~leaving
            active, cur, b, dev, eta, mu = active[keep], cur[:, keep], b[keep], dev[keep], eta[:, keep], mu[:, keep]
    if active.size:  # the iterations ran out
        extreme = np.abs(eta).max(axis=0) >= LR_SEPARATED_ETA
        stat[active] = null_dev - dev
        gamma[active] = b[:, 2]
        status[active] = np.where(extreme, FIT_SEPARATED, FIT_NOT_CONVERGED)
    return np.maximum(stat, 0.0), gamma, status


REPLAY_CONSTANTS = ("LR_MAX_ITER", "LR_TOL", "LR_SEPARATED_ETA", "FIT_OK", "FIT_SEPARATED", "FIT_NOT_CONVERGED")
REPLAY_CODE = (deviance_and_mu, logistic_null, null_vectors, score_products, score_statistic, signed_root_p, lrt_fit)
