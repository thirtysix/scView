"""Spot composition: each Visium spot's cell-type shares from a single-cell reference (C047).

The reference is ANOTHER library dataset with a cell-type column. Written from the
equations of RCTD (Cable et al. 2022, Nat Biotech 40:517, "full" mode; clean room:
spacexr is GPL-3, so no code is taken from it or its ports):

- Profiles: ``mu_kj`` = the mean over type k's cells of ``X_cj / n_c`` (``n_c`` the
  cell's counts over all its genes; raw counts only), ``S_k`` the mean ``n_c`` (cell
  size, for cell shares). Types with fewer than ``MIN_TYPE_CELLS`` cells are left out.
- Genes: shared by name (``label_transfer.match_genes``), not mitochondrial,
  ribosomal, Y-linked or XIST (``deconvolution._excluded``), with spot counts. The
  platform set: genes with ``mu >= PLATFORM_MIN_MU`` and ``ln(mu / mean over types)
  >= PLATFORM_MIN_LFC`` in some type; the fit set the same at ``FIT_MIN_MU`` /
  ``FIT_MIN_LFC`` (the paper's "about 3,000" genes; the thresholds are ours).
- Platform effect, per section (``library_id``): the section's pseudo-bulk ``s_j =
  sum_i Y_ij / sum_i N_i`` is fitted as ``mu b`` on the log scale (NNLS, then
  ``GN_ROUNDS`` relative-error NNLS rounds), ``g_j = ln(s_j / (mu b)_j)``; with the
  Poisson variance ``v_j = 1 / S_j`` the gene effect's spread is ``s_g^2 = max(var g -
  mean v, SG2_FLOOR)`` and ``gamma_j = g_j s_g^2 / (s_g^2 + v_j)`` (shrunk, centred).
  The profiles become ``mu_kj e^gamma_j``: one effect shared by every type (the
  paper's stated limitation).
- Per spot: ``Y_ij | eps ~ Poisson(m_ij e^eps)``, ``m_ij = N_i sum_k w_ik mu'_kj``,
  ``eps ~ N(0, s_e^2)``, ``w >= 0``; the RNA shares are ``w / sum w``. The likelihood
  of a count given ``t = ln m`` is tabulated (Gauss-Hermite, ``GH_NODES`` nodes, ``t``
  on a ``T_STEP`` grid) with its first two derivatives (exact: the posterior mean and
  variance of the rate). Newton steps with an NNLS box step on the PSD-floored
  Hessian (the Poisson information where it has no curvature), halved until the
  likelihood rises. ``s_e`` is chosen on ``SE_SPOTS`` seeded spots from ``SE_GRID``
  (their fits on a ``SEARCH_T_STEP`` table, scored exactly). Counts above ``Y_MAX`` are
  clipped (counted in the notes).
  The paper's heavy-tailed (Cauchy-Gaussian) noise term is NOT in this version.
- ``nnls``: ``w = nnls(mu'^T, Y_i / N_i)`` per spot on the fit genes, the baseline
  the benchmarks keep (Sang-aram 2024, Spotless) and the Newton fit's start.

Spots with fewer than ``MIN_SPOT_UMI`` counts are not fitted (NaN, "low counts").
Cell shares are ``beta_k / S_k`` renormalised: they assume the reference's cell sizes
hold in the tissue. The self-check (``iter_check``) draws synthetic spots from a
held-out half of the reference (Spotless "synthspot": 2-10 cells, Dirichlet(0.5)
types, thinned to a Visium-like depth) and scores both methods.
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
import uuid
import weakref
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import nnls
from scipy.special import gammaln

from scview.core import results_store

logger = logging.getLogger(__name__)

KIND = "spot_deconvolution"
CHECK_KIND = "spot_deconvolution_check"
TAG = "result:spotdeconv"  # the co-pilot's fact (claims.py family "spotdeconv")
STEP = "spot_deconvolution"
UNS_KEY = "scview_spot_deconv"
OBSM_RNA = "spot_deconv"
OBSM_CELLS = "spot_deconv_cells"
OBS_TOP = "spot_deconv_top"
OBS_TOP_SHARE = "spot_deconv_top_share"
LOW_COUNTS = "low counts"
METHODS = ("rctd", "nnls")
METHOD_NAMES = {"rctd": "RCTD-style Poisson-lognormal fit", "nnls": "NNLS on the fit genes"}

MIN_TYPE_CELLS = 25
MAX_TYPES = 50
MIN_REFERENCE_CELLS = 50
MIN_SPOT_UMI = 100
MIN_SHARED_GENES = 500  # shared by name before any selection: below this the check warns
# Tuned on V1_Human_Lymph_Node x the cell2location lymphoid atlas toward the paper's "about
# 3,000" fit genes: RCTD's own 1.25e-4 / 2e-4 kept 2,952 / 1,595 of its 9,941 usable genes.
PLATFORM_MIN_MU = 6.25e-5
PLATFORM_MIN_LFC = 0.5
FIT_MIN_MU = 1e-4
FIT_MIN_LFC = 0.75
MIN_FIT_GENES = 200  # fewer: the platform set is fitted
MIN_GENES = 50  # fewer: refused
GN_ROUNDS = 20
SG2_FLOOR = 1e-4
GH_NODES = 80
T_MIN, T_MAX, T_STEP = -12.0, 11.0, 0.01
SEARCH_T_STEP = 0.05  # the noise-level search's coarser table (5x quicker; it only ranks s_e values)
Y_MAX = 1000
SE_GRID = tuple(round(0.1 * i, 1) for i in range(1, 16))
SE_START = 0.8
SE_SPOTS = 200
SE_ROUNDS = 4
MAX_ITER = 50
TOL = 1e-3
BLOCK = 256  # spots per fit block (one progress event each)
HESS_BLOCK = 32  # spots per Hessian product (memory: HESS_BLOCK x types x genes)
EIG_FLOOR = 1e-8
TABLE_ROWS = 16  # counts per table chunk
SEED = 0

# The self-check.
CHECK_SPOTS = 1000
CHECK_CELLS = (2, 10)
CHECK_DEPTHS = (20000, 5000)
CHECK_ALPHA = 0.5

# Estimates at 2 CPUs (calibrated on V1_Human_Lymph_Node x the lymphoid atlas, see estimate_seconds).
SECONDS_FIXED = 8.0
SECONDS_PER_REF_CELL = 2.5e-4
SECONDS_PER_SPOT_TYPE2 = 1.2e-5
NEED_FIXED = 400 * 1024 * 1024
NEED_COUNTS_COPIES = 2.5
# A large side (core/large_mode.py) is never read whole: one pass over its counts by row
# blocks (the reference's self-check a second), each block (MAX_READ_BYTES / 32 values,
# float32 + int64 index) held about LARGE_BLOCK_COPIES times; the reference's profiles
# (types x all its genes, float64, twice); the self-check's drawn cells' rows. Seconds
# per stored value of the reading (gzip counts, 2 CPUs; measured on the lymphoid atlas as
# a large dataset, 93M values: 26 s to prepare the fit, 32 s the self-check's two passes).
LARGE_BLOCK_COPIES = 2.0
LARGE_SECONDS_PER_VALUE = 2e-7
LARGE_CHECK_FACTOR = 1.5

NOTE = ("Shares are fractions of each spot's or bin's RNA (or, estimated, of its cells), not counts. A cell type "
        "missing from the reference is given to its nearest type, so its spots look like that type. The "
        "platform correction assumes one effect shared by every type, so single-nucleus references and "
        "probe-based (FFPE) sections differ most.")

REPLAY_CONSTANTS = ("MIN_SPOT_UMI", "PLATFORM_MIN_MU", "PLATFORM_MIN_LFC", "FIT_MIN_MU", "FIT_MIN_LFC",
                    "MIN_FIT_GENES", "MIN_GENES", "GN_ROUNDS", "SG2_FLOOR", "GH_NODES", "T_MIN", "T_MAX",
                    "T_STEP", "Y_MAX", "SE_GRID", "SE_START", "SE_SPOTS", "SE_ROUNDS", "MAX_ITER", "TOL",
                    "BLOCK", "HESS_BLOCK", "EIG_FLOOR", "TABLE_ROWS", "SEARCH_T_STEP")


class SpotDeconvolutionError(RuntimeError):
    """A request this cannot serve; ``status`` is the HTTP code (422 by default)."""

    def __init__(self, message: str, status: int = 422) -> None:
        super().__init__(message)
        self.status = status


class SpotDeconvolutionCancelled(RuntimeError):
    """The client went away: the run stops after the block it is on, nothing is saved."""


# ---------------------------------------------------------------------------
# The model (self-contained: the replay script carries these functions' source)
# ---------------------------------------------------------------------------


def sd_profiles(M, totals, codes, n_types):
    """Per type: mu (types x genes, the mean of counts / cell total over its cells),
    S (the mean cell total) and its cell count. ``codes``: type per cell, -1 none."""
    ok = (codes >= 0) & (totals > 0)
    cols = np.flatnonzero(ok)
    n = np.bincount(codes[ok], minlength=n_types).astype(np.float64)
    W = sp.csr_matrix((1.0 / (totals[ok] * n[codes[ok]]), (codes[ok], cols)), shape=(n_types, M.shape[0]))
    mu = W @ M
    mu = np.asarray(mu.toarray() if sp.issparse(mu) else mu, dtype=np.float64)
    S = np.bincount(codes[ok], weights=totals[ok], minlength=n_types) / np.maximum(n, 1)
    return mu, S, n.astype(np.int64)


def sd_select_genes(mu, keep):
    """(platform set, fit set): positions of the genes in ``keep`` with mu >= the floor
    and ln(mu / the types' mean) >= the cut in at least one type."""
    mean = mu.mean(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        lfc = np.log(mu / mean[None, :])
    lfc = np.nan_to_num(lfc, nan=-np.inf)
    plat = keep & ((mu >= PLATFORM_MIN_MU) & (lfc >= PLATFORM_MIN_LFC)).any(axis=0)
    fit = keep & ((mu >= FIT_MIN_MU) & (lfc >= FIT_MIN_LFC)).any(axis=0)
    return np.flatnonzero(plat), np.flatnonzero(fit)


def sd_platform_effect(mu, ysum, ntot):
    """One section's gene effect (gamma, per gene of ``mu``: types x genes), its bulk
    weights b and s_g, from the section's summed counts ``ysum`` and summed totals
    ``ntot``. Genes without counts get gamma 0. (zeros, None, None) when it cannot be fitted."""
    gamma = np.zeros(mu.shape[1])
    ok = ysum > 0
    if ntot <= 0 or int(ok.sum()) < mu.shape[0] + 2:
        return gamma, None, None
    A = mu[:, ok].T
    s = ysum[ok] / ntot
    b = nnls(A, s, maxiter=50 * A.shape[1])[0]
    for _ in range(GN_ROUNDS):
        m = np.maximum(A @ b, 0.01 * s)  # a gene the fit leaves near 0 weighs at most 100x its own
        b = nnls(A / m[:, None], s / m, maxiter=50 * A.shape[1])[0]
    m = A @ b
    pos = m > 0
    if int(pos.sum()) < 2:
        return gamma, None, None
    g = np.log(s[pos] / m[pos])
    v = 1.0 / ysum[ok][pos]
    sg2 = max(float(g.var() - v.mean()), SG2_FLOOR)
    gam = g * sg2 / (sg2 + v)
    gamma[np.flatnonzero(ok)[pos]] = gam - gam.mean()
    return gamma, b, math.sqrt(sg2)


def sd_gh(s_e):
    """Gauss-Hermite nodes on the noise (eps) and their log weights, for N(0, s_e^2)."""
    x, w = np.polynomial.hermite.hermgauss(GH_NODES)
    return np.sqrt(2.0) * s_e * x, np.log(w / np.sqrt(np.pi))


def sd_t_grid(step=T_STEP):
    return T_MIN + step * np.arange(int(round((T_MAX - T_MIN) / step)) + 1)


def sd_table(s_e, yvals, step=T_STEP):
    """For each count in ``yvals`` (sorted, unique) and each t on the grid: l(y, t) =
    ln E_eps Pois(y; e^(t + eps)), and its first and second derivatives in t (the
    posterior rate's mean and variance: l_t = y - E[lam], l_tt = Var[lam] - E[lam]).
    Shape (3, counts, grid)."""
    t = sd_t_grid(step)
    eps, logw = sd_gh(s_e)
    te = t[:, None] + eps[None, :]
    lam = np.exp(te)
    lam2 = lam * lam
    out = np.empty((3, len(yvals), t.size))
    for start in range(0, len(yvals), TABLE_ROWS):
        y = np.asarray(yvals[start:start + TABLE_ROWS], dtype=np.float64)
        lp = y[:, None, None] * te[None]
        lp -= lam[None]
        lp += logw[None, None, :]
        mx = lp.max(axis=2, keepdims=True)
        lp -= mx
        np.exp(lp, out=lp)  # the posterior weights, unnormalised
        z = lp.sum(axis=2)
        el = np.einsum("ktn,tn->kt", lp, lam) / z
        el2 = np.einsum("ktn,tn->kt", lp, lam2) / z
        rows = slice(start, start + y.size)
        out[0, rows] = mx[..., 0] + np.log(z) - gammaln(y + 1)[:, None]
        out[1, rows] = y[:, None] - el
        out[2, rows] = np.maximum(el2 - el * el, 0.0) - el
    return out


def sd_loglik_direct(y, t, s_e, chunk=50000):
    """l(y, t) evaluated directly (no table), for flat arrays of counts and t."""
    eps, logw = sd_gh(s_e)
    out = np.empty(y.size)
    for start in range(0, y.size, chunk):
        yy = y[start:start + chunk].astype(np.float64)[:, None]
        te = t[start:start + chunk][:, None] + eps[None, :]
        lp = yy * te - np.exp(te) - gammaln(yy + 1) + logw[None, :]
        mx = lp.max(axis=1, keepdims=True)
        out[start:start + chunk] = (mx + np.log(np.exp(lp - mx).sum(axis=1, keepdims=True)))[:, 0]
    return out


def sd_lookup(tab, rows, t, which):
    """Table values (``which``: 0 = l, 1 = l_t, 2 = l_tt) at counts' table rows and t,
    linear in t; t outside the grid is clipped."""
    n_t = tab.shape[2]
    f = (np.clip(t, T_MIN, T_MAX) - T_MIN) / ((T_MAX - T_MIN) / (n_t - 1))
    i0 = np.minimum(f.astype(np.int64), n_t - 2)
    f = f - i0
    pos = rows * n_t + i0
    out = []
    for k in which:
        flat = tab[k].ravel()
        out.append(flat[pos] * (1.0 - f) + flat[pos + 1] * f)
    return out


def sd_fit_nnls(mu, Y, N):
    """Per spot: w = nnls(mu^T, y / N) on the genes of ``mu`` (types x genes)."""
    A = mu.T
    W = np.zeros((Y.shape[0], mu.shape[0]))
    for i in range(Y.shape[0]):
        y = np.asarray(Y[i], dtype=np.float64).ravel()
        W[i] = nnls(A, y / N[i], maxiter=50 * A.shape[1])[0]
    return W


def sd_spot_loglik(mu, Y, rows, N, W, tab):
    m = np.maximum(N[:, None] * (W @ mu), math.exp(T_MIN))
    (ll,) = sd_lookup(tab, rows, np.log(m), (0,))
    return ll.sum(axis=1)


def sd_fit_spots(mu, Y, rows, N, W0, tab):
    """Newton fits of w >= 0 per spot (rows of Y, dense counts on mu's genes) from W0.
    Returns (W, iterations per spot, converged per spot)."""
    B, K = W0.shape
    W = np.array(W0, dtype=np.float64)
    empty = W.sum(axis=1) <= 0
    if empty.any():
        scale = Y[empty].sum(axis=1) / N[empty] / max(float(mu.sum(axis=1).mean()), 1e-12)
        W[empty] = np.maximum(scale, 1e-6)[:, None] / K
    iters = np.zeros(B, dtype=np.int64)
    done = np.zeros(B, dtype=bool)
    ll = sd_spot_loglik(mu, Y, rows, N, W, tab)
    floor_m = math.exp(T_MIN)
    for it in range(MAX_ITER):
        act = np.flatnonzero(~done)
        if act.size == 0:
            break
        Wa, Na, Ya, Ra = W[act], N[act], Y[act], rows[act]
        m = np.maximum(Na[:, None] * (Wa @ mu), floor_m)
        lt, ltt = sd_lookup(tab, Ra, np.log(m), (1, 2))
        r = Na[:, None] / m
        g = (lt * r) @ mu.T
        c = (ltt - lt) * r * r
        Z = np.empty_like(Wa)
        for s0 in range(0, act.size, HESS_BLOCK):
            cb = c[s0:s0 + HESS_BLOCK]
            H = np.matmul(cb[:, None, :] * mu[None, :, :], mu.T)
            for j in range(cb.shape[0]):
                Q = -0.5 * (H[j] + H[j].T)
                vals, vecs = np.linalg.eigh(Q)
                if not vals.max() > 0:  # no curvature to use: the Poisson information instead
                    Q = (mu * (Na[s0 + j] ** 2 / m[s0 + j])[None, :]) @ mu.T
                    vals, vecs = np.linalg.eigh(Q)
                vals = np.maximum(vals, EIG_FLOOR * float(np.abs(vals).sum()) + 1e-300)
                w = Wa[s0 + j]
                rhs = (vecs.T @ (g[s0 + j] + vecs @ (vals * (vecs.T @ w)))) / np.sqrt(vals)
                with np.errstate(all="ignore"):
                    z = nnls(np.sqrt(vals)[:, None] * vecs.T, rhs, maxiter=50 * K)[0]
                Z[s0 + j] = z if np.isfinite(z).all() else w
        eta = np.ones(act.size)
        new = Wa + (Z - Wa)
        ll_new = sd_spot_loglik(mu, Ya, Ra, Na, new, tab)
        worse = ll_new < ll[act] - 1e-9
        for _ in range(12):
            if not worse.any():
                break
            eta[worse] /= 2
            idx = np.flatnonzero(worse)
            new[idx] = Wa[idx] + eta[idx, None] * (Z[idx] - Wa[idx])
            ll_new[idx] = sd_spot_loglik(mu, Ya[idx], Ra[idx], Na[idx], new[idx], tab)
            worse = ll_new < ll[act] - 1e-9
        stuck = worse
        new[stuck] = Wa[stuck]
        ll_new[stuck] = ll[act][stuck]
        tot_old = np.maximum(Wa.sum(axis=1, keepdims=True), 1e-300)
        tot_new = np.maximum(new.sum(axis=1, keepdims=True), 1e-300)
        moved = np.abs(new / tot_new - Wa / tot_old).max(axis=1)
        W[act] = new
        ll[act] = ll_new
        iters[act] = it + 1
        done[act[(moved < TOL) | stuck]] = True
    return W, iters, done


def sd_rows_for(Y):
    """The sorted unique counts of Y (clipped at Y_MAX, with 0) and each count's table row."""
    yvals = np.union1d(np.unique(np.minimum(np.asarray(Y), Y_MAX).astype(np.int64)), [0])
    rowmap = np.zeros(Y_MAX + 1, dtype=np.int64)
    rowmap[yvals] = np.arange(yvals.size)
    return yvals, rowmap


def sd_choose_sigma(mus, Y, N, W0, libs):
    """s_e for the spots given (dense counts Y, clipped; their sections ``libs``): fit
    them at s, score the fits at every value of SE_GRID, move to the best; at most
    SE_ROUNDS rounds. ``mus``: profiles per section. Fits and scores use SEARCH_T_STEP
    tables, built once per value."""
    s = SE_START
    yvals, rowmap = sd_rows_for(Y)
    rows = rowmap[Y]
    tables = {}

    def table(v):
        if v not in tables:
            tables[v] = sd_table(v, yvals, SEARCH_T_STEP)
        return tables[v]

    for _ in range(SE_ROUNDS):
        tab = table(s)
        fits = []
        for lib in np.unique(libs):
            idx = np.flatnonzero(libs == lib)
            W, _, _ = sd_fit_spots(mus[lib], Y[idx], rows[idx], N[idx], W0[idx], tab)
            m = np.maximum(N[idx, None] * (W @ mus[lib]), math.exp(T_MIN))
            fits.append((rows[idx], np.log(m)))
        scores = [sum(float(sd_lookup(table(v), r, t, (0,))[0].sum()) for r, t in fits) for v in SE_GRID]
        best = SE_GRID[int(np.argmax(scores))]
        if math.isclose(best, s):
            break
        s = best
    return s


def sd_iter_deconvolve(mu_all, S, Y, N, libs, method, keep, genes=None, s_e=None, seed=0):
    """The whole fit as events: ("progress", {...}) per stage and block, then ("fitted",
    {...}). ``mu_all``: types x genes profiles; ``Y``: spots x the same genes (CSR
    counts); ``N``: spot totals (all genes); ``libs``: section per spot; ``keep``: genes
    allowed; ``genes`` = (platform, fit) positions to use as recorded; ``s_e`` likewise."""
    notes = []
    Y = sp.csr_matrix(Y)
    if genes is None:
        has = np.asarray(Y.sum(axis=0)).ravel() > 0
        plat, fit = sd_select_genes(mu_all, keep & has & (mu_all.sum(axis=0) > 0))
        if fit.size < MIN_FIT_GENES:
            notes.append(f"Only {fit.size} genes passed the fit-set cut; the {plat.size} platform-set genes were fitted.")
            fit = plat
        if fit.size < MIN_GENES:
            raise ValueError(f"Only {fit.size} informative shared genes (at least {MIN_GENES} are needed): the "
                             "reference and the spots share too few genes that tell the types apart.")
    else:
        plat, fit = (np.asarray(g, dtype=np.int64) for g in genes)
    yield "progress", {"step": "genes", "n_platform": int(plat.size), "n_fit": int(fit.size)}
    in_plat = np.searchsorted(plat, fit)
    Yp = Y[:, plat].tocsr()
    fitted = N >= MIN_SPOT_UMI
    mus, s_g, bulk = {}, {}, {}
    for lib in np.unique(libs):
        rows_l = np.flatnonzero(libs == lib)
        ysum = np.asarray(Yp[rows_l].sum(axis=0)).ravel()
        gamma, b, sg = sd_platform_effect(mu_all[:, plat], ysum, float(N[rows_l].sum()))
        mus[lib] = (mu_all[:, plat] * np.exp(gamma)[None, :])[:, in_plat]
        s_g[lib] = sg
        bulk[lib] = (b / b.sum()).tolist() if b is not None and b.sum() > 0 else None
    yield "progress", {"step": "platform", "s_g": s_g}
    K = mu_all.shape[0]
    Yf = Yp[:, in_plat].tocsr()
    W = np.full((Y.shape[0], K), np.nan)
    idx_fit = np.flatnonzero(fitted)
    W0 = np.zeros((Y.shape[0], K))
    for lib in mus:
        sel = idx_fit[libs[idx_fit] == lib]
        for s0 in range(0, sel.size, BLOCK * 4):
            part = sel[s0:s0 + BLOCK * 4]
            W0[part] = sd_fit_nnls(mus[lib], Yf[part].toarray(), N[part])
            yield "progress", {"step": "baseline", "done": int(min(s0 + BLOCK * 4, sel.size)), "library": str(lib)}
    n_clipped = int((Yf.data > Y_MAX).sum())
    iters = np.zeros(Y.shape[0], dtype=np.int64)
    conv = np.ones(Y.shape[0], dtype=bool)
    if method == "nnls":
        W[idx_fit] = W0[idx_fit]
    else:
        if s_e is None:
            rng = np.random.default_rng(seed)
            picks = np.sort(rng.choice(idx_fit, size=min(SE_SPOTS, idx_fit.size), replace=False))
            if picks.size == 0:
                raise ValueError(f"No spot has {MIN_SPOT_UMI}+ counts: nothing to fit.")
            yield "progress", {"step": "sigma", "n_spots": int(picks.size)}
            Yd = np.minimum(Yf[picks].toarray(), Y_MAX).astype(np.int64)
            s_e = sd_choose_sigma(mus, Yd, N[picks], W0[picks], libs[picks])
        yield "progress", {"step": "table", "s_e": float(s_e)}
        yvals, rowmap = sd_rows_for(Yf.data if Yf.nnz else np.zeros(1))
        tab = sd_table(float(s_e), yvals)
        total, done_n = idx_fit.size, 0
        for lib in mus:
            sel = idx_fit[libs[idx_fit] == lib]
            for s0 in range(0, sel.size, BLOCK):
                part = sel[s0:s0 + BLOCK]
                Yc = np.minimum(Yf[part].toarray(), Y_MAX).astype(np.int64)
                Wb, it, ok = sd_fit_spots(mus[lib], Yc, rowmap[Yc], N[part], W0[part], tab)
                W[part], iters[part], conv[part] = Wb, it, ok
                done_n += part.size
                yield "progress", {"step": "fit", "done": int(done_n), "total": int(total)}
    tot = W.sum(axis=1, keepdims=True)
    fitted = fitted & (np.nan_to_num(tot[:, 0]) > 0)
    beta = np.where(tot > 0, W / np.where(tot > 0, tot, 1), np.nan)
    beta[~fitted] = np.nan
    cells = beta / S[None, :]
    cells = cells / cells.sum(axis=1, keepdims=True)
    if n_clipped:
        notes.append(f"{n_clipped:,} spot x gene counts above {Y_MAX} were clipped to {Y_MAX}.")
    yield "fitted", {"beta": beta, "cells": cells, "fitted": fitted, "s_e": None if s_e is None else float(s_e),
                     "s_g": s_g, "bulk": bulk, "platform": plat, "fit": fit, "iterations": iters,
                     "converged": conv, "n_clipped": n_clipped, "notes": notes}


REPLAY_CODE = (sd_profiles, sd_select_genes, sd_platform_effect, sd_gh, sd_t_grid, sd_table, sd_loglik_direct,
               sd_lookup, sd_fit_nnls, sd_spot_loglik, sd_fit_spots, sd_rows_for, sd_choose_sigma,
               sd_iter_deconvolve)


# ---------------------------------------------------------------------------
# Inputs: counts, labels, genes, species
# ---------------------------------------------------------------------------


def _strip_versions(var: pd.DataFrame) -> pd.DataFrame:
    """Ensembl ids without their version suffix ("ENSG...12.3" -> "ENSG...12")."""
    from scview.core.deconvolution import _ENSEMBL, _VERSION

    names = pd.Series(var.index.astype(str))
    if names.str.match(_ENSEMBL).mean() > 0.5:
        return var.set_axis(pd.Index(names.str.replace(_VERSION, "", regex=True)))
    return var


def counts_with_var(adata, fallback_paths: list[str], who: str) -> tuple[Any, pd.DataFrame, str]:
    """``adata``'s raw counts (cells x genes), their var and where they came from: the
    dataset's own (``pseudobulk.raw_counts``), else its source files
    (``integration.recover_counts``). Log values are refused."""
    from scview.core.integration import IntegrationError, recover_counts
    from scview.core.pseudobulk import PseudobulkError, raw_counts

    try:
        M, names, source = raw_counts(adata)
        var = adata.raw.var if source == ".raw" and adata.raw is not None else adata.var
        var = var.copy()
        var.index = pd.Index(np.asarray(names).astype(str))
        return M, var, source
    except PseudobulkError:
        pass
    try:
        rc = recover_counts(adata, fallback_paths)
        return rc.X, rc.var.copy(), rc.source
    except IntegrationError:
        pass
    raise SpotDeconvolutionError(
        f"{who} has no raw counts (layers['counts'], .raw, X or its source files): spot composition compares "
        "counts, never log values. Re-load it from a file that keeps its counts.")


def _first_positions(index: pd.Index, names: list[str]) -> np.ndarray:
    first: dict[str, int] = {}
    for i, g in enumerate(index.astype(str)):
        first.setdefault(g, i)
    return np.asarray([first[g] for g in names], dtype=np.int64)


def type_counts(labels: pd.Series) -> tuple[list[str], list[str], pd.Series]:
    """(types kept, types left out with fewer than MIN_TYPE_CELLS cells, cells per label)."""
    from scview.core.deconvolution import _natural

    vc = labels.dropna().astype(str).value_counts()
    kept = sorted((str(t) for t, n in vc.items() if n >= MIN_TYPE_CELLS), key=_natural)
    small = sorted((str(t) for t, n in vc.items() if n < MIN_TYPE_CELLS), key=_natural)
    return kept, small, vc


@dataclass
class Prepared:
    """What a run needs, copied out of both datasets (they can close meanwhile)."""

    types: list[str]
    mu: np.ndarray  # types x shared genes
    S: np.ndarray  # mean counts per reference cell, per type
    n_type_cells: np.ndarray
    genes: list[str]  # shared genes, the spots' names
    ref_genes: list[str]
    keep: np.ndarray  # bool per shared gene: not excluded
    Y: sp.csr_matrix  # spots x shared genes, counts
    N: np.ndarray  # spot totals, all genes
    libraries: np.ndarray  # section per spot (str)
    clusters: pd.Series | None
    cluster_column: str | None
    small_types: list[str] = field(default_factory=list)
    info: dict[str, Any] = field(default_factory=dict)
    check: dict[str, Any] | None = None  # the self-check's synthetic spots (prepare(check=True))


def library_labels(adata) -> np.ndarray:
    from scview.core.spatial import DEFAULT_LIBRARY, LIBRARY_COLUMN

    if LIBRARY_COLUMN in adata.obs.columns:
        s = adata.obs[LIBRARY_COLUMN]
        return s.astype(str).where(s.notna(), DEFAULT_LIBRARY).to_numpy(dtype=object).astype(str)
    return np.full(adata.n_obs, DEFAULT_LIBRARY, dtype=object).astype(str)


def _row_sums(M) -> np.ndarray:
    return np.asarray(M.sum(axis=1), dtype=np.float64).ravel()


# A large dataset's counts (core/large_mode.py: the counts sidecar, read by row blocks,
# never whole) give the same inputs in one pass each.


def _on_disk(M) -> bool:
    """Whether ``M`` is a large dataset's counts on disk (``OnDiskMatrix``: row reads only)."""
    return hasattr(M, "row_blocks")


def _profiles_by_blocks(M, codes: np.ndarray, n_types: int, rows: np.ndarray | None = None):
    """``sd_profiles`` of on-disk counts in one pass by row blocks: (cell totals, mu, S,
    n, and the rows at the sorted positions ``rows`` as a CSR, or None). Each block adds
    its cells' ``X_c / n_c`` per type and mu is divided by the type's cells at the end:
    sd_profiles' mu up to the order of the additions (relative ~1e-15); the totals, S,
    n and the rows are exact."""
    totals = np.zeros(M.shape[0])
    sums = np.zeros((n_types, M.shape[1]))
    picked = []
    for start, block in M.row_blocks():
        block = sp.csr_matrix(block)
        stop = start + block.shape[0]
        t = _row_sums(block)
        totals[start:stop] = t
        c = codes[start:stop]
        ok = (c >= 0) & (t > 0)
        if ok.any():
            W = sp.csr_matrix((1.0 / t[ok], (c[ok], np.flatnonzero(ok))), shape=(n_types, block.shape[0]))
            sums += (W @ block).toarray()
        if rows is not None:
            a, b = np.searchsorted(rows, [start, stop])
            if b > a:
                picked.append(block[rows[a:b] - start])
    ok = (codes >= 0) & (totals > 0)
    n = np.bincount(codes[ok], minlength=n_types).astype(np.float64)
    mu = sums / np.maximum(n, 1)[:, None]
    S = np.bincount(codes[ok], weights=totals[ok], minlength=n_types) / np.maximum(n, 1)
    R = None
    if rows is not None:
        R = sp.vstack(picked, format="csr") if picked else sp.csr_matrix((0, M.shape[1]), dtype=M.dtype)
    return totals, mu, S, n.astype(np.int64), R


def _columns_by_blocks(M, cols: np.ndarray) -> tuple[sp.csr_matrix, np.ndarray]:
    """(``M[:, cols]`` as a float32 CSR, each row's total over every gene) of on-disk
    counts in one pass by row blocks: what the in-memory path takes, exactly."""
    parts, totals = [], np.zeros(M.shape[0])
    for start, block in M.row_blocks():
        block = sp.csr_matrix(block)
        totals[start:start + block.shape[0]] = _row_sums(block)
        parts.append(sp.csr_matrix(block[:, cols], dtype=np.float32))
    return sp.csr_matrix(sp.vstack(parts, format="csr"), dtype=np.float32), totals


@contextmanager
def large_reference(path: str):
    """A large reference (core/large_mode.py) opened for one run: an AnnData of its obs
    and var alone (nothing else is read) and its counts sidecar (``OnDiskMatrix``, row
    reads only, checked against those cells and genes), closed afterwards. Its own
    handles, not the cached dataset's: closing that (evicted for room) would close its
    sidecar in the middle of the pass, and its graphs and embeddings are not needed."""
    import anndata as ad
    import h5py
    from anndata.io import read_elem

    from scview.core import large_layout, large_mode

    with h5py.File(path, "r") as f:
        obs, var = read_elem(f["obs"]), read_elem(f["var"])
        uns = {large_layout.KEY: read_elem(f["uns"][large_layout.KEY])}
    try:
        counts = large_layout.counts_path(path, uns)
    except large_layout.LargeLayoutError as e:
        raise SpotDeconvolutionError(f"The reference's counts cannot be read: {e}") from None
    why = large_layout.counts_mismatch(counts, obs.index, var.index)
    if why is not None:
        raise SpotDeconvolutionError(f"The reference's counts file cannot be used ({why}): spot composition "
                                     "compares counts, never log values.")
    handle, M = large_mode.open_matrix(counts, large_layout.COUNTS_KEY, threading.RLock(), "the reference's counts")
    try:
        names = var.copy()
        names.index = pd.Index(var.index.astype(str))
        yield ad.AnnData(obs=obs, var=var), (M, names, "the counts file")
    finally:
        handle.close()


def prepare(ref, spots, *, celltype_column: str, ref_info: dict[str, Any], ref_fallback: list[str],
            spot_fallback: list[str], cluster_column: str | None, check: bool = False,
            seed: int = SEED, ref_counts: tuple[Any, pd.DataFrame, str] | None = None) -> Prepared:
    """Check the two datasets and copy out what a run (or, ``check``, the self-check)
    needs: the reference's profiles on the shared genes, the spots' counts.
    ``ref_counts``: the reference's (counts, var, source) when already open
    (``large_reference``)."""
    from scview.core import label_transfer as lt
    from scview.core import spatial
    from scview.core.deconvolution import _excluded
    from scview.core.species import detect_species

    if not spatial.is_spot_data(spots):
        raise SpotDeconvolutionError(NOT_SPOTS)
    try:
        labels = lt.check_labels(ref, celltype_column)
    except lt.LabelTransferError as e:
        raise SpotDeconvolutionError(str(e), getattr(e, "status", 422)) from None
    refusal = lt.check_species(detect_species(ref).species, detect_species(spots).species)
    if refusal:
        raise SpotDeconvolutionError(refusal.replace("Cell types do not map", "Cell-type profiles do not transfer"))
    types, small, _vc = type_counts(labels)
    if len(types) < 2:
        raise SpotDeconvolutionError(f"'{celltype_column}' has fewer than 2 types with {MIN_TYPE_CELLS}+ cells: "
                                     "nothing to split the spots into.")
    if len(types) > MAX_TYPES:
        raise SpotDeconvolutionError(f"'{celltype_column}' has {len(types)} types with {MIN_TYPE_CELLS}+ cells "
                                     f"(at most {MAX_TYPES}): pick a coarser cell-type column.")
    RM, rvar, rsource = (ref_counts if ref_counts is not None
                         else counts_with_var(ref, ref_fallback, "The reference"))
    SM, svar, ssource = counts_with_var(spots, spot_fallback, "This dataset")
    # Duplicate names (symbols): the first of each, on both sides; versions off Ensembl ids.
    rkey, skey = _strip_versions(rvar), _strip_versions(svar)
    match = lt.match_genes(rkey[~rkey.index.duplicated()], skey[~skey.index.duplicated()])
    pairs = list(dict.fromkeys(zip(match.shared, match.ref_names, strict=True)))
    seen_r, shared, ref_names = set(), [], []
    for s_name, r_name in pairs:
        if r_name in seen_r:
            continue
        seen_r.add(r_name)
        shared.append(s_name)
        ref_names.append(r_name)
    if len(shared) < MIN_GENES:
        raise SpotDeconvolutionError(
            f"The reference and this dataset share {len(shared):,} genes (reference {match.n_ref:,}, this dataset "
            f"{match.n_query:,}): check that both name genes the same way (symbols or Ensembl ids) and are the "
            "same species.")
    rpos = _first_positions(rkey.index, ref_names)
    spos = _first_positions(skey.index, shared)
    # Each file's own names (with versions), so the replay finds them as they are.
    shared = [str(g) for g in svar.index[spos]]
    ref_names = [str(g) for g in rvar.index[rpos]]
    codes_s = pd.Categorical(labels.astype(str).where(labels.notna(), None).to_numpy(), categories=types).codes
    codes = np.asarray(codes_s, dtype=np.int64)
    if _on_disk(RM):  # a large reference: totals and profiles in one pass over its counts
        totals, mu_full, S, n, _ = _profiles_by_blocks(RM, codes, len(types))
    else:
        totals = _row_sums(RM)
    codes = np.where(totals > 0, codes, -1)
    if int((codes >= 0).sum()) < MIN_REFERENCE_CELLS:
        raise SpotDeconvolutionError(f"Only {int((codes >= 0).sum())} reference cells have a type with counts; "
                                     f"a reference needs at least {MIN_REFERENCE_CELLS}.")
    if not _on_disk(RM):
        mu_full, S, n = sd_profiles(RM, totals, codes, len(types))
    mu = mu_full[:, rpos]
    keep = np.fromiter((not (_excluded(g) or _excluded(r)) for g, r in zip(shared, ref_names, strict=True)), bool,
                       len(shared))
    if _on_disk(SM):  # large spots: the shared genes and the totals in one pass over their counts
        Y, N = _columns_by_blocks(SM, spos)
    else:
        Y = sp.csr_matrix(SM[:, spos] if sp.issparse(SM) else np.asarray(SM)[:, spos], dtype=np.float32)
        N = _row_sums(SM)
    clusters = None
    if cluster_column and cluster_column in spots.obs.columns:
        clusters = spots.obs[cluster_column].astype(str).copy()
    synth = None
    if check:
        synth = _synthetic(RM, totals, codes, len(types), rpos, seed)
    return Prepared(
        types=types, mu=mu, S=S, n_type_cells=n, genes=shared, ref_genes=ref_names, keep=keep, Y=Y, N=N,
        libraries=library_labels(spots), clusters=clusters,
        cluster_column=cluster_column if clusters is not None else None, small_types=small,
        info={**ref_info, "celltype_column": celltype_column, "reference_source": rsource, "spot_source": ssource,
              "renamed": match.renamed, "n_ref_genes": match.n_ref, "n_spot_genes": match.n_query,
              "unit": spatial.technology(spots)["unit"] if spatial.has_spatial(spots) else "spot",
              "n_reference_cells": int((codes >= 0).sum())},
        check=synth,
    )


NOT_SPOTS = ("Spot composition needs Visium spots or Visium HD bins: this dataset has no capture grid. "
             "Imaging-based data already resolves single cells: label them from a reference instead "
             "(Data Assessment > Label from a reference dataset).")


# ---------------------------------------------------------------------------
# A run: prepared inputs -> streamed blocks -> outputs
# ---------------------------------------------------------------------------


def estimate_seconds(n_ref: int, n_spots: int, n_types: int, method: str = "rctd", *,
                     ref_values: int = 0, spot_values: int = 0, check: bool = False) -> float:
    """Wall time at 2 CPUs: reading the reference, then the fit (types^2 per spot dominate).
    ``ref_values`` / ``spot_values``: the stored values of a large side's counts, read by
    row blocks (a large reference's second pass for the self-check), instead of the
    reference read in."""
    if ref_values:
        secs = SECONDS_FIXED + LARGE_SECONDS_PER_VALUE * ref_values * (LARGE_CHECK_FACTOR if check else 1)
    else:
        secs = SECONDS_FIXED + SECONDS_PER_REF_CELL * n_ref
    secs += LARGE_SECONDS_PER_VALUE * spot_values
    if method == "rctd":
        secs += SECONDS_PER_SPOT_TYPE2 * n_spots * max(n_types, 2) ** 2
    else:
        secs += 2e-3 * n_spots
    return round(secs, 1)


def _large_pass_bytes(h) -> int:
    """One block of a pass over a large dataset's counts (``OnDiskMatrix.row_blocks``)."""
    from scview.core import large_mode

    values = min(stored_values(h), large_mode.MAX_READ_BYTES // 32)
    return int(LARGE_BLOCK_COPIES * values * 12)


def stored_values(h) -> int:
    x = h.x
    return int(x.nnz) if x is not None else 0


def need_bytes(ref_header, spot_header, ref_open: bool, check: bool = False) -> int:
    """Peak memory beyond the open datasets: the reference read from its file (unless open),
    a copy of each dataset's counts, the likelihood table and the dense blocks. A large
    side (``Header.large``) is read by row blocks instead: the reference's obs and var,
    one block, its profiles and (``check``) the drawn cells' rows; the spots' shared genes
    (about their counts, held twice while stacked) and one block."""
    from scview.core import preflight

    need = 0
    if ref_header.large:
        n, m = ref_header.n_cells or 0, ref_header.n_genes or 0
        need += preflight._frames_bytes(ref_header) + _large_pass_bytes(ref_header) + 2 * MAX_TYPES * m * 8
        if check:
            drawn = min(n, CHECK_SPOTS * CHECK_CELLS[1])
            need += 2 * 12 * stored_values(ref_header) * drawn // max(n, 1)
    else:
        need += 0 if ref_open else preflight.estimate_load(ref_header, ingest=False).peak_bytes
        need += NEED_COUNTS_COPIES * preflight._lt_counts_bytes(ref_header)
    if spot_header.large:
        need += _large_pass_bytes(spot_header) + 2 * preflight._lt_counts_bytes(spot_header)
    else:
        need += NEED_COUNTS_COPIES * preflight._lt_counts_bytes(spot_header)
    return int(need + NEED_FIXED)


def iter_fit(p: Prepared, *, method: str, stop: Callable[[], bool] | None = None,
             seed: int = SEED) -> Iterator[dict[str, Any]]:
    """The run as events: ``progress`` per stage and block, then ``fitted`` with the outcome."""
    if method not in METHODS:
        raise SpotDeconvolutionError(f"Unknown method '{method}': one of {', '.join(METHODS)}.")
    t0 = time.monotonic()
    gen = sd_iter_deconvolve(p.mu, p.S, p.Y, p.N, p.libraries, method, p.keep, seed=seed)
    messages = {"genes": "Choosing the genes", "platform": "Estimating each section's platform effect",
                "baseline": "Starting values (NNLS per spot)", "sigma": "Choosing the noise level on 200 spots",
                "table": "Tabulating the likelihood", "fit": "Fitting the spots"}
    try:
        for event, data in gen:
            if stop is not None and stop():
                raise SpotDeconvolutionCancelled()
            if event == "fitted":
                data["seconds"] = round(time.monotonic() - t0, 1)
                yield {"type": "fitted", "out": data}
            else:
                yield {"type": "progress", "message": messages.get(data["step"], data["step"]), **data}
    except ValueError as e:
        raise SpotDeconvolutionError(str(e)) from None
    finally:
        gen.close()


def _share_stats(M: np.ndarray, types: list[str]) -> list[dict[str, Any]]:
    ok = np.isfinite(M).all(axis=1)
    sub = M[ok]
    out = []
    for k, t in enumerate(types):
        col = sub[:, k] if sub.size else np.zeros(0)
        out.append({"type": t, "mean": round(float(col.mean()), 5) if col.size else None,
                    "max": round(float(col.max()), 5) if col.size else None})
    return out


def _cluster_means(M: np.ndarray, clusters: pd.Series | None) -> dict[str, list[float | None]]:
    if clusters is None:
        return {}
    ok = np.isfinite(M).all(axis=1)
    out: dict[str, list[float | None]] = {}
    labels = clusters.to_numpy()
    from scview.core.deconvolution import _natural

    for c in sorted(set(labels[ok].tolist()), key=lambda x: _natural(str(x))):
        rows = ok & (labels == c)
        out[str(c)] = [round(float(v), 4) for v in M[rows].mean(axis=0)]
    return out


def outputs(p: Prepared, out: dict[str, Any]) -> dict[str, Any]:
    """The arrays saved: RNA and cell shares (float32), dominant type and its share."""
    beta, cells, fitted = out["beta"], out["cells"], out["fitted"]
    top_idx = np.where(fitted, np.nanargmax(np.where(np.isfinite(beta), beta, -1.0), axis=1), -1)
    top = np.where(fitted, np.asarray(p.types, dtype=object)[np.maximum(top_idx, 0)], LOW_COUNTS)
    share = np.where(fitted, beta[np.arange(beta.shape[0]), np.maximum(top_idx, 0)], np.nan)
    return {"rna": beta.astype(np.float32), "cells": cells.astype(np.float32), "top": top.astype(str),
            "top_share": share.astype(np.float32)}


def summarize(p: Prepared, out: dict[str, Any], *, method: str) -> dict[str, Any]:
    arr = outputs(p, out)
    fitted = out["fitted"]
    rna, cells = arr["rna"].astype(np.float64), arr["cells"].astype(np.float64)
    top_counts = pd.Series(arr["top"][fitted]).value_counts()
    libs = sorted(set(p.libraries.tolist()))
    n_unconv = int((~out["converged"][fitted]).sum()) if method == "rctd" else 0
    notes = list(out["notes"])
    if p.small_types:
        notes.insert(0, f"Left out of the reference (fewer than {MIN_TYPE_CELLS} cells): "
                     + ", ".join(p.small_types[:10]) + (" ..." if len(p.small_types) > 10 else "") + ".")
    n_skipped = int((~fitted).sum())
    unit = p.info.get("unit", "spot")
    if n_skipped:
        notes.append(f"{n_skipped:,} {unit}s with fewer than {MIN_SPOT_UMI} counts were not fitted (\"{LOW_COUNTS}\").")
    if n_unconv:
        notes.append(f"{n_unconv:,} {unit}s did not settle within {MAX_ITER} rounds; their last estimate is kept.")
    if p.info.get("renamed"):
        notes.append(f"Genes were matched after re-indexing {p.info['renamed']}.")
    stats_rna, stats_cells = _share_stats(rna, p.types), _share_stats(cells, p.types)
    return {
        "method": method, "method_name": METHOD_NAMES[method], "types": list(p.types),
        "type_stats": [{"type": t, "n_reference_cells": int(p.n_type_cells[k]), "cell_size": round(float(p.S[k]), 1),
                        "rna_mean": stats_rna[k]["mean"], "rna_max": stats_rna[k]["max"],
                        "cell_mean": stats_cells[k]["mean"], "cell_max": stats_cells[k]["max"],
                        "n_top": int(top_counts.get(t, 0))} for k, t in enumerate(p.types)],
        "small_types": list(p.small_types),
        "reference": {key: p.info.get(key) for key in ("id", "name", "path", "mtime", "celltype_column",
                                                       "n_reference_cells")},
        "celltype_column": p.info.get("celltype_column"),
        "sources": {"reference": p.info.get("reference_source"), "spots": p.info.get("spot_source"),
                    "renamed": p.info.get("renamed")},
        "s_e": out["s_e"],
        "libraries": [{"id": lib, "n_spots": int((p.libraries == lib).sum()),
                       "n_fitted": int(((p.libraries == lib) & fitted).sum()),
                       "s_g": None if out["s_g"].get(lib) is None else round(float(out["s_g"][lib]), 4),
                       "bulk": None if out["bulk"].get(lib) is None else [round(float(v), 4) for v in out["bulk"][lib]]}
                      for lib in libs],
        "n_shared_genes": len(p.genes), "n_genes": int(out["fit"].size), "n_platform_genes": int(out["platform"].size),
        "n_spots": int(len(fitted)), "n_fitted": int(fitted.sum()), "n_skipped": n_skipped,
        "n_unconverged": n_unconv, "n_clipped": int(out["n_clipped"]),
        "cluster_column": p.cluster_column,
        "clusters": {"rna": _cluster_means(rna, p.clusters), "cells": _cluster_means(cells, p.clusters)},
        "columns": {"rna": OBSM_RNA, "cells": OBSM_CELLS, "top": OBS_TOP, "top_share": OBS_TOP_SHARE},
        "seconds": out.get("seconds"), "notes": notes, "note": NOTE,
        "created_at": time.time(),
    }


def save(adata, p: Prepared, out: dict[str, Any], summary: dict[str, Any], resources: dict | None = None
         ) -> dict[str, Any]:
    """Write obsm spot_deconv / spot_deconv_cells, obs spot_deconv_top(_share), the
    summary in uns (a JSON string: labels may hold "/") and the provenance step."""
    from scview.core import provenance

    arr = outputs(p, out)
    adata.obsm[OBSM_RNA] = arr["rna"]
    adata.obsm[OBSM_CELLS] = arr["cells"]
    present = [t for t in p.types if (arr["top"] == t).any()]
    adata.obs[OBS_TOP] = pd.Categorical(arr["top"], categories=[*present, *([LOW_COUNTS] if (arr["top"] == LOW_COUNTS)
                                                                             .any() else [])])
    adata.obs[OBS_TOP_SHARE] = arr["top_share"]
    adata.uns[UNS_KEY] = json.dumps(summary)  # replaced, never mutated in place
    try:
        ref = summary["reference"]
        provenance.record_step(
            adata, step=STEP, tool=f"scview spot composition ({METHOD_NAMES[summary['method']]}, Cable 2022)",
            params={
                "reference": ref.get("name"), "reference_id": ref.get("id"),
                "reference_file": os.path.basename(str(ref.get("path") or "")), "reference_mtime": ref.get("mtime"),
                "celltype_column": summary["celltype_column"], "method": summary["method"], "types": list(p.types),
                "s_e": summary["s_e"], "genes": [p.genes[i] for i in out["platform"]],
                "ref_genes": [p.ref_genes[i] for i in out["platform"]],
                "fit_genes": [int(i) for i in np.searchsorted(out["platform"], out["fit"])],
                "n_genes": summary["n_genes"], "renamed": p.info.get("renamed"),
                "reference_source": p.info.get("reference_source"), "spot_source": p.info.get("spot_source"),
                "outputs": list(summary["columns"].values()), "n_skipped": summary["n_skipped"],
            },
            resources=resources,
        )
    except Exception as e:  # provenance must never block the result
        logger.warning("Could not record spot composition provenance: %s", e)
    return summary


def saved(adata) -> dict[str, Any] | None:
    """The saved run's summary while its arrays are still in the dataset, else None."""
    raw = adata.uns.get(UNS_KEY) if hasattr(adata, "uns") else None
    if raw is None or OBSM_RNA not in getattr(adata, "obsm", {}):
        return None
    try:
        data = json.loads(raw) if isinstance(raw, str) else dict(raw)
    except (TypeError, ValueError):
        return None
    if not isinstance(data, dict) or np.shape(adata.obsm[OBSM_RNA])[1] != len(data.get("types") or []):
        return None
    return data


def share_values(adata, cell_type: str, kind: str = "rna") -> np.ndarray:
    """One type's share per spot (NaN unfitted) from the saved run."""
    s = saved(adata)
    if s is None:
        raise SpotDeconvolutionError("No spot composition saved on this dataset yet.", 404)
    if cell_type not in s["types"]:
        raise SpotDeconvolutionError(f"'{cell_type}' is not one of the saved run's types.", 404)
    key = OBSM_CELLS if kind == "cell" else OBSM_RNA
    return np.asarray(adata.obsm[key], dtype=np.float32)[:, s["types"].index(cell_type)]


# ---------------------------------------------------------------------------
# The self-check: synthetic spots from a held-out half of the reference
# ---------------------------------------------------------------------------


def _synthetic(RM, totals, codes, n_types, rpos, seed) -> dict[str, Any]:
    """Split each type's cells 50/50 (seeded): profiles from half A; CHECK_SPOTS spots
    of CHECK_CELLS cells from half B (Dirichlet(CHECK_ALPHA) types), summed, then thinned
    to each of CHECK_DEPTHS (normal, sd a quarter of the depth). Truth: RNA shares."""
    rng = np.random.default_rng(seed)
    a_codes = np.full(codes.size, -1)
    b_cells: list[np.ndarray] = []
    for k in range(n_types):
        idx = rng.permutation(np.flatnonzero(codes == k))
        half = idx.size // 2
        a_codes[idx[:half]] = k
        b_cells.append(idx[half:])
    n_cells = rng.integers(CHECK_CELLS[0], CHECK_CELLS[1] + 1, size=CHECK_SPOTS)
    rows, cols, truth_rna, truth_cells = [], [], np.zeros((CHECK_SPOTS, n_types)), np.zeros((CHECK_SPOTS, n_types))
    for i in range(CHECK_SPOTS):
        props = rng.dirichlet(np.full(n_types, CHECK_ALPHA))
        nk = rng.multinomial(n_cells[i], props)
        for k in np.flatnonzero(nk):
            picked = rng.choice(b_cells[k], size=nk[k], replace=True)
            rows.extend([i] * picked.size)
            cols.extend(picked.tolist())
            truth_rna[i, k] += totals[picked].sum()
            truth_cells[i, k] += picked.size
    A = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(CHECK_SPOTS, RM.shape[0]))
    if _on_disk(RM):
        # A large reference: half A's profiles and the drawn cells' rows in one more pass.
        # A's columns cut to the drawn cells (same order), so the product is A @ RM's.
        used = np.unique(np.asarray(cols, dtype=np.int64))
        _t, mu_a, S_a, n_a, R = _profiles_by_blocks(RM, a_codes, n_types, rows=used)
        summed = sp.csr_matrix(A[:, used] @ R, dtype=np.float64)
    else:
        mu_a, S_a, n_a = sd_profiles(RM, totals, a_codes, n_types)
        summed = sp.csr_matrix(A @ RM, dtype=np.float64)
    summed.data = np.round(summed.data)
    depth_sets = []
    for depth in CHECK_DEPTHS:
        tot = np.asarray(summed.sum(axis=1)).ravel()
        target = np.maximum(rng.normal(depth, depth / 4, size=CHECK_SPOTS), 200)
        frac = np.minimum(1.0, target / np.maximum(tot, 1))
        M = summed.copy()
        M.data = rng.binomial(M.data.astype(np.int64), np.repeat(frac, np.diff(M.indptr))).astype(np.float64)
        M.eliminate_zeros()
        depth_sets.append({"depth": depth, "Y": sp.csr_matrix(M[:, rpos], dtype=np.float32),
                           "N": np.asarray(M.sum(axis=1)).ravel()})
    return {"mu": mu_a[:, rpos], "S": S_a, "n": n_a, "sets": depth_sets,
            "truth_rna": truth_rna / truth_rna.sum(axis=1, keepdims=True),
            "truth_cells": truth_cells / truth_cells.sum(axis=1, keepdims=True), "seed": seed}


def _jsd(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Jensen-Shannon divergence per row (base 2, 0..1)."""
    m = 0.5 * (p + q)
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.where(p > 0, p * np.log2(p / m), 0.0).sum(axis=1)
        b = np.where(q > 0, q * np.log2(q / m), 0.0).sum(axis=1)
    return 0.5 * (a + b)


def _aupr(truth: np.ndarray, est: np.ndarray) -> float | None:
    """Average precision of the estimates for "the type is in the spot"."""
    y = (truth > 0).ravel()
    s = est.ravel()
    if not y.any() or y.all():
        return None
    order = np.argsort(-s, kind="stable")
    hits = y[order]
    prec = np.cumsum(hits) / np.arange(1, hits.size + 1)
    return float((prec * hits).sum() / hits.sum())


def metrics(truth: np.ndarray, est: np.ndarray, types: list[str], truth_cells: np.ndarray | None = None,
            est_cells: np.ndarray | None = None) -> dict[str, Any]:
    ok = np.isfinite(est).all(axis=1)
    t, e = truth[ok], est[ok]
    per_type = []
    for k, name in enumerate(types):
        x, y = t[:, k], e[:, k]
        r = float(np.corrcoef(x, y)[0, 1]) if x.size >= 3 and x.std() > 0 and y.std() > 0 else None
        per_type.append({"type": name, "r": r, "bias": float((y - x).mean()) if x.size else None,
                         "rmse": float(np.sqrt(((y - x) ** 2).mean())) if x.size else None})
    out = {"rmse": float(np.sqrt(((e - t) ** 2).mean())) if t.size else None,
           "jsd": float(_jsd(t, e).mean()) if t.size else None, "aupr": _aupr(t, e), "n_spots": int(ok.sum()),
           "per_type": per_type}
    if truth_cells is not None and est_cells is not None:
        tc, ec = truth_cells[ok], est_cells[ok]
        out["rmse_cells"] = float(np.sqrt(((ec - tc) ** 2).mean())) if tc.size else None
    return out


def iter_check(p: Prepared, *, methods: tuple[str, ...] = METHODS,
               stop: Callable[[], bool] | None = None) -> Iterator[dict[str, Any]]:
    """The self-check as events: ``progress`` per method and depth, then ``checked``."""
    if p.check is None:
        raise SpotDeconvolutionError("The self-check's synthetic spots were not prepared.", 500)
    t0 = time.monotonic()
    c = p.check
    runs = []
    total = len(methods) * len(c["sets"])
    done = 0
    yield {"type": "progress", "step": "check", "done": 0, "total": total,
           "message": f"Fitting {CHECK_SPOTS:,} synthetic spots"}
    for ds in c["sets"]:
        libs = np.full(ds["Y"].shape[0], "synthetic", dtype=object).astype(str)
        for method in methods:
            gen = sd_iter_deconvolve(c["mu"], c["S"], ds["Y"], ds["N"], libs, method, p.keep, seed=c["seed"])
            res = None
            try:
                for event, data in gen:
                    if stop is not None and stop():
                        raise SpotDeconvolutionCancelled()
                    if event == "fitted":
                        res = data
            except ValueError as e:
                raise SpotDeconvolutionError(str(e)) from None
            finally:
                gen.close()
            assert res is not None
            m = metrics(c["truth_rna"], res["beta"], p.types, c["truth_cells"], res["cells"])
            runs.append({"method": method, "depth": ds["depth"], "s_e": res["s_e"],
                         "median_umi": float(np.median(ds["N"])), "n_genes": int(res["fit"].size), **m})
            done += 1
            yield {"type": "progress", "step": "check", "done": done, "total": total,
                   "message": f"{METHOD_NAMES[method]} at {ds['depth']:,} counts per spot"}
    yield {"type": "checked", "out": {
        "id": uuid.uuid4().hex[:12], "types": list(p.types), "runs": runs, "n_spots": CHECK_SPOTS,
        "cells_per_spot": list(CHECK_CELLS), "depths": list(CHECK_DEPTHS), "seed": c["seed"],
        "reference": {key: p.info.get(key) for key in ("id", "name", "celltype_column")},
        "seconds": round(time.monotonic() - t0, 1), "created_at": time.time(),
        "note": ("Synthetic spots from held-out reference cells: no platform effect and no type missing from "
                 "the reference, so real sections fit worse than this.")}}


# ---------------------------------------------------------------------------
# Kept results and the co-pilot's fact
# ---------------------------------------------------------------------------


@dataclass
class Kept:
    """A summary (run or check) kept in the results store."""

    id: str
    summary: dict[str, Any]
    cells: str = ""


_last: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_last_check: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_lock = threading.Lock()


def remember(adaptor, summary: dict[str, Any], result_id: str) -> None:
    k = Kept(result_id, summary, results_store.cells_hash(adaptor.adata))
    try:
        with _lock:
            _last[adaptor] = k
    except TypeError:  # a test double
        pass
    results_store.keep(adaptor, KIND, k, result_id=result_id, key="",
                       label=f"Spot composition from {summary['reference'].get('name')} "
                             f"({len(summary['types'])} types, {summary['method']})")


def remember_check(adaptor, out: dict[str, Any]) -> None:
    k = Kept(out["id"], out, results_store.cells_hash(adaptor.adata))
    try:
        with _lock:
            _last_check[adaptor] = k
    except TypeError:
        pass
    results_store.keep(adaptor, CHECK_KIND, k, result_id=out["id"], key="",
                       label=f"Spot composition self-check ({out['reference'].get('name')})")


def last_check(adaptor) -> dict[str, Any] | None:
    try:
        cells = results_store.cells_hash(adaptor.adata)
        k = results_store.remembered(_last_check, _lock, adaptor, CHECK_KIND)
        if k is None:
            k = results_store.latest(adaptor, CHECK_KIND, current_only=False)
            if k is not None and k.cells == cells:
                with _lock:
                    _last_check.setdefault(adaptor, k)
    except Exception:  # a test double, an unreadable result
        return None
    return k.summary if k is not None and k.cells == cells else None


def grounding_lines(adaptor) -> tuple[list[str], str | None]:
    """For the co-pilot: the saved run's dominant types and each section's shares."""
    s = saved(adaptor.adata)
    if s is None:
        return [], None
    ref = s.get("reference") or {}
    detail = (f"Spot composition from {ref.get('name')} ({s.get('celltype_column')}, {len(s['types'])} types), "
              f"{s['method_name']}, {s['n_fitted']:,} of {s['n_spots']:,} spots fitted on {s['n_genes']:,} genes")
    lines = [f"\n## Spot composition (saved run) [{TAG}]", f"- {detail} [{TAG}]."]
    stats = sorted(s["type_stats"], key=lambda r: -(r["n_top"] or 0))
    tops = [f"{r['type']} {r['n_top']:,}" for r in stats if r["n_top"]][:8]
    if tops:
        lines.append("- Spots by dominant type (largest RNA share): " + "; ".join(tops) + ".")
    for lib in s["libraries"][:4]:
        if lib.get("bulk"):
            order = np.argsort(lib["bulk"])[::-1][:5]
            lines.append(f"- Section {lib['id']} RNA shares (section-wide): "
                         + ", ".join(f"{s['types'][i]} {lib['bulk'][i]:.0%}" for i in order) + ".")
    means = sorted(s["type_stats"], key=lambda r: -(r["rna_mean"] or 0))[:5]
    lines.append("- Mean RNA share over spots: " + ", ".join(f"{r['type']} {(r['rna_mean'] or 0):.0%}" for r in means)
                 + ".")
    lines.append("- Shares are estimates of each spot's RNA from the reference's profiles, not counts of cells; a "
                 "type missing from the reference is given to its nearest type.")
    return lines, detail


def estimate_note(seconds: float) -> str:
    return f"about {math.ceil(seconds / 5) * 5:.0f} s" if seconds < 90 else f"about {math.ceil(seconds / 60):.0f} min"
