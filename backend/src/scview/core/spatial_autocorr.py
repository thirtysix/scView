"""Spatially variable genes: global Moran's I per gene (C099).

Equal to squidpy 1.6's defaults, ``sq.gr.spatial_neighbors(adata, library_key=
"library_id")`` then ``sq.gr.spatial_autocorr(adata, mode="moran")``, without
importing squidpy (source read: squidpy/gr/_build.py, _ppatterns.py, BSD-3; and
scanpy/metrics/_morans_i.py):

- Neighbours, one graph per ``obs["library_id"]`` (block-diagonal: never across
  sections). ``grid`` (Visium or Visium HD: ``uns["scview_spatial"]["technology"]``,
  or, with no such record, squidpy's own rule: ``uns["spatial"]`` exists): each
  spot's 6 nearest spots (Visium's hexagons; a Visium HD bin's 4 nearest, the bins it
  shares a side with: squidpy's ``coord_type="grid", n_neighs=4``; Euclidean on
  ``obsm["spatial"]``), kept when nearer than 1.3 x the median of those distances in
  its section, so an edge spot keeps its fewer real neighbours. ``knn`` otherwise: the
  6 nearest, no cut. Directed, as
  squidpy builds it (j among i's nearest, not necessarily i among j's). A spot
  matches itself out as sklearn's ``kneighbors()`` does (with duplicates at one
  position, the first hit is dropped when the spot itself is not among them).
- Weights row-normalised (``transformation=True``); a spot with no neighbour keeps
  a zero row and still counts in n and the mean, as in squidpy.
- I = n / S0 * sum_ij w_ij z_i z_j / sum_i z_i^2. One section: z = x - its mean, as
  squidpy. Several sections (all of them in one run): z = x - the mean of ITS section,
  so a gene that is simply higher on one slide than on another (a batch or a condition)
  does not count as spatial pattern; squidpy's global mean would (8 Kuppe 2022 slides:
  pooled MALAT1 0.61 against at most 0.48 on any one slide). Each section's own I, z
  and p come with it (``by_library``), each equal to a run on that section alone. The
  pooled p is the normal approximation for residuals of the section means (Cliff and
  Ord 1981): E[I] = n / S0 tr(MW) / (n - L), Var[I] = (n / S0)^2 (tr(MWMW') + tr(MWMW)
  + tr(MW)^2) / ((n - L)(n - L + 2)) - E[I]^2, M the centring within L sections; with
  L = 1 these are squidpy's. Computed from per-gene sums in row blocks, so neither
  the matrix nor a gene block is ever made dense whole. A large dataset (``core/
  large_mode.py``) gives the same sums from the gene store, a block of genes at a time
  (``moran_sums_by_genes``), equal to a relative 1e-9 (the additions come in another order).
- p under the normality assumption (``n_perms=None``): E[I] = -1/(n-1), Var[I] =
  (n^2 S1 - n S2 + 3 S0^2) / ((n-1)(n+1) S0^2) - E[I]^2; one-tailed
  (``two_tailed=False``) in the direction of z. squidpy computes ``1 - cdf(z)``,
  which is 0 beyond z ~ 8.3; here the normal tail (``sf``) is used, equal to 1e-12
  absolute and gives the real small p. Benjamini-Hochberg over the genes (fdr_bh).
- Genes: ``highly_variable`` when the dataset has them, else all (squidpy's
  default), or all, or a list. Constant genes (squidpy: I = NaN) are left out and
  counted. Values: the log-normalised matrix (``activity.lognorm_matrix``: .raw
  after scView's pipeline, else X when it holds log values, else the shifted log of
  the counts), or a named layer; squidpy's run on that matrix as X gives the same.

The functions in ``REPLAY_CODE`` use numpy and scipy only (copied into the replay
script with ``REPLAY_CONSTANTS``).
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
import uuid
import weakref
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats
from scipy.spatial import cKDTree

from scview.core import large_mode, off_tissue, results_store

logger = logging.getLogger(__name__)

KIND = "spatial_autocorr"
TAG = "result:spatial_genes"
UNS_KEY = "moranI"  # squidpy's key, so an exported h5ad reads in squidpy as its own result
STEP = "spatial_autocorr"
LIBRARY_COLUMN = "library_id"

N_NEIGHS = 6
HD_NEIGHS = 4  # a square grid: the bins sharing a side
GRID_SWAY = 1.3  # squidpy: "there's a small amount of sway" in a grid's spacing
BLOCK_BYTES = 64_000_000  # per row block of W @ X (dense values)
GENE_CHOICES = ("hvg", "all")
MAX_LIST_GENES = 50_000
TOP_FACT_GENES = 10
FACT_PADJ = 0.05
# Measured on 2 cores (synthetic Visium grid, spots in random order, 10% non-zero, CSR
# float32, sorted into blocks by spatial_order): 5,184 spots x 2,000 genes 0.1 s, x 20,000
# genes 1.3 s; 20,736 x 20,000 4.8 s.
SECONDS_PER_VALUE = 1.3e-8
# A large dataset, from its gene store (sparse blocks of genes): 393,491 bins x 18,821 genes, 17 s.
LARGE_SECONDS_PER_VALUE = 2.3e-9
LONG_SECONDS = 4.0  # the UI says how long above this
GENE_BLOCK = 64  # genes read at a time from a large dataset's gene store
NEED_PER_OBS = 400  # KD-tree, its query's distances and indices, W and W + W'
NEED_LARGE_BLOCK = 256 * 2**20  # a block of genes and W times it (8 um bins, all genes: 350 MB in all)

REPLAY_CONSTANTS = ("N_NEIGHS", "GRID_SWAY", "BLOCK_BYTES")


class SpatialAutocorrError(ValueError):
    """A run that cannot start (no spatial data, no such section, no genes): a 422."""


# ---------------------------------------------------------------------------
# The computation (REPLAY_CODE: copied into the replay script)
# ---------------------------------------------------------------------------


def knn_edges(coords, n_neighs, grid):
    """(rows, cols) of each point's ``n_neighs`` nearest other points, as squidpy's
    spatial_neighbors builds them for one section; ``grid``: only those nearer than
    GRID_SWAY x the median of all those distances (Visium's hexagonal grid)."""
    coords = np.asarray(coords, dtype=np.float64)
    n = coords.shape[0]
    k = min(int(n_neighs), n - 1)
    if k < 1:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    dist, idx = cKDTree(coords).query(coords, k=k + 1)
    dist, idx = np.atleast_2d(dist), np.atleast_2d(idx)
    # Drop the point itself (sklearn kneighbors() rule: when it is not among the hits,
    # duplicates at its position, the first hit goes).
    keep = idx != np.arange(n)[:, None]
    no_self = keep.all(axis=1)
    keep[no_self, 0] = False
    dist = dist[keep].reshape(n, k).ravel()
    cols = idx[keep].reshape(n, k).ravel().astype(np.int64)
    rows = np.repeat(np.arange(n, dtype=np.int64), k)
    if grid:
        near = dist < np.median(dist) * GRID_SWAY
        rows, cols = rows[near], cols[near]
    return rows, cols


def spatial_weights(coords, libraries, grid, n_neighs=N_NEIGHS):
    """Row-normalised spatial weights (CSR, n x n): a kNN graph per section (no edge
    between sections), each row summing to 1 (0 for a spot without neighbours)."""
    coords = np.asarray(coords, dtype=np.float64)
    n = coords.shape[0]
    libraries = np.asarray(libraries)
    rows, cols = [], []
    for lib in pd.unique(libraries):
        where = np.flatnonzero(libraries == lib)
        r, c = knn_edges(coords[where], n_neighs, grid)
        rows.append(where[r])
        cols.append(where[c])
    rows = np.concatenate(rows) if rows else np.zeros(0, dtype=np.int64)
    cols = np.concatenate(cols) if cols else np.zeros(0, dtype=np.int64)
    W = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    deg = np.asarray(W.sum(axis=1)).ravel()
    scale = np.divide(1.0, deg, out=np.zeros_like(deg), where=deg > 0)
    return sp.csr_matrix(sp.diags(scale) @ W)


def weight_moments(W):
    """S0, S1, S2 of a weights matrix (pysal / squidpy's _g_moments)."""
    W = sp.csr_matrix(W)
    s0 = float(W.sum())
    t = W + W.T
    s1 = float(t.multiply(t).sum()) / 2.0
    s2 = float(((np.asarray(W.sum(axis=1)).ravel() + np.asarray(W.sum(axis=0)).ravel()) ** 2).sum())
    return s0, s1, s2


def moran_sums(W, M, order=None):
    """Per gene (columns of M, spots x genes, CSR or dense): (x'Wx, sum x, sum x^2,
    x . rowsums(W), x . colsums(W), constant). Row blocks: a block's spots and their
    neighbours are made dense (at most BLOCK_BYTES), never the whole matrix; ``order``
    (spots sorted by position) keeps a block's neighbours few."""
    W = sp.csr_matrix(W)
    n, g = M.shape
    if sp.issparse(M):
        M = sp.csr_matrix(M)
    r = np.asarray(W.sum(axis=1)).ravel()
    c = np.asarray(W.sum(axis=0)).ravel()
    xwx, s, ss, xr, xc = (np.zeros(g) for _ in range(5))
    lo, hi = np.full(g, np.inf), np.full(g, -np.inf)
    step = max(1, int(BLOCK_BYTES // (8 * max(g, 1) * (N_NEIGHS + 1))))
    order = np.arange(n) if order is None else np.asarray(order)
    for a in range(0, n, step):
        rows = order[a:a + step]
        Wb = W[rows]
        near = np.unique(np.concatenate([rows, Wb.indices]))  # sorted
        XR = M[near]
        XR = XR.toarray() if sp.issparse(XR) else np.asarray(XR)
        XR = XR.astype(np.float64, copy=False)
        Wb = sp.csr_matrix((Wb.data, np.searchsorted(near, Wb.indices), Wb.indptr), shape=(len(rows), len(near)))
        Xb = XR[np.searchsorted(near, rows)]
        xwx += (Xb * (Wb @ XR)).sum(axis=0)
        s += Xb.sum(axis=0)
        ss += (Xb * Xb).sum(axis=0)
        lo = np.minimum(lo, Xb.min(axis=0))
        hi = np.maximum(hi, Xb.max(axis=0))
        xr += r[rows] @ Xb
        xc += c[rows] @ Xb
    return xwx, s, ss, xr, xc, lo == hi


def bh_adjust(p):
    """Benjamini-Hochberg adjusted p (statsmodels' fdr_bh)."""
    p = np.asarray(p, dtype=np.float64)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(adj, 1.0)
    return out


def spatial_order(coords, libraries):
    """Spots sorted by section, then row (y), then x: row blocks of neighbours."""
    coords = np.asarray(coords, dtype=np.float64)
    return np.lexsort((coords[:, 0], coords[:, 1], np.asarray(libraries).astype(str)))


def moran_table(W, sums, genes):
    """Moran's I per gene from ``moran_sums``'s sums, with squidpy's analytic p-values,
    sorted by I (highest first); constant genes are left out. Returns (table, number of
    constant genes)."""
    n = W.shape[0]
    xwx, s, ss, xr, xc, constant = sums
    s0, s1, s2 = weight_moments(W)
    mean = s / n
    num = xwx - mean * xr - mean * xc + mean * mean * s0
    den = ss - n * mean * mean
    keep = ~constant
    with np.errstate(divide="ignore", invalid="ignore"):
        score = n / s0 * num[keep] / den[keep]
    expected = -1.0 / (n - 1)
    var = (n * n * s1 - n * s2 + 3 * s0 * s0) / ((n - 1) * (n + 1) * s0 * s0) - expected ** 2
    z = (score - expected) / np.sqrt(var)
    p = np.where(z > 0, stats.norm.sf(z), stats.norm.cdf(z))
    table = pd.DataFrame({
        "gene": np.asarray(genes, dtype=object)[keep], "I": score, "expected_I": expected,
        "var_I": var, "z": z, "pval": p, "pval_adj": bh_adjust(p),
    })
    table = table.sort_values("I", ascending=False, kind="mergesort").reset_index(drop=True)
    return table, int(constant.sum())


def morans_i(W, M, genes, order=None):
    """Moran's I per gene of a matrix read by rows (``moran_table``)."""
    return moran_table(W, moran_sums(W, M, order), genes)


def section_sums(W, M, sections, n_sections, order=None):
    """``moran_sums`` split by section (``sections``: codes 0..n_sections - 1; ``W`` links no
    two sections): each sum as sections x genes, and constant per section."""
    W = sp.csr_matrix(W)
    n, g = M.shape
    if sp.issparse(M):
        M = sp.csr_matrix(M)
    sections = np.asarray(sections, dtype=np.int64)
    r = np.asarray(W.sum(axis=1)).ravel()
    c = np.asarray(W.sum(axis=0)).ravel()
    xwx, s, ss, xr, xc = (np.zeros((n_sections, g)) for _ in range(5))
    lo, hi = np.full((n_sections, g), np.inf), np.full((n_sections, g), -np.inf)
    step = max(1, int(BLOCK_BYTES // (8 * max(g, 1) * (N_NEIGHS + 1))))
    order = np.arange(n) if order is None else np.asarray(order)
    for a in range(0, n, step):
        rows = order[a:a + step]
        Wb = W[rows]
        near = np.unique(np.concatenate([rows, Wb.indices]))
        XR = M[near]
        XR = XR.toarray() if sp.issparse(XR) else np.asarray(XR)
        XR = XR.astype(np.float64, copy=False)
        Wb = sp.csr_matrix((Wb.data, np.searchsorted(near, Wb.indices), Wb.indptr), shape=(len(rows), len(near)))
        Xb = XR[np.searchsorted(near, rows)]
        sec = sections[rows]
        P = sp.csr_matrix((np.ones(len(rows)), (sec, np.arange(len(rows)))), shape=(n_sections, len(rows)))
        xwx += P @ (Xb * (Wb @ XR))
        s += P @ Xb
        ss += P @ (Xb * Xb)
        xr += P @ (r[rows][:, None] * Xb)
        xc += P @ (c[rows][:, None] * Xb)
        for k in np.unique(sec):
            part = Xb[sec == k]
            lo[k] = np.minimum(lo[k], part.min(axis=0))
            hi[k] = np.maximum(hi[k], part.max(axis=0))
    return xwx, s, ss, xr, xc, lo == hi


def centred_moments(W, sections, n_sections):
    """(E[I], Var[I]) of Moran's I on values centred within each section, under the normality
    assumption (Cliff and Ord 1981, regression residuals on section indicators). One
    section gives squidpy's -1 / (n - 1) and its variance."""
    W = sp.csr_matrix(W)
    sections = np.asarray(sections, dtype=np.int64)
    n = W.shape[0]
    n_l = np.bincount(sections, minlength=n_sections).astype(np.float64)
    r = np.asarray(W.sum(axis=1)).ravel()
    c = np.asarray(W.sum(axis=0)).ravel()
    s0_l = np.bincount(sections, weights=r, minlength=n_sections)
    s0 = float(s0_l.sum())
    used = n_l > 0
    inv = np.divide(1.0, n_l, out=np.zeros_like(n_l), where=used)
    tr_mw = -float((s0_l * inv).sum())
    jwj = float((s0_l * s0_l * inv * inv).sum())
    tr_mwmwt = (float(W.multiply(W).sum()) - float((np.bincount(sections, weights=c * c, minlength=n_sections) * inv).sum())
                - float((np.bincount(sections, weights=r * r, minlength=n_sections) * inv).sum()) + jwj)
    tr_mwmw = (float(W.multiply(W.T).sum()) - 2 * float((np.bincount(sections, weights=c * r, minlength=n_sections)
                                                          * inv).sum()) + jwj)
    k = int(used.sum())
    scale = n / s0
    expected = scale * tr_mw / (n - k)
    var = scale * scale * (tr_mwmwt + tr_mwmw + tr_mw * tr_mw) / ((n - k) * (n - k + 2)) - expected * expected
    return expected, var


def section_tables(W, sums, sections, names, genes):
    """From ``section_sums``: (the pooled table, values centred within each section, as
    ``moran_table`` lays it out; each section's own table, its I as ``moran_table`` gives
    for that section alone, with library_id and n_obs; the number of genes constant in
    every section). A gene constant in one section only is left out of that section's table."""
    W = sp.csr_matrix(W)
    sections = np.asarray(sections, dtype=np.int64)
    xwx, s, ss, xr, xc, constant = sums
    L = len(names)
    n = W.shape[0]
    n_l = np.bincount(sections, minlength=L).astype(np.float64)
    genes = np.asarray(genes, dtype=object)
    nums, dens, parts = [], [], []
    for k in range(L):
        where = np.flatnonzero(sections == k)
        Wk = W[where][:, where]
        s0, s1, s2 = weight_moments(Wk)
        m = s[k] / n_l[k]
        num = xwx[k] - m * xr[k] - m * xc[k] + m * m * s0
        den = ss[k] - n_l[k] * m * m
        nums.append(num)
        dens.append(np.where(constant[k], 0.0, den))
        keep = ~constant[k]
        if s0 <= 0 or n_l[k] < 3 or not keep.any():
            continue
        nk = n_l[k]
        with np.errstate(divide="ignore", invalid="ignore"):
            score = nk / s0 * num[keep] / den[keep]
        expected = -1.0 / (nk - 1)
        var = (nk * nk * s1 - nk * s2 + 3 * s0 * s0) / ((nk - 1) * (nk + 1) * s0 * s0) - expected ** 2
        z = (score - expected) / np.sqrt(var)
        p = np.where(z > 0, stats.norm.sf(z), stats.norm.cdf(z))
        parts.append(pd.DataFrame({"library_id": names[k], "n_obs": int(nk), "gene": genes[keep], "I": score,
                                   "z": z, "pval": p, "pval_adj": bh_adjust(p)}))
    keep = ~constant.all(axis=0)
    s0 = float(W.sum())
    with np.errstate(divide="ignore", invalid="ignore"):
        score = n / s0 * np.sum(nums, axis=0)[keep] / np.sum(dens, axis=0)[keep]
    expected, var = centred_moments(W, sections, L)
    z = (score - expected) / np.sqrt(var)
    p = np.where(z > 0, stats.norm.sf(z), stats.norm.cdf(z))
    table = pd.DataFrame({"gene": genes[keep], "I": score, "expected_I": expected, "var_I": var, "z": z,
                          "pval": p, "pval_adj": bh_adjust(p)})
    table = table.sort_values("I", ascending=False, kind="mergesort").reset_index(drop=True)
    by = (pd.concat(parts, ignore_index=True) if parts else
          pd.DataFrame(columns=["library_id", "n_obs", "gene", "I", "z", "pval", "pval_adj"]))
    return table, by, int((~keep).sum())


def morans_i_sections(W, M, genes, sections, names, order=None):
    """Moran's I per gene pooled over several sections, each centred on its own mean, and
    each section's own (``section_tables``)."""
    return section_tables(W, section_sums(W, M, sections, len(names), order), sections, names, genes)


REPLAY_CODE = (knn_edges, spatial_weights, weight_moments, moran_sums, bh_adjust, spatial_order, moran_table,
               morans_i, section_sums, centred_moments, section_tables, morans_i_sections)


def moran_sums_by_genes(W, read, n_genes, block=GENE_BLOCK, progress=None):
    """``moran_sums`` of a matrix read a block of genes at a time: ``read(a, b)`` gives genes
    a..b as a sparse spots x genes matrix (a large dataset's gene store). Stays sparse.
    ``progress(done, total)`` after each block (it may raise to stop the run)."""
    W = sp.csr_matrix(W)
    r = np.asarray(W.sum(axis=1)).ravel()
    c = np.asarray(W.sum(axis=0)).ravel()
    xwx, s, ss, xr, xc = (np.zeros(n_genes) for _ in range(5))
    constant = np.zeros(n_genes, dtype=bool)
    for a in range(0, n_genes, block):
        b = min(a + block, n_genes)
        B = sp.csc_matrix(read(a, b), dtype=np.float64)
        xwx[a:b] = np.asarray(B.multiply(W @ B).sum(axis=0)).ravel()
        s[a:b] = np.asarray(B.sum(axis=0)).ravel()
        ss[a:b] = np.asarray(B.multiply(B).sum(axis=0)).ravel()
        xr[a:b] = r @ B
        xc[a:b] = c @ B
        constant[a:b] = B.max(axis=0).toarray().ravel() == B.min(axis=0).toarray().ravel()
        if progress is not None:
            progress(b, n_genes)
    return xwx, s, ss, xr, xc, constant


def section_sums_by_genes(W, read, n_genes, sections, n_sections, block=GENE_BLOCK, progress=None):
    """``section_sums`` of a matrix read a block of genes at a time (``moran_sums_by_genes``)."""
    W = sp.csr_matrix(W)
    sections = np.asarray(sections, dtype=np.int64)
    n = W.shape[0]
    P = sp.csr_matrix((np.ones(n), (sections, np.arange(n))), shape=(n_sections, n))
    r = np.asarray(W.sum(axis=1)).ravel()
    c = np.asarray(W.sum(axis=0)).ravel()
    Pr, Pc = sp.csr_matrix(P.multiply(r[None, :])), sp.csr_matrix(P.multiply(c[None, :]))
    xwx, s, ss, xr, xc = (np.zeros((n_sections, n_genes)) for _ in range(5))
    constant = np.zeros((n_sections, n_genes), dtype=bool)
    where = [np.flatnonzero(sections == k) for k in range(n_sections)]
    for a in range(0, n_genes, block):
        b = min(a + block, n_genes)
        B = sp.csc_matrix(read(a, b), dtype=np.float64)
        xwx[:, a:b] = (P @ sp.csc_matrix(B.multiply(W @ B))).toarray()
        s[:, a:b] = (P @ B).toarray()
        ss[:, a:b] = (P @ sp.csc_matrix(B.multiply(B))).toarray()
        xr[:, a:b] = (Pr @ B).toarray()
        xc[:, a:b] = (Pc @ B).toarray()
        for k, rows in enumerate(where):
            part = sp.csc_matrix(B[rows])
            constant[k, a:b] = part.max(axis=0).toarray().ravel() == part.min(axis=0).toarray().ravel()
        if progress is not None:
            progress(b, n_genes)
    return xwx, s, ss, xr, xc, constant


# ---------------------------------------------------------------------------
# On a dataset
# ---------------------------------------------------------------------------


@dataclass
class Result:
    """One Moran's I run (results store kind ``spatial_autocorr``)."""

    id: str
    dataset_id: str
    n_obs: int
    n_genes: int
    neighbors: dict[str, Any]
    libraries: list[str]
    genes_choice: str  # hvg | all | list
    matrix_source: str
    table: pd.DataFrame
    n_constant: int = 0
    notes: list[str] = field(default_factory=list)
    unit: str = "spot"
    elapsed_seconds: float = 0.0
    created_at: float = 0.0
    cells: str = ""  # results_store.cells_hash of the dataset it ran on
    # "global": one mean (one section, squidpy's I); "section": several sections pooled, each
    # centred on its own mean, with each section's own values in ``by_library``.
    centring: str = "global"
    sections: list[dict[str, Any]] = field(default_factory=list)  # library_id, n_obs (several sections)
    by_library: pd.DataFrame | None = None  # library_id, n_obs, gene, I, z, pval, pval_adj
    section_run: bool = False  # run on one section of the dataset (``library_id``)


def has_spatial(adata) -> bool:
    try:
        coords = adata.obsm["spatial"]
    except KeyError:
        return False
    return getattr(coords, "ndim", 0) == 2 and coords.shape[1] >= 2


def is_grid(adata) -> bool:
    """Visium's or Visium HD's grid (squidpy coord_type "grid"), else generic kNN. Visium HD's
    segmented cells (unit "cell") are on no grid."""
    info = adata.uns.get("scview_spatial")
    if isinstance(info, dict) and info.get("technology"):
        return str(info["technology"]) in ("visium", "visium_hd") and info.get("unit") != "cell"
    return "spatial" in adata.uns


def neighbours_of(adata) -> int:
    """Each point's neighbours: 4 on Visium HD's square grid, else 6."""
    info = adata.uns.get("scview_spatial")
    bins = isinstance(info, dict) and info.get("technology") == "visium_hd" and info.get("unit") != "cell"
    return HD_NEIGHS if bins else N_NEIGHS


def unit_of(adata) -> str:
    info = adata.uns.get("scview_spatial")
    if isinstance(info, dict) and info.get("unit") in ("spot", "bin", "cell"):
        return str(info["unit"])
    return "spot" if is_grid(adata) else "cell"


def library_labels(adata) -> np.ndarray:
    if LIBRARY_COLUMN in adata.obs.columns:
        s = adata.obs[LIBRARY_COLUMN]
        return s.astype(str).where(s.notna(), "").to_numpy()
    return np.full(adata.n_obs, "", dtype=object)


def values_matrix(adata, layer: str | None, rows: np.ndarray | None = None):
    """(spots x genes matrix, gene names, where from); ``rows``: only those spots (ascending
    positions; None: all). A backed dataset's matrix is read: only ``rows`` of it, with the
    room held by the endpoint (``read_bytes``)."""
    if layer:
        if layer not in adata.layers:
            raise KeyError(f"Layer '{layer}' not found.")
        M = adata.layers[layer]
        return (M if rows is None else M[rows]), adata.var_names, f"layer:{layer}"
    from scview.core.activity import lognorm_source

    try:
        choice = lognorm_source(adata)
    except ValueError as e:
        raise SpatialAutocorrError(str(e)) from None
    return (choice.whole() if rows is None else choice.rows(rows)), choice.names, choice.source


def read_bytes(adaptor, library_id: str | None, layer: str | None) -> int:
    """Memory ``values_matrix`` adds reading a backed dataset's matrix (the section's spots):
    0 for a large dataset (its gene store), a layer (in memory) or a dataset in memory."""
    from scview.core.activity import lognorm_bytes

    if layer or large_mode.is_large(adaptor):
        return 0
    n = int((library_labels(adaptor.adata) == str(library_id)).sum()) if library_id else None
    try:
        return lognorm_bytes(adaptor, n)
    except ValueError:  # nothing to test: the run says so
        return 0


def select_genes(adata, names, choice) -> tuple[np.ndarray, str, list[str]]:
    """(column indices into the matrix, the choice actually made, notes)."""
    names = pd.Index(names)
    notes: list[str] = []
    if isinstance(choice, (list, tuple)):
        wanted = [str(g) for g in choice][:MAX_LIST_GENES]
        idx = names.get_indexer(pd.Index(wanted).unique())
        missing = int((idx < 0).sum())
        idx = idx[idx >= 0]
        if missing:
            notes.append(f"{missing} of the genes asked for are not in the data.")
        if not len(idx):
            raise SpatialAutocorrError("None of these genes are in the data.")
        return np.asarray(idx), "list", notes
    if choice not in GENE_CHOICES:
        raise SpatialAutocorrError(f"Unknown gene choice '{choice}': use hvg, all or a list of genes.")
    if choice == "hvg":
        hv = adata.var.get("highly_variable")
        if hv is not None and bool(np.asarray(hv, dtype=bool).any()):
            flagged = adata.var_names[np.asarray(hv, dtype=bool)]
            idx = names.get_indexer(flagged)
            idx = idx[idx >= 0]
            if len(idx):
                return np.sort(idx), "hvg", notes
        notes.append("The dataset has no variable genes marked, so every gene was tested.")
    return np.arange(len(names)), "all", notes


def estimate_seconds(n_obs: int, n_genes: int, large: bool = False) -> float:
    rate = LARGE_SECONDS_PER_VALUE if large else SECONDS_PER_VALUE
    return round(max(0.2, rate * n_obs * n_genes), 1)


def need_bytes(adaptor) -> int:
    """Memory a run holds beyond the open dataset: the neighbour search and the weights, and a
    block of genes (a large dataset) or the chosen genes' copy and a block of rows."""
    adata = adaptor.adata
    n = int(adata.n_obs)
    if large_mode.is_large(adaptor):
        return NEED_PER_OBS * n + NEED_LARGE_BLOCK
    x = adata.raw.X if adata.raw is not None else adata.X
    nnz = int(getattr(x, "nnz", 0) or 0)
    return NEED_PER_OBS * n + 3 * BLOCK_BYTES + 12 * nnz // 4


def _large_values(adata):
    """A large dataset's log-normalised X (read from its gene store) and gene names."""
    return adata.X, adata.var_names, "log-normalised X"


def run(adata, *, genes: Any = "hvg", library_id: str | None = None, layer: str | None = None,
        dataset_id: str = "", progress=None) -> Result:
    """Moran's I for the chosen genes on all sections (or one). ``progress(done, total)``: genes
    done, for a large dataset's blocks of genes (a run in memory takes seconds)."""
    if not has_spatial(adata):
        raise SpatialAutocorrError("This dataset has no spatial coordinates (obsm['spatial']).")
    t0 = time.time()
    libs = library_labels(adata)
    rows = np.arange(adata.n_obs)
    if library_id:
        rows = np.flatnonzero(libs == str(library_id))
        if not len(rows):
            raise SpatialAutocorrError(f"No spots in section '{library_id}'.")
    if len(rows) < 3:
        raise SpatialAutocorrError("Moran's I needs at least 3 spots.")
    coords = np.asarray(adata.obsm["spatial"])[rows, :2]
    if not np.isfinite(coords).all():
        raise SpatialAutocorrError("Some spatial coordinates are missing (NaN).")
    large = large_mode.is_large(adata)
    if large and layer:
        raise SpatialAutocorrError("A large dataset's layers stay on disk: run this on its expression values.")
    some = rows if len(rows) < adata.n_obs else None
    M, names, source = _large_values(adata) if large else values_matrix(adata, layer, some)
    idx, choice, notes = select_genes(adata, names, genes)
    if not large and len(idx) < M.shape[1]:
        M = M[:, idx]
    grid = is_grid(adata)
    k = neighbours_of(adata)
    W = spatial_weights(coords, libs[rows], grid, k)
    if W.nnz == 0:
        raise SpatialAutocorrError(f"No {unit_of(adata)} has a neighbour: the positions are too far apart.")
    section_names = [str(x) for x in pd.unique(libs[rows])]
    pooled = len(section_names) > 1
    codes = pd.Index(section_names).get_indexer(libs[rows]) if pooled else None
    gene_names = np.asarray(names)[idx]
    by_library = None
    if large:
        whole = len(rows) == adata.n_obs

        def read(a, b):
            block = M.columns(idx[a:b])
            return block if whole else block[rows]

        if pooled:
            table, by_library, n_constant = section_tables(
                W, section_sums_by_genes(W, read, len(idx), codes, len(section_names), progress=progress),
                codes, section_names, gene_names)
        else:
            table, n_constant = moran_table(W, moran_sums_by_genes(W, read, len(idx), progress=progress), gene_names)
    elif pooled:
        table, by_library, n_constant = morans_i_sections(W, M, gene_names, codes, section_names,
                                                          spatial_order(coords, libs[rows]))
    else:
        table, n_constant = morans_i(W, M, gene_names, spatial_order(coords, libs[rows]))
    unit = unit_of(adata)
    if n_constant:
        where = " of every section" if pooled else ""
        notes.append(f"{n_constant:,} gene(s) with the same value in every {unit}{where} were left out.")
    sections: list[dict[str, Any]] = []
    if pooled:
        sizes = np.bincount(codes, minlength=len(section_names))
        sections = [{"library_id": name, "n_obs": int(sizes[i])} for i, name in enumerate(section_names)]
        notes.append(f"{len(section_names)} sections pooled, each centred on its own mean: a gene that is only higher "
                     "in one section than in another does not count as spatial pattern. Each section's own I is "
                     "listed beside it.")
    isolated = int((np.diff(W.indptr) == 0).sum())
    if isolated:
        notes.append(f"{isolated:,} {unit}(s) have no neighbour within reach; they count in the mean only.")
    off = off_tissue.note(adata, unit)
    if off:  # imported before off-tissue spots were left out at ingest
        notes.append(off)
    return Result(
        id=uuid.uuid4().hex[:12], dataset_id=dataset_id, n_obs=int(len(rows)), n_genes=int(len(table)),
        neighbors={"kind": "grid" if grid else "knn", "k": k},
        libraries=[str(x) for x in pd.unique(libs[rows])], genes_choice=choice, matrix_source=source,
        table=table, n_constant=n_constant, notes=notes, unit=unit,
        elapsed_seconds=round(time.time() - t0, 2), created_at=time.time(),
        cells=results_store.cells_hash(adata), centring="section" if pooled else "global",
        sections=sections, by_library=by_library, section_run=bool(library_id),
    )


def uns_frame(r: Result) -> pd.DataFrame:
    """squidpy's uns["moranI"] layout: genes as the index, I, pval_norm, var_norm,
    pval_norm_fdr_bh."""
    t = r.table
    return pd.DataFrame({"I": t["I"].to_numpy(), "pval_norm": t["pval"].to_numpy(),
                         "var_norm": t["var_I"].to_numpy(), "pval_norm_fdr_bh": t["pval_adj"].to_numpy()},
                        index=pd.Index(t["gene"].astype(str).to_numpy()))


def record(adata, r: Result, genes: Any, library_id: str | None, layer: str | None) -> None:
    """The run in provenance (History, replay script) and squidpy's uns["moranI"]."""
    from scview.core import provenance

    adata.uns[UNS_KEY] = uns_frame(r)
    provenance.record_step(
        adata, step=STEP, tool="scview.spatial_autocorr (= squidpy 1.6 spatial_neighbors + spatial_autocorr)",
        params={"genes": genes if isinstance(genes, str) else [str(g) for g in genes],
                "genes_used": r.genes_choice, "library_id": library_id, "layer": layer,
                "matrix_source": r.matrix_source, "neighbors": r.neighbors["kind"], "n_neighs": r.neighbors["k"],
                "n_obs": r.n_obs, "n_genes": r.n_genes, "n_constant": r.n_constant, "centring": r.centring},
        resources={"seconds": r.elapsed_seconds},
    )


# ---------------------------------------------------------------------------
# Kept results
# ---------------------------------------------------------------------------

_last: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_lock = threading.Lock()


ALL_SECTIONS = "all"


def section_of(r: Result) -> str:
    """The section a run is about: its library_id, or "all" (every section of the dataset)."""
    return r.libraries[0] if len(r.libraries) == 1 and r.libraries[0] and r.section_run else ALL_SECTIONS


def result_key(r: Result) -> str:
    """A run's identity in the results store: its section and its genes (and layer), so a run on
    section B does not supersede one on section A, nor all genes the variable ones."""
    genes = r.genes_choice
    if genes == "list":
        genes = "list:" + hashlib.sha256("\n".join(sorted(map(str, r.table["gene"]))).encode()).hexdigest()[:12]
    layer = r.matrix_source if r.matrix_source.startswith("layer:") else ""
    return "|".join(x for x in (section_of(r), genes, layer) if x)


def remember(adaptor, r: Result) -> None:
    """Keep the run for GET and the co-pilot, in memory and in the results store."""
    try:
        with _lock:
            _last[adaptor] = r
    except TypeError:  # a test double
        pass
    section = section_of(r)
    where = f" in section {section}" if section != ALL_SECTIONS else (
        f", {len(r.libraries)} sections" if len(r.libraries) > 1 else "")
    results_store.keep(adaptor, KIND, r, result_id=r.id, key=result_key(r),
                       params={"library_id": section, "genes": r.genes_choice},
                       label=f"Spatially variable genes (Moran's I, {r.n_genes:,} genes){where}")


def last_for_section(adaptor, library_id: str) -> Result | None:
    """The newest kept run on one section (or ``ALL_SECTIONS``) on the spots the dataset holds now."""
    r = last(adaptor)
    if r is not None and section_of(r) == library_id:
        return r
    dataset_id = results_store.dataset_id_of(adaptor)
    if dataset_id is None:
        return None
    try:
        cells = results_store.cells_hash(adaptor.adata)
        for e in results_store.entries(dataset_id, KIND):
            if (e.get("params") or {}).get("library_id") != library_id:
                continue
            got = results_store.recall(dataset_id, e["id"], KIND)
            if got is not None and got.cells == cells:
                return got
    except Exception:  # a test double, an unreadable result
        return None
    return None


def last(adaptor) -> Result | None:
    """The last run, while the dataset holds the same spots: a step recorded since
    (labels, a kept layout) does not hide it, filtering or a re-import does."""
    try:
        cells = results_store.cells_hash(adaptor.adata)
        r = results_store.remembered(_last, _lock, adaptor, KIND)
        if r is None:
            r = results_store.latest(adaptor, KIND, current_only=False)
            if r is not None and r.cells == cells:
                with _lock:
                    _last.setdefault(adaptor, r)
    except Exception:  # a test double, an unreadable result
        return None
    return r if r is not None and r.cells == cells else None


def summary_counts(r: Result) -> dict[str, int]:
    t = r.table
    sig = t["pval_adj"] < FACT_PADJ
    return {"n_significant": int((sig & (t["I"] > 0)).sum()), "n_dispersed": int((sig & (t["I"] < 0)).sum())}


def grounding_lines(adaptor) -> tuple[list[str], str | None]:
    """For the co-pilot: the last run's top spatially variable genes."""
    r = last(adaptor)
    if r is None or r.table.empty:
        return [], None
    t = r.table
    c = summary_counts(r)
    top = t[(t["I"] > 0) & (t["pval_adj"] < FACT_PADJ)].head(TOP_FACT_GENES)
    sections = (f"{len(r.libraries)} sections, neighbours within each"
                + (", each centred on its own mean" if r.centring == "section" else "")
                if len(r.libraries) > 1 else "one section")
    detail = (f"Moran's I on {r.n_obs:,} {r.unit}s ({sections}; {r.neighbors['kind']} neighbours, k "
              f"{r.neighbors['k']}), {r.n_genes:,} genes: {c['n_significant']:,} spatially variable "
              f"(I > 0, adjusted p < {FACT_PADJ:g})")
    lines = [f"\n## Spatially variable genes (last run; Moran's I) [{TAG}]", f"- {detail} [{TAG}]."]
    if len(top):
        lines.append("- Highest Moran's I: " + ", ".join(
            f"{row.gene} (I {row.I:.3f}, adjusted p {row.pval_adj:.2g})" for row in top.itertuples()) + ".")
    else:
        lines.append(f"- No gene has I > 0 with adjusted p < {FACT_PADJ:g}.")
    by = r.by_library
    if r.centring == "section" and by is not None and len(top):
        lines.append(f"- The pooled I is within-section (each section centred on its own mean, so a section that is "
                     f"only higher overall does not count as pattern). Each section's own I [{TAG}]:")
        for g in top["gene"].head(3):
            one = by[by["gene"] == g]
            lines.append(f"  - {g}: " + ", ".join(f"{row.library_id} {row.I:.3f}" for row in one.itertuples()) + ".")
    lines.append(f"- Moran's I is 1 when neighbouring {r.unit}s have alike values, about 0 when the pattern "
                 "is random and below 0 when neighbours differ; it does not say which cell types drive it.")
    return lines, detail
