"""Batch-correction methods on the PCA: Harmony, Scanorama and BBKNN.

The pipeline's ``batch_correction`` step runs one of ``METHODS`` (``PipelineParams.
batch_method``):

* ``harmony``: corrected PCs in obsm ``X_pca_harmony`` (harmonypy).
* ``scanorama``: corrected PCs in obsm ``X_scanorama``, as
  ``sc.external.pp.scanorama_integrate(adata, key, approx=False)`` makes them
  (``scanorama_correct`` below).
* ``bbknn``: no embedding. The neighbour GRAPH is built batch by batch (each cell's
  nearest neighbours in every batch, then UMAP connectivities and trimming), as
  ``bbknn.bbknn(adata, batch_key, computation="cKDTree")`` makes it
  (``bbknn_graph``); it replaces the neighbours step, which then keeps it.

Each method removes what the others left, so one corrected representation is ever
held, and the neighbours step reads it (``neighbour_rep``).

Scanorama and BBKNN are ports, not dependencies: scanorama pulls fbpca (BSD with
Facebook's patent-grant clause) and annoy, and bbknn annoy (a C++ build on every
image architecture), for code paths neither port uses. Both are MIT licensed:

    Scanorama: Copyright (c) 2018 Brian Hie (MIT License). Hie, Bryson & Berger,
    "Efficient integration of heterogeneous single-cell transcriptomes using
    Scanorama", Nat Biotechnol 37:685 (2019). Ported from scanorama 1.7.4
    (``find_alignments_table``, ``fill_table``, ``assemble``, ``transform``,
    ``batch_bias``) with exact neighbours (``approx=False``).

    BBKNN: Copyright (c) 2018 The Python Packaging Authority (bbknn's licence file;
    MIT License). Polanski et al., "BBKNN: fast batch alignment of single cell
    transcriptomes", Bioinformatics 36:964 (2020). Ported from bbknn 1.6.0
    (``matrix.get_graph``, ``matrix.bbknn``, ``matrix.trimming``) with the exact
    cKDTree search.

    Permission is hereby granted, free of charge, to any person obtaining a copy of
    this software and associated documentation files (the "Software"), to deal in the
    Software without restriction, including without limitation the rights to use,
    copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
    Software, and to permit persons to whom the Software is furnished to do so,
    subject to the following conditions: The above copyright notice and this
    permission notice shall be included in all copies or substantial portions of the
    Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.

Agreement with the packages (Kang, 13,836 cells, ``stim`` as batch, 50 PCs, 2026-10-24):
Scanorama max |difference| vs scanorama 1.7.4 via ``sc.external.pp.scanorama_integrate(
approx=False)`` 1.8e-6 (values up to 28; float summation order); BBKNN distances and
connectivities identical to bbknn 1.6.0 ``computation="cKDTree"`` (max |diff| 0).

Scanorama's Gaussian kernel (gamma = sigma / 2 = 7.5) on scanpy's PCs underflows to 0
for cells far from every matched cell; those cells keep their position (as in the
package).
"""

from __future__ import annotations

import logging
import operator
import warnings
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

logger = logging.getLogger(__name__)

HARMONY = "harmony"
SCANORAMA = "scanorama"
BBKNN = "bbknn"
METHODS: tuple[str, ...] = (HARMONY, SCANORAMA, BBKNN)

# obsm key of each embedding method's corrected PCs (BBKNN makes a graph instead).
REPRESENTATION = {HARMONY: "X_pca_harmony", SCANORAMA: "X_scanorama"}
CORRECTED_KEYS: tuple[str, ...] = tuple(REPRESENTATION.values())

LABELS = {HARMONY: "Harmony", SCANORAMA: "Scanorama", BBKNN: "BBKNN", "none": "None"}

# Scanorama's defaults (sc.external.pp.scanorama_integrate / scanorama.assemble).
SCANORAMA_KNN = 20
SCANORAMA_SIGMA = 15.0
SCANORAMA_ALPHA = 0.10
SCANORAMA_BATCH_SIZE = 5000
# Rows of the kernel computed at once (the package does all a dataset's cells at once:
# cells x 5,000 float64, 4 GB at 100k cells); the sums are the same.
_KERNEL_ROWS = 4096

# BBKNN: its graph is trimmed to this many times the neighbours per cell (bbknn's default).
BBKNN_TRIM_FACTOR = 10
N_WORKERS = 2  # the laptop rule: never more than 2 cores


class BatchMethodError(ValueError):
    """A batch-correction method that cannot run on these cells (named in the message)."""


def check_method(value: Any) -> str:
    """``PipelineParams.batch_method`` as one of ``METHODS``; a ValueError naming them."""
    method = str(value if value is not None else HARMONY).strip().lower()
    if method not in METHODS:
        raise ValueError(f"Unknown batch correction method '{value}'; choose from "
                         f"{', '.join(METHODS)}.")
    return method


def is_bbknn_graph(adata: ad.AnnData) -> bool:
    """The neighbour graph held is a BBKNN graph (its params carry a ``bbknn`` block)."""
    params = (adata.uns.get("neighbors") or {}).get("params") or {}
    return isinstance(params, dict) and "bbknn" in params


def neighbour_rep(adata: ad.AnnData) -> str:
    """The representation a neighbour graph is built from: corrected PCs, else X_pca."""
    return next((k for k in CORRECTED_KEYS if k in adata.obsm), "X_pca")


def clear(adata: ad.AnnData, keep: str | None = None) -> None:
    """Drop every method's result but ``keep``'s: corrected PCs, and the mark that the
    graph is BBKNN's (the graph itself stays until the neighbours step replaces it)."""
    for method, key in REPRESENTATION.items():
        if method != keep:
            adata.obsm.pop(key, None)
    if keep != BBKNN and is_bbknn_graph(adata):
        adata.uns["neighbors"]["params"].pop("bbknn", None)


def _batches(obs: pd.DataFrame, key: str) -> np.ndarray:
    if not key:
        raise BatchMethodError("batch_key parameter is required for batch correction.")
    if key not in obs.columns:
        raise BatchMethodError(f"Batch key '{key}' not found in obs columns.")
    col = obs[key]
    if col.isna().any():
        raise BatchMethodError(f"Batch key '{key}' has cells with no value; give every cell "
                               "a batch first.")
    return col.astype(str).to_numpy()


def _groups_in_order(batches: np.ndarray) -> list[np.ndarray]:
    """Cell positions of each batch, batches in order of first appearance (the order
    scanpy's wrapper takes them in, when the batches are contiguous)."""
    _, first, inverse = np.unique(batches, return_index=True, return_inverse=True)
    return [np.flatnonzero(inverse == code) for code in np.argsort(first, kind="stable")]


# ---------------------------------------------------------------------------
# Harmony
# ---------------------------------------------------------------------------


def harmony_correct(X: np.ndarray, obs: pd.DataFrame, key: str) -> np.ndarray:
    """Harmony's corrected PCs, cells x PCs.

    harmonypy directly rather than scanpy's harmony_integrate: scanpy transposes
    Harmony's output unconditionally, which breaks with harmonypy >= 2.0 (Z_corr is
    already cells x PCs). Oriented here, so either version works."""
    import harmonypy

    _batches(obs, key)
    ho = harmonypy.run_harmony(X, obs, [key])
    Z = np.asarray(ho.Z_corr)
    n = X.shape[0]
    if Z.shape[0] != n and Z.ndim == 2 and Z.shape[1] == n:
        Z = Z.T  # old harmonypy returns PCs x cells
    if Z.shape[0] != n:
        raise ValueError(f"Harmony output shape {Z.shape} does not match n_cells={n}.")
    return np.ascontiguousarray(Z)


# ---------------------------------------------------------------------------
# Scanorama (port of scanorama 1.7.4, exact neighbours)
# ---------------------------------------------------------------------------


def _l2_rows(X: np.ndarray) -> np.ndarray:
    from sklearn.preprocessing import normalize

    return normalize(X, axis=1)


def _fill_table(table: dict, i: int, curr: np.ndarray, datasets: list[np.ndarray],
                base_ds: int, knn: int) -> None:
    """scanorama.fill_table: the knn nearest cells (Manhattan, on unit rows) of each cell
    of dataset ``i`` among ``datasets`` pooled, as (cell in i, cell in j) pairs."""
    from sklearn.neighbors import NearestNeighbors

    ref = np.concatenate(datasets)
    ind = NearestNeighbors(n_neighbors=knn, p=1, n_jobs=N_WORKERS).fit(ref) \
        .kneighbors(curr, return_distance=False)
    starts = np.cumsum([0] + [d.shape[0] for d in datasets])
    d = np.repeat(np.arange(curr.shape[0]), knn)
    r = ind.ravel()
    which = np.searchsorted(starts, r, side="right") - 1
    for off in range(len(datasets)):
        m = which == off
        if m.any():
            table[(i, base_ds + off)] = (d[m], r[m] - starts[off])


def _alignments(datasets: list[np.ndarray], knn: int, alpha: float):
    """scanorama.find_alignments: dataset pairs by the share of cells with a mutual
    nearest neighbour (above ``alpha``, best first), and each pair's matches."""
    norm = [_l2_rows(ds) for ds in datasets]
    table: dict = {}
    for i in range(len(norm)):
        if i > 0:
            _fill_table(table, i, norm[i], norm[:i], 0, knn)
        if i + 1 < len(norm):
            _fill_table(table, i, norm[i], norm[i + 1:], i + 1, knn)
    matches: dict = {}
    score: dict = {}
    for i in range(len(norm)):
        for j in range(i + 1, len(norm)):
            if (i, j) not in table or (j, i) not in table:
                continue
            n_j = norm[j].shape[0]
            a, b = table[(i, j)]
            b2, a2 = table[(j, i)]
            mutual = np.intersect1d(a.astype(np.int64) * n_j + b, a2.astype(np.int64) * n_j + b2)
            pairs = np.column_stack([mutual // n_j, mutual % n_j])
            matches[(i, j)] = pairs
            score[(i, j)] = max(np.unique(pairs[:, 0]).size / norm[i].shape[0],
                                np.unique(pairs[:, 1]).size / n_j)
    ranked = reversed(sorted(score.items(), key=operator.itemgetter(1)))
    return [ij for ij, val in ranked if val > alpha], matches


def _batch_bias(curr: np.ndarray, match: np.ndarray, bias: np.ndarray, batch_size: int,
                sigma: float) -> np.ndarray:
    """scanorama.batch_bias: each cell's Gaussian-weighted mean of the matched cells'
    translation vectors, by blocks of matched cells (and of cells, for memory)."""
    from sklearn.metrics.pairwise import rbf_kernel

    avg = np.zeros(curr.shape)
    denom = np.zeros(curr.shape[0])
    for base in range(0, match.shape[0], batch_size):
        m = match[base:base + batch_size]
        bb = bias[base:base + batch_size]
        for r0 in range(0, curr.shape[0], _KERNEL_ROWS):
            w = rbf_kernel(curr[r0:r0 + _KERNEL_ROWS], m, gamma=0.5 * sigma)
            avg[r0:r0 + _KERNEL_ROWS] += w @ bb
            denom[r0:r0 + _KERNEL_ROWS] += w.sum(axis=1)
    denom[denom == 0.0] = 1.0
    return avg / denom[:, None]


def _transform(curr: np.ndarray, ref: np.ndarray, ds_ind, ref_ind, sigma: float,
               batch_size: int) -> np.ndarray:
    ds_ind = np.asarray(ds_ind, dtype=np.int64)
    ref_ind = np.asarray(ref_ind, dtype=np.int64)
    match = curr[ds_ind]
    bias = ref[ref_ind] - match
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=RuntimeWarning)
        try:
            return _batch_bias(curr, match, bias, batch_size, sigma)
        except RuntimeWarning:  # as the package: refuse to correct this one
            logger.warning("Scanorama: oversmoothing detected, this dataset left uncorrected")
            return np.zeros(curr.shape)


def _matches_into(match: list, pairs: np.ndarray, flip: bool, base_a: int, base_b: int) -> None:
    if pairs.size:
        a, b = (pairs[:, 1], pairs[:, 0]) if flip else (pairs[:, 0], pairs[:, 1])
        match.append(np.column_stack([a + base_a, b + base_b]))


def _pairs(match: list) -> tuple[np.ndarray, np.ndarray]:
    if not match:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    m = np.concatenate(match)
    return m[:, 0], m[:, 1]


def _assemble(datasets: list[np.ndarray], knn: int, sigma: float, alpha: float,
              batch_size: int) -> list[np.ndarray]:
    """scanorama.assemble on low-dimensional datasets (no expression matrices)."""
    if len(datasets) == 1:
        return datasets
    alignments, matches = _alignments(datasets, knn, alpha)
    sizes = [d.shape[0] for d in datasets]
    assembled: dict[int, int] = {}
    panoramas: list[list[int]] = []
    for i, j in alignments:
        assembled[i] = assembled.get(i, 0) + 1
        assembled[j] = assembled.get(j, 0) + 1
        if assembled[i] > 3 and assembled[j] > 3:
            continue
        pi = [p for p in panoramas if i in p]
        pj = [p for p in panoramas if j in p]
        if not pi and not pj:
            if sizes[i] < sizes[j]:
                i, j = j, i
            panoramas.append([i])
            pi = [panoramas[-1]]

        if not pi or not pj:  # map one dataset onto a panorama
            ds, pano = (i, pj[0]) if not pi else (j, pi[0])
            curr = datasets[ds]
            ref = np.concatenate([datasets[p] for p in pano])
            match: list = []
            base = 0
            for p in pano:
                if ds < p and (ds, p) in matches:
                    _matches_into(match, matches[(ds, p)], False, 0, base)
                elif ds > p and (p, ds) in matches:
                    _matches_into(match, matches[(p, ds)], True, 0, base)
                base += sizes[p]
            ds_ind, ref_ind = _pairs(match)
            datasets[ds] = curr + _transform(curr, ref, ds_ind, ref_ind, sigma, batch_size)
            pano.append(ds)
            continue

        # Merge two panoramas: move all of panorama i
        curr = np.concatenate([datasets[p] for p in pi[0]])
        ref = np.concatenate([datasets[p] for p in pj[0]])
        base_i = 0
        for p in pi[0]:
            if p == i:
                break
            base_i += sizes[p]
        base_j = 0
        for p in pj[0]:
            if p == j:
                break
            base_j += sizes[p]
        match = []
        base = 0
        for p in pi[0]:
            if p == i and j < p and (j, p) in matches:
                _matches_into(match, matches[(j, p)], True, base, base_j)
            elif p == i and j > p and (p, j) in matches:
                _matches_into(match, matches[(p, j)], False, base, base_j)
            base += sizes[p]
        base = 0
        for p in pj[0]:
            if p == j and i < p and (i, p) in matches:
                _matches_into(match, matches[(i, p)], False, base_i, base)
            elif p == j and i > p and (p, i) in matches:
                _matches_into(match, matches[(p, i)], True, base_i, base)
            base += sizes[p]
        ds_ind, ref_ind = _pairs(match)
        curr = curr + _transform(curr, ref, ds_ind, ref_ind, sigma, batch_size)
        base = 0
        for p in pi[0]:
            datasets[p] = curr[base:base + sizes[p]]
            base += sizes[p]
        if pi[0] is not pj[0]:
            pi[0] += pj[0]
            panoramas.remove(pj[0])
    return datasets


def scanorama_correct(X: np.ndarray, batches: np.ndarray, *, knn: int = SCANORAMA_KNN,
                      sigma: float = SCANORAMA_SIGMA, alpha: float = SCANORAMA_ALPHA,
                      batch_size: int = SCANORAMA_BATCH_SIZE) -> np.ndarray:
    """Scanorama's corrected embedding of ``X`` (cells x PCs), rows in ``X``'s order.

    The package needs each batch's cells together (scanpy's wrapper refuses
    non-contiguous batches); here each batch's rows are gathered, corrected and put
    back where they were, so the user's cell order never changes."""
    groups = _groups_in_order(np.asarray(batches).astype(str))
    small = [g.size for g in groups if g.size <= knn]
    if len(groups) < 2:
        raise BatchMethodError("Scanorama needs at least two batches.")
    if small:
        raise BatchMethodError(f"Scanorama needs more than {knn} cells in every batch; the "
                               f"smallest has {min(small)}.")
    X = np.asarray(X)
    datasets = _assemble([X[g] for g in groups], knn, sigma, alpha, batch_size)
    out = np.empty((X.shape[0], X.shape[1]), dtype=np.result_type(X.dtype, np.float64))
    for g, ds in zip(groups, datasets):
        out[g] = ds
    return out


# ---------------------------------------------------------------------------
# BBKNN (port of bbknn 1.6.0, exact cKDTree search)
# ---------------------------------------------------------------------------


def bbknn_within(n_neighbors: int, n_batches: int) -> int:
    """Neighbours taken from each batch so a cell has about ``n_neighbors`` in all (the
    same graph size as the other methods' neighbour step); at least 1."""
    return max(1, int(round(n_neighbors / max(1, n_batches))))


def _trim(cnts: sp.csr_matrix, trim: int) -> sp.csr_matrix:
    """bbknn.matrix.trimming: keep each cell's ``trim`` strongest connections, rows then
    columns with the same thresholds."""
    vals = np.zeros(cnts.shape[0])
    for i in range(cnts.shape[0]):
        row = cnts.data[cnts.indptr[i]:cnts.indptr[i + 1]]
        if row.shape[0] > trim:
            vals[i] = row[np.argsort(row)[-trim]]
    for _ in range(2):
        for i in range(cnts.shape[0]):
            row = cnts.data[cnts.indptr[i]:cnts.indptr[i + 1]]
            row[row < vals[i]] = 0
        cnts.eliminate_zeros()
        cnts = cnts.T.tocsr()
    return cnts


def bbknn_graph(X: np.ndarray, batches: np.ndarray, *, neighbors_within_batch: int,
                n_pcs: int | None = None, trim: int | None = None
                ) -> tuple[sp.csr_matrix, sp.csr_matrix, dict[str, Any]]:
    """(distances, connectivities, params) of the batch-balanced kNN graph, as
    ``bbknn.matrix.bbknn(..., computation="cKDTree")`` returns them."""
    from scipy.spatial import cKDTree
    from umap.umap_ import fuzzy_simplicial_set

    X = np.asarray(X)
    n_pcs = int(min(n_pcs or X.shape[1], X.shape[1]))
    batches = np.asarray([str(b) for b in batches])
    names, counts = np.unique(batches, return_counts=True)
    k = int(neighbors_within_batch)
    if len(names) < 2:
        raise BatchMethodError("BBKNN needs at least two batches.")
    if counts.min() < k:
        raise BatchMethodError(f"BBKNN takes {k} neighbours from every batch; the smallest "
                               f"batch has {int(counts.min())} cells.")
    n = X.shape[0]
    dists = np.zeros((n, k * len(names)))
    inds = np.zeros((n, k * len(names)), dtype=int)
    data = X[:, :n_pcs]
    for b, name in enumerate(names):
        pos = np.flatnonzero(batches == name)
        d, i = cKDTree(data[pos]).query(x=data, k=k, workers=N_WORKERS)
        if k == 1:
            d, i = d[:, None], i[:, None]
        cols = slice(b * k, (b + 1) * k)
        inds[:, cols] = pos[i]
        dists[:, cols] = d
    order = np.argsort(dists, axis=1)
    rows = np.arange(n)[:, None]
    inds, dists = inds[rows, order], dists[rows, order]
    total = inds.shape[1]

    empty = sp.coo_matrix(([], ([], [])), shape=(n, 1))
    conn = fuzzy_simplicial_set(empty, total, None, None, knn_indices=inds, knn_dists=dists,
                                set_op_mix_ratio=1.0, local_connectivity=1.0)
    if isinstance(conn, tuple):
        conn = conn[0]
    vals = np.where(inds == np.arange(n)[:, None], 0.0, dists)
    dist = sp.coo_matrix((vals.ravel(), (np.repeat(np.arange(n), total), inds.ravel())),
                         shape=(n, n))
    dist.eliminate_zeros()
    trim = BBKNN_TRIM_FACTOR * total if trim is None else int(trim)
    cnts = conn.tocsr()
    if trim > 0:
        cnts = _trim(cnts, trim)
    params = {"n_neighbors": total, "method": "umap", "metric": "euclidean", "n_pcs": n_pcs,
              "bbknn": {"trim": trim, "computation": "cKDTree",
                        "neighbors_within_batch": k}}
    return dist.tocsr(), cnts, params


# ---------------------------------------------------------------------------
# On an AnnData (the pipeline step, the comparison)
# ---------------------------------------------------------------------------


def apply(adata: ad.AnnData, method: str, batch_key: str, *, n_neighbors: int = 15
          ) -> dict[str, Any]:
    """Run ``method`` on ``adata.obsm["X_pca"]`` and store its result; what to record."""
    method = check_method(method)
    if "X_pca" not in adata.obsm:
        raise ValueError("PCA must be computed before batch correction.")
    batches = _batches(adata.obs, batch_key)
    X = np.asarray(adata.obsm["X_pca"])
    if method == HARMONY:
        adata.obsm[REPRESENTATION[HARMONY]] = harmony_correct(X, adata.obs, batch_key)
        clear(adata, keep=HARMONY)
        return {"representation": REPRESENTATION[HARMONY]}
    if method == SCANORAMA:
        adata.obsm[REPRESENTATION[SCANORAMA]] = scanorama_correct(X, batches)
        clear(adata, keep=SCANORAMA)
        return {"representation": REPRESENTATION[SCANORAMA], "knn": SCANORAMA_KNN,
                "sigma": SCANORAMA_SIGMA, "alpha": SCANORAMA_ALPHA, "approx": False}
    within = bbknn_within(n_neighbors, np.unique(batches).size)
    dist, conn, params = bbknn_graph(X, batches, neighbors_within_batch=within)
    clear(adata, keep=BBKNN)
    params["use_rep"] = "X_pca"
    params["bbknn"]["batch_key"] = batch_key
    adata.uns["neighbors"] = {"params": params, "connectivities_key": "connectivities",
                              "distances_key": "distances"}
    adata.obsp["distances"] = dist
    adata.obsp["connectivities"] = conn
    return {"graph": "bbknn", "neighbors_within_batch": within,
            "n_neighbors_total": int(params["n_neighbors"]), "n_pcs": int(params["n_pcs"]),
            "trim": int(params["bbknn"]["trim"])}


def graph_neighbours(conn: sp.csr_matrix, query: np.ndarray, k: int) -> np.ndarray:
    """Each query cell's ``k`` strongest graph neighbours (by connectivity; the graph
    clustering and UMAP use), itself excluded. A row with fewer cycles through the ones
    it has, so every row has ``k`` (k is cut to the longest row when all are shorter)."""
    conn = sp.csr_matrix(conn)
    rows = []
    longest = 0
    for q in query:
        lo, hi = conn.indptr[q], conn.indptr[q + 1]
        cols, w = conn.indices[lo:hi], conn.data[lo:hi]
        keep = cols != q
        cols, w = cols[keep], w[keep]
        order = np.argsort(-w, kind="stable")[:k]
        rows.append(cols[order])
        longest = max(longest, order.size)
    k = max(1, min(k, longest))
    out = np.empty((len(rows), k), dtype=np.int64)
    for r, cols in enumerate(rows):
        if cols.size == 0:
            out[r] = query[r]
        else:
            out[r] = np.resize(cols, k) if cols.size < k else cols[:k]
    return out
