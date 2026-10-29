"""Weighted nearest neighbours (WNN): one neighbour graph from RNA and protein (C046).

CITE-seq measures two things in each cell. Some cell types are told apart by the
RNA (dendritic cells, platelets, progenitors: the antibody panel has few markers
for them), others by the proteins (naive vs memory T cells: CD45RA / CD45RO, which
the RNA barely separates). WNN (Hao et al. 2021, Cell 184:3573, PMID 34062119)
learns, for each cell, how much to trust each modality, and builds one neighbour
graph from both. This module is that method as Seurat's ``FindMultiModalNeighbors``
runs it (Seurat 5, R/clustering.R ``FindModalityWeights`` / ``MultiModalNN`` /
``PredictAssay``, src/snn.cpp ``SNN_SmallestNonzero_Dist`` / ``ComputeSNN``;
Seurat is MIT-licensed, Copyright (c) 2021 Seurat authors), written in numpy with
pynndescent in place of Annoy (both approximate; exact neighbours up to
``EXACT_MAX_CELLS`` cells).

Inputs. RNA: the PCs the dataset's neighbour graph was built on (``X_pca``, or
``X_pca_harmony`` after Harmony), as many as that graph used. Protein: the CLR
values (obsm ``protein_clr``) of the panel without its isotype controls, each
protein centred and scaled to unit variance (clipped at 10, as ``ScaleData``),
then PCA, ``min(18, proteins - 1)`` components (the vignette's 18 of 25). When the
RNA side is Harmony-corrected, the protein PCs are corrected on the same batch
column. Each modality's rows are L2-normalised (an all-zero row stays zero);
everything below works on those.

Per modality m (k = 20 neighbours, the cell itself included; c = the distance to
the nearest other cell):

1. Prediction from m's own neighbours (the 19 others, averaged in m) and from the
   other modality's neighbours (averaged in m's coordinates). Distances to the
   cell: ``max(||x - prediction|| - c, 0)``.
2. Bandwidth sigma: the shared-neighbour graph of the k-NN sets (Jaccard
   ``s / (2k - s)``, no pruning); of the cells sharing any neighbour, the 20 with
   the smallest Jaccard (all ties at the 20th kept), their distances minus c
   (floored at 0); sigma = the mean of the 20 largest of those.
3. Affinities ``exp(-d / sigma)``; score ``S = clip(within / (cross + 1e-4), 0, 200)``.
4. Weights: ``w_m = exp(S_m) / (exp(S_rna) + exp(S_protein))``, summing to 1.

Joint neighbours: the union of each cell's 199 nearest others in either modality;
similarity ``sum_m w_m(i) exp(-max(||x_m(i) - x_m(j)|| - c_m(i), 0) / sigma_m(i))``;
the 20 most similar kept, distance ``sqrt(clip((1 - similarity) / 2, 0, 1))``.

Then, in scanpy's terms: UMAP connectivities of that 20-NN graph (obsp
``wnn_connectivities`` / ``wnn_distances``, uns ``wnn``; the dataset's own
``neighbors`` graph is left alone), its UMAP (obsm ``X_umap_wnn``), a UMAP of the
protein PCs alone (obsm ``X_umap_protein``, the protein PCs in obsm
``protein_pca``), and Leiden on the WNN connectivities (obs ``wnn_leiden``). The
weights go into obs ``wnn_rna_weight`` / ``wnn_protein_weight``.

Reading the weights. They are relative: a high RNA weight says the RNA predicted
the cell's neighbours better than the proteins did, which is expected wherever
the panel has no marker for the cell's type. They are not a quality score of
either modality. Cells with no antibody counts at all have identical protein rows
(their count is reported); their protein weight means nothing.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp

logger = logging.getLogger(__name__)

TAG = "result:wnn"
UNS_KEY = "scview_wnn"
GRAPH_KEY = "wnn"  # uns["wnn"], obsp["wnn_connectivities"] / ["wnn_distances"]
CONN_KEY = "wnn_connectivities"
DIST_KEY = "wnn_distances"
OBS_RNA = "wnn_rna_weight"
OBS_PROTEIN = "wnn_protein_weight"
OBS_CLUSTERS = "wnn_leiden"
OBSM_WNN = "X_umap_wnn"
OBSM_PROTEIN = "X_umap_protein"
OBSM_PCA = "protein_pca"  # not X_...: every obsm key starting X_ is listed as an embedding
OBS_KEYS = (OBS_RNA, OBS_PROTEIN, OBS_CLUSTERS)
OBSM_KEYS = (OBSM_WNN, OBSM_PROTEIN, OBSM_PCA)
OBSP_KEYS = (CONN_KEY, DIST_KEY)
VERSION = 1

K_NN = 20  # Seurat's k.nn
KNN_RANGE = 200  # Seurat's knn.range: candidates per modality
CROSS_EPS = 1e-4  # Seurat's cross.contant.list
MAX_SCORE = 200.0
SD_SCALE = 1.0
MAX_PROTEIN_PCS = 18
SCALE_MAX = 10.0  # ScaleData's scale.max
MIN_PROTEINS = 5  # fewer (controls left out) is refused
WARN_PROTEINS = 10
MIN_CELLS = 50
EXACT_MAX_CELLS = 4096  # scanpy's switch to approximate neighbours
SIGMA_FLOOR = float(np.finfo(np.float32).eps)  # identical cells: sigma 0 would give exp(-0/0)
SEED = 0
DEFAULT_RESOLUTION = 1.0  # sc.tl.leiden's
DEFAULT_K_PROTEIN = 15  # protein UMAP neighbours when the dataset recorded none
DEFAULT_MIN_DIST = 0.5  # the pipeline's umap_min_dist
MAX_GROUPS = 100

STAGES = (
    ("protein_pca", "Protein PCA"),
    ("knn_rna", "RNA neighbours"),
    ("knn_protein", "Protein neighbours"),
    ("bandwidth", "Kernel bandwidths"),
    ("weights", "Modality weights"),
    ("joint_knn", "Joint neighbours"),
    ("umap_wnn", "WNN UMAP"),
    ("umap_protein", "Protein UMAP"),
    ("leiden", "WNN clusters (Leiden)"),
)

# Measured at 2 CPUs on 10x pbmc10k CITE-seq (7,798 cells, 14 proteins, 50 RNA PCs): 42.6 s a
# run (RNA / protein neighbours, bandwidths, joint neighbours ~10 s, two UMAPs ~9 s each,
# Leiden 4 s), 79 s for the first run after a start (numba compiles pynndescent); peak
# 0.9 GB over the loaded dataset.
FIRST_RUN_SECONDS = 35.0
_SECONDS_PER_CELL = 5.5e-3


class WnnError(ValueError):
    """A run that cannot start or a request that cannot be answered (400 / 404)."""


class WnnCancelled(Exception):
    """The client went away; the run stops before its next stage, nothing is saved."""


# ---------------------------------------------------------------------------
# The algorithm (numpy, scipy, pynndescent): copied into replay scripts as is
# ---------------------------------------------------------------------------


def l2_rows(X):
    """Each row divided by its length; an all-zero row stays zero (Seurat's L2Norm)."""
    X = np.asarray(X, dtype=np.float64)
    norm = np.sqrt((X * X).sum(axis=1, keepdims=True))
    return np.divide(X, norm, out=np.zeros_like(X), where=norm > 0)


def scale_proteins(M, scale_max=10.0):
    """Seurat's ScaleData on proteins (columns): centred, divided by the standard
    deviation (n - 1), values above ``scale_max`` set to it. A constant protein is 0."""
    M = np.asarray(M, dtype=np.float64)
    centred = M - M.mean(axis=0, keepdims=True)
    sd = M.std(axis=0, ddof=1, keepdims=True) if M.shape[0] > 1 else np.ones((1, M.shape[1]))
    Z = np.divide(centred, sd, out=np.zeros_like(centred), where=sd > 0)
    return np.minimum(Z, scale_max)


def protein_pcs(Z, n_pcs):
    """The first ``n_pcs`` principal components of the scaled proteins (cells x PCs),
    each component's sign set so its largest loading is positive."""
    U, S, Vt = np.linalg.svd(Z - Z.mean(axis=0, keepdims=True), full_matrices=False)
    signs = np.sign(Vt[np.arange(Vt.shape[0]), np.abs(Vt).argmax(axis=1)])
    signs[signs == 0] = 1
    return (U[:, :n_pcs] * S[:n_pcs]) * signs[:n_pcs]


def _self_first(idx, dist):
    """Each row's own cell in column 0 at distance 0 (exact duplicates can put another
    cell first), the others after it in their order."""
    n = idx.shape[0]
    rows = np.arange(n)[:, None]
    is_self = idx == rows
    has = is_self.any(axis=1)
    order = np.argsort(~is_self, axis=1, kind="stable")
    idx = np.where(has[:, None], np.take_along_axis(idx, order, axis=1),
                   np.concatenate([rows, idx[:, :-1]], axis=1))
    dist = np.where(has[:, None], np.take_along_axis(dist, order, axis=1),
                    np.concatenate([np.zeros((n, 1), dist.dtype), dist[:, :-1]], axis=1))
    dist[:, 0] = 0
    return idx.astype(np.int64), dist.astype(np.float64)


def knn(X, k, exact_max=4096, seed=0, chunk=512):
    """Each cell's ``k`` nearest cells (Euclidean), itself first: exact up to
    ``exact_max`` cells, else pynndescent (umap-learn's settings)."""
    X = np.ascontiguousarray(X, dtype=np.float32)
    n = X.shape[0]
    k = min(k, n)
    if n <= exact_max:
        X64 = X.astype(np.float64)
        sq = (X64 * X64).sum(axis=1)
        idx = np.empty((n, k), dtype=np.int64)
        dist = np.empty((n, k), dtype=np.float64)
        for s in range(0, n, chunk):
            e = min(n, s + chunk)
            d2 = np.maximum(sq[s:e, None] + sq[None, :] - 2 * X64[s:e] @ X64.T, 0)
            d2[np.arange(e - s), np.arange(s, e)] = -1.0  # the cell itself first
            o = np.argsort(d2, axis=1, kind="stable")[:, :k]
            idx[s:e] = o
            dist[s:e] = np.sqrt(np.maximum(np.take_along_axis(d2, o, axis=1), 0))
    else:
        from pynndescent import NNDescent

        index = NNDescent(X, n_neighbors=k, metric="euclidean", random_state=seed,
                          n_trees=min(64, 5 + int(round(n ** 0.5 / 20.0))),
                          n_iters=max(5, int(round(np.log2(n)))), max_candidates=60,
                          low_memory=True, verbose=False, compressed=False)
        idx, dist = index.neighbor_graph
    return _self_first(np.asarray(idx), np.asarray(dist))


def _row_dist(X, rows, cols):
    """||X[rows] - X[cols]|| for paired index arrays."""
    d = X[rows] - X[cols]
    return np.sqrt((d * d).sum(axis=-1))


def snn_bandwidth(X, nn, nearest, k=20, chunk=2000):
    """Seurat's kernel width per cell (``ComputeSNN`` with no pruning, then
    ``SNN_SmallestNonzero_Dist``). ``nn``: the k-NN sets with the cell itself
    (cells x k); ``nearest``: each cell's distance to its nearest other cell."""
    n, kk = nn.shape
    A = sp.csr_matrix((np.ones(n * kk, dtype=np.float32), nn.ravel(), np.arange(0, n * kk + 1, kk)),
                      shape=(n, n))
    At = A.T.tocsr()
    sigma = np.empty(n, dtype=np.float64)
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        S = (A[s:e] @ At).tocsr()
        S.sort_indices()  # ascending cell index: the order Seurat's ties keep
        counts = np.diff(S.indptr)
        width = int(counts.max())
        m = e - s
        at = np.repeat(np.arange(m), counts)
        pos = np.arange(S.nnz) - S.indptr[:-1][at]
        vals = np.full((m, width), np.inf)
        cols = np.zeros((m, width), dtype=np.int64)
        vals[at, pos] = S.data / (2.0 * kk - S.data)
        cols[at, pos] = S.indices
        order = np.argsort(vals, axis=1, kind="stable")
        vals = np.take_along_axis(vals, order, axis=1)
        cols = np.take_along_axis(cols, order, axis=1)
        n_i = np.minimum(k, counts)
        cut = vals[np.arange(m), n_i - 1]
        keep = (np.arange(width)[None, :] < n_i[:, None]) | (vals == cut[:, None])
        rr, cc = np.nonzero(keep)
        d = _row_dist(X, rr + s, cols[rr, cc])
        c = nearest[rr + s]
        d = np.where(c > 0, np.maximum(d - c, 0), d)
        far = np.full((m, width), -np.inf)
        far[rr, cc] = d
        far = -np.sort(-far, axis=1)  # largest first
        top = np.where(np.arange(width)[None, :] < n_i[:, None], far, 0.0)
        sigma[s:e] = top.sum(axis=1) / n_i
    return sigma


def modality_scores(X, own, other, nearest, sigma, eps=1e-4, max_score=200.0):
    """One modality's score S: how much better its own neighbours (``own``, cells x
    k, itself first) predict a cell in its coordinates ``X`` than the other
    modality's neighbours (``other``) do."""
    within = X[own[:, 1:]].mean(axis=1)
    cross = X[other[:, 1:]].mean(axis=1)
    d_within = np.maximum(np.sqrt(((X - within) ** 2).sum(axis=1)) - nearest, 0)
    d_cross = np.maximum(np.sqrt(((X - cross) ** 2).sum(axis=1)) - nearest, 0)
    s = np.maximum(sigma, np.finfo(np.float32).eps)
    return np.clip(np.exp(-d_within / s) / (np.exp(-d_cross / s) + eps), 0, max_score)


def modality_weights(score_rna, score_protein):
    """Softmax of the two scores: (RNA weight, protein weight), summing to 1."""
    w_rna = 1.0 / (1.0 + np.exp(np.clip(score_protein - score_rna, -700, 700)))
    return w_rna, 1.0 - w_rna


def joint_knn(Xs, cands, nearests, sigmas, weights, k=20, chunk=500):
    """Seurat's ``MultiModalNN``: per cell, the ``k`` most similar of the union of
    each modality's candidates (cells x K, itself first), by the weighted sum of the
    modalities' affinities. Returns (indices, distances), cells x k, itself excluded."""
    n = cands[0].shape[0]
    C_all = np.concatenate([c[:, 1:] for c in cands], axis=1)
    out_idx = np.empty((n, k), dtype=np.int64)
    out_sim = np.empty((n, k), dtype=np.float64)
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        rows = np.arange(s, e)
        C = C_all[s:e]
        o = np.argsort(C, axis=1, kind="stable")
        sorted_c = np.take_along_axis(C, o, axis=1)
        dup_sorted = np.zeros(C.shape, dtype=bool)
        dup_sorted[:, 1:] = sorted_c[:, 1:] == sorted_c[:, :-1]
        dup = np.empty_like(dup_sorted)
        np.put_along_axis(dup, o, dup_sorted, axis=1)
        sim = np.zeros(C.shape, dtype=np.float64)
        for X, c, sg, w in zip(Xs, nearests, sigmas, weights):
            d = _row_dist(X, np.broadcast_to(rows[:, None], C.shape), C)
            d = np.maximum(d - c[rows][:, None], 0)
            sim += w[rows][:, None] * np.exp(-d / np.maximum(sg[rows], np.finfo(np.float32).eps)[:, None])
        sim[dup | (C == rows[:, None])] = -np.inf
        top = np.argsort(-sim, axis=1, kind="stable")[:, :k]
        out_idx[s:e] = np.take_along_axis(C, top, axis=1)
        out_sim[s:e] = np.take_along_axis(sim, top, axis=1)
    dist = np.sqrt(np.clip((1 - out_sim) / 2, 0, 1))
    return out_idx, dist


def wnn_graphs(idx, dist):
    """The joint neighbours as scanpy keeps a graph: UMAP connectivities (umap-learn's
    fuzzy_simplicial_set on the cell itself + its ``k`` neighbours, as scanpy's
    neighbours step) and the distances (csr, ``k`` per row)."""
    from umap.umap_ import fuzzy_simplicial_set

    n, k = idx.shape
    knn_idx = np.concatenate([np.arange(n)[:, None], idx], axis=1)
    knn_dist = np.concatenate([np.zeros((n, 1)), dist], axis=1).astype(np.float32)
    conn, _, _ = fuzzy_simplicial_set(sp.coo_matrix((n, 1)), k + 1, None, None,
                                      knn_indices=knn_idx, knn_dists=knn_dist)
    distances = sp.csr_matrix((dist.ravel().astype(np.float32), idx.ravel(), np.arange(0, n * k + 1, k)),
                              shape=(n, n))
    distances.sort_indices()
    return conn.tocsr(), distances


def wnn_steps(rna, prot, k=20, knn_range=200, exact_max=4096, sd_scale=1.0):
    """Seurat's FindMultiModalNeighbors on two embeddings (cells x PCs, not yet
    L2-normalised), as a generator: yields each stage's name before it runs and
    returns a dict (weights, the joint neighbours idx / dist, sigma, scores)."""
    n = rna.shape[0]
    K = min(knn_range, n)
    k = min(k, n - 1)
    Xr, Xp = l2_rows(rna), l2_rows(prot)
    yield "knn_rna"
    nn_r, d_r = knn(Xr, K, exact_max)
    yield "knn_protein"
    nn_p, d_p = knn(Xp, K, exact_max)
    c_r, c_p = d_r[:, 1], d_p[:, 1]
    del d_r, d_p
    yield "bandwidth"
    sig_r = snn_bandwidth(Xr, nn_r[:, :k], c_r, k) * sd_scale
    sig_p = snn_bandwidth(Xp, nn_p[:, :k], c_p, k) * sd_scale
    yield "weights"
    s_r = modality_scores(Xr, nn_r[:, :k], nn_p[:, :k], c_r, sig_r)
    s_p = modality_scores(Xp, nn_p[:, :k], nn_r[:, :k], c_p, sig_p)
    w_r, w_p = modality_weights(s_r, s_p)
    yield "joint_knn"
    idx, dist = joint_knn((Xr, Xp), (nn_r, nn_p), (c_r, c_p), (sig_r, sig_p), (w_r, w_p), k)
    return {"rna_weight": w_r, "protein_weight": w_p, "idx": idx, "dist": dist,
            "sigma_rna": sig_r, "sigma_protein": sig_p, "score_rna": s_r, "score_protein": s_p,
            "nearest_rna": c_r, "nearest_protein": c_p}


def wnn_neighbors(rna, prot, k=20, knn_range=200, exact_max=4096, sd_scale=1.0):
    """``wnn_steps`` run to its end: the result dict."""
    steps = wnn_steps(rna, prot, k, knn_range, exact_max, sd_scale)
    while True:
        try:
            next(steps)
        except StopIteration as done:
            return done.value


REPLAY_CONSTANTS = ("K_NN", "KNN_RANGE", "SCALE_MAX", "EXACT_MAX_CELLS")
REPLAY_CODE = (l2_rows, scale_proteins, protein_pcs, _self_first, knn, _row_dist, snn_bandwidth,
               modality_scores, modality_weights, joint_knn, wnn_graphs, wnn_steps, wnn_neighbors)


# ---------------------------------------------------------------------------
# What a run uses (plan) and the run itself
# ---------------------------------------------------------------------------


def _last_params(adata, step: str) -> dict[str, Any] | None:
    from scview.core import provenance

    for h in reversed(provenance.read_provenance(adata).get("history", [])):
        if isinstance(h, dict) and h.get("step") == step and isinstance(h.get("params"), dict):
            return h["params"]
    return None


def panel(adata) -> tuple[list[str], list[str]]:
    """(proteins used, isotype controls left out), in the panel's order."""
    from scview.core import protein

    names = protein.protein_names(adata)
    info = adata.uns.get(protein.UNS_KEY) if protein.UNS_KEY in adata.uns else None
    found = info.get("controls") if isinstance(info, dict) else None  # an array once read from h5ad
    listed = {str(c) for c in np.asarray(found, dtype=object).ravel()} if found is not None else set()
    controls = [n for n in names if n in listed or protein.is_control(n)]
    return [n for n in names if n not in controls], controls


def _rna_rep(adata) -> tuple[str, int, list[str]]:
    """The representation and PC count the dataset's neighbour graph used."""
    from scview.core import batch_methods

    notes: list[str] = []
    nb = adata.uns.get("neighbors") if "neighbors" in adata.uns else None
    params = nb.get("params") if isinstance(nb, dict) and isinstance(nb.get("params"), dict) else {}
    rep = params.get("use_rep")
    if params.get("bbknn") is not None or batch_methods.is_bbknn_graph(adata):
        rep = "X_pca"
        notes.append("The RNA graph is BBKNN's (built batch by batch); WNN uses the uncorrected "
                     "PCA for the RNA side.")
    rep = str(rep) if rep else batch_methods.neighbour_rep(adata)
    if rep not in adata.obsm:
        raise WnnError(f"There is no {rep} for the RNA side: run the pipeline's PCA and neighbours "
                       "steps first.")
    width = int(np.asarray(adata.obsm[rep]).shape[1])
    n_pcs = params.get("n_pcs")
    d_r = min(int(n_pcs), width) if n_pcs else width
    return rep, d_r, notes


def _batch(adata, rep: str) -> tuple[str | None, str | None]:
    """The batch column Harmony corrected the RNA PCs on (when ``rep`` is Harmony's)."""
    if rep != "X_pca_harmony":
        return None, None
    rec = _last_params(adata, "batch_correction") or {}
    key = rec.get("batch_key")
    if key and key in adata.obs.columns and not adata.obs[key].isna().any():
        return str(key), (f"The RNA PCs are Harmony-corrected on '{key}': the protein PCs are "
                          "corrected on it too.")
    return None, ("The RNA PCs are Harmony-corrected, but the batch column is not recorded: the "
                  "protein PCs are not corrected.")


def zero_protein_cells(adata, proteins: list[str]) -> int:
    from scview.core import protein

    m = adata.obsm.get(protein.COUNTS_KEY) if protein.COUNTS_KEY in adata.obsm else None
    if not isinstance(m, pd.DataFrame):
        return 0
    cols = [p for p in proteins if p in m.columns]
    return int((m[cols].to_numpy().sum(axis=1) == 0).sum()) if cols else 0


@dataclass
class WnnPlan:
    n_cells: int
    proteins: list[str]
    controls: list[str]
    rep: str
    d_rna: int
    d_protein: int
    batch_key: str | None
    n_zero_protein: int
    k: int = K_NN
    knn_range: int = KNN_RANGE
    k_protein_umap: int = DEFAULT_K_PROTEIN
    min_dist: float = DEFAULT_MIN_DIST
    resolution: float = DEFAULT_RESOLUTION
    n_iterations: int = -1
    exact: bool = False
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"n_cells": self.n_cells, "proteins": list(self.proteins), "controls": list(self.controls),
                "n_proteins": len(self.proteins), "rep": self.rep, "d_rna": self.d_rna,
                "d_protein": self.d_protein, "batch_key": self.batch_key,
                "n_zero_protein": self.n_zero_protein, "k": self.k, "knn_range": self.knn_range,
                "k_protein_umap": self.k_protein_umap, "min_dist": self.min_dist,
                "resolution": self.resolution, "n_iterations": self.n_iterations, "exact": self.exact,
                "warnings": list(self.warnings), "notes": list(self.notes),
                "estimate_seconds": estimate_seconds(self.n_cells)}


def estimate_seconds(n_cells: int) -> float:
    """A run's time at 2 CPUs (both kNN searches, bandwidths, joint neighbours, two
    UMAPs, Leiden), plus numba's first compile."""
    return round(FIRST_RUN_SECONDS + _SECONDS_PER_CELL * n_cells, 0)


def plan(adata, protein_pcs: int | None = None, resolution: float | None = None) -> WnnPlan:
    """What a run would use; WnnError for a dataset it cannot run on."""
    from scview.core import leiden, protein

    if not protein.has_protein(adata):
        raise WnnError("This dataset has no protein (CITE-seq antibody) counts.")
    used, controls = panel(adata)
    if len(used) < MIN_PROTEINS:
        raise WnnError(f"WNN needs at least {MIN_PROTEINS} proteins besides isotype controls; this "
                       f"panel has {len(used)}.")
    if adata.n_obs < MIN_CELLS:
        raise WnnError(f"WNN needs at least {MIN_CELLS} cells; this dataset has {adata.n_obs}.")
    rep, d_r, notes = _rna_rep(adata)
    batch_key, batch_note = _batch(adata, rep)
    if batch_note:
        notes.append(batch_note)
    if rep == "X_scanorama":
        notes.append("The RNA side is Scanorama-corrected; the protein PCs are not corrected.")
    top = len(used) - 1
    d_p = min(MAX_PROTEIN_PCS, top) if protein_pcs is None else int(protein_pcs)
    if not 2 <= d_p <= top:
        raise WnnError(f"Protein PCs must be between 2 and {top} (the proteins used minus one).")
    warnings: list[str] = []
    if len(used) < WARN_PROTEINS:
        warnings.append(f"Only {len(used)} proteins: RNA weight is expected wherever the panel has no "
                        "markers.")
    n_zero = zero_protein_cells(adata, used)
    if n_zero:
        warnings.append(f"{n_zero:,} cells have no antibody counts at all; their protein weight means "
                        "nothing.")
    emb = _last_params(adata, "embeddings") or {}
    nbr = _last_params(adata, "neighbors") or {}
    nb = adata.uns.get("neighbors") if "neighbors" in adata.uns else None
    k_umap = (nb.get("params") or {}).get("n_neighbors") if isinstance(nb, dict) else None
    k_umap = int(k_umap or nbr.get("n_neighbors") or DEFAULT_K_PROTEIN)
    res = DEFAULT_RESOLUTION if resolution is None else float(resolution)
    if not 0 < res <= 10:
        raise WnnError("Resolution must be above 0 and at most 10.")
    return WnnPlan(n_cells=int(adata.n_obs), proteins=used, controls=controls, rep=rep, d_rna=d_r,
                   d_protein=d_p, batch_key=batch_key, n_zero_protein=n_zero,
                   k_protein_umap=min(k_umap, adata.n_obs - 1),
                   min_dist=float(emb["umap_min_dist"] if emb.get("umap_min_dist") is not None
                                  else DEFAULT_MIN_DIST), resolution=res,
                   n_iterations=leiden.recorded_iterations(adata),
                   exact=adata.n_obs <= EXACT_MAX_CELLS, warnings=warnings, notes=notes)


@dataclass
class WnnInputs:
    plan: WnnPlan
    rna: np.ndarray  # cells x d_rna (float32 copy)
    clr: np.ndarray  # cells x proteins used (float32 copy)
    batches: pd.DataFrame | None  # the batch column, for Harmony
    fingerprint: str  # of the inputs: a later change to them makes the result stale


def inputs_fingerprint(rna: np.ndarray, clr: np.ndarray) -> str:
    h = hashlib.sha256(np.ascontiguousarray(rna, dtype=np.float32).tobytes())
    h.update(np.ascontiguousarray(clr, dtype=np.float32).tobytes())
    return h.hexdigest()[:24]


def _inputs(adata, p: WnnPlan) -> tuple[np.ndarray, np.ndarray]:
    from scview.core import protein

    rna = np.asarray(adata.obsm[p.rep])[:, :p.d_rna].astype(np.float32, copy=True)
    clr = adata.obsm[protein.CLR_KEY][p.proteins].to_numpy(dtype=np.float32, copy=True)
    return rna, clr


def prepare(adata, p: WnnPlan) -> WnnInputs:
    """The run's own copies of the RNA PCs and the CLR values (a few MB): the dataset
    stays free for other requests while it runs."""
    rna, clr = _inputs(adata, p)
    batches = adata.obs[[p.batch_key]].copy() if p.batch_key else None
    return WnnInputs(plan=p, rna=rna, clr=clr, batches=batches, fingerprint=inputs_fingerprint(rna, clr))


@dataclass
class WnnResult:
    rna_weight: np.ndarray
    protein_weight: np.ndarray
    connectivities: sp.csr_matrix
    distances: sp.csr_matrix
    umap_wnn: np.ndarray
    umap_protein: np.ndarray
    protein_pca: np.ndarray
    clusters: pd.Categorical
    sigma_rna: np.ndarray
    sigma_protein: np.ndarray
    seconds: float
    stage_seconds: dict[str, float]


def _protein_pca(inp: WnnInputs) -> np.ndarray:
    p = inp.plan
    pcs = protein_pcs(scale_proteins(inp.clr, SCALE_MAX), p.d_protein)
    if p.batch_key and inp.batches is not None:
        from scview.core.batch_methods import harmony_correct

        pcs = harmony_correct(pcs, inp.batches, p.batch_key)
    return np.asarray(pcs, dtype=np.float32)


def _umap(holder, neighbors_key: str, key_added: str, min_dist: float) -> np.ndarray:
    import scanpy as sc

    sc.tl.umap(holder, neighbors_key=neighbors_key, key_added=key_added, min_dist=min_dist,
               random_state=SEED)
    return np.asarray(holder.obsm[key_added], dtype=np.float32)


def iter_wnn(inp: WnnInputs, stop: Callable[[], bool] = lambda: False) -> Iterator[dict[str, Any]]:
    """Run WNN: yields ``{"type": "stage", ...}`` as each stage starts, then
    ``{"type": "result", "result": WnnResult}``. ``stop`` is checked between stages
    (WnnCancelled: nothing is kept)."""
    import anndata as ad
    import scanpy as sc

    from scview.core import leiden

    p = inp.plan
    t0 = time.monotonic()
    spent: dict[str, float] = {}
    names = [s for s, _ in STAGES]
    labels = dict(STAGES)
    current: list[Any] = [None, t0]

    def begin(name: str) -> dict[str, Any]:
        if stop():
            raise WnnCancelled()
        now = time.monotonic()
        if current[0] is not None:
            spent[current[0]] = round(now - current[1], 2)
        current[0], current[1] = name, now
        return {"type": "stage", "stage": name, "label": labels[name], "index": names.index(name) + 1,
                "total": len(names), "elapsed_s": round(now - t0, 1)}

    yield begin("protein_pca")
    pca = _protein_pca(inp)
    steps = wnn_steps(inp.rna, pca, p.k, p.knn_range, EXACT_MAX_CELLS, SD_SCALE)
    while True:
        try:
            name = next(steps)
        except StopIteration as done:
            result = done.value
            break
        yield begin(name)
    conn, distances = wnn_graphs(result["idx"], result["dist"])
    n = inp.rna.shape[0]
    holder = ad.AnnData(obs=pd.DataFrame(index=[str(i) for i in range(n)]))
    holder.obsm[p.rep] = inp.rna  # its first d_rna columns: what the UMAP reads for components
    holder.obsm[OBSM_PCA] = pca
    holder.obsp[CONN_KEY] = conn.copy()  # umap-learn trims the graph it is given in place
    holder.obsp[DIST_KEY] = distances
    holder.uns[GRAPH_KEY] = graph_uns(p)
    yield begin("umap_wnn")
    umap_wnn = _umap(holder, GRAPH_KEY, OBSM_WNN, p.min_dist)
    yield begin("umap_protein")
    sc.pp.neighbors(holder, n_neighbors=p.k_protein_umap, use_rep=OBSM_PCA, key_added="protein",
                    random_state=SEED)
    umap_protein = _umap(holder, "protein", OBSM_PROTEIN, p.min_dist)
    del holder
    yield begin("leiden")
    clusters = leiden.labels(conn, p.resolution, n_iterations=p.n_iterations)
    now = time.monotonic()
    spent[current[0]] = round(now - current[1], 2)
    yield {"type": "result", "result": WnnResult(
        rna_weight=result["rna_weight"].astype(np.float32),
        protein_weight=result["protein_weight"].astype(np.float32),
        connectivities=conn, distances=distances, umap_wnn=umap_wnn, umap_protein=umap_protein,
        protein_pca=pca, clusters=clusters, sigma_rna=result["sigma_rna"],
        sigma_protein=result["sigma_protein"], seconds=now - t0, stage_seconds=spent)}


def run(inp: WnnInputs) -> WnnResult:
    """``iter_wnn`` to its end."""
    for item in iter_wnn(inp):
        if item["type"] == "result":
            return item["result"]
    raise WnnError("The run ended without a result.")  # pragma: no cover


def graph_uns(p: WnnPlan) -> dict[str, Any]:
    """uns["wnn"] as scanpy's neighbours step writes it (sc.tl.umap reads the params)."""
    return {"connectivities_key": CONN_KEY, "distances_key": DIST_KEY,
            "params": {"method": "umap", "n_neighbors": p.k + 1, "use_rep": p.rep, "n_pcs": p.d_rna,
                       "metric": "euclidean", "random_state": SEED, "wnn": True}}


# ---------------------------------------------------------------------------
# Saved result
# ---------------------------------------------------------------------------


def method_text(p: WnnPlan) -> str:
    batch = f", Harmony-corrected on '{p.batch_key}' as the RNA" if p.batch_key else ""
    return (
        "Weighted nearest neighbours (Hao et al. 2021; Seurat FindMultiModalNeighbors, k = 20, "
        f"200 candidates per modality): RNA {p.rep} (first {p.d_rna} PCs); protein CLR of "
        f"{len(p.proteins)} antibodies (isotype controls left out) scaled per protein, PCA "
        f"({p.d_protein} PCs{batch}); rows L2-normalised. Per cell, a modality's weight is how much "
        "better its own neighbours predict the cell than the other modality's do (softmax of the two "
        "scores). Joint graph: UMAP connectivities of the 20 most similar cells by the weighted "
        f"affinities; Leiden at resolution {p.resolution:g} on it."
    )


def replay_params(p: WnnPlan, adata) -> dict[str, Any]:
    """What provenance records and the replay script reads."""
    from scview.core import protein

    info = adata.uns.get(protein.UNS_KEY) if protein.UNS_KEY in adata.uns else None
    original = info.get("original") if isinstance(info, dict) and isinstance(info.get("original"), dict) else {}
    names = protein.protein_names(adata)
    return {"proteins": list(p.proteins), "controls": list(p.controls), "panel": names,
            "features": [str(original.get(n, n)) for n in names], "rep": p.rep, "n_pcs_rna": p.d_rna,
            "n_pcs_protein": p.d_protein, "batch_key": p.batch_key, "k": p.k, "knn_range": p.knn_range,
            "k_protein_umap": p.k_protein_umap, "min_dist": p.min_dist, "resolution": p.resolution,
            "n_iterations": p.n_iterations, "exact_max_cells": EXACT_MAX_CELLS}


def save(adata, inp: WnnInputs, res: WnnResult, *, resources: dict | None = None,
         cells_fingerprint: str | None = None) -> dict[str, Any]:
    """Store the result in ``adata`` (obs, obsm, obsp, uns, a provenance step); replaces
    the uns entries (never mutates them), so ``write_derived`` sees them as changed."""
    from scview.core import provenance

    p = inp.plan
    created = datetime.now(timezone.utc).isoformat()
    adata.obs[OBS_RNA] = res.rna_weight
    adata.obs[OBS_PROTEIN] = res.protein_weight
    adata.obs[OBS_CLUSTERS] = pd.Categorical(np.asarray(res.clusters).astype(str),
                                             categories=list(res.clusters.categories))
    adata.obsm[OBSM_WNN] = res.umap_wnn
    adata.obsm[OBSM_PROTEIN] = res.umap_protein
    adata.obsm[OBSM_PCA] = res.protein_pca
    adata.obsp[CONN_KEY] = res.connectivities
    adata.obsp[DIST_KEY] = res.distances
    adata.uns[GRAPH_KEY] = graph_uns(p)
    info = {
        "version": VERSION, "run_id": f"{created}|{adata.n_obs}", "created_at": created,
        "plan": {k: v for k, v in p.to_dict().items() if k != "estimate_seconds"},
        "cells_fingerprint": cells_fingerprint, "inputs_fingerprint": inp.fingerprint,
        "n_clusters": int(len(res.clusters.categories)),
        "mean_rna_weight": round(float(np.mean(res.rna_weight)), 4),
        "sigma_rna_median": round(float(np.median(res.sigma_rna)), 5),
        "sigma_protein_median": round(float(np.median(res.sigma_protein)), 5),
        "seconds": round(res.seconds, 1), "stage_seconds": dict(res.stage_seconds),
        "method": method_text(p),
    }
    adata.uns[UNS_KEY] = json.dumps(info)
    provenance.record_step(adata, step="wnn", tool="scview.wnn (Seurat FindMultiModalNeighbors algorithm)",
                           params=replay_params(p, adata), resources=resources)
    return summary(adata)


def stored(adata) -> dict[str, Any] | None:
    """The saved run's info, or None (none saved, or its columns are gone or of
    another length)."""
    raw = adata.uns.get(UNS_KEY) if UNS_KEY in adata.uns else None
    if raw is None:
        return None
    try:
        info = json.loads(str(raw))
    except (TypeError, ValueError):
        logger.warning("Ignoring an unreadable %s", UNS_KEY)
        return None
    if not isinstance(info, dict) or any(c not in adata.obs.columns for c in (OBS_RNA, OBS_PROTEIN)):
        return None
    return info


def stale_reason(adata, info: dict, cells_fingerprint: str | None = None) -> str | None:
    """Why the saved result no longer fits the dataset (other cells, other PCs or
    protein values), or None."""
    if cells_fingerprint is not None and info.get("cells_fingerprint") not in (None, cells_fingerprint):
        return "The cells changed since WNN ran (a filter or a merge): run it again."
    p = info.get("plan") or {}
    try:
        rep, d_r, proteins = p["rep"], int(p["d_rna"]), list(p["proteins"])
        from scview.core import protein

        if rep not in adata.obsm or not protein.has_protein(adata) or any(
                n not in adata.obsm[protein.CLR_KEY].columns for n in proteins):
            return "The PCs or the protein values WNN used are gone: run it again."
        rna = np.asarray(adata.obsm[rep])[:, :d_r]
        clr = adata.obsm[protein.CLR_KEY][proteins].to_numpy(dtype=np.float32)
    except (KeyError, TypeError, ValueError):
        return "The saved run does not say what it used: run it again."
    if inputs_fingerprint(rna, clr) != info.get("inputs_fingerprint"):
        return "The PCs or the protein values changed since WNN ran (a pipeline run): run it again."
    return None


def _groups(adata, column: str) -> tuple[np.ndarray, list[str]]:
    if column not in adata.obs.columns:
        raise WnnError(f"Column '{column}' not found.")
    s = adata.obs[column]
    labels = s.astype(str).where(s.notna()).to_numpy(dtype=object)
    present = {x for x in labels if x is not None and x == x}
    order = [str(c) for c in s.cat.categories] if hasattr(s, "cat") else sorted(present, key=str)
    groups = [g for g in order if g in present]
    if len(groups) > MAX_GROUPS:
        raise WnnError(f"'{column}' has {len(groups)} groups; at most {MAX_GROUPS} are shown.")
    return labels, groups


def by_group(adata, column: str) -> dict[str, Any]:
    """Mean and median RNA weight per group of ``column`` (groups in its order)."""
    labels, groups = _groups(adata, column)
    w = adata.obs[OBS_RNA].to_numpy(dtype=np.float64)
    rows = []
    for g in groups:
        v = w[labels == g]
        rows.append({"group": g, "n_cells": int(v.size), "mean_rna_weight": round(float(v.mean()), 4),
                     "median_rna_weight": round(float(np.median(v)), 4)})
    return {"column": column, "rows": rows}


def violin(adata, column: str) -> dict[str, list[float]]:
    labels, groups = _groups(adata, column)
    w = adata.obs[OBS_RNA].to_numpy(dtype=np.float64)
    return {g: [round(float(x), 4) for x in w[labels == g]] for g in groups}


def weights(adata) -> np.ndarray:
    return adata.obs[OBS_RNA].to_numpy(dtype=np.float32)


def summary(adata, groupby: str | None = None, cells_fingerprint: str | None = None) -> dict[str, Any]:
    """The saved run for the UI: ``{"available": False}`` without one."""
    info = stored(adata)
    if info is None:
        return {"available": False}
    out: dict[str, Any] = {"available": True, **info}
    out["stale_reason"] = stale_reason(adata, info, cells_fingerprint)
    out["stale"] = out["stale_reason"] is not None
    out["embeddings"] = [k for k in ("X_umap", OBSM_PROTEIN, OBSM_WNN) if k in adata.obsm]
    out["clusters_column"] = OBS_CLUSTERS if OBS_CLUSTERS in adata.obs.columns else None
    if groupby:
        try:
            out["by_group"] = by_group(adata, groupby)
        except WnnError as exc:
            out["by_group"] = None
            out["by_group_error"] = str(exc)
    return out


def runnable(adata) -> dict[str, Any]:
    """Whether a run can start with the defaults, and the plan when it can."""
    try:
        return {"ok": True, "reason": None, "plan": plan(adata).to_dict()}
    except WnnError as exc:
        return {"ok": False, "reason": str(exc), "plan": None}


def drop(adata) -> None:
    """Remove a saved result (a merge: it was made on other cells and another panel)."""
    for key in OBS_KEYS:
        if key in adata.obs.columns:
            del adata.obs[key]
    for key in OBSM_KEYS:
        adata.obsm.pop(key, None)
    for key in OBSP_KEYS:
        if key in adata.obsp:
            del adata.obsp[key]
    for key in (GRAPH_KEY, UNS_KEY):
        adata.uns.pop(key, None)


# ---------------------------------------------------------------------------
# Co-pilot
# ---------------------------------------------------------------------------


def grounding_lines(adata) -> tuple[list[str], str]:
    """The saved run as a fact: mean RNA / protein weight per label group (an author
    cell-type column, else the clustering), the protein panel, and how to read the
    weights. Nothing without a run, or when it no longer fits the cells."""
    from scview.core.qc_filter import label_grouping

    info = stored(adata)
    if info is None or stale_reason(adata, info) is not None:
        return [], ""
    p = info.get("plan") or {}
    column, _why = label_grouping(adata)
    parts: list[str] = []
    if column:
        try:
            rows = sorted(by_group(adata, column)["rows"], key=lambda r: -r["mean_rna_weight"])
        except WnnError:
            rows = []
        parts = [f"{r['group']} RNA {r['mean_rna_weight']:.2f} / protein {1 - r['mean_rna_weight']:.2f} "
                 f"(n = {r['n_cells']:,})" for r in rows if r["n_cells"] >= 10]
    detail = (f"WNN on {len(p.get('proteins') or [])} proteins and {p.get('d_rna')} RNA PCs; mean RNA weight "
              f"{info.get('mean_rna_weight', 0):.2f}")
    line = (f"- Joint RNA + protein neighbours (WNN, Hao 2021): {detail}; {info.get('n_clusters')} WNN "
            f"clusters (obs {OBS_CLUSTERS}). "
            + (f"Mean modality weight per {column}, highest RNA weight first: " + "; ".join(parts) + ". "
               if parts else "")
            + "A cell's weights sum to 1 and are relative: a high RNA weight means the RNA predicted its "
              "neighbours better than the proteins, expected where the antibody panel has no marker for the "
              f"type; they are not quality scores [{TAG}]")
    return ["\n## Joint RNA + protein (WNN)", line], detail
