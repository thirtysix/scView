"""RNA velocity: steady-state and stochastic models, native, on the modality store's counts.

Ported by behaviour from scVelo 0.3.3 (BSD-3; Bergen 2020, La Manno 2018), whose stochastic mode
does not run on this image's numpy and which is not a dependency. Step for step as
``scv.pp.filter_and_normalize(min_shared_counts=20, n_top_genes=2000)``,
``scv.pp.moments(n_pcs=30, n_neighbors=30)``, ``scv.tl.velocity(mode=...)``,
``scv.tl.velocity_graph()``, ``scv.tl.velocity_embedding()``, ``scv.tl.velocity_confidence()``:

- genes: spliced + unspliced summed over the cells where both are non-zero, at least 20;
- each matrix scaled per cell to its own median total (totals before the gene filter);
- 2,000 variable genes by normalised dispersion of spliced (Seurat flavour, 20 mean bins);
- neighbours: the dataset's X_pca and graph when they cover the cells and hold k >= the request,
  else computed here (never saved over the dataset's own); the summary says which;
- first moments: Ms, Mu = the mean over each cell's graph neighbours and itself, unweighted;
- gamma: least squares through the origin on the cells in the top 5% of scaled S + U; r2 of that
  fit over all cells; velocity genes r2 > 0.01, gamma > 0.01;
- stochastic: second moments, then generalised least squares over both moments, velocity genes only;
- graph: cosine between each cell's velocity and its displacement to its neighbours and their
  neighbours (2 recursions); sqrt transform for stochastic only, as scVelo decides it;
- arrows: expected displacement under the transition matrix (scale 10, self transitions, negative
  cosines count against), per cell; the grid as ``compute_velocity_on_grid``.

Not offered: the dynamical model and latent time. ``velocity_<embedding>`` is scVelo's before its
autoscale (a figure-size scalar from matplotlib's quiver): the plot sets the arrow length itself.

Everything runs a block of cells at a time from the store, so a large dataset (X on disk) runs too:
only obs_names, obsm, obsp and uns of the dataset are read, never X. Ms and Mu are dense on the 2,000
genes; past ``MEMMAP_BYTES`` they are memory-mapped files beside the store, so their pages are file
cache, which the memory gauge leaves out.

Kept: obs ``velocity_length``, ``velocity_confidence``, ``velocity_unspliced_fraction``; obsm
``velocity_<embedding>`` per 2D embedding; var ``velocity_gamma``, ``velocity_r2``, ``velocity_gene``;
uns ``scview_velocity`` (a JSON string: ``source`` + ``run``: method, params, gene table, grid). Too big
for the h5ad, kept in the store folder: ``velocity_run.h5`` (per cell: the moments' connectivities and
the signed cosine graph, for re-projecting onto another embedding) and ``velocity_fit.h5`` (per GENE:
normalised S and U, so a phase portrait reads one row).
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import math
import os
import shutil
import threading
import uuid
import weakref
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp

from scview.core import modality_store as ms
from scview.core import results_store, velocity_input as vi

logger = logging.getLogger(__name__)

KIND = "velocity"
TAG = "result:velocity"  # the co-pilot's fact (claims.py family "velocity")
STEP = "velocity"
UNS_KEY = vi.UNS_KEY
OBS_LENGTH = "velocity_length"
OBS_CONFIDENCE = "velocity_confidence"
OBS_UNSPLICED = "velocity_unspliced_fraction"
VAR_GAMMA = "velocity_gamma"
VAR_R2 = "velocity_r2"
VAR_GENE = "velocity_gene"
RUN_STORE = "velocity_run"
FIT_STORE = "velocity_fit"
METHODS = ("stochastic", "steady_state")
METHOD_NAMES = {"stochastic": "Stochastic (second moments)", "steady_state": "Steady state (deterministic)"}

MIN_SHARED_COUNTS = 20
N_TOP_GENES = 2000
N_NEIGHBORS = 30
N_PCS = 30
MIN_R2 = 0.01
MIN_RATIO = 0.01
PERCENTILE = 95
SCALE = 10.0
GRID_POINTS = 50
SMOOTH = 0.5
MIN_MASS = 1.0
MAX_GRID_NEIGHBOURS = 1000  # scVelo takes n_cells / 50 cells per grid point: past 50,000 cells, this many (see vel_grid)
MIN_CELLS = 50
MIN_GENES = 10
BLOCK_CELLS = 2048
FIT_BLOCK_GENES = 256
FIT_BLOCK_BYTES = 512 * 1024**2  # at most this in the fit's working arrays at once (about 35 B per cell x gene)
MEMMAP_BYTES = 2 * 1024**3
PHASE_MAX_POINTS = 20_000
TOP_GENES = 5
GENE_DETECTED_SHARE = 0.1
LOW_UNSPLICED = 0.05  # scVelo's documentation gives 10-25% unspliced as typical
LOW_CONFIDENCE = 0.5
FEW_VELOCITY_GENES = 50
NOTE = ("Arrows show direction on a 2D plot, not a fate claim: the projection loses most of the high-dimensional "
        "signal, and the model assumes one splicing rate per gene. The dynamical model and latent time are not offered.")


class VelocityError(RuntimeError):
    """A run that cannot go on; the message says what to do."""

    def __init__(self, message: str, status: int = 422) -> None:
        super().__init__(message)
        self.status = status


class VelocityCancelled(RuntimeError):
    """The client went away: nothing saved."""


# ---------------------------------------------------------------------------
# The method on arrays (also copied into a replay script: numpy, scipy, pandas only)
# ---------------------------------------------------------------------------


def vel_row_scale(sizes):
    """What scales each cell's row to the median total (scv.pp.normalize_per_cell)."""
    target = np.median(sizes)
    target = target + (target == 0)
    counts = sizes / target
    counts = counts + (counts == 0)
    return 1.0 / counts


def vel_select_genes(mean, mean_sq, n_cells, n_top):
    """Variable genes by normalised dispersion, Seurat flavour on 20 bins of log1p(mean)
    (scv.pp.filter_genes_dispersion): a mask, all True when there are n_top genes or fewer."""
    if mean.size < n_top:
        return np.ones(mean.size, dtype=bool)
    n = max(n_cells, 2)
    var = np.nan_to_num((mean_sq - mean ** 2) * (n / (n - 1)))
    mean = np.nan_to_num(mean).copy()
    mean[mean == 0] = 1e-12
    dispersion = var / mean
    dispersion[dispersion == 0] = np.nan
    df = pd.DataFrame({"mean": np.log1p(mean), "dispersion": np.log(dispersion)})
    df["mean_bin"] = pd.cut(df["mean"], bins=20)
    grouped = df.groupby("mean_bin", observed=False)["dispersion"]
    bin_mean, bin_std = grouped.mean(), grouped.std(ddof=1)
    single = bin_std.isnull()
    bin_std[single] = bin_mean[single].values
    bin_mean[single] = 0
    mu = bin_mean[df["mean_bin"].values].values
    std = bin_std[df["mean_bin"].values].values
    norm = ((df["dispersion"] - mu) / std).fillna(0)
    cut = norm.nlargest(n_top).values[-1]
    return norm.values >= cut


def vel_top_per_row(graph, k, largest):
    """Keep each row's k largest (else smallest) stored values (scVelo's select_connectivities,
    select_distances); k is at most the fewest any row has."""
    graph = sp.csr_matrix(graph, copy=True)
    counts = np.diff(graph.indptr)
    k = min(int(counts.min()), int(k))
    for row in np.flatnonzero(counts > k):
        a, b = graph.indptr[row], graph.indptr[row + 1]
        values = graph.data[a:b]
        order = np.argsort(-values if largest else values)
        keep = np.zeros(b - a, dtype=bool)
        keep[order[:k]] = True
        values[~keep] = 0
    graph.eliminate_zeros()
    return graph


def vel_moment_weights(connectivities, n_neighbors=None, stored_k=None):
    """The moments' weights (scVelo's get_connectivities): each neighbour and the cell itself
    equally. ``n_neighbors`` under what the graph holds keeps each row's strongest."""
    C = connectivities
    if n_neighbors is not None and stored_k is not None and n_neighbors < stored_k:
        C = vel_top_per_row(C, n_neighbors, largest=True)
    B = sp.csr_matrix(C > 0, dtype=np.float32) + sp.identity(C.shape[0], dtype=np.float32, format="csr")
    B = sp.csr_matrix(B)
    B.data[:] = 1.0
    sizes = np.asarray(B.sum(axis=1)).ravel()
    return (sp.diags(1.0 / sizes, format="csr", dtype=np.float32) @ B).tocsr()


def vel_knn_indices(distances):
    """cells x k positions of each cell's nearest neighbours (scVelo's get_indices)."""
    D = sp.csr_matrix(distances, copy=True)
    D.data = D.data + 1e-6
    k = int(np.diff(D.indptr).min())
    D = vel_top_per_row(D, k, largest=False)
    return D.indices.reshape((-1, k)).astype(np.int64)


def vel_moments(weights, matrix, out, start=0, stop=None, block=2048):
    """weights @ matrix, dense, into ``out``, a block of cells at a time."""
    n = weights.shape[0]
    stop = n if stop is None else stop
    for a in range(start, stop, block):
        b = min(a + block, stop)
        out[a:b] = (weights[a:b] @ matrix).toarray().astype(np.float32)


def vel_quantile_fit(Ms, Mu, perc=95):
    """Gamma through the origin on the cells in the top of scaled S + U, and r2 of that fit
    over all cells (scvelo.core.LinearRegression, Velocity.compute_deterministic)."""
    norm = Ms / np.clip(Ms.max(axis=0), 1e-3, None) + Mu / np.clip(Mu.max(axis=0), 1e-3, None)
    keep = norm >= np.percentile(norm, perc, axis=0)
    x = np.asarray(Ms, dtype=np.float64) * keep
    y = np.asarray(Mu, dtype=np.float64) * keep
    with np.errstate(divide="ignore", invalid="ignore"):
        gamma = (x * y).sum(axis=0) / (x * x).sum(axis=0)
    gamma = np.nan_to_num(gamma, nan=0.0, posinf=0.0, neginf=0.0)
    residual = Mu - gamma.astype(np.float32) * Ms
    total = Mu - Mu.mean(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = 1.0 - np.square(residual, dtype=np.float64).sum(axis=0) / np.square(total, dtype=np.float64).sum(axis=0)
    return gamma, np.nan_to_num(r2, nan=0.0), residual.std(axis=0)


def vel_stochastic_gamma(Ms, Mu, Mss, Mus, res_std, perc=95):
    """Gamma from first and second moments by generalised least squares (scVelo's
    compute_stochastic and leastsq_generalized, no offsets)."""
    var_ss = 2 * Mss - Ms
    cov_us = 2 * Mus + Mu
    with np.errstate(divide="ignore", invalid="ignore"):
        gamma2 = np.nan_to_num((var_ss * cov_us).sum(axis=0, dtype=np.float64)
                               / (var_ss * var_ss).sum(axis=0, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        res2_std = (cov_us - gamma2.astype(np.float32) * var_ss).std(axis=0)
        joint = Ms / np.clip(Ms.max(axis=0), 1e-3, None) + Mu / np.clip(Mu.max(axis=0), 1e-3, None)
        keep = (joint >= np.percentile(joint, perc, axis=0)) | (Ms >= np.percentile(Ms, perc, axis=0))
        x1 = np.asarray(Ms * keep, dtype=np.float64) / res_std
        y1 = np.asarray(Mu * keep, dtype=np.float64) / res_std
        x2 = np.asarray(var_ss, dtype=np.float64) / res2_std
        y2 = np.asarray(cov_us, dtype=np.float64) / res2_std
        gamma = ((x1 * y1).sum(axis=0) + (x2 * y2).sum(axis=0)) / ((x1 * x1).sum(axis=0) + (x2 * x2).sum(axis=0))
    return np.nan_to_num(gamma, nan=0.0, posinf=0.0, neginf=0.0)


def vel_velocity_genes(gamma, r2, Ms_any, Mu_any, min_r2=0.01, min_ratio=0.01):
    """Velocity genes (scVelo's compute_deterministic), with its fallback on very low signal."""
    genes = (r2 > min_r2) & (gamma > min_ratio) & Ms_any & Mu_any
    if int(genes.sum()) < 2:
        genes = r2 > np.percentile(r2, 80)
    return genes


def vel_neighbour_sets(indices):
    """Cell i's row: the cell, its neighbours, and theirs (scVelo's get_iterative_indices at 2
    recursions) as the sparse matrix whose square gives them; ``recursed[a:b] @ recursed``."""
    n, k = indices.shape
    rows = np.repeat(np.arange(n), k)
    B = sp.csr_matrix((np.ones(n * k, dtype=np.int32), (rows, indices.ravel())), shape=(n, n))
    return (B + sp.identity(n, dtype=np.int32, format="csr")).tocsr()


def vel_cosine_rows(X, V, B, a, b, sqrt_transform):
    """Rows a..b of the signed cosine graph (scVelo's _compute_cosines): CSR, b - a rows."""
    reach = (B[a:b] @ B).tocsr()
    reach.sort_indices()
    indptr = [0]
    cols = []
    vals = []
    for r, i in enumerate(range(a, b)):
        vi_ = V[i]
        norm_v = float(np.sqrt(np.dot(vi_, vi_)))
        if norm_v == 0:
            indptr.append(indptr[-1])
            continue
        idx = reach.indices[reach.indptr[r]:reach.indptr[r + 1]]
        idx = idx[idx != i]
        dX = X[idx] - X[i]
        if sqrt_transform:
            dX = np.sqrt(np.abs(dX)) * np.sign(dX)
        dX = dX - dX.mean(axis=1, keepdims=True)
        norms = np.sqrt(np.einsum("ij,ij->i", dX, dX))
        with np.errstate(divide="ignore", invalid="ignore"):
            cos = dX.dot(vi_) / (norms * norm_v)
        cos = np.nan_to_num(cos, nan=0.0, posinf=0.0, neginf=0.0)
        ok = cos != 0
        cols.append(idx[ok])
        vals.append(cos[ok].astype(np.float32))
        indptr.append(indptr[-1] + int(ok.sum()))
    data = np.concatenate(vals) if vals else np.zeros(0, dtype=np.float32)
    indices = np.concatenate(cols) if cols else np.zeros(0, dtype=np.int32)
    return sp.csr_matrix((data, indices, np.asarray(indptr, dtype=np.int64)), shape=(b - a, X.shape[0]))


def vel_self_probability(best_cosine):
    """Each cell's self-transition (scVelo's transition_matrix): how far under the 98th
    percentile its best positive cosine is."""
    return np.clip(np.percentile(best_cosine, 98) - best_cosine, 0, 1)


def vel_project_rows(graph_rows, row_ids, self_prob, emb, scale=10.0):
    """Arrows of these cells on ``emb`` (scv.tl.velocity_embedding with autoscale off): the
    transition matrix row by row, its self transition counted in the row's normalisation."""
    coo = sp.coo_matrix(graph_rows)
    r, c, v = coo.row, coo.col, coo.data.astype(np.float64)
    n = graph_rows.shape[0]
    out = np.zeros((n, emb.shape[1]))
    if not r.size:
        return out
    t = np.where(v > 0, np.expm1(v * scale), -np.expm1(-v * scale))
    total = np.bincount(r, weights=np.abs(t), minlength=n) + np.expm1(self_prob[row_ids] * scale)
    p = t / total[r]
    dX = emb[c] - emb[row_ids[r]]
    with np.errstate(divide="ignore", invalid="ignore"):
        dX = dX / np.sqrt((dX * dX).sum(axis=1, keepdims=True))
    dX = np.nan_to_num(dX, nan=0.0)
    mean_p = np.bincount(r, weights=p, minlength=n) / np.maximum(np.bincount(r, minlength=n), 1)
    w = p - mean_p[r]
    for d in range(emb.shape[1]):
        out[:, d] = np.bincount(r, weights=w * dX[:, d], minlength=n)
    return out


def vel_confidence(V, indices, block=256, alloc=np.zeros):
    """velocity_length and velocity_confidence (scv.tl.velocity_confidence): each cell's velocity
    against its neighbours', correlation over the velocity genes, floored at 0."""
    n = V.shape[0]
    unit = alloc(V.shape, dtype=np.float32)
    norm = np.zeros(n, dtype=np.float32)
    for a in range(0, n, block):
        Vc = np.asarray(V[a:a + block], dtype=np.float32)
        Vc = Vc - Vc.mean(axis=1, keepdims=True)
        norm[a:a + block] = np.sqrt(np.einsum("ij,ij->i", Vc, Vc))
        with np.errstate(divide="ignore", invalid="ignore"):
            unit[a:a + block] = np.nan_to_num(Vc / norm[a:a + block, None], nan=0.0, posinf=0.0, neginf=0.0)
    R = np.zeros(n)
    for a in range(0, n, block):
        R[a:a + block] = np.einsum("ikj,ij->ik", np.asarray(unit[indices[a:a + block]]),
                                   np.asarray(unit[a:a + block])).mean(axis=1)
    return norm.round(2), np.clip(R, 0, None)


def vel_grid(X_emb, V_emb, density=1.0, smooth=0.5, min_mass=1.0, max_neighbours=1000):
    """Arrows on a grid (scVelo's compute_velocity_on_grid, autoscale off): (X_grid, V_grid).
    scVelo gives each grid point the nearest n_cells / 50 cells under a Gaussian of half a grid
    step, so far cells weigh nothing: past 50,000 cells the nearest ``max_neighbours``."""
    from sklearn.neighbors import NearestNeighbors

    ok = np.isfinite(X_emb.sum(axis=1) + V_emb.sum(axis=1))
    X_emb, V_emb = X_emb[ok], V_emb[ok]
    n_obs = X_emb.shape[0]
    grids = []
    for d in range(X_emb.shape[1]):
        lo, hi = X_emb[:, d].min(), X_emb[:, d].max()
        lo = lo - 0.01 * abs(hi - lo)
        hi = hi + 0.01 * abs(hi - lo)
        grids.append(np.linspace(lo, hi, int(50 * density)))
    X_grid = np.vstack([m.flat for m in np.meshgrid(*grids)]).T
    k = min(max(1, int(n_obs / 50)), max_neighbours, n_obs)
    nn = NearestNeighbors(n_neighbors=k, n_jobs=1).fit(X_emb)
    scale = np.mean([g[1] - g[0] for g in grids]) * smooth
    mass = np.zeros(X_grid.shape[0])
    V_grid = np.zeros(X_grid.shape)
    for a in range(0, X_grid.shape[0], 512):  # k x 512 x dims values per block
        b = a + 512
        dists, neighs = nn.kneighbors(X_grid[a:b])
        weight = np.exp(-0.5 * (dists / scale) ** 2) / (scale * np.sqrt(2 * np.pi))
        mass[a:b] = weight.sum(axis=1)
        V_grid[a:b] = (V_emb[neighs] * weight[:, :, None]).sum(axis=1)
    V_grid /= np.maximum(1, mass)[:, None]
    keep = mass > min_mass * np.percentile(mass, 99) / 100
    return X_grid[keep], V_grid[keep]


REPLAY_CODE = (vel_row_scale, vel_select_genes, vel_top_per_row, vel_moment_weights, vel_knn_indices, vel_moments,
               vel_quantile_fit, vel_stochastic_gamma, vel_velocity_genes, vel_neighbour_sets, vel_cosine_rows,
               vel_self_probability, vel_project_rows, vel_confidence)


REPLAY_CONSTANTS = ("MIN_R2", "MIN_RATIO", "PERCENTILE", "SCALE")


# ---------------------------------------------------------------------------
# A run: prepared inputs -> streamed stages -> outputs
# ---------------------------------------------------------------------------


@dataclass
class Params:
    method: str = "stochastic"
    n_top_genes: int = N_TOP_GENES
    n_neighbors: int = N_NEIGHBORS
    n_pcs: int = N_PCS
    min_shared_counts: int = MIN_SHARED_COUNTS
    group_by: str | None = None


@dataclass
class Prepared:
    dataset_id: str
    data_dir: str
    params: Params
    store_rows: np.ndarray  # the store's row of each cell with counts, in the dataset's order
    positions: np.ndarray  # those cells' positions in the dataset
    names: list[str]
    n_dataset_cells: int
    fingerprint: str
    pca: np.ndarray | None
    pca_note: str
    connectivities: Any
    distances: Any
    stored_k: int | None
    neighbours_note: str
    embeddings: dict[str, np.ndarray]
    groups: pd.Series | None
    group_column: str | None
    source: dict[str, Any] = field(default_factory=dict)
    var_names: list[str] = field(default_factory=list)


def embedding_names(adata) -> list[str]:
    """obsm keys arrows can be drawn on: X_* with two columns, and X_pca (its first two)."""
    out = []
    for key in adata.obsm.keys():
        if not str(key).startswith("X_"):
            continue
        shape = np.shape(adata.obsm[key])
        if len(shape) == 2 and shape[0] == adata.n_obs and (shape[1] == 2 or key == "X_pca"):
            out.append(str(key))
    return out


def obsm_key(embedding: str) -> str:
    """scVelo's name: X_umap's arrows are ``velocity_umap``."""
    return "velocity_" + (embedding[2:] if embedding.startswith("X_") else embedding)


def _stored_neighbours(adata, wanted: int, positions: np.ndarray) -> tuple[Any, Any, int | None, str | None]:
    """(connectivities, distances, k, why not): the dataset's graph if it covers the run."""
    if "connectivities" not in adata.obsp or "distances" not in adata.obsp:
        return None, None, None, "the dataset has no neighbour graph"
    conn, dist = adata.obsp["connectivities"], adata.obsp["distances"]
    params = (adata.uns.get("neighbors") or {}).get("params") if hasattr(adata.uns, "get") else None
    k = params.get("n_neighbors") if hasattr(params, "get") else None
    if k is None:
        return None, None, None, "the dataset's graph does not say its k"
    k = int(np.ravel(k)[0])  # a pre-0.7 file holds it as a one-value array
    if k < wanted:
        return None, None, None, f"the dataset's graph has k = {k}"
    if positions.size != adata.n_obs:
        return None, None, None, "the dataset's graph includes cells without velocity counts"
    conn, dist = sp.csr_matrix(conn), sp.csr_matrix(dist)
    if np.diff(dist.indptr).min() == 0:  # a batch-balanced graph, say: scVelo refuses it as corrupted
        return None, None, None, "a cell in the dataset's graph has no neighbours"
    return conn, dist, k, None


def prepare(adaptor, data_dir: str | Path, dataset_id: str, params: Params) -> Prepared:
    """Copy out what the run needs (no X); refuses what cannot run. In a worker thread."""
    if params.method not in METHODS:
        raise VelocityError(f"Unknown velocity method '{params.method}': choose {' or '.join(METHODS)}.")
    adata = adaptor.adata
    path = vi.store_path(data_dir, dataset_id)
    if not path.exists():
        raise VelocityError("This dataset has no spliced and unspliced counts yet: attach them first (a velocyto "
                            ".loom, an .h5ad with those layers, or a STARsolo Velocyto folder).")
    with ms.ModalityStore(path) as store:
        rows = store.rows_of(adata.obs_names)
        var_names = list(map(str, store.features))
    mask = rows >= 0
    positions = np.flatnonzero(mask)
    if positions.size < MIN_CELLS:
        raise VelocityError(f"Only {positions.size:,} of this dataset's {adata.n_obs:,} cells have velocity counts; a "
                            f"run needs at least {MIN_CELLS}.")
    if adata.n_obs - positions.size:
        logger.info("Velocity on %s: %d cells have no counts", dataset_id, adata.n_obs - positions.size)
    pca = None
    pca_note = ""
    if "X_pca" in adata.obsm and np.shape(adata.obsm["X_pca"])[1] >= params.n_pcs:
        pca = np.asarray(adata.obsm["X_pca"])[positions, :params.n_pcs].astype(np.float32)
        pca_note = f"The dataset's PCA ({params.n_pcs} PCs)"
    conn, dist, k, why = _stored_neighbours(adata, params.n_neighbors, positions)
    if conn is not None:
        note = f"Neighbours: the dataset's stored graph (k = {k})."
    else:
        on = f"the dataset's PCA ({params.n_pcs} PCs)" if pca is not None else (
            f"a PCA of normalised spliced counts ({params.n_pcs} PCs)")
        note = f"Neighbours: {params.n_neighbors} computed for this run on {on}, because {why}; its own graph is unchanged."
    embeddings = {e: np.asarray(adata.obsm[e])[:, :2].astype(np.float64) for e in embedding_names(adata)}
    column = params.group_by or adaptor.active_clustering_column()
    groups = None
    if column is not None:
        if column not in adata.obs.columns:
            raise VelocityError(f"This dataset has no column '{column}' to group the table by.")
        groups = adata.obs[column].astype(object).where(adata.obs[column].notna(), None).astype(str)
    return Prepared(dataset_id, str(data_dir), params, rows[mask], positions, [str(adata.obs_names[i]) for i in positions],
                    int(adata.n_obs), adaptor.cells_fingerprint(), pca, pca_note, conn, dist, k, note, embeddings, groups,
                    column, vi.read_record(adata).get("source") or {}, var_names)


def fit_block_genes(n_cells: int) -> int:
    """Genes fitted at once: FIT_BLOCK_GENES, fewer past 60,000 cells to bound the block's copies."""
    return int(np.clip(FIT_BLOCK_BYTES // max(n_cells * 35, 1), 16, FIT_BLOCK_GENES))


def _alloc(shape: tuple[int, int], folder: Path, name: str):
    """A float32 array, memory-mapped beside the store past MEMMAP_BYTES."""
    if shape[0] * shape[1] * 4 <= MEMMAP_BYTES:
        return np.zeros(shape, dtype=np.float32)
    folder.mkdir(parents=True, exist_ok=True)
    return np.memmap(folder / f"{name}.bin", dtype=np.float32, mode="w+", shape=shape)


def _progress(stage: str, message: str, **extra) -> dict[str, Any]:
    return {"type": "progress", "step": stage, "message": message, **extra}


def _next_path(p: Prepared, name: str) -> Path:
    return ms.store_path(p.data_dir, p.dataset_id, f"{name}.next")


def _pca_of(S_hvg: sp.csr_matrix, n_pcs: int) -> np.ndarray:
    """scVelo's internal PCA: log1p of normalised spliced on the variable genes, arpack."""
    import anndata as ad
    import scanpy as sc

    X = S_hvg.copy()
    X.data = np.log1p(X.data)
    tmp = ad.AnnData(X)
    n = min(n_pcs, X.shape[1] - 1, X.shape[0] - 1)
    sc.pp.pca(tmp, n_comps=n, svd_solver="arpack", random_state=0)
    return np.asarray(tmp.obsm["X_pca"], dtype=np.float32)


_knn_compiled = False  # numba compiles the neighbour search on first use in a process


def knn_compiled() -> bool:
    return _knn_compiled


def _neighbours_of(pca: np.ndarray, n_neighbors: int, n_pcs: int) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    """As scv.pp.neighbors: scanpy's graph, method umap, seed 0, on this PCA."""
    import anndata as ad
    import scanpy as sc

    global _knn_compiled

    tmp = ad.AnnData(sp.csr_matrix((pca.shape[0], 1), dtype=np.float32), obsm={"X_pca": pca})
    sc.pp.neighbors(tmp, n_neighbors=n_neighbors, n_pcs=min(n_pcs, pca.shape[1]), use_rep="X_pca", random_state=0,
                    method="umap", metric="euclidean")
    _knn_compiled = True
    return sp.csr_matrix(tmp.obsp["connectivities"]), sp.csr_matrix(tmp.obsp["distances"])


def iter_run(p: Prepared, stop: Callable[[], bool] | None = None) -> Iterator[dict[str, Any]]:
    """The run, in stages: yields progress events and, last, ``{"type": "result", "out": ...}``.
    Nothing in the dataset changes here; the two ``.next`` stores become current at ``save``."""
    stopped = stop or (lambda: False)

    def check() -> None:
        if stopped():
            raise VelocityCancelled

    params = p.params
    n = p.store_rows.size
    work = ms.store_dir(p.data_dir, p.dataset_id) / f".velocity-work-{uuid.uuid4().hex[:8]}"
    run_next, fit_next = _next_path(p, RUN_STORE), _next_path(p, FIT_STORE)
    try:
        yield _progress("scan", f"Reading spliced and unspliced counts of {n:,} cells")
        with ms.ModalityStore(vi.store_path(p.data_dir, p.dataset_id)) as store:
            n_genes = store.n_vars
            size_s, size_u = np.zeros(n), np.zeros(n)
            shared = np.zeros(n_genes)
            for a in range(0, n, BLOCK_CELLS):
                check()
                rows = p.store_rows[a:a + BLOCK_CELLS]
                S, U = store.read("spliced", rows), store.read("unspliced", rows)
                size_s[a:a + BLOCK_CELLS] = np.asarray(S.sum(axis=1), dtype=np.float64).ravel()
                size_u[a:a + BLOCK_CELLS] = np.asarray(U.sum(axis=1), dtype=np.float64).ravel()
                both = S.multiply(U > 0) + U.multiply(S > 0)
                shared += np.asarray(both.sum(axis=0), dtype=np.float64).ravel()
            kept = np.flatnonzero(shared >= params.min_shared_counts)
            if kept.size < MIN_GENES:
                raise VelocityError(f"Only {kept.size:,} of {n_genes:,} genes have {params.min_shared_counts}+ counts in "
                                    "cells with both spliced and unspliced: too few to fit. Check the file's unspliced "
                                    "layer.")
            scale_s, scale_u = vel_row_scale(size_s), vel_row_scale(size_u)
            yield _progress("genes", f"{kept.size:,} of {n_genes:,} genes pass the shared-counts filter")

            mean, mean_sq = np.zeros(kept.size), np.zeros(kept.size)
            for a in range(0, n, BLOCK_CELLS):
                check()
                S = store.read("spliced", p.store_rows[a:a + BLOCK_CELLS])[:, kept]
                S = sp.diags(scale_s[a:a + BLOCK_CELLS].astype(np.float32), format="csr") @ S
                mean += np.asarray(S.sum(axis=0), dtype=np.float64).ravel()
                mean_sq += np.asarray(S.multiply(S).sum(axis=0), dtype=np.float64).ravel()
            mean, mean_sq = mean / n, mean_sq / n
            hvg = kept[vel_select_genes(mean, mean_sq, n, params.n_top_genes)]
            genes = [str(store.features[i]) for i in hvg]

            S_parts, U_parts = [], []
            for a in range(0, n, BLOCK_CELLS):
                check()
                rows = p.store_rows[a:a + BLOCK_CELLS]
                S_parts.append(sp.diags(scale_s[a:a + BLOCK_CELLS].astype(np.float32), format="csr")
                               @ store.read("spliced", rows)[:, hvg])
                U_parts.append(sp.diags(scale_u[a:a + BLOCK_CELLS].astype(np.float32), format="csr")
                               @ store.read("unspliced", rows)[:, hvg])
        S = sp.vstack(S_parts, format="csr", dtype=np.float32)
        U = sp.vstack(U_parts, format="csr", dtype=np.float32)
        del S_parts, U_parts
        unspliced_fraction = size_u / np.maximum(size_u + size_s, 1)

        check()
        conn, dist, stored_k = p.connectivities, p.distances, p.stored_k
        pca_note = p.pca_note
        if conn is None:
            pca = p.pca
            if pca is None:
                yield _progress("pca", f"PCA of normalised spliced counts on {len(genes):,} genes")
                pca = _pca_of(S, params.n_pcs)
                pca_note = f"PCA computed for this run ({pca.shape[1]} PCs on normalised spliced counts)"
            check()
            yield _progress("neighbours", f"{params.n_neighbors} neighbours on {pca.shape[1]} PCs")
            conn, dist = _neighbours_of(pca, params.n_neighbors, params.n_pcs)
            stored_k = params.n_neighbors
        W1 = vel_moment_weights(conn, params.n_neighbors, stored_k)
        W2 = vel_moment_weights(conn)
        indices = vel_knn_indices(dist)

        check()
        yield _progress("moments", f"First moments on {len(genes):,} genes")
        shape = (n, len(genes))
        Ms, Mu = _alloc(shape, work, "Ms"), _alloc(shape, work, "Mu")
        for a in range(0, n, BLOCK_CELLS):
            check()
            b = min(a + BLOCK_CELLS, n)
            vel_moments(W1, S, Ms, a, b)
            vel_moments(W1, U, Mu, a, b)

        check()
        yield _progress("fit", f"Fitting gamma per gene ({METHOD_NAMES[params.method]})")
        m = len(genes)
        gamma, r2, res_std = np.zeros(m), np.zeros(m), np.zeros(m)
        ms_any, mu_any = np.zeros(m, dtype=bool), np.zeros(m, dtype=bool)
        fit_block = fit_block_genes(n)
        for a in range(0, m, fit_block):
            check()
            b = min(a + fit_block, m)
            xs, xu = np.asarray(Ms[:, a:b]), np.asarray(Mu[:, a:b])
            gamma[a:b], r2[a:b], res_std[a:b] = vel_quantile_fit(xs, xu, PERCENTILE)
            ms_any[a:b], mu_any[a:b] = (xs > 0).any(axis=0), (xu > 0).any(axis=0)
        velocity_genes = vel_velocity_genes(gamma, r2, ms_any, mu_any, MIN_R2, MIN_RATIO)
        vidx = np.flatnonzero(velocity_genes)
        if params.method == "stochastic" and vidx.size:
            S_v, U_v = S[:, vidx].tocsr(), U[:, vidx].tocsr()
            SS, US = S_v.multiply(S_v).tocsr(), S_v.multiply(U_v).tocsr()
            del S_v, U_v
            for a in range(0, vidx.size, fit_block):
                check()
                g = vidx[a:a + fit_block]
                sel = slice(a, a + fit_block)
                Mss = np.zeros((n, g.size), dtype=np.float32)
                Mus = np.zeros((n, g.size), dtype=np.float32)
                vel_moments(W2, SS[:, sel], Mss)
                vel_moments(W2, US[:, sel], Mus)
                gamma[g] = vel_stochastic_gamma(np.asarray(Ms[:, g]), np.asarray(Mu[:, g]), Mss, Mus, res_std[g],
                                                PERCENTILE)
            del SS, US
        if vidx.size < 2:
            raise VelocityError("Too few genes show splicing dynamics to fit velocity here "
                                f"({vidx.size} velocity gene{'s' if vidx.size != 1 else ''}).")
        check()
        yield _progress("graph", f"Velocity graph over {vidx.size:,} velocity genes")
        X = _alloc((n, vidx.size), work, "X")
        V = _alloc((n, vidx.size), work, "V")
        gamma_v = gamma[vidx].astype(np.float32)
        for a in range(0, n, BLOCK_CELLS):
            xs = np.asarray(Ms[a:a + BLOCK_CELLS][:, vidx])
            xu = np.asarray(Mu[a:a + BLOCK_CELLS][:, vidx])
            X[a:a + BLOCK_CELLS] = xs
            V[a:a + BLOCK_CELLS] = xu - gamma_v * xs
        del Ms, Mu
        Vt = _alloc((n, vidx.size), work, "Vt")
        sqrt_transform = params.method == "stochastic"
        for a in range(0, n, BLOCK_CELLS):
            v = np.asarray(V[a:a + BLOCK_CELLS])
            if sqrt_transform:
                v = np.sqrt(np.abs(v)) * np.sign(v)
            Vt[a:a + BLOCK_CELLS] = v - np.nanmean(v, axis=1, keepdims=True)
        B = vel_neighbour_sets(indices)
        run_next.parent.mkdir(parents=True, exist_ok=True)
        best = np.zeros(n)
        with ms.StoreWriter(run_next, p.names, p.names, attrs={"embedding_note": "signed cosine"}) as w:
            for a in range(0, n, BLOCK_CELLS):
                b = min(a + BLOCK_CELLS, n)
                w.append("connectivities", W1[a:b])
            for a in range(0, n, BLOCK_CELLS):
                check()
                b = min(a + BLOCK_CELLS, n)
                rows_g = vel_cosine_rows(X, Vt, B, a, b, sqrt_transform)
                w.append("graph", rows_g)
                best[a:b] = np.maximum(np.asarray(rows_g.max(axis=1).todense()).ravel(), 0)
                if (a // BLOCK_CELLS) % 4 == 3:
                    yield _progress("graph", f"Velocity graph: {b:,} of {n:,} cells", done=b, total=n)
        del Vt, X
        ms.write(fit_next, {"spliced": S.T.tocsr(), "unspliced": U.T.tocsr()}, genes, p.names,
                 attrs={"gene_row": True})

        check()
        yield _progress("confidence", "Velocity length and confidence")
        length, confidence = vel_confidence(V, indices, alloc=lambda shape, dtype: _alloc(shape, work, "unit"))
        check()
        yield _progress("project", f"Projecting onto {len(p.embeddings)} embedding{'s' if len(p.embeddings) != 1 else ''}")
        self_prob = vel_self_probability(best)
        arrows: dict[str, np.ndarray] = {}
        with ms.ModalityStore(run_next) as run_store:
            for name, emb in p.embeddings.items():
                check()
                arrows[name] = _project(run_store, np.arange(n), self_prob, emb[p.positions])
        table = _cluster_table(p, confidence, length, size_u, size_s, V, U, vidx, genes)
        out = {"genes": genes, "gamma": gamma, "r2": r2, "velocity_genes": velocity_genes, "n_kept": int(kept.size),
               "n_store_genes": int(n_genes), "length": length, "confidence": confidence,
               "unspliced_fraction": unspliced_fraction, "arrows": arrows, "table": table,
               "self_prob": self_prob, "pca_note": pca_note, "sum_u": float(size_u.sum()), "sum_s": float(size_s.sum()),
               "n_velocity_genes": int(vidx.size)}
        del V, U, S
        yield {"type": "result", "out": out}
    except BaseException:
        run_next.unlink(missing_ok=True)
        fit_next.unlink(missing_ok=True)
        raise
    finally:
        shutil.rmtree(work, ignore_errors=True)
        gc.collect()


def _project(run_store: ms.ModalityStore, rows: np.ndarray, self_prob: np.ndarray, emb: np.ndarray) -> np.ndarray:
    """Arrows of the run's cells on ``emb``, reading the graph a block of cells at a time."""
    out = np.zeros((rows.size, emb.shape[1]))
    for a in range(0, rows.size, BLOCK_CELLS):
        ids = rows[a:a + BLOCK_CELLS]
        out[a:a + BLOCK_CELLS] = vel_project_rows(run_store.read("graph", ids), ids, self_prob, emb, SCALE)
    return out


def _cluster_table(p: Prepared, confidence, length, size_u, size_s, V, U, vidx, genes) -> list[dict[str, Any]]:
    """Per group: cells, coherence (mean confidence), mean length, unspliced share of the raw
    counts, velocity genes with unspliced counts in 10%+ of its cells, and the highest mean velocities."""
    if p.groups is None:
        labels = np.full(p.positions.size, "all cells", dtype=object)
    else:
        labels = p.groups.iloc[p.positions].to_numpy(dtype=object)
    names, codes = np.unique(labels.astype(str), return_inverse=True)
    n_groups = names.size
    sums = np.zeros((n_groups, vidx.size))
    detected = np.zeros((n_groups, vidx.size))
    seen = (U[:, vidx] > 0).astype(np.float32).tocsr()
    for a in range(0, V.shape[0], BLOCK_CELLS):
        code = codes[a:a + BLOCK_CELLS]
        block = np.asarray(V[a:a + BLOCK_CELLS])
        np.add.at(sums, code, block)
        one_hot = sp.csr_matrix((np.ones(code.size, dtype=np.float32), (code, np.arange(code.size))),
                                shape=(n_groups, code.size))
        detected += (one_hot @ seen[a:a + BLOCK_CELLS]).toarray()
    out = []
    for g, name in enumerate(names):
        mask = codes == g
        k = int(mask.sum())
        mean_v = sums[g] / max(k, 1)
        order = np.argsort(-mean_v)[:TOP_GENES]
        us, ss = float(size_u[mask].sum()), float(size_s[mask].sum())
        out.append({
            "group": str(name), "n_cells": k, "coherence": round(float(np.mean(confidence[mask])), 4),
            "length": round(float(np.mean(length[mask])), 3), "unspliced_fraction": round(us / max(us + ss, 1), 4),
            "n_velocity_genes": int((detected[g] >= GENE_DETECTED_SHARE * k).sum()),
            "top_genes": [genes[vidx[i]] for i in order if mean_v[i] > 0],
        })
    out.sort(key=lambda r: -r["n_cells"])
    return out


def warnings_for(unspliced: float, median_confidence: float, n_velocity_genes: int, cells_without: int) -> list[str]:
    """What the user needs before reading the arrows; thresholds from scVelo's guidance."""
    out = []
    if unspliced < LOW_UNSPLICED:
        out.append(f"Only {unspliced:.1%} of the counts are unspliced (scVelo's documentation calls 10-25% typical; "
                   f"under {LOW_UNSPLICED:.0%} is flagged here), so the fit has little to work on: read the arrows with "
                   "caution, or check how the counts were made.")
    if median_confidence < LOW_CONFIDENCE:
        out.append(f"Median velocity confidence is {median_confidence:.2f} (scVelo's pancreas and dentate gyrus examples: "
                   f"0.72 to 0.85; under {LOW_CONFIDENCE} is flagged here): neighbouring cells disagree on direction, so "
                   "the arrows may be noise.")
    if n_velocity_genes < FEW_VELOCITY_GENES:
        out.append(f"Only {n_velocity_genes} genes pass as velocity genes: direction rests on few genes.")
    if cells_without:
        out.append(f"{cells_without:,} cells have no velocity counts and no arrow.")
    return out


# ---------------------------------------------------------------------------
# Saving and reading back
# ---------------------------------------------------------------------------


def _hash_embedding(emb: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(emb, dtype=np.float32).tobytes()).hexdigest()[:16]


def _pack_grid(X_grid: np.ndarray, V_grid: np.ndarray) -> dict[str, list[float]]:
    return {"x": np.round(X_grid[:, 0], 4).tolist(), "y": np.round(X_grid[:, 1], 4).tolist(),
            "u": [float(f"{v:.5g}") for v in V_grid[:, 0]], "v": [float(f"{v:.5g}") for v in V_grid[:, 1]]}


def clear_saved(adata) -> None:
    """Remove a run's columns from the dataset (new counts were attached)."""
    for col in (OBS_LENGTH, OBS_CONFIDENCE, OBS_UNSPLICED):
        if col in adata.obs.columns:
            del adata.obs[col]
    for col in (VAR_GAMMA, VAR_R2, VAR_GENE):
        if col in adata.var.columns:
            del adata.var[col]
    for key in [k for k in adata.obsm.keys() if str(k).startswith("velocity_")]:
        del adata.obsm[key]


def drop_run_files(data_dir: str | Path, dataset_id: str) -> None:
    for name in (RUN_STORE, FIT_STORE):
        ms.store_path(data_dir, dataset_id, name).unlink(missing_ok=True)


def drop_next_files(data_dir: str | Path, dataset_id: str) -> None:
    """A run's stores that never became current (its save was refused or failed)."""
    for name in (RUN_STORE, FIT_STORE):
        ms.store_path(data_dir, dataset_id, f"{name}.next").unlink(missing_ok=True)


def save(adata, p: Prepared, out: dict[str, Any], summary_extra: dict[str, Any], resources: dict | None = None,
         ) -> dict[str, Any]:
    """Write the columns, arrows, gene table and grid into the dataset (the caller saves it) and
    make the run's two stores current. Returns the run's summary."""
    from scview.core import provenance

    n_total = adata.n_obs
    full = np.full(n_total, np.nan, dtype=np.float32)
    for col, values in ((OBS_LENGTH, out["length"]), (OBS_CONFIDENCE, out["confidence"]),
                        (OBS_UNSPLICED, out["unspliced_fraction"])):
        arr = full.copy()
        arr[p.positions] = values
        adata.obs[col] = arr
    old = [k for k in adata.obsm.keys() if str(k).startswith("velocity_")]
    for key in old:
        del adata.obsm[key]
    grids = {}
    hashes = {}
    keys = {}
    for emb, arrows in out["arrows"].items():
        arr = np.full((n_total, 2), np.nan, dtype=np.float32)
        arr[p.positions] = arrows.astype(np.float32)
        key = obsm_key(emb)
        adata.obsm[key] = arr
        keys[emb] = key
        hashes[emb] = _hash_embedding(p.embeddings[emb])
        grids[emb] = _pack_grid(*vel_grid(p.embeddings[emb][p.positions], arrows, density=1.0))
    position = {g: i for i, g in enumerate(out["genes"])}
    gamma, r2, vg = out["gamma"], out["r2"], out["velocity_genes"]
    var_names = list(map(str, adata.var_names))
    v_gamma, v_r2 = np.full(len(var_names), np.nan, dtype=np.float32), np.full(len(var_names), np.nan, dtype=np.float32)
    v_gene = np.zeros(len(var_names), dtype=bool)
    in_var = 0
    for j, g in enumerate(var_names):
        i = position.get(g)
        if i is not None:
            v_gamma[j], v_r2[j], v_gene[j] = gamma[i], r2[i], bool(vg[i])
            in_var += 1
    adata.var[VAR_GAMMA], adata.var[VAR_R2], adata.var[VAR_GENE] = v_gamma, v_r2, v_gene
    conf = out["confidence"]
    unspliced = out["sum_u"] / max(out["sum_u"] + out["sum_s"], 1)
    params = p.params
    cells_without = int(n_total - p.positions.size)
    warnings = warnings_for(unspliced, float(np.median(conf)), out["n_velocity_genes"], cells_without)
    run = {
        **summary_extra,
        "method": params.method, "method_name": METHOD_NAMES[params.method],
        "params": {"n_top_genes": params.n_top_genes, "n_neighbors": params.n_neighbors, "n_pcs": params.n_pcs,
                   "min_shared_counts": params.min_shared_counts, "min_r2": MIN_R2, "min_ratio": MIN_RATIO,
                   "percentile": PERCENTILE, "scale": SCALE, "n_recurse_neighbors": 2},
        "fingerprint": p.fingerprint, "n_cells": int(p.positions.size), "n_dataset_cells": int(n_total),
        "cells_without": cells_without, "n_genes_store": out["n_store_genes"], "n_genes_kept": out["n_kept"],
        "n_genes_fit": len(out["genes"]), "n_velocity_genes": out["n_velocity_genes"], "genes_in_var": in_var,
        "median_confidence": round(float(np.median(conf)), 4), "median_length": round(float(np.median(out["length"])), 3),
        "unspliced_fraction": round(float(unspliced), 4), "neighbours_note": p.neighbours_note, "pca_note": out["pca_note"],
        "group_column": p.group_column, "clusters": out["table"], "warnings": warnings, "note": NOTE,
        "embeddings": sorted(out["arrows"]), "embedding_hashes": hashes, "obsm_keys": keys,
        "gene_table": {"gene": out["genes"], "gamma": [float(f"{g:.5g}") for g in gamma],
                       "r2": [round(float(x), 4) for x in r2], "velocity_gene": [bool(x) for x in vg]},
        "grid": grids,
    }
    record = vi.read_record(adata)
    adata.uns[UNS_KEY] = json.dumps({"source": record.get("source") or p.source, "run": run})
    for name in (RUN_STORE, FIT_STORE):
        next_path = _next_path(p, name)
        if next_path.exists():
            os.replace(next_path, ms.store_path(p.data_dir, p.dataset_id, name))
    try:
        src = record.get("source") or p.source
        provenance.record_step(
            adata, step=STEP, tool=f"scview velocity ({METHOD_NAMES[params.method]}, ported from scVelo 0.3.3)",
            params={
                "method": params.method, "n_top_genes": params.n_top_genes, "n_neighbors": params.n_neighbors,
                "n_pcs": params.n_pcs, "min_shared_counts": params.min_shared_counts,
                "source_kind": src.get("kind"), "source_file": src.get("file"), "source_stored_as": src.get("stored_as"),
                "source_files": src.get("files"), "group_column": p.group_column,
                "embeddings": sorted(out["arrows"]), "n_velocity_genes": out["n_velocity_genes"],
                "neighbours": p.neighbours_note,
                "outputs": [OBS_LENGTH, OBS_CONFIDENCE, OBS_UNSPLICED, *keys.values(), VAR_GAMMA, VAR_R2, VAR_GENE],
            },
            resources=resources,
        )
    except Exception as e:  # provenance must never block the result
        logger.warning("Could not record velocity provenance: %s", e)
    return run


def public_run(run: dict[str, Any] | None) -> dict[str, Any] | None:
    """A run for the panel: without the gene table and the grids (their own endpoints)."""
    if run is None:
        return None
    return {k: v for k, v in run.items() if k not in ("gene_table", "grid", "embedding_hashes")}


def saved(adata) -> dict[str, Any] | None:
    """The last run (from uns) while its columns are in the dataset."""
    run = vi.read_record(adata).get("run")
    if not isinstance(run, dict) or OBS_CONFIDENCE not in adata.obs.columns:
        return None
    return run


def status(adaptor, data_dir: str | Path, dataset_id: str) -> dict[str, Any]:
    """Whether counts are attached, what matched, and the last run; ``stale`` when the cells
    changed since it."""
    adata = adaptor.adata
    record = vi.read_record(adata)
    attached = vi.has_store(data_dir, dataset_id)
    run = saved(adata)
    stale = None
    if run is not None and run.get("fingerprint") != adaptor.cells_fingerprint():
        stale = (f"The cells changed since this run ({run.get('n_dataset_cells', 0):,} then, {adata.n_obs:,} now): run "
                 "velocity again.")
    source = record.get("source") or ({"kind": "store"} if attached else None)
    if attached and source is not None and "cells_without" not in source:
        source = {**source, "cells_without": None}
    if attached:
        with ms.ModalityStore(vi.store_path(data_dir, dataset_id)) as store:
            n_have = int((store.rows_of(adata.obs_names) >= 0).sum())
        source = {**(source or {}), "cells_now": n_have, "cells_without_now": int(adata.n_obs - n_have)}
    return {
        "attached": attached, "source": source, "run": public_run(run), "stale": stale, "methods": list(METHODS),
        "method_names": METHOD_NAMES, "defaults": {"method": METHODS[0], "n_top_genes": N_TOP_GENES,
                                                   "n_neighbors": N_NEIGHBORS, "n_pcs": N_PCS},
        "note": NOTE, "n_cells": int(adata.n_obs),
    }


_projected: dict[tuple[str, str, str, str], np.ndarray] = {}  # (dataset, run id, embedding, hash): arrows
_projected_lock = threading.Lock()


def arrows_for(adaptor, data_dir: str | Path, dataset_id: str, embedding: str) -> tuple[np.ndarray, np.ndarray, bool]:
    """(the embedding's coordinates, each cell's arrow, re-projected): the saved arrows when the
    embedding is the one the run saw, else projected now from the run's graph (a block pass;
    seconds; kept in memory, last 4). In a worker thread."""
    adata = adaptor.adata
    run = saved(adata)
    if run is None:
        raise VelocityError("Run velocity on this dataset first.", 404)
    if run.get("fingerprint") != adaptor.cells_fingerprint():
        raise VelocityError("The cells changed since the velocity run: run it again.", 409)
    if embedding not in adata.obsm or np.ndim(adata.obsm[embedding]) != 2 or np.shape(adata.obsm[embedding])[1] < 2:
        raise VelocityError(f"This dataset has no embedding '{embedding}'.", 404)
    if np.shape(adata.obsm[embedding])[1] > 2 and embedding != "X_pca":
        raise VelocityError(f"Arrows are drawn on 2D embeddings; '{embedding}' has "
                            f"{np.shape(adata.obsm[embedding])[1]} dimensions.", 422)
    emb = np.asarray(adata.obsm[embedding])[:, :2].astype(np.float64)
    digest = _hash_embedding(emb)
    key = obsm_key(embedding)
    if run.get("embedding_hashes", {}).get(embedding) == digest and key in adata.obsm:
        return emb, np.asarray(adata.obsm[key], dtype=np.float64), False
    cache_key = (dataset_id, str(run.get("result_id")), embedding, digest)
    with _projected_lock:
        hit = _projected.get(cache_key)
    if hit is not None:
        return emb, hit, True
    path = ms.store_path(data_dir, dataset_id, RUN_STORE)
    if not path.exists():
        raise VelocityError("The velocity run's graph is missing: run velocity again.", 409)
    with ms.ModalityStore(path) as store:
        rows = store.rows_of(adata.obs_names)
        mask = rows >= 0
        positions = np.flatnonzero(mask)
        best = np.zeros(store.n_obs)
        for a in range(0, store.n_obs, BLOCK_CELLS):
            ids = np.arange(a, min(a + BLOCK_CELLS, store.n_obs))
            block = store.read("graph", ids)
            best[ids] = np.maximum(np.asarray(block.max(axis=1).todense()).ravel(), 0)
        self_prob = vel_self_probability(best)
        order = np.full(store.n_obs, -1, dtype=np.int64)
        order[rows[mask]] = positions
        run_emb = emb[order]
        projected = _project(store, np.arange(store.n_obs), self_prob, run_emb)
    full = np.full((adata.n_obs, 2), np.nan)
    full[order] = projected
    with _projected_lock:
        if len(_projected) >= 4:
            _projected.pop(next(iter(_projected)))
        _projected[cache_key] = full
    return emb, full, True


def grid_for(adaptor, data_dir: str | Path, dataset_id: str, embedding: str, density: float = 1.0) -> dict[str, Any]:
    """The arrows on a grid for this embedding. The stored one at the default density."""
    run = saved(adaptor.adata)
    emb, arrows, reprojected = arrows_for(adaptor, data_dir, dataset_id, embedding)
    if not reprojected and density == 1.0 and embedding in (run or {}).get("grid", {}):
        packed = run["grid"][embedding]
    else:
        packed = _pack_grid(*vel_grid(emb, arrows, density=density, smooth=SMOOTH, min_mass=MIN_MASS,
                                      max_neighbours=MAX_GRID_NEIGHBOURS))
    u, v = np.asarray(packed["u"]), np.asarray(packed["v"])
    lengths = np.hypot(u, v)
    return {"embedding": embedding, "density": density, "n": len(packed["x"]), "reprojected": reprojected, **packed,
            "step": _grid_step(emb, density), "length_p95": float(np.percentile(lengths, 95)) if lengths.size else 0.0}


def _grid_step(emb: np.ndarray, density: float) -> float:
    ok = np.isfinite(emb).all(axis=1)
    if not ok.any():
        return 1.0
    span = emb[ok].max(axis=0) - emb[ok].min(axis=0)
    return float(span.mean() / max(GRID_POINTS * density - 1, 1))


def gene_table(adata) -> dict[str, Any]:
    run = saved(adata)
    if run is None:
        raise VelocityError("Run velocity on this dataset first.", 404)
    t = run["gene_table"]
    rows = [{"gene": g, "gamma": c, "r2": r, "velocity_gene": bool(v)}
            for g, c, r, v in zip(t["gene"], t["gamma"], t["r2"], t["velocity_gene"])]
    rows.sort(key=lambda r: (-int(r["velocity_gene"]), -r["r2"]))
    return {"genes": rows, "n_genes": len(rows), "n_velocity_genes": sum(1 for r in rows if r["velocity_gene"]),
            "sort": "velocity genes first, then by r2"}


def phase(adaptor, data_dir: str | Path, dataset_id: str, gene: str, color_by: str | None = None) -> dict[str, Any]:
    """A gene's phase portrait: Ms and Mu of up to PHASE_MAX_POINTS cells (a seeded sample past
    that), the fitted line, and the colour column's values. Reads one gene's row and those cells'
    connectivities, never a dense matrix."""
    adata = adaptor.adata
    run = saved(adata)
    if run is None:
        raise VelocityError("Run velocity on this dataset first.", 404)
    t = run["gene_table"]
    try:
        i = t["gene"].index(gene)
    except ValueError:
        raise VelocityError(f"'{gene}' was not among the {len(t['gene']):,} genes fitted: pick one from the table.", 404)
    fit_path, run_path = ms.store_path(data_dir, dataset_id, FIT_STORE), ms.store_path(data_dir, dataset_id, RUN_STORE)
    if not fit_path.exists() or not run_path.exists():
        raise VelocityError("The velocity run's files are missing: run velocity again.", 409)
    with ms.ModalityStore(fit_path) as fit:
        row = fit.rows_of([gene])
        s = fit.read("spliced", row)
        u = fit.read("unspliced", row)
    with ms.ModalityStore(run_path) as store:
        n = store.n_obs
        take = np.arange(n)
        if n > PHASE_MAX_POINTS:
            take = np.sort(np.random.default_rng(0).choice(n, PHASE_MAX_POINTS, replace=False))
        W = store.read("connectivities", take)
        barcodes = store.barcodes[take]
    s_col, u_col = s.T.tocsr(), u.T.tocsr()
    Ms = np.asarray((W @ s_col).todense()).ravel()
    Mu = np.asarray((W @ u_col).todense()).ravel()
    positions = adata.obs_names.get_indexer(barcodes)
    ok = positions >= 0
    color = _color_values(adata, color_by, positions[ok])
    return {"gene": gene, "gamma": t["gamma"][i], "r2": t["r2"][i], "velocity_gene": bool(t["velocity_gene"][i]),
            "spliced": [float(f"{x:.4g}") for x in Ms[ok]], "unspliced": [float(f"{x:.4g}") for x in Mu[ok]],
            "n_points": int(ok.sum()), "n_cells": int(n), "capped": bool(n > PHASE_MAX_POINTS),
            "max_points": PHASE_MAX_POINTS, "color": color, "method": run["method"]}


def _color_values(adata, column: str | None, positions: np.ndarray) -> dict[str, Any] | None:
    if not column or column not in adata.obs.columns:
        return None
    values = adata.obs[column].iloc[positions]
    if pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(values):
        return {"column": column, "kind": "numeric",
                "values": [None if not np.isfinite(v) else float(f"{v:.4g}") for v in values.to_numpy(dtype=np.float64)]}
    cat = pd.Categorical(values.astype(object).where(values.notna(), "NA").astype(str))
    full = adata.obs[column]
    order = list(map(str, full.cat.categories)) if isinstance(full.dtype, pd.CategoricalDtype) else list(map(str, cat.categories))
    index = {c: i for i, c in enumerate(order)}
    extras = [c for c in cat.categories if c not in index]
    for c in extras:
        index[c] = len(order)
        order.append(str(c))
    return {"column": column, "kind": "category", "categories": order,
            "codes": [index[str(c)] for c in cat.astype(str)]}


# ---------------------------------------------------------------------------
# Estimates, results store, the co-pilot
# ---------------------------------------------------------------------------


def stored_values(data_dir: str | Path, dataset_id: str) -> tuple[int, int, int, int]:
    """(cells, genes, spliced nnz, unspliced nnz) of the store, from its header."""
    with ms.ModalityStore(vi.store_path(data_dir, dataset_id)) as store:
        return store.n_obs, store.n_vars, store.nnz("spliced"), store.nnz("unspliced")


@dataclass
class Kept:
    id: str
    summary: dict[str, Any]
    cells: str = ""


_last: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_lock = threading.Lock()


def remember(adaptor, run: dict[str, Any], result_id: str) -> None:
    k = Kept(result_id, public_run(run) or {}, results_store.cells_hash(adaptor.adata))
    try:
        with _lock:
            _last[adaptor] = k
    except TypeError:  # a test double
        pass
    results_store.keep(adaptor, KIND, k, result_id=result_id, key="",
                       label=f"RNA velocity ({run['method']}, {run['n_velocity_genes']:,} velocity genes)")


def grounding_lines(adaptor) -> tuple[list[str], str | None]:
    """For the co-pilot: the last run's method, genes, confidence, and the warnings."""
    run = saved(adaptor.adata)
    if run is None:
        return [], None
    detail = (f"RNA velocity, {run['method_name']}: {run['n_velocity_genes']:,} velocity genes of {run['n_genes_fit']:,} "
              f"fitted, {run['n_cells']:,} cells, median confidence {run['median_confidence']:.2f}, unspliced "
              f"{run['unspliced_fraction']:.1%}")
    lines = [f"\n## RNA velocity (saved run) [{TAG}]", f"- {detail} [{TAG}]."]
    rows = sorted(run.get("clusters") or [], key=lambda r: -(r.get("coherence") or 0))
    if rows:
        lines.append(f"- By {run.get('group_column') or 'group'}, coherence (mean confidence): "
                     + "; ".join(f"{r['group']} {r['coherence']:.2f}" for r in rows[:12]) + ".")
        tops = [f"{r['group']}: {', '.join(r['top_genes'][:3])}" for r in rows[:6] if r.get("top_genes")]
        if tops:
            lines.append("- Highest mean velocity per group: " + "; ".join(tops) + ".")
    lines.extend(f"- Caution: {w}" for w in run.get("warnings") or [])
    lines.append("- Arrows show direction on a 2D plot, not a fate claim; the dynamical model and latent time are not "
                 "offered.")
    return lines, detail


def estimate_note(seconds: float) -> str:
    return f"about {math.ceil(seconds / 5) * 5:.0f} s" if seconds < 90 else f"about {math.ceil(seconds / 60):.0f} min"
