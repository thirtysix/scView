"""Spatial domains (by expression) and niches (by composition).

A spatial domain is a stretch of tissue whose cells look alike and whose surroundings
look alike too, such as a cortical layer or a tumour's invasive edge: clustering that
also sees each observation's neighbours, so it follows the tissue, not only cell type.

Domains: CellCharter (Varrone et al. 2024, Nat Genet 56:74, PMID 38066188), ported from
its source without torch (``gr/_build.py``, ``gr/_aggr.py``, ``tl/_gmm.py``,
``tl/_autok.py``; BSD 3-Clause License, Copyright (c) 2022, Marco Varrone):

- Graph, one per ``obs["library_id"]``, never between sections. Cells: Delaunay
  triangulation (``sq.gr.spatial_neighbors(coord_type="generic", delaunay=True)``) with
  links above the 99th percentile of the section's link lengths removed
  (``remove_long_links``; CellCharter takes the percentile over the whole dataset, here
  per section, as sections can differ in scale). Spots and bins: the capture grid
  (``spatial_autocorr.knn_edges``: 6 on Visium's hexagons, 4 on Visium HD's squares),
  made symmetric.
- Features (``aggregate_neighbors``): an observation's own vector, then the mean of its
  neighbours at exactly 1, 2, ... L hops (L = 3 by default), side by side, from sparse
  powers of the adjacency; no dense matrix. Vectors: the first ``n_pcs`` of ``X_pca``
  (``X_pca_harmony`` when present and there is more than one section). CellCharter uses
  a 10-dimensional scVI or trVAE latent space (both tools' default size), and a full
  covariance over (L + 1) x dimensions has to be estimated per domain, so 10 is the
  default here too. A ring that is empty (a cell in a small island) takes the nearest
  ring within reach, its own vector at worst; CellCharter leaves it at zero, and a
  mixture then spends a component on those rows, whose variance is zero.
- Clustering (``tl/_gmm.py``): a Gaussian mixture with full covariances, k-means
  initialisation, tolerance 1e-3, regularisation 1e-6 (times 10 on a failed Cholesky),
  sklearn's ``GaussianMixture`` in place of torchgmm. Above ``MAX_FIT_OBS`` observations
  it is fitted on a seeded sample and predicts all, and the result says so. Domains are
  numbered by size.
- Choosing k (``tl/_autok.py``, ``ClusterAutoK``): repeated fits per k, each k's
  stability the mean adjusted Rand index (CellCharter's default is Fowlkes-Mallows) of
  its runs against the other seeds' runs at k - 1 and k + 1, and the best local peak of
  that curve. k = 1 is not fitted, so k = 2 is compared with k = 3 only; an end of the
  range can never be a peak. Above ``MAX_FIT_OBS`` all fits share one seeded sample.

Niches, neither of which reads X:

- Neighbourhood enrichment: squidpy 1.6.5 ``nhood_enrichment`` (``gr/_nhood.py``, BSD
  3-Clause License, Copyright (c) 2021, Theislab): the links between each pair of labels
  on the same graph, against the same labels shuffled within each section, z =
  (count - mean) / sd over ``N_PERMS`` permutations (sd with ddof = 0, as squidpy).
- Composition niches (Schurch et al. 2020, Cell 182:1341): each cell's window is its
  k nearest cells in its section (itself included, as in the paper; k = 20 here), its
  features the label fractions in the window, and k-means on those (8 niches by default)
  gives ``obs["spatial_niche"]``. Refused on Visium spots and on Visium HD bins other
  than 8 um: a window of spots that each mix cells says little about composition.

Observations without a label are left out of both niche analyses; they stay NaN in
``spatial_niche``. No X read anywhere here: a large dataset needs only its tables.

``REPLAY_CODE`` uses numpy, scipy, pandas and sklearn, and is copied into the replay
script (with ``knn_edges`` from ``core/spatial_autocorr.py``).
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.signal import find_peaks
from scipy.spatial import Delaunay, QhullError, cKDTree
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture

from scview.core import spatial
from scview.core.spatial_autocorr import knn_edges

logger = logging.getLogger(__name__)

TAG = "result:spatial_domains"
DOMAINS_KEY = "scview_spatial_domains"  # uns, JSON: the runs, one per column
NICHES_KEY = "scview_spatial_niches"
ENRICHMENT_KEY = "scview_nhood_enrichment"
DOMAIN_COLUMN = "spatial_domain"
NICHE_COLUMN = "spatial_niche"
STEP_DOMAINS = "spatial_domains"
STEP_NICHES = "spatial_niches"
STEP_ENRICHMENT = "nhood_enrichment"
VERSION = 1

GRID_NEIGHBOURS = {"visium": 6, "visium_hd": 4}
LONG_LINK_PERCENTILE = 99.0  # CellCharter's remove_long_links
DEFAULT_HOPS = 3  # CellCharter's n_layers
MAX_HOPS = 6
DEFAULT_PCS = 10  # scVI's and trVAE's default latent size, CellCharter's input
MIN_PCS = 2
DEFAULT_K = 7
MAX_K = 40
SCAN_RANGE = (2, 15)
SCAN_REPEATS = 5
MAX_FIT_OBS = 50_000
SEED = 0
N_PERMS = 1000
MAX_PERMS = 10_000
PERM_BLOCK = 25  # permutations between a progress event and a stop check
NICHE_NEIGHBOURS = 20
MAX_NICHE_NEIGHBOURS = 200
DEFAULT_NICHES = 8
MAX_NICHES = 40
NICHE_N_INIT = 4
MAX_LABELS = 100
MIN_OBS = 50
HD_NICHE_BIN_UM = 8.0
HOLE_SHARE = 0.05  # QC removed at least this share of a grid's spots: say so


COLUMN_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.\-]{0,62}$")


class SpatialDomainsError(ValueError):
    """A run that cannot start (no positions, no PCA, a name in use): a 4xx with a sentence."""


class SpatialDomainsCancelled(Exception):
    """The client went away; the run stops before its next piece of work, nothing is saved."""


class ColumnInUse(SpatialDomainsError):
    """The name asked for holds another column, and the request did not say to replace it."""


# ---------------------------------------------------------------------------
# The computation (REPLAY_CODE: copied into the replay script)
# ---------------------------------------------------------------------------


def delaunay_edges(coords, percentile=99.0):
    """(rows, cols) of one section's Delaunay graph, both directions, without its links
    above ``percentile`` of the positive link lengths. Fewer than 4 points, or points on
    one line, have no triangulation: each is linked to its two nearest."""
    coords = np.asarray(coords, dtype=np.float64)
    n = coords.shape[0]
    if n < 2:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    try:
        if n < 4:
            raise QhullError("too few points")
        indptr, indices = Delaunay(coords).vertex_neighbor_vertices
    except QhullError:
        rows, cols = knn_edges(coords, 2, False)
        return np.concatenate([rows, cols]), np.concatenate([cols, rows])
    rows = np.repeat(np.arange(n, dtype=np.int64), np.diff(indptr))
    cols = np.asarray(indices, dtype=np.int64)
    dist = np.linalg.norm(coords[rows] - coords[cols], axis=1)
    if (dist > 0).any():
        keep = dist <= np.percentile(dist[dist > 0], percentile)
        rows, cols = rows[keep], cols[keep]
    return rows, cols


def spatial_graph(coords, libraries, grid_neighbours=None, percentile=99.0):
    """The symmetric neighbour graph (CSR, int8, no diagonal): one per section, no link
    between sections; Delaunay, or the capture grid with ``grid_neighbours`` per point."""
    coords = np.asarray(coords, dtype=np.float64)
    n = coords.shape[0]
    libraries = np.asarray(libraries)
    all_rows, all_cols = [], []
    for lib in pd.unique(libraries):
        where = np.flatnonzero(libraries == lib)
        if grid_neighbours:
            r, c = knn_edges(coords[where], grid_neighbours, True)
        else:
            r, c = delaunay_edges(coords[where], percentile)
        all_rows.append(where[r])
        all_cols.append(where[c])
    rows = np.concatenate(all_rows) if all_rows else np.zeros(0, dtype=np.int64)
    cols = np.concatenate(all_cols) if all_cols else np.zeros(0, dtype=np.int64)
    keep = rows != cols
    # Both directions, once each: a grid's cut can keep one direction of a link.
    pairs = np.unique(np.concatenate([rows[keep] * n + cols[keep], cols[keep] * n + rows[keep]]))
    rows, cols = pairs // n, pairs % n
    return sp.csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(n, n))


def aggregate_neighbors(adj, X, n_hops=3):
    """CellCharter's aggregate_neighbors: each row's own vector, then the mean of its
    neighbours at exactly 1, 2, ... ``n_hops`` hops, side by side (float32). An empty ring
    takes the ring before it (a row with no neighbour, its own vector)."""
    X = np.asarray(X, dtype=np.float32)
    n, d = X.shape
    out = np.empty((n, d * (n_hops + 1)), dtype=np.float32)
    out[:, :d] = X
    adj = sp.csr_matrix(adj, dtype=np.int32)
    visited = (adj + sp.identity(n, dtype=np.int32, format="csr")).tocsr()
    visited.data[:] = 1
    hop = adj.copy()
    for h in range(1, n_hops + 1):
        if h > 1:
            hop = (hop @ adj).tocsr()
            hop.data[:] = 1
            hop = (hop - hop.multiply(visited)).tocsr()
            hop.eliminate_zeros()
            visited = (visited + hop).tocsr()
            visited.data[:] = 1
        deg = np.asarray(hop.sum(axis=1)).ravel().astype(np.float32)
        scale = np.divide(1.0, deg, out=np.zeros_like(deg), where=deg > 0)
        ring = sp.diags(scale, format="csr", dtype=np.float32) @ hop.astype(np.float32)
        out[:, d * h:d * (h + 1)] = ring @ X
        empty = deg == 0
        if empty.any():
            out[empty, d * h:d * (h + 1)] = out[empty, d * (h - 1):d * h]
    return out


def seeded_sample(n, size, seed):
    """Sorted row numbers of a seeded sample; all rows when ``size`` is not below ``n``."""
    if n <= size:
        return np.arange(n)
    return np.sort(np.random.RandomState(seed).choice(n, size, replace=False))


def fit_mixture(Z, k, seed):
    """A seeded full-covariance Gaussian mixture; the regularisation goes up ten-fold
    while the covariances stay singular, as CellCharter's fit does."""
    reg = 1e-6
    while True:
        model = GaussianMixture(n_components=int(k), covariance_type="full", init_params="kmeans", tol=1e-3,
                                reg_covar=reg, max_iter=100, n_init=1, random_state=int(seed))
        try:
            return model.fit(Z)
        except ValueError:  # sklearn: "ill-defined empirical covariance"
            if reg >= 1:
                raise
            reg *= 10


def predict_blocks(model, Z, block=20_000, stop=None):
    """The most likely component of each row, a block of rows at a time."""
    out = np.empty(Z.shape[0], dtype=np.int32)
    for a in range(0, Z.shape[0], block):
        if stop is not None and stop():
            return None
        out[a:a + block] = model.predict(Z[a:a + block])
    return out


def by_size(codes, k):
    """Relabel 0..k-1 so that 0 is the largest group (ties: the lower code first)."""
    sizes = np.bincount(codes, minlength=k)
    order = np.argsort(-sizes, kind="stable")
    rank = np.empty(k, dtype=np.int32)
    rank[order] = np.arange(k, dtype=np.int32)
    return rank[codes]


def fit_domains(Z, k, seed=0, max_fit=50_000, stop=None):
    """(domain per row, numbered by size; rows the mixture was fitted on). ``None`` labels
    when ``stop`` asked to stop while predicting."""
    n = Z.shape[0]
    fit_rows = seeded_sample(n, max_fit, seed)
    model = fit_mixture(Z if len(fit_rows) == n else Z[fit_rows], k, seed)
    codes = predict_blocks(model, Z, stop=stop)
    return (None if codes is None else by_size(codes, k)), int(len(fit_rows))


def nhood_counts(rows, cols, codes, n_labels):
    """Links from each label to each label (n_labels x n_labels; a directed count, as
    squidpy's, so a symmetric graph gives a symmetric matrix)."""
    return np.bincount(codes[rows].astype(np.int64) * n_labels + codes[cols], minlength=n_labels * n_labels
                       ).reshape(n_labels, n_labels)


def shuffle_within(codes, groups, rng):
    """``codes`` shuffled within each group of rows (``groups``: a list of row numbers)."""
    out = codes.copy()
    for where in groups:
        out[where] = rng.permutation(codes[where])
    return out


def nhood_steps(adj, codes, n_labels, libraries, n_perms=1000, seed=0, block=25):
    """squidpy's nhood_enrichment on ``adj`` with labels ``codes`` (0-based, every row
    labelled), shuffled within each section: yields the permutations done after each
    block, returns (z, count). NaN z where the count cannot vary."""
    adj = sp.csr_matrix(adj)
    rows = np.repeat(np.arange(adj.shape[0], dtype=np.int64), np.diff(adj.indptr))
    cols = adj.indices.astype(np.int64)
    codes = np.asarray(codes, dtype=np.int64)
    libraries = np.asarray(libraries)
    groups = [np.flatnonzero(libraries == lib) for lib in pd.unique(libraries)]
    count = nhood_counts(rows, cols, codes, n_labels)
    rng = np.random.default_rng(seed)
    total = np.zeros((n_labels, n_labels), dtype=np.float64)
    squares = np.zeros((n_labels, n_labels), dtype=np.float64)
    for i in range(n_perms):
        c = nhood_counts(rows, cols, shuffle_within(codes, groups, rng), n_labels).astype(np.float64)
        total += c
        squares += c * c
        if (i + 1) % block == 0 or i + 1 == n_perms:
            yield i + 1
    mean = total / n_perms
    sd = np.sqrt(np.maximum(squares / n_perms - mean * mean, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(sd > 0, (count - mean) / sd, np.nan)
    return z, count


def nhood_enrichment(adj, codes, n_labels, libraries, n_perms=1000, seed=0):
    """(z, count): ``nhood_steps`` to its end."""
    steps = nhood_steps(adj, codes, n_labels, libraries, n_perms, seed)
    while True:
        try:
            next(steps)
        except StopIteration as done:
            return done.value


def window_fractions(coords, codes, n_labels, libraries, k=20, block=50_000):
    """Each row's label fractions among its ``k`` nearest rows in its section (itself
    included; fewer when the section is smaller). Float32, rows x labels."""
    coords = np.asarray(coords, dtype=np.float64)
    codes = np.asarray(codes, dtype=np.int64)
    libraries = np.asarray(libraries)
    out = np.zeros((coords.shape[0], n_labels), dtype=np.float32)
    for lib in pd.unique(libraries):
        where = np.flatnonzero(libraries == lib)
        m = min(int(k), len(where))
        tree = cKDTree(coords[where])
        for a in range(0, len(where), block):
            rows = where[a:a + block]
            _, idx = tree.query(coords[rows], k=m)
            idx = np.asarray(idx).reshape(len(rows), m)
            flat = np.repeat(np.arange(len(rows), dtype=np.int64), m) * n_labels + codes[where[idx.ravel()]]
            out[rows] = (np.bincount(flat, minlength=len(rows) * n_labels).reshape(len(rows), n_labels) / m)
    return out


def fit_niches(fractions, n_niches, seed=0, n_init=4):
    """Niches: seeded k-means on the window fractions, numbered by size."""
    model = KMeans(n_clusters=int(n_niches), n_init=int(n_init), random_state=int(seed))
    return by_size(model.fit_predict(fractions).astype(np.int32), int(n_niches))


REPLAY_CODE = (delaunay_edges, spatial_graph, aggregate_neighbors, seeded_sample, fit_mixture, predict_blocks, by_size,
               fit_domains, nhood_counts, shuffle_within, nhood_steps, nhood_enrichment, window_fractions, fit_niches)


# ---------------------------------------------------------------------------
# What a run needs
# ---------------------------------------------------------------------------

# Measured at 2 CPUs on real slides: Xenium breast (164,000 cells) and Visium HD 16 um (98,917
# bins), k = 8, 10 PCs, 3 hops: fit 9.2 s and 6.6 s on 50,000; DLPFC (3,639 spots): 0.47 s a fit,
# 53 s for the default scan. A fit is about _GMM_SECONDS x n x k x dims^2 x _GMM_ITERATIONS with
# its k-means start (easy synthetic data converges up to 5 times sooner); predicting costs
# _PREDICT_SECONDS x n x k x dims^2. Graphs and permutations also on 100,000 and 400,000
# synthetic observations.
_GMM_SECONDS = 1.0e-9
_GMM_ITERATIONS = 12
_GMM_FLOOR = 0.05
_PREDICT_SECONDS = 1.0e-10
_GRAPH_SECONDS_PER_OBS = 1.1e-5  # Delaunay and its long links
_GRID_SECONDS_PER_OBS = 4e-6  # the capture grid's kNN
_RING_SECONDS_PER_OBS = 2e-6  # per hop, Delaunay's rings; a grid's are smaller
_GRID_RING_SECONDS_PER_OBS = 5e-7
_PERM_SECONDS_PER_LINK = 9e-9  # one permutation, per directed link of the graph
_DELAUNAY_LINKS = 6  # links per observation of a Delaunay graph
_WINDOW_SECONDS_PER_VALUE = 1.8e-7  # the tree and its queries, per obs x window size
_KMEANS_SECONDS_PER_VALUE = 5e-9  # per obs x labels x niches x start
# Peak RSS over the start: Delaunay 0.15 GB at 100,000, 0.25 at 164,000 (Xenium), 0.59 at 400,000;
# grid 0.10 at 98,917 (HD 16 um), 0.40 at 399,424.
BASE_BYTES = 20 * 1024**2  # allocator slack, sklearn's workspace
_RING_BYTES_PER_OBS = 950  # three hops' sparse rings per observation, freed after the features
_GRAPH_BYTES_PER_OBS = 400  # Delaunay's triangles, the links both ways
_GRID_BYTES_PER_OBS = 850  # a grid's kNN arrays, the links both ways and its smaller rings


def fit_seconds(n_obs: int, k: int, dims: int) -> float:
    """One mixture fit and the prediction of the rows beyond the fitted ones."""
    fitted = min(n_obs, MAX_FIT_OBS)
    cost = k * dims * dims
    return _GMM_SECONDS * _GMM_ITERATIONS * cost * fitted + _PREDICT_SECONDS * cost * max(0, n_obs - fitted) + _GMM_FLOOR


def graph_seconds(n_obs: int, n_hops: int, grid: int | None = None) -> float:
    """The graph and the rings of neighbours."""
    if grid:
        return n_obs * (_GRID_SECONDS_PER_OBS + _GRID_RING_SECONDS_PER_OBS * n_hops)
    return n_obs * (_GRAPH_SECONDS_PER_OBS + _RING_SECONDS_PER_OBS * n_hops)


def scan_seconds(n_obs: int, dims: int, k_min: int, k_max: int, repeats: int) -> float:
    """Every fit a k scan makes: ``k_min - 1`` (not 1) to ``k_max + 1``, ``repeats`` each."""
    fitted = min(n_obs, MAX_FIT_OBS)
    return float(sum(fit_seconds(fitted, k, dims) for k in range(max(2, k_min - 1), k_max + 2)) * repeats)


def domain_bytes(n_obs: int, dims: int, k: int, n_hops: int, grid: int | None = None) -> int:
    """What a run holds beyond the dataset: its copies, the features, the rings, the fit."""
    fitted = min(n_obs, MAX_FIT_OBS)
    features = 4 * n_obs * dims
    fit = 8 * fitted * (2 * k + dims) + 8 * k * dims * dims * 2
    graph = _GRID_BYTES_PER_OBS if grid else _GRAPH_BYTES_PER_OBS + _RING_BYTES_PER_OBS * n_hops / 3
    return int(BASE_BYTES + features + fit + n_obs * graph)


def enrichment_seconds(n_obs: int, n_perms: int = N_PERMS, grid: int | None = None) -> float:
    links = (grid or _DELAUNAY_LINKS) * n_obs
    return round(graph_seconds(n_obs, 0, grid) + _PERM_SECONDS_PER_LINK * links * n_perms + 0.1, 1)


def enrichment_bytes(n_obs: int) -> int:
    return int(BASE_BYTES + 220 * n_obs)  # the links' row and column numbers, two label arrays


def niche_seconds(n_obs: int, n_labels: int, k_window: int, n_niches: int) -> float:
    return round(_WINDOW_SECONDS_PER_VALUE * n_obs * k_window
                 + _KMEANS_SECONDS_PER_VALUE * n_obs * n_labels * n_niches * NICHE_N_INIT + 0.1, 1)


def niche_bytes(n_obs: int, n_labels: int, k_window: int) -> int:
    block = min(n_obs, 50_000) * (k_window * 16 + n_labels * 8)
    return int(BASE_BYTES + n_obs * (4 * n_labels * 3 + 40) + block)


def has_spatial(adata) -> bool:
    return spatial.has_spatial(adata)


def _libraries(adata) -> np.ndarray:
    if spatial.LIBRARY_COLUMN in adata.obs.columns:
        s = adata.obs[spatial.LIBRARY_COLUMN]
        return s.astype(str).where(s.notna(), "").to_numpy()
    return np.full(adata.n_obs, "", dtype=object)


def _coords(adata) -> np.ndarray:
    coords = np.asarray(adata.obsm[spatial.KEY])[:, :2].astype(np.float64, copy=True)
    if not np.isfinite(coords).all():
        raise SpatialDomainsError("Some spatial positions are missing (NaN): this needs a position for every "
                                  "observation, so re-import the dataset with its positions file.")
    return coords


def choose_rep(adata, n_libraries: int) -> str:
    """Harmony's PCs for more than one section when the dataset has them, else the PCA."""
    if n_libraries > 1 and "X_pca_harmony" in adata.obsm:
        return "X_pca_harmony"
    if "X_pca" in adata.obsm:
        return "X_pca"
    raise SpatialDomainsError("This dataset has no PCA: run the pipeline up to its PCA step, then find domains.")


def grid_neighbours(adata) -> int | None:
    """6 on Visium's grid, 4 on Visium HD's, None (Delaunay) for cells."""
    return GRID_NEIGHBOURS.get(spatial.technology(adata)["technology"])


def _graph_text(grid: int | None) -> str:
    if grid:
        return f"the capture grid ({grid} neighbours)"
    return f"Delaunay triangulation without links above the {LONG_LINK_PERCENTILE:g}th percentile of length"


def _taken(adata, column: str, own: dict, overwrite: bool, name_it: str) -> None:
    """Refuse a name that is not a plain column name, one that holds another column, or
    (without ``overwrite``, the user's explicit choice) one holding an earlier result."""
    if not COLUMN_RE.match(column or ""):
        raise SpatialDomainsError(f"'{column}' is not a usable column name: start with a letter and use letters, "
                                  "digits, '_', '.' or '-'.")
    if column not in adata.obs.columns:
        return
    if column not in own:
        raise ColumnInUse(f"The dataset already has a column '{column}' that is not a {name_it} result: "
                          "choose another name.")
    if not overwrite:
        raise ColumnInUse(f"'{column}' already holds a {name_it} result: choose another name, or tick Replace.")


def free_column(adata, base: str, k: int) -> str:
    """The name a new run offers: ``base``, else ``base_k<k>``, else with a number."""
    if base not in adata.obs.columns:
        return base
    name = f"{base}_k{k}"
    n = 2
    while name in adata.obs.columns:
        name = f"{base}_k{k}_{n}"
        n += 1
    return name


def _load(adata, key: str) -> dict[str, Any]:
    """A JSON result block of uns: ``{"latest", "runs"}``, empty when absent or unreadable."""
    raw = adata.uns.get(key) if key in adata.uns else None
    if raw is None:
        return {"latest": None, "runs": {}}
    try:
        out = json.loads(str(raw))
    except (TypeError, ValueError):
        logger.warning("Ignoring an unreadable %s", key)
        return {"latest": None, "runs": {}}
    if not isinstance(out, dict) or not isinstance(out.get("runs"), dict):
        return {"latest": None, "runs": {}}
    return out


def _fingerprint(*arrays) -> str:
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:24]


def _labels_key(codes: np.ndarray, labels: list[str]) -> str:
    """Which observation has which label: a move or a rename makes a result stale."""
    return _fingerprint(codes, np.frombuffer("\0".join(labels).encode(), dtype=np.uint8))


def _labels_of(adata, column: str) -> tuple[np.ndarray, list[str]]:
    """(0-based codes, -1 for no label; the labels present, in the column's order)."""
    if column not in adata.obs.columns:
        raise SpatialDomainsError(f"Column '{column}' is not in this dataset: pick another label column.")
    s = adata.obs[column]
    if pd.api.types.is_float_dtype(s):
        raise SpatialDomainsError(f"'{column}' holds numbers, not labels: pick a label column such as clusters or "
                                  "cell types.")
    text = s.astype(str).where(s.notna()).to_numpy(dtype=object)
    present = {x for x in text if isinstance(x, str)}
    order = ([str(c) for c in s.cat.categories] if isinstance(s.dtype, pd.CategoricalDtype)
             else sorted(present))
    labels = [g for g in order if g in present]
    if len(labels) < 2:
        raise SpatialDomainsError(f"'{column}' has {len(labels)} label(s) here: this needs at least 2.")
    if len(labels) > MAX_LABELS:
        raise SpatialDomainsError(f"'{column}' has {len(labels):,} labels; at most {MAX_LABELS} can be compared: "
                                  "pick a coarser column.")
    index = {g: i for i, g in enumerate(labels)}
    codes = np.fromiter((index.get(x, -1) if isinstance(x, str) else -1 for x in text), dtype=np.int32, count=len(text))
    return codes, labels


# ---------------------------------------------------------------------------
# Domains: plan, run, save
# ---------------------------------------------------------------------------


@dataclass
class DomainPlan:
    n_obs: int
    technology: str
    unit: str
    libraries: list[str]
    rep: str
    n_pcs: int
    n_pcs_available: int
    n_hops: int
    dims: int
    k: int | None  # None: suggest it from the stability scan
    k_min: int
    k_max: int
    repeats: int
    seed: int
    column: str
    overwrite: bool
    n_fit: int
    grid: int | None
    graph: str
    notes: list[str] = field(default_factory=list)
    estimate_seconds: float = 0.0
    estimate_scan_seconds: float = 0.0
    estimate_bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _hole_note(adata, unit: str) -> str | None:
    """A note when QC filtering took a good share of a grid's spots: its holes cut links."""
    rec = adata.uns.get("scview_qc_removed") if "scview_qc_removed" in adata.uns else None
    if not isinstance(rec, dict) or unit not in spatial.SPOT_UNITS:
        return None
    try:
        removed, before = int(np.asarray(rec["n_removed"]).ravel()[0]), int(np.asarray(rec["n_before"]).ravel()[0])
    except (KeyError, TypeError, ValueError, IndexError):
        return None
    if before <= 0 or removed / before < HOLE_SHARE:
        return None
    return (f"QC filtering removed {removed:,} of {before:,} {unit}s ({removed / before:.0%}), so the tissue graph has "
            f"holes and domains see fewer neighbours; review the filtering if those {unit}s were real tissue.")


def plan_domains(adata, *, k: int | None = DEFAULT_K, n_hops: int = DEFAULT_HOPS, n_pcs: int = DEFAULT_PCS,
                 seed: int = SEED, column: str = DOMAIN_COLUMN, overwrite: bool = False,
                 k_min: int = SCAN_RANGE[0], k_max: int = SCAN_RANGE[1], repeats: int = SCAN_REPEATS,
                 check_column: bool = True) -> DomainPlan:
    """What a run would do and need; ``SpatialDomainsError`` when it cannot start. Reads
    obs, obsm and uns only."""
    if not has_spatial(adata):
        raise SpatialDomainsError("This dataset has no spatial positions: domains need spot or cell positions "
                                  "(obsm['spatial'], e.g. a Space Ranger folder).")
    n = int(adata.n_obs)
    if n < MIN_OBS:
        raise SpatialDomainsError(f"Domains need at least {MIN_OBS} observations; this dataset has {n}.")
    libs = [str(x) for x in pd.unique(_libraries(adata))]
    rep = choose_rep(adata, len(libs))
    width = int(np.shape(adata.obsm[rep])[1])
    use = min(int(n_pcs), width)
    if use < MIN_PCS:
        raise SpatialDomainsError(f"{rep} has {width} component(s); domains need at least {MIN_PCS}: run the "
                                  "pipeline's PCA step again.")
    if not 0 <= int(n_hops) <= MAX_HOPS:
        raise SpatialDomainsError(f"Hops must be between 0 and {MAX_HOPS}.")
    if k is not None and not 2 <= int(k) <= min(MAX_K, n - 1):
        raise SpatialDomainsError(f"k must be between 2 and {min(MAX_K, n - 1)}.")
    if not 2 <= int(k_min) < int(k_max) <= min(MAX_K, n - 2):
        raise SpatialDomainsError(f"The k range must run upwards from at least 2 to at most {min(MAX_K, n - 2)}.")
    if not 2 <= int(repeats) <= 20:
        raise SpatialDomainsError("Repeats per k must be between 2 and 20.")
    if check_column:
        _taken(adata, column, _load(adata, DOMAINS_KEY)["runs"], overwrite, "domain")
    tech = spatial.technology(adata)
    grid = GRID_NEIGHBOURS.get(tech["technology"])
    dims = use * (int(n_hops) + 1)
    notes: list[str] = []
    fitted = min(n, MAX_FIT_OBS)
    if n > MAX_FIT_OBS:
        notes.append(f"Above {MAX_FIT_OBS:,} {tech['unit']}s the model is fitted on a seeded sample of {fitted:,} and "
                     "then assigns all of them.")
    if len(libs) > 1:
        notes.append("Uses the Harmony-corrected PCs, as there is more than one section." if rep == "X_pca_harmony" else
                     "Uses the uncorrected PCA of several sections, so domains may split by section; run the pipeline "
                     "with a batch column for corrected PCs.")
    hole = _hole_note(adata, tech["unit"])
    if hole:
        notes.append(hole)
    k_used = int(k) if k is not None else DEFAULT_K
    return DomainPlan(
        n_obs=n, technology=tech["technology"], unit=tech["unit"], libraries=libs, rep=rep, n_pcs=use,
        n_pcs_available=width, n_hops=int(n_hops), dims=dims, k=int(k) if k is not None else None, k_min=int(k_min),
        k_max=int(k_max), repeats=int(repeats), seed=int(seed), column=column, overwrite=bool(overwrite), n_fit=fitted,
        grid=grid, graph=_graph_text(grid), notes=notes,
        estimate_seconds=round(graph_seconds(n, int(n_hops), grid) + fit_seconds(n, k_used, dims), 1),
        estimate_scan_seconds=round(scan_seconds(n, dims, int(k_min), int(k_max), int(repeats)), 1),
        estimate_bytes=domain_bytes(n, dims, k_used if k is not None else int(k_max) + 1, int(n_hops), grid),
    )


@dataclass
class DomainInputs:
    plan: DomainPlan
    coords: np.ndarray
    libraries: np.ndarray
    vectors: np.ndarray
    fingerprint: str  # of positions + vectors + settings: which scan a later run may reuse


def _scan_key(coords: np.ndarray, vectors: np.ndarray, n_hops: int, seed: int) -> str:
    return _fingerprint(coords, vectors, np.array([n_hops, seed], dtype=np.int64))


def prepare_domains(adata, p: DomainPlan) -> DomainInputs:
    """The run's own copies (a few MB): the dataset stays free for other requests meanwhile."""
    coords = _coords(adata)
    vectors = np.asarray(adata.obsm[p.rep])[:, :p.n_pcs].astype(np.float32, copy=True)
    return DomainInputs(plan=p, coords=coords, libraries=_libraries(adata), vectors=vectors,
                        fingerprint=_scan_key(coords, vectors, p.n_hops, p.seed))


@dataclass
class DomainResult:
    labels: np.ndarray
    k: int
    n_fit: int
    stability: dict[str, Any] | None
    n_links: int
    n_isolated: int
    coherence: dict[str, float]
    seconds: float
    stage_seconds: dict[str, float]


class _Stages:
    """Stage events and their timing; ``begin`` stops a cancelled run."""

    def __init__(self, stages: tuple[tuple[str, str], ...], stop: Callable[[], bool]) -> None:
        self.stages, self.stop = stages, stop
        self.t0 = time.monotonic()
        self.spent: dict[str, float] = {}
        self.current: str | None = None
        self.since = self.t0

    def begin(self, name: str) -> dict[str, Any]:
        if self.stop():
            raise SpatialDomainsCancelled()
        now = time.monotonic()
        self.close(now)
        self.current, self.since = name, now
        names = [s for s, _ in self.stages]
        return {"type": "stage", "stage": name, "label": dict(self.stages)[name], "index": names.index(name) + 1,
                "total": len(names), "elapsed_s": round(now - self.t0, 1)}

    def close(self, now: float | None = None) -> float:
        now = time.monotonic() if now is None else now
        if self.current is not None:
            self.spent[self.current] = round(now - self.since, 2)
        return now - self.t0


DOMAIN_STAGES = (("graph", "Tissue graph"), ("features", "Neighbour features"), ("scan", "Stability of each k"),
                 ("fit", "Gaussian mixture"), ("assign", "Assigning domains"))


def stability_curve(labels: dict[int, list[np.ndarray]], k_min: int, k_max: int) -> dict[str, Any]:
    """Each k's mean ARI to the other seeds' runs at k - 1 and k + 1, and the best local
    peak (the highest k of the curve when it has no peak)."""
    ks = list(range(k_min, k_max + 1))
    means, sds = [], []
    for k in ks:
        scores = []
        for other in (k - 1, k + 1):
            for i, a in enumerate(labels.get(k, [])):
                for j, b in enumerate(labels.get(other, [])):
                    if i != j:
                        scores.append(adjusted_rand_score(a, b))
        means.append(float(np.mean(scores)) if scores else 0.0)
        sds.append(float(np.std(scores)) if scores else 0.0)
    peaks, _ = find_peaks(np.asarray(means))
    peak_ks = [ks[int(i)] for i in peaks]
    best = max(peak_ks, key=lambda k: means[ks.index(k)]) if peak_ks else ks[int(np.argmax(means))]
    return {"ks": ks, "mean": [round(m, 4) for m in means], "sd": [round(s, 4) for s in sds], "peaks": peak_ks,
            "suggested": int(best), "has_peak": bool(peak_ks)}


def edge_coherence(adj, labels, seed=0) -> dict[str, float]:
    """The share of graph links within one group, and the same labels shuffled: how far a
    labelling follows the tissue (1 = every neighbour alike)."""
    adj = sp.csr_matrix(adj)
    rows = np.repeat(np.arange(adj.shape[0]), np.diff(adj.indptr))
    labels = np.asarray(labels)
    if not adj.nnz:
        return {"within": 0.0, "shuffled": 0.0}
    shuffled = np.random.default_rng(seed).permutation(labels)
    return {"within": round(float((labels[rows] == labels[adj.indices]).mean()), 4),
            "shuffled": round(float((shuffled[rows] == shuffled[adj.indices]).mean()), 4)}


def iter_domains(inp: DomainInputs, stop: Callable[[], bool] = lambda: False) -> Iterator[dict[str, Any]]:
    """Run: ``stage`` events, ``scan`` after each fit of a k scan, ``curve`` when the scan
    ends, then ``{"type": "result", "result": DomainResult}``. ``stop`` is checked between
    pieces of work (SpatialDomainsCancelled: nothing is kept)."""
    p = inp.plan
    stages = _Stages(DOMAIN_STAGES if p.k is None else tuple(s for s in DOMAIN_STAGES if s[0] != "scan"), stop)
    yield stages.begin("graph")
    adj = spatial_graph(inp.coords, inp.libraries, p.grid, LONG_LINK_PERCENTILE)
    n_isolated = int((np.diff(adj.indptr) == 0).sum())
    yield stages.begin("features")
    Z = aggregate_neighbors(adj, inp.vectors, p.n_hops)
    stability: dict[str, Any] | None = None
    k = p.k
    if k is None:
        yield stages.begin("scan")
        rows = seeded_sample(Z.shape[0], MAX_FIT_OBS, p.seed)
        sample = Z if len(rows) == Z.shape[0] else Z[rows]
        seeds = [int(s.generate_state(1)[0]) for s in np.random.SeedSequence(p.seed).spawn(p.repeats)]
        fit_ks = list(range(max(2, p.k_min - 1), p.k_max + 2))
        total, done = len(fit_ks) * p.repeats, 0
        labels: dict[int, list[np.ndarray]] = {x: [] for x in fit_ks}
        for run, s in enumerate(seeds):
            for fit_k in fit_ks:
                if stop():
                    raise SpatialDomainsCancelled()
                labels[fit_k].append(predict_blocks(fit_mixture(sample, fit_k, s), sample))
                done += 1
                yield {"type": "scan", "k": fit_k, "run": run + 1, "done": done, "total": total,
                       "elapsed_s": round(time.monotonic() - stages.t0, 1)}
        stability = {**stability_curve(labels, p.k_min, p.k_max), "repeats": p.repeats, "n_fit": int(len(rows)),
                     "seed": p.seed, "inputs": inp.fingerprint}
        del labels, sample
        k = int(stability["suggested"])
        yield {"type": "curve", "stability": stability}
    yield stages.begin("fit")
    fit_rows = seeded_sample(Z.shape[0], MAX_FIT_OBS, p.seed)
    model = fit_mixture(Z if len(fit_rows) == Z.shape[0] else Z[fit_rows], k, p.seed)
    yield stages.begin("assign")
    codes = predict_blocks(model, Z, stop=stop)
    if codes is None:
        raise SpatialDomainsCancelled()
    domains = by_size(codes, k)
    seconds = stages.close()
    yield {"type": "result", "result": DomainResult(
        labels=domains, k=k, n_fit=int(len(fit_rows)), stability=stability, n_links=int(adj.nnz // 2),
        n_isolated=n_isolated, coherence=edge_coherence(adj, domains, p.seed), seconds=seconds,
        stage_seconds=dict(stages.spent))}


def run_domains(inp: DomainInputs) -> DomainResult:
    """``iter_domains`` to its end."""
    for item in iter_domains(inp):
        if item["type"] == "result":
            return item["result"]
    raise SpatialDomainsError("The run ended without a result.")  # pragma: no cover


def _per_library(codes: np.ndarray, libraries: np.ndarray, n: int) -> dict[str, list[int]]:
    """Observations per group in each section; ``codes`` -1 (no label) left out."""
    return {str(lib): np.bincount(codes[(libraries == lib) & (codes >= 0)], minlength=n).astype(int).tolist()
            for lib in pd.unique(libraries)}


def method_text(p: DomainPlan, k: int, n_fit: int) -> str:
    sample = f", fitted on a seeded sample of {n_fit:,}" if n_fit < p.n_obs else ""
    return (f"CellCharter-style spatial domains (Varrone et al. 2024): a graph per section from {p.graph}; features: "
            f"the first {p.n_pcs} components of {p.rep} of each {p.unit} with the means of its neighbours at 1 to "
            f"{p.n_hops} hops; full-covariance Gaussian mixture, k = {k}, seed {p.seed}{sample}.")


def replay_params(p: DomainPlan, res: DomainResult) -> dict[str, Any]:
    return {"column": p.column, "k": res.k, "rep": p.rep, "n_pcs": p.n_pcs, "n_hops": p.n_hops, "seed": p.seed,
            "grid_neighbours": p.grid, "suggested": res.stability is not None}


def save_domains(adata, inp: DomainInputs, res: DomainResult, *, resources: dict | None = None,
                 cells_fingerprint: str | None = None) -> dict[str, Any]:
    """Store in ``adata``: obs column, uns JSON (replaced, never mutated) and a provenance
    step. The column's markers and enrichment, if it held an earlier result, are dropped."""
    from scview.core import provenance
    from scview.core.obs_writes import forget_column_results

    p = inp.plan
    created = datetime.now(timezone.utc).isoformat()
    adata.obs[p.column] = pd.Categorical.from_codes(res.labels, categories=[str(i) for i in range(res.k)])
    forget_column_results(adata, p.column)
    block = _load(adata, DOMAINS_KEY)
    # A run at a chosen k keeps the curve of the same inputs' scan: the k was picked from it.
    scan = res.stability or block.get("scan")
    if isinstance(scan, dict) and scan.get("inputs") != inp.fingerprint:
        scan = None
    notes = list(p.notes)
    if res.n_isolated:
        notes.append(f"{res.n_isolated:,} {p.unit}(s) have no neighbour on the graph, so only their own profile "
                     "places them.")
    info = {
        "version": VERSION, "run_id": f"{created}|{adata.n_obs}", "created_at": created, "column": p.column,
        "k": res.k, "suggested": res.stability is not None, "technology": p.technology, "unit": p.unit, "rep": p.rep,
        "n_pcs": p.n_pcs, "n_hops": p.n_hops, "seed": p.seed, "n_obs": p.n_obs, "n_fit": res.n_fit, "graph": p.graph,
        "grid": p.grid, "n_links": res.n_links, "n_isolated": res.n_isolated, "libraries": p.libraries, "notes": notes,
        "sizes": np.bincount(res.labels, minlength=res.k).astype(int).tolist(),
        "per_library": _per_library(res.labels, inp.libraries, res.k), "coherence": res.coherence,
        "inputs_fingerprint": inp.fingerprint, "cells_fingerprint": cells_fingerprint,
        "seconds": round(res.seconds, 1),
        "stage_seconds": dict(res.stage_seconds), "method": method_text(p, res.k, res.n_fit),
    }
    adata.uns[DOMAINS_KEY] = json.dumps({"latest": p.column, "runs": {**block["runs"], p.column: info}, "scan": scan})
    provenance.record_step(adata, step=STEP_DOMAINS, tool="scview.spatial_domains (CellCharter algorithm, sklearn "
                           "GaussianMixture)", params=replay_params(p, res), resources=resources)
    return domains_summary(adata, cells_fingerprint)


def domains_stale(adata, info: dict, cells_fingerprint: str | None) -> str | None:
    """Why a saved run no longer fits the dataset, or None."""
    unit = info.get("unit", "cell")
    if cells_fingerprint is not None and info.get("cells_fingerprint") not in (None, cells_fingerprint):
        return f"The {unit}s changed since this ran (a filter or a merge): run it again."
    if int(adata.n_obs) != info.get("n_obs"):
        return f"The {unit}s changed since this ran (a filter or a merge): run it again."
    rep, n_pcs = info.get("rep"), int(info.get("n_pcs") or 0)
    if rep not in adata.obsm or np.shape(adata.obsm[rep])[1] < n_pcs:
        return "The PCs this used are gone: run it again."
    vectors = np.asarray(adata.obsm[rep])[:, :n_pcs].astype(np.float32, copy=False)
    key = _scan_key(np.asarray(adata.obsm[spatial.KEY])[:, :2].astype(np.float64, copy=False), vectors,
                    int(info.get("n_hops", 0)), int(info.get("seed", 0)))
    if key != info.get("inputs_fingerprint"):
        return "The PCs changed since this ran (a pipeline run): run it again."
    return None


def domains_summary(adata, cells_fingerprint: str | None = None) -> dict[str, Any]:
    """The saved runs for the UI: ``available`` False without one; ``scan`` the last k scan."""
    block = _load(adata, DOMAINS_KEY)
    runs = {c: i for c, i in block["runs"].items() if isinstance(i, dict) and c in adata.obs.columns}
    latest = block.get("latest") if block.get("latest") in runs else next(iter(runs), None)
    if latest is None:
        return {"available": False, "columns": [], "scan": block.get("scan")}
    reason = domains_stale(adata, runs[latest], cells_fingerprint)
    return {"available": True, **runs[latest], "stale": reason is not None, "stale_reason": reason,
            "columns": list(runs), "scan": block.get("scan")}


# ---------------------------------------------------------------------------
# Neighbourhood enrichment
# ---------------------------------------------------------------------------


@dataclass
class EnrichmentPlan:
    n_obs: int
    n_labelled: int
    unit: str
    column: str
    labels: list[str]
    n_perms: int
    seed: int
    grid: int | None
    graph: str
    notes: list[str] = field(default_factory=list)
    estimate_seconds: float = 0.0
    estimate_bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EnrichmentInputs:
    plan: EnrichmentPlan
    coords: np.ndarray
    libraries: np.ndarray
    codes: np.ndarray


@dataclass
class EnrichmentResult:
    zscore: np.ndarray
    count: np.ndarray
    sizes: list[int]
    seconds: float


ENRICHMENT_STAGES = (("graph", "Tissue graph"), ("permutations", "Shuffling the labels"))


def _need_positions(adata) -> None:
    if not has_spatial(adata):
        raise SpatialDomainsError("This dataset has no spatial positions: this needs spot or cell positions "
                                  "(obsm['spatial'], e.g. a Space Ranger folder).")


def plan_enrichment(adata, column: str, n_perms: int = N_PERMS, seed: int = SEED) -> EnrichmentPlan:
    _need_positions(adata)
    if not 10 <= int(n_perms) <= MAX_PERMS:
        raise SpatialDomainsError(f"Permutations must be between 10 and {MAX_PERMS:,}.")
    codes, labels = _labels_of(adata, column)
    grid = grid_neighbours(adata)
    unit = spatial.technology(adata)["unit"]
    n_labelled = int((codes >= 0).sum())
    notes = []
    if n_labelled < adata.n_obs:
        notes.append(f"{adata.n_obs - n_labelled:,} {unit}(s) without a label are left out.")
    return EnrichmentPlan(n_obs=int(adata.n_obs), n_labelled=n_labelled, unit=unit, column=column, labels=labels,
                          n_perms=int(n_perms), seed=int(seed), grid=grid, graph=_graph_text(grid), notes=notes,
                          estimate_seconds=enrichment_seconds(n_labelled, int(n_perms), grid),
                          estimate_bytes=enrichment_bytes(int(adata.n_obs)))


def prepare_enrichment(adata, p: EnrichmentPlan) -> EnrichmentInputs:
    codes, _ = _labels_of(adata, p.column)
    return EnrichmentInputs(plan=p, coords=_coords(adata), libraries=_libraries(adata), codes=codes)


def iter_enrichment(inp: EnrichmentInputs, stop: Callable[[], bool] = lambda: False) -> Iterator[dict[str, Any]]:
    """Run: ``stage``, ``progress`` every few permutations, then the result."""
    p = inp.plan
    stages = _Stages(ENRICHMENT_STAGES, stop)
    yield stages.begin("graph")
    adj = spatial_graph(inp.coords, inp.libraries, p.grid, LONG_LINK_PERCENTILE)
    keep = inp.codes >= 0
    if not keep.all():
        adj = adj[keep][:, keep]
    codes = inp.codes[keep]
    yield stages.begin("permutations")
    steps = nhood_steps(adj, codes, len(p.labels), inp.libraries[keep], p.n_perms, p.seed, PERM_BLOCK)
    while True:
        if stop():
            raise SpatialDomainsCancelled()
        try:
            done = next(steps)
        except StopIteration as finished:
            z, count = finished.value
            break
        yield {"type": "progress", "done": done, "total": p.n_perms,
               "elapsed_s": round(time.monotonic() - stages.t0, 1)}
    seconds = stages.close()
    yield {"type": "result", "result": EnrichmentResult(
        zscore=z, count=count, sizes=np.bincount(codes, minlength=len(p.labels)).astype(int).tolist(), seconds=seconds)}


def run_enrichment(inp: EnrichmentInputs) -> EnrichmentResult:
    for item in iter_enrichment(inp):
        if item["type"] == "result":
            return item["result"]
    raise SpatialDomainsError("The run ended without a result.")  # pragma: no cover


def squidpy_key(column: str) -> str:
    """squidpy's uns key for the result (no "/": it would nest in the h5ad)."""
    return f"{column.replace('/', '_')}_nhood_enrichment"


def save_enrichment(adata, inp: EnrichmentInputs, res: EnrichmentResult, *, resources: dict | None = None,
                    cells_fingerprint: str | None = None) -> dict[str, Any]:
    """Stored where squidpy stores it (uns ``<column>_nhood_enrichment``: zscore, count), so an
    exported file reads in squidpy as its own, with its labels and settings beside it."""
    from scview.core import provenance

    p = inp.plan
    created = datetime.now(timezone.utc).isoformat()
    adata.uns[squidpy_key(p.column)] = {"zscore": res.zscore.astype(np.float64), "count": res.count.astype(np.int64)}
    block = _load(adata, ENRICHMENT_KEY)
    info = {"version": VERSION, "created_at": created, "column": p.column, "labels": p.labels, "sizes": res.sizes,
            "n_perms": p.n_perms, "seed": p.seed, "n_obs": p.n_obs, "n_labelled": p.n_labelled, "unit": p.unit,
            "graph": p.graph, "notes": p.notes, "seconds": round(res.seconds, 1), "cells_fingerprint": cells_fingerprint,
            "labels_fingerprint": _labels_key(inp.codes, p.labels)}
    adata.uns[ENRICHMENT_KEY] = json.dumps({"latest": p.column, "runs": {**block["runs"], p.column: info}})
    provenance.record_step(adata, step=STEP_ENRICHMENT, tool="scview.spatial_domains (= squidpy 1.6 nhood_enrichment)",
                           params={"column": p.column, "n_perms": p.n_perms, "seed": p.seed, "grid_neighbours": p.grid,
                                   "labels": p.labels}, resources=resources)
    return enrichment_summary(adata, p.column, cells_fingerprint)


def _labels_changed(adata, info: dict, cells_fingerprint: str | None) -> str | None:
    """Why a result built on a label column no longer fits it, or None."""
    column = str(info.get("label_column") or info.get("column"))
    if cells_fingerprint is not None and info.get("cells_fingerprint") not in (None, cells_fingerprint):
        return f"The {info.get('unit', 'cell')}s changed since this ran (a filter or a merge): run it again."
    try:
        codes, labels = _labels_of(adata, column)
    except SpatialDomainsError:
        return f"The labels of '{column}' are gone: run it again."
    if _labels_key(codes, labels) != info.get("labels_fingerprint"):
        return f"The labels in '{column}' changed since this ran: run it again."
    return None


def _finite(a: np.ndarray, digits: int = 3) -> list[list[float | None]]:
    return [[round(float(v), digits) if np.isfinite(v) else None for v in row] for row in np.asarray(a, dtype=float)]


def enrichment_summary(adata, column: str | None = None, cells_fingerprint: str | None = None) -> dict[str, Any]:
    """A saved run with its matrices (``column``'s, else the latest); ``available`` False without one."""
    block = _load(adata, ENRICHMENT_KEY)
    runs = {c: i for c, i in block["runs"].items() if isinstance(i, dict) and squidpy_key(c) in adata.uns}
    want = column if column in runs else (block.get("latest") if block.get("latest") in runs else next(iter(runs), None))
    if want is None:
        return {"available": False, "columns": []}
    stored = adata.uns[squidpy_key(want)]
    reason = _labels_changed(adata, runs[want], cells_fingerprint)
    return {"available": True, **runs[want], "zscore": _finite(np.asarray(stored["zscore"])),
            "count": np.asarray(stored["count"]).astype(int).tolist(), "stale": reason is not None,
            "stale_reason": reason, "columns": list(runs)}


# ---------------------------------------------------------------------------
# Composition niches
# ---------------------------------------------------------------------------


def niches_allowed(adata) -> dict[str, Any]:
    """``{"ok", "reason", "note"}``: refused on spots that mix cells, with a sentence."""
    tech = spatial.technology(adata)["technology"]
    if tech == "visium":
        return {"ok": False, "note": None, "reason": (
            "Composition niches need one label per cell, and a Visium spot mixes several cells: use neighbourhood "
            "enrichment here, or Spot composition for each spot's cell-type shares.")}
    if tech == "visium_hd":
        uns = adata.uns.get(spatial.KEY)
        sizes = {spatial._bin_size(uns.get(lib)) for lib in spatial._libraries(adata)} if uns is not None else set()
        sizes.discard(None)
        if sizes != {HD_NICHE_BIN_UM}:
            shown = ", ".join(f"{s:g}" for s in sorted(sizes))
            return {"ok": False, "note": None, "reason": (
                f"Composition niches need about one cell per observation, and these Visium HD bins are {shown} um: "
                f"import the {HD_NICHE_BIN_UM:g} um bins for this.")}
        return {"ok": True, "reason": None, "note": (
            f"Visium HD {HD_NICHE_BIN_UM:g} um bins are close to one cell each but some hold parts of two: read the "
            "niches as approximate.")}
    return {"ok": True, "reason": None, "note": None}


@dataclass
class NichePlan:
    n_obs: int
    n_labelled: int
    unit: str
    label_column: str
    labels: list[str]
    k_window: int
    n_niches: int
    seed: int
    column: str
    overwrite: bool
    notes: list[str] = field(default_factory=list)
    estimate_seconds: float = 0.0
    estimate_bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class NicheInputs:
    plan: NichePlan
    coords: np.ndarray
    libraries: np.ndarray
    codes: np.ndarray


@dataclass
class NicheResult:
    labels: np.ndarray  # niche per observation, -1 without a label
    fractions: np.ndarray  # niches x labels: mean window fraction
    seconds: float


NICHE_STAGES = (("windows", "Each observation's neighbours"), ("kmeans", "Grouping into niches"))


def plan_niches(adata, label_column: str, *, k_window: int = NICHE_NEIGHBOURS, n_niches: int = DEFAULT_NICHES,
                seed: int = SEED, column: str = NICHE_COLUMN, overwrite: bool = False) -> NichePlan:
    _need_positions(adata)
    allowed = niches_allowed(adata)
    if not allowed["ok"]:
        raise SpatialDomainsError(allowed["reason"])
    if not 2 <= int(k_window) <= MAX_NICHE_NEIGHBOURS:
        raise SpatialDomainsError(f"The window must hold between 2 and {MAX_NICHE_NEIGHBOURS} neighbours.")
    if not 2 <= int(n_niches) <= MAX_NICHES:
        raise SpatialDomainsError(f"The number of niches must be between 2 and {MAX_NICHES}.")
    _taken(adata, column, _load(adata, NICHES_KEY)["runs"], overwrite, "niche")
    codes, labels = _labels_of(adata, label_column)
    n_labelled = int((codes >= 0).sum())
    if n_labelled <= int(n_niches):
        raise SpatialDomainsError(f"Only {n_labelled:,} labelled observations for {n_niches} niches: pick fewer niches.")
    unit = spatial.technology(adata)["unit"]
    notes = [allowed["note"]] if allowed["note"] else []
    if n_labelled < adata.n_obs:
        notes.append(f"{adata.n_obs - n_labelled:,} {unit}(s) without a label are left out.")
    return NichePlan(n_obs=int(adata.n_obs), n_labelled=n_labelled, unit=unit, label_column=label_column, labels=labels,
                     k_window=int(k_window), n_niches=int(n_niches), seed=int(seed), column=column,
                     overwrite=bool(overwrite), notes=notes,
                     estimate_seconds=niche_seconds(n_labelled, len(labels), int(k_window), int(n_niches)),
                     estimate_bytes=niche_bytes(n_labelled, len(labels), int(k_window)))


def prepare_niches(adata, p: NichePlan) -> NicheInputs:
    codes, _ = _labels_of(adata, p.label_column)
    return NicheInputs(plan=p, coords=_coords(adata), libraries=_libraries(adata), codes=codes)


def iter_niches(inp: NicheInputs, stop: Callable[[], bool] = lambda: False) -> Iterator[dict[str, Any]]:
    p = inp.plan
    stages = _Stages(NICHE_STAGES, stop)
    yield stages.begin("windows")
    keep = inp.codes >= 0
    fractions = window_fractions(inp.coords[keep], inp.codes[keep], len(p.labels), inp.libraries[keep], p.k_window)
    yield stages.begin("kmeans")
    niche = fit_niches(fractions, p.n_niches, p.seed, NICHE_N_INIT)
    if stop():
        raise SpatialDomainsCancelled()
    mean = np.stack([fractions[niche == i].mean(axis=0) if (niche == i).any() else np.zeros(len(p.labels))
                     for i in range(p.n_niches)])
    labels = np.full(inp.codes.shape[0], -1, dtype=np.int32)
    labels[keep] = niche
    seconds = stages.close()
    yield {"type": "result", "result": NicheResult(labels=labels, fractions=mean, seconds=seconds)}


def run_niches(inp: NicheInputs) -> NicheResult:
    for item in iter_niches(inp):
        if item["type"] == "result":
            return item["result"]
    raise SpatialDomainsError("The run ended without a result.")  # pragma: no cover


def save_niches(adata, inp: NicheInputs, res: NicheResult, *, resources: dict | None = None,
                cells_fingerprint: str | None = None) -> dict[str, Any]:
    from scview.core import provenance
    from scview.core.obs_writes import forget_column_results

    p = inp.plan
    created = datetime.now(timezone.utc).isoformat()
    names = [str(i) for i in range(p.n_niches)]
    adata.obs[p.column] = pd.Categorical.from_codes(res.labels, categories=names)
    forget_column_results(adata, p.column)
    block = _load(adata, NICHES_KEY)
    info = {"version": VERSION, "created_at": created, "column": p.column, "label_column": p.label_column,
            "labels": p.labels, "niches": names, "fractions": _finite(res.fractions, 4), "k_window": p.k_window,
            "n_niches": p.n_niches, "seed": p.seed, "n_obs": p.n_obs, "n_labelled": p.n_labelled, "unit": p.unit,
            "sizes": np.bincount(res.labels[res.labels >= 0], minlength=p.n_niches).astype(int).tolist(),
            "per_library": _per_library(res.labels, inp.libraries, p.n_niches), "notes": p.notes,
            "seconds": round(res.seconds, 1), "cells_fingerprint": cells_fingerprint,
            "labels_fingerprint": _labels_key(inp.codes, p.labels)}
    adata.uns[NICHES_KEY] = json.dumps({"latest": p.column, "runs": {**block["runs"], p.column: info}})
    provenance.record_step(adata, step=STEP_NICHES, tool="scview.spatial_domains (Schurch et al. 2020 windows, sklearn "
                           "KMeans)", params={"column": p.column, "label_column": p.label_column, "labels": p.labels,
                                              "k_window": p.k_window, "n_niches": p.n_niches, "seed": p.seed},
                           resources=resources)
    return niches_summary(adata, cells_fingerprint)


def niches_summary(adata, cells_fingerprint: str | None = None) -> dict[str, Any]:
    block = _load(adata, NICHES_KEY)
    runs = {c: i for c, i in block["runs"].items() if isinstance(i, dict) and c in adata.obs.columns}
    latest = block.get("latest") if block.get("latest") in runs else next(iter(runs), None)
    if latest is None:
        return {"available": False, "columns": []}
    reason = _labels_changed(adata, runs[latest], cells_fingerprint)
    return {"available": True, **runs[latest], "stale": reason is not None, "stale_reason": reason,
            "columns": list(runs)}


# ---------------------------------------------------------------------------
# Co-pilot
# ---------------------------------------------------------------------------


def _strongest_pairs(summary: dict[str, Any], n: int = 3) -> tuple[list[str], list[str]]:
    """The most enriched and most avoided pairs of different labels, as text."""
    labels, z = summary["labels"], summary["zscore"]
    pairs = [(z[i][j], labels[i], labels[j]) for i in range(len(labels)) for j in range(i + 1, len(labels))
             if z[i][j] is not None]
    high = sorted((p for p in pairs if p[0] > 2), key=lambda p: -p[0])[:n]
    low = sorted((p for p in pairs if p[0] < -2), key=lambda p: p[0])[:n]

    def text(rows: list) -> list[str]:
        return [f"{a} with {b} (z {v:.1f})" for v, a, b in rows]

    return text(high), text(low)


def grounding_lines(adata, cells_fingerprint: str | None = None) -> tuple[list[str], str]:
    """The saved results as one fact: the domains' k, sizes and stability, the niches and
    the strongest enrichment pairs. Nothing when none fits the dataset any more."""
    out = ["\n## Spatial domains and niches (last runs)"]
    detail = ""
    d = domains_summary(adata, cells_fingerprint)
    if d.get("available") and not d.get("stale"):
        sizes = ", ".join(f"domain {i}: {n:,}" for i, n in enumerate(d["sizes"]))
        detail = f"{d['k']} spatial domains in '{d['column']}' on {d['n_obs']:,} {d['unit']}s"
        chosen = "suggested by the stability scan" if d.get("suggested") else "chosen by the user"
        out.append(f"- {detail} ({sizes}); k {chosen}; {d['coherence']['within']:.0%} of neighbouring {d['unit']}s share "
                   f"a domain (shuffled labels {d['coherence']['shuffled']:.0%}); first {d['n_pcs']} components of "
                   f"{d['rep']}, {d['n_hops']} hops [{TAG}].")
        scan = d.get("scan")
        if isinstance(scan, dict) and scan.get("ks"):
            mean = scan["mean"][scan["ks"].index(d["k"])] if d["k"] in scan["ks"] else None
            shown = f"; stability (mean ARI to the runs at k - 1 and k + 1) at k = {d['k']}: {mean:.2f}" if mean is not None else ""
            peak = "" if scan.get("has_peak") else " (no local peak, so the highest value)"
            out.append(f"- Stability scan over k = {scan['ks'][0]} to {scan['ks'][-1]}, {scan['repeats']} repeats: "
                       f"suggested k = {scan['suggested']}{peak}{shown} [{TAG}].")
    n = niches_summary(adata, cells_fingerprint)
    if n.get("available") and not n.get("stale"):
        tops = []
        for i, row in enumerate(n["fractions"]):
            best = int(np.argmax(np.asarray(row, dtype=float)))
            tops.append(f"niche {i} ({n['sizes'][i]:,}): mostly {n['labels'][best]} {row[best]:.0%}")
        text = (f"{n['n_niches']} composition niches in '{n['column']}' from '{n['label_column']}', windows of the "
                f"{n['k_window']} nearest")
        detail = detail or text
        out.append(f"- {text}: " + "; ".join(tops) + f" [{TAG}].")
    e = enrichment_summary(adata, None, cells_fingerprint)
    if e.get("available") and not e.get("stale"):
        high, low = _strongest_pairs(e)
        text = f"Neighbourhood enrichment of '{e['column']}' ({e['n_perms']:,} permutations)"
        parts = ([f"most often neighbours: {', '.join(high)}"] if high else []) + (
            [f"most avoided: {', '.join(low)}"] if low else [])
        detail = detail or text
        out.append(f"- {text}: {'; '.join(parts) if parts else 'no pair of different labels beyond |z| 2'} [{TAG}].")
    if len(out) == 1:
        return [], ""
    out.append("- A domain is a stretch of tissue with one profile and one kind of surroundings, not a cell type; the "
               "number of domains is a choice the stability scan only advises on.")
    return out, detail
