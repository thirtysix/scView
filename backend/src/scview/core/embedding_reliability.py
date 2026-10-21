"""Per-cell embedding reliability: does the plot put each cell near its real neighbours?

A UMAP (or t-SNE) places each cell from the kNN graph and has to squeeze that
graph into two or three dimensions. Where it fails, a cell lands away from its
neighbours: a stray inside another group's island, or a bridge between islands.

The per-cell score is **placement**: the share of the cell's k graph neighbours
(the kNN graph the clustering used) that lie in the part of the plot holding its
nearest ``WINDOW`` share of cells (5%, K = round(0.05 * (n - 1)) cells, at least
k), i.e. within the distance of its K-th nearest cell on the plot:

    placement_i = |{j in graph neighbours of i : d_plot(i, j) <= r_K(i)}| / k

1 means every graph neighbour sits nearby; a low value (below ``LOW`` = 0.5:
fewer than half its neighbours nearby) means the cell sits away from its real
neighbours, so its position should not be read closely.

Why a share of the cells and not a multiple of k: the plot rank of a cell's graph
neighbours grows with the number of cells (median rank 184 on Kang's 2D UMAP,
13,836 cells; 60 on paul15's, 2,730 cells), so no fixed K = m * k fits both. For
m = 1, 3, 5, 10 Kang's median was 0.07, 0.13, 0.27, 0.47 (58% of cells still low at
m = 10) while paul15's reached 0.79 (6% low) at m = 10; Kang needed m of about 50.
A window of 5% of the cells left typical cells high in all five layouts measured
(median 0.93 / 1.0 Kang 2D / 3D, 0.79 / 0.86 paul15, 0.87 a 37,104-cell ovary atlas;
2-9% of cells low) and still flagged 96-100% of 1% of cells moved into another
group's island (seeded). Wider windows let strays through (10%: 85% of paul15's
2D strays flagged); narrower ones mark a fifth to a third of all cells (2-3%).

The strict score, the share of the k graph neighbours that are also the cell's
k nearest cells on the plot (``exact``), is kept as context: UMAP keeps few
exact neighbours anywhere (median 1 of 15 on Kang), so positions within an
island should be read loosely, but colouring every cell by it said only that.

The graph's own neighbours define k (the self entry left out: scanpy stores
k - 1 neighbours, older files store the cell itself as an explicit 0). Without a
stored graph, an exact kNN is built in the representation
``uns["neighbors"]["params"]`` names (``use_rep``, ``n_pcs``, ``metric``), and the
result says so.

Baselines: chance, K / (n - 1) for placement (k / (n - 1) for the exact score),
the expected score when a cell's neighbours are a random draw; and one seeded
shuffle of the coordinates (random placement), scored the same way.

The window radius is exact up to ``EXACT_MAX`` cells (the K-th nearest cell by
KD-tree); above, it is the distance to the matching rank among a seeded sample
of ``REF_CELLS`` cells (the true count inside it was within 12% of K for 90% of
cells at 55k and 249k cells; forced on Kang, it changed 10% of the cells' scores, by
0.007 on average). Group summaries (n, median, IQR, share low) come
from the same per-cell scores, worst groups first. Above ``MAX_CELLS`` cells a
seeded sample of ``SAMPLE_CELLS`` cells is scored (against neighbours among all
cells); the others are left out (NaN) and the result says so.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp

from scview.core.object_cache import ObjectCache
from scview.core.qc_groups import group_codes

logger = logging.getLogger(__name__)

TAG = "result:embedding_reliability"

WINDOW = 0.05  # placement window: the cell's nearest 5% of cells on the plot (see above)
# "Low": fewer than half of a cell's graph neighbours lie in its window.
LOW = 0.5
LOW_REASON = "below 0.5: fewer than half of the cell's graph neighbours lie near it on the plot"
EXACT_MAX = 20_000  # window radius exact up to here (13,836 cells: 0.6 s on 2 cores)
REF_CELLS = 5_000  # above it, estimated from a seeded sample of this many cells
MAX_CELLS = 250_000  # score every cell up to here
SAMPLE_CELLS = 100_000  # above it, a seeded sample of this many
SEED = 0
N_JOBS = 2  # the laptop rule: never more than 2 cores
MAX_K = 100
DEFAULT_GRAPH_K = 14  # scanpy's n_neighbors=15, self excluded (no graph, no params)
MAX_GROUPS = 200
_PER_DATASET = 6  # cached results per loaded dataset
_CHUNK = 2_000  # rows per KD-tree query chunk (K can be in the thousands)

METHOD = (
    "Placement, per cell: the share of its k graph neighbours (the kNN graph the clustering "
    "used, self excluded) that lie within the distance of its K-th nearest cell on the plot, "
    "K = 5% of the cells (at least k); 1 = all nearby. Exact neighbours, for context: the share "
    "of the k graph neighbours that are also its k nearest cells on the plot. k defaults to the "
    "graph's own neighbour count. Chance = K / (n - 1) (k / (n - 1) for exact). Random "
    "placement = the score after one seeded shuffle of the coordinates. Low = " + LOW_REASON + "."
)


class ReliabilityError(ValueError):
    """A request that cannot be answered (the API answers 422)."""


@dataclass
class Graph:
    """Each cell's graph neighbours, nearest first: ``idx`` padded with -1."""

    idx: np.ndarray  # (n, width) int64
    count: np.ndarray  # (n,) neighbours per cell (self excluded)
    k: int  # the graph's own k: the most common count
    source: str  # "stored" | "computed"
    key: str | None = None  # uns entry of a stored graph
    use_rep: str | None = None
    n_pcs: int | None = None
    metric: str = "euclidean"
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Graph neighbours
# ---------------------------------------------------------------------------


def _graph_entry(adata) -> tuple[str | None, dict]:
    """The uns entry of the graph the clustering used and its obsp distances key.
    scView clusters on ``uns["neighbors"]``; a graph stored under a custom
    ``key_added`` is used when that is the only one."""
    nb = adata.uns.get("neighbors")
    if isinstance(nb, dict):
        return "neighbors", nb
    for key, val in adata.uns.items():
        if isinstance(val, dict) and isinstance(val.get("distances_key"), str) \
                and val["distances_key"] in adata.obsp:
            return str(key), val
    return None, {}


def _padded_rows(m: sp.spmatrix, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For each of ``rows``: its column indices in ``m``, the cell itself removed
    and nearest first, padded with -1 to the longest row; and each row's count."""
    m = sp.csr_matrix(m)[rows]
    n = m.shape[0]
    counts = np.diff(m.indptr)
    at = np.repeat(np.arange(n), counts)
    keep = m.indices != rows[at]  # the cell itself, stored as an explicit 0 by some versions
    at, cols, dist = at[keep], m.indices[keep].astype(np.int64), m.data[keep].astype(np.float64)
    counts = np.bincount(at, minlength=n)
    width = int(counts.max()) if n else 0
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    pos = np.arange(at.size) - starts[at]
    idx = np.full((n, width), -1, dtype=np.int64)
    dd = np.full((n, width), np.inf)
    idx[at, pos] = cols
    dd[at, pos] = dist
    order = np.argsort(dd, axis=1, kind="stable")  # nearest first; padding (inf) last
    return np.take_along_axis(idx, order, axis=1), counts


def _mode(counts: np.ndarray) -> int:
    pos = counts[counts > 0]
    return int(np.bincount(pos).argmax()) if pos.size else 0


def _rep_params(adata, params: dict) -> tuple[str, int | None, str]:
    use_rep = params.get("use_rep")
    rep = str(use_rep) if use_rep else "X_pca"
    n_pcs = params.get("n_pcs")
    metric = str(params.get("metric") or "euclidean")
    return rep, (int(n_pcs) if n_pcs is not None else None), metric


def stored_graph(adata, rows: np.ndarray) -> Graph | None:
    """The stored graph's neighbours of ``rows`` (None without a stored graph)."""
    key, entry = _graph_entry(adata)
    dkey = entry.get("distances_key", "distances") if entry else "distances"
    if dkey not in adata.obsp:
        return None
    idx, counts = _padded_rows(adata.obsp[dkey], rows)
    params = entry.get("params") if isinstance(entry.get("params"), dict) else {}
    rep, n_pcs, metric = _rep_params(adata, params)
    g = Graph(idx=idx, count=counts, k=_mode(counts), source="stored", key=key,
              use_rep=rep if params else None, n_pcs=n_pcs, metric=metric)
    if key is None:
        g.notes.append(f"No neighbour parameters were recorded; the graph is obsp['{dkey}'].")
    elif key != "neighbors":
        g.notes.append(f"The graph stored under uns['{key}'] (obsp['{dkey}']).")
    return g


def computed_graph(adata, k: int, rows: np.ndarray) -> Graph:
    """The ``k`` exact nearest neighbours of ``rows`` (among all cells) in the
    representation the neighbour parameters name (else X_pca, all columns): the
    graph scanpy would have stored."""
    from sklearn.neighbors import NearestNeighbors

    _, entry = _graph_entry(adata)
    params = entry.get("params") if isinstance(entry.get("params"), dict) else {}
    rep, n_pcs, metric = _rep_params(adata, params)
    if rep not in adata.obsm:
        raise ReliabilityError(
            f"No neighbour graph is stored and there is no {rep} to build one from: run the "
            "pipeline's neighbors step first.")
    X = np.asarray(adata.obsm[rep], dtype=np.float64)
    if n_pcs is not None:
        X = X[:, :n_pcs]
    nn = NearestNeighbors(n_neighbors=min(k + 1, X.shape[0]), metric=metric, n_jobs=N_JOBS).fit(X)
    _, ind = nn.kneighbors(X[rows])
    idx = drop_self(ind, rows, k)
    counts = np.full(rows.size, idx.shape[1], dtype=np.int64)
    where = f"{rep}" + (f", first {n_pcs} columns" if n_pcs else "")
    g = Graph(idx=idx, count=counts, k=int(idx.shape[1]), source="computed", key=None,
              use_rep=rep, n_pcs=n_pcs, metric=metric)
    g.notes.append(
        f"No neighbour graph is stored: the graph neighbours are an exact {k}-nearest-neighbour "
        f"search in {where} ({metric}), as the neighbour step would have built them.")
    return g


def drop_self(ind: np.ndarray, rows: np.ndarray, k: int) -> np.ndarray:
    """The first ``k`` neighbours of each query row other than the row itself.
    Not simply column 0: cells at identical coordinates tie at distance 0 and
    the cell itself need not come first."""
    is_self = ind == rows[:, None]
    order = np.argsort(is_self, axis=1, kind="stable")  # self (True) moves to the end
    return np.take_along_axis(ind, order, axis=1)[:, :k].astype(np.int64)


# ---------------------------------------------------------------------------
# Plot neighbours and the overlap
# ---------------------------------------------------------------------------


def plot_neighbours(coords: np.ndarray, k: int, rows: np.ndarray) -> np.ndarray:
    """The ``k`` nearest cells on the plot (exact; KD-tree in 2D / 3D) to each of
    ``rows``, among all cells, self excluded."""
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="kd_tree", n_jobs=N_JOBS).fit(coords)
    _, ind = nn.kneighbors(coords[rows])
    return drop_self(ind, rows, k)


def overlap(graph_idx: np.ndarray, plot_idx: np.ndarray) -> np.ndarray:
    """Per row, how many of ``graph_idx`` (-1 = padding) are in ``plot_idx``."""
    m = graph_idx.shape[0]
    out = np.zeros(m, dtype=np.int64)
    if m == 0:
        return out
    chunk = max(1, int(2e7 // max(1, graph_idx.shape[1] * plot_idx.shape[1])))
    for s in range(0, m, chunk):
        g = graph_idx[s:s + chunk, :, None]
        p = plot_idx[s:s + chunk, None, :]
        out[s:s + chunk] = (g == p).any(axis=2).sum(axis=1)
    return out


def exact_scores(graph: Graph, coords: np.ndarray, k: int, rows: np.ndarray) -> np.ndarray:
    """The strict score of ``rows`` (``graph`` holds their neighbours, in that order):
    the share of the k graph neighbours that are also the k nearest cells on the plot."""
    g = graph.idx[:, :k]
    denom = np.minimum(graph.count, k).astype(np.float64)
    hits = overlap(g, plot_neighbours(coords, k, rows))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denom > 0, hits / denom, np.nan)


def _threaded(fn, cells: np.ndarray) -> np.ndarray:
    """``fn`` over chunks of ``cells`` on ``N_JOBS`` threads (KD-tree queries release the GIL)."""
    chunks = [cells[i:i + _CHUNK] for i in range(0, cells.size, _CHUNK)]
    if len(chunks) <= 1:
        return fn(cells)
    with ThreadPoolExecutor(N_JOBS) as pool:
        return np.concatenate(list(pool.map(fn, chunks)))


def window_radius(coords: np.ndarray, window: int, cells: np.ndarray,
                  exact_max: int | None = None, ref_cells: int | None = None) -> tuple[np.ndarray, int | None]:
    """For each of ``cells``: the distance on the plot to its ``window``-th nearest other
    cell. Exact (KD-tree) up to ``exact_max`` cells; above, the distance to the matching
    rank among a seeded sample of ``ref_cells`` cells. Returns (radii, the sample size or
    None when exact)."""
    from sklearn.neighbors import KDTree

    exact_max = EXACT_MAX if exact_max is None else exact_max
    ref_cells = REF_CELLS if ref_cells is None else ref_cells
    n = coords.shape[0]
    if n <= exact_max:
        tree = KDTree(coords)

        def exact(chunk):
            d, _ = tree.query(coords[chunk], k=window + 1)  # the cell itself is at distance 0
            return d[:, -1]

        return _threaded(exact, cells), None
    ref_cells = min(ref_cells, n)
    ref = np.sort(np.random.default_rng(SEED + 2).choice(n, size=ref_cells, replace=False))
    is_ref = np.zeros(n, dtype=bool)
    is_ref[ref] = True
    rank = max(1, int(round(window * ref_cells / n)))
    tree = KDTree(coords[ref])

    def estimated(chunk):
        d, _ = tree.query(coords[chunk], k=rank + 1)
        # A cell of the sample finds itself first, at distance 0.
        return np.where(is_ref[chunk], d[:, rank], d[:, rank - 1])

    return _threaded(estimated, cells), ref_cells


def placement_scores(graph: Graph, coords: np.ndarray, k: int, rows: np.ndarray,
                     radius: np.ndarray) -> np.ndarray:
    """The share of each row's k graph neighbours within ``radius`` (its window) of it on
    the plot. ``graph`` and ``radius`` follow ``rows``."""
    g = graph.idx[:, :k]
    valid = g >= 0
    safe = np.where(valid, g, rows[:, None])
    hits = np.zeros(rows.size, dtype=np.int64)
    for i in range(0, rows.size, 50_000):
        sl = slice(i, i + 50_000)
        d = np.linalg.norm(coords[rows[sl]][:, None, :] - coords[safe[sl]], axis=2)
        hits[sl] = ((d <= radius[sl, None]) & valid[sl]).sum(axis=1)
    denom = np.minimum(graph.count, k).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denom > 0, hits / denom, np.nan)


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------


def stats(values: np.ndarray, low: float = LOW) -> dict:
    v = values[np.isfinite(values)]
    if v.size == 0:
        return {"n": 0, "median": None, "q1": None, "q3": None, "mean": None, "share_low": None}
    q1, med, q3 = np.quantile(v, [0.25, 0.5, 0.75])
    return {"n": int(v.size), "median": float(med), "q1": float(q1), "q3": float(q3),
            "mean": float(v.mean()), "share_low": float(np.mean(v < low))}


def group_summary(scores: np.ndarray, labels: pd.Series, low: float = LOW,
                  exact: np.ndarray | None = None) -> list[dict]:
    """Per group: its scored cells' n, median, IQR, mean and share below ``low``
    (and the median ``exact`` score, when given); worst first (lowest median, then
    the larger share low). Groups with no scored cell are left out."""
    names, codes = group_codes(labels)
    if len(names) > MAX_GROUPS:
        raise ReliabilityError(
            f"'{labels.name}' has {len(names):,} groups; summarise by a column with at most "
            f"{MAX_GROUPS}.")
    rows = []
    for i, name in enumerate(names):
        mask = codes == i
        s = stats(scores[mask], low)
        if s["n"]:
            row = {"group": name, **s}
            if exact is not None:
                row["exact_median"] = stats(exact[mask], low)["median"]
            rows.append(row)
    rows.sort(key=lambda r: (r["median"], -r["share_low"], r["group"]))
    return rows


def default_group_column(adata) -> tuple[str | None, str | None]:
    from scview.core.anndata_adaptor import CLUSTERING_CANDIDATES

    active = adata.uns.get("scview_active_clustering")
    if isinstance(active, str) and active in adata.obs.columns:
        return active, "the active clustering"
    for c in CLUSTERING_CANDIDATES:
        if c in adata.obs.columns:
            return c, "the clustering column"
    return None, None


# ---------------------------------------------------------------------------
# The whole computation, cached on the loaded dataset
# ---------------------------------------------------------------------------


_cache = ObjectCache()
_lock = threading.Lock()


@dataclass
class Scored:
    scores: np.ndarray  # (n,) float32 placement, NaN = not scored
    exact: np.ndarray  # (n,) float32 strict same-k score, NaN = not scored
    shuffled: np.ndarray  # placement of the scored rows after one shuffle
    meta: dict
    computed_at: float


def _coords(adata, embedding: str, dims: list[int] | None) -> tuple[np.ndarray, list[int], list[str]]:
    if embedding not in adata.obsm:
        raise KeyError(f"Embedding '{embedding}' not found.")
    arr = np.asarray(adata.obsm[embedding], dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ReliabilityError(f"'{embedding}' has fewer than two columns: nothing to plot.")
    notes: list[str] = []
    if dims is not None:
        cols = list(dims)
    elif arr.shape[1] > 3:
        cols = [0, 1]
        notes.append(f"'{embedding}' has {arr.shape[1]} columns; scored as plotted, the first two.")
    else:
        cols = list(range(arr.shape[1]))
    coords = arr[:, cols]
    if not np.isfinite(coords).all():
        raise ReliabilityError(f"'{embedding}' has missing coordinates: it cannot be scored.")
    return np.ascontiguousarray(coords), cols, notes


def compute(adata, embedding: str, k: int | None = None, dims: list[int] | None = None,
            max_cells: int | None = None, sample_cells: int | None = None,
            window: float | None = None, exact_max: int | None = None) -> Scored:
    """Per-cell scores and the baselines for one embedding (see the module docstring).
    ``window`` (share of cells) and ``exact_max`` are for measurements and tests."""
    t0 = time.perf_counter()
    max_cells = MAX_CELLS if max_cells is None else max_cells
    sample_cells = SAMPLE_CELLS if sample_cells is None else sample_cells
    share = WINDOW if window is None else window
    coords, cols, notes = _coords(adata, embedding, dims)
    n = coords.shape[0]
    if n < 3:
        raise ReliabilityError("Fewer than three cells: nothing to compare.")
    if k is not None and (k < 1 or k > MAX_K):
        raise ReliabilityError(f"k must be between 1 and {MAX_K}, not {k}.")
    sampled = n > max_cells
    n_sample = min(sample_cells, n)
    rng = np.random.default_rng(SEED)
    rows = np.sort(rng.choice(n, size=n_sample, replace=False)) if sampled else np.arange(n)
    graph = stored_graph(adata, rows)
    if graph is not None and graph.k == 0:
        graph = None
    if graph is None:
        want = k if k is not None else DEFAULT_GRAPH_K
        graph = computed_graph(adata, min(want, n - 2), rows)
    elif k is not None and k > graph.k:
        raise ReliabilityError(
            f"The stored graph keeps {graph.k} neighbours per cell, so k can be at most "
            f"{graph.k}, not {k}.")
    kk = k if k is not None else graph.k
    kk = min(kk, graph.idx.shape[1], n - 2)
    window_cells = min(n - 1, max(kk, int(round(share * (n - 1)))))
    notes = graph.notes + notes
    short = int(np.sum((graph.count > 0) & (graph.count < kk)))
    if short:
        notes.append(f"{short:,} cells have fewer than {kk} graph neighbours; each is scored "
                     "over the neighbours it has.")
    if sampled:
        notes.append(f"{n:,} cells: a seeded sample of {n_sample:,} was scored (their plot "
                     "neighbours searched among all cells); the others are left out.")

    perm = np.random.default_rng(SEED + 1).permutation(n)
    radius, ref = window_radius(coords, window_cells, rows if sampled else np.arange(n), exact_max)
    if ref:
        notes.append(f"The window (each cell's nearest {window_cells:,} cells) was measured "
                     f"against a seeded sample of {ref:,} cells, so it holds about that many "
                     "(within 10% for most cells).")
    scores = np.full(n, np.nan, dtype=np.float32)
    scores[rows] = placement_scores(graph, coords, kk, rows, radius if sampled else radius[rows])
    # Random placement: cell i takes cell perm[i]'s position, and with it that position's window.
    shuffled_radius = (window_radius(coords, window_cells, perm[rows], exact_max)[0] if sampled
                       else radius[perm[rows]])
    shuffled = placement_scores(graph, coords[perm], kk, rows, shuffled_radius).astype(np.float32)
    exact = np.full(n, np.nan, dtype=np.float32)
    exact[rows] = exact_scores(graph, coords, kk, rows)

    if embedding == "X_umap_uncorrected":
        notes.append("This UMAP was laid out from the uncorrected graph; it is scored against "
                     "the batch-corrected graph the clustering used.")
    meta = {
        "embedding": embedding,
        "dims": cols,
        "n_dims": len(cols),
        "k": int(kk),
        "window": {"share": share, "cells": int(window_cells),
                   "radius": "estimated" if ref else "exact", "ref_cells": ref},
        "graph": {
            "source": graph.source, "key": graph.key, "k": int(graph.k),
            "use_rep": graph.use_rep, "n_pcs": graph.n_pcs, "metric": graph.metric,
        },
        "n_cells": int(n),
        "n_scored": int(rows.size),
        "sampled": bool(sampled),
        "seed": SEED,
        "chance": float(window_cells / (n - 1)),
        "exact_chance": float(kk / (n - 1)),
        "low_threshold": LOW,
        "low_reason": LOW_REASON,
        "notes": notes,
        "method": METHOD,
        "seconds": round(time.perf_counter() - t0, 3),
    }
    logger.info("Plot placement %s dims=%s k=%d window=%d: %d cells in %.2f s",
                embedding, cols, kk, window_cells, rows.size, meta["seconds"])
    return Scored(scores=scores, exact=exact, shuffled=shuffled, meta=meta, computed_at=time.time())


def _key(adata, embedding: str, k: int | None, dims: list[int] | None) -> tuple:
    gkey, entry = _graph_entry(adata)
    dkey = entry.get("distances_key", "distances") if entry else "distances"
    graph_id = id(adata.obsp[dkey]) if dkey in adata.obsp else None
    emb_id = id(adata.obsm[embedding]) if embedding in adata.obsm else None
    return (embedding, tuple(dims) if dims is not None else None, k, emb_id, graph_id)


def scored(adata, embedding: str, k: int | None = None, dims: list[int] | None = None) -> tuple[Scored, bool]:
    """``compute``, cached on the loaded dataset (the dataset manager loads a new
    object after every pipeline run, so a cached result cannot outlive its graph
    or coordinates). Returns (result, came from the cache)."""
    key = _key(adata, embedding, k, dims)
    with _lock:  # one at a time: no double work, at most N_JOBS cores
        store = _cache.get(adata)
        if store is not None and key in store:
            store.move_to_end(key)
            return store[key], True
        result = compute(adata, embedding, k=k, dims=dims)
        if store is None:
            store = OrderedDict()
            _cache.put(adata, store)
        store[key] = result
        while len(store) > _PER_DATASET:
            store.popitem(last=False)
    return result, False


def latest(adata) -> Scored | None:
    """The most recently computed result for this dataset, without computing."""
    store = _cache.get(adata)
    if not store:
        return None
    return max(store.values(), key=lambda r: r.computed_at)


def summarise(adata, result: Scored, group_by: str | None = None) -> dict:
    """The JSON summary: overall stats, the baselines, and per group."""
    s = result.scores
    reason = "requested"
    if not group_by:
        group_by, reason = default_group_column(adata)
    elif group_by not in adata.obs.columns:
        raise KeyError(f"Column '{group_by}' not found in obs.")
    groups = group_summary(s, adata.obs[group_by], exact=result.exact) if group_by else []
    shuffled = stats(result.shuffled)
    exact = stats(result.exact)
    return {
        **result.meta,
        "overall": stats(s),
        "exact": {"median": exact["median"], "q1": exact["q1"], "q3": exact["q3"],
                  "mean": exact["mean"], "chance": result.meta["exact_chance"]},
        "shuffled": {"median": shuffled["median"], "mean": shuffled["mean"], "seed": SEED + 1},
        "group_by": group_by,
        "group_reason": reason,
        "groups": groups,
    }


def for_adaptor(adaptor, embedding: str, k: int | None = None, group_by: str | None = None,
                dims: list[int] | None = None) -> tuple[np.ndarray, dict]:
    """(per-cell scores, summary) for a loaded dataset."""
    adata = adaptor.adata
    result, cached = scored(adata, embedding, k=k, dims=dims)
    summary = summarise(adata, result, group_by)
    summary["cached"] = cached
    return result.scores, summary


# ---------------------------------------------------------------------------
# The co-pilot's fact
# ---------------------------------------------------------------------------


def _f(v: float | None) -> str:
    """Two decimals, or two significant digits below 0.1 (a chance level of 0.0011)."""
    if v is None:
        return "n/a"
    if v == 0:
        return "0"
    return f"{v:.2f}" if abs(v) >= 0.1 else f"{v:.2g}"


def exact_sentence(k: int, exact_median: float | None, plot: str = "the plot") -> str:
    """The strict same-k number as context: how many of k exact neighbours a typical
    cell keeps, and what that means when it is few."""
    kept = round((exact_median or 0.0) * k)
    text = f"Exact neighbours kept: median {kept} of {k}"
    if (exact_median or 0.0) < 0.5:
        text += (f"; {plot} keeps few exact neighbours anywhere, so read positions within an "
                 "island loosely")
    return text + "."


def grounding_lines(adaptor) -> tuple[list[str], str]:
    """One fact line, only when a result has been computed for this dataset:
    the overall placement, its chance and random-placement baselines, the strict
    exact-neighbour number as context and the three worst groups of the active
    clustering."""
    adata = adaptor.adata
    result = latest(adata)
    if result is None:
        return [], ""
    d = summarise(adata, result)
    o = d["overall"]
    w = d["window"]["cells"]
    plot = d["embedding"] + (f" (columns {d['dims'][0] + 1} and {d['dims'][1] + 1})"
                             if d["embedding"] and "pca" in d["embedding"].lower() else "")
    sample = f" (a seeded sample of {d['n_scored']:,})" if d["sampled"] else ""
    worst = "; ".join(
        f"{g['group']} median {_f(g['median'])}, {g['share_low']:.0%} of its {g['n']:,} cells low"
        for g in d["groups"][:3])
    groups = (f" Worst groups of '{d['group_by']}': {worst}." if worst else "")
    detail = (f"{plot}: median placement {_f(o['median'])} (share of {d['k']} graph neighbours "
              f"within each cell's nearest {w:,} cells); {o['share_low']:.0%} low; chance {_f(d['chance'])}")
    line = (f"- Plot reliability (placement) on {plot}: per cell, the share of its {d['k']} graph "
            f"neighbours (the kNN graph the clustering used) that lie among its nearest {w:,} cells "
            f"on the plot ({d['window']['share']:.0%} of the cells). Median {_f(o['median'])} over "
            f"{o['n']:,} cells{sample} (IQR {_f(o['q1'])} to {_f(o['q3'])}); {o['share_low']:.0%} of "
            f"cells are low ({LOW_REASON}): those sit away from their real neighbours on the plot, so "
            f"their position should not be read closely. Chance {_f(d['chance'])}; random placement "
            f"mean {_f(d['shuffled']['mean'])}. {exact_sentence(d['k'], d['exact']['median'])}"
            f"{groups} A group smaller than the window ({w:,} cells) scores near 1 whenever it forms "
            f"its own island, whatever the order inside it [{TAG}]")
    return [line], detail
