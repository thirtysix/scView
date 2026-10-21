"""Per-cell embedding reliability: how much of each cell's neighbourhood a plot keeps.

A UMAP (or t-SNE) places each cell from the kNN graph and has to squeeze that
graph into two or three dimensions, so some cells end up beside cells that are
not their neighbours in the analysis space. For each cell this is the share of
its graph neighbours (the kNN graph the clustering used) that are also among
its k nearest cells on the plot:

    score_i = |graph neighbours of i  intersect  k nearest cells to i on the plot| / k

1 means the plot keeps every one of the cell's neighbours around it; a low value
means the cells around it on the plot are mostly not its real neighbours, so its
position should not be read closely. The graph's own neighbours define k (the
self entry left out on both sides: scanpy stores k - 1 neighbours, older files
store the cell itself as an explicit 0). Without a stored graph, an exact kNN is
built in the representation ``uns["neighbors"]["params"]`` names (``use_rep``,
``n_pcs``, ``metric``), and the result says so.

Two baselines make the number readable: chance, k / (n - 1), the expected score
when a cell's plot neighbours are a random draw; and one seeded shuffle of the
coordinates (random placement), scored the same way.

Group summaries (n, median, IQR, share below ``LOW``) come from the same
per-cell scores, worst groups first. Above ``MAX_CELLS`` cells a seeded sample of
``SAMPLE_CELLS`` cells is scored (against neighbours among all cells); the others
are left out (NaN) and the result says so.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp

from scview.core.object_cache import ObjectCache
from scview.core.qc_groups import group_codes

logger = logging.getLogger(__name__)

TAG = "result:embedding_reliability"

# "Low": fewer than 1 in 5 of a cell's graph neighbours are beside it on the plot.
LOW = 0.2
LOW_REASON = (
    "below 0.2: fewer than 1 in 5 of the cell's graph neighbours are among its nearest "
    "cells on the plot"
)
MAX_CELLS = 250_000  # score every cell up to here
SAMPLE_CELLS = 100_000  # above it, a seeded sample of this many
SEED = 0
N_JOBS = 2  # the laptop rule: never more than 2 cores
MAX_K = 100
DEFAULT_GRAPH_K = 14  # scanpy's n_neighbors=15, self excluded (no graph, no params)
MAX_GROUPS = 200
_PER_DATASET = 6  # cached results per loaded dataset

METHOD = (
    "Per cell: the share of its k graph neighbours (the kNN graph the clustering used, "
    "self excluded) that are also among its k nearest cells on the plot (exact kNN on the "
    "plotted coordinates, KD-tree, self excluded); 1 = all kept. k defaults to the graph's own "
    "neighbour count. Chance = k / (n - 1). Random placement = the same score after one seeded "
    "shuffle of the coordinates. Low = " + LOW_REASON + "."
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


def _scores(graph: Graph, coords: np.ndarray, k: int, rows: np.ndarray) -> np.ndarray:
    """``graph`` holds the neighbours of ``rows``, in that order."""
    g = graph.idx[:, :k]
    denom = np.minimum(graph.count, k).astype(np.float64)
    hits = overlap(g, plot_neighbours(coords, k, rows))
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


def group_summary(scores: np.ndarray, labels: pd.Series, low: float = LOW) -> list[dict]:
    """Per group: its scored cells' n, median, IQR, mean and share below ``low``;
    worst first (lowest median, then the larger share low). Groups with no
    scored cell are left out."""
    names, codes = group_codes(labels)
    if len(names) > MAX_GROUPS:
        raise ReliabilityError(
            f"'{labels.name}' has {len(names):,} groups; summarise by a column with at most "
            f"{MAX_GROUPS}.")
    rows = []
    for i, name in enumerate(names):
        s = stats(scores[codes == i], low)
        if s["n"]:
            rows.append({"group": name, **s})
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
    scores: np.ndarray  # (n,) float32, NaN = not scored
    shuffled: np.ndarray  # scores of the scored rows after one shuffle
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
            max_cells: int | None = None, sample_cells: int | None = None) -> Scored:
    """Per-cell scores and the baselines for one embedding (see the module docstring)."""
    t0 = time.perf_counter()
    max_cells = MAX_CELLS if max_cells is None else max_cells
    sample_cells = SAMPLE_CELLS if sample_cells is None else sample_cells
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
    notes = graph.notes + notes
    short = int(np.sum((graph.count > 0) & (graph.count < kk)))
    if short:
        notes.append(f"{short:,} cells have fewer than {kk} graph neighbours; each is scored "
                     "over the neighbours it has.")
    if sampled:
        notes.append(f"{n:,} cells: a seeded sample of {n_sample:,} was scored (their plot "
                     "neighbours searched among all cells); the others are left out.")
    scores = np.full(n, np.nan, dtype=np.float32)
    scores[rows] = _scores(graph, coords, kk, rows)
    perm = np.random.default_rng(SEED + 1).permutation(n)
    shuffled = _scores(graph, coords[perm], kk, rows).astype(np.float32)

    if embedding == "X_umap_uncorrected":
        notes.append("This UMAP was laid out from the uncorrected graph; it is scored against "
                     "the batch-corrected graph the clustering used.")
    meta = {
        "embedding": embedding,
        "dims": cols,
        "n_dims": len(cols),
        "k": int(kk),
        "graph": {
            "source": graph.source, "key": graph.key, "k": int(graph.k),
            "use_rep": graph.use_rep, "n_pcs": graph.n_pcs, "metric": graph.metric,
        },
        "n_cells": int(n),
        "n_scored": int(rows.size),
        "sampled": bool(sampled),
        "seed": SEED,
        "chance": float(kk / (n - 1)),
        "low_threshold": LOW,
        "low_reason": LOW_REASON,
        "notes": notes,
        "method": METHOD,
        "seconds": round(time.perf_counter() - t0, 3),
    }
    logger.info("Neighbour preservation %s dims=%s k=%d: %d cells in %.2f s",
                embedding, cols, kk, rows.size, meta["seconds"])
    return Scored(scores=scores, shuffled=shuffled, meta=meta, computed_at=time.time())


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
    groups = group_summary(s, adata.obs[group_by]) if group_by else []
    shuffled = stats(result.shuffled)
    return {
        **result.meta,
        "overall": stats(s),
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


def grounding_lines(adaptor) -> tuple[list[str], str]:
    """One fact line, only when a result has been computed for this dataset:
    the overall median, the chance and random-placement baselines and the
    three worst groups of the active clustering."""
    adata = adaptor.adata
    result = latest(adata)
    if result is None:
        return [], ""
    d = summarise(adata, result)
    o = d["overall"]
    plot = d["embedding"] + (f" (columns {d['dims'][0] + 1} and {d['dims'][1] + 1})"
                             if d["embedding"] and "pca" in d["embedding"].lower() else "")
    sample = f" (a seeded sample of {d['n_scored']:,})" if d["sampled"] else ""
    worst = "; ".join(
        f"{g['group']} median {_f(g['median'])}, {g['share_low']:.0%} of its {g['n']:,} cells low"
        for g in d["groups"][:3])
    groups = (f" Worst groups of '{d['group_by']}': {worst}." if worst else "")
    detail = (f"{plot}: median share kept {_f(o['median'])} of {d['k']} graph neighbours per cell; "
              f"chance {_f(d['chance'])}")
    line = (f"- Plot reliability (neighbour preservation) on {plot}: per cell, the share of its "
            f"{d['k']} graph neighbours (the kNN graph the clustering used) that are also among its "
            f"{d['k']} nearest cells on the plot. Median {_f(o['median'])} over {o['n']:,} cells"
            f"{sample} (IQR {_f(o['q1'])} to {_f(o['q3'])}); {o['share_low']:.0%} of cells are low "
            f"({LOW_REASON}). Chance {_f(d['chance'])}; random placement median "
            f"{_f(d['shuffled']['median'])}, mean {_f(d['shuffled']['mean'])}.{groups} Small groups "
            f"score higher by chance alone (fewer cells to land among). A low cell's position on the "
            f"plot should not be read closely [{TAG}]")
    return [line], detail
