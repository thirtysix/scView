"""Which structure in a 2D layout survives another seed or neighbourhood size (C089).

A UMAP is one draw of a stochastic method with a chosen ``n_neighbors`` and
``min_dist``. Islands that touch, a bridge between two clusters, a cluster that
sits between two others: these are read as biology. This module lays the same
cells out again, from the same PCA (or Harmony) representation, under other
seeds and neighbour counts, and says which group-level relations hold in every
layout and which appear in only some.

Layouts: seeds ``SEEDS`` x n_neighbors {10, the dataset's own k, 30} (scanpy's
UMAP, the dataset's ``min_dist``). For each n_neighbors the kNN graph is REBUILT
from the same representation (``uns["neighbors"]["params"]``: ``use_rep``,
``n_pcs``, metric), as the pipeline's neighbours step builds it; seed 0 with the
dataset's own k on all cells is the pipeline's own layout again. Above
``MAX_CELLS`` cells a seeded sample of ``MAX_CELLS`` is laid out (the result says so).

Touching, per layout and pair of groups A, B. Each cell's ``PLOT_K`` nearest
cells on the plot are found (KD-tree). A group's **main body** is the cells in a
part of it that holds at least ``BODY_SHARE`` of its cells (connected components
of the plot-kNN graph restricted to the group's own cells): a stray cell or a
small clump of a group inside another group's island is not part of its main
body. A **contact cell** of A towards B is a main-body cell of A with a
main-body cell of B among its ``PLOT_K`` nearest cells. A and B **touch** when
their contact cells, both sides together, number at least ``MIN_CONTACT`` (or,
for a group with fewer cells, all of its cells' worth).

Why not a share of each group's cells, or every cell: measured on Kang (13,836
PBMCs, 13 cell types) and paul15 (2,730 cells, 19 clusters), 9 layouts each.
Counting every cell with a neighbour of the other group, CD14 monocytes "touched"
CD4 memory T cells through 193 stray cells in every layout. Keeping only cells
with half their neighbours in their own group, CD14 and CD16 monocytes (one
contiguous island) touched in 7 of 9 layouts: intermixed borders vanish. The
main body removed the strays and clumps (residual contacts of 1-5 cells) and kept
true contacts at 20-400 cells. A fixed share fails at both ends: CD4 naive and
CD8 T cells (2,600 / 753 cells) touch with 1.2-4% of their cells at the contact,
paul15's 9-cell eosinophil group with all of them.

Across layouts, per pair: the share of layouts in which they touch (``every``,
``some``, ``never``), and whether they touch in the user's current plot. A
relation in the current plot but in fewer than half of the layouts is flagged
(``rare_in_plot``), as is one missing from the plot but in more than half
(``hidden_in_plot``). Per group: its median placement score (batch 13,
``core/embedding_reliability.py``: the share of a cell's graph neighbours within
its nearest 5% of cells on the plot) in each layout, against one reference graph
(the dataset's own k, rebuilt on the laid-out cells), and the range across layouts.

Results are transient: the latest run is kept against the loaded dataset object
(a pipeline run loads a new one). "Keep this layout" saves one as obsm
``X_umap_s<seed>_k<k>`` (``keep``), recorded in provenance and the replay script.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp

from scview.core import embedding_reliability as er
from scview.core import provenance
from scview.core.object_cache import ObjectCache
from scview.core.qc_groups import group_codes

logger = logging.getLogger(__name__)

TAG = "result:layouts"

SEEDS = (0, 1, 2)
NEIGHBOR_CHOICES = (10, 30)  # besides the dataset's own k (scanpy's default 15)
EXTRA_NEIGHBORS = (15, 50, 5)  # when the dataset's own k is 10 or 30: still three values
MIN_NEIGHBORS, MAX_NEIGHBORS = 2, 100
MAX_SEEDS = 5
MAX_LAYOUTS = 12
MAX_CELLS = 20_000  # laid out in full up to here; above, a seeded sample of this many
SAMPLE_SEED = 0
DEFAULT_K = 15  # scanpy's n_neighbors when nothing is recorded
DEFAULT_MIN_DIST = 0.5  # the pipeline's umap_min_dist

PLOT_K = 15  # nearest cells on the plot, per cell
BODY_SHARE = 0.05  # a part of a group holding at least 5% of its cells is main body
MIN_CONTACT = 10  # contact cells (both sides) for two groups to touch
MIN_GROUP_CELLS = 5  # smaller groups are left out of the relations
MAX_GROUPS = 100
MAX_ROWS = 500  # relation rows listed (flagged and touching first)
THUMB_CELLS = 5_000  # cells per thumbnail (the same cells in every layout)
MAX_THUMB_CELLS = 20_000
SAME_TOL = 1e-3  # a layout equal to the current plot within this (max abs difference)

FIRST_RUN_SECONDS = 30  # numba compiles UMAP and the neighbour search once per server start
# Measured on 2 cores: Kang 13,836 cells 200 epochs 4.3 / 4.9 / 6.2 s per layout at
# k 10 / 15 / 30 and 2.0 s per graph; paul15 2,730 cells 500 epochs 2.0 / 2.3 / 2.9 s.
_SECONDS_PER_EPOCH_CELL = 1.6e-6
_SECONDS_PER_GRAPH_CELL = 1.5e-4

TOUCH_RULE = (
    f"Two groups touch in a layout when at least {MIN_CONTACT} of their cells (both groups "
    "together; for a smaller group, as many as it has) sit at the contact: cells in their "
    f"group's main body with a main-body cell of the other group among their {PLOT_K} nearest "
    f"cells on the plot. A group's main body is the parts of it holding at least "
    f"{BODY_SHARE:.0%} of its cells, so a stray cell or a small clump inside another island "
    "does not count."
)


class LayoutError(ValueError):
    """A request that cannot be answered (the API answers 400 / 422)."""


class Cancelled(Exception):
    """The client went away; the run stops before its next layout."""


# ---------------------------------------------------------------------------
# What the dataset recorded
# ---------------------------------------------------------------------------


@dataclass
class Recorded:
    """How the dataset's graph and UMAP were made, as far as it recorded."""

    use_rep: str
    n_pcs: int | None
    metric: str
    n_neighbors: int
    min_dist: float | None
    a: float | None = None  # the stored UMAP curve, when min_dist is not recorded
    b: float | None = None
    notes: list[str] = field(default_factory=list)

    def umap_kwargs(self) -> dict[str, float]:
        if self.min_dist is None and self.a is not None and self.b is not None:
            return {"a": self.a, "b": self.b}
        return {"min_dist": self.min_dist if self.min_dist is not None else DEFAULT_MIN_DIST}

    def to_dict(self) -> dict[str, Any]:
        return {"use_rep": self.use_rep, "n_pcs": self.n_pcs, "metric": self.metric,
                "n_neighbors": self.n_neighbors, "min_dist": self.min_dist, "a": self.a,
                "b": self.b, "notes": list(self.notes)}


def _last_params(adata, step: str) -> dict[str, Any] | None:
    for h in reversed(provenance.read_provenance(adata).get("history", [])):
        if isinstance(h, dict) and h.get("step") == step and isinstance(h.get("params"), dict):
            return h["params"]
    return None


def recorded(adata) -> Recorded:
    """The representation, k and min_dist the dataset's own layout was made with."""
    notes: list[str] = []
    _, entry = er._graph_entry(adata)
    params = entry.get("params") if isinstance(entry.get("params"), dict) else {}
    rep, n_pcs, metric = er._rep_params(adata, params)
    if rep not in adata.obsm:
        raise LayoutError(
            f"There is no {rep} to lay the cells out from: run the pipeline's PCA and "
            "neighbours steps first.")
    if not params:
        notes.append(f"No neighbour parameters are recorded: the layouts use {rep}"
                     + (f" (first {n_pcs} columns)" if n_pcs else "") + ".")
    k = params.get("n_neighbors")
    if k is None:
        k = (_last_params(adata, "neighbors") or {}).get("n_neighbors")
    if k is None:
        k = DEFAULT_K
        notes.append(f"The neighbour count is not recorded: assuming {DEFAULT_K}, scanpy's default.")
    min_dist, a, b = None, None, None
    emb = _last_params(adata, "embeddings")
    if emb is not None and emb.get("umap_min_dist") is not None:
        min_dist = float(emb["umap_min_dist"])
    else:
        curve = (adata.uns.get("umap") or {}).get("params") if isinstance(adata.uns.get("umap"), dict) else None
        if isinstance(curve, dict) and curve.get("a") is not None and curve.get("b") is not None:
            a, b = float(curve["a"]), float(curve["b"])
            notes.append("min_dist is not recorded: the layouts use the stored UMAP curve "
                         f"(a = {a:.3g}, b = {b:.3g}).")
        else:
            min_dist = DEFAULT_MIN_DIST
            notes.append(f"min_dist is not recorded: using {DEFAULT_MIN_DIST}, scView's default.")
    return Recorded(use_rep=rep, n_pcs=n_pcs, metric=metric, n_neighbors=int(k),
                    min_dist=min_dist, a=a, b=b, notes=notes)


def default_neighbors(k_current: int, n_cells: int) -> list[int]:
    """{10, the dataset's k, 30}; three distinct values below the cell count."""
    ks: list[int] = []
    for k in (*NEIGHBOR_CHOICES, k_current, *EXTRA_NEIGHBORS):
        if k not in ks and MIN_NEIGHBORS <= k < n_cells and len(ks) < 3:
            ks.append(int(k))
    if k_current not in ks and MIN_NEIGHBORS <= k_current < n_cells:
        ks[-1] = int(k_current)
    return sorted(ks)


def estimate_seconds(n_cells: int, ks: list[int], seeds: list[int]) -> float:
    """Graph + UMAP time on 2 cores (see the measured rates above), first run excluded."""
    m = min(n_cells, MAX_CELLS)
    epochs = 500 if m <= 10_000 else 200
    total = len(ks) * _SECONDS_PER_GRAPH_CELL * m
    for k in ks:
        total += len(seeds) * (0.3 + epochs * m * _SECONDS_PER_EPOCH_CELL * (0.6 + 0.4 * k / 15))
    return round(total, 1)


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------


@dataclass
class Plan:
    rec: Recorded
    seeds: list[int]
    ks: list[int]
    rows: np.ndarray  # the laid-out cells (obs indices, sorted)
    n_cells: int
    sampled: bool
    group_by: str | None
    current: str | None  # the embedding compared as "your plot" (2D)
    notes: list[str]

    @property
    def layouts(self) -> list[tuple[int, int]]:
        """(seed, k) in display order: one row per k, one column per seed."""
        return [(seed, k) for k in self.ks for seed in self.seeds]

    def to_dict(self) -> dict[str, Any]:
        return {
            "recorded": self.rec.to_dict(),
            "seeds": list(self.seeds),
            "n_neighbors": list(self.ks),
            "layouts": [{"index": i, "seed": s, "n_neighbors": k, "key": layout_key(s, k),
                         "label": layout_label(s, k)}
                        for i, (s, k) in enumerate(self.layouts)],
            "n_cells": self.n_cells,
            "n_laid_out": int(self.rows.size),
            "sampled": self.sampled,
            "group_by": self.group_by,
            "current": self.current,
            "notes": list(self.notes),
        }


def layout_key(seed: int, k: int) -> str:
    return f"X_umap_s{seed}_k{k}"


def layout_label(seed: int, k: int) -> str:
    return f"seed {seed}, {k} neighbours"


def sample_rows(n: int, max_cells: int = MAX_CELLS, seed: int = SAMPLE_SEED) -> tuple[np.ndarray, bool]:
    """Every cell up to ``max_cells``; above, a seeded sample of that many (sorted)."""
    if n <= max_cells:
        return np.arange(n), False
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=max_cells, replace=False)), True


def current_embedding(adata, requested: str | None) -> tuple[str | None, str | None]:
    """The 2D embedding compared as the user's plot, and a note when it is not the
    one asked for. A 3D or many-column embedding falls back to X_umap."""
    def two_d(name: str) -> bool:
        arr = adata.obsm[name] if name in adata.obsm else None
        return arr is not None and getattr(arr, "ndim", 0) == 2 and arr.shape[1] == 2

    if requested and requested in adata.obsm and two_d(requested):
        return requested, None
    fallback = "X_umap" if two_d("X_umap") else None
    if requested and requested in adata.obsm:
        if fallback:
            return fallback, (f"'{requested}' is not a 2D plot; the layouts are compared with "
                              f"{fallback}.")
        return None, f"'{requested}' is not a 2D plot, and there is no 2D UMAP to compare with."
    if requested and requested not in adata.obsm:
        raise KeyError(f"Embedding '{requested}' not found.")
    if fallback is None:
        return None, "There is no 2D UMAP yet: the layouts are compared with each other only."
    return fallback, None


def check_group_column(adata, group_by: str | None) -> str | None:
    if not group_by:
        col, _ = er.default_group_column(adata)
        return col
    if group_by not in adata.obs.columns:
        raise KeyError(f"Column '{group_by}' not found in obs.")
    col = adata.obs[group_by]
    if pd.api.types.is_float_dtype(col) or (pd.api.types.is_integer_dtype(col) and col.nunique() > MAX_GROUPS):
        raise LayoutError(f"'{group_by}' holds numbers, not groups: pick a grouping column.")
    n = col.nunique()
    if n > MAX_GROUPS:
        raise LayoutError(f"'{group_by}' has {n:,} groups; pick a column with at most {MAX_GROUPS}.")
    return group_by


def make_plan(adata, embedding: str | None = None, group_by: str | None = None,
              seeds: list[int] | None = None, ks: list[int] | None = None,
              max_cells: int = MAX_CELLS) -> Plan:
    rec = recorded(adata)
    n = int(adata.n_obs)
    if n < 10:
        raise LayoutError("Fewer than 10 cells: there is nothing to lay out.")
    seeds = sorted({int(s) for s in (seeds if seeds else SEEDS)})
    if len(seeds) > MAX_SEEDS or any(s < 0 or s > 2**31 - 1 for s in seeds):
        raise LayoutError(f"Pick at most {MAX_SEEDS} seeds, each a whole number from 0.")
    if ks:
        ks = sorted({int(k) for k in ks})
        bad = [k for k in ks if k < MIN_NEIGHBORS or k > MAX_NEIGHBORS or k >= n]
        if bad:
            raise LayoutError(f"Neighbour counts must be between {MIN_NEIGHBORS} and "
                              f"{min(MAX_NEIGHBORS, n - 1)}, not {bad[0]}.")
    else:
        ks = default_neighbors(rec.n_neighbors, n)
    if len(seeds) * len(ks) > MAX_LAYOUTS:
        raise LayoutError(f"{len(seeds) * len(ks)} layouts asked for; at most {MAX_LAYOUTS}.")
    rows, sampled = sample_rows(n, max_cells)
    current, note = current_embedding(adata, embedding)
    notes = list(rec.notes)
    if note:
        notes.append(note)
    notes.append(f"For each neighbour count the kNN graph is rebuilt from {rec.use_rep}"
                 + (f" (first {rec.n_pcs} columns)" if rec.n_pcs else "")
                 + "; only the seed and the neighbour count change between layouts.")
    if sampled:
        notes.append(f"{n:,} cells: a seeded sample of {rows.size:,} was laid out; your plot is "
                     "compared on the same cells.")
        if current:
            notes.append(f"Your plot was laid out with all {n:,} cells and these layouts with the "
                         "sample, which can change a layout by itself: read a contact in your plot "
                         "that no layout repeats with that in mind.")
    return Plan(rec=rec, seeds=seeds, ks=ks, rows=rows, n_cells=n, sampled=sampled,
                group_by=check_group_column(adata, group_by), current=current, notes=notes)


# ---------------------------------------------------------------------------
# One layout
# ---------------------------------------------------------------------------


def _representation(adata, rec: Recorded, rows: np.ndarray) -> np.ndarray:
    X = adata.obsm[rec.use_rep]
    X = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)
    if rec.n_pcs is not None:
        X = X[:, :rec.n_pcs]
    return np.ascontiguousarray(X[rows]) if rows.size != X.shape[0] else np.ascontiguousarray(X)


def build_graph(X: np.ndarray, k: int, metric: str = "euclidean"):
    """scanpy's kNN graph of ``X`` (an AnnData holding it), as the neighbours step builds it."""
    import anndata as ad
    import scanpy as sc

    tmp = ad.AnnData(obs=pd.DataFrame(index=[str(i) for i in range(X.shape[0])]))
    tmp.obsm["X_rep"] = X
    sc.pp.neighbors(tmp, n_neighbors=min(k, X.shape[0] - 1), use_rep="X_rep", metric=metric)
    return tmp


def umap_layout(graph_holder, seed: int, rec: Recorded) -> np.ndarray:
    """scanpy's 2D UMAP of the holder's graph with ``seed``. umap-learn trims the
    graph it is given in place, so it gets a copy (the next seed sees the same graph)."""
    import scanpy as sc

    graph = graph_holder.obsp["connectivities"]
    graph_holder.obsp["connectivities"] = graph.copy()
    try:
        sc.tl.umap(graph_holder, random_state=seed, **rec.umap_kwargs())
        return np.asarray(graph_holder.obsm.pop("X_umap"), dtype=np.float32)
    finally:
        graph_holder.obsp["connectivities"] = graph
        graph_holder.uns.pop("umap", None)


def plot_neighbours(coords: np.ndarray, k: int = PLOT_K) -> np.ndarray:
    """Each cell's ``k`` nearest other cells on the plot (KD-tree), int32."""
    m = coords.shape[0]
    k = min(k, m - 1)
    return er.plot_neighbours(coords, k, np.arange(m)).astype(np.int32)


def reference_graph(graph_holder) -> er.Graph:
    """The laid-out cells' graph neighbours (the dataset's own k), for placement."""
    m = graph_holder.n_obs
    idx, counts = er._padded_rows(graph_holder.obsp["distances"], np.arange(m))
    return er.Graph(idx=idx, count=counts, k=er._mode(counts), source="computed")


def placement(ref: er.Graph, coords: np.ndarray) -> np.ndarray:
    """Batch 13's placement score of every laid-out cell on ``coords``."""
    m = coords.shape[0]
    k = min(ref.k, ref.idx.shape[1], m - 2)
    window = min(m - 1, max(k, int(round(er.WINDOW * (m - 1)))))
    radius, _ = er.window_radius(coords.astype(np.float64), window, np.arange(m))
    return er.placement_scores(ref, coords.astype(np.float64), k, np.arange(m), radius).astype(np.float32)


@dataclass
class Layout:
    index: int  # -1: the current plot
    seed: int | None
    n_neighbors: int | None
    coords: np.ndarray  # (m, 2) float32, the laid-out cells in plan.rows order
    nn: np.ndarray  # (m, PLOT_K) int32 plot neighbours
    placement: np.ndarray  # (m,) float32
    seconds: float
    source_id: int = 0  # the current plot: id() of the obsm array it was read from

    @property
    def key(self) -> str | None:
        return None if self.seed is None else layout_key(self.seed, self.n_neighbors)

    @property
    def label(self) -> str:
        return "your plot" if self.seed is None else layout_label(self.seed, self.n_neighbors)


def make_layout(index: int, seed: int | None, k: int | None, coords: np.ndarray, ref: er.Graph,
                seconds: float = 0.0) -> Layout:
    coords = np.ascontiguousarray(coords, dtype=np.float32)
    return Layout(index=index, seed=seed, n_neighbors=k, coords=coords, nn=plot_neighbours(coords),
                  placement=placement(ref, coords), seconds=seconds)


# ---------------------------------------------------------------------------
# Relations
# ---------------------------------------------------------------------------


def main_body(nn: np.ndarray, codes: np.ndarray, n_groups: int) -> np.ndarray:
    """Per cell: is it in a part of its group holding at least ``BODY_SHARE`` of the
    group's cells (a connected component of the plot-kNN graph restricted to the
    group's own cells)? Cells without a group (code -1) are not."""
    from scipy.sparse.csgraph import connected_components

    m = codes.size
    sizes = np.bincount(codes[codes >= 0], minlength=n_groups)
    same = (codes[nn] == codes[:, None]) & (codes[:, None] >= 0)
    i, j = np.nonzero(same)
    adj = sp.csr_matrix((np.ones(i.size, dtype=np.int8), (i, nn[i, j])), shape=(m, m))
    _, comp = connected_components(adj, directed=True, connection="weak")
    csize = np.bincount(comp)
    need = np.where(codes >= 0, np.maximum(BODY_SHARE * sizes[np.maximum(codes, 0)], 1.0), np.inf)
    return csize[comp] >= need


def contact_matrix(nn: np.ndarray, codes: np.ndarray, n_groups: int,
                   body: np.ndarray | None = None) -> np.ndarray:
    """(G, G) int: ``c[a, b]`` = main-body cells of a with a main-body cell of b
    among their plot neighbours."""
    if body is None:
        body = main_body(nn, codes, n_groups)
    nc = codes[nn]
    ok = body[:, None] & body[nn] & (nc != codes[:, None]) & (nc >= 0) & (codes[:, None] >= 0)
    i, j = np.nonzero(ok)
    pairs = np.unique(i.astype(np.int64) * n_groups + nc[i, j])
    flat = codes[pairs // n_groups].astype(np.int64) * n_groups + pairs % n_groups
    return np.bincount(flat, minlength=n_groups * n_groups).reshape(n_groups, n_groups)


def touch_matrix(c: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    """(G, G) bool, symmetric: contact cells both sides >= min(MIN_CONTACT, smaller group)."""
    total = c + c.T
    need = np.minimum(MIN_CONTACT, np.minimum.outer(sizes, sizes))
    out = (total >= need) & (total > 0)
    np.fill_diagonal(out, False)
    return out


@dataclass
class Run:
    plan: Plan
    layouts: list[Layout]
    ref: er.Graph | None = None
    rep_id: int = 0
    complete: bool = False
    stopped: bool = False
    started_at: float = field(default_factory=time.time)
    seconds: float = 0.0
    thumb: np.ndarray | None = None  # positions (into plan.rows) drawn in thumbnails
    current: dict[str, Layout] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)


def _codes(adata, plan: Plan, group_by: str) -> tuple[list[str], np.ndarray, np.ndarray]:
    names, codes = group_codes(adata.obs[group_by])
    codes = np.asarray(codes)[plan.rows].astype(np.int64)
    sizes = np.bincount(codes[codes >= 0], minlength=len(names))
    # Too small to have a shape on the plot: left out.
    small = sizes < MIN_GROUP_CELLS
    if small.any():
        codes = np.where((codes >= 0) & small[np.maximum(codes, 0)], -1, codes)
        sizes = np.where(small, 0, sizes)
    return names, codes, sizes


def current_layout(adata, run: Run, embedding: str | None) -> Layout | None:
    """The user's plot on the laid-out cells, scored like a layout (cached per run)."""
    if not embedding or run.ref is None:
        return None
    with run.lock:
        hit = run.current.get(embedding)
    arr = adata.obsm[embedding] if embedding in adata.obsm else None
    if arr is None or getattr(arr, "ndim", 0) != 2 or arr.shape[1] != 2:
        return None
    if hit is not None and hit.source_id == id(arr):
        return hit
    coords = np.asarray(arr, dtype=np.float32)[run.plan.rows]
    if not np.isfinite(coords).all():
        return None
    lay = make_layout(-1, None, None, coords, run.ref)
    lay.source_id = id(arr)  # a replaced embedding misses
    with run.lock:
        run.current[embedding] = lay
    return lay


def _same(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and float(np.max(np.abs(a.astype(np.float64) - b))) < SAME_TOL


def _status(n: int, total: int) -> str:
    if total and n == total:
        return "every"
    return "never" if n == 0 else "some"


def _flag(cur: bool | None, n: int, total: int) -> str | None:
    if cur is None or not total:
        return None
    if cur and n < total / 2:
        return "rare_in_plot"
    if not cur and n > total / 2:
        return "hidden_in_plot"
    return None


_ORDER = {"rare_in_plot": 0, "hidden_in_plot": 1}
_STATUS_ORDER = {"some": 0, "every": 1, "never": 2}


def relations(names: list[str], codes: np.ndarray, sizes: np.ndarray, layouts: list[Layout],
              current: Layout | None) -> tuple[list[dict], dict]:
    """Per pair of groups: in which layouts they touch, in the current plot or not,
    the status and flag; plus the counts. Sorted: rare in the plot first."""
    G = len(names)
    L = len(layouts)
    touch = np.zeros((L, G, G), dtype=bool)
    cells = np.zeros((L, G, G), dtype=np.int64)
    for i, lay in enumerate(layouts):
        c = contact_matrix(lay.nn, codes, G)
        cells[i] = c
        touch[i] = touch_matrix(c, sizes)
    cur_touch = cur_cells = None
    if current is not None:
        cur_cells = contact_matrix(current.nn, codes, G)
        cur_touch = touch_matrix(cur_cells, sizes)
    rows = []
    counts = {"every": 0, "some": 0, "never": 0, "rare_in_plot": 0, "hidden_in_plot": 0}
    present = np.flatnonzero(sizes > 0)
    for ai, a in enumerate(present):
        for b in present[ai + 1:]:
            per = touch[:, a, b]
            n = int(per.sum())
            cur = None if cur_touch is None else bool(cur_touch[a, b])
            status = _status(n, L)
            flag = _flag(cur, n, L)
            counts[status] += 1
            if flag:
                counts[flag] += 1
            share_a = [int(x) for x in cells[:, a, b]]
            share_b = [int(x) for x in cells[:, b, a]]
            rows.append({
                "a": names[a], "b": names[b],
                "n_a": int(sizes[a]), "n_b": int(sizes[b]),
                "layouts": [bool(x) for x in per],
                "n_touch": n, "n_layouts": L, "share": round(n / L, 4) if L else None,
                "current": cur, "status": status, "flag": flag,
                "contact_a": share_a, "contact_b": share_b,
                "current_contact": None if cur_cells is None else [int(cur_cells[a, b]), int(cur_cells[b, a])],
            })
    rows.sort(key=lambda r: (_ORDER.get(r["flag"], 2), _STATUS_ORDER[r["status"]],
                             r["share"] if r["flag"] == "rare_in_plot" else -(r["share"] or 0),
                             r["a"], r["b"]))
    return rows, counts


def group_placement(names: list[str], codes: np.ndarray, sizes: np.ndarray, layouts: list[Layout],
                    current: Layout | None) -> list[dict]:
    """Per group: its median placement in each layout and the current plot, and the
    range across layouts; the widest range first."""
    out = []
    for g, name in enumerate(names):
        if sizes[g] == 0:
            continue
        mask = codes == g
        per = [float(np.nanmedian(lay.placement[mask])) for lay in layouts]
        cur = float(np.nanmedian(current.placement[mask])) if current is not None else None
        lo, hi = (min(per), max(per)) if per else (None, None)
        out.append({"group": name, "n_cells": int(sizes[g]), "placement": [round(v, 4) for v in per],
                    "current": None if cur is None else round(cur, 4),
                    "min": None if lo is None else round(lo, 4), "max": None if hi is None else round(hi, 4),
                    "range": None if lo is None else round(hi - lo, 4)})
    out.sort(key=lambda r: (-(r["range"] or 0), r["group"]))
    return out


def summarise(adata, run: Run, group_by: str | None = None, embedding: str | None = None) -> dict:
    """The JSON summary of a (possibly partial) run for one grouping."""
    plan = run.plan
    group_by = check_group_column(adata, group_by or plan.group_by)
    emb = embedding if embedding is not None else plan.current
    if embedding is not None:
        emb, _ = current_embedding(adata, embedding)
    with run.lock:
        layouts = list(run.layouts)
    cur = current_layout(adata, run, emb)
    notes = list(plan.notes)
    same = set()
    if cur is not None and not plan.sampled:
        same = {lay.index for lay in layouts if _same(lay.coords, cur.coords)}
        if layouts and not same and run.complete:
            notes.append(f"None of the layouts reproduces your plot ({emb}) exactly: it was laid "
                         "out with other settings, from another graph, or outside scView.")
    if run.stopped:
        notes.append(f"The run was stopped after {len(layouts)} of {len(plan.layouts)} layouts; "
                     "the relations are over those.")
    out: dict[str, Any] = {
        **plan.to_dict(),
        "current": emb,
        "complete": run.complete,
        "stopped": run.stopped,
        "n_done": len(layouts),
        "seconds": round(run.seconds, 1),
        "done": [{"index": lay.index, "seed": lay.seed, "n_neighbors": lay.n_neighbors,
                  "key": lay.key, "label": lay.label, "seconds": round(lay.seconds, 2),
                  "same_as_current": lay.index in same, "exists": lay.key in adata.obsm}
                 for lay in layouts],
        "group_by": group_by,
        "rule": TOUCH_RULE,
        "plot_k": PLOT_K, "min_contact": MIN_CONTACT, "body_share": BODY_SHARE,
        "relations": [], "n_pairs": 0, "n_listed": 0, "counts": {}, "groups": [],
        "notes": notes,
    }
    if not group_by:
        notes.append("No grouping column: pick one to compare relations.")
        return out
    names, codes, sizes = _codes(adata, plan, group_by)
    left_out = [n for n, s in zip(names, sizes) if s == 0]
    if left_out:
        notes.append(f"{len(left_out)} group(s) with fewer than {MIN_GROUP_CELLS} laid-out cells are "
                     "left out: " + ", ".join(left_out[:5]) + ("…" if len(left_out) > 5 else "") + ".")
    rows, counts = relations(names, codes, sizes, layouts, cur)
    out.update({
        "relations": rows[:MAX_ROWS], "n_pairs": len(rows), "n_listed": min(len(rows), MAX_ROWS),
        "counts": counts, "groups": group_placement(names, codes, sizes, layouts, cur),
        "n_groups": int((sizes > 0).sum()),
    })
    return out


# ---------------------------------------------------------------------------
# The run, cached on the loaded dataset
# ---------------------------------------------------------------------------

_cache = ObjectCache()


def latest(adata) -> Run | None:
    """The latest run for this loaded dataset (complete or not)."""
    run = _cache.get(adata)
    if run is None:
        return None
    rep = adata.obsm[run.plan.rec.use_rep] if run.plan.rec.use_rep in adata.obsm else None
    if rep is None or id(rep) != run.rep_id:
        return None  # the representation was replaced: the layouts are not of it
    return run


def _check_stop(stop: Callable[[], bool] | None) -> None:
    if stop is not None and stop():
        raise Cancelled()


LayoutFn = Callable[[Any, int, Recorded], np.ndarray]


def iter_run(adata, plan: Plan, stop: Callable[[], bool] | None = None,
             layout_fn: LayoutFn | None = None) -> Iterator[tuple[str, dict]]:
    """Lay the cells out under each (seed, k) of ``plan`` and yield
    ``("start", plan)``, one ``("layout", ...)`` per layout and ``("summary", ...)``.
    ``adata`` is not changed. ``stop`` is checked before each graph and layout;
    once it returns True the run raises :class:`Cancelled`, keeping what it laid out.
    ``layout_fn(graph_holder, seed, rec)``: for tests (default ``umap_layout``)."""
    layout_fn = layout_fn or umap_layout
    t0 = time.perf_counter()
    rec = plan.rec
    run = Run(plan=plan, layouts=[], rep_id=id(adata.obsm[rec.use_rep]))
    run.thumb = np.sort(np.random.default_rng(SAMPLE_SEED + 1).choice(
        plan.rows.size, size=min(THUMB_CELLS, plan.rows.size), replace=False))
    est = estimate_seconds(plan.n_cells, plan.ks, plan.seeds)
    yield "start", {**plan.to_dict(), "estimate_seconds": est, "first_run_seconds": FIRST_RUN_SECONDS}
    X = _representation(adata, rec, plan.rows)
    # The reference graph for placement: the dataset's own k, on the laid-out cells.
    k_ref = rec.n_neighbors if rec.n_neighbors < X.shape[0] else X.shape[0] - 1
    graphs: dict[int, Any] = {}
    _check_stop(stop)
    graphs[k_ref] = build_graph(X, k_ref, rec.metric)
    run.ref = reference_graph(graphs[k_ref])
    _cache.put(adata, run)
    try:
        for index, (seed, k) in enumerate(plan.layouts):
            _check_stop(stop)
            t = time.perf_counter()
            if k not in graphs:
                for old in [g for g in graphs if g != k_ref]:
                    del graphs[old]  # one graph besides the reference at a time
                graphs[k] = build_graph(X, k, rec.metric)
                _check_stop(stop)
            coords = layout_fn(graphs[k], seed, rec)
            lay = make_layout(index, seed, k, coords, run.ref, time.perf_counter() - t)
            with run.lock:
                run.layouts.append(lay)
            run.seconds = time.perf_counter() - t0
            yield "layout", {"index": index, "seed": seed, "n_neighbors": k, "key": lay.key,
                             "label": lay.label, "seconds": round(lay.seconds, 2),
                             "exists": lay.key in adata.obsm}
    except (Cancelled, GeneratorExit):  # stopped, or the consumer closed the generator
        run.stopped = True
        run.seconds = time.perf_counter() - t0
        raise
    run.complete = True
    run.seconds = time.perf_counter() - t0
    logger.info("Layout stability: %d layouts of %d cells in %.1f s", len(run.layouts),
                plan.rows.size, run.seconds)
    yield "summary", summarise(adata, run)


def run_all(adata, plan: Plan, **kw) -> dict:
    """The whole run; the summary (tests and scripts)."""
    out: dict = {}
    for event, data in iter_run(adata, plan, **kw):
        if event == "summary":
            out = data
    return out


# ---------------------------------------------------------------------------
# Thumbnails and keep
# ---------------------------------------------------------------------------


def thumbnail(adata, run: Run, index: str | int, max_points: int = THUMB_CELLS,
              embedding: str | None = None) -> dict:
    """One layout's coordinates for a thumbnail: the run's thumbnail cells (the same
    in every layout), ``index`` a layout number or ``"current"`` (the plot the run
    compared with, or ``embedding`` when given)."""
    plan = run.plan
    if str(index) == "current":
        name = current_embedding(adata, embedding)[0] if embedding else plan.current
        lay = current_layout(adata, run, name)
        if lay is None:
            raise LayoutError("There is no 2D plot of yours to show.")
    else:
        try:
            i = int(index)
        except (TypeError, ValueError):
            raise LayoutError(f"Layout '{index}' is not a number or 'current'.") from None
        name = None
        with run.lock:
            done = list(run.layouts)
        if i < 0 or i >= len(plan.layouts):
            raise KeyError(f"There is no layout {i}: the run has {len(plan.layouts)}.")
        if i >= len(done):
            raise LayoutError(f"Layout {i} is not laid out yet.")
        lay = done[i]
    pos = run.thumb if run.thumb is not None else np.arange(plan.rows.size)
    if max_points < pos.size:
        pos = pos[np.linspace(0, pos.size - 1, max(1, max_points)).astype(int)]
    xy = lay.coords[pos].astype(np.float64)
    return {
        "index": lay.index if lay.seed is not None else "current",
        "embedding": name if lay.seed is None else None,
        "seed": lay.seed, "n_neighbors": lay.n_neighbors, "key": lay.key, "label": lay.label,
        "cells": plan.rows[pos].astype(int).tolist(),
        "x": np.round(xy[:, 0], 3).tolist(), "y": np.round(xy[:, 1], 3).tolist(),
        "n_laid_out": int(plan.rows.size), "n_shown": int(pos.size),
    }


def keep(adata, run: Run, index: int) -> dict:
    """Save layout ``index`` as obsm ``X_umap_s<seed>_k<k>`` (unchanged when that key
    exists). A layout of a sample is laid out again on all cells with the same seed
    and neighbours. Recorded in provenance (step ``keep_layout``)."""
    with run.lock:
        done = list(run.layouts)
    if index < 0 or index >= len(done):
        raise KeyError(f"Layout {index} is not laid out in this run.")
    lay = done[index]
    plan, rec = run.plan, run.plan.rec
    key = lay.key
    if key in adata.obsm:
        return {"key": key, "created": False, "n_cells": int(adata.n_obs), "recomputed": False, "note": None}
    note = None
    if plan.sampled:
        X = _representation(adata, rec, np.arange(adata.n_obs))
        coords = umap_layout(build_graph(X, lay.n_neighbors, rec.metric), lay.seed, rec)
        note = (f"The layout shown was of a {plan.rows.size:,}-cell sample; the kept one is laid "
                f"out on all {adata.n_obs:,} cells with the same seed and neighbours, so it can "
                "differ from the thumbnail.")
    else:
        coords = lay.coords
    adata.obsm[key] = np.asarray(coords, dtype=np.float32)
    try:
        provenance.record_step(
            adata, step="keep_layout",
            tool="scanpy.pp.neighbors + scanpy.tl.umap (layout stability check)",
            params={"key": key, "seed": lay.seed, "n_neighbors": lay.n_neighbors,
                    **rec.umap_kwargs(), "use_rep": rec.use_rep, "n_pcs": rec.n_pcs,
                    "metric": rec.metric, "from": "layout stability check"},
        )
    except Exception as e:  # provenance must never block the result
        logger.warning("Could not record keep_layout provenance: %s", e)
    return {"key": key, "created": True, "n_cells": int(adata.n_obs), "recomputed": plan.sampled,
            "note": note}


# ---------------------------------------------------------------------------
# The co-pilot's fact
# ---------------------------------------------------------------------------


def _pair(r: dict) -> str:
    return f"{r['a']} and {r['b']}"


def grounding_lines(adaptor) -> tuple[list[str], str]:
    """One fact line, only when a complete run is held for this dataset: the
    relations in the current plot that appear in fewer than half of the layouts
    (and those missing from the plot but in most), with the counts."""
    adata = adaptor.adata
    run = latest(adata)
    if run is None or not run.complete:
        return [], ""
    d = summarise(adata, run)
    if not d["group_by"] or not d["relations"]:
        return [], ""
    L = len(d["done"])
    c = d["counts"]
    rare = [r for r in d["relations"] if r["flag"] == "rare_in_plot"]
    hidden = [r for r in d["relations"] if r["flag"] == "hidden_in_plot"]
    sample = f" (a seeded sample of {d['n_laid_out']:,} of {d['n_cells']:,} cells)" if d["sampled"] else ""
    head = (f"- Layout stability by '{d['group_by']}': {L} UMAP layouts of the same "
            f"{d['recorded']['use_rep']}{sample}, seeds {', '.join(map(str, d['seeds']))} x "
            f"{', '.join(map(str, d['n_neighbors']))} neighbours (graph rebuilt for each). "
            f"{TOUCH_RULE} Of {d['n_pairs']} pairs of groups, {c.get('every', 0)} touch in every "
            f"layout, {c.get('some', 0)} in some, {c.get('never', 0)} in none.")
    if d["current"] is None:
        line = head + " There is no 2D plot of the user's to compare with"
    elif rare:
        items = "; ".join(f"{_pair(r)} ({r['n_touch']} of {L} layouts)" for r in rare[:6])
        line = (head + f" Touching in the current plot ({d['current']}) but in fewer than half "
                f"of the layouts, so not to be read as biology: {items}")
    else:
        line = head + (f" Every relation in the current plot ({d['current']}) appears in at least "
                       "half of the layouts")
    if hidden:
        line += "; apart in the current plot but touching in most layouts: " + "; ".join(
            f"{_pair(r)} ({r['n_touch']} of {L})" for r in hidden[:4])
    line += f" [{TAG}]"
    detail = (f"{L} layouts by '{d['group_by']}': {len(rare)} relation(s) in the plot but rare, "
              f"{c.get('every', 0)} in every layout")
    return [line], detail
