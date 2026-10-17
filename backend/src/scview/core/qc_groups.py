"""QC metrics per group of one obs column: per sample, batch or cluster.

The Data Assessment histograms pool every cell, which hides one bad sample or a
cluster of damaged cells. Here each group's per-cell QC values (genes and counts
per cell, % mito when measurable) are summarised for violins, and the
counts-vs-genes scatter is sampled with each point's group, for colouring.

The payload does not grow with the cell count: a group's distribution is sent as
its quantiles (``N_QUANTILES`` values from 0 to 100 %), not as its cells, and the
scatter is a sample stratified by group. Plotly draws a violin from those values
as it would from the raw ones, and its box quartiles are exact because the 25th,
50th and 75th percentiles are among the quantiles sent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scview.core import cluster_qc
from scview.core.confounding import batch_columns, condition_columns, recorded_batch_key
from scview.core.replicates import donor_candidates, pick_donor

MAX_GROUPS = 50  # more violins than this are unreadable side by side
N_QUANTILES = 201  # 0, 0.5 %, ..., 100 %
N_SCATTER = 5000  # as the pooled scatter (AnnDataAdaptor.qc_distributions)
MIN_SCATTER_PER_GROUP = 30  # a small sample stays visible next to a large one

# (metric name in the response, key in AnnDataAdaptor.qc_per_cell's arrays)
METRICS = (
    ("n_genes_by_counts", "n_genes"),
    ("total_counts", "total_counts"),
    ("pct_counts_mt", "pct_mt"),
)


def _n_groups(obs: pd.DataFrame, col: str) -> int:
    return int(obs[col].nunique(dropna=True))


def _usable(obs: pd.DataFrame, col: str | None) -> bool:
    return bool(col) and col in obs.columns and 2 <= _n_groups(obs, col) <= MAX_GROUPS


def sample_grouping(adata) -> tuple[str | None, str | None]:
    """(column, why) for the samples: the donor/sample column, picked as the DE
    replicate guard and the QC report pick it (core/replicates.py), else a batch
    column. Each with 2 to MAX_GROUPS groups."""
    obs = adata.obs
    # A sample column that only repeats the condition loses to a real donor column.
    candidates = [c for c in donor_candidates(obs) if _usable(obs, c)]
    levels = [(obs[c].astype(str) == lvl).to_numpy()
              for c in condition_columns(obs) for lvl in obs[c].astype(str).unique()]
    donor = pick_donor(obs, candidates, levels)
    if donor:
        return donor, "donor"
    key = recorded_batch_key(adata)
    batches = [c for c in batch_columns(obs, [key] if key else []) if _usable(obs, c)]
    if batches:
        return batches[0], "batch"
    return None, None


def default_grouping(adata) -> tuple[str | None, str | None]:
    """(column, why) when none is named: the samples (``sample_grouping``), else
    the clustering, with 2 to MAX_GROUPS groups."""
    col, reason = sample_grouping(adata)
    if col:
        return col, reason
    clustering = cluster_qc.default_grouping(adata)
    if _usable(adata.obs, clustering):
        return clustering, "clustering"
    return None, None


def _group_order(labels: pd.Series, present: set[str]) -> list[str]:
    """The column's own category order (cluster 0, 1, 2 ...), as cluster QC shows it."""
    if hasattr(labels, "cat"):
        order = [str(c) for c in labels.cat.categories]
    else:
        order = sorted(present, key=lambda s: (0, int(s), "") if s.isdigit() else (1, 0, s))
    return [g for g in order if g in present]


def group_codes(raw: pd.Series) -> tuple[list[str], np.ndarray]:
    """The groups of one obs column in its own order, and each cell's index into
    them (-1 = no value)."""
    if isinstance(raw.dtype, pd.CategoricalDtype):
        # The codes as they are, minus unused categories: no string per cell. The QC
        # filter preview asks on every threshold change (200k cells: 66 -> 11 ms).
        codes = raw.cat.codes.to_numpy()
        used = np.unique(codes[codes >= 0])
        remap = np.full(len(raw.cat.categories) + 1, -1)  # code -1 reads the last slot
        remap[used] = np.arange(used.size)
        return [str(raw.cat.categories[i]) for i in used], remap[codes]
    labels = raw.astype(str).where(raw.notna())
    order = _group_order(raw, set(labels.dropna().unique()))
    return order, np.asarray(pd.Categorical(labels, categories=order).codes)


def members(codes: np.ndarray, k: int) -> list[np.ndarray]:
    """Each group's cell indices (``codes`` from ``group_codes``), in one sort
    rather than one pass over the cells per group."""
    idx = np.argsort(codes, kind="stable")
    ordered = codes[idx]
    starts = np.searchsorted(ordered, np.arange(k), side="left")
    ends = np.searchsorted(ordered, np.arange(k), side="right")
    return [idx[s:e] for s, e in zip(starts, ends)]


def _quantiles(values: np.ndarray, k: int) -> list[float]:
    """``k`` evenly spaced quantiles of ``values``, or all of them, sorted, when fewer."""
    values = values[np.isfinite(values)]
    out = np.sort(values) if values.size <= k else np.quantile(values, np.linspace(0, 1, k))
    return np.round(out, 3).tolist()


def _stratified_sample(members: list[np.ndarray], n: int, floor: int) -> np.ndarray:
    """Cell indices for the scatter: each group's share of ``n``, at least ``floor``
    cells (or all it has), evenly spaced through the group."""
    total = sum(m.size for m in members)
    if total <= n:
        return np.sort(np.concatenate(members)) if members else np.array([], dtype=int)
    picks = []
    for m in members:
        k = min(m.size, max(floor, round(n * m.size / total)))
        picks.append(m if k >= m.size else m[np.linspace(0, m.size - 1, k).astype(int)])
    return np.sort(np.concatenate(picks))


def no_grouping(adata) -> dict:
    """The response when no column was named and none could be picked."""
    return {
        "column": None, "reason": None, "n_cells": int(adata.n_obs), "n_groups": 0,
        "max_groups": MAX_GROUPS, "groups": [], "metrics": {}, "scatter": None,
        "note": "No sample, batch or clustering column found: pick a column to group by.",
    }


def qc_by_group(adata, column: str, cells: dict, reason: str = "requested", *,
                n_quantiles: int = N_QUANTILES, n_scatter: int = N_SCATTER) -> dict:
    """Per-group QC distributions of ``column`` (``cells`` = AnnDataAdaptor.qc_per_cell()).

    Raises KeyError for a missing column and ValueError for one with no labelled
    cells or more than MAX_GROUPS groups."""
    if column not in adata.obs.columns:
        raise KeyError(f"Column '{column}' not found in obs.")
    order, codes = group_codes(adata.obs[column])
    if not order:
        raise ValueError(f"'{column}' has no labelled cells.")
    if len(order) > MAX_GROUPS:
        raise ValueError(
            f"'{column}' has {len(order)} groups; QC by group shows at most {MAX_GROUPS}. "
            "Pick a column with fewer groups, such as the sample, batch or clustering."
        )
    members = [np.flatnonzero(codes == i) for i in range(len(order))]

    has_mt = bool(cells.get("has_mt", True))
    metrics: dict[str, dict[str, list[float]]] = {}
    for name, key in METRICS:
        if name == "pct_counts_mt" and not has_mt:
            continue  # no MT- genes: zeros everywhere would read as "clean"
        values = np.asarray(cells[key], dtype=np.float64)
        metrics[name] = {g: _quantiles(values[m], n_quantiles) for g, m in zip(order, members)}

    idx = _stratified_sample(members, n_scatter, MIN_SCATTER_PER_GROUP)
    total_counts = np.asarray(cells["total_counts"], dtype=np.float64)
    n_genes = np.asarray(cells["n_genes"], dtype=np.float64)
    return {
        "column": column,
        "reason": reason,
        "n_cells": int(adata.n_obs),
        "n_groups": len(order),
        "max_groups": MAX_GROUPS,
        "n_unlabelled": int((codes < 0).sum()),
        "has_mt": has_mt,
        "computed_on_demand": bool(cells.get("computed")),
        "n_quantiles": n_quantiles,
        "groups": [{"name": g, "n_cells": int(m.size)} for g, m in zip(order, members)],
        "metrics": metrics,
        "scatter": {
            "x": np.round(total_counts[idx], 3).tolist(),
            "y": np.round(n_genes[idx], 3).tolist(),
            "group": codes[idx].astype(int).tolist(),  # index into ``groups``
            "x_label": "total_counts",
            "y_label": "n_genes_by_counts",
            "n_shown": int(idx.size),
        },
    }
