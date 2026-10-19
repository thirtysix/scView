"""Per-cell expression of a few genes in a stratified sample of cells, for the
marker heatmap's "Per cell" view (C034).

A heatmap of every cell is unreadable past a few thousand columns and lets the
largest group fill the width, so each group gives at most ``per_group`` cells.
If that still comes to more than ``max_cells``, every group's share is scaled
down in proportion (largest remainders fill up to ``max_cells``), and a group
with cells always keeps at least one, so a total can exceed ``max_cells`` only
when there are more groups than that.

Sampling is seeded and deterministic: each group draws from its own generator,
seeded by the seed and the group's label, and takes a prefix of one random
permutation of its cells. So a group's sample does not depend on the other
groups, and the 100-cell sample of a group is part of its 200-cell sample.

Cells come out grouped in the column's stored category order
(``core/category_order.py``; a text or bool column is sorted naturally), and in
their original order within a group.
"""

from __future__ import annotations

import json
import zlib
from typing import Literal

import numpy as np
import pandas as pd
import pyarrow as pa

from scview.core.category_order import natural_key

Scale = Literal["zscore", "none"]

MAX_GENES = 100
MAX_GROUPS = 200
# Seurat's DoHeatmap draws scaled data clipped to -2.5..2.5 (disp.min / disp.max),
# so a few extreme cells cannot wash out the colour of everything else.
ZSCORE_CLIP = 2.5
METADATA_KEY = b"scview"


class HeatmapCellsError(ValueError):
    """A request that cannot be served; ``status`` is the HTTP code."""

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


def group_codes(col: pd.Series, name: str = "") -> tuple[list[str], np.ndarray]:
    """The non-empty groups of a label column in its stored order, and each
    cell's index into them (-1: no label). Raises for a numeric column."""
    if isinstance(col.dtype, pd.CategoricalDtype):
        labels = [str(c) for c in col.cat.categories]
        codes = col.cat.codes.to_numpy().astype(np.int64)
    elif pd.api.types.is_bool_dtype(col) or col.dtype == object or pd.api.types.is_string_dtype(col):
        text = col.map(lambda v: str(v) if pd.notna(v) else None)
        labels = sorted(text.dropna().unique(), key=natural_key)
        codes = pd.Categorical(text, categories=labels).codes.astype(np.int64)
    else:
        raise HeatmapCellsError(f"'{name}' is numeric; group the cells by a label column.")
    counts = np.bincount(codes[codes >= 0], minlength=len(labels))
    keep = np.flatnonzero(counts > 0)
    if keep.size > MAX_GROUPS:
        raise HeatmapCellsError(f"'{name}' has {keep.size} groups; at most {MAX_GROUPS} fit a heatmap.")
    remap = np.full(len(labels) + 1, -1, dtype=np.int64)  # the last slot maps code -1
    remap[keep] = np.arange(keep.size)
    return [labels[i] for i in keep], remap[codes]


def sample_quota(sizes, per_group: int, max_cells: int) -> np.ndarray:
    """Cells to take from each group: min(size, per_group), scaled down in
    proportion when the total exceeds ``max_cells`` (see the module docstring)."""
    sizes = np.asarray(sizes, dtype=np.int64)
    want = np.minimum(sizes, per_group)
    total = int(want.sum())
    if total <= max_cells:
        return want
    exact = want * (max_cells / total)
    base = np.minimum(np.maximum(np.floor(exact).astype(np.int64), (sizes > 0).astype(np.int64)), want)
    spare = max_cells - int(base.sum())
    if spare > 0:
        frac = exact - np.floor(exact)
        for k in sorted(range(len(sizes)), key=lambda i: (-frac[i], i)):  # ties: the earlier group
            if spare == 0:
                break
            if base[k] < want[k]:
                base[k] += 1
                spare -= 1
    return base


def sample_cells(codes: np.ndarray, groups: list[str], quota: np.ndarray, seed: int) -> np.ndarray:
    """Row indices of the sample, grouped in ``groups`` order, ascending within a group."""
    out = []
    for k, label in enumerate(groups):
        members = np.flatnonzero(codes == k)
        q = int(quota[k])
        if q < members.size:
            rng = np.random.default_rng([seed, zlib.crc32(label.encode("utf-8"))])
            members = np.sort(members[rng.permutation(members.size)[:q]])
        out.append(members)
    return np.concatenate(out) if out else np.zeros(0, dtype=np.int64)


def read_cells(X, rows: np.ndarray, cols: list[int]) -> np.ndarray:
    """Dense float32 (rows x cols) of a sparse, dense or backed matrix, rows in
    the order given. Rows are read in ascending order (an on-disk matrix needs it)."""
    order = np.argsort(rows, kind="stable")
    block = X[rows[order]]
    block = block[:, cols]
    dense = block.toarray() if hasattr(block, "toarray") else np.asarray(block)
    out = np.empty_like(dense, dtype=np.float32)
    out[order] = dense
    return np.nan_to_num(out, nan=0.0)


def zscore(values: np.ndarray) -> np.ndarray:
    """Each gene (column) centred and divided by its standard deviation (ddof=1,
    as R's ``scale`` and scanpy's ``pp.scale``) over the cells given, then
    clipped to +-ZSCORE_CLIP. A gene with no variation is 0 throughout."""
    v = values.astype(np.float64)
    if v.shape[0] < 2:
        return np.zeros_like(v, dtype=np.float32)
    mean = v.mean(axis=0)
    sd = v.std(axis=0, ddof=1)
    z = np.divide(v - mean, sd, out=np.zeros_like(v), where=sd > 0)
    return np.clip(z, -ZSCORE_CLIP, ZSCORE_CLIP).astype(np.float32)


def heatmap_cells(
    X,
    var_names,
    column: pd.Series,
    genes: list[str],
    *,
    groupby: str = "",
    layer: str = "",
    per_group: int = 200,
    max_cells: int = 5000,
    seed: int = 0,
    scale: Scale = "zscore",
    layer_range: tuple[float, float] | None = None,
) -> dict:
    """The sampled cells' values of ``genes`` (cells x genes, float32) with the
    cells' rows and group codes and the facts the plot labels itself with.
    ``layer_range`` is the layer's fixed colour range, used for scale "none"."""
    genes = [g for g in dict.fromkeys(g.strip() for g in genes) if g]
    if not genes:
        raise HeatmapCellsError("No genes specified.")
    if len(genes) > MAX_GENES:
        raise HeatmapCellsError(f"At most {MAX_GENES} genes per heatmap ({len(genes)} asked for).")
    index = {str(g): i for i, g in enumerate(var_names)}
    found = [g for g in genes if g in index]
    missing = [g for g in genes if g not in index]
    if not found:
        raise HeatmapCellsError(f"None of the genes are in layer '{layer}': {', '.join(missing[:10])}.")

    groups, codes = group_codes(column, groupby)
    n_total = np.bincount(codes[codes >= 0], minlength=len(groups))
    quota = sample_quota(n_total, per_group, max_cells)
    rows = sample_cells(codes, groups, quota, seed)
    values = read_cells(X, rows, [index[g] for g in found])
    if scale == "zscore":
        values = zscore(values)
        value_range = [-ZSCORE_CLIP, ZSCORE_CLIP]
    else:
        lo, hi = layer_range if layer_range is not None else (0.0, float(values.max(initial=1.0)))
        value_range = [float(lo), float(hi)]
    return {
        "values": values,
        "cells": rows.astype(np.int32),
        "codes": codes[rows].astype(np.int32),
        "meta": {
            "groupby": groupby,
            "layer": layer,
            "genes": found,
            "missing_genes": missing,
            "groups": groups,
            "n_sampled": [int(q) for q in quota],
            "n_total": [int(n) for n in n_total],
            "n_cells": int(len(codes)),
            "scale": scale,
            "value_range": value_range,
            "seed": int(seed),
            "per_group": int(per_group),
            "max_cells": int(max_cells),
        },
    }


def to_arrow_ipc(result: dict) -> bytes:
    """Arrow IPC stream: column 0 ``cell`` (row in the dataset, int32), column 1
    ``group`` (index into meta.groups, int32), then one float32 column per gene in
    meta.genes order (read them by position: a gene may be called "cell").
    The JSON ``meta`` rides in the schema metadata under ``scview``."""
    values = result["values"]
    arrays = [pa.array(result["cells"], type=pa.int32()), pa.array(result["codes"], type=pa.int32())]
    arrays += [pa.array(np.ascontiguousarray(values[:, j]), type=pa.float32()) for j in range(values.shape[1])]
    names = ["cell", "group", *result["meta"]["genes"]]
    batch = pa.RecordBatch.from_arrays(arrays, names=names)
    batch = batch.replace_schema_metadata({METADATA_KEY: json.dumps(result["meta"]).encode("utf-8")})
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, batch.schema) as writer:
        writer.write_batch(batch)
    return sink.getvalue().to_pybytes()
