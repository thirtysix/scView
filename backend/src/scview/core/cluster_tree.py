"""A clustering tree across resolutions (clustree, Zappia & Oshlack 2018).

Each resolution of a sweep is a level of the tree and each of its clusters a
node, sized by its cells. An edge joins a cluster at one resolution to a
cluster at the next, weighted by the cells they share: ``count``, and
``in_prop``, the share of the child's cells that came from that parent.
Edges under ``PROP_FILTER`` of the child are dropped, as clustree's default
``prop_filter = 0.1`` does.

Two signs of an unstable cluster are reported per node:

- ``sc3_stability``: clustree's SC3 stability index (Kiselev et al. 2017, as
  ``calc_sc3_stability_cluster`` computes it): high when the cluster's cells stay
  together at the other resolutions, lower the more clusters they spread over.
  It reads every resolution, so it is computed once the sweep has finished.
- ``mixed``: the cluster draws ``MIXED_PROP`` (20%) or more of its cells from
  each of two or more parents at the resolution before. A tree that only splits
  has no mixed nodes; cells moving between branches, which clustree draws as
  crossing edges, make them.
"""

from __future__ import annotations

from typing import Any

import numpy as np

PROP_FILTER = 0.1  # clustree's default: edges carrying under 10% of the child's cells are not drawn
MIXED_PROP = 0.2  # a parent giving a child this share of its cells counts towards "mixed"


def _table(a: np.ndarray, b: np.ndarray, ka: int, kb: int) -> np.ndarray:
    return np.bincount(a * kb + b, minlength=ka * kb).reshape(ka, kb)


def build(levels: list[tuple[float, np.ndarray, list[str]]]) -> dict[str, Any]:
    """The tree for ``levels``: (resolution, codes 0..k-1 per cell, cluster names),
    lowest resolution first; all levels label the same cells."""
    # Categorical codes are int8 under 128 groups: a * kb + b would overflow.
    levels = [(r, np.asarray(codes, dtype=np.int64), names) for r, codes, names in levels]
    sizes = [np.bincount(codes, minlength=len(names)) for _, codes, names in levels]
    n_levels = len(levels)

    # SC3 stability (clustree's formula): for cluster i at level k, over every other
    # level l, the clusters j of l holding any of its cells (N_l of them) add
    # |i & j| / (|j| * N_l^2); the sum is divided by the number of levels.
    stability = [np.zeros(len(names)) for _, _, names in levels]
    for k in range(n_levels):
        for m in range(n_levels):
            if m == k:
                continue
            t = _table(levels[k][1], levels[m][1], len(levels[k][2]), len(levels[m][2])).astype(np.float64)
            n_hit = (t > 0).sum(axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                part = (t / np.maximum(sizes[m], 1)[None, :]).sum(axis=1) / np.maximum(n_hit, 1) ** 2
            stability[k] += part
    if n_levels:
        stability = [s / n_levels for s in stability]

    edges: list[dict[str, Any]] = []
    n_parents = [np.zeros(len(names), dtype=int) for _, _, names in levels]
    mixed = [np.zeros(len(names), dtype=bool) for _, _, names in levels]
    for k in range(1, n_levels):
        (r0, c0, n0), (r1, c1, n1) = levels[k - 1], levels[k]
        t = _table(c0, c1, len(n0), len(n1))
        with np.errstate(divide="ignore", invalid="ignore"):
            in_prop = np.nan_to_num(t / np.maximum(sizes[k], 1)[None, :])
        n_parents[k] = (in_prop >= PROP_FILTER).sum(axis=0)
        mixed[k] = (in_prop >= MIXED_PROP).sum(axis=0) >= 2
        for p, c in zip(*np.nonzero(in_prop >= PROP_FILTER)):
            edges.append({
                "from_resolution": r0, "from": n0[p], "to_resolution": r1, "to": n1[c],
                "count": int(t[p, c]), "in_prop": round(float(in_prop[p, c]), 4),
            })

    nodes = [
        {"resolution": r, "cluster": name, "size": int(sizes[k][i]),
         "sc3_stability": round(float(stability[k][i]), 4),
         "n_parents": int(n_parents[k][i]), "mixed": bool(mixed[k][i])}
        for k, (r, _, names) in enumerate(levels)
        for i, name in enumerate(names)
        if sizes[k][i] > 0
    ]
    return {
        "prop_filter": PROP_FILTER,
        "mixed_prop": MIXED_PROP,
        "nodes": nodes,
        "edges": edges,
        "n_mixed": int(sum(int(m.sum()) for m in mixed)),
    }
