"""Trying several clustering resolutions without changing the dataset.

Every resolution is clustered on the dataset's own neighbour graph with the
method and seed the pipeline uses (``_run_clustering``), and Leiden with the passes
the dataset's clustering was made with (fast Leiden or the full optimisation,
``leiden.recorded_iterations``), so keeping one gives exactly the labels the pipeline
would have made at that resolution, under the pipeline's name for them
(``scview_leiden_r0.5``).

For each resolution the sweep reports the number of clusters and their sizes,
a silhouette score in the representation the graph was built from (on a fixed,
seeded subsample of cells, so the scores are comparable), and the adjusted Rand
index to the previous resolution: how much the partition changed. Once every
resolution is done it adds the clustering tree across them
(``core/cluster_tree.py``).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd

from scview.core import cluster_tree, leiden, provenance
from scview.core.obs_writes import forget_column_results

logger = logging.getLogger(__name__)

METHODS = ("leiden", "louvain")
DEFAULT_RESOLUTIONS = (0.2, 0.4, 0.6, 0.8, 1.0, 1.5)
MAX_RESOLUTIONS = 8
MIN_RESOLUTION, MAX_RESOLUTION = 0.01, 5.0
# Leiden takes about 7 s per resolution on 50k cells and grows with the graph;
# results stream one resolution at a time, so no single wait is that long.
MAX_CELLS = 200_000
SILHOUETTE_CELLS = 5_000
SEED = 0  # the subsample's; clustering uses the pipeline's own seed (0)


class SweepError(ValueError):
    """A sweep that cannot run (no neighbour graph, too many cells): a 400."""


def column_name(method: str, resolution: float) -> str:
    """The pipeline's column for a clustering (``_run_clustering``)."""
    return f"scview_{method}_r{float(resolution)}"


def existing_column(adata, method: str, resolution: float) -> str | None:
    """The dataset's column for this clustering, if it has one.

    A whole-number resolution has two spellings: the pipeline names the column
    from the value it was given, and one sent from the browser as JSON ``1``
    arrives as an int (``scview_leiden_r1``), not ``1.0``.
    """
    names = [column_name(method, resolution)]
    if float(resolution).is_integer():
        names.append(f"scview_{method}_r{int(resolution)}")
    return next((n for n in names if n in adata.obs.columns), None)


def clean_resolutions(values: list[float] | None) -> list[float]:
    """Resolutions rounded to 0.01, deduplicated and sorted; SweepError if unusable."""
    if not values:
        return list(DEFAULT_RESOLUTIONS)
    out: list[float] = []
    for v in values:
        r = round(float(v), 2)
        if not np.isfinite(r) or not MIN_RESOLUTION <= r <= MAX_RESOLUTION:
            raise SweepError(
                f"Resolutions must be between {MIN_RESOLUTION} and {MAX_RESOLUTION} (got {v})."
            )
        out.append(r)
    out = sorted(set(out))
    if len(out) > MAX_RESOLUTIONS:
        raise SweepError(f"Try at most {MAX_RESOLUTIONS} resolutions at a time.")
    return out


def _graph(adata):
    nb = adata.uns.get("neighbors")
    key = nb.get("connectivities_key", "connectivities") if isinstance(nb, dict) else None
    if key is None or key not in adata.obsp:
        return None
    return adata.obsp[key]


def check(adata) -> None:
    """Raise SweepError when this dataset cannot be swept, saying why."""
    if _graph(adata) is None:
        raise SweepError(
            "This dataset has no neighbour graph, which clustering runs on. Run the "
            "processing pipeline (through the neighbours step) in Data Assessment first."
        )
    if adata.n_obs > MAX_CELLS:
        raise SweepError(
            f"The sweep runs on at most {MAX_CELLS:,} cells; this dataset has {adata.n_obs:,}. "
            "Re-cluster at a chosen resolution from Data Assessment instead."
        )


def dataset_method(adata) -> str:
    """The method the dataset was clustered with (Leiden when unknown)."""
    recorded = (provenance.read_provenance(adata)["current"].get("clustering") or {}).get("method")
    if recorded in METHODS:
        return recorded
    m = re.match(r"scview_(leiden|louvain)_r", str(adata.uns.get("scview_active_clustering") or ""))
    return m.group(1) if m else "leiden"


def cluster(adata, method: str, resolution: float, n_iterations: int | None = None) -> pd.Categorical:
    """Cluster labels at ``resolution``, numbered by size; ``adata`` is not changed.
    Leiden makes the passes ``n_iterations`` says, by default as many as the dataset's
    clustering was made with (``leiden.recorded_iterations``: fast Leiden's 2, or until
    none improves), so a resolution kept is the pipeline's clustering at it."""
    from scview.core.pipeline import _louvain_igraph

    graph = _graph(adata)
    if method == "leiden":
        if n_iterations is None:
            n_iterations = leiden.recorded_iterations(adata)
        return leiden.labels(graph, resolution, n_iterations=n_iterations)
    work = ad.AnnData(obs=pd.DataFrame(index=adata.obs_names.copy()))
    work.obsp["connectivities"] = graph
    _louvain_igraph(work, resolution, "c")
    return pd.Categorical(work.obs["c"])


def _representation(adata) -> tuple[str, np.ndarray] | tuple[None, None]:
    """The space the neighbour graph was built in (PCA, or Harmony-corrected PCA)."""
    params = (adata.uns.get("neighbors") or {}).get("params") or {}
    rep = params.get("use_rep")
    if rep not in adata.obsm:
        rep = next((r for r in ("X_pca_harmony", "X_pca") if r in adata.obsm), None)
    if rep is None:
        return None, None
    X = np.asarray(adata.obsm[rep])
    n_pcs = params.get("n_pcs")
    if isinstance(n_pcs, (int, np.integer)) and 0 < n_pcs < X.shape[1]:
        X = X[:, : int(n_pcs)]
    return rep, X


def iter_sweep(adata, resolutions: list[float], method: str) -> Iterator[dict[str, Any]]:
    """Cluster at each resolution and score it, one resolution at a time.

    Yields a header (method, cells, the silhouette's representation and
    subsample size), one result per resolution, then ``{"tree": ...}``, the
    clustering tree across them. ``adata`` is not changed.
    """
    from sklearn.metrics import adjusted_rand_score, silhouette_score

    check(adata)
    rep, X = _representation(adata)
    sub = None
    if X is not None:
        rng = np.random.default_rng(SEED)
        sub = np.sort(rng.choice(adata.n_obs, size=min(adata.n_obs, SILHOUETTE_CELLS), replace=False))
        X = np.asarray(X[sub], dtype=np.float32)
    n_iterations = leiden.recorded_iterations(adata) if method == "leiden" else None
    yield {
        "method": method,
        # Leiden's passes, as the dataset's clustering made them (-1: until none improves)
        "leiden_iterations": n_iterations,
        "n_cells": int(adata.n_obs),
        "resolutions": resolutions,
        "representation": rep,
        "silhouette_cells": None if sub is None else int(sub.size),
    }

    prev: np.ndarray | None = None
    levels: list[tuple[float, np.ndarray, list[str]]] = []
    for r in resolutions:
        existing = existing_column(adata, method, r)
        # A resolution the dataset already has a column for is shown as that
        # column, which "Keep this one" returns: re-clustering may not reproduce it
        # (files processed before the UMAP graph fix carry an altered graph).
        labels = (pd.Categorical(adata.obs[existing].astype(str).where(adata.obs[existing].notna()))
                  if existing is not None else cluster(adata, method, r, n_iterations))
        codes = np.asarray(labels.codes)
        names = [str(c) for c in labels.categories]
        if (codes < 0).any():  # missing labels: one group
            codes = np.where(codes < 0, len(names), codes)
            names.append("(no label)")
        levels.append((r, codes, names))
        sizes = np.sort(np.bincount(codes))[::-1]
        silhouette = None
        if sub is not None and 2 <= np.unique(codes[sub]).size < sub.size:
            silhouette = float(silhouette_score(X, codes[sub]))
        yield {
            "resolution": r,
            "n_clusters": int(sizes.size),
            "sizes": sizes.astype(int).tolist(),
            "silhouette": silhouette,
            "ari_previous": None if prev is None else float(adjusted_rand_score(prev, codes)),
            "column": existing or column_name(method, r),
            "exists": existing is not None,
        }
        prev = codes
    yield {"tree": cluster_tree.build(levels)}


def sweep(adata, resolutions: list[float], method: str) -> dict[str, Any]:
    """:func:`iter_sweep` collected: the header with ``results`` and ``tree`` added."""
    rows = iter_sweep(adata, resolutions, method)
    head = next(rows)
    *results, tree = rows
    return {**head, "results": results, **tree}


def keep(adata, method: str, resolution: float) -> dict[str, Any]:
    """Save the clustering at ``resolution`` as its pipeline column, unless that
    column exists already. Does not change the dataset's active clustering."""
    existing = existing_column(adata, method, resolution)
    if existing is not None:
        return {"column": existing, "created": False, "n_clusters": int(adata.obs[existing].nunique())}
    column = column_name(method, resolution)
    check(adata)
    n_iterations = leiden.recorded_iterations(adata) if method == "leiden" else None
    labels = cluster(adata, method, resolution, n_iterations)
    adata.obs[column] = labels
    forget_column_results(adata, column)  # left by an earlier column of that name
    params: dict[str, Any] = {"clustering_method": method, "clustering_resolution": resolution,
                              "column": column, "from": "resolution sweep"}
    if n_iterations is not None:
        params["leiden_iterations"] = n_iterations
    try:
        provenance.record_step(
            adata, step="keep_resolution",
            tool="scanpy.tl.leiden" if method == "leiden" else "igraph multilevel (Louvain)",
            params=params,
        )
    except Exception as e:  # provenance must never block the result
        logger.warning("Could not record keep_resolution provenance: %s", e)
    return {"column": column, "created": True, "n_clusters": int(len(labels.categories))}
