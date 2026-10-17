"""Sub-clustering one group on its own graph.

A cluster that mixes two cell types at the dataset-wide resolution (on Kang,
CD8 T cells and NK cells share one Leiden cluster at 0.5) often separates once
its own cells get their own variable genes, PCA and neighbour graph: genes that
vary within the group are swamped by the dataset-wide differences otherwise.
Raising the global resolution instead splits every other cluster too.

The result goes into ``<column>_sub``: a copy of the column in which the group
is replaced by ``<group>.0``, ``<group>.1``, ... (numbered by size). Splitting
another group of the same column, or the same group again, edits that column.
"""

from __future__ import annotations

import logging
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd

from scview.core import leiden, provenance
from scview.core.obs_writes import forget_column_results

logger = logging.getLogger(__name__)

MIN_CELLS = 30
DEFAULT_RESOLUTION = 0.5
MIN_RESOLUTION, MAX_RESOLUTION = 0.01, 5.0
N_TOP_GENES = 2000
N_PCS = 30
N_NEIGHBORS = 15
SEED = 0


class SubclusterError(ValueError):
    """A request that can't be served (too few cells, unknown group): a 400."""


def target_column(column: str) -> str:
    return column if column.endswith("_sub") else f"{column}_sub"


def _log_expression(adata: ad.AnnData, mask: np.ndarray) -> ad.AnnData:
    """The group's cells, log-normalised, all genes: from counts when the dataset
    keeps them, else .raw or X when they hold log-normalised values."""
    import scanpy as sc

    from scview.core.pipeline import looks_like_counts

    def _from_counts(X, var_names) -> ad.AnnData:
        sub = ad.AnnData(X=X.astype(np.float32), var=pd.DataFrame(index=var_names))
        sc.pp.normalize_total(sub, target_sum=1e4)
        sc.pp.log1p(sub)
        return sub

    if "counts" in adata.layers:
        return _from_counts(adata.layers["counts"][mask], adata.var_names)
    sources = [(adata.raw.X, adata.raw.var_names)] if adata.raw is not None else []
    for X, names in [*sources, (adata.X, adata.var_names)]:
        X = X[mask]
        if looks_like_counts(X):
            return _from_counts(X, names)
        if float(X.min()) >= 0:  # log-normalised (scaled data has negatives)
            return ad.AnnData(X=X.astype(np.float32), var=pd.DataFrame(index=names))
    raise SubclusterError(
        "Sub-clustering needs counts or log-normalised expression; this dataset only has "
        "scaled values. Re-run the pipeline from the counts."
    )


def _ordered_categories(base: pd.Series, group: str, new: list[str]) -> list[str]:
    """The column's own order, with the group (or its earlier split) replaced
    in place by its new sub-clusters."""
    cats = list(base.cat.categories) if hasattr(base, "cat") else sorted(base.dropna().unique())
    out: list[str] = []
    for c in map(str, cats):
        if c == group or c.startswith(f"{group}."):
            if new[0] not in out:
                out.extend(new)
        else:
            out.append(c)
    return out + [c for c in new if c not in out]


def subcluster(
    adata: ad.AnnData,
    column: str,
    group: str,
    resolution: float = DEFAULT_RESOLUTION,
    batch_key: str | None = None,
) -> dict[str, Any]:
    """Split ``group`` of ``column`` into sub-clusters on its own graph, writing
    ``target_column(column)``. ``batch_key`` (the dataset's Harmony key) is
    corrected for within the group too, as for the whole dataset."""
    import scanpy as sc

    if column not in adata.obs.columns:
        raise SubclusterError(f"Column '{column}' not found.")
    if not MIN_RESOLUTION <= resolution <= MAX_RESOLUTION:
        raise SubclusterError(f"Resolution must be between {MIN_RESOLUTION} and {MAX_RESOLUTION}.")
    labels = adata.obs[column]
    mask = (labels.astype(str).where(labels.notna()) == group).to_numpy()
    n = int(mask.sum())
    if n == 0:
        raise SubclusterError(f"No cells have {column} = '{group}'.")
    if n < MIN_CELLS:
        raise SubclusterError(f"'{group}' has {n} cells; sub-clustering needs at least {MIN_CELLS}.")

    sub = _log_expression(adata, mask)
    sub.obs_names = adata.obs_names[mask]
    use_batch = bool(batch_key and batch_key in adata.obs.columns
                     and adata.obs.loc[mask, batch_key].nunique() >= 2)
    if use_batch:
        sub.obs[batch_key] = adata.obs.loc[mask, batch_key].astype(str).to_numpy()

    sc.pp.highly_variable_genes(sub, n_top_genes=min(N_TOP_GENES, sub.n_vars), flavor="seurat")
    sub = sub[:, sub.var["highly_variable"]].copy()
    sc.pp.scale(sub, max_value=10)
    n_comps = max(2, min(N_PCS, n - 1, sub.n_vars - 1))
    sc.tl.pca(sub, n_comps=n_comps, random_state=SEED)
    rep = "X_pca"
    if use_batch:
        from scview.core.pipeline import PipelineParams, _run_batch_correction

        _run_batch_correction(sub, PipelineParams(batch_key=batch_key))
        rep = "X_pca_harmony"
    sc.pp.neighbors(sub, n_neighbors=min(N_NEIGHBORS, n - 1), use_rep=rep, random_state=SEED)
    leiden.leiden(sub, resolution, "s", random_state=SEED)
    codes = sub.obs["s"].astype(str).to_numpy()  # Leiden numbers communities by size
    new_cats = [f"{group}.{k}" for k in sorted(set(codes), key=int)]

    target = target_column(column)
    base = adata.obs[target] if target in adata.obs.columns else labels
    values = base.astype(str).where(base.notna()).astype(object).to_numpy()
    values[mask] = [f"{group}.{k}" for k in codes]
    adata.obs[target] = pd.Categorical(values, categories=_ordered_categories(base, group, new_cats))
    forget_column_results(adata, target)  # markers / enrichment of the old groups

    sizes = [int((codes == k.split(".")[-1]).sum()) for k in new_cats]
    params = {"column": column, "group": group, "resolution": resolution, "target": target,
              "n_top_genes": int(sub.n_vars), "n_pcs": n_comps, "n_neighbors": min(N_NEIGHBORS, n - 1),
              "batch_key": batch_key if use_batch else None}
    try:
        provenance.record_step(adata, step="subcluster",
                               tool="scanpy HVG + PCA + neighbors + leiden on the group's cells",
                               params=params)
    except Exception as e:  # provenance must never block the result
        logger.warning("Could not record sub-clustering provenance: %s", e)
    return {"column": target, "group": group, "n_cells": n, "n_subclusters": len(new_cats),
            "groups": new_cats, "sizes": sizes, "representation": rep,
            "batch_key": batch_key if use_batch else None}
