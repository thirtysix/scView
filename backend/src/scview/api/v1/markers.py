"""Marker gene endpoints."""

import logging

import math

import anndata as ad
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response

from scview.dependencies import get_dataset_manager
from scview.core.dataset_manager import DatasetManager
from scview.core.arrow_serializer import dataframe_to_arrow_ipc
from scview.core.pipeline import drop_enrichment_cache
from scview.core.uns_keys import markers_key

logger = logging.getLogger(__name__)

router = APIRouter()

ARROW_CONTENT_TYPE = "application/vnd.apache.arrow.stream"

# The pipeline's limit too: every gene is kept per group, so a column with
# thousands of levels (barcodes, free text) would cost a lot to compute and store.
MAX_MARKER_GROUPS = 100


def _compute_markers(adata: ad.AnnData, column: str) -> dict:
    """Wilcoxon markers for every group of ``column``, as the pipeline computes them.

    Runs on a light AnnData sharing the dataset's matrix rather than on a copy of
    the whole dataset, and leaves the dataset's own obs column as it is.
    """
    import scanpy as sc

    if getattr(adata, "isbacked", False):
        adata = adata.to_memory()
    # scanpy ranks .raw when there is one; give it that matrix directly.
    use_raw = adata.raw is not None
    src = adata.raw if use_raw else adata
    labels = adata.obs[column]
    # rank_genes_groups needs a categorical groupby; coerce bool / numeric /
    # object columns (e.g. predicted_doublet) the way the pipeline does.
    if not isinstance(labels.dtype, pd.CategoricalDtype):
        labels = labels.astype(str).astype("category")
    work = ad.AnnData(
        X=src.X,
        obs=pd.DataFrame({column: labels.values}, index=adata.obs_names),
        var=pd.DataFrame(index=src.var_names),
    )
    sc.tl.rank_genes_groups(work, groupby=column, method="wilcoxon", pts=True, use_raw=False)
    rgg = work.uns["rank_genes_groups"]
    rgg["params"]["use_raw"] = use_raw  # what the ranking actually read
    return rgg


@router.get("/datasets/{dataset_id}/markers")
async def get_markers(
    dataset_id: str,
    groupby: str = Query(default="", description="Filter to a specific group"),
    groupby_column: str = Query(default="", description="Compute markers for this obs column"),
    format: str = Query(default="arrow", description="Response format: arrow or json"),
    n_genes: int = Query(default=100, ge=1, le=5000, description="Max genes per group"),
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """Return marker gene table from rank_genes_groups.

    If groupby_column is provided, returns that column's markers, computing and
    saving them the first time. ``groupby`` limits the table to one group.
    """
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    # Markers for a specific obs column: stored ones, else computed once and stored
    # under the column's key (as the pipeline does), so the next visit is instant.
    if groupby_column:
        if not adaptor.has_markers(column=groupby_column):
            adata = adaptor.adata
            if groupby_column not in adata.obs.columns:
                raise HTTPException(status_code=404, detail=f"Column '{groupby_column}' not found in obs.")
            n_groups = adata.obs[groupby_column].nunique()
            if n_groups < 2:
                raise HTTPException(
                    status_code=400,
                    detail=f"Column '{groupby_column}' has only one group; need at least two to compute markers.",
                )
            if n_groups > MAX_MARKER_GROUPS:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Column '{groupby_column}' has {n_groups} groups; markers are computed "
                        f"for at most {MAX_MARKER_GROUPS}."
                    ),
                )
            try:
                key = markers_key(groupby_column)
                adata.uns[key] = _compute_markers(adata, groupby_column)
                drop_enrichment_cache(adata, groupby_column)
            except Exception as e:
                logger.error("On-demand marker computation failed: %s", e)
                raise HTTPException(status_code=500, detail=f"Marker computation failed: {e!s}")
            try:
                dm.persist_uns(dataset_id, adaptor, [key])
            except Exception as e:  # still served from memory; recomputed after a reload
                logger.warning("Could not save markers for '%s': %s", groupby_column, e)
        df = adaptor.get_markers(
            groupby=groupby if groupby else None,
            column=groupby_column,
            n_genes=n_genes,
        )
    else:
        if not adaptor.has_markers():
            raise HTTPException(
                status_code=404,
                detail="No marker genes found. Run sc.tl.rank_genes_groups first.",
            )
        df = adaptor.get_markers(groupby=groupby if groupby else None, n_genes=n_genes)

    if df is None or df.empty:
        raise HTTPException(status_code=404, detail="No marker genes found for the specified group.")

    if format == "json":
        records = df.to_dict(orient="records")
        # Replace NaN/Inf with None so JSON serialization succeeds
        for rec in records:
            for key, val in rec.items():
                if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
                    rec[key] = None
        # Group by cluster/group to match frontend MarkersResponse shape
        groups = list(dict.fromkeys(r["group"] for r in records))
        markers_by_group = {g: [r for r in records if r["group"] == g] for g in groups}
        return {"groups": groups, "markers": markers_by_group}

    ipc_bytes = dataframe_to_arrow_ipc(df)
    return Response(content=ipc_bytes, media_type=ARROW_CONTENT_TYPE)
