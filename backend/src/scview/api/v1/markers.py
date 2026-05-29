"""Marker gene endpoints."""

import logging

import math

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response

from scview.dependencies import get_dataset_manager
from scview.core.dataset_manager import DatasetManager
from scview.core.arrow_serializer import dataframe_to_arrow_ipc

logger = logging.getLogger(__name__)

router = APIRouter()

ARROW_CONTENT_TYPE = "application/vnd.apache.arrow.stream"


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

    If groupby_column is provided, computes markers on-the-fly for that column.
    """
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    # On-demand marker computation for a specific obs column
    if groupby_column:
        # First check for pre-computed per-column markers
        if adaptor.has_markers(column=groupby_column):
            df = adaptor.get_markers(
                groupby=groupby if groupby else None,
                column=groupby_column,
                n_genes=n_genes,
            )
        else:
            # Fall back to on-demand computation
            try:
                import scanpy as sc

                work_adata = adaptor.adata.to_memory() if hasattr(adaptor.adata, "to_memory") else adaptor.adata.copy()
                if groupby_column not in work_adata.obs.columns:
                    raise HTTPException(status_code=404, detail=f"Column '{groupby_column}' not found in obs.")

                sc.tl.rank_genes_groups(
                    work_adata,
                    groupby=groupby_column,
                    method="wilcoxon",
                    pts=True,
                    n_genes=n_genes,
                )
                rgg = work_adata.uns["rank_genes_groups"]
                groups = list(rgg["names"].dtype.names) if hasattr(rgg["names"].dtype, "names") else []
                rows = []
                for group in groups:
                    total_genes = len(rgg["names"][group])
                    limit = min(n_genes, total_genes)
                    for i in range(limit):
                        gene_name = str(rgg["names"][group][i])
                        row = {"group": str(group), "gene": gene_name}
                        if "logfoldchanges" in rgg:
                            row["logfoldchange"] = float(rgg["logfoldchanges"][group][i])
                        if "pvals" in rgg:
                            row["pval"] = float(rgg["pvals"][group][i])
                        if "pvals_adj" in rgg:
                            row["pval_adj"] = float(rgg["pvals_adj"][group][i])
                        if "pts" in rgg and rgg["pts"] is not None:
                            pts = rgg["pts"]
                            try:
                                if hasattr(pts, "loc"):
                                    row["pct_in"] = float(pts.loc[gene_name, group])
                                elif hasattr(pts.dtype, "names") and group in pts.dtype.names:
                                    row["pct_in"] = float(pts[group][i])
                            except (KeyError, IndexError):
                                pass
                        if "pts_rest" in rgg and rgg["pts_rest"] is not None:
                            pts_rest = rgg["pts_rest"]
                            try:
                                if hasattr(pts_rest, "loc"):
                                    row["pct_out"] = float(pts_rest.loc[gene_name, group])
                                elif hasattr(pts_rest.dtype, "names") and group in pts_rest.dtype.names:
                                    row["pct_out"] = float(pts_rest[group][i])
                            except (KeyError, IndexError):
                                pass
                        rows.append(row)

                if not rows:
                    raise HTTPException(status_code=404, detail="No marker genes computed.")

                import pandas as pd
                df = pd.DataFrame(rows)

            except HTTPException:
                raise
            except Exception as e:
                logger.error("On-demand marker computation failed: %s", e)
                raise HTTPException(status_code=500, detail=f"Marker computation failed: {e!s}")
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
