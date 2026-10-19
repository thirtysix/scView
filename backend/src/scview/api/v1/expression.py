"""Expression / gene query endpoints."""

import asyncio
from typing import Literal

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel, Field

from scview.dependencies import get_dataset_manager
from scview.core.dataset_manager import DatasetManager
from scview.core.arrow_serializer import expression_to_arrow_ipc
from scview.core import coexpression, donor_expression, heatmap_cells

router = APIRouter()

ARROW_CONTENT_TYPE = "application/vnd.apache.arrow.stream"


@router.get("/datasets/{dataset_id}/expression")
async def get_expression(
    dataset_id: str,
    genes: str = Query(default="", description="Comma-separated gene names"),
    layer: str = Query(default="", description="Expression layer key (e.g. raw, X, counts)"),
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """Return expression values for one or more genes as Arrow IPC."""
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    gene_list = [g.strip() for g in genes.split(",") if g.strip()]
    if not gene_list:
        raise HTTPException(status_code=400, detail="No genes specified.")

    layer_key = layer if layer else None
    try:
        expr = adaptor.get_expression(gene_list, layer=layer_key)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Only include genes that were found in the selected layer
    try:
        layer_var = adaptor._var_names_for_layer(layer_key or adaptor.default_expression_layer())
    except KeyError:
        layer_var = adaptor.gene_names()
    found_genes = [g for g in gene_list if g in layer_var]

    if expr.shape[1] == 0:
        layer_label = layer if layer else adaptor.default_expression_layer()
        raise HTTPException(
            status_code=404,
            detail=f"None of the requested genes were found in layer '{layer_label}': {gene_list}. Try switching to a different expression layer.",
        )

    ipc_bytes = expression_to_arrow_ipc(expr, found_genes)
    return Response(content=ipc_bytes, media_type=ARROW_CONTENT_TYPE)


@router.get("/datasets/{dataset_id}/expression/violin")
async def get_expression_violin(
    dataset_id: str,
    gene: str = Query(..., description="Gene name"),
    groupby: str = Query(default="", description="Obs column to group by"),
    layer: str = Query(default="", description="Expression layer key (e.g. raw, X, counts)"),
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """Return expression values grouped by obs column for violin plot."""
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    # Use active clustering column as default if groupby not specified
    if not groupby:
        groupby = adaptor.active_clustering_column() or "leiden"

    layer_key = layer if layer else None
    try:
        result = adaptor.get_expression_for_violin(gene, groupby, layer=layer_key)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

    if not result:
        layer_label = layer if layer else adaptor.default_expression_layer()
        raise HTTPException(
            status_code=404,
            detail=f"Gene '{gene}' not found in layer '{layer_label}'. Try switching to a different expression layer.",
        )

    return {"gene": gene, "groupby": groupby, "groups": result}


@router.get("/datasets/{dataset_id}/expression/per-donor")
async def get_expression_per_donor(
    dataset_id: str,
    gene: str = Query(..., description="Gene name"),
    groupby: str = Query(default="", description="Obs column to group by"),
    sample: str = Query(default="", description="Donor / sample column (default: detected)"),
    condition: str = Query(default="", description="Condition column to split each group by"),
    method: str = Query(default="mean", description="mean (log-normalised) or cpm (log2 CPM of summed counts)"),
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """One pseudobulk value per donor (x condition) in each group, for a box +
    strip plot (core/donor_expression.py). No donor column: 200 with a message."""
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    if not groupby:
        groupby = adaptor.active_clustering_column() or ""
    if not groupby:
        raise HTTPException(status_code=400, detail="No grouping column.")

    def _run() -> dict:
        return donor_expression.per_donor_expression(
            adaptor.adata, gene, groupby, sample_key=sample or None,
            condition_key=condition or None, method=method,  # type: ignore[arg-type]
        )

    try:
        # A column of every cell, and the counts' row sums for CPM: off the loop.
        return await asyncio.to_thread(_run)
    except donor_expression.DonorExpressionError as exc:
        status = 404 if "not found" in str(exc) else 400
        raise HTTPException(status_code=status, detail=str(exc))


@router.get("/datasets/{dataset_id}/expression/by-group")
async def get_expression_by_group(
    dataset_id: str,
    genes: list[str] = Query(..., description="Gene names (repeat the parameter)"),
    groupby: str = Query(default="", description="Obs column to group by"),
    layer: str = Query(default="", description="Expression layer key (e.g. raw, X, counts)"),
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """Mean expression (over all cells and over expressing cells only) and the
    fraction of expressing cells per group, with the layer's fixed colour range,
    for the marker dot plot and heatmap."""
    if len(genes) > 300:
        raise HTTPException(status_code=400, detail="At most 300 genes per request.")
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    if not groupby:
        groupby = adaptor.active_clustering_column() or ""
    try:
        # Reads every cell's values for up to 300 genes: off the loop.
        result = await asyncio.to_thread(adaptor.expression_by_group, genes, groupby, layer=layer or None)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    if not result["genes"]:
        raise HTTPException(status_code=404, detail="None of the requested genes are in this layer.")
    return result


@router.get("/datasets/{dataset_id}/expression/heatmap-cells")
async def get_heatmap_cells(
    dataset_id: str,
    genes: str = Query(..., description="Comma-separated gene names (at most 100)"),
    groupby: str = Query(default="", description="Label column to group the cells by"),
    layer: str = Query(default="", description="Expression layer key (default: the dataset's)"),
    per_group: int = Query(default=200, ge=1, le=2000, description="Most cells sampled per group"),
    max_cells: int = Query(default=5000, ge=100, le=20000, description="Most cells in all"),
    seed: int = Query(default=0, ge=0, le=2**32 - 1),
    scale: Literal["zscore", "none"] = Query(default="zscore"),
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """Per-cell values of a few genes in a seeded, stratified sample of cells,
    grouped in the column's stored order, for the marker heatmap's "Per cell"
    view (core/heatmap_cells.py). Arrow IPC: ``cell``, ``group``, one column per
    gene; genes, groups, cells sampled and in total per group, the scale and its
    range and the genes not found ride in the schema metadata (``scview``)."""
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    if not groupby:
        groupby = adaptor.active_clustering_column() or ""
    if groupby not in adaptor.adata.obs.columns:
        raise HTTPException(status_code=404, detail=f"Column '{groupby}' not found.")
    layer_key = layer or adaptor.default_expression_layer()
    try:
        var_names = adaptor._var_names_for_layer(layer_key)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e))

    def _run() -> bytes:
        result = heatmap_cells.heatmap_cells(
            adaptor._get_matrix_for_layer(layer_key), var_names, adaptor.adata.obs[groupby],
            genes.split(","), groupby=groupby, layer=layer_key, per_group=per_group,
            max_cells=max_cells, seed=seed, scale=scale,
            layer_range=adaptor.layer_scale_range(layer_key) if scale == "none" else None,
        )
        return heatmap_cells.to_arrow_ipc(result)

    try:
        # Reads a few thousand rows of the matrix (and the layer's range): off the loop.
        ipc_bytes = await asyncio.to_thread(_run)
    except heatmap_cells.HeatmapCellsError as exc:
        raise HTTPException(status_code=exc.status, detail=str(exc))
    return Response(content=ipc_bytes, media_type=ARROW_CONTENT_TYPE)


class CoexpressionRequest(BaseModel):
    """A query gene, and the cells to correlate over: every cell, a selection
    (``indices``) or one group (``column`` = ``value``)."""

    gene: str = Field(min_length=1, max_length=200)
    method: Literal["pearson", "spearman"] = "pearson"
    indices: list[int] | None = None
    column: str | None = None
    value: str | None = None
    top_n: int = Field(default=coexpression.TOP_N, ge=1, le=coexpression.MAX_TOP_N)


@router.post("/datasets/{dataset_id}/expression/coexpression")
async def get_coexpression(
    dataset_id: str,
    body: CoexpressionRequest,
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """The genes most positively and most negatively correlated with ``gene``
    across the chosen cells, on the log-normalised expression the overlays use
    (core/coexpression.py), with the number of cells used and of genes skipped."""
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    adata = adaptor.adata
    if body.indices is not None and body.column:
        raise HTTPException(status_code=400, detail="Send a selection or a group, not both.")
    rows = None
    scope = "all cells"
    if body.indices is not None:
        rows = np.asarray(body.indices, dtype=np.int64)
        scope = "selection"
    elif body.column:
        if body.column not in adata.obs.columns:
            raise HTTPException(status_code=404, detail=f"Column '{body.column}' not found.")
        if body.value is None:
            raise HTTPException(status_code=400, detail="Name the group's value too.")
        rows = np.flatnonzero(adata.obs[body.column].astype(str).to_numpy() == body.value)
        scope = f"{body.column} = {body.value}"

    def _run() -> dict:
        return coexpression.coexpression(adata, body.gene, rows=rows, method=body.method,
                                         top_n=body.top_n)

    try:
        # One pass over every gene of the matrix: off the loop.
        out = await asyncio.to_thread(_run)
    except coexpression.CoexpressionError as exc:
        status = 404 if "not found" in str(exc) else 400
        raise HTTPException(status_code=status, detail=str(exc))
    return {**out, "scope": scope}


@router.get("/datasets/{dataset_id}/genes")
async def list_genes(
    dataset_id: str,
    dm: DatasetManager = Depends(get_dataset_manager),
):
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    return {"genes": adaptor.gene_names()}


@router.get("/datasets/{dataset_id}/genes/search")
async def search_genes(
    dataset_id: str,
    q: str = Query(default="", description="Search prefix"),
    limit: int = Query(default=20, ge=1, le=100),
    dm: DatasetManager = Depends(get_dataset_manager),
):
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    if not q:
        return {"query": q, "results": []}

    results = adaptor.search_genes(q, limit=limit)
    return {"query": q, "results": results}
