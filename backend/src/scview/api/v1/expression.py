"""Expression / gene query endpoints."""

import asyncio

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response

from scview.dependencies import get_dataset_manager
from scview.core.dataset_manager import DatasetManager
from scview.core.arrow_serializer import expression_to_arrow_ipc
from scview.core import donor_expression

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
