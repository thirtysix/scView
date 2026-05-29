"""Expression / gene query endpoints."""

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response

from scview.dependencies import get_dataset_manager
from scview.core.dataset_manager import DatasetManager
from scview.core.arrow_serializer import expression_to_arrow_ipc

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
