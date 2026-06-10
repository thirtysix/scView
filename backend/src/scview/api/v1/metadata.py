"""Metadata endpoints – obs column info, values, and summaries."""

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel

from scview.dependencies import get_dataset_manager
from scview.core.dataset_manager import DatasetManager
from scview.core.arrow_serializer import series_to_arrow_ipc

router = APIRouter()

ARROW_CONTENT_TYPE = "application/vnd.apache.arrow.stream"


@router.get("/datasets/{dataset_id}/metadata")
async def list_metadata_columns(
    dataset_id: str,
    dm: DatasetManager = Depends(get_dataset_manager),
):
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    return adaptor.obs_columns_info()


@router.get("/datasets/{dataset_id}/metadata/summary")
async def metadata_summary(
    dataset_id: str,
    groupby: str = Query(default=""),
    dm: DatasetManager = Depends(get_dataset_manager),
):
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    if not groupby:
        # Default: try common categorical columns
        for candidate in ("leiden", "louvain", "seurat_clusters", "cluster", "celltype"):
            try:
                adaptor.get_obs_column(candidate)
                groupby = candidate
                break
            except KeyError:
                continue

    if not groupby:
        return {"groupby": None, "counts": {}}

    try:
        counts = adaptor.get_obs_summary(groupby)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Column '{groupby}' not found.")

    return {"groupby": groupby, "counts": counts}


@router.get("/datasets/{dataset_id}/metadata/cell/{index}")
async def get_cell_metadata(
    dataset_id: str,
    index: int,
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """Return all obs column values for a single cell at the given index."""
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    try:
        return adaptor.get_cell_metadata(index)
    except IndexError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/datasets/{dataset_id}/metadata/crosstab")
async def get_crosstab(
    dataset_id: str,
    row: str = Query(...),
    col: str = Query(...),
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """Return a cross-tabulation of two obs columns."""
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    try:
        return adaptor.get_obs_crosstab(row, col)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/datasets/{dataset_id}/metadata/{column}")
async def get_metadata_column(
    dataset_id: str,
    column: str,
    format: str = Query(default="arrow", description="Response format: arrow or json"),
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """Return values for a single obs column as Arrow IPC or JSON."""
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    try:
        series = adaptor.get_obs_column(column)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Column '{column}' not found.")

    if format == "json":
        values = series.tolist()
        return {"column": column, "values": values}

    ipc_bytes = series_to_arrow_ipc(series, name=column)
    return Response(content=ipc_bytes, media_type=ARROW_CONTENT_TYPE)


class RenameCategoryBody(BaseModel):
    old: str
    new: str


@router.post("/datasets/{dataset_id}/metadata/{column}/rename")
async def rename_obs_category(
    dataset_id: str,
    column: str,
    body: RenameCategoryBody,
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """Rename a category in a categorical obs column (e.g. correct a cell-type label).

    Persists to the derived layer, never the original upload. If the new label already
    exists, the two categories are merged.
    """
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    new = body.new.strip()
    if not new:
        raise HTTPException(status_code=400, detail="New label must not be empty.")

    adata = adaptor.adata
    if getattr(adata, "isbacked", False):
        adata = adata.to_memory()
    if column not in adata.obs.columns:
        raise HTTPException(status_code=404, detail=f"Column '{column}' not found.")
    col = adata.obs[column]
    if col.dtype.name != "category":
        col = col.astype("category")
    cats = list(col.cat.categories)
    if body.old not in cats:
        raise HTTPException(status_code=404, detail=f"'{body.old}' is not a category of '{column}'.")
    if new in cats and new != body.old:
        adata.obs[column] = col.astype(str).replace({body.old: new}).astype("category")
    else:
        adata.obs[column] = col.cat.rename_categories({body.old: new})

    output_path = str(dm.derived_h5ad_path(dataset_id, adaptor.h5ad_path))
    adata.write_h5ad(output_path)
    # Reload so subsequent reads see the rename.
    if dataset_id in dm._datasets:
        dm._datasets[dataset_id].close()
        del dm._datasets[dataset_id]
    if dataset_id in dm._load_order:
        dm._load_order.remove(dataset_id)
    await dm.load_dataset(dataset_id)
    return {"column": column, "old": body.old, "new": new}
