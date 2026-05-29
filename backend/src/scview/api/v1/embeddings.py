"""Embedding endpoints – serve UMAP/tSNE/PCA coordinates as Arrow IPC."""

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response

from scview.dependencies import get_dataset_manager
from scview.core.dataset_manager import DatasetManager
from scview.core.arrow_serializer import embedding_to_arrow_ipc

router = APIRouter()

ARROW_CONTENT_TYPE = "application/vnd.apache.arrow.stream"


@router.get("/datasets/{dataset_id}/embeddings")
async def list_embeddings(
    dataset_id: str,
    dm: DatasetManager = Depends(get_dataset_manager),
):
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    return adaptor.available_embeddings()


@router.get("/datasets/{dataset_id}/embeddings/{name}")
async def get_embedding(
    dataset_id: str,
    name: str,
    color_by: str = Query(default="", description="Obs column to use for coloring"),
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """Return embedding coordinates as Arrow IPC binary.

    Optionally includes a color column from obs metadata.
    """
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    try:
        coords = adaptor.get_embedding(name)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

    color_values = None
    if color_by:
        try:
            col = adaptor.get_obs_column(color_by)
            if hasattr(col, "cat") or col.dtype.name == "category":
                color_values = col.cat.codes.values.astype(np.int32)
            elif np.issubdtype(col.dtype, np.number):
                color_values = col.values.astype(np.float32)
            else:
                # String column — encode as integer categories
                categories = {v: i for i, v in enumerate(col.unique())}
                color_values = col.map(categories).values.astype(np.int32)
        except KeyError:
            pass  # Ignore invalid color_by, return without color

    ipc_bytes = embedding_to_arrow_ipc(coords, color_values, color_name=color_by or "color")
    return Response(content=ipc_bytes, media_type=ARROW_CONTENT_TYPE)
