"""Embedding endpoints – serve UMAP/tSNE/PCA coordinates as Arrow IPC."""

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response

from scview.dependencies import get_dataset_manager
from scview.core.dataset_manager import DatasetManager
from scview.core.arrow_serializer import embedding_to_arrow_ipc

router = APIRouter()

ARROW_CONTENT_TYPE = "application/vnd.apache.arrow.stream"


def parse_dims(dims: str, n_cols: int) -> list[int]:
    """Two distinct 0-based column indices from ``"i,j"``, each under ``n_cols``."""
    try:
        idx = [int(p) for p in dims.split(",")]
    except ValueError:
        raise HTTPException(status_code=400, detail=f"dims must be two integers like '0,1', not '{dims}'.")
    if len(idx) != 2 or idx[0] == idx[1] or any(i < 0 or i >= n_cols for i in idx):
        raise HTTPException(
            status_code=400,
            detail=f"dims must name two different columns between 0 and {n_cols - 1}, not '{dims}'.")
    return idx


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
    dims: str = Query(default="", description="Two 0-based columns to serve as x,y (e.g. '2,3'); default the first two"),
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """Return embedding coordinates as Arrow IPC binary.

    Optionally includes a color column from obs metadata. ``dims`` picks which two
    columns of a many-column embedding (a PCA) become x and y.
    """
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    try:
        coords = adaptor.get_embedding(name)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    if dims:
        coords = coords[:, parse_dims(dims, coords.shape[1])]

    color_values = None
    if color_by:
        try:
            col = adaptor.get_obs_column(color_by)
            if hasattr(col, "cat") or col.dtype.name == "category":
                color_values = col.cat.codes.values.astype(np.int32)
            elif pd.api.types.is_bool_dtype(col):
                # Fixed codes matching the listed values ["False", "True"]; the
                # string branch below numbered them in order of first appearance.
                color_values = np.where(col.isna(), -1, col.fillna(False).astype(int)).astype(np.int32)
            elif np.issubdtype(col.dtype, np.number):
                color_values = col.values.astype(np.float32)
                # inf (pseudotime of a cell the root never reached) stretched the
                # colour range to infinity, flattening every finite value; NaN is
                # left out of the range and drawn grey.
                color_values[np.isinf(color_values)] = np.nan
            else:
                # String column — encode as integer categories
                categories = {v: i for i, v in enumerate(col.unique())}
                color_values = col.map(categories).values.astype(np.int32)
        except KeyError:
            pass  # Ignore invalid color_by, return without color

    ipc_bytes = embedding_to_arrow_ipc(coords, color_values, color_name=color_by or "color")
    return Response(content=ipc_bytes, media_type=ARROW_CONTENT_TYPE)
