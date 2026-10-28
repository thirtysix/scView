"""Spatial endpoints: the tissue images of a spatial dataset (``core/spatial.py``).

``GET /datasets/{id}/spatial/{library_id}/image?res=lowres|hires``: one library's
tissue image as PNG, encoded once per open dataset and cached; 404 for an unknown
library or a resolution the library has no image at. The spot coordinates come from
the embeddings endpoint (``/embeddings/spatial``); what the dataset holds (libraries,
image sizes and scale factors, bounds) from ``GET /datasets/{id}`` (``spatial``).
"""

from __future__ import annotations

import asyncio
import hashlib
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response

from scview.core import spatial
from scview.core.dataset_manager import DatasetManager
from scview.dependencies import get_dataset_manager

router = APIRouter()


def _etag(dataset_id: str, library_id: str, res: str) -> str:
    digest = hashlib.sha256(f"{dataset_id}\n{library_id}\n{res}".encode()).hexdigest()[:24]
    return f'"{digest}"'


@router.get("/datasets/{dataset_id}/spatial/{library_id}/image")
async def get_tissue_image(
    dataset_id: str,
    library_id: str,
    request: Request,
    res: Literal["lowres", "hires"] = Query(default="lowres"),
    dm: DatasetManager = Depends(get_dataset_manager),
):
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    adata = adaptor.adata
    if not spatial.has_spatial(adata):
        raise HTTPException(status_code=404, detail="This dataset has no spatial coordinates.")
    headers = {"Cache-Control": "private, max-age=3600", "ETag": _etag(dataset_id, library_id, res)}
    try:
        spatial.image_array(adata, library_id, res)  # 404 before anything else
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e).strip("'\""))
    if request.headers.get("if-none-match") == headers["ETag"]:
        return Response(status_code=304, headers=headers)
    # Encoding a 2,000 x 2,000 hires image takes a moment: off the loop (cached after).
    data = await asyncio.to_thread(spatial.image_png, adata, library_id, res)
    return Response(content=data, media_type="image/png", headers=headers)
