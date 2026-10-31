"""Spatial endpoints: the tissue images of a spatial dataset (``core/spatial.py``).

``GET /datasets/{id}/spatial/{library_id}/image?res=lowres|hires``: one library's
tissue image as PNG, encoded once per open dataset and cached; 404 for an unknown
library or a resolution the library has no image at. The spot coordinates come from
the embeddings endpoint (``/embeddings/spatial``); what the dataset holds (libraries,
image sizes and scale factors, bounds) from ``GET /datasets/{id}`` (``spatial``).

``POST /datasets/{id}/spatial/outlines?name=``: a Xenium boundaries file as the raw request body
(streamed to disk), read by row group into the outline store when it fits the dataset's cells.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from starlette.requests import ClientDisconnect

from scview.config import Settings
from scview.core import cell_outlines, raw_upload, spatial, xenium
from scview.core.dataset_manager import DatasetManager
from scview.dependencies import get_dataset_manager, get_settings_dep

logger = logging.getLogger(__name__)

router = APIRouter()

_uploading_outlines: set[tuple[str, str]] = set()  # (dataset, kind) being written


@router.get("/datasets/{dataset_id}/spatial/outlines")
async def get_outlines(
    dataset_id: str,
    kind: Literal["cell", "nucleus"] = Query(default="cell"),
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """The segmented outlines, one per cell in obs order (wire format: ``core/cell_outlines.py``).
    404 when this dataset has none of that kind."""
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    path = cell_outlines.store_path(dm.data_dir, dataset_id, kind)
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"This dataset has no {kind} outlines.")
    obs_names = adaptor.adata.obs_names

    def work() -> bytes:
        return cell_outlines.encode(*cell_outlines.read_for(path, obs_names))

    return Response(content=await asyncio.to_thread(work), media_type="application/octet-stream")


@router.post("/datasets/{dataset_id}/spatial/outlines")
async def upload_outlines(dataset_id: str, request: Request, name: str = Query(max_length=512),
                          settings: Settings = Depends(get_settings_dep),
                          dm: DatasetManager = Depends(get_dataset_manager)):
    """One boundaries file as the raw body: ``curl --data-binary @cell_boundaries.parquet ...?name=...``.
    Refused, with the counts, when under half the cells match it by id and position. What the
    dataset has now: ``outlines``, with the file's fit."""
    part = xenium.outline_part(name)
    if part is None:
        raise HTTPException(status_code=422, detail=f"'{name}' is not a Xenium boundaries file: send "
                                                    "cell_boundaries.parquet or nucleus_boundaries.parquet.")
    raw_upload.refuse_multipart(request)
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    adata = adaptor.adata
    if not spatial.has_spatial(adata):
        raise HTTPException(status_code=422, detail="This dataset has no spatial coordinates, so outlines have "
                                                    "nothing to sit on: import the Xenium run's folder.")
    key = (dataset_id, part)
    if key in _uploading_outlines:
        raise HTTPException(status_code=409, detail=f"This dataset's {name} is already being uploaded.")
    _uploading_outlines.add(key)
    folder = cell_outlines.store_dir(dm.data_dir, dataset_id)
    # Its own name, so the reader knows the format from the name it was sent with; never a store.
    held = folder / f".upload-{part}{'.csv.gz' if name.lower().endswith('.gz') else '.parquet'}"
    try:
        await asyncio.to_thread(folder.mkdir, parents=True, exist_ok=True)
        await raw_upload.stream_to_file(request, held, settings.max_upload_bytes, settings.MAX_UPLOAD_SIZE_MB)
        fit = await asyncio.to_thread(xenium.write_outlines, held, part, adata, dm.data_dir, dataset_id, name)
    except xenium.XeniumError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except raw_upload.UploadRefused as e:
        raise HTTPException(status_code=e.status, detail=str(e))
    except ClientDisconnect:
        raise HTTPException(status_code=400, detail="The upload stopped before the end: send the file again.")
    except OSError as e:
        logger.warning("Could not store %s for %s: %s", name, dataset_id, e)
        raise HTTPException(status_code=507, detail=f"Could not store {name} on this machine's disk: free some "
                                                    "space and send it again.")
    finally:
        held.unlink(missing_ok=True)
        _uploading_outlines.discard(key)
    return {**fit, "outlines": cell_outlines.kinds_of(dm.data_dir, dataset_id)}


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
