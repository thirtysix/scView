"""Clustering resolution sweep (``core/resolution_sweep.py``).

``GET /clustering/sweep`` says whether a sweep can run and with what defaults;
``POST /clustering/sweep`` streams one SSE ``result`` per resolution without
changing the dataset; ``POST /clustering/keep`` saves one resolution's labels.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from scview.core import resolution_sweep as rs
from scview.core.dataset_manager import DatasetManager
from scview.core.obs_writes import dataset_lock, write_derived
from scview.dependencies import get_dataset_manager

logger = logging.getLogger(__name__)

router = APIRouter()

_DONE = object()
_running: set[str] = set()  # datasets with a sweep streaming: one at a time each


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


class SweepRequest(BaseModel):
    resolutions: list[float] | None = None  # default rs.DEFAULT_RESOLUTIONS
    method: str | None = None  # default: the dataset's clustering method


class KeepRequest(BaseModel):
    resolution: float
    method: str | None = None


def _method(adata, requested: str | None) -> str:
    method = requested or rs.dataset_method(adata)
    if method not in rs.METHODS:
        raise HTTPException(status_code=400, detail=f"Unknown clustering method '{method}'.")
    return method


async def _adata(dm: DatasetManager, dataset_id: str):
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    return adaptor


@router.get("/datasets/{dataset_id}/clustering/sweep")
async def sweep_info(
    dataset_id: str,
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """Whether a sweep can run on this dataset (and if not, why), and its limits."""
    adata = (await _adata(dm, dataset_id)).adata
    try:
        rs.check(adata)
        reason = None
    except rs.SweepError as exc:
        reason = str(exc)
    return {
        "available": reason is None,
        "reason": reason,
        "method": rs.dataset_method(adata),
        "n_cells": int(adata.n_obs),
        "default_resolutions": list(rs.DEFAULT_RESOLUTIONS),
        "max_resolutions": rs.MAX_RESOLUTIONS,
        "min_resolution": rs.MIN_RESOLUTION,
        "max_resolution": rs.MAX_RESOLUTION,
        "max_cells": rs.MAX_CELLS,
        "silhouette_cells": rs.SILHOUETTE_CELLS,
    }


@router.post("/datasets/{dataset_id}/clustering/sweep")
async def run_sweep(
    dataset_id: str,
    body: SweepRequest,
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """Cluster at each resolution and stream the scores, leaving the dataset as it is.

    Events: ``start`` (method, cells, the silhouette's representation and
    subsample), one ``result`` per resolution, then ``complete``, or ``error``.
    """
    adata = (await _adata(dm, dataset_id)).adata
    method = _method(adata, body.method)
    try:
        resolutions = rs.clean_resolutions(body.resolutions)
        rs.check(adata)
    except rs.SweepError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if dataset_id in _running:
        raise HTTPException(status_code=409, detail="A resolution sweep is already running on this dataset.")

    async def events():
        if dataset_id in _running:  # another request started between the check and here
            yield _sse("error", {"error": "A resolution sweep is already running on this dataset."})
            return
        _running.add(dataset_id)
        gen = rs.iter_sweep(adata, resolutions, method)
        try:
            event = "start"
            while True:
                # Each resolution clusters in a worker thread: the server stays responsive.
                item = await asyncio.to_thread(next, gen, _DONE)
                if item is _DONE:
                    break
                yield _sse(event, item)
                event = "result"
            yield _sse("complete", {})
        except Exception as e:  # noqa: BLE001 — surface to the client, keep the server up
            logger.error("Resolution sweep failed for %s: %s", dataset_id, e, exc_info=True)
            yield _sse("error", {"error": str(e)})
        finally:
            _running.discard(dataset_id)
            try:
                gen.close()
            except ValueError:  # client gone mid-resolution: the thread finishes it
                pass

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/datasets/{dataset_id}/clustering/keep")
async def keep_resolution(
    dataset_id: str,
    body: KeepRequest,
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """Save one resolution's clusters as the pipeline's ``scview_<method>_r<res>``
    column (unchanged when it exists), then reload the dataset."""
    async with dataset_lock(dataset_id):
        adaptor = await _adata(dm, dataset_id)
        adata = adaptor.adata
        method = _method(adata, body.method)
        try:
            resolution = rs.clean_resolutions([body.resolution])[0]
            if rs.existing_column(adata, method, resolution) is not None:
                return rs.keep(adata, method, resolution)
            rs.check(adata)
        except rs.SweepError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        output_path = str(dm.derived_h5ad_path(dataset_id, adaptor.h5ad_path))

        def _run():
            work = adata.to_memory() if getattr(adata, "isbacked", False) else adata
            result = rs.keep(work, method, resolution)
            write_derived(work, output_path)
            return result

        try:
            result = await asyncio.to_thread(_run)
        except Exception as exc:
            logger.error("Keeping resolution %s failed for %s: %s", resolution, dataset_id, exc,
                         exc_info=True)
            raise HTTPException(status_code=500, detail=f"Could not save the clustering: {exc!s}")
        finally:
            dm.evict(dataset_id)  # the cached copy may hold the column without it being saved
        await dm.load_dataset(dataset_id)
        return result
