"""Count-split markers: marker p-values on held-out counts (``core/split_markers.py``).

``GET /markers/count-split`` says whether the check can run on a column (and if
not, why) with its time estimate; ``POST /markers/count-split`` streams the run
(SSE: ``start``, ``progress``, ``result``, ``complete`` or ``error``) without
changing the dataset; ``POST /markers/count-split/keep`` stores one group's
split test with the DE results (GSEA by result id, the overlap view).
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from scview.core import merge_check as mc
from scview.core import split_markers as sm
from scview.core.dataset_manager import DatasetManager
from scview.core.sse import keepalive
from scview.dependencies import get_dataset_manager

logger = logging.getLogger(__name__)

router = APIRouter()

_DONE = object()
_running: set[str] = set()  # datasets with a check running: one at a time each


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


class SplitMarkersRequest(BaseModel):
    column: str | None = None  # default: the active clustering


class SplitKeepRequest(SplitMarkersRequest):
    group: str


async def _adaptor(dm: DatasetManager, dataset_id: str):
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    return adaptor


@router.get("/datasets/{dataset_id}/markers/count-split")
async def split_markers_info(
    dataset_id: str,
    column: str | None = Query(None),
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """Whether the count-split check can run on ``column`` (and if not, why), and
    its time estimate."""
    adata = (await _adaptor(dm, dataset_id)).adata

    def _info() -> dict[str, Any]:
        base = {"n_cells": int(adata.n_obs), "first_run_seconds": mc.FIRST_RUN_SECONDS}
        try:
            p = sm.plan(adata, column)
        except sm.SplitMarkersError as exc:
            return {"available": False, "reason": str(exc), **base}
        return {"available": True, "reason": None, **base, "reference": p.reference,
                "n_groups": len(p.names), "recorded": p.rec.to_dict(),
                "cached": sm.cached(adata, p) is not None,
                "estimate_seconds": sm.estimate_seconds(adata.n_obs, p.rec)}

    return await asyncio.to_thread(_info)


@router.post("/datasets/{dataset_id}/markers/count-split")
async def run_split_markers(
    dataset_id: str,
    body: SplitMarkersRequest,
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """Split the counts, re-cluster the training half and test each cluster
    against the rest on the held-out half, streaming progress. 422 when the
    check cannot run here (no raw counts, a large dataset). A client that goes
    away stops the run before its next step."""
    adaptor = await _adaptor(dm, dataset_id)
    adata = adaptor.adata
    try:
        p = await asyncio.to_thread(sm.plan, adata, body.column)
    except sm.SplitMarkersError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    if dataset_id in _running:
        raise HTTPException(status_code=409, detail="A count-split check is already running on this dataset.")

    async def events():
        if dataset_id in _running:  # another request started between the check and here
            yield _sse("error", {"error": "A count-split check is already running on this dataset."})
            return
        _running.add(dataset_id)
        stop = threading.Event()
        gen = sm.iter_split_markers(adata, p, stop=stop.is_set)
        pending: asyncio.Future | None = None

        def finish(fut: asyncio.Future | None = None) -> None:
            if fut is not None and not fut.cancelled():
                fut.exception()  # retrieved: a cancelled run is not an error
            gen.close()
            _running.discard(dataset_id)

        try:
            first = True
            while True:
                # Each step in a worker thread, shielded so a cancelled request
                # waits for the step to end before the generator is closed.
                pending = asyncio.ensure_future(asyncio.to_thread(next, gen, _DONE))
                item = await asyncio.shield(pending)
                pending = None
                if item is _DONE:
                    break
                if first:
                    yield _sse("start", item)
                    first = False
                elif "step" in item:
                    yield _sse("progress", item)
                else:
                    sm.remember(adaptor, item)
                    yield _sse("result", item)
            yield _sse("complete", {})
        except Exception as e:  # noqa: BLE001 — surface to the client, keep the server up
            logger.error("Count-split markers failed for %s: %s", dataset_id, e, exc_info=True)
            yield _sse("error", {"error": str(e)})
        finally:
            stop.set()
            if pending is not None and not pending.done():
                pending.add_done_callback(finish)
            else:
                finish()

    return StreamingResponse(
        keepalive(events()),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/datasets/{dataset_id}/markers/count-split/keep")
async def keep_split_result(
    dataset_id: str,
    body: SplitKeepRequest,
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """Store one group's split test with the DE results; 404 when the run is no
    longer kept (run the check again)."""
    adata = (await _adaptor(dm, dataset_id)).adata

    def _keep() -> dict[str, Any]:
        p = sm.plan(adata, body.column)
        stage = sm.cached(adata, p)
        if stage is None:
            raise LookupError
        rid = sm.keep_group(dataset_id, stage, body.group)
        from scview.core import de_results

        stored = de_results.recall(dataset_id, rid)
        return {"result_id": rid, "label": stored.label if stored else ""}

    try:
        return await asyncio.to_thread(_keep)
    except sm.SplitMarkersError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except LookupError:
        raise HTTPException(status_code=404, detail="That count-split run is no longer kept: run the check again.")
