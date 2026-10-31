"""Compare batch-correction methods on the current PCA (``core/batch_compare.py``).

``GET /datasets/{id}/batch/compare`` says whether the comparison can run for a batch
column (and if not, why), with the label column conservation would use, the time
estimate and the background text. ``POST /datasets/{id}/batch/compare-stream`` runs
none / Harmony / Scanorama / BBKNN without changing the dataset and streams
``start``, ``method_start`` and ``method`` per method, ``result`` (rows and the
recommendation) and ``complete`` (or ``stopped`` / ``error``). A client that goes
away stops the comparison before its next method.
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

from scview.core import batch_compare as bc
from scview.core.dataset_manager import DatasetManager
from scview.core.large_mode import LargeDatasetError
from scview.core.sse import keepalive
from scview.dependencies import get_dataset_manager

logger = logging.getLogger(__name__)

router = APIRouter()

_DONE = object()
_running: set[str] = set()  # one comparison at a time per dataset

LARGE = LargeDatasetError(
    "Comparing batch-correction methods",
    "Scanorama and BBKNN do not run in large-dataset mode (Harmony does: run it as a step)")


class CompareRequest(BaseModel):
    batch_key: str
    label_column: str | None = None  # default: a cell-type-like column, if any
    methods: list[str] | None = None  # default: all of bc.METHODS


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def _adata(dm: DatasetManager, dataset_id: str):
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    if adaptor.is_large:
        raise HTTPException(status_code=422, detail=str(LARGE))
    return adaptor.adata


@router.get("/datasets/{dataset_id}/batch/compare")
async def compare_info(
    dataset_id: str,
    batch_key: str = Query(..., description="The obs column holding each cell's batch"),
    label_column: str | None = Query(None, description="Label for conservation; default auto"),
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """Whether the comparison can run, which label column it would score, and how long."""
    adata = await _adata(dm, dataset_id)
    base = {"methods": list(bc.METHODS), "background": bc.BACKGROUND,
            "background_tip": bc.BACKGROUND_TIP, "n_cells": int(adata.n_obs),
            "running": dataset_id in _running}
    try:
        label, kind = bc.check(adata, batch_key, label_column)
    except bc.CompareError as e:
        return {"available": False, "reason": str(e), **base}
    return {"available": True, "reason": None, **base, "label_column": label,
            "label_kind": kind or None,
            "estimate_seconds": round(bc.estimate_seconds(int(adata.n_obs)), 1),
            "first_run": not bc.bbknn_warm()}  # the estimate counts BBKNN's compilation


@router.post("/datasets/{dataset_id}/batch/compare-stream")
async def compare_stream(
    dataset_id: str,
    body: CompareRequest,
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """Run each method on the current PCA and stream the scores; nothing is saved."""
    adata = await _adata(dm, dataset_id)
    unknown = [m for m in (body.methods or []) if m not in bc.METHODS]
    if unknown:
        raise HTTPException(status_code=422, detail=f"Unknown method(s) {', '.join(unknown)}; "
                                                    f"choose from {', '.join(bc.METHODS)}.")
    try:
        bc.check(adata, body.batch_key, body.label_column)
    except bc.CompareError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if dataset_id in _running:
        raise HTTPException(status_code=409, detail="A comparison is already running on this dataset.")

    async def events():
        if dataset_id in _running:
            yield _sse("error", {"error": "A comparison is already running on this dataset."})
            return
        _running.add(dataset_id)
        stop = threading.Event()
        gen = bc.iter_compare(adata, body.batch_key, label_column=body.label_column,
                              methods=body.methods, stop=stop.is_set)
        pending: asyncio.Future | None = None

        def finish(fut: asyncio.Future | None = None) -> None:
            if fut is not None and not fut.cancelled():
                fut.exception()  # retrieved: a stopped comparison is not an error
            gen.close()
            _running.discard(dataset_id)

        try:
            while True:
                # Each method runs in a worker thread; shielded, so a client going away
                # cancels this wait, not the thread's future (whose callback cleans up).
                pending = asyncio.ensure_future(asyncio.to_thread(next, gen, _DONE))
                item = await asyncio.shield(pending)
                pending = None
                if item is _DONE:
                    break
                yield _sse(*item)
            yield _sse("complete", {})
        except Exception as e:  # noqa: BLE001 — surface to the client, keep the server up
            logger.error("Batch method comparison failed for %s: %s", dataset_id, e, exc_info=True)
            yield _sse("error", {"error": str(e)})
        finally:
            stop.set()  # a thread mid-method stops before the next one
            if pending is not None and not pending.done():
                pending.add_done_callback(finish)
            else:
                finish()

    return StreamingResponse(
        keepalive(events()),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
