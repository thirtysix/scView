"""Minimal marker panel (``core/marker_panel.py``): the fewest genes that still tell
the groups of a column apart.

``POST /markers/panel`` plans the run and reads the candidate genes of the sampled
cells (a 400 / 404 / 422 before any stream starts), then streams ``start``, one
``step`` per gene added, ``baseline``, ``result`` and ``complete`` (or ``error``).
``GET /markers/panel`` gives the latest result held for the dataset (memory only:
gone when the dataset is reloaded).
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from scview.core import marker_panel as mp
from scview.core.dataset_manager import DatasetManager
from scview.core.sse import keepalive
from scview.dependencies import get_dataset_manager

logger = logging.getLogger(__name__)

router = APIRouter()

_DONE = object()
_running: set[str] = set()  # one panel run per dataset at a time


class PanelRequest(BaseModel):
    groupby: str
    size: int = Field(mp.SIZE_DEFAULT, ge=1, le=mp.SIZE_MAX)
    per_group: int = Field(mp.PER_GROUP_DEFAULT, ge=1, le=mp.PER_GROUP_MAX)
    required: list[str] = Field(default_factory=list, max_length=mp.SIZE_MAX)
    excluded: list[str] = Field(default_factory=list, max_length=5000)
    restrict: list[str] | None = Field(None, max_length=20000)  # e.g. a saved gene list


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def _adaptor(dm: DatasetManager, dataset_id: str):
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    return adaptor


@router.get("/datasets/{dataset_id}/markers/panel")
async def latest_panel(dataset_id: str, dm: DatasetManager = Depends(get_dataset_manager)):
    """The latest panel made for this dataset since it was loaded, or ``result: null``."""
    adaptor = await _adaptor(dm, dataset_id)
    return {"result": mp.latest(adaptor.adata), "running": dataset_id in _running}


@router.post("/datasets/{dataset_id}/markers/panel")
async def run_panel(dataset_id: str, body: PanelRequest,
                    dm: DatasetManager = Depends(get_dataset_manager)):
    """Choose the panel by greedy forward selection and stream the path. A client that
    goes away stops the run before its next gene."""
    adaptor = await _adaptor(dm, dataset_id)
    if dataset_id in _running:
        raise HTTPException(status_code=409, detail="A marker panel is already being made for this dataset.")

    def _prepare():
        plan = mp.make_plan(adaptor, body.groupby, size=body.size, per_group=body.per_group,
                            required=body.required, excluded=body.excluded, restrict=body.restrict)
        return plan, mp.read_matrix(adaptor, plan.candidates, plan.cells)

    try:
        # Reads the candidate genes of the sampled cells (a gene-store read when large).
        plan, matrix = await asyncio.to_thread(_prepare)
    except mp.PanelError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc).strip("'\""))

    async def events():
        if dataset_id in _running:
            yield _sse("error", {"error": "A marker panel is already being made for this dataset."})
            return
        _running.add(dataset_id)
        stop = threading.Event()
        gen = mp.iter_run(adaptor, plan, stop=stop.is_set, matrix=matrix)
        pending: asyncio.Future | None = None

        def finish(fut: asyncio.Future | None = None) -> None:
            if fut is not None and not fut.cancelled():
                fut.exception()  # retrieved: a stopped run is not an error
            gen.close()
            _running.discard(dataset_id)

        try:
            while True:
                # Each step (a few dozen model fits, ~2 s on Kang) runs in a worker thread;
                # shielded so a client going away cancels this wait, not the thread.
                pending = asyncio.ensure_future(asyncio.to_thread(next, gen, _DONE))
                item = await asyncio.shield(pending)
                pending = None
                if item is _DONE:
                    break
                event, data = item
                yield _sse(event, data)
            yield _sse("complete", {})
        except Exception as e:  # noqa: BLE001 — surface to the client, keep the server up
            logger.error("Marker panel failed for %s: %s", dataset_id, e, exc_info=True)
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
