"""Layout stability (``core/layout_stability.py``): which structure in a 2D layout
survives another seed or neighbour count.

``GET /layouts`` says whether the check can run, with what defaults and time
estimate, and gives the latest run's summary (for ``group_by`` / ``embedding``)
when one is held. ``POST /layouts/run-stream`` lays the cells out under each seed
x n_neighbors without changing the dataset and streams ``start``, one ``layout``
per layout, ``summary`` and ``complete`` (or ``error``). ``GET /layouts/{i}``
gives one layout's coordinates for a thumbnail (``current``: the user's plot).
``POST /layouts/{i}/keep`` saves layout ``i`` as obsm ``X_umap_s<seed>_k<k>``.
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

from scview.core import layout_stability as ls
from scview.core.dataset_manager import DatasetManager
from scview.core.obs_writes import dataset_edit, uns_snapshot, write_derived
from scview.core.sse import keepalive
from scview.dependencies import get_dataset_manager

logger = logging.getLogger(__name__)

router = APIRouter()

_DONE = object()
_running: set[str] = set()  # datasets with a run streaming: one at a time each

NOT_HELD = ("No layouts are held for this dataset: run the check again (they are kept in memory "
            "only until the dataset changes or the server restarts).")


class LayoutRunRequest(BaseModel):
    group_by: str | None = None  # default: the active clustering
    embedding: str | None = None  # the plot to compare with; default X_umap
    seeds: list[int] | None = None  # default ls.SEEDS
    n_neighbors: list[int] | None = None  # default {10, the dataset's k, 30}


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def _adaptor(dm: DatasetManager, dataset_id: str):
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    return adaptor


def _limits() -> dict[str, Any]:
    return {"max_cells": ls.MAX_CELLS, "max_layouts": ls.MAX_LAYOUTS, "max_seeds": ls.MAX_SEEDS,
            "neighbor_range": [ls.MIN_NEIGHBORS, ls.MAX_NEIGHBORS], "max_groups": ls.MAX_GROUPS,
            "first_run_seconds": ls.FIRST_RUN_SECONDS, "rule": ls.TOUCH_RULE}


@router.get("/datasets/{dataset_id}/layouts")
async def layouts_info(
    dataset_id: str,
    group_by: str | None = Query(None, description="Grouping for the relations; default the clustering"),
    embedding: str | None = Query(None, description="The plot to compare with; default X_umap"),
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """Whether the check can run (and if not, why), the default layouts and their
    time estimate, and the latest run's summary when one is held."""
    adata = (await _adaptor(dm, dataset_id)).adata

    def _info() -> dict[str, Any]:
        base = {"n_cells": int(adata.n_obs), **_limits(), "running": dataset_id in _running}
        try:
            plan = ls.make_plan(adata, embedding=embedding, group_by=group_by)
        except ls.LayoutError as exc:
            return {"available": False, "reason": str(exc), **base, "result": None}
        run = ls.latest(adata)
        return {
            "available": True, "reason": None, **base,
            "plan": plan.to_dict(),
            "estimate_seconds": ls.estimate_seconds(plan.n_cells, plan.ks, plan.seeds),
            # A run summarised for this grouping and plot (a few hundred ms: the plot's
            # own neighbours and placement the first time).
            "result": ls.summarise(adata, run, group_by=plan.group_by, embedding=embedding)
            if run is not None else None,
        }

    try:
        return await asyncio.to_thread(_info)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc).strip("'\""))


@router.post("/datasets/{dataset_id}/layouts/run-stream")
async def run_layouts(
    dataset_id: str,
    body: LayoutRunRequest,
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """Lay the cells out under each seed x n_neighbors and stream the comparison,
    leaving the dataset as it is. A client that goes away stops the run before its
    next layout; what it laid out stays held."""
    adata = (await _adaptor(dm, dataset_id)).adata

    def _plan():
        return ls.make_plan(adata, embedding=body.embedding, group_by=body.group_by,
                            seeds=body.seeds, ks=body.n_neighbors)

    try:
        plan = await asyncio.to_thread(_plan)
    except ls.LayoutError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc).strip("'\""))
    if dataset_id in _running:
        raise HTTPException(status_code=409, detail="A layout check is already running on this dataset.")

    async def events():
        if dataset_id in _running:  # another request started between the check and here
            yield _sse("error", {"error": "A layout check is already running on this dataset."})
            return
        _running.add(dataset_id)
        stop = threading.Event()
        gen = ls.iter_run(adata, plan, stop=stop.is_set)
        pending: asyncio.Future | None = None

        def finish(fut: asyncio.Future | None = None) -> None:
            if fut is not None and not fut.cancelled():
                fut.exception()  # retrieved: a stopped run is not an error
            gen.close()
            _running.discard(dataset_id)

        try:
            while True:
                # Each layout (a graph and a UMAP, seconds each) runs in a worker thread.
                # Shielded, so a client going away cancels this wait, not the thread's
                # future, whose callback then releases the dataset.
                pending = asyncio.ensure_future(asyncio.to_thread(next, gen, _DONE))
                item = await asyncio.shield(pending)
                pending = None
                if item is _DONE:
                    break
                event, data = item
                yield _sse(event, data)
            yield _sse("complete", {})
        except Exception as e:  # noqa: BLE001 — surface to the client, keep the server up
            logger.error("Layout check failed for %s: %s", dataset_id, e, exc_info=True)
            yield _sse("error", {"error": str(e)})
        finally:
            stop.set()  # a thread mid-layout stops before the next one
            if pending is not None and not pending.done():
                pending.add_done_callback(finish)
            else:
                finish()

    return StreamingResponse(
        keepalive(events()),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/datasets/{dataset_id}/layouts/{index}")
async def layout_coordinates(
    dataset_id: str,
    index: str,
    max_points: int = Query(ls.THUMB_CELLS, ge=100, le=ls.MAX_THUMB_CELLS),
    embedding: str | None = Query(None, description="For 'current': the plot; default the run's"),
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """Layout ``index`` (or ``current``, the user's plot) on the run's thumbnail
    cells, the same cells in every layout: obs indices and x, y."""
    if index != "current" and not index.isdigit():
        raise HTTPException(status_code=422, detail=f"Layout '{index}' is not a number or 'current'.")
    adata = (await _adaptor(dm, dataset_id)).adata
    run = ls.latest(adata)
    if run is None:
        raise HTTPException(status_code=404, detail=NOT_HELD)
    try:
        # The user's plot is scored once on first use (KD-tree, placement): off the loop.
        return await asyncio.to_thread(ls.thumbnail, adata, run, index, max_points, embedding)
    except ls.LayoutError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc).strip("'\""))


@router.post("/datasets/{dataset_id}/layouts/{index}/keep")
async def keep_layout(
    dataset_id: str,
    index: int,
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """Save layout ``index`` of the held run as obsm ``X_umap_s<seed>_k<k>``
    (unchanged when it exists), so it appears in the embedding chooser."""
    async with dataset_edit(dataset_id) as held:
        adaptor = await _adaptor(dm, dataset_id)
        adata = adaptor.adata
        run = ls.latest(adata)
        if run is None:
            raise HTTPException(status_code=404, detail=NOT_HELD)
        output_path = str(dm.derived_h5ad_path(dataset_id, adaptor.h5ad_path))

        def _run() -> dict[str, Any]:
            before = uns_snapshot(adata)
            result = ls.keep(adata, run, index)
            if result["created"]:
                write_derived(adata, output_path, adaptor.h5ad_path, before, obsm_keys=[result["key"]])
            return result

        try:
            # A layout of a sample is laid out again on all cells (seconds to a minute).
            result = await held.run(_run)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc).strip("'\""))
        except Exception as exc:
            dm.evict(dataset_id)  # the cached copy may hold the layout without it being saved
            logger.error("Keeping layout %s failed for %s: %s", index, dataset_id, exc, exc_info=True)
            raise HTTPException(status_code=500, detail=f"Could not save the layout: {exc!s}")
        if result["created"]:
            dm.saved_in_place(dataset_id, adaptor, output_path)
        return result
