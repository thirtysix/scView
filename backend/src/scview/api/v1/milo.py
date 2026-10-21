"""Milo neighbourhood differential abundance (core/milo.py): which regions of
the cell graph change in abundance between two conditions, with the samples
as replicates and no clusters needed.

``POST /milo/plan`` checks a request and returns its design and a time
estimate; ``POST /milo/run`` streams progress and the result (SSE);
``POST /milo/cells`` returns the cells of neighbourhoods of the last run.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from scview.core import milo
from scview.core.confounding import recorded_batch_key
from scview.core.dataset_manager import DatasetManager
from scview.core.replicates import donor_candidates, pick_donor
from scview.core.sse import keepalive
from scview.dependencies import get_dataset_manager

logger = logging.getLogger(__name__)

router = APIRouter()

_DONE = object()
_running: set[str] = set()  # datasets with a Milo run streaming: one at a time each


class MiloRequest(BaseModel):
    condition_key: str
    # Donor / sample column; left out, the most donor-like one with at least two
    # values in each condition is used (core/replicates.py).
    sample_key: str | None = None
    conditions: list[str] | None = None  # [reference, tested]; default: the first two
    covariates: list[str] | None = None  # per-sample columns (batch, sex) in the design
    paired: bool | None = None  # None: paired when every donor is in both conditions
    label_key: str | None = None  # cell types / clusters to label neighbourhoods by
    prop: float | None = None  # share of cells sampled as index cells


class CellsRequest(BaseModel):
    nhoods: list[int]


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def _adaptor(dm: DatasetManager, dataset_id: str):
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    return adaptor


def _plan(adata, body: MiloRequest) -> tuple[milo.Plan, bool]:
    obs = adata.obs
    for col in (body.condition_key, body.sample_key, body.label_key, *(body.covariates or [])):
        if col and col not in obs.columns:
            raise milo.MiloError(f"Column '{col}' not found.")
    key = body.sample_key
    if not key:
        cond = obs[body.condition_key]
        labels = cond.astype(str).where(cond.notna())
        values = body.conditions or [str(v) for v in labels.dropna().unique()]
        masks = [(labels == v).to_numpy() for v in values]
        exclude = {body.condition_key, body.label_key, *(body.covariates or [])} - {None}
        key = pick_donor(obs, donor_candidates(obs, exclude), masks)
    p = milo.plan(adata, body.condition_key, key, conditions=body.conditions,
                  covariates=body.covariates, paired=body.paired, prop=body.prop,
                  batch_key=recorded_batch_key(adata))
    return p, not body.sample_key


@router.post("/datasets/{dataset_id}/milo/plan")
async def milo_plan(
    dataset_id: str,
    body: MiloRequest,
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """The design a run would use (samples per condition, pairing, formula),
    how many neighbourhoods it would sample and about how long it would take,
    with the design's warnings. Nothing is computed on the graph."""
    adata = (await _adaptor(dm, dataset_id)).adata
    try:
        p, detected = await asyncio.to_thread(_plan, adata, body)
    except milo.MiloError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {**p.to_dict(), "sample_key_detected": detected, "n_cells_total": int(adata.n_obs)}


@router.post("/datasets/{dataset_id}/milo/run")
async def milo_run(
    dataset_id: str,
    body: MiloRequest,
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """Run Milo and stream it. Events: ``start`` (the plan), ``progress`` (one
    per step), ``result`` (the full result), then ``complete``, or ``error``.
    The dataset is not changed."""
    adaptor = await _adaptor(dm, dataset_id)
    adata = adaptor.adata
    try:
        p, detected = await asyncio.to_thread(_plan, adata, body)
    except milo.MiloError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if dataset_id in _running:
        raise HTTPException(status_code=409, detail="Milo is already running on this dataset.")

    async def events():
        if dataset_id in _running:  # another request started between the check and here
            yield _sse("error", {"error": "Milo is already running on this dataset."})
            return
        _running.add(dataset_id)
        stop = threading.Event()
        gen = milo.iter_milo(adata, p, label_key=body.label_key, stop=stop.is_set)
        pending: asyncio.Future | None = None

        def finish(fut: asyncio.Future | None = None) -> None:
            if fut is not None and not fut.cancelled():
                fut.exception()  # retrieved: a cancelled run is not an error
            gen.close()
            _running.discard(dataset_id)

        try:
            yield _sse("start", {**p.to_dict(), "sample_key_detected": detected})
            while True:
                # Every step runs in a worker thread: the server stays responsive.
                # Shielded, so a client going away cancels this wait, not the
                # thread's future, whose callback then releases the dataset.
                pending = asyncio.ensure_future(asyncio.to_thread(next, gen, _DONE))
                item = await asyncio.shield(pending)
                pending = None
                if item is _DONE:
                    break
                if "result" in item:
                    result: milo.MiloResult = item["result"]
                    milo.remember(adaptor, result)
                    payload = await asyncio.to_thread(
                        lambda: result.to_dict(milo.layouts(adata, result.index)))
                    yield _sse("result", {**payload, "sample_key_detected": detected})
                else:
                    yield _sse("progress", item)
            yield _sse("complete", {})
        except milo.MiloError as e:
            yield _sse("error", {"error": str(e)})
        except Exception as e:  # noqa: BLE001 — surface to the client, keep the server up
            logger.error("Milo failed for %s: %s", dataset_id, e, exc_info=True)
            yield _sse("error", {"error": str(e)})
        finally:
            stop.set()  # a thread mid-step stops at its next step
            if pending is not None and not pending.done():
                pending.add_done_callback(finish)
            else:
                finish()

    return StreamingResponse(
        keepalive(events()),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/datasets/{dataset_id}/milo/cells")
async def milo_cells(
    dataset_id: str,
    body: CellsRequest,
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """The cells (row indices) in the given neighbourhoods of the last run, to
    select them on the scatter."""
    adaptor = await _adaptor(dm, dataset_id)
    result = milo.last_result(adaptor)
    if result is None:
        raise HTTPException(status_code=404, detail="No Milo result for this dataset: run it again.")
    cells = milo.nhood_cells(result, body.nhoods)
    return {"cells": cells.astype(int).tolist(), "n_cells": int(cells.size)}
