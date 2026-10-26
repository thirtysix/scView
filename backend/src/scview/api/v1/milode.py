"""Neighbourhood DE, miloDE-style (core/milode.py): DE between two conditions
inside each neighbourhood of the cell graph, with the samples as replicates
and no clusters needed.

``POST /milode/plan`` checks a request, builds the neighbourhoods and returns
how many can be tested and a time estimate; ``POST /milode/run`` streams
progress and the result (SSE, Cancel = close the stream); ``GET /milode/gene``
and ``GET /milode/nhood/{i}`` read the last run by gene and by neighbourhood;
``POST /milode/cells`` returns the cells of neighbourhoods.
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

from scview.core import milo, milode
from scview.core.confounding import recorded_batch_key
from scview.core.dataset_manager import DatasetManager
from scview.core.replicates import donor_candidates, pick_donor
from scview.core.sse import keepalive
from scview.dependencies import get_dataset_manager

logger = logging.getLogger(__name__)

router = APIRouter()

_DONE = object()
_running: set[str] = set()  # datasets with a run streaming: one at a time each


class MiloDERequest(BaseModel):
    condition_key: str
    sample_key: str | None = None  # left out: the most donor-like column
    conditions: list[str] | None = None  # [reference, tested]
    covariates: list[str] | None = None
    paired: bool | None = None
    label_key: str | None = None
    k: int = milode.K
    prop: float = milode.PROP
    genes: str = milode.HVG  # "hvg" | "expressed"


class CellsRequest(BaseModel):
    nhoods: list[int]


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def _adaptor(dm: DatasetManager, dataset_id: str):
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    return adaptor


def _plan(adata, body: MiloDERequest) -> tuple[milode.Plan, bool]:
    obs = adata.obs
    for col in (body.condition_key, body.sample_key, body.label_key, *(body.covariates or [])):
        if col and col not in obs.columns:
            raise milode.MiloDEError(f"Column '{col}' not found.")
    key = body.sample_key
    if not key:
        cond = obs[body.condition_key]
        labels = cond.astype(str).where(cond.notna())
        values = body.conditions or [str(v) for v in labels.dropna().unique()]
        masks = [(labels == v).to_numpy() for v in values]
        exclude = {body.condition_key, body.label_key, *(body.covariates or [])} - {None}
        key = pick_donor(obs, donor_candidates(obs, exclude), masks)
    p = milode.plan(adata, body.condition_key, key, conditions=body.conditions,
                    covariates=body.covariates, paired=body.paired, k=body.k, prop=body.prop,
                    genes=body.genes, batch_key=recorded_batch_key(adata))
    return p, not body.sample_key


async def _planned(adata, body: MiloDERequest) -> tuple[milode.Plan, bool]:
    try:
        return await asyncio.to_thread(_plan, adata, body)
    except (milode.MiloDEError, milo.MiloError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/datasets/{dataset_id}/milode/plan")
async def milode_plan(
    dataset_id: str,
    body: MiloDERequest,
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """The design, the neighbourhoods (how many, cells and samples in each, how
    many can be tested) and a time estimate. The neighbourhoods are kept for
    the run."""
    adata = (await _adaptor(dm, dataset_id)).adata
    p, detected = await _planned(adata, body)
    return {**p.to_dict(), "sample_key_detected": detected, "n_cells_total": int(adata.n_obs)}


@router.post("/datasets/{dataset_id}/milode/run")
async def milode_run(
    dataset_id: str,
    body: MiloDERequest,
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """Run and stream: ``start`` (the plan), ``progress`` (steps; the test in
    chunks with ``done`` / ``total``), ``result``, then ``complete``, or
    ``error``. Closing the stream stops the run at its next chunk. The dataset
    is not changed."""
    adaptor = await _adaptor(dm, dataset_id)
    adata = adaptor.adata
    p, detected = await _planned(adata, body)
    if dataset_id in _running:
        raise HTTPException(status_code=409,
                            detail="Neighbourhood DE is already running on this dataset.")

    async def events():
        if dataset_id in _running:
            yield _sse("error", {"error": "Neighbourhood DE is already running on this dataset."})
            return
        _running.add(dataset_id)
        stop = threading.Event()
        gen = milode.iter_milode(adata, p, label_key=body.label_key, stop=stop.is_set)
        pending: asyncio.Future | None = None

        def finish(fut: asyncio.Future | None = None) -> None:
            if fut is not None and not fut.cancelled():
                fut.exception()  # retrieved: a cancelled run is not an error
            gen.close()
            _running.discard(dataset_id)

        try:
            yield _sse("start", {**p.to_dict(), "sample_key_detected": detected})
            while True:
                # Each step runs in a worker thread; shielded, so a client going
                # away cancels this wait, not the thread's future.
                pending = asyncio.ensure_future(asyncio.to_thread(next, gen, _DONE))
                item = await asyncio.shield(pending)
                pending = None
                if item is _DONE:
                    break
                if "result" in item:
                    result: milode.MiloDEResult = item["result"]
                    milode.remember(adaptor, result)
                    payload = await asyncio.to_thread(
                        lambda: result.to_dict(milo.layouts(adata, result.index)))
                    yield _sse("result", {**payload, "sample_key_detected": detected})
                else:
                    yield _sse("progress", item)
            yield _sse("complete", {})
        except (milode.MiloDEError, milo.MiloError) as e:
            yield _sse("error", {"error": str(e)})
        except Exception as e:  # noqa: BLE001 — surface to the client, keep the server up
            logger.error("Neighbourhood DE failed for %s: %s", dataset_id, e, exc_info=True)
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


async def _last(dm: DatasetManager, dataset_id: str) -> milode.MiloDEResult:
    adaptor = await _adaptor(dm, dataset_id)
    result = await asyncio.to_thread(milode.last_result, adaptor)
    if result is None:
        raise HTTPException(status_code=404,
                            detail="No neighbourhood DE result for this dataset: run it again.")
    return result


@router.get("/datasets/{dataset_id}/milode/gene")
async def milode_gene(
    dataset_id: str,
    gene: str,
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """One gene in every neighbourhood of the last run: log2FC, p, spatial FDR."""
    result = await _last(dm, dataset_id)
    try:
        return result.gene_detail(gene)
    except milode.MiloDEError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/datasets/{dataset_id}/milode/nhood/{nhood}")
async def milode_nhood(
    dataset_id: str,
    nhood: int,
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """One neighbourhood's genes in the last run, smallest p first."""
    result = await _last(dm, dataset_id)
    try:
        return await asyncio.to_thread(result.nhood_detail, nhood)
    except milode.MiloDEError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/datasets/{dataset_id}/milode/cells")
async def milode_cells(
    dataset_id: str,
    body: CellsRequest,
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """The cells (row indices) in neighbourhoods of the last run."""
    result = await _last(dm, dataset_id)
    cells = milode.nhood_cells(result, body.nhoods)
    return {"cells": cells.astype(int).tolist(), "n_cells": int(cells.size)}
