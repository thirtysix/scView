"""Study planner (core/study_planner.py): how many donors per group would a
follow-up need, from the open dataset's own donor-to-donor variability.

``POST /planner/pilot`` says what a plan would use (obs only, fast);
``POST /planner/composition-stream`` and ``/planner/de-stream`` simulate the
grid and stream progress per grid point and the result (SSE). The dataset is
not changed.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from scview.core import study_planner as planner
from scview.core.dataset_manager import DatasetManager
from scview.core.replicates import donor_candidates, pick_donor
from scview.core.sse import keepalive
from scview.dependencies import get_dataset_manager

logger = logging.getLogger(__name__)

router = APIRouter()

_DONE = object()
_running: set[str] = set()  # datasets with a plan streaming: one at a time each


class _Pilot(BaseModel):
    group_key: str  # cell types (or clusters)
    condition_key: str | None = None  # left out: all cells as one condition
    # Donor / sample column; left out, the most donor-like one with at least two
    # values in every condition is used (core/replicates.py).
    sample_key: str | None = None
    reference: str | None = None
    conditions: list[str] | None = Field(default=None, max_length=2)


class PilotRequest(_Pilot):
    pass


class CompositionPlanRequest(_Pilot):
    groups: list[str] | None = Field(default=None, max_length=planner.MAX_TARGETS)
    fold_changes: list[float] = Field(default=list(planner.DEFAULT_FOLD_CHANGES), max_length=12)
    donors: list[int] = Field(default=list(planner.DEFAULT_DONORS), max_length=12)
    cells: list[int] = Field(default=list(planner.DEFAULT_CELLS), max_length=12)
    alpha: float = planner.ALPHA
    target_power: float = planner.TARGET_POWER
    design: Literal["paired", "unpaired"] | None = None  # None: the pilot's
    reps: int | None = None  # None: from the grid size (time budget)
    # A donor-to-donor SD of the log-odds share to plan with instead of the
    # pilot's (required when the pilot has no replicates).
    assumed_sd: float | None = None
    transform: Literal["logit", "asin"] = "logit"
    robust: bool = True
    seed: int = 0


class DEPlanRequest(_Pilot):
    group: str  # the cell type whose pseudobulk DE is planned
    log2fc: list[float] = Field(default=list(planner.DE_DEFAULT_LOG2FC), max_length=12)
    donors: list[int] = Field(default=list(planner.DEFAULT_DONORS), max_length=12)
    cells: list[int] = Field(default=list(planner.DE_DEFAULT_CELLS), max_length=12)
    de_share: float = planner.DE_SHARE
    alpha: float = planner.ALPHA
    target_power: float = planner.TARGET_POWER
    design: Literal["paired", "unpaired"] | None = None
    reps: int | None = None
    n_genes: int = planner.DE_GENES
    seed: int = 0


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def _adaptor(dm: DatasetManager, dataset_id: str):
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    return adaptor


def _sample_key(obs, body: _Pilot, extra: set[str]) -> tuple[str | None, bool]:
    """(the donor column, detected?): the request's, else the most donor-like one
    with two or more values in each compared condition."""
    if body.sample_key:
        return body.sample_key, False
    for col in (body.group_key, body.condition_key):
        if col and col not in obs.columns:
            raise planner.PlannerError(f"Column '{col}' not found.")
    if body.condition_key:
        cond = obs[body.condition_key]
        labels = cond.astype(str).where(cond.notna())
        values = body.conditions or [str(v) for v in labels.dropna().unique()][:2]
        masks = [(labels == v).to_numpy() for v in values]
    else:
        masks = [obs.index.notna()]
    exclude = {body.group_key, body.condition_key, *extra} - {None}
    key = pick_donor(obs, donor_candidates(obs, exclude), masks)
    return key, key is not None


@router.post("/datasets/{dataset_id}/planner/pilot")
async def planner_pilot(
    dataset_id: str,
    body: PilotRequest,
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """The pilot a plan would use: its design, samples per condition, cells per
    sample, the types' shares (with three suggested: abundant, mid, rare) and,
    without replicates, the plain message why a plan needs them."""
    adata = (await _adaptor(dm, dataset_id)).adata

    def _run() -> dict:
        key, detected = _sample_key(adata.obs, body, set())
        out = planner.check_pilot(adata, body.group_key, body.condition_key, key,
                                  reference=body.reference, conditions=body.conditions)
        return {**out, "sample_key_detected": detected}

    try:
        return await asyncio.to_thread(_run)
    except planner.PlannerError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


def _stream(dataset_id: str, adaptor, iterate, start_extra: dict):
    """The SSE response for a prepared plan: ``start``, ``progress`` per grid
    point, ``result``, ``complete`` (or ``error``); every step in a worker thread."""
    if dataset_id in _running:
        raise HTTPException(status_code=409, detail="A study plan is already running on this dataset.")

    async def events():
        if dataset_id in _running:  # another request started between the check and here
            yield _sse("error", {"error": "A study plan is already running on this dataset."})
            return
        _running.add(dataset_id)
        stop = threading.Event()
        gen = iterate(stop.is_set)
        pending: asyncio.Future | None = None

        def finish(fut: asyncio.Future | None = None) -> None:
            if fut is not None and not fut.cancelled():
                fut.exception()  # retrieved: a cancelled run is not an error
            gen.close()
            _running.discard(dataset_id)

        try:
            while True:
                # Shielded, so a client going away cancels this wait, not the
                # thread's future, whose callback then releases the dataset.
                pending = asyncio.ensure_future(asyncio.to_thread(next, gen, _DONE))
                item = await asyncio.shield(pending)
                pending = None
                if item is _DONE:
                    break
                if "start" in item:
                    yield _sse("start", {**item["start"], **start_extra})
                elif "progress" in item:
                    yield _sse("progress", item["progress"])
                elif "result" in item:
                    payload = {**item["result"], **start_extra}
                    planner.remember(adaptor, payload)
                    yield _sse("result", payload)
            yield _sse("complete", {})
        except planner.PlannerError as e:
            yield _sse("error", {"error": str(e)})
        except Exception as e:  # noqa: BLE001 — surface to the client, keep the server up
            logger.error("Study plan failed for %s: %s", dataset_id, e, exc_info=True)
            yield _sse("error", {"error": str(e)})
        finally:
            stop.set()  # a thread mid-point stops at the next grid point
            if pending is not None and not pending.done():
                pending.add_done_callback(finish)
            else:
                finish()

    return StreamingResponse(
        keepalive(events()),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/datasets/{dataset_id}/planner/composition-stream")
async def planner_composition(
    dataset_id: str,
    body: CompositionPlanRequest,
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """Power to detect a fold change in each chosen cell type's share with the
    app's composition test, over donors per group x cells per donor x fold
    change, simulated from the pilot's spreads (or an entered one)."""
    adaptor = await _adaptor(dm, dataset_id)
    obs = adaptor.adata.obs
    t0 = time.perf_counter()

    def _prepare():
        key, detected = _sample_key(obs, body, set())
        req = planner.CompositionRequestModel(**{**body.model_dump(), "sample_key": key})
        return planner.prepare_composition(obs, req), detected

    try:
        (model, grid), detected = await asyncio.to_thread(_prepare)
    except planner.PlannerError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return _stream(
        dataset_id, adaptor,
        lambda stop: planner.iter_composition_plan(model, grid, body.target_power, stop, t0),
        {"sample_key_detected": detected})


@router.post("/datasets/{dataset_id}/planner/de-stream")
async def planner_de(
    dataset_id: str,
    body: DEPlanRequest,
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """Power of pseudobulk DE in one cell type (share of changed genes found at
    FDR < alpha) over donors per group x cells of that type per donor x log2
    fold change, simulated from the pilot's per-gene means and dispersions."""
    adaptor = await _adaptor(dm, dataset_id)
    adata = adaptor.adata
    t0 = time.perf_counter()

    def _prepare():
        key, detected = _sample_key(adata.obs, body, set())
        req = planner.DERequestModel(**{**body.model_dump(), "sample_key": key})
        return planner.prepare_de(adata, req), detected

    try:
        (model, grid), detected = await asyncio.to_thread(_prepare)
    except planner.PlannerError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return _stream(
        dataset_id, adaptor,
        lambda stop: planner.iter_de_plan(model, grid, body.target_power, stop, t0),
        {"sample_key_detected": detected})
