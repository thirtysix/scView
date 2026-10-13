"""Joint multi-dataset integration endpoints.

``POST /integrate/plan`` checks a set of datasets (raw counts available, gene
overlap, shared annotations) without changing anything; ``POST
/integrate/run-stream`` builds the joint dataset and processes it, streaming
progress as Server-Sent Events. See ``core/integration.py``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from scview.config import Settings
from scview.core.dataset_manager import DatasetManager
from scview.core.ingestion.merge import MergeJoin
from scview.core.integration import (
    METHODS,
    IntegrationPlan,
    build_plan,
    load_units,
    run_integration_streamed,
)
from scview.dependencies import get_dataset_manager, get_settings_dep

logger = logging.getLogger(__name__)

router = APIRouter()

_DONE = object()


class IntegratePlanRequest(BaseModel):
    dataset_ids: list[str]
    labels: list[str] | None = None  # optional per-dataset labels for the `dataset` column


class IntegrateRequest(IntegratePlanRequest):
    name: str | None = None
    method: str = "harmony"  # harmony | none
    join: MergeJoin = MergeJoin.inner
    apply_reconciliation: bool = True
    params: dict[str, Any] = Field(default_factory=dict)


def _check_ids(ids: list[str], settings: Settings) -> None:
    # Cap before touching any file, so an oversized request costs nothing.
    if len(ids) > settings.MAX_INTEGRATE_DATASETS:
        raise HTTPException(
            status_code=400,
            detail=f"At most {settings.MAX_INTEGRATE_DATASETS} datasets can be integrated at once.",
        )


async def _plan(body: IntegratePlanRequest, dm: DatasetManager, settings: Settings):
    _check_ids(body.dataset_ids, settings)
    checks, units = await asyncio.to_thread(load_units, dm, body.dataset_ids, body.labels)
    plan = build_plan(
        checks, units,
        max_datasets=settings.MAX_INTEGRATE_DATASETS,
        max_cells=settings.MAX_INTEGRATE_CELLS,
    )
    return plan, units


@router.post("/integrate/plan", response_model=IntegrationPlan)
async def plan_integration(
    body: IntegratePlanRequest,
    dm: DatasetManager = Depends(get_dataset_manager),
    settings: Settings = Depends(get_settings_dep),
) -> IntegrationPlan:
    """Read-only compatibility check for a joint embedding."""
    plan, _ = await _plan(body, dm, settings)
    return plan


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@router.post("/integrate/run-stream")
async def run_integration(
    body: IntegrateRequest,
    dm: DatasetManager = Depends(get_dataset_manager),
    settings: Settings = Depends(get_settings_dep),
):
    """Build + process a joint dataset, streaming progress as SSE.

    Events: ``stage`` (merge/persist), ``created`` (the new dataset id), the
    pipeline's ``step_start``/``step_done``/``step_error``/``substep``,
    ``pipeline_complete``, then ``complete`` once the dataset is registered — or
    ``error``.
    """
    if body.method not in METHODS:
        raise HTTPException(status_code=400, detail=f"Unknown method '{body.method}'.")
    plan, units = await _plan(body, dm, settings)
    if not plan.ok:
        raise HTTPException(status_code=400, detail=plan.blocking[0])

    dataset_id = uuid.uuid4().hex
    name = (body.name or "").strip() or "joint: " + " + ".join(units)
    gen = run_integration_streamed(
        units,
        dataset_id=dataset_id,
        name=name,
        data_dir=Path(settings.DATA_DIR),
        output_path_for=lambda src: dm.derived_h5ad_path(dataset_id, src),
        join=body.join,
        method=body.method,
        apply_reconciliation=body.apply_reconciliation,
        overrides=body.params,
    )

    async def events():
        try:
            # Each step runs in a worker thread so the server stays responsive
            # (other panels keep loading) during a minutes-long integration.
            while True:
                item = await asyncio.to_thread(next, gen, _DONE)
                if item is _DONE:
                    break
                yield _sse(*item)
            await dm.load_dataset(dataset_id)
            yield _sse("complete", {"dataset_id": dataset_id, "name": name})
        except Exception as e:  # noqa: BLE001 — surface to the client, keep the server up
            logger.error("Integration %s failed: %s", dataset_id, e, exc_info=True)
            yield _sse("error", {"error": str(e)})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
