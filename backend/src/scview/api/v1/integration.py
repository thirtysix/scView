"""Joint multi-dataset integration endpoints.

``POST /integrate/plan`` checks a set of datasets (raw counts available, gene
overlap, shared annotations) without changing anything; ``POST
/integrate/run-stream`` builds the joint dataset and processes it, streaming
progress as Server-Sent Events; ``GET /datasets/{id}/integration/summary``
reports a joint dataset's batch mixing. See ``core/integration.py``.
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

from scview.core.sse import keepalive
from scview.config import Settings
from scview.core import provenance, working_copy
from scview.core.dataset_manager import DatasetManager
from scview.core.obs_writes import dataset_edit
from scview.core.ingestion.merge import MergeJoin
from scview.core.integration import (
    LABEL_COLUMN,
    METHODS,
    IntegrationPlan,
    build_plan,
    load_units,
    read_need,
    read_batch_mixing,
    record_batch_mixing,
    run_integration_streamed,
)
from scview.core.uns_keys import BATCH_MIXING_KEY
from scview.dependencies import get_dataset_manager, get_settings_dep

logger = logging.getLogger(__name__)

router = APIRouter()

_DONE = object()


class IntegratePlanRequest(BaseModel):
    dataset_ids: list[str]
    labels: list[str] | None = None  # optional per-dataset labels for the `dataset` column


class IntegrateRequest(IntegratePlanRequest):
    name: str | None = None
    method: str = "harmony"  # harmony | scanorama | bbknn | none (core/integration.py METHODS)
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
    # A dataset a pipeline run is changing is read once the run has finished.
    for did in body.dataset_ids:
        await dm.wait_until_idle(did)
    # Every dataset's counts are read here, for a plan too: room first, and the reservation
    # holds until they are read (a 507 the Compare panel shows, ``JointEmbedding``).
    ids = body.dataset_ids
    need = await asyncio.to_thread(read_need, dm, ids)
    what = f"A joint embedding of {' + '.join(dm.dataset_name(d) for d in ids)}" if need else ""
    async with working_copy.reading(dm, ids[0] if ids else "", need, what, also=ids[1:]) as held:
        checks, units = await held.run(load_units, dm, ids, body.labels)
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
        keepalive(events()),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/datasets/{dataset_id}/integration/summary")
async def integration_summary(
    dataset_id: str,
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """For a joint dataset: its batch-mixing score and the columns to break it down by.

    ``joint`` is false for any other dataset. A joint dataset built before the
    score existed (or whose PCA was redone since) is scored now and the score saved.
    """
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    adata = adaptor.adata
    origin = provenance.read_provenance(adata)["source"].get("origin")
    info = dm.get_dataset_info(dataset_id) or {}
    if origin != "integrated" and info.get("source") != "integrated":
        return {"dataset_id": dataset_id, "joint": False}

    def _stale(m):  # scored before the iLISI / conservation metrics existed
        return m is None or "ilisi" not in m

    mixing = read_batch_mixing(adata)
    if _stale(mixing):
        async with dataset_edit(dataset_id) as held:  # saving writes the dataset's file
            mixing = read_batch_mixing(adata)  # another request may have scored it
            if _stale(mixing):
                mixing = await held.run(record_batch_mixing, adata)
                if mixing is not None:
                    try:
                        await held.run(
                            dm.persist_uns, dataset_id, adaptor, [BATCH_MIXING_KEY]
                        )
                    except Exception as e:  # still served from memory
                        logger.warning("Could not save the mixing score of %s: %s", dataset_id, e)
    return {
        "dataset_id": dataset_id,
        "joint": True,
        "label_column": LABEL_COLUMN if LABEL_COLUMN in adata.obs.columns else None,
        "cluster_column": adaptor.active_clustering_column(),
        "mixing": mixing,
    }
