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
import threading
import uuid
from contextlib import aclosing
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from scview.core.sse import keepalive
from scview.config import Settings
from scview.core import preflight, provenance, working_copy
from scview.core.dataset_manager import DatasetManager
from scview.core.obs_writes import dataset_edit
from scview.core.ingestion.merge import MergeJoin
from scview.core.integration import (
    LABEL_COLUMN,
    METHOD_HARMONY,
    METHODS,
    IntegrationPlan,
    build_plan,
    estimate_joint,
    joint_before_step,
    joint_genes,
    default_name,
    joint_preflight,
    load_units,
    read_need,
    read_batch_mixing,
    record_batch_mixing,
    run_integration_streamed,
)
from scview.core.pipeline import SAVED_STATUSES
from scview.core.processing import iterate_new
from scview.core.run_watchdog import RunControl
from scview.core.uns_keys import BATCH_MIXING_KEY
from scview.dependencies import get_dataset_manager, get_settings_dep

logger = logging.getLogger(__name__)

router = APIRouter()


class IntegratePlanRequest(BaseModel):
    dataset_ids: list[str]
    labels: list[str] | None = None  # optional per-dataset labels for the `dataset` column


class IntegrateRequest(IntegratePlanRequest):
    name: str | None = None
    method: str = "harmony"  # harmony | scanorama | bbknn | none (core/integration.py METHODS)
    join: MergeJoin = MergeJoin.inner
    apply_reconciliation: bool = True
    # Filter the datasets scView never filtered first, with a single run's QC (the plan's
    # ``filtering`` says what that removes); datasets filtered before are left as they are.
    filter_units: bool = True
    params: dict[str, Any] = Field(default_factory=dict)


def _check_ids(ids: list[str], settings: Settings) -> None:
    # Cap before touching any file, so an oversized request costs nothing.
    if len(ids) > settings.MAX_INTEGRATE_DATASETS:
        raise HTTPException(
            status_code=400,
            detail=f"At most {settings.MAX_INTEGRATE_DATASETS} datasets can be integrated at once.",
        )


async def _plan(body: IntegratePlanRequest, dm: DatasetManager, settings: Settings, *,
                join: MergeJoin = MergeJoin.inner, method: str = METHOD_HARMONY,
                overrides: dict[str, Any] | None = None):
    """The plan, the units read for it, and the joint run's memory estimate (None when
    the datasets cannot be merged). The plan carries the estimate's note (``preflight``),
    and is blocked when the run cannot fit even with every other dataset closed."""
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
    est = None
    genes = joint_genes(plan, join)
    if len(units) >= 2 and genes > 0:
        try:
            est = await asyncio.to_thread(estimate_joint, units, genes, method=method,
                                          overrides=overrides)
            note = await asyncio.to_thread(joint_preflight, est, preflight.available_memory(),
                                           preflight.cached_others(dm))
            plan.preflight = note["verdict"]
            if note["blocking"]:
                plan.blocking.append(note["blocking"])
                plan.ok = False
        except Exception as e:  # noqa: BLE001 - never block a plan on the estimate itself
            logger.warning("No joint run estimate: %s", e, exc_info=True)
    return plan, units, est


@router.post("/integrate/plan", response_model=IntegrationPlan)
async def plan_integration(
    body: IntegratePlanRequest,
    dm: DatasetManager = Depends(get_dataset_manager),
    settings: Settings = Depends(get_settings_dep),
) -> IntegrationPlan:
    """Read-only compatibility check for a joint embedding, with its memory and time
    estimate (``preflight``)."""
    plan, units, _ = await _plan(body, dm, settings)
    del units
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

    Events: ``stage`` (merge/persist; the first one carries the new dataset id, which
    Cancel takes: ``POST /datasets/{id}/assessment/cancel``), ``created``, the pipeline's
    ``step_start``/``step_done``/``step_error``/``substep``, ``watchdog`` (memory, slow
    step, waiting: as a pipeline run's), ``pipeline_complete``, then ``complete`` once
    the dataset is registered, or ``error``.

    The run gets a pipeline run's protection (``processing.iterate_new``): the joint
    run's estimated growth (``integration.estimate_joint``) is made room for and reserved
    until it ends, it waits behind earlier runs, and the merge and each step are checked
    before they start (``integration.joint_before_step``). A run that is stopped, cancelled
    or fails ends with ``error`` and leaves no joint dataset (the merged counts already
    written are removed); the source datasets are never changed.
    """
    if body.method not in METHODS:
        raise HTTPException(status_code=400, detail=f"Unknown method '{body.method}'.")
    plan, units, est = await _plan(body, dm, settings, join=body.join, method=body.method,
                                   overrides=body.params)
    if not plan.ok:
        raise HTTPException(status_code=400, detail=plan.blocking[0])

    dataset_id = uuid.uuid4().hex
    name = (body.name or "").strip() or default_name(list(units))
    merged = threading.Event()
    control = RunControl(before_step=joint_before_step(est.merge_bytes if est is not None else 0))
    needs = est.step_needs() if est is not None else {}
    if est is not None:
        control.watchdog.expected = {e.step: e.seconds for e in est.run.steps}
    ended: dict[str, Any] = {}

    def remaining() -> int:
        done = control.watchdog.steps  # filled as each pipeline step finishes
        return max((b for s, b in needs.items()
                    if s not in done and not (s == "merge" and merged.is_set())), default=0)

    def make_events():
        for event, data in run_integration_streamed(
            units,
            dataset_id=dataset_id,
            name=name,
            data_dir=Path(settings.DATA_DIR),
            output_path_for=lambda src: dm.derived_h5ad_path(dataset_id, src),
            join=body.join,
            method=body.method,
            apply_reconciliation=body.apply_reconciliation,
            overrides=body.params,
            control=control,
            on_merged=merged.set,
            filter_units_first=body.filter_units,
        ):
            if event == "pipeline_complete":
                ended.update(data)
            yield event, data

    def saved() -> bool:
        return ended.get("status") in SAVED_STATUSES

    async def events():
        try:
            # Steps run in a worker thread so the server stays responsive (other panels
            # keep loading) during a minutes-long integration. aclosing: a client that goes
            # away stops the run after its current step.
            async with aclosing(iterate_new(dm, dataset_id, make_events, control, name=name,
                                            need=est.need if est is not None else 0,
                                            remaining=remaining, saved=saved)) as items:
                async for event, data in items:
                    yield _sse(event, data)
            if not saved():
                why = ended.get("stop_reason") or "The run ended before it finished."
                if ended.get("status") == "cancelled":  # the pipeline's sentence speaks of a source dataset
                    why = (why.split(":")[0] + ": the joint dataset was not kept; the source datasets "
                           "are unchanged.")
                yield _sse("error", {"error": f"The joint embedding did not finish. {why}",
                                     "status": ended.get("status")})
                return
            await dm.load_dataset(dataset_id)
            yield _sse("complete", {"dataset_id": dataset_id, "name": name})
        except Exception as e:  # noqa: BLE001 — surface to the client, keep the server up
            logger.error("Integration %s failed: %s", dataset_id, e, exc_info=True)
            detail = e.detail if isinstance(e, HTTPException) else str(e)
            yield _sse("error", {"error": f"The joint embedding failed: {detail} The joint "
                                          "dataset was not kept; the source datasets are unchanged."})

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
