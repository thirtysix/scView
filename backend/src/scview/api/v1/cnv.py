"""Copy-number inference from expression, inferCNV-style (core/cnv.py).

``GET /cnv`` says whether a run can start (genome, genes with a position, the
time and memory estimate) and gives the saved result with each group's score
box and the CNV clusters; ``POST /cnv/run`` runs it and streams it (SSE), then
saves it into the dataset; ``GET /cnv/heatmap`` gives sampled cells x genome.

The run reads the dataset's matrix block by block and keeps only its own
cells x windows result, so the dataset stays open to other requests meanwhile;
only the final save holds the dataset's edit lock. A client that goes away (the
Cancel button) stops the run after the block it is on, or ends the Leiden child,
and nothing is saved.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from scview.core import cnv
from scview.core import large_mode
from scview.core import preflight
from scview.core import working_copy
from scview.core.dataset_manager import DatasetManager
from scview.core.obs_writes import dataset_edit, uns_snapshot, write_derived
from scview.core.run_watchdog import RunControl, RunWatchdog
from scview.core.sse import keepalive
from scview.dependencies import get_dataset_manager

logger = logging.getLogger(__name__)

router = APIRouter()

_DONE = object()
_running: set[str] = set()  # datasets with a run streaming: one at a time each

_WHAT = "Copy-number inference"
_WHY = "it reads every cell's expression of every positioned gene"


class RunRequest(BaseModel):
    reference_column: str
    reference_groups: list[str] = Field(min_length=1)
    window: int | None = None
    step: int | None = None
    exclude_chromosomes: list[str] | None = None
    min_frac: float | None = None
    # Values below this many SDs set to 0; null turns denoising off.
    threshold: float | None = cnv.DEFAULT_THRESHOLD
    genome: str = "auto"


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def _adaptor(dm: DatasetManager, dataset_id: str):
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    return adaptor


def _defaults() -> dict[str, Any]:
    return {"window": cnv.DEFAULT_WINDOW, "step": cnv.DEFAULT_STEP, "exclude": list(cnv.DEFAULT_EXCLUDE),
            "min_frac": cnv.DEFAULT_MIN_FRAC, "threshold": cnv.DEFAULT_THRESHOLD,
            "min_window": cnv.MIN_WINDOW, "max_window": cnv.MAX_WINDOW, "max_step": cnv.MAX_STEP,
            "min_ref_cells": cnv.MIN_REF_CELLS, "high_ratio": cnv.HIGH_RATIO}


def _estimate(n_cells: int, n_genes: int | None, window: int, step: int) -> dict[str, Any]:
    n_genes = int(n_genes or 0)
    n_windows = max(1, n_genes // step)
    need = preflight.cnv_need(n_cells, n_genes, n_windows)
    avail = preflight.available_memory()
    return {"need_bytes": need, "available_bytes": avail.available_bytes,
            "level": preflight.memory_level(need, avail.available_bytes),
            "seconds": cnv.estimate_seconds(n_cells, n_genes, n_windows)}


@router.get("/datasets/{dataset_id}/cnv")
async def get_cnv(
    dataset_id: str,
    groupby: str = Query(default="", description="Score box per group of this column (default: the reference column)"),
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """The saved result (``available``; per group of ``groupby`` the score box, the
    CNV clusters with gained / lost chromosomes), ``runnable`` (genome, genes with a
    position, where the values come from), the estimate, the defaults, and whether
    a run is working on the dataset now."""
    adata = (await _adaptor(dm, dataset_id)).adata

    def work() -> dict[str, Any]:
        if large_mode.is_large(adata):
            runnable = {"ok": False, "reason": str(large_mode.LargeDatasetError(_WHAT, _WHY)),
                        "genome": None, "assembly": None, "genome_why": None, "source": None,
                        "n_placed": None, "match": None}
        else:
            runnable = cnv.runnable(adata)
        out = {**cnv.summary(adata, groupby or None), "runnable": runnable, "defaults": _defaults(),
               "n_cells_total": int(adata.n_obs)}
        if runnable["ok"]:
            out["estimate"] = _estimate(adata.n_obs, runnable["n_placed"], cnv.DEFAULT_WINDOW,
                                        cnv.DEFAULT_STEP)
        return out

    out = await asyncio.to_thread(work)
    out["running"] = dataset_id in _running
    return out


async def _save(dm: DatasetManager, dataset_id: str, plan: cnv.CnvPlan, res: cnv.CnvResult,
                resources: dict, fingerprint: str) -> dict[str, Any]:
    """Store the result in the dataset under its edit lock and save it."""
    async with dataset_edit(dataset_id) as held:
        adaptor = await _adaptor(dm, dataset_id)
        if adaptor.cells_fingerprint() != fingerprint:
            raise cnv.CnvError("The dataset's cells changed while copy number ran (a pipeline run or a "
                               "filter); nothing was saved. Run it again.")
        output_path = str(dm.derived_h5ad_path(dataset_id, adaptor.h5ad_path))
        backed = await working_copy.make_room(dm, dataset_id, adaptor)

        def work() -> dict[str, Any]:
            data = working_copy.in_memory(adaptor)
            before = uns_snapshot(data)
            out = cnv.save(data, plan, res, resources=resources, fingerprint=fingerprint)
            write_derived(data, output_path, adaptor.h5ad_path, before, obsm_keys=(cnv.OBSM_KEY,))
            return out

        try:
            result = await held.run(work)
        except Exception:
            dm.evict(dataset_id)  # the cached copy may hold a result that was never saved
            raise
        if backed:
            dm.evict(dataset_id)
            await dm.load_dataset(dataset_id)
        else:
            dm.saved_in_place(dataset_id, adaptor, output_path)
        return result


@router.post("/datasets/{dataset_id}/cnv/run")
async def run_cnv(
    dataset_id: str,
    body: RunRequest,
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """Infer copy number and stream it. Events: ``start`` (genes, windows, the
    reference, the estimate), ``progress`` after each block of cells, ``watchdog``
    memory warnings, ``result`` (the saved summary), then ``complete``, or
    ``error``. Saved only when the run reaches its end."""
    adaptor = await _adaptor(dm, dataset_id)
    large_mode.check(adaptor, _WHAT, _WHY)
    if dataset_id in _running:
        raise HTTPException(status_code=409, detail="Copy number is already running on this dataset.")
    adata = adaptor.adata
    try:
        plan = await asyncio.to_thread(
            cnv.plan, adata, body.reference_column, body.reference_groups, window=body.window,
            step=body.step, exclude=body.exclude_chromosomes, min_frac=body.min_frac,
            threshold=body.threshold, genome=body.genome)
    except cnv.CnvError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    fingerprint = await asyncio.to_thread(adaptor.cells_fingerprint)
    del adata, adaptor

    async def events():
        if dataset_id in _running:  # another request started between the check and here
            yield _sse("error", {"error": "Copy number is already running on this dataset."})
            return
        _running.add(dataset_id)
        stop = threading.Event()
        watchdog = RunWatchdog()
        control = RunControl(watchdog=watchdog, before_step=None)
        sent = 0
        gen = cnv.iter_cnv(plan, stop=stop.is_set)
        pending: asyncio.Future | None = None

        def step_next():
            # The Leiden child registers with the control, so Cancel ends it.
            with control.step("cnv"):
                return next(gen, _DONE)

        def finish(fut: asyncio.Future | None = None) -> None:
            if fut is not None and not fut.cancelled():
                fut.exception()  # retrieved: a stopped run is not an error
            gen.close()
            watchdog.stop()
            _running.discard(dataset_id)

        try:
            watchdog.start()
            n_windows = int(plan.W.shape[1])
            yield _sse("start", {
                "n_cells": plan.n_cells, "n_genes": len(plan.genes), "n_windows": n_windows,
                "genome": plan.genome, "assembly": cnv.GENOMES[plan.genome]["assembly"],
                "reference_column": plan.ref_column, "reference_groups": plan.ref_groups,
                "n_reference_cells": plan.n_ref_cells, "source": plan.src.text, "match": plan.match,
                "n_low": plan.n_low, "notes": plan.notes,
                "estimate_seconds": cnv.estimate_seconds(plan.n_cells, len(plan.genes), n_windows),
            })
            while True:
                # Each block runs in a worker thread. Shielded, so a client going away
                # cancels this wait, not the thread's future, whose callback closes the run.
                pending = asyncio.ensure_future(asyncio.to_thread(step_next))
                item = await asyncio.shield(pending)
                pending = None
                for warning in watchdog.events[sent:]:
                    yield _sse("watchdog", warning)
                sent = len(watchdog.events)
                if item is _DONE:
                    break
                if item["type"] == "result":
                    resources = watchdog.summary()
                    result = await _save(dm, dataset_id, plan, item["result"],
                                         {"peak_gb": resources["peak_gb"], "limit_gb": resources["limit_gb"],
                                          "seconds": round(item["result"].seconds, 1)}, fingerprint)
                    yield _sse("result", {**result, "resources": resources})
                else:
                    yield _sse(item["type"], {k: v for k, v in item.items() if k != "type"})
            yield _sse("complete", {})
        except cnv.CnvCancelled:
            return
        except cnv.CnvError as e:
            yield _sse("error", {"error": str(e)})
        except Exception as e:  # noqa: BLE001 — surface to the client, keep the server up
            logger.error("Copy number failed for %s: %s", dataset_id, e, exc_info=True)
            detail = e.detail if isinstance(e, HTTPException) else str(e)
            yield _sse("error", {"error": f"Copy number failed: {detail}"})
        finally:
            stop.set()  # a thread mid-block stops before the next one
            control.cancel()  # and a Leiden child now
            if pending is not None and not pending.done():
                pending.add_done_callback(finish)
            else:
                finish()

    return StreamingResponse(
        keepalive(events()),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/datasets/{dataset_id}/cnv/heatmap")
async def get_cnv_heatmap(
    dataset_id: str,
    groupby: str = Query(default="", description="Group the rows by this column (default: the reference column)"),
    max_cells: int = Query(default=cnv.MAX_HEATMAP_CELLS, ge=100, le=5000),
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """Sampled cells (at most ``max_cells``) x genome, grouped by ``groupby``: int8
    codes in base64 (``value = code * scale``), with each chromosome's columns."""
    adata = (await _adaptor(dm, dataset_id)).adata
    try:
        return await asyncio.to_thread(cnv.heatmap, adata, groupby or None, max_cells)
    except cnv.CnvError as exc:
        raise HTTPException(status_code=404 if cnv.stored(adata) is None else 400, detail=str(exc))
