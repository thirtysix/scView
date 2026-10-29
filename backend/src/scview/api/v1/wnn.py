"""WNN joint RNA + protein neighbours (core/wnn.py).

``GET /wnn`` says whether a run can start (the proteins used, the controls left
out, the PCs, the estimate) and gives the saved result, whether it still fits the
dataset, and the mean RNA weight per group of ``groupby``; ``POST /wnn/run-stream``
runs it and streams each stage (SSE), then saves it into the dataset;
``GET /wnn/weights`` gives the per-cell RNA weight (the overlay) and its values by
group (the violin).

The run works on its own copies of the RNA PCs and the protein values (a few MB),
so the dataset stays open to other requests meanwhile; only the final save holds
the dataset's edit lock. A client that goes away (the Cancel button) stops the run
before its next stage, or ends the Leiden child, and nothing is saved.
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

from scview.core import large_mode
from scview.core import wnn
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

_WHAT = "WNN (joint RNA + protein neighbours)"
_WHY = "it writes a neighbour graph of every cell into the dataset"
# What a run holds beyond the dataset (both modalities' 200 candidates per cell, the
# neighbour index, two UMAPs): pbmc10k (7,798 cells) peaked 0.9 GB over the loaded dataset.
_BASE_BYTES = 500 * 1024**2
_BYTES_PER_CELL = 20_000


class RunRequest(BaseModel):
    protein_pcs: int | None = Field(default=None, ge=2)  # default min(18, proteins - 1)
    resolution: float | None = Field(default=None, gt=0, le=10)  # Leiden on the WNN graph


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def _adaptor(dm: DatasetManager, dataset_id: str):
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    return adaptor


def _defaults() -> dict[str, Any]:
    return {"k": wnn.K_NN, "knn_range": wnn.KNN_RANGE, "max_protein_pcs": wnn.MAX_PROTEIN_PCS,
            "min_proteins": wnn.MIN_PROTEINS, "warn_proteins": wnn.WARN_PROTEINS,
            "resolution": wnn.DEFAULT_RESOLUTION}


@router.get("/datasets/{dataset_id}/wnn")
async def get_wnn(
    dataset_id: str,
    groupby: str = Query(default="", description="Mean RNA weight per group of this column"),
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """The saved result (``available``; ``stale`` with its reason when the cells, the
    PCs or the protein values changed since), ``runnable`` with the plan a run would
    use, the defaults, and whether a run is working on the dataset now."""
    adaptor = await _adaptor(dm, dataset_id)
    adata = adaptor.adata

    def work() -> dict[str, Any]:
        if large_mode.is_large(adata):
            runnable = {"ok": False, "reason": str(large_mode.LargeDatasetError(_WHAT, _WHY)), "plan": None}
        else:
            runnable = wnn.runnable(adata)
        return {**wnn.summary(adata, groupby or None, adaptor.cells_fingerprint()), "runnable": runnable,
                "defaults": _defaults(), "n_cells_total": int(adata.n_obs)}

    out = await asyncio.to_thread(work)
    out["running"] = dataset_id in _running
    return out


@router.get("/datasets/{dataset_id}/wnn/weights")
async def get_wnn_weights(
    dataset_id: str,
    groupby: str = Query(default="", description="Also split the values by this column (violin)"),
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """Each cell's RNA weight (its protein weight is 1 minus it), in the dataset's cell
    order, and split by ``groupby`` when given."""
    adata = (await _adaptor(dm, dataset_id)).adata

    def work() -> dict[str, Any]:
        info = wnn.stored(adata)
        if info is None:
            raise wnn.WnnError("No WNN result for this dataset yet: run it on the Protein tab.")
        values = wnn.weights(adata)
        out: dict[str, Any] = {"run_id": info["run_id"], "values": [round(float(v), 4) for v in values],
                               "violin": None, "column": groupby or None}
        if groupby:
            try:
                out["violin"] = wnn.violin(adata, groupby)
            except wnn.WnnError as exc:
                out["violin_error"] = str(exc)
        return out

    try:
        return await asyncio.to_thread(work)
    except wnn.WnnError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


async def _save(dm: DatasetManager, dataset_id: str, inp: wnn.WnnInputs, res: wnn.WnnResult,
                resources: dict, fingerprint: str) -> dict[str, Any]:
    """Store the result in the dataset under its edit lock and save it."""
    async with dataset_edit(dataset_id) as held:
        adaptor = await _adaptor(dm, dataset_id)
        if adaptor.cells_fingerprint() != fingerprint:
            raise wnn.WnnError("The dataset's cells changed while WNN ran (a pipeline run or a filter); "
                               "nothing was saved. Run it again.")
        output_path = str(dm.derived_h5ad_path(dataset_id, adaptor.h5ad_path))
        backed = await working_copy.make_room(dm, dataset_id, adaptor)

        def work() -> dict[str, Any]:
            data = working_copy.in_memory(adaptor)
            before = uns_snapshot(data)
            out = wnn.save(data, inp, res, resources=resources, cells_fingerprint=fingerprint)
            write_derived(data, output_path, adaptor.h5ad_path, before, obsm_keys=wnn.OBSM_KEYS,
                          obsp_keys=wnn.OBSP_KEYS)
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


@router.post("/datasets/{dataset_id}/wnn/run-stream")
async def run_wnn(
    dataset_id: str,
    body: RunRequest,
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """Run WNN and stream it. Events: ``start`` (the plan and the estimate),
    ``stage`` as each stage starts, ``watchdog`` memory warnings, ``result`` (the saved
    summary), then ``complete``, or ``error``. Saved only when the run reaches its end."""
    adaptor = await _adaptor(dm, dataset_id)
    large_mode.check(adaptor, _WHAT, _WHY)
    if dataset_id in _running:
        raise HTTPException(status_code=409, detail="WNN is already running on this dataset.")
    adata = adaptor.adata
    try:
        plan = await asyncio.to_thread(wnn.plan, adata, body.protein_pcs, body.resolution)
        inp = await asyncio.to_thread(wnn.prepare, adata, plan)
    except wnn.WnnError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    fingerprint = await asyncio.to_thread(adaptor.cells_fingerprint)
    await dm.free_memory_for(_BASE_BYTES + _BYTES_PER_CELL * plan.n_cells, keep=dataset_id)
    del adata, adaptor  # the run needs only its copies

    async def events():
        if dataset_id in _running:  # another request started between the check and here
            yield _sse("error", {"error": "WNN is already running on this dataset."})
            return
        _running.add(dataset_id)
        stop = threading.Event()
        watchdog = RunWatchdog()
        control = RunControl(watchdog=watchdog, before_step=None)
        sent = 0
        gen = wnn.iter_wnn(inp, stop=stop.is_set)
        pending: asyncio.Future | None = None

        def step_next():
            # The Leiden child registers with the control, so Cancel ends it.
            with control.step("wnn"):
                return next(gen, _DONE)

        def finish(fut: asyncio.Future | None = None) -> None:
            if fut is not None and not fut.cancelled():
                fut.exception()  # retrieved: a stopped run is not an error
            gen.close()
            watchdog.stop()
            _running.discard(dataset_id)

        try:
            watchdog.start()
            yield _sse("start", {**plan.to_dict(), "stages": [{"stage": s, "label": label}
                                                              for s, label in wnn.STAGES]})
            while True:
                # Each stage runs in a worker thread. Shielded, so a client going away
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
                    res = item["result"]
                    result = await _save(dm, dataset_id, inp, res,
                                         {"peak_gb": resources["peak_gb"], "limit_gb": resources["limit_gb"],
                                          "seconds": round(res.seconds, 1)}, fingerprint)
                    yield _sse("result", {**result, "resources": resources})
                else:
                    yield _sse(item["type"], {k: v for k, v in item.items() if k != "type"})
            yield _sse("complete", {})
        except wnn.WnnCancelled:
            return
        except wnn.WnnError as e:
            yield _sse("error", {"error": str(e)})
        except Exception as e:  # noqa: BLE001 — surface to the client, keep the server up
            logger.error("WNN failed for %s: %s", dataset_id, e, exc_info=True)
            detail = e.detail if isinstance(e, HTTPException) else str(e)
            yield _sse("error", {"error": f"WNN failed: {detail}"})
        finally:
            stop.set()  # a thread mid-stage stops before the next one
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
