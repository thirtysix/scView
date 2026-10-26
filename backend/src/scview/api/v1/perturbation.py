"""CRISPR screens: guide assignment (``core/crispr.py``) and Mixscape (``core/mixscape.py``).

``GET /perturbation`` lists the guide sources (guide counts, Cell Ranger calls,
guide-like obs columns), the saved assignment and the last Mixscape run;
``POST /perturbation/preview`` shows what an assignment would write;
``POST /perturbation/assign`` saves ``guide_call`` / ``perturbation`` /
``n_guides``; ``POST /perturbation/mixscape/run-stream`` runs Mixscape and
streams it (SSE), then saves ``mixscape_class`` / ``mixscape_class_global`` /
``mixscape_p_ko``; ``GET /perturbation/mixscape`` gives the last run.

Mixscape reads the dataset (no copy beyond the tested genes of the used cells)
in worker threads; only the final save holds the edit lock. A client that goes
away stops the run at its next step and nothing is saved.
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

from scview.core import crispr, mixscape
from scview.core.dataset_manager import DatasetManager
from scview.core.obs_writes import dataset_edit, edit_room, editable, save_edit, uns_snapshot
from scview.core.sse import keepalive
from scview.dependencies import get_dataset_manager

logger = logging.getLogger(__name__)

router = APIRouter()

_DONE = object()
_running: set[str] = set()  # datasets with a Mixscape run streaming: one at a time each


class AssignRequest(BaseModel):
    source: str | None = None  # cellranger | counts | column; None: the suggested one
    column: str | None = None
    min_umi: float = crispr.MIN_UMI
    min_ratio: float = crispr.MIN_RATIO
    mapping: dict[str, str] = Field(default_factory=dict)  # guide -> target overrides
    control: str | None = None


class MixscapeRequest(BaseModel):
    labels: str | None = None  # default: the saved assignment's perturbation column
    control: str | None = None  # default: the saved assignment's control label
    split_by: str | None = None
    n_neighbors: int = mixscape.N_NEIGHBORS
    n_dims: int = mixscape.N_DIMS
    logfc_threshold: float = mixscape.LOGFC
    pval_cutoff: float = mixscape.PVAL
    min_de_genes: int = mixscape.MIN_DE_GENES
    iter_num: int = mixscape.ITER_NUM
    lda: bool = True


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=_plain)}\n\n"


def _plain(v: Any) -> Any:
    try:
        return v.item()  # numpy scalars
    except AttributeError:
        return str(v)


async def _adaptor(dm: DatasetManager, dataset_id: str):
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")
    return adaptor


def _overview(adaptor) -> dict[str, Any]:
    out = crispr.summary(adaptor.adata)
    r = mixscape.last_result(adaptor)
    out["mixscape"] = r.to_dict() if r is not None else None
    out["mixscape_saved"] = mixscape.saved_record(adaptor.adata) if r is None else None
    out["mixscape_defaults"] = {"n_neighbors": mixscape.N_NEIGHBORS, "n_dims": mixscape.N_DIMS,
                                "logfc_threshold": mixscape.LOGFC, "pval_cutoff": mixscape.PVAL,
                                "min_de_genes": mixscape.MIN_DE_GENES, "iter_num": mixscape.ITER_NUM}
    return out


@router.get("/datasets/{dataset_id}/perturbation")
async def get_perturbation(dataset_id: str, dm: DatasetManager = Depends(get_dataset_manager)):
    adaptor = await _adaptor(dm, dataset_id)
    out = await asyncio.to_thread(_overview, adaptor)
    out["running"] = dataset_id in _running
    return out


@router.post("/datasets/{dataset_id}/perturbation/preview")
async def post_preview(dataset_id: str, body: AssignRequest, dm: DatasetManager = Depends(get_dataset_manager)):
    """What ``assign`` would write: the guide -> target mapping, the control, cells
    per target, multiplets and unassigned cells. Nothing is changed."""
    adaptor = await _adaptor(dm, dataset_id)
    try:
        p = await asyncio.to_thread(crispr.preview, adaptor.adata, body.source, body.column,
                                    min_umi=body.min_umi, min_ratio=body.min_ratio, mapping=body.mapping,
                                    control=body.control)
    except crispr.CrisprError as e:
        raise HTTPException(status_code=e.status, detail=str(e))
    p.pop("_columns", None)
    return p


@router.post("/datasets/{dataset_id}/perturbation/assign")
async def post_assign(dataset_id: str, body: AssignRequest, dm: DatasetManager = Depends(get_dataset_manager)):
    """Save each cell's guide and target (``guide_call``, ``perturbation``, ``n_guides``)."""
    async with dataset_edit(dataset_id) as held:
        adaptor = await _adaptor(dm, dataset_id)
        output_path = str(dm.derived_h5ad_path(dataset_id, adaptor.h5ad_path))
        backed = await edit_room(dm, dataset_id, adaptor)

        def _run():
            work = editable(adaptor)
            before = uns_snapshot(work)
            result = crispr.assign(work, body.source, body.column, min_umi=body.min_umi,
                                   min_ratio=body.min_ratio, mapping=body.mapping, control=body.control)
            save_edit(adaptor, work, output_path, before)
            return result

        try:
            result = await held.run(_run)
        except crispr.CrisprError as e:
            raise HTTPException(status_code=e.status, detail=str(e))  # raised before any change
        except Exception as e:
            dm.evict(dataset_id)  # the cached copy may hold columns that were never saved
            logger.error("Guide assignment failed for %s: %s", dataset_id, e, exc_info=True)
            raise HTTPException(status_code=500, detail=f"Could not save the guide assignment: {e!s}")
        if backed:
            dm.evict(dataset_id)
            await dm.load_dataset(dataset_id)
        else:
            dm.saved_in_place(dataset_id, adaptor, output_path)
        return result


def _params(adata, body: MixscapeRequest) -> mixscape.Params:
    call = crispr.saved_call(adata)
    labels = body.labels or (call["columns"]["perturbation"] if call else None)
    control = body.control or (call["control"] if call else None)
    if not labels or not control:
        raise mixscape.MixscapeError("Assign the guides first (or name the label column and the control).")
    return mixscape.Params(labels=labels, control=control, split_by=body.split_by or None,
                           n_neighbors=body.n_neighbors, n_dims=body.n_dims, logfc_threshold=body.logfc_threshold,
                           pval_cutoff=body.pval_cutoff, min_de_genes=body.min_de_genes,
                           iter_num=body.iter_num, lda=body.lda)


async def _save(dm: DatasetManager, dataset_id: str, result: mixscape.MixscapeResult, columns: dict,
                fingerprint: str) -> dict:
    async with dataset_edit(dataset_id) as held:
        adaptor = await _adaptor(dm, dataset_id)
        if adaptor.cells_fingerprint() != fingerprint:
            raise mixscape.MixscapeError("The dataset's cells changed during the run (a pipeline run or a "
                                         "filter); nothing was saved. Run it again.", 409)
        output_path = str(dm.derived_h5ad_path(dataset_id, adaptor.h5ad_path))
        backed = await edit_room(dm, dataset_id, adaptor)

        def _run():
            work = editable(adaptor)
            before = uns_snapshot(work)
            out = mixscape.save(work, result, columns)
            save_edit(adaptor, work, output_path, before)
            return out

        try:
            out = await held.run(_run)
        except mixscape.MixscapeError:
            raise
        except Exception:
            dm.evict(dataset_id)
            raise
        if backed:
            dm.evict(dataset_id)
            adaptor = await dm.load_dataset(dataset_id)
        else:
            dm.saved_in_place(dataset_id, adaptor, output_path)
        mixscape.remember(adaptor, result)
        return out


@router.post("/datasets/{dataset_id}/perturbation/mixscape/run-stream")
async def run_mixscape(dataset_id: str, body: MixscapeRequest, dm: DatasetManager = Depends(get_dataset_manager)):
    """Run Mixscape and stream it. Events: ``start`` (the plan), ``progress`` per
    step (neighbours, DE, signature, each target, LDA), ``result`` (saved), then
    ``complete``, or ``error``."""
    adaptor = await _adaptor(dm, dataset_id)
    adata = adaptor.adata
    try:
        params = _params(adata, body)
        pl = await asyncio.to_thread(mixscape.plan, adata, params)
    except mixscape.MixscapeError as e:
        raise HTTPException(status_code=e.status, detail=str(e))
    if dataset_id in _running:
        raise HTTPException(status_code=409, detail="Mixscape is already running on this dataset.")
    fingerprint = await asyncio.to_thread(adaptor.cells_fingerprint)

    async def events():
        if dataset_id in _running:  # another request started between the check and here
            yield _sse("error", {"error": "Mixscape is already running on this dataset."})
            return
        _running.add(dataset_id)
        stop = threading.Event()
        gen = mixscape.iter_mixscape(adata, pl, stop=stop.is_set)
        pending: asyncio.Future | None = None

        def finish(fut: asyncio.Future | None = None) -> None:
            if fut is not None and not fut.cancelled():
                fut.exception()  # retrieved: a stopped run is not an error
            gen.close()
            _running.discard(dataset_id)

        try:
            yield _sse("start", pl.to_dict())
            while True:
                # Each step in a worker thread; shielded, so a client going away cancels
                # this wait, not the thread, whose callback then closes the run.
                pending = asyncio.ensure_future(asyncio.to_thread(next, gen, _DONE))
                item = await asyncio.shield(pending)
                pending = None
                if item is _DONE:
                    break
                if item["type"] == "result":
                    out = await _save(dm, dataset_id, item["result"], item["columns"], fingerprint)
                    yield _sse("result", out)
                else:
                    yield _sse("progress", {k: v for k, v in item.items() if k != "type"})
            yield _sse("complete", {})
        except mixscape.MixscapeCancelled:
            return
        except mixscape.MixscapeError as e:
            yield _sse("error", {"error": str(e)})
        except Exception as e:  # noqa: BLE001 — surface to the client, keep the server up
            logger.error("Mixscape failed for %s: %s", dataset_id, e, exc_info=True)
            yield _sse("error", {"error": f"Mixscape failed: {e}"})
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


@router.get("/datasets/{dataset_id}/perturbation/mixscape")
async def get_mixscape(dataset_id: str, dm: DatasetManager = Depends(get_dataset_manager)):
    """The last Mixscape run on the current cells (per-target table, histograms, LDA)."""
    adaptor = await _adaptor(dm, dataset_id)
    r = await asyncio.to_thread(mixscape.last_result, adaptor)
    if r is None:
        raise HTTPException(status_code=404, detail="No Mixscape run for these cells: run it again.")
    return r.to_dict()
