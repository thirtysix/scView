"""Spot composition endpoints (``core/spot_deconvolution.py``): Visium spots split into
the cell types of a reference dataset.

``GET /datasets/{id}/spot-deconvolution`` gives the methods, whether the dataset can be
fitted (``reason`` when not: no spots, large mode), the saved run and whether one is
working; ``GET .../reference/{ref}`` reads a candidate reference's label columns, cells
per type (fewer than 25 flagged), species and shared genes from its file (obs and var
only) with a memory and time estimate; ``POST .../run-stream`` fits the spots and saves
obsm ``spot_deconv`` / ``spot_deconv_cells`` and obs ``spot_deconv_top(_share)``
(SSE); ``GET .../share?type=`` one type's share per spot (the tissue overlay); ``POST
.../check-stream`` scores both methods on synthetic spots from held-out reference
cells (SSE, saves nothing); ``GET .../check`` the last check.

Both streamed runs copy what they need out of the two datasets first (in a worker
thread), then fit in blocks of spots, each in a worker thread; only the final save
holds the dataset's edit lock. A client that goes away stops the run after its block.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
from typing import Any, Literal

import anndata as ad
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from scview.core import large_mode, preflight, working_copy
from scview.core import spot_deconvolution as sd
from scview.core.dataset_manager import DatasetManager
from scview.core.obs_writes import dataset_edit, uns_snapshot, write_derived
from scview.core.run_watchdog import RunWatchdog
from scview.core.sse import keepalive
from scview.dependencies import get_dataset_manager

logger = logging.getLogger(__name__)

router = APIRouter()

_DONE = object()
_running: set[str] = set()  # spot datasets with a run or check streaming: one at a time each

_WHAT = "Spot composition"
_WHY = "it reads every spot's and every reference cell's counts into memory"


class RunRequest(BaseModel):
    reference_id: str = Field(max_length=128)
    celltype_column: str = Field(max_length=256)
    method: Literal["rctd", "nnls"] = "rctd"


class CheckRequest(BaseModel):
    reference_id: str = Field(max_length=128)
    celltype_column: str = Field(max_length=256)
    seed: int = Field(default=sd.SEED, ge=0, le=2**31 - 1)


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def _adaptor(dm: DatasetManager, dataset_id: str):
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    return adaptor


def _reason(adata) -> str | None:
    from scview.core import spatial

    if large_mode.is_large(adata):
        return str(large_mode.LargeDatasetError(_WHAT, _WHY))
    if not spatial.is_spot_data(adata):
        return sd.NOT_SPOTS
    return None


@router.get("/datasets/{dataset_id}/spot-deconvolution")
async def get_spot_deconvolution(dataset_id: str, dm: DatasetManager = Depends(get_dataset_manager)):
    """The methods and limits, whether the dataset can be fitted, the saved run, and
    whether a run is working on it now."""
    adata = (await _adaptor(dm, dataset_id)).adata
    reason = _reason(adata)
    return {"methods": list(sd.METHODS), "method": "rctd", "method_names": sd.METHOD_NAMES,
            "min_type_cells": sd.MIN_TYPE_CELLS, "max_types": sd.MAX_TYPES, "min_spot_umi": sd.MIN_SPOT_UMI,
            "min_shared_genes": sd.MIN_SHARED_GENES, "available": reason is None, "reason": reason,
            "saved": await asyncio.to_thread(sd.saved, adata) if reason is None else None,
            "n_spots": int(adata.n_obs), "running": dataset_id in _running, "note": sd.NOTE}


def _read_frames(path: str):
    from scview.api.v1.label_transfer import _read_frames as read

    return read(path)


def _estimate(ref_path: str, ref_open: bool, spots_adata, n_types: int, method: str) -> dict[str, Any]:
    ref_h = preflight.read_header(ref_path, "h5ad")
    spot_h = preflight.header_from_adata(spots_adata)
    secs = sd.estimate_seconds(ref_h.n_cells or 0, spots_adata.n_obs, n_types, method)
    return {"peak_bytes": sd.need_bytes(ref_h, spot_h, ref_open), "seconds": secs,
            "seconds_range": [round(secs * 0.5, 1), round(secs * 2, 1)], "note": sd.estimate_note(secs)}


def _spot_genes(adata) -> set[str]:
    names = adata.raw.var_names if adata.raw is not None and adata.raw.n_vars > adata.n_vars else adata.var_names
    return set(map(str, names))


_TYPE_WORDS = ("cell_type", "celltype", "subset", "annot", "label")


def _guess_column(obs, columns: list[dict[str, Any]]) -> str | None:
    """The reference's cell-type column: the first named like one whose types with
    MIN_TYPE_CELLS+ cells number 2 to MAX_TYPES (a lymphoid atlas's 'PrelimCellType' has
    54, its 'Subset' 34), else the first named like one, else the first label column."""
    named = [c["column"] for c in columns if any(k in c["column"].lower() for k in _TYPE_WORDS)]
    for col in named:
        types, _small, _vc = sd.type_counts(obs[col].astype(object).where(obs[col].notna(), None))
        if 2 <= len(types) <= sd.MAX_TYPES:
            return col
    if named:
        return named[0]
    return columns[0]["column"] if columns else None


@router.get("/datasets/{dataset_id}/spot-deconvolution/reference/{reference_id}")
async def reference_info(
    dataset_id: str,
    reference_id: str,
    celltype_column: str | None = Query(default=None, max_length=256),
    method: Literal["rctd", "nnls"] = "rctd",
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """A candidate reference: its cells, label columns, the chosen (or guessed) column's
    cells per type with those under 25 flagged, species against this dataset's, genes
    shared by name, and the run's memory and time estimate. Reads obs and var only."""
    from scview.core import label_transfer as lt
    from scview.core.species import detect_species

    if reference_id == dataset_id:
        raise HTTPException(status_code=400, detail="Pick another dataset as the reference.")
    paths = dm.h5ad_candidates(reference_id)
    if not paths:
        raise HTTPException(status_code=404, detail="Reference dataset not found.")
    spots = (await _adaptor(dm, dataset_id)).adata
    cached = dm.get_dataset(reference_id)

    def work() -> dict[str, Any]:
        if large_mode.file_is_large(paths[0]):
            return {"ok": False, "reason": str(large_mode.LargeDatasetError(_WHAT, _WHY))}
        obs, var, _umap = _read_frames(paths[0])
        skel = ad.AnnData(obs=obs, var=var)
        columns = lt.label_columns(skel)
        col = celltype_column or _guess_column(obs, columns)
        types, small, vc = ([], [], None)
        if col is not None:
            if col not in obs.columns:
                return {"ok": False, "reason": f"The reference has no column '{col}'.", "label_columns": columns}
            types, small, vc = sd.type_counts(obs[col].astype(object).where(obs[col].notna(), None))
        ref_species = detect_species(skel).species
        spot_species = detect_species(spots).species
        spot_genes = _spot_genes(spots)
        shared = sum(1 for g in var.index.astype(str) if g in spot_genes)
        refusal = lt.check_species(ref_species, spot_species)
        if refusal is None and col is not None and len(types) < 2:
            refusal = f"'{col}' has fewer than 2 types with {sd.MIN_TYPE_CELLS}+ cells."
        if refusal is None and len(types) > sd.MAX_TYPES:
            refusal = f"'{col}' has {len(types)} types with {sd.MIN_TYPE_CELLS}+ cells (at most {sd.MAX_TYPES})."
        return {
            "ok": refusal is None, "reason": refusal, "n_cells": int(len(obs)), "n_genes": int(len(var)),
            "label_columns": columns, "celltype_column": col,
            "types": [{"type": t, "n_cells": int(vc[t]), "small": t in small}
                      for t in [*types, *small]] if vc is not None else [],
            "n_small": len(small), "species": ref_species, "dataset_species": spot_species,
            "shared_genes_by_name": shared, "min_shared_genes": sd.MIN_SHARED_GENES,
            "estimate": _estimate(paths[0], cached is not None, spots, len(types), method),
        }

    try:
        return await asyncio.to_thread(work)
    except Exception as e:  # noqa: BLE001 - an unreadable file is reported, not a 500
        logger.warning("Could not read reference %s: %s", reference_id, e, exc_info=True)
        raise HTTPException(status_code=422, detail=f"Could not read the reference: {e}")


async def _setup(dm: DatasetManager, dataset_id: str, reference_id: str, method: str):
    """Shared refusals of both streams; the estimate, the reference's info and paths."""
    if reference_id == dataset_id:
        raise HTTPException(status_code=400, detail="Pick another dataset as the reference.")
    if dataset_id in _running:
        raise HTTPException(status_code=409, detail="A spot composition run is already working on this dataset.")
    paths = dm.h5ad_candidates(reference_id)
    if not paths:
        raise HTTPException(status_code=404, detail="Reference dataset not found.")
    if large_mode.file_is_large(paths[0]):
        raise large_mode.LargeDatasetError(f"{_WHAT} from this reference", _WHY)
    await dm.wait_until_idle(reference_id)  # a pipeline run on it finishes first
    adaptor = await _adaptor(dm, dataset_id)
    large_mode.check(adaptor, _WHAT, _WHY)
    reason = _reason(adaptor.adata)
    if reason:
        raise HTTPException(status_code=422, detail=reason)
    est = await asyncio.to_thread(_estimate, paths[0], dm.get_dataset(reference_id) is not None, adaptor.adata,
                                  20, method)
    # Room for the reference's copy: other datasets closed first (least recently used),
    # a 507 when it cannot fit even then.
    await dm.reserve(dataset_id, est["peak_bytes"], what=_WHAT)
    info = dm.get_dataset_info(reference_id) or {}
    ref_info = {"id": reference_id, "name": str(info.get("name") or reference_id), "path": paths[0],
                "mtime": round(os.path.getmtime(paths[0]), 3)}
    return adaptor, paths, est, ref_info


def _preparer(dm: DatasetManager, dataset_id: str, reference_id: str, paths: list[str], column: str,
              ref_info: dict, cluster_column: str | None, check: bool, seed: int = sd.SEED):
    def prepare(spot_adaptor) -> sd.Prepared:
        cached = dm.get_dataset(reference_id)
        if cached is not None and not getattr(cached.adata, "isbacked", False):
            ref = cached.adata
        else:
            ref = ad.read_h5ad(paths[0])  # read directly: opening it in the cache could close this one
        spots = spot_adaptor.adata
        if getattr(spots, "isbacked", False):
            spots = spots.to_memory()
        return sd.prepare(ref, spots, celltype_column=column, ref_info=ref_info, ref_fallback=paths[1:],
                          spot_fallback=dm.h5ad_candidates(dataset_id)[1:], cluster_column=cluster_column,
                          check=check, seed=seed)
    return prepare


async def _save(dm: DatasetManager, dataset_id: str, p: sd.Prepared, out: dict, summary: dict,
                resources: dict, fingerprint: str) -> dict[str, Any]:
    """Write the arrays and columns into the dataset under its edit lock and save it."""
    async with dataset_edit(dataset_id) as held:
        adaptor = await _adaptor(dm, dataset_id)
        if adaptor.cells_fingerprint() != fingerprint:
            raise sd.SpotDeconvolutionError("This dataset's spots changed while the fit ran (a pipeline run or a "
                                            "filter); nothing was saved. Run it again.", 409)
        output_path = str(dm.derived_h5ad_path(dataset_id, adaptor.h5ad_path))
        backed = await working_copy.make_room(dm, dataset_id, adaptor)

        def work() -> dict[str, Any]:
            data = working_copy.in_memory(adaptor)
            before = uns_snapshot(data)
            result = sd.save(data, p, out, summary, resources=resources)
            write_derived(data, output_path, adaptor.h5ad_path, before, obsm_keys=(sd.OBSM_RNA, sd.OBSM_CELLS))
            return result

        try:
            result = await held.run(work)
        except Exception:
            dm.evict(dataset_id)  # the cached copy may hold arrays that were never saved
            raise
        if backed:
            dm.evict(dataset_id)
            await dm.load_dataset(dataset_id)
            adaptor = await _adaptor(dm, dataset_id)
        else:
            dm.saved_in_place(dataset_id, adaptor, output_path)
        sd.remember(adaptor, result, summary["result_id"])
        return result


def _stream(dataset_id: str, events_of) -> StreamingResponse:
    return StreamingResponse(keepalive(events_of()), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def _events(dm: DatasetManager, dataset_id: str, start: dict, prepare, iterate, finish_item, label: str):
    """The stream both endpoints share: prepare in a worker thread, then each item of
    ``iterate(prepared, stop)`` in a worker thread, ``finish_item`` on the last one."""

    async def events():
        if dataset_id in _running:  # another request started between the check and here
            yield _sse("error", {"error": "A spot composition run is already working on this dataset."})
            return
        _running.add(dataset_id)
        stop = threading.Event()
        watchdog = RunWatchdog()
        sent = 0
        gen = None
        pending: asyncio.Future | None = None

        def finish(fut: asyncio.Future | None = None) -> None:
            if fut is not None and not fut.cancelled():
                fut.exception()  # retrieved: a stopped run is not an error
            if gen is not None:
                gen.close()
            watchdog.stop()
            _running.discard(dataset_id)

        try:
            watchdog.start()
            yield _sse("start", start)
            yield _sse("progress", {"step": "read", "message": "Reading the reference's and the spots' counts"})
            watchdog.step_started(sd.STEP)
            spot_adaptor = await _adaptor(dm, dataset_id)
            pending = asyncio.ensure_future(asyncio.to_thread(prepare, spot_adaptor))
            del spot_adaptor  # the thread holds it while it copies; the run needs only the copies
            prepared = await asyncio.shield(pending)
            pending = None
            yield _sse("progress", {"step": "prepared", "message": (
                f"{len(prepared.types)} types from {prepared.info.get('n_reference_cells', 0):,} reference cells, "
                f"{prepared.Y.shape[0]:,} spots, {len(prepared.genes):,} shared genes"),
                "n_shared_genes": len(prepared.genes), "types": prepared.types,
                "small_types": prepared.small_types})
            gen = iterate(prepared, stop.is_set)
            while True:
                # Each block in a worker thread. Shielded, so a client going away cancels
                # this wait, not the thread's future, whose callback then closes the run.
                pending = asyncio.ensure_future(asyncio.to_thread(next, gen, _DONE))
                item = await asyncio.shield(pending)
                pending = None
                for warning in watchdog.events[sent:]:
                    yield _sse("watchdog", warning)
                sent = len(watchdog.events)
                if item is _DONE:
                    break
                if item["type"] == "progress":
                    yield _sse("progress", {k: v for k, v in item.items() if k not in ("type", "s_g")})
                else:
                    watchdog.step_finished(sd.STEP)
                    body = await finish_item(prepared, item["out"], watchdog.summary())
                    yield _sse("result", body)
            yield _sse("complete", {})
        except sd.SpotDeconvolutionCancelled:
            return
        except sd.SpotDeconvolutionError as e:
            yield _sse("error", {"error": str(e)})
        except Exception as e:  # noqa: BLE001 - surface to the client, keep the server up
            logger.error("%s failed for %s: %s", label, dataset_id, e, exc_info=True)
            detail = e.detail if isinstance(e, HTTPException) else str(e)
            yield _sse("error", {"error": f"{label} failed: {detail}"})
        finally:
            stop.set()  # a thread mid-block stops before the next one
            if pending is not None and not pending.done():
                pending.add_done_callback(finish)
            else:
                finish()

    return events


@router.post("/datasets/{dataset_id}/spot-deconvolution/run-stream")
async def run_spot_deconvolution(dataset_id: str, body: RunRequest,
                                 dm: DatasetManager = Depends(get_dataset_manager)):
    """Fit the spots and stream it. Events: ``start`` (the estimate), ``progress`` per
    stage and block, ``watchdog`` memory warnings, ``result`` (the saved summary), then
    ``complete``, or ``error``. Saved only when the fit reaches its end."""
    import uuid

    adaptor, paths, est, ref_info = await _setup(dm, dataset_id, body.reference_id, body.method)
    cluster_column = adaptor.active_clustering_column()
    fingerprint = await asyncio.to_thread(adaptor.cells_fingerprint)
    prepare = _preparer(dm, dataset_id, body.reference_id, paths, body.celltype_column, ref_info, cluster_column,
                        check=False)
    del adaptor

    async def finish_item(prepared, out, res):
        summary = sd.summarize(prepared, out, method=body.method)
        summary["result_id"] = uuid.uuid4().hex[:12]
        resources = {"peak_gb": res["peak_gb"], "limit_gb": res["limit_gb"], "seconds": out.get("seconds")}
        return {**await _save(dm, dataset_id, prepared, out, summary, resources, fingerprint), "resources": res}

    start = {"reference": ref_info["name"], "reference_id": body.reference_id,
             "celltype_column": body.celltype_column, "method": body.method, "estimate": est}
    return _stream(dataset_id, _events(dm, dataset_id, start, prepare,
                                       lambda p, stop: sd.iter_fit(p, method=body.method, stop=stop),
                                       finish_item, "Spot composition"))


@router.post("/datasets/{dataset_id}/spot-deconvolution/check-stream")
async def run_check(dataset_id: str, body: CheckRequest, dm: DatasetManager = Depends(get_dataset_manager)):
    """Score both methods on synthetic spots drawn from held-out reference cells (SSE:
    ``progress`` per method and depth, ``result``, ``complete``). Saves nothing in the
    dataset; the check is kept in the results store."""
    adaptor, paths, est, ref_info = await _setup(dm, dataset_id, body.reference_id, "rctd")
    prepare = _preparer(dm, dataset_id, body.reference_id, paths, body.celltype_column, ref_info, None,
                        check=True, seed=body.seed)
    del adaptor

    async def finish_item(prepared, out, res):
        adaptor = await _adaptor(dm, dataset_id)
        sd.remember_check(adaptor, out)
        return {**out, "resources": res}

    start = {"reference": ref_info["name"], "reference_id": body.reference_id,
             "celltype_column": body.celltype_column, "n_spots": sd.CHECK_SPOTS}
    return _stream(dataset_id, _events(dm, dataset_id, start, prepare,
                                       lambda p, stop: sd.iter_check(p, stop=stop),
                                       finish_item, "The spot composition self-check"))


@router.get("/datasets/{dataset_id}/spot-deconvolution/check")
async def last_check(dataset_id: str, allow_empty: bool = False,
                     dm: DatasetManager = Depends(get_dataset_manager)):
    """The last self-check on this dataset, or 404; with ``allow_empty`` null instead (the panel asks on open,
    and a 404 is a console error in the browser)."""
    c = sd.last_check(await _adaptor(dm, dataset_id))
    if c is None:
        if allow_empty:
            return None
        raise HTTPException(status_code=404, detail="No spot composition self-check on this dataset yet.")
    return c


@router.get("/datasets/{dataset_id}/spot-deconvolution/share")
async def share(
    dataset_id: str,
    type: str = Query(max_length=512),  # noqa: A002 - the API's word
    kind: Literal["rna", "cell"] = "rna",
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """One type's share per spot from the saved run (null: not fitted), in spot order."""
    import numpy as np

    adata = (await _adaptor(dm, dataset_id)).adata
    try:
        values = await asyncio.to_thread(sd.share_values, adata, type, kind)
    except sd.SpotDeconvolutionError as e:
        raise HTTPException(status_code=e.status, detail=str(e))
    return {"type": type, "kind": kind,
            "values": [None if not np.isfinite(v) else round(float(v), 5) for v in values.tolist()]}
