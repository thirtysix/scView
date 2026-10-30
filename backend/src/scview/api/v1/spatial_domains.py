"""Spatial domains and niches (core/spatial_domains.py).

``GET /spatial/domains`` says whether a run can start and what it would need, and gives
the saved domains, niches and enrichment. Three runs stream their progress (SSE) and
then save into the dataset, each only when it reaches its end:
``POST /spatial/domains/run-stream``, ``POST /spatial/niches/run-stream``,
``POST /spatial/nhood-enrichment/run-stream``.

A run works on its own copies of the positions, the PCs and the labels, so the dataset
stays open to other requests meanwhile; only the final save holds its edit lock. No
run reads X, so a large dataset runs as one in memory. A client that goes away (the
Cancel button) stops the run before its next piece of work and nothing is saved.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from collections.abc import Callable, Iterator
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from scview.core import spatial_domains as sd
from scview.core.dataset_manager import DatasetManager
from scview.core.obs_writes import dataset_edit, edit_room, editable, save_edit, uns_snapshot
from scview.core.run_watchdog import RunControl, RunWatchdog
from scview.core.sse import keepalive
from scview.dependencies import get_dataset_manager

logger = logging.getLogger(__name__)

router = APIRouter()

_DONE = object()
_running: set[str] = set()  # datasets with one of these runs streaming: one at a time each


class DomainsRequest(BaseModel):
    k: int | None = Field(default=sd.DEFAULT_K, ge=2, le=sd.MAX_K)  # null: suggest it from the stability scan
    n_hops: int = Field(default=sd.DEFAULT_HOPS, ge=0, le=sd.MAX_HOPS)
    n_pcs: int = Field(default=sd.DEFAULT_PCS, ge=sd.MIN_PCS, le=500)
    seed: int = Field(default=sd.SEED, ge=0, le=2**31 - 1)
    column: str = Field(default=sd.DOMAIN_COLUMN, max_length=64)
    overwrite: bool = False  # the user's explicit choice to replace an earlier run's column
    k_min: int = Field(default=sd.SCAN_RANGE[0], ge=2, le=sd.MAX_K)
    k_max: int = Field(default=sd.SCAN_RANGE[1], ge=2, le=sd.MAX_K)
    repeats: int = Field(default=sd.SCAN_REPEATS, ge=2, le=20)


class EnrichmentRequest(BaseModel):
    column: str = Field(max_length=256)
    n_perms: int = Field(default=sd.N_PERMS, ge=10, le=sd.MAX_PERMS)
    seed: int = Field(default=sd.SEED, ge=0, le=2**31 - 1)


class NichesRequest(BaseModel):
    label_column: str = Field(max_length=256)
    k_window: int = Field(default=sd.NICHE_NEIGHBOURS, ge=2, le=sd.MAX_NICHE_NEIGHBOURS)
    n_niches: int = Field(default=sd.DEFAULT_NICHES, ge=2, le=sd.MAX_NICHES)
    seed: int = Field(default=sd.SEED, ge=0, le=2**31 - 1)
    column: str = Field(default=sd.NICHE_COLUMN, max_length=64)
    overwrite: bool = False


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def _adaptor(dm: DatasetManager, dataset_id: str):
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    return adaptor


def _http(exc: sd.SpatialDomainsError) -> HTTPException:
    return HTTPException(status_code=409 if isinstance(exc, sd.ColumnInUse) else 422, detail=str(exc))


def _defaults() -> dict[str, Any]:
    return {"k": sd.DEFAULT_K, "n_hops": sd.DEFAULT_HOPS, "n_pcs": sd.DEFAULT_PCS, "seed": sd.SEED,
            "k_min": sd.SCAN_RANGE[0], "k_max": sd.SCAN_RANGE[1], "repeats": sd.SCAN_REPEATS,
            "max_fit_obs": sd.MAX_FIT_OBS, "n_perms": sd.N_PERMS, "k_window": sd.NICHE_NEIGHBOURS,
            "n_niches": sd.DEFAULT_NICHES, "domain_column": sd.DOMAIN_COLUMN, "niche_column": sd.NICHE_COLUMN}


@router.get("/datasets/{dataset_id}/spatial/domains")
async def get_domains(
    dataset_id: str,
    k: int = Query(default=sd.DEFAULT_K, ge=2, le=sd.MAX_K),
    n_hops: int = Query(default=sd.DEFAULT_HOPS, ge=0, le=sd.MAX_HOPS),
    n_pcs: int = Query(default=sd.DEFAULT_PCS, ge=sd.MIN_PCS, le=500),
    enrichment_column: str = Query(default="", max_length=256),
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """The saved results (``domains``, ``niches``, ``enrichment``: each ``available``, ``stale``
    with its reason), ``runnable`` with the plan and estimates a run would use for these
    settings, ``free_column`` (a free name for ``k``), and whether a run is working now."""
    adaptor = await _adaptor(dm, dataset_id)
    adata = adaptor.adata

    def work() -> dict[str, Any]:
        fp = adaptor.cells_fingerprint()
        try:
            plan = sd.plan_domains(adata, k=k, n_hops=n_hops, n_pcs=n_pcs, check_column=False).to_dict()
            runnable = {"ok": True, "reason": None, "plan": plan}
        except sd.SpatialDomainsError as exc:
            runnable = {"ok": False, "reason": str(exc), "plan": None}
        positions = sd.has_spatial(adata)
        return {
            "runnable": runnable, "defaults": _defaults(), "n_obs": int(adata.n_obs),
            "free_column": sd.free_column(adata, sd.DOMAIN_COLUMN, k),
            "free_suggested_column": sd.free_suggested_column(adata),
            "free_niche_column": sd.free_column(adata, sd.NICHE_COLUMN, sd.DEFAULT_NICHES),
            "niches_allowed": sd.niches_allowed(adata) if positions else {"ok": False, "reason": None, "note": None},
            "estimates": {"enrichment_seconds": sd.enrichment_seconds(
                              int(adata.n_obs), grid=sd.grid_neighbours(adata) if positions else None),
                          "enrichment_bytes": sd.enrichment_bytes(int(adata.n_obs))},
            "domains": sd.domains_summary(adata, fp),
            "niches": sd.niches_summary(adata, fp),
            "enrichment": sd.enrichment_summary(adata, enrichment_column or None, fp),
        }

    out = await asyncio.to_thread(work)
    out["running"] = dataset_id in _running
    return out


async def _save(dm: DatasetManager, dataset_id: str, fingerprint: str, what: str,
                write: Callable[[Any], dict[str, Any]]) -> dict[str, Any]:
    """Store a result in the dataset under its edit lock and save it (a large dataset's
    obs and uns written into its h5ad in place, X untouched)."""
    async with dataset_edit(dataset_id) as held:
        adaptor = await _adaptor(dm, dataset_id)
        if adaptor.cells_fingerprint() != fingerprint:
            raise sd.SpatialDomainsError(f"The dataset's cells changed while {what} ran (a pipeline run or a filter); "
                                         "nothing was saved. Run it again.")
        output_path = str(dm.derived_h5ad_path(dataset_id, adaptor.h5ad_path))
        backed = await edit_room(dm, dataset_id, adaptor)

        def work() -> dict[str, Any]:
            data = editable(adaptor)
            before = uns_snapshot(data)
            out = write(data)
            save_edit(adaptor, data, output_path, before)
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


def _stream(dataset_id: str, dm: DatasetManager, what: str, start: dict[str, Any], fingerprint: str,
            make: Callable[[Callable[[], bool]], Iterator[dict[str, Any]]],
            write: Callable[[Any, Any, dict[str, Any]], dict[str, Any]]) -> StreamingResponse:
    """The run as SSE: ``start``, ``stage`` / ``scan`` / ``progress`` / ``curve`` as they come,
    ``watchdog`` memory warnings, ``result`` (the saved summary), ``complete``, or ``error``."""

    async def events():
        if dataset_id in _running:  # another request started between the check and here
            yield _sse("error", {"error": "A spatial domains or niches run is already working on this dataset."})
            return
        _running.add(dataset_id)
        stop = threading.Event()
        watchdog = RunWatchdog()
        control = RunControl(watchdog=watchdog, before_step=None)
        sent = 0
        gen = make(stop.is_set)
        pending: asyncio.Future | None = None

        def step_next():
            with control.step("spatial_domains"):
                return next(gen, _DONE)

        def finish(fut: asyncio.Future | None = None) -> None:
            if fut is not None and not fut.cancelled():
                fut.exception()  # retrieved: a stopped run is not an error
            gen.close()
            watchdog.stop()
            _running.discard(dataset_id)

        try:
            watchdog.start()
            yield _sse("start", start)
            while True:
                # Each piece of work runs in a worker thread. Shielded, so a client going away
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
                    saved = await _save(dm, dataset_id, fingerprint, what, lambda data, r=res, rs=resources: write(
                        data, r, {"peak_gb": rs["peak_gb"], "limit_gb": rs["limit_gb"], "seconds": round(r.seconds, 1)}))
                    yield _sse("result", {**saved, "resources": resources})
                else:
                    yield _sse(item["type"], {k: v for k, v in item.items() if k != "type"})
            yield _sse("complete", {})
        except sd.SpatialDomainsCancelled:
            return
        except sd.SpatialDomainsError as e:
            yield _sse("error", {"error": str(e)})
        except Exception as e:  # noqa: BLE001 — surface to the client, keep the server up
            logger.error("%s failed for %s: %s", what, dataset_id, e, exc_info=True)
            detail = e.detail if isinstance(e, HTTPException) else str(e)
            yield _sse("error", {"error": f"{what} failed: {detail}"})
        finally:
            stop.set()  # a thread mid-piece stops before the next one
            control.cancel()
            if pending is not None and not pending.done():
                pending.add_done_callback(finish)
            else:
                finish()

    return StreamingResponse(
        keepalive(events()),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _busy(dataset_id: str) -> None:
    if dataset_id in _running:
        raise HTTPException(status_code=409, detail="A spatial domains or niches run is already working on this dataset.")


@router.post("/datasets/{dataset_id}/spatial/domains/run-stream")
async def run_domains(dataset_id: str, body: DomainsRequest, dm: DatasetManager = Depends(get_dataset_manager)):
    """Find domains at ``k``, or (``k`` null) scan the stability of each k and use its best
    peak. An earlier run's column is replaced only with ``overwrite``."""
    adaptor = await _adaptor(dm, dataset_id)
    _busy(dataset_id)
    adata = adaptor.adata
    try:
        plan = await asyncio.to_thread(sd.plan_domains, adata, **body.model_dump())
        inp = await asyncio.to_thread(sd.prepare_domains, adata, plan)
    except sd.SpatialDomainsError as exc:
        raise _http(exc)
    fingerprint = await asyncio.to_thread(adaptor.cells_fingerprint)
    await dm.reserve(dataset_id, plan.estimate_bytes, what="Spatial domains")
    del adata, adaptor  # the run needs only its copies
    start = {**plan.to_dict(), "stages": [{"stage": s, "label": x} for s, x in sd.DOMAIN_STAGES
                                          if plan.k is None or s != "scan"]}
    return _stream(dataset_id, dm, "Finding spatial domains", start, fingerprint,
                   lambda stop: sd.iter_domains(inp, stop),
                   lambda data, res, resources: sd.save_domains(data, inp, res, resources=resources,
                                                                cells_fingerprint=fingerprint))


@router.post("/datasets/{dataset_id}/spatial/nhood-enrichment/run-stream")
async def run_enrichment(dataset_id: str, body: EnrichmentRequest, dm: DatasetManager = Depends(get_dataset_manager)):
    """Neighbourhood enrichment of a label column (= squidpy's): which labels sit next to
    which more often than the same labels shuffled."""
    adaptor = await _adaptor(dm, dataset_id)
    _busy(dataset_id)
    adata = adaptor.adata
    try:
        plan = await asyncio.to_thread(sd.plan_enrichment, adata, body.column, body.n_perms, body.seed)
        inp = await asyncio.to_thread(sd.prepare_enrichment, adata, plan)
    except sd.SpatialDomainsError as exc:
        raise _http(exc)
    fingerprint = await asyncio.to_thread(adaptor.cells_fingerprint)
    await dm.reserve(dataset_id, plan.estimate_bytes, what="Neighbourhood enrichment")
    del adata, adaptor
    start = {**plan.to_dict(), "stages": [{"stage": s, "label": x} for s, x in sd.ENRICHMENT_STAGES]}
    return _stream(dataset_id, dm, "Neighbourhood enrichment", start, fingerprint,
                   lambda stop: sd.iter_enrichment(inp, stop),
                   lambda data, res, resources: sd.save_enrichment(data, inp, res, resources=resources,
                                                                   cells_fingerprint=fingerprint))


@router.post("/datasets/{dataset_id}/spatial/niches/run-stream")
async def run_niches(dataset_id: str, body: NichesRequest, dm: DatasetManager = Depends(get_dataset_manager)):
    """Composition niches from a label column. Refused on Visium spots and on Visium HD
    bins other than 8 um. An earlier run's column is replaced only with ``overwrite``."""
    adaptor = await _adaptor(dm, dataset_id)
    _busy(dataset_id)
    adata = adaptor.adata
    try:
        plan = await asyncio.to_thread(sd.plan_niches, adata, **body.model_dump())
        inp = await asyncio.to_thread(sd.prepare_niches, adata, plan)
    except sd.SpatialDomainsError as exc:
        raise _http(exc)
    fingerprint = await asyncio.to_thread(adaptor.cells_fingerprint)
    await dm.reserve(dataset_id, plan.estimate_bytes, what="Composition niches")
    del adata, adaptor
    start = {**plan.to_dict(), "stages": [{"stage": s, "label": x} for s, x in sd.NICHE_STAGES]}
    return _stream(dataset_id, dm, "Composition niches", start, fingerprint,
                   lambda stop: sd.iter_niches(inp, stop),
                   lambda data, res, resources: sd.save_niches(data, inp, res, resources=resources,
                                                               cells_fingerprint=fingerprint))
