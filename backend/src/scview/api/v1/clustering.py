"""Clustering resolution sweep (``core/resolution_sweep.py``) and the
variable-gene sensitivity check (``core/hvg_sensitivity.py``).

``GET /clustering/sweep`` says whether a sweep can run and with what defaults;
``POST /clustering/sweep`` streams one SSE ``result`` per resolution without
changing the dataset; ``POST /clustering/keep`` saves one resolution's labels.
``/clustering/hvg-sensitivity`` (GET, POST, POST ``/keep``) does the same for
re-clustering under other variable-gene settings. ``GET /clustering/distances``
gives the distances between a column's groups in the graph's PC space
(``core/cluster_distances.py``).
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from scview.core import cluster_distances as cd
from scview.core import hvg_sensitivity as hs
from scview.core import resolution_sweep as rs
from scview.core import subcluster as sc_sub
from scview.core.confounding import recorded_batch_key
from scview.core.dataset_manager import DatasetManager
from scview.core.obs_writes import dataset_edit, uns_snapshot, write_derived
from scview.dependencies import get_dataset_manager

logger = logging.getLogger(__name__)

router = APIRouter()

_DONE = object()
_running: set[str] = set()  # datasets with a sweep streaming: one at a time each


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


class SweepRequest(BaseModel):
    resolutions: list[float] | None = None  # default rs.DEFAULT_RESOLUTIONS
    method: str | None = None  # default: the dataset's clustering method


class KeepRequest(BaseModel):
    resolution: float
    method: str | None = None


class SubclusterRequest(BaseModel):
    column: str
    group: str
    resolution: float = sc_sub.DEFAULT_RESOLUTION


class HvgSetting(BaseModel):
    flavor: str
    n_top_genes: int


class HvgSensitivityRequest(BaseModel):
    settings: list[HvgSetting] | None = None  # default hs.default_settings
    reference: str | None = None  # default: the active clustering


class HvgKeepRequest(BaseModel):
    flavor: str
    n_top_genes: int
    reference: str | None = None


def _method(adata, requested: str | None) -> str:
    method = requested or rs.dataset_method(adata)
    if method not in rs.METHODS:
        raise HTTPException(status_code=400, detail=f"Unknown clustering method '{method}'.")
    return method


async def _adata(dm: DatasetManager, dataset_id: str):
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    return adaptor


@router.get("/datasets/{dataset_id}/clustering/sweep")
async def sweep_info(
    dataset_id: str,
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """Whether a sweep can run on this dataset (and if not, why), and its limits."""
    adata = (await _adata(dm, dataset_id)).adata
    try:
        rs.check(adata)
        reason = None
    except rs.SweepError as exc:
        reason = str(exc)
    return {
        "available": reason is None,
        "reason": reason,
        "method": rs.dataset_method(adata),
        "n_cells": int(adata.n_obs),
        "default_resolutions": list(rs.DEFAULT_RESOLUTIONS),
        "max_resolutions": rs.MAX_RESOLUTIONS,
        "min_resolution": rs.MIN_RESOLUTION,
        "max_resolution": rs.MAX_RESOLUTION,
        "max_cells": rs.MAX_CELLS,
        "silhouette_cells": rs.SILHOUETTE_CELLS,
    }


@router.post("/datasets/{dataset_id}/clustering/sweep")
async def run_sweep(
    dataset_id: str,
    body: SweepRequest,
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """Cluster at each resolution and stream the scores, leaving the dataset as it is.

    Events: ``start`` (method, cells, the silhouette's representation and
    subsample), one ``result`` per resolution, then ``complete``, or ``error``.
    """
    adata = (await _adata(dm, dataset_id)).adata
    method = _method(adata, body.method)
    try:
        resolutions = rs.clean_resolutions(body.resolutions)
        rs.check(adata)
    except rs.SweepError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if dataset_id in _running:
        raise HTTPException(status_code=409, detail="A resolution sweep is already running on this dataset.")

    async def events():
        if dataset_id in _running:  # another request started between the check and here
            yield _sse("error", {"error": "A resolution sweep is already running on this dataset."})
            return
        _running.add(dataset_id)
        gen = rs.iter_sweep(adata, resolutions, method)
        try:
            event = "start"
            while True:
                # Each resolution clusters in a worker thread: the server stays responsive.
                item = await asyncio.to_thread(next, gen, _DONE)
                if item is _DONE:
                    break
                yield _sse(event, item)
                event = "result"
            yield _sse("complete", {})
        except Exception as e:  # noqa: BLE001 — surface to the client, keep the server up
            logger.error("Resolution sweep failed for %s: %s", dataset_id, e, exc_info=True)
            yield _sse("error", {"error": str(e)})
        finally:
            _running.discard(dataset_id)
            try:
                gen.close()
            except ValueError:  # client gone mid-resolution: the thread finishes it
                pass

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/datasets/{dataset_id}/clustering/keep")
async def keep_resolution(
    dataset_id: str,
    body: KeepRequest,
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """Save one resolution's clusters as the pipeline's ``scview_<method>_r<res>``
    column (unchanged when it exists), then reload the dataset."""
    async with dataset_edit(dataset_id) as held:
        adaptor = await _adata(dm, dataset_id)
        adata = adaptor.adata
        method = _method(adata, body.method)
        try:
            resolution = rs.clean_resolutions([body.resolution])[0]
            if rs.existing_column(adata, method, resolution) is not None:
                return rs.keep(adata, method, resolution)
            rs.check(adata)
        except rs.SweepError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        output_path = str(dm.derived_h5ad_path(dataset_id, adaptor.h5ad_path))

        backed = getattr(adata, "isbacked", False)

        def _run():
            work = adata.to_memory() if backed else adata
            before = uns_snapshot(work)
            result = rs.keep(work, method, resolution)
            write_derived(work, output_path, adaptor.h5ad_path, before)
            return result

        try:
            result = await held.run(_run)
        except Exception as exc:
            dm.evict(dataset_id)  # the cached copy may hold the column without it being saved
            logger.error("Keeping resolution %s failed for %s: %s", resolution, dataset_id, exc,
                         exc_info=True)
            raise HTTPException(status_code=500, detail=f"Could not save the clustering: {exc!s}")
        if backed:
            dm.evict(dataset_id)
            await dm.load_dataset(dataset_id)
        else:
            # The cached dataset is the edited one and now matches the file.
            dm.saved_in_place(dataset_id, adaptor, output_path)
        return result


@router.post("/datasets/{dataset_id}/clustering/subcluster")
async def subcluster_group(
    dataset_id: str,
    body: SubclusterRequest,
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """Split one group of a column into sub-clusters on its own graph, saved as
    ``<column>_sub`` (core/subcluster.py)."""
    async with dataset_edit(dataset_id) as held:
        adaptor = await _adata(dm, dataset_id)
        output_path = str(dm.derived_h5ad_path(dataset_id, adaptor.h5ad_path))
        backed = getattr(adaptor.adata, "isbacked", False)

        def _run():
            work = adaptor.adata.to_memory() if backed else adaptor.adata
            before = uns_snapshot(work)
            result = sc_sub.subcluster(work, body.column, body.group, body.resolution,
                                       batch_key=recorded_batch_key(work))
            write_derived(work, output_path, adaptor.h5ad_path, before)
            return result

        try:
            # HVG + PCA + graph + Leiden on the group: seconds, so in a worker thread.
            result = await held.run(_run)
        except sc_sub.SubclusterError as exc:
            raise HTTPException(status_code=400, detail=str(exc))  # raised before any change
        except Exception as exc:
            dm.evict(dataset_id)  # the cached copy may hold a column that was never saved
            logger.error("Sub-clustering failed for %s: %s", dataset_id, exc, exc_info=True)
            raise HTTPException(status_code=500, detail=f"Sub-clustering failed: {exc!s}")
        if backed:
            dm.evict(dataset_id)
            await dm.load_dataset(dataset_id)
        else:
            dm.saved_in_place(dataset_id, adaptor, output_path)
        return result


# ---------------------------------------------------------------------------
# Variable-gene sensitivity check (core/hvg_sensitivity.py)
# ---------------------------------------------------------------------------

_hvg_running: set[str] = set()  # datasets with a check running: one at a time each


def _hvg_limits() -> dict[str, Any]:
    return {
        "max_settings": hs.MAX_SETTINGS,
        "min_genes": hs.MIN_GENES,
        "max_genes": hs.MAX_GENES,
        "max_cells": hs.MAX_CELLS,
        "first_run_seconds": hs.FIRST_RUN_SECONDS,
    }


@router.get("/datasets/{dataset_id}/clustering/hvg-sensitivity")
async def hvg_sensitivity_info(
    dataset_id: str,
    reference: str | None = Query(None),
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """Whether the check can run (and if not, why), what the dataset's clustering
    was made with, the default settings and their time estimate."""
    adata = (await _adata(dm, dataset_id)).adata

    def _info() -> dict[str, Any]:
        base = {"n_cells": int(adata.n_obs), "n_genes": int(adata.n_vars), **_hvg_limits()}
        try:
            hs.check(adata)
            rec = hs.recorded(adata, reference)
        except hs.SensitivityError as exc:
            return {"available": False, "reason": str(exc), **base}
        settings = hs.default_settings(adata, rec)
        return {
            "available": True,
            "reason": None,
            **base,
            "recorded": rec.to_dict(),
            "current": rec.current.to_dict(),
            "default_settings": [s.to_dict() for s in settings],
            "flavors": [{"flavor": f, "label": hs.FLAVOR_LABELS[f], "unavailable": why}
                        for f, why in hs.flavors_available(adata).items()],
            "estimate_seconds": hs.estimate_seconds(adata.n_obs, len(settings), rec),
            "seconds_per_setting": hs.estimate_seconds(adata.n_obs, 1, rec),
        }

    # Finding the counts can copy a matrix (counts held in .raw): off the loop.
    return await asyncio.to_thread(_info)


@router.post("/datasets/{dataset_id}/clustering/hvg-sensitivity")
async def run_hvg_sensitivity(
    dataset_id: str,
    body: HvgSensitivityRequest,
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """Re-cluster under each variable-gene setting and stream the comparison with
    the current clustering, leaving the dataset as it is.

    Events: ``start`` (the recorded settings, the time estimate), one ``result``
    per setting, then ``complete``, or ``error``. A client that goes away stops
    the run before its next sub-step.
    """
    adata = (await _adata(dm, dataset_id)).adata
    requested = [s.model_dump() for s in body.settings] if body.settings else None

    def _plan():
        hs.check(adata)
        rec = hs.recorded(adata, body.reference)
        return rec, hs.clean_settings(adata, rec, requested)

    try:
        rec, settings = await asyncio.to_thread(_plan)
    except hs.SensitivityError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if dataset_id in _hvg_running:
        raise HTTPException(status_code=409, detail="A variable-gene check is already running on this dataset.")

    async def events():
        if dataset_id in _hvg_running:  # another request started between the check and here
            yield _sse("error", {"error": "A variable-gene check is already running on this dataset."})
            return
        _hvg_running.add(dataset_id)
        stop = threading.Event()
        gen = hs.iter_sensitivity(adata, settings, rec, stop=stop.is_set)
        pending: asyncio.Future | None = None

        def finish(fut: asyncio.Future | None = None) -> None:
            if fut is not None and not fut.cancelled():
                fut.exception()  # retrieved: a cancelled run is not an error
            gen.close()
            _hvg_running.discard(dataset_id)

        try:
            event = "start"
            while True:
                # Each setting runs in a worker thread: the server stays responsive.
                # Shielded, so a client going away cancels this wait, not the
                # thread's future, whose callback then releases the dataset.
                pending = asyncio.ensure_future(asyncio.to_thread(next, gen, _DONE))
                item = await asyncio.shield(pending)
                pending = None
                if item is _DONE:
                    break
                yield _sse(event, item)
                event = "result"
            yield _sse("complete", {})
        except Exception as e:  # noqa: BLE001 — surface to the client, keep the server up
            logger.error("Variable-gene check failed for %s: %s", dataset_id, e, exc_info=True)
            yield _sse("error", {"error": str(e)})
        finally:
            stop.set()  # a thread mid-setting stops at its next sub-step
            if pending is not None and not pending.done():
                pending.add_done_callback(finish)
            else:
                finish()

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/datasets/{dataset_id}/clustering/hvg-sensitivity/keep")
async def keep_hvg_setting(
    dataset_id: str,
    body: HvgKeepRequest,
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """Save one setting's clusters as ``scview_<method>_r<res>_hvg_<flavor>_<n>``
    (unchanged when it exists): the labels the check showed, while still cached."""
    async with dataset_edit(dataset_id) as held:
        adaptor = await _adata(dm, dataset_id)
        adata = adaptor.adata
        setting_in = [{"flavor": body.flavor, "n_top_genes": body.n_top_genes}]

        def _plan():
            hs.check(adata)
            rec = hs.recorded(adata, body.reference)
            return rec, hs.clean_settings(adata, rec, setting_in)[0]

        try:
            rec, setting = await held.run(_plan)
        except hs.SensitivityError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        column = hs.column_name(rec, setting)
        if column in adata.obs.columns:
            return {"column": column, "created": False, "n_clusters": int(adata.obs[column].nunique())}
        output_path = str(dm.derived_h5ad_path(dataset_id, adaptor.h5ad_path))

        def _run():
            before = uns_snapshot(adata)
            result = hs.keep(adata, setting, rec)
            write_derived(adata, output_path, adaptor.h5ad_path, before)
            return result

        try:
            # A cache miss re-runs the setting (seconds to a minute): in a worker thread.
            result = await held.run(_run)
        except Exception as exc:
            dm.evict(dataset_id)  # the cached copy may hold the column without it being saved
            logger.error("Keeping variable-gene setting %s failed for %s: %s", setting.key, dataset_id,
                         exc, exc_info=True)
            raise HTTPException(status_code=500, detail=f"Could not save the clustering: {exc!s}")
        dm.saved_in_place(dataset_id, adaptor, output_path)
        return result


@router.get("/datasets/{dataset_id}/clustering/distances")
async def group_distances(
    dataset_id: str,
    column: str = Query(..., min_length=1),
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """Distances between the centroids of ``column``'s groups in the PCs the
    neighbour graph was built on (Harmony PCs when it used them; ``representation``
    says which), with the groups' order under complete-linkage clustering, the
    clustering's tree (scipy linkage rows) and the centroids' Pearson correlation."""
    adata = (await _adata(dm, dataset_id)).adata
    if column not in adata.obs.columns:
        raise HTTPException(status_code=404, detail=f"Column '{column}' not found.")
    try:
        # One pass over every cell's PCs: off the loop.
        return await asyncio.to_thread(cd.group_distances, adata, column)
    except cd.DistanceError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
