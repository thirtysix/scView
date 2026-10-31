"""Reference mapping / label transfer endpoints (``core/label_transfer.py``).

``GET /datasets/{id}/label-transfer`` gives the methods, defaults, the saved
mappings and whether a run is working on the dataset; ``GET
.../label-transfer/reference/{ref}`` reads a candidate reference's label columns,
size and species from its file (obs and var only) with a memory and time estimate;
``POST .../label-transfer/run-stream`` maps the dataset onto the reference and
streams it (SSE), then saves the label, score and "not in reference" columns.

The run copies what it needs out of both datasets first (the reference's labelled
cells and the query on their shared genes), so the reference can close meanwhile;
only the final save holds the query's edit lock. A client that goes away stops the
run after the step it is on, and nothing is saved.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
from typing import Any

import anndata as ad
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from scview.core import label_transfer as lt
from scview.core import large_mode, preflight, working_copy
from scview.core.anndata_adaptor import opens_backed
from scview.core.dataset_manager import DatasetManager
from scview.core.obs_writes import dataset_edit, uns_snapshot, write_derived
from scview.core.run_watchdog import RunWatchdog
from scview.core.sse import keepalive
from scview.dependencies import get_dataset_manager

logger = logging.getLogger(__name__)

router = APIRouter()

_DONE = object()
_running: set[str] = set()  # query datasets with a mapping streaming: one at a time each

_WHAT = "Reference mapping"
_WHY = "it reads every cell's values for the shared genes of both datasets into memory"


class RunRequest(BaseModel):
    reference_id: str
    label_column: str
    method: str = lt.METHOD_PROJECTION
    k: int = Field(default=lt.DEFAULT_K, ge=lt.MIN_K, le=lt.MAX_K)
    # Prefix of the new columns; the reference's name when left out
    name: str | None = Field(default=None, max_length=60)


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def _adaptor(dm: DatasetManager, dataset_id: str):
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    return adaptor


def _defaults() -> dict[str, Any]:
    return {"methods": list(lt.METHODS), "method": lt.METHOD_PROJECTION, "k": lt.DEFAULT_K,
            "min_k": lt.MIN_K, "max_k": lt.MAX_K, "n_hvg": lt.N_HVG, "n_pcs": lt.N_PCS,
            "percentile": lt.UNMAPPED_PERCENTILE, "min_shared_genes": lt.MIN_SHARED_GENES,
            "cluster_unmapped_share": lt.CLUSTER_UNMAPPED_SHARE, "note": lt.NOTE}


@router.get("/datasets/{dataset_id}/label-transfer")
async def get_label_transfer(dataset_id: str, dm: DatasetManager = Depends(get_dataset_manager)):
    """The methods and defaults, the saved mappings (newest last), and whether the
    dataset can be mapped (``reason`` when not) and a run is working on it now."""
    adaptor = await _adaptor(dm, dataset_id)
    adata = adaptor.adata
    reason = str(large_mode.LargeDatasetError(_WHAT, _WHY)) if large_mode.is_large(adata) else None
    return {**_defaults(), "saved": await asyncio.to_thread(lt.saved, adata),
            "available": reason is None, "reason": reason,
            "n_cells": int(adata.n_obs), "running": dataset_id in _running}


def _read_frames(path: str) -> tuple[Any, Any, bool]:
    """A file's obs and the var of its widest gene list (.raw's when larger), read
    without its matrices."""
    import h5py
    from anndata.io import read_elem

    with h5py.File(path, "r") as f:
        obs = read_elem(f["obs"])
        var = read_elem(f["var"])
        if "raw" in f and "var" in f["raw"]:
            raw_var = read_elem(f["raw"]["var"])
            if len(raw_var) > len(var):
                var = raw_var
        umap = "obsm" in f and "X_umap" in f["obsm"]
    return obs, var, umap


def _estimate(dm: DatasetManager, ref_path: str, ref_cached: bool, query_adaptor, method: str) -> dict:
    ref_h = preflight.read_header(ref_path, "h5ad")
    query_h = preflight.header_from_adata(query_adaptor.adata)
    need = preflight.label_transfer_need(ref_h, query_h, method=method, ref_open=ref_cached)
    secs = lt.estimate_seconds(ref_h.n_cells or 0, query_h.n_cells or 0, method)
    return {"peak_bytes": int(need), "seconds": secs, "seconds_range": [round(secs * 0.5, 1), round(secs * 2, 1)]}


@router.get("/datasets/{dataset_id}/label-transfer/reference/{reference_id}")
async def reference_info(
    dataset_id: str,
    reference_id: str,
    method: str = lt.METHOD_PROJECTION,
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """A candidate reference: its cells, label columns (categorical, 2-200 values),
    species against this dataset's, whether it has a UMAP, shared genes by name, and
    the run's memory and time estimate. Reads the reference's obs and var only."""
    from scview.core.species import detect_species

    if reference_id == dataset_id:
        raise HTTPException(status_code=400, detail="Pick another dataset as the reference.")
    paths = dm.h5ad_candidates(reference_id)
    if not paths:
        raise HTTPException(status_code=404, detail="Reference dataset not found.")
    query = await _adaptor(dm, dataset_id)
    cached = dm.get_dataset(reference_id)

    def work() -> dict[str, Any]:
        if large_mode.file_is_large(paths[0]):
            return {"ok": False, "reason": str(large_mode.LargeDatasetError(_WHAT, _WHY))}
        obs, var, umap = _read_frames(paths[0])
        skel = ad.AnnData(obs=obs, var=var)
        ref_species = detect_species(skel).species
        query_species = detect_species(query.adata).species
        qnames = set(map(str, (query.adata.raw.var_names if query.adata.raw is not None
                               and query.adata.raw.n_vars > query.adata.n_vars else query.adata.var_names)))
        shared = sum(1 for g in var.index.astype(str) if g in qnames)
        refusal = lt.check_species(ref_species, query_species)
        return {
            "ok": refusal is None, "reason": refusal,
            "n_cells": int(len(obs)), "n_genes": int(len(var)), "has_umap": bool(umap),
            "label_columns": lt.label_columns(skel),
            "species": ref_species, "dataset_species": query_species,
            "shared_genes_by_name": shared, "min_shared_genes": lt.MIN_SHARED_GENES,
            "estimate": _estimate(dm, paths[0], cached is not None, query, method),
        }

    try:
        return await asyncio.to_thread(work)
    except Exception as e:  # noqa: BLE001 - an unreadable file is reported, not a 500
        logger.warning("Could not read reference %s: %s", reference_id, e, exc_info=True)
        raise HTTPException(status_code=422, detail=f"Could not read the reference: {e}")


def _expression(fallback: list[str]):
    """A dataset's log-normalised matrix, its var and where it came from: counts from the
    dataset or its source files (``integration.recover_counts``), else log-normalised
    values as they are."""
    def get(adata):
        return lt.expression_of(adata, fallback)
    return get


async def _save(dm: DatasetManager, dataset_id: str, p: lt.Prepared, out: dict, summary: dict,
                resources: dict, fingerprint: str) -> dict[str, Any]:
    """Write the columns into the query under its edit lock and save it."""
    async with dataset_edit(dataset_id) as held:
        adaptor = await _adaptor(dm, dataset_id)
        if adaptor.cells_fingerprint() != fingerprint:
            raise lt.LabelTransferError("This dataset's cells changed while the mapping ran (a pipeline "
                                        "run or a filter); nothing was saved. Run it again.", 409)
        output_path = str(dm.derived_h5ad_path(dataset_id, adaptor.h5ad_path))
        backed = await working_copy.make_room(dm, dataset_id, adaptor)
        emb = lt.columns_for(summary["prefix"])["embedding"]

        def work() -> dict[str, Any]:
            data = working_copy.in_memory(adaptor)
            before = uns_snapshot(data)
            result = lt.save(data, p, out, summary, resources=resources)
            write_derived(data, output_path, adaptor.h5ad_path, before, obsm_keys=(emb,))
            return result

        try:
            result = await held.run(work)
        except Exception:
            dm.evict(dataset_id)  # the cached copy may hold columns that were never saved
            raise
        if backed:
            dm.evict(dataset_id)
            await dm.load_dataset(dataset_id)
        else:
            dm.saved_in_place(dataset_id, adaptor, output_path)
        return result


@router.post("/datasets/{dataset_id}/label-transfer/run-stream")
async def run_label_transfer(
    dataset_id: str,
    body: RunRequest,
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """Map the dataset onto the reference and stream it. Events: ``start`` (sizes,
    genes, the estimate), ``progress`` per step, ``watchdog`` memory warnings,
    ``result`` (the saved summary), then ``complete``, or ``error``. Saved only
    when the run reaches its end."""
    if body.method not in lt.METHODS:
        raise HTTPException(status_code=400, detail=f"Unknown method '{body.method}'.")
    if body.reference_id == dataset_id:
        raise HTTPException(status_code=400, detail="Pick another dataset as the reference.")
    if dataset_id in _running:
        raise HTTPException(status_code=409, detail="A reference mapping is already running on this dataset.")
    paths = dm.h5ad_candidates(body.reference_id)
    if not paths:
        raise HTTPException(status_code=404, detail="Reference dataset not found.")
    if large_mode.file_is_large(paths[0]):
        raise large_mode.LargeDatasetError(f"{_WHAT} from this reference", _WHY)
    await dm.wait_until_idle(body.reference_id)  # a pipeline run on it finishes first
    adaptor = await _adaptor(dm, dataset_id)
    large_mode.check(adaptor, _WHAT, _WHY)
    ref_cached = dm.get_dataset(body.reference_id)
    est = await asyncio.to_thread(_estimate, dm, paths[0], ref_cached is not None, adaptor, body.method)
    # Room for the reference's copy and the run's matrices: other datasets closed first
    # (least recently used, never the two read here), a 507 when it cannot fit even then.
    # Held once the stream starts, until the run ends (``hold_read``).
    what = f"{_WHAT} on {dm.dataset_name(dataset_id)}"
    guard = dm.hold_read((dataset_id, body.reference_id), 0)
    try:
        await dm.reserve(dataset_id, est["peak_bytes"], what=what)
    finally:
        guard()
    info = dm.get_dataset_info(body.reference_id) or {}
    ref_name = str(info.get("name") or body.reference_id)
    prefix = lt.slug(body.name or ref_name)
    ref_info = {"id": body.reference_id, "name": ref_name, "path": paths[0],
                "mtime": round(os.path.getmtime(paths[0]), 3)}
    cluster_column = adaptor.active_clustering_column()

    def prepare(query_adaptor) -> lt.Prepared:
        cached = dm.get_dataset(body.reference_id)
        if cached is not None:
            ref, opened = cached.adata, None
        else:  # read directly: opening it in the cache could close this one
            # A file that opens backed stays on disk but for its counts, as the estimate says.
            big = opens_backed(os.path.getsize(paths[0]))
            ref = opened = ad.read_h5ad(paths[0], backed="r" if big else None)
        try:
            return lt.prepare(ref, query_adaptor.adata, label_column=body.label_column, ref_info=ref_info,
                              ref_expression=_expression(paths[1:]),
                              query_expression=_expression(dm.h5ad_candidates(dataset_id)[1:]),
                              cluster_column=cluster_column)
        finally:
            if opened is not None and opened.isbacked:
                opened.file.close()

    fingerprint = await asyncio.to_thread(adaptor.cells_fingerprint)
    del adaptor, ref_cached  # the run needs only its copies

    async def events():
        if dataset_id in _running:  # another request started between the check and here
            yield _sse("error", {"error": "A reference mapping is already running on this dataset."})
            return
        _running.add(dataset_id)
        release = dm.hold_read((dataset_id, body.reference_id), est["peak_bytes"], what)
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
            release()
            watchdog.stop()
            _running.discard(dataset_id)

        try:
            watchdog.start()
            yield _sse("start", {"reference": ref_name, "reference_id": body.reference_id,
                                 "label_column": body.label_column, "method": body.method, "k": body.k,
                                 "prefix": prefix, "estimate": est})
            yield _sse("progress", {"step": "read", "message": "Reading both datasets' values"})
            watchdog.step_started(lt.STEP)
            query_adaptor = await _adaptor(dm, dataset_id)
            pending = asyncio.ensure_future(asyncio.to_thread(prepare, query_adaptor))
            del query_adaptor  # the thread holds it while it copies; the run needs only the copies
            prepared = await asyncio.shield(pending)
            pending = None
            yield _sse("progress", {"step": "prepared", "message": (
                f"{prepared.ref_log.shape[0]:,} reference cells, {prepared.query_log.shape[0]:,} cells here, "
                f"{len(prepared.genes):,} shared genes"), "n_shared_genes": len(prepared.genes)})
            gen = lt.iter_transfer(prepared, method=body.method, k=body.k, stop=stop.is_set)
            while True:
                # Each step runs in a worker thread. Shielded, so a client going away
                # cancels this wait, not the thread's future, whose callback then
                # closes the run.
                pending = asyncio.ensure_future(asyncio.to_thread(next, gen, _DONE))
                item = await asyncio.shield(pending)
                pending = None
                for warning in watchdog.events[sent:]:
                    yield _sse("watchdog", warning)
                sent = len(watchdog.events)
                if item is _DONE:
                    break
                if item["type"] == "mapped":
                    watchdog.step_finished(lt.STEP)
                    res = watchdog.summary()
                    out = item["out"]
                    summary = lt.summarize(prepared, out, method=body.method, prefix=prefix)
                    resources = {"peak_gb": res["peak_gb"], "limit_gb": res["limit_gb"],
                                 "seconds": out["seconds"]}
                    result = await _save(dm, dataset_id, prepared, out, summary, resources, fingerprint)
                    yield _sse("result", {**result, "resources": res})
                else:
                    yield _sse(item["type"], {k_: v for k_, v in item.items() if k_ != "type"})
            yield _sse("complete", {})
        except lt.LabelTransferCancelled:
            return
        except lt.LabelTransferError as e:
            yield _sse("error", {"error": str(e)})
        except Exception as e:  # noqa: BLE001 — surface to the client, keep the server up
            logger.error("Reference mapping failed for %s: %s", dataset_id, e, exc_info=True)
            detail = e.detail if isinstance(e, HTTPException) else str(e)
            yield _sse("error", {"error": f"Reference mapping failed: {detail}"})
        finally:
            stop.set()  # a thread mid-step stops before the next one
            if pending is not None and not pending.done():
                pending.add_done_callback(finish)
            else:
                finish()

    return StreamingResponse(
        keepalive(events()),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
