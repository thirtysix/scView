"""Gene programs: consensus NMF of the variable genes (core/gene_programs.py).

``GET /programs`` says whether a run can start (with the defaults) and gives the
saved programs, with each group's mean usage for ``groupby``;
``POST /programs/run-stream`` runs NMF and streams it (SSE), then saves the
programs into the dataset; ``GET /programs/{p}/usage`` gives one program's
per-cell usage for the scatter (and per group for the violin);
``POST /programs/{p}/enrichment`` tests the program's top genes for
over-representation (MSigDB Hallmark by default) against the genes NMF used.

The factorisation runs on its own copy of the variable-gene matrix (tens of MB),
so the dataset stays open to other requests meanwhile; only the final save
holds the dataset's edit lock. A client that goes away stops the run after the
replicate it is working on, and nothing is saved.
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

from scview.config import Settings
from scview.core import gene_programs as gp
from scview.core.dataset_manager import DatasetManager
from scview.core.obs_writes import dataset_edit, uns_snapshot, write_derived
from scview.core.run_watchdog import RunWatchdog
from scview.core.sse import keepalive
from scview.dependencies import get_dataset_manager, get_settings_dep

logger = logging.getLogger(__name__)

router = APIRouter()

_DONE = object()
_running: set[str] = set()  # datasets with a programs run streaming: one at a time each

HALLMARK = "h.all"


class RunRequest(BaseModel):
    k: int | None = None  # default gp.DEFAULT_K
    seeds: int | None = None  # replicates; default gp.DEFAULT_SEEDS
    n_hvg: int | None = Field(default=None, ge=gp.MIN_GENES)  # default: every flagged gene
    sweep: list[int] | None = None  # other values of k to compare (stability, error)


class EnrichRequest(BaseModel):
    collections: list[str] = [HALLMARK]
    n_genes: int = Field(default=gp.N_TOP, ge=5, le=gp.N_TOP)
    # Ribosomal / mitochondrial genes out of the query, as in the other ORA requests.
    exclude_ribo_mito: bool = True
    max_terms: int = Field(default=50, ge=1, le=1000)


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def _adaptor(dm: DatasetManager, dataset_id: str):
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    return adaptor


def _defaults() -> dict[str, Any]:
    return {"k": gp.DEFAULT_K, "seeds": gp.DEFAULT_SEEDS, "sweep": list(gp.DEFAULT_SWEEP),
            "min_k": gp.MIN_K, "max_k": gp.MAX_K, "max_seeds": gp.MAX_SEEDS, "max_sweep": gp.MAX_SWEEP,
            "min_genes": gp.MIN_GENES, "n_top": gp.N_TOP, "stable_correlation": gp.STABLE_CORR}


@router.get("/datasets/{dataset_id}/programs")
async def get_programs(
    dataset_id: str,
    groupby: str = Query(default="", description="Mean usage of each program per group of this column"),
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """The saved programs (``available``; per program its stability, usage share and
    top genes; the sweep, if one ran), ``runnable`` (whether a run can start and on
    what input), the defaults, and whether a run is working on the dataset now."""
    adata = (await _adaptor(dm, dataset_id)).adata

    def work() -> dict[str, Any]:
        return {**gp.summary(adata, groupby or None), "runnable": gp.runnable(adata),
                "defaults": _defaults(), "n_cells_total": int(adata.n_obs)}

    out = await asyncio.to_thread(work)
    out["running"] = dataset_id in _running
    return out


async def _save(dm: DatasetManager, dataset_id: str, fit: gp.KFit, inp: gp.ProgramInput,
                n_hvg: int | None, sweep: list[dict], resources: dict, fingerprint: str) -> dict[str, Any]:
    """Store the programs in the dataset under its edit lock and save it."""
    async with dataset_edit(dataset_id) as held:
        adaptor = await _adaptor(dm, dataset_id)
        if adaptor.cells_fingerprint() != fingerprint:
            raise gp.ProgramsError("The dataset's cells changed while the programs ran (a pipeline run "
                                   "or a filter); nothing was saved. Run them again.")
        output_path = str(dm.derived_h5ad_path(dataset_id, adaptor.h5ad_path))
        backed = getattr(adaptor.adata, "isbacked", False)

        def work() -> dict[str, Any]:
            data = adaptor.adata.to_memory() if backed else adaptor.adata
            before = uns_snapshot(data)
            out = gp.save(data, fit, inp, n_hvg=n_hvg, sweep=sweep, resources=resources,
                          fingerprint=fingerprint)
            write_derived(data, output_path, adaptor.h5ad_path, before,
                          obsm_keys=(gp.OBSM_KEY, gp.OBSM_RAW_KEY))
            return out

        try:
            result = await held.run(work)
        except Exception:
            dm.evict(dataset_id)  # the cached copy may hold programs that were never saved
            raise
        if backed:
            dm.evict(dataset_id)
            await dm.load_dataset(dataset_id)
        else:
            dm.saved_in_place(dataset_id, adaptor, output_path)
        return result


@router.post("/datasets/{dataset_id}/programs/run-stream")
async def run_programs(
    dataset_id: str,
    body: RunRequest,
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """Run consensus NMF and stream it. Events: ``start`` (the input and a time
    estimate), ``progress`` after each replicate, ``k_result`` after each k (its
    silhouette, mean correlation, stable count, relative error), ``watchdog``
    memory warnings, ``result`` (the saved programs), then ``complete``, or
    ``error``. Saved only when the run reaches its end."""
    adaptor = await _adaptor(dm, dataset_id)
    try:
        k, seeds, ks = gp.clean_request(body.k, body.seeds, body.sweep)
    except gp.ProgramsError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if dataset_id in _running:
        raise HTTPException(status_code=409, detail="Gene programs are already running on this dataset.")
    adata = adaptor.adata
    try:
        # Copies the variable genes out of the matrix (row chunks for a dense one).
        inp = await asyncio.to_thread(gp.prepare_input, adata, body.n_hvg)
    except gp.ProgramsError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    fingerprint = await asyncio.to_thread(adaptor.cells_fingerprint)
    del adata, adaptor  # the run needs only its copy

    async def events():
        if dataset_id in _running:  # another request started between the check and here
            yield _sse("error", {"error": "Gene programs are already running on this dataset."})
            return
        _running.add(dataset_id)
        stop = threading.Event()
        watchdog = RunWatchdog()
        sent = 0
        gen = gp.iter_programs(inp, k, seeds, ks, stop=stop.is_set)
        pending: asyncio.Future | None = None

        def finish(fut: asyncio.Future | None = None) -> None:
            if fut is not None and not fut.cancelled():
                fut.exception()  # retrieved: a stopped run is not an error
            gen.close()
            watchdog.stop()
            _running.discard(dataset_id)

        try:
            watchdog.start()
            yield _sse("start", {
                "k": k, "seeds": seeds, "ks": ks, "n_cells": int(inp.X.shape[0]), "n_genes": len(inp.genes),
                "n_flagged": inp.n_flagged, "source": inp.source, "notes": inp.notes,
                "estimate_seconds": gp.estimate_seconds(inp.X, ks, len(seeds)),
            })
            watchdog.step_started("gene_programs")
            while True:
                # Each replicate runs in a worker thread. Shielded, so a client going
                # away cancels this wait, not the thread's future, whose callback then
                # closes the run.
                pending = asyncio.ensure_future(asyncio.to_thread(next, gen, _DONE))
                item = await asyncio.shield(pending)
                pending = None
                for warning in watchdog.events[sent:]:
                    yield _sse("watchdog", warning)
                sent = len(watchdog.events)
                if item is _DONE:
                    break
                if item["type"] == "fit":
                    watchdog.step_finished("gene_programs")
                    resources = watchdog.summary()
                    result = await _save(dm, dataset_id, item["fit"], inp, body.n_hvg, item["sweep"],
                                         {"peak_gb": resources["peak_gb"], "limit_gb": resources["limit_gb"],
                                          "seconds": round(item["fit"].seconds, 1)}, fingerprint)
                    yield _sse("result", {**result, "resources": resources})
                else:
                    yield _sse(item["type"], {k_: v for k_, v in item.items() if k_ != "type"})
            yield _sse("complete", {})
        except gp.ProgramsCancelled:
            return
        except gp.ProgramsError as e:
            yield _sse("error", {"error": str(e)})
        except Exception as e:  # noqa: BLE001 — surface to the client, keep the server up
            logger.error("Gene programs failed for %s: %s", dataset_id, e, exc_info=True)
            detail = e.detail if isinstance(e, HTTPException) else str(e)
            yield _sse("error", {"error": f"Gene programs failed: {detail}"})
        finally:
            stop.set()  # a thread mid-replicate stops before the next one
            if pending is not None and not pending.done():
                pending.add_done_callback(finish)
            else:
                finish()

    return StreamingResponse(
        keepalive(events()),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/datasets/{dataset_id}/programs/{index}/usage")
async def get_program_usage(
    dataset_id: str,
    index: int,
    groupby: str = Query(default="", description="Also split the values by this column (violin)"),
    raw: bool = Query(default=False, description="Usages before scaling each cell's to sum 1"),
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """One program's per-cell usage share (``raw``: the unscaled usage), in the
    dataset's cell order, and split by ``groupby`` when given."""
    adata = (await _adaptor(dm, dataset_id)).adata

    def work() -> dict[str, Any]:
        values = gp.program_usage(adata, index, raw=raw)
        info = gp.stored(adata)
        prog = info["programs"][index - 1]
        out: dict[str, Any] = {
            "index": index, "name": prog["name"], "raw": raw, "run_id": info["run_id"],
            "top_genes": [g["gene"] for g in prog["top_genes"][:5]],
            "values": [round(float(v), 5) for v in values],
            "violin": None,
        }
        if groupby and groupby in adata.obs.columns:
            out["violin"] = gp.violin(adata, values, groupby) or None
        return out

    try:
        return await asyncio.to_thread(work)
    except gp.ProgramsError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.post("/datasets/{dataset_id}/programs/{index}/enrichment")
async def enrich_program(
    dataset_id: str,
    index: int,
    body: EnrichRequest,
    dm: DatasetManager = Depends(get_dataset_manager),
    settings: Settings = Depends(get_settings_dep),
):
    """Over-representation of the program's top genes (by gene score) in the chosen
    collections (MSigDB Hallmark by default), with the genes NMF used as the
    universe. The best terms are kept with the run for the co-pilot."""
    from scview.api.v1.enrichment import EnrichmentResult, _ora_records
    from scview.core.custom_genesets import get_custom_store, is_custom_id
    from scview.core.gene_filters import select_query_genes
    from scview.core.geneset_sources import label_records
    from scview.core.msigdb_loader import get_msigdb_loader
    from scview.core.species import detect_species, human_genesets_warning

    adaptor = await _adaptor(dm, dataset_id)
    adata = adaptor.adata
    info = gp.stored(adata)
    if info is None:
        raise HTTPException(status_code=404, detail="No gene programs for this dataset yet: run them first.")
    if not 1 <= index <= info["k"]:
        raise HTTPException(status_code=404, detail=f"There are {info['k']} programs; no P{index}.")
    loader = get_msigdb_loader(settings.MSIGDB_DIR)
    store = get_custom_store(settings.DATA_DIR)
    if loader is None and not any(is_custom_id(c) for c in body.collections):
        raise HTTPException(status_code=500, detail="MSigDB directory not configured.")
    run_id = info["run_id"]

    def work() -> dict[str, Any]:
        genes = gp.query_genes(info, index, body.n_genes)
        background = gp.background_genes(adata)
        kept = select_query_genes(genes, len(genes), exclude_ribo_mito=body.exclude_ribo_mito)
        records = _ora_records(kept, background, body.collections, loader, store) if kept else []
        return {
            "index": index, "name": info["programs"][index - 1]["name"], "run_id": run_id,
            "collections": list(body.collections), "query": kept, "n_query": len(kept),
            "n_excluded_ribo_mito": len(genes) - len(kept), "background_n": len(background),
            "n_terms": len(records),
            "n_significant": sum(r.get("adjusted_pvalue", 1.0) < 0.05 for r in records),
            "results": [EnrichmentResult(**r).model_dump()
                        for r in label_records(records[: body.max_terms], store)],
            "warning": human_genesets_warning(detect_species(adata)),
            "_records": records,
        }

    try:
        out = await asyncio.to_thread(work)
    except ImportError:
        raise HTTPException(status_code=500, detail="gseapy is required for enrichment analysis.")
    records = out.pop("_records")
    # Kept with the run (a small uns entry, rewritten in place), for the co-pilot.
    try:
        async with dataset_edit(dataset_id) as held:
            current = await _adaptor(dm, dataset_id)
            stored_now = gp.stored(current.adata)
            if stored_now is not None and stored_now["run_id"] == run_id:
                def keep() -> None:
                    gp.remember_enrichment(current.adata, index, body.collections, records)
                    dm.persist_uns(dataset_id, current, [gp.ENRICH_KEY])

                await held.run(keep)
    except Exception as e:  # the result is still returned; only the co-pilot misses it
        logger.warning("Could not keep the enrichment of P%s for %s: %s", index, dataset_id, e)
    return out
