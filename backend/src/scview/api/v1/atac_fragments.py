"""ATAC fragments endpoints (``core/atac_fragments.py``).

``POST /datasets/{id}/atac/fragments?name=``: one file as the raw request body (the fragments
file or its ``.tbi``, in either order, told apart by ``name``), streamed straight to the
dataset's modality folder and renamed there when whole: never spooled, never read twice.
``GET`` the same path: what is attached, the QC's estimate, the saved QC. ``DELETE`` removes
the files. ``POST .../atac/fragments/qc-stream``: the QC pass as SSE, one at a time per dataset;
a client that goes away stops it after its block of text, with nothing saved.
``GET .../atac/fragments/values`` and ``.../lengths``: per-cell values and length histograms
by group for the plots. ``GET /datasets/{id}/atac/coverage``: a track per group.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from collections.abc import Iterator
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from starlette.requests import ClientDisconnect

from scview.config import Settings
from scview.core import atac_fragments as af
from scview.core import fragment_file as ff
from scview.core import gene_annotation
from scview.core.dataset_manager import DatasetManager
from scview.core.obs_writes import dataset_edit, edit_room, editable, save_edit, uns_snapshot
from scview.core.run_watchdog import RunControl, RunWatchdog
from scview.core.sse import keepalive
from scview.dependencies import get_dataset_manager, get_settings_dep

logger = logging.getLogger(__name__)

router = APIRouter()

_DONE = object()
_running: set[str] = set()  # datasets with a QC pass streaming
_uploading: set[tuple[str, str]] = set()  # (dataset, role) being written
_WRITE_SIZE = 1024 * 1024
_WHAT = "The ATAC fragment QC"


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def _adaptor(dm: DatasetManager, dataset_id: str):
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")
    return adaptor


def _http(e: af.FragmentsError) -> HTTPException:
    return HTTPException(status_code=e.status, detail=str(e))


def _species(adata) -> str | None:
    from scview.core.species import detect_species

    call = detect_species(adata).species
    return None if call == "unknown" else call


def _cell_names(adata) -> list[str]:
    return list(map(str, adata.obs_names))


def _verify(settings: Settings, dataset_id: str, adata, file_name: str | None) -> dict[str, Any]:
    return af.verify(settings.DATA_DIR, dataset_id, _cell_names(adata), file_name=file_name,
                     species=_species(adata), auto_download=settings.ANNOTATION_AUTO_DOWNLOAD)


@router.get("/datasets/{dataset_id}/atac/fragments")
async def get_fragments(dataset_id: str, settings: Settings = Depends(get_settings_dep),
                        dm: DatasetManager = Depends(get_dataset_manager)):
    adata = (await _adaptor(dm, dataset_id)).adata

    def work() -> dict[str, Any]:
        return {**af.status(settings.DATA_DIR, dataset_id, int(adata.n_obs)), "qc": af.saved(adata),
                "n_cells": int(adata.n_obs), "max_window": af.MAX_WINDOW, "gene_pad": af.GENE_PAD,
                "min_barcode_share": af.MIN_BARCODE_SHARE}

    return {**await asyncio.to_thread(work), "running": dataset_id in _running}


@router.post("/datasets/{dataset_id}/atac/fragments")
async def upload_fragments(dataset_id: str, request: Request, name: str = Query(max_length=512),
                           settings: Settings = Depends(get_settings_dep),
                           dm: DatasetManager = Depends(get_dataset_manager)):
    """One file as the raw body: ``curl --data-binary @atac_fragments.tsv.gz ...?name=...``.
    Says what is still missing once it is in."""
    role = af.role_of(name)
    if role is None:
        raise HTTPException(status_code=422, detail=f"'{name}' is neither a fragments file (.tsv.gz) nor its "
                                                    "index (.tbi): send atac_fragments.tsv.gz and its .tbi.")
    if (request.headers.get("content-type") or "").startswith("multipart/"):
        raise HTTPException(status_code=415, detail="Send the file as the raw request body, not a multipart form.")
    adaptor = await _adaptor(dm, dataset_id)
    if dataset_id in _running:
        raise HTTPException(status_code=409, detail="A fragment QC is reading this dataset's file: wait for it, or "
                                                    "stop it first.")
    key = (dataset_id, role)
    if key in _uploading:
        raise HTTPException(status_code=409, detail=f"This dataset's {role} file is already being uploaded.")
    _uploading.add(key)
    target = af.final_path(settings.DATA_DIR, dataset_id, role)
    part = af.part_path(settings.DATA_DIR, dataset_id, role)
    try:
        await asyncio.to_thread(part.parent.mkdir, parents=True, exist_ok=True)
        await _stream_to(request, part, role, settings.max_upload_bytes, settings.MAX_UPLOAD_SIZE_MB)
        await asyncio.to_thread(part.replace, target)
    except af.FragmentsError as e:
        raise _http(e)
    except ClientDisconnect:
        raise HTTPException(status_code=400, detail="The upload stopped before the end: send the file again.")
    except OSError as e:
        logger.warning("Could not store %s for %s: %s", name, dataset_id, e)
        raise HTTPException(status_code=507, detail=f"Could not store {name} on this machine's disk: free some "
                                                    "space and send it again.")
    finally:
        part.unlink(missing_ok=True)
        _uploading.discard(key)
    adata = adaptor.adata
    try:
        await asyncio.to_thread(_verify, settings, dataset_id, adata, name if role == "fragments" else None)
    except af.FragmentsError as e:
        raise _http(e)
    return await asyncio.to_thread(af.status, settings.DATA_DIR, dataset_id, int(adata.n_obs))


async def _stream_to(request: Request, part, role: str, max_bytes: int, max_mb: int) -> None:
    """The body into ``part``, in writes of about 1 MB, off the event loop. The first bytes
    decide whether a fragments file is BGZF, before the rest travels."""
    total = 0
    pending: list[bytes] = []
    size = 0
    checked = False
    fh = await asyncio.to_thread(open, part, "wb")
    try:
        async for chunk in request.stream():
            if not chunk:
                continue
            total += len(chunk)
            if total > max_bytes:
                raise af.FragmentsError(f"That file is larger than the maximum upload size of {max_mb} MB.", 413)
            pending.append(chunk)
            size += len(chunk)
            if not checked and size >= 1024:
                af.check_head(b"".join(pending)[:1024], role)
                checked = True
            if size >= _WRITE_SIZE:
                await asyncio.to_thread(fh.write, b"".join(pending))
                pending, size = [], 0
        if not total:
            raise af.FragmentsError("That upload was empty: send the file as the request body.")
        if not checked:
            af.check_head(b"".join(pending)[:1024], role)
        if pending:
            await asyncio.to_thread(fh.write, b"".join(pending))
    finally:
        await asyncio.to_thread(fh.close)


@router.delete("/datasets/{dataset_id}/atac/fragments")
async def delete_fragments(dataset_id: str, settings: Settings = Depends(get_settings_dep)):
    if dataset_id in _running:
        raise HTTPException(status_code=409, detail="A fragment QC is reading this dataset's file: stop it first.")
    await asyncio.to_thread(af.detach, settings.DATA_DIR, dataset_id)
    return {"deleted": True}


async def _save_qc(dm: DatasetManager, dataset_id: str, fingerprint: str, res: af.QcResult, genome: str | None,
                   source: str | None, file_name: str | None, resources: dict) -> dict[str, Any]:
    async with dataset_edit(dataset_id) as held:
        adaptor = await _adaptor(dm, dataset_id)
        if adaptor.cells_fingerprint() != fingerprint:
            raise af.FragmentsError("The dataset's cells changed while the QC ran (a pipeline run or a filter); "
                                    "nothing was saved. Run it again.", 409)
        output_path = str(dm.derived_h5ad_path(dataset_id, adaptor.h5ad_path))
        backed = await edit_room(dm, dataset_id, adaptor)  # a large dataset: written in place

        def work() -> dict[str, Any]:
            data = editable(adaptor)
            before = uns_snapshot(data)
            out = af.save(data, res, genome=genome, source=source, file_name=file_name, resources=resources)
            save_edit(adaptor, data, output_path, before)
            return out

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


@router.post("/datasets/{dataset_id}/atac/fragments/qc-stream")
async def qc_stream(dataset_id: str, settings: Settings = Depends(get_settings_dep),
                    dm: DatasetManager = Depends(get_dataset_manager)):
    """One pass over the file. Events: ``start`` (the estimate), ``progress``, ``watchdog``,
    ``result`` (the saved QC), then ``complete``, or ``error``. Saved only at the end."""
    adaptor = await _adaptor(dm, dataset_id)
    if dataset_id in _running:
        raise HTTPException(status_code=409, detail="A fragment QC is already working on this dataset.")
    try:
        path, _ = af.require_files(settings.DATA_DIR, dataset_id)
    except af.FragmentsError as e:
        raise _http(e)
    state = await asyncio.to_thread(af.status, settings.DATA_DIR, dataset_id, int(adaptor.adata.n_obs))
    genome = state.get("genome")
    if not genome:
        raise HTTPException(status_code=422, detail="scView has gene positions for GRCh38 and mm10, and this file's "
                                                    "chromosome names match neither, so TSS enrichment cannot be "
                                                    "computed; coverage tracks still work.")
    try:
        tss = await asyncio.to_thread(af.tss_table, genome, settings.DATA_DIR, settings.ANNOTATION_AUTO_DOWNLOAD)
    except gene_annotation.AnnotationError as e:
        raise HTTPException(status_code=503, detail=str(e))
    est = state["estimate"]
    await dm.reserve(dataset_id, est["peak_bytes"], what=_WHAT)
    fingerprint = await asyncio.to_thread(adaptor.cells_fingerprint)
    cells = await asyncio.to_thread(_cell_names, adaptor.adata)
    del adaptor
    source = gene_annotation.source_of(genome)
    file_name = state.get("name")
    data_dir = settings.DATA_DIR

    async def events():
        if dataset_id in _running:  # another request started between the check and here
            yield _sse("error", {"error": "A fragment QC is already working on this dataset."})
            return
        _running.add(dataset_id)
        stop = threading.Event()
        watchdog = RunWatchdog()
        control = RunControl(watchdog=watchdog, before_step=None)
        gen: Iterator[dict[str, Any]] = af.iter_qc(path, cells, tss, stop=stop.is_set)
        sent = 0
        pending: asyncio.Future | None = None
        dm.reserve_run(dataset_id, est["peak_bytes"])

        def step_next():
            with control.step(af.STEP):
                return next(gen, _DONE)

        def finish(fut: asyncio.Future | None = None) -> None:
            if fut is not None and not fut.cancelled():
                fut.exception()  # retrieved: a stopped run is not an error
            gen.close()
            watchdog.stop()
            dm.release_run(dataset_id)
            _running.discard(dataset_id)

        try:
            watchdog.start()
            yield _sse("start", {"estimate": est, "n_cells": len(cells), "genome": genome})
            while True:
                # Shielded, so a client going away cancels this wait, not the thread's future,
                # whose callback then closes the run.
                pending = asyncio.ensure_future(asyncio.to_thread(step_next))
                item = await asyncio.shield(pending)
                pending = None
                for warning in watchdog.events[sent:]:
                    yield _sse("watchdog", warning)
                sent = len(watchdog.events)
                if item is _DONE:
                    break
                if item["type"] == "progress":
                    yield _sse("progress", {k: v for k, v in item.items() if k != "type"})
                    continue
                res: af.QcResult = item["result"]
                await asyncio.to_thread(af.write_lengths, res, data_dir, dataset_id)
                summary = watchdog.summary()
                saved = await _save_qc(dm, dataset_id, fingerprint, res, genome, source, file_name,
                                       {"peak_gb": summary["peak_gb"], "limit_gb": summary["limit_gb"],
                                        "seconds": round(res.seconds, 1)})
                yield _sse("result", {**saved, "resources": summary})
            yield _sse("complete", {})
        except af.FragmentsCancelled:
            return
        except (af.FragmentsError, ff.FragmentFileError) as e:
            yield _sse("error", {"error": str(e)})
        except Exception as e:  # noqa: BLE001 - surface to the client, keep the server up
            logger.error("Fragment QC failed for %s: %s", dataset_id, e, exc_info=True)
            detail = e.detail if isinstance(e, HTTPException) else str(e)
            yield _sse("error", {"error": f"The ATAC fragment QC failed: {detail}"})
        finally:
            stop.set()  # a thread mid-block stops before the next one
            control.cancel()
            if pending is not None and not pending.done():
                pending.add_done_callback(finish)
            else:
                finish()

    return StreamingResponse(keepalive(events()), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.get("/datasets/{dataset_id}/atac/fragments/values")
async def qc_values(dataset_id: str, group_by: str | None = Query(default=None, max_length=256),
                    dm: DatasetManager = Depends(get_dataset_manager)):
    """Per-cell QC values by group (at most 2,000 per group) for the violins."""
    adata = (await _adaptor(dm, dataset_id)).adata
    try:
        return await asyncio.to_thread(af.values_by_group, adata, group_by or None)
    except af.FragmentsError as e:
        raise _http(e)


@router.get("/datasets/{dataset_id}/atac/fragments/lengths")
async def qc_lengths(dataset_id: str, group_by: str | None = Query(default=None, max_length=256),
                     settings: Settings = Depends(get_settings_dep),
                     dm: DatasetManager = Depends(get_dataset_manager)):
    """Fragment lengths per group, from the per-cell store the QC wrote."""
    adata = (await _adaptor(dm, dataset_id)).adata
    try:
        return await asyncio.to_thread(af.lengths_by_group, adata, settings.DATA_DIR, dataset_id, group_by or None)
    except af.FragmentsError as e:
        raise _http(e)


@router.get("/datasets/{dataset_id}/atac/coverage")
async def coverage(
    dataset_id: str,
    region: str | None = Query(default=None, max_length=128),
    gene: str | None = Query(default=None, max_length=128),
    group_by: str | None = Query(default=None, max_length=256),
    groups: list[str] | None = Query(default=None),
    bins: int = Query(default=af.DEFAULT_BINS, ge=10, le=af.MAX_BINS),
    settings: Settings = Depends(get_settings_dep),
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """Per-group normalised depth per bin, the peaks and genes in the window. ``gene`` = its
    body +- 5 kb; ``region`` = chr1:1000-5000."""
    if not region and not gene:
        raise HTTPException(status_code=422, detail="Give a gene or a region like chr1:1000-5000.")
    adata = (await _adaptor(dm, dataset_id)).adata
    data_dir = settings.DATA_DIR

    def work() -> dict[str, Any]:
        t0 = time.monotonic()
        frag, idx = af.require_files(data_dir, dataset_id)
        genome = af.status(data_dir, dataset_id).get("genome")
        table, gene_note = None, None
        try:
            if genome:
                table = gene_annotation.load(genome, data_dir, auto_download=settings.ANNOTATION_AUTO_DOWNLOAD)
            else:
                gene_note = "This file's chromosome names match neither GRCh38 nor mm10, so no genes are drawn."
        except gene_annotation.AnnotationError as e:
            gene_note = str(e)
        focus = None
        if gene:
            if table is None:
                raise af.FragmentsError(gene_note or "Gene positions are not available: give a region like "
                                                     "chr1:1000-5000.", 503)
            chrom, start, end, focus = af.gene_window(table, gene)
        else:
            chrom, start, end = af.parse_region(region or "")
        reader = ff.FragmentReader(frag, idx, _cell_names(adata))
        out = af.coverage(reader, adata, chrom, start, end, group_by=group_by or None, groups=groups, bins=bins)
        out["peaks"] = af.peaks_in(data_dir, dataset_id, chrom, start, end)
        out["genes"] = af.genes_in(table, chrom, start, end)
        out["gene"] = focus
        out["genes_note"] = gene_note or ("Gene bodies from GENCODE, without exons." if table is not None else None)
        out["seconds"] = round(time.monotonic() - t0, 3)
        return out

    try:
        return await asyncio.to_thread(work)
    except af.FragmentsError as e:
        raise _http(e)
    except ff.FragmentFileError as e:
        raise HTTPException(status_code=422, detail=str(e))
