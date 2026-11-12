"""Motifs in ATAC peaks (core/atac_motifs.py, core/atac_chromvar.py).

``GET /datasets/{id}/atac/motifs``: what the scan and chromVAR would use and need, whether each is
done. ``POST .../atac/motifs/scan``: downloads what is missing and scans the peaks in a child process,
streamed (SSE); the scan outlives the request and a second request joins it, ``.../scan/cancel``
stops it. ``POST .../atac/motifs/enrichment``: Signac's FindMotifs on a stored differential result;
``GET .../enrichment[/{id}[/csv|/export]]``. ``GET .../atac/motifs/logo/{motif}``: a motif's logo.
``POST .../atac/motifs/chromvar``: chromVAR deviations in a child process (SSE); ``GET .../chromvar``
the saved run, ``.../chromvar/groups?column=`` groups x motifs, ``.../chromvar/motif?motif_id=`` a
motif's z per cell, ``.../chromvar/csv`` the ranking.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import shutil
import threading
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from scview.api.v1 import atac as atac_api
from scview.config import Settings
from scview.core import atac, atac_chromvar as cv, atac_motifs as am, genome_build, genome_seq, modality_store, results_store, tf_motifs, wnn
from scview.core.dataset_manager import DatasetManager
from scview.core.gene_annotation import AnnotationError, canonical_genome
from scview.core.sse import keepalive
from scview.dependencies import get_dataset_manager, get_settings_dep

logger = logging.getLogger(__name__)

router = APIRouter()

KIND_CHROMVAR = "atac_chromvar"
atac_api._WHAT[KIND_CHROMVAR] = "chromVAR"  # noqa: SLF001 - one stream at a time per dataset, with the other ATAC runs
POLL_SECONDS = 0.5
ENRICH_BYTES = 300 * 1024**2  # the hit table's query rows, the TF reads
MAX_PAGE = 1000


class EnrichmentRequest(BaseModel):
    da_id: str | None = Field(default=None, max_length=64)
    groups: list[str] | None = Field(default=None, max_length=atac.MAX_GROUPS)
    p_column: Literal["pval", "pval_adj"] = "pval"
    p_max: float = Field(default=am.QUERY_P, gt=0, le=1)
    min_pct: float = Field(default=am.QUERY_PCT, ge=0, le=1)
    min_log2fc: float = Field(default=am.QUERY_LOG2FC)
    background: int = Field(default=am.BACKGROUND_PEAKS, ge=100, le=500_000)
    seed: int = Field(default=am.SEED, ge=0, le=2**31 - 1)


class ChromVarRequest(BaseModel):
    seed: int = Field(default=cv.SEED, ge=0, le=2**31 - 1)


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


class _Peaks:
    """A store's peak set: its digest and sizes, read once per file."""

    def __init__(self, store: modality_store.ModalityStore) -> None:
        self.genome_name = store.meta.get("genome")
        chrom, start, end = am.peak_coordinates(store)
        self.digest = am.peak_digest(chrom, start, end)
        self.n_peaks = int(chrom.size)
        self.n_bases = int(np.where((start >= 0) & (end > start), end - start, 0).sum())


_peaks: dict[str, tuple[float, _Peaks]] = {}
_peaks_lock = threading.Lock()


def _peak_set(path: Path) -> _Peaks:
    mtime = path.stat().st_mtime
    with _peaks_lock:
        hit = _peaks.get(str(path))
    if hit and hit[0] == mtime:
        return hit[1]
    with modality_store.ModalityStore(path) as store:
        out = _Peaks(store)
    with _peaks_lock:
        while len(_peaks) >= 8:
            _peaks.pop(next(iter(_peaks)))
        _peaks[str(path)] = (mtime, out)
    return out


def _genome(name: str | None) -> str:
    try:
        return canonical_genome(name)
    except AnnotationError as e:
        if genome_build.old_build(name):
            raise atac.AtacError(str(e), 422) from None
        raise atac.AtacError(f"scView has genome sequence for GRCh38 and mm10, not {name!r}: motifs cannot be read from peaks "
                             "on another reference.", 422) from None


async def _adaptor(dm: DatasetManager, dataset_id: str):
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    return adaptor


def _where(adata, dm: DatasetManager, dataset_id: str) -> tuple[Path, _Peaks, str, str]:
    if not atac.has_atac(adata, dm.data_dir, dataset_id):
        raise atac.AtacError(atac.NO_ATAC, 404)
    path = atac.store_path(dm.data_dir, dataset_id)
    peaks = _peak_set(path)
    genome = _genome(peaks.genome_name)
    return path, peaks, genome, am.cache_key(genome, peaks.digest)


def _table_or_409(dm: DatasetManager, genome: str, key: str) -> am.PeakMotifTable:
    table = am.load_table(dm.data_dir, genome, key)
    if table is None:
        raise HTTPException(status_code=409, detail="The peaks have not been scanned for motifs yet: press Scan peaks first.")
    return table


def _http(e: atac.AtacError | cv.ChromVarError) -> HTTPException:
    return HTTPException(status_code=e.status, detail=str(e))


def _chromvar_state(summary: dict[str, Any], fingerprint: str | None) -> dict[str, Any]:
    keep = ("run_id", "n_cells", "n_peaks_used", "n_peaks_no_counts", "n_peaks_left_out", "seed", "iterations", "w", "bs",
            "genome", "seconds", "peak_mb", "created_at", "method")
    return {**{k: summary.get(k) for k in keep},
            "stale": fingerprint is not None and summary.get("cells_fingerprint") not in (None, fingerprint),
            "ranking": cv.ranking(summary, 100), "caveats": cv.CAVEATS, "citation": cv.CITATION}


@router.get("/datasets/{dataset_id}/atac/motifs")
async def state(
    dataset_id: str,
    dm: DatasetManager = Depends(get_dataset_manager),
    settings: Settings = Depends(get_settings_dep),
):
    """What each step would use and need; never hashes or reads the genome."""
    adaptor = await _adaptor(dm, dataset_id)
    adata = adaptor.adata

    def work() -> dict[str, Any]:
        path, peaks, genome, key = _where(adata, dm, dataset_id)
        table = am.load_table(dm.data_dir, genome, key)
        build = am.current_build(key)
        n_cells = int(adata.n_obs)
        with modality_store.ModalityStore(path) as store:
            nnz = int(store.nnz(atac.MATRIX) * n_cells / max(1, store.n_obs))
        jaspar = tf_motifs.JASPAR
        scan: dict[str, Any] = {
            "ready": table is not None, "running": build is not None, "progress": build.last if build else None,
            "n_peaks": peaks.n_peaks, "n_bases": peaks.n_bases, "estimate_seconds": round(am.estimate_scan_seconds(peaks.n_bases)),
            "estimate_bytes": am.estimate_scan_bytes(peaks.n_peaks), "meta": table.meta if table is not None else None,
            "n_motifs_expected": 879, "p_cutoff": am.MOTIF_P_CUTOFF,
        }
        mean_hits = (table.meta.get("hit_rate", 0.135) * table.n_motifs) if table is not None else 0.135 * 879
        summary = cv.read_summary(dm.data_dir, dataset_id)
        return {
            "available": True, "genome": genome, "key": key, "scan": scan,
            "files": {"genome": genome_seq.status(dm.data_dir, genome), "genome_auto": settings.GENOME_AUTO_DOWNLOAD,
                      "jaspar": {"name": jaspar.name, "url": jaspar.url, "size": jaspar.size,
                                 "downloaded": am.jaspar_path(dm.data_dir).is_file()}, "jaspar_auto": settings.TF_MOTIFS_AUTO_DOWNLOAD},
            "chromvar": _chromvar_state(summary, adaptor.cells_fingerprint()) if summary else None,
            "chromvar_estimate": {"n_cells": n_cells, "estimate_seconds": round(cv.estimate_seconds(nnz, cv.N_ITERATIONS, mean_hits)),
                                  "estimate_bytes": cv.estimate_bytes(n_cells, peaks.n_peaks, 879, cv.N_ITERATIONS, mean_hits),
                                  "disk_bytes": cv.estimate_disk_bytes(n_cells, 879), "iterations": cv.N_ITERATIONS},
            "enrichment_defaults": {"p_column": "pval", "p_max": am.QUERY_P, "min_pct": am.QUERY_PCT, "min_log2fc": am.QUERY_LOG2FC,
                                    "background": am.BACKGROUND_PEAKS, "seed": am.SEED},
            "caveats": am.CAVEATS, "citation": am.CITATION,
        }

    try:
        body = await asyncio.to_thread(work)
    except atac.AtacError as e:
        return {"available": False, "reason": str(e)}
    body["running"] = atac_api._running.get(dataset_id)  # noqa: SLF001
    return body


def _stream(events) -> StreamingResponse:
    return StreamingResponse(keepalive(events), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.post("/datasets/{dataset_id}/atac/motifs/scan")
async def scan(
    dataset_id: str,
    dm: DatasetManager = Depends(get_dataset_manager),
    settings: Settings = Depends(get_settings_dep),
):
    """Scan the peaks. Events: ``start`` (the estimate), ``progress``, ``result`` (the table's numbers),
    ``complete``; ``cancelled`` or ``error``. The scan keeps going when the client goes away."""
    adaptor = await _adaptor(dm, dataset_id)
    adata = adaptor.adata
    try:
        path, peaks, genome, key = await asyncio.to_thread(_where, adata, dm, dataset_id)
    except atac.AtacError as e:
        raise _http(e)
    need = am.estimate_scan_bytes(peaks.n_peaks)
    table = await asyncio.to_thread(am.load_table, dm.data_dir, genome, key)
    joined = am.current_build(key) is not None
    if table is None and not joined:
        await dm.reserve(dataset_id, need, what="The motif scan")
    del adata

    async def events():
        ready = await asyncio.to_thread(am.load_table, dm.data_dir, genome, key)
        if ready is None:
            run_key = f"motif-scan:{key}"

            def end() -> None:
                dm.release_run(run_key)

            existing = am.current_build(key)
            if existing is None:
                dm.reserve_run(run_key, need)
            build = am.start_build(dm.data_dir, genome, key, path, genome_auto=settings.GENOME_AUTO_DOWNLOAD,
                                   jaspar_auto=settings.TF_MOTIFS_AUTO_DOWNLOAD, on_end=None if existing else end)
            yield _sse("start", {"genome": genome, "n_peaks": peaks.n_peaks, "estimate_seconds": round(am.estimate_scan_seconds(peaks.n_bases)),
                                 "estimate_bytes": need, "joined": existing is not None})
            sent = 0
            while True:
                done = build.done  # read before the events, so none is missed
                for e in build.events[sent:]:
                    yield _sse("progress", e)
                sent = len(build.events)
                if done:
                    break
                await asyncio.sleep(POLL_SECONDS)
            if build.cancelled:
                yield _sse("cancelled", {"message": "The motif scan was stopped; nothing was kept."})
                yield _sse("complete", {})
                return
            if build.error:
                yield _sse("error", {"error": build.error})
                return
            ready = await asyncio.to_thread(am.load_table, dm.data_dir, genome, key)
            if ready is None:
                yield _sse("error", {"error": "The motif scan finished but its result cannot be read: scan again."})
                return
        yield _sse("result", {"genome": genome, "key": key, "meta": ready.meta})
        yield _sse("complete", {})

    return _stream(events())


@router.post("/datasets/{dataset_id}/atac/motifs/scan/cancel")
async def cancel_scan(dataset_id: str, dm: DatasetManager = Depends(get_dataset_manager)):
    """Stop the running scan of this dataset's peaks (whoever started it)."""
    adaptor = await _adaptor(dm, dataset_id)
    try:
        _p, _s, _g, key = await asyncio.to_thread(_where, adaptor.adata, dm, dataset_id)
    except atac.AtacError as e:
        raise _http(e)
    build = am.current_build(key)
    if build is None:
        return {"cancelled": False}
    build.stop.set()
    return {"cancelled": True}


@router.post("/datasets/{dataset_id}/atac/motifs/enrichment")
async def run_enrichment(dataset_id: str, body: EnrichmentRequest, dm: DatasetManager = Depends(get_dataset_manager)):
    """Each group's differential peaks against a GC-matched background (seconds); kept by id."""
    adaptor = await _adaptor(dm, dataset_id)
    adata = adaptor.adata
    try:
        _path, _peaks_set, genome, key = await asyncio.to_thread(_where, adata, dm, dataset_id)
    except atac.AtacError as e:
        raise _http(e)
    table = await asyncio.to_thread(_table_or_409, dm, genome, key)

    def pick() -> atac.DaResult | None:
        return atac.recall_da(dataset_id, body.da_id) if body.da_id else atac.latest_da(adaptor)

    da = await asyncio.to_thread(pick)
    if da is None:
        raise HTTPException(status_code=404, detail="No differential peaks to test: run Differential peaks first, on these cells.")
    unknown = [g for g in (body.groups or []) if g not in da.groups]
    if unknown:
        raise HTTPException(status_code=404, detail=f"The differential result has no group '{unknown[0]}'.")
    groups = list(body.groups or da.groups)
    await dm.reserve(dataset_id, ENRICH_BYTES, what="Motif enrichment")
    fingerprint = await asyncio.to_thread(adaptor.cells_fingerprint)
    params = {"da_id": da.result_id, "da_test": atac.da_test(da), "p_column": body.p_column, "p_max": body.p_max,
              "min_pct": body.min_pct, "min_log2fc": body.min_log2fc, "background": body.background, "seed": body.seed,
              "groups": body.groups, "da_params": da.params}

    def work() -> am.MotifEnrichment:
        frame, summary = am.enrich_groups(table, da.table, groups, p_column=body.p_column, p_max=body.p_max, min_pct=body.min_pct,
                                          min_log2fc=body.min_log2fc, n_background=body.background, seed=body.seed,
                                          kept_per_group=atac.KEEP_PER_GROUP)
        if len(frame):
            symbols = {t for m in table.motif_names.tolist() for t in tf_motifs.tf_members(m)}
            am.annotate_expression(frame, am.tf_statistics(adaptor, da.column, symbols))
        return am.MotifEnrichment(
            result_id=uuid.uuid4().hex[:12], da_id=da.result_id, column=da.column, groups=groups, table=frame, summary=summary,
            params=params, method=am.ENRICH_METHOD.format(n_background=body.background, n_motifs=table.n_motifs), genome=genome,
            scan={"key": key, "n_motifs": table.n_motifs, "n_peaks": table.n_peaks, "n_left_out": table.meta.get("n_left_out"),
                  "n_best_word_only": table.meta.get("n_best_word_only"), "p_cutoff": table.meta.get("p_cutoff")},
            created_at=datetime.now(timezone.utc).isoformat())

    try:
        res = await asyncio.to_thread(work)
    except am.EnrichmentError as e:
        raise HTTPException(status_code=422, detail=str(e))
    await atac_api._save(dm, dataset_id, fingerprint, (), (), "motif enrichment", lambda data: am.record(data, res))  # noqa: SLF001
    a = await _adaptor(dm, dataset_id)
    am.remember(a, dataset_id, res)
    return await asyncio.to_thread(am.payload, res)


@router.get("/datasets/{dataset_id}/atac/motifs/enrichment")
async def latest_enrichment(dataset_id: str, limit: int = Query(default=100, ge=1, le=MAX_PAGE),
                            dm: DatasetManager = Depends(get_dataset_manager)):
    """The last result on these cells, or null."""
    adaptor = await _adaptor(dm, dataset_id)

    def work():
        res = am.latest(adaptor)
        return am.payload(res, limit=limit) if res is not None else None

    return await asyncio.to_thread(work)


def _recall(dataset_id: str, result_id: str) -> am.MotifEnrichment:
    if not results_store.safe_id(result_id):
        raise HTTPException(status_code=404, detail="That motif enrichment result is gone. Run it again.")
    res = am.recall(dataset_id, result_id)
    if res is None:
        raise HTTPException(status_code=404, detail="That motif enrichment result is gone. Run it again.")
    return res


@router.get("/datasets/{dataset_id}/atac/motifs/enrichment/{result_id}")
async def get_enrichment(dataset_id: str, result_id: str, group: str = Query(default="", max_length=512),
                         offset: int = Query(default=0, ge=0), limit: int = Query(default=100, ge=1, le=MAX_PAGE),
                         dm: DatasetManager = Depends(get_dataset_manager)):
    await _adaptor(dm, dataset_id)
    res = await asyncio.to_thread(_recall, dataset_id, result_id)
    return await asyncio.to_thread(am.payload, res, group or None, offset, limit)


def _csv_text(res: am.MotifEnrichment) -> str:
    buf = io.StringIO()
    (res.table if res.table is not None else pd.DataFrame()).to_csv(buf, index=False)
    return buf.getvalue()


@router.get("/datasets/{dataset_id}/atac/motifs/enrichment/{result_id}/csv")
async def enrichment_csv(dataset_id: str, result_id: str, dm: DatasetManager = Depends(get_dataset_manager)):
    await _adaptor(dm, dataset_id)
    res = await asyncio.to_thread(_recall, dataset_id, result_id)
    body = await asyncio.to_thread(_csv_text, res)
    return StreamingResponse(iter([body]), media_type="text/csv", headers={
        "Content-Disposition": f'attachment; filename="motif_enrichment_{result_id}.csv"'})


@router.get("/datasets/{dataset_id}/atac/motifs/enrichment/{result_id}/export")
async def enrichment_export(dataset_id: str, result_id: str, dm: DatasetManager = Depends(get_dataset_manager)):
    """A zip: the table, the method, and the logo of every motif with FDR < 5% in some group."""
    await _adaptor(dm, dataset_id)
    res = await asyncio.to_thread(_recall, dataset_id, result_id)

    def build() -> bytes:
        motifs = {m.id: m for m in am.read_jaspar(dm.data_dir)} if am.jaspar_path(dm.data_dir).is_file() else {}
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("motif_enrichment.csv", _csv_text(res))
            z.writestr("README.txt", f"{res.method}\n\n{am.CAVEATS}\n\nCitations: {am.CITATION}\n\nParameters: {json.dumps(res.params, default=str)}\n")
            t = res.table
            if t is not None and len(t):
                for motif_id in t.loc[t["pval_adj"] < 0.05, "motif_id"].astype(str).unique().tolist():
                    m = motifs.get(motif_id)
                    if m is not None:
                        z.writestr(f"logos/{motif_id}.svg", am.logo_svg(m.counts, f"{m.name} ({m.id})"))
        return buf.getvalue()

    body = await asyncio.to_thread(build)
    return StreamingResponse(iter([body]), media_type="application/zip", headers={
        "Content-Disposition": f'attachment; filename="motif_enrichment_{result_id}.zip"'})


@router.get("/datasets/{dataset_id}/atac/motifs/logo/{motif_id}")
async def motif_logo(dataset_id: str, motif_id: str, dm: DatasetManager = Depends(get_dataset_manager)):
    """A motif's sequence logo (standalone SVG), its matrix's length and consensus."""
    await _adaptor(dm, dataset_id)
    if not am.jaspar_path(dm.data_dir).is_file():
        raise HTTPException(status_code=409, detail="The JASPAR motifs are not downloaded yet: press Scan peaks first.")

    def work() -> dict[str, Any] | None:
        for m in am.read_jaspar(dm.data_dir):
            if m.id == motif_id:
                return {"motif_id": m.id, "motif": m.name, "length": int(m.counts.shape[1]), "consensus": am.consensus(m.counts),
                        "svg": am.logo_svg(m.counts, f"{m.name} ({m.id})")}
        return None

    out = await asyncio.to_thread(work)
    if out is None:
        raise HTTPException(status_code=404, detail=f"JASPAR 2024 has no motif '{motif_id}'.")
    return out


@router.post("/datasets/{dataset_id}/atac/motifs/chromvar")
async def run_chromvar(dataset_id: str, body: ChromVarRequest, dm: DatasetManager = Depends(get_dataset_manager)):
    """chromVAR deviations per cell. Events: ``start``, ``stage``, ``watchdog``, ``result``, ``complete``,
    or ``error``. A client that goes away ends the child, and nothing is kept."""
    adaptor = await _adaptor(dm, dataset_id)
    atac_api._refuse_running(dataset_id)  # noqa: SLF001
    adata = adaptor.adata
    try:
        path, peaks, genome, key = await asyncio.to_thread(_where, adata, dm, dataset_id)
        table = await asyncio.to_thread(_table_or_409, dm, genome, key)
        store = await asyncio.to_thread(atac.open_store, adata, dm.data_dir, dataset_id)
        rows = await asyncio.to_thread(atac.cell_rows, store, adata)
        nnz = int(store.nnz(atac.MATRIX) * rows.size / max(1, store.n_obs))
    except atac.AtacError as e:
        raise _http(e)
    fingerprint = await asyncio.to_thread(adaptor.cells_fingerprint)
    mean_hits = table.meta.get("hit_rate", 0.135) * table.n_motifs
    need = cv.estimate_bytes(int(rows.size), peaks.n_peaks, table.n_motifs, cv.N_ITERATIONS, mean_hits)
    await dm.reserve(dataset_id, need, what="chromVAR")
    out_dir = cv.staging(dm.data_dir, dataset_id)
    table_file = am.table_path(dm.data_dir, genome, key)
    del adata

    def open_gen(stop):
        def wrapped():
            try:
                yield from cv.iter_run(path, rows, table_file, out_dir, seed=body.seed, stop=stop)
            except am.ScanCancelled:
                raise atac.AtacCancelled() from None
            except (am.ScanError, cv.ChromVarError) as e:
                raise atac.AtacError(str(e)) from e

        return wrapped()

    def clean() -> None:
        store.close()
        shutil.rmtree(out_dir, ignore_errors=True)

    async def on_result(item, res):
        summary = {**item["summary"], "cells_fingerprint": fingerprint, "created_at": datetime.now(timezone.utc).isoformat(),
                   "method": cv.METHOD.format(iterations=cv.N_ITERATIONS, w=cv.WIDTH, bs=cv.N_BINS, seed=body.seed)}
        resources = {"peak_gb": res["peak_gb"], "limit_gb": res["limit_gb"], "seconds": summary.get("seconds")}
        await atac_api._save(dm, dataset_id, fingerprint, (), (), "chromVAR",  # noqa: SLF001
                             lambda data: cv.record(data, summary, resources))
        await asyncio.to_thread(cv.publish, out_dir, dm.data_dir, dataset_id, summary)
        return {**_chromvar_state(summary, fingerprint), "resources": res}

    start = {"n_cells": int(rows.size), "n_peaks": peaks.n_peaks, "seed": body.seed, "iterations": cv.N_ITERATIONS,
             "estimate_seconds": round(cv.estimate_seconds(nnz, cv.N_ITERATIONS, mean_hits)), "estimate_bytes": need,
             "stages": [{"stage": "totals", "label": "Peak totals"}, {"stage": "background", "label": "Background peaks"},
                        {"stage": "deviations", "label": "Deviations"}]}
    return atac_api._stream(atac_api._events(  # noqa: SLF001
        dm, dataset_id, KIND_CHROMVAR, start, need, open_gen, on_result, "chromVAR", closers=[clean]))


@router.get("/datasets/{dataset_id}/atac/motifs/chromvar")
async def get_chromvar(dataset_id: str, dm: DatasetManager = Depends(get_dataset_manager)):
    """The saved run, or null."""
    adaptor = await _adaptor(dm, dataset_id)
    summary = await asyncio.to_thread(cv.read_summary, dm.data_dir, dataset_id)
    if summary is None:
        return None
    return _chromvar_state(summary, await asyncio.to_thread(adaptor.cells_fingerprint))


def _summary_or_404(dm: DatasetManager, dataset_id: str) -> dict[str, Any]:
    summary = cv.read_summary(dm.data_dir, dataset_id)
    if summary is None:
        raise HTTPException(status_code=404, detail="No chromVAR run on this dataset yet: run it on the Accessibility tab.")
    return summary


@router.get("/datasets/{dataset_id}/atac/motifs/chromvar/groups")
async def chromvar_groups(dataset_id: str, column: str = Query(max_length=256), dm: DatasetManager = Depends(get_dataset_manager)):
    """The most variable motifs x groups (mean z), and each group's motifs against the rest."""
    adaptor = await _adaptor(dm, dataset_id)
    adata = adaptor.adata

    def work() -> dict[str, Any]:
        summary = _summary_or_404(dm, dataset_id)
        codes, every, _chosen = atac.group_codes(adata, column, None)
        return cv.group_stats(dm.data_dir, dataset_id, summary, column, codes, every, adata.obs_names)

    try:
        return await asyncio.to_thread(work)
    except (atac.AtacError, cv.ChromVarError) as e:
        raise _http(e)


@router.get("/datasets/{dataset_id}/atac/motifs/chromvar/motif")
async def chromvar_motif(dataset_id: str, motif_id: str = Query(max_length=64), groupby: str = Query(default="", max_length=256),
                         dm: DatasetManager = Depends(get_dataset_manager)):
    """One motif's z per cell (the overlay), and by group when asked."""
    adaptor = await _adaptor(dm, dataset_id)
    adata = adaptor.adata

    def work() -> dict[str, Any]:
        summary = _summary_or_404(dm, dataset_id)
        z = cv.motif_values(dm.data_dir, dataset_id, summary, motif_id, adata.obs_names)
        finite = np.isfinite(z)
        out: dict[str, Any] = {"motif_id": motif_id, "motif": summary["motif_names"][summary["motif_ids"].index(motif_id)],
                               "n_cells": int(z.size), "n_nan": int((~finite).sum()),
                               "values": [None if not f else round(float(v), 3) for v, f in zip(z.tolist(), finite.tolist())],
                               "violin": None, "column": groupby or None}
        if groupby:
            try:
                labels, groups = wnn._groups(adata, groupby)  # noqa: SLF001
                out["violin"] = {g: [round(float(x), 3) for x in z[(labels == g) & finite]] for g in groups}
            except wnn.WnnError as exc:
                out["violin_error"] = str(exc)
        return out

    try:
        return await asyncio.to_thread(work)
    except cv.ChromVarError as e:
        raise _http(e)


@router.get("/datasets/{dataset_id}/atac/motifs/chromvar/csv")
async def chromvar_csv(dataset_id: str, dm: DatasetManager = Depends(get_dataset_manager)):
    """Every motif's variability, highest first."""
    await _adaptor(dm, dataset_id)

    def work() -> str:
        summary = _summary_or_404(dm, dataset_id)
        buf = io.StringIO()
        pd.DataFrame(cv.ranking(summary, len(summary["motif_ids"]))).to_csv(buf, index=False)
        return buf.getvalue()

    body = await asyncio.to_thread(work)
    return StreamingResponse(iter([body]), media_type="text/csv", headers={
        "Content-Disposition": 'attachment; filename="chromvar_variability.csv"'})
