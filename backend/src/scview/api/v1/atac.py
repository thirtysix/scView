"""10x Multiome ATAC endpoints (core/atac.py, core/atac_import.py).

``GET /datasets/{id}/atac`` is the state: whether the dataset has ATAC (``allow_empty``: a
sentence instead of a 404), the import's numbers, the saved runs, what a run would use and
need. ``POST .../atac/run-stream``: LSI, graph, UMAP, Leiden, gene activity (SSE).
``POST .../atac/wnn/run-stream``: WNN RNA + ATAC (SSE). ``GET .../atac/peaks``: every peak's
totals and context, paged. ``GET .../atac/activity?gene=``: one gene's activity per cell, by
group, beside its RNA. ``GET .../atac/qc``: the QC columns per cell. ``POST .../atac/metrics``:
attach a per_barcode_metrics.csv. ``POST .../atac/differential``: Wilcoxon per group (SSE),
kept by result id; ``GET .../atac/differential/{id}``, ``.../csv``. ``GET .../atac/export``: an
h5ad of the peak matrix.

Each run works on the store and its own copies, in worker threads, block by block; only the
final save holds the dataset's edit lock. A client that goes away stops it before its next block,
or ends the Leiden child, and nothing is saved.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import threading
import uuid
from collections.abc import Callable, Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from scview.config import Settings, get_settings
from scview.core import atac, atac_import, results_store
from scview.core.dataset_manager import DatasetManager
from scview.core.obs_writes import dataset_edit, edit_room, editable, save_edit, uns_snapshot
from scview.core.run_watchdog import RunControl, RunWatchdog
from scview.core.sse import keepalive
from scview.dependencies import get_dataset_manager, get_settings_dep

logger = logging.getLogger(__name__)

router = APIRouter()

_DONE = object()
_running: dict[str, str] = {}  # dataset -> what is streaming on it: one at a time each
_WHAT = {"run": "ATAC analysis (LSI, UMAP, clusters, gene activity)", "wnn": "WNN (joint RNA + ATAC neighbours)",
         "differential": "Differential accessibility"}
_PEAK_COLUMNS = ("peak", "chrom", "start", "end", "total", "n_cells", "pct_cells", "selected", "nearest_gene",
                 "distance_to_tss", "peak_class")
MAX_PAGE = 1000
MAX_SCATTER_CELLS = 5000


class RunRequest(BaseModel):
    n_lsi: int | None = Field(default=None, ge=3, le=200)
    dims_to: int | None = Field(default=None, ge=2, le=200)  # the graph's last LSI component
    drop_first: Literal["auto", "yes", "no"] = "auto"
    resolution: float | None = Field(default=None, gt=0, le=10)
    activity: bool = True


class WnnRequest(BaseModel):
    lsi_first: int = Field(default=2, ge=1)  # dims 2:n by default, as Seurat's multiome vignette
    lsi_last: int | None = Field(default=None, ge=2)
    resolution: float | None = Field(default=None, gt=0, le=10)


class DifferentialRequest(BaseModel):
    column: str = Field(max_length=256)
    groups: list[str] | None = Field(default=None, max_length=atac.MAX_GROUPS)
    min_pct: float = Field(default=atac.MIN_PCT, ge=0, le=1)
    max_peaks: int = Field(default=atac.MAX_PEAKS_PER_GROUP, ge=10, le=500_000)


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def _adaptor(dm: DatasetManager, dataset_id: str):
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    return adaptor


def _http(e: atac.AtacError) -> HTTPException:
    return HTTPException(status_code=e.status, detail=str(e))


def _auto_download() -> bool:
    return bool(get_settings().ANNOTATION_AUTO_DOWNLOAD)


def _refuse_running(dataset_id: str) -> None:
    what = _running.get(dataset_id)
    if what:
        raise HTTPException(status_code=409, detail=f"{_WHAT[what]} is already running on this dataset.")


@router.get("/datasets/{dataset_id}/atac")
async def get_atac(
    dataset_id: str,
    groupby: str = Query(default="", max_length=256, description="Mean RNA weight per group of this column"),
    allow_empty: bool = False,
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """The state of ATAC on this dataset. Without ATAC: a 404, or with ``allow_empty``
    ``{"available": False, "reason": ...}``."""
    adaptor = await _adaptor(dm, dataset_id)
    adata = adaptor.adata

    def work() -> dict[str, Any]:
        if not atac.has_atac(adata, dm.data_dir, dataset_id):
            return {"available": False,
                    "reason": atac.NO_STORE if atac_import.UNS_KEY in adata.uns else atac.NO_ATAC}
        out: dict[str, Any] = {"available": True, "reason": None, "n_cells": int(adata.n_obs),
                               **atac.summary(adata, groupby or None, adaptor.cells_fingerprint())}
        try:
            out["plan"], out["plan_reason"] = atac.plan(adata, dm.data_dir, dataset_id).to_dict(), None
        except atac.AtacError as e:
            out["plan"], out["plan_reason"] = None, str(e)
        try:
            out["wnn_plan"], out["wnn_reason"] = atac.plan_wnn(adata).to_dict(), None
        except atac.AtacError as e:
            out["wnn_plan"], out["wnn_reason"] = None, str(e)
        out["qc_columns"] = [c for c in atac_import.QC_COLUMNS if c in adata.obs.columns]
        return out

    body = await asyncio.to_thread(work)
    if not body["available"] and not allow_empty:
        raise HTTPException(status_code=404, detail=body["reason"])
    body["running"] = _running.get(dataset_id)
    return body


def _stream(events) -> StreamingResponse:
    return StreamingResponse(keepalive(events), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def _events(dm: DatasetManager, dataset_id: str, kind: str, start: dict, need: int, open_gen: Callable[[Any], Iterator],
            on_result: Callable, label: str, closers: list[Callable[[], None]] | None = None):
    """The stream every run shares. ``open_gen(stop)`` in a worker thread gives the run's
    generator; each of its items comes from a worker thread; ``on_result(item, resources)``
    saves the last one and returns the ``result`` event's body. A client going away stops it."""

    async def events():
        if dataset_id in _running:  # another request started between the check and here
            yield _sse("error", {"error": f"{_WHAT[_running[dataset_id]]} is already running on this dataset."})
            return
        _running[dataset_id] = kind
        stop = threading.Event()
        watchdog = RunWatchdog()
        control = RunControl(watchdog=watchdog, before_step=None)
        reservation = dm.reserve_run(dataset_id, need)
        sent = 0
        gen: Iterator | None = None
        pending: asyncio.Future | None = None

        def step_next():
            # The Leiden child registers with the control, so Cancel ends it.
            with control.step("atac"):
                return next(gen, _DONE)  # type: ignore[arg-type]

        def finish(fut: asyncio.Future | None = None) -> None:
            if fut is not None and not fut.cancelled():
                fut.exception()  # retrieved: a stopped run is not an error
            if gen is not None and hasattr(gen, "close"):
                gen.close()
            for close in closers or []:
                close()
            watchdog.stop()
            if reservation is not None:
                dm.release_run(dataset_id)
            _running.pop(dataset_id, None)

        try:
            watchdog.start()
            yield _sse("start", start)
            pending = asyncio.ensure_future(asyncio.to_thread(open_gen, stop.is_set))
            gen = await asyncio.shield(pending)
            pending = None
            while True:
                # Each item in a worker thread. Shielded, so a client going away cancels
                # this wait, not the thread's future, whose callback then closes the run.
                pending = asyncio.ensure_future(asyncio.to_thread(step_next))
                item = await asyncio.shield(pending)
                pending = None
                for warning in watchdog.events[sent:]:
                    yield _sse("watchdog", warning)
                sent = len(watchdog.events)
                if item is _DONE:
                    break
                if item["type"] == "result":
                    yield _sse("result", await on_result(item, watchdog.summary()))
                else:
                    yield _sse(item["type"], {k: v for k, v in item.items() if k != "type"})
            yield _sse("complete", {})
        except atac.AtacCancelled:
            return
        except atac.AtacError as e:
            yield _sse("error", {"error": str(e)})
        except Exception as e:  # noqa: BLE001 - surface to the client, keep the server up
            logger.error("%s failed for %s: %s", label, dataset_id, e, exc_info=True)
            detail = e.detail if isinstance(e, HTTPException) else str(e)
            yield _sse("error", {"error": f"{label} failed: {detail}"})
        finally:
            stop.set()  # a thread mid-block stops before the next one
            control.cancel()  # and a Leiden child now
            if pending is not None and not pending.done():
                pending.add_done_callback(finish)
            else:
                finish()

    return events()


@router.post("/datasets/{dataset_id}/atac/run-stream")
async def run_atac(dataset_id: str, body: RunRequest, dm: DatasetManager = Depends(get_dataset_manager)):
    """LSI, graph, UMAP, Leiden, gene activity. Events: ``start`` (the plan), ``stage``,
    ``watchdog``, ``result`` (the saved summary), ``complete``, or ``error``."""
    adaptor = await _adaptor(dm, dataset_id)
    _refuse_running(dataset_id)
    adata = adaptor.adata
    try:
        p = await asyncio.to_thread(atac.plan, adata, dm.data_dir, dataset_id, n_lsi=body.n_lsi, dims_to=body.dims_to,
                                    drop_first=body.drop_first, resolution=body.resolution, activity=body.activity)
    except atac.AtacError as e:
        raise _http(e)
    fingerprint = await asyncio.to_thread(adaptor.cells_fingerprint)
    need = atac.estimate_bytes(p.n_cells, p.n_peaks, n_lsi=p.n_lsi)
    await dm.reserve(dataset_id, need, what=_WHAT["run"])
    store = await asyncio.to_thread(atac.open_store, adata, dm.data_dir, dataset_id)
    rows = await asyncio.to_thread(atac.cell_rows, store, adata)
    del adata

    def open_gen(stop):
        # The first run fetches the gene positions: in a worker thread, and a failure is a
        # sentence on the activity card, never a lost run.
        chosen = atac.choose_genes(adaptor.adata, p.genome, dm.data_dir, _auto_download()) if p.activity else None
        return atac.iter_run(store, rows, p, data_dir=dm.data_dir, dataset_id=dataset_id, genes=chosen, stop=stop)

    async def on_result(item, res):
        result: atac.AtacResult = item["result"]
        resources = {"peak_gb": res["peak_gb"], "limit_gb": res["limit_gb"], "seconds": round(result.seconds, 1)}
        saved = await _save(dm, dataset_id, fingerprint, atac.RUN_OBSM, atac.RUN_OBSP, "ATAC analysis",
                            lambda data: atac.save_run(data, p, result, resources=resources,
                                                       cells_fingerprint=fingerprint))
        return {**saved, "resources": res}

    start = {**p.to_dict(), "stages": [{"stage": s, "label": label} for s, label in atac.STAGES
                                       if s != "activity" or p.activity]}
    return _stream(_events(dm, dataset_id, "run", start, need, open_gen, on_result, "ATAC analysis",
                           closers=[store.close]))


async def _save(dm: DatasetManager, dataset_id: str, fingerprint: str, obsm: tuple, obsp: tuple, what: str,
                write: Callable[[Any], dict[str, Any]]) -> dict[str, Any]:
    """Store a result in the dataset under its edit lock and save it (a large dataset's obs,
    obsm, obsp and uns written into its h5ad in place, X untouched)."""
    async with dataset_edit(dataset_id) as held:
        adaptor = await _adaptor(dm, dataset_id)
        if adaptor.cells_fingerprint() != fingerprint:
            raise atac.AtacError(f"The dataset's cells changed while {what} ran (a pipeline run or a filter); "
                                 "nothing was saved. Run it again.", 409)
        output_path = str(dm.derived_h5ad_path(dataset_id, adaptor.h5ad_path))
        backed = await edit_room(dm, dataset_id, adaptor)

        def work() -> dict[str, Any]:
            data = editable(adaptor)
            before = uns_snapshot(data)
            out = write(data)
            save_edit(adaptor, data, output_path, before, obsm_keys=obsm, obsp_keys=obsp)
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


@router.post("/datasets/{dataset_id}/atac/wnn/run-stream")
async def run_wnn(dataset_id: str, body: WnnRequest, dm: DatasetManager = Depends(get_dataset_manager)):
    """WNN RNA + ATAC. Events as the run's."""
    adaptor = await _adaptor(dm, dataset_id)
    _refuse_running(dataset_id)
    adata = adaptor.adata
    if not atac.has_atac(adata, dm.data_dir, dataset_id):
        raise HTTPException(status_code=404, detail=atac.NO_ATAC)
    try:
        p = await asyncio.to_thread(atac.plan_wnn, adata, lsi_first=body.lsi_first, lsi_last=body.lsi_last,
                                    resolution=body.resolution)
        inp = await asyncio.to_thread(atac.prepare_wnn, adata, p)
    except atac.AtacError as e:
        raise _http(e)
    fingerprint = await asyncio.to_thread(adaptor.cells_fingerprint)
    from scview.core import wnn

    need = wnn.estimate_bytes(p.n_cells, p.d_rna, len(p.lsi_dims))
    await dm.reserve(dataset_id, need, what=_WHAT["wnn"])
    del adata, adaptor  # the run needs only its copies

    async def on_result(item, res):
        result: atac.WnnAtacResult = item["result"]
        resources = {"peak_gb": res["peak_gb"], "limit_gb": res["limit_gb"], "seconds": round(result.seconds, 1)}
        saved = await _save(dm, dataset_id, fingerprint, atac.WNN_OBSM, atac.WNN_OBSP, "WNN",
                            lambda data: atac.save_wnn(data, inp, result, resources=resources,
                                                       cells_fingerprint=fingerprint))
        return {**saved, "resources": res}

    start = {**p.to_dict(), "stages": [{"stage": s, "label": label} for s, label in atac.WNN_STAGES]}
    return _stream(_events(dm, dataset_id, "wnn", start, need, lambda stop: atac.iter_wnn(inp, stop), on_result, "WNN"))


@router.get("/datasets/{dataset_id}/atac/peaks")
async def get_peaks(
    dataset_id: str,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=MAX_PAGE),
    sort: str = Query(default="total"),
    descending: bool = True,
    search: str = Query(default="", max_length=128),
    selected_only: bool = False,
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """Every peak: totals and detection over the dataset's cells, whether LSI selects it,
    nearest gene, distance to its TSS, class. ``search``: within the peak or gene name."""
    if sort not in _PEAK_COLUMNS:
        raise HTTPException(status_code=422, detail=f"Peaks cannot be sorted by '{sort}'. Use one of: "
                                                    f"{', '.join(_PEAK_COLUMNS)}.")
    adaptor = await _adaptor(dm, dataset_id)
    adata = adaptor.adata

    def work() -> dict[str, Any]:
        with atac.open_store(adata, dm.data_dir, dataset_id) as store:
            rows = atac.cell_rows(store, adata)
            stats = atac.cached_stats(store, rows, dm.data_dir, dataset_id, results_store.cells_hash(adata))
            table, note = atac.peak_table(store, rows, dm.data_dir, stats=stats, auto_download=_auto_download())
        if selected_only:
            table = table[table["selected"]]
        if search:
            hit = (table["peak"].str.contains(search, case=False, regex=False)
                   | table["nearest_gene"].astype(str).str.contains(search, case=False, regex=False))
            table = table[hit]
        table = table.sort_values(sort, ascending=not descending, kind="stable")
        page = table.iloc[offset:offset + limit]
        return {"total": int(len(table)), "offset": offset, "limit": limit, "note": note,
                "rows": json.loads(page.to_json(orient="records"))}

    try:
        return await asyncio.to_thread(work)
    except atac.AtacError as e:
        raise _http(e)


@router.get("/datasets/{dataset_id}/atac/qc")
async def get_qc(
    dataset_id: str,
    groupby: str = Query(default="", max_length=256),
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """Each ATAC QC column per cell (histograms) and each cell's group (violins)."""
    from scview.core import wnn

    adata = (await _adaptor(dm, dataset_id)).adata
    try:
        return await asyncio.to_thread(atac.qc_values, adata, groupby or None)
    except wnn.WnnError as e:
        raise HTTPException(status_code=422, detail=str(e))


@router.post("/datasets/{dataset_id}/atac/metrics")
async def attach_metrics(
    dataset_id: str,
    file: UploadFile = File(...),
    sample: str = Form(""),
    column: str = Form(""),
    settings: Settings = Depends(get_settings_dep),
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """Attach a per_barcode_metrics.csv afterwards (to one ``sample`` of ``column`` in a merged
    dataset): adds atac_frac_peaks, atac_frac_tss, atac_frac_mito."""
    adaptor = await _adaptor(dm, dataset_id)
    if not atac.has_atac(adaptor.adata, dm.data_dir, dataset_id):
        raise HTTPException(status_code=404, detail=atac.NO_ATAC)
    if column and column not in adaptor.adata.obs.columns:
        raise HTTPException(status_code=404, detail=f"Column '{column}' not found.")
    if column and not sample:
        raise HTTPException(status_code=422, detail=f"Choose which '{column}' value this file belongs to.")
    stage = Path(settings.DATA_DIR) / "tmp" / f"atac-metrics-{uuid.uuid4().hex}"
    stage.mkdir(parents=True, exist_ok=True)
    try:
        dest = stage / Path(file.filename or "per_barcode_metrics.csv").name  # never the client's path
        total = 0
        with open(dest, "wb") as fh:
            while chunk := await file.read(1024 * 1024):
                total += len(chunk)
                if total > settings.max_upload_bytes:
                    raise HTTPException(status_code=413, detail=f"The file exceeds the maximum upload size of "
                                                                f"{settings.MAX_UPLOAD_SIZE_MB} MB.")
                fh.write(chunk)

        async with dataset_edit(dataset_id) as held:
            adaptor = await _adaptor(dm, dataset_id)
            output_path = str(dm.derived_h5ad_path(dataset_id, adaptor.h5ad_path))
            backed = await edit_room(dm, dataset_id, adaptor)

            def work() -> dict[str, Any]:
                data = editable(adaptor)
                before = uns_snapshot(data)
                suffix = f":{sample}" if column else ""
                out = atac_import.attach_metrics(data, dest, sample_column=column or None,
                                                 sample=sample or None, suffix=suffix)
                save_edit(adaptor, data, output_path, before)
                return out

            try:
                out = await held.run(work)
            except (atac_import.AtacImportError, ValueError, pd.errors.ParserError) as e:
                raise HTTPException(status_code=422, detail=str(e))
            if backed:
                dm.evict(dataset_id)
                await dm.load_dataset(dataset_id)
            else:
                dm.saved_in_place(dataset_id, adaptor, output_path)
        return out
    finally:
        import shutil

        shutil.rmtree(stage, ignore_errors=True)


@router.get("/datasets/{dataset_id}/atac/activity")
async def get_activity(
    dataset_id: str,
    gene: str = Query(max_length=256),
    groupby: str = Query(default="", max_length=256),
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """One gene's activity per cell (log1p per 10k, the overlay), its values by group when
    asked, and beside its RNA: a sample of cells for the scatter, Pearson r over all cells."""
    from scview.core import wnn

    adaptor = await _adaptor(dm, dataset_id)
    adata = adaptor.adata

    def work() -> dict[str, Any]:
        out = atac.activity_gene(adata, dm.data_dir, dataset_id, gene)
        values = out["values"]
        body: dict[str, Any] = {"gene": gene, "caption": out["caption"], "n_cells": int(values.size),
                                "n_nonzero": int(np.count_nonzero(out["counts"])),
                                "values": [round(float(v), 4) for v in values.tolist()], "violin": None,
                                "column": groupby or None, "rna": None, "r": None}
        if groupby:
            try:
                labels, groups = wnn._groups(adata, groupby)
                body["violin"] = {g: [round(float(x), 3) for x in values[labels == g]] for g in groups}
            except wnn.WnnError as exc:
                body["violin_error"] = str(exc)
        rna = adaptor.get_expression([gene])
        if rna.shape[1] == 1:
            r = rna[:, 0].astype(np.float64)
            a = values.astype(np.float64)
            if r.std() > 0 and a.std() > 0:
                body["r"] = round(float(np.corrcoef(r, a)[0, 1]), 4)
            n = r.size
            take = np.arange(n) if n <= MAX_SCATTER_CELLS else np.linspace(0, n - 1, MAX_SCATTER_CELLS).astype(np.int64)
            body["rna"] = {"rna": [round(float(x), 3) for x in r[take]], "activity": [round(float(x), 3) for x in a[take]],
                           "n_shown": int(take.size)}
        return body

    try:
        return await asyncio.to_thread(work)
    except atac.AtacError as e:
        raise _http(e)


@router.post("/datasets/{dataset_id}/atac/differential")
async def run_differential(dataset_id: str, body: DifferentialRequest, dm: DatasetManager = Depends(get_dataset_manager)):
    """Wilcoxon on TF-IDF per group vs the rest, streamed: ``start``, ``progress``, ``result``
    (the id, each group's counts and its first rows), ``complete``, or ``error``. Kept by id."""
    adaptor = await _adaptor(dm, dataset_id)
    _refuse_running(dataset_id)
    adata = adaptor.adata
    try:
        codes, every, chosen = await asyncio.to_thread(atac.group_codes, adata, body.column, body.groups)
        store = await asyncio.to_thread(atac.open_store, adata, dm.data_dir, dataset_id)
        rows = await asyncio.to_thread(atac.cell_rows, store, adata)
    except atac.AtacError as e:
        raise _http(e)
    cells = await asyncio.to_thread(results_store.cells_hash, adata)
    need = atac.estimate_da_bytes(store.n_vars, len(every))
    await dm.reserve(dataset_id, need, what=_WHAT["differential"])
    genome = store.meta.get("genome")

    async def on_result(item, res):
        def finish() -> dict[str, Any]:
            table, note = atac.attach_context(item["frame"], store, genome, dm.data_dir, _auto_download())
            params = {"min_pct": body.min_pct, "max_peaks": body.max_peaks, "test": "wilcoxon", "on": "TF-IDF"}
            result_id = uuid.uuid4().hex[:12]
            kept = atac.DaResult(result_id=result_id, column=body.column, groups=chosen, table=table,
                                 n_tested=item["n_tested"], n_significant=item["n_significant"], params=params,
                                 method=atac.DA_METHOD.format(min_pct=body.min_pct), cells=cells,
                                 created_at=datetime.now(timezone.utc).isoformat())
            return {"kept": kept, "note": note}

        out = await asyncio.to_thread(finish)
        a = await _adaptor(dm, dataset_id)
        atac.remember_da(a, dataset_id, out["kept"])
        return {**atac.da_payload(out["kept"], limit=100), "note": out["note"], "resources": res}

    start = {"column": body.column, "groups": chosen, "n_cells": int(rows.size), "min_pct": body.min_pct,
             "estimate_seconds": atac.estimate_da_seconds(store.nnz(atac.MATRIX)), "estimate_bytes": need}
    return _stream(_events(
        dm, dataset_id, "differential", start, need,
        lambda stop: atac.iter_differential(store, rows, codes, every, chosen, min_pct=body.min_pct,
                                            max_peaks=body.max_peaks, stop=stop),
        on_result, "Differential accessibility", closers=[store.close]))


def _recall(dataset_id: str, result_id: str, adaptor) -> atac.DaResult:
    kept = atac.recall_da(dataset_id, result_id)
    if kept is None:
        raise HTTPException(status_code=404, detail="That differential accessibility result is gone. Run it again.")
    return kept


@router.get("/datasets/{dataset_id}/atac/differential")
async def latest_differential(dataset_id: str, limit: int = Query(default=100, ge=1, le=MAX_PAGE),
                              dm: DatasetManager = Depends(get_dataset_manager)):
    """The last result on these cells, or null."""
    adaptor = await _adaptor(dm, dataset_id)

    def work():
        kept = atac.latest_da(adaptor)
        return atac.da_payload(kept, limit=limit) if kept is not None else None

    return await asyncio.to_thread(work)


@router.get("/datasets/{dataset_id}/atac/differential/{result_id}")
async def get_differential(
    dataset_id: str, result_id: str, group: str = Query(default="", max_length=512),
    offset: int = Query(default=0, ge=0), limit: int = Query(default=100, ge=1, le=MAX_PAGE),
    dm: DatasetManager = Depends(get_dataset_manager),
):
    adaptor = await _adaptor(dm, dataset_id)
    kept = await asyncio.to_thread(_recall, dataset_id, result_id, adaptor)
    return await asyncio.to_thread(atac.da_payload, kept, group or None, offset, limit)


@router.get("/datasets/{dataset_id}/atac/differential/{result_id}/csv")
async def differential_csv(dataset_id: str, result_id: str, dm: DatasetManager = Depends(get_dataset_manager)):
    adaptor = await _adaptor(dm, dataset_id)
    kept = await asyncio.to_thread(_recall, dataset_id, result_id, adaptor)

    def text() -> str:
        buf = io.StringIO()
        kept.table.drop(columns=["peak_index"], errors="ignore").to_csv(buf, index=False)
        return buf.getvalue()

    body = await asyncio.to_thread(text)
    return StreamingResponse(iter([body]), media_type="text/csv", headers={
        "Content-Disposition": f'attachment; filename="differential_accessibility_{result_id}.csv"'})


@router.get("/datasets/{dataset_id}/atac/export")
async def export_peaks(dataset_id: str, dm: DatasetManager = Depends(get_dataset_manager)):
    """An h5ad of the peak counts of the dataset's cells (cells x peaks), for Signac and muon."""
    adaptor = await _adaptor(dm, dataset_id)
    try:
        path = await asyncio.to_thread(atac.export_h5ad, adaptor.adata, dm.data_dir, dataset_id)
    except atac.AtacError as e:
        raise _http(e)
    info = dm.get_dataset_info(dataset_id) or {}
    from scview.api.v1.export import file_stem

    return FileResponse(path, media_type="application/x-hdf5",
                        filename=f"{file_stem(str(info.get('name') or dataset_id))}_peaks.h5ad")

