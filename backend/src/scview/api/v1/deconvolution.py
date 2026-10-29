"""Bulk deconvolution with the open dataset as the reference (``core/deconvolution.py``).

``POST /deconvolution/upload`` (multipart: the bulk file, optional on a re-run, and
``options`` JSON) runs it, keeps the file beside the dataset and records a History
step; ``GET /deconvolution`` gives the last run (results store) or 404, ``GET
/deconvolution/{id}.csv`` its fractions. ``POST /deconvolution/validate`` streams the
leave-one-donor-out self-check (SSE, ~20 s on Kang); ``GET /deconvolution/check`` the
last one. ``GET /deconvolution/reference.zip`` streams the reference for other tools.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import logging
import os
import re
import threading
from pathlib import Path
from typing import Any, Literal

import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, ValidationError

from scview.core import deconvolution as dc
from scview.core import results_store
from scview.core.dataset_manager import DatasetManager
from scview.core.obs_writes import dataset_edit, save_edit, uns_snapshot
from scview.core.sse import keepalive
from scview.dependencies import get_dataset_manager

logger = logging.getLogger(__name__)

router = APIRouter()

MAX_POINTS = 4000  # self-check points per method sent for the scatter
_SAFE = re.compile(r"[^A-Za-z0-9._-]+")
_DONE = object()
_checking: set[str] = set()


class DeconvOptions(BaseModel):
    celltype_column: str | None = Field(default=None, max_length=256)
    # None: detect the donors; "": no donor column
    donor_column: str | None = Field(default=None, max_length=256)
    method: Literal["auto", "weighted", "nnls"] = "auto"
    # Only the cells with cells_column = cells_value (one condition) form the reference.
    cells_column: str | None = Field(default=None, max_length=256)
    cells_value: str | None = Field(default=None, max_length=512)


class CheckRequest(DeconvOptions):
    n_mixtures: int = Field(default=dc.N_MIXTURES, ge=1, le=dc.MAX_MIXTURES)
    seed: int = Field(default=0, ge=0, le=2**31 - 1)


def _num(v: Any, digits: int = 4) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return round(f, digits) if np.isfinite(f) else None


def result_body(r: dc.Result) -> dict[str, Any]:
    by_sample: dict[str, dict] = {s: {"cell": {}, "rna": {}} for s in r.samples}
    for row in r.fractions.itertuples(index=False):
        by_sample[row.sample]["cell"][row.cell_type] = _num(row.cell_fraction)
        by_sample[row.sample]["rna"][row.cell_type] = _num(row.rna_fraction)
    fit = {row.sample: row for row in r.fit.itertuples(index=False)}
    flagged: dict[str, list[str]] = {}
    for f in r.flags:
        flagged.setdefault(f["sample"], []).append(f["cell_type"])
    samples = []
    for s in r.samples:
        f = fit.get(s)
        samples.append({
            "sample": s, "cell_fractions": by_sample[s]["cell"], "rna_fractions": by_sample[s]["rna"],
            "r2": _num(f.r2) if f is not None else None, "residual": _num(f.residual) if f is not None else None,
            "genes": int(f.genes) if f is not None else 0, "iterations": int(f.iterations) if f is not None else 0,
            "converged": bool(f.converged) if f is not None else False,
            "flagged": flagged.get(s, []),
        })
    return {
        "result_id": r.id, "method": r.method, "method_name": dc.METHOD_NAMES[r.method],
        "method_requested": r.method_requested, "method_reason": r.method_reason,
        "celltype_column": r.celltype_column, "donor_column": r.donor_column, "donor_reason": r.donor_reason,
        "types": r.types, "cell_size": [_num(x, 1) for x in r.cell_size], "n_donors": r.n_donors,
        "n_reference_cells": r.n_reference_cells, "n_reference_donors": r.n_reference_donors,
        "counts_source": r.counts_source, "filename": r.filename, "n_samples": len(r.samples),
        "samples": samples, "n_genes_file": r.n_genes_file, "n_genes_matched": r.n_genes_matched,
        "gene_share": _num(r.gene_share), "n_genes_used": r.n_genes_used, "marker_source": r.marker_source,
        "id_kind": r.id_kind, "input_kind": r.input_kind, "warnings": r.warnings, "notes": r.notes,
        "elapsed_seconds": r.elapsed_seconds, "created_at": r.created_at,
        "reference_cells": r.reference_cells, "type_share": [_num(x) for x in r.type_share],
        "flags": [{**f, "fraction": _num(f["fraction"]), "reference_share": _num(f["reference_share"])}
                  for f in r.flags],
    }


def check_body(c: dc.CheckResult) -> dict[str, Any]:
    pts = c.points
    types = list(c.types)
    tindex = {t: i for i, t in enumerate(types)}
    points: dict[str, dict[str, list]] = {}
    capped = False
    rng = np.random.default_rng(c.seed)
    for meth, g in pts.groupby("method", sort=False):
        if len(g) > MAX_POINTS:
            g = g.iloc[np.sort(rng.choice(len(g), MAX_POINTS, replace=False))]
            capped = True
        points[str(meth)] = {
            "type": [tindex.get(t, -1) for t in g["cell_type"]],
            "truth": [_num(v) for v in g["truth"]], "estimate": [_num(v) for v in g["estimate"]],
            "pseudobulk": [bool(m < 0) for m in g["mixture"]],
        }
    return {
        "result_id": c.id, "celltype_column": c.celltype_column, "donor_column": c.donor_column,
        "methods": c.methods, "method_names": {m: dc.METHOD_NAMES[m] for m in c.methods},
        "n_mixtures": c.n_mixtures, "seed": c.seed, "donors": c.donors, "types": types,
        "overall": [{k: (_num(v) if k in ("r", "rmse", "bias") else v) for k, v in o.items()} for o in c.overall],
        "per_type": [{"method": row.method, "cell_type": row.cell_type, "r": _num(row.r), "rmse": _num(row.rmse),
                      "bias": _num(row.bias), "n": int(row.n)} for row in c.per_type.itertuples(index=False)],
        "points": points, "points_capped": capped,
        "n_points": {str(m): int(n) for m, n in pts.groupby("method", sort=False).size().items()}, "notes": c.notes,
        "elapsed_seconds": c.elapsed_seconds, "created_at": c.created_at,
        "reference_cells": c.reference_cells,
    }


async def _adaptor(dm: DatasetManager, dataset_id: str):
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")
    return adaptor


def _folder(dm: DatasetManager, dataset_id: str) -> Path:
    if not results_store.safe_id(dataset_id):
        raise HTTPException(status_code=400, detail="Bad dataset id.")
    return Path(dm.data_dir) / "uploads" / dataset_id / "deconvolution"


async def _store_upload(folder: Path, file: UploadFile) -> tuple[Path, str, str]:
    """Stream the upload into the dataset's deconvolution folder: (path, original name, sha256).
    The newest ``KEEP_FILES`` files are kept, so a reload or a re-run finds the last one."""
    folder.mkdir(parents=True, exist_ok=True)
    name = Path(file.filename or "bulk.tsv").name
    partial = folder / f".upload-{os.getpid()}-{id(file)}.partial"
    digest = hashlib.sha256()
    total = 0
    try:
        with open(partial, "wb") as fh:
            while chunk := await file.read(1024 * 1024):
                total += len(chunk)
                if total > dc.MAX_BULK_BYTES:
                    raise HTTPException(status_code=413, detail=(
                        f"File exceeds the {dc.MAX_BULK_BYTES // (1024 * 1024)} MB limit for bulk tables."))
                digest.update(chunk)
                fh.write(chunk)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    if total == 0:
        partial.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail="The bulk file is empty.")
    sha = digest.hexdigest()
    dest = folder / f"{sha[:12]}_{_SAFE.sub('_', name).strip('_') or 'bulk'}"
    os.replace(partial, dest)
    os.utime(dest)
    kept = sorted((p for p in folder.iterdir() if p.is_file() and not p.name.startswith(".")),
                  key=lambda p: p.stat().st_mtime, reverse=True)
    for old in kept[dc.KEEP_FILES:]:
        old.unlink(missing_ok=True)
    return dest, name, sha


def _options(raw: str) -> DeconvOptions:
    try:
        return DeconvOptions(**(json.loads(raw) if raw.strip() else {}))
    except (json.JSONDecodeError, TypeError) as e:
        raise HTTPException(status_code=422, detail=f"options must be a JSON object: {e}")
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors(include_url=False, include_context=False))


@router.get("/datasets/{dataset_id}/deconvolution/defaults")
async def deconvolution_defaults(dataset_id: str, dm: DatasetManager = Depends(get_dataset_manager)):
    """The columns a run uses when none are named, and what the donors allow."""
    adata = (await _adaptor(dm, dataset_id)).adata

    def _run() -> dict[str, Any]:
        from scview.core.de_composition import default_column

        ct = default_column(adata)
        donor, reason = dc.default_donor_column(adata)
        n_donors = int(adata.obs[donor].nunique()) if donor else 0
        return {"celltype_column": ct, "donor_column": donor, "donor_reason": reason, "n_donors": n_donors,
                "min_donors_weighted": dc.MIN_DONORS_WEIGHTED, "min_donors_check": dc.MIN_DONORS_CHECK,
                "check_seconds_per_donor": round(dc.check_estimate_seconds(1), 2),
                "max_bulk_mb": dc.MAX_BULK_BYTES // (1024 * 1024), "max_samples": dc.MAX_BULK_SAMPLES,
                "reference_columns": dc.reference_columns(adata, exclude=(ct, donor))}

    return await asyncio.to_thread(_run)


# "upload" in the path: main.py's body-size limit (512 KB for JSON) skips upload routes,
# and a bulk table is often tens of MB.
@router.post("/datasets/{dataset_id}/deconvolution/upload")
async def run_deconvolution(
    dataset_id: str,
    file: UploadFile | None = File(default=None),
    options: str = Form(default=""),
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """Deconvolve the bulk file's samples. Without a file, the last run's file is used again."""
    opts = _options(options)
    folder = _folder(dm, dataset_id)
    await _adaptor(dm, dataset_id)
    if file is not None and (file.filename or file.size):
        path, name, sha = await _store_upload(folder, file)
    else:
        prev = dc.last(await _adaptor(dm, dataset_id))
        path = folder / prev.stored_file if prev is not None and prev.stored_file else None
        if prev is None or path is None or not path.is_file():
            raise HTTPException(status_code=422, detail="Choose a bulk expression file to deconvolve.")
        name, sha = prev.filename, prev.sha256

    async with dataset_edit(dataset_id) as held:
        adaptor = await _adaptor(dm, dataset_id)
        adata = adaptor.adata
        output_path = str(dm.derived_h5ad_path(dataset_id, adaptor.h5ad_path))

        def _run() -> dc.Result:
            r = dc.run(adata, path.read_bytes(), name, celltype_column=opts.celltype_column,
                       donor_column=opts.donor_column, method=opts.method, dataset_id=dataset_id, sha256=sha,
                       stored_file=path.name, cells_column=opts.cells_column or None, cells_value=opts.cells_value)
            before = uns_snapshot(adata)
            dc.record(adata, r)
            save_edit(adaptor, adata, output_path, before)
            return r

        try:
            r = await held.run(_run)
        except dc.DeconvolutionError as e:
            raise HTTPException(status_code=422, detail=str(e))
        except Exception as e:
            dm.evict(dataset_id)  # the cached copy may hold the step without it being saved
            logger.error("Deconvolution failed for %s: %s", dataset_id, e, exc_info=True)
            raise HTTPException(status_code=500, detail=f"Could not deconvolve the bulk samples: {e!s}")
        dm.saved_in_place(dataset_id, adaptor, output_path)
        dc.remember(adaptor, r)
        return result_body(r)


@router.get("/datasets/{dataset_id}/deconvolution")
async def last_deconvolution(dataset_id: str, allow_empty: bool = False,
                             dm: DatasetManager = Depends(get_dataset_manager)):
    """The last run on this dataset, or 404; with ``allow_empty`` null instead (the panel asks on open,
    and a 404 is a console error in the browser)."""
    r = dc.last(await _adaptor(dm, dataset_id))
    if r is None:
        if allow_empty:
            return None
        raise HTTPException(status_code=404, detail="No bulk deconvolution on this dataset yet.")
    return result_body(r)


@router.get("/datasets/{dataset_id}/deconvolution/check")
async def last_check(dataset_id: str, allow_empty: bool = False,
                     dm: DatasetManager = Depends(get_dataset_manager)):
    """The last self-check on this dataset, or 404; with ``allow_empty`` null instead (the panel asks on open,
    and a 404 is a console error in the browser)."""
    c = dc.last_check(await _adaptor(dm, dataset_id))
    if c is None:
        if allow_empty:
            return None
        raise HTTPException(status_code=404, detail="No deconvolution self-check on this dataset yet.")
    return check_body(c)


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@router.post("/datasets/{dataset_id}/deconvolution/validate")
async def run_check(dataset_id: str, body: CheckRequest, dm: DatasetManager = Depends(get_dataset_manager)):
    """Hold out each donor in turn and score the estimates of its pseudobulk and mixtures
    (SSE: ``progress`` per donor, ``result``, ``complete``). Reads the dataset, saves nothing."""
    adaptor = await _adaptor(dm, dataset_id)
    adata = adaptor.adata
    if dataset_id in _checking:
        raise HTTPException(status_code=409, detail="A self-check is already running on this dataset.")
    stop = threading.Event()
    gen = dc.iter_check(adata, celltype_column=body.celltype_column, donor_column=body.donor_column,
                        method=body.method, n_mixtures=body.n_mixtures, seed=body.seed, dataset_id=dataset_id,
                        stop=stop.is_set, cells_column=body.cells_column or None, cells_value=body.cells_value)
    try:  # the first event (the reference built, the donors counted) before the stream starts
        first = await asyncio.to_thread(next, gen, _DONE)
    except dc.DeconvolutionError as e:
        raise HTTPException(status_code=422, detail=str(e))

    async def events():
        if dataset_id in _checking:  # another request started between the check and here
            gen.close()
            yield _sse("error", {"error": "A self-check is already running on this dataset."})
            return
        _checking.add(dataset_id)
        pending: asyncio.Future | None = None

        def finish(fut: asyncio.Future | None = None) -> None:
            if fut is not None and not fut.cancelled():
                fut.exception()  # retrieved: a stopped check is not an error
            gen.close()
            _checking.discard(dataset_id)

        try:
            item = first
            while item is not _DONE:
                event, data = item
                if event == "result":
                    dc.remember_check(adaptor, data)
                    yield _sse("result", check_body(data))
                else:
                    yield _sse(event, data)
                # Each held-out donor (a few seconds) in a worker thread; shielded, so a client
                # going away cancels this wait, not the thread.
                pending = asyncio.ensure_future(asyncio.to_thread(next, gen, _DONE))
                item = await asyncio.shield(pending)
                pending = None
            yield _sse("complete", {})
        except dc.DeconvolutionError as e:
            yield _sse("error", {"error": str(e)})
        except Exception as e:  # noqa: BLE001 - surface to the client, keep the server up
            logger.error("Deconvolution self-check failed for %s: %s", dataset_id, e, exc_info=True)
            yield _sse("error", {"error": str(e)})
        finally:
            stop.set()
            if pending is not None and not pending.done():
                pending.add_done_callback(finish)
            else:
                finish()

    return StreamingResponse(keepalive(events()), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.get("/datasets/{dataset_id}/deconvolution/reference.zip")
async def reference_zip(
    dataset_id: str,
    celltype_column: str | None = Query(default=None, max_length=256),
    donor_column: str | None = Query(default=None, max_length=256),
    cells_column: str | None = Query(default=None, max_length=256),
    cells_value: str | None = Query(default=None, max_length=512),
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """The reference as files for MuSiC, BayesPrism, CIBERSORTx and others, streamed."""
    adata = (await _adaptor(dm, dataset_id)).adata
    gen = dc.iter_reference_zip(adata, celltype_column=celltype_column or None, donor_column=donor_column,
                                dataset_id=dataset_id, cells_column=cells_column or None, cells_value=cells_value)
    try:
        first = await asyncio.to_thread(next, gen, b"")
    except dc.DeconvolutionError as e:
        raise HTTPException(status_code=422, detail=str(e))

    async def chunks():
        try:
            if first:
                yield first
            while True:
                part = await asyncio.to_thread(next, gen, _DONE)
                if part is _DONE:
                    break
                if part:
                    yield part
        finally:
            gen.close()

    return StreamingResponse(chunks(), media_type="application/zip", headers={
        "Content-Disposition": f'attachment; filename="scview_reference_{dataset_id}.zip"'})


@router.get("/datasets/{dataset_id}/deconvolution/{result_id}.csv")
async def fractions_csv(dataset_id: str, result_id: str, dm: DatasetManager = Depends(get_dataset_manager)):
    """A kept run's fractions, one row per bulk sample and kind (cell / RNA)."""
    adaptor = await _adaptor(dm, dataset_id)
    r = dc.last(adaptor)
    if r is None or r.id != result_id:
        r = results_store.recall(dataset_id, result_id, dc.KIND) if results_store.safe_id(result_id) else None
    if r is None:
        raise HTTPException(status_code=404, detail="That result is no longer kept on the server: run it again.")
    buf = io.StringIO()
    for kind, col in (("cell_fraction", "cell_fraction"), ("rna_fraction", "rna_fraction")):
        wide = r.fractions.pivot(index="sample", columns="cell_type", values=col).reindex(index=r.samples,
                                                                                          columns=r.types)
        wide.insert(0, "fraction", kind)
        wide.round(6).to_csv(buf, header=kind == "cell_fraction", index_label="sample", lineterminator="\n")
    return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv", headers={
        "Content-Disposition": f'attachment; filename="deconvolution_{result_id}.csv"'})
