"""Spatially variable genes (``core/spatial_autocorr.py``): Moran's I per gene.

``POST /spatial/autocorr`` runs it on the dataset's spots (all sections, or one) and
saves it with the dataset (uns ``moranI``, a provenance step the replay script
repeats); ``GET /spatial/autocorr`` gives the last run (results store) or 404, with
``library_id`` the last run on that section ("all": the run over every section);
``GET /spatial/autocorr/{result_id}.csv`` every gene of a kept run. Over several sections
the pooled I is within-section (``centring`` "section") and each row carries each
section's own values (``by_library``). ``POST
/spatial/autocorr/run-stream`` is the same run as SSE, what the UI uses: all genes on a
large dataset take many minutes, past what a proxy waits for a silent response.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import threading
from typing import Any, Literal

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from scview.core import results_store, working_copy
from scview.core import spatial_autocorr as sa
from scview.core.dataset_manager import DatasetManager
from scview.core.obs_writes import dataset_edit, save_edit, uns_snapshot
from scview.core.sse import keepalive
from scview.dependencies import get_dataset_manager

logger = logging.getLogger(__name__)

router = APIRouter()

MAX_ROWS = 2000  # rows sent to the browser, highest I first; the CSV has all
WHAT = "Spatially variable genes"


class AutocorrRequest(BaseModel):
    genes: Literal["hvg", "all"] | list[str] = "hvg"
    library_id: str | None = Field(default=None, max_length=256)
    layer: str | None = Field(default=None, max_length=256)


def _num(a, digits: int = 6) -> list:
    out: list = []
    for v in np.asarray(a, dtype=np.float64).tolist():
        out.append(None if not np.isfinite(v) else float(f"{v:.{digits}g}"))
    return out


BY_LIBRARY = ("I", "z", "pval", "pval_adj")


def _by_library(r: sa.Result, genes: list[str]) -> list[dict[str, dict[str, float | None]]]:
    """Per row, each section's own values: {library_id: {I, z, pval, pval_adj}} (a section
    where the gene is constant is left out)."""
    by = r.by_library
    out: list[dict[str, dict[str, float | None]]] = [{} for _ in genes]
    if by is None or by.empty:
        return out
    where = {g: i for i, g in enumerate(genes)}
    sub = by[by["gene"].astype(str).isin(where)]
    cols = {c: _num(sub[c]) for c in BY_LIBRARY}
    for j, (lib, g) in enumerate(zip(sub["library_id"].astype(str), sub["gene"].astype(str), strict=True)):
        out[where[g]][lib] = {c: cols[c][j] for c in BY_LIBRARY}
    return out


def result_body(r: sa.Result) -> dict[str, Any]:
    t = r.table.head(MAX_ROWS)
    cols = {c: _num(t[c]) for c in ("I", "expected_I", "var_I", "z", "pval", "pval_adj")}
    genes = [str(g) for g in t["gene"]]
    rows = [{"gene": g, **{c: cols[c][i] for c in cols}} for i, g in enumerate(genes)]
    pooled = r.centring == "section"
    if pooled:
        for row, by in zip(rows, _by_library(r, genes), strict=True):
            row["by_library"] = by
    return {
        "result_id": r.id, "method": "moran", "n_obs": r.n_obs, "n_genes": r.n_genes,
        "neighbors": r.neighbors, "libraries": r.libraries, "genes": r.genes_choice,
        "matrix_source": r.matrix_source, "unit": r.unit, "n_constant": r.n_constant,
        "library_id": sa.section_of(r), "centring": r.centring, "sections": r.sections,
        "notes": r.notes, "rows_shown": len(rows), **sa.summary_counts(r),
        "elapsed_seconds": r.elapsed_seconds, "created_at": r.created_at, "rows": rows,
    }


async def _adaptor(dm: DatasetManager, dataset_id: str):
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")
    return adaptor


NO_SPATIAL = ("This dataset has no spatial coordinates: spatially variable genes need spot or cell "
              "positions (obsm['spatial'], e.g. a Space Ranger folder).")


class _Stopped(Exception):
    """The client went away: the run stops before its next block of genes, nothing saved."""


async def _run_and_save(dm: DatasetManager, dataset_id: str, body: AutocorrRequest, held, progress=None) -> dict:
    """The run under the dataset's edit lock (``held``), saved with the dataset; the result's body."""
    adaptor = await _adaptor(dm, dataset_id)
    adata = adaptor.adata
    if not sa.has_spatial(adata):
        raise HTTPException(status_code=422, detail=NO_SPATIAL)
    output_path = str(dm.derived_h5ad_path(dataset_id, adaptor.h5ad_path))
    await dm.reserve(dataset_id, sa.need_bytes(adaptor), what=WHAT)

    def _run() -> sa.Result:
        r = sa.run(adata, genes=body.genes, library_id=body.library_id, layer=body.layer,
                   dataset_id=dataset_id, progress=progress)
        before = uns_snapshot(adata)
        sa.record(adata, r, body.genes, body.library_id, body.layer)
        save_edit(adaptor, adata, output_path, before)
        return r

    # A backed dataset's matrix (the section's spots) is read: room for it held until the run ends.
    need = await asyncio.to_thread(sa.read_bytes, adaptor, body.library_id, body.layer)
    async with working_copy.reading(dm, dataset_id, need, working_copy.subject(dm, dataset_id, need, WHAT)) as reading:
        try:
            r = await reading.run_with(held, _run)
        except sa.SpatialAutocorrError as e:
            raise HTTPException(status_code=422, detail=str(e))
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e).strip("'\""))
        except _Stopped:
            raise
        except Exception as e:
            dm.evict(dataset_id)  # the cached copy may hold the step without it being saved
            logger.error("Moran's I failed for %s: %s", dataset_id, e, exc_info=True)
            raise HTTPException(status_code=500, detail=f"Could not compute spatially variable genes: {e!s}")
    dm.saved_in_place(dataset_id, adaptor, output_path)
    sa.remember(adaptor, r)  # after the step is recorded: stored as current
    return result_body(r)


@router.post("/datasets/{dataset_id}/spatial/autocorr")
async def run_autocorr(
    dataset_id: str,
    body: AutocorrRequest,
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """Moran's I for the chosen genes; saved with the dataset and kept for GET."""
    async with dataset_edit(dataset_id) as held:
        return await _run_and_save(dm, dataset_id, body, held)


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@router.post("/datasets/{dataset_id}/spatial/autocorr/run-stream")
async def run_autocorr_stream(
    dataset_id: str,
    body: AutocorrRequest,
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """The run as SSE: ``progress`` (genes done, of a large dataset's blocks), then ``result``
    (as POST /spatial/autocorr answers) and ``complete``, or ``error``. A client that goes
    away stops the run before its next block of genes and nothing is saved."""

    async def events():
        loop = asyncio.get_running_loop()
        updates: asyncio.Queue = asyncio.Queue()
        stop = threading.Event()

        def progress(done: int, total: int) -> None:
            if stop.is_set():
                raise _Stopped
            loop.call_soon_threadsafe(updates.put_nowait, {"done": done, "total": total})

        try:
            async with dataset_edit(dataset_id) as held:
                work = asyncio.ensure_future(_run_and_save(dm, dataset_id, body, held, progress))
                try:
                    while not work.done():
                        waiter = asyncio.ensure_future(updates.get())
                        await asyncio.wait({work, waiter}, return_when=asyncio.FIRST_COMPLETED)
                        if waiter.done():
                            yield _sse("progress", waiter.result())
                        else:
                            waiter.cancel()
                    result = work.result()
                finally:
                    stop.set()
                    if not work.done():
                        work.cancel()
            yield _sse("result", result)
            yield _sse("complete", {})
        except HTTPException as e:
            yield _sse("error", {"error": str(e.detail)})
        except _Stopped:
            return
        except Exception as e:  # noqa: BLE001 - said to the client, the server stays up
            logger.error("Moran's I stream failed for %s: %s", dataset_id, e, exc_info=True)
            yield _sse("error", {"error": f"Could not compute spatially variable genes: {e!s}"})

    return StreamingResponse(keepalive(events()), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.get("/datasets/{dataset_id}/spatial/autocorr")
async def last_autocorr(dataset_id: str, allow_empty: bool = False,
                        library_id: str | None = Query(default=None, max_length=256),
                        dm: DatasetManager = Depends(get_dataset_manager)):
    """The last run on this dataset, or 404; with ``allow_empty`` null instead (the panel asks on open,
    and a 404 is a console error in the browser). ``library_id``: the last run on that section,
    "all" the last run over every section."""
    adaptor = await _adaptor(dm, dataset_id)
    r = await asyncio.to_thread(sa.last_for_section, adaptor, library_id) if library_id else sa.last(adaptor)
    if r is None:
        if allow_empty:
            return None
        raise HTTPException(status_code=404, detail="No spatially variable genes computed for this dataset yet.")
    return result_body(r)


@router.get("/datasets/{dataset_id}/spatial/autocorr/{result_id}.csv")
async def autocorr_csv(dataset_id: str, result_id: str, dm: DatasetManager = Depends(get_dataset_manager)):
    """Every tested gene of a kept run, highest I first."""
    adaptor = await _adaptor(dm, dataset_id)
    r = sa.last(adaptor)
    if r is None or r.id != result_id:  # an older run, or the store has it only
        r = results_store.recall(dataset_id, result_id, sa.KIND) if results_store.safe_id(result_id) else None
    if r is None:
        raise HTTPException(status_code=404, detail="That result is no longer kept on the server: run it again.")
    table = r.table
    if r.by_library is not None and not r.by_library.empty:  # each section's own I and adjusted p beside it
        wide = r.by_library.pivot(index="gene", columns="library_id", values=["I", "pval_adj"])
        wide.columns = [f"{c}_{lib}" for c, lib in wide.columns]
        table = table.join(wide, on="gene")
    buf = io.StringIO()
    table.to_csv(buf, index=False)
    return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv", headers={
        "Content-Disposition": f'attachment; filename="moran_{result_id}.csv"'})
