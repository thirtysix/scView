"""Peak-gene links endpoints (core/atac_links.py).

``POST .../atac/links/estimate`` says what a run would need; ``POST .../atac/links`` runs it, streamed
(``start``, ``progress``, ``watchdog``, ``result``, ``complete``, or ``error``), in the stream every
ATAC run shares (``api/v1/atac.py``), one run per dataset; ``GET .../atac/links``: the last result on
these cells and what is kept; ``GET .../atac/links/{id}``: a page, by gene or searched;
``.../csv``; ``.../locus?gene=``: one gene's arcs with the peaks and genes in their window.
"""

from __future__ import annotations

import asyncio
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from scview.api.v1.atac import MAX_PAGE, _adaptor, _auto_download, _events, _http, _refuse_running, _save, _stream, _WHAT
from scview.config import get_settings
from scview.core import atac, atac_links, genome_seq, results_store
from scview.core.dataset_manager import DatasetManager
from scview.dependencies import get_dataset_manager

router = APIRouter()

_SORTS = (*atac_links.SORTS, "abs_score", "abs_distance")


class LinksRequest(BaseModel):
    genes: Literal["variable", "all", "list"] = "variable"
    gene_list: list[str] | None = None  # the plan says when it is too long
    cells_column: str | None = Field(default=None, max_length=256)
    cells_value: list[str] | None = Field(default=None, max_length=atac.MAX_GROUPS)


def _genome_auto() -> bool:
    return bool(get_settings().GENOME_AUTO_DOWNLOAD)


def _estimate(adaptor, p: atac_links.LinkPlan, store, data_dir) -> dict[str, Any]:
    nnz = int(store.nnz(atac.MATRIX) * (p.n_cells / max(1, store.n_obs)))
    status = genome_seq.status(data_dir, p.genome)
    seconds = atac_links.estimate_seconds(nnz, store.n_vars, len(p.genes), mode=p.mode,
                                           cell_share=p.n_cells / max(1, p.n_dataset_cells))
    if not status["downloaded"]:
        seconds += atac_links.DOWNLOAD_SECONDS
    return {"genes": p.mode, "n_genes": len(p.genes), "n_asked": p.n_asked, "missing": p.missing[:20], "n_cells": p.n_cells,
            "n_dataset_cells": p.n_dataset_cells, "n_peaks": store.n_vars, "estimate_seconds": seconds,
            "estimate_bytes": atac_links.estimate_bytes(nnz, store.n_vars, len(p.genes), p.n_cells),
            "estimate_disk_bytes": atac_links.estimate_disk_bytes(nnz), "genome": status}


def _plan(adaptor, dm, store, body: LinksRequest) -> atac_links.LinkPlan:
    return atac_links.plan_links(adaptor, dm.data_dir, genome=store.meta.get("genome"), mode=body.genes, gene_list=body.gene_list,
                                 cells_column=body.cells_column, cells_value=body.cells_value, auto_download=_auto_download())


@router.post("/datasets/{dataset_id}/atac/links/estimate")
async def links_estimate(dataset_id: str, body: LinksRequest, dm: DatasetManager = Depends(get_dataset_manager)):
    """What a run with these choices would need, before it runs; states a first download. A POST: a pasted gene
    list would not fit in a URL behind nginx."""
    adaptor = await _adaptor(dm, dataset_id)

    def work() -> dict[str, Any]:
        with atac.open_store(adaptor.adata, dm.data_dir, dataset_id) as store:
            return _estimate(adaptor, _plan(adaptor, dm, store, body), store, dm.data_dir)

    try:
        return await asyncio.to_thread(work)
    except atac.AtacError as e:
        raise _http(e)


@router.post("/datasets/{dataset_id}/atac/links")
async def run_links(dataset_id: str, body: LinksRequest, dm: DatasetManager = Depends(get_dataset_manager)):
    """The links, streamed. The first use of a genome downloads it inside the stream."""
    adaptor = await _adaptor(dm, dataset_id)
    _refuse_running(dataset_id)
    adata = adaptor.adata
    try:
        store = await asyncio.to_thread(atac.open_store, adata, dm.data_dir, dataset_id)
    except atac.AtacError as e:
        raise _http(e)
    try:
        p = await asyncio.to_thread(_plan, adaptor, dm, store, body)
    except atac.AtacError as e:
        store.close()
        raise _http(e)
    fingerprint = await asyncio.to_thread(adaptor.cells_fingerprint)
    start = await asyncio.to_thread(_estimate, adaptor, p, store, dm.data_dir)
    need = start["estimate_bytes"]
    try:
        await dm.reserve(dataset_id, need, what=_WHAT["links"])
    except Exception:
        store.close()
        raise
    del adata

    async def on_result(item, res):
        kept = await asyncio.to_thread(atac_links.build_result, p, item, body.gene_list)
        resources = {"peak_gb": res["peak_gb"], "limit_gb": res["limit_gb"], "seconds": kept.seconds}
        await _save(dm, dataset_id, fingerprint, (), (), "the peak-gene links",
                    lambda data: atac_links.record(data, kept, resources))
        a = await _adaptor(dm, dataset_id)
        atac_links.remember(a, dataset_id, kept)
        return {**atac_links.payload(kept), "resources": res}

    def open_gen(stop):
        return atac_links.iter_run(adaptor, store, p, data_dir=dm.data_dir, auto_download=_genome_auto(), stop=stop)

    return _stream(_events(dm, dataset_id, "links", start, need, open_gen, on_result, "Peak-gene links", closers=[store.close]))


@router.get("/datasets/{dataset_id}/atac/links")
async def latest_links(dataset_id: str, limit: int = Query(default=100, ge=1, le=MAX_PAGE),
                       dm: DatasetManager = Depends(get_dataset_manager)):
    """The last result on these cells (null when none), and every kept run with whether it still applies."""
    adaptor = await _adaptor(dm, dataset_id)

    def work() -> dict[str, Any]:
        results_store.flush()  # a run that just ended is on its way to disk
        kept = atac_links.latest(adaptor)
        return {"result": atac_links.payload(kept, limit=limit) if kept is not None else None, "runs": atac_links.kept_runs(adaptor)}

    return await asyncio.to_thread(work)


def _recall(dataset_id: str, result_id: str) -> atac_links.LinksResult:
    kept = atac_links.recall(dataset_id, result_id)
    if kept is None:
        raise HTTPException(status_code=404, detail="That peak-gene links result is gone. Run it again.")
    return kept


@router.get("/datasets/{dataset_id}/atac/links/{result_id}")
async def get_links(
    dataset_id: str, result_id: str,
    gene: str = Query(default="", max_length=256),
    search: str = Query(default="", max_length=256),
    sort: str = Query(default="p"),
    descending: bool = False,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=MAX_PAGE),
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """A page of a kept result, all of it or one gene's; ``stale`` when the cells or the data changed since."""
    if sort not in _SORTS:
        raise HTTPException(status_code=422, detail=f"Links cannot be sorted by '{sort}'. Use one of: {', '.join(_SORTS)}.")
    adaptor = await _adaptor(dm, dataset_id)

    def work() -> dict[str, Any]:
        kept = _recall(dataset_id, result_id)
        stale = next((r["stale"] for r in atac_links.kept_runs(adaptor) if r["id"] == result_id), False) \
            or not atac_links.is_current(adaptor, kept)
        return atac_links.payload(kept, gene=gene, search=search, sort=sort, descending=descending, offset=offset, limit=limit,
                                  stale=stale)

    return await asyncio.to_thread(work)


@router.get("/datasets/{dataset_id}/atac/links/{result_id}/csv")
async def links_csv(dataset_id: str, result_id: str, dm: DatasetManager = Depends(get_dataset_manager)):
    await _adaptor(dm, dataset_id)

    def text() -> str:
        return atac_links.csv_text(_recall(dataset_id, result_id))

    body = await asyncio.to_thread(text)
    return StreamingResponse(iter([body]), media_type="text/csv", headers={
        "Content-Disposition": f'attachment; filename="peak_gene_links_{result_id}.csv"'})


@router.get("/datasets/{dataset_id}/atac/links/{result_id}/locus")
async def links_locus(dataset_id: str, result_id: str, gene: str = Query(max_length=256),
                      dm: DatasetManager = Depends(get_dataset_manager)):
    adaptor = await _adaptor(dm, dataset_id)

    def work() -> dict[str, Any]:
        kept = _recall(dataset_id, result_id)
        with atac.open_store(adaptor.adata, dm.data_dir, dataset_id) as store:
            return atac_links.locus(kept, store, store.meta.get("genome"), dm.data_dir, gene, _auto_download())

    try:
        return await asyncio.to_thread(work)
    except atac.AtacError as e:
        raise _http(e)
