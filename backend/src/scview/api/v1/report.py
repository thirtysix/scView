"""Cluster and gene reports (C064; ``core/report.py``).

``GET /datasets/{id}/report/cluster?column=&value=`` and
``GET /datasets/{id}/report/gene?gene=&groupby=`` return the report as JSON
sections; the same paths with ``.html`` return it as one self-contained HTML file
to download (``core/report_html.py``). Both read the dataset only: nothing is
saved, no database is downloaded.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response

from scview.api.v1.export import file_stem
from scview.config import Settings
from scview.core import report, report_html
from scview.core.dataset_manager import DatasetManager
from scview.core.msigdb_loader import get_msigdb_loader
from scview.dependencies import get_dataset_manager, get_settings_dep

logger = logging.getLogger(__name__)
router = APIRouter()


async def _adaptor(dm: DatasetManager, dataset_id: str):
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    return adaptor


def _sources(dm: DatasetManager, dataset_id: str, settings: Settings) -> dict:
    info = dm.get_dataset_info(dataset_id) or {}
    return {"dataset_name": str(info.get("name") or dataset_id),
            "msigdb_loader": get_msigdb_loader(settings.MSIGDB_DIR),
            "marker_db_dir": Path(settings.DATA_DIR) / "marker_db"}


async def _run(fn, *args, **kwargs) -> dict:
    try:
        # Markers, QC, ORA and co-expression together take a second or more: off the loop.
        return await asyncio.to_thread(fn, *args, **kwargs)
    except report.ReportError as exc:
        detail = {"message": str(exc), "suggestion": exc.suggestion} if exc.suggestion else str(exc)
        raise HTTPException(status_code=exc.status, detail=detail)


def _html(body: dict, stem: str) -> Response:
    page = report_html.render(body)
    return Response(content=page, media_type="text/html; charset=utf-8",
                    headers={"Content-Disposition": f'attachment; filename="{stem}.html"'})


@router.get("/datasets/{dataset_id}/report/cluster")
async def cluster_report(
    dataset_id: str,
    column: str = Query(..., min_length=1, description="The grouping column"),
    value: str = Query(..., description="The group to report on"),
    dm: DatasetManager = Depends(get_dataset_manager),
    settings: Settings = Depends(get_settings_dep),
):
    """One group's report: identity, labels, markers (+ marker databases),
    HALLMARK enrichment, QC, composition and methods, each section with a caption
    (with n), a method line and a status (ok / not_computed / not_applicable)."""
    adaptor = await _adaptor(dm, dataset_id)
    return await _run(report.cluster_report, adaptor, column, value, **_sources(dm, dataset_id, settings))


@router.get("/datasets/{dataset_id}/report/cluster.html")
async def cluster_report_html(
    dataset_id: str,
    column: str = Query(..., min_length=1),
    value: str = Query(...),
    dm: DatasetManager = Depends(get_dataset_manager),
    settings: Settings = Depends(get_settings_dep),
):
    """The cluster report as a self-contained HTML file (a download)."""
    adaptor = await _adaptor(dm, dataset_id)
    src = _sources(dm, dataset_id, settings)
    body = await _run(report.cluster_report, adaptor, column, value, **src)
    return _html(body, f"{file_stem(src['dataset_name'])}_cluster_{file_stem(value, 'group')}_report")


@router.get("/datasets/{dataset_id}/report/gene")
async def gene_report(
    dataset_id: str,
    gene: str = Query(..., min_length=1),
    groupby: str = Query("", description="Column to split the cells by (default: the active clustering)"),
    dm: DatasetManager = Depends(get_dataset_manager),
    settings: Settings = Depends(get_settings_dep),
):
    """One gene's report: expression per group, marker status, marker databases,
    per-donor pseudobulk, co-expressed genes, HVG status, MSigDB sets, protein and
    methods. An unknown gene is a 404 whose detail carries a close match
    (``{"message", "suggestion"}``) when there is one."""
    adaptor = await _adaptor(dm, dataset_id)
    return await _run(report.gene_report, adaptor, gene, groupby or None, **_sources(dm, dataset_id, settings))


@router.get("/datasets/{dataset_id}/report/gene.html")
async def gene_report_html(
    dataset_id: str,
    gene: str = Query(..., min_length=1),
    groupby: str = Query(""),
    dm: DatasetManager = Depends(get_dataset_manager),
    settings: Settings = Depends(get_settings_dep),
):
    """The gene report as a self-contained HTML file (a download)."""
    adaptor = await _adaptor(dm, dataset_id)
    src = _sources(dm, dataset_id, settings)
    body = await _run(report.gene_report, adaptor, gene, groupby or None, **src)
    return _html(body, f"{file_stem(src['dataset_name'])}_gene_{file_stem(body['gene'], 'gene')}_report")
