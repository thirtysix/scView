"""Marker off-target expression (``core/offtarget.py``): where else in the body a
group's markers are expressed, from the Human Protein Atlas.

``GET /datasets/{id}/markers/offtarget`` looks up a group's markers (the one marker
choice, ``marker_selection.group_markers``); ``POST /offtarget/genes`` an explicit
list (e.g. a saved gene list). ``GET /offtarget/status`` says what is downloaded.
The HPA tables are fetched on first use (a 503 says why when they cannot be).
Reads no expression from the dataset, so a large dataset needs nothing special.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from scview.api.v1.markers import ensure_specificity
from scview.config import Settings
from scview.core import offtarget as ot
from scview.core.dataset_manager import DatasetManager
from scview.dependencies import get_dataset_manager, get_settings_dep

router = APIRouter()


class GenesRequest(BaseModel):
    genes: list[str] = Field(..., min_length=1, max_length=ot.MAX_GENES)
    species: Literal["human", "mouse"] = "human"
    level: Literal["tissue", "region", "organ"] = "tissue"


def _dest(settings: Settings) -> Path:
    return Path(settings.DATA_DIR) / "hpa"  # the data volume


@router.get("/offtarget/status")
async def offtarget_status(settings: Settings = Depends(get_settings_dep)):
    """Which HPA tables are on disk (URL, version, licence, sha256, sizes)."""
    st = await asyncio.to_thread(ot.status, _dest(settings))
    st["auto_download"] = settings.HPA_AUTO_DOWNLOAD
    return st


@router.post("/offtarget/genes")
async def offtarget_genes(body: GenesRequest, settings: Settings = Depends(get_settings_dep)):
    """HPA tissue and cell-type specificity of an explicit gene list."""
    try:
        # first use downloads and converts the tables: off the loop
        return await asyncio.to_thread(ot.lookup, body.genes, species=body.species, dest=_dest(settings),
                                       auto_download=settings.HPA_AUTO_DOWNLOAD, level=body.level)
    except ot.HpaUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@router.get("/datasets/{dataset_id}/markers/offtarget")
async def offtarget_group(
    dataset_id: str,
    group_col: str,
    group: str,
    n: int = Query(ot.N_DEFAULT, ge=1, le=ot.N_MAX),
    level: Literal["tissue", "region", "organ"] = "tissue",
    dm: DatasetManager = Depends(get_dataset_manager),
    settings: Settings = Depends(get_settings_dep),
):
    """A group's markers (tissues at ``level``: tissue, region or organ) with their in-dataset tau, HPA category, top tissues and
    cell types, a verdict and a flag when specific here but broad in HPA."""
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    if group_col not in adaptor.adata.obs.columns:
        raise HTTPException(status_code=404, detail=f"Column '{group_col}' not found.")
    # Markers saved before the specificity columns get their tau now, as in the table.
    adaptor = await ensure_specificity(dm, dataset_id, adaptor, group_col)
    adata = adaptor.adata
    try:
        res = await asyncio.to_thread(ot.for_group, adata, group_col, group, n, dest=_dest(settings),
                                      auto_download=settings.HPA_AUTO_DOWNLOAD, level=level)
    except ot.HpaUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except LookupError as exc:  # no markers stored (KeyError: unknown group)
        code = 404 if isinstance(exc, KeyError) else 422
        raise HTTPException(status_code=code, detail=str(exc.args[0]) if exc.args else str(exc))
    ot.remember(adaptor, res)
    return res
