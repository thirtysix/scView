"""Cell Ontology terms for labels (``core/cell_ontology.py``).

``GET /ontology/cell-types?q=`` searches the Cell Ontology (fetched on first use with
the marker databases' ``cl-basic.obo``; 503 when it is not on disk and cannot be fetched).
``GET|POST /datasets/{id}/ontology/terms`` reads or sets a label column's label -> term
mapping (kept in uns, saved like any obs edit); ``POST .../terms/suggest`` proposes terms
for every label, which are shown and never stored.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from scview.core import cell_ontology as co
from scview.core.dataset_manager import DatasetManager
from scview.core.obs_writes import dataset_edit, edit_room, editable, save_edit, uns_snapshot
from scview.dependencies import get_dataset_manager

logger = logging.getLogger(__name__)

router = APIRouter()

MAX_QUERY = 120
MAX_CHANGES = 500


async def _index() -> co.Index:
    try:
        return await asyncio.to_thread(co.get_index)
    except co.OntologyUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc))


def _local_index() -> co.Index | None:
    """The index if the ontology is on disk (no fetching): to name and check terms."""
    try:
        return co.loaded()
    except (OSError, ValueError):
        return None


@router.get("/ontology/cell-types")
async def search_cell_types(
    q: str = Query("", max_length=MAX_QUERY),
    limit: int = Query(20, ge=1, le=co.MAX_RESULTS),
):
    """Cell Ontology terms for ``q`` (a name, synonym, part of one, or a CL id), best first."""
    idx = await _index()
    terms = await asyncio.to_thread(idx.search, q, limit)
    return {"query": q, "terms": terms, "version": idx.version, "n_terms": len(idx)}


async def _adaptor(dm: DatasetManager, dataset_id: str):
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    return adaptor


@router.get("/datasets/{dataset_id}/ontology/terms")
async def get_label_terms(
    dataset_id: str,
    column: str = Query(...),
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """Each label of ``column`` with the term the user gave it, and whether all have one."""
    adaptor = await _adaptor(dm, dataset_id)
    idx = await asyncio.to_thread(_local_index)
    try:
        out = await asyncio.to_thread(co.summary, adaptor.adata, column, idx)
    except co.TermError as exc:
        raise HTTPException(status_code=exc.status, detail=str(exc))
    out["columns"] = co.mapped_columns(adaptor.adata)
    return out


class TermsBody(BaseModel):
    column: str
    terms: dict[str, str | None] = {}  # label -> CL id, "unknown", or None to clear
    clear: bool = False  # clear every term of the column


@router.post("/datasets/{dataset_id}/ontology/terms")
async def set_label_terms(
    dataset_id: str,
    body: TermsBody,
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """Set or clear terms of a column's labels (user-confirmed only; saved to the
    derived layer, never the original upload). Answers with the new mapping and each
    changed label's previous term (``previous``)."""
    if len(body.terms) > MAX_CHANGES:
        raise HTTPException(status_code=400, detail=f"At most {MAX_CHANGES} labels at once.")
    idx = await asyncio.to_thread(_local_index)
    async with dataset_edit(dataset_id) as held:
        adaptor = await _adaptor(dm, dataset_id)
        output_path = str(dm.derived_h5ad_path(dataset_id, adaptor.h5ad_path))
        backed = await edit_room(dm, dataset_id, adaptor)

        def _run() -> dict[str, Any]:
            adata = editable(adaptor)
            before = uns_snapshot(adata)
            changes: dict[str, Any] = dict(body.terms)
            if body.clear:
                changes = {label: None for label in co.stored(adata, body.column)
                           if label in set(co.labels_of(adata, body.column)[0])}
            previous = co.set_terms(adata, body.column, changes, idx)
            if previous:
                save_edit(adaptor, adata, output_path, before)
            return {"previous": previous, "summary": co.summary(adata, body.column, idx),
                    "columns": co.mapped_columns(adata)}

        try:
            result = await held.run(_run)
        except co.TermError as exc:  # raised before anything changed
            raise HTTPException(status_code=exc.status, detail=str(exc))
        except Exception as exc:
            dm.evict(dataset_id)  # may hold an edit that was never saved
            logger.error("Saving Cell Ontology terms failed for %s: %s", dataset_id, exc, exc_info=True)
            raise HTTPException(status_code=500, detail=f"Could not save the terms: {exc!s}")
        if result["previous"]:
            if backed:
                dm.evict(dataset_id)
                await dm.load_dataset(dataset_id)
            else:
                dm.saved_in_place(dataset_id, adaptor, output_path)
    return {**result["summary"], "previous": result["previous"], "columns": result["columns"]}


class SuggestBody(BaseModel):
    column: str


@router.post("/datasets/{dataset_id}/ontology/terms/suggest")
async def suggest_label_terms(
    dataset_id: str,
    body: SuggestBody,
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """Up to three terms per label of ``column`` from its name (the marker databases'
    name matcher plus narrower terms). Nothing is stored: the user accepts each one."""
    adaptor = await _adaptor(dm, dataset_id)
    idx = await _index()
    try:
        return await asyncio.to_thread(co.suggest, adaptor.adata, body.column, idx)
    except co.TermError as exc:
        raise HTTPException(status_code=exc.status, detail=str(exc))
