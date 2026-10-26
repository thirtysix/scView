"""Hand labels for selected cells (``core/user_labels.py``).

``POST /labels`` labels cells and answers with their previous values; the
frontend keeps those on an undo stack and sends one back to ``POST /labels/undo``.
Both save the derived file, so the dataset is evicted and reloaded afterwards.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from scview.core import cell_ontology
from scview.core.dataset_manager import DatasetManager
from scview.core.obs_writes import dataset_edit, edit_room, editable, save_edit, uns_snapshot
from scview.core.user_labels import UNASSIGNED, LabelError, label_cells, restore, user_columns
from scview.dependencies import get_dataset_manager

logger = logging.getLogger(__name__)

router = APIRouter()


class LabelRequest(BaseModel):
    column: str
    label: str
    indices: list[int]  # cell indices (a lasso or a legend selection)
    new_column: bool = False  # create ``column`` (every other cell "Unassigned")
    # A Cell Ontology term the user picked for the label (core/cell_ontology.py); None
    # leaves the label's term as it is.
    ontology_id: str | None = None


class PreviousGroup(BaseModel):
    value: str | None
    indices: list[int]


class UndoRequest(BaseModel):
    column: str
    previous: list[PreviousGroup]
    created: bool = False  # the undone labelling created the column
    # The labelling's ``previous_terms``: label -> the term it had before (None: none)
    previous_terms: dict[str, str | None] = {}


async def _edit_and_save(
    dm: DatasetManager, dataset_id: str, edit: Callable[[Any], dict[str, Any]]
) -> dict[str, Any]:
    """Run ``edit`` on the dataset, save it, and reload it, one request at a time."""
    async with dataset_edit(dataset_id) as held:
        adaptor = await dm.get_or_load_dataset(dataset_id)
        if adaptor is None:
            raise HTTPException(status_code=404, detail="Dataset not found.")
        output_path = str(dm.derived_h5ad_path(dataset_id, adaptor.h5ad_path))

        backed = await edit_room(dm, dataset_id, adaptor)

        def _run():
            work = editable(adaptor)
            before = uns_snapshot(work)
            result = edit(work)
            save_edit(adaptor, work, output_path, before)
            return result

        try:
            result = await held.run(_run)
        except LabelError as exc:
            # Raised before anything changed: the cached copy is still the file.
            raise HTTPException(status_code=exc.status, detail=str(exc))
        except Exception as exc:
            dm.evict(dataset_id)  # may hold an edit that was never saved
            logger.error("Labelling failed for %s: %s", dataset_id, exc, exc_info=True)
            raise HTTPException(status_code=500, detail=f"Could not save the labels: {exc!s}")
        if backed:
            dm.evict(dataset_id)
            await dm.load_dataset(dataset_id)
        else:
            # The cached dataset is the edited one and now matches the file.
            dm.saved_in_place(dataset_id, adaptor, output_path)
        return result


@router.get("/datasets/{dataset_id}/labels/columns")
async def list_label_columns(
    dataset_id: str,
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """The dataset's user annotation columns: the ones new labels can go into."""
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    return {"columns": user_columns(adaptor.adata), "unassigned": UNASSIGNED}


@router.post("/datasets/{dataset_id}/labels")
async def label_selection(
    dataset_id: str,
    body: LabelRequest,
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """Label the selected cells in a user annotation column, new or existing.

    Answers with each changed cell's previous value (``previous``), which
    ``/labels/undo`` takes back to undo this labelling. With ``ontology_id`` the label
    gets that Cell Ontology term; ``previous_terms`` holds what the undo puts back.
    """
    try:
        index = await asyncio.to_thread(cell_ontology.loaded) if body.ontology_id else None
    except (OSError, ValueError):
        index = None  # the id's form is still checked
    try:
        term = cell_ontology.clean_term(body.ontology_id, index)
    except cell_ontology.TermError as exc:
        raise HTTPException(status_code=exc.status, detail=str(exc))

    def edit(adata):
        out = label_cells(adata, body.column, body.label, body.indices, body.new_column)
        out["previous_terms"] = (
            cell_ontology.set_terms(adata, out["column"], {out["label"]: term}) if term else {})
        return out

    return await _edit_and_save(dm, dataset_id, edit)


@router.post("/datasets/{dataset_id}/labels/undo")
async def undo_label_selection(
    dataset_id: str,
    body: UndoRequest,
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """Give cells back the values a labelling replaced (its ``previous``)."""
    previous = [g.model_dump() for g in body.previous]
    try:
        terms = {k: cell_ontology.clean_term(v) for k, v in body.previous_terms.items()}
    except cell_ontology.TermError as exc:
        raise HTTPException(status_code=exc.status, detail=str(exc))

    def edit(adata):
        out = restore(adata, body.column, previous, body.created)
        if out["removed"]:
            cell_ontology.drop_column(adata, body.column)
        else:
            cell_ontology.restore_terms(adata, body.column, terms)
        return out

    return await _edit_and_save(dm, dataset_id, edit)
