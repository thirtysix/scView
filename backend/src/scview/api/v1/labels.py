"""Hand labels for selected cells (``core/user_labels.py``).

``POST /labels`` labels cells and answers with their previous values; the
frontend keeps those on an undo stack and sends one back to ``POST /labels/undo``.
Both save the derived file, so the dataset is evicted and reloaded afterwards.
``POST /labels/import`` (JSON) and ``/labels/import-file`` (a CSV/TSV upload) write
a per-cell label table into a user column (``core/label_import.py``).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Literal

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, ConfigDict, Field

from scview.core import cell_ontology, label_import
from scview.core.dataset_manager import DatasetManager
from scview.core.obs_writes import dataset_edit, edit_room, editable, save_edit, uns_snapshot
from scview.core.user_labels import (
    COMBINE_SEPARATOR, UNASSIGNED, LabelError, combine_columns, label_cells, restore, user_columns,
)
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


class CombineRequest(BaseModel):
    columns: list[str]  # 2 to 4 obs columns, in the order their values are joined
    name: str  # the new user annotation column
    separator: str = COMBINE_SEPARATOR


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
            # Labels change obs and uns only: the matrices are copied, not rewritten.
            save_edit(adaptor, work, output_path, before, source_unchanged=True)
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


@router.post("/datasets/{dataset_id}/labels/combine")
async def combine_label_columns(
    dataset_id: str,
    body: CombineRequest,
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """A new user annotation column from two to four obs columns, each cell's values
    joined ("HFD01 | ING"): a sample id from donor and depot in one call. A cell
    missing a value in any of them gets none."""
    return await _edit_and_save(
        dm, dataset_id, lambda adata: combine_columns(adata, body.columns, body.name, body.separator))


class ImportRequest(BaseModel):
    """One of ``labels`` (barcode -> label), ``values`` (one label per cell, in order) or
    ``table`` (CSV/TSV text with a header row)."""

    model_config = ConfigDict(extra="forbid")

    column: str
    labels: dict[str, str | None] | None = None
    values: list[str | None] | None = None
    table: str | None = None
    filename: str | None = Field(None, max_length=255)  # the table's file name, for the record
    barcode_column: str | None = None  # table: default its first column
    label_column: str | None = None  # table: default its one other column
    match: Literal["name", "order"] = "name"  # table: by barcode, or row i = cell i
    new_column: bool = True
    strip_suffix: bool = False
    min_match: float = Field(label_import.MIN_MATCH, ge=0, le=1)
    dry_run: bool = False


def _table_of(body: ImportRequest) -> label_import.LabelTable:
    given = [k for k in ("labels", "values", "table") if getattr(body, k) is not None]
    if len(given) != 1:
        raise LabelError("Give one of labels (barcode -> label), values (one per cell, in order) or table.")
    if body.labels is not None:
        return label_import.LabelTable(labels=list(body.labels.values()), barcodes=list(body.labels),
                                       source={"input": "mapping"})
    if body.values is not None:
        return label_import.LabelTable(labels=list(body.values), source={"input": "values"})
    return label_import.read_table(body.table.encode("utf-8"), body.filename or "", body.barcode_column,
                                   body.label_column, by_order=body.match == "order")


async def _import(dm: DatasetManager, dataset_id: str, make_table: Callable[[], label_import.LabelTable],
                  body_opts: dict[str, Any]) -> dict[str, Any]:
    try:
        table = await asyncio.to_thread(make_table)  # a big table parses off the event loop
    except LabelError as exc:
        raise HTTPException(status_code=exc.status, detail=str(exc))

    def edit(adata):
        return label_import.import_labels(adata, body_opts["column"], table, **{
            k: body_opts[k] for k in ("new_column", "strip_suffix", "min_match", "dry_run")})

    if body_opts["dry_run"]:
        adaptor = await dm.get_or_load_dataset(dataset_id)
        if adaptor is None:
            raise HTTPException(status_code=404, detail="Dataset not found.")
        try:
            return await asyncio.to_thread(edit, adaptor.adata)
        except LabelError as exc:
            raise HTTPException(status_code=exc.status, detail=str(exc))
    return await _edit_and_save(dm, dataset_id, edit)


@router.post("/datasets/{dataset_id}/labels/import")
async def import_label_table(
    dataset_id: str,
    body: ImportRequest,
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """A per-cell label table into a user annotation column, matched by cell name (or by
    order with one row per cell): says how many rows matched, the rows naming no cell
    (``n_extra``) and the cells left out (``n_unmatched_cells``); refused when under
    ``min_match`` of the labelled rows name a cell. ``dry_run`` only reports."""
    return await _import(dm, dataset_id, lambda: _table_of(body), body.model_dump())


@router.post("/datasets/{dataset_id}/labels/import-file")
async def import_label_file(
    dataset_id: str,
    file: UploadFile = File(...),
    column: str = Form(...),
    barcode_column: str | None = Form(None),
    label_column: str | None = Form(None),
    match: Literal["name", "order"] = Form("name"),
    new_column: bool = Form(True),
    strip_suffix: bool = Form(False),
    min_match: float = Form(label_import.MIN_MATCH, ge=0, le=1),
    dry_run: bool = Form(False),
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """``/labels/import`` with the table as a CSV/TSV upload (gzipped too), e.g.
    ``curl -F file=@labels.csv -F column=author_fine .../labels/import-file``."""
    data = await file.read(label_import.MAX_TABLE_BYTES + 1)
    if len(data) > label_import.MAX_TABLE_BYTES:
        raise HTTPException(status_code=413, detail=(
            f"The table is larger than {label_import.MAX_TABLE_BYTES // 2**20} MB."))
    name = (file.filename or "").rsplit("/", 1)[-1][:255]
    return await _import(
        dm, dataset_id,
        lambda: label_import.read_table(data, name, barcode_column or None, label_column or None,
                                        by_order=match == "order"),
        {"column": column, "new_column": new_column, "strip_suffix": strip_suffix, "min_match": min_match,
         "dry_run": dry_run})
