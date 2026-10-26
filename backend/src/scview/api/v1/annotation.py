"""Cell-type annotation endpoints.

CellTypist models are tissue/system specific (there is no single universal model),
so the UI offers the catalog below and the user picks the one matching their sample.
The default, ``Immune_All_Low``, suits the common PBMC/immune case.

``/datasets/{id}/annotation/confidence`` checks how well each cell's label in a
column fits its expression neighbourhood, with split conformal prediction sets
(``core/annotation_confidence.py``): GET returns the last check of a column,
POST runs one, POST ``.../unknown`` copies the column with rejected cells set to
"Unknown".

``/datasets/{id}/annotation/crosscheck`` compares 2-6 label columns made by
different methods (``core/annotation_crosscheck.py``): GET lists the columns that
look like annotations, POST runs the comparison, POST ``.../save`` writes the
consensus and per-cell agreement as obs columns.

``/datasets/{id}/annotation/marker-db`` checks a label column against the
PanglaoDB + CellMarker 2.0 marker databases (``core/marker_db.py``): GET
``.../status`` says what is downloaded, POST ``.../check`` fetches them on first
use and returns each label's database support and the 3 best database types.
"""

from __future__ import annotations

import asyncio
import logging
from functools import lru_cache
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from scview.core import annotation_confidence as conf
from scview.core import annotation_crosscheck as xc
from scview.config import Settings
from scview.core import celltypist_fit as fit
from scview.core import marker_db as mdb
from scview.core.dataset_manager import DatasetManager
from scview.core.obs_writes import dataset_edit, edit_room, editable, save_edit, uns_snapshot
from scview.dependencies import get_dataset_manager, get_settings_dep

logger = logging.getLogger(__name__)
router = APIRouter()

DEFAULT_CELLTYPIST_MODEL = "Immune_All_Low.pkl"


class CellTypistModel(BaseModel):
    model: str
    description: str
    # human | mouse | other, from the name and description (core/celltypist_fit.py)
    species: str = "human"


class CellTypistModelsResponse(BaseModel):
    default: str
    models: list[CellTypistModel]


@lru_cache(maxsize=1)
def _model_catalog() -> list[tuple[str, str]]:
    """(name, description) for every available CellTypist model. Cached; may hit the
    network once to fetch the model index, then reads the local copy."""
    from celltypist import models as ct_models

    df = ct_models.models_description()
    return [(str(r["model"]), str(r["description"])) for _, r in df.iterrows()]


@router.get("/annotation/celltypist-models", response_model=CellTypistModelsResponse)
def list_celltypist_models() -> CellTypistModelsResponse:
    """List CellTypist models (name + description) for the annotation picker."""
    try:
        items = [CellTypistModel(model=m, description=d, species=fit.model_species(m, d))
                 for m, d in _model_catalog()]
    except Exception as e:  # offline / celltypist unavailable: still return the default
        logger.warning("Could not list CellTypist models (%s); returning default only.", e)
        items = [CellTypistModel(
            model=DEFAULT_CELLTYPIST_MODEL,
            description="immune sub-populations combined from 20 tissues of 18 studies",
        )]
    return CellTypistModelsResponse(default=DEFAULT_CELLTYPIST_MODEL, models=items)


@router.get("/datasets/{dataset_id}/annotation/celltypist")
async def celltypist_summary(
    dataset_id: str,
    column: str | None = None,
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """The columns CellTypist annotated, the dataset's species, and for ``column``
    the run's model, species and gene overlap with a per-label confidence table
    and its cautions (null when the column has no CellTypist confidence)."""
    adata = (await _adaptor(dm, dataset_id)).adata

    def _read():
        from scview.core.species import detect_species

        return {
            "columns": fit.run_columns(adata),
            "dataset_species": detect_species(adata).species,
            "result": fit.summarize(adata, column) if column else None,
        }

    return await asyncio.to_thread(_read)  # a median per label over every cell: off the loop


class ConfidenceRequest(BaseModel):
    column: str
    alpha: float = conf.DEFAULT_ALPHA
    classifier: str = conf.DEFAULT_CLASSIFIER


class UnknownRequest(BaseModel):
    column: str


async def _adaptor(dm: DatasetManager, dataset_id: str):
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    return adaptor


@router.get("/datasets/{dataset_id}/annotation/confidence")
async def annotation_confidence(
    dataset_id: str,
    column: str | None = None,
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """Whether the check can run here (and if not, why), its limits, the columns
    checked so far, and the last check of ``column`` (null when there is none)."""
    adata = (await _adaptor(dm, dataset_id)).adata

    def _read():
        try:
            rep, n_pcs = conf.representation(adata)
            reason = None
        except conf.ConfidenceError as exc:
            rep, n_pcs, reason = None, None, str(exc)
        checked = [c for c in conf.registry(adata) if conf.summary(adata, c) is not None]
        return {
            "available": reason is None,
            "reason": reason,
            "representation": rep,
            "n_pcs": n_pcs,
            "default_alpha": conf.DEFAULT_ALPHA,
            "min_alpha": conf.MIN_ALPHA,
            "max_alpha": conf.MAX_ALPHA,
            "min_label_cells": conf.MIN_LABEL_CELLS,
            "max_labels": conf.MAX_LABELS,
            "classifiers": list(conf.CLASSIFIERS),
            "note": conf.NOTE,
            "reference_note": conf.REFERENCE_NOTE,
            # Columns CellTypist annotated, which its probabilities can score: {column: model}
            "reference_columns": conf.reference_columns(adata),
            "checked_columns": checked,
            "result": conf.summary(adata, column) if column else None,
        }

    return await asyncio.to_thread(_read)  # hashing a column's labels: off the loop


async def _edit(dm: DatasetManager, dataset_id: str, edit, what: str):
    """Run ``edit(adata)`` in a worker thread holding the dataset's edit lock, then
    save the obs/uns change (core/obs_writes.write_derived)."""
    async with dataset_edit(dataset_id) as held:
        adaptor = await _adaptor(dm, dataset_id)
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
        except (conf.ConfidenceError, xc.CrosscheckError) as exc:  # raised before any change
            raise HTTPException(status_code=exc.status, detail=str(exc))
        except Exception as exc:
            dm.evict(dataset_id)  # the cached copy may hold columns that were never saved
            logger.error("%s failed for %s: %s", what, dataset_id, exc, exc_info=True)
            raise HTTPException(status_code=500, detail=f"{what} failed: {exc!s}")
        if backed:
            dm.evict(dataset_id)
            await dm.load_dataset(dataset_id)
        else:
            dm.saved_in_place(dataset_id, adaptor, output_path)
        return result


@router.post("/datasets/{dataset_id}/annotation/confidence")
async def run_annotation_confidence(
    dataset_id: str,
    body: ConfidenceRequest,
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """Check how well each cell's label in ``column`` fits at error rate ``alpha``,
    saving the flag, fit, set-size and competing-label columns."""
    return await _edit(
        dm, dataset_id,
        lambda adata: conf.assess(adata, body.column, body.alpha, body.classifier),
        "The confidence check",
    )


@router.post("/datasets/{dataset_id}/annotation/confidence/unknown")
async def annotation_confidence_unknown(
    dataset_id: str,
    body: UnknownRequest,
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """Copy ``column`` with every rejected cell (unsure or doesn't fit in its last
    check) set to "Unknown", as the check's own column."""
    return await _edit(
        dm, dataset_id,
        lambda adata: conf.make_unknown_column(adata, body.column),
        "The Unknown copy",
    )


class CrosscheckRequest(BaseModel):
    columns: list[str]
    # One of ``columns``; the first when left out
    reference: str | None = None


class CrosscheckSaveRequest(CrosscheckRequest):
    name: str = xc.DEFAULT_NAME


@router.get("/datasets/{dataset_id}/annotation/crosscheck")
async def annotation_crosscheck_info(
    dataset_id: str,
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """The columns that look like annotations (column -> the method that made
    it), the active clustering, the limits, and the saved cross-checks."""
    adaptor = await _adaptor(dm, dataset_id)
    return xc.info(adaptor.adata, adaptor.active_clustering_column())


@router.post("/datasets/{dataset_id}/annotation/crosscheck")
async def run_annotation_crosscheck(
    dataset_id: str,
    body: CrosscheckRequest,
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """Compare 2-6 label columns: pairwise ARI/AMI, the reference against each
    other column (contingency, best matches, label mapping), per-label agreement
    and the consensus counts. Changes nothing."""
    adata = (await _adaptor(dm, dataset_id)).adata
    try:
        # contingency tables and AMI over every cell: off the loop
        return await asyncio.to_thread(xc.crosscheck, adata, body.columns, body.reference)
    except xc.CrosscheckError as exc:
        raise HTTPException(status_code=exc.status, detail=str(exc))


@router.post("/datasets/{dataset_id}/annotation/crosscheck/save")
async def save_annotation_crosscheck(
    dataset_id: str,
    body: CrosscheckSaveRequest,
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """Write ``{name}_consensus`` and ``{name}_agreement`` to obs."""
    return await _edit(
        dm, dataset_id,
        lambda adata: xc.save(adata, body.columns, body.reference, body.name),
        "Saving the cross-check",
    )


class MarkerDbCheckRequest(BaseModel):
    column: str
    # The dataset's species (core/species.py) when left out
    species: Literal["human", "mouse"] | None = None
    # Try a source whose download failed a moment ago again now
    retry: bool = False


def _marker_db_dest(settings: Settings) -> Path:
    return Path(settings.DATA_DIR) / "marker_db"  # the data volume


@router.get("/datasets/{dataset_id}/annotation/marker-db/status")
async def marker_db_status(
    dataset_id: str,
    dm: DatasetManager = Depends(get_dataset_manager),
    settings: Settings = Depends(get_settings_dep),
):
    """Which marker databases are downloaded (source, release, date, sha256, row
    counts), which failed and why, whether downloading is on, and the dataset's
    species."""
    adata = (await _adaptor(dm, dataset_id)).adata

    def _read():
        from scview.core.species import detect_species

        return {
            **mdb.status(_marker_db_dest(settings), auto_download=settings.MARKER_DB_AUTO_DOWNLOAD),
            "dataset_species": detect_species(adata).species,
        }

    return await asyncio.to_thread(_read)


class MarkerDbDownloadRequest(BaseModel):
    retry: bool = False


@router.post("/datasets/{dataset_id}/annotation/marker-db/download")
async def marker_db_download(
    dataset_id: str,
    body: MarkerDbDownloadRequest | None = None,
    dm: DatasetManager = Depends(get_dataset_manager),
    settings: Settings = Depends(get_settings_dep),
):
    """Start fetching the missing marker databases in the background and return the
    status at once; poll ``/status`` until ``state`` is no longer "downloading"."""
    await _adaptor(dm, dataset_id)
    return await asyncio.to_thread(
        mdb.start_download, _marker_db_dest(settings), auto_download=settings.MARKER_DB_AUTO_DOWNLOAD,
        retry=bool(body and body.retry),
    )


@router.post("/datasets/{dataset_id}/annotation/marker-db/check")
async def marker_db_check(
    dataset_id: str,
    body: MarkerDbCheckRequest,
    dm: DatasetManager = Depends(get_dataset_manager),
    settings: Settings = Depends(get_settings_dep),
):
    """Each label of ``column``: its database type(s) and how the name matched, the
    verdict (supported / no database support / too few database markers / not in
    database), p, overlapping genes, and the 3 best database types for its cells.
    Downloads the databases on first use. Changes nothing."""
    adata = (await _adaptor(dm, dataset_id)).adata
    if body.column not in adata.obs.columns:
        raise HTTPException(status_code=404, detail=f"Column '{body.column}' not found.")

    def _run():
        db = mdb.get_db(_marker_db_dest(settings), auto_download=settings.MARKER_DB_AUTO_DOWNLOAD,
                        retry=body.retry)
        return mdb.check(adata, body.column, body.species, db)

    try:
        # a download on first use, markers when none are stored: off the loop
        return await asyncio.to_thread(_run)
    except mdb.MarkerDBUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
