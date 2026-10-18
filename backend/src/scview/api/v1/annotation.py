"""Cell-type annotation endpoints.

CellTypist models are tissue/system specific (there is no single universal model),
so the UI offers the catalog below and the user picks the one matching their sample.
The default, ``Immune_All_Low``, suits the common PBMC/immune case.

``/datasets/{id}/annotation/confidence`` checks how well each cell's label in a
column fits its expression neighbourhood, with split conformal prediction sets
(``core/annotation_confidence.py``): GET returns the last check of a column,
POST runs one, POST ``.../unknown`` copies the column with rejected cells set to
"Unknown".
"""

from __future__ import annotations

import asyncio
import logging
from functools import lru_cache

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from scview.core import annotation_confidence as conf
from scview.core import celltypist_fit as fit
from scview.core.dataset_manager import DatasetManager
from scview.core.obs_writes import dataset_edit, uns_snapshot, write_derived
from scview.dependencies import get_dataset_manager

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
        backed = getattr(adaptor.adata, "isbacked", False)

        def _run():
            work = adaptor.adata.to_memory() if backed else adaptor.adata
            before = uns_snapshot(work)
            result = edit(work)
            write_derived(work, output_path, adaptor.h5ad_path, before)
            return result

        try:
            result = await held.run(_run)
        except conf.ConfidenceError as exc:  # raised before any change
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
