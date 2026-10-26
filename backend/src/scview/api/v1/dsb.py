"""DSB protein normalisation endpoints (``core/dsb.py``).

``GET /datasets/{id}/protein/dsb``: whether DSB can run, the empty droplets of each
sample with a raw matrix (and their density for the window plot), the default isotypes
and windows, and the last run's record. ``POST /datasets/{id}/protein/dsb`` computes it
under the dataset's edit lock in a worker thread and saves ``obsm["protein_dsb"]`` (the
``protein_dsb`` layer) with the record and a provenance step.
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from scview.core import dsb, results_store
from scview.core.dataset_manager import DatasetManager
from scview.core.obs_writes import dataset_edit, edit_room, editable, save_edit, uns_snapshot
from scview.dependencies import get_dataset_manager

logger = logging.getLogger(__name__)

router = APIRouter()


class DsbRequest(BaseModel):
    # auto: empty droplets when a raw matrix gave them, else ModelNegativeADTnorm
    mode: str = "auto"
    rna_window: tuple[float, float] = dsb.RNA_WINDOW  # log10 RNA UMIs, [lo, hi)
    prot_window: tuple[float, float] = dsb.PROT_WINDOW  # log10 protein UMIs, [lo, hi)
    isotypes: list[str] | None = Field(None, max_length=50)  # default: the panel's controls
    denoise: bool = True  # step II (dsb denoise.counts)


async def _adaptor(dm: DatasetManager, dataset_id: str):
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")
    return adaptor


@router.get("/datasets/{dataset_id}/protein/dsb")
async def dsb_status(dataset_id: str, dm: DatasetManager = Depends(get_dataset_manager)):
    adaptor = await _adaptor(dm, dataset_id)
    return await asyncio.to_thread(dsb.status, adaptor.adata)


@router.post("/datasets/{dataset_id}/protein/dsb")
async def run_dsb(dataset_id: str, body: DsbRequest, dm: DatasetManager = Depends(get_dataset_manager)):
    """Compute DSB and save it; answers with the run's record."""
    async with dataset_edit(dataset_id) as held:
        adaptor = await _adaptor(dm, dataset_id)
        output_path = str(dm.derived_h5ad_path(dataset_id, adaptor.h5ad_path))
        backed = await edit_room(dm, dataset_id, adaptor)

        def _run():
            work = editable(adaptor)
            before = uns_snapshot(work)
            record = dsb.run(work, mode=body.mode, rna_window=body.rna_window,
                             prot_window=body.prot_window, isotypes=body.isotypes, denoise=body.denoise)
            save_edit(adaptor, work, output_path, before, obsm_keys=(dsb.DSB_KEY,))
            return record

        try:
            record = await held.run(_run)
        except dsb.DsbError as exc:
            raise HTTPException(status_code=exc.status, detail=str(exc))  # raised before any change
        except Exception as exc:
            dm.evict(dataset_id)  # the cached copy may hold values that were never saved
            logger.error("DSB failed for %s: %s", dataset_id, exc, exc_info=True)
            raise HTTPException(status_code=500, detail=f"Could not compute DSB: {exc!s}")
        if backed:
            dm.evict(dataset_id)
            adaptor = await dm.load_dataset(dataset_id)
        else:
            dm.saved_in_place(dataset_id, adaptor, output_path)
    results_store.keep(dataset_id, "dsb", dsb.results_value(record), key=dsb.DSB_KEY,
                       label="DSB protein normalisation", adaptor=adaptor,
                       params={"mode": record["mode"], "isotypes": record["isotypes"]})
    return record
