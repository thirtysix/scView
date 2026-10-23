"""Cell hashing demultiplexing (``core/hashing.py``).

``GET /hashing`` lists the tag counts and the antibody features that could be
hashtags, and summarises the saved call; ``POST /hashing/demux`` calls every
cell (HTODemux or hashedDrops) and saves the ``hash_*`` columns;
``POST /hashing/samples`` writes a sample column from a tag -> sample mapping;
``GET /hashing/plot-data`` gives the per-tag CLR histograms with their cutoffs
and the tag x call heatmap. Saves run in a worker thread under the edit lock.
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from scview.core import hashing, protein
from scview.core.dataset_manager import DatasetManager
from scview.core.obs_writes import dataset_edit, uns_snapshot, write_derived
from scview.core import working_copy
from scview.dependencies import get_dataset_manager

logger = logging.getLogger(__name__)

router = APIRouter()

# Entries a save can change: the tag counts, and the protein modality the
# confirmed hashtags leave (its CLR is made again without them).
_OBSM_KEYS = (hashing.HTO_KEY, protein.COUNTS_KEY, protein.CLR_KEY)


class DemuxRequest(BaseModel):
    tags: list[str] = Field(default_factory=list)
    method: str = hashing.HTODEMUX
    quantile: float = hashing.POSITIVE_QUANTILE


class SamplesRequest(BaseModel):
    mapping: dict[str, str] = Field(default_factory=dict)
    column: str = hashing.DEFAULT_SAMPLE_COLUMN


async def _adaptor(dm: DatasetManager, dataset_id: str):
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")
    return adaptor


@router.get("/datasets/{dataset_id}/hashing")
async def get_hashing(dataset_id: str, dm: DatasetManager = Depends(get_dataset_manager)):
    """Tags found (``obsm hto_counts``), antibody features suggested as hashtags,
    the saved call's counts (on the current cells) and the sample column."""
    adaptor = await _adaptor(dm, dataset_id)
    return await asyncio.to_thread(hashing.summary, adaptor.adata)


@router.get("/datasets/{dataset_id}/hashing/plot-data")
async def get_hashing_plot_data(dataset_id: str, dm: DatasetManager = Depends(get_dataset_manager)):
    """Per tag: CLR histogram split by the call, with the cutoff in CLR units;
    the tag x call heatmap (mean CLR); hashedDrops' LogFC / LogFC2 histograms."""
    adaptor = await _adaptor(dm, dataset_id)
    try:
        return await asyncio.to_thread(hashing.plot_data, adaptor.adata)
    except hashing.HashingError as e:
        raise HTTPException(status_code=e.status, detail=str(e))


async def _save(dm: DatasetManager, dataset_id: str, edit, what: str):
    """Run ``edit(work)`` in a worker thread under the edit lock and save the
    dataset (obs, uns and the tag / protein obsm entries)."""
    async with dataset_edit(dataset_id) as held:
        adaptor = await _adaptor(dm, dataset_id)
        output_path = str(dm.derived_h5ad_path(dataset_id, adaptor.h5ad_path))
        backed = await working_copy.make_room(dm, dataset_id, adaptor)

        def _run():
            work = working_copy.in_memory(adaptor)
            before = uns_snapshot(work)
            result = edit(work)
            write_derived(work, output_path, adaptor.h5ad_path, before, obsm_keys=_OBSM_KEYS)
            return result

        try:
            result = await held.run(_run)
        except hashing.HashingError as e:
            raise HTTPException(status_code=e.status, detail=str(e))  # raised before any change
        except Exception as e:
            dm.evict(dataset_id)  # the cached copy may hold columns that were never saved
            logger.error("Hashing %s failed for %s: %s", what, dataset_id, e, exc_info=True)
            raise HTTPException(status_code=500, detail=f"Could not save the {what}: {e!s}")
        if backed:
            dm.evict(dataset_id)
            await dm.load_dataset(dataset_id)
        else:
            dm.saved_in_place(dataset_id, adaptor, output_path)
        return result


@router.post("/datasets/{dataset_id}/hashing/demux")
async def post_demux(dataset_id: str, body: DemuxRequest, dm: DatasetManager = Depends(get_dataset_manager)):
    """Call each cell's sample from ``tags`` (tag counts and/or antibody features,
    which are then moved out of the protein modality) and save ``hash_call``,
    ``hash_class``, ``hash_max``, ``hash_second``, ``hash_margin``."""
    return await _save(dm, dataset_id,
                       lambda work: hashing.demux(work, body.tags, body.method, body.quantile),
                       "hashing call")


@router.post("/datasets/{dataset_id}/hashing/samples")
async def post_samples(dataset_id: str, body: SamplesRequest, dm: DatasetManager = Depends(get_dataset_manager)):
    """Write ``column`` from the saved call: each singlet's sample name (``mapping``,
    tag -> name), doublets and negatives left empty."""

    def _edit(work):
        hashing.sample_column(work, body.mapping, body.column)
        return hashing.summary(work)

    return await _save(dm, dataset_id, _edit, "sample column")
