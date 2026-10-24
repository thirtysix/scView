"""Ambient RNA endpoint: each sample's soup and contamination fraction, and the markers
it may drive (``core/ambient.py``).

``GET /datasets/{id}/qc/ambient?column=``: per sample the soup's top genes (from the empty
droplets of an attached raw matrix, else the cells' own counts, labelled approximate), its
SoupX-style contamination fraction rho with its range and notes, the ambient-looking
markers per group of ``column`` (default: the active clustering) and whether the counts
were corrected. 422 for a large dataset (it reads every cell's counts).
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Depends, HTTPException

from scview.core import ambient, large_mode
from scview.core.dataset_manager import DatasetManager
from scview.dependencies import get_dataset_manager

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/datasets/{dataset_id}/qc/ambient")
async def get_ambient(
    dataset_id: str,
    column: str = "",
    dm: DatasetManager = Depends(get_dataset_manager),
):
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")
    large_mode.check(adaptor, "The ambient RNA estimate", "it reads every cell's counts")
    if column and column not in adaptor.adata.obs.columns:
        raise HTTPException(status_code=404, detail=f"Column '{column}' not found.")
    try:
        # Sums each sample's counts and tests its marker genes: off the loop.
        return await asyncio.to_thread(ambient.diagnose_cached, adaptor.adata, column or None)
    except ambient.AmbientError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Ambient RNA estimate failed for %s: %s", dataset_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ambient RNA estimate failed: {e}")
