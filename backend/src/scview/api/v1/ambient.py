"""Ambient RNA endpoint: each sample's soup and contamination fraction, and the markers
it may drive (``core/ambient.py``).

``GET /datasets/{id}/qc/ambient?column=``: per sample the soup's top genes (from the empty
droplets of an attached raw matrix, else the cells' own counts, labelled approximate), its
SoupX-style contamination fraction rho with its range and notes, the ambient-looking
markers per group of ``column`` (default: the active clustering) and whether the counts
were corrected. 422 for a large dataset (it reads every cell's counts).
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from scview.core import ambient, assessment_gate, assessment_memory, large_mode
from scview.core.dataset_manager import DatasetManager, DatasetMemoryError
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
        # Sums each sample's counts and tests its marker genes: off the loop, in the memory budget.
        return await assessment_gate.run(
            dataset_id, ("ambient", column, id(adaptor.adata)), ambient.diagnose_cached, adaptor.adata,
            column or None, budget=assessment_gate.Budget(
                dm, lambda: assessment_memory.ambient_need(adaptor.adata, column or None),
                "The ambient RNA estimate"))
    except DatasetMemoryError:
        raise  # a 507 (main.py), not a 500
    except ambient.AmbientError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Ambient RNA estimate failed for %s: %s", dataset_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ambient RNA estimate failed: {e}")
