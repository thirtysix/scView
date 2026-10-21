"""PCA diagnostics: variance per component, elbow, loadings (core/pca_diagnostics.py)."""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Depends, HTTPException, Query

from scview.core import pca_diagnostics
from scview.core.dataset_manager import DatasetManager
from scview.dependencies import get_dataset_manager

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/datasets/{dataset_id}/pca/diagnostics")
async def get_pca_diagnostics(
    dataset_id: str,
    top_k: int = Query(default=pca_diagnostics.TOP_K, ge=1, le=50,
                       description="Genes per sign per PC in the loadings"),
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """Each PC's variance ratio and the cumulative curve, the Kneedle elbow and the
    PCs-to-90% count (``method`` says how each is computed), the PCA params, how many
    PCs the neighbour graph used, and the top +/- loading genes per PC. A dataset
    without a PCA answers ``{"available": false, "reason": ...}``."""
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")
    try:
        # Sorts every PC's loadings (genes x 50): off the loop.
        return await asyncio.to_thread(pca_diagnostics.pca_diagnostics, adaptor.adata, top_k)
    except Exception as e:
        logger.error("PCA diagnostics failed for %s: %s", dataset_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"PCA diagnostics failed: {e}")
