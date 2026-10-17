"""Normalisation diagnostics: each gene's mean against its variance, before and after.

Backs the Data Assessment mean-variance plot (``core/normalization.py``): counts,
the shifted log and analytic Pearson residuals side by side, with the dataset's
current variable genes marked, so the choice between the two methods rests on
what the data shows.
"""

from __future__ import annotations

import asyncio
import logging

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException

from scview.core import normalization
from scview.core.assessor import recorded_normalization_method
from scview.core.dataset_manager import DatasetManager
from scview.core.object_cache import ObjectCache
from scview.dependencies import get_dataset_manager

logger = logging.getLogger(__name__)

router = APIRouter()

# Per loaded dataset object: a reload (after any pipeline run) is a new object,
# so a cached diagnostic never outlives the data it describes.
_CACHE = ObjectCache()


def _sig(values, digits: int = 4) -> list[float]:
    """Floats rounded to a few significant digits: the payload is one per gene."""
    return [float(f"{v:.{digits}g}") for v in np.asarray(values, dtype=np.float64)]


def mean_variance_payload(adata) -> dict:
    counts, names, source = normalization.counts_all_genes(adata)
    mv = normalization.mean_variance(counts, names)
    keep = mv.pop("keep")
    genes = pd.Index(mv["genes"])
    hvg = None
    if "highly_variable" in adata.var.columns:
        flags = pd.Series(adata.var["highly_variable"].astype(bool).to_numpy(), index=adata.var_names)
        hvg = flags.reindex(genes, fill_value=False).astype(bool).tolist()
    return {
        "n_cells": mv["n_cells"],
        "n_cells_used": mv["n_cells_used"],
        "n_genes": int(keep.sum()),
        "n_genes_undetected": int((~keep).sum()),
        "theta": mv["theta"],
        "clip": round(mv["clip"], 3),
        "counts_source": source,
        "normalization_method": recorded_normalization_method(adata),
        "genes": mv["genes"],
        "mean": _sig(mv["mean"]),
        "var_counts": _sig(mv["var_counts"]),
        "var_log": _sig(mv["var_log"]),
        "var_residuals": _sig(mv["var_residuals"]),
        "hvg": hvg,
    }


@router.get("/datasets/{dataset_id}/qc/mean-variance")
async def get_mean_variance(
    dataset_id: str,
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """Per-gene mean vs variance of the counts, the shifted log and the Pearson
    residuals (on up to 20,000 sampled cells), and the current variable genes."""
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")
    adata = adaptor.adata
    cached = _CACHE.get(adata)
    if cached is not None:
        return cached
    try:
        # Residuals of every gene, a block at a time: seconds on 14k cells.
        payload = await asyncio.to_thread(mean_variance_payload, adata)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        logger.error("Mean-variance diagnostic failed for %s: %s", dataset_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Mean-variance diagnostic failed: {e}")
    _CACHE.put(adata, payload)
    return payload
