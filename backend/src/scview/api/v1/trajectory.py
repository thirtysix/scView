"""Trajectory / pseudotime analysis endpoints – Phase 5.

Provides endpoints for discovering, retrieving, and visualising pseudotime
columns and gene expression along pseudotime.
"""

from __future__ import annotations

import logging

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query

from scview.core.dataset_manager import DatasetManager
from scview.dependencies import get_dataset_manager

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/datasets/{dataset_id}/trajectory")
async def list_pseudotime_columns(
    dataset_id: str,
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """List available pseudotime columns from obs.

    Uses the adaptor's ``available_pseudotime_columns()`` which detects columns
    containing keywords like "pseudotime", "dpt_", "latent_time", "monocle".
    """
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    columns = adaptor.available_pseudotime_columns()

    return {
        "dataset_id": dataset_id,
        "pseudotime_columns": columns,
        "n_columns": len(columns),
    }


@router.get("/datasets/{dataset_id}/trajectory/{column}")
async def get_pseudotime_values(
    dataset_id: str,
    column: str,
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """Return pseudotime values for a specific column as a JSON array."""
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    available = adaptor.available_pseudotime_columns()
    if column not in available:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Column '{column}' is not a recognised pseudotime column.  "
                f"Available: {available}"
            ),
        )

    values = adaptor.adata.obs[column].values.astype(float)
    # Unreachable cells get inf (DPT, Monocle) and unassigned ones NaN: send null
    # for them rather than 0 / 1.8e308, which flattened the colour scale.
    finite = np.isfinite(values)
    if not finite.any():
        raise HTTPException(status_code=422, detail=f"Column '{column}' has no finite values.")

    return {
        "dataset_id": dataset_id,
        "column": column,
        "n_cells": int(len(values)),
        "values": [float(v) if ok else None for v, ok in zip(values, finite)],
        "min": float(values[finite].min()),
        "max": float(values[finite].max()),
    }


@router.get("/datasets/{dataset_id}/trajectory/{column}/genes")
async def get_expression_along_pseudotime(
    dataset_id: str,
    column: str,
    genes: str = Query(
        ...,
        description="Comma-separated list of gene names",
    ),
    n_bins: int = Query(
        default=100,
        ge=10,
        le=500,
        description="Number of bins for smoothing",
    ),
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """Return expression values along pseudotime for selected genes.

    For each gene the response includes:
    - raw (pseudotime, expression) pairs
    - binned/smoothed expression values for line-plot rendering

    Large datasets are sub-sampled to at most 5 000 points per gene for the
    raw scatter, while the binned values always cover the full dataset.
    """
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    available = adaptor.available_pseudotime_columns()
    if column not in available:
        raise HTTPException(
            status_code=404,
            detail=f"Column '{column}' is not a recognised pseudotime column.",
        )

    gene_list = [g.strip() for g in genes.split(",") if g.strip()]
    if not gene_list:
        raise HTTPException(status_code=400, detail="At least one gene name is required.")

    if len(gene_list) > 10:
        raise HTTPException(
            status_code=400,
            detail="At most 10 genes can be queried at once.",
        )

    pseudotime = adaptor.adata.obs[column].values.astype(float)
    # Cells without a finite pseudotime are left out of the trends.
    finite = np.isfinite(pseudotime)
    if not finite.any():
        raise HTTPException(status_code=422, detail=f"Column '{column}' has no finite values.")
    pseudotime = pseudotime[finite]

    # The same expression layer as every other expression view (log-normalised
    # .raw when present), not X, which the pipeline leaves scaled.
    layer = adaptor.default_expression_layer()
    present = set(adaptor._var_names_for_layer(layer))
    results: dict[str, dict] = {}

    for gene in gene_list:
        if gene not in present:
            results[gene] = {"found": False}
            continue

        expr = adaptor.get_expression([gene], layer=layer)[:, 0].astype(float)[finite]

        # ---- Binned / smoothed values ----
        pt_min, pt_max = float(np.min(pseudotime)), float(np.max(pseudotime))
        if pt_max <= pt_min:
            pt_max = pt_min + 1.0

        bin_edges = np.linspace(pt_min, pt_max, n_bins + 1)
        bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_indices = np.digitize(pseudotime, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        bin_means = np.zeros(n_bins)
        bin_counts = np.zeros(n_bins)
        for i in range(len(pseudotime)):
            b = bin_indices[i]
            bin_means[b] += expr[i]
            bin_counts[b] += 1

        mask = bin_counts > 0
        bin_means[mask] /= bin_counts[mask]

        # ---- Sub-sampled raw scatter ----
        max_scatter = 5000
        n_cells = len(pseudotime)
        if n_cells > max_scatter:
            idx = np.random.choice(n_cells, max_scatter, replace=False)
            idx.sort()
            scatter_pt = pseudotime[idx].tolist()
            scatter_expr = expr[idx].tolist()
        else:
            scatter_pt = pseudotime.tolist()
            scatter_expr = expr.tolist()

        results[gene] = {
            "found": True,
            "scatter_pseudotime": scatter_pt,
            "scatter_expression": scatter_expr,
            "binned_pseudotime": bin_centres.tolist(),
            "binned_expression": bin_means.tolist(),
            "binned_counts": bin_counts.astype(int).tolist(),
        }

    return {
        "dataset_id": dataset_id,
        "pseudotime_column": column,
        "genes": results,
    }
