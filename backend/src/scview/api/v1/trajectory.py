"""Trajectory / pseudotime analysis endpoints – Phase 5.

Provides endpoints for discovering, retrieving, and visualising pseudotime
columns and gene expression along pseudotime, for computing diffusion
pseudotime and the PAGA group graph in-app (``core/trajectory.py``), and for
ranking the genes that change along a pseudotime (``core/pseudotime_genes.py``).
"""

from __future__ import annotations

import asyncio
import logging
import os

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from scview.core import pseudotime_genes as ptg
from scview.core import large_mode
from scview.core.dataset_manager import DatasetManager, DatasetMemoryError
from scview.core.obs_writes import uns_snapshot, write_derived
from scview.core.processing import run_exclusive
from scview.core.trajectory import (
    TrajectoryError,
    compute_dpt,
    compute_paga,
    graph_source,
    paga_view,
    read_params,
)
from scview.dependencies import get_dataset_manager

logger = logging.getLogger(__name__)

router = APIRouter()

# RNA velocity needs both (scVelo's names); scView does not compute it yet.
VELOCITY_LAYERS = ("spliced", "unspliced")


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
        # How the in-app column was computed (root group, graph), if it was.
        "computed": read_params(adaptor.adata),
        # Spliced/unspliced counts are there, so RNA velocity could be run elsewhere.
        "velocity_layers": all(k in adaptor.adata.layers for k in VELOCITY_LAYERS),
    }


# Before /trajectory/{column}, which would otherwise take "paga" as a column name.
@router.get("/datasets/{dataset_id}/trajectory/paga")
async def get_paga(
    dataset_id: str,
    embedding: str = Query(default="", description="obsm key to place the groups on"),
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """The saved PAGA graph (null when none), with each group's median position
    on ``embedding``, whether its grouping has changed since, and the group that
    holds the pseudotime root cell."""
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    try:
        view = await asyncio.to_thread(paga_view, adaptor.adata, embedding or None)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc.args[0]))
    return {"dataset_id": dataset_id, **view}


class PAGARequest(BaseModel):
    groupby: str  # obs column whose groups become the graph's nodes


@router.post("/datasets/{dataset_id}/trajectory/paga")
async def compute_paga_graph(
    dataset_id: str,
    body: PAGARequest,
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """Compute PAGA between the groups of one column on (a copy of) the
    neighbour graph and save it with the dataset: a uns record plus provenance."""
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    if body.groupby not in adaptor.adata.obs.columns:
        raise HTTPException(status_code=404, detail=f"Column '{body.groupby}' not found.")
    try:
        graph_source(adaptor.adata)
    except TrajectoryError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    del adaptor  # no hold on the cached copy through run_exclusive's reload

    def _run(adaptor):
        work = adaptor.adata  # in memory: run_exclusive reads a backed dataset in first
        before = uns_snapshot(work)
        record = compute_paga(work, body.groupby)
        output_path = str(dm.derived_h5ad_path(dataset_id, adaptor.h5ad_path))
        write_derived(work, output_path, adaptor.h5ad_path, before)
        return record

    try:
        # In a worker thread with the dataset to itself; then served from disk.
        record = await run_exclusive(dm, dataset_id, _run)
    except (HTTPException, DatasetMemoryError):  # a 507 from reading a backed dataset in
        raise
    except TrajectoryError as exc:
        raise HTTPException(status_code=400, detail=str(exc))  # raised before any change
    except Exception as exc:
        logger.error("PAGA failed for %s: %s", dataset_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"PAGA failed: {exc!s}")
    return {"dataset_id": dataset_id, **record}


class DPTRequest(BaseModel):
    root_column: str  # obs column holding the start group (cluster / cell type)
    root_value: str


@router.post("/datasets/{dataset_id}/trajectory/dpt")
async def compute_pseudotime(
    dataset_id: str,
    body: DPTRequest,
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """Compute diffusion pseudotime rooted in one group, save it as an obs
    column of the dataset's derived file, and reload the dataset."""
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    adata = adaptor.adata
    if body.root_column not in adata.obs.columns:
        raise HTTPException(status_code=404, detail=f"Column '{body.root_column}' not found.")
    # Cheap checks first, so a bad request costs no copy, write or reload.
    col = adata.obs[body.root_column]
    if not (col.astype(str).where(col.notna()) == body.root_value).any():
        raise HTTPException(
            status_code=400,
            detail=f"No cells have {body.root_column} = '{body.root_value}'.",
        )
    try:
        graph_source(adata)
    except TrajectoryError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    del adaptor, adata, col  # no hold on the cached copy through run_exclusive's reload

    def _run(adaptor):
        work = adaptor.adata  # in memory: run_exclusive reads a backed dataset in first
        params = compute_dpt(work, body.root_column, body.root_value)
        # Write beside, then swap in: a backed dataset may be reading this path.
        output_path = str(dm.derived_h5ad_path(dataset_id, adaptor.h5ad_path))
        partial = output_path + ".partial"
        work.write_h5ad(partial, compression="gzip")
        os.replace(partial, output_path)
        return params

    try:
        # In a worker thread with the dataset to itself (busy, locked until the
        # thread ends even if the client goes away); then served from disk.
        params = await run_exclusive(dm, dataset_id, _run)
    except (HTTPException, DatasetMemoryError):  # a 507 from reading a backed dataset in
        raise
    except TrajectoryError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error("Pseudotime failed for %s: %s", dataset_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Pseudotime failed: {exc!s}")
    return {"dataset_id": dataset_id, **params}


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

        bin_counts = np.bincount(bin_indices, minlength=n_bins)
        bin_sums = np.bincount(bin_indices, weights=expr, minlength=n_bins)
        # A bin no cell falls in (DPT's sparse tail) is a gap, not a mean of 0.
        bin_means = [
            float(s / c) if c > 0 else None for s, c in zip(bin_sums, bin_counts)
        ]

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
            "binned_expression": bin_means,
            "binned_counts": bin_counts.astype(int).tolist(),
        }

    return {
        "dataset_id": dataset_id,
        "pseudotime_column": column,
        "genes": results,
    }


class TrendGenesRequest(BaseModel):
    n_top: int = Field(default=ptg.DEFAULT_TOP, ge=1, le=ptg.MAX_TOP)
    n_bins: int = Field(default=ptg.DEFAULT_BINS, ge=5, le=ptg.MAX_BINS)
    # Only the cells in these groups of this column (a branch: the groups on a PAGA path).
    groups_column: str | None = None
    groups: list[str] | None = Field(default=None, max_length=200)


@router.post("/datasets/{dataset_id}/trajectory/{column}/trend-genes")
async def rank_trend_genes(
    dataset_id: str,
    column: str,
    body: TrendGenesRequest,
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """The genes whose log-normalised expression changes most along the
    pseudotime ``column`` (spline F-test, core/pseudotime_genes.py), each with
    its binned trend. Changes nothing."""
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    large_mode.check(adaptor, "Genes along pseudotime",
                     "it fits every gene's values along the cells in memory")
    if column not in adaptor.available_pseudotime_columns():
        raise HTTPException(
            status_code=404,
            detail=f"Column '{column}' is not a recognised pseudotime column.",
        )
    try:
        result = await asyncio.to_thread(
            ptg.trend_genes, adaptor.adata, column, n_top=body.n_top, n_bins=body.n_bins,
            groups_column=body.groups_column, groups=body.groups,
        )
    except ptg.TrendError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"dataset_id": dataset_id, **result}
