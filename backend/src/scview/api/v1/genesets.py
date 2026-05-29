"""Gene set endpoints – Phase 5.

Provides gene set collection listing, search (via MSigDB), and per-cell
scoring via scanpy.tl.score_genes().
"""

from __future__ import annotations

import logging

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from scview.config import Settings
from scview.core.dataset_manager import DatasetManager
from scview.core.msigdb_loader import get_msigdb_loader
from scview.dependencies import get_dataset_manager, get_settings_dep

logger = logging.getLogger(__name__)

router = APIRouter()

# ---------------------------------------------------------------------------
# MSigDB collection catalogue — now uses subcategory-level loader
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class GeneSetScoreRequest(BaseModel):
    """Body for the gene-set scoring endpoint."""

    gene_set: list[str]
    score_name: str = "gene_set_score"


class GeneSetScoreResponse(BaseModel):
    """Per-cell scores returned by the scoring endpoint."""

    score_name: str
    n_cells: int
    scores: list[float]
    min_score: float
    max_score: float
    genes_found: list[str]
    genes_missing: list[str]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/datasets/{dataset_id}/genesets/collections")
async def list_geneset_collections(
    dataset_id: str,
    dm: DatasetManager = Depends(get_dataset_manager),
    settings: Settings = Depends(get_settings_dep),
):
    """List available gene-set collections (subcategory-level)."""
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    loader = get_msigdb_loader(settings.MSIGDB_DIR)
    if loader is None:
        return {"dataset_id": dataset_id, "collections": []}

    collections = loader.available_collections()
    return {
        "dataset_id": dataset_id,
        "collections": collections,
    }


@router.get("/datasets/{dataset_id}/genesets/search")
async def search_genesets(
    dataset_id: str,
    q: str = Query(default="", description="Search query for gene set name"),
    collection: str = Query(default="", description="Filter by collection id"),
    limit: int = Query(default=50, ge=1, le=200, description="Max results"),
    dm: DatasetManager = Depends(get_dataset_manager),
    settings: Settings = Depends(get_settings_dep),
):
    """Search MSigDB gene-set collections by name."""
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    loader = get_msigdb_loader(settings.MSIGDB_DIR)
    if loader is None:
        return {
            "results": [],
            "total": 0,
            "query": q,
            "collection": collection,
            "message": "MSigDB directory not configured. Set MSIGDB_DIR environment variable.",
        }

    if not q.strip():
        # No query — list all gene sets in the collection (or first collection)
        results = loader.search("", collection=collection, limit=limit)
        return {
            "results": results,
            "total": len(results),
            "query": q,
            "collection": collection,
        }

    results = loader.search(q, collection=collection, limit=limit)
    return {
        "results": results,
        "total": len(results),
        "query": q,
        "collection": collection,
    }


@router.post("/datasets/{dataset_id}/genesets/score")
async def score_genesets(
    dataset_id: str,
    body: GeneSetScoreRequest,
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """Compute per-cell gene-set scores via scanpy.tl.score_genes().

    Accepts a list of gene names and returns a score for every cell in the
    dataset.
    """
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    if not body.gene_set or len(body.gene_set) == 0:
        raise HTTPException(
            status_code=400,
            detail="gene_set must contain at least one gene name.",
        )

    adata = adaptor.adata

    # Determine which genes are present in the dataset
    available_genes = set(adaptor.gene_names())
    genes_found = [g for g in body.gene_set if g in available_genes]
    genes_missing = [g for g in body.gene_set if g not in available_genes]

    if len(genes_found) == 0:
        raise HTTPException(
            status_code=400,
            detail=(
                f"None of the {len(body.gene_set)} provided genes were found "
                "in the dataset."
            ),
        )

    try:
        import scanpy as sc

        score_key = body.score_name

        # Materialize backed sparse datasets — scanpy score_genes cannot
        # operate on anndata._core.sparse_dataset._CSRDataset.
        work_adata = adata.to_memory() if hasattr(adata, "to_memory") else adata.copy()

        # Filter genes_found to those actually present in the expression matrix
        # that score_genes will use (raw if use_raw, else X).
        has_raw = work_adata.raw is not None
        if has_raw:
            raw_var = set(work_adata.raw.var_names.tolist())
            genes_in_raw = [g for g in genes_found if g in raw_var]
        else:
            genes_in_raw = []
        main_var = set(work_adata.var_names.tolist())
        genes_in_main = [g for g in genes_found if g in main_var]

        # Try with raw first (log-normalised), fall back to main X
        scored = False
        for use_raw, gene_list in [(True, genes_in_raw), (False, genes_in_main)]:
            if not gene_list or (use_raw and not has_raw):
                continue
            try:
                sc.tl.score_genes(
                    work_adata,
                    gene_list=gene_list,
                    score_name=score_key,
                    use_raw=use_raw,
                )
                scored = True
                # Update genes_found to reflect what was actually used
                genes_found = gene_list
                genes_missing = [g for g in body.gene_set if g not in set(genes_found)]
                break
            except Exception as inner_err:
                logger.warning(
                    "score_genes failed with use_raw=%s: %s — trying fallback",
                    use_raw, inner_err,
                )
                continue

        if not scored:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Could not score gene set. {len(genes_found)} genes were found "
                    "but scoring failed. This may happen if too few genes overlap "
                    "with the expression matrix."
                ),
            )

        scores = work_adata.obs[score_key].values.astype(float)
        scores = np.nan_to_num(scores, nan=0.0)

        return GeneSetScoreResponse(
            score_name=score_key,
            n_cells=int(len(scores)),
            scores=scores.tolist(),
            min_score=float(np.min(scores)),
            max_score=float(np.max(scores)),
            genes_found=genes_found,
            genes_missing=genes_missing,
        )

    except HTTPException:
        raise
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="scanpy is required for gene set scoring but is not installed.",
        )
    except Exception as e:
        logger.error("Gene set scoring failed for dataset %s: %s", dataset_id, e)
        raise HTTPException(status_code=500, detail=f"Scoring failed: {e!s}")
