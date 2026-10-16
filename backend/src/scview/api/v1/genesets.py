"""Gene set endpoints – Phase 5.

Provides gene set collection listing, search (MSigDB and custom collections),
and per-cell scoring via scanpy.tl.score_genes().
"""

from __future__ import annotations

import logging

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from scview.config import Settings
from scview.core.custom_genesets import get_custom_store, is_custom_id
from scview.core.dataset_manager import DatasetManager
from scview.core.geneset_sources import custom_collections_for_dataset
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
    """List available gene-set collections (subcategory-level).

    MSigDB collections first, then the custom ones (category "Custom"), which
    also say how many of their genes this dataset measures.
    """
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    loader = get_msigdb_loader(settings.MSIGDB_DIR)
    collections = loader.available_collections() if loader is not None else []
    collections += custom_collections_for_dataset(
        adaptor, get_custom_store(settings.DATA_DIR)
    )
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
    """Search gene-set collections (MSigDB, then custom) by name.

    An empty query lists the collection's sets. A set whose name matches the
    query exactly comes first, so looking a result term up by name finds that
    term rather than the first longer name containing it.
    """
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    loader = get_msigdb_loader(settings.MSIGDB_DIR)
    store = get_custom_store(settings.DATA_DIR)
    custom = is_custom_id(collection)
    query = q.strip()

    results: list[dict] = []
    if query:
        exact = None
        if not custom and loader is not None:
            exact = loader.get_gene_set(query, collection=collection)
        if exact is None and (custom or not collection):
            exact = store.get_gene_set(query, collection=collection)
        if exact is not None:
            results.append(exact)
    if not custom and loader is not None:
        results += loader.search(query, collection=collection, limit=limit)
    if custom or not collection:
        results += store.search(query, collection=collection, limit=limit)

    seen: set[tuple[str, str]] = set()
    unique: list[dict] = []
    for r in results:
        key = (r["collection"], r["name"])
        if key not in seen:
            seen.add(key)
            unique.append(r)
    unique = unique[:limit]

    body = {
        "results": unique,
        "total": len(unique),
        "query": q,
        "collection": collection,
    }
    if loader is None and not custom:
        body["message"] = "MSigDB directory not configured. Set MSIGDB_DIR environment variable."
    return body


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

    # Determine which genes are present in the dataset. A name matches exactly,
    # else ignoring case, the way ORA matches MSigDB's upper-case human symbols
    # (and custom sets, stored upper-case) to a mouse dataset's Cd3e.
    available_genes = set(adaptor.gene_names())
    by_upper: dict[str, str] = {}
    for g in available_genes:
        by_upper.setdefault(g.upper(), g)
    genes_found = []
    genes_missing = []
    for g in body.gene_set:
        hit = g if g in available_genes else by_upper.get(g.upper())
        if hit is None:
            genes_missing.append(g)
        elif hit not in genes_found:
            genes_found.append(hit)

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
                used = {g.upper() for g in gene_list}
                genes_missing = [g for g in body.gene_set if g.upper() not in used]
                genes_found = gene_list
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
