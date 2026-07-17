"""Enrichment analysis endpoints – Phase 5.

Provides pre-computed enrichment retrieval and on-demand enrichment via
marker gene lists.  When gseapy is installed, full pathway enrichment is
performed; otherwise the endpoint returns the top marker genes as a
starting point.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from scview.config import Settings
from scview.core import enrichment_network as network
from scview.core.dataset_manager import DatasetManager
from scview.core.gene_filters import select_query_genes
from scview.core.msigdb_loader import DEFAULT_COLLECTIONS, get_msigdb_loader
from scview.core.ora_background import (
    cache_is_current,
    get_background_genes,
    write_meta,
)
from scview.dependencies import get_dataset_manager, get_settings_dep

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class EnrichmentComputeRequest(BaseModel):
    """Body for the on-demand enrichment endpoint."""

    column: str = ""  # obs column whose markers to use (preferred)
    groupby: str = ""  # kept for backward compat — falls back if column is empty
    group: str
    n_genes: int = 100
    # Drop ribosomal/mitochondrial genes from the query. They dominate scRNA-seq
    # markers as an artifact and cross-react with every cell-type signature; see
    # core/gene_filters.py. Opt out for a ribosome-biology study.
    exclude_ribo_mito: bool = True
    gene_sets: list[str] = [
        "GO_Biological_Process_2025",
        "GO_Molecular_Function_2025",
        "GO_Cellular_Component_2025",
        "KEGG_2026",
        "MSigDB_Hallmark_2020",
        "Reactome_Pathways_2024",
    ]


class LocalEnrichmentRequest(BaseModel):
    """Body for the local MSigDB enrichment endpoint."""

    column: str = ""
    group: str
    n_genes: int = 100
    # See EnrichmentComputeRequest.exclude_ribo_mito.
    exclude_ribo_mito: bool = True
    collections: list[str] = DEFAULT_COLLECTIONS


class EnrichmentNetworkRequest(LocalEnrichmentRequest):
    """Body for the enrichment-network endpoint.

    Extends the local enrichment request, so the network is built from exactly the
    result the compute-local endpoint would return for the same parameters.
    """

    fdr: float = network.DEFAULT_FDR
    metric: str = network.DEFAULT_METRIC
    min_similarity: float = network.DEFAULT_MIN_SIMILARITY
    resolution: float = network.DEFAULT_RESOLUTION
    seed: int = network.DEFAULT_SEED
    min_term_size: int = network.DEFAULT_MIN_TERM_SIZE
    max_term_size: int = network.DEFAULT_MAX_TERM_SIZE
    # The collapsed cluster view needs no edges, and there are tens of thousands of
    # them (10,379 for one CD14 Mono run). Callers rendering only clusters/nodes
    # should turn this off rather than pay for the payload.
    include_edges: bool = True


class EnrichmentResult(BaseModel):
    """A single enrichment hit."""

    term: str
    pvalue: float
    adjusted_pvalue: float
    overlap_count: int
    gene_count: int
    genes: list[str]
    collection: str = ""


class EnrichmentResponse(BaseModel):
    """Full enrichment response."""

    group: str
    groupby: str
    n_genes_used: int
    results: list[EnrichmentResult]
    source: str  # "gseapy" | "precomputed" | "marker_list" | "msigdb_local"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/datasets/{dataset_id}/enrichment")
async def get_enrichment(
    dataset_id: str,
    group: str = Query(default="", description="Specific group to retrieve"),
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """Return pre-computed enrichment results from adata.uns if available."""
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    adata = adaptor.adata

    # Check for pre-computed enrichment stored in adata.uns
    enrichment_key = "enrichment_results"
    if enrichment_key not in adata.uns:
        # Also check common alternative keys
        alt_keys = ["enrich_results", "pathway_enrichment", "gsea_results"]
        found_key = None
        for k in alt_keys:
            if k in adata.uns:
                found_key = k
                break

        if found_key is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    "No pre-computed enrichment results found.  "
                    "Use the POST /enrichment/compute endpoint to run enrichment analysis."
                ),
            )
        enrichment_key = found_key

    raw = adata.uns[enrichment_key]

    # Try to parse into our response format
    results = _parse_precomputed_enrichment(raw, group)

    return {
        "dataset_id": dataset_id,
        "source": "precomputed",
        "enrichment_key": enrichment_key,
        "results": results,
    }


@router.get("/enrichment/msigdb-collections")
async def get_msigdb_collections(
    settings: Settings = Depends(get_settings_dep),
):
    """Return available MSigDB collections with hierarchical structure."""
    loader = get_msigdb_loader(settings.MSIGDB_DIR)
    if loader is None:
        return {"collections": [], "hierarchy": [], "defaults": DEFAULT_COLLECTIONS}

    return {
        "collections": loader.available_collections(),
        "hierarchy": loader.available_collections_hierarchical(),
        "defaults": DEFAULT_COLLECTIONS,
    }


@router.get("/datasets/{dataset_id}/enrichment/columns")
async def list_enrichment_columns(
    dataset_id: str,
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """List obs columns available for enrichment (marker columns + all categorical)."""
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    marker_cols = adaptor.marker_columns()

    # Also return all categorical obs columns (for enrichment even without markers)
    categorical_columns: list[str] = []
    for col_name in adaptor.adata.obs.columns:
        dtype = adaptor.adata.obs[col_name].dtype
        if hasattr(dtype, "name") and dtype.name == "category":
            categorical_columns.append(col_name)
        elif dtype == "object":
            categorical_columns.append(col_name)
        elif str(dtype).startswith("int"):
            n_unique = adaptor.adata.obs[col_name].nunique()
            if n_unique <= 100:
                categorical_columns.append(col_name)

    return {
        "dataset_id": dataset_id,
        "columns": marker_cols,
        "categorical_columns": categorical_columns,
    }


@router.get("/datasets/{dataset_id}/enrichment/groups")
async def list_enrichment_groups(
    dataset_id: str,
    column: str = Query(default="", description="Obs column to get groups for"),
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """List groups for which markers are available (for enrichment analysis)."""
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    # Determine which uns key to read
    if column:
        uns_key = f"rank_genes_groups__{column}"
        if uns_key not in adaptor.adata.uns:
            # Fall back to default if the column matches
            if "rank_genes_groups" in adaptor.adata.uns:
                rgg = adaptor.adata.uns["rank_genes_groups"]
                if rgg.get("params", {}).get("groupby") == column:
                    uns_key = "rank_genes_groups"
                else:
                    raise HTTPException(status_code=404, detail=f"No markers for column '{column}'.")
            else:
                raise HTTPException(status_code=404, detail=f"No markers for column '{column}'.")
    else:
        if not adaptor.has_markers():
            raise HTTPException(
                status_code=404,
                detail="No marker genes found. Run sc.tl.rank_genes_groups first.",
            )
        uns_key = "rank_genes_groups"

    rgg = adaptor.adata.uns[uns_key]
    groups: list[str] = []
    if hasattr(rgg["names"].dtype, "names") and rgg["names"].dtype.names:
        groups = list(rgg["names"].dtype.names)

    # Check which groups have pre-computed enrichment that is still servable.
    # A payload predating the background fix is reported as not computed, since
    # requesting it will recompute rather than return the cached result.
    col_name = column or rgg.get("params", {}).get("groupby", "")
    enrichment_computed = {
        g: (
            f"enrichment__{col_name}__{g}" in adaptor.adata.uns
            and cache_is_current(adaptor.adata, col_name, g)
        )
        for g in groups
    }

    return {
        "dataset_id": dataset_id,
        "column": col_name,
        "groups": groups,
        "enrichment_computed": enrichment_computed,
    }


@router.post("/datasets/{dataset_id}/enrichment/network")
async def enrichment_network(
    dataset_id: str,
    body: EnrichmentNetworkRequest,
    dm: DatasetManager = Depends(get_dataset_manager),
    settings: Settings = Depends(get_settings_dep),
):
    """Collapse an enrichment result into a term-similarity network.

    Significant terms become nodes, gene overlap becomes edges, and Leiden groups
    the redundant terms into communities, so a few hundred near-duplicate GO terms
    read as a handful of programs instead of a top-N list of rephrasings.

    Delegates to compute-local for the enrichment itself, so the network always
    reflects exactly what the table would show for the same parameters.
    """
    enrichment = await compute_enrichment_local(
        dataset_id,
        LocalEnrichmentRequest(
            column=body.column,
            group=body.group,
            n_genes=body.n_genes,
            collections=body.collections,
            exclude_ribo_mito=body.exclude_ribo_mito,
        ),
        dm=dm,
        settings=settings,
    )

    try:
        graph = network.build_network(
            [r.model_dump() for r in enrichment.results],
            fdr=body.fdr,
            metric=body.metric,
            min_similarity=body.min_similarity,
            resolution=body.resolution,
            seed=body.seed,
            min_term_size=body.min_term_size,
            max_term_size=body.max_term_size,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    n_edges = len(graph["edges"])
    if not body.include_edges:
        graph["edges"] = []

    return {
        "dataset_id": dataset_id,
        "group": enrichment.group,
        "groupby": enrichment.groupby,
        "n_genes_used": enrichment.n_genes_used,
        "n_significant": graph["n_terms"],
        "n_edges": n_edges,
        **graph,
    }


@router.post("/datasets/{dataset_id}/enrichment/compute-local")
async def compute_enrichment_local(
    dataset_id: str,
    body: LocalEnrichmentRequest,
    dm: DatasetManager = Depends(get_dataset_manager),
    settings: Settings = Depends(get_settings_dep),
):
    """Compute enrichment using local MSigDB subcategory files.

    Uses the top N marker genes for the specified group and runs them through
    gseapy.enrich() with gene sets loaded from local MSigDB JSON files.
    """
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    loader = get_msigdb_loader(settings.MSIGDB_DIR)
    if loader is None:
        raise HTTPException(status_code=500, detail="MSigDB directory not configured.")

    col = body.column
    if not col:
        if not adaptor.has_markers():
            raise HTTPException(
                status_code=404,
                detail="No marker genes available. Run differential expression first.",
            )
    else:
        if not adaptor.has_markers(column=col) and not adaptor.has_markers():
            raise HTTPException(
                status_code=404,
                detail=f"No marker genes available for column '{col}'.",
            )

    # Check for cached result. Payloads without a current provenance sidecar were
    # computed against gseapy's implicit universe and are recomputed rather than served.
    if col:
        cached_key = f"enrichment__{col}__{body.group}"
        if cached_key in adaptor.adata.uns and cache_is_current(
            adaptor.adata,
            col,
            body.group,
            n_genes=body.n_genes,
            collections=body.collections,
            exclude_ribo_mito=body.exclude_ribo_mito,
        ):
            cached = adaptor.adata.uns[cached_key]
            if isinstance(cached, str):
                cached = json.loads(cached)
            normalized = [_normalize_gseapy_record(r) for r in cached] if cached else []
            results = [EnrichmentResult(**r) for r in normalized]
            return EnrichmentResponse(
                group=body.group,
                groupby=col,
                n_genes_used=0,
                results=results,
                source="precomputed",
            )

    # Existence check. n_genes must be passed through: get_markers defaults to 100,
    # which previously capped this endpoint no matter what the caller asked for.
    df = adaptor.get_markers(
        groupby=body.group, column=col if col else None, n_genes=body.n_genes
    )
    if df is None or df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No marker genes found for group '{body.group}'.",
        )

    # Query and background both come from the full rank-ordered marker list. The
    # query drops ribosomal/mitochondrial genes (an scRNA-seq artifact) BEFORE
    # taking the top-n, so the ranked list must be the full one, not the truncated
    # df. The background keeps every measured gene. No re-sort: ORA scores the
    # query as a set, so ordering it cannot change the result.
    background = get_background_genes(adaptor.adata, col, body.group)
    ranked = background if background else df["gene"].tolist()
    top_genes = select_query_genes(
        ranked, body.n_genes, exclude_ribo_mito=body.exclude_ribo_mito
    )

    try:
        import gseapy as gp

        # Load selected MSigDB collections as a merged dict
        gene_sets_dict = loader.get_multiple_collections_as_dict(body.collections)
        if not gene_sets_dict:
            raise HTTPException(
                status_code=400,
                detail="No gene sets found for the selected collections.",
            )

        enr = gp.enrich(
            gene_list=top_genes,
            gene_sets=gene_sets_dict,
            background=background,
            outdir=None,
            no_plot=True,
            cutoff=0.5,
        )

        all_results: list[dict[str, Any]] = []
        if enr.results is not None and not enr.results.empty:
            for _, row in enr.results.iterrows():
                rec = _normalize_gseapy_record(row.to_dict())
                # Try to identify which collection the term belongs to
                rec["collection"] = _identify_collection(rec["term"], body.collections, loader)
                all_results.append(rec)

        # Sort by adjusted p-value
        all_results.sort(key=lambda r: r.get("adjusted_pvalue", 1.0))

        # Cache the result
        if col:
            cache_key = f"enrichment__{col}__{body.group}"
            adaptor.adata.uns[cache_key] = json.dumps(all_results)
            write_meta(
                adaptor.adata,
                col,
                body.group,
                background_n=len(background) if background else None,
                n_genes=body.n_genes,
                collections=body.collections,
                exclude_ribo_mito=body.exclude_ribo_mito,
            )

        return EnrichmentResponse(
            group=body.group,
            groupby=col,
            n_genes_used=len(top_genes),
            results=[EnrichmentResult(**r) for r in all_results],
            source="msigdb_local",
        )

    except ImportError:
        raise HTTPException(
            status_code=500, detail="gseapy is required for enrichment analysis."
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Local enrichment failed for dataset %s group %s: %s",
            dataset_id, body.group, e,
        )
        raise HTTPException(status_code=500, detail=f"Enrichment analysis failed: {e!s}")


@router.post("/datasets/{dataset_id}/enrichment/compute")
async def compute_enrichment(
    dataset_id: str,
    body: EnrichmentComputeRequest,
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """Compute enrichment for a specific cluster / group.

    Uses the top N marker genes for the specified group and runs them through
    gseapy.enrich() when available.  Falls back to returning the raw marker
    gene list if gseapy is not installed.
    """
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    # Resolve which obs column's markers to use
    col = body.column or body.groupby
    if not col:
        # Fall back to whatever markers are available
        if not adaptor.has_markers():
            raise HTTPException(
                status_code=404,
                detail="No marker genes available. Run differential expression first.",
            )
    else:
        if not adaptor.has_markers(column=col) and not adaptor.has_markers():
            raise HTTPException(
                status_code=404,
                detail=f"No marker genes available for column '{col}'.",
            )

    # Check for cached enrichment result. Payloads without a current provenance
    # sidecar predate the background fix and are recomputed rather than served.
    if col:
        cached_key = f"enrichment__{col}__{body.group}"
        if cached_key in adaptor.adata.uns and cache_is_current(
            adaptor.adata,
            col,
            body.group,
            n_genes=body.n_genes,
            collections=body.gene_sets,
            exclude_ribo_mito=body.exclude_ribo_mito,
        ):
            cached = adaptor.adata.uns[cached_key]
            # uns values may be JSON strings (h5ad-safe) or already parsed lists
            if isinstance(cached, str):
                cached = json.loads(cached)
            normalized = [_normalize_gseapy_record(r) for r in cached] if cached else []
            results = [EnrichmentResult(**r) for r in normalized]
            return EnrichmentResponse(
                group=body.group,
                groupby=col,
                n_genes_used=0,
                results=results,
                source="precomputed",
            )

    # Existence check; see compute-local above for the query/background split.
    df = adaptor.get_markers(
        groupby=body.group, column=col if col else None, n_genes=body.n_genes
    )
    if df is None or df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No marker genes found for group '{body.group}'.",
        )

    # Attempt gseapy enrichment
    try:
        import gseapy as gp

        background = get_background_genes(adaptor.adata, col, body.group)
        ranked = background if background else df["gene"].tolist()
        top_genes = select_query_genes(
            ranked, body.n_genes, exclude_ribo_mito=body.exclude_ribo_mito
        )

        enr = gp.enrich(
            gene_list=top_genes,
            gene_sets=body.gene_sets,
            background=background,
            outdir=None,
            no_plot=True,
            cutoff=0.5,
        )

        results: list[dict[str, Any]] = []
        if enr.results is not None and not enr.results.empty:
            for _, row in enr.results.iterrows():
                results.append(_normalize_gseapy_record(row.to_dict()))

        # Cache the result for future lookups
        if col:
            cache_key = f"enrichment__{col}__{body.group}"
            adaptor.adata.uns[cache_key] = json.dumps(results)
            write_meta(
                adaptor.adata,
                col,
                body.group,
                background_n=len(background) if background else None,
                n_genes=body.n_genes,
                collections=body.gene_sets,
                exclude_ribo_mito=body.exclude_ribo_mito,
            )

        return EnrichmentResponse(
            group=body.group,
            groupby=col or body.groupby,
            n_genes_used=len(top_genes),
            results=[EnrichmentResult(**r) for r in results],
            source="gseapy",
        )

    except ImportError:
        logger.info(
            "gseapy not installed; returning marker gene list for group '%s'",
            body.group,
        )

        # Fallback: return the marker genes themselves as pseudo-enrichment
        marker_results = []
        for _, row in df.head(body.n_genes).iterrows():
            pval = float(row.get("pval", 1.0))
            pval_adj = float(row.get("pval_adj", 1.0))

            # Handle NaN values
            if np.isnan(pval):
                pval = 1.0
            if np.isnan(pval_adj):
                pval_adj = 1.0

            marker_results.append(
                EnrichmentResult(
                    term=str(row["gene"]),
                    pvalue=pval,
                    adjusted_pvalue=pval_adj,
                    overlap_count=1,
                    gene_count=1,
                    genes=[str(row["gene"])],
                )
            )

        return EnrichmentResponse(
            group=body.group,
            groupby=body.groupby,
            n_genes_used=len(top_genes),
            results=marker_results,
            source="marker_list",
        )

    except Exception as e:
        logger.error("Enrichment failed for dataset %s group %s: %s", dataset_id, body.group, e)
        raise HTTPException(status_code=500, detail=f"Enrichment analysis failed: {e!s}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize_gseapy_record(row: dict[str, Any]) -> dict[str, Any]:
    """Convert a raw gseapy result dict to our EnrichmentResult schema."""
    # Already in our schema (has 'term' key) — return as-is
    if "term" in row:
        return row

    overlap_str = str(row.get("Overlap", "0/0"))
    parts = overlap_str.split("/")
    overlap_count = int(parts[0]) if len(parts) >= 1 and parts[0].isdigit() else 0
    gene_count = int(parts[1]) if len(parts) >= 2 and parts[1].isdigit() else 0
    genes_str = str(row.get("Genes", ""))
    genes = [g.strip() for g in genes_str.split(";") if g.strip()]

    return {
        "term": str(row.get("Term", "")),
        "pvalue": float(row.get("P-value", 1.0)),
        "adjusted_pvalue": float(row.get("Adjusted P-value", 1.0)),
        "overlap_count": overlap_count,
        "gene_count": gene_count,
        "genes": genes,
    }


def _identify_collection(
    term_name: str, collection_ids: list[str], loader: Any
) -> str:
    """Try to identify which MSigDB collection a term belongs to."""
    for cid in collection_ids:
        data = loader._load_collection(cid)
        if data and term_name in data:
            return cid
    return ""


def _parse_precomputed_enrichment(raw: Any, group_filter: str) -> list[dict]:
    """Best-effort parsing of pre-computed enrichment stored in adata.uns."""
    results: list[dict] = []

    # If it is a DataFrame-like (pandas)
    try:
        import pandas as pd

        if isinstance(raw, pd.DataFrame):
            for _, row in raw.iterrows():
                entry = row.to_dict()
                # Normalise NaN -> None
                entry = {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in entry.items()}
                results.append(entry)
            return results
    except Exception:
        pass

    # If it is a dict of dicts or list of dicts
    if isinstance(raw, dict):
        if group_filter and group_filter in raw:
            sub = raw[group_filter]
            if isinstance(sub, list):
                return sub
            if isinstance(sub, dict):
                return [sub]
        # Return all values flattened
        for k, v in raw.items():
            if isinstance(v, list):
                results.extend(v)
            elif isinstance(v, dict):
                results.append({"group": k, **v})

    if isinstance(raw, list):
        return raw

    return results
