"""Enrichment analysis endpoints – Phase 5.

Provides pre-computed enrichment retrieval and on-demand enrichment via
marker gene lists.  When gseapy is installed, full pathway enrichment is
performed; otherwise the endpoint returns the top marker genes as a
starting point.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from typing import Any, Literal

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from scview.config import Settings
from scview.core import enrichment_network as network
from scview.core import gsea
from scview.core.custom_genesets import get_custom_store, is_custom_id
from scview.core.dataset_manager import DatasetManager
from scview.core.gene_filters import select_query_genes
from scview.core.geneset_sources import (
    dataset_gene_names,
    label_records,
    match_query_case,
    resolve_gene_sets,
    tag_records,
)
from scview.core.msigdb_loader import DEFAULT_COLLECTIONS, get_msigdb_loader
from scview.core.ora_background import (
    cache_is_current,
    get_background_genes,
    resolve_markers_key,
    write_meta,
)
from scview.core.species import detect_species, human_genesets_warning
from scview.core.uns_keys import enrichment_key
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
    # A custom collection's name (MSigDB results leave it empty), and whether
    # the collection has been deleted since this result was computed.
    collection_name: str = ""
    collection_deleted: bool = False


class EnrichmentResponse(BaseModel):
    """Full enrichment response."""

    group: str
    groupby: str
    n_genes_used: int
    results: list[EnrichmentResult]
    source: str  # "gseapy" | "precomputed" | "marker_list" | "msigdb_local"
    # Set when the gene sets (human symbols) may not fit the dataset's species.
    warning: str = ""


# Far above any genome; caps the request body, not a real list.
MAX_GENE_LIST = 100_000


class GeneListEnrichmentRequest(BaseModel):
    """Body for ORA on explicit gene lists (a DE result's significant genes)."""

    up: list[str] = Field(default=[], max_length=MAX_GENE_LIST)
    down: list[str] = Field(default=[], max_length=MAX_GENE_LIST)
    # The genes that were tested, i.e. the universe both lists were drawn from.
    background: list[str] = Field(min_length=1, max_length=MAX_GENE_LIST)
    collections: list[str] = DEFAULT_COLLECTIONS
    # See EnrichmentComputeRequest.exclude_ribo_mito (query only, never the background).
    exclude_ribo_mito: bool = True
    # Most significant terms returned per list. A 500-gene DE list overlaps ~8,000
    # terms of the default collections (Kang CD14 Mono STIM vs CTRL: 4.5 MB of
    # JSON); past the first thousand they are 1-gene overlaps with p near 1.
    max_terms: int = Field(default=1000, ge=1, le=20_000)


class GeneListSide(BaseModel):
    """ORA of one list: how many genes went in, and the enriched terms."""

    n_genes_requested: int
    n_genes_used: int  # in the background and not filtered out
    n_excluded_ribo_mito: int
    n_terms: int  # terms sharing a gene with the list; ``results`` keeps the top max_terms
    n_significant: int  # of n_terms, adjusted p < 0.05
    results: list[EnrichmentResult]


class GeneListEnrichmentResponse(BaseModel):
    background_n: int  # tested genes the dataset measures
    collections: list[str]
    exclude_ribo_mito: bool
    up: GeneListSide
    down: GeneListSide
    source: str = "msigdb_local"
    warning: str = ""  # e.g. a mouse dataset against human gene sets


class GseaGene(BaseModel):
    """One gene of a DE result: GSEA ranks by ``score`` when every gene has one."""

    gene: str
    logfoldchange: float | None = None
    pval: float | None = None
    score: float | None = None


class GseaRequest(BaseModel):
    """Body for preranked GSEA: a DE result's genes, or a marker group."""

    # Every gene the DE tested (core/gsea.py says how they are ranked) ...
    genes: list[GseaGene] = Field(default=[], max_length=MAX_GENE_LIST)
    # ... or a marker group of an obs column, ranked by scanpy's score.
    column: str = ""
    group: str = ""
    rank_by: Literal["auto", "score", "signed_p"] = "auto"
    collections: list[str] = DEFAULT_COLLECTIONS
    # Left out of the ranking altogether (see EnrichmentComputeRequest.exclude_ribo_mito).
    exclude_ribo_mito: bool = True
    min_size: int = Field(default=gsea.MIN_SIZE, ge=5, le=500)
    max_size: int = Field(default=gsea.MAX_SIZE, ge=15, le=2000)
    # None: as many as the cost budget allows (core/gsea.py permutations_for).
    permutations: int | None = Field(default=None, ge=gsea.MIN_PERMUTATIONS, le=gsea.MAX_PERMUTATIONS)
    seed: int = 123
    # Terms returned per direction, most significant first.
    max_terms: int = Field(default=500, ge=1, le=20_000)


class GseaTerm(BaseModel):
    term: str
    es: float
    nes: float  # > 0: the set's genes sit at the top of the ranking
    pvalue: float  # nominal, from permutations; 0 means below 1/permutations
    fdr: float  # GSEA's FDR q-value
    set_size: int  # the set's genes in the ranking
    leading_edge: list[str]
    collection: str = ""
    collection_name: str = ""
    collection_deleted: bool = False


class GseaSide(BaseModel):
    n_terms: int
    n_significant: int  # FDR < 0.05
    results: list[GseaTerm]


class GseaResponse(BaseModel):
    rank_metric: str  # wilcoxon_score | signed_log10_p | marker_score
    rank_metric_label: str
    n_genes_ranked: int
    n_excluded_ribo_mito: int
    n_sets_tested: int
    permutations: int
    min_size: int
    max_size: int
    collections: list[str]
    exclude_ribo_mito: bool
    up: GseaSide  # NES > 0
    down: GseaSide  # NES < 0
    seconds: float  # compute time of the run (0 when served from the cache)
    cached: bool = False
    column: str = ""
    group: str = ""
    warning: str = ""


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

    # The column's own markers, or scanpy's bare key when its groupby matches.
    uns_key = resolve_markers_key(adaptor.adata.uns, column)
    if uns_key is None:
        if column:
            raise HTTPException(status_code=404, detail=f"No markers for column '{column}'.")
        raise HTTPException(
            status_code=404,
            detail="No marker genes found. Run sc.tl.rank_genes_groups first.",
        )

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
            enrichment_key(col_name, g) in adaptor.adata.uns
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
        # Leiden over the term-similarity graph: off the event loop too.
        graph = await asyncio.to_thread(
            lambda: network.build_network(
                [r.model_dump() for r in enrichment.results],
                fdr=body.fdr,
                metric=body.metric,
                min_similarity=body.min_similarity,
                resolution=body.resolution,
                seed=body.seed,
                min_term_size=body.min_term_size,
                max_term_size=body.max_term_size,
            )
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
        "warning": enrichment.warning,
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
    """Compute enrichment using local MSigDB subcategory files and custom collections.

    Uses the top N marker genes for the specified group and runs them through
    gseapy.enrich() with gene sets loaded from local MSigDB JSON files and/or
    the user's custom collections (ids starting "custom-").
    """
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    # The background genes and gseapy take seconds: in a worker thread, so the
    # server keeps answering meanwhile.
    return await asyncio.to_thread(_compute_local, dataset_id, adaptor, body, settings)


def _compute_local(
    dataset_id: str, adaptor, body: LocalEnrichmentRequest, settings: Settings
) -> EnrichmentResponse:
    loader = get_msigdb_loader(settings.MSIGDB_DIR)
    store = get_custom_store(settings.DATA_DIR)
    # Custom collections need no MSigDB, so only an MSigDB-only request fails here.
    if loader is None and not any(is_custom_id(c) for c in body.collections):
        raise HTTPException(status_code=500, detail="MSigDB directory not configured.")
    warning = human_genesets_warning(detect_species(adaptor.adata))

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
        cached_key = enrichment_key(col, body.group)
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
            # A custom collection deleted since then is shown as deleted.
            results = [EnrichmentResult(**r) for r in label_records(normalized, store)]
            return EnrichmentResponse(
                group=body.group,
                groupby=col,
                n_genes_used=0,
                results=results,
                source="precomputed",
                warning=warning,
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
        all_results = _ora_records(top_genes, background, body.collections, loader, store)

        # Cache the result
        if col:
            cache_key = enrichment_key(col, body.group)
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
            results=[EnrichmentResult(**r) for r in label_records(all_results, store)],
            source="msigdb_local",
            warning=warning,
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
    # As compute-local: off the event loop.
    return await asyncio.to_thread(_compute, dataset_id, adaptor, body)


def _compute(dataset_id: str, adaptor, body: EnrichmentComputeRequest) -> EnrichmentResponse:
    warning = human_genesets_warning(detect_species(adaptor.adata))
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
        cached_key = enrichment_key(col, body.group)
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
                warning=warning,
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
            cache_key = enrichment_key(col, body.group)
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
            warning=warning,
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
            warning=warning,
        )

    except Exception as e:
        logger.error("Enrichment failed for dataset %s group %s: %s", dataset_id, body.group, e)
        raise HTTPException(status_code=500, detail=f"Enrichment analysis failed: {e!s}")


@router.post(
    "/datasets/{dataset_id}/enrichment/gene-lists",
    response_model=GeneListEnrichmentResponse,
)
async def compute_enrichment_gene_lists(
    dataset_id: str,
    body: GeneListEnrichmentRequest,
    dm: DatasetManager = Depends(get_dataset_manager),
    settings: Settings = Depends(get_settings_dep),
) -> GeneListEnrichmentResponse:
    """Over-representation of explicit gene lists, e.g. a DE result's up and down genes.

    ``background`` is the set the lists were drawn from (the genes the DE tested),
    so the hypergeometric test asks "more of this set than chance among the tested
    genes", not among every gene MSigDB names. Each list is tested on its own:
    pooling up and down would mix two programs into one weaker one.
    """
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    # Two gseapy runs over the selected collections: off the event loop.
    return await asyncio.to_thread(_compute_gene_lists, dataset_id, adaptor, body, settings)


def _compute_gene_lists(
    dataset_id: str, adaptor, body: GeneListEnrichmentRequest, settings: Settings
) -> GeneListEnrichmentResponse:
    loader = get_msigdb_loader(settings.MSIGDB_DIR)
    store = get_custom_store(settings.DATA_DIR)
    if loader is None and not any(is_custom_id(c) for c in body.collections):
        raise HTTPException(status_code=500, detail="MSigDB directory not configured.")

    # Only genes this dataset measures: a list from another dataset (or a typo)
    # must not widen the universe the test is conditioned on.
    measured = dataset_gene_names(adaptor)
    background = [g for g in dict.fromkeys(body.background) if g in measured]
    if not background:
        raise HTTPException(
            status_code=400, detail="None of the background genes are in this dataset."
        )
    universe = set(background)

    def side(genes: list[str]) -> GeneListSide:
        # A query gene outside the universe would break the hypergeometric test.
        query = [g for g in dict.fromkeys(genes) if g in universe]
        kept = select_query_genes(query, len(query), exclude_ribo_mito=body.exclude_ribo_mito)
        records = _ora_records(kept, background, body.collections, loader, store) if kept else []
        top = records[: body.max_terms]  # already most significant first
        return GeneListSide(
            n_genes_requested=len(genes),
            n_genes_used=len(kept),
            n_excluded_ribo_mito=len(query) - len(kept),
            n_terms=len(records),
            n_significant=sum(r.get("adjusted_pvalue", 1.0) < 0.05 for r in records),
            results=[EnrichmentResult(**r) for r in label_records(top, store)],
        )

    try:
        up, down = side(body.up), side(body.down)
    except ImportError:
        raise HTTPException(status_code=500, detail="gseapy is required for enrichment analysis.")
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Gene-list enrichment failed for dataset %s: %s", dataset_id, e)
        raise HTTPException(status_code=500, detail=f"Enrichment analysis failed: {e!s}")

    return GeneListEnrichmentResponse(
        background_n=len(background),
        collections=list(body.collections),
        exclude_ribo_mito=body.exclude_ribo_mito,
        up=up,
        down=down,
        warning=human_genesets_warning(detect_species(adaptor.adata)),
    )


@router.post("/datasets/{dataset_id}/enrichment/gsea", response_model=GseaResponse)
async def compute_gsea(
    dataset_id: str,
    body: GseaRequest,
    dm: DatasetManager = Depends(get_dataset_manager),
    settings: Settings = Depends(get_settings_dep),
) -> GseaResponse:
    """Preranked GSEA of a DE result (``genes``) or a marker group (``column`` +
    ``group``) against the chosen collections: NES, nominal p, FDR and the
    leading-edge genes of each set. The same ranking and parameters again are
    served from a cache."""
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    # Seconds to tens of seconds of permutations: in a worker thread (gseapy's
    # Rust core releases the GIL), so the server keeps answering meanwhile.
    return await asyncio.to_thread(_compute_gsea, adaptor, body, settings)


def _gsea_ranking(adaptor, body: GseaRequest) -> gsea.Ranking:
    if body.genes and body.group:
        raise HTTPException(status_code=400, detail="Send a DE result's genes or a marker group, not both.")
    try:
        if body.genes:
            return gsea.ranking_from_de(
                [g.model_dump() for g in body.genes],
                rank_by=body.rank_by, exclude_ribo_mito=body.exclude_ribo_mito,
            )
        if not body.group:
            raise HTTPException(status_code=400, detail="Send a DE result's genes or a marker group.")
        key = resolve_markers_key(adaptor.adata.uns, body.column)
        if key is None:
            raise HTTPException(status_code=404, detail=f"No marker genes for column '{body.column}'.")
        return gsea.ranking_from_markers(
            adaptor.adata.uns[key], body.group, exclude_ribo_mito=body.exclude_ribo_mito
        )
    except gsea.GseaError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


def _collections_fingerprint(collections: list[str], store) -> list[str]:
    """Collection ids for the cache key; a custom collection can be edited, so its
    current contents count too (MSigDB files are versioned and fixed)."""
    out = []
    for cid in collections:
        if is_custom_id(cid):
            sets = store.get_sets(cid) if store else None
            blob = json.dumps(sets, sort_keys=True, default=str).encode()
            cid = f"{cid}:{hashlib.sha256(blob).hexdigest()[:16]}"
        out.append(cid)
    return out


def _compute_gsea(adaptor, body: GseaRequest, settings: Settings) -> GseaResponse:
    loader = get_msigdb_loader(settings.MSIGDB_DIR)
    store = get_custom_store(settings.DATA_DIR)
    if loader is None and not any(is_custom_id(c) for c in body.collections):
        raise HTTPException(status_code=500, detail="MSigDB directory not configured.")
    ranking = _gsea_ranking(adaptor, body)

    params = {
        "min_size": body.min_size, "max_size": body.max_size,
        "permutations": body.permutations, "seed": body.seed,
    }
    key = gsea.cache_key(ranking, _collections_fingerprint(body.collections, store), params)
    run = gsea.cached(key)
    hit = run is not None
    if run is None:
        resolved = resolve_gene_sets(body.collections, loader, store)
        if resolved.missing_custom:
            raise HTTPException(
                status_code=400,
                detail=(
                    "A selected custom collection no longer exists (it was deleted): "
                    f"{', '.join(resolved.missing_custom)}. Deselect it and run again."
                ),
            )
        if not resolved.gene_sets:
            raise HTTPException(status_code=400, detail="No gene sets found for the selected collections.")
        sets = match_query_case(resolved, ranking.genes, ranking.genes)
        try:
            run = gsea.run_prerank(
                ranking, sets, min_size=body.min_size, max_size=body.max_size,
                permutations=body.permutations, seed=body.seed,
            )
        except gsea.GseaError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except ImportError:
            raise HTTPException(status_code=500, detail="gseapy is required for GSEA.")
        except Exception as e:
            logger.error("GSEA failed: %s", e, exc_info=True)
            raise HTTPException(status_code=500, detail=f"GSEA failed: {e!s}")
        tag_records(run.up + run.down, resolved)
        gsea.remember(key, run)
        logger.info(
            "GSEA: %d genes, %d sets, %d permutations in %.1f s",
            len(ranking.genes), run.n_sets, run.permutations, run.seconds,
        )

    def side(records: list[dict]) -> GseaSide:
        top = records[: body.max_terms]
        return GseaSide(
            n_terms=len(records),
            n_significant=sum(r["fdr"] < gsea.SIGNIFICANT_FDR for r in records),
            results=[GseaTerm(**r) for r in label_records(top, store)],
        )

    return GseaResponse(
        rank_metric=ranking.metric,
        rank_metric_label=gsea.RANK_LABELS[ranking.metric],
        n_genes_ranked=len(ranking.genes),
        n_excluded_ribo_mito=ranking.n_excluded_ribo_mito,
        n_sets_tested=run.n_sets,
        permutations=run.permutations,
        min_size=body.min_size,
        max_size=body.max_size,
        collections=list(body.collections),
        exclude_ribo_mito=body.exclude_ribo_mito,
        up=side(run.up),
        down=side(run.down),
        seconds=0.0 if hit else round(run.seconds, 2),
        cached=hit,
        column=body.column if body.group else "",
        group=body.group,
        warning=human_genesets_warning(detect_species(adaptor.adata)),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ora_records(
    query: list[str],
    background: list[str] | None,
    collections: list[str],
    loader,
    store,
) -> list[dict[str, Any]]:
    """gseapy ORA of ``query`` against the selected MSigDB/custom collections over
    ``background``: normalised records tagged with their collection, most
    significant first. A deleted custom collection or an empty selection is a 400."""
    import gseapy as gp

    # Load the selected MSigDB and custom collections as one merged dict
    resolved = resolve_gene_sets(collections, loader, store)
    if resolved.missing_custom:
        raise HTTPException(
            status_code=400,
            detail=(
                "A selected custom collection no longer exists (it was deleted): "
                f"{', '.join(resolved.missing_custom)}. Deselect it and run again."
            ),
        )
    if not resolved.gene_sets:
        raise HTTPException(
            status_code=400,
            detail="No gene sets found for the selected collections.",
        )
    gene_sets_dict = match_query_case(resolved, query, background)

    enr = gp.enrich(
        gene_list=query,
        gene_sets=gene_sets_dict,
        background=background,
        outdir=None,
        no_plot=True,
        cutoff=0.5,
    )

    records: list[dict[str, Any]] = []
    # With no overlap at all gseapy leaves results as an empty list, not a frame.
    if not getattr(enr.results, "empty", True):
        for _, row in enr.results.iterrows():
            records.append(_normalize_gseapy_record(row.to_dict()))
    # Which collection each term came from (and a custom one's name)
    tag_records(records, resolved)

    # Sort by adjusted p-value
    records.sort(key=lambda r: r.get("adjusted_pvalue", 1.0))
    return records


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
