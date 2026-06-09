"""Preprocessing pipeline — run standard scanpy steps on an AnnData object.

This module applies standard preprocessing steps (normalization, HVG selection,
PCA, neighbor graph, clustering, UMAP) to an AnnData object in-place,
then saves the result to disk.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Generator

import anndata as ad
import numpy as np
import scanpy as sc

from scview.core import provenance

logger = logging.getLogger(__name__)

# Callback for sub-step progress: (message, current_index, total_count)
SubstepCallback = Callable[[str, int, int], None]

# No-op default for runners that don't support callbacks
_NOOP_CB: SubstepCallback = lambda _msg, _cur, _tot: None


@dataclass
class PipelineParams:
    """Parameters for each preprocessing step."""

    # Filtering
    min_genes: int = 200
    min_cells: int = 3
    max_pct_mt: float = 20.0
    drop_doublets: bool = False  # drop cells flagged by doublet detection (opt-in)

    # Doublet detection
    doublet_method: str = "scrublet"
    expected_doublet_rate: float = 0.06

    # Normalization
    target_sum: float = 1e4

    # HVG
    n_top_genes: int = 2000
    hvg_flavor: str = "seurat"

    # PCA
    n_comps: int = 50

    # Neighbors
    n_neighbors: int = 15

    # Clustering
    clustering_resolution: float = 0.5
    clustering_method: str = "leiden"

    # Batch correction
    batch_key: str = ""  # obs column for batch identity (empty = skip)
    batch_method: str = "harmony"

    # UMAP
    umap_min_dist: float = 0.5

    # Marker genes: obs columns to compute markers for (empty = single-column fallback)
    marker_columns: list[str] = field(default_factory=list)

    # Enrichment: obs columns to compute enrichment for (empty = skip)
    enrichment_columns: list[str] = field(default_factory=list)
    enrichment_n_genes: int = 100
    enrichment_collections: list[str] = field(default_factory=lambda: [
        "h.all", "c2.cp.kegg_medicus", "c2.cp.reactome", "c2.cp.wikipathways",
        "c5.go.bp", "c5.go.cc", "c5.go.mf", "c8.all",
    ])
    # Legacy field for backward compat — ignored if enrichment_collections is set
    enrichment_gene_sets: list[str] = field(default_factory=list)

    # Cell-type annotation
    annotation_method: str = "llm"               # llm (default, any-tissue) | celltypist | marker_score
    celltypist_model: str = "Immune_All_Low.pkl"  # CellTypist model (immune default)
    annotation_groupby: str = ""                 # cluster column for consensus (empty = active clustering / "cluster")
    annotation_target: str = "cell_type"         # obs column to write
    annotation_llm_model: str = ""               # DeepInfra model id for method="llm"
    annotation_tissue: str = ""                  # optional tissue hint for method="llm"


@dataclass
class PipelineResult:
    """Result of running the preprocessing pipeline."""

    steps_run: list[str] = field(default_factory=list)
    steps_skipped: list[str] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    output_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Ordered list of all steps
ALL_STEPS = [
    "reset_to_counts",
    "qc_metrics",
    "doublet_detection",
    "filtering",
    "normalization",
    "log_transform",
    "highly_variable_genes",
    "scaling",
    "pca",
    "batch_correction",
    "neighbors",
    "clustering",
    "embeddings",
    "marker_genes",
    "enrichment",
    "cell_cycle",
    "cell_type_annotation",
]


def _run_reset_to_counts(adata: ad.AnnData, params: PipelineParams) -> ad.AnnData:
    """Reset adata.X to raw counts, clearing all preprocessing artifacts.

    Returns a fresh AnnData ready for reprocessing from scratch.
    """
    # Restore counts from layers or raw
    if "counts" in adata.layers:
        adata.X = adata.layers["counts"].copy()
        logger.info("Restored counts from adata.layers['counts']")
    elif adata.raw is not None:
        logger.warning("No counts layer found; restoring from adata.raw (may be normalized)")
        adata = adata.raw.to_adata()
    else:
        logger.warning("No counts layer or raw found; proceeding with current X")
        return adata

    # Clear preprocessing artifacts
    adata.raw = None

    # Remove HVG annotation from var
    for col in ["highly_variable", "means", "dispersions", "dispersions_norm"]:
        if col in adata.var.columns:
            del adata.var[col]

    # Clear embeddings
    keys_to_clear = [k for k in adata.obsm_keys() if k.startswith("X_")]
    for k in keys_to_clear:
        del adata.obsm[k]

    # Clear neighbor graphs
    for k in list(adata.obsp.keys()):
        del adata.obsp[k]

    # Clear uns entries (fixed keys + multi-column marker/enrichment keys)
    for key in ["neighbors", "rank_genes_groups", "scview_active_clustering",
                "pca", "umap", "leiden", "louvain"]:
        if key in adata.uns:
            del adata.uns[key]
    multi_keys = [k for k in list(adata.uns.keys())
                  if k.startswith("rank_genes_groups__") or k.startswith("enrichment__")]
    for key in multi_keys:
        del adata.uns[key]

    # Remove scview-generated obs columns
    scview_cols = [c for c in adata.obs.columns if c.startswith("scview_")]
    if scview_cols:
        adata.obs.drop(columns=scview_cols, inplace=True, errors="ignore")

    logger.info("Reset to counts (%d cells, %d genes)", adata.n_obs, adata.n_vars)
    return adata


def _run_qc_metrics(adata: ad.AnnData, params: PipelineParams) -> None:
    """Calculate QC metrics including mitochondrial gene percentage."""
    adata.var["mt"] = adata.var_names.str.startswith("MT-") | adata.var_names.str.startswith("mt-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)
    logger.info(
        "QC metrics: median genes/cell=%d, median counts/cell=%d",
        int(adata.obs["n_genes_by_counts"].median()),
        int(adata.obs["total_counts"].median()),
    )


def _run_doublet_detection(adata: ad.AnnData, params: PipelineParams) -> None:
    """Detect doublets with Scrublet; annotate ``.obs`` only (removal is opt-in).

    Writes ``doublet_score`` (float) and ``predicted_doublet`` (bool) to ``.obs``.
    Ordered before normalization so Scrublet sees raw counts. The flags are
    informational here — actual removal happens in the filtering step when
    ``params.drop_doublets`` is set, so users can inspect the QC distribution first.
    """
    method = (params.doublet_method or "scrublet").lower()
    if method != "scrublet":
        logger.warning("Unknown doublet method '%s'; falling back to scrublet", method)
    sc.pp.scrublet(adata, expected_doublet_rate=params.expected_doublet_rate)
    n_dbl = (
        int(adata.obs["predicted_doublet"].sum())
        if "predicted_doublet" in adata.obs.columns
        else 0
    )
    logger.info(
        "Doublet detection (scrublet): %d / %d cells flagged as doublets",
        n_dbl, adata.n_obs,
    )


def _run_filtering(adata: ad.AnnData, params: PipelineParams) -> ad.AnnData:
    """Filter cells and genes based on QC thresholds. Returns new AnnData."""
    n_before = adata.n_obs
    sc.pp.filter_cells(adata, min_genes=params.min_genes)
    sc.pp.filter_genes(adata, min_cells=params.min_cells)

    # Filter by mitochondrial percentage if QC metrics are available
    if "pct_counts_mt" in adata.obs.columns:
        adata = adata[adata.obs["pct_counts_mt"] < params.max_pct_mt, :].copy()

    # Drop predicted doublets (opt-in; requires the doublet_detection step to have run)
    if params.drop_doublets and "predicted_doublet" in adata.obs.columns:
        n_dbl = int(adata.obs["predicted_doublet"].astype(bool).sum())
        adata = adata[~adata.obs["predicted_doublet"].astype(bool), :].copy()
        logger.info("Filtering: dropped %d predicted doublets", n_dbl)

    logger.info("Filtering: %d -> %d cells", n_before, adata.n_obs)
    return adata


def _run_normalization(adata: ad.AnnData, params: PipelineParams) -> None:
    """Normalize to target sum per cell, store raw counts."""
    # Save raw counts before normalization
    if adata.raw is None and "counts" not in adata.layers:
        adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=params.target_sum)
    logger.info("Normalized to target sum %.0f", params.target_sum)


def _run_log_transform(adata: ad.AnnData, params: PipelineParams) -> None:
    """Log-transform expression values (log1p)."""
    sc.pp.log1p(adata)
    logger.info("Applied log1p transformation")


def _run_hvg(adata: ad.AnnData, params: PipelineParams) -> None:
    """Identify highly variable genes."""
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=params.n_top_genes,
        flavor=params.hvg_flavor,
    )
    n_hvg = int(adata.var["highly_variable"].sum())
    logger.info("Selected %d highly variable genes (flavor=%s)", n_hvg, params.hvg_flavor)


def _run_scaling(adata: ad.AnnData, params: PipelineParams) -> None:
    """Scale expression to unit variance and zero mean."""
    # Store the full data before subsetting to HVGs
    adata.raw = adata
    sc.pp.scale(adata, max_value=10)
    logger.info("Scaled expression (max_value=10)")


def _run_pca(adata: ad.AnnData, params: PipelineParams) -> None:
    """Run PCA."""
    n_comps = min(params.n_comps, adata.n_obs - 1, adata.n_vars - 1)
    sc.tl.pca(adata, n_comps=n_comps, svd_solver="arpack")
    logger.info("PCA: %d components", n_comps)


def _run_batch_correction(adata: ad.AnnData, params: PipelineParams) -> None:
    """Run Harmony integration to correct batch effects in PCA space."""
    if not params.batch_key:
        raise ValueError("batch_key parameter is required for batch correction.")
    if params.batch_key not in adata.obs.columns:
        raise ValueError(f"Batch key '{params.batch_key}' not found in obs columns.")
    if "X_pca" not in adata.obsm:
        raise ValueError("PCA must be computed before batch correction.")

    # Call harmonypy directly rather than scanpy's harmony_integrate: scanpy
    # unconditionally transposes Harmony's output, which breaks with harmonypy
    # >= 2.0 (where Z_corr is already cells × PCs) — producing a (n_pcs, n_cells)
    # array that fails obsm validation. Orient to (n_cells, n_pcs) ourselves so
    # the step works across harmonypy versions.
    import harmonypy

    ho = harmonypy.run_harmony(adata.obsm["X_pca"], adata.obs, [params.batch_key])
    Z = np.asarray(ho.Z_corr)
    n = adata.n_obs
    if Z.shape[0] != n and Z.ndim == 2 and Z.shape[1] == n:
        Z = Z.T  # old harmonypy returns PCs × cells
    if Z.shape[0] != n:
        raise ValueError(
            f"Harmony output shape {Z.shape} does not match n_cells={n}."
        )
    adata.obsm["X_pca_harmony"] = np.ascontiguousarray(Z)
    logger.info("Harmony batch correction using key '%s' -> X_pca_harmony %s",
                params.batch_key, adata.obsm["X_pca_harmony"].shape)


def _run_neighbors(adata: ad.AnnData, params: PipelineParams) -> None:
    """Compute nearest neighbor graph."""
    # Use harmony-corrected PCA if available
    use_rep = "X_pca_harmony" if "X_pca_harmony" in adata.obsm else "X_pca"
    sc.pp.neighbors(adata, n_neighbors=params.n_neighbors, use_rep=use_rep)
    logger.info("Neighbor graph: k=%d, use_rep=%s", params.n_neighbors, use_rep)


def _run_clustering(adata: ad.AnnData, params: PipelineParams) -> None:
    """Run clustering (Leiden or Louvain).

    Stores results in a custom column ``scview_{method}_r{resolution}`` so that
    the original dataset's clustering column (if any) is preserved.  The active
    column name is recorded in ``adata.uns["scview_active_clustering"]`` for
    downstream steps (marker genes, violin groupby, etc.).
    """
    method = params.clustering_method
    if method == "leiden":
        sc.tl.leiden(adata, resolution=params.clustering_resolution)
    else:
        sc.tl.louvain(adata, resolution=params.clustering_resolution)

    # Build custom column name to preserve the original
    scview_col = f"scview_{method}_r{params.clustering_resolution}"
    adata.obs[scview_col] = adata.obs[method].copy()
    adata.uns["scview_active_clustering"] = scview_col

    n_clusters = adata.obs[scview_col].nunique()
    logger.info(
        "Clustering (%s): %d clusters at resolution %.2f → column '%s'",
        method, n_clusters, params.clustering_resolution, scview_col,
    )


def _run_embeddings(adata: ad.AnnData, params: PipelineParams) -> None:
    """Compute 2D and 3D UMAP embeddings."""
    # 3D UMAP first (stored temporarily in X_umap, then moved)
    sc.tl.umap(adata, min_dist=params.umap_min_dist, n_components=3)
    adata.obsm["X_umap_3d"] = adata.obsm["X_umap"].copy()

    # 2D UMAP (overwrites X_umap)
    sc.tl.umap(adata, min_dist=params.umap_min_dist, n_components=2)
    logger.info("UMAP computed: 2D and 3D (min_dist=%.2f)", params.umap_min_dist)


def _run_marker_genes(adata: ad.AnnData, params: PipelineParams, cb: SubstepCallback = _NOOP_CB) -> None:
    """Find marker genes, optionally for multiple obs columns."""
    columns = list(params.marker_columns) if params.marker_columns else []

    # Auto-include active clustering column created by a preceding clustering step
    active_clust = adata.uns.get("scview_active_clustering")
    if active_clust and active_clust in adata.obs.columns and active_clust not in columns:
        columns.insert(0, active_clust)
        logger.info("Auto-including active clustering column for markers: %s", active_clust)

    if not columns:
        # Backward-compatible: single column fallback
        groupby = adata.uns.get("scview_active_clustering")
        if groupby and groupby in adata.obs.columns:
            columns = [groupby]
        else:
            for candidate in ("leiden", "louvain", "cluster", "clusters",
                              "seurat_clusters", "cell_type"):
                if candidate in adata.obs.columns:
                    columns = [candidate]
                    break
        if not columns:
            raise ValueError("No clustering column found. Run clustering first.")

    active_clustering = adata.uns.get("scview_active_clustering", "")
    computed = 0

    for i, col in enumerate(columns):
        if col not in adata.obs.columns:
            logger.warning("Column '%s' not in obs, skipping markers", col)
            continue
        n_unique = adata.obs[col].nunique()
        if n_unique < 2:
            logger.warning("Column '%s' has < 2 unique values, skipping markers", col)
            continue
        if n_unique > 100:
            logger.warning("Column '%s' has %d unique values (> 100), skipping markers", col, n_unique)
            continue

        logger.info("Computing markers for column '%s' (%d/%d)", col, i + 1, len(columns))
        cb(f"Markers for '{col}'", i, len(columns))
        # rank_genes_groups needs a categorical groupby — coerce bool/numeric/object
        # columns (e.g. predicted_doublet) so they don't raise a .cat accessor error.
        if adata.obs[col].dtype.name != "category":
            adata.obs[col] = adata.obs[col].astype(str).astype("category")
        sc.tl.rank_genes_groups(adata, groupby=col, method="wilcoxon", pts=True)
        # Store under namespaced key
        adata.uns[f"rank_genes_groups__{col}"] = adata.uns["rank_genes_groups"].copy()
        computed += 1

    # Leave rank_genes_groups pointing to the active clustering for backward compat
    if active_clustering and f"rank_genes_groups__{active_clustering}" in adata.uns:
        adata.uns["rank_genes_groups"] = adata.uns[f"rank_genes_groups__{active_clustering}"]
    elif computed > 0:
        # Point to the first computed column
        first_key = next(k for k in adata.uns if k.startswith("rank_genes_groups__"))
        adata.uns["rank_genes_groups"] = adata.uns[first_key]

    logger.info("Marker genes computed for %d column(s)", computed)


def _run_enrichment(adata: ad.AnnData, params: PipelineParams, cb: SubstepCallback = _NOOP_CB) -> None:
    """Run pathway enrichment for marker genes across specified obs columns.

    Uses local MSigDB JSON files via MSigDBLoader when enrichment_collections
    is specified; falls back to Enrichr string library names via
    enrichment_gene_sets for backward compatibility.
    """
    columns = list(params.enrichment_columns) if params.enrichment_columns else []

    # Auto-include active clustering column if set by a preceding clustering step
    active_clust = adata.uns.get("scview_active_clustering")
    if active_clust and active_clust in adata.obs.columns and active_clust not in columns:
        columns.insert(0, active_clust)
        logger.info("Auto-including active clustering column for enrichment: %s", active_clust)

    if not columns:
        logger.info("No enrichment columns specified, skipping")
        return

    try:
        import gseapy as gp
    except ImportError:
        logger.warning("gseapy not installed; skipping enrichment pipeline step")
        return

    # Determine gene_sets source: local MSigDB dicts or Enrichr names
    gene_sets_arg: Any  # dict[str, list[str]] or list[str]
    if params.enrichment_collections:
        import os
        from scview.core.msigdb_loader import get_msigdb_loader

        msigdb_dir = os.environ.get("MSIGDB_DIR", "/msigdb")
        loader = get_msigdb_loader(msigdb_dir)
        if loader:
            gene_sets_arg = loader.get_multiple_collections_as_dict(params.enrichment_collections)
            if not gene_sets_arg:
                logger.warning("No gene sets loaded from MSigDB collections: %s", params.enrichment_collections)
                return
            logger.info("Using %d local MSigDB gene sets from %d collections", len(gene_sets_arg), len(params.enrichment_collections))
        else:
            logger.warning("MSigDB loader not available; falling back to Enrichr libraries")
            gene_sets_arg = params.enrichment_gene_sets or ["GO_Biological_Process_2025"]
    elif params.enrichment_gene_sets:
        gene_sets_arg = params.enrichment_gene_sets
    else:
        logger.warning("No enrichment collections or gene sets specified, skipping")
        return

    # Build a flat list of (col, uns_key, groups) for progress tracking
    work_items: list[tuple[str, str, str]] = []  # (col, uns_key, group)
    for col in columns:
        uns_key = f"rank_genes_groups__{col}"
        if uns_key not in adata.uns:
            if "rank_genes_groups" in adata.uns:
                rgg = adata.uns["rank_genes_groups"]
                if rgg.get("params", {}).get("groupby") == col:
                    uns_key = "rank_genes_groups"
                else:
                    logger.warning("No markers for column '%s', skipping enrichment", col)
                    continue
            else:
                logger.warning("No markers for column '%s', skipping enrichment", col)
                continue

        rgg = adata.uns[uns_key]
        groups = list(rgg["names"].dtype.names) if hasattr(rgg["names"].dtype, "names") else []
        for group in groups:
            work_items.append((col, uns_key, group))

    total_items = len(work_items)
    total_groups = 0

    for wi, (col, uns_key, group) in enumerate(work_items):
        cb(f"Enrichment: {col} / {group}", wi, total_items)

        rgg = adata.uns[uns_key]
        total_genes = len(rgg["names"][group])
        limit = min(params.enrichment_n_genes, total_genes)
        top_genes = [str(rgg["names"][group][i]) for i in range(limit)]

        try:
            enr = gp.enrich(
                gene_list=top_genes,
                gene_sets=gene_sets_arg,
                outdir=None,
                no_plot=True,
                cutoff=0.5,
            )
            store_key = f"enrichment__{col}__{group}"
            if enr.results is not None and not enr.results.empty:
                from scview.api.v1.enrichment import _normalize_gseapy_record
                normalized = [
                    _normalize_gseapy_record(r) for r in enr.results.to_dict(orient="records")
                ]
                adata.uns[store_key] = json.dumps(normalized)
            else:
                adata.uns[store_key] = "[]"
            total_groups += 1
        except Exception as e:
            logger.warning("Enrichment failed for %s/%s: %s", col, group, e)

    logger.info("Enrichment completed: %d total group results", total_groups)


def _run_cell_cycle(adata: ad.AnnData, params: PipelineParams) -> None:
    """Score cell cycle phases using scanpy's built-in gene sets."""
    try:
        s_genes = [
            "MCM5", "PCNA", "TYMS", "FEN1", "MCM2", "MCM4", "RRM1", "UNG",
            "GINS2", "MCM6", "CDCA7", "DTL", "PRIM1", "UHRF1", "MLF1IP",
            "HELLS", "RFC2", "RPA2", "NASP", "RAD51AP1", "GMNN", "WDR76",
            "SLBP", "CCNE2", "UBR7", "POLD3", "MSH2", "ATAD2", "RAD51",
            "RRM2", "CDC45", "CDC6", "EXO1", "TIPIN", "DSCC1", "BLM",
            "CASP8AP2", "USP1", "CLSPN", "POLA1", "CHAF1B", "BRIP1", "E2F8",
        ]
        g2m_genes = [
            "HMGB2", "CDK1", "NUSAP1", "UBE2C", "BIRC5", "TPX2", "TOP2A",
            "NDC80", "CKS2", "NUF2", "CKS1B", "MKI67", "TMPO", "CENPF",
            "TACC3", "FAM64A", "SMC4", "CCNB2", "CKAP2L", "CKAP2", "AURKB",
            "BUB1", "KIF11", "ANP32E", "TUBB4B", "GTSE1", "KIF20B", "HJURP",
            "CDCA3", "HN1", "CDC20", "TTK", "CDC25C", "KIF2C", "RANGAP1",
            "NCAPD2", "DLGAP5", "CDCA2", "CDCA8", "ECT2", "KIF23", "HMMR",
            "AURKA", "PSRC1", "ANLN", "LBR", "CKAP5", "CENPE", "CTCF",
            "NEK2", "G2E3", "GAS2L3", "CBX5", "CENPA",
        ]
        # Filter to genes present in the dataset
        var_names_upper = [g.upper() for g in adata.var_names]
        s_present = [g for g in s_genes if g.upper() in var_names_upper]
        g2m_present = [g for g in g2m_genes if g.upper() in var_names_upper]

        if len(s_present) < 5 or len(g2m_present) < 5:
            logger.warning(
                "Too few cell cycle genes found (S: %d, G2M: %d). Skipping.",
                len(s_present),
                len(g2m_present),
            )
            raise ValueError(
                f"Too few cell cycle genes found (S: {len(s_present)}, G2M: {len(g2m_present)}). "
                "Need at least 5 of each."
            )

        sc.tl.score_genes_cell_cycle(adata, s_genes=s_present, g2m_genes=g2m_present)
        logger.info("Cell cycle scored (%d S genes, %d G2M genes)", len(s_present), len(g2m_present))
    except Exception as e:
        raise ValueError(f"Cell cycle scoring failed: {e}") from e


def _resolve_annotation_groupby(adata: ad.AnnData, params: PipelineParams) -> str:
    """Clustering column to take per-cluster consensus over, or "" if none exists.

    When a clustering is present we pass it to CellTypist as ``over_clustering`` so
    labels align 1:1 with the user's clusters; when "", CellTypist computes its own
    over-clustering for majority voting (the canonical no-clustering call)."""
    groupby = params.annotation_groupby or adata.uns.get("scview_active_clustering") or ""
    if groupby and groupby in adata.obs.columns:
        return groupby
    for cand in ("cluster", "leiden", "louvain"):
        if cand in adata.obs.columns and adata.obs[cand].nunique() > 1:
            return cand
    return ""  # no clustering: let CellTypist over-cluster internally


def _annotate_celltypist(
    adata: ad.AnnData, params: PipelineParams, groupby: str, target: str
) -> None:
    """Reference-based annotation with CellTypist, per-cluster consensus (majority voting)."""
    try:
        import celltypist
        from celltypist import models
    except ImportError as e:  # pragma: no cover - dependency guard
        raise ValueError(
            "CellTypist is not installed. Add 'celltypist' to the backend image."
        ) from e

    model_name = params.celltypist_model or "Immune_All_Low.pkl"
    if not model_name.endswith(".pkl"):
        model_name += ".pkl"
    try:
        models.download_models(model=[model_name], force_update=False)
    except Exception as e:  # already cached, or offline — fail only if load fails below
        logger.warning("CellTypist model fetch issue (%s); trying cached copy.", e)

    # CellTypist expects log1p of counts-per-10k. Build that from raw counts so the
    # result is correct regardless of the current X state (scaled/HVG-subset/etc.).
    if adata.raw is not None:
        counts, var_names = adata.raw.X, list(adata.raw.var_names)
    elif "counts" in adata.layers:
        counts, var_names = adata.layers["counts"], list(adata.var_names)
    else:
        counts, var_names = adata.X, list(adata.var_names)
    inp = ad.AnnData(X=counts.copy() if hasattr(counts, "copy") else counts)
    inp.obs_names = adata.obs_names
    inp.var_names = var_names
    sc.pp.normalize_total(inp, target_sum=1e4)
    sc.pp.log1p(inp)
    annotate_kwargs: dict[str, Any] = {"model": model_name, "majority_voting": True}
    if groupby:
        inp.obs[groupby] = adata.obs[groupby].values  # consensus over the user's clusters
        annotate_kwargs["over_clustering"] = groupby

    pred = celltypist.annotate(inp, **annotate_kwargs)
    labels = pred.predicted_labels
    col = "majority_voting" if "majority_voting" in labels.columns else "predicted_labels"
    adata.obs[target] = labels[col].astype(str).astype("category").values
    if "predicted_labels" in labels.columns and col != "predicted_labels":
        adata.obs[f"{target}_percell"] = (
            labels["predicted_labels"].astype(str).astype("category").values
        )
    if getattr(pred, "probability_matrix", None) is not None:
        adata.obs[f"{target}_confidence"] = np.asarray(
            pred.probability_matrix.max(axis=1), dtype="float32"
        )
    logger.info(
        "CellTypist: annotated %d cells -> obs['%s'] (model=%s, groupby=%s, %d types)",
        adata.n_obs, target, model_name, groupby or "celltypist-internal",
        int(adata.obs[target].nunique()),
    )


def _top_markers_per_group(adata: ad.AnnData, groupby: str, n: int = 15) -> dict[str, list[str]]:
    """Top-n marker genes per cluster, reusing precomputed rank_genes_groups when present."""
    key = f"rank_genes_groups__{groupby}"
    if key in adata.uns:
        rgg = adata.uns[key]
    elif (
        "rank_genes_groups" in adata.uns
        and adata.uns["rank_genes_groups"].get("params", {}).get("groupby") == groupby
    ):
        rgg = adata.uns["rank_genes_groups"]
    else:
        if adata.obs[groupby].dtype.name != "category":
            adata.obs[groupby] = adata.obs[groupby].astype(str).astype("category")
        sc.tl.rank_genes_groups(adata, groupby=groupby, method="wilcoxon", n_genes=n)
        rgg = adata.uns["rank_genes_groups"]
    names = rgg["names"]
    groups = list(names.dtype.names) if hasattr(names.dtype, "names") else []
    return {g: [str(names[g][i]) for i in range(min(n, len(names[g])))] for g in groups}


def _parse_cluster_labels(raw: str, clusters: list[str]) -> dict[str, str]:
    """Parse an LLM reply ('cluster_id: cell type' per line) into {cluster: label}."""
    cluster_set = {str(c) for c in clusters}
    out: dict[str, str] = {}
    for line in raw.splitlines():
        line = line.strip().lstrip("-*0123456789. ").strip()
        if ":" not in line:
            continue
        cid, label = (p.strip() for p in line.split(":", 1))
        if cid in cluster_set:
            out[cid] = label
        else:
            for c in cluster_set:
                if cid.endswith(c):
                    out[c] = label
                    break
    if not out:  # fall back to positional zip if the model ignored the id format
        labels = [ln.strip().split(":", 1)[-1].strip() for ln in raw.splitlines() if ln.strip()]
        out = {str(c): lab for c, lab in zip(clusters, labels)}
    return out


def _annotate_llm(adata: ad.AnnData, params: PipelineParams, groupby: str, target: str) -> None:
    """Tissue-agnostic annotation: name each cluster's cell type from its top markers via an LLM.

    No reference model to choose — works for any tissue (GPTCelltype-style). Best treated as a
    reviewable first pass; pair with CellTypist when a matching reference model exists.
    """
    if not groupby:
        raise ValueError("LLM-from-markers annotation needs a clustering; run clustering first.")
    marker_map = _top_markers_per_group(adata, groupby, n=15)
    if not marker_map:
        raise ValueError(f"No marker genes available for '{groupby}'.")

    from scview.config import get_settings

    settings = get_settings()
    api_key = settings.DEEPINFRA_API_KEY
    if not api_key:
        raise ValueError("LLM annotation requires DEEPINFRA_API_KEY to be set.")
    model = params.annotation_llm_model or settings.RAG_CHAT_MODEL

    from openai import OpenAI

    tissue = f"{params.annotation_tissue.strip()} " if params.annotation_tissue.strip() else ""
    listing = "\n".join(f"{g}: {', '.join(genes)}" for g, genes in marker_map.items())
    prompt = (
        "You are an expert in single-cell RNA-seq. For each cluster below, identify the most likely "
        f"cell type of these {tissue}cells from its top marker genes. Output exactly one line per "
        "cluster as 'cluster_id: cell type', using a concise standard cell-type name (a mixture is "
        "allowed). Do not add any other commentary.\n\nClusters and marker genes:\n" + listing
    )
    client = OpenAI(api_key=api_key, base_url="https://api.deepinfra.com/v1/openai")
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=800,
    )
    raw = resp.choices[0].message.content or ""
    labels = _parse_cluster_labels(raw, list(marker_map.keys()))

    mapping = {str(g): labels.get(str(g), "Unknown") for g in marker_map}
    adata.obs[target] = (
        adata.obs[groupby].astype(str).map(lambda g: mapping.get(g, "Unknown")).astype("category")
    )
    adata.uns[f"{target}_llm_mapping"] = mapping
    logger.info(
        "LLM annotation: %d clusters -> obs['%s'] (model=%s, groupby=%s)",
        len(mapping), target, model, groupby,
    )


def _run_cell_type_annotation(adata: ad.AnnData, params: PipelineParams) -> None:
    """Annotate cell types per cluster, writing a reviewable obs[target] column.

    Methods: 'celltypist' (reference-based, tissue-specific model) and 'llm'
    (LLM-from-markers, tissue-agnostic). 'marker_score' is planned — see
    docs/CELLTYPE_ANNOTATION.md.
    """
    method = (params.annotation_method or "llm").lower()
    target = params.annotation_target or "cell_type"
    groupby = _resolve_annotation_groupby(adata, params)

    # Default method is the any-tissue LLM; degrade to CellTypist when no LLM key is set.
    if method == "llm":
        from scview.config import get_settings

        if not get_settings().DEEPINFRA_API_KEY:
            logger.warning(
                "LLM annotation requested but DEEPINFRA_API_KEY is unset; "
                "falling back to CellTypist (%s).", params.celltypist_model,
            )
            method = "celltypist"

    if method == "celltypist":
        _annotate_celltypist(adata, params, groupby, target)
    elif method == "llm":
        _annotate_llm(adata, params, groupby, target)
    elif method == "marker_score":
        raise ValueError(
            "Annotation method 'marker_score' is not implemented yet; use 'celltypist' or 'llm'."
        )
    else:
        raise ValueError(
            f"Unknown annotation method '{method}' (expected celltypist | llm | marker_score)."
        )


# Map step names to their runner functions
_STEP_RUNNERS: dict[str, Any] = {
    "reset_to_counts": _run_reset_to_counts,
    "qc_metrics": _run_qc_metrics,
    "doublet_detection": _run_doublet_detection,
    "filtering": _run_filtering,
    "normalization": _run_normalization,
    "log_transform": _run_log_transform,
    "highly_variable_genes": _run_hvg,
    "scaling": _run_scaling,
    "pca": _run_pca,
    "batch_correction": _run_batch_correction,
    "neighbors": _run_neighbors,
    "clustering": _run_clustering,
    "embeddings": _run_embeddings,
    "marker_genes": _run_marker_genes,
    "enrichment": _run_enrichment,
    "cell_cycle": _run_cell_cycle,
    "cell_type_annotation": _run_cell_type_annotation,
}

# Steps that return a new AnnData object (instead of modifying in-place)
_STEPS_RETURNING_ADATA = {"filtering", "reset_to_counts"}


# step -> (human tool label, PipelineParams fields worth recording)
_STEP_PROVENANCE: dict[str, tuple[str, tuple[str, ...]]] = {
    "reset_to_counts": ("scview.reset_to_counts", ()),
    "qc_metrics": ("scanpy.pp.calculate_qc_metrics", ()),
    "doublet_detection": ("scanpy.pp.scrublet", ("doublet_method", "expected_doublet_rate")),
    "filtering": ("scanpy.pp.filter_cells/genes",
                  ("min_genes", "min_cells", "max_pct_mt", "drop_doublets")),
    "normalization": ("scanpy.pp.normalize_total", ("target_sum",)),
    "log_transform": ("scanpy.pp.log1p", ()),
    "highly_variable_genes": ("scanpy.pp.highly_variable_genes", ("n_top_genes", "hvg_flavor")),
    "scaling": ("scanpy.pp.scale", ()),
    "pca": ("scanpy.pp.pca", ("n_comps",)),
    "batch_correction": ("scview.batch_correction", ("batch_key", "batch_method")),
    "neighbors": ("scanpy.pp.neighbors", ("n_neighbors", "n_comps")),
    "clustering": ("scanpy.tl.leiden/louvain", ("clustering_method", "clustering_resolution")),
    "embeddings": ("scanpy.tl.umap", ("umap_min_dist",)),
    "marker_genes": ("scanpy.tl.rank_genes_groups", ("marker_columns",)),
    "enrichment": ("scview.enrichment",
                   ("enrichment_columns", "enrichment_n_genes", "enrichment_collections")),
    "cell_cycle": ("scanpy.tl.score_genes_cell_cycle", ()),
    "cell_type_annotation": ("scview.cell_type_annotation",
                             ("annotation_method", "celltypist_model", "annotation_llm_model",
                              "annotation_tissue", "annotation_groupby", "annotation_target")),
}


def _record_pipeline_step(adata: ad.AnnData, step_name: str, params: PipelineParams) -> None:
    """Append a provenance entry for a completed step. Never raises."""
    tool, keys = _STEP_PROVENANCE.get(step_name, (f"scview.{step_name}", ()))
    try:
        provenance.record_step(
            adata, step=step_name, tool=tool,
            params={k: getattr(params, k) for k in keys},
        )
    except Exception as e:  # provenance must never break the pipeline
        logger.warning("Could not record provenance for step '%s': %s", step_name, e)


def _finalize_current(adata: ad.AnnData, params: PipelineParams, steps_run: list[str]) -> None:
    """Update the denormalised ``current`` state summary after a pipeline run."""
    try:
        ran = set(steps_run)
        current: dict[str, Any] = {}
        if "normalization" in ran:
            current["normalized"] = True
        if "log_transform" in ran:
            current["log1p"] = True
        if "scaling" in ran:
            current["scaled"] = True
        if "pca" in ran or "X_pca" in adata.obsm:
            current["pca"] = True
        embs = sorted(k for k in adata.obsm.keys() if k.startswith("X_"))
        if embs:
            current["embeddings"] = embs
        active = adata.uns.get("scview_active_clustering")
        if active:
            current["clustering"] = {
                "column": active,
                "method": params.clustering_method,
                "resolution": params.clustering_resolution,
            }
        markers_for = sorted(
            k.split("__", 1)[1]
            for k in adata.uns
            if isinstance(k, str) and k.startswith("rank_genes_groups__")
        )
        if markers_for:
            current["markers_for"] = markers_for
        enr_for = sorted(
            {
                k.split("__")[1]
                for k in adata.uns
                if isinstance(k, str) and k.startswith("enrichment__") and len(k.split("__")) >= 2
            }
        )
        if enr_for:
            current["enrichment_for"] = enr_for
        anchors = []
        if "counts" in adata.layers:
            anchors.append("counts")
        if adata.raw is not None:
            anchors.append("lognorm")  # pre-scaling state stored in adata.raw
        if anchors:
            current["anchors"] = anchors
        if current:
            provenance.set_current(adata, **current)
    except Exception as e:
        logger.warning("Could not finalise provenance state: %s", e)


def run_pipeline(
    adata: ad.AnnData,
    steps: list[str],
    params: PipelineParams | None = None,
    output_path: str | None = None,
) -> tuple[ad.AnnData, PipelineResult]:
    """Run a subset of preprocessing steps on the AnnData object.

    Parameters
    ----------
    adata
        The AnnData object to process (modified in-place where possible).
    steps
        List of step keys to execute (see ALL_STEPS).
    params
        Pipeline parameters. Defaults are used if None.
    output_path
        If provided, save the processed AnnData to this path.

    Returns
    -------
    Tuple of (processed AnnData, PipelineResult).
    """
    if params is None:
        params = PipelineParams()

    # Fix common pandas conflict: obs.index.name clashes with a column name.
    # This happens when e.g. index is named 'cell_id' and a 'cell_id' column
    # also exists with different values. Rename the index to avoid the error.
    if adata.obs.index.name is not None and adata.obs.index.name in adata.obs.columns:
        logger.warning(
            "obs.index.name '%s' conflicts with a column of the same name; "
            "renaming index to avoid pandas error.",
            adata.obs.index.name,
        )
        adata.obs.index.name = None

    result = PipelineResult()
    t0 = time.time()

    # Ensure steps are in canonical order
    ordered_steps = [s for s in ALL_STEPS if s in steps]
    skipped_requested = [s for s in steps if s not in ALL_STEPS]
    if skipped_requested:
        logger.warning("Unknown steps ignored: %s", skipped_requested)

    for step_name in ordered_steps:
        runner = _STEP_RUNNERS.get(step_name)
        if runner is None:
            result.steps_skipped.append(step_name)
            continue

        try:
            logger.info("Running step: %s", step_name)
            ret = runner(adata, params)
            # Some steps return a new AnnData object
            if step_name in _STEPS_RETURNING_ADATA and isinstance(ret, ad.AnnData):
                provenance.carry(adata, ret)  # keep history across the new object
                adata = ret
            result.steps_run.append(step_name)
            _record_pipeline_step(adata, step_name, params)
        except Exception as e:
            logger.error("Step '%s' failed: %s", step_name, e)
            result.errors[step_name] = str(e)

    _finalize_current(adata, params, result.steps_run)
    result.elapsed_seconds = round(time.time() - t0, 2)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        adata.write_h5ad(output_path, compression="gzip")
        result.output_path = output_path
        logger.info("Saved processed dataset to %s", output_path)

    return adata, result


def run_pipeline_streamed(
    adata: ad.AnnData,
    steps: list[str],
    params: PipelineParams | None = None,
    output_path: str | None = None,
) -> Generator[tuple[str, dict[str, Any]], None, tuple[ad.AnnData, PipelineResult]]:
    """Run preprocessing steps, yielding progress events after each step.

    Yields
    ------
    Tuples of (event_type, event_data) where event_type is one of:
      "step_start", "step_done", "step_error", "substep", "complete".
    """
    if params is None:
        params = PipelineParams()

    if adata.obs.index.name is not None and adata.obs.index.name in adata.obs.columns:
        adata.obs.index.name = None

    result = PipelineResult()
    t0 = time.time()

    ordered_steps = [s for s in ALL_STEPS if s in steps]
    total = len(ordered_steps)

    # Runners that accept a substep progress callback (3rd arg)
    _SUBSTEP_RUNNERS = {"marker_genes", "enrichment"}
    _SENTINEL = object()  # marks thread completion

    for idx, step_name in enumerate(ordered_steps):
        runner = _STEP_RUNNERS.get(step_name)
        if runner is None:
            result.steps_skipped.append(step_name)
            continue

        yield ("step_start", {"step": step_name, "index": idx, "total": total})
        step_t0 = time.time()

        try:
            if step_name in _SUBSTEP_RUNNERS:
                # Run in a thread so substep events can stream in real-time
                q: queue.Queue = queue.Queue()
                exc_holder: list[Exception] = []
                ret_holder: list[Any] = [None]

                def _substep_cb(msg: str, current: int, sub_total: int) -> None:
                    q.put({"message": msg, "current": current, "total": sub_total})

                def _run_in_thread(r=runner, a=adata, p=params, cb=_substep_cb):
                    try:
                        ret_holder[0] = r(a, p, cb)
                    except Exception as e:
                        exc_holder.append(e)
                    finally:
                        q.put(_SENTINEL)

                t = threading.Thread(target=_run_in_thread, daemon=True)
                t.start()

                # Drain substep events from the queue while thread runs
                while True:
                    item = q.get()
                    if item is _SENTINEL:
                        break
                    yield ("substep", {"step": step_name, **item})

                t.join()
                if exc_holder:
                    raise exc_holder[0]
                ret = ret_holder[0]
            else:
                ret = runner(adata, params)

            if step_name in _STEPS_RETURNING_ADATA and isinstance(ret, ad.AnnData):
                provenance.carry(adata, ret)  # keep history across the new object
                adata = ret
            result.steps_run.append(step_name)
            _record_pipeline_step(adata, step_name, params)
            elapsed = round(time.time() - step_t0, 2)
            yield ("step_done", {"step": step_name, "index": idx, "elapsed": elapsed})
        except Exception as e:
            logger.error("Step '%s' failed: %s", step_name, e)
            result.errors[step_name] = str(e)
            yield ("step_error", {"step": step_name, "index": idx, "error": str(e)})

    _finalize_current(adata, params, result.steps_run)
    result.elapsed_seconds = round(time.time() - t0, 2)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        adata.write_h5ad(output_path, compression="gzip")
        result.output_path = output_path

    yield ("complete", result.to_dict())

    return adata, result
