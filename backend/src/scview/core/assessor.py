"""Preprocessing state detector for AnnData objects.

Inspects an AnnData object and determines which preprocessing steps
have already been applied, returning a structured report with confidence levels.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from pydantic import BaseModel
from scipy import sparse

from scview.core import provenance

logger = logging.getLogger(__name__)

# scView provenance step name -> PreprocessingState field.
_PROV_STEP_TO_FIELD = {
    "qc_metrics": "qc_metrics",
    "filtering": "filtering",
    "normalization": "normalization",
    "log_transform": "log_transform",
    "highly_variable_genes": "highly_variable_genes",
    "scaling": "scaling",
    "pca": "pca",
    "batch_correction": "batch_correction",
    "neighbors": "neighbors",
    "clustering": "clustering",
    "embeddings": "embeddings",
    "marker_genes": "marker_genes",
    "enrichment": "enrichment",
    "cell_cycle": "cell_cycle",
}


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class StepStatus(BaseModel):
    """Status of a single preprocessing step."""

    done: bool
    confidence: str  # "high", "medium", "low"
    details: str  # human-readable description of what was detected


class PreprocessingState(BaseModel):
    """Full preprocessing state of an AnnData object."""

    qc_metrics: StepStatus
    filtering: StepStatus
    normalization: StepStatus
    log_transform: StepStatus
    highly_variable_genes: StepStatus
    scaling: StepStatus
    pca: StepStatus
    batch_correction: StepStatus
    neighbors: StepStatus
    clustering: StepStatus
    embeddings: StepStatus
    marker_genes: StepStatus
    enrichment: StepStatus
    cell_cycle: StepStatus


# ---------------------------------------------------------------------------
# Individual step detectors
# ---------------------------------------------------------------------------


def _check_qc_metrics(adata: Any) -> StepStatus:
    """Check if QC metrics (n_genes_by_counts, total_counts, etc.) are present."""
    try:
        obs_cols = set(adata.obs.columns)
        qc_columns = {"n_genes_by_counts", "total_counts", "n_genes"}
        found = qc_columns & obs_cols

        # Also check for mitochondrial percentage columns
        mt_cols = [c for c in obs_cols if "pct_counts_mt" in c.lower() or "percent_mt" in c.lower()]

        if found and mt_cols:
            return StepStatus(
                done=True,
                confidence="high",
                details=f"Found QC columns: {', '.join(sorted(found | set(mt_cols)))}",
            )
        elif found:
            return StepStatus(
                done=True,
                confidence="high",
                details=f"Found QC columns: {', '.join(sorted(found))} (no MT% column detected)",
            )
        elif mt_cols:
            return StepStatus(
                done=True,
                confidence="medium",
                details=f"Found MT% columns only: {', '.join(mt_cols)}",
            )
        else:
            return StepStatus(
                done=False,
                confidence="high",
                details="No standard QC metric columns found in obs",
            )
    except Exception as exc:
        logger.warning("Error checking QC metrics: %s", exc)
        return StepStatus(done=False, confidence="low", details=f"Check failed: {exc}")


def _check_filtering(adata: Any) -> StepStatus:
    """Heuristic: if QC columns exist and have no very low values, filtering was likely applied."""
    try:
        obs_cols = set(adata.obs.columns)

        if "n_genes_by_counts" in obs_cols:
            min_genes = int(adata.obs["n_genes_by_counts"].min())
            if min_genes >= 200:
                return StepStatus(
                    done=True,
                    confidence="medium",
                    details=f"Min genes/cell = {min_genes} (>= 200 suggests filtering was applied)",
                )
            elif min_genes > 0:
                return StepStatus(
                    done=False,
                    confidence="medium",
                    details=f"Min genes/cell = {min_genes} (low value suggests no strict filtering)",
                )
            else:
                return StepStatus(
                    done=False,
                    confidence="medium",
                    details="Cells with 0 genes detected; filtering likely not applied",
                )
        elif "n_genes" in obs_cols:
            min_genes = int(adata.obs["n_genes"].min())
            if min_genes >= 200:
                return StepStatus(
                    done=True,
                    confidence="medium",
                    details=f"Min genes/cell (n_genes) = {min_genes} (>= 200 suggests filtering)",
                )
            else:
                return StepStatus(
                    done=False,
                    confidence="low",
                    details=f"Min genes/cell (n_genes) = {min_genes}; unclear if filtered",
                )
        else:
            return StepStatus(
                done=False,
                confidence="low",
                details="No gene-count columns found; cannot determine filtering status",
            )
    except Exception as exc:
        logger.warning("Error checking filtering: %s", exc)
        return StepStatus(done=False, confidence="low", details=f"Check failed: {exc}")


def _check_normalization(adata: Any) -> StepStatus:
    """Check dtype of X (float -> likely normalized), presence of raw or counts layer."""
    try:
        if _is_raw_counts(adata):
            return StepStatus(
                done=False,
                confidence="high",
                details="Raw integer counts — not normalized yet",
            )
        X = adata.X
        is_float = False

        if sparse.issparse(X):
            is_float = np.issubdtype(X.dtype, np.floating)
        else:
            is_float = np.issubdtype(np.asarray(X).dtype, np.floating)

        has_raw = adata.raw is not None
        has_counts_layer = "counts" in adata.layers if hasattr(adata, "layers") else False

        evidence: list[str] = []
        if is_float:
            evidence.append("X dtype is float")
        if has_raw:
            evidence.append("adata.raw is set (common post-normalization pattern)")
        if has_counts_layer:
            evidence.append("'counts' layer exists (raw counts preserved)")

        if is_float and (has_raw or has_counts_layer):
            return StepStatus(
                done=True,
                confidence="high",
                details="; ".join(evidence),
            )
        elif is_float:
            return StepStatus(
                done=True,
                confidence="medium",
                details="; ".join(evidence) + " (but no raw/counts backup found)",
            )
        elif has_raw or has_counts_layer:
            return StepStatus(
                done=True,
                confidence="medium",
                details="; ".join(evidence) + " (X dtype is not float, unusual)",
            )
        else:
            return StepStatus(
                done=False,
                confidence="medium",
                details="X dtype is not float, no raw or counts layer found",
            )
    except Exception as exc:
        logger.warning("Error checking normalization: %s", exc)
        return StepStatus(done=False, confidence="low", details=f"Check failed: {exc}")


def _dense_sample(X: Any, n_rows: int, n_cols: int | None = None) -> np.ndarray:
    """Extract a dense float64 ndarray sample from any anndata X type.

    Handles sparse matrices, np.matrix, h5py datasets, and anndata
    internal array wrappers that ``np.asarray`` may fail to convert.
    """
    sliced = X[:n_rows] if n_cols is None else X[:n_rows, :n_cols]
    if sparse.issparse(sliced):
        arr = sliced.toarray()
    elif hasattr(sliced, "toarray"):
        arr = sliced.toarray()
    elif hasattr(sliced, "todense"):
        arr = np.asarray(sliced.todense())
    else:
        arr = sliced
    # dtype=np.float64 + copy=True forces a fresh plain ndarray regardless
    # of the input wrapper type (avoids 0-d object array from np.array).
    return np.array(arr, dtype=np.float64, copy=True)


def _is_raw_counts(adata: Any) -> bool:
    """True if X looks like raw counts — non-negative and integer-valued, even
    when stored as float (10x matrices are float-typed counts). This is the
    definitive tell that the data has NOT been normalised / logged / scaled."""
    try:
        n_cols = min(2000, adata.n_vars)
        sample = _dense_sample(adata.X, min(200, adata.n_obs), n_cols).ravel()
        finite = sample[np.isfinite(sample)]
        if finite.size == 0:
            return False
        return bool(np.all(finite >= 0) and np.allclose(finite, np.round(finite)))
    except Exception:
        return False


def _check_log_transform(adata: Any) -> StepStatus:
    """Check if data appears log-transformed by examining value range."""
    try:
        if _is_raw_counts(adata):
            return StepStatus(
                done=False,
                confidence="high",
                details="Raw integer counts — not log-transformed yet",
            )
        has_raw = adata.raw is not None

        # Prefer raw.X (pre-scaling data) for a cleaner signal; fall back to X
        if has_raw:
            raw_X = adata.raw.X
            n_sample = min(1000, raw_X.shape[0])
            sample = _dense_sample(raw_X, n_sample).ravel()
            source = "raw.X"
        else:
            n_sample = min(1000, adata.n_obs)
            sample = _dense_sample(adata.X, n_sample).ravel()
            source = "X"

        max_val = float(np.nanmax(sample))

        if max_val < 0:
            return StepStatus(
                done=True,
                confidence="low",
                details=f"{source} max = {max_val:.2f} (negative values; data may be scaled after log)",
            )
        elif max_val <= 20:
            confidence = "high" if has_raw else "medium"
            extra = "; raw data preserved" if has_raw else "; no raw to compare"
            return StepStatus(
                done=True,
                confidence=confidence,
                details=f"{source} max = {max_val:.2f} (<= 20, consistent with log1p){extra}",
            )
        else:
            return StepStatus(
                done=False,
                confidence="medium",
                details=f"{source} max = {max_val:.2f} (> 20, likely not log-transformed)",
            )
    except Exception as exc:
        logger.warning("Error checking log transform: %s", exc)
        return StepStatus(done=False, confidence="low", details=f"Check failed: {exc}")


def _check_highly_variable_genes(adata: Any) -> StepStatus:
    """Check for 'highly_variable' column in var."""
    try:
        if "highly_variable" in adata.var.columns:
            n_hvg = int(adata.var["highly_variable"].sum())
            n_total = adata.n_vars
            return StepStatus(
                done=True,
                confidence="high",
                details=f"Found 'highly_variable' in var: {n_hvg}/{n_total} genes selected",
            )
        else:
            return StepStatus(
                done=False,
                confidence="high",
                details="No 'highly_variable' column in var",
            )
    except Exception as exc:
        logger.warning("Error checking HVGs: %s", exc)
        return StepStatus(done=False, confidence="low", details=f"Check failed: {exc}")


def _check_scaling(adata: Any) -> StepStatus:
    """Check if X has been z-score scaled (mean ~ 0, std ~ 1 per gene)."""
    try:
        if _is_raw_counts(adata):
            return StepStatus(
                done=False,
                confidence="high",
                details="Raw counts — not scaled (z-scoring would produce negative values)",
            )

        n_genes_sample = min(100, adata.n_vars)
        n_cells_sample = min(2000, adata.n_obs)

        # _dense_sample guarantees a plain float64 ndarray
        sample = _dense_sample(adata.X, n_cells_sample, n_genes_sample)

        # Z-score scaling produces negative values centred near 0. Non-negative
        # data (counts / normalised / log1p) is definitely not scaled.
        if float(np.nanmin(sample)) >= 0:
            return StepStatus(
                done=False,
                confidence="high",
                details="All values are non-negative — not z-score scaled",
            )

        # Compute per-gene statistics — ravel() guarantees 1D
        gene_means = np.ravel(np.nanmean(sample, axis=0))
        gene_stds = np.ravel(np.nanstd(sample, axis=0))

        # Filter out zero-variance genes for a fair check
        nonzero_mask = gene_stds > 1e-6
        n_nonzero = int(nonzero_mask.sum())
        if n_nonzero < 5:
            return StepStatus(
                done=False,
                confidence="low",
                details="Too few non-zero-variance genes to assess scaling",
            )

        mean_of_means = float(np.mean(gene_means[nonzero_mask]))
        mean_of_stds = float(np.mean(gene_stds[nonzero_mask]))

        if abs(mean_of_means) < 0.5 and 0.5 < mean_of_stds < 1.5:
            return StepStatus(
                done=True,
                confidence="high",
                details=(
                    f"Gene means avg = {mean_of_means:.3f}, stds avg = {mean_of_stds:.3f} "
                    f"(consistent with z-score scaling)"
                ),
            )
        elif abs(mean_of_means) < 1.0 and 0.3 < mean_of_stds < 2.0:
            return StepStatus(
                done=True,
                confidence="medium",
                details=(
                    f"Gene means avg = {mean_of_means:.3f}, stds avg = {mean_of_stds:.3f} "
                    f"(possibly scaled, but not perfectly centered)"
                ),
            )
        else:
            return StepStatus(
                done=False,
                confidence="medium",
                details=(
                    f"Gene means avg = {mean_of_means:.3f}, stds avg = {mean_of_stds:.3f} "
                    f"(does not appear z-score scaled)"
                ),
            )
    except Exception as exc:
        logger.warning("Error checking scaling: %s", exc)
        return StepStatus(done=False, confidence="low", details=f"Check failed: {exc}")


def _check_pca(adata: Any) -> StepStatus:
    """Check for X_pca in obsm."""
    try:
        if "X_pca" in adata.obsm:
            n_pcs = adata.obsm["X_pca"].shape[1]
            has_variance_ratio = "pca" in adata.uns and "variance_ratio" in adata.uns.get("pca", {})
            extra = ""
            if has_variance_ratio:
                extra = "; variance ratios stored in uns"
            return StepStatus(
                done=True,
                confidence="high",
                details=f"Found X_pca with {n_pcs} components{extra}",
            )
        else:
            return StepStatus(
                done=False,
                confidence="high",
                details="No 'X_pca' in obsm",
            )
    except Exception as exc:
        logger.warning("Error checking PCA: %s", exc)
        return StepStatus(done=False, confidence="low", details=f"Check failed: {exc}")


def _check_batch_correction(adata: Any) -> StepStatus:
    """Check for Harmony-corrected PCA embedding (X_pca_harmony) in obsm."""
    try:
        if "X_pca_harmony" in adata.obsm:
            n_dims = adata.obsm["X_pca_harmony"].shape[1]
            return StepStatus(
                done=True,
                confidence="high",
                details=f"Found X_pca_harmony with {n_dims} dimensions",
            )
        else:
            return StepStatus(
                done=False,
                confidence="high",
                details="No 'X_pca_harmony' in obsm (batch correction not applied)",
            )
    except Exception as exc:
        logger.warning("Error checking batch correction: %s", exc)
        return StepStatus(done=False, confidence="low", details=f"Check failed: {exc}")


def _check_neighbors(adata: Any) -> StepStatus:
    """Check for connectivity/distance graphs in obsp."""
    try:
        has_connectivities = "connectivities" in adata.obsp
        has_distances = "distances" in adata.obsp
        has_neighbors_uns = "neighbors" in adata.uns

        if has_connectivities or has_distances:
            parts = []
            if has_connectivities:
                parts.append("connectivities")
            if has_distances:
                parts.append("distances")
            extra = ""
            if has_neighbors_uns:
                params = adata.uns.get("neighbors", {}).get("params", {})
                if params:
                    extra = f"; params: {params}"
            return StepStatus(
                done=True,
                confidence="high",
                details=f"Found {' and '.join(parts)} in obsp{extra}",
            )
        else:
            return StepStatus(
                done=False,
                confidence="high",
                details="No 'connectivities' or 'distances' in obsp",
            )
    except Exception as exc:
        logger.warning("Error checking neighbors: %s", exc)
        return StepStatus(done=False, confidence="low", details=f"Check failed: {exc}")


def _check_clustering(adata: Any) -> StepStatus:
    """Check for leiden, louvain, seurat_clusters, or cluster columns in obs."""
    try:
        obs_cols = set(adata.obs.columns)
        cluster_candidates = ["leiden", "louvain", "seurat_clusters", "cluster", "clusters"]
        found = [c for c in cluster_candidates if c in obs_cols]

        if found:
            details_parts = []
            for col in found:
                n_clusters = adata.obs[col].nunique()
                details_parts.append(f"'{col}' ({n_clusters} clusters)")
            return StepStatus(
                done=True,
                confidence="high",
                details=f"Found clustering columns: {', '.join(details_parts)}",
            )
        else:
            # Also check for any column that looks like cluster assignments
            possible = [
                c for c in obs_cols
                if "clust" in c.lower() or "community" in c.lower()
            ]
            if possible:
                return StepStatus(
                    done=True,
                    confidence="medium",
                    details=f"Possible clustering columns: {', '.join(possible)}",
                )
            return StepStatus(
                done=False,
                confidence="high",
                details="No standard clustering columns found in obs",
            )
    except Exception as exc:
        logger.warning("Error checking clustering: %s", exc)
        return StepStatus(done=False, confidence="low", details=f"Check failed: {exc}")


def _check_embeddings(adata: Any) -> StepStatus:
    """Check for X_umap, X_tsne, or other 2D embeddings in obsm."""
    try:
        embedding_keys = []
        for key in adata.obsm_keys():
            if key in ("X_umap", "X_tsne"):
                embedding_keys.append(key)
            elif key.startswith("X_") and key != "X_pca":
                # Other possible embeddings (X_draw_graph_fa, X_diffmap, etc.)
                arr = adata.obsm[key]
                if hasattr(arr, "shape") and arr.ndim == 2 and arr.shape[1] in (2, 3):
                    embedding_keys.append(key)

        if embedding_keys:
            return StepStatus(
                done=True,
                confidence="high",
                details=f"Found embeddings: {', '.join(sorted(embedding_keys))}",
            )
        else:
            return StepStatus(
                done=False,
                confidence="high",
                details="No UMAP, t-SNE, or other 2D embeddings found in obsm",
            )
    except Exception as exc:
        logger.warning("Error checking embeddings: %s", exc)
        return StepStatus(done=False, confidence="low", details=f"Check failed: {exc}")


def _check_marker_genes(adata: Any) -> StepStatus:
    """Check for rank_genes_groups in uns (single or multi-column)."""
    try:
        # Check for per-column marker results
        multi_keys = [k for k in adata.uns if k.startswith("rank_genes_groups__")]
        columns_done = [k.split("__", 1)[1] for k in multi_keys]
        has_default = "rank_genes_groups" in adata.uns

        if multi_keys:
            details_parts = []
            for col in columns_done:
                rgg = adata.uns[f"rank_genes_groups__{col}"]
                method = rgg.get("params", {}).get("method", "unknown")
                details_parts.append(f"'{col}' ({method})")
            return StepStatus(
                done=True,
                confidence="high",
                details=f"Markers computed for {len(multi_keys)} column(s): {', '.join(details_parts)}",
            )
        elif has_default:
            rgg = adata.uns["rank_genes_groups"]
            method = rgg.get("params", {}).get("method", "unknown")
            groupby = rgg.get("params", {}).get("groupby", "unknown")
            return StepStatus(
                done=True,
                confidence="high",
                details=f"Found rank_genes_groups (method: {method}, groupby: {groupby})",
            )
        else:
            return StepStatus(
                done=False,
                confidence="high",
                details="No 'rank_genes_groups' in uns",
            )
    except Exception as exc:
        logger.warning("Error checking marker genes: %s", exc)
        return StepStatus(done=False, confidence="low", details=f"Check failed: {exc}")


def _check_enrichment(adata: Any) -> StepStatus:
    """Check for pre-computed enrichment results in uns."""
    try:
        enrichment_keys = [k for k in adata.uns if k.startswith("enrichment__")]
        if enrichment_keys:
            columns = set()
            for k in enrichment_keys:
                parts = k.split("__")
                if len(parts) >= 2:
                    columns.add(parts[1])
            return StepStatus(
                done=True,
                confidence="high",
                details=(
                    f"Enrichment computed for {len(columns)} column(s): "
                    f"{', '.join(sorted(columns))}; {len(enrichment_keys)} total group results"
                ),
            )

        # Also check legacy keys
        legacy = [k for k in ("enrichment_results", "enrich_results",
                               "pathway_enrichment", "gsea_results") if k in adata.uns]
        if legacy:
            return StepStatus(
                done=True,
                confidence="medium",
                details=f"Found legacy enrichment key(s): {', '.join(legacy)}",
            )

        return StepStatus(
            done=False,
            confidence="high",
            details="No enrichment results found in uns",
        )
    except Exception as exc:
        logger.warning("Error checking enrichment: %s", exc)
        return StepStatus(done=False, confidence="low", details=f"Check failed: {exc}")


def _check_cell_cycle(adata: Any) -> StepStatus:
    """Check for S_score and G2M_score in obs."""
    try:
        obs_cols = set(adata.obs.columns)
        has_s = "S_score" in obs_cols
        has_g2m = "G2M_score" in obs_cols
        has_phase = "phase" in obs_cols

        if has_s and has_g2m:
            extra = " with phase assignments" if has_phase else ""
            return StepStatus(
                done=True,
                confidence="high",
                details=f"Found S_score and G2M_score in obs{extra}",
            )
        elif has_s or has_g2m:
            found = "S_score" if has_s else "G2M_score"
            return StepStatus(
                done=True,
                confidence="medium",
                details=f"Found {found} but missing the other score",
            )
        elif has_phase:
            return StepStatus(
                done=True,
                confidence="medium",
                details="Found 'phase' column but no S_score/G2M_score",
            )
        else:
            return StepStatus(
                done=False,
                confidence="high",
                details="No cell cycle scores (S_score, G2M_score) found in obs",
            )
    except Exception as exc:
        logger.warning("Error checking cell cycle: %s", exc)
        return StepStatus(done=False, confidence="low", details=f"Check failed: {exc}")


# ---------------------------------------------------------------------------
# Main assessment function
# ---------------------------------------------------------------------------


def assess_preprocessing(adata: Any) -> PreprocessingState:
    """Inspect an AnnData object and return the full preprocessing state.

    Parameters
    ----------
    adata
        An anndata.AnnData object (or anything with the same interface).

    Returns
    -------
    PreprocessingState
        Structured report of which preprocessing steps have been applied.
    """
    logger.info(
        "Assessing preprocessing state for dataset with %d cells x %d genes",
        adata.n_obs,
        adata.n_vars,
    )

    state = PreprocessingState(
        qc_metrics=_check_qc_metrics(adata),
        filtering=_check_filtering(adata),
        normalization=_check_normalization(adata),
        log_transform=_check_log_transform(adata),
        highly_variable_genes=_check_highly_variable_genes(adata),
        scaling=_check_scaling(adata),
        pca=_check_pca(adata),
        batch_correction=_check_batch_correction(adata),
        neighbors=_check_neighbors(adata),
        clustering=_check_clustering(adata),
        embeddings=_check_embeddings(adata),
        marker_genes=_check_marker_genes(adata),
        enrichment=_check_enrichment(adata),
        cell_cycle=_check_cell_cycle(adata),
    )
    return _apply_recorded_provenance(adata, state)


def _apply_recorded_provenance(adata: Any, state: PreprocessingState) -> PreprocessingState:
    """Overlay what scView actually recorded doing: a step in the provenance
    history is authoritatively 'done' (high confidence), overriding heuristics."""
    try:
        history = provenance.read_provenance(adata).get("history", [])
        last_ts: dict[str, str] = {}
        for h in history:
            field = _PROV_STEP_TO_FIELD.get(h.get("step", ""))
            if field:
                last_ts[field] = h.get("timestamp", "")
        for field, ts in last_ts.items():
            when = ts.split("T")[0] if ts else ""
            setattr(
                state,
                field,
                StepStatus(
                    done=True,
                    confidence="high",
                    details=f"Run by scView{(' on ' + when) if when else ''} (recorded)",
                ),
            )
    except Exception as exc:  # provenance overlay must never break assessment
        logger.warning("Provenance overlay failed: %s", exc)
    return state
