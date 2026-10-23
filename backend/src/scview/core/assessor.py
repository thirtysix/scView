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
from scview.core.cell_cycle import near_zero_phase_calls, phase_caution
from scview.core.stress import SCORE_COLUMN
from scview.core.uns_keys import enrichment_column, markers_column

logger = logging.getLogger(__name__)

# scView provenance step name -> PreprocessingState field.
_PROV_STEP_TO_FIELD = {
    "qc_metrics": "qc_metrics",
    "doublet_detection": "doublet_detection",
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
    "dissociation_stress": "dissociation_stress",
}


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class StepStatus(BaseModel):
    """Status of a single preprocessing step."""

    done: bool
    confidence: str  # "high", "medium", "low"
    details: str  # human-readable description of what was detected
    caution: str | None = None  # a one-line warning about the step's result, if any


class PreprocessingState(BaseModel):
    """Full preprocessing state of an AnnData object."""

    qc_metrics: StepStatus
    doublet_detection: StepStatus
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
    dissociation_stress: StepStatus


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


def _check_doublet_detection(adata: Any) -> StepStatus:
    """Check if doublet detection (Scrublet) has annotated obs."""
    try:
        obs_cols = set(adata.obs.columns)
        dbl_cols = {"doublet_score", "predicted_doublet"} & obs_cols
        if dbl_cols:
            from scview.core import doublet_source

            # Text "True"/"False" after a round trip: astype(bool) would count them all.
            scr = doublet_source.scrublet_calls(adata.obs)
            detail = f"Found doublet columns: {', '.join(sorted(dbl_cols))}"
            if scr is not None:
                detail += f" ({int(scr.sum())} cells flagged)"
            hashing = doublet_source.hashing_calls(adata.obs)
            if hashing is not None:
                detail += (f"; cell hashing calls {int(hashing.sum())} doublets, the doublet views' "
                           "primary call (Scrublet is the second opinion)")
            return StepStatus(done=True, confidence="high", details=detail)
        return StepStatus(
            done=False, confidence="high", details="No doublet annotation found in obs"
        )
    except Exception as exc:
        logger.warning("Error checking doublet detection: %s", exc)
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
    """Normalization check, plus an advisory on the library-size scale (C011)."""
    status = _check_normalization_basic(adata)
    try:
        note = _scale_note(adata)
    except Exception as exc:  # advisory only; never break the check
        logger.warning("Error checking normalization scale: %s", exc)
        note = ""
    if note:
        status = status.model_copy(update={"details": f"{status.details}. {note}"})
    return status


_SCALE_SAMPLE_CELLS = 2000
_SCALE_TARGETS = ((1e4, "1e4, the scanpy/Seurat default"), (1e6, "1e6 (CPM)"))
_BAND_LOG10 = 0.3  # +/- factor ~2 around a target


def _counts_recoverable(adata: Any) -> str:
    """Say where raw counts can be read from, without running any recovery."""
    from scview.core.pipeline import looks_like_counts  # lazy: pipeline is heavy

    try:
        layer = adata.layers["counts"] if "counts" in adata.layers else None
        if layer is not None and looks_like_counts(layer[:200]):
            return "Raw counts are recoverable from layers['counts']"
        if adata.raw is not None and looks_like_counts(adata.raw.X[:200]):
            return "Raw counts are recoverable from .raw"
    except Exception as exc:
        logger.debug("counts recovery probe failed: %s", exc)
    return "No raw counts found in layers['counts'] or .raw"


_SCALE_CHUNK = 250


def _sample_totals(X: Any, n_rows: int) -> tuple[bool, np.ndarray] | None:
    """(logged, per-cell totals of expm1(X) if logged else X) over the first
    ``n_rows`` cells, or None for scaled or unreadable values. Read in chunks of
    rows: the whole sample dense, with its copies, took 1 GB on a 20k-gene dataset,
    and three requests compute it at once when a dataset opens."""
    if sparse.issparse(X):  # the stored values alone: zeros add nothing to either total
        sub = sparse.csr_matrix(X[:n_rows], dtype=np.float64, copy=True)
        d = sub.data
        ok = np.isfinite(d)
        d[~ok] = 0.0
        lo = min(0.0, float(d[ok].min())) if ok.any() else 0.0
        hi = max(0.0, float(d[ok].max())) if ok.any() else 0.0
        if lo < 0:
            return None
        logged = hi <= 20
        if logged:
            np.expm1(d, out=d)
        return logged, np.asarray(sub.sum(axis=1)).ravel()
    lo, hi, any_finite = np.inf, -np.inf, False
    raw_sums, log_sums = [], []
    for start in range(0, n_rows, _SCALE_CHUNK):
        block = _dense_sample(X[start:min(n_rows, start + _SCALE_CHUNK)], _SCALE_CHUNK)
        ok = np.isfinite(block)
        if ok.any():
            any_finite = True
            lo, hi = min(lo, float(block[ok].min())), max(hi, float(block[ok].max()))
        block[~ok] = 0.0
        raw_sums.append(block.sum(axis=1))
        with np.errstate(over="ignore"):
            log_sums.append(np.expm1(block, out=block).sum(axis=1))
    if not any_finite or lo < 0:
        return None
    logged = hi <= 20
    return logged, np.concatenate(log_sums if logged else raw_sums)


def _scale_note(adata: Any) -> str:
    """Median per-cell total on a cell sample, classified into scale bands.

    Bands (not equality) because gene filtering / HVG subsetting makes the sums
    smaller than the normalisation target. Empty string when X is raw counts,
    scaled (negative) or otherwise unreadable.
    """
    if _is_raw_counts(adata):
        return ""
    found = _sample_totals(adata.X, min(_SCALE_SAMPLE_CELLS, adata.n_obs))
    if found is None:
        return ""  # scaled data: totals are meaningless
    logged, totals = found
    median = float(np.median(totals))
    if median <= 0:
        return ""
    n_ref = adata.raw.n_vars if adata.raw is not None else adata.n_vars
    if "counts" in adata.layers:
        n_ref = max(n_ref, adata.layers["counts"].shape[1])
    hvg_subset = adata.n_vars < 0.5 * n_ref
    form = "expm1(X)" if logged else "X"
    msg = f"Median per-cell total of {form} = {median:.3g}"
    near = [
        label for target, label in _SCALE_TARGETS
        if abs(np.log10(median / target)) <= _BAND_LOG10
    ]
    if near and near[0].startswith("1e4"):
        msg += " (~1e4, the scanpy/Seurat default; fine)"
    elif near:
        msg += (
            " (~1e6, CPM-scale: WARNING, not the usual 1e4; counts-per-million "
            "inflates log values and can distort HVG selection and DE effect sizes)"
        )
    else:
        msg += " (non-standard scale"
        msg += (
            f"; X holds {adata.n_vars} of {n_ref} genes, so totals are lower than "
            "the normalisation target, likely normalised before gene subsetting)"
            if hvg_subset and median < 1e6
            else ")"
        )
    if hvg_subset and near:
        msg += f"; X is a gene subset ({adata.n_vars} of {n_ref}), totals may sit below the target"
    return msg + ". " + _counts_recoverable(adata)


def _check_normalization_basic(adata: Any) -> StepStatus:
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

        # A backed matrix (a large dataset's, a file over 2 GB) has a dtype but is no array.
        dtype = getattr(X, "dtype", None)
        if dtype is None or not (sparse.issparse(X) or hasattr(X, "shape")):
            dtype = np.asarray(X).dtype
        is_float = np.issubdtype(dtype, np.floating)

        has_raw = adata.raw is not None
        has_counts_layer = "counts" in adata.layers if hasattr(adata, "layers") else False
        if not has_counts_layer:  # a large dataset keeps its counts in a file of their own
            from scview.core import large_mode

            try:
                has_counts_layer = large_mode.counts_of(adata) is not None
            except large_mode.LargeDatasetError:
                pass

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
        multi_keys = [k for k in adata.uns if markers_column(k) is not None]
        has_default = "rank_genes_groups" in adata.uns

        if multi_keys:
            details_parts = []
            for key in multi_keys:
                rgg = adata.uns[key]
                method = rgg.get("params", {}).get("method", "unknown")
                details_parts.append(f"'{markers_column(key)}' ({method})")
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
        enrichment_keys = [k for k in adata.uns if enrichment_column(k) is not None]
        if enrichment_keys:
            columns = {enrichment_column(k) for k in enrichment_keys}
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
                caution=phase_caution(near_zero_phase_calls(adata.obs)),
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


def _check_dissociation_stress(adata: Any) -> StepStatus:
    """Check for the per-cell dissociation-stress score in obs."""
    if SCORE_COLUMN in adata.obs.columns:
        return StepStatus(done=True, confidence="high", details=f"Found {SCORE_COLUMN} in obs")
    return StepStatus(
        done=False, confidence="high", details="No dissociation-stress score in obs"
    )


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
        doublet_detection=_check_doublet_detection(adata),
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
        dissociation_stress=_check_dissociation_stress(adata),
    )
    return _apply_recorded_provenance(adata, state)


_PEARSON_DETAILS = {
    "highly_variable_genes": "Ranked by Pearson residual variance (analytic Pearson residuals)",
    "scaling": "X holds analytic Pearson residuals of the counts (.raw keeps log values)",
}


def _scaling_details(params: dict, method: str | None) -> str | None:
    """What a recorded scaling step did since it plans only the matrix PCA reads
    (core/scaling.py; ``scaled_genes`` recorded), or None for an older record (it
    wrote z-scores or residuals of every gene into X)."""
    genes = params.get("scaled_genes")
    if genes is None:
        return None
    n = params.get("n_scaled_genes")
    which = (f"the {n:,} variable genes" if isinstance(n, int) else "the variable genes") \
        if genes == "highly_variable" else "every gene"
    made = ("analytic Pearson residuals of the counts" if method == "pearson_residuals"
            else "z-scores")
    return f"PCA reads {made} of {which}; X keeps the log-normalised values"


def recorded_normalization_method(adata: Any) -> str | None:
    """The normalisation method scView last processed this dataset with ("log" or
    "pearson_residuals"), or None when it never ran the HVG or scaling step."""
    try:
        data = provenance.read_provenance(adata)
        method = data.get("current", {}).get("normalization_method")
        if method:
            return str(method)
        for h in reversed(data.get("history", [])):
            if h.get("step") in ("highly_variable_genes", "scaling"):
                return str((h.get("params") or {}).get("normalization_method") or "log")
    except Exception as exc:  # never break the assessment
        logger.warning("Could not read the recorded normalization method: %s", exc)
    return None


def _apply_recorded_provenance(adata: Any, state: PreprocessingState) -> PreprocessingState:
    """Overlay what scView actually recorded doing: a step in the provenance
    history is authoritatively 'done' (high confidence), overriding heuristics.
    A caution read from the data itself is kept."""
    try:
        history = provenance.read_provenance(adata).get("history", [])
        last: dict[str, dict] = {}
        for h in history:
            field = _PROV_STEP_TO_FIELD.get(h.get("step", ""))
            if field:
                last[field] = h
        for field, h in last.items():
            ts = h.get("timestamp", "")
            when = ts.split("T")[0] if ts else ""
            details = f"Run by scView{(' on ' + when) if when else ''} (recorded)"
            # Under Pearson residuals these two steps did something else than their
            # names say: residual-variance HVGs, residuals in X instead of z-scores.
            params = h.get("params") or {}
            method = params.get("normalization_method")
            what = _scaling_details(params, method) if field == "scaling" else None
            if what is None and field in _PEARSON_DETAILS and method == "pearson_residuals":
                what = _PEARSON_DETAILS[field]
            if what:
                details = f"{what}; {details[0].lower()}{details[1:]}"
            setattr(
                state,
                field,
                StepStatus(
                    done=True,
                    confidence="high",
                    details=details,
                    caution=getattr(state, field).caution,
                ),
            )
    except Exception as exc:  # provenance overlay must never break assessment
        logger.warning("Provenance overlay failed: %s", exc)
    return state
