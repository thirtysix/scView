"""Preprocessing pipeline — run standard scanpy steps on an AnnData object.

This module applies standard preprocessing steps (normalization, HVG selection,
PCA, neighbor graph, clustering, UMAP) to an AnnData object in-place,
then saves the result to disk.
"""

from __future__ import annotations

import contextlib
import contextvars
import json
import logging
import os
import queue
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Generator

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

from scview.core import (
    doublets,
    leiden,
    marker_specificity,
    normalization,
    provenance,
    qc_filter,
    scaling,
    stress,
)
from scview.core.run_watchdog import RunCancelled, RunControl, cancelled_reason
from scview.core.anndata_adaptor import per_cell_qc
from scview.core.cell_cycle import near_zero_phase_calls
from scview.core.llm import chat_extra
from scview.core.geneset_sources import (
    ResolvedGeneSets,
    match_query_case,
    resolve_gene_sets,
    tag_records,
)
from scview.core.ora_background import write_meta
from scview.core.species import detect_species, match_genes
from scview.core.uns_keys import (
    BATCH_MIXING_KEY,
    enrichment_column,
    enrichment_key,
    enrichment_prefix,
    markers_column,
    markers_key,
    meta_prefix,
)

logger = logging.getLogger(__name__)

# Callback for sub-step progress: (message, current_index, total_count)
SubstepCallback = Callable[[str, int, int], None]

def _NOOP_CB(_msg: str, _cur: int, _tot: int) -> None:
    """No-op default for runners that don't support callbacks."""


@dataclass
class PipelineParams:
    """Parameters for each preprocessing step."""

    # Filtering (cells: see core/qc_filter.py; 0 turns a threshold off)
    # "mad": each sample's outliers (median +/- k MADs); "fixed": the cut-offs below.
    qc_threshold_mode: str = "mad"
    qc_mad_k: float = 5.0  # counts and genes per cell (log1p), below the median
    qc_mad_k_mt: float = 3.0  # % mito, above only
    qc_mad_upper: bool = False  # also cut counts / genes above median + k MADs (opt-in)
    qc_sample_column: str = ""  # the samples for MAD mode (empty = detected)
    min_genes: int = 200  # fixed mode only, as are max_genes and min_counts
    max_genes: int = 0
    min_counts: int = 0
    min_cells: int = 3
    # Used only when % mito can be measured (MT- genes); a ceiling in MAD mode.
    max_pct_mt: float = 20.0
    # "global": the rule above for every cell; "per_type": each cell type's own bound
    # (median + qc_mad_k_mt MADs) instead, for types that can have one (core/qc_filter.py).
    qc_mito_mode: str = "global"
    qc_mito_type_column: str = ""  # per_type: the cell types (empty = mito_by_type's choice)
    # per_type: never stricter than the global rule (a type's bound = max(own, global)); opt-in
    qc_mito_type_floor: bool = False
    drop_doublets: bool = False  # drop cells flagged by doublet detection (opt-in)
    # Whose doublets: "auto" (cell hashing when run, else Scrublet) | "hashing" |
    # "scrublet" | "either" (core/doublet_source.py); hashing negatives are a separate opt-in.
    qc_doublet_source: str = "auto"
    drop_hash_negatives: bool = False

    # Doublet detection: Scrublet in each capture channel (core/doublets.py)
    doublet_method: str = "scrublet"
    # None: each channel's own, from its cells (a rule of thumb); a number: every channel's
    expected_doublet_rate: float | None = None
    # The channels: an obs column, "__all__" (all cells as one channel), or empty: a
    # channel column, else the barcodes' -N / PREFIX_, else donor / batch, else all cells
    doublet_channel_key: str = ""

    # Normalization: "log" = shifted log, normalize_total + log1p (the default) |
    # "pearson_residuals" = analytic Pearson residuals of the counts, for HVG
    # selection and PCA only; .raw stays log-normalised (see core/normalization.py).
    normalization_method: str = "log"
    target_sum: float = 1e4

    # HVG
    n_top_genes: int = 2000
    hvg_flavor: str = "seurat"
    hvg_batch_key: str = ""  # obs column for batch-aware HVG selection (empty = pooled)

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
    # Drop ribosomal/mitochondrial genes from the query (scRNA-seq artifact); see
    # core/gene_filters.py.
    enrichment_exclude_ribo_mito: bool = True
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
    annotation_marker_sets: str = "databases"    # marker_score: databases (core/marker_db.py) | builtin (14 immune sets)

    # Large-dataset mode (core/large_run.py): "auto" runs the steps with the matrices on
    # disk, in row blocks, when the run would not fit the memory budget in memory; "on"
    # always, "off" never.
    chunked: str = "auto"


CHUNKED_MODES: tuple[str, ...] = ("auto", "on", "off")


def check_chunked(value: Any) -> str:
    """``PipelineParams.chunked`` as one of ``CHUNKED_MODES`` (true/false as on/off); a
    ValueError naming the choices for anything else."""
    if isinstance(value, bool):
        return "on" if value else "off"
    mode = str(value if value is not None else "auto").strip().lower()
    if mode not in CHUNKED_MODES:
        raise ValueError(f"Unknown chunked mode '{value}'; choose from {', '.join(CHUNKED_MODES)}.")
    return mode


@dataclass
class PipelineResult:
    """Result of running the preprocessing pipeline."""

    steps_run: list[str] = field(default_factory=list)
    steps_skipped: list[str] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)
    # Steps that ran but had nothing to do, and why (not errors; see StepSkipped).
    notes: dict[str, str] = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    output_path: str | None = None
    # "complete" (ran to the end, errors or not) | "stopped" (the pre-step check
    # stopped it before a step) | "cancelled"; neither of the last two saves
    # anything. See core/run_watchdog.py.
    status: str = "complete"
    stop_reason: str | None = None
    # Requested names that are not steps (the run endpoints refuse these with 422).
    unknown_steps: list[str] = field(default_factory=list)
    # What the run used (RunWatchdog.summary + write_s), when it was watched.
    resources: dict[str, Any] | None = None
    # Ran in large-dataset mode (core/large_run.py), and how the mode was chosen
    # (``ModeChoice.to_dict``: chunked, requested, reason, need / free bytes).
    chunked: bool = False
    mode: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class StepSkipped(Exception):
    """A step with nothing to do on this dataset, e.g. too few signature genes.

    Not an error: the run goes on, nothing is recorded in provenance, and the
    reason is returned in ``PipelineResult.notes``."""


# The large-dataset run's stand-ins for the steps below that read the matrix
# (core/large_run.py ``LargeBackend``): marker tests from the gene store, gene scores and
# CellTypist by row blocks, so those steps run unchanged on a copy of the dataset without
# X. None: in memory.
_LARGE: contextvars.ContextVar[Any] = contextvars.ContextVar("scview_large_backend",
                                                             default=None)


@contextlib.contextmanager
def large_backend(backend: Any):
    """Run steps in this thread with ``backend`` reading the matrices (see ``_LARGE``)."""
    token = _LARGE.set(backend)
    try:
        yield backend
    finally:
        _LARGE.reset(token)


def _wilcoxon(adata: ad.AnnData, groupby: str, key_added: str | None = None) -> None:
    """Wilcoxon markers with pts, as ``sc.tl.rank_genes_groups`` writes them (into
    ``uns[key_added or "rank_genes_groups"]``), by blocks of genes (``core/rank_genes.py``,
    bit for bit scanpy's); in a large-dataset run from its gene store. Either way with the
    specificity frame already in (``sparse_markers``) unless scanpy ran."""
    backend = _LARGE.get()
    if backend is not None:
        backend.rank_genes_groups(adata, groupby, key_added=key_added)
    else:
        from scview.core import rank_genes

        rank_genes.rank_genes_groups(adata, groupby, method="wilcoxon", pts=True,
                                     key_added=key_added, specificity=True)


def _attach_specificity(adata: ad.AnnData, rgg: Any, col: str) -> None:
    if not marker_specificity.has_specificity(rgg):  # markers from a gene store have it
        marker_specificity.attach(adata, rgg, col)


def _score_genes(adata: ad.AnnData, genes, *, score_name: str, use_raw=None,
                 ctrl_size: int = 50, random_state: int = 0) -> None:
    """``sc.tl.score_genes``; in a large-dataset run the same scores by row blocks."""
    backend = _LARGE.get()
    if backend is not None:
        backend.score_genes(adata, genes, score_name=score_name, ctrl_size=ctrl_size,
                            random_state=random_state)
    else:
        sc.tl.score_genes(adata, genes, score_name=score_name, use_raw=use_raw,
                          ctrl_size=ctrl_size, random_state=random_state)


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
    "uncorrected_embedding",
    "batch_correction",
    "neighbors",
    "clustering",
    "embeddings",
    "marker_genes",
    "enrichment",
    "cell_cycle",
    "dissociation_stress",
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
    clear_processing(adata)
    logger.info("Reset to counts (%d cells, %d genes)", adata.n_obs, adata.n_vars)
    return adata


def clear_processing(adata: ad.AnnData) -> None:
    """What resetting to counts clears besides X: .raw, the variable-gene columns,
    embeddings, graphs, the results in uns and scView's obs columns (the large-dataset
    run clears the same on its matrix-free copy, core/large_run.py)."""
    adata.raw = None

    # Remove HVG annotation from var (both flavours' columns)
    for col in ["highly_variable", "means", "dispersions", "dispersions_norm",
                "variances", "residual_variances", "highly_variable_rank",
                "highly_variable_nbatches", "highly_variable_intersection"]:
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
                "pca", "umap", "leiden", "louvain", BATCH_MIXING_KEY, scaling.PLAN_KEY]:
        if key in adata.uns:
            del adata.uns[key]
    multi_keys = [k for k in list(adata.uns.keys())
                  if k.startswith("rank_genes_groups__") or k.startswith("enrichment__")]
    for key in multi_keys:
        del adata.uns[key]

    # Remove scview-generated obs columns, except the cell types a per-type % mito
    # filter judges by (core/mito_refilter.py): an input to filtering, not a result.
    scview_cols = [c for c in adata.obs.columns if c.startswith("scview_") and c != "scview_mito_type"]
    if scview_cols:
        adata.obs.drop(columns=scview_cols, inplace=True, errors="ignore")


def _run_qc_metrics(adata: ad.AnnData, params: PipelineParams) -> None:
    """Calculate QC metrics including mitochondrial gene percentage."""
    adata.var["mt"] = adata.var_names.str.startswith("MT-") | adata.var_names.str.startswith("mt-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)
    logger.info(
        "QC metrics: median genes/cell=%d, median counts/cell=%d",
        int(adata.obs["n_genes_by_counts"].median()),
        int(adata.obs["total_counts"].median()),
    )


def _holds_processed_values(M) -> bool:
    """Values that are not counts (negative or fractional); an empty matrix is not."""
    if sp.issparse(M):
        has = M.nnz > 0
    else:
        has = bool(np.any(np.asarray(M).ravel()[:_COUNTS_SAMPLE * 4]))
    return has and not looks_like_counts(M)


def _doublet_counts(adata: ad.AnnData):
    """(matrix, gene names, where from) of the counts beside a processed X, or None."""
    if "counts" in adata.layers and looks_like_counts(adata.layers["counts"]):
        return adata.layers["counts"], adata.var_names, "layers['counts']"
    if adata.raw is not None and looks_like_counts(adata.raw.X):
        return adata.raw.X, adata.raw.var_names, ".raw"
    return None


def _doublet_matrix(adata: ad.AnnData):
    """(counts, where from) for Scrublet: X, or the counts beside a processed X (where:
    "layers['counts']" / ".raw"; None for X)."""
    if not _holds_processed_values(adata.X):
        return adata.X, None
    # Run alone on a processed dataset, X is scaled: Scrublet then filled a 6 GB
    # backend and failed ("Bin edges must be unique: nan"). Score the counts.
    counts = _doublet_counts(adata)
    if counts is None:
        raise StepSkipped("Doublet detection needs raw counts: X holds processed values and "
                          "there is no counts layer or counts in .raw.")
    return counts[0], counts[2]


def _run_doublet_detection(adata: ad.AnnData, params: PipelineParams) -> dict[str, Any]:
    """Detect doublets with Scrublet in each capture channel; annotate ``.obs`` only
    (removal is opt-in).

    Writes ``doublet_score`` (float; NaN for a cell not scored) and ``predicted_doublet``
    (bool) to ``.obs``. Ordered before normalization so Scrublet sees raw counts. The
    flags are informational here — actual removal happens in the filtering step when
    ``params.drop_doublets`` is set, so users can inspect the QC distribution first.

    Scrublet runs on each capture channel's cells alone, one after another
    (``doublets.run_scrublet``: the channels from ``doublet_channel_key``, else a channel
    column, the barcodes, the donors / batches, else all cells as one; channels read from
    the barcodes are written to ``capture_channel``). Each channel gets its own expected
    rate from its cells unless ``expected_doublet_rate`` is set, its own threshold, and
    exact neighbours up to ``doublets.EXACT_KNN_MAX_CELLS`` cells; a channel under
    ``doublets.MIN_CHANNEL_CELLS`` is not scored. The channels and each one's rate,
    threshold and calls are recorded in provenance. Scrublet scores a sparse matrix of
    the counts alone (``scrublet_lean.scrublet_counts``: the scores ``sc.pp.scrublet``
    gives, without copying the whole dataset); they come from ``layers['counts']`` or
    ``.raw`` when X holds processed values (recorded as ``input``).
    """
    method = (params.doublet_method or "scrublet").lower()
    if method != "scrublet":
        logger.warning("Unknown doublet method '%s'; falling back to scrublet", method)
    counts, where = _doublet_matrix(adata)
    # A channel column that is not there raises DoubletError: the step's error.
    found = doublets.run_scrublet(adata, counts, params.expected_doublet_rate,
                                  params.doublet_channel_key or None)
    if where is not None:
        found["input"] = where
    return found


def _run_filtering(adata: ad.AnnData, params: PipelineParams) -> tuple[ad.AnnData, dict[str, Any]]:
    """Filter cells on QC thresholds, then genes seen in too few cells.

    Returns the new AnnData and what to record beside the params: in MAD mode
    each sample's thresholds and loss (``qc_thresholds``). Cells are judged on
    the per-cell QC values the QC plots show (``per_cell_qc``: obs QC columns,
    else computed from the counts) by ``qc_filter.judge``, as the Data Assessment
    preview is, so its loss is what this removes, and a re-run on processed data
    (X scaled, every gene "detected") still counts genes in the counts. % mito is
    used only when measurable; predicted doublets are dropped when
    ``drop_doublets`` is set (``qc_doublet_source``: cell hashing when it was run,
    else Scrublet, by default), hashing negatives when ``drop_hash_negatives`` is.
    V(D)J clone sizes are recomputed on the cells kept (``qc_filter.after_filter``).

    The cells removed are recorded in ``uns[qc_filter.RECORD_KEY]`` (their QC
    values and labels), so the Data Assessment audit can show who was removed
    after they are gone.

    With ``qc_mito_mode`` "per_type" each cell type's own % mito bound replaces
    the global rule for its cells (``qc_mito_types``: every type's rule and loss);
    with ``qc_mito_type_floor`` never stricter than the global rule.
    """
    n_before = adata.n_obs
    t, keep, record, found = filter_judgement(adata, params)
    if not keep.all():
        adata = subset_lean(adata, obs=keep)
    adata.uns[qc_filter.RECORD_KEY] = record
    # Whose doublets were dropped; V(D)J clone sizes recomputed on the cells kept.
    found.update(qc_filter.after_filter(adata, t, int((~keep).sum())))
    adata = _filter_genes_lean(adata, params.min_cells)
    logger.info("Filtering (%s): %d -> %d cells", t.mode, n_before, adata.n_obs)
    return adata, found


def filter_judgement(adata: ad.AnnData, params: PipelineParams):
    """The cells the filtering step keeps, from obs alone: (thresholds, keep mask, removal
    record, what to record beside the params). ``_run_filtering`` and the large-dataset
    run (core/large_run.py) then cut the matrices each its own way."""
    n_before = adata.n_obs
    t = qc_filter.Thresholds.from_params(params)
    samples = qc_filter.samples_of(adata, params.qc_sample_column or None)
    cells = per_cell_qc(adata)
    removed, bounds, types = qc_filter.judge_types(adata, cells, t, samples)
    keep = qc_filter.keep_mask(removed, n_before)
    record = qc_filter.removal_record(adata, cells, removed, ~keep, t, samples.column, types=types)
    for name, mask in removed.items():
        logger.info("Filtering: %s removes %d cells on its own", name, int(mask.sum()))
    # Only the params this mode used, so the record says what was applied.
    used = (("min_genes", "max_genes", "min_counts") if t.mode == "fixed"
            else ("qc_mad_k", "qc_mad_k_mt", "qc_mad_upper", "qc_sample_column"))
    found: dict[str, Any] = {k: getattr(params, k) for k in used}
    if bounds is not None:
        found["qc_thresholds"] = bounds.to_dict(~keep)
    if t.mito_mode == "per_type":  # a global run records what it always did
        found.update(qc_mito_mode="per_type", qc_mad_k_mt=params.qc_mad_k_mt,
                     qc_mito_type_floor=bool(params.qc_mito_type_floor))  # absent in old records: off
        if types is not None:
            found.update(qc_mito_type_column=types.column, qc_mito_types=types.to_dict())
        else:
            found.update(qc_mito_type_column=params.qc_mito_type_column,
                         qc_mito_note="% mito can't be measured (no MT- genes): no per-type bounds.")
    return t, keep, record, found


def _plain_matrix(M) -> bool:
    return M is None or isinstance(M, np.ndarray) or sp.issparse(M)


def subset_lean(adata: ad.AnnData, obs=None, var=None) -> ad.AnnData:
    """``adata[obs, var].copy()`` (boolean masks, one axis at a time), taking one copy of
    each matrix instead of two, and none of a matrix whose genes all stay.

    anndata subsets X, every layer and .raw into a view and then copies that: two new
    arrays per matrix while ``adata`` still holds its own. On the hashing PBMC (a dense
    scaled X of every gene, 1.7 GiB) that filled a 6 GB backend. Here X, the layers and
    .raw are taken off ``adata`` first, anndata subsets the rest as it always does (obs
    with its unused categories and their colours in uns, var, uns, obsm, obsp, varm,
    varp), and each matrix is cut once, the old one let go as soon as its cut exists.

    ``adata`` gives its matrices up: it is the copy of the dataset the run replaces
    (core/processing.py evicts it and reloads the dataset from disk when the run ends,
    saved, stopped or cancelled). If anndata's part fails, ``adata`` gets them back
    untouched. A backed or view object, or a matrix that is not numpy or scipy, takes
    anndata's own way."""
    raw = adata.raw
    # (no list of the matrices kept: it would hold each old one until the end)
    plain = all(_plain_matrix(M) for M in (adata.X, *adata.layers.values(),
                                            raw.X if raw is not None else None))
    if adata.isbacked or adata.is_view or not plain:
        return adata[slice(None) if obs is None else obs,
                     slice(None) if var is None else var].copy()
    rows = None if obs is None else np.flatnonzero(obs)
    cols = None if var is None or np.all(var) else np.flatnonzero(var)
    X, layers = adata.X, dict(adata.layers)
    adata.X = None
    for k in layers:
        del adata.layers[k]
    adata.raw = None
    try:
        new = adata[slice(None) if rows is None else rows,
                    slice(None) if var is None else np.asarray(var, dtype=bool)].copy()
    except BaseException:
        adata.X = X
        for k, M in layers.items():
            adata.layers[k] = M
        if raw is not None:
            adata.raw = ad.AnnData(X=raw.X, var=raw.var, varm=dict(raw.varm))
        raise

    def cut(M, x: bool = False):
        if rows is not None:
            return np.ascontiguousarray(M[rows]) if x and isinstance(M, np.ndarray) else M[rows]
        if cols is None:
            return M  # every gene stays: the same matrix (anndata would re-lay it out)
        # As anndata's copy lays them out: a dense X row-major (np.take: in one copy;
        # M[:, cols] is column-major), a dense layer as numpy's indexing leaves it.
        if x and isinstance(M, np.ndarray):
            return np.take(M, cols, axis=1)
        return M[:, cols]

    if X is not None:
        new.X = cut(X, x=True)
        del X
    for k in list(layers):
        new.layers[k] = cut(layers.pop(k))
    if raw is not None:  # rows only: subsetting genes leaves .raw as it is
        new.raw = ad.AnnData(X=raw.X[rows] if rows is not None else raw.X, var=raw.var.copy(),
                             varm={k: v.copy() for k, v in raw.varm.items()})
    return new


def _filter_genes_lean(adata: ad.AnnData, min_cells: int) -> ad.AnnData:
    """``sc.pp.filter_genes(adata, min_cells=...)`` (``var["n_cells"]``, then the genes kept)
    through ``subset_lean``: scanpy subsets every matrix even when every gene stays."""
    gene_subset, number = sc.pp.filter_genes(adata, min_cells=min_cells, inplace=False)
    adata.var["n_cells"] = number
    return subset_lean(adata, var=np.asarray(gene_subset, dtype=bool))


# Non-zero values sampled when deciding whether a matrix holds raw counts.
_COUNTS_SAMPLE = 20_000
_COUNTS_ROWS = 2_000  # rows read from an on-disk matrix to judge it


def looks_like_counts(M) -> bool:
    """Non-negative integers with some signal: a raw-count matrix."""
    if M is None:
        return False
    if not sp.issparse(M) and not isinstance(M, np.ndarray) and hasattr(M, "shape"):
        # An on-disk (backed) matrix, e.g. a file over 2 GB: read the first rows only.
        # np.asarray of the whole thing failed (sparse) or read every value (dense).
        try:
            M = M[: min(M.shape[0], _COUNTS_ROWS)]
        except Exception:  # noqa: BLE001 - not sliceable: nothing to judge
            return False
    if sp.issparse(M):
        vals = np.asarray(M.data[:_COUNTS_SAMPLE])
    else:
        arr = np.asarray(M)
        vals = arr.ravel()[:_COUNTS_SAMPLE * 4]
        vals = vals[vals != 0][:_COUNTS_SAMPLE]
    if vals.size == 0:
        return False
    vals = vals.astype(np.float64, copy=False)
    return bool(np.all(vals >= 0) and np.all(np.mod(vals, 1) == 0))


def _keep_raw_counts(adata: ad.AnnData) -> None:
    """Copy counts held only in ``.raw`` into ``layers['counts']``.

    Uploads often carry counts in ``.raw`` alone. ``_run_scaling`` replaces
    ``.raw`` with the log-normalised matrix, which would leave no counts anywhere.
    """
    if "counts" in adata.layers or adata.raw is None:
        return
    if not looks_like_counts(adata.raw.X):
        return
    if len(adata.var_names.difference(adata.raw.var_names)):
        logger.warning("adata.raw lacks some genes in X; raw counts not copied to layers")
        return
    adata.layers["counts"] = adata.raw[:, adata.var_names].X.copy()
    logger.info("Kept raw counts from adata.raw in layers['counts']")


def _run_normalization(adata: ad.AnnData, params: PipelineParams) -> None:
    """Normalize to target sum per cell, store raw counts.

    Under both normalisation methods: with Pearson residuals, this shifted log
    still becomes .raw, which markers, DE and the scores read."""
    normalization.check_method(params.normalization_method)  # fail before any work
    # Save raw counts before normalization
    if "counts" not in adata.layers:
        if adata.raw is None or looks_like_counts(adata.X):
            adata.layers["counts"] = adata.X.copy()
        else:
            _keep_raw_counts(adata)
    sc.pp.normalize_total(adata, target_sum=params.target_sum)
    scaling.drop_plan(adata)  # X changed: a scaling plan's statistics no longer hold
    logger.info("Normalized to target sum %.0f", params.target_sum)


def _run_log_transform(adata: ad.AnnData, params: PipelineParams) -> None:
    """Log-transform expression values (log1p)."""
    sc.pp.log1p(adata)
    scaling.drop_plan(adata)
    logger.info("Applied log1p transformation")


# Flavours offered in the UI. `seurat_v3` ranks the raw counts (loess via scikit-misc).
HVG_FLAVORS: tuple[str, ...] = ("seurat", "cell_ranger", "seurat_v3")


def _var_from_hvg_frame(adata: ad.AnnData, df: pd.DataFrame) -> None:
    """Replace the variable-gene columns in ``var`` with one flavour's results."""
    for col in ("dispersions", "dispersions_norm", "variances", "variances_norm",
                "highly_variable_rank", "residual_variances"):  # another flavour's, now stale
        if col in adata.var.columns and col not in df.columns:
            del adata.var[col]
    for col in df.columns:
        adata.var[col] = df[col].to_numpy()
    adata.var["highly_variable"] = adata.var["highly_variable"].astype(bool)


def _wide_ints(counts):
    """Dense counts of a small integer type as int64. scanpy squares a dense matrix in
    its own dtype for the gene variances, so an int32 count over 46,340 overflowed
    into a negative variance and wrong Seurat v3 variable genes."""
    if isinstance(counts, np.ndarray) and counts.dtype.kind in "iu" and counts.dtype.itemsize < 8:
        return counts.astype(np.int64)
    return counts


def _seurat_v3_hvg(adata: ad.AnnData, params: PipelineParams, batch_key: str | None) -> dict[str, Any]:
    """Variable genes ranked by the standardised variance of the counts (Seurat v3)."""
    _keep_raw_counts(adata)
    try:
        counts, source = normalization.find_counts(adata)
    except ValueError:
        raise ValueError(
            "The Seurat v3 flavour ranks raw counts (layers['counts'], .raw or X), and this "
            "dataset has none: choose the Seurat or Cell Ranger flavour."
        ) from None
    tmp = ad.AnnData(X=_wide_ints(counts), obs=adata.obs[[batch_key]] if batch_key else None)
    tmp.var_names = adata.var_names
    df = sc.pp.highly_variable_genes(
        tmp, flavor="seurat_v3", n_top_genes=min(params.n_top_genes, adata.n_vars),
        batch_key=batch_key, inplace=False,
    )
    _var_from_hvg_frame(adata, df)
    logger.info("Selected %d highly variable genes (Seurat v3; counts from %s, batch_key=%s)",
                int(adata.var["highly_variable"].sum()), source, batch_key)
    return {"counts_source": source}


def _pearson_hvg(adata: ad.AnnData, params: PipelineParams, batch_key: str | None) -> dict[str, Any]:
    """Variable genes ranked by the variance of their Pearson residuals, from the counts."""
    _keep_raw_counts(adata)  # counts held only in .raw: kept, as scaling would keep them
    counts, source = normalization.find_counts(adata)
    tmp = ad.AnnData(X=counts, obs=adata.obs[[batch_key]] if batch_key else None)
    tmp.var_names = adata.var_names
    df = sc.experimental.pp.highly_variable_genes(
        tmp, flavor="pearson_residuals", n_top_genes=min(params.n_top_genes, adata.n_vars),
        batch_key=batch_key, theta=normalization.THETA, inplace=False,
    )
    _var_from_hvg_frame(adata, df)
    logger.info("Selected %d highly variable genes (Pearson residuals; counts from %s, "
                "batch_key=%s)", int(adata.var["highly_variable"].sum()), source, batch_key)
    return {"hvg_flavor": "pearson_residuals", "theta": normalization.THETA,
            "counts_source": source}


def _run_hvg(adata: ad.AnnData, params: PipelineParams) -> dict[str, Any] | None:
    """Identify highly variable genes: on the log values, or with Pearson residuals
    chosen, by residual variance (``hvg_flavor`` is then not used)."""
    method = normalization.check_method(params.normalization_method)
    if method == "log" and params.hvg_flavor not in HVG_FLAVORS:
        raise ValueError(
            f"Unsupported HVG flavor '{params.hvg_flavor}'; choose from {', '.join(HVG_FLAVORS)}."
        )
    # A batch key ranks genes by how many batches call them variable, so a gene
    # that only differs *between* batches isn't picked as biology.
    batch_key = params.hvg_batch_key or None
    if batch_key and batch_key not in adata.obs.columns:
        raise ValueError(f"HVG batch key '{batch_key}' not found in obs columns.")
    if method == "pearson_residuals":
        return _pearson_hvg(adata, params, batch_key)
    if params.hvg_flavor == "seurat_v3":
        return _seurat_v3_hvg(adata, params, batch_key)
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=params.n_top_genes,
        flavor=params.hvg_flavor,
        batch_key=batch_key,
    )
    n_hvg = int(adata.var["highly_variable"].sum())
    logger.info("Selected %d highly variable genes (flavor=%s, batch_key=%s)",
                n_hvg, params.hvg_flavor, batch_key)
    return None


def _run_scaling(adata: ad.AnnData, params: PipelineParams) -> dict[str, Any]:
    """Plan the matrix PCA reads (core/scaling.py): z-scores of the log values of the
    variable genes (every gene without them), or, with Pearson residuals chosen, the
    residuals of the counts. Nothing dense is made here: X keeps the log-normalised
    matrix of every gene, .raw is set to it (markers, DE and the scores read it), and
    the PCA step builds the scaled matrix for its own length. The old step made X dense
    float32 for every gene and z-scored it: 38 of a 46 GB run peak at 250k cells."""
    method = normalization.check_method(params.normalization_method)
    # Counts held only in .raw: kept in layers['counts'] before .raw is replaced.
    _keep_raw_counts(adata)
    if method == "pearson_residuals":
        counts, source = normalization.find_counts(adata)
        if looks_like_counts(adata.X):
            raise ValueError(
                "Pearson residuals replace only the matrix PCA reads: run normalization "
                "and log transform too, so markers, DE and plots keep log-normalised values."
            )
        _raw_from_x(adata)
        clip = normalization.default_clip(adata.n_obs)
        found = scaling.plan_residuals(adata, normalization.THETA, clip, source)
        return {"theta": normalization.THETA, "clip": round(clip, 3), "counts_source": source,
                **found}
    _raw_from_x(adata)
    return scaling.plan_zscores(adata, max_value=scaling.MAX_VALUE)


def _raw_from_x(adata: ad.AnnData) -> None:
    """``.raw = adata``: the log-normalised matrix, X itself. Over an X an older scView
    scaled (z-scores or residuals), .raw keeps the log values it has."""
    if adata.raw is not None and scaling.holds_scaled(adata.X):
        logger.info("X holds scaled values (an older run): .raw kept as it is")
        return
    adata.raw = adata


def _run_pca(adata: ad.AnnData, params: PipelineParams) -> dict[str, Any] | None:
    """PCA on the matrix the scaling step planned (``scaling.run_pca``: the variable
    genes, scaled for this step only), else on X as it is: the log values when scaling
    was skipped, or the z-scores an older scView wrote into X. Stored as sc.tl.pca
    stores it either way."""
    n_comps = min(params.n_comps, adata.n_obs - 1, adata.n_vars - 1)
    found = None
    if scaling.read_plan(adata) is not None:
        found = scaling.run_pca(adata, n_comps)
    else:
        sc.tl.pca(adata, n_comps=n_comps, svd_solver="arpack")
    # A Harmony embedding of the previous PCA would otherwise still be picked up
    # by the neighbour graph; the batch-correction step rebuilds it if it runs.
    adata.obsm.pop("X_pca_harmony", None)
    adata.uns.pop(BATCH_MIXING_KEY, None)  # scored the previous PCA
    logger.info("PCA: %d components", n_comps)
    return found


def _run_uncorrected_embedding(adata: ad.AnnData, params: PipelineParams) -> None:
    """UMAP of the *uncorrected* PCA, stored as ``X_umap_uncorrected``.

    Run before batch correction so the corrected and uncorrected layouts can be
    compared side by side — the most direct check of whether integration mixed
    the batches (or over-corrected real differences). Uses its own neighbor
    graph so the main ``neighbors`` graph is left for the corrected space.
    """
    if "X_pca" not in adata.obsm:
        raise ValueError("PCA must be computed before the uncorrected embedding.")
    key = "uncorrected"
    sc.pp.neighbors(adata, n_neighbors=params.n_neighbors, use_rep="X_pca", key_added=key)
    sc.tl.umap(adata, min_dist=params.umap_min_dist, neighbors_key=key,
               key_added="X_umap_uncorrected")
    # Only the layout is kept; the graph would roughly double the file size.
    for k in (f"{key}_connectivities", f"{key}_distances"):
        adata.obsp.pop(k, None)
    adata.uns.pop(key, None)
    adata.uns.pop("X_umap_uncorrected", None)
    logger.info("Uncorrected UMAP -> X_umap_uncorrected %s", adata.obsm["X_umap_uncorrected"].shape)


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
    adata.uns.pop(BATCH_MIXING_KEY, None)  # scored the previous correction
    logger.info("Harmony batch correction using key '%s' -> X_pca_harmony %s",
                params.batch_key, adata.obsm["X_pca_harmony"].shape)


def _run_neighbors(adata: ad.AnnData, params: PipelineParams) -> None:
    """Compute nearest neighbor graph."""
    # Use harmony-corrected PCA if available
    use_rep = "X_pca_harmony" if "X_pca_harmony" in adata.obsm else "X_pca"
    sc.pp.neighbors(adata, n_neighbors=params.n_neighbors, use_rep=use_rep)
    logger.info("Neighbor graph: k=%d, use_rep=%s", params.n_neighbors, use_rep)


def _louvain_igraph(adata: ad.AnnData, resolution: float, key_added: str, seed: int = 0) -> None:
    """Louvain clustering on the neighbour graph with python-igraph.

    ``sc.tl.louvain`` needs the unmaintained ``louvain`` package, and its igraph
    flavor ignores ``resolution``; igraph's own multilevel method honours it.
    Clusters are numbered by size, largest first, as Leiden's are.
    """
    import random

    import igraph as ig
    import pandas as pd

    adj = sp.triu(adata.obsp["connectivities"], k=1).tocoo()
    g = ig.Graph(n=adata.n_obs, edges=np.column_stack([adj.row, adj.col]).tolist(), directed=False)
    g.es["weight"] = adj.data.tolist()
    ig.set_random_number_generator(random.Random(seed))
    try:
        membership = np.asarray(g.community_multilevel(weights="weight", resolution=resolution).membership)
    finally:
        ig.set_random_number_generator(random)
    order = np.argsort(-np.bincount(membership), kind="stable")
    rank = np.empty_like(order)
    rank[order] = np.arange(order.size)
    labels = rank[membership].astype(str)
    adata.obs[key_added] = pd.Categorical(labels, categories=[str(i) for i in range(order.size)])


def _run_clustering(adata: ad.AnnData, params: PipelineParams) -> None:
    """Run clustering (Leiden or Louvain).

    Stores results in a custom column ``scview_{method}_r{resolution}`` so that
    the original dataset's clustering column (if any) is preserved.  The active
    column name is recorded in ``adata.uns["scview_active_clustering"]`` for
    downstream steps (marker genes, violin groupby, etc.).
    """
    method = params.clustering_method
    # Written straight to the scview_* column: scanpy's default key ("leiden" /
    # "louvain") would overwrite a clustering the dataset came with.
    scview_col = f"scview_{method}_r{params.clustering_resolution}"
    if method == "leiden":
        leiden.leiden(adata, params.clustering_resolution, scview_col)  # in a child process
    else:
        _louvain_igraph(adata, params.clustering_resolution, scview_col)
    adata.uns["scview_active_clustering"] = scview_col

    n_clusters = adata.obs[scview_col].nunique()
    logger.info(
        "Clustering (%s): %d clusters at resolution %.2f → column '%s'",
        method, n_clusters, params.clustering_resolution, scview_col,
    )


def _run_embeddings(adata: ad.AnnData, params: PipelineParams) -> None:
    """Compute 2D and 3D UMAP embeddings."""
    # sc.tl.umap hands umap-learn the neighbour graph without copying it, and
    # umap-learn zeroes the graph's weakest edges in place. The saved graph was
    # then not the one the clustering ran on: re-clustering it (the resolution
    # sweep, "Keep this one") gave 11 clusters on Kang where the pipeline's had 10.
    key = (adata.uns.get("neighbors") or {}).get("connectivities_key", "connectivities")
    graph = adata.obsp[key].copy() if key in adata.obsp else None

    # 3D UMAP first (stored temporarily in X_umap, then moved)
    sc.tl.umap(adata, min_dist=params.umap_min_dist, n_components=3)
    adata.obsm["X_umap_3d"] = adata.obsm["X_umap"].copy()

    # 2D UMAP (overwrites X_umap)
    sc.tl.umap(adata, min_dist=params.umap_min_dist, n_components=2)
    if graph is not None:
        adata.obsp[key] = graph
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
        _wilcoxon(adata, col)
        # Store under namespaced key
        adata.uns[markers_key(col)] = adata.uns["rank_genes_groups"].copy()
        _attach_specificity(adata, adata.uns[markers_key(col)], col)
        drop_enrichment_cache(adata, col)
        computed += 1

    # Leave rank_genes_groups pointing to the active clustering for backward compat
    if active_clustering and markers_key(active_clustering) in adata.uns:
        adata.uns["rank_genes_groups"] = adata.uns[markers_key(active_clustering)]
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
    # Track the gene-set selection actually used, so the cache sidecar records it and a
    # later request with a different selection recomputes instead of reusing this payload.
    used_collections: list[str]
    # MSigDB + custom collections merged, when local gene sets are used.
    resolved: ResolvedGeneSets | None = None
    if params.enrichment_collections:
        import os
        from scview.config import get_settings
        from scview.core.custom_genesets import get_custom_store
        from scview.core.msigdb_loader import get_msigdb_loader

        msigdb_dir = os.environ.get("MSIGDB_DIR", "/msigdb")
        loader = get_msigdb_loader(msigdb_dir)
        resolved = resolve_gene_sets(
            params.enrichment_collections, loader, get_custom_store(get_settings().DATA_DIR)
        )
        if resolved.missing_custom:
            logger.warning("Custom gene-set collections not found (deleted?): %s",
                           resolved.missing_custom)
        if loader or resolved.custom_names:
            gene_sets_arg = resolved.gene_sets
            if not gene_sets_arg:
                logger.warning("No gene sets loaded from collections: %s", params.enrichment_collections)
                return
            logger.info("Using %d local gene sets from %d collections", len(gene_sets_arg), len(params.enrichment_collections))
            used_collections = list(params.enrichment_collections)
        else:
            logger.warning("MSigDB loader not available; falling back to Enrichr libraries")
            resolved = None
            gene_sets_arg = params.enrichment_gene_sets or ["GO_Biological_Process_2025"]
            used_collections = list(gene_sets_arg)
    elif params.enrichment_gene_sets:
        gene_sets_arg = params.enrichment_gene_sets
        used_collections = list(gene_sets_arg)
    else:
        logger.warning("No enrichment collections or gene sets specified, skipping")
        return

    # Build a flat list of (col, uns_key, groups) for progress tracking
    work_items: list[tuple[str, str, str]] = []  # (col, uns_key, group)
    for col in columns:
        uns_key = markers_key(col)
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
    queries: dict[str, dict[str, list[str]]] = {}
    from scview.core.marker_selection import group_markers

    for wi, (col, uns_key, group) in enumerate(work_items):
        cb(f"Enrichment: {col} / {group}", wi, total_items)

        rgg = adata.uns[uns_key]
        # The full ranked list is the set of genes the DE test could see, and is
        # the correct ORA universe. Without it gseapy tests against every gene in
        # the selected collections and p-values come out anti-conservative.
        background = [str(g) for g in rgg["names"][group]]
        # The query is marker_selection's choice (in >= 20% of the group, adjusted
        # p < 0.05, by fold change), computed once per column for all its groups.
        if col not in queries:
            queries[col] = group_markers(
                adata, col, n=params.enrichment_n_genes, key=uns_key,
                exclude_ribo_mito=params.enrichment_exclude_ribo_mito,
            )
        top_genes = queries[col].get(str(group), [])

        try:
            enr = gp.enrich(
                gene_list=top_genes,
                gene_sets=(
                    match_query_case(resolved, top_genes, background)
                    if resolved is not None else gene_sets_arg
                ),
                background=background,
                outdir=None,
                no_plot=True,
                cutoff=0.5,
            )
            store_key = enrichment_key(col, group)
            # With no overlap at all gseapy leaves results as an empty list.
            if not getattr(enr.results, "empty", True):
                from scview.api.v1.enrichment import _normalize_gseapy_record
                normalized = [
                    _normalize_gseapy_record(r) for r in enr.results.to_dict(orient="records")
                ]
                if resolved is not None:
                    tag_records(normalized, resolved)
                adata.uns[store_key] = json.dumps(normalized)
            else:
                adata.uns[store_key] = "[]"
            write_meta(
                adata,
                col,
                group,
                background_n=len(background),
                n_genes=params.enrichment_n_genes,
                collections=used_collections,
                exclude_ribo_mito=params.enrichment_exclude_ribo_mito,
            )
            total_groups += 1
        except Exception as e:
            logger.warning("Enrichment failed for %s/%s: %s", col, group, e)

    logger.info("Enrichment completed: %d total group results", total_groups)


# Tirosh et al. 2016 (Science 352:189) S and G2/M genes, as in scanpy's tutorial
# list and Seurat's cc.genes. Written as human symbols and matched ignoring case:
# each gene's mouse ortholog is the title-cased symbol (MCM5 -> Mcm5).
CELL_CYCLE_S_GENES = (
    "MCM5", "PCNA", "TYMS", "FEN1", "MCM2", "MCM4", "RRM1", "UNG",
    "GINS2", "MCM6", "CDCA7", "DTL", "PRIM1", "UHRF1", "MLF1IP",
    "HELLS", "RFC2", "RPA2", "NASP", "RAD51AP1", "GMNN", "WDR76",
    "SLBP", "CCNE2", "UBR7", "POLD3", "MSH2", "ATAD2", "RAD51",
    "RRM2", "CDC45", "CDC6", "EXO1", "TIPIN", "DSCC1", "BLM",
    "CASP8AP2", "USP1", "CLSPN", "POLA1", "CHAF1B", "BRIP1", "E2F8",
)
CELL_CYCLE_G2M_GENES = (
    "HMGB2", "CDK1", "NUSAP1", "UBE2C", "BIRC5", "TPX2", "TOP2A",
    "NDC80", "CKS2", "NUF2", "CKS1B", "MKI67", "TMPO", "CENPF",
    "TACC3", "FAM64A", "SMC4", "CCNB2", "CKAP2L", "CKAP2", "AURKB",
    "BUB1", "KIF11", "ANP32E", "TUBB4B", "GTSE1", "KIF20B", "HJURP",
    "CDCA3", "HN1", "CDC20", "TTK", "CDC25C", "KIF2C", "RANGAP1",
    "NCAPD2", "DLGAP5", "CDCA2", "CDCA8", "ECT2", "KIF23", "HMMR",
    "AURKA", "PSRC1", "ANLN", "LBR", "CKAP5", "CENPE", "CTCF",
    "NEK2", "G2E3", "GAS2L3", "CBX5", "CENPA",
)
# Three genes were renamed in both species after 2016 (Seurat's
# cc.genes.updated.2019; mouse Cenpu, Pimreg, Jpt1). References differ, and one can
# mix them: Kang 2018 has CENPU but HN1. That list's MCM2->MCM7, RPA2->POLR1B and
# BRIP1->MRPL36 are alias mix-ups, not renames, and are not followed.
CELL_CYCLE_ALIASES = {"MLF1IP": ("CENPU",), "FAM64A": ("PIMREG",), "HN1": ("JPT1",)}


def _lognorm_scoring_names(adata: ad.AnnData, what: str) -> tuple[bool, list[str]]:
    """``use_raw`` for scanpy's gene scoring and that matrix's gene names, on
    log-normalised expression: ``.raw`` after scaling, else X. Never counts: a fresh
    upload can hold counts in ``.raw`` beside a log-normalised X (Kang), and
    scanpy's default would score the counts."""
    if adata.raw is not None and not looks_like_counts(adata.raw.X):
        return True, list(adata.raw.var_names)
    if not looks_like_counts(adata.X):
        return False, list(adata.var_names)
    raise ValueError(
        f"{what} needs log-normalised expression: run normalization and log transform first."
    )


def _run_cell_cycle(adata: ad.AnnData, params: PipelineParams) -> dict[str, Any]:
    """Score cell cycle phases (Tirosh 2016 genes), for human or mouse gene names."""
    try:
        species = detect_species(adata).species
        # The dataset's own spelling: scanpy drops a gene whose name differs only
        # in case, so a mouse dataset had none of the (human) list.
        use_raw, names = _lognorm_scoring_names(adata, "Cell cycle scoring")
        s_present = match_genes(CELL_CYCLE_S_GENES, names, CELL_CYCLE_ALIASES)
        g2m_present = match_genes(CELL_CYCLE_G2M_GENES, names, CELL_CYCLE_ALIASES)

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

        backend = _LARGE.get()
        if backend is not None:
            backend.score_genes_cell_cycle(adata, s_genes=s_present, g2m_genes=g2m_present)
        else:
            sc.tl.score_genes_cell_cycle(
                adata, s_genes=s_present, g2m_genes=g2m_present, use_raw=use_raw
            )
        logger.info("Cell cycle scored (%s; %d S genes, %d G2M genes)",
                    species, len(s_present), len(g2m_present))
    except Exception as e:
        raise ValueError(f"Cell cycle scoring failed: {e}") from e
    # Recorded with the step: how many phase calls rest on scores that are noise.
    return {"species": species, "n_s_genes": len(s_present), "n_g2m_genes": len(g2m_present),
            **(near_zero_phase_calls(adata.obs) or {})}


def _run_dissociation_stress(adata: ad.AnnData, params: PipelineParams) -> dict[str, Any]:
    """Score each cell for the dissociation-stress signature (see core/stress.py).

    On log-normalised expression: ``.raw`` after scaling, else X. Skipped, not
    failed, when too few signature genes are present."""
    use_raw, names = _lognorm_scoring_names(adata, "The dissociation-stress score")
    genes = stress.signature_genes(names)
    if len(genes) < stress.MIN_GENES:
        raise StepSkipped(
            f"Only {len(genes)} of the {len(stress.STRESS_GENES)} dissociation-stress genes "
            f"are in this dataset (at least {stress.MIN_GENES} are needed), so no score "
            "was written."
        )
    n_pool = len(names) - len(genes)
    _score_genes(
        adata, genes, score_name=stress.SCORE_COLUMN, use_raw=use_raw,
        ctrl_size=max(1, min(50, n_pool)), random_state=0,
    )
    species = detect_species(adata).species
    logger.info("Dissociation stress scored (%s; %d genes) -> obs['%s']",
                species, len(genes), stress.SCORE_COLUMN)
    return {"species": species, "n_genes": len(genes), "genes": genes}


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
) -> dict[str, Any]:
    """Reference-based annotation with CellTypist, per-cluster consensus (majority voting).

    Returns what the run used (model, species, gene overlap) for provenance; the
    same record is kept in uns for the per-label summary (core/celltypist_fit.py)."""
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
    # layers['counts'] first: after scaling, .raw holds log-normalised values.
    backend = _LARGE.get()
    inp = None
    if backend is not None:  # the counts' genes; the input is built a block of cells at a time
        input_genes = pd.Index(adata.var_names)
    else:
        if "counts" in adata.layers:
            counts, var_names = adata.layers["counts"], list(adata.var_names)
        elif adata.raw is not None:
            counts, var_names = adata.raw.X, list(adata.raw.var_names)
        else:
            counts, var_names = adata.X, list(adata.var_names)
        inp = ad.AnnData(X=counts.copy() if hasattr(counts, "copy") else counts)
        inp.obs_names = adata.obs_names
        inp.var_names = var_names
        sc.pp.normalize_total(inp, target_sum=1e4)
        sc.pp.log1p(inp)
        input_genes = inp.var_names

    # How well the model suits the data (core/celltypist_fit.py): its species
    # against the dataset's, and the share of its genes the dataset has.
    from scview.core import celltypist_fit
    from scview.core.species import detect_species

    dataset_species = detect_species(adata).species
    model: Any = model_name
    run: dict[str, Any] = {"model": model_name, "dataset_species": dataset_species,
                           "model_species": celltypist_fit.model_species(model_name)}
    try:
        model = models.Model.load(model=model_name)
        details = str((getattr(model, "description", None) or {}).get("details", ""))
        run["model_species"] = celltypist_fit.model_species(model_name, details)
        run.update(celltypist_fit.gene_overlap(model.features, input_genes))
    except Exception as e:  # no local copy: CellTypist's own load below says why
        logger.warning("CellTypist model %s not loaded for the overlap check: %s", model_name, e)
    if run.get("n_shared_genes") == 0:
        warning = celltypist_fit.species_warning(run["model_species"], dataset_species)
        raise ValueError(
            f"None of the {run['n_model_genes']:,} genes of {model_name} are in this dataset. "
            + (warning or "CellTypist needs gene symbols matching the model's (not Ensembl ids).")
        )
    if backend is not None:  # per block of cells, then the vote over the clustering
        pred = backend.celltypist(adata, model, groupby)
    else:
        annotate_kwargs: dict[str, Any] = {"model": model, "majority_voting": True}
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
        # The probabilities of the target's labels, for the annotation confidence
        # check to score cells by the reference model (core/annotation_confidence.py).
        from scview.core.annotation_confidence import store_reference_probabilities

        store_reference_probabilities(adata, target, pred.probability_matrix, model_name)
    else:
        adata.obs.drop(columns=[f"{target}_confidence"], errors="ignore", inplace=True)
    celltypist_fit.record_run(adata, target, run)
    logger.info(
        "CellTypist: annotated %d cells -> obs['%s'] (model=%s, groupby=%s, %d types, "
        "gene overlap %s)",
        adata.n_obs, target, model_name, groupby or "celltypist-internal",
        int(adata.obs[target].nunique()), run.get("gene_overlap"),
    )
    return run


def _top_markers_per_group(adata: ad.AnnData, groupby: str, n: int = 25) -> dict[str, list[str]]:
    """Each cluster's markers for the LLM, best first (``core/marker_selection.py``:
    expressed in >= 20% of the cluster, adjusted p < 0.05, by fold change, no
    ribosomal/mito genes).

    Computes wilcoxon markers under the column's own key first when none are
    stored: the default rank_genes_groups follows the clustering, and the Marker
    Genes panel reads it."""
    from scview.core.marker_selection import group_markers
    from scview.core.ora_background import resolve_markers_key

    if resolve_markers_key(adata.uns, groupby) is None:
        if adata.obs[groupby].dtype.name != "category":
            adata.obs[groupby] = adata.obs[groupby].astype(str).astype("category")
        _wilcoxon(adata, groupby, key_added=markers_key(groupby))
    return group_markers(adata, groupby, n=n)


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
        labels = [ln.strip().split(":", 1)[-1].strip() for ln in raw.splitlines()
                  if ln.strip() and not ln.strip().lower().startswith("tissue:")]
        out = {str(c): lab for c, lab in zip(clusters, labels)}
    return out


def _annotate_llm(adata: ad.AnnData, params: PipelineParams, groupby: str, target: str) -> None:
    """Tissue-agnostic annotation: name each cluster's cell type from its top markers via an LLM.

    No reference model to choose — works for any tissue (GPTCelltype-style). Best treated as a
    reviewable first pass; pair with CellTypist when a matching reference model exists.
    """
    if not groupby:
        raise ValueError("LLM-from-markers annotation needs a clustering; run clustering first.")
    # Ribosomal and mitochondrial genes say nothing about cell type, yet they can
    # fill a large cluster's whole top list (Kang's main T-cell cluster: 15 of 15
    # were RPS/RPL, and the model called it "Ribosomal cells"); score order put
    # FTL, CD63, ANXA5 first in CD14 monocytes. See core/marker_selection.py.
    marker_map = _top_markers_per_group(adata, groupby)
    if not marker_map:
        raise ValueError(f"No marker genes available for '{groupby}'.")

    from scview.config import get_settings

    settings = get_settings()
    api_key = settings.DEEPINFRA_API_KEY
    if not api_key:
        raise ValueError("LLM annotation requires DEEPINFRA_API_KEY to be set.")
    model = params.annotation_llm_model or settings.RAG_CHAT_MODEL

    from openai import OpenAI

    hint = params.annotation_tissue.strip()
    listing = "\n".join(f"{g}: {', '.join(genes)}" for g, genes in marker_map.items())
    prompt = _annotation_prompt(listing, hint)
    client = OpenAI(api_key=api_key, base_url="https://api.deepinfra.com/v1/openai")
    resp = client.chat.completions.create(
        model=model,
        **chat_extra(model),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=800,
    )
    raw = resp.choices[0].message.content or ""
    tissue = _parse_tissue_line(raw) or hint
    labels = _parse_cluster_labels(raw, list(marker_map.keys()))

    mapping = {str(g): _label_case(labels.get(str(g), "Unknown")) for g in marker_map}
    adata.obs[target] = (
        adata.obs[groupby].astype(str).map(lambda g: mapping.get(g, "Unknown")).astype("category")
    )
    adata.uns[f"{target}_llm_mapping"] = mapping
    if tissue:
        adata.uns[f"{target}_llm_tissue"] = tissue
    logger.info(
        "LLM annotation: %d clusters -> obs['%s'] (model=%s, groupby=%s, tissue=%s)",
        len(mapping), target, model, groupby, tissue or "?",
    )


def _annotation_prompt(listing: str, tissue_hint: str = "") -> str:
    """The marker-based annotation prompt.

    The tissue comes first, from every cluster together (or the user's hint), so
    cells are named as they occur there. Without it, blood monocytes (S100A8, LYZ)
    were called "macrophages" and plasmacytoid DCs (JCHAIN, IRF7) "plasma cells".
    """
    tissue = (
        f"These cells come from {tissue_hint}. Start with the line 'tissue: {tissue_hint}'."
        if tissue_hint else
        "First infer the tissue or sample type from all the clusters together, and write it "
        "as the first line, 'tissue: <tissue>'."
    )
    return (
        "You are an expert in single-cell RNA-seq. " + tissue + " Then, for each cluster, "
        "identify the most likely cell type from its top marker genes, naming cells as they "
        "occur in that tissue: in blood, CD14/LYZ/S100A8 cells are monocytes, not macrophages. "
        "Decide by lineage-defining genes when markers overlap: CD3D/CD3E/CD8A mean T cells "
        "even with NKG7/GZMB; plasmacytoid dendritic cells (LILRA4, IL3RA, CLEC4C, TCF4, IRF7) "
        "are not plasma cells (MZB1, XBP1, SDC1 with high immunoglobulin genes), though both "
        "express JCHAIN. If a cluster shows two lineages, name the mixture (e.g. "
        "'NK cells / CD8+ T cells'). Output one line per cluster as 'cluster_id: cell type', "
        "with concise standard cell-type names, and nothing else.\n\n"
        "Clusters and marker genes:\n" + listing
    )


def _parse_tissue_line(raw: str) -> str:
    for line in raw.splitlines():
        head, _, rest = line.strip().lstrip("-*# ").partition(":")
        if head.strip().lower() == "tissue" and rest.strip():
            return rest.strip().rstrip(".")
    return ""


def _label_case(label: str) -> str:
    """Sentence case for labels the model wrote in lower case ("macrophages" and
    "Macrophages" came back on different runs); "pDC" and "CD8+ T" are kept."""
    return label[:1].upper() + label[1:] if label[:1].islower() and not label[1:2].isupper() else label


# Curated marker sets for the offline 'marker_score' method (common PBMC/immune plus
# a few stromal/epithelial types): the fallback when the marker databases
# (core/marker_db.py) are not downloaded or their download is turned off.
_MARKER_SETS: dict[str, list[str]] = {
    "T cell": ["CD3D", "CD3E", "CD3G", "TRAC"],
    "CD4+ T cell": ["CD4", "IL7R"],
    "CD8+ T cell": ["CD8A", "CD8B", "GZMK"],
    "B cell": ["CD79A", "CD79B", "MS4A1", "CD19"],
    "Plasma cell": ["MZB1", "IGHG1", "JCHAIN", "XBP1"],
    "NK cell": ["NKG7", "GNLY", "KLRD1", "NCAM1"],
    "Monocyte": ["CD14", "LYZ", "S100A8", "S100A9", "FCN1"],
    "Macrophage": ["CD68", "C1QA", "C1QB", "MARCO"],
    "Dendritic cell": ["FCER1A", "CD1C", "CLEC9A", "LILRA4"],
    "Megakaryocyte": ["PPBP", "PF4", "ITGA2B"],
    "Erythrocyte": ["HBB", "HBA1", "HBA2", "GYPA"],
    "Epithelial cell": ["EPCAM", "KRT8", "KRT18"],
    "Fibroblast": ["COL1A1", "DCN", "LUM", "PDGFRB"],
    "Endothelial cell": ["PECAM1", "VWF", "CLDN5"],
}


def _marker_db_or_none():
    """The merged marker database (fetched on first use when allowed), or None."""
    from scview.core import marker_db

    try:
        return marker_db.get_db()
    except Exception as exc:  # not downloaded, download off or failed: use the built-in sets
        logger.info("marker_score: marker databases unavailable (%s); using the built-in sets", exc)
        return None


def _annotate_marker_score(
    adata: ad.AnnData, params: PipelineParams, groupby: str, target: str
) -> dict[str, Any]:
    """Offline annotation, one cell type per cluster. Deterministic, no LLM.

    With the marker databases (PanglaoDB + CellMarker 3.0 curated rows, or 2.0 + ACT; see
    core/marker_db.py) each cluster gets the database type its top 25 markers
    over-represent most (hypergeometric against the full database set), "Unknown"
    when no type shares a marker. Without them, the 14 curated sets above are
    scored per cell and each cluster gets the highest mean. Which one ran is
    recorded in ``uns[f"{target}_markerscore_source"]`` and returned for the
    provenance."""
    if not groupby:
        raise ValueError("marker_score annotation needs a clustering; run clustering first.")
    from scview.core import marker_db

    # "builtin" is the user's choice, not a fallback: on PBMC the 14 sets named 8 of
    # Kang's 11 Leiden clusters right against the databases' 5 (fine T subtypes,
    # CD14 Mono), while the databases win elsewhere (paul15 11 of 19 against 2).
    builtin = (params.annotation_marker_sets or "databases").lower() == "builtin"
    db = None if builtin else _marker_db_or_none()
    if db is not None:
        # In a large-dataset run the database check reads stored markers only (it would
        # rank the matrix otherwise): the clustering's, for the check alone.
        key = markers_key(groupby)
        temporary = (_LARGE.get() is not None and key not in adata.uns
                     and adata.obs[groupby].dtype.name == "category")
        if temporary:
            _wilcoxon(adata, groupby, key_added=key)
        try:
            mapping, info = marker_db.annotate(adata, groupby, db=db)
        finally:
            if temporary:
                adata.uns.pop(key, None)
    else:
        mapping = _marker_set_mapping(adata, groupby)
        info = {"marker_source": "built-in", "n_types": len(_MARKER_SETS),
                "chosen": builtin}
    adata.obs[target] = (
        adata.obs[groupby].astype(str).map(lambda g: mapping.get(g, "Unknown")).astype("category")
    )
    adata.uns[f"{target}_markerscore_mapping"] = mapping
    adata.uns[f"{target}_markerscore_source"] = info
    logger.info("marker_score: %d clusters -> obs['%s'] (%s)", len(mapping), target, info)
    return info


def _marker_set_mapping(adata: ad.AnnData, groupby: str) -> dict[str, str]:
    """Score the built-in marker sets per cell; each cluster -> the set with the top mean."""
    upper = {str(g).upper(): str(g) for g in adata.var_names}
    score_cols: dict[str, str] = {}
    try:
        for ct, genes in _MARKER_SETS.items():
            present = [upper[g.upper()] for g in genes if g.upper() in upper]
            if len(present) < 2:
                continue
            key = f"_ms__{ct}"
            _score_genes(adata, present, score_name=key, ctrl_size=min(50, adata.n_vars - 1))
            score_cols[ct] = key
        if not score_cols:
            raise ValueError("None of the marker-set genes are present in this dataset.")
        inv = {v: k for k, v in score_cols.items()}
        means = adata.obs.groupby(groupby, observed=True)[list(score_cols.values())].mean()
        return {str(idx): inv[means.loc[idx].idxmax()] for idx in means.index}
    finally:
        adata.obs.drop(columns=[k for k in score_cols.values() if k in adata.obs.columns], inplace=True)


def _run_cell_type_annotation(
    adata: ad.AnnData, params: PipelineParams
) -> dict[str, Any] | None:
    """Annotate cell types per cluster, writing a reviewable obs[target] column.

    Methods: 'llm' (LLM-from-markers, tissue-agnostic, default), 'celltypist'
    (reference-based, tissue-specific model), and 'marker_score' (offline curated
    marker scoring). See docs/CELLTYPE_ANNOTATION.md.
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

    fit: dict[str, Any] | None = None
    if method == "celltypist":
        fit = _annotate_celltypist(adata, params, groupby, target)
    elif method == "llm":
        _annotate_llm(adata, params, groupby, target)
    elif method == "marker_score":
        fit = _annotate_marker_score(adata, params, groupby, target)
    else:
        raise ValueError(
            f"Unknown annotation method '{method}' (expected celltypist | llm | marker_score)."
        )
    if method != "celltypist":
        from scview.core.annotation_confidence import drop_reference_probabilities
        from scview.core.celltypist_fit import drop_run

        drop_reference_probabilities(adata, target)  # an earlier CellTypist run's
        drop_run(adata, target)

    # Marker genes for the new column, so enrichment and the marker plots can use it
    # straight away. The default rank_genes_groups stays on the clustering.
    _markers_for_column(adata, target)
    # Recorded with the step: the model's species and gene overlap (CellTypist only).
    return {k: v for k, v in (fit or {}).items() if k != "model"} or None


def drop_enrichment_cache(adata: ad.AnnData, col: str) -> None:
    """Forget cached enrichment for a column whose markers or groups changed.

    The cache is keyed by column and group name only, so after re-annotation or
    new markers a surviving label would otherwise be served its old result.
    """
    for prefix in (enrichment_prefix(col), meta_prefix(col)):
        for k in [k for k in adata.uns if isinstance(k, str) and k.startswith(prefix)]:
            del adata.uns[k]


def _markers_for_column(adata: ad.AnnData, col: str) -> None:
    """Store wilcoxon markers for one obs column under rank_genes_groups__{col}.
    Best effort: a failure is logged and leaves the column without markers."""
    if col not in adata.obs.columns or not 2 <= adata.obs[col].nunique() <= 100:
        return
    try:
        if adata.obs[col].dtype.name != "category":
            adata.obs[col] = adata.obs[col].astype(str).astype("category")
        _wilcoxon(adata, col, key_added=markers_key(col))
        _attach_specificity(adata, adata.uns[markers_key(col)], col)
        drop_enrichment_cache(adata, col)
    except Exception as exc:
        logger.warning("Markers for '%s' failed: %s", col, exc)


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
    "uncorrected_embedding": _run_uncorrected_embedding,
    "batch_correction": _run_batch_correction,
    "neighbors": _run_neighbors,
    "clustering": _run_clustering,
    "embeddings": _run_embeddings,
    "marker_genes": _run_marker_genes,
    "enrichment": _run_enrichment,
    "cell_cycle": _run_cell_cycle,
    "dissociation_stress": _run_dissociation_stress,
    "cell_type_annotation": _run_cell_type_annotation,
}

# Steps that return a new AnnData object (instead of modifying in-place), alone or
# with a dict of what to record (see _new_adata)
_STEPS_RETURNING_ADATA = {"filtering", "reset_to_counts"}


def _new_adata(step_name: str, ret: Any) -> tuple[ad.AnnData | None, Any]:
    """(the step's new AnnData or None, what to record) from a runner's return."""
    if step_name not in _STEPS_RETURNING_ADATA:
        return None, ret
    if isinstance(ret, tuple) and len(ret) == 2 and isinstance(ret[0], ad.AnnData):
        return ret
    return (ret, None) if isinstance(ret, ad.AnnData) else (None, ret)


# step -> (human tool label, PipelineParams fields worth recording)
_STEP_PROVENANCE: dict[str, tuple[str, tuple[str, ...]]] = {
    "reset_to_counts": ("scview.reset_to_counts", ()),
    "qc_metrics": ("scanpy.pp.calculate_qc_metrics", ()),
    "doublet_detection": ("scanpy.pp.scrublet (per capture channel)",
                          ("doublet_method", "expected_doublet_rate", "doublet_channel_key")),
    # The mode's own params (fixed cut-offs, or k and per-sample thresholds) come
    # back from the step itself.
    "filtering": ("scview.qc_filter + scanpy.pp.filter_genes",
                  ("qc_threshold_mode", "min_cells", "max_pct_mt", "drop_doublets")),
    "normalization": ("scanpy.pp.normalize_total", ("target_sum", "normalization_method")),
    "log_transform": ("scanpy.pp.log1p", ()),
    "highly_variable_genes": ("scanpy.pp.highly_variable_genes",
                              ("n_top_genes", "hvg_flavor", "hvg_batch_key",
                               "normalization_method")),
    "scaling": ("scanpy.pp.scale", ("normalization_method",)),
    "pca": ("scanpy.pp.pca", ("n_comps",)),
    "uncorrected_embedding": ("scanpy.tl.umap (uncorrected PCA)", ("n_neighbors", "umap_min_dist")),
    "batch_correction": ("scview.batch_correction", ("batch_key", "batch_method")),
    "neighbors": ("scanpy.pp.neighbors", ("n_neighbors", "n_comps")),
    "clustering": ("scanpy.tl.leiden/louvain", ("clustering_method", "clustering_resolution")),
    "embeddings": ("scanpy.tl.umap", ("umap_min_dist",)),
    "marker_genes": ("scanpy.tl.rank_genes_groups", ("marker_columns",)),
    "enrichment": ("scview.enrichment",
                   ("enrichment_columns", "enrichment_n_genes", "enrichment_collections")),
    "cell_cycle": ("scanpy.tl.score_genes_cell_cycle", ()),
    "dissociation_stress": ("scanpy.tl.score_genes (van den Brink 2017 dissociation genes)", ()),
    "cell_type_annotation": ("scview.cell_type_annotation",
                             ("annotation_method", "celltypist_model", "annotation_llm_model",
                              "annotation_tissue", "annotation_marker_sets", "annotation_groupby",
                              "annotation_target")),
}


# What those steps run instead when Pearson residuals are chosen.
_PEARSON_TOOLS = {
    "highly_variable_genes": "scanpy.experimental.pp.highly_variable_genes (pearson_residuals)",
    "scaling": "analytic Pearson residuals (scanpy.experimental.pp.normalize_pearson_residuals)",
}


def _record_pipeline_step(
    adata: ad.AnnData, step_name: str, params: PipelineParams, ret: Any = None,
    resources: dict[str, Any] | None = None,
) -> None:
    """Append a provenance entry for a completed step. Never raises.

    A runner that returns a dict adds it to the recorded params: what the step
    found rather than was told (the species, how many signature genes it had).
    ``resources`` (the step's seconds and sampled peak memory, from a watched
    run) is kept beside the params, not in them, so replay ignores it."""
    tool, keys = _STEP_PROVENANCE.get(step_name, (f"scview.{step_name}", ()))
    if params.normalization_method == "pearson_residuals":
        tool = _PEARSON_TOOLS.get(step_name, tool)
    try:
        recorded = {k: getattr(params, k) for k in keys}
        if isinstance(ret, dict):
            recorded.update(ret)
        provenance.record_step(adata, step=step_name, tool=tool, params=recorded,
                               resources=resources)
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
        if ran & {"highly_variable_genes", "scaling"}:
            # What X and the variable genes came from: z-scored log values or residuals.
            current["normalization_method"] = params.normalization_method
        if "pca" in ran or "X_pca" in adata.obsm:
            current["pca"] = True
        embs = sorted(k for k in adata.obsm.keys() if k.startswith("X_"))
        if embs:
            current["embeddings"] = embs
        active = adata.uns.get("scview_active_clustering")
        if active and "clustering" in ran:
            current["clustering"] = {
                "column": active,
                "method": params.clustering_method,
                "resolution": params.clustering_resolution,
            }
        elif active:
            # Clustering did not run, so params.clustering_* are just defaults: keep
            # what the run that made the column recorded.
            recorded = provenance.read_provenance(adata)["current"].get("clustering") or {}
            if recorded.get("column") != active:
                current["clustering"] = {"column": active}
        markers_for = sorted(
            c for c in (markers_column(k) for k in adata.uns) if c is not None
        )
        if markers_for:
            current["markers_for"] = markers_for
        enr_for = sorted(
            {c for c in (enrichment_column(k) for k in adata.uns) if c is not None}
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


def _step_scope(control: RunControl | None, step: str):
    """Times and watches the step when the run is watched (``RunControl.step``)."""
    return control.step(step) if control is not None else contextlib.nullcontext()


def _step_resources(control: RunControl | None, step: str) -> dict[str, Any] | None:
    return control.watchdog.steps.get(step) if control is not None else None


def _unknown_requested(steps: list[str]) -> list[str]:
    unknown = list(dict.fromkeys(s for s in steps if s not in ALL_STEPS))
    if unknown:
        logger.warning("Unknown steps ignored: %s", unknown)
    return unknown


def _halted(result: PipelineResult, halt: tuple[str, str]) -> None:
    result.status, result.stop_reason = halt
    logger.info("Run %s: %s", result.status, result.stop_reason)


def _cancelled_during(result: PipelineResult, step: str) -> None:
    _halted(result, ("cancelled", cancelled_reason("during", step)))


def _finish_run(
    adata: ad.AnnData,
    params: PipelineParams,
    result: PipelineResult,
    t0: float,
    output_path: str | None,
    control: RunControl | None,
) -> None:
    """Finalise and save a run that reached its end. A cancelled run, or one the
    pre-step check stopped, saves nothing: a reprocess or re-filter stopped halfway
    would otherwise replace the dataset with a half-processed one (reset to counts,
    not yet normalised). The dataset reopens as it was before the run.

    The file is written beside the derived file and then swapped in, so a backend
    killed mid-write (out of memory) leaves the previous file whole."""
    if result.status == "complete" and control is not None and control.cancelled:
        # Cancelled after the last step ended but before the save.
        _halted(result, ("cancelled", cancelled_reason("before", "saving")))
    save = result.status == "complete"
    if save:
        _finalize_current(adata, params, result.steps_run)
    result.elapsed_seconds = round(time.time() - t0, 2)
    write_s = None
    if output_path and save:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        tw = time.time()
        partial = output_path + ".partial"
        try:
            adata.write_h5ad(partial, compression="gzip")
            os.replace(partial, output_path)
        except BaseException:
            Path(partial).unlink(missing_ok=True)
            raise
        write_s = round(time.time() - tw, 2)
        result.output_path = output_path
        logger.info("Saved processed dataset to %s", output_path)
    if control is not None:
        result.resources = {**control.watchdog.summary(), "write_s": write_s}


def run_pipeline(
    adata: ad.AnnData,
    steps: list[str],
    params: PipelineParams | None = None,
    output_path: str | None = None,
    control: RunControl | None = None,
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
    control
        Checked before each step (cancel, the pre-step hook) and times and
        watches each step (``core/run_watchdog.py``). None: neither.

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
    result.unknown_steps = _unknown_requested(steps)

    for step_name in ordered_steps:
        runner = _STEP_RUNNERS.get(step_name)
        if runner is None:
            result.steps_skipped.append(step_name)
            continue
        halt = control.halt_before(step_name, adata, params) if control is not None else None
        if halt is not None:
            _halted(result, halt)
            break

        try:
            logger.info("Running step: %s", step_name)
            with _step_scope(control, step_name):
                ret = runner(adata, params)
            # Some steps return a new AnnData object
            new, ret = _new_adata(step_name, ret)
            if new is not None:
                provenance.carry(adata, new)  # keep history across the new object
                adata = new
            result.steps_run.append(step_name)
            _record_pipeline_step(adata, step_name, params, ret,
                                  _step_resources(control, step_name))
        except StepSkipped as e:
            logger.info("Step '%s' skipped: %s", step_name, e)
            result.notes[step_name] = str(e)
        except Exception as e:
            if control is not None and control.cancelled:  # e.g. its Leiden child was ended
                _cancelled_during(result, step_name)
                break
            logger.error("Step '%s' failed: %s", step_name, e)
            result.errors[step_name] = str(e)

    _finish_run(adata, params, result, t0, output_path, control)
    return adata, result


def run_pipeline_streamed(
    adata: ad.AnnData,
    steps: list[str],
    params: PipelineParams | None = None,
    output_path: str | None = None,
    control: RunControl | None = None,
) -> Generator[tuple[str, dict[str, Any]], None, tuple[ad.AnnData, PipelineResult]]:
    """Run preprocessing steps, yielding progress events after each step.

    ``control`` as in ``run_pipeline``; a streamed run also stops between the
    substeps of marker genes and enrichment when cancelled.

    Yields
    ------
    Tuples of (event_type, event_data) where event_type is one of:
      "step_start", "step_done", "step_error", "substep", "complete". The
    "complete" data is the result, with ``status`` complete | stopped | cancelled.
    """
    if params is None:
        params = PipelineParams()

    if adata.obs.index.name is not None and adata.obs.index.name in adata.obs.columns:
        adata.obs.index.name = None

    result = PipelineResult()
    t0 = time.time()

    ordered_steps = [s for s in ALL_STEPS if s in steps]
    total = len(ordered_steps)
    result.unknown_steps = _unknown_requested(steps)

    # Runners that accept a substep progress callback (3rd arg)
    _SUBSTEP_RUNNERS = {"marker_genes", "enrichment"}
    _SENTINEL = object()  # marks thread completion

    for idx, step_name in enumerate(ordered_steps):
        runner = _STEP_RUNNERS.get(step_name)
        if runner is None:
            result.steps_skipped.append(step_name)
            continue
        halt = control.halt_before(step_name, adata, params) if control is not None else None
        if halt is not None:
            _halted(result, halt)
            break

        yield ("step_start", {"step": step_name, "index": idx, "total": total})
        step_t0 = time.time()

        try:
            with _step_scope(control, step_name):
                if step_name in _SUBSTEP_RUNNERS:
                    # Run in a thread so substep events can stream in real-time
                    q: queue.Queue = queue.Queue()
                    exc_holder: list[Exception] = []
                    ret_holder: list[Any] = [None]

                    def _substep_cb(msg: str, current: int, sub_total: int) -> None:
                        if control is not None and control.cancelled:
                            raise RunCancelled()  # ends the step between substeps
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

                    # Drain substep events from the queue while thread runs. If the
                    # consumer stops at a substep, still wait for the step's thread:
                    # whoever runs this generator owns the dataset until it returns.
                    try:
                        while True:
                            item = q.get()
                            if item is _SENTINEL:
                                break
                            yield ("substep", {"step": step_name, **item})
                    finally:
                        t.join()
                    if exc_holder:
                        raise exc_holder[0]
                    ret = ret_holder[0]
                else:
                    ret = runner(adata, params)

            new, ret = _new_adata(step_name, ret)
            if new is not None:
                provenance.carry(adata, new)  # keep history across the new object
                adata = new
            result.steps_run.append(step_name)
            _record_pipeline_step(adata, step_name, params, ret,
                                  _step_resources(control, step_name))
            elapsed = round(time.time() - step_t0, 2)
            yield ("step_done", {"step": step_name, "index": idx, "elapsed": elapsed})
        except StepSkipped as e:
            logger.info("Step '%s' skipped: %s", step_name, e)
            result.notes[step_name] = str(e)
            yield ("step_done", {"step": step_name, "index": idx, "skipped": True,
                                 "note": str(e), "elapsed": round(time.time() - step_t0, 2)})
        except Exception as e:
            if control is not None and control.cancelled:  # its Leiden child ended, or a substep
                _cancelled_during(result, step_name)
                break
            logger.error("Step '%s' failed: %s", step_name, e)
            result.errors[step_name] = str(e)
            yield ("step_error", {"step": step_name, "index": idx, "error": str(e)})

    _finish_run(adata, params, result, t0, output_path, control)
    yield ("complete", result.to_dict())

    return adata, result
