"""Data assessment endpoints -- detect preprocessing state, run pipeline, get suggestions."""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import aclosing
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from scview.core.sse import keepalive
from scview.config import Settings
from scview.dependencies import get_dataset_manager, get_settings_dep
from scview.core.dataset_manager import DatasetManager, DatasetMemoryError
from scview.core.assessor import assess_preprocessing, recorded_normalization_method
from scview.core import (
    cluster_qc, doublets, qc_filter, qc_groups, qc_report, sample_pseudobulk, sex_check,
)
from scview.core import large_mode, large_run, leiden
from scview.core import batch_methods, qc_states, scaling, spatial, targeted_panel
from scview.core.obs_writes import (
    dataset_edit, edit_room, editable, save_edit, uns_snapshot,
)
from scview.core.pipeline import PipelineParams, check_chunked, check_umap_3d
from scview.core.processing import cancel_run, iterate_exclusive, run_exclusive
from scview.core.run_watchdog import RunControl
from scview.core.step_names import unknown_steps, unknown_steps_message
from scview.core.llm_advisor import get_llm_suggestions, get_rule_based_suggestions
from scview.core import provenance
from scview.core.rerun import plan_rerun
from scview.core.species import detect_species

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class RunPipelineRequest(BaseModel):
    """Request body for the pipeline run endpoint."""
    steps: list[str]
    params: dict[str, Any] | None = None


class SuggestRequest(BaseModel):
    """Request body for the suggestions endpoint."""
    preprocessing_state: dict[str, Any]


class RerunRequest(BaseModel):
    """Re-run a step (and its downstream) with new parameters."""
    edited_step: str
    params: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# GET /datasets/{dataset_id}/assessment
# ---------------------------------------------------------------------------


@router.get("/datasets/{dataset_id}/assessment")
async def get_assessment(
    dataset_id: str,
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """Assess which preprocessing steps have been applied to the dataset."""
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")

    def _assess(adata) -> dict:
        # Not steps, so beside the step entries rather than among them.
        filtered = qc_filter.recorded_filtering(adata)
        return {**assess_preprocessing(adata).model_dump(),
                "species": detect_species(adata).model_dump(),
                "targeted_panel": targeted_panel.assessment(adata),
                "normalization_method": recorded_normalization_method(adata),
                # Re-filtering keeps the mode the dataset was filtered with (fixed
                # before MAD existed); None: never filtered by scView.
                "qc_threshold_mode": filtered["qc_threshold_mode"] if filtered else None,
                # Likewise the % mito rule (per-type bounds keep their type column).
                "qc_mito_mode": (filtered.get("qc_mito_mode") or "global") if filtered else None,
                "qc_mito_type_column": (filtered.get("qc_mito_type_column")
                                        if filtered and filtered.get("qc_mito_mode") == "per_type" else None),
                "qc_mito_type_floor": bool(filtered and filtered.get("qc_mito_mode") == "per_type"
                                           and filtered.get("qc_mito_type_floor"))}

    try:
        # Samples the matrix to judge its scale: ~1.6 s on Kang, so off the loop.
        return await asyncio.to_thread(_assess, adaptor.adata)
    except Exception as e:
        logger.error("Assessment failed for %s: %s", dataset_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Assessment failed: {e}")


# ---------------------------------------------------------------------------
# GET /datasets/{dataset_id}/cluster-qc
# ---------------------------------------------------------------------------


@router.get("/datasets/{dataset_id}/cluster-qc")
async def get_cluster_qc(
    dataset_id: str,
    groupby: str | None = None,
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """Per-cluster QC (size, median genes/counts/% mito/% ribo, doublets, % haemoglobin,
    cycling cells) with plain-language flags for clusters that stand out, erythroid
    notes, ambient haemoglobin and technical scores that track the clustering's
    leading PCs (deterministic, no LLM). ``groupby`` defaults to the active clustering."""
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")
    try:
        # Per-cell metrics may need a pass over the counts: keep it off the loop.
        report = await asyncio.to_thread(cluster_qc.for_adaptor, adaptor, groupby)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Cluster QC failed for %s: %s", dataset_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Cluster QC failed: {e}")
    return report.model_dump()


# ---------------------------------------------------------------------------
# GET /datasets/{dataset_id}/qc/by-group
# ---------------------------------------------------------------------------


@router.get("/datasets/{dataset_id}/qc/by-group")
async def get_qc_by_group(
    dataset_id: str,
    column: str = "",
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """QC distributions (genes, counts, % mito when measurable) per group of one
    categorical obs column, as quantiles for violins, plus a counts-vs-genes scatter
    sample carrying each point's group. With no ``column``, the donor/sample column
    is used, else a batch column, else the clustering (``reason`` says which)."""
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")
    if column and column not in adaptor.adata.obs.columns:
        raise HTTPException(status_code=404, detail=f"Column '{column}' not found.")

    def _run() -> dict:
        col, reason = (column, "requested") if column else qc_groups.default_grouping(adaptor.adata)
        if col is None:
            return qc_groups.no_grouping(adaptor.adata)
        return qc_groups.qc_by_group(adaptor.adata, col, adaptor.qc_per_cell(), reason)

    try:
        # Per-cell metrics may need a pass over the counts: keep it off the loop.
        return await asyncio.to_thread(_run)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("QC by group failed for %s: %s", dataset_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"QC by group failed: {e}")


# ---------------------------------------------------------------------------
# GET /datasets/{dataset_id}/qc/sex
# ---------------------------------------------------------------------------


@router.get("/datasets/{dataset_id}/qc/sex")
async def get_sex_check(
    dataset_id: str,
    column: str = "",
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """Each sample's sex from its expression (XIST vs Y-chromosome genes, core/sex_check.py):
    female / male / mixed / ambiguous / undetermined with the evidence, compared with any
    sex metadata column, plus warnings for mixed samples, mismatches (possible swaps) and
    sex x condition confounding. ``column`` names the samples (default: detected)."""
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")
    if column and column not in adaptor.adata.obs.columns:
        raise HTTPException(status_code=404, detail=f"Column '{column}' not found.")
    try:
        # Reads a few gene columns and every cell's total counts: off the loop.
        return await asyncio.to_thread(sex_check.sex_check, adaptor.adata, column or None)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Sex check failed for %s: %s", dataset_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Sex check failed: {e}")


# ---------------------------------------------------------------------------
# GET /datasets/{dataset_id}/qc/pseudobulk
# ---------------------------------------------------------------------------


@router.get("/datasets/{dataset_id}/qc/pseudobulk")
async def get_pseudobulk_qc(
    dataset_id: str,
    sample_column: str = "",
    split_by: str | None = None,
    celltype_column: str = "",
    celltype: str = "",
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """Sample-level QC on pseudobulk profiles (core/sample_pseudobulk.py): PCA of each
    sample's summed counts (per sample x condition when the samples carry several),
    each PC's association with the sample-level covariates (incl. the inferred sex),
    and outlier profiles. ``split_by`` absent: detected; empty: none. ``celltype``
    limits the profiles to one value of ``celltype_column``."""
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")
    adata = adaptor.adata

    def _run() -> dict:
        column = sample_column or qc_groups.sample_grouping(adata)[0]
        calls = None
        if column:
            try:
                sex = sex_check.sex_check(adata, column)
                calls = {g["name"]: g["call"] for g in sex["groups"]}
            except Exception as exc:  # the sex covariate is optional
                logger.debug("pseudobulk QC: no sex calls: %s", exc)
        return sample_pseudobulk.pseudobulk_qc(
            adata, sample_column or None, split_by, celltype_column=celltype_column or None,
            celltype=celltype or None, sex_calls=calls)

    try:
        # Sums the counts of every cell and runs a PCA: off the loop.
        return await asyncio.to_thread(_run)
    except sample_pseudobulk.SampleQcError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Pseudobulk QC failed for %s: %s", dataset_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Pseudobulk QC failed: {e}")


# ---------------------------------------------------------------------------
# GET /datasets/{dataset_id}/qc/filter-preview
# ---------------------------------------------------------------------------


@router.get("/datasets/{dataset_id}/qc/filter-preview")
async def get_filter_preview(
    dataset_id: str,
    min_genes: float | None = None,
    max_genes: float | None = None,
    min_counts: float | None = None,
    max_pct_mt: float | None = None,
    drop_doublets: bool | None = None,
    mode: str | None = None,
    mad_k: float | None = None,
    mad_k_mt: float | None = None,
    mad_upper: bool | None = None,
    mito_mode: str | None = None,
    mito_type_column: str = "",
    mito_type_floor: bool | None = None,
    sample_column: str = "",
    cluster_column: str = "",
    label_column: str = "",
    condition_column: str = "",
    doublet_source: str | None = None,
    drop_hash_negatives: bool | None = None,
    counts_floor: float | None = None,
    bin_mito: bool = False,
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """The cells the filtering step's thresholds would remove (core/qc_filter.py):
    overall, per threshold, per sample and per cluster, with warnings for a sample
    or cluster lost whole or more than half, each sample's QC medians and range
    flags, and in MAD mode (``mode``, the default) each sample's thresholds.
    ``sample_column`` names the samples (default: detected), as the step's
    ``qc_sample_column`` does. Thresholds left out take the pipeline's defaults.
    ``audit`` is who they remove per ``label_column`` group (default: an author
    cell-type column, else the clustering) x ``condition_column`` group (default:
    a condition-like column, else the samples), with a chi-square per group and
    flags; ``recorded`` the same for what the last filtering step removed.
    ``mito_mode`` "per_type" judges % mito by each type of ``mito_type_column``
    (default: as the step picks it) with ``mito_types`` the per-type table;
    ``mito_type_floor`` makes it never stricter than the global rule.
    ``doublet_source`` (auto | hashing | scrublet | classifier | consensus | either) is whose doublets
    ``drop_doublets`` drops, ``drop_hash_negatives`` drops cell hashing's
    negatives; ``doublets`` describes the choice (core/doublet_source.py).
    Small and fast at any cell count (one pass over cached per-cell values), so
    the QC plots ask again on every threshold change."""
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")
    params = PipelineParams()
    for key, value in (("min_genes", min_genes), ("max_genes", max_genes),
                       ("min_counts", min_counts), ("max_pct_mt", max_pct_mt),
                       ("drop_doublets", drop_doublets), ("qc_threshold_mode", mode),
                       ("qc_mad_k", mad_k), ("qc_mad_k_mt", mad_k_mt),
                       ("qc_mad_upper", mad_upper), ("qc_mito_mode", mito_mode),
                       ("qc_mito_type_column", mito_type_column or None),
                       ("qc_mito_type_floor", mito_type_floor),
                       ("qc_doublet_source", doublet_source), ("drop_hash_negatives", drop_hash_negatives),
                       ("qc_counts_floor", counts_floor)):
        if value is not None:
            setattr(params, key, value)
            params.explicit = (*params.explicit, key)
    params.qc_bin_mito = bool(bin_mito)
    # A targeted panel's counts floor, as its run applies it (core/targeted_panel.py).
    params = targeted_panel.for_run(adaptor.adata, params)
    thresholds = qc_filter.Thresholds.from_params(params)

    def _run() -> dict:
        t, cells = qc_filter.bin_rules(adaptor.adata, thresholds, adaptor.qc_per_cell(), params)
        out = qc_filter.preview(adaptor.adata, cells, t, sample_column or None, cluster_column or None,
                                label_column or None, condition_column or None)
        if t.max_pct_mt is None and thresholds.max_pct_mt is not None:
            out["bin_note"] = qc_filter.BIN_NOTE
        return out

    try:
        # Per-cell metrics may need a pass over the counts the first time: off the loop.
        return await asyncio.to_thread(_run)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Filter preview failed for %s: %s", dataset_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Filter preview failed: {e}")


# ---------------------------------------------------------------------------
# Doublet score threshold (core/doublets.py)
# ---------------------------------------------------------------------------


class DoubletThresholdRequest(BaseModel):
    """Call doublets above ``threshold``; ``None`` puts Scrublet's own call back."""
    threshold: float | None = None
    column: str = ""  # the channels for the per-channel counts (default: the last run's)


@router.get("/datasets/{dataset_id}/qc/doublets")
async def get_doublets(
    dataset_id: str,
    threshold: float | None = None,
    column: str = "",
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """The doublet-score histogram and how many cells ``threshold`` (default: the
    applied one, else Scrublet's when it ran once over every cell) calls doublets,
    overall and per capture channel, with each channel's expected rate. ``column``
    names the channels (an obs column, or ``__all__``: all cells as one; default: the
    last run's, else the doublet step's own choice). Without scores only the
    suggestion (the channels the step would use) is returned (``available`` false)."""
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")
    try:
        return await asyncio.to_thread(doublets.summary, adaptor.adata, threshold, column or None)
    except doublets.DoubletError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Doublet summary failed for %s: %s", dataset_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Doublet summary failed: {e}")


@router.post("/datasets/{dataset_id}/qc/doublets/threshold")
async def apply_doublet_threshold(
    dataset_id: str,
    body: DoubletThresholdRequest,
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """Write the calls at ``threshold`` into ``predicted_doublet`` (what the
    filtering step drops), keeping Scrublet's own call in
    ``predicted_doublet_scrublet``; save, and answer with the new summary."""
    async with dataset_edit(dataset_id) as held:
        adaptor = await dm.get_or_load_dataset(dataset_id)
        if adaptor is None:
            raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")
        output_path = str(dm.derived_h5ad_path(dataset_id, adaptor.h5ad_path))
        # A large dataset is edited where it is (its X stays on disk), written in place.
        backed = getattr(adaptor.adata, "isbacked", False) and not adaptor.is_large

        def _run():
            work = adaptor.adata.to_memory() if backed else adaptor.adata
            before = uns_snapshot(work)
            result = doublets.apply_threshold(work, body.threshold, body.column or None)
            save_edit(adaptor, work, output_path, before)
            return result

        try:
            result = await held.run(_run)
        except doublets.DoubletError as e:
            raise HTTPException(status_code=400, detail=str(e))  # raised before any change
        except Exception as e:
            dm.evict(dataset_id)  # the cached copy may hold calls that were never saved
            logger.error("Doublet threshold failed for %s: %s", dataset_id, e, exc_info=True)
            raise HTTPException(status_code=500, detail=f"Could not save the doublet calls: {e!s}")
        if backed:
            dm.evict(dataset_id)
            await dm.load_dataset(dataset_id)
        else:
            dm.saved_in_place(dataset_id, adaptor, output_path)
        return result


# ---------------------------------------------------------------------------
# POST /datasets/{dataset_id}/assessment/run
# ---------------------------------------------------------------------------


def _check_steps(steps: list[str]) -> None:
    """400 for no steps, 422 for names that are not steps (with the likely one meant)."""
    if not steps:
        raise HTTPException(status_code=400, detail="No steps specified.")
    unknown = unknown_steps(steps)
    if unknown:
        raise HTTPException(status_code=422, detail=unknown_steps_message(unknown))


def _run_params(raw: dict[str, Any] | None) -> PipelineParams:
    """PipelineParams from a request body; 422 for a large-dataset mode or a Leiden
    iteration count that is not one (before loading: a wrong value costs nothing)."""
    params = PipelineParams()
    for key, value in (raw or {}).items():
        if hasattr(params, key):
            setattr(params, key, value)
    params.explicit = tuple(k for k in (raw or {}) if k != "explicit" and hasattr(params, k))
    try:
        params.chunked = check_chunked(params.chunked)
        params.leiden_iterations = leiden.check_iterations(params.leiden_iterations)
        params.umap_3d = check_umap_3d(params.umap_3d)
        params.regress_out = scaling.check_regress_out(params.regress_out)
        params.batch_method = batch_methods.check_method(params.batch_method)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return params



@router.post("/datasets/{dataset_id}/assessment/run")
async def run_assessment_pipeline(
    dataset_id: str,
    body: RunPipelineRequest,
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """Run specified preprocessing steps on the dataset and save the result."""
    _check_steps(body.steps)  # before loading: a wrong name costs nothing
    params = _run_params(body.params)
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")
    control = RunControl(plan=(body.steps, params))
    # Drop this request's hold on the cached copy before the run: run_exclusive
    # reloads the dataset from disk afterwards, and a reference kept here would
    # hold the old AnnData through that reload (two copies: ovary at 6 GB was
    # OOM-killed reloading after a run the pre-step check had stopped).
    del adaptor

    def _run(adaptor):
        # Write processing output to the derived layer, never the original
        # upload or conversion — those stay immutable. In memory or in large-dataset
        # mode, as core/processing.py chose (control.mode).
        output_path = str(dm.derived_h5ad_path(dataset_id, adaptor.h5ad_path))
        return large_run.run_for(adaptor, body.steps, params, output_path, control)

    try:
        # In a worker thread with the dataset to itself; the cached copy is
        # dropped afterwards, so what is served is what is on disk.
        result = await run_exclusive(dm, dataset_id, _run, control)
        return result.to_dict()

    except (HTTPException, DatasetMemoryError):  # a 507 from reading a backed dataset in
        raise
    except Exception as e:
        logger.error("Pipeline run failed for %s: %s", dataset_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Pipeline run failed: {e}")


# ---------------------------------------------------------------------------
# POST /datasets/{dataset_id}/assessment/run-stream
# ---------------------------------------------------------------------------


@router.post("/datasets/{dataset_id}/assessment/run-stream")
async def run_assessment_pipeline_stream(
    dataset_id: str,
    body: RunPipelineRequest,
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """Run preprocessing steps with Server-Sent Events progress streaming."""
    _check_steps(body.steps)  # before loading: a wrong name costs nothing
    params = _run_params(body.params)
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")
    del adaptor  # as /assessment/run: no hold on the copy the run replaces
    control = RunControl(plan=(body.steps, params))

    def _events(adaptor):
        # Write processing output to the derived layer, never the original
        # upload or conversion — those stay immutable. In memory or in large-dataset
        # mode, as core/processing.py chose (control.mode).
        return large_run.run_streamed_for(
            adaptor, body.steps, params,
            str(dm.derived_h5ad_path(dataset_id, adaptor.h5ad_path)), control)

    async def event_generator():
        # Steps run in a worker thread with the dataset to itself. When the
        # client goes away mid-run (tab closed, proxy timeout) the run stops
        # after its current step without saving, and the cached copy, which may
        # hold unsaved steps, is dropped.
        try:
            # aclosing: stopping this generator must stop the run at once, not
            # whenever the inner generator is garbage-collected.
            async with aclosing(iterate_exclusive(dm, dataset_id, _events, control)) as events:
                async for event_type, event_data in events:
                    yield f"event: {event_type}\ndata: {json.dumps(event_data)}\n\n"
        except Exception as e:
            logger.error("Streamed pipeline failed for %s: %s", dataset_id, e, exc_info=True)
            detail = e.detail if isinstance(e, HTTPException) else str(e)
            yield f"event: error\ndata: {json.dumps({'error': detail})}\n\n"

    return StreamingResponse(
        keepalive(event_generator()),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# POST /datasets/{dataset_id}/assessment/suggest
# ---------------------------------------------------------------------------


@router.post("/datasets/{dataset_id}/assessment/suggest")
async def suggest_improvements(
    dataset_id: str,
    body: SuggestRequest,
    dm: DatasetManager = Depends(get_dataset_manager),
    settings: Settings = Depends(get_settings_dep),
):
    """Get AI-powered or rule-based suggestions for preprocessing."""
    # Verify the dataset exists and get summary info
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")

    dataset_summary = {
        "n_cells": adaptor.n_cells(),
        "n_genes": adaptor.n_genes(),
    }
    species = detect_species(adaptor.adata).species
    if species != "unknown":
        dataset_summary["species"] = species
    if spatial.targeted_panel(adaptor.adata):
        dataset_summary["targeted_panel"] = targeted_panel.n_genes(adaptor.adata)

    try:
        # Try LLM suggestions if API key is configured
        if settings.DEEPINFRA_API_KEY:
            try:
                result = await get_llm_suggestions(
                    preprocessing_state=body.preprocessing_state,
                    dataset_summary=dataset_summary,
                    api_key=settings.DEEPINFRA_API_KEY,
                    model=settings.RAG_CHAT_MODEL,
                )
                return result.model_dump()
            except Exception as e:
                logger.warning(
                    "LLM suggestions failed, falling back to rule-based: %s", e
                )

        # Fallback to rule-based suggestions
        result = get_rule_based_suggestions(
            preprocessing_state=body.preprocessing_state,
            dataset_summary=dataset_summary,
        )
        return result.model_dump()

    except Exception as e:
        logger.error("Suggestions failed for %s: %s", dataset_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Suggestions failed: {e}")


# ---------------------------------------------------------------------------
# Dependency-aware re-run ("edit & re-run from step k")
# ---------------------------------------------------------------------------


def _history_steps(adaptor) -> list[str]:
    return [h["step"] for h in provenance.read_provenance(adaptor.adata).get("history", [])]


@router.get("/datasets/{dataset_id}/rerun-plan")
async def get_rerun_plan(
    dataset_id: str,
    step: str,
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """Preview what re-running a step would recompute vs keep (no execution)."""
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")
    return plan_rerun(_history_steps(adaptor), step).model_dump()


@router.post("/datasets/{dataset_id}/rerun")
async def rerun_step(
    dataset_id: str,
    body: RerunRequest,
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """Re-run an in-place-rerunnable step (and its downstream) with new params,
    reusing existing upstream artifacts (PCA / neighbour graph)."""
    if unknown_steps([body.edited_step]):  # before loading: a wrong name costs nothing
        raise HTTPException(status_code=422, detail=unknown_steps_message([body.edited_step]))
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")

    plan = plan_rerun(_history_steps(adaptor), body.edited_step)
    if not plan.rerun_steps:
        raise HTTPException(status_code=400, detail=plan.message or "Nothing to re-run.")
    if plan.requires_reprocess:
        raise HTTPException(status_code=409, detail=plan.message)

    params = _run_params(body.params)
    control = RunControl(plan=(plan.rerun_steps, params))
    del adaptor  # as /assessment/run: no hold on the copy the run replaces

    def _run(adaptor):
        output_path = str(dm.derived_h5ad_path(dataset_id, adaptor.h5ad_path))
        return large_run.run_for(adaptor, plan.rerun_steps, params, output_path, control)

    try:
        # As /assessment/run: a worker thread, then serve what is on disk.
        result = await run_exclusive(dm, dataset_id, _run, control)
        return {"plan": plan.model_dump(), "result": result.to_dict()}
    except (HTTPException, DatasetMemoryError):  # a 507 from reading a backed dataset in
        raise
    except Exception as e:
        logger.error("Re-run failed for %s: %s", dataset_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Re-run failed: {e}")


# ---------------------------------------------------------------------------
# Narrative QC report ("Explain this data")
# ---------------------------------------------------------------------------


class QCReportRequest(BaseModel):
    """Request body for the QC report endpoints."""
    regenerate: bool = False  # skip the cached report and write a new one


async def _qc_facts(dataset_id: str, dm: DatasetManager) -> qc_report.QCFacts:
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")
    try:
        # Per-cell metrics may need a pass over the counts: keep it off the loop.
        return await asyncio.to_thread(qc_report.build_facts, adaptor)
    except Exception as e:
        logger.error("QC facts failed for %s: %s", dataset_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"QC report failed: {e}")


def _report_kwargs(dataset_id: str, dm: DatasetManager, settings: Settings) -> dict:
    return dict(
        api_key=settings.DEEPINFRA_API_KEY,
        model=settings.RAG_CHAT_MODEL,
        cache_file=qc_report.cache_path(dm.data_dir, dataset_id),
    )


@router.get("/datasets/{dataset_id}/qc-report")
async def get_qc_report(
    dataset_id: str,
    dm: DatasetManager = Depends(get_dataset_manager),
    settings: Settings = Depends(get_settings_dep),
):
    """The cached report for the dataset's current facts, or an empty report
    with the facts. Never calls the LLM, so opening the tab costs nothing;
    ``llm_available`` says whether a new report would be written or laid out."""
    facts = await _qc_facts(dataset_id, dm)
    return qc_report.cached_report(facts, **_report_kwargs(dataset_id, dm, settings)).model_dump()


@router.post("/datasets/{dataset_id}/qc-report")
async def post_qc_report(
    dataset_id: str,
    body: QCReportRequest,
    dm: DatasetManager = Depends(get_dataset_manager),
    settings: Settings = Depends(get_settings_dep),
):
    """Write (or replay from the cache) the QC report in four fixed sections,
    grounded only on the deterministic facts. Deterministic layout with no key."""
    facts = await _qc_facts(dataset_id, dm)
    report = await qc_report.build_report(
        facts, regenerate=body.regenerate, **_report_kwargs(dataset_id, dm, settings)
    )
    return report.model_dump()


@router.post("/datasets/{dataset_id}/qc-report-stream")
async def stream_qc_report(
    dataset_id: str,
    body: QCReportRequest,
    dm: DatasetManager = Depends(get_dataset_manager),
    settings: Settings = Depends(get_settings_dep),
):
    """Streaming variant (SSE): one ``facts`` event, ``delta`` text, then ``done``."""
    facts = await _qc_facts(dataset_id, dm)

    async def event_generator():
        try:
            async for ev in qc_report.report_events(
                facts, regenerate=body.regenerate, **_report_kwargs(dataset_id, dm, settings)
            ):
                yield f"data: {json.dumps(ev)}\n\n"
        except Exception as e:
            logger.error("QC report stream failed for %s: %s", dataset_id, e, exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

    return StreamingResponse(
        keepalive(event_generator()),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# GET /datasets/{dataset_id}/qc/mito-by-type
# ---------------------------------------------------------------------------


@router.get("/datasets/{dataset_id}/qc/mito-by-type")
async def get_mito_by_type(
    dataset_id: str,
    column: str = "",
    ceiling: float | None = None,
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """% mito per cell type (median, MAD, quantiles, n), each type's own bound
    (median + 3 MADs) and the cells the global ceiling (``ceiling``, else the
    recorded or default ``max_pct_mt``) would remove from each type vs its own
    bound; flags types the ceiling cuts hard. A dataset filtered with per-type
    bounds is read with them. ``column`` defaults to the column that filtering
    used, else the author's cell types, else the clustering. After filtering, the
    removed cells are added back from its record (a clustering made after it: each
    placed by expression, ``core/mito_refilter.py``)."""
    from scview.core import mito_by_type, mito_refilter

    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")
    if column and column not in adaptor.adata.obs.columns:
        raise HTTPException(status_code=404, detail=f"Column '{column}' not found.")

    def _assign(col: str):
        return mito_refilter.assigned_removed(dm, dataset_id, adaptor, col)

    try:
        # Obs + cached per-cell QC (a few ms), but the arrays may need a pass first,
        # and placing removed cells reads them from the original upload once.
        report = await asyncio.to_thread(mito_by_type.for_adaptor, adaptor, column or None, ceiling, _assign)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Mito by type failed for %s: %s", dataset_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Mito by type failed: {e}")
    return report.model_dump()


# ---------------------------------------------------------------------------
# QC states: low-RP / high-mito second state per cell type (core/qc_states.py)
# ---------------------------------------------------------------------------


@router.get("/datasets/{dataset_id}/qc/states")
async def get_qc_states(
    dataset_id: str,
    column: str = "",
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """Per group of ``column`` (default: a cell-type column, else the clustering)
    with 100+ cells: 1 vs 2 Gaussian components on the ribosomal-protein share of
    non-mito counts and % mito (RP share alone without MT- genes), the components,
    delta BIC, separation, the flagged state's share overall and per condition (or
    sample), and a seeded scatter sample (<= 2,000 cells per group)."""
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")
    if column and column not in adaptor.adata.obs.columns:
        raise HTTPException(status_code=404, detail=f"Column '{column}' not found.")
    try:
        result, _states = await asyncio.to_thread(qc_states.for_adaptor, adaptor, column or None)
        return result
    except qc_states.QcStatesError as e:
        raise HTTPException(status_code=e.status, detail=str(e))
    except Exception as e:
        logger.error("QC states failed for %s: %s", dataset_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"QC states failed: {e}")


class QcStateLabelRequest(BaseModel):
    column: str = ""  # the grouping; default as GET /qc/states picks it
    name: str = qc_states.DEFAULT_NAME


@router.post("/datasets/{dataset_id}/qc/states/label")
async def save_qc_state_label(
    dataset_id: str,
    body: QcStateLabelRequest,
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """Save every cell's QC state as categorical obs column ``name``: the flagged
    state of a two-state group (``low-RP high-mito``, ``low-RP`` without mito, or
    ``second state``), ``main`` for every other cell."""
    async with dataset_edit(dataset_id) as held:
        adaptor = await dm.get_or_load_dataset(dataset_id)
        if adaptor is None:
            raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")
        if body.column and body.column not in adaptor.adata.obs.columns:
            raise HTTPException(status_code=404, detail=f"Column '{body.column}' not found.")
        output_path = str(dm.derived_h5ad_path(dataset_id, adaptor.h5ad_path))
        backed = await edit_room(dm, dataset_id, adaptor)

        def _run():
            work = editable(adaptor)
            result, states = qc_states.for_adaptor(adaptor, body.column or None)
            if result.get("column") is None or states is None:
                raise qc_states.QcStatesError(result.get("note") or "No column to label by.")
            name = qc_states.check_name(work, result["column"], body.name)
            before = uns_snapshot(work)
            saved = qc_states.label(work, result, states, name)
            save_edit(adaptor, work, output_path, before)
            return saved

        try:
            saved = await held.run(_run)
        except qc_states.QcStatesError as e:
            raise HTTPException(status_code=e.status, detail=str(e))  # raised before any change
        except Exception as e:
            dm.evict(dataset_id)  # the cached copy may hold a column that was never saved
            logger.error("QC state label failed for %s: %s", dataset_id, e, exc_info=True)
            raise HTTPException(status_code=500, detail=f"Could not save the QC state label: {e!s}")
        if backed:
            dm.evict(dataset_id)
            await dm.load_dataset(dataset_id)
        else:
            dm.saved_in_place(dataset_id, adaptor, output_path)
        return saved


# ---------------------------------------------------------------------------
# POST /datasets/{dataset_id}/assessment/cancel
# ---------------------------------------------------------------------------


@router.post("/datasets/{dataset_id}/assessment/cancel")
async def cancel_pipeline_run(dataset_id: str):
    """Cancel the pipeline run working on this dataset (run, run-stream, rerun).

    The run stops before its next step (a Leiden clustering child process is
    ended at once; other steps finish first) and saves nothing: the dataset
    reopens as it was before the run. The run's own response or stream ends
    with ``status: "cancelled"``. 404 when no run is working on the dataset.
    """
    control = cancel_run(dataset_id)
    if control is None:
        raise HTTPException(status_code=404, detail="No pipeline run is working on this dataset.")
    return {"cancelling": True, "step": control.current_step}


# ---------------------------------------------------------------------------
# Re-filter with per-type % mito bounds, from the original upload (core/mito_refilter.py)
# ---------------------------------------------------------------------------


class MitoRefilterRequest(BaseModel):
    column: str = ""  # the cell types (default: as GET /qc/mito-by-type picks it)
    mito_mode: str = "per_type"  # "global" re-filters with one rule, as before
    floor: bool = False  # per type never stricter than the global rule (qc_mito_type_floor)


@router.get("/datasets/{dataset_id}/qc/mito-refilter")
async def preview_mito_refilter(
    dataset_id: str,
    column: str = "",
    mito_mode: str = "per_type",
    floor: bool = False,
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """What re-filtering from the original upload with per-type % mito bounds on
    ``column`` would do: per type its cells, median, rule and bound, cells removed
    now vs after, brought back and newly removed; how the cells filtering removed
    were typed (the original's column, else the nearest cluster by expression),
    and what it can and cannot bring back. ``floor``: never stricter than the
    global rule. Reads the original upload (seconds)."""
    from scview.core import mito_refilter

    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")
    large_mode.check(adaptor, "Re-filtering from the original upload",
                     "it reads the original upload and re-processes it in memory")
    try:
        return await asyncio.to_thread(mito_refilter.preview, dm, dataset_id, adaptor, column or None,
                                       mito_mode, floor)
    except mito_refilter.RefilterError as e:
        raise HTTPException(status_code=e.status, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Mito re-filter preview failed for %s: %s", dataset_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Re-filter preview failed: {e}")


@router.post("/datasets/{dataset_id}/qc/mito-refilter-stream")
async def run_mito_refilter(
    dataset_id: str,
    body: MitoRefilterRequest,
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """Re-filter from the original upload with per-type % mito bounds and re-run
    the steps the dataset ran (Server-Sent Events: ``refilter`` stages, then the
    pipeline's events). Saved, and recorded in provenance as ``mito_refilter``
    followed by the steps, only when the run reaches its end."""
    from scview.core import mito_refilter

    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")
    large_mode.check(adaptor, "Re-filtering from the original upload",
                     "it reads the original upload and re-processes it in memory")
    try:  # refuse before streaming what can be refused cheaply
        await asyncio.to_thread(mito_refilter.check_runnable, dm, dataset_id, adaptor, body.column or None,
                                body.mito_mode)
    except mito_refilter.RefilterError as e:
        raise HTTPException(status_code=e.status, detail=str(e))

    control = RunControl()  # watchdog, Cancel and the pre-step memory check, as for any run

    def _events(adaptor):
        output = str(dm.derived_h5ad_path(dataset_id, adaptor.h5ad_path))
        return mito_refilter.run_events(dm, dataset_id, adaptor, body.column or None, output, body.mito_mode,
                                        control=control, floor=body.floor)

    async def event_generator():
        try:
            async with aclosing(iterate_exclusive(dm, dataset_id, _events, control)) as events:
                async for event_type, event_data in events:
                    yield f"event: {event_type}\ndata: {json.dumps(event_data)}\n\n"
        except Exception as e:
            logger.error("Mito re-filter failed for %s: %s", dataset_id, e, exc_info=True)
            detail = e.detail if isinstance(e, HTTPException) else str(e)
            yield f"event: error\ndata: {json.dumps({'error': detail})}\n\n"

    return StreamingResponse(
        keepalive(event_generator()),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
