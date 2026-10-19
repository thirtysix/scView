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
from scview.core.dataset_manager import DatasetManager
from scview.core.assessor import assess_preprocessing, recorded_normalization_method
from scview.core import (
    cluster_qc, doublets, qc_filter, qc_groups, qc_report, sample_pseudobulk, sex_check,
)
from scview.core.obs_writes import dataset_edit, uns_snapshot, write_derived
from scview.core.pipeline import PipelineParams, run_pipeline, run_pipeline_streamed
from scview.core.processing import iterate_exclusive, run_exclusive
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
                "normalization_method": recorded_normalization_method(adata),
                # Re-filtering keeps the mode the dataset was filtered with (fixed
                # before MAD existed); None: never filtered by scView.
                "qc_threshold_mode": filtered["qc_threshold_mode"] if filtered else None}

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
    sample_column: str = "",
    cluster_column: str = "",
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """The cells the filtering step's thresholds would remove (core/qc_filter.py):
    overall, per threshold, per sample and per cluster, with warnings for a sample
    or cluster lost whole or more than half, each sample's QC medians and range
    flags, and in MAD mode (``mode``, the default) each sample's thresholds.
    ``sample_column`` names the samples (default: detected), as the step's
    ``qc_sample_column`` does. Thresholds left out take the pipeline's defaults.
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
                       ("qc_mad_upper", mad_upper)):
        if value is not None:
            setattr(params, key, value)
    thresholds = qc_filter.Thresholds.from_params(params)

    def _run() -> dict:
        return qc_filter.preview(adaptor.adata, adaptor.qc_per_cell(), thresholds,
                                 sample_column or None, cluster_column or None)

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
    column: str = ""  # the sample column for the per-sample counts (default: detected)


@router.get("/datasets/{dataset_id}/qc/doublets")
async def get_doublets(
    dataset_id: str,
    threshold: float | None = None,
    column: str = "",
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """The doublet-score histogram and how many cells ``threshold`` (default: the
    applied one, else Scrublet's) calls doublets, overall and per sample, with
    the expected doublet rate suggested from the cells per sample. Without scores
    only the suggestion is returned (``available`` false)."""
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
        backed = getattr(adaptor.adata, "isbacked", False)

        def _run():
            work = adaptor.adata.to_memory() if backed else adaptor.adata
            before = uns_snapshot(work)
            result = doublets.apply_threshold(work, body.threshold, body.column or None)
            write_derived(work, output_path, adaptor.h5ad_path, before)
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


@router.post("/datasets/{dataset_id}/assessment/run")
async def run_assessment_pipeline(
    dataset_id: str,
    body: RunPipelineRequest,
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """Run specified preprocessing steps on the dataset and save the result."""
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")

    if not body.steps:
        raise HTTPException(status_code=400, detail="No steps specified.")

    # Build PipelineParams from the request body
    params = PipelineParams()
    if body.params:
        for key, value in body.params.items():
            if hasattr(params, key):
                setattr(params, key, value)

    def _run(adaptor):
        # Write processing output to the derived layer, never the original
        # upload or conversion — those stay immutable.
        output_path = str(dm.derived_h5ad_path(dataset_id, adaptor.h5ad_path))
        return run_pipeline(
            adata=adaptor.adata,
            steps=body.steps,
            params=params,
            output_path=output_path,
        )[1]

    try:
        # In a worker thread with the dataset to itself; the cached copy is
        # dropped afterwards, so what is served is what is on disk.
        result = await run_exclusive(dm, dataset_id, _run)
        return result.to_dict()

    except HTTPException:
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
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")

    if not body.steps:
        raise HTTPException(status_code=400, detail="No steps specified.")

    params = PipelineParams()
    if body.params:
        for key, value in body.params.items():
            if hasattr(params, key):
                setattr(params, key, value)

    def _events(adaptor):
        # Write processing output to the derived layer, never the original
        # upload or conversion — those stay immutable.
        return run_pipeline_streamed(
            adata=adaptor.adata,
            steps=body.steps,
            params=params,
            output_path=str(dm.derived_h5ad_path(dataset_id, adaptor.h5ad_path)),
        )

    async def event_generator():
        # Steps run in a worker thread with the dataset to itself. When the
        # client goes away mid-run (tab closed, proxy timeout) the run stops
        # after its current step without saving, and the cached copy, which may
        # hold unsaved steps, is dropped.
        try:
            # aclosing: stopping this generator must stop the run at once, not
            # whenever the inner generator is garbage-collected.
            async with aclosing(iterate_exclusive(dm, dataset_id, _events)) as events:
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
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")

    plan = plan_rerun(_history_steps(adaptor), body.edited_step)
    if not plan.rerun_steps:
        raise HTTPException(status_code=400, detail=plan.message or "Nothing to re-run.")
    if plan.requires_reprocess:
        raise HTTPException(status_code=409, detail=plan.message)

    params = PipelineParams()
    if body.params:
        for key, value in body.params.items():
            if hasattr(params, key):
                setattr(params, key, value)

    def _run(adaptor):
        output_path = str(dm.derived_h5ad_path(dataset_id, adaptor.h5ad_path))
        return run_pipeline(
            adata=adaptor.adata, steps=plan.rerun_steps, params=params, output_path=output_path
        )[1]

    try:
        # As /assessment/run: a worker thread, then serve what is on disk.
        result = await run_exclusive(dm, dataset_id, _run)
        return {"plan": plan.model_dump(), "result": result.to_dict()}
    except HTTPException:
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
