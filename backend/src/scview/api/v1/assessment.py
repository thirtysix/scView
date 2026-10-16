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

from scview.config import Settings
from scview.dependencies import get_dataset_manager, get_settings_dep
from scview.core.dataset_manager import DatasetManager
from scview.core.assessor import assess_preprocessing
from scview.core import cluster_qc, qc_report
from scview.core.pipeline import PipelineParams, run_pipeline, run_pipeline_streamed
from scview.core.processing import iterate_exclusive, run_exclusive
from scview.core.llm_advisor import get_llm_suggestions, get_rule_based_suggestions
from scview.core import provenance
from scview.core.rerun import plan_rerun

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

    try:
        # Samples the matrix to judge its scale: ~1.6 s on Kang, so off the loop.
        state = await asyncio.to_thread(assess_preprocessing, adaptor.adata)
        return state.model_dump()
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
    """Per-cluster QC (size, median genes/counts/% mito/% ribo, doublets) with
    plain-language flags for clusters that stand out (deterministic, no LLM).
    ``groupby`` defaults to the active clustering."""
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
        event_generator(),
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
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
