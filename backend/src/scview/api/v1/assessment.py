"""Data assessment endpoints -- detect preprocessing state, run pipeline, get suggestions."""

from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from scview.config import Settings
from scview.dependencies import get_dataset_manager, get_settings_dep
from scview.core.dataset_manager import DatasetManager
from scview.core.assessor import assess_preprocessing
from scview.core.pipeline import PipelineParams, run_pipeline, run_pipeline_streamed
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
        state = assess_preprocessing(adaptor.adata)
        return state.model_dump()
    except Exception as e:
        logger.error("Assessment failed for %s: %s", dataset_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Assessment failed: {e}")


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

    try:
        # Write processing output to the derived layer, never the original
        # upload or conversion — those stay immutable.
        output_path = str(dm.derived_h5ad_path(dataset_id, adaptor.h5ad_path))

        # Run the pipeline
        new_adata, result = run_pipeline(
            adata=adaptor.adata,
            steps=body.steps,
            params=params,
            output_path=output_path,
        )

        # Reload the dataset to pick up changes
        # Clear the cached adaptor so it reloads from disk
        if dataset_id in dm._datasets:
            dm._datasets[dataset_id].close()
            del dm._datasets[dataset_id]
        if dataset_id in dm._load_order:
            dm._load_order.remove(dataset_id)

        # Reload
        await dm.load_dataset(dataset_id)

        return result.to_dict()

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

    # Write processing output to the derived layer, never the original
    # upload or conversion — those stay immutable.
    output_path = str(dm.derived_h5ad_path(dataset_id, adaptor.h5ad_path))

    async def event_generator():
        try:
            gen = run_pipeline_streamed(
                adata=adaptor.adata,
                steps=body.steps,
                params=params,
                output_path=output_path,
            )
            for event_type, event_data in gen:
                line = f"event: {event_type}\ndata: {json.dumps(event_data)}\n\n"
                yield line

            # Reload dataset after pipeline completes
            if dataset_id in dm._datasets:
                dm._datasets[dataset_id].close()
                del dm._datasets[dataset_id]
            if dataset_id in dm._load_order:
                dm._load_order.remove(dataset_id)
            await dm.load_dataset(dataset_id)

        except Exception as e:
            logger.error("Streamed pipeline failed for %s: %s", dataset_id, e, exc_info=True)
            error_line = f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"
            yield error_line

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

    try:
        output_path = str(dm.derived_h5ad_path(dataset_id, adaptor.h5ad_path))
        _, result = run_pipeline(
            adata=adaptor.adata, steps=plan.rerun_steps, params=params, output_path=output_path
        )
        # Reload so the new state is served.
        if dataset_id in dm._datasets:
            dm._datasets[dataset_id].close()
            del dm._datasets[dataset_id]
        if dataset_id in dm._load_order:
            dm._load_order.remove(dataset_id)
        await dm.load_dataset(dataset_id)
        return {"plan": plan.model_dump(), "result": result.to_dict()}
    except Exception as e:
        logger.error("Re-run failed for %s: %s", dataset_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Re-run failed: {e}")
