"""LLM-powered and rule-based preprocessing advisor.

Uses DeepInfra (Llama 3.1 8B) to suggest next preprocessing steps based on
the current state of an AnnData object, with a deterministic rule-based fallback.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class LLMSuggestion(BaseModel):
    """A single preprocessing suggestion."""

    step: str
    recommended: bool
    reasoning: str
    suggested_params: dict[str, Any]


class AdvisorResponse(BaseModel):
    """Full advisor response with structured suggestions and raw LLM output."""

    suggestions: list[LLMSuggestion]
    raw_response: str


# ---------------------------------------------------------------------------
# System prompt for the LLM
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert single-cell RNA-seq bioinformatician. You are given a summary \
of a scRNA-seq dataset and its current preprocessing state. Your job is to \
recommend the next preprocessing steps.

For each step, provide:
- Whether it is recommended (true/false)
- Your reasoning (1-2 sentences)
- Suggested parameter values if applicable

Respond ONLY with valid JSON in this exact format (no markdown, no extra text):
{
  "suggestions": [
    {
      "step": "step_name",
      "recommended": true,
      "reasoning": "Why this step should be run.",
      "suggested_params": {"param_name": value}
    }
  ]
}

The possible preprocessing steps are:
- qc_metrics: Calculate quality control metrics (n_genes, total_counts, pct_mt)
- doublet_detection: Flag likely doublets with Scrublet, writing doublet_score + predicted_doublet to obs (params: expected_doublet_rate). Run before filtering on raw counts; removal is opt-in via filtering's drop_doublets.
- filtering: Filter low-quality cells and genes (params: min_genes, min_cells, max_pct_mt, drop_doublets)
- normalization: Normalize counts per cell (params: target_sum)
- log_transform: Log1p transform the data
- highly_variable_genes: Select highly variable genes (params: n_top_genes)
- scaling: Z-score scale the data (params: max_value_scale)
- pca: Principal component analysis (params: n_pcs)
- neighbors: Compute neighbor graph (params: n_neighbors, n_pcs)
- clustering: Cluster cells (params: resolution, clustering_method)
- embeddings: Compute UMAP/t-SNE embedding
- marker_genes: Find marker genes per cluster
- cell_cycle: Score cell cycle phase

Consider the dataset size when suggesting parameters:
- Small datasets (<5k cells): lower n_neighbors (10-15), fewer PCs (20-30)
- Medium datasets (5k-50k cells): standard parameters
- Large datasets (>50k cells): can use more neighbors (20-30), more PCs (50)

Only suggest steps that have NOT already been completed. Follow the standard \
scanpy workflow order."""


# ---------------------------------------------------------------------------
# LLM-based suggestions
# ---------------------------------------------------------------------------


def _build_user_message(
    preprocessing_state: dict[str, Any],
    dataset_summary: dict[str, Any],
) -> str:
    """Format the preprocessing state and dataset summary into a user prompt."""
    lines = ["## Dataset Summary"]
    lines.append(f"- Cells: {dataset_summary.get('n_cells', 'unknown')}")
    lines.append(f"- Genes: {dataset_summary.get('n_genes', 'unknown')}")

    species = dataset_summary.get("species", None)
    if species:
        lines.append(f"- Species: {species}")

    organism = dataset_summary.get("organism", None)
    if organism and organism != species:
        lines.append(f"- Organism: {organism}")

    lines.append("")
    lines.append("## Current Preprocessing State")

    for step_name, status in preprocessing_state.items():
        if isinstance(status, dict):
            done = status.get("done", False)
            confidence = status.get("confidence", "unknown")
            details = status.get("details", "")
            state_str = "DONE" if done else "NOT DONE"
            lines.append(
                f"- {step_name}: {state_str} (confidence: {confidence}) "
                f"-- {details}"
            )

    lines.append("")
    lines.append(
        "Based on this state, what preprocessing steps should be run next? "
        "Provide your suggestions as JSON."
    )

    return "\n".join(lines)


def _parse_llm_response(raw: str) -> list[LLMSuggestion]:
    """Attempt to parse the LLM response into structured suggestions."""
    # Try to extract JSON from the response
    text = raw.strip()

    # Handle markdown code blocks
    if "```json" in text:
        text = text.split("```json", 1)[1]
        text = text.split("```", 1)[0]
    elif "```" in text:
        text = text.split("```", 1)[1]
        text = text.split("```", 1)[0]

    text = text.strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Try to find a JSON object in the response
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                data = json.loads(text[start:end])
            except json.JSONDecodeError:
                logger.warning("Could not parse LLM response as JSON")
                return []
        else:
            return []

    suggestions = []
    raw_suggestions = data.get("suggestions", [])
    if not isinstance(raw_suggestions, list):
        return []

    for item in raw_suggestions:
        if not isinstance(item, dict):
            continue
        try:
            suggestions.append(
                LLMSuggestion(
                    step=str(item.get("step", "")),
                    recommended=bool(item.get("recommended", False)),
                    reasoning=str(item.get("reasoning", "")),
                    suggested_params=item.get("suggested_params", {}) or {},
                )
            )
        except Exception as exc:
            logger.warning("Failed to parse suggestion item: %s -- %s", item, exc)

    return suggestions


async def get_llm_suggestions(
    preprocessing_state: dict[str, Any],
    dataset_summary: dict[str, Any],
    api_key: str,
) -> AdvisorResponse:
    """Get preprocessing suggestions from DeepInfra's Llama 3.1 8B model.

    Parameters
    ----------
    preprocessing_state
        Dictionary representation of a PreprocessingState (from assess_preprocessing).
    dataset_summary
        Dictionary with at least 'n_cells' and 'n_genes' keys.
    api_key
        DeepInfra API key.

    Returns
    -------
    AdvisorResponse
        Structured suggestions and the raw LLM response text.
        Falls back to rule-based suggestions if the API call fails.
    """
    if not api_key:
        logger.info("No API key provided; falling back to rule-based suggestions")
        return get_rule_based_suggestions(preprocessing_state, dataset_summary)

    try:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.deepinfra.com/v1/openai",
        )

        user_message = _build_user_message(preprocessing_state, dataset_summary)

        response = await client.chat.completions.create(
            model="meta-llama/Meta-Llama-3.1-8B-Instruct",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.3,
            max_tokens=2048,
        )

        raw_response = response.choices[0].message.content or ""
        suggestions = _parse_llm_response(raw_response)

        if not suggestions:
            logger.warning(
                "LLM returned no parseable suggestions; "
                "supplementing with rule-based fallback"
            )
            fallback = get_rule_based_suggestions(
                preprocessing_state, dataset_summary
            )
            return AdvisorResponse(
                suggestions=fallback.suggestions,
                raw_response=raw_response,
            )

        return AdvisorResponse(
            suggestions=suggestions,
            raw_response=raw_response,
        )

    except ImportError:
        logger.error(
            "openai package not installed; falling back to rule-based suggestions"
        )
        return get_rule_based_suggestions(preprocessing_state, dataset_summary)
    except Exception as exc:
        logger.error(
            "LLM API call failed: %s; falling back to rule-based suggestions", exc
        )
        fallback = get_rule_based_suggestions(preprocessing_state, dataset_summary)
        return AdvisorResponse(
            suggestions=fallback.suggestions,
            raw_response=f"API error: {exc}",
        )


# ---------------------------------------------------------------------------
# Rule-based fallback
# ---------------------------------------------------------------------------


def _is_step_done(state: dict[str, Any], step: str) -> bool:
    """Check if a step is marked as done with at least medium confidence."""
    status = state.get(step, {})
    if isinstance(status, dict):
        return status.get("done", False) and status.get("confidence") in (
            "high",
            "medium",
        )
    return False


def get_rule_based_suggestions(
    preprocessing_state: dict[str, Any],
    dataset_summary: dict[str, Any] | None = None,
) -> AdvisorResponse:
    """Generate preprocessing suggestions using standard scRNA-seq best practices.

    This function works without any API key and applies deterministic rules
    based on the detected preprocessing state.

    Parameters
    ----------
    preprocessing_state
        Dictionary representation of a PreprocessingState.
    dataset_summary
        Optional dictionary with 'n_cells' and 'n_genes' keys for parameter tuning.

    Returns
    -------
    AdvisorResponse
        Structured suggestions based on best-practice rules.
    """
    if dataset_summary is None:
        dataset_summary = {}

    n_cells = dataset_summary.get("n_cells", 10000)
    n_genes = dataset_summary.get("n_genes", 20000)

    suggestions: list[LLMSuggestion] = []

    # Determine size-appropriate parameters
    if n_cells < 5000:
        default_n_neighbors = 10
        default_n_pcs = 20
        default_resolution = 0.8
    elif n_cells < 50000:
        default_n_neighbors = 15
        default_n_pcs = 50
        default_resolution = 1.0
    else:
        default_n_neighbors = 20
        default_n_pcs = 50
        default_resolution = 1.0

    # --- QC metrics ---
    if not _is_step_done(preprocessing_state, "qc_metrics"):
        suggestions.append(
            LLMSuggestion(
                step="qc_metrics",
                recommended=True,
                reasoning=(
                    "QC metrics have not been calculated. This is the essential "
                    "first step to assess data quality before any filtering."
                ),
                suggested_params={},
            )
        )

    # --- Filtering ---
    if not _is_step_done(preprocessing_state, "filtering"):
        suggestions.append(
            LLMSuggestion(
                step="filtering",
                recommended=True,
                reasoning=(
                    "Cells and genes have not been filtered. Removing low-quality "
                    "cells (few genes, high mitochondrial %) and rarely-expressed "
                    "genes is critical."
                ),
                suggested_params={
                    "min_genes": 200,
                    "min_cells": 3,
                    "max_pct_mt": 20.0,
                },
            )
        )

    # --- Normalization ---
    if not _is_step_done(preprocessing_state, "normalization"):
        suggestions.append(
            LLMSuggestion(
                step="normalization",
                recommended=True,
                reasoning=(
                    "Data does not appear to be normalized. Library-size "
                    "normalization is necessary to make expression values "
                    "comparable across cells."
                ),
                suggested_params={"target_sum": 1e4},
            )
        )

    # --- Log transform ---
    if not _is_step_done(preprocessing_state, "log_transform"):
        suggestions.append(
            LLMSuggestion(
                step="log_transform",
                recommended=True,
                reasoning=(
                    "Data does not appear log-transformed. Log1p transformation "
                    "reduces skewness and is standard practice before downstream "
                    "analysis."
                ),
                suggested_params={},
            )
        )

    # --- Highly variable genes ---
    if not _is_step_done(preprocessing_state, "highly_variable_genes"):
        n_top = (
            min(2000, n_genes - 1) if n_genes > 2000 else max(500, n_genes // 2)
        )
        suggestions.append(
            LLMSuggestion(
                step="highly_variable_genes",
                recommended=True,
                reasoning=(
                    "Highly variable genes have not been selected. HVG selection "
                    "focuses on biologically informative genes and reduces noise "
                    "from housekeeping genes."
                ),
                suggested_params={"n_top_genes": n_top},
            )
        )

    # --- Scaling ---
    if not _is_step_done(preprocessing_state, "scaling"):
        suggestions.append(
            LLMSuggestion(
                step="scaling",
                recommended=True,
                reasoning=(
                    "Data has not been scaled. Z-score scaling ensures that highly "
                    "expressed genes do not dominate PCA and downstream analyses."
                ),
                suggested_params={"max_value_scale": 10.0},
            )
        )

    # --- PCA ---
    if not _is_step_done(preprocessing_state, "pca"):
        suggestions.append(
            LLMSuggestion(
                step="pca",
                recommended=True,
                reasoning=(
                    "PCA has not been computed. Dimensionality reduction via PCA "
                    "is needed before computing the neighbor graph and embeddings."
                ),
                suggested_params={"n_pcs": default_n_pcs},
            )
        )

    # --- Neighbors ---
    if not _is_step_done(preprocessing_state, "neighbors"):
        suggestions.append(
            LLMSuggestion(
                step="neighbors",
                recommended=True,
                reasoning=(
                    "The neighborhood graph has not been computed. This is "
                    "required for clustering (Leiden/Louvain) and UMAP embedding."
                ),
                suggested_params={
                    "n_neighbors": default_n_neighbors,
                    "n_pcs": default_n_pcs,
                },
            )
        )

    # --- Clustering ---
    if not _is_step_done(preprocessing_state, "clustering"):
        suggestions.append(
            LLMSuggestion(
                step="clustering",
                recommended=True,
                reasoning=(
                    "No clustering has been performed. Leiden clustering is the "
                    "standard approach for identifying cell populations in "
                    "scRNA-seq data."
                ),
                suggested_params={
                    "resolution": default_resolution,
                    "clustering_method": "leiden",
                },
            )
        )

    # --- Embeddings ---
    if not _is_step_done(preprocessing_state, "embeddings"):
        suggestions.append(
            LLMSuggestion(
                step="embeddings",
                recommended=True,
                reasoning=(
                    "No 2D embedding (UMAP/t-SNE) has been computed. UMAP "
                    "provides an intuitive visualization of cell populations "
                    "and cluster structure."
                ),
                suggested_params={},
            )
        )

    # --- Marker genes ---
    if not _is_step_done(preprocessing_state, "marker_genes"):
        # Only recommend if clustering is done or will be done
        has_clustering = _is_step_done(preprocessing_state, "clustering")
        clustering_planned = any(s.step == "clustering" for s in suggestions)

        if has_clustering or clustering_planned:
            suggestions.append(
                LLMSuggestion(
                    step="marker_genes",
                    recommended=True,
                    reasoning=(
                        "Marker genes have not been identified. Differential "
                        "expression analysis reveals which genes define each "
                        "cluster, aiding cell type annotation."
                    ),
                    suggested_params={},
                )
            )
        else:
            suggestions.append(
                LLMSuggestion(
                    step="marker_genes",
                    recommended=False,
                    reasoning=(
                        "Marker gene analysis requires clustering to be "
                        "completed first. Run clustering before marker gene "
                        "detection."
                    ),
                    suggested_params={},
                )
            )

    # --- Cell cycle (optional, not always recommended) ---
    if not _is_step_done(preprocessing_state, "cell_cycle"):
        suggestions.append(
            LLMSuggestion(
                step="cell_cycle",
                recommended=False,
                reasoning=(
                    "Cell cycle scoring is optional and depends on the "
                    "biological question. It is useful when cell cycle effects "
                    "may confound clustering, e.g., in proliferating tissues "
                    "or cancer samples."
                ),
                suggested_params={},
            )
        )

    return AdvisorResponse(
        suggestions=suggestions,
        raw_response="Rule-based suggestions (no LLM API call)",
    )
