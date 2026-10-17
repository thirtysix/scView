"""Replicate-aware composition test: do a column's groups (cell types, clusters)
make up different shares of the cells between conditions, with the donors or
samples as replicates (core/composition.py, propeller-style)."""

from __future__ import annotations

import asyncio
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from scview.core import composition as comp
from scview.core.confounding import recorded_batch_key
from scview.core.dataset_manager import DatasetManager
from scview.core.replicates import donor_candidates, pick_donor
from scview.dependencies import get_dataset_manager

router = APIRouter()


class CompositionRequest(BaseModel):
    condition_key: str
    group_key: str
    # Donor / sample column; left out, the most donor-like one with at least two
    # values in every condition is used (core/replicates.py).
    sample_key: str | None = None
    conditions: list[str] | None = None  # values to compare (default: all)
    reference: str | None = None  # the baseline condition (default: the first)
    transform: Literal["logit", "asin"] = "logit"
    paired: bool | None = None  # None: paired when every donor is in every condition


def _detect_sample_key(obs, body: CompositionRequest) -> str | None:
    cond = obs[body.condition_key]
    values = body.conditions or [str(v) for v in cond.dropna().astype(str).unique()]
    labels = cond.astype(str).where(cond.notna())
    masks = [(labels == v).to_numpy() for v in values]
    candidates = donor_candidates(obs, {body.condition_key, body.group_key})
    return pick_donor(obs, candidates, masks)


@router.post("/datasets/{dataset_id}/composition/test")
async def composition_test(
    dataset_id: str,
    body: CompositionRequest,
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """Per group: the mean share of the cells per condition over its samples, the
    fold change, a moderated t (two conditions) or F statistic, p and BH FDR."""
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    adata = adaptor.adata
    obs = adata.obs
    for col in (body.condition_key, body.group_key, body.sample_key):
        if col and col not in obs.columns:
            raise HTTPException(status_code=404, detail=f"Column '{col}' not found.")

    def _run() -> tuple[comp.CompositionResult, bool]:
        key = body.sample_key or _detect_sample_key(obs, body)
        result = comp.composition_test(
            obs, body.condition_key, key, body.group_key,
            conditions=body.conditions, reference=body.reference,
            transform=body.transform, paired=body.paired,
            batch_key=recorded_batch_key(adata),
        )
        return result, not body.sample_key and key is not None

    try:
        # A crosstab and a few small fits, but over every cell: off the loop.
        result, detected = await asyncio.to_thread(_run)
    except comp.CompositionError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    comp.remember(adaptor, result)
    return {**result.to_dict(), "sample_key_detected": detected}
