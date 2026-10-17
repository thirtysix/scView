"""Cell filters (``core/cell_filter.py``): which cells match conditions on genes
and metadata. The Unified View's filter builder asks for the count while the
filter is edited and for the cells themselves on "Select"."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from scview.core.cell_filter import CellFilter, FilterError, evaluate, mask_to_base64
from scview.core.dataset_manager import DatasetManager
from scview.dependencies import get_dataset_manager

router = APIRouter()


class FilterRequest(BaseModel):
    filter: CellFilter
    # False: the count only (the live count while editing).
    mask: bool = True


class FilterResponse(BaseModel):
    n_cells: int
    n_matching: int
    # Packed bitmask, base64: cell i is bit i % 8 (least significant first) of byte i // 8.
    mask: str | None = None


@router.post("/datasets/{dataset_id}/cells/filter", response_model=FilterResponse)
async def filter_cells(
    dataset_id: str,
    body: FilterRequest,
    dm: DatasetManager = Depends(get_dataset_manager),
) -> FilterResponse:
    """The cells matching a filter: their number, and (``mask``) which they are."""
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    def _run() -> FilterResponse:
        hits = evaluate(adaptor, body.filter)
        return FilterResponse(
            n_cells=int(hits.size),
            n_matching=int(hits.sum()),
            mask=mask_to_base64(hits) if body.mask else None,
        )

    # Reading a gene's column from a large matrix takes a while: off the loop.
    try:
        return await asyncio.to_thread(_run)
    except FilterError as exc:
        raise HTTPException(status_code=exc.status, detail=str(exc)) from None
