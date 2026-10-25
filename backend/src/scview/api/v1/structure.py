"""``GET /datasets/{id}/structure``: what the AnnData object holds (``core/structure.py``):
matrices summarised from a sample (never read whole), tables, embeddings, graphs, uns
as a tree, and the problems worth a look."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException

from scview.core import structure
from scview.core.dataset_manager import DatasetManager
from scview.dependencies import get_dataset_manager

router = APIRouter()


@router.get("/datasets/{dataset_id}/structure")
async def dataset_structure(dataset_id: str, dm: DatasetManager = Depends(get_dataset_manager)):
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    # Samples the matrices (a few slices of an on-disk file): off the event loop.
    return await asyncio.to_thread(structure.summarise, adaptor)
