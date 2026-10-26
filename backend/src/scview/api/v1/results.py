"""The per-dataset results store (``core/results_store.py``): list, read, delete."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query

from scview.core import results_store
from scview.core.dataset_manager import DatasetManager
from scview.dependencies import get_dataset_manager

router = APIRouter()

_FIELDS = ("id", "kind", "key", "label", "status", "version", "bytes", "stale")


def _summary(e: dict) -> dict:
    out = {k: e.get(k) for k in _FIELDS}
    out["created_at"] = datetime.fromtimestamp(e["created_at"], tz=timezone.utc).isoformat()
    return out


def _known(dm: DatasetManager, dataset_id: str) -> None:
    if dm.get_dataset_info(dataset_id) is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")


@router.get("/datasets/{dataset_id}/results")
async def list_results(
    dataset_id: str,
    kind: str | None = Query(default=None, max_length=40),
    dm: DatasetManager = Depends(get_dataset_manager),
) -> dict:
    """The dataset's stored results, newest first. ``stale`` is true for one
    computed on an earlier version of the data (cells or recipe changed), null
    when the dataset is not open (nothing to compare with)."""
    _known(dm, dataset_id)
    rows = await asyncio.to_thread(results_store.entries, dataset_id, kind, dm.get_dataset(dataset_id))
    return {"results": [_summary(e) for e in rows], "keep_superseded": results_store.KEEP_SUPERSEDED,
            "max_bytes": results_store.MAX_BYTES, "total_bytes": sum(e.get("bytes") or 0 for e in rows)}


@router.get("/datasets/{dataset_id}/results/{result_id}")
async def get_result(dataset_id: str, result_id: str,
                     dm: DatasetManager = Depends(get_dataset_manager)) -> dict:
    """One stored result's entry with its parameters and fingerprint."""
    _known(dm, dataset_id)
    rows = await asyncio.to_thread(results_store.entries, dataset_id, None, dm.get_dataset(dataset_id))
    e = next((e for e in rows if e["id"] == result_id), None)
    if e is None:
        raise HTTPException(status_code=404, detail="No such stored result.")
    return {**_summary(e), "params": e.get("params") or {}, "fingerprint": e.get("fingerprint")}


@router.delete("/datasets/{dataset_id}/results/{result_id}")
async def delete_result(dataset_id: str, result_id: str,
                        dm: DatasetManager = Depends(get_dataset_manager)) -> dict:
    _known(dm, dataset_id)
    if not await asyncio.to_thread(results_store.delete, dataset_id, result_id):
        raise HTTPException(status_code=404, detail="No such stored result.")
    return {"deleted": result_id}
