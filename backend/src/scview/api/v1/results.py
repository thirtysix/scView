"""The per-dataset results store (``core/results_store.py``): list, read, delete, and
restore an earlier markers / enrichment result (``core/uns_history.py``)."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query

from scview.core import results_store, uns_history
from scview.core.dataset_manager import DatasetManager
from scview.core.obs_writes import dataset_edit, edit_room, editable, save_edit, uns_snapshot
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


@router.post("/datasets/{dataset_id}/results/{result_id}/restore")
async def restore_result(dataset_id: str, result_id: str,
                         dm: DatasetManager = Depends(get_dataset_manager)) -> dict:
    """Make an earlier markers or enrichment result current again (the one it
    replaces is kept in its place). Restored markers drop the column's enrichment,
    which was computed from the markers they replace. 409 when the column's groups
    have changed since the result was computed."""
    _known(dm, dataset_id)
    async with dataset_edit(dataset_id) as held:
        adaptor = await dm.get_or_load_dataset(dataset_id)
        if adaptor is None:
            raise HTTPException(status_code=404, detail="Dataset not found.")
        try:
            entry, value = await asyncio.to_thread(uns_history.check_restorable, dataset_id, result_id, adaptor)
        except uns_history.RestoreError as exc:
            raise HTTPException(status_code=exc.status, detail=exc.detail)
        output_path = str(dm.derived_h5ad_path(dataset_id, adaptor.h5ad_path))
        backed = await edit_room(dm, dataset_id, adaptor)

        def _run() -> list[str]:
            adata = editable(adaptor)
            before = uns_snapshot(adata)
            keys = uns_history.apply(adata, value)
            save_edit(adaptor, adata, output_path, before)
            return keys

        try:
            keys = await held.run(_run)
        except Exception as exc:
            dm.evict(dataset_id)  # may hold a restore that was never saved
            raise HTTPException(status_code=500, detail=f"Could not restore the result: {exc!s}")
        if backed:
            dm.evict(dataset_id)
            await dm.load_dataset(dataset_id)
        else:
            dm.saved_in_place(dataset_id, adaptor, output_path)
    # It is the current result now (in the h5ad); the one it replaced took its place here.
    await asyncio.to_thread(results_store.delete, dataset_id, result_id)
    return {"restored": result_id, "kind": entry["kind"], "key": entry["key"], "uns_keys": keys}
