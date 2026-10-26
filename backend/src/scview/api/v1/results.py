"""The per-dataset results store (``core/results_store.py``): list, read, delete, and
restore an earlier markers / enrichment result (``core/uns_history.py``)."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

from scview.core import results_export, results_store, uns_history
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


class PinBody(BaseModel):
    pinned: bool = True


@router.post("/datasets/{dataset_id}/results/{result_id}/pin")
async def pin_result(dataset_id: str, result_id: str, body: PinBody,
                     dm: DatasetManager = Depends(get_dataset_manager)) -> dict:
    """Pin a stored result (kept until deleted, never dropped by retention), or unpin it."""
    _known(dm, dataset_id)
    if not await asyncio.to_thread(results_store.set_pinned, dataset_id, result_id, body.pinned):
        raise HTTPException(status_code=404, detail="No such stored result.")
    return {"id": result_id, "pinned": body.pinned}


async def _entry_value(dm: DatasetManager, dataset_id: str, result_id: str):
    _known(dm, dataset_id)
    rows = await asyncio.to_thread(results_store.entries, dataset_id, None, dm.get_dataset(dataset_id))
    e = next((e for e in rows if e["id"] == result_id), None)
    got = await asyncio.to_thread(results_store.load, dataset_id, result_id) if e else None
    if e is None or got is None:
        raise HTTPException(status_code=404, detail="No such stored result.")
    return e, got[1]


@router.get("/datasets/{dataset_id}/results/{result_id}/table.tsv.gz")
async def result_table(dataset_id: str, result_id: str,
                       dm: DatasetManager = Depends(get_dataset_manager)) -> Response:
    """The result as one tab-separated table, gzipped (header + rows; the parameters are
    in the results zip's manifest)."""
    e, value = await _entry_value(dm, dataset_id, result_id)

    def _run() -> bytes | None:
        df = results_export.table(e, value)
        return None if df is None else results_export.tsv_gz(df)

    body = await asyncio.to_thread(_run)
    if body is None:
        raise HTTPException(status_code=404, detail="This result holds no table to export.")
    name = f"{results_export.file_stem(e)}.tsv.gz"
    return Response(body, media_type="application/gzip", headers={
        "Content-Disposition": f'attachment; filename="{name}"', "Content-Encoding": "identity"})


@router.get("/datasets/{dataset_id}/results.zip")
async def results_zip(dataset_id: str, dm: DatasetManager = Depends(get_dataset_manager)) -> FileResponse:
    """Every stored result as ``results/{label}.tsv.gz`` plus ``manifest.json`` (parameters,
    scView version, created, fingerprint, stale flag of each)."""
    _known(dm, dataset_id)
    out = dm.data_dir / "cache" / dataset_id / "results.zip"
    out.parent.mkdir(parents=True, exist_ok=True)
    await asyncio.to_thread(results_export.results_zip, dataset_id, out, dm.get_dataset(dataset_id))
    return FileResponse(out, media_type="application/zip", filename=f"scview_results_{dataset_id[:8]}.zip",
                        headers={"Content-Encoding": "identity"})


@router.get("/datasets/{dataset_id}/results/{result_id}/compare")
async def compare_with_current(dataset_id: str, result_id: str,
                               n: int = Query(default=50, ge=5, le=500),
                               dm: DatasetManager = Depends(get_dataset_manager)) -> dict:
    """An earlier markers result against the column's current markers (top ``n`` genes
    shared per group), or an earlier enrichment against the group's current one
    (significant terms shared)."""
    e, value = await _entry_value(dm, dataset_id, result_id)
    if e["kind"] not in uns_history.KINDS:
        raise HTTPException(status_code=400, detail="Only markers and enrichment compare with a current result.")
    adaptor = await dm.get_or_load_dataset(dataset_id)
    current = adaptor.adata.uns.get(value["uns_key"]) if adaptor is not None else None
    if current is None:
        raise HTTPException(status_code=404, detail="There is no current result to compare with.")
    if e["kind"] == "markers":
        out = await asyncio.to_thread(results_export.compare_markers, value["value"], current, n)
    else:
        out = await asyncio.to_thread(results_export.compare_enrichment, value["value"], current, min(n, 50))
    return {"id": result_id, "kind": e["kind"], "key": e["key"], "stale": e["stale"], **out}
