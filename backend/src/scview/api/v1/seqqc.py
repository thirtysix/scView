"""Sequencing QC endpoints: the Cell Ranger metrics and barcode ranks of a dataset.

``GET /datasets/{id}/seq-metrics``: per sample, the parsed ``metrics_summary.csv``, the
Cell Ranger alerts it crosses, and the raw matrix's knee / inflection summary with a
downsampled rank curve (``core/seq_metrics.py``); 404 when none is attached.
``POST /datasets/{id}/seq-metrics``: attach those files to a dataset uploaded without
them (multipart ``files``; optional ``sample`` + ``column`` for one sample of a merged
dataset); ``DELETE ...?sample=`` removes one sample's record. Only that ``uns`` entry
is written (the file is copied and the entry rewritten; the matrix is never read).
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import uuid
from pathlib import Path

import h5py
from anndata.io import write_elem
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from scview.config import Settings
from scview.core import seq_metrics
from scview.core.dataset_manager import DatasetManager
from scview.core.obs_writes import dataset_edit, save_edit, uns_snapshot
from scview.dependencies import get_dataset_manager, get_settings_dep

logger = logging.getLogger(__name__)

router = APIRouter()

_MAX_FILES = 6
_EMPTY = {"samples": [], "n_samples": 0, "metrics_info": []}


@router.get("/datasets/{dataset_id}/seq-metrics")
async def get_seq_metrics(
    dataset_id: str, allow_empty: bool = False, dm: DatasetManager = Depends(get_dataset_manager)
):
    """404 when none is attached; with ``allow_empty`` an empty answer instead (the card
    asks on every dataset, and a 404 is a console error in the browser)."""
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")
    body = await asyncio.to_thread(seq_metrics.payload, adaptor.adata)
    if body is None:
        if allow_empty:
            return _EMPTY
        raise HTTPException(status_code=404, detail="No sequencing metrics are attached to this "
                                                    "dataset.")
    return body


def _save_uns_entry(source: str, output: str, value: str | None) -> None:
    """Save only the sequencing entry: copy the dataset's file and rewrite that one uns
    key, so the matrix is never read, written or recompressed (a full write of a
    freshly uploaded 2 GB file takes minutes and its memory)."""
    partial = output + ".partial"
    shutil.copyfile(source, partial)  # content only: a read-only original gives a writable copy
    try:
        with h5py.File(partial, "a") as f:
            if "uns" not in f:
                write_elem(f, "uns", {})
            if seq_metrics.UNS_KEY in f["uns"]:
                del f["uns"][seq_metrics.UNS_KEY]
            if value is not None:  # None: the entry is removed
                write_elem(f["uns"], seq_metrics.UNS_KEY, value)
    except Exception:
        os.remove(partial)
        raise
    os.replace(partial, output)


@router.post("/datasets/{dataset_id}/seq-metrics")
async def attach_seq_metrics(
    dataset_id: str,
    files: list[UploadFile] = File(...),
    sample: str = Form(""),
    column: str = Form(""),
    settings: Settings = Depends(get_settings_dep),
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """Attach a metrics_summary.csv and/or a raw matrix (.h5, or matrix.mtx + barcodes.tsv
    + features.tsv) to the dataset (or to one ``sample`` of ``column``)."""
    if not files:
        raise HTTPException(status_code=400, detail="No files were sent.")
    if len(files) > _MAX_FILES:
        raise HTTPException(status_code=400, detail=f"At most {_MAX_FILES} files at once.")
    if (await dm.get_or_load_dataset(dataset_id)) is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")

    stage = Path(settings.DATA_DIR) / "tmp" / f"seqqc-{uuid.uuid4().hex}"
    stage.mkdir(parents=True, exist_ok=True)
    try:
        paths: list[Path] = []
        total = 0
        for f in files:
            dest = stage / Path(f.filename or "unnamed").name  # never the client's path
            with open(dest, "wb") as fh:
                while chunk := await f.read(1024 * 1024):
                    total += len(chunk)
                    if total > settings.max_upload_bytes:
                        raise HTTPException(
                            status_code=413,
                            detail=f"Files exceed the maximum upload size of "
                                   f"{settings.MAX_UPLOAD_SIZE_MB} MB.")
                    fh.write(chunk)
            paths.append(dest)
        return await _attach(dataset_id, paths, sample.strip(), column.strip(), dm)
    finally:
        shutil.rmtree(stage, ignore_errors=True)


@router.delete("/datasets/{dataset_id}/seq-metrics")
async def remove_seq_metrics(
    dataset_id: str, sample: str, dm: DatasetManager = Depends(get_dataset_manager)
):
    """Remove one sample's attached metrics (a wrong file, say); 404 when it has none.
    Answers with what is left (``samples`` empty when nothing is)."""

    def edit(work) -> None:
        if not seq_metrics.remove(work, sample):
            raise seq_metrics.SeqQcError(f"No sequencing metrics are attached for '{sample}'.", 404)

    return await _edit(dataset_id, dm, edit) or _EMPTY


async def _attach(dataset_id: str, paths: list[Path], sample: str, column: str,
                  dm: DatasetManager) -> dict:
    name = sample or (dm.get_dataset_info(dataset_id) or {}).get("name") or "sample"
    return await _edit(
        dataset_id, dm, lambda work: seq_metrics.attach_files(work, paths, name, column or None))


async def _edit(dataset_id: str, dm: DatasetManager, edit) -> dict | None:
    """Run ``edit(adata)`` in a worker thread under the dataset's edit lock, save the
    sequencing entry alone, and answer with the new payload."""
    async with dataset_edit(dataset_id) as held:
        adaptor = await dm.get_or_load_dataset(dataset_id)
        if adaptor is None:
            raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")
        output_path = str(dm.derived_h5ad_path(dataset_id, adaptor.h5ad_path))
        large = adaptor.is_large
        backed = getattr(adaptor.adata, "isbacked", False) and not large

        def _run():
            work = adaptor.adata
            before = uns_snapshot(work)
            edit(work)
            if large:  # written into its h5ad in place (core/large_mode.py)
                save_edit(adaptor, work, output_path, before)
            else:
                _save_uns_entry(adaptor.h5ad_path, output_path, work.uns.get(seq_metrics.UNS_KEY))
            return seq_metrics.payload(work)

        try:
            body = await held.run(_run)
        except seq_metrics.SeqQcError as e:
            raise HTTPException(status_code=e.status, detail=str(e))  # raised before any change
        except Exception as e:
            dm.evict(dataset_id)  # the cached copy may hold an entry that was never saved
            logger.error("Saving sequencing metrics failed for %s: %s", dataset_id, e,
                         exc_info=True)
            raise HTTPException(status_code=500,
                                detail=f"Could not save the sequencing metrics: {e!s}")
        if backed:
            dm.evict(dataset_id)
            await dm.load_dataset(dataset_id)
        else:
            dm.saved_in_place(dataset_id, adaptor, output_path)
        return body
