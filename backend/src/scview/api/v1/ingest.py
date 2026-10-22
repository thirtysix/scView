"""Ingestion engine HTTP endpoints — the guided multi-file import flow.

Stateful sessions: create one, upload files into it incrementally (one
experiment at a time, or a bulk drop), inspect the detection/bundle/validation
state and the multi-sample merge plan, set options, then commit into a dataset.
See ``docs/INGESTION_ENGINE.md`` §5.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from scview.core.dataset_manager import DatasetManager
from scview.core.ingestion.paths import UploadPathError, flat_name, junk_reason, safe_relative_path
from scview.core.ingestion.session import (
    IngestCommitError,
    IngestOptions,
    IngestSessionManager,
    StagingConflict,
)
from scview.dependencies import get_dataset_manager, get_ingest_session_manager

logger = logging.getLogger(__name__)

router = APIRouter()


def _require(mgr: IngestSessionManager, sid: str) -> None:
    if not mgr.exists(sid):
        raise HTTPException(status_code=404, detail="Ingest session not found.")


@router.post("/ingest/session", status_code=201)
async def create_session(mgr: IngestSessionManager = Depends(get_ingest_session_manager)):
    return {"session_id": mgr.create_session()}


@router.post("/ingest/session/{sid}/files")
async def add_files(
    sid: str,
    files: list[UploadFile] = File(...),
    paths: list[str] | None = Form(None),
    mgr: IngestSessionManager = Depends(get_ingest_session_manager),
):
    """Stage files. ``paths`` (one per file, in order): each file's path inside an
    uploaded folder (the browser's ``webkitRelativePath``), kept so same-named files
    in different subfolders stay apart; without it each file is staged by its name.
    A path that is unsafe is a 400; two files on one path (in this request, or a
    different file where one is staged) a 409, and nothing of the request is staged.
    Files scView never reads (BAM, Loupe, HTML report) are skipped and listed."""
    _require(mgr, sid)
    if paths is not None and len(paths) != len(files):
        raise HTTPException(status_code=400, detail=(
            f"{len(paths)} paths were sent for {len(files)} files: send one path per file."))
    try:
        rels = [safe_relative_path(paths[i] if paths is not None else flat_name(f.filename))
                for i, f in enumerate(files)]
    except UploadPathError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    items: list[tuple[str, Path, str]] = []
    skipped: list[dict] = []
    try:
        for f, rel in zip(files, rels):
            reason = junk_reason(rel)
            if reason:
                skipped.append({"path": rel, "reason": reason, "bytes": await _drain(f)})
                continue
            partial = mgr.partial_path(sid)
            items.append((rel, partial, await _write(f, partial)))
        await asyncio.to_thread(mgr.stage, sid, items)
    except StagingConflict as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    finally:
        for _, partial, _ in items:  # whatever stage did not move
            partial.unlink(missing_ok=True)
    mgr.record_skipped(sid, skipped)
    return await asyncio.to_thread(mgr.get_state, sid)


async def _write(f: UploadFile, dest: Path) -> str:
    """Stream an upload to ``dest``; its sha256. Nothing is left behind on failure."""
    digest = hashlib.sha256()
    try:
        with open(dest, "wb") as fh:
            while chunk := await f.read(1024 * 1024):
                digest.update(chunk)
                fh.write(chunk)
    except BaseException:
        dest.unlink(missing_ok=True)
        raise
    return digest.hexdigest()


async def _drain(f: UploadFile) -> int:
    """The size of an upload that is not kept."""
    if f.size is not None:
        return int(f.size)
    total = 0
    while chunk := await f.read(1024 * 1024):
        total += len(chunk)
    return total


@router.get("/ingest/session/{sid}")
async def get_session(
    sid: str, mgr: IngestSessionManager = Depends(get_ingest_session_manager)
):
    _require(mgr, sid)
    return mgr.get_state(sid)


@router.get("/ingest/session/{sid}/merge-plan")
async def get_merge_plan(
    sid: str, mgr: IngestSessionManager = Depends(get_ingest_session_manager)
):
    _require(mgr, sid)
    plan = mgr.merge_plan(sid)
    if plan is None:
        return {"is_merge": False}
    return plan.model_dump(mode="json")


@router.post("/ingest/session/{sid}/options")
async def set_options(
    sid: str,
    options: IngestOptions,
    mgr: IngestSessionManager = Depends(get_ingest_session_manager),
):
    _require(mgr, sid)
    mgr.set_options(sid, options)
    return mgr.get_state(sid)


@router.post("/ingest/session/{sid}/commit")
async def commit_session(
    sid: str,
    mgr: IngestSessionManager = Depends(get_ingest_session_manager),
    dm: DatasetManager = Depends(get_dataset_manager),
):
    _require(mgr, sid)
    try:
        dataset_id = await mgr.commit(sid, dm)
    except IngestCommitError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return {"dataset_id": dataset_id}


@router.delete("/ingest/session/{sid}")
async def discard_session(
    sid: str, mgr: IngestSessionManager = Depends(get_ingest_session_manager)
):
    _require(mgr, sid)
    mgr.discard(sid)
    return {"status": "discarded"}
