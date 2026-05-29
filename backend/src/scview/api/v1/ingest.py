"""Ingestion engine HTTP endpoints — the guided multi-file import flow.

Stateful sessions: create one, upload files into it incrementally (one
experiment at a time, or a bulk drop), inspect the detection/bundle/validation
state and the multi-sample merge plan, set options, then commit into a dataset.
See ``docs/INGESTION_ENGINE.md`` §5.
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from scview.core.dataset_manager import DatasetManager
from scview.core.ingestion.session import (
    IngestCommitError,
    IngestOptions,
    IngestSessionManager,
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
    mgr: IngestSessionManager = Depends(get_ingest_session_manager),
):
    _require(mgr, sid)
    dest_dir = mgr.files_dir(sid)
    for f in files:
        # basename only — never trust the client's path separators.
        dest = dest_dir / Path(f.filename or "unnamed").name
        with open(dest, "wb") as fh:
            while chunk := await f.read(1024 * 1024):
                fh.write(chunk)
    return mgr.get_state(sid)


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
