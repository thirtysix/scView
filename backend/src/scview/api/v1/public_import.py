"""Import a public dataset from CELLxGENE Discover (``core/public_import.py``).

``GET /public-import/status``; ``GET /public-import/discover/datasets`` (search the
cached listing); ``GET /public-import/discover/datasets/{version_id}`` (one version,
with the disk check); ``POST /public-import/discover/import-stream`` (download +
register, SSE); ``POST /public-import/jobs/{job_id}/cancel``. All 403 unless
``PUBLIC_IMPORT_ENABLED`` (default: on in private mode, off in public mode).
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import threading
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from scview.config import Settings
from scview.core import public_import as pi
from scview.core.dataset_manager import DatasetManager
from scview.core.sse import keepalive
from scview.dependencies import get_dataset_manager, get_settings_dep

logger = logging.getLogger(__name__)

router = APIRouter()

_DONE = object()
_catalogs: dict[str, pi.DiscoverCatalog] = {}
# Imports running now: job id -> its cancel flag.
_jobs: dict[str, threading.Event] = {}


def catalog(settings: Settings) -> pi.DiscoverCatalog:
    c = _catalogs.get(settings.DATA_DIR)
    if c is None:
        c = _catalogs[settings.DATA_DIR] = pi.DiscoverCatalog(settings.DATA_DIR)
    return c


def _enabled(settings: Settings) -> None:
    if not settings.public_import_enabled:
        raise HTTPException(status_code=403, detail=(
            "Importing public datasets is turned off on this server (PUBLIC_IMPORT_ENABLED)."))


def _http(e: pi.PublicImportError) -> HTTPException:
    return HTTPException(status_code=e.status, detail=str(e))


@router.get("/public-import/status")
async def status(settings: Settings = Depends(get_settings_dep)):
    return {"enabled": settings.public_import_enabled, "source": pi.SOURCE,
            "api": pi.API_BASE, "files": pi.FILES_BASE, "license": pi.LICENSE,
            "license_url": pi.LICENSE_URL}


@router.get("/public-import/discover/datasets")
async def search_datasets(
    q: str | None = Query(default=None, max_length=200),
    organism: str | None = Query(default=None, max_length=100),
    tissue: str | None = Query(default=None, max_length=200),
    disease: str | None = Query(default=None, max_length=200),
    assay: str | None = Query(default=None, max_length=200),
    max_mb: float | None = Query(default=None, gt=0),
    sort: str = Query(default="cells", pattern="^(cells|size|title|newest)$"),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    refresh: bool = False,
    settings: Settings = Depends(get_settings_dep),
):
    """Search Discover's dataset listing (fetched once a day, then cached)."""
    _enabled(settings)
    try:
        listing = await asyncio.to_thread(catalog(settings).listing, refresh)
    except pi.PublicImportError as e:
        raise _http(e)
    max_bytes = int(max_mb * 1024 * 1024) if max_mb else None
    result = await asyncio.to_thread(pi.search, listing.datasets, q, organism, tissue, disease,
                                     assay, max_bytes, sort, limit, offset)
    return {**result, "fetched_at": listing.fetched_at, "n_datasets": len(listing.datasets)}


def _disk(settings: Settings, size: int) -> dict[str, Any]:
    data_dir = Path(settings.DATA_DIR)
    free = shutil.disk_usage(data_dir).free if data_dir.exists() else None
    need = 2 * size + pi.DISK_MARGIN
    return {"free_bytes": free, "need_bytes": need, "ok": free is None or free >= need}


@router.get("/public-import/discover/datasets/{version_id}")
async def dataset_detail(
    version_id: str,
    dataset_id: str | None = None,
    settings: Settings = Depends(get_settings_dep),
):
    """One dataset version, with the URL it downloads from and the disk it needs."""
    _enabled(settings)
    try:
        rec = await asyncio.to_thread(catalog(settings).find, version_id, dataset_id)
    except pi.PublicImportError as e:
        raise _http(e)
    return {**rec, "download_url": pi.file_url(rec["dataset_version_id"]),
            "disk": _disk(settings, rec["size_bytes"]), "license_url": pi.LICENSE_URL}


class ImportBody(BaseModel):
    dataset_version_id: str
    dataset_id: str | None = None  # needed only for a version no longer current
    name: str | None = None


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


@router.post("/public-import/discover/import-stream")
async def import_stream(
    body: ImportBody,
    settings: Settings = Depends(get_settings_dep),
    dm: DatasetManager = Depends(get_dataset_manager),
):
    """Download one dataset version into the library, streaming progress (SSE).

    Events: ``job`` (its id, for cancel), ``stage``, ``progress``, ``memory``, then
    ``complete`` (the dataset id) once it is registered and opened, or ``error`` /
    ``cancelled``. Leaving the stream cancels the import; nothing is kept then."""
    _enabled(settings)
    try:
        rec = await asyncio.to_thread(catalog(settings).find, body.dataset_version_id, body.dataset_id)
        await asyncio.to_thread(pi.check_disk, Path(settings.DATA_DIR), rec["size_bytes"])
    except pi.PublicImportError as e:
        raise _http(e)
    job_id = uuid.uuid4().hex
    dataset_id = str(uuid.uuid4())
    cancel = threading.Event()
    _jobs[job_id] = cancel
    gen = pi.run_import(rec, Path(settings.DATA_DIR), dataset_id, cancel, name=body.name)

    async def events():
        registered = False
        try:
            yield _sse("job", {"job_id": job_id, "dataset_version_id": rec["dataset_version_id"],
                               "url": pi.file_url(rec["dataset_version_id"]),
                               "size_bytes": rec["size_bytes"]})
            while True:
                item = await asyncio.to_thread(next, gen, _DONE)
                if item is _DONE:
                    break
                registered = registered or item[0] == "registered"
                if item[0] != "registered":
                    yield _sse(*item)
            await dm.load_dataset(dataset_id)
            yield _sse("complete", {"dataset_id": dataset_id})
        except pi.ImportCancelled:
            yield _sse("cancelled", {"message": "Import cancelled; nothing was kept."})
        except pi.PublicImportError as e:
            yield _sse("error", {"error": str(e), "status": e.status})
        except Exception as e:  # noqa: BLE001 — surface to the client, keep the server up
            logger.error("Public import %s failed: %s", rec["dataset_version_id"], e, exc_info=True)
            if registered:  # the file is in; only opening it failed
                yield _sse("error", {"error": f"Imported, but it could not be opened: {e}",
                                     "dataset_id": dataset_id})
            else:
                yield _sse("error", {"error": str(e)})
        finally:
            cancel.set()  # a client that left stops the download
            _jobs.pop(job_id, None)
            await asyncio.to_thread(gen.close)

    return StreamingResponse(keepalive(events()), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.post("/public-import/jobs/{job_id}/cancel")
async def cancel_job(job_id: str, settings: Settings = Depends(get_settings_dep)):
    _enabled(settings)
    flag = _jobs.get(job_id)
    if flag is None:
        raise HTTPException(status_code=404, detail="No import with that id is running.")
    flag.set()
    return {"cancelled": True}
