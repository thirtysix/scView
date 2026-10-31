"""Xenium transcripts endpoints (``core/transcripts.py``).

``POST /datasets/{id}/spatial/transcripts?name=transcripts.parquet``: the file as the raw request
body, streamed to the dataset's folder and renamed there when whole. ``POST
.../spatial/transcripts/index-stream``: builds the store as SSE, one at a time per dataset; a client
that goes away stops it after its row group, with nothing kept. The parquet is deleted once indexed.
``GET .../spatial/transcripts/status``: what is attached, the genes, the estimate (``?size=`` a file
not yet sent). ``GET .../spatial/transcripts?genes=&x0=&x1=&y0=&y1=&limit=``: the points in a view,
binary (``transcripts.encode``), with ``X-Transcripts-Total`` (in the view), ``X-Transcripts-Shown``
and ``X-Transcripts-Per-Gene``. ``DELETE`` the same path removes it all.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from collections.abc import Iterator
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from fastapi.responses import StreamingResponse
from starlette.requests import ClientDisconnect

from scview.config import Settings
from scview.core import raw_upload, spatial
from scview.core import transcripts as tr
from scview.core.dataset_manager import DatasetManager
from scview.core.run_watchdog import RunControl, RunWatchdog
from scview.core.sse import keepalive
from scview.dependencies import get_dataset_manager, get_settings_dep

logger = logging.getLogger(__name__)

router = APIRouter()

_DONE = object()
_running: set[str] = set()  # datasets with a build streaming
_uploading: set[str] = set()
_WHAT = "Indexing the transcripts"


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _http(e: tr.TranscriptsError) -> HTTPException:
    return HTTPException(status_code=e.status, detail=str(e))


async def _section(dm: DatasetManager, dataset_id: str):
    """(the adaptor, the section's bounds); a dataset with no single section is refused."""
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")
    info = spatial.info(adaptor.adata)
    if info is None:
        raise HTTPException(status_code=422, detail="This dataset has no spatial coordinates, so transcripts have "
                                                    "nothing to sit on: import the Xenium run's folder.")
    if len(info["libraries"]) != 1:
        raise HTTPException(status_code=422, detail=f"This dataset merges {len(info['libraries'])} sections and a "
                                                    "transcripts file holds one: import the run on its own to "
                                                    "attach its transcripts.")
    return adaptor, info["libraries"][0]["bounds"]


@router.get("/datasets/{dataset_id}/spatial/transcripts/status")
async def get_status(dataset_id: str, size: int | None = Query(default=None, ge=1),
                     settings: Settings = Depends(get_settings_dep),
                     dm: DatasetManager = Depends(get_dataset_manager)):
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")
    n_genes = int(adaptor.adata.n_vars)

    def work() -> dict[str, Any]:
        out = tr.status(settings.DATA_DIR, dataset_id, n_genes)
        out["disk_free"] = tr.disk_free(tr.store_dir(settings.DATA_DIR, dataset_id))
        if size is not None:
            out["estimate_for_size"] = tr.estimate_for_size(size, n_genes)
        return out

    return {**await asyncio.to_thread(work), "running": dataset_id in _running}


def _check_head(head: bytes) -> None:
    if not head.startswith(tr.PARQUET_MAGIC):
        raise raw_upload.UploadRefused("That file is not parquet: send transcripts.parquet from the Xenium output "
                                       "folder.")


@router.post("/datasets/{dataset_id}/spatial/transcripts")
async def upload_transcripts(dataset_id: str, request: Request, name: str = Query(max_length=512),
                             settings: Settings = Depends(get_settings_dep),
                             dm: DatasetManager = Depends(get_dataset_manager)):
    """The file as the raw body: ``curl --data-binary @transcripts.parquet ...?name=...``. On disk,
    never in memory; refused before it travels when the disk cannot hold it and its index."""
    if not name.lower().endswith(".parquet"):
        raise HTTPException(status_code=422, detail=f"'{name}' is not a parquet file: send transcripts.parquet from "
                                                    "the Xenium output folder.")
    raw_upload.refuse_multipart(request)
    adaptor, _ = await _section(dm, dataset_id)
    n_genes = int(adaptor.adata.n_vars)
    del adaptor
    if dataset_id in _running:
        raise HTTPException(status_code=409, detail="The transcripts of this dataset are being indexed: wait for it, "
                                                    "or stop it first.")
    if dataset_id in _uploading:
        raise HTTPException(status_code=409, detail="A transcripts file of this dataset is already being uploaded.")
    target = tr.upload_path(settings.DATA_DIR, dataset_id)
    part = target.with_name(target.name + ".part")
    length = request.headers.get("content-length")
    if length and length.isdigit():
        est = tr.estimate_for_size(int(length), n_genes)
        need = int(length) + est["disk_bytes"]
        free = await asyncio.to_thread(tr.disk_free, target.parent)
        if need > free:
            raise HTTPException(status_code=507, detail=f"This file and its index need about {need / 1024**3:.1f} GB "
                                                        f"of disk and {free / 1024**3:.1f} GB are free: free some "
                                                        "space and send it again.")
    _uploading.add(dataset_id)
    try:
        await asyncio.to_thread(part.parent.mkdir, parents=True, exist_ok=True)
        await raw_upload.stream_to_file(request, part, settings.max_upload_bytes, settings.MAX_UPLOAD_SIZE_MB, _check_head)
        await asyncio.to_thread(tr.read_plan, part)
        await asyncio.to_thread(part.replace, target)
    except tr.TranscriptsError as e:
        raise _http(e)
    except raw_upload.UploadRefused as e:
        raise HTTPException(status_code=e.status, detail=str(e))
    except ClientDisconnect:
        raise HTTPException(status_code=400, detail="The upload stopped before the end: send the file again.")
    except OSError as e:
        logger.warning("Could not store %s for %s: %s", name, dataset_id, e)
        raise HTTPException(status_code=507, detail=f"Could not store {name} on this machine's disk: free some "
                                                    "space and send it again.")
    finally:
        part.unlink(missing_ok=True)
        _uploading.discard(dataset_id)
    return await asyncio.to_thread(tr.status, settings.DATA_DIR, dataset_id, n_genes)


@router.post("/datasets/{dataset_id}/spatial/transcripts/index-stream")
async def index_stream(dataset_id: str, settings: Settings = Depends(get_settings_dep),
                       dm: DatasetManager = Depends(get_dataset_manager)):
    """Events: ``start`` (the estimate), ``progress``, ``watchdog``, ``result`` (the summary), then
    ``complete``, or ``error``. The store replaces any earlier one only at the end."""
    adaptor, bounds = await _section(dm, dataset_id)
    n_genes = int(adaptor.adata.n_vars)
    del adaptor
    if dataset_id in _running:
        raise HTTPException(status_code=409, detail="The transcripts of this dataset are already being indexed.")
    data_dir = settings.DATA_DIR
    source = tr.upload_path(data_dir, dataset_id)
    if not source.is_file():
        raise HTTPException(status_code=422, detail="No transcripts file is waiting: send transcripts.parquet first.")
    try:
        plan = await asyncio.to_thread(tr.read_plan, source)
    except tr.TranscriptsError as e:
        raise _http(e)
    est = tr.estimate(plan["rows"], plan["largest_row_group"], n_genes)
    free = await asyncio.to_thread(tr.disk_free, source.parent)
    if est["disk_bytes"] > free:
        raise HTTPException(status_code=507, detail=f"The index needs about {est['disk_bytes'] / 1024**3:.1f} GB of "
                                                    f"disk and {free / 1024**3:.1f} GB are free: free some space and "
                                                    "build it again.")
    await dm.reserve(dataset_id, est["peak_bytes"], what=_WHAT)
    matrix = await asyncio.to_thread(tr.find_matrix, data_dir, dataset_id)

    async def events():
        if dataset_id in _running:  # another request started between the check and here
            yield _sse("error", {"error": "The transcripts of this dataset are already being indexed."})
            return
        _running.add(dataset_id)
        stop = threading.Event()
        watchdog = RunWatchdog()
        control = RunControl(watchdog=watchdog, before_step=None)
        gen: Iterator[dict[str, Any]] = tr.iter_index(source, tr.store_path(data_dir, dataset_id), bounds,
                                                      stop=stop.is_set, matrix=matrix)
        sent = 0
        pending: asyncio.Future | None = None
        dm.reserve_run(dataset_id, est["peak_bytes"])

        def step_next():
            with control.step(tr.STEP):
                return next(gen, _DONE)

        def finish(fut: asyncio.Future | None = None) -> None:
            if fut is not None and not fut.cancelled():
                fut.exception()  # retrieved: a stopped run is not an error
            gen.close()
            watchdog.stop()
            dm.release_run(dataset_id)
            _running.discard(dataset_id)

        try:
            watchdog.start()
            yield _sse("start", {"estimate": est, "row_groups": plan["row_groups"]})
            while True:
                # Shielded, so a client going away cancels this wait, not the thread's future,
                # whose callback then closes the run.
                pending = asyncio.ensure_future(asyncio.to_thread(step_next))
                item = await asyncio.shield(pending)
                pending = None
                for warning in watchdog.events[sent:]:
                    yield _sse("watchdog", warning)
                sent = len(watchdog.events)
                if item is _DONE:
                    break
                if item["type"] == "progress":
                    yield _sse("progress", {k: v for k, v in item.items() if k != "type"})
                    continue
                await asyncio.to_thread(source.unlink, missing_ok=True)
                yield _sse("result", {**item["summary"], "resources": watchdog.summary()})
            yield _sse("complete", {})
        except tr.TranscriptsCancelled:
            return
        except tr.TranscriptsError as e:
            yield _sse("error", {"error": str(e)})
        except OSError as e:
            logger.warning("Transcript index failed for %s: %s", dataset_id, e)
            yield _sse("error", {"error": f"The transcript index could not be written ({e}): free some disk space and "
                                          "build it again."})
        except Exception as e:  # noqa: BLE001 - surface to the client, keep the server up
            logger.error("Transcript index failed for %s: %s", dataset_id, e, exc_info=True)
            yield _sse("error", {"error": f"Indexing the transcripts failed: {e}"})
        finally:
            stop.set()  # a thread mid-row-group stops before the next one
            control.cancel()
            if pending is not None and not pending.done():
                pending.add_done_callback(finish)
            else:
                finish()

    return StreamingResponse(keepalive(events()), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.get("/datasets/{dataset_id}/spatial/transcripts")
async def get_transcripts(
    dataset_id: str,
    genes: str = Query(max_length=2048),
    x0: float | None = None,
    x1: float | None = None,
    y0: float | None = None,
    y1: float | None = None,
    limit: int = Query(default=tr.DEFAULT_LIMIT, ge=1, le=tr.MAX_LIMIT),
    settings: Settings = Depends(get_settings_dep),
):
    """The points of up to 8 genes (comma-separated) in a view; all four bounds, or none."""
    path = tr.store_path(settings.DATA_DIR, dataset_id)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="This dataset has no transcripts attached: send its "
                                                    "transcripts.parquet first.")
    bounds = [x0, y0, x1, y1]
    if any(b is None for b in bounds) and not all(b is None for b in bounds):
        raise HTTPException(status_code=422, detail="Give all of x0, x1, y0, y1, or none for the whole section.")
    box = None if bounds[0] is None else bounds
    if box is not None and (box[2] < box[0] or box[3] < box[1]):
        raise HTTPException(status_code=422, detail="x1 and y1 must not be below x0 and y0.")
    wanted = [g.strip() for g in genes.split(",") if g.strip()]

    def work() -> tuple[bytes, tr.View]:
        view = tr.query(path, wanted, box, limit)
        return tr.encode(view), view

    try:
        content, view = await asyncio.to_thread(work)
    except tr.TranscriptsError as e:
        raise _http(e)
    return Response(content=content, media_type="application/octet-stream", headers={
        "X-Transcripts-Total": str(view.total), "X-Transcripts-Shown": str(len(view.gene)),
        "X-Transcripts-Per-Gene": ",".join(map(str, view.per_gene)),
        "X-Transcripts-Seconds": f"{view.seconds:.3f}",
    })


@router.delete("/datasets/{dataset_id}/spatial/transcripts")
async def delete_transcripts(dataset_id: str, settings: Settings = Depends(get_settings_dep)):
    if dataset_id in _running or dataset_id in _uploading:
        raise HTTPException(status_code=409, detail="The transcripts of this dataset are being written: stop that first.")
    await asyncio.to_thread(tr.remove, settings.DATA_DIR, dataset_id)
    return {"deleted": True}
