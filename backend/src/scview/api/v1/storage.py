"""How a dataset's matrices are stored, and the offered rewrite by rows (``core/store_by_rows.py``).

``GET /datasets/{id}/storage``: which matrices are column-stored (read from the active file's
header: the one source of truth, a dataset stored by rows has none), whether that blocks
large-dataset mode, and what the rewrite costs (seconds, disk needed and free, memory).
``POST /datasets/{id}/storage/by-rows``: the rewrite as SSE (``start``, ``progress``, ``result``,
``complete`` or ``error``). It runs only when asked: nothing rewrites a dataset on open.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from scview.core import store_by_rows
from scview.core.dataset_manager import DatasetManager, DatasetMemoryError
from scview.core.obs_writes import dataset_edit
from scview.core.sse import keepalive
from scview.dependencies import get_dataset_manager

logger = logging.getLogger(__name__)
router = APIRouter()


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _known(dm: DatasetManager, dataset_id: str) -> None:
    if dm.get_dataset_info(dataset_id) is None or dm._resolve_h5ad(dataset_id) is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")


@router.get("/datasets/{dataset_id}/storage")
async def storage(dataset_id: str, dm: DatasetManager = Depends(get_dataset_manager)):
    """The dataset's column-stored matrices (``column_stored``, [] when stored by rows or
    dense), the ones large-dataset mode cannot read (``blocks_large_mode``), and the cost of
    storing it by rows: ``seconds``, ``disk_bytes`` against ``disk_free_bytes``
    (``enough_disk``), ``memory_bytes``, and ``cost`` / ``text`` saying them. Reads file headers only."""
    _known(dm, dataset_id)
    return await asyncio.to_thread(store_by_rows.plan, dm, dataset_id)


@router.post("/datasets/{dataset_id}/storage/by-rows")
async def store_by_rows_stream(dataset_id: str, dm: DatasetManager = Depends(get_dataset_manager)):
    """Store the dataset's column-stored matrices by rows: same values, dtype and order, the
    original upload untouched, the step recorded in History. Refused before anything is
    written: 409 while a run works on the dataset, 507 when the disk (the message names the GB
    needed and free) or the memory does not fit. Then SSE: ``start`` (the plan), ``progress``
    (``done`` of ``total`` stored values), ``result`` (``converted``, ``seconds``, the file's
    size before and after) and ``complete``, or ``error``. A dataset with nothing
    column-stored answers ``result`` with ``converted`` empty and writes nothing. A client
    that goes away does not stop it: the new file is swapped in when it is whole."""
    _known(dm, dataset_id)
    if dm.is_busy(dataset_id):
        raise HTTPException(status_code=409, detail="A run is working on this dataset: store it by rows "
                                                    "when the run has finished.")
    plan = await asyncio.to_thread(store_by_rows.plan, dm, dataset_id)
    if plan["files"] and not plan["enough_disk"]:
        raise HTTPException(status_code=507, detail=str(
            store_by_rows.NoDiskSpace(plan["disk_bytes"], plan["disk_free_bytes"])))

    async def events():
        loop = asyncio.get_running_loop()
        updates: asyncio.Queue = asyncio.Queue()

        def progress(done: int, total: int) -> None:
            loop.call_soon_threadsafe(updates.put_nowait, {"done": done, "total": total})

        try:
            async with dataset_edit(dataset_id) as held:
                # Asked again under the lock: an edit that just ended may have changed the file.
                todo = await asyncio.to_thread(store_by_rows.plan, dm, dataset_id)
                if todo["files"] and not todo["enough_disk"]:
                    raise store_by_rows.NoDiskSpace(todo["disk_bytes"], todo["disk_free_bytes"])
                yield _sse("start", todo)
                if todo["memory_bytes"]:
                    await dm.reserve(dataset_id, todo["memory_bytes"], what="Storing this dataset by rows",
                                     strict=True)
                    held.on_end(dm.hold_read(dataset_id, todo["memory_bytes"], "storing a dataset by rows"))
                work = asyncio.ensure_future(held.run(store_by_rows.run, dm, dataset_id, progress))
                # Also when the client has gone: the next open reads the new file.
                work.add_done_callback(lambda f: None if f.cancelled() or f.exception() or not f.result()["files"]
                                       else dm.evict(dataset_id))
                while not work.done():
                    waiter = asyncio.ensure_future(updates.get())
                    await asyncio.wait({work, waiter}, return_when=asyncio.FIRST_COMPLETED)
                    if waiter.done():
                        yield _sse("progress", waiter.result())
                    else:
                        waiter.cancel()
                result = work.result()
            yield _sse("result", result)
            yield _sse("complete", {})
        except (store_by_rows.NoDiskSpace, DatasetMemoryError) as e:
            yield _sse("error", {"error": str(e)})
        except Exception as e:  # noqa: BLE001 - said to the client, the server stays up
            logger.error("Store by rows failed for %s: %s", dataset_id, e, exc_info=True)
            yield _sse("error", {"error": f"Could not store the dataset by rows ({e!s}). It is as it was."})

    return StreamingResponse(keepalive(events()), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
