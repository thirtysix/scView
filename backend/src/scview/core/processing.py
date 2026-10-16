"""Running a pipeline on a dataset without stalling the server.

Pipeline steps take from under a second to minutes (doublet detection on 14k
cells: ~70 s) and change the cached AnnData in place. They run in a worker
thread, so the server keeps answering while they work (the production backend
is one process: a step on the event loop stalled every request, including the
healthcheck). While a run works on a dataset:

- it holds the dataset's edit lock (``obs_writes.dataset_lock``), so labels, a
  kept resolution, pseudotime or a second run wait their turn;
- the dataset is marked busy, so ``DatasetManager.get_or_load_dataset`` makes
  other requests for it wait instead of reading a half-processed dataset.

Both are released, and the cached copy dropped, only when the worker thread has
stopped, even if the client went away first: the thread can't be interrupted
mid-step, and until it stops it may still be writing the derived file.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any, TypeVar

from fastapi import HTTPException

from scview.core.anndata_adaptor import AnnDataAdaptor
from scview.core.dataset_manager import DatasetManager
from scview.core.obs_writes import dataset_lock

logger = logging.getLogger(__name__)

T = TypeVar("T")


async def _claim(dm: DatasetManager, dataset_id: str) -> tuple[AnnDataAdaptor, Callable[[], None]]:
    """Take the dataset's lock, load it, and mark it busy. Returns the adaptor and
    the function that undoes all three (and drops the cached copy)."""
    lock = dataset_lock(dataset_id)
    await lock.acquire()
    try:
        adaptor = await dm.get_or_load_dataset(dataset_id)
    except BaseException:
        lock.release()
        raise
    if adaptor is None:
        lock.release()
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")
    dm.mark_busy(dataset_id)

    released = False

    def release() -> None:
        nonlocal released
        if released:
            return
        released = True
        dm.evict(dataset_id)  # steps changed it in place; the file is the truth
        dm.mark_idle(dataset_id)
        lock.release()

    return adaptor, release


async def run_exclusive(
    dm: DatasetManager, dataset_id: str, work: Callable[[AnnDataAdaptor], T]
) -> T:
    """Run ``work(adaptor)`` in a worker thread with the dataset to itself, then
    reload the dataset from disk."""
    adaptor, release = await _claim(dm, dataset_id)
    try:
        task = asyncio.ensure_future(asyncio.to_thread(work, adaptor))
    except BaseException:
        release()
        raise
    # Runs on the loop once the thread returns, even if this request is cancelled.
    task.add_done_callback(lambda _t: release())
    result = await asyncio.shield(task)
    await dm.load_dataset(dataset_id)
    return result


async def iterate_exclusive(
    dm: DatasetManager,
    dataset_id: str,
    make_events: Callable[[AnnDataAdaptor], Iterator[Any]],
) -> AsyncIterator[Any]:
    """Yield the items of the generator ``make_events(adaptor)``, which runs in a
    worker thread with the dataset to itself.

    When the consumer stops early (the client went away), the thread finishes the
    item it is working on and then closes the generator, so a pipeline run ends
    after its current step without saving. A run that reaches its end leaves the
    dataset reloaded from disk.
    """
    adaptor, release = await _claim(dm, dataset_id)
    loop = asyncio.get_running_loop()
    items: asyncio.Queue = asyncio.Queue()
    stop = threading.Event()

    def post(kind: str, value: Any = None) -> None:
        try:
            loop.call_soon_threadsafe(items.put_nowait, (kind, value))
        except RuntimeError:  # the loop is gone (server shutting down)
            pass

    def drive() -> None:
        try:
            gen = make_events(adaptor)
            try:
                for item in gen:
                    post("item", item)
                    if stop.is_set():
                        break
            finally:
                gen.close()
        except BaseException as exc:  # noqa: BLE001 — handed to the consumer
            post("error", exc)
        else:
            post("done")

    try:
        task = asyncio.ensure_future(asyncio.to_thread(drive))
    except BaseException:
        release()
        raise
    task.add_done_callback(lambda _t: release())

    finished = False
    try:
        while True:
            kind, value = await items.get()
            if kind == "item":
                yield value
            elif kind == "error":
                raise value
            else:
                finished = True
                break
    finally:
        if not finished:
            stop.set()
            dm.evict(dataset_id)  # nothing may serve the half-run copy
    # The thread has posted its last item; wait for it to return (and release).
    await asyncio.shield(task)
    await dm.load_dataset(dataset_id)
