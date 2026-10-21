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

A pipeline run passes its ``RunControl`` (``core/run_watchdog.py``): while the
run works it is registered for ``cancel_run`` (``POST .../assessment/cancel``)
and its watchdog samples memory, sending warnings into a streamed run's events.
Cancel takes effect between steps (a Leiden child is ended at once); the derived
file is written only after the last step, so a cancelled run saves nothing, and
the cached copy, which steps changed in place, is dropped as after any run: the
dataset reopens as it was before the run.
"""

from __future__ import annotations

import asyncio
import ctypes
import gc
import logging
import threading
import time
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any, TypeVar

from fastapi import HTTPException

from scview.core.anndata_adaptor import AnnDataAdaptor
from scview.core.dataset_manager import DatasetManager
from scview.core.obs_writes import dataset_lock
from scview.core.run_watchdog import RunControl

logger = logging.getLogger(__name__)

T = TypeVar("T")


def _give_back_memory() -> None:
    """Free the run's copy of the dataset before the saved file is read back.

    A run leaves its arrays to the garbage collector, and glibc keeps freed memory
    instead of returning it: after Kang the process held 3.1 GB, where a fresh load
    of the result takes 1.25 GB; this brings it to 2.0 GB. That matters where the
    container has little memory (WSL gives Podman/Docker half of a Windows PC's).
    The arrays are in reference cycles (an AnnData and its .raw), so trimming alone
    frees nothing. The collection holds the GIL even in a worker thread: 0.25 s
    after a Kang run, once per run."""
    t = time.perf_counter()
    gc.collect()
    logger.info("Freed the run's memory (garbage collection %.0f ms)", 1000 * (time.perf_counter() - t))
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):  # not glibc
        pass


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


# The pipeline run working on each dataset now (one at most: it holds the lock).
_runs: dict[str, RunControl] = {}


def active_run(dataset_id: str) -> RunControl | None:
    """The control of the pipeline run working on ``dataset_id``, if one is."""
    return _runs.get(dataset_id)


def _watch(dataset_id: str, control: RunControl | None, emit=None) -> Callable[[], None]:
    """Register ``control`` for cancel and start its watchdog; returns the undo
    (run on the loop once the worker thread has ended, errors included)."""
    if control is None:
        return lambda: None
    _runs[dataset_id] = control
    control.watchdog.emit = emit
    try:
        control.watchdog.start()
    except BaseException:
        del _runs[dataset_id]
        raise

    def unwatch() -> None:
        control.watchdog.stop()
        if _runs.get(dataset_id) is control:
            del _runs[dataset_id]

    return unwatch


async def run_exclusive(
    dm: DatasetManager,
    dataset_id: str,
    work: Callable[[AnnDataAdaptor], T],
    control: RunControl | None = None,
) -> T:
    """Run ``work(adaptor)`` in a worker thread with the dataset to itself, then
    reload the dataset from disk. ``control``: the pipeline run ``work`` makes
    with it, registered for cancel and watched (no events: nothing streams)."""
    adaptor, claimed = await _claim(dm, dataset_id)
    try:
        unwatch = _watch(dataset_id, control)
    except BaseException:
        claimed()
        raise

    def release() -> None:
        unwatch()
        claimed()

    try:
        task = asyncio.ensure_future(asyncio.to_thread(work, adaptor))
    except BaseException:
        release()
        raise
    # Runs on the loop once the thread returns, even if this request is cancelled.
    task.add_done_callback(lambda _t: release())
    result = await asyncio.shield(task)
    del adaptor
    await asyncio.to_thread(_give_back_memory)
    await dm.load_dataset(dataset_id)
    return result


async def iterate_exclusive(
    dm: DatasetManager,
    dataset_id: str,
    make_events: Callable[[AnnDataAdaptor], Iterator[Any]],
    control: RunControl | None = None,
) -> AsyncIterator[Any]:
    """Yield the items of the generator ``make_events(adaptor)``, which runs in a
    worker thread with the dataset to itself.

    When the consumer stops early (the client went away), the thread finishes the
    item it is working on and then closes the generator, so a pipeline run ends
    after its current step without saving. A run that reaches its end leaves the
    dataset reloaded from disk. With ``control`` (the pipeline run's), the run is
    registered for cancel and its watchdog's warnings are yielded among the items
    as ``("watchdog", event)``, the pipeline's event shape.
    """
    adaptor, claimed = await _claim(dm, dataset_id)
    loop = asyncio.get_running_loop()
    items: asyncio.Queue = asyncio.Queue()
    stop = threading.Event()

    def post(kind: str, value: Any = None) -> None:
        try:
            loop.call_soon_threadsafe(items.put_nowait, (kind, value))
        except RuntimeError:  # the loop is gone (server shutting down)
            pass

    try:
        unwatch = _watch(dataset_id, control, lambda event: post("item", ("watchdog", event)))
    except BaseException:
        claimed()
        raise

    def release() -> None:
        unwatch()
        claimed()

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
    adaptor = None  # the thread is done with it  # noqa: F841
    await asyncio.to_thread(_give_back_memory)
    await dm.load_dataset(dataset_id)
