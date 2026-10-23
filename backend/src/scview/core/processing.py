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

Steps need the dataset in memory: one the adaptor opened backed (a file over
``SCVIEW_BACKED_MIN_MB``) is read in first, within the memory budget
(``core/working_copy.py``), in the worker thread. Unless the run goes to large-dataset
mode (``core/large_run.py``): with a plan, the mode is chosen first (``choose_mode``, on
``PipelineParams.chunked``), and a run in that mode reads the file in blocks itself, so
the cached copy is dropped instead and the work gets a fresh adaptor that has read
nothing (``control.mode`` tells the run endpoints which way to run).

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
import logging
import threading
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any, TypeVar

from fastapi import HTTPException

from scview.core import memory_budget, working_copy
from scview.core.anndata_adaptor import AnnDataAdaptor
from scview.core.dataset_manager import DatasetManager
from scview.core.obs_writes import dataset_lock
from scview.core.run_watchdog import RunControl

logger = logging.getLogger(__name__)

T = TypeVar("T")


def _give_back_memory() -> None:
    """Free the run's copy of the dataset before the saved file is read back
    (``memory_budget.release_memory``): where the container has little memory
    (WSL gives Podman/Docker half of a Windows PC's) the run's leftovers and the
    reloaded result must not sit side by side."""
    memory_budget.release_memory()


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


async def _make_run_room(dm: DatasetManager, dataset_id: str, adaptor: AnnDataAdaptor,
                         control: RunControl | None) -> AnnDataAdaptor:
    """Before a run: with a plan, the mode (``large_run.choose_mode``, set as
    ``control.mode``). In memory: room to read a backed dataset into memory
    (``working_copy``, a 507 when that cannot fit) and close other open datasets whose
    memory its estimated peak needs (the pre-step check, in the worker thread, can only
    stop the run). In large-dataset mode: the cached copy dropped, room for that mode's
    peak, and a fresh adaptor of the same file (it reads nothing until asked). Returns
    the adaptor the work gets."""
    need = None
    if control is not None and control.plan is not None:
        from scview.core import large_run, preflight

        steps, params = control.plan
        try:
            control.mode = await asyncio.to_thread(large_run.choose_mode, dm, dataset_id, adaptor,
                                                   steps, params)
        except ValueError:  # a chunked value that is not a mode: the endpoints check it first
            raise
        except Exception as e:  # never block a run on the estimate itself
            logger.warning("No mode estimate for %s (in memory): %s", dataset_id, e)
            control.mode = large_run.ModeChoice(False, "auto", f"No estimate ({e}): in memory.")
        logger.info("Run on %s: %s", dataset_id, control.mode.reason)
        if control.mode.chunked:
            path = adaptor.h5ad_path
            dm.evict(dataset_id)  # the run reads the file itself; nothing may serve a stale copy
            try:
                peak = await asyncio.to_thread(large_run.run_need, path, steps, params)
            except Exception as e:  # never block a run on the estimate itself
                logger.warning("No large-run estimate for %s: %s", dataset_id, e)
                peak = 0
            await dm.free_memory_for(peak, keep=dataset_id)
            return AnnDataAdaptor(path)
        need = control.mode.run_bytes
        if need is None:
            try:
                need = await asyncio.to_thread(preflight.run_need, adaptor.adata, steps, params)
            except Exception as e:  # never block a run on the estimate itself
                logger.warning("No run estimate for %s: %s", dataset_id, e)
    await working_copy.make_room(dm, dataset_id, adaptor, then=need or 0)
    return adaptor


def _chunked(control: RunControl | None) -> bool:
    return bool(getattr(getattr(control, "mode", None), "chunked", False))


def _in_memory(work: Callable[[AnnDataAdaptor], T],
               control: RunControl | None = None) -> Callable[[AnnDataAdaptor], T]:
    """``work``, called with the dataset in memory (``working_copy.in_memory``: a backed
    one is read in first, in the worker thread): steps need every value at hand. Not for
    a run in large-dataset mode, which reads the file in blocks itself."""

    def run(adaptor: AnnDataAdaptor) -> T:
        if not _chunked(control):
            working_copy.in_memory(adaptor)
        return work(adaptor)

    return run


# The pipeline run working on each dataset now (one at most: it holds the lock).
_runs: dict[str, RunControl] = {}


def active_run(dataset_id: str) -> RunControl | None:
    """The control of the pipeline run working on ``dataset_id``, if one is."""
    return _runs.get(dataset_id)


def cancel_run(dataset_id: str) -> RunControl | None:
    """Ask the run working on ``dataset_id`` to stop; None when none is."""
    control = _runs.get(dataset_id)
    if control is not None:
        control.cancel()
    return control


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
        adaptor = await _make_run_room(dm, dataset_id, adaptor, control)
        if _chunked(control):
            await asyncio.to_thread(_give_back_memory)  # the copy dropped for the run
        unwatch = _watch(dataset_id, control)
    except BaseException:
        claimed()
        raise

    def release() -> None:
        unwatch()
        claimed()

    try:
        task = asyncio.ensure_future(asyncio.to_thread(_in_memory(work, control), adaptor))
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
    registered for cancel, a consumer that stops early also cancels it, and its
    watchdog's warnings are yielded among the items as ``("watchdog", event)``,
    the pipeline's event shape.
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
        adaptor = await _make_run_room(dm, dataset_id, adaptor, control)
        if _chunked(control):
            await asyncio.to_thread(_give_back_memory)  # the copy dropped for the run
        unwatch = _watch(dataset_id, control, lambda event: post("item", ("watchdog", event)))
    except BaseException:
        claimed()
        raise

    def release() -> None:
        unwatch()
        claimed()

    def drive() -> None:
        try:
            gen = _in_memory(make_events, control)(adaptor)
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
            if control is not None:
                control.cancel()  # also ends a Leiden child the step is waiting on
            dm.evict(dataset_id)  # nothing may serve the half-run copy
    # The thread has posted its last item; wait for it to return (and release).
    await asyncio.shield(task)
    adaptor = None  # the thread is done with it  # noqa: F841
    await asyncio.to_thread(_give_back_memory)
    await dm.load_dataset(dataset_id)
