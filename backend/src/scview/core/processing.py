"""Running a pipeline on a dataset without stalling the server.

Pipeline steps take from under a second to minutes (doublet detection on 14k
cells: ~70 s) and change the cached AnnData in place. They run in a worker
thread, so the server keeps answering while they work (the production backend
is one process: a step on the event loop stalled every request, including the
healthcheck). While a run works on a dataset:

- it holds the dataset's edit lock (``obs_writes.dataset_lock``), so labels, a
  kept resolution, pseudotime or a second run wait their turn;
- the dataset is marked busy, so ``DatasetManager.get_or_load_dataset`` makes
  other requests for it wait instead of reading a half-processed dataset;
- its estimated peak is reserved (``DatasetManager.reserve_run``), so a load of
  another dataset counts what the run has still to grow by as used, and is refused
  (a 507 naming the run) rather than let in on the memory free at that moment.

Both are released, and the cached copy dropped, only when the worker thread has
stopped, even if the client went away first: the thread can't be interrupted
mid-step, and until it stops it may still be writing the derived file.

Steps need the dataset in memory: one the adaptor opened backed (a file over
``SCVIEW_BACKED_MIN_MB``) is read in first, within the memory budget
(``core/working_copy.py``), in the worker thread. Unless the run goes to large-dataset
mode (``core/large_run.py``): with a plan, the mode is chosen first (``choose_mode``, on
``PipelineParams.chunked``), and a run in that mode reads the file in blocks itself, so
the cached copy is dropped instead and the work gets a fresh adaptor that has read
nothing (``control.mode`` tells the run endpoints which way to run). Work that needs only
obs, obsm, obsp and uns (PAGA, pseudotime: ``run_exclusive(light=True)``) gets the same
on a large dataset and reads it without its matrices (``large_run.read_light``). Any
other work on a large dataset is refused (422): it would read the matrices into memory.

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
        dm.release_run(dataset_id)  # its growth is over, however it ended
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
    peak, and a fresh adaptor of the same file (it reads nothing until asked). Either
    way the run's expected growth is reserved (``_reserve_run``) until it ends. Returns
    the adaptor the work gets."""
    need = None
    if control is not None:  # its pre-step check counts what the others still grow by
        control.other_growth = lambda used: dm.outstanding(used, dataset_id)
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
            path = getattr(adaptor, "h5ad_path", None)
            large = await asyncio.to_thread(large_run.large_info, path) is not None
            control.mode = large_run.ModeChoice(large, "auto", f"No estimate ({e}).")
        logger.info("Run on %s: %s", dataset_id, control.mode.reason)
        try:  # each step's estimated time, for the minute notes while it runs
            control.watchdog.expected = await asyncio.to_thread(
                preflight.step_times, dm, dataset_id, steps, params, control.mode.chunked)
        except Exception as e:  # never block a run on the estimate itself
            logger.warning("No step times for %s: %s", dataset_id, e)
        if control.mode.chunked:
            why = large_run.refused(steps, params)  # a step this mode does not run: before anything
            if why is not None:
                raise HTTPException(status_code=422, detail=why)
            path = adaptor.h5ad_path
            dm.evict(dataset_id)  # the run reads the file itself; nothing may serve a stale copy
            try:
                peak = await asyncio.to_thread(large_run.run_need, path, steps, params)
            except Exception as e:  # never block a run on the estimate itself
                logger.warning("No large-run estimate for %s: %s", dataset_id, e)
                peak = 0
            await dm.free_memory_for(peak, keep=dataset_id)
            await _reserve_run(dm, dataset_id, control, peak,
                               lambda: large_run.run_step_needs(path, steps, params))
            return AnnDataAdaptor(path)
        need = control.mode.run_bytes
        if need is None:
            try:
                need = await asyncio.to_thread(preflight.run_need, adaptor.adata, steps, params)
            except Exception as e:  # never block a run on the estimate itself
                logger.warning("No run estimate for %s: %s", dataset_id, e)
    else:  # work that reads the dataset into memory: never a large one
        from scview.core import large_run

        path = getattr(adaptor, "h5ad_path", None)
        if await asyncio.to_thread(large_run.large_info, path) is not None:
            raise HTTPException(status_code=422, detail=LARGE_REFUSED)
    await working_copy.make_room(dm, dataset_id, adaptor, then=need or 0)
    if need:
        from scview.core import preflight

        # The growth from now: a backed dataset read in (``need_bytes`` counts it when
        # the mode was estimated), then the run's peak over the copy.
        growth = control.mode.need_bytes
        if growth is None:
            try:
                growth = need + await asyncio.to_thread(working_copy.copy_bytes, adaptor)
            except Exception:  # noqa: BLE001 - the run's own part then
                growth = need
        copy = max(0, growth - need)
        adata = adaptor.adata
        await _reserve_run(dm, dataset_id, control, growth, lambda: {
            s: copy + b for s, b in preflight.run_step_needs(adata, steps, params).items()})
    return adaptor


async def _reserve_run(dm: DatasetManager, dataset_id: str, control: RunControl, need: int,
                       step_needs: Callable[[], dict[str, int]]) -> None:
    """Reserve the run's expected growth (``need`` bytes from now) with the manager until
    it ends, so a load of another dataset meanwhile counts it as used. With the per-step
    needs (``step_needs``, in a worker thread; the whole need when they cannot be
    estimated) the reservation shrinks to the peak of the steps still to run: the
    watchdog records each finished step, so this reads nothing from the run's thread."""
    if need <= 0:
        return
    remaining = None
    try:
        by_step = await asyncio.to_thread(step_needs)
    except Exception as e:  # noqa: BLE001 - never block a run on the estimate itself
        logger.warning("No per-step estimate for the run on %s: %s", dataset_id, e)
        by_step = {}
    if by_step:
        done = control.watchdog.steps  # filled as each step finishes

        def remaining() -> int:
            return max((b for s, b in by_step.items() if s not in done), default=0)

    dm.reserve_run(dataset_id, need, remaining)


def _chunked(control: RunControl | None) -> bool:
    return bool(getattr(getattr(control, "mode", None), "chunked", False))


async def _for_light_work(dm: DatasetManager, dataset_id: str,
                          adaptor: AnnDataAdaptor) -> AnnDataAdaptor | None:
    """For work that needs no matrix on a large dataset: its cached copy dropped and a
    fresh adaptor (it reads nothing until asked); None for any other dataset."""
    from scview.core import large_run

    path = adaptor.h5ad_path
    if await asyncio.to_thread(large_run.large_info, path) is None:
        return None
    dm.evict(dataset_id)
    return AnnDataAdaptor(path)


LARGE_REFUSED = ("This dataset is in large-dataset layout (its matrices stay on disk), and "
                 "this work reads them into memory, which large datasets never are. Run the "
                 "pipeline on it instead (it works in large-dataset mode).")


def _in_memory(work: Callable[[AnnDataAdaptor], T],
               on_disk: bool = False) -> Callable[[AnnDataAdaptor], T]:
    """``work``, called with the dataset in memory (``working_copy.in_memory``: a backed
    one is read in first, in the worker thread): steps need every value at hand. Not for
    work that reads a large dataset itself (``on_disk``: a run in large-dataset mode, or
    light work on one); any other work on a large dataset is refused (a 422)."""

    def run(adaptor: AnnDataAdaptor) -> T:
        if not on_disk:
            from scview.core import large_run

            if large_run.large_info(adaptor.h5ad_path) is not None:
                raise HTTPException(status_code=422, detail=LARGE_REFUSED)
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
    *,
    light: bool = False,
) -> T:
    """Run ``work(adaptor)`` in a worker thread with the dataset to itself, then
    reload the dataset from disk. ``control``: the pipeline run ``work`` makes
    with it, registered for cancel and watched (no events: nothing streams).
    ``light``: ``work`` needs no matrix and reads a large dataset itself
    (``large_run.read_light``), so a large one is not refused."""
    adaptor, claimed = await _claim(dm, dataset_id)
    try:
        fresh = await _for_light_work(dm, dataset_id, adaptor) if light else None
        if fresh is not None:
            adaptor = fresh
        else:
            adaptor = await _make_run_room(dm, dataset_id, adaptor, control)
        on_disk = fresh is not None or _chunked(control)
        if on_disk:
            await asyncio.to_thread(_give_back_memory)  # the copy dropped for the work
        unwatch = _watch(dataset_id, control)
    except BaseException:
        claimed()
        raise

    def release() -> None:
        unwatch()
        claimed()

    try:
        task = asyncio.ensure_future(asyncio.to_thread(_in_memory(work, on_disk), adaptor))
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
            gen = _in_memory(make_events, _chunked(control))(adaptor)
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
