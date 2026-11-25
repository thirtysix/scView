"""Data Assessment's heavy work, a little at a time per dataset.

Opening Data Assessment asks a dozen endpoints at once (the assessment, cluster QC, the QC
report facts, the sex check...), each a worker thread that reads the dataset's matrices. On
a 39k x 35k kidney file each grew the backend by 0.4-0.6 GB over the 2.6 GB the open dataset
holds, and one page peaked at 4.3 GB; two pages (two users, or two tabs) opening it at once
ran every one of them twice and stopped a 6 GB backend.

So ``run`` gives each dataset two lanes (``MAX_AT_ONCE``): heavy work beyond that waits its
turn, without blocking the event loop or another dataset's work. A request identical to one
already running or waiting (same endpoint, arguments and loaded dataset) does not run again:
it waits for that one and gets its result, so a second page costs nothing more. The work runs
in its own task, so a client that goes away does not cancel it for the others.

Each computation is also held in the memory budget (``Budget``): once it has a lane, its
estimated peak (worked out then, so a cache an earlier request filled counts as nothing to
read) is reserved with ``working_copy.reading`` until it ends. Requests that share it share
the one reservation; requests the page fires together each count the others' reservations,
and one that fits only once those end waits for them (``QUEUE_SECONDS``) rather than being
refused. One that cannot fit even with the other datasets closed is a 507 naming it, and the
page's other cards still load. A 3 GB backend holding a deep 1.6 GB file (23,578 x 34,209)
was killed within 3 s of opening Data Assessment before this.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Hashable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

MAX_AT_ONCE = 2  # heavy computations per dataset at a time
# How long a computation waits for room that other requests' reads hold, past the dataset
# manager's own few seconds: under nginx's 300 s read timeout with the lane wait before it.
QUEUE_SECONDS = 120.0


@dataclass
class Budget:
    """How a computation is held in the memory budget: ``need()`` its estimated peak in bytes
    (called in a worker thread once it has a lane; 0: nothing held), ``what`` the subject of a
    refusal ("Cluster QC"), ``dm`` the dataset manager."""

    dm: Any
    need: Callable[[], int]
    what: str


async def _held(dataset_id: str, budget: Budget | None, fn: Callable[..., Any], *args: Any) -> Any:
    """``fn(*args)`` in a worker thread, inside ``budget``'s reservation when there is one."""
    if budget is None:
        return await asyncio.to_thread(fn, *args)
    from scview.core import working_copy

    need = max(0, int(await asyncio.to_thread(budget.need)))
    if need:  # work that reads the matrix: a backed one's spans (working_copy.BACKED_READ_BYTES)
        get = getattr(budget.dm, "get_dataset", None)
        adaptor = get(dataset_id) if get is not None else None
        need = max(need, await asyncio.to_thread(working_copy.backed_read_bytes, adaptor))
    if need:
        logger.info("%s (%s): holding %.2f GB while it runs", budget.what, dataset_id, need / 1024**3)
    what = working_copy.subject(budget.dm, dataset_id, need, budget.what)
    async with working_copy.reading(budget.dm, dataset_id, need, what, wait=QUEUE_SECONDS) as held:
        return await held.run(fn, *args)


class _Gate:
    def __init__(self) -> None:
        self.lanes = asyncio.Semaphore(MAX_AT_ONCE)
        self.inflight: dict[Hashable, asyncio.Task] = {}


# (event loop, dataset id) -> its gate; a gate goes once nothing is in flight.
_gates: dict[tuple[int, str], _Gate] = {}


def _retrieve(task: asyncio.Task) -> None:
    if not task.cancelled():
        task.exception()  # read, so an error nobody awaited any more is not logged as lost


async def run(dataset_id: str, key: Hashable, fn: Callable[..., Any], *args: Any, lane: bool = True,
              budget: Budget | None = None) -> Any:
    """``fn(*args)`` in a worker thread, at most ``MAX_AT_ONCE`` per dataset at a time; a call
    with the same ``key`` while one is in flight gets that one's result (or its error).
    ``lane`` False: shared the same way, but not queued behind the lanes (the assessment
    itself, which the page's checklist waits for: 16 s behind cluster QC and the QC report).
    ``budget``: held in the memory budget while it runs (see the module); a
    ``DatasetMemoryError`` (507) reaches every caller sharing it."""
    loop = asyncio.get_running_loop()
    gid = (id(loop), dataset_id)
    gate = _gates.get(gid)
    if gate is None:
        gate = _gates[gid] = _Gate()
    task = gate.inflight.get(key)
    if task is None:
        async def work() -> Any:
            if not lane:
                return await _held(dataset_id, budget, fn, *args)
            async with gate.lanes:
                return await _held(dataset_id, budget, fn, *args)

        task = loop.create_task(work())
        gate.inflight[key] = task

        def done(t: asyncio.Task) -> None:
            if gate.inflight.get(key) is t:
                del gate.inflight[key]
            if not gate.inflight and _gates.get(gid) is gate:
                del _gates[gid]
            _retrieve(t)

        task.add_done_callback(done)
    return await asyncio.shield(task)


def in_flight(dataset_id: str) -> int:
    """Heavy computations running or waiting for ``dataset_id`` (tests, logs)."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return 0
    gate = _gates.get((id(loop), dataset_id))
    return len(gate.inflight) if gate is not None else 0
