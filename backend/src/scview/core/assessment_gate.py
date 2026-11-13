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
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Hashable
from typing import Any

MAX_AT_ONCE = 2  # heavy computations per dataset at a time


class _Gate:
    def __init__(self) -> None:
        self.lanes = asyncio.Semaphore(MAX_AT_ONCE)
        self.inflight: dict[Hashable, asyncio.Task] = {}


# (event loop, dataset id) -> its gate; a gate goes once nothing is in flight.
_gates: dict[tuple[int, str], _Gate] = {}


def _retrieve(task: asyncio.Task) -> None:
    if not task.cancelled():
        task.exception()  # read, so an error nobody awaited any more is not logged as lost


async def run(dataset_id: str, key: Hashable, fn: Callable[..., Any], *args: Any) -> Any:
    """``fn(*args)`` in a worker thread, at most ``MAX_AT_ONCE`` per dataset at a time; a call
    with the same ``key`` while one is in flight gets that one's result (or its error)."""
    loop = asyncio.get_running_loop()
    gid = (id(loop), dataset_id)
    gate = _gates.get(gid)
    if gate is None:
        gate = _gates[gid] = _Gate()
    task = gate.inflight.get(key)
    if task is None:
        async def work() -> Any:
            async with gate.lanes:
                return await asyncio.to_thread(fn, *args)

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
