"""An edit's lock outlives a cancelled request until its worker thread ends.

Endpoints edit a dataset in a worker thread while holding its edit lock. When a
client went away, the request was cancelled and `async with dataset_lock` released
the lock at once, while the thread could still be writing the derived file: the
next edit could start on a half-written file.
"""

from __future__ import annotations

import asyncio
import threading

from scview.core.obs_writes import dataset_edit, is_locked


async def test_a_cancelled_edit_keeps_the_lock_until_its_thread_ends():
    started, finish = threading.Event(), threading.Event()

    def slow_write():
        started.set()
        finish.wait(5)
        return "written"

    async def request():
        async with dataset_edit("ds") as edit:
            return await edit.run(slow_write)

    task = asyncio.ensure_future(request())
    await asyncio.to_thread(started.wait, 5)
    task.cancel()  # the client went away mid-write
    await asyncio.sleep(0.05)
    assert is_locked("ds")  # the thread is still writing
    finish.set()
    for _ in range(100):
        if not is_locked("ds"):
            break
        await asyncio.sleep(0.01)
    assert not is_locked("ds")


async def test_a_finished_or_failed_edit_releases_the_lock():
    async with dataset_edit("ds2") as edit:
        assert await edit.run(lambda x: x + 1, 1) == 2
    assert not is_locked("ds2")

    def boom():
        raise ValueError("bad")

    try:
        async with dataset_edit("ds2") as edit:
            await edit.run(boom)
    except ValueError:
        pass
    assert not is_locked("ds2")
