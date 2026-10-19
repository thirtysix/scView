"""Keep a server-sent event stream alive while a long step sends nothing.

The production frontend's nginx closes an API response that has been silent for
``proxy_read_timeout`` (300 s), and a pipeline step sends no event until it ends.
On a slow machine one step can outlast that: doublet detection on Kang (14k cells)
took over 5 minutes in a Windows VM under Podman, the browser showed "network
error", and the run stopped unsaved after the step (``processing.iterate_exclusive``).
``keepalive`` sends an SSE comment line whenever the stream has been quiet for
``interval`` seconds; every client reader skips lines starting with ``:``.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator

KEEPALIVE_SECONDS = 15.0
COMMENT = ": keepalive\n\n"


async def keepalive(
    events: AsyncIterator[str], interval: float = KEEPALIVE_SECONDS,
) -> AsyncIterator[str]:
    """``events``, with ``COMMENT`` sent after every ``interval`` s of silence.

    When the client goes away the pending read of ``events`` is cancelled and the
    generator closed, as it would be without the wrapper."""
    it = aiter(events)
    pending = asyncio.ensure_future(anext(it))
    try:
        while True:
            done, _ = await asyncio.wait({pending}, timeout=interval)
            if not done:
                yield COMMENT
                continue
            try:
                item = pending.result()
            except StopAsyncIteration:
                return
            yield item
            pending = asyncio.ensure_future(anext(it))
    finally:
        if not pending.done():
            pending.cancel()
            with contextlib.suppress(BaseException):
                await pending
        aclose = getattr(it, "aclose", None)
        if aclose is not None:
            await aclose()
