"""SSE keep-alive comments while a stream is quiet (core/sse.py)."""

from __future__ import annotations

import asyncio

from scview.core.sse import COMMENT, keepalive


async def _slow(delays, closed):
    try:
        for d in delays:
            await asyncio.sleep(d)
            yield f"data: {d}\n\n"
    finally:
        closed.append(True)


async def _collect(agen):
    return [x async for x in agen]


async def test_quiet_stream_gets_comments_between_events():
    closed = []
    out = await _collect(keepalive(_slow([0.25, 0.0], closed), interval=0.05))
    events = [x for x in out if x != COMMENT]
    assert events == ["data: 0.25\n\n", "data: 0.0\n\n"]
    # Comments come before the slow event, none after the fast one.
    assert out.index("data: 0.25\n\n") >= 3
    assert out[-2:] == events
    assert closed == [True]


async def test_busy_stream_is_unchanged():
    closed = []
    out = await _collect(keepalive(_slow([0.0, 0.0, 0.0], closed), interval=1.0))
    assert COMMENT not in out and len(out) == 3


async def test_client_going_away_closes_the_inner_stream():
    closed = []
    wrapped = keepalive(_slow([10.0], closed), interval=0.05)
    assert await anext(wrapped) == COMMENT
    await wrapped.aclose()  # what Starlette does when the client disconnects
    assert closed == [True]
