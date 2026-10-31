"""One file as a raw request body, streamed to disk (``curl --data-binary @file``): never spooled,
never held in memory, written about 1 MB at a time off the event loop. The caller renames the part
file when it is whole."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path

from fastapi import HTTPException, Request

_WRITE_SIZE = 1024 * 1024
_HEAD = 1024


class UploadRefused(RuntimeError):
    """A sentence for the client; ``status`` is the HTTP code."""

    def __init__(self, message: str, status: int = 422):
        super().__init__(message)
        self.status = status


def refuse_multipart(request: Request) -> None:
    if (request.headers.get("content-type") or "").startswith("multipart/"):
        raise HTTPException(status_code=415, detail="Send the file as the raw request body, not a multipart form.")


async def stream_to_file(request: Request, path: Path, max_bytes: int, max_mb: int,
                         check_head: Callable[[bytes], None] | None = None) -> int:
    """The body into ``path``; the bytes written. ``check_head`` sees the first bytes before the
    rest travels (it raises UploadRefused)."""
    total = 0
    pending: list[bytes] = []
    size = 0
    checked = check_head is None
    fh = await asyncio.to_thread(open, path, "wb")
    try:
        async for chunk in request.stream():
            if not chunk:
                continue
            total += len(chunk)
            if total > max_bytes:
                raise UploadRefused(f"That file is larger than the maximum upload size of {max_mb} MB.", 413)
            pending.append(chunk)
            size += len(chunk)
            if not checked and size >= _HEAD:
                check_head(b"".join(pending)[:_HEAD])
                checked = True
            if size >= _WRITE_SIZE:
                await asyncio.to_thread(fh.write, b"".join(pending))
                pending, size = [], 0
        if not total:
            raise UploadRefused("That upload was empty: send the file as the request body.")
        if not checked:
            check_head(b"".join(pending)[:_HEAD])
        if pending:
            await asyncio.to_thread(fh.write, b"".join(pending))
    finally:
        await asyncio.to_thread(fh.close)
    return total
