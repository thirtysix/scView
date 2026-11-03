"""The MCP tools' calls to scView's own ``/api/v1`` routes, in process.

A call goes through the whole FastAPI app (its middleware, the access gate, the
route's validation, edit locks and memory checks), exactly as the browser's
request does, without a socket. It is a plain ASGI call rather than httpx's
ASGITransport, which holds a whole response before returning it: a streamed
run (SSE) has to be read as it arrives, for a job's progress, and has to be
able to end as a browser closing the tab ends it (``disconnect``).
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import Any
from urllib.parse import urlencode

API = "/api/v1"

EventFn = Callable[[str, Any], Awaitable[None] | None]


class RouteError(Exception):
    """A route's error answer: its status and its one-sentence ``detail``."""

    def __init__(self, status: int, detail: str, large_dataset: bool = False) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail
        self.large_dataset = large_dataset


def app():
    from scview.main import app as _app

    return _app


def dependency(fn):
    """A FastAPI dependency's value as the routes get it (tests override them)."""
    return app().dependency_overrides.get(fn, fn)()


def settings():
    from scview.dependencies import get_settings_dep

    return dependency(get_settings_dep)


def dataset_manager():
    from scview.dependencies import get_dataset_manager

    return dependency(get_dataset_manager)


def _detail(status: int, body: bytes) -> RouteError:
    try:
        data = json.loads(body or b"{}")
    except ValueError:
        data = None
    if isinstance(data, dict) and "detail" in data:
        detail = data["detail"]
        if isinstance(detail, list):  # FastAPI's validation errors
            detail = "; ".join(
                f"{'.'.join(str(p) for p in d.get('loc', [])[1:])}: {d.get('msg')}" for d in detail)
        return RouteError(status, str(detail), bool(data.get("large_dataset")))
    text = (body or b"").decode("utf-8", "replace").strip()[:300]
    return RouteError(status, text or f"The request failed ({status}).")


def _scope(method: str, path: str, query: dict[str, Any] | None, body: bytes) -> dict[str, Any]:
    qs = urlencode({k: v for k, v in (query or {}).items() if v is not None}, doseq=True)
    headers = [(b"host", b"127.0.0.1"), (b"accept-encoding", b"identity"),
               (b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]
    token = settings().ACCESS_TOKEN
    if token:
        headers.append((b"authorization", f"Bearer {token}".encode()))
    return {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"}, "http_version": "1.1",
            "method": method, "scheme": "http", "path": path, "raw_path": path.encode(),
            "query_string": qs.encode(), "root_path": "", "headers": headers,
            "client": ("127.0.0.1", 0), "server": ("127.0.0.1", 80), "state": {}}


async def _call(method: str, path: str, query: dict[str, Any] | None, body: Any,
                on_chunk: Callable[[bytes], Awaitable[None]] | None,
                disconnect: asyncio.Event | None) -> tuple[int, dict[str, str], bytes]:
    payload = b"" if body is None else json.dumps(body).encode()
    scope = _scope(method, API + path, query, payload)
    sent = False
    status = 0
    headers: dict[str, str] = {}
    chunks: list[bytes] = []
    stop = disconnect or asyncio.Event()

    async def receive() -> dict[str, Any]:
        nonlocal sent
        if not sent:
            sent = True
            return {"type": "http.request", "body": payload, "more_body": False}
        await stop.wait()  # the "client" stays until told to leave
        return {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        nonlocal status
        if message["type"] == "http.response.start":
            status = int(message["status"])
            headers.update({k.decode().lower(): v.decode() for k, v in message.get("headers", [])})
        elif message["type"] == "http.response.body":
            data = message.get("body", b"")
            if on_chunk is not None and status < 400 and data:
                await on_chunk(data)
            else:
                chunks.append(data)

    try:
        await app()(scope, receive, send)
    except Exception as e:  # noqa: BLE001 — an unhandled error: the browser would get a 500
        raise RouteError(500, f"The server failed on this request: {e}") from e
    return status, headers, b"".join(chunks)


async def request(method: str, path: str, query: dict[str, Any] | None = None,
                  body: Any = None) -> tuple[dict[str, str], bytes]:
    """One route's answer (headers, body); RouteError for a status of 400 or more."""
    status, headers, data = await _call(method, path, query, body, None, None)
    if status >= 400:
        raise _detail(status, data)
    return headers, data


async def get(path: str, **query: Any) -> Any:
    _, data = await request("GET", path, query)
    return json.loads(data)


async def post(path: str, body: Any = None, **query: Any) -> Any:
    _, data = await request("POST", path, query, body if body is not None else {})
    return json.loads(data)


async def stream(path: str, body: Any, on_event: EventFn,
                 disconnect: asyncio.Event | None = None) -> None:
    """POST to a streaming (SSE) route and hand each event (type, data) to ``on_event``
    as it arrives. Setting ``disconnect`` leaves the stream, as a closed tab does."""
    buffer = bytearray()

    async def on_chunk(data: bytes) -> None:
        buffer.extend(data.replace(b"\r\n", b"\n"))
        while b"\n\n" in buffer:
            raw, _, rest = bytes(buffer).partition(b"\n\n")
            buffer[:] = rest
            event, lines = "message", []
            for line in raw.decode("utf-8", "replace").split("\n"):
                if line.startswith(":"):
                    continue  # keepalive comments
                if line.startswith("event:"):
                    event = line[6:].strip()
                elif line.startswith("data:"):
                    lines.append(line[5:].lstrip())
            if not lines:
                continue
            try:
                parsed = json.loads("\n".join(lines))
            except ValueError:
                parsed = "\n".join(lines)
            out = on_event(event, parsed)
            if asyncio.iscoroutine(out):
                await out

    status, _, data = await _call("POST", path, None, body, on_chunk, disconnect)
    if status >= 400:
        raise _detail(status, data)
