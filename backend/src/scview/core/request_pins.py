"""Every dataset a request is handed stays readable until the request has ended.

A backed dataset's X (and .raw's) is read from its file when used. The file is closed when
the adaptor lets the backed object go: an edit or a run reads the dataset into memory
(``AnnDataAdaptor.to_memory``), or the dataset manager drops it to make room. A request
reading the matrix in a worker thread at that moment fails (h5py: invalid identifier, a
500). ``AnnDataAdaptor.pin`` keeps the file open for a reader, but each route had to take
it, and take it before its first wait: the expression routes, the plots and half of the
analysis routes took none, and two that took one after a wait were fixed in v0.85.

So the pin is taken where the adaptor is handed out. ``Middleware`` gives each HTTP
request (an MCP tool's in-process call is one too) a scope; ``DatasetManager.load_dataset``
and ``get_dataset`` pin what they return into it (``hold``), and the scope unpins when the
response has been sent, a streamed one included. A route needs to do nothing, and cannot
forget to.

Two things are not covered by the request's scope, and say so themselves:

* work that goes on after the request (a computation other requests share, a background
  job) takes ``keep()`` when it starts and calls the release when it ends;
* a request that reads the dataset into memory itself (``to_memory``), or drops it
  (``DatasetManager.evict``: a run), gives up its own pin on it (``let_go``): it no longer
  reads the backed object, and a run must not keep the old tables alive until its stream ends.

Outside a request (startup, a test calling the manager) nothing is pinned, as before.
"""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Callable
from contextvars import ContextVar
from typing import Any


class _Scope:
    def __init__(self) -> None:
        self._lock = threading.Lock()  # worker threads see the request's scope too
        self._pins: dict[int, tuple[Any, Callable[[], None]]] = {}
        self._open = True

    def hold(self, adaptor: Any) -> None:
        with self._lock:
            if self._open and id(adaptor) not in self._pins:
                self._pins[id(adaptor)] = (adaptor, adaptor.pin())

    def let_go(self, adaptor: Any) -> None:
        with self._lock:
            pin = self._pins.pop(id(adaptor), None)
        if pin is not None:
            pin[1]()

    def adaptors(self) -> list[Any]:
        with self._lock:
            return [a for a, _ in self._pins.values()]

    def close(self) -> None:
        with self._lock:
            self._open = False  # a task that outlives the request pins nothing into it
            pins, self._pins = list(self._pins.values()), {}
        for _, unpin in pins:
            unpin()


_current: ContextVar[_Scope | None] = ContextVar("scview_request_pins", default=None)


def hold(adaptor: Any) -> None:
    """Pin ``adaptor`` (``AnnDataAdaptor.pin``) until the current request ends; once per
    request, nothing outside one."""
    scope = _current.get()
    if scope is not None and adaptor is not None and hasattr(adaptor, "pin"):
        scope.hold(adaptor)


def let_go(adaptor: Any) -> None:
    """The current request no longer reads ``adaptor``'s backed object (see the module)."""
    scope = _current.get()
    if scope is not None:
        scope.let_go(adaptor)


def keep() -> Callable[[], None]:
    """For work that may outlive the request that starts it: the datasets the request
    holds now, pinned again until the release this returns is called (safe to call twice)."""
    scope = _current.get()
    unpins = [a.pin() for a in scope.adaptors()] if scope is not None else []

    def release() -> None:
        for u in unpins:
            u()

    return release


@contextlib.contextmanager
def scope():
    """A request's scope for code that is not an HTTP request (an MCP tool's own reads)."""
    s = _Scope()
    token = _current.set(s)
    try:
        yield s
    finally:
        _current.reset(token)
        s.close()


class Middleware:
    """ASGI: one scope per HTTP request, closed when the response has been sent (plain
    ASGI, so a streamed body is inside it; outermost, so every inner task sees it)."""

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope_, receive, send) -> None:
        if scope_.get("type") != "http":
            await self.app(scope_, receive, send)
            return
        with scope():
            await self.app(scope_, receive, send)
