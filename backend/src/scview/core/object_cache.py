"""Results cached against one loaded dataset object, computed on demand.

The dataset manager holds each loaded dataset as one AnnData object and drops it
after every pipeline run (the next request loads a new one from disk), so a result
cached against the object cannot outlive the expression values it came from.
AnnData is not hashable, so entries are keyed by ``id()`` and checked against a
weak reference: a dead object's entry is removed, and a recycled id misses.
"""

from __future__ import annotations

import threading
import weakref
from typing import Any


class ObjectCache:
    """``(object, key) -> value``, entries dropped when the object is collected."""

    def __init__(self) -> None:
        self._items: dict[int, tuple[weakref.ref, dict[Any, Any]]] = {}
        self._lock = threading.Lock()

    def get(self, obj: Any, key: Any = None) -> Any | None:
        with self._lock:
            entry = self._items.get(id(obj))
            if entry is None or entry[0]() is not obj:
                return None
            return entry[1].get(key)

    def put(self, obj: Any, value: Any, key: Any = None) -> None:
        oid = id(obj)

        def _drop(_ref, oid=oid) -> None:
            with self._lock:
                entry = self._items.get(oid)
                if entry is not None and entry[0] is _ref:
                    del self._items[oid]

        with self._lock:
            entry = self._items.get(oid)
            if entry is None or entry[0]() is not obj:
                entry = (weakref.ref(obj, _drop), {})
                self._items[oid] = entry
            entry[1][key] = value

    def __len__(self) -> int:
        with self._lock:
            return sum(1 for ref, _ in self._items.values() if ref() is not None)
