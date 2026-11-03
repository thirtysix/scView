"""Route answers made small enough for an agent's context.

Numbers keep 4 significant digits (relative rounding, so a p value of 1e-300
stays 1e-300 and never becomes 0); a table keeps its first ``limit`` rows with
the total; whatever is still too long is cut list by list, longest first,
until the text fits ``MAX_CHARS``.
"""

from __future__ import annotations

import json
import math
from typing import Any

DEFAULT_LIMIT = 20
MAX_LIMIT = 200
MAX_CHARS = 8000  # at default settings; a larger ``limit`` asks for more on purpose


def num(x: Any) -> Any:
    """``x`` to 4 significant digits; NaN and infinities become None (JSON has neither)."""
    if isinstance(x, bool) or not isinstance(x, float):
        if hasattr(x, "item") and not isinstance(x, (list, dict, str)):  # numpy scalars
            try:
                return num(x.item())
            except (TypeError, ValueError):
                return x
        return x
    if not math.isfinite(x):
        return None
    if x == 0:
        return 0.0
    return float(f"{x:.4g}")


def rounded(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): rounded(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [rounded(v) for v in obj]
    return num(obj)


def rows(table: list[dict[str, Any]], columns: list[str] | None = None,
         limit: int = DEFAULT_LIMIT) -> dict[str, Any]:
    """The first ``limit`` rows of a table (only ``columns`` when given), with the total."""
    picked = [{c: r.get(c) for c in columns if c in r} if columns else r for r in table[:limit]]
    out: dict[str, Any] = {"rows": rounded(picked), "n_total": len(table)}
    if len(table) > limit:
        out["n_shown"] = limit
    return out


def table(records: list[dict[str, Any]], columns: list[str], limit: int = DEFAULT_LIMIT) -> dict[str, Any]:
    """Records as {"columns", "rows": [[...]], "n_total"}: the column names once, not per row."""
    cols = [c for c in columns if any(c in r for r in records[:limit])]
    out: dict[str, Any] = {"columns": cols, "rows": [[num(r.get(c)) for c in cols] for r in records[:limit]],
                           "n_total": len(records)}
    if len(records) > limit:
        out["n_shown"] = limit
    return out


def clamp(limit: int | None) -> int:
    return max(1, min(MAX_LIMIT, int(limit or DEFAULT_LIMIT)))


def _longest_list(obj: Any, path: tuple = ()) -> tuple[int, tuple]:
    best = (0, ())
    if isinstance(obj, dict):
        for k, v in obj.items():
            best = max(best, _longest_list(v, (*path, k)), key=lambda t: t[0])
    elif isinstance(obj, list):
        size = len(json.dumps(obj, default=str))
        if len(obj) > 1:
            best = (size, path)
        for i, v in enumerate(obj):
            best = max(best, _longest_list(v, (*path, i)), key=lambda t: t[0])
    return best


def dumps(obj: Any, max_chars: int = MAX_CHARS) -> str:
    """Compact JSON of ``obj`` (rounded), cut to about ``max_chars``: the list
    taking the most room is halved, again and again, and says how long it was."""
    data = rounded(obj)
    text = json.dumps(data, separators=(",", ":"), default=str)
    notes: dict[str, int] = {}
    for _ in range(60):
        if len(text) <= max_chars:
            break
        size, path = _longest_list(data)
        if not path and not isinstance(data, list):
            break
        parent = data
        for p in path[:-1]:
            parent = parent[p]
        key = path[-1] if path else None
        lst = parent[key] if path else data
        if len(lst) <= 1:
            break
        name = ".".join(str(p) for p in path)
        notes.setdefault(name, len(lst))
        keep = max(1, len(lst) // 2)
        if path:
            parent[key] = lst[:keep]
        else:
            data = lst[:keep]
        text = json.dumps(data, separators=(",", ":"), default=str)
    if notes and isinstance(data, dict):
        data["truncated"] = {k: f"{len(_at(data, k))} of {n} shown" for k, n in notes.items()}
        text = json.dumps(data, separators=(",", ":"), default=str)
    return text


def _at(data: Any, dotted: str) -> Any:
    cur = data
    for part in dotted.split("."):
        cur = cur[int(part)] if isinstance(cur, list) else cur[part]
    return cur


def pick(d: dict[str, Any] | None, *keys: str) -> dict[str, Any]:
    """The keys of ``d`` that are present and not None."""
    if not d:
        return {}
    return {k: d[k] for k in keys if d.get(k) is not None}
