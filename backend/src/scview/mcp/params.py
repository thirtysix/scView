"""The pipeline's parameters as an input schema for ``run_pipeline`` and ``preflight``.

Built from ``PipelineParams`` itself (names, types, defaults, and the comments
above or beside each field as its description), so the schema cannot drift from
the run. Unknown names are refused here: the run endpoints ignore a name they do
not know, which made a misspelt parameter a run with the default instead.
"""

from __future__ import annotations

import dataclasses
import difflib
import inspect
import re
import typing
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, create_model

from scview.core.pipeline import PipelineParams

_HIDDEN = {"explicit", "enrichment_gene_sets"}  # bookkeeping, and a legacy alias


def _comments() -> dict[str, str]:
    """Each field's comment: the lines just above it and the one beside it."""
    out: dict[str, str] = {}
    above: list[str] = []
    for line in inspect.getsource(PipelineParams).splitlines()[1:]:
        s = line.strip()
        if not s:
            above = []
            continue
        if s.startswith("#"):
            above.append(s.lstrip("# ").strip())
            continue
        m = re.match(r"([a-z_][a-z0-9_]*)\s*:", s)
        if m:
            side = s.split("#", 1)[1].strip() if "#" in s else ""
            out[m.group(1)] = " ".join([*above, side]).strip()
        above = []
    return out


def fields() -> dict[str, tuple[Any, Any]]:
    """name -> (type, default) of every settable pipeline parameter."""
    hints = typing.get_type_hints(PipelineParams)
    out = {}
    for f in dataclasses.fields(PipelineParams):
        if f.name in _HIDDEN:
            continue
        default = f.default if f.default is not dataclasses.MISSING else f.default_factory()  # type: ignore[misc]
        out[f.name] = (hints[f.name], default)
    return out


def _model() -> type[BaseModel]:
    notes = _comments()
    spec: dict[str, Any] = {}
    for name, (tp, default) in fields().items():
        text = notes.get(name, "")
        desc = f"{text} Default: {default!r}." if text else f"Default: {default!r}."
        spec[name] = (tp | None, Field(default=None, description=desc))
    return create_model("PipelineParamsInput", __config__=ConfigDict(extra="forbid"), **spec)


PipelineParamsInput = _model()


def unknown_message(names: list[str]) -> str | None:
    """None when every name is a pipeline parameter, else one sentence naming the
    unknown ones (with the likely meant name) and where the list is."""
    known = list(fields())
    bad = [n for n in names if n not in known]
    if not bad:
        return None
    parts = []
    for n in bad:
        close = difflib.get_close_matches(n, known, n=1, cutoff=0.6)
        parts.append(f"'{n}' (did you mean '{close[0]}'?)" if close else f"'{n}'")
    return (f"Unknown pipeline parameter{'s' if len(bad) > 1 else ''}: {', '.join(parts)}. "
            "The run would ignore it; the valid names and defaults are in this tool's schema.")
