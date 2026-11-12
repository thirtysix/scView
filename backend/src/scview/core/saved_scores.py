"""Per-cell scores kept as obs columns: a scored gene set or one TF / pathway's activity.

Scoring in the app (``api/v1/genesets.py``, ``api/v1/activity.py``) draws a score on the
plot; with ``save`` it is also written into the dataset as one float column, so Compare
colourings, Colour by, the histogram, sessions and the h5ad export see it like any other
column. Only the score the user put on the plot is saved, never a whole matrix of them.

Column names: ``score_<set>`` for a gene set, ``activity_<source>`` for an activity,
cleaned to letters, digits, ``_ . -`` and at most ``NAME_MAX`` characters, made unique.
The record, uns ``scview_scores`` (a JSON string, as the provenance), says per column what
it holds: kind, name, method, the genes used and missing, the matrix scored, the network,
when. Scoring the same set again (any method) or the same source replaces its column and
its entry. Only a column in the record is ever replaced or removed: a column the user or
the file made is never touched.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any

import numpy as np

from scview.core import provenance

logger = logging.getLogger(__name__)

RECORD_KEY = "scview_scores"
KINDS = ("geneset", "activity")
PREFIXES = {"geneset": "score_", "activity": "activity_"}
NAME_MAX = 64  # whole column name, as a hand-label column (core/user_labels.COLUMN_MAX)
MAX_GENES_RECORDED = 2000  # a set's genes in the record (the scoring endpoint's own cap is larger)

# Scores the pipeline already keeps per cell: offered with the saved ones, never removed.
BUILTIN_SCORE_COLUMNS = ("S_score", "G2M_score", "dissociation_stress_score")

TOOL = "scview.saved_scores"
_UNSAFE = re.compile(r"[^A-Za-z0-9_.\-]+")


class SavedScoreError(ValueError):
    """A request about a saved score that cannot be served; ``status`` is the HTTP code."""

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


def record(adata) -> dict[str, dict[str, Any]]:
    """The saved scores: column -> entry, only for columns still in obs."""
    raw = adata.uns.get(RECORD_KEY)
    if raw is None:
        return {}
    try:
        data = json.loads(raw) if isinstance(raw, (str, bytes)) else dict(raw)
    except (TypeError, ValueError):
        logger.warning("Unreadable %s record; ignored", RECORD_KEY)
        return {}
    if not isinstance(data, dict):
        return {}
    cols = set(map(str, adata.obs.columns))
    return {str(k): v for k, v in data.items() if isinstance(v, dict) and str(k) in cols}


def obs_flag(entry: dict[str, Any]) -> dict[str, Any]:
    """What the obs column listing carries per saved score (the UI groups on it)."""
    return {k: entry.get(k) for k in ("kind", "name", "method", "network")}


def _same(entry: dict[str, Any], kind: str, name: str, network: str | None) -> bool:
    return (entry.get("kind") == kind and entry.get("name") == name
            and (kind != "activity" or entry.get("network") == network))


def clean_name(kind: str, name: str) -> str:
    """``score_<name>`` / ``activity_<name>`` with only safe characters, capped."""
    prefix = PREFIXES[kind]
    body = _UNSAFE.sub("_", str(name)).strip("_.-") or "unnamed"
    return (prefix + body)[:NAME_MAX].rstrip("_.-")


def column_for(adata, kind: str, name: str, network: str | None = None) -> str:
    """The column this score goes into: its saved column when the same set (or source)
    was saved before, else a new name no obs column has."""
    for col, entry in record(adata).items():
        if _same(entry, kind, name, network):
            return col
    base = clean_name(kind, name)
    taken = {str(c).lower() for c in adata.obs.columns}
    col, i = base, 2
    while col.lower() in taken:
        suffix = f"_{i}"
        col = base[: NAME_MAX - len(suffix)] + suffix
        i += 1
    return col


def _write_record(adata, entries: dict[str, dict[str, Any]]) -> None:
    # A new value, never an in-place change: obs_writes.write_derived saves the uns
    # entries whose identity changed.
    if entries:
        adata.uns[RECORD_KEY] = json.dumps(entries, default=str)
    else:
        adata.uns.pop(RECORD_KEY, None)


def _recorded(record) -> None:
    """Run ``record`` (a ``provenance.record_step`` call); provenance never blocks the edit."""
    try:
        record()
    except Exception as e:
        logger.warning("Could not record a saved score's provenance: %s", e)


def save(adata, *, kind: str, name: str, values, method: str | None = None,
         genes: list[str] | None = None, genes_missing: list[str] | None = None,
         source: str | None = None, network: str | None = None,
         resource: str | None = None, extra: dict[str, Any] | None = None) -> str:
    """Write one per-cell score into ``adata.obs`` and the record; the column name.

    ``genes``: the genes scored (a gene set); ``source``: the matrix scored; ``network``
    and ``resource``: an activity's network and where it came from; ``extra``: other
    settings worth keeping (AUCell's ``max_rank``). Replaces the column of the same set or
    source saved before. Records the step ``save_score``."""
    if kind not in KINDS:
        raise SavedScoreError(f"Unknown score kind '{kind}'.")
    if not str(name).strip():
        raise SavedScoreError("A saved score needs a name.")
    v = np.asarray(values, dtype=np.float32).ravel()
    if v.shape[0] != adata.n_obs:
        raise SavedScoreError(
            f"{v.shape[0]:,} scores for {adata.n_obs:,} cells: the dataset changed while it was "
            "scored. Score it again.", status=409)
    entries = record(adata)
    column = column_for(adata, kind, name, network)
    replaced = column in entries
    entry: dict[str, Any] = {
        "kind": kind,
        "name": str(name),
        "method": method,
        "source": source,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if genes is not None:
        entry["genes"] = [str(g) for g in genes][:MAX_GENES_RECORDED]
        entry["n_genes"] = len(genes)
        entry["genes_missing"] = [str(g) for g in (genes_missing or [])][:MAX_GENES_RECORDED]
    if network is not None:
        entry["network"] = network
    if resource is not None:
        entry["resource"] = resource
    if extra:
        entry.update({k: v for k, v in extra.items() if v is not None})
    adata.obs[column] = v
    entries[column] = entry
    _write_record(adata, entries)
    params = {"column": column, "replaced": replaced, **{k: v for k, v in entry.items() if k != "created"}}
    _recorded(lambda: provenance.record_step(adata, step="save_score", tool=TOOL, params=params))
    return column


def remove(adata, column: str) -> dict[str, Any]:
    """Drop a saved score's column and its record entry; the entry removed. Only a column
    in the record: any other is a 404, so a file's own columns are never touched."""
    entries = record(adata)
    entry = entries.pop(column, None)
    if entry is None:
        raise SavedScoreError(f"'{column}' is not a saved score.", status=404)
    del adata.obs[column]
    _write_record(adata, entries)
    params = {"column": column, "kind": entry.get("kind"), "name": entry.get("name")}
    _recorded(lambda: provenance.record_step(adata, step="remove_score", tool=TOOL, params=params))
    return entry
