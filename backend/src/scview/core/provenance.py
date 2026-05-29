"""Data provenance & history recorded inside the AnnData.

scView records what it does *into the data itself* so that the next person (or
future-you) opening the file can see exactly what happened — closing the
"what's been done to this file?" loop that scView exists to solve, including for
scView's own outputs. See ``docs/PROVENANCE.md``.

Storage: a single JSON string in ``adata.uns['scview_provenance']``. A JSON
string round-trips through h5ad cleanly across anndata versions (unlike nested
lists-of-dicts in ``uns``), and reads are defensive — a malformed block never
breaks loading.

Block shape (see PROVENANCE.md for the full schema):
    {
      "schema_version": 1,
      "source":  { origin, original_filename, format, ingested_at, n_cells,
                   n_genes, merged_from?, merge? },
      "history": [ { step, tool, params, timestamp, scview_version, effect, note? } ],
      "current": { qc?, normalized?, pca?, clustering?, embeddings?, markers_for?, … }
    }
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

UNS_KEY = "scview_provenance"
SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# Read / write
# ---------------------------------------------------------------------------


def _empty() -> dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, "source": {}, "history": [], "current": {}}


def read_provenance(adata) -> dict[str, Any]:
    """Return the provenance block, or an empty one. Never raises."""
    raw = adata.uns.get(UNS_KEY) if hasattr(adata, "uns") else None
    if raw is None:
        return _empty()
    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
        if not isinstance(data, dict):
            raise ValueError("provenance is not an object")
    except (ValueError, TypeError) as e:
        logger.warning("Ignoring malformed %s: %s", UNS_KEY, e)
        return _empty()
    # Repair missing keys defensively.
    data.setdefault("schema_version", SCHEMA_VERSION)
    data.setdefault("source", {})
    data.setdefault("history", [])
    data.setdefault("current", {})
    return data


def _write(adata, data: dict[str, Any]) -> None:
    adata.uns[UNS_KEY] = json.dumps(data, default=_json_default)


def has_provenance(adata) -> bool:
    p = read_provenance(adata)
    return bool(p.get("history") or p.get("source"))


def carry(src, dst) -> None:
    """Copy the provenance block from src to dst when a pipeline step returns a
    new AnnData object that doesn't carry it (so history isn't lost)."""
    try:
        if UNS_KEY in src.uns and UNS_KEY not in dst.uns:
            dst.uns[UNS_KEY] = src.uns[UNS_KEY]
    except (AttributeError, TypeError):
        pass


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------


def init_source(
    adata,
    *,
    origin: str,
    original_filename: str,
    fmt: str,
    merged_from: list[dict] | None = None,
    merge: dict | None = None,
    when: str | None = None,
    overwrite: bool = False,
) -> None:
    """Record where this dataset came from. No-op on an existing source unless
    ``overwrite`` (so re-loads don't clobber the original provenance)."""
    data = read_provenance(adata)
    if data["source"] and not overwrite:
        return
    source: dict[str, Any] = {
        "origin": origin,
        "original_filename": original_filename,
        "format": fmt,
        "ingested_at": when or _now(),
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
    }
    if merged_from:
        source["merged_from"] = merged_from
    if merge:
        source["merge"] = merge
    data["source"] = source
    _write(adata, data)


def record_step(
    adata,
    *,
    step: str,
    tool: str,
    params: dict | None = None,
    note: str | None = None,
    when: str | None = None,
) -> None:
    """Append one step to the history — a replayable recipe entry that is also
    a git-style commit: it carries a content-derived ``commit_id`` and a
    ``parent`` pointer to the previous commit (the DAG backbone)."""
    data = read_provenance(adata)
    history = data["history"]
    parent = history[-1].get("commit_id") if history else None
    ts = when or _now()
    cleaned = _clean(params or {})
    entry: dict[str, Any] = {
        "commit_id": _commit_id(parent, step, cleaned, ts),
        "parent": parent,
        "step": step,
        "tool": tool,
        "params": cleaned,
        "timestamp": ts,
        "scview_version": _scview_version(),
        "effect": {"n_cells": int(adata.n_obs), "n_genes": int(adata.n_vars)},
    }
    if note:
        entry["note"] = note
    history.append(entry)
    data.setdefault("current", {})["head"] = entry["commit_id"]
    _write(adata, data)


def recipe(adata) -> list[dict[str, Any]]:
    """Return the ordered, replayable recipe (step + params per commit) — a
    portable record that reproduces this dataset's processing elsewhere."""
    return [
        {"commit_id": h.get("commit_id"), "step": h["step"], "params": h.get("params", {})}
        for h in read_provenance(adata).get("history", [])
    ]


def set_current(adata, **fields: Any) -> None:
    """Merge fields into the denormalised ``current`` state summary."""
    data = read_provenance(adata)
    data["current"].update(_clean(fields))
    _write(adata, data)


# ---------------------------------------------------------------------------
# Reconciliation — recorded vs actual
# ---------------------------------------------------------------------------


def reconcile(adata) -> list[str]:
    """Return human-readable mismatches between recorded ``current`` and the
    actual data (e.g. a file edited outside scView). Empty list = consistent."""
    data = read_provenance(adata)
    current = data.get("current", {})
    issues: list[str] = []

    obsm = set(getattr(adata, "obsm", {}).keys())
    for emb in current.get("embeddings", []) or []:
        if emb not in obsm:
            issues.append(f"recorded embedding '{emb}' is not present in the data")

    clustering = current.get("clustering") or {}
    col = clustering.get("column")
    if col and col not in getattr(adata, "obs", {}).columns:
        issues.append(f"recorded clustering column '{col}' is missing from the data")

    uns = getattr(adata, "uns", {})
    for mcol in current.get("markers_for", []) or []:
        if f"rank_genes_groups__{mcol}" not in uns and "rank_genes_groups" not in uns:
            issues.append(f"recorded markers for '{mcol}' are not present in the data")

    return issues


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _commit_id(parent: str | None, step: str, params: dict, ts: str) -> str:
    """Content-derived commit id (git-style): hashes parent + step + params + time."""
    payload = json.dumps([parent, step, params, ts], sort_keys=True, default=_json_default)
    return hashlib.sha1(payload.encode()).hexdigest()[:12]


def _scview_version() -> str:
    try:
        return version("scview")
    except PackageNotFoundError:
        return "0.0.0"


def _json_default(o: Any) -> Any:
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _clean(obj: Any) -> Any:
    """Coerce to plain JSON-safe types (handles numpy scalars/arrays)."""
    return json.loads(json.dumps(obj, default=_json_default))
