"""Older markers and enrichment, kept when a re-run replaces them (results store, phase 2).

Markers (``rank_genes_groups__{column}``) and enrichment (``enrichment__{column}__{group}``
with its ``ora_meta__`` sidecar) live in the h5ad, one per column (and group): a re-run
with another method, a re-clustering or a relabelling replaces or drops them. Here the
entries a dataset had when it was opened (or last saved) are remembered by reference
(``capture``); every save of the dataset first calls ``before_save``, which hands each
entry that was replaced by a different result, or removed, to the results store as an
archived (superseded) result, then remembers the new ones.

The h5ad keeps only the current result, so exports are unchanged. An archived entry
carries the fingerprint of the column it was computed on (cells + labels): one whose
column has since changed is stale and cannot be restored onto the new groups.
``restore`` puts an archived entry back, archiving the one it replaces.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any

from scview.core import results_store
from scview.core.uns_keys import ENRICHMENT_PREFIX, MARKERS_PREFIX, META_PREFIX, decode, enrichment_column

logger = logging.getLogger(__name__)

KINDS = ("markers", "enrichment")
_SEP = "__"
_MISSING = object()


@dataclass
class _Seen:
    value: Any
    meta: Any  # enrichment's ora_meta sidecar (None for markers)
    computed_on: dict | None
    digest: str | None = None


_baseline: dict[str, dict[str, _Seen]] = {}
_lock = threading.Lock()


def _tracked(uns) -> list[str]:
    return [k for k in uns if isinstance(k, str) and k.startswith((MARKERS_PREFIX, ENRICHMENT_PREFIX))]


def meta_of(key: str) -> str:
    """The ora_meta sidecar key of an enrichment key."""
    return META_PREFIX + key[len(ENRICHMENT_PREFIX):]


def column_of(key: str) -> str:
    if key.startswith(MARKERS_PREFIX):
        return decode(key[len(MARKERS_PREFIX):])
    return enrichment_column(key) or ""


def group_of(key: str) -> str:
    return decode(key[len(ENRICHMENT_PREFIX):].split(_SEP, 1)[1]) if key.startswith(ENRICHMENT_PREFIX) else ""


def _dataset(target: Any) -> str | None:
    """A dataset id, or the dataset of an adaptor or a dataset file's path."""
    if isinstance(target, str) and results_store.safe_id(target):
        return target
    return results_store.dataset_id_of(target)


def capture(target: Any, adata) -> None:
    """Remember the markers and enrichment ``adata`` holds as the saved ones."""
    dataset_id = _dataset(target)
    if dataset_id is None:
        return
    try:
        uns = adata.uns
        columns: dict[str, dict | None] = {}
        seen = {}
        cells = None
        for key in _tracked(uns):
            col = column_of(key)
            if col not in columns:
                cells = cells or results_store.cells_hash(adata)
                columns[col] = results_store.column_fingerprint(adata, col, cells)
            meta = uns.get(meta_of(key)) if key.startswith(ENRICHMENT_PREFIX) else None
            seen[key] = _Seen(uns[key], meta, columns[col])
    except Exception:
        logger.exception("could not note the markers / enrichment of %s", dataset_id)
        return
    with _lock:
        _baseline[dataset_id] = seen


def forget(dataset_id: str) -> None:
    with _lock:
        _baseline.pop(dataset_id, None)


def _same(seen: _Seen, now: Any) -> bool:
    try:
        if seen.digest is None:
            seen.digest = results_store.digest(seen.value)
        return seen.digest == results_store.digest(now)
    except Exception:
        return False


def _archive(dataset_id: str, key: str, seen: _Seen) -> None:
    col = column_of(key)
    if key.startswith(MARKERS_PREFIX):
        params = dict(seen.value.get("params") or {}) if hasattr(seen.value, "get") else {}
        method = params.get("method")
        label = f"Markers of '{col}'" + (f" ({method})" if method else "")
        results_store.keep(dataset_id, "markers", {"uns_key": key, "value": seen.value}, key=col,
                           label=label, params=_plain(params), computed_on=seen.computed_on, archived=True)
    else:
        group = group_of(key)
        results_store.keep(dataset_id, "enrichment",
                           {"uns_key": key, "value": seen.value, "meta_key": meta_of(key), "meta": seen.meta},
                           key=f"{col}|{group}", label=f"Enrichment of '{col}' {group}",
                           computed_on=seen.computed_on, archived=True)


def _plain(params: dict) -> dict:
    out = {}
    for k, v in params.items():
        v = v.item() if hasattr(v, "item") and getattr(v, "ndim", 0) == 0 else v
        out[str(k)] = v if isinstance(v, (str, int, float, bool, type(None))) else str(v)
    return out


def before_save(target: Any, adata) -> int:
    """Archive the markers / enrichment the save is about to replace with a different
    result or drop, then note the new ones as saved. Returns how many were archived.
    Never raises: a save must not fail over its history."""
    dataset_id = _dataset(target)
    if dataset_id is None:
        return 0
    try:
        with _lock:
            before = _baseline.get(dataset_id)
        n = 0
        uns = adata.uns
        for key, seen in (before or {}).items():
            now = uns.get(key, _MISSING)
            if now is seen.value or (now is not _MISSING and _same(seen, now)):
                continue
            _archive(dataset_id, key, seen)
            n += 1
        if n:
            logger.info("Kept %d superseded marker / enrichment result(s) of %s", n, dataset_id)
    except Exception:
        logger.exception("could not keep the superseded results of %s", dataset_id)
        n = 0
    capture(dataset_id, adata)
    return n


class RestoreError(Exception):
    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status, self.detail = status, detail


def check_restorable(dataset_id: str, result_id: str, adaptor) -> tuple[dict, dict]:
    """The archived entry and its value, or RestoreError (404 gone, 400 not a markers /
    enrichment result, 409 computed on other groups than the column has now)."""
    entry = next((e for e in results_store.entries(dataset_id, adaptor=adaptor) if e["id"] == result_id), None)
    got = results_store.load(dataset_id, result_id) if entry else None
    if entry is None or got is None:
        raise RestoreError(404, "No such stored result.")
    if entry["kind"] not in KINDS or not entry.get("archived"):
        raise RestoreError(400, "Only earlier markers and enrichment can be made current again.")
    if entry["stale"]:
        col = (entry.get("fingerprint") or {}).get("column", "")
        raise RestoreError(409, f"This result was computed on earlier groups of '{col}' (re-clustered, "
                                "relabelled or cells filtered since), so it does not describe the groups now.")
    return entry, got[1]


def apply(adata, value: dict) -> list[str]:
    """Put an archived markers / enrichment value back into ``adata.uns`` (the edit;
    the caller saves). Restored markers drop the column's enrichment, which was
    computed from the markers they replace. Returns the uns keys set."""
    key = value["uns_key"]
    adata.uns[key] = value["value"]
    keys = [key]
    if key.startswith(MARKERS_PREFIX):
        col = column_of(key)
        rgg = adata.uns.get("rank_genes_groups")
        if hasattr(rgg, "get") and (rgg.get("params") or {}).get("groupby") == col:
            adata.uns["rank_genes_groups"] = value["value"]
            keys.append("rank_genes_groups")
        from scview.core.pipeline import drop_enrichment_cache
        drop_enrichment_cache(adata, col)
    else:
        if value.get("meta") is None:
            adata.uns.pop(value["meta_key"], None)
        else:
            adata.uns[value["meta_key"]] = value["meta"]
            keys.append(value["meta_key"])
    return keys
