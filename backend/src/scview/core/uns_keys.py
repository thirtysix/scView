"""Keys of the per-column results scView caches in ``adata.uns``.

Markers live at ``rank_genes_groups__{column}``, enrichment at
``enrichment__{column}__{group}`` and its provenance sidecar at
``ora_meta__{column}__{group}``. Build and parse them only through this module.

h5py treats "/" in a key as a path separator, so a CellTypist label such as
"Tcm/Naive helper T cells" was written as a nested group and read back as
``enrichment__cell_type__Tcm`` holding ``{"Naive helper T cells": ...}``: the
cached result was never found again (anndata also warns that it will refuse such
keys outright). Each name is therefore encoded before it goes into a key, "/"
becoming "%2F". A name without "/" encodes to itself, so every key written by
an earlier version still resolves.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable, Iterator

MARKERS_PREFIX = "rank_genes_groups__"
ENRICHMENT_PREFIX = "enrichment__"
META_PREFIX = "ora_meta__"

# A joint dataset's batch-mixing score (core/integration.py). It describes the
# current PCA spaces, so re-running PCA or batch correction drops it.
BATCH_MIXING_KEY = "scview_batch_mixing"

_SEP = "__"
_SLASH = "%2F"


def encode(name: object) -> str:
    """A column or group name as it appears inside a key."""
    return str(name).replace("/", _SLASH)


def decode(part: str) -> str:
    return part.replace(_SLASH, "/")


def markers_key(column: object) -> str:
    return f"{MARKERS_PREFIX}{encode(column)}"


def enrichment_prefix(column: object) -> str:
    return f"{ENRICHMENT_PREFIX}{encode(column)}{_SEP}"


def enrichment_key(column: object, group: object) -> str:
    return f"{enrichment_prefix(column)}{encode(group)}"


def meta_prefix(column: object) -> str:
    return f"{META_PREFIX}{encode(column)}{_SEP}"


def meta_key(column: object, group: object) -> str:
    return f"{meta_prefix(column)}{encode(group)}"


def markers_column(key: object) -> str | None:
    """The obs column of a ``rank_genes_groups__{column}`` key, else None."""
    if isinstance(key, str) and key.startswith(MARKERS_PREFIX):
        return decode(key[len(MARKERS_PREFIX):])
    return None


def enrichment_column(key: object) -> str | None:
    """The obs column of an ``enrichment__{column}__{group}`` key, else None."""
    if isinstance(key, str) and key.startswith(ENRICHMENT_PREFIX):
        return decode(key[len(ENRICHMENT_PREFIX):].split(_SEP, 1)[0])
    return None


# ---------------------------------------------------------------------------
# Files written before the encoding
# ---------------------------------------------------------------------------


def _is_markers(value: Any) -> bool:
    return not isinstance(value, Mapping) or "names" in value or "params" in value


def _is_payload(value: Any) -> bool:
    # Payloads and sidecars are JSON strings; a sidecar may be a parsed dict.
    return not isinstance(value, Mapping) or "ora_version" in value


def _flatten(
    key: str, value: Any, is_leaf: Callable[[Any], bool]
) -> Iterator[tuple[str, Any]]:
    if is_leaf(value):
        yield key, value
        return
    for child, sub in value.items():
        yield from _flatten(f"{key}/{child}", sub, is_leaf)


def repair_slash_keys(uns: Any) -> int:
    """Re-key results that an earlier version saved under a raw "/" key.

    Such a file reads back with the result nested one level per "/"; flattening
    it to the encoded key keeps it usable. Returns the number of entries moved.
    """
    moved = 0
    for key in [k for k in uns if isinstance(k, str)]:
        if key.startswith(MARKERS_PREFIX):
            is_leaf = _is_markers
        elif key.startswith((ENRICHMENT_PREFIX, META_PREFIX)):
            is_leaf = _is_payload
        else:
            continue
        if is_leaf(uns[key]):
            continue
        for raw, value in list(_flatten(key, uns.pop(key), is_leaf)):
            uns[encode(raw)] = value
            moved += 1
    return moved
