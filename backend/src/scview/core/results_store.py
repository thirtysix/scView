"""Per-dataset results store: results kept on disk beside the dataset.

Many results live only in process memory (the last DE results, the last
ligand-receptor run, the last composition test, ...) and are lost on a restart,
when the dataset drops out of the cache, or when newer runs push them out. Each
module's ``remember`` also hands its result here; its reads fall back to here
when the in-memory copy is gone.

Layout (design: research/results-store/DESIGN.md)::

    DATA_DIR/results/{dataset_id}/
      index.json             one entry per result
      {result_id}/result.json  the result's structure and small values
      {result_id}/t0.parquet   each DataFrame it holds (zstd, narrowed types)
      {result_id}/arrays.npz   its numpy arrays and sparse matrices

A result is any dataclass / dict / list of plain values, DataFrames, arrays and
sparse matrices (``encode``); dataclasses are rebuilt by class name, only from
scview's own modules. Floats narrow to float32 except p-value columns (DE p goes
below float32's 1.2e-38), integers to the smallest type that holds them.

Each entry records a fingerprint of the data it was computed on (the cells, in
order, and the recipe's step count): a result whose fingerprint no longer
matches is ``stale`` and is not given to the co-pilot as a current fact.

Retention: per (kind, key) the newest is ``primary`` and older ones
``superseded``, of which the last ``KEEP_SUPERSEDED`` are kept; at most
``KEEP_PER_KIND`` primaries per kind; then unpinned results, superseded first and
oldest first, go until the folder is under ``MAX_BYTES`` (compressed, on disk).
Pinned results are never dropped.

Writes run on one background thread, so an endpoint never waits on the disk and
two writes to one index never race; a failed write is logged, never raised.
"""

from __future__ import annotations

import dataclasses
import importlib
import json
import logging
import re
import shutil
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp

logger = logging.getLogger(__name__)

KEEP_SUPERSEDED = 3
KEEP_PER_KIND = 20
MAX_BYTES = 200 * 1024**2
INDEX = "index.json"
FORMAT = 1
# p-values and their adjusted forms stay float64 (scRNA-seq DE gives p < 1e-38).
_P_COLUMN = re.compile(r"(^|_)(p|pval|pvals|pvalue|p_value|padj|pval_adj|pvals_adj|fdr|qval|q_value|"
                       r"spatial_fdr|p_adj|adj_p|fdr_bh)(_|$)|pval|fdr|padj|qval", re.IGNORECASE)
_SUBDIRS = ("derived", "ingested", "converted", "uploads", "cache")
_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")
# Dataclasses are rebuilt only from these modules (a result file names its classes).
TRUSTED_MODULES = ("scview.",)

_root: Path | None = None
_lock = threading.RLock()  # index reads and writes
_writer = ThreadPoolExecutor(max_workers=1, thread_name_prefix="results-store")


def configure(data_dir: str | Path) -> None:
    """Point the store at ``data_dir/results`` (the dataset manager does, so tests
    with their own data dir get their own store)."""
    global _root
    _root = Path(data_dir) / "results"


def root() -> Path:
    if _root is None:
        from scview.config import get_settings
        configure(get_settings().DATA_DIR)
    assert _root is not None
    return _root


def _safe(value: str | None) -> bool:
    return bool(value) and bool(_ID_RE.fullmatch(value))


def dataset_id_of(adaptor: Any) -> str | None:
    """The dataset an adaptor was opened for, from its file's path
    (``DATA_DIR/{derived|ingested|...}/{id}/...``), or None."""
    path = getattr(adaptor, "h5ad_path", None)
    if not path:
        return None
    parts = Path(path).parts
    for i in range(len(parts) - 2, -1, -1):
        if parts[i] in _SUBDIRS and _safe(parts[i + 1]):
            return parts[i + 1]
    return None


def fingerprint(adaptor: Any) -> dict[str, Any] | None:
    """The cells (in order) and the recipe's step count the adaptor holds now."""
    try:
        from scview.core import provenance
        adata = adaptor.adata
        return {"n_cells": int(adata.n_obs), "cells": adaptor.cells_fingerprint(),
                "n_steps": len(provenance.read_provenance(adata).get("history") or [])}
    except Exception:  # a test double, or no data
        return None


def _version() -> str:
    from scview.core import provenance
    return provenance._scview_version()


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

class _Parts:
    def __init__(self) -> None:
        self.tables: dict[str, pd.DataFrame] = {}
        self.arrays: dict[str, np.ndarray] = {}


def _encode(obj: Any, parts: _Parts) -> Any:
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, pd.DataFrame):
        name = f"t{len(parts.tables)}"
        parts.tables[name] = obj
        return {"__df__": name}
    if isinstance(obj, pd.Series):
        name = f"t{len(parts.tables)}"
        parts.tables[name] = obj.to_frame(name="__value__")
        return {"__series__": name, "name": _encode(obj.name, parts)}
    if sp.issparse(obj):
        m = obj.tocsc() if obj.format == "csc" else obj.tocsr()
        return {"__sparse__": m.format, "shape": list(m.shape),
                "data": _encode(m.data, parts), "indices": _encode(m.indices, parts),
                "indptr": _encode(m.indptr, parts)}
    if isinstance(obj, np.ndarray):
        if obj.dtype == object:
            if all(isinstance(v, str) for v in obj.ravel()):
                obj = obj.astype(str)
            else:
                return {"__list_array__": _encode(obj.tolist(), parts)}
        name = f"a{len(parts.arrays)}"
        parts.arrays[name] = obj
        return {"__nd__": name}
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        cls = type(obj)
        return {"__dc__": f"{cls.__module__}:{cls.__qualname__}",
                "f": {f.name: _encode(getattr(obj, f.name), parts) for f in dataclasses.fields(obj)}}
    if isinstance(obj, tuple):
        return {"__tuple__": [_encode(v, parts) for v in obj]}
    if isinstance(obj, (list, set, frozenset)):
        return [_encode(v, parts) for v in obj]
    if isinstance(obj, dict):
        if all(isinstance(k, str) and not k.startswith("__") for k in obj):
            return {k: _encode(v, parts) for k, v in obj.items()}
        return {"__items__": [[_encode(k, parts), _encode(v, parts)] for k, v in obj.items()]}
    raise TypeError(f"cannot store a {type(obj).__name__}")


def _decode(obj: Any, tables: dict[str, pd.DataFrame], arrays: dict[str, np.ndarray]) -> Any:
    if isinstance(obj, list):
        return [_decode(v, tables, arrays) for v in obj]
    if not isinstance(obj, dict):
        return obj
    if "__df__" in obj:
        return tables[obj["__df__"]]
    if "__series__" in obj:
        s = tables[obj["__series__"]]["__value__"]
        s.name = _decode(obj["name"], tables, arrays)
        return s
    if "__nd__" in obj:
        return arrays[obj["__nd__"]]
    if "__list_array__" in obj:
        vals = _decode(obj["__list_array__"], tables, arrays)
        out = np.empty(len(vals), dtype=object)
        out[:] = vals
        return out
    if "__sparse__" in obj:
        cls = sp.csc_matrix if obj["__sparse__"] == "csc" else sp.csr_matrix
        return cls((_decode(obj["data"], tables, arrays), _decode(obj["indices"], tables, arrays),
                    _decode(obj["indptr"], tables, arrays)), shape=tuple(obj["shape"]))
    if "__tuple__" in obj:
        return tuple(_decode(v, tables, arrays) for v in obj["__tuple__"])
    if "__items__" in obj:
        return {_hashable(_decode(k, tables, arrays)): _decode(v, tables, arrays) for k, v in obj["__items__"]}
    if "__dc__" in obj:
        module, _, name = obj["__dc__"].partition(":")
        if not module.startswith(TRUSTED_MODULES):
            raise ValueError(f"refusing to rebuild {obj['__dc__']}")
        cls: Any = importlib.import_module(module)
        for part in name.split("."):
            cls = getattr(cls, part)
        if not dataclasses.is_dataclass(cls):
            raise ValueError(f"{obj['__dc__']} is not a dataclass")
        inst = cls.__new__(cls)
        for k, v in obj["f"].items():
            object.__setattr__(inst, k, _decode(v, tables, arrays))
        return inst
    return {k: _decode(v, tables, arrays) for k, v in obj.items()}


def _hashable(v: Any) -> Any:
    return tuple(_hashable(x) for x in v) if isinstance(v, list) else v


def is_p_column(name: Any) -> bool:
    return bool(_P_COLUMN.search(str(name)))


def narrow(df: pd.DataFrame) -> pd.DataFrame:
    """Floats to float32 (not p-value columns), integers to the smallest type
    that holds their range. Repeated strings need nothing: Parquet's dictionary
    encoding stores each once (and a category dtype would change groupby)."""
    out = {}
    for col in df.columns:
        s = df[col]
        kind = s.dtype.kind
        if kind == "f" and s.dtype.itemsize > 4 and not is_p_column(col):
            s = s.astype(np.float32)
        elif kind in "iu" and len(s):
            s = pd.to_numeric(s, downcast="unsigned" if kind == "u" or s.min() >= 0 else "integer")
        out[col] = s
    narrowed = pd.DataFrame(out, index=df.index)
    narrowed.columns = df.columns
    return narrowed


def _write_table(df: pd.DataFrame, path: Path) -> None:
    df = narrow(df)
    renamed = [str(c) for c in df.columns]
    if renamed != list(df.columns):
        if len(set(renamed)) != len(renamed):
            raise TypeError("column names collide as text")
        df = df.copy()
        df.columns = renamed
    df.to_parquet(path, engine="pyarrow", compression="zstd", compression_level=3, index=True)


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------

def _dir(dataset_id: str) -> Path:
    if not _safe(dataset_id):
        raise ValueError("bad dataset id")
    return root() / dataset_id


def _read_index(dataset_id: str) -> list[dict[str, Any]]:
    path = _dir(dataset_id) / INDEX
    try:
        data = json.loads(path.read_text())
        return list(data.get("results") or [])
    except FileNotFoundError:
        return []
    except (ValueError, OSError) as exc:
        logger.warning("results index for %s unreadable (%s); starting a new one", dataset_id, exc)
        return []


def _write_index(dataset_id: str, entries: list[dict[str, Any]]) -> None:
    d = _dir(dataset_id)
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / f".{INDEX}.{uuid.uuid4().hex}"
    tmp.write_text(json.dumps({"format": FORMAT, "results": entries}, indent=1, default=str))
    tmp.replace(d / INDEX)


def _dir_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) if path.exists() else 0


def _retain(dataset_id: str, entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mark superseded results and drop what retention lets go (oldest first)."""
    entries = sorted(entries, key=lambda e: e["created_at"], reverse=True)
    seen: dict[tuple[str, str], int] = {}
    primaries: dict[str, int] = {}
    keep: list[dict[str, Any]] = []
    for e in entries:
        k = (e["kind"], e["key"])
        n = seen.get(k, 0)
        seen[k] = n + 1
        if e["status"] != "pinned":
            e["status"] = "primary" if n == 0 else "superseded"
        if e["status"] == "superseded" and n > KEEP_SUPERSEDED:
            continue
        if e["status"] == "primary":
            primaries[e["kind"]] = primaries.get(e["kind"], 0) + 1
            if primaries[e["kind"]] > KEEP_PER_KIND:
                continue
        keep.append(e)
    total = sum(e.get("bytes", 0) for e in keep)
    if total > MAX_BYTES:
        order = sorted((e for e in keep if e["status"] != "pinned"),
                       key=lambda e: (e["status"] == "primary", e["created_at"]))
        drop = set()
        for e in order:
            if total <= MAX_BYTES:
                break
            drop.add(e["id"])
            total -= e.get("bytes", 0)
        keep = [e for e in keep if e["id"] not in drop]
    kept_ids = {e["id"] for e in keep}
    for e in entries:
        if e["id"] not in kept_ids:
            shutil.rmtree(_dir(dataset_id) / e["id"], ignore_errors=True)
    return keep


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _save(dataset_id: str, entry: dict[str, Any], value: Any) -> None:
    rdir = _dir(dataset_id) / entry["id"]
    tmp = rdir.with_name(f".{entry['id']}.{uuid.uuid4().hex}")
    try:
        parts = _Parts()
        tree = _encode(value, parts)
        tmp.mkdir(parents=True)
        for name, df in parts.tables.items():
            _write_table(df, tmp / f"{name}.parquet")
        if parts.arrays:
            np.savez_compressed(tmp / "arrays.npz", **parts.arrays)
        (tmp / "result.json").write_text(json.dumps({"entry": entry, "value": tree}))
        entry["bytes"] = _dir_bytes(tmp)
        with _lock:
            shutil.rmtree(rdir, ignore_errors=True)
            tmp.replace(rdir)
            entries = [e for e in _read_index(dataset_id) if e["id"] != entry["id"]]
            _write_index(dataset_id, _retain(dataset_id, entries + [entry]))
    except Exception:
        logger.exception("could not store %s result %s for %s", entry.get("kind"), entry.get("id"), dataset_id)
        shutil.rmtree(tmp, ignore_errors=True)


def keep(
    target: Any,
    kind: str,
    value: Any,
    *,
    key: str = "",
    label: str = "",
    params: dict[str, Any] | None = None,
    result_id: str | None = None,
    adaptor: Any = None,
) -> Future | None:
    """Store ``value`` for a dataset (``target`` is its adaptor or its id) in the
    background; returns the write's future (tests wait on it), or None when the
    dataset is unknown. ``key`` names what the result is about: a newer result of
    the same kind and key supersedes it."""
    if isinstance(target, str):
        dataset_id = target
    else:
        adaptor = target
        dataset_id = dataset_id_of(target)
    if not _safe(dataset_id) or (result_id is not None and not _safe(result_id)):
        return None
    entry = {
        "id": result_id or uuid.uuid4().hex, "kind": kind, "key": str(key), "label": label,
        "params": params or {}, "created_at": time.time(), "version": _version(),
        "status": "primary", "fingerprint": fingerprint(adaptor) if adaptor is not None else None,
    }
    return _writer.submit(_save, dataset_id, entry, value)


def flush() -> None:
    """Wait for the writes queued so far."""
    _writer.submit(lambda: None).result()


def entries(dataset_id: str, kind: str | None = None, adaptor: Any = None) -> list[dict[str, Any]]:
    """The dataset's stored results, newest first, each with ``stale`` when the
    adaptor is given (None when it is not, or the entry has no fingerprint)."""
    if not _safe(dataset_id):
        return []
    with _lock:
        out = [dict(e) for e in _read_index(dataset_id) if kind is None or e["kind"] == kind]
    now = fingerprint(adaptor) if adaptor is not None else None
    for e in out:
        e["stale"] = None if now is None or not e.get("fingerprint") else e["fingerprint"] != now
    return sorted(out, key=lambda e: e["created_at"], reverse=True)


def load(dataset_id: str, result_id: str) -> tuple[dict[str, Any], Any] | None:
    """A stored result's entry and value, or None when it is gone or unreadable."""
    if not _safe(dataset_id) or not _safe(result_id):
        return None
    rdir = _dir(dataset_id) / result_id
    try:
        doc = json.loads((rdir / "result.json").read_text())
        tables = {p.stem: pd.read_parquet(p) for p in rdir.glob("t*.parquet")}
        arrays: dict[str, np.ndarray] = {}
        if (rdir / "arrays.npz").exists():
            with np.load(rdir / "arrays.npz", allow_pickle=False) as z:
                arrays = {k: z[k] for k in z.files}
        return doc["entry"], _decode(doc["value"], tables, arrays)
    except FileNotFoundError:
        return None
    except Exception:
        logger.exception("could not read stored result %s for %s", result_id, dataset_id)
        return None


def latest(adaptor: Any, kind: str, key: str | None = None, *, current_only: bool = True) -> Any | None:
    """The newest stored value of a kind (and key) for the adaptor's dataset;
    with ``current_only``, only one computed on the data the adaptor holds now."""
    dataset_id = dataset_id_of(adaptor)
    if dataset_id is None:
        return None
    for e in entries(dataset_id, kind, adaptor=adaptor if current_only else None):
        if key is not None and e["key"] != key:
            continue
        if current_only and e["stale"]:
            continue
        got = load(dataset_id, e["id"])
        if got is not None:
            return got[1]
    return None


def delete(dataset_id: str, result_id: str) -> bool:
    def _run() -> bool:
        with _lock:
            current = _read_index(dataset_id)
            left = [e for e in current if e["id"] != result_id]
            if len(left) == len(current):
                return False
            shutil.rmtree(_dir(dataset_id) / result_id, ignore_errors=True)
            _write_index(dataset_id, _retain(dataset_id, left))
            return True
    if not _safe(dataset_id) or not _safe(result_id):
        return False
    return _writer.submit(_run).result()


def delete_dataset(dataset_id: str) -> None:
    """Remove a dataset's results (after the writes queued for it)."""
    if _safe(dataset_id):
        _writer.submit(shutil.rmtree, _dir(dataset_id), True).result()


def remembered(cache: Any, lock: Any, adaptor: Any, kind: str, key: str | None = None) -> Any | None:
    """``cache[adaptor]`` (a module's in-memory last result), else the newest
    current stored one, put back into the cache. None for a test double."""
    try:
        with lock:
            value = cache.get(adaptor)
    except TypeError:  # not weak-referenceable
        return None
    if value is not None:
        return value
    value = latest(adaptor, kind, key)
    if value is not None:
        with lock:
            cache.setdefault(adaptor, value)
    return value


def recall(dataset_id: str, result_id: str, kind: str) -> Any | None:
    """A stored result's value by id, when it is of that kind."""
    got = load(dataset_id, result_id)
    return got[1] if got is not None and got[0].get("kind") == kind else None
