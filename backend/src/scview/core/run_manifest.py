"""RUN_MANIFEST: what a dataset's processing ran on, beside the replay script (C006).

One JSON document: the app and package versions, the dataset (id, name, shape),
its input (the recorded source and the stored file's content hash), and each
recorded step with its params, time, scView version and how the replay script
(``core/replay_script.py``) reproduces it.
"""

from __future__ import annotations

import hashlib
import platform
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from scview.core import provenance, replay_script

MANIFEST_VERSION = 1

# Distribution names whose versions shape the results (those installed are listed).
PACKAGES = (
    "scview", "scanpy", "anndata", "numpy", "scipy", "pandas", "scikit-learn", "umap-learn",
    "pynndescent", "igraph", "leidenalg", "harmonypy", "scikit-misc", "celltypist", "gseapy",
    "pydeseq2", "decoupler", "h5py", "numba",
)

_HASH_CHUNK = 1 << 20
# path -> (size, mtime_ns, sha256): a source file is never rewritten, so hash it once.
_hash_cache: dict[str, tuple[int, int, str]] = {}


def package_versions() -> dict[str, str]:
    out: dict[str, str] = {}
    for name in PACKAGES:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            continue
    return out


def file_sha256(path: str | Path) -> str:
    """The file's SHA-256 (hex), read in 1 MB chunks; cached by size and mtime."""
    p = Path(path)
    st = p.stat()
    key = str(p.resolve())
    hit = _hash_cache.get(key)
    if hit and hit[:2] == (st.st_size, st.st_mtime_ns):
        return hit[2]
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(_HASH_CHUNK), b""):
            h.update(chunk)
    digest = h.hexdigest()
    _hash_cache[key] = (st.st_size, st.st_mtime_ns, digest)
    return digest


def input_file(candidates: list[str]) -> dict[str, Any] | None:
    """The stored source file (the first candidate not under ``derived/``: the
    import, the conversion or the upload, never rewritten by processing), with
    its size and SHA-256. None when there is none."""
    for c in candidates:
        p = Path(c)
        if p.parent.parent.name == "derived" or not p.is_file():
            continue
        try:
            return {"name": p.name, "stored_as": p.parent.parent.name, "size_bytes": p.stat().st_size,
                    "sha256": file_sha256(p)}
        except OSError:
            return {"name": p.name, "stored_as": p.parent.parent.name, "sha256": None}
    return None


def build_manifest(adata, *, dataset_id: str, dataset_name: str,
                   candidates: list[str] | None = None) -> dict[str, Any]:
    """The manifest for ``adata``; ``candidates`` are the dataset's stored .h5ad
    files (``DatasetManager.h5ad_candidates``) for the input hash."""
    prov = provenance.read_provenance(adata)
    history = [h for h in prov.get("history", []) if isinstance(h, dict)]
    replay = replay_script.describe(adata)
    steps = []
    for i, (h, r) in enumerate(zip(history, replay), 1):
        steps.append({
            "index": i,
            "step": h.get("step"),
            "tool": h.get("tool"),
            "params": h.get("params", {}),
            "timestamp": h.get("timestamp"),
            "scview_version": h.get("scview_version"),
            "effect": h.get("effect"),
            "commit_id": h.get("commit_id"),
            "parent": h.get("parent"),
            "replay": r,
        })
    source = prov.get("source") or {}
    inp = input_file(candidates or [])
    layers = list(getattr(adata, "layers", {}).keys())
    return {
        "manifest_version": MANIFEST_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "app": {"name": "scView", "version": provenance._scview_version()},
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "packages": package_versions(),
        },
        "dataset": {
            "id": dataset_id,
            "name": dataset_name,
            "n_cells": int(adata.n_obs),
            "n_genes": int(adata.n_vars),
            "layers": layers,
            "raw": getattr(adata, "raw", None) is not None,
            "embeddings": sorted(k for k in getattr(adata, "obsm", {}).keys()),
            "active_clustering": _str_or_none(adata.uns.get("scview_active_clustering")),
        },
        "input": {
            "original_filename": source.get("original_filename"),
            "format": source.get("format"),
            "origin": source.get("origin"),
            "ingested_at": source.get("ingested_at"),
            "n_cells": source.get("n_cells"),
            "n_genes": source.get("n_genes"),
            "merged_from": source.get("merged_from"),
            "stored_file": inp,
            "hash_note": None if inp and inp.get("sha256") else
            "No stored source file to hash: the dataset may have been processed outside scView.",
        },
        "head": (prov.get("current") or {}).get("head"),
        "reconcile_issues": provenance.reconcile(adata),
        "replay_summary": _summary(replay),
        "steps": steps,
    }


def _summary(replay: list[dict]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in replay:
        out[r["status"]] = out.get(r["status"], 0) + 1
    return out


def _str_or_none(v: Any) -> str | None:
    return None if v is None else str(v)
