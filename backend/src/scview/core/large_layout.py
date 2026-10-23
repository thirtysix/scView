"""Files of a dataset processed in large (chunked) mode: the one place that says
where they are and what they hold.

A large processed dataset is its derived h5ad plus two sidecars beside it:

- ``<stem>.h5ad``: X = the log-normalised values, CSR float32, every kept gene;
  obs, var, obsm, obsp, varm and uns as the in-memory pipeline stores them. No
  ``.raw`` (X already is the log-normalised matrix, so .raw would be a second
  copy) and no layers (``backed="r"`` reads layers fully into memory).
- ``<stem>.counts.h5``: the filtered counts, CSR, under ``counts``
  (``chunked.write_counts``), with the cell and gene names' digests as attributes.
- ``<stem>.genes.h5``: the gene store of X (``core/gene_store.py``), gene-major,
  for per-gene reads and the marker test.

``uns["scview_large"]`` marks the dataset and names the sidecars relative to the
h5ad, so a renamed or copied folder still resolves them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py

from scview.core import gene_store

KEY = "scview_large"
VERSION = 1
COUNTS_KEY = "counts"
COUNTS_SUFFIX = ".counts.h5"
GENES_SUFFIX = ".genes.h5"
_CELLS_ATTR = "scview_cells_digest"
_GENES_ATTR = "scview_genes_digest"


class LargeLayoutError(RuntimeError):
    """A sidecar is missing or holds other cells or genes than the dataset."""


def sidecar_paths(h5ad_path: str | Path) -> tuple[Path, Path]:
    """(counts, genes) sidecar paths of a derived h5ad."""
    p = Path(h5ad_path)
    return p.with_name(p.stem + COUNTS_SUFFIX), p.with_name(p.stem + GENES_SUFFIX)


def record(uns: dict, h5ad_path: str | Path) -> dict:
    """Mark ``uns`` as a large dataset whose sidecars sit beside ``h5ad_path``."""
    counts, genes = sidecar_paths(h5ad_path)
    uns[KEY] = {"version": VERSION, "counts": counts.name, "genes": genes.name}
    return uns[KEY]


def is_large(uns_or_adata: Any) -> bool:
    uns = getattr(uns_or_adata, "uns", uns_or_adata)
    return isinstance(uns, dict) and isinstance(uns.get(KEY), dict)


def _named(h5ad_path: str | Path, uns: dict, which: str) -> Path:
    info = uns.get(KEY) or {}
    name = info.get(which)
    if not name:
        raise LargeLayoutError(f"Not a large dataset: uns['{KEY}'] names no {which} sidecar.")
    return Path(h5ad_path).with_name(str(name))


def counts_path(h5ad_path: str | Path, uns: dict) -> Path:
    return _named(h5ad_path, uns, "counts")


def genes_path(h5ad_path: str | Path, uns: dict) -> Path:
    return _named(h5ad_path, uns, "genes")


def stamp_counts(path: str | Path, obs_names, var_names) -> None:
    """Write the cell and gene names' digests on a counts sidecar."""
    with h5py.File(path, "a") as f:
        f[COUNTS_KEY].attrs[_CELLS_ATTR] = gene_store.names_digest(obs_names)
        f[COUNTS_KEY].attrs[_GENES_ATTR] = gene_store.names_digest(var_names)


def counts_mismatch(path: str | Path, obs_names, var_names) -> str | None:
    """Why the counts sidecar does not belong to these cells and genes, or None."""
    path = Path(path)
    if not path.exists():
        return f"{path.name} is missing"
    with h5py.File(path, "r") as f:
        if COUNTS_KEY not in f:
            return f"{path.name} holds no '{COUNTS_KEY}' matrix"
        attrs = f[COUNTS_KEY].attrs
        if attrs.get(_CELLS_ATTR) != gene_store.names_digest(obs_names):
            return f"{path.name} holds other cells"
        if attrs.get(_GENES_ATTR) != gene_store.names_digest(var_names):
            return f"{path.name} holds other genes"
    return None


def remove_sidecars(h5ad_path: str | Path) -> None:
    """Delete both sidecars (and any unfinished gene-store build) of a derived h5ad."""
    counts, genes = sidecar_paths(h5ad_path)
    counts.unlink(missing_ok=True)
    counts.with_name(f".{counts.name}.part").unlink(missing_ok=True)
    gene_store.remove(genes)
