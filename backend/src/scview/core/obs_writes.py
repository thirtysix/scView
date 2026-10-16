"""Saving a dataset after an obs edit made outside the pipeline.

Hand labels (``core/user_labels.py``) and a kept clustering resolution
(``core/resolution_sweep.py``) each add or change one obs column, which only a
full write of the derived file can save. Writes to one dataset are serialised:
two requests writing the same ``.partial`` file at once would corrupt it.
"""

from __future__ import annotations

import asyncio
import os

from scview.core.uns_keys import markers_key

_locks: dict[str, asyncio.Lock] = {}


def forget_column_results(adata, column: str) -> None:
    """Drop the markers and cached enrichment computed for ``column``.

    Both are keyed by group name, so after cells change group a surviving label
    would otherwise be served results for its old members. Markers are computed
    again on the next visit to the column.
    """
    from scview.core.pipeline import drop_enrichment_cache

    adata.uns.pop(markers_key(column), None)
    rgg = adata.uns.get("rank_genes_groups")
    if isinstance(rgg, dict) and rgg.get("params", {}).get("groupby") == column:
        del adata.uns["rank_genes_groups"]
    drop_enrichment_cache(adata, column)


def dataset_lock(dataset_id: str) -> asyncio.Lock:
    """The lock held while a request edits and saves this dataset."""
    lock = _locks.get(dataset_id)
    if lock is None:
        lock = _locks[dataset_id] = asyncio.Lock()
    return lock


def write_derived(adata, output_path: str) -> None:
    """Write ``adata`` to its derived file: beside it first, then swapped in, so a
    failed write leaves the old file and a backed dataset reading it intact."""
    partial = output_path + ".partial"
    adata.write_h5ad(partial, compression="gzip")
    os.replace(partial, output_path)
