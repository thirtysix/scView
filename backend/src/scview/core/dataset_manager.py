"""DatasetManager – load, cache (LRU), and manage single-cell datasets."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

from scview.core.anndata_adaptor import AnnDataAdaptor

logger = logging.getLogger(__name__)


class DatasetManager:
    """Manages the lifecycle of dataset adaptors with LRU eviction."""

    def __init__(self, data_dir: str, max_datasets: int = 3) -> None:
        self.data_dir = Path(data_dir)
        self.max_datasets = max_datasets
        self._datasets: dict[str, AnnDataAdaptor] = {}
        self._load_order: list[str] = []

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    async def load_dataset(self, dataset_id: str) -> AnnDataAdaptor | None:
        """Load (or promote in LRU) a dataset and return its adaptor."""
        if dataset_id in self._datasets:
            self._load_order.remove(dataset_id)
            self._load_order.append(dataset_id)
            return self._datasets[dataset_id]

        h5ad_path = self._resolve_h5ad(dataset_id)
        if h5ad_path is None:
            return None

        # Evict least-recently-used if at capacity
        while len(self._datasets) >= self.max_datasets and self._load_order:
            evict_id = self._load_order.pop(0)
            adaptor = self._datasets.pop(evict_id, None)
            if adaptor:
                adaptor.close()
                logger.info("Evicted dataset %s from cache", evict_id)

        adaptor = AnnDataAdaptor(h5ad_path)
        self._datasets[dataset_id] = adaptor
        self._load_order.append(dataset_id)

        # Update metadata sidecar with actual cell/gene counts and embeddings
        self._update_metadata_from_adaptor(dataset_id, adaptor)

        return adaptor

    def get_dataset(self, dataset_id: str) -> AnnDataAdaptor | None:
        return self._datasets.get(dataset_id)

    async def get_or_load_dataset(self, dataset_id: str) -> AnnDataAdaptor | None:
        """Get from cache or load from disk."""
        adaptor = self.get_dataset(dataset_id)
        if adaptor is None:
            adaptor = await self.load_dataset(dataset_id)
        return adaptor

    # ------------------------------------------------------------------
    # Listing / info
    # ------------------------------------------------------------------

    # Statuses for datasets still being prepared (no data file yet is expected).
    _IN_PROGRESS_STATUSES = ("converting", "pending")

    def _is_openable(self, dataset_id: str, status: str) -> bool:
        """A dataset is openable if it's still being prepared, or has a data file."""
        if status in self._IN_PROGRESS_STATUSES:
            return True
        return self._resolve_h5ad(dataset_id) is not None

    def list_datasets(self) -> list[dict[str, Any]]:
        """Return only openable datasets — those with a resolvable data file (or
        still converting). Orphaned entries (failed conversions, missing files)
        are skipped so the picker never shows a dataset that can't be opened.
        """
        results: list[dict[str, Any]] = []
        uploads_dir = self.data_dir / "uploads"
        if not uploads_dir.exists():
            return results
        for entry in sorted(uploads_dir.iterdir()):
            if entry.is_dir():
                meta_path = entry / "metadata.json"
                if meta_path.exists():
                    with open(meta_path) as fh:
                        meta = json.load(fh)
                    if not self._is_openable(entry.name, meta.get("status", "")):
                        continue
                    # The picker only needs id/name/counts/embeddings and obs
                    # column names — never the inlined category `values`, which
                    # can be tens of MB per dataset for high-cardinality columns.
                    # Full values are served per-dataset by GET /datasets/{id}.
                    for col in meta.get("obs_columns") or []:
                        if isinstance(col, dict):
                            col.pop("values", None)
                    results.append(meta)
        return results

    def prune_orphans(self) -> list[dict[str, str]]:
        """Delete dataset dirs that can't be opened (failed conversions, missing
        data). Returns the removed [{id, name}] so the caller can report them."""
        removed: list[dict[str, str]] = []
        uploads_dir = self.data_dir / "uploads"
        if not uploads_dir.exists():
            return removed
        for entry in sorted(uploads_dir.iterdir()):
            meta_path = entry / "metadata.json"
            if not (entry.is_dir() and meta_path.exists()):
                continue
            with open(meta_path) as fh:
                meta = json.load(fh)
            if not self._is_openable(entry.name, meta.get("status", "")):
                if self.remove_dataset(entry.name):
                    removed.append({"id": entry.name, "name": meta.get("name", entry.name)})
        return removed

    def get_dataset_info(self, dataset_id: str) -> dict[str, Any] | None:
        meta_path = self.data_dir / "uploads" / dataset_id / "metadata.json"
        if not meta_path.exists():
            return None
        with open(meta_path) as fh:
            return json.load(fh)

    def update_dataset_metadata(self, dataset_id: str, updates: dict[str, Any]) -> None:
        """Merge updates into the metadata sidecar JSON."""
        meta_path = self.data_dir / "uploads" / dataset_id / "metadata.json"
        if not meta_path.exists():
            return
        with open(meta_path) as fh:
            meta = json.load(fh)
        meta.update(updates)
        with open(meta_path, "w") as fh:
            json.dump(meta, fh, indent=2)

    # ------------------------------------------------------------------
    # Removal
    # ------------------------------------------------------------------

    def remove_dataset(self, dataset_id: str) -> bool:
        adaptor = self._datasets.pop(dataset_id, None)
        if adaptor:
            adaptor.close()
        if dataset_id in self._load_order:
            self._load_order.remove(dataset_id)

        found = False
        for subdir in ("uploads", "converted", "cache", "derived"):
            d = self.data_dir / subdir / dataset_id
            if d.exists():
                shutil.rmtree(d)
                found = True
        return found

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    # Subdirectory precedence for resolving a dataset's active .h5ad.
    # `derived` holds pipeline output and is preferred so processing results
    # are seen; `ingested` holds the canonical h5ad produced by the ingestion
    # engine (10x/CSV/loom → h5ad); `converted` holds RDS→h5ad output; `uploads`
    # holds the user's original file(s). Originals, ingests and conversions are
    # NEVER written to after the initial ingest — all processing writes to
    # `derived` (see derived_h5ad_path).
    _H5AD_SUBDIRS = ("derived", "ingested", "converted", "uploads")

    def _resolve_h5ad(self, dataset_id: str) -> str | None:
        """Find the active .h5ad (derived > ingested > converted > uploaded)."""
        for subdir in self._H5AD_SUBDIRS:
            d = self.data_dir / subdir / dataset_id
            if d.exists():
                for f in sorted(d.iterdir()):
                    if f.suffix == ".h5ad":
                        return str(f)
        return None

    def derived_h5ad_path(self, dataset_id: str, source_path: str) -> Path:
        """Return the output path for pipeline/processing results.

        Processing output always lands under ``derived/{id}/`` so the user's
        original upload (``uploads/``) and any RDS→h5ad conversion
        (``converted/``) remain untouched — an interrupted or buggy pipeline
        run can never clobber an irreplaceable source file. Re-running the
        pipeline overwrites only the (regenerable) derived file.
        """
        stem = Path(source_path).stem
        d = self.data_dir / "derived" / dataset_id
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{stem}.h5ad"

    def _update_metadata_from_adaptor(
        self, dataset_id: str, adaptor: AnnDataAdaptor
    ) -> None:
        """Update the metadata JSON with info extracted from the loaded AnnData."""
        try:
            embeddings = adaptor.available_embeddings()
            obs_info = adaptor.obs_columns_info()
            self.update_dataset_metadata(dataset_id, {
                "n_cells": adaptor.n_cells(),
                "n_genes": adaptor.n_genes(),
                "available_embeddings": [e["name"] for e in embeddings],
                "embedding_dimensions": {e["name"]: e["dimensions"] for e in embeddings},
                "obs_columns": obs_info,
                "active_clustering": adaptor.active_clustering_column(),
                "status": "ready",
            })
        except Exception as e:
            logger.error("Failed to update metadata for %s: %s", dataset_id, e)
