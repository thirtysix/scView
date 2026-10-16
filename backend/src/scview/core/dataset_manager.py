"""DatasetManager – load, cache (LRU), and manage single-cell datasets."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import shutil
import unicodedata
from pathlib import Path
from typing import Any

from scview.core.anndata_adaptor import AnnDataAdaptor
from scview.core.obs_writes import is_locked

logger = logging.getLogger(__name__)

# Dataset and ingest-session ids are generated here (uuid4, hex or with dashes).
# Anything else, e.g. ".." arriving as %2E%2E in a URL, must never become a path.
_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")


def is_safe_id(value: str) -> bool:
    """True for an id that is safe to use as one path component."""
    return bool(_ID_RE.fullmatch(value or ""))


DATASET_NAME_MAX = 120
# Control characters, invisible format characters (bidi overrides, zero-width
# spaces) and line/paragraph separators: none belong in a one-line display name.
_BAD_NAME_CATEGORIES = {"Cc", "Cf", "Cs", "Zl", "Zp"}


def clean_dataset_name(value: object) -> str:
    """Return a user-supplied dataset name trimmed, or raise ValueError saying why
    it can't be used."""
    if not isinstance(value, str):
        raise ValueError("Name must be text.")
    name = value.strip()
    if not name:
        raise ValueError("Name must not be empty.")
    if len(name) > DATASET_NAME_MAX:
        raise ValueError(f"Name is too long (max {DATASET_NAME_MAX} characters).")
    if any(unicodedata.category(ch) in _BAD_NAME_CATEGORIES for ch in name):
        raise ValueError("Name must not contain control or invisible characters.")
    return name


# How long a request waits for a pipeline run on its dataset: under nginx's
# proxy_read_timeout (300 s), so the client gets a 503 it can retry, not a 504.
BUSY_WAIT_SECONDS = 240.0


class DatasetBusyError(Exception):
    """A pipeline run kept the dataset busy for longer than a request waits."""

    def __init__(self, dataset_id: str) -> None:
        super().__init__(f"Dataset {dataset_id} is being processed.")
        self.dataset_id = dataset_id


class DatasetManager:
    """Manages the lifecycle of dataset adaptors with LRU eviction."""

    def __init__(self, data_dir: str, max_datasets: int = 3) -> None:
        self.data_dir = Path(data_dir)
        self.max_datasets = max_datasets
        self._datasets: dict[str, AnnDataAdaptor] = {}
        self._load_order: list[str] = []
        # Loads in progress, so concurrent requests for one dataset share a read.
        self._loading: dict[str, asyncio.Future] = {}
        # Bumped by evict/remove: a load that started before must not be cached.
        self._generation: dict[str, int] = {}
        # Datasets a pipeline run is changing in a worker thread (see core/processing.py).
        self._busy: dict[str, asyncio.Event] = {}

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    async def load_dataset(self, dataset_id: str) -> AnnDataAdaptor | None:
        """Load (or promote in LRU) a dataset and return its adaptor.

        The file is read in a worker thread (seconds for a large dataset), so the
        server keeps answering other requests meanwhile.
        """
        if dataset_id in self._datasets:
            self._load_order.remove(dataset_id)
            self._load_order.append(dataset_id)
            return self._datasets[dataset_id]

        pending = self._loading.get(dataset_id)
        if pending is None:
            pending = asyncio.ensure_future(self._load(dataset_id))
            self._loading[dataset_id] = pending

            def _forget(fut: asyncio.Future, did: str = dataset_id) -> None:
                if self._loading.get(did) is fut:
                    del self._loading[did]

            pending.add_done_callback(_forget)
        # Shielded: one caller giving up must not cancel the read the others wait on.
        return await asyncio.shield(pending)

    async def _load(self, dataset_id: str) -> AnnDataAdaptor | None:
        h5ad_path = self._resolve_h5ad(dataset_id)
        if h5ad_path is None:
            return None
        generation = self._generation.get(dataset_id, 0)

        # Make room first, so the new dataset never sits beside a full cache.
        self._evict_lru(keep=dataset_id, room_for_one=True)

        adaptor = AnnDataAdaptor(h5ad_path)
        updates = await asyncio.to_thread(self._metadata_updates, dataset_id, adaptor)
        if self._generation.get(dataset_id, 0) != generation:
            # Evicted while reading (its file was rewritten or removed): this copy
            # may be stale, so hand it to the requests waiting on it but don't cache it.
            return adaptor
        if updates:
            # Update metadata sidecar with actual cell/gene counts and embeddings
            self.update_dataset_metadata(dataset_id, updates)
        self._datasets[dataset_id] = adaptor
        self._load_order.append(dataset_id)
        self._evict_lru(keep=dataset_id)  # two loads may have finished at once
        return adaptor

    def _evict_lru(self, keep: str, room_for_one: bool = False) -> None:
        """Drop least-recently-used datasets beyond capacity, never ``keep`` or a
        dataset a pipeline run is still working on."""
        limit = self.max_datasets - 1 if room_for_one else self.max_datasets
        for evict_id in list(self._load_order):
            if len(self._datasets) <= limit:
                break
            # Never one a request is working on: its adaptor would re-read the file
            # and lose what the request computed (markers, a pipeline run).
            if evict_id == keep or evict_id in self._busy or is_locked(evict_id):
                continue
            self._load_order.remove(evict_id)
            adaptor = self._datasets.pop(evict_id, None)
            if adaptor:
                adaptor.close()
                logger.info("Evicted dataset %s from cache", evict_id)

    def get_dataset(self, dataset_id: str) -> AnnDataAdaptor | None:
        return self._datasets.get(dataset_id)

    async def get_or_load_dataset(self, dataset_id: str) -> AnnDataAdaptor | None:
        """Get from cache or load from disk.

        While a pipeline run is changing the dataset this waits for it to finish,
        so no request reads a half-processed dataset; after BUSY_WAIT_SECONDS it
        raises DatasetBusyError (a 503), before a proxy gives up on the request.
        """
        if dataset_id in self._busy:
            try:
                await asyncio.wait_for(self.wait_until_idle(dataset_id), BUSY_WAIT_SECONDS)
            except TimeoutError:
                raise DatasetBusyError(dataset_id) from None
        # load_dataset also marks a cached dataset as recently used.
        return await self.load_dataset(dataset_id)

    # ------------------------------------------------------------------
    # Processing state
    # ------------------------------------------------------------------

    def is_busy(self, dataset_id: str) -> bool:
        """True while a pipeline run is changing this dataset."""
        return dataset_id in self._busy

    def mark_busy(self, dataset_id: str) -> None:
        self._busy.setdefault(dataset_id, asyncio.Event())

    def mark_idle(self, dataset_id: str) -> None:
        event = self._busy.pop(dataset_id, None)
        if event is not None:
            event.set()

    async def wait_until_idle(self, dataset_id: str) -> None:
        while (event := self._busy.get(dataset_id)) is not None:
            await event.wait()

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
        if not is_safe_id(dataset_id):
            return None
        meta_path = self.data_dir / "uploads" / dataset_id / "metadata.json"
        if not meta_path.exists():
            return None
        with open(meta_path) as fh:
            return json.load(fh)

    def update_dataset_metadata(self, dataset_id: str, updates: dict[str, Any]) -> None:
        """Merge updates into the metadata sidecar JSON."""
        if not is_safe_id(dataset_id):
            return
        meta_path = self.data_dir / "uploads" / dataset_id / "metadata.json"
        if not meta_path.exists():
            return
        with open(meta_path) as fh:
            meta = json.load(fh)
        meta.update(updates)
        with open(meta_path, "w") as fh:
            json.dump(meta, fh, indent=2)

    def rename_dataset(self, dataset_id: str, name: object) -> str | None:
        """Change a dataset's display name. Returns the stored name, or None if the
        dataset doesn't exist; raises ValueError for a name that can't be used.

        The name lives only in the metadata sidecar (never in the .h5ad), so this
        doesn't touch the data or the cached adaptor."""
        if self.get_dataset_info(dataset_id) is None:
            return None
        clean = clean_dataset_name(name)
        self.update_dataset_metadata(dataset_id, {"name": clean})
        return clean

    # ------------------------------------------------------------------
    # Removal
    # ------------------------------------------------------------------

    def evict(self, dataset_id: str) -> None:
        """Drop the cached adaptor so the next access reloads from disk.

        Pipeline steps change the cached AnnData in place and only write the file
        at the end, so after a failed or abandoned run the cache can hold changes
        that were never saved; re-running steps on it would apply them twice.
        """
        self._generation[dataset_id] = self._generation.get(dataset_id, 0) + 1
        self._loading.pop(dataset_id, None)  # the next access reads the file again
        adaptor = self._datasets.pop(dataset_id, None)
        if adaptor:
            adaptor.close()
        if dataset_id in self._load_order:
            self._load_order.remove(dataset_id)

    def remove_dataset(self, dataset_id: str) -> bool:
        if not is_safe_id(dataset_id):
            return False
        self._generation[dataset_id] = self._generation.get(dataset_id, 0) + 1
        self._loading.pop(dataset_id, None)
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
        if not is_safe_id(dataset_id):
            return None
        for subdir in self._H5AD_SUBDIRS:
            d = self.data_dir / subdir / dataset_id
            if d.exists():
                for f in sorted(d.iterdir()):
                    if f.suffix == ".h5ad":
                        return str(f)
        return None

    def h5ad_candidates(self, dataset_id: str) -> list[str]:
        """Every .h5ad stored for a dataset, in the same precedence as
        :meth:`_resolve_h5ad` (active file first, then its sources)."""
        if not is_safe_id(dataset_id):
            return []
        found: list[str] = []
        for subdir in self._H5AD_SUBDIRS:
            d = self.data_dir / subdir / dataset_id
            if d.exists():
                found.extend(str(f) for f in sorted(d.iterdir()) if f.suffix == ".h5ad")
        return found

    def derived_h5ad_path(self, dataset_id: str, source_path: str) -> Path:
        """Return the output path for pipeline/processing results.

        Processing output always lands under ``derived/{id}/`` so the user's
        original upload (``uploads/``) and any RDS→h5ad conversion
        (``converted/``) remain untouched — an interrupted or buggy pipeline
        run can never clobber an irreplaceable source file. Re-running the
        pipeline overwrites only the (regenerable) derived file.
        """
        if not is_safe_id(dataset_id):
            raise ValueError(f"Invalid dataset id: {dataset_id!r}")
        stem = Path(source_path).stem
        d = self.data_dir / "derived" / dataset_id
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{stem}.h5ad"

    def persist_uns(self, dataset_id: str, adaptor: AnnDataAdaptor, keys: list[str]) -> bool:
        """Save ``uns`` entries the cached dataset gained since it was loaded.

        When the loaded file is already the derived one, only those entries are
        rewritten in place, far cheaper than writing the whole file again;
        otherwise the dataset is written to ``derived/`` like any processing
        result, so originals stay untouched. A backed dataset's file is open
        read-only, so its entries stay in memory. Returns whether it saved.
        """
        adata = adaptor.adata
        if getattr(adata, "isbacked", False):
            logger.info("Dataset %s is backed; not saving %s", dataset_id, keys)
            return False
        target = self.derived_h5ad_path(dataset_id, adaptor.h5ad_path)
        if Path(adaptor.h5ad_path) == target and target.exists():
            import h5py
            from anndata.io import write_elem

            with h5py.File(target, "a") as f:
                if "uns" not in f:
                    write_elem(f, "uns", {})
                group = f["uns"]
                for key in keys:
                    if key in group:
                        del group[key]
                    write_elem(group, key, adata.uns[key])
        else:
            adata.write_h5ad(target, compression="gzip")
            adaptor.h5ad_path = str(target)
        return True

    def saved_in_place(self, dataset_id: str, adaptor: AnnDataAdaptor, path: str) -> None:
        """Record that the cached ``adaptor`` was edited and saved to ``path``, so
        it stays cached (no reload) and the library sees its new columns."""
        adaptor.h5ad_path = str(path)
        self._update_metadata_from_adaptor(dataset_id, adaptor)

    def _update_metadata_from_adaptor(
        self, dataset_id: str, adaptor: AnnDataAdaptor
    ) -> None:
        """Update the metadata JSON with info extracted from the loaded AnnData."""
        updates = self._metadata_updates(dataset_id, adaptor)
        if updates:
            self.update_dataset_metadata(dataset_id, updates)

    @staticmethod
    def _metadata_updates(dataset_id: str, adaptor: AnnDataAdaptor) -> dict[str, Any] | None:
        """The sidecar fields read from the AnnData (reading the file on first use),
        or None when it can't be read."""
        try:
            embeddings = adaptor.available_embeddings()
            return {
                "n_cells": adaptor.n_cells(),
                "n_genes": adaptor.n_genes(),
                "available_embeddings": [e["name"] for e in embeddings],
                "embedding_dimensions": {e["name"]: e["dimensions"] for e in embeddings},
                "obs_columns": adaptor.obs_columns_info(),
                "active_clustering": adaptor.active_clustering_column(),
                "user_annotation_columns": adaptor.user_annotation_columns(),
                # Kept so the dataset can be described while a run works on it.
                "expression_layers": adaptor.available_expression_layers(),
                "default_expression_layer": adaptor.default_expression_layer(),
                "status": "ready",
            }
        except Exception as e:
            logger.error("Failed to update metadata for %s: %s", dataset_id, e)
            return None
