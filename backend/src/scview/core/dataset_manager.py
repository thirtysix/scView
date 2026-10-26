"""DatasetManager – load, cache (LRU), and manage single-cell datasets."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import dataclasses
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scview.core import large_layout, large_mode, memory_budget, results_store, uns_history
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


class DatasetMemoryError(Exception):
    """Opening a dataset would need more memory than scView has, even with every
    other dataset closed: refused, instead of the system stopping the backend."""

    def __init__(self, dataset_id: str, need: int, available: int, limit: int,
                 what: str = "Opening this dataset") -> None:
        gb = 1024**3
        super().__init__(
            f"{what} needs about {need / gb:.1f} GB at its peak; scView has "
            f"{available / gb:.1f} GB free of its {limit / gb:.1f} GB limit, even with the other "
            "datasets closed. Raise SCVIEW_MEMORY_GB in .env (then restart scView).")
        self.dataset_id = dataset_id


# Room to leave beside a dataset's estimated load peak before opening it.
LOAD_MARGIN = 1.1
# Memory kept free beyond a load or run's own need, for the requests that follow it:
# opening Kang beside a cached ovary (3.2 GB) left 0.4 GB of a 6 GB limit, and the
# page's first requests (QC, cluster QC, the co-pilot insight) got the backend killed.
RESERVE_SHARE = 0.25
RESERVE_MIN = 1024**3
# How long a load that does not fit waits for memory a finished run is still freeing.
REFUSAL_WAIT_SECONDS = 10


def working_reserve(limit_bytes: int) -> int:
    return max(RESERVE_MIN, int(RESERVE_SHARE * limit_bytes))


# Memory a gene-store rebuild of a large dataset takes while it runs: one staged row chunk
# and its CSC copy, then one gene block (core/gene_store.py: 0.80 GB peak RSS for the whole
# process at 100k cells, 0.93 GB at 1.3M).
GENE_STORE_BUILD_BYTES = 768 * 1024**2


@dataclass
class LoadPlan:
    """How a dataset opens: its load peak in memory (None when it opens backed whatever
    happens: a large dataset, a file over ``BACKED_MIN_BYTES``) and backed."""

    in_memory: int | None
    backed: int


def load_plan(path: str) -> LoadPlan:
    """Memory opening ``path`` peaks at, in memory and backed (``preflight.estimate_load``
    from its header). A large dataset opens backed at any size: its X and sidecars stay
    on disk, so it holds its tables, embeddings, graphs and results, plus its open files
    (``large_mode``)."""
    from scview.core import preflight

    h = preflight.read_header(path)
    h5ad = getattr(h, "format", None) == "h5ad"
    large = h5ad and large_mode.file_is_large(path)
    in_memory = None
    if not (large or getattr(h, "backed", False)):
        in_memory = int(preflight.estimate_load(h, ingest=False).peak_bytes)
    if h5ad:
        h = dataclasses.replace(h, backed=True)
    backed = int(preflight.estimate_load(h, ingest=False).peak_bytes)
    held = large_mode.held_bytes(getattr(h, "n_cells", 0) or 0) if large else 0
    return LoadPlan(in_memory, backed + held)


def load_peak_bytes(path: str) -> int:
    """The load peak of ``path`` as it would open with memory to spare (``load_plan``)."""
    plan = load_plan(path)
    return plan.backed if plan.in_memory is None else plan.in_memory


def large_derived(adaptor: AnnDataAdaptor, target: str | Path) -> None:
    """Before an in-place edit of a large dataset opened from somewhere other than its
    derived file (an uploaded large layout): the h5ad is copied to ``target`` (once; X is
    copied, never rewritten) and the sidecars linked beside it, and the adaptor reads the
    copy from then on. Originals are never written."""
    target = Path(target)
    source = Path(adaptor.h5ad_path)
    if source == target:
        return
    adata = adaptor.adata
    sidecars = (large_layout.counts_path(source, adata.uns), large_layout.genes_path(source, adata.uns))
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(f".{target.name}.part")
    shutil.copyfile(source, partial)
    partial.chmod(0o644)
    os.replace(partial, target)
    for side in sidecars:
        dest = target.with_name(side.name)
        if side.exists() and not dest.exists():
            try:
                os.link(side, dest)  # the same volume: no second copy
            except OSError:
                shutil.copyfile(side, dest)
    with adaptor._large.lock:
        adata.file.close()
        adata.file.open(str(target), "r")
    adaptor.h5ad_path = str(target)
    adaptor._large.opened(str(target), True)


class DatasetManager:
    """Manages the lifecycle of dataset adaptors with LRU eviction."""

    def __init__(self, data_dir: str, max_datasets: int = 3) -> None:
        self.data_dir = Path(data_dir)
        results_store.configure(self.data_dir)
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
        if self._evict_lru(keep=dataset_id, room_for_one=True):
            await asyncio.to_thread(memory_budget.release_memory)
        backed = await self._make_memory_room(dataset_id, h5ad_path)

        adaptor = AnnDataAdaptor(h5ad_path, backed=backed)
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
        await self._rebuild_gene_store(dataset_id, adaptor)
        return adaptor

    async def _rebuild_gene_store(self, dataset_id: str, adaptor: AnnDataAdaptor) -> None:
        """A large dataset whose gene store is missing or stale gets it rebuilt in the
        background (gene views read the on-disk rows meanwhile, slowly), once there is
        room for the build (``GENE_STORE_BUILD_BYTES``); never blocks the open."""
        try:
            if not await asyncio.to_thread(adaptor.store_rebuild_needed):
                return
            free = await self.free_memory_for(GENE_STORE_BUILD_BYTES, keep=dataset_id)
            if free is not None and free < GENE_STORE_BUILD_BYTES:
                logger.warning("No room to rebuild the gene store of %s", dataset_id)
                return
            if await asyncio.to_thread(adaptor.start_store_rebuild):
                logger.info("Rebuilding the gene store of %s in the background", dataset_id)
        except Exception as e:  # noqa: BLE001 - the dataset still opens; its note says why
            logger.warning("Gene store rebuild of %s not started: %s", dataset_id, e)

    def _evictable(self, keep: str) -> list[str]:
        """Cached datasets that may be dropped, least recently used first: never
        ``keep`` or one a request is working on (its adaptor would re-read the file
        and lose what the request computed: markers, a pipeline run)."""
        return [d for d in self._load_order
                if d != keep and d not in self._busy and not is_locked(d)]

    def _drop(self, dataset_id: str) -> None:
        self._load_order.remove(dataset_id)
        adaptor = self._datasets.pop(dataset_id, None)
        if adaptor:
            adaptor.close()
            logger.info("Evicted dataset %s from cache", dataset_id)

    def _evict_lru(self, keep: str, room_for_one: bool = False) -> int:
        """Drop least-recently-used datasets beyond capacity (see ``_evictable``);
        how many were dropped."""
        limit = self.max_datasets - 1 if room_for_one else self.max_datasets
        dropped = 0
        for evict_id in self._evictable(keep):
            if len(self._datasets) <= limit:
                break
            self._drop(evict_id)
            dropped += 1
        return dropped

    async def free_memory_for(self, need: int, keep: str) -> int | None:
        """Drop cached datasets other than ``keep``, least recently used first, freeing
        each for real, until ``need`` bytes (with ``LOAD_MARGIN``) and the working
        reserve fit in the memory left under the container limit. Returns the bytes
        free after, or None when there is no container limit (nothing is dropped)."""
        budget = memory_budget.read_budget()
        if not budget.limited or budget.limit_bytes is None or budget.used_bytes is None:
            return None
        target = max(0, need) * LOAD_MARGIN + working_reserve(budget.limit_bytes)

        def free() -> int:
            return budget.limit_bytes - budget.used_bytes

        if free() >= target:
            return free()
        await asyncio.to_thread(memory_budget.release_memory)  # earlier drops may await collection
        budget = memory_budget.read_budget()
        for victim in self._evictable(keep):
            if free() >= target:
                break
            self._drop(victim)
            await asyncio.to_thread(memory_budget.release_memory)
            budget = memory_budget.read_budget()
        return free()

    async def _make_memory_room(self, dataset_id: str, path: str) -> bool:
        """Before opening ``path``: make room for its estimated load peak
        (``preflight``, from the file header) with ``free_memory_for``. Returns whether
        it must open backed although it is under the size threshold (its in-memory
        peak does not fit).

        Three cached datasets fit on a big machine and not under a small limit: on
        a 6 GB backend holding ovary (3.2 GB, dense counts) and pbmc, opening Kang
        got the backend killed. Refused (``DatasetMemoryError``) only when even the
        low end of the estimate's error band is more than what is free with every
        other dataset closed; without a container limit nothing is checked."""
        try:
            plan = await asyncio.to_thread(load_plan, path)
        except Exception as e:  # never block opening on the estimate itself
            logger.warning("No load estimate for %s: %s", path, e)
            return False
        if plan.in_memory is not None:
            # A file under the backed threshold can still be too big to read in (a
            # streamed ingest of a million cells): it opens backed when its in-memory
            # peak does not fit with the working reserve, even with the others closed.
            free = await self.free_memory_for(plan.in_memory, keep=dataset_id)
            budget = memory_budget.read_budget()
            if (free is None or budget.limit_bytes is None
                    or free >= plan.in_memory * LOAD_MARGIN + working_reserve(budget.limit_bytes)):
                await self.reserve(dataset_id, plan.in_memory)
                return False
            logger.info("Opening %s backed: about %.1f GB in memory, %.1f GB free",
                        dataset_id, plan.in_memory / 1024**3, free / 1024**3)
        await self.reserve(dataset_id, plan.backed)
        return plan.in_memory is not None

    async def reserve(self, dataset_id: str, need: int, extra: int = 0,
                      what: str = "Opening this dataset") -> None:
        """Make room for ``need`` bytes (and ``extra`` the work needs beyond them) with
        ``free_memory_for``; ``DatasetMemoryError`` (a 507) when even the low end of
        ``need``'s error band is more than what is free with every other dataset closed.
        ``extra`` is room asked for, never a reason to refuse (a run's pre-step check
        stops a step that cannot fit). Without a container limit nothing is checked."""
        from scview.core import preflight

        def refused(free: int | None) -> bool:
            return free is not None and need * (1 - preflight.ERROR_BAND) > free

        free = await self.free_memory_for(need + extra, keep=dataset_id)
        # A run that just ended may still hold its copy for a moment (the request that
        # follows a stopped ovary run got a 507): give it a few seconds to let go.
        for i in range(REFUSAL_WAIT_SECONDS):
            if not refused(free):
                return
            if i == 0:
                logger.info("%s (%s) needs about %.1f GB; %.1f GB free: waiting up to %d s for memory",
                            what, dataset_id, need / 1024**3, max(0, free or 0) / 1024**3,
                            REFUSAL_WAIT_SECONDS)
            await asyncio.sleep(1)
            free = await self.free_memory_for(need + extra, keep=dataset_id)
        if refused(free):
            raise DatasetMemoryError(dataset_id, need, max(0, free),
                                     memory_budget.read_budget().limit_bytes, what)

    def get_dataset(self, dataset_id: str) -> AnnDataAdaptor | None:
        return self._datasets.get(dataset_id)

    def cached_ids(self) -> list[str]:
        """The datasets held in memory, least recently used first."""
        return list(self._load_order)

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
            else:
                self.prune_sidecars(entry.name)
        return removed

    def prune_sidecars(self, dataset_id: str) -> int:
        """Delete large-dataset sidecars in the dataset's derived folder that no h5ad there
        names (a run that stopped before saving its h5ad, an h5ad since processed in
        memory) and gene-store builds of a process that is gone. Never while a run or an
        edit works on the dataset: a run writes its sidecars before its h5ad."""
        if not is_safe_id(dataset_id) or self.is_busy(dataset_id) or is_locked(dataset_id):
            return 0
        d = self.data_dir / "derived" / dataset_id
        stray = large_mode.stray_sidecars(d)
        if stray:
            logger.info("Removing sidecars no dataset names in %s: %s", d, [p.name for p in stray])
        return large_mode.remove_paths(stray)

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
        in_flight = self._loading.pop(dataset_id, None)  # the next access reads the file again
        adaptor = self._datasets.pop(dataset_id, None)
        if adaptor:
            adaptor.close()
        if dataset_id in self._load_order:
            self._load_order.remove(dataset_id)
        logger.info("Dropped dataset %s (%s)", dataset_id,
                    "a read was in progress" if in_flight is not None and not in_flight.done() else "cached copy")

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

        # A large dataset's sidecars sit beside its h5ad and go with the folder below;
        # named here too, so sidecars beside an h5ad elsewhere cannot be left behind.
        for h5ad in self.h5ad_candidates(dataset_id):
            if large_mode.file_is_large(h5ad):
                large_layout.remove_sidecars(h5ad)
        found = False
        for subdir in ("uploads", "ingested", "converted", "cache", "derived"):
            d = self.data_dir / subdir / dataset_id
            if d.exists():
                shutil.rmtree(d)
                found = True
        results_store.delete_dataset(dataset_id)
        uns_history.forget(dataset_id)
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
        if adaptor.is_large:  # into its h5ad, X untouched
            large_derived(adaptor, self.derived_h5ad_path(dataset_id, adaptor.h5ad_path))
            adaptor.write_in_place(obs=False, uns_keys=keys)
            return True
        if getattr(adata, "isbacked", False):
            logger.info("Dataset %s is backed; not saving %s", dataset_id, keys)
            return False
        target = self.derived_h5ad_path(dataset_id, adaptor.h5ad_path)
        uns_history.before_save(dataset_id, adata)
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
                "protein": adaptor.protein_info(),
                # A large dataset: matrices on disk, and how its gene store and counts are.
                "large": adaptor.large_info(),
                "status": "ready",
            }
        except Exception as e:
            logger.error("Failed to update metadata for %s: %s", dataset_id, e)
            return None
