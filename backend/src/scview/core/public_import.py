"""Import a public dataset from CELLxGENE Discover, pinned to its dataset version.

Discover (https://cellxgene.cziscience.com) publishes curated, annotated datasets;
every dataset version is immutable and has its own id. scView lists them through
Discover's public curation API (``GET /curation/v1/datasets``, documented at
https://api.cellxgene.cziscience.com/curation/ui/), caches the listing in
``DATA_DIR/public_import`` for a day, and downloads one version's h5ad from the
fixed host ``datasets.cellxgene.cziscience.com/{dataset_version_id}.h5ad``. The
URL is built here from a validated UUID: nothing the client sends becomes a URL
(no SSRF), and redirects are not followed.

A Discover h5ad (schema 7) holds normalised values in X, the raw counts in
``raw.X``, Ensembl ids as the gene index with symbols in ``var["feature_name"]``,
and curated obs columns (``cell_type``, ``tissue``, ``disease``, ...). The
downloaded file is kept as sent (``uploads/{id}/``, read-only, its SHA-256
recorded); the dataset's working source (``ingested/{id}/``) is a copy whose gene
index is the symbols (Ensembl ids kept in ``var["gene_ids"]``), with ``cell_type``
first among the obs columns (the author annotation every per-type view picks),
and with the source recorded in provenance and in ``uns["about_*"]`` (read by the
co-pilot, the methods writer and the report). Only those small tables are
rewritten; the matrices are copied byte for byte.

Licence: Discover distributes every dataset under CC BY 4.0 (the submission
policy requires it; the Census on the AWS Open Data registry lists CC-BY 4.0), so
the citation must travel with the data: it is shown in the dataset info and the
methods / report.

Census slices (``cellxgene_census`` / tiledbsoma) were evaluated and not built:
see the batch 18 report.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

SOURCE = "CELLxGENE Discover"
API_BASE = "https://api.cellxgene.cziscience.com/curation/v1"
FILES_BASE = "https://datasets.cellxgene.cziscience.com"
LICENSE = "CC BY 4.0"
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
LISTING_TTL_SECONDS = 24 * 3600
CACHE_DIR = "public_import"
LISTING_FILE = "discover_datasets.json"
CHUNK = 1024 * 1024
# Disk kept free beyond the two copies (the download and the working copy).
DISK_MARGIN = 1024**3
TIMEOUT = httpx.Timeout(60.0, connect=20.0)

_UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


def new_client() -> httpx.Client:
    """Every request to Discover: fixed hosts, redirects not followed."""
    return httpx.Client(timeout=TIMEOUT, follow_redirects=False)


class PublicImportError(Exception):
    """A request that cannot be served; ``status`` is the HTTP status to answer with."""

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


class ImportCancelled(Exception):
    pass


def check_uuid(value: str, what: str = "dataset version id") -> str:
    v = (value or "").strip().lower()
    if not _UUID_RE.fullmatch(v):
        raise PublicImportError(f"Not a valid {what}: expected a UUID.", 422)
    return v


def file_url(version_id: str) -> str:
    """The h5ad of one dataset version, on Discover's fixed file host."""
    return f"{FILES_BASE}/{check_uuid(version_id)}.h5ad"


# ---------------------------------------------------------------------------
# The listing
# ---------------------------------------------------------------------------


def _labels(items: Any) -> list[str]:
    if not isinstance(items, list):
        return []
    return [str(i.get("label")) for i in items if isinstance(i, dict) and i.get("label")]


def slim(rec: dict) -> dict | None:
    """The fields scView shows and records, from one curation API dataset record;
    None for one without an h5ad or not public."""
    if rec.get("tombstone") or rec.get("visibility", "PUBLIC") != "PUBLIC":
        return None
    h5ad = [a for a in rec.get("assets") or [] if isinstance(a, dict) and a.get("filetype") == "H5AD"]
    vid = str(rec.get("dataset_version_id") or "").lower()
    if not h5ad or not _UUID_RE.fullmatch(vid):
        return None
    return {
        "dataset_id": rec.get("dataset_id"),
        "dataset_version_id": vid,
        "collection_id": rec.get("collection_id"),
        "collection_version_id": rec.get("collection_version_id"),
        "collection_name": rec.get("collection_name"),
        "collection_doi": rec.get("collection_doi"),
        "collection_doi_label": rec.get("collection_doi_label"),
        "title": rec.get("title") or "",
        "citation": rec.get("citation") or "",
        "organism": _labels(rec.get("organism")),
        "tissue": _labels(rec.get("tissue")),
        "disease": _labels(rec.get("disease")),
        "assay": _labels(rec.get("assay")),
        "n_cell_types": len(_labels(rec.get("cell_type"))),
        "suspension_type": [str(s) for s in rec.get("suspension_type") or []],
        "cell_count": int(rec.get("cell_count") or 0),
        "size_bytes": int(h5ad[0].get("filesize") or 0),
        "schema_version": rec.get("schema_version"),
        "published_at": rec.get("published_at"),
        "revised_at": rec.get("revised_at"),
        "explorer_url": rec.get("explorer_url"),
        "license": LICENSE,
    }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Listing:
    datasets: list[dict]
    fetched_at: str
    from_cache: bool


class DiscoverCatalog:
    """Discover's dataset listing, cached in ``DATA_DIR/public_import``."""

    def __init__(self, data_dir: str | Path, client_factory: Callable[[], httpx.Client] | None = None,
                 ttl: float = LISTING_TTL_SECONDS) -> None:
        self.dir = Path(data_dir) / CACHE_DIR
        self.ttl = ttl
        self._client_factory = client_factory
        self._lock = threading.Lock()
        self._mem: Listing | None = None

    def _client(self) -> httpx.Client:
        return (self._client_factory or new_client)()

    def _get_json(self, path: str) -> Any:
        with self._client() as c:
            try:
                r = c.get(f"{API_BASE}{path}")
            except httpx.HTTPError as e:
                raise PublicImportError(f"CELLxGENE Discover could not be reached ({e}).", 502)
        if r.status_code == 404:
            raise PublicImportError("CELLxGENE Discover does not know that dataset.", 404)
        if r.status_code != 200:
            raise PublicImportError(f"CELLxGENE Discover answered {r.status_code}.", 502)
        try:
            return r.json()
        except ValueError:
            raise PublicImportError("CELLxGENE Discover sent a reply that is not JSON.", 502)

    def listing(self, refresh: bool = False) -> Listing:
        """The slimmed listing: from memory or the cache file while under a day old,
        else fetched again (a stale cache is used when Discover cannot be reached)."""
        with self._lock:
            path = self.dir / LISTING_FILE
            cached: Listing | None = self._mem
            if cached is None and path.is_file():
                try:
                    data = json.loads(path.read_text())
                    cached = Listing(list(data["datasets"]), str(data["fetched_at"]), True)
                except (OSError, ValueError, KeyError, TypeError):
                    cached = None
            if cached is not None and not refresh and self._fresh(cached.fetched_at):
                self._mem = cached
                return Listing(cached.datasets, cached.fetched_at, True)
            try:
                raw = self._get_json("/datasets")
            except PublicImportError:
                if cached is not None:
                    logger.warning("Discover unreachable: using the listing from %s", cached.fetched_at)
                    self._mem = cached
                    return Listing(cached.datasets, cached.fetched_at, True)
                raise
            if not isinstance(raw, list):
                raise PublicImportError("CELLxGENE Discover's listing has an unexpected shape.", 502)
            datasets = [s for s in (slim(r) for r in raw if isinstance(r, dict)) if s]
            fresh = Listing(datasets, _now(), False)
            self.dir.mkdir(parents=True, exist_ok=True)
            part = path.with_suffix(".part")
            part.write_text(json.dumps({"fetched_at": fresh.fetched_at, "datasets": datasets}))
            part.replace(path)
            self._mem = fresh
            return fresh

    def _fresh(self, fetched_at: str) -> bool:
        try:
            t = datetime.fromisoformat(fetched_at).timestamp()
        except ValueError:
            return False
        return time.time() - t < self.ttl

    def find(self, version_id: str, dataset_id: str | None = None) -> dict:
        """One dataset version: from the listing (current versions), else, given its
        dataset id, from that dataset's version history (an older, pinned version)."""
        vid = check_uuid(version_id)
        for d in self.listing().datasets:
            if d["dataset_version_id"] == vid:
                return d
        if dataset_id:
            did = check_uuid(dataset_id, "dataset id")
            versions = self._get_json(f"/datasets/{did}/versions")
            for rec in versions if isinstance(versions, list) else []:
                if isinstance(rec, dict) and str(rec.get("dataset_version_id", "")).lower() == vid:
                    s = slim({**rec, "visibility": "PUBLIC"})
                    if s:
                        return s
        raise PublicImportError("That dataset version is not in CELLxGENE Discover's listing.", 404)


def _matches(values: list[str], wanted: str | None) -> bool:
    if not wanted:
        return True
    w = wanted.strip().lower()
    return any(v.lower() == w for v in values)


def search(datasets: list[dict], q: str | None = None, organism: str | None = None,
           tissue: str | None = None, disease: str | None = None, assay: str | None = None,
           max_bytes: int | None = None, sort: str = "cells", limit: int = 50,
           offset: int = 0) -> dict:
    """Datasets matching every given filter (``q``: all words, in the title, collection,
    tissues, diseases, assays or DOI label), with counts per organism and tissue over
    the text-matched set so the filters show what is there."""
    words = [w for w in (q or "").lower().split() if w]

    def text(d: dict) -> str:
        return " ".join([d.get("title") or "", d.get("collection_name") or "",
                         d.get("collection_doi_label") or "", *d["tissue"], *d["disease"],
                         *d["assay"], *d["organism"]]).lower()

    hits = [d for d in datasets if all(w in text(d) for w in words)]
    organisms = Counter(o for d in hits for o in d["organism"])
    tissues = Counter(t for d in hits if _matches(d["organism"], organism) for t in d["tissue"])
    hits = [d for d in hits
            if _matches(d["organism"], organism) and _matches(d["tissue"], tissue)
            and _matches(d["disease"], disease) and _matches(d["assay"], assay)
            and (max_bytes is None or d["size_bytes"] <= max_bytes)]
    key = {"cells": lambda d: -d["cell_count"], "size": lambda d: d["size_bytes"],
           "title": lambda d: d["title"].lower(),
           "newest": lambda d: "".join(chr(0x10FFFF - ord(c)) for c in (d.get("published_at") or ""))}
    hits.sort(key=key.get(sort, key["cells"]))
    return {
        "total": len(hits),
        "results": hits[offset:offset + limit],
        "facets": {"organism": organisms.most_common(), "tissue": tissues.most_common(60)},
    }


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


def check_disk(where: Path, size_bytes: int) -> None:
    """Room for the download and the working copy beside it, with a margin."""
    where.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(where).free
    need = 2 * size_bytes + DISK_MARGIN
    if free < need:
        gb = 1024**3
        raise PublicImportError(
            f"Not enough disk: the import needs about {need / gb:.1f} GB (the file twice: as "
            f"downloaded and scView's working copy) and {free / gb:.1f} GB is free.", 507)


def download(version_id: str, dest: Path, expected_size: int, cancel: threading.Event,
             client: httpx.Client | None = None) -> Iterator[tuple[int, int]]:
    """Stream the version's h5ad to ``dest`` (through a ``.part`` file), yielding
    (bytes so far, total). Raises ``ImportCancelled`` when ``cancel`` is set and
    ``PublicImportError`` on an HTTP error or a size mismatch; the partial file is
    removed either way."""
    url = file_url(version_id)
    part = dest.with_name(f".{dest.name}.part")
    own = client is None
    client = client or new_client()
    try:
        with client.stream("GET", url) as r:
            if r.status_code != 200:
                raise PublicImportError(f"The download from CELLxGENE Discover failed (HTTP "
                                        f"{r.status_code}).", 502)
            total = int(r.headers.get("content-length") or expected_size or 0)
            done = 0
            with open(part, "wb") as fh:
                for chunk in r.iter_bytes(CHUNK):
                    if cancel.is_set():
                        raise ImportCancelled()
                    fh.write(chunk)
                    done += len(chunk)
                    yield done, total
        if expected_size and done != expected_size:
            raise PublicImportError(f"The download is incomplete: {done:,} of {expected_size:,} "
                                    "bytes.", 502)
        part.replace(dest)
    except httpx.HTTPError as e:
        part.unlink(missing_ok=True)
        raise PublicImportError(f"The download from CELLxGENE Discover failed ({e}).", 502)
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    finally:
        if own:
            client.close()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(8 * CHUNK):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# The working copy
# ---------------------------------------------------------------------------

# Obs columns Discover curates, in the order they are put first (cell_type leads:
# the author annotation every per-type view takes, before its ontology-id column).
LEADING_OBS = ("cell_type", "tissue", "disease", "donor_id", "assay", "sex")


def unique_symbols(symbols: list[str], ids: list[str]) -> list[str]:
    """Symbols as the gene index: an empty one becomes its Ensembl id; repeats get
    ``-1``, ``-2`` ... (``anndata.utils.make_index_unique``, as the replay does)."""
    import anndata as ad
    import pandas as pd

    names = [s if s and s.lower() != "nan" else i for s, i in zip(symbols, ids)]
    return [str(n) for n in ad.utils.make_index_unique(pd.Index(names))]


def _symbol_var(group) -> dict | None:
    """Rewrite one ``var`` table (``/var`` or ``/raw/var``) to symbols in place."""
    from anndata.io import read_elem, write_elem

    var = read_elem(group)
    if "feature_name" not in var.columns or "gene_ids" in var.columns:
        return None
    ids = [str(v) for v in var.index]
    var["gene_ids"] = ids
    var.index = unique_symbols([str(v) for v in var["feature_name"].astype(str)], ids)
    parent, name = group.parent, group.name.rsplit("/", 1)[-1]
    del parent[name]
    write_elem(parent, name, var)
    return {"n_genes": len(ids), "renamed": int(sum(a != b for a, b in zip(ids, var.index)))}


def about(rec: dict, sha256: str, size: int, downloaded_at: str) -> dict[str, str]:
    """``uns["about_*"]`` and the dataset info: where the data came from, to cite it."""
    doi = rec.get("collection_doi") or ""
    return {
        "about_title": str(rec.get("title") or ""),
        "about_source": SOURCE,
        "about_citation": str(rec.get("citation") or ""),
        "about_publication": str(rec.get("collection_doi_label") or ""),
        "about_doi": str(doi),
        "about_collection": str(rec.get("collection_name") or ""),
        "about_collection_id": str(rec.get("collection_id") or ""),
        "about_dataset_id": str(rec.get("dataset_id") or ""),
        "about_dataset_version_id": str(rec.get("dataset_version_id") or ""),
        "about_schema_version": str(rec.get("schema_version") or ""),
        "about_license": LICENSE,
        "about_url": file_url(rec["dataset_version_id"]),
        "about_downloaded_at": downloaded_at,
        "about_sha256": sha256,
        "about_size_bytes": str(size),
    }


def citation_text(info: dict) -> str:
    """One line to cite: the publication, then Discover's own citation, licence."""
    parts = []
    if info.get("about_publication") or info.get("about_doi"):
        pub = info.get("about_publication") or ""
        doi = info.get("about_doi")
        parts.append(f"{pub} doi:{doi}".strip() if doi else pub)
    if info.get("about_citation"):
        parts.append(info["about_citation"])
    parts.append(f"Licence {info.get('about_license') or LICENSE}.")
    return " ".join(p if p.endswith(".") else p + "." for p in parts if p)


def prepare(src: Path, out: Path, rec: dict, info: dict[str, str]) -> dict:
    """Write the working copy ``out`` of the downloaded ``src``: symbols as the gene
    index (X and raw), ``cell_type`` first in obs, provenance and ``about_*`` in uns."""
    import anndata as ad
    import h5py
    import pandas as pd
    from anndata.io import read_elem, write_elem

    from scview.core import provenance

    part = out.with_name(f".{out.name}.part")
    shutil.copyfile(src, part)
    part.chmod(0o644)
    summary: dict[str, Any] = {}
    try:
        with h5py.File(part, "r+") as f:
            if "var" in f:
                summary["var"] = _symbol_var(f["var"])
            if "raw" in f and "var" in f["raw"]:
                summary["raw_var"] = _symbol_var(f["raw"]["var"])
            obs = f["obs"]
            order = [c.decode() if isinstance(c, bytes) else str(c)
                     for c in obs.attrs.get("column-order", [])]
            lead = [c for c in LEADING_OBS if c in order]
            if lead:
                obs.attrs["column-order"] = lead + [c for c in order if c not in lead]
            summary["author_annotation"] = "cell_type" if "cell_type" in order else None
            n_obs = len(read_elem(obs[obs.attrs["_index"]])) if obs.attrs.get("_index") in obs else 0
            n_vars = int(f["X"].attrs["shape"][1]) if "shape" in f["X"].attrs else int(f["X"].shape[1])
            holder = ad.AnnData(obs=pd.DataFrame(index=[str(i) for i in range(n_obs)]),
                                var=pd.DataFrame(index=[str(i) for i in range(n_vars)]))
            uns = f["uns"] if "uns" in f else f.create_group("uns")
            if provenance.UNS_KEY in uns:
                holder.uns[provenance.UNS_KEY] = read_elem(uns[provenance.UNS_KEY])
            provenance.init_source(holder, origin=SOURCE.lower().replace(" ", "_"),
                                   original_filename=src.name, fmt="h5ad",
                                   when=info["about_downloaded_at"], overwrite=True)
            params = {"public_import": {
                "source": SOURCE,
                "dataset_id": rec.get("dataset_id"),
                "dataset_version_id": rec.get("dataset_version_id"),
                "collection_id": rec.get("collection_id"),
                "collection_version_id": rec.get("collection_version_id"),
                "doi": rec.get("collection_doi"),
                "schema_version": rec.get("schema_version"),
                "license": LICENSE,
                "url": info["about_url"],
                "sha256": info["about_sha256"],
                "size_bytes": int(info["about_size_bytes"]),
                "var_names": "feature_name (Ensembl ids in var['gene_ids'])",
                "author_annotation": summary["author_annotation"],
            }}
            provenance.record_step(holder, step="ingest", tool="scview.public_import", params=params)
            holder.uns.update(info)
            for key, value in holder.uns.items():
                if key in uns:
                    del uns[key]
                write_elem(uns, key, value)
        part.replace(out)
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    return summary


# ---------------------------------------------------------------------------
# Memory, and the import as a whole
# ---------------------------------------------------------------------------


def memory_verdict(path: Path) -> dict:
    """How the downloaded file would open (``dataset_manager.load_plan``, from its
    header): ``fits`` in memory, ``backed`` (matrices stay on disk: large-dataset
    handling), or ``wont_fit`` even backed with every other dataset closed."""
    from scview.core import memory_budget
    from scview.core.dataset_manager import LOAD_MARGIN, load_plan, working_reserve

    plan = load_plan(str(path))
    budget = memory_budget.read_budget()
    gb = 1024**3
    out: dict[str, Any] = {"in_memory_bytes": plan.in_memory, "backed_bytes": plan.backed,
                           "limit_bytes": budget.limit_bytes}
    if not budget.limited or budget.limit_bytes is None:
        out.update(verdict="fits", message="No memory limit is set for scView.")
        return out
    room = budget.limit_bytes - working_reserve(budget.limit_bytes)
    if plan.in_memory is not None and plan.in_memory * LOAD_MARGIN <= room:
        out.update(verdict="fits", message=f"Opens in memory: about {plan.in_memory / gb:.1f} GB at "
                                           f"its peak of scView's {budget.limit_bytes / gb:.1f} GB.")
    elif plan.backed * LOAD_MARGIN <= room:
        out.update(verdict="backed", message=(
            "Too big to read into memory: it opens from disk (the expression matrix stays in the "
            f"file), about {plan.backed / gb:.1f} GB at its peak; processing runs in "
            "large-dataset mode."))
    else:
        out.update(verdict="wont_fit", message=(
            f"Needs about {plan.backed / gb:.1f} GB even with its matrices on disk; scView has "
            f"{budget.limit_bytes / gb:.1f} GB. Raise SCVIEW_MEMORY_GB in .env (then restart scView)."))
    return out


def dataset_name(rec: dict) -> str:
    from scview.core.dataset_manager import DATASET_NAME_MAX

    name = " ".join(str(rec.get("title") or "CELLxGENE dataset").split())
    return name[:DATASET_NAME_MAX]


def run_import(rec: dict, data_dir: Path, dataset_id: str, cancel: threading.Event,
               name: str | None = None, client: httpx.Client | None = None,
               verdict: Callable[[Path], dict] | None = None) -> Iterator[tuple[str, dict]]:
    """Download, check and register one dataset version, yielding SSE (event, data):
    ``stage``, ``progress`` (four a second at most), ``memory``, then ``registered``
    with the metadata. Everything written is removed when it fails or is cancelled."""
    vid = rec["dataset_version_id"]
    size = int(rec.get("size_bytes") or 0)
    upload_dir = data_dir / "uploads" / dataset_id
    ingest_dir = data_dir / "ingested" / dataset_id
    ok = False
    try:
        check_disk(data_dir, size)
        upload_dir.mkdir(parents=True, exist_ok=True)
        src = upload_dir / f"{vid}.h5ad"
        yield "stage", {"stage": "download", "message": f"Downloading {file_url(vid)}",
                        "total": size}
        last = 0.0
        for done, total in download(vid, src, size, cancel, client=client):
            now = time.monotonic()
            if now - last >= 0.25 or done == total:
                last = now
                yield "progress", {"done": done, "total": total}
        if cancel.is_set():
            raise ImportCancelled()
        yield "stage", {"stage": "verify", "message": "Checking the file and the memory it needs"}
        sha = file_sha256(src)
        mem = (verdict or memory_verdict)(src)
        yield "memory", mem
        if mem.get("verdict") == "wont_fit":
            raise PublicImportError(mem["message"], 507)
        if cancel.is_set():
            raise ImportCancelled()
        yield "stage", {"stage": "prepare", "message": "Writing scView's working copy (gene symbols, "
                                                       "source and citation)"}
        info = about(rec, sha, src.stat().st_size, _now())
        ingest_dir.mkdir(parents=True, exist_ok=True)
        out = ingest_dir / f"{vid}.h5ad"
        summary = prepare(src, out, rec, info)
        for p in (src, out):
            try:
                p.chmod(0o444)
            except OSError:
                pass
        meta = {
            "id": dataset_id,
            "name": (name or "").strip()[:120] or dataset_name(rec),
            "filename": src.name,
            "extension": ".h5ad",
            "size_bytes": src.stat().st_size,
            "status": "ready",
            "created_at": _now(),
            "n_cells": None,
            "n_genes": None,
            "available_embeddings": [],
            "obs_columns": [],
            "public_source": {**info, "citation_text": citation_text(info),
                              "author_annotation": summary.get("author_annotation")},
        }
        (upload_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
        ok = True
        yield "registered", meta
    finally:
        if not ok:
            shutil.rmtree(upload_dir, ignore_errors=True)
            shutil.rmtree(ingest_dir, ignore_errors=True)
