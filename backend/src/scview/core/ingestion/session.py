"""Ingestion engine — stage [5] session orchestration.

An ingest *session* is a staging directory (``data/ingest/{session_id}/files/``)
plus a small ``session.json`` of user options. Files are uploaded into it
incrementally; the session reports detection/bundle/validation state at any
point, and on commit runs the full pipeline (build → validate → load → merge),
writes a canonical gzip h5ad into the immutable ``ingested/{dataset_id}/`` layer,
copies the originals (read-only) into ``uploads/{dataset_id}/``, writes the
``metadata.json`` sidecar, and registers the result through the DatasetManager
so it appears in the picker like any other dataset.

Abandoned sessions are removed by :meth:`sweep_expired` (24 h TTL). See
``docs/INGESTION_ENGINE.md`` §5.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from scview.core.ingestion.bundling import UnitFormat, build_bundle
from scview.core.ingestion.loading import load_unit
from scview.core.ingestion.merge import MergeJoin, MergePlan, merge_units, plan_merge
from scview.core.ingestion.validation import IssueSeverity, validate_bundle

if TYPE_CHECKING:
    from scview.core.dataset_manager import DatasetManager

logger = logging.getLogger(__name__)

_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


class IngestCommitError(RuntimeError):
    """Raised when a session cannot be committed into a dataset."""


class IngestOptions(BaseModel):
    """User-resolved choices for a session (persisted in session.json)."""

    name: str | None = None
    join: MergeJoin = MergeJoin.inner
    sample_label: str = "sample"
    apply_reconciliation: bool = True
    genes_in_rows: bool = True


class IngestSessionManager:
    """Manage ingest staging sessions and commit them into datasets."""

    def __init__(self, data_dir: str) -> None:
        self.data_dir = Path(data_dir)
        self.ingest_dir = self.data_dir / "ingest"

    # -- session lifecycle ------------------------------------------------

    def create_session(self) -> str:
        sid = uuid.uuid4().hex
        self.files_dir(sid).mkdir(parents=True, exist_ok=True)
        self._write_session(sid, {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "options": IngestOptions().model_dump(mode="json"),
        })
        return sid

    def exists(self, sid: str) -> bool:
        return self._session_dir(sid).is_dir()

    def discard(self, sid: str) -> bool:
        d = self._session_dir(sid)
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
            return True
        return False

    def files_dir(self, sid: str) -> Path:
        return self._session_dir(sid) / "files"

    def staged_paths(self, sid: str) -> list[Path]:
        fd = self.files_dir(sid)
        if not fd.is_dir():
            return []
        return sorted(p for p in fd.iterdir() if p.is_file() or p.is_dir())

    # -- options ----------------------------------------------------------

    def get_options(self, sid: str) -> IngestOptions:
        data = self._read_session(sid)
        return IngestOptions.model_validate(data.get("options", {}))

    def set_options(self, sid: str, options: IngestOptions) -> None:
        data = self._read_session(sid)
        data["options"] = options.model_dump(mode="json")
        self._write_session(sid, data)

    # -- inspection -------------------------------------------------------

    def get_state(self, sid: str) -> dict[str, Any]:
        """Cheap state: detection + bundle + validation (no full data read)."""
        bundle = build_bundle(self.staged_paths(sid))
        report = validate_bundle(bundle)
        return {
            "session_id": sid,
            "bundle": bundle.model_dump(mode="json"),
            "validation": report.model_dump(mode="json"),
            "options": self.get_options(sid).model_dump(mode="json"),
        }

    def merge_plan(self, sid: str) -> MergePlan | None:
        """Load the units and analyse a multi-sample merge (heavier — reads data)."""
        bundle = build_bundle(self.staged_paths(sid))
        loadable = [u for u in bundle.units if u.format != UnitFormat.unknown and u.complete]
        if len(loadable) < 2:
            return None
        opts = self.get_options(sid)
        adatas = {u.label: load_unit(u, {"genes_in_rows": opts.genes_in_rows}) for u in loadable}
        return plan_merge(adatas)

    # -- commit -----------------------------------------------------------

    async def commit(self, sid: str, dm: DatasetManager) -> str:
        if not self.exists(sid):
            raise IngestCommitError("Ingest session not found.")
        files = self.staged_paths(sid)
        if not files:
            raise IngestCommitError("No files have been uploaded to this session.")

        bundle = build_bundle(files)
        report = validate_bundle(bundle)
        if not report.ok:
            errs = [i.message for i in report.issues if i.severity == IssueSeverity.error]
            raise IngestCommitError(errs[0] if errs else "The uploaded files didn't validate.")

        loadable = [u for u in bundle.units if u.format != UnitFormat.unknown]
        if not loadable:
            raise IngestCommitError("No recognisable single-cell dataset was found.")

        opts = self.get_options(sid)
        adata = self._load_and_merge(loadable, opts)

        dataset_id = uuid.uuid4().hex
        name = opts.name or (loadable[0].label if len(loadable) == 1 else "merged_dataset")
        self._persist_dataset(dataset_id, name, adata, files)
        await dm.load_dataset(dataset_id)  # populates n_cells / embeddings / obs_columns

        self.discard(sid)
        logger.info("Committed ingest session %s as dataset %s (%s)", sid, dataset_id, name)
        return dataset_id

    # -- TTL --------------------------------------------------------------

    def sweep_expired(self, ttl_hours: int = 24) -> int:
        """Remove staging sessions older than the TTL. Returns count removed."""
        if not self.ingest_dir.is_dir():
            return 0
        cutoff = datetime.now(timezone.utc) - timedelta(hours=ttl_hours)
        removed = 0
        for d in self.ingest_dir.iterdir():
            if not d.is_dir():
                continue
            created = self._created_at(d.name)
            if created is None or created < cutoff:
                shutil.rmtree(d, ignore_errors=True)
                removed += 1
        if removed:
            logger.info("Swept %d expired ingest session(s)", removed)
        return removed

    # -- internals --------------------------------------------------------

    def _load_and_merge(self, loadable, opts: IngestOptions):
        load_opts = {"genes_in_rows": opts.genes_in_rows}
        adatas = {u.label: load_unit(u, load_opts) for u in loadable}
        if len(adatas) == 1:
            return next(iter(adatas.values()))
        plan = plan_merge(adatas)
        recon = plan.reconciliation if opts.apply_reconciliation else None
        return merge_units(
            adatas, join=opts.join, sample_label=opts.sample_label, reconciliation=recon
        )

    def _persist_dataset(self, dataset_id: str, name: str, adata, originals: list[Path]) -> None:
        # Originals → uploads/{id}/ (read-only), the untouched source.
        up = self.data_dir / "uploads" / dataset_id
        up.mkdir(parents=True, exist_ok=True)
        total = 0
        for src in originals:
            if src.is_dir():
                continue  # store directories (e.g. zarr) is a later refinement
            dest = up / src.name
            shutil.copyfile(src, dest)
            total += dest.stat().st_size
            _make_readonly(dest)

        # Canonical h5ad → ingested/{id}/ (read-only, gzip-compressed).
        ing = self.data_dir / "ingested" / dataset_id
        ing.mkdir(parents=True, exist_ok=True)
        h5_path = ing / f"{_safe_name(name)}.h5ad"
        adata.write_h5ad(h5_path, compression="gzip")
        _make_readonly(h5_path)

        # metadata sidecar (uploads/) — what list_datasets reads.
        meta = {
            "id": dataset_id,
            "name": name,
            "filename": h5_path.name,
            "extension": ".h5ad",
            "size_bytes": total,
            "status": "ready",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "n_cells": None,
            "n_genes": None,
            "available_embeddings": [],
            "obs_columns": [],
            "source": "ingested",
        }
        with open(up / "metadata.json", "w") as fh:
            json.dump(meta, fh, indent=2)

    def _session_dir(self, sid: str) -> Path:
        return self.ingest_dir / sid

    def _session_file(self, sid: str) -> Path:
        return self._session_dir(sid) / "session.json"

    def _read_session(self, sid: str) -> dict[str, Any]:
        p = self._session_file(sid)
        if not p.exists():
            return {}
        with open(p) as fh:
            return json.load(fh)

    def _write_session(self, sid: str, data: dict[str, Any]) -> None:
        with open(self._session_file(sid), "w") as fh:
            json.dump(data, fh, indent=2)

    def _created_at(self, sid: str) -> datetime | None:
        raw = self._read_session(sid).get("created_at")
        try:
            return datetime.fromisoformat(raw) if raw else None
        except (TypeError, ValueError):
            return None


def _safe_name(name: str) -> str:
    cleaned = _SAFE_NAME_RE.sub("_", name).strip("_")
    return cleaned or "dataset"


def _make_readonly(path: Path) -> None:
    try:
        path.chmod(0o444)
    except OSError as e:
        logger.warning("Could not mark %s read-only: %s", path, e)
