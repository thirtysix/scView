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

import asyncio
import json
import logging
import os
import re
import shutil
import uuid
import zlib
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel

from scview.core import crispr, protein, provenance
from scview.core.dataset_manager import is_safe_id
from scview.core.ingestion import streamed
from scview.core.ingestion.bundling import IngestUnit, SideKind, UnitFormat, build_bundle
from scview.core.ingestion.detection import is_store_dir
from scview.core.ingestion.loading import IngestLoadError, load_unit
from scview.core.ingestion.merge import MergeJoin, MergePlan, merge_units, plan_merge
from scview.core.ingestion.validation import IssueSeverity, validate_bundle

if TYPE_CHECKING:
    from scview.core.dataset_manager import DatasetManager

logger = logging.getLogger(__name__)

_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


class IngestCommitError(RuntimeError):
    """Raised when a session cannot be committed into a dataset."""


class StagingConflict(RuntimeError):
    """An upload would put a file where another one is staged; nothing was staged."""


# Uploads land under this name first and move into place once the whole request is in.
PARTIAL_PREFIX = ".scview-partial-"


class IngestOptions(BaseModel):
    """User-resolved choices for a session (persisted in session.json)."""

    name: str | None = None
    join: MergeJoin = MergeJoin.inner
    sample_label: str = "sample"
    apply_reconciliation: bool = True
    genes_in_rows: bool = True
    # Read and save the matrix a block of cells at a time (``streamed.py``): "auto" when
    # reading it whole would not fit the memory free; "on" / "off" force it.
    streamed: Literal["auto", "on", "off"] = "auto"


def _seqqc_view(unit: IngestUnit) -> IngestUnit:
    """The unit with only its sequencing-QC side files (metrics, raw matrix): its
    contig tables, guide calls and spatial files are joined to the cells by
    ``_attach_vdj`` / ``_attach_guide_calls`` / ``_attach_spatial`` instead."""
    keep = [f for f in unit.side_files
            if f.side_kind not in (SideKind.vdj, SideKind.crispr, SideKind.spatial)]
    return unit.model_copy(update={"side_files": keep})


def _protein_params(adata) -> dict:
    """The CITE-seq panel split off at load, for the ingest history entry."""
    names = protein.protein_names(adata)
    if not names:
        return {}
    return {"protein": {
        "n_proteins": len(names),
        "controls": [n for n in names if protein.is_control(n)],
        "normalisation": "CLR per cell (Seurat margin = 2)",
        "location": "obsm protein_counts / protein_clr",
    }}


class StagedPaths(list):
    """A session's staged files, with the staging folder they sit in (``root``), so
    ``build_bundle`` keeps their folder paths for any caller that passes the list."""

    def __init__(self, items, root: Path) -> None:
        super().__init__(items)
        self.root = root


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
        return is_safe_id(sid) and self._session_dir(sid).is_dir()

    def discard(self, sid: str) -> bool:
        if not is_safe_id(sid):
            return False
        d = self._session_dir(sid)
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
            return True
        return False

    def files_dir(self, sid: str) -> Path:
        return self._session_dir(sid) / "files"

    def staged_paths(self, sid: str) -> list[Path]:
        """Every staged file, in subfolders too (a folder upload keeps its layout); a
        Zarr store is one entry, never walked into."""
        fd = self.files_dir(sid)
        if not fd.is_dir():
            return StagedPaths([], fd)
        out: list[Path] = []

        def walk(d: Path) -> None:
            for p in sorted(d.iterdir()):
                if p.name.startswith(PARTIAL_PREFIX) or p.is_symlink():
                    continue
                if p.is_dir():
                    if is_store_dir(p):
                        out.append(p)
                    else:
                        walk(p)
                elif p.is_file():
                    out.append(p)

        walk(fd)
        return StagedPaths(out, fd)

    def partial_path(self, sid: str) -> Path:
        """Where one upload is written before ``stage`` moves it into place."""
        return self.files_dir(sid) / f"{PARTIAL_PREFIX}{uuid.uuid4().hex}"

    def stage(self, sid: str, items: list[tuple[str, Path, str]]) -> list[str]:
        """Move uploaded files (``(relative path, partial file, sha256)``) to their
        paths under the session, all or none. The same file sent again (same path,
        same content) is kept once; a different file at a path that is taken, a file
        where a staged folder is (or the reverse), raises StagingConflict and stages
        nothing. Returns the paths newly staged."""
        root = self.files_dir(sid)
        rels = {rel for rel, _, _ in items}
        if len(rels) < len(items):
            seen: set[str] = set()
            dup = next(rel for rel, _, _ in items if rel in seen or seen.add(rel))
            raise StagingConflict(f"Two files of this upload have the same path, {dup}.")
        moves: list[tuple[Path, Path | None]] = []
        for rel, partial, sha in items:
            dest = root / rel
            if not dest.resolve().is_relative_to(root.resolve()):  # safe_relative_path already refused it
                raise StagingConflict(f"{rel} would land outside the upload folder.")
            for parent in PurePosixPath(rel).parents:
                if str(parent) == ".":
                    break
                if (root / parent).is_file() or str(parent) in rels:
                    raise StagingConflict(f"{parent} is a file, so {rel} cannot be put in a folder of "
                                          "that name.")
            if dest.is_dir():
                raise StagingConflict(f"{rel} is already a folder of this upload.")
            if dest.is_file():
                from scview.core.run_manifest import file_sha256

                if file_sha256(dest) != sha:
                    raise StagingConflict(
                        f"A different file is already staged as {rel}. Start over to replace it, "
                        "or upload the second one from a folder with another name.")
                moves.append((partial, None))  # the same file again
                continue
            moves.append((partial, dest))
        staged = []
        for partial, dest in moves:
            if dest is None:
                partial.unlink(missing_ok=True)
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            partial.replace(dest)
            staged.append(dest.relative_to(root).as_posix())
        return staged

    def record_skipped(self, sid: str, skipped: list[dict]) -> None:
        """Remember files an upload left out (``{"path", "reason", "bytes"}``) for the
        state's sentence about what was skipped."""
        if not skipped:
            return
        data = self._read_session(sid)
        known = {s.get("path"): s for s in data.get("skipped", [])}
        known.update({s["path"]: s for s in skipped})
        data["skipped"] = list(known.values())
        self._write_session(sid, data)

    def _bundle(self, sid: str):
        return build_bundle(self.staged_paths(sid), root=self.files_dir(sid),
                            skipped=self._read_session(sid).get("skipped", []))

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
        bundle = self._bundle(sid)
        report = validate_bundle(bundle)
        return {
            "session_id": sid,
            "bundle": bundle.model_dump(mode="json"),
            "validation": report.model_dump(mode="json"),
            "options": self.get_options(sid).model_dump(mode="json"),
        }

    def merge_plan(self, sid: str) -> MergePlan | None:
        """Load the units and analyse a multi-sample merge (heavier — reads data)."""
        bundle = self._bundle(sid)
        loadable = [
            u for u in bundle.units
            if u.format not in (UnitFormat.unknown, UnitFormat.seurat) and u.complete
        ]
        if len(loadable) < 2:
            return None
        opts = self.get_options(sid)
        try:
            if all(u.format in streamed.STREAMABLE for u in loadable):
                # The cell and gene tables only: a plan never needs the matrices.
                try:
                    units = [streamed.open_unit(u, self._session_dir(sid)) for u in loadable]
                    return plan_merge({su.label: su.meta for su in units}, streamed.counts_like(units))
                except streamed.StreamUnsupported as e:
                    logger.info("Merge plan from the samples read whole (%s)", e)
            adatas = {
                u.label: load_unit(u, {"genes_in_rows": opts.genes_in_rows}) for u in loadable
            }
        except IngestLoadError as e:
            # No plan for unreadable data; commit reports the error to the user.
            logger.warning("Merge plan skipped for session %s: %s", sid, e)
            return None
        return plan_merge(adatas)

    # -- commit -----------------------------------------------------------

    async def commit(self, sid: str, dm: DatasetManager) -> str:
        if not self.exists(sid):
            raise IngestCommitError("Ingest session not found.")
        files = self.staged_paths(sid)
        if not files:
            raise IngestCommitError("No files have been uploaded to this session.")

        # Reading, merging and writing the files takes seconds to minutes: in a
        # worker thread, so the server keeps answering meanwhile.
        dataset_id = await asyncio.to_thread(self._build_dataset, sid)
        self.discard(sid)  # saved: the staged files are no longer needed
        from scview.core.dataset_manager import DatasetMemoryError

        try:
            await dm.load_dataset(dataset_id)  # populates n_cells / embeddings / obs_columns
        except DatasetMemoryError as e:
            # Saved, but too large to open in the memory free now; opening it says so again.
            logger.warning("Ingested dataset %s not opened: %s", dataset_id, e)
        return dataset_id

    def _build_dataset(self, sid: str) -> str:
        """Validate, load, merge and save the staged files as a new dataset."""
        bundle = self._bundle(sid)
        report = validate_bundle(bundle)
        if not report.ok:
            errs = [i.message for i in report.issues if i.severity == IssueSeverity.error]
            raise IngestCommitError(errs[0] if errs else "The uploaded files didn't validate.")

        loadable = [u for u in bundle.units if u.format != UnitFormat.unknown]
        if not loadable:
            raise IngestCommitError("No recognisable single-cell dataset was found.")

        opts = self.get_options(sid)
        part = self._session_dir(sid) / ".scview-matrices.part.h5"
        try:
            adata, matrices = self._load_or_stream(loadable, opts, self._session_dir(sid), part)
        except IngestLoadError as e:
            part.unlink(missing_ok=True)
            raise IngestCommitError(str(e)) from e
        try:
            spatial_params = self._attach_spatial(adata, loadable, opts)
        except IngestCommitError:
            part.unlink(missing_ok=True)
            raise
        self._record_ingest_provenance(adata, loadable, opts, spatial_params)
        # Cell Ranger metrics / raw matrices that came with a sample (never raises).
        from scview.core import seq_metrics

        seq_metrics.attach_from_units(adata, [_seqqc_view(u) for u in loadable], opts.sample_label)

        self._attach_vdj(adata, loadable, opts)
        self._attach_guide_calls(adata, loadable, opts)

        dataset_id = uuid.uuid4().hex
        name = opts.name or (loadable[0].label if len(loadable) == 1 else "merged_dataset")
        # The originals, in their folders; not the files the bundle skipped.
        skip = {s.path for s in bundle.skipped}
        root = self.files_dir(sid)
        originals = [p for p in self.staged_paths(sid) if _rel_to(p, root) not in skip]
        try:
            self._persist_dataset(dataset_id, name, adata, originals, root, matrices=matrices)
        finally:
            part.unlink(missing_ok=True)
        logger.info("Committed ingest session %s as dataset %s (%s)%s", sid, dataset_id, name,
                    " in blocks" if matrices is not None else "")
        return dataset_id

    def _load_or_stream(self, loadable, opts: IngestOptions, workdir: Path, part: Path):
        """(dataset, None) read into memory, or (skeleton, ``part``) with the matrices
        written to ``part`` a block at a time, when reading them whole would not fit the
        memory free or the session asks for it (``streamed.choose``)."""
        try:
            choice = streamed.choose(loadable, opts.streamed, opts.genes_in_rows)
        except streamed.StreamUnsupported as e:
            raise IngestCommitError(f"This upload cannot be read in blocks: {e}.") from e
        if not choice.streamed:
            return self._load_and_merge(loadable, opts), None
        logger.info("Ingesting %s in blocks: %s", ", ".join(u.label for u in loadable), choice.reason)
        need, free = streamed.disk_need(loadable), shutil.disk_usage(self.data_dir).free
        if need > free:
            raise IngestCommitError(
                f"Saving this dataset needs about {need / 1024**3:.1f} GB of disk; "
                f"{free / 1024**3:.1f} GB is free where scView keeps its data. Free some disk "
                "space (or delete datasets you no longer need) and try again.")
        try:
            units = [streamed.open_unit(u, workdir) for u in loadable]
            recon = None
            if len(units) > 1 and opts.apply_reconciliation:
                recon = plan_merge({su.label: su.meta for su in units},
                                   streamed.counts_like(units)).reconciliation
            meta = streamed.ingest(units, part, join=opts.join, sample_label=opts.sample_label,
                                   reconciliation=recon)
        except streamed.StreamUnsupported as e:
            part.unlink(missing_ok=True)
            if opts.streamed == "on" or len(loadable) > 1:
                raise IngestCommitError(f"This upload cannot be read in blocks: {e}.") from e
            logger.warning("Read into memory instead of in blocks: %s", e)
            return self._load_and_merge(loadable, opts), None
        except IngestLoadError:
            part.unlink(missing_ok=True)
            raise
        except Exception as e:  # a write that failed (disk full, a damaged block)
            part.unlink(missing_ok=True)
            logger.exception("Ingest in blocks failed")
            raise IngestCommitError(f"The dataset could not be read and saved in blocks: {e}") from e
        return meta, part

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

    def _record_ingest_provenance(self, adata, loadable, opts: IngestOptions,
                                  spatial_params: dict | None = None) -> None:
        """Write the source block + an ingest history entry. Never raises."""
        try:
            if len(loadable) > 1:
                provenance.init_source(
                    adata,
                    origin="ingested",
                    original_filename=", ".join(u.label for u in loadable),
                    fmt="merged",
                    merged_from=[
                        {"sample": u.label, "format": u.format.value, "n_files": len(u.files)}
                        for u in loadable
                    ],
                    merge={
                        "join": opts.join.value,
                        "sample_label": opts.sample_label,
                        "apply_reconciliation": opts.apply_reconciliation,
                    },
                )
            else:
                u = loadable[0]
                provenance.init_source(
                    adata,
                    origin="ingested",
                    original_filename=", ".join(f.name for f in u.files),
                    fmt=u.format.value,
                )
            provenance.record_step(
                adata,
                step="ingest",
                tool="scview.ingestion",
                params={
                    "formats": sorted({u.format.value for u in loadable}),
                    "genes_in_rows": opts.genes_in_rows,
                    **_protein_params(adata),
                    **({"spatial": spatial_params} if spatial_params else {}),
                },
            )
        except Exception as e:  # provenance must never block ingest
            logger.warning("Could not record ingest provenance: %s", e)

    def _attach_vdj(self, adata, loadable, opts: IngestOptions) -> None:
        """TCR/BCR contig tables uploaded beside a unit's matrix (``core/vdj.py``),
        each matched to its own sample's cells in a merge."""
        from scview.core import vdj
        from scview.core.run_manifest import file_sha256

        merged = len(loadable) > 1
        for u in loadable:
            for f in [f for f in u.side_files if f.side_kind == SideKind.vdj]:
                try:
                    # Named by its folder path: a T and a B table share Cell Ranger's file name.
                    vdj.attach(adata, f.path, file_name=f.rel_path or f.name, sha256=file_sha256(f.path),
                               sample_column=opts.sample_label if merged else None,
                               sample=u.label if merged else None, stored_as=f.rel_path or f.name)
                except (vdj.VdjError, OSError, EOFError, zlib.error, ValueError) as e:
                    # A gzip cut off past the part detection read surfaces only here.
                    raise IngestCommitError(f"The contig table {f.name} could not be attached: {e}") from e

    def _attach_spatial(self, adata, loadable, opts: IngestOptions) -> dict | None:
        """Space Ranger spatial files uploaded with a unit (``core/spatial.py``): each
        sample's spots placed on its own cells, one library per sample (its label).
        Returns the summary for the ingest history entry (None without spatial files)."""
        from scview.core import spatial

        with_files = [u for u in loadable
                      if any(f.side_kind == SideKind.spatial for f in u.side_files)]
        if not with_files:
            return None
        if len(with_files) < len(loadable):
            without = ", ".join(u.label for u in loadable if u not in with_files)
            raise IngestCommitError(
                f"Only some samples came with Space Ranger spatial files ({without} did not). "
                "Upload the spatial samples together without the others, or add each sample's "
                "spatial folder.")
        merged = len(loadable) > 1
        for u in with_files:
            parts = [(f.spatial_part or "", f.path, f.rel_path or f.name)
                     for f in u.side_files if f.side_kind == SideKind.spatial]
            try:
                sr = spatial.read_spaceranger(parts)
                spatial.attach(adata, sr, u.label, sample_column=opts.sample_label if merged else None,
                               sample=u.label if merged else None, label=u.label)
            except (spatial.SpatialError, OSError, EOFError, zlib.error) as e:
                raise IngestCommitError(f"The spatial files of {u.label} could not be used: {e}") from e
        return spatial.finish(adata)

    def _attach_guide_calls(self, adata, loadable, opts: IngestOptions) -> None:
        """Cell Ranger's protospacer calls uploaded with a unit (``core/crispr.py``),
        each matched to its own sample's cells in a merge."""
        merged = len(loadable) > 1
        for u in loadable:
            for f in [f for f in u.side_files if f.side_kind == SideKind.crispr]:
                try:
                    crispr.attach_calls(adata, f.path, file_name=f.rel_path or f.name,
                                        sample_column=opts.sample_label if merged else None,
                                        sample=u.label if merged else None)
                except (crispr.CrisprError, OSError, EOFError, zlib.error, ValueError) as e:
                    raise IngestCommitError(f"The guide calls {f.name} could not be attached: {e}") from e

    def _load_and_merge(self, loadable, opts: IngestOptions):
        load_opts = {"genes_in_rows": opts.genes_in_rows}
        adatas = {u.label: load_unit(u, load_opts) for u in loadable}
        if len(adatas) == 1:
            return next(iter(adatas.values()))
        plan = plan_merge(adatas)
        recon = plan.reconciliation if opts.apply_reconciliation else None
        protein.common_panel(list(adatas.values()))  # CITE-seq: the shared antibodies, or none
        crispr.common_guides(list(adatas.values()))  # CRISPR guides: all of them, 0 where absent
        merged = merge_units(
            adatas, join=opts.join, sample_label=opts.sample_label, reconciliation=recon
        )
        protein.renormalise(merged)
        return merged

    def _persist_dataset(self, dataset_id: str, name: str, adata, originals: list[Path],
                         root: Path | None = None, matrices: Path | None = None) -> None:
        """``matrices``: the h5 a streamed ingest wrote the matrices to
        (``streamed.ingest``); ``adata`` is then the dataset without them."""
        # Originals → uploads/{id}/ (read-only), the untouched source, in the folders
        # they were uploaded in (``root``: the staging folder).
        up = self.data_dir / "uploads" / dataset_id
        up.mkdir(parents=True, exist_ok=True)
        total = 0
        for src in originals:
            if src.is_dir():
                continue  # store directories (e.g. zarr) is a later refinement
            dest = up / (_rel_to(src, root) if root is not None else src.name)
            dest.parent.mkdir(parents=True, exist_ok=True)
            _link_or_copy(src, dest)
            total += dest.stat().st_size
            _make_readonly(dest)

        # Canonical h5ad → ingested/{id}/ (read-only, gzip-compressed).
        ing = self.data_dir / "ingested" / dataset_id
        ing.mkdir(parents=True, exist_ok=True)
        h5_path = ing / f"{_safe_name(name)}.h5ad"
        _clear_index_name_clashes(adata)
        try:
            if matrices is not None:
                streamed.finish_h5ad(adata, matrices, h5_path)
            else:
                adata.write_h5ad(h5_path, compression="gzip")
        except Exception as e:
            # Leave no half-made dataset behind (an uploads dir without metadata).
            shutil.rmtree(up, ignore_errors=True)
            shutil.rmtree(ing, ignore_errors=True)
            raise IngestCommitError(f"The dataset could not be saved: {e}") from e
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
        if not is_safe_id(sid):
            raise ValueError(f"Invalid session id: {sid!r}")
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


def _link_or_copy(src: Path, dest: Path) -> None:
    """A second name for the staged file (its session is discarded after the commit), or a
    copy where no link can be made: a 4 GB upload is not copied a second time."""
    try:
        os.link(src, dest)
    except OSError:
        shutil.copyfile(src, dest)


def _rel_to(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name


def _safe_name(name: str) -> str:
    cleaned = _SAFE_NAME_RE.sub("_", name).strip("_")
    return cleaned or "dataset"


def _clear_index_name_clashes(adata) -> None:
    """Drop an obs/var index name that is also a column name (e.g. "cell_id" with
    different values): anndata refuses to write it. The column keeps its values."""
    for frame in (adata.obs, adata.var):
        if frame.index.name is not None and frame.index.name in frame.columns:
            logger.info("Index name %r is also a column; dropping the index name", frame.index.name)
            frame.index.name = None


def _make_readonly(path: Path) -> None:
    try:
        path.chmod(0o444)
    except OSError as e:
        logger.warning("Could not mark %s read-only: %s", path, e)
