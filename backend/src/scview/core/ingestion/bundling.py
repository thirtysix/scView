"""Ingestion engine — stage [2] BUNDLE.

Group detected files into *units* and a *bundle*:

* **Unit** = one experiment/sample. A 10x MEX unit is up to three files
  (matrix + barcodes + features/genes); every other format is a single file or
  store directory.
* **Bundle** = the output of one ingest session: a single unit, or several units
  to be merged (a ``sample`` label is added at merge time — see §3a of the design).

The bundler runs detection (stage [1]) on each path, assigns a role, groups the
10x MEX parts by GEO-style prefix / parent directory, and reports completeness
— precisely which companion files a partial unit is still missing. It performs
no loading and has no side effects beyond reading. See ``docs/INGESTION_ENGINE.md``.
"""

from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field

from scview.core.ingestion.detection import DetectionResult, FileKind, detect_file

logger = logging.getLogger(__name__)


class UnitFormat(str, Enum):
    """The recipe by which a unit will be loaded."""

    tenx_mex = "tenx_mex"
    tenx_h5 = "tenx_h5"
    anndata = "anndata"
    loom = "loom"
    zarr = "zarr"
    dense_table = "dense_table"
    seurat = "seurat"
    unknown = "unknown"


class FileRole(str, Enum):
    """A file's role within its unit."""

    matrix = "matrix"
    barcodes = "barcodes"
    features = "features"
    data = "data"  # the single file of a single-file format
    unknown = "unknown"


# A 10x MEX unit needs exactly one of each of these.
_MEX_REQUIRED: tuple[FileRole, ...] = (FileRole.matrix, FileRole.barcodes, FileRole.features)

# FileKind → (UnitFormat for a single-file unit, FileRole). MEX parts map to a
# role but no standalone format (they only exist inside a tenx_mex unit).
_KIND_MAP: dict[FileKind, tuple[UnitFormat | None, FileRole]] = {
    FileKind.anndata_h5ad: (UnitFormat.anndata, FileRole.data),
    FileKind.tenx_h5: (UnitFormat.tenx_h5, FileRole.data),
    FileKind.loom: (UnitFormat.loom, FileRole.data),
    FileKind.zarr_dir: (UnitFormat.zarr, FileRole.data),
    FileKind.dense_table: (UnitFormat.dense_table, FileRole.data),
    FileKind.seurat_rds: (UnitFormat.seurat, FileRole.data),
    FileKind.mtx_matrix: (None, FileRole.matrix),
    FileKind.tenx_barcodes: (None, FileRole.barcodes),
    FileKind.tenx_features: (None, FileRole.features),
    FileKind.unknown: (None, FileRole.unknown),
}

# Tokens stripped when deriving a 10x MEX grouping prefix from a filename.
_STRIP_EXTS = (".gz", ".mtx", ".tsv", ".csv", ".txt")
_ROLE_TOKENS = ("matrix", "barcodes", "features", "genes")


class BundleFile(BaseModel):
    """One file placed into a unit."""

    path: str
    name: str
    kind: FileKind
    role: FileRole
    confidence: float = 0.0


class IngestUnit(BaseModel):
    """One experiment/sample, possibly still incomplete."""

    label: str
    format: UnitFormat
    files: list[BundleFile] = Field(default_factory=list)
    complete: bool = False
    missing_roles: list[FileRole] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)


class Bundle(BaseModel):
    """The set of units staged in one ingest session."""

    units: list[IngestUnit] = Field(default_factory=list)
    is_merge: bool = False  # more than one loadable unit
    complete: bool = False  # every (non-unknown) unit is complete
    issues: list[str] = Field(default_factory=list)


def build_bundle(paths: list[Path]) -> Bundle:
    """Detect, role-assign and group a set of staged paths into a Bundle."""
    detected = [(Path(p), detect_file(Path(p))) for p in paths]
    return bundle_from_detections(detected)


def bundle_from_detections(detected: list[tuple[Path, DetectionResult]]) -> Bundle:
    """Group already-detected files (split out for testing / reuse)."""
    single_files: list[BundleFile] = []
    mex_parts: list[tuple[Path, BundleFile]] = []
    unknown_files: list[BundleFile] = []

    for path, det in detected:
        fmt, role = _KIND_MAP.get(det.kind, (None, FileRole.unknown))
        bf = BundleFile(
            path=str(path), name=path.name, kind=det.kind, role=role, confidence=det.confidence
        )
        if role == FileRole.data and fmt is not None:
            single_files.append(bf)
        elif role in _MEX_REQUIRED:
            mex_parts.append((path, bf))
        else:
            unknown_files.append(bf)

    units: list[IngestUnit] = []
    for bf in single_files:
        fmt, _ = _KIND_MAP[bf.kind]
        units.append(
            IngestUnit(
                label=Path(bf.name).stem,
                format=fmt,  # type: ignore[arg-type]
                files=[bf],
                complete=True,
            )
        )

    units.extend(_group_mex_units(mex_parts))

    bundle_issues: list[str] = []
    if unknown_files:
        names = ", ".join(f.name for f in unknown_files)
        bundle_issues.append(
            f"{len(unknown_files)} file(s) could not be recognised and were ignored: {names}"
        )

    loadable = [u for u in units if u.format != UnitFormat.unknown]
    return Bundle(
        units=sorted(units, key=lambda u: u.label),
        is_merge=len(loadable) > 1,
        complete=bool(loadable) and all(u.complete for u in loadable),
        issues=bundle_issues,
    )


# ---------------------------------------------------------------------------
# 10x MEX grouping
# ---------------------------------------------------------------------------


def _group_mex_units(mex_parts: list[tuple[Path, BundleFile]]) -> list[IngestUnit]:
    """Group matrix/barcodes/features files into MEX units by parent dir + prefix."""
    if not mex_parts:
        return []

    groups: dict[str, list[BundleFile]] = {}
    labels: dict[str, str] = {}
    for path, bf in mex_parts:
        prefix = _mex_prefix(path.name)
        key = f"{path.parent}|{prefix}"
        groups.setdefault(key, []).append(bf)
        labels.setdefault(key, prefix or path.parent.name or "sample")

    units: list[IngestUnit] = []
    for key, files in groups.items():
        units.append(_assemble_mex_unit(labels[key], files))
    return units


def _assemble_mex_unit(label: str, files: list[BundleFile]) -> IngestUnit:
    present = {f.role for f in files}
    missing = [r for r in _MEX_REQUIRED if r not in present]

    issues: list[str] = []
    # Duplicate roles (e.g. two matrix files grouped together) — flag, don't guess.
    for role in _MEX_REQUIRED:
        if sum(1 for f in files if f.role == role) > 1:
            issues.append(f"More than one {role.value} file in this sample — keep only one.")
    if missing:
        issues.append(
            "Missing " + ", ".join(r.value for r in missing) + " for this 10x sample."
        )

    return IngestUnit(
        label=label,
        format=UnitFormat.tenx_mex,
        files=sorted(files, key=lambda f: f.role.value),
        complete=not missing and not issues,
        missing_roles=missing,
        issues=issues,
    )


def _mex_prefix(name: str) -> str:
    """Derive a grouping prefix by stripping extensions and the role token.

    ``GSM4711_barcodes.tsv.gz`` → ``gsm4711``; a bare ``barcodes.tsv`` → ``""``
    (so a standard triplet in one folder groups together).
    """
    s = name.lower()
    changed = True
    while changed:
        changed = False
        for ext in _STRIP_EXTS:
            if s.endswith(ext):
                s = s[: -len(ext)]
                changed = True
    for tok in _ROLE_TOKENS:
        if s.endswith(tok):
            s = s[: -len(tok)]
            break
    return s.strip(" _-.")
