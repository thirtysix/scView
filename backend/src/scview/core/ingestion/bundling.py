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
import re
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
    side = "side"  # Cell Ranger metrics / raw matrix attached to a unit, never loaded
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

# Directory names that are NOT sample-specific (CellRanger conventions + our
# upload-staging dir), so they make poor default labels for a prefix-less MEX.
_GENERIC_MEX_DIRS = {
    "files", "filtered_feature_bc_matrix", "raw_feature_bc_matrix", "outs", "matrix",
}


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
    # metrics_summary.csv / raw matrix of this sample, read for sequencing QC (core/seq_metrics.py)
    side_files: list[BundleFile] = Field(default_factory=list)


class Bundle(BaseModel):
    """The set of units staged in one ingest session."""

    units: list[IngestUnit] = Field(default_factory=list)
    is_merge: bool = False  # more than one loadable unit
    complete: bool = False  # every (non-unknown) unit is complete
    issues: list[str] = Field(default_factory=list)


# nf-core/scrnaseq's mtx_conversions/ emits "<sample>_{raw,filtered,cellbender_filter}_matrix.h5ad"
# (+ "combined_matrix.h5ad" for the all-samples concat). Strip these so the dataset
# label is the sample name rather than the verbose nf-core filename.
_NFCORE_MATRIX_SUFFIXES = (
    "_raw_matrix",
    "_filtered_matrix",
    "_cellbender_filter_matrix",
    "_unfiltered_matrix",
    "_matrix",
)


def _clean_unit_label(stem: str) -> str:
    """Best label for a single-file unit, cleaning nf-core matrix-export suffixes."""
    if stem == "combined_matrix":
        return "combined"
    for suffix in _NFCORE_MATRIX_SUFFIXES:
        if stem.endswith(suffix) and len(stem) > len(suffix):
            return stem[: -len(suffix)]
    return stem


def build_bundle(paths: list[Path]) -> Bundle:
    """Detect, role-assign and group a set of staged paths into a Bundle."""
    detected = [(Path(p), detect_file(Path(p))) for p in paths]
    return bundle_from_detections(detected)


def bundle_from_detections(detected: list[tuple[Path, DetectionResult]]) -> Bundle:
    """Group already-detected files (split out for testing / reuse)."""
    single_files: list[BundleFile] = []
    mex_parts: list[tuple[Path, BundleFile]] = []
    unknown_files: list[BundleFile] = []

    side_files: list[BundleFile] = []

    for path, det in detected:
        if _is_side_file(det):
            side_files.append(BundleFile(path=str(path), name=path.name, kind=det.kind,
                                         role=FileRole.side, confidence=det.confidence))
            continue
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
                label=_clean_unit_label(Path(bf.name).stem),
                format=fmt,  # type: ignore[arg-type]
                files=[bf],
                complete=True,
            )
        )

    units.extend(_group_mex_units(mex_parts))
    units, raw_mex = _split_raw_mex(units)

    bundle_issues: list[str] = _attach_side_files(units, side_files, raw_mex)
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
        labels.setdefault(key, _mex_label(prefix, path.parent.name))

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


def _mex_label(prefix: str, parent_name: str) -> str:
    """Best label for a MEX unit: the filename prefix, else a meaningful parent
    directory, else a neutral default (the user renames it in the wizard)."""
    if prefix:
        return prefix
    if parent_name and parent_name.lower() not in _GENERIC_MEX_DIRS:
        return parent_name
    return "sample"


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


# ---------------------------------------------------------------------------
# Side files: Cell Ranger metrics_summary.csv and the raw (unfiltered) matrix
# ---------------------------------------------------------------------------

_RAW_TOKEN_RE = re.compile(r"(^|[_.\-])raw([_.\-]|$)")
_RAW_MEX_DIRS = ("raw_feature_bc_matrix", "raw_gene_bc_matrices")
_SIDE_EXTS = (".gz", ".h5", ".h5ad", ".csv", ".mtx", ".tsv", ".txt")
# Cell Ranger's own file stems, stripped to leave the sample prefix ("pbmc_1k_v3").
_TENX_TOKENS = (
    "sample_filtered_feature_bc_matrix", "sample_raw_feature_bc_matrix",
    "filtered_feature_bc_matrix", "raw_feature_bc_matrix",
    "filtered_gene_bc_matrices", "raw_gene_bc_matrices", "metrics_summary",
)


def _is_side_file(det: DetectionResult) -> bool:
    return det.kind == FileKind.tenx_metrics or (
        det.kind == FileKind.tenx_h5 and bool(det.details.get("raw_matrix"))
    )


def _strip_tenx(s: str) -> tuple[str, bool]:
    for tok in _TENX_TOKENS:
        if s.endswith(tok):
            return s[: -len(tok)], True
    return s, False


def sample_key(name: str) -> str:
    """The sample prefix of a Cell Ranger file name, shared by its metrics, raw and
    filtered files: ``pbmc_1k_v3_metrics_summary.csv``, ``pbmc_1k_v3_raw_feature_bc_matrix.h5``
    and ``pbmc_1k_v3_filtered_feature_bc_matrix.h5`` all give ``pbmc_1k_v3``; a bare
    ``metrics_summary.csv`` gives ""."""
    s = name.lower()
    changed = True
    while changed:
        changed = False
        for ext in _SIDE_EXTS:
            if s.endswith(ext):
                s = s[: -len(ext)]
                changed = True
    s, hit = _strip_tenx(s)
    if not hit:  # a MEX part: "<sample>_raw_feature_bc_matrix_barcodes"
        for tok in _ROLE_TOKENS:
            if s.endswith(tok) and len(s) > len(tok):
                s = s[: -len(tok)].rstrip(" _-.")
                break
        s, _ = _strip_tenx(s)
    s = s.rstrip(" _-.")
    for tok in ("raw", "filtered"):
        if s == tok or (s.endswith(tok) and s[-len(tok) - 1] in " _-."):
            s = s[: -len(tok)]
            break
    return s.strip(" _-.")


def _is_raw_mex(unit: IngestUnit) -> bool:
    if unit.format != UnitFormat.tenx_mex or not unit.files:
        return False
    prefix = _mex_prefix(unit.files[0].name)
    parent = Path(unit.files[0].path).parent.name.lower()
    named_raw = bool(_RAW_TOKEN_RE.search(prefix)) and "filtered" not in prefix
    return named_raw or any(d in parent for d in _RAW_MEX_DIRS)


def _split_raw_mex(units: list[IngestUnit]) -> tuple[list[IngestUnit], list[IngestUnit]]:
    """Raw (unfiltered) MEX triplets, named so, leave the data units: every barcode of
    the whitelist is in them, nearly all empty droplets."""
    raw = [u for u in units if _is_raw_mex(u)]
    return [u for u in units if u not in raw], raw


def _unit_key(unit: IngestUnit) -> str:
    if not unit.files:
        return unit.label.lower()
    first = unit.files[0]
    name = _mex_prefix(first.name) if unit.format == UnitFormat.tenx_mex else first.name
    return f"{Path(first.path).parent}|{sample_key(name)}"


def _side_key(bf: BundleFile) -> str:
    return f"{Path(bf.path).parent}|{sample_key(bf.name)}"


def _attach_side_files(
    units: list[IngestUnit], side_files: list[BundleFile], raw_mex: list[IngestUnit]
) -> list[str]:
    """Attach each metrics file / raw matrix to the unit with the same sample prefix
    (or to the only unit); return sentences for what could not be attached."""
    issues: list[str] = []
    # One item per metrics file or raw matrix (a raw MEX triplet is one item).
    items: list[tuple[str, str, list[BundleFile]]] = []  # (key, "metrics" | "raw", files)
    for bf in side_files:
        items.append((_side_key(bf), "metrics" if bf.kind == FileKind.tenx_metrics else "raw", [bf]))
    for u in raw_mex:
        names = ", ".join(f.name for f in u.files)
        if u.missing_roles:
            missing = ", ".join(r.value for r in u.missing_roles)
            issues.append(f"The raw 10x matrix files ({names}) are missing their {missing} file "
                          "and were not attached.")
            continue
        parts = [f.model_copy(update={"role": FileRole.side}) for f in u.files]
        key = f"{Path(u.files[0].path).parent}|{sample_key(_mex_prefix(u.files[0].name))}"
        items.append((key, "raw", parts))

    loadable = [u for u in units if u.format != UnitFormat.unknown]
    by_key: dict[str, IngestUnit] = {}
    for u in loadable:
        by_key.setdefault(_unit_key(u), u)
    for key, what, files in items:
        names = ", ".join(f.name for f in files)
        noun = "metrics summary" if what == "metrics" else "raw (unfiltered) matrix"
        unit = by_key.get(key) or (loadable[0] if len(loadable) == 1 else None)
        if unit is None:
            if not loadable:
                issues.append(
                    f"{names}: a Cell Ranger {noun} is attached to a sample for sequencing QC, "
                    "never loaded as the data. Add the filtered matrix "
                    "(filtered_feature_bc_matrix) as the dataset."
                )
            else:
                example = ("metrics_summary.csv" if what == "metrics"
                           else "raw_feature_bc_matrix.h5")
                issues.append(
                    f"{names} (a Cell Ranger {noun}) could not be matched to a sample by name and "
                    f"was not attached. Name it after its sample, e.g. "
                    f"{loadable[0].label}_{example}."
                )
            continue
        is_metrics = [f.kind == FileKind.tenx_metrics for f in unit.side_files]
        if (what == "metrics" and any(is_metrics)) or (what == "raw" and not all(is_metrics)):
            issues.append(f"{names}: sample {unit.label} already has a Cell Ranger {noun}; this "
                          "one was not attached.")
            continue
        unit.side_files.extend(files)
    return issues
