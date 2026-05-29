"""Ingestion engine — stage [3] VALIDATE.

Turn a :class:`Bundle` into a list of structured, human-readable
:class:`IngestIssue`s — the "forgiving & helpful" layer. Every problem is phrased
for a non-technical user with a concrete next step, instead of a raw exception.

Pre-load checks only (no full matrix read): completeness messaging, 10x MEX
dimension agreement (matrix header vs barcode/feature counts), and gzip
integrity. Deeper checks that need the whole matrix (zero-variance genes, etc.)
happen during stage [4] LOAD. See ``docs/INGESTION_ENGINE.md`` §2[3] and §4.
"""

from __future__ import annotations

import gzip
import logging
import zlib
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field

from scview.core.ingestion.bundling import Bundle, FileRole, IngestUnit, UnitFormat

logger = logging.getLogger(__name__)


class IssueSeverity(str, Enum):
    info = "info"
    warn = "warn"
    error = "error"  # blocks ingest until resolved


class IngestIssue(BaseModel):
    severity: IssueSeverity
    code: str
    message: str
    suggestion: str = ""
    unit_label: str = ""


class ValidationReport(BaseModel):
    issues: list[IngestIssue] = Field(default_factory=list)
    ok: bool = True  # False if any error-severity issue is present


def validate_bundle(bundle: Bundle) -> ValidationReport:
    """Validate every unit in a bundle and collect issues."""
    issues: list[IngestIssue] = []
    for unit in bundle.units:
        issues.extend(_validate_unit(unit))
    ok = not any(i.severity == IssueSeverity.error for i in issues)
    return ValidationReport(issues=issues, ok=ok)


# ---------------------------------------------------------------------------
# Per-unit validation
# ---------------------------------------------------------------------------


def _validate_unit(unit: IngestUnit) -> list[IngestIssue]:
    issues: list[IngestIssue] = []
    issues.extend(_check_gzip_integrity(unit))
    if unit.format == UnitFormat.tenx_mex:
        issues.extend(_validate_mex(unit))
    return issues


def _validate_mex(unit: IngestUnit) -> list[IngestIssue]:
    issues: list[IngestIssue] = []

    # 1) Missing companions — the most common case, phrased helpfully.
    if unit.missing_roles:
        missing = _join([r.value for r in unit.missing_roles])
        issues.append(
            IngestIssue(
                severity=IssueSeverity.error,
                code="mex_incomplete",
                message=(
                    f"This 10x sample is missing its {missing} file(s). A 10x matrix "
                    "can't be read on its own."
                ),
                suggestion=(
                    "Add the companion files from the same folder: barcodes.tsv(.gz) "
                    "(cell IDs) and features.tsv(.gz) (the gene list; older datasets call "
                    "it genes.tsv)."
                ),
                unit_label=unit.label,
            )
        )

    # 2) Duplicate roles flagged by the bundler.
    for issue_text in unit.issues:
        if "more than one" in issue_text.lower():
            issues.append(
                IngestIssue(
                    severity=IssueSeverity.error,
                    code="mex_duplicate_role",
                    message=issue_text,
                    suggestion="Remove the extra file so each sample has one of each.",
                    unit_label=unit.label,
                )
            )

    # 3) Dimension agreement — only when the triplet is otherwise complete.
    if not unit.missing_roles:
        issues.extend(_check_mex_dimensions(unit))
    return issues


def _check_mex_dimensions(unit: IngestUnit) -> list[IngestIssue]:
    matrix = _file_for_role(unit, FileRole.matrix)
    barcodes = _file_for_role(unit, FileRole.barcodes)
    features = _file_for_role(unit, FileRole.features)
    if not (matrix and barcodes and features):
        return []

    dims = _read_mtx_dims(Path(matrix.path))
    if dims is None:
        return [
            IngestIssue(
                severity=IssueSeverity.warn,
                code="mtx_header_unreadable",
                message="Couldn't read the matrix dimensions from the .mtx header.",
                suggestion="The file may be truncated — try re-uploading it.",
                unit_label=unit.label,
            )
        ]

    n_rows, n_cols, _ = dims
    n_barcodes = _count_lines(Path(barcodes.path))
    n_features = _count_lines(Path(features.path))

    # 10x convention: matrix is features (rows) x barcodes (cols).
    if n_rows == n_features and n_cols == n_barcodes:
        return []
    if n_rows == n_barcodes and n_cols == n_features:
        return [
            IngestIssue(
                severity=IssueSeverity.warn,
                code="mtx_transposed",
                message="The matrix appears transposed (cells in rows, genes in columns).",
                suggestion="scView will orient it automatically (features × cells).",
                unit_label=unit.label,
            )
        ]
    return [
        IngestIssue(
            severity=IssueSeverity.error,
            code="mtx_dimension_mismatch",
            message=(
                f"The matrix is {n_rows} × {n_cols}, but there are {n_features} genes and "
                f"{n_barcodes} cell barcodes — these don't line up."
            ),
            suggestion=(
                "The barcodes/features files are probably from a different sample than the "
                "matrix. Re-upload the files that came from the same folder."
            ),
            unit_label=unit.label,
        )
    ]


def _check_gzip_integrity(unit: IngestUnit) -> list[IngestIssue]:
    issues: list[IngestIssue] = []
    for bf in unit.files:
        p = Path(bf.path)
        if p.suffix.lower() != ".gz":
            continue
        try:
            with gzip.open(p, "rb") as fh:
                fh.read(1024)
        except (OSError, EOFError, zlib.error):
            issues.append(
                IngestIssue(
                    severity=IssueSeverity.error,
                    code="gzip_truncated",
                    message=f"{bf.name} looks cut off (the gzip data ends early).",
                    suggestion="The upload may have been interrupted — please re-upload it.",
                    unit_label=unit.label,
                )
            )
    return issues


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def _read_mtx_dims(path: Path) -> tuple[int, int, int] | None:
    """Return (n_rows, n_cols, n_entries) from a MatrixMarket header, or None."""
    try:
        opener = gzip.open if path.suffix.lower() == ".gz" else open
        with opener(path, "rt", encoding="utf-8", errors="strict") as fh:
            for line in fh:
                s = line.strip()
                if not s or s.startswith("%"):
                    continue
                parts = s.split()
                if len(parts) >= 3:
                    return int(parts[0]), int(parts[1]), int(parts[2])
                return None
    except (OSError, ValueError, EOFError, zlib.error) as e:
        logger.debug("Could not read mtx dims from %s: %s", path, e)
    return None


def _count_lines(path: Path) -> int:
    """Count non-empty lines (one barcode / feature per line)."""
    try:
        opener = gzip.open if path.suffix.lower() == ".gz" else open
        with opener(path, "rt", encoding="utf-8", errors="strict") as fh:
            return sum(1 for line in fh if line.strip())
    except (OSError, EOFError, zlib.error) as e:
        logger.debug("Could not count lines in %s: %s", path, e)
        return -1


def _file_for_role(unit: IngestUnit, role: FileRole):
    return next((f for f in unit.files if f.role == role), None)


def _join(items: list[str]) -> str:
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]
