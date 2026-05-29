"""Ingestion engine — multi-format, multi-file single-cell data import.

See ``docs/INGESTION_ENGINE.md`` for the overall design. Stage [1] (detection)
is implemented here; bundling, validation, loading and merge follow.
"""

from scview.core.ingestion.bundling import (
    Bundle,
    BundleFile,
    FileRole,
    IngestUnit,
    UnitFormat,
    build_bundle,
    bundle_from_detections,
)
from scview.core.ingestion.detection import (
    DetectionResult,
    FileKind,
    detect_file,
)
from scview.core.ingestion.validation import (
    IngestIssue,
    IssueSeverity,
    ValidationReport,
    validate_bundle,
)

__all__ = [
    "Bundle",
    "BundleFile",
    "DetectionResult",
    "FileKind",
    "FileRole",
    "IngestIssue",
    "IngestUnit",
    "IssueSeverity",
    "UnitFormat",
    "ValidationReport",
    "build_bundle",
    "bundle_from_detections",
    "detect_file",
    "validate_bundle",
]
