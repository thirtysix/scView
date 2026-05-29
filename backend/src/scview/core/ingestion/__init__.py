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
from scview.core.ingestion.loading import (
    IngestLoadError,
    load_unit,
)
from scview.core.ingestion.merge import (
    MergeJoin,
    MergePlan,
    Reconciliation,
    VarBasis,
    VarReset,
    classify_var_basis,
    merge_units,
    plan_merge,
)
from scview.core.ingestion.session import (
    IngestCommitError,
    IngestOptions,
    IngestSessionManager,
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
    "IngestCommitError",
    "IngestLoadError",
    "IngestOptions",
    "IngestSessionManager",
    "IngestUnit",
    "IssueSeverity",
    "MergeJoin",
    "MergePlan",
    "Reconciliation",
    "UnitFormat",
    "ValidationReport",
    "VarBasis",
    "VarReset",
    "build_bundle",
    "bundle_from_detections",
    "classify_var_basis",
    "detect_file",
    "load_unit",
    "merge_units",
    "plan_merge",
    "validate_bundle",
]
