"""Ingestion engine — multi-format, multi-file single-cell data import.

See ``docs/INGESTION_ENGINE.md`` for the overall design. Stage [1] (detection)
is implemented here; bundling, validation, loading and merge follow.
"""

from scview.core.ingestion.detection import (
    DetectionResult,
    FileKind,
    detect_file,
)

__all__ = ["DetectionResult", "FileKind", "detect_file"]
