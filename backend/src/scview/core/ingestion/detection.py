"""File-type detection for the ingestion engine — stage [1] DETECT.

Content-first sniffing: classify a single uploaded file into a :class:`FileKind`
by inspecting its magic bytes and a short content peek, using the filename only
as a hint. Detection never trusts the extension alone — GEO downloads routinely
rename files (e.g. ``GSM123_barcodes.tsv.gz``), so a renamed barcodes file must
still resolve correctly.

This module looks at one path at a time and has no side effects beyond reading.
The bundler (stage [2]) is responsible for grouping detected files into units
and deciding completeness. See ``docs/INGESTION_ENGINE.md``.
"""

from __future__ import annotations

import gzip
import logging
import re
import zlib
from enum import Enum
from pathlib import Path

import h5py
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Signatures / heuristics
# ---------------------------------------------------------------------------

_HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"
_GZIP_MAGIC = b"\x1f\x8b"
_MTX_BANNER = "%%MatrixMarket"

# How many (decompressed) bytes to sniff for text content.
_SNIFF_BYTES = 8192

# Ensembl gene ids: human ENSG, mouse ENSMUSG, and other species ENS<species>G.
_ENSEMBL_RE = re.compile(r"^ENS[A-Z]*G\d{6,}", re.IGNORECASE)
# A 10x cell barcode: a run of ACGT(N), optionally a "-1" / ".1" lane suffix.
_BARCODE_RE = re.compile(r"^[ACGTN]{8,}([-.]\d+)?$", re.IGNORECASE)
# A plausible gene symbol / identifier token.
_GENEISH_RE = re.compile(r"^[A-Za-z0-9._-]{1,30}$")

# Filename substrings → the kind they hint at (a confidence nudge, never an
# override of content).
_NAME_HINTS: list[tuple[str, "FileKind"]] = []  # populated after FileKind defined


class FileKind(str, Enum):
    """What a single ingest file appears to be."""

    anndata_h5ad = "anndata_h5ad"
    tenx_h5 = "tenx_h5"
    loom = "loom"
    zarr_dir = "zarr_dir"
    mtx_matrix = "mtx_matrix"
    tenx_barcodes = "tenx_barcodes"
    tenx_features = "tenx_features"
    dense_table = "dense_table"
    seurat_rds = "seurat_rds"
    unknown = "unknown"


_NAME_HINTS = [
    ("barcode", FileKind.tenx_barcodes),
    ("feature", FileKind.tenx_features),
    ("genes", FileKind.tenx_features),
    ("matrix.mtx", FileKind.mtx_matrix),
]


class DetectionResult(BaseModel):
    """Outcome of sniffing one file."""

    kind: FileKind
    confidence: float = 0.0  # 0..1
    # anndata | 10x | loom | zarr | matrix_market | table | seurat | hdf5
    format_family: str = ""
    reason: str = ""  # human-readable; feeds user messaging / debugging
    details: dict = Field(default_factory=dict)


def detect_file(path: Path) -> DetectionResult:
    """Classify a single file (or store directory) by content, then filename."""
    path = Path(path)
    if not path.exists():
        return DetectionResult(kind=FileKind.unknown, reason=f"Path does not exist: {path}")
    if path.is_dir():
        return _detect_dir(path)

    name = path.name.lower()

    # R serialization is binary and gzip-compressed by default (so it would look
    # like a gzipped text file); trust the extension here.
    if name.endswith((".rds", ".rdata", ".rda")):
        return DetectionResult(
            kind=FileKind.seurat_rds,
            confidence=0.7,
            format_family="seurat",
            reason="R data file (.rds/.RData) — object type is verified during conversion",
        )

    head = _read_head(path, 16)

    # HDF5 family: h5ad / 10x .h5 / loom all share this magic.
    if head.startswith(_HDF5_MAGIC):
        return _classify_hdf5(path)

    gzipped = head.startswith(_GZIP_MAGIC)
    text = _peek_text(path, gzipped)
    if text is None:
        result = DetectionResult(
            kind=FileKind.unknown,
            confidence=0.0,
            reason="Unrecognised binary file",
            details={"gzipped": gzipped},
        )
        return _apply_name_hint(result, name)

    result = _classify_text(text, gzipped)
    return _apply_name_hint(result, name)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _detect_dir(path: Path) -> DetectionResult:
    """Directories are only recognised as Zarr stores; MEX dirs are the bundler's job."""
    is_zarr = (
        path.suffix.lower() == ".zarr"
        or (path / ".zgroup").exists()
        or (path / ".zattrs").exists()
        or (path / "zarr.json").exists()  # zarr v3
    )
    if is_zarr:
        return DetectionResult(
            kind=FileKind.zarr_dir,
            confidence=0.9,
            format_family="zarr",
            reason="Zarr store directory",
        )
    return DetectionResult(
        kind=FileKind.unknown,
        confidence=0.0,
        reason="Directory is not a recognised store (expected a .zarr store)",
    )


def _read_head(path: Path, n: int) -> bytes:
    try:
        with open(path, "rb") as fh:
            return fh.read(n)
    except OSError as e:
        logger.warning("Could not read %s: %s", path, e)
        return b""


def _peek_text(path: Path, gzipped: bool) -> str | None:
    """Return up to _SNIFF_BYTES of decoded text, or None if the file isn't text."""
    try:
        if gzipped:
            with gzip.open(path, "rt", encoding="utf-8", errors="strict") as fh:
                return fh.read(_SNIFF_BYTES)
        with open(path, "rt", encoding="utf-8", errors="strict") as fh:
            return fh.read(_SNIFF_BYTES)
    except (OSError, UnicodeDecodeError, EOFError, zlib.error) as e:
        # gzip.BadGzipFile subclasses OSError; truncated deflate raises zlib.error.
        logger.debug("Text peek failed for %s: %s", path, e)
        return None


def _classify_hdf5(path: Path) -> DetectionResult:
    """Disambiguate the HDF5 family by top-level structure."""
    try:
        with h5py.File(path, "r") as f:
            keys = set(f.keys())
            details = {"hdf5_keys": sorted(keys)}

            # 10x CellRanger .h5: a /matrix *group* with CSC arrays + features.
            if "matrix" in keys and isinstance(f["matrix"], h5py.Group):
                mkeys = set(f["matrix"].keys())
                if {"data", "indptr"} <= mkeys:
                    return DetectionResult(
                        kind=FileKind.tenx_h5,
                        confidence=0.95,
                        format_family="10x",
                        reason="10x CellRanger HDF5 (/matrix group with sparse arrays)",
                        details=details,
                    )

            # Loom: row_attrs / col_attrs groups beside a /matrix *dataset*.
            if {"row_attrs", "col_attrs"} <= keys:
                return DetectionResult(
                    kind=FileKind.loom,
                    confidence=0.95,
                    format_family="loom",
                    reason="Loom file (row_attrs / col_attrs)",
                    details=details,
                )

            # AnnData h5ad: obs + var (+ usually X).
            if {"obs", "var"} <= keys:
                return DetectionResult(
                    kind=FileKind.anndata_h5ad,
                    confidence=0.95,
                    format_family="anndata",
                    reason="AnnData h5ad (obs / var groups)",
                    details=details,
                )

            return DetectionResult(
                kind=FileKind.unknown,
                confidence=0.2,
                format_family="hdf5",
                reason="HDF5 file with unrecognised structure",
                details=details,
            )
    except (OSError, KeyError, RuntimeError) as e:
        logger.warning("HDF5 inspection failed for %s: %s", path, e)
        return DetectionResult(
            kind=FileKind.unknown, confidence=0.0, reason=f"Could not open as HDF5: {e}"
        )


def _classify_text(text: str, gzipped: bool) -> DetectionResult:
    """Classify delimited / line-oriented text content."""
    details: dict = {"gzipped": gzipped}
    stripped = text.lstrip()

    if stripped.startswith(_MTX_BANNER):
        return DetectionResult(
            kind=FileKind.mtx_matrix,
            confidence=0.97,
            format_family="matrix_market",
            reason="Matrix Market sparse matrix (%%MatrixMarket banner)",
            details=details,
        )

    lines = [ln for ln in stripped.splitlines() if ln.strip()]
    if not lines:
        return DetectionResult(
            kind=FileKind.unknown,
            confidence=0.0,
            reason="Empty or whitespace-only file",
            details=details,
        )

    delim, ncols = _guess_delimiter(lines)
    details["delimiter"] = {"\t": "tab", ",": "comma", None: "whitespace"}.get(delim, delim)
    details["n_columns"] = ncols

    rows = [_split(ln, delim) for ln in lines[:10]]
    col0 = [r[0].strip() for r in rows if r and r[0].strip()]

    # Single-column: cell barcodes, or a bare gene list (genes.txt sidecar).
    if ncols == 1:
        if _fraction_matching(col0, _BARCODE_RE) >= 0.6:
            return DetectionResult(
                kind=FileKind.tenx_barcodes,
                confidence=0.85,
                format_family="10x",
                reason="Single column of cell barcodes",
                details=details,
            )
        if _looks_geneish(col0):
            return DetectionResult(
                kind=FileKind.tenx_features,
                confidence=0.5,
                format_family="10x",
                reason="Single column of gene-like identifiers (feature list)",
                details=details,
            )
        return DetectionResult(
            kind=FileKind.unknown,
            confidence=0.2,
            reason="Single-column text of unknown content",
            details=details,
        )

    # 2–3 tab-separated columns with gene ids → 10x features/genes table.
    if delim == "\t" and 2 <= ncols <= 3:
        if _fraction_matching(col0, _ENSEMBL_RE) >= 0.5 or _looks_geneish(col0):
            return DetectionResult(
                kind=FileKind.tenx_features,
                confidence=0.85,
                format_family="10x",
                reason=f"{ncols}-column features/genes table (gene id + symbol)",
                details=details,
            )

    # Wide delimited grid → dense expression table.
    if ncols > 3:
        return DetectionResult(
            kind=FileKind.dense_table,
            confidence=0.7,
            format_family="table",
            reason=f"Dense delimited table ({ncols} columns, {details['delimiter']}-separated)",
            details=details,
        )

    return DetectionResult(
        kind=FileKind.unknown,
        confidence=0.2,
        reason="Delimited text that did not match a known layout",
        details=details,
    )


def _guess_delimiter(lines: list[str]) -> tuple[str | None, int]:
    """Pick tab/comma/whitespace by counting on a data line; return (delim, n_columns)."""
    probe = lines[1] if len(lines) > 1 else lines[0]
    tab, comma = probe.count("\t"), probe.count(",")
    if tab == 0 and comma == 0:
        ws_cols = len(probe.split())
        return (None, ws_cols) if ws_cols > 1 else ("\t", 1)
    if tab >= comma:
        return "\t", tab + 1
    return ",", comma + 1


def _split(line: str, delim: str | None) -> list[str]:
    return line.split(delim) if delim else line.split()


def _fraction_matching(tokens: list[str], regex: re.Pattern) -> float:
    toks = [t for t in tokens if t]
    if not toks:
        return 0.0
    return sum(1 for t in toks if regex.match(t)) / len(toks)


def _looks_geneish(tokens: list[str]) -> bool:
    toks = [t for t in tokens if t]
    if not toks:
        return False
    ok = sum(1 for t in toks if _GENEISH_RE.match(t) and not t.isdigit())
    return ok / len(toks) >= 0.6


def _apply_name_hint(result: DetectionResult, name: str) -> DetectionResult:
    """Use the filename as a tiebreaker: boost agreeing confidence, or rescue an unknown."""
    hint = next((kind for sub, kind in _NAME_HINTS if sub in name), None)
    if hint is None:
        return result
    if result.kind == hint:
        result.confidence = min(1.0, result.confidence + 0.1)
        result.details["name_hint"] = "agrees"
    elif result.kind == FileKind.unknown:
        result.kind = hint
        result.confidence = 0.4
        result.format_family = "10x"
        result.reason = f"{result.reason}; filename suggests {hint.value}".lstrip("; ")
        result.details["name_hint"] = "applied"
    else:
        result.details["name_hint"] = f"conflicts ({hint.value})"
    return result
