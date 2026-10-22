"""Ingestion engine — upload paths.

A folder upload sends each file's path relative to the chosen folder (the
browser's ``webkitRelativePath``, e.g. ``pbmc/outs/raw_feature_bc_matrix/barcodes.tsv.gz``),
so that Cell Ranger's same-named files (the raw and filtered ``barcodes.tsv.gz``,
the T and B ``filtered_contig_annotations.csv``) stay apart and the folders can
say which sample a file belongs to. This module makes such a path safe to stage
under the session directory, and names the files a folder holds that scView
never reads (alignments, reports, Loupe files), so they are skipped with a reason.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import PurePosixPath

MAX_DEPTH = 16  # folders + file name
MAX_PATH_CHARS = 1024
MAX_PART_BYTES = 255  # one name, as most file systems allow

_DRIVE_RE = re.compile(r"^[A-Za-z]:")


class UploadPathError(ValueError):
    """A client-sent path that cannot be staged; the message says why."""


def safe_relative_path(raw: str) -> str:
    """``raw`` as a relative POSIX path that stays under the staging directory.

    Backslashes count as separators (a Windows client); unicode is NFC-normalised
    (macOS sends decomposed names, so the same name in two forms is one path).
    Refused, with a sentence naming the path: an empty path, an absolute one or
    a drive letter (``/x``, ``C:\\x``, ``\\\\server\\x``), a ``..``, ``.`` or empty
    part, NUL or another control character, and paths deeper than ``MAX_DEPTH``
    parts, longer than ``MAX_PATH_CHARS`` or with a part over ``MAX_PART_BYTES``.
    """
    if not isinstance(raw, str) or not raw.strip():
        raise UploadPathError("A file was sent without a name.")
    s = unicodedata.normalize("NFC", raw).replace("\\", "/")
    shown = s[:120]
    if any(ord(c) < 32 or ord(c) == 127 for c in s):
        raise UploadPathError(f"The path {shown!r} holds a control character.")
    if s.startswith("/") or _DRIVE_RE.match(s):
        raise UploadPathError(f"The path {shown!r} is absolute; only paths inside the chosen "
                              "folder are accepted.")
    parts = s.split("/")
    if any(p in ("", ".", "..") for p in parts):
        raise UploadPathError(f"The path {shown!r} has an empty, '.' or '..' part.")
    if len(parts) > MAX_DEPTH:
        raise UploadPathError(f"The path {shown!r} is more than {MAX_DEPTH} folders deep.")
    if len(s) > MAX_PATH_CHARS or any(len(p.encode("utf-8")) > MAX_PART_BYTES for p in parts):
        raise UploadPathError(f"The path {shown}... is too long.")
    return "/".join(parts)


def flat_name(filename: str | None) -> str:
    """A plain (non-folder) upload's file name: the last part of whatever the
    client sent, either separator, as plain uploads always stored it."""
    name = (filename or "").replace("\\", "/").rsplit("/", 1)[-1]
    return name or "unnamed"


# ---------------------------------------------------------------------------
# Files a folder holds that scView never reads
# ---------------------------------------------------------------------------

SYSTEM = "system files"
ALIGNMENTS = "read alignments (scView starts from the count matrix)"
REPORT = "Cell Ranger's HTML report"
LOUPE = "Loupe Browser files"
MOLECULES = "per-molecule read information, not a count matrix"

_JUNK_NAMES = {
    ".ds_store": SYSTEM, "thumbs.db": SYSTEM, "desktop.ini": SYSTEM,
    "web_summary.html": REPORT, "molecule_info.h5": MOLECULES,
    "sample_molecule_info.h5": MOLECULES,
}
_JUNK_SUFFIXES = (
    (".bam", ALIGNMENTS), (".bai", ALIGNMENTS), (".bam.bai", ALIGNMENTS), (".csi", ALIGNMENTS),
    (".cram", ALIGNMENTS), (".crai", ALIGNMENTS), (".cloupe", LOUPE), (".vloupe", LOUPE),
)


def junk_reason(rel: str) -> str | None:
    """Why scView skips this file of a folder (a reason above), or None to keep it.
    The frontend skips the same names before sending (``lib/ingestUpload.ts``), so
    a BAM is never uploaded; this is the server's own check."""
    name = PurePosixPath(rel).name.lower()
    if name in _JUNK_NAMES:
        return _JUNK_NAMES[name]
    if name.startswith("._"):  # macOS AppleDouble resource forks
        return SYSTEM
    for suffix, reason in _JUNK_SUFFIXES:
        if name.endswith(suffix):
            return reason
    return None


def skipped_sentence(skipped: list[dict]) -> str:
    """One sentence for every skipped file, grouped by reason: what and why.
    ``skipped``: ``{"path", "reason", "bytes"?}`` dicts."""
    if not skipped:
        return ""
    groups: dict[str, list[dict]] = {}
    for item in skipped:
        groups.setdefault(str(item.get("reason") or "not read"), []).append(item)
    parts = []
    for reason, items in groups.items():
        names = [PurePosixPath(str(i.get("path"))).name for i in items]
        shown = ", ".join(names[:3]) + (f" and {len(names) - 3} more" if len(names) > 3 else "")
        size = sum(int(i.get("bytes") or 0) for i in items)
        parts.append(f"{shown} ({reason}{', ' + _size(size) if size else ''})")
    n = len(skipped)
    return f"Skipped {n} file{'s' if n != 1 else ''} scView does not read: " + "; ".join(parts) + "."


def _size(n: int) -> str:
    for unit, scale in (("GB", 1024**3), ("MB", 1024**2), ("KB", 1024)):
        if n >= scale:
            return f"{n / scale:.1f} {unit}"
    return f"{n} bytes"
