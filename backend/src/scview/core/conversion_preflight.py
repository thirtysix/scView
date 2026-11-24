"""Converter preflight: the memory an .rds / .RData conversion needs, before it starts.

An R file has no header to read: its object is one serialized stream, compressed
(gzip by default, or bzip2 / xz). What the converter holds at its peak is that whole
object as R reads it (readRDS loads all of it, layers it does not write included),
plus a fixed base for R, Python and anndata. So the estimate decompresses the stream
without keeping it, counting its bytes (the serialized size is close to the object's
size in memory), and adds the base:

    peak = CONVERTER_BASE_BYTES + CONVERTER_PER_SERIALIZED_BYTE * serialized bytes

Measured (v0.82.0, the converter process's VmHWM; serialized size -> peak, GiB):
synthetic Seurat v5 objects with counts, data, scale.data, pca/umap and nn/snn graphs,
8,000 cells x 20,000 genes in 3 split layers 0.56 -> 1.12, 12,000 x 34,209 1.60 -> 1.96,
23,578 x 34,209 (the shape of the 331 MB object the hub converted) 3.14 -> 3.50, and
a real 2,199-cell object 0.43 -> 0.88: within 10% of the fit. The 23,578-cell one
converted at a 3.6 GiB limit and was killed at 3.2 GiB (0.81.0's converter was killed
at 12 GB). The compressed size alone is a poor guide: these objects compress 2 to 5
times, others more. The decompression is timed: after ``COUNT_SECONDS`` the rest is
extrapolated from the share read so far.

The verdict is a warning, never a refusal, worded as the backend preflight's
(``core/preflight.py``), against the converter's own memory (its GET /health reads
its cgroup).
"""

from __future__ import annotations

import bz2
import logging
import lzma
import os
import time
import zlib
from dataclasses import dataclass
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

GiB = 1024 ** 3

# Fitted on the measurements above (see the module docstring).
CONVERTER_BASE_BYTES = int(0.45 * GiB)
CONVERTER_PER_SERIALIZED_BYTE = 1.0
# Largest error of the fit on the measured objects, rounded up.
ERROR_BAND = 0.15
# The conversion's time (2 CPUs, gzip): a few seconds, then per GB of serialized object.
CONVERT_SECONDS = 3.0
CONVERT_SECONDS_PER_GB = 5.5

COUNT_SECONDS = 20.0
_READ = 4 * 1024 * 1024

_GZIP = b"\x1f\x8b"
_BZIP2 = b"BZh"
_XZ = b"\xfd7zXZ\x00"


@dataclass
class Serialized:
    bytes: int
    exact: bool  # False: the end was extrapolated, or the stream could not be read
    compression: str  # gzip | bzip2 | xz | none


# R's own serialization starts with these (format, then a newline); an RData file with RDX.
_PLAIN = (b"X\n", b"A\n", b"B\n", b"RDX", b"RDA")
# Bytes one decompress call may return: a run of zeros can expand a lot.
_OUT = 64 * _READ


def _decompressor(head: bytes):
    if head.startswith(_GZIP):
        return "gzip", zlib.decompressobj(16 + zlib.MAX_WBITS)
    if head.startswith(_BZIP2):
        return "bzip2", bz2.BZ2Decompressor()
    if head.startswith(_XZ):
        return "xz", lzma.LZMADecompressor()
    return "none", None


def _feed(d, kind: str, data: bytes) -> int:
    """Bytes ``data`` decompresses to (all of it, up to the end of its stream)."""
    n = len(d.decompress(data, _OUT))
    if kind == "gzip":
        while d.unconsumed_tail and not d.eof:
            n += len(d.decompress(d.unconsumed_tail, _OUT))
    else:
        while not d.eof and not d.needs_input:
            n += len(d.decompress(b"", _OUT))
    return n


def serialized_size(path: str | Path, seconds: float = COUNT_SECONDS) -> Serialized:
    """The uncompressed size of an R file's serialized stream, read without keeping it.
    A file of several compressed streams one after another (gzip members) is read to
    its end; past ``seconds`` the rest is extrapolated from the share read."""
    path = Path(path)
    size = path.stat().st_size
    with open(path, "rb") as fh:
        head = fh.read(6)
        kind, d = _decompressor(head)
        if d is None:
            return Serialized(size, head.startswith(_PLAIN), kind)
        fh.seek(0)
        mode = kind
        out = fed = 0
        t0 = time.monotonic()
        while True:
            chunk = fh.read(_READ)
            if not chunk:
                return Serialized(out, True, kind)
            fed += len(chunk)
            while chunk:
                if d is None or d.eof:
                    mode, d = _decompressor(chunk[:6])
                    if d is None:  # trailing bytes after the last stream
                        return Serialized(out, True, kind)
                out += _feed(d, mode, chunk)
                chunk = d.unused_data if d.eof else b""
            if fed < size and time.monotonic() - t0 > seconds:
                return Serialized(int(out * size / fed), False, kind)


_CACHE: dict[tuple[str, int, int], Serialized] = {}


def serialized_size_cached(path: str | Path) -> Serialized:
    """``serialized_size``, once per file (path, size, mtime): the ingest preflight asks
    again whenever the staged files change."""
    st = os.stat(path)
    key = (str(path), st.st_size, st.st_mtime_ns)
    hit = _CACHE.get(key)
    if hit is None:
        hit = _CACHE[key] = serialized_size(path)
    return hit


def estimate_peak(serialized_bytes: int) -> int:
    return int(CONVERTER_BASE_BYTES + CONVERTER_PER_SERIALIZED_BYTE * serialized_bytes)


# ---------------------------------------------------------------------------
# The converter's memory (its GET /health)
# ---------------------------------------------------------------------------

@dataclass
class ConverterMemory:
    limit_bytes: int | None  # None: no limit set, or not readable
    used_bytes: int | None
    reachable: bool


def converter_memory(converter_url: str, transport: httpx.BaseTransport | None = None) -> ConverterMemory:
    """The converter's limit and use, from its /health. It answers nothing while it
    converts: then (and when it is down) the limit is unknown."""
    timeout = httpx.Timeout(connect=3.0, read=3.0, write=3.0, pool=3.0)
    try:
        with httpx.Client(timeout=timeout, transport=transport) as client:
            r = client.get(f"{converter_url}/health")
        data = r.json() if r.status_code == 200 else {}
    except (httpx.HTTPError, ValueError):
        return ConverterMemory(None, None, False)
    if not isinstance(data, dict):
        data = {}

    def num(k: str) -> int | None:
        v = data.get(k)
        return int(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None

    return ConverterMemory(num("memory_limit_bytes"), num("memory_used_bytes"), True)


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

def _gb(b: float) -> str:
    g = b / GiB
    return f"{g:.0f}" if g >= 10 else f"{g:.1f}"


def converter_verdict(path: str | Path, mem: ConverterMemory, *, hosted: bool = False) -> dict:
    """The converter's verdict for one R file, in the shape PreflightNote reads (as
    ``preflight.make_verdict``): level fits | tight | wont_fit | unknown and a sentence."""
    from scview.core import conversion, preflight

    name = Path(path).name
    ser = serialized_size_cached(path)
    need = estimate_peak(ser.bytes)
    secs = CONVERT_SECONDS + ser.bytes / GiB * CONVERT_SECONDS_PER_GB
    seconds = (round(secs * 0.5, 1), round(secs * 2.0, 1))
    avail = None
    if mem.limit_bytes is not None:
        avail = max(0, mem.limit_bytes - (mem.used_bytes or 0))
    level = preflight._level(need, avail) if avail is not None else "unknown"

    lead = f"Converting {name} takes about {_gb(need)} GB in the R converter"
    if level == "unknown":
        if not mem.reachable:
            budget = "The converter did not say how much memory it has (it is busy or not running)."
        else:
            budget = "No memory limit is set for the converter."
    else:
        budget = f"The converter has {_gb(avail)} GB free of its {_gb(mem.limit_bytes)} GB limit."
    sentence = f"{lead}, {preflight.time_text(seconds)}. {budget}"
    suggestions: list[str] = []
    if level in ("tight", "wont_fit") and mem.limit_bytes is not None:
        target = conversion.target_gb(need, mem.used_bytes or 0)
        suggestions.append(conversion.converter_memory_advice(mem.limit_bytes, target, hosted=hosted))
        suggestions.append("Or save a smaller object in R first: DietSeurat(obj, layers = \"counts\") "
                           "keeps the counts and drops the data and scale.data layers, which scView "
                           "does not read.")
    basis = ("Estimated from the size of the R object once decompressed (the converter reads all of it, "
             f"layers it does not write too): {_gb(ser.bytes)} GB"
             + ("" if ser.exact else ", extrapolated from the first part of the file")
             + f", plus about {_gb(CONVERTER_BASE_BYTES)} GB for R and Python. Measured on test objects, "
             f"it came within {round(ERROR_BAND * 100)}% of the converter's peak.")
    return {
        "level": level, "sentence": sentence, "need_bytes": need, "peak_bytes": need,
        "peak_stage": "reading the R object", "available_bytes": avail,
        "limit_bytes": mem.limit_bytes, "used_bytes": mem.used_bytes,
        "limited": mem.limit_bytes is not None, "source": "converter",
        "seconds": list(seconds),
        "cached": {"count": 0, "bytes": 0, "frees_enough": False},
        "suggestions": suggestions, "error_band": ERROR_BAND,
        "breakdown": [{"label": "The R object, decompressed", "bytes": ser.bytes},
                      {"label": "R, Python and anndata", "bytes": CONVERTER_BASE_BYTES}],
        "notes": [], "basis": basis,
        "serialized_bytes": ser.bytes, "serialized_exact": ser.exact, "compression": ser.compression,
    }


def verdict_for(path: str | Path, converter_url: str, *, hosted: bool = False,
                transport: httpx.BaseTransport | None = None) -> dict:
    """Asks the converter for its memory, then judges ``path``: blocking (a thread)."""
    return converter_verdict(path, converter_memory(converter_url, transport), hosted=hosted)
