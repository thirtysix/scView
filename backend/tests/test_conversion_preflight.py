"""The converter preflight: an .rds / .RData conversion's memory, estimated before it
starts from the decompressed size of the R file, against the converter's own memory
(its /health); shown on the upload and in the ingest preflight."""

from __future__ import annotations

import bz2
import gzip
import lzma
from pathlib import Path

import httpx
import pytest

from scview.core import conversion, conversion_preflight

GiB = 1024 ** 3
URL = "http://conv:8001"


# ---------------------------------------------------------------------------
# Converter preflight: the decompressed size, and the verdict
# ---------------------------------------------------------------------------

PAYLOAD = b"X\n" + bytes(range(256)) * 20000 + b"\x00" * 3_000_000

# Measured conversions (v0.83.0 converter, 8 GiB limit, 2 CPUs) of synthetic Seurat v5
# objects: serialized bytes -> the conversion's peak (VmHWM, GiB). A converter's /health
# (last_conversion_peak_bytes) and a converted dataset's report (conversion.memory) give
# the same pair for real objects; the estimate must stay at or above every one.
MEASURED = [
    ("counts + data + scale.data + graphs, 3 split layers", 601_456_838, 1.121),
    ("counts + data + scale.data + graphs, 12,000 cells", 1_715_598_813, 1.958),
    ("counts + data + scale.data + graphs, 23,578 cells", 3_369_723_301, 3.500),
    ("counts only, 20,000 cells", 1_345_873_342, 1.979),
    ("counts only, 5,000 cells", 337_317_754, 0.999),
]


@pytest.mark.parametrize("what,serialized,peak_gib", MEASURED)
def test_the_estimate_is_never_under_a_measured_peak(what, serialized, peak_gib):
    est = conversion_preflight.estimate_peak(serialized)
    assert est >= peak_gib * GiB, what
    assert est <= (1 + conversion_preflight.ERROR_BAND) * peak_gib * GiB, what


def test_counts_only_objects_are_never_under_at_any_size():
    """A counts-only object needs about 0.7 GiB beyond its own size at every size measured
    (blocks of X on top of an object nothing of which is freed). So the estimate keeps at
    least that over the object from small to large: a factor above 1 with a smaller base
    comes out under on small ones, a factor below 1 on large ones."""
    beyond = max(peak_gib * GiB - serialized for what, serialized, peak_gib in MEASURED
                 if what.startswith("counts only"))
    for gib in (0.1, 0.3, 1, 2, 4, 8, 16):
        size = int(gib * GiB)
        assert conversion_preflight.estimate_peak(size) - size >= beyond, gib


@pytest.mark.parametrize("kind,write", [
    ("gzip", gzip.compress), ("bzip2", bz2.compress), ("xz", lzma.compress), ("none", lambda b: b),
])
def test_serialized_size_is_the_decompressed_length(tmp_path, kind, write):
    p = tmp_path / "obj.rds"
    p.write_bytes(write(PAYLOAD))
    s = conversion_preflight.serialized_size(p)
    assert (s.bytes, s.exact, s.compression) == (len(PAYLOAD), True, kind)


def test_serialized_size_follows_several_gzip_members(tmp_path):
    p = tmp_path / "obj.rds"
    p.write_bytes(gzip.compress(PAYLOAD) + gzip.compress(PAYLOAD))
    assert conversion_preflight.serialized_size(p).bytes == 2 * len(PAYLOAD)


def test_serialized_size_extrapolates_past_its_time(tmp_path, monkeypatch):
    monkeypatch.setattr(conversion_preflight, "_READ", 64 * 1024)
    rnd = bytes((i * 7919 + (i >> 8) * 31) % 256 for i in range(2_000_000))  # compresses poorly
    p = tmp_path / "obj.rds"
    p.write_bytes(gzip.compress(b"X\n" + rnd * 3))
    s = conversion_preflight.serialized_size(p, seconds=0)
    assert not s.exact
    assert s.bytes == pytest.approx(len(rnd) * 3 + 2, rel=0.3)


def test_an_unknown_start_is_taken_as_its_size_but_not_exact(tmp_path):
    p = tmp_path / "obj.rds"
    p.write_bytes(b"\x28\xb5\x2f\xfd" + bytes(100))  # zstd, which this does not read
    s = conversion_preflight.serialized_size(p)
    assert s.bytes == 104 and not s.exact


def _rds(tmp_path: Path, n_bytes: int) -> Path:
    p = tmp_path / "object.rds"
    p.write_bytes(gzip.compress(b"X\n" + bytes(n_bytes - 2)))
    return p


def test_verdict_wont_fit_names_the_need_and_what_to_do(tmp_path):
    p = _rds(tmp_path, 40 * 1024 ** 2)
    need = conversion_preflight.estimate_peak(40 * 1024 ** 2)
    mem = conversion_preflight.ConverterMemory(limit_bytes=int(0.4 * GiB), used_bytes=int(0.1 * GiB),
                                               reachable=True)
    v = conversion_preflight.converter_verdict(p, mem)
    assert v["level"] == "wont_fit" and v["need_bytes"] == need
    assert v["sentence"].startswith("Converting object.rds takes about 0.8 GB in the R converter")
    assert "The converter has 0.3 GB free of its 0.4 GB limit." in v["sentence"]
    assert any("SCVIEW_MEMORY_GB" in s for s in v["suggestions"])
    assert any(s.startswith("Or, if it has data or scale.data layers, save a smaller object in R first: "
                            "DietSeurat(") for s in v["suggestions"])
    assert v["source"] == "converter" and "decompressed" in v["basis"]


def test_verdict_fits_with_room(tmp_path):
    p = _rds(tmp_path, 1024 ** 2)
    mem = conversion_preflight.ConverterMemory(limit_bytes=8 * GiB, used_bytes=GiB // 10, reachable=True)
    v = conversion_preflight.converter_verdict(p, mem)
    assert v["level"] == "fits" and v["suggestions"] == []


def test_verdict_unknown_when_the_converter_is_busy_or_unlimited(tmp_path):
    p = _rds(tmp_path, 1024 ** 2)
    busy = conversion_preflight.converter_verdict(p, conversion_preflight.ConverterMemory(None, None, False))
    assert busy["level"] == "unknown" and "busy or not running" in busy["sentence"]
    free = conversion_preflight.converter_verdict(p, conversion_preflight.ConverterMemory(None, 10, True))
    assert free["level"] == "unknown" and "No memory limit" in free["sentence"]


def test_converter_memory_reads_its_health():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/health"
        return httpx.Response(200, json={"status": "ok", "memory_limit_bytes": 3221225472,
                                         "memory_used_bytes": 150000000, "oom_kills": 0})

    m = conversion_preflight.converter_memory(URL, httpx.MockTransport(handler))
    assert (m.limit_bytes, m.used_bytes, m.reachable) == (3221225472, 150000000, True)


def test_converter_memory_of_an_old_or_silent_converter_is_unknown():
    old = conversion_preflight.converter_memory(
        URL, httpx.MockTransport(lambda r: httpx.Response(200, json={"status": "ok"})))
    assert (old.limit_bytes, old.reachable) == (None, True)

    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("busy", request=request)

    assert not conversion_preflight.converter_memory(URL, httpx.MockTransport(down)).reachable


def test_the_size_is_read_once_per_file(tmp_path, monkeypatch):
    p = _rds(tmp_path, 4096)
    calls = []
    real = conversion_preflight.serialized_size
    monkeypatch.setattr(conversion_preflight, "serialized_size", lambda path: calls.append(path) or real(path))
    conversion_preflight.serialized_size_cached(p)
    conversion_preflight.serialized_size_cached(p)
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# The counts in the stream: what the converted file holds once open
# ---------------------------------------------------------------------------

MiB = 1024 ** 2
DGC = b"\x00\x00\x00\x09dgCMatrix"  # a CHARSXP class name, as R's XDR stream writes it
ASSAY = b"\x00\x00\x00\x05Assay"
ASSAY5 = b"\x00\x00\x00\x06Assay5"


def _stream(*parts) -> bytes:
    """An R-like stream: ints are filler bytes, bytes go in as they are."""
    return b"X\n" + b"".join(bytes(p) if isinstance(p, int) else p for p in parts)


def _ser(tmp_path, data: bytes, name="obj.rds"):
    p = tmp_path / name
    p.write_bytes(gzip.compress(data))
    return conversion_preflight.serialized_size(p)


def test_a_v4_objects_counts_are_its_first_matrix(tmp_path):
    """Seurat v3 / v4: counts, then data, then scale.data; the class name ends each."""
    data = _stream(3 * MiB, DGC, 3 * MiB, DGC, 5 * MiB, 100, ASSAY, 1000)
    s = _ser(tmp_path, data)
    assert (s.seurat_assay, s.exact) == ("Assay", True)
    assert s.first_matrix_bytes == 3 * MiB + 2
    assert conversion_preflight.counts_bytes(s) == 3 * MiB + 2


def test_an_empty_counts_matrix_is_passed_over(tmp_path):
    """An assay made from data alone: an empty counts matrix, then the data (X)."""
    data = _stream(200, DGC, 2 * MiB, DGC, ASSAY)
    s = _ser(tmp_path, data)
    assert s.first_matrix_bytes == 2 * MiB


def test_a_v5_object_counts_as_all_counts(tmp_path):
    """Seurat v5 can split its counts in layers (the first is one sample): the whole object."""
    data = _stream(2 * MiB, DGC, 2 * MiB, DGC, ASSAY5, 100)
    s = _ser(tmp_path, data)
    assert s.seurat_assay == "Assay5"
    assert conversion_preflight.counts_bytes(s) == s.bytes == len(data)


def test_no_seurat_assay_or_no_end_counts_as_all(tmp_path, monkeypatch):
    sce = _ser(tmp_path, _stream(2 * MiB, DGC, 2 * MiB, DGC), "sce.rds")
    assert sce.seurat_assay == "" and conversion_preflight.counts_bytes(sce) == sce.bytes
    monkeypatch.setattr(conversion_preflight, "_READ", 64 * 1024)
    rnd = bytes((i * 7919 + (i >> 8) * 31) % 256 for i in range(2_000_000))
    p = tmp_path / "big.rds"
    p.write_bytes(gzip.compress(_stream(rnd, DGC, rnd, rnd, ASSAY)))
    cut = conversion_preflight.serialized_size(p, seconds=0)  # read in part: an Assay5 may follow
    assert not cut.exact and conversion_preflight.counts_bytes(cut) == cut.bytes


def test_a_class_name_split_between_two_pieces_is_found():
    m = conversion_preflight._Markers()
    piece = bytes(2 * MiB) + DGC + bytes(10) + ASSAY
    for cut in (2 * MiB + 5, 2 * MiB + 11):
        m = conversion_preflight._Markers()
        m.feed(piece[:cut])
        m.feed(piece[cut:])
        assert m.result() == {"seurat_assay": "Assay", "first_matrix_bytes": 2 * MiB}


# ---------------------------------------------------------------------------
# Where it shows: the upload's metadata, and the ingest preflight
# ---------------------------------------------------------------------------

async def test_upload_keeps_the_converter_estimate_while_it_converts(client, monkeypatch):
    seen = {}

    def memory(url, transport=None):
        seen["url"] = url
        return conversion_preflight.ConverterMemory(3 * GiB, GiB // 10, True)

    async def refuse(rds_path, output_path, converter_url, progress_path=None, transport=None):
        raise conversion.ConversionError("no")

    monkeypatch.setattr(conversion_preflight, "converter_memory", memory)
    monkeypatch.setattr(conversion, "trigger_conversion", refuse)
    files = {"file": ("obj.rds", gzip.compress(b"X\n" + bytes(1000)), "application/octet-stream")}
    did = (await client.post("/api/v1/datasets/upload", files=files)).json()["id"]
    task = conversion.task_for(did)
    if task is not None:
        await task
    info = (await client.get(f"/api/v1/datasets/{did}")).json()
    v = info["conversion_preflight"]
    assert v["level"] == "fits" and v["source"] == "converter" and v["serialized_bytes"] == 1002
    assert seen["url"].endswith(":8001")


async def test_ingest_preflight_adds_the_converter_for_an_unconverted_rds(client, tmp_data_dir, monkeypatch):
    from scview.core.ingestion.session import IngestSessionManager
    from scview.dependencies import get_ingest_session_manager
    from scview.main import app

    monkeypatch.setattr(conversion_preflight, "converter_memory",
                        lambda url, transport=None: conversion_preflight.ConverterMemory(2 * GiB, 0, True))
    mgr = IngestSessionManager(str(tmp_data_dir), converter_url=URL)
    app.dependency_overrides[get_ingest_session_manager] = lambda: mgr
    try:
        sid = mgr.create_session()
        (mgr.files_dir(sid) / "obj.rds").write_bytes(gzip.compress(b"X\n" + bytes(5000)))
        body = (await client.get(f"/api/v1/ingest/session/{sid}/preflight")).json()
    finally:
        app.dependency_overrides.pop(get_ingest_session_manager, None)
    unit = body["units"][0]
    assert unit["format"] == "seurat"
    assert unit["converter"]["source"] == "converter" and unit["converter"]["level"] == "fits"
    assert unit["converter"]["sentence"].startswith("Converting obj.rds takes about 0.8 GB in the R converter")
