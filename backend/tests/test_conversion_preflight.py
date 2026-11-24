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
    assert v["sentence"].startswith("Converting object.rds takes about 0.5 GB in the R converter")
    assert "The converter has 0.3 GB free of its 0.4 GB limit." in v["sentence"]
    assert any("SCVIEW_MEMORY_GB" in s for s in v["suggestions"])
    assert any("DietSeurat" in s for s in v["suggestions"])
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
                        lambda url, transport=None: conversion_preflight.ConverterMemory(GiB, 0, True))
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
    assert unit["converter"]["sentence"].startswith("Converting obj.rds takes about 0.5 GB in the R converter")
