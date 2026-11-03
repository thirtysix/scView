"""The memory opening a Discover dataset needs, before it is downloaded (batch 39 item A).

The file's header (shapes, stored values, what it holds) is read from Discover's file host
by HTTP ranges, so the preflight is the downloaded file's own: Liu 2021 (972 MB) and
Otero-Garcia 2022 (1.26 GB) in 11 and 13 requests, 0.3 s each, the same load plan as
their downloaded files. HTTP is mocked here (a file served by ranges).
"""

from __future__ import annotations

import io
import re

import httpx
import pytest

from scview.core import memory_budget, preflight, public_import as pi
from scview.core.dataset_manager import load_plan, plan_of_header
from tests.test_mcp import _call, msigdb_dir, test_settings  # noqa: F401 — fixtures
from tests.test_public_import import LIVER_B, LISTING, _events, discover_h5ad, fake  # noqa: F401

API = "/api/v1"
GB = 1024**3


@pytest.fixture(autouse=True)
def _no_cached_headers():
    pi._HEADERS.clear()
    yield
    pi._HEADERS.clear()


class Ranged:
    """A file host that serves byte ranges of ``body`` (or ignores them: ``ranges`` False)."""

    def __init__(self, body: bytes, ranges: bool = True) -> None:
        self.body, self.ranges = body, ranges
        self.ranged: list[str] = []
        self.whole = 0

    def __call__(self, request):
        rng = request.headers.get("range")
        if rng and self.ranges:
            a, b = (int(x) for x in re.fullmatch(r"bytes=(\d+)-(\d+)", rng).groups())
            self.ranged.append(rng)
            return httpx.Response(206, content=self.body[a:b + 1])
        self.whole += 1
        return httpx.Response(200, content=self.body)

    def client(self):
        return httpx.Client(transport=httpx.MockTransport(self))


def _rec(body: bytes) -> dict:
    return {"dataset_version_id": LIVER_B, "size_bytes": len(body)}


def test_the_remote_header_is_the_downloaded_file_s(tmp_path):
    src = tmp_path / "d.h5ad"
    discover_h5ad(src)
    body = src.read_bytes()
    host = Ranged(body)
    h = pi.remote_header(_rec(body), client=host.client())
    local = preflight.read_header(src, "h5ad")
    assert (h.n_cells, h.n_genes) == (local.n_cells, local.n_genes)
    assert [(m.where, m.nnz, m.nbytes) for m in h.matrices] == [(m.where, m.nnz, m.nbytes) for m in local.matrices]
    assert plan_of_header(h) == load_plan(str(src))
    assert host.whole == 0 and 0 < len(host.ranged) <= pi.HEADER_MAX_REQUESTS
    n = len(host.ranged)
    assert pi.remote_header(_rec(body), client=host.client()) is h and len(host.ranged) == n  # kept


def test_small_blocks_read_the_same_header(tmp_path, monkeypatch):
    src = tmp_path / "d.h5ad"
    discover_h5ad(src)
    body = src.read_bytes()
    with pi.RangeFile("https://x/f.h5ad", len(body), Ranged(body).client(), block=4096,
                      max_requests=10_000) as fh:
        h = preflight._read_h5ad(fh)
        assert fh.requests > 1
        fh.seek(-10, 2)
        assert fh.read(10) == body[-10:] and fh.tell() == len(body)
    assert h.n_cells == preflight.read_header(src, "h5ad").n_cells


def test_no_ranges_or_too_many_is_no_estimate_not_a_download(tmp_path, monkeypatch):
    src = tmp_path / "d.h5ad"
    discover_h5ad(src)
    body = src.read_bytes()
    host = Ranged(body, ranges=False)
    v = pi.memory_before_download(_rec(body), client=host.client())
    assert v["verdict"] == "unknown" and "checked once it is downloaded" in v["message"]
    monkeypatch.setattr(pi, "HEADER_MAX_REQUESTS", 0)
    v = pi.memory_before_download(_rec(body), client=Ranged(body).client())
    assert v["verdict"] == "unknown" and "spread too widely" in v["message"]
    assert pi.memory_before_download({"dataset_version_id": LIVER_B, "size_bytes": 0})["verdict"] == "unknown"


def _budget(monkeypatch, limit_gb: float) -> None:
    monkeypatch.setattr(memory_budget, "read_budget", lambda *a, **k: memory_budget.MemoryBudget(
        int(limit_gb * GB), int(0.1 * GB), True, "container limit"))


@pytest.fixture()
def ranged(fake, monkeypatch):  # noqa: F811
    """The Discover stand-in, its file host serving ranges."""
    host = Ranged(fake.body)
    orig = fake.handler

    def handler(request):
        if str(request.url).startswith(pi.FILES_BASE + "/") and request.headers.get("range"):
            return host(request)
        return orig(request)

    fake.handler = handler
    return host


async def test_the_detail_says_the_memory_and_a_file_that_cannot_open_is_not_downloaded(
        client, ranged, fake, monkeypatch):  # noqa: F811
    _budget(monkeypatch, 8)
    r = await client.get(f"{API}/public-import/discover/datasets/{LIVER_B}")
    mem = r.json()["memory"]
    assert mem["verdict"] == "fits" and mem["source"].startswith("the file's header")
    assert mem["n_cells"] == preflight._read_h5ad(io.BytesIO(fake.body)).n_cells and mem["in_memory_bytes"] > 0
    monkeypatch.setattr(pi, "plan_verdict", lambda plan: {"verdict": "wont_fit", "message": "Needs about 9.0 GB."})
    pi._HEADERS.clear()
    before = len(fake.urls)
    r = await client.post(f"{API}/public-import/discover/import-stream", json={"dataset_version_id": LIVER_B})
    assert r.status_code == 507, r.text
    assert r.json()["detail"] == "Needs about 9.0 GB. Nothing was downloaded."
    assert not any(u == pi.file_url(LIVER_B) for u in fake.urls[before:] if u) or ranged.whole == 0


async def test_an_import_says_the_memory_before_the_download(client, ranged, fake, monkeypatch):  # noqa: F811
    _budget(monkeypatch, 8)
    r = await client.post(f"{API}/public-import/discover/import-stream", json={"dataset_version_id": LIVER_B})
    events = _events(r.text)
    names = [e for e, _ in events]
    assert names[:2] == ["job", "memory"] and names[-1] == "complete", names
    assert events[1][1]["stage"] == "before download" and events[1][1]["verdict"] == "fits"


async def test_mcp_check_and_import_refusal(client, ranged, fake, monkeypatch):  # noqa: F811
    _budget(monkeypatch, 8)
    r = await _call("check_public_dataset", {"dataset_version_id": LIVER_B})
    assert r["memory"]["verdict"] == "fits" and r["memory"]["in_memory_gb"] >= 0 and r["disk"]["ok"]
    monkeypatch.setattr(pi, "plan_verdict", lambda plan: {"verdict": "wont_fit", "message": "Needs about 9.0 GB."})
    pi._HEADERS.clear()
    text = await _call("import_public_dataset", {"dataset_version_id": LIVER_B}, error=True)
    assert "Not downloaded: Needs about 9.0 GB." in text
    assert ranged.whole == 0
