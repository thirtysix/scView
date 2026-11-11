"""The sidebar's version and newer-release notice (core/version_check.py); no network."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from scview.core import version_check as V


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    V._reset()
    monkeypatch.setattr(V, "current_version", lambda: "0.69.1")
    yield
    V._reset()


def _client(tag: str | None = "v0.70.0", status: int = 200, calls: list | None = None) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        if calls is not None:
            calls.append(request)
        if status != 200:
            return httpx.Response(status)
        return httpx.Response(200, json={"tag_name": tag, "html_url": f"https://github.com/x/releases/tag/{tag}"})
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def test_versions_compare_as_numbers_and_pre_releases_never_count():
    assert V.is_newer("0.70.0", "0.69.1") and V.is_newer("v0.69.10", "0.69.9")
    assert not V.is_newer("0.69.1", "0.69.1") and not V.is_newer("0.68.9", "0.69.1")
    assert not V.is_newer("0.70.0-rc1", "0.69.1") and not V.is_newer("0.70.0", "unknown")


def test_a_newer_release_is_offered_with_its_page_and_the_request_names_nobody():
    calls: list = []
    out = asyncio.run(V.status(True, _client(calls=calls)))
    assert out == {"version": "0.69.1", "check": "on", "latest": "0.70.0", "update_available": True,
                   "release_url": "https://github.com/x/releases/tag/v0.70.0"}
    (req,) = calls
    assert str(req.url) == V.RELEASES_API and req.headers["User-Agent"] == "scView-update-check"


def test_the_same_version_is_not_an_update():
    out = asyncio.run(V.status(True, _client("v0.69.1")))
    assert out["latest"] == "0.69.1" and out["update_available"] is False and out["release_url"] is None


def test_off_sends_nothing():
    calls: list = []
    out = asyncio.run(V.status(False, _client(calls=calls)))
    assert calls == [] and out["check"] == "off" and out["latest"] is None


def test_github_is_asked_once_a_day_and_again_an_hour_after_a_failure():
    calls: list = []
    ok = _client(calls=calls)
    asyncio.run(V.latest_release(ok, now=0.0))
    asyncio.run(V.latest_release(ok, now=V.CHECK_EVERY - 1))
    assert len(calls) == 1
    asyncio.run(V.latest_release(ok, now=V.CHECK_EVERY))
    assert len(calls) == 2

    V._reset()
    calls.clear()
    bad = _client(status=403, calls=calls)  # rate-limited
    assert asyncio.run(V.latest_release(bad, now=0.0)) == {"latest": None, "url": None}
    asyncio.run(V.latest_release(bad, now=V.RETRY_AFTER_FAILURE - 1))
    assert len(calls) == 1
    asyncio.run(V.latest_release(bad, now=V.RETRY_AFTER_FAILURE))
    assert len(calls) == 2


def test_offline_keeps_the_last_good_answer():
    asyncio.run(V.latest_release(_client("v0.70.0"), now=0.0))
    out = asyncio.run(V.latest_release(_client(status=500), now=V.CHECK_EVERY))
    assert out["latest"] == "0.70.0"


def test_the_route_reports_the_version(monkeypatch):
    from fastapi.testclient import TestClient

    from scview.config import get_settings
    from scview.main import app

    monkeypatch.setenv("UPDATE_CHECK", "false")
    get_settings.cache_clear()
    try:
        body = TestClient(app).get("/api/v1/system/version").json()
    finally:
        get_settings.cache_clear()
    assert body["version"] == "0.69.1" and body["check"] == "off" and body["update_available"] is False
