"""Memory advice on a hosted site (DEPLOYMENT_MODE=public) names the site admin: its users
cannot edit .env or restart scView. On the hub the preflight note and the unclean-restart
notice said "raise SCVIEW_MEMORY_GB in .env"."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from scview.api.v1 import export_rds
from scview.config import get_settings
from scview.core import integration, memory_budget, preflight, run_watchdog
from scview.core.dataset_manager import DatasetMemoryError

GB = 1024**3
API = "/api/v1"


def _mode(monkeypatch, mode: str) -> None:
    monkeypatch.setattr(get_settings(), "DEPLOYMENT_MODE", mode)  # the settings are cached


@pytest.fixture
def public(monkeypatch):
    _mode(monkeypatch, "public")


@pytest.fixture
def private(monkeypatch):
    _mode(monkeypatch, "private")


def test_the_helper_in_both_modes(private, monkeypatch):
    assert memory_budget.hosted() is False
    assert memory_budget.more_memory_advice() == "Raise SCVIEW_MEMORY_GB in .env (then restart scView)."
    assert memory_budget.more_memory_advice("leave this step out") == (
        "Raise SCVIEW_MEMORY_GB in .env (then restart scView), or leave this step out.")
    _mode(monkeypatch, "public")
    assert memory_budget.hosted() is True
    assert memory_budget.more_memory_advice(limit_bytes=6 * GB) == (
        "Ask the site admin for more memory (this site has 6.0 GB).")
    assert memory_budget.more_memory_advice("leave this step out", limit_bytes=12 * GB, target_gb=14) == (
        "Ask the site admin for more memory (this site has 12 GB; this needs at least 14 GB), "
        "or leave this step out.")
    assert memory_budget.more_memory_advice(limit_bytes=3 * GB, service="the R converter") == (
        "Ask the site admin to give the R converter more memory (it has 3.0 GB).")


def test_the_budget_limit_is_named_when_none_is_given(public, monkeypatch):
    monkeypatch.setattr(memory_budget, "read_budget",
                        lambda *_a, **_k: memory_budget.MemoryBudget(8 * GB, GB, True, "container limit"))
    assert memory_budget.more_memory_advice() == "Ask the site admin for more memory (this site has 8.0 GB)."


@pytest.mark.parametrize("mode", ["private", "public"])
def test_each_message_follows_the_mode(mode, monkeypatch):
    _mode(monkeypatch, mode)
    hosted = mode == "public"
    texts = {
        "507": str(DatasetMemoryError("x", 5 * GB, 2 * GB, 6 * GB)),
        "step": run_watchdog.raise_limit(6 * GB),
        "joint": integration.joint_fix(),
        "rds": export_rds._too_big({"needs_bytes": 9 * GB, "limit_bytes": 3 * GB}),
        "preflight": preflight._raise_limit(preflight.Available(6 * GB, 4 * GB, 2 * GB, True, "container limit"),
                                            3 * GB),
    }
    for name, text in texts.items():
        assert ("site admin" in text) is hosted, (name, text)
        assert (".env" in text or "SCVIEW_MEMORY_GB" in text) is (not hosted), (name, text)
    if hosted:
        assert texts["507"].endswith("Ask the site admin for more memory (this site has 6.0 GB).")
        assert texts["step"].endswith(", or leave this step out.")
        assert "this needs at least 9 GB" in texts["preflight"]
        assert "give the R converter more memory (it has 3.0 GB), or subset the dataset first" in texts["rds"]


@pytest.mark.parametrize("mode", ["private", "public"])
def test_the_verdict_says_the_mode_and_words_closing_datasets(mode, monkeypatch):
    _mode(monkeypatch, mode)
    v = preflight.make_verdict(
        need=5 * GB, peak=6 * GB, stage="pca", seconds=(10.0, 30.0),
        avail=preflight.Available(6 * GB, 4 * GB, 2 * GB, True, "container limit"),
        cached=preflight.Cached(count=2, bytes=GB), lead="About 5 GB", breakdown=[], notes=[],
        alternatives=[])
    assert v["hosted"] is (mode == "public")
    text = " ".join(v["suggestions"])
    if mode == "public":
        assert "Ask the site admin to restart scView, then open only this one" in text
        assert ".env" not in text and "SCVIEW_MEMORY_GB" not in text
    else:
        assert "SCVIEW_MEMORY_GB" in text and "Close the 2 other open datasets" in text


@pytest.mark.parametrize("mode", ["private", "public"])
async def test_the_system_route_says_whether_the_site_is_hosted(client, mode, monkeypatch):
    _mode(monkeypatch, mode)
    r = await client.get(f"{API}/system/memory")
    assert r.status_code == 200
    assert r.json()["hosted"] is (mode == "public")


def test_no_memory_advice_outside_the_helper():
    """Every "raise SCVIEW_MEMORY_GB" a user can read goes through ``more_memory_advice``
    (preflight's and the converter's detailed local sentences aside, which they guard with
    ``hosted``)."""
    src = Path(memory_budget.__file__).resolve().parents[1]
    allowed = {"core/memory_budget.py", "core/preflight.py", "core/conversion.py"}
    found = []
    for path in src.rglob("*.py"):
        rel = path.relative_to(src).as_posix()
        if rel in allowed:
            continue
        for i, line in enumerate(path.read_text().splitlines(), 1):
            code = line.split("#", 1)[0]
            if re.search(r"[\"'].*Raise SCVIEW_MEMORY_GB", code):
                found.append(f"{rel}:{i}")
    assert not found, found
