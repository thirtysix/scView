"""Two runs whose next steps each fit alone but not together (``run_watchdog.room_for_step``).

Each run's pre-step check counts what the other is still expected to grow by
(``DatasetManager.outstanding``). Before, both stopped ("Run it again once that work has
finished"): nothing queued. Now the later run waits for the earlier one, sends a
``kind: "waiting"`` event naming it, and goes on once its reservation is released; a run
that started earlier never waits on a later one that is waiting, so two runs cannot wait on
each other; Cancel ends the wait at once, and the wait is capped (``QUEUE_WAIT_SECONDS``).
"""

from __future__ import annotations

import threading
import time

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp

from scview.core import memory_budget, preflight, processing, run_watchdog
from scview.core.run_watchdog import RunControl, RunWatchdog

GB = 1024**3


class _Avail:
    """10 GB limit, 2 GB used: 8 GB free."""

    limited, limit_bytes, available_bytes, used_bytes = True, 10 * GB, 8 * GB, 2 * GB


@pytest.fixture()
def budget(monkeypatch):
    monkeypatch.setattr(memory_budget, "read_budget", lambda *_a, **_k: memory_budget.MemoryBudget(
        10 * GB, 2 * GB, True, "container limit"))
    monkeypatch.setattr(memory_budget, "release_memory", lambda: None)
    monkeypatch.setattr(preflight, "available_memory", lambda *_a, **_k: _Avail())
    # Every step needs 4 GB (3.2 at the low end): fits in the 8 free, not beside 5 more.
    monkeypatch.setattr(preflight, "step_peak_for", lambda *_a, **_k: 4 * GB)
    monkeypatch.setattr(run_watchdog, "QUEUE_POLL_SECONDS", 0.02)
    monkeypatch.setattr(run_watchdog, "QUEUE_WAIT_SECONDS", 10.0)  # a test that hangs fails instead


def _register(dm, did: str) -> None:
    import json
    from pathlib import Path

    d = Path(dm.data_dir) / "uploads" / did
    d.mkdir(parents=True, exist_ok=True)
    (d / "metadata.json").write_text(json.dumps({"id": did, "name": f"{did} name", "status": "ready"}))


def _run(dm, did: str, need: float = 5.0) -> tuple[RunControl, list[dict]]:
    """A run on ``did`` that reserved ``need`` GB, wired to the manager as processing
    wires every run (``processing._queue``); its events collected."""
    _register(dm, did)
    events: list[dict] = []
    control = RunControl(RunWatchdog(read=lambda: run_watchdog.Memory(None, None, False)))
    control.watchdog.emit = events.append
    dm.reserve_run(did, int(need * GB))
    processing._queue(dm, did, control)
    return control, events


_ADATA = ad.AnnData(sp.csr_matrix(np.ones((3, 2), dtype=np.float32)))


def _check(control: RunControl, out: dict, key: str) -> threading.Thread:
    def go() -> None:
        out[key] = control.halt_before("pca", _ADATA, None)

    t = threading.Thread(target=go, daemon=True)
    t.start()
    return t


def _until(cond, seconds: float = 5.0) -> None:
    end = time.monotonic() + seconds
    while not cond():
        assert time.monotonic() < end, "timed out"
        time.sleep(0.01)


def test_the_later_run_waits_for_the_earlier_and_goes_on_when_it_ends(dataset_manager, budget):
    first, _ = _run(dataset_manager, "first")
    second, events = _run(dataset_manager, "second")
    out: dict = {}
    waiting = _check(second, out, "second")
    _until(lambda: events)
    assert waiting.is_alive() and dataset_manager.run_reservations()["second"].waiting
    [event] = events
    assert event["kind"] == "waiting" and event["step"] == "pca"
    assert event["waiting_for"] == ["the run on first name"]
    assert event["message"].startswith("Waiting before pca: it needs about 4.0 GB more at its peak, and of the "
                                       "8.0 GB free, the run on first name will still take about 5.0 GB.")
    assert "Cancel stops it without saving" in event["message"]
    # The earlier run leaves the waiting later one out: it goes on at once.
    assert first.halt_before("pca", _ADATA, None) is None
    dataset_manager.release_run("first")  # the earlier run ended
    waiting.join(5)
    assert not waiting.is_alive() and out["second"] is None
    assert not dataset_manager.run_reservations()["second"].waiting


def test_two_runs_waiting_at_once_do_not_wait_on_each_other(dataset_manager, budget):
    first, first_events = _run(dataset_manager, "first")
    second, second_events = _run(dataset_manager, "second")
    out: dict = {}
    a = _check(first, out, "first")  # the later one is not waiting yet: the earlier waits too
    _until(lambda: first_events)
    b = _check(second, out, "second")
    a.join(5)  # once the later one waits, the earlier one leaves it out and goes on
    assert not a.is_alive() and out["first"] is None
    assert b.is_alive() and second_events
    dataset_manager.release_run("first")
    b.join(5)
    assert out["second"] is None


def test_cancel_during_the_wait_stops_the_run_cleanly(dataset_manager, budget):
    _run(dataset_manager, "first")
    second, events = _run(dataset_manager, "second")
    out: dict = {}
    t = _check(second, out, "second")
    _until(lambda: events)
    second.cancel()
    t.join(5)
    assert not t.is_alive()
    status, why = out["second"]
    assert status == "cancelled" and why.startswith("Cancelled before pca: nothing was saved")
    assert not dataset_manager.run_reservations()["second"].waiting


def test_the_wait_is_capped_and_then_stops_as_before(dataset_manager, budget, monkeypatch):
    monkeypatch.setattr(run_watchdog, "QUEUE_WAIT_SECONDS", 1.1)
    _run(dataset_manager, "first")
    second, events = _run(dataset_manager, "second")
    status, why = second.halt_before("pca", _ADATA, None)
    assert status == "stopped" and "other work still running will take about 5.0 GB" in why
    assert "It waited 1 s for that work to finish first." in why and "Run it again" in why
    assert events and not dataset_manager.run_reservations()["second"].waiting


def test_a_step_that_cannot_fit_even_alone_stops_at_once(dataset_manager, budget, monkeypatch):
    monkeypatch.setattr(preflight, "step_peak_for", lambda *_a, **_k: 20 * GB)
    _run(dataset_manager, "first")
    second, events = _run(dataset_manager, "second")
    status, why = second.halt_before("pca", _ADATA, None)
    assert status == "stopped" and "Raise SCVIEW_MEMORY_GB" in why
    assert events == []


def test_a_run_alone_or_with_room_never_waits(dataset_manager, budget):
    alone, events = _run(dataset_manager, "alone")
    assert alone.halt_before("pca", _ADATA, None) is None and events == []
    dataset_manager.release_run("alone")
    _run(dataset_manager, "small", need=1.0)
    other, events = _run(dataset_manager, "other")  # 4 GB beside 1 more: fits
    assert other.halt_before("pca", _ADATA, None) is None and events == []


def test_loads_still_count_a_waiting_run(dataset_manager, budget):
    # Only the pre-step checks of runs that started earlier leave a waiting run out: a load
    # meanwhile counts its growth, which comes back once it goes on.
    _run(dataset_manager, "first", need=1.0)
    dataset_manager.reserve_run("second", 3 * GB)
    dataset_manager.set_waiting("second", True)
    assert dataset_manager.outstanding(2 * GB) == 4 * GB
    assert dataset_manager.outstanding(2 * GB, "first", pre_step=True) == 0
    assert dataset_manager.outstanding(2 * GB, "second", pre_step=True) == GB
