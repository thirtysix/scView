"""The run watchdog (core/run_watchdog.py): memory warnings at 80/90% of the limit,
slow-step events, and the run's peak memory and step seconds recorded.

No real memory pressure here: the memory reader and the clock are fakes.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pytest

from scview.core import pipeline, provenance, run_watchdog
from scview.core.dataset_manager import DatasetManager
from scview.core.processing import active_run, iterate_exclusive, run_exclusive
from scview.core.run_watchdog import GB, Memory, RunControl, RunWatchdog, read_memory

API = "/api/v1"


class FakeMemory:
    """A memory reader whose use the test sets (in GB of an 8 GB limit)."""

    def __init__(self, used_gb: float = 1.0, limit_gb: float | None = 8.0, limited: bool = True):
        self.used_gb, self.limit_gb, self.limited = used_gb, limit_gb, limited

    def __call__(self) -> Memory:
        limit = None if self.limit_gb is None else int(self.limit_gb * GB)
        return Memory(int(self.used_gb * GB), limit, self.limited)


class FakeClock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def _watchdog(mem, clock=None, events=None):
    events = [] if events is None else events
    return RunWatchdog(events.append, read=mem, clock=clock or FakeClock(), slow_every=60), events


def _memory(events):
    return [e for e in events if e["kind"] == "memory"]


def test_memory_warnings_at_80_and_90_percent_once_each():
    mem = FakeMemory(4.0)
    wd, events = _watchdog(mem)
    wd.sample()
    assert events == []
    mem.used_gb = 6.5  # 81%
    wd.sample()
    wd.sample()  # still above 80%: no repeat
    mem.used_gb = 7.3  # 91%
    wd.sample()
    mem.used_gb = 7.5
    wd.sample()
    got = _memory(events)
    assert [e["threshold_pct"] for e in got] == [80, 90]
    assert got[0] == {"type": "watchdog", "kind": "memory", "threshold_pct": 80, "used_gb": 6.5,
                      "limit_gb": 8.0, "pct": 81, "limited": True, "step": None}
    assert got[1]["pct"] == 91 and got[1]["used_gb"] == 7.3


def test_a_threshold_fires_again_only_after_dropping_back_below():
    mem = FakeMemory(6.5)  # 81%
    wd, events = _watchdog(mem)
    wd.sample()
    mem.used_gb = 6.3  # 78.75%: below 80 but within the re-arm margin (2 points)
    wd.sample()
    mem.used_gb = 6.5
    wd.sample()
    assert len(_memory(events)) == 1
    mem.used_gb = 5.0  # well below
    wd.sample()
    mem.used_gb = 6.6
    wd.sample()
    assert [e["threshold_pct"] for e in _memory(events)] == [80, 80]


def test_a_jump_past_both_thresholds_sends_one_warning():
    mem = FakeMemory(2.0)
    wd, events = _watchdog(mem)
    wd.sample()
    mem.used_gb = 7.6  # 95%
    wd.sample()
    mem.used_gb = 7.1  # 88.75%: 90 not re-armed (needs < 88%), 80 still disarmed
    wd.sample()
    mem.used_gb = 7.4  # 92.5%
    wd.sample()
    assert [(e["threshold_pct"], e["pct"]) for e in _memory(events)] == [(90, 95)]


def test_no_limit_known_means_no_memory_warnings_but_the_peak_is_kept():
    mem = FakeMemory(7.9, limit_gb=None, limited=False)
    wd, events = _watchdog(mem)
    wd.sample()
    assert events == []
    assert wd.summary()["peak_gb"] == 7.9


def test_the_machine_limit_is_said_to_be_the_machine():
    mem = FakeMemory(14.6, limit_gb=16.0, limited=False)
    wd, events = _watchdog(mem)
    wd.sample()
    assert events[0]["limited"] is False and events[0]["pct"] == 91


def test_slow_step_event_every_60_seconds_of_one_step():
    clock = FakeClock()
    wd, events = _watchdog(FakeMemory(1.0), clock)
    wd.step_started("doublet_detection")
    for t in (30, 59.9, 60, 61, 119, 121, 150, 185):
        clock.t = 1000 + t
        wd.sample()
    slow = [e for e in events if e["kind"] == "slow_step"]
    assert slow == [
        {"type": "watchdog", "kind": "slow_step", "step": "doublet_detection", "elapsed_s": 60},
        {"type": "watchdog", "kind": "slow_step", "step": "doublet_detection", "elapsed_s": 121},
        {"type": "watchdog", "kind": "slow_step", "step": "doublet_detection", "elapsed_s": 185},
    ]
    # A delayed sample (the GIL held for minutes) sends one event, then the next minute.
    wd.step_finished("doublet_detection")
    wd.step_started("embeddings")
    clock.t += 250
    wd.sample()
    clock.t += 30  # 280 s: the next one is due at 300
    wd.sample()
    clock.t += 20
    wd.sample()
    later = [e["elapsed_s"] for e in events if e["kind"] == "slow_step" and e["step"] == "embeddings"]
    assert later == [250, 300]


def test_a_new_step_starts_its_own_clock():
    clock = FakeClock()
    wd, events = _watchdog(FakeMemory(1.0), clock)
    wd.step_started("pca")
    clock.t += 50
    wd.step_finished("pca")
    wd.step_started("neighbors")
    clock.t += 50
    wd.sample()
    assert not [e for e in events if e["kind"] == "slow_step"]


def test_step_seconds_and_peaks_are_recorded():
    clock = FakeClock()
    mem = FakeMemory(1.0)
    wd, _ = _watchdog(mem, clock)
    wd.sample()
    wd.step_started("pca")
    mem.used_gb = 3.0
    wd.sample()
    mem.used_gb = 2.0
    clock.t += 12.5
    wd.step_finished("pca")
    wd.step_started("neighbors")
    mem.used_gb = 2.5
    clock.t += 3
    wd.step_finished("neighbors")
    s = wd.summary()
    assert s["steps"] == {"pca": {"elapsed_s": 12.5, "peak_gb": 3.0},
                          "neighbors": {"elapsed_s": 3.0, "peak_gb": 2.5}}
    assert s["peak_gb"] == 3.0 and s["start_gb"] == 1.0 and s["limit_gb"] == 8.0


def test_a_failing_reader_or_emitter_never_raises():
    def broken() -> Memory:
        raise OSError("no cgroup")

    def emit(_e):
        raise RuntimeError("stream gone")

    wd = RunWatchdog(emit, read=broken, clock=FakeClock())
    wd.sample()
    wd2 = RunWatchdog(emit, read=FakeMemory(7.9), clock=FakeClock())
    wd2.sample()  # the warning is still recorded
    assert wd2.summary()["warnings"][0]["pct"] == 99


def test_the_thread_samples_until_stopped():
    mem = FakeMemory(1.0)
    wd = RunWatchdog(read=mem, interval=0.01)
    wd.start()
    time.sleep(0.15)
    assert wd.running
    wd.stop()
    assert not wd.running
    n = wd.n_samples
    assert n >= 3
    time.sleep(0.05)
    assert wd.n_samples == n


def test_read_memory_takes_the_inactive_file_cache_off(tmp_path):
    (tmp_path / "memory.max").write_text("8589934592\n")
    (tmp_path / "memory.current").write_text(str(6 * GB) + "\n")
    (tmp_path / "memory.stat").write_text(
        f"anon {4 * GB}\nfile {2 * GB}\nactive_file {GB // 2}\ninactive_file {int(1.5 * GB)}\n")
    m = read_memory(tmp_path)
    assert m == Memory(int(4.5 * GB), 8 * GB, True)
    (tmp_path / "memory.stat").unlink()
    assert read_memory(tmp_path).used == 6 * GB  # no stat file: the raw reading


# -- through the pipeline and core/processing.py ---------------------------------


def _counts(n: int = 120, g: int = 40) -> ad.AnnData:
    a = ad.AnnData(np.random.default_rng(0).poisson(1.0, (n, g)).astype("float32"))
    a.var_names = [f"G{i}" for i in range(g)]
    a.obs_names = [f"c{i}" for i in range(n)]
    return a


def _register(data_dir: Path, did: str = "ds") -> str:
    d = data_dir / "uploads" / did
    d.mkdir(parents=True, exist_ok=True)
    (d / "metadata.json").write_text(json.dumps({"id": did, "name": did, "status": "ready"}))
    _counts().write_h5ad(d / "data.h5ad")
    return did


def test_run_result_and_provenance_carry_step_seconds_and_peak(tmp_path):
    mem = FakeMemory(2.0)
    control = RunControl(RunWatchdog(read=mem))
    out = tmp_path / "out.h5ad"
    _, result = pipeline.run_pipeline(_counts(), ["qc_metrics", "normalization", "log_transform"],
                                      output_path=str(out), control=control)
    res = result.resources
    assert set(res["steps"]) == {"qc_metrics", "normalization", "log_transform"}
    assert all(v["elapsed_s"] >= 0 and v["peak_gb"] == 2.0 for v in res["steps"].values())
    assert res["peak_gb"] == 2.0 and res["limit_gb"] == 8.0 and res["write_s"] is not None
    saved = provenance.read_provenance(ad.read_h5ad(out))["history"]
    assert [h["step"] for h in saved] == ["qc_metrics", "normalization", "log_transform"]
    for h in saved:
        assert h["resources"]["peak_gb"] == 2.0
        assert h["resources"]["elapsed_s"] == res["steps"][h["step"]]["elapsed_s"]
    # Replay reads the params only: resources stay out of the recipe.
    assert all("resources" not in r for r in provenance.recipe(ad.read_h5ad(out)))


def test_a_run_without_control_records_no_resources():
    _, result = pipeline.run_pipeline(_counts(), ["qc_metrics"])
    assert result.resources is None
    assert "resources" not in provenance.read_provenance(_)["history"][0]


async def test_non_streamed_run_is_watched_without_events(tmp_path):
    did = _register(tmp_path)
    dm = DatasetManager(str(tmp_path))
    mem = FakeMemory(7.5)  # 94%: a warning is recorded, not sent anywhere
    control = RunControl(RunWatchdog(read=mem, interval=0.01))
    seen_active = []

    def work(adaptor):
        seen_active.append(active_run(did) is control)
        return pipeline.run_pipeline(adaptor.adata, ["qc_metrics"], control=control)[1]

    result = await run_exclusive(dm, did, work, control)
    assert seen_active == [True]
    assert active_run(did) is None
    assert not control.watchdog.running  # stopped with the run
    assert result.resources["warnings"][0]["threshold_pct"] == 90


async def test_streamed_run_yields_watchdog_events_and_stops_the_sampler(tmp_path, monkeypatch):
    did = _register(tmp_path)
    dm = DatasetManager(str(tmp_path))
    mem = FakeMemory(1.0)
    control = RunControl(RunWatchdog(read=mem, interval=0.01, slow_every=0.05))
    real_pca = pipeline._STEP_RUNNERS["pca"]
    release = threading.Event()

    def slow_pca(adata, params):
        mem.used_gb = 7.4  # 92.5% while PCA works
        release.wait(5)
        return real_pca(adata, params)

    monkeypatch.setitem(pipeline._STEP_RUNNERS, "pca", slow_pca)

    def events(adaptor):
        return pipeline.run_pipeline_streamed(adaptor.adata, ["normalization", "log_transform", "pca"],
                                              control=control)

    got = []
    async for kind, data in iterate_exclusive(dm, did, events, control):
        got.append((kind, data))
        if kind == "watchdog" and data["kind"] == "slow_step" and data["step"] == "pca":
            release.set()
    kinds = [k for k, _ in got]
    watch = [d for k, d in got if k == "watchdog"]
    assert any(d["kind"] == "memory" and d["threshold_pct"] == 90 and d["step"] == "pca" for d in watch)
    assert any(d["kind"] == "slow_step" and d["step"] == "pca" for d in watch)
    assert kinds[-1] == "complete"
    assert got[-1][1]["resources"]["peak_gb"] == 7.4
    assert not control.watchdog.running
    assert active_run(did) is None


async def test_the_sampler_stops_when_the_run_fails(tmp_path, monkeypatch):
    did = _register(tmp_path)
    dm = DatasetManager(str(tmp_path))
    control = RunControl(RunWatchdog(read=FakeMemory(1.0), interval=0.01))

    def work(_adaptor):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        await run_exclusive(dm, did, work, control)
    assert not control.watchdog.running
    assert active_run(did) is None


async def test_run_stream_endpoint_sends_watchdog_events(client, dataset_manager, monkeypatch):
    did = _register(Path(dataset_manager.data_dir))
    monkeypatch.setattr(run_watchdog, "read_memory", FakeMemory(7.4))
    r = await client.post(f"{API}/datasets/{did}/assessment/run-stream", json={"steps": ["qc_metrics"]})
    assert r.status_code == 200
    text = r.text
    assert "event: watchdog" in text
    blocks = [b for b in text.split("\n\n") if b.startswith("event: watchdog")]
    data = json.loads(blocks[0].split("data: ", 1)[1])
    assert data["kind"] == "memory" and data["pct"] == 92 and data["limit_gb"] == 8.0
    complete = [b for b in text.split("\n\n") if b.startswith("event: complete")][0]
    result = json.loads(complete.split("data: ", 1)[1])
    assert result["status"] == "complete"
    assert result["resources"]["steps"]["qc_metrics"]["peak_gb"] == 7.4
