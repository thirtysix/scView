"""Stopping a pipeline run between steps (core/run_watchdog.py RunControl): the
pre-step hook ``before_step`` and Cancel (``POST .../assessment/cancel``).

Steps change the cached AnnData in place and the derived file is written only
after the last step, so a cancelled run saves nothing and the dataset reopens
as it was; a run the pre-step check stops keeps (and saves) the steps it did.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from scview.core import leiden, pipeline, run_watchdog
from scview.core.dataset_manager import DatasetManager
from scview.core.processing import active_run, cancel_run, iterate_exclusive, run_exclusive
from scview.core.run_watchdog import Memory, RunControl, RunWatchdog

API = "/api/v1"
STEPS = ["qc_metrics", "normalization", "log_transform", "pca"]


def _quiet() -> RunWatchdog:
    return RunWatchdog(read=lambda: Memory(1 << 30, 8 << 30, True), interval=0.01)


def _counts(n: int = 120, g: int = 40) -> ad.AnnData:
    a = ad.AnnData(np.random.default_rng(0).poisson(1.0, (n, g)).astype("float32"))
    a.var_names = [f"G{i}" for i in range(g)]
    a.obs_names = [f"c{i}" for i in range(n)]
    a.obs["grp"] = pd.Categorical(np.where(np.arange(n) < n // 2, "a", "b"))
    return a


def _register(data_dir: Path, did: str = "ds") -> str:
    d = data_dir / "uploads" / did
    d.mkdir(parents=True, exist_ok=True)
    (d / "metadata.json").write_text(json.dumps({"id": did, "name": did, "status": "ready"}))
    _counts().write_h5ad(d / "data.h5ad")
    return did


def _state(adata: ad.AnnData) -> tuple:
    """What a cancelled run must leave as it was: shape, obs columns, uns keys, X."""
    X = adata.X.toarray() if sp.issparse(adata.X) else np.asarray(adata.X)
    return adata.shape, list(adata.obs.columns), sorted(adata.uns.keys()), X.tobytes()


# -- before_step --------------------------------------------------------------


def test_before_step_is_called_with_step_adata_and_params_in_order():
    calls = []

    def hook(step, adata, params):
        calls.append((step, adata.shape, params.n_comps))
        return None

    params = pipeline.PipelineParams(n_comps=7)
    _, result = pipeline.run_pipeline(_counts(), STEPS, params,
                                      control=RunControl(_quiet(), before_step=hook))
    assert [c[0] for c in calls] == STEPS
    assert all(c[1] == (120, 40) and c[2] == 7 for c in calls)
    assert result.status == "complete" and result.stop_reason is None


def test_a_message_stops_the_run_before_that_step_and_saves_nothing(tmp_path):
    msg = "PCA needs about 9.0 GB and 2.1 GB is free."
    out = tmp_path / "out.h5ad"

    def hook(step, _adata, _params):
        return msg if step == "pca" else None

    _, result = pipeline.run_pipeline(_counts(), STEPS, output_path=str(out),
                                      control=RunControl(_quiet(), before_step=hook))
    assert result.status == "stopped"
    assert result.stop_reason == msg
    assert result.steps_run == ["qc_metrics", "normalization", "log_transform"]
    # A reprocess stopped halfway must not replace the dataset with a half-made one.
    assert not out.exists()


def test_the_streamed_run_stops_without_starting_the_step(tmp_path):
    def hook(step, _adata, _params):
        return "No room for PCA." if step == "pca" else None

    events = list(pipeline.run_pipeline_streamed(
        _counts(), STEPS, control=RunControl(_quiet(), before_step=hook)))
    starts = [d["step"] for k, d in events if k == "step_start"]
    assert starts == ["qc_metrics", "normalization", "log_transform"]
    kind, data = events[-1]
    assert kind == "complete"
    assert data["status"] == "stopped" and data["stop_reason"] == "No room for PCA."
    assert data["steps_run"] == starts


def test_stopped_before_the_first_step_saves_nothing(tmp_path):
    out = tmp_path / "out.h5ad"
    _, result = pipeline.run_pipeline(_counts(), STEPS, output_path=str(out),
                                      control=RunControl(_quiet(), before_step=lambda *_: "No."))
    assert result.status == "stopped" and result.steps_run == []
    assert not out.exists()


def test_a_failing_check_is_logged_and_the_step_runs(caplog):
    def broken(*_a):
        raise KeyError("estimate")

    _, result = pipeline.run_pipeline(_counts(), ["qc_metrics"],
                                      control=RunControl(_quiet(), before_step=broken))
    assert result.status == "complete" and result.steps_run == ["qc_metrics"]
    assert "pre-step check failed" in caplog.text


def _free(monkeypatch, *, limited=True, free_gb=1.0, limit_gb=4.0):
    from scview.core import preflight
    monkeypatch.setattr(preflight, "available_memory", lambda *a, **k: preflight.Available(
        int(limit_gb * run_watchdog.GB), int((limit_gb - free_gb) * run_watchdog.GB),
        int(free_gb * run_watchdog.GB), limited, "container limit" if limited else "machine"))


def test_the_default_check_stops_before_a_step_that_cannot_fit(monkeypatch):
    from scview.core import preflight
    _free(monkeypatch, free_gb=1.0)
    need = {"doublet_detection": 2.0, "pca": 1.1}
    monkeypatch.setattr(preflight, "step_peak_for",
                        lambda adata, step, params=None: int(need.get(step, 0) * run_watchdog.GB))
    msg = run_watchdog.default_before_step("doublet_detection", None, None)
    assert msg.startswith("Stopped before doublet detection: it needs about 2.0 GB")
    assert "1.0 GB free of its 4.0 GB limit" in msg and "SCVIEW_MEMORY_GB" in msg
    # 1.1 GB against 1.0 free is inside the estimate's error band: let it run.
    assert run_watchdog.default_before_step("pca", None, None) is None
    # Without a container limit the "limit" is the whole machine: no stop.
    _free(monkeypatch, limited=False, free_gb=1.0)
    assert run_watchdog.default_before_step("doublet_detection", None, None) is None


def test_the_default_check_stops_a_real_run(monkeypatch):
    _free(monkeypatch, free_gb=0.0)  # a tiny test dataset needs well under 1 MB per step
    _, result = pipeline.run_pipeline(_counts(), STEPS, control=RunControl(_quiet()))
    assert result.status == "stopped" and result.stop_reason.startswith("Stopped before ")
    _free(monkeypatch, free_gb=64.0, limit_gb=128.0)
    _, result = pipeline.run_pipeline(_counts(), STEPS, control=RunControl(_quiet()))
    assert result.status == "complete"


def test_the_default_check_is_where_preflight_plugs_in(monkeypatch):
    monkeypatch.setattr(run_watchdog, "default_before_step",
                        lambda step, *_: "stop" if step == "normalization" else None)
    _, result = pipeline.run_pipeline(_counts(), STEPS, control=RunControl(_quiet()))
    assert result.status == "stopped" and result.steps_run == ["qc_metrics"]
    # None switches the check off.
    _, result = pipeline.run_pipeline(_counts(), STEPS, control=RunControl(_quiet(), before_step=None))
    assert result.status == "complete"


# -- cancel: the pipeline loop ----------------------------------------------------


def _held(monkeypatch, step: str, started: threading.Event, go: threading.Event):
    real = pipeline._STEP_RUNNERS[step]

    def held(adata, params):
        started.set()
        go.wait(10)
        return real(adata, params)

    monkeypatch.setitem(pipeline._STEP_RUNNERS, step, held)


def test_cancel_between_steps_saves_nothing(tmp_path, monkeypatch):
    started, go = threading.Event(), threading.Event()
    _held(monkeypatch, "normalization", started, go)
    out = tmp_path / "out.h5ad"
    control = RunControl(_quiet())
    box = {}

    t = threading.Thread(target=lambda: box.update(r=pipeline.run_pipeline(
        _counts(), STEPS, output_path=str(out), control=control)[1]))
    t.start()
    assert started.wait(10)
    control.cancel()
    go.set()
    t.join(20)
    result = box["r"]
    assert result.status == "cancelled"
    assert result.steps_run == ["qc_metrics", "normalization"]  # done in memory, not saved
    assert result.stop_reason.startswith("Cancelled before log transform: nothing was saved")
    assert result.errors == {}
    assert not out.exists()


def test_cancel_after_the_last_step_skips_the_save(tmp_path, monkeypatch):
    started, go = threading.Event(), threading.Event()
    _held(monkeypatch, "qc_metrics", started, go)
    out = tmp_path / "out.h5ad"
    control = RunControl(_quiet())
    box = {}
    t = threading.Thread(target=lambda: box.update(r=pipeline.run_pipeline(
        _counts(), ["qc_metrics"], output_path=str(out), control=control)[1]))
    t.start()
    assert started.wait(10)
    control.cancel()
    go.set()
    t.join(20)
    assert box["r"].status == "cancelled"
    assert "before saving" in box["r"].stop_reason
    assert not out.exists()


def test_cancel_stops_marker_genes_between_its_substeps(monkeypatch):
    control = RunControl(_quiet())
    seen = []

    def fake_markers(adata, params, cb):
        for i in range(5):
            cb(f"Markers for 'c{i}'", i, 5)
            seen.append(i)
            if i == 1:
                control.cancel()

    monkeypatch.setitem(pipeline._STEP_RUNNERS, "marker_genes", fake_markers)
    events = list(pipeline.run_pipeline_streamed(_counts(), ["qc_metrics", "marker_genes"],
                                                 control=control))
    assert seen == [0, 1]
    kind, data = events[-1]
    assert data["status"] == "cancelled"
    assert data["stop_reason"].startswith("Cancelled during marker genes")
    assert "marker_genes" not in data["errors"]
    assert not [k for k, _ in events if k == "step_error"]


class _BlockingChild:
    """A Leiden child that runs until terminated."""

    instances: list = []

    def __init__(self, *_a, **_k):
        self.returncode = None
        self.ended = threading.Event()
        _BlockingChild.instances.append(self)

    def communicate(self, _input=None):
        self.ended.wait(10)
        return b"", b"Terminated"

    def terminate(self):
        self.returncode = -15
        self.ended.set()


def _with_graph() -> ad.AnnData:
    a = _counts()
    rng = np.random.default_rng(1)
    g = sp.random(120, 120, density=0.05, random_state=rng, format="csr")
    a.obsp["connectivities"] = g + g.T
    return a


def test_cancel_ends_a_running_leiden_child(monkeypatch):
    _BlockingChild.instances = []
    monkeypatch.setattr(leiden.subprocess, "Popen", _BlockingChild)
    control = RunControl(_quiet())
    box = {}
    t = threading.Thread(target=lambda: box.update(r=pipeline.run_pipeline(
        _with_graph(), ["clustering", "marker_genes"], control=control)[1]))
    t.start()
    for _ in range(200):
        if _BlockingChild.instances:
            break
        time.sleep(0.01)
    assert _BlockingChild.instances, "the Leiden child never started"
    t0 = time.perf_counter()
    control.cancel()
    t.join(10)
    assert time.perf_counter() - t0 < 5
    result = box["r"]
    assert result.status == "cancelled"
    assert result.stop_reason.startswith("Cancelled during clustering")
    assert result.steps_run == [] and result.errors == {}


def test_leiden_without_a_run_still_reports_a_failed_child(monkeypatch):
    class Failed(_BlockingChild):
        def communicate(self, _input=None):
            self.returncode = 1
            return b"", b"MemoryError"

    monkeypatch.setattr(leiden.subprocess, "Popen", Failed)
    with pytest.raises(leiden.LeidenError, match="exit 1"):
        leiden.membership(_with_graph().obsp["connectivities"], 1.0)


def test_cancel_terminates_a_real_child_process():
    control = RunControl(_quiet())
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    with control.child(proc):
        control.cancel()
        assert proc.wait(5) != 0
    # A child registered after the cancel is ended at once.
    proc2 = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    with control.child(proc2):
        assert proc2.wait(5) != 0


# -- cancel: core/processing.py and the endpoint -------------------------------------


async def _wait_for(cond, seconds: float = 10.0):
    for _ in range(int(seconds / 0.01)):
        if cond():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("timed out")


async def test_cancel_run_reaches_the_run_and_the_dataset_reopens_as_before(tmp_path, monkeypatch):
    did = _register(tmp_path)
    dm = DatasetManager(str(tmp_path))
    before = _state((await dm.get_or_load_dataset(did)).adata)
    started, go = threading.Event(), threading.Event()
    _held(monkeypatch, "log_transform", started, go)
    control = RunControl(_quiet())

    def work(adaptor):
        out = str(dm.derived_h5ad_path(did, adaptor.h5ad_path))
        return pipeline.run_pipeline(adaptor.adata, STEPS, output_path=out, control=control)[1]

    task = asyncio.ensure_future(run_exclusive(dm, did, work, control))
    await _wait_for(started.is_set)
    assert active_run(did) is control
    assert cancel_run(did) is control
    go.set()
    result = await task
    assert result.status == "cancelled"
    assert result.steps_run == ["qc_metrics", "normalization", "log_transform"]
    assert active_run(did) is None and cancel_run(did) is None
    assert not (tmp_path / "derived" / did / "data.h5ad").exists()
    assert _state((await dm.get_or_load_dataset(did)).adata) == before


async def test_cancel_endpoint_404_when_no_run(client, dataset_manager):
    did = _register(Path(dataset_manager.data_dir))
    r = await client.post(f"{API}/datasets/{did}/assessment/cancel")
    assert r.status_code == 404
    assert "No pipeline run" in r.json()["detail"]


async def test_cancel_endpoint_stops_a_streamed_run(client, dataset_manager, monkeypatch):
    did = _register(Path(dataset_manager.data_dir))
    before = _state((await dataset_manager.get_or_load_dataset(did)).adata)
    monkeypatch.setattr(run_watchdog, "read_memory", lambda: Memory(1 << 30, 8 << 30, True))
    started, go = threading.Event(), threading.Event()
    _held(monkeypatch, "normalization", started, go)

    async def run():
        return await client.post(f"{API}/datasets/{did}/assessment/run-stream",
                                 json={"steps": STEPS})

    task = asyncio.ensure_future(run())
    await _wait_for(started.is_set)
    r = await client.post(f"{API}/datasets/{did}/assessment/cancel")
    assert r.status_code == 200
    assert r.json() == {"cancelling": True, "step": "normalization"}
    go.set()
    resp = await task
    complete = [b for b in resp.text.split("\n\n") if b.startswith("event: complete")][0]
    result = json.loads(complete.split("data: ", 1)[1])
    assert result["status"] == "cancelled"
    assert result["steps_run"] == ["qc_metrics", "normalization"]
    assert "step_start" in resp.text and '"step": "log_transform"' not in resp.text
    assert (await client.post(f"{API}/datasets/{did}/assessment/cancel")).status_code == 404
    after = await dataset_manager.get_or_load_dataset(did)
    assert _state(after.adata) == before
    assert list((Path(dataset_manager.data_dir) / "derived" / did).glob("*.h5ad")) == []


async def test_cancel_endpoint_stops_a_plain_run(client, dataset_manager, monkeypatch):
    did = _register(Path(dataset_manager.data_dir))
    before = _state((await dataset_manager.get_or_load_dataset(did)).adata)
    started, go = threading.Event(), threading.Event()
    _held(monkeypatch, "normalization", started, go)

    task = asyncio.ensure_future(client.post(f"{API}/datasets/{did}/assessment/run",
                                             json={"steps": STEPS}))
    await _wait_for(started.is_set)
    assert (await client.post(f"{API}/datasets/{did}/assessment/cancel")).status_code == 200
    go.set()
    r = await task
    assert r.status_code == 200
    assert r.json()["status"] == "cancelled"
    assert _state((await dataset_manager.get_or_load_dataset(did)).adata) == before
    assert list((Path(dataset_manager.data_dir) / "derived" / did).glob("*.h5ad")) == []


async def test_a_client_that_goes_away_cancels_the_run(tmp_path, monkeypatch):
    _BlockingChild.instances = []
    monkeypatch.setattr(leiden.subprocess, "Popen", _BlockingChild)
    did = "g"
    d = tmp_path / "uploads" / did
    d.mkdir(parents=True)
    (d / "metadata.json").write_text(json.dumps({"id": did, "name": did, "status": "ready"}))
    _with_graph().write_h5ad(d / "data.h5ad")
    dm = DatasetManager(str(tmp_path))
    control = RunControl(_quiet())

    def events(adaptor):
        return pipeline.run_pipeline_streamed(adaptor.adata, ["clustering"], control=control)

    it = iterate_exclusive(dm, did, events, control)
    async for kind, _ in it:
        if kind == "step_start":
            break
    await _wait_for(lambda: bool(_BlockingChild.instances))
    await it.aclose()  # the client went away: the Leiden child is ended, not waited for
    assert control.cancelled
    assert _BlockingChild.instances[0].returncode == -15


async def test_a_run_with_a_plan_makes_room_first(tmp_path, monkeypatch):
    from scview.core import preflight, processing

    asked = []

    class FakeDM:
        async def free_memory_for(self, need, keep):
            asked.append((need, keep))

    monkeypatch.setattr(preflight, "run_need", lambda adata, steps, params: 12345)
    a = _counts()
    await processing._make_run_room(FakeDM(), "d1", type("A", (), {"adata": a})(),
                                    RunControl(_quiet(), plan=(STEPS, pipeline.PipelineParams())))
    await processing._make_run_room(FakeDM(), "d1", type("A", (), {"adata": a})(), RunControl(_quiet()))
    assert asked == [(12345, "d1")]  # the run without a plan asked nothing


@pytest.mark.parametrize("path, body", [
    ("assessment/run", {"steps": ["qc_metrics", "normalization"]}),
    ("rerun", None),
])
async def test_a_run_endpoint_holds_no_copy_through_the_reload(client, dataset_manager, monkeypatch, path, body):
    # The endpoint used to keep its own `adaptor` across run_exclusive, so the reload
    # read the file while the old AnnData was still alive: two copies at once (ovary
    # at 6 GB was OOM-killed reloading after a run the pre-step check had stopped).
    import gc
    import weakref

    did = _register(Path(dataset_manager.data_dir))
    old = weakref.ref((await dataset_manager.get_or_load_dataset(did)).adata)
    if body is None:  # rerun needs a recorded history: run the steps once first
        r = await client.post(f"{API}/datasets/{did}/assessment/run",
                              json={"steps": ["qc_metrics", "normalization", "log_transform", "pca", "neighbors",
                                              "embeddings"]})
        assert r.status_code == 200
        old = weakref.ref((await dataset_manager.get_or_load_dataset(did)).adata)
        body = {"edited_step": "embeddings", "params": {}}
    alive_at_reload = []
    load = dataset_manager.load_dataset

    async def watched(dataset_id, *a, **k):
        gc.collect()
        alive_at_reload.append(old() is not None)
        return await load(dataset_id, *a, **k)

    monkeypatch.setattr(dataset_manager, "load_dataset", watched)
    monkeypatch.setattr(run_watchdog, "default_before_step",
                        lambda step, *_: "No room." if step in ("normalization", "embeddings") else None)
    r = await client.post(f"{API}/datasets/{did}/{path}", json=body)
    assert r.status_code == 200, r.text
    assert alive_at_reload and alive_at_reload[-1] is False
