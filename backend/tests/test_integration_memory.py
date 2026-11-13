"""A joint embedding gets a pipeline run's memory protection (QA round 3: 8 Visium slides
OOM-killed the backend in the variable-gene step while other requests opened datasets).

The run's estimated growth is reserved until it ends (a load meanwhile is refused, naming
it), the merge and each step are checked before they start, a stop or Cancel ends with an
``error`` event and no half-made joint dataset, and the server goes on. The plan carries
the estimate and blocks a run that cannot fit even with every other dataset closed.
"""

from __future__ import annotations

import asyncio
import json
import threading
import weakref

import pytest

from scview.core import integration, memory_budget, preflight, processing, run_watchdog
from scview.core.dataset_manager import RunMemoryError
from scview.core.integration import estimate_joint, load_units, release_units, run_integration_streamed
from scview.core.run_watchdog import RunControl, RunWatchdog

from tests.test_integration import _add_dataset, _counts_dataset

GB = 1024**3


class _Avail:
    """10 GB limit, 2 GB used: 8 GB free."""

    limited, limit_bytes, available_bytes, used_bytes, source = True, 10 * GB, 8 * GB, 2 * GB, "container limit"


@pytest.fixture()
def budget(monkeypatch):
    monkeypatch.setattr(memory_budget, "read_budget", lambda *_a, **_k: memory_budget.MemoryBudget(
        10 * GB, 2 * GB, True, "container limit"))
    monkeypatch.setattr(memory_budget, "release_memory", lambda: None)
    monkeypatch.setattr(preflight, "available_memory", lambda *_a, **_k: _Avail())


def _two(data_dir) -> None:
    _add_dataset(data_dir, "a", "A", _counts_dataset(seed=1))
    _add_dataset(data_dir, "b", "B", _counts_dataset(seed=2, batch_shift=2.0))


def _events(text: str) -> list[tuple[str, dict]]:
    out, event = [], None
    for line in text.splitlines():
        if line.startswith("event: "):
            event = line[7:]
        elif line.startswith("data: ") and event:
            out.append((event, json.loads(line[6:])))
            event = None
    return out


_RUN = {"dataset_ids": ["a", "b"], "name": "AB", "params": {"n_top_genes": 100, "n_comps": 20}}


def _left_behind(dm, data_dir) -> list[str]:
    """Joint datasets still on disk or registered (anything but the two sources)."""
    on_disk = {p.name for sub in ("uploads", "ingested", "derived") if (data_dir / sub).exists()
               for p in (data_dir / sub).iterdir()}
    return sorted(on_disk - {"a", "b"})


@pytest.mark.asyncio
async def test_a_step_that_cannot_fit_stops_the_run_with_an_error_event(
        client, tmp_data_dir, dataset_manager, budget, monkeypatch):
    _two(tmp_data_dir)
    real = preflight.step_peak_for
    # The variable-gene step needs 20 GB (16 at the low end): more than the 8 GB free.
    monkeypatch.setattr(preflight, "step_peak_for", lambda a, step, p=None:
                        20 * GB if step == "highly_variable_genes" else real(a, step, p))
    r = await client.post("/api/v1/integrate/run-stream", json=_RUN)
    assert r.status_code == 200
    events = _events(r.text)
    kinds = [e for e, _ in events]
    assert "created" in kinds and "complete" not in kinds
    assert ("step_done", "normalization") in [(e, d.get("step")) for e, d in events]
    assert "highly_variable_genes" not in [d.get("step") for e, d in events if e == "step_start"]
    kind, data = events[-1]
    assert kind == "error" and data["status"] == "stopped"
    assert data["error"].startswith("The joint embedding did not finish. Stopped before highly "
                                    "variable genes: it needs about 20.0 GB more at its peak")
    assert "Nothing was kept: the joint dataset was removed, and the source datasets are unchanged." in data["error"]
    assert "integrate fewer or smaller datasets" in data["error"]
    # No half-made joint dataset, nothing held, and the server goes on.
    assert _left_behind(dataset_manager, tmp_data_dir) == []
    assert dataset_manager.run_reservations() == {} and not processing._runs
    joint_id = next(d["dataset_id"] for e, d in events if e == "created")
    assert not dataset_manager.is_busy(joint_id)
    assert (await client.get("/health")).status_code == 200
    monkeypatch.setattr(preflight, "step_peak_for", real)
    r = await client.post("/api/v1/integrate/run-stream", json=_RUN)
    assert _events(r.text)[-1][0] == "complete", r.text[-400:]


@pytest.mark.asyncio
async def test_a_merge_that_cannot_fit_stops_before_anything_is_written(
        client, tmp_data_dir, dataset_manager, budget, monkeypatch):
    _two(tmp_data_dir)
    real = run_watchdog.memory_stop
    monkeypatch.setattr(run_watchdog, "memory_stop", lambda step, need, **kw: real(
        step, 30 * GB if step == "merge" else need, **kw))
    r = await client.post("/api/v1/integrate/run-stream", json=_RUN)
    events = _events(r.text)
    assert "created" not in [e for e, _ in events]
    kind, data = events[-1]
    assert kind == "error" and "Stopped before merging the datasets: it needs about 30.0 GB" in data["error"]
    assert "Nothing was kept: the source datasets are unchanged." in data["error"]
    assert _left_behind(dataset_manager, tmp_data_dir) == []


@pytest.mark.asyncio
async def test_cancel_reaches_a_joint_run_and_removes_it(client, tmp_data_dir, dataset_manager, monkeypatch):
    _two(tmp_data_dir)
    real = preflight.step_peak_for

    def peak(a, step, p=None):
        if step == "scaling":  # the run is registered under its new id: Cancel it as the UI does
            [joint_id] = list(processing._runs)
            processing.cancel_run(joint_id)
        return real(a, step, p)

    monkeypatch.setattr(preflight, "step_peak_for", peak)
    # A container limit, so the check runs (and calls step_peak_for) before each step.
    monkeypatch.setattr(preflight, "available_memory", lambda *_a, **_k: _Avail())
    r = await client.post("/api/v1/integrate/run-stream", json=_RUN)
    kind, data = _events(r.text)[-1]
    assert kind == "error" and data["status"] == "cancelled"
    assert "Cancelled before scaling" in data["error"]
    assert "joint dataset was not kept" in data["error"] and "as it was before" not in data["error"]
    assert _left_behind(dataset_manager, tmp_data_dir) == []


@pytest.mark.asyncio
async def test_a_load_during_the_run_cannot_take_its_reserved_room(dataset_manager, budget):
    dm = dataset_manager
    gate, inside = threading.Event(), threading.Event()

    def make_events():
        inside.set()
        gate.wait(10)
        yield ("stage", {"stage": "merge"})

    control = RunControl(RunWatchdog(read=lambda: run_watchdog.Memory(None, None, False)))
    items = processing.iterate_new(dm, "joint1", make_events, control, name="joint: A + B",
                                   need=5 * GB, saved=lambda: False)
    first = asyncio.ensure_future(items.__anext__())
    await asyncio.to_thread(inside.wait, 10)
    r = dm.run_reservations()["joint1"]
    assert r.need == 5 * GB and r.name == "joint: A + B"
    assert dm.is_busy("joint1")
    # 5 GB more for another dataset fits in the 8 free alone, not beside the run's 5.
    with pytest.raises(RunMemoryError) as err:
        await dm.reserve("other", 5 * GB)
    assert str(err.value).startswith("A run on joint: A + B needs about 5.0 GB more at its peak")
    assert "or cancel it" in str(err.value)
    gate.set()
    assert (await first) == ("stage", {"stage": "merge"})
    with pytest.raises(StopAsyncIteration):
        await items.__anext__()
    assert dm.run_reservations() == {} and not dm.is_busy("joint1")
    await dm.reserve("other", 5 * GB)  # the room is back once the run has ended


@pytest.mark.asyncio
async def test_the_plan_gives_the_estimate_and_blocks_a_run_that_cannot_fit(
        client, tmp_data_dir, budget, monkeypatch):
    _two(tmp_data_dir)
    r = await client.post("/api/v1/integrate/plan", json={"dataset_ids": ["a", "b"]})
    plan = r.json()
    assert plan["ok"] is True
    v = plan["preflight"]
    assert v["level"] == "fits" and v["peak_bytes"] > 0 and v["limited"] is True
    assert v["sentence"].endswith("scView has 8.0 GB free of its 10 GB limit.")
    assert "360 cells x 120 genes" in v["notes"][0]
    # Ten thousand copies of the joint counts while merging: more than the limit holds.
    monkeypatch.setattr(integration, "MERGE_COPIES", 1e5)
    r = await client.post("/api/v1/integrate/plan", json={"dataset_ids": ["a", "b"]})
    plan = r.json()
    assert plan["ok"] is False and plan["preflight"]["level"] == "wont_fit"
    [why] = plan["blocking"]
    assert why.startswith("The joint run needs about ") and "(merging)" in why
    assert "even with every other dataset closed" in why
    r = await client.post("/api/v1/integrate/run-stream", json=_RUN)
    assert r.status_code == 400 and "even with every other dataset closed" in r.json()["detail"]


def test_the_estimate_follows_the_joint_shape(tmp_path):
    from scview.core.dataset_manager import DatasetManager

    dm = DatasetManager(str(tmp_path))
    _two(tmp_path)
    _, units = load_units(dm, ["a", "b"])
    est = estimate_joint(units, 120)
    h = est.header
    assert (h.n_cells, h.n_genes) == (360, 120)
    assert h.x.nnz == sum(u.adata.X.nnz for u in units.values())
    needs = est.step_needs()
    assert needs["merge"] == est.merge_bytes > 0
    assert {"highly_variable_genes", "pca", "batch_correction", "save"} <= set(needs)
    assert est.need == max(needs.values()) and est.peak_bytes == est.units_bytes + est.need
    # Half the genes shared: half of each unit's stored values kept.
    assert estimate_joint(units, 60).header.x.nnz == pytest.approx(h.x.nnz / 2, rel=0.01)


def test_the_units_counts_are_dropped_once_merged(tmp_path):
    from scview.core.dataset_manager import DatasetManager

    dm = DatasetManager(str(tmp_path))
    _two(tmp_path)
    _, units = load_units(dm, ["a", "b"])
    refs = [weakref.ref(u.adata) for u in units.values()]
    gen = run_integration_streamed(units, dataset_id="j", name="joint", data_dir=tmp_path,
                                   output_path_for=lambda src: dm.derived_h5ad_path("j", src),
                                   overrides={"n_top_genes": 100, "n_comps": 20})
    assert next(gen)[0] == "stage"  # merging
    assert next(gen)[1]["stage"] == "persist"
    assert all(u.adata is None for u in units.values())
    import gc

    gc.collect()
    assert all(r() is None for r in refs)
    assert [u.n_cells for u in units.values()] == [180, 180]
    events = list(gen)
    assert events[-1][0] == "pipeline_complete" and events[-1][1]["status"] == "complete"
    release_units(units)  # twice is harmless


def test_a_joint_unit_leaves_the_full_resolution_image_behind():
    """CELLxGENE Visium slides carry a 7,874 px full-resolution image scView never draws
    (186 MB each): the unit keeps hires and lowres only; the source keeps all of them."""
    import numpy as np

    from scview.core import spatial
    from scview.core.integration import build_unit

    a = _counts_dataset()
    a.obsm["spatial"] = np.random.default_rng(0).uniform(0, 100, (a.n_obs, 2))
    images = {"hires": np.zeros((20, 20, 3), np.uint8), "lowres": np.zeros((5, 5, 3), np.uint8),
              "fullres": np.zeros((80, 80, 3), np.uint8)}
    a.uns["spatial"] = {"lib1": {"images": images, "scalefactors": {"tissue_hires_scalef": 0.2}},
                        "is_single": True}
    unit = build_unit(a, dataset_id="a", name="A", label="A", fallback_paths=[])
    entry = unit.adata.uns[spatial.KEY]["lib1"]
    assert sorted(entry["images"]) == ["hires", "lowres"]
    assert entry["scalefactors"] == {"tissue_hires_scalef": 0.2}
    assert unit.adata.uns[spatial.KEY]["is_single"] is True
    assert "fullres" in a.uns["spatial"]["lib1"]["images"]  # the source is not changed
