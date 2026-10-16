"""Heavy work runs off the event loop, and a pipeline run has its dataset to itself.

The production backend is one process. A pipeline step ran on the event loop, so
every request waited for the whole step: on Kang, /health timed out for the ~70 s
of doublet detection. Loading a file, on-demand markers, DE and enrichment each
stalled it for seconds too.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from scview.api.v1.enrichment import EnrichmentResponse
from scview.core import pipeline
from scview.core.dataset_manager import DatasetManager
from scview.core.obs_writes import dataset_lock
from scview.core.processing import iterate_exclusive, run_exclusive

MAIN = threading.main_thread()


def _dataset(data_dir: Path, did: str = "ds", n: int = 120, g: int = 40) -> None:
    d = data_dir / "uploads" / did
    d.mkdir(parents=True, exist_ok=True)
    (d / "metadata.json").write_text(json.dumps({"id": did, "name": did, "status": "ready"}))
    a = ad.AnnData(np.random.default_rng(0).poisson(1.0, (n, g)).astype("float32"))
    a.var_names = [f"G{i}" for i in range(g)]
    a.obs_names = [f"c{i}" for i in range(n)]
    a.obs["grp"] = pd.Categorical(np.where(np.arange(n) < n // 2, "a", "b"))
    a.write_h5ad(d / "data.h5ad")


async def _worst_stall(coro, interval: float = 0.01):
    """Await ``coro`` beside a 10 ms heartbeat; return its result and the longest
    the heartbeat was kept waiting beyond its interval (= time the loop was blocked)."""
    gaps: list[float] = []
    done = asyncio.Event()

    async def beat():
        last = time.perf_counter()
        while not done.is_set():
            await asyncio.sleep(interval)
            now = time.perf_counter()
            gaps.append(now - last - interval)
            last = now

    hb = asyncio.create_task(beat())
    try:
        result = await coro
    finally:
        done.set()
        await hb
    return result, max(gaps, default=0.0)


@pytest.fixture()
def slow_reads(monkeypatch):
    """Make reading an h5ad take 0.3 s and record which threads read."""
    readers: list[threading.Thread] = []
    real = ad.read_h5ad

    def slow(*args, **kwargs):
        readers.append(threading.current_thread())
        time.sleep(0.3)
        return real(*args, **kwargs)

    monkeypatch.setattr(ad, "read_h5ad", slow)
    return readers


# --- loading ------------------------------------------------------------------


async def test_concurrent_loads_share_one_read_off_the_loop(tmp_path, slow_reads):
    _dataset(tmp_path)
    dm = DatasetManager(str(tmp_path))

    async def three():
        return await asyncio.gather(*(dm.load_dataset("ds") for _ in range(3)))

    adaptors, stall = await _worst_stall(three())
    assert adaptors[0] is adaptors[1] is adaptors[2]
    assert len(slow_reads) == 1 and slow_reads[0] is not MAIN
    assert stall < 0.15
    assert dm.get_dataset("ds") is adaptors[0]


async def test_a_load_overtaken_by_an_evict_is_not_cached(tmp_path, slow_reads):
    _dataset(tmp_path)
    dm = DatasetManager(str(tmp_path))
    load = asyncio.create_task(dm.load_dataset("ds"))
    await asyncio.sleep(0.1)  # reading
    dm.evict("ds")  # e.g. a run rewrote the file meanwhile
    adaptor = await load
    assert adaptor is not None  # the waiting request still gets an answer
    assert dm.get_dataset("ds") is None  # but the possibly stale copy isn't kept


# --- pipeline runs ------------------------------------------------------------


async def test_a_run_leaves_the_loop_free_and_holds_its_dataset(tmp_path):
    _dataset(tmp_path)
    dm = DatasetManager(str(tmp_path))
    await dm.load_dataset("ds")
    threads: list[threading.Thread] = []

    def steps(adaptor):
        for i in range(3):
            threads.append(threading.current_thread())
            time.sleep(0.2)
            yield i

    seen: dict[str, object] = {}

    async def consume():
        out = []
        async for item in iterate_exclusive(dm, "ds", steps):
            if item == 0:
                seen["busy"] = dm.is_busy("ds")
                seen["locked"] = dataset_lock("ds").locked()
                seen["reader"] = asyncio.create_task(dm.get_or_load_dataset("ds"))
                await asyncio.sleep(0.05)
                seen["reader_waiting"] = not seen["reader"].done()
            out.append(item)
        return out

    items, stall = await _worst_stall(consume())
    assert items == [0, 1, 2]
    assert stall < 0.15
    assert all(t is not MAIN for t in threads)
    assert seen["busy"] and seen["locked"] and seen["reader_waiting"]
    # Afterwards: released, and the reader got the reloaded dataset.
    assert not dm.is_busy("ds") and not dataset_lock("ds").locked()
    assert await seen["reader"] is dm.get_dataset("ds") is not None


async def test_an_abandoned_run_stops_after_its_step_and_only_then_releases(tmp_path):
    _dataset(tmp_path)
    dm = DatasetManager(str(tmp_path))
    await dm.load_dataset("ds")
    second_done = threading.Event()
    third_started = threading.Event()

    def steps(adaptor):
        yield "first"
        time.sleep(0.3)
        second_done.set()
        yield "second"
        third_started.set()
        yield "third"

    events = iterate_exclusive(dm, "ds", steps)
    assert await events.__anext__() == "first"
    await events.aclose()  # the client went away during the second step

    assert dm.get_dataset("ds") is None  # nothing serves the half-run copy
    assert dm.is_busy("ds") and dataset_lock("ds").locked()  # the thread is mid-step
    await dm.wait_until_idle("ds")
    assert second_done.is_set() and not third_started.is_set()
    assert not dataset_lock("ds").locked()


async def test_a_cancelled_request_keeps_the_dataset_until_the_thread_returns(tmp_path):
    _dataset(tmp_path)
    dm = DatasetManager(str(tmp_path))
    finished = threading.Event()

    def work(adaptor):
        time.sleep(0.3)
        finished.set()
        return "ok"

    task = asyncio.create_task(run_exclusive(dm, "ds", work))
    await asyncio.sleep(0.1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert dm.is_busy("ds") and dataset_lock("ds").locked()
    await dm.wait_until_idle("ds")
    assert finished.is_set() and not dataset_lock("ds").locked()


async def test_streamed_run_does_not_stall_other_requests(client, dataset_manager, monkeypatch):
    _dataset(Path(dataset_manager.data_dir))
    real = pipeline._STEP_RUNNERS["normalization"]

    def slow_normalization(adata, params):
        time.sleep(0.6)
        return real(adata, params)

    monkeypatch.setitem(pipeline._STEP_RUNNERS, "normalization", slow_normalization)
    resp, stall = await _worst_stall(client.post(
        "/api/v1/datasets/ds/assessment/run-stream",
        json={"steps": ["normalization", "log_transform"]},
    ))
    assert "event: complete" in resp.text
    assert stall < 0.3  # the 0.6 s step did not hold the loop (it did: 0.6 s before the fix)


# --- request handlers -----------------------------------------------------------


async def test_on_demand_markers_compute_in_a_worker_thread(client, dataset_manager, monkeypatch):
    from scview.api.v1 import markers

    _dataset(Path(dataset_manager.data_dir))
    threads: list[threading.Thread] = []
    real = markers._compute_markers

    def spy(adata, column):
        threads.append(threading.current_thread())
        return real(adata, column)

    monkeypatch.setattr(markers, "_compute_markers", spy)
    first, second = await asyncio.gather(
        client.get("/api/v1/datasets/ds/markers", params={"groupby_column": "grp", "format": "json"}),
        client.get("/api/v1/datasets/ds/markers", params={"groupby_column": "grp", "format": "json"}),
    )
    assert first.status_code == second.status_code == 200
    assert first.json()["groups"] == ["a", "b"]
    assert len(threads) == 1 and threads[0] is not MAIN  # computed once, off the loop


async def test_de_runs_in_a_worker_thread(client, dataset_manager, monkeypatch):
    from scview.api.v1 import de

    _dataset(Path(dataset_manager.data_dir))
    threads: list[threading.Thread] = []
    real = de._wilcoxon

    def spy(*args, **kwargs):
        threads.append(threading.current_thread())
        return real(*args, **kwargs)

    monkeypatch.setattr(de, "_wilcoxon", spy)
    sel = await client.post("/api/v1/datasets/ds/de", json={"indices": list(range(60))})
    cmp_ = await client.post("/api/v1/datasets/ds/de/compare",
                             json={"split_by": "grp", "group_a": "a", "group_b": "b"})
    assert sel.status_code == cmp_.status_code == 200
    assert len(threads) == 2 and all(t is not MAIN for t in threads)


async def test_enrichment_computes_in_a_worker_thread(client, dataset_manager, monkeypatch):
    from scview.api.v1 import enrichment

    _dataset(Path(dataset_manager.data_dir))
    threads: list[threading.Thread] = []

    def stub(dataset_id, adaptor, body, settings):
        threads.append(threading.current_thread())
        return EnrichmentResponse(group=body.group, groupby=body.column, n_genes_used=0,
                                  results=[], source="stub")

    monkeypatch.setattr(enrichment, "_compute_local", stub)
    r = await client.post("/api/v1/datasets/ds/enrichment/compute-local",
                          json={"column": "grp", "group": "a"})
    assert r.status_code == 200 and r.json()["source"] == "stub"
    assert threads and threads[0] is not MAIN


async def test_dataset_info_answers_from_the_sidecar_while_a_run_works(client, dataset_manager):
    _dataset(Path(dataset_manager.data_dir))
    dataset_manager.mark_busy("ds")
    try:
        r = await asyncio.wait_for(client.get("/api/v1/datasets/ds"), timeout=2)
    finally:
        dataset_manager.mark_idle("ds")
    assert r.status_code == 200 and r.json()["id"] == "ds"
