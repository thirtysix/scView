"""Data Assessment's heavy work per dataset (core/assessment_gate.py): two pages opening one
dataset at once ran every heavy endpoint twice and stopped a 6 GB backend. Identical requests
now share one computation, at most two run per dataset at a time, other datasets and the event
loop are not held up, and a client that goes away does not cancel the work for the others."""

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
import scipy.sparse as sp

from scview.core import assessment_gate, cluster_qc

API = "/api/v1"


async def test_identical_requests_share_one_computation():
    calls = []
    release = threading.Event()

    def heavy(x):
        calls.append(x)
        release.wait(5)
        return x * 2

    first = asyncio.create_task(assessment_gate.run("d1", ("k", 1), heavy, 21))
    await asyncio.sleep(0.05)
    second = asyncio.create_task(assessment_gate.run("d1", ("k", 1), heavy, 21))
    await asyncio.sleep(0.05)
    assert assessment_gate.in_flight("d1") == 1
    release.set()
    assert await first == await second == 42
    assert calls == [21]  # the second waited for the first and got its result
    assert assessment_gate.in_flight("d1") == 0
    assert await assessment_gate.run("d1", ("k", 1), heavy, 21) == 42 and calls == [21, 21]  # after: runs again


async def test_at_most_two_per_dataset_and_other_datasets_are_not_held_up():
    running, most = [], [0]
    lock = threading.Lock()
    release = threading.Event()

    def heavy(name):
        with lock:
            running.append(name)
            most[0] = max(most[0], len(running))
        release.wait(5)
        with lock:
            running.remove(name)
        return name

    tasks = [asyncio.create_task(assessment_gate.run("big", ("k", i), heavy, f"big{i}")) for i in range(5)]
    await asyncio.sleep(0.2)
    assert len(running) == assessment_gate.MAX_AT_ONCE == 2
    # Another dataset runs at once, and the event loop answers meanwhile.
    t0 = time.monotonic()
    other = asyncio.create_task(assessment_gate.run("small", ("k",), lambda: "small"))
    assert await asyncio.wait_for(other, 2) == "small"
    assert time.monotonic() - t0 < 1.0
    # Work outside the lanes (the assessment itself) does not queue behind them.
    quick = asyncio.create_task(assessment_gate.run("big", ("assessment",), lambda: "checklist", lane=False))
    assert await asyncio.wait_for(quick, 2) == "checklist"
    release.set()
    assert sorted(await asyncio.gather(*tasks)) == [f"big{i}" for i in range(5)]
    assert most[0] == 2


async def test_errors_reach_every_waiter_and_a_cancelled_client_does_not_cancel_the_work():
    release = threading.Event()

    def boom():
        release.wait(5)
        raise ValueError("bad column")

    a = asyncio.create_task(assessment_gate.run("d2", "k", boom))
    await asyncio.sleep(0.05)
    b = asyncio.create_task(assessment_gate.run("d2", "k", boom))
    await asyncio.sleep(0.05)
    a.cancel()  # the first client went away
    release.set()
    with pytest.raises(ValueError, match="bad column"):
        await b
    with pytest.raises(asyncio.CancelledError):
        await a


def _register(dm, a: ad.AnnData, did: str) -> str:
    d = Path(dm.data_dir) / "uploads" / did
    d.mkdir(parents=True)
    (d / "metadata.json").write_text(json.dumps({"id": did, "name": did, "status": "ready"}))
    a.write_h5ad(d / "data.h5ad")
    return did


def _dataset(n: int = 400) -> ad.AnnData:
    rng = np.random.default_rng(0)
    X = sp.csr_matrix(rng.poisson(1.0, size=(n, 60)).astype(np.float32))
    obs = pd.DataFrame({"leiden": pd.Categorical(rng.choice(["0", "1", "2"], n))}, index=[f"c{i}" for i in range(n)])
    a = ad.AnnData(X, obs=obs, var=pd.DataFrame(index=[f"G{i}" for i in range(59)] + ["MT-CO1"]))
    return a


async def test_two_pages_at_once_compute_cluster_qc_once(client, dataset_manager, monkeypatch):
    did = _register(dataset_manager, _dataset(), "two_pages")
    real = cluster_qc.for_adaptor
    calls = []

    def counted(adaptor, groupby):
        calls.append(groupby)
        time.sleep(0.3)  # long enough for the second page to arrive meanwhile
        return real(adaptor, groupby)

    monkeypatch.setattr(cluster_qc, "for_adaptor", counted)
    url = f"{API}/datasets/{did}/cluster-qc"
    r1, r2 = await asyncio.gather(client.get(url), client.get(url))
    assert r1.status_code == r2.status_code == 200
    assert r1.json() == r2.json() and len(calls) == 1


async def test_work_waiting_for_a_lane_keeps_the_backed_file_open(tmp_path, monkeypatch):
    # Seen live (1.66 GB file, 3 GB backend, 1 run in 6): the mean-variance card took the backed
    # object and waited for a lane; an edit read the dataset in and then dropped it, so the card
    # pinned nothing when it started; another card's unpin closed the file under its read
    # (h5py: invalid identifier, a 500). The gate pins the dataset before any wait.
    from scview.core import anndata_adaptor
    from scview.core.anndata_adaptor import AnnDataAdaptor

    monkeypatch.setattr(anndata_adaptor, "BACKED_MIN_BYTES", 0)
    a = ad.AnnData(sp.random(50, 20, density=0.3, format="csr", dtype=np.float32, random_state=0))
    a.write_h5ad(tmp_path / "d.h5ad")
    adaptor = AnnDataAdaptor(str(tmp_path / "d.h5ad"))
    backed = adaptor.adata
    assert backed.isbacked
    cached = [adaptor]

    class DM:
        def get_dataset(self, _id):
            return cached[0]

        def dataset_name(self, _id):
            return "d"

    budget = assessment_gate.Budget(DM(), lambda: 0, "The card")
    go, started, mid = threading.Event(), threading.Event(), threading.Event()
    blockers = [asyncio.ensure_future(assessment_gate.run("d", ("block", i), go.wait))
                for i in range(assessment_gate.MAX_AT_ONCE)]
    await asyncio.sleep(0.05)

    def read(x):
        counts = x.X  # the on-disk matrix, read a block at a time
        started.set()
        mid.wait(10)
        return float(counts[np.arange(5)].sum())

    card = asyncio.ensure_future(assessment_gate.run("d", "card", read, backed, budget=budget))
    await asyncio.sleep(0.05)  # waiting for a lane
    other = adaptor.pin()  # another card, reading
    await asyncio.to_thread(adaptor.to_memory)  # the edit's read-in: the backed object is let go
    cached[0] = None  # and its reload: the dataset is dropped
    go.set()
    await asyncio.gather(*blockers)
    assert await asyncio.to_thread(started.wait, 10)
    other()  # the other card ends while this one reads
    mid.set()
    assert await card == pytest.approx(float(a.X[:5].sum()))
    assert adaptor._pins == 0 and not adaptor._retired  # the file is closed once it is done
