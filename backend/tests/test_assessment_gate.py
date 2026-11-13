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
