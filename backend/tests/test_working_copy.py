"""Processing a dataset the adaptor opened backed (core/working_copy.py).

A file over ``SCVIEW_BACKED_MIN_MB`` (2 GiB by default) opens in anndata's backed
mode. The run endpoints handed that object to the pipeline: QC metrics failed at
once (``AxisError: axis 1 is out of bounds for array of dimension 0``) and every
later step but doublet detection failed with it. The threshold is set to 0 here, so
every test file opens backed.
"""

from __future__ import annotations

import gc
import json
import weakref
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from scview.core import anndata_adaptor, memory_budget, preflight, working_copy
from scview.core.anndata_adaptor import AnnDataAdaptor

API = "/api/v1"
GB = 1024**3
STEPS = ["qc_metrics", "filtering", "normalization", "log_transform",
         "highly_variable_genes", "scaling", "pca"]
PARAMS = {"qc_threshold_mode": "fixed", "min_genes": 1, "min_cells": 1, "max_pct_mt": 100.0,
          "n_top_genes": 30, "n_comps": 10}


def _counts(n: int = 300, g: int = 60) -> ad.AnnData:
    rng = np.random.default_rng(0)
    X = sp.csr_matrix(rng.poisson(rng.gamma(1.0, 1.0, g), (n, g)).astype(np.float32))
    a = ad.AnnData(X)
    a.var_names = [f"MT-{i}" if i < 3 else f"G{i}" for i in range(g)]
    a.obs_names = [f"c{i}" for i in range(n)]
    a.obs["grp"] = pd.Categorical(np.where(np.arange(n) % 3 == 0, "a", np.where(np.arange(n) % 3 == 1, "b", "c")))
    a.raw = a.copy()  # counts in .raw too: its X stays on disk as well
    return a


def _register(data_dir: Path, did: str = "big") -> str:
    d = data_dir / "uploads" / did
    d.mkdir(parents=True, exist_ok=True)
    (d / "metadata.json").write_text(json.dumps({"id": did, "name": did, "status": "ready"}))
    _counts().write_h5ad(d / "data.h5ad")
    return did


@pytest.fixture()
def backed(monkeypatch):
    """Every file opens backed."""
    monkeypatch.setattr(anndata_adaptor, "BACKED_MIN_BYTES", 0)


async def _is_backed(dm, did: str) -> bool:
    return bool((await dm.get_or_load_dataset(did)).adata.isbacked)


# -- the threshold ---------------------------------------------------------------


@pytest.mark.parametrize("value, expected", [
    ("", 2048 * 1024**2), ("100", 100 * 1024**2), ("0.5", 512 * 1024), ("lots", 2048 * 1024**2),
])
def test_the_backed_threshold_comes_from_the_environment(monkeypatch, value, expected):
    monkeypatch.setenv("SCVIEW_BACKED_MIN_MB", value)
    assert anndata_adaptor._backed_min_bytes() == expected


def test_the_header_and_the_adaptor_agree_on_backed(tmp_path, monkeypatch):
    p = tmp_path / "d.h5ad"
    _counts().write_h5ad(p)
    size = p.stat().st_size
    monkeypatch.setattr(anndata_adaptor, "BACKED_MIN_BYTES", size)  # not over: in memory
    assert not preflight.read_header(p).backed and not AnnDataAdaptor(str(p)).adata.isbacked
    monkeypatch.setattr(anndata_adaptor, "BACKED_MIN_BYTES", size - 1)
    h = preflight.read_header(p)
    assert h.backed and AnnDataAdaptor(str(p)).adata.isbacked
    # X and .raw's X stay on disk: they are what reading the dataset in adds.
    x, raw = h.matrix("X"), h.matrix("raw/X")
    assert preflight.backed_bytes(h) == x.nbytes + raw.nbytes
    h.backed = False
    assert preflight.backed_bytes(h) == 0
    assert preflight.resident_bytes(h) - x.nbytes - raw.nbytes == preflight.resident_bytes(
        preflight.read_header(p))


def test_a_backed_dataset_s_run_is_costed_as_read_in(tmp_path, backed):
    p = tmp_path / "d.h5ad"
    _counts().write_h5ad(p)
    on_disk = ad.read_h5ad(p, backed="r")
    in_memory = ad.read_h5ad(p)
    # header_from_adata would take the on-disk X for a dense matrix of every value.
    assert preflight.run_need(on_disk, STEPS) == preflight.run_need(in_memory, STEPS)


# -- the run endpoints -----------------------------------------------------------


@pytest.mark.parametrize("path", ["assessment/run", "assessment/run-stream"])
async def test_a_backed_dataset_runs_every_step(client, dataset_manager, backed, path):
    did = _register(Path(dataset_manager.data_dir))
    assert await _is_backed(dataset_manager, did)
    r = await client.post(f"{API}/datasets/{did}/{path}", json={"steps": STEPS, "params": PARAMS})
    assert r.status_code == 200, r.text
    if path.endswith("stream"):
        events = [line for line in r.text.splitlines() if line.startswith("data: ")]
        result = json.loads(events[-1][len("data: "):])
    else:
        result = r.json()
    assert result["errors"] == {} and result["steps_run"] == STEPS
    # Saved, and the result opens backed again (it is over the threshold too).
    adaptor = await dataset_manager.get_or_load_dataset(did)
    assert adaptor.adata.isbacked and "X_pca" in adaptor.adata.obsm


async def test_a_run_lets_go_of_the_backed_object_and_its_copy_before_the_reload(
        client, dataset_manager, backed, monkeypatch):
    # The copy read into memory replaces the backed object in the adaptor; the run's
    # reload must find both gone (lesson 2026-10-22: a frame still holding the old
    # object kept two copies through the reload).
    did = _register(Path(dataset_manager.data_dir))

    async def refs():
        a = (await dataset_manager.get_or_load_dataset(did)).adata
        return weakref.ref(a), a.file._file

    old, h5 = await refs()
    assert old().isbacked and h5.id.valid
    copies = []
    to_memory = AnnDataAdaptor.to_memory

    def spy(self):
        out = to_memory(self)
        copies.append(weakref.ref(out))
        return out

    monkeypatch.setattr(AnnDataAdaptor, "to_memory", spy)
    at_reload = []
    load = dataset_manager.load_dataset

    async def watched(dataset_id, *a, **k):
        gc.collect()
        at_reload.append((old() is not None, any(c() is not None for c in copies), bool(h5.id.valid)))
        return await load(dataset_id, *a, **k)

    monkeypatch.setattr(dataset_manager, "load_dataset", watched)
    r = await client.post(f"{API}/datasets/{did}/assessment/run", json={"steps": STEPS, "params": PARAMS})
    assert r.status_code == 200 and r.json()["errors"] == {}, r.text
    assert copies, "the run never read the dataset into memory"
    # The endpoint's own lookups come first (the backed object is still the dataset
    # then); the reload after the run is the last call.
    assert at_reload[0] == (True, False, True)
    assert at_reload[-1] == (False, False, False)


async def test_a_copy_that_cannot_fit_is_refused_with_507_and_frees_the_dataset(
        client, dataset_manager, backed, monkeypatch):
    did = _register(Path(dataset_manager.data_dir))
    assert await _is_backed(dataset_manager, did)
    monkeypatch.setattr(memory_budget, "read_budget",
                        lambda *a, **k: memory_budget.MemoryBudget(8 * GB, 1 * GB, True, "container limit"))
    monkeypatch.setattr(memory_budget, "release_memory", lambda: None)
    monkeypatch.setattr("scview.core.dataset_manager.REFUSAL_WAIT_SECONDS", 0)
    monkeypatch.setattr(preflight, "backed_bytes", lambda h: 100 * GB)
    # In memory by choice: by default ("auto") such a run goes to large-dataset mode
    # (tests/test_large_run.py).
    off = {**PARAMS, "chunked": "off"}
    r = await client.post(f"{API}/datasets/{did}/assessment/run", json={"steps": STEPS, "params": off})
    assert r.status_code == 507, r.text
    assert "Reading this dataset into memory needs about 100.0 GB" in r.json()["detail"]
    # Nothing ran, and the dataset is neither locked nor busy: the next run works.
    monkeypatch.setattr(preflight, "backed_bytes", lambda h: 1)
    r = await client.post(f"{API}/datasets/{did}/assessment/run", json={"steps": STEPS, "params": PARAMS})
    assert r.status_code == 200 and r.json()["errors"] == {}, r.text


async def test_make_room_asks_for_the_copy_and_the_run_together(monkeypatch):
    asked = []

    class FakeDM:
        async def reserve(self, did, need, extra=0, what=""):
            asked.append(("reserve", need, extra, what))

        async def free_memory_for(self, need, keep):
            asked.append(("free", need, keep))

    class FakeAdaptor:
        def __init__(self, on_disk: bool):
            self.adata = type("A", (), {"isbacked": on_disk})()
            self.h5ad_path = "x.h5ad"

    monkeypatch.setattr(preflight, "read_header", lambda path, fmt=None: "header")
    monkeypatch.setattr(preflight, "backed_bytes", lambda h: 500)
    assert await working_copy.make_room(FakeDM(), "d", FakeAdaptor(True), then=70) is True
    assert await working_copy.make_room(FakeDM(), "d", FakeAdaptor(False), then=70) is False
    assert await working_copy.make_room(FakeDM(), "d", FakeAdaptor(False)) is False
    assert asked == [("reserve", 500, 70, "Reading this dataset into memory"), ("free", 70, "d")]


# -- the endpoints that edit and save --------------------------------------------


async def test_an_edit_of_a_backed_dataset_is_saved_and_it_reopens_backed(client, dataset_manager, backed):
    did = _register(Path(dataset_manager.data_dir))
    assert await _is_backed(dataset_manager, did)
    r = await client.post(f"{API}/datasets/{did}/metadata/grp/order", json={"order": ["c", "b", "a"]})
    assert r.status_code == 200, r.text
    adaptor = await dataset_manager.get_or_load_dataset(did)
    assert adaptor.adata.isbacked and list(adaptor.adata.obs["grp"].cat.categories) == ["c", "b", "a"]


async def test_markers_on_a_backed_dataset_are_computed_in_memory_and_kept(client, dataset_manager, backed):
    did = _register(Path(dataset_manager.data_dir))
    r = await client.get(f"{API}/datasets/{did}/markers", params={"groupby_column": "grp", "format": "json"})
    assert r.status_code == 200, r.text
    assert set(r.json()["groups"]) == {"a", "b", "c"}
    adaptor = await dataset_manager.get_or_load_dataset(did)
    assert not adaptor.adata.isbacked  # read in for the markers, and served from memory since
    assert adaptor.has_markers(column="grp")


async def test_reading_a_backed_dataset_in_during_a_run_on_another_gets_the_run_sentence(
        client, dataset_manager, backed, monkeypatch):
    # Light work on a second dataset (markers, scoring, a PAGA) reads it in through the
    # same path as a load: a run's expected growth is counted, and the refusal names the run.
    from scview.core.dataset_manager import RunMemoryError

    did = _register(Path(dataset_manager.data_dir))
    adaptor = await dataset_manager.get_or_load_dataset(did)
    assert adaptor.adata.isbacked
    monkeypatch.setattr(memory_budget, "read_budget",
                        lambda *a, **k: memory_budget.MemoryBudget(8 * GB, 1 * GB, True, "container limit"))
    monkeypatch.setattr(memory_budget, "release_memory", lambda: None)
    monkeypatch.setattr(preflight, "backed_bytes", lambda h: 2 * GB)  # 2.2 + 2 reserve: fits in 7
    assert await working_copy.make_room(dataset_manager, did, adaptor) is True
    dataset_manager.mark_busy("run")
    dataset_manager.reserve_run("run", 6 * GB)  # 1 used + 6 to come: 1 GB left
    with pytest.raises(RunMemoryError, match="A run on run needs about 6.0 GB more .* before reading big into memory"):
        await working_copy.make_room(dataset_manager, did, adaptor)
    r = await client.get(f"{API}/datasets/{did}/markers", params={"groupby_column": "grp", "format": "json"})
    assert r.status_code == 507 and "A run on run needs about 6.0 GB" in r.json()["detail"]
    dataset_manager.release_run("run")
    assert await working_copy.make_room(dataset_manager, did, adaptor) is True
