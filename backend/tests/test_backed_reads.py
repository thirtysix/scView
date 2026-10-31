"""Reads of a backed dataset's values outside a pipeline run (``working_copy.reading``):
DE, the doublet threshold, a joint embedding, a reference mapping, a learned gate and the
re-filter read only what they use, say what they need first, refuse it with a sentence
when it cannot fit, and hold the room until they end, so two at once count each other.
Every file opens backed here; the budget is monkeypatched.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
import scipy.sparse as sp

from scview.api.v1 import de as de_api
from scview.core import anndata_adaptor, integration, memory_budget, preflight, transcriptome_size, working_copy
from scview.core.dataset_manager import DatasetMemoryError, RunMemoryError

API = "/api/v1"
GB = 1024**3


def _dataset(n: int = 400, g: int = 80, dense_x: bool = False, seed: int = 0) -> ad.AnnData:
    """Counts in layers['counts'], log values in .raw, a scaled-like X: as a processed file."""
    rng = np.random.default_rng(seed)
    counts = rng.poisson(rng.gamma(1.0, 1.0, g), (n, g)).astype(np.float32)
    groups = np.where(np.arange(n) % 3 == 0, "a", np.where(np.arange(n) % 3 == 1, "b", "c"))
    counts[groups == "a", :10] *= 4
    a = ad.AnnData(sp.csr_matrix(counts))
    a.var_names = [f"G{i}" for i in range(g)]
    a.obs_names = [f"c{i}" for i in range(n)]
    a.obs["grp"] = pd.Categorical(groups)
    a.obs["donor"] = pd.Categorical([f"d{i % 6}" for i in range(n)])
    a.layers["counts"] = a.X.copy()
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    a.raw = a.copy()
    if dense_x:
        a.X = np.asarray(a.X.todense())
    return a


def _register(data_dir: Path, did: str, adata: ad.AnnData) -> str:
    d = data_dir / "uploads" / did
    d.mkdir(parents=True, exist_ok=True)
    (d / "metadata.json").write_text(json.dumps({"id": did, "name": f"{did} name", "status": "ready"}))
    adata.write_h5ad(d / "data.h5ad")
    return did


@pytest.fixture()
def backed(monkeypatch):
    monkeypatch.setattr(anndata_adaptor, "BACKED_MIN_BYTES", 0)


def _budget(monkeypatch, *, limit: float = 8.0, used: float = 1.0):
    state = {"used": used}

    def read(*_a, **_k):
        return memory_budget.MemoryBudget(int(limit * GB), int(state["used"] * GB), True, "container limit")

    monkeypatch.setattr(memory_budget, "read_budget", read)
    monkeypatch.setattr(memory_budget, "release_memory", lambda: None)
    return state


def _needs(monkeypatch, gb: float) -> None:
    """Every DE test on a backed dataset needs ``gb``."""
    monkeypatch.setattr(de_api, "_test_bytes", lambda *_a: int(gb * GB))
    monkeypatch.setattr(de_api, "_counts_bytes", lambda *_a: int(gb * GB))


COMPARE = {"split_by": "grp", "group_a": "a", "group_b": "b"}
BODIES = {"/de": {"indices": list(range(0, 120))}, "/de/methods": {"indices": list(range(0, 120))},
          "/de/compare": COMPARE, "/de/compare/methods": COMPARE,
          "/de/compare#pseudobulk": {**COMPARE, "method": "pseudobulk", "replicate_key": "donor"}}


async def _post(client, did: str, where: str):
    path = where.split("#")[0]
    return await client.post(f"{API}/datasets/{did}{path}", json=BODIES[where])


def _rows(r) -> list:
    return [(g["gene"], g["logfoldchange"], g["pval"], g["pval_adj"]) for g in r.json()["genes"]]


# -- DE: the same results, from the rows it uses ---------------------------------------


@pytest.mark.parametrize("dense_x", [False, True])
@pytest.mark.parametrize("where", ["/de", "/de/compare", "/de/compare#pseudobulk"])
async def test_a_backed_test_is_bitwise_the_in_memory_one(client, dataset_manager, tmp_data_dir, monkeypatch,
                                                          where, dense_x):
    _register(Path(tmp_data_dir), "x", _dataset(dense_x=dense_x))
    mem = await _post(client, "x", where)
    assert mem.status_code == 200, mem.text
    assert not (await dataset_manager.get_or_load_dataset("x")).adata.isbacked
    dataset_manager.evict("x")
    monkeypatch.setattr(anndata_adaptor, "BACKED_MIN_BYTES", 0)
    file_read = await _post(client, "x", where)
    assert file_read.status_code == 200, file_read.text
    assert (await dataset_manager.get_or_load_dataset("x")).adata.isbacked
    assert _rows(file_read) == _rows(mem)
    assert file_read.json().get("transcriptome_size") == mem.json().get("transcriptome_size")


async def test_a_backed_test_never_reads_the_whole_object_or_closes_the_file(client, dataset_manager, tmp_data_dir,
                                                                             backed, monkeypatch):
    _register(Path(tmp_data_dir), "x", _dataset())
    adata = (await dataset_manager.get_or_load_dataset("x")).adata

    def boom(*_a, **_k):
        raise AssertionError("read the whole object")

    monkeypatch.setattr(ad.AnnData, "to_memory", boom)
    for where in BODIES:
        r = await _post(client, "x", where)
        assert r.status_code == 200, f"{where}: {r.text}"
        assert adata.file.is_open


def test_read_rows_matches_a_read_of_the_whole_matrix(tmp_path):
    a = _dataset(dense_x=True, n=9000, g=5)
    a.write_h5ad(tmp_path / "d.h5ad")
    b = ad.read_h5ad(tmp_path / "d.h5ad", backed="r")
    keep = np.zeros(a.n_obs, dtype=bool)
    keep[[0, 1, 4095, 4096, 8999]] = True
    assert working_copy.on_disk(b.X) and working_copy.on_disk(b.raw.X)
    assert np.array_equal(working_copy.read_rows(b.X, keep), a.X[keep])
    assert (working_copy.read_rows(b.raw.X, keep) != a.raw.X[keep]).nnz == 0
    assert (working_copy.read_rows(b.raw.X, np.ones(a.n_obs, bool)) != a.raw.X).nnz == 0
    b.file.close()


def test_totals_of_a_backed_dataset_read_only_the_compared_blocks(tmp_path, monkeypatch):
    a = _dataset()
    del a.layers["counts"]
    a.raw = ad.AnnData(sp.csr_matrix(np.expm1(a.X.toarray())), var=a.var, obs=a.obs)  # not counts
    a.X = sp.csr_matrix(np.rint(a.X.toarray()) + 1)  # X holds whole numbers
    a.write_h5ad(tmp_path / "d.h5ad")
    b = ad.read_h5ad(tmp_path / "d.h5ad", backed="r")
    monkeypatch.setattr(transcriptome_size, "SUM_ROWS", 50)
    keep = np.zeros(a.n_obs, dtype=bool)
    keep[60:70] = True
    totals, source = transcriptome_size.per_cell_totals(b, keep)
    assert source == "X"
    assert np.array_equal(totals[50:100], np.asarray(a.X[50:100].sum(axis=1), dtype=np.float64).ravel())
    assert totals[:50].sum() == 0  # no compared cell in that block: not read
    b.file.close()


# -- the estimate ----------------------------------------------------------------------


def test_the_test_estimate_scales_with_the_cells_read_and_the_header_sees_legacy_raw(tmp_path):
    a = _dataset()
    a.write_h5ad(tmp_path / "d.h5ad")
    h = preflight.read_header(tmp_path / "d.h5ad", "h5ad")
    raw = h.matrix("raw/X")
    assert preflight.rows_bytes(raw, raw.n_cells) == raw.nbytes
    assert preflight.rows_bytes(raw, raw.n_cells // 2) < raw.nbytes * 0.6
    assert preflight.de_read_need(raw, raw.n_cells) > preflight.de_read_need(raw, raw.n_cells // 2)
    assert preflight.de_read_need(raw, 100, logreg=True) > preflight.de_read_need(raw, 100)
    assert preflight.de_read_need(None, 100) == 0

    import h5py

    with h5py.File(tmp_path / "d.h5ad", "a") as f:  # anndata < 0.7 kept it at the top level
        f.move("raw/X", "raw.X")
        del f["raw"]
    h = preflight.read_header(tmp_path / "d.h5ad", "h5ad")
    assert h.matrix("raw/X") is not None and h.matrix("raw/X").nnz == raw.nnz


async def test_an_in_memory_dataset_is_not_checked(client, dataset_manager, tmp_data_dir, monkeypatch):
    _register(Path(tmp_data_dir), "x", _dataset())
    await dataset_manager.get_or_load_dataset("x")
    _budget(monkeypatch, limit=1.0, used=1.0)  # nothing free at all
    calls = []
    monkeypatch.setattr(dataset_manager, "hold_read", lambda *a, **k: calls.append(a) or (lambda: None))
    r = await client.post(f"{API}/datasets/x/de/compare", json=COMPARE)
    assert r.status_code == 200, r.text
    assert calls == []


# -- refused, runs, released -----------------------------------------------------------


@pytest.mark.parametrize("where", list(BODIES))
async def test_a_test_that_cannot_fit_is_a_507_naming_the_dataset_and_the_numbers(
        client, dataset_manager, tmp_data_dir, backed, monkeypatch, where):
    _register(Path(tmp_data_dir), "x", _dataset())
    await dataset_manager.get_or_load_dataset("x")
    _budget(monkeypatch, limit=8.0, used=1.0)
    _needs(monkeypatch, 100.0)
    monkeypatch.setattr("scview.core.dataset_manager.REFUSAL_WAIT_SECONDS", 1)
    r = await _post(client, "x", where)
    assert r.status_code == 507, r.text
    detail = r.json()["detail"]
    assert "x name" in detail and "100.0 GB" in detail and "7.0 GB free of its 8.0 GB limit" in detail
    assert dataset_manager._reads == {} and not dataset_manager._reading


@pytest.mark.parametrize("where", list(BODIES))
async def test_a_test_that_fits_runs_and_lets_go(client, dataset_manager, tmp_data_dir, backed, monkeypatch, where):
    _register(Path(tmp_data_dir), "x", _dataset())
    await dataset_manager.get_or_load_dataset("x")
    _budget(monkeypatch, limit=8.0, used=1.0)
    _needs(monkeypatch, 1.0)
    held = []
    real = de_api.transcriptome_size.check

    def spy(*a):
        held.append(dataset_manager.outstanding(GB))
        return real(*a)

    monkeypatch.setattr(de_api.transcriptome_size, "check", spy)
    r = await _post(client, "x", where)
    assert r.status_code == 200, r.text
    if "methods" not in where:
        assert held == [GB]  # counted while it worked
    assert dataset_manager._reads == {} and not dataset_manager._reading


async def test_the_reservation_goes_when_the_test_fails(client, dataset_manager, tmp_data_dir, backed, monkeypatch):
    _register(Path(tmp_data_dir), "x", _dataset())
    await dataset_manager.get_or_load_dataset("x")
    _budget(monkeypatch)
    _needs(monkeypatch, 1.0)

    def fails(*_a):
        raise RuntimeError("no")

    monkeypatch.setattr(de_api, "_run_method", fails)
    with pytest.raises(RuntimeError):
        await client.post(f"{API}/datasets/x/de/compare", json=COMPARE)
    assert dataset_manager._reads == {} and not dataset_manager._reading


async def test_a_cancelled_request_holds_until_its_thread_ends(dataset_manager, monkeypatch):
    import threading

    _budget(monkeypatch)
    go, started = threading.Event(), threading.Event()

    def work():
        started.set()
        go.wait(5)

    async def request():
        async with working_copy.reading(dataset_manager, "x", GB, "Work on x") as held:
            await held.run(work)

    task = asyncio.ensure_future(request())
    while not started.is_set():
        await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert dataset_manager.outstanding(GB) == GB  # the thread still reads
    go.set()
    for _ in range(200):
        if not dataset_manager._reads:
            break
        await asyncio.sleep(0.01)
    assert dataset_manager._reads == {} and not dataset_manager._reading


async def test_two_at_once_count_each_other(client, dataset_manager, tmp_data_dir, backed, monkeypatch):
    """Each of 5 GB fits alone in the 7 GB free; the second waits, and is refused naming
    the first while it still reads, then runs once the first is done."""
    import threading

    _register(Path(tmp_data_dir), "x", _dataset())
    _register(Path(tmp_data_dir), "y", _dataset(seed=1))
    await dataset_manager.get_or_load_dataset("x")
    await dataset_manager.get_or_load_dataset("y")
    _budget(monkeypatch, limit=8.0, used=1.0)
    _needs(monkeypatch, 5.0)
    monkeypatch.setattr("scview.core.dataset_manager.REFUSAL_WAIT_SECONDS", 1)
    go, started = threading.Event(), threading.Event()
    real = de_api._cell_test

    def slow(*a, **k):
        started.set()
        go.wait(10)
        return real(*a, **k)

    monkeypatch.setattr(de_api, "_cell_test", slow)
    first = asyncio.ensure_future(client.post(f"{API}/datasets/x/de/compare", json=COMPARE))
    while not started.is_set():
        await asyncio.sleep(0.01)
    assert dataset_manager.outstanding(GB) == 5 * GB
    second = await client.post(f"{API}/datasets/y/de/compare", json=COMPARE)
    assert second.status_code == 507, second.text
    assert second.json()["detail"] == (
        "Differential expression on x name needs about 5.0 GB more at its peak and scView has 7.0 GB free of "
        "its 8.0 GB limit: wait for it to finish, before differential expression on y name.")
    go.set()
    r = await first
    assert r.status_code == 200, r.text
    third = await client.post(f"{API}/datasets/y/de/compare", json=COMPARE)
    assert third.status_code == 200, third.text


async def test_a_dataset_being_read_is_not_dropped_for_another_request(client, dataset_manager, tmp_data_dir,
                                                                       backed, monkeypatch):
    _register(Path(tmp_data_dir), "x", _dataset())
    _register(Path(tmp_data_dir), "y", _dataset(seed=1))
    await dataset_manager.get_or_load_dataset("x")
    await dataset_manager.get_or_load_dataset("y")
    release = dataset_manager.hold_read("x", 0)
    assert dataset_manager._evictable("y") == []
    release()
    release()  # safe twice
    assert dataset_manager._evictable("y") == ["x"]


def test_runs_and_reads_stack_their_growth(dataset_manager, monkeypatch):
    state = _budget(monkeypatch, limit=20.0, used=2.0)
    dataset_manager.reserve_run("a", 4 * GB)
    r = dataset_manager.reserve_run("b", 3 * GB)
    assert r.base_used == 6 * GB  # 2 used + a's 4 to come
    release = dataset_manager.hold_read("c", GB, "A read")
    assert dataset_manager.outstanding(2 * GB) == 8 * GB
    state["used"] = 5.0  # a grew by 3
    assert dataset_manager.outstanding(5 * GB) == GB + 3 * GB + GB
    assert dataset_manager.outstanding(5 * GB, keep="a") == 3 * GB + GB
    release()
    dataset_manager.release_run("a")
    dataset_manager.release_run("b")
    assert dataset_manager.outstanding(5 * GB) == 0


async def test_a_run_holding_the_room_is_named_at_once_and_a_read_only_after_the_wait(dataset_manager, tmp_data_dir,
                                                                                      monkeypatch):
    _budget(monkeypatch, limit=8.0, used=1.0)
    monkeypatch.setattr("scview.core.dataset_manager.REFUSAL_WAIT_SECONDS", 2)
    slept = []

    async def sleep(s):
        slept.append(s)

    monkeypatch.setattr("scview.core.dataset_manager.asyncio.sleep", sleep)
    dataset_manager.reserve_run("run", 5 * GB)
    with pytest.raises(RunMemoryError, match="A run on run needs about 5.0 GB"):
        await dataset_manager.reserve("q", 5 * GB)
    assert slept == []
    dataset_manager.release_run("run")
    release = dataset_manager.hold_read("r", 5 * GB, "A read on r")
    with pytest.raises(RunMemoryError, match="A read on r needs about 5.0 GB .*: wait for it to finish, before"):
        await dataset_manager.reserve("q", 5 * GB)
    assert slept == [1, 1]
    release()


# -- the doublet threshold writes no matrix ---------------------------------------------


@pytest.mark.parametrize("derived_first", [False, True])
async def test_the_doublet_threshold_of_a_backed_dataset_reads_no_matrix(client, dataset_manager, tmp_data_dir, backed,
                                                                         monkeypatch, derived_first):
    a = _dataset()
    rng = np.random.default_rng(0)
    a.obs["doublet_score"] = rng.uniform(0, 1, a.n_obs).astype(np.float32)
    a.obs["predicted_doublet"] = a.obs["doublet_score"] > 0.9
    _register(Path(tmp_data_dir), "x", a)
    if derived_first:  # it reads its own derived file: replaced, not next to it
        adaptor = await dataset_manager.get_or_load_dataset("x")
        out = dataset_manager.derived_h5ad_path("x", adaptor.h5ad_path)
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        a.write_h5ad(out)
        dataset_manager.evict("x")
    await dataset_manager.get_or_load_dataset("x")

    def boom(*_a, **_k):
        raise AssertionError("read the whole object")

    monkeypatch.setattr(ad.AnnData, "to_memory", boom)
    r = await client.post(f"{API}/datasets/x/qc/doublets/threshold", json={"threshold": 0.5})
    assert r.status_code == 200, r.text
    monkeypatch.undo()
    saved = ad.read_h5ad(dataset_manager.derived_h5ad_path("x", "data.h5ad"))
    assert int(saved.obs["predicted_doublet"].sum()) == int((a.obs["doublet_score"] > 0.5).sum())
    assert (saved.X != a.X).nnz == 0 and (saved.raw.X != a.raw.X).nnz == 0
    assert (saved.layers["counts"] != a.layers["counts"]).nnz == 0


# -- counts of a joint embedding, a mapping, a learned gate ------------------------------


def _pair(tmp_data_dir):
    _register(Path(tmp_data_dir), "x", _dataset(n=300))
    _register(Path(tmp_data_dir), "y", _dataset(n=240, seed=1))


async def test_a_plan_reads_only_counts_and_refuses_with_both_names(client, dataset_manager, tmp_data_dir, backed,
                                                                    monkeypatch):
    _pair(tmp_data_dir)
    real = ad.AnnData.to_memory

    def boom(*_a, **_k):
        raise AssertionError("read the whole object")

    monkeypatch.setattr(ad.AnnData, "to_memory", boom)
    body = {"dataset_ids": ["x", "y"]}
    r = await client.post(f"{API}/integrate/plan", json=body)
    assert r.status_code == 200, r.text
    assert all(d["error"] is None for d in r.json()["datasets"])
    assert [d["counts_source"] for d in r.json()["datasets"]] == ["layers['counts']", "layers['counts']"]
    monkeypatch.setattr(ad.AnnData, "to_memory", real)
    _budget(monkeypatch, limit=8.0, used=1.0)
    monkeypatch.setattr("scview.api.v1.integration.read_need", lambda *_a: 100 * GB)
    monkeypatch.setattr("scview.core.dataset_manager.REFUSAL_WAIT_SECONDS", 1)
    r = await client.post(f"{API}/integrate/plan", json=body)
    assert r.status_code == 507
    assert "A joint embedding of x name + y name needs about 100.0 GB" in r.json()["detail"]
    assert dataset_manager._reads == {} and not dataset_manager._reading


def test_counts_on_disk_equal_a_whole_read(tmp_path):
    a = _dataset()
    del a.layers["counts"]
    a.X = a.X.copy()
    counts = sp.csr_matrix(np.rint(np.expm1(a.raw.X.toarray()) / 100))
    a.raw = ad.AnnData(counts, var=a.var, obs=a.obs)
    a.write_h5ad(tmp_path / "d.h5ad")
    mem = integration.recover_counts(ad.read_h5ad(tmp_path / "d.h5ad"), [])
    b = ad.read_h5ad(tmp_path / "d.h5ad", backed="r")
    got = integration.recover_counts(b, [])
    assert got.source == mem.source == "raw.X"
    assert (got.X != mem.X).nnz == 0 and got.X.dtype == np.float32 and b.file.is_open
    b.file.close()
    # and from a source file, which it closes
    c = a.copy()
    del c.raw
    c.X = sp.csr_matrix(np.ones((a.n_obs, a.n_vars)) * 0.5)
    fb = integration.recover_counts(c, [str(tmp_path / "d.h5ad")])
    assert (fb.X != mem.X).nnz == 0


def test_the_read_estimate_counts_every_unit_and_the_largest_open(tmp_data_dir, dataset_manager):
    _pair(tmp_data_dir)
    need = integration.read_need(dataset_manager, ["x", "y", "gone"])
    hx = preflight.read_header(Path(tmp_data_dir) / "uploads/x/data.h5ad", "h5ad")
    hy = preflight.read_header(Path(tmp_data_dir) / "uploads/y/data.h5ad", "h5ad")
    assert need > preflight._lt_counts_bytes(hx) + preflight._lt_counts_bytes(hy)


async def test_a_mapping_query_opened_backed_is_not_read_whole_and_holds_until_it_ends(
        client, dataset_manager, tmp_data_dir, backed, monkeypatch):
    _register(Path(tmp_data_dir), "q", _dataset(n=300, g=600))
    ref = _dataset(n=300, g=600, seed=2)
    ref.obs["ct"] = ref.obs["grp"]
    _register(Path(tmp_data_dir), "r", ref)
    await dataset_manager.get_or_load_dataset("q")
    state = _budget(monkeypatch, limit=64.0, used=1.0)
    assert state
    seen = []

    from scview.api.v1 import label_transfer as api
    real = api.lt.prepare

    def spy(*a, **k):
        seen.append(dataset_manager.outstanding(GB) > 0)
        assert a[1].isbacked  # the query as it is
        return real(*a, **k)

    monkeypatch.setattr(api.lt, "prepare", spy)
    r = await client.post(f"{API}/datasets/q/label-transfer/run-stream",
                          json={"reference_id": "r", "label_column": "ct", "method": "projection"})
    assert r.status_code == 200, r.text
    assert "event: error" not in r.text.split("event: result")[0], r.text[:600]
    assert seen == [True]
    assert dataset_manager._reads == {} and not dataset_manager._reading


async def test_a_learned_gate_on_a_backed_dataset_reads_counts_only_and_can_be_refused(
        client, dataset_manager, tmp_data_dir, backed, monkeypatch):
    _register(Path(tmp_data_dir), "x", _dataset(n=400))
    await dataset_manager.get_or_load_dataset("x")
    _budget(monkeypatch, limit=8.0, used=1.0)
    monkeypatch.setattr(working_copy, "counts_bytes", lambda *_a: 100 * GB)
    monkeypatch.setattr("scview.core.dataset_manager.REFUSAL_WAIT_SECONDS", 1)
    body = {"cells": {"n_cells": 400, "indices": list(range(0, 100))}, "name": "g"}
    r = await client.post(f"{API}/datasets/x/learned-gates/train-stream", json=body)
    assert r.status_code == 507 and "Learned gates on x name needs about 100.0 GB" in r.json()["detail"]
    monkeypatch.setattr(working_copy, "counts_bytes", lambda *_a: GB)

    def boom(*_a, **_k):
        raise AssertionError("read the whole object")

    monkeypatch.setattr(ad.AnnData, "to_memory", boom)
    r = await client.post(f"{API}/datasets/x/learned-gates/train-stream", json=body)
    assert r.status_code == 200, r.text
    assert "event: start" in r.text
    assert dataset_manager._reads == {} and not dataset_manager._reading


async def test_a_refilter_preview_says_what_reading_the_upload_needs(client, dataset_manager, tmp_data_dir, monkeypatch):
    from scview.core import mito_refilter

    _register(Path(tmp_data_dir), "x", _dataset())
    await dataset_manager.get_or_load_dataset("x")
    _budget(monkeypatch, limit=8.0, used=1.0)
    monkeypatch.setattr(mito_refilter, "read_need", lambda *_a: 100 * GB)
    monkeypatch.setattr("scview.core.dataset_manager.REFUSAL_WAIT_SECONDS", 1)
    r = await client.get(f"{API}/datasets/x/qc/mito-refilter")
    assert r.status_code == 507, r.text
    assert "Previewing a re-filter of x name from its original upload needs about 100.0 GB" in r.json()["detail"]


def test_the_refilter_read_is_the_whole_file_whatever_its_size(tmp_path, backed):
    a = _dataset()
    a.write_h5ad(tmp_path / "d.h5ad")
    h = preflight.read_header(tmp_path / "d.h5ad", "h5ad")
    assert h.backed  # the header says it would open with its matrices on disk
    assert working_copy.file_bytes(str(tmp_path / "d.h5ad")) >= h.matrix("raw/X").nbytes + h.matrix("X").nbytes


def test_refusals_stay_dataset_memory_errors():
    assert issubclass(RunMemoryError, DatasetMemoryError)
