"""Tests for DatasetManager list filtering + orphan pruning."""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pytest

from scview.core.dataset_manager import DatasetManager


@pytest.fixture()
def dm(tmp_path: Path) -> DatasetManager:
    for sub in ("uploads", "converted", "cache"):
        (tmp_path / sub).mkdir()
    return DatasetManager(data_dir=str(tmp_path))


def _meta(dm: DatasetManager, did: str, name: str, status: str) -> Path:
    d = Path(dm.data_dir) / "uploads" / did
    d.mkdir(parents=True)
    (d / "metadata.json").write_text(json.dumps({"id": did, "name": name, "status": status}))
    return d


def _h5ad(dm: DatasetManager, did: str) -> None:
    d = Path(dm.data_dir) / "ingested" / did
    d.mkdir(parents=True)
    ad.AnnData(X=np.zeros((3, 2), dtype="float32")).write_h5ad(d / "data.h5ad")


def test_list_skips_orphans(dm):
    _meta(dm, "good", "Good", "ready")
    _h5ad(dm, "good")  # has a resolvable data file
    _meta(dm, "orphan", "Failed", "error")  # no data file
    _meta(dm, "converting", "Busy", "converting")  # in-progress, no file yet

    listed = {d["id"] for d in dm.list_datasets()}
    assert "good" in listed
    assert "converting" in listed  # in-progress kept
    assert "orphan" not in listed  # unopenable hidden


def test_prune_removes_only_orphans(dm):
    _meta(dm, "good", "Good", "ready")
    _h5ad(dm, "good")
    _meta(dm, "orphan", "Failed", "error")
    _meta(dm, "converting", "Busy", "converting")

    removed = dm.prune_orphans()
    removed_ids = {r["id"] for r in removed}

    assert removed_ids == {"orphan"}
    assert (Path(dm.data_dir) / "uploads" / "good").exists()
    assert (Path(dm.data_dir) / "uploads" / "converting").exists()
    assert not (Path(dm.data_dir) / "uploads" / "orphan").exists()


def test_prune_empty_when_all_openable(dm):
    _meta(dm, "good", "Good", "ready")
    _h5ad(dm, "good")
    assert dm.prune_orphans() == []


# --- opening a dataset makes room in memory first ---------------------------------

GB = 1024**3


def _three(dm: DatasetManager) -> None:
    for did in ("a", "b", "c"):
        _meta(dm, did, did, "ready")
        _h5ad(dm, did)


def _fake_memory(monkeypatch, dm, *, need_gb: float, limited: bool = True, per_dataset_gb: float = 2.0):
    """6 GB limit, 1 GB base, each cached dataset ``per_dataset_gb``; every file
    needs ``need_gb`` to open."""
    from scview.core import memory_budget, preflight

    def budget(*_a, **_k):
        used = int((1 + per_dataset_gb * len(dm._datasets)) * GB)
        return memory_budget.MemoryBudget(6 * GB, used, limited,
                                          "container limit" if limited else "machine")

    released = []
    monkeypatch.setattr(memory_budget, "read_budget", budget)
    monkeypatch.setattr(memory_budget, "release_memory", lambda: released.append(1))
    monkeypatch.setattr(preflight, "read_header", lambda path: path)
    monkeypatch.setattr(preflight, "estimate_load", lambda h, ingest=True: preflight.LoadEstimate(
        int(need_gb * GB), int(need_gb * GB), "reading the file", (1.0, 2.0)))
    return released


async def test_opening_drops_least_recently_used_datasets_until_it_fits(dm, monkeypatch):
    _three(dm)
    _fake_memory(monkeypatch, dm, need_gb=99, limited=False)  # load a and b without a limit first
    await dm.load_dataset("a")
    await dm.load_dataset("b")
    # 1 + 2 + 2 = 5 GB used, 1 GB free; c needs 1.1 GB + a 1.5 GB working reserve
    released = _fake_memory(monkeypatch, dm, need_gb=1.0)
    await dm.load_dataset("c")
    assert dm.cached_ids() == ["b", "c"]  # a (least recent) went; one drop made 3 GB free
    assert released


async def test_a_dataset_that_cannot_fit_even_alone_is_refused(dm, monkeypatch):
    from scview.core.dataset_manager import DatasetMemoryError

    _three(dm)
    _fake_memory(monkeypatch, dm, need_gb=99, limited=False)
    await dm.load_dataset("a")
    _fake_memory(monkeypatch, dm, need_gb=7.0)  # 5 GB free with nothing cached; 7 x 0.8 > 5
    monkeypatch.setattr("scview.core.dataset_manager.REFUSAL_WAIT_SECONDS", 0)
    with pytest.raises(DatasetMemoryError, match="needs about 7.0 GB .* 5.0 GB free of its 6.0 GB"):
        await dm.load_dataset("c")
    assert dm.cached_ids() == []
    # Within the estimate's error band (6 x 0.8 = 4.8 <= 5 GB free): opened.
    _fake_memory(monkeypatch, dm, need_gb=6.0)
    await dm.load_dataset("c")
    assert dm.cached_ids() == ["c"]


async def test_a_load_keeps_a_working_reserve_free(dm, monkeypatch):
    # Two datasets of 2 GB on 6 GB leave 1 GB: enough for a 0.5 GB load, not for the
    # requests after it, so the least recent one goes (reserve 1.5 GB).
    _three(dm)
    _fake_memory(monkeypatch, dm, need_gb=0.1, limited=False)
    await dm.load_dataset("a")
    await dm.load_dataset("b")
    _fake_memory(monkeypatch, dm, need_gb=0.5)
    await dm.load_dataset("c")
    assert dm.cached_ids() == ["b", "c"]


async def test_a_load_waits_for_memory_a_finished_run_is_still_freeing(dm, monkeypatch):
    from scview.core import memory_budget

    _three(dm)
    _fake_memory(monkeypatch, dm, need_gb=4.0)
    held = {"gb": 4.5}  # a finished run's copy, gone after the second look

    def budget(*_a, **_k):
        used = int((1 + held["gb"]) * GB)
        held["gb"] = max(0.0, held["gb"] - 2.5)
        return memory_budget.MemoryBudget(6 * GB, used, True, "container limit")

    monkeypatch.setattr(memory_budget, "read_budget", budget)
    monkeypatch.setattr("asyncio.sleep", _no_sleep)
    await dm.load_dataset("a")
    assert dm.cached_ids() == ["a"]


async def _no_sleep(_s):
    return None


async def test_without_a_container_limit_nothing_is_dropped_for_memory(dm, monkeypatch):
    _three(dm)
    _fake_memory(monkeypatch, dm, need_gb=50, limited=False)
    for did in ("a", "b", "c"):
        await dm.load_dataset(did)
    assert dm.cached_ids() == ["a", "b", "c"]


async def test_free_memory_for_a_run_drops_other_datasets_only_as_needed(dm, monkeypatch):
    _three(dm)
    _fake_memory(monkeypatch, dm, need_gb=0.1, limited=False)
    for did in ("a", "b", "c"):
        await dm.load_dataset(did)
    _fake_memory(monkeypatch, dm, need_gb=0.1)  # 1 + 3 x 2 = 7 GB "used" of 6: over, as after a load
    free = await dm.free_memory_for(int(1.0 * GB), keep="c")  # 1.1 + 1.5 reserve: a and b go
    assert dm.cached_ids() == ["c"] and free == 3 * GB  # LRU order; c, the run's, stays
    assert await dm.free_memory_for(int(0.5 * GB), keep="c") == 3 * GB  # already fits: nothing dropped
    _fake_memory(monkeypatch, dm, need_gb=0.1, limited=False)
    assert await dm.free_memory_for(int(99 * GB), keep="c") is None
    assert dm.cached_ids() == ["c"]


def test_a_dropped_adaptor_never_reads_its_file_again(tmp_path, monkeypatch):
    # A request that took the adaptor before a run and read .adata after the drop
    # used to re-read the whole file outside the cache (3.2 GB on ovary: killed).
    from scview.core.anndata_adaptor import AnnDataAdaptor

    path = tmp_path / "d.h5ad"
    ad.AnnData(X=np.ones((4, 3), dtype="float32")).write_h5ad(path)
    a = AnnDataAdaptor(str(path))
    first = a.adata
    reads = []
    monkeypatch.setattr(ad, "read_h5ad", lambda *x, **k: reads.append(x) or first)
    a.close()
    assert a.adata is first and reads == []
