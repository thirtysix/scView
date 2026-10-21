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
    released = _fake_memory(monkeypatch, dm, need_gb=2.0)  # 1 + 2 + 2 = 5 GB used, 1 GB free
    await dm.load_dataset("c")
    assert dm.cached_ids() == ["b", "c"]  # a (least recent) went; one drop made 3 GB free
    assert released


async def test_a_dataset_that_cannot_fit_even_alone_is_refused(dm, monkeypatch):
    from scview.core.dataset_manager import DatasetMemoryError

    _three(dm)
    _fake_memory(monkeypatch, dm, need_gb=99, limited=False)
    await dm.load_dataset("a")
    _fake_memory(monkeypatch, dm, need_gb=7.0)  # 5 GB free with nothing cached; 7 x 0.8 > 5
    with pytest.raises(DatasetMemoryError, match="needs about 7.0 GB .* 5.0 GB free of its 6.0 GB"):
        await dm.load_dataset("c")
    assert dm.cached_ids() == []
    # Within the estimate's error band (6 x 0.8 = 4.8 <= 5 GB free): opened.
    _fake_memory(monkeypatch, dm, need_gb=6.0)
    await dm.load_dataset("c")
    assert dm.cached_ids() == ["c"]


async def test_without_a_container_limit_nothing_is_dropped_for_memory(dm, monkeypatch):
    _three(dm)
    _fake_memory(monkeypatch, dm, need_gb=50, limited=False)
    for did in ("a", "b", "c"):
        await dm.load_dataset(did)
    assert dm.cached_ids() == ["a", "b", "c"]
