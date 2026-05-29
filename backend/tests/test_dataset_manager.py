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
