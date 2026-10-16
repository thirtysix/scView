"""The cell fingerprint a saved session checks (GET /datasets/{id} cells_fingerprint)."""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from scview.core.anndata_adaptor import AnnDataAdaptor

API = "/api/v1"


def _adata(names: list[str]) -> ad.AnnData:
    a = ad.AnnData(np.ones((len(names), 3), dtype="float32"))
    a.obs_names = names
    return a


def _fp(a: ad.AnnData) -> str:
    adaptor = AnnDataAdaptor("unused.h5ad")
    adaptor._adata = a
    return adaptor.cells_fingerprint()


def test_same_cells_same_fingerprint_whatever_else_changes():
    a = _adata(["c0", "c1", "c2"])
    b = _adata(["c0", "c1", "c2"])
    b.obs["manual_labels"] = pd.Categorical(["x", "y", "x"])  # a label column
    b.obsm["X_umap"] = np.zeros((3, 2))
    assert _fp(a) == _fp(b)
    assert len(_fp(a)) == 24


def test_other_cells_or_order_change_it():
    base = _fp(_adata(["c0", "c1", "c2"]))
    assert _fp(_adata(["c1", "c0", "c2"])) != base  # reordered
    assert _fp(_adata(["c0", "c1"])) != base  # filtered
    assert _fp(_adata(["c0", "c1", "c3"])) != base  # another cell


def test_cached_until_the_cells_change():
    adaptor = AnnDataAdaptor("unused.h5ad")
    adaptor._adata = _adata(["c0", "c1", "c2"])
    first = adaptor.cells_fingerprint()
    assert adaptor.cells_fingerprint() == first
    adaptor._adata = adaptor._adata[[0, 1]].copy()
    assert adaptor.cells_fingerprint() != first


async def test_dataset_info_carries_it(client, dataset_manager):
    d = Path(dataset_manager.data_dir) / "uploads" / "fp"
    d.mkdir(parents=True)
    (d / "metadata.json").write_text(json.dumps({"id": "fp", "name": "fp", "status": "ready"}))
    a = _adata(["c0", "c1", "c2"])
    a.write_h5ad(d / "data.h5ad")
    info = (await client.get(f"{API}/datasets/fp")).json()
    assert info["cells_fingerprint"] == _fp(a)
    assert info["n_cells"] == 3
