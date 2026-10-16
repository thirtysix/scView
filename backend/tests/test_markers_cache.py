"""On-demand markers are computed once, saved, and filtered by group.

They used to recompute rank_genes_groups on a full copy of the dataset on every
visit (~28 s on the 14k-cell Kang data) and ignored the ``groupby`` filter.
"""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scanpy as sc

from scview.api.v1 import markers as markers_api
from scview.core.uns_keys import enrichment_key, markers_key

API = "/api/v1"


def _register(dm, did: str = "mk", subdir: str = "uploads") -> Path:
    """Log-normalised data with .raw set (scanpy ranks .raw), an object label column
    and a bool column; written under ``subdir`` with the metadata in uploads/."""
    d = Path(dm.data_dir) / subdir / did
    d.mkdir(parents=True, exist_ok=True)
    meta = Path(dm.data_dir) / "uploads" / did
    meta.mkdir(parents=True, exist_ok=True)
    (meta / "metadata.json").write_text(json.dumps({"id": did, "name": did, "status": "ready"}))
    rng = np.random.default_rng(0)
    X = rng.poisson(0.4, size=(120, 40)).astype("float32")
    X[:40, :8] += 6
    X[40:80, 8:16] += 6
    a = ad.AnnData(X)
    a.var_names = [f"G{i}" for i in range(40)]
    a.obs_names = [f"C{i}" for i in range(120)]
    a.obs["cell_type"] = np.array(["T", "B", "NK"]).repeat(40)  # object, not categorical
    a.obs["flag"] = np.arange(120) < 30
    sc.pp.normalize_total(a)
    sc.pp.log1p(a)
    a.raw = a
    a.uns[enrichment_key("cell_type", "T")] = "[]"  # stale: must go with new markers
    path = d / "data.h5ad"
    a.write_h5ad(path)
    return path


@pytest.fixture()
def count_computes(monkeypatch):
    calls = []
    real = markers_api._compute_markers

    def counting(adata, column):
        calls.append(column)
        return real(adata, column)

    monkeypatch.setattr(markers_api, "_compute_markers", counting)
    return calls


async def _markers(client, did, **params):
    r = await client.get(f"{API}/datasets/{did}/markers",
                         params={"format": "json", "groupby_column": "cell_type", **params})
    assert r.status_code == 200, r.text
    return r.json()


async def test_computed_once_then_served_from_the_cache(client, dataset_manager, count_computes):
    _register(dataset_manager)
    first = await _markers(client, "mk", n_genes=5)
    assert set(first["groups"]) == {"T", "B", "NK"}
    assert all(len(rows) == 5 for rows in first["markers"].values())
    # Every gene is kept, so a longer table later needs no recompute either.
    second = await _markers(client, "mk", n_genes=20)
    assert all(len(rows) == 20 for rows in second["markers"].values())
    assert count_computes == ["cell_type"]

    uns = dataset_manager.get_dataset("mk").adata.uns
    assert markers_key("cell_type") in uns
    assert enrichment_key("cell_type", "T") not in uns


async def test_saved_to_the_derived_file_and_reused_after_reload(
    client, dataset_manager, count_computes
):
    original = _register(dataset_manager)
    before = original.stat().st_mtime_ns
    await _markers(client, "mk")
    assert original.stat().st_mtime_ns == before, "the original upload is never written"
    derived = Path(dataset_manager.data_dir) / "derived" / "mk" / "data.h5ad"
    assert markers_key("cell_type") in ad.read_h5ad(derived).uns

    dataset_manager.evict("mk")
    again = await _markers(client, "mk", groupby="B")
    assert again["groups"] == ["B"]
    assert count_computes == ["cell_type"]


async def test_a_derived_file_gets_only_the_new_entry(client, dataset_manager, count_computes):
    path = _register(dataset_manager, subdir="derived")
    await _markers(client, "mk")
    r = await client.get(f"{API}/datasets/mk/markers",
                         params={"format": "json", "groupby_column": "flag"})
    assert r.status_code == 200, r.text
    saved = ad.read_h5ad(path)
    assert markers_key("cell_type") in saved.uns and markers_key("flag") in saved.uns
    assert saved.shape == (120, 40) and saved.raw is not None
    assert count_computes == ["cell_type", "flag"]


async def test_group_filter_is_honoured(client, dataset_manager):
    _register(dataset_manager)
    body = await _markers(client, "mk", groupby="NK", n_genes=3)
    assert body["groups"] == ["NK"]
    assert len(body["markers"]["NK"]) == 3


async def test_matches_the_pipeline_and_leaves_obs_alone(client, dataset_manager):
    path = _register(dataset_manager)
    await _markers(client, "mk")
    adata = dataset_manager.get_dataset("mk").adata
    assert adata.obs["flag"].dtype == bool
    got = adata.uns[markers_key("cell_type")]
    assert got["params"]["use_raw"] is True

    ref = ad.read_h5ad(path)
    ref.obs["cell_type"] = ref.obs["cell_type"].astype("category")
    sc.tl.rank_genes_groups(ref, "cell_type", method="wilcoxon", pts=True)
    want = ref.uns["rank_genes_groups"]
    for g in ("T", "B", "NK"):
        assert list(got["names"][g]) == list(want["names"][g])
        np.testing.assert_allclose(got["scores"][g], want["scores"][g], rtol=1e-5)
    pd.testing.assert_frame_equal(got["pts"], want["pts"])


async def test_too_many_groups_is_400(client, dataset_manager):
    path = _register(dataset_manager)
    a = ad.read_h5ad(path)
    a.obs["barcode_like"] = [f"b{i}" for i in range(a.n_obs)]
    a.write_h5ad(path)
    r = await client.get(f"{API}/datasets/mk/markers",
                         params={"format": "json", "groupby_column": "barcode_like"})
    assert r.status_code == 400
    assert "at most" in r.json()["detail"]
