"""Renaming a category carries over to markers and cached enrichment; merging keeps NaN."""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

API = "/api/v1"


def _register(dm) -> str:
    d = Path(dm.data_dir) / "uploads" / "rn"
    d.mkdir(parents=True)
    (d / "metadata.json").write_text(json.dumps({"id": "rn", "name": "rn", "status": "ready"}))
    rng = np.random.default_rng(0)
    X = rng.poisson(1.0, (90, 20)).astype("float32")
    X[:30, :5] += 8
    X[30:60, 5:10] += 8
    a = ad.AnnData(X)
    a.var_names = [f"G{i}" for i in range(20)]
    labels = np.array(["T"] * 30 + ["B"] * 30 + ["NK"] * 30, dtype=object)
    labels[[0, 1, 2]] = np.nan
    a.obs["cell_type"] = pd.Categorical(labels, categories=["T", "B", "NK"])
    sc.tl.rank_genes_groups(a, "cell_type", method="wilcoxon", pts=True,
                            key_added="rank_genes_groups__cell_type")
    a.uns["enrichment__cell_type__T"] = "[]"
    a.write_h5ad(d / "data.h5ad")
    return "rn"


async def _groups(client, did):
    r = await client.get(f"{API}/datasets/{did}/markers", params={"groupby_column": "cell_type", "format": "json"})
    assert r.status_code == 200, r.text
    return set(r.json()["groups"])


async def test_rename_follows_into_markers_and_enrichment(client, dataset_manager):
    did = _register(dataset_manager)
    r = await client.post(f"{API}/datasets/{did}/metadata/cell_type/rename", json={"old": "T", "new": "CD4 T"})
    assert r.status_code == 200, r.text
    assert await _groups(client, did) == {"CD4 T", "B", "NK"}
    uns = dataset_manager.get_dataset(did).adata.uns
    assert "enrichment__cell_type__CD4 T" in uns and "enrichment__cell_type__T" not in uns


async def test_merge_keeps_missing_values_and_recomputes_markers(client, dataset_manager):
    did = _register(dataset_manager)
    r = await client.post(f"{API}/datasets/{did}/metadata/cell_type/rename", json={"old": "B", "new": "NK"})
    assert r.status_code == 200, r.text
    obs = dataset_manager.get_dataset(did).adata.obs["cell_type"]
    assert list(obs.cat.categories) == ["T", "NK"]
    assert obs.isna().sum() == 3
    assert await _groups(client, did) == {"T", "NK"}
    assert "enrichment__cell_type__T" not in dataset_manager.get_dataset(did).adata.uns


async def test_metadata_json_sends_null_for_missing(client, dataset_manager):
    did = _register(dataset_manager)
    r = await client.get(f"{API}/datasets/{did}/metadata/cell_type", params={"format": "json"})
    assert r.status_code == 200, r.text
    assert r.json()["values"][:4] == [None, None, None, "T"]
