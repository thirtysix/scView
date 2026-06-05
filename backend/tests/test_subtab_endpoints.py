"""Phase 6 §3: Unified View subtab endpoints — markers (on-demand) + gene sets.

Uses the httpx client fixture (asyncio_mode=auto). MSIGDB_DIR is unset in tests,
so MSigDB-backed collection listing degrades gracefully; gene-set *scoring* takes
an explicit gene list and works offline.
"""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

API = "/api/v1"


def _register(dm, did: str = "ds1", *, with_cluster: bool = True,
              n: int = 120, g: int = 80) -> str:
    """Register a small structured dataset (2 groups) under uploads/<id>/."""
    d = Path(dm.data_dir) / "uploads" / did
    d.mkdir(parents=True, exist_ok=True)
    (d / "metadata.json").write_text(
        json.dumps({"id": did, "name": did, "status": "ready"})
    )
    rng = np.random.default_rng(0)
    X = rng.poisson(0.3, size=(n, g)).astype("float32")
    X[: n // 2, :20] += rng.poisson(6.0, size=(n // 2, 20)).astype("float32")
    a = ad.AnnData(X)
    a.var_names = [f"G{i}" for i in range(g)]
    a.obs_names = [f"C{i}" for i in range(n)]
    if with_cluster:
        a.obs["cluster"] = pd.Categorical(["0"] * (n // 2) + ["1"] * (n - n // 2))
    a.write_h5ad(d / "data.h5ad")
    return did


async def test_markers_on_demand_for_column(client, dataset_manager):
    did = _register(dataset_manager)
    r = await client.get(
        f"{API}/datasets/{did}/markers?groupby_column=cluster&format=json"
    )
    assert r.status_code == 200, r.text
    # response carries per-group/gene marker rows
    assert "group" in r.text and "gene" in r.text


async def test_markers_unknown_column_404(client, dataset_manager):
    did = _register(dataset_manager)
    r = await client.get(
        f"{API}/datasets/{did}/markers?groupby_column=does_not_exist&format=json"
    )
    assert r.status_code == 404


async def test_markers_missing_dataset_404(client, dataset_manager):
    r = await client.get(
        f"{API}/datasets/ghost/markers?groupby_column=cluster&format=json"
    )
    assert r.status_code == 404


async def test_genesets_collections_endpoint(client, dataset_manager):
    """Returns 200 with a collections list — empty when MSIGDB_DIR is unset,
    populated (each item well-formed) when configured."""
    did = _register(dataset_manager)
    r = await client.get(f"{API}/datasets/{did}/genesets/collections")
    assert r.status_code == 200
    collections = r.json()["collections"]
    assert isinstance(collections, list)
    if collections:
        assert "category" in collections[0]


async def test_geneset_score_offline(client, dataset_manager):
    did = _register(dataset_manager)
    body = {"gene_set": ["G0", "G1", "G2", "NOSUCHGENE"], "score_name": "mysig"}
    r = await client.post(f"{API}/datasets/{did}/genesets/score", json=body)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["n_cells"] == 120
    assert len(data["scores"]) == 120
    assert "G0" in data["genes_found"]
    assert "NOSUCHGENE" in data["genes_missing"]


async def test_geneset_score_empty_list_400(client, dataset_manager):
    did = _register(dataset_manager)
    r = await client.post(
        f"{API}/datasets/{did}/genesets/score", json={"gene_set": []}
    )
    assert r.status_code == 400
