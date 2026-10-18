"""Per-group expression summary behind the marker dot plot and heatmap."""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

API = "/api/v1"


def _register(dm, did: str = "ds1") -> str:
    """4 cells in 2 groups; G0 is expressed only in group A, G1 in one cell of each."""
    d = Path(dm.data_dir) / "uploads" / did
    d.mkdir(parents=True, exist_ok=True)
    (d / "metadata.json").write_text(json.dumps({"id": did, "name": did, "status": "ready"}))
    X = np.array([[2, 1], [4, 0], [0, 3], [0, 0]], dtype="float32")
    a = ad.AnnData(X)
    a.var_names = ["G0", "G1"]
    a.obs_names = [f"C{i}" for i in range(4)]
    a.obs["cluster"] = pd.Categorical(["A", "A", "B", "B"], categories=["B", "A", "unused"])
    a.write_h5ad(d / "data.h5ad")
    return did


async def test_mean_and_fraction_per_group(client, dataset_manager):
    did = _register(dataset_manager)
    r = await client.get(
        f"{API}/datasets/{did}/expression/by-group",
        params={"genes": ["G0", "G1"], "groupby": "cluster", "layer": "X"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    # category order is kept; categories with no cells are dropped
    assert body["groups"] == ["B", "A"]
    assert body["genes"] == ["G0", "G1"]
    assert body["mean"] == [[0.0, 1.5], [3.0, 0.5]]
    assert body["fraction"] == [[0.0, 0.5], [1.0, 0.5]]
    # over the expressing cells only: no expressing cell gives 0, not NaN
    assert body["mean_expressing"] == [[0.0, 3.0], [3.0, 1.0]]
    # the layer's fixed range: 99th percentile of the non-zero values (1-4), rounded up
    assert body["scale_range"] == [0.0, 4.0]


async def test_scale_range_does_not_depend_on_the_genes_or_groups(client, dataset_manager):
    did = _register(dataset_manager)
    url = f"{API}/datasets/{did}/expression/by-group"
    one = (await client.get(url, params={"genes": ["G1"], "groupby": "cluster", "layer": "X"})).json()
    both = (await client.get(url, params={"genes": ["G0", "G1"], "groupby": "cluster", "layer": "X"})).json()
    assert one["scale_range"] == both["scale_range"] == [0.0, 4.0]


async def test_unknown_genes_are_dropped(client, dataset_manager):
    did = _register(dataset_manager)
    r = await client.get(
        f"{API}/datasets/{did}/expression/by-group",
        params={"genes": ["G1", "NOPE"], "groupby": "cluster", "layer": "X"},
    )
    assert r.status_code == 200, r.text
    assert r.json()["genes"] == ["G1"]


async def test_no_known_genes_is_404(client, dataset_manager):
    did = _register(dataset_manager)
    r = await client.get(
        f"{API}/datasets/{did}/expression/by-group",
        params={"genes": ["NOPE"], "groupby": "cluster", "layer": "X"},
    )
    assert r.status_code == 404


async def test_unknown_column_is_404(client, dataset_manager):
    did = _register(dataset_manager)
    r = await client.get(
        f"{API}/datasets/{did}/expression/by-group",
        params={"genes": ["G0"], "groupby": "nope", "layer": "X"},
    )
    assert r.status_code == 404
