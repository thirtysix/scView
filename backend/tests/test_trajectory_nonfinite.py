"""Pseudotime with inf / NaN cells (unreachable or unassigned) must not break the panel."""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np

API = "/api/v1"


def _register(dm) -> str:
    d = Path(dm.data_dir) / "uploads" / "pt"
    d.mkdir(parents=True)
    (d / "metadata.json").write_text(json.dumps({"id": "pt", "name": "pt", "status": "ready"}))
    rng = np.random.default_rng(0)
    a = ad.AnnData(rng.poisson(2.0, (40, 5)).astype("float32"))
    a.var_names = [f"G{i}" for i in range(5)]
    pt = np.linspace(0, 1, 40)
    pt[:3] = np.inf
    pt[3:5] = np.nan
    a.obs["dpt_pseudotime"] = pt
    a.write_h5ad(d / "data.h5ad")
    return "pt"


async def test_values_send_null_for_missing_and_finite_range(client, dataset_manager):
    did = _register(dataset_manager)
    r = await client.get(f"{API}/datasets/{did}/trajectory/dpt_pseudotime")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["values"][:5] == [None] * 5
    assert body["max"] == 1.0 and body["min"] > 0


async def test_gene_trends_skip_missing_cells(client, dataset_manager):
    did = _register(dataset_manager)
    r = await client.get(f"{API}/datasets/{did}/trajectory/dpt_pseudotime/genes", params={"genes": "G1,NOPE"})
    assert r.status_code == 200, r.text
    g = r.json()["genes"]
    assert g["NOPE"] == {"found": False}
    assert sum(g["G1"]["binned_counts"]) == 35
    assert all(np.isfinite(g["G1"]["binned_pseudotime"]))
