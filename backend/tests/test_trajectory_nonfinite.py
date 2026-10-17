"""Pseudotime with inf / NaN cells (unreachable or unassigned) must not break the panel."""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pytest

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
    a.obsm["X_umap"] = rng.normal(size=(40, 2)).astype("float32")
    a.write_h5ad(d / "data.h5ad")
    return "pt"


async def test_values_send_null_for_missing_and_finite_range(client, dataset_manager):
    did = _register(dataset_manager)
    r = await client.get(f"{API}/datasets/{did}/trajectory/dpt_pseudotime")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["values"][:5] == [None] * 5
    assert body["max"] == 1.0 and body["min"] > 0


async def test_colour_by_pseudotime_sends_nan_not_inf(client, dataset_manager):
    import pyarrow as pa

    did = _register(dataset_manager)
    r = await client.get(f"{API}/datasets/{did}/embeddings/X_umap", params={"color_by": "dpt_pseudotime"})
    assert r.status_code == 200, r.text
    table = pa.ipc.open_stream(r.content).read_all()
    vals = np.array(table.column(table.num_columns - 1).to_pylist(), dtype=float)
    assert np.isnan(vals[:5]).all() and np.isfinite(vals[5:]).all()


async def test_gene_trends_skip_missing_cells(client, dataset_manager):
    did = _register(dataset_manager)
    r = await client.get(f"{API}/datasets/{did}/trajectory/dpt_pseudotime/genes", params={"genes": "G1,NOPE"})
    assert r.status_code == 200, r.text
    g = r.json()["genes"]
    assert g["NOPE"] == {"found": False}
    assert sum(g["G1"]["binned_counts"]) == 35
    assert all(np.isfinite(g["G1"]["binned_pseudotime"]))


async def test_gene_trends_leave_empty_bins_as_gaps(client, dataset_manager):
    # 35 cells over 100 bins: most bins hold no cell, and were drawn as a mean of 0.
    did = _register(dataset_manager)
    r = await client.get(f"{API}/datasets/{did}/trajectory/dpt_pseudotime/genes", params={"genes": "G1"})
    g = r.json()["genes"]["G1"]
    counts, means = g["binned_counts"], g["binned_expression"]
    assert any(c == 0 for c in counts)
    assert all((m is None) == (c == 0) for m, c in zip(means, counts))
    a = ad.read_h5ad(Path(dataset_manager.data_dir) / "uploads" / did / "data.h5ad")
    expected = np.asarray(a.X)[5:, 1].mean()  # the cells with a finite pseudotime
    assert sum(m * c for m, c in zip(means, counts) if c) / 35 == pytest.approx(expected)
