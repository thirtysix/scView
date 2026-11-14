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


async def test_gene_trends_have_no_empty_bins(client, dataset_manager):
    # 35 cells over 100 bins by value: most bins held no cell (drawn first as 0, then as gaps).
    # Bins of equal cell counts hold every cell and leave none empty.
    did = _register(dataset_manager)
    r = await client.get(f"{API}/datasets/{did}/trajectory/dpt_pseudotime/genes", params={"genes": "G1"})
    g = r.json()["genes"]["G1"]
    counts, means = g["binned_counts"], g["binned_expression"]
    assert min(counts) >= 10 and None not in means
    a = ad.read_h5ad(Path(dataset_manager.data_dir) / "uploads" / did / "data.h5ad")
    expected = np.asarray(a.X)[5:, 1].mean()  # the cells with a finite pseudotime
    assert sum(m * c for m, c in zip(means, counts) if c) / 35 == pytest.approx(expected)


def test_equal_count_bins_hold_a_far_out_tail():
    from scview.core.pseudotime_genes import equal_count_bins

    rng = np.random.default_rng(1)
    pt = np.r_[rng.uniform(0, 0.016, 9797), rng.uniform(0.92, 1.0, 18)]  # Pijuan-Sala after Harmony
    bin_of, bin_pt, counts = equal_count_bins(pt, 20)
    assert len(bin_pt) == 20 and counts.min() >= 490 and counts.max() <= 491
    assert np.all(np.diff(bin_pt) >= 0) and bin_pt[0] < 0.001
    for b in (0, 19):
        assert np.median(pt[bin_of == b]) == bin_pt[b]
    # At least 10 cells a bin: 35 cells give 3 bins, not 100.
    assert len(equal_count_bins(np.linspace(0, 1, 35), 100)[1]) == 3


async def test_gene_trends_bin_by_cell_counts(client, dataset_manager):
    did = _register(dataset_manager)
    r = await client.get(f"{API}/datasets/{did}/trajectory/dpt_pseudotime/genes",
                         params={"genes": "G1", "n_bins": 10})
    body = r.json()
    assert body["n_bins"] == 3 and "equal cell counts" in body["binning"]
    g = body["genes"]["G1"]
    assert g["binned_counts"] == [12, 12, 11] and None not in g["binned_expression"]
