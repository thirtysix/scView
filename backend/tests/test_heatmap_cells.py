"""Per-cell marker heatmap: stratified, seeded sample of cells (C034)."""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
import scipy.sparse as sp

from scview.core import category_order, heatmap_cells as hc

API = "/api/v1"


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def test_quota_is_the_group_size_capped_at_per_group():
    assert hc.sample_quota([10, 3, 0], per_group=5, max_cells=100).tolist() == [5, 3, 0]


def test_quota_scales_down_in_proportion_over_max_cells():
    # want 200 + 200 + 10 = 410 > 200: exact shares 97.56, 97.56, 4.88; floors
    # 97 + 97 + 4 = 198, and the 2 spare go to the largest remainders (4.88 -> 5,
    # then the earlier of the two tied 97.56 -> 98).
    q = hc.sample_quota([1000, 1000, 10], per_group=200, max_cells=200)
    assert q.tolist() == [98, 97, 5]
    assert q.sum() == 200


def test_quota_keeps_one_cell_of_every_non_empty_group():
    # 5000 * 100 / 5001 -> 99.98 and 0.02: the small group still gets a cell.
    assert hc.sample_quota([10000, 1, 0], per_group=5000, max_cells=100).tolist() == [99, 1, 0]


def _labels(sizes: dict[str, int], categories: list[str] | None = None) -> pd.Series:
    values = [g for g, n in sizes.items() for _ in range(n)]
    rng = np.random.default_rng(1)
    values = list(np.asarray(values, dtype=object)[rng.permutation(len(values))])  # interleave groups
    return pd.Series(pd.Categorical(values, categories=categories or list(sizes)))


def test_sample_is_seeded_and_deterministic():
    col = _labels({"A": 300, "B": 50})
    groups, codes = hc.group_codes(col)
    quota = hc.sample_quota([300, 50], 100, 5000)
    one = hc.sample_cells(codes, groups, quota, seed=0)
    assert np.array_equal(one, hc.sample_cells(codes, groups, quota, seed=0))
    assert not np.array_equal(one, hc.sample_cells(codes, groups, quota, seed=1))
    # A's 100 cells, then all of B's 50; ascending within each group.
    assert len(one) == 150
    a, b = one[:100], one[100:]
    assert (codes[a] == 0).all() and (codes[b] == 1).all()
    assert (np.diff(a) > 0).all() and (np.diff(b) > 0).all()
    assert len(set(one.tolist())) == 150


def test_a_smaller_sample_is_part_of_a_larger_one():
    col = _labels({"A": 300, "B": 300})
    groups, codes = hc.group_codes(col)
    small = hc.sample_cells(codes, groups, np.array([100, 100]), seed=7)
    large = hc.sample_cells(codes, groups, np.array([200, 200]), seed=7)
    assert set(small.tolist()) <= set(large.tolist())
    # and a group's sample does not depend on the other group's quota
    other = hc.sample_cells(codes, groups, np.array([100, 5]), seed=7)
    assert np.array_equal(other[:100], small[:100])


def test_groups_follow_the_stored_category_order():
    adata = ad.AnnData(np.zeros((6, 1), dtype=np.float32))
    adata.obs["cl"] = pd.Categorical(["A", "B", "A", "B", "A", "B"], categories=["B", "A", "empty"])
    groups, _ = hc.group_codes(adata.obs["cl"])
    assert groups == ["B", "A"]  # stored order; the empty category is left out
    category_order.reorder(adata, "cl", order=["A", "empty", "B"])
    groups, codes = hc.group_codes(adata.obs["cl"])
    assert groups == ["A", "B"]
    rows = hc.sample_cells(codes, groups, np.array([3, 3]), seed=0)
    assert rows.tolist() == [0, 2, 4, 1, 3, 5]


def test_a_text_column_is_ordered_naturally_and_a_numeric_one_refused():
    groups, codes = hc.group_codes(pd.Series(["10", "2", None, "2"], dtype=object))
    assert groups == ["2", "10"]
    assert codes.tolist() == [1, 0, -1, 0]  # no label: in no group
    with pytest.raises(hc.HeatmapCellsError, match="numeric"):
        hc.group_codes(pd.Series([0.5, 1.5]), "score")


# ---------------------------------------------------------------------------
# Values
# ---------------------------------------------------------------------------


def test_zscore_is_per_gene_and_clipped_at_2_5():
    # Nine 0s and one 100: mean 10, sd (ddof=1) sqrt(9000 / 9) = 31.62, so the
    # 100 is at 2.85 (clipped to 2.5) and each 0 at -10 / 31.62 = -0.316.
    v = np.zeros((10, 2), dtype=np.float32)
    v[9, 0] = 100
    v[:, 1] = 3.0  # a flat gene: 0 throughout, not NaN
    z = hc.zscore(v)
    assert z[9, 0] == pytest.approx(2.5)
    assert z[0, 0] == pytest.approx(-10 / np.sqrt(1000), rel=1e-5)
    assert (z[:, 1] == 0).all()
    assert hc.ZSCORE_CLIP == 2.5


def _matrix(n: int = 40) -> np.ndarray:
    rng = np.random.default_rng(3)
    X = rng.poisson(1.0, size=(n, 5)).astype(np.float32)
    X[X < 1] = 0
    return X


@pytest.mark.parametrize("make", [np.asarray, sp.csr_matrix, sp.csc_matrix], ids=["dense", "csr", "csc"])
def test_values_are_the_sampled_cells_in_plot_order(make):
    X = _matrix()
    col = _labels({"A": 25, "B": 15}, categories=["B", "A"])
    out = hc.heatmap_cells(make(X), [f"G{i}" for i in range(5)], col, ["G3", "G0"],
                           per_group=10, scale="none", layer_range=(0.0, 5.0))
    rows = out["cells"]
    assert out["meta"]["groups"] == ["B", "A"]
    assert out["codes"].tolist() == [0] * 10 + [1] * 10
    # genes in the order asked for, values straight from the matrix
    assert np.array_equal(out["values"], X[rows][:, [3, 0]])
    assert out["meta"]["value_range"] == [0.0, 5.0]
    assert out["meta"]["n_sampled"] == [10, 10] and out["meta"]["n_total"] == [15, 25]


def test_dense_and_sparse_give_the_same_zscores():
    X = _matrix()
    col = _labels({"A": 25, "B": 15})
    names = [f"G{i}" for i in range(5)]
    dense = hc.heatmap_cells(X, names, col, names, per_group=10)
    sparse = hc.heatmap_cells(sp.csr_matrix(X), names, col, names, per_group=10)
    assert np.array_equal(dense["cells"], sparse["cells"])
    assert np.allclose(dense["values"], sparse["values"])
    assert dense["meta"]["value_range"] == [-2.5, 2.5]
    assert np.abs(dense["values"]).max() <= 2.5


def test_gene_list_rules():
    X = _matrix()
    col = _labels({"A": 25, "B": 15})
    names = [f"G{i}" for i in range(5)]
    out = hc.heatmap_cells(X, names, col, ["G1", "NOPE", "G1"])
    assert out["meta"]["genes"] == ["G1"] and out["meta"]["missing_genes"] == ["NOPE"]
    with pytest.raises(hc.HeatmapCellsError, match="None of the genes"):
        hc.heatmap_cells(X, names, col, ["NOPE"])
    with pytest.raises(hc.HeatmapCellsError, match="At most 100"):
        hc.heatmap_cells(X, names, col, [f"X{i}" for i in range(101)])


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


def _register(dm, did: str = "hm1") -> tuple[str, ad.AnnData]:
    """30 cells, X (scaled, 2 genes) differs from .raw (log-normalised, 3 genes)."""
    d = Path(dm.data_dir) / "uploads" / did
    d.mkdir(parents=True, exist_ok=True)
    (d / "metadata.json").write_text(json.dumps({"id": did, "name": did, "status": "ready"}))
    rng = np.random.default_rng(0)
    raw = rng.poisson(2.0, size=(30, 3)).astype(np.float32)
    full = ad.AnnData(sp.csr_matrix(raw))
    full.var_names = ["CD3E", "MS4A1", "LYZ"]
    a = ad.AnnData(np.full((30, 2), -1.0, dtype=np.float32))
    a.var_names = ["CD3E", "MS4A1"]
    a.obs_names = [f"c{i}" for i in range(30)]
    a.raw = full
    a.obs["cell_type"] = pd.Categorical(["T"] * 20 + ["B"] * 10, categories=["B", "T"])
    a.obs["score"] = np.linspace(0, 1, 30)
    a.write_h5ad(d / "data.h5ad")
    return did, a


def _decode(content: bytes) -> tuple[pa.Table, dict]:
    table = pa.ipc.open_stream(content).read_all()
    return table, json.loads(table.schema.metadata[b"scview"])


async def test_endpoint_returns_arrow_with_its_facts(client, dataset_manager):
    did, a = _register(dataset_manager)
    url = f"{API}/datasets/{did}/expression/heatmap-cells"
    r = await client.get(url, params={"genes": "LYZ,NOPE,CD3E", "groupby": "cell_type",
                                      "per_group": 5, "scale": "none"})
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("application/vnd.apache.arrow.stream")
    table, meta = _decode(r.content)
    assert meta["genes"] == ["LYZ", "CD3E"] and meta["missing_genes"] == ["NOPE"]
    assert meta["groups"] == ["B", "T"]
    assert meta["n_sampled"] == [5, 5] and meta["n_total"] == [10, 20] and meta["n_cells"] == 30
    assert meta["layer"] == "raw" and meta["scale"] == "none" and meta["seed"] == 0
    cells = table.column(0).to_numpy()
    assert table.column(1).to_pylist() == [0] * 5 + [1] * 5
    assert (cells[:5] >= 20).all() and (cells[5:] < 20).all()  # B cells are rows 20-29
    # values from the log-normalised .raw (LYZ is not even in X), not from X
    raw = a.raw.X.toarray()
    assert np.array_equal(table.column(2).to_numpy(), raw[cells, 2])
    assert np.array_equal(table.column(3).to_numpy(), raw[cells, 0])
    # scale "none": the layer's fixed range, as the dot plot's
    rng = (await client.get(f"{API}/datasets/{did}/expression/by-group",
                            params={"genes": ["CD3E"], "groupby": "cell_type"})).json()["scale_range"]
    assert meta["value_range"] == rng

    again = await client.get(url, params={"genes": "LYZ,NOPE,CD3E", "groupby": "cell_type",
                                          "per_group": 5, "scale": "none"})
    assert again.content == r.content  # same seed, same cells

    z = await client.get(url, params={"genes": "LYZ", "groupby": "cell_type"})
    _, zmeta = _decode(z.content)
    assert zmeta["scale"] == "zscore" and zmeta["value_range"] == [-2.5, 2.5]


async def test_endpoint_errors(client, dataset_manager):
    did, _ = _register(dataset_manager)
    url = f"{API}/datasets/{did}/expression/heatmap-cells"
    r = await client.get(url, params={"genes": "NOPE,ALSO", "groupby": "cell_type"})
    assert r.status_code == 400 and "NOPE" in r.json()["detail"]
    r = await client.get(url, params={"genes": ",".join(f"G{i}" for i in range(101)), "groupby": "cell_type"})
    assert r.status_code == 400
    r = await client.get(url, params={"genes": "CD3E", "groupby": "score"})
    assert r.status_code == 400 and "numeric" in r.json()["detail"]
    r = await client.get(url, params={"genes": "CD3E", "groupby": "nope"})
    assert r.status_code == 404
    r = await client.get(url, params={"genes": "CD3E", "groupby": "cell_type", "scale": "minmax"})
    assert r.status_code == 422
