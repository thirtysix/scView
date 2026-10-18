"""Distances between a column's groups in the graph's PC space
(core/cluster_distances.py + GET /clustering/distances)."""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from scview.core import cluster_distances as cd

API = "/api/v1"


def _adata(graph: dict | None = None) -> ad.AnnData:
    """Groups A and B close together in PC space, C far from both; an unused
    category D; two cells without a label."""
    rng = np.random.default_rng(0)
    centres = {"A": [0, 0, 0, 0], "B": [1, 0, 0, 0], "C": [10, 5, 0, 0]}
    labels = ["A"] * 30 + ["B"] * 20 + ["C"] * 10
    pca = np.array([centres[g] for g in labels], dtype="float32") + rng.normal(0, 0.05, (60, 4)).astype("float32")
    a = ad.AnnData(np.zeros((62, 3), dtype="float32"))
    a.obs_names = [f"c{i}" for i in range(62)]
    a.obs["cluster"] = pd.Categorical(labels + [None, None], categories=["C", "A", "D", "B"])
    a.obsm["X_pca"] = np.vstack([pca, np.zeros((2, 4), dtype="float32")])
    if graph is not None:
        a.uns["neighbors"] = {"params": graph}
    return a


def _centroid(a: ad.AnnData, key: str, group: str, n_pcs: int | None = None) -> np.ndarray:
    X = np.asarray(a.obsm[key], dtype=np.float64)[(a.obs["cluster"] == group).to_numpy()]
    return X.mean(axis=0)[:n_pcs]


def test_distances_are_between_the_group_centroids_and_order_puts_close_groups_together():
    a = _adata({"n_neighbors": 15})
    out = cd.group_distances(a, "cluster")
    # Category order kept, the empty D left out, the unlabelled cells counted.
    assert out["groups"] == ["C", "A", "B"]
    assert out["sizes"] == [10, 30, 20]
    assert out["n_left_out"] == 2
    D = np.array(out["distances"])
    assert D.shape == (3, 3) and np.allclose(np.diag(D), 0) and np.allclose(D, D.T)
    for i, g in enumerate(out["groups"]):
        for j, h in enumerate(out["groups"]):
            expected = np.linalg.norm(_centroid(a, "X_pca", g) - _centroid(a, "X_pca", h))
            assert D[i, j] == pytest.approx(expected, abs=1e-3)
    order = [out["groups"][i] for i in out["order"]]
    assert sorted(out["order"]) == [0, 1, 2]
    assert abs(order.index("A") - order.index("B")) == 1  # the two close groups side by side
    assert out["representation"] == "X_pca" and out["rep_source"] == "graph"
    assert out["n_pcs"] == 4 and out["linkage"] == "complete" and out["metric"] == "euclidean"
    json.dumps(out)  # plain JSON types


def test_the_space_is_the_graphs_harmony_pcs_and_its_n_pcs():
    a = _adata({"use_rep": "X_pca_harmony", "n_pcs": 2})
    a.obsm["X_pca_harmony"] = np.asarray(a.obsm["X_pca"]) * 2
    out = cd.group_distances(a, "cluster")
    assert out["representation"] == "X_pca_harmony" and out["rep_source"] == "graph" and out["n_pcs"] == 2
    expected = np.linalg.norm(_centroid(a, "X_pca_harmony", "A", 2) - _centroid(a, "X_pca_harmony", "C", 2))
    assert out["distances"][1][0] == pytest.approx(expected, abs=1e-3)


def test_representation_without_a_graph_or_with_its_space_missing():
    a = _adata()
    assert cd.representation(a)[0::2] == ("X_pca", "no_graph")
    a.obsm["X_pca_harmony"] = np.asarray(a.obsm["X_pca"])
    assert cd.representation(a)[0::2] == ("X_pca_harmony", "no_graph")
    # A graph recorded without use_rep was built on X_pca (scanpy's default),
    # even though Harmony PCs are in the file.
    a.uns["neighbors"] = {"params": {"n_neighbors": 15}}
    assert cd.representation(a)[0::2] == ("X_pca", "graph")
    a.uns["neighbors"] = {"params": {"use_rep": "X_scvi", "n_pcs": 2}}
    rep, X, how = cd.representation(a)
    assert (rep, how, X.shape[1]) == ("X_pca_harmony", "missing", 4)  # n_pcs belongs to the missing space


def test_the_tree_is_the_linkage_behind_the_order_and_the_correlation_is_of_the_centroids():
    from scipy.cluster.hierarchy import leaves_list

    a = _adata({"n_neighbors": 15})
    out = cd.group_distances(a, "cluster")
    tree = np.array(out["tree"], dtype=float)
    assert tree.shape == (2, 4)  # n - 1 merges of [a, b, height, size]
    assert leaves_list(tree).tolist() == out["order"]
    # The first merge joins A (index 1) and B (index 2) at their distance; the
    # root's height is the complete-linkage (largest) distance to C.
    assert sorted(tree[0, :2].tolist()) == [1, 2]
    assert tree[0, 2] == pytest.approx(out["distances"][1][2], abs=1e-3)
    assert tree[1, 2] == pytest.approx(max(out["distances"][0][1], out["distances"][0][2]), abs=1e-3)
    assert tree[-1, 3] == 3
    centroids = np.array([_centroid(a, "X_pca", g) for g in out["groups"]])
    R = np.array(out["correlation"])
    assert np.allclose(R, np.corrcoef(centroids), atol=1e-3)
    assert np.allclose(np.diag(R), 1) and np.allclose(R, R.T)
    assert out["correlation_method"] == "pearson"
    json.dumps(out)


def test_a_flat_centroid_correlates_zero_not_nan():
    R = cd.centroid_correlation(np.array([[1.0, 1.0, 1.0], [1.0, 2.0, 3.0], [3.0, 2.0, 1.0]]))
    assert R[0].tolist() == [1.0, 0.0, 0.0]
    assert R[1, 2] == pytest.approx(-1.0)


def test_errors_without_pcs_or_with_too_few_or_too_many_groups(monkeypatch):
    a = _adata()
    del a.obsm["X_pca"]
    with pytest.raises(cd.DistanceError, match="PCA"):
        cd.group_distances(a, "cluster")
    a = _adata()
    a.obs["one"] = "x"
    with pytest.raises(cd.DistanceError, match="at least two"):
        cd.group_distances(a, "one")
    monkeypatch.setattr(cd, "MAX_GROUPS", 2)
    with pytest.raises(cd.DistanceError, match="at most 2"):
        cd.group_distances(a, "cluster")
    with pytest.raises(cd.DistanceError, match="not found"):
        cd.group_distances(a, "nope")


def test_a_plain_column_and_cells_with_non_finite_pcs():
    a = _adata()
    a.obs["kind"] = np.where(np.arange(62) < 31, "late", "early")
    a.obsm["X_pca"][0, 0] = np.nan
    out = cd.group_distances(a, "kind")
    assert out["groups"] == ["early", "late"] and out["sizes"] == [31, 30]
    assert out["n_left_out"] == 1
    assert np.isfinite(out["distances"]).all()


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


def _register(dm, a: ad.AnnData) -> str:
    d = Path(dm.data_dir) / "uploads" / "dist"
    d.mkdir(parents=True)
    (d / "metadata.json").write_text(json.dumps({"id": "dist", "name": "dist", "status": "ready"}))
    a.write_h5ad(d / "data.h5ad")
    return "dist"


async def test_distances_endpoint(client, dataset_manager):
    a = _adata({"use_rep": "X_pca_harmony"})
    a.obsm["X_pca_harmony"] = np.asarray(a.obsm["X_pca"])
    a.obs["one"] = "x"
    did = _register(dataset_manager, a)
    r = await client.get(f"{API}/datasets/{did}/clustering/distances", params={"column": "cluster"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["groups"] == ["C", "A", "B"] and body["representation"] == "X_pca_harmony"
    assert len(body["distances"]) == 3 and len(body["order"]) == 3
    assert len(body["tree"]) == 2 and len(body["correlation"]) == 3
    r = await client.get(f"{API}/datasets/{did}/clustering/distances", params={"column": "nope"})
    assert r.status_code == 404
    r = await client.get(f"{API}/datasets/{did}/clustering/distances", params={"column": "one"})
    assert r.status_code == 400 and "at least two" in r.json()["detail"]
    r = await client.get(f"{API}/datasets/{did}/clustering/distances")
    assert r.status_code == 422
    r = await client.get(f"{API}/datasets/missing/clustering/distances", params={"column": "cluster"})
    assert r.status_code == 404
