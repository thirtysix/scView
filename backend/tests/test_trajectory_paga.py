"""PAGA group graph (core/trajectory.py compute_paga / paga_view + /trajectory/paga)."""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from scview.core import provenance
from scview.core.trajectory import (
    PAGA_KEY,
    TrajectoryError,
    compute_dpt,
    compute_paga,
    paga_view,
    read_paga,
)
from tests.test_trajectory_dpt import N_ISLAND, N_PATH, _path_adata, _register

API = "/api/v1"


def _with_graph() -> tuple[ad.AnnData, np.ndarray]:
    import scanpy as sc

    a, t = _path_adata()
    sc.pp.neighbors(a, n_neighbors=15, use_rep="X_pca")
    a.obsm["X_umap"] = a.obsm["X_pca"][:, :2].astype("float32")
    a.obsm["X_umap_3d"] = a.obsm["X_pca"][:, :3].astype("float32")
    return a, t


def _weights(record) -> dict[tuple[str, str], float]:
    g = record["groups"]
    return {(g[i], g[j]): w for i, j, w in record["edges"]}


def test_paga_links_neighbouring_stages_only_and_leaves_the_graph_alone():
    a, _ = _with_graph()
    graph = a.obsp["connectivities"].copy()
    record = compute_paga(a, "stage")

    assert record["groups"] == ["early", "island", "late", "mid"]
    assert record["sizes"] == [
        int((a.obs["stage"] == g).sum()) for g in record["groups"]
    ] and sum(record["sizes"]) == N_PATH + N_ISLAND
    w = _weights(record)
    # The path runs early -> mid -> late; nothing reaches the island. On a thin
    # path only the cells at a boundary link two stages, hence the small weights.
    assert w[("early", "mid")] > 0.02 and w[("late", "mid")] > 0.02
    assert ("early", "late") not in w
    assert not any("island" in pair for pair in w)
    g = record["groups"]
    assert sorted((g[i], g[j]) for i, j in record["tree"]) == [("early", "mid"), ("late", "mid")]
    assert record["graph"] == "existing neighbour graph (use_rep=X_pca, k=15)"

    # Saved as one uns record + provenance; the dataset's own graph is untouched
    # (same edges, same weights) and scanpy's uns['paga'] stays on the scratch copy.
    assert read_paga(a) == record and isinstance(a.uns[PAGA_KEY], str)
    assert (a.obsp["connectivities"] != graph).nnz == 0
    assert "paga" not in a.uns and "stage_sizes" not in a.uns
    assert provenance.read_provenance(a)["history"][-1]["step"] == "paga"


def test_paga_builds_a_graph_from_the_pca_when_there_is_none():
    a, _ = _path_adata()
    record = compute_paga(a, "stage")
    assert record["graph"] == "kNN graph on X_pca (k=15)"
    assert _weights(record)[("early", "mid")] > 0.02
    assert "neighbors" not in a.uns and "connectivities" not in a.obsp


def test_paga_leaves_out_unlabelled_cells_and_refuses_what_it_cannot_group():
    a, _ = _with_graph()
    stage = a.obs["stage"].astype(object)
    stage.iloc[:10] = np.nan
    a.obs["stage_gaps"] = pd.Categorical(stage)
    record = compute_paga(a, "stage_gaps")
    assert record["n_cells"] == a.n_obs - 10 and sum(record["sizes"]) == a.n_obs - 10

    a.obs["one"] = "x"
    a.obs["score"] = np.linspace(0, 1, a.n_obs)
    a.obs["barcode"] = [f"b{i}" for i in range(a.n_obs)]
    for column, match in [
        ("nope", "not found"),
        ("one", "at least two groups"),
        ("score", "continuous"),
        ("barcode", "at most 100"),
    ]:
        with pytest.raises(TrajectoryError, match=match):
            compute_paga(a, column)
    del a.obsm["X_pca"], a.uns["neighbors"]
    with pytest.raises(TrajectoryError, match="PCA"):
        compute_paga(a, "stage")


def test_paga_view_places_groups_and_notices_a_changed_grouping():
    a, _ = _with_graph()
    assert paga_view(a, "X_umap") == {"paga": None, "positions": None, "stale": None, "root_group": None}
    compute_paga(a, "stage")

    view = paga_view(a, "X_umap")
    assert view["stale"] is None and view["root_group"] is None
    for g, pos in zip(view["paga"]["groups"], view["positions"]):
        expected = np.median(a.obsm["X_umap"][(a.obs["stage"] == g).to_numpy()], axis=0)
        assert np.allclose(pos, expected, atol=1e-3)
    assert all(len(p) == 3 for p in paga_view(a, "X_umap_3d")["positions"])
    assert paga_view(a)["positions"] is None
    with pytest.raises(KeyError):
        paga_view(a, "X_tsne")

    # A branch starts at the group holding the pseudotime root cell.
    compute_dpt(a, "stage", "early")
    assert paga_view(a, "X_umap")["root_group"] == "early"

    # Relabelled cells: the drawn graph no longer describes the column.
    a.obs["stage"] = a.obs["stage"].cat.rename_categories({"late": "end"})
    assert "changed" in paga_view(a, "X_umap")["stale"]
    del a.obs["stage"]
    assert "gone" in paga_view(a, "X_umap")["stale"]


async def test_endpoints_save_paga_and_serve_it_placed(client, dataset_manager):
    a, _ = _with_graph()
    a.layers["spliced"] = a.X.copy()
    did, original = _register(dataset_manager, a)
    url = f"{API}/datasets/{did}/trajectory/paga"

    r = await client.get(url, params={"embedding": "X_umap"})
    assert r.status_code == 200, r.text  # not taken for a pseudotime column named "paga"
    assert r.json()["paga"] is None
    listed = (await client.get(f"{API}/datasets/{did}/trajectory")).json()
    assert listed["velocity_layers"] is False  # spliced alone is not enough

    r = await client.post(url, json={"groupby": "stage"})
    assert r.status_code == 200, r.text
    assert r.json()["groups"] == ["early", "island", "late", "mid"]

    body = (await client.get(url, params={"embedding": "X_umap"})).json()
    assert body["paga"]["groupby"] == "stage" and len(body["positions"]) == 4
    assert body["stale"] is None
    r = await client.get(url, params={"embedding": "X_nope"})
    assert r.status_code == 404

    # Written to the derived file; the upload is never touched.
    derived = Path(dataset_manager.data_dir) / "derived" / did / "data.h5ad"
    saved = ad.read_h5ad(derived)
    assert json.loads(saved.uns[PAGA_KEY])["groupby"] == "stage"
    assert PAGA_KEY not in ad.read_h5ad(original).uns
    assert not list(derived.parent.glob("*.partial"))


async def test_endpoint_rejects_bad_requests(client, dataset_manager):
    a, _ = _path_adata()
    a.obs["one"] = "x"
    did, _ = _register(dataset_manager, a)
    url = f"{API}/datasets/{did}/trajectory/paga"
    assert (await client.post(url, json={"groupby": "nope"})).status_code == 404
    r = await client.post(url, json={"groupby": "one"})
    assert r.status_code == 400 and "two groups" in r.json()["detail"]
    assert (await client.get(f"{API}/datasets/missing/trajectory/paga")).status_code == 404


async def test_listing_flags_velocity_layers(client, dataset_manager):
    a, _ = _path_adata()
    a.layers["spliced"] = a.X.copy()
    a.layers["unspliced"] = a.X.copy()
    did, _ = _register(dataset_manager, a)
    listed = (await client.get(f"{API}/datasets/{did}/trajectory")).json()
    assert listed["velocity_layers"] is True
