"""Per-cell plot reliability (core/embedding_reliability.py): placement (graph
neighbours within the cell's nearest 5% of cells on the plot) and the strict
exact-neighbour score.

Expected values are known in advance (a faithful layout keeps every neighbour, a
shuffled one scores about the window's share of the cells, a cell moved into
another island scores 0) or computed independently here with plain numpy (all
pairwise distances, argsort).
"""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
import scipy.sparse as sp

from scview.core import embedding_reliability as er
from scview.core.anndata_adaptor import AnnDataAdaptor
from scview.core.assistant import build_grounding_context

API = "/api/v1"


def _blobs(n_per: int = 150, groups: int = 2, dims: int = 10, seed: int = 0, sparse: bool = False):
    """``groups`` well-separated blobs. X_pca: 2D positions padded with zero
    columns, so distances in X_pca equal distances in the 2D layout; the graph
    is scanpy's (exact for this few cells)."""
    rng = np.random.default_rng(seed)
    pos = np.concatenate([rng.normal(size=(n_per, 2)) + 40 * g for g in range(groups)])
    n = pos.shape[0]
    pca = np.zeros((n, dims))
    pca[:, :2] = pos
    X = rng.poisson(1.0, size=(n, 20)).astype(np.float32)
    a = ad.AnnData(X=sp.csr_matrix(X) if sparse else X)
    a.obs_names = [f"c{i}" for i in range(n)]
    a.obs["group"] = pd.Categorical(np.repeat([f"g{g}" for g in range(groups)], n_per))
    a.obsm["X_pca"] = pca
    sc.pp.neighbors(a, n_neighbors=15, use_rep="X_pca")
    # Rotated and scaled: the same neighbours.
    theta = 0.7
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    a.obsm["X_umap"] = 3.0 * pos @ rot.T
    a.uns["scview_active_clustering"] = "group"
    return a


def _knn_brute(X: np.ndarray, k: int) -> np.ndarray:
    """The k nearest other cells of every cell, by all pairwise distances."""
    d = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d, np.inf)
    return np.argsort(d, axis=1, kind="stable")[:, :k]


def _placement_brute(graph_nn: np.ndarray, Y: np.ndarray, window: int) -> np.ndarray:
    """Share of each cell's graph neighbours no farther on the plot than its
    window-th nearest other cell, by all pairwise distances."""
    d = np.sqrt(((Y[:, None, :] - Y[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    r = np.sort(d, axis=1)[:, window - 1]
    return np.array([np.mean(d[i, graph_nn[i]] <= r[i]) for i in range(len(Y))])


def test_faithful_layout_keeps_every_neighbour():
    a = _blobs()
    res = er.compute(a, "X_umap")
    assert res.meta["k"] == 14 and res.meta["graph"]["source"] == "stored"  # n_neighbors 15 - self
    assert np.all(res.scores == 1.0) and np.all(res.exact == 1.0)
    # The window: 5% of the 299 other cells, 15 cells (at least k).
    assert res.meta["window"] == {"share": 0.05, "cells": 15, "radius": "exact", "ref_cells": None}
    assert res.meta["chance"] == pytest.approx(15 / 299)
    assert res.meta["exact_chance"] == pytest.approx(14 / 299)
    assert res.meta["n_dims"] == 2 and res.meta["dims"] == [0, 1]


@pytest.mark.parametrize("sparse", [False, True])
def test_shuffled_layout_scores_about_chance(sparse):
    a = _blobs(n_per=500, groups=4, sparse=sparse)
    perm = np.random.default_rng(5).permutation(a.n_obs)
    a.obsm["X_shuffled"] = a.obsm["X_umap"][perm]
    res = er.compute(a, "X_shuffled")
    window = round(0.05 * 1999)
    chance = window / 1999
    assert res.meta["window"]["cells"] == window and res.meta["chance"] == pytest.approx(chance)
    # Each graph neighbour lands in the 100-cell window with probability ~0.05.
    assert res.scores.mean() == pytest.approx(chance, abs=0.01)
    assert np.mean(res.scores < er.LOW) > 0.99
    exact_chance = 14 / 1999
    assert res.exact.mean() == pytest.approx(exact_chance, abs=3 * exact_chance)
    # The built-in random-placement baseline is a shuffle too, of the faithful layout.
    faithful = er.summarise(a, er.compute(a, "X_umap"))
    assert faithful["overall"]["median"] == 1.0 and faithful["exact"]["median"] == 1.0
    assert faithful["shuffled"]["mean"] == pytest.approx(chance, abs=0.01)


def test_group_moved_into_another_is_low_and_listed_first():
    a = _blobs(n_per=150, groups=3)
    lay = a.obsm["X_umap"].copy()
    g = a.obs["group"].to_numpy()
    b_cells = np.flatnonzero(g == "g1")
    c_cells = np.flatnonzero(g == "g2")
    rng = np.random.default_rng(1)
    # Each g2 cell is put beside a random g1 cell: inside g1's region, its own neighbours scattered.
    lay[c_cells] = lay[rng.choice(b_cells, size=c_cells.size)] + rng.normal(scale=0.01, size=(c_cells.size, 2))
    a.obsm["X_moved"] = lay
    summary = er.summarise(a, er.compute(a, "X_moved"))
    assert summary["group_by"] == "group" and summary["group_reason"] == "the active clustering"
    order = [r["group"] for r in summary["groups"]]
    assert order == ["g2", "g1", "g0"]
    by = {r["group"]: r for r in summary["groups"]}
    assert by["g0"]["median"] == 1.0 and by["g0"]["share_low"] == 0.0
    assert by["g2"]["median"] < 0.2 and by["g2"]["share_low"] > 0.9
    assert 0.5 <= by["g1"]["median"] < 1.0  # crowded by the newcomers, still near its own
    assert by["g0"]["exact_median"] == 1.0 and by["g2"]["exact_median"] < 0.2


def test_a_cell_moved_into_another_island_scores_zero():
    a = _blobs(n_per=150, groups=2)
    lay = a.obsm["X_umap"].copy()
    lay[7] = lay[200] + 0.01  # cell 7 (island g0) put inside island g1
    a.obsm["X_stray"] = lay
    res = er.compute(a, "X_stray")
    assert res.scores[7] == 0.0
    others = np.delete(res.scores, 7)
    assert np.median(others) == 1.0 and others.min() >= 13 / 14  # one neighbour gone or crowded out


def test_scores_match_a_plain_numpy_computation():
    rng = np.random.default_rng(3)
    n = 220
    a = ad.AnnData(X=np.zeros((n, 3), dtype=np.float32))
    a.obsm["X_pca"] = rng.normal(size=(n, 6))
    sc.pp.neighbors(a, n_neighbors=11, use_rep="X_pca")
    a.obsm["X_umap"] = rng.normal(size=(n, 2))  # unrelated layout: partial overlap at best
    a.obs["grp"] = pd.Categorical(rng.choice(["x", "y", "z"], size=n))
    k = 10
    window = round(0.05 * (n - 1))  # 11 cells
    g = _knn_brute(a.obsm["X_pca"], k)
    p = _knn_brute(a.obsm["X_umap"], k)
    exact = np.array([len(set(g[i]) & set(p[i])) / k for i in range(n)])
    placement = _placement_brute(g, a.obsm["X_umap"], window)
    res = er.compute(a, "X_umap")
    assert res.meta["k"] == k and res.meta["window"]["cells"] == window
    np.testing.assert_allclose(res.scores, placement, atol=1e-6)
    np.testing.assert_allclose(res.exact, exact, atol=1e-6)
    assert (placement >= exact).all()  # the window holds the k nearest
    s = er.summarise(a, res, "grp")
    assert s["overall"]["median"] == pytest.approx(np.median(placement))
    assert s["overall"]["share_low"] == pytest.approx(np.mean(placement < 0.5))
    assert s["exact"]["median"] == pytest.approx(np.median(exact))
    for row in s["groups"]:
        mask = a.obs["grp"].to_numpy() == row["group"]
        v = placement[mask]
        assert row["n"] == v.size
        assert row["median"] == pytest.approx(np.median(v))
        assert row["q1"] == pytest.approx(np.quantile(v, 0.25))
        assert row["q3"] == pytest.approx(np.quantile(v, 0.75))
        assert row["share_low"] == pytest.approx(np.mean(v < 0.5))
        assert row["exact_median"] == pytest.approx(np.median(exact[mask]))
    medians = [r["median"] for r in s["groups"]]
    assert medians == sorted(medians)  # worst first
    # A smaller k: the graph's 5 nearest, the plot's 5 nearest, the same window.
    res5 = er.compute(a, "X_umap", k=5)
    g5, p5 = _knn_brute(a.obsm["X_pca"], 5), _knn_brute(a.obsm["X_umap"], 5)
    np.testing.assert_allclose(res5.exact, [len(set(g5[i]) & set(p5[i])) / 5 for i in range(n)], atol=1e-6)
    np.testing.assert_allclose(res5.scores, _placement_brute(g5, a.obsm["X_umap"], window), atol=1e-6)
    # Another window share.
    res20 = er.compute(a, "X_umap", window=0.2)
    np.testing.assert_allclose(res20.scores, _placement_brute(g, a.obsm["X_umap"], round(0.2 * (n - 1))), atol=1e-6)


def test_window_estimated_from_a_sample_above_the_exact_limit(monkeypatch):
    monkeypatch.setattr(er, "REF_CELLS", 500)
    rng = np.random.default_rng(6)
    n = 1200
    a = ad.AnnData(X=np.zeros((n, 2), dtype=np.float32))
    a.obsm["X_pca"] = rng.normal(size=(n, 4))
    sc.pp.neighbors(a, n_neighbors=15, use_rep="X_pca")
    a.obsm["X_umap"] = a.obsm["X_pca"][:, :2] + rng.normal(scale=0.3, size=(n, 2))  # roughly kept
    exact = er.compute(a, "X_umap")
    est = er.compute(a, "X_umap", exact_max=1000)
    assert exact.meta["window"]["radius"] == "exact"
    assert est.meta["window"] == {"share": 0.05, "cells": 60, "radius": "estimated", "ref_cells": 500}
    assert any("seeded sample of 500 cells" in m for m in est.meta["notes"])
    # The estimated window holds about 60 cells, so the scores barely move.
    assert np.mean(np.abs(est.scores - exact.scores)) < 0.05
    assert abs(np.median(est.scores) - np.median(exact.scores)) <= 1 / 14


def test_self_stored_as_explicit_zero_is_not_a_neighbour():
    # Some scanpy versions store the cell itself in the distances as an explicit 0
    # (the Kang file: 16 entries per row, 15 neighbours).
    a = _blobs()
    d = a.obsp["distances"].tocsr()
    n = a.n_obs
    rows = np.concatenate([np.repeat(np.arange(n), np.diff(d.indptr)), np.arange(n)])
    cols = np.concatenate([d.indices, np.arange(n)])
    vals = np.concatenate([d.data, np.zeros(n)])
    with_self = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    assert with_self.nnz == d.nnz + n  # the zeros are stored, not dropped
    a.obsp["distances"] = with_self
    res = er.compute(a, "X_umap")
    assert res.meta["k"] == 14 and np.all(res.scores == 1.0) and np.all(res.exact == 1.0)


def test_drop_self_with_tied_duplicates():
    # Row 0's cell has a duplicate (cell 1) at distance 0 that came back first.
    ind = np.array([[1, 0, 2], [1, 0, 3], [4, 5, 6]])
    out = er.drop_self(ind, np.array([0, 1, 9]), 2)
    assert out.tolist() == [[1, 2], [0, 3], [4, 5]]
    # Two cells at identical plot coordinates still each find the other, not themselves.
    a = _blobs()
    lay = a.obsm["X_umap"].copy()
    lay[1] = lay[0]
    a.obsm["X_dup"] = lay
    res = er.compute(a, "X_dup")
    assert np.isfinite(res.scores).all() and res.scores.max() <= 1.0
    assert np.isfinite(res.exact).all() and res.exact.max() <= 1.0


def test_k_above_the_graph_is_rejected_and_bad_k():
    a = _blobs()
    with pytest.raises(er.ReliabilityError, match="at most 14"):
        er.compute(a, "X_umap", k=20)
    with pytest.raises(er.ReliabilityError):
        er.compute(a, "X_umap", k=0)
    assert er.compute(a, "X_umap", k=14).meta["k"] == 14


def test_3d_layout():
    a = _blobs()
    pos3 = np.column_stack([a.obsm["X_pca"][:, :2], np.zeros(a.n_obs)])
    a.obsm["X_umap_3d"] = pos3 + 5.0
    res = er.compute(a, "X_umap_3d")
    assert res.meta["n_dims"] == 3 and res.meta["dims"] == [0, 1, 2]
    assert np.all(res.scores == 1.0)
    # A 3D layout that keeps the first two dims only of a graph built on three.
    rng = np.random.default_rng(2)
    b = ad.AnnData(X=np.zeros((300, 2), dtype=np.float32))
    b.obsm["X_pca"] = rng.normal(size=(300, 3))
    sc.pp.neighbors(b, n_neighbors=15, use_rep="X_pca")
    b.obsm["X_umap_3d"] = b.obsm["X_pca"].copy()
    b.obsm["X_flat"] = np.column_stack([b.obsm["X_pca"][:, :2], np.zeros(300)])
    full = er.compute(b, "X_umap_3d")
    flat = er.compute(b, "X_flat")
    assert np.all(full.scores == 1.0) and flat.scores.mean() < 0.95 and flat.exact.mean() < full.exact.mean()


def test_pca_slice_scores_lower_than_a_faithful_layout():
    rng = np.random.default_rng(4)
    n = 400
    a = ad.AnnData(X=np.zeros((n, 2), dtype=np.float32))
    a.obsm["X_pca"] = rng.normal(size=(n, 10))  # ten equally strong directions
    sc.pp.neighbors(a, n_neighbors=15, use_rep="X_pca")
    res = er.compute(a, "X_pca", dims=[0, 1])
    assert res.meta["dims"] == [0, 1] and np.median(res.scores) < 0.5 and np.median(res.exact) < 0.3
    # No dims on a many-column embedding: the first two, as the plot shows them.
    res2 = er.compute(a, "X_pca")
    assert res2.meta["dims"] == [0, 1] and any("first two" in m for m in res2.meta["notes"])
    np.testing.assert_array_equal(res.scores, res2.scores)


def test_no_stored_graph_builds_one_in_the_named_representation():
    a = _blobs()
    a.obsm["X_other"] = a.obsm["X_pca"].copy()
    a.obsm["X_pca"] = np.random.default_rng(0).normal(size=a.obsm["X_pca"].shape)  # a decoy
    del a.obsp["distances"], a.obsp["connectivities"]
    a.uns["neighbors"] = {"params": {"use_rep": "X_other", "n_pcs": 2, "metric": "euclidean"}}
    res = er.compute(a, "X_umap")
    g = res.meta["graph"]
    assert g["source"] == "computed" and g["use_rep"] == "X_other" and g["n_pcs"] == 2
    assert res.meta["k"] == 14 and np.all(res.scores == 1.0)
    assert any("No neighbour graph is stored" in m for m in res.meta["notes"])
    # No graph and no parameters: X_pca, all columns, k as asked.
    del a.uns["neighbors"]
    a.obsm["X_pca"] = a.obsm["X_other"]
    res = er.compute(a, "X_umap", k=8)
    assert res.meta["graph"]["use_rep"] == "X_pca" and res.meta["k"] == 8
    assert np.all(res.scores == 1.0)
    # Nothing to build one from.
    del a.obsm["X_pca"]
    with pytest.raises(er.ReliabilityError, match="neighbors step"):
        er.compute(a, "X_umap")


def test_custom_key_added_graph():
    a = _blobs()
    sc.pp.neighbors(a, n_neighbors=10, use_rep="X_pca", key_added="mine")
    del a.uns["neighbors"], a.obsp["distances"], a.obsp["connectivities"]
    res = er.compute(a, "X_umap")
    assert res.meta["graph"]["key"] == "mine" and res.meta["k"] == 9
    assert np.all(res.scores == 1.0)


def test_sample_above_the_cap():
    a = _blobs(n_per=150, groups=2)
    res = er.compute(a, "X_umap", max_cells=200, sample_cells=120)
    assert res.meta["sampled"] and res.meta["n_scored"] == 120
    assert np.isnan(res.scores).sum() == a.n_obs - 120
    assert np.all(res.scores[np.isfinite(res.scores)] == 1.0)
    assert any("seeded sample of 120" in m for m in res.meta["notes"])
    again = er.compute(a, "X_umap", max_cells=200, sample_cells=120)
    np.testing.assert_array_equal(np.isnan(res.scores), np.isnan(again.scores))  # seeded
    s = er.summarise(a, res)
    assert sum(r["n"] for r in s["groups"]) == 120


def test_bad_inputs():
    a = _blobs()
    with pytest.raises(KeyError):
        er.compute(a, "X_nope")
    a.obsm["X_line"] = a.obsm["X_umap"][:, :1]
    with pytest.raises(er.ReliabilityError, match="fewer than two"):
        er.compute(a, "X_line")
    lay = a.obsm["X_umap"].copy()
    lay[3, 0] = np.nan
    a.obsm["X_holes"] = lay
    with pytest.raises(er.ReliabilityError, match="missing coordinates"):
        er.compute(a, "X_holes")
    with pytest.raises(KeyError):
        er.summarise(a, er.compute(a, "X_umap"), "no_such_column")
    a.obs["barcode"] = [f"b{i}" for i in range(a.n_obs)]
    with pytest.raises(er.ReliabilityError, match="groups"):
        er.summarise(a, er.compute(a, "X_umap"), "barcode")


def test_cached_on_the_loaded_dataset():
    a = _blobs()
    r1, hit1 = er.scored(a, "X_umap")
    r2, hit2 = er.scored(a, "X_umap")
    assert not hit1 and hit2 and r1 is r2
    assert er.latest(a) is r1
    r3, hit3 = er.scored(a, "X_umap", k=5)
    assert not hit3 and er.latest(a) is r3
    # New coordinates under the same name: a new entry.
    a.obsm["X_umap"] = a.obsm["X_umap"] * 2
    assert not er.scored(a, "X_umap")[1]
    assert er.latest(_blobs()) is None


# ---------------------------------------------------------------------------
# Endpoint and co-pilot
# ---------------------------------------------------------------------------


def _register(dm, adata, did: str = "ds") -> str:
    d = Path(dm.data_dir) / "uploads" / did
    d.mkdir(parents=True, exist_ok=True)
    (d / "metadata.json").write_text(json.dumps({"id": did, "name": did, "status": "ready"}))
    adata.write_h5ad(d / "data.h5ad")
    return did


async def test_endpoint(client, dataset_manager):
    a = _blobs(n_per=100, groups=3)
    lay = a.obsm["X_umap"].copy()
    g = a.obs["group"].to_numpy()
    lay[g == "g2"] = lay[np.flatnonzero(g == "g1")] + 0.01  # g2 laid over g1
    a.obsm["X_moved"] = lay
    did = _register(dataset_manager, a)
    url = f"{API}/datasets/{did}/embeddings/X_moved/reliability"
    r = await client.get(url)
    assert r.status_code == 200, r.text
    d = r.json()
    assert len(d["scores"]) == a.n_obs and d["k"] == 14 and d["n_scored"] == a.n_obs
    assert d["groups"][0]["group"] == "g2" and d["group_by"] == "group"
    assert d["window"]["cells"] == 15 and d["chance"] == pytest.approx(15 / 299)
    assert d["exact"]["chance"] == pytest.approx(14 / 299) and "exact_median" in d["groups"][0]
    assert {"median", "mean"} <= set(d["shuffled"]) and d["low_threshold"] == 0.5
    assert d["cached"] is False
    s = await client.get(f"{url}/summary", params={"group_by": "group"})
    assert s.status_code == 200 and "scores" not in s.json() and s.json()["cached"] is True
    assert s.json()["groups"] == d["groups"]
    # 3D, a PCA pair, k.
    r = await client.get(f"{API}/datasets/{did}/embeddings/X_pca/reliability", params={"dims": "0,1"})
    assert r.status_code == 200 and r.json()["dims"] == [0, 1]
    r = await client.get(url, params={"k": 5})
    assert r.status_code == 200 and r.json()["k"] == 5
    # Errors.
    assert (await client.get(url, params={"k": 20})).status_code == 422
    assert (await client.get(url, params={"k": 0})).status_code == 422
    assert (await client.get(url, params={"group_by": "nope"})).status_code == 404
    assert (await client.get(f"{API}/datasets/{did}/embeddings/X_nope/reliability")).status_code == 404
    assert (await client.get(f"{API}/datasets/nope/embeddings/X_umap/reliability")).status_code == 404
    assert (await client.get(f"{url}/summary", params={"dims": "0,0"})).status_code == 400


async def test_endpoint_sampled_cells_are_null(client, dataset_manager, monkeypatch):
    monkeypatch.setattr(er, "MAX_CELLS", 100)
    monkeypatch.setattr(er, "SAMPLE_CELLS", 60)
    a = _blobs(n_per=80, groups=2)
    did = _register(dataset_manager, a, "big")
    r = await client.get(f"{API}/datasets/{did}/embeddings/X_umap/reliability")
    assert r.status_code == 200, r.text
    scores = r.json()["scores"]
    assert sum(v is None for v in scores) == a.n_obs - 60 and r.json()["sampled"] is True


def test_copilot_fact_only_once_computed(tmp_path: Path):
    a = _blobs(n_per=100, groups=3)
    p = tmp_path / "d.h5ad"
    a.write_h5ad(p)
    adaptor = AnnDataAdaptor(str(p))
    _, sources = build_grounding_context(adaptor)
    assert er.TAG not in {s.ref for s in sources}
    er.for_adaptor(adaptor, "X_umap")
    context, sources = build_grounding_context(adaptor)
    fact = next(s for s in sources if s.ref == er.TAG)
    assert f"[{er.TAG}]" in context
    assert "Median 1.00" in fact.evidence and "Chance 0.05" in fact.evidence
    assert "nearest 15 cells on the plot (5% of the cells)" in fact.evidence
    assert "Exact neighbours kept: median 14 of 14." in fact.evidence  # faithful: no caveat
    assert "Worst groups of 'group'" in fact.evidence
    assert fact.detail.startswith("X_umap: median placement 1.00 (share of 14 graph neighbours within "
                                  "each cell's nearest 15 cells)")
    assert "A group smaller than the window (15 cells) scores near 1" in fact.evidence


def test_exact_sentence_says_few_only_when_few():
    assert er.exact_sentence(15, 1 / 15, "UMAP") == (
        "Exact neighbours kept: median 1 of 15; UMAP keeps few exact neighbours anywhere, so read "
        "positions within an island loosely.")
    assert er.exact_sentence(15, 0.6) == "Exact neighbours kept: median 9 of 15."


def test_claims_resolve_the_tag():
    from scview.core.claims import resolve_tag

    refs = ["dataset:summary", er.TAG, "result:pca"]
    assert resolve_tag(er.TAG, refs) == [er.TAG]
    assert resolve_tag("result:reliability", refs) == [er.TAG]
    assert resolve_tag("embedding_reliability", refs) == [er.TAG]
