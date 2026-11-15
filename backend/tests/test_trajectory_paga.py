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


# --- ties in PAGA's tree ---------------------------------------------------------------------

def test_group_links_capped_are_scanpys_connectivities():
    import scanpy as sc
    import scipy.sparse as sp

    from scview.core.trajectory import group_links

    a, _ = _with_graph()
    sc.tl.paga(a, groups="stage")
    _, ratio = group_links(a.obsp["distances"], a.obs["stage"].cat.codes.to_numpy(),
                           len(a.obs["stage"].cat.categories))
    assert np.allclose(np.minimum(ratio, 1), sp.csr_matrix(a.uns["paga"]["connectivities"]).toarray(), atol=1e-9)


def _picks_adata(picks: dict[str, dict[str, float]], n: int = 100, k: int = 10, seed: int = 3) -> ad.AnnData:
    """Groups of ``n`` cells on a hand-made kNN graph (``k`` recorded as its neighbours):
    ``picks[g][h]`` neighbours each cell of g takes among h's cells (under 1: that share of
    g's cells take one)."""
    import scipy.sparse as sp

    rng = np.random.default_rng(seed)
    names = list(picks)
    start = {g: i * n for i, g in enumerate(names)}
    rows, cols = [], []
    for g in names:
        for c in range(start[g], start[g] + n):
            for h, m in picks[g].items():
                m = int(m) if m >= 1 else int(rng.random() < m)
                pool = [x for x in range(start[h], start[h] + n) if x != c]
                for x in rng.choice(pool, size=m, replace=False):
                    rows.append(c)
                    cols.append(int(x))
    total = len(names) * n
    dist = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(total, total))
    a = ad.AnnData(np.zeros((total, 2), dtype="float32"))
    a.obs_names = [f"c{i}" for i in range(total)]
    a.obs["grp"] = pd.Categorical(np.repeat(names, n))
    a.obsp["distances"] = dist
    a.obsp["connectivities"] = ((dist + dist.T) > 0).astype(np.float64)
    a.uns["neighbors"] = {"connectivities_key": "connectivities", "distances_key": "distances",
                          "params": {"n_neighbors": k, "method": "umap", "use_rep": "X_pca"}}
    return a


def _tie_adata() -> ad.AnnData:
    """Four groups of 100 cells (k = 10): A, B and C all link more than random wiring would
    (uncapped A-B 1.2, A-C 1.6, B-C 2.0: every one capped at 1), D hangs on C."""
    return _picks_adata({"A": {"A": 3, "B": 3, "C": 4}, "B": {"B": 2, "A": 3, "C": 5},
                         "C": {"C": 1, "A": 4, "B": 5}, "D": {"D": 9, "C": 1}})


def test_tied_tree_edges_are_broken_by_the_uncapped_ratio_and_reported():
    a = _tie_adata()
    record = compute_paga(a, "grp")
    g = record["groups"]
    w = _weights(record)
    assert w[("A", "B")] == w[("A", "C")] == w[("B", "C")] == 1.0  # saturated: a three-way tie
    assert sorted((g[i], g[j]) for i, j in record["tree"]) == [("A", "C"), ("B", "C"), ("C", "D")]
    assert record["tie_break"] == "uncapped ratio"
    ties = [((g[t["kept"][0]], g[t["kept"][1]]), (g[t["instead_of"][0]], g[t["instead_of"][1]]), t["close"])
            for t in record["ties"]]
    # closest first: A - C (about 1.6) beat A - B (1.2) by under 1.5 times; B - C (2.0) clearly
    assert ties == [(("A", "C"), ("A", "B"), True), (("B", "C"), ("A", "B"), False)]
    assert all(t["ratio_kept"] > t["ratio_instead_of"] for t in record["ties"])
    (warning,) = record["warnings"]
    assert warning.startswith("The tree depends on 1 close tie: each tied pair has the same connectivity")
    assert "'A - C' over 'A - B' (" in warning and "'B - C'" not in warning
    assert warning.endswith("Read those steps of the tree as uncertain.")


def test_ties_broken_by_a_clear_margin_are_kept_but_not_warned_about():
    from scview.core.trajectory import ties_warning

    clear = [{"kept": [0, 1], "instead_of": [1, 2], "connectivity": 1.0, "ratio_kept": 8.0,
              "ratio_instead_of": 1.07, "close": False}]
    assert ties_warning(["a", "b", "c"], clear) is None and ties_warning(["a"], []) is None


def test_a_tree_without_ties_is_scanpys_and_says_nothing():
    import scanpy as sc
    import scipy.sparse as sp

    a, _ = _with_graph()
    record = compute_paga(a, "stage")
    assert record["ties"] == [] and record["warnings"] == []
    sc.tl.paga(a, groups="stage")
    tree = sp.coo_matrix(a.uns["paga"]["connectivities_tree"])
    assert [list(e) for e in record["tree"]] == sorted(
        [min(i, j), max(i, j)] for i, j in zip(tree.row.tolist(), tree.col.tolist()))


def test_spanning_tree_prefers_the_capped_connectivity_then_the_ratio():
    from scview.core.trajectory import spanning_tree, tree_ties

    capped = np.array([[0, 1.0, 0.5], [1.0, 0, 1.0], [0.5, 1.0, 0]])
    ratio = np.array([[0, 3.0, 0.5], [3.0, 0, 1.5], [0.5, 1.5, 0]])
    assert spanning_tree(3, capped, ratio) == [(0, 1), (1, 2)]
    assert tree_ties([(0, 1), (1, 2)], capped, ratio) == []  # 0-2 is weaker: no tie
    ratio[0, 1] = ratio[1, 0] = 1.2  # the order of ratios never beats a lower capped value
    assert spanning_tree(3, capped, ratio) == [(0, 1), (1, 2)]
    split = np.zeros((3, 3))
    split[0, 1] = split[1, 0] = 0.4
    assert spanning_tree(3, split, split) == [(0, 1)]  # a graph in two parts: a forest


# --- DPT across lineages -----------------------------------------------------------------------

def _two_lineages() -> ad.AnnData:
    """A path P1 - P2 - P3 and a second population Q joined to P3 by a handful of edges."""
    return _picks_adata({"P1": {"P1": 8, "P2": 2}, "P2": {"P2": 6, "P1": 2, "P3": 2},
                         "P3": {"P3": 8, "P2": 2}, "Q": {"Q": 10, "P3": 0.05}}, n=120)


def test_dpt_warns_when_groups_sit_off_the_roots_lineage():
    a = _two_lineages()
    got = compute_dpt(a, "grp", "P1")
    lineage = got["root_checks"]["lineage"]
    assert lineage["lineage"] == ["P1", "P2", "P3"] and lineage["column"] == "grp"
    (q,) = lineage["off_lineage"]
    assert q["group"] == "Q" and q["n_cells"] == 120 and q["widest_share"] < 0.01
    (w,) = [x for x in got["warnings"] if "not on a path" in x]
    assert w.startswith("1 groups of grp (120 of 480 cells, 25%) are not on a path from 'P1'")
    assert f"'Q' ({q['median_pseudotime']:.2g})" in w and "'P1', 'P2', 'P3'" in w
    # Q's own root: P1 - P3 are off its lineage the same way
    other = compute_dpt(a, "grp", "Q")
    assert [x["group"] for x in other["root_checks"]["lineage"]["off_lineage"]] == ["P1", "P2", "P3"]


def test_dpt_along_one_lineage_says_nothing():
    a, _ = _path_adata()
    got = compute_dpt(a, "stage", "early")
    lineage = got["root_checks"]["lineage"]
    assert lineage["off_lineage"] == [] and set(lineage["lineage"]) == {"early", "mid", "late"}
    assert not any("not on a path" in w for w in got["warnings"])


def test_a_root_whose_tree_position_rests_on_a_close_tie_is_not_judged_inside():
    from scview.core.trajectory import tied_links

    rec = {"groups": ["MEP", "Ery", "GMP", "Mk"], "sizes": [100, 400, 400, 50],
           "tree": [[0, 1], [0, 2], [2, 3]],
           "ties": [{"kept": [2, 3], "instead_of": [0, 3], "close": True},
                    {"kept": [0, 2], "instead_of": [1, 2], "close": False}]}
    assert tied_links(rec, "MEP") == ["GMP - Mk over MEP - Mk"]  # the clear tie is not listed
    assert tied_links(rec, "Ery") == [] and tied_links(rec, "nope") == []
    a = _tie_adata()
    compute_paga(a, "grp")  # A - C kept over A - B by under 1.5x: B's and C's place is open
    got = compute_dpt(a, "grp", "B")
    assert got["root_checks"]["paga"]["close_ties"] == ["A - C over A - B"]
    assert not any("sits inside the PAGA tree" in w for w in got["warnings"])
