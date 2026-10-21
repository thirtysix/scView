"""Clustering tree across resolutions (core/cluster_tree.py, clustree-style)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from scview.core import cluster_tree as ct
from scview.core import resolution_sweep as rs


def _level(r: float, labels: list[str]) -> tuple[float, np.ndarray, list[str]]:
    cat = pd.Categorical(labels)
    return r, np.asarray(cat.codes), [str(c) for c in cat.categories]


def _sc3_reference(clusterings: np.ndarray, res: int, cluster) -> float:
    """clustree's calc_sc3_stability_cluster, line for line."""
    is_c1 = clusterings[:, res] == cluster
    s = 0.0
    for res2 in range(clusterings.shape[1]):
        if res2 == res:
            continue
        clusters = np.unique(clusterings[is_c1, res2])
        for cluster2 in clusters:
            is_c2 = clusterings[:, res2] == cluster2
            s += (is_c1 & is_c2).sum() / (is_c2.sum() * len(clusters) ** 2)
    return s / clusterings.shape[1]


def test_edges_carry_counts_and_the_childs_share_and_drop_small_ones():
    # 0.1: A (60) B (40); 0.5: A splits into a1 (30) a2 (30); x takes 35 of B and 5 of A.
    low = ["A"] * 60 + ["B"] * 40
    high = ["a1"] * 30 + ["a2"] * 25 + ["x"] * 5 + ["x"] * 35 + ["b"] * 5
    tree = ct.build([_level(0.1, low), _level(0.5, high)])
    edges = {(e["from"], e["to"]): e for e in tree["edges"]}
    assert edges[("A", "a1")]["count"] == 30 and edges[("A", "a1")]["in_prop"] == 1.0
    assert edges[("B", "x")]["in_prop"] == 0.875
    assert ("A", "x") in edges and edges[("A", "x")]["in_prop"] == 0.125  # 5/40, above 0.1
    assert all(e["from_resolution"] == 0.1 and e["to_resolution"] == 0.5 for e in tree["edges"])
    nodes = {(n["resolution"], n["cluster"]): n for n in tree["nodes"]}
    assert nodes[(0.1, "A")]["size"] == 60 and nodes[(0.5, "b")]["size"] == 5
    assert nodes[(0.5, "x")]["n_parents"] == 2 and not nodes[(0.5, "x")]["mixed"]  # 12.5% < 20%


def test_many_clusters_do_not_overflow_small_codes():
    # Categorical codes are int8 below 128 groups; 30 x 30 clusters overflowed them.
    low = [str(i // 10) for i in range(300)]
    high = [str(i // 10) for i in range(300)]
    tree = ct.build([_level(0.5, low), _level(1.0, high)])
    assert len(tree["edges"]) == 30 and all(e["in_prop"] == 1.0 for e in tree["edges"])
    # clustree divides by every level, the cluster's own too: two identical levels give 1/2.
    assert all(n["sc3_stability"] == 0.5 for n in tree["nodes"])


def test_a_child_drawing_on_two_parents_is_mixed():
    low = ["A"] * 50 + ["B"] * 50
    high = ["p"] * 40 + ["q"] * 10 + ["q"] * 10 + ["r"] * 40  # q: half from A, half from B
    tree = ct.build([_level(0.2, low), _level(0.4, high)])
    mixed = {n["cluster"] for n in tree["nodes"] if n["mixed"]}
    assert mixed == {"q"} and tree["n_mixed"] == 1


def test_sc3_stability_matches_clustree():
    rng = np.random.default_rng(0)
    base = rng.integers(0, 3, 300)
    cols = [base, base * 2 + rng.integers(0, 2, 300), np.where(rng.random(300) < 0.2, rng.integers(0, 7, 300),
                                                                 base * 2)]
    levels = [_level(r, [str(v) for v in c]) for r, c in zip((0.2, 0.6, 1.0), cols)]
    tree = ct.build(levels)
    clusterings = np.column_stack(cols)
    for n in tree["nodes"]:
        k = (0.2, 0.6, 1.0).index(n["resolution"])
        assert abs(n["sc3_stability"] - _sc3_reference(clusterings, k, int(n["cluster"]))) < 1e-4


def test_a_cluster_that_never_changes_has_the_highest_stability():
    same = ["A"] * 50 + ["B"] * 50
    split = ["A"] * 50 + ["B1"] * 25 + ["B2"] * 25
    tree = ct.build([_level(0.2, same), _level(0.4, split), _level(0.6, split)])
    s = {(n["resolution"], n["cluster"]): n["sc3_stability"] for n in tree["nodes"]}
    assert s[(0.2, "A")] > s[(0.2, "B")]
    assert s[(0.4, "A")] == s[(0.6, "A")]


def test_the_sweep_ends_with_the_tree_of_its_own_labels():
    from tests.test_resolution_sweep import _adata

    a = _adata()
    out = rs.sweep(a, [0.2, 0.6, 1.5], "leiden")
    tree = out["tree"]
    for row in out["results"]:
        sizes = sorted((n["size"] for n in tree["nodes"] if n["resolution"] == row["resolution"]), reverse=True)
        assert sizes == row["sizes"]
    # Every child at 0.6 and 1.5 has at least its biggest parent drawn.
    children = {(n["resolution"], n["cluster"]) for n in tree["nodes"] if n["resolution"] > 0.2}
    assert children == {(e["to_resolution"], e["to"]) for e in tree["edges"]}


def test_missing_labels_are_one_node():
    from tests.test_resolution_sweep import _adata

    a = _adata()
    a.obs["scview_leiden_r0.5"] = pd.Categorical(["x"] * 200 + [None] * 200)
    tree = rs.sweep(a, [0.5], "leiden")["tree"]
    assert {n["cluster"]: n["size"] for n in tree["nodes"]} == {"x": 200, "(no label)": 200}
