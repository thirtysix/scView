"""Layout stability (core/layout_stability.py, api/v1/layouts.py): which group
relations in a 2D layout hold across other seeds and neighbour counts.

Expected values are known in advance (separated blobs never touch, overlapping
blobs always do, a planted bridge touches only in the layouts that draw it) or
computed independently here: plot neighbours by all pairwise distances and a
group's main body by breadth-first search over its own cells.
"""

from __future__ import annotations

import asyncio
import json
import threading
from collections import deque
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
import scipy.sparse as sp

from scview.core import layout_stability as ls
from scview.core import provenance, replay_script
from scview.core.anndata_adaptor import AnnDataAdaptor
from scview.core.assistant import build_grounding_context

API = "/api/v1"
N_PER = 120
# g0, g1, g2 far apart; g3 and g4 overlap (one island).
CENTRES = np.array([[0, 0], [40, 0], [80, 0], [0, 40], [2.5, 40]], float)
BRIDGE = np.arange(2 * N_PER - 40, 2 * N_PER)  # the last 40 cells of g1


def _data(sparse: bool = False, seed: int = 0) -> ad.AnnData:
    """Five groups; X_pca is the 2D positions padded with zero columns, so the
    graph's neighbours are the positions' neighbours. The graph is scanpy's."""
    rng = np.random.default_rng(seed)
    pos = np.concatenate([rng.normal(size=(N_PER, 2)) + c for c in CENTRES])
    n = pos.shape[0]
    pca = np.zeros((n, 10), dtype=np.float32)
    pca[:, :2] = pos
    X = rng.poisson(1.0, size=(n, 20)).astype(np.float32)
    a = ad.AnnData(X=sp.csr_matrix(X) if sparse else X)
    a.obs_names = [f"c{i}" for i in range(n)]
    a.var_names = [f"G{i}" for i in range(20)]
    a.obs["group"] = pd.Categorical(np.repeat([f"g{g}" for g in range(5)], N_PER))
    a.obs["score"] = rng.normal(size=n)
    a.obsm["X_pca"] = pca
    sc.pp.neighbors(a, n_neighbors=15, use_rep="X_pca")
    a.uns["scview_active_clustering"] = "group"
    return a


def _bridge_layout(pos: np.ndarray, seed: int, k: int) -> np.ndarray:
    """The positions, rotated by the seed (the same neighbours); with 30
    neighbours the bridge cells of g1 run as a line into g2's island."""
    Y = pos[:, :2].astype(np.float64).copy()
    if k == 30:
        t = np.linspace(41.5, 79.0, BRIDGE.size)
        Y[BRIDGE] = np.column_stack([t, np.random.default_rng(7).normal(scale=0.3, size=BRIDGE.size)])
    theta = 0.4 * seed
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return (Y @ rot.T).astype(np.float32)


def fake_layout(holder, seed: int, rec) -> np.ndarray:
    k = holder.uns["neighbors"]["params"]["n_neighbors"]
    return _bridge_layout(holder.obsm["X_rep"], seed, k)


def _pair(summary: dict, a: str, b: str) -> dict:
    return next(r for r in summary["relations"] if {r["a"], r["b"]} == {a, b})


# --- independent implementations ----------------------------------------------------------


def _knn_brute(Y: np.ndarray, k: int) -> np.ndarray:
    d = ((Y[:, None, :].astype(np.float64) - Y[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d, np.inf)
    return np.argsort(d, axis=1, kind="stable")[:, :k]


def _contacts_brute(Y: np.ndarray, codes: np.ndarray, k: int, share: float) -> np.ndarray:
    nn = _knn_brute(Y, k)
    n, G = len(codes), int(codes.max()) + 1
    sizes = np.bincount(codes, minlength=G)
    # Main body: breadth-first search over each group's own kNN edges (both directions).
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in nn[i]:
            if codes[j] == codes[i]:
                adj[i].add(int(j))
                adj[int(j)].add(i)
    comp = -np.ones(n, dtype=int)
    for s in range(n):
        if comp[s] >= 0:
            continue
        comp[s] = s
        q = deque([s])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if comp[v] < 0:
                    comp[v] = s
                    q.append(v)
    csize = np.bincount(comp, minlength=n)
    body = csize[comp] >= np.maximum(share * sizes[codes], 1)
    c = np.zeros((G, G), dtype=int)
    for i in range(n):
        if not body[i]:
            continue
        seen = {int(codes[j]) for j in nn[i] if body[j] and codes[j] != codes[i]}
        for g in seen:
            c[codes[i], g] += 1
    return c


# --- the touching rule --------------------------------------------------------------------


def test_contacts_match_a_brute_force_count_with_strays_and_clumps():
    rng = np.random.default_rng(3)
    Y = np.concatenate([rng.normal(size=(150, 2)), rng.normal(size=(150, 2)) + [2.2, 0],
                        rng.normal(size=(80, 2)) + [20, 0]])
    codes = np.repeat([0, 1, 2], [150, 150, 80])
    codes[[5, 9]] = 2  # strays of g2 inside g0's island
    codes[300:304] = 1  # a 4-cell clump of g1 inside g2's island (under 5% of g1)
    got = ls.contact_matrix(ls.plot_neighbours(Y.astype(np.float32)), codes, 3)
    np.testing.assert_array_equal(got, _contacts_brute(Y, codes, ls.PLOT_K, ls.BODY_SHARE))
    # The strays and the clump are not main body: g0~g2 and g1~g2 have no contact.
    assert got[0, 2] == got[2, 0] == got[1, 2] == got[2, 1] == 0
    assert got[0, 1] > 0 and got[1, 0] > 0


def test_a_stray_cell_inside_another_island_is_not_a_contact():
    rng = np.random.default_rng(1)
    Y = np.concatenate([rng.normal(size=(200, 2)), rng.normal(size=(200, 2)) + [30, 0]])
    codes = np.repeat([0, 1], 200)
    Y[:3] = rng.normal(size=(3, 2)) + [30, 0]  # three g0 cells drawn inside g1
    nn = ls.plot_neighbours(Y.astype(np.float32))
    naive = (codes[nn] != codes[:, None]).any(axis=1).sum()
    assert naive > 3  # counting every cell would make them touch
    c = ls.contact_matrix(nn, codes, 2)
    assert c.sum() == 0
    assert not ls.touch_matrix(c, np.array([200, 200]))[0, 1]


def test_touch_needs_ten_contact_cells_or_a_small_groups_worth():
    sizes = np.array([100, 100, 6])
    c = np.array([[0, 6, 0], [3, 0, 3], [0, 2, 0]])
    t = ls.touch_matrix(c, sizes)
    assert not t[0, 1]  # 9 < 10
    assert not t[1, 2]  # 5 < 6, the small group's size
    c[2, 1] = 3
    t = ls.touch_matrix(c, sizes)
    assert t[1, 2] and t[2, 1]  # 6 >= min(10, 6)
    c[1, 0] = 4
    assert ls.touch_matrix(c, sizes)[0, 1]  # 10
    assert not ls.touch_matrix(c, sizes).diagonal().any()


# --- the run with planted layouts ---------------------------------------------------------


def _planted_run(a: ad.AnnData, **plan_kw) -> dict:
    a.obsm["X_umap"] = _bridge_layout(a.obsm["X_pca"], 0, 30)  # the user's plot draws the bridge
    plan = ls.make_plan(a, embedding="X_umap", group_by="group", **plan_kw)
    return ls.run_all(a, plan, layout_fn=fake_layout)


@pytest.mark.parametrize("sparse", [False, True])
def test_relations_every_never_and_a_planted_bridge_in_some(sparse):
    a = _data(sparse=sparse)
    s = _planted_run(a)
    assert s["n_neighbors"] == [10, 15, 30] and s["seeds"] == [0, 1, 2] and len(s["done"]) == 9
    assert [(d["seed"], d["n_neighbors"]) for d in s["done"]][:4] == [(0, 10), (1, 10), (2, 10), (0, 15)]
    touching = _pair(s, "g3", "g4")
    assert touching["status"] == "every" and touching["n_touch"] == 9 and touching["current"] is True
    for x, y in [("g0", "g1"), ("g0", "g2"), ("g0", "g3"), ("g1", "g4"), ("g2", "g3")]:
        r = _pair(s, x, y)
        assert r["status"] == "never" and r["n_touch"] == 0 and r["current"] is False
    bridge = _pair(s, "g1", "g2")
    assert bridge["status"] == "some" and bridge["n_touch"] == 3
    assert bridge["layouts"] == [False] * 6 + [True] * 3  # the 30-neighbour row
    assert bridge["current"] is True and bridge["flag"] == "rare_in_plot"
    assert s["relations"][0]["flag"] == "rare_in_plot"  # rare in the plot first
    assert s["counts"] == {"every": 1, "some": 1, "never": 8, "rare_in_plot": 1, "hidden_in_plot": 0}
    # The plot is the seed-0, 30-neighbour layout.
    assert [d["index"] for d in s["done"] if d["same_as_current"]] == [6]


def test_placement_range_follows_the_bridge():
    s = _planted_run(_data())
    groups = {g["group"]: g for g in s["groups"]}
    # A rotated copy of the positions keeps every graph neighbour (hand-known: 1.0).
    for g in ("g0", "g2", "g3", "g4"):
        assert groups[g]["min"] == groups[g]["max"] == 1.0
    g1 = groups["g1"]
    assert g1["placement"][:6] == [1.0] * 6 and max(g1["placement"][6:]) < 1.0
    assert g1["range"] > 0 and s["groups"][0]["group"] == "g1"  # widest range first
    assert g1["current"] == g1["placement"][6]


def test_the_plot_missing_a_common_relation_is_flagged_hidden():
    a = _data()
    a.obsm["X_umap"] = _bridge_layout(a.obsm["X_pca"], 0, 10)  # no bridge in the plot
    plan = ls.make_plan(a, embedding="X_umap", group_by="group", ks=[10, 30, 50])
    s = ls.run_all(a, plan, layout_fn=lambda h, seed, rec: _bridge_layout(
        h.obsm["X_rep"], seed, 30 if h.uns["neighbors"]["params"]["n_neighbors"] >= 30 else 10))
    bridge = _pair(s, "g1", "g2")
    assert bridge["n_touch"] == 6 and bridge["current"] is False and bridge["flag"] == "hidden_in_plot"


def test_another_grouping_needs_no_new_layouts():
    a = _data()
    _planted_run(a)
    run = ls.latest(a)
    a.obs["half"] = pd.Categorical(np.where(np.arange(a.n_obs) < 300, "left", "right"))
    s = ls.summarise(a, run, group_by="half")
    assert s["group_by"] == "half" and s["n_pairs"] == 1
    with pytest.raises(ls.LayoutError, match="numbers"):
        ls.summarise(a, run, group_by="score")


def test_without_a_2d_plot_the_layouts_are_compared_with_each_other():
    a = _data()
    plan = ls.make_plan(a, group_by="group")
    assert plan.current is None and any("no 2D UMAP" in n for n in plan.notes)
    s = ls.run_all(a, plan, layout_fn=fake_layout)
    assert s["current"] is None and all(r["current"] is None and r["flag"] is None for r in s["relations"])
    a.obsm["X_umap_3d"] = np.zeros((a.n_obs, 3), dtype=np.float32)
    with pytest.raises(KeyError):
        ls.make_plan(a, embedding="X_tsne")
    p3 = ls.make_plan(a, embedding="X_umap_3d")
    assert p3.current is None and any("not a 2D plot" in n for n in p3.notes)


def test_plan_reads_the_recorded_settings_and_validates():
    a = _data()
    rec = ls.recorded(a)
    assert (rec.use_rep, rec.n_neighbors, rec.min_dist) == ("X_pca", 15, 0.5)
    assert any("min_dist is not recorded" in n for n in rec.notes)
    provenance.record_step(a, step="embeddings", tool="scanpy.tl.umap", params={"umap_min_dist": 0.3})
    assert ls.recorded(a).min_dist == 0.3 and not ls.recorded(a).notes
    assert ls.default_neighbors(15, 1000) == [10, 15, 30]
    assert ls.default_neighbors(20, 1000) == [10, 20, 30]
    assert ls.default_neighbors(10, 1000) == [10, 15, 30]
    assert ls.default_neighbors(15, 25) == [5, 10, 15]
    with pytest.raises(ls.LayoutError, match="at most 12"):
        ls.make_plan(a, seeds=[0, 1, 2, 3], ks=[5, 10, 15, 20])
    with pytest.raises(ls.LayoutError, match="between"):
        ls.make_plan(a, ks=[1])
    del a.obsm["X_pca"]
    with pytest.raises(ls.LayoutError, match="PCA"):
        ls.recorded(a)


def test_a_sample_is_seeded_and_the_same_every_time():
    r1, s1 = ls.sample_rows(50_000, 20_000)
    r2, _ = ls.sample_rows(50_000, 20_000)
    assert s1 and r1.size == 20_000 and np.array_equal(r1, r2) and np.all(np.diff(r1) > 0)
    assert not ls.sample_rows(500, 20_000)[1]
    a = _data()
    p1 = ls.make_plan(a, max_cells=300)
    p2 = ls.make_plan(a, max_cells=300)
    assert p1.sampled and p1.rows.size == 300 and np.array_equal(p1.rows, p2.rows)
    assert any("seeded sample of 300" in n for n in p1.notes)
    s = ls.run_all(a, p1, layout_fn=lambda h, seed, rec: h.obsm["X_rep"][:, :2].astype(np.float32))
    assert s["n_laid_out"] == 300 and s["sampled"] is True
    assert sum(g["n_cells"] for g in s["groups"]) == 300


def test_a_stopped_run_keeps_what_it_laid_out():
    a = _data()
    plan = ls.make_plan(a, group_by="group")
    gen = ls.iter_run(a, plan, stop=lambda: len(ls.latest(a).layouts) >= 2 if ls.latest(a) else False,
                      layout_fn=fake_layout)
    events = []
    with pytest.raises(ls.Cancelled):
        for ev, _ in gen:
            events.append(ev)
    assert events == ["start", "layout", "layout"]
    run = ls.latest(a)
    assert run.stopped and not run.complete and len(run.layouts) == 2
    s = ls.summarise(a, run)
    assert s["n_done"] == 2 and any("stopped after 2 of 9" in n for n in s["notes"])
    assert _pair(s, "g3", "g4")["n_layouts"] == 2


def test_a_new_representation_drops_the_held_run():
    a = _data()
    _planted_run(a)
    assert ls.latest(a) is not None
    a.obsm["X_pca"] = a.obsm["X_pca"].copy()
    assert ls.latest(a) is None


# --- thumbnails, keep, replay --------------------------------------------------------------


def test_thumbnails_show_the_same_cells_in_every_layout():
    a = _data()
    _planted_run(a)
    run = ls.latest(a)
    t0 = ls.thumbnail(a, run, 0, max_points=200)
    t6 = ls.thumbnail(a, run, "6", max_points=200)
    cur = ls.thumbnail(a, run, "current", max_points=200)
    assert t0["cells"] == t6["cells"] == cur["cells"] and t0["n_shown"] == 200
    assert cur["index"] == "current" and cur["label"] == "your plot"
    exp = _bridge_layout(a.obsm["X_pca"], 0, 30)[t6["cells"]]
    np.testing.assert_allclose(t6["x"], exp[:, 0], atol=1e-3)
    np.testing.assert_allclose(t6["y"], exp[:, 1], atol=1e-3)
    assert cur["x"] == t6["x"]
    with pytest.raises(KeyError):
        ls.thumbnail(a, run, 9)


def test_keep_saves_the_layout_once_and_records_it():
    a = _data()
    _planted_run(a)
    run = ls.latest(a)
    out = ls.keep(a, run, 1)
    assert out == {"key": "X_umap_s1_k10", "created": True, "n_cells": 600, "recomputed": False, "note": None}
    np.testing.assert_array_equal(a.obsm["X_umap_s1_k10"], _bridge_layout(a.obsm["X_pca"], 1, 10))
    step = provenance.read_provenance(a)["history"][-1]
    assert step["step"] == "keep_layout"
    assert step["params"]["seed"] == 1 and step["params"]["n_neighbors"] == 10
    assert step["params"]["use_rep"] == "X_pca" and step["params"]["min_dist"] == 0.5
    assert ls.keep(a, run, 1)["created"] is False
    assert replay_script.describe(a)[-1]["status"] == "exact"


@pytest.fixture(scope="module")
def umap_run():
    """Real UMAPs (umap-learn via scanpy) of the five groups: 9 layouts."""
    a = _data()
    # The pipeline's own layout (core/pipeline.py _run_embeddings): seed 0, the stored graph.
    sc.tl.umap(a, min_dist=0.5)
    a.uns.pop("umap", None)
    plan = ls.make_plan(a, embedding="X_umap", group_by="group")
    return a, ls.run_all(a, plan)


def test_real_umap_layouts_keep_the_touching_pair_and_the_separated_ones(umap_run):
    a, s = umap_run
    assert _pair(s, "g3", "g4")["status"] == "every"
    for x, y in [("g0", "g1"), ("g0", "g2"), ("g1", "g2"), ("g0", "g3"), ("g2", "g4")]:
        assert _pair(s, x, y)["status"] == "never"
    # Seed 0 with the dataset's own 15 neighbours, graph rebuilt: the pipeline's layout again.
    assert [d["index"] for d in s["done"] if d["same_as_current"]] == [3]


def test_a_kept_real_layout_is_replayed_exactly(umap_run):
    a, _ = umap_run
    fresh = a.copy()
    run = ls.latest(a)
    out = ls.keep(a, run, 1)  # seed 1, 10 neighbours
    step = provenance.read_provenance(a)["history"][-1]
    code = "\n".join(replay_script._keep_layout(step["params"], replay_script._Ctx(adata=fresh)).code)
    exec(code, {"sc": sc, "adata": fresh})  # noqa: S102 — the script's own lines
    np.testing.assert_allclose(fresh.obsm[out["key"]], a.obsm[out["key"]], atol=1e-5)
    assert "layout_check" not in fresh.uns and "layout_check_distances" not in fresh.obsp


def test_keep_of_a_sample_lays_out_all_cells_again(umap_run):
    a, _ = umap_run
    b = a.copy()
    plan = ls.make_plan(b, group_by="group", seeds=[2], ks=[10], max_cells=300)
    ls.run_all(b, plan)
    out = ls.keep(b, ls.latest(b), 0)
    assert out["recomputed"] is True and "300-cell sample" in out["note"]
    assert b.obsm["X_umap_s2_k10"].shape == (600, 2)


# --- co-pilot -----------------------------------------------------------------------------


def test_the_copilot_fact_names_the_relation_rare_across_layouts(tmp_path):
    a = _data()
    a.obsm["X_umap"] = _bridge_layout(a.obsm["X_pca"], 0, 30)
    p = tmp_path / "d.h5ad"
    a.write_h5ad(p)
    adaptor = AnnDataAdaptor(str(p))
    _, sources = build_grounding_context(adaptor)
    assert ls.TAG not in {s.ref for s in sources}
    plan = ls.make_plan(adaptor.adata, embedding="X_umap", group_by="group")
    ls.run_all(adaptor.adata, plan, layout_fn=fake_layout)
    context, sources = build_grounding_context(adaptor)
    fact = next(s for s in sources if s.ref == ls.TAG)
    assert f"[{ls.TAG}]" in context
    assert "g1 and g2 (3 of 9 layouts)" in fact.evidence
    assert "1 touch in every layout, 1 in some, 8 in none" in fact.evidence


# --- endpoints ----------------------------------------------------------------------------


def _register(dm, a: ad.AnnData, did: str = "lay") -> str:
    d = Path(dm.data_dir) / "uploads" / did
    d.mkdir(parents=True)
    (d / "metadata.json").write_text(json.dumps({"id": did, "name": did, "status": "ready"}))
    a.write_h5ad(d / "data.h5ad")
    return did


def _events(text: str) -> list[tuple[str, dict]]:
    out = []
    for block in text.strip().split("\n\n"):
        if block.startswith(":"):
            continue
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        out.append((lines["event"], json.loads(lines["data"])))
    return out


@pytest.fixture()
def planted(monkeypatch):
    monkeypatch.setattr(ls, "umap_layout", fake_layout)


async def test_endpoints_run_show_and_keep(client, dataset_manager, planted):
    a = _data()
    a.obsm["X_umap"] = _bridge_layout(a.obsm["X_pca"], 0, 30)
    did = _register(dataset_manager, a)
    info = (await client.get(f"{API}/datasets/{did}/layouts")).json()
    assert info["available"] and info["result"] is None
    assert info["plan"]["n_neighbors"] == [10, 15, 30] and info["plan"]["current"] == "X_umap"
    assert info["plan"]["group_by"] == "group" and info["estimate_seconds"] > 0
    r = await client.get(f"{API}/datasets/{did}/layouts/0")
    assert r.status_code == 404 and "run the check" in r.json()["detail"]

    r = await client.post(f"{API}/datasets/{did}/layouts/run-stream", json={"embedding": "X_umap"})
    ev = _events(r.text)
    assert [e for e, _ in ev] == ["start"] + ["layout"] * 9 + ["summary", "complete"]
    assert ev[1][1]["key"] == "X_umap_s0_k10" and ev[0][1]["n_laid_out"] == 600
    assert _pair(ev[-2][1], "g1", "g2")["flag"] == "rare_in_plot"

    t = (await client.get(f"{API}/datasets/{did}/layouts/6", params={"max_points": 100})).json()
    assert t["n_shown"] == 100 and len(t["x"]) == len(t["cells"]) == 100
    cur = (await client.get(f"{API}/datasets/{did}/layouts/current", params={"max_points": 100})).json()
    assert cur["x"] == t["x"]
    assert (await client.get(f"{API}/datasets/{did}/layouts/12")).status_code == 404
    assert (await client.get(f"{API}/datasets/{did}/layouts/abc")).status_code == 422

    info = (await client.get(f"{API}/datasets/{did}/layouts", params={"group_by": "group"})).json()
    assert info["result"]["complete"] and info["result"]["counts"]["rare_in_plot"] == 1

    r = await client.post(f"{API}/datasets/{did}/layouts/2/keep")
    assert r.status_code == 200, r.text
    assert r.json()["key"] == "X_umap_s2_k10" and r.json()["created"] is True
    derived = ad.read_h5ad(Path(dataset_manager.data_dir) / "derived" / did / "data.h5ad")
    np.testing.assert_allclose(derived.obsm["X_umap_s2_k10"], _bridge_layout(a.obsm["X_pca"], 2, 10), atol=1e-6)
    meta = (await client.get(f"{API}/datasets/{did}")).json()
    assert "X_umap_s2_k10" in meta["available_embeddings"]
    assert (await client.post(f"{API}/datasets/{did}/layouts/2/keep")).json()["created"] is False
    # Saved in place: the held run survives, and the next keep writes only obsm + uns.
    r = await client.post(f"{API}/datasets/{did}/layouts/5/keep")
    assert r.json()["created"] is True
    derived = ad.read_h5ad(Path(dataset_manager.data_dir) / "derived" / did / "data.h5ad")
    assert {"X_umap_s2_k10", "X_umap_s2_k15", "X_umap", "X_pca"} <= set(derived.obsm.keys())
    assert [h["step"] for h in provenance.read_provenance(derived)["history"]][-2:] == ["keep_layout"] * 2
    assert (await client.post(f"{API}/datasets/{did}/layouts/40/keep")).status_code == 404


async def test_endpoints_explain_what_cannot_run(client, dataset_manager):
    a = _data()
    del a.obsm["X_pca"]
    did = _register(dataset_manager, a, "nopca")
    info = (await client.get(f"{API}/datasets/{did}/layouts")).json()
    assert info["available"] is False and "PCA" in info["reason"]
    r = await client.post(f"{API}/datasets/{did}/layouts/run-stream", json={})
    assert r.status_code == 400
    did2 = _register(dataset_manager, _data(), "bad")
    r = await client.post(f"{API}/datasets/{did2}/layouts/run-stream", json={"group_by": "nope"})
    assert r.status_code == 404
    r = await client.post(f"{API}/datasets/{did2}/layouts/run-stream", json={"n_neighbors": [700]})
    assert r.status_code == 400
    assert (await client.post(f"{API}/datasets/{did2}/layouts/0/keep")).status_code == 404


async def test_a_second_run_on_the_same_dataset_is_refused(client, dataset_manager, planted):
    from scview.api.v1 import layouts

    did = _register(dataset_manager, _data(), "busy")
    layouts._running.add(did)
    try:
        r = await client.post(f"{API}/datasets/{did}/layouts/run-stream", json={})
        assert r.status_code == 409
    finally:
        layouts._running.discard(did)


async def test_a_client_going_away_stops_the_run_and_releases_it(tmp_path, monkeypatch):
    from scview.api.v1 import layouts
    from scview.api.v1.layouts import LayoutRunRequest, run_layouts
    from scview.core.dataset_manager import DatasetManager

    dm = DatasetManager(str(tmp_path))
    did = _register(dm, _data(), "away")
    entered, release = threading.Event(), threading.Event()
    calls: list[int] = []

    def held_layout(holder, seed, rec):
        calls.append(seed)
        entered.set()
        release.wait(10)
        return fake_layout(holder, seed, rec)

    monkeypatch.setattr(ls, "umap_layout", held_layout)
    resp = await run_layouts(did, LayoutRunRequest(), dm)
    chunks: list[str] = []

    async def consume():
        async for chunk in resp.body_iterator:
            chunks.append(chunk)

    task = asyncio.create_task(consume())
    assert await asyncio.to_thread(entered.wait, 10)
    task.cancel()  # the client goes away mid-layout
    with pytest.raises(asyncio.CancelledError):
        await task
    assert did in layouts._running  # the thread still works on the layout
    release.set()
    for _ in range(500):
        if did not in layouts._running:
            break
        await asyncio.sleep(0.01)
    assert did not in layouts._running
    assert calls == [0]  # stopped before the second layout
    adata = (await dm.get_or_load_dataset(did)).adata
    run = ls.latest(adata)
    assert run is not None and len(run.layouts) == 1 and not run.complete


async def test_the_run_leaves_the_loop_free(client, dataset_manager, monkeypatch):
    from tests.test_event_loop import _worst_stall

    did = _register(dataset_manager, _data(), "loop")

    def slow_layout(holder, seed, rec):
        threading.Event().wait(0.3)
        return fake_layout(holder, seed, rec)

    monkeypatch.setattr(ls, "umap_layout", slow_layout)
    resp, stall = await _worst_stall(client.post(
        f"{API}/datasets/{did}/layouts/run-stream", json={"seeds": [0], "n_neighbors": [10, 15]}))
    assert "event: complete" in resp.text
    assert stall < 0.3
