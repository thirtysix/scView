"""Minimal marker panel (core/marker_panel.py, POST/GET /markers/panel)."""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
import scipy.sparse as sp

from scview.core import marker_panel as mp
from scview.core.anndata_adaptor import AnnDataAdaptor
from scview.core.assistant import build_grounding_context
from scview.core.uns_keys import markers_key
from tests.large_fixture import counts_adata, register, small_twin

API = "/api/v1"
GROUPS = ["a", "b", "c", "d"]


def _data(n_per: int = 120, seed: int = 0, small: int = 0) -> ad.AnnData:
    """Four groups, each with 5 genes up (g*5 .. g*5+4); "c" and "d" share their
    first gene, so one gene alone cannot split them. ``small``: a fifth group of that
    many cells. Log-normalised X, the column's Wilcoxon markers stored."""
    rng = np.random.default_rng(seed)
    labels = np.repeat(GROUPS, n_per)
    if small:
        labels = np.concatenate([labels, ["tiny"] * small])
    n, genes = labels.size, 60
    rate = np.full((n, genes), 0.5)
    for g, name in enumerate(GROUPS):
        rate[labels == name, g * 5:(g + 1) * 5] = 6.0
    rate[labels == "d", 10] = 6.0  # c's first marker is up in d too
    counts = rng.poisson(rate).astype(np.float32)
    a = ad.AnnData(X=sp.csr_matrix(counts),
                   obs=pd.DataFrame({"group": pd.Categorical(labels)},
                                    index=[f"c{i}" for i in range(n)]),
                   var=pd.DataFrame(index=[f"G{i}" for i in range(genes)]))
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    a.X = sp.csr_matrix(a.X, dtype=np.float32)
    sc.tl.rank_genes_groups(a, "group", method="wilcoxon", pts=True, key_added=markers_key("group"))
    return a


class _Mem:
    is_large = False

    def __init__(self, adata):
        self.adata = adata


# --- plan ---------------------------------------------------------------------------------


def test_plan_takes_candidates_from_the_one_marker_choice():
    from scview.core.marker_selection import group_markers

    a = _data()
    plan = mp.make_plan(_Mem(a), "group", size=6, per_group=3)
    chosen = group_markers(a, "group", n=3)
    assert set(plan.candidates) == {g for genes in chosen.values() for g in genes}
    # baseline order: each group's best in turn
    assert plan.baseline[:4] == [chosen[g][0] for g in GROUPS]
    assert plan.groups == GROUPS and plan.y.size == plan.cells.size == 480
    assert np.all(np.diff(plan.cells) > 0)


def test_plan_required_excluded_restricted_and_missing():
    a = _data()
    plan = mp.make_plan(_Mem(a), "group", size=5, required=["G20", "NOPE", "G0"],
                        excluded=["G1", "G5"], restrict=[f"G{i}" for i in range(0, 15)])
    assert plan.required == ["G20", "G0"] and plan.missing == ["NOPE"]
    assert plan.candidates[:2] == ["G20", "G0"]
    assert "G1" not in plan.candidates and "G5" not in plan.candidates
    assert all(int(g[1:]) < 15 for g in plan.baseline)
    assert any("NOPE" in n for n in plan.notes)
    assert "d" in plan.to_dict()["groups"] and plan.restricted == 15


def test_plan_samples_with_a_fixed_seed_and_drops_tiny_groups(monkeypatch):
    monkeypatch.setattr(mp, "CELLS_PER_GROUP", 50)
    a = _data(small=4)
    p1 = mp.make_plan(_Mem(a), "group")
    p2 = mp.make_plan(_Mem(a), "group")
    assert np.array_equal(p1.cells, p2.cells)
    assert np.bincount(p1.y).tolist() == [50] * 4
    assert p1.dropped_groups == ["tiny"] and "tiny" not in p1.groups


def test_plan_refusals():
    a = _data()
    with pytest.raises(KeyError):
        mp.make_plan(_Mem(a), "nope")
    a.obs["one"] = "x"
    with pytest.raises(mp.PanelError, match="two groups"):
        mp.make_plan(_Mem(a), "one")
    a.obs["other"] = a.obs["group"].astype(str)
    with pytest.raises(mp.PanelError, match="No marker genes"):
        mp.make_plan(_Mem(a), "other")
    with pytest.raises(mp.PanelError, match="No candidate"):
        mp.make_plan(_Mem(a), "group", restrict=["G59"], excluded=["G59"])


# --- selection ----------------------------------------------------------------------------


def test_score_test_ranks_the_gene_that_splits_the_remaining_groups():
    rng = np.random.default_rng(1)
    y = np.repeat([0, 1], 200)
    Zc = rng.normal(size=(400, 3))
    Zc[:, 2] += 2.0 * (y == 1)  # informative
    P = np.full((400, 2), 0.5)
    assert np.argmax(mp.score_test(mp.standardise(Zc), y, P, 2)) == 2


def test_run_finds_a_small_panel_better_than_the_top_markers():
    a = _data()
    plan = mp.make_plan(_Mem(a), "group", size=4, per_group=5)
    res = mp.run(_Mem(a), plan)
    assert len(res["panel"]) == len(res["path"]) == 4 == len(res["baseline"])
    accs = [p["balanced_accuracy"] for p in res["path"]]
    assert accs[-1] > 0.95
    # one gene per group is enough; the shared c/d gene alone is not
    assert res["balanced_accuracy"] >= res["baseline"][-1]["balanced_accuracy"] - 1e-9
    conf = np.asarray(res["confusion"]["counts"])
    assert conf.sum() == 480 and conf.shape == (4, 4)
    assert np.isclose(np.mean(np.diag(conf) / conf.sum(axis=1)), res["balanced_accuracy"], atol=1e-4)
    for g in res["genes"]:
        idx = int(g["gene"][1:])
        if idx < 20 and idx != 10:
            assert GROUPS[idx // 5] in g["helps"]
        assert set(g["coef"]) == set(GROUPS)
    assert res["best_n"] <= 4 and "no SHAP" in res["importance_note"]
    assert set(res["path"][-1]["recall"]) == set(GROUPS)


def test_required_genes_go_first_and_count_towards_the_size():
    a = _data()
    plan = mp.make_plan(_Mem(a), "group", size=3, required=["G40"])
    res = mp.run(_Mem(a), plan)
    assert res["panel"][0] == "G40" and len(res["panel"]) == 3
    assert res["path"][0]["required"] and not res["path"][1]["required"]
    assert res["genes"][0]["required"]


def test_screening_matches_trying_every_candidate_here(monkeypatch):
    a = _data()
    plan = mp.make_plan(_Mem(a), "group", size=3, per_group=4)
    screened = mp.run(_Mem(a), plan)["path"]
    monkeypatch.setattr(mp, "SCREEN", 1000)
    full = mp.run(_Mem(a), plan)["path"]
    assert [p["balanced_accuracy"] for p in screened] == pytest.approx(
        [p["balanced_accuracy"] for p in full], abs=0.02)


def test_stop_cancels_before_the_next_step():
    a = _data()
    plan = mp.make_plan(_Mem(a), "group", size=4)
    calls = {"n": 0}

    def stop():
        calls["n"] += 1
        return calls["n"] > 2

    gen = mp.iter_run(_Mem(a), plan, stop=stop)
    events = []
    with pytest.raises(mp.Cancelled):
        for ev, _ in gen:
            events.append(ev)
    assert events == ["start", "step", "step"]


def test_reads_the_raw_log_values_when_x_is_scaled():
    a = _data()
    a.raw = a.copy()
    sc.pp.scale(a)
    plan = mp.make_plan(_Mem(a), "group", size=2)
    X = mp.read_matrix(_Mem(a), plan.candidates, plan.cells)
    assert X.min() >= 0
    j = a.raw.var_names.get_loc(plan.candidates[0])
    np.testing.assert_allclose(X[:, 0], a.raw.X[plan.cells][:, j].toarray().ravel(), rtol=1e-6)


def test_grounding_line_once_a_panel_is_made(tmp_path):
    a = _data()
    p = tmp_path / "d.h5ad"
    a.write_h5ad(p)
    adaptor = AnnDataAdaptor(str(p))
    _, sources = build_grounding_context(adaptor)
    assert mp.TAG not in {s.ref for s in sources}
    mp.run(adaptor, mp.make_plan(adaptor, "group", size=3))
    context, sources = build_grounding_context(adaptor)
    fact = next(s for s in sources if s.ref == mp.TAG)
    assert f"[{mp.TAG}]" in context and "Minimal marker panel for 'group'" in fact.evidence
    assert "balanced accuracy" in fact.evidence


# --- endpoints ----------------------------------------------------------------------------


def _register(dm, a: ad.AnnData, did: str = "pan") -> str:
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


async def test_endpoint_streams_the_path_and_keeps_the_result(client, dataset_manager):
    did = _register(dataset_manager, _data())
    r = await client.get(f"{API}/datasets/{did}/markers/panel")
    assert r.status_code == 200 and r.json()["result"] is None
    r = await client.post(f"{API}/datasets/{did}/markers/panel", json={"groupby": "group", "size": 3})
    assert r.status_code == 200, r.text
    ev = _events(r.text)
    assert [e for e, _ in ev] == ["start", "step", "step", "step", "baseline", "result", "complete"]
    assert ev[0][1]["n_cells"] == 480 and ev[0][1]["cv"].startswith("stratified 3-fold")
    held = (await client.get(f"{API}/datasets/{did}/markers/panel")).json()["result"]
    assert held["panel"] == ev[-2][1]["panel"]


async def test_endpoint_refusals(client, dataset_manager):
    a = _data()
    a.obs["other"] = a.obs["group"].astype(str)
    did = _register(dataset_manager, a)
    url = f"{API}/datasets/{did}/markers/panel"
    assert (await client.post(url, json={"groupby": "nope"})).status_code == 404
    r = await client.post(url, json={"groupby": "other"})
    assert r.status_code == 400 and "Markers table" in r.json()["detail"]
    assert (await client.post(url, json={"groupby": "group", "size": 51})).status_code == 422
    assert (await client.post(f"{API}/datasets/none/markers/panel", json={"groupby": "g"})).status_code == 404


async def test_large_dataset_reads_the_candidates_from_the_gene_store(client, tmp_data_dir):
    counts = counts_adata()
    register(tmp_data_dir, "bigP", counts)
    up = tmp_data_dir / "uploads" / "smallP"
    up.mkdir(parents=True)
    small_twin(counts).write_h5ad(up / "twin.h5ad")
    (up / "metadata.json").write_text(json.dumps({"id": "smallP", "name": "twin", "status": "ready"}))
    out = {}
    for did in ("bigP", "smallP"):
        r = await client.get(f"{API}/datasets/{did}/markers",
                             params={"groupby_column": "leiden", "format": "json"})
        assert r.status_code == 200, r.text
        r = await client.post(f"{API}/datasets/{did}/markers/panel", json={"groupby": "leiden", "size": 3})
        assert r.status_code == 200, r.text
        out[did] = _events(r.text)[-2][1]
    assert out["bigP"]["panel"] == out["smallP"]["panel"]
    assert out["bigP"]["balanced_accuracy"] == pytest.approx(out["smallP"]["balanced_accuracy"], abs=1e-6)
