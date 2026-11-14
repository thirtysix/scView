"""The velocity direction check (core/velocity_direction.py): orders, cross-boundary scores,
warnings, and the check in a run and over the API."""

from __future__ import annotations

import json

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from scview.core import trajectory
from scview.core import velocity as vel
from scview.core import velocity_direction as vd
from scview.core import velocity_input as vi
from tests.test_velocity import API, _run, _sse, dataset_with_store, run_core
from tests.velocity_fixture import Adaptor, splicing_counts


def _obs_adata(obs: dict, n: int | None = None) -> ad.AnnData:
    n = n or len(next(iter(obs.values())))
    return ad.AnnData(sp.csr_matrix((n, 1), dtype=np.float32), obs=pd.DataFrame(obs, index=[f"c{i}" for i in range(n)]))


def _line(n_per=200, groups=("A", "B", "C"), seed=0):
    """Groups side by side along x, each a band of n_per cells; returns coords and labels."""
    rng = np.random.default_rng(seed)
    X = np.vstack([np.c_[rng.uniform(i, i + 1, n_per), rng.uniform(0, 1, n_per)] for i in range(len(groups))])
    labels = np.repeat(np.asarray(groups, dtype=object), n_per)
    return X, labels


def _ordering(labels, t, transitions, kind="order"):
    return vd.Ordering(kind, "test order", "g", labels, np.asarray(t, dtype=np.float64), transitions)


# -- resolving an order ---------------------------------------------------------------


def test_order_gives_consecutive_transitions_and_ranks():
    a = _obs_adata({"g": pd.Categorical(["A", "B", "C", "A", None])})
    o, why = vd.resolve(a, {"order": ["A", "B", "C"]}, "g")
    assert why is None and o.kind == "order" and o.transitions == [("A", "B"), ("B", "C")]
    assert o.t[:4].tolist() == [0, 1, 2, 0] and np.isnan(o.t[4]) and o.labels[4] is None
    with pytest.raises(vd.DirectionError, match="no group 'D'"):
        vd.resolve(a, {"order": ["A", "D"]}, "g")


def test_transitions_give_depths_and_a_cycle_is_refused():
    a = _obs_adata({"g": ["A", "B", "C", "D"]})
    o, _ = vd.resolve(a, {"transitions": [["A", "B"], ["B", "C"], ["A", "D"], ["D", "C"]]}, "g")
    assert o.t.tolist() == [0, 1, 2, 1]
    with pytest.raises(vd.DirectionError, match="circle"):
        vd.resolve(a, {"transitions": [["A", "B"], ["B", "A"]]}, "g")


def test_root_follows_the_paga_tree_of_that_column():
    a = _obs_adata({"g": ["A", "B", "C", "D"]})
    with pytest.raises(vd.DirectionError, match="run PAGA"):
        vd.resolve(a, {"root": "A"}, "g")
    a.uns[trajectory.PAGA_KEY] = json.dumps({"groupby": "g", "groups": ["A", "B", "C", "D"], "tree": [[0, 1], [1, 2], [1, 3]]})
    o, _ = vd.resolve(a, {"root": "B"}, "g")
    assert sorted(o.transitions) == [("B", "A"), ("B", "C"), ("B", "D")]
    assert o.t.tolist() == [1, 0, 1, 1]


def test_columns_numeric_stage_ordered_and_given_order():
    a = _obs_adata({"g": ["A", "A", "B", "B"], "pt": [0.1, 0.2, np.inf, 0.9], "stage": ["E7.5", "E7.5", "E8.0", "E8.25"],
                    "day": pd.Categorical(["late", "early", "late", "mid"], categories=["early", "mid", "late"], ordered=True),
                    "word": ["x", "y", "x", "y"], "flag": [True, False, True, False]})
    o, _ = vd.resolve(a, {"column": "pt"}, "g")
    assert o.kind == "column" and o.transitions is None and np.isnan(o.t[2]) and o.t[3] == 0.9
    assert vd.resolve(a, {"column": "stage"}, "g")[0].t.tolist() == [7.5, 7.5, 8.0, 8.25]
    assert vd.resolve(a, {"column": "day"}, "g")[0].t.tolist() == [2, 0, 2, 1]
    assert vd.resolve(a, {"column": "word", "column_order": ["y", "x"]}, "g")[0].t.tolist() == [1, 0, 1, 0]
    with pytest.raises(vd.DirectionError, match="give its values in order"):
        vd.resolve(a, {"column": "word"}, "g")
    with pytest.raises(vd.DirectionError, match="true/false"):
        vd.resolve(a, {"column": "flag"}, "g")
    # The stage column as the groups too: its own order, step by step.
    o, _ = vd.resolve(a, {"column": "stage", "group_by": "stage"}, None)
    assert o.transitions == [("E7.5", "E8.0"), ("E8.0", "E8.25")]


def test_default_is_scviews_pseudotime_else_unchecked():
    a = _obs_adata({"g": ["A", "B", "A", "B"]})
    o, why = vd.resolve(a, None, "g")
    assert o is None and "no pseudotime" in why
    a.obs[trajectory.DPT_COLUMN] = [0.0, 1.0, 0.1, 0.9]
    a.uns[trajectory.PARAMS_KEY] = json.dumps({"root_column": "g", "root_value": "A"})
    o, _ = vd.resolve(a, {}, "g")
    assert o.kind == "pseudotime" and "from g = A" in o.label
    assert vd.resolve(a, None, None)[0] is None  # no groups to check between
    with pytest.raises(vd.DirectionError, match="continuous"):
        vd.resolve(a, None, trajectory.DPT_COLUMN)


# -- the scores -------------------------------------------------------------------------


def test_arrows_along_the_order_score_forward_and_against_it_backward():
    X, labels = _line()
    V = np.tile([1.0, 0.0], (X.shape[0], 1))
    o = _ordering(labels, np.repeat([0, 1, 2], 200), [("A", "B"), ("B", "C")])
    d = vd.evaluate(o, X, V, np.arange(X.shape[0]), "X_umap")
    assert [r["status"] for r in d["transitions"]] == ["forward", "forward"]
    assert all(r["mean_cos"] > 0.8 and r["share_forward"] == 1 and r["n_cells"] >= vd.MIN_BOUNDARY for r in d["transitions"])
    assert d["n_backward"] == 0 and vd.warnings(d) == []
    assert d["cells"]["mean_cos"] > 0.8 and d["cells"]["n_cells"] < X.shape[0]  # only cells next to another group
    back = vd.evaluate(o, X, -V, np.arange(X.shape[0]), "X_umap")
    assert back["n_backward"] == 2 and all(r["high"] < 0 for r in back["transitions"])
    w = vd.warnings(back)
    assert len(w) == 1 and "A -> B (mean cosine -" in w[0] and "boundary cells, 0% forward" in w[0] and "X_umap" in w[0]


def test_sideways_noise_is_unclear_and_a_far_pair_has_no_boundary():
    X, labels = _line()
    rng = np.random.default_rng(1)
    V = rng.normal(size=X.shape)
    o = _ordering(labels, np.repeat([0, 1, 2], 200), [("A", "B"), ("A", "C")])
    d = vd.evaluate(o, X, V, np.arange(X.shape[0]), "X_umap")
    assert d["transitions"][0]["status"] == "unclear"
    assert d["transitions"][1] == {"from": "A", "to": "C", "n_cells": 0, "status": "no_boundary"}


def test_cells_without_arrows_are_left_out_and_positions_map_to_the_dataset():
    X, labels = _line(n_per=60)
    V = np.tile([1.0, 0.0], (X.shape[0], 1))
    V[:5] = np.nan
    # The run holds every other cell of the dataset: positions say which.
    full_labels = np.repeat(labels, 2)
    full_t = np.repeat(np.repeat([0.0, 1.0, 2.0], 60), 2)
    o = _ordering(full_labels, full_t, [("A", "B")])
    d = vd.evaluate(o, X, V, np.arange(0, 2 * X.shape[0], 2), "X_umap")
    assert d["transitions"][0]["status"] == "forward" and d["cells"]["n_run_cells"] == X.shape[0]


def test_a_numeric_order_derives_transitions_where_it_separates_the_groups():
    X, labels = _line(groups=("A", "B", "C", "D"))
    rng = np.random.default_rng(2)
    t = np.r_[rng.uniform(0, 1, 200), rng.uniform(1, 2, 200), rng.uniform(2, 3, 200), rng.uniform(2, 3, 200)]
    V = np.tile([1.0, 0.0], (X.shape[0], 1))
    d = vd.evaluate(_ordering(labels, t, None, "column"), X, V, np.arange(X.shape[0]), "X_umap")
    assert [(r["from"], r["to"]) for r in d["transitions"]] == [("A", "B"), ("B", "C")]
    assert d["skipped"] == [{"groups": ["C", "D"], "auc": pytest.approx(0.5, abs=0.1)}]
    assert d["derived"] and d["order_auc"] == vd.ORDER_AUC
    assert d["cells"]["n_cells"] == 800 and d["cells"]["mean_cos"] > 0.1  # t is noise inside each group


def test_few_boundary_cells_get_no_verdict():
    X, labels = _line(n_per=200, groups=("A", "B"))
    labels = labels.copy()
    X = X.copy()
    X[200:, 0] += 5  # B far away ...
    X[:5, 0] = 6.5  # ... but for five cells of A inside it
    o = _ordering(labels, np.repeat([0, 1], 200), [("A", "B")])
    d = vd.evaluate(o, X, np.tile([1.0, 0.0], (400, 1)), np.arange(400), "X_umap")
    r = d["transitions"][0]
    assert 0 < r["n_cells"] < vd.MIN_BOUNDARY and r["status"] == "few_cells"


# -- in a run and over the API -------------------------------------------------------------


def test_run_checks_the_direction_it_is_given(tmp_path):
    _, p, out = run_core(tmp_path)
    assert out["direction"]["checked"] is False and "no pseudotime" in out["direction"]["reason"]
    adata = splicing_counts()
    adata.obs["time"] = adata.obsm["X_umap"][:, 0]
    vi.adopt_anndata(adata, "ds", tmp_path / "b", file_name="x.h5ad")
    for spec, want in (({"order": ["early", "late"]}, "forward"), ({"order": ["late", "early"]}, "backward")):
        params = vel.Params(n_top_genes=200, n_neighbors=20, n_pcs=10, direction=spec)
        p = vel.prepare(Adaptor(adata), tmp_path / "b", "ds", params)
        out = next(ev["out"] for ev in vel.iter_run(p) if ev["type"] == "result")
        d = out["direction"]
        assert d["embedding"] == "X_umap" and d["transitions"][0]["status"] == want, d
    params = vel.Params(n_top_genes=200, n_neighbors=20, n_pcs=10, direction={"column": "time"})
    p = vel.prepare(Adaptor(adata), tmp_path / "b", "ds", params)
    out = next(ev["out"] for ev in vel.iter_run(p) if ev["type"] == "result")
    assert out["direction"]["cells"]["mean_cos"] > 0.3
    assert [(r["from"], r["to"]) for r in out["direction"]["transitions"]] == [("early", "late")]
    with pytest.raises(vel.VelocityError, match="no group"):
        vel.prepare(Adaptor(adata), tmp_path / "b", "ds", vel.Params(direction={"order": ["early", "nope"]}))


async def test_api_checks_after_the_run_keeps_the_check_and_its_warning(client, dataset_manager):
    did, _a = dataset_with_store(dataset_manager, "veldir")
    r = await client.post(f"{API}/datasets/{did}/velocity/direction", json={"order": ["early", "late"]})
    assert r.status_code == 404 and "Run velocity" in r.json()["detail"]
    events = await _run(client, did, direction={"order": ["early", "late"]}, embedding="X_umap")
    run = dict(events)["result"]["run"]
    assert run["direction"]["transitions"][0]["status"] == "forward"
    assert not any("backward" in w for w in run["warnings"])

    r = await client.post(f"{API}/datasets/{did}/velocity/direction", json={"order": ["late", "early"]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["direction"]["transitions"][0]["status"] == "backward"
    assert any("point backward" in w for w in body["warnings"])
    s = (await client.get(f"{API}/datasets/{did}/velocity")).json()
    assert s["run"]["direction"]["transitions"][0]["status"] == "backward"
    assert any("point backward" in w for w in s["run"]["warnings"])
    assert s["run"]["result_id"] == run["result_id"]
    dataset_manager.evict(did)  # saved, not only in the cached copy
    adaptor = await dataset_manager.get_or_load_dataset(did)
    assert vel.saved(adaptor.adata)["direction"]["spec"]["order"] == ["late", "early"]

    r = await client.post(f"{API}/datasets/{did}/velocity/direction", json={"order": ["early", "nope"]})
    assert r.status_code == 422 and "no group 'nope'" in r.json()["detail"]
    r = await client.post(f"{API}/datasets/{did}/velocity/direction", json={"embedding": "X_nope", "order": ["early", "late"]})
    assert r.status_code == 422 and "2D plot" in r.json()["detail"]
    r = await client.post(f"{API}/datasets/{did}/velocity/direction", json={})
    assert r.status_code == 200 and r.json()["direction"]["checked"] is False
    s = (await client.get(f"{API}/datasets/{did}/velocity")).json()
    assert not any("point backward" in w for w in s["run"]["warnings"])  # the warning went with its check


async def test_run_with_an_order_that_names_nothing_is_refused_before_it_runs(client, dataset_manager):
    did, _a = dataset_with_store(dataset_manager, "veldir2")
    r = await client.post(f"{API}/datasets/{did}/velocity/run-stream",
                          json={"n_top_genes": 200, "n_neighbors": 20, "direction": {"root": "early"}})
    events = _sse(r.text)
    assert events[-1][0] == "error" and "PAGA" in events[-1][1]["error"]
