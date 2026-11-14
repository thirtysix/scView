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


# -- genes that turn the direction (gene_flags) ----------------------------------------------


def _flag_stores(tmp_path, n_a=100, n_b=100):
    """Two groups; moments = the counts (identity weights). murk: spliced doubles A -> B, its
    unspliced share triples, so with B's gamma A reads as switching off. induced: spliced doubles
    with the unspliced share falling (unspliced leads). faint: barely expressed in A."""
    from scview.core import modality_store as ms

    rng = np.random.default_rng(0)
    n = n_a + n_b
    names = [f"c{i}" for i in range(n)]
    in_b = np.arange(n) >= n_a
    s = {"murk": np.where(in_b, 4.0, 2.0), "induced": np.where(in_b, 4.0, 2.0), "faint": np.where(in_b, 4.0, 0.05)}
    u = {"murk": s["murk"] * np.where(in_b, 0.3, 0.1), "induced": s["induced"] * np.where(in_b, 0.2, 0.6),
         "faint": s["faint"] * np.where(in_b, 0.3, 0.1)}
    genes = list(s)
    S = np.c_[[s[g] * rng.uniform(0.9, 1.1, n) for g in genes]]  # genes x cells
    U = np.c_[[u[g] * rng.uniform(0.9, 1.1, n) for g in genes]]
    run_path, fit_path = tmp_path / "run.h5", tmp_path / "fit.h5"
    ms.write(run_path, {"connectivities": sp.identity(n, format="csr", dtype=np.float32)}, names, names)
    ms.write(fit_path, {"spliced": sp.csr_matrix(S, dtype=np.float32), "unspliced": sp.csr_matrix(U, dtype=np.float32)},
             genes, names)
    table = {"gene": genes + ["unfit"], "gamma": [0.3, 0.3, 0.3, 0.3], "r2": [0.5, 0.9, 0.4, 0.8],
             "velocity_gene": [True, True, True, False]}
    labels = np.where(in_b, "B", "A").astype(object)
    return run_path, fit_path, table, labels


def test_gene_flags_find_rising_expression_read_as_switching_off(tmp_path):
    run_path, fit_path, table, labels = _flag_stores(tmp_path)
    o = _ordering(labels, np.where(labels == "B", 1.0, 0.0), [("A", "B")])
    steps = [{"from": "A", "to": "B", "status": "backward"}]
    f = vd.gene_flags(o, steps, run_path, fit_path, np.arange(labels.size), table)
    assert f["checked"] and f["n_velocity_genes"] == 3 and f["n_flagged"] == 1
    g = f["genes"][0]
    assert g["gene"] == "murk" and g["rank"] == 2  # by r2 among velocity genes: induced 0.9, murk 0.5, faint 0.4
    st = g["steps"][0]
    assert st["from"] == "A" and st["status"] == "backward" and st["share_negative"] == 1.0
    assert st["ratio_to"] / st["ratio_from"] == pytest.approx(3, rel=0.1) and st["spliced_to"] > st["spliced_from"]
    assert f["n_in_top"] == 1 and f["top_n"] == 3 and "MURK" in f["rule"]
    # In the warning a backward step earns, with how many of the top genes they are.
    d = {"checked": True, "label": "x", "embedding": "X_umap", "gene_flags": f,
         "transitions": [{"from": "A", "to": "B", "status": "backward", "mean_cos": -0.3, "n_cells": 50, "share_forward": 0.2}]}
    assert "1 of 3 velocity genes show that pattern here (murk; 1 in the top 3)" in vd.warnings(d)[0]
    # Cells the dataset no longer has are left out; too few on a side: no flags.
    few = vd.gene_flags(o, steps, run_path, fit_path, np.r_[np.arange(10), np.full(190, -1)], table)
    assert few["checked"] is False and "20+" in few["reason"]


def test_run_flags_a_planted_murk_gene_and_the_api_keeps_params_and_warnings(tmp_path):
    a = splicing_counts()
    t = a.obsm["X_umap"][:, 0] / 10
    rng = np.random.default_rng(3)
    late = t >= 0.5
    # Levels after each cell is scaled to the median total (the fixture's totals grow with t):
    # spliced rising along the order, the unspliced share stepping up in the late group.
    tot_s = np.asarray(a.layers["spliced"].sum(axis=1)).ravel()
    tot_u = np.asarray(a.layers["unspliced"].sum(axis=1)).ravel()
    level = 20 + 60 * t
    s = level * tot_s / np.median(tot_s)
    u = level * np.where(late, 0.5, 0.15) * tot_u / np.median(tot_u)
    extra_s = np.c_[[rng.poisson(s * k) for k in (1.0, 1.1, 0.9)]].T
    extra_u = np.c_[[rng.poisson(u * k) for k in (1.0, 1.1, 0.9)]].T
    b = ad.AnnData(sp.csr_matrix(np.c_[a.X.toarray(), extra_s], dtype=np.float32), obs=a.obs.copy(),
                   var=pd.DataFrame(index=[*a.var_names, "M0", "M1", "M2"]))
    b.layers["spliced"] = sp.csr_matrix(np.c_[a.layers["spliced"].toarray(), extra_s], dtype=np.float32)
    b.layers["unspliced"] = sp.csr_matrix(np.c_[a.layers["unspliced"].toarray(), extra_u], dtype=np.float32)
    b.obsm["X_umap"] = a.obsm["X_umap"]
    vi.adopt_anndata(b, "ds", tmp_path, file_name="x.h5ad")
    params = vel.Params(n_top_genes=300, n_neighbors=20, n_pcs=10, direction={"order": ["early", "late"]})
    p = vel.prepare(Adaptor(b), tmp_path, "ds", params)
    out = next(ev["out"] for ev in vel.iter_run(p) if ev["type"] == "result")
    flags = out["direction"]["gene_flags"]  # computed inside the run, from its stores
    named = {g["gene"] for g in flags["genes"]}
    assert {"M0", "M1", "M2"} <= named, flags
    assert len(named) <= 10  # the synthetic dynamic genes switch on with unspliced leading: not flagged


def test_copilot_reads_the_check_and_routes_its_words(tmp_path):
    from scview.core.rag import router

    adata = splicing_counts()
    vi.adopt_anndata(adata, "ds", tmp_path, file_name="x.h5ad")
    params = vel.Params(n_top_genes=200, n_neighbors=20, n_pcs=10, direction={"order": ["late", "early"]})
    p = vel.prepare(Adaptor(adata), tmp_path, "ds", params)
    events = list(vel.iter_run(p))
    vel.save(adata, p, next(ev["out"] for ev in events if ev["type"] == "result"), {"result_id": "g1"})
    lines, _ = vel.grounding_lines(Adaptor(adata))
    assert any(x.startswith("- Direction checked against the order of clusters given on X_umap: late -> early backward")
               for x in lines)
    assert any("point backward" in x for x in lines)
    for q in ("are the arrows pointing the wrong way", "what are murk genes doing here", "did the direction check pass"):
        assert "groups" in router.named_result_facets(q), q


async def test_stored_result_has_its_params_and_warnings(client, dataset_manager):
    did, _a = dataset_with_store(dataset_manager, "velres")
    events = await _run(client, did, direction={"order": ["late", "early"]}, exclude_genes=["G0"])
    rid = dict(events)["result"]["run"]["result_id"]
    r = (await client.get(f"{API}/datasets/{did}/results/{rid}")).json()
    assert r["kind"] == "velocity" and r["params"]["method"] == "stochastic" and r["params"]["n_top_genes"] == 200
    assert r["params"]["n_excluded"] == 1 and r["params"]["exclude_given"] == ["G0"]
    assert r["params"]["direction"]["order"] == ["late", "early"] and r["params"]["direction"]["embedding"] == "X_umap"
    assert any("point backward" in w for w in r["warnings"])
    await client.post(f"{API}/datasets/{did}/velocity/direction", json={"order": ["early", "late"]})
    r = (await client.get(f"{API}/datasets/{did}/results/{rid}")).json()
    assert r["params"]["direction"]["order"] == ["early", "late"] and not any("point backward" in w for w in r["warnings"])
    body = (await client.get(f"{API}/datasets/{did}/velocity")).json()["run"]["direction"]
    assert body["gene_flags"]["checked"] is True and "rule" in body["gene_flags"]
