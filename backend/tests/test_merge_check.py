"""Merge check: count splitting + held-out tests (core/merge_check.py + /clustering/merge-check)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import anndata as ad

from scview.core import merge_check as mc
from scview.core import provenance, replay_script
from scview.core.pipeline import PipelineParams, run_pipeline
from tests.test_hvg_sensitivity import STEPS, _events, _register

API = "/api/v1"


@pytest.fixture(autouse=True)
def _empty_cache():
    mc._stages.clear()
    yield
    mc._stages.clear()


def _nb(rng, mu, theta):
    return rng.negative_binomial(theta, theta / (theta + mu))


def _types(n_per: int = 200, g: int = 300, seed: int = 0) -> ad.AnnData:
    """Three types with 30 strong markers each, NB counts (theta 5): over-clustered,
    each type splits into pieces that are one population."""
    rng = np.random.default_rng(seed)
    types = np.repeat(np.arange(3), n_per)
    base = rng.gamma(2.0, 1.0, g)
    mu = np.tile(base, (types.size, 1)) * rng.lognormal(0, 0.2, types.size)[:, None]
    for t in range(3):
        mu[types == t, 30 * t:30 * t + 30] *= 8
    a = ad.AnnData(sp.csr_matrix(_nb(rng, mu, 5.0).astype("float32")))
    a.var_names = [f"G{i}" for i in range(g)]
    a.obs_names = [f"c{i}" for i in range(types.size)]
    a.obs["type"] = pd.Categorical([f"T{t}" for t in types])
    return a


def _processed(resolution: float = 2.0) -> ad.AnnData:
    a, res = run_pipeline(_types(), STEPS, PipelineParams(n_top_genes=200, n_comps=15,
                                                          clustering_resolution=resolution))
    assert not res.errors
    return a


# --- overdispersion and the split ---------------------------------------------------------


def test_overdispersion_recovers_theta_and_poisson_genes():
    rng = np.random.default_rng(1)
    n, g = 3000, 200
    totals = rng.lognormal(8, 0.4, n)
    p = rng.lognormal(-8, 1.0, g)
    mu = totals[:, None] * p[None, :]
    X = np.where(np.arange(g) < 100, _nb(rng, mu, 2.0), rng.poisson(mu)).astype(float)
    theta = mc._fit_theta(sp.csr_matrix(X), totals)
    assert 1.6 < np.median(theta[:100]) < 2.5
    assert np.median(theta[100:]) > 50  # Poisson genes: near THETA_MAX or far above 2
    smoothed, info = mc.estimate_overdispersion(sp.csr_matrix(X), np.random.default_rng(0))
    assert info["n_genes_fit"] == g and smoothed.shape == (g,) and np.all(smoothed > 0)


def test_split_keeps_every_count_and_halves_it():
    rng = np.random.default_rng(0)
    X = sp.csr_matrix(_nb(rng, np.full((2000, 20), 4.0), 2.0).astype("float32"))
    train, test = mc.split_counts(X, np.full(20, 2.0), np.random.default_rng(0))
    assert (train + test - X).nnz == 0
    assert abs(train.sum() / X.sum() - mc.EPSILON) < 0.02
    assert np.all(train.data >= 0) and np.all(test.data >= 0)


def test_nb_splitting_makes_the_halves_independent_and_poisson_does_not():
    """Neufeld et al. 2023, Theorems 2 and 3: at the true overdispersion the halves of
    a homogeneous population are uncorrelated; Poisson splitting correlates them."""
    rng = np.random.default_rng(0)
    theta = 1.0
    X = sp.csr_matrix(_nb(rng, np.full((20000, 5), 5.0), theta).astype("float32"))

    def corr(th):
        tr, te = mc.split_counts(X, np.full(5, th), np.random.default_rng(1))
        tr, te = tr.toarray(), te.toarray()
        return np.mean([np.corrcoef(tr[:, j], te[:, j])[0, 1] for j in range(5)])

    assert abs(corr(theta)) < 0.03
    assert corr(mc.THETA_MAX) > 0.5  # theory: 0.83 at mu 5, b 1


# --- pairs, matching, merging -------------------------------------------------------------


def test_adjacency_and_neighbour_pairs():
    codes = np.array([0, 0, 1, 1, 2, 2])
    graph = sp.csr_matrix(np.array([
        [0, 1, 1, 0, 0, 0], [1, 0, 0, 0, 0, 0], [1, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 1, 0]], dtype=float))
    A = mc.adjacency(graph, codes, 3)
    assert A[0, 1] == pytest.approx(1 / 3) and A[0, 2] == 0 and A[1, 1] == 0  # 1 of 3 edge ends
    assert mc.neighbour_pairs(A) == [(0, 1)]


def _row(i, j, n):
    return {"i": i, "j": j, "a": str(i), "b": str(j), "tested": True, "n_genes": n, "up": [], "down": []}


def test_supported_groups_join_indistinct_pairs_but_never_a_distinct_one():
    # 0~1 and 1~2 look alike, but 0 vs 2 is distinct: 2 must stay out of 0's group.
    groups = mc.supported_groups(4, [_row(0, 1, 0), _row(1, 2, 5), _row(0, 2, 80), _row(2, 3, 2)], 10)
    assert groups[0] == groups[1] and groups[2] == groups[3] and groups[0] != groups[2]
    assert len(set(mc.supported_groups(3, [_row(0, 1, 50)], 10))) == 3


def _plan(names, codes):
    rec = mc.hs.Recorded(reference="cl", method="leiden", resolution=1.0, flavor="seurat", n_top_genes=200,
                         hvg_batch_key=None, target_sum=1e4, n_comps=10, n_neighbors=15, batch_key=None)
    return mc.Plan("cl", True, np.asarray(codes), names, rec)


def _stage(train_codes, tested, A_ref):
    k = int(max(train_codes)) + 1
    return mc._Stage(train_codes=np.asarray(train_codes), train_names=[str(i) for i in range(k)],
                     train_resolution=1.0, comparison={"ari": 0.5, "ami": 0.5, "sizes": []},
                     A_ref=np.asarray(A_ref, dtype=float), tested=tested, theta_info={}, seconds={})


def test_summarise_reads_pairs_of_the_checked_clustering():
    # Checked clusters a, b, c, d (10 cells each). Training: a and b fall in training 0
    # and 1 (tested alike); c and d share training 2, distinct from 1.
    codes = [0] * 10 + [1] * 10 + [2] * 10 + [3] * 10
    train = [0] * 10 + [1] * 10 + [2] * 20
    A = np.array([[0, .3, 0, 0], [.3, 0, .2, 0], [0, .2, 0, .4], [0, 0, .4, 0]])
    s = _stage(train, [_row(0, 1, 3), _row(1, 2, 150)], A)
    r = mc.summarise(_plan(["a", "b", "c", "d"], codes), s, max_genes=10)
    by = {tuple(sorted((x["a"], x["b"]))): x for x in r["pairs"]}
    assert by[("a", "b")]["candidate"] and by[("a", "b")]["n_genes"] == 3
    assert by[("b", "c")]["evidence"] == "held_out" and not by[("b", "c")]["candidate"]
    assert by[("b", "c")]["n_genes"] == 150
    assert by[("c", "d")]["evidence"] == "together" and not by[("c", "d")]["candidate"]
    assert r["n_candidates"] == 1 and sorted(map(sorted, r["groups"])) == [["a", "b"], ["c", "d"]]
    # A stricter threshold reads the same tests again: a and b are now kept apart.
    r = mc.summarise(_plan(["a", "b", "c", "d"], codes), s, max_genes=2)
    assert r["n_candidates"] == 0 and r["thresholds"]["merge_max_genes"] == 2


def test_summarise_reads_the_test_between_the_pairs_own_parts():
    # One group of three training clusters; a, b, c each in its own. a + b reads the
    # test of training 0 vs 1 (2 genes), not the group's most distinct (8).
    codes = [0] * 10 + [1] * 10 + [2] * 10
    A = np.array([[0, .3, .1], [.3, 0, .2], [.1, .2, 0]])
    s = _stage(codes, [_row(0, 1, 2), _row(1, 2, 8)], A)
    by = {tuple(sorted((x["a"], x["b"]))): x for x in mc.summarise(_plan(["a", "b", "c"], codes), s, 10)["pairs"]}
    assert by[("a", "b")]["n_genes"] == 2 and by[("a", "b")]["train_pairs"] == ["0+1"]
    assert by[("b", "c")]["n_genes"] == 8
    assert by[("a", "c")]["candidate"] and by[("a", "c")]["n_genes"] == 8  # no direct test: the group's


def test_resolution_for_count_reaches_the_reference_count():
    a = _processed(resolution=1.0)
    graph = a.obsp["connectivities"]
    labels = mc._cluster_graph(graph, "leiden", 0.5)
    target = len(mc._cluster_graph(graph, "leiden", 2.0).categories)
    assert len(labels.categories) < target
    r, found = mc.resolution_for_count(graph, "leiden", 0.5, labels, target)
    # The search counts with two Leiden passes; the full run can land a cluster or two off.
    assert abs(len(found.categories) - target) <= 2 and 0.5 < r <= 2.0
    assert len(found.categories) > len(labels.categories)


def test_merged_labels_join_sets_in_the_columns_order():
    s = pd.Series(pd.Categorical(["a", "b", "c", "d", None], categories=["d", "c", "b", "a"]))
    out = mc.merged_labels(s, [("a", "c"), ("c", "b")])
    assert list(out.categories) == ["d", "c+b+a"]
    assert out.astype(object).tolist()[:4] == ["c+b+a", "c+b+a", "c+b+a", "d"] and pd.isna(out[4])


def test_clean_merges_refuses_unknown_or_self_pairs():
    with pytest.raises(mc.MergeCheckError, match="at least one"):
        mc.clean_merges(["0", "1"], [])
    with pytest.raises(mc.MergeCheckError, match="two different"):
        mc.clean_merges(["0", "1"], [["0", "0"]])
    with pytest.raises(mc.MergeCheckError, match="No cluster '7'"):
        mc.clean_merges(["0", "1"], [["0", "7"]])


# --- the run --------------------------------------------------------------------------------


def test_plan_takes_the_active_clustering_or_a_sweep_resolution():
    a = _processed()
    p = mc.plan(a)
    assert p.reference == "scview_leiden_r2.0" and p.exists and p.rec.resolution == 2.0
    q = mc.plan(a, resolution=0.3)
    assert q.reference == "scview_leiden_r0.3" and not q.exists and q.rec.resolution == 0.3
    assert q.codes.size == a.n_obs
    no_counts = a.copy()
    del no_counts.layers["counts"]
    no_counts.raw = None
    X = no_counts.X.toarray() if sp.issparse(no_counts.X) else np.asarray(no_counts.X)  # X stays sparse now
    no_counts.X = sp.csr_matrix(np.abs(X) + 0.5)
    with pytest.raises(mc.MergeCheckError, match="raw counts"):
        mc.plan(no_counts)


def test_over_split_pieces_merge_and_distinct_types_do_not():
    a = _processed()
    p = mc.plan(a)
    obs_before = list(a.obs.columns)
    steps = list(mc.iter_merge_check(a, p))
    assert [s.get("step") for s in steps[1:-1]][:3] == ["overdispersion", "split", "train"]
    r = steps[-1]
    assert list(a.obs.columns) == obs_before
    assert r["train"]["n_clusters"] == r["n_reference_clusters"]  # matched count
    type_of = pd.crosstab(a.obs[p.reference], a.obs["type"]).idxmax(axis=1).astype(str).to_dict()
    # The groups the held-out half can't split are the three types.
    assert sorted(sorted({type_of[c] for c in g}) for g in r["groups"] if g) == [["T0"], ["T1"], ["T2"]]
    cands = [x for x in r["pairs"] if x["candidate"]]
    assert cands and all(type_of[x["a"]] == type_of[x["b"]] for x in cands)
    assert all(x["n_genes"] < mc.MERGE_MAX_GENES for x in cands)
    assert r["n_candidates"] == len(cands)
    # Again: the tests are re-read, not re-run.
    again = list(mc.iter_merge_check(a, p, max_genes=1))
    assert again[0]["cached"] is True and len(again) == 2
    assert again[-1]["thresholds"]["merge_max_genes"] == 1


def test_save_writes_the_merged_column_with_provenance_and_replay():
    a = _processed()
    p = mc.plan(a)
    names = p.names
    res = mc.save(a, p, [[names[0], names[1]]])
    col = "scview_leiden_r2.0_merged"
    assert res["column"] == col and res["n_clusters"] == len(names) - 1
    assert f"{names[0]}+{names[1]}" in a.obs[col].cat.categories
    step = provenance.read_provenance(a)["history"][-1]
    assert step["step"] == "merge_clusters" and step["params"]["target"] == col
    script = replay_script.build_script(a)
    compile(script, "replay.py", "exec")
    assert replay_script.describe(a)[-1]["status"] == "exact"
    # The replayed line gives the saved labels.
    merged = step["params"]["mapping"]
    again = pd.Categorical(a.obs["scview_leiden_r2.0"].astype(str).map(merged),
                           categories=step["params"]["categories"])
    assert list(again) == list(a.obs[col]) and list(again.categories) == list(a.obs[col].cat.categories)


def test_save_of_a_sweep_resolution_keeps_it_first():
    a = _processed()
    p = mc.plan(a, resolution=0.6)
    res = mc.save(a, p, [[p.names[0], p.names[1]]])
    assert "scview_leiden_r0.6" in a.obs.columns and res["column"] == "scview_leiden_r0.6_merged"
    steps = [s["step"] for s in provenance.read_provenance(a)["history"]][-2:]
    assert steps == ["keep_resolution", "merge_clusters"]


def test_grounding_names_the_candidates():
    class Adaptor:  # weak-referenceable stand-in
        pass

    ad_ = Adaptor()
    pair = {"a": "1", "b": "2", "adjacency": 0.3, "evidence": "held_out", "n_genes": 3, "up": ["X"],
            "down": [], "candidate": True}
    apart = {**pair, "a": "1", "b": "5", "n_genes": 200, "candidate": False, "up": ["CD14", "LYZ"]}
    r = {"reference": "cl", "reference_exists": True, "method": "leiden", "resolution": 1.0,
         "train_resolution": 1.2, "train": {"n_clusters": 6, "ari": 0.8}, "pairs": [pair, apart],
         "thresholds": {"merge_max_genes": 10, "padj": 0.05}}
    mc.remember(ad_, r)
    obs = pd.DataFrame({"cl": ["1", "2"]})
    lines, detail = mc.grounding_lines(ad_, obs)
    assert "1 merge candidates" in detail
    assert any("merge candidate: 1 + 2: 3 genes" in line for line in lines)
    assert any("kept apart: 1 vs 5" in line for line in lines)
    assert mc.grounding_lines(ad_, pd.DataFrame({"other": [1]})) == ([], None)  # column gone


# --- endpoints ------------------------------------------------------------------------------


async def test_endpoints_check_stream_and_save(client, dataset_manager):
    did = _register(dataset_manager, _processed(), "mc")
    info = (await client.get(f"{API}/datasets/{did}/clustering/merge-check")).json()
    assert info["available"] is True and info["reference"] == "scview_leiden_r2.0"
    assert info["estimate_seconds"] > 0 and info["merge_max_genes"] == mc.MERGE_MAX_GENES
    r = await client.post(f"{API}/datasets/{did}/clustering/merge-check", json={})
    assert r.status_code == 200, r.text
    events = _events(r.text)
    kinds = [e for e, _ in events]
    assert kinds[0] == "start" and kinds[-2:] == ["result", "complete"] and "progress" in kinds
    result = events[-2][1]
    pair = next(x for x in result["pairs"] if x["candidate"])
    # The run reaches the co-pilot.
    from scview.core.assistant import build_grounding_context

    context, sources = build_grounding_context(await dataset_manager.get_or_load_dataset(did), {"groups"})
    assert "Merge check (last run; count splitting" in context
    assert f"merge candidate: {pair['a']} + {pair['b']}" in context
    assert any(s.ref == "result:merge_check" for s in sources)
    r = await client.post(f"{API}/datasets/{did}/clustering/merge-check", json={"max_genes": 1})
    events = _events(r.text)
    assert events[0][1]["cached"] is True and [e for e, _ in events] == ["start", "result", "complete"]
    assert events[1][1]["thresholds"]["merge_max_genes"] == 1

    r = await client.post(f"{API}/datasets/{did}/clustering/merge-check/save",
                          json={"pairs": [[pair["a"], pair["b"]]]})
    assert r.status_code == 200, r.text
    assert r.json()["column"] == "scview_leiden_r2.0_merged"
    ds = (await client.get(f"{API}/datasets/{did}")).json()
    assert "scview_leiden_r2.0_merged" in [c["name"] for c in ds["obs_columns"]]
    r = await client.post(f"{API}/datasets/{did}/clustering/merge-check/save", json={"pairs": [["0", "nope"]]})
    assert r.status_code == 400 and "nope" in r.json()["detail"]


async def test_endpoint_explains_a_dataset_without_clusters(client, dataset_manager):
    did = _register(dataset_manager, _types(), "mc2")
    info = (await client.get(f"{API}/datasets/{did}/clustering/merge-check")).json()
    assert info["available"] is False and "no clustering" in info["reason"]
    r = await client.post(f"{API}/datasets/{did}/clustering/merge-check", json={})
    assert r.status_code == 400
