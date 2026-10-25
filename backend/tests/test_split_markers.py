"""Count-split markers: calibrated marker p-values (core/split_markers.py + /markers/count-split)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from scipy import stats

from scview.core import de_results
from scview.core import merge_check as mc
from scview.core import split_markers as sm
from tests.test_hvg_sensitivity import _events, _register
from tests.test_merge_check import _processed, _types

API = "/api/v1"


@pytest.fixture(autouse=True)
def _empty_cache():
    sm._stages.clear()
    yield
    sm._stages.clear()


def _nb(rng, mu, theta):
    return rng.negative_binomial(theta, theta / (theta + mu))


def _two_clusters(counts) -> tuple[sp.csr_matrix, np.ndarray]:
    """Log-normalised counts and a forced 2-cluster k-means of their PCs."""
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    log = mc._normalise(counts, 1e4)
    pcs = PCA(10, random_state=0).fit_transform(log.toarray())
    return log, KMeans(2, n_init=10, random_state=0).fit_predict(pcs).astype(np.int64)


def test_split_pvalues_are_uniform_on_one_population_and_ordinary_ones_are_not():
    """One homogeneous NB population forced into two clusters: the ordinary
    (double-dipped) p-values pile up near 0, the count-split ones are uniform."""
    rng = np.random.default_rng(1)
    n, g = 800, 300
    mu = np.tile(rng.gamma(4, 1.5, g), (n, 1)) * rng.lognormal(0, 0.1, n)[:, None]
    X = sp.csr_matrix(_nb(rng, mu, 2.0).astype("float32"))
    var = pd.Index([f"G{i}" for i in range(g)])

    log, codes = _two_clusters(X)
    ordinary = sm.groups_vs_rest(log, codes, var)
    p_ord = ordinary.pvals[ordinary.row(0)]

    srng = np.random.default_rng(101)
    theta, _ = mc.estimate_overdispersion(X, srng)
    train, test = mc.split_counts(X, theta, srng)
    _, train_codes = _two_clusters(train)
    split = sm.groups_vs_rest(mc._normalise(test, 1e4), train_codes, var)
    p_split = split.pvals[split.row(0)]

    assert stats.kstest(p_ord, "uniform").pvalue < 1e-10
    assert np.mean(p_ord < 0.05) > 0.2
    assert stats.kstest(p_split, "uniform").pvalue > 0.05
    assert np.mean(p_split < 0.05) < 0.1


def test_groups_vs_rest_leaves_out_unlabelled_and_tiny_groups():
    rng = np.random.default_rng(0)
    X = sp.csr_matrix(rng.poisson(3.0, (60, 20)).astype("float32"))
    codes = np.array([0] * 30 + [1] * 28 + [2] * 2)
    codes[:3] = -1
    t = sm.groups_vs_rest(mc._normalise(X, 1e4), codes, pd.Index([f"G{i}" for i in range(20)]))
    assert t.groups == [0, 1] and t.row(2) is None
    assert t.pvals.shape == (2, 20) and np.all(np.isfinite(t.pvals))
    assert np.all((t.pct_in >= 0) & (t.pct_in <= 1))


def test_summarise_counts_kept_and_lost_and_skips_unmatched_groups():
    g = 4
    var = pd.Index(["A", "B", "RPL3", "D"])

    def tests(groups, lfc, padj):
        k = len(groups)
        return sm.GroupTests(groups, np.ones((k, g)), np.asarray(lfc, float), np.asarray(padj, float),
                             np.asarray(padj, float), np.full((k, g), 0.5), np.full((k, g), 0.1))

    # Ordinary: group x has A, B, D (RPL3 left out as ribosomal); y has A.
    ordinary = tests([0, 1], [[1, 1, 1, 1], [1, 0, 0, 0]], [[1e-9, 1e-9, 1e-9, 1e-9], [1e-9, 1, 1, 1]])
    # Split: training cluster 0 (x's match) keeps A only.
    split = tests([0, 1], [[1, 1, 1, -1], [1, 0, 0, 0]], [[1e-5, 0.3, 1e-5, 1e-5], [1, 1, 1, 1]])
    rec = mc.hs.Recorded(reference="cl", method="leiden", resolution=1.0, flavor="seurat", n_top_genes=200,
                         hvg_batch_key=None, target_sum=1e4, n_comps=10, n_neighbors=15, batch_key=None)
    p = mc.Plan("cl", True, np.array([0] * 5 + [1] * 5), ["x", "y"], rec)
    comparison = {"ari": 0.5, "ami": 0.5, "clusters": [
        {"cluster": "x", "n_cells": 5, "jaccard": 0.9, "match": "t0"},
        {"cluster": "y", "n_cells": 5, "jaccard": 0.05, "match": "t1"}]}
    r, matched = sm.summarise(p, ordinary, split, comparison, ["t0", "t1"], var)
    x, y = r["groups"]
    assert matched == [0, None]
    assert (x["n_ordinary"], x["n_kept"], x["n_lost"]) == (3, 1, 2)
    assert x["kept"] == ["A"] and set(x["lost"]) == {"B", "D"}  # D is down on the held-out half
    assert not y["tested"] and "no training cluster matches it" in y["reason"]
    assert r["n_tested_groups"] == 1 and r["n_ordinary"] == 3 and r["n_kept"] == 1


def test_run_keeps_the_true_markers_and_loses_the_double_dipped_ones():
    a = _processed()  # three types over-clustered at resolution 2.0
    p = sm.plan(a)
    obs_before = list(a.obs.columns)
    steps = list(sm.iter_split_markers(a, p))
    assert [s.get("step") for s in steps[1:-1]][:4] == ["overdispersion", "ordinary", "split", "train"]
    r = steps[-1]
    assert list(a.obs.columns) == obs_before
    # Over-split pieces come back as other pieces on the training half (Jaccard 0.11-0.28).
    assert r["n_groups"] == len(p.names) and r["n_tested_groups"] >= r["n_groups"] - 2
    assert abs(r["train"]["n_clusters"] - len(p.names)) <= 2
    type_of = pd.crosstab(a.obs[p.reference], a.obs["type"]).idxmax(axis=1).astype(str).to_dict()
    for g in r["groups"]:
        if not g["tested"]:
            assert g["reason"] and g["n_ordinary"] == 0
            continue
        assert g["n_kept"] + g["n_lost"] == g["n_ordinary"]
        assert len(g["genes"]) == min(sm.GENES_PER_GROUP, a.n_vars)
        t = int(type_of[g["group"]][1])
        true = {f"G{i}" for i in range(30 * t, 30 * t + 30)}
        # The type's own 30 markers survive the held-out test; what is lost is noise
        # that set the over-split piece apart only on the counts that made it.
        assert g["n_kept"] >= 25 and set(g["kept"]) <= true
        assert len(true & set(g["lost"])) <= 2  # a weak true marker can miss at half depth
    assert sum(g["n_lost"] for g in r["groups"]) > 0
    # Again: the kept run is returned at once.
    again = list(sm.iter_split_markers(a, p))
    assert again[0]["cached"] is True and len(again) == 2 and again[-1] is r


def test_keep_group_stores_a_de_result():
    a = _processed()
    p = sm.plan(a)
    sm.run(a, p)
    stage = sm.cached(a, p)
    rid = sm.keep_group("ds", stage, p.names[0])
    stored = de_results.recall("ds", rid)
    assert stored is not None and "count split" in stored.label and stored.method == "wilcoxon"
    assert len(stored.genes) == a.n_vars and np.all(np.isfinite(stored.stats))
    rows = stored.rows()
    assert rows[0]["score"] is not None  # GSEA ranks it by its statistic
    with pytest.raises(sm.SplitMarkersError, match="No group"):
        sm.keep_group("ds", stage, "nope")


def test_plan_refuses_a_dataset_without_counts():
    a = _processed()
    del a.layers["counts"]
    a.raw = None
    X = a.X.toarray() if sp.issparse(a.X) else np.asarray(a.X)
    a.X = sp.csr_matrix(np.abs(X) + 0.5)
    with pytest.raises(sm.SplitMarkersError, match="raw counts"):
        sm.plan(a)


def test_grounding_names_groups_that_keep_few_markers():
    class Adaptor:  # weak-referenceable stand-in
        pass

    ad_ = Adaptor()
    g = {"tested": True, "lost": [], "kept": []}
    r = {"reference": "cl", "reference_exists": True, "n_keep_enough": 1, "n_kept": 40, "n_ordinary": 90,
         "train": {"ari": 0.8}, "thresholds": {"keeps_enough": 10},
         "groups": [{**g, "group": "1", "n_kept": 38, "n_ordinary": 50},
                    {**g, "group": "5", "n_kept": 2, "n_ordinary": 40, "lost": ["RPL3", "FOS"]}]}
    sm.remember(ad_, r)
    lines, detail = sm.grounding_lines(ad_, pd.DataFrame({"cl": ["1"]}))
    assert "1/2 groups keep >= 10 markers" in detail
    assert any("5: keeps 2 of 40 markers; lost e.g. RPL3, FOS" in line for line in lines)
    assert sm.grounding_lines(ad_, pd.DataFrame({"other": [1]})) == ([], None)


# --- endpoints ------------------------------------------------------------------------------


async def test_endpoints_info_stream_keep_and_copilot(client, dataset_manager):
    did = _register(dataset_manager, _processed(), "cs")
    info = (await client.get(f"{API}/datasets/{did}/markers/count-split")).json()
    assert info["available"] is True and info["reference"] == "scview_leiden_r2.0"
    assert info["estimate_seconds"] > 0 and info["cached"] is False
    r = await client.post(f"{API}/datasets/{did}/markers/count-split/keep", json={"group": "0"})
    assert r.status_code == 404  # nothing run yet
    r = await client.post(f"{API}/datasets/{did}/markers/count-split", json={})
    assert r.status_code == 200, r.text
    events = _events(r.text)
    kinds = [e for e, _ in events]
    assert kinds[0] == "start" and kinds[-2:] == ["result", "complete"] and "progress" in kinds
    result = events[-2][1]
    group = result["groups"][0]["group"]
    from scview.core.assistant import build_grounding_context

    context, sources = build_grounding_context(await dataset_manager.get_or_load_dataset(did), {"markers"})
    assert "Count-split marker check" in context
    assert any(s.ref == "result:count_split" for s in sources)
    r = await client.post(f"{API}/datasets/{did}/markers/count-split/keep", json={"group": group})
    assert r.status_code == 200, r.text
    listed = (await client.get(f"{API}/datasets/{did}/de/results")).json()["results"]
    assert r.json()["result_id"] in [x["result_id"] for x in listed]
    info = (await client.get(f"{API}/datasets/{did}/markers/count-split")).json()
    assert info["cached"] is True


async def test_endpoints_refuse_with_a_reason(client, dataset_manager):
    did = _register(dataset_manager, _types(), "cs2")
    info = (await client.get(f"{API}/datasets/{did}/markers/count-split")).json()
    assert info["available"] is False and "no clustering" in info["reason"]
    r = await client.post(f"{API}/datasets/{did}/markers/count-split", json={})
    assert r.status_code == 422 and "no clustering" in r.json()["detail"]
    r = await client.post(f"{API}/datasets/{did}/markers/count-split", json={"column": "type"})
    assert r.status_code == 200  # any grouping column, re-clustered to its number of groups
