"""Milo neighbourhood differential abundance (core/milo.py, api/v1/milo.py).

The reference numbers in ``fixtures/milo/pertpy_milo.tsv`` come from pertpy
1.0.3's ``Milo`` (``make_nhoods(prop=0.1, seed=0)``, ``count_nhoods``,
``da_nhoods(design="~donor+cond", solver="pydeseq2")``) run on ``synthetic()``
below; ``fixtures/milo/pertpy_milo.py`` regenerates them.
"""

from __future__ import annotations

import json
import random
import threading
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from scipy import stats

from scview.core import milo, provenance
from scview.core.milo import MiloError

MAIN = threading.main_thread()
FIXTURE = Path(__file__).parent / "fixtures" / "milo" / "pertpy_milo.tsv"


def synthetic(shift: float = 3.0, separate: bool = False, seed: int = 0,
              k: int = 50) -> ad.AnnData:
    """Three cell types in a 10-D PCA, 4 donors each in conditions A and B
    (paired). Type t0 is ``shift`` times as abundant in B. ``separate`` moves
    every B cell away from its A counterpart (a change of state). The graph is
    an exact k-NN graph (symmetric binary connectivities + distances)."""
    from sklearn.neighbors import NearestNeighbors

    rng = np.random.default_rng(seed)
    centers = np.zeros((3, 10))
    centers[1, 0], centers[2, 1] = 8.0, 8.0
    rows, X = [], []
    for donor in ("d1", "d2", "d3", "d4"):
        for cond in ("A", "B"):
            for t, n in enumerate((40, 60, 50)):
                n = int(round(n * (shift if (cond == "B" and t == 0) else 1.0)))
                pts = centers[t] + rng.normal(size=(n, 10))
                if separate and cond == "B":
                    pts[:, 2] += 10.0
                X.append(pts)
                rows += [(donor, cond, f"t{t}")] * n
    X = np.vstack(X).astype(np.float32)
    obs = pd.DataFrame(rows, columns=["donor", "cond", "type"])
    obs.index = [f"c{i}" for i in range(len(obs))]
    for c in obs.columns:
        obs[c] = obs[c].astype("category")
    a = ad.AnnData(np.zeros((len(obs), 3), dtype=np.float32), obs=obs)
    a.var_names = ["G0", "G1", "G2"]
    a.obsm["X_pca"] = X
    a.obsm["X_umap"] = X[:, :2].copy()
    nn = NearestNeighbors(n_neighbors=k).fit(X)
    dist, idx = nn.kneighbors(X)
    n = len(obs)
    r = np.repeat(np.arange(n), k - 1)
    D = sp.csr_matrix((dist[:, 1:].ravel(), (r, idx[:, 1:].ravel())), shape=(n, n))
    C = sp.csr_matrix((np.ones(r.size), (r, idx[:, 1:].ravel())), shape=(n, n))
    C = ((C + C.T) > 0).astype(np.float32)
    a.obsp["connectivities"] = sp.csr_matrix(C)
    a.obsp["distances"] = D
    a.uns["neighbors"] = {"connectivities_key": "connectivities", "distances_key": "distances",
                          "params": {"use_rep": "X_pca", "n_neighbors": k, "method": "umap"}}
    return a


def _run(a, **kw) -> milo.MiloResult:
    p = milo.plan(a, "cond", "donor", conditions=["A", "B"], **kw)
    return milo.run(a, p, label_key="type")


# --- neighbourhoods and spatial FDR -------------------------------------------


def _pertpy_make_nhoods(conn, X, dist, prop, seed):
    """pertpy 1.0.3 ``Milo.make_nhoods``, its loop as written."""
    from sklearn.metrics.pairwise import euclidean_distances

    knn = conn.copy().tocsr()
    knn[knn != 0] = 1
    n = knn.shape[0]
    random.seed(seed)
    verts = sorted(random.sample(range(n), k=int(np.round(n * prop))))
    sub = knn[verts, :]
    rws, cls = sub.nonzero()
    refined = np.empty(len(verts))
    for i in range(len(verts)):
        nn = cls[rws == i]
        pos = np.median(X[nn, :], 0).reshape(-1, 1)
        refined[i] = nn[euclidean_distances(X[nn, :], pos.T).argmin()]
    refined = np.unique(refined.astype(int))
    kth = dist[refined, :].max(1).toarray().ravel()
    return refined, knn[:, refined], kth


def test_neighbourhoods_are_pertpys():
    a = synthetic()
    conn, dist = milo.graph(a)
    nh = milo.make_nhoods(conn, a.obsm["X_pca"], dist, prop=0.1, seed=0)
    ref, members, kth = _pertpy_make_nhoods(conn, a.obsm["X_pca"], dist, 0.1, 0)
    assert np.array_equal(nh.index, ref)
    assert (nh.membership != sp.csc_matrix(members)).nnz == 0
    assert np.allclose(nh.kth_distance, kth)
    assert len(nh.index) < int(round(a.n_obs * 0.1))  # refinement collapsed some


def test_spatial_fdr_is_weighted_bh():
    rng = np.random.default_rng(1)
    p = rng.uniform(size=50) ** 3
    # Equal weights: plain Benjamini-Hochberg.
    assert np.allclose(milo.spatial_fdr(p, np.ones(50)), stats.false_discovery_control(p))
    # Weighted, by hand: q_(i) = sum(w) p_(i) / cumsum(w)_(i), running min from the top.
    w = rng.uniform(0.5, 2, size=50)
    o = np.argsort(p)
    q = w.sum() * p[o] / np.cumsum(w[o])
    q = np.minimum(np.minimum.accumulate(q[::-1])[::-1], 1)
    expect = np.empty(50)
    expect[o] = q
    assert np.allclose(milo.spatial_fdr(p, 1 / w), expect)
    # NaN p-values are left out and stay NaN; a zero distance has weight 0.
    p2 = p.copy()
    p2[[3, 7]] = np.nan
    out = milo.spatial_fdr(p2, np.r_[0.0, np.ones(49)])
    assert np.isnan(out[[3, 7]]).all() and np.isfinite(np.delete(out, [3, 7])).all()


# --- the test -------------------------------------------------------------------


def test_matches_pertpy_pydeseq2():
    ref = pd.read_csv(FIXTURE, sep="\t")
    r = _run(synthetic())
    assert r.index_cells == ref["index_cell"].tolist()
    assert np.allclose(r.log_fc, ref["logFC"], atol=1e-4)
    assert np.allclose(r.pval, ref["PValue"], rtol=1e-3, atol=1e-8)
    assert np.allclose(r.spatial_fdr, ref["SpatialFDR"], rtol=1e-3, atol=1e-8)
    assert r.labels == ref["label"].where(ref["label_fraction"] >= 0.7, "Mixed").tolist()
    assert r.formula == "~ donor + cond" and r.design == "paired"


def test_a_planted_shift_is_found_and_nothing_else():
    r = _run(synthetic(shift=3.0))
    sig = r.significant
    labels = np.array(r.labels)
    t0 = labels == "t0"
    assert sig[t0].mean() > 0.8 and (r.log_fc[t0 & sig] > 0).all()
    # The other types lose share only through t0's growth: small, negative.
    assert np.median(r.log_fc[labels == "t1"]) < 0
    d = r.to_dict()
    assert d["n_up"] >= int((t0 & sig).sum()) and d["n_significant"] == int(sig.sum())
    by = {x["label"]: x for x in d["labels"]}
    assert by["t0"]["n_up"] == int((t0 & sig).sum()) and by["t0"]["median_logfc"] > 0.5
    # No planted shift: nothing passes.
    null = _run(synthetic(shift=1.0))
    assert null.significant.sum() == 0


def test_unpaired_and_a_covariate():
    a = synthetic()
    a.obs["site"] = a.obs["donor"].map({"d1": "x", "d2": "x", "d3": "y", "d4": "y"})
    p = milo.plan(a, "cond", "donor", conditions=["A", "B"], paired=False, covariates=["site"])
    assert p.formula == "~ site + cond" and p.design == "unpaired"
    r = milo.run(a, p, label_key="type")
    assert r.significant.sum() > 0
    # A per-donor covariate is absorbed by pairing.
    with pytest.raises(MiloError, match="absorbed by pairing"):
        milo.plan(a, "cond", "donor", conditions=["A", "B"], covariates=["site"])
    # A per-cell column is not a covariate.
    with pytest.raises(MiloError, match="varies within a sample"):
        milo.plan(a, "cond", "donor", conditions=["A", "B"], covariates=["type"])


def test_labels_and_mixed():
    a = synthetic()
    conn, dist = milo.graph(a)
    nh = milo.make_nhoods(conn, a.obsm["X_pca"], dist)
    labels, frac = milo.annotate(a.obs["type"], nh.membership)
    assert set(labels) <= {"t0", "t1", "t2", "Mixed"} and (frac > 0.5).all()
    # Random labels: every neighbourhood is mixed.
    noise = pd.Series(np.random.default_rng(0).choice(list("abcd"), a.n_obs), index=a.obs_names)
    labels, frac = milo.annotate(noise, nh.membership)
    assert set(labels) == {"Mixed"} and (frac < 0.7).all()


def test_bad_requests():
    a = synthetic()
    with pytest.raises(MiloError, match="sample or donor column"):
        milo.plan(a, "cond", None)
    with pytest.raises(MiloError, match="must be different"):
        milo.plan(a, "cond", "cond")
    with pytest.raises(MiloError, match="not a value"):
        milo.plan(a, "cond", "donor", conditions=["A", "Z"])
    with pytest.raises(MiloError, match="compares two values"):
        milo.plan(a, "cond", "donor", conditions=["A"])
    with pytest.raises(MiloError, match="between"):
        milo.plan(a, "cond", "donor", prop=0.9)
    one = a[a.obs["donor"].isin(["d1"]) | (a.obs["cond"] == "A")].copy()
    with pytest.raises(MiloError, match="at least 2 per condition"):
        milo.plan(one, "cond", "donor")
    no_graph = a.copy()
    del no_graph.obsp["connectivities"]
    with pytest.raises(MiloError, match="no neighbour graph"):
        milo.plan(no_graph, "cond", "donor")


def test_design_warnings():
    a = synthetic()
    # The graph was corrected for the condition itself.
    p = milo.plan(a, "cond", "donor", batch_key="cond")
    assert any("batch-corrected for 'cond'" in w for w in p.warnings)
    # Conditions that separate on the embedding: a change of state everywhere.
    r = _run(synthetic(separate=True))
    assert any("almost only one condition" in w for w in r.warnings)
    assert not any("Milo suggests" in n for n in r.notes)
    # Small neighbourhoods (a 10-NN graph) for 8 samples.
    r = _run(synthetic(k=10))
    assert any("Milo suggests at least 5 per sample (40)" in n for n in r.notes)


def test_cells_of_neighbourhoods():
    a = synthetic()
    r = _run(a)
    cells = milo.nhood_cells(r, [0, 1])
    expect = np.union1d(r.membership[:, 0].indices, r.membership[:, 1].indices)
    assert np.array_equal(cells, expect)
    assert milo.nhood_cells(r, [10_000]).size == 0


def test_result_payload():
    a = synthetic()
    r = _run(a)
    d = r.to_dict(milo.layouts(a, r.index))
    n = d["n_nhoods"]
    assert set(d["layouts"]) == {"X_pca", "X_umap"} and len(d["layouts"]["X_umap"]) == n
    for key in ("index_cell", "size", "label", "log_fc", "pval", "spatial_fdr"):
        assert len(d["nhoods"][key]) == n
    assert set(d["nhoods"]["counts"]) == {"A", "B"}
    assert all(0 <= i < n and 0 <= j < n and w > 0 for i, j, w in d["edges"])
    json.dumps(d)  # JSON-safe (no NaN)


# --- endpoints and co-pilot -------------------------------------------------------


def _dataset(root: Path, a: ad.AnnData, did: str = "ds") -> None:
    d = root / "uploads" / did
    d.mkdir(parents=True)
    (d / "metadata.json").write_text(json.dumps({"id": did, "name": did, "status": "ready"}))
    a.write_h5ad(d / "data.h5ad")


def _events(text: str) -> list[tuple[str, dict]]:
    out = []
    for block in text.split("\n\n"):
        lines = [x for x in block.splitlines() if x and not x.startswith(":")]
        if len(lines) == 2 and lines[0].startswith("event: "):
            out.append((lines[0][7:], json.loads(lines[1][6:])))
    return out


async def test_plan_endpoint_detects_the_donor_column(client, dataset_manager):
    a = synthetic()
    a.obs = a.obs.rename(columns={"donor": "Donor", "cond": "stim"})
    a.obs["orig_ident"] = a.obs["stim"]  # sample-like by name, but one per condition
    _dataset(Path(dataset_manager.data_dir), a)
    r = await client.post("/api/v1/datasets/ds/milo/plan", json={"condition_key": "stim"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["sample_key"] == "Donor" and body["sample_key_detected"] is True
    assert body["design"] == "paired" and body["reference"] == "A"
    assert body["n_samples"] == {"A": 4, "B": 4} and body["estimate_seconds"] > 0
    r = await client.post("/api/v1/datasets/ds/milo/plan",
                          json={"condition_key": "stim", "sample_key": "orig_ident"})
    assert r.status_code == 400 and "at least 2 per condition" in r.json()["detail"]
    r = await client.post("/api/v1/datasets/ds/milo/plan", json={"condition_key": "nope"})
    assert r.status_code == 400 and "not found" in r.json()["detail"]


async def test_run_streams_off_the_loop_and_reaches_the_copilot(
        client, dataset_manager, monkeypatch):
    from scview.core.assistant import build_grounding_context

    _dataset(Path(dataset_manager.data_dir), synthetic())
    threads: list[threading.Thread] = []
    real = milo._deseq2

    def spy(*args, **kwargs):
        threads.append(threading.current_thread())
        return real(*args, **kwargs)

    monkeypatch.setattr(milo, "_deseq2", spy)
    r = await client.post("/api/v1/datasets/ds/milo/cells", json={"nhoods": [0]})
    assert r.status_code == 404
    r = await client.post("/api/v1/datasets/ds/milo/run",
                          json={"condition_key": "cond", "sample_key": "donor",
                                "label_key": "type"})
    assert r.status_code == 200
    events = _events(r.text)
    kinds = [e for e, _ in events]
    assert kinds[0] == "start" and kinds[-1] == "complete" and "result" in kinds
    assert [d["step"] for e, d in events if e == "progress"] == [
        "nhoods", "counts", "test", "annotate"]
    result = next(d for e, d in events if e == "result")
    assert result["n_significant"] > 0 and result["label_key"] == "type"
    assert "X_umap" in result["layouts"]
    assert len(threads) == 1 and threads[0] is not MAIN

    r = await client.post("/api/v1/datasets/ds/milo/cells", json={"nhoods": [0, 1]})
    assert r.status_code == 200 and r.json()["n_cells"] > 0

    adaptor = await dataset_manager.get_or_load_dataset("ds")
    context, sources = build_grounding_context(adaptor, {"groups"})
    assert "Milo neighbourhood differential abundance (last run" in context
    assert "B vs A, samples from 'donor' (A 4, B 4; paired)" in context
    assert "- t0: " in context
    assert any(s.ref == "result:milo" for s in sources)


async def test_run_rejects_a_bad_request_before_streaming(client, dataset_manager):
    _dataset(Path(dataset_manager.data_dir), synthetic())
    r = await client.post("/api/v1/datasets/ds/milo/run",
                          json={"condition_key": "cond", "sample_key": "cond"})
    assert r.status_code == 400
    r = await client.post("/api/v1/datasets/missing/milo/run", json={"condition_key": "cond"})
    assert r.status_code == 404


async def test_run_warns_when_the_graph_was_corrected_for_the_condition(
        client, dataset_manager):
    a = synthetic()
    provenance.record_step(a, step="batch_correction", tool="harmony",
                           params={"batch_key": "cond", "batch_method": "harmony"})
    _dataset(Path(dataset_manager.data_dir), a)
    r = await client.post("/api/v1/datasets/ds/milo/plan", json={"condition_key": "cond"})
    assert r.status_code == 200
    assert any("batch-corrected for 'cond'" in w for w in r.json()["warnings"])


def test_grounding_drops_a_result_whose_columns_went():
    a = synthetic()

    class Adaptor:
        pass

    ad_ = Adaptor()
    ad_.adata = a
    milo.remember(ad_, _run(a))
    lines, detail = milo.grounding_lines(ad_)
    assert detail and "neighbourhoods" in detail
    del a.obs["type"]
    assert milo.grounding_lines(ad_) == ([], None)
