"""Neighbourhood DE, miloDE-style (core/milode.py, api/v1/milode.py).

``fixtures/milode/limma_voom.tsv`` holds edgeR 4.10.7 / limma 3.68.5 results
(filterByExpr as miloDE 0.1.0 calls it, TMM, voom, lmFit, eBayes) for the pseudobulk table in
``fixtures/milode/counts.tsv``; ``fixtures/milode/limma_voom.R`` regenerates it.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from scview.core import milode, results_export, results_store
from scview.core.milode import MiloDEError

MAIN = threading.main_thread()
FIXTURES = Path(__file__).parent / "fixtures" / "milode"
N_GENES = 120
PLANTED = [f"G{i}" for i in range(6)]


def synthetic(effect: float = 4.0, seed: int = 0, donors: int = 4,
              separate: bool = False) -> ad.AnnData:
    """Three cell types in a 10-D PCA, ``donors`` donors each in conditions A
    and B (paired). Raw counts in X: negative binomial around per-type gene
    means, each cell with its own depth and each donor its own small offset.
    Genes G0-G5 are ``effect`` times higher in B, but only in type t0 cells
    with PC 3 > 0: half of one type, the region a per-cluster test dilutes.
    ``separate`` moves 90% of the B cells of type t2 away from their A
    counterparts (a change of state the embedding did not integrate)."""
    from sklearn.neighbors import NearestNeighbors

    rng = np.random.default_rng(seed)
    centers = np.zeros((3, 10))
    centers[1, 0], centers[2, 1] = 10.0, 10.0
    base = rng.gamma(2.0, 2.0, size=(3, N_GENES))
    donor_fx = rng.normal(0, 0.15, size=(donors, N_GENES))
    rows, X, counts = [], [], []
    for d in range(donors):
        for cond in ("A", "B"):
            for t, n in enumerate((120, 100, 100)):
                pts = centers[t] + rng.normal(size=(n, 10))
                if separate and cond == "B" and t == 2:
                    pts[: int(n * 0.9), 4] += 12.0
                mu = base[t] * np.exp(donor_fx[d]) * rng.uniform(0.6, 1.4, size=(n, 1))
                if cond == "B" and t == 0:
                    region = pts[:, 3] > 0
                    mu[np.ix_(region, np.arange(len(PLANTED)))] *= effect
                r = 5.0
                counts.append(rng.negative_binomial(r, r / (r + mu)))
                X.append(pts)
                rows += [(f"d{d}", cond, f"t{t}")] * n
    obs = pd.DataFrame(rows, columns=["donor", "cond", "type"])
    obs.index = [f"c{i}" for i in range(len(obs))]
    for c in obs.columns:
        obs[c] = obs[c].astype("category")
    a = ad.AnnData(sp.csr_matrix(np.vstack(counts).astype(np.float32)), obs=obs)
    a.var_names = [f"G{i}" for i in range(N_GENES)]
    Xp = np.vstack(X).astype(np.float32)
    a.obsm["X_pca"] = Xp
    a.obsm["X_umap"] = Xp[:, :2].copy()
    k = 15
    nn = NearestNeighbors(n_neighbors=k).fit(Xp)
    dist, idx = nn.kneighbors(Xp)
    n = len(obs)
    r_ = np.repeat(np.arange(n), k - 1)
    C = sp.csr_matrix((np.ones(r_.size), (r_, idx[:, 1:].ravel())), shape=(n, n))
    a.obsp["connectivities"] = sp.csr_matrix(((C + C.T) > 0).astype(np.float32))
    a.obsp["distances"] = sp.csr_matrix((dist[:, 1:].ravel(), (r_, idx[:, 1:].ravel())),
                                        shape=(n, n))
    a.uns["neighbors"] = {"connectivities_key": "connectivities", "distances_key": "distances",
                          "params": {"use_rep": "X_pca", "n_neighbors": k}}
    return a


def _run(a, **kw) -> milode.MiloDEResult:
    kw.setdefault("genes", "expressed")
    p = milode.plan(a, "cond", "donor", conditions=["A", "B"], **kw)
    return milode.run(a, p, label_key="type")


# --- neighbourhoods ----------------------------------------------------------------


def test_second_order_neighbourhoods_and_redundancy_filter():
    a = synthetic()
    X = a.obsm["X_pca"]
    A = milode.knn_graph(X, 10)
    assert (A != A.T).nnz == 0 and A.diagonal().sum() == 0
    assert np.asarray(A.sum(axis=1)).min() >= 10
    index = np.array([0, 500, 1500])
    M = milode.second_order(A, index)
    for j, v in enumerate(index):
        one = set(A[v].indices) | {v}
        two = set(one)
        for u in one:
            two |= set(A[u].indices)
        assert set(M[:, j].indices) == two
    nh = milode.make_nhoods(X, k=10, prop=0.1)
    full = milode.make_nhoods(X, k=10, prop=0.1, filtering=False)
    assert len(nh.index) < full.n_sampled == nh.n_sampled
    covered = np.asarray(full.membership.sum(axis=1)).ravel() > 0
    kept = np.asarray(nh.membership.sum(axis=1)).ravel() > 0
    assert np.array_equal(covered, kept)  # dropping redundant ones keeps the cover


def test_spatial_fdr_weights_by_overlap():
    """miloDE's spatial_pval_adjustment: per gene, weighted BH over the
    neighbourhoods where it was tested, weight 1 / rowSums(crossprod(nhoods))."""
    M = sp.csc_matrix(np.array([[1, 1, 0, 0], [1, 1, 0, 0], [1, 0, 0, 0], [0, 1, 1, 0],
                                [0, 0, 1, 1], [0, 0, 0, 1], [0, 0, 0, 1]], dtype=np.int32))
    p = np.array([[0.01, 0.2], [0.02, np.nan], [0.03, 0.04], [0.5, 0.001]])
    got = milode.spatial_fdr(p, M)

    def wbh(pv, w):
        o = np.argsort(pv)
        adj = np.empty(len(pv))
        adj[o] = np.minimum.accumulate((w.sum() * pv[o] / np.cumsum(w[o]))[::-1])[::-1]
        return np.minimum(adj, 1)

    overlap = (M.T @ M).toarray()
    assert np.allclose(got[:, 0], wbh(p[:, 0], 1 / overlap.sum(axis=1)), atol=1e-6)
    rows = [0, 2, 3]
    w = 1 / overlap[np.ix_(rows, rows)].sum(axis=1)  # only the neighbourhoods that tested it
    assert np.allclose(got[rows, 1], wbh(p[rows, 1], w), atol=1e-6)
    assert np.isnan(got[1, 1])


# --- the per-neighbourhood engine vs edgeR + limma ----------------------------------


def test_engine_matches_edger_limma_voom():
    counts = pd.read_csv(FIXTURES / "counts.tsv", sep="\t", index_col=0)  # genes x samples
    design = pd.read_csv(FIXTURES / "design.tsv", sep="\t", index_col=0).to_numpy(float)
    lib = pd.read_csv(FIXTURES / "lib.tsv", sep="\t", index_col=0)["lib"].to_numpy(float)
    want = pd.read_csv(FIXTURES / "limma_voom.tsv", sep="\t", index_col=0)
    res = milode.test_nhood(counts.to_numpy().T, lib, design)
    got = pd.DataFrame({"logFC": res.log_fc, "p": res.pval}, index=counts.index)
    tested = got.dropna().index
    assert set(tested) == set(want.index)  # filterByExpr keeps the same genes
    w = want.loc[tested]
    assert np.allclose(got.loc[tested, "logFC"], w["logFC"], atol=1e-7)
    assert np.allclose(np.log10(got.loc[tested, "p"]), np.log10(w["P.Value"]), atol=1e-7)


def test_filter_and_tmm_basics():
    rng = np.random.default_rng(1)
    counts = rng.poisson(20, size=(6, 50)).astype(float)
    counts[:, :5] = 0
    lib = counts.sum(axis=1)
    keep = milode.filter_by_expr(counts, lib)
    assert not keep[:5].any() and keep[5:].all()
    one = np.zeros((6, 2))
    one[0] = [3, 3]  # 3 counts in one sample: a CPM above the cutoff there, but 3 < 4 in all
    one[1, 1] = 1
    flat = np.full(6, 1e4)  # cutoff: 3 / 1e4 * 1e6 = 300 CPM
    assert list(milode.filter_by_expr(np.hstack([counts, one]), flat)[-2:]) == [False, True]
    f = milode.tmm_factors(counts[:, 5:], lib)
    assert np.isclose(np.exp(np.log(f).mean()), 1.0) and np.allclose(f, 1.0, atol=0.1)
    doubled = counts.copy()
    doubled[0, 5:10] *= 20  # a few genes way up in one sample inflate its library...
    f2 = milode.tmm_factors(doubled[:, 5:], doubled.sum(axis=1))
    assert f2[0] < 0.6  # ...which TMM, trimming them, scales back down


# --- the run -------------------------------------------------------------------------


def test_a_planted_regional_effect_is_found_and_little_else():
    a = synthetic()
    r = _run(a, k=15, prop=0.1)
    assert r.tested.sum() >= 5
    genes = {g["gene"]: g for g in r.gene_table()}
    pc3 = a.obsm["X_pca"][r.index, 3]
    t0 = np.array(r.labels) == "t0"
    region = t0 & (pc3 > 0.5)
    outside = ~t0
    sig = r.significant
    gi = [r.genes.index(g) for g in PLANTED]
    # Up in B where it was planted, in most of those neighbourhoods...
    hit = sig[np.ix_(region & r.tested, gi)]
    assert hit.mean() > 0.7
    assert (r.log_fc[np.ix_(region & r.tested, gi)] > 0).all()
    # ...and not in the other cell types, nor for the other genes.
    assert sig[np.ix_(outside & r.tested, gi)].mean() < 0.05
    others = [i for i, g in enumerate(r.genes) if g not in PLANTED]
    assert sig[np.ix_(r.tested, others)].mean() < 0.02
    assert genes["G0"]["n_de"] > 0 and genes["G0"]["share"] > 0.1
    d = r.to_dict(milode.milo.layouts(a, r.index))
    assert d["genes"][0]["gene"] in PLANTED
    assert "X_umap" in d["layouts"] and d["n_tested"] == int(r.tested.sum())
    json.dumps(d)


def test_neighbourhood_needs_two_samples_per_condition():
    a = synthetic()
    # Donors d2 and d3 have no B cells in type t1: its neighbourhoods keep one B sample... none.
    drop = (a.obs["type"] == "t1") & (a.obs["cond"] == "B") & a.obs["donor"].isin(["d1", "d2", "d3"])
    a = a[~drop.to_numpy()].copy()
    p = milode.plan(a, "cond", "donor", conditions=["A", "B"], k=15, genes="expressed")
    r = milode.run(a, p, label_key="type")
    t1 = np.array(r.labels) == "t1"
    assert t1.any() and not r.tested[t1].any()
    assert r.tested[~t1].any()
    assert any("were not tested" in n for n in r.notes)


def test_one_sided_neighbourhoods_are_flagged(monkeypatch):
    r = _run(synthetic(), k=15)
    assert not any("one condition" in w for w in r.warnings)
    assert np.nanmin(r.minority_share[r.tested]) > 0.3
    monkeypatch.setattr(milode, "ONE_SIDED_WARN", 0.01)
    r = _run(synthetic(separate=True), k=30)
    t2 = r.tested & (np.array(r.labels) == "t2")
    assert t2.any() and (r.minority_share[t2] < milode.ONE_SIDED).all()
    assert any("from one condition" in w for w in r.warnings)
    d = r.to_dict()
    assert set(d["nhoods"]["cells"]) == {"A", "B"} and d["one_sided"] == milode.ONE_SIDED


def test_stop_ends_the_run_between_chunks():
    a = synthetic()
    p = milode.plan(a, "cond", "donor", k=15, genes="expressed")
    events = list(milode.iter_milode(a, p, stop=lambda: True))
    assert events and not any("result" in e for e in events)


def test_unpaired_design_and_gene_and_nhood_reads():
    a = synthetic()
    r = _run(a, k=15, paired=False)
    assert r.design == "unpaired" and set(x for x in r.nhood_design if x) == {"unpaired"}
    g = r.gene_detail("g0")  # case-insensitive
    assert g["gene"] == "G0" and len(g["log_fc"]) == len(r.index)
    with pytest.raises(MiloDEError):
        r.gene_detail("NOPE")
    i = int(np.flatnonzero(r.tested)[0])
    nd = r.nhood_detail(i, limit=5)
    assert nd["tested"] and len(nd["genes"]) == 5 and nd["n_genes_tested"] >= 5
    ps = [x["pval"] for x in nd["genes"]]
    assert ps == sorted(ps)
    with pytest.raises(MiloDEError):
        r.nhood_detail(10_000)


def test_bad_requests(monkeypatch):
    a = synthetic()
    with pytest.raises(MiloDEError, match="k must be"):
        milode.plan(a, "cond", "donor", k=3)
    with pytest.raises(MiloDEError, match="sampled share"):
        milode.plan(a, "cond", "donor", prop=0.9)
    with pytest.raises(MiloDEError, match="not found"):
        milode.plan(a, "nope", "donor")
    with pytest.raises(MiloDEError, match="'hvg' or 'expressed'"):
        milode.plan(a, "cond", "donor", genes="all")
    b = a.copy()
    b.X = sp.csr_matrix(np.log1p(b.X.toarray()) + 0.37)
    with pytest.raises(MiloDEError, match="raw counts"):
        milode.plan(b, "cond", "donor")
    # No HVG flag: falls back to the expressed genes, with a note.
    p = milode.plan(a, "cond", "donor", k=15)
    assert p.genes_mode == "expressed" and any("highly variable" in n for n in p.notes)
    a.var["highly_variable"] = [i < 40 for i in range(N_GENES)]
    p = milode.plan(a, "cond", "donor", k=15)
    assert p.genes_mode == "hvg" and len(p.genes) == 40
    # Above MAX_SAMPLED centres the sampled share is lowered, with a note.
    monkeypatch.setattr(milode, "MAX_SAMPLED", 100)
    p = milode.plan(a, "cond", "donor", k=15)
    assert p.prop == round(100 / a.n_obs, 4) and any("sampled as neighbourhood centres" in n for n in p.notes)


def test_results_store_round_trip_and_export(tmp_path):
    a = synthetic()
    r = _run(a, k=15)

    class Adaptor:
        h5ad_path = str(tmp_path / "uploads" / "dsx" / "data.h5ad")

    ad_ = Adaptor()
    ad_.adata = a
    milode.remember(ad_, r)
    results_store.flush()
    entries = results_store.entries("dsx", "milode")
    assert len(entries) == 1
    got = results_store.recall("dsx", entries[0]["id"], "milode")
    assert isinstance(got, milode.MiloDEResult)
    assert np.allclose(got.log_fc, r.log_fc, equal_nan=True)
    df = results_export.table(entries[0], got)
    assert {"gene", "nhood", "logFC", "pval", "spatial_fdr"} <= set(df.columns)
    assert len(df) == int(np.isfinite(r.pval).sum())
    lines, detail = milode.grounding_lines(ad_)
    assert "[result:milode]" in lines[1] and detail
    del a.obs["type"]
    assert milode.grounding_lines(ad_) == ([], None)


def test_claims_tag_is_its_own_family():
    from scview.core.claims import resolve_tag

    refs = ["result:milo", "result:milode"]
    assert resolve_tag("result:milode", refs) == ["result:milode"]
    assert resolve_tag("milode", refs) == ["result:milode"]
    assert resolve_tag("milo", refs) == ["result:milo"]
    assert resolve_tag("result:milo", refs) == ["result:milo"]


# --- endpoints -------------------------------------------------------------------------


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


async def test_plan_endpoint(client, dataset_manager):
    a = synthetic()
    a.obs = a.obs.rename(columns={"donor": "Donor", "cond": "stim"})
    _dataset(Path(dataset_manager.data_dir), a)
    body = {"condition_key": "stim", "k": 15, "genes": "expressed"}
    r = await client.post("/api/v1/datasets/ds/milode/plan", json=body)
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["sample_key"] == "Donor" and d["sample_key_detected"] is True
    assert d["design"] == "paired" and d["order"] == 2 and d["k"] == 15
    assert d["n_nhoods"] > 0 and d["n_testable"] > 0 and d["median_samples_per_nhood"] > 0
    assert d["estimate_seconds"] > 0 and d["n_genes"] > 50
    r = await client.post("/api/v1/datasets/ds/milode/plan", json={**body, "k": 99})
    assert r.status_code == 400 and "k must be" in r.json()["detail"]
    r = await client.post("/api/v1/datasets/missing/milode/plan", json=body)
    assert r.status_code == 404


async def test_run_streams_off_the_loop_and_serves_reads(client, dataset_manager, monkeypatch):
    from scview.core.assistant import build_grounding_context

    _dataset(Path(dataset_manager.data_dir), synthetic())
    threads: list[threading.Thread] = []
    real = milode.test_nhood

    def spy(*args, **kwargs):
        threads.append(threading.current_thread())
        return real(*args, **kwargs)

    monkeypatch.setattr(milode, "test_nhood", spy)
    r = await client.get("/api/v1/datasets/ds/milode/gene", params={"gene": "G0"})
    assert r.status_code == 404
    body = {"condition_key": "cond", "sample_key": "donor", "label_key": "type", "k": 15,
            "genes": "expressed"}
    r = await client.post("/api/v1/datasets/ds/milode/run", json=body)
    assert r.status_code == 200
    events = _events(r.text)
    kinds = [e for e, _ in events]
    assert kinds[0] == "start" and kinds[-1] == "complete" and "result" in kinds
    steps = [d for e, d in events if e == "progress"]
    assert steps[0]["step"] == "counts" and steps[-1]["step"] == "fdr"
    assert steps[-2]["done"] == steps[-2]["total"]
    result = next(d for e, d in events if e == "result")
    assert result["n_tested"] > 0 and result["genes"][0]["gene"] in PLANTED
    assert threads and all(t is not MAIN for t in threads)

    r = await client.get("/api/v1/datasets/ds/milode/gene", params={"gene": "G0"})
    assert r.status_code == 200 and any(r.json()["significant"])
    r = await client.get("/api/v1/datasets/ds/milode/gene", params={"gene": "NOPE"})
    assert r.status_code == 400
    i = result["nhoods"]["tested"].index(True)
    r = await client.get(f"/api/v1/datasets/ds/milode/nhood/{i}")
    assert r.status_code == 200 and r.json()["genes"]
    r = await client.post("/api/v1/datasets/ds/milode/cells", json={"nhoods": [i]})
    assert r.status_code == 200 and r.json()["n_cells"] == result["nhoods"]["size"][i]

    adaptor = await dataset_manager.get_or_load_dataset("ds")
    context, sources = build_grounding_context(adaptor, {"groups"})
    assert "Neighbourhood DE (miloDE-style" in context
    assert any(s.ref == "result:milode" for s in sources)


async def test_run_rejects_a_bad_request_before_streaming(client, dataset_manager):
    _dataset(Path(dataset_manager.data_dir), synthetic())
    r = await client.post("/api/v1/datasets/ds/milode/run",
                          json={"condition_key": "cond", "sample_key": "cond"})
    assert r.status_code == 400
