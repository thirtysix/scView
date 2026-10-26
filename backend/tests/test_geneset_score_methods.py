"""POST /genesets/score with method score_genes | aucell, and the depth check (C097)."""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from scipy import stats

from scview.core import aucell

API = "/api/v1"
SET = ["G0", "G1", "G2", "G3", "G4", "g5", "NOSUCHGENE"]


def _adata(n: int = 240, g: int = 200, seed: int = 0, dense: bool = False) -> ad.AnnData:
    """Counts with a 10-fold spread in depth, two clusters (the first raises G0-G9),
    log-normalised .raw, counts in layers and obs, a scaled X and a gene never detected."""
    rng = np.random.default_rng(seed)
    size = rng.lognormal(0.0, 0.8, size=(n, 1))
    mu = np.full((n, g), 0.6)
    mu[: n // 2, :10] *= 5
    X = rng.poisson(mu * size).astype(np.float32)
    X[:, g - 1] = 0
    a = ad.AnnData(X=sp.csr_matrix(X))
    a.var_names = [f"G{i}" for i in range(g)]
    a.obs_names = [f"C{i}" for i in range(n)]
    a.obs["cluster"] = pd.Categorical(["0"] * (n // 2) + ["1"] * (n - n // 2))
    a.obs["total_counts"] = X.sum(axis=1).astype(np.float64)
    a.layers["counts"] = a.X.copy()
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    if dense:
        a.X = a.X.toarray()
    a.raw = a
    a.X = a.X.toarray() if sp.issparse(a.X) else a.X.copy()  # dense: never scale .raw's array
    sc.pp.scale(a, max_value=10)
    return a


def _register(dm, a: ad.AnnData, did: str = "ds1") -> str:
    d = Path(dm.data_dir) / "uploads" / did
    d.mkdir(parents=True, exist_ok=True)
    (d / "metadata.json").write_text(json.dumps({"id": did, "name": did, "status": "ready"}))
    a.write_h5ad(d / "data.h5ad")
    return did


async def test_default_method_is_score_genes_unchanged(client, dataset_manager):
    a = _adata()
    did = _register(dataset_manager, a)
    r = await client.post(f"{API}/datasets/{did}/genesets/score", json={"gene_set": SET, "score_name": "s"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["method"] == "score_genes" and body["max_rank"] is None
    assert body["genes_found"] == ["G0", "G1", "G2", "G3", "G4", "G5"]
    assert body["genes_missing"] == ["NOSUCHGENE"]
    ref = a.copy()
    sc.tl.score_genes(ref, ["G0", "G1", "G2", "G3", "G4", "G5"], score_name="s", use_raw=True)
    np.testing.assert_allclose(body["scores"], ref.obs["s"].to_numpy(), rtol=1e-6, atol=1e-7)


async def test_aucell_method(client, dataset_manager):
    a = _adata()
    did = _register(dataset_manager, a)
    r = await client.post(f"{API}/datasets/{did}/genesets/score",
                          json={"gene_set": [*SET, "G199"], "method": "aucell"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["method"] == "aucell"
    assert body["source"] == "log-normalised .raw"  # never the scaled X
    assert body["max_rank"] == 10  # ceil(5% of the 199 detected genes)
    assert body["genes_found"] == ["G0", "G1", "G2", "G3", "G4", "G5"]
    assert body["genes_missing"] == ["NOSUCHGENE"]
    assert body["genes_undetected"] == ["G199"]
    # Ranks of the counts are the ranks of the log values: the expected scores from the
    # counts layer, built directly.
    rk = aucell.build(a.layers["counts"], a.var_names, max_rank=10)
    np.testing.assert_allclose(body["scores"], aucell.score(rk, range(6)), rtol=0, atol=1e-12)
    assert 0 <= body["min_score"] <= body["max_score"] <= 1
    assert body["share_below_max_rank"] == round(float(np.mean(rk.n_detected < 10)), 4)
    # The cluster that raises G0-G9 scores higher.
    s = np.asarray(body["scores"])
    assert s[:120].mean() > s[120:].mean() + 0.2


async def test_aucell_dense_matches_sparse(client, dataset_manager):
    did_s = _register(dataset_manager, _adata(), "sparse")
    did_d = _register(dataset_manager, _adata(dense=True), "dense")
    body = {"gene_set": SET, "method": "aucell", "max_rank": 25}
    rs = await client.post(f"{API}/datasets/{did_s}/genesets/score", json=body)
    rd = await client.post(f"{API}/datasets/{did_d}/genesets/score", json=body)
    assert rs.status_code == rd.status_code == 200
    assert rs.json()["max_rank"] == 25
    np.testing.assert_array_equal(rs.json()["scores"], rd.json()["scores"])


async def test_aucell_errors(client, dataset_manager):
    did = _register(dataset_manager, _adata())
    url = f"{API}/datasets/{did}/genesets/score"
    r = await client.post(url, json={"gene_set": ["G0"], "method": "aucell", "max_rank": 500})
    assert r.status_code == 422 and "max_rank" in r.json()["detail"]
    r = await client.post(url, json={"gene_set": ["G0"], "method": "aucell", "max_rank": 1})
    assert r.status_code == 422
    r = await client.post(url, json={"gene_set": ["G0"], "method": "bogus"})
    assert r.status_code == 422
    for method in ("score_genes", "aucell"):
        r = await client.post(url, json={"gene_set": ["NOPE1", "NOPE2"], "method": method})
        assert r.status_code == 400 and "None of the 2" in r.json()["detail"]
    r = await client.post(url, json={"gene_set": ["G199"], "method": "aucell"})
    assert r.status_code == 400 and "never detected" in r.json()["detail"]


async def test_depth_check_matches_scipy(client, dataset_manager):
    a = _adata()
    a.obs["half"] = a.obs["cluster"].astype(str)  # not a clustering column
    did = _register(dataset_manager, a)
    genes = ["G0", "G1", "G2", "G3", "G4", "G20", "G21", "G22"]
    url = f"{API}/datasets/{did}/genesets/score"
    sg = np.asarray((await client.post(url, json={"gene_set": genes})).json()["scores"])
    au = np.asarray((await client.post(url, json={"gene_set": genes, "method": "aucell"})).json()["scores"])
    r = await client.post(f"{url}/compare", json={"gene_set": genes, "groupby": "half"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["depth"]["total_counts"] == "obs['total_counts']"
    lead = body["metric"]
    assert lead in ("total_counts", "n_genes")
    assert body["groupby"] == "half" and body["n_groups"] == 2

    depth = a.obs["total_counts"].to_numpy()
    detected = np.asarray((a.layers["counts"] > 0).sum(axis=1)).ravel()
    halves = (slice(0, 120), slice(120, 240))
    for name, s in (("score_genes", sg), ("aucell", au)):
        for metric, d in (("total_counts", depth), ("n_genes", detected)):
            got = body["methods"][name][metric]
            assert got["overall"] == pytest_approx(stats.spearmanr(s, d).statistic)
            within = np.median([stats.spearmanr(s[h], d[h]).statistic for h in halves])
            assert got["within"] == pytest_approx(within)
    assert body["between"]["overall"] == pytest_approx(stats.spearmanr(sg, au).statistic)
    # The sentence reports the within-group numbers.
    sg_w = body["methods"]["score_genes"][lead]["within"]
    au_w = body["methods"]["aucell"][lead]["within"]
    other = "n_genes" if lead == "total_counts" else "total_counts"
    assert max(abs(sg_w), abs(au_w)) >= max(abs(body["methods"][m][other]["within"]) for m in ("score_genes", "aucell"))
    assert body["sentence"].startswith(f"score_genes {aucell.verdict(sg_w)} on this data ")
    assert f"(rho {sg_w:.2f} with {aucell.METRIC_TEXT[lead]}, within half groups)" in body["sentence"]
    assert f"AUCell {aucell.verdict(au_w)} (rho {au_w:.2f}); ssGSEA " in body["sentence"]


async def test_depth_check_defaults_to_the_active_clustering(client, dataset_manager):
    a = _adata()
    a.uns["scview_active_clustering"] = "cluster"
    did = _register(dataset_manager, a)
    r = await client.post(f"{API}/datasets/{did}/genesets/score/compare", json={"gene_set": ["G0", "G1", "G2"]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["groupby"] == "cluster" and "within clusters" in body["sentence"]
    r = await client.post(f"{API}/datasets/{did}/genesets/score/compare",
                          json={"gene_set": ["G0"], "groupby": "nope"})
    assert r.status_code == 404


def pytest_approx(v):
    import pytest

    return pytest.approx(v, abs=1e-9)
