"""POST /genesets/score with method ssgsea | gsva, and both in the depth check (C097)."""

from __future__ import annotations

import numpy as np

from scview.core import gsva, ssgsea
from tests.test_geneset_score_methods import API, SET, _adata, _register


async def test_ssgsea_method(client, dataset_manager):
    a = _adata()
    did = _register(dataset_manager, a)
    r = await client.post(f"{API}/datasets/{did}/genesets/score",
                          json={"gene_set": [*SET, "G199"], "method": "ssgsea"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["method"] == "ssgsea" and body["source"] == "log-normalised .raw"
    # Every gene is ranked, also one no cell detects (G199), as GSVA does.
    assert body["genes_found"] == ["G0", "G1", "G2", "G3", "G4", "G5", "G199"]
    assert body["genes_missing"] == ["NOSUCHGENE"] and body["genes_undetected"] == []
    assert "alpha 0.25" in body["note"]
    # Ranks of the counts are the ranks of the log values.
    want = ssgsea.score(a.layers["counts"], [0, 1, 2, 3, 4, 5, 199])
    np.testing.assert_allclose(body["scores"], want, rtol=1e-9, atol=1e-12)
    s = np.asarray(body["scores"])
    assert s[:120].mean() > s[120:].mean()  # the cluster that raises G0-G9


async def test_gsva_method(client, dataset_manager):
    a = _adata()
    did = _register(dataset_manager, a)
    r = await client.post(f"{API}/datasets/{did}/genesets/score",
                          json={"gene_set": [*SET, "G199"], "method": "gsva"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["method"] == "gsva" and body["source"] == "log-normalised .raw"
    assert body["genes_found"] == ["G0", "G1", "G2", "G3", "G4", "G5"]
    assert body["genes_undetected"] == ["G199"]  # the same (0) in every cell: left out
    assert "empirical CDF over all 240 cells" in body["note"]
    raw = a.raw.X
    want = gsva.score(gsva.prepare(raw), raw, range(6))
    np.testing.assert_allclose(body["scores"], want, rtol=0, atol=1e-12)
    s = np.asarray(body["scores"])
    assert -1 <= s.min() and s.max() <= 1
    assert s[:120].mean() > s[120:].mean() + 0.3


async def test_dense_matches_sparse(client, dataset_manager):
    did_s = _register(dataset_manager, _adata(), "sparse")
    did_d = _register(dataset_manager, _adata(dense=True), "dense")
    for method in ("ssgsea", "gsva"):
        body = {"gene_set": SET, "method": method}
        rs = await client.post(f"{API}/datasets/{did_s}/genesets/score", json=body)
        rd = await client.post(f"{API}/datasets/{did_d}/genesets/score", json=body)
        assert rs.status_code == rd.status_code == 200
        np.testing.assert_allclose(rs.json()["scores"], rd.json()["scores"], rtol=1e-12, atol=1e-12)


async def test_errors_and_size_limit(client, dataset_manager, monkeypatch):
    did = _register(dataset_manager, _adata())
    url = f"{API}/datasets/{did}/genesets/score"
    for method in ("ssgsea", "gsva"):
        r = await client.post(url, json={"gene_set": ["NOPE1", "NOPE2"], "method": method})
        assert r.status_code == 422 and "None of the 2" in r.json()["detail"]
    r = await client.post(url, json={"gene_set": ["G199"], "method": "gsva"})
    assert r.status_code == 422 and "varies between cells" in r.json()["detail"]
    monkeypatch.setattr(ssgsea, "MAX_VALUES", 1000)
    for method, label in (("ssgsea", "ssGSEA"), ("gsva", "GSVA")):
        r = await client.post(url, json={"gene_set": ["G0", "G1"], "method": method})
        assert r.status_code == 422, r.text
        assert r.json()["detail"].startswith(f"{label} scores up to 0 million non-zero values")


async def test_depth_check_has_all_four_methods(client, dataset_manager, monkeypatch):
    a = _adata()
    a.uns["scview_active_clustering"] = "cluster"
    did = _register(dataset_manager, a)
    genes = ["G0", "G1", "G2", "G3", "G20", "G21"]
    url = f"{API}/datasets/{did}/genesets/score"
    r = await client.post(f"{url}/compare", json={"gene_set": genes})
    assert r.status_code == 200, r.text
    body = r.json()
    assert list(body["methods"]) == ["score_genes", "aucell", "ssgsea", "gsva"]
    assert list(body["agreement"]) == ["aucell", "ssgsea", "gsva"] and body["skipped"] == {}
    gs = np.asarray((await client.post(url, json={"gene_set": genes, "method": "gsva"})).json()["scores"])
    sg = np.asarray((await client.post(url, json={"gene_set": genes})).json()["scores"])
    from scview.core import aucell

    assert body["agreement"]["gsva"]["overall"] == aucell.spearman(sg, gs)
    assert body["between"]["overall"] == body["agreement"]["aucell"]["overall"]
    assert "; ssGSEA " in body["sentence"] and "; GSVA " in body["sentence"]
    # Narrowed, and too large: left out with the reason.
    r = await client.post(f"{url}/compare", json={"gene_set": genes, "methods": ["gsva"]})
    assert list(r.json()["methods"]) == ["score_genes", "aucell", "gsva"]
    monkeypatch.setattr(ssgsea, "MAX_VALUES", 1000)
    r = await client.post(f"{url}/compare", json={"gene_set": genes})
    body = r.json()
    assert list(body["methods"]) == ["score_genes", "aucell"]
    assert set(body["skipped"]) == {"ssgsea", "gsva"} and "AUCell or score_genes" in body["skipped"]["gsva"]
