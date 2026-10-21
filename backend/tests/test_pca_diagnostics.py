"""PCA diagnostics (core/pca_diagnostics.py, GET /pca/diagnostics): variance ratios,
Kneedle elbow, PCs to 90%, loadings without the zero rows, the neighbour graph's PC
count, dense and sparse inputs, a dataset without a PCA, and the co-pilot line."""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
import scipy.sparse as sp

from scview.core import pca_diagnostics as pd_
from scview.core.anndata_adaptor import AnnDataAdaptor
from scview.core.assistant import build_grounding_context

API = "/api/v1"


def _adata(sparse: bool, n_cells: int = 120, n_genes: int = 60, seed: int = 0) -> ad.AnnData:
    """Log-normalised cells with two groups; PCA on 30 variable genes with 10 PCs."""
    rng = np.random.default_rng(seed)
    X = rng.poisson(1.0, (n_cells, n_genes)).astype(np.float32)
    X[: n_cells // 2, :5] += 8  # the first five genes split the two groups: PC1
    X = np.log1p(X)
    a = ad.AnnData(sp.csr_matrix(X) if sparse else X)
    a.var_names = [f"G{i}" for i in range(n_genes)]
    a.obs_names = [f"c{i}" for i in range(n_cells)]
    a.obs["group"] = pd.Categorical(["A"] * (n_cells // 2) + ["B"] * (n_cells - n_cells // 2))
    a.var["highly_variable"] = [i < 30 for i in range(n_genes)]
    sc.pp.pca(a, n_comps=10, mask_var="highly_variable")
    return a


@pytest.mark.parametrize("sparse", [False, True])
def test_diagnostics_from_stored_pca(sparse):
    a = _adata(sparse)
    d = pd_.pca_diagnostics(a)
    assert d["available"] and d["n_cells"] == 120 and d["n_pcs"] == 10
    assert d["n_genes_total"] == 60 and d["n_genes_used"] == 30
    assert d["params"] == {"n_comps": 10, "use_highly_variable": True,
                           "mask_var": "highly_variable", "zero_center": True}
    vr = np.asarray(d["variance_ratio"])
    assert len(vr) == 10 and np.all(np.diff(vr) <= 1e-9)  # decreasing
    assert vr[0] == pytest.approx(float(a.uns["pca"]["variance_ratio"][0]))
    assert d["cumulative"][-1] == pytest.approx(d["retained"]) and d["retained"] < 1
    assert d["ratio_basis"] == "total"
    assert 1 <= d["elbow"] <= 10 and 1 <= d["pcs_90"] <= 10
    assert d["neighbors"] is None  # no graph yet
    assert "Kneedle" in d["method"] and "90%" in d["method"]
    # PC1 separates the groups: the planted genes top its loadings, no zero rows.
    pc1 = d["loadings"][0]
    assert pc1["pc"] == 1
    top = {g["gene"] for g in pc1["positive"][:5]} | {g["gene"] for g in pc1["negative"][:5]}
    assert {"G0", "G1", "G2", "G3", "G4"} <= top
    for entry in d["loadings"]:
        genes = [g["gene"] for g in entry["positive"] + entry["negative"]]
        assert all(int(g[1:]) < 30 for g in genes)  # only variable genes load
        assert all(g["loading"] > 0 for g in entry["positive"])
        assert all(g["loading"] < 0 for g in entry["negative"])
        assert len(entry["positive"]) <= 15 and len(entry["negative"]) <= 15


def test_top_k_and_signs():
    a = _adata(False)
    d = pd_.pca_diagnostics(a, k=3)
    assert all(len(e["positive"]) <= 3 and len(e["negative"]) <= 3 for e in d["loadings"])
    pos = d["loadings"][0]["positive"]
    assert [g["loading"] for g in pos] == sorted((g["loading"] for g in pos), reverse=True)


def test_kneedle_elbow_and_share_rules():
    # A sharp bend after the 3rd value: Kneedle lands on it.
    assert pd_.kneedle_elbow(np.array([10, 5, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5])) == 3
    assert pd_.kneedle_elbow(np.array([1.0, 1.0, 1.0])) == 3  # flat: no bend
    assert pd_.kneedle_elbow(np.array([2.0, 1.0])) == 2
    assert pd_.pcs_to_share(np.array([0.5, 0.3, 0.1, 0.1])) == 3  # 0.9 reached at the 3rd
    assert pd_.pcs_to_share(np.array([0.9, 0.1])) == 1
    assert pd_.pcs_to_share(np.array([0.0, 0.0])) == 2


def test_neighbors_pc_count_from_params():
    a = _adata(False)
    sc.pp.neighbors(a, n_neighbors=8, n_pcs=6)
    nb = pd_.pca_diagnostics(a)["neighbors"]
    assert nb == {"use_rep": "X_pca", "n_pcs": 6, "n_neighbors": 8, "from_pca": True}
    # Only use_rep recorded (scanpy's default): every column of it.
    a.obsm["X_pca_harmony"] = a.obsm["X_pca"].copy()
    a.uns["neighbors"]["params"] = {"n_neighbors": 15, "use_rep": "X_pca_harmony", "method": "umap"}
    nb = pd_.pca_diagnostics(a)["neighbors"]
    assert nb["use_rep"] == "X_pca_harmony" and nb["n_pcs"] == 10 and nb["from_pca"]
    a.obsm["X_scvi"] = np.zeros((a.n_obs, 4), dtype=np.float32)
    a.uns["neighbors"]["params"] = {"n_neighbors": 15, "use_rep": "X_scvi"}
    nb = pd_.pca_diagnostics(a)["neighbors"]
    assert nb["use_rep"] == "X_scvi" and nb["n_pcs"] == 4 and not nb["from_pca"]


def test_no_pca_and_pca_without_uns():
    a = _adata(False)
    del a.obsm["X_pca"]
    assert pd_.pca_diagnostics(a) == {"available": False,
                                      "reason": "No PCA yet: run the pipeline's PCA step first."}
    # A converted object: X_pca without uns["pca"] or varm["PCs"].
    b = _adata(False)
    del b.uns["pca"], b.varm["PCs"]
    d = pd_.pca_diagnostics(b)
    assert d["available"] and d["ratio_basis"] == "retained"
    assert d["cumulative"][-1] == pytest.approx(1.0) and d["loadings"] == []
    assert d["n_genes_used"] is None and d["params"]["use_highly_variable"] is None
    assert "no stored variance ratios" in d["method"]


def _register(dm, adata, did: str = "ds") -> str:
    d = Path(dm.data_dir) / "uploads" / did
    d.mkdir(parents=True, exist_ok=True)
    (d / "metadata.json").write_text(json.dumps({"id": did, "name": did, "status": "ready"}))
    adata.write_h5ad(d / "data.h5ad")
    return did


async def test_endpoint(client, dataset_manager):
    did = _register(dataset_manager, _adata(True))
    r = await client.get(f"{API}/datasets/{did}/pca/diagnostics")
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["available"] and d["n_pcs"] == 10 and len(d["loadings"]) == 10
    r = await client.get(f"{API}/datasets/{did}/pca/diagnostics", params={"top_k": 2})
    assert all(len(e["positive"]) <= 2 for e in r.json()["loadings"])
    assert (await client.get(f"{API}/datasets/nope/pca/diagnostics")).status_code == 404
    a = _adata(False)
    del a.obsm["X_pca"]
    did2 = _register(dataset_manager, a, "raw")
    r = await client.get(f"{API}/datasets/{did2}/pca/diagnostics")
    assert r.status_code == 200 and r.json()["available"] is False


def test_grounding_line_for_the_copilot(tmp_path: Path):
    a = _adata(False)
    sc.pp.neighbors(a, n_neighbors=8, n_pcs=6)
    p = tmp_path / "d.h5ad"
    a.write_h5ad(p)
    context, sources = build_grounding_context(AnnDataAdaptor(str(p)))
    fact = next(s for s in sources if s.ref == "result:pca")
    assert fact.detail.startswith("10 PCs: PC1 ")
    assert "elbow at PC" in fact.detail and "reach 90% of the retained variance" in fact.detail
    assert "the neighbour graph used 6 PCs of X_pca" in fact.detail
    assert "[result:pca]" in context and "PC1 top loadings: positive" in fact.evidence
    assert "Caveat" in fact.evidence
    del a.obsm["X_pca"]
    a.write_h5ad(p)
    assert "result:pca" not in {s.ref for s in build_grounding_context(AnnDataAdaptor(str(p)))[1]}


def test_claims_resolve_pca_tag():
    from scview.core.claims import resolve_tag

    refs = ["dataset:summary", "result:pca", "result:qc:1"]
    assert resolve_tag("result:pca", refs) == ["result:pca"]
    assert resolve_tag("pca", refs) == ["result:pca"]
