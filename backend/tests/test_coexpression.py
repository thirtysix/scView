"""Co-expression search (core/coexpression.py, POST .../expression/coexpression)."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from scipy import stats

from scview.core import coexpression as cx

API = "/api/v1"


def _data(n: int = 300, sparse: bool = True, seed: int = 0) -> ad.AnnData:
    """Log-normalised values. G1 follows G0, G2 opposes it, G3 is expressed in
    5 cells only, G4 never varies (all zero), G5.. are noise. Cell type T holds
    the first half."""
    rng = np.random.default_rng(seed)
    g = 30
    base = rng.gamma(2.0, 1.0, n)
    X = rng.poisson(0.8, (n, g)).astype(np.float64)
    X[:, 0] = rng.poisson(base * 2)
    X[:, 1] = rng.poisson(base * 2 + 0.2)
    X[:, 2] = rng.poisson(np.maximum(6 - base * 2, 0.1))
    X[:, 3] = 0
    X[rng.choice(n, 5, replace=False), 3] = 3
    X[:, 4] = 0
    X = np.log1p(X)
    a = ad.AnnData(sp.csr_matrix(X.astype(np.float32)) if sparse else X.astype(np.float32))
    a.var_names = [f"G{i}" for i in range(g)]
    a.obs_names = [f"C{i}" for i in range(n)]
    a.obs["cell_type"] = pd.Categorical(np.where(np.arange(n) < n // 2, "T", "B"))
    return a


def _dense(a: ad.AnnData) -> np.ndarray:
    return a.X.toarray() if sp.issparse(a.X) else np.asarray(a.X)


@pytest.mark.parametrize("sparse", [True, False])
@pytest.mark.parametrize("method", ["pearson", "spearman"])
def test_matches_scipy_on_sparse_and_dense(sparse, method, monkeypatch):
    monkeypatch.setattr(cx, "GENE_CHUNK", 7)  # dense: several gene chunks
    a = _data(sparse=sparse)
    out = cx.coexpression(a, "G0", method=method, top_n=50)
    X = _dense(a).astype(np.float64)
    ref = stats.pearsonr if method == "pearson" else stats.spearmanr
    for row in out["positive"] + out["negative"]:
        j = int(row["gene"][1:])
        assert row["r"] == pytest.approx(ref(X[:, 0], X[:, j]).statistic, abs=1e-4)
        assert row["n_expressing"] == int((X[:, j] > 0).sum())
    assert out["positive"][0]["gene"] == "G1"
    assert out["negative"][0]["gene"] == "G2"
    # (a tiny negative r rounds to -0.0)
    assert all(r["r"] >= 0 for r in out["positive"]) and all(r["r"] <= 0 for r in out["negative"])


def test_skips_rare_and_constant_genes_and_the_query():
    out = cx.coexpression(_data(), "G0")
    genes = {r["gene"] for r in out["positive"] + out["negative"]}
    assert not genes & {"G0", "G3", "G4"}
    assert out["min_expressing"] == cx.MIN_CELLS  # 1% of 300 cells is under 10
    # G3 (5 cells) and G4 (no cells) fall under the expressing-cell minimum.
    assert out["n_skipped_rare"] == 2 and out["n_skipped_constant"] == 0
    assert out["n_genes_tested"] == 30 - 3
    assert out["n_cells"] == 300 and out["values"] == "log-normalised" and out["layer"] == "X"
    for r in out["positive"]:
        assert np.isfinite(r["r"])


def test_restricts_to_the_chosen_cells():
    a = _data()
    rows = np.arange(150)
    out = cx.coexpression(a, "G0", rows=rows, top_n=50)
    assert out["n_cells"] == 150 and out["n_cells_total"] == 300
    X = _dense(a)[:150].astype(np.float64)
    g1 = next(r for r in out["positive"] if r["gene"] == "G1")
    assert g1["r"] == pytest.approx(np.corrcoef(X[:, 0], X[:, 1])[0, 1], abs=1e-4)
    # Duplicates and out-of-range indices are dropped; every cell means all cells.
    every = cx.coexpression(a, "G0", rows=np.r_[np.arange(300), 0, 999])
    assert every["n_cells"] == 300


def test_a_minimum_of_one_percent_of_many_cells():
    out = cx.coexpression(_data(n=2000), "G0")
    assert out["min_expressing"] == 20


def test_refuses_unknown_rare_or_constant_query_genes():
    a = _data()
    with pytest.raises(cx.CoexpressionError, match="not found"):
        cx.coexpression(a, "NOPE")
    with pytest.raises(cx.CoexpressionError, match="expressed in 5 of these 300 cells"):
        cx.coexpression(a, "G3")
    with pytest.raises(cx.CoexpressionError, match="expressed in 0"):
        cx.coexpression(a, "G4")
    with pytest.raises(cx.CoexpressionError, match="at least 10 cells"):
        cx.coexpression(a, "G0", rows=np.arange(4))


def test_counts_are_log_normalised_first_and_raw_is_preferred():
    rng = np.random.default_rng(3)
    counts = rng.poisson(2.0, (200, 12)).astype(np.float32)
    counts[:, 1] = counts[:, 0] * 2
    a = ad.AnnData(sp.csr_matrix(counts))
    a.var_names = [f"G{i}" for i in range(12)]
    out = cx.coexpression(a, "G0", top_n=20)
    assert out["values"] == "log-normalised from counts"
    lib = counts.sum(axis=1, keepdims=True)
    logn = np.log1p(counts / lib * 1e4)
    g1 = next(r for r in out["positive"] if r["gene"] == "G1")
    assert g1["r"] == pytest.approx(np.corrcoef(logn[:, 0], logn[:, 1])[0, 1], abs=1e-4)

    # A .raw (log-normalised, all genes) beside an HVG-subset X is what is read.
    b = _data()
    b.raw = b.copy()
    b = b[:, :10].copy()
    out = cx.coexpression(b, "G0", top_n=50)
    assert out["layer"] == "raw" and out["n_genes_tested"] == 27


def test_scaled_values_skip_the_expressing_minimum():
    a = _data(sparse=False)
    X = _dense(a)
    a.X = ((X - X.mean(0)) / (X.std(0) + 1e-9)).astype(np.float32)
    out = cx.coexpression(a, "G0", top_n=50)
    assert out["values"] == "scaled" and out["min_expressing"] is None
    assert out["positive"][0]["gene"] == "G1"
    assert "G4" not in {r["gene"] for r in out["positive"] + out["negative"]}  # constant


# --- the endpoint --------------------------------------------------------------------


def _register(dm, did: str = "cx") -> str:
    d = Path(dm.data_dir) / "uploads" / did
    d.mkdir(parents=True)
    (d / "metadata.json").write_text(json.dumps({"id": did, "name": did, "status": "ready"}))
    _data().write_h5ad(d / "data.h5ad")
    return did


async def test_endpoint_scopes_and_errors(client, dataset_manager):
    did = _register(dataset_manager)
    url = f"{API}/datasets/{did}/expression/coexpression"
    r = await client.post(url, json={"gene": "G0"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["scope"] == "all cells" and body["positive"][0]["gene"] == "G1"
    assert body["method"] == "pearson"

    r = await client.post(url, json={"gene": "G0", "column": "cell_type", "value": "T",
                                     "method": "spearman", "top_n": 3})
    body = r.json()
    assert (body["scope"], body["n_cells"], body["method"]) == ("cell_type = T", 150, "spearman")
    assert len(body["positive"]) <= 3

    r = await client.post(url, json={"gene": "G0", "indices": list(range(40))})
    assert r.json()["n_cells"] == 40 and r.json()["scope"] == "selection"

    assert (await client.post(url, json={"gene": "NOPE"})).status_code == 404
    assert (await client.post(url, json={"gene": "G3"})).status_code == 400
    both = await client.post(url, json={"gene": "G0", "indices": [1, 2], "column": "cell_type", "value": "T"})
    assert both.status_code == 400
    missing = await client.post(url, json={"gene": "G0", "column": "nope", "value": "T"})
    assert missing.status_code == 404


async def test_endpoint_runs_in_a_worker_thread(client, dataset_manager, monkeypatch):
    did = _register(dataset_manager)
    threads: list[threading.Thread] = []
    real = cx.coexpression

    def spy(*args, **kwargs):
        threads.append(threading.current_thread())
        return real(*args, **kwargs)

    monkeypatch.setattr(cx, "coexpression", spy)
    r = await client.post(f"{API}/datasets/{did}/expression/coexpression", json={"gene": "G0"})
    assert r.status_code == 200
    assert len(threads) == 1 and threads[0] is not threading.main_thread()
