"""Tests for the on-demand QC distributions (adaptor.qc_distributions)."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd

from scview.core.anndata_adaptor import AnnDataAdaptor


def _write_h5ad(tmp_path, obs=None, n_cells=200, n_genes=60, with_mt=True):
    rng = np.random.default_rng(0)
    X = rng.poisson(1.0, size=(n_cells, n_genes)).astype("float32")
    a = ad.AnnData(X)
    genes = [f"G{i}" for i in range(n_genes)]
    if with_mt:
        genes[0] = "MT-CO1"
        genes[1] = "MT-ND1"
    a.var_names = genes
    a.obs_names = [f"C{i}" for i in range(n_cells)]
    if obs is not None:
        a.obs = obs
    p = tmp_path / "d.h5ad"
    a.write_h5ad(p)
    return str(p)


def test_qc_computed_on_demand_when_obs_missing(tmp_path):
    path = _write_h5ad(tmp_path)
    qc = AnnDataAdaptor(path).qc_distributions()
    assert qc["computed_on_demand"] is True
    assert qc["n_cells"] == 200
    assert set(qc["metrics"]) >= {"n_genes_by_counts", "total_counts", "pct_counts_mt"}
    m = qc["metrics"]["total_counts"]
    assert m["max"] >= m["median"] >= m["min"]
    assert len(m["hist"]["counts"]) == 40
    assert len(m["hist"]["bin_edges"]) == 41


def test_qc_reads_existing_obs_columns(tmp_path):
    obs = pd.DataFrame(
        {
            "total_counts": np.arange(200, dtype=float) + 1,
            "n_genes_by_counts": np.arange(200, dtype=float) + 1,
            "pct_counts_mt": np.linspace(0, 10, 200),
        },
        index=[f"C{i}" for i in range(200)],
    )
    path = _write_h5ad(tmp_path, obs=obs)
    qc = AnnDataAdaptor(path).qc_distributions()
    assert qc["computed_on_demand"] is False
    # median of total_counts 1..200 is 100.5
    assert abs(qc["metrics"]["total_counts"]["median"] - 100.5) < 1e-6


def test_qc_includes_doublet_score_when_present(tmp_path):
    obs = pd.DataFrame(
        {"doublet_score": np.linspace(0, 1, 200)},
        index=[f"C{i}" for i in range(200)],
    )
    path = _write_h5ad(tmp_path, obs=obs)
    qc = AnnDataAdaptor(path).qc_distributions()
    assert "doublet_score" in qc["metrics"]


def test_qc_grows_with_optional_metrics(tmp_path):
    """Optional QC metrics appear only once present in obs (plot set grows)."""
    idx = [f"C{i}" for i in range(200)]
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    base = _write_h5ad(base_dir)
    assert set(AnnDataAdaptor(base).qc_distributions()["metrics"]) == {
        "n_genes_by_counts", "total_counts", "pct_counts_mt",
    }
    obs = pd.DataFrame(
        {
            "S_score": np.linspace(-1, 1, 200),
            "G2M_score": np.linspace(-1, 1, 200),
            "pct_counts_ribo": np.linspace(0, 40, 200),
        },
        index=idx,
    )
    enr_dir = tmp_path / "enriched"
    enr_dir.mkdir()
    enriched = _write_h5ad(enr_dir, obs=obs)
    metrics = AnnDataAdaptor(enriched).qc_distributions()["metrics"]
    assert {"S_score", "G2M_score", "pct_counts_ribo"} <= set(metrics)


def test_qc_scatter_downsampled_and_cached(tmp_path):
    path = _write_h5ad(tmp_path, n_cells=12000)
    adaptor = AnnDataAdaptor(path)
    qc = adaptor.qc_distributions(n_scatter=5000)
    assert qc["scatter"]["n_shown"] == 5000
    assert len(qc["scatter"]["x"]) == 5000
    # second call returns the cached object
    assert adaptor.qc_distributions() is qc


def test_row_sums_in_small_blocks_equal_one_pass(monkeypatch):
    import scipy.sparse as sp

    from scview.core import anndata_adaptor as mod

    rng = np.random.default_rng(1)
    M = sp.random(300, 50, density=0.3, format="csr", random_state=2, dtype=np.float32)
    M.data = rng.integers(1, 9, M.nnz).astype(np.float32)
    mask = np.zeros(50, bool)
    mask[:3] = True
    whole = mod._row_sums(M, [mask])
    monkeypatch.setattr(mod, "ROW_BLOCK_VALUES", 37)  # many uneven blocks, some of one row
    spans = mod._row_spans(M, 20000)
    assert len(spans) > 50 and spans[0][0] == 0 and spans[-1][1] == 300
    assert all(a < b for a, b in spans) and all(s[1] == t[0] for s, t in zip(spans, spans[1:]))
    for dense in (False, True):
        got = mod._row_sums(M.toarray() if dense else M, [mask])
        for w, g in zip((whole[0], whole[1], whole[2][0]), (got[0], got[1], got[2][0])):
            np.testing.assert_allclose(g, w)


def test_concurrent_first_calls_make_one_pass(tmp_path, monkeypatch):
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor

    from scview.core import anndata_adaptor as mod

    adaptor = AnnDataAdaptor(_write_h5ad(tmp_path))
    real, calls, lock = mod.per_cell_qc, [], threading.Lock()

    def slow(*a, **k):
        with lock:
            calls.append(1)
        time.sleep(0.2)
        return real(*a, **k)

    monkeypatch.setattr(mod, "per_cell_qc", slow)
    with ThreadPoolExecutor(4) as ex:
        out = list(ex.map(lambda _: adaptor.qc_per_cell(ribo=True), range(4)))
    assert len(calls) == 1 and all(o is out[0] for o in out)
