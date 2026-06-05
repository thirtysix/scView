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


def test_qc_scatter_downsampled_and_cached(tmp_path):
    path = _write_h5ad(tmp_path, n_cells=12000)
    adaptor = AnnDataAdaptor(path)
    qc = adaptor.qc_distributions(n_scatter=5000)
    assert qc["scatter"]["n_shown"] == 5000
    assert len(qc["scatter"]["x"]) == 5000
    # second call returns the cached object
    assert adaptor.qc_distributions() is qc
