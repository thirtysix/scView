"""Tests for the provenance / history recorder (uns['scview_provenance'])."""

from __future__ import annotations

import anndata as ad
import numpy as np

from scview.core import provenance as prov


def _adata(n_obs=10, n_var=5):
    a = ad.AnnData(X=np.zeros((n_obs, n_var), dtype="float32"))
    a.obs_names = [f"c{i}" for i in range(n_obs)]
    a.var_names = [f"g{i}" for i in range(n_var)]
    return a


def test_read_empty():
    p = prov.read_provenance(_adata())
    assert p == {"schema_version": 1, "source": {}, "history": [], "current": {}}


def test_init_source_and_no_clobber():
    a = _adata()
    prov.init_source(a, origin="ingested", original_filename="pbmc.h5", fmt="10x_h5")
    src = prov.read_provenance(a)["source"]
    assert src["origin"] == "ingested"
    assert src["format"] == "10x_h5"
    assert src["n_cells"] == 10 and src["n_genes"] == 5
    # second call must not clobber
    prov.init_source(a, origin="uploaded", original_filename="other.h5ad", fmt="h5ad")
    assert prov.read_provenance(a)["source"]["origin"] == "ingested"
    # unless overwrite
    prov.init_source(a, origin="uploaded", original_filename="o.h5ad", fmt="h5ad", overwrite=True)
    assert prov.read_provenance(a)["source"]["origin"] == "uploaded"


def test_record_steps_ordered_with_effect():
    a = _adata()
    prov.record_step(a, step="normalization", tool="sc.pp.normalize_total",
                     params={"target_sum": 1e4})
    prov.record_step(a, step="pca", tool="sc.pp.pca", params={"n_comps": 50})
    hist = prov.read_provenance(a)["history"]
    assert [h["step"] for h in hist] == ["normalization", "pca"]
    assert hist[0]["params"]["target_sum"] == 1e4
    assert hist[0]["effect"] == {"n_cells": 10, "n_genes": 5}
    assert "timestamp" in hist[0] and "scview_version" in hist[0]


def test_numpy_params_are_cleaned():
    a = _adata()
    prov.record_step(a, step="clustering", tool="sc.tl.leiden",
                     params={"resolution": np.float64(0.5), "seed": np.int64(0)})
    p = prov.read_provenance(a)["history"][0]["params"]
    assert p["resolution"] == 0.5 and p["seed"] == 0


def test_set_current():
    a = _adata()
    prov.set_current(a, normalized=True, embeddings=["X_pca", "X_umap"])
    prov.set_current(a, clustering={"method": "leiden", "resolution": 0.5, "column": "leiden"})
    cur = prov.read_provenance(a)["current"]
    assert cur["normalized"] is True
    assert cur["embeddings"] == ["X_pca", "X_umap"]
    assert cur["clustering"]["method"] == "leiden"


def test_malformed_block_is_ignored():
    a = _adata()
    a.uns[prov.UNS_KEY] = "not valid json {{"
    assert prov.read_provenance(a) == {"schema_version": 1, "source": {}, "history": [], "current": {}}


def test_h5ad_roundtrip(tmp_path):
    a = _adata()
    prov.init_source(a, origin="uploaded", original_filename="ovary.h5ad", fmt="h5ad")
    prov.record_step(a, step="qc_metrics", tool="sc.pp.calculate_qc_metrics")
    prov.set_current(a, qc=True)
    p = tmp_path / "x.h5ad"
    a.write_h5ad(p)
    b = ad.read_h5ad(p)
    pr = prov.read_provenance(b)
    assert pr["source"]["original_filename"] == "ovary.h5ad"
    assert [h["step"] for h in pr["history"]] == ["qc_metrics"]
    assert pr["current"]["qc"] is True


def test_reconcile_detects_missing():
    a = _adata()
    prov.set_current(a, embeddings=["X_umap"], clustering={"column": "leiden"}, markers_for=["ct"])
    issues = prov.reconcile(a)
    assert any("X_umap" in i for i in issues)
    assert any("leiden" in i for i in issues)
    assert any("ct" in i for i in issues)
    # add the real artifacts → no issues
    a.obsm["X_umap"] = np.zeros((10, 2), dtype="float32")
    a.obs["leiden"] = ["0"] * 10
    a.uns["rank_genes_groups__ct"] = {}
    assert prov.reconcile(a) == []
