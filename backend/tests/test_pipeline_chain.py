"""Phase 6 §3: pipeline integration — clustering → markers with
scview_active_clustering auto-inject."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pytest

from scview.core.pipeline import (
    PipelineParams,
    _run_marker_genes,
    run_pipeline,
)


def _structured(n_per: int = 80, n_genes: int = 120, seed: int = 0) -> ad.AnnData:
    """Three well-separated populations, each with its own high-expression gene
    block, so Leiden reliably finds >= 2 clusters with distinct markers."""
    rng = np.random.default_rng(seed)
    blocks = [(0, 40), (40, 80), (80, 120)]
    parts = []
    for lo, hi in blocks:
        X = rng.poisson(0.2, size=(n_per, n_genes)).astype("float32")
        X[:, lo:hi] += rng.poisson(6.0, size=(n_per, hi - lo)).astype("float32")
        parts.append(X)
    X = np.vstack(parts)
    a = ad.AnnData(X)
    a.var_names = [f"G{i}" for i in range(n_genes)]
    a.obs_names = [f"C{i}" for i in range(X.shape[0])]
    return a


def _prep_chain():
    return ["normalization", "log_transform", "pca", "neighbors", "clustering"]


def test_clustering_sets_active_clustering():
    adata = _structured()
    out, res = run_pipeline(adata, _prep_chain(), PipelineParams(n_comps=30))
    assert res.errors == {}
    active = out.uns.get("scview_active_clustering")
    assert active == "scview_leiden_r0.5"
    assert active in out.obs.columns
    assert out.obs[active].nunique() >= 2


def test_markers_auto_inject_active_clustering():
    adata = _structured()
    out, res = run_pipeline(
        adata, [*_prep_chain(), "marker_genes"], PipelineParams(n_comps=30)
    )
    assert res.errors == {}
    active = out.uns["scview_active_clustering"]
    # markers were computed for the active clustering under the namespaced key
    assert f"rank_genes_groups__{active}" in out.uns
    assert "marker_genes" in res.steps_run


def test_full_chain_produces_clustering_and_markers():
    adata = _structured()
    out, res = run_pipeline(
        adata,
        ["normalization", "log_transform", "pca", "neighbors", "clustering",
         "embeddings", "marker_genes"],
        PipelineParams(n_comps=30),
    )
    assert res.errors == {}
    assert "X_umap" in out.obsm  # embeddings ran
    active = out.uns["scview_active_clustering"]
    assert f"rank_genes_groups__{active}" in out.uns


def test_marker_genes_without_clustering_raises():
    """Edge case: no clustering column anywhere → clear error."""
    adata = _structured()
    with pytest.raises(ValueError, match="clustering"):
        _run_marker_genes(adata, PipelineParams())


def test_marker_genes_explicit_columns():
    """marker_columns param computes markers for a pre-existing categorical."""
    adata = _structured()
    adata.obs["cell_type"] = (["A"] * 80 + ["B"] * 80 + ["C"] * 80)
    _run_marker_genes(adata, PipelineParams(marker_columns=["cell_type"]))
    assert "rank_genes_groups__cell_type" in adata.uns
