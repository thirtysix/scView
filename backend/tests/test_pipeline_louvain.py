"""Louvain runs on python-igraph (no `louvain` package) and honours resolution."""

from __future__ import annotations

import numpy as np
import anndata as ad
import scanpy as sc

from scview.core.pipeline import _louvain_igraph


def _graph(seed: int = 0) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    centres = rng.normal(0, 6, (6, 20))
    x = np.vstack([c + rng.normal(0, 1, (80, 20)) for c in centres]).astype("float32")
    a = ad.AnnData(x)
    sc.pp.pca(a, n_comps=10)
    sc.pp.neighbors(a, n_neighbors=15)
    return a


def test_louvain_resolution_changes_cluster_count():
    # One diffuse population: well-separated blobs give the same clusters at
    # every resolution, so they cannot show whether resolution is honoured.
    a = ad.AnnData(np.random.default_rng(0).normal(0, 1, (480, 20)).astype("float32"))
    sc.pp.pca(a, n_comps=10)
    sc.pp.neighbors(a, n_neighbors=15)
    _louvain_igraph(a, 0.5, "low")
    _louvain_igraph(a, 2.0, "high")
    assert a.obs["low"].nunique() < a.obs["high"].nunique()


def test_louvain_is_deterministic_and_size_ordered():
    a = _graph()
    _louvain_igraph(a, 1.0, "first")
    _louvain_igraph(a, 1.0, "second")
    assert (a.obs["first"] == a.obs["second"]).all()
    sizes = a.obs["first"].value_counts().reindex(a.obs["first"].cat.categories).to_numpy()
    assert (np.diff(sizes) <= 0).all()
    assert list(a.obs["first"].cat.categories) == [str(i) for i in range(len(sizes))]
