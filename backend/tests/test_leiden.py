"""Leiden in a child process (core/leiden.py): same clusters as sc.tl.leiden."""

from __future__ import annotations

import subprocess

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scanpy as sc

from scview.core import leiden
from scview.core.pipeline import PipelineParams, _run_clustering


def _adata() -> ad.AnnData:
    rng = np.random.default_rng(0)
    pca = rng.normal(0, 1, (400, 10)).astype("float32")
    pca[:150, 0] += 3
    a = ad.AnnData(np.zeros((400, 3), dtype="float32"))
    a.obs_names = [f"c{i}" for i in range(400)]
    a.obsm["X_pca"] = pca
    sc.pp.neighbors(a, n_neighbors=15, use_rep="X_pca")
    return a


@pytest.mark.parametrize(("resolution", "seed"), [(0.3, 0), (1.0, 0), (2.0, 0), (1.0, 7)])
def test_labels_are_scanpys(resolution, seed):
    a = _adata()
    sc.tl.leiden(a, resolution=resolution, key_added="ref", random_state=seed)
    got = leiden.labels(a.obsp["connectivities"], resolution, seed)
    pd.testing.assert_series_equal(
        pd.Series(got, index=a.obs_names, name="ref"), a.obs["ref"], check_categorical=True,
    )


def test_a_quick_count_uses_fewer_passes_like_scanpys():
    a = _adata()
    sc.tl.leiden(a, resolution=1.0, key_added="ref", n_iterations=2)
    got = leiden.membership(a.obsp["connectivities"], 1.0, n_iterations=2)
    assert np.unique(got).size == a.obs["ref"].nunique()


def test_pipeline_column_and_params_are_scanpys():
    a = _adata()
    ref = a.copy()
    sc.tl.leiden(ref, resolution=0.8, key_added="scview_leiden_r0.8")
    _run_clustering(a, PipelineParams(clustering_resolution=0.8))
    pd.testing.assert_series_equal(a.obs["scview_leiden_r0.8"], ref.obs["scview_leiden_r0.8"])
    assert a.uns["scview_leiden_r0.8"] == ref.uns["scview_leiden_r0.8"]  # hvg_sensitivity reads it
    assert a.uns["scview_active_clustering"] == "scview_leiden_r0.8"


def test_runs_outside_this_process(monkeypatch):
    import leidenalg

    def in_process(*_a, **_k):
        raise AssertionError("leidenalg ran in the server process (holds the GIL)")

    monkeypatch.setattr(leidenalg, "find_partition", in_process)
    assert len(leiden.labels(_adata().obsp["connectivities"], 1.0).categories) > 1


def test_child_failure_is_reported(monkeypatch):
    def failed(*_a, **_k):
        return subprocess.CompletedProcess([], 1, b"", b"Traceback ...\n  ...\nMemoryError: out of memory\n")

    monkeypatch.setattr(leiden.subprocess, "run", failed)
    with pytest.raises(leiden.LeidenError, match="exit 1.*MemoryError: out of memory"):
        leiden.labels(_adata().obsp["connectivities"], 1.0)


def test_needs_a_neighbour_graph():
    a = _adata()
    del a.obsp["connectivities"]
    with pytest.raises(ValueError, match="neighbour graph"):
        leiden.leiden(a, 1.0, "x")
