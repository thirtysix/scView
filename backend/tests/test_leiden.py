"""Leiden in a child process (core/leiden.py): same clusters as sc.tl.leiden."""

from __future__ import annotations

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


class _FailedChild:
    """A child process that exits 1 with a traceback on stderr."""

    returncode = None

    def __init__(self, *_a, **_k):
        pass

    def communicate(self, _input=None):
        self.returncode = 1
        return b"", b"Traceback ...\n  ...\nMemoryError: out of memory\n"

    def terminate(self):
        pass


def test_child_failure_is_reported(monkeypatch):
    monkeypatch.setattr(leiden.subprocess, "Popen", _FailedChild)
    with pytest.raises(leiden.LeidenError, match="exit 1.*MemoryError: out of memory"):
        leiden.labels(_adata().obsp["connectivities"], 1.0)


def test_needs_a_neighbour_graph():
    a = _adata()
    del a.obsp["connectivities"]
    with pytest.raises(ValueError, match="neighbour graph"):
        leiden.leiden(a, 1.0, "x")


# --- fast Leiden (leiden_iterations 2) -----------------------------------------


def test_fast_leiden_is_scanpys_two_passes():
    a = _adata()
    ref = a.copy()
    sc.tl.leiden(ref, resolution=0.8, key_added="scview_leiden_r0.8", n_iterations=2)
    found = _run_clustering(a, PipelineParams(clustering_resolution=0.8, leiden_iterations=2))
    pd.testing.assert_series_equal(a.obs["scview_leiden_r0.8"], ref.obs["scview_leiden_r0.8"])
    assert a.uns["scview_leiden_r0.8"] == ref.uns["scview_leiden_r0.8"]  # n_iterations 2 in params
    assert found == {"leiden_iterations": 2}


def _spy(monkeypatch) -> list[int]:
    """The ``n_iterations`` of every Leiden call from here on."""
    seen: list[int] = []
    real = leiden.membership

    def spy(adjacency, resolution, random_state=0, n_iterations=-1):
        seen.append(n_iterations)
        return real(adjacency, resolution, random_state, n_iterations)

    monkeypatch.setattr(leiden, "membership", spy)
    return seen


def test_the_pipeline_passes_the_iterations_to_the_child(monkeypatch):
    seen = _spy(monkeypatch)
    _run_clustering(_adata(), PipelineParams(leiden_iterations=2))
    _run_clustering(_adata(), PipelineParams())
    assert seen == [2, -1]


@pytest.mark.parametrize("value", [-1, 2, "2", "-1", 2.0])
def test_check_iterations_takes_the_two_settings(value):
    assert leiden.check_iterations(value) in (-1, 2)


@pytest.mark.parametrize("value", [0, 1, 3, 10, "fast", None, True, 2.5])
def test_check_iterations_refuses_anything_else(value):
    with pytest.raises(ValueError, match="Leiden iterations"):
        leiden.check_iterations(value)


async def test_a_run_request_with_another_setting_is_refused_before_loading(client):
    for path in ("assessment/run", "assessment/run-stream"):
        r = await client.post(f"/api/v1/datasets/none/{path}",
                              json={"steps": ["clustering"], "params": {"leiden_iterations": 3}})
        assert r.status_code == 422 and "Leiden iterations" in r.json()["detail"], path


def test_an_unknown_setting_is_the_clustering_steps_error():
    from scview.core.pipeline import run_pipeline

    _, res = run_pipeline(_adata(), ["clustering"], PipelineParams(leiden_iterations=5))
    assert "Unknown Leiden iterations" in res.errors["clustering"]


def test_fast_leiden_is_recorded_in_provenance_and_current():
    from scview.core import provenance
    from scview.core.pipeline import run_pipeline

    out, res = run_pipeline(_adata(), ["clustering"], PipelineParams(leiden_iterations=2))
    assert res.errors == {}
    prov = provenance.read_provenance(out)
    assert prov["history"][-1]["params"]["leiden_iterations"] == 2
    assert prov["current"]["clustering"]["leiden_iterations"] == 2
    assert leiden.recorded_iterations(out) == 2
    # Louvain has no passes to record
    out, _ = run_pipeline(_adata(), ["clustering"], PipelineParams(clustering_method="louvain"))
    assert "leiden_iterations" not in provenance.read_provenance(out)["history"][-1]["params"]


def test_recorded_iterations_reads_the_column_then_the_active_clustering():
    a = _adata()
    assert leiden.recorded_iterations(a) == -1  # never clustered: scanpy's default
    leiden.leiden(a, 1.0, "scview_leiden_r1.0", n_iterations=2)
    a.uns["scview_active_clustering"] = "scview_leiden_r1.0"
    assert leiden.recorded_iterations(a) == 2
    a.obs["cell_type"] = a.obs["scview_leiden_r1.0"]  # a column scView did not cluster
    assert leiden.recorded_iterations(a, "cell_type") == 2  # follows the active one
    leiden.leiden(a, 0.5, "other")  # a full-optimisation column of its own
    assert leiden.recorded_iterations(a, "other") == -1
    # Read back from an h5ad the params are numpy integers
    a.uns["scview_leiden_r1.0"]["params"]["n_iterations"] = np.int64(2)
    assert leiden.recorded_iterations(a) == 2


def test_rerunning_the_clustering_follows_the_recorded_setting(monkeypatch):
    """The sweep, "keep", sub-clustering and the variable-gene check's control re-cluster
    with the passes the dataset's clustering was made with."""
    from scview.core import provenance
    from scview.core import resolution_sweep as rs
    from scview.core.subcluster import subcluster

    a = _adata()
    _run_clustering(a, PipelineParams(leiden_iterations=2))
    seen = _spy(monkeypatch)
    out = rs.sweep(a, [0.4, 0.9], "leiden")
    assert out["leiden_iterations"] == 2 and seen == [2, 2]
    rs.keep(a, "leiden", 1.3)
    assert seen[-1] == 2
    assert provenance.read_provenance(a)["history"][-1]["params"]["leiden_iterations"] == 2
    # Clustered in full: the sweep follows that
    b = _adata()
    _run_clustering(b, PipelineParams())
    rs.sweep(b, [0.4], "leiden")
    assert seen[-1] == -1
    # Sub-clustering a group of the fast clustering (its own graph, from the counts)
    base = _adata()
    c = ad.AnnData(np.random.default_rng(1).poisson(2.0, (base.n_obs, 50)).astype("float32"),
                   obs=base.obs.copy(), obsm=dict(base.obsm), obsp=dict(base.obsp),
                   uns=dict(base.uns))
    c.var_names = [f"G{i}" for i in range(50)]
    _run_clustering(c, PipelineParams(leiden_iterations=2))
    subcluster(c, c.uns["scview_active_clustering"], "0", 0.5)
    assert seen[-1] == 2
    assert provenance.read_provenance(c)["history"][-1]["params"]["leiden_iterations"] == 2
