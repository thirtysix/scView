"""HVG flavour + batch-aware HVG: params reach scanpy and provenance; offered flavours run."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pytest
import scanpy as sc

from scview.core import provenance
from scview.core.pipeline import HVG_FLAVORS, PipelineParams, _record_pipeline_step, _run_hvg


def _logged(n: int = 300, g: int = 200, seed: int = 0) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    a = ad.AnnData(rng.poisson(2.0, size=(n, g)).astype("float32"))
    a.var_names = [f"G{i}" for i in range(g)]
    a.obs_names = [f"C{i}" for i in range(n)]
    a.obs["batch"] = ["A"] * (n // 2) + ["B"] * (n - n // 2)
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    return a


def test_cell_ranger_with_batch_key_is_recorded_in_provenance():
    a = _logged()
    params = PipelineParams(n_top_genes=50, hvg_flavor="cell_ranger", hvg_batch_key="batch")
    _run_hvg(a, params)
    assert int(a.var["highly_variable"].sum()) > 0
    assert "highly_variable_nbatches" in a.var.columns  # batch-aware path ran

    _record_pipeline_step(a, "highly_variable_genes", params)
    step = provenance.read_provenance(a)["history"][-1]
    assert step["step"] == "highly_variable_genes"
    assert step["params"]["hvg_flavor"] == "cell_ranger"
    assert step["params"]["hvg_batch_key"] == "batch"


@pytest.mark.parametrize("flavor", HVG_FLAVORS)
def test_every_offered_hvg_flavor_runs(flavor):
    # Guards against offering a flavour whose dependency is missing (the Louvain class of bug).
    a = _with_counts()
    _run_hvg(a, PipelineParams(n_top_genes=50, hvg_flavor=flavor))
    assert int(a.var["highly_variable"].sum()) > 0


def test_unsupported_hvg_flavor_is_rejected():
    with pytest.raises(ValueError, match="Unsupported HVG flavor"):
        _run_hvg(_logged(), PipelineParams(hvg_flavor="bogus"))


def _with_counts(n: int = 300, g: int = 200, seed: int = 0) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    counts = rng.negative_binomial(2, 0.3, size=(n, g)).astype("float32")
    a = _logged(n, g, seed)
    a.X = counts.copy()
    a.layers["counts"] = counts
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    return a


def test_seurat_v3_ranks_the_counts_not_the_log_values():
    a = _with_counts()
    ret = _run_hvg(a, PipelineParams(n_top_genes=50, hvg_flavor="seurat_v3", hvg_batch_key="batch"))
    expected = sc.pp.highly_variable_genes(
        ad.AnnData(a.layers["counts"], obs=a.obs[["batch"]]), flavor="seurat_v3", n_top_genes=50,
        batch_key="batch", inplace=False,
    )
    assert np.array_equal(a.var["highly_variable"].to_numpy(), expected["highly_variable"].to_numpy())
    assert a.var["highly_variable"].dtype == bool and int(a.var["highly_variable"].sum()) == 50
    assert ret == {"counts_source": "layers['counts']"}


def test_seurat_v3_drops_another_flavours_columns():
    a = _with_counts()
    _run_hvg(a, PipelineParams(n_top_genes=50, hvg_flavor="seurat"))
    assert "dispersions_norm" in a.var.columns
    _run_hvg(a, PipelineParams(n_top_genes=50, hvg_flavor="seurat_v3"))
    assert "dispersions_norm" not in a.var.columns and "variances_norm" in a.var.columns


def test_seurat_v3_without_counts_says_so():
    with pytest.raises(ValueError, match="Seurat v3 flavour ranks raw counts"):
        _run_hvg(_logged(), PipelineParams(hvg_flavor="seurat_v3"))
