"""Harmony batch-correction step — regression test.

Guards against the harmonypy>=2.0 incompatibility where scanpy's
``harmony_integrate`` transposed Harmony's (already cells × PCs) output, yielding
a (n_pcs, n_cells) array that failed obsm validation. We call harmonypy directly
and orient to (n_cells, n_pcs); this test asserts the resulting shape.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pytest

from scview.core.pipeline import PipelineParams, _run_batch_correction, _run_pca


def _two_batch(n: int = 400, g: int = 200, seed: int = 0) -> ad.AnnData:
    """Two batches with an additive offset, so there is a batch effect to correct."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, size=(n, g)).astype("float32")
    X[: n // 2, :50] += 3.0  # batch A shifted on a block of genes
    a = ad.AnnData(np.abs(X))
    a.var_names = [f"G{i}" for i in range(g)]
    a.obs_names = [f"C{i}" for i in range(n)]
    a.obs["batch"] = ["A"] * (n // 2) + ["B"] * (n - n // 2)
    return a


def test_batch_correction_produces_correctly_shaped_harmony():
    a = _two_batch()
    _run_pca(a, PipelineParams(n_comps=20))
    assert a.obsm["X_pca"].shape == (a.n_obs, 20)

    _run_batch_correction(a, PipelineParams(batch_key="batch"))

    assert "X_pca_harmony" in a.obsm
    # the bug produced (n_pcs, n_cells); correct output is (n_cells, n_pcs)
    assert a.obsm["X_pca_harmony"].shape[0] == a.n_obs
    assert a.obsm["X_pca_harmony"].shape == a.obsm["X_pca"].shape


def test_batch_correction_requires_pca_and_key():
    a = _two_batch()
    with pytest.raises(ValueError):
        _run_batch_correction(a, PipelineParams(batch_key="batch"))  # no PCA yet
    _run_pca(a, PipelineParams(n_comps=20))
    with pytest.raises(ValueError):
        _run_batch_correction(a, PipelineParams(batch_key=""))  # no key
