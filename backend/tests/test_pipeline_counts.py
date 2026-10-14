"""Raw counts survive processing when an upload carries them only in .raw."""

from __future__ import annotations

import anndata as ad
import numpy as np
import scanpy as sc
import scipy.sparse as sp

from scview.core.pipeline import PipelineParams, looks_like_counts, run_pipeline


def _counts(n_cells: int = 120, n_genes: int = 80, seed: int = 0) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    X = sp.csr_matrix(rng.poisson(1.5, size=(n_cells, n_genes)).astype("float32"))
    a = ad.AnnData(X)
    a.var_names = [f"G{i}" for i in range(n_genes)]
    a.obs_names = [f"C{i}" for i in range(n_cells)]
    return a


def _counts_in_raw_only(lognorm_x: bool) -> tuple[ad.AnnData, np.ndarray]:
    """Counts in .raw, X either the same counts or log-normalised, no counts layer."""
    a = _counts()
    expected = a.X.toarray()
    a.raw = a.copy()
    if lognorm_x:
        sc.pp.normalize_total(a, target_sum=1e4)
        sc.pp.log1p(a)
    return a, expected


def _steps(*extra: str) -> list[str]:
    return [*extra, "hvg", "scaling", "pca"]


def test_counts_in_raw_survive_full_processing():
    a, expected = _counts_in_raw_only(lognorm_x=False)
    out, res = run_pipeline(a, _steps("normalization", "log_transform"), PipelineParams(n_comps=10, n_top_genes=40))
    assert res.errors == {}
    assert "counts" in out.layers
    assert looks_like_counts(out.layers["counts"])
    np.testing.assert_array_equal(np.asarray(out.layers["counts"].todense()), expected)
    # .raw now holds the log-normalised matrix, as before
    assert not looks_like_counts(out.raw.X)


def test_counts_in_raw_survive_scaling_of_lognorm_upload():
    a, expected = _counts_in_raw_only(lognorm_x=True)
    out, res = run_pipeline(a, _steps(), PipelineParams(n_comps=10, n_top_genes=40))
    assert res.errors == {}
    np.testing.assert_array_equal(np.asarray(out.layers["counts"].todense()), expected)


def test_counts_layer_aligned_to_x_when_raw_has_more_genes():
    a, expected = _counts_in_raw_only(lognorm_x=True)
    keep = a.var_names[::2]
    a = a[:, keep].copy()
    out, res = run_pipeline(a, _steps(), PipelineParams(n_comps=10, n_top_genes=20))
    assert res.errors == {}
    assert list(out.var_names) == list(keep)
    np.testing.assert_array_equal(np.asarray(out.layers["counts"].todense()), expected[:, ::2])


def test_plain_counts_upload_unchanged():
    a = _counts()
    expected = a.X.toarray()
    out, res = run_pipeline(a, _steps("normalization", "log_transform"), PipelineParams(n_comps=10, n_top_genes=40))
    assert res.errors == {}
    np.testing.assert_array_equal(np.asarray(out.layers["counts"].todense()), expected)
