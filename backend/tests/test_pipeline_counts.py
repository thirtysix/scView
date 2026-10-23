"""Raw counts survive processing when an upload carries them only in .raw."""

from __future__ import annotations

import anndata as ad
import numpy as np
import scanpy as sc
import scipy.sparse as sp

from scview.core import scaling
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


def test_scaling_a_dense_x_leaves_raw_log_normalised():
    # anndata's .raw shares a dense X, and scaling used to work on X in place: .raw
    # ended up scaled too, so markers read z-scores (paul15: ~95% NaN fold changes).
    # Now .raw is X itself (log values) and PCA z-scores a copy: neither changes.
    a = _counts()
    a.X = a.X.toarray()
    out, res = run_pipeline(a, ["normalization", "log_transform", "highly_variable_genes", "scaling",
                                "pca"], PipelineParams(n_top_genes=40, n_comps=10))
    assert res.errors == {}
    raw = out.raw.X.toarray() if sp.issparse(out.raw.X) else np.asarray(out.raw.X)
    assert raw.min() >= 0
    assert np.asarray(out.X).min() >= 0
    assert scaling.pca_matrix(out)[0].min() < 0  # what PCA read: z-scores


def test_the_matrix_pca_reads_is_dense_float32_of_the_variable_genes_alone():
    # sc.pp.scale makes a sparse matrix dense float64; on Kang that was 1.5 GB. Then
    # every gene was scaled into X as dense float32 (38 GB at 250k cells): now X keeps
    # its own form and only the variable genes are made dense, for PCA.
    for to_dense_f64 in (False, True):
        a = _counts()
        if to_dense_f64:
            a.X = a.X.toarray().astype(np.float64)
        out, res = run_pipeline(a, ["normalization", "log_transform", "highly_variable_genes", "scaling"],
                                PipelineParams(n_top_genes=40))
        assert res.errors == {}
        assert sp.issparse(out.X) != to_dense_f64  # X as it came: sparse stays sparse
        M, mask = scaling.pca_matrix(out)
        assert M.dtype == np.float32 and M.shape == (a.n_obs, 40) and mask.sum() == 40
        assert M.min() < 0
        raw = out.raw.X.toarray() if sp.issparse(out.raw.X) else np.asarray(out.raw.X)
        assert raw.min() >= 0  # .raw stays log-normalised
