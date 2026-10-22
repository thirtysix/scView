"""Scrublet on the counts alone (core/scrublet_lean.py): the scores sc.pp.scrublet gives on
the whole dataset, bitwise, without copying the dataset."""

from __future__ import annotations

import tracemalloc

import anndata as ad
import numpy as np
import pytest
import scanpy as sc
import scipy.sparse as sp

from scview.core import scrublet_lean
from scview.core.pipeline import PipelineParams, _run_doublet_detection


def _counts(n_per: int = 120, n_genes: int = 300, seed: int = 0) -> np.ndarray:
    """Three cell populations with their own gene blocks (so there are variable genes),
    12 doublets of two populations, two near-empty cells and genes seen in under 3 cells
    (Scrublet filters both out)."""
    rng = np.random.default_rng(seed)
    parts = []
    for lo in (0, 60, 120):
        X = rng.poisson(0.4, size=(n_per, n_genes)).astype(np.float32)
        X[:, lo:lo + 60] += rng.poisson(4.0, size=(n_per, 60))
        parts.append(X)
    X = np.vstack(parts)
    X[-12:] = X[:12] + X[n_per:n_per + 12]  # doublets of populations 1 and 2
    X[:2] = 0
    X[0, 5] = X[1, 7] = 1  # one gene each: under Scrublet's 3
    X[:, -4:] = 0
    X[3, -1] = 2  # a gene in one cell
    return X


def _whole(X, **layers) -> ad.AnnData:
    """A dataset as the doublet step sees it: counts in X, plus what Scrublet copies."""
    a = ad.AnnData(X=X)
    a.obs_names = [f"C{i}" for i in range(a.n_obs)]
    a.var_names = [f"G{i}" for i in range(a.n_vars)]
    for k, v in layers.items():
        a.layers[k] = v
    a.obsm["X_other"] = np.zeros((a.n_obs, 5))
    return a


def _same(score, call, uns, ref: ad.AnnData):
    """Bitwise the same as sc.pp.scrublet wrote on ``ref``."""
    assert np.array_equal(score, ref.obs["doublet_score"].to_numpy(), equal_nan=True)
    assert [str(v) for v in call] == [str(v) for v in ref.obs["predicted_doublet"].to_numpy()]
    want = ref.uns["scrublet"]
    assert np.array_equal(uns["doublet_scores_sim"], want["doublet_scores_sim"])
    assert np.array_equal(uns["doublet_parents"], want["doublet_parents"])
    assert uns.get("threshold") == want.get("threshold")
    assert uns["parameters"] == want["parameters"]


@pytest.mark.parametrize("kind", ["dense float32", "csr float64", "dense int64", "csc float32"])
def test_scores_are_bitwise_sc_pp_scrublets_own(kind):
    X = _counts()
    M = {"dense float32": X, "csr float64": sp.csr_matrix(X.astype(np.float64)),
         "dense int64": X.astype(np.int64), "csc float32": sp.csc_matrix(X)}[kind]
    ref = _whole(M.copy(), counts=M.copy())
    sc.pp.scrublet(ref, expected_doublet_rate=0.06, use_approx_neighbors=False)
    assert ref.obs["doublet_score"].isna().sum() == 2  # the near-empty cells
    assert ref.obs["predicted_doublet"].eq(True).sum() > 0
    before = M.copy()
    score, call, uns = scrublet_lean.scrublet_counts(M, expected_doublet_rate=0.06,
                                                     use_approx_neighbors=False)
    _same(score, call, uns, ref)
    # the counts are the dataset's: never changed (Scrublet normalises its copy in place)
    after = M.toarray() if sp.issparse(M) else M
    assert np.array_equal(after, before.toarray() if sp.issparse(before) else before)


def test_a_dense_and_a_sparse_input_need_the_dense_step_to_agree():
    # Why the observed cells' matrix goes dense again: scored as plain CSR, paul15's scores
    # moved by up to 9e-4. If this starts to pass bitwise, the toarray() can go.
    X = _counts(seed=3)
    ref = _whole(X.copy())
    sc.pp.scrublet(ref, expected_doublet_rate=0.06, use_approx_neighbors=False)
    plain = _whole(sp.csr_matrix(X))
    sc.pp.scrublet(plain, expected_doublet_rate=0.06, use_approx_neighbors=False)
    lean, _, _ = scrublet_lean.scrublet_counts(X, expected_doublet_rate=0.06, use_approx_neighbors=False)
    assert np.array_equal(lean, ref.obs["doublet_score"].to_numpy(), equal_nan=True)
    assert not np.array_equal(plain.obs["doublet_score"].to_numpy(),
                              ref.obs["doublet_score"].to_numpy(), equal_nan=True)


@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32])
def test_dense_rows_to_csr_is_scipys_csr(order, dtype):
    rng = np.random.default_rng(1)
    M = (rng.poisson(0.3, size=(53, 41)) * (rng.random((53, 41)) < 0.5)).astype(dtype)
    M[7] = 0  # an empty row
    M = np.asarray(M, order=order)
    got = scrublet_lean.dense_rows_to_csr(M, rows=8)  # blocks that do not divide the rows
    want = sp.csr_matrix(M)
    assert got.dtype == M.dtype and got.has_sorted_indices
    assert np.array_equal(got.indptr, want.indptr)
    assert np.array_equal(got.indices, want.indices)
    assert np.array_equal(got.data, want.data)


def test_dense_rows_to_csr_holds_one_block_not_the_whole_matrix():
    # sp.csr_matrix(dense) takes a mask and int64 coordinates of every non-zero at once
    # (ovary's 3 GB counts layer: measured in the batch 16 report); by blocks, the result
    # and one block.
    rng = np.random.default_rng(2)
    M = (rng.random((4000, 2000)) < 0.1).astype(np.float32)  # 32 MB, 800k non-zeros
    result = M.nonzero()[0].size * (4 + 4)

    def peak(fn):
        tracemalloc.start()
        out = fn()
        _, top = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        del out
        return top

    blocks = peak(lambda: scrublet_lean.dense_rows_to_csr(M, rows=256))
    whole = peak(lambda: sp.csr_matrix(M))
    assert blocks < result + 256 * 2000 * 2 + 4 * 2**20
    assert blocks < whole / 2


def test_the_doublet_step_never_hands_scrublet_the_dataset(monkeypatch):
    # sc.pp.scrublet copies the whole AnnData it is given and then its layers again: given
    # the dataset (ovary: a dense X beside a dense counts layer) it held several copies of
    # both. It now gets the observed cells' variable genes alone.
    seen = []
    real = sc.pp.scrublet

    def spy(adata, *args, **kw):
        seen.append({"layers": list(adata.layers), "obsm": list(adata.obsm), "raw": adata.raw,
                     "shape": adata.shape, "sim": kw.get("adata_sim") is not None})
        return real(adata, *args, **kw)

    monkeypatch.setattr(sc.pp, "scrublet", spy)
    X = _counts()
    a = _whole(X.copy(), counts=X.copy())
    a.raw = a.copy()
    _run_doublet_detection(a, PipelineParams(expected_doublet_rate=0.06))
    assert len(seen) == 1
    s = seen[0]
    assert s["layers"] == [] and s["obsm"] == [] and s["raw"] is None and s["sim"]
    assert s["shape"][1] < a.n_vars  # variable genes only
    assert a.obs["doublet_score"].notna().sum() == a.n_obs - 2


def test_the_doublet_step_writes_what_sc_pp_scrublet_wrote_on_the_dataset():
    # main ran sc.pp.scrublet on the dataset itself; the step's columns and uns are its own.
    X = _counts(seed=5)
    ref = _whole(X.copy(), counts=X.copy())
    sc.pp.scrublet(ref, expected_doublet_rate=0.06, use_approx_neighbors=False)
    a = _whole(X.copy(), counts=X.copy())
    found = _run_doublet_detection(a, PipelineParams(expected_doublet_rate=0.06))
    assert found["neighbors"] == "exact" and "input" not in found
    _same(a.obs["doublet_score"].to_numpy(), a.obs["predicted_doublet"].to_numpy(),
          a.uns["scrublet"], ref)
    assert a.obs["predicted_doublet"].dtype == ref.obs["predicted_doublet"].dtype


def test_a_processed_x_is_scored_on_its_counts_layer():
    X = _counts(seed=6)
    ref = _whole(sp.csr_matrix(X))
    sc.pp.scrublet(ref, expected_doublet_rate=0.06, use_approx_neighbors=False)
    a = _whole(np.log1p(X / X.sum(1, keepdims=True).clip(1) * 1e4), counts=sp.csr_matrix(X))
    found = _run_doublet_detection(a, PipelineParams(expected_doublet_rate=0.06))
    assert found["input"] == "layers['counts']"
    _same(a.obs["doublet_score"].to_numpy(), a.obs["predicted_doublet"].to_numpy(),
          a.uns["scrublet"], ref)
