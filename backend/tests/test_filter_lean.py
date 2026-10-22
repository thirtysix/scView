"""Filtering's subsets (pipeline.subset_lean): what ``adata[keep].copy()`` and
``sc.pp.filter_genes`` produce, with one copy of each matrix instead of two."""

from __future__ import annotations

import tracemalloc

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
import scipy.sparse as sp

from scview.core.pipeline import PipelineParams, _filter_genes_lean, _run_filtering, subset_lean

N, M = 60, 40


def _dataset(kind: str, seed: int = 0) -> ad.AnnData:
    """Everything filtering carries: X (dense, CSR, CSC or column-major), dense, sparse and
    column-major layers, .raw, obsm arrays and a DataFrame (as vdj_chains), obsp, varm,
    varp, a categorical whose first category only the removed cells have (and its colours),
    a categorical in var, nested uns."""
    rng = np.random.default_rng(seed)
    Xd = (rng.poisson(0.5, (N, M)) * (rng.random((N, M)) < 0.6)).astype(np.float32)
    X = {"dense": Xd, "csr": sp.csr_matrix(Xd), "csc": sp.csc_matrix(Xd),
         "fortran": np.asfortranarray(Xd)}[kind]
    a = ad.AnnData(X=X)
    a.obs_names = [f"c{i}" for i in range(N)]
    a.var_names = [f"g{i}" for i in range(M)]
    a.layers["dense"] = Xd * 2
    a.layers["sparse"] = sp.csr_matrix(Xd * 3)
    a.layers["fortran"] = np.asfortranarray(Xd * 4)
    a.obs["cat"] = pd.Categorical(np.where(np.arange(N) < 5, "rare",
                                           np.where(np.arange(N) % 2, "a", "b")))
    a.uns["cat_colors"] = ["#111111", "#222222", "#333333"]
    a.obs["num"] = rng.random(N)
    a.var["vcat"] = pd.Categorical(np.where(np.arange(M) < 3, "x", "y"))
    a.obsm["X_pca"] = rng.random((N, 5))
    a.obsm["vdj_chains"] = pd.DataFrame({"n": rng.integers(0, 3, N)}, index=a.obs_names)
    a.obsp["distances"] = sp.random(N, N, density=0.1, format="csr", random_state=1)
    a.varm["PCs"] = rng.random((M, 5))
    a.varp["pairs"] = sp.random(M, M, density=0.1, format="csr", random_state=2)
    a.uns["nested"] = {"a": [1, 2], "b": np.arange(3)}
    a.raw = ad.AnnData(X=sp.csr_matrix(np.log1p(Xd)),
                       var=pd.DataFrame({"r": np.arange(M)}, index=a.var_names))
    return a


def _same_matrix(x, y, layout: bool = True) -> None:
    if sp.issparse(x) or sp.issparse(y):
        assert sp.issparse(x) and sp.issparse(y) and x.format == y.format
        assert x.dtype == y.dtype and x.shape == y.shape and (x != y).nnz == 0
        return
    x, y = np.asarray(x), np.asarray(y)
    assert x.dtype == y.dtype and np.array_equal(x, y)
    if layout:
        assert x.flags.c_contiguous == y.flags.c_contiguous


def _same_frame(x: pd.DataFrame, y: pd.DataFrame) -> None:
    pd.testing.assert_frame_equal(x, y, check_categorical=True)
    for c in x.columns:
        if isinstance(x[c].dtype, pd.CategoricalDtype):
            assert list(x[c].cat.categories) == list(y[c].cat.categories)


def _same(want: ad.AnnData, got: ad.AnnData, layout: bool = True) -> None:
    _same_matrix(want.X, got.X, layout)
    _same_frame(want.obs, got.obs)
    _same_frame(want.var, got.var)
    assert list(want.layers) == list(got.layers)
    for k in want.layers:
        _same_matrix(want.layers[k], got.layers[k], layout)
    assert list(want.obsm) == list(got.obsm)
    _same_frame(want.obsm["vdj_chains"], got.obsm["vdj_chains"])
    _same_matrix(want.obsm["X_pca"], got.obsm["X_pca"])
    _same_matrix(want.obsp["distances"], got.obsp["distances"])
    _same_matrix(want.varm["PCs"], got.varm["PCs"])
    _same_matrix(want.varp["pairs"], got.varp["pairs"])
    assert sorted(want.uns) == sorted(got.uns)
    assert list(want.uns["cat_colors"]) == list(got.uns["cat_colors"])
    assert want.uns["nested"]["a"] == got.uns["nested"]["a"]
    _same_matrix(want.raw.X, got.raw.X)
    _same_frame(want.raw.var, got.raw.var)


def _keep(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    keep = np.ones(N, dtype=bool)
    keep[:5] = False  # every "rare" cell: the category and its colour go
    keep[rng.choice(np.arange(5, N), 10, replace=False)] = False
    return keep


@pytest.mark.parametrize("kind", ["dense", "csr", "csc", "fortran"])
def test_the_cells_kept_are_what_anndata_copies(kind):
    a = _dataset(kind)
    want = a[_keep(), :].copy()
    got = subset_lean(a, obs=_keep())
    _same(want, got)
    assert list(got.obs["cat"].cat.categories) == ["a", "b"]
    assert list(got.uns["cat_colors"]) == ["#111111", "#222222"]  # categories a, b, rare
    # the dataset's copy gave its matrices up (the run replaces it)
    assert a.X is None and len(a.layers) == 0 and a.raw is None


@pytest.mark.parametrize("kind", ["dense", "csr", "csc", "fortran"])
@pytest.mark.parametrize("min_cells", [0, 3, 12])
def test_the_genes_kept_are_what_filter_genes_leaves(kind, min_cells):
    a = _dataset(kind)[_keep(), :].copy()
    want = a.copy()
    sc.pp.filter_genes(want, min_cells=min_cells)
    got = _filter_genes_lean(a.copy(), min_cells)
    # scanpy re-lays the layers out even when every gene stays; the values are the same
    _same(want, got, layout=want.n_vars < M)
    assert "n_cells" in got.var.columns
    if min_cells == 12:
        assert want.n_vars < M  # some genes went, so their layout was compared too


def test_if_anndata_fails_the_dataset_keeps_its_matrices(monkeypatch):
    a = _dataset("csr")
    X, layer, raw_X = a.X, a.layers["dense"], a.raw.X

    def boom(self, *args, **kwargs):
        raise RuntimeError("copy failed")

    monkeypatch.setattr(ad.AnnData, "copy", boom)
    with pytest.raises(RuntimeError):
        subset_lean(a, obs=_keep())
    assert a.X is X and a.layers["dense"] is layer and a.raw.X is raw_X
    assert list(a.layers) == ["dense", "sparse", "fortran"]


def test_when_every_gene_stays_no_matrix_is_copied():
    a = _dataset("dense")
    X, layer = a.X, a.layers["sparse"]
    got = subset_lean(a, var=np.ones(M, dtype=bool))
    assert got.X is X and got.layers["sparse"] is layer


def _processed(n: int = 3000, m: int = 2000, seed: int = 0) -> ad.AnnData:
    """A dataset filtered and scaled before: a dense float32 X, sparse counts, 10% of the
    cells newly called doublets."""
    rng = np.random.default_rng(seed)
    counts = sp.random(n, m, density=0.05, format="csr", random_state=seed, dtype=np.float32)
    counts.data = np.ceil(counts.data * 9)
    a = ad.AnnData(X=rng.standard_normal((n, m), dtype=np.float32))
    a.layers["counts"] = counts
    a.obs_names = [f"c{i}" for i in range(n)]
    a.var_names = [f"g{i}" for i in range(m)]
    a.obs["predicted_doublet"] = np.arange(n) % 10 == 0
    a.obsm["X_pca"] = rng.random((n, 20))
    return a


def test_refiltering_a_processed_dataset_takes_one_copy_of_x():
    # adata[keep].copy() and then filter_genes' own subset made four copies of the dense X
    # in turn while the dataset kept its own: a 6 GB backend died re-filtering the hashing
    # PBMC (1.7 GiB scaled X). Now X is cut once, and the old X goes as it is.
    a = _processed()
    x_bytes = a.X.nbytes
    params = PipelineParams(qc_threshold_mode="fixed", min_genes=0, min_cells=0, drop_doublets=True)
    tracemalloc.start()
    out, _ = _run_filtering(a, params)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert out.n_obs == 2700 and out.n_vars == 2000
    assert peak < 1.4 * x_bytes  # one 90% cut of X and its x > 0 mask (a byte per value)
