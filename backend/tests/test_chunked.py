"""Chunked (out-of-core) QC..PCA against the in-memory pipeline (core/chunked.py).

Every expected value is computed in memory, by the pipeline's own step or scanpy, on the
same file read with ``ad.read_h5ad``; PCA is also checked against an exact float64 SVD of
the in-memory scaled matrix, since scanpy's float32 ARPACK result is itself only good to
float32 precision. Fixtures are small but read in several blocks with a ragged last one,
as sparse (gzip) and dense (lzf) files, and are built so the paths under test change the
answer: batches that disagree, rare genes that hit the scaling clip, explicit zeros.

Real data: set SCVIEW_TESTDATA to a directory holding kang2018_ifnb_pbmc.h5ad and/or
paul15.h5ad (sparse with counts in .raw / dense counts in X) to run the same checks there.
"""

from __future__ import annotations

import os
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
import scipy.sparse as sp
from sklearn.utils.extmath import svd_flip

from scview.core import chunked, normalization
from scview.core.pipeline import (
    PipelineParams,
    _doublet_matrix,
    _run_hvg,
    _run_log_transform,
    _run_normalization,
    _run_qc_metrics,
)

ROWS = 700  # 2,300 cells -> blocks of 700, 700, 700, 200
N_CELLS, N_GENES = 2300, 420
QC_OBS = ["n_genes_by_counts", "total_counts", "total_counts_mt", "pct_counts_mt"]
QC_VAR = ["mt", "n_cells_by_counts", "mean_counts", "pct_dropout_by_counts", "total_counts"]


def _counts(seed: int = 0, n: int = N_CELLS, g: int = N_GENES, zero_cell: bool = False):
    """NB counts: four cell groups with their own genes, a batch effect on 30 genes in one
    of three batches, mito genes high in some cells, three genes never seen and five rare
    genes (a few cells each, far beyond 10 sd once scaled)."""
    rng = np.random.default_rng(seed)
    names = ([f"MT-{i}" for i in range(8)] + ["mt-Co1", "mt-Nd1"]
             + [f"G{i}" for i in range(g - 10)])
    group = rng.integers(0, 4, n)
    batch = rng.choice(["b1", "b2", "b3"], n, p=[0.5, 0.3, 0.2])
    size = rng.lognormal(0.0, 0.6, n)
    base = rng.gamma(0.5, 1.0, g)
    mu = size[:, None] * base[None, :] * 2.0
    for k in range(4):
        mu[group == k, 20 + 25 * k: 45 + 25 * k] *= 6.0
    mu[batch == "b2", 150:180] *= 4.0
    mu[rng.random(n) < 0.1, :10] *= 8.0
    X = rng.negative_binomial(2, 2.0 / (2.0 + mu)).astype(np.float32)
    X[:, 200:203] = 0
    for j in range(g - 5, g):
        X[:, j] = 0
        X[rng.choice(n, 4, replace=False), j] = rng.integers(5, 40, 4)
    if zero_cell:
        X[3] = 0
    obs = pd.DataFrame({"batch": pd.Categorical(batch), "group": group.astype(str)},
                       index=[f"c{i}" for i in range(n)])
    return X, obs, pd.Index(names)


def _write(path: Path, X, obs, var_names, *, dense: bool, compression: str, layers=None,
           raw: ad.AnnData | None = None, explicit_zeros: bool = False) -> Path:
    M = np.asarray(X) if dense else sp.csr_matrix(X)
    if explicit_zeros and not dense:
        M.data[::97] = 0  # stored zeros: QC must not count them as detected
    a = ad.AnnData(M, obs=obs.copy(), var=pd.DataFrame(index=var_names))
    for k, v in (layers or {}).items():
        a.layers[k] = v
    if raw is not None:
        a.raw = raw
    a.write_h5ad(path, compression=compression)
    return path


@pytest.fixture(params=["sparse", "dense"], scope="module")
def counts_file(request, tmp_path_factory) -> Path:
    X, obs, names = _counts()
    dense = request.param == "dense"
    path = tmp_path_factory.mktemp("chunked") / f"{request.param}.h5ad"
    return _write(path, X, obs, names, dense=dense, compression="lzf" if dense else "gzip")


@pytest.fixture(scope="module")
def processed(counts_file) -> tuple[chunked.LazyCounts, ad.AnnData, object]:
    """(lazy counts, in-memory adata after qc/normalization/log, lazy log-normalised X)."""
    m = chunked.open_counts(counts_file, rows=ROWS)
    a = ad.read_h5ad(counts_file)
    p = PipelineParams()
    _run_qc_metrics(a, p)
    _run_normalization(a, p)
    _run_log_transform(a, p)
    return m, a, chunked.log_normalize(m.X, p.target_sum)


def _dense(M) -> np.ndarray:
    return M.toarray() if sp.issparse(M) else np.asarray(M)


def _assert_same_matrix(got, want) -> None:
    """Bitwise: same dtype, same values (sparse: same stored entries)."""
    assert got.dtype == want.dtype
    if sp.issparse(want):
        got, want = sp.csr_matrix(got), sp.csr_matrix(want)
        assert np.array_equal(got.indptr, want.indptr)
        assert np.array_equal(got.indices, want.indices)
        assert np.array_equal(got.data, want.data)
    else:
        assert np.array_equal(np.asarray(got), np.asarray(want))


# ---------------------------------------------------------------------------
# Opening, blocks, scheduling
# ---------------------------------------------------------------------------

def test_open_reads_ragged_blocks(counts_file):
    m = chunked.open_counts(counts_file, rows=ROWS)
    assert m.X.chunks[0] == (700, 700, 700, 200)
    assert m.source == "X"
    assert m.dense == ("dense" in counts_file.name)
    _assert_same_matrix(m.X.compute(), ad.read_h5ad(counts_file).X)


def test_rows_per_block_follow_memory_target():
    sparse_rows = chunked.rows_per_block(30_000, nnz_per_row=2_000, chunk_bytes=256 * 2**20)
    dense_rows = chunked.rows_per_block(30_000, dense=True, chunk_bytes=256 * 2**20)
    assert dense_rows < sparse_rows
    one = chunked.block_bytes(1, 30_000, nnz_per_row=2_000)
    assert abs(sparse_rows * one - 256 * 2**20) <= one
    assert chunked.rows_per_block(10, chunk_bytes=1) == chunked.MIN_ROWS
    assert chunked.step_bytes(1000, 30_000, nnz_per_row=2_000, workers=3) == 3 * \
        chunked.block_bytes(1000, 30_000, nnz_per_row=2_000)


def test_workers_follow_cpu_quota(monkeypatch):
    import dask

    monkeypatch.setattr(chunked, "_memory_limit", lambda: None)
    monkeypatch.setattr(chunked, "cpu_quota", lambda: 2.5)
    assert chunked.n_workers() == 2
    monkeypatch.setattr(chunked, "cpu_quota", lambda: 16.0)
    assert chunked.n_workers() == chunked.MAX_WORKERS
    # a 4 GB limit: a quarter holds 4 blocks of 256 MiB, whatever the CPUs
    monkeypatch.setattr(chunked, "_memory_limit", lambda: 4 * 2**30)
    assert chunked.n_workers() == 4
    assert chunked.step_bytes(chunked.rows_per_block(28_000, nnz_per_row=2_000), 28_000,
                              nnz_per_row=2_000) <= 1.1 * 2**30
    monkeypatch.setattr(chunked, "_memory_limit", lambda: 2**29)
    assert chunked.n_workers() == 1
    monkeypatch.setattr(chunked, "_memory_limit", lambda: None)
    monkeypatch.setattr(chunked, "cpu_quota", lambda: 0.5)
    assert chunked.n_workers() == 1
    with chunked.scheduler(3) as w:
        assert w == 3
        assert dask.config.get("scheduler") == "threads"
        assert dask.config.get("num_workers") == 3


def test_results_do_not_depend_on_worker_count(processed, monkeypatch):
    m, a, L = processed
    hvg = a.var_names.isin(a.var_names[:200])
    out = {}
    for w in (1, 3):
        monkeypatch.setattr(chunked, "n_workers", lambda w=w: w)
        obs, var = chunked.qc_metrics(m)
        h = chunked.highly_variable_genes(L, m.var_names, flavor="seurat", n_top_genes=100)
        out[w] = (obs, var, h, chunked.pca_scaled(L, hvg, n_comps=10))
    pd.testing.assert_frame_equal(out[1][0], out[3][0], check_exact=True)
    pd.testing.assert_frame_equal(out[1][1], out[3][1], check_exact=True)
    pd.testing.assert_frame_equal(out[1][2], out[3][2], check_exact=True)
    assert np.array_equal(out[1][3]["X_pca"], out[3][3]["X_pca"])


def test_counts_source_follows_find_counts(tmp_path):
    X, obs, names = _counts(n=900)
    lognorm = normalization.shifted_log(sp.csr_matrix(X)).astype(np.float32)
    # X processed, counts in a layer
    p1 = _write(tmp_path / "layer.h5ad", lognorm, obs, names, dense=False, compression="gzip",
                layers={"counts": sp.csr_matrix(X)})
    # X processed, counts only in .raw, which has extra genes in another order
    extra = pd.Index([f"R{i}" for i in range(15)])
    raw_names = extra.append(names)[::-1]
    rng = np.random.default_rng(1)
    raw_X = np.hstack([rng.poisson(1.0, (900, 15)).astype(np.float32), X])[:, ::-1]
    raw = ad.AnnData(sp.csr_matrix(raw_X), var=pd.DataFrame(index=raw_names))
    p2 = _write(tmp_path / "raw.h5ad", lognorm, obs, names, dense=False, compression="gzip",
                raw=raw)
    # nothing holds counts
    p3 = _write(tmp_path / "none.h5ad", lognorm, obs, names, dense=False, compression="gzip")

    a1, a2 = ad.read_h5ad(p1), ad.read_h5ad(p2)
    m1 = chunked.open_counts(p1, rows=250)
    want, where = normalization.find_counts(a1)
    assert (m1.source, where) == ("layers['counts']", "layers['counts']")
    _assert_same_matrix(m1.X.compute(), want)

    m2 = chunked.open_counts(p2, rows=250)
    want, where = normalization.find_counts(a2)
    assert m2.source == where == ".raw"
    assert list(m2.var_names) == list(a2.var_names)
    assert np.array_equal(_dense(m2.X.compute()), _dense(want))

    all2 = chunked.open_counts(p2, rule="all_genes", rows=250)
    want, genes, where = normalization.counts_all_genes(a2)
    assert all2.source == where == ".raw"
    assert list(all2.var_names) == list(genes)
    _assert_same_matrix(all2.X.compute(), want)

    for p, a in ((p1, a1), (p2, a2)):
        d = chunked.open_counts(p, rule="doublets", rows=250)
        want, where = _doublet_matrix(a)
        assert d.source == where
        _assert_same_matrix(d.X.compute(), want)

    with pytest.raises(ValueError, match="No raw counts"):
        chunked.open_counts(p3, rows=250)


def test_old_encoding_is_refused(tmp_path):
    p = tmp_path / "old.h5ad"
    M = sp.csr_matrix(np.eye(5, dtype=np.float32))
    with h5py.File(p, "w") as f:
        g = f.create_group("X")
        g.attrs["h5sparse_format"] = "csr"
        for k in ("data", "indices", "indptr"):
            g[k] = getattr(M, k)
    with pytest.raises(ValueError, match="anndata 0.7 or later"):
        chunked.read_lazy(p, "X", chunk_rows=2)


# ---------------------------------------------------------------------------
# QC metrics and filtering helpers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dense", [False, True])
def test_qc_metrics_equal_pipeline_step(tmp_path, dense):
    X, obs, names = _counts(zero_cell=True)
    p = _write(tmp_path / "qc.h5ad", X, obs, names, dense=dense, compression="gzip",
               explicit_zeros=True)
    with h5py.File(p, "r") as f:
        assert dense or np.any(f["X/data"][:] == 0)  # the explicit zeros are stored
    a = ad.read_h5ad(p)
    _run_qc_metrics(a, PipelineParams())
    got_obs, got_var = chunked.qc_metrics(chunked.open_counts(p, rows=ROWS))
    pd.testing.assert_frame_equal(got_obs, a.obs[QC_OBS], check_exact=True)
    pd.testing.assert_frame_equal(got_var, a.var[QC_VAR], check_exact=True)
    assert got_var["mt"].sum() == 10


def test_qc_n_genes_keeps_stored_index_dtype(tmp_path):
    X, obs, names = _counts(n=600)
    M = sp.csr_matrix(X)
    M.indptr, M.indices = M.indptr.astype(np.int64), M.indices.astype(np.int64)
    p = tmp_path / "i64.h5ad"
    ad.AnnData(M, obs=obs, var=pd.DataFrame(index=names)).write_h5ad(p)
    a = ad.read_h5ad(p)
    _run_qc_metrics(a, PipelineParams())
    got, _ = chunked.qc_metrics(chunked.open_counts(p, rows=250))
    assert a.obs["n_genes_by_counts"].dtype == np.int64
    pd.testing.assert_frame_equal(got, a.obs[QC_OBS], check_exact=True)


def test_gene_detection_and_subset_equal_in_memory(processed):
    m, a, _ = processed
    counts = ad.read_h5ad(m.path).X
    _, number = sc.pp.filter_genes(ad.AnnData(counts.copy()), min_cells=3, inplace=False)
    det = chunked.gene_detection(m.X)
    assert det.dtype == np.int64
    assert np.array_equal(det, number)

    rng = np.random.default_rng(3)
    cells = rng.random(m.shape[0]) < 0.6
    cells[700:1400] = False  # the second block loses every cell: dropped
    genes = det >= 3
    sub = m.subset(obs=cells, var=genes)
    assert sum(sub.X.chunks[0]) == cells.sum()
    assert len(sub.X.chunks[0]) == 3 and all(c > 0 for c in sub.X.chunks[0])
    want = counts[np.flatnonzero(cells)][:, np.flatnonzero(genes)]
    assert np.array_equal(_dense(sub.X.compute()), _dense(want))
    assert list(sub.obs_names) == list(m.obs_names[cells])
    assert list(sub.var_names) == list(m.var_names[genes])


def test_subset_drops_unused_categories(processed):
    m, _, _ = processed
    keep = (m.obs["batch"] != "b3").to_numpy()
    assert list(m.subset(obs=keep).obs["batch"].cat.categories) == ["b1", "b2"]


# ---------------------------------------------------------------------------
# Normalisation and log
# ---------------------------------------------------------------------------

def test_normalize_and_log_equal_pipeline_bitwise(processed):
    m, a, L = processed
    _assert_same_matrix(L.compute(), a.X)
    step = chunked.log1p(chunked.normalize_total(m.X, 1e4)).compute()
    _assert_same_matrix(step, a.X)


@pytest.mark.parametrize("dense", [False, True])
def test_normalize_median_target_equals_scanpy(tmp_path, dense):
    """target_sum=None: scanpy takes the median over every cell for a CSR matrix and over
    cells with counts for a dense one; a fifth of the cells have none, so the two differ."""
    X, obs, names = _counts()
    X[::5] = 0
    p = _write(tmp_path / "zeros.h5ad", X, obs, names, dense=dense, compression="gzip")
    m = chunked.open_counts(p, rows=ROWS)
    want = sc.pp.normalize_total(ad.read_h5ad(p), target_sum=None, inplace=False)["X"]
    _assert_same_matrix(chunked.normalize_total(m.X, None).compute(), want)
    totals = X.sum(axis=1)
    assert np.median(totals) != np.median(totals[totals > 0])


# ---------------------------------------------------------------------------
# Highly variable genes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("flavor", ["seurat", "cell_ranger", "seurat_v3"])
@pytest.mark.parametrize("batch_key", ["", "batch"])
def test_hvg_equal_pipeline_step(processed, flavor, batch_key):
    m, a, L = processed
    a = a.copy()
    n_top = 120
    _run_hvg(a, PipelineParams(hvg_flavor=flavor, hvg_batch_key=batch_key, n_top_genes=n_top))
    batch = m.obs[batch_key] if batch_key else None
    X = m.X if flavor == "seurat_v3" else L
    got = chunked.highly_variable_genes(X, m.var_names, flavor=flavor,
                                        n_top_genes=min(n_top, m.shape[1]), batch=batch)
    want = a.var[got.columns]
    assert list(got.index) == list(want.index)
    assert got["highly_variable"].sum() == n_top
    assert got["highly_variable"].equals(want["highly_variable"])
    for c in got.columns:
        assert got[c].dtype == want[c].dtype, c
        if got[c].dtype.kind in "fc":
            np.testing.assert_allclose(got[c], want[c], rtol=1e-10, atol=1e-12, err_msg=c)
        else:
            assert got[c].equals(want[c]), c
    if batch_key:  # the fixture's batches disagree, so the batched path decides some genes
        nb = got["highly_variable_nbatches"][got["highly_variable"]]
        assert nb.max() == 3 and (nb < 3).any()


@pytest.mark.parametrize("batch_key", ["", "batch"])
def test_pearson_hvg_equal_pipeline_step(processed, batch_key):
    m, a, _ = processed
    a = a.copy()
    p = PipelineParams(normalization_method="pearson_residuals", hvg_batch_key=batch_key,
                       n_top_genes=120)
    _run_hvg(a, p)
    got = chunked.pearson_hvg(m.X, m.var_names, n_top_genes=min(120, m.shape[1]),
                              batch=m.obs[batch_key] if batch_key else None)
    want = a.var[got.columns]
    assert got["highly_variable"].equals(want["highly_variable"])
    for c in got.columns:
        assert got[c].dtype == want[c].dtype, c
        if got[c].dtype.kind == "f":
            np.testing.assert_allclose(got[c], want[c], rtol=1e-10, atol=1e-12, err_msg=c)
        else:
            assert got[c].equals(want[c]), c


@pytest.mark.parametrize("dense", [False, True])
def test_integer_counts_equal_pipeline(tmp_path, dense):
    """int32 counts: QC sums as int64, normalisation to float32, the count-based HVG
    flavours. The sparse file has counts above 46,340, whose square overflows int32:
    scanpy's CSR kernel squares into int64 and so does this module (float64). Its dense
    path squares in int32 and overflows (test_dense_int_squares_do_not_overflow), so the
    dense file stays below that."""
    X, obs, names = _counts(n=1500)
    if not dense:
        X[:40, 30] = 60_000
    p = _write(tmp_path / "int.h5ad", X.astype(np.int32), obs, names, dense=dense,
               compression="gzip")
    m = chunked.open_counts(p, rows=400)
    assert m.X.dtype == np.int32
    a = ad.read_h5ad(p)
    pp = PipelineParams(n_top_genes=100)
    _run_qc_metrics(a, pp)
    got_obs, got_var = chunked.qc_metrics(m)
    pd.testing.assert_frame_equal(got_obs, a.obs[QC_OBS], check_exact=True)
    pd.testing.assert_frame_equal(got_var, a.var[QC_VAR], check_exact=True)
    _run_normalization(a, pp)
    _run_log_transform(a, pp)
    _assert_same_matrix(chunked.log_normalize(m.X).compute(), a.X)
    for method, flavor in (("log", "seurat_v3"), ("pearson_residuals", "seurat")):
        b = a.copy()
        _run_hvg(b, PipelineParams(n_top_genes=100, hvg_flavor=flavor,
                                   normalization_method=method))
        if method == "log":
            got = chunked.highly_variable_genes(m.X, m.var_names, flavor=flavor, n_top_genes=100)
        else:
            got = chunked.pearson_hvg(m.X, m.var_names, n_top_genes=100)
        assert got["highly_variable"].equals(b.var["highly_variable"]), method
        np.testing.assert_allclose(got["variances"], b.var["variances"], rtol=1e-12)


def test_dense_int_squares_do_not_overflow(tmp_path):
    """A dense int32 count of 60,000: the chunked variance is the exact one. (scanpy's
    ``_get_mean_var`` squares a dense int32 matrix in int32 and returns a negative
    variance here, so the in-memory seurat_v3 / Pearson steps are wrong on such data.)"""
    X, obs, names = _counts(n=1500)
    X[:40, 30] = 60_000
    p = _write(tmp_path / "int.h5ad", X.astype(np.int32), obs, names, dense=True,
               compression="gzip")
    got = chunked.highly_variable_genes(chunked.open_counts(p, rows=400).X, names,
                                        flavor="seurat_v3", n_top_genes=100)
    exact = X.astype(np.float64)
    np.testing.assert_allclose(got["variances"], exact.var(axis=0, ddof=1), rtol=1e-12)
    np.testing.assert_allclose(got["means"], exact.mean(axis=0), rtol=1e-12)


def test_hvg_rejects_unknown_flavor(processed):
    m, _, L = processed
    with pytest.raises(ValueError, match="Unsupported HVG flavor"):
        chunked.highly_variable_genes(L, m.var_names, flavor="nope")


# ---------------------------------------------------------------------------
# Scaling + PCA
# ---------------------------------------------------------------------------

def _scanpy_pca(M: np.ndarray, n_comps: int, scale: bool) -> ad.AnnData:
    """The reference: copy of the variable-gene columns (float32), sc.pp.scale(max_value=10)
    (scaled path), sc.tl.pca(svd_solver='arpack'). The copy keeps var['highly_variable'],
    so scanpy stores the loadings as it does with a mask (float64)."""
    sub = ad.AnnData(np.array(M, dtype=np.float32))
    sub.var["highly_variable"] = True
    if scale:
        sc.pp.scale(sub, max_value=10)
    sc.tl.pca(sub, n_comps=n_comps, svd_solver="arpack")
    return sub


def _exact_pca(Z: np.ndarray, n_comps: int):
    """Exact PCA of the reference's own input matrix in float64: SVD, sklearn's signs."""
    Z = Z.astype(np.float64)
    C = Z - Z.mean(axis=0)
    U, S, Vt = np.linalg.svd(C, full_matrices=False)
    U, Vt = svd_flip(U, Vt, u_based_decision=False)
    n = Z.shape[0]
    var = S**2 / (n - 1)
    return (U * S)[:, :n_comps], Vt[:n_comps].T, var[:n_comps], var[:n_comps] / var.sum()


def _check_pca(got: dict, ref: ad.AnnData, Z: np.ndarray, hvg: np.ndarray, n_comps: int):
    k = got["n_comps"]
    assert k == n_comps
    assert got["X_pca"].dtype == np.float32 and got["X_pca"].shape == (Z.shape[0], k)
    assert got["PCs"].dtype == ref.varm["PCs"].dtype == np.float64
    assert got["PCs"].shape == (len(hvg), k) and not got["PCs"][~hvg].any()
    assert got["variance"].dtype == ref.uns["pca"]["variance"].dtype
    assert got["variance_ratio"].dtype == ref.uns["pca"]["variance_ratio"].dtype
    # Exact float64 PCA of the in-memory input: equal to float32 rounding of the output.
    xe, ve, vare, ratioe = _exact_pca(Z, k)
    scale = np.abs(xe).max(axis=0)
    np.testing.assert_allclose(got["X_pca"] / scale, xe / scale, atol=2e-6)
    np.testing.assert_allclose(got["PCs"][hvg], ve, atol=2e-6)
    np.testing.assert_allclose(got["variance"], vare, rtol=2e-6)
    np.testing.assert_allclose(got["variance_ratio"], ratioe, rtol=2e-6)
    # scanpy's float32 ARPACK, against the same exact result: the chunked result is at
    # least as close on every component (its own error is the ceiling for agreement).
    xr = ref.obsm["X_pca"]
    err_ref = np.abs(xr - xe).max(axis=0) / scale
    err_got = np.abs(got["X_pca"] - xe).max(axis=0) / scale
    assert np.all(err_got <= np.maximum(err_ref, 2e-6))
    np.testing.assert_allclose(got["variance"], ref.uns["pca"]["variance"], rtol=1e-4)


def test_pca_scaled_equals_scale_then_arpack(processed):
    m, a, L = processed
    hvg = sc.pp.highly_variable_genes(a, n_top_genes=150, flavor="seurat",
                                      inplace=False)["highly_variable"].to_numpy()
    got = chunked.pca_scaled(L, hvg, n_comps=30)
    ref = _scanpy_pca(_dense(a.X[:, hvg]), 30, scale=True)
    Z = ref.X
    assert Z.max() == 10.0  # rare genes hit the clip: the PCA must re-centre after it
    assert abs(Z.mean(axis=0)).max() > 1e-3
    _check_pca(got, ref, Z, hvg, 30)


def test_pca_scaled_with_given_statistics(processed):
    m, a, L = processed
    hvg = sc.pp.highly_variable_genes(a, n_top_genes=150, flavor="seurat",
                                      inplace=False)["highly_variable"].to_numpy()
    cols = np.flatnonzero(hvg)
    stats = chunked.scaled_stats(L, cols)
    want = _dense(a.X[:, hvg]).astype(np.float32)
    np.testing.assert_allclose(stats[0], want.mean(axis=0, dtype=np.float64), rtol=1e-12)
    got = chunked.pca_scaled(L, hvg, n_comps=20, stats=stats)
    base = chunked.pca_scaled(L, hvg, n_comps=20)
    np.testing.assert_array_equal(got["X_pca"], base["X_pca"])
    # mean 0, sd 1, no clip: PCA of the values as they are (sc.tl.pca without scaling)
    raw = chunked.pca_scaled(L, hvg, n_comps=20, max_value=None,
                             stats=(np.zeros(cols.size), np.ones(cols.size)))
    _check_pca(raw, _scanpy_pca(want, 20, scale=False), want, hvg, 20)


def test_block_hook_sees_every_batch_and_can_stop_a_pass(processed):
    m, _, _ = processed
    seen = []
    with chunked.on_blocks(lambda done, total: seen.append((done, total))):
        chunked.gene_detection(m.X)
    nb = m.X.numblocks[0]
    assert seen[-1] == (nb, nb) and seen[0] == (0, nb)

    class Stop(Exception):
        pass

    def stop(done, total):
        if done:
            raise Stop

    with chunked.on_blocks(stop), pytest.raises(Stop):
        chunked.gene_detection(m.X)
    chunked.gene_detection(m.X)  # the hook ends with its block


def test_csr_appender_writes_blocks_and_an_empty_matrix(tmp_path):
    rng = np.random.default_rng(1)
    blocks = [sp.random(n, 30, density=0.2, format="csr", dtype=np.float32, random_state=rng)
              for n in (7, 0, 5)]
    with h5py.File(tmp_path / "m.h5", "w") as f:
        out = chunked.CsrAppender(f, "M")
        for b in blocks:
            out.add(b)
        out.finish(30, np.float32)
        empty = chunked.CsrAppender(f, "E")
        empty.finish(30, np.float32)
    with h5py.File(tmp_path / "m.h5", "r") as f:
        got = ad.io.read_elem(f["M"])
        assert ad.io.read_elem(f["E"]).shape == (0, 30)
    np.testing.assert_array_equal(got.toarray(), sp.vstack(blocks).toarray())


def test_pca_scaled_equals_todays_scale_all_then_mask(processed):
    """Scaling every gene and masking gives the same PCA (scaling is per gene)."""
    m, a, L = processed
    a = a.copy()
    hvg = sc.pp.highly_variable_genes(a, n_top_genes=150, flavor="seurat",
                                      inplace=False)["highly_variable"].to_numpy()
    a.var["highly_variable"] = hvg
    X = a.X.astype(np.float32)
    a.X = _dense(X)
    sc.pp.scale(a, max_value=10)
    sc.tl.pca(a, n_comps=20, svd_solver="arpack")
    got = chunked.pca_scaled(L, hvg, n_comps=20)
    xe, *_ = _exact_pca(a.X[:, hvg], 20)
    scale = np.abs(xe).max(axis=0)
    np.testing.assert_allclose(got["X_pca"] / scale, xe / scale, atol=2e-6)
    assert got["params"] == {k: a.uns["pca"]["params"][k] for k in got["params"]}


def test_pca_pearson_equals_residuals_then_arpack(processed):
    m, a, _ = processed
    counts = ad.read_h5ad(m.path).X
    hvg = np.zeros(m.shape[1], dtype=bool)
    hvg[np.argsort(-normalization.residual_variance(counts))[:150]] = True
    R = normalization.pearson_residuals(counts, clip=normalization.default_clip(m.shape[0]),
                                        columns=np.flatnonzero(hvg))
    got = chunked.pca_pearson(m.X, hvg, n_comps=30)
    ref = _scanpy_pca(R, 30, scale=False)
    _check_pca(got, ref, R, hvg, 30)


def test_pca_scaled_float64_counts(tmp_path):
    """float64 counts give float64 log values; the reference scales a float32 copy."""
    X, obs, names = _counts(n=1200)
    p = _write(tmp_path / "f64.h5ad", X.astype(np.float64), obs, names, dense=False,
               compression="gzip")
    a = ad.read_h5ad(p)
    _run_normalization(a, PipelineParams())
    _run_log_transform(a, PipelineParams())
    assert a.X.dtype == np.float64
    L = chunked.log_normalize(chunked.open_counts(p, rows=300).X)
    _assert_same_matrix(L.compute(), a.X)
    hvg = np.zeros(len(names), dtype=bool)
    hvg[10:190] = True
    ref = _scanpy_pca(_dense(a.X[:, hvg]), 20, scale=True)
    _check_pca(chunked.pca_scaled(L, hvg, n_comps=20), ref, ref.X, hvg, 20)


def test_pca_n_comps_capped(processed):
    m, a, L = processed
    hvg = np.zeros(m.shape[1], dtype=bool)
    hvg[20:32] = True
    assert chunked.pca_scaled(L, hvg, n_comps=50)["n_comps"] == 11


# ---------------------------------------------------------------------------
# Streaming writers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("compression", ["gzip", "lzf", None])
def test_write_h5ad_opens_backed(processed, tmp_path, compression):
    m, a, L = processed
    obs, var = chunked.qc_metrics(m)
    obs = m.obs.join(obs)
    rng = np.random.default_rng(0)
    obsm = {"X_pca": rng.standard_normal((m.shape[0], 5)).astype(np.float32)}
    uns = {"log1p": {"base": None}, "note": "chunked"}
    out = tmp_path / f"w_{compression}.h5ad"
    chunked.write_h5ad(out, L, obs=obs, var=var, obsm=obsm, uns=uns, compression=compression)
    assert not list(tmp_path.glob(".*.part"))
    b = ad.read_h5ad(out, backed="r")
    assert b.shape == m.shape
    assert np.array_equal(_dense(b.X[100:900]), _dense(a.X[100:900]))
    b.file.close()
    c = ad.read_h5ad(out)
    assert np.array_equal(_dense(c.X), _dense(a.X)) and c.X.dtype == a.X.dtype
    assert sp.issparse(c.X) == sp.issparse(a.X)
    pd.testing.assert_frame_equal(c.obs, obs.assign(group=obs["group"].astype("category")))
    pd.testing.assert_frame_equal(c.var, var)
    assert np.array_equal(c.obsm["X_pca"], obsm["X_pca"])
    assert c.uns["note"] == "chunked"
    with h5py.File(out, "r") as f:
        ds = f["X/data"] if isinstance(f["X"], h5py.Group) else f["X"]
        assert ds.compression == compression


def test_write_counts_sidecar_round_trip(processed, tmp_path):
    m, _, _ = processed
    keep = np.ones(m.shape[0], dtype=bool)
    keep[::3] = False
    sub = m.subset(obs=keep)
    out = tmp_path / "x.counts.h5"
    chunked.write_counts(out, sub.X, compression="lzf")
    back = chunked.read_lazy(out, "counts", chunk_rows=500)
    assert np.array_equal(_dense(back.compute()), _dense(sub.X.compute()))


def test_write_h5ad_checks_shape(processed, tmp_path):
    m, _, L = processed
    with pytest.raises(ValueError, match="obs x var"):
        chunked.write_h5ad(tmp_path / "bad.h5ad", L, obs=m.obs.iloc[:10], var=m.var)


# ---------------------------------------------------------------------------
# Real data (opt-in: SCVIEW_TESTDATA)
# ---------------------------------------------------------------------------

TESTDATA = Path(os.environ.get("SCVIEW_TESTDATA", "/nonexistent"))


def _modern(src: Path, tmp_path: Path) -> Path:
    """The file as the app stores it (ingestion re-writes every upload with anndata)."""
    with h5py.File(src, "r") as f:
        if f["X"].attrs.get("encoding-type"):
            return src
    out = tmp_path / src.name
    ad.read_h5ad(src).write_h5ad(out, compression="gzip")
    return out


_MEMORY_RUN = """
import json, resource, sys
from scview.core import chunked
chunked.n_workers = lambda: 2
m = chunked.open_counts(sys.argv[1])
obs, _ = chunked.qc_metrics(m)
cells = m.subset(obs=((obs["n_genes_by_counts"] >= 200) & (obs["pct_counts_mt"] < 20)).to_numpy())
f = cells.subset(var=chunked.gene_detection(cells.X) >= 3)
L = chunked.log_normalize(f.X)
hv = chunked.highly_variable_genes(L, f.var_names, flavor="seurat", n_top_genes=2000)
pca = chunked.pca_scaled(L, hv["highly_variable"].to_numpy(), n_comps=50)
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 / 2**30
print(json.dumps({"peak_gb": peak, "cells": f.shape[0], "blocks": len(f.X.chunks[0])}))
"""


def test_memory_bounded_on_100k_cells():
    """QC..PCA by blocks on 100k cells (10x 1.3M brain subset, 201M non-zeros, gzip) in a
    child process at 2 workers. Measured 2026-10-23: peak 1.15 GB, ~51 s at 2 CPUs; the
    same steps in memory peaked 6.3 GB with HVG-only scaling (18.5 GB on Roihu scaling
    every gene). The bound fails a step that stops being block-bounded."""
    import json
    import subprocess
    import sys

    src = TESTDATA / "neurons_100000.h5ad"
    if not src.exists():
        pytest.skip("neurons_100000.h5ad not in SCVIEW_TESTDATA")
    r = subprocess.run([sys.executable, "-c", _MEMORY_RUN, str(src)], capture_output=True,
                       text=True, timeout=1200, check=True)
    got = json.loads(r.stdout.strip().splitlines()[-1])
    assert got["blocks"] > 10
    assert got["peak_gb"] < 2.0, got


@pytest.mark.parametrize("name", ["kang2018_ifnb_pbmc.h5ad", "paul15.h5ad"])
def test_real_data_qc_to_pca(name, tmp_path):
    src = TESTDATA / name
    if not src.exists():
        pytest.skip(f"{name} not in SCVIEW_TESTDATA")
    path = _modern(src, tmp_path)
    m = chunked.open_counts(path, rows=1500)
    assert len(m.X.chunks[0]) > 1 and m.X.chunks[0][-1] != 1500
    a = ad.read_h5ad(path)
    counts, where = normalization.find_counts(a)
    assert m.source == where
    a = ad.AnnData(counts.copy(), obs=a.obs, var=a.var)  # the pipeline after reset_to_counts
    p = PipelineParams()
    _run_qc_metrics(a, p)
    obs, var = chunked.qc_metrics(m)
    pd.testing.assert_frame_equal(obs, a.obs[QC_OBS], check_exact=True)
    pd.testing.assert_frame_equal(var, a.var[QC_VAR], check_exact=True)
    _run_normalization(a, p)
    _run_log_transform(a, p)
    L = chunked.log_normalize(m.X, p.target_sum)
    _assert_same_matrix(L.compute(), a.X)
    for flavor in ("seurat", "cell_ranger", "seurat_v3"):
        b = a.copy()
        _run_hvg(b, PipelineParams(hvg_flavor=flavor))
        got = chunked.highly_variable_genes(m.X if flavor == "seurat_v3" else L, m.var_names,
                                            flavor=flavor, n_top_genes=min(2000, m.shape[1]))
        assert got["highly_variable"].equals(b.var["highly_variable"]), flavor
    hvg = b.var["highly_variable"].to_numpy()
    got = chunked.pca_scaled(L, hvg, n_comps=50)
    ref = _scanpy_pca(_dense(a.X[:, hvg]), 50, scale=True)
    _check_pca(got, ref, ref.X, hvg, 50)
    # Pearson residuals: variable genes, then PCA of their residuals
    b = a.copy()
    _run_hvg(b, PipelineParams(normalization_method="pearson_residuals"))
    got = chunked.pearson_hvg(m.X, m.var_names, n_top_genes=min(2000, m.shape[1]))
    assert got["highly_variable"].equals(b.var["highly_variable"])
    hvg = got["highly_variable"].to_numpy()
    R = normalization.pearson_residuals(b.layers["counts"], columns=np.flatnonzero(hvg),
                                        clip=normalization.default_clip(m.shape[0]))
    _check_pca(chunked.pca_pearson(m.X, hvg, n_comps=50), _scanpy_pca(R, 50, scale=False),
               R, hvg, 50)
