"""QC distributions come from counts, and expression reads work in any gene order."""

from __future__ import annotations

import anndata as ad
import numpy as np
import scipy.sparse as sp

from scview.core.anndata_adaptor import AnnDataAdaptor, _row_stats


def _write(tmp_path, a) -> AnnDataAdaptor:
    path = tmp_path / "x.h5ad"
    a.write_h5ad(path)
    return AnnDataAdaptor(str(path))


def test_qc_uses_raw_counts_when_x_is_scaled(tmp_path):
    rng = np.random.default_rng(0)
    counts = rng.poisson(2.0, (50, 10)).astype("float32")
    a = ad.AnnData(counts.copy())
    a.var_names = [f"G{i}" for i in range(9)] + ["MT-CO1"]
    a.raw = a
    a.X = (counts - counts.mean(0)) / (counts.std(0) + 1e-6)  # scaled: negative values
    qc = _write(tmp_path, a).qc_distributions()
    tc = qc["metrics"]["total_counts"] if "metrics" in qc else qc["total_counts"]
    assert tc["median"] == float(np.median(counts.sum(1)))


def test_qc_reads_existing_n_counts(tmp_path):
    a = ad.AnnData(np.full((4, 3), -1.0, dtype="float32"))  # X useless for counts
    a.obs["n_counts"] = [10.0, 20.0, 30.0, 40.0]
    a.obs["n_genes"] = [1.0, 2.0, 3.0, 3.0]
    a.obs["percent_mito"] = [0.0, 0.0, 0.0, 0.0]
    qc = _write(tmp_path, a).qc_distributions()
    tc = qc["metrics"]["total_counts"] if "metrics" in qc else qc["total_counts"]
    assert tc["median"] == 25.0


def test_row_stats_in_chunks_match_whole_matrix():
    M = sp.random(30, 8, density=0.4, format="csr", random_state=0) * 10
    mask = np.array([False] * 7 + [True])
    tc, ng, mt = _row_stats(M, mask, chunk=7)
    assert np.allclose(tc, np.asarray(M.sum(1)).ravel())
    assert np.allclose(ng, np.asarray((M > 0).sum(1)).ravel())
    assert np.allclose(mt, np.asarray(M[:, 7].sum(1)).ravel())


def test_expression_keeps_requested_order_and_duplicates(tmp_path):
    a = ad.AnnData(np.arange(12, dtype="float32").reshape(3, 4))
    a.var_names = ["G0", "G1", "G2", "G3"]
    ad_ = _write(tmp_path, a)
    out = ad_.get_expression(["G3", "G1", "G3"], layer="X")
    assert out[:, 0].tolist() == [3, 7, 11] and out[:, 1].tolist() == [1, 5, 9]
    assert out[:, 2].tolist() == [3, 7, 11]
