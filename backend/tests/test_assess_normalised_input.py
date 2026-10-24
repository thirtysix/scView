"""C011: an upload that is already normalised says so in the Data Assessment (core/assessor.py).

Checked 2026-10-24 on Kang copies (CPM with and without counts, log1p(CPM), log1p(1e4) with
and without a counts layer): the normalisation check names the scale (1e4 fine, 1e6 a
warning) and where the counts can be recovered from. One gap: with counts in .raw beside
a log-normalised X (Kang as published), the log check read .raw and said "not
log-transformed", so the step was pre-selected and would have logged X twice.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import scanpy as sc
import scipy.sparse as sp

from scview.core.assessor import assess_preprocessing


def _counts(n: int = 300, g: int = 500, seed: int = 0) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    a = ad.AnnData(sp.csr_matrix(rng.poisson(rng.gamma(0.5, 2.0, g), (n, g)).astype(np.float32)))
    a.var_names = [f"G{i}" for i in range(g)]
    return a


def _normalised(target: float, log: bool) -> ad.AnnData:
    a = _counts()
    sc.pp.normalize_total(a, target_sum=target)
    if log:
        sc.pp.log1p(a)
    return a


def test_cpm_is_a_warning_and_says_where_the_counts_are():
    cpm = _normalised(1e6, log=False)
    cpm.raw = _counts()
    s = assess_preprocessing(cpm)
    assert "CPM-scale: WARNING" in s.normalization.details
    assert "Raw counts are recoverable from .raw" in s.normalization.details
    assert not s.log_transform.done  # CPM values, not logged
    bare = assess_preprocessing(_normalised(1e6, log=True))
    assert "Median per-cell total of expm1(X) = 1e+06" in bare.normalization.details
    assert "No raw counts found" in bare.normalization.details and bare.log_transform.done


def test_log1p_of_1e4_is_fine_and_the_counts_layer_is_named():
    a = _normalised(1e4, log=True)
    a.layers["counts"] = _counts().X
    s = assess_preprocessing(a)
    assert "~1e4, the scanpy/Seurat default; fine" in s.normalization.details
    assert "recoverable from layers['counts']" in s.normalization.details
    assert s.log_transform.done


def test_counts_in_raw_beside_a_logged_x_is_logged():
    a = _normalised(1e4, log=True)
    a.raw = _counts()  # Kang as published
    s = assess_preprocessing(a)
    assert s.log_transform.done and s.log_transform.confidence == "high"
    assert s.log_transform.details.startswith("X max = ")
    assert ".raw holds the raw counts" in s.log_transform.details
    # A normalised but unlogged X beside counts in .raw is still not logged
    b = _normalised(1e4, log=False)
    b.raw = _counts()
    assert not assess_preprocessing(b).log_transform.done
