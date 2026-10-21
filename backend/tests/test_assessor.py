"""Tests for preprocessing-state assessment — raw-counts detection + provenance overlay."""

from __future__ import annotations

import anndata as ad
import numpy as np

from scview.core import provenance
from scview.core.assessor import assess_preprocessing

_RNG = np.random.default_rng(0)


def _counts(n_obs=60, n_var=40):
    return ad.AnnData(X=_RNG.poisson(3.0, (n_obs, n_var)).astype("float32"))


def test_raw_counts_not_normalized_scaled_or_logged():
    # 10x-style counts: integer-valued but float-typed.
    s = assess_preprocessing(_counts())
    for step in (s.normalization, s.log_transform, s.scaling):
        assert step.done is False
        assert step.confidence == "high"
    assert "raw" in s.normalization.details.lower()


def test_scaled_data_is_detected():
    # z-scored data: non-integer, has negatives, mean~0 std~1.
    a = ad.AnnData(X=_RNG.standard_normal((200, 50)).astype("float32"))
    assert assess_preprocessing(a).scaling.done is True


def test_lognorm_data_not_flagged_as_raw():
    a = ad.AnnData(X=np.log1p(_RNG.poisson(3.0, (60, 40)).astype("float32")))
    s = assess_preprocessing(a)
    assert s.log_transform.done is True  # values <= ~log1p(max), consistent with log1p
    assert s.scaling.done is False  # non-negative → not scaled


def test_provenance_overlay_overrides_heuristic():
    # Data still looks like raw counts, but scView recorded normalizing it →
    # the recorded fact wins.
    a = _counts()
    provenance.record_step(a, step="normalization", tool="sc.pp.normalize_total", params={})
    s = assess_preprocessing(a)
    assert s.normalization.done is True
    assert "recorded" in s.normalization.details.lower()
    # a step NOT recorded stays not-done
    assert s.scaling.done is False


def _norm_log(target, n_obs=80, n_var=60, hvg_of=None):
    c = _RNG.poisson(3.0, (n_obs, n_var)).astype("float64") + 1
    x = np.log1p(c / c.sum(1, keepdims=True) * target).astype("float32")
    return ad.AnnData(X=x)


def test_cpm_scale_warns():
    s = assess_preprocessing(_norm_log(1e6))
    d = s.normalization.details
    assert "CPM" in d and "1e6" in d
    assert "No raw counts" in d


def test_1e4_scale_no_warning():
    d = assess_preprocessing(_norm_log(1e4)).normalization.details
    assert "CPM" not in d and "WARNING" not in d
    assert "scanpy/Seurat default" in d


def test_counts_layer_reported_recoverable():
    a = _norm_log(1e6)
    a.layers["counts"] = _RNG.poisson(3.0, a.shape).astype("float32")
    assert "recoverable from layers['counts']" in assess_preprocessing(a).normalization.details


def test_hvg_subset_softens_message():
    a = _norm_log(1e4, n_var=60)
    full = _norm_log(1e4, n_var=60)[:, :10].copy()
    full.raw = a
    d = assess_preprocessing(full).normalization.details
    assert "gene subset" in d or "gene subsetting" in d
    assert "recoverable" not in d  # raw here is normalised, not counts


def test_raw_counts_have_no_scale_note():
    d = assess_preprocessing(_counts()).normalization.details
    assert "Median per-cell total" not in d


def test_sample_totals_in_chunks_equal_the_whole_sample():
    # _scale_note reads its 2,000-cell sample in chunks of rows (the whole sample
    # dense, with its copies, took 1 GB on ovary's 20k genes).
    import scipy.sparse as sp

    from scview.core import assessor

    rng = np.random.default_rng(0)
    counts = rng.poisson(1.0, (600, 50)).astype(np.float64)
    logx = np.log1p(counts / counts.sum(axis=1, keepdims=True) * 1e4)
    for X in (logx, sp.csr_matrix(logx)):
        logged, totals = assessor._sample_totals(X, 600)  # 3 chunks, the last one partial
        assert logged and np.allclose(totals, np.expm1(logx).sum(axis=1))
    logged, totals = assessor._sample_totals(counts * 50, 600)
    assert not logged and np.allclose(totals, (counts * 50).sum(axis=1))
    assert assessor._sample_totals(logx - 1.0, 600) is None  # scaled: negative values
