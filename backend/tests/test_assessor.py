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
