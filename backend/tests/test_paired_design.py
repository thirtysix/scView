"""Pairing detection and the paired tests (core/paired_design.py) behind the
per-donor slope charts."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from scview.core import paired_design as pdn

CTRL = [1.0, 2.5, 3.0, 4.2, 5.1, 6.3, 7.0, 8.4]
STIM = [2.0, 2.4, 4.5, 5.0, 6.9, 6.0, 9.1, 9.8]


def test_detects_paired_disjoint_and_longitudinal_designs():
    paired = pdn.detect_design([("D0", "CTRL"), ("D0", "STIM"), ("D1", "CTRL"), ("D1", "STIM"),
                                ("D2", "CTRL")], ["CTRL", "STIM"])
    assert paired == {"paired": True, "n_pairs": 2, "conditions": ["CTRL", "STIM"]}
    disjoint = pdn.detect_design([("D0", "CTRL"), ("D1", "CTRL"), ("D2", "STIM"), ("D3", "STIM")],
                                 ["CTRL", "STIM"])
    assert disjoint == {"paired": False, "n_pairs": 0, "conditions": ["CTRL", "STIM"]}
    times = ["T0", "T1", "T2"]
    longitudinal = pdn.detect_design([(d, t) for d in ("D0", "D1") for t in times] + [("D2", "T0")], times)
    assert longitudinal == {"paired": True, "n_pairs": 2, "conditions": times}
    # Unsplit ("" condition) or one condition: nothing to pair.
    assert pdn.detect_design([("D0", ""), ("D1", "")], [""])["paired"] is False
    assert pdn.detect_design([("D0", "CTRL"), ("D0", "STIM")], ["CTRL"])["paired"] is False


def test_paired_test_matches_scipy_by_hand():
    r = pdn.paired_test(CTRL, STIM)
    d = np.array(STIM) - np.array(CTRL)
    assert r["n_pairs"] == 8 and r["caution"] is None
    assert r["mean_diff"] == pytest.approx(d.mean())
    assert r["wilcoxon_p"] == pytest.approx(
        stats.wilcoxon(d, zero_method="wilcox", method="exact").pvalue)
    assert r["ttest_p"] == pytest.approx(stats.ttest_rel(STIM, CTRL).pvalue)
    # Exact for 8 pairs without ties: the two negative differences (-0.1, -0.3) are
    # the two smallest, so W- = 1 + 2 = 3; five of the 256 sign patterns give a sum
    # of 3 or less ({}, {1}, {2}, {3}, {1, 2}): p = 2 * 5 / 256.
    assert r["wilcoxon_p"] == pytest.approx(10 / 256)


def test_few_pairs_are_descriptive_and_too_few_untested():
    r = pdn.paired_test(CTRL[:4], STIM[:4])
    assert r["n_pairs"] == 4 and r["wilcoxon_p"] is not None and r["ttest_p"] is not None
    assert "4 pairs: read these p-values as descriptive" in r["caution"]
    assert "cannot go below p = 0.125" in r["caution"]
    one = pdn.paired_test([1.0], [2.0])
    assert one["n_pairs"] == 1 and one["wilcoxon_p"] is None and one["ttest_p"] is None
    assert one["mean_diff"] == 1.0 and "nothing to test" in one["caution"]
    none = pdn.paired_test([], [])
    assert none["n_pairs"] == 0 and none["mean_diff"] is None and "No donor" in none["caution"]
    same = pdn.paired_test([1.0] * 6, [1.0] * 6)
    assert same["wilcoxon_p"] is None and same["ttest_p"] is None and "same value" in same["caution"]
    with pytest.raises(ValueError, match="one value per pair"):
        pdn.paired_test([1.0, 2.0], [1.0])


def test_group_tests_use_only_the_donors_in_both_conditions():
    pts = [{"group": "A", "condition": c, "sample": f"D{i}", "value": v}
           for i, (a, b) in enumerate(zip(CTRL, STIM)) for c, v in (("CTRL", a), ("STIM", b))]
    pts.append({"group": "A", "condition": "STIM", "sample": "D9", "value": 50.0})  # no CTRL
    pts.append({"group": "B", "condition": "CTRL", "sample": "D0", "value": 1.0})
    pts.append({"group": "B", "condition": "STIM", "sample": "D1", "value": 2.0})
    tests = pdn.group_paired_tests(pts, ["CTRL", "STIM"], ["A", "B"])
    assert tests["A"]["n_pairs"] == 8
    assert tests["A"]["wilcoxon_p"] == pytest.approx(10 / 256)
    assert tests["B"]["n_pairs"] == 0 and tests["B"]["wilcoxon_p"] is None


def test_unpaired_test_matches_scipy_and_names_the_exact_floor():
    patients = [0.05, 0.08, 0.10, 0.11, 0.13, 0.15, 0.24]
    controls = [0.20, 0.30, 0.35, 0.40, 0.45, 0.50]
    r = pdn.unpaired_test(controls, patients)
    assert r["n_a"] == 6 and r["n_b"] == 7 and r["method"] == "exact"
    assert r["mannwhitney_p"] == pytest.approx(stats.mannwhitneyu(patients, controls, method="exact").pvalue)
    assert r["welch_p"] == pytest.approx(stats.ttest_ind(patients, controls, equal_var=False).pvalue)
    assert r["mean_diff"] == pytest.approx(np.mean(patients) - np.mean(controls))
    assert r["min_p"] == pytest.approx(2 / 1716) and r["caution"] is None  # 0.24 sits among controls
    apart = pdn.unpaired_test(controls, [v - 0.2 for v in patients])
    assert apart["mannwhitney_p"] == pytest.approx(2 / 1716)
    assert "smallest two-sided p" in apart["caution"] and "C(13, 6)" in apart["caution"]
    tied = pdn.unpaired_test([1.0, 1.0, 2.0], [3.0, 3.0, 4.0])
    assert tied["method"].startswith("normal approximation") and tied["mannwhitney_p"] is not None
    few = pdn.unpaired_test([1.0, 2.0], [3.0, 4.0, 5.0])
    assert few["mannwhitney_p"] is None and "at least 3" in few["caution"]


def test_group_unpaired_tests_per_group():
    pts = [{"group": g, "condition": c, "sample": f"{c}{i}", "value": float(i + (5 if c == "B" else 0))}
           for g in ("x", "y") for c in ("A", "B") for i in range(4)]
    out = pdn.group_unpaired_tests(pts, ["A", "B"], ["x", "y"])
    assert set(out) == {"x", "y"} and out["x"]["n_a"] == out["x"]["n_b"] == 4
    assert out["x"]["mean_diff"] == pytest.approx(5.0) and out["x"]["mannwhitney_p"] == pytest.approx(2 / 70)
