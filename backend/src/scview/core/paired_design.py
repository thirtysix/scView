"""Paired (before-after) and longitudinal designs over per-donor values.

A per-donor plot shows one pseudobulk value (or one proportion) per donor in
each condition. When the same donors were sampled in more than one condition,
as on Kang (8 donors, each CTRL and STIM), the two conditions' points are not
independent samples: each donor is its own control, and a line joining a
donor's values shows the change the box plots hide. ``detect_design`` reads
that off the (donor, condition) samples; ``paired_test`` compares two
conditions pair by pair with a Wilcoxon signed-rank test (scipy, exact for
small n without ties) and a paired t-test, on the plotted units. Both are
descriptive below ``MIN_PAIRS`` pairs: an exact signed-rank test on 4 pairs
cannot go under p = 0.125 two-sided, and a paired t on so few differences
leans entirely on their normality.

Used by core/donor_expression.py (a gene per donor) and core/composition.py
(a group's share per donor). With more than two conditions the design is still
paired (longitudinal: the lines are drawn) but no test is made here.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Iterable, Sequence

import numpy as np
from scipy import stats

MIN_PAIRS = 5  # fewer: the p-values are descriptive
MIN_PAIRS_TEST = 2  # fewer: no test at all


def detect_design(samples: Iterable[tuple[str, str]], conditions: Sequence[str]) -> dict:
    """``{"paired", "n_pairs", "conditions"}`` from the (donor, condition)
    samples: paired when some donor occurs in more than one condition;
    ``n_pairs`` is the donors present in every condition."""
    conds = [c for c in conditions if c != ""]
    by_donor: dict[str, set[str]] = {}
    for donor, cond in samples:
        if cond in conds:
            by_donor.setdefault(donor, set()).add(cond)
    paired = len(conds) > 1 and any(len(s) > 1 for s in by_donor.values())
    n_pairs = sum(1 for s in by_donor.values() if s == set(conds)) if paired else 0
    return {"paired": paired, "n_pairs": n_pairs, "conditions": conds}


def paired_test(first: Sequence[float], second: Sequence[float]) -> dict:
    """Second minus first over the pairs: ``n_pairs``, ``mean_diff``,
    ``wilcoxon_p`` (signed-rank, zeros dropped as Wilcoxon did, exact when scipy
    can), ``ttest_p`` (paired t), and a ``caution`` below MIN_PAIRS pairs.
    Fewer than two pairs or no non-zero difference: both p None."""
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("The two conditions must give one value per pair.")
    d = b - a
    n = int(d.size)
    out: dict = {"n_pairs": n, "mean_diff": _finite(d.mean()) if n else None,
                 "wilcoxon_p": None, "ttest_p": None, "caution": None}
    if n < MIN_PAIRS_TEST:
        out["caution"] = (f"{n} pair{'s' if n != 1 else ''}: nothing to test." if n else
                          "No donor has a value in both conditions here.")
        return out
    if np.any(d != 0):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out["wilcoxon_p"] = _finite(stats.wilcoxon(b, a, zero_method="wilcox").pvalue)
            out["ttest_p"] = _finite(stats.ttest_rel(b, a).pvalue)
    if n < MIN_PAIRS:
        out["caution"] = (
            f"{n} pairs: read these p-values as descriptive. An exact signed-rank test on "
            f"{n} pairs cannot go below p = {_min_wilcoxon_p(n):.3g} two-sided, and a paired "
            "t on so few differences rests on their normality.")
    elif out["wilcoxon_p"] is None:
        out["caution"] = "Every donor has the same value in both conditions: no test."
    return out


def group_paired_tests(points: Iterable[dict], conditions: Sequence[str],
                       groups: Sequence[str]) -> dict[str, dict]:
    """``paired_test`` per group over the donors with a value in both of the
    two ``conditions`` (points: dicts with group, condition, sample, value)."""
    first, second = conditions[0], conditions[1]
    by_group: dict[str, dict[str, dict[str, float]]] = {g: {} for g in groups}
    for p in points:
        if p["condition"] in (first, second) and p["group"] in by_group:
            by_group[p["group"]].setdefault(p["sample"], {})[p["condition"]] = float(p["value"])
    out = {}
    for g in groups:
        donors = sorted(d for d, v in by_group[g].items() if first in v and second in v)
        out[g] = paired_test([by_group[g][d][first] for d in donors],
                             [by_group[g][d][second] for d in donors])
    return out


def _min_wilcoxon_p(n: int) -> float:
    """The smallest two-sided p an exact signed-rank test can give with n pairs."""
    return min(1.0, 2.0 / 2 ** n)


def _finite(v) -> float | None:
    v = float(v)
    return v if math.isfinite(v) else None
