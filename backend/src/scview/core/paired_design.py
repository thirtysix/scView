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

Different donors in the two conditions (unpaired: patients vs controls) are
compared by ``unpaired_test``: a Wilcoxon rank-sum (Mann-Whitney U, exact when
no two values tie) and a Welch t on the per-donor values, the test Wilk et al.
(2020) used on donor means. With so few donors the exact rank-sum has a floor:
7 vs 6 donors cannot go below p = 2 / C(13, 6) = 0.0012 two-sided, whatever
the separation; the result says so when the p is that floor.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Iterable, Sequence

import numpy as np
from scipy import stats

MIN_PAIRS = 5  # fewer: the p-values are descriptive
MIN_PAIRS_TEST = 2  # fewer: no test at all
MIN_DONORS_UNPAIRED = 3  # per condition, for the rank-sum and Welch tests


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


def unpaired_test(first: Sequence[float], second: Sequence[float]) -> dict:
    """Second against first, each donor once: ``n_a``, ``n_b``, ``mean_diff`` (second's
    mean minus first's), ``mannwhitney_p`` (two-sided rank-sum, exact without ties, else
    the normal approximation: ``method`` says which), ``welch_p``, ``min_p`` (the
    smallest p the exact rank-sum can give at these sizes) and a ``caution`` when a side
    has under MIN_DONORS_UNPAIRED donors (no test) or the p is at that floor."""
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    na, nb = int(a.size), int(b.size)
    out: dict = {"n_a": na, "n_b": nb,
                 "mean_diff": _finite(b.mean() - a.mean()) if na and nb else None,
                 "mannwhitney_p": None, "welch_p": None, "method": None,
                 "min_p": _min_ranksum_p(na, nb) if na and nb else None, "caution": None}
    if min(na, nb) < MIN_DONORS_UNPAIRED:
        out["caution"] = (f"{na} vs {nb} donors: a donor-level test needs at least "
                          f"{MIN_DONORS_UNPAIRED} in each condition.")
        return out
    both = np.concatenate([a, b])
    if np.ptp(both) == 0:
        out["caution"] = "Every donor has the same value: no test."
        return out
    exact = np.unique(both).size == both.size
    out["method"] = "exact" if exact else "normal approximation (tied values)"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out["mannwhitney_p"] = _finite(stats.mannwhitneyu(
            b, a, alternative="two-sided", method="exact" if exact else "asymptotic").pvalue)
        out["welch_p"] = _finite(stats.ttest_ind(b, a, equal_var=False).pvalue)
    p = out["mannwhitney_p"]
    if exact and p is not None and p <= out["min_p"] * (1 + 1e-9):
        out["caution"] = (
            f"{na} vs {nb} donors: p = {p:.2g} is the smallest two-sided p an exact rank-sum test "
            f"can give at these sizes (2 / C({na + nb}, {na})): every donor of one condition is "
            "beyond every donor of the other, and more donors would be needed for a smaller p.")
    return out


def group_unpaired_tests(points: Iterable[dict], conditions: Sequence[str],
                         groups: Sequence[str]) -> dict[str, dict]:
    """``unpaired_test`` per group over its donors' values in the two ``conditions``
    (points: dicts with group, condition, sample, value)."""
    first, second = conditions[0], conditions[1]
    vals: dict[str, dict[str, list[float]]] = {g: {first: [], second: []} for g in groups}
    for p in points:
        if p["condition"] in (first, second) and p["group"] in vals:
            vals[p["group"]][p["condition"]].append(float(p["value"]))
    return {g: unpaired_test(vals[g][first], vals[g][second]) for g in groups}


def _min_ranksum_p(na: int, nb: int) -> float:
    """The smallest two-sided p an exact rank-sum test can give with na and nb values."""
    return min(1.0, 2.0 / math.comb(na + nb, na))


def _min_wilcoxon_p(n: int) -> float:
    """The smallest two-sided p an exact signed-rank test can give with n pairs."""
    return min(1.0, 2.0 / 2 ** n)


def _finite(v) -> float | None:
    v = float(v)
    return v if math.isfinite(v) else None
