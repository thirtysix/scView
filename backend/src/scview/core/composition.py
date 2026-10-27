"""Replicate-aware composition testing: do the groups of a column (cell types,
clusters) make up different shares of the cells in different conditions, with
the samples as the replicates?

A test on pooled cell counts (chi-square, Fisher) treats every cell as an
independent observation. Cells from one sample share that sample's make-up,
and samples differ (donor biology, dissociation, how many of each type a sample
happened to catch), so such a test reads ordinary sample-to-sample variation as
a shift, and the more cells, the smaller its p-values. propeller (Phipson et al.
2022, Bioinformatics 38:4720, PMID 36005887) tests the per-sample proportions
instead, which is what this module does:

1. each sample's share of every group, where a sample is one donor (or sample
   id) in one condition;
2. a logit transform of the shares, with 0.5 added to every count first
   (propeller's default; arcsin-square-root is its alternative, offered here
   too). Rare types sit near 0, where a proportion's variance shrinks with its
   mean; on the log-odds scale a shift is a ratio, so a rare type that doubles
   weighs like an abundant one that doubles, and the pseudocount keeps samples
   with none of a type finite;
3. a linear model per group, with the donor as a blocking factor when every
   donor was sampled in every condition (paired, as on Kang), then a moderated
   t-test for two conditions or an F-test for more. The residual variances are
   shrunk towards their common value by limma's empirical Bayes (``squeezeVar``,
   reimplemented here in numpy/scipy). By default robustly, as propeller calls it
   (``robust=TRUE``, ``fitFDistRobustly``): a group whose variance is an outlier
   gets a smaller prior df, so it is shrunk less and cannot inflate the prior
   for the rest. With fewer than ten groups limma's robust fit is the plain one;
4. Benjamini-Hochberg across the groups.

When the donors pair up across two conditions each group also carries a
``paired_test`` on its raw per-sample shares (Wilcoxon signed-rank and paired
t, core/paired_design.py), for the per-donor slope chart; ``pairing`` says
whether any donor occurs in more than one condition and how many are in all.

With fewer than two samples in a condition there is no replicate variance to
test against, so only the proportions are returned.
"""

from __future__ import annotations

import re
import threading
import weakref
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from scipy import optimize, special, stats

from scview.core import paired_design, results_store
from scview.core.confounding import check, condition_columns, for_comparison

Transform = Literal["logit", "asin"]

MIN_SAMPLES = 2  # per condition, for a test
MIN_CELLS_PER_SAMPLE = 10  # a smaller sample's shares are mostly noise: left out
MAX_GROUPS = 100
MAX_CONDITIONS = 8
MAX_SAMPLES = 500  # more and the column is per-cell, not per-sample
FDR = 0.05


class CompositionError(ValueError):
    """A request the test cannot serve (unknown column, one condition): a 400."""


@dataclass
class GroupResult:
    group: str
    props: dict[str, float | None]  # condition -> mean share over its samples
    n_cells: dict[str, int]  # condition -> cells of this group
    fold_change: float | None = None  # mean share, other condition / reference (2 conditions)
    stat: float | None = None  # moderated t (2 conditions) or F (more)
    pval: float | None = None
    fdr: float | None = None
    df_prior: float | None = None  # this group's prior df (robust: smaller for an outlier)
    # Wilcoxon signed-rank + paired t on the raw shares (%) of the donors in both
    # conditions (two conditions, paired donors only).
    paired_test: dict | None = None


@dataclass
class SampleRow:
    sample: str
    condition: str
    n_cells: int
    props: list[float]  # share of each group, in ``groups`` order


@dataclass
class CompositionResult:
    condition_key: str
    sample_key: str | None
    group_key: str
    conditions: list[str]  # reference first
    groups: list[GroupResult]
    samples: list[SampleRow]
    n_samples: dict[str, int]
    n_cells: dict[str, int]
    transform: str
    robust: bool = True
    tested: bool = False
    design: str | None = None  # "paired" | "unpaired"
    test: str | None = None  # "moderated t" | "moderated F"
    df_residual: int | None = None
    df_prior: float | None = None  # None with ``tested``: infinite (variances pooled)
    n_outliers: int = 0  # robust: groups given a smaller prior df than the rest
    n_significant: int = 0
    # The same groups tested with a chi-square on the pooled cell counts, for the
    # warning only: how many it would call significant at the same FDR.
    n_significant_cell_level: int | None = None
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    dropped_samples: list[str] = field(default_factory=list)
    # {"paired", "n_pairs", "conditions"}: whether the donors recur across conditions.
    pairing: dict = field(default_factory=lambda: {"paired": False, "n_pairs": 0, "conditions": []})

    def to_dict(self) -> dict:
        return {
            "condition_key": self.condition_key, "sample_key": self.sample_key,
            "group_key": self.group_key, "conditions": self.conditions,
            "reference": self.conditions[0], "transform": self.transform,
            "tested": self.tested, "design": self.design, "test": self.test,
            "robust": self.robust, "n_outliers": self.n_outliers,
            "df_residual": self.df_residual, "df_prior": self.df_prior,
            "n_samples": self.n_samples, "n_cells": self.n_cells,
            "n_significant": self.n_significant, "fdr_threshold": FDR,
            "n_significant_cell_level": self.n_significant_cell_level,
            "groups": [g.__dict__ for g in self.groups],
            "samples": [s.__dict__ for s in self.samples],
            "warnings": self.warnings, "notes": self.notes,
            "dropped_samples": self.dropped_samples, "pairing": self.pairing,
        }


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def transform_props(counts: np.ndarray, kind: Transform = "logit") -> np.ndarray:
    """Per-sample shares (rows = samples, columns = groups) on the test's scale,
    as propeller's getTransformedProps computes them."""
    counts = np.asarray(counts, dtype=np.float64)
    if kind == "logit":
        p = (counts + 0.5) / (counts + 0.5).sum(axis=1, keepdims=True)
        return np.log(p / (1 - p))
    if kind == "asin":
        return np.arcsin(np.sqrt(counts / counts.sum(axis=1, keepdims=True)))
    raise CompositionError(f"Unknown transform '{kind}' (logit or asin).")


def _trigamma_inverse(x: float) -> float:
    """y with trigamma(y) = x (limma's trigammaInverse, Newton's method)."""
    if x > 1e7:
        return 1.0 / np.sqrt(x)
    if x < 1e-6:
        return 1.0 / x
    y = 0.5 + 1.0 / x
    for _ in range(50):
        tri = special.polygamma(1, y)
        dif = tri * (1 - tri / x) / special.polygamma(2, y)
        y += dif
        if -dif / y < 1e-8:
            break
    return float(y)


def fit_f_dist(s2: np.ndarray, df: float) -> tuple[float, float]:
    """(prior variance, prior df) of residual variances ``s2`` that each have
    ``df`` degrees of freedom: limma's fitFDist, moment estimates on log(s2).
    The prior df is inf when the variances vary no more than chance allows."""
    x = np.asarray(s2, dtype=np.float64)
    x = np.maximum(x[np.isfinite(x)], 0)
    if len(x) < 2:
        return float("nan"), float("nan")
    m = float(np.median(x)) or 1.0  # limma: more than half exactly zero
    x = np.maximum(x, 1e-5 * m)
    e = np.log(x) - special.digamma(df / 2) + np.log(df / 2)
    emean = float(e.mean())
    evar = float(((e - emean) ** 2).sum() / (len(x) - 1)) - float(special.polygamma(1, df / 2))
    if evar > 0:
        df2 = 2 * _trigamma_inverse(evar)
        return float(np.exp(emean + special.digamma(df2 / 2) - np.log(df2 / 2))), df2
    return float(x.mean()), float("inf")


# 128-point Gauss-Legendre on [0, 1] (statmod's gauss.quad.prob, uniform).
_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(128)
_GL_NODES, _GL_WEIGHTS = (_GL_NODES + 1) / 2, _GL_WEIGHTS / 2


def _f_ppf(p: np.ndarray, df1: float, df2: float) -> np.ndarray:
    if np.isinf(df2):
        return stats.chi2.ppf(p, df1) / df1
    return stats.f.ppf(p, df1, df2)


def _f_pdf(x: np.ndarray, df1: float, df2: float) -> np.ndarray:
    if np.isinf(df2):
        return df1 * stats.chi2.pdf(df1 * x, df1)
    return stats.f.pdf(x, df1, df2)


def _winsorized_moments(df1: float, df2: float, tail: tuple[float, float]) -> tuple[float, float]:
    """Mean and variance of log(F(df1, df2)) winsorized at the ``tail``
    probabilities (limma's winsorizedMoments, by quadrature on x / (1 + x))."""
    fq = _f_ppf(np.array([tail[0], 1 - tail[1]]), df1, df2)
    zq = np.log(fq)
    q = fq / (1 + fq)
    nodes = q[0] + (q[1] - q[0]) * _GL_NODES
    fnodes = nodes / (1 - nodes)
    znodes = np.log(fnodes)
    f = _f_pdf(fnodes, df1, df2) / (1 - nodes) ** 2
    q21 = q[1] - q[0]
    m = q21 * float((_GL_WEIGHTS * f * znodes).sum()) + float((zq * tail).sum())
    v = (q21 * float((_GL_WEIGHTS * f * (znodes - m) ** 2).sum())
         + float(((zq - m) ** 2 * tail).sum()))
    return m, v


def fit_f_dist_robust(s2: np.ndarray, df: float, tail: tuple[float, float] = (0.05, 0.1),
                      ) -> tuple[float, np.ndarray]:
    """(prior variance, prior df per value) of residual variances ``s2`` with
    ``df`` degrees of freedom each: limma's fitFDistRobustly (no covariate, one
    df). The prior df is estimated from the winsorized log-variances, so a few
    outlying variances do not drag it down, and each outlier then gets a smaller
    prior df of its own, by how unlikely its variance is under the prior."""
    x = np.asarray(s2, dtype=np.float64)
    n = len(x)
    s0, df2_plain = fit_f_dist(x, df)
    if n < 3 or all(t < 1 / n for t in tail):
        return s0, np.full(n, df2_plain)  # limma: the plain fit
    m = float(np.median(x))
    if m <= 0:
        raise CompositionError("The groups' variances are mostly zero.")
    x = np.maximum(x, m * 1e-12)
    tail_a = np.array(tail, dtype=np.float64)

    z = np.log(x)
    ztrend = float(stats.trim_mean(z, tail[1]))
    zresid = z - ztrend
    lo, hi = np.quantile(zresid, [tail[0], 1 - tail[1]])
    zwins = np.clip(zresid, lo, hi)
    zwmean = float(zwins.mean())
    zwvar = float(((zwins - zwmean) ** 2).mean()) * n / (n - 1)

    mean_inf, var_inf = _winsorized_moments(df, np.inf, tail_a)
    funval_inf = np.log(zwvar / var_inf)
    rank = stats.rankdata(z)  # = rank of the F statistics (same order)
    if funval_inf <= 0:
        s20 = float(np.exp(ztrend + zwmean - mean_inf))
        fstat = np.exp(z) / s20
        tail_p = stats.chi2.sf(fstat * df, df)
        not_outlier = np.minimum(tail_p / ((n - rank + 0.5) / n), 1)
        shrunk = np.full(n, np.inf)
        out = not_outlier < 1
        if out.any():
            shrunk[out] = not_outlier[out] * n * df
            o = np.argsort(tail_p, kind="stable")
            shrunk[o] = np.maximum.accumulate(shrunk[o])
        return s20, shrunk

    if not np.isfinite(df2_plain):
        return s0, np.full(n, df2_plain)

    def fun(u: float) -> float:
        if u >= 1:
            return funval_inf
        return float(np.log(zwvar / _winsorized_moments(df, u / (1 - u), tail_a)[1]))

    rbx = df2_plain / (1 + df2_plain)
    funval_low = fun(rbx)
    if funval_low >= 0:
        df2 = df2_plain
    else:
        u = optimize.brentq(fun, rbx, 1, xtol=1e-8)
        df2 = u / (1 - u) if u < 1 else np.inf
    mean_df2, _ = _winsorized_moments(df, df2, tail_a)
    s20 = float(np.exp(ztrend + zwmean - mean_df2))
    fstat = np.exp(z) / s20
    log_tail_p = stats.f.logsf(fstat, df, df2)
    log_not_outlier = np.minimum(log_tail_p - (np.log(n - rank + 0.5) - np.log(n)), 0)
    if not (log_not_outlier < 0).any():
        return s20, np.full(n, df2)
    not_outlier = np.exp(log_not_outlier)
    p_outlier = -np.expm1(log_not_outlier)
    min_log_tail_p = float(log_tail_p.min())
    if min_log_tail_p == -np.inf:
        shrunk = not_outlier * df2
    else:
        df2_out = np.log(0.5) / min_log_tail_p * df2
        df2_out = np.log(0.5) / float(stats.f.logsf(fstat.max(), df, df2_out)) * df2_out
        shrunk = not_outlier * df2 + p_outlier * df2_out
    o = np.argsort(log_tail_p, kind="stable")
    ordered = shrunk[o]
    running = np.cumsum(ordered) / np.arange(1, n + 1)
    imin = int(np.argmin(running))
    ordered[: imin + 1] = running[imin]
    shrunk[o] = np.maximum.accumulate(ordered)
    return s20, shrunk


@dataclass
class _Fit:
    stat: np.ndarray
    pval: np.ndarray
    df_residual: int
    df_prior: float  # the common prior df (robust: the non-outliers')
    df_prior_each: np.ndarray  # per group


def moderated_test(Y: np.ndarray, X: np.ndarray, coefs: list[int], robust: bool = False) -> _Fit:
    """Per column of ``Y`` (samples x groups), least squares on design ``X``,
    variances squeezed by empirical Bayes, then a moderated t-test of the one
    coefficient in ``coefs`` or an F-test of them all (limma lmFit + eBayes,
    ``robust`` as eBayes' argument)."""
    n, p = X.shape
    if np.linalg.matrix_rank(X) < p:
        raise CompositionError("The design has no residual degrees of freedom to test against.")
    df = n - p
    if df < 1:
        raise CompositionError("Too few samples for the design: no residual degrees of freedom.")
    V = np.linalg.inv(X.T @ X)  # unscaled covariance of the coefficients
    beta = V @ X.T @ Y  # p x groups
    resid = Y - X @ beta
    s2 = (resid**2).sum(axis=0) / df

    G = Y.shape[1]
    if G < 3:  # limma's squeezeVar: nothing to borrow from, plain t / F
        s0, prior = float("nan"), np.zeros(G)
    elif robust:
        s0, prior = fit_f_dist_robust(s2, df)
    else:
        s0, d = fit_f_dist(s2, df)
        prior = np.full(G, d)
    if not np.isfinite(s0):
        s_post, prior = s2, np.zeros(G)
    else:
        with np.errstate(invalid="ignore"):
            s_post = np.where(np.isfinite(prior), (df * s2 + prior * s0) / (df + prior), s0)
    df_prior = float(np.max(prior)) if len(prior) else 0.0
    # limma caps the total df at the residual df pooled over every group.
    df_total = np.minimum(df + prior, df * G)

    with np.errstate(divide="ignore", invalid="ignore"):
        if len(coefs) == 1:
            j = coefs[0]
            stat = beta[j] / np.sqrt(s_post * V[j, j])
            pval = 2 * stats.t.sf(np.abs(stat), df_total)
        else:
            b = beta[coefs]  # r x groups
            W = np.linalg.inv(V[np.ix_(coefs, coefs)])
            r = len(coefs)
            stat = np.einsum("ig,ij,jg->g", b, W, b) / (r * s_post)
            # limma's F (classifyTestsF) takes df.residual + df.prior uncapped,
            # and a chi-square once that passes 1e6 (an infinite prior df).
            df2 = df + prior
            pval = np.where(df2 > 1e6, stats.chi2.sf(r * stat, r),
                            stats.f.sf(stat, r, np.minimum(df2, 1e6)))
    return _Fit(stat=stat, pval=pval, df_residual=df, df_prior=df_prior, df_prior_each=prior)


def _bh(p: np.ndarray) -> np.ndarray:
    out = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    if ok.any():
        out[ok] = stats.false_discovery_control(p[ok], method="bh")
    return out


def cell_level_pvals(cell_counts: np.ndarray) -> np.ndarray:
    """Per group (column of ``cell_counts``, conditions x groups), a chi-square
    test of its cells vs the rest across conditions: the cells-as-replicates test
    the warning is about. Returned for comparison, never as a result."""
    total = cell_counts.sum(axis=1)
    out = np.full(cell_counts.shape[1], np.nan)
    for g in range(cell_counts.shape[1]):
        table = np.vstack([cell_counts[:, g], total - cell_counts[:, g]])
        if (table.sum(axis=1) > 0).all():
            out[g] = stats.chi2_contingency(table)[1]
    return out


# ---------------------------------------------------------------------------
# The test
# ---------------------------------------------------------------------------


def _labels(obs: pd.DataFrame, col: str) -> pd.Series:
    s = obs[col]
    return s.astype(str).where(s.notna())


def _order(values: pd.Series) -> list[str]:
    """Distinct values as text, in the column's category order, else sorted
    (numbers by value: cluster 10 after cluster 9)."""
    present = set(values.dropna().astype(str))
    if isinstance(values.dtype, pd.CategoricalDtype):
        return [str(c) for c in values.cat.categories if str(c) in present]

    def key(v: str):
        try:
            return (0, float(v), v)
        except ValueError:
            return (1, 0.0, v)

    return sorted(present, key=key)


def composition_test(
    obs: pd.DataFrame,
    condition_key: str,
    sample_key: str | None,
    group_key: str,
    *,
    conditions: list[str] | None = None,
    reference: str | None = None,
    transform: Transform = "logit",
    paired: bool | None = None,
    batch_key: str | None = None,
    robust: bool = True,
) -> CompositionResult:
    """Test each group of ``group_key`` for a different share of the cells
    between the conditions of ``condition_key``, with the samples of
    ``sample_key`` (one per value per condition) as replicates. ``paired`` None
    pairs by sample when every sample value occurs in every condition.
    ``batch_key`` is the dataset's batch-correction key, if any. ``robust``
    (propeller's default) squeezes the variances robustly."""
    for col in (condition_key, group_key, sample_key):
        if col and col not in obs.columns:
            raise CompositionError(f"Column '{col}' not found.")
    if len({condition_key, group_key, sample_key} - {None}) < (3 if sample_key else 2):
        raise CompositionError("The condition, sample and group columns must be different.")
    if transform not in ("logit", "asin"):
        raise CompositionError(f"Unknown transform '{transform}' (logit or asin).")

    cond = _labels(obs, condition_key)
    available = _order(obs[condition_key].where(cond.notna()))
    chosen = list(dict.fromkeys(conditions)) if conditions else available
    unknown = [c for c in chosen if c not in available]
    if unknown:
        raise CompositionError(f"'{unknown[0]}' is not a value of '{condition_key}'.")
    if len(chosen) < 2:
        raise CompositionError(f"Pick at least two values of '{condition_key}' to compare.")
    if len(chosen) > MAX_CONDITIONS:
        raise CompositionError(
            f"'{condition_key}' has {len(chosen)} values; compare at most {MAX_CONDITIONS}.")
    if reference is not None:
        if reference not in chosen:
            raise CompositionError(f"The reference '{reference}' is not among the compared values.")
        chosen = [reference, *(c for c in chosen if c != reference)]

    group = _labels(obs, group_key)
    keep = cond.isin(chosen) & group.notna()
    sample = _labels(obs, sample_key) if sample_key else None
    if sample is not None:
        keep &= sample.notna()
    if not keep.any():
        raise CompositionError("No cells have all of the chosen columns set.")
    groups = _order(obs[group_key].where(keep))
    if len(groups) < 2:
        raise CompositionError(f"'{group_key}' has fewer than two groups in these cells.")
    if len(groups) > MAX_GROUPS:
        raise CompositionError(
            f"'{group_key}' has {len(groups)} groups; the test takes at most {MAX_GROUPS}.")

    c = cond[keep].to_numpy()
    g_codes = pd.Categorical(group[keep], categories=groups).codes

    # One sample per (sample value, condition).
    if sample is not None:
        s = sample[keep].to_numpy()
        sample_ids = pd.Series(c, dtype=object) + "\x1f" + pd.Series(s, dtype=object)
    else:
        sample_ids = pd.Series(c, dtype=object) + "\x1f"
    s_codes, s_names = pd.factorize(sample_ids, sort=True)
    counts = np.zeros((len(s_names), len(groups)), dtype=np.int64)
    np.add.at(counts, (s_codes, g_codes), 1)
    s_cond = np.array([x.split("\x1f", 1)[0] for x in s_names])
    s_donor = np.array([x.split("\x1f", 1)[1] for x in s_names])
    n_per = counts.sum(axis=1)
    if len(s_names) > MAX_SAMPLES:
        raise CompositionError(
            f"{len(s_names)} samples is too many (at most {MAX_SAMPLES}); '{sample_key}' "
            "looks like a per-cell column rather than a donor or sample.")

    result = CompositionResult(
        condition_key=condition_key, sample_key=sample_key, group_key=group_key,
        conditions=chosen, groups=[], samples=[], n_samples={}, n_cells={},
        transform=transform, robust=robust,
    )
    ok = n_per >= MIN_CELLS_PER_SAMPLE if sample is not None else np.ones(len(n_per), bool)
    result.dropped_samples = [f"{s_donor[i]} ({s_cond[i]}, {n_per[i]} cells)"
                              for i in np.flatnonzero(~ok)]
    counts, s_cond, s_donor, n_per = counts[ok], s_cond[ok], s_donor[ok], n_per[ok]
    props = counts / np.maximum(n_per, 1)[:, None]
    cell_counts = np.vstack([counts[s_cond == k].sum(axis=0) for k in chosen])

    result.n_samples = {k: int((s_cond == k).sum()) for k in chosen}
    result.n_cells = {k: int(cell_counts[i].sum()) for i, k in enumerate(chosen)}
    result.samples = [
        SampleRow(sample=s_donor[i] if sample is not None else s_cond[i], condition=s_cond[i],
                  n_cells=int(n_per[i]), props=[round(float(v), 6) for v in props[i]])
        for i in range(len(n_per))
    ]
    for gi, name in enumerate(groups):
        means = {k: float(props[s_cond == k, gi].mean()) if (s_cond == k).any() else None
                 for k in chosen}
        a, b = means[chosen[0]], means[chosen[-1]]
        fc = b / a if len(chosen) == 2 and a and b is not None else None
        result.groups.append(GroupResult(
            group=name, props=means, fold_change=fc,
            n_cells={k: int(cell_counts[i, gi]) for i, k in enumerate(chosen)}))
    if sample is not None:
        result.pairing = paired_design.detect_design(zip(s_donor, s_cond), chosen)
        if result.pairing["paired"] and len(chosen) == 2:
            # On the shares as served (rounded), so the numbers match the plot.
            points = [{"group": g, "condition": row.condition, "sample": row.sample,
                       "value": row.props[gi] * 100}
                      for row in result.samples for gi, g in enumerate(groups)]
            tests = paired_design.group_paired_tests(points, chosen, groups)
            for gr in result.groups:
                gr.paired_test = tests[gr.group]
    warnings, notes = design_warnings(obs, condition_key, group_key, keep.to_numpy(), batch_key)
    result.warnings += warnings
    result.notes += notes
    only = _one_condition_groups(cell_counts, groups)
    if only:
        shown = ", ".join(only[:5]) + (f" and {len(only) - 5} more" if len(only) > 5 else "")
        result.warnings.append(
            f"{len(only)} of {len(groups)} groups are almost only in one condition (95%+ of their "
            f"cells, adjusted for condition size): {shown}. Clusters built where the conditions "
            "separate (a stimulation response, an uncorrected batch) mark a change of state, not "
            "of abundance; compare cell types, or clusters from a batch-corrected embedding.")

    if sample is None:
        result.warnings.append(
            "No sample or donor column: the cells cannot be grouped into replicates, so "
            "only the proportions are shown. Pick the column that names each donor or sample.")
        return result
    few = [k for k in chosen if result.n_samples[k] < MIN_SAMPLES]
    if few:
        result.warnings.append(
            f"{', '.join(f'{k}: {result.n_samples[k]}' for k in few)} sample(s) of "
            f"'{sample_key}' (with {MIN_CELLS_PER_SAMPLE}+ cells): a test needs at least "
            f"{MIN_SAMPLES} per condition, so only the proportions are shown. One sample per "
            "condition cannot separate a condition effect from that sample's own make-up.")
        return result

    # Pair by donor when every donor was sampled in every condition.
    donors = sorted(set(s_donor))
    complete = all(set(s_cond[s_donor == d]) == set(chosen) for d in donors)
    design = "paired" if (complete if paired is None else paired) else "unpaired"
    if design == "paired" and not complete:
        raise CompositionError(
            f"Pairing needs every '{sample_key}' value in every condition; some are missing.")
    if paired is None and not complete and any(
            len(set(s_cond[s_donor == d])) > 1 for d in donors):
        result.notes.append(
            f"Some '{sample_key}' values occur in more than one condition but not in all, so "
            "the samples are treated as independent (unpaired).")

    X_cols = [np.ones(len(s_cond))]
    X_cols += [(s_cond == k).astype(float) for k in chosen[1:]]
    if design == "paired":
        X_cols += [(s_donor == d).astype(float) for d in donors[1:]]
    X = np.column_stack(X_cols)
    Y = transform_props(counts, transform)
    fit = moderated_test(Y, X, list(range(1, len(chosen))), robust=robust)

    fdr = _bh(fit.pval)
    for gi, gr in enumerate(result.groups):
        gr.stat = _finite(fit.stat[gi])
        gr.pval = _finite(fit.pval[gi])
        gr.fdr = _finite(fdr[gi])
        gr.df_prior = _finite(fit.df_prior_each[gi])
    result.tested = True
    result.design = design
    result.test = "moderated t" if len(chosen) == 2 else "moderated F"
    result.df_residual = fit.df_residual
    result.df_prior = fit.df_prior if np.isfinite(fit.df_prior) else None
    result.n_outliers = int((fit.df_prior_each < fit.df_prior).sum()) if robust else 0
    result.n_significant = int(np.nansum(fdr < FDR))
    cell_fdr = _bh(cell_level_pvals(cell_counts))
    result.n_significant_cell_level = int(np.nansum(cell_fdr < FDR))
    return result


def _finite(v) -> float | None:
    v = float(v)
    return v if np.isfinite(v) else None


ONE_CONDITION_SHARE = 0.95
ONE_CONDITION_MIN_CELLS = 20


def _one_condition_groups(cell_counts: np.ndarray, groups: list[str]) -> list[str]:
    """Groups whose cells come almost all from one condition, after scaling each
    condition to the same size (conditions x groups counts)."""
    totals = cell_counts.sum(axis=1, keepdims=True)
    if (totals == 0).any():
        return []
    rate = cell_counts / totals
    share = rate.max(axis=0) / np.maximum(rate.sum(axis=0), 1e-300)
    big = cell_counts.sum(axis=0) >= ONE_CONDITION_MIN_CELLS
    return [g for g, s, b in zip(groups, share, big) if b and s >= ONE_CONDITION_SHARE]


# Clusterings scView (or scanpy, Seurat) makes on an embedding; imported labels
# and per-cell annotations are not built on it.
CLUSTERING_RE = re.compile(r"^scview_|^(leiden|louvain|seurat_clusters)$|_sub$", re.I)


def design_warnings(obs: pd.DataFrame, condition_key: str, group_key: str, mask: np.ndarray,
                    batch_key: str | None) -> tuple[list[str], list[str]]:
    """(warnings, notes) where the design, not the test, limits what a
    composition difference means (core/confounding.py): conditions that are also
    batches, and groups built on an embedding corrected for the condition."""
    warnings: list[str] = []
    notes: list[str] = []
    extra = [batch_key] if batch_key and batch_key != condition_key else []
    hit = for_comparison(obs, condition_key, mask, extra)
    if hit is not None:
        warnings.append(f"{hit.message} A difference in composition is then also a difference "
                        "between batches (capture, dissociation, sequencing run).")
    if batch_key and batch_key in obs.columns:
        same = batch_key == condition_key
        if same or check(obs, batch_key, condition_key, mask) is not None:
            how = "is the condition" if same else "lines up with the condition"
            if CLUSTERING_RE.search(group_key):
                warnings.append(
                    f"The embedding was batch-corrected for '{batch_key}', which {how}. Clusters "
                    "built on it are pulled towards an even mix of the conditions, which can hide "
                    "composition differences.")
            else:
                notes.append(
                    f"The embedding was batch-corrected for '{batch_key}', which {how}: if "
                    f"'{group_key}' was built on that embedding, the correction can hide "
                    "composition differences (labels imported or assigned from expression are "
                    "unaffected).")
    return warnings, notes


# ---------------------------------------------------------------------------
# Co-pilot grounding: the last test run on each loaded dataset
# ---------------------------------------------------------------------------

_last: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_last_lock = threading.Lock()


def remember(adaptor, result: CompositionResult) -> None:
    """Keep the last result for the co-pilot. Held as long as the loaded dataset
    is: a re-processed (reloaded) dataset starts without one."""
    try:
        with _last_lock:
            _last[adaptor] = result
    except TypeError:  # not weak-referenceable (a test double)
        return
    results_store.keep(adaptor, "composition", result,
                       key=f"{result.condition_key}|{result.group_key}",
                       label=f"Composition of '{result.group_key}' by '{result.condition_key}'")


_CAUTION = (
    "- Cell-type proportions between conditions are tested with the samples (donors) as "
    "replicates (Compare > Composition > Between conditions). Never suggest a chi-square or "
    "Fisher test on cell counts: it treats cells as independent and overstates significance."
)


# How many stored runs the co-pilot looks through for one that still fits.
_MAX_RUNS = 12


def _runs(adaptor):
    """The dataset's composition runs, newest first: the one remembered in memory,
    then the stored ones computed on the data it holds now."""
    try:
        with _last_lock:
            mem = _last.get(adaptor)
    except TypeError:  # not weak-referenceable (a test double)
        return
    if mem is not None:
        yield mem
    did = results_store.dataset_id_of(adaptor)
    if not did:
        return
    for e in results_store.entries(did, "composition", adaptor=adaptor)[:_MAX_RUNS]:
        if e.get("stale"):
            continue
        got = results_store.load(did, e["id"])
        if got is not None and isinstance(got[1], CompositionResult):
            yield got[1]


def _fits(r: CompositionResult, obs: pd.DataFrame) -> bool:
    """Whether a run's columns and groups are still the dataset's."""
    if any(k and k not in obs.columns for k in (r.condition_key, r.sample_key, r.group_key)):
        return False
    return set(_order(obs[r.group_key])) == {g.group for g in r.groups}  # not renamed / re-clustered


def _pick(adaptor, obs: pd.DataFrame, group_key: str | None = None
          ) -> tuple[CompositionResult | None, CompositionResult | None]:
    """The newest run that still fits and was tested (else the newest that fits),
    and the newest run when it is not that one (its columns changed, or it was not
    tested). ``group_key``: only runs on that column."""
    newest = untested = None
    for r in _runs(adaptor):
        if group_key is not None and r.group_key != group_key:
            continue
        newest = newest or r
        if not _fits(r, obs):
            continue
        if r.tested:
            return r, (newest if newest is not r else None)
        untested = untested or r
    return untested, (newest if newest is not untested else None)


def _last_result(adaptor, obs: pd.DataFrame, group_key: str | None = None) -> CompositionResult | None:
    return _pick(adaptor, obs, group_key)[0]


def _run_name(r: CompositionResult) -> str:
    return f"'{r.group_key}' by '{r.condition_key}'"


def grounding_lines(adaptor, obs: pd.DataFrame) -> tuple[list[str], str | None]:
    """For the co-pilot: the newest valid test's summary (and a citable detail line
    for it), plus the caution on cell-level tests wherever conditions exist."""
    r, newer = _pick(adaptor, obs)
    if r is None:
        return ([f"\n## Composition between conditions\n{_CAUTION}"]
                if condition_columns(obs) else []), None
    which = "last run" if newer is None else "earlier run"
    skipped = []
    if newer is not None:
        why = "not tested" if _fits(newer, obs) else "its columns or groups changed since"
        skipped = [f"- The newest run ({_run_name(newer)}) is not used: {why}."]
    vs = " vs ".join(reversed(r.conditions)) if len(r.conditions) == 2 else ", ".join(r.conditions)
    head = f"{vs} by '{r.group_key}'"
    run = f" (run: {_run_name(r)})"
    if not r.tested:
        detail = f"{head}: proportions only, not tested{run}. " + " ".join(r.warnings)
        return [f"\n## Composition test ({which})", f"- {detail} [result:composition]",
                *skipped, _CAUTION], detail
    reps = ", ".join(f"{k} {n}" for k, n in r.n_samples.items())
    detail = (f"{head}, samples from '{r.sample_key}' ({reps}; {r.design}), {r.test}: "
              f"{r.n_significant} of {len(r.groups)} groups at FDR < {FDR}{run}")
    lines = [f"\n## Composition test ({which}; samples as replicates, propeller-style)",
             f"- {detail} [result:composition].", *skipped]
    for g in sorted(r.groups, key=lambda g: (g.fdr is None, g.fdr or 1.0))[:5]:
        shares = " -> ".join("n/a" if g.props[k] is None else f"{g.props[k] * 100:.1f}%"
                             for k in r.conditions)
        fdr = "n/a" if g.fdr is None else f"{g.fdr:.2g}"
        lines.append(f"- {g.group}: {shares} ({' -> '.join(r.conditions)}), FDR {fdr}")
    lines.append(_CAUTION)
    return lines, detail
