"""Study planner: how many donors would a follow-up experiment need, judged
from the open dataset's own donor-to-donor variability?

Single-cell power rests on the donors far more than on the cells: the donor
is the replicate, so adding cells sharpens each donor's own estimate and does
nothing for the spread between donors. Both planners here take that spread
from the pilot (the open dataset), simulate studies of other sizes from it,
and run on every simulated study the test the app itself would run.

Composition (``composition_*``): power to detect a fold change in one cell
type's share with the propeller-style test of core/composition.py.

* The pilot: per sample (a donor in a condition) the cells of each type, the
  samples as composition_test forms them (under 10 cells left out).
* The model, on the test's own scale: the log-odds of type j's share in
  sample i is ``mu_cj + u_dj + e_ij``, with a condition mean, a donor effect
  shared by that donor's samples (sd ``tau_j``) and a residual (sd
  ``sigma_j``), and the sample's count of type j is binomial on its cells.
  propeller's own simulations draw each sample's share from a Beta and its
  count from a binomial, type by type (Phipson et al. 2022, PMID 36005887);
  the logit-normal share here splits the donor part from the rest the way the
  test's linear model (donor as a block) does.
* The estimate: the residual variance of each type in the app's own linear
  model on the pilot (logit shares, donor-blocked when the donors pair up) is
  donor-to-donor variance plus counting noise. The variances are chosen so
  that simulated pilots of the pilot's exact design (its samples, its cells
  per sample) give, on average, the pilot's own residual variances: a method
  of simulated moments, which removes the counting noise, the pseudocount and
  the share-to-logit distortion at once. The uncertainty is the chi-square
  interval of that residual variance on its residual df: with 8 donors it
  spans about a factor of four, and the planner says so.
* The simulation: n donors per group (paired: each donor in both conditions,
  only from a paired pilot), m cells per sample, the target type's mean share
  times the fold change and the other types' shares scaled down to make room
  (as propeller's simulations do), then ``transform_props`` +
  ``moderated_test`` + Benjamini-Hochberg across all types, exactly the app's
  test. Power = share of simulated studies whose target type has adjusted
  p < alpha in the right direction.

No replicates (one sample per condition, no donor column): there is nothing
to estimate the spread from; the planner says so and can run on a
user-entered spread instead, labelled as an assumption.
"""

from __future__ import annotations

import math
import threading
import time
import weakref
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import special, stats

from scview.core import composition as comp

# ---------------------------------------------------------------------------
# Defaults and bounds
# ---------------------------------------------------------------------------

DEFAULT_DONORS = (3, 4, 6, 8, 12, 16)
DEFAULT_CELLS = (500, 2000, 5000)
DEFAULT_FOLD_CHANGES = (1.5, 2.0, 3.0)
TARGET_POWER = 0.8
ALPHA = 0.05
MAX_GRID_VALUES = 12  # per axis
MAX_DONORS = 100
MAX_CELLS = 100_000
MAX_TARGETS = 6
MAX_SHARE = 0.98  # a fold change that would push a type above this share is infeasible
MIN_SHARE = 1e-4  # a type absent from the reference is simulated at this share

# Monte Carlo: studies simulated per grid point. Chosen from the grid size so a
# typical grid fits the time budget; the Monte Carlo error is reported.
MIN_REPS, MAX_REPS = 40, 400
TIME_BUDGET_S = 35.0
# Seconds per moderated test (2 cores, measured 2026-10-22): limma's robust
# variance fit (10+ groups) costs ~1.4 ms, the plain one ~0.2 ms.
_COST_ROBUST, _COST_PLAIN = 1.6e-3, 0.3e-3

# Method of simulated moments: pilots simulated per round, rounds.
CAL_REPS, CAL_ROUNDS = 300, 6

# The spread assumed when the pilot has no replicates: the donor-to-donor SD of
# a type's log-odds share, the median over the 13 PBMC types of Kang et al. 2018
# (8 donors, PMID 29227470) as this planner estimates it (unpaired, CTRL and
# STIM samples within condition). A solid tissue, which dissociates unevenly,
# usually varies more; enter its own value when one is known.
ASSUMED_SD = 0.35
ASSUMED_SD_SOURCE = (
    "median donor-to-donor SD of the log-odds share of the 13 PBMC cell types in "
    "Kang et al. 2018 (8 donors, PMID 29227470), estimated with this planner")


class PlannerError(ValueError):
    """A request the planner cannot serve (unknown column, no replicates): a 400."""


class NoReplicatesError(PlannerError):
    """The pilot has no replicate samples to estimate a spread from."""


# ---------------------------------------------------------------------------
# The pilot: samples x types, as composition_test forms them
# ---------------------------------------------------------------------------


@dataclass
class CompositionPilot:
    group_key: str
    condition_key: str | None
    sample_key: str | None
    conditions: list[str]  # reference first; [] without a condition column
    groups: list[str]
    counts: np.ndarray  # samples x groups
    s_cond: np.ndarray  # condition code per sample (0 = reference)
    s_donor: np.ndarray  # donor code per sample
    donors: list[str]
    design: str  # "paired" | "unpaired" | "single" (one condition) | "none" (no replicates)
    n_samples: dict[str, int]
    dropped_samples: list[str] = field(default_factory=list)

    @property
    def n_cells(self) -> np.ndarray:
        return self.counts.sum(axis=1)

    @property
    def replicated(self) -> bool:
        return self.design != "none"


def _two_conditions(obs: pd.DataFrame, condition_key: str, reference: str | None,
                    conditions: Sequence[str] | None) -> list[str]:
    cond = comp._labels(obs, condition_key)
    available = comp._order(obs[condition_key].where(cond.notna()))
    chosen = list(dict.fromkeys(conditions)) if conditions else available[:2]
    unknown = [c for c in chosen if c not in available]
    if unknown:
        raise PlannerError(f"'{unknown[0]}' is not a value of '{condition_key}'.")
    if reference is not None:
        if reference not in available:
            raise PlannerError(f"'{reference}' is not a value of '{condition_key}'.")
        chosen = [reference, *(c for c in (chosen if conditions else available) if c != reference)]
    chosen = chosen[:2]
    if len(chosen) < 2:
        raise PlannerError(f"'{condition_key}' has one value here: pick another column, or none.")
    return chosen


def composition_pilot(obs: pd.DataFrame, group_key: str, condition_key: str | None = None,
                      sample_key: str | None = None, *, reference: str | None = None,
                      conditions: Sequence[str] | None = None) -> CompositionPilot:
    """The pilot's per-sample type counts. With ``condition_key`` the two
    compared conditions (reference first), else all cells as one condition."""
    for col in (group_key, condition_key, sample_key):
        if col and col not in obs.columns:
            raise PlannerError(f"Column '{col}' not found.")
    if len({group_key, condition_key, sample_key} - {None}) < 1 + bool(condition_key) + bool(sample_key):
        raise PlannerError("The cell type, condition and donor columns must be different.")

    group = comp._labels(obs, group_key)
    keep = group.notna()
    cond = comp._labels(obs, condition_key) if condition_key else None
    chosen = _two_conditions(obs, condition_key, reference, conditions) if condition_key else []
    if cond is not None:
        keep &= cond.isin(chosen)
    sample = comp._labels(obs, sample_key) if sample_key else None
    if sample is not None:
        keep &= sample.notna()
    c = cond[keep].to_numpy() if cond is not None else np.full(int(keep.sum()), "", dtype=object)
    if not keep.any():
        raise PlannerError("No cells have all of the chosen columns set.")
    groups = comp._order(obs[group_key].where(keep))
    if len(groups) < 2:
        raise PlannerError(f"'{group_key}' has fewer than two cell types in these cells.")
    if len(groups) > comp.MAX_GROUPS:
        raise PlannerError(f"'{group_key}' has {len(groups)} groups; at most {comp.MAX_GROUPS}.")
    g_codes = pd.Categorical(group[keep], categories=groups).codes
    s = sample[keep].to_numpy() if sample is not None else np.full(len(c), "", dtype=object)
    ids = pd.Series(c, dtype=object).astype(str) + "\x1f" + pd.Series(s, dtype=object).astype(str)
    s_codes, s_names = pd.factorize(ids, sort=True)
    if len(s_names) > comp.MAX_SAMPLES:
        raise PlannerError(
            f"{len(s_names)} samples is too many (at most {comp.MAX_SAMPLES}); '{sample_key}' "
            "looks like a per-cell column rather than a donor or sample.")
    counts = np.zeros((len(s_names), len(groups)), dtype=np.int64)
    np.add.at(counts, (s_codes, g_codes), 1)
    sc = np.array([x.split("\x1f", 1)[0] for x in s_names], dtype=object)
    sd = np.array([x.split("\x1f", 1)[1] for x in s_names], dtype=object)
    n_per = counts.sum(axis=1)
    ok = n_per >= comp.MIN_CELLS_PER_SAMPLE
    dropped = [f"{sd[i]} ({sc[i] or 'all'}, {n_per[i]} cells)" for i in np.flatnonzero(~ok)] \
        if sample is not None else []
    if sample is not None:
        counts, sc, sd = counts[ok], sc[ok], sd[ok]
    cond_names = chosen or [""]
    n_samples = {k or "all": int((sc == k).sum()) for k in cond_names}
    s_cond = np.array([cond_names.index(k) for k in sc], dtype=np.int64)
    donors = sorted(set(sd))
    s_donor = np.array([donors.index(d) for d in sd], dtype=np.int64)

    if sample is None or min(n_samples.values()) < comp.MIN_SAMPLES:
        design = "none"
    elif not condition_key:
        design = "single"
    else:
        complete = all(len(set(s_cond[s_donor == d])) == 2 for d in range(len(donors)))
        design = "paired" if complete else "unpaired"
    return CompositionPilot(
        group_key=group_key, condition_key=condition_key, sample_key=sample_key,
        conditions=chosen, groups=groups, counts=counts, s_cond=s_cond, s_donor=s_donor,
        donors=donors, design=design, n_samples=n_samples, dropped_samples=dropped)


# ---------------------------------------------------------------------------
# Shared numerics
# ---------------------------------------------------------------------------

_GH_X, _GH_W = np.polynomial.hermite.hermgauss(40)
_GH_W = _GH_W / math.sqrt(math.pi)


def logit_location(share: np.ndarray, var: np.ndarray) -> np.ndarray:
    """mu with E[expit(mu + sqrt(var) Z)] = share, Z standard normal (Gauss-
    Hermite + bisection, elementwise): the logit-normal location whose mean
    share is ``share``."""
    share = np.clip(np.asarray(share, dtype=np.float64), 1e-12, 1 - 1e-12)
    s = np.sqrt(2 * np.maximum(np.asarray(var, dtype=np.float64), 0))[..., None]
    lo = np.full(share.shape, -40.0)
    hi = np.full(share.shape, 40.0)
    for _ in range(64):
        mid = (lo + hi) / 2
        mean = (special.expit(mid[..., None] + s * _GH_X) * _GH_W).sum(axis=-1)
        high = mean > share
        hi = np.where(high, mid, hi)
        lo = np.where(high, lo, mid)
    return (lo + hi) / 2


def _design(s_cond: np.ndarray, s_donor: np.ndarray, paired: bool, two: bool) -> np.ndarray:
    """composition_test's design: intercept, the tested condition, and the donors
    (all but the first) when paired."""
    cols = [np.ones(len(s_cond))]
    if two:
        cols.append((s_cond == 1).astype(float))
    if paired:
        cols += [(s_donor == d).astype(float) for d in np.unique(s_donor)[1:]]
    return np.column_stack(cols)


def _resid_var(Y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, int]:
    """Least-squares residual variance per column of ``Y`` (``(..., samples,
    groups)``) on design ``X``, as moderated_test's ``s2``; and its df."""
    Q, _ = np.linalg.qr(X)
    resid = Y - Q @ (np.swapaxes(Q, 0, 1) @ Y)
    df = X.shape[0] - np.linalg.matrix_rank(X)
    return (resid ** 2).sum(axis=-2) / max(df, 1), int(df)


def simulate_counts(rng: np.random.Generator, reps: int, mu: np.ndarray, s_cond: np.ndarray,
                    s_donor: np.ndarray, cells: np.ndarray, sd_within: np.ndarray,
                    sd_donor: np.ndarray) -> np.ndarray:
    """``reps`` simulated count tables (reps x samples x types): sample i's
    log-odds share of type j is ``mu[s_cond[i], j]`` + a donor effect (sd
    ``sd_donor``, shared by the donor's samples) + a residual (sd
    ``sd_within``); its count is binomial on ``cells[i]``."""
    n_s, G = len(s_cond), mu.shape[1]
    z = rng.standard_normal((reps, n_s, G)) * sd_within
    if np.any(sd_donor > 0):
        n_d = int(s_donor.max()) + 1
        z += (rng.standard_normal((reps, n_d, G)) * sd_donor)[:, s_donor, :]
    p = special.expit(mu[s_cond][None, :, :] + z)
    return rng.binomial(np.asarray(cells, dtype=np.int64)[None, :, None], p)


# ---------------------------------------------------------------------------
# The pilot's spread, by simulated moments
# ---------------------------------------------------------------------------


@dataclass
class TypeSpread:
    group: str
    share: float  # mean share over the reference samples (the baseline)
    n_cells: int  # in the pilot
    sd_within: float  # residual SD of the log-odds share (paired: within donor)
    sd_donor: float  # donor SD (paired pilots; 0 otherwise)
    s2: float  # the pilot's residual variance in the app's model (logit scale)
    noise_var: float  # of which counting noise (simulated)
    sd_low: float | None = None  # 95% interval of the spread a future design uses
    sd_high: float | None = None
    share_tested: float | None = None  # mean share in the other condition (pilot)

    @property
    def sd_total(self) -> float:
        return math.sqrt(self.sd_within ** 2 + self.sd_donor ** 2)


@dataclass
class CompositionModel:
    pilot: CompositionPilot | None
    groups: list[str]
    share: np.ndarray  # baseline share per type
    sd_within: np.ndarray
    sd_donor: np.ndarray
    spreads: list[TypeSpread]
    source: str  # "pilot" | "assumed"
    df: int | None  # residual df of the pilot fit
    df_unpaired: int | None = None
    cal_reps: int = 0
    assumed_sd: float | None = None

    @property
    def can_pair(self) -> bool:
        return self.pilot is not None and self.pilot.design == "paired" and self.source == "pilot"


def _cal_targets(pilot: CompositionPilot, transform: str):
    paired = pilot.design == "paired"
    two = pilot.design in ("paired", "unpaired")
    X = _design(pilot.s_cond, pilot.s_donor, paired, two)
    Y = comp.transform_props(pilot.counts, transform)
    s2, df = _resid_var(Y, X)
    if paired:
        Xu = _design(pilot.s_cond, pilot.s_donor, False, True)
        s2u, dfu = _resid_var(Y, Xu)
    else:
        Xu, s2u, dfu = None, None, None
    return X, s2, df, Xu, s2u, dfu


def fit_composition(pilot: CompositionPilot, *, transform: str = "logit", seed: int = 0,
                    cal_reps: int = CAL_REPS, rounds: int = CAL_ROUNDS) -> CompositionModel:
    """Each type's donor-to-donor spread on the logit scale, from the pilot's
    residual variances in the app's own model, counting noise removed by
    simulated moments (see the module docstring)."""
    if not pilot.replicated:
        raise NoReplicatesError(no_replicates_message(pilot))
    if transform not in ("logit", "asin"):
        raise PlannerError(f"Unknown transform '{transform}' (logit or asin).")
    X, s2, df, Xu, s2u, dfu = _cal_targets(pilot, transform)
    if df < 1:
        raise PlannerError("Too few samples for the pilot's design: no residual degrees of freedom.")
    n_cond = len(pilot.conditions) or 1
    counts, n = pilot.counts, pilot.n_cells
    props = counts / np.maximum(n, 1)[:, None]
    share_c = np.vstack([props[pilot.s_cond == k].mean(axis=0) for k in range(n_cond)])
    share_c = np.maximum(share_c, MIN_SHARE)
    paired = pilot.design == "paired"

    # Start: residual variance minus the empirical-logit counting variance.
    exp_n = n[:, None] * share_c[pilot.s_cond]
    noise0 = (1 / (exp_n + 0.5) + 1 / (n[:, None] - exp_n + 0.5)).mean(axis=0)
    sw2 = np.maximum(s2 - noise0, 0.0)
    sd2 = np.maximum(s2u - s2, 0.0) if paired else np.zeros_like(s2)
    for _ in range(rounds):
        rng = np.random.default_rng([seed, 7919])  # the same draws every round
        mu = logit_location(share_c, (sw2 + sd2)[None, :])
        sim = simulate_counts(rng, cal_reps, mu, pilot.s_cond, pilot.s_donor, n,
                              np.sqrt(sw2), np.sqrt(sd2))
        Ysim = comp.transform_props(sim.reshape(-1, sim.shape[-1]), transform).reshape(sim.shape)
        s2_sim = _resid_var(Ysim, X)[0].mean(axis=0)
        sw2 = np.maximum(sw2 + (s2 - s2_sim), 0.0)
        if paired:
            gap_sim = _resid_var(Ysim, Xu)[0].mean(axis=0) - s2_sim
            sd2 = np.maximum(sd2 + ((s2u - s2) - gap_sim), 0.0)
    noise = np.maximum(s2_sim - sw2, 0.0)

    spreads = []
    lo_q, hi_q = stats.chi2.ppf([0.975, 0.025], df)
    for j, g in enumerate(pilot.groups):
        # 95% interval of the residual variance (chi-square on its df), counting
        # noise taken off: the within-donor spread a paired design sees.
        lo = max(df * s2[j] / lo_q - noise[j], 0.0)
        hi = max(df * s2[j] / hi_q - noise[j], 0.0)
        spreads.append(TypeSpread(
            group=g, share=float(share_c[0, j]), n_cells=int(counts[:, j].sum()),
            sd_within=float(math.sqrt(sw2[j])), sd_donor=float(math.sqrt(sd2[j])),
            s2=float(s2[j]), noise_var=float(noise[j]),
            sd_low=math.sqrt(lo), sd_high=math.sqrt(hi),
            share_tested=float(share_c[1, j]) if n_cond > 1 else None))
    return CompositionModel(
        pilot=pilot, groups=list(pilot.groups), share=share_c[0].copy(),
        sd_within=np.sqrt(sw2), sd_donor=np.sqrt(sd2), spreads=spreads, source="pilot",
        df=df, df_unpaired=dfu, cal_reps=cal_reps)


def spread_interval(model: CompositionModel, design: str,
                    transform: str = "logit") -> tuple[np.ndarray, np.ndarray]:
    """Per type, the 95% interval (low, high) of the spread ``design`` sees: the
    within-donor SD for a paired design, the total for an unpaired one."""
    if model.source != "pilot":
        sd = np.array([s.sd_total for s in model.spreads])
        return sd, sd
    pilot = model.pilot
    if design == "paired" or pilot.design != "paired":
        return (np.array([s.sd_low for s in model.spreads]),
                np.array([s.sd_high for s in model.spreads]))
    # Unpaired from a paired pilot: the total spread, from the condition-only fit
    # (its df treat the paired samples as independent: approximate).
    _, _, _, _, s2u, dfu = _cal_targets(pilot, transform)
    lo_q, hi_q = stats.chi2.ppf([0.975, 0.025], dfu)
    lo = np.zeros(len(model.groups))
    hi = np.zeros(len(model.groups))
    for j, s in enumerate(model.spreads):
        lo[j] = math.sqrt(min(max(dfu * s2u[j] / lo_q - s.noise_var, 0.0), s.sd_total ** 2))
        hi[j] = math.sqrt(max(dfu * s2u[j] / hi_q - s.noise_var, s.sd_total ** 2))
    return lo, hi


def assumed_composition(pilot: CompositionPilot, sd: float) -> CompositionModel:
    """A model with one entered spread for every type (no replicates, or the
    user's own value): pooled shares of the reference cells, unpaired."""
    if not (0 <= sd <= 5):
        raise PlannerError("The entered spread must be a log-odds SD between 0 and 5.")
    counts = pilot.counts[pilot.s_cond == 0].sum(axis=0)
    share = np.maximum(counts / max(counts.sum(), 1), MIN_SHARE)
    spreads = [TypeSpread(group=g, share=float(share[j]), n_cells=int(pilot.counts[:, j].sum()),
                          sd_within=float(sd), sd_donor=0.0, s2=float("nan"), noise_var=0.0,
                          sd_low=None, sd_high=None)
               for j, g in enumerate(pilot.groups)]
    return CompositionModel(pilot=pilot, groups=list(pilot.groups), share=share,
                            sd_within=np.full(len(share), float(sd)),
                            sd_donor=np.zeros(len(share)), spreads=spreads, source="assumed",
                            df=None, assumed_sd=float(sd))


def no_replicates_message(pilot: CompositionPilot) -> str:
    if pilot.sample_key is None:
        what = "no donor or sample column"
    else:
        per = ", ".join(f"{k}: {v}" for k, v in pilot.n_samples.items())
        what = (f"fewer than {comp.MIN_SAMPLES} samples of '{pilot.sample_key}' per condition "
                f"({per}, each with {comp.MIN_CELLS_PER_SAMPLE}+ cells)")
    return (f"This dataset has {what}, so it holds no donor-to-donor variability to plan "
            "from: a power estimate needs a pilot with replicate donors (at least two per "
            "condition). The composition planner can still run on an entered spread, "
            f"labelled as an assumption (default {ASSUMED_SD}, the {ASSUMED_SD_SOURCE}).")


# ---------------------------------------------------------------------------
# Composition power
# ---------------------------------------------------------------------------


@dataclass
class CompositionGrid:
    targets: list[int]  # type indices
    fold_changes: list[float]
    donors: list[int]  # per group
    cells: list[int]  # per sample
    design: str  # "paired" | "unpaired"
    alpha: float = ALPHA
    reps: int = 100
    transform: str = "logit"
    robust: bool = True
    seed: int = 0
    pessimistic: bool = True  # also at the upper end of the spread's interval

    @property
    def middle_cells(self) -> int:
        return self.cells[len(self.cells) // 2]


def pick_targets(model: CompositionModel, wanted: Sequence[str] | None = None) -> list[int]:
    """The requested types, else three: the most abundant, the one nearest the
    median share and the rarest with at least 0.5% of the cells."""
    if wanted:
        missing = [w for w in wanted if w not in model.groups]
        if missing:
            raise PlannerError(f"'{missing[0]}' is not a group of '{model.pilot.group_key}'.")
        out = [model.groups.index(w) for w in dict.fromkeys(wanted)]
        if len(out) > MAX_TARGETS:
            raise PlannerError(f"Plan for at most {MAX_TARGETS} cell types at once.")
        return out
    share = model.share
    order = np.argsort(-share, kind="stable")
    abundant = int(order[0])
    mid = int(np.argmin(np.abs(np.log(share) - np.median(np.log(share)))))
    rare_ok = [int(j) for j in order[::-1] if share[j] >= 0.005]
    rare = rare_ok[0] if rare_ok else int(order[-1])
    return list(dict.fromkeys([abundant, mid, rare]))


def auto_reps(n_points: int, n_groups: int, robust: bool) -> int:
    cost = _COST_ROBUST if robust and n_groups >= 10 else _COST_PLAIN
    reps = int(TIME_BUDGET_S / max(n_points, 1) / cost)
    return int(min(MAX_REPS, max(MIN_REPS, reps // 10 * 10)))


def shifted_shares(share: np.ndarray, target: int, fc: float) -> np.ndarray | None:
    """The tested group's mean shares: the target's times ``fc``, the others
    scaled to make room; None when the target would pass MAX_SHARE."""
    new_t = share[target] * fc
    if fc <= 0 or new_t >= MAX_SHARE:
        return None
    out = share * (1 - new_t) / (1 - share[target])
    out[target] = new_t
    return np.maximum(out, MIN_SHARE)


def _future_design(design: str, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(condition code, donor code, X) of a future study, n donors per group."""
    if design == "paired":
        s_cond = np.tile([0, 1], n)
        s_donor = np.repeat(np.arange(n), 2)
    else:
        s_cond = np.repeat([0, 1], n)
        s_donor = np.arange(2 * n)
    return s_cond, s_donor, _design(s_cond, s_donor, design == "paired", True)


@dataclass
class PowerPoint:
    group: str
    fold_change: float
    donors: int
    cells: int
    power: float | None  # None: infeasible fold change
    se: float | None
    power_raw: float | None = None  # unadjusted p < alpha (calibration)
    pessimistic: bool = False


def power_point(model: CompositionModel, grid: CompositionGrid, target: int, fc: float,
                n: int, m: int, *, sd_scale: np.ndarray | None = None) -> PowerPoint:
    """Power to detect ``fc`` in type ``target`` with n donors per group and m
    cells per sample, by ``grid.reps`` simulated studies each run through the
    app's composition test (transform_props, moderated_test, BH)."""
    name = model.groups[target]
    shares1 = shifted_shares(model.share, target, fc)
    if shares1 is None:
        return PowerPoint(name, fc, n, m, None, None, pessimistic=sd_scale is not None)
    design = grid.design if (grid.design == "unpaired" or model.can_pair) else "unpaired"
    if design == "paired":
        sw, sdn = model.sd_within.copy(), model.sd_donor.copy()
    else:
        sw, sdn = np.sqrt(model.sd_within ** 2 + model.sd_donor ** 2), np.zeros(len(model.share))
    if sd_scale is not None:  # pessimistic: the spread at its interval's upper end
        sw = np.where(np.isfinite(sd_scale), np.maximum(sd_scale, sw), sw)
    var = sw ** 2 + sdn ** 2
    mu = np.vstack([logit_location(model.share, var), logit_location(shares1, var)])
    s_cond, s_donor, X = _future_design(design, n)
    rng = np.random.default_rng([grid.seed, n, m, target, int(sd_scale is not None)])
    sim = simulate_counts(rng, grid.reps, mu, s_cond, s_donor, np.full(len(s_cond), m), sw, sdn)
    direction = np.sign(math.log(fc)) if fc != 1 else 0.0
    hits = raw = done = 0
    for r in range(grid.reps):
        Y = comp.transform_props(sim[r], grid.transform)
        try:
            fit = comp.moderated_test(Y, X, [1], robust=grid.robust)
        except comp.CompositionError:
            done += 1
            continue
        done += 1
        fdr = comp._bh(fit.pval)
        right = direction == 0 or np.sign(fit.stat[target]) == direction
        hits += bool(right and fdr[target] < grid.alpha)
        raw += bool(right and fit.pval[target] < grid.alpha)
    power = hits / max(done, 1)
    return PowerPoint(name, fc, n, m, power, math.sqrt(power * (1 - power) / max(done, 1)),
                      power_raw=raw / max(done, 1), pessimistic=sd_scale is not None)


def composition_points(grid: CompositionGrid) -> int:
    n = len(grid.targets) * len(grid.fold_changes) * len(grid.donors)
    return n * len(grid.cells) + (n if grid.pessimistic else 0)


def iter_composition_power(model: CompositionModel, grid: CompositionGrid,
                           stop: Callable[[], bool] | None = None) -> Iterator[PowerPoint]:
    """Every grid point, then (``grid.pessimistic``, pilot spreads only) each
    type x fold change again at the middle cells value with the spread at the
    upper end of its 95% interval."""
    for t in grid.targets:
        for fc in grid.fold_changes:
            for m in grid.cells:
                for n in grid.donors:
                    if stop and stop():
                        return
                    yield power_point(model, grid, t, fc, n, m)
    if not grid.pessimistic or model.source != "pilot":
        return
    design = grid.design if model.can_pair else "unpaired"
    upper = spread_interval(model, design, grid.transform)[1]
    for t in grid.targets:
        for fc in grid.fold_changes:
            for n in grid.donors:
                if stop and stop():
                    return
                yield power_point(model, grid, t, fc, n, grid.middle_cells, sd_scale=upper)


def donors_needed(points: Sequence[PowerPoint], target: float = TARGET_POWER) -> float | None:
    """Donors per group where the power curve (points of one type, fold change
    and cells value) first reaches ``target``, linearly interpolated between
    grid values and rounded up; None when the largest grid value falls short."""
    pts = sorted((p for p in points if p.power is not None), key=lambda p: p.donors)
    prev = None
    for p in pts:
        if p.power >= target:
            if prev is None or prev.power >= target:
                return float(p.donors)
            frac = (target - prev.power) / max(p.power - prev.power, 1e-9)
            # Rounded first: 8 + 0.5 * 4 can come out as 10.000000000000007.
            return float(math.ceil(round(prev.donors + frac * (p.donors - prev.donors), 6)))
        prev = p
    return None


# ---------------------------------------------------------------------------
# Pseudobulk DE: one cell type's per-sample count sums
# ---------------------------------------------------------------------------
#
# The pilot: the chosen type's raw counts summed per sample (a donor in a
# condition), as core/pseudobulk.py sums them, plus each sample's per-cell
# sums of squares. Per gene, from those: the pseudobulk dispersion (DESeq2's
# estimator, below; within donor when the donors pair up), a donor effect (the
# extra dispersion of the condition-only fit) and the cells' own overdispersion
# within a sample. Summing m cells whose counts have mean q and variance
# q + a_cell q^2 gives a sample variance mq + (a_cell / m) (mq)^2, so the
# dispersion a study with m cells per sample sees is a_bio + a_cell / m: more
# cells raise the counts (less Poisson noise) and average out the cells' own
# spread, and leave the donors' spread (a_bio, the donor effect) untouched.
# hierarchicell (Zimmerman et al. 2021, PMID 33531494) and scPower (Schmid et
# al. 2021, PMID 34785648) model pseudobulk power the same way: the donor is
# the replicate, the cells only sharpen each donor's mean.
#
# The test, on the pilot and on every simulated study: DESeq2's negative
# binomial Wald test as pydeseq2 runs it for the app's pseudobulk DE (median-
# of-ratios size factors; method-of-moments start; Cox-Reid profile-likelihood
# gene-wise dispersions; parametric mean-dispersion trend; log-normal prior
# with MAD variance, floor 0.25; MAP dispersions, gene-wise kept for genes far
# above the trend; Wald z on the condition coefficient; Benjamini-Hochberg),
# vectorised over genes for the two designs the app uses (``~condition`` and
# ``~donor + condition``). Left out: Cook's-distance refits and independent
# filtering. tests/test_study_planner.py checks it against pydeseq2.

DE_DEFAULT_CELLS = (50, 200, 1000)  # cells of the type per sample
DE_DEFAULT_LOG2FC = (0.5, 1.0)
DE_SHARE = 0.10  # share of the simulated genes that change
DE_GENES = 2000  # genes simulated per study (a random draw of the pilot's tested genes)
DE_MIN_REPS, DE_MAX_REPS = 4, 40
DE_TIME_BUDGET_S = 45.0
# Seconds per simulated study per (gene x sample), measured 2026-10-22 on 2 cores
# (Kang CD14 Mono, 2,000 genes): ~3.1e-6 paired, ~1.4e-6 unpaired.
_DE_COST = {"paired": 3.2e-6, "unpaired": 1.5e-6}
MIN_DISP = 1e-8
MIN_MU = 0.5  # pydeseq2's floor on fitted means
_CHUNK_ELEMENTS = 8_000_000  # dense rows read per chunk: ~64 MB as float64


@dataclass
class _Design:
    """A design as blocks + an optional shared condition coefficient: paired =
    one block per donor and the condition shared (``~donor + condition``);
    unpaired = one block per condition (``~condition``); single = one block."""

    block: np.ndarray  # per sample
    cond: np.ndarray  # per sample, 0/1
    n_blocks: int
    paired: bool
    two: bool

    @property
    def p(self) -> int:
        return self.n_blocks + int(self.paired)

    @property
    def B(self) -> np.ndarray:
        return (self.block[:, None] == np.arange(self.n_blocks)[None, :]).astype(np.float64)

    @classmethod
    def of(cls, s_cond: np.ndarray, s_donor: np.ndarray, kind: str) -> _Design:
        s_cond = np.asarray(s_cond, dtype=np.int64)
        if kind == "paired":
            codes = pd.factorize(np.asarray(s_donor), sort=True)[0]
            return cls(codes, s_cond, int(codes.max()) + 1, True, True)
        if kind == "unpaired":
            return cls(s_cond.copy(), s_cond, 2, False, True)
        return cls(np.zeros(len(s_cond), dtype=np.int64), s_cond, 1, False, False)


def size_factors(counts: np.ndarray) -> np.ndarray:
    """DESeq2's median-of-ratios size factors (samples x genes counts), over the
    genes with no zero; with none, over each sample's positive counts."""
    logc = np.log(np.where(counts > 0, counts, np.nan))
    full = np.all(counts > 0, axis=0)
    if full.any():
        lg = logc[:, full].mean(axis=0)
        sf = np.exp(np.median(logc[:, full] - lg, axis=1))
    else:  # poscounts-like fallback
        lg = np.nanmean(logc, axis=0)
        ok = np.isfinite(lg)
        sf = np.exp(np.nanmedian(logc[:, ok] - lg[ok], axis=1))
        sf = sf / np.exp(np.mean(np.log(sf)))
    return sf


def _info(w: np.ndarray, d: _Design, B: np.ndarray):
    """Fisher information blocks from weights ``w`` (..., samples)."""
    Iaa = w @ B
    if not d.paired:
        return Iaa, None, None
    wc = w * d.cond
    Iab = wc @ B
    Ibb = wc.sum(axis=-1)
    return Iaa, Iab, Ibb


def _logdet(Iaa, Iab, Ibb) -> np.ndarray:
    out = np.log(np.maximum(Iaa, 1e-300)).sum(axis=-1)
    if Iab is not None:
        out = out + np.log(np.maximum(Ibb - (Iab ** 2 / np.maximum(Iaa, 1e-300)).sum(axis=-1), 1e-300))
    return out


def _nb_irls(Y: np.ndarray, logs: np.ndarray, d: _Design, alpha: np.ndarray,
             a: np.ndarray | None = None, b: np.ndarray | None = None,
             iters: int = 50, tol: float = 1e-7):
    """Negative binomial GLM by Fisher scoring, genes along axis 0 (``Y`` genes x
    samples), log link, offsets ``logs``, fixed dispersions ``alpha``. Returns
    (block coefficients, condition coefficient, mu, weights)."""
    B = d.B
    G = Y.shape[0]
    if a is None:
        norm = Y / np.exp(logs)
        a = np.log(np.maximum((norm @ B) / np.maximum(B.sum(axis=0), 1), 1e-3))
        b = np.zeros(G)
    if b is None:
        b = np.zeros(G)
    al = alpha[:, None]
    for _ in range(iters):
        eta = logs + a[:, d.block] + (b[:, None] * d.cond if d.paired else 0.0)
        mu = np.maximum(np.exp(np.minimum(eta, 50)), MIN_MU)
        w = mu / (1 + al * mu)
        r = (Y - mu) / (1 + al * mu)
        Ua = r @ B
        Iaa, Iab, Ibb = _info(w, d, B)
        Iaa = np.maximum(Iaa, 1e-12)
        if d.paired:
            Ub = (r * d.cond).sum(axis=1)
            schur = np.maximum(Ibb - (Iab ** 2 / Iaa).sum(axis=1), 1e-12)
            db = (Ub - (Iab * Ua / Iaa).sum(axis=1)) / schur
            da = (Ua - Iab * db[:, None]) / Iaa
        else:
            db = np.zeros(G)
            da = Ua / Iaa
        da = np.clip(da, -5, 5)
        db = np.clip(db, -5, 5)
        a = np.clip(a + da, -40, 40)
        b = np.clip(b + db, -40, 40)
        if max(float(np.abs(da).max(initial=0)), float(np.abs(db).max(initial=0))) < tol:
            break
    eta = logs + a[:, d.block] + (b[:, None] * d.cond if d.paired else 0.0)
    mu = np.maximum(np.exp(np.minimum(eta, 50)), MIN_MU)
    return a, b, mu, mu / (1 + al * mu)


def _disp_objective(Y: np.ndarray, mu: np.ndarray, d: _Design, B: np.ndarray,
                    log_alpha: np.ndarray, prior: tuple[np.ndarray, float] | None = None
                    ) -> np.ndarray:
    """Cox-Reid adjusted profile log likelihood of the dispersion (DESeq2's), at
    ``log_alpha`` (A x genes), mu fixed; plus the log-normal prior when given."""
    alpha = np.exp(log_alpha)[..., None]  # A x G x 1
    r = 1 / alpha
    am = alpha * mu[None]
    ll = (special.gammaln(Y[None] + r) - special.gammaln(r) - r * np.log1p(am)
          + Y[None] * (np.log(am) - np.log1p(am))).sum(axis=-1)
    w = mu[None] / (1 + am)
    f = ll - 0.5 * _logdet(*_info(w, d, B))
    if prior is not None:
        trend, var = prior
        f = f - (log_alpha - np.log(trend)) ** 2 / (2 * var)
    return f


_GRID_CHUNK = 4_000_000  # grid points x genes x samples evaluated at once (~32 MB a temporary)


def _chunked(fun: Callable[[slice], np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    """``fun`` over slices of the genes (axis 0 of ``shape``) small enough that a
    17-point grid over them stays under ``_GRID_CHUNK`` elements."""
    G, n = shape
    step = max(1, _GRID_CHUNK // (17 * max(n, 1)))
    return np.concatenate([fun(slice(i, min(i + step, G))) for i in range(0, G, step)]) \
        if G else np.zeros(0)


def _grid_argmax(fun: Callable[[np.ndarray], np.ndarray], lo: np.ndarray, hi: np.ndarray,
                 points: Sequence[int] = (17, 9, 9)) -> np.ndarray:
    """Per gene, the maximiser of ``fun`` over [lo, hi] by successively finer grids."""
    lo, hi = lo.astype(np.float64).copy(), hi.astype(np.float64).copy()
    cols = np.arange(len(lo))
    floor, ceil = lo.copy(), hi.copy()
    best = (lo + hi) / 2
    for n in points:
        grid = lo[None] + (hi - lo)[None] * np.linspace(0, 1, n)[:, None]
        k = np.argmax(fun(grid), axis=0)
        best = grid[k, cols]
        step = (hi - lo) / (n - 1)
        lo, hi = np.maximum(best - step, floor), np.minimum(best + step, ceil)
    return best


def _trend(base_mean: np.ndarray, disp: np.ndarray) -> tuple[np.ndarray, str]:
    """DESeq2's parametric trend a0 + a1 / mean (gamma GLM, identity link,
    refitted without genes far off the curve); the mean when that fails."""
    use = (disp >= 100 * MIN_DISP) & (base_mean > 0)
    x, y = 1 / base_mean[use], disp[use]
    coef = None
    keep = np.ones(len(x), dtype=bool)
    old = np.array([0.1, 1.0])
    for _ in range(10):
        if keep.sum() < 3:
            coef = None
            break
        X = np.column_stack([np.ones(keep.sum()), x[keep]])
        yy = y[keep]
        c = np.array([max(float(yy.mean()), 1e-6), 0.0])
        for _ in range(50):  # IRLS, gamma family, identity link: weights 1 / fitted^2
            pred = np.maximum(X @ c, 1e-12)
            Wt = 1 / pred ** 2
            c_new = np.linalg.solve(X.T @ (X * Wt[:, None]), X.T @ (yy * Wt))
            if np.all(np.abs(c_new - c) <= 1e-8 * (np.abs(c) + 1e-8)):
                c = c_new
                break
            c = c_new
        if not np.all(c > 1e-10):
            coef = None
            break
        coef = c
        ratio = y / (coef[0] + coef[1] * x)
        keep = (ratio >= 1e-4) & (ratio < 15)
        if np.sum(np.log(np.abs(coef / old)) ** 2) < 1e-6:
            break
        old = coef
    if coef is not None:
        return coef[0] + coef[1] / np.maximum(base_mean, 1e-12), "parametric"
    m = float(stats.trim_mean(y, 0.001)) if len(y) else 0.1
    return np.full(len(base_mean), m), "mean"


@dataclass
class NBResult:
    log2fc: np.ndarray
    stat: np.ndarray
    pval: np.ndarray
    padj: np.ndarray
    dispersion: np.ndarray  # MAP (gene-wise for dispersion outliers)
    dispersion_genewise: np.ndarray
    base_mean: np.ndarray
    size_factors: np.ndarray
    trend: str


def nb_wald_test(counts: np.ndarray, s_cond: np.ndarray, s_donor: np.ndarray,
                 kind: str) -> NBResult:
    """DESeq2's pipeline (see above) on ``counts`` (samples x genes): the Wald
    test of the tested condition (code 1) vs the reference (0) for ``kind``
    "paired" or "unpaired"; "single" fits the dispersions only (no test)."""
    K = np.asarray(counts, dtype=np.float64)
    n = K.shape[0]
    d = _Design.of(s_cond, s_donor, kind)
    B = d.B
    if n - d.p < 1:
        raise PlannerError("No residual degrees of freedom for the test.")
    sf = size_factors(K)
    logs = np.log(sf)
    Y = K.T  # genes x samples
    norm = Y / sf
    base_mean = norm.mean(axis=1)

    # Method-of-moments start: min of the rough (linear model) and moments estimates.
    X = np.column_stack([B] + ([d.cond.astype(float)] if d.paired else []))
    Q, _ = np.linalg.qr(X)
    yhat = norm @ Q @ Q.T
    rough = np.maximum((((norm - np.maximum(yhat, 1)) ** 2 - np.maximum(yhat, 1))
                        / np.maximum(yhat, 1) ** 2).sum(axis=1) / (n - d.p), 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        moments = np.nan_to_num((norm.var(axis=1, ddof=1) - (1 / sf).mean() * base_mean)
                                / base_mean ** 2)
    max_disp = max(10.0, float(n))
    alpha0 = np.clip(np.minimum(rough, moments), MIN_DISP, max_disp)

    # mu for the dispersion fits: a linear model when the design is saturated.
    if not d.paired:
        mu = np.maximum(np.maximum(yhat, 0) * sf, MIN_MU)
    else:
        _, _, mu, _ = _nb_irls(Y, logs, d, alpha0)
    lo = np.full(len(base_mean), math.log(MIN_DISP))
    hi = np.full(len(base_mean), math.log(max_disp))
    lg = _chunked(lambda sl: _grid_argmax(
        lambda la: _disp_objective(Y[sl], mu[sl], d, B, la), lo[sl], hi[sl]), Y.shape)
    gw = np.exp(lg)

    trend, kind_trend = _trend(base_mean, gw)
    use = gw >= 100 * MIN_DISP
    resid = np.log(gw[use]) - np.log(trend[use])
    mad = float(np.median(np.abs(resid - np.median(resid))) / stats.norm.ppf(0.75)) if use.any() else 0.0
    prior_var = max(mad ** 2 - float(special.polygamma(1, (n - d.p) / 2)), 0.25)
    lt = np.log(trend)
    lo2 = np.maximum(np.minimum(lg, lt) - 1, math.log(MIN_DISP))
    hi2 = np.minimum(np.maximum(lg, lt) + 1, math.log(max_disp))
    lmap = _chunked(lambda sl: _grid_argmax(
        lambda la: _disp_objective(Y[sl], mu[sl], d, B, la, (trend[sl], prior_var)),
        lo2[sl], hi2[sl], points=(13, 9, 9)), Y.shape)
    outlier = lg > lt + 2 * math.sqrt(prior_var)
    disp = np.where(outlier, gw, np.exp(lmap))

    G = Y.shape[0]
    if not d.two:
        nan = np.full(G, np.nan)
        return NBResult(nan, nan, nan, nan, disp, gw, base_mean, sf, kind_trend)
    a, b, mu, w = _nb_irls(Y, logs, d, disp)
    Iaa, Iab, Ibb = _info(w, d, B)
    if d.paired:
        lfc = b
        var = 1 / np.maximum(Ibb - (Iab ** 2 / np.maximum(Iaa, 1e-300)).sum(axis=1), 1e-300)
    else:
        lfc = a[:, 1] - a[:, 0]
        var = 1 / np.maximum(Iaa[:, 0], 1e-300) + 1 / np.maximum(Iaa[:, 1], 1e-300)
    stat = lfc / np.sqrt(var)
    pval = 2 * stats.norm.sf(np.abs(stat))
    return NBResult(lfc / math.log(2), stat, pval, comp._bh(pval), disp, gw, base_mean, sf,
                    kind_trend)


@dataclass
class DEPilot:
    group_key: str
    group: str
    condition_key: str | None
    sample_key: str | None
    conditions: list[str]
    s_cond: np.ndarray
    s_donor: np.ndarray
    donors: list[str]
    design: str  # "paired" | "unpaired" | "single" | "none"
    n_cells: np.ndarray  # cells of the type per sample
    sums: np.ndarray  # samples x genes: summed counts (all genes)
    sumsq: np.ndarray  # samples x genes: summed squared counts
    gene_names: np.ndarray
    tested: np.ndarray  # gene indices passing the app's pseudobulk filter
    counts_source: str
    n_samples: dict[str, int]
    dropped_samples: list[str] = field(default_factory=list)


def _sample_sums(M, rows: np.ndarray, codes: np.ndarray, n_samples: int
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Per sample, the sums and the sums of squares of rows ``rows`` of ``M``
    (cells x genes) by ``codes``; dense matrices in row chunks, never whole."""
    import scipy.sparse as sps

    n_genes = M.shape[1]
    S1 = np.zeros((n_samples, n_genes))
    S2 = np.zeros((n_samples, n_genes))
    if sps.issparse(M):
        sub = sps.csr_matrix(M[rows]) if len(rows) < M.shape[0] else sps.csr_matrix(M)
        ind = sps.csr_matrix((np.ones(len(codes)), (codes, np.arange(len(codes)))),
                             shape=(n_samples, len(codes)))
        S1 += (ind @ sub).toarray()
        sq = sub.copy()
        sq.data = sq.data.astype(np.float64) ** 2
        S2 += (ind @ sq).toarray()
        return S1, S2
    step = max(64, _CHUNK_ELEMENTS // max(n_genes, 1))
    for start in range(0, len(rows), step):
        r = rows[start:start + step]
        block = np.asarray(M[r], dtype=np.float64)
        c = codes[start:start + step]
        np.add.at(S1, c, block)
        np.add.at(S2, c, block * block)
    return S1, S2


def de_pilot(adata, group_key: str, group: str, condition_key: str | None = None,
             sample_key: str | None = None, *, reference: str | None = None,
             conditions: Sequence[str] | None = None) -> DEPilot:
    """The chosen cell type's per-sample count sums (raw counts, as the app's
    pseudobulk DE takes them) and the app's gene filter on them."""
    from scview.core import pseudobulk

    obs = adata.obs
    for col in (group_key, condition_key, sample_key):
        if col and col not in obs.columns:
            raise PlannerError(f"Column '{col}' not found.")
    if len({group_key, condition_key, sample_key} - {None}) < 1 + bool(condition_key) + bool(sample_key):
        raise PlannerError("The cell type, condition and donor columns must be different.")
    labels = comp._labels(obs, group_key)
    if group not in set(labels.dropna()):
        raise PlannerError(f"'{group}' is not a group of '{group_key}'.")
    keep = (labels == group).to_numpy()
    chosen = _two_conditions(obs, condition_key, reference, conditions) if condition_key else []
    cond = comp._labels(obs, condition_key) if condition_key else None
    if cond is not None:
        keep &= cond.isin(chosen).to_numpy()
    sample = comp._labels(obs, sample_key) if sample_key else None
    if sample is not None:
        keep &= sample.notna().to_numpy()
    rows = np.flatnonzero(keep)
    if len(rows) == 0:
        raise PlannerError(f"No '{group}' cells have all of the chosen columns set.")
    c = cond.to_numpy()[rows] if cond is not None else np.full(len(rows), "", dtype=object)
    s = sample.to_numpy()[rows] if sample is not None else np.full(len(rows), "", dtype=object)
    ids = pd.Series(c, dtype=object).astype(str) + "\x1f" + pd.Series(s, dtype=object).astype(str)
    codes, names = pd.factorize(ids, sort=True)
    if len(names) > pseudobulk.MAX_SAMPLES:
        raise PlannerError(f"{len(names)} samples is too many (at most {pseudobulk.MAX_SAMPLES}).")
    sc = np.array([x.split("\x1f", 1)[0] for x in names], dtype=object)
    sd = np.array([x.split("\x1f", 1)[1] for x in names], dtype=object)
    n_cells = np.bincount(codes, minlength=len(names))
    ok = n_cells >= pseudobulk.MIN_CELLS_PER_SAMPLE
    dropped = [f"{sd[i]} ({sc[i] or 'all'}, {n_cells[i]} cells)" for i in np.flatnonzero(~ok)] \
        if sample is not None else []
    cond_names = chosen or [""]
    n_samples = {k or "all": int(((sc == k) & ok).sum()) for k in cond_names}
    if sample is None or min(n_samples.values()) < pseudobulk.MIN_REPLICATES:
        design = "none"
    elif not condition_key:
        design = "single"
    else:
        kept_d = sd[ok]
        kept_c = sc[ok]
        complete = all(len(set(kept_c[kept_d == d])) == 2 for d in set(kept_d))
        design = "paired" if complete else "unpaired"
    s_cond_all = np.array([cond_names.index(k) for k in sc], dtype=np.int64)
    if design == "none":  # before any counts are read
        return DEPilot(group_key, group, condition_key, sample_key, chosen, s_cond_all, sd,
                       sorted(set(sd)), design, n_cells, np.zeros((len(names), 0)),
                       np.zeros((len(names), 0)), np.asarray(adata.var_names),
                       np.zeros(0, dtype=int), "", n_samples, dropped)
    try:
        M, gene_names, source = pseudobulk.raw_counts(adata)
    except pseudobulk.PseudobulkError as exc:
        raise PlannerError(str(exc)) from exc
    S1, S2 = _sample_sums(M, rows, codes, len(names))
    S1, S2, sc, sd, n_cells = S1[ok], S2[ok], sc[ok], sd[ok], n_cells[ok]
    s_cond = np.array([cond_names.index(k) for k in sc], dtype=np.int64)
    tested = pseudobulk._filter_genes(S1, min(n_samples.values()))
    if len(tested) < 50:
        raise PlannerError(
            f"Only {len(tested)} genes have {pseudobulk.MIN_GENE_COUNT}+ counts in enough "
            f"'{group}' samples: too little signal to plan from. Pick a more abundant type.")
    donors = sorted(set(sd))
    s_donor = np.array([donors.index(x) for x in sd], dtype=np.int64)
    return DEPilot(group_key, group, condition_key, sample_key, chosen, s_cond, s_donor, donors,
                   design, n_cells, S1, S2, np.asarray(gene_names), tested, source, n_samples,
                   dropped)


@dataclass
class DEModel:
    pilot: DEPilot
    genes: np.ndarray  # names of the simulated genes
    q: np.ndarray  # mean count per cell (reference samples, normalised)
    alpha_bio: np.ndarray  # pseudobulk dispersion that more cells do not remove
    alpha_cell: np.ndarray  # the cells' own overdispersion within a sample
    tau2: np.ndarray  # donor effect variance (log scale; paired pilots)
    alpha_pilot: np.ndarray  # MAP pseudobulk dispersion of the pilot's design
    n_tested: int  # pilot genes passing the app's filter
    pilot_n_significant: int | None  # the fast test on the pilot itself (FDR < 0.05)
    median_cells: float
    trend: str


def fit_de(pilot: DEPilot, *, n_genes: int = DE_GENES, seed: int = 0) -> DEModel:
    """Per gene: per-cell mean, pilot dispersion (DESeq2's), donor effect and
    the cells' own overdispersion; ``n_genes`` of the tested genes drawn for
    the simulation."""
    if pilot.design == "none":
        raise NoReplicatesError(
            f"'{pilot.sample_key or 'no donor column'}': fewer than 2 samples of "
            f"{pilot.group} per condition (each with 10+ cells). A DE power estimate needs a "
            "pilot with replicate donors; this dataset has none to estimate their spread from.")
    K = np.round(pilot.sums[:, pilot.tested]).astype(np.int64)
    kind = pilot.design if pilot.design != "single" else "single"
    res = nb_wald_test(K, pilot.s_cond, pilot.s_donor, kind)
    tau2 = np.zeros(K.shape[1])
    if pilot.design == "paired":
        tot = nb_wald_test(K, pilot.s_cond, pilot.s_donor, "unpaired")
        tau2 = np.maximum(np.log((1 + tot.dispersion) / (1 + res.dispersion)), 0)
    n = pilot.n_cells.astype(np.float64)[:, None]
    S1 = pilot.sums[:, pilot.tested]
    S2 = pilot.sumsq[:, pilot.tested]
    mean = S1 / n
    var = np.maximum(S2 - n * mean ** 2, 0) / np.maximum(n - 1, 1)
    wts = np.maximum(n - 1, 0)
    alpha_cell = np.maximum((wts * (var - mean)).sum(axis=0)
                            / np.maximum((wts * mean ** 2).sum(axis=0), 1e-300), 0)
    inv_n = float((1 / n).mean())
    alpha_bio = np.maximum(res.dispersion - alpha_cell * inv_n, 0)
    # Per-cell mean of the reference samples, on the normalised scale with the
    # size factors scaled to cells: a sample of m cells has mean m * q.
    sf_per_cell = res.size_factors / n[:, 0]
    sf_per_cell = sf_per_cell / np.exp(np.mean(np.log(sf_per_cell)))
    ref = pilot.s_cond == 0
    q = (S1[ref] / sf_per_cell[ref, None]).sum(axis=0) / n[ref, 0].sum()
    rng = np.random.default_rng([seed, 104729])
    pick = np.sort(rng.choice(K.shape[1], size=min(n_genes, K.shape[1]), replace=False))
    n_sig = int(np.nansum(res.padj < ALPHA)) if pilot.design != "single" else None
    return DEModel(
        pilot=pilot, genes=pilot.gene_names[pilot.tested][pick], q=q[pick],
        alpha_bio=alpha_bio[pick], alpha_cell=alpha_cell[pick], tau2=tau2[pick],
        alpha_pilot=res.dispersion[pick], n_tested=int(K.shape[1]), pilot_n_significant=n_sig,
        median_cells=float(np.median(pilot.n_cells)), trend=res.trend)


@dataclass
class DEGrid:
    log2fc: list[float]
    donors: list[int]
    cells: list[int]  # cells of the type per sample
    design: str
    de_share: float = DE_SHARE
    alpha: float = ALPHA
    reps: int = 10
    seed: int = 0


def de_auto_reps(grid: DEGrid, n_genes: int) -> int:
    """Studies per grid point so the grid takes about DE_TIME_BUDGET_S."""
    per = _DE_COST["paired" if grid.design == "paired" else "unpaired"] * n_genes
    cost = per * sum(2 * n for n in grid.donors) * len(grid.cells) * len(grid.log2fc)
    reps = int(DE_TIME_BUDGET_S / max(cost, 1e-9))
    return int(min(DE_MAX_REPS, max(DE_MIN_REPS, reps)))


@dataclass
class DEPoint:
    log2fc: float
    donors: int
    cells: int
    power: float
    se: float
    fdp: float | None  # false discoveries / discoveries, averaged over the studies
    tested_share: float  # DE genes passing the app's count filter in the simulated study
    power_by_expression: list[float] = field(default_factory=list)  # by tertile of q


def de_power_point(model: DEModel, grid: DEGrid, L: float, n: int, m: int) -> DEPoint:
    """Share of the changed genes the fast DESeq2-style test finds (BH < alpha,
    right sign) in ``grid.reps`` simulated studies of n donors per group with m
    cells of the type per sample."""
    from scview.core import pseudobulk

    paired = grid.design == "paired" and model.pilot.design == "paired"
    G = len(model.q)
    s_cond, s_donor, _ = _future_design("paired" if paired else "unpaired", n)
    n_s = len(s_cond)
    alpha = model.alpha_bio + model.alpha_cell / m
    tau = np.sqrt(model.tau2)
    rng = np.random.default_rng([grid.seed, n, m, int(round(L * 1000))])
    tert = np.quantile(model.q, [1 / 3, 2 / 3])
    tier = np.digitize(model.q, tert)
    found, fdps, tested_share = [], [], []
    by_tier = np.zeros(3)
    tier_n = np.zeros(3)
    n_de = max(1, int(round(grid.de_share * G)))
    for _ in range(grid.reps):
        de = rng.choice(G, size=n_de, replace=False)
        sign = rng.choice([-1.0, 1.0], size=n_de)
        lfc = np.zeros(G)
        lfc[de] = sign * L * math.log(2)
        n_units = n if paired else n_s
        u = rng.standard_normal((n_units, G)) * tau - tau ** 2 / 2
        u = u[s_donor] if paired else u
        mean = m * model.q[None, :] * np.exp(u + lfc[None, :] * s_cond[:, None])
        shape = 1 / np.maximum(alpha, MIN_DISP)
        lam = rng.gamma(shape[None, :], mean / shape[None, :])
        K = rng.poisson(lam)
        ok = pseudobulk._filter_genes(K, n)
        hit = np.zeros(G, dtype=bool)
        if len(ok) >= 10:
            try:
                res = nb_wald_test(K[:, ok], s_cond, s_donor, "paired" if paired else "unpaired")
                sig = (res.padj < grid.alpha) & np.isfinite(res.padj)
                right = np.sign(res.log2fc) == np.sign(lfc[ok])
                hit[ok[sig & right]] = True
                called = ok[sig]
                if called.size:
                    fdps.append(float(np.mean(lfc[called] == 0)))
            except PlannerError:
                pass
        is_de = np.zeros(G, dtype=bool)
        is_de[de] = True
        found.append(float(hit[de].mean()))
        tested_share.append(float(np.isin(de, ok).mean()))
        for t in range(3):
            sel = is_de & (tier == t)
            by_tier[t] += hit[sel].sum()
            tier_n[t] += sel.sum()
    power = float(np.mean(found))
    se = float(np.std(found, ddof=1) / math.sqrt(len(found))) if len(found) > 1 else 0.0
    return DEPoint(L, n, m, power, se, float(np.mean(fdps)) if fdps else None,
                   float(np.mean(tested_share)),
                   [float(by_tier[t] / tier_n[t]) if tier_n[t] else 0.0 for t in range(3)])


def de_points(grid: DEGrid) -> int:
    return len(grid.log2fc) * len(grid.donors) * len(grid.cells)


def iter_de_power(model: DEModel, grid: DEGrid,
                  stop: Callable[[], bool] | None = None) -> Iterator[DEPoint]:
    for L in grid.log2fc:
        for m in grid.cells:
            for n in grid.donors:
                if stop and stop():
                    return
                yield de_power_point(model, grid, L, n, m)


def de_donors_needed(points: Sequence[DEPoint], target: float = TARGET_POWER) -> float | None:
    """As ``donors_needed``, for DE points of one log2FC and cells value."""
    proxy = [PowerPoint("", 0, p.donors, p.cells, p.power, None) for p in points]
    return donors_needed(proxy, target)


# ---------------------------------------------------------------------------
# Runs: a planner request -> progress events -> one result payload
# ---------------------------------------------------------------------------


def _check_grid(values: Sequence, name: str, lo: float, hi: float, integer: bool) -> list:
    vals = sorted(dict.fromkeys(values))
    if not vals:
        raise PlannerError(f"Give at least one value for {name}.")
    if len(vals) > MAX_GRID_VALUES:
        raise PlannerError(f"At most {MAX_GRID_VALUES} values for {name}.")
    for v in vals:
        if not (lo <= v <= hi) or (integer and int(v) != v):
            raise PlannerError(f"{name} must be {'whole numbers' if integer else 'numbers'} "
                               f"from {lo:g} to {hi:g} (got {v}).")
    return [int(v) for v in vals] if integer else [float(v) for v in vals]


def _cells_summary(n: np.ndarray) -> dict:
    return {"median": float(np.median(n)), "min": int(n.min()), "max": int(n.max())} if len(n) \
        else {"median": 0.0, "min": 0, "max": 0}


def _fmt_share(v: float) -> str:
    return f"{v * 100:.1f}%" if v >= 0.001 else f"{v * 100:.2g}%"


def _pilot_line(pilot, unit: str) -> str:
    per = ", ".join(f"{v} {k}" for k, v in pilot.n_samples.items())
    how = {"paired": f"each donor in both conditions (paired by '{pilot.sample_key}')",
           "unpaired": f"donors of '{pilot.sample_key}' not shared between the conditions",
           "single": f"one condition, donors of '{pilot.sample_key}'"}.get(pilot.design, "")
    cells = _cells_summary(np.asarray(pilot.n_cells))
    line = (f"Pilot: {len(pilot.donors)} donors, {per} samples ({how}); {cells['median']:,.0f} "
            f"{unit} per sample (median; {cells['min']:,} to {cells['max']:,}).")
    if pilot.dropped_samples:
        line += f" Left out (under 10 cells): {', '.join(pilot.dropped_samples)}."
    return line


@dataclass
class CompositionRequestModel:
    """What a composition plan asks for (the API body, validated)."""

    group_key: str
    condition_key: str | None = None
    sample_key: str | None = None
    reference: str | None = None
    conditions: list[str] | None = None
    groups: list[str] | None = None
    fold_changes: list[float] = field(default_factory=lambda: list(DEFAULT_FOLD_CHANGES))
    donors: list[int] = field(default_factory=lambda: list(DEFAULT_DONORS))
    cells: list[int] = field(default_factory=lambda: list(DEFAULT_CELLS))
    alpha: float = ALPHA
    target_power: float = TARGET_POWER
    design: str | None = None  # None: the pilot's (paired only from a paired pilot)
    reps: int | None = None
    assumed_sd: float | None = None
    transform: str = "logit"
    robust: bool = True
    seed: int = 0


def prepare_composition(obs: pd.DataFrame, req: CompositionRequestModel
                        ) -> tuple[CompositionModel, CompositionGrid]:
    """The pilot, its fitted spreads and the grid (fast; errors are 400s)."""
    fcs = _check_grid(req.fold_changes, "fold changes", 0.05, 20, False)
    donors = _check_grid(req.donors, "donors per group", 2, MAX_DONORS, True)
    cells = _check_grid(req.cells, "cells per donor", 20, MAX_CELLS, True)
    if not (0 < req.alpha < 0.5) or not (0 < req.target_power < 1):
        raise PlannerError("alpha must be in (0, 0.5) and the target power in (0, 1).")
    pilot = composition_pilot(obs, req.group_key, req.condition_key, req.sample_key,
                              reference=req.reference, conditions=req.conditions)
    if req.assumed_sd is not None:
        model = assumed_composition(pilot, req.assumed_sd)
    else:
        model = fit_composition(pilot, transform=req.transform, seed=req.seed)
    design = req.design or ("paired" if model.can_pair else "unpaired")
    if design not in ("paired", "unpaired"):
        raise PlannerError("design must be 'paired' or 'unpaired'.")
    if design == "paired" and not model.can_pair:
        raise PlannerError(
            "A paired design needs a paired pilot (every donor in both conditions), so the "
            "donor part of the spread can be told from the rest; plan an unpaired study.")
    targets = pick_targets(model, req.groups)
    grid = CompositionGrid(targets=targets, fold_changes=fcs, donors=donors, cells=cells,
                           design=design, alpha=req.alpha, transform=req.transform,
                           robust=req.robust, seed=req.seed,
                           pessimistic=model.source == "pilot")
    n_points = composition_points(grid)
    grid.reps = int(req.reps) if req.reps else auto_reps(n_points, len(model.groups), req.robust)
    if not (10 <= grid.reps <= 2000):
        raise PlannerError("reps must be from 10 to 2000.")
    return model, grid


def _composition_assumptions(model: CompositionModel, grid: CompositionGrid,
                             points: list[PowerPoint]) -> tuple[list[str], list[str]]:
    pilot = model.pilot
    out, notes = [], []
    if model.source == "pilot":
        out.append(_pilot_line(pilot, "cells"))
        used = "within-donor" if grid.design == "paired" else "total (donor + within-donor)"
        lo, hi = spread_interval(model, grid.design, grid.transform)
        examples = "; ".join(
            f"{model.spreads[t].group} "
            f"{(model.spreads[t].sd_within if grid.design == 'paired' else model.spreads[t].sd_total):.2f}"
            f" (95% interval {lo[t]:.2f} to {hi[t]:.2f})" for t in grid.targets[:3])
        out.append(
            "Spread: each cell type's donor-to-donor SD of its log-odds share, from the pilot's "
            "residual variance in the app's own model (logit shares"
            + (", donor as a block" if pilot.design == "paired" else "")
            + f", {model.df} residual df) with the counting noise removed by simulating pilots "
            f"of the same design. The simulation uses the {used} spread: {examples}. With "
            f"{len(pilot.donors)} donors these intervals are wide; the dashed curves use their "
            "upper ends.")
        zero = [s.group for s in model.spreads if s.sd_within < 0.02]
        if pilot.design == "paired" and len(zero) >= max(2, len(model.spreads) // 3):
            notes.append(
                f"Within a donor, {len(zero)} of {len(model.spreads)} types vary no more than "
                "counting noise allows: the paired samples look like aliquots of one sample "
                "(as in a split-and-stimulate design). Paired samples taken separately (before "
                "and after a treatment) will vary more: read the unpaired design as the safer "
                "guide for such a study.")
    else:
        src = f" ({ASSUMED_SD_SOURCE})" if model.assumed_sd == ASSUMED_SD else ""
        out.append(
            f"Spread: ENTERED, not estimated from this dataset: a donor-to-donor SD of "
            f"{model.assumed_sd:g} on the log-odds share for every cell type{src}. Shares are "
            "the pooled shares of the reference cells.")
    design = ("paired: every donor sampled in both conditions" if grid.design == "paired"
              else "unpaired: different donors in each group")
    out.append(f"Design simulated: {design}; n donors per group, m cells per sample, every "
               "sample exactly m cells (counts binomial per cell type).")
    out.append("Effect: the cell type's mean share times the fold change in the tested group, "
               "the other types scaled down to make room (as in propeller's own simulations, "
               f"Phipson et al. 2022); a change that would take a type past {MAX_SHARE:.0%} of "
               "the cells is left out.")
    ses = [p.se for p in points if p.se is not None and not p.pessimistic]
    test = "moderated t (limma empirical Bayes" + (", robust" if grid.robust else "") + ")"
    out.append(
        f"Test: the app's composition test on every simulated study: {grid.transform}-"
        f"transformed shares, {test}, Benjamini-Hochberg across all {len(model.groups)} types. "
        f"Power = share of {grid.reps} simulated studies per point with adjusted p < "
        f"{grid.alpha:g} in the right direction (Monte Carlo SE up to "
        f"{(max(ses) if ses else 0) * 100:.0f} points).")
    out.append("The pilot's shares and spreads stand in for the new study's: another tissue, "
               "protocol or population can vary more, and more variability needs more donors.")
    return out, notes


def composition_payload(model: CompositionModel, grid: CompositionGrid,
                        points: list[PowerPoint], target_power: float, elapsed: float) -> dict:
    pilot = model.pilot
    lower, upper = (spread_interval(model, grid.design, grid.transform)
                    if model.source == "pilot" else (None, None))
    types = []
    for j, s in enumerate(model.spreads):
        used = s.sd_within if grid.design == "paired" else s.sd_total
        types.append({
            "group": s.group, "share": s.share, "share_tested": s.share_tested,
            "n_cells": s.n_cells, "sd_within": s.sd_within, "sd_donor": s.sd_donor,
            "sd_total": s.sd_total, "sd_used": used, "sd_low": s.sd_low, "sd_high": s.sd_high,
            "sd_low_used": float(lower[j]) if lower is not None else None,
            "sd_upper_used": float(upper[j]) if upper is not None else None,
            "targeted": j in grid.targets})
    needed = []
    for t in grid.targets:
        name = model.groups[t]
        for fc in grid.fold_changes:
            for m in grid.cells:
                pts = [p for p in points if p.group == name and p.fold_change == fc
                       and p.cells == m and not p.pessimistic]
                needed.append({"group": name, "fold_change": fc, "cells": m,
                               "pessimistic": False,
                               "feasible": any(p.power is not None for p in pts),
                               "donors_needed": donors_needed(pts, target_power)})
            pes = [p for p in points if p.group == name and p.fold_change == fc and p.pessimistic]
            if pes:
                needed.append({"group": name, "fold_change": fc, "cells": grid.middle_cells,
                               "pessimistic": True,
                               "feasible": any(p.power is not None for p in pes),
                               "donors_needed": donors_needed(pes, target_power)})
    assumptions, notes = _composition_assumptions(model, grid, points)
    return {
        "kind": "composition", "group_key": pilot.group_key,
        "condition_key": pilot.condition_key, "sample_key": pilot.sample_key,
        "conditions": pilot.conditions, "reference": pilot.conditions[0] if pilot.conditions else None,
        "source": model.source, "assumed_sd": model.assumed_sd,
        "assumed_sd_default": ASSUMED_SD, "assumed_sd_source": ASSUMED_SD_SOURCE,
        "pilot": {"design": pilot.design, "n_samples": pilot.n_samples,
                  "n_donors": len(pilot.donors), "df": model.df,
                  "cells_per_sample": _cells_summary(pilot.n_cells),
                  "dropped_samples": pilot.dropped_samples},
        "design": grid.design, "can_pair": model.can_pair, "alpha": grid.alpha,
        "target_power": target_power, "reps": grid.reps, "transform": grid.transform,
        "robust": grid.robust, "donors": grid.donors, "cells": grid.cells,
        "fold_changes": grid.fold_changes, "middle_cells": grid.middle_cells,
        "types": types, "points": [p.__dict__ for p in points], "needed": needed,
        "assumptions": assumptions, "notes": notes, "elapsed_s": round(elapsed, 2),
    }


def iter_composition_plan(model: CompositionModel, grid: CompositionGrid, target_power: float,
                          stop: Callable[[], bool] | None = None,
                          t0: float | None = None) -> Iterator[dict]:
    """Events for a prepared composition plan: ``{"start": ...}``, one
    ``{"progress": ...}`` per grid point, then ``{"result": payload}``."""
    t0 = time.perf_counter() if t0 is None else t0
    total = composition_points(grid)
    yield {"start": {"kind": "composition", "n_points": total, "reps": grid.reps,
                     "design": grid.design, "source": model.source,
                     "groups": [model.groups[t] for t in grid.targets]}}
    points = []
    for i, p in enumerate(iter_composition_power(model, grid, stop)):
        points.append(p)
        yield {"progress": {"done": i + 1, "total": total, "group": p.group,
                            "fold_change": p.fold_change, "donors": p.donors, "cells": p.cells,
                            "power": p.power, "pessimistic": p.pessimistic}}
    if stop and stop():
        return
    yield {"result": composition_payload(model, grid, points, target_power,
                                         time.perf_counter() - t0)}


@dataclass
class DERequestModel:
    group_key: str
    group: str
    condition_key: str | None = None
    sample_key: str | None = None
    reference: str | None = None
    conditions: list[str] | None = None
    log2fc: list[float] = field(default_factory=lambda: list(DE_DEFAULT_LOG2FC))
    donors: list[int] = field(default_factory=lambda: list(DEFAULT_DONORS))
    cells: list[int] = field(default_factory=lambda: list(DE_DEFAULT_CELLS))
    de_share: float = DE_SHARE
    alpha: float = ALPHA
    target_power: float = TARGET_POWER
    design: str | None = None
    reps: int | None = None
    n_genes: int = DE_GENES
    seed: int = 0


def prepare_de(adata, req: DERequestModel) -> tuple[DEModel, DEGrid]:
    """The pilot's per-sample sums, its per-gene fit and the grid (reads one
    cell type's counts; errors are 400s)."""
    lfcs = _check_grid(req.log2fc, "log2 fold changes", 0.1, 5, False)
    donors = _check_grid(req.donors, "donors per group", 2, MAX_DONORS, True)
    cells = _check_grid(req.cells, "cells per donor", 10, MAX_CELLS, True)
    if not (0 < req.de_share <= 0.5):
        raise PlannerError("The share of changed genes must be in (0, 0.5].")
    if not (0 < req.alpha < 0.5) or not (0 < req.target_power < 1):
        raise PlannerError("alpha must be in (0, 0.5) and the target power in (0, 1).")
    if not (200 <= req.n_genes <= 10_000):
        raise PlannerError("n_genes must be from 200 to 10,000.")
    pilot = de_pilot(adata, req.group_key, req.group, req.condition_key, req.sample_key,
                     reference=req.reference, conditions=req.conditions)
    model = fit_de(pilot, n_genes=req.n_genes, seed=req.seed)
    design = req.design or ("paired" if pilot.design == "paired" else "unpaired")
    if design not in ("paired", "unpaired"):
        raise PlannerError("design must be 'paired' or 'unpaired'.")
    if design == "paired" and pilot.design != "paired":
        raise PlannerError("A paired design needs a paired pilot (every donor in both "
                           "conditions); plan an unpaired study.")
    grid = DEGrid(log2fc=lfcs, donors=donors, cells=cells, design=design,
                  de_share=req.de_share, alpha=req.alpha, seed=req.seed)
    grid.reps = int(req.reps) if req.reps else de_auto_reps(grid, len(model.q))
    if not (2 <= grid.reps <= 500):
        raise PlannerError("reps must be from 2 to 500.")
    return model, grid


def _de_assumptions(model: DEModel, grid: DEGrid, points: list[DEPoint]) -> list[str]:
    from scview.core import pseudobulk

    pilot = model.pilot
    med = lambda v: float(np.median(v))  # noqa: E731
    out = [_pilot_line(pilot, f"{pilot.group} cells")]
    out.append(
        f"Genes: {model.n_tested:,} pass the app's pseudobulk filter on the pilot "
        f"({pseudobulk.MIN_GENE_COUNT}+ counts in as many samples as the smaller condition "
        f"has); {len(model.q):,} of them, drawn at random, are simulated.")
    within = " within donor" if pilot.design == "paired" else ""
    donor = (f"; a donor effect (median SD {math.sqrt(med(model.tau2)):.2f} on the log scale) "
             "comes on top in an unpaired study" if pilot.design == "paired" else "")
    out.append(
        f"Per gene: the mean count per cell and the pilot's pseudobulk dispersion{within} "
        f"(DESeq2's estimate, median {med(model.alpha_pilot):.3f}). The part from the cells' "
        f"own spread (per-cell overdispersion, median {med(model.alpha_cell):.2f}, divided by "
        f"the cells per sample) shrinks with more cells; the donors' part (median "
        f"{med(model.alpha_bio):.3f}) does not{donor}.")
    design = ("paired: every donor sampled in both conditions" if grid.design == "paired"
              else "unpaired: different donors in each group")
    out.append(f"Design simulated: {design}; m cells of {pilot.group} per sample.")
    out.append(
        f"Effect: {grid.de_share:.0%} of the simulated genes change by the log2 fold change "
        "(half up, half down at random), the rest not. Power = share of the changed genes "
        f"found at FDR < {grid.alpha:g} with the right sign, mean of {grid.reps} simulated "
        "studies per point. A gene below the app's count filter in a study counts as not "
        "found: that is how too few cells per donor lose genes.")
    fdps = [p.fdp for p in points if p.fdp is not None]
    sig = (f"; on the pilot itself it calls {model.pilot_n_significant:,} genes at FDR < 0.05"
           if model.pilot_n_significant is not None else "")
    out.append(
        "Test: DESeq2's negative binomial Wald test as the app's pseudobulk DE runs it "
        "(pydeseq2), reimplemented vectorised so hundreds of studies run in seconds; Cook's "
        f"refits and independent filtering are left out{sig}. Across the simulated studies "
        f"false discoveries were {np.mean(fdps) * 100:.0f}% of the calls (target "
        f"{grid.alpha:.0%})." if fdps else "Test: DESeq2's negative binomial Wald test, "
        "reimplemented vectorised (no calls in the simulated studies).")
    out.append(f"The dispersions are estimates from {len(pilot.donors)} donors: another tissue, "
               "protocol or population can vary more, and more variability needs more donors.")
    return out


def de_payload(model: DEModel, grid: DEGrid, points: list[DEPoint], target_power: float,
               elapsed: float) -> dict:
    pilot = model.pilot
    needed = []
    for L in grid.log2fc:
        for m in grid.cells:
            pts = [p for p in points if p.log2fc == L and p.cells == m]
            needed.append({"log2fc": L, "cells": m,
                           "donors_needed": de_donors_needed(pts, target_power)})
    tert = np.quantile(model.q, [1 / 3, 2 / 3])
    return {
        "kind": "de", "group_key": pilot.group_key, "group": pilot.group,
        "condition_key": pilot.condition_key, "sample_key": pilot.sample_key,
        "conditions": pilot.conditions, "reference": pilot.conditions[0] if pilot.conditions else None,
        "pilot": {"design": pilot.design, "n_samples": pilot.n_samples,
                  "n_donors": len(pilot.donors), "cells_per_sample": _cells_summary(pilot.n_cells),
                  "n_tested": model.n_tested, "n_simulated": int(len(model.q)),
                  "counts_source": pilot.counts_source,
                  "n_significant": model.pilot_n_significant,
                  "dropped_samples": pilot.dropped_samples},
        "dispersion": {"pilot_median": float(np.median(model.alpha_pilot)),
                       "bio_median": float(np.median(model.alpha_bio)),
                       "cell_median": float(np.median(model.alpha_cell)),
                       "donor_sd_median": float(math.sqrt(np.median(model.tau2)))},
        "expression_tertiles": [float(t) for t in tert],
        "design": grid.design, "can_pair": pilot.design == "paired",
        "log2fc": grid.log2fc, "donors": grid.donors, "cells": grid.cells,
        "de_share": grid.de_share, "alpha": grid.alpha, "target_power": target_power,
        "reps": grid.reps, "points": [p.__dict__ for p in points], "needed": needed,
        "assumptions": _de_assumptions(model, grid, points), "notes": [],
        "elapsed_s": round(elapsed, 2),
    }


def iter_de_plan(model: DEModel, grid: DEGrid, target_power: float,
                 stop: Callable[[], bool] | None = None,
                 t0: float | None = None) -> Iterator[dict]:
    """Events for a prepared DE plan, as ``iter_composition_plan``."""
    t0 = time.perf_counter() if t0 is None else t0
    total = de_points(grid)
    yield {"start": {"kind": "de", "n_points": total, "reps": grid.reps, "design": grid.design,
                     "group": model.pilot.group, "n_genes": int(len(model.q))}}
    points = []
    for i, p in enumerate(iter_de_power(model, grid, stop)):
        points.append(p)
        yield {"progress": {"done": i + 1, "total": total, "log2fc": p.log2fc,
                            "donors": p.donors, "cells": p.cells, "power": p.power}}
    if stop and stop():
        return
    yield {"result": de_payload(model, grid, points, target_power, time.perf_counter() - t0)}


def check_pilot(adata, group_key: str, condition_key: str | None, sample_key: str | None,
                reference: str | None = None, conditions: Sequence[str] | None = None) -> dict:
    """What a plan would use, without simulating (obs only): the pilot's design
    and samples, the types' shares, and whether it has replicates at all."""
    pilot = composition_pilot(adata.obs, group_key, condition_key, sample_key,
                              reference=reference, conditions=conditions)
    counts = pilot.counts[pilot.s_cond == 0].sum(axis=0)
    share = counts / max(counts.sum(), 1)
    model_stub = CompositionModel(pilot=pilot, groups=pilot.groups, share=np.maximum(share, MIN_SHARE),
                                  sd_within=np.zeros(len(share)), sd_donor=np.zeros(len(share)),
                                  spreads=[], source="stub", df=None)
    return {
        "group_key": group_key, "condition_key": condition_key, "sample_key": sample_key,
        "conditions": pilot.conditions, "design": pilot.design, "replicated": pilot.replicated,
        "n_samples": pilot.n_samples, "n_donors": len(pilot.donors),
        "cells_per_sample": _cells_summary(pilot.n_cells),
        "groups": [{"group": g, "share": float(share[j]), "n_cells": int(pilot.counts[:, j].sum())}
                   for j, g in enumerate(pilot.groups)],
        "suggested": [pilot.groups[t] for t in pick_targets(model_stub)],
        "message": None if pilot.replicated else no_replicates_message(pilot),
        "assumed_sd_default": ASSUMED_SD, "assumed_sd_source": ASSUMED_SD_SOURCE,
        "dropped_samples": pilot.dropped_samples,
    }


# ---------------------------------------------------------------------------
# Co-pilot grounding: the last plan of each kind on each loaded dataset
# ---------------------------------------------------------------------------

TAG = "result:planner"
_last: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_last_lock = threading.Lock()


def remember(adaptor, payload: dict) -> None:
    """Keep the last payload of its kind for the co-pilot, as long as the loaded
    dataset is kept."""
    try:
        with _last_lock:
            _last.setdefault(adaptor, {})[payload["kind"]] = payload
    except TypeError:  # not weak-referenceable (a test double)
        pass


def last_plans(adaptor) -> dict:
    try:
        with _last_lock:
            return dict(_last.get(adaptor, {}))
    except TypeError:
        return {}


def _n_text(v: float | None, top: int) -> str:
    return f"{v:.0f}" if v is not None else f">{top}"


def grounding_lines(adaptor) -> tuple[list[str], str | None]:
    """For the co-pilot: the last composition and DE plans, donors per group
    needed at the target power for each effect and cells value."""
    plans = last_plans(adaptor)
    obs_cols = set(getattr(getattr(getattr(adaptor, "adata", None), "obs", None), "columns", []))
    plans = {k: p for k, p in plans.items() if p.get("group_key") in obs_cols}
    if not plans:
        return [], None
    lines = ["\n## Study planner (last run; studies simulated from this pilot's donors) "
             f"[{TAG}]"]
    details = []
    c = plans.get("composition")
    if c:
        src = ("spreads estimated from the pilot" if c["source"] == "pilot"
               else f"ENTERED spread {c['assumed_sd']:g}, not estimated")
        lines.append(
            f"- Composition ({c['design']} design, {src}; power {c['target_power']:.0%}, "
            f"adjusted p < {c['alpha']:g}, the app's composition test): donors per group needed "
            f"at {' / '.join(f'{m:,}' for m in c['cells'])} cells per donor:")
        top = max(c["donors"])
        for t in (t for t in c["types"] if t["targeted"]):
            parts = []
            for fc in c["fold_changes"]:
                rows = [r for r in c["needed"] if r["group"] == t["group"]
                        and r["fold_change"] == fc and not r["pessimistic"]]
                if not any(r["feasible"] for r in rows):
                    parts.append(f"x{fc:g} not possible")
                    continue
                parts.append(f"x{fc:g} " + "/".join(_n_text(r["donors_needed"], top) for r in rows))
            lines.append(f"  - {t['group']} (share {_fmt_share(t['share'])}, spread SD "
                         f"{t['sd_used']:.2f}): " + "; ".join(parts))
        details.append(f"composition by '{c['group_key']}' ({c['design']})")
    d = plans.get("de")
    if d:
        top = max(d["donors"])
        parts = []
        for L in d["log2fc"]:
            rows = [r for r in d["needed"] if r["log2fc"] == L]
            parts.append(f"log2FC {L:g}: " + "/".join(_n_text(r["donors_needed"], top) for r in rows))
        lines.append(
            f"- Pseudobulk DE in {d['group']} ({d['design']} design, {d['de_share']:.0%} of genes "
            f"changed, power = share found at FDR < {d['alpha']:g}, target {d['target_power']:.0%}): "
            f"donors per group needed at {' / '.join(f'{m:,}' for m in d['cells'])} {d['group']} "
            "cells per donor: " + "; ".join(parts))
        details.append(f"DE in {d['group']} ({d['design']})")
    lines.append("- These are simulations from the pilot's own variability, itself an estimate "
                 "from few donors: give them as estimates, never as guarantees.")
    return lines, "Study planner: " + "; ".join(details)
