"""Batch x condition confounding: when a technical grouping and a biological one
coincide, their effects can't be told apart.

Two shapes matter (Hicks et al. 2018; Luecken & Theis 2019):

- **confounded**: each batch holds one condition and each condition one batch
  (1:1). A difference between the conditions is equally a difference between
  the batches, so DE between them measures both, and batch correction on that
  column removes the condition effect along with the batch effect.
- **nested**: each batch holds one condition, but a condition spans several
  batches. Batches can then serve as replicates (pseudobulk), but correcting
  for the batch still removes the condition difference from the embedding.

Used by DE between groups (api/v1/de.py), the joint-embedding plan
(core/integration.py) and the QC report's study design (core/qc_report.py).

A third shape is checked for cell-level DE and the composition test
(``one_sided_levels``): a technology or batch value present on one side only, holding
a fair share of that side's cells (Wilson 2022: the one 3' library is 37% of the
control nuclei, every DKD library 5'; half the cell-level DKD genes are also 5'-vs-3'
genes). Such a column is crossed with the condition, so ``check`` lets it pass.
Donor columns are not batches here: donors nested in conditions are the normal
design, handled by the replicate guard and pseudobulk.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

CONDITION_RE = re.compile(
    r"condition|stim|treat|disease|status|genotype|timepoint|time_point|infect|perturb|group",
    re.I,
)
BATCH_RE = re.compile(r"batch|library|lane|plate|chip|^dataset$|^run$|seq_?run", re.I)
# Technology columns (CELLxGENE ``assay``, a chemistry or kit): few values, and a value
# on one side only makes a cell-level comparison partly a technology comparison.
TECH_RE = re.compile(r"assay|chemistry|technolog|platform|protocol|^kit|_kit|suspension_type|"
                     r"seq_?tech|library_prep", re.I)

# A batch counts as holding a condition only with this many of its cells (and
# this share of them): a few stray or mislabelled cells must not hide a design
# that is confounded for every practical purpose.
MIN_CELLS = 10
MIN_SHARE = 0.02


@dataclass
class Confounding:
    batch: str
    condition: str
    kind: str  # "confounded" | "nested"
    n_batches: int
    n_conditions: int
    role: str = "comparison"  # "comparison" (DE between groups) | "correction" (a batch key)

    @property
    def message(self) -> str:
        if self.batch == self.condition and self.role == "correction":
            return (f"Batch correction used '{self.batch}', which looks like a condition: it "
                    f"removes the '{self.batch}' difference from the embedding and clusters "
                    "(expression values and DE are unaffected).")
        if self.batch == self.condition:
            return (f"The two sides are different '{self.batch}' groups, i.e. different batches, "
                    "so their differences mix technical and biological effects.")
        if self.kind == "confounded":
            return (f"Each '{self.batch}' group holds one '{self.condition}' group and vice versa, "
                    f"so '{self.condition}' differences can't be told apart from '{self.batch}' "
                    "(technical) differences.")
        return (f"Every '{self.batch}' group holds a single '{self.condition}' group, so "
                f"correcting for '{self.batch}' can also remove the '{self.condition}' difference.")

    def to_dict(self) -> dict:
        return {"batch": self.batch, "condition": self.condition, "kind": self.kind,
                "n_batches": self.n_batches, "n_conditions": self.n_conditions,
                "role": self.role, "message": self.message}


def check(obs: pd.DataFrame, batch: str, condition: str,
          mask: np.ndarray | None = None) -> Confounding | None:
    """How ``batch`` and ``condition`` relate over the cells in ``mask``:
    confounded, nested, or None (crossed, so the two can be separated)."""
    if batch not in obs.columns or condition not in obs.columns:
        return None
    sub = obs if mask is None else obs.loc[np.asarray(mask, dtype=bool)]
    b = sub[batch].astype(str).where(sub[batch].notna())
    c = sub[condition].astype(str).where(sub[condition].notna())
    keep = b.notna() & c.notna()
    if keep.sum() == 0:
        return None
    table = pd.crosstab(b[keep], c[keep])
    if table.shape[1] < 2:
        return None  # one condition: nothing to confound
    sizes = table.sum(axis=1)
    present = (table >= MIN_CELLS) & (table.div(sizes, axis=0) >= MIN_SHARE)
    present = present[present.any(axis=1)]  # batches too small to hold anything are ignored
    per_batch = present.sum(axis=1)
    if present.shape[0] < 1 or (per_batch != 1).any():
        return None  # some batch holds several conditions: crossed
    per_condition = present.sum(axis=0)
    per_condition = per_condition[per_condition > 0]
    kind = "confounded" if (per_condition == 1).all() else "nested"
    return Confounding(batch=batch, condition=condition, kind=kind,
                       n_batches=int(present.shape[0]), n_conditions=int(per_condition.size))


def batch_columns(obs: pd.DataFrame, extra: Sequence[str] = ()) -> list[str]:
    """Batch-like obs columns (by name) plus ``extra`` (e.g. the pipeline's batch
    key), each with 2-50 values."""
    out: list[str] = []
    for col in [*extra, *(str(c) for c in obs.columns if BATCH_RE.search(str(c)))]:
        if col in obs.columns and col not in out and 2 <= obs[col].nunique(dropna=True) <= 50:
            out.append(col)
    return out


def condition_columns(obs: pd.DataFrame) -> list[str]:
    """Condition-like obs columns (by name) with 2-8 values."""
    return [str(c) for c in obs.columns
            if CONDITION_RE.search(str(c)) and 2 <= obs[c].nunique(dropna=True) <= 8]


def recorded_batch_key(adata) -> str | None:
    """The batch key of the dataset's latest batch-correction run, if any."""
    from scview.core import provenance

    for entry in reversed(provenance.read_provenance(adata).get("history", [])):
        if entry.get("step") == "batch_correction":
            key = (entry.get("params") or {}).get("batch_key")
            return str(key) if key else None
    return None


def for_comparison(obs: pd.DataFrame, split_by: str, mask: np.ndarray,
                   batch_extra: Sequence[str] = ()) -> Confounding | None:
    """For DE between two groups of ``split_by`` (cells in ``mask``): a batch
    whose groups line up with the two sides, so the comparison is also between
    batches. ``split_by`` itself counts when it is batch-like."""
    for batch in batch_columns(obs, batch_extra):
        found = check(obs, batch, split_by, mask)
        if found is not None and found.kind == "confounded":
            return found
    return None


def in_design(adata) -> list[Confounding]:
    """Every batch/condition pair of the dataset that is confounded or nested,
    the batch-correction key first."""
    obs = adata.obs
    key = recorded_batch_key(adata)
    found: list[Confounding] = []
    for batch in batch_columns(obs, [key] if key else []):
        for condition in condition_columns(obs):
            if condition == batch and batch != key:
                continue  # a condition column that merely looks batch-like by name
            hit = check(obs, batch, condition)
            if hit is not None:
                hit.role = "correction" if batch == key else "comparison"
                found.append(hit)
    return found


# A value on one side only is worth a warning when it holds at least this share of that
# side's cells; a column with more values than this is per sample, not a technology.
ONE_SIDED_SHARE = 0.10
ONE_SIDED_MAX_LEVELS = 8
MAX_SAMPLES_NAMED = 5


def one_sided_levels(obs: pd.DataFrame, sides: dict[str, np.ndarray], *, skip: Sequence[str] = (),
                     sample_key: str | None = None, extra: Sequence[str] = (),
                     what: str = "de") -> list[dict]:
    """Technology or batch values found on one side only (``sides``: label -> cell mask),
    each holding at least ONE_SIDED_SHARE of that side's cells: ``{"column", "value",
    "side", "share", "n_cells", "samples", "n_samples", "message"}``, largest share first.
    ``samples`` names the ``sample_key`` values behind it (one sample: the effect rests on
    one library). ``what`` ("de" or "composition") words the consequence."""
    labels = list(sides)
    if len(labels) < 2:
        return []
    side_of = np.full(len(obs), None, dtype=object)
    for k in labels:
        m = np.asarray(sides[k], bool) & pd.isna(side_of)
        side_of[m] = k
    keep = pd.notna(side_of)
    if not keep.any():
        return []
    sub = obs.loc[keep]
    side = pd.Series(side_of[keep], index=sub.index)
    skipped = set(skip) | ({sample_key} if sample_key else set())
    named = [str(c) for c in obs.columns if TECH_RE.search(str(c)) or BATCH_RE.search(str(c))]
    out: list[dict] = []
    for col in dict.fromkeys([*extra, *named]):
        if col not in obs.columns or col in skipped:
            continue
        v = sub[col].astype(str).where(sub[col].notna())
        ok = v.notna()
        if not 2 <= v[ok].nunique() <= ONE_SIDED_MAX_LEVELS:
            continue
        counts = pd.crosstab(v[ok], side[ok]).reindex(columns=labels, fill_value=0)
        sizes = counts.sum(axis=1)
        present = (counts >= MIN_CELLS) & (counts.div(sizes, axis=0) >= MIN_SHARE)
        if (present.sum(axis=0) == 0).any():
            continue  # a side with no value present: too few cells to say anything
        n_sides = present.sum(axis=1)
        if (n_sides[n_sides > 0] == 1).all() and not (
                TECH_RE.search(col) and (present.sum(axis=0) == 1).all()):
            continue  # every value on one side: samples nested in the sides (the usual design)
        side_sizes = counts.sum(axis=0)
        for level in counts.index[n_sides == 1]:
            k = present.columns[present.loc[level].to_numpy()][0]
            share = float(counts.loc[level, k]) / max(float(side_sizes[k]), 1.0)
            if share < ONE_SIDED_SHARE:
                continue
            samples: list[str] = []
            if sample_key and sample_key in obs.columns:
                cells = ((v == level) & (side == k)).to_numpy()
                samples = sorted(set(sub.loc[cells, sample_key].dropna().astype(str)))
            out.append({"column": col, "value": str(level), "side": str(k), "share": round(share, 4),
                        "n_cells": int(counts.loc[level, k]), "samples": samples[:MAX_SAMPLES_NAMED],
                        "n_samples": len(samples),
                        "message": _one_sided_message(col, str(level), str(k), share, samples,
                                                      sample_key, what)})
    out.sort(key=lambda d: -d["share"])
    return out


def _one_sided_message(col: str, level: str, side: str, share: float, samples: list[str],
                       sample_key: str | None, what: str) -> str:
    if len(samples) == 1:
        src, drop = f", all from one '{sample_key}' value: {samples[0]}", f"leave {samples[0]} out"
    elif samples:
        src, drop = f", from {len(samples)} '{sample_key}' values", "leave those samples out"
    else:
        src, drop = "", "leave those cells out"
    head = f"'{col}' value '{level}' is only in {side} ({share:.0%} of its cells{src})"
    if what == "composition":
        return (f"{head}: technologies and batches recover cell types differently, so a difference "
                f"in composition is partly a '{col}' difference. Add '{col}' as a covariate, or {drop}.")
    return (f"{head}: a cell-level test between the sides is partly a '{level}' vs other '{col}' "
            f"test. Compare donors (pseudobulk) with '{col}' as a covariate, or {drop}.")


def one_level_samples(values: Sequence, sample_names: Sequence[str]) -> list[tuple[str, str]]:
    """(value, sample) for each value of a per-sample covariate held by one sample only:
    as a covariate it fits that sample exactly, so the condition is estimated without it."""
    s = pd.Series([str(v) for v in values])
    counts = s.value_counts()
    return [(str(v), str(sample_names[int(np.flatnonzero(s.to_numpy() == v)[0])]))
            for v in counts.index[counts == 1]]
