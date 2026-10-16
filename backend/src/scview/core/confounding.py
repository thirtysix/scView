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
