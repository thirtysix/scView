"""Finding a dataset's donor / sample column.

Used by the DE replicate guard and pseudobulk (api/v1/de.py) and the QC
report's study design (core/qc_report.py), so both name the same column.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

# Most specific first, as in the frontend's lib/de.ts: a donor column beats a
# per-sample one, which often just encodes the condition (Kang's orig_ident is
# CTRL/STIM while Donor holds the 8 people).
DONOR_PATTERNS = [
    re.compile(p, re.I)
    for p in (
        r"donor", r"patient", r"individual", r"^ind$", r"subject",
        r"participant", r"replicate", r"sample", r"orig[._]ident",
    )
]


def donor_candidates(obs, exclude: Iterable[str] = ()) -> list[str]:
    """Donor-like obs columns with more than one value, best-named first."""
    skip = set(exclude)
    out: list[str] = []
    for pattern in DONOR_PATTERNS:
        for col in obs.columns:
            name = str(col)
            if name in skip or name in out or not pattern.search(name):
                continue
            if obs[col].nunique(dropna=True) > 1:
                out.append(name)
    return out


# Columns that name a person (not a sample, replicate or library): the level samples nest in.
PERSON_PATTERNS = DONOR_PATTERNS[:6]
FEW_DONORS = 3  # fewer on a side: the between-donor variance rests on one or two people
MAX_NAMED = 8  # donors named in a warning


def _differs_within(obs, frame: pd.DataFrame, cand: str, skip: set[str]) -> str | None:
    """A column (2-8 values) that is one value per sample but differs between one
    person's samples: what the person's samples are (a depot, a time point)."""
    rows = frame.index
    many = frame.drop_duplicates("s").groupby("d", observed=True)["s"].transform("size") > 1
    if not many.any():
        return None
    for col in obs.columns:
        name = str(col)
        if name in skip or name == cand:
            continue
        v = obs.loc[rows, col]
        if not 2 <= v.nunique(dropna=True) <= 8 or v.isna().any():
            continue
        f = pd.DataFrame({"s": frame["s"].to_numpy(), "d": frame["d"].to_numpy(),
                          "v": v.astype(str).to_numpy()})
        if (f.groupby("s", observed=True)["v"].nunique() > 1).any():
            continue  # varies within a sample: a per-cell column
        if (f.drop_duplicates("s").groupby("d", observed=True)["v"].nunique() > 1).any():
            return name
    return None


def repeated_donors(obs, sample_key: str, sides: Sequence, donor_key: str | None = None,
                    exclude: Iterable[str] = (), covariates: Iterable[str] = ()) -> dict | None:
    """When the replicates (``sample_key``, one per value per side) nest in a person
    column with more than one replicate for some person among the cells of ``sides``
    (boolean masks, one per side): ``{"donor_key", "n_samples", "n_donors", "donors":
    {donor: n replicates}, "within", "blocked", "message"}``. ``donor_key`` None tries the
    person-named columns; None when there is none, or every person gives one replicate.

    The advice fits the design: ``within`` is set when some person has samples on both
    sides (the compared factor varies inside a person, as a depot does inside a mouse), so
    the person is blocked on, not pooled; ``blocked`` (``message`` None) when the person
    column is already in ``covariates``, which is that blocking."""
    covs = set(covariates)
    if donor_key is not None:
        candidates = [donor_key] if donor_key != sample_key and donor_key in obs.columns else []
    else:
        skip = set(exclude) | {sample_key}
        candidates = [c for c in donor_candidates(obs, skip)
                      if any(p.search(c) for p in PERSON_PATTERNS)]
        candidates.sort(key=lambda c: c not in covs)  # a blocked person column first
    if not candidates:
        return None
    in_any = np.zeros(len(obs), bool)
    side_of = np.full(len(obs), -1)
    for i, mask in enumerate(sides):
        m = np.asarray(mask, bool)
        side_of[m & ~in_any] = i
        in_any |= m
    sample = obs[sample_key].astype(str).where(obs[sample_key].notna())
    for cand in candidates:
        person = obs[cand].astype(str).where(obs[cand].notna())
        frame = pd.DataFrame({"s": sample.to_numpy(), "d": person.to_numpy(), "side": side_of},
                             index=obs.index)
        frame = frame[in_any].dropna()
        if frame.empty or (frame.groupby("s", observed=True)["d"].nunique() > 1).any():
            continue  # not nested: a sample with cells of several people (pooled) or no person
        # Distinct sample values per person: a donor used as its own replicate column on
        # both sides (paired) is one value, and pseudobulk already blocks on it.
        reps = frame.drop_duplicates("s")
        per = reps.groupby("d", observed=True).size()
        many = per[per > 1].sort_values(ascending=False)
        if many.empty:
            continue  # one sample per person here (a copy of the sample column, or truly one each)
        donors = {str(k): int(v) for k, v in many.items()}
        shown = ", ".join(f"{k}: {v}" for k, v in list(donors.items())[:MAX_NAMED])
        more = f" and {len(donors) - MAX_NAMED} more" if len(donors) > MAX_NAMED else ""
        both = reps.drop_duplicates(["d", "side"]).groupby("d", observed=True).size()
        n_within = int((both > 1).sum())
        out = {"donor_key": cand, "n_samples": int(len(reps)), "n_donors": int(per.size),
               "donors": donors, "within": n_within > 0, "blocked": cand in covs}
        if cand in covs:
            out["message"] = None
            return out
        head = (f"Samples are not independent: {len(donors)} of {per.size} '{cand}' values give "
                f"more than one '{sample_key}' value here ({shown}{more}). Each sample is counted "
                "as an independent replicate, so p values are optimistic")
        if n_within:
            out["message"] = (
                f"{head}. The compared factor varies within '{cand}' ({n_within} of them have "
                f"samples on both sides), so block on it: add '{cand}' as a covariate (each "
                f"one's samples are then compared within it), or use '{cand}' as the replicate "
                "column (its cells are summed per side and paired).")
            return out
        other = _differs_within(obs, frame, cand, set(exclude) | {sample_key} | covs)
        if other:
            out["differs_by"] = other
            out["message"] = (
                f"{head}. A '{cand}' value's samples differ in '{other}': use '{cand}' as the "
                f"replicate column to count each one once (that pools its '{other}' levels), "
                f"or test within one '{other}' value.")
        else:
            out["message"] = (f"{head}; use '{cand}' as the replicate column to count each "
                              f"'{cand}' value once.")
        return out
    return None


def on_both_sides(obs, key: str, mask_a, mask_b, min_cells: int) -> list[str]:
    """The ``key`` values with at least ``min_cells`` cells on each side."""
    col = obs[key].astype(str).where(obs[key].notna())
    a = col[np.asarray(mask_a, bool)].value_counts()
    b = col[np.asarray(mask_b, bool)].value_counts()
    return sorted(set(a[a >= min_cells].index) & set(b[b >= min_cells].index))


def partly_paired(shared: Sequence[str], key: str, design: str) -> str:
    """The warning for donors on both sides of an unpaired design (not every donor is)."""
    shown = ", ".join(list(shared)[:MAX_NAMED]) + (" ..." if len(shared) > MAX_NAMED else "")
    return (f"{len(shared)} '{key}' value{'s are' if len(shared) != 1 else ' is'} on both sides "
            f"({shown}) but not every one, so the design cannot block on them ({design}): each "
            "one's two samples count as independent and p values are optimistic. Compare only the "
            "donors on both sides (paired), or keep one sample per donor.")


def few_donors(counts: dict[str, int], unit: str) -> str | None:
    """A warning when a side has fewer than FEW_DONORS (``counts``: side -> people)."""
    few = {k: n for k, n in counts.items() if n < FEW_DONORS}
    if not few:
        return None
    where = ", ".join(f"{k}: {n}" for k, n in few.items())
    return (f"Fewer than {FEW_DONORS} {unit} on a side ({where}): the variance between them rests "
            "on one or two people, so one unusual person can make or hide a difference.")


def pick_donor(obs, candidates: Sequence[str], groups: Sequence) -> str | None:
    """The first candidate with at least 2 donors in every group (boolean masks
    over the cells), else the best-named one. A sample column that only repeats
    the condition has one value per group, so it loses to a real donor column."""
    for name in candidates:
        if all(obs[name][mask].nunique() >= 2 for mask in groups):
            return name
    return candidates[0] if candidates else None
