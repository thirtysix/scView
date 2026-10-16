"""Finding a dataset's donor / sample column.

Used by the DE replicate guard and pseudobulk (api/v1/de.py) and the QC
report's study design (core/qc_report.py), so both name the same column.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

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


def pick_donor(obs, candidates: Sequence[str], groups: Sequence) -> str | None:
    """The first candidate with at least 2 donors in every group (boolean masks
    over the cells), else the best-named one. A sample column that only repeats
    the condition has one value per group, so it loses to a real donor column."""
    for name in candidates:
        if all(obs[name][mask].nunique() >= 2 for mask in groups):
            return name
    return candidates[0] if candidates else None
