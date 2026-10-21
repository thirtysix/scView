"""Composition of the two sides of a DE between groups: each side's share of
every cluster or state among the compared cells.

When condition A has 42% monocytes and condition B 18%, a DE of A vs B over all
cells finds monocyte genes "up in A" whether or not any cell changed. That is
the composition caveat (Crowell et al. 2020 muscat; Squair et al. 2021): the
test then partly measures who the cells are, not what they do. A pseudobulk
test does not protect against it either: each donor's sum carries the same mix.

Shares are column-normalised within each side (a group's cells over that
side's cells), so unequal side sizes do not show as a shift. A group is
flagged when the two shares differ by more than ``WARN_POINTS`` percentage
points, it holds at least ``MIN_CELLS`` cells on one side, and the difference
is unlikely to be chance (Fisher's exact test of the group's cells on each side,
p < ``MAX_P``): with 7 cells on one side, one cell is 14 points.

The composition column is the active clustering unless one is asked for. A
comparison already restricted to one group of that column is checked by the
next label column that splits those cells (a ``<column>_sub`` subclustering, a
QC state, another clustering); without one, it is skipped with a note. When
the compared column's groups ARE the composition groups (each group lies on
one side: two clusters compared by the clustering), composition is the
comparison itself, which a note says instead of warning.

Used by ``POST /de/compare`` (attached as ``composition``) and ``GET
/de/composition`` (the same check before a DE is run). Obs only, no matrix.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

from scview.core.anndata_adaptor import CLUSTERING_CANDIDATES

# A group whose two shares differ by more than this many percentage points
# makes the DE partly a composition comparison.
WARN_POINTS = 10.0
# ... and only with this many cells on at least one side: a handful of cells
# swing a share by 10 points in a small comparison.
MIN_CELLS = 10
# ... and only when Fisher's exact test (group vs other cells, side A vs B) gives
# p below this: large comparisons pass easily, tiny sides do not.
MAX_P = 0.05
# Groups returned, largest difference first (the check itself uses every group).
MAX_GROUPS = 50
# A flagged group this one-sided (A's share over the sum of both shares, or B's)
# may be a state the compared difference itself creates, not a population.
ONE_SIDED = 0.8
# Every group at least this one-sided: the sides are different groups of the
# composition column, so composition is the comparison (two clusters).
NESTED = 0.95

_STATE_RE = re.compile(r"state", re.I)
_CLUSTERING_RE = re.compile(r"^scview_(leiden|louvain)_|_sub$|^(leiden|louvain)", re.I)
# A finer column must split the compared cells into at least two groups this big.
_MIN_SPLIT = MIN_CELLS


def default_column(adata) -> str | None:
    """An author cell-type column when there is one (as the per-cell-type QC
    picks it), else the active clustering (``AnnDataAdaptor.active_clustering_column``).
    A clustering can split one cell type by the condition itself (Kang: CD14
    monocytes in two Leiden clusters, one per stimulation), which reads as a
    composition shift that is really the effect being tested."""
    from scview.core.sample_pseudobulk import default_celltype_column

    obs = adata.obs
    typed = default_celltype_column(obs)
    if typed is not None:
        return typed
    active = adata.uns.get("scview_active_clustering")
    if isinstance(active, str) and active in obs.columns:
        return active
    return next((c for c in CLUSTERING_CANDIDATES if c in obs.columns), None)


def _is_label(series: pd.Series) -> bool:
    return (isinstance(series.dtype, pd.CategoricalDtype) or series.dtype == object
            or pd.api.types.is_string_dtype(series.dtype) or pd.api.types.is_integer_dtype(series.dtype))


def _splits(series: pd.Series, mask: np.ndarray) -> bool:
    """At least two groups of ``_MIN_SPLIT`` cells among the ``mask`` cells."""
    counts = series[mask].value_counts(dropna=True)
    return int((counts >= _MIN_SPLIT).sum()) >= 2


def finer_column(obs: pd.DataFrame, column: str, mask: np.ndarray,
                 exclude: set[str]) -> str | None:
    """A label column that splits the ``mask`` cells, all of one ``column``
    group: its subclustering first, then a QC state, then another clustering."""
    sub = column if column.endswith("_sub") else f"{column}_sub"
    names = [str(c) for c in obs.columns]
    ordered = [sub] if sub != column else []
    ordered += [c for c in names if _STATE_RE.search(c)]
    ordered += [c for c in CLUSTERING_CANDIDATES if c in names]
    ordered += [c for c in names if _CLUSTERING_RE.search(c)]
    seen: set[str] = set()
    for col in ordered:
        if col in seen or col == column or col in exclude or col not in obs.columns:
            continue
        seen.add(col)
        if _is_label(obs[col]) and _splits(obs[col], mask):
            return col
    return None


def _pct(share: float) -> str:
    p = share * 100
    if p == 0:
        return "0%"
    return f"{p:.1f}%" if p < 1 else f"{p:.0f}%"


def _name(column: str, group: str) -> str:
    """A group as a sentence names it: a cell type by itself, a numbered cluster with its column."""
    return f"'{column}' group {group}" if re.fullmatch(r"-?\d+", group) else group


def _result(column: str | None, label_a: str, label_b: str, *, auto: bool,
            note: str | None = None, **rest) -> dict:
    out = {"column": column, "label_a": label_a, "label_b": label_b, "auto": auto,
           "n_a": 0, "n_b": 0, "groups": [], "n_groups": 0, "warn": False,
           "message": None, "note": note, "threshold_points": WARN_POINTS,
           "min_cells": MIN_CELLS, "max_p": MAX_P, "n_missing": 0, "nested": False}
    out.update(rest)
    return out


def _message(column: str, flagged: list[dict], label_a: str, label_b: str) -> str:
    top = flagged[0]
    name = _name(column, top["group"])
    text = (f"{name} is {_pct(top['share_a'])} of {label_a} cells but {_pct(top['share_b'])} of "
            f"{label_b} cells ({top['diff_points']:+.0f} points): genes specific to {name} will look "
            f"like {label_a} vs {label_b} differences.")
    if top["one_sided"]:
        rare = label_b if top["share_a"] > top["share_b"] else label_a
        text += (f" It is nearly absent from {rare}, so it may instead be a state the {label_a} vs "
                 f"{label_b} difference creates (its genes then belong to that difference): check "
                 "the composition by a cell-type column too.")
    if len(flagged) > 1:
        more = len(flagged) - 1
        text += (f" {more} more group{'s' if more > 1 else ''} differ{'' if more > 1 else 's'} by more "
                 f"than {WARN_POINTS:g} points.")
    return text + " Compare within one group, or test composition (Compare > Composition)."


def for_comparison(obs: pd.DataFrame, mask_a: np.ndarray, mask_b: np.ndarray,
                   label_a: str, label_b: str, *, split_by: str,
                   column: str | None = None, default: str | None = None,
                   within_column: str | None = None, within_value: str | None = None) -> dict:
    """Each side's share of every ``column`` group (``default``, the active
    clustering, when None) among the ``mask_a | mask_b`` cells, and whether
    the DE of A vs B then partly measures composition."""
    auto = column is None
    column = column or default
    if not column or column not in obs.columns:
        return _result(None, label_a, label_b, auto=auto,
                       note="No clustering or label column to check composition by.")
    mask_a = np.asarray(mask_a, dtype=bool)
    mask_b = np.asarray(mask_b, dtype=bool)
    keep = mask_a | mask_b
    notes: list[str] = []

    if within_column and column == within_column:
        scope = f" ({within_value})" if within_value is not None else ""
        finer = finer_column(obs, column, keep, {split_by, within_column})
        if finer is None:
            return _result(None, label_a, label_b, auto=auto, note=(
                f"The comparison is already within one '{column}' group{scope}, and no finer label "
                f"column (a subclustering such as '{column}_sub', or a QC state) splits it, so "
                "composition was not checked."))
        notes.append(f"Within one '{column}' group{scope}, composition is checked by '{finer}'.")
        column = finer

    if column == split_by:
        return _result(column, label_a, label_b, auto=auto, note=(
            f"Composition by '{column}', the compared column itself, is the comparison: each side "
            "is all one group. Pick another label column to check composition by."))

    labels = obs[column].to_numpy()
    codes, uniques = pd.factorize(labels[keep], use_na_sentinel=True)
    side_a = mask_a[keep]
    valid = codes >= 0
    n_missing = int((~valid).sum())
    k = len(uniques)
    count_a = np.bincount(codes[valid & side_a], minlength=k)
    count_b = np.bincount(codes[valid & ~side_a], minlength=k)
    n_a, n_b = int(count_a.sum()), int(count_b.sum())
    if n_missing:
        notes.append(f"{n_missing:,} compared cells have no '{column}' label and are left out.")
    if n_a == 0 or n_b == 0:
        empty = label_a if n_a == 0 else label_b
        notes.append(f"No {empty} cells carry a '{column}' label, so there is nothing to compare.")
        return _result(column, label_a, label_b, auto=auto, n_a=n_a, n_b=n_b, n_missing=n_missing,
                       note=" ".join(notes))

    share_a = count_a / n_a
    share_b = count_b / n_b
    diff = (share_a - share_b) * 100
    total = share_a + share_b
    dominance = np.where(total > 0, np.maximum(share_a, share_b) / np.where(total > 0, total, 1), 0)
    flagged = (np.abs(diff) > WARN_POINTS) & (np.maximum(count_a, count_b) >= MIN_CELLS)
    pvals = np.full(len(diff), np.nan)
    for i in np.flatnonzero(flagged):  # only the candidates: one 2x2 table each
        a, b = int(count_a[i]), int(count_b[i])
        pvals[i] = fisher_exact([[a, n_a - a], [b, n_b - b]])[1]
    flagged &= np.nan_to_num(pvals, nan=1.0) < MAX_P
    present = (count_a + count_b) > 0
    nested = bool(present.any() and (dominance[present] >= NESTED).all())

    # Largest difference first (as shown, so float noise does not reorder ties), then by name.
    order = np.lexsort((np.array([str(u) for u in uniques]), -np.round(np.abs(diff), 1)))
    groups = [{
        "group": str(uniques[i]),
        "n_a": int(count_a[i]), "n_b": int(count_b[i]),
        "share_a": round(float(share_a[i]), 4), "share_b": round(float(share_b[i]), 4),
        "diff_points": round(float(diff[i]), 1),
        "p": None if np.isnan(pvals[i]) else float(f"{pvals[i]:.3g}"),
        "flagged": bool(flagged[i]) and not nested,
        "one_sided": bool(dominance[i] >= ONE_SIDED),
    } for i in order]
    if len(groups) > MAX_GROUPS:
        notes.append(f"{len(groups)} '{column}' groups; the {MAX_GROUPS} with the largest "
                     "difference are listed (the check used all).")

    warn = bool(flagged.any()) and not nested
    message = None
    if nested:
        notes.append(f"Each '{column}' group lies (almost) entirely on one side, so the two sides are "
                     f"different '{column}' groups: composition is the comparison itself, not a "
                     "confounder.")
    elif warn:
        message = _message(column, [g for g in groups if g["flagged"]], label_a, label_b)
    return _result(column, label_a, label_b, auto=auto, n_a=n_a, n_b=n_b,
                   groups=groups[:MAX_GROUPS], n_groups=len(groups), warn=warn,
                   message=message, note=" ".join(notes) or None, n_missing=n_missing,
                   nested=nested)
