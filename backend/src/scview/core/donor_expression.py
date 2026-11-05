"""One gene's expression per donor within each group, for a box + strip plot
with one point per donor (the Unified View violin's "Per donor" mode).

A violin of cells hides how many people the cells came from: 3,000 cells of
two donors look as solid as 3,000 cells of eight. Here each (donor, condition)
sample in each group is one point, so the spread between donors, which is
what a replicate-aware test compares, is what the plot shows.

Two pseudobulk values are offered:

- ``mean`` (default): the mean of the log-normalised expression over the
  sample's cells. It is on the violin's own scale, needs no counts (a processed
  file often has none left), and gives every cell one vote, so a few very deep
  cells cannot carry a donor.
- ``cpm``: log2(CPM + 1) of the sample's summed raw counts, the quantity a
  pseudobulk DE test (limma-voom or DESeq2 on donor sums, core/pseudobulk.py) works from.
  Deep cells weigh more. Needs raw counts.

A sample with fewer than ``MIN_CELLS`` cells of a group is left out, as in
pseudobulk DE and the composition test: its mean is mostly noise.

When a donor occurs in more than one condition (Kang: every donor in CTRL and
STIM) the design is paired, and ``design`` says so: the plot joins each donor's
values with a line, and with exactly two conditions each group gets a Wilcoxon
signed-rank and a paired t-test on the plotted values (core/paired_design.py).

The tests compare two levels of the condition: ``conditions`` picks them (and
``reference`` the baseline, else the first in the column's order), so a third
level (a second control group, the "Unassigned" cells of a labelled column)
does not stop the test. Without ``conditions``, a column with two levels besides
"Unassigned" tests those two. The other levels stay in ``points``; ``difference``
names the direction of every ``mean_diff`` ("tested - reference").
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
import scipy.sparse as sp

from scview.core import paired_design
from scview.core.replicates import donor_candidates, pick_donor

Method = Literal["mean", "cpm"]

MIN_CELLS = 10  # per donor x condition x group sample
FEW_DONORS = 3  # fewer in a condition: a box says little, read the points
MAX_SAMPLES = 300  # as pseudobulk DE: more and the column is per-cell
MAX_GROUPS = 60
MAX_DROPPED_LISTED = 100  # dropped donor-group samples named in ``dropped``
UNASSIGNED = "Unassigned"  # user_labels.UNASSIGNED: cells a labelling did not reach

VALUE_LABELS = {
    "mean": "mean log-normalised expression per donor",
    "cpm": "log2(CPM + 1) of the donor's summed counts",
}


class DonorExpressionError(ValueError):
    """A request this cannot serve (unknown gene or column, no counts): a 4xx."""


def _labels(s: pd.Series) -> pd.Series:
    return s.astype(str).where(s.notna())


def _order(s: pd.Series, present: set[str]) -> list[str]:
    """The values present, in the column's category order, else sorted naturally."""
    if isinstance(s.dtype, pd.CategoricalDtype):
        return [str(c) for c in s.cat.categories if str(c) in present]

    def key(v: str):
        try:
            return (0, float(v), v)
        except ValueError:
            return (1, 0.0, v)

    return sorted(present, key=key)


def _column(M, j: int) -> np.ndarray:
    col = M[:, j]
    col = col.toarray() if sp.issparse(col) else np.asarray(col)
    return np.asarray(col, dtype=np.float64).ravel()


def _non_negative(M) -> bool:
    if sp.issparse(M):
        sample = M.data[:200_000]
    else:  # dense, or on disk (h5py, backed): the first rows
        head = M[:1000]
        sample = head.data if sp.issparse(head) else np.asarray(head)
    return sample.size == 0 or float(np.min(sample)) >= 0


def lognorm_gene(adata, gene: str) -> tuple[np.ndarray, str]:
    """One gene's log-normalised expression per cell, and where it came from:
    ``.raw`` or X when they hold log values (never counts, z-scores or
    residuals), else the shifted log of the counts (log1p of counts per 10k)."""
    from scview.core.pipeline import looks_like_counts

    def lognorm(M) -> bool:
        return not looks_like_counts(M) and _non_negative(M)

    sources = []
    if adata.raw is not None:
        sources.append((adata.raw.X, adata.raw.var_names, "log-normalised .raw"))
    sources.append((adata.X, adata.var_names, "log-normalised X"))
    for M, names, where in sources:
        if gene in names and lognorm(M):
            return _column(M, names.get_loc(gene)), where
    if not any(gene in names for _, names, _ in sources):
        raise DonorExpressionError(f"Gene '{gene}' not found.")
    c, totals, source = _counts_gene(adata, gene, for_lognorm=True)
    scale = np.divide(1e4, totals, out=np.zeros_like(totals), where=totals > 0)
    return np.log1p(c * scale), f"shifted log of the counts in {source}"


def _counts_gene(adata, gene: str, for_lognorm: bool = False) -> tuple[np.ndarray, np.ndarray, str]:
    """(the gene's counts per cell, each cell's total counts, where from)."""
    from scview.core.pseudobulk import PseudobulkError, raw_counts

    try:
        M, names, source = raw_counts(adata)
    except PseudobulkError:
        raise DonorExpressionError(
            "This dataset has no log-normalised expression and no raw counts to show per "
            "donor (X looks scaled): run normalisation first." if for_lognorm else
            "log2 CPM sums raw counts, and this dataset has none (not in layers['counts'], "
            ".raw or X). Use the mean of the log-normalised expression instead."
        ) from None
    names = pd.Index(names)
    if gene not in names:
        raise DonorExpressionError(f"Gene '{gene}' not found in the raw counts ({source}).")
    if hasattr(M, "row_blocks"):  # a large dataset's counts on disk: one pass by row blocks
        from scview.core.large_views import column_and_totals

        col, totals = column_and_totals(M, names.get_loc(gene))
        return col, totals, source
    totals = np.asarray(M.sum(axis=1), dtype=np.float64).ravel()
    return _column(M, names.get_loc(gene)), totals, source


def detect_sample_key(obs: pd.DataFrame, group_key: str, condition_key: str | None) -> str | None:
    """The most donor-like column with 2+ donors in every condition (all cells
    without one), as the composition test picks it."""
    exclude = {group_key} | ({condition_key} if condition_key else set())
    candidates = donor_candidates(obs, exclude)
    if condition_key:
        cond = _labels(obs[condition_key])
        masks = [(cond == v).to_numpy() for v in cond.dropna().unique()]
    else:
        masks = [np.ones(len(obs), bool)]
    return pick_donor(obs, candidates, masks)


def _tested_levels(levels: list[str], condition_key: str, conditions: list[str] | None,
                   reference: str | None) -> tuple[list[str] | None, str | None]:
    """(the two levels the tests compare, reference first, or None; a note on the choice)."""
    if conditions:
        chosen = list(dict.fromkeys(str(c) for c in conditions))
        unknown = [c for c in chosen if c not in levels]
        if unknown:
            raise DonorExpressionError(
                f"'{unknown[0]}' is not a value of '{condition_key}' in these cells "
                f"(values: {', '.join(levels[:12])}).")
        if len(chosen) != 2:
            raise DonorExpressionError(
                f"Pick exactly two values of '{condition_key}' to compare (got {len(chosen)}).")
        note = None
    elif len(levels) == 2:
        chosen, note = list(levels), None
    else:
        named = [c for c in levels if c != UNASSIGNED]
        if len(named) == 2 and len(levels) == 3:
            chosen = named
            note = (f"'{UNASSIGNED}' cells are shown but left out of the tests, which compare "
                    f"{named[0]} and {named[1]}.")
        elif reference is not None:
            raise DonorExpressionError(
                f"'{condition_key}' has {len(levels)} values: name the two to compare in "
                "conditions, with the reference among them.")
        else:
            return None, (f"'{condition_key}' has {len(levels)} values and a test compares two: "
                          "pick them (conditions) for a donor-level test.") if len(levels) > 2 else None
    if reference is not None:
        if str(reference) not in chosen:
            raise DonorExpressionError(
                f"The reference '{reference}' is not one of the compared values "
                f"({' and '.join(chosen)}).")
        chosen = [str(reference), *(c for c in chosen if c != str(reference))]
    return chosen, note


def per_donor_expression(
    adata,
    gene: str,
    group_key: str,
    *,
    sample_key: str | None = None,
    condition_key: str | None = None,
    method: Method = "mean",
    conditions: list[str] | None = None,
    reference: str | None = None,
    groups: list[str] | None = None,
) -> dict:
    """One point per (donor, condition) sample in each group of ``group_key``.

    ``sample_key`` None detects the donor column; when there is none the result
    has ``sample_key`` None, no points and a message (a dataset without donors is
    not an error, it just cannot be shown this way). ``conditions`` (two values)
    and ``reference`` choose the tested levels; ``groups`` keeps only those groups,
    so donors and dropped samples are counted over them alone."""
    obs = adata.obs
    for col in (group_key, sample_key, condition_key):
        if col and col not in obs.columns:
            raise DonorExpressionError(f"Column '{col}' not found.")
    if method not in VALUE_LABELS:
        raise DonorExpressionError(f"Unknown method '{method}' (mean or cpm).")
    if condition_key and condition_key in (group_key, sample_key):
        condition_key = None
    if (conditions or reference) and not condition_key:
        raise DonorExpressionError("Choosing conditions or a reference needs a condition column.")
    detected = False
    if not sample_key:
        sample_key = detect_sample_key(obs, group_key, condition_key)
        detected = sample_key is not None
    if sample_key == group_key:
        raise DonorExpressionError("The donor and group columns must be different.")

    out: dict = {
        "gene": gene, "group_key": group_key, "sample_key": sample_key,
        "sample_key_detected": detected, "condition_key": condition_key,
        "method": method, "value_label": VALUE_LABELS[method], "source": None,
        "min_cells": MIN_CELLS, "groups": [], "conditions": [], "points": [],
        "n_donors": {}, "n_dropped": 0, "dropped": [], "warnings": [], "notes": [],
        "message": None, "tested_conditions": None, "reference": None, "difference": None,
        "design": {"paired": False, "n_pairs": 0, "conditions": []}, "paired_tests": {},
        "unpaired_tests": {},
    }
    if sample_key is None:
        out["message"] = (
            "Per-donor values need a sample or donor column (one value per person or "
            "sample), and none was found in this dataset.")
        return out

    if method == "mean":
        values, out["source"] = lognorm_gene(adata, gene)
        weights = None
    else:
        values, weights, out["source"] = _counts_gene(adata, gene)

    group = _labels(obs[group_key])
    donor = _labels(obs[sample_key])
    cond = _labels(obs[condition_key]) if condition_key else pd.Series("", index=obs.index)
    keep = (group.notna() & donor.notna() & cond.notna()).to_numpy()
    if groups:
        wanted = {str(x) for x in groups}
        present = set(group.dropna().unique())
        if not wanted & present:
            raise DonorExpressionError(
                f"None of the groups ({', '.join(sorted(wanted)[:5])}) is in '{group_key}'.")
        out["groups_not_found"] = sorted(wanted - present)
        keep &= group.isin(wanted).to_numpy()
    if not keep.any():
        raise DonorExpressionError("No cells have all of the chosen columns set.")
    g, d, c = group.to_numpy()[keep], donor.to_numpy()[keep], cond.to_numpy()[keep]
    v = values[keep]
    w = weights[keep] if weights is not None else None

    group_order = _order(obs[group_key], set(g))
    levels = _order(obs[condition_key], set(c)) if condition_key else [""]
    if len(group_order) > MAX_GROUPS:
        raise DonorExpressionError(
            f"'{group_key}' has {len(group_order)} groups; per-donor plots take at most {MAX_GROUPS}.")
    n_samples = len(set(zip(d, c)))
    if n_samples > MAX_SAMPLES:
        raise DonorExpressionError(
            f"{n_samples} samples is too many (at most {MAX_SAMPLES}); '{sample_key}' looks "
            "like a per-cell column rather than a donor or sample.")
    tested: list[str] | None = None
    if condition_key:
        tested, note = _tested_levels(levels, condition_key, conditions, reference)
        if note:
            out["notes"].append(note)

    frame = pd.DataFrame({"g": g, "d": d, "c": c, "v": v, "w": w if w is not None else 0.0})
    agg = frame.groupby(["g", "c", "d"], sort=False, observed=True).agg(
        n=("v", "size"), v=("v", "sum"), w=("w", "sum"))
    points, dropped = [], []
    for (gi, ci, di), row in agg.iterrows():
        if row["n"] < MIN_CELLS:
            dropped.append({"group": gi, "condition": ci, "sample": di, "n_cells": int(row["n"])})
            continue
        if method == "mean":
            value = row["v"] / row["n"]
        else:
            value = np.log2(row["v"] / row["w"] * 1e6 + 1) if row["w"] > 0 else 0.0
        points.append({"group": gi, "condition": ci, "sample": di,
                       "n_cells": int(row["n"]), "value": round(float(value), 5)})
    gpos = {x: i for i, x in enumerate(group_order)}
    cpos = {x: i for i, x in enumerate(levels)}
    points.sort(key=lambda p: (gpos[p["group"]], cpos[p["condition"]], p["sample"]))
    dropped.sort(key=lambda p: (gpos[p["group"]], cpos[p["condition"]], p["sample"]))

    out["groups"], out["conditions"], out["points"] = group_order, levels, points
    out["n_dropped"], out["dropped"] = len(dropped), dropped[:MAX_DROPPED_LISTED]
    out["n_donors"] = {k: len({p["sample"] for p in points if p["condition"] == k})
                       for k in levels}
    # Pairing is the dataset's design (which donors were sampled in which of the
    # tested conditions), so it is read off the cells, not the samples kept; the
    # tests use the pairs with a value in both conditions in each group.
    if condition_key:
        out["design"] = paired_design.detect_design(set(zip(d, c)), tested or levels)
    if tested:
        out["tested_conditions"], out["reference"] = tested, tested[0]
        out["difference"] = f"{tested[1]} - {tested[0]}"
        if out["design"]["paired"]:
            out["paired_tests"] = paired_design.group_paired_tests(points, tested, group_order)
        else:  # different donors in each: rank-sum + Welch on donor values
            out["unpaired_tests"] = paired_design.group_unpaired_tests(points, tested, group_order)
    few = {k: n for k, n in out["n_donors"].items() if n < FEW_DONORS}
    if few:
        where = ", ".join(f"{k or 'all cells'}: {n}" for k, n in few.items())
        out["warnings"].append(
            f"Fewer than {FEW_DONORS} donors ({where}): a box of so few points shows their "
            "range, not a distribution, and cannot separate a condition effect from those "
            "donors' own levels. Read the points.")
    return out
