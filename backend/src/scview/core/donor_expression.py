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
  pseudobulk DE test (DESeq2 on donor sums, core/pseudobulk.py) works from.
  Deep cells weigh more. Needs raw counts.

A sample with fewer than ``MIN_CELLS`` cells of a group is left out, as in
pseudobulk DE and the composition test: its mean is mostly noise.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
import scipy.sparse as sp

from scview.core.replicates import donor_candidates, pick_donor

Method = Literal["mean", "cpm"]

MIN_CELLS = 10  # per donor x condition x group sample
FEW_DONORS = 3  # fewer in a condition: a box says little, read the points
MAX_SAMPLES = 300  # as pseudobulk DE: more and the column is per-cell
MAX_GROUPS = 60

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
    sample = M.data[:200_000] if sp.issparse(M) else np.asarray(M[:1000])
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


def per_donor_expression(
    adata,
    gene: str,
    group_key: str,
    *,
    sample_key: str | None = None,
    condition_key: str | None = None,
    method: Method = "mean",
) -> dict:
    """One point per (donor, condition) sample in each group of ``group_key``.

    ``sample_key`` None detects the donor column; when there is none the result
    has ``sample_key`` None, no points and a message (a dataset without donors is
    not an error, it just cannot be shown this way)."""
    obs = adata.obs
    for col in (group_key, sample_key, condition_key):
        if col and col not in obs.columns:
            raise DonorExpressionError(f"Column '{col}' not found.")
    if method not in VALUE_LABELS:
        raise DonorExpressionError(f"Unknown method '{method}' (mean or cpm).")
    if condition_key and condition_key in (group_key, sample_key):
        condition_key = None
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
        "n_donors": {}, "n_dropped": 0, "warnings": [], "message": None,
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
    if not keep.any():
        raise DonorExpressionError("No cells have all of the chosen columns set.")
    g, d, c = group.to_numpy()[keep], donor.to_numpy()[keep], cond.to_numpy()[keep]
    v = values[keep]
    w = weights[keep] if weights is not None else None

    groups = _order(obs[group_key], set(g))
    conditions = _order(obs[condition_key], set(c)) if condition_key else [""]
    if len(groups) > MAX_GROUPS:
        raise DonorExpressionError(
            f"'{group_key}' has {len(groups)} groups; per-donor plots take at most {MAX_GROUPS}.")
    n_samples = len(set(zip(d, c)))
    if n_samples > MAX_SAMPLES:
        raise DonorExpressionError(
            f"{n_samples} samples is too many (at most {MAX_SAMPLES}); '{sample_key}' looks "
            "like a per-cell column rather than a donor or sample.")

    frame = pd.DataFrame({"g": g, "d": d, "c": c, "v": v, "w": w if w is not None else 0.0})
    agg = frame.groupby(["g", "c", "d"], sort=False, observed=True).agg(
        n=("v", "size"), v=("v", "sum"), w=("w", "sum"))
    points = []
    for (gi, ci, di), row in agg.iterrows():
        if row["n"] < MIN_CELLS:
            out["n_dropped"] += 1
            continue
        if method == "mean":
            value = row["v"] / row["n"]
        else:
            value = np.log2(row["v"] / row["w"] * 1e6 + 1) if row["w"] > 0 else 0.0
        points.append({"group": gi, "condition": ci, "sample": di,
                       "n_cells": int(row["n"]), "value": round(float(value), 5)})
    gpos = {x: i for i, x in enumerate(groups)}
    cpos = {x: i for i, x in enumerate(conditions)}
    points.sort(key=lambda p: (gpos[p["group"]], cpos[p["condition"]], p["sample"]))

    out["groups"], out["conditions"], out["points"] = groups, conditions, points
    out["n_donors"] = {k: len({p["sample"] for p in points if p["condition"] == k})
                       for k in conditions}
    few = {k: n for k, n in out["n_donors"].items() if n < FEW_DONORS}
    if few:
        where = ", ".join(f"{k or 'all cells'}: {n}" for k, n in few.items())
        out["warnings"].append(
            f"Fewer than {FEW_DONORS} donors ({where}): a box of so few points shows their "
            "range, not a distribution, and cannot separate a condition effect from those "
            "donors' own levels. Read the points.")
    return out
