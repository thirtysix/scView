"""Do the annotation methods agree? A cross-check of 2-6 label columns (C005).

Each method (CellTypist, the LLM, marker scores, a clustering, hand labels)
writes its own column, in its own vocabulary: "CD14 Mono" in one is "Classical
monocytes" in another. The check reports:

- ARI and AMI between every pair of columns (vocabulary-free: they compare the
  partitions, not the names);
- for one reference column against each other one: the contingency table, and
  for each reference label its best-matching label by Jaccard with what became
  of it (stable / splits / merges / dissolves, ``hvg_sensitivity.compare``);
- per-cell agreement. Each other column's labels are mapped onto reference
  labels by majority overlap (a label goes to the reference label most of its
  cells carry), so a renamed or finer labelling agrees fully. A cell's agreement
  is the share of the other columns whose mapped label is its reference label,
  and its consensus is the plurality of the reference label and the mapped
  labels ("Unresolved" on a tie);
- exact-name agreement, apart from the above: the share of cells whose labels
  are the same name after normalising case, spaces, "+" and a plural "s". Zero
  for columns that share no names, however well they agree.

Mapping by majority is one-way: a finer column (sub-clusters of one type)
agrees fully, while a coarser one ("T cell" over the reference's CD4 and CD8)
maps to whichever reference label is the bigger part and disagrees on the rest.
Agreement measures consistency with the reference, not which column is right.

Everything is keyed by label, never by position.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any

import numpy as np
import pandas as pd

from scview.core.hvg_sensitivity import compare

logger = logging.getLogger(__name__)

MIN_COLUMNS = 2
MAX_COLUMNS = 6
MAX_CATEGORIES = 200
UNRESOLVED = "Unresolved"
TOP_CALLS = 3  # what an annotator calls a reference label: its commonest labels there

NOTE = (
    "Labels of each column are mapped to the reference label most of their cells carry, so "
    "different names for the same cells agree. Agreement is consistency with the reference, "
    "not correctness: a coarser column agrees only on the bigger part of what it merges."
)


class CrosscheckError(Exception):
    """A request the check cannot serve; ``status`` is the HTTP code to answer with."""

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


# ---------------------------------------------------------------------------
# Columns
# ---------------------------------------------------------------------------


def _labels(adata, name: str) -> tuple[np.ndarray, list[str]]:
    """(codes, names) of a categorical column: names in the column's category
    order (or sorted, for strings), present labels only; -1 for missing."""
    if name not in adata.obs.columns:
        raise CrosscheckError(f"Column '{name}' not found.", status=404)
    col = adata.obs[name]
    categorical = isinstance(col.dtype, pd.CategoricalDtype)
    if not categorical and pd.api.types.is_numeric_dtype(col) and not pd.api.types.is_bool_dtype(col):
        raise CrosscheckError(
            f"'{name}' holds numbers, not labels; pick a categorical column (cell types, clusters)."
        )
    present = col.notna().to_numpy()
    text = np.full(col.size, None, dtype=object)
    text[present] = col[present].map(str).to_numpy(dtype=object)
    seen = set(text[present])
    if categorical:
        order = list(dict.fromkeys(str(c) for c in col.cat.categories if str(c) in seen))
    else:
        order = sorted(seen)
    if len(order) > MAX_CATEGORIES:
        raise CrosscheckError(
            f"'{name}' has {len(order)} labels; the cross-check takes columns with up to "
            f"{MAX_CATEGORIES}."
        )
    if len(order) < 2:
        raise CrosscheckError(f"'{name}' has fewer than two labels; there is nothing to compare.")
    codes = np.asarray(pd.Categorical(text, categories=order).codes, dtype=np.int64)
    return codes, order


def _check_request(columns: list[str], reference: str | None) -> tuple[list[str], str]:
    cols = list(dict.fromkeys(str(c) for c in columns))
    if not MIN_COLUMNS <= len(cols) <= MAX_COLUMNS:
        raise CrosscheckError(f"Pick {MIN_COLUMNS} to {MAX_COLUMNS} different columns to compare.")
    ref = reference or cols[0]
    if ref not in cols:
        raise CrosscheckError(f"The reference '{ref}' must be one of the compared columns.")
    return cols, ref


def normalise_label(label: str) -> str:
    """A label's name for exact-name agreement: lower case, no "+", words joined
    without spaces, a plural "s" dropped from each word ("CD4+ T cells" and
    "cd4 T cell" are the same name)."""
    words = re.split(r"[\s_]+", str(label).lower().replace("+", " ").strip())
    out = [w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w for w in words if w]
    return "".join(out)


# ---------------------------------------------------------------------------
# The check
# ---------------------------------------------------------------------------


def _table(r: np.ndarray, c: np.ndarray, k_r: int, k_c: int) -> np.ndarray:
    keep = (r >= 0) & (c >= 0)
    return np.bincount(r[keep] * k_c + c[keep], minlength=k_r * k_c).reshape(k_r, k_c)


def _mapping(table: np.ndarray) -> np.ndarray:
    """Per other label (column of ``table``), the reference label (row) holding
    most of its cells; of two holding as many, the one it is the bigger share of,
    then the first. -1 for a label no reference label overlaps."""
    size_r = np.maximum(table.sum(axis=1), 1)
    out = np.full(table.shape[1], -1, dtype=np.int64)
    rows = np.arange(table.shape[0])
    for j in range(table.shape[1]):
        if table[:, j].sum() == 0:
            continue
        # lexsort: last key primary; -rows makes the first row win a full tie
        out[j] = int(np.lexsort((-rows, table[:, j] / size_r, table[:, j]))[-1])
    return out


def _consensus(votes: np.ndarray) -> np.ndarray:
    """Plurality of each row's votes (codes, -1 = no vote): the label, -2 on a
    tie, -1 when the row has no votes."""
    valid = votes >= 0
    k = votes.shape[1]
    counts = np.zeros(votes.shape, dtype=np.int64)
    for u in range(k):
        counts += valid & valid[:, [u]] & (votes == votes[:, [u]])
    counts[~valid] = 0
    best = counts.max(axis=1)
    first = np.argmax(counts == best[:, None], axis=1)
    label = votes[np.arange(votes.shape[0]), first]
    tied = ((counts == best[:, None]) & (votes != label[:, None])).any(axis=1)
    return np.where(best == 0, -1, np.where(tied, -2, label))


def _compute(adata, columns: list[str], reference: str | None):
    cols, ref = _check_request(columns, reference)
    labs = {c: _labels(adata, c) for c in cols}
    ref_codes, ref_names = labs[ref]
    others = [c for c in cols if c != ref]
    k_r = len(ref_names)
    ref_norm = [normalise_label(x) for x in ref_names]

    mapped, exact, per = {}, {}, {}
    for o in others:
        codes, names = labs[o]
        table = _table(ref_codes, codes, k_r, len(names))
        to = _mapping(table)
        m = np.where(codes >= 0, to[np.maximum(codes, 0)], -1)
        mapped[o] = m
        norm = np.array([normalise_label(x) for x in names] + [""], dtype=object)
        ref_n = np.array(ref_norm + [""], dtype=object)
        both = (ref_codes >= 0) & (codes >= 0)
        exact[o] = np.where(both, norm[codes] == ref_n[ref_codes], False)
        per[o] = (codes, names, table, to, both)

    M = np.column_stack([mapped[o] for o in others])
    has_ref = ref_codes >= 0
    voted = M >= 0
    den = voted.sum(axis=1)
    num = (voted & (M == ref_codes[:, None])).sum(axis=1)
    compared = has_ref & (den > 0)
    agreement = np.full(adata.n_obs, np.nan)
    agreement[compared] = num[compared] / den[compared]
    cons = _consensus(np.column_stack([ref_codes, M]))
    return cols, ref, labs, others, per, mapped, exact, agreement, compared, cons


def _consensus_labels(cons: np.ndarray, ref_names: list[str]) -> pd.Categorical:
    out = np.full(cons.size, None, dtype=object)
    names = np.array(ref_names, dtype=object)
    out[cons >= 0] = names[cons[cons >= 0]]
    out[cons == -2] = UNRESOLVED
    cats = [n for n in ref_names if n != UNRESOLVED] + [UNRESOLVED]
    return pd.Categorical(out, categories=cats).remove_unused_categories()


def _r(x: float) -> float:
    return round(float(x), 4)


def crosscheck(adata, columns: list[str], reference: str | None = None) -> dict[str, Any]:
    """Compare ``columns`` (see the module doc); ``reference`` defaults to the first."""
    from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

    t0 = time.perf_counter()
    cols, ref, labs, others, per, mapped, exact, agreement, compared, cons = _compute(
        adata, columns, reference)
    ref_codes, ref_names = labs[ref]

    ari: dict[str, dict[str, float]] = {c: {c: 1.0} for c in cols}
    ami: dict[str, dict[str, float]] = {c: {c: 1.0} for c in cols}
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            ca, cb = labs[a][0], labs[b][0]
            keep = (ca >= 0) & (cb >= 0)
            va = _r(adjusted_rand_score(ca[keep], cb[keep])) if keep.any() else None
            vm = _r(adjusted_mutual_info_score(ca[keep], cb[keep])) if keep.any() else None
            ari[a][b] = ari[b][a] = va
            ami[a][b] = ami[b][a] = vm

    comparisons: dict[str, Any] = {}
    for o in others:
        codes, names, table, to, both = per[o]
        cmp = compare(ref_codes, ref_names, pd.Categorical.from_codes(codes, categories=names))
        size_c = table.sum(axis=0)
        size_r = table.sum(axis=1)
        counts = {ref_names[i]: {names[j]: int(table[i, j]) for j in np.flatnonzero(table[i])}
                  for i in range(len(ref_names)) if size_r[i]}
        fraction = {r: {n: _r(v / size_r[ref_names.index(r)]) for n, v in row.items()}
                    for r, row in counts.items()}
        n_both = int(both.sum())
        shared = sorted({normalise_label(n) for n in names} & {normalise_label(n) for n in ref_names})
        comparisons[o] = {
            "ari": cmp["ari"],
            "ami": cmp["ami"],
            "n_cells": n_both,
            "mapped_agreement": _r((mapped[o][both] == ref_codes[both]).mean()) if n_both else None,
            "mapping": {names[j]: {"to": ref_names[to[j]], "share": _r(table[to[j], j] / size_c[j])}
                        for j in range(len(names)) if to[j] >= 0},
            "exact_name": {
                "n_shared_labels": len(shared),
                "shared_labels": [n for n in ref_names if normalise_label(n) in set(shared)],
                "agreement": _r(exact[o][both].mean()) if n_both else None,
            },
            "best_match": {row["cluster"]: {k: v for k, v in row.items() if k != "cluster"}
                           for row in cmp["clusters"]},
            "contingency": {
                "rows": [n for n in ref_names if counts.get(n)],
                "cols": [names[j] for j in range(len(names)) if size_c[j]],
                "counts": counts,
                "row_fraction": fraction,
            },
        }

    labels: dict[str, Any] = {}
    for i, name in enumerate(ref_names):
        cells = ref_codes == i
        n = int(cells.sum())
        if not n:
            continue
        mine = cells & compared
        annotators = {}
        for o in others:
            codes, names, _, to, _ = per[o]
            has = cells & (codes >= 0)
            k = int(has.sum())
            if not k:
                continue
            share = float((mapped[o][has] == i).mean())
            calls = pd.Series(codes[has]).value_counts().head(TOP_CALLS)
            annotators[o] = {
                "agreement": _r(share),
                "disagrees": share < 0.5,
                "calls": [{"label": names[c], "share": _r(v / k),
                           "maps_to": ref_names[to[c]] if to[c] >= 0 else None}
                          for c, v in calls.items()],
            }
        labels[name] = {
            "n_cells": n,
            "mean_agreement": _r(np.nanmean(agreement[mine])) if mine.any() else None,
            "disagreeing": [o for o, a in annotators.items() if a["disagrees"]],
            "annotators": annotators,
        }

    consensus = _consensus_labels(cons, ref_names)
    vc = pd.Series(agreement[compared]).round(4).value_counts().sort_index()
    any_exact = np.column_stack([exact[o] for o in others])
    both_any = np.column_stack([per[o][4] for o in others])
    n_exact = both_any.sum(axis=1)
    exact_cell = np.where(n_exact > 0, any_exact.sum(axis=1) / np.maximum(n_exact, 1), np.nan)
    has_exact = compared & (n_exact > 0)
    out = {
        "columns": cols,
        "reference": ref,
        "n_cells": int(adata.n_obs),
        "n_compared": int(compared.sum()),
        "pairwise": {"ari": ari, "ami": ami},
        "mean_agreement": _r(agreement[compared].mean()) if compared.any() else None,
        "agreement_counts": {f"{k:g}": int(v) for k, v in vc.items()},
        "exact_name_agreement": _r(np.mean(exact_cell[has_exact])) if has_exact.any() else None,
        "consensus_counts": {str(k): int(v) for k, v in pd.Series(consensus).value_counts().items()},
        "n_unresolved": int((cons == -2).sum()),
        "comparisons": comparisons,
        "labels": labels,
        "note": NOTE,
        "unresolved_label": UNRESOLVED,
        "elapsed_seconds": 0.0,
    }
    out["elapsed_seconds"] = round(time.perf_counter() - t0, 3)
    logger.info("Annotation cross-check of %s (reference %s): mean agreement %s; %.2f s",
                cols, ref, out["mean_agreement"], out["elapsed_seconds"])
    return out


# ---------------------------------------------------------------------------
# The info for the picker
# ---------------------------------------------------------------------------


def annotation_columns(adata, active_clustering: str | None) -> dict[str, str]:
    """Columns that look like annotations, with the method that made them: a
    CellTypist run, an LLM or marker-score mapping, and the active clustering."""
    from scview.core import celltypist_fit

    obs = adata.obs.columns
    found: dict[str, str] = {}
    for c in celltypist_fit.run_columns(adata):
        found[c] = "CellTypist"
    for key in adata.uns.keys():
        for suffix, method in (("_llm_mapping", "LLM"), ("_markerscore_mapping", "marker score")):
            if key.endswith(suffix) and key[: -len(suffix)] in obs:
                found.setdefault(key[: -len(suffix)], method)
    if active_clustering and active_clustering in obs:
        found.setdefault(active_clustering, "clustering")
    return found


def info(adata, active_clustering: str | None) -> dict[str, Any]:
    return {
        "suggested": annotation_columns(adata, active_clustering),
        "active_clustering": active_clustering,
        "min_columns": MIN_COLUMNS,
        "max_columns": MAX_COLUMNS,
        "max_categories": MAX_CATEGORIES,
        "note": NOTE,
    }
