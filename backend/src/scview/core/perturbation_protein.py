"""Protein check of a Mixscape run: regulators the transcriptome cannot see.

Mixscape (core/mixscape.py) calls a target's cells KO only when their transcriptome
moved; Papalexi 2021 (Nat Genet 53:322) name the limit in their Discussion: a
perturbation that changes protein levels "but exhibit[s] no evidence of transcriptomic
change will be classified as non-perturbed". On their THP-1 screen CMTM6, which keeps
PD-L1 at the cell surface (Burr 2017, Mezzadra 2017), has no DE genes, so all of its
cells are NP and a KO-filtered analysis drops it, yet its guide cells carry less PD-L1
protein (QA 2026-11-14: -0.19 CLR, lower in 3 of 3 replicates).

With a protein modality (obsm ``protein_clr``, CLR per cell as the paper's Seurat
``NormalizeData(method = "CLR", margin = 2)``), per target and protein:

- the shift of the mean CLR from the controls' in ALL the target's guide cells, in its
  KO cells and in its NP cells;
- a cell-level two-sided Wilcoxon rank-sum test of each against the controls (the
  paper's Fig. 4b test), Benjamini-Hochberg over the targets x proteins of each group;
- with a replicate column (Mixscape's ``replicate_column``, else ``split_by``): the shift
  per replicate (the target's mean minus the controls' mean in the same replicate), and a
  paired t test over the replicates (one-sample t of the shifts; n = replicates with at
  least ``MIN_CELLS`` target and control cells, 3 or more), BH as above.

``significant`` (ALL cells): a mean shift of at least ``MIN_SHIFT`` with cell-level BH
p < 0.05 and, with 3 or more replicates, the shift in the same direction in EVERY
replicate (the replicates as the unit of agreement). The
replicate t test is shown but does not decide: with 3 replicates (df 2) and BH over
the targets x proteins no shift can pass (Papalexi: JAK2 -0.35 in 3 of 3 replicates,
BH 0.17), and uncorrected it splits near-identical patterns by chance (CMTM6 on PD-L1
p 0.046, CD86 on its own protein p 0.065, both 3 of 3).

A target whose ALL-cells shift is significant while Mixscape called (almost) none of its
cells KO (``ko_fraction`` <= ``KO_SHARE_NONE``) is flagged "protein effect without an
RNA KO class". With a small panel the per-cell CLR is compositional: a fall in one
protein lifts the others' values (``SMALL_PANEL``).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

MIN_CELLS = 5  # target or control cells in a group (or in one replicate) for a test
MIN_REPLICATES = 3  # replicates for a replicate-level test
ALPHA = 0.05
MIN_SHIFT = 0.1  # |mean CLR shift| for "significant": about a 10% change against the cell's other proteins
KO_SHARE_NONE = 0.1  # at most this share called KO: "no RNA KO class" (real KO classes: 22% and up)
SMALL_PANEL = 10  # proteins: below this the per-cell CLR is noticeably compositional
MAX_PAIRS = 6000  # target x protein rows kept in the result (the significant ones first)
GROUPS = ("all", "ko", "np")


def _bh(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p values (NaN kept as NaN)."""
    p = np.asarray(p, dtype=np.float64)
    out = np.full(p.shape, np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    if ok.size == 0:
        return out
    order = ok[np.argsort(p[ok], kind="stable")]
    ranked = p[order] * ok.size / np.arange(1, ok.size + 1)
    out[order] = np.minimum(1.0, np.minimum.accumulate(ranked[::-1])[::-1])
    return out


def _wilcoxon(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Two-sided Wilcoxon rank-sum p per column of ``x`` (target cells) vs ``y`` (controls):
    normal approximation with tie and continuity correction (R ``wilcox.test`` default
    for large samples, Seurat ``FindMarkers``)."""
    from scipy.stats import mannwhitneyu

    with np.errstate(all="ignore"):
        p = mannwhitneyu(x, y, alternative="two-sided", method="asymptotic", use_continuity=True, axis=0).pvalue
    p = np.atleast_1d(np.asarray(p, dtype=np.float64))
    same = np.all(np.isclose(np.vstack([x, y]), np.vstack([x, y])[:1]), axis=0)  # a constant column
    p[same] = 1.0
    return p


def _paired_t(d: np.ndarray) -> float:
    """Two-sided one-sample t test of per-replicate shifts against 0 (NaN without one)."""
    from scipy.stats import ttest_1samp

    d = d[np.isfinite(d)]
    if d.size < MIN_REPLICATES:
        return float("nan")
    if np.allclose(d, d[0]):
        return 0.0 if d[0] != 0 else 1.0
    return float(ttest_1samp(d, 0.0).pvalue)


def check(values: np.ndarray, proteins: list[str], labels: np.ndarray, glob: np.ndarray, control: str,
          targets: list[dict], replicates: np.ndarray | None = None, replicate_column: str | None = None,
          layer: str = "protein_clr") -> dict[str, Any]:
    """The protein check (see the module docstring). ``values``: cells x proteins (the
    cells Mixscape used, in its order); ``labels`` each cell's target or ``control``;
    ``glob`` its KO / NP / NT class; ``targets`` Mixscape's per-target rows."""
    from scview.core.mixscape import KO, NP

    V = np.asarray(values, dtype=np.float64)
    is_ctrl = labels == control
    C = V[is_ctrl]
    ko_share = {t["target"]: float(t["ko_fraction"]) for t in targets}
    reps = None
    if replicates is not None:
        reps = np.asarray(replicates).astype(str)
        rep_values = [r for r in pd.unique(reps) if r != ""]
        ctrl_mean = {r: (V[is_ctrl & (reps == r)].mean(axis=0) if (is_ctrl & (reps == r)).sum() >= MIN_CELLS
                         else None) for r in rep_values}
    rows: list[dict[str, Any]] = []
    for t in [t["target"] for t in targets]:
        is_t = labels == t
        masks = {"all": is_t, "ko": is_t & (glob == KO), "np": is_t & (glob == NP)}
        stats: dict[str, Any] = {}
        for g, m in masks.items():
            n = int(m.sum())
            if n < MIN_CELLS or C.shape[0] < MIN_CELLS:
                stats[g] = None
                continue
            X = V[m]
            stats[g] = {"n": n, "diff": X.mean(axis=0) - C.mean(axis=0), "p": _wilcoxon(X, C)}
        per_rep: list[tuple[str, int, np.ndarray]] = []
        if reps is not None:
            for r in rep_values:
                m = is_t & (reps == r)
                if m.sum() >= MIN_CELLS and ctrl_mean[r] is not None:
                    per_rep.append((r, int(m.sum()), V[m].mean(axis=0) - ctrl_mean[r]))
        for j, prot in enumerate(proteins):
            row: dict[str, Any] = {"target": t, "protein": prot, "ko_fraction": ko_share.get(t, 0.0)}
            for g in GROUPS:
                s = stats[g]
                row[g] = None if s is None else {"n": s["n"], "diff": float(s["diff"][j]), "p": float(s["p"][j])}
            if per_rep:
                d = np.array([x[2][j] for x in per_rep])
                if len(per_rep) < MIN_REPLICATES:
                    row["note"] = f"{len(per_rep)} replicate(s) with {MIN_CELLS}+ cells: judged on cells"
                row["replicates"] = [{"replicate": r, "n": n, "diff": float(x[j])} for r, n, x in per_rep]
                row["n_replicates"] = len(per_rep)
                ref = row["all"]["diff"] if row["all"] else float(d.mean())
                row["same_sign"] = int(np.sum(np.sign(d) == np.sign(ref))) if ref != 0 else 0
                p_rep = _paired_t(d)
                row["p_replicate"] = p_rep if np.isfinite(p_rep) else None
            rows.append(row)
    # BH per group (cell level) and over the replicate tests.
    for g in GROUPS:
        idx = [i for i, r in enumerate(rows) if r[g] is not None]
        adj = _bh(np.array([rows[i][g]["p"] for i in idx]))
        for i, a in zip(idx, adj):
            rows[i][g]["padj"] = None if not np.isfinite(a) else float(a)
    idx = [i for i, r in enumerate(rows) if r.get("p_replicate") is not None]
    for i, a in zip(idx, _bh(np.array([rows[i]["p_replicate"] for i in idx]))):
        rows[i]["padj_replicate"] = float(a)
    n_rep_max = max((r.get("n_replicates", 0) for r in rows), default=0)
    for r in rows:
        a = r["all"] or {}
        ok = a.get("padj") is not None and a["padj"] < ALPHA and abs(a.get("diff") or 0.0) >= MIN_SHIFT
        if r.get("n_replicates", 0) >= MIN_REPLICATES:
            r["unit"] = "replicate"
            r["significant"] = bool(ok and r["same_sign"] == r["n_replicates"])
        else:
            r["unit"] = "cell"
            r["significant"] = bool(ok)
        r["flag"] = bool(r["significant"] and r["ko_fraction"] <= KO_SHARE_NONE)
    flags = [_flag_text(r) for r in sorted((r for r in rows if r["flag"]), key=lambda r: _rank(r))]
    n_pairs = len(rows)
    if n_pairs > MAX_PAIRS:
        rows = sorted(rows, key=_rank)[:MAX_PAIRS]
    notes = []
    unit_rep = any(r["unit"] == "replicate" for r in rows)
    if unit_rep:
        notes.append(f"Unit: the {n_rep_max} replicates in '{replicate_column}': a shift counts when it is at "
                     f"least {MIN_SHIFT} in the same direction in every replicate (and cell-level BH p < {ALPHA}). "
                     f"The paired t p over {n_rep_max} replicates is shown, not used: with so few it is descriptive.")
    elif replicate_column:
        notes.append(f"Fewer than {MIN_REPLICATES} replicates in '{replicate_column}' with {MIN_CELLS}+ cells each: "
                     "significance from cells (Wilcoxon, BH), which treats cells as independent.")
    else:
        notes.append("No replicate column (pick one in 'Within'): significance from cells (Wilcoxon, BH), which "
                     "treats cells as independent; a screen's independent units are its replicates.")
    notes.append(f"Shift: mean CLR of the target's cells minus the controls'; a shift counts from {MIN_SHIFT}. "
                 f"Flag: a shift that counts with at most {KO_SHARE_NONE:.0%} of the target's cells called KO.")
    if len(proteins) < SMALL_PANEL:
        notes.append(f"CLR per cell over {len(proteins)} proteins: a fall in one protein lifts the others' values, "
                     "so read shifts of the other proteins with care.")
    if n_pairs > MAX_PAIRS:
        notes.append(f"{MAX_PAIRS:,} of {n_pairs:,} target x protein rows kept (the strongest).")
    return {"layer": layer, "proteins": list(proteins), "n_proteins": len(proteins), "control": control,
            "n_control": int(is_ctrl.sum()), "replicate_column": replicate_column if reps is not None else None,
            "min_cells": MIN_CELLS, "ko_share_none": KO_SHARE_NONE, "alpha": ALPHA, "n_pairs": n_pairs,
            "rows": rows, "flags": flags, "notes": notes, "default_protein": _default_protein(rows, proteins)}


def _rank(r: dict) -> tuple:
    p = (r["all"] or {}).get("padj")
    return (not r["significant"], 1.0 if p is None else p, r["target"], r["protein"])


def _default_protein(rows: list[dict], proteins: list[str]) -> str | None:
    """The protein with the most significant targets (ties: the first in the panel)."""
    if not proteins:
        return None
    n = {p: 0 for p in proteins}
    for r in rows:
        n[r["protein"]] += int(r["significant"])
    return max(proteins, key=lambda p: (n[p], -proteins.index(p)))


def _flag_text(r: dict) -> dict:
    a = r["all"] or {}
    if r["unit"] == "replicate":
        shifts = ", ".join(f"{y['diff']:+.2f}" for y in r["replicates"])
        how = (f"the same way in {r['same_sign']} of {r['n_replicates']} replicates ({shifts}), cell-level BH p "
               f"{a.get('padj', float('nan')):.2g}")
    else:
        how = f"Wilcoxon on {a.get('n', 0):,} cells, BH p {a.get('padj', float('nan')):.2g}"
    text = (f"{r['target']}: {r['protein']} {a.get('diff', 0):+.2f} (CLR) in all {a.get('n', 0):,} guide cells, "
            f"{how}; {r['ko_fraction']:.0%} of its cells called KO on RNA.")
    return {"target": r["target"], "protein": r["protein"], "diff": a.get("diff"), "unit": r["unit"],
            "ko_fraction": r["ko_fraction"], "text": text}


def has_values(adata) -> bool:
    from scview.core import protein

    return protein.has_protein(adata)


def run(adata, rows: np.ndarray, labels: np.ndarray, glob: np.ndarray, control: str, targets: list[dict],
        replicates: np.ndarray | None, replicate_column: str | None) -> dict[str, Any] | None:
    """The check on a dataset's ``protein_clr`` for the cells Mixscape used (``rows``), or
    None without a protein modality."""
    from scview.core import protein

    if not protein.has_protein(adata):
        return None
    m = adata.obsm[protein.CLR_KEY]
    vals = np.asarray(m.to_numpy(dtype=np.float32))[rows]
    return check(vals, [str(c) for c in m.columns], labels, glob, control, targets, replicates,
                 replicate_column, layer=protein.CLR_KEY)
