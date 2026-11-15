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
RNA KO class".

**Which values** (``layer``; ``auto`` by panel size):

- ``protein_clr``: the per-cell CLR (Papalexi's and the Seurat Mixscape vignette's
  ``NormalizeData(method = "CLR", margin = 2)``). Each cell's values are divided by the
  geometric mean of its OWN panel, so on a small panel it is compositional: a fall in one
  protein lifts the others. Papalexi (4 proteins; v0.80 measurement): the IFN-gamma pathway
  KOs "raise" CD86 +0.12 to +0.13 (3 of 3 replicates) and PDCD1LG2's own PD-L2 fall reads
  -0.07 (not significant). The shift of that reference per target is reported
  (``reference_shift``: the cells' mean log1p over the panel minus the controls').
- ``protein_counts``: log1p of the antibody counts, not normalised per cell. Not
  compositional: one protein's fall cannot move another. Guides are assigned at random, so
  droplet capture is balanced between a target and its controls, and the replicate unit
  compares within a replicate. Papalexi: CD86 in the IFN-gamma KOs -0.02 to -0.11, PD-L2
  falls with them (-0.27 to -0.35: it is IFN-gamma inducible too), PDCD1LG2 PD-L2 -0.31
  (3 of 3, flagged), CMTM6 PD-L1 -0.43 (flagged). Rejected scales: counts over the cell's
  total protein (the same compositional problem on 4 proteins, worse here: CD86 +0.33 and
  CD366 +0.38 in the IFN-gamma KOs) and over its RNA UMIs (not compositional, but the
  control cells' SD rose on all 4 proteins, +0% to +17%: log RNA and log protein correlate
  only 0.10 to 0.33 here, and a KO that shrinks the cell, MYC at 0.89x RNA, moves every
  protein).
- ``protein_dsb``: DSB values (core/dsb.py) when the dataset has them. Empty droplets give
  the background; without them dsb's ModelNegativeADTnorm step I is log counts minus a
  per-protein constant (the same test as ``protein_counts``). Its step II removes a
  per-cell technical component fitted on the cell's own proteins: compositional on a small
  panel too (Papalexi, no empty droplets: CD86 +0.24 and CD366 +0.27 in the IFN-gamma KOs).

``auto`` takes ``protein_counts`` below ``SMALL_PANEL`` proteins (when the counts are
there) and ``protein_clr`` from it. The cut is measured (pbmc_10k_protein_v3, a planted
-0.9 log fall of one abundant protein in a fifth of the cells, the size of the strongest
Papalexi shift, JAK2 on PD-L1): the largest CLR shift of an UNCHANGED protein was 0.146
on 4 proteins, 0.104 on 10, 0.101 on 12 and 0.092 on 14; unchanged proteins were called
significant on panels of 12 or fewer (9 of 15 on 4, 1 of 55 on 12), none on 14.
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
SMALL_PANEL = 15  # proteins: below this the per-cell CLR is compositional enough to make shifts (docstring)
REFERENCE_NOTE = 0.05  # |CLR reference shift| named in the notes: half a shift that counts
CLR, COUNTS, DSB, AUTO = "protein_clr", "protein_counts", "protein_dsb", "auto"
LAYERS = (CLR, COUNTS, DSB)
LAYER_CHOICES = (AUTO, *LAYERS)
LAYER_LABEL = {CLR: "CLR per cell", COUNTS: "log counts", DSB: "DSB"}
LAYER_UNIT = {CLR: "CLR", COUNTS: "log1p counts", DSB: "DSB"}
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
          layer: str = CLR, reference: np.ndarray | None = None) -> dict[str, Any]:
    """The protein check (see the module docstring). ``values``: cells x proteins (the
    cells Mixscape used, in its order); ``labels`` each cell's target or ``control``;
    ``glob`` its KO / NP / NT class; ``targets`` Mixscape's per-target rows. ``reference``
    (CLR only): each cell's CLR reference, its mean log1p count over the panel."""
    from scview.core.mixscape import KO, NP

    unit_name = LAYER_UNIT.get(layer, layer)

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
    flags = [_flag_text(r, unit_name) for r in sorted((r for r in rows if r["flag"]), key=lambda r: _rank(r))]
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
    notes.append(f"Shift: mean {unit_name} of the target's cells minus the controls'; a shift counts from "
                 f"{MIN_SHIFT}. Flag: a shift that counts with at most {KO_SHARE_NONE:.0%} of the target's cells "
                 "called KO.")
    ref_shift = _reference_shifts(reference, labels, control, [t["target"] for t in targets]) \
        if reference is not None and layer == CLR else None
    if layer == CLR and len(proteins) < SMALL_PANEL:
        notes.append(f"CLR per cell over {len(proteins)} proteins: each cell is divided by its own panel's mean, so "
                     "a fall in one protein lifts the others' values (compositional); read shifts of the other "
                     "proteins with care. Log counts are not compositional.")
    if ref_shift:
        moved = sorted(((t, v) for t, v in ref_shift.items() if abs(v) >= REFERENCE_NOTE), key=lambda x: x[1])
        if moved:
            notes.append("The CLR reference (each cell's mean log count over the panel) moved in "
                         + ", ".join(f"{t} {v:+.2f}" for t, v in moved[:8])
                         + (f" and {len(moved) - 8} more" if len(moved) > 8 else "")
                         + ": a fall of the reference lifts every protein's CLR in that target, a rise lowers it.")
    if layer == COUNTS:
        notes.append("Log counts are not normalised per cell: not compositional, but a target that changes cell "
                     "size or antibody capture moves every protein the same way.")
    if n_pairs > MAX_PAIRS:
        notes.append(f"{MAX_PAIRS:,} of {n_pairs:,} target x protein rows kept (the strongest).")
    return {"layer": layer, "layer_label": LAYER_LABEL.get(layer, layer), "unit_name": unit_name,
            "reference_shift": ref_shift, "proteins": list(proteins), "n_proteins": len(proteins), "control": control,
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


def _reference_shifts(reference: np.ndarray, labels: np.ndarray, control: str,
                      targets: list[str]) -> dict[str, float] | None:
    """Per target: the mean CLR reference (log1p counts averaged over the panel) of its
    cells minus the controls'. A CLR shift of protein j is about its log-count shift minus
    this (damped by log1p at low counts), so a target that lowers one abundant protein
    "raises" the rest (Papalexi: JAK2 -0.36, CD86 +0.13 on CLR, -0.11 on log counts)."""
    ref = np.asarray(reference, dtype=np.float64)
    is_ctrl = labels == control
    if is_ctrl.sum() < MIN_CELLS:
        return None
    base = float(ref[is_ctrl].mean())
    out = {}
    for t in targets:
        m = labels == t
        if m.sum() >= MIN_CELLS:
            out[t] = round(float(ref[m].mean()) - base, 4)
    return out


def _flag_text(r: dict, unit_name: str = LAYER_UNIT[CLR]) -> dict:
    a = r["all"] or {}
    if r["unit"] == "replicate":
        shifts = ", ".join(f"{y['diff']:+.2f}" for y in r["replicates"])
        how = (f"the same way in {r['same_sign']} of {r['n_replicates']} replicates ({shifts}), cell-level BH p "
               f"{a.get('padj', float('nan')):.2g}")
    else:
        how = f"Wilcoxon on {a.get('n', 0):,} cells, BH p {a.get('padj', float('nan')):.2g}"
    text = (f"{r['target']}: {r['protein']} {a.get('diff', 0):+.2f} ({unit_name}) in all {a.get('n', 0):,} guide cells, "
            f"{how}; {r['ko_fraction']:.0%} of its cells called KO on RNA.")
    return {"target": r["target"], "protein": r["protein"], "diff": a.get("diff"), "unit": r["unit"],
            "ko_fraction": r["ko_fraction"], "text": text}


class ProteinLayerError(ValueError):
    """A layer the check cannot use here; ``status`` is the HTTP code."""

    def __init__(self, message: str, status: int = 422) -> None:
        super().__init__(message)
        self.status = status


def has_values(adata) -> bool:
    from scview.core import protein

    return protein.has_protein(adata)


def _frame(adata, key: str) -> pd.DataFrame | None:
    m = adata.obsm.get(key) if hasattr(adata, "obsm") else None
    return m if isinstance(m, pd.DataFrame) and m.shape[1] > 0 else None


def _dsb_compositional(adata, n_proteins: int) -> str | None:
    from scview.core import dsb

    rec = dsb.read_record(adata) or {}
    if rec.get("denoise") and n_proteins < SMALL_PANEL:
        return (f"DSB step II removes a per-cell technical part fitted on the cell's own {n_proteins} proteins: "
                "on a small panel that is compositional too (a fall in one protein moves the others).")
    return None


def layers(adata) -> list[dict[str, Any]]:
    """Each layer the check can use on this dataset, with why one cannot be used."""
    n = len(_frame(adata, CLR).columns) if _frame(adata, CLR) is not None else 0
    out = [{"layer": CLR, "label": LAYER_LABEL[CLR], "available": _frame(adata, CLR) is not None,
            "reason": None if _frame(adata, CLR) is not None else "No protein modality.",
            "compositional": n < SMALL_PANEL}]
    has_counts = _frame(adata, COUNTS) is not None
    out.append({"layer": COUNTS, "label": LAYER_LABEL[COUNTS], "available": has_counts,
                "reason": None if has_counts else "The dataset keeps no antibody counts (obsm 'protein_counts'), "
                                                  "only normalised values.",
                "compositional": False})
    has_dsb = _frame(adata, DSB) is not None
    if has_dsb:
        why = _dsb_compositional(adata, n)
        out.append({"layer": DSB, "label": LAYER_LABEL[DSB], "available": True, "reason": why,
                    "compositional": why is not None})
    else:
        out.append({"layer": DSB, "label": LAYER_LABEL[DSB], "available": False, "compositional": False,
                    "reason": "No DSB values: compute them under Protein > DSB. DSB measures each antibody's "
                              "background in empty droplets, so it needs the raw (unfiltered) matrix; without it, "
                              "dsb's no-empty-droplet mode is log counts minus a constant per protein (the same "
                              "test as log counts) plus a per-cell step that is compositional on small panels."
                              if has_counts else "No DSB values, and no antibody counts to compute them from."})
    return out


def default_layer(adata) -> str:
    """``auto``: log counts below ``SMALL_PANEL`` proteins (when kept), else the CLR."""
    clr = _frame(adata, CLR)
    if clr is not None and clr.shape[1] < SMALL_PANEL and _frame(adata, COUNTS) is not None:
        return COUNTS
    return CLR


def resolve_layer(adata, requested: str | None) -> str:
    """The layer to use: ``auto`` (or None) by panel size; a named one must be there."""
    req = (requested or AUTO).strip()
    if req not in LAYER_CHOICES:
        raise ProteinLayerError(f"Unknown protein layer '{req}': use one of {', '.join(LAYER_CHOICES)}.")
    if req == AUTO:
        return default_layer(adata)
    for lay in layers(adata):
        if lay["layer"] == req and not lay["available"]:
            raise ProteinLayerError(lay["reason"])
    return req


def _values(adata, layer: str, rows: np.ndarray) -> tuple[np.ndarray, list[str], np.ndarray | None]:
    """(cells x proteins values, protein names, the CLR reference or None) for ``rows``."""
    m = _frame(adata, layer)
    names = [str(c) for c in m.columns]
    vals = np.asarray(m.to_numpy(dtype=np.float64))[rows]
    if layer == COUNTS:
        vals = np.log1p(np.clip(vals, 0, None))
    ref = None
    counts = _frame(adata, COUNTS)
    if layer == CLR and counts is not None and [str(c) for c in counts.columns] == names:
        ref = np.log1p(np.clip(np.asarray(counts.to_numpy(dtype=np.float64))[rows], 0, None)).mean(axis=1)
    return vals, names, ref


def _with_layer_info(out: dict[str, Any], adata, requested: str | None, layer: str) -> dict[str, Any]:
    out["layer_choice"] = (requested or AUTO)
    out["default_layer"] = default_layer(adata)
    out["layers"] = layers(adata)
    n = out["n_proteins"]
    if (requested or AUTO) == AUTO:
        why = (f"{n} proteins, fewer than {SMALL_PANEL}: log counts by default, as the per-cell CLR is "
               "compositional on a small panel." if layer == COUNTS else
               f"{n} proteins: CLR per cell by default." if n >= SMALL_PANEL else
               f"{n} proteins: CLR per cell, as the dataset keeps no antibody counts.")
        out["notes"].insert(0, why)
    if layer == DSB:
        why = _dsb_compositional(adata, n)
        if why:
            out["notes"].append(why)
    return out


def run(adata, rows: np.ndarray, labels: np.ndarray, glob: np.ndarray, control: str, targets: list[dict],
        replicates: np.ndarray | None, replicate_column: str | None, layer: str | None = AUTO) -> dict[str, Any] | None:
    """The check on ``layer`` (``auto``: by panel size) for the cells Mixscape used
    (``rows``), or None without a protein modality."""
    if not has_values(adata):
        return None
    use = resolve_layer(adata, layer)
    vals, names, ref = _values(adata, use, rows)
    out = check(vals, names, labels, glob, control, targets, replicates, replicate_column, layer=use,
                reference=ref)
    return _with_layer_info(out, adata, layer, use)


def recheck(adata, result: dict[str, Any], layer: str | None) -> dict[str, Any]:
    """The check of a saved Mixscape run (its record: ``labels``, ``control``, ``targets``,
    ``replicate_column``) again on another layer, from the class column it wrote. Changes nothing."""
    from scview.core import mixscape

    if not has_values(adata):
        raise ProteinLayerError("This dataset has no protein modality.", 404)
    obs = adata.obs
    lab_col, rep_col = result.get("labels"), result.get("replicate_column")
    if mixscape.GLOBAL not in obs.columns or lab_col not in obs.columns:
        raise ProteinLayerError("No Mixscape classes on these cells: run Mixscape again.", 404)
    if rep_col and rep_col not in obs.columns:
        rep_col = None
    g = obs[mixscape.GLOBAL]
    rows = np.flatnonzero(g.notna().to_numpy())
    glob = g.astype(object).to_numpy()[rows].astype(str)
    labels = obs[lab_col].astype(object).to_numpy()[rows].astype(str)
    reps = mixscape._text(obs[rep_col], rows) if rep_col else None
    return run(adata, rows, labels, glob, str(result.get("control")), list(result.get("targets") or []), reps,
               rep_col, layer)


def with_current_layers(result: dict[str, Any] | None, adata) -> dict[str, Any] | None:
    """A Mixscape result dict whose protein check lists the layers the dataset has NOW
    (DSB computed after the run is then offered); the check itself is unchanged."""
    if not result or not isinstance(result.get("protein"), dict) or not has_values(adata):
        return result
    result["protein"] = {**result["protein"], "layers": layers(adata), "default_layer": default_layer(adata)}
    return result


def layer_of(check_result: dict | None) -> str:
    """The layer a stored check used: results before v0.80 without one were CLR."""
    return str((check_result or {}).get("layer") or CLR)
