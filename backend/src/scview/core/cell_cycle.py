"""How far to trust cell-cycle phase calls.

scanpy's ``score_genes_cell_cycle`` (like Seurat's CellCycleScoring) calls a cell
G1 only when both its S and G2M scores are below 0, else the higher score's
phase. In cells that are not dividing both scores are 0 plus noise, so about
half of them get called S or G2M. The phase column is left as scanpy wrote it;
this counts the cells whose call rests on that noise, so the assessment can say
so.

The margin, measured on scView's own steps (counts -> normalise to 1e4 -> log1p
-> score), 2026-10-16:

- Kang 2018 IFN-b PBMC (13,836 cells, 628 genes per cell): G1 6,009 / S 4,161 /
  G2M 3,666, yet MKI67 is detected in 15 cells. The negative side of each score
  (cells below 0 are not cycling) has an RMS spread of 0.044, so +-0.1 is about
  two noise SDs: 95% of cells have both scores inside it, and MKI67 is detected
  in 0.0% of them (1.7% of the 663 S/G2M calls outside).
- 10x pbmc_1k_v3 (1,222 cells, 1,919 genes per cell): G1 684 / S 493 / G2M 45;
  85% inside +-0.1 (MKI67 detected in 1.0% of them, 5.2% of the calls outside).
- paul15 mouse marrow progenitors (2,730 cells, really dividing): scores spread
  to +-0.5 and beyond; 5.6% of cells inside +-0.1.

A margin scaled to each dataset's own negative-side spread was tried and
rejected: in a dividing population negative scores are biology (S-phase cells'
G2M score), and it put 80% of paul15 inside.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Both scores within this distance of 0: the phase call is noise.
NEAR_ZERO_MARGIN = 0.1
# Caution when at least this share of cells is near zero (paul15, which really
# divides, has 5.6%; the PBMC sets 85-95%).
CAUTION_SHARE = 0.10


def near_zero_phase_calls(obs: Any) -> dict[str, Any] | None:
    """Counts of cells whose S and G2M scores are both within NEAR_ZERO_MARGIN of
    0, from obs ``S_score`` / ``G2M_score`` (and ``phase`` when present); None
    when the scores are missing."""
    if "S_score" not in obs.columns or "G2M_score" not in obs.columns:
        return None
    s = np.asarray(obs["S_score"], dtype=float)
    g = np.asarray(obs["G2M_score"], dtype=float)
    ok = np.isfinite(s) & np.isfinite(g)
    near = ok & (np.abs(s) < NEAR_ZERO_MARGIN) & (np.abs(g) < NEAR_ZERO_MARGIN)
    out: dict[str, Any] = {
        "n_cells": int(ok.sum()),
        "n_near_zero": int(near.sum()),
        "near_zero_margin": NEAR_ZERO_MARGIN,
    }
    if "phase" in obs.columns:
        cycling = ok & (np.asarray(obs["phase"]).astype(str) != "G1")
        out["n_called_cycling"] = int(cycling.sum())
        out["n_called_cycling_near_zero"] = int((cycling & near).sum())
    return out


def phase_caution(counts: dict[str, Any] | None) -> str | None:
    """The one-line caution for the assessment, or None when few cells are near 0."""
    if not counts or counts["n_cells"] == 0:
        return None
    n, near = counts["n_cells"], counts["n_near_zero"]
    if near / n < CAUTION_SHARE:
        return None
    called = ""
    if counts.get("n_called_cycling_near_zero"):
        called = f", {counts['n_called_cycling_near_zero']:,} of them called S or G2M"
    return (
        f"{near:,} of {n:,} cells ({near / n:.0%}) have both S and G2M scores within "
        f"±{counts['near_zero_margin']:g} of 0{called}: phase calls for these cells are "
        "not reliable."
    )
