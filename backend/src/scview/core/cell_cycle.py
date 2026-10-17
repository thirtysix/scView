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

The per-cluster ``cycling`` flag (``core/cluster_qc.py``) counts a cell as cycling
only when its S or G2M score clears the same margin, so calls that rest on noise
never make a cluster look proliferative. Measured on scView's own processing,
2026-10-17: Kang's 13 cell types have 0.2-12.6% such cells (T activated the
most) although 28-75% are called S or G2M; paul15's typical cluster 61%, its
erythroid progenitors 1Ery-4Ery 86-98% (median G2M score 0.13-0.45), the
megakaryocyte-erythroid progenitors 17%.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Both scores within this distance of 0: the phase call is noise.
NEAR_ZERO_MARGIN = 0.1
# Caution when at least this share of cells is near zero (paul15, which really
# divides, has 5.6%; the PBMC sets 85-95%).
CAUTION_SHARE = 0.10
# The per-cluster flag: most of a cluster's cells cycling, and clearly more than
# in the typical cluster, so a dataset where every cluster divides (progenitors)
# flags only the ones that stand out. A PBMC cluster would need four times
# Kang's highest share.
CYCLING_SHARE = 0.5
CYCLING_MARGIN = 0.25
# Score columns as scanpy and Seurat (CellCycleScoring) name them.
_S_NAMES = ("S_score", "S.Score")
_G2M_NAMES = ("G2M_score", "G2M.Score")

FLAG_HELP = (
    "Mostly cycling cells: most of its cells have clear S or G2M cell-cycle scores, far "
    "more than in the typical cluster. It may be a proliferating state of another cell type "
    "rather than a cell type of its own; if the cell cycle is not the biology of interest, "
    "regress out the S and G2M scores before clustering."
)


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


def cell_cycle_scores(obs: Any) -> tuple[np.ndarray, np.ndarray] | None:
    """Per-cell S and G2M scores from obs (scanpy or Seurat names), or None."""
    s = next((c for c in _S_NAMES if c in obs.columns), None)
    g = next((c for c in _G2M_NAMES if c in obs.columns), None)
    if s is None or g is None:
        return None
    return (np.asarray(obs[s], dtype=float), np.asarray(obs[g], dtype=float))


def cycling_cells(obs: Any) -> np.ndarray | None:
    """Per cell: True when its S or G2M score is at least NEAR_ZERO_MARGIN, i.e.
    clearly cycling, not a phase call resting on noise. None without scores (a
    phase column alone cannot be checked against the noise, so it is not used)."""
    scores = cell_cycle_scores(obs)
    if scores is None:
        return None
    s, g = scores
    with np.errstate(invalid="ignore"):
        return np.fmax(s, g) >= NEAR_ZERO_MARGIN
