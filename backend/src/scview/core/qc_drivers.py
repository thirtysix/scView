"""Do technical scores line up with the space the clusters were made in?

For each per-cell technical score (log total counts, % mito, % ribosomal,
% haemoglobin, the S and G2M cell-cycle scores): its Pearson correlation with
each leading principal component of the space the neighbour graph was built in
(PCA, or Harmony-corrected PCA). A score that correlates strongly with a leading
PC is part of what the clustering sees.

The same correlation can mean two things, so it is computed a second time after
taking out each cluster's mean (from the score and from the PC):

- Inside clusters too: the score moves along the PC within clusters, so it is an
  axis of the embedding (sequencing depth, a continuous state) that can split a
  cell type.
- Only between clusters: the clusters differ in it, which cell types often do
  (RNA content, ribosomal share); a concern only if it is what splits one.

Measured 2026-10-17 on scView's own processing, first 10 PCs:

- Kang PBMC (Harmony PCs): log total counts vs PC1 r = 0.69 and ribosomal %
  r = -0.78, but inside its 13 cell types only 0.17 and -0.34 (PC1 separates
  monocytes from lymphocytes, which differ in both); haemoglobin, S and G2M
  |r| <= 0.25.
- paul15 marrow (MARS-seq, PCA): log total counts vs PC2 r = 0.89 and 0.95
  inside clusters: a depth axis. Ribosomal % -0.61 (-0.45 inside), G2M 0.33.

Haemoglobin is correlated over non-erythroid cells only (red cells forming their
own PC is biology, not a driver) and on a log scale (log1p of the %), since most
cells have none and a few a great deal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import BaseModel

LEADING_PCS = 10  # the PCs with most of the variance; later ones carry little
DRIVER_R = 0.5  # |r| >= 0.5: the score accounts for a quarter of that PC's variance
WITHIN_R = 0.5  # and the same inside clusters: an axis, not a cluster difference
MAX_CELLS = 50_000  # a seeded subsample beyond this; r is then within ~0.01
MIN_CELLS = 50
SEED = 0

LABELS = {
    "log_counts": "Log total counts",
    "pct_mt": "Mitochondrial %",
    "pct_ribo": "Ribosomal %",
    "pct_hb": "Haemoglobin %",
    "S_score": "S-phase score",
    "G2M_score": "G2M score",
}


class Driver(BaseModel):
    score: str  # key in LABELS
    label: str
    pc: int  # 1-based
    r: float
    r_within: float | None = None  # inside clusters (cluster means removed)
    pc_share: float  # the PC's share of the variance in the space used
    inside_clusters: bool = False
    reason: str = ""


class DriverCheck(BaseModel):
    representation: str | None = None
    n_pcs: int = 0  # PCs looked at
    checked: list[str] = []  # labels of the scores looked at
    drivers: list[Driver] = []


def _corr(x: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Pearson r of ``x`` with each column of ``Y`` (NaN for a constant one)."""
    xc = x - x.mean()
    Yc = Y - Y.mean(axis=0)
    denom = np.sqrt((xc ** 2).sum() * (Yc ** 2).sum(axis=0))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denom > 0, xc @ Yc / denom, np.nan)


def _centre(values: np.ndarray, codes: np.ndarray, n_groups: int) -> np.ndarray:
    """``values`` (1-D or cells x k) minus each cluster's mean."""
    counts = np.bincount(codes, minlength=n_groups).astype(np.float64)
    counts[counts == 0] = 1.0
    if values.ndim == 1:
        return values - (np.bincount(codes, values, n_groups) / counts)[codes]
    means = np.stack([np.bincount(codes, values[:, j], n_groups) for j in range(values.shape[1])],
                     axis=1) / counts[:, None]
    return values - means[codes]


def _reason(d: Driver) -> str:
    head = (f"{d.label} correlates with PC{d.pc} (r = {d.r:.2f}; PC{d.pc} carries "
            f"{d.pc_share:.0%} of the variance)")
    rw = "n/a" if d.r_within is None else f"{d.r_within:.2f}"
    if d.inside_clusters:
        head += f", and inside clusters too (r = {rw})"
        if d.score in ("S_score", "G2M_score"):
            return (head + ": the cell cycle is an axis of the embedding. If it is not the "
                    "biology of interest, regress out the S and G2M scores (Scaling step, "
                    "Regress out).")
        fix = (" Pearson-residual normalisation models depth and usually removes it."
               if d.score == "log_counts" else "")
        return (head + ": it is an axis of the embedding, not only a difference between cell "
                "types, and can split one cell type into clusters." + fix)
    head += f", but mostly between clusters (r = {rw} inside them)"
    if d.score in ("S_score", "G2M_score"):
        return head + ": proliferating cells form or lead some clusters."
    if d.score == "pct_hb":
        return head + ": some non-erythroid clusters carry more haemoglobin RNA than others."
    return (head + ": the clusters differ in it, as cell types often do (RNA content, "
            "ribosomal share). A concern only if it is what splits one cell type.")


def check(rep: str | None, X, labels: np.ndarray, scores: dict[str, np.ndarray],
          masks: dict[str, np.ndarray] | None = None) -> DriverCheck:
    """Correlate each score with the leading PCs of ``X`` (cells x PCs, the space
    named ``rep``), over all cells and inside clusters (``labels``). ``masks``
    limit a score to some cells (haemoglobin: non-erythroid cells)."""
    if X is None:
        return DriverCheck()
    X = np.asarray(X)
    n = X.shape[0]
    if X.ndim != 2 or n != len(labels) or n < MIN_CELLS:
        return DriverCheck()
    var = X.var(axis=0, dtype=np.float64)
    total = float(var.sum())
    k = min(LEADING_PCS, X.shape[1])
    keep = np.arange(n)
    if n > MAX_CELLS:
        keep = np.sort(np.random.default_rng(SEED).choice(n, size=MAX_CELLS, replace=False))
    lead = np.asarray(X[keep, :k], dtype=np.float64)  # the leading PCs, subsampled
    lab = np.asarray(labels, dtype=object)[keep]
    ok_lab = ~pd.isna(lab)  # unlabelled cells sit in no cluster
    out = DriverCheck(representation=rep, n_pcs=k)
    for key, values in scores.items():
        if values is None or len(values) != n:
            continue
        v = np.asarray(values, dtype=np.float64)[keep]
        use = np.isfinite(v) & ok_lab
        if masks and key in masks:
            use &= np.asarray(masks[key], dtype=bool)[keep]
        if use.sum() < MIN_CELLS or np.ptp(v[use]) == 0:
            continue
        out.checked.append(LABELS[key])
        Y = lead[use]
        r = _corr(v[use], Y)
        if not np.any(np.isfinite(r)):
            continue
        j = int(np.nanargmax(np.abs(r)))
        if not abs(r[j]) >= DRIVER_R:
            continue
        _, codes = np.unique(lab[use].astype(str), return_inverse=True)
        n_groups = int(codes.max()) + 1
        r_w = float(_corr(_centre(v[use], codes, n_groups), _centre(Y[:, [j]], codes, n_groups))[0])
        d = Driver(
            score=key, label=LABELS[key], pc=j + 1, r=round(float(r[j]), 2),
            r_within=round(r_w, 2) if np.isfinite(r_w) else None,
            pc_share=round(float(var[j]) / total, 3) if total > 0 else 0.0,
            inside_clusters=bool(np.isfinite(r_w) and abs(r_w) >= WITHIN_R),
        )
        d.reason = _reason(d)
        out.drivers.append(d)
    out.drivers.sort(key=lambda d: -abs(d.r))
    return out


def for_adata(adata, labels: np.ndarray, scores: dict[str, np.ndarray],
              masks: dict[str, np.ndarray] | None = None) -> DriverCheck:
    """``check`` in the space the dataset's neighbour graph was built in."""
    from scview.core.resolution_sweep import _representation  # PCA or Harmony, n_pcs used

    rep, X = _representation(adata)
    return check(rep, X, labels, scores, masks)
