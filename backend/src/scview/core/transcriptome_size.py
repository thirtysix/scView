"""Transcriptome-size caution for DE between groups that differ in RNA content.

Cell types hold different amounts of RNA (a monocyte more than a naive T cell),
and library-size normalisation scales every cell to the same total, which hides
it (Lu et al. 2025, PMID 39893178). A gene with the same number of transcripts
per cell on both sides then gets a log2 fold change of -log2(r) when one side's
cells hold r times more RNA, so fold changes of broadly expressed genes shift
together and the bigger cells' genes look "down".

The caution fires at a 1.5-fold difference in median total counts: the shift is
then log2 1.5 = 0.58, over half the |log2FC| cut-off of 1 the DE table starts
from (at 2-fold it is the whole cut-off). On Kang's PBMCs, lymphocytes vs CD4
Naive T stay at or under 1.28 (T subsets 1.02-1.12, B 1.06, NK 1.24, B
activated 1.28), as do STIM vs CTRL within CD14 Mono (1.12), while myeloid cells
reach 1.79 (pDC, CD14 Mono) to 2.76 (DC): a 2-fold threshold would stay silent
for CD14 Mono vs CD4 Naive T, the textbook case.

Total counts are raw counts per cell: the obs column a QC step or another tool
wrote (scanpy, older scanpy, Seurat), else summed from the counts matrix.
Never normalised X, whose totals are equal by construction, nor ``.raw``, which
holds log-normalised values in this app.
"""

from __future__ import annotations

import numpy as np

CAUTION_RATIO = 1.5
TOTAL_COUNT_COLUMNS = ("total_counts", "n_counts", "nCount_RNA")
VIOLIN_MAX = 2000  # cells per side sent for the violin plot


def per_cell_totals(adata) -> tuple[np.ndarray, str] | None:
    """Every cell's total raw count and where it came from, or None when the
    dataset holds no counts. ``adata`` is in memory (the compared cells)."""
    for col in TOTAL_COUNT_COLUMNS:
        if col in adata.obs.columns:
            try:
                v = np.asarray(adata.obs[col].to_numpy(), dtype=np.float64)
            except (TypeError, ValueError):
                continue
            if np.all(np.isfinite(v)) and np.all(v >= 0):
                return v, f"obs['{col}']"
    from scview.core.normalization import counts_all_genes

    try:
        M, _, source = counts_all_genes(adata)
    except ValueError:
        return None
    return np.asarray(M.sum(axis=1), dtype=np.float64).ravel(), source


def _sample(v: np.ndarray, rng: np.random.Generator) -> list[float]:
    if v.size > VIOLIN_MAX:
        v = rng.choice(v, VIOLIN_MAX, replace=False)
    return [round(float(x), 1) for x in v]


def compare(totals: np.ndarray, source: str, mask_a: np.ndarray, mask_b: np.ndarray) -> dict:
    """Median total counts of the two sides, their ratio (larger over smaller),
    whether it calls for the caution, and up to ``VIOLIN_MAX`` cells' totals per
    side (a seeded sample, so a rerun draws the same violin)."""
    a, b = totals[mask_a], totals[mask_b]
    med_a, med_b = float(np.median(a)), float(np.median(b))
    lo, hi = sorted((med_a, med_b))
    ratio = hi / lo if lo > 0 else float("inf")
    rng = np.random.default_rng(0)
    return {
        "source": source,
        "median_a": med_a,
        "median_b": med_b,
        "ratio": round(ratio, 3) if np.isfinite(ratio) else None,
        "caution": bool(ratio >= CAUTION_RATIO),
        "threshold": CAUTION_RATIO,
        "values_a": _sample(a, rng),
        "values_b": _sample(b, rng),
    }


def check(adata, mask_a: np.ndarray, mask_b: np.ndarray) -> dict | None:
    """``compare`` for the two sides of a DE comparison (``adata`` in memory,
    before anything normalises its X), or None without counts."""
    found = per_cell_totals(adata)
    if found is None:
        return None
    return compare(found[0], found[1], mask_a, mask_b)
