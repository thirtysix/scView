"""Cell QC thresholds: which cells the filtering step removes, before it runs.

The QC histograms on the Data Assessment tab carry the filtering step's cell
thresholds (min / max genes, min counts, max % mito), and ``preview`` says what
they remove overall, per sample and per cluster, with a warning when one empties
a sample or a cluster or takes more than half of one: a real cell type with few
genes or much mitochondrial RNA (platelets, neutrophils, cardiomyocytes) is lost
that way without anyone noticing.

The filtering step judges cells with the same masks (``removed_by``) on the same
per-cell values (``per_cell_qc``: obs QC columns when present, else computed
from the counts, as the QC plots show them), so the preview is what the step
does. A cell without a value for a metric is not removed by it. % mito is used
only when it can be measured (the dataset has MT- genes).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scview.core import cluster_qc, qc_groups

HALF = 0.5  # warn when a threshold removes more than this share of a group
MAX_CLUSTERS = cluster_qc.MAX_GROUPS

# criterion -> what the user calls it (the threshold's label on the plots)
CRITERIA = {
    "min_genes": "Min genes",
    "max_genes": "Max genes",
    "min_counts": "Min counts",
    "max_pct_mt": "Max % mito",
    "doublets": "Predicted doublets",
}


@dataclass(frozen=True)
class Thresholds:
    """Cell thresholds; 0 (or None for % mito) turns one off."""

    min_genes: float = 0.0  # keep cells with at least this many genes
    max_genes: float = 0.0  # keep cells with at most this many genes
    min_counts: float = 0.0  # keep cells with at least this many counts
    max_pct_mt: float | None = None  # keep cells with % mito below this
    drop_doublets: bool = False  # remove cells called doublets (predicted_doublet)

    @classmethod
    def from_params(cls, params) -> Thresholds:
        """From the pipeline's PipelineParams (or anything with the same fields)."""
        mt = getattr(params, "max_pct_mt", None)
        return cls(
            min_genes=float(getattr(params, "min_genes", 0) or 0),
            max_genes=float(getattr(params, "max_genes", 0) or 0),
            min_counts=float(getattr(params, "min_counts", 0) or 0),
            max_pct_mt=None if mt is None else float(mt),
            drop_doublets=bool(getattr(params, "drop_doublets", False)),
        )


def _values(cells: dict, key: str) -> np.ndarray:
    return np.asarray(cells[key], dtype=np.float64)


def removed_by(cells: dict, t: Thresholds, doublets: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """For each threshold that is on, the cells it removes on its own (boolean masks).

    ``cells`` is ``per_cell_qc``'s output; ``doublets`` the per-cell doublet calls
    (``cluster_qc.doublet_calls``), used when ``t.drop_doublets``."""
    genes = _values(cells, "n_genes")
    counts = _values(cells, "total_counts")
    out: dict[str, np.ndarray] = {}
    with np.errstate(invalid="ignore"):
        if t.min_genes > 0:
            out["min_genes"] = np.isfinite(genes) & (genes < t.min_genes)
        if t.max_genes > 0:
            out["max_genes"] = np.isfinite(genes) & (genes > t.max_genes)
        if t.min_counts > 0:
            out["min_counts"] = np.isfinite(counts) & (counts < t.min_counts)
        if t.max_pct_mt is not None and cells.get("has_mt", True) and cells.get("pct_mt") is not None:
            pct = _values(cells, "pct_mt")
            # Kept below the threshold (strictly), as the filtering step always has.
            out["max_pct_mt"] = np.isfinite(pct) & (pct >= t.max_pct_mt)
    if t.drop_doublets and doublets is not None:
        out["doublets"] = np.asarray(doublets, dtype=bool)
    return out


def keep_mask(removed: dict[str, np.ndarray], n_cells: int) -> np.ndarray:
    """Cells no threshold removes."""
    drop = np.zeros(n_cells, dtype=bool)
    for mask in removed.values():
        drop |= mask
    return ~drop


def _grouped(adata, column: str | None, reason: str | None, removed: dict[str, np.ndarray],
             drop: np.ndarray, max_groups: int) -> dict | None:
    if not column:
        return None
    if column not in adata.obs.columns:
        raise ValueError(f"Column '{column}' not found.")
    order, codes = qc_groups.group_codes(adata.obs[column])
    if not order:
        return None
    if len(order) > max_groups:
        return {"column": column, "reason": reason, "groups": [],
                "note": f"'{column}' has {len(order)} groups; the loss is shown for at most {max_groups}."}
    labelled = codes >= 0
    k = len(order)
    sizes = np.bincount(codes[labelled], minlength=k)
    lost = np.bincount(codes[labelled & drop], minlength=k)
    alone = {name: np.bincount(codes[labelled & mask], minlength=k) for name, mask in removed.items()}
    groups = []
    for i, name in enumerate(order):
        n, r = int(sizes[i]), int(lost[i])
        groups.append({
            "name": name,
            "n_cells": n,
            "n_removed": r,
            "fraction": round(r / n, 4) if n else 0.0,
            "by_criterion": {c: int(v[i]) for c, v in alone.items()},
        })
    return {"column": column, "reason": reason, "groups": groups}


def _warnings(kind: str, grouped: dict | None) -> list[dict]:
    if not grouped:
        return []
    out = []
    for g in grouped["groups"]:
        n, r = g["n_cells"], g["n_removed"]
        if not n or (r < n and r / n <= HALF):
            continue
        active = {c: v for c, v in g["by_criterion"].items() if v > 0}
        main = max(active, key=active.get) if active else None
        whole = r >= n
        what = (f"all {n:,} cells of {kind} {g['name']}" if whole
                else f"{r / n:.0%} of {kind} {g['name']} ({r:,} of {n:,} cells)")
        by = ("" if main is None
              else f", by {CRITERIA[main]}" if len(active) == 1
              else f", mostly by {CRITERIA[main]}")
        out.append({
            "kind": kind,
            "column": grouped["column"],
            "group": g["name"],
            "n_cells": n,
            "n_removed": r,
            "fraction": g["fraction"],
            "whole": whole,
            "criterion": main,
            "message": f"Removes {what}{by}.",
        })
    return out


def preview(adata, cells: dict, t: Thresholds, sample_column: str | None = None,
            cluster_column: str | None = None) -> dict:
    """What ``t`` removes: overall, per threshold on its own, per sample (the
    donor / sample column, else a batch column) and per cluster (the active
    clustering), plus warnings for groups it empties or more than halves.

    ``cells`` is ``AnnDataAdaptor.qc_per_cell()``. Raises ValueError for a named
    column that does not exist."""
    n = int(adata.n_obs)
    calls = cluster_qc.doublet_calls(adata.obs)
    removed = removed_by(cells, t, calls)
    drop = ~keep_mask(removed, n)
    n_removed = int(drop.sum())

    if sample_column:
        s_col, s_reason = sample_column, "requested"
    else:
        s_col, s_reason = qc_groups.sample_grouping(adata)
    c_col = cluster_column or cluster_qc.default_grouping(adata)
    c_reason = "requested" if cluster_column else ("clustering" if c_col else None)
    samples = _grouped(adata, s_col, s_reason, removed, drop, qc_groups.MAX_GROUPS)
    clusters = None if c_col and c_col == s_col else _grouped(
        adata, c_col, c_reason, removed, drop, MAX_CLUSTERS)

    has_mt = bool(cells.get("has_mt", True))
    warnings = _warnings("sample", samples) + _warnings("cluster", clusters)
    warnings.sort(key=lambda w: (not w["whole"], -w["fraction"]))
    return {
        "n_cells": n,
        "n_removed": n_removed,
        "n_kept": n - n_removed,
        "fraction_removed": round(n_removed / n, 4) if n else 0.0,
        "thresholds": {
            "min_genes": t.min_genes,
            "max_genes": t.max_genes,
            "min_counts": t.min_counts,
            "max_pct_mt": t.max_pct_mt if has_mt else None,
            "drop_doublets": t.drop_doublets,
        },
        "has_mt": has_mt,
        "doublet_calls": calls is not None,  # so "drop doublets" has something to drop
        "by_criterion": {c: {"label": CRITERIA[c], "n_removed": int(m.sum())} for c, m in removed.items()},
        "samples": samples,
        "clusters": clusters,
        "warnings": warnings,
    }
