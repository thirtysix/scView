"""Cell-type-aware % mito bounds: does one global ceiling fit every cell type?

Cell types differ in their baseline mitochondrial share: lymphocytes and
fibroblasts sit at a few %, hepatocytes, cardiomyocytes and some cancer lines far
higher. A fixed ceiling that suits one type can remove most of another, healthy
cells included (a lab case: a 10 % ceiling removed 65 % of LNCaP cells, median
~12 %, and 1-5 % of the fibroblasts beside them). Osorio & Cai 2021 (Bioinformatics,
doi:10.1093/bioinformatics/btaa751) found the usual 5 % fails in 13 of 44 human
tissues and that human data run higher than mouse; MitoChontrol (bioRxiv 2026,
PMID 41993278) sets the threshold within each cluster instead of once.

For each group of a label column (default: the author's cell types, else the
clustering) this gives the % mito distribution (n, median, MAD, quantiles) and a
within-type bound, median + ``MAD_K`` MADs (the raw MAD, as the filtering step's
MAD mode uses; MitoChontrol's per-cluster idea with the MAD rule in place of its
mixture model). It counts the cells the GLOBAL ceiling (the dataset's recorded
``max_pct_mt``, else the filtering step's default) would remove from each type
against those the type's own bound would, and flags a type when the ceiling
removes more than ``LOSS_FLAG`` of it or sits below its median.

Advisory only: the filtering step (``core/qc_filter.py``) is unchanged. A type
made mostly of damaged cells has a high median and would get a high bound too, so
a per-type bound assumes each type's typical cell is healthy: a type whose median
is ``DAMAGED_MEDIAN`` % or more (pbmc_1k_v3: a 57-cell cluster at 96.5 %) is
called damaged instead, with no bound of its own and no flag.
"""

from __future__ import annotations

import re
from collections.abc import Callable

import numpy as np
import pandas as pd
from pydantic import BaseModel

from scview.core import cluster_qc, qc_filter, qc_groups

MAD_K = 3.0  # within-type bound: median + this many (raw) MADs of % mito
LOSS_FLAG = 0.30  # flag a type when the global ceiling removes more than this share
MIN_TYPE_CELLS = 30  # fewer: no bound of its own (the MAD's relative SE ~ 1.7 / sqrt(n))
DAMAGED_MEDIAN = 50.0  # median % mito at or above this: damaged cells, not a high baseline
MAX_GROUPS = cluster_qc.MAX_GROUPS
DEFAULT_CEILING = 20.0  # PipelineParams.max_pct_mt (core/pipeline.py)
QUANTILES = (0.05, 0.25, 0.75, 0.95)

# Columns that hold an author's cell types: a cell-type-like name, or a cluster
# column whose labels are mostly names ("CD14 Mono"), not numbers.
_TYPE_RE = re.compile(r"cell_?type|celltype|annotation|^labels?$|cell_?ontology", re.I)
_CLUSTER_RE = re.compile(r"cluster", re.I)

NOT_MEASURABLE = ("No mitochondrial (MT-) genes in this dataset, so % mito can't be measured "
                  "and no per-type mito bound can be set.")

RULE = (
    f"Each type's own bound is its median % mito + {MAD_K:g} MADs (raw MAD; types with "
    f"fewer than {MIN_TYPE_CELLS} cells, or no spread, get none). A type is flagged when the "
    f"global ceiling removes more than {LOSS_FLAG:.0%} of it or sits below its median; one with "
    f"a median of {DAMAGED_MEDIAN:g}% or more is called damaged instead."
)
CITATIONS = (
    "Osorio & Cai 2021, Bioinformatics, doi:10.1093/bioinformatics/btaa751 (mito % varies by "
    "tissue and species; a fixed 5 % fails in 13 of 44 human tissues); MitoChontrol, bioRxiv "
    "2026, PMID 41993278 (thresholds set within each cluster)."
)


class TypeMito(BaseModel):
    group: str
    n_cells: int
    median: float | None = None  # % mito
    mad: float | None = None
    quantiles: dict[str, float] = {}  # "q05", "q25", "q75", "q95"
    bound: float | None = None  # median + MAD_K MADs; None: too few cells or no spread
    removed_global: int = 0  # cells above the global ceiling
    removed_global_frac: float = 0.0
    removed_bound: int | None = None  # cells above the type's own bound
    removed_bound_frac: float | None = None
    flagged: bool = False
    damaged: bool = False  # median >= DAMAGED_MEDIAN: likely damaged cells, no bound of its own
    reasons: list[str] = []
    note: str = ""


class MitoByType(BaseModel):
    column: str | None
    reason: str | None = None  # "requested" | "cell type" | "clustering"
    measurable: bool = True
    n_cells: int
    n_labelled: int = 0
    ceiling: float | None = None
    ceiling_source: str | None = None  # "requested" | "recorded" | "default"
    filter_mode: str | None = None  # the recorded filtering mode ("mad" | "fixed"), if any
    already_filtered: bool = False
    includes_removed: int = 0  # cells the filtering step removed, added back from its record
    mad_k: float = MAD_K
    loss_flag: float = LOSS_FLAG
    min_cells: int = MIN_TYPE_CELLS
    types: list[TypeMito] = []
    n_flagged: int = 0
    removed_global_total: int = 0
    removed_bound_total: int = 0  # by the per-type bounds, where a type has one
    summary: str = ""
    note: str = ""
    rule: str = RULE
    citations: str = CITATIONS


def default_column(adata) -> tuple[str | None, str | None]:
    """(column, why): an author's cell-type column if obs has one (a ``cell_type``-like
    name, or a cluster column with named labels), else the clustering."""
    obs = adata.obs

    def usable(c) -> bool:
        s = obs[c]
        if str(c).startswith("scview_") or not (
                isinstance(s.dtype, pd.CategoricalDtype) or pd.api.types.is_object_dtype(s)
                or pd.api.types.is_string_dtype(s)):
            return False
        return s.notna().mean() >= 0.5 and 2 <= s.nunique(dropna=True) <= MAX_GROUPS

    def named(c) -> bool:
        labels = obs[c].dropna().astype(str).unique()
        return sum(not s.strip().isdigit() for s in labels) > len(labels) / 2

    for c in obs.columns:
        if _TYPE_RE.search(str(c)) and usable(c):
            return str(c), "cell type"
    for c in obs.columns:
        if _CLUSTER_RE.search(str(c)) and usable(c) and named(c):
            return str(c), "cell type"
    col = cluster_qc.default_grouping(adata)
    return (col, "clustering") if col else (None, None)


def _ceiling(adata, requested: float | None) -> tuple[float, str, dict | None]:
    recorded = qc_filter.recorded_filtering(adata)
    if requested is not None:
        return float(requested), "requested", recorded
    mt = (recorded or {}).get("max_pct_mt")
    try:
        if mt is not None and float(mt) > 0:
            return float(mt), "recorded", recorded
    except (TypeError, ValueError):
        pass
    return DEFAULT_CEILING, "default", recorded


def _pct(v: float) -> str:
    return f"{v:.1f}%"


def mito_by_type(adata, column: str | None = None, cells: dict | Callable[[], dict] | None = None,
                 ceiling: float | None = None) -> MitoByType:
    """Per-type % mito distribution, bound and the global ceiling's loss.

    ``cells`` are the per-cell QC arrays (``AnnDataAdaptor.qc_per_cell``) or a
    function returning them. Raises ``ValueError`` for an unknown column, one
    with too many groups, or a negative ceiling."""
    n = int(adata.n_obs)
    if ceiling is not None and not (ceiling >= 0):
        raise ValueError("The % mito ceiling must be a number >= 0.")
    if column:
        if column not in adata.obs.columns:
            raise ValueError(f"Column '{column}' not found in obs.")
        reason = "requested"
    else:
        column, reason = default_column(adata)
    if not column:
        return MitoByType(column=None, n_cells=n,
                          note="No cell types or clustering yet: run clustering, or pick a label column.")
    order, codes = qc_groups.group_codes(adata.obs[column])
    if len(order) > MAX_GROUPS:
        raise ValueError(f"'{column}' has {len(order)} groups; at most {MAX_GROUPS} are compared.")
    cells = cells() if callable(cells) else (cells if cells is not None else _per_cell(adata))
    labelled = codes >= 0
    base = dict(column=column, reason=reason, n_cells=n, n_labelled=int(labelled.sum()))
    if not qc_filter._mito_measurable(cells):
        return MitoByType(**base, measurable=False, note=NOT_MEASURABLE)

    cut, source, recorded = _ceiling(adata, ceiling)
    mt = np.asarray(cells["pct_mt"], dtype=np.float64)
    # After filtering, the cells the ceiling took are gone from the dataset; the
    # step's record keeps their % mito and labels, so the table shows the cells
    # as they were before it (a type the ceiling emptied comes back as a row).
    gone = qc_filter.recorded_removed_mito(adata, column)
    extra: dict[str, np.ndarray] = {}
    if gone is not None:
        g_labels, g_mt, g_when = gone
        lab = np.asarray(g_labels, dtype=object)
        for name in dict.fromkeys(x for x in g_labels if x):
            extra[name] = g_mt[lab == name]
    rows: list[TypeMito] = []
    for g, idx in zip(order, qc_groups.members(codes, len(order))):
        v = mt[idx]
        rows.append(_type_row(g, np.concatenate([v, extra.pop(g)]) if g in extra else v, cut))
    for g, v in extra.items():  # types with no cell left
        rows.append(_type_row(g, v, cut))

    flagged = [r for r in rows if r.flagged]
    removed_global = int(sum(r.removed_global for r in rows))
    removed_bound = int(sum(r.removed_bound or 0 for r in rows))
    notes = []
    n_back = 0 if gone is None else int(sum(1 for x in gone[0] if x))
    if gone is not None:
        when = gone[2][:10]
        notes.append(f"Includes the {n_back:,} labelled cells the filtering step removed"
                     f"{' on ' + when if when else ''} (from its record): the table shows the cells as "
                     "they were before that filter.")
    elif recorded:
        notes.append(f"This dataset was filtered by scView ({recorded.get('qc_threshold_mode')} mode): "
                     "cells it removed are gone, so the counts are what filtering again would remove.")
        if recorded.get("qc_threshold_mode") == "mad":
            notes.append("In MAD mode each sample's own bound (median + 3 MADs) can sit below the "
                         "ceiling; the filter preview shows that.")
    if int((~labelled).sum()):
        notes.append(f"{int((~labelled).sum()):,} cells have no '{column}' label and are left out.")
    return MitoByType(
        **base, ceiling=cut, ceiling_source=source,
        filter_mode=(recorded or {}).get("qc_threshold_mode"), already_filtered=bool(recorded),
        includes_removed=n_back,
        types=rows, n_flagged=len(flagged), removed_global_total=removed_global,
        removed_bound_total=removed_bound, summary=_summary(rows, flagged, cut),
        note=" ".join(notes),
    )


def _per_cell(adata) -> dict:
    from scview.core.anndata_adaptor import per_cell_qc
    return per_cell_qc(adata)


def _type_row(group: str, v: np.ndarray, cut: float) -> TypeMito:
    v = v[np.isfinite(v)]
    k = int(v.size)
    row = TypeMito(group=group, n_cells=k)
    if not k:
        row.note = "No cells with a % mito value."
        return row
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med)))
    row.median, row.mad = round(med, 3), round(mad, 3)
    row.quantiles = {f"q{round(q * 100):02d}": round(float(np.quantile(v, q)), 3) for q in QUANTILES}
    row.removed_global = int((v > cut).sum())
    row.removed_global_frac = round(row.removed_global / k, 4)
    if med >= DAMAGED_MEDIAN:
        row.damaged = True
        row.note = (f"Median {_pct(med)} mito: most of its reads are mitochondrial, likely damaged "
                    "or dying cells rather than a type with a high baseline; no bound of its own.")
        return row
    if k < MIN_TYPE_CELLS:
        row.note = f"Only {k} cells: too few for a bound of its own."
    elif mad <= 0:
        row.note = "No spread in % mito (MAD 0): no bound of its own."
    else:
        bound = med + MAD_K * mad
        row.bound = round(bound, 3)
        row.removed_bound = int((v > bound).sum())
        row.removed_bound_frac = round(row.removed_bound / k, 4)
    own = (f"; its own bound, {_pct(row.bound)}, would remove {row.removed_bound_frac:.0%}"
           if row.bound is not None else "")
    if k >= MIN_TYPE_CELLS and cut < med:
        row.flagged = True
        row.reasons.append(f"The {_pct(cut)} ceiling sits below its median ({_pct(med)}): it removes "
                           f"{row.removed_global_frac:.0%} of its {k:,} cells{own}.")
    elif k >= MIN_TYPE_CELLS and row.removed_global_frac > LOSS_FLAG:
        row.flagged = True
        row.reasons.append(f"The {_pct(cut)} ceiling removes {row.removed_global_frac:.0%} of its "
                           f"{k:,} cells (median {_pct(med)}){own}.")
    return row


def _summary(rows: list[TypeMito], flagged: list[TypeMito], cut: float) -> str:
    if not rows:
        return ""
    damaged = [r for r in rows if r.damaged]
    tail = "" if not damaged else " " + "; ".join(
        f"{r.group} (median {_pct(r.median or 0.0)})" for r in damaged
    ) + (" looks" if len(damaged) == 1 else " look") + " damaged, not a high-baseline type."
    return _flag_summary([r for r in rows if not r.damaged], flagged, cut) + tail


def _flag_summary(rows: list[TypeMito], flagged: list[TypeMito], cut: float) -> str:
    if not rows:
        return ""
    if flagged:
        worst = max(flagged, key=lambda r: r.removed_global_frac)
        own = (f"; its own bound ({_pct(worst.bound)}) would remove {worst.removed_bound_frac:.0%}"
               if worst.bound is not None else "")
        more = f" {len(flagged) - 1} more type(s) flagged." if len(flagged) > 1 else ""
        return (f"A {_pct(cut)} ceiling removes {worst.removed_global_frac:.0%} of {worst.group} "
                f"(median {_pct(worst.median or 0.0)}){own}.{more}")
    sized = [r for r in rows if r.n_cells >= MIN_TYPE_CELLS] or rows  # a 9-cell group says little
    top = max(sized, key=lambda r: r.removed_global_frac)
    return (f"A {_pct(cut)} ceiling removes at most {top.removed_global_frac:.0%} of any type "
            f"({top.group}, median {_pct(top.median or 0.0)}): no type is flagged.")


def for_adaptor(adaptor, column: str | None = None, ceiling: float | None = None) -> MitoByType:
    """``mito_by_type`` on a loaded dataset, reusing its cached per-cell QC arrays."""
    fn = getattr(adaptor, "qc_per_cell", None)
    return mito_by_type(adaptor.adata, column, fn if callable(fn) else None, ceiling)


TAG = "result:mito_by_type"


def grounding_lines(adaptor) -> tuple[list[str], str]:
    """One co-pilot fact line: the per-type mito summary (or "not measurable"),
    with ``(lines, detail)`` as the other grounding helpers return them."""
    r = for_adaptor(adaptor)
    if not r.column:
        return [], ""
    if not r.measurable:
        line = (f"- % mito by cell type ('{r.column}'): not measurable, the dataset has no "
                f"mitochondrial (MT-) genes [{TAG}]")
        return [line], f"{r.column}: % mito not measurable"
    flagged = ", ".join(f"{t.group} ({t.removed_global_frac:.0%} removed, median {_pct(t.median or 0.0)})"
                        for t in r.types if t.flagged) or "none"
    line = (f"- % mito by cell type ('{r.column}', {r.ceiling_source} ceiling {_pct(r.ceiling or 0.0)}; "
            f"advisory, filtering unchanged): {r.summary} Flagged types: {flagged}. "
            f"Rule: {r.rule} [{TAG}]")
    return [line], f"{r.column}: {r.n_flagged} of {len(r.types)} types flagged at {_pct(r.ceiling or 0.0)}"
