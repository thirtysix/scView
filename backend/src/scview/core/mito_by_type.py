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

The filtering step applies these bounds when asked (``qc_mito_mode``
"per_type", ``core/qc_filter.py``: the same rule, with the cap at
``qc_filter.TYPE_BOUND_CAP``); the table here previews it. A dataset filtered that
way is read with the rule it used: a type that had its own bound is not flagged
against a ceiling it was never judged by. A type made mostly of damaged cells has
a high median and would get a high bound too, so a per-type bound assumes each
type's typical cell is healthy: a type whose median is ``DAMAGED_MEDIAN`` % or
more (pbmc_1k_v3: a 57-cell cluster at 96.5 %) is called damaged instead, with no
bound of its own and no flag.

After filtering, the cells it removed are gone; the step's record keeps their %
mito and the labels they had then. A clustering made after filtering has no label
for them, so ``assigned`` can give each one the cluster its expression is closest
to (``core/mito_refilter.py``), and the table shows the types as they were.
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
MIN_TYPE_CELLS = qc_filter.TYPE_MIN_CELLS  # fewer: no bound of its own (MAD's rel. SE ~ 1.7/sqrt(n))
DAMAGED_MEDIAN = qc_filter.TYPE_DAMAGED_MEDIAN  # median % mito at or above: damaged, not a baseline
BOUND_CAP = qc_filter.TYPE_BOUND_CAP  # no type's own bound above this, as the filtering step
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
    f"Each type's own bound is its median % mito + {MAD_K:g} MADs (raw MAD, at most "
    f"{BOUND_CAP:g}%; types with fewer than {MIN_TYPE_CELLS} cells, or no spread, get none). "
    "A type is flagged when the "
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
    # Cells the filtering step's % mito rule actually removed (from its record; None
    # when the dataset was not filtered or has no record). In MAD mode each sample's
    # bound can sit below the ceiling, so this can be far above removed_global.
    removed_filter: int | None = None
    removed_filter_frac: float | None = None
    flagged: bool = False
    damaged: bool = False  # median >= DAMAGED_MEDIAN: likely damaged cells, no bound of its own
    # How the dataset's per-type filtering judged it ("own" | "global"), and its bound
    # then; None when it was not filtered per type on this column.
    applied_rule: str | None = None
    applied_bound: float | None = None
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
    # The recorded % mito rule ("global" | "per_type") and, per type, its column.
    filter_mito_mode: str | None = None
    filter_type_column: str | None = None
    already_filtered: bool = False
    includes_removed: int = 0  # cells the filtering step removed, added back from its record
    assigned_removed: int = 0  # of those, cells given a cluster by expression (``assigned``)
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
    name, or a cluster column with named labels), else the clustering. (The card
    itself first takes the column a per-type filtering used: ``_column``.)"""
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


def _column(adata, recorded: dict | None) -> tuple[str | None, str | None]:
    """The column a per-type filtering used (``reason`` "filtering"), else ``default_column``."""
    col = (recorded or {}).get("qc_mito_type_column")
    if (recorded or {}).get("qc_mito_mode") == "per_type" and col and col in adata.obs.columns:
        return str(col), "filtering"
    return default_column(adata)


def _applied(recorded: dict | None, column: str) -> dict[str, float | None] | None:
    """Per type, the bound the last per-type filtering judged it by (None: the
    global rule), when it filtered per type on ``column``; else None."""
    if (recorded or {}).get("qc_mito_mode") != "per_type":
        return None
    rule = recorded.get("qc_mito_types")
    if not isinstance(rule, dict) or rule.get("column") != column:
        return None
    out: dict[str, float | None] = {}
    for r in rule.get("types") or []:
        if isinstance(r, dict) and isinstance(r.get("name"), str):
            b = r.get("bound")
            out[r["name"]] = float(b) if isinstance(b, (int, float)) else None
    return out


def mito_by_type(adata, column: str | None = None, cells: dict | Callable[[], dict] | None = None,
                 ceiling: float | None = None,
                 assigned: tuple[list[str], np.ndarray] | Callable[[str], tuple | None] | None = None
                 ) -> MitoByType:
    """Per-type % mito distribution, bound and the global ceiling's loss.

    ``cells`` are the per-cell QC arrays (``AnnDataAdaptor.qc_per_cell``) or a
    function returning them. ``assigned``: the removed cells' labels in ``column``
    and their % mito, for a column the removal record has no labels for (a
    clustering made after filtering; ``mito_refilter.assigned_removed``). Raises
    ``ValueError`` for an unknown column, one with too many groups, or a negative
    ceiling."""
    n = int(adata.n_obs)
    if ceiling is not None and not (ceiling >= 0):
        raise ValueError("The % mito ceiling must be a number >= 0.")
    recorded = qc_filter.recorded_filtering(adata)
    if column:
        if column not in adata.obs.columns:
            raise ValueError(f"Column '{column}' not found in obs.")
        reason = "requested"
    else:
        column, reason = _column(adata, recorded)
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

    cut, source, _ = _ceiling(adata, ceiling)
    mt = np.asarray(cells["pct_mt"], dtype=np.float64)
    # After filtering, the cells the ceiling took are gone from the dataset; the
    # step's record keeps their % mito and labels, so the table shows the cells
    # as they were before it (a type the ceiling emptied comes back as a row). A
    # column made after filtering has no recorded labels: ``assigned`` gives them.
    gone = qc_filter.recorded_removed_mito(adata, column)
    how = "record"
    unlabelled = qc_filter.recorded_removed_mito(adata, None) if gone is None and assigned is not None else None
    if unlabelled is not None:
        got = assigned(column) if callable(assigned) else assigned
        if got is not None and len(got[0]) == len(unlabelled[1]):
            gone, how = (list(got[0]), np.asarray(got[1], dtype=np.float64), unlabelled[2]), "assigned"
    by_filter = qc_filter.recorded_mito_bits(adata) if gone is not None else None
    extra: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if gone is not None:
        g_labels, g_mt, g_when = gone
        lab = np.asarray(g_labels, dtype=object)
        bits = by_filter if by_filter is not None and by_filter.size == lab.size else np.zeros(lab.size, bool)
        for name in dict.fromkeys(x for x in g_labels if x):
            extra[name] = (g_mt[lab == name], bits[lab == name])
    applied = _applied(recorded, column)
    judged = applied is not None and source != "requested"  # a typed ceiling is a what-if
    rows: list[TypeMito] = []
    for g, idx in zip(order, qc_groups.members(codes, len(order))):
        v, f = mt[idx], np.zeros(idx.size, dtype=bool)
        if g in extra:
            ev, ef = extra.pop(g)
            v, f = np.concatenate([v, ev]), np.concatenate([f, ef])
        rows.append(_type_row(g, v, cut, f if gone is not None else None))
    for g, (v, f) in extra.items():  # types with no cell left
        rows.append(_type_row(g, v, cut, f))
    if applied is not None:
        for r in rows:
            _recorded_rule(r, applied, judged)

    flagged = [r for r in rows if r.flagged]
    removed_global = int(sum(r.removed_global for r in rows))
    removed_bound = int(sum(r.removed_bound or 0 for r in rows))
    notes = []
    n_back = 0 if gone is None else int(sum(1 for x in gone[0] if x))
    if gone is not None:
        when = gone[2][:10]
        notes.append(f"Includes the {n_back:,} labelled cells the filtering step removed"
                     f"{' on ' + when if when else ''} (from its record"
                     + (f"; '{column}' was made after filtering, so each was given the group its "
                        "expression is closest to" if how == "assigned" else "")
                     + "): the table shows the cells as they were before that filter.")
    elif recorded:
        notes.append(f"This dataset was filtered by scView ({recorded.get('qc_threshold_mode')} mode): "
                     "cells it removed are gone, so the counts are what filtering again would remove.")
        if recorded.get("qc_threshold_mode") == "mad":
            notes.append("In MAD mode each sample's own bound (median + 3 MADs) can sit below the "
                         "ceiling; the filter preview shows that.")
    if applied is not None and source == "requested":
        notes.append("The filtering step used per-type bounds; the typed ceiling is a what-if.")
    if int((~labelled).sum()):
        notes.append(f"{int((~labelled).sum()):,} cells have no '{column}' label and are left out.")
    mito_mode = ((recorded or {}).get("qc_mito_mode") or "global") if recorded else None
    return MitoByType(
        **base, ceiling=cut, ceiling_source=source,
        filter_mode=(recorded or {}).get("qc_threshold_mode"),
        filter_mito_mode=mito_mode,
        filter_type_column=(recorded or {}).get("qc_mito_type_column") if mito_mode == "per_type" else None,
        already_filtered=bool(recorded),
        includes_removed=n_back, assigned_removed=n_back if how == "assigned" else 0,
        types=rows, n_flagged=len(flagged), removed_global_total=removed_global,
        removed_bound_total=removed_bound,
        summary=_summary(rows, flagged, cut, column if judged else None,
                         "cluster " if reason == "clustering" else ""),
        note=" ".join(notes),
    )


def _recorded_rule(row: TypeMito, applied: dict[str, float | None], judged: bool) -> None:
    """How the per-type filtering judged the type; one judged by its own bound is not
    flagged against a ceiling it never met (unless a ceiling was typed: a what-if)."""
    if row.group not in applied:
        return
    b = applied[row.group]
    row.applied_rule, row.applied_bound = ("own", round(b, 3)) if b is not None else ("global", None)
    if row.applied_rule == "own" and judged:
        row.flagged = False
        row.reasons = []
        row.note = " ".join(x for x in (f"Filtered with its own bound, {_pct(b)}.", row.note) if x)


def _per_cell(adata) -> dict:
    from scview.core.anndata_adaptor import per_cell_qc
    return per_cell_qc(adata)


def _type_row(group: str, v: np.ndarray, cut: float, by_filter: np.ndarray | None = None) -> TypeMito:
    """One type's row; ``by_filter`` (per value): removed by the filtering step's %
    mito rule, from its record (None: not filtered, or no record)."""
    finite = np.isfinite(v)
    v = v[finite]
    k = int(v.size)
    row = TypeMito(group=group, n_cells=k)
    if not k:
        row.note = "No cells with a % mito value."
        return row
    if by_filter is not None:
        row.removed_filter = int(np.asarray(by_filter)[finite].sum())
        row.removed_filter_frac = round(row.removed_filter / k, 4)
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
        bound = min(med + MAD_K * mad, BOUND_CAP)
        row.bound = round(bound, 3)
        row.removed_bound = int(((v > bound) | (v >= BOUND_CAP)).sum())
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
    elif k >= MIN_TYPE_CELLS and (row.removed_filter_frac or 0.0) > LOSS_FLAG:
        # MAD mode: each sample's own bound (median + 3 MADs) can sit well below the ceiling.
        row.flagged = True
        row.reasons.append(f"The filtering step's % mito rule removed {row.removed_filter_frac:.0%} of "
                           f"its {k:,} cells (median {_pct(med)}){own}.")
    return row


def _summary(rows: list[TypeMito], flagged: list[TypeMito], cut: float,
             per_type_column: str | None = None, kind: str = "") -> str:
    """``kind``: "cluster " before each name when the groups are a clustering's numbers."""
    if not rows:
        return ""
    damaged = [r for r in rows if r.damaged]
    tail = "" if not damaged else " " + "; ".join(
        f"{kind}{r.group} (median {_pct(r.median or 0.0)})" for r in damaged
    ) + (" looks" if len(damaged) == 1 else " look") + " damaged, not a high-baseline type."
    if per_type_column:  # filtered per type: the ceiling judged only the types without a bound
        own = sum(1 for r in rows if r.applied_rule == "own")
        rest = [r for r in rows if r.applied_rule != "own" and not r.damaged]
        head = (f"Filtered with per-type bounds on '{per_type_column}': {own} of {len(rows)} types "
                f"used their own bound, the rest the global rule.")
        if flagged:
            return f"{head} {_flag_summary(rest, flagged, cut, kind)}{tail}"
        return f"{head} No type that kept the global rule is flagged.{tail}"
    return _flag_summary([r for r in rows if not r.damaged], flagged, cut, kind) + tail


def _flag_summary(rows: list[TypeMito], flagged: list[TypeMito], cut: float, kind: str = "") -> str:
    if not rows:
        return ""
    if flagged:
        worst = max(flagged, key=lambda r: max(r.removed_global_frac, r.removed_filter_frac or 0.0))
        own = (f"; its own bound ({_pct(worst.bound)}) would remove {worst.removed_bound_frac:.0%}"
               if worst.bound is not None else "")
        more = f" {len(flagged) - 1} more type(s) flagged." if len(flagged) > 1 else ""
        if (worst.removed_filter_frac or 0.0) > worst.removed_global_frac:
            return (f"The filtering step's % mito rule removed {worst.removed_filter_frac:.0%} of "
                    f"{kind}{worst.group} (median {_pct(worst.median or 0.0)}){own}.{more}")
        return (f"A {_pct(cut)} ceiling removes {worst.removed_global_frac:.0%} of {kind}{worst.group} "
                f"(median {_pct(worst.median or 0.0)}){own}.{more}")
    sized = [r for r in rows if r.n_cells >= MIN_TYPE_CELLS] or rows  # a 9-cell group says little
    top = max(sized, key=lambda r: r.removed_global_frac)
    return (f"A {_pct(cut)} ceiling removes at most {top.removed_global_frac:.0%} of any type "
            f"({kind}{top.group}, median {_pct(top.median or 0.0)}): no type is flagged.")


def for_adaptor(adaptor, column: str | None = None, ceiling: float | None = None,
                assign: Callable[[str], tuple[list[str], np.ndarray] | None] | None = None) -> MitoByType:
    """``mito_by_type`` on a loaded dataset, reusing its cached per-cell QC arrays.
    ``assign(column)``: the removed cells' labels and % mito when the record has no
    labels in that column (``mito_refilter.assigned_removed``), asked only then."""
    fn = getattr(adaptor, "qc_per_cell", None)
    return mito_by_type(adaptor.adata, column, fn if callable(fn) else None, ceiling, assign)


TAG = "result:mito_by_type"


def filtering_rule(r: MitoByType) -> str:
    """Which % mito rule the dataset's filtering used, in a few words."""
    if not r.already_filtered:
        return "not filtered by scView yet"
    if r.filter_mito_mode == "per_type":
        return f"filtering used per-type bounds on '{r.filter_type_column}'"
    return f"filtering used one global rule ({r.filter_mode} mode)"


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
    flagged = ", ".join(f"{t.group} ({max(t.removed_global_frac, t.removed_filter_frac or 0.0):.0%} "
                        f"removed, median {_pct(t.median or 0.0)})" for t in r.types if t.flagged) or "none"
    line = (f"- % mito by cell type ('{r.column}', {r.ceiling_source} ceiling {_pct(r.ceiling or 0.0)}; "
            f"{filtering_rule(r)}): {r.summary} Flagged types: {flagged}. "
            f"Rule: {r.rule} [{TAG}]")
    return [line], f"{r.column}: {r.n_flagged} of {len(r.types)} types flagged at {_pct(r.ceiling or 0.0)}"
