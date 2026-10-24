"""Cell QC thresholds: which cells the filtering step removes, before it runs.

The QC histograms on the Data Assessment tab carry the filtering step's cell
thresholds (min / max genes, min counts, max % mito), and ``preview`` says what
they remove overall, per sample and per cluster, with a warning when one empties
a sample or a cluster or takes more than half of one: a real cell type with few
genes or much mitochondrial RNA (platelets, neutrophils, cardiomyocytes) is lost
that way without anyone noticing.

Two modes. ``fixed``: the cut-offs above, one set for every cell. ``mad`` (the
default for new runs): each sample's own outliers, after the single-cell best
practices book (Heumos et al. 2023): cells below median - 5 MADs of
log1p(counts) or of log1p(genes) per cell in their sample, and above median +
3 MADs of % mito. The book also cuts above median + 5 MADs of counts and genes;
that is opt-in here (``mad_upper``): on Kang it took 12.3 % of the dendritic
cells, 10 % of the erythrocytes and 9.3 % of the pDCs (large cells with more RNA),
and doublet detection already covers the high tail. The MAD is the raw median
absolute deviation (scipy's default scale), as in the book. In MAD mode the fixed
genes / counts cut-offs are not used; "max % mito" stays on as a ceiling over
the per-sample bound, because a
sample whose cells are mostly damaged has a high median and a wide MAD, and
median + 3 MADs would keep them (the book adds a hard 8 % for PBMCs; the 20 %
default suits tissues with more mitochondrial RNA, such as heart or kidney).

The samples are the donor / sample column ``qc_groups.sample_grouping`` picks
(or one named). Without one, all cells are one pooled group. A sample with fewer
than ``MIN_SAMPLE_CELLS`` cells, and cells without a sample, are judged against
the pooled median and MAD: the MAD's relative standard error is about
1.7 / sqrt(n) (17 % at 100 cells, 31 % at 30), so a bound 5 MADs out moves too
far on a small sample's own spread. A metric whose MAD is 0 in a sample (most
cells alike, e.g. 0 % mito in nuclei) sets no bound there.

Already-filtered data: the cells left have a narrower spread, so bounds worked
out again from them would trim the tails on every re-run. When scView's last
filtering of the dataset was a MAD run with the same settings, its recorded
thresholds are used again (``reused_from``); new settings work them out afresh
from the cells left. Data filtered elsewhere can only be judged as it is.

The filtering step judges cells with the same masks (``judge``) on the same
per-cell values (``per_cell_qc``: obs QC columns when present, else computed
from the counts, as the QC plots show them), so the preview is what the step
does. A cell without a value for a metric is not removed by it. % mito is used
only when it can be measured (the dataset has MT- genes).

% mito per cell type (``mito_mode`` "per_type"; the default "global" is the
rule above, unchanged). Cell types differ in their healthy mitochondrial share,
so one ceiling can remove most of a type with a high baseline (a 10 % ceiling
took 65 % of LNCaP cells, median ~12 %). Per type, each type with
``TYPE_MIN_CELLS`` cells or more and some spread gets its own bound, median +
``mad_k_mt`` raw MADs of % mito over all its cells, used INSTEAD of the global
rule (fixed: the ceiling; MAD: the sample's bound and the ceiling) for its cells.
A type that is too small, has no spread, or whose median is
``TYPE_DAMAGED_MEDIAN`` % or more (damaged cells, not a high baseline) keeps the
global rule, as do cells without a type. The counts and genes rules are not
changed. The bound works both ways: a type with a low baseline gets a bound
below the global one (ovary oocytes, median 0.8 %: 23 of 80 cells removed vs 5),
so the preview's per-type table shows each type's loss under both rules.

The floor (``mito_type_floor``, off by default): a type's bound is the higher of
its own and the global rule's, so per-type mode can only relax the global rule
for a type with a high baseline, never tighten it for one with a low baseline. A
cell is then removed for % mito only when both rules remove it, and no type ever
loses more cells than under the global rule. The global bound is the one the
global rule applies to that cell: the ceiling in fixed mode, and in MAD mode its
sample's bound (median + ``mad_k_mt`` MADs of its sample, under the ceiling), so
in MAD mode a type can keep its own bound in one sample and take the global one
in another. Without a global % mito rule (fixed mode, no ceiling) the floor
leaves nothing for % mito to remove.

Choices, with the reasons:
- Per type pooled over the samples, not per sample x type. The baseline is a
  property of the type, and the MAD of a small group is unreliable (relative SE
  ~1.7 / sqrt(n): 31 % at 30 cells); splitting 13 types over 8 donors leaves
  most sample x type groups well under the 100 cells a sample needs for MAD
  bounds of its own. A sample whose cells of one type are damaged would also raise
  that group's own median and so its own bound, keeping the damaged cells;
  pooled, the healthy samples set the type's baseline. A sample with damaged
  cells is still visible in the per-sample table.
- A hard cap, ``TYPE_BOUND_CAP`` (50 %): no type's bound goes above it, so a cell
  with half or more of its reads mitochondrial is removed whatever its type (the
  same line that calls a whole type damaged). A bound above ``TYPE_HIGH_BOUND``
  (30 %) and above the global ceiling is kept but warned about, with the number:
  it keeps cells a third of whose reads are mitochondrial.

Who gets removed (``audit``): the loss per cell type (an author label column,
else the clustering) and per condition (a condition-like column, else the
samples), with a chi-square test per cell type of removed ~ condition. A fixed
10 % mito cut-off that took 65 % of one cell type and 1-5 % of another, at a rate
that followed the drug (p = 2e-21), is a filter that selects a cell state, not
damaged cells; the flags say so with the numbers. The filtering step keeps a
record of the cells it removed (``RECORD_KEY`` in uns: their QC values and
labels), so the same audit is shown for what it actually removed
(``recorded_audit``), after the cells are gone.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from scview.core import cluster_qc, confounding, doublet_source, provenance, qc_groups

logger = logging.getLogger(__name__)

HALF = 0.5  # warn when a threshold removes more than this share of a group
MAX_CLUSTERS = cluster_qc.MAX_GROUPS

MODES = ("mad", "fixed")
MAD_K = 5.0  # counts and genes: below median - this many MADs (log1p), above too if mad_upper
MAD_K_MT = 3.0  # % mito: above median + this many MADs
MIN_SAMPLE_CELLS = 100  # fewer cells: the pooled median and MAD (module docstring)
MAX_SAMPLES = cluster_qc.MAX_GROUPS  # per-sample thresholds for at most this many samples

MITO_MODES = ("global", "per_type")
TYPE_MIN_CELLS = 30  # fewer cells: the type keeps the global % mito rule
TYPE_DAMAGED_MEDIAN = 50.0  # median % mito at or above this: damaged cells, the global rule
TYPE_BOUND_CAP = 50.0  # no type's own bound above this (module docstring)
TYPE_HIGH_BOUND = 30.0  # a type's own bound above this (and the ceiling) is warned about
MAX_TYPES = cluster_qc.MAX_GROUPS

# criterion -> what the user calls it (the threshold's label on the plots)
CRITERIA = {
    "min_genes": "Min genes",
    "max_genes": "Max genes",
    "min_counts": "Min counts",
    "max_counts": "Max counts",
    "max_pct_mt": "Max % mito",
    "doublets": "Predicted doublets",
    "hash_negatives": "Hashing negatives",
}

# MAD mode: (per-cell metric, criterion below the bound, criterion above it)
_MAD_METRICS = (
    ("total_counts", "min_counts", "max_counts"),
    ("n_genes", "min_genes", "max_genes"),
)


@dataclass(frozen=True)
class Thresholds:
    """Cell thresholds; 0 (or None for % mito) turns one off.

    ``mode`` "mad" replaces the genes / counts cut-offs with each sample's
    outliers (``mad_bounds``); ``max_pct_mt`` is then a ceiling."""

    min_genes: float = 0.0  # keep cells with at least this many genes
    max_genes: float = 0.0  # keep cells with at most this many genes
    min_counts: float = 0.0  # keep cells with at least this many counts
    max_pct_mt: float | None = None  # keep cells with % mito below this
    drop_doublets: bool = False  # remove cells called doublets (predicted_doublet)
    mode: str = "fixed"  # "fixed" | "mad"
    mad_k: float = MAD_K
    mad_k_mt: float = MAD_K_MT
    mad_upper: bool = False  # MAD mode: also cut counts / genes above median + k MADs
    # "per_type": each cell type's own % mito bound (median + mad_k_mt MADs) instead
    # of the global rule (module docstring); "global": the rule above for every cell.
    mito_mode: str = "global"
    mito_type_column: str = ""  # per_type: the cell types ("" = mito_by_type.default_column)
    # per_type: never stricter than the global rule (a type's bound = max(own, global))
    mito_type_floor: bool = False
    # Whose doublet calls drop_doublets drops (doublet_source.SOURCES): "auto" = cell
    # hashing when it was run, else Scrublet; and hashing negatives, a separate opt-in.
    doublet_source: str = "auto"
    drop_hash_negatives: bool = False

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
            mode=str(getattr(params, "qc_threshold_mode", "fixed") or "fixed"),
            mad_k=float(getattr(params, "qc_mad_k", MAD_K)),
            mad_k_mt=float(getattr(params, "qc_mad_k_mt", MAD_K_MT)),
            mad_upper=bool(getattr(params, "qc_mad_upper", False)),
            mito_mode=str(getattr(params, "qc_mito_mode", "global") or "global"),
            mito_type_column=str(getattr(params, "qc_mito_type_column", "") or ""),
            mito_type_floor=bool(getattr(params, "qc_mito_type_floor", False)),
            doublet_source=str(getattr(params, "qc_doublet_source", "auto") or "auto"),
            drop_hash_negatives=bool(getattr(params, "drop_hash_negatives", False)),
        )

    def check(self) -> None:
        """Raise ValueError for an unknown mode or a MAD multiplier that is not positive."""
        if self.mode not in MODES:
            raise ValueError(f"Unknown QC threshold mode '{self.mode}': use 'mad' or 'fixed'.")
        if self.mito_mode not in MITO_MODES:
            raise ValueError(f"Unknown % mito mode '{self.mito_mode}': use 'global' or 'per_type'.")
        if self.doublet_source not in doublet_source.SOURCES:
            raise ValueError(f"Unknown doublet source '{self.doublet_source}': use "
                             f"{', '.join(doublet_source.SOURCES)}.")
        checks = [("MAD multiplier", self.mad_k)] if self.mode == "mad" else []
        if self.mode == "mad" or self.mito_mode == "per_type":
            checks.append(("% mito MAD multiplier", self.mad_k_mt))
        for name, k in checks:
            if not (math.isfinite(k) and k > 0):
                raise ValueError(f"The {name} must be a positive number, not {k}.")


@dataclass
class Samples:
    """The cells' samples: the column, why it was picked, its groups and each
    cell's index into them (-1 = no value, or no column: one pooled group)."""

    column: str | None
    reason: str | None
    names: list[str]
    codes: np.ndarray


def samples_of(adata, sample_column: str | None = None) -> Samples:
    """The sample column named, else the donor / sample (or batch) column
    ``qc_groups.sample_grouping`` picks, else none. Raises ValueError for a named
    column that does not exist."""
    if sample_column:
        if sample_column not in adata.obs.columns:
            raise ValueError(f"Column '{sample_column}' not found.")
        column, reason = sample_column, "requested"
    else:
        column, reason = qc_groups.sample_grouping(adata)
    if not column:
        return Samples(None, None, [], np.full(int(adata.n_obs), -1))
    names, codes = qc_groups.group_codes(adata.obs[column])
    return Samples(column, reason, names, np.asarray(codes))


@dataclass
class MadBounds:
    """Each sample's outlier bounds, and per cell the bounds it is judged by.

    ``lo`` / ``hi``: per-cell bounds on the log1p scale for counts and genes, in %
    for mito (-inf / inf where a metric sets none). ``groups`` and ``pooled``
    are the bounds for display and provenance, in counts, genes and %."""

    t: Thresholds
    samples: Samples
    groups: list[dict]
    pooled: dict
    lo: dict[str, np.ndarray] = field(default_factory=dict)
    hi: dict[str, np.ndarray] = field(default_factory=dict)
    reused_from: str | None = None  # when the bounds are a previous run's (its timestamp)

    def to_dict(self, drop: np.ndarray | None = None) -> dict:
        """The thresholds applied; with ``drop`` (cells removed), each sample's loss."""
        groups = [dict(g) for g in self.groups]
        if drop is not None and groups:
            labelled = self.samples.codes >= 0
            lost = np.bincount(self.samples.codes[labelled & drop], minlength=len(groups))
            for g, n in zip(groups, lost):
                g["n_removed"] = int(n)
        return {
            "column": self.samples.column,
            "reason": self.samples.reason,
            "k": self.t.mad_k,
            "k_mt": self.t.mad_k_mt,
            "upper": self.t.mad_upper,
            "ceiling_pct_mt": self.t.max_pct_mt if "pct_mt" in self.hi else None,
            "min_sample_cells": MIN_SAMPLE_CELLS,
            "samples": groups,
            "pooled": self.pooled,
            "n_unlabelled": int((self.samples.codes < 0).sum()) if self.samples.column else 0,
            "reused_from": self.reused_from,
        }


def _values(cells: dict, key: str) -> np.ndarray:
    return np.asarray(cells[key], dtype=np.float64)


def _mito_measurable(cells: dict) -> bool:
    return bool(cells.get("has_mt", True)) and cells.get("pct_mt") is not None


def _median_mad(v: np.ndarray) -> tuple[float, float]:
    v = v[np.isfinite(v)]
    if not v.size:
        return math.nan, math.nan
    med = float(np.median(v))
    return med, float(np.median(np.abs(v - med)))


_BOUNDS = ("min_counts", "max_counts", "min_genes", "max_genes", "max_pct_mt")


def _bounds(stats: dict[str, tuple[float, float]], t: Thresholds, ceiling: float | None) -> dict:
    """One sample's bounds in counts, genes and % from the (median, MAD) of each
    metric (log1p for counts and genes); None where a MAD of 0 sets none, and for
    the counts / genes upper bounds unless ``t.mad_upper``."""
    skipped = [m for m, (_, mad) in stats.items() if not (mad > 0)]  # 0, or no values
    out: dict = {}
    for key, below, above in _MAD_METRICS:
        med, mad = stats[key]
        on = key not in skipped
        out[below] = max(0.0, math.expm1(med - t.mad_k * mad)) if on else None
        out[above] = math.expm1(med + t.mad_k * mad) if on and t.mad_upper else None
    mt = None
    if "pct_mt" in stats:
        med, mad = stats["pct_mt"]
        bound = med + t.mad_k_mt * mad if "pct_mt" not in skipped else math.inf
        mt = min(bound, ceiling) if ceiling is not None else bound
    out["max_pct_mt"] = mt if mt is not None and math.isfinite(mt) else None
    out["skipped"] = skipped
    return out


def _reusable(recorded: dict | None, t: Thresholds, column: str | None,
              ceiling: float | None) -> dict | None:
    """The thresholds the last filtering recorded, when it ran in MAD mode with the
    settings this run has. The cells left are those that kept within them, so
    using them again removes nothing new; bounds worked out afresh from the
    cells left, whose spread is narrower, would trim the tails again on every
    re-run (Reprocess from Scratch: Kang lost 52 more cells, pbmc_1k_v3 41)."""
    if not recorded or recorded.get("qc_threshold_mode") != "mad":
        return None
    rec = recorded.get("qc_thresholds")
    if not isinstance(rec, dict) or not isinstance(rec.get("pooled"), dict):
        return None
    rows = [rec["pooled"], *(rec.get("samples") or [])]
    if not all(isinstance(r, dict) and all(r.get(b) is None or isinstance(r.get(b), (int, float))
                                           for b in _BOUNDS) for r in rows):
        return None  # not a record this code wrote: work them out afresh
    # Records from before the upper bounds were opt-in had them on.
    same = (rec.get("k"), rec.get("k_mt"), bool(rec.get("upper", True)), rec.get("ceiling_pct_mt"),
            rec.get("column"))
    return rec if same == (t.mad_k, t.mad_k_mt, t.mad_upper, ceiling, column) else None


def mad_bounds(cells: dict, t: Thresholds, samples: Samples | None = None,
               recorded: dict | None = None) -> MadBounds:
    """Per-sample bounds: median - ``t.mad_k`` MADs of log1p(counts) and of
    log1p(genes) (and + ``t.mad_k`` MADs when ``t.mad_upper``), median +
    ``t.mad_k_mt`` MADs of % mito (capped by ``t.max_pct_mt``). Small samples and
    unlabelled cells take the pooled bounds; a MAD of 0 sets no bound (that metric
    is listed in ``skipped``).

    ``recorded`` is ``recorded_filtering``'s output: a MAD run with the same
    settings already filtered these cells, and its thresholds are used again."""
    n = len(_values(cells, "n_genes"))
    if samples is None:
        samples = Samples(None, None, [], np.full(n, -1))
    k = len(samples.names)
    if k > MAX_SAMPLES:
        raise ValueError(
            f"'{samples.column}' has {k} groups; per-sample thresholds take at most "
            f"{MAX_SAMPLES}. Pick the sample column, or use fixed cut-offs."
        )
    has_mt = _mito_measurable(cells)
    ceiling = t.max_pct_mt if has_mt else None
    groups = qc_groups.members(samples.codes, k)
    reuse = _reusable(recorded, t, samples.column, ceiling)
    if reuse is not None:
        rec_rows = {r.get("name"): r for r in reuse.get("samples") or []}
        pooled = {"n_cells": n, **{b: reuse["pooled"].get(b) for b in _BOUNDS},
                  "skipped": reuse["pooled"].get("skipped", [])}
        rows = []
        for name, m in zip(samples.names, groups):
            r = rec_rows.get(name)
            src = r if r is not None else pooled
            rows.append({"name": name, "n_cells": int(m.size),
                         "pooled": bool(r.get("pooled")) if r is not None else True,
                         **{b: src.get(b) for b in _BOUNDS}, "skipped": src.get("skipped", [])})
    else:
        with np.errstate(invalid="ignore"):
            values = {key: np.log1p(_values(cells, key)) for key, _, _ in _MAD_METRICS}
        if has_mt:
            values["pct_mt"] = _values(cells, "pct_mt")
        pooled_stats = {m: _median_mad(v) for m, v in values.items()}
        rows = []
        for name, m in zip(samples.names, groups):
            own = m.size >= MIN_SAMPLE_CELLS
            stats = {key: _median_mad(v[m]) for key, v in values.items()} if own else pooled_stats
            rows.append({"name": name, "n_cells": int(m.size), "pooled": not own,
                         **_bounds(stats, t, ceiling)})
        pooled = {"n_cells": n, **_bounds(pooled_stats, t, ceiling)}

    # Per cell, on the scale the masks compare: log1p for counts and genes, % for
    # mito. Row i < k is sample i; row k the pooled bounds (unlabelled cells).
    table = rows + [pooled]

    def column(name: str, log: bool, missing: float) -> np.ndarray:
        vals = [r.get(name) for r in table]
        return np.array([missing if v is None else (math.log1p(v) if log else v) for v in vals])

    lo, hi = {}, {}
    for key, below, above in _MAD_METRICS:
        lo[key] = column(below, True, -np.inf)
        hi[key] = column(above, True, np.inf)
    if has_mt:
        hi["pct_mt"] = column("max_pct_mt", False, np.inf)
    row_of = np.where(samples.codes >= 0, samples.codes, k)
    return MadBounds(t, samples, rows, pooled,
                     lo={m: a[row_of] for m, a in lo.items()},
                     hi={m: a[row_of] for m, a in hi.items()},
                     reused_from=recorded.get("timestamp") if reuse is not None else None)


@dataclass
class TypeBounds:
    """Per-type % mito bounds (``mito_mode`` "per_type") and, per cell, the rule it
    is judged by: ``own`` cells by their type's bound ``hi`` (inf elsewhere), the
    rest by the global rule. ``rows`` are the types in ``column``'s order: cells,
    median, MAD, ``bound`` (None: the global rule, ``why`` says why). ``judge_types``
    fills ``removed_global`` / ``removed`` (the global rule's % mito mask and the one
    applied) for the per-type loss table, and ``floor_info`` the floor's columns:
    ``removed_type`` / ``removed_floor`` (the per-type mask with the floor off and
    on) and ``global_hi`` (per cell, the bound the global rule applies to it)."""

    column: str
    reason: str | None
    k: float
    codes: np.ndarray
    rows: list[dict]
    own: np.ndarray
    hi: np.ndarray
    ceiling: float | None = None
    reused_from: str | None = None
    removed_global: np.ndarray | None = None
    removed: np.ndarray | None = None
    floor: bool = False
    removed_type: np.ndarray | None = None
    removed_floor: np.ndarray | None = None
    global_hi: np.ndarray | None = None

    def to_dict(self) -> dict:
        """The rule per type for the preview, the record and provenance, with each
        type's loss under the global rule and under the one applied (and, once
        ``floor_info`` ran, with the floor off and on, and which bound the floor
        picks: ``floor_rows``)."""
        rows = [dict(r) for r in self.rows]
        k = len(rows)
        labelled = self.codes >= 0
        for key, mask in (("removed_global", self.removed_global), ("removed", self.removed),
                          ("removed_type", self.removed_type), ("removed_floor", self.removed_floor)):
            if mask is None:
                continue
            lost = np.bincount(self.codes[labelled & mask], minlength=k)
            for r, v in zip(rows, lost):
                r[key] = int(v)
        floor_rows(self, rows)
        own = [r for r in rows if r["bound"] is not None]
        return {
            "column": self.column,
            "reason": self.reason,
            "k": self.k,
            "cap": TYPE_BOUND_CAP,
            "min_cells": TYPE_MIN_CELLS,
            "damaged_median": TYPE_DAMAGED_MEDIAN,
            "ceiling_pct_mt": self.ceiling,
            "types": rows,
            "n_own": len(own),
            "n_global": k - len(own),
            "n_unlabelled": int((~labelled).sum()),
            "reused_from": self.reused_from,
            "warnings": [r["warning"] for r in rows if r.get("warning")],
            "floor": self.floor,
        }


def _type_rule(name: str, v: np.ndarray, k: float, ceiling: float | None) -> dict:
    """One type's % mito rule from its cells' values (module docstring)."""
    v = v[np.isfinite(v)]
    row: dict = {"name": name, "n_cells": int(v.size), "median": None, "mad": None, "bound": None,
                 "rule": "global", "why": None, "capped": False, "note": "", "warning": None}
    if not v.size:
        row.update(why="no_values", note="No cells with a % mito value: the global rule applies.")
        return row
    med, mad = _median_mad(v)
    row.update(median=round(med, 3), mad=round(mad, 3))
    if med >= TYPE_DAMAGED_MEDIAN:
        row.update(why="damaged", note=(f"Median {med:.1f}% mito: most of its reads are mitochondrial, "
                                        "likely damaged cells rather than a high baseline, so it keeps "
                                        "the global rule."))
    elif v.size < TYPE_MIN_CELLS:
        row.update(why="too_few", note=f"Only {v.size} cells: too few for a bound of its own, so it "
                                       "keeps the global rule.")
    elif not (mad > 0):
        row.update(why="no_spread", note="No spread in % mito (MAD 0): no bound of its own, so it "
                                         "keeps the global rule.")
    else:
        bound = med + k * mad
        row.update(rule="own", bound=round(min(bound, TYPE_BOUND_CAP), 4))
        if bound > TYPE_BOUND_CAP:
            row.update(capped=True, note=(f"Median + {k:g} MADs is {bound:.1f}%: capped at "
                                          f"{TYPE_BOUND_CAP:g}%, so cells with half or more of their "
                                          "reads mitochondrial still go."))
        if row["bound"] > TYPE_HIGH_BOUND and (ceiling is None or row["bound"] > ceiling):
            row["warning"] = (f"{name}: its own bound, {row['bound']:.1f}%, keeps cells with up to that "
                              f"share of mitochondrial reads (median {med:.1f}%). Check the type is not "
                              "damaged cells (few genes, no clear markers) before keeping them.")
    return row


def _type_column(adata, requested: str, recorded: dict | None) -> tuple[str, str | None]:
    if requested:
        if requested not in adata.obs.columns:
            raise ValueError(f"Column '{requested}' not found.")
        return requested, "requested"
    rec_col = (recorded or {}).get("qc_mito_type_column")
    if (recorded or {}).get("qc_mito_mode") == "per_type" and rec_col in adata.obs.columns:
        return str(rec_col), "filtering"  # the column the last per-type filtering used
    from scview.core.mito_by_type import default_column  # it imports this module

    column, reason = default_column(adata)
    if not column:
        raise ValueError("Per-type % mito bounds need cell types (an author annotation or a "
                         "clustering) and this dataset has none: pick a column, or use the "
                         "global rule.")
    return column, reason


def _reusable_types(recorded: dict | None, t: Thresholds, column: str) -> dict[str, dict] | None:
    """The per-type rows the last filtering recorded, by type, when it ran per type
    with this column and multiplier: the cells left kept within them, so bounds
    worked out again from those cells, whose spread is narrower, would trim them
    again on every re-run (as ``_reusable`` for the samples' bounds)."""
    if not recorded or recorded.get("qc_mito_mode") != "per_type":
        return None
    rec = recorded.get("qc_mito_types")
    if not isinstance(rec, dict) or rec.get("column") != column or rec.get("k") != t.mad_k_mt:
        return None
    rows = rec.get("types")
    if not isinstance(rows, list) or not all(
            isinstance(r, dict) and isinstance(r.get("name"), str)
            and (r.get("bound") is None or isinstance(r.get("bound"), (int, float))) for r in rows):
        return None
    return {r["name"]: r for r in rows}


def type_bounds(adata, cells: dict, t: Thresholds, recorded: dict | None = None) -> TypeBounds:
    """Each type's % mito rule (``t.mito_type_column``; empty: the column the last
    per-type filtering used, else ``mito_by_type.default_column``) and the per-cell
    arrays the masks compare. ``recorded`` is ``recorded_filtering``'s output: a
    per-type run with the same column and multiplier is reused. Raises ValueError
    for a missing column, none at all, or one with too many groups."""
    column, reason = _type_column(adata, t.mito_type_column, recorded)
    names, codes = qc_groups.group_codes(adata.obs[column])
    k = len(names)
    if k > MAX_TYPES:
        raise ValueError(f"'{column}' has {k} groups; per-type % mito bounds take at most {MAX_TYPES}.")
    pct = _values(cells, "pct_mt")
    ceiling = t.max_pct_mt
    reuse = _reusable_types(recorded, t, column)
    rows = []
    for name, m in zip(names, qc_groups.members(codes, k)):
        rec = (reuse or {}).get(name)
        if rec is not None:
            n_cells = int(np.isfinite(pct[m]).sum())
            rows.append({**_type_rule(name, np.array([]), t.mad_k_mt, ceiling),
                         **{key: rec.get(key) for key in ("median", "mad", "bound", "rule", "why",
                                                          "capped", "note", "warning")},
                         "n_cells": n_cells})
        else:
            rows.append(_type_rule(name, pct[m], t.mad_k_mt, ceiling))
    bound_of = np.array([np.inf if r["bound"] is None else float(r["bound"]) for r in rows] + [np.inf])
    own_of = np.array([r["bound"] is not None for r in rows] + [False])
    row_of = np.where(codes >= 0, codes, k)
    return TypeBounds(column, reason, t.mad_k_mt, np.asarray(codes), rows, own_of[row_of],
                      bound_of[row_of], ceiling,
                      reused_from=recorded.get("timestamp") if reuse is not None else None)


def _per_type_mito(cells: dict, types: TypeBounds, global_mask: np.ndarray | None) -> np.ndarray:
    """% mito per type: a type with its own bound removes its cells above it (or at
    the cap); the other cells keep ``global_mask``, the global rule's."""
    pct = _values(cells, "pct_mt")
    with np.errstate(invalid="ignore"):
        mine = np.isfinite(pct) & ((pct > types.hi) | (pct >= TYPE_BOUND_CAP))
    base = global_mask if global_mask is not None else np.zeros(pct.size, dtype=bool)
    return np.where(types.own, mine, base)


# --- the floor: per type never stricter than the global rule (module docstring) ---------


def _floored_mito(per_type: np.ndarray, global_mask: np.ndarray | None) -> np.ndarray:
    """``mito_type_floor``: each cell's bound is the higher of its type's own and the
    global rule's for that cell, so it goes only when both rules remove it (a cell
    without a bound of its own already follows the global rule alone)."""
    if global_mask is None:  # no global % mito rule: its bound is infinite
        return np.zeros(per_type.size, dtype=bool)
    return per_type & global_mask


def global_mito_bound(cells: dict, t: Thresholds, bounds: MadBounds | None) -> np.ndarray:
    """Per cell, the % mito bound the global rule applies to it: in MAD mode its
    sample's bound (already under the ceiling), in fixed mode the ceiling; inf
    where there is none."""
    if t.mode == "mad" and bounds is not None and "pct_mt" in bounds.hi:
        return np.asarray(bounds.hi["pct_mt"], dtype=np.float64)
    n = len(_values(cells, "n_genes"))
    return np.full(n, np.inf if t.max_pct_mt is None else float(t.max_pct_mt))


def floor_info(types: TypeBounds, cells: dict, t: Thresholds, bounds: MadBounds | None,
               global_mask: np.ndarray | None) -> None:
    """Fill in the floor's columns on ``types`` (the per-type % mito mask with the
    floor off and on, each cell's global bound), whichever is applied, so the
    preview shows both."""
    own = _per_type_mito(cells, types, global_mask)
    types.floor = t.mito_type_floor
    types.removed_type = own
    types.removed_floor = _floored_mito(own, global_mask)
    types.global_hi = global_mito_bound(cells, t, bounds)


def floor_rows(types: TypeBounds, rows: list[dict]) -> None:
    """Per type with a bound of its own, which bound the floor uses for its cells:
    ``floor_uses`` "type" (its own is the higher for every cell), "global" (the
    global rule's for every cell) or "mixed" (MAD mode: the samples' bounds differ),
    ``n_floored`` (its cells whose global bound is the higher) and ``global_bound``
    ([lowest, highest] global bound over its cells; None when there is none). A
    type without a bound of its own follows the global rule (``floor_uses`` None)."""
    if types.global_hi is None:
        return
    k = len(rows)
    labelled = types.codes >= 0
    g = types.global_hi
    lifted = labelled & types.own & (g > types.hi)
    n_lifted = np.bincount(types.codes[lifted], minlength=k)
    n_own = np.bincount(types.codes[labelled & types.own], minlength=k)
    fin = labelled & np.isfinite(g)
    lo, hi = np.full(k, np.inf), np.full(k, -np.inf)
    np.minimum.at(lo, types.codes[fin], g[fin])
    np.maximum.at(hi, types.codes[fin], g[fin])
    for i, r in enumerate(rows):
        r["global_bound"] = [round(float(lo[i]), 3), round(float(hi[i]), 3)] if lo[i] <= hi[i] else None
        if r["bound"] is None:
            r.update(floor_uses=None, n_floored=0)
            continue
        n, m = int(n_lifted[i]), int(n_own[i])
        r.update(floor_uses="type" if n == 0 else "global" if n == m else "mixed", n_floored=n)


def removed_by(cells: dict, t: Thresholds, doublets: np.ndarray | None = None,
               bounds: MadBounds | None = None, types: TypeBounds | None = None
               ) -> dict[str, np.ndarray]:
    """For each threshold that is on, the cells it removes on its own (boolean masks).

    ``cells`` is ``per_cell_qc``'s output; ``doublets`` the per-cell doublet calls
    (an array, used when ``t.drop_doublets``) or ``cell_calls``' ``CellCalls`` (the
    source's doublets, and hashing negatives for ``t.drop_hash_negatives``). In MAD mode
    ``bounds`` are the samples' (``mad_bounds``); without them all cells pool.
    ``types`` (``type_bounds``) replaces the % mito rule for the cells of a type
    with a bound of its own (``mito_mode`` "per_type"); with ``t.mito_type_floor``
    never stricter than the global rule."""
    if t.mode == "mad":
        out = _removed_by_mad(cells, t, bounds if bounds is not None else mad_bounds(cells, t))
    else:
        out = _removed_by_fixed(cells, t)
    if types is not None and _mito_measurable(cells):
        glob = out.get("max_pct_mt")
        out["max_pct_mt"] = _per_type_mito(cells, types, glob)
        if t.mito_type_floor:
            out["max_pct_mt"] = _floored_mito(out["max_pct_mt"], glob)
    out.update(_call_masks(t, doublets))
    return out


def _removed_by_fixed(cells: dict, t: Thresholds) -> dict[str, np.ndarray]:
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
    return out


def _removed_by_mad(cells: dict, t: Thresholds, bounds: MadBounds) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    with np.errstate(invalid="ignore"):
        for key, below, above in _MAD_METRICS:
            v = np.log1p(_values(cells, key))
            ok = np.isfinite(v)
            out[below] = ok & (v < bounds.lo[key])  # strictly outside, as the book has it
            if t.mad_upper:
                out[above] = ok & (v > bounds.hi[key])
        if "pct_mt" in bounds.hi:
            pct = _values(cells, "pct_mt")
            ok = np.isfinite(pct)
            mask = ok & (pct > bounds.hi["pct_mt"])
            if t.max_pct_mt is not None:
                mask |= ok & (pct >= t.max_pct_mt)  # the ceiling keeps below it, as fixed mode
            out["max_pct_mt"] = mask
    return out


def judge(adata, cells: dict, t: Thresholds, samples: Samples | None = None
          ) -> tuple[dict[str, np.ndarray], MadBounds | None]:
    """What ``t`` removes from ``adata`` (``cells`` = its ``per_cell_qc``): the
    masks per criterion and, in MAD mode, the samples' bounds (``judge_types``
    without the per-type rows).

    ``samples`` defaults to ``samples_of(adata)``. Raises ValueError for a bad
    mode or multiplier, or too many samples."""
    removed, bounds, _ = judge_types(adata, cells, t, samples)
    return removed, bounds


def judge_types(adata, cells: dict, t: Thresholds, samples: Samples | None = None
                ) -> tuple[dict[str, np.ndarray], MadBounds | None, TypeBounds | None]:
    """``judge`` plus, with ``mito_mode`` "per_type" and % mito measurable, each
    type's rule (``type_bounds``, with the global and applied % mito masks filled
    in, and the floor's columns: ``floor_info``). The preview and the filtering
    step both call this, so the preview is what the step does. Raises ValueError as ``judge`` does, and for a per-type column
    that is missing or has too many groups."""
    t.check()
    calls = cell_calls(adata.obs, t)
    per_type = t.mito_mode == "per_type" and _mito_measurable(cells)
    if t.mode != "mad" and not per_type:
        return removed_by(cells, t, calls), None, None
    recorded = recorded_filtering(adata)
    bounds = None
    if t.mode == "mad":
        bounds = mad_bounds(cells, t, samples if samples is not None else samples_of(adata), recorded)
    if not per_type:
        return removed_by(cells, t, calls, bounds), bounds, None
    types = type_bounds(adata, cells, t, recorded)
    removed = removed_by(cells, t, calls, bounds, types)
    glob = removed_by(cells, t, None, bounds).get("max_pct_mt")
    types.removed_global = glob if glob is not None else np.zeros(int(adata.n_obs), dtype=bool)
    types.removed = removed["max_pct_mt"]
    floor_info(types, cells, t, bounds, glob)
    return removed, bounds, types


def applied_summary(applied: dict) -> str:
    """A MAD run's recorded thresholds (``MadBounds.to_dict``) in one line, for a
    methods text: the rule, the samples, and the range of each bound across them."""
    samples = applied.get("samples") or []
    rows = samples or [applied.get("pooled") or {}]

    def span(key: str, digits: int = 0) -> str | None:
        vals = [float(r[key]) for r in rows if isinstance(r.get(key), (int, float))]
        if not vals:
            return None
        lo, hi = f"{min(vals):,.{digits}f}", f"{max(vals):,.{digits}f}"
        return lo if lo == hi else f"{lo}-{hi}"

    parts = []
    for label, below, above in (("counts", "min_counts", "max_counts"),
                                ("genes", "min_genes", "max_genes")):
        a, b = span(below), span(above)
        if a and b:
            parts.append(f"{label} per cell kept from {a} to {b}")
        elif a or b:
            parts.append(f"{label} per cell at least {a}" if a else f"{label} per cell at most {b}")
    mt = span("max_pct_mt", 1)
    if mt:
        parts.append(f"% mito below {mt}")
    n_pooled = sum(1 for s in samples if s.get("pooled"))
    where = (f"within each {applied['column']} ({len(samples)} samples"
             + (f", {n_pooled} too small and judged on all cells" if n_pooled else "") + ")"
             if applied.get("column") else "over all cells (no sample column)")
    side = "beyond median +/-" if applied.get("upper", True) else "below median -"
    rule = (f"outliers {side} {applied.get('k')} MADs of log1p counts and genes"
            f" and above median + {applied.get('k_mt')} MADs of % mito")
    return f"{rule}, {where}" + (": " + "; ".join(parts) if parts else "")


def recorded_filtering(adata) -> dict | None:
    """How scView last filtered this dataset: the filtering step's recorded params
    plus its timestamp (``qc_threshold_mode`` "fixed" for runs from before MAD
    mode), or None when it never ran."""
    try:
        for h in reversed(provenance.read_provenance(adata).get("history", [])):
            if h.get("step") == "filtering":
                params = dict(h.get("params") or {})
                params.setdefault("qc_threshold_mode", "fixed")
                params["timestamp"] = h.get("timestamp")
                return params
    except Exception:  # a malformed history never breaks the preview
        return None
    return None


def keep_mask(removed: dict[str, np.ndarray], n_cells: int) -> np.ndarray:
    """Cells no threshold removes."""
    drop = np.zeros(n_cells, dtype=bool)
    for mask in removed.values():
        drop |= mask
    return ~drop


def _grouped(adata, column: str | None, reason: str | None, removed: dict[str, np.ndarray],
             drop: np.ndarray, max_groups: int,
             coded: tuple[list[str], np.ndarray] | None = None) -> dict | None:
    if not column:
        return None
    if column not in adata.obs.columns:
        raise ValueError(f"Column '{column}' not found.")
    order, codes = coded if coded is not None else qc_groups.group_codes(adata.obs[column])
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
            cluster_column: str | None = None, label_column: str | None = None,
            condition_column: str | None = None) -> dict:
    """What ``t`` removes: overall, per threshold on its own, per sample (the
    donor / sample column, else a batch column) and per cluster (the active
    clustering), plus warnings for groups it empties or more than halves.

    Each sample also carries its QC medians and range flags
    (``qc_groups.sample_summary``); in MAD mode ``mad`` holds each sample's
    thresholds (``MadBounds.to_dict``), the samples being the same groups.
    ``previous`` is how scView last filtered the dataset, if it did.

    ``audit``: who the thresholds remove per label group (``label_column``,
    default an author cell-type column, else the clustering) x condition
    (``condition_column``, default a condition-like column, else the samples),
    with a removed ~ condition test per group and flags (``audit_flags``; a group
    the warnings already cover is not flagged again). ``recorded``: the same for
    the cells the last filtering step actually removed (``recorded_audit``).

    ``mito_types`` (per-type mode, % mito measurable): per cell type
    (``type_bounds``) its cells, median, the rule and bound used, and the cells
    removed for % mito under the global rule vs the rule applied, and per type
    with the floor off vs on (``removed_type`` / ``removed_floor``, ``floor_uses``).

    ``cells`` is ``AnnDataAdaptor.qc_per_cell()``. Raises ValueError for a named
    column that does not exist, or for thresholds ``judge`` refuses."""
    n = int(adata.n_obs)
    samp = samples_of(adata, sample_column)
    removed, bounds, types = judge_types(adata, cells, t, samp)
    drop = ~keep_mask(removed, n)
    n_removed = int(drop.sum())

    c_col = cluster_column or cluster_qc.default_grouping(adata)
    c_reason = "requested" if cluster_column else ("clustering" if c_col else None)
    samples = _grouped(adata, samp.column, samp.reason, removed, drop, qc_groups.MAX_GROUPS,
                       coded=(samp.names, samp.codes))
    if samples and samples["groups"]:
        summary = qc_groups.sample_summary(cells, samp.names, samp.codes, MIN_SAMPLE_CELLS)
        for g, row in zip(samples["groups"], summary["rows"]):
            g.update({k: v for k, v in row.items() if k not in ("name", "n_cells")})
        samples.update(typical=summary["typical"], flag_help=summary["flag_help"],
                       compared=summary["compared"])
    clusters = None if c_col and c_col == samp.column else _grouped(
        adata, c_col, c_reason, removed, drop, MAX_CLUSTERS)

    has_mt = bool(cells.get("has_mt", True))
    warnings = _warnings("sample", samples) + _warnings("cluster", clusters)
    warnings.sort(key=lambda w: (not w["whole"], -w["fraction"]))
    previous = recorded_filtering(adata)
    who = audit(adata, drop, removed, label_column, condition_column, samp.column,
                skip={(w["column"], w["group"]) for w in warnings})
    return {
        "n_cells": n,
        "n_removed": n_removed,
        "n_kept": n - n_removed,
        "fraction_removed": round(n_removed / n, 4) if n else 0.0,
        "thresholds": {
            "mode": t.mode,
            "min_genes": t.min_genes,
            "max_genes": t.max_genes,
            "min_counts": t.min_counts,
            "max_pct_mt": t.max_pct_mt if has_mt else None,
            "drop_doublets": t.drop_doublets,
            "mad_k": t.mad_k,
            "mad_k_mt": t.mad_k_mt,
            "mad_upper": t.mad_upper,
            "mito_mode": t.mito_mode,
            "mito_type_column": t.mito_type_column,
            "mito_type_floor": t.mito_type_floor,
        },
        "has_mt": has_mt,
        "mito_types": _types_table(types),
        "doublet_calls": cluster_qc.doublet_calls(adata.obs) is not None,  # "drop doublets" can act
        "doublets": doublet_choices(adata.obs, t),  # whose calls, and hashing negatives
        "by_criterion": {c: {"label": CRITERIA[c], "n_removed": int(m.sum())} for c, m in removed.items()},
        "mad": bounds.to_dict(drop) if bounds is not None else None,
        "samples": samples,
        "clusters": clusters,
        "warnings": warnings,
        "previous": None if previous is None else {
            "mode": previous["qc_threshold_mode"], "timestamp": previous.get("timestamp")},
        "audit": who,
        "recorded": recorded_audit(adata, label_column, condition_column),
    }


def _types_table(types: TypeBounds | None) -> dict | None:
    """The preview's per-type % mito table (per-type mode only: the MitoByType card
    previews the bounds before the mode is chosen, and a global preview stays as
    small as it was on every threshold drag)."""
    return None if types is None else {**types.to_dict(), "applied": "per_type"}


# --- who gets removed: per cell type x condition ------------------------------------

AUDIT_LOSS = 0.30  # flag a cell type that loses more than this share
AUDIT_P = 0.01  # and a removal rate that depends on the condition at this p ...
AUDIT_DIFF_PP = 10.0  # ... with at least this many percentage points between conditions
AUDIT_MIN_CELLS = confounding.MIN_CELLS  # a condition is compared in a cell type with this many cells
MAX_AUDIT_CONDITIONS = 20  # more conditions (or samples) than this are not split by
MAX_AUDIT_GROUPS = 100  # label groups (rows); a column with more is summarised by a note
AUDIT_RULE = (f"Flagged when a filter removes more than {AUDIT_LOSS:.0%} of a group, or its removal "
              f"rate differs between conditions by {AUDIT_DIFF_PP:g} or more percentage points with "
              f"p < {AUDIT_P:g} (chi-square of removed vs kept by condition, Yates-corrected for two "
              f"conditions; conditions with fewer than {AUDIT_MIN_CELLS} cells in the group are left out).")

# Author labels: a cell-type / annotation column, or the author's clusters
# (paul15_clusters, seurat_clusters), never one scView made (it is remade after
# filtering, so it cannot label the removed cells).
_LABEL_RE = re.compile(r"cell[ ._-]?type|celltype|annotation|^labels?$|clusters?$", re.I)
_MADE_HERE_RE = re.compile(r"^scview_|leiden|louvain|^phase$|doublet", re.I)
MAX_LABEL_VALUES = MAX_AUDIT_GROUPS


def _few_values(s: pd.Series, most: int = MAX_LABEL_VALUES) -> bool:
    """A categorical, text, bool or integer column with 2..``most`` values."""
    if pd.api.types.is_float_dtype(s.dtype):
        return False
    return 2 <= int(s.nunique(dropna=True)) <= most


def _author_label(name: str) -> bool:
    return bool(_LABEL_RE.search(name)) and not _MADE_HERE_RE.search(name)


def label_grouping(adata, label_column: str | None = None) -> tuple[str | None, str | None]:
    """(column, why) for the audit's rows: the one named, else an author cell-type
    column, else the active clustering. Raises ValueError for a named column that
    does not exist."""
    obs = adata.obs
    if label_column:
        if label_column not in obs.columns:
            raise ValueError(f"Column '{label_column}' not found.")
        return label_column, "requested"
    for c in obs.columns:
        if _author_label(str(c)) and _few_values(obs[c]):
            return str(c), "labels"
    col = cluster_qc.default_grouping(adata)
    return (col, "clustering") if col else (None, None)


def condition_grouping(adata, condition_column: str | None = None, label_column: str | None = None,
                       sample_column: str | None = None) -> tuple[str | None, str | None]:
    """(column, why) for the audit's columns: the one named, else the first
    condition-like column (``confounding.condition_columns``), else the samples,
    else none. Raises ValueError for a named column that does not exist."""
    obs = adata.obs
    if condition_column:
        if condition_column not in obs.columns:
            raise ValueError(f"Column '{condition_column}' not found.")
        return condition_column, "requested"
    for c in confounding.condition_columns(obs):
        if c != label_column:
            return c, "condition"
    if sample_column and sample_column != label_column:
        return sample_column, "sample"
    return None, None


def _fmt_p(p: float) -> str:
    if p <= 0 or p < 1e-300:
        return "p < 1e-300"
    return f"p = {p:.0e}" if p < 1e-3 else f"p = {p:.2g}"


def rate_test(removed: np.ndarray, sizes: np.ndarray, names: list[str]) -> dict | None:
    """Removed ~ condition over the conditions with ``AUDIT_MIN_CELLS`` or more
    cells: chi-square on the 2 x k table (removed, kept), the highest and lowest
    removal rate and their difference in percentage points. None with fewer than
    two such conditions; ``p`` None when nothing or everything was removed."""
    use = np.flatnonzero(np.asarray(sizes) >= AUDIT_MIN_CELLS)
    if use.size < 2:
        return None
    r = np.asarray(removed, dtype=np.int64)[use]
    n = np.asarray(sizes, dtype=np.int64)[use]
    rates = r / n
    hi, lo = int(np.argmax(rates)), int(np.argmin(rates))
    out = {
        "compared": [names[i] for i in use],
        "high": names[use[hi]], "high_rate": round(float(rates[hi]), 4),
        "low": names[use[lo]], "low_rate": round(float(rates[lo]), 4),
        "max_diff_pp": round(100.0 * float(rates[hi] - rates[lo]), 1),
        "p": None, "statistic": None, "dof": None, "low_expected": False,
    }
    if 0 < int(r.sum()) < int(n.sum()):
        chi2, p, dof, expected = chi2_contingency(np.vstack([r, n - r]))  # Yates for 2 x 2
        out.update(p=float(p), statistic=round(float(chi2), 3), dof=int(dof),
                   low_expected=bool((expected < 5).any()))
    return out


def _depends(test: dict | None) -> bool:
    return bool(test and test["p"] is not None and test["p"] < AUDIT_P
                and test["max_diff_pp"] >= AUDIT_DIFF_PP)


def _main_criterion(by_criterion: dict[str, int]) -> tuple[str | None, int]:
    active = {c: v for c, v in by_criterion.items() if v > 0}
    return (max(active, key=active.get) if active else None), len(active)


def _cell_counts(codes: np.ndarray, k: int, drop: np.ndarray, removed: dict[str, np.ndarray],
                 where: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Per code (cells in ``where``): cells, removed, removed by each criterion alone."""
    c = codes[where]
    return (np.bincount(c, minlength=k), np.bincount(codes[where & drop], minlength=k),
            {name: np.bincount(codes[where & m], minlength=k) for name, m in removed.items()})


def _row(name: str | None, n: int, r: int, alone: dict[str, int], cond: dict | None) -> dict:
    n, r = int(n), int(r)
    row = {"name": name, "n_cells": n, "n_removed": r,
           "fraction": round(r / n, 4) if n else 0.0, "by_criterion": {c: int(v) for c, v in alone.items()}}
    if cond is not None:
        row["conditions"] = {"n_cells": [int(x) for x in cond["n"]], "n_removed": [int(x) for x in cond["r"]],
                             "by_criterion": {c: [int(x) for x in v] for c, v in cond["alone"].items()}}
        row["test"] = rate_test(cond["r"], cond["n"], cond["names"])
    return row


def audit_table(label: tuple[list[str], np.ndarray] | None, condition: tuple[list[str], np.ndarray] | None,
                drop: np.ndarray, removed: dict[str, np.ndarray], max_groups: int = MAX_AUDIT_GROUPS) -> dict:
    """The cross-tab on codes (-1 = no value): per label group, its loss overall,
    per condition and per criterion, and the removed ~ condition test; ``overall``
    the same over all cells, ignoring the labels."""
    n = int(drop.size)
    c_names, c_codes = condition if condition is not None else ([], np.full(n, -1))
    k_c = len(c_names)
    has_cond = k_c > 0
    out: dict = {"by_type": [], "note": None, "n_unlabelled": 0}

    def cond_cells(where: np.ndarray) -> dict | None:
        if not has_cond:
            return None
        cn, cr, ca = _cell_counts(c_codes, k_c, drop, removed, where & (c_codes >= 0))
        return {"n": cn, "r": cr, "alone": ca, "names": c_names}

    everyone = np.ones(n, dtype=bool)
    out["overall"] = _row(None, n, int(drop.sum()), {c: int(m.sum()) for c, m in removed.items()},
                          cond_cells(everyone))
    if label is None or not label[0]:
        return out
    l_names, l_codes = label
    k_l = len(l_names)
    labelled = l_codes >= 0
    out["n_unlabelled"] = int((~labelled).sum())
    if k_l > max_groups:
        out["note"] = f"{k_l} groups; the audit is shown for at most {max_groups}."
        return out
    sizes, lost, alone = _cell_counts(l_codes, k_l, drop, removed, labelled)
    per_cond = None
    if has_cond:
        both = labelled & (c_codes >= 0)
        flat = np.where(both, l_codes * k_c + np.maximum(c_codes, 0), -1)
        n_lc, r_lc, a_lc = _cell_counts(flat, k_l * k_c, drop, removed, both)
        per_cond = (n_lc.reshape(k_l, k_c), r_lc.reshape(k_l, k_c),
                    {c: v.reshape(k_l, k_c) for c, v in a_lc.items()})
    for i, name in enumerate(l_names):
        cond = None if per_cond is None else {
            "n": per_cond[0][i], "r": per_cond[1][i], "alone": {c: v[i] for c, v in per_cond[2].items()},
            "names": c_names}
        out["by_type"].append(_row(name, sizes[i], lost[i], {c: v[i] for c, v in alone.items()}, cond))
    return out


def _by(main: str | None, n_active: int) -> str:
    return ("" if main is None else f", by {CRITERIA[main]}" if n_active == 1
            else f", mostly by {CRITERIA[main]}")


def _rate_text(row: dict, test: dict, side: str) -> str:
    names = row["_cond_names"]
    i = names.index(test[side])
    r, n = row["conditions"]["n_removed"][i], row["conditions"]["n_cells"][i]
    return f"{test[side + '_rate']:.0%} of {test[side]} cells ({r:,} of {n:,})"


def audit_flags(audit: dict, skip: set[tuple[str, str]] = frozenset()) -> list[dict]:
    """Plain sentences with the numbers, shaped as the preview's warnings: a group
    losing more than ``AUDIT_LOSS`` (unless ``skip`` holds its (column, group): the
    preview already warns about it), and a removal rate that depends on the
    condition, per group and over all cells."""
    label_col = audit["label"]["column"]
    cond_col = audit["condition"]["column"] if audit.get("condition") else None
    cond_names = audit["condition"]["names"] if cond_col else []
    bare = bool(label_col) and _author_label(label_col)
    flags = []

    def who(name: str) -> str:
        return name if bare else f"{label_col} {name}"

    def base(kind: str, row: dict, criterion: str | None, message: str) -> dict:
        return {"kind": kind, "column": label_col if row["name"] is not None else cond_col,
                "group": row["name"], "n_cells": row["n_cells"], "n_removed": row["n_removed"],
                "fraction": row["fraction"], "whole": row["n_removed"] >= row["n_cells"] > 0,
                "criterion": criterion, "message": message}

    rows = [*audit["by_type"], audit["overall"]]
    for row in rows:
        main, n_active = _main_criterion(row["by_criterion"])
        n, r = row["n_cells"], row["n_removed"]
        if (row["name"] is not None and n and r / n > AUDIT_LOSS
                and (label_col, row["name"]) not in skip):
            what = (f"all {n:,} cells of {who(row['name'])}" if r >= n
                    else f"{r / n:.0%} of {who(row['name'])} ({r:,} of {n:,} cells)")
            flags.append(base("loss", row, main, f"Removes {what}{_by(main, n_active)}."))
        test = row.get("test")
        if not _depends(test):
            continue
        row = {**row, "_cond_names": cond_names}
        agent = ("the thresholds" if main is None else CRITERIA[main] if n_active == 1
                 else f"the thresholds (mostly {CRITERIA[main]})")
        where = "Over all cells" if row["name"] is None else f"In {who(row['name'])}"
        msg = (f"{where}, {agent} remove{'s' if n_active == 1 else ''} {_rate_text(row, test, 'high')} "
               f"but {_rate_text(row, test, 'low')} ({_fmt_p(test['p'])}): the loss depends on "
               f"'{cond_col}'.")
        flag = base("condition", row, main, msg)
        flag.update(condition=cond_col, p=test["p"], max_diff_pp=test["max_diff_pp"])
        flags.append(flag)
    flags.sort(key=lambda f: (f["kind"] != "condition", -f["fraction"]))
    return flags


def _codes_for(adata, column: str | None) -> tuple[list[str], np.ndarray] | None:
    return qc_groups.group_codes(adata.obs[column]) if column else None


def _assemble(label_col: str | None, label_reason: str | None, label: tuple[list[str], np.ndarray] | None,
              cond_col: str | None, cond_reason: str | None, cond: tuple[list[str], np.ndarray] | None,
              drop: np.ndarray, removed: dict[str, np.ndarray], skip: set[tuple[str, str]] = frozenset(),
              notes: list[str] | None = None) -> dict:
    notes = list(notes or [])
    if cond_col and cond_col == label_col:
        cond = None
        notes.append(f"'{cond_col}' is also the label column, so the cells are not split by it.")
    elif cond is not None and len(cond[0]) > MAX_AUDIT_CONDITIONS:
        notes.append(f"'{cond_col}' has {len(cond[0])} groups; the audit splits by at most "
                     f"{MAX_AUDIT_CONDITIONS}.")
        cond = None
    table = audit_table(label, cond, drop, removed)
    if table["note"]:
        notes.append(f"'{label_col}' has {table['note']}")
    out = {
        "label": {"column": label_col, "reason": label_reason, "n_groups": len(label[0]) if label else 0},
        "condition": None if cond is None else {"column": cond_col, "reason": cond_reason, "names": cond[0]},
        **table,
        "note": " ".join(notes) or None,
        "rule": AUDIT_RULE,
    }
    out["flags"] = audit_flags(out, skip)
    return out


def audit(adata, drop: np.ndarray, removed: dict[str, np.ndarray], label_column: str | None = None,
          condition_column: str | None = None, sample_column: str | None = None,
          skip: set[tuple[str, str]] = frozenset()) -> dict:
    """Who ``removed`` takes from ``adata``: per label group x condition (columns
    named, else ``label_grouping`` / ``condition_grouping``), with tests and flags."""
    l_col, l_reason = label_grouping(adata, label_column)
    c_col, c_reason = condition_grouping(adata, condition_column, l_col, sample_column)
    return _assemble(l_col, l_reason, _codes_for(adata, l_col), c_col, c_reason, _codes_for(adata, c_col),
                     drop, removed, skip)


# --- what the filtering step removed: a record in uns ------------------------------

RECORD_KEY = "scview_qc_removed"
RECORD_VERSION = 1
RECORD_MAX_CELLS = 200_000  # removed cells kept in the record (the counts say how many there were)
RECORD_MAX_COLUMNS = 8


def record_columns(adata, sample_column: str | None, type_column: str | None = None) -> list[str]:
    """The obs columns the record keeps for each removed cell: the samples, the
    condition-like columns, author label columns, then other categorical author
    columns, each with 2..100 values; never one scView remakes after filtering,
    except ``type_column``, the cell types a per-type % mito filter judged by."""
    obs = adata.obs
    out: list[str] = [type_column] if type_column and type_column in obs.columns else []

    def add(c, any_dtype: bool = True) -> None:
        name = str(c)
        if (name in obs.columns and name not in out and not _MADE_HERE_RE.search(name)
                and (any_dtype or isinstance(obs[c].dtype, pd.CategoricalDtype)
                     or obs[c].dtype == object or pd.api.types.is_bool_dtype(obs[c].dtype))
                and _few_values(obs[c])):
            out.append(name)

    if sample_column:
        add(sample_column)
    for c in confounding.condition_columns(obs):
        add(c)
    for c in obs.columns:
        if _author_label(str(c)):
            add(c)
    for c in obs.columns:
        add(c, any_dtype=False)
    return out[:RECORD_MAX_COLUMNS]


def removal_record(adata, cells: dict, removed: dict[str, np.ndarray], drop: np.ndarray, t: Thresholds,
                   sample_column: str | None, when: str | None = None,
                   types: TypeBounds | None = None) -> dict:
    """What the filtering step removes, for ``uns[RECORD_KEY]``: per removed cell
    (at most ``RECORD_MAX_CELLS``) its name, the criteria that removed it (bits in
    the order of ``criteria``), its QC values and its labels in ``record_columns``
    (codes into each column's categories). Arrays and strings only, so the h5ad
    stays writable (no None: "" and NaN stand for none). With per-type % mito
    bounds (``types``) also the type column and each type's bound (NaN: the global
    rule), and ``mito_type_floor`` when the floor was on; a global run's record is
    as it always was."""
    idx = np.flatnonzero(drop)
    crit = list(removed)
    keep_idx = idx[:RECORD_MAX_CELLS]
    bits = np.zeros(keep_idx.size, dtype=np.uint8)
    for b, c in enumerate(crit):
        bits |= np.asarray(removed[c])[keep_idx].astype(np.uint8) << b
    cols = record_columns(adata, sample_column, types.column if types is not None else None)
    columns = {}
    for j, c in enumerate(cols):
        names, codes = qc_groups.group_codes(adata.obs[c])
        columns[f"c{j}"] = {"name": c, "categories": np.array(names, dtype=object),
                            "codes": codes[keep_idx].astype(np.int16)}
    has_mt = _mito_measurable(cells)
    conds = set(confounding.condition_columns(adata.obs))
    pct = _values(cells, "pct_mt")[keep_idx] if has_mt else np.full(keep_idx.size, np.nan)
    extra = {} if types is None else {
        "mito_mode": "per_type",
        "mito_type_column": types.column,
        "mito_type_names": np.array([r["name"] for r in types.rows], dtype=object),
        "mito_type_bounds": np.array([np.nan if r["bound"] is None else r["bound"] for r in types.rows],
                                     dtype=np.float64),
        "mito_type_k": float(types.k),
        **({"mito_type_floor": True} if t.mito_type_floor else {}),  # absent: off, as before
    }
    return {
        **extra,
        "version": RECORD_VERSION,
        "timestamp": when or datetime.now(timezone.utc).isoformat(),
        "mode": t.mode,
        "n_before": int(drop.size),
        "n_removed": int(idx.size),
        "n_kept": int(drop.size - idx.size),
        "truncated": bool(idx.size > keep_idx.size),
        "criteria": np.array(crit, dtype=object),
        "obs_names": np.asarray(adata.obs_names[keep_idx], dtype=object),
        "criterion_bits": bits,
        "total_counts": _values(cells, "total_counts")[keep_idx].astype(np.float32),
        "n_genes": _values(cells, "n_genes")[keep_idx].astype(np.float32),
        "has_mt": bool(has_mt),
        "pct_mt": pct.astype(np.float32),
        "sample_column": sample_column or "",
        "condition_columns": np.array([c for c in cols if c in conds], dtype=object),
        # The types a per-type filter judged by come first (even a scView clustering).
        "label_columns": np.array([c for c in cols if _author_label(c) or (types is not None and c == types.column)],
                                  dtype=object),
        "columns": columns,
    }


def _strs(v) -> list[str]:
    return [str(x) for x in np.atleast_1d(np.asarray(v, dtype=object))] if v is not None else []


def _median(v: np.ndarray) -> float | None:
    v = v[np.isfinite(v)]
    return round(float(np.median(v)), 2) if v.size else None


def recorded_audit(adata, label_column: str | None = None, condition_column: str | None = None) -> dict | None:
    """The audit of the cells the last filtering step removed (``uns[RECORD_KEY]``):
    the removed cells from the record, the kept cells from ``adata`` as it is now.
    None for a dataset without a record (filtered before the record existed, or
    never); ``stale`` when cells were added or removed since, so the two no longer
    add up. Columns not recorded fall back to the record's own defaults."""
    rec = adata.uns.get(RECORD_KEY)
    if not isinstance(rec, dict):
        return None
    try:
        return _recorded_audit(adata, rec, label_column, condition_column)
    except Exception as e:  # a malformed record never breaks the preview
        logger.warning("QC removal record unreadable: %s", e)
        return None


def recorded_mito_rule(rec: dict) -> dict:
    """How a removal record's run judged % mito: ``mito_mode`` "global", or
    "per_type" with the type column and each type's bound (None: the global rule),
    and ``mito_type_floor`` True when it was never stricter than the global rule."""
    if str(rec.get("mito_mode") or "global") != "per_type":
        return {"mito_mode": "global"}
    names = _strs(rec.get("mito_type_names"))
    bounds = np.atleast_1d(np.asarray(rec.get("mito_type_bounds", []), dtype=np.float64))
    return {"mito_mode": "per_type", "mito_type_column": str(rec.get("mito_type_column") or ""),
            "mito_type_bounds": {n: (None if not np.isfinite(b) else round(float(b), 4))
                                 for n, b in zip(names, bounds)},
            **({"mito_type_floor": True} if bool(rec.get("mito_type_floor", False)) else {})}


def recorded_removed_mito(adata, column: str | None) -> tuple[list[str], np.ndarray, str] | None:
    """The cells the last filtering step removed, from its record: each one's
    ``column`` label ("" when it had none) and % mito, and the record's timestamp.
    None unless the record still adds up (not stale or truncated), measured % mito
    and recorded ``column``. ``column`` None: no labels (all ""), only whether the
    record is usable and its cells' % mito."""
    rec = adata.uns.get(RECORD_KEY)
    if not isinstance(rec, dict) or not bool(rec.get("has_mt", False)):
        return None
    try:
        if int(rec["n_kept"]) != int(adata.n_obs) or bool(rec.get("truncated", False)):
            return None
        if column is None:
            pct = np.asarray(rec["pct_mt"], dtype=np.float64)
            return [""] * pct.size, pct, str(rec.get("timestamp") or "")
        for entry in (rec.get("columns") or {}).values():
            if str(entry["name"]) != column:
                continue
            cats = _strs(entry["categories"])
            codes = np.asarray(entry["codes"], dtype=np.int64)
            labels = [cats[c] if 0 <= c < len(cats) else "" for c in codes]
            return labels, np.asarray(rec["pct_mt"], dtype=np.float64), str(rec.get("timestamp") or "")
    except Exception as e:  # a malformed record only loses the addition
        logger.warning("QC removal record unreadable: %s", e)
    return None


def recorded_names(adata) -> list[str] | None:
    """The names of the cells the last filtering step removed, from its record, in
    its order; None unless the record still adds up (as ``recorded_removed_mito``)."""
    rec = adata.uns.get(RECORD_KEY)
    if not isinstance(rec, dict):
        return None
    try:
        if int(rec["n_kept"]) != int(adata.n_obs) or bool(rec.get("truncated", False)):
            return None
        return _strs(rec.get("obs_names"))
    except Exception as e:  # a malformed record only loses the addition
        logger.warning("QC removal record unreadable: %s", e)
        return None


def recorded_mito_bits(adata) -> np.ndarray | None:
    """Per cell of the removal record, in its order: removed by the % mito rule
    (its ``max_pct_mt`` criterion bit). None without a usable record; all False when
    % mito was not a criterion."""
    return recorded_bits(adata, "max_pct_mt")


def recorded_bits(adata, criterion: str) -> np.ndarray | None:
    """Per cell of the removal record, in its order: removed by ``criterion`` (one
    of ``CRITERIA``). None without a record; all False when it was not a criterion."""
    rec = adata.uns.get(RECORD_KEY)
    if not isinstance(rec, dict):
        return None
    try:
        bits = np.asarray(rec.get("criterion_bits"), dtype=np.uint8)
        crit = _strs(rec.get("criteria"))
        if criterion not in crit:
            return np.zeros(bits.size, dtype=bool)
        return (bits >> crit.index(criterion)) & 1 > 0
    except Exception as e:
        logger.warning("QC removal record unreadable: %s", e)
        return None


def _recorded_audit(adata, rec: dict, label_column: str | None, condition_column: str | None) -> dict:
    head = {"timestamp": str(rec.get("timestamp") or ""), "mode": str(rec.get("mode") or ""),
            "n_before": int(rec["n_before"]), "n_removed": int(rec["n_removed"]),
            "truncated": bool(rec.get("truncated", False)), "has_mt": bool(rec.get("has_mt", False)),
            **recorded_mito_rule(rec)}
    n_kept, rec_kept = int(adata.n_obs), int(rec["n_kept"])
    if n_kept != rec_kept:
        return {**head, "stale": True, "columns": [],
                "note": (f"The dataset has {n_kept:,} cells now, but {rec_kept:,} were kept when it was "
                         "filtered: cells were added or removed since, so the record no longer adds up.")}
    cols: dict[str, tuple[list[str], np.ndarray]] = {}
    for entry in (rec.get("columns") or {}).values():
        name = str(entry["name"])
        if name in adata.obs.columns:  # the kept cells' values come from obs
            cols[name] = (_strs(entry["categories"]), np.asarray(entry["codes"], dtype=np.int64))
    labels = [c for c in _strs(rec.get("label_columns")) if c in cols]
    conds = [c for c in _strs(rec.get("condition_columns")) if c in cols]
    sample = str(rec.get("sample_column") or "")
    sample = sample if sample in cols else None
    notes: list[str] = []

    def pick(requested: str | None, defaults: list[tuple[str | None, str]]) -> tuple[str | None, str | None]:
        if requested:
            if requested in cols:
                return requested, "requested"
            notes.append(f"'{requested}' was not recorded when the cells were filtered.")
        return next(((c, r) for c, r in defaults if c), (None, None))

    l_col, l_reason = pick(label_column, [(labels[0] if labels else None, "labels")])
    c_col, c_reason = pick(condition_column, [(next((c for c in conds if c != l_col), None), "condition"),
                                              (sample if sample != l_col else None, "sample")])
    m = len(_strs(rec.get("obs_names")))

    def combined(col: str | None) -> tuple[list[str], np.ndarray] | None:
        if not col:
            return None
        names, codes = qc_groups.group_codes(adata.obs[col])
        rec_names, rec_codes = cols[col]
        seen = set(names)
        names = [*names, *(x for x in rec_names if x not in seen)]
        pos = {x: i for i, x in enumerate(names)}
        remap = np.array([pos[x] for x in rec_names] + [-1])  # code -1 reads the last slot
        return names, np.concatenate([codes, remap[rec_codes]])

    drop = np.concatenate([np.zeros(n_kept, dtype=bool), np.ones(m, dtype=bool)])
    bits = np.asarray(rec.get("criterion_bits"), dtype=np.uint8)
    removed = {c: np.concatenate([np.zeros(n_kept, dtype=bool), (bits >> b) & 1 > 0])
               for b, c in enumerate(_strs(rec.get("criteria")))}
    if head["truncated"]:
        notes.append(f"Only the first {m:,} of the {head['n_removed']:,} removed cells were recorded, "
                     "so the losses below are too low.")
    label = combined(l_col)
    out = _assemble(l_col, l_reason, label, c_col, c_reason, combined(c_col), drop, removed, notes=notes)
    if label is not None and out["by_type"]:  # the removed cells' own QC, per group
        rec_codes = label[1][n_kept:]
        vals = {k: np.asarray(rec[k], dtype=np.float64) for k in ("total_counts", "n_genes", "pct_mt")}
        for i, row in enumerate(out["by_type"]):
            sel = rec_codes == i
            row["removed_median"] = {k: _median(v[sel]) for k, v in vals.items()
                                     if k != "pct_mt" or head["has_mt"]} if sel.any() else None
    return {**head, "stale": False, "columns": list(cols), **out}


# --- doublet calls: whose, and the hashing negatives (core/doublet_source.py) ----------


@dataclass(frozen=True)
class CellCalls:
    """Per-cell calls the filtering step can drop beside the thresholds: the
    doublets of the chosen source, and cell hashing's negatives."""

    doublets: np.ndarray | None
    negatives: np.ndarray | None = None
    source: str | None = None  # the source the doublets came from (doublet_source.Calls)


def cell_calls(obs: pd.DataFrame, t: Thresholds) -> CellCalls:
    """The doublets ``t.doublet_source`` stands for ("auto": cell hashing when it was
    run, else Scrublet, else a text column) and the hashing negatives."""
    used = doublet_source.resolve(obs, t.doublet_source)
    return CellCalls(None if used is None else used.calls, doublet_source.hash_negatives(obs),
                     None if used is None else used.source)


def _call_masks(t: Thresholds, calls) -> dict[str, np.ndarray]:
    """``removed_by``'s doublet / negative criteria from an array of doublet calls
    (as before the source choice) or a ``CellCalls``."""
    if calls is None:
        return {}
    if not isinstance(calls, CellCalls):
        calls = CellCalls(np.asarray(calls, dtype=bool))
    out: dict[str, np.ndarray] = {}
    if t.drop_doublets and calls.doublets is not None:
        out["doublets"] = np.asarray(calls.doublets, dtype=bool)
    if t.drop_hash_negatives and calls.negatives is not None:
        out["hash_negatives"] = np.asarray(calls.negatives, dtype=bool)
    return out


def doublet_choices(obs: pd.DataFrame, t: Thresholds) -> dict | None:
    """For the preview's doublet control: the sources that mean something here, the
    one ``t.doublet_source`` stands for and what auto picks, each source's count,
    how many cells hashing and Scrublet both call, the hashing negatives, and the
    cells the chosen calls leave unscored (Scrublet's small channels: kept, not called).
    None without any doublet calls."""
    auto = doublet_source.describe(obs)
    if auto is None:
        return None
    used = doublet_source.resolve(obs, t.doublet_source)
    unscored = None if used is None or used.unscored is None else int(used.unscored.sum())
    n_by: dict[str, int | None] = {}
    for s in auto["options"]:
        c = doublet_source.resolve(obs, s)
        n_by[s] = None if c is None else c.n
    neg = doublet_source.hash_negatives(obs)
    return {
        "source": t.doublet_source,
        "used": None if used is None else {"source": used.source, "label": used.label, "n": used.n},
        "auto": {"source": auto["source"], "label": auto["label"]},
        "options": auto["options"],
        "n_by_source": n_by,
        "both": auto["both"]["n"] if auto.get("both") else None,
        "n_cells": auto["n_cells"],
        "hash_negatives": None if neg is None else int(neg.sum()),
        "drop_hash_negatives": t.drop_hash_negatives,
        "negative_note": doublet_source.NEGATIVE_NOTE,
        "unscored": unscored,
    }


def after_filter(adata, t: Thresholds, n_removed: int) -> dict:
    """The filtering step's own record beside its params, once the cells are gone
    (``adata`` = the cells kept): whose doublets ``drop_doublets`` dropped (the
    choice, and the source it stood for, which the replay uses), the hashing
    negatives choice, and the V(D)J clone sizes recomputed on the cells kept
    (``vdj.recompute_after_filter``) when cells were removed. A record without
    ``qc_doublet_source`` is from before the choice: its replay keeps Scrublet first."""
    from scview.core import vdj

    out: dict = {}
    if t.drop_doublets:
        used = doublet_source.resolve(adata.obs, t.doublet_source)
        out.update(qc_doublet_source=t.doublet_source,
                   qc_doublet_source_used=None if used is None else used.source)
        if used is not None and used.unscored is not None and used.unscored.any():
            # Cells Scrublet did not score (a channel too small for it) are kept: no call.
            out["doublets_unscored_kept"] = int(used.unscored.sum())
    if t.drop_hash_negatives:
        out["drop_hash_negatives"] = True
    if n_removed:
        clones = vdj.recompute_after_filter(adata)
        if clones is not None:
            out["vdj_clonotypes"] = clones
    return out


# --- the co-pilot's fact --------------------------------------------------------

GROUNDING_TAG = "result:qc:removal"
GROUNDING_FLAGS = 4  # flag sentences in the fact; the rest are counted


def grounding_lines(adata, cells: dict, t: Thresholds) -> tuple[list[str], str]:
    """The co-pilot's fact about who the QC removes, so "did the QC remove a cell
    type?" is answered from the numbers: what the last filtering step removed
    (its record), else what ``t`` (the pipeline's defaults) would remove, with the
    audit's flags. ``(lines, detail)``; no lines without a label column."""
    rec = recorded_audit(adata)
    if rec is not None and not rec["stale"]:
        au = rec
        when = rec["timestamp"][:10]
        what = (f"the filtering step{f' on {when}' if when else ''} removed {rec['n_removed']:,} "
                f"of {rec['n_before']:,} cells")
    else:
        au = preview(adata, cells, t)["audit"]
        what = (f"the default thresholds ({'per-sample MAD' if t.mode == 'mad' else 'fixed cut-offs'}) "
                f"would remove {au['overall']['n_removed']:,} of {au['overall']['n_cells']:,} cells "
                "(not applied yet)")
    label = au["label"]["column"]
    if not label:
        return [], ""
    cond = au["condition"]["column"] if au.get("condition") else None
    split = f"per '{label}' group" + (f" x '{cond}'" if cond else "")
    flags = au["flags"]
    if flags:
        found = " ".join(f["message"] for f in flags[:GROUNDING_FLAGS])
        if len(flags) > GROUNDING_FLAGS:
            found += f" ({len(flags) - GROUNDING_FLAGS} more flags.)"
    else:
        found = (f"No '{label}' group loses more than {AUDIT_LOSS:.0%}"
                 + (f", and no group's removal rate differs between '{cond}' groups by "
                    f"{AUDIT_DIFF_PP:g}+ points with p < {AUDIT_P:g} (chi-square)." if cond
                    else " (no condition or sample column to test the removal against)."))
    lines = ["\n## QC removal audit (deterministic: who the cell QC removes)",
             f"- {what[0].upper()}{what[1:]}, {split}: {found} [{GROUNDING_TAG}]"]
    detail = f"{what}; {len(flags)} flag(s) {split}"
    return lines, detail
