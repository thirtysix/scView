"""Cell QC thresholds: which cells the filtering step removes, before it runs.

The QC histograms on the Data Assessment tab carry the filtering step's cell
thresholds (min / max genes, min counts, max % mito), and ``preview`` says what
they remove overall, per sample and per cluster, with a warning when one empties
a sample or a cluster or takes more than half of one: a real cell type with few
genes or much mitochondrial RNA (platelets, neutrophils, cardiomyocytes) is lost
that way without anyone noticing.

Two modes. ``fixed``: the cut-offs above, one set for every cell. ``mad`` (the
default for new runs): each sample's own outliers, as the single-cell best
practices book has them (Heumos et al. 2023): cells outside median +/- 5 MADs of
log1p(counts) and of log1p(genes) per cell in their sample, and above median +
3 MADs of % mito. The MAD is the raw median absolute deviation (scipy's default
scale), as in the book. In MAD mode the fixed genes / counts cut-offs are not
used; "max % mito" stays on as a ceiling over the per-sample bound, because a
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
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from scview.core import cluster_qc, provenance, qc_groups

HALF = 0.5  # warn when a threshold removes more than this share of a group
MAX_CLUSTERS = cluster_qc.MAX_GROUPS

MODES = ("mad", "fixed")
MAD_K = 5.0  # counts and genes: outside median +/- this many MADs (log1p values)
MAD_K_MT = 3.0  # % mito: above median + this many MADs
MIN_SAMPLE_CELLS = 100  # fewer cells: the pooled median and MAD (module docstring)
MAX_SAMPLES = cluster_qc.MAX_GROUPS  # per-sample thresholds for at most this many samples

# criterion -> what the user calls it (the threshold's label on the plots)
CRITERIA = {
    "min_genes": "Min genes",
    "max_genes": "Max genes",
    "min_counts": "Min counts",
    "max_counts": "Max counts",
    "max_pct_mt": "Max % mito",
    "doublets": "Predicted doublets",
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
        )

    def check(self) -> None:
        """Raise ValueError for an unknown mode or a MAD multiplier that is not positive."""
        if self.mode not in MODES:
            raise ValueError(f"Unknown QC threshold mode '{self.mode}': use 'mad' or 'fixed'.")
        if self.mode == "mad":
            for name, k in (("MAD multiplier", self.mad_k), ("% mito MAD multiplier", self.mad_k_mt)):
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
    metric (log1p for counts and genes); None where a MAD of 0 sets none."""
    skipped = [m for m, (_, mad) in stats.items() if not (mad > 0)]  # 0, or no values
    out: dict = {}
    for key, below, above in _MAD_METRICS:
        med, mad = stats[key]
        on = key not in skipped
        out[below] = max(0.0, math.expm1(med - t.mad_k * mad)) if on else None
        out[above] = math.expm1(med + t.mad_k * mad) if on else None
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
    same = (rec.get("k"), rec.get("k_mt"), rec.get("ceiling_pct_mt"), rec.get("column"))
    return rec if same == (t.mad_k, t.mad_k_mt, ceiling, column) else None


def mad_bounds(cells: dict, t: Thresholds, samples: Samples | None = None,
               recorded: dict | None = None) -> MadBounds:
    """Per-sample bounds: median +/- ``t.mad_k`` MADs of log1p(counts) and of
    log1p(genes), median + ``t.mad_k_mt`` MADs of % mito (capped by
    ``t.max_pct_mt``). Small samples and unlabelled cells take the pooled bounds;
    a MAD of 0 sets no bound (that metric is listed in ``skipped``).

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


def removed_by(cells: dict, t: Thresholds, doublets: np.ndarray | None = None,
               bounds: MadBounds | None = None) -> dict[str, np.ndarray]:
    """For each threshold that is on, the cells it removes on its own (boolean masks).

    ``cells`` is ``per_cell_qc``'s output; ``doublets`` the per-cell doublet calls
    (``cluster_qc.doublet_calls``), used when ``t.drop_doublets``. In MAD mode
    ``bounds`` are the samples' (``mad_bounds``); without them all cells pool."""
    if t.mode == "mad":
        out = _removed_by_mad(cells, t, bounds if bounds is not None else mad_bounds(cells, t))
        if t.drop_doublets and doublets is not None:
            out["doublets"] = np.asarray(doublets, dtype=bool)
        return out
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


def _removed_by_mad(cells: dict, t: Thresholds, bounds: MadBounds) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    with np.errstate(invalid="ignore"):
        for key, below, above in _MAD_METRICS:
            v = np.log1p(_values(cells, key))
            ok = np.isfinite(v)
            out[below] = ok & (v < bounds.lo[key])  # strictly outside, as the book has it
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
    masks per criterion and, in MAD mode, the samples' bounds. The preview and the
    filtering step both call this, so the preview is what the step does.

    ``samples`` defaults to ``samples_of(adata)``. Raises ValueError for a bad
    mode or multiplier, or too many samples."""
    t.check()
    calls = cluster_qc.doublet_calls(adata.obs)
    if t.mode != "mad":
        return removed_by(cells, t, calls), None
    bounds = mad_bounds(cells, t, samples if samples is not None else samples_of(adata),
                        recorded_filtering(adata))
    return removed_by(cells, t, calls, bounds), bounds


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
            cluster_column: str | None = None) -> dict:
    """What ``t`` removes: overall, per threshold on its own, per sample (the
    donor / sample column, else a batch column) and per cluster (the active
    clustering), plus warnings for groups it empties or more than halves.

    Each sample also carries its QC medians and range flags
    (``qc_groups.sample_summary``); in MAD mode ``mad`` holds each sample's
    thresholds (``MadBounds.to_dict``), the samples being the same groups.
    ``previous`` is how scView last filtered the dataset, if it did.

    ``cells`` is ``AnnDataAdaptor.qc_per_cell()``. Raises ValueError for a named
    column that does not exist, or for thresholds ``judge`` refuses."""
    n = int(adata.n_obs)
    samp = samples_of(adata, sample_column)
    removed, bounds = judge(adata, cells, t, samp)
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
        },
        "has_mt": has_mt,
        "doublet_calls": cluster_qc.doublet_calls(adata.obs) is not None,  # "drop doublets" can act
        "by_criterion": {c: {"label": CRITERIA[c], "n_removed": int(m.sum())} for c, m in removed.items()},
        "mad": bounds.to_dict(drop) if bounds is not None else None,
        "samples": samples,
        "clusters": clusters,
        "warnings": warnings,
        "previous": None if previous is None else {
            "mode": previous["qc_threshold_mode"], "timestamp": previous.get("timestamp")},
    }
