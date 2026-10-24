"""Sequencing QC from Cell Ranger outputs: ``metrics_summary.csv`` and the raw matrix.

A 10x run's ``metrics_summary.csv`` says whether the library was sequenced deeply
enough and mapped well, and the raw (unfiltered) barcode matrix shows how cells were
called. Users upload the filtered matrix and rarely look at either, so the ingest
flow attaches them as side files of a sample (``core/ingestion/bundling.py``), and
``POST /datasets/{id}/seq-metrics`` attaches them to a dataset uploaded earlier.
One record per sample is kept as a JSON string in ``uns["scview_seq_metrics"]``; the
raw matrix's empty droplets also give the sample's ambient RNA profile, kept beside it in
``uns["scview_ambient_soup"]`` (``core/ambient.py``).

Metrics: both layouts Cell Ranger writes are read. ``cellranger count`` writes one
header row of metric names and one row of values ("1,222", "97.4%");
``cellranger multi`` writes one metric per row (Category, Library Type, Grouped By,
Group Name, Metric Name, Metric Value), and Cell Ranger 10 writes its fractions there
as plain decimals (0.976) rather than percentages. Only Gene Expression rows are read.

Flags are Cell Ranger's own web-summary alerts (10.1.0), with their thresholds:
count ``lib/python/cellranger/websummary/metrics.csv`` (column "targeted" = warning
line, "acceptable" = error line),
https://github.com/10XGenomics/cellranger/blob/cellranger-10.1.0/lib/python/cellranger/websummary/metrics.csv
and multi ``lib/rust/cr_websummary/src/multi/metrics_etl.toml`` (+ ``metrics_etl_special.toml``),
https://github.com/10XGenomics/cellranger/blob/cellranger-10.1.0/lib/rust/cr_websummary/src/multi/metrics_etl.toml
Sequencing depth is not an alert there: the depth note uses 10x's sequencing
requirements (minimum 20,000 read pairs per cell for 3' Gene Expression; Flex 10,000),
https://www.10xgenomics.com/support/single-cell-gene-expression/documentation/steps/sequencing/sequencing-requirements-for-single-cell-3

Barcode ranks: per-barcode UMI totals (Gene Expression features only) sorted into the
log-log rank curve, with the knee and inflection points of DropletUtils
``barcodeRanks`` (the curve-tracing version, DropletUtils >= 1.30, commit 8f18cce,
August 2025: a window of length 1 in log10 units is traced along the curve; the knee
is the midpoint of the window above its chord with the shortest chord, the
inflection the midpoint of the window with the steepest chord),
https://github.com/MarioniLab/DropletUtils/blob/devel/R/barcodeRanks.R
The raw matrix (6.8 million barcodes for 3' v3) is read in blocks of matrix entries,
never densified, and only barcodes with at least one UMI are kept.
"""

from __future__ import annotations

import csv
import gzip
import io
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import pandas as pd

from scview.core import ambient
from scview.core.ingestion.detection import looks_like_metrics

logger = logging.getLogger(__name__)

UNS_KEY = "scview_seq_metrics"
TAG = "result:seqqc"
REPORT_TAG = "qc:seqqc"
VERSION = 1

MAX_CURVE_POINTS = 2000
_CURVE_BINS = MAX_CURVE_POINTS // 2 - 2  # one cell + one other barcode per bin, + the last
_NNZ_CHUNK = 8_000_000  # matrix entries read at a time from a raw matrix
_BC_CHUNK = 1_000_000  # barcodes read at a time
_MEX_ROWS = 1_000_000  # Matrix Market lines parsed at a time

# barcodeRanks defaults (DropletUtils).
BR_LOWER = 100
BR_EXCLUDE_FROM = 50
BR_WINDOW = 1.0
BR_GRADIENT_THRESHOLD = -1.0

CELLRANGER_COUNT_URL = (
    "https://github.com/10XGenomics/cellranger/blob/cellranger-10.1.0/"
    "lib/python/cellranger/websummary/metrics.csv"
)
CELLRANGER_MULTI_URL = (
    "https://github.com/10XGenomics/cellranger/blob/cellranger-10.1.0/"
    "lib/rust/cr_websummary/src/multi/metrics_etl.toml"
)
DEPTH_URL = (
    "https://www.10xgenomics.com/support/single-cell-gene-expression/documentation/"
    "steps/sequencing/sequencing-requirements-for-single-cell-3"
)
FLEX_DEPTH_URL = (
    "https://www.10xgenomics.com/support/flex-gene-expression/documentation/steps/sequencing/"
    "sequencing-requirements-for-single-cell-gene-expression-flex"
)
DROPLETUTILS_URL = "https://github.com/MarioniLab/DropletUtils/blob/devel/R/barcodeRanks.R"
_ALERT_SOURCE = "Cell Ranger 10.1 web summary alert"


class SeqQcError(ValueError):
    """A request that can't be answered, as a sentence for the user."""

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Metric:
    key: str
    label: str
    kind: str  # "int" | "frac"
    count: tuple[str, ...]  # cellranger count column names (lower case)
    multi: tuple[tuple[str, str], ...]  # (category, metric name) in order of preference
    tip: str


METRICS: tuple[Metric, ...] = (
    Metric("estimated_cells", "Estimated cells", "int", ("estimated number of cells",),
           (("cells", "cells"), ("library", "estimated number of cells"), ("library", "cells")),
           "Barcodes Cell Ranger called as cells (the filtered matrix)."),
    Metric("mean_reads_per_cell", "Mean reads per cell", "int", ("mean reads per cell",),
           (("cells", "mean reads per cell"), ("library", "mean reads per cell")),
           "All sequenced reads divided by the estimated cells, reads outside cells included."),
    Metric("median_genes_per_cell", "Median genes per cell", "int", ("median genes per cell",),
           (("cells", "median genes per cell"),),
           "Median number of genes detected (at least one UMI) per cell."),
    Metric("median_umi_per_cell", "Median UMIs per cell", "int", ("median umi counts per cell",),
           (("cells", "median umi counts per cell"),),
           "Median UMI count per cell."),
    Metric("total_genes_detected", "Total genes detected", "int", ("total genes detected",),
           (("cells", "total genes detected"),),
           "Genes with at least one UMI in any cell."),
    Metric("n_reads", "Number of reads", "int", ("number of reads",),
           (("library", "number of reads"), ("library", "number of reads in the library")),
           "Read pairs sequenced for the library."),
    Metric("valid_barcodes", "Valid barcodes", "frac", ("valid barcodes",),
           (("library", "valid barcodes"), ("library", "valid gem barcodes")),
           "Share of reads whose cell barcode matches the whitelist (after correction)."),
    Metric("valid_umis", "Valid UMIs", "frac", ("valid umis", "valid umi sequences"),
           (("library", "valid umis"), ("library", "valid umi sequences")),
           "Share of reads with a valid UMI sequence (no N, no homopolymer)."),
    Metric("sequencing_saturation", "Sequencing saturation", "frac", ("sequencing saturation",),
           (("library", "sequencing saturation"),),
           "Share of reads that are duplicates of an already-seen UMI: 1 - unique UMIs / "
           "reads. High = deeper sequencing would find few new molecules."),
    Metric("q30_barcode", "Q30 barcode", "frac", ("q30 bases in barcode",),
           (("library", "q30 barcodes"), ("library", "q30 gem barcodes")),
           "Share of cell-barcode bases with a Phred quality of 30 or more."),
    Metric("q30_rna_read", "Q30 RNA read", "frac", ("q30 bases in rna read",),
           (("library", "q30 rna read"),),
           "Share of RNA-read bases with a Phred quality of 30 or more."),
    Metric("q30_umi", "Q30 UMI", "frac", ("q30 bases in umi",),
           (("library", "q30 umi"),),
           "Share of UMI bases with a Phred quality of 30 or more."),
    Metric("mapped_genome", "Mapped to genome", "frac", ("reads mapped to genome",),
           (("library", "mapped to genome"), ("cells", "mapped to genome")),
           "Share of reads that map to the genome."),
    Metric("conf_genome", "Confidently mapped to genome", "frac",
           ("reads mapped confidently to genome",),
           (("library", "confidently mapped to genome"), ("cells", "confidently mapped to genome")),
           "Share of reads that map uniquely to the genome."),
    Metric("conf_exonic", "Exonic", "frac", ("reads mapped confidently to exonic regions",),
           (("library", "confidently mapped to exonic regions"),
            ("cells", "confidently mapped to exonic regions")),
           "Share of reads that map uniquely to an exon."),
    Metric("conf_intronic", "Intronic", "frac", ("reads mapped confidently to intronic regions",),
           (("library", "confidently mapped to intronic regions"),
            ("cells", "confidently mapped to intronic regions")),
           "Share of reads that map uniquely to an intron (high in single-nucleus data)."),
    Metric("conf_intergenic", "Intergenic", "frac",
           ("reads mapped confidently to intergenic regions",),
           (("library", "confidently mapped to intergenic regions"),
            ("cells", "confidently mapped to intergenic regions")),
           "Share of reads that map uniquely outside genes."),
    Metric("conf_transcriptome", "Mapped to transcriptome", "frac",
           ("reads mapped confidently to transcriptome",),
           (("library", "confidently mapped to transcriptome"),
            ("cells", "confidently mapped to transcriptome")),
           "Share of reads that map uniquely to a gene (the reads that are counted as UMIs)."),
    Metric("conf_probe_set", "Mapped to probe set", "frac", (),
           (("library", "reads confidently mapped to probe set"),
            ("cells", "reads confidently mapped to probe set")),
           "Flex: share of reads that map uniquely to a probe of the probe set."),
    Metric("antisense", "Antisense", "frac", ("reads mapped antisense to gene",),
           (("library", "confidently mapped antisense"), ("cells", "confidently mapped antisense")),
           "Share of reads that map to the opposite strand of a gene."),
    Metric("frac_reads_in_cells", "Reads in cells", "frac", ("fraction reads in cells",),
           (("cells", "confidently mapped reads in cells"),
            ("library", "confidently mapped reads in cells"),
            ("library", "fraction reads in cells"), ("cells", "fraction reads in cells")),
           "Share of confidently mapped, valid-barcode reads that come from cell barcodes. "
           "Low = much ambient RNA, or cells the caller missed."),
)
_BY_KEY = {m.key: m for m in METRICS}


@dataclass(frozen=True)
class Alert:
    key: str
    direction: str  # "below" | "above": the value is flagged when it is on this side
    warn: float
    error: float | None
    title: str
    ideal: str
    meaning: str
    source: str = _ALERT_SOURCE
    url: str = CELLRANGER_COUNT_URL
    level: str = "warn"  # crossing ``warn``: "warn", or "info" for a recommendation


ALERTS: tuple[Alert, ...] = (
    Alert("estimated_cells", "below", 100, 1, "Low number of cells detected", "Ideal > 100",
          "This usually indicates poor cell handling, poor library quality or poor "
          "sequencing quality."),
    Alert("valid_barcodes", "below", 0.75, 0.5, "Low fraction valid barcodes", "Ideal > 75%",
          "This may indicate a quality issue with the R1 read."),
    Alert("valid_umis", "below", 0.75, 0.5, "Low fraction valid UMI sequences", "Ideal > 75%",
          "This may indicate a quality issue with the R1 read."),
    Alert("conf_transcriptome", "below", 0.30, 0.20,
          "Low fraction reads confidently mapped to transcriptome", "Ideal > 30%",
          "This can indicate the wrong reference transcriptome, a reference with overlapping "
          "genes, poor library or sequencing quality, or reads shorter than the recommended "
          "minimum."),
    Alert("conf_probe_set", "below", 0.5, 0.2,
          "Low fraction reads confidently mapped to probe set", "Ideal > 50%",
          "This can indicate low total expression, the wrong probe set, suboptimal sample "
          "preparation, or FASTQs from a product other than Flex.", url=CELLRANGER_MULTI_URL),
    Alert("antisense", "above", 0.10, 0.30, "High fraction of reads mapped antisense to genes",
          "Ideal < 10%",
          "This can indicate an unsupported chemistry (for example V(D)J libraries counted as "
          "gene expression). Single-nucleus samples counted with introns run higher: 10x's "
          "multi summary then warns only above 20% (error above 40%)."),
    Alert("q30_barcode", "below", 0.55, 0.45, "Low barcode Q30 fraction", "Ideal > 55%",
          "Low base quality in the cell barcode (R1); application performance may be affected."),
    Alert("q30_rna_read", "below", 0.65, 0.55, "Low RNA read Q30 fraction", "Ideal > 65%",
          "Low base quality in the RNA read; application performance may be affected."),
    Alert("q30_umi", "below", 0.75, 0.65, "Low UMI Q30 fraction", "Ideal > 75%",
          "Low base quality in the UMI (R1); application performance may be affected."),
    Alert("frac_reads_in_cells", "below", 0.70, 0.50, "Low fraction reads in cells",
          "Ideal > 70%",
          "Many reads were not assigned to cell barcodes: high ambient RNA, or cells with low "
          "RNA content that the cell caller missed (Cell Ranger's --force-cells addresses the "
          "latter)."),
    Alert("mean_reads_per_cell", "below", 20_000, None, "Below 10x's recommended depth",
          "Minimum 20,000 read pairs per cell",
          "10x recommends at least 20,000 read pairs per cell for 3' and 5' Gene Expression "
          "(10,000 for Flex); check the sequencing saturation before sequencing more.",
          source="10x sequencing requirements", url=DEPTH_URL, level="info"),
)

# Flex (probe-based; its metrics carry "Reads confidently mapped to probe set"): 10x's
# minimum is 10,000 read pairs per cell.
_FLEX_DEPTH = Alert(
    "mean_reads_per_cell", "below", 10_000, None, "Below 10x's recommended depth",
    "Minimum 10,000 read pairs per cell (Flex)",
    "10x recommends at least 10,000 read pairs per cell for Flex Gene Expression; check the "
    "sequencing saturation before sequencing more.",
    source="10x sequencing requirements", url=FLEX_DEPTH_URL, level="info",
)

# cellranger multi, per-sample "Cells": at least 10 cells must be assigned to a sample
# (metrics_etl_special.toml, gex_sample_hero_metrics.total_singlets).
_MULTI_SAMPLE_CELLS = Alert(
    "estimated_cells", "below", 10, 1, "Low number of cells assigned to sample", "Ideal >= 10",
    "At least 10 cells must be assigned to a sample for Cell Ranger's secondary analysis; this "
    "usually indicates poor cell handling, library or sequencing quality.",
    url=CELLRANGER_MULTI_URL,
)


# ---------------------------------------------------------------------------
# Parsing metrics_summary.csv
# ---------------------------------------------------------------------------

_NUM_RE = re.compile(r"^\s*([-+]?(?:\d[\d,]*)?\.?\d+(?:[eE][-+]?\d+)?)\s*(%?)")
_MISSING = {"", "na", "nan", "n/a", "-", "---", "none", "null"}


def parse_value(raw: object, kind: str) -> float | int | None:
    """A metric value as Cell Ranger writes it: "1,234", "95.2%", "0.952" (multi, 10.x),
    "2,616 (5.29%)" (the leading number). Fractions come back as 0..1."""
    s = str(raw).strip().strip('"').strip()
    if s.lower() in _MISSING:
        return None
    m = _NUM_RE.match(s)
    if not m:
        return None
    try:
        v = float(m.group(1).replace(",", ""))
    except ValueError:
        return None
    if not np.isfinite(v):
        return None
    if kind == "frac":
        if m.group(2) or v > 1.0:  # "95.2%", or a bare percentage
            v /= 100.0
        return round(v, 6)
    if kind == "int":
        return int(round(v))
    return v


def _read_text(path: Path) -> str:
    raw = Path(path).read_bytes()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return raw.decode("utf-8-sig", errors="replace")


def parse_metrics(path: Path | str) -> dict:
    """Parse a metrics_summary.csv file; see :func:`parse_metrics_text`."""
    out = parse_metrics_text(_read_text(Path(path)))
    out["file"] = Path(path).name
    return out


def parse_metrics_text(text: str) -> dict:
    """The typed record of a metrics_summary.csv (count or multi layout).

    ``values``: metric key -> number (fractions 0..1, missing metrics absent);
    ``scope``: where each came from ("count", "cells" = the sample, "library");
    ``raw``: every row as [name, value] strings, in file order; ``notes``: sentences."""
    rows = [r for r in csv.reader(io.StringIO(text)) if any(c.strip() for c in r)]
    if not rows or not looks_like_metrics(",".join(rows[0])):
        raise SeqQcError("This is not a Cell Ranger metrics_summary.csv (no metric names found).")
    header = [c.strip() for c in rows[0]]
    if "metric name" in [h.lower() for h in header]:
        return _parse_multi(header, rows[1:])
    return _parse_count(header, rows[1:])


def _parse_count(header: list[str], rows: list[list[str]]) -> dict:
    notes: list[str] = []
    if not rows:
        raise SeqQcError("The metrics_summary.csv has a header but no row of values.")
    if len(rows) > 1:
        notes.append(f"The file has {len(rows)} rows of values; the first is used.")
    row = rows[0]
    raw = [[h, row[i].strip() if i < len(row) else ""] for i, h in enumerate(header)]
    by_name = {h.lower(): v for h, v in raw}
    values: dict[str, Any] = {}
    scope: dict[str, str] = {}
    for m in METRICS:
        for name in m.count:
            v = parse_value(by_name.get(name, ""), m.kind)
            if v is not None:
                values[m.key] = v
                scope[m.key] = "count"
                break
    return {"format": "count", "values": values, "scope": scope, "raw": raw, "notes": notes}


def _parse_multi(header: list[str], rows: list[list[str]]) -> dict:
    cols = {h.lower(): i for i, h in enumerate(header)}

    def col(*names: str) -> int | None:
        return next((cols[n] for n in names if n in cols), None)

    i_cat = col("category", "library or sample")
    i_type, i_by, i_group = col("library type"), col("grouped by"), col("group name")
    i_name, i_value = col("metric name"), col("metric value")
    if i_name is None or i_value is None:
        raise SeqQcError("The metrics_summary.csv has no Metric Name / Metric Value columns.")

    def cell(r: list[str], i: int | None) -> str:
        return r[i].strip() if i is not None and i < len(r) else ""

    raw: list[list[str]] = []
    gex: list[tuple[str, str, str, str, str]] = []  # (category, grouped by, group, name, value)
    for r in rows:
        cat, ltype, by, group = cell(r, i_cat), cell(r, i_type), cell(r, i_by), cell(r, i_group)
        name, value = cell(r, i_name), cell(r, i_value)
        label = " · ".join(p for p in (cat, ltype, f"{by} {group}".strip(), name) if p)
        raw.append([label, value])
        if ltype.lower() not in ("gene expression", ""):
            continue
        scope = "cells" if cat.lower() in ("cells", "sample") else "library"
        gex.append((scope, by.lower(), group, name.lower(), value))

    values: dict[str, Any] = {}
    scope_of: dict[str, str] = {}
    for m in METRICS:
        for want_scope, want_name in m.multi:
            hits = [g for g in gex if g[0] == want_scope and g[3] == want_name]
            # A physical library's row before a FASTQ's, then file order.
            hits.sort(key=lambda g: 0 if g[1] in ("physical library id", "") else 1)
            v = next((x for x in (parse_value(h[4], m.kind) for h in hits) if x is not None), None)
            if v is not None:
                values[m.key] = v
                scope_of[m.key] = want_scope
                break
    notes: list[str] = []
    libs = sorted({g[2] for g in gex if g[0] == "library" and g[1] == "physical library id" and g[2]})
    if len(libs) > 1:
        notes.append(f"{len(libs)} Gene Expression libraries ({', '.join(libs)}); library-level "
                     f"metrics are {libs[0]}'s.")
    if not gex:
        notes.append("No Gene Expression rows were found in this metrics file.")
    return {"format": "multi", "values": values, "scope": scope_of, "raw": raw, "notes": notes}


# ---------------------------------------------------------------------------
# Flags (Cell Ranger's web-summary alerts)
# ---------------------------------------------------------------------------


def _fmt(key: str, v: float) -> str:
    m = _BY_KEY.get(key)
    if m is not None and m.kind == "frac":
        return f"{v * 100:.1f}%"
    return f"{v:,.0f}"


def flags_for(metrics: dict | None) -> list[dict]:
    """Each 10x alert the metrics cross, worst first: level error / warn / info, the
    value, the threshold crossed, a plain sentence and the source."""
    if not metrics:
        return []
    values = metrics.get("values") or {}
    scope = metrics.get("scope") or {}
    out: list[dict] = []
    for alert in ALERTS:
        v = values.get(alert.key)
        if v is None:
            continue
        a = alert
        if alert.key == "estimated_cells" and scope.get(alert.key) == "cells" \
                and metrics.get("format") == "multi":
            a = _MULTI_SAMPLE_CELLS
        elif alert.key == "mean_reads_per_cell" and values.get("conf_probe_set") is not None:
            a = _FLEX_DEPTH
        below = a.direction == "below"

        def crosses(t: float) -> bool:
            return v < t if below else v > t

        if a.error is not None and crosses(a.error):
            level, line = "error", a.error
        elif crosses(a.warn):
            level, line = a.level, a.warn
        else:
            continue
        side = "below" if below else "above"
        label = _BY_KEY[a.key].label
        if level == "info":
            msg = f"{label} is {_fmt(a.key, v)}. {a.meaning}"
        else:
            other = (f" (warning {side} {_fmt(a.key, a.warn)})" if level == "error"
                     else f" (error {side} {_fmt(a.key, a.error)})" if a.error is not None else "")
            msg = (f"{label} is {_fmt(a.key, v)}, {side} Cell Ranger's {level} line of "
                   f"{_fmt(a.key, line)}{other}. {a.meaning}")
        out.append({
            "metric": a.key, "label": label, "level": level, "title": a.title,
            "value": v, "threshold": line, "warn": a.warn, "error": a.error,
            "direction": a.direction, "ideal": a.ideal, "message": msg,
            "source": a.source, "url": a.url,
        })
    rank = {"error": 0, "warn": 1, "info": 2}
    return sorted(out, key=lambda f: rank[f["level"]])


def checked_metrics(metrics: dict | None) -> list[str]:
    """The metrics present that have an alert (what "no flags" rests on)."""
    values = (metrics or {}).get("values") or {}
    return [a.key for a in ALERTS if values.get(a.key) is not None]


# ---------------------------------------------------------------------------
# Barcode ranks: knee + inflection (DropletUtils barcodeRanks, curve tracing)
# ---------------------------------------------------------------------------


def _interpolate_on_curve(target, cumdist, step, x, y):
    # R: findInterval(target, cumdist, left.open=TRUE) = #{cumdist < target}
    idx = np.searchsorted(cumdist, target, side="left") - 1
    idx = np.clip(idx, 0, len(x) - 2)
    prop = (target - cumdist[idx]) / step[idx + 1]
    return x[idx] + prop * (x[idx + 1] - x[idx]), y[idx] + prop * (y[idx + 1] - y[idx])


def barcode_ranks(
    totals,
    lower: float = BR_LOWER,
    exclude_from: int = BR_EXCLUDE_FROM,
    window: float = BR_WINDOW,
    gradient_threshold: float = BR_GRADIENT_THRESHOLD,
) -> dict[str, float]:
    """Knee and inflection of the log-log barcode-rank curve, as DropletUtils
    ``barcodeRanks`` (curve tracing, >= 1.30) computes them: ties share their mid-rank,
    totals at or below ``lower`` and the first ``exclude_from`` ranks are left out, and a
    window of ``window`` log10 units is traced along the curve. Returns the UMI totals
    ``{"knee", "inflection"}``; raises ValueError with fewer than 2 usable points."""
    t = np.asarray(totals, dtype=np.float64)
    t = np.sort(t[np.isfinite(t)])[::-1]
    if t.size == 0:
        raise ValueError("no barcodes")
    starts = np.r_[0, np.flatnonzero(np.diff(t)) + 1]
    lengths = np.diff(np.r_[starts, t.size])
    run_totals = t[starts]
    run_rank = np.cumsum(lengths) - (lengths - 1) / 2.0  # mid-rank of each run of ties
    keep = run_totals > lower
    keep[run_rank <= exclude_from] = False
    if int(keep.sum()) < 2:
        raise ValueError("insufficient unique points for computing knee/inflection points")
    y = np.log10(run_totals[keep])
    x = np.log10(run_rank[keep])

    step = np.r_[0.0, np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)]
    cumdist = np.cumsum(step)
    rhs = cumdist + window
    to_scan = rhs <= cumdist[-1]
    if not to_scan.any():
        knee = inflection = 10 ** y[-1]
        return {"knee": float(knee), "inflection": float(inflection)}

    left_x, left_y = x[to_scan], y[to_scan]
    right_x, right_y = _interpolate_on_curve(rhs[to_scan], cumdist, step, x, y)
    gap = np.sqrt((left_x - right_x) ** 2 + (left_y - right_y) ** 2)
    gradient = (right_y - left_y) / (right_x - left_x)
    intercept = right_y - gradient * right_x
    mid_x, mid_y = _interpolate_on_curve(cumdist[to_scan] + window / 2, cumdist, step, x, y)
    mid_above = mid_y > gradient * mid_x + intercept

    elbows = np.flatnonzero(~mid_above & (gradient < gradient_threshold))
    if elbows.size == 0:
        infl_window = int(np.argmin(gradient))
        maybe_knee = np.flatnonzero(mid_above)
    else:
        first = int(elbows[0])
        maybe_knee = np.flatnonzero(mid_above[: first + 1])
        # Windows that start past the midpoint of the first elbow window are skipped.
        before = int(np.searchsorted(left_x, mid_x[first], side="right"))
        infl_window = int(np.argmin(gradient[: max(before, 1)]))
    knee_window = int(maybe_knee[np.argmin(gap[maybe_knee])]) if maybe_knee.size else infl_window
    return {"knee": float(10 ** mid_y[knee_window]), "inflection": float(10 ** mid_y[infl_window])}


# ---------------------------------------------------------------------------
# Reading UMI totals from a raw (unfiltered) matrix
# ---------------------------------------------------------------------------


def raw_totals_h5(path: Path | str) -> tuple[np.ndarray, np.ndarray, dict]:
    """(barcodes, UMI totals) of every barcode with at least one Gene Expression UMI in a
    Cell Ranger HDF5 matrix, read in blocks of matrix entries (never the whole matrix)."""
    with h5py.File(path, "r") as f:
        if "matrix" not in f:
            raise SeqQcError(f"{Path(path).name} is not a Cell Ranger v3+ HDF5 matrix.")
        m = f["matrix"]
        n_features, n_bc = (int(v) for v in m["shape"][:2])
        indptr = np.asarray(m["indptr"][:], dtype=np.int64)  # n_barcodes + 1 (54 MB for 3' v3)
        gex: np.ndarray | None = None
        feats = m.get("features")
        if feats is not None and "feature_type" in feats:
            types = feats["feature_type"][:]
            is_gex = types == b"Gene Expression"
            if not is_gex.all():
                gex = is_gex
        data, indices = m["data"], m["indices"]
        totals = np.zeros(n_bc, dtype=np.int64)
        b0 = 0
        while b0 < n_bc:
            s = int(indptr[b0])
            b1 = int(np.searchsorted(indptr, s + _NNZ_CHUNK, side="right")) - 1
            # At most _BC_CHUNK barcodes too: the per-barcode temporaries stay small.
            b1 = min(max(b1, b0 + 1), n_bc, b0 + _BC_CHUNK)
            e = int(indptr[b1])
            if e > s:
                vals = data[s:e].astype(np.int64)
                if gex is not None:
                    vals *= gex[indices[s:e]]
                cs = np.concatenate(([0], np.cumsum(vals)))
                seg = indptr[b0:b1 + 1] - s
                totals[b0:b1] = cs[seg[1:]] - cs[seg[:-1]]
            b0 = b1
        del indptr
        nz = np.flatnonzero(totals > 0)
        bcs = m["barcodes"]
        out = []
        for start in range(0, n_bc, _BC_CHUNK):
            stop = min(n_bc, start + _BC_CHUNK)
            lo, hi = np.searchsorted(nz, [start, stop])
            if hi > lo:
                out.append(bcs[start:stop][nz[lo:hi] - start])
    barcodes = (np.concatenate(out) if out else np.zeros(0, dtype="S1")).astype(str)
    info = {"n_barcodes_in_file": n_bc, "n_features": n_features,
            "gex_features": int(gex.sum()) if gex is not None else n_features,
            "feature_types": "gene expression only" if gex is None else "gene expression of a mixed matrix"}
    return barcodes, totals[nz], info


def _open_text(path: Path):
    with open(path, "rb") as fh:
        gz = fh.read(2) == b"\x1f\x8b"
    return gzip.open(path, "rt", encoding="utf-8", errors="replace") if gz else open(
        path, "rt", encoding="utf-8", errors="replace")


def raw_totals_mex(matrix: Path, barcodes: Path, features: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """(barcodes, UMI totals) from a raw 10x MEX triplet, streamed in chunks."""
    with _open_text(features) as fh:
        ftypes = [ln.rstrip("\n").split("\t")[2] if ln.count("\t") >= 2 else "Gene Expression"
                  for ln in fh if ln.strip()]
    is_gex = np.array([t == "Gene Expression" for t in ftypes], dtype=bool)
    n_features = len(ftypes)
    n_header = 0
    dims: list[int] | None = None
    with _open_text(matrix) as fh:
        for line in fh:
            n_header += 1
            s = line.strip()
            if not s or s.startswith("%"):
                continue
            dims = [int(p) for p in s.split()[:3]]
            break
    if not dims:
        raise SeqQcError(f"{Path(matrix).name} has no Matrix Market dimensions line.")
    n_rows, n_cols, _ = dims
    if n_rows == n_features:
        feat_col, bc_col, n_bc = 0, 1, n_cols
    elif n_cols == n_features:
        feat_col, bc_col, n_bc = 1, 0, n_rows
    else:
        raise SeqQcError(f"The raw matrix is {n_rows} x {n_cols} but its features file lists "
                         f"{n_features} features.")
    totals = np.zeros(n_bc, dtype=np.float64)
    all_gex = bool(is_gex.all())
    with _open_text(matrix) as fh:
        reader = pd.read_csv(fh, sep=" ", header=None, skiprows=n_header, usecols=[0, 1, 2],
                             dtype={0: np.int32, 1: np.int32, 2: np.float64},
                             chunksize=_MEX_ROWS)
        for chunk in reader:
            w = chunk[2].to_numpy()
            if not all_gex:
                w = w * is_gex[chunk[feat_col].to_numpy() - 1]
            totals += np.bincount(chunk[bc_col].to_numpy() - 1, weights=w, minlength=n_bc)
            del chunk, w
    totals_i = np.rint(totals).astype(np.int64)
    nz = np.flatnonzero(totals_i > 0)
    keep = np.zeros(n_bc, dtype=bool)
    keep[nz] = True
    out: list[np.ndarray] = []
    pos = 0
    with _open_text(barcodes) as fh:
        for chunk in pd.read_csv(fh, sep="\t", header=None, usecols=[0], dtype=str,
                                 na_filter=False, chunksize=_MEX_ROWS // 4):
            col = chunk[0].to_numpy(dtype=str)
            if pos + len(col) > n_bc:
                pos += len(col)
                break
            out.append(col[keep[pos:pos + len(col)]])
            pos += len(col)
    if pos != n_bc:
        raise SeqQcError(f"The raw barcodes file does not list the matrix's {n_bc:,} barcodes "
                         f"({'more' if pos > n_bc else 'fewer'} lines).")
    bc = np.concatenate(out) if out else np.zeros(0, dtype=str)
    info = {"n_barcodes_in_file": n_bc, "n_features": n_features, "gex_features": int(is_gex.sum()),
            "feature_types": "gene expression only" if all_gex else "gene expression of a mixed matrix"}
    return bc, totals_i[nz], info


# ---------------------------------------------------------------------------
# Matching the dataset's cells to raw barcodes; knee summary + curve
# ---------------------------------------------------------------------------

_CORE_RE = r"([ACGTN]{8,})"


def _core(names: np.ndarray) -> np.ndarray:
    return pd.Series(names, dtype=object).str.extract(_CORE_RE, expand=False).fillna("").to_numpy(dtype=str)


def match_cells(raw_barcodes: np.ndarray, cells: Iterable[str]) -> tuple[np.ndarray, int, str]:
    """(raw barcode is one of the cells, cells found, how they were matched). Full
    barcodes first; when under half the cells match, the bare barcode sequence (the
    "-1" suffix and any sample prefix dropped), as merged or re-labelled data name them."""
    cells_arr = np.asarray(list(cells), dtype=str)
    if cells_arr.size == 0 or raw_barcodes.size == 0:
        return np.zeros(raw_barcodes.size, dtype=bool), 0, "barcode"
    mask = np.isin(raw_barcodes, cells_arr)
    found = int(np.isin(cells_arr, raw_barcodes).sum())
    if found >= 0.5 * cells_arr.size:
        return mask, found, "barcode"
    raw_core = np.char.partition(raw_barcodes.astype(str), "-")[:, 0]
    cell_core = _core(cells_arr)
    cell_core = cell_core[cell_core != ""]
    mask2 = np.isin(raw_core, cell_core)
    found2 = int(np.isin(cell_core, raw_core).sum())
    if found2 > found:
        return mask2, found2, "sequence"
    return mask, found, "barcode"


def _curve(totals_sorted: np.ndarray, kept_sorted: np.ndarray) -> dict:
    """At most MAX_CURVE_POINTS points of the rank curve: per log-spaced rank bin, the
    first cell and the first other barcode (so mixed bins show both), plus the last."""
    n = totals_sorted.size
    if n == 0:
        return {"rank": [], "umi": [], "kept": []}
    edges = np.unique(np.floor(np.logspace(0, np.log10(n), _CURVE_BINS + 1)).astype(np.int64))
    ranks = np.arange(1, n + 1)
    bin_of = np.searchsorted(edges, ranks, side="right") - 1
    key = bin_of * 2 + kept_sorted.astype(np.int64)
    _, first = np.unique(key, return_index=True)
    idx = np.unique(np.r_[first, n - 1])[:MAX_CURVE_POINTS]
    return {
        "rank": (idx + 1).tolist(),
        "umi": totals_sorted[idx].astype(np.int64).tolist(),
        "kept": kept_sorted[idx].astype(int).tolist(),
    }


def knee_summary(barcodes: np.ndarray, totals: np.ndarray, cells: Iterable[str]) -> dict:
    """Barcode-rank summary of a raw matrix against the dataset's cells: knee and
    inflection, how many barcodes lie above each, the cells' UMI floor and the curve."""
    cells = list(cells)
    kept, found, how = match_cells(barcodes, cells)
    order = np.argsort(-totals, kind="stable")
    t_sorted = totals[order]
    k_sorted = kept[order]
    out: dict[str, Any] = {
        "n_barcodes": int(totals.size),
        "total_umis": int(totals.sum()),
        "n_cells": len(cells),
        "n_cells_matched": found,
        "match": how,
        "knee": None, "inflection": None, "n_above_knee": None, "n_above_inflection": None,
        "method": ("DropletUtils barcodeRanks (curve tracing): lower = "
                   f"{BR_LOWER}, exclude.from = {BR_EXCLUDE_FROM}, window = {BR_WINDOW:g}, "
                   f"gradient.threshold = {BR_GRADIENT_THRESHOLD:g}"),
        "url": DROPLETUTILS_URL,
    }
    try:
        br = barcode_ranks(totals)
        out["knee"] = round(br["knee"], 1)
        out["inflection"] = round(br["inflection"], 1)
        out["n_above_knee"] = int((totals >= br["knee"]).sum())
        out["n_above_inflection"] = int((totals >= br["inflection"]).sum())
    except ValueError as e:
        out["knee_note"] = (f"No knee or inflection: fewer than 2 distinct UMI totals above "
                            f"{BR_LOWER} ({e}).")
    if kept.any():
        kt = totals[kept]
        out["cells_umi_min"] = int(kt.min())
        out["cells_umi_median"] = float(np.median(kt))
        out["frac_umis_in_cells"] = round(float(kt.sum()) / max(int(totals.sum()), 1), 4)
        if out["knee"] is not None:
            out["n_cells_below_knee"] = int((kt < br["knee"]).sum())
            out["n_above_knee_not_cells"] = int(((totals >= br["knee"]) & ~kept).sum())
    out["curve"] = _curve(t_sorted, k_sorted)
    return out


# ---------------------------------------------------------------------------
# Records: build, store, read
# ---------------------------------------------------------------------------


def _raw_parts(files: list) -> tuple[Path | None, tuple[Path, Path, Path] | None]:
    """From side BundleFiles (or (path, kind) pairs): the raw .h5, or the raw MEX triplet."""
    h5 = None
    mex: dict[str, Path] = {}
    for f in files:
        kind = getattr(f, "kind", None)
        kind = getattr(kind, "value", kind)
        path = Path(getattr(f, "path", f))
        if kind == "tenx_h5":
            h5 = h5 or path
        elif kind in ("mtx_matrix", "tenx_barcodes", "tenx_features"):
            mex.setdefault(kind, path)
    if h5 is not None:
        return h5, None
    if len(mex) == 3:
        return None, (mex["mtx_matrix"], mex["tenx_barcodes"], mex["tenx_features"])
    return None, None


def build_record(
    sample: str,
    files: list,
    cells: Iterable[str],
    *,
    column: str | None = None,
    cells_source: str = "ingest",
) -> dict:
    """One sample's record from its side files (metrics CSV and/or raw matrix).

    ``files``: BundleFiles (``path`` + ``kind``). ``cells``: the dataset's barcodes for
    this sample; ``cells_source`` "ingest" = the uploaded filtered matrix (Cell Ranger's
    call), "dataset" = the dataset's cells when attached later (after any filtering)."""
    notes: list[str] = []
    rec: dict[str, Any] = {
        "sample": str(sample), "column": column, "cells_source": cells_source,
        "attached_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "files": {"metrics": None, "raw": None}, "metrics": None, "knee": None, "notes": notes,
    }
    metrics_files = [f for f in files if getattr(getattr(f, "kind", None), "value",
                                                 getattr(f, "kind", None)) == "tenx_metrics"]
    if metrics_files:
        mf = Path(getattr(metrics_files[0], "path", metrics_files[0]))
        rec["files"]["metrics"] = mf.name
        try:
            rec["metrics"] = parse_metrics(mf)
        except (SeqQcError, OSError, UnicodeDecodeError) as e:
            notes.append(f"{mf.name} could not be read: {e}")
    h5, mex = _raw_parts(files)
    if h5 is not None or mex is not None:
        names = [h5.name] if h5 is not None else [p.name for p in mex]  # type: ignore[union-attr]
        rec["files"]["raw"] = names
        try:
            bc, totals, info = raw_totals_h5(h5) if h5 is not None else raw_totals_mex(*mex)  # type: ignore[misc]
            summary = knee_summary(bc, totals, cells)
            summary.update(info)
            rec["knee"] = summary
            if summary["n_cells"] and summary["n_cells_matched"] < 0.5 * summary["n_cells"]:
                notes.append(
                    f"Only {summary['n_cells_matched']:,} of the {summary['n_cells']:,} cells were "
                    "found in the raw matrix: it may be from another sample.")
            elif summary["n_barcodes"] and summary["n_cells_matched"] >= 0.9 * summary["n_barcodes"]:
                notes.append(
                    f"{summary['n_cells_matched']:,} of the matrix's {summary['n_barcodes']:,} "
                    "barcodes are cells: it looks filtered, so the curve has no empty droplets. "
                    "The knee needs the raw (unfiltered) raw_feature_bc_matrix.")
        except (SeqQcError, OSError, KeyError, ValueError) as e:
            notes.append(f"The raw matrix could not be read: {e}")
        if rec["knee"] is not None:
            _add_soup(rec, h5, mex, notes)
    return rec


# Fewer empty droplets than this (a filtered matrix sent as the raw one): no soup.
MIN_SOUP_DROPLETS = 100


def _add_soup(rec: dict, h5: Path | None, mex: tuple[Path, Path, Path] | None,
              notes: list[str]) -> None:
    """The ambient RNA profile of the raw matrix's empty droplets (``core/ambient.py``):
    a summary in the record, the profile itself under ``_soup`` until ``store`` moves it
    to ``uns[ambient.UNS_SOUP]``."""
    try:
        soup = ambient.soup_for_files(h5, mex)
    except (OSError, KeyError, ValueError) as e:
        notes.append(f"The ambient RNA profile (empty droplets) could not be read: {e}")
        return
    if soup is None:
        return
    if soup["n_droplets"] < MIN_SOUP_DROPLETS:
        notes.append(f"Only {soup['n_droplets']:,} droplets with fewer than {soup['umi_max']} UMIs: "
                     "too few empty droplets for an ambient RNA profile.")
        return
    rec["soup"] = ambient.soup_summary(soup)
    rec["_soup"] = soup


def read(adata) -> dict | None:
    """The stored ``{"version", "samples": {label: record}}``, or None."""
    raw = getattr(adata, "uns", {}).get(UNS_KEY)
    if raw is None:
        return None
    try:
        data = json.loads(raw) if isinstance(raw, (str, bytes)) else dict(raw)
    except (TypeError, ValueError):
        logger.warning("Unreadable %s entry; ignored", UNS_KEY)
        return None
    if not isinstance(data, dict) or not data.get("samples"):
        return None
    return data


def store(adata, records: list[dict]) -> None:
    """Add (or replace, by sample) records; a new JSON string so write_derived saves it."""
    data = read(adata) or {"version": VERSION, "samples": {}}
    samples = dict(data.get("samples") or {})
    soups: dict[str, dict] = {}
    for rec in records:
        soup = rec.pop("_soup", None)
        if soup is not None:
            soups[rec["sample"]] = soup
        elif rec.get("knee") is not None:  # a new raw matrix without a soup replaces one
            ambient.remove_soup(adata, rec["sample"])
        samples[rec["sample"]] = rec
    adata.uns[UNS_KEY] = json.dumps({"version": VERSION, "samples": samples})
    if soups:  # the profiles beside it (``core/ambient.py``)
        ambient.store_soups(adata, soups)


def summary_sentence(rec: dict, flags: list[dict]) -> str:
    """One plain sentence on a sample: the alerts, or why there are none."""
    metrics = rec.get("metrics")
    if not metrics:
        return "No metrics_summary.csv is attached, so no Cell Ranger alert can be checked."
    worst = [f for f in flags if f["level"] in ("error", "warn")]
    n_checked = len(checked_metrics(metrics))
    if not worst:
        return (f"No Cell Ranger alert fires: the {n_checked} metrics with an alert are all on "
                "the good side of 10x's warning lines.")
    names = ", ".join(f["label"].lower() for f in worst)
    return f"{len(worst)} of {n_checked} checked metrics cross a Cell Ranger alert: {names}."


def payload(adata) -> dict | None:
    """The GET answer: every sample's record with its flags (computed now, so the
    thresholds in this module apply to records stored earlier), or None."""
    data = read(adata)
    if not data:
        return None
    samples = []
    for rec in data["samples"].values():
        flags = flags_for(rec.get("metrics"))
        samples.append({**rec, "flags": flags, "summary": summary_sentence(rec, flags)})
    return {
        "samples": samples,
        "n_samples": len(samples),
        "metrics_info": [{"key": m.key, "label": m.label, "kind": m.kind, "tip": m.tip}
                         for m in METRICS],
        "alerts": [{"metric": a.key, "direction": a.direction, "warn": a.warn, "error": a.error,
                    "title": a.title, "level": a.level, "source": a.source, "url": a.url}
                   for a in ALERTS],
    }


# ---------------------------------------------------------------------------
# Attaching: at ingest (units) and to an existing dataset (endpoint)
# ---------------------------------------------------------------------------


def _unit_cells(adata, label: str, sample_label: str | None) -> list[str]:
    names = adata.obs_names.astype(str)
    if sample_label and sample_label in adata.obs.columns:
        mask = (adata.obs[sample_label].astype(str) == str(label)).to_numpy()
        suffix = f":{label}"  # merge_units' index_unique=":"
        return [n[: -len(suffix)] if n.endswith(suffix) else n for n in names[mask]]
    return list(names)


def attach_from_units(adata, units: list, sample_label: str | None = None) -> int:
    """At ingest: one record per unit that came with side files (metrics, raw matrix).
    ``sample_label``: the merged dataset's sample column (None for one unit). Never
    raises; returns the number of records stored."""
    records = []
    merged = len(units) > 1
    for u in units:
        side = list(getattr(u, "side_files", None) or [])
        if not side:
            continue
        try:
            cells = _unit_cells(adata, u.label, sample_label if merged else None)
            records.append(build_record(u.label, side, cells,
                                        column=sample_label if merged else None))
        except Exception as e:  # sequencing QC must never block ingest
            logger.warning("Sequencing metrics for %s skipped: %s", u.label, e)
    if records:
        try:
            store(adata, records)
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("Could not store sequencing metrics: %s", e)
            return 0
    return len(records)


def classify_files(paths: list[Path]) -> list:
    """Uploaded attach files as side BundleFiles (metrics CSV, raw .h5 or MEX parts);
    anything else raises a sentence naming it."""
    from scview.core.ingestion.bundling import BundleFile, FileRole
    from scview.core.ingestion.detection import FileKind, detect_file

    allowed = {FileKind.tenx_metrics, FileKind.tenx_h5, FileKind.mtx_matrix,
               FileKind.tenx_barcodes, FileKind.tenx_features}
    out, bad = [], []
    for p in paths:
        det = detect_file(p)
        if det.kind not in allowed:
            bad.append(p.name)
            continue
        out.append(BundleFile(path=str(p), name=p.name, kind=det.kind, role=FileRole.side,
                              confidence=det.confidence))
    if bad:
        raise SeqQcError(
            f"Not a Cell Ranger metrics or raw matrix file: {', '.join(bad)}. Attach "
            "metrics_summary.csv and/or raw_feature_bc_matrix.h5 (or the raw matrix.mtx, "
            "barcodes.tsv and features.tsv).")
    kinds = {f.kind for f in out}
    mex = {FileKind.mtx_matrix: "matrix.mtx", FileKind.tenx_barcodes: "barcodes.tsv",
           FileKind.tenx_features: "features.tsv"}
    parts = kinds & set(mex)
    if parts and parts != set(mex):
        names = ", ".join(f.name for f in out if f.kind in mex)
        missing = ", ".join(v for k, v in mex.items() if k not in parts)
        raise SeqQcError(f"A raw matrix in 10x MEX form is three files; {names} needs "
                         f"{missing} as well.")
    return out


def dataset_cells(adata, sample: str | None, column: str | None) -> list[str]:
    """The dataset's barcodes for ``sample`` of ``column`` (all cells without a column)."""
    if column:
        if column not in adata.obs.columns:
            raise SeqQcError(f"Column '{column}' not found.", 404)
        if not sample:
            raise SeqQcError(f"Choose which '{column}' value these files belong to.")
        mask = (adata.obs[column].astype(str) == str(sample)).to_numpy()
        if not mask.any():
            raise SeqQcError(f"No cells have '{column}' = '{sample}'.", 404)
        return list(adata.obs_names.astype(str)[mask])
    return list(adata.obs_names.astype(str))


def attach_files(adata, paths: list[Path], sample: str, column: str | None) -> dict:
    """Build the record for files attached to an existing dataset and store it."""
    files = classify_files(paths)
    cells = dataset_cells(adata, sample if column else None, column)
    rec = build_record(sample, files, cells, column=column or None, cells_source="dataset")
    if rec["metrics"] is None and rec["knee"] is None:
        raise SeqQcError(" ".join(rec["notes"]) or "None of the files could be read.")
    k = rec["knee"]
    if k is not None and k["n_cells"] and not k["n_cells_matched"]:
        raise SeqQcError(
            f"None of the {k['n_cells']:,} cells is in this raw matrix: it is from another sample, "
            "or the dataset's cell names are not 10x barcodes. Nothing was attached.")
    store(adata, [rec])
    return rec


def remove(adata, sample: str) -> bool:
    """Drop one sample's record (the key goes when none is left). False: no such sample."""
    data = read(adata)
    if not data or sample not in data["samples"]:
        return False
    samples = {k: v for k, v in data["samples"].items() if k != sample}
    ambient.remove_soup(adata, sample)
    if samples:
        adata.uns[UNS_KEY] = json.dumps({"version": VERSION, "samples": samples})
    else:
        del adata.uns[UNS_KEY]
    return True


# ---------------------------------------------------------------------------
# Plain sentences for the QC report and the co-pilot
# ---------------------------------------------------------------------------


def _pct(v: float | None) -> str:
    return f"{v * 100:.1f}%" if v is not None else "n/a"


def sample_line(rec: dict, flags: list[dict]) -> str:
    """Sequencing facts of one sample in one line."""
    bits: list[str] = []
    vals = (rec.get("metrics") or {}).get("values") or {}
    if vals:
        parts = []
        if "estimated_cells" in vals:
            parts.append(f"{vals['estimated_cells']:,} estimated cells")
        if "mean_reads_per_cell" in vals:
            parts.append(f"{vals['mean_reads_per_cell']:,} mean reads per cell")
        if "median_genes_per_cell" in vals:
            parts.append(f"median {vals['median_genes_per_cell']:,} genes per cell")
        for key, text in (("sequencing_saturation", "sequencing saturation"),
                          ("frac_reads_in_cells", "reads in cells"),
                          ("conf_transcriptome", "mapped confidently to the transcriptome"),
                          ("valid_barcodes", "valid barcodes"),
                          ("q30_rna_read", "Q30 in the RNA read")):
            if key in vals:
                parts.append(f"{_pct(vals[key])} {text}")
        bits.append("Cell Ranger metrics: " + ", ".join(parts) + ".")
        worst = [f for f in flags if f["level"] in ("error", "warn")]
        if worst:
            bits.append("10x alerts: " + "; ".join(
                f"{f['title']} ({_fmt(f['metric'], f['value'])}, {f['level']} line "
                f"{_fmt(f['metric'], f['threshold'])})" for f in worst) + ".")
        else:
            bits.append(f"No 10x alert fires ({len(checked_metrics(rec.get('metrics')))} metrics "
                        "checked).")
        info = [f for f in flags if f["level"] == "info"]
        if info:
            bits.append(info[0]["message"])
    k = rec.get("knee")
    if k:
        if k.get("knee") is not None:
            bits.append(
                f"Barcode ranks ({k['n_barcodes']:,} barcodes with UMIs): knee at "
                f"{k['knee']:,.0f} UMIs ({k['n_above_knee']:,} barcodes above), inflection at "
                f"{k['inflection']:,.0f} UMIs ({k['n_above_inflection']:,} above).")
        what = "the filtered matrix's" if rec.get("cells_source") == "ingest" else "the dataset's"
        if k.get("cells_umi_min") is not None:
            bits.append(
                f"{what.capitalize()} {k['n_cells_matched']:,} cells found in the raw matrix have "
                f"at least {k['cells_umi_min']:,} UMIs (median {k['cells_umi_median']:,.0f}) and "
                f"hold {_pct(k.get('frac_umis_in_cells'))} of its UMIs.")
    return " ".join(bits) or "No readable sequencing metrics."


def grounding_lines(adata) -> tuple[list[str], str]:
    """Co-pilot fact lines: one per sample, tagged [result:seqqc]; ``(lines, detail)``."""
    data = read(adata)
    if not data:
        return [], ""
    lines = ["\n## Sequencing QC from Cell Ranger outputs (per sample; alerts are Cell "
             f"Ranger's own web-summary thresholds) [{TAG}]"]
    details = []
    for rec in data["samples"].values():
        flags = flags_for(rec.get("metrics"))
        lines.append(f"- {rec['sample']}: {sample_line(rec, flags)} [{TAG}]")
        worst = [f["title"] for f in flags if f["level"] in ("error", "warn")]
        details.append(f"{rec['sample']}: {', '.join(worst) if worst else 'no 10x alert'}")
    return lines, "; ".join(details)


def report_facts(adata) -> tuple[list[str], list[str]]:
    """(fact lines, concern sentences) for the QC report: one fact per sample, one
    concern per error or warning alert (at most 4)."""
    data = read(adata)
    if not data:
        return [], []
    facts, concerns = [], []
    for rec in data["samples"].values():
        flags = flags_for(rec.get("metrics"))
        facts.append(f"Sample {rec['sample']}: {sample_line(rec, flags)}")
        for f in flags:
            if f["level"] in ("error", "warn"):
                concerns.append(f"Sample {rec['sample']}: {f['title'].lower()} "
                                f"({_fmt(f['metric'], f['value'])}; Cell Ranger's {f['level']} "
                                f"line is {_fmt(f['metric'], f['threshold'])}).")
    return facts, concerns[:4]
