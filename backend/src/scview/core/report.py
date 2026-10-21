"""Cluster and gene reports (C064): one page per cluster and one per gene that
collects the evidence scView already has, with n in every caption and a methods
paragraph, for a colleague to read without opening six tabs.

Nothing here is a new analysis. Each section reads a stored result (markers, the
label confidence check, the filtering step's removal record, the last composition
test) or makes a cheap computation the app already makes elsewhere (the cluster
QC table, ORA of the shared marker choice against HALLMARK, per-donor pseudobulk,
co-expression). A section whose input was never computed, and is slow to compute,
says so and names the tab that computes it (status ``not_computed``); one that
cannot apply to the dataset (no donors, no protein panel) says why
(``not_applicable``). Nothing is silently missing, the report never downloads a
database and it never writes to the dataset.

Every section is ``{id, title, caption, method, status, note, data}``: the caption
is a plain sentence with the n, the method one sentence on how the numbers were
made. ``cluster_report`` lives in ``core/report_cluster.py``, ``gene_report`` in
``core/report_gene.py``, and ``core/report_html.py`` renders either as one
self-contained HTML file.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from scview.core.confounding import condition_columns
from scview.core.donor_expression import detect_sample_key

logger = logging.getLogger(__name__)

OK = "ok"
NOT_COMPUTED = "not_computed"
NOT_APPLICABLE = "not_applicable"

MAX_GROUPS = 200  # a column with more values is per-cell, not a grouping
MAX_ROWS = 60  # rows of a per-donor or per-group table before it is capped

# Where in the app each missing input is made, for "Not computed yet: ..." notes.
WHERE = {
    "markers": "Unified View > Markers",
    "annotation": "Data Assessment > Cell-type annotation",
    "confidence": "Data Assessment > Annotation confidence",
    "marker_db": "Data Assessment > Annotation cross-check > Marker database check",
    "doublets": "Data Assessment (processing pipeline, doublet detection)",
    "hvg": "Data Assessment (processing pipeline, highly variable genes)",
    "composition": "Compare > Composition",
}

# Clusterings, cell types and QC calls look like conditions to the condition regex
# ("louvain_group", "doublet_status"): never a condition here, as in the QC report.
_NOT_CONDITION = re.compile(r"cell[ ._-]?type|celltypeanno|leiden|louvain|cluster|phase|doublet", re.I)


class ReportError(ValueError):
    """A request a report cannot serve: ``status`` is the HTTP code, ``suggestion``
    a close match for an unknown gene."""

    def __init__(self, message: str, status: int = 400, suggestion: str | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.suggestion = suggestion


def section(
    id: str, title: str, *, caption: str = "", method: str = "", status: str = OK,
    note: str | None = None, data: dict | None = None,
) -> dict:
    return {"id": id, "title": title, "caption": caption, "method": method,
            "status": status, "note": note, "data": data or {}}


def guarded(id: str, title: str, build: Callable[[], dict]) -> dict:
    """``build()``, or a section saying it failed: one broken input must not lose
    the rest of the report."""
    try:
        return build()
    except ReportError:
        raise
    except Exception as exc:  # a malformed stored result, an unexpected dataset
        logger.warning("report section %s failed: %s", id, exc, exc_info=True)
        return section(id, title, status=NOT_COMPUTED,
                       note=f"This section could not be made for this dataset: {exc}")


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------


def labels(s: pd.Series) -> pd.Series:
    """Values as strings, missing values kept missing."""
    return s.astype(str).where(s.notna())


def group_order(s: pd.Series) -> list[str]:
    """The values present, in the column's category order, else naturally sorted."""
    present = set(labels(s).dropna())
    if isinstance(s.dtype, pd.CategoricalDtype):
        return [str(c) for c in s.cat.categories if str(c) in present]

    def key(v: str):
        try:
            return (0, float(v), v)
        except ValueError:
            return (1, 0.0, v)

    return sorted(present, key=key)


def grouping(obs: pd.DataFrame, column: str, what: str = "column") -> pd.Series:
    """``column``'s labels, checked to be a grouping (2 to ``MAX_GROUPS`` values)."""
    if column not in obs.columns:
        raise ReportError(f"Column '{column}' not found.", 404)
    lab = labels(obs[column])
    k = int(lab.nunique(dropna=True))
    if k > MAX_GROUPS:
        raise ReportError(
            f"'{column}' has {k:,} values; a report needs a grouping {what} with at most {MAX_GROUPS}.", 422)
    return lab


def pct(x: float | None, digits: int = 1) -> str:
    return "n/a" if x is None else f"{100.0 * x:.{digits}f}%"


def num(v, digits: int = 4) -> float | None:
    """A finite float rounded, else None (JSON has no NaN)."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return round(f, digits) if np.isfinite(f) else None


def plural(n: int, word: str, many: str | None = None) -> str:
    return f"{n:,} {word if n == 1 else (many or word + 's')}"


def design(obs: pd.DataFrame, column: str) -> tuple[str | None, str | None]:
    """(condition column, donor column) for a report on ``column``: a
    condition-like name with 2 to 8 values (``confounding.condition_columns``),
    and the donor column the composition test and per-donor plots pick
    (``donor_expression.detect_sample_key``: a donor-like name with 2+ donors in
    every condition)."""
    conds = [c for c in condition_columns(obs) if c != column and not _NOT_CONDITION.search(c)]
    condition = conds[0] if conds else None
    donor = detect_sample_key(obs, column, condition)
    if donor is not None and condition is not None and donor == condition:
        donor = None
    return condition, donor


# ---------------------------------------------------------------------------
# Stored markers and the marker databases (both reports)
# ---------------------------------------------------------------------------

MARKER_FILTER_TEXT = (
    "expressed in at least {pts:.0%} of the group's cells, log fold change at least {lfc:g}, "
    "adjusted p under {padj:g}, ribosomal and mitochondrial genes left out"
)


def marker_filter_text() -> str:
    from scview.core.marker_selection import MAX_PADJ, MIN_LOGFC, MIN_PTS

    return MARKER_FILTER_TEXT.format(pts=MIN_PTS, lfc=MIN_LOGFC, padj=MAX_PADJ)


def stored_markers(adata, column: str) -> tuple[str | None, str | None]:
    """(uns key of ``column``'s stored markers, None) when they cover the column's
    current groups, else (None, a note saying why and where to compute them). A
    column relabelled since its markers ran no longer matches them."""
    from scview.core.ora_background import resolve_markers_key

    where = WHERE["markers"]
    key = resolve_markers_key(adata.uns, column)
    if key is None:
        return None, f"Not computed yet: compute marker genes for '{column}' in {where}."
    try:
        stored = {str(g) for g in adata.uns[key]["names"].dtype.names}
    except (KeyError, TypeError, AttributeError):
        return None, f"The stored markers for '{column}' are unreadable: compute them again in {where}."
    if stored != set(group_order(adata.obs[column])):
        return None, (f"The stored markers for '{column}' were computed for other groups than the "
                      f"column has now (it changed since): compute them again in {where}.")
    return key, None


def marker_params(adata, key: str) -> dict:
    try:
        params = dict(adata.uns[key].get("params", {}))
    except (AttributeError, TypeError):
        params = {}
    return {"method": str(params.get("method") or "wilcoxon"),
            "corr_method": str(params.get("corr_method") or "benjamini-hochberg")}


def marker_stats(adata, column: str, key: str, group: str, genes: list[str]) -> dict[str, dict]:
    """Per gene of ``genes`` in ``group``'s stored marker result: its rank in the
    stored (score) order, log fold change, adjusted p and the fraction of cells
    expressing it in the group and in the rest (stored pts, else counted)."""
    from scview.core.marker_selection import _fractions, _lookup, _stored

    rgg = adata.uns[key]
    names = np.asarray(rgg["names"][group]).astype(str)
    pos = {g: i for i, g in enumerate(names)}

    def field(name: str) -> np.ndarray | None:
        try:
            return np.asarray(rgg[name][group], dtype=float)[: names.size]
        except (KeyError, TypeError, ValueError, IndexError):
            return None

    lfc, padj = field("logfoldchanges"), field("pvals_adj")
    pts, rest = _stored(rgg, "pts"), _stored(rgg, "pts_rest")
    if pts is None:
        counted = _fractions(adata, column, [str(g) for g in rgg["names"].dtype.names])
        pts, rest = counted if counted is not None else (None, None)
    pin, pout = _lookup(pts, group, names), _lookup(rest, group, names)

    def at(arr: np.ndarray | None, i: int | None) -> float | None:
        return None if arr is None or i is None or i >= arr.size else num(arr[i])

    out = {}
    for g in genes:
        i = pos.get(g)
        out[g] = {"gene": g, "rank": None if i is None else i + 1, "logfc": at(lfc, i),
                  "padj": None if padj is None or i is None else p_value(padj[i]),
                  "pct_in": at(pin, i), "pct_out": at(pout, i)}
    return out


def p_value(v) -> float | None:
    """A p-value for JSON: finite, not rounded away (1e-300 stays 1e-300)."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def marker_db_status(dest) -> tuple[object | None, str | None]:
    """(the merged marker database on disk, None), else (None, why not). Never
    downloads: the report reads only what the database check fetched."""
    from pathlib import Path

    from scview.core import marker_db

    where = WHERE["marker_db"]
    try:
        db = marker_db.load(Path(dest)) if dest else None
    except Exception as exc:  # an unreadable table: the check would refetch it
        logger.warning("marker databases unreadable: %s", exc)
        db = None
    if db is None:
        return None, (f"Not downloaded yet: the marker databases are fetched the first time {where} "
                      "runs. The report reads them only once they are on disk, it never downloads.")
    return db, None


def scview_version() -> str:
    from scview.core.provenance import _scview_version

    return _scview_version()


def meta(adaptor, dataset_name: str | None) -> dict:
    adata = adaptor.adata
    return {
        "dataset": {"name": dataset_name or "dataset", "n_cells": int(adata.n_obs),
                    "n_genes": int(adata.n_vars)},
        "scview_version": scview_version(),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def methods_section(adaptor, report_methods: list[str]) -> dict:
    """The recorded analysis steps, as the co-pilot's methods writer sees them
    (``assistant._methods_skeleton``), and how each report section was made."""
    from scview.core.assistant import _methods_skeleton

    lines = [ln.strip() for ln in _methods_skeleton(adaptor).splitlines() if ln.strip()]
    steps = [ln[2:] for ln in lines if ln.startswith("- ")]
    head = [ln for ln in lines if not ln.startswith("- ") and not ln.startswith("Recorded analysis steps")]
    caption = (f"{plural(len(steps), 'analysis step')} recorded in the dataset's provenance, in order."
               if steps else "No processing steps are recorded in the dataset's provenance.")
    return section(
        "methods", "Methods",
        caption=caption,
        method=("Read from the dataset's provenance record (the digest the co-pilot's methods writer "
                "uses), with how each section of this report was computed; nothing is inferred."),
        data={"summary": head, "steps": steps, "sections": [m for m in report_methods if m]},
    )


def sections_methods(sections: list[dict]) -> list[str]:
    return [f"{s['title']}: {s['method']}" for s in sections if s.get("method")]


def cluster_report(adaptor, column: str, value: str, **kwargs) -> dict:
    """The cluster report (``core/report_cluster.py``)."""
    from scview.core.report_cluster import build

    return build(adaptor, column, value, **kwargs)


def gene_report(adaptor, gene: str, groupby: str | None = None, **kwargs) -> dict:
    """The gene report (``core/report_gene.py``)."""
    from scview.core.report_gene import build

    return build(adaptor, gene, groupby, **kwargs)
