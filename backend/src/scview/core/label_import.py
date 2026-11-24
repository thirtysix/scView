"""Per-cell labels from a table (an author annotation from GEO, a label file from a
collaborator) into a user annotation column (``core/user_labels.py``).

Cells are matched by name (the table's barcode column against the dataset's obs
names, optionally with a "-1"-style suffix removed on both sides) or, with no
barcode column, by order, which needs exactly one row per cell. The import says how
many rows matched, which table rows name no cell ("extra") and which cells the table
leaves out, and is refused when too small a share of the table's rows match: a
barcode format that differs (a sample prefix, a suffix) labels nothing, it does not
label a few cells and leave the rest silently "Unassigned".

By order, a matching row count is not enough: a table written with its samples in
another order (B then A where the dataset has A then B) has the right count and labels
every cell wrongly. When the table carries a sample column (named, or any other column
whose values are a column of the dataset, found by itself) the rows are checked sample
by sample against the dataset's column, and the import is refused with the samples
that do not line up. Into an existing column the answer also says how many cells
change label (``n_changed``).

QA round 7 (Garrido-Trigo 2023, 46,700 cells, 91 labels) took 91 ``POST /labels``
calls by cell indices and an order join checked by hand; this is one call.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from scview.core import provenance
from scview.core.obs_writes import forget_column_results
from scview.core.user_labels import (
    MAX_LABELS, UI_MAX_GROUPS, UNASSIGNED, LabelError, _set_user_columns, _user_column,
    clean_column_name, clean_label, user_columns,
)

logger = logging.getLogger(__name__)

MIN_MATCH = 0.9  # share of the table's labelled rows that must name a cell (by name)
MAX_TABLE_BYTES = 512 * 1024 * 1024  # a table as read (after gunzip)
MAX_ROWS = 5_000_000
EXAMPLES = 5  # names listed for extra rows and cells left out
MAX_OTHERS = 4  # by order: other table columns looked at for a sample column
MAX_SAMPLES = 1000  # a sample column has at most this many values
SAMPLES_LISTED = 4  # samples named when the order does not line up
TOP_LABELS = 20  # labels listed in the answer
# Read as "no label": the cell keeps its value (UNASSIGNED in a new column).
BLANK = frozenset({"", "NA", "N/A", "NaN", "nan", "None", "null", "NULL"})
SUFFIX_RE = r"[-.]\d+$"  # Cell Ranger's "-1" (a merge's "-2"), R make.unique's ".1"


@dataclass
class LabelTable:
    """Rows of labels: ``barcodes`` None means one row per cell in the dataset's order."""

    labels: list[Any]
    barcodes: list[Any] | None = None
    source: dict[str, Any] = field(default_factory=dict)  # file, columns, separator: for the record
    # By order: the table's other columns (a sample column checks the order sample by sample);
    # ``sample_column`` names the one to use, else one is looked for among them.
    others: dict[str, list[Any]] = field(default_factory=dict)
    sample_column: str | None = None


def _separator(first_line: str, filename: str) -> str:
    """The header row's separator: a tab, else a comma, else a semicolon; a one-column
    header goes by the file's extension."""
    for sep in ("\t", ",", ";"):
        if sep in first_line:
            return sep
    return "," if filename.lower().removesuffix(".gz").endswith(".csv") else "\t"


def read_table(data: bytes, filename: str = "", barcode_column: str | None = None,
               label_column: str | None = None, by_order: bool = False,
               sample_column: str | None = None) -> LabelTable:
    """A CSV or TSV (gzipped or not) with a header row as a LabelTable. Barcodes: the
    ``barcode_column`` (default: the first column) unless ``by_order``; labels: the
    ``label_column`` (default: the one other column). By order the other columns are kept for
    the sample check (``sample_column``: the one to use)."""
    if data[:2] == b"\x1f\x8b":
        try:
            with gzip.GzipFile(fileobj=io.BytesIO(data)) as fh:
                data = fh.read(MAX_TABLE_BYTES + 1)
        except (OSError, EOFError) as exc:
            raise LabelError(f"Could not read the gzipped table: {exc}.") from None
    if len(data) > MAX_TABLE_BYTES:
        raise LabelError(f"The table is larger than {MAX_TABLE_BYTES // 2**20} MB.", status=413)
    text = data.decode("utf-8-sig", "replace")
    first = text.split("\n", 1)[0]
    sep = _separator(first, filename)
    try:
        df = pd.read_csv(io.StringIO(text), sep=sep, dtype=str, keep_default_na=False, na_filter=False)
    except (ValueError, pd.errors.ParserError) as exc:
        raise LabelError(f"Could not read the table: {str(exc).splitlines()[0]}") from None
    cols = [str(c) for c in df.columns]
    df.columns = cols
    if df.empty or not cols:
        raise LabelError("The table has no rows.")
    if barcode_column and barcode_column not in cols:
        raise LabelError(f"No column '{barcode_column}' in the table (columns: {', '.join(cols[:12])}).")
    if label_column and label_column not in cols:
        raise LabelError(f"No column '{label_column}' in the table (columns: {', '.join(cols[:12])}).")
    if sample_column and sample_column not in cols:
        raise LabelError(f"No column '{sample_column}' in the table (columns: {', '.join(cols[:12])}).")
    bc = None if by_order else (barcode_column or cols[0])
    if not label_column:
        rest = [c for c in cols if c != bc]
        if len(rest) != 1:
            raise LabelError(
                f"Name the label column: the table has {len(cols)} columns ({', '.join(cols[:12])}).")
        label_column = rest[0]
    if label_column == bc:
        raise LabelError("The label column and the barcode column must differ.")
    if sample_column and sample_column == label_column:
        raise LabelError("The label column and the sample column must differ.")
    source = {"file": filename or None, "barcode_column": bc, "label_column": label_column,
              "sep": sep}
    others: dict[str, list[Any]] = {}
    if bc is None:
        for c in ([sample_column] if sample_column else [c for c in cols if c != label_column][:MAX_OTHERS]):
            others[c] = df[c].tolist()
    return LabelTable(labels=df[label_column].tolist(),
                      barcodes=None if bc is None else df[bc].tolist(), source=source,
                      others=others, sample_column=sample_column or None)


def _clean_values(values: list[Any]) -> np.ndarray:
    """Labels trimmed; blank ones None. A label scView cannot store is refused with its row."""
    out = np.empty(len(values), dtype=object)
    cache: dict[str, str] = {}
    for i, v in enumerate(values):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            out[i] = None
            continue
        s = str(v).strip()
        if s in BLANK:
            out[i] = None
            continue
        if s not in cache:
            try:
                cache[s] = clean_label(s)
            except LabelError as exc:
                raise LabelError(f"Row {i + 1}: {exc}") from None
        out[i] = cache[s]
    return out


def _strip(names: pd.Index) -> pd.Index:
    return pd.Index(names.astype(str).str.replace(SUFFIX_RE, "", regex=True))


def _examples(values) -> list[str]:
    return [str(v) for v in list(values)[:EXAMPLES]]


def _hints(obs: pd.Index, keys: pd.Series, share: float, strip_suffix: bool) -> dict[str, str]:
    """A suggestion when most rows miss: a suffix to strip, or a sample prefix the dataset's
    names carry ("CTRL_AAAC..." vs "AAAC..."), which is never stripped for the user: the
    same barcode is often in several samples."""
    if not len(keys):
        return {}
    if not strip_suffix:
        alt = _strip(pd.Index(keys)).isin(set(_strip(obs)))
        if alt.mean() > share + 0.05:
            return {"hint": f"With a '-1' / '.1' suffix removed from both sides, {int(alt.sum()):,} of "
                            f"{len(keys):,} rows would match: import with strip_suffix."}
    core = obs.str.replace(r"^.*[_:#]", "", regex=True)
    if strip_suffix:
        core = _strip(core)
    hit = pd.Index(keys).isin(set(core))
    if hit.mean() > share + 0.05:
        first = obs[core.get_indexer([keys[hit].iloc[0]])[0]] if core.is_unique else obs[0]
        same = "" if core.is_unique else " The same barcode is in several samples, so it alone cannot say which cell."
        return {"hint": f"{int(hit.sum()):,} of {len(keys):,} rows match once the dataset's prefix is set aside "
                        f"(e.g. '{first}'): write the cell names as the dataset has them.{same}"}
    return {}


def plan(adata, table: LabelTable, *, strip_suffix: bool = False, min_match: float = MIN_MATCH
         ) -> tuple[np.ndarray, dict[str, Any]]:
    """Each cell's imported label (None: none) and the match report; LabelError when
    the table cannot be matched (too few rows match, wrong row count, conflicting rows)."""
    n_rows = len(table.labels)
    if n_rows == 0:
        raise LabelError("The table has no rows.")
    if n_rows > MAX_ROWS:
        raise LabelError(f"The table has {n_rows:,} rows (at most {MAX_ROWS:,}).", status=413)
    labels = _clean_values(table.labels)
    blank = np.array([v is None for v in labels])
    n_cells = int(adata.n_obs)
    report: dict[str, Any] = {"n_rows": n_rows, "n_blank": int(blank.sum()), "n_cells": n_cells}

    if table.barcodes is None:
        report["match"] = "order"
        if n_rows != n_cells:
            raise LabelError(
                f"The table has {n_rows:,} rows and the dataset {n_cells:,} cells: matching by order "
                "needs one row per cell, in the dataset's order. Match by barcode instead, or find the "
                "cells that differ first (count the rows per sample on both sides).")
        report["order_check"] = order_check(adata, table)
        per_cell = labels
        report.update(n_matched=int((~blank).sum()), n_extra=0, n_duplicates=0)
    else:
        report["match"] = "name"
        if len(table.barcodes) != n_rows:
            raise LabelError("Give one barcode per label.")
        obs = pd.Index(adata.obs_names.astype(str))
        keys = pd.Index([str(b).strip() for b in table.barcodes])
        if strip_suffix:
            obs, keys = _strip(obs), _strip(keys)
        if not obs.is_unique:
            dup = obs[obs.duplicated()][0]
            raise LabelError(
                f"The dataset's cell names are not unique ('{dup}' more than once"
                + (" with the suffix removed" if strip_suffix else "")
                + "): match by order instead, with a table of one row per cell.")
        frame = pd.DataFrame({"key": keys, "label": labels})[~blank]
        dups = frame[frame["key"].duplicated(keep=False)]
        if len(dups):
            clash = dups.groupby("key")["label"].nunique()
            clash = clash[clash > 1]
            if len(clash):
                k = clash.index[0]
                both = sorted(map(str, dups.loc[dups["key"] == k, "label"].unique()))[:2]
                raise LabelError(
                    f"{len(clash):,} barcodes have more than one label in the table, e.g. '{k}' "
                    f"('{both[0]}' and '{both[1]}'). Keep one row per cell: a barcode repeated with "
                    "other labels is most often the same barcode in two samples, so name the cells as "
                    "the dataset does (sample prefix and all), or match by order.")
        n_dup = int(frame["key"].duplicated().sum())
        frame = frame.drop_duplicates("key")
        where = obs.get_indexer(frame["key"])
        found = where >= 0
        n_labelled = len(frame)
        share = float(found.sum()) / n_labelled if n_labelled else 0.0
        report.update(n_matched=int(found.sum()), n_extra=int((~found).sum()), n_duplicates=n_dup,
                      share_matched=round(share, 4))
        if (~found).any():
            report["extra_examples"] = _examples(frame["key"][~found])
        if share < 1.0:
            report.update(_hints(obs, frame["key"], share, strip_suffix))
        if n_labelled == 0 or share < min_match:
            seen = _examples(obs[:EXAMPLES])
            raise LabelError(
                f"Only {int(found.sum()):,} of {n_labelled:,} labelled rows ({share:.0%}) name a cell of "
                f"this dataset (at least {min_match:.0%} must): the barcodes differ, e.g. the table's "
                f"{', '.join(repr(x) for x in report.get('extra_examples', [])[:3])} vs the dataset's "
                f"{', '.join(repr(x) for x in seen[:3])}. "
                + (report.get("hint") or "Fix the barcode column, or lower min_match if the table "
                   "really holds other cells too."))
        per_cell = np.full(n_cells, None, dtype=object)
        per_cell[where[found]] = frame["label"].to_numpy()[found]

    got = np.array([v is not None for v in per_cell])
    report["n_unmatched_cells"] = int((~got).sum())
    if (~got).any():
        report["unmatched_examples"] = _examples(adata.obs_names[~got])
    if not got.any():
        raise LabelError("No cell gets a label from this table.")
    names = pd.unique(per_cell[got])
    if len(names) > MAX_LABELS:
        raise LabelError(f"The table has {len(names):,} labels (at most {MAX_LABELS}): its label column "
                         "looks per-cell, not a grouping.")
    return per_cell, report


def _as_text(values) -> np.ndarray:
    out = pd.Series(values, dtype=object).astype(str).str.strip().to_numpy(dtype=object)
    return out


def _sample_columns(adata, values: set[str]) -> list[str]:
    """The dataset's columns that hold every one of ``values`` (a sample column's values)."""
    out = []
    for c in adata.obs.columns:
        col = adata.obs[c]
        if isinstance(col.dtype, pd.CategoricalDtype):
            have = set(map(str, col.cat.categories))
        elif col.dtype == object or pd.api.types.is_string_dtype(col.dtype):
            if col.nunique(dropna=True) > MAX_SAMPLES:
                continue
            have = set(map(str, col.dropna().unique()))
        else:
            continue
        if values <= have:
            out.append(str(c))
    return out


def _runs(values: np.ndarray) -> list[tuple[str, int, int]]:
    """(value, first row, last row) of each run of equal values, in order (1-based rows)."""
    out: list[tuple[str, int, int]] = []
    start = 0
    for i in range(1, len(values) + 1):
        if i == len(values) or values[i] != values[start]:
            out.append((str(values[start]), start + 1, i))
            start = i
    return out


def _rows_text(runs: list[tuple[str, int, int]], name: str) -> str:
    """Where a sample's rows are: its first three runs ('1-845, 2,001-2,010, ...')."""
    mine = [(a, b) for v, a, b in runs if v == name]
    parts = [f"{a:,}-{b:,}" if a != b else f"{a:,}" for a, b in mine[:3]]
    return (", ".join(parts) + (", ..." if len(mine) > 3 else "")) or "none"


def order_check(adata, table: LabelTable) -> dict[str, Any] | None:
    """By order: the table's sample column against the dataset's, row by row. None when the
    table has no column whose values are a column of the dataset; the check's summary when the
    rows line up; LabelError naming the samples that do not."""
    n = int(adata.n_obs)
    chosen: tuple[str, str, np.ndarray, np.ndarray] | None = None
    best = -1.0
    for tcol, raw in table.others.items():
        if len(raw) != n:
            continue
        tv = _as_text(raw)
        vals = set(tv) - BLANK
        if not 2 <= len(vals) <= MAX_SAMPLES:
            if tcol == table.sample_column:
                raise LabelError(f"The table's '{tcol}' has {len(vals):,} values: a sample column needs 2 to "
                                 f"{MAX_SAMPLES:,}.")
            continue
        for dcol in _sample_columns(adata, vals):
            dv = _as_text(adata.obs[dcol].astype(object).to_numpy())
            agree = float(np.mean(tv == dv))
            if agree > best:
                best, chosen = agree, (tcol, dcol, tv, dv)
    if chosen is None:
        if table.sample_column:
            raise LabelError(f"The values of the table's '{table.sample_column}' are not all in any column of "
                             "the dataset, so the order cannot be checked sample by sample.")
        return None
    tcol, dcol, tv, dv = chosen
    names = list(pd.unique(dv))
    if best == 1.0:
        return {"table_column": tcol, "dataset_column": dcol, "n_samples": len(names), "lined_up": True}
    t_runs, d_runs = _runs(tv), _runs(dv)
    t_count = pd.Series(tv).value_counts()
    bad = [s for s in pd.unique(np.concatenate([dv, tv])) if not np.all(tv[dv == s] == s)
           or not np.all(dv[tv == s] == s)]
    lines = []
    for s in bad[:SAMPLES_LISTED]:
        lines.append(f"{s}: {int(t_count.get(s, 0)):,} table rows at {_rows_text(t_runs, s)}, "
                     f"{int((dv == s).sum()):,} dataset cells at {_rows_text(d_runs, s)}")
    more = f", and {len(bad) - SAMPLES_LISTED} more" if len(bad) > SAMPLES_LISTED else ""
    same_counts = all(int(t_count.get(s, 0)) == int((dv == s).sum()) for s in names)
    why = ("the counts per sample agree, so the samples are in another order" if same_counts
           else "the rows per sample differ")
    raise LabelError(
        f"Matched by order, the table's '{tcol}' does not line up with the dataset's '{dcol}': "
        f"{len(bad)} of {len(set(names) | set(tv))} samples are on other rows ({why}): "
        f"{'; '.join(lines)}{more}. Sort the table in the dataset's order, sample by sample, or match "
        "by barcode.")


def fingerprint(per_cell: np.ndarray) -> str:
    """A short hash of the cells' imported labels, so a replay can say it matched."""
    text = "\x1e".join("" if v is None else str(v) for v in per_cell)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def import_labels(adata, column: str, table: LabelTable, *, new_column: bool = True,
                  strip_suffix: bool = False, min_match: float = MIN_MATCH,
                  dry_run: bool = False) -> dict[str, Any]:
    """Write the table's labels into the user annotation column ``column`` (a new one:
    every other cell UNASSIGNED; an existing one: the cells the table leaves out keep
    their value). Checked whole before ``adata`` changes; ``dry_run`` only reports."""
    if new_column:
        column = clean_column_name(column, adata.obs.columns)
        base = pd.Categorical.from_codes(np.zeros(adata.n_obs, dtype=np.int8), categories=[UNASSIGNED])
    else:
        base = _user_column(adata, column)
    per_cell, report = plan(adata, table, strip_suffix=strip_suffix, min_match=min_match)
    got = np.array([v is not None for v in per_cell])
    values = np.asarray(base.astype(object), dtype=object)
    before = values.copy()
    values[got] = per_cell[got]
    n_changed = int(sum(1 for a, b in zip(before[got], per_cell[got]) if a != b))
    imported = list(pd.unique(per_cell[got]))
    left = set(values[~got])
    kept = [c for c in base.categories if c in left]
    cats = [*kept, *(v for v in imported if v not in set(kept))]
    if len(cats) > MAX_LABELS:
        raise LabelError(f"'{column}' would have {len(cats):,} labels (at most {MAX_LABELS}).")
    counts = pd.Series(per_cell[got]).value_counts()
    out: dict[str, Any] = {
        "column": column, "created": bool(new_column), **report, "n_labels": len(cats),
        "n_changed": n_changed,
        # cells left Unassigned after the import (a new column's unmatched cells, mostly)
        "n_unassigned": int(sum(1 for v in values if v == UNASSIGNED)),
        "labels": [{"label": str(k), "n_cells": int(v)} for k, v in counts.head(TOP_LABELS).items()],
    }
    if len(counts) > TOP_LABELS:
        out["n_imported_labels"] = int(len(counts))
    if len(cats) > UI_MAX_GROUPS:
        out["note"] = (f"'{column}' has {len(cats)} labels: the UI's group pickers list columns of at "
                       f"most {UI_MAX_GROUPS}, so they leave it out; filters and tests still take it.")
    if dry_run:
        out["dry_run"] = True
        return out
    adata.obs[column] = pd.Categorical(values, categories=cats)
    if new_column:
        _set_user_columns(adata, [*user_columns(adata), column])
    forget_column_results(adata, column)
    params = {"column": column, "new_column": bool(new_column), "match": report["match"],
              "strip_suffix": bool(strip_suffix), "n_rows": report["n_rows"], "n_matched": report["n_matched"],
              "n_labels": len(cats), "n_changed": n_changed, "fingerprint": fingerprint(per_cell),
              **{k: v for k, v in table.source.items() if v is not None}}
    if report.get("order_check"):
        params["order_checked_by"] = [report["order_check"]["table_column"], report["order_check"]["dataset_column"]]
    try:
        provenance.record_step(adata, step="import_labels", tool="scview.label_import", params=params)
    except Exception as e:  # provenance must never block the edit
        logger.warning("Could not record import_labels provenance: %s", e)
    return out
