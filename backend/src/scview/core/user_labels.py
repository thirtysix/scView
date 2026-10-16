"""Labelling selected cells by hand ("Label selection…" in Unified View).

Labels go only into user annotation columns: obs columns that this module
created, listed by name in ``uns[USER_COLUMNS_KEY]`` (a JSON list). A clustering
or an imported annotation is never written to. A new column starts with every
cell ``UNASSIGNED``.

Each labelling returns the previous value of every cell it changed, grouped by
value, and :func:`restore` puts those values back: that is the undo. Undoing the
labelling that created a column removes the column once no labels are left in it.
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

from scview.core import provenance
from scview.core.obs_writes import forget_column_results

logger = logging.getLogger(__name__)

USER_COLUMNS_KEY = "scview_user_annotations"
UNASSIGNED = "Unassigned"

LABEL_MAX = 80
COLUMN_MAX = 64
# Pickers and markers handle at most 100 groups per column.
MAX_LABELS = 100
# A letter first, then letters, digits, space, "_", "-" or ".". No "/": h5py
# would read it as a path separator when the file is written.
_COLUMN_RE = re.compile(r"[A-Za-z][A-Za-z0-9 _.\-]*")
# The pipeline's own columns (scview_leiden_r0.5, scview_dpt_pseudotime, ...).
RESERVED_PREFIX = "scview_"
# Control characters, invisible format characters (bidi overrides, zero-width
# spaces) and line/paragraph separators.
_BAD_CATEGORIES = {"Cc", "Cf", "Cs", "Zl", "Zp"}


class LabelError(ValueError):
    """A labelling request that cannot be served; ``status`` is the HTTP code."""

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


def user_columns(adata) -> list[str]:
    """Names of the user annotation columns the dataset still has."""
    raw = adata.uns.get(USER_COLUMNS_KEY)
    try:
        names = json.loads(raw) if isinstance(raw, str) else []
    except ValueError:
        names = []
    if not isinstance(names, list):
        return []
    return [n for n in names if isinstance(n, str) and n in adata.obs.columns]


def _set_user_columns(adata, names: list[str]) -> None:
    adata.uns[USER_COLUMNS_KEY] = json.dumps(names)


def clean_label(value: object) -> str:
    """A label trimmed, or LabelError saying why it can't be used."""
    if not isinstance(value, str) or not value.strip():
        raise LabelError("Type a label for the selected cells.")
    label = value.strip()
    if len(label) > LABEL_MAX:
        raise LabelError(f"Label is too long (max {LABEL_MAX} characters).")
    if any(unicodedata.category(ch) in _BAD_CATEGORIES for ch in label):
        raise LabelError("Label must not contain control or invisible characters.")
    return label


def clean_column_name(value: object, existing: Iterable[str]) -> str:
    """A new column's name trimmed, or LabelError saying why it can't be used."""
    name = value.strip() if isinstance(value, str) else ""
    if not name:
        raise LabelError("Name the new column.")
    if len(name) > COLUMN_MAX:
        raise LabelError(f"Column name is too long (max {COLUMN_MAX} characters).")
    if not _COLUMN_RE.fullmatch(name):
        raise LabelError(
            "Column name must start with a letter and use only letters, digits, "
            "spaces, '_', '-' or '.'."
        )
    if name.lower().startswith(RESERVED_PREFIX):
        raise LabelError(f"Names starting with '{RESERVED_PREFIX}' are kept for scView's own columns.")
    taken = {str(c).lower() for c in existing}
    if name.lower() in taken:
        raise LabelError(f"A column named '{name}' already exists.", status=409)
    return name


def _cell_indices(indices: Iterable[int], n_obs: int) -> np.ndarray:
    idx = np.unique(np.asarray(list(indices), dtype=np.int64))
    if idx.size == 0:
        raise LabelError("Select some cells first.")
    if idx[0] < 0 or idx[-1] >= n_obs:
        raise LabelError("The selection does not match this dataset's cells; select them again.")
    return idx


def _user_column(adata, column: str) -> pd.Categorical:
    if column not in adata.obs.columns:
        raise LabelError(f"Column '{column}' not found.", status=404)
    if column not in user_columns(adata):
        raise LabelError(
            f"'{column}' is not a user annotation column. Labels go into a column "
            "made for them: pick one, or start a new one."
        )
    col = adata.obs[column]
    if not isinstance(col.dtype, pd.CategoricalDtype):
        col = col.astype(str).astype("category")
    return pd.Categorical(col)


def _assign(cat: pd.Categorical, idx: np.ndarray, value: str | None) -> pd.Categorical:
    """``cat`` with the cells at ``idx`` set to ``value`` (None: missing)."""
    cats = list(cat.categories)
    codes = np.array(cat.codes, dtype=np.int64)  # int8 codes would overflow past 127
    if value is None:
        codes[idx] = -1
    else:
        if value not in cats:
            cats.append(value)
        codes[idx] = cats.index(value)
    return pd.Categorical.from_codes(codes, categories=cats)


def _previous(cat: pd.Categorical, idx: np.ndarray) -> list[dict[str, Any]]:
    """The cells' current values grouped by value: what an undo restores."""
    codes = np.asarray(cat.codes)[idx]
    groups = []
    for code in np.unique(codes):
        value = None if code < 0 else str(cat.categories[code])
        groups.append({"value": value, "indices": idx[codes == code].tolist()})
    return groups


def _record(adata, step: str, params: dict[str, Any]) -> None:
    try:
        provenance.record_step(adata, step=step, tool="scview.label_selection", params=params)
    except Exception as e:  # provenance must never block the edit
        logger.warning("Could not record %s provenance: %s", step, e)


def label_cells(
    adata, column: str, label: object, indices: Iterable[int], new_column: bool
) -> dict[str, Any]:
    """Give the cells at ``indices`` the label ``label`` in a user annotation column.

    With ``new_column`` the column is created (every other cell UNASSIGNED);
    otherwise it must be an existing user annotation column. Everything is
    validated before ``adata`` changes, so a LabelError leaves it untouched.
    """
    clean = clean_label(label)
    idx = _cell_indices(indices, adata.n_obs)
    if new_column:
        column = clean_column_name(column, adata.obs.columns)
        cat = pd.Categorical.from_codes(np.zeros(adata.n_obs, dtype=np.int8), categories=[UNASSIGNED])
    else:
        cat = _user_column(adata, column)
    if clean not in cat.categories and len(cat.categories) >= MAX_LABELS:
        raise LabelError(f"'{column}' already has {MAX_LABELS} labels; reuse one or start a new column.")
    previous = _previous(cat, idx)
    names = user_columns(adata)

    adata.obs[column] = _assign(cat, idx, clean).remove_unused_categories()
    if new_column:
        _set_user_columns(adata, [*names, column])
    forget_column_results(adata, column)
    _record(adata, "label_cells", {
        "column": column, "label": clean, "n_cells": int(idx.size), "new_column": new_column,
    })
    return {
        "column": column,
        "label": clean,
        "n_cells": int(idx.size),
        "created": new_column,
        "previous": previous,
    }


def restore(
    adata, column: str, previous: list[dict[str, Any]], created: bool
) -> dict[str, Any]:
    """Undo a labelling: give each group of cells in ``previous`` its old value.

    When the undone labelling created the column (``created``) and only
    UNASSIGNED is left in it, the column is removed.
    """
    cat = _user_column(adata, column)
    groups: list[tuple[np.ndarray, str | None]] = []
    for group in previous:
        value = group.get("value")
        groups.append((
            _cell_indices(group.get("indices") or [], adata.n_obs),
            None if value is None else clean_label(value),
        ))
    if not groups:
        raise LabelError("Nothing to undo.")

    for idx, value in groups:
        cat = _assign(cat, idx, value)
    cat = cat.remove_unused_categories()
    n_cells = int(sum(idx.size for idx, _ in groups))
    dropped = created and list(cat.categories) == [UNASSIGNED] and not (cat.codes < 0).any()
    if dropped:
        del adata.obs[column]
        _set_user_columns(adata, [c for c in user_columns(adata) if c != column])
    else:
        adata.obs[column] = cat
    forget_column_results(adata, column)
    _record(adata, "undo_label_cells", {
        "column": column, "n_cells": n_cells, "removed_column": dropped,
    })
    return {"column": column, "n_cells": n_cells, "removed": dropped}
