"""The stored order of a label column's categories.

The order of an obs categorical is what every view lists its groups by: the
legend, the violin and dot plots, the composition crosstab, and the colour codes
the embedding endpoint sends (so a group's palette colour follows its position,
as in scanpy). ``reorder`` sets it from an explicit order or a quick sort; a text
column becomes a categorical in that order. A scanpy ``uns['<column>_colors']``
list is permuted with it so it stays matched to its categories.
"""

from __future__ import annotations

import re
from typing import Literal

import numpy as np
import pandas as pd

Sort = Literal["natural", "size"]
_DIGITS = re.compile(r"(\d+)")


class OrderError(ValueError):
    """A request that cannot be served; ``status`` is the HTTP code."""

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


def natural_key(label: str) -> tuple:
    """Sort key putting "2" before "10" and "T cells 2" before "T cells 10",
    ignoring case."""
    parts = _DIGITS.split(str(label).lower())
    return tuple((0, int(p), "") if p.isdigit() else (1, 0, p) for p in parts if p != "")


def _categories(col: pd.Series) -> list:
    if isinstance(col.dtype, pd.CategoricalDtype):
        return list(col.cat.categories)
    if pd.api.types.is_bool_dtype(col) or not (
        col.dtype == object or pd.api.types.is_string_dtype(col)
    ):
        raise OrderError("Only a categorical or text column has an order to set.")
    return list(pd.unique(col.dropna()))


def reorder(adata, column: str, order: list[str] | None = None, sort: Sort | None = None) -> list[str]:
    """Set ``column``'s category order and return it (as strings).

    ``order`` lists every category once; ``sort`` is "natural" (A-Z, numbers by
    value) or "size" (most cells first). Raises OrderError before any change."""
    if column not in adata.obs.columns:
        raise OrderError(f"Column '{column}' not found.", status=404)
    col = adata.obs[column]
    cats = _categories(col)
    by_name = {str(c): c for c in cats}
    if len(by_name) != len(cats):
        raise OrderError(f"'{column}' has categories that read the same as text.")
    names = list(by_name)
    if order is not None:
        order = [str(o) for o in order]
        if len(order) != len(names) or set(order) != set(names):
            missing = sorted(set(names) - set(order))[:5]
            extra = sorted(set(order) - set(names))[:5]
            detail = "; ".join(p for p in (
                f"missing {', '.join(missing)}" if missing else "",
                f"not in the column: {', '.join(extra)}" if extra else "",
                "a category is listed twice" if not missing and not extra else "",
            ) if p)
            raise OrderError(f"The order must list each category of '{column}' once ({detail}).")
    elif sort == "natural":
        order = sorted(names, key=natural_key)
    elif sort == "size":
        counts = col.astype(object).map(lambda v: str(v) if pd.notna(v) else None).value_counts()
        # Most cells first; ties (and empty categories) keep their current place.
        order = sorted(names, key=lambda n: (-int(counts.get(n, 0)), names.index(n)))
    else:
        raise OrderError("Give either an order or a sort ('natural' or 'size').")

    new = [by_name[n] for n in order]
    if isinstance(col.dtype, pd.CategoricalDtype):
        adata.obs[column] = col.cat.reorder_categories(new)
        perm = [cats.index(c) for c in new]
    else:
        adata.obs[column] = pd.Categorical(col, categories=new)
        perm = None
    key = f"{column}_colors"
    colors = adata.uns.get(key)
    if perm is not None and colors is not None and len(colors) == len(perm):
        adata.uns[key] = np.asarray(colors, dtype=object)[perm].astype(str)
    return order
