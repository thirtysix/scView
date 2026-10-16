"""Cells matching a filter on genes and metadata, used as a selection.

A filter is a list of items joined by AND or OR; an item is one condition or a
group of conditions with its own AND/OR (one level of grouping, so
``CD14 > 1 AND (stim = STIM OR cell_type in {B, NK})`` can be written, deeper
nesting cannot). Conditions:

- ``gene``: a gene's expression on the given layer above or below a value;
- ``category``: an obs column's value in a set (compared as text, as the
  frontend lists them);
- ``range``: a numeric obs column within [min, max], either end open.

A missing value (NaN) matches no condition. The result is a boolean mask over
the cells, sent to the frontend as a packed bitmask (``mask_to_base64``).
"""

from __future__ import annotations

import base64
from typing import Annotated, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, model_validator

MAX_ITEMS = 20
MAX_CONDITIONS = 40  # in the whole filter, groups included
MAX_VALUES = 200
MAX_NAME = 256


class FilterError(ValueError):
    """A filter that can't be evaluated on this dataset. ``status`` is the HTTP
    code: 404 for a gene, column or layer it doesn't have, 400 otherwise."""

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


class GeneCondition(BaseModel):
    kind: Literal["gene"]
    gene: str = Field(min_length=1, max_length=MAX_NAME)
    op: Literal[">", ">=", "<", "<="]
    value: float = Field(allow_inf_nan=False)


class CategoryCondition(BaseModel):
    kind: Literal["category"]
    column: str = Field(min_length=1, max_length=MAX_NAME)
    values: list[Annotated[str, Field(max_length=MAX_NAME)]] = Field(min_length=1, max_length=MAX_VALUES)


class RangeCondition(BaseModel):
    kind: Literal["range"]
    column: str = Field(min_length=1, max_length=MAX_NAME)
    min: float | None = Field(default=None, allow_inf_nan=False)
    max: float | None = Field(default=None, allow_inf_nan=False)

    @model_validator(mode="after")
    def _bounds(self) -> RangeCondition:
        if self.min is None and self.max is None:
            raise ValueError("A range needs a minimum, a maximum, or both.")
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError("The range's minimum is above its maximum.")
        return self


Condition = Annotated[GeneCondition | CategoryCondition | RangeCondition, Field(discriminator="kind")]


class ConditionGroup(BaseModel):
    kind: Literal["group"]
    op: Literal["and", "or"]
    conditions: list[Condition] = Field(min_length=1, max_length=MAX_ITEMS)


Item = Annotated[GeneCondition | CategoryCondition | RangeCondition | ConditionGroup, Field(discriminator="kind")]


class CellFilter(BaseModel):
    op: Literal["and", "or"]
    items: list[Item] = Field(min_length=1, max_length=MAX_ITEMS)
    # Expression layer the gene conditions read ("" = the dataset's default).
    layer: str = Field(default="", max_length=MAX_NAME)

    @model_validator(mode="after")
    def _size(self) -> CellFilter:
        n = sum(len(i.conditions) if isinstance(i, ConditionGroup) else 1 for i in self.items)
        if n > MAX_CONDITIONS:
            raise ValueError(f"At most {MAX_CONDITIONS} conditions per filter.")
        return self


def _conditions(spec: CellFilter):
    for item in spec.items:
        if isinstance(item, ConditionGroup):
            yield from item.conditions
        else:
            yield item


def _gene_values(adaptor, spec: CellFilter) -> dict[str, np.ndarray]:
    """Each gene's values on the filter's layer, read in one pass over the matrix."""
    genes = list(dict.fromkeys(c.gene for c in _conditions(spec) if isinstance(c, GeneCondition)))
    if not genes:
        return {}
    layer = spec.layer or adaptor.default_expression_layer()
    try:
        var_names = set(adaptor._var_names_for_layer(layer))
    except KeyError:
        raise FilterError(f"Layer '{layer}' not found.", 404) from None
    missing = [g for g in genes if g not in var_names]
    if missing:
        raise FilterError(f"Gene '{missing[0]}' is not in layer '{layer}'.", 404)
    expr = adaptor.get_expression(genes, layer=layer)
    return {g: expr[:, i] for i, g in enumerate(genes)}


def _column(obs: pd.DataFrame, name: str) -> pd.Series:
    if name not in obs.columns:
        raise FilterError(f"Column '{name}' not found.", 404)
    return obs[name]


def _category_mask(series: pd.Series, values: list[str]) -> np.ndarray:
    wanted = set(values)
    if isinstance(series.dtype, pd.CategoricalDtype):
        # Compare the categories' text once, then look the codes up (a missing
        # value has code -1 and is in no set).
        hit = np.array([str(c) in wanted for c in series.cat.categories], dtype=bool)
        if not hit.any():
            return np.zeros(len(series), dtype=bool)
        codes = series.cat.codes.to_numpy()
        return (codes >= 0) & hit[np.maximum(codes, 0)]
    notna = series.notna().to_numpy()
    return notna & series.astype(str).isin(wanted).to_numpy()


def _range_mask(series: pd.Series, cond: RangeCondition) -> np.ndarray:
    if pd.api.types.is_bool_dtype(series) or not pd.api.types.is_numeric_dtype(series):
        raise FilterError(f"Column '{cond.column}' is not numeric: pick values instead of a range.")
    v = series.to_numpy(dtype=np.float64, na_value=np.nan)
    mask = ~np.isnan(v)
    if cond.min is not None:
        mask &= v >= cond.min
    if cond.max is not None:
        mask &= v <= cond.max
    return mask


_OPS = {
    ">": np.greater,
    ">=": np.greater_equal,
    "<": np.less,
    "<=": np.less_equal,
}


def _condition_mask(cond, obs: pd.DataFrame, genes: dict[str, np.ndarray]) -> np.ndarray:
    if isinstance(cond, GeneCondition):
        return _OPS[cond.op](genes[cond.gene], cond.value)
    if isinstance(cond, CategoryCondition):
        return _category_mask(_column(obs, cond.column), cond.values)
    return _range_mask(_column(obs, cond.column), cond)


def _combine(op: str, masks: list[np.ndarray]) -> np.ndarray:
    return np.logical_and.reduce(masks) if op == "and" else np.logical_or.reduce(masks)


def evaluate(adaptor, spec: CellFilter) -> np.ndarray:
    """Boolean mask of the cells matching ``spec`` (length n_cells). Reads the
    dataset only: run it off the event loop."""
    obs = adaptor.adata.obs
    genes = _gene_values(adaptor, spec)
    masks = []
    for item in spec.items:
        if isinstance(item, ConditionGroup):
            masks.append(_combine(item.op, [_condition_mask(c, obs, genes) for c in item.conditions]))
        else:
            masks.append(_condition_mask(item, obs, genes))
    return np.asarray(_combine(spec.op, masks), dtype=bool)


def mask_to_base64(mask: np.ndarray) -> str:
    """Cell ``i`` is bit ``i % 8`` (least significant first) of byte ``i // 8``."""
    return base64.b64encode(np.packbits(mask, bitorder="little").tobytes()).decode("ascii")
