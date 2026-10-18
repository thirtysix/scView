"""A fixed colour range for an expression layer, for marker plots that can be
compared with each other.

Scanpy-style dot plots and heatmaps scale each gene to 0-1 across the groups
shown, so a colour means something different in every plot. A fixed range
needs an upper end that depends on the layer alone, not on the genes or groups
asked for: here the 99th percentile of the layer's non-zero values in an evenly
spaced sample of cells, rounded up to a round number (Kang PBMC log-normalised:
5.0; paul15 mouse: 4.5). The lower end is 0, or the 1st percentile of all
values for a layer with negative values (scaled data).
"""

from __future__ import annotations

import math

import numpy as np
import scipy.sparse as sp

SAMPLE_CELLS = 5000
UPPER_PERCENTILE = 99.0
LOWER_PERCENTILE = 1.0


def nice_ceil(x: float) -> float:
    """``x`` rounded up to a round number: steps of 0.5 from 2 up, else two
    significant figures (0.0137 -> 0.014)."""
    if not math.isfinite(x) or x <= 0:
        return 1.0
    if x >= 2:
        return math.ceil(x * 2 - 1e-9) / 2
    step = 10 ** (math.floor(math.log10(x)) - 1)
    return round(math.ceil(x / step - 1e-9) * step, 10)


def _nice_floor(x: float) -> float:
    return -nice_ceil(-x) if x < 0 else 0.0


def layer_range(X) -> tuple[float, float]:
    """(low, high) colour range of a cells x genes matrix (sparse, dense or
    backed): see the module docstring."""
    n = X.shape[0]
    if n == 0 or X.shape[1] == 0:
        return 0.0, 1.0
    rows = np.unique(np.linspace(0, n - 1, min(SAMPLE_CELLS, n)).astype(int))
    sub = X[rows]
    if sp.issparse(sub):
        values = np.asarray(sub.data, dtype=np.float64)
        has_zeros = sub.nnz < sub.shape[0] * sub.shape[1]
    else:
        values = np.asarray(sub.toarray() if hasattr(sub, "toarray") else sub, dtype=np.float64).ravel()
        has_zeros = False
    values = values[np.isfinite(values)]
    positive = values[values > 0]
    high = nice_ceil(float(np.percentile(positive, UPPER_PERCENTILE))) if positive.size else 1.0
    low = 0.0
    if values.size and values.min() < 0:
        pool = np.concatenate([values, [0.0]]) if has_zeros else values
        low = _nice_floor(float(np.percentile(pool, LOWER_PERCENTILE)))
    return low, high
