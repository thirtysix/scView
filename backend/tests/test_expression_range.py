"""A layer's fixed colour range for the marker plots (core/expression_range.py)."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from scview.core import expression_range as er


@pytest.mark.parametrize(
    ("x", "expected"),
    [(4.84, 5.0), (4.39, 4.5), (2.0, 2.0), (1.37, 1.4), (0.0137, 0.014), (0.0, 1.0), (float("nan"), 1.0)],
)
def test_nice_ceil_rounds_up_to_a_round_number(x, expected):
    assert er.nice_ceil(x) == pytest.approx(expected)


def test_sparse_and_dense_give_the_same_range_and_zeros_do_not_count():
    rng = np.random.default_rng(0)
    dense = np.where(rng.random((200, 30)) < 0.2, rng.gamma(2.0, 1.0, (200, 30)), 0.0).astype("float32")
    low, high = er.layer_range(dense)
    assert low == 0.0
    positive = dense[dense > 0]
    assert high == er.nice_ceil(float(np.percentile(positive, 99)))
    assert er.layer_range(sp.csr_matrix(dense)) == (low, high)


def test_a_layer_with_negative_values_gets_a_negative_lower_end():
    rng = np.random.default_rng(1)
    scaled = rng.normal(0, 1, (300, 20))
    low, high = er.layer_range(scaled)
    assert low < 0 < high
    assert low == -er.nice_ceil(-float(np.percentile(scaled, 1)))


def test_samples_at_most_sample_cells_evenly(monkeypatch):
    monkeypatch.setattr(er, "SAMPLE_CELLS", 10)
    X = np.zeros((1000, 2))
    X[::100, 0] = 3.0  # rows 0, 100, ... 900: hit by the 10 evenly spaced rows
    X[1:100, 1] = 50.0  # never sampled
    assert er.layer_range(X) == (0.0, 3.0)


def test_an_empty_or_all_zero_layer_falls_back_to_0_to_1():
    assert er.layer_range(np.zeros((0, 3))) == (0.0, 1.0)
    assert er.layer_range(np.zeros((5, 3))) == (0.0, 1.0)
