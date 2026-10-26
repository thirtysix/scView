"""ssGSEA scores (C097): a hand-checked walk, the closed form against the walk itself
(R GSVA's ``ssgseaParam`` procedure written out with numpy), sparse == dense, blocks.

Agreement with R GSVA 2.6 ``gsva(ssgseaParam(...))`` on a Kang subsample (2,000
cells, 50 Hallmark sets) is in the C097 report; R is not in the app image.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
from scipy import stats

from scview.core import ssgsea


def walk_by_hand(x: np.ndarray, S, alpha: float) -> float:
    """One cell, written out: ranks upwards with ties averaged and truncated, genes
    visited by rank (highest first, ties in gene order), the running difference
    summed over every position."""
    p = x.size
    r = np.floor(stats.rankdata(x, method="average"))
    order = np.argsort(-r, kind="stable")
    inside = np.isin(order, S)
    w = np.where(inside, r[order] ** alpha, 0.0)
    walk_in = np.cumsum(w) / w.sum()
    walk_out = np.cumsum(~inside) / (p - inside.sum())
    return float(np.sum(walk_in - walk_out))


def test_walk_by_hand():
    # Ranks: 3 -> 4, 1 -> 3, the two zeros 1.5 -> 1. Walk order G0, G2, G1, G3.
    # Set {G0, G1}, alpha 1: weights 4 and 1. Running difference 0.8, 0.3, 0.5, 0.
    x = np.array([[3.0, 0.0, 1.0, 0.0]])
    es = ssgsea.score(sp.csr_matrix(x), [0, 1], alpha=1.0, normalize=False)
    assert es[0] == pytest.approx(1.6)
    assert walk_by_hand(x[0], [0, 1], 1.0) == pytest.approx(1.6)


def _data(n_cells=60, n_genes=40, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.poisson(rng.gamma(0.4, 2.0, size=n_genes), size=(n_cells, n_genes)).astype(np.float64)
    X[5] = 0  # a cell with nothing detected
    X[:, 7] = 0  # a gene no cell detects
    X[9, :] = 3  # a cell where every gene ties
    return X


@pytest.mark.parametrize("alpha", [0.25, 1.0])
@pytest.mark.parametrize("S", [[0, 3, 7, 11, 20], [1], list(range(0, 39, 2))])
def test_closed_form_matches_the_walk(alpha, S):
    X = _data()
    es = ssgsea.score(sp.csr_matrix(X), S, alpha=alpha, normalize=False)
    want = [walk_by_hand(x, S, alpha) for x in X]
    np.testing.assert_allclose(es, want, rtol=1e-10, atol=1e-10)


def test_log_values_rank_like_counts():
    X = _data()
    totals = X.sum(axis=1, keepdims=True)
    L = np.log1p(np.divide(X, totals, out=np.zeros_like(X), where=totals > 0) * 1e4)
    S = [2, 4, 6, 8]
    np.testing.assert_allclose(ssgsea.score(sp.csr_matrix(L), S), ssgsea.score(sp.csr_matrix(X), S))


def test_sparse_dense_and_blocks_agree():
    X = _data(n_cells=101)
    S = [0, 5, 9, 13]
    a = ssgsea.score(sp.csr_matrix(X), S)
    np.testing.assert_allclose(ssgsea.score(X, S), a, rtol=1e-12)
    np.testing.assert_allclose(ssgsea.score(sp.csr_matrix(X), S, block_rows=7), a, rtol=1e-12)
    # Explicitly stored zeros are zeros.
    E = sp.csr_matrix(X)
    E.data[::3] = 0.0
    np.testing.assert_allclose(ssgsea.score(E, S), ssgsea.score(E.toarray(), S), rtol=1e-12)


def test_normalised_by_the_range():
    X = _data()
    raw = ssgsea.score(X, [1, 2, 3], normalize=False)
    norm = ssgsea.score(X, [1, 2, 3])
    np.testing.assert_allclose(norm, raw / (raw.max() - raw.min()))
    assert norm.max() - norm.min() == pytest.approx(1.0)


def test_set_edges():
    X = np.eye(5)
    with pytest.raises(ValueError, match="at least one"):
        ssgsea.score(X, [])
    with pytest.raises(ValueError, match="every gene"):
        ssgsea.score(X, [0, 1, 2, 3, 4])
