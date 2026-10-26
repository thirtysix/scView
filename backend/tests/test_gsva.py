"""GSVA scores (C097): the walk statistic by hand, the blocked ECDF ranks against the
procedure written out with numpy, the kernel below 200 cells, sparse == dense, blocks,
the gene filter, and a sampled reference against the full one.

Agreement with R GSVA 2.6 ``gsva(gsvaParam(...))`` on a Kang subsample (2,000 cells,
50 Hallmark sets; and 150 cells for the kernel) is in the C097 report.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp
from scipy import special, stats

from scview.core import gsva


def _data(n_cells=300, n_genes=50, seed=2):
    rng = np.random.default_rng(seed)
    C = rng.poisson(rng.gamma(0.5, 2.0, size=n_genes), size=(n_cells, n_genes)).astype(np.float64)
    C[:, 3] = 0  # never detected: left out
    C[:, 9] = 0
    C[0, 9] = 2  # seen in one cell: kept
    totals = C.sum(axis=1, keepdims=True)
    X = np.log1p(np.divide(C, totals, out=np.zeros_like(C), where=totals > 0) * 1e4)
    X[:, 4] = 1.5  # one value in every cell: left out
    return X


def by_hand(X: np.ndarray, S, kernel: bool = False, max_diff: bool = True) -> np.ndarray:
    """GSVA written out: filter, per-gene CDF, per-cell ranks (ties: later gene lower),
    |p/2 - r| weights and the explicit walk."""
    keep = X.max(axis=0) > X.min(axis=0)
    Y = X[:, keep]
    n, p = Y.shape
    kept = np.flatnonzero(keep)
    S = [int(np.flatnonzero(kept == s)[0]) for s in S if keep[s]]
    if kernel:
        h = Y.std(axis=0, ddof=1) / 4
        F = np.array([[special.ndtr((Y[c, g] - Y[:, g]) / h[g]).mean() for g in range(p)] for c in range(n)])
    else:
        F = np.array([[np.mean(Y[:, g] <= Y[c, g]) for g in range(p)] for c in range(n)])
    out = []
    for c in range(n):
        # ascending ranks, ties.method="last": among equal F the later gene ranks lower
        order = sorted(range(p), key=lambda g: (F[c, g], -g))
        r = np.empty(p)
        r[order] = np.arange(1, p + 1)
        walk_order = np.argsort(-r)
        inside = np.isin(walk_order, S)
        w = np.where(inside, np.abs(p / 2 - r[walk_order]), 0.0)
        with np.errstate(invalid="ignore"):  # all weights 0: no walk (NaN)
            walk = np.cumsum(w) / w.sum() - np.cumsum(~inside) / (p - len(S))
        top, bottom = max(walk.max(), 0), min(walk.min(), 0)
        out.append(top + bottom if max_diff else (top if top > -bottom else bottom))
    return np.array(out)


def test_walk_by_hand():
    # p = 4 genes, a one-gene set. At the top: the walk is 1, 2/3, 1/3, 0 -> 1. At the
    # bottom: -1/3, -2/3, -1, 0 -> -1. Second: -1/3, 2/3, 1/3, 0 -> 2/3 - 1/3.
    es = gsva.walk_scores(np.array([[1], [4], [2]]), np.array([[1.0], [1.0], [1.0]]), 4)
    np.testing.assert_allclose(es, [1.0, -1.0, 1 / 3])
    # Not max_diff: the deviation furthest from 0.
    es = gsva.walk_scores(np.array([[2]]), np.array([[1.0]]), 4, max_diff=False)
    assert es[0] == pytest.approx(2 / 3)
    # All set weights 0: no walk, as in GSVA.
    assert np.isnan(gsva.walk_scores(np.array([[2]]), np.array([[0.0]]), 4)[0])


@pytest.mark.parametrize("S", [[0, 1, 2, 5, 8, 13], [6], [3, 4, 10, 20, 30, 40, 41]])
def test_ecdf_matches_the_procedure(S):
    X = _data()
    prep = gsva.prepare(sp.csr_matrix(X))
    assert not prep.kernel and not prep.sampled
    np.testing.assert_allclose(gsva.score(prep, sp.csr_matrix(X), S), by_hand(X, S), atol=1e-12)


def test_max_diff_off():
    X = _data(n_cells=220)
    prep = gsva.prepare(X)
    np.testing.assert_allclose(gsva.score(prep, X, [1, 2, 9], max_diff=False),
                               by_hand(X, [1, 2, 9], max_diff=False), atol=1e-12)


def test_kernel_below_200_cells():
    X = _data(n_cells=40, n_genes=20)
    prep = gsva.prepare(X)
    assert prep.kernel
    np.testing.assert_allclose(gsva.score(prep, X, [0, 2, 7]), by_hand(X, [0, 2, 7], kernel=True), atol=1e-12)


def test_filter_drops_constant_genes():
    X = _data()
    prep = gsva.prepare(X)
    assert not prep.keep[3] and not prep.keep[4] and prep.keep[0] and prep.keep[9]
    with pytest.raises(ValueError, match="varies"):
        gsva.score(prep, X, [3, 4])


def test_sparse_dense_blocks_and_counts_agree():
    X = _data(n_cells=257)
    S = [0, 1, 2, 5, 8]
    a = gsva.score(gsva.prepare(sp.csr_matrix(X)), sp.csr_matrix(X), S)
    np.testing.assert_allclose(gsva.score(gsva.prepare(X), X, S), a, atol=1e-12)
    np.testing.assert_allclose(gsva.score(gsva.prepare(X), X, S, block_rows=13), a, atol=1e-12)
    E = sp.csr_matrix(X)
    E.data[::4] = 0.0  # explicitly stored zeros are zeros
    np.testing.assert_allclose(gsva.score(gsva.prepare(E), E, S), gsva.score(gsva.prepare(E.toarray()), E.toarray(), S),
                               atol=1e-12)


def test_counts_are_log_normalised_in_blocks():
    rng = np.random.default_rng(3)
    C = rng.poisson(1.0, size=(250, 30)).astype(np.float32)
    a = ad.AnnData(X=sp.csr_matrix(C))
    a.layers["counts"] = a.X.copy()
    a.X = sp.csr_matrix(np.where(C > 0, C - 2.5, -1.0))  # scaled: not used
    M, genes, source = gsva.gsva_matrix(a)
    assert "log-normalised" in source and "counts" in source
    L = np.log1p(C / C.sum(axis=1, keepdims=True) * 1e4)
    np.testing.assert_allclose(sp.vstack([M[0:100], M[100:250]]).toarray(), L, rtol=1e-6)


def test_gsva_matrix_prefers_log_raw():
    rng = np.random.default_rng(4)
    C = rng.poisson(1.0, size=(50, 10)).astype(np.float32)
    L = np.log1p(C / np.maximum(C.sum(axis=1, keepdims=True), 1) * 1e4)
    a = ad.AnnData(X=sp.csr_matrix(L))
    a.raw = a
    a.X = sp.csr_matrix(L - L.mean(axis=0))
    assert gsva.gsva_matrix(a)[2] == "log-normalised .raw"


def test_sampled_reference_stays_close():
    # Error of an ECDF from a sample: DKW bounds each gene's F within
    # sqrt(ln(2 / 0.05) / (2 m)) of the full one; ranks then move little.
    X = _data(n_cells=3000, n_genes=60, seed=5)
    S = [0, 1, 2, 5, 8, 13, 21, 34]
    full = gsva.score(gsva.prepare(X), X, S)
    sub = gsva.prepare(X, ref_cells=1000)
    assert sub.sampled and sub.n_ref == 1000
    rho = stats.spearmanr(gsva.score(sub, X, S), full).statistic
    assert rho > 0.98
    assert gsva.reference_cells(10, 20).tolist() == list(range(10))
    np.testing.assert_array_equal(gsva.reference_cells(3000, 1000), gsva.reference_cells(3000, 1000))


def test_too_little_to_rank():
    with pytest.raises(ValueError, match="2 cells"):
        gsva.prepare(np.ones((1, 5)))
    with pytest.raises(ValueError, match="vary"):
        gsva.prepare(np.ones((300, 5)))


def test_note_warns_when_the_set_is_mostly_zero():
    # On sparse single-cell data GSVA's scores bunch together (Kang's interferon set:
    # -0.75 to -0.56): the note says so when most of the set's values are 0.
    rng = np.random.default_rng(7)
    n, p = 400, 60
    C = rng.poisson(2.0, size=(n, p)).astype(np.float64)
    C[:, :10] *= rng.random((n, 10)) < 0.1  # the set: 90% zeros
    a = ad.AnnData(X=sp.csr_matrix(np.log1p(C)))
    a.var_names = [f"G{i}" for i in range(p)]
    sparse_note = gsva.score_set(a, [f"G{i}" for i in range(10)]).note
    assert "scores bunch together" in sparse_note and "ssGSEA or AUCell" in sparse_note
    dense_note = gsva.score_set(a, [f"G{i}" for i in range(20, 30)]).note
    assert "bunch together" not in dense_note
