"""AUCell gene-set scores (C097): ranks, AUC, the reference's tie rule, the depth check.

Agreement with decoupler 2.2's ``dc.mt.aucell`` on Kang and paul15 (max |diff| 0 over
9 sets) is in the C097 report; decoupler is not installed in the app image, so here
the AUC is checked by hand on a tiny matrix and against decoupler's procedure written
out with scipy (ordinal ranks over the shuffled columns, then its recovery-curve sum).
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp
from scipy import stats

from scview.core import aucell

GENES = [f"G{i}" for i in range(10)]

# decoupler breaks ties in numpy's default_rng(0) order of the detected genes; for 10
# genes that order is G0, G9, G1, G2, G4, G6, G3, G7, G5, G8.
TIES_10 = [0, 9, 1, 2, 4, 6, 3, 7, 5, 8]

HAND = np.array([
    # A: G1 5 (rank 1); G2 = G3 = 3, G2 first in tie order (ranks 2, 3); G7 2 (4); G5 1 (5)
    [0, 5, 3, 3, 0, 1, 0, 2, 0, 0],
    # B: G3 4 (1), G0 2 (2), G9 1 (3); then undetected genes in tie order: G1 (4), G2 (5)
    [2, 0, 0, 4, 0, 0, 0, 0, 0, 1],
    # C: G4, G5, G6, G8 tie at 1, in tie order G4, G6, G5, G8 (ranks 1-4); G0 (5)
    [0, 0, 0, 0, 1, 1, 1, 0, 1, 0],
    # D: nothing detected
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=np.float32)


def test_tie_order_is_decouplers_shuffle():
    assert aucell.tie_order(np.ones(10, dtype=bool)).tolist() == TIES_10
    # Over the detected genes only: an undetected column is skipped, not shuffled in.
    det = np.ones(11, dtype=bool)
    det[5] = False
    idx = np.flatnonzero(det)
    assert aucell.tie_order(det).tolist() == idx[TIES_10].tolist()


def test_auc_by_hand_with_ties():
    rk = aucell.build(sp.csr_matrix(HAND), GENES, max_rank=5)
    assert rk.max_rank == 5 and rk.n_ranked == 10
    assert rk.top[0].tolist() == [1, 2, 3, 7]
    assert rk.top[1].tolist() == [3, 0, 9, 1]
    assert rk.top[2].tolist() == [4, 6, 5, 8]
    # Set G1, G3, G5; ranks 1-4 add 4, 3, 2, 1; best possible 4 + 3 + 2 = 9.
    s = aucell.score(rk, [1, 3, 5])
    np.testing.assert_allclose(s, [
        (4 + 2) / 9,  # A: G1 rank 1, G3 rank 3 (G5 at rank 5 adds 0)
        (4 + 1) / 9,  # B: G3 rank 1, G1 rank 4 (undetected, placed by the tie order)
        2 / 9,  # C: G5 rank 3
        0.0,  # D: an empty cell scores 0
    ])


def test_sparse_and_dense_agree():
    a = aucell.build(sp.csr_matrix(HAND), GENES, max_rank=5)
    b = aucell.build(HAND, GENES, max_rank=5)
    np.testing.assert_array_equal(a.top, b.top)
    np.testing.assert_array_equal(aucell.score(a, [1, 3, 5]), aucell.score(b, [1, 3, 5]))


def test_explicit_zeros_rank_as_zeros():
    M = sp.csr_matrix(HAND)
    M.data[M.data == 3] = 0  # stored zeros: still undetected genes
    ref = HAND.copy()
    ref[ref == 3] = 0
    a = aucell.build(M, GENES, max_rank=5, detected=np.ones(10, dtype=bool))
    b = aucell.build(ref, GENES, max_rank=5, detected=np.ones(10, dtype=bool))
    np.testing.assert_array_equal(a.top, b.top)


# --- against decoupler's procedure, written out ---------------------------------


def _decoupler_aucell(X: np.ndarray, sets: list[list[int]], n_up: int | None = None) -> np.ndarray:
    """decoupler 2.2 ``dc.mt.aucell``: drop empty genes, shuffle the columns with
    default_rng(0), ordinal ranks of -x per cell, recovery-curve sum over ranks <= n_up,
    divided by the best a set of that size can do. Empty cells (dropped there) are 0."""
    det = np.count_nonzero(X, axis=0) > 0
    cols = np.flatnonzero(det)
    perm = np.random.default_rng(seed=0).choice(np.arange(cols.size), cols.size, replace=False)
    shuffled = cols[perm]
    nvar = cols.size
    if n_up is None:
        n_up = int(np.clip(np.ceil(0.05 * nvar), 2, nvar))
    out = np.zeros((X.shape[0], len(sets)))
    for i in range(X.shape[0]):
        if not X[i].any():
            continue
        row = stats.rankdata(-X[i, shuffled].astype(float), method="ordinal")
        rank_of = dict(zip(shuffled.tolist(), row.tolist()))
        for j, s in enumerate(sets):
            fset = [g for g in s if det[g]]
            x_th = np.arange(1, len(fset) + 1)
            x_th = x_th[x_th < n_up]
            max_auc = np.sum(np.diff(np.append(x_th, n_up)) * x_th)
            x = np.array([rank_of[g] for g in fset])
            x = np.sort(x[x <= n_up])
            y = np.arange(x.shape[0]) + 1
            x = np.append(x, n_up)
            out[i, j] = np.sum(np.diff(x) * y) / max_auc
    return out


def _counts(n_cells=60, n_genes=120, seed=0):
    """Low counts (many ties), varied depth (some cells detect fewer genes than
    max_rank), two never-detected genes and one empty cell."""
    rng = np.random.default_rng(seed)
    depth = rng.gamma(2.0, 0.5, size=(n_cells, 1))
    depth[:20] *= 0.1  # a third of the cells detect only a few genes
    mu = rng.gamma(0.6, 1.0, size=(1, n_genes)) * depth
    X = rng.poisson(mu).astype(np.float32)
    X[-1] = 1 + rng.poisson(2.0, n_genes)  # one deep cell: every gene detected somewhere
    X[:, [7, 70]] = 0
    X[5] = 0
    return X


SETS = [[0, 1, 2, 3, 4, 5, 6, 7], [10, 20, 30], list(range(40, 100)), [99]]


@pytest.mark.parametrize("n_up", [None, 2, 3, 17, 118])
def test_matches_decoupler_procedure(n_up):
    X = _counts()
    rk = aucell.build(sp.csr_matrix(X), [f"G{i}" for i in range(X.shape[1])], max_rank=n_up)
    assert rk.n_ranked == 118
    assert rk.max_rank == (n_up or 6)  # ceil(5% of 118)
    if n_up in (None, 17):
        assert rk.share_below_max_rank() >= 0.2  # the undetected-gene ties are exercised
    ref = _decoupler_aucell(X, SETS, n_up)
    for j, s in enumerate(SETS):
        np.testing.assert_allclose(aucell.score(rk, s), ref[:, j], rtol=0, atol=1e-12)


def test_log_normalised_values_rank_like_counts():
    X = _counts()
    lib = X.sum(axis=1, keepdims=True)
    L = np.log1p(np.divide(X, lib, out=np.zeros_like(X), where=lib > 0) * 1e4)
    a = aucell.build(X, [f"G{i}" for i in range(120)])
    b = aucell.build(sp.csr_matrix(L), [f"G{i}" for i in range(120)])
    np.testing.assert_array_equal(a.top, b.top)


def test_scoring_in_blocks_matches_one_block(monkeypatch):
    X = _counts()
    rk = aucell.build(X, [f"G{i}" for i in range(120)])
    whole = aucell.score(rk, SETS[2])
    monkeypatch.setattr(aucell, "_SCORE_ROWS", 7)  # 60 cells: 9 blocks, the last of 4
    np.testing.assert_array_equal(aucell.score(rk, SETS[2]), whole)


def test_max_rank_edges():
    X = _counts()
    names = [f"G{i}" for i in range(120)]
    two = aucell.build(X, names, max_rank=2)
    assert two.top.shape == (60, 1)
    s = aucell.score(two, [int(two.top[0, 0])])
    assert s[0] == 1.0  # the cell's top gene, alone in the set, is the best possible
    whole = aucell.build(X, names, max_rank=118)
    assert whole.top.shape == (60, 117)
    for bad in (1, 119, 0):
        with pytest.raises(ValueError, match="max_rank"):
            aucell.build(X, names, max_rank=bad)
    assert aucell.default_max_rank(10) == 2 and aucell.default_max_rank(14044) == 703


def test_a_set_larger_than_max_rank_reaches_one():
    rk = aucell.build(sp.csr_matrix(HAND), GENES, max_rank=3)
    assert aucell.max_auc(9, 3) == aucell.max_auc(2, 3) == 3.0  # 2 + 1
    s = aucell.score(rk, [1, 2, 0, 4, 5, 6, 7, 8, 9])  # A's ranks 1 and 2 (G1, G2) in it
    assert s[0] == 1.0


def test_match_reports_missing_and_undetected_genes():
    X = np.hstack([HAND, np.zeros((4, 1), dtype=np.float32)])
    names = [*GENES, "Never"]
    rk = aucell.build(X, names, max_rank=5)
    m = aucell.match(rk, ["g1", "G3", "G3", "never", "NOPE"])
    assert m.found == ["G1", "G3"] and m.idx == [1, 3]
    assert m.undetected == ["Never"] and m.missing == ["NOPE"]
    # The undetected gene is left out of the set's size, as decoupler prunes it.
    np.testing.assert_array_equal(aucell.score(rk, [1, 3, 10]), aucell.score(rk, [1, 3]))


# --- what is ranked, and the cache ----------------------------------------------


def _adata(X, raw=None, counts=None):
    a = ad.AnnData(X=X)
    a.var_names = [f"G{i}" for i in range(X.shape[1])]
    if raw is not None:
        r = ad.AnnData(X=raw)
        r.var_names = a.var_names
        a.raw = r
    if counts is not None:
        a.layers["counts"] = counts
    return a


def test_rank_matrix_never_ranks_scaled_values():
    X = _counts()
    scaled = (X - X.mean(axis=0)) / (X.std(axis=0) + 1)
    _, _, src = aucell.rank_matrix(_adata(scaled, raw=np.log1p(X)))
    assert src == "log-normalised .raw"
    _, _, src = aucell.rank_matrix(_adata(scaled, counts=sp.csr_matrix(X)))
    assert src == "counts in layers['counts']"
    _, _, src = aucell.rank_matrix(_adata(X))
    assert src == "counts X"
    with pytest.raises(ValueError, match="scaled"):
        aucell.rank_matrix(_adata(scaled))


def test_rankings_are_cached_per_dataset_and_max_rank():
    a = _adata(sp.csr_matrix(_counts()))
    r1 = aucell.rankings(a)
    assert aucell.rankings(a) is r1
    assert aucell.rankings(a, r1.max_rank) is r1  # the default asked for by number
    r2 = aucell.rankings(a, 10)
    assert r2 is not r1 and r2.max_rank == 10
    aucell.rankings(a, 12)  # a third max_rank pushes out the oldest
    assert aucell.rankings(a, 10) is r2
    assert aucell.rankings(a) is not r1


# --- the depth check ------------------------------------------------------------


def test_spearman_matches_scipy():
    rng = np.random.default_rng(1)
    x = rng.poisson(3, 200).astype(float)  # ties
    y = x + rng.normal(0, 2, 200)
    assert aucell.spearman(x, y) == pytest.approx(stats.spearmanr(x, y).statistic, abs=1e-12)
    assert aucell.spearman(np.ones(10), y[:10]) is None
    assert aucell.spearman([1, 2], [2, 1]) is None


def test_within_group_rho_is_the_median_of_groups():
    rng = np.random.default_rng(2)
    n = 300
    codes = np.repeat([0, 1, 2], 100)
    codes[:5] = -1  # unlabelled cells count overall only
    depth = rng.gamma(3, 1, n) + codes.clip(0) * 5  # groups differ in depth
    score = codes.clip(0) * 2.0 + rng.normal(0, 1, n)  # and in score, not with depth inside
    c = aucell.correlations(score, depth, codes)
    expect = np.median([stats.spearmanr(score[codes == g], depth[codes == g]).statistic for g in (0, 1, 2)])
    assert c["within"] == pytest.approx(expect, abs=1e-12)
    assert c["n_groups"] == 3
    assert c["overall"] == pytest.approx(stats.spearmanr(score, depth).statistic, abs=1e-12)
    assert c["overall"] > 0.5 and abs(c["within"]) < 0.2  # between-group depth is not depth sensitivity
    small = aucell.correlations(score, depth, codes, min_cells=101)
    assert small["within"] is None and small["n_groups"] == 0


def test_depth_check_and_sentence():
    rng = np.random.default_rng(3)
    n = 400
    codes = np.repeat([0, 1], n // 2)
    depth = rng.gamma(4, 500, n)
    follows = np.log(depth) + rng.normal(0, 0.3, n)
    flat = rng.normal(0, 1, n)
    check = aucell.depth_check({"score_genes": follows, "aucell": flat},
                               {"total_counts": depth, "n_genes": depth ** 0.5}, codes)
    sg = check["methods"]["score_genes"]["total_counts"]
    au = check["methods"]["aucell"]["total_counts"]
    assert sg["within"] > 0.3 and abs(au["within"]) < 0.1
    assert check["between"]["overall"] == pytest.approx(stats.spearmanr(follows, flat).statistic, abs=1e-12)
    text = aucell.sentence(check, {"score_genes": "score_genes", "aucell": "AUCell"},
                           "total_counts", "within clusters")
    assert text == (f"score_genes follows depth on this data (rho {sg['within']:.2f} within clusters); "
                    f"AUCell does not follow depth (rho {au['within']:.2f}).")
    assert aucell.verdict(0.2) == "follows depth weakly" and aucell.verdict(-0.35) == "follows depth"
    assert aucell.verdict(None) == "cannot be checked against depth"
