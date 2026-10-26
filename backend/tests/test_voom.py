"""limma-voom / voomByGroup (core/voom.py) and RUVg (core/ruv.py) against
reference numbers from R on the same small fixture.

References: limma 3.68.5, edgeR 4.10.7, RUVSeq 1.46.0 (Bioconductor via r2u),
voomByGroup from github.com/YOU-k/voomByGroup (run only, never copied); 8 samples
(4 donors x conditions a/b, b the reference level), 60 genes, design
~donor + cond, lib.size = colSums x TMM.
"""

from __future__ import annotations

import numpy as np
import pytest

from scview.core import ruv, voom

_COUNTS = """
127,136,272,308,388,291,364,499,128,8,16,101,199,19,56,32,31,178,87,82,174,38,153,1034,4,154,229,172,28,15,15,71,213,23,6,76,110,50,374,47,82,17,47,79,166,586,277,66,6,65,158,6,83,78,55,27,6,20,39,30
202,98,85,187,48,383,81,119,13,12,13,48,12,5,91,4,34,151,65,16,52,11,56,77,26,108,91,232,23,45,15,65,101,25,8,57,74,25,203,28,37,4,19,51,93,187,113,115,4,33,139,32,90,36,41,20,16,8,54,29
88,117,302,232,140,528,123,180,15,9,18,44,36,21,35,33,65,51,68,78,199,9,259,620,9,77,144,199,41,35,9,77,107,18,3,191,264,44,410,68,54,42,14,94,209,158,194,100,4,23,616,27,183,51,55,102,13,14,176,47
45,149,96,288,41,147,178,112,26,8,21,4,44,4,27,28,12,259,189,40,70,6,65,400,14,69,95,40,32,19,6,45,76,14,3,37,73,36,56,22,66,12,28,62,208,113,87,64,2,12,276,21,70,48,37,63,28,18,13,9
53,52,27,86,39,53,65,41,8,4,8,53,47,18,50,23,20,127,69,57,102,6,67,269,6,84,141,73,45,58,13,40,125,16,4,47,118,52,159,48,51,10,30,99,84,165,138,67,3,8,143,66,71,39,122,76,18,27,76,36
48,20,23,41,43,69,52,42,12,7,7,38,34,7,57,24,25,91,65,33,29,10,43,113,13,69,70,165,22,37,13,50,181,26,5,25,79,20,154,36,62,10,37,106,84,237,86,61,6,38,245,8,60,18,130,37,14,26,71,36
23,12,31,23,10,36,17,37,3,0,5,31,24,4,15,10,20,42,20,20,35,11,21,66,7,37,44,26,10,18,4,26,44,1,0,43,82,23,76,13,21,12,18,56,80,77,52,35,1,1,84,14,38,13,48,33,7,9,17,17
18,10,7,29,10,35,16,8,4,4,4,41,18,1,16,6,8,41,16,41,32,6,39,110,4,39,46,57,18,22,2,22,66,14,2,22,20,14,76,26,27,7,18,49,40,57,30,45,2,4,114,8,31,15,41,13,5,10,22,14
"""
X = np.array([[float(v) for v in row.split(",")] for row in _COUNTS.strip().splitlines()])
COND = np.array(["a"] * 4 + ["b"] * 4)


def _design() -> np.ndarray:
    donors = np.tile(np.arange(4), 2)
    dummies = (donors[:, None] == np.arange(1, 4)[None, :]).astype(float)
    return np.column_stack([np.ones(8), dummies, (COND == "a").astype(float)])


def _eff_lib() -> np.ndarray:
    lib = X.sum(axis=1)
    return lib * voom.norm_factors(X, "TMM", lib)


def test_tmm_and_upper_quartile_match_edger():
    tmm = [0.846173525488, 1.03393379255, 0.828165153518, 0.913157474782, 1.15750571425,
           1.07999474084, 1.17935806728, 1.02516728629]
    uq = [0.957014740827, 1.01292108592, 1.12758622108, 0.811882086357, 0.958637752226,
          0.960058457434, 1.0501819303, 1.16585971049]
    assert voom.norm_factors(X, "TMM") == pytest.approx(tmm, rel=1e-9)
    assert voom.norm_factors(X, "upperquartile") == pytest.approx(uq, rel=1e-9)
    assert np.all(voom.norm_factors(X, "none") == 1)
    # Identical samples need no correction.
    assert voom.norm_factors(np.vstack([X[0], X[0], X[0]]), "TMM") == pytest.approx([1, 1, 1])


def test_voom_weights_and_moderated_t_match_limma():
    D = _design()
    E, w = voom.voom(X, D, lib_size=_eff_lib())
    assert E.shape == w.shape == X.shape
    # limma's weights are genes x samples: v$weights[1:3, 1] = sample 1, genes 1-3.
    assert w[0, :3] == pytest.approx([4.35679114956, 4.50479581261, 4.52151106293], rel=1e-8)
    eb = voom.ebayes(voom.lm_fit_weighted(E, D, w), D.shape[1] - 1)
    assert eb.t[:5] == pytest.approx(
        [2.00788260133, 4.33915870018, 6.4523448837, 5.93210497999, 2.80423700602], rel=1e-8)
    assert eb.p[:5] == pytest.approx(
        [0.0881014356867, 0.00411966339848, 0.000490053927857, 0.000786447345596,
         0.0287007510272], rel=1e-7)
    assert eb.df_prior == pytest.approx(3.45052239138, rel=1e-8)
    assert eb.s2_prior == pytest.approx(0.83202050051, rel=1e-8)


def test_voombygroup_matches_the_authors_code():
    D = _design()
    E, w = voom.voom(X, D, group=COND, lib_size=_eff_lib())
    assert w[4, :3] == pytest.approx([6.28561311144, 5.42620000313, 5.46709155542], rel=1e-8)
    eb = voom.ebayes(voom.lm_fit_weighted(E, D, w), D.shape[1] - 1)
    assert eb.t[:5] == pytest.approx(
        [1.87534472829, 4.47617489895, 6.54902063254, 5.97886175795, 2.63048367055], rel=1e-8)
    assert eb.df_prior == pytest.approx(3.3158108223, rel=1e-8)


def test_voombygroup_equals_voom_when_the_groups_are_alike():
    """Two groups holding the same samples have the same trend, so a trend per
    group gives voom's weights."""
    Xd = np.vstack([X[:4], X[:4]])
    cond = np.array(["a"] * 4 + ["b"] * 4)
    D = np.column_stack([np.ones(8), (cond == "a").astype(float)])
    f_one = voom.voom_fit(Xd, D)
    f_grp = voom.voom_fit(Xd, D, group=cond)
    assert f_grp.weights == pytest.approx(f_one.weights, rel=1e-9)
    assert f_grp.s2["a"] == pytest.approx(f_grp.s2["b"])
    pts = f_grp.trends["a"].points(10)
    assert 2 <= len(pts) <= 10 and all(len(p) == 2 for p in pts)


def test_voom_without_replication_gives_unit_weights():
    D = np.eye(2)
    f = voom.voom_fit(X[:2], D)
    assert np.all(f.weights == 1) and f.trends["all"] is None


def test_lm_fit_rejects_a_singular_design_and_ebayes_needs_df():
    D = np.column_stack([np.ones(8), np.ones(8)])
    with pytest.raises(voom.VoomError):
        voom.lm_fit_weighted(np.log1p(X), D)
    fit = voom.lm_fit_weighted(np.log1p(X), np.eye(8))
    with pytest.raises(voom.VoomError):
        voom.ebayes(fit, 0)


def test_choose_span_follows_limma():
    assert voom.choose_span(10) == 1.0
    assert voom.choose_span(10_000) == pytest.approx(0.3 + 0.7 * (50 / 10_000) ** (1 / 3))


def test_full_rank_columns_drops_repeats_in_order():
    D = np.column_stack([np.ones(4), [1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 0, 0]])
    assert voom.full_rank_columns(D) == [0, 1, 3]


def test_ruvg_factors_match_ruvseq():
    """RUVSeq RUVg(center = TRUE) on TMM-scaled counts, controls = genes 31-60."""
    lib = _eff_lib()
    W = ruv.ruvg_factors(ruv.log_normalised(X, lib), np.arange(30, 60), 2)
    ref1 = [-0.490125970726, -0.349206737437, 0.202926502957, 0.106808934365,
            0.199770436735, -0.308498289522, 0.670186043598, -0.0318609199705]
    ref2 = [0.556427840459, 0.0821010098864, 0.202775922701, 0.237097609419,
            -0.29060619565, -0.51248546782, 0.179616463743, -0.454927182738]
    for j, ref in enumerate((ref1, ref2)):  # equal up to sign
        assert np.abs(W[:, j]) == pytest.approx(np.abs(ref), rel=1e-8)
        assert abs(np.corrcoef(W[:, j], ref)[0, 1]) == pytest.approx(1.0)
        assert W[np.argmax(np.abs(W[:, j])), j] > 0  # the sign rule


def test_ruvg_with_a_nuisance_design_is_orthogonal_to_it():
    D = _design()[:, :4]  # intercept + donors
    W = ruv.ruvg_factors(ruv.log_normalised(X, _eff_lib()), np.arange(30, 60), 3, nuisance=D)
    assert W.shape == (8, 3)
    assert np.abs(D.T @ W).max() < 1e-10


def test_empirical_controls_take_the_largest_p_at_most_half():
    p = np.array([0.9, 0.01, 0.5, np.nan, 0.99, 0.2])
    assert ruv.empirical_controls(p, n=2).tolist() == [0, 4]
    assert ruv.empirical_controls(p, n=1000).tolist() == [0, 2, 4]
    assert len(ruv.empirical_controls(np.array([0.5, 0.6]))) == 0
