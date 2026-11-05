"""DESeq2 vectorised over genes (core/deseq2_vec.py) and its NB GLM core
(core/nb_glm.py).

Expected values come from pydeseq2 0.5.4 run on the same counts and design
(the reference the module follows), from finite differences, or by hand;
never from the module's own output.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from scview.core import deseq2_vec as dv
from scview.core import nb_glm


def nb_counts(rng, mu, alpha):
    """Negative binomial counts with mean ``mu`` (samples x genes) and
    dispersion ``alpha`` (genes): gamma-Poisson."""
    lam = rng.gamma(1 / alpha, mu * alpha)
    return rng.poisson(lam)


def sim(n_per, G=200, seed=0, lfc=None, extra=None):
    """Two groups of ``n_per`` samples, gene means 5..2000, dispersions on a
    DESeq2-like trend with log-normal scatter, size factors 0.6..1.6."""
    rng = np.random.default_rng(seed)
    n = 2 * n_per
    base = np.exp(rng.uniform(np.log(5), np.log(2000), G))
    alpha = (0.05 + 1.5 / base) * np.exp(rng.normal(0, 0.5, G))
    cond = np.repeat([0.0, 1.0], n_per)
    eff = np.zeros(G) if lfc is None else lfc
    sf = rng.uniform(0.6, 1.6, n)
    log_mu = np.log(base)[None] + cond[:, None] * eff[None] + np.log(sf)[:, None]
    if extra is not None:
        log_mu = log_mu + extra
    return nb_counts(rng, np.exp(log_mu), alpha[None]), cond


def run_pydeseq2(counts, X, contrast):
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.default_inference import DefaultInference
    from pydeseq2.ds import DeseqStats

    idx = [f"s{i}" for i in range(len(counts))]
    cols = [f"x{j}" for j in range(X.shape[1])]
    design = pd.DataFrame(X, index=idx, columns=cols)
    df = pd.DataFrame(counts, index=idx, columns=[f"g{j}" for j in range(counts.shape[1])])
    inf = DefaultInference(n_cpus=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dds = DeseqDataSet(counts=df, metadata=pd.DataFrame(index=idx), design=design,
                           inference=inf, quiet=True)
        dds.deseq2()
        st = DeseqStats(dds, contrast=np.asarray(contrast, float), inference=inf, quiet=True)
        st.summary()
    return dds, st.results_df


def dlog10(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    return np.abs(np.log10(np.maximum(a[ok], 1e-300)) - np.log10(np.maximum(b[ok], 1e-300)))


def assert_agrees(res, dds, r, *, p_med=2e-3, p_q99=0.03, disp_med=2e-3):
    assert np.array_equal(np.isnan(res.pvalue), r["pvalue"].isna().to_numpy())
    np.testing.assert_allclose(res.size_factors, dds.obs["size_factors"].to_numpy(), rtol=1e-10)
    d = dlog10(res.dispersions, dds.var["dispersions"].to_numpy())
    assert np.median(d) < disp_med
    dp = dlog10(res.pvalue, r["pvalue"].to_numpy())
    assert np.median(dp) < p_med and np.quantile(dp, 0.99) < p_q99
    lfc = r["log2FoldChange"].to_numpy()
    ok = np.isfinite(lfc)
    assert np.median(np.abs(res.log2fc[ok] - lfc[ok])) < 2e-3
    ca = res.padj < 0.05
    cb = r["padj"].to_numpy() < 0.05
    assert (ca == cb).mean() >= 0.99


# ---------------------------------------------------------------------------
# Agreement with pydeseq2 by design
# ---------------------------------------------------------------------------


def test_condition_design_agrees_with_pydeseq2():
    G = 300
    lfc = np.where(np.arange(G) < 30, 1.5, 0.0)
    counts, cond = sim(4, G=G, seed=1, lfc=lfc)
    X = np.column_stack([np.ones(len(cond)), cond])
    res = dv.deseq2_wald(counts, X, pydeseq2_trend=True)
    dds, r = run_pydeseq2(counts, X, [0, 1])
    assert_agrees(res, dds, r)
    # Gene-wise and MAP dispersions and the trend, not only the end result.
    assert np.median(dlog10(res.genewise_dispersions,
                            dds.var["genewise_dispersions"].to_numpy())) < 1e-3
    np.testing.assert_allclose(res.trend_coeffs, dds.uns["trend_coeffs"].to_numpy(), rtol=0.05)
    assert res.prior_disp_var == pytest.approx(float(dds.uns["prior_disp_var"]), rel=0.05)
    assert (res.padj[:30] < 0.05).mean() > 0.5


def test_paired_design_agrees_with_pydeseq2():
    """~donor + condition: not saturated, so the dispersion means come from IRLS."""
    rng = np.random.default_rng(2)
    d = 4
    donor_eff = rng.normal(0, 0.5, (d, 1))
    extra = np.vstack([donor_eff, donor_eff])  # samples ordered cond 0 (d donors), cond 1
    counts, cond = sim(d, G=250, seed=3, extra=extra,
                       lfc=np.where(np.arange(250) < 25, -1.2, 0.0))
    donors = np.tile(np.arange(d), 2)
    X = np.column_stack([np.ones(2 * d)] + [(donors == k).astype(float) for k in range(1, d)]
                        + [cond])
    res = dv.deseq2_wald(counts, X, pydeseq2_trend=True)
    dds, r = run_pydeseq2(counts, X, np.eye(X.shape[1])[-1])
    assert_agrees(res, dds, r)


def test_covariate_design_agrees_with_pydeseq2():
    """~covariate + condition with a continuous covariate (every design row distinct)."""
    rng = np.random.default_rng(4)
    age = rng.uniform(-1, 1, 10)
    extra = 0.4 * age[:, None] * rng.normal(0, 1, (1, 220))
    counts, cond = sim(5, G=220, seed=5, extra=extra,
                       lfc=np.where(np.arange(220) < 20, 1.0, 0.0))
    X = np.column_stack([np.ones(10), age, cond])
    res = dv.deseq2_wald(counts, X, coef=2, pydeseq2_trend=True)
    dds, r = run_pydeseq2(counts, X, [0, 0, 1])
    assert_agrees(res, dds, r)


def test_contrast_vector_equals_coefficient():
    counts, cond = sim(3, G=80, seed=6)
    X = np.column_stack([np.ones(6), cond])
    a = dv.deseq2_wald(counts, X, coef=1)
    b = dv.deseq2_wald(counts, X, coef=np.array([0.0, 1.0]))
    np.testing.assert_array_equal(a.pvalue, b.pvalue)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_zero_genes_and_one_sided_zeros_match_pydeseq2():
    counts, cond = sim(4, G=120, seed=7)
    counts[:, 0] = 0  # all zero: no test
    counts[cond == 1, 1] = 0  # zero on one side only
    counts[cond == 0, 2] = 0
    X = np.column_stack([np.ones(8), cond])
    res = dv.deseq2_wald(counts, X, pydeseq2_trend=True)
    dds, r = run_pydeseq2(counts, X, [0, 1])
    assert np.isnan(res.pvalue[0]) and np.isnan(res.log2fc[0]) and res.base_mean[0] == 0
    assert np.isnan(res.dispersions[0])
    for g in (1, 2):
        assert res.log2fc[g] == pytest.approx(r["log2FoldChange"].iloc[g], abs=0.05)
        assert res.pvalue[g] == pytest.approx(r["pvalue"].iloc[g], rel=0.05)
    assert_agrees(res, dds, r)


def test_cooks_outlier_loses_its_p_value_as_in_pydeseq2():
    """Three replicates per side: no refit (needs 7), Cook's filter on."""
    counts, cond = sim(3, G=150, seed=8)
    counts[0, 5] = counts[:, 5].max() * 60 + 500
    X = np.column_stack([np.ones(6), cond])
    res = dv.deseq2_wald(counts, X, pydeseq2_trend=True)
    dds, r = run_pydeseq2(counts, X, [0, 1])
    assert res.cooks_outlier[5] and np.isnan(res.pvalue[5])
    np.testing.assert_array_equal(res.cooks_outlier, dds.var["_pvalue_cooks_outlier"].to_numpy())
    assert not res.replaced.any()
    assert_agrees(res, dds, r)
    off = dv.deseq2_wald(counts, X, cooks_filter=False, pydeseq2_trend=True)
    assert np.isfinite(off.pvalue[5])


def test_outlier_refit_with_seven_replicates_matches_pydeseq2():
    counts, cond = sim(7, G=150, seed=9, lfc=np.where(np.arange(150) < 15, 1.0, 0.0))
    counts[2, 40] = counts[:, 40].max() * 40 + 300
    counts[9, 41] = counts[:, 41].max() * 40 + 300
    X = np.column_stack([np.ones(14), cond])
    res = dv.deseq2_wald(counts, X, pydeseq2_trend=True)
    dds, r = run_pydeseq2(counts, X, [0, 1])
    np.testing.assert_array_equal(res.replaced, dds.var["replaced"].to_numpy())
    np.testing.assert_array_equal(res.refitted, dds.var["refitted"].to_numpy())
    assert res.refitted[40] and res.refitted[41]
    np.testing.assert_allclose(res.base_mean, r["baseMean"].to_numpy(), rtol=1e-10)
    assert_agrees(res, dds, r)
    # Without the refit the outlier gene keeps its counts and is filtered instead.
    plain = dv.deseq2_wald(counts, X, refit_cooks=False, pydeseq2_trend=True)
    assert not plain.refitted.any()


def test_every_gene_with_a_zero_uses_iterative_size_factors():
    rng = np.random.default_rng(10)
    counts, cond = sim(4, G=120, seed=10)
    for g in range(counts.shape[1]):
        counts[rng.integers(0, 8), g] = 0
    X = np.column_stack([np.ones(8), cond])
    res = dv.deseq2_wald(counts, X, pydeseq2_trend=True)
    dds, r = run_pydeseq2(counts, X, [0, 1])
    np.testing.assert_allclose(res.size_factors, dds.obs["size_factors"].to_numpy(), rtol=0.02)
    dp = dlog10(res.pvalue, r["pvalue"].to_numpy())
    assert np.median(dp) < 0.01


def test_genes_the_newton_search_cannot_handle_fall_back_to_the_grid(monkeypatch):
    counts, cond = sim(4, G=100, seed=11)
    X = np.column_stack([np.ones(8), cond])
    ref = dv.deseq2_wald(counts, X, pydeseq2_trend=True)
    real = nb_glm.maximise_1d

    def flaky(derivs, x0, lo, hi, **kw):
        x, ok = real(derivs, x0, lo, hi, **kw)
        ok = ok.copy()
        ok[::7] = False
        return x, ok

    monkeypatch.setattr(nb_glm, "maximise_1d", flaky)
    res = dv.deseq2_wald(counts, X, pydeseq2_trend=True)
    assert not res.genewise_converged[::7].all()
    # The 100 + 100 point grid lands within one fine step (~0.004 in log alpha).
    moved = ref.genewise_dispersions > 1e-6
    d = np.abs(np.log(res.genewise_dispersions) - np.log(ref.genewise_dispersions))[moved]
    assert d.max() < 0.01


def test_default_trend_leaves_out_genes_on_the_dispersion_floor():
    """The default trend is pydeseq2's gamma GLM on the genes above
    100 * min_disp only (pydeseq2_trend=True: on every gene)."""
    counts, cond = sim(2, G=300, seed=15)
    X = np.column_stack([np.ones(4), cond])
    res = dv.deseq2_wald(counts, X)
    py = dv.deseq2_wald(counts, X, pydeseq2_trend=True)
    floor = res.genewise_dispersions < 100 * dv.MIN_DISP
    assert floor.sum() > 10  # 2 + 2 samples: many genes look Poisson
    mean = res.base_mean
    want = dv._parametric_trend(mean[~floor], res.genewise_dispersions[~floor])
    np.testing.assert_allclose(res.trend_coeffs, want)
    np.testing.assert_allclose(py.trend_coeffs,
                               dv._parametric_trend(mean, py.genewise_dispersions))


def test_chunked_fit_equals_one_chunk(monkeypatch):
    counts, cond = sim(4, G=150, seed=17, lfc=np.where(np.arange(150) < 15, 1.0, 0.0))
    X = np.column_stack([np.ones(8), cond])
    whole = dv.deseq2_wald(counts, X)
    monkeypatch.setattr(dv, "_CHUNK_ELEMENTS", 8 * 40)  # 40 genes a chunk
    parts = dv.deseq2_wald(counts, X)
    # Chunks change the matrix shapes, so BLAS rounding (CPU-dependent) differs at ~1e-9.
    np.testing.assert_allclose(parts.pvalue, whole.pvalue, rtol=1e-7)
    np.testing.assert_allclose(parts.dispersions, whole.dispersions, rtol=1e-7)


def test_independent_filtering_matches_plain_bh_when_off():
    counts, cond = sim(4, G=150, seed=12, lfc=np.where(np.arange(150) < 40, 2.0, 0.0))
    X = np.column_stack([np.ones(8), cond])
    res = dv.deseq2_wald(counts, X, independent_filter=False)
    from scipy import stats

    ok = np.isfinite(res.pvalue)
    np.testing.assert_allclose(res.padj[ok], stats.false_discovery_control(res.pvalue[ok]))


def test_bad_input_is_refused():
    X = np.column_stack([np.ones(4), [0, 0, 1, 1]])
    with pytest.raises(ValueError, match="integers"):
        dv.deseq2_wald(np.full((4, 3), 1.5), X)
    with pytest.raises(ValueError, match="full rank"):
        dv.deseq2_wald(np.ones((4, 3), int), np.column_stack([X, X[:, 1]]))
    with pytest.raises(ValueError, match="degrees of freedom"):
        dv.deseq2_wald(np.ones((2, 3), int), X[:2] + np.array([[0, 0], [0, 1]]))


# ---------------------------------------------------------------------------
# NB GLM core
# ---------------------------------------------------------------------------


def test_dispersion_derivatives_match_finite_differences():
    rng = np.random.default_rng(13)
    X = np.column_stack([np.ones(6), [0, 0, 0, 1, 1, 1], rng.normal(size=6)])
    XX = nb_glm.outer_design(X)
    mu = rng.uniform(1, 200, (5, 6))
    Y = rng.poisson(mu).astype(float)
    th = np.log(np.array([1e-3, 0.02, 0.1, 0.5, 3.0]))
    eps = 1e-5

    def f(t):
        a = np.exp(t)
        return -nb_glm.nb_nll(Y, mu, a) + nb_glm.cox_reid(mu, a, XX, 3)

    def d(t):
        g1, h1 = nb_glm.nb_loglik_derivs(Y, mu, np.exp(t))
        g2, h2 = nb_glm.cox_reid_derivs(mu, np.exp(t), XX, 3)
        return g1 + g2, h1 + h2

    g, h = d(th)
    np.testing.assert_allclose(g, (f(th + eps) - f(th - eps)) / (2 * eps), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(h, (d(th + eps)[0] - d(th - eps)[0]) / (2 * eps), rtol=1e-5,
                               atol=1e-6)


def test_maximise_1d_finds_the_maximum_uphill_and_the_bounds():
    # f(x) = -(x - c)^2 with maxima inside, beyond hi and below lo.
    c = np.array([0.3, 5.0, -5.0, 1.0])

    def derivs(idx, x):
        return -2 * (x - c[idx]), np.full(len(idx), -2.0)

    x, ok = nb_glm.maximise_1d(derivs, np.array([-1.0, 0.0, 0.0, 1.0]), np.full(4, -2.0),
                               np.full(4, 2.0))
    np.testing.assert_allclose(x, [0.3, 2.0, -2.0, 1.0], atol=1e-8)
    assert ok.all()

    # Two maxima (a quartic): uphill from x0 = 0.5 reaches the one at +1, not -1.
    def quartic(idx, x):
        return -4 * x ** 3 + 4 * x, -12 * x ** 2 + 4

    x, ok = nb_glm.maximise_1d(quartic, np.array([0.5]), np.array([-3.0]), np.array([3.0]))
    assert x[0] == pytest.approx(1.0, abs=1e-8) and ok[0]


def test_irls_matches_pydeseq2_solver():
    from pydeseq2.utils import irls_solver

    counts, cond = sim(4, G=30, seed=14, lfc=np.linspace(-2, 2, 30))
    X = np.column_stack([np.ones(8), cond])
    sf = dv.size_factors(counts)
    alpha = np.full(30, 0.1)
    fit = nb_glm.irls(counts.T.astype(float), np.log(sf), X, alpha)
    for g in range(30):
        beta, mu, hat, conv = irls_solver(counts[:, g], sf, X, 0.1)
        np.testing.assert_allclose(fit.beta[g], beta, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(fit.mu[g], mu, rtol=1e-6)
        np.testing.assert_allclose(fit.hat[g], hat, rtol=1e-6, atol=1e-10)


def test_nb_deviance_by_hand():
    Y = np.array([[0.0, 3.0, 10.0]])
    mu = np.array([[1.0, 2.0, 12.0]])
    a = 0.2
    want = 0.0
    for y, m in zip(Y[0], mu[0]):
        t1 = y * np.log(y / m) if y > 0 else 0.0
        want += 2 * (t1 - (y + 1 / a) * np.log((1 + a * y) / (1 + a * m)))
    assert nb_glm.nb_deviance(Y, mu, np.array([a]))[0] == pytest.approx(want)


def test_trigamma_matches_scipy():
    from scipy import special

    x = np.concatenate([np.linspace(1e-3, 30, 2000), np.logspace(-4, 9, 500)])
    np.testing.assert_allclose(nb_glm.trigamma(x), special.polygamma(1, x), rtol=1e-11)


# ---------------------------------------------------------------------------
# Prior variance rules at low residual df
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("df", [1, 2, 3, 6])
def test_mad_chisq_spread_matches_a_monte_carlo_mad(df):
    from scipy import stats

    x = np.log(np.random.default_rng(df).chisquare(df, 400_000))
    mad = np.median(np.abs(x - np.median(x))) / stats.norm.ppf(0.75)
    assert dv.mad_chisq_spread(df) == pytest.approx(mad ** 2, rel=0.01)


def test_mad_chisq_spread_approaches_trigamma_with_df():
    from scipy import special

    assert dv.mad_chisq_spread(3) < special.polygamma(1, 1.5)  # 0.79 vs 0.93
    assert dv.mad_chisq_spread(400) == pytest.approx(special.polygamma(1, 200), rel=0.01)


def test_prior_rule_applies_only_at_low_df():
    counts, cond = sim(2, G=300, seed=20)  # 2 v 2: n - p = 2
    X = np.column_stack([np.ones(4), cond])
    base = dv.deseq2_wald(counts, X)
    mc = dv.deseq2_wald(counts, X, prior_var_rule="mad_chisq")
    assert base.prior_var_rule == "trigamma" and mc.prior_var_rule == "mad_chisq"
    assert mc.squared_logres == base.squared_logres
    assert mc.prior_sampling_var == pytest.approx(dv.mad_chisq_spread(2))
    assert mc.prior_disp_var == pytest.approx(max(mc.squared_logres - dv.mad_chisq_spread(2), 0.25))
    counts4, cond4 = sim(4, G=200, seed=21)  # n - p = 6: the rule is not used
    X4 = np.column_stack([np.ones(8), cond4])
    a = dv.deseq2_wald(counts4, X4)
    b = dv.deseq2_wald(counts4, X4, prior_var_rule="simulate")
    assert b.prior_var_rule == "trigamma"
    np.testing.assert_array_equal(a.pvalue, b.pvalue)
    c = dv.deseq2_wald(counts4, X4, prior_var_rule="mad_chisq", prior_var_max_df=10)
    assert c.prior_var_rule == "mad_chisq"
    with pytest.raises(ValueError, match="prior_var_rule"):
        dv.deseq2_wald(counts4, X4, prior_var_rule="nope")


def test_simulated_spread_is_seeded_from_the_data_and_matches_no_extra_spread():
    """Counts drawn on the trend itself (no gene-to-gene spread): the observed
    spread is all sampling, so the simulated term should come close to it."""
    rng = np.random.default_rng(22)
    G = 2000
    base = np.exp(rng.uniform(np.log(20), np.log(3000), G))
    alpha = 0.08 + 2.0 / base
    cond = np.array([0.0, 0, 1, 1])
    mu = base[None] * np.array([0.8, 1.2, 1.0, 1.1])[:, None]
    counts = nb_counts(rng, mu, alpha[None])
    X = np.column_stack([np.ones(4), cond])
    a = dv.deseq2_wald(counts, X, prior_var_rule="simulate")
    b = dv.deseq2_wald(counts, X, prior_var_rule="simulate")
    assert a.prior_sampling_var == b.prior_sampling_var  # same data, same draw
    c = dv.deseq2_wald(counts, X, prior_var_rule="simulate", prior_sim_seed=1)
    assert c.prior_sampling_var != a.prior_sampling_var
    assert a.prior_sampling_var == pytest.approx(a.squared_logres, rel=0.25)
    assert c.prior_sampling_var == pytest.approx(a.prior_sampling_var, rel=0.25)
