"""edgeR v4 quasi-likelihood (core/edger_ql.py) against R edgeR 4.6.3 / limma 3.64.3.

The reference numbers in ``fixtures/edger_ql/`` come from ``fixtures/edger_ql/edger_ql.R``
(glmQLFit legacy=FALSE + glmQLFTest on simulated counts): two groups 4 v 4 with genes silent in
one group, very low counts 3 v 3, a paired design (4 donors x 2 conditions) and a group +
continuous covariate design testing the covariate.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy import optimize, special

from scview.core import edger_ql as eq
from scview.core import voom

FIX = Path(__file__).parent / "fixtures" / "edger_ql"
CASES = ("two_group", "low", "paired", "covariate")


def _case(name: str):
    data = name.replace("_robust", "")
    C = np.loadtxt(FIX / f"{data}.counts.tsv").T  # samples x genes
    X = np.loadtxt(FIX / f"{data}.design.tsv")
    genes = np.genfromtxt(FIX / f"{name}.genes.tsv", names=True, delimiter="\t")
    with open(FIX / f"{name}.fit.tsv") as f:
        keys = f.readline().rstrip("\n").split("\t")
        vals = f.readline().rstrip("\n").split("\t")
    fit = dict(zip(keys, vals))
    return C, X, genes, fit


@pytest.fixture(scope="module", params=CASES)
def case(request):
    C, X, genes, fit = _case(request.param)
    res = eq.glm_ql(C, X, int(fit["coef"]) - 1)
    return request.param, res, genes, fit


def _dlog10p(a, b):
    return np.abs(np.log10(np.maximum(a, 1e-300)) - np.log10(np.maximum(b, 1e-300)))


def test_matches_r_hyperparameters(case):
    name, res, genes, fit = case
    nf = np.array([float(v) for v in fit["norm_factors"].split(",")])
    np.testing.assert_allclose(res.norm_factors, nf, rtol=1e-12)
    np.testing.assert_allclose(res.AveLogCPM, genes["AveLogCPM"], rtol=0, atol=1e-9)
    assert res.dispersion == pytest.approx(float(fit["dispersion"]), rel=1e-7)
    assert res.average_ql_dispersion == pytest.approx(float(fit["average_ql"]), rel=1e-3)
    assert res.df_prior == pytest.approx(genes["df_prior"][0], rel=5e-3)
    np.testing.assert_allclose(res.s2_prior, genes["s2_prior"], rtol=2e-3)


def test_matches_r_tests(case):
    name, res, genes, fit = case
    reg = genes["df_residual_adj"] > 0.5  # leaves out genes fitted by diverging coefficients
    np.testing.assert_allclose(res.df_resid[reg], genes["df_residual_adj"][reg], rtol=2e-3)
    np.testing.assert_allclose(res.deviance_adj[reg], genes["deviance_adj"][reg], rtol=5e-3, atol=1e-6)
    np.testing.assert_allclose(res.logFC[reg], genes["logFC"][reg], rtol=0, atol=2e-4)
    d = _dlog10p(res.pvalue, genes["PValue"])
    assert np.median(d) < 1e-4
    assert np.max(d) < 0.02
    assert np.mean((res.pvalue < 0.05) == (genes["PValue"] < 0.05)) > 0.99
    assert np.mean((res.padj < 0.05) == (genes["FDR"] < 0.05)) > 0.99


def test_two_group_close_to_r():
    """The one-way case agrees to the precision of R's own moment approximation."""
    C, X, genes, fit = _case("two_group")
    res = eq.glm_ql(C, X, 1)
    d = _dlog10p(res.pvalue, genes["PValue"])
    assert np.median(d) < 5e-5
    assert np.max(d) < 1e-2
    np.testing.assert_allclose(res.F, genes["F"], rtol=2e-3, atol=1e-6)


def test_robust_not_implemented():
    C, X, _g, _f = _case("two_group")
    with pytest.raises(NotImplementedError):
        eq.glm_ql(C, X, 1, robust=True)


def test_rejects_bad_designs():
    C, X, _g, _f = _case("two_group")
    with pytest.raises(ValueError):
        eq.glm_ql(C, X[:, :1], 0)
    with pytest.raises(ValueError):
        eq.glm_ql(C, np.column_stack([X, X[:, 1]]), 1)
    with pytest.raises(ValueError):
        eq.glm_ql(C[:5], X, 1)


# ---------------------------------------------------------------------------
# building blocks
# ---------------------------------------------------------------------------


def test_brent_min_finds_the_minimum():
    f = lambda x: (x - 0.3141) ** 2 + 0.1 * np.cos(5 * x)  # noqa: E731
    x = eq.brent_min(f, 0.0, 1.0, 1e-10)
    ref = optimize.minimize_scalar(f, bounds=(0, 1), method="bounded", options={"xatol": 1e-12}).x
    assert x == pytest.approx(ref, abs=1e-8)


def test_tmm_factors_equal_voom():
    rng = np.random.default_rng(3)
    for _ in range(5):
        C = np.round(rng.negative_binomial(5, 0.3, (6, 500)) * rng.uniform(0.5, 2, (6, 1)))
        lib = C.sum(axis=1)
        np.testing.assert_allclose(eq.tmm_factors(C, lib), voom.norm_factors(C, "TMM", lib), rtol=1e-14)


def _brute_moments(m, phi):
    r = 1 / phi
    y = np.arange(0, int(m + 40 * np.sqrt(m * (1 + phi * m)) + 60 * (1 + phi * m) + 50), dtype=float)
    p = np.exp(special.gammaln(y + r) - special.gammaln(r) - special.gammaln(y + 1)
               + r * np.log(r / (r + m)) + y * np.log(m / (r + m)))
    d = eq.unit_deviance(y, m, phi)
    E = (p * d).sum() / p.sum()
    return E, (p * (d - E) ** 2).sum() / p.sum()


@pytest.mark.parametrize("phi", [0.01, 0.2, 2.0])
def test_deviance_moments_against_brute_force(phi):
    ms = np.array([1e-4, 0.03, 0.7, 4.0, 60.0, 900.0, 7000.0])
    E, V = eq.deviance_moments(ms, phi)
    for i, m in enumerate(ms):
        Eb, Vb = _brute_moments(m, phi)
        assert E[i] == pytest.approx(Eb, rel=2e-5)
        assert V[i] == pytest.approx(Vb, rel=2e-5)
    k, s = eq.UnitMoments(phi)(ms)
    np.testing.assert_allclose(k, 2 * E**2 / V, rtol=2e-4)
    np.testing.assert_allclose(s, V / (2 * E), rtol=2e-4)


def test_lowess_delta_is_the_fixed_point():
    x = np.random.default_rng(1).normal(size=3000)
    d = eq.lowess_delta(x, 200)
    gaps = np.diff(np.unique(x))
    assert d * 200 == pytest.approx(np.minimum(gaps, d).sum(), rel=1e-12)
    assert eq.lowess_delta(np.repeat(np.arange(50.0), 3), 200) == 0.0


def test_compiled_kernels_match_numpy_paths(monkeypatch):
    """numba kernels and the numpy fallbacks give the same smoothers, fits and adjusted deviances."""
    rng = np.random.default_rng(2)
    x = rng.normal(size=800)
    y = np.sin(x) + rng.normal(scale=0.3, size=800)
    w = rng.gamma(2.0, size=800)
    C, X, _g, _f = _case("low")
    off = np.log(C.sum(axis=1))
    groups = eq.design_groups(X)
    with_k = (eq.weighted_lowess(x, y, w, 0.6, eq.lowess_delta(x)), eq.r_lowess(x, y, 0.5, 3),
              eq.glm_fit(C, X, off, 0.2, groups)[1], eq.adjusted_deviance(C, eq.glm_fit(C, X, off, 0.2, groups)[1], X,
                                                                          0.25, 1.2, groups)[:2],
              eq.common_dispersion(C, X, off, groups))
    monkeypatch.setattr(eq, "_KERNELS", {})
    without = (eq.weighted_lowess(x, y, w, 0.6, eq.lowess_delta(x)), eq.r_lowess(x, y, 0.5, 3),
               eq.glm_fit(C, X, off, 0.2, groups)[1], eq.adjusted_deviance(C, eq.glm_fit(C, X, off, 0.2, groups)[1], X,
                                                                           0.25, 1.2, groups)[:2],
               eq.common_dispersion(C, X, off, groups))
    np.testing.assert_allclose(with_k[0], without[0], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(with_k[1], without[1], rtol=1e-7, atol=1e-9)  # statsmodels ~ R to 1e-9
    np.testing.assert_allclose(with_k[2], without[2], rtol=1e-8)
    np.testing.assert_allclose(with_k[3][0], without[3][0], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(with_k[3][1], without[3][1], rtol=1e-10, atol=1e-12)
    assert with_k[4] == pytest.approx(without[4], rel=1e-6)
