"""Cell-level DE tests side by side (core/de_methods.py, POST .../de with method,
POST .../de/methods and .../de/compare/methods)."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from scipy import stats

from scview.api.v1.de import (
    DECompareMethodsRequest,
    DECompareRequest,
    DEMethodsRequest,
    DERequest,
    differential_expression,
    differential_expression_compare,
    differential_expression_compare_methods,
    differential_expression_methods,
)
from scview.core import de_methods, de_results, gsea


class _DM:
    def __init__(self, adata):
        self._a = type("A", (), {"adata": adata})()

    async def get_or_load_dataset(self, _id):
        return self._a


def _log_norm(n: int = 120, g: int = 40, seed: int = 0) -> ad.AnnData:
    """Sparse log-normalised data; the first half over-expresses G0-G4 and
    under-expresses G35-G39; G30 is never detected."""
    rng = np.random.default_rng(seed)
    counts = rng.poisson(1.0, (n, g)).astype("float32")
    counts[: n // 2, :5] += rng.poisson(6.0, (n // 2, 5))
    counts[n // 2:, 35:] += rng.poisson(6.0, (n - n // 2, 5))
    counts[:, 30] = 0
    a = ad.AnnData(sp.csr_matrix(counts))
    a.var_names = [f"G{i}" for i in range(g)]
    a.obs["cond"] = pd.Categorical(np.where(np.arange(n) < n // 2, "A", "B"))
    a.obs["type"] = pd.Categorical(np.tile(["X", "Y"], n // 2))
    import scanpy as sc

    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    return a


@pytest.fixture(autouse=True)
def _fresh():
    de_results.clear()
    yield
    de_results.clear()


@pytest.mark.parametrize("method", ["t-test", "logreg"])
async def test_each_method_finds_the_planted_genes(method):
    a = _log_norm()
    res = await differential_expression(
        "x", DERequest(indices=list(range(60)), label="A", method=method), _DM(a))
    assert res.method == method
    top5 = {g.gene for g in res.genes[:5]}  # highest statistic first
    assert top5 == {f"G{i}" for i in range(5)}
    bottom5 = {g.gene for g in res.genes[-5:]}
    assert bottom5 == {f"G{i}" for i in range(35, 40)}
    assert all(g.logfoldchange > 0 for g in res.genes[:5])
    # The 1% prefilter applies to every method: the undetected gene is not tested.
    assert res.n_genes_skipped == 1 and "G30" not in {g.gene for g in res.genes}


async def test_logreg_gives_coefficients_and_no_p_values():
    a = _log_norm()
    res = await differential_expression(
        "x", DERequest(indices=list(range(60)), method="logreg"), _DM(a))
    assert all(g.pval is None and g.pval_adj is None for g in res.genes)
    assert all(g.stat is not None for g in res.genes)
    dumped = res.model_dump()["genes"][0]
    assert dumped["pval"] is None and dumped["pval_adj"] is None
    # Fold changes as scanpy takes them, from the same tested matrix.
    wil = await differential_expression("x", DERequest(indices=list(range(60))), _DM(a))
    lfc_w = {g.gene: g.logfoldchange for g in wil.genes}
    for g in res.genes:
        assert g.logfoldchange == pytest.approx(lfc_w[g.gene], rel=1e-4, abs=1e-4)


async def test_the_t_test_is_welchs():
    a = _log_norm()
    res = await differential_expression(
        "x", DERequest(indices=list(range(60)), method="t-test", min_pct=0), _DM(a))
    X = a.X.toarray()
    for g in res.genes[:3]:
        j = list(a.var_names).index(g.gene)
        t, p = stats.ttest_ind(X[:60, j], X[60:, j], equal_var=False)
        assert g.stat == pytest.approx(t, rel=1e-3)
        assert g.pval == pytest.approx(p, rel=1e-3, abs=1e-300)


async def test_wilcoxon_stays_the_default():
    a = _log_norm()
    res = await differential_expression("x", DERequest(indices=list(range(60))), _DM(a))
    assert res.method == "wilcoxon" and all(g.pval is not None for g in res.genes)


async def test_stored_results_record_the_method():
    a = _log_norm()
    for method, metric in (("wilcoxon", "wilcoxon_score"), ("t-test", "t_score"),
                           ("logreg", "logreg_coef")):
        res = await differential_expression(
            "x", DERequest(indices=list(range(60)), method=method), _DM(a))
        stored = de_results.recall("x", res.result_id)
        assert stored.method == method and stored.score_metric == metric
        rows = stored.rows()
        assert all(r["score"] is not None for r in rows)  # GSEA ranks by the statistic
        ranking = gsea.ranking_from_de(rows, exclude_ribo_mito=False,
                                       score_metric=stored.score_metric)
        assert ranking.metric == metric and metric in gsea.RANK_LABELS
        genes, stats_ = stored.profile()  # TF / pathway activity input
        assert len(genes) == len(stats_) == len(res.genes)
    # A logreg result has no p-value to fall back on: its pvals are NaN.
    assert np.isnan(stored.pvals).all()


async def test_compare_runs_the_chosen_cell_level_method():
    a = _log_norm()
    body = DECompareRequest(split_by="cond", group_a="A", group_b="B",
                            within_column="type", within_value="X", method="t-test")
    res = await differential_expression_compare("x", body, _DM(a))
    assert res.method == "t-test" and res.n_selected == res.n_rest == 30
    assert {g.gene for g in res.genes[:5]} == {f"G{i}" for i in range(5)}
    res = await differential_expression_compare(
        "x", body.model_copy(update={"method": "logreg"}), _DM(a))
    assert res.method == "logreg" and res.genes[0].pval is None


async def test_methods_agree_on_a_clear_difference():
    a = _log_norm()
    res = await differential_expression_methods(
        "x", DEMethodsRequest(indices=list(range(60)), label="A", top_n=5), _DM(a))
    assert [m.method for m in res.methods] == ["wilcoxon", "t-test", "logreg"]
    by = {m.method: m for m in res.methods}
    assert by["logreg"].has_pvalues is False and by["logreg"].n_significant is None
    assert by["wilcoxon"].n_significant >= 10 and by["t-test"].n_significant >= 10
    assert by["t-test"].label == "Welch t-test"
    assert "MAST" in res.unavailable
    assert res.n_genes_tested == 39 and res.n_genes_skipped == 1 and res.min_pct == 0.01
    assert len(res.pairs) == 3
    for p in res.pairs:
        # Top 5 up = the 5 planted genes for every test, and the same for down.
        assert p.jaccard_up == 1.0 and p.shared_up == 5
        assert p.jaccard_down == 1.0
        assert p.spearman is not None and -1 <= p.spearman <= 1
    assert res.unique == []
    wt = next(p for p in res.pairs if {p.a, p.b} == {"wilcoxon", "t-test"})
    assert wt.spearman > 0.8


async def test_compare_methods_scopes_the_cells():
    a = _log_norm()
    body = DECompareMethodsRequest(split_by="cond", group_a="A", group_b="B",
                                   within_column="type", within_value="Y", top_n=5)
    res = await differential_expression_compare_methods("x", body, _DM(a))
    assert (res.n_selected, res.n_rest, res.label) == (30, 30, "A (Y)")
    assert len(res.methods) == 3 and res.top_n == 5


def _result(method: str, scores: list[float]) -> de_methods.MethodResult:
    g = np.array([f"G{i}" for i in range(len(scores))])
    s = np.asarray(scores, dtype=float)
    return de_methods.MethodResult(method, g, s, s, None, None, 0.0)


def test_agreement_measures_overlap_rank_correlation_and_unique_genes():
    a = _result("wilcoxon", [5, 4, 3, 2, 1, 0, -1, -2])
    b = _result("t-test", [4, 5, 2, 3, 1, 0, -2, -1])  # same tops, order swapped
    c = _result("logreg", [0, 1, 5, 2, 4, 3, -2, -1])  # tops G2, G4
    out = de_methods.agreement([a, b, c], top_n=2)
    pairs = {(p["a"], p["b"]): p for p in out["pairs"]}
    ab = pairs[("wilcoxon", "t-test")]
    assert ab["jaccard_up"] == 1.0 and ab["shared_up"] == 2 and ab["jaccard_down"] == 1.0
    assert ab["spearman"] == pytest.approx(stats.spearmanr(a.scores, b.scores).statistic, abs=1e-4)
    ac = pairs[("wilcoxon", "logreg")]
    assert ac["jaccard_up"] == 0.0 and ac["shared_up"] == 0
    only = {(u["method"], u["side"], u["gene"]): u["ranks"] for u in out["unique"]}
    # G0 / G1 top both univariate tests, so neither has a gene of its own.
    assert set(only) == {("logreg", "up", "G2"), ("logreg", "up", "G4")}
    assert only[("logreg", "up", "G2")] == {"wilcoxon": 3, "t-test": 4, "logreg": 1}


def test_agreement_skips_non_finite_scores_and_constant_ones():
    a = _result("wilcoxon", [1.0, np.nan, 3.0, 2.0])
    b = _result("t-test", [0.0, 0.0, 0.0, 0.0])
    out = de_methods.agreement([a, b], top_n=2)
    assert out["pairs"][0]["spearman"] is None  # b has no ranking at all
    ranks = {u["gene"]: u["ranks"] for u in out["unique"] if u["method"] == "wilcoxon"}
    assert "G1" not in ranks  # a NaN statistic is never ranked


def test_an_unknown_method_is_refused():
    with pytest.raises(de_methods.MethodError):
        de_methods.run(ad.AnnData(np.ones((4, 2))), "mast")
