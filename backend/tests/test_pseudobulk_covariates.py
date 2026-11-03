"""Pseudobulk DE with per-sample covariates (sex, batch, age) in the design, and the
replicate cautions: samples nested in donors, a side with under 3 donors (batch 39)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi import HTTPException

from scview.api.v1.de import DECompareRequest, differential_expression_compare
from scview.core import de_results, voom
from tests.test_pseudobulk_voom import _DM, _counts

A = [f"a{i}" for i in range(6)]
B = [f"b{i}" for i in range(6)]
FEMALE = {"a0", "a1", "a2", "b0", "b1", "b2"}


def _with_sex(a, fold: float = 6.0):
    """Sex per donor (3 F, 3 M a side) and G2 ``fold``x higher in the female donors' cells."""
    a.obs["sex"] = pd.Categorical(np.where(a.obs["donor"].astype(str).isin(FEMALE), "F", "M"))
    X = a.X.toarray()
    f = (a.obs["sex"] == "F").to_numpy()
    X[f, 2] = np.round(X[f, 2] * fold)
    a.X = type(a.X)(X)
    return a


def _req(**kw) -> DECompareRequest:
    base = dict(split_by="cond", group_a="A", group_b="B", method="pseudobulk", replicate_key="donor")
    return DECompareRequest(**{**base, **kw})


def _pseudobulk(a) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Each donor's summed counts and its condition and sex, as pseudobulk_de builds them."""
    obs = a.obs.astype(str)
    X = pd.DataFrame(a.X.toarray(), index=a.obs_names, columns=a.var_names)
    sums = X.groupby(obs["donor"]).sum().round().astype(np.int64)
    meta = obs.groupby("donor")[["cond", "sex"]].first().loc[sums.index]
    keep = (sums >= 10).sum(axis=0) >= 6
    return sums.loc[:, keep], meta


@pytest.mark.asyncio
async def test_deseq2_with_sex_matches_pydeseq2_on_the_same_design():
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.default_inference import DefaultInference
    from pydeseq2.ds import DeseqStats

    a = _with_sex(_counts(A, B, seed=3))
    res = await differential_expression_compare("x", _req(covariates=["sex"]), _DM(a))
    assert res.design == "~sex + condition" and res.covariates == ["sex"] and res.covariate_notes == []
    counts, meta = _pseudobulk(a)
    inf = DefaultInference(n_cpus=1)
    dds = DeseqDataSet(counts=counts, metadata=meta, design="~sex + cond", inference=inf, quiet=True)
    dds.deseq2()
    st = DeseqStats(dds, contrast=["cond", "A", "B"], inference=inf, quiet=True)
    st.summary()
    ref = st.results_df.dropna(subset=["log2FoldChange", "pvalue"])
    got = {g.gene: g for g in res.genes}
    assert set(got) == set(ref.index)
    for gene, row in ref.iterrows():
        assert got[gene].logfoldchange == pytest.approx(row["log2FoldChange"], rel=1e-6, abs=1e-9)
        assert got[gene].pval == pytest.approx(row["pvalue"], rel=1e-6, abs=1e-12)
    stored = de_results.recall("x", res.result_id)
    assert stored.comparison["covariates"] == ["sex"] and stored.comparison["design"] == "~sex + condition"
    assert stored.label == "All cells: A vs B (pseudobulk DESeq2 + sex)"


@pytest.mark.asyncio
async def test_voom_with_sex_is_the_moderated_t_of_that_design():
    """limma: voom(counts, model.matrix(~sex + cond)), lmFit, eBayes, coef 'condA'."""
    a = _with_sex(_counts(A, B, seed=4))
    res = await differential_expression_compare("x", _req(covariates=["sex"], pb_method="voom"), _DM(a))
    without = await differential_expression_compare("x", _req(pb_method="voom"), _DM(a))
    assert res.design == "~sex + condition"
    counts, meta = _pseudobulk(a)
    C = counts.to_numpy(dtype=np.float64)
    X = np.column_stack([np.ones(len(C)), (meta["sex"] == "M").to_numpy(float),
                         (meta["cond"] == "A").to_numpy(float)])
    lib = C.sum(axis=1) * voom.norm_factors(C, "TMM", C.sum(axis=1))
    E, w = voom.voom(C, X, lib_size=lib)
    eb = voom.ebayes(voom.lm_fit_weighted(E, X, w), 2)
    by = {g.gene: g for g in res.genes}
    for j, gene in enumerate(counts.columns):
        assert by[gene].logfoldchange == pytest.approx(eb.logfc[j], abs=1e-9)
        assert by[gene].pval == pytest.approx(eb.p[j], rel=1e-9)
    # The sex gene's residual variance falls once sex is in the model.
    assert by["G2"].stat is not None and {g.gene for g in without.genes} == set(by)


@pytest.mark.asyncio
async def test_a_numeric_covariate_is_one_slope_and_few_values_a_factor():
    a = _counts(A, B, seed=5)
    age = {d: 30 + 7 * i for i, d in enumerate(A + B)}
    a.obs["age"] = a.obs["donor"].astype(str).map(age).astype(float)
    a.obs["batch"] = a.obs["donor"].astype(str).map({d: (i % 3) + 1 for i, d in enumerate(A + B)}).astype(int)
    res = await differential_expression_compare("x", _req(covariates=["age", "batch"], pb_method="voom"), _DM(a))
    assert res.design == "~age + batch + condition" and res.covariates == ["age", "batch"]
    assert res.n_samples_a == 6 and res.genes


@pytest.mark.asyncio
async def test_bad_covariates_are_refused_in_one_sentence():
    a = _with_sex(_counts(A, B, seed=6))
    a.obs["per_cell"] = pd.Categorical(np.arange(a.n_obs) % 2)
    with pytest.raises(HTTPException) as e:
        await differential_expression_compare("x", _req(covariates=["per_cell"]), _DM(a))
    assert e.value.status_code == 400 and "varies within a sample" in e.value.detail
    a.obs["site"] = a.obs["cond"].map({"A": "S1", "B": "S2"}).astype("category")
    with pytest.raises(HTTPException) as e:
        await differential_expression_compare("x", _req(covariates=["site"]), _DM(a))
    assert "confounded with the condition (A: S1; B: S2)" in e.value.detail
    for bad, word in (("nope", "not found"), ("cond", "cannot also be a covariate"),
                      ("donor", "cannot also be a covariate")):
        with pytest.raises(HTTPException) as e:
            await differential_expression_compare("x", _req(covariates=[bad]), _DM(a))
        assert word in e.value.detail


@pytest.mark.asyncio
async def test_paired_design_already_holds_a_donor_constant():
    donors = [f"d{i}" for i in range(5)]
    a = _counts(donors, donors, seed=7)
    a.obs["sex"] = pd.Categorical(np.where(a.obs["donor"].isin(["d0", "d1"]), "F", "M"))
    res = await differential_expression_compare("x", _req(covariates=["sex"]), _DM(a))
    assert res.design == "~donor + condition" and res.covariates == []
    assert "constant within each donor" in res.covariate_notes[0]


@pytest.mark.asyncio
async def test_samples_nested_in_donors_and_few_donors_are_warned():
    # Liu-like: replicate = sample, 2 samples of p0 and p1 on side A (two time points).
    samples_a, samples_b = ["p0_t1", "p0_t2", "p1_t1", "p1_t2", "p2_t1"], ["h0", "h1", "h2", "h3"]
    a = _counts(samples_a, samples_b, seed=8)
    a.obs["sample_id"] = a.obs["donor"]
    a.obs["donor_id"] = a.obs["donor"].astype(str).str.split("_").str[0].astype("category")
    res = await differential_expression_compare("x", _req(replicate_key="sample_id"), _DM(a))
    assert res.n_samples_a == 5 and res.n_donors_a == 3 and res.n_donors_b == 4
    w = res.replicate_warnings
    assert len(w) == 1 and "Samples are not independent: 2 of 7 'donor_id' values" in w[0]
    assert "p0: 2, p1: 2" in w[0] and "use 'donor_id' as the replicate column" in w[0]
    # Summed by donor: no warning, and the paired design (when donors recur) is kept.
    res = await differential_expression_compare("x", _req(replicate_key="donor_id"), _DM(a))
    assert res.replicate_warnings == [] and res.n_samples_a == 3
    two = _counts(["a0", "a1"], ["b0", "b1", "b2"], seed=9)
    res = await differential_expression_compare("x", _req(pb_method="voom"), _DM(two))
    assert any("Fewer than 3 donors on a side (A: 2)" in x for x in res.replicate_warnings)
