"""Pseudobulk DE with limma-voom / voomByGroup, the per-group variance check and
RUV factors (core/pseudobulk.py through /de/compare)."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from fastapi import HTTPException
from pydantic import ValidationError

from scview.api.v1.de import DECompareRequest, differential_expression_compare
from scview.core import claims, de_results, pseudobulk


class _Adaptor:
    def __init__(self, adata):
        self.adata = adata


class _DM:
    def __init__(self, adata):
        self.adaptor = _Adaptor(adata)

    async def get_or_load_dataset(self, _id):
        return self.adaptor


def _counts(donors_a, donors_b, *, sd_a=0.3, sd_b=0.3, tech=None, cells=30, g=150, seed=0):
    """NB counts per donor x condition; G0 4x up in A, G1 4x up in B. ``sd_*``:
    the donor-to-donor log spread per condition; ``tech``: per-sample multipliers
    on the odd genes (unwanted variation unrelated to the condition)."""
    rng = np.random.default_rng(seed)
    base = rng.gamma(2.0, 1.0, size=g) + 0.5
    rows, cond, donor = [], [], []
    k = 0
    for c, ds, sd in (("A", donors_a, sd_a), ("B", donors_b, sd_b)):
        for d in ds:
            mu = np.tile(base * rng.lognormal(0, sd, size=g), (cells, 1))
            mu[:, 0 if c == "A" else 1] *= 4.0
            if tech is not None:
                mu[:, 3::2] *= tech[k]
            k += 1
            rows.append(rng.poisson(rng.gamma(1 / 0.1, mu * 0.1)))
            cond += [c] * cells
            donor += [d] * cells
    a = ad.AnnData(sp.csr_matrix(np.vstack(rows).astype("float32")))
    a.var_names = [f"G{i}" for i in range(g)]
    a.obs_names = [f"c{i}" for i in range(a.n_obs)]
    a.obs["cond"] = pd.Categorical(cond)
    a.obs["donor"] = pd.Categorical(donor)
    return a


def _req(**kw) -> DECompareRequest:
    return DECompareRequest(split_by="cond", group_a="A", group_b="B", method="pseudobulk",
                            replicate_key="donor", **kw)


DONORS = [f"d{i}" for i in range(5)]


@pytest.mark.asyncio
@pytest.mark.parametrize("pb", ["voom", "voombygroup"])
async def test_voom_methods_find_the_planted_genes(pb):
    a = _counts(DONORS, DONORS)
    res = await differential_expression_compare("x", _req(pb_method=pb), _DM(a))
    assert res.method == "pseudobulk" and res.pb_method == pb and res.ruv_k == 0
    assert res.design == "~donor + condition"
    by = {g.gene: g for g in res.genes}
    assert by["G0"].pval_adj < 1e-3 and by["G0"].logfoldchange == pytest.approx(2.0, abs=0.5)
    # This seed's donors put G1 at -2.76 under DESeq2 too.
    assert by["G1"].pval_adj < 1e-3 and by["G1"].logfoldchange == pytest.approx(-2.76, abs=0.15)
    assert sum(g.pval_adj < 0.05 for g in res.genes if g.gene not in ("G0", "G1")) <= 3
    assert res.genes[0].gene == "G0" and res.genes[-1].gene == "G1"  # ranked by t
    assert by["G0"].stat > 0 > by["G1"].stat
    stored = de_results.recall("x", res.result_id)
    assert stored.comparison["pb_method"] == pb and stored.comparison["ruv_k"] == 0
    name = "limma-voom" if pb == "voom" else "voomByGroup"
    assert stored.label == f"All cells: A vs B (pseudobulk {name})"


@pytest.mark.asyncio
async def test_default_stays_deseq2_and_adds_the_variance_check():
    a = _counts(DONORS, DONORS)
    res = await differential_expression_compare("x", _req(), _DM(a))
    assert res.pb_method == "deseq2" and res.ruv_k == 0 and res.ruv is None
    assert res.design == "~donor + condition"
    assert de_results.recall("x", res.result_id).label == "All cells: A vs B (pseudobulk DESeq2)"
    h = res.heteroscedasticity
    assert [g["group"] for g in h["groups"]] == ["A", "B"]
    assert all(g["n_samples"] == 5 and len(g["trend"]) >= 2 for g in h["groups"])
    assert h["flag"] is False and h["ratio"] < pseudobulk.HETERO_RATIO


@pytest.mark.asyncio
async def test_variance_check_flags_a_group_whose_donors_vary_more():
    a = _counts(["a0", "a1", "a2", "a3", "a4"], ["b0", "b1", "b2", "b3", "b4"], sd_a=0.9, sd_b=0.1)
    res = await differential_expression_compare("x", _req(pb_method="voom"), _DM(a))
    h = res.heteroscedasticity
    assert h["flag"] is True and h["higher"] == "A" and h["ratio"] >= 2
    assert "Try voomByGroup" in h["message"]
    res = await differential_expression_compare("x", _req(pb_method="voombygroup"), _DM(a))
    assert "(in use)" in res.heteroscedasticity["message"]


@pytest.mark.asyncio
@pytest.mark.parametrize("pb", ["deseq2", "voom"])
async def test_ruv_factor_recovers_planted_unwanted_variation(pb):
    tech = np.array([1.0, 2.5, 0.5, 1.8, 0.7, 0.6, 2.2, 1.0, 0.4, 1.5])
    a = _counts(DONORS, DONORS, tech=tech, sd_a=0.1, sd_b=0.1)
    res = await differential_expression_compare("x", _req(pb_method=pb, ruv_k=1), _DM(a))
    assert res.ruv_k == 1 and res.design == "~W1 + donor + condition"
    r = res.ruv
    assert r["k"] == 1 and len(r["factors"]) == 1 and len(r["samples"]) == 10
    assert r["n_controls"] == res.n_genes_tested // 2
    # Samples come in (condition, donor) order: A d0..d4, then B d0..d4.
    w = np.array([s["W"][0] for s in sorted(r["samples"], key=lambda s: (s["group"], s["donor"]))])
    # Paired: the factor is the within-donor part of the planted effect.
    lt = np.log(tech)
    within = lt - np.tile((lt[:5] + lt[5:]) / 2, 2)
    assert abs(np.corrcoef(w, within)[0, 1]) > 0.95
    f = r["factors"][0]
    assert f["r2_donor"] is not None and f["r2_donor"] < 0.5
    by = {g.gene: g for g in res.genes}
    assert by["G0"].pval_adj < 0.01 and by["G1"].pval_adj < 0.01
    stored = de_results.recall("x", res.result_id)
    assert stored.comparison["ruv_k"] == 1 and "+ RUV k=1" in stored.label


@pytest.mark.asyncio
async def test_ruv_needs_residual_degrees_of_freedom():
    a = _counts(["a0", "a1"], ["b0", "b1"])
    with pytest.raises(HTTPException) as e:
        await differential_expression_compare("x", _req(pb_method="voom", ruv_k=2), _DM(a))
    assert e.value.status_code == 400 and "smaller k" in e.value.detail
    res = await differential_expression_compare("x", _req(pb_method="voom", ruv_k=1), _DM(a))
    assert res.ruv_k == 1 and res.ruv["factors"][0]["r2_donor"] is None  # unpaired


def test_request_bounds():
    with pytest.raises(ValidationError):
        _req(ruv_k=4)
    with pytest.raises(ValidationError):
        _req(pb_method="edger")
    with pytest.raises(pseudobulk.PseudobulkError):
        pseudobulk.pseudobulk_de(_counts(DONORS, DONORS), np.ones(1, bool), pd.Series(["d"]),
                                 method="edger")


@pytest.mark.asyncio
async def test_copilot_line_names_the_last_run():
    a = _counts(["a0", "a1", "a2", "a3", "a4"], ["b0", "b1", "b2", "b3", "b4"], sd_a=0.9, sd_b=0.1)
    dm = _DM(a)
    assert pseudobulk.grounding_lines(dm.adaptor) == ([], None)
    await differential_expression_compare("x", _req(pb_method="voom"), dm)
    lines, detail = pseudobulk.grounding_lines(dm.adaptor)
    assert "limma-voom" in detail and "5 vs 5" in detail
    assert pseudobulk.TAG in lines[1]
    assert any("voomByGroup suggested" in ln for ln in lines)
    assert claims.resolve_tag("pseudobulk", [pseudobulk.TAG]) == [pseudobulk.TAG]
