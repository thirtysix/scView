"""Selection-vs-rest differential expression (volcano backend)."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from fastapi import HTTPException

from scview.api.v1.de import (
    DECompareRequest,
    DERequest,
    differential_expression,
    differential_expression_compare,
)


class _DM:
    """Minimal dataset-manager stub returning a fixed adaptor."""

    def __init__(self, adata):
        self._a = type("A", (), {"adata": adata})()

    async def get_or_load_dataset(self, _id):
        return self._a


def _two_pop(n: int = 60, g: int = 30) -> ad.AnnData:
    rng = np.random.default_rng(0)
    X = rng.normal(5, 1, size=(n, g)).astype("float32")
    X[: n // 2, :5] += 6.0  # first half over-expresses the first 5 genes
    a = ad.AnnData(np.abs(X))
    a.var_names = [f"G{i}" for i in range(g)]
    return a


@pytest.mark.asyncio
async def test_de_finds_upregulated_genes():
    a = _two_pop()
    dm = _DM(a)
    sel = list(range(a.n_obs // 2))  # the over-expressing half
    res = await differential_expression("x", DERequest(indices=sel, label="A"), dm)
    assert res.n_selected == len(sel)
    assert res.n_rest == a.n_obs - len(sel)
    assert len(res.genes) == a.n_vars
    top = sorted(res.genes, key=lambda d: d.pval_adj)[:5]
    # The planted genes should be the most significant and up-regulated.
    assert {g.gene for g in top} == {f"G{i}" for i in range(5)}
    assert all(g.logfoldchange > 0 for g in top)


@pytest.mark.asyncio
async def test_de_rejects_tiny_or_full_selection():
    a = _two_pop()
    dm = _DM(a)
    with pytest.raises(HTTPException):  # too few selected
        await differential_expression("x", DERequest(indices=[0, 1]), dm)
    with pytest.raises(HTTPException):  # nothing left to compare against
        await differential_expression("x", DERequest(indices=list(range(a.n_obs))), dm)


@pytest.mark.asyncio
async def test_de_dedupes_and_clamps_indices():
    a = _two_pop()
    dm = _DM(a)
    # Duplicates + an out-of-range index are cleaned, not fatal.
    res = await differential_expression("x", DERequest(indices=[0, 0, 1, 2, 3, 9999]), dm)
    assert res.n_selected == 4


def _two_conditions() -> ad.AnnData:
    """Cell types X/Y in conditions A/B; G0–G4 are up in A, but only in X cells."""
    rng = np.random.default_rng(1)
    n, g = 80, 30
    X = np.abs(rng.normal(5, 1, size=(n, g))).astype("float32")
    cond = np.array(["A", "B"] * (n // 2))
    ct = np.array(["X"] * (n // 2) + ["Y"] * (n // 2))
    X[(cond == "A") & (ct == "X"), :5] += 6.0
    a = ad.AnnData(X)
    a.var_names = [f"G{i}" for i in range(g)]
    a.obs["cond"] = pd.Categorical(cond)
    a.obs["ct"] = pd.Categorical(ct)
    return a


@pytest.mark.asyncio
async def test_de_compare_within_subgroup():
    dm = _DM(_two_conditions())
    body = DECompareRequest(split_by="cond", group_a="A", group_b="B",
                            within_column="ct", within_value="X")
    res = await differential_expression_compare("x", body, dm)
    assert (res.n_selected, res.n_rest) == (20, 20)
    assert res.label == "A (X)"
    top = sorted(res.genes, key=lambda d: d.pval_adj)[:5]
    assert {g.gene for g in top} == {f"G{i}" for i in range(5)}
    assert all(g.logfoldchange > 0 for g in top)

    # Within Y there is no planted difference.
    body_y = DECompareRequest(split_by="cond", group_a="A", group_b="B",
                              within_column="ct", within_value="Y")
    res_y = await differential_expression_compare("x", body_y, dm)
    assert min(g.pval_adj for g in res_y.genes) > 0.05


@pytest.mark.asyncio
async def test_de_compare_rejects_bad_requests():
    dm = _DM(_two_conditions())
    with pytest.raises(HTTPException) as e:
        await differential_expression_compare(
            "x", DECompareRequest(split_by="nope", group_a="A", group_b="B"), dm)
    assert e.value.status_code == 404
    with pytest.raises(HTTPException) as e:
        await differential_expression_compare(
            "x", DECompareRequest(split_by="cond", group_a="A", group_b="A"), dm)
    assert e.value.status_code == 400
    with pytest.raises(HTTPException) as e:  # empty subgroup
        await differential_expression_compare(
            "x", DECompareRequest(split_by="cond", group_a="A", group_b="B",
                                  within_column="ct", within_value="Z"), dm)
    assert e.value.status_code == 400


@pytest.mark.asyncio
async def test_de_on_hvg_subset_keeps_down_regulated_genes():
    # The usual imported layout: .raw holds every gene, X only the variable ones.
    full = _two_pop(g=30)
    full.X[: full.n_obs // 2, 20:25] = 0.0  # the selection loses G20-G24
    full.raw = full
    a = full[:, :10].copy()
    sel = list(range(a.n_obs // 2))
    res = await differential_expression("x", DERequest(indices=sel, label="A"), _DM(a))
    assert len(res.genes) == 30
    down = {g.gene for g in res.genes if g.logfoldchange < 0}
    assert {f"G{i}" for i in range(20, 25)} <= down
