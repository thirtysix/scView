"""Selection-vs-rest differential expression (volcano backend)."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pytest
from fastapi import HTTPException

from scview.api.v1.de import DERequest, differential_expression


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
