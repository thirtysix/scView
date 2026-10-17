"""Wilcoxon DE reads log-normalised expression, never counts.

An unprocessed upload can carry counts in .raw beside a log-normalised X (Kang),
or counts alone. scanpy's rank_genes_groups uses .raw by default and takes the
fold change of whatever it is given, so on counts it reported log2 fold changes
of ~184 for ISG15. The ranks, and so the p-values, were unaffected.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import scanpy as sc

from scview.api.v1.de import DERequest, differential_expression


def _counts() -> ad.AnnData:
    rng = np.random.default_rng(0)
    X = rng.poisson(2.0, (80, 30)).astype("float32")
    X[:40, :3] += rng.poisson(20.0, (40, 3))  # three genes up in the selection
    a = ad.AnnData(X)
    a.var_names = [f"G{i}" for i in range(30)]
    a.obs_names = [f"C{i}" for i in range(80)]
    return a


def _lognorm(a: ad.AnnData) -> ad.AnnData:
    b = a.copy()
    sc.pp.normalize_total(b, target_sum=1e4)
    sc.pp.log1p(b)
    return b


class _DM:
    def __init__(self, adata):
        self.adata = adata

    async def get_or_load_dataset(self, _id):
        return type("A", (), {"adata": self.adata})()


async def _lfc(adata) -> dict[str, float]:
    res = await differential_expression("x", DERequest(indices=list(range(40))), _DM(adata))
    return {g.gene: g.logfoldchange for g in res.genes}


async def _expected() -> dict[str, float]:
    return await _lfc(_lognorm(_counts()))  # log-normalised X, no .raw


async def test_counts_in_raw_beside_lognormalised_x():
    a = _lognorm(_counts())
    a.raw = _counts()
    got, want = await _lfc(a), await _expected()
    assert max(abs(v) for v in got.values()) < 10
    assert np.allclose([got[g] for g in want], list(want.values()), atol=1e-4)


async def test_counts_only_are_log_normalised_first():
    got, want = await _lfc(_counts()), await _expected()
    assert np.allclose([got[g] for g in want], list(want.values()), atol=1e-4)
