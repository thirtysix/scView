"""Transcriptome-size caution before DE between groups that differ in RNA content
(core/transcriptome_size.py)."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
import scipy.sparse as sp

from scview.api.v1.de import (
    DECompareRequest,
    DERequest,
    differential_expression,
    differential_expression_compare,
)
from scview.core import transcriptome_size as ts


class _DM:
    def __init__(self, adata):
        self._a = type("A", (), {"adata": adata})()

    async def get_or_load_dataset(self, _id):
        return self._a


def _counts(seed: int = 0) -> ad.AnnData:
    """120 cells x 50 genes: 'big' cells (the first 60) hold 3x the RNA of 'small'
    ones, in both conditions; the same share of each gene per cell."""
    rng = np.random.default_rng(seed)
    share = rng.dirichlet(np.ones(50))
    depth = np.r_[np.full(60, 3000.0), np.full(60, 1000.0)]
    X = rng.poisson(np.outer(depth, share)).astype("float32")
    a = ad.AnnData(sp.csr_matrix(X))
    a.var_names = [f"G{i}" for i in range(50)]
    a.obs_names = [f"C{i}" for i in range(120)]
    a.obs["cell_type"] = pd.Categorical(["big"] * 60 + ["small"] * 60)
    a.obs["cond"] = pd.Categorical(["STIM", "CTRL"] * 60)
    return a


def _processed(a: ad.AnnData) -> ad.AnnData:
    """The pipeline's layout: counts in layers, log-normalised X and .raw."""
    b = a.copy()
    b.layers["counts"] = b.X.copy()
    sc.pp.normalize_total(b, target_sum=1e4)
    sc.pp.log1p(b)
    b.raw = b.copy()
    return b


# --- core -----------------------------------------------------------------------


def test_totals_come_from_obs_first_then_the_counts():
    a = _counts()
    sums = np.asarray(a.X.sum(axis=1)).ravel()
    v, src = ts.per_cell_totals(a)
    assert src == "X" and np.allclose(v, sums)
    b = _processed(a)
    v, src = ts.per_cell_totals(b)
    assert src == "layers['counts']" and np.allclose(v, sums)
    b.obs["nCount_RNA"] = 7.0
    b.obs["n_counts"] = 5.0
    assert ts.per_cell_totals(b)[1] == "obs['n_counts']"  # scanpy's name before Seurat's
    b.obs["total_counts"] = np.nan  # an unusable column is passed over
    assert ts.per_cell_totals(b)[1] == "obs['n_counts']"


def test_normalised_values_are_never_totals():
    b = _processed(_counts())
    del b.layers["counts"]  # log X and log .raw only: no counts anywhere
    assert ts.per_cell_totals(b) is None
    assert ts.check(b, np.arange(120) < 60, np.arange(120) >= 60) is None


def test_compare_medians_ratio_and_a_capped_seeded_sample():
    totals = np.r_[np.full(3000, 4000.0), np.full(500, 1000.0)]
    a = np.arange(3500) < 3000
    out = ts.compare(totals, "obs['n_counts']", a, ~a)
    assert (out["median_a"], out["median_b"], out["ratio"], out["caution"]) == (4000.0, 1000.0, 4.0, True)
    assert (len(out["values_a"]), len(out["values_b"])) == (ts.VIOLIN_MAX, 500)
    assert ts.compare(totals, "x", a, ~a)["values_a"] == out["values_a"]  # same draw each time
    # The ratio is larger over smaller whichever side is bigger; under 1.5x no caution.
    half = np.arange(20) < 10
    out = ts.compare(np.r_[np.full(10, 1000.0), np.full(10, 1450.0)], "x", half, ~half)
    assert (out["ratio"], out["caution"], out["threshold"]) == (1.45, False, 1.5)
    out = ts.compare(np.r_[np.full(10, 1500.0), np.full(10, 1000.0)], "x", half, ~half)
    assert (out["ratio"], out["caution"]) == (1.5, True)  # equal-copy genes shift by log2 FC 0.58


# --- endpoints --------------------------------------------------------------------


async def test_cross_type_compare_carries_the_caution():
    dm = _DM(_processed(_counts()))
    res = await differential_expression_compare(
        "x", DECompareRequest(split_by="cell_type", group_a="big", group_b="small"), dm)
    size = res.transcriptome_size
    assert size.source == "layers['counts']"
    assert size.caution is True and size.ratio == pytest.approx(3.0, rel=0.05)
    assert size.median_a == pytest.approx(3000, rel=0.05) and size.median_b == pytest.approx(1000, rel=0.05)
    assert len(size.values_a) == len(size.values_b) == 60
    # Within one cell type the two conditions hold the same RNA: no caution.
    res = await differential_expression_compare(
        "x", DECompareRequest(split_by="cond", group_a="STIM", group_b="CTRL",
                              within_column="cell_type", within_value="big"), dm)
    assert res.transcriptome_size.caution is False
    assert res.transcriptome_size.ratio < 1.2


async def test_selection_vs_rest_reads_totals_before_x_is_normalised():
    # Counts only in X: the Wilcoxon log-normalises its copy; the totals are the counts'.
    a = _counts()
    res = await differential_expression("x", DERequest(indices=list(range(60))), _DM(a))
    sums = np.asarray(a.X.sum(axis=1)).ravel()
    assert res.transcriptome_size.source == "X"
    assert res.transcriptome_size.median_a == pytest.approx(float(np.median(sums[:60])))
    assert res.transcriptome_size.caution is True


async def test_pseudobulk_carries_it_too_and_no_counts_means_none():
    a = _counts()
    a.obs["donor"] = pd.Categorical([f"d{i % 4}" for i in range(120)])
    a.obs["n_counts"] = np.asarray(a.X.sum(axis=1)).ravel()
    res = await differential_expression_compare(
        "x", DECompareRequest(split_by="cell_type", group_a="big", group_b="small",
                              method="pseudobulk", replicate_key="donor"), _DM(a))
    assert res.method == "pseudobulk"
    assert res.transcriptome_size.source == "obs['n_counts']" and res.transcriptome_size.caution

    b = _processed(_counts())
    del b.layers["counts"]
    res = await differential_expression_compare(
        "x", DECompareRequest(split_by="cell_type", group_a="big", group_b="small"), _DM(b))
    assert res.transcriptome_size is None
