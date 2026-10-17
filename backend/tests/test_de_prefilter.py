"""DE gene prefilter: test only genes detected in enough cells of either group
(core/de_prefilter.py), and say how many were tested and skipped."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
import scipy.sparse as sp
from fastapi import HTTPException

from scview.api.v1.de import (
    DECompareRequest,
    DERequest,
    differential_expression,
    differential_expression_compare,
)
from scview.core import de_prefilter, de_results


class _DM:
    def __init__(self, adata):
        self._a = type("A", (), {"adata": adata})()

    async def get_or_load_dataset(self, _id):
        return self._a


@pytest.fixture(autouse=True)
def _fresh_store():
    de_results.clear()
    yield
    de_results.clear()


def _cells(n: int = 100, seed: int = 0, rare: tuple[int, ...] = (0, 1, 2, 50, 51, 52)) -> ad.AnnData:
    """Counts for 100 cells (A = the first 40) x 40 genes: G0-G29 common, G30-G34
    in the ``rare`` cells only (3 of each side: 7.5% of A, 5% of B), G35-G39 in 10
    of the 40 A cells (25%) only."""
    rng = np.random.default_rng(seed)
    X = rng.poisson(3.0, (n, 40)).astype("float32")
    X[:40, :3] += rng.poisson(15.0, (40, 3))  # three genes up in A
    X[:, 30:] = 0
    X[list(rare), 30:35] = 4
    X[5:15, 35:40] = 6  # 25% of A, none of B
    a = ad.AnnData(sp.csr_matrix(X))
    a.var_names = [f"G{i}" for i in range(40)]
    a.obs_names = [f"C{i}" for i in range(n)]
    a.obs["grp"] = pd.Categorical(["A"] * 40 + ["B"] * (n - 40))
    a.layers["counts"] = a.X.copy()
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    a.raw = a.copy()  # .raw log-normalised, as the pipeline leaves it
    return a


SEL = list(range(40))


# --- core -----------------------------------------------------------------------


def test_prefilter_keeps_genes_detected_in_either_group():
    M = np.zeros((10, 4))
    in_a = np.array([True] * 4 + [False] * 6)
    M[:1, 0] = 1  # 1/4 of A
    M[4:5, 1] = 1  # 1/6 of B
    M[:, 2] = 1  # everywhere
    pre = de_prefilter.prefilter(sp.csr_matrix(M), in_a, 0.2)
    assert pre.keep.tolist() == [True, False, True, False]
    assert (pre.n_tested, pre.n_skipped, pre.min_pct) == (2, 2, 0.2)
    # The larger fraction per gene on a 0.01 % grid: 0.25, 0.1667, 1, 0.
    h = np.asarray(pre.histogram)
    assert h.size == de_prefilter.HIST_BINS + 1 and h.sum() == 4
    assert h[2500] == 1 and h[1666] == 1 and h[10_000] == 1 and h[0] == 1


def test_a_cut_off_on_the_grid_is_exact():
    # 3 of 30 cells is exactly 10%: tested at 0.1, and in the 10% bin, not below it.
    M = np.zeros((40, 2))
    in_a = np.array([True] * 30 + [False] * 10)
    M[:3, 0] = 1
    M[:2, 1] = 1
    pre = de_prefilter.prefilter(M, in_a, 0.1)
    assert pre.keep.tolist() == [True, False]
    assert pre.histogram[1000] == 1 and pre.histogram[666] == 1


def test_zero_tests_every_gene_and_scaled_values_cannot_be_filtered():
    M = np.zeros((10, 3))
    M[0, 0] = 1
    in_a = np.arange(10) < 5
    pre = de_prefilter.prefilter(M, in_a, 0.0)
    assert pre.keep.all() and pre.min_pct == 0.0 and sum(pre.histogram) == 3
    z = np.random.default_rng(0).normal(size=(10, 3))  # z-scores: no zero for "absent"
    pre = de_prefilter.prefilter(z, in_a, 0.5)
    assert pre.keep.all() and pre.min_pct is None and pre.histogram == []


# --- endpoints --------------------------------------------------------------------


async def _de(a, **kw):
    return await differential_expression("x", DERequest(indices=SEL, label="A", **kw), _DM(a))


async def test_selection_tests_only_detected_genes():
    res = await _de(_cells(), min_pct=0.1)
    tested = {g.gene for g in res.genes}
    assert tested == {f"G{i}" for i in range(30)} | {f"G{i}" for i in range(35, 40)}
    assert (res.min_pct, res.n_genes_tested, res.n_genes_skipped) == (0.1, 35, 5)
    assert len(res.detection) == de_prefilter.HIST_BINS + 1 and sum(res.detection) == 40
    assert res.detection[750] == 5  # G30-G34: 7.5% of A is the larger fraction
    assert res.detection[2500] == 5  # G35-G39: 25% of A


async def test_the_default_is_seurat_v5s_one_percent():
    assert DERequest(indices=[0]).min_pct == de_prefilter.DEFAULT_MIN_PCT == 0.01
    assert DECompareRequest(split_by="g", group_a="a", group_b="b").min_pct == 0.01
    res = await _de(_cells())
    assert res.min_pct == 0.01 and res.n_genes_skipped == 0  # 5% is enough
    res = await _de(_cells(rare=()))  # G30-G34 in no cell
    assert res.n_genes_skipped == 5 and "G30" not in {g.gene for g in res.genes}


async def test_zero_is_the_unfiltered_test_and_the_correction_covers_fewer_genes():
    a = _cells()
    full = await _de(a, min_pct=0)
    assert (full.min_pct, full.n_genes_tested, full.n_genes_skipped) == (0.0, 40, 0)
    # Unfiltered = scanpy's own test on .raw, gene for gene.
    ref = a.copy()
    ref.obs["g"] = pd.Categorical(np.where(np.arange(a.n_obs) < 40, "selection", "rest"))
    sc.tl.rank_genes_groups(ref, "g", groups=["selection"], reference="rest",
                            method="wilcoxon", use_raw=True, n_genes=ref.raw.n_vars)
    rgg = ref.uns["rank_genes_groups"]
    want = {str(g): (float(p), float(lf)) for g, p, lf in
            zip(rgg["names"]["selection"], rgg["pvals_adj"]["selection"],
                rgg["logfoldchanges"]["selection"])}
    for g in full.genes:
        assert g.pval_adj == pytest.approx(want[g.gene][0])
        assert g.logfoldchange == pytest.approx(want[g.gene][1], abs=1e-5)
    # Same p-values with the filter on, but BH over 35 tests instead of 40.
    filt = {g.gene: g for g in (await _de(a, min_pct=0.1)).genes}
    for g in full.genes:
        if g.gene in filt:
            assert filt[g.gene].pval == pytest.approx(g.pval)
            assert filt[g.gene].pval_adj <= g.pval_adj + 1e-12


async def test_a_cut_off_nothing_passes_is_a_400():
    X = np.zeros((60, 5), dtype="float32")
    X[::2] = 1.5  # every gene in half the cells of each side
    a = ad.AnnData(X)
    with pytest.raises(HTTPException) as e:
        await _de(a, min_pct=0.9)
    assert e.value.status_code == 400 and "90%" in e.value.detail
    assert (await _de(a, min_pct=0.5)).n_genes_tested == 5


async def test_compare_takes_the_cut_off_and_pseudobulk_ignores_it():
    a = _cells()
    res = await differential_expression_compare(
        "x", DECompareRequest(split_by="grp", group_a="A", group_b="B", min_pct=0.2), _DM(a))
    assert (res.min_pct, res.n_genes_tested, res.n_genes_skipped) == (0.2, 35, 5)
    assert {"G30", "G35"} & {g.gene for g in res.genes} == {"G35"}

    # Pseudobulk filters on summed counts (core/pseudobulk.py), not on detection.
    rng = np.random.default_rng(2)
    pb = ad.AnnData(sp.csr_matrix(rng.poisson(4.0, (80, 30)).astype("float32")))
    pb.var_names = [f"G{i}" for i in range(30)]
    pb.obs["grp"] = pd.Categorical(["A", "B"] * 40)
    pb.obs["donor"] = pd.Categorical([f"d{i % 4}" for i in range(80)])
    res = await differential_expression_compare(
        "x", DECompareRequest(split_by="grp", group_a="A", group_b="B", method="pseudobulk",
                              replicate_key="donor", min_pct=0.5), _DM(pb))
    assert res.method == "pseudobulk"
    assert res.min_pct is None and res.n_genes_skipped is None and res.detection is None
    assert de_results.recall("x", res.result_id).min_pct is None


async def test_the_stored_result_carries_the_cut_off():
    res = await _de(_cells(), min_pct=0.1)
    stored = de_results.recall("x", res.result_id)
    assert stored.min_pct == 0.1 and stored.n_skipped == 5
    assert len(stored.genes) == 35
