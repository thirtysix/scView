"""The one marker / DE call (core/rank_genes.py) against scanpy's own, run in the test.

Every expected value is what ``sc.tl.rank_genes_groups`` (scanpy 1.11) stores for the
same call on the same data. On the blocks path names (order included), scores, p-values,
adjusted p-values and pts are bit for bit scanpy's; on float32 log values the fold
changes and the t-test are too (their float64 sums are exact). While the blocks path
runs, scanpy's function is replaced by one that fails, so a test cannot pass by quietly
falling back. Data: cuts of Kang 2018 (sparse) and paul15 (dense), raw counts in
tests/fixtures/sparse_markers; small ``max_entries`` split the genes into many blocks.
"""

from __future__ import annotations

import os
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
import scipy.sparse as sp

from scview.core import marker_specificity, matrix_genes, rank_genes
from tests.test_matrix_genes import _reorder_rows
from tests.test_sparse_markers import (
    FIELDS_EXACT,
    _assert_close,
    _assert_exact,
    _assert_order_up_to_ties,
    _assert_structure,
    _kang,
    _paul15,
)

ALL_FIELDS = (*FIELDS_EXACT, "logfoldchanges")


@pytest.fixture()
def no_scanpy(monkeypatch):
    """Inside the block: sc.tl.rank_genes_groups raises (the blocks path must not use it)."""
    real = sc.tl.rank_genes_groups

    class Guard:
        def __enter__(self):
            def fail(*a, **k):
                raise AssertionError("scanpy's rank_genes_groups was called")
            monkeypatch.setattr(sc.tl, "rank_genes_groups", fail)

        def __exit__(self, *exc):
            monkeypatch.setattr(sc.tl, "rank_genes_groups", real)

    return Guard()


def _both(a, groupby, no_scanpy, **kw):
    """(scanpy's uns entry, ours) for the same call; ours must take the blocks path."""
    sc.tl.rank_genes_groups(a, groupby, key_added="ref", **kw)
    with no_scanpy:
        path = rank_genes.rank_genes_groups(a, groupby, key_added="got", max_entries=700, **kw)
    assert path == rank_genes.SPARSE
    return a.uns.pop("ref"), a.uns.pop("got")


def _data(name):
    return (_kang(), "cluster") if name == "kang" else (_paul15(), "paul15_clusters")


@pytest.mark.parametrize("data", ["kang", "paul15"])
@pytest.mark.parametrize("method", ["wilcoxon", "t-test", "t-test_overestim_var"])
def test_each_group_against_the_rest(no_scanpy, data, method):
    a, col = _data(data)
    ref, got = _both(a, col, no_scanpy, method=method, pts=True)
    _assert_structure(ref, got)
    _assert_exact(ref, got, ALL_FIELDS)


@pytest.mark.parametrize("data", ["kang", "paul15"])
@pytest.mark.parametrize("method", ["wilcoxon", "t-test", "t-test_overestim_var"])
def test_reference_group_equals_scanpys_reference_mode(no_scanpy, data, method):
    a, col = _data(data)
    cats = list(a.obs[col].cat.categories)
    # a few groups against one (ranked within each pair's cells), the rest left out
    ref, got = _both(a, col, no_scanpy, method=method, pts=True, groups=cats[2:5],
                     reference=cats[0])
    assert got["params"]["reference"] == cats[0]
    assert "pts_rest" not in got and list(got["pts"].columns) == [*cats[2:5], cats[0]]
    _assert_structure(ref, got)
    _assert_exact(ref, got, ALL_FIELDS)
    # every group against one, and one pair (merge check, DE between groups)
    for kw in ({"reference": cats[1]}, {"groups": [cats[3]], "reference": cats[1]}):
        ref, got = _both(a, col, no_scanpy, method=method, pts=True, **kw)
        _assert_structure(ref, got)
        _assert_exact(ref, got, ALL_FIELDS)


def test_reference_is_not_the_rest_of_a_pair_inside_more_cells(no_scanpy):
    # The reference test must leave out the other cells: on this data it differs from
    # the same pair against everything else, so a version ranking all cells fails here.
    a, col = _data("kang")
    cats = list(a.obs[col].cat.categories)
    ref, got = _both(a, col, no_scanpy, method="wilcoxon", groups=[cats[3]], reference=cats[1])
    sc.tl.rank_genes_groups(a, col, method="wilcoxon", groups=[cats[3]], key_added="rest")
    assert not np.array_equal(a.uns["rest"]["scores"][cats[3]], ref["scores"][cats[3]])
    _assert_exact(ref, got, ALL_FIELDS)


@pytest.mark.parametrize("method", ["wilcoxon", "t-test"])
def test_subset_of_groups_against_all_the_rest(no_scanpy, method):
    a, col = _data("kang")
    cats = list(a.obs[col].cat.categories)
    ref, got = _both(a, col, no_scanpy, method=method, pts=True, groups=[cats[4], cats[1]])
    assert list(got["pts"].columns) == [cats[4], cats[1]]  # scanpy's order: as given
    _assert_structure(ref, got)
    _assert_exact(ref, got, ALL_FIELDS)


@pytest.mark.parametrize("layout", ["csc", "descending", "unordered", "dense"])
def test_every_matrix_layout(no_scanpy, layout):
    a, col = _data("kang")
    X = a.X
    a.X = {"csc": lambda: sp.csc_matrix(X), "descending": lambda: _reorder_rows(X, "descending"),
           "unordered": lambda: _reorder_rows(X, "unordered"), "dense": lambda: X.toarray()}[layout]()
    assert matrix_genes.MatrixGenes(a.X, a.var_names).layout == layout
    ref, got = _both(a, col, no_scanpy, method="wilcoxon", pts=True)
    _assert_structure(ref, got)
    _assert_exact(ref, got, ALL_FIELDS)
    ref, got = _both(a, col, no_scanpy, method="t-test", pts=True)
    _assert_structure(ref, got)
    if layout != "csc":
        _assert_exact(ref, got, ALL_FIELDS)
    else:  # scanpy squares a CSC matrix's values in float64: those sums are not exact
        _assert_order_up_to_ties(ref, got)
        _assert_close(ref, got, "pvals", rtol=1e-9)
        _assert_exact(ref, got, ("logfoldchanges",))


def test_float64_values(no_scanpy):
    a, col = _data("kang")
    a.X = a.X.astype(np.float64)
    a.X.data += np.linspace(0, 1e-3, a.X.nnz)  # not float32 values: sums are not exact
    ref, got = _both(a, col, no_scanpy, method="wilcoxon", pts=True)
    _assert_structure(ref, got)
    _assert_exact(ref, got)  # ranks: exact in any dtype
    _assert_close(ref, got, "logfoldchanges", rtol=1e-6)
    ref, got = _both(a, col, no_scanpy, method="t-test", pts=True)
    _assert_order_up_to_ties(ref, got)
    _assert_close(ref, got, "pvals", rtol=1e-9)
    _assert_close(ref, got, "logfoldchanges", rtol=1e-6)


def test_raw_layer_and_string_column(no_scanpy):
    a, col = _data("kang")
    a.layers["log"] = a.X.copy()
    a.raw = a.copy()
    sc.pp.scale(a)  # X scaled (dense, negative values); markers rank .raw by default
    a.obs["donor"] = a.obs["Donor"].astype(str).to_numpy()  # becomes categorical
    for kw in ({}, {"use_raw": False, "layer": "log"}):
        ref, got = _both(a, "donor", no_scanpy, method="wilcoxon", pts=True, **kw)
        assert got["params"]["use_raw"] is ref["params"]["use_raw"]
        _assert_structure(ref, got)
        _assert_exact(ref, got, ALL_FIELDS)
    assert a.obs["donor"].dtype.name == "category"
    # the scaled X: ranks exact; its means of mixed-sign values are not exact sums
    ref, got = _both(a, "donor", no_scanpy, method="wilcoxon", pts=True, use_raw=False)
    _assert_structure(ref, got)
    _assert_exact(ref, got)
    _assert_close(ref, got, "logfoldchanges", rtol=1e-5)


def test_two_cell_group_unlabelled_cells_and_one_cell_error(no_scanpy):
    a, col = _data("kang")
    labels = a.obs[col].astype(str).to_numpy().copy()
    labels[[3, 400]] = "tiny"
    labels[10:25] = None  # in every group's rest, in no group (left out of a reference test)
    a.obs["g"] = pd.Categorical(labels)
    for kw in ({"method": "wilcoxon", "pts": True}, {"method": "t-test", "pts": True},
               {"method": "wilcoxon", "groups": ["tiny"], "reference": "B", "pts": True}):
        ref, got = _both(a, "g", no_scanpy, **kw)
        _assert_structure(ref, got)
        _assert_exact(ref, got, ALL_FIELDS)
    labels[7] = "single"
    a.obs["g"] = pd.Categorical(labels)
    with pytest.raises(ValueError) as theirs:
        sc.tl.rank_genes_groups(a, "g", method="wilcoxon")
    with no_scanpy, pytest.raises(ValueError) as ours:
        rank_genes.rank_genes_groups(a, "g", method="wilcoxon")
    assert str(ours.value) == str(theirs.value)
    # a one-cell group outside the compared ones is no error, in scanpy or here
    ref, got = _both(a, "g", no_scanpy, method="wilcoxon", groups=["tiny", "B"], reference="B")
    _assert_exact(ref, got, ALL_FIELDS)


def test_n_genes_rankby_abs_bonferroni_log_base(no_scanpy):
    a = _kang("counts")
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a, base=2)
    ref, got = _both(a, "cluster", no_scanpy, method="wilcoxon", n_genes=25, rankby_abs=True,
                     corr_method="bonferroni", pts=True)
    assert len(got["names"]) == 25
    _assert_structure(ref, got)
    _assert_exact(ref, got, ALL_FIELDS)


def test_specificity_as_marker_specificity_attaches_it(no_scanpy):
    a, col = _data("kang")
    with no_scanpy:
        rank_genes.rank_genes_groups(a, col, method="wilcoxon", pts=True, specificity=True,
                                     key_added="got")
    sc.tl.rank_genes_groups(a, col, method="wilcoxon", pts=True, key_added="ref")
    assert marker_specificity.attach(a, a.uns["ref"], col)
    pd.testing.assert_frame_equal(a.uns["got"][marker_specificity.SPECIFICITY_FIELD],
                                  a.uns["ref"][marker_specificity.SPECIFICITY_FIELD])


def test_scanpy_runs_what_the_blocks_path_does_not(monkeypatch):
    a, col = _data("kang")
    a.layers["counts"] = sp.csr_matrix(a.X.toarray().round().astype(np.int64))
    dup = a.copy()
    p = dup.X.indptr[2]
    dup.X.indices[p + 1] = dup.X.indices[p]  # a stored duplicate
    dup.X.has_sorted_indices = False
    calls = []
    real = sc.tl.rank_genes_groups

    def spy(*args, **kw):
        calls.append(kw.get("method"))
        return real(*args, **kw)

    monkeypatch.setattr(sc.tl, "rank_genes_groups", spy)
    cases = [
        (a, {"method": "logreg", "max_iter": 50}),
        (a, {"method": "wilcoxon", "tie_correct": True}),
        (a, {"method": "wilcoxon", "layer": "counts", "use_raw": False}),  # integer values
        (dup, {"method": "wilcoxon"}),  # found while reading: nothing written, then scanpy
    ]
    for adata, kw in cases:
        assert rank_genes.rank_genes_groups(adata, col, **kw) == rank_genes.SCANPY
        assert "names" in adata.uns["rank_genes_groups"]
    assert len(calls) == len(cases)
    # scanpy's own errors for arguments it refuses
    with pytest.raises(ValueError, match="reference = nope"):
        rank_genes.rank_genes_groups(a, col, method="wilcoxon", reference="nope")
    with pytest.raises(IndexError):
        rank_genes.rank_genes_groups(a, col, method="wilcoxon", groups=["nope"])


# The full datasets, when present (not in the repo): SCVIEW_TESTDATA=<dir with
# kang2018_ifnb_pbmc.h5ad and paul15.h5ad>.
_DATA = Path(os.environ.get("SCVIEW_TESTDATA", "/nonexistent"))


def _full(name):
    if name == "kang":
        k = ad.read_h5ad(_DATA / "kang2018_ifnb_pbmc.h5ad")
        a = ad.AnnData(sp.csr_matrix(k.raw.X, dtype=np.float32), obs=k.obs.copy(),
                       var=k.raw.var.copy())
        col = "cluster"
    else:
        p = ad.read_h5ad(_DATA / "paul15.h5ad")
        a = ad.AnnData(np.asarray(p.X, dtype=np.float32), obs=p.obs.copy(), var=p.var.copy())
        col = "paul15_clusters"
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    a.obs[col] = a.obs[col].astype(str).astype("category")
    return a, col


@pytest.mark.parametrize("name", ["kang", "paul15"])
def test_full_datasets(no_scanpy, name):
    file = "kang2018_ifnb_pbmc.h5ad" if name == "kang" else "paul15.h5ad"
    if not (_DATA / file).exists():
        pytest.skip(f"full {name} absent")
    a, col = _full(name)
    cats = list(a.obs[col].cat.categories)
    for kw in ({"method": "wilcoxon", "pts": True}, {"method": "t-test", "pts": True},
               {"method": "wilcoxon", "groups": [cats[0]], "reference": cats[2]}):
        sc.tl.rank_genes_groups(a, col, key_added="ref", **kw)
        with no_scanpy:
            assert rank_genes.rank_genes_groups(a, col, key_added="got", **kw) == rank_genes.SPARSE
        ref, got = a.uns.pop("ref"), a.uns.pop("got")
        _assert_structure(ref, got)
        _assert_exact(ref, got, ALL_FIELDS)


@pytest.mark.parametrize("where", ["raw", "X"])
def test_float16_matrix_is_ranked_as_float32(where):
    """A half-precision matrix (a published h5ad) is widened in place, then ranked like float32."""
    rng = np.random.default_rng(0)
    vals = np.log1p(rng.poisson(1.0, size=(60, 12)).astype(np.float32))
    vals[:30, 0] += 2.0
    obs = pd.DataFrame({"g": pd.Categorical(["a"] * 30 + ["b"] * 30)}, index=[f"c{i}" for i in range(60)])
    var = pd.DataFrame(index=[f"G{i}" for i in range(12)])
    half = vals.astype(np.float16)  # dense: scipy.sparse has no float16
    a = ad.AnnData(X=half if where == "X" else sp.csr_matrix(vals), obs=obs, var=var)
    if where == "raw":
        a.raw = ad.AnnData(X=half, obs=obs[[]], var=var)
    ran = rank_genes.rank_genes_groups(a, "g", method="wilcoxon", use_raw=where == "raw")
    m = a.raw.X if where == "raw" else a.X
    assert m.dtype == np.float32 and ran == rank_genes.SPARSE
    assert a.uns["rank_genes_groups"]["names"]["a"][0] == "G0"
