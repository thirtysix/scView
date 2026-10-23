"""Markers from a gene store (core/sparse_markers.py) against scanpy's own output.

Every expected value is what ``sc.tl.rank_genes_groups`` (scanpy 1.11) stores, run
in the test on the same data, never this module's output. Data: cuts of Kang 2018
(sparse; 520 cells, 13 cell types, 176 genes of which 6 never detected) and paul15
(369 cells, 19 clusters; densified, so scanpy takes its dense path), raw counts in
tests/fixtures/sparse_markers. Log counts give many tied values.

Wilcoxon: names (order included), scores, pvals, pvals_adj, pts and pts_rest must
be bit for bit scanpy's; the rank sums are exact, so this holds by construction.
logfoldchanges and the t-test go through float64 means, compared to float precision
with the gene order allowed to differ only inside runs of equal (float32) scores.
Small ``max_entries`` split the genes into many blocks with a ragged last one.
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

from scview.core import gene_store, marker_specificity
from scview.core import sparse_markers as sm

FIXTURES = Path(__file__).parent / "fixtures" / "sparse_markers"
FIELDS_EXACT = ("names", "scores", "pvals", "pvals_adj")


def _kang(kind: str = "lognorm") -> ad.AnnData:
    a = ad.read_h5ad(FIXTURES / "kang_counts.h5ad")
    a.X = sp.csr_matrix(a.X, dtype=np.float32)
    if kind == "lognorm":  # what the pipeline leaves in .raw / X
        sc.pp.normalize_total(a, target_sum=1e4)
        sc.pp.log1p(a)
    elif kind == "logcounts":  # log1p of integer counts: heavy ties
        a.X = np.log1p(a.X)
    return a


def _paul15() -> ad.AnnData:
    a = ad.read_h5ad(FIXTURES / "paul15_counts.h5ad")
    a.X = np.asarray(a.X.toarray(), dtype=np.float32)  # dense, as paul15 ships
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    return a


def _store(adata, tmp_path, name="x", source="X", **kw):
    path = gene_store.build_from_adata(adata, tmp_path / f"{name}.genes.h5", source=source,
                                       stage_entries=3000, block_entries=2000, **kw)
    return gene_store.GeneStore(path)


def _both(adata, groupby, store, **kw):
    """(scanpy's uns entry, ours) for the same call."""
    sc.tl.rank_genes_groups(adata, groupby, key_added="ref", **kw)
    sm.rank_genes_groups(adata, groupby, store, key_added="got", max_entries=700, **kw)
    return adata.uns.pop("ref"), adata.uns.pop("got")


def _assert_structure(ref, got):
    assert list(got) == list(ref)  # same keys, same order
    assert got["params"] == ref["params"]
    for field in ("names", "scores", "logfoldchanges", "pvals", "pvals_adj"):
        assert type(got[field]) is type(ref[field])
        assert got[field].dtype == ref[field].dtype  # field names = groups, dtypes
    for field in ("pts", "pts_rest"):
        if field in ref:
            pd.testing.assert_frame_equal(got[field], ref[field], check_exact=True)


def _assert_exact(ref, got, fields=FIELDS_EXACT):
    for field in fields:
        for g in ref[field].dtype.names:
            np.testing.assert_array_equal(got[field][g], ref[field][g], err_msg=f"{field} {g}")


def _by_gene(rec, names, g):
    return pd.Series(rec[g], index=names[g]).astype(np.float64)


def _assert_close(ref, got, field, rtol):
    for g in ref["names"].dtype.names:
        r = _by_gene(ref[field], ref["names"], g)
        q = _by_gene(got[field], got["names"], g).reindex(r.index)
        np.testing.assert_allclose(q.to_numpy(), r.to_numpy(), rtol=rtol, atol=0,
                                   err_msg=f"{field} {g}")


def _assert_order_up_to_ties(ref, got):
    """Same scores by rank; where names differ, only inside a run of equal scores."""
    for g in ref["names"].dtype.names:
        rs, qs = ref["scores"][g], got["scores"][g]
        np.testing.assert_allclose(qs, rs, rtol=1e-6, atol=0)
        edges = np.flatnonzero(np.concatenate([[True], rs[1:] != rs[:-1], [True]]))
        for a, b in zip(edges[:-1], edges[1:]):
            assert set(got["names"][g][a:b]) == set(ref["names"][g][a:b]), (g, a, b)


@pytest.mark.parametrize("kind", ["lognorm", "logcounts"])
def test_wilcoxon_equals_scanpy_on_kang(tmp_path, kind):
    a = _kang(kind)
    with _store(a, tmp_path) as store:
        ref, got = _both(a, "cluster", store, method="wilcoxon", pts=True)
    _assert_structure(ref, got)
    _assert_exact(ref, got)
    _assert_close(ref, got, "logfoldchanges", rtol=1e-6)
    # the fixture has genes no cell expresses: tied scores of exactly 0, kept in order
    assert (ref["scores"]["B"] == 0).sum() >= 6


def test_wilcoxon_equals_scanpy_on_dense_paul15(tmp_path):
    a = _paul15()
    assert isinstance(a.X, np.ndarray)
    with _store(a, tmp_path) as store:
        ref, got = _both(a, "paul15_clusters", store, method="wilcoxon", pts=True)
    _assert_structure(ref, got)
    _assert_exact(ref, got)
    _assert_close(ref, got, "logfoldchanges", rtol=1e-6)


def test_wilcoxon_with_negative_values(tmp_path):
    # a scaled matrix: zeros sit between negative and positive values in the ranks
    a = _paul15()
    sc.pp.scale(a)
    assert (a.X < 0).any()
    with _store(a, tmp_path) as store:
        ref, got = _both(a, "paul15_clusters", store, method="wilcoxon", pts=True)
    _assert_exact(ref, got)


def test_wilcoxon_float64_store(tmp_path):
    # float64 values take the indirect-sort path
    a = _kang("counts")
    a.X = np.log1p(sp.csr_matrix(a.X, dtype=np.float64))
    with _store(a, tmp_path, dtype=np.float64) as store:
        assert store.dtype == np.float64
        ref, got = _both(a, "cluster", store, method="wilcoxon", pts=True)
    _assert_structure(ref, got)
    _assert_exact(ref, got)


@pytest.mark.parametrize("method", ["t-test", "t-test_overestim_var"])
@pytest.mark.parametrize("data", ["kang", "paul15"])
def test_t_test_equals_scanpy(tmp_path, method, data):
    a, col = (_kang(), "cluster") if data == "kang" else (_paul15(), "paul15_clusters")
    with _store(a, tmp_path) as store:
        ref, got = _both(a, col, store, method=method, pts=True)
    _assert_structure(ref, got)
    _assert_order_up_to_ties(ref, got)
    _assert_close(ref, got, "pvals", rtol=1e-9)
    _assert_close(ref, got, "pvals_adj", rtol=1e-9)
    _assert_close(ref, got, "logfoldchanges", rtol=1e-6)


def test_tiny_group_unlabelled_cells_and_zero_genes(tmp_path):
    a = _kang()
    labels = a.obs["cluster"].astype(str).to_numpy().copy()
    labels[[3, 400]] = "tiny"  # a two-cell group
    labels[10:25] = None  # unlabelled: in every group's rest, in no group
    a.obs["groups"] = pd.Categorical(labels)
    with _store(a, tmp_path) as store:
        for method in ("wilcoxon", "t-test"):
            ref, got = _both(a, "groups", store, method=method, pts=True)
            _assert_structure(ref, got)
            if method == "wilcoxon":
                _assert_exact(ref, got)
            else:
                _assert_order_up_to_ties(ref, got)
    # most genes are all zero in the tiny group
    assert (ref["pts"]["tiny"] == 0).sum() > 50


def test_one_cell_group_is_refused_like_scanpy(tmp_path):
    a = _kang()
    labels = a.obs["cluster"].astype(str).to_numpy().copy()
    labels[7] = "single"
    a.obs["groups"] = pd.Categorical(labels)
    with pytest.raises(ValueError, match="single") as theirs:
        sc.tl.rank_genes_groups(a, "groups", method="wilcoxon")
    with _store(a, tmp_path) as store, pytest.raises(ValueError, match="single") as ours:
        sm.rank_genes_groups(a, "groups", store)
    assert str(ours.value) == str(theirs.value)
    # an unused category counts as a group with fewer than two cells too
    a.obs["groups"] = pd.Categorical(a.obs["cluster"], categories=[*a.obs["cluster"].cat.categories, "none"])
    with _store(a, tmp_path, "y") as store, pytest.raises(ValueError, match="none"):
        sm.rank_genes_groups(a, "groups", store)


def test_block_size_does_not_change_the_result(tmp_path):
    a = _kang()
    with _store(a, tmp_path) as store:
        results = []
        for max_entries in (1, 997, 10**9):  # one gene per block, ragged, a single block
            sm.rank_genes_groups(a, "cluster", store, pts=True, max_entries=max_entries,
                                 key_added="m")
            results.append(a.uns.pop("m"))
        assert len(list(store.blocks(997))) > 10
    for other in results[1:]:
        _assert_exact(results[0], other, fields=(*FIELDS_EXACT, "logfoldchanges"))


def test_use_raw_default_and_stale_store(tmp_path):
    # as the pipeline leaves it: .raw log-normalised, X scaled
    a = _kang()
    a.raw = a.copy()
    sc.pp.scale(a)
    with _store(a, tmp_path, "raw", source="raw") as store:
        ref, got = _both(a, "cluster", store, method="wilcoxon", pts=True)
        assert got["params"]["use_raw"] is True
        _assert_structure(ref, got)
        _assert_exact(ref, got)
        with pytest.raises(sm.StaleGeneStore, match="X"):
            sm.rank_genes_groups(a, "cluster", store, use_raw=False)
    with _store(a, tmp_path, "x") as store:
        # a store of X while scanpy's default would rank .raw
        with pytest.raises(sm.StaleGeneStore):
            sm.rank_genes_groups(a, "cluster", store)
        ref, got = _both(a, "cluster", store, method="wilcoxon", use_raw=False)
        _assert_exact(ref, got)
    b = a[a.obs["cluster"] != "B"].copy()  # cells filtered after the store was built
    with _store(a, tmp_path, "raw2", source="raw") as store:
        with pytest.raises(sm.StaleGeneStore, match="cells"):
            sm.rank_genes_groups(b, "cluster", store)


def test_log1p_base_n_genes_rankby_abs_bonferroni(tmp_path):
    a = _kang("counts")
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a, base=2)
    with _store(a, tmp_path) as store:
        ref, got = _both(a, "cluster", store, method="wilcoxon", pts=True)
        _assert_exact(ref, got)
        _assert_close(ref, got, "logfoldchanges", rtol=1e-6)
        ref, got = _both(a, "cluster", store, method="wilcoxon", n_genes=25, rankby_abs=True,
                         corr_method="bonferroni")
        assert len(got["names"]) == 25
        _assert_structure(ref, got)
        _assert_exact(ref, got)


def test_string_column_and_refused_options(tmp_path):
    a = _kang()
    a.obs["donor_str"] = a.obs["Donor"].astype(str).to_numpy()  # scanpy makes it categorical
    b = a.copy()
    with _store(a, tmp_path) as store:
        sm.rank_genes_groups(b, "donor_str", store, key_added="got")  # ours first, on strings
        assert b.obs["donor_str"].dtype.name == "category"
        sc.tl.rank_genes_groups(a, "donor_str", method="wilcoxon", key_added="ref")
        _assert_exact(a.uns["ref"], b.uns["got"])
        with pytest.raises(ValueError, match="Method"):
            sm.rank_genes_groups(a, "cluster", store, method="logreg")
        with pytest.raises(ValueError, match="tie_correct"):
            sm.rank_genes_groups(a, "cluster", store, tie_correct=True)
        with pytest.raises(ValueError, match="layer"):
            a.raw = a.copy()
            sm.rank_genes_groups(a, "cluster", store, layer="counts", use_raw=True)


def test_specificity_matches_marker_specificity(tmp_path):
    a = _kang()
    with _store(a, tmp_path) as store:
        sm.rank_genes_groups(a, "cluster", store, pts=True, specificity=True, key_added="got")
    sc.tl.rank_genes_groups(a, "cluster", method="wilcoxon", pts=True, key_added="ref")
    assert marker_specificity.attach(a, a.uns["ref"], "cluster")
    pd.testing.assert_frame_equal(a.uns["got"][marker_specificity.SPECIFICITY_FIELD],
                                  a.uns["ref"][marker_specificity.SPECIFICITY_FIELD])


def test_int64_store(tmp_path, monkeypatch):
    monkeypatch.setattr(gene_store, "_INDEX_LIMIT", 300)  # 520 cells: int64 cell indices
    a = _kang("logcounts")
    with _store(a, tmp_path) as store:
        ref, got = _both(a, "cluster", store, method="wilcoxon", pts=True)
    _assert_exact(ref, got)


# The full datasets, when present (not in the repo): SCVIEW_TESTDATA=<dir with
# kang2018_ifnb_pbmc.h5ad and paul15.h5ad>.
_DATA = Path(os.environ.get("SCVIEW_TESTDATA", "/nonexistent"))


@pytest.mark.skipif(not (_DATA / "kang2018_ifnb_pbmc.h5ad").exists(), reason="full Kang absent")
def test_full_kang(tmp_path):
    k = ad.read_h5ad(_DATA / "kang2018_ifnb_pbmc.h5ad")
    a = ad.AnnData(sp.csr_matrix(k.raw.X, dtype=np.float32), obs=k.obs.copy(), var=k.raw.var.copy())
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    a.obs["cluster"] = a.obs["cluster"].astype(str).astype("category")
    path = gene_store.build_from_adata(a, tmp_path / "kang.genes.h5")
    with gene_store.GeneStore(path) as store:
        ref, got = _both(a, "cluster", store, method="wilcoxon", pts=True)
        _assert_structure(ref, got)
        _assert_exact(ref, got, fields=(*FIELDS_EXACT, "logfoldchanges"))
        ref, got = _both(a, "cluster", store, method="t-test", pts=True)
        _assert_order_up_to_ties(ref, got)


@pytest.mark.skipif(not (_DATA / "paul15.h5ad").exists(), reason="full paul15 absent")
def test_full_paul15(tmp_path):
    p = ad.read_h5ad(_DATA / "paul15.h5ad")
    a = ad.AnnData(np.asarray(p.X, dtype=np.float32), obs=p.obs.copy(), var=p.var.copy())
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    a.obs["paul15_clusters"] = a.obs["paul15_clusters"].astype(str).astype("category")
    path = gene_store.build_from_adata(a, tmp_path / "paul15.genes.h5")
    with gene_store.GeneStore(path) as store:
        ref, got = _both(a, "paul15_clusters", store, method="wilcoxon", pts=True)
        _assert_structure(ref, got)
        _assert_exact(ref, got, fields=(*FIELDS_EXACT, "logfoldchanges"))
