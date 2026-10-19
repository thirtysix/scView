"""Group markers for annotation and the co-pilot (core/marker_selection.py).

Expected orders are worked out by hand from the stored values below, not taken
from the function: a gene with a high wilcoxon score but a low fold change, a
ribosomal gene leading the score order, and a gene in under 20% of the group.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from scview.core.marker_selection import group_markers, ora_query_genes
from scview.core.uns_keys import markers_key

N = 10  # cells per group

# Fraction of cells expressing each gene, per group; with two groups, each
# group's pts_rest is the other group's fraction.
FRAC = {
    "A": {"RPL13": 1.0, "BROAD": 0.9, "RARE": 0.1, "SPEC": 0.5, "MT-CO1": 1.0, "MID": 0.6,
          "LOWFC": 0.8, "RPLP0": 1.0, "RPSA": 1.0, "Rps3": 1.0, "NANFC": 0.9, "CD79A": 0.0},
    "B": {"RPL13": 1.0, "BROAD": 0.8, "RARE": 0.0, "SPEC": 0.3, "MT-CO1": 1.0, "MID": 0.0,
          "LOWFC": 0.7, "RPLP0": 1.0, "RPSA": 1.0, "Rps3": 1.0, "NANFC": 0.1, "CD79A": 0.7},
}
GENES = list(FRAC["A"])
# Stored (score) order per group: gene, score, logFC.
ROWS = {
    "A": [
        ("RPL13", 20.0, 3.0),    # ribosomal, top score: out
        ("BROAD", 15.0, 0.3),    # second score, smallest kept fold change
        ("RARE", 12.0, 5.0),     # biggest fold change, 1 of 10 cells: out
        ("SPEC", 10.0, 4.0),
        ("MT-CO1", 9.0, 2.0),    # mitochondrial: out
        ("MID", 8.0, 2.0),
        ("LOWFC", 7.0, 0.1),     # fold change under 0.25: out
        ("RPLP0", 6.0, 3.0),     # out
        ("RPSA", 5.0, 3.0),      # out
        ("Rps3", 4.0, 3.0),      # mouse ribosomal: out
        ("NANFC", 3.0, np.nan),  # fold change unknown: out
        ("CD79A", -9.0, -3.0),
    ],
    # Its own order; only CD79A passes (MID: in none of B's cells).
    "B": [("CD79A", 9.0, 3.0), ("MID", 1.0, 0.5)] + [
        (g, -1.0, -1.0) for g in GENES if g not in {"CD79A", "MID"}],
}
A_LOGFC = ["SPEC", "MID", "BROAD"]  # fold change 4, 2, 0.3


def _rec(cols: dict[str, list], dtype) -> np.recarray:
    """scanpy's layout: one field per group, rows by rank."""
    return np.rec.fromarrays([np.asarray(v, dtype=dtype) for v in cols.values()],
                             names=list(cols))


def _result(with_pts: bool = True) -> dict:
    res = {
        "params": {"groupby": "cluster", "method": "wilcoxon"},
        "names": _rec({g: [r[0] for r in rows] for g, rows in ROWS.items()}, object),
        "scores": _rec({g: [r[1] for r in rows] for g, rows in ROWS.items()}, float),
        "logfoldchanges": _rec({g: [r[2] for r in rows] for g, rows in ROWS.items()}, float),
    }
    if with_pts:
        res["pts"] = pd.DataFrame(FRAC)
        res["pts_rest"] = pd.DataFrame({"A": FRAC["B"], "B": FRAC["A"]})
    return res


def _adata(with_pts: bool = True, sparse: bool = False) -> ad.AnnData:
    """Log-like values (not integers), with exactly FRAC of each group expressing."""
    X = np.zeros((2 * N, len(GENES)), dtype=np.float32)
    for gi, group in enumerate(("A", "B")):
        for j, gene in enumerate(GENES):
            X[gi * N: gi * N + int(round(FRAC[group][gene] * N)), j] = 1.5
    a = ad.AnnData(sp.csr_matrix(X) if sparse else X)
    a.var_names = GENES
    a.obs_names = [f"c{i}" for i in range(2 * N)]
    a.obs["cluster"] = pd.Categorical(["A"] * N + ["B"] * N)
    a.uns[markers_key("cluster")] = _result(with_pts)
    return a


def test_logfc_order_drops_ribo_mito_rare_and_weak_genes():
    assert group_markers(_adata(), "cluster") == {"A": A_LOGFC, "B": ["CD79A"]}


def test_score_and_pts_diff_orders():
    a = _adata()
    assert group_markers(a, "cluster", order="score")["A"] == ["BROAD", "SPEC", "MID"]
    # pts - pts_rest: MID 0.6, SPEC 0.2, BROAD 0.1
    assert group_markers(a, "cluster", order="pts_diff")["A"] == ["MID", "SPEC", "BROAD"]
    with pytest.raises(ValueError):
        group_markers(a, "cluster", order="pval")


def test_thresholds_and_n():
    a = _adata()
    assert group_markers(a, "cluster", n=2)["A"] == ["SPEC", "MID"]
    # RARE comes first once 10% of the group is enough.
    assert group_markers(a, "cluster", min_pts=0.1)["A"] == ["RARE", *A_LOGFC]
    assert group_markers(a, "cluster", min_logfc=1.0)["A"] == ["SPEC", "MID"]
    # Without the exclusion the ribosomal/mito genes come back, by fold change
    # (ties keep the score order).
    assert group_markers(a, "cluster", exclude_ribo_mito=False, n=4)["A"] == [
        "SPEC", "RPL13", "RPLP0", "RPSA"]


def test_adjusted_p_filter():
    a = _adata()
    rgg = a.uns[markers_key("cluster")]
    # A's adjusted p by rank: MID (rank 6) and SPEC (rank 4) not significant.
    padj = {"A": [1e-9, 1e-6, 1e-6, 0.2, 1e-3, 0.06] + [1e-3] * 6, "B": [1e-4] * 12}
    rgg["pvals_adj"] = _rec(padj, float)
    assert group_markers(a, "cluster")["A"] == ["BROAD"]  # 0.05 by default
    assert ora_query_genes(a, "cluster", "A", 25) == ["BROAD"]
    assert group_markers(a, "cluster", max_padj=None)["A"] == A_LOGFC
    assert group_markers(a, "cluster", max_padj=0.1)["A"] == ["MID", "BROAD"]
    # A result without p-values (logreg) is not filtered by them.
    rgg["pvals_adj"] = _rec({g: [np.nan] * 12 for g in padj}, float)
    assert group_markers(a, "cluster", max_padj=0.05)["A"] == A_LOGFC
    del rgg["pvals_adj"]
    assert group_markers(a, "cluster", max_padj=0.05)["A"] == A_LOGFC


@pytest.mark.parametrize("sparse", [False, True])
def test_fraction_expressing_counted_when_pts_are_not_stored(sparse):
    a = _adata(with_pts=False, sparse=sparse)
    assert group_markers(a, "cluster") == {"A": A_LOGFC, "B": ["CD79A"]}
    # pts_rest counted as well: MID 0.6, SPEC 0.2, BROAD 0.1.
    assert group_markers(a, "cluster", order="pts_diff")["A"] == ["MID", "SPEC", "BROAD"]


@pytest.mark.parametrize("sparse", [False, True])
def test_fraction_read_from_log_normalised_raw_not_scaled_x(sparse):
    a = _adata(with_pts=False, sparse=sparse)
    a.raw = a.copy()
    a.X = np.full(a.shape, -0.5, dtype=np.float32)  # z-scores: every cell "expresses"
    assert group_markers(a, "cluster")["A"] == A_LOGFC


def test_reads_the_columns_own_key_never_another_columns_markers():
    a = _adata()
    del a.uns[markers_key("cluster")]
    a.obs["leiden"] = a.obs["cluster"]
    a.uns["rank_genes_groups"] = {**_result(), "params": {"groupby": "leiden"}}
    assert group_markers(a, "cluster") == {}  # the bare key is leiden's
    a.uns["rank_genes_groups"]["params"] = {"groupby": "cluster"}
    assert group_markers(a, "cluster")["A"] == A_LOGFC  # now it is this column's
    a.uns["other"] = _result()
    assert group_markers(a, "nothing", key="other")["B"] == ["CD79A"]  # explicit key


def test_labels_survive_an_h5ad_round_trip(tmp_path):
    p = tmp_path / "d.h5ad"
    _adata().write_h5ad(p)
    assert group_markers(ad.read_h5ad(p), "cluster") == {"A": A_LOGFC, "B": ["CD79A"]}


def test_with_scanpys_own_result_a_specific_gene_beats_a_broad_one():
    """Real rank_genes_groups: BROAD is up a little in every A cell (the higher
    score), SPEC strongly up in 40% of A cells (the bigger fold change)."""
    import scanpy as sc

    rng = np.random.default_rng(0)
    n = 50
    genes = ["BROAD", "SPEC", "RPL3", "FLAT1", "FLAT2"]
    X = rng.poisson(1.0, (2 * n, len(genes))).astype(np.float32)
    X[:n, 0] += 2  # every A cell, a little
    X[: int(0.4 * n), 1] += 30  # 40% of A cells, a lot
    X[:n, 2] += 3  # ribosomal, every A cell
    a = ad.AnnData(np.log1p(X))
    a.var_names = genes
    a.obs["cluster"] = pd.Categorical(["A"] * n + ["B"] * n)
    sc.tl.rank_genes_groups(a, "cluster", method="wilcoxon", pts=True,
                            key_added=markers_key("cluster"))
    score_order = [str(g) for g in a.uns[markers_key("cluster")]["names"]["A"]]
    assert score_order.index("BROAD") < score_order.index("SPEC")
    assert group_markers(a, "cluster")["A"][:2] == ["SPEC", "BROAD"]


def test_ora_query_is_the_fold_change_choice_not_the_top_of_the_score_order():
    a = _adata()
    # Score order with only ribo/mito dropped (the old query) would start BROAD, RARE.
    assert ora_query_genes(a, "cluster", "A", 2) == ["SPEC", "MID"]
    assert ora_query_genes(a, "cluster", "A", 100) == A_LOGFC  # fewer than n pass
    assert ora_query_genes(a, "cluster", "nope", 5) == []


def test_ora_query_reads_the_groupby_of_a_bare_result():
    a = _adata()
    a.uns["rank_genes_groups"] = a.uns.pop(markers_key("cluster"))
    assert ora_query_genes(a, "", "A", 3) == A_LOGFC

