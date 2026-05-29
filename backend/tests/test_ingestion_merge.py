"""Tests for multi-sample merge (§3a)."""

from __future__ import annotations

import anndata as ad
import numpy as np

from scview.core.ingestion import (
    MergeJoin,
    VarBasis,
    classify_var_basis,
    merge_units,
    plan_merge,
)


def _ens(i: int) -> str:
    return f"ENSG{i:011d}"


def _adata(var_names, n_cells=2, gene_ids=None, x=None):
    n_var = len(var_names)
    X = x if x is not None else np.arange(1, n_cells * n_var + 1, dtype="float32").reshape(
        n_cells, n_var
    )
    a = ad.AnnData(X)
    a.obs_names = [f"AAACCTGA{i:08d}-1" for i in range(n_cells)]
    a.var_names = list(var_names)
    if gene_ids is not None:
        a.var["gene_ids"] = list(gene_ids)
    return a


# ---------------------------------------------------------------------------
# Basis classification
# ---------------------------------------------------------------------------


def test_classify_basis():
    assert classify_var_basis([_ens(i) for i in range(5)]) == VarBasis.ensembl
    assert classify_var_basis(["CD3D", "CD8A", "IL7R"]) == VarBasis.symbol
    assert classify_var_basis([]) == VarBasis.unknown


# ---------------------------------------------------------------------------
# Plain merges
# ---------------------------------------------------------------------------


def test_merge_full_overlap_inner(tmp_path):
    a = _adata(["G1", "G2", "G3"])
    b = _adata(["G1", "G2", "G3"])
    plan = plan_merge({"sA": a, "sB": b})
    assert plan.suspicious_low_overlap is False
    assert plan.intersection == 3
    merged = merge_units({"sA": a, "sB": b}, reconciliation=plan.reconciliation)
    assert merged.shape == (4, 3)
    assert list(merged.obs["sample"]) == ["sA", "sA", "sB", "sB"]
    assert len(set(merged.obs_names)) == 4  # barcodes made unique by sample suffix


def test_subset_same_basis_not_flagged(tmp_path):
    # 3-gene sample is a subset of a 5-gene sample, same basis → legitimate.
    a = _adata(["G0", "G1", "G2"])
    b = _adata(["G0", "G1", "G2", "G3", "G4"])
    plan = plan_merge({"small": a, "big": b})
    assert plan.suspicious_low_overlap is False  # 3 == 100% of the smaller set
    assert plan.reconciliation is None
    merged = merge_units({"small": a, "big": b}, join=MergeJoin.inner)
    assert merged.shape == (4, 3)


def test_union_join_fills_zeros(tmp_path):
    a = _adata(["G1", "G2", "G3"])
    b = _adata(["G2", "G3", "G4"])
    merged = merge_units({"sA": a, "sB": b}, join=MergeJoin.outer)
    assert merged.shape == (4, 4)  # union of {G1..G4}
    # G4 is absent from sA, so sA cells are zero there
    g4_col = list(merged.var_names).index("G4")
    assert merged.X[0, g4_col] == 0


# ---------------------------------------------------------------------------
# Identifier reconciliation (the Ensembl <-> symbol trap)
# ---------------------------------------------------------------------------


def test_reconciliation_feasible(tmp_path):
    a = _adata([_ens(i) for i in range(3)])  # Ensembl-indexed
    b = _adata(["GENE0", "GENE1", "GENE2"], gene_ids=[_ens(i) for i in range(3)])  # symbols + ids
    plan = plan_merge({"sA": a, "sB": b})
    assert plan.suspicious_low_overlap is True
    assert plan.reconciliation is not None
    recon = plan.reconciliation
    assert recon.needed and recon.feasible
    assert recon.target_basis == VarBasis.ensembl
    assert recon.overlap_before == 0
    assert recon.overlap_after == 3
    assert [r.sample for r in recon.resets] == ["sB"]

    merged = merge_units({"sA": a, "sB": b}, reconciliation=recon)
    assert merged.shape == (4, 3)  # reconciled onto Ensembl ids
    assert list(merged.var_names) == [_ens(i) for i in range(3)]


def test_reconciliation_infeasible(tmp_path):
    a = _adata([_ens(i) for i in range(3)])  # Ensembl
    b = _adata(["GENE0", "GENE1", "GENE2"])  # symbols, NO gene_ids column
    plan = plan_merge({"sA": a, "sB": b})
    assert plan.suspicious_low_overlap is True
    assert plan.reconciliation.needed is True
    assert plan.reconciliation.feasible is False
    assert "union" in plan.reconciliation.message.lower()


def test_low_overlap_same_basis_is_informational(tmp_path):
    # Same basis (symbols) but disjoint gene sets → informational, not reconcilable.
    a = _adata(["A1", "A2", "A3", "A4"])
    b = _adata(["B1", "B2", "B3", "B4"])
    plan = plan_merge({"sA": a, "sB": b})
    assert plan.suspicious_low_overlap is True
    assert plan.reconciliation.needed is False
    assert plan.reconciliation.feasible is False
    assert "pre-filtered" in plan.reconciliation.message.lower()


# ---------------------------------------------------------------------------
# Warnings
# ---------------------------------------------------------------------------


def test_large_merge_warns(tmp_path):
    a = ad.AnnData(np.zeros((200_001, 1), dtype="float32"))
    a.var_names = ["G1"]
    b = _adata(["G1"], n_cells=2)
    plan = plan_merge({"big": a, "sB": b})
    assert any("sluggish" in w.lower() for w in plan.warnings)


def test_raw_vs_normalized_warns(tmp_path):
    counts = _adata(["G1", "G2", "G3"], x=np.array([[1, 2, 3], [4, 5, 6]], dtype="float32"))
    norm = _adata(["G1", "G2", "G3"], x=np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], "float32"))
    plan = plan_merge({"raw": counts, "norm": norm})
    assert any("normalized" in w.lower() for w in plan.warnings)
