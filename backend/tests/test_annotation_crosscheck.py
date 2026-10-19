"""Annotation cross-check (core/annotation_crosscheck.py): labellings small enough
that every ARI, mapping and agreement below is worked out by hand."""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from scview.core import annotation_crosscheck as xc


def _adata(**cols) -> ad.AnnData:
    n = len(next(iter(cols.values())))
    a = ad.AnnData(X=np.zeros((n, 2), dtype=np.float32))
    a.obs_names = [f"c{i}" for i in range(n)]
    for k, v in cols.items():
        a.obs[k] = pd.Categorical(v) if not isinstance(v, pd.Categorical) else v
    return a


REF = ["A"] * 4 + ["B"] * 4 + ["C"] * 2


def test_identical_columns_agree_on_everything():
    a = _adata(ref=REF, same=REF)
    out = xc.crosscheck(a, ["ref", "same"], "ref")
    assert out["pairwise"]["ari"]["ref"]["same"] == 1.0
    assert out["pairwise"]["ami"]["same"]["ref"] == 1.0
    assert out["mean_agreement"] == 1.0 and out["exact_name_agreement"] == 1.0
    assert out["agreement_counts"] == {"1": 10}
    assert out["consensus_counts"] == {"A": 4, "B": 4, "C": 2} and out["n_unresolved"] == 0
    cmp = out["comparisons"]["same"]
    assert {k: v["status"] for k, v in cmp["best_match"].items()} == {"A": "stable", "B": "stable", "C": "stable"}
    assert cmp["best_match"]["C"]["jaccard"] == 1.0 and cmp["best_match"]["C"]["match"] == "C"
    assert cmp["exact_name"]["n_shared_labels"] == 3


def test_renamed_labels_agree_by_overlap_but_not_by_name():
    names = {"A": "Classical monocytes", "B": "Naive B", "C": "pDC"}
    a = _adata(ref=REF, other=[names[x] for x in REF])
    out = xc.crosscheck(a, ["ref", "other"], "ref")
    cmp = out["comparisons"]["other"]
    assert cmp["ari"] == 1.0 and out["pairwise"]["ari"]["other"]["ref"] == 1.0
    assert cmp["mapping"] == {v: {"to": k, "share": 1.0} for k, v in names.items()}
    assert out["mean_agreement"] == 1.0 and cmp["mapped_agreement"] == 1.0
    # No name in common: exact-name agreement is zero, however well they agree.
    assert out["exact_name_agreement"] == 0.0
    assert cmp["exact_name"] == {"n_shared_labels": 0, "shared_labels": [], "agreement": 0.0}
    assert out["labels"]["A"]["annotators"]["other"]["calls"] == [
        {"label": "Classical monocytes", "share": 1.0, "maps_to": "A"}]


def test_one_split_cluster():
    # Reference A (4 cells) and B (4); the other column splits A in two halves.
    a = _adata(ref=["A"] * 4 + ["B"] * 4, split=["A1", "A1", "A2", "A2"] + ["B"] * 4)
    out = xc.crosscheck(a, ["ref", "split"], "ref")
    cmp = out["comparisons"]["split"]
    # Pairs together in both: 1 + 1 + 6 = 8; rows 6 + 6 = 12, columns 1 + 1 + 6 = 8,
    # all pairs 28: ARI = (8 - 12*8/28) / ((12 + 8)/2 - 12*8/28) = 16/23.
    assert cmp["ari"] == pytest.approx(16 / 23, abs=1e-4)
    # Both halves map to A, so every cell agrees once mapped...
    assert cmp["mapping"]["A1"]["to"] == "A" and cmp["mapping"]["A2"]["to"] == "A"
    assert out["mean_agreement"] == 1.0
    # ...but only B's 4 of 8 cells carry the same name.
    assert cmp["exact_name"]["agreement"] == 0.5 and cmp["exact_name"]["shared_labels"] == ["B"]
    # A's best match holds 2 of its 4 cells (Jaccard 2/4) and is pure: it splits.
    assert cmp["best_match"]["A"]["jaccard"] == 0.5 and cmp["best_match"]["A"]["status"] == "splits"
    assert cmp["best_match"]["B"]["status"] == "stable"
    assert cmp["contingency"]["counts"]["A"] == {"A1": 2, "A2": 2}
    assert cmp["contingency"]["row_fraction"]["A"] == {"A1": 0.5, "A2": 0.5}
    assert "B" not in cmp["contingency"]["counts"]["A"]  # zeros are left out


def _three_way() -> ad.AnnData:
    """30 cells, 10 per reference label. Cell 0 (A): o1 says y (-> B), o2 says r (-> C):
    a three-way tie. Cell 1 (A): both say B's label: the consensus overrules A."""
    ref = ["A"] * 10 + ["B"] * 10 + ["C"] * 10
    o1 = ["x"] * 10 + ["y"] * 10 + ["z"] * 10
    o2 = ["p"] * 10 + ["q"] * 10 + ["r"] * 10
    o1[0], o2[0] = "y", "r"
    o1[1], o2[1] = "y", "q"
    return _adata(ref=ref, o1=o1, o2=o2)


def test_three_way_tie_is_unresolved_and_two_votes_overrule_the_reference():
    a = _three_way()
    out = xc.crosscheck(a, ["ref", "o1", "o2"], "ref")
    # y has 12 cells, 10 in B; r 11, 10 in C; q 11, 10 in B: majority mapping holds.
    assert out["comparisons"]["o1"]["mapping"]["y"] == {"to": "B", "share": round(10 / 12, 4)}
    assert out["comparisons"]["o2"]["mapping"]["r"]["to"] == "C"
    # Cells 0 and 1 agree with nobody; the other 28 fully.
    assert out["mean_agreement"] == round(28 / 30, 4)
    assert out["agreement_counts"] == {"0": 2, "1": 28}
    assert out["n_unresolved"] == 1
    assert out["consensus_counts"] == {"A": 8, "B": 11, "C": 10, "Unresolved": 1}
    lab = out["labels"]["A"]
    assert lab["n_cells"] == 10 and lab["mean_agreement"] == 0.8
    assert lab["annotators"]["o1"]["agreement"] == 0.8 and lab["disagreeing"] == []
    assert lab["annotators"]["o1"]["calls"][:2] == [
        {"label": "x", "share": 0.8, "maps_to": "A"}, {"label": "y", "share": 0.2, "maps_to": "B"}]
    assert out["labels"]["B"]["mean_agreement"] == 1.0
    # Every pair is in the matrix, both ways, keyed by name.
    assert set(out["pairwise"]["ari"]) == {"ref", "o1", "o2"}
    assert out["pairwise"]["ari"]["o1"]["o2"] == out["pairwise"]["ari"]["o2"]["o1"]


def test_two_columns_that_disagree_tie():
    # One cell where the other column's label maps elsewhere: 1 vote each.
    a = _adata(ref=["A"] * 3 + ["B"] * 3, other=["a", "a", "b", "b", "b", "b"])
    out = xc.crosscheck(a, ["ref", "other"])
    assert out["reference"] == "ref"  # the first column by default
    assert out["n_unresolved"] == 1 and out["consensus_counts"]["Unresolved"] == 1
    lab = out["labels"]["A"]
    assert lab["mean_agreement"] == round(2 / 3, 4)


def test_a_coarser_column_disagrees_on_the_smaller_part():
    # "T" covers CD4 (6) and CD8 (2): it maps to CD4, so CD8 cells disagree.
    a = _adata(ref=["CD4"] * 6 + ["CD8"] * 2 + ["B"] * 2, other=["T"] * 8 + ["B"] * 2)
    out = xc.crosscheck(a, ["ref", "other"], "ref")
    assert out["comparisons"]["other"]["mapping"]["T"] == {"to": "CD4", "share": 0.75}
    assert out["labels"]["CD8"]["mean_agreement"] == 0.0
    assert out["labels"]["CD8"]["disagreeing"] == ["other"]
    assert out["comparisons"]["other"]["best_match"]["CD8"]["status"] == "merges"


def test_missing_labels_are_left_out():
    ref = pd.Categorical(["A", "A", None, "B", "B", "B"])
    other = pd.Categorical(["a", None, "a", "b", "b", "b"])
    out = xc.crosscheck(_adata(ref=ref, other=other), ["ref", "other"], "ref")
    # Cell 1 has no other label, cell 2 no reference: 4 cells compared, all agree.
    assert out["n_compared"] == 4 and out["mean_agreement"] == 1.0
    assert out["comparisons"]["other"]["n_cells"] == 4


def test_label_names_normalise_case_spaces_plus_and_plural():
    n = xc.normalise_label
    assert n("CD4+ T cells") == n("cd4 T cell") == n("CD4_T_Cell")
    assert n("NK cells") == n("NK cell") and n("Classical monocytes") == n("classical Monocyte")
    assert n("Mast") != n("Mas") and n("glass") == "glass"
    assert n("B cells") != n("T cells")
    a = _adata(ref=["CD4+ T cells"] * 2 + ["B cells"] * 2, other=["cd4 T cell"] * 2 + ["B_cell"] * 2)
    assert xc.crosscheck(a, ["ref", "other"])["exact_name_agreement"] == 1.0


def test_requests_that_cannot_be_served_say_why():
    a = _adata(ref=REF, other=REF)
    a.obs["score"] = np.linspace(0, 1, a.n_obs)
    big = _adata(ref=[f"g{i}" for i in range(201)] + ["g0"], other=["x"] * 202)
    with pytest.raises(xc.CrosscheckError, match="numbers") as e:
        xc.crosscheck(a, ["ref", "score"])
    assert e.value.status == 400
    with pytest.raises(xc.CrosscheckError, match="201 labels"):
        xc.crosscheck(big, ["ref", "other"])
    with pytest.raises(xc.CrosscheckError, match="2 to 6"):
        xc.crosscheck(a, ["ref"])
    with pytest.raises(xc.CrosscheckError, match="2 to 6"):
        xc.crosscheck(a, ["ref", "ref"])
    with pytest.raises(xc.CrosscheckError, match="reference"):
        xc.crosscheck(a, ["ref", "other"], "nope")
    with pytest.raises(xc.CrosscheckError) as e:
        xc.crosscheck(a, ["ref", "missing"])
    assert e.value.status == 404


def test_string_and_bool_columns_count_as_labels():
    a = _adata(ref=REF)
    a.obs["text"] = np.array(REF, dtype=object)
    a.obs["flag"] = np.array([x == "A" for x in REF])
    out = xc.crosscheck(a, ["ref", "text", "flag"], "ref")
    assert out["comparisons"]["text"]["ari"] == 1.0
    assert out["comparisons"]["flag"]["mapping"]["True"]["to"] == "A"


def test_annotation_columns_are_found_by_what_made_them():
    a = _adata(cell_type_llm=REF, cell_type_ms=REF, cell_type_ct=REF, leiden=REF, other=REF)
    a.uns["cell_type_llm_llm_mapping"] = {"0": "A"}
    a.uns["cell_type_ms_markerscore_mapping"] = {"0": "A"}
    a.uns["gone_llm_mapping"] = {"0": "A"}  # its column was deleted
    a.uns["scview_celltypist"] = json.dumps({"cell_type_ct": {"model": "Immune_All_Low.pkl"}})
    found = xc.annotation_columns(a, "leiden")
    assert found == {"cell_type_ct": "CellTypist", "cell_type_llm": "LLM",
                     "cell_type_ms": "marker score", "leiden": "clustering"}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


def _save(dataset_manager, a: ad.AnnData) -> None:
    d = Path(dataset_manager.data_dir) / "uploads" / "ds"
    d.mkdir(parents=True, exist_ok=True)
    (d / "metadata.json").write_text(json.dumps({"id": "ds", "name": "ds", "status": "ready"}))
    a.write_h5ad(d / "data.h5ad")


async def test_endpoints_check_and_list(client, dataset_manager):
    a = _three_way()
    a.uns["o1_llm_mapping"] = {"0": "x"}
    _save(dataset_manager, a)
    url = "/api/v1/datasets/ds/annotation/crosscheck"
    info = (await client.get(url)).json()
    assert info["suggested"]["o1"] == "LLM" and info["max_columns"] == 6

    r = await client.post(url, json={"columns": ["ref", "o1", "o2"], "reference": "ref"})
    assert r.status_code == 200, r.text
    assert r.json()["n_unresolved"] == 1 and r.json()["mean_agreement"] == round(28 / 30, 4)



async def test_endpoint_errors(client, dataset_manager):
    a = _three_way()
    a.obs["score"] = np.linspace(0, 1, a.n_obs)
    _save(dataset_manager, a)
    url = "/api/v1/datasets/ds/annotation/crosscheck"
    assert (await client.post(url, json={"columns": ["ref", "score"]})).status_code == 400
    assert (await client.post(url, json={"columns": ["ref"]})).status_code == 400
    assert (await client.post(url, json={"columns": ["ref", "nope"]})).status_code == 404
    assert (await client.post("/api/v1/datasets/missing/annotation/crosscheck",
                              json={"columns": ["ref", "o1"]})).status_code == 404
