"""Pseudobulk says which genes its filters left out, and why for the genes asked about:
QA round 7's NRG1 (IDA vs M2 macrophages) was untestable (M2 donor sums near 0) and only
visible as a missing row."""

from __future__ import annotations

import pytest
from fastapi import HTTPException  # noqa: F401 - the route raises it

from scview.api.v1.de import differential_expression_compare
from scview.core import pseudobulk
from tests.test_de_pseudobulk import DONORS, _DM, _donor_counts, _req


def _with_rare_gene():
    a = _donor_counts(DONORS, DONORS, g=60)
    X = a.X.tolil()
    X[:, 59] = 0  # G59: a few counts in one A donor's cells only
    a_cells = (a.obs["cond"] == "A").to_numpy().nonzero()[0]
    for i in a_cells[:12]:
        X[i, 59] = 3
    a.X = X.tocsr()
    return a


@pytest.mark.asyncio
async def test_gene_filter_counts_each_filter_and_explains_asked_genes(monkeypatch):
    monkeypatch.setattr(pseudobulk, "MAX_GENES", 50)
    a = _with_rare_gene()
    res = await differential_expression_compare(
        "x", _req(replicate_key="donor", genes=["G0", "g59", "NOPE", "G1"]), _DM(a))
    f = res.gene_filter
    assert f["n_genes"] == 60 and f["n_tested"] == len(res.genes)
    assert f["dropped"]["low_counts"] >= 1 and f["dropped"]["gene_cap"] >= 1
    assert f["n_genes"] - sum(f["dropped"].values()) == f["n_tested"]
    assert "10 counts in at least 5 of the 10 donor samples" in f["rule"]
    by = {g["gene"]: g for g in res.asked_genes}
    assert by["G0"]["status"] == "tested" and by["G0"]["log2fc"] > 1 and by["G0"]["pval_adj"] < 0.05
    rare = by["G59"]  # matched whatever the case
    assert rare["status"] == "low_counts" and rare["needs"] == 5
    assert "in 1 donor sample (A 1 of 5, B 0 of 5)" in rare["reason"] and rare["counts"]["B"] == {d: 0 for d in DONORS}
    assert by["NOPE"]["status"] == "not_found"
    assert [g["gene"] for g in res.asked_genes] == ["G0", "G59", "NOPE", "G1"]


@pytest.mark.asyncio
async def test_no_genes_asked_gives_the_filter_summary_only():
    a = _donor_counts(DONORS, DONORS, g=40)
    res = await differential_expression_compare("x", _req(replicate_key="donor"), _DM(a))
    assert res.asked_genes is None and res.gene_filter["n_genes"] == 40


@pytest.mark.asyncio
async def test_mcp_differential_expression_reports_asked_genes(client, dataset_manager):
    from tests.test_label_import import _mcp
    from tests.test_mcp import _register

    did = _register(dataset_manager)
    r = await _mcp("differential_expression", {
        "dataset_id": did, "column": "stim", "group_a": "STIM", "group_b": "CTRL", "method": "pseudobulk",
        "replicate_key": "donor_id", "genes": ["G12", "NOPE"]})
    assert r["gene_filter"]["n_genes"] == 40 and "dropped" in r["gene_filter"]
    by = {g["gene"]: g for g in r["asked_genes"]}
    assert by["G12"]["status"] == "tested" and by["NOPE"]["status"] == "not_found"
    r = await _mcp("differential_expression", {
        "dataset_id": did, "column": "stim", "group_a": "STIM", "group_b": "CTRL", "genes": ["G12", "NOPE"]})
    by = {g["gene"]: g for g in r["asked_genes"]}
    assert by["G12"]["status"] == "tested" and by["NOPE"]["status"] == "not_tested"
    assert "gene_filter" not in r
