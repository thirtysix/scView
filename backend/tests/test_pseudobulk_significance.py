"""A pseudobulk run with 0 significant genes says how close it came (Wilson 2022 kidney:
voom found 0 of 12,294 in PT, smallest adjusted p .69, only readable from table.tsv.gz)."""

from __future__ import annotations

import pytest

from scview.api.v1.de import differential_expression_compare
from scview.core.pseudobulk import significance
from tests.test_pseudobulk_covariates import _req
from tests.test_mcp import _call, _register, msigdb_dir, test_settings  # noqa: F401 - fixtures
from tests.test_pseudobulk_voom import _DM, _counts


def test_zero_significant_names_the_smallest_padj_and_nominal_hits():
    rows = [(f"G{i}", 0.1, 0.5, 0.9) for i in range(100)]
    rows[7] = ("PCK1", 1.2, 0.001, 0.69)
    rows += [(f"N{i}", 0.3, 0.02, 0.8) for i in range(3)]
    s = significance(rows)
    assert s["n_significant"] == 0 and s["min_padj"] == pytest.approx(0.69) and s["min_padj_gene"] == "PCK1"
    assert s["n_nominal"] == 4 and s["n_nominal_expected"] == pytest.approx(5.2)
    assert s["note"] == ("No gene reaches adjusted p < 0.05: the smallest adjusted p is 0.69 (PCK1). "
                         "4 of 103 genes have p < 0.05 before adjustment, about what chance alone gives.")
    # Many nominal hits that the correction removes: said so.
    many = [(f"G{i}", 0.5, 0.01, 0.2) for i in range(50)] + [(f"H{i}", 0.0, 0.6, 0.9) for i in range(50)]
    assert "more than the 5 chance alone gives" in significance(many)["note"]
    # Something significant: no note.
    assert significance([("A", 2.0, 1e-8, 1e-6), ("B", 0.1, 0.5, 0.9)])["note"] is None


@pytest.mark.asyncio
async def test_pseudobulk_response_carries_the_summary():
    a = _counts([f"a{i}" for i in range(4)], [f"b{i}" for i in range(4)], seed=11)
    res = await differential_expression_compare("x", _req(pb_method="voom"), _DM(a))
    s = res.significance
    assert s is not None and s["n_tested"] == res.n_genes_tested
    assert s["n_significant"] == sum(1 for g in res.genes if g.pval_adj < 0.05) >= 1
    assert s["min_padj_gene"] in {"G0", "G1"} and s["note"] is None


async def test_mcp_says_why_nothing_is_listed(client, dataset_manager):
    did = _register(dataset_manager)
    # Sex has no effect in this data: 2 female vs 2 male donors, nothing significant.
    r = await _call("differential_expression", {
        "dataset_id": did, "column": "sex", "group_a": "female", "group_b": "male",
        "method": "pseudobulk", "replicate_key": "donor_id"})
    assert r["n_up"] == 0 and r["n_down"] == 0
    assert r["none_significant"].startswith("No gene reaches adjusted p < 0.05: the smallest adjusted p is")
