"""Small API leftovers from the Emont 2022 QA: ``within_value`` is one value or a list on
every route (the composition test always took a list, DE and plots a string), and a DE
result names the column its ``n_donors_*`` count (a sample column is not donors)."""

from __future__ import annotations

import json

import pytest

from scview.api.v1.de import (
    DEAuditRequest, DECompareRequest, _CompareScope, differential_expression_compare,
    differential_expression_composition,
)
from scview.mcp import tools
from tests.test_mcp import _call, _register
from tests.test_pseudobulk_voom import _DM, _counts


def _typed(seed: int = 0):
    a = _counts([f"d{i}" for i in range(4)], [f"e{i}" for i in range(4)], seed=seed)
    a.obs["type"] = (["T", "B", "NK"] * (a.n_obs // 3 + 1))[: a.n_obs]
    a.obs["type"] = a.obs["type"].astype("category")
    return a


def test_within_value_is_one_value_or_a_list():
    one = _CompareScope(split_by="cond", group_a="A", group_b="B", within_column="type", within_value=["T"])
    assert one.within_value == "T" and one.within_values() == ["T"] and one.within_text() == "T"
    two = _CompareScope(split_by="cond", group_a="A", group_b="B", within_column="type", within_value=["T", "B"])
    assert two.within_values() == ["T", "B"] and two.within_text() == "T, B"
    assert _CompareScope(split_by="c", group_a="A", group_b="B", within_value=[]).within_value is None
    assert DEAuditRequest(within_value=["T", "B"]).within_value == ["T", "B"]
    assert DEAuditRequest(within_value="T").within_value == "T"


@pytest.mark.asyncio
async def test_de_within_a_list_pools_the_subgroups_and_a_one_item_list_is_the_value():
    a = _typed()
    base = dict(split_by="cond", group_a="A", group_b="B", within_column="type", replicate_key="donor")
    as_str = await differential_expression_compare("x", DECompareRequest(**base, within_value="T"), _DM(a))
    as_list = await differential_expression_compare("x", DECompareRequest(**base, within_value=["T"]), _DM(a))
    assert (as_list.n_selected, as_list.n_rest, as_list.label) == (as_str.n_selected, as_str.n_rest, as_str.label)
    both = await differential_expression_compare("x", DECompareRequest(**base, within_value=["T", "B"]), _DM(a))
    t_b = int(((a.obs["type"].isin(["T", "B"])) & (a.obs["cond"] == "A")).sum())
    assert both.n_selected == t_b and both.label == "A (T, B)"
    comp = await differential_expression_composition(
        "x", split_by="cond", group_a="A", group_b="B", within_column="type", within_value=["T", "B"],
        column=None, dm=_DM(a))
    assert comp is not None


@pytest.mark.asyncio
async def test_n_donors_names_the_column_it_counts():
    a = _counts([f"m{i}_ING" for i in range(4)], [f"m{i}_PG" for i in range(4)], seed=3)
    a.obs["donor_id"] = a.obs["donor"].astype(str).str.split("_").str[0].astype("category")
    cell = await differential_expression_compare(
        "x", DECompareRequest(split_by="cond", group_a="A", group_b="B", replicate_key="donor"), _DM(a))
    assert cell.replicate_key == "donor" and cell.n_donors_column == "donor"  # samples, said so
    pb = await differential_expression_compare(
        "x", DECompareRequest(split_by="cond", group_a="A", group_b="B", replicate_key="donor",
                              method="pseudobulk"), _DM(a))
    assert pb.n_donors_column == "donor_id" and pb.n_donors_a == 4  # the mice the samples nest in


async def test_mcp_plot_within_a_list_and_markers_say_why_no_result_id(client, dataset_manager):
    did = _register(dataset_manager)
    r = await _call("plot", {"dataset_id": did, "kind": "dotplot", "genes": ["G0", "G1"], "groupby": "stim",
                             "within_column": "cluster", "within_value": ["A", "B"]})
    caption = next(b.text for b in r if getattr(b, "text", None))
    assert "cluster = A, B" in caption and "2 groups of stim" in caption
    r = await _call("plot", {"dataset_id": did, "kind": "violin", "gene": "G0", "groupby": "stim",
                             "within_column": "cluster", "within_value": "A"})
    assert any("cluster = A" in getattr(b, "text", "") for b in r)
    r = await _call("differential_expression", {"dataset_id": did, "column": "stim", "group_a": "STIM",
                                                "group_b": "CTRL", "within_column": "cluster",
                                                "within_value": ["A"]})
    assert "STIM (A)" in r["label"]
    assert r["n_donors_column"] == "donor_id"
    doc = tools.get_markers.__doc__ or ""
    assert "list_results does not" in doc and "no result_id" in doc
    assert json.dumps(r)  # plain JSON
