"""Co-pilot actions for Compare + DE between groups, and multi-step plans (C026).

The model's JSON is faked: these cover the allow-list validation of the new actions,
the ordering and binding of plan steps, and malformed replies.
"""

from __future__ import annotations

import json

from scview.core.assistant import (
    MAX_PLAN_STEPS,
    _coerce_actions,
    _deterministic_actions,
    extract_actions,
)

CTX = dict(
    columns=["cell_type", "stim", "ind", "leiden"], embeddings=["X_umap"], genes_upper={},
    n_cells=13836, active_clustering="leiden",
    values={"stim": ["CTRL", "STIM"], "cell_type": ["B cells", "NK cells", "CD14+ Monocytes"],
            "ind": ["101", "1015", "1016"], "leiden": ["0", "1", "2", "3"]},
)


def _coerce(items) -> list:
    return _coerce_actions(json.dumps(items), **CTX)


# --- A. Compare tabs ---------------------------------------------------------------


def test_open_compare_tab_prefills_valid_columns():
    [a] = _coerce([{"type": "open_compare_tab", "tab": "composition",
                    "condition": "stim", "group": "cell_type", "sample": "ind"}])
    assert (a.type, a.panel, a.tab) == ("open_compare_tab", "compare", "composition")
    assert (a.condition, a.column, a.sample) == ("stim", "cell_type", "ind")
    assert "Composition" in a.label and "stim" in a.label and not a.requires_confirm


def test_open_compare_tab_drops_unknown_columns_but_keeps_the_tab():
    [a] = _coerce([{"type": "open_compare_tab", "tab": "milo", "condition": "nope",
                    "group": ["cell_type"], "sample": "ind"}])
    assert a.tab == "milo" and a.condition is None and a.column is None and a.sample == "ind"


def test_open_compare_tab_one_column_cannot_be_condition_and_group():
    [a] = _coerce([{"type": "open_compare_tab", "tab": "planner", "condition": "stim", "group": "stim"}])
    assert a.condition == "stim" and a.column is None


def test_open_compare_tab_joint_takes_no_columns_and_bad_tabs_drop():
    [a] = _coerce([{"type": "open_compare_tab", "tab": "Joint", "condition": "stim"}])
    assert a.tab == "joint" and a.condition is None
    assert _coerce([{"type": "open_compare_tab", "tab": "volcano"}]) == []
    assert _coerce([{"type": "open_compare_tab"}]) == []


# --- B. DE between groups --------------------------------------------------------------


def test_open_de_compare_validates_groups_against_the_categories():
    [a] = _coerce([{"type": "open_de_compare", "column": "stim", "group_a": "stim", "group_b": "ctrl"}])
    assert (a.type, a.subtab, a.column, a.group_a, a.group_b) == (
        "open_de_compare", "de", "stim", "STIM", "CTRL")  # case-insensitive -> the real names
    assert not a.requires_confirm and "Click Run" in a.label


def test_open_de_compare_two_level_column_fills_the_other_group():
    [a] = _coerce([{"type": "open_de_compare", "column": "stim", "group_a": "STIM"}])
    assert (a.group_a, a.group_b) == ("STIM", "CTRL")


def test_open_de_compare_rejects_bad_requests():
    bad = [
        {"type": "open_de_compare", "column": "stim", "group_a": "STIM", "group_b": "STIM"},  # same
        {"type": "open_de_compare", "column": "stim", "group_a": "STIM", "group_b": "IFN"},  # not one
        {"type": "open_de_compare", "column": "nope", "group_a": "a", "group_b": "b"},  # no column
        {"type": "open_de_compare", "column": "cell_type", "group_a": "B cells"},  # 3 levels, no B
        {"type": "open_de_compare", "column": {"x": 1}, "group_a": "STIM", "group_b": "CTRL"},
    ]
    for item in bad:
        assert _coerce([item]) == [], item
    # No category list for the column (more than 100 levels): cannot be validated -> dropped
    no_vals = dict(CTX, values={})
    assert _coerce_actions(json.dumps([{"type": "open_de_compare", "column": "stim",
                                        "group_a": "STIM", "group_b": "CTRL"}]), **no_vals) == []


# --- C. Plans ------------------------------------------------------------------------------


def test_recluster_markers_enrichment_becomes_one_ordered_plan_bound_to_the_new_column():
    [plan] = _coerce([
        {"type": "cluster", "resolution": 1},
        {"type": "compute_markers"},
        {"type": "run_enrichment", "group": "3"},
    ])
    assert plan.type == "run_plan" and plan.requires_confirm
    assert [s.type for s in plan.steps] == ["cluster", "compute_markers", "run_enrichment"]
    clus, mk, enr = plan.steps
    assert clus.column == "scview_leiden_r1" and clus.from_step is None
    # Both later steps work on the NEW clustering (step 0), resolved by the UI after it runs
    assert (mk.column, mk.from_step, mk.params) == (
        "scview_leiden_r1", 0, {"marker_columns": ["scview_leiden_r1"]})
    assert (enr.column, enr.from_step, enr.value) == ("scview_leiden_r1", 0, "3")
    assert "3 steps" in plan.label and plan.advisory == ""


def test_plan_steps_name_the_new_column_loosely_or_an_existing_one():
    [plan] = _coerce([
        {"type": "cluster", "resolution": 0.8},
        {"type": "highlight_cluster", "column": "new_clusters", "value": "2"},
        {"type": "set_color_by", "column": "cell_type"},  # an existing column stays itself
        {"type": "open_de_compare", "column": "scview_leiden_r0.8", "group_a": "1", "group_b": "2"},
    ])
    _, hl, color, de = plan.steps
    assert (hl.column, hl.from_step, hl.value) == ("scview_leiden_r0.8", 0, "2")
    assert (color.column, color.from_step) == ("cell_type", None)
    assert (de.column, de.from_step, de.group_a, de.group_b) == ("scview_leiden_r0.8", 0, "1", "2")


def test_without_a_recluster_nothing_binds_and_markers_use_the_active_clustering():
    [plan] = _coerce([{"type": "compute_markers"}, {"type": "run_enrichment", "group": "9"},
                      {"type": "highlight_cluster", "column": "new_clusters", "value": "2"}])
    mk, enr = plan.steps  # the highlight on a column that does not exist is dropped
    assert mk.column == "leiden" and mk.from_step is None
    assert enr.column == "leiden" and enr.value is None  # "9" is not a leiden group
    assert "1 step I could not map was left out." in plan.advisory


def test_plan_is_capped_and_unknown_steps_are_noted():
    items = [{"type": "detect_doublets"}, {"type": "rm_rf"}, "junk"] + [
        {"type": "set_color_by", "column": "stim"} for _ in range(8)]
    [plan] = _coerce(items)
    assert len(plan.steps) == MAX_PLAN_STEPS
    assert plan.steps[0].type == "detect_doublets"
    assert f"first {MAX_PLAN_STEPS} steps" in plan.advisory
    assert "2 steps I could not map were left out." in plan.advisory


def test_view_only_lists_and_single_runs_are_not_plans():
    acts = _coerce([{"type": "set_color_by", "column": "stim"}, {"type": "open_panel", "panel": "compare"}])
    assert [a.type for a in acts] == ["set_color_by", "open_panel"]
    [one] = _coerce([{"type": "cluster", "resolution": 1}])
    assert one.type == "cluster" and one.requires_confirm and not one.steps


def test_malformed_replies_give_no_actions():
    for raw in ("", "no json", "[{]", '{"type": "cluster"}', "[1, 2, null]", '[{"type": 5}]'):
        assert _coerce_actions(raw, **CTX) == [], raw


def test_deterministic_fallback_keeps_the_order_the_message_names():
    q = "re-cluster at resolution 1, then compute markers and run enrichment for cluster 3"
    [plan] = _deterministic_actions(q, **CTX)
    assert [s.type for s in plan.steps] == ["cluster", "compute_markers", "run_enrichment"]
    assert plan.steps[0].params["clustering_resolution"] == 1.0
    assert plan.steps[2].value == "3" and plan.steps[2].from_step == 0
    two = _deterministic_actions("run enrichment after you compute markers", **CTX)
    assert [s.type for s in two[0].steps] == ["run_enrichment", "compute_markers"]  # as written


async def test_extract_actions_reads_the_plan_from_one_model_call(monkeypatch):
    calls: list[dict] = []
    reply = ('Here is the plan:\n[{"type":"cluster","resolution":1},'
             '{"type":"compute_markers"},{"type":"run_enrichment","group":"3"}]')

    class _Resp:
        choices = [type("C", (), {"message": type("M", (), {"content": reply})()})()]

    class _Client:
        def __init__(self, **_):
            self.chat = type("Ch", (), {"completions": self})()

        async def create(self, **kw):
            calls.append(kw)
            return _Resp()

    monkeypatch.setattr("openai.AsyncOpenAI", _Client)
    [plan] = await extract_actions(
        "re-cluster at resolution 1, then compute markers and run enrichment for cluster 3",
        None, CTX["columns"], CTX["embeddings"], {}, 100, "leiden", "key", "m", values=CTX["values"])
    assert len(calls) == 1  # no second round trip for the plan
    prompt = calls[0]["messages"][0]["content"]
    assert "open_compare_tab" in prompt and "open_de_compare" in prompt and "order" in prompt
    assert [s.type for s in plan.steps] == ["cluster", "compute_markers", "run_enrichment"]
