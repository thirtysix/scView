"""One label column from one column per dataset in a joint embedding (core/integration.py
``combine_labels``; v0.82).

Hub practice run (2026-11-18): a 2026 Seurat object labelled its cells in ``Celltype``, a
2024 one in ``cell_line``; the joint dataset kept both, each NA for the other dataset's
cells, so no column named all 8 samples and a label table had to be imported."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from scview.core import provenance, replay_script
from scview.core.integration import (
    CombineColumns,
    DatasetCheck,
    build_plan as _build_plan,
    build_unit,
    merge_joint,
    record_integration_provenance,
)
from scview.core.ingestion.merge import MergeJoin
from tests.test_integration import _add_dataset, _counts_dataset
from tests.test_mcp import msigdb_dir, test_settings  # noqa: F401 — fixtures


def _a():
    a = _counts_dataset(seed=1)
    n = a.n_obs
    a.obs["Celltype"] = pd.Categorical(["LineA"] * (n // 2) + ["LineA resistant"] * (n - n // 2))
    a.obs["orig.ident"] = pd.Categorical(["s26"] * n)
    a.obs["RNA_snn_res.0.5"] = pd.Categorical(["0", "1", "2"] * (n // 3))
    return a


def _b():
    b = _counts_dataset(seed=2)
    n = b.n_obs
    b.obs["cell_line"] = pd.Categorical(["LineB parental"] * (n // 2) + ["LineB resistant"] * (n - n // 2))
    b.obs["orig.ident"] = pd.Categorical(["s24"] * n)
    return b


def build_plan(checks, units, **kw):
    """The plan with a check per unit (an empty list blocks: fewer than two datasets)."""
    checks = checks or [DatasetCheck(id=u.dataset_id, name=u.name, label=u.label, n_cells=u.n_cells)
                        for u in units.values()]
    return _build_plan(checks, units, **kw)


def _units():
    return {lab: build_unit(x, dataset_id=lab.lower(), name=lab, label=lab, fallback_paths=[])
            for lab, x in (("A", _a()), ("B", _b()))}


def test_the_plan_suggests_combining_columns_each_in_one_dataset():
    units = _units()
    plan = build_plan([], units, max_datasets=8, max_cells=10**6)
    s = plan.label_suggestion
    assert s is not None
    # a cell-line / cell-type name wins over the other only-here column (RNA_snn_res.0.5)
    assert s.columns == {"a": "Celltype", "b": "cell_line"} and s.name == "Celltype"
    assert s.shared_values == []
    assert "Celltype" in plan.label_columns["a"] and "orig.ident" in plan.label_columns["b"]  # one value is enough
    assert "cell_type" in plan.label_columns["a"]  # shared columns can be picked too
    assert plan.combined == [] and plan.ok


def _b_two_lines():
    """QA-8 shape: ``Cell_line`` (first in obs, a coarser value, 30 cells empty) and a complete
    ``cell_line``; the suggestion took ``Cell_line``."""
    b = _b()
    n = b.n_obs
    coarse = ["LineB"] * (n // 2) + ["LineB resistant"] * (n - n // 2)
    coarse[:30] = [None] * 30
    b.obs.insert(0, "Cell_line", pd.Categorical(coarse))
    return b


def test_the_suggestion_takes_the_complete_column_and_says_how_many_cells_have_none():
    units = {lab: build_unit(x, dataset_id=lab.lower(), name=lab, label=lab, fallback_paths=[])
             for lab, x in (("A", _a()), ("B", _b_two_lines()))}
    plan = build_plan([], units, max_datasets=8, max_cells=10**6)
    s = plan.label_suggestion
    assert s.columns == {"a": "Celltype", "b": "cell_line"}
    assert s.n_missing == {"a": 0, "b": 0} and s.n_cells == {"a": 180, "b": 180}
    # a column the user picks with empty cells is allowed, and says so
    plan = build_plan([], units, max_datasets=8, max_cells=10**6,
                      combine=[CombineColumns(name="line", columns={"a": "Celltype", "b": "Cell_line"})])
    assert plan.ok, plan.blocking
    c = plan.combined[0]
    assert c.n_missing == {"a": 0, "b": 30} and c.n_cells == {"a": 180, "b": 180}
    assert int(merge_joint(units).obs["line"].isna().sum()) == 30


def test_a_specific_name_beats_a_generic_ident_and_placeholders_count_as_empty():
    """QA-8 real data: ``old.ident`` is complete but holds "SeuratProject" for the 30 cells
    ``Cell_line`` leaves empty; the complete ``cell_line`` must win, and the placeholder counts."""
    b = _b_two_lines()
    old = b.obs["Cell_line"].astype(object).where(b.obs["Cell_line"].notna(), "SeuratProject")
    b.obs.insert(0, "old.ident", pd.Categorical(old))
    units = {lab: build_unit(x, dataset_id=lab.lower(), name=lab, label=lab, fallback_paths=[])
             for lab, x in (("A", _a()), ("B", b))}
    s = build_plan([], units, max_datasets=8, max_cells=10**6).label_suggestion
    assert s.columns["b"] == "cell_line" and s.n_missing["b"] == 0
    plan = build_plan([], units, max_datasets=8, max_cells=10**6,
                      combine=[CombineColumns(name="line", columns={"a": "Celltype", "b": "old.ident"})])
    assert plan.combined[0].n_missing == {"a": 0, "b": 30}


def test_a_label_name_still_beats_a_complete_other_column():
    a = _a()
    a.obs["Celltype"] = a.obs["Celltype"].astype(object)
    a.obs.loc[a.obs.index[:5], "Celltype"] = None  # a few empty cells; RNA_snn_res.0.5 is complete
    a.obs["Celltype"] = a.obs["Celltype"].astype("category")
    units = {lab: build_unit(x, dataset_id=lab.lower(), name=lab, label=lab, fallback_paths=[])
             for lab, x in (("A", a), ("B", _b()))}
    s = build_plan([], units, max_datasets=8, max_cells=10**6).label_suggestion
    assert s.columns["a"] == "Celltype" and s.n_missing["a"] == 5


def test_a_prefixed_combined_column_never_replaces_a_source_column():
    """QA-8 joint 0eebab85: 'cell_line' with prefix replaced B's own cell_line values."""
    spec = CombineColumns(name="cell_line", columns={"a": "Celltype", "b": "cell_line"}, prefix=True)
    plan = build_plan([], _units(), max_datasets=8, max_cells=10**6, combine=[spec])
    assert not plan.ok
    assert plan.blocking == ["'cell_line' is B's own column: with the dataset label before each value, the "
                             "combined column would replace it. Give the combined column another name."]
    # without the prefix the name is fine: B's cells keep exactly their own values
    units = _units()
    plan = build_plan([], units, max_datasets=8, max_cells=10**6,
                      combine=[spec.model_copy(update={"prefix": False})])
    assert plan.ok
    joint = merge_joint(units)
    b = joint.obs["dataset"] == "B"
    assert (joint.obs["cell_line"][b].astype(str).isin(["LineB parental", "LineB resistant"])).all()
    # another name with the prefix: both columns kept
    units = _units()
    plan = build_plan([], units, max_datasets=8, max_cells=10**6,
                      combine=[spec.model_copy(update={"name": "line"})])
    assert plan.ok
    joint = merge_joint(units)
    assert joint.obs["line"][b].astype(str).str.startswith("B: ").all()
    assert not joint.obs["cell_line"][b].astype(str).str.startswith("B: ").any()


def test_combined_columns_are_user_annotation_columns():
    from scview.core.integration import register_combined
    from scview.core.user_labels import label_cells, user_columns

    units = _units()
    plan = build_plan([], units, max_datasets=8, max_cells=10**6,
                      combine=[CombineColumns(name="line", columns={"a": "Celltype", "b": "cell_line"})])
    joint = merge_joint(units)
    assert register_combined(joint, plan.combined) == ["line"]
    assert user_columns(joint) == ["line"]
    out = label_cells(joint, "line", "LineB fixed", [0, 1, 2], new_column=False)
    assert out["n_cells"] == 3 and joint.obs["line"].iloc[0] == "LineB fixed"
    assert register_combined(joint, []) == [] and user_columns(joint) == ["line"]


def test_no_suggestion_when_the_label_columns_are_shared():
    units = {lab: build_unit(_counts_dataset(seed=i), dataset_id=lab.lower(), name=lab, label=lab,
                             fallback_paths=[]) for i, lab in enumerate("AB")}
    assert build_plan([], units, max_datasets=8, max_cells=10**6).label_suggestion is None


def test_combined_column_names_every_cell_values_as_they_are():
    units = _units()
    spec = CombineColumns(name="Celltype", columns={"a": "Celltype", "b": "cell_line"})
    plan = build_plan([], units, max_datasets=8, max_cells=10**6, combine=[spec])
    assert plan.ok and plan.blocking == []
    c = plan.combined[0]
    assert c.name == "Celltype" and c.n_values == 4 and c.labels == {"a": "A", "b": "B"}
    assert "Celltype" in plan.shared_columns
    joint = merge_joint(units)
    col = joint.obs["Celltype"]
    assert col.notna().all()
    assert set(col.cat.categories) == {"LineA", "LineA resistant", "LineB parental", "LineB resistant"}
    assert (col[joint.obs["dataset"] == "B"].isin(["LineB parental", "LineB resistant"])).all()
    assert "cell_line" in joint.obs.columns  # the source columns stay


def test_prefix_keeps_values_of_two_datasets_apart():
    units = _units()
    units["B"].adata.obs["orig.ident"] = "s26"
    plan = build_plan([], units, max_datasets=8, max_cells=10**6,
                      combine=[CombineColumns(name="sample", columns={"a": "orig.ident", "b": "orig.ident"},
                                              prefix=True)])
    assert plan.ok, plan.blocking
    assert plan.combined[0].shared_values == ["s26"]
    joint = merge_joint(units)
    assert set(joint.obs["sample"].cat.categories) == {"A: s26", "B: s26"}


def test_a_dataset_left_out_has_no_value_and_bad_specs_block():
    units = _units()
    c = _counts_dataset(seed=3)
    units["C"] = build_unit(c, dataset_id="c", name="C", label="C", fallback_paths=[])
    plan = build_plan([], units, max_datasets=8, max_cells=10**6,
                      combine=[CombineColumns(name="line", columns={"a": "Celltype", "b": "cell_line"})])
    assert plan.ok
    joint = merge_joint(units)
    assert joint.obs["line"][joint.obs["dataset"] == "C"].isna().all()
    assert joint.obs["line"][joint.obs["dataset"] != "C"].notna().all()

    def blocked(spec):
        p = build_plan([], _units(), max_datasets=8, max_cells=10**6, combine=[spec])
        assert not p.ok
        return " | ".join(p.blocking)

    assert "is also a column of A" in blocked(CombineColumns(name="orig.ident",
                                                              columns={"a": "Celltype", "b": "cell_line"}))
    assert "not carried" in blocked(CombineColumns(name="x", columns={"a": "barcode", "b": "cell_line"}))
    assert "two or more datasets" in blocked(CombineColumns(name="x", columns={"a": "Celltype"}))
    assert "not in this joint run" in blocked(CombineColumns(name="x", columns={"a": "Celltype", "zz": "q"}))
    assert "cannot be the combined" in blocked(CombineColumns(name="dataset",
                                                              columns={"a": "Celltype", "b": "cell_line"}))


def test_recorded_and_in_the_replay_comment():
    units = _units()
    plan = build_plan([], units, max_datasets=8, max_cells=10**6,
                      combine=[CombineColumns(name="Celltype", columns={"a": "Celltype", "b": "cell_line"})])
    joint = merge_joint(units)
    record_integration_provenance(joint, units, join=MergeJoin.inner, method="harmony",
                                  apply_reconciliation=True, combined=plan.combined)
    step = provenance.read_provenance(joint)["history"][-1]
    assert step["params"]["combined_labels"] == [
        {"name": "Celltype", "prefix": False, "columns": {"A": "Celltype", "B": "cell_line"}}]
    snip = replay_script._integrate(step["params"], None)
    text = "\n".join(snip.code)
    assert "#   'Celltype' <- A: 'Celltype', B: 'cell_line'" in text


@pytest.mark.asyncio
async def test_routes_plan_and_run_with_a_combined_column(client, tmp_data_dir):
    _add_dataset(tmp_data_dir, "a", "A", _a())
    _add_dataset(tmp_data_dir, "b", "B", _b())
    r = await client.post("/api/v1/integrate/plan", json={"dataset_ids": ["a", "b"]})
    plan = r.json()
    assert plan["label_suggestion"]["columns"] == {"a": "Celltype", "b": "cell_line"}
    combine = [{"name": "line", "columns": plan["label_suggestion"]["columns"]}]
    r = await client.post("/api/v1/integrate/plan", json={"dataset_ids": ["a", "b"], "combine": combine})
    assert r.json()["combined"][0]["n_values"] == 4 and r.json()["ok"]
    r = await client.post("/api/v1/integrate/plan", json={
        "dataset_ids": ["a", "b"], "combine": [{"name": "line", "columns": {"a": "nope", "b": "cell_line"}}]})
    assert r.json()["ok"] is False and "'nope' is not a label column of A" in r.json()["blocking"][-1]
    r = await client.post("/api/v1/integrate/run-stream", json={
        "dataset_ids": ["a", "b"], "name": "AB", "combine": combine,
        "params": {"n_top_genes": 100, "n_comps": 20}})
    last = r.text.strip().splitlines()[-1]
    assert last.startswith("data: ") and "dataset_id" in last, r.text[-400:]
    new_id = json.loads(last[6:])["dataset_id"]
    info = (await client.get(f"/api/v1/datasets/{new_id}")).json()
    assert "line" in {c["name"] for c in info["obs_columns"]}
    recipe = (await client.get(f"/api/v1/datasets/{new_id}/recipe")).json()
    steps = recipe.get("steps") or recipe.get("recipe") or recipe
    integ = next(s for s in steps if s["step"] == "integrate")
    assert integ["params"]["combined_labels"][0]["columns"] == {"A": "Celltype", "B": "cell_line"}
    # QA-8 J2: the combined column takes labels like a column the user made (after a reload too)
    assert (await client.get(f"/api/v1/datasets/{new_id}/labels/columns")).json()["columns"] == ["line"]
    n = info["n_cells"]
    table = "\n".join(["label", *["LineA fixed"] * 4, *["NA"] * (n - 4)]) + "\n"
    r = await client.post(f"/api/v1/datasets/{new_id}/labels/import-file",
                          files={"file": ("fix.csv", table.encode(), "text/csv")},
                          data={"column": "line", "new_column": "false", "match": "order", "label_column": "label"})
    assert r.status_code == 200, r.text
    assert r.json()["n_changed"] == 4 and r.json()["n_unassigned"] == 0
    r = await client.post(f"/api/v1/datasets/{new_id}/metadata/line/rename",
                          json={"old": "LineA fixed", "new": "LineA parental"})
    assert r.status_code == 200, r.text
    script = (await client.get(f"/api/v1/datasets/{new_id}/recipe/script")).text
    assert "#   'line' <- A: 'Celltype', B: 'cell_line'" in script
    assert "existing=adata.obs['line']" in script


async def test_mcp_plan_suggests_and_checks_a_combination(client, tmp_data_dir):
    from tests.test_mcp import _call

    _add_dataset(tmp_data_dir, "a", "A", _a())
    _add_dataset(tmp_data_dir, "b", "B", _b())
    plan = await _call("joint_embedding_plan", {"dataset_ids": ["a", "b"]})
    assert plan["label_suggestion"]["columns"] == {"a": "Celltype", "b": "cell_line"}
    plan = await _call("joint_embedding_plan", {"dataset_ids": ["a", "b"], "combine": [
        {"name": "line", "columns": {"a": "Celltype", "b": "cell_line"}, "prefix": True}]})
    assert plan["ok"] is True
    combined = plan["combined"]
    assert combined[0]["name"] == "line" and combined[0]["n_values"] == 4 and combined[0]["prefix"] is True
    assert combined[0]["n_missing"] == {"a": 0, "b": 0} and combined[0]["n_cells"] == {"a": 180, "b": 180}
    plan = await _call("joint_embedding_plan", {"dataset_ids": ["a", "b"], "combine": [
        {"name": "cell_line", "columns": {"a": "Celltype", "b": "cell_line"}, "prefix": True}]})
    assert plan["ok"] is False and "is B's own column" in plan["blocking"][0]
