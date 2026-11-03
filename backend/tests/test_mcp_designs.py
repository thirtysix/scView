"""MCP statistics tools with richer designs (batch 39): per_donor_expression's two tested
levels, composition_test's subsets and stored ids, pseudobulk covariates and the
repeated-sample and cell-level sex warnings. Same in-process client as tests/test_mcp.py."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from tests.test_mcp import _adata, _call, msigdb_dir, test_settings  # noqa: F401 — fixtures


def _register(dm, did: str, edit=None) -> str:
    a = _adata().copy()
    if edit is not None:
        edit(a)
    d = Path(dm.data_dir) / "uploads" / did
    d.mkdir(parents=True, exist_ok=True)
    (d / "metadata.json").write_text(json.dumps({"id": did, "name": did, "status": "ready"}))
    a.write_h5ad(d / "data.h5ad")
    return did


def _with_unassigned(a) -> None:
    """A labelled column with a third level, as set_labels leaves it: D3 'Unassigned'."""
    lab = a.obs["stim"].astype(str).where(a.obs["donor_id"].astype(str) != "D3", "Unassigned")
    a.obs["lab"] = pd.Categorical(lab, categories=["Unassigned", "CTRL", "STIM"])


async def test_per_donor_expression_two_levels_of_three(client, dataset_manager):
    did = _register(dataset_manager, "pd3", _with_unassigned)
    args = {"dataset_id": did, "gene": "G12", "groupby": "cluster", "condition": "lab",
            "sample": "donor_id", "groups": ["A"]}
    r = await _call("per_donor_expression", args)
    # Before: 'Unassigned' as a third level gave paired, n_pairs 0 and no test.
    assert r["tested_conditions"] == ["CTRL", "STIM"] and r["difference"] == "STIM - CTRL"
    assert r["design"] == {"paired": True, "n_pairs": 3, "conditions": ["CTRL", "STIM"]}
    assert r["paired_tests"]["A"]["n_pairs"] == 3 and r["paired_tests"]["A"]["mean_diff"] > 0
    assert set(r["values_by_group"]["A"]) == {"Unassigned", "CTRL", "STIM"}
    flip = await _call("per_donor_expression", {**args, "conditions": ["STIM", "CTRL"], "reference": "STIM"})
    assert flip["difference"] == "CTRL - STIM" and flip["paired_tests"]["A"]["mean_diff"] < 0
    assert "groups_not_found" not in flip and flip["n_dropped"] == 0
    text = await _call("per_donor_expression", {**args, "conditions": ["STIM", "nope"]}, error=True)
    assert "not a value of 'lab'" in text


def _with_samples(a) -> None:
    """Two libraries per donor and condition (sample), and a broad class over the clusters."""
    lib = a.obs["donor_id"].astype(str) + "_" + a.obs["stim"].astype(str) + "_" + (
        (pd.Series(range(a.n_obs), index=a.obs.index) % 2).astype(str))
    a.obs["sample"] = pd.Categorical(lib)
    a.obs["klass"] = pd.Categorical(["all"] * a.n_obs)


async def test_composition_subset_leave_out_and_stored_ids(client, dataset_manager):
    did = _register(dataset_manager, "cmp", _with_samples)
    base = {"dataset_id": did, "condition_key": "stim", "group_key": "cluster", "sample_key": "sample"}
    r = await _call("composition_test", base)
    # Before: result_id null and every comparison stored under one label.
    assert r["result_id"] and r["comparison"] == "'cluster' by 'stim' (STIM vs CTRL)"
    assert r["repeated_donors"]["donor_key"] == "donor_id"
    assert any("Samples are not independent: 4 of 4 'donor_id' values" in w for w in r["warnings"])
    r2 = await _call("composition_test", {**base, "within_column": "klass", "within_value": ["all"],
                                          "exclude_samples": ["D0_CTRL_0", "D0_CTRL_1"]})
    assert r2["result_id"] != r["result_id"] and r2["within"]["values"] == ["all"]
    assert r2["n_samples"] == {"CTRL": 6, "STIM": 8}
    assert r2["comparison"].endswith("within klass = all; without D0_CTRL_0, D0_CTRL_1")
    text = await _call("composition_test", {**base, "exclude_samples": ["nobody"]}, error=True)
    assert "not a value of 'sample'" in text


async def test_pseudobulk_covariates_and_repeated_samples(client, dataset_manager):
    def edit(a):
        _with_samples(a)
        a.obs["batch"] = pd.Categorical(a.obs["donor_id"].astype(str).map(
            {"D0": "b1", "D1": "b2", "D2": "b1", "D3": "b2"}))

    did = _register(dataset_manager, "pbc", edit)
    base = {"dataset_id": did, "column": "stim", "group_a": "STIM", "group_b": "CTRL",
            "within_column": "cluster", "within_value": "A", "method": "pseudobulk"}
    # Before: no covariates argument; the design was always ~condition (or ~donor + condition).
    r = await _call("differential_expression", {**base, "replicate_key": "donor_id", "covariates": ["sex"]})
    assert r["design"] == "~donor_id + condition" and "covariates" not in r
    assert any(w.startswith("covariate: 'sex' is constant within each donor") for w in r["warnings"])
    r = await _call("differential_expression", {**base, "replicate_key": "sample", "covariates": ["batch"],
                                                "pb_method": "voom"})
    assert r["design"] == "~batch + condition" and r["covariates"] == ["batch"]
    assert r["n_samples_a"] == 8 and r["n_donors_a"] == 4
    assert any(w.startswith("replicates: Samples are not independent: 4 of 4 'donor_id'") for w in r["warnings"])
    text = await _call("differential_expression", {**base, "replicate_key": "sample", "covariates": ["stim"]},
                       error=True)
    assert "cannot also be a covariate" in text
    text = await _call("differential_expression", {"dataset_id": did, "column": "stim", "group_a": "STIM",
                                                   "group_b": "CTRL", "covariates": ["sex"]}, error=True)
    assert "method='pseudobulk'" in text


async def test_cell_level_de_warns_on_cell_sex_shares(client, dataset_manager):
    def edit(a):  # female D0, D1 keep 1 in 4 STIM cells: 2 F + 2 M donors a side, 20% vs 50% of cells
        keep = ~(a.obs["donor_id"].isin(["D0", "D1"]) & (a.obs["stim"] == "STIM")).to_numpy()
        keep |= (pd.Series(range(a.n_obs), index=a.obs.index) % 4 == 0).to_numpy()
        a._inplace_subset_obs(keep)

    did = _register(dataset_manager, "sexc", edit)
    r = await _call("differential_expression", {"dataset_id": did, "column": "stim", "group_a": "STIM",
                                                "group_b": "CTRL", "replicate_key": "donor_id"})
    assert any(w.startswith("sex: Cells differ in sex between the sides") for w in r["warnings"])
