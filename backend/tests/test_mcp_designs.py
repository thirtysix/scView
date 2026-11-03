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
