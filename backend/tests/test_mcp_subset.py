"""MCP subset_dataset: a new dataset of some cells, the next call named."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.test_mcp import _call, _register

pytestmark = pytest.mark.asyncio


async def test_subset_dataset_by_groups_and_by_filter(client, dataset_manager):
    did = _register(dataset_manager)
    dry = await _call("subset_dataset", {"dataset_id": did, "column": "cluster", "values": ["A"], "dry_run": True})
    assert dry["n_cells"] == 240 and "dataset_id" not in dry and "without dry_run" in dry["next"]

    r = await _call("subset_dataset", {"dataset_id": did, "column": "cluster", "values": ["A"]})
    new = r["dataset_id"]
    assert r["n_cells"] == 240 and r["n_parent_cells"] == 480 and r["parent_id"] == did
    assert r["name"] == f"{did} - A" and r["counts_source"] == "layers['counts']"
    assert f"run_pipeline(dataset_id='{new}'" in r["next"] and "'clustering'" in r["next"]
    meta = json.loads((Path(dataset_manager.data_dir) / "uploads" / new / "metadata.json").read_text())
    assert meta["subset_of"]["id"] == did

    flt = {"op": "and", "items": [{"kind": "category", "column": "stim", "values": ["STIM"]},
                                  {"kind": "gene", "gene": "G15", "op": ">", "value": 1}]}
    r = await _call("subset_dataset", {"dataset_id": did, "filter": flt, "name": "stim cells"})
    assert r["name"] == "stim cells" and 50 <= r["n_cells"] <= 240
    # a lasso: the mask the cell filter route gives
    import base64

    import numpy as np

    bits = np.zeros(480, dtype=bool)
    bits[::4] = True
    mask = base64.b64encode(np.packbits(bits, bitorder="little").tobytes()).decode()
    m = await _call("subset_dataset", {"dataset_id": did, "mask": mask})
    assert m["n_cells"] == 120 and m["described"] == "a selection of 120 cells"
    got = await _call("get_dataset", {"dataset_id": m["dataset_id"]})
    assert got["source"] == "subset" and got["subset_of"]["id"] == did and got["subset_of"]["n_cells"] == 120
    listed = await _call("list_datasets", {})
    assert {new, r["dataset_id"]} <= {d["id"] for d in (listed["datasets"] if isinstance(listed, dict) else listed)}


async def test_subset_dataset_refusals(client, dataset_manager):
    did = _register(dataset_manager)
    text = await _call("subset_dataset", {"dataset_id": did}, error=True)
    assert "column + values" in text
    text = await _call("subset_dataset", {"dataset_id": did, "mask": "AAAA"}, error=True)
    assert "Select the cells again" in text
    text = await _call("subset_dataset", {"dataset_id": did, "column": "cluster", "values": ["A"],
                                          "filter": {"op": "and", "items": [{"kind": "category", "column": "stim",
                                                                             "values": ["STIM"]}]}}, error=True)
    assert "not several" in text
    text = await _call("subset_dataset", {"dataset_id": did, "column": "cluster", "values": ["A", "B"]}, error=True)
    assert "whole dataset" in text
    text = await _call("subset_dataset", {"dataset_id": did, "column": "clusterz", "values": ["A"]}, error=True)
    assert "clusterz" in text
