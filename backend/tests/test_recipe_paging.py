"""GET /recipe and MCP get_recipe page a long recipe: QA round 7's 210 steps (196 hand
labellings first) showed the first 26 with no way to read the pipeline run."""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np

from scview.core import provenance

API = "/api/v1"


def _register(dm) -> str:
    a = ad.AnnData(np.ones((10, 3), dtype="float32"))
    for i in range(150):
        provenance.record_step(a, step="label_cells", tool="scview", params={"column": "c", "label": f"L{i}"})
    for name in ("qc_metrics", "filtering", "normalization", "pca", "clustering"):
        provenance.record_step(a, step=name, tool="scanpy", params={"x": 1}, resources={"elapsed_s": 2.0})
    d = Path(dm.data_dir) / "uploads" / "rp"
    d.mkdir(parents=True)
    (d / "metadata.json").write_text(json.dumps({"id": "rp", "name": "rp", "status": "ready"}))
    a.write_h5ad(d / "data.h5ad")
    return "rp"


async def test_rest_recipe_filters_pages_and_counts(client, dataset_manager):
    did = _register(dataset_manager)
    whole = (await client.get(f"{API}/datasets/{did}/recipe")).json()
    assert len(whole["recipe"]) == whole["n_total"] == whole["n_matched"] == 155
    assert whole["step_counts"] == {"label_cells": 150, "qc_metrics": 1, "filtering": 1, "normalization": 1,
                                    "pca": 1, "clustering": 1}
    r = (await client.get(f"{API}/datasets/{did}/recipe",
                          params=[("step", "pca"), ("step", "clustering")])).json()
    assert [s["step"] for s in r["recipe"]] == ["pca", "clustering"] and r["n_matched"] == 2
    assert [s["index"] for s in r["recipe"]] == [153, 154]
    r = (await client.get(f"{API}/datasets/{did}/recipe", params={"offset": 148, "limit": 4})).json()
    assert [s["index"] for s in r["recipe"]] == [148, 149, 150, 151] and r["n_total"] == 155


async def test_mcp_get_recipe_pages_with_next_offset(client, dataset_manager):
    from tests.test_label_import import _mcp

    did = _register(dataset_manager)
    r = await _mcp("get_recipe", {"dataset_id": did})
    assert r["n_total"] == 155 and r["step_counts"]["label_cells"] == 150
    assert r["steps"][0]["index"] == 0 and r["next_offset"] == len(r["steps"])
    assert "truncated" not in r
    seen = len(r["steps"])
    while "next_offset" in r:
        r = await _mcp("get_recipe", {"dataset_id": did, "offset": r["next_offset"], "limit": 200})
        assert r["steps"][0]["index"] == seen
        seen += len(r["steps"])
    assert seen == 155 and r["steps"][-1]["step"] == "clustering"
    r = await _mcp("get_recipe", {"dataset_id": did, "steps": ["qc_metrics", "pca"]})
    assert [s["step"] for s in r["steps"]] == ["qc_metrics", "pca"] and r["n_matched"] == 2
    assert r["steps"][1]["measured"] == {"seconds": 2.0}
