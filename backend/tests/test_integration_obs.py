"""Numeric obs columns in a joint dataset (QA round 3: in_tissue, percent.mt and the authors'
cell2location shares vanished in the merge, and the plan did not say so).

Numbers every dataset has are carried (ints stay ints when none is missing); per-cell QC
metrics the joint run computes again, ids and numbers only some datasets have are left out,
and the plan lists every column left out with the reason."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from scview.core.integration import (
    WHY_CLUSTERS,
    WHY_ID,
    WHY_RECOMPUTED,
    WHY_SOME,
    build_plan,
    build_unit,
    carried_obs,
    merge_joint,
)
from tests.test_integration import _add_dataset, _counts_dataset


def _with_numbers(seed: int, extra: str | None = None):
    a = _counts_dataset(seed=seed)
    n = a.n_obs
    rng = np.random.default_rng(seed)
    a.obs["in_tissue"] = np.r_[np.zeros(10, np.int64), np.ones(n - 10, np.int64)]
    a.obs["percent.mt"] = rng.uniform(0, 5, n)
    a.obs["Cardiomyocyte"] = rng.uniform(0, 1, n).astype(np.float32)
    a.obs["total_counts"] = rng.uniform(1000, 5000, n)  # recomputed by the joint QC
    a.obs["pct_counts_mt"] = rng.uniform(0, 5, n)
    a.obs["spot_index"] = np.arange(n)  # one value per cell
    if extra:
        a.obs[extra] = rng.uniform(0, 1, n)
    return a


def _unit(a, label):
    return build_unit(a, dataset_id=label.lower(), name=label, label=label, fallback_paths=[])


def test_numbers_are_carried_and_qc_metrics_and_ids_are_not():
    a = _with_numbers(1)
    a.obs["n_mapped"] = pd.array([1, None] + [3] * (a.n_obs - 2), dtype="Int64")  # a missing value
    obs = carried_obs(a, "A")
    assert obs["in_tissue"].dtype == np.int64
    assert obs["percent.mt"].dtype == np.float64 and obs["Cardiomyocyte"].dtype == np.float32
    assert obs["n_mapped"].dtype == np.float64 and np.isnan(obs["n_mapped"].iloc[1])
    assert not {"total_counts", "pct_counts_mt", "spot_index", "barcode"} & set(obs.columns)
    assert _unit(a, "A").dropped == {"barcode": WHY_ID, "total_counts": WHY_RECOMPUTED,
                                     "pct_counts_mt": WHY_RECOMPUTED, "spot_index": WHY_ID}


def test_the_merge_keeps_shared_numbers_and_the_plan_lists_what_it_leaves_out():
    a = _with_numbers(1, extra="Fibroblast")
    a.obs["leiden"] = pd.Categorical(["0", "1"] * (a.n_obs // 2))
    units = {"A": _unit(a, "A"), "B": _unit(_with_numbers(2), "B")}
    plan = build_plan([], units, max_datasets=8, max_cells=10**6)
    dropped = {d.column: (d.reason, d.datasets) for d in plan.dropped_columns}
    assert dropped["Fibroblast"] == (WHY_SOME, ["A"])
    assert dropped["total_counts"] == (WHY_RECOMPUTED, ["A", "B"])
    assert dropped["spot_index"] == (WHY_ID, ["A", "B"]) and dropped["barcode"][0] == WHY_ID
    assert dropped["leiden"] == (WHY_CLUSTERS, ["A"])
    assert plan.dropped_columns[0].column == "Fibroblast"  # the ones only some have, first
    assert {"in_tissue", "percent.mt", "Cardiomyocyte"} <= set(plan.shared_columns)

    joint = merge_joint(units)
    assert joint.obs["in_tissue"].dtype == np.int64
    assert joint.obs["in_tissue"].sum() == 2 * (a.n_obs - 10)
    assert pd.api.types.is_float_dtype(joint.obs["percent.mt"])
    assert pd.api.types.is_float_dtype(joint.obs["Cardiomyocyte"])
    assert "Fibroblast" not in joint.obs.columns
    assert isinstance(joint.obs["cell_type"].dtype, pd.CategoricalDtype)


@pytest.mark.asyncio
async def test_a_joint_dataset_holds_the_numbers_and_colours_by_them(client, tmp_data_dir):
    _add_dataset(tmp_data_dir, "a", "A", _with_numbers(1, extra="Fibroblast"))
    _add_dataset(tmp_data_dir, "b", "B", _with_numbers(2))
    r = await client.post("/api/v1/integrate/plan", json={"dataset_ids": ["a", "b"]})
    plan = r.json()
    assert {"column": "Fibroblast", "reason": WHY_SOME, "datasets": ["A"]} in plan["dropped_columns"]
    r = await client.post("/api/v1/integrate/run-stream", json={
        "dataset_ids": ["a", "b"], "name": "AB", "params": {"n_top_genes": 100, "n_comps": 20}})
    last = r.text.strip().splitlines()[-1]
    assert last.startswith("data: ") and "dataset_id" in last, r.text[-400:]
    new_id = json.loads(last[6:])["dataset_id"]
    info = (await client.get(f"/api/v1/datasets/{new_id}")).json()
    cols = {c["name"]: c for c in info["obs_columns"]}
    assert {"in_tissue", "percent.mt", "Cardiomyocyte"} <= set(cols)
    assert "Fibroblast" not in cols
    # A continuous column colours the plot like any dataset's.
    assert cols["in_tissue"]["dtype"].startswith("int") and cols["percent.mt"]["dtype"].startswith("float")
    r = await client.get(f"/api/v1/datasets/{new_id}/embeddings/X_umap", params={"color_by": "percent.mt"})
    assert r.status_code == 200, r.text
