"""Regression: on-demand markers on a non-categorical (e.g. boolean) obs column.

Selecting `predicted_doublet` (a bool column) in the Markers tab used to 500 with
"Can only use .cat accessor with a 'category' dtype" because rank_genes_groups
needs a categorical groupby. The endpoint now coerces the column first.
"""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

API = "/api/v1"


def _register_bool(dm, did: str = "dsb", n: int = 200, g: int = 120) -> str:
    d = Path(dm.data_dir) / "uploads" / did
    d.mkdir(parents=True, exist_ok=True)
    (d / "metadata.json").write_text(json.dumps({"id": did, "name": did, "status": "ready"}))
    rng = np.random.default_rng(0)
    X = rng.poisson(0.4, size=(n, g)).astype("float32")
    # make a block of genes separate the two boolean groups
    flag = np.zeros(n, dtype=bool)
    flag[: n // 4] = True
    X[flag, :15] += rng.poisson(6.0, size=(flag.sum(), 15)).astype("float32")
    a = ad.AnnData(X)
    a.var_names = [f"G{i}" for i in range(g)]
    a.obs_names = [f"C{i}" for i in range(n)]
    a.obs["predicted_doublet"] = flag                 # bool (not categorical)
    a.obs["n_genes"] = (X > 0).sum(1).astype(int)     # numeric column too
    a.write_h5ad(d / "data.h5ad")
    return did


async def test_markers_on_boolean_column(client, dataset_manager):
    did = _register_bool(dataset_manager)
    r = await client.get(
        f"{API}/datasets/{did}/markers?groupby_column=predicted_doublet&format=json&n_genes=5"
    )
    assert r.status_code == 200, r.text
    body = r.text
    # both boolean groups present (coerced to 'True'/'False')
    assert "True" in body and "False" in body
    assert "gene" in body and "group" in body


async def test_markers_single_group_column_400(client, dataset_manager):
    """A column with only one value can't yield markers — friendly 400, not 500."""
    did = _register_bool(dataset_manager, did="dsb1")
    # overwrite predicted_doublet with a single value via a fresh dataset
    d = Path(dataset_manager.data_dir) / "uploads" / "dsone"
    d.mkdir(parents=True, exist_ok=True)
    (d / "metadata.json").write_text(json.dumps({"id": "dsone", "name": "dsone", "status": "ready"}))
    a = ad.read_h5ad(Path(dataset_manager.data_dir) / "uploads" / did / "data.h5ad")
    a.obs["all_false"] = pd.Series(False, index=a.obs_names)
    a.write_h5ad(d / "data.h5ad")
    r = await client.get(f"{API}/datasets/dsone/markers?groupby_column=all_false&format=json")
    assert r.status_code == 400, r.text
