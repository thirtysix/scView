"""The Observations crosstab, alone and split by a third column (one bar per
sample, faceted by condition)."""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

API = "/api/v1"


def _register(dm, did: str = "ds1") -> str:
    """6 cells: donors d1/d2, conditions ctrl/stim (d2 has no stim cells), types A/B."""
    d = Path(dm.data_dir) / "uploads" / did
    d.mkdir(parents=True, exist_ok=True)
    (d / "metadata.json").write_text(json.dumps({"id": did, "name": did, "status": "ready"}))
    a = ad.AnnData(np.zeros((6, 1), dtype="float32"))
    a.obs_names = [f"C{i}" for i in range(6)]
    a.obs["donor"] = ["d1", "d1", "d1", "d2", "d2", "d2"]
    a.obs["stim"] = pd.Categorical(
        ["stim", "ctrl", "stim", "ctrl", "ctrl", "ctrl"], categories=["stim", "ctrl", "unused"])
    a.obs["type"] = ["A", "B", "A", "A", "B", "B"]
    a.write_h5ad(d / "data.h5ad")
    return did


async def test_plain_crosstab_is_unchanged(client, dataset_manager):
    did = _register(dataset_manager)
    r = await client.get(f"{API}/datasets/{did}/metadata/crosstab",
                         params={"row": "donor", "col": "type"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body == {"row_values": ["d1", "d2"], "col_values": ["A", "B"],
                    "counts": [[2, 1], [1, 2]]}


async def test_facet_splits_the_counts_in_category_order(client, dataset_manager):
    did = _register(dataset_manager)
    r = await client.get(f"{API}/datasets/{did}/metadata/crosstab",
                         params={"row": "donor", "col": "type", "facet": "stim"})
    assert r.status_code == 200, r.text
    body = r.json()
    # The facet keeps its stored category order and drops the empty category.
    assert body["facet_values"] == ["stim", "ctrl"]
    # Same rows and columns in every facet; a donor without cells there is zeros.
    assert body["facet_counts"] == [
        [[2, 0], [0, 0]],
        [[0, 1], [1, 2]],
    ]
    total = np.sum(body["facet_counts"], axis=0).tolist()
    assert total == body["counts"]


async def test_unknown_facet_is_404(client, dataset_manager):
    did = _register(dataset_manager)
    r = await client.get(f"{API}/datasets/{did}/metadata/crosstab",
                         params={"row": "donor", "col": "type", "facet": "nope"})
    assert r.status_code == 404
