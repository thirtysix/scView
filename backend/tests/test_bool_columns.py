"""Bool obs columns (e.g. predicted_doublet) behave as two-level categoricals."""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pyarrow as pa

API = "/api/v1"


def _register(dm) -> str:
    d = Path(dm.data_dir) / "uploads" / "bl"
    d.mkdir(parents=True)
    (d / "metadata.json").write_text(json.dumps({"id": "bl", "name": "bl", "status": "ready"}))
    a = ad.AnnData(np.ones((4, 3), dtype="float32"))
    a.obs["predicted_doublet"] = np.array([True, False, False, True])  # first cell True
    a.obsm["X_umap"] = np.zeros((4, 2), dtype="float32")
    a.write_h5ad(d / "data.h5ad")
    return "bl"


async def test_bool_column_has_values_and_fixed_codes(client, dataset_manager):
    did = _register(dataset_manager)
    info = (await client.get(f"{API}/datasets/{did}")).json()
    col = next(c for c in info["obs_columns"] if c["name"] == "predicted_doublet")
    assert col["values"] == ["False", "True"]
    r = await client.get(f"{API}/datasets/{did}/embeddings/X_umap", params={"color_by": "predicted_doublet"})
    assert r.status_code == 200, r.text
    table = pa.ipc.open_stream(r.content).read_all()
    codes = table.column(table.num_columns - 1).to_pylist()
    assert codes == [1, 0, 0, 1]  # True is 1 whatever the first cell is
