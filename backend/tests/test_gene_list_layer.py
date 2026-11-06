"""``GET /datasets/{id}/genes?layer=``: the genes ``GET /expression`` finds in a layer, so
a client checks a name first instead of asking for it and getting a 404."""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

API = "/api/v1"


def _register(dm) -> str:
    full = ad.AnnData(X=np.ones((5, 4), dtype=np.float32),
                      obs=pd.DataFrame(index=[f"c{i}" for i in range(5)]),
                      var=pd.DataFrame(index=["CD3E", "CD19", "MS4A1", "XIST"]))
    a = full[:, ["CD3E", "CD19"]].copy()  # X holds the variable genes only
    a.raw = full
    a.layers["counts"] = a.X.copy()
    d = Path(dm.data_dir) / "uploads" / "gl"
    d.mkdir(parents=True)
    (d / "metadata.json").write_text(json.dumps({"id": "gl", "name": "gl", "status": "ready"}))
    a.write_h5ad(d / "data.h5ad")
    return "gl"


async def test_genes_by_layer_match_what_expression_finds(client, dataset_manager):
    did = _register(dataset_manager)
    url = f"{API}/datasets/{did}/genes"
    assert (await client.get(url)).json()["genes"] == ["CD3E", "CD19", "MS4A1", "XIST"]  # default: .raw
    assert (await client.get(url, params={"layer": "raw"})).json()["genes"] == ["CD3E", "CD19", "MS4A1", "XIST"]
    x = (await client.get(url, params={"layer": "X"})).json()
    assert x == {"genes": ["CD3E", "CD19"], "layer": "X"}
    assert (await client.get(url, params={"layer": "counts"})).json()["genes"] == ["CD3E", "CD19"]
    assert (await client.get(url, params={"layer": "nope"})).status_code == 400
    # A gene listed for X is found there; one that is not gives the 404 a client now avoids.
    ok = await client.get(f"{API}/datasets/{did}/expression", params={"genes": "CD19", "layer": "X"})
    assert ok.status_code == 200
    miss = await client.get(f"{API}/datasets/{did}/expression", params={"genes": "MS4A1", "layer": "X"})
    assert miss.status_code == 404
