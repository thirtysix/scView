"""GET /markers on an unprocessed dataset: 200 with an empty table and a reason.

It used to 404 ("Run sc.tl.rank_genes_groups first"), so opening any fresh upload
logged a failed request. 404 stays for a dataset or a column that does not exist.
"""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pyarrow as pa

from scview.api.v1.markers import NO_MARKERS

API = "/api/v1"


def _register(dm, did: str, *, cluster: bool) -> str:
    d = Path(dm.data_dir) / "uploads" / did
    d.mkdir(parents=True, exist_ok=True)
    (d / "metadata.json").write_text(json.dumps({"id": did, "name": did, "status": "ready"}))
    rng = np.random.default_rng(0)
    X = rng.poisson(0.3, size=(120, 60)).astype("float32")
    X[:60, :15] += rng.poisson(6.0, size=(60, 15)).astype("float32")
    a = ad.AnnData(X)
    a.var_names = [f"G{i}" for i in range(60)]
    a.obs_names = [f"C{i}" for i in range(120)]
    if cluster:
        a.obs["cluster"] = pd.Categorical(["0"] * 60 + ["1"] * 60)
    a.write_h5ad(d / "data.h5ad")
    return did


async def test_no_markers_is_200_with_an_empty_list_and_a_reason(client, dataset_manager):
    did = _register(dataset_manager, "raw1", cluster=False)
    r = await client.get(f"{API}/datasets/{did}/markers?format=json")
    assert r.status_code == 200, r.text
    assert r.json() == {"groups": [], "markers": {}, "reason": NO_MARKERS}
    assert NO_MARKERS == "No marker genes yet: run the pipeline or pick a column."


async def test_the_arrow_form_is_an_empty_table_with_the_reason_in_a_header(client, dataset_manager):
    did = _register(dataset_manager, "raw2", cluster=False)
    r = await client.get(f"{API}/datasets/{did}/markers")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/vnd.apache.arrow.stream")
    assert r.headers["x-scview-reason"] == NO_MARKERS
    table = pa.ipc.open_stream(r.content).read_all()
    assert table.num_rows == 0 and {"group", "gene", "pval_adj"} <= set(table.column_names)


async def test_a_group_with_no_markers_is_200_empty_and_a_found_one_unchanged(client, dataset_manager):
    did = _register(dataset_manager, "raw3", cluster=True)
    url = f"{API}/datasets/{did}/markers?format=json&groupby_column=cluster"
    full = (await client.get(url)).json()
    assert full["groups"] == ["0", "1"] and "reason" not in full  # the populated shape as before
    r = await client.get(url + "&groupby=nope")
    assert r.status_code == 200
    assert r.json() == {"groups": [], "markers": {}, "reason": "No marker genes for group 'nope'."}
    # the bare request reads the default markers only, which on-demand ones are not: empty, 200
    bare = await client.get(f"{API}/datasets/{did}/markers?format=json")
    assert bare.status_code == 200 and bare.json()["reason"] == NO_MARKERS


async def test_an_unknown_column_or_dataset_is_still_404(client, dataset_manager):
    did = _register(dataset_manager, "raw4", cluster=False)
    r = await client.get(f"{API}/datasets/{did}/markers?format=json&groupby_column=nope")
    assert r.status_code == 404 and "not found" in r.json()["detail"]
    r = await client.get(f"{API}/datasets/ghost/markers?format=json")
    assert r.status_code == 404 and r.json()["detail"] == "Dataset not found."
