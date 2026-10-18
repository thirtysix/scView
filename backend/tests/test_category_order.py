"""Setting a label column's category order (N05): explicit, natural or by size."""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from scview.core import category_order as co

API = "/api/v1"


def _adata() -> ad.AnnData:
    labels = ["10"] * 5 + ["2"] * 30 + ["1"] * 10 + [None] * 2
    a = ad.AnnData(X=np.zeros((len(labels), 3), dtype=np.float32))
    a.obs_names = [f"c{i}" for i in range(len(labels))]
    a.obs["cl"] = pd.Categorical(labels, categories=["10", "2", "1", "empty"])
    a.obs["donor"] = np.array(["d2", "d10", "d1"] * 15 + [None, "d1"], dtype=object)
    a.obs["score"] = np.arange(len(labels), dtype=float)
    a.obs["flag"] = a.obs["score"] > 5
    a.uns["cl_colors"] = np.array(["#aa0000", "#00bb00", "#0000cc", "#dddddd"], dtype=object)
    return a


def test_natural_key_orders_numbers_by_value():
    assert sorted(["T 10", "t 2", "B", "1", "10", "2"], key=co.natural_key) == [
        "1", "2", "10", "B", "t 2", "T 10"]


def test_explicit_order_keeps_every_cell_and_moves_its_colour():
    a = _adata()
    before = a.obs["cl"].astype(str).copy()
    assert co.reorder(a, "cl", order=["1", "2", "10", "empty"]) == ["1", "2", "10", "empty"]
    assert list(a.obs["cl"].cat.categories) == ["1", "2", "10", "empty"]
    assert (a.obs["cl"].astype(str) == before).all()  # labels unchanged, NaN kept
    assert a.obs["cl"].isna().sum() == 2
    assert list(a.uns["cl_colors"]) == ["#0000cc", "#00bb00", "#aa0000", "#dddddd"]


def test_quick_sorts():
    a = _adata()
    assert co.reorder(a, "cl", sort="natural") == ["1", "2", "10", "empty"]
    # Most cells first; the empty category last.
    assert co.reorder(a, "cl", sort="size") == ["2", "1", "10", "empty"]


def test_a_text_column_becomes_a_categorical_in_that_order():
    a = _adata()
    assert co.reorder(a, "donor", sort="natural") == ["d1", "d2", "d10"]
    col = a.obs["donor"]
    assert isinstance(col.dtype, pd.CategoricalDtype) and list(col.cat.categories) == ["d1", "d2", "d10"]
    assert col.isna().sum() == 1 and (col == "d1").sum() == 16


@pytest.mark.parametrize("kwargs, status, text", [
    ({"order": ["1", "2", "10"]}, 400, "missing empty"),
    ({"order": ["1", "2", "10", "empty", "x"]}, 400, "not in the column: x"),
    ({"order": ["1", "1", "2", "10"]}, 400, "missing empty"),
    ({}, 400, "Give either an order or a sort"),
])
def test_bad_orders_change_nothing(kwargs, status, text):
    a = _adata()
    with pytest.raises(co.OrderError, match=text) as err:
        co.reorder(a, "cl", **kwargs)
    assert err.value.status == status
    assert list(a.obs["cl"].cat.categories) == ["10", "2", "1", "empty"]


def test_columns_without_an_order_are_refused():
    a = _adata()
    for col in ("score", "flag"):
        with pytest.raises(co.OrderError, match="categorical or text"):
            co.reorder(a, col, sort="natural")
    with pytest.raises(co.OrderError) as err:
        co.reorder(a, "nope", sort="natural")
    assert err.value.status == 404


def _register(dm) -> str:
    d = Path(dm.data_dir) / "uploads" / "od"
    d.mkdir(parents=True)
    (d / "metadata.json").write_text(json.dumps({"id": "od", "name": "od", "status": "ready"}))
    _adata().write_h5ad(d / "data.h5ad")
    return "od"


async def test_endpoint_saves_the_order_for_every_view(client, dataset_manager):
    did = _register(dataset_manager)
    r = await client.post(f"{API}/datasets/{did}/metadata/cl/order", json={"sort": "size"})
    assert r.status_code == 200, r.text
    assert r.json() == {"column": "cl", "order": ["2", "1", "10", "empty"]}
    info = (await client.get(f"{API}/datasets/{did}")).json()
    assert next(c for c in info["obs_columns"] if c["name"] == "cl")["values"] == ["2", "1", "10", "empty"]
    ct = (await client.get(f"{API}/datasets/{did}/metadata/crosstab",
                           params={"row": "cl", "col": "donor"})).json()
    assert ct["row_values"][:3] == ["2", "1", "10"]
    saved = ad.read_h5ad(dataset_manager.derived_h5ad_path(did, "data.h5ad"))
    assert list(saved.obs["cl"].cat.categories) == ["2", "1", "10", "empty"]
    assert list(saved.uns["cl_colors"]) == ["#00bb00", "#0000cc", "#aa0000", "#dddddd"]

    r = await client.post(f"{API}/datasets/{did}/metadata/cl/order",
                          json={"order": ["empty", "10", "2", "1"]})
    assert r.status_code == 200, r.text
    dataset_manager.evict(did)  # read back from disk
    info = (await client.get(f"{API}/datasets/{did}")).json()
    assert next(c for c in info["obs_columns"] if c["name"] == "cl")["values"] == ["empty", "10", "2", "1"]


async def test_endpoint_errors(client, dataset_manager):
    did = _register(dataset_manager)
    url = f"{API}/datasets/{did}/metadata"
    assert (await client.post(f"{url}/cl/order", json={"order": ["1"]})).status_code == 400
    assert (await client.post(f"{url}/score/order", json={"sort": "natural"})).status_code == 400
    assert (await client.post(f"{url}/nope/order", json={"sort": "natural"})).status_code == 404
    assert (await client.post(f"{url}/cl/order", json={"sort": "zigzag"})).status_code == 422
    assert (await client.post(f"{API}/datasets/missing/metadata/cl/order",
                              json={"sort": "natural"})).status_code == 404
