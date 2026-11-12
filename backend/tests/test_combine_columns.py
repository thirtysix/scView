"""A new annotation column from two to four columns in one call (core/user_labels.py
combine_columns, POST /labels/combine, MCP combine_columns). Emont 2022 took 24
set_labels calls to build a mouse x depot sample column."""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from scview.core import provenance
from scview.core.user_labels import LabelError, combine_columns, label_cells, user_columns
from tests.test_mcp import _call, _register as _register_mcp, msigdb_dir, test_settings  # noqa: F401

API = "/api/v1"


def _adata() -> ad.AnnData:
    rng = np.random.default_rng(0)
    a = ad.AnnData(rng.poisson(1.0, (60, 12)).astype("float32"))
    a.var_names = [f"G{i}" for i in range(12)]
    a.obs["leiden"] = pd.Categorical(["0"] * 20 + ["1"] * 20 + ["2"] * 20)
    a.obs["mouse"] = pd.Categorical(["M2", "M1", "M3"] * 20, categories=["M1", "M2", "M3"])
    a.obs["depot"] = ["ING", "PG"] * 30
    return a


def _register(dm, a: ad.AnnData) -> str:
    d = Path(dm.data_dir) / "uploads" / "cb"
    d.mkdir(parents=True)
    (d / "metadata.json").write_text(json.dumps({"id": "cb", "name": "cb", "status": "ready"}))
    a.write_h5ad(d / "data.h5ad")
    return "cb"


def test_values_are_joined_in_column_order():
    a = _adata()
    a.obs.loc[a.obs_names[5], "depot"] = None
    out = combine_columns(a, ["mouse", "depot"], " sample ")
    assert out["column"] == "sample" and out["n_labels"] == 6
    assert out["n_missing"] == 1 and out["n_cells"] == 59
    assert a.obs["sample"].iloc[0] == "M2 | ING" and a.obs["sample"].iloc[1] == "M1 | PG"
    assert pd.isna(a.obs["sample"].iloc[5])
    # Categories follow the first column's order, then the second's.
    assert list(a.obs["sample"].cat.categories) == [
        "M1 | ING", "M1 | PG", "M2 | ING", "M2 | PG", "M3 | ING", "M3 | PG"]
    assert user_columns(a) == ["sample"]
    last = provenance.read_provenance(a)["history"][-1]
    assert last["step"] == "combine_columns" and last["params"]["columns"] == ["mouse", "depot"]
    # A combined column takes hand labels afterwards.
    label_cells(a, "sample", "fixed", [0], new_column=False)
    assert a.obs["sample"].iloc[0] == "fixed"


def test_another_separator():
    a = _adata()
    combine_columns(a, ["depot", "mouse", "leiden"], "s", separator="_")
    assert a.obs["s"].iloc[0] == "ING_M2_0"


@pytest.mark.parametrize("columns, name, sep, status", [
    (["mouse"], "s", " | ", 400),
    (["mouse", "nope"], "s", " | ", 404),
    (["mouse", "depot"], "leiden", " | ", 409),
    (["mouse", "depot"], "s", "", 400),
    (["mouse", "depot"], "s", "\n", 400),
    (["mouse", "depot", "leiden", "mouse2", "x"], "s", " | ", 400),
])
def test_bad_requests_change_nothing(columns, name, sep, status):
    a = _adata()
    with pytest.raises(LabelError) as err:
        combine_columns(a, columns, name, sep)
    assert err.value.status == status
    assert "s" not in a.obs.columns


def test_a_per_cell_combination_is_refused(monkeypatch):
    from scview.core import user_labels

    monkeypatch.setattr(user_labels, "MAX_LABELS", 5)
    with pytest.raises(LabelError, match="per-cell"):
        user_labels.combine_columns(_adata(), ["mouse", "depot"], "s")


def test_the_step_replays_exactly():
    from scview.core.replay_script import EXACT, _combine_columns

    snip = _combine_columns({"column": "sample", "columns": ["mouse", "depot"], "separator": " | "}, None)
    assert snip.status == EXACT
    a = _adata()
    a.obs.loc[a.obs_names[5], "depot"] = None
    exec("\n".join(snip.code), {"adata": a})
    assert a.obs["sample"].iloc[0] == "M2 | ING" and pd.isna(a.obs["sample"].iloc[5])


async def test_endpoint_saves_the_column(client, dataset_manager):
    did = _register(dataset_manager, _adata())
    r = await client.post(f"{API}/datasets/{did}/labels/combine",
                          json={"columns": ["mouse", "depot"], "name": "sample"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n_labels"] == 6 and body["values"][0]["n_cells"] == 10
    derived = Path(dataset_manager.data_dir) / "derived" / did / "data.h5ad"
    assert ad.read_h5ad(derived).obs["sample"].iloc[0] == "M2 | ING"
    info = (await client.get(f"{API}/datasets/{did}")).json()
    assert info["user_annotation_columns"] == ["sample"]
    r = await client.post(f"{API}/datasets/{did}/labels/combine",
                          json={"columns": ["mouse", "depot"], "name": "sample"})
    assert r.status_code == 409


async def test_mcp_tool_makes_a_sample_column_in_one_call(client, dataset_manager):
    did = _register_mcp(dataset_manager)
    r = await _call("combine_columns", {"dataset_id": did, "columns": ["donor_id", "stim"], "name": "sample"})
    assert r["column"] == "sample" and r["n_labels"] == 8 and r["n_missing"] == 0
    r = await _call("describe_column", {"dataset_id": did, "column": "sample"})
    assert dict(r["counts"])["D0 | CTRL"] == 60
    text = await _call("combine_columns", {"dataset_id": did, "columns": ["donor_id", "nope"],
                                           "name": "s2"}, error=True)
    assert "nope" in text
