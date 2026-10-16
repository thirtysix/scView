"""Labelling selected cells by hand, and undoing it (core/user_labels.py + /labels)."""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scanpy as sc

from scview.core import provenance
from scview.core.user_labels import (
    UNASSIGNED,
    USER_COLUMNS_KEY,
    LabelError,
    clean_column_name,
    clean_label,
    label_cells,
    restore,
    user_columns,
)

API = "/api/v1"


def _adata() -> ad.AnnData:
    rng = np.random.default_rng(0)
    X = rng.poisson(1.0, (60, 12)).astype("float32")
    X[:20, :4] += 6
    a = ad.AnnData(X)
    a.var_names = [f"G{i}" for i in range(12)]
    a.obs["leiden"] = pd.Categorical(["0"] * 20 + ["1"] * 20 + ["2"] * 20)
    return a


def _register(dm, a: ad.AnnData | None = None) -> str:
    d = Path(dm.data_dir) / "uploads" / "lb"
    d.mkdir(parents=True)
    (d / "metadata.json").write_text(json.dumps({"id": "lb", "name": "lb", "status": "ready"}))
    (a if a is not None else _adata()).write_h5ad(d / "data.h5ad")
    return "lb"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_clean_label_trims_and_rejects_empty_long_and_invisible():
    assert clean_label("  CD4 T  ") == "CD4 T"
    for bad in ("", "   ", "x" * 81, "T​cell", "a\nb", None):
        with pytest.raises(LabelError):
            clean_label(bad)


@pytest.mark.parametrize("name", ["my labels", "Cell_type-2", "a.b"])
def test_clean_column_name_accepts_plain_names(name):
    assert clean_column_name(f" {name} ", ["leiden"]) == name


@pytest.mark.parametrize("name", ["", "2nd", "a/b", "scview_mine", "SCVIEW_x", "x" * 65, "_index"])
def test_clean_column_name_rejects(name):
    with pytest.raises(LabelError):
        clean_column_name(name, ["leiden"])


def test_clean_column_name_refuses_an_existing_column_whatever_its_case():
    with pytest.raises(LabelError) as err:
        clean_column_name("Leiden", ["leiden"])
    assert err.value.status == 409


# ---------------------------------------------------------------------------
# Labelling and undo, on the AnnData
# ---------------------------------------------------------------------------


def test_new_column_starts_unassigned_and_is_listed():
    a = _adata()
    res = label_cells(a, "my labels", "T cells", [0, 1, 2, 2], new_column=True)
    col = a.obs["my labels"]
    assert res["created"] and res["n_cells"] == 3
    assert list(col.cat.categories) == [UNASSIGNED, "T cells"]
    assert (col[:3] == "T cells").all() and (col[3:] == UNASSIGNED).all()
    assert res["previous"] == [{"value": UNASSIGNED, "indices": [0, 1, 2]}]
    assert user_columns(a) == ["my labels"]
    step = provenance.read_provenance(a)["history"][-1]
    assert step["step"] == "label_cells"
    assert step["params"] == {"column": "my labels", "label": "T cells", "n_cells": 3, "new_column": True}


def test_only_user_columns_take_labels():
    a = _adata()
    with pytest.raises(LabelError):
        label_cells(a, "leiden", "x", [0], new_column=False)
    with pytest.raises(LabelError) as err:
        label_cells(a, "nope", "x", [0], new_column=False)
    assert err.value.status == 404
    assert list(a.obs.columns) == ["leiden"]


@pytest.mark.parametrize("indices", [[], [-1], [60]])
def test_bad_selection_changes_nothing(indices):
    a = _adata()
    with pytest.raises(LabelError):
        label_cells(a, "mine", "x", indices, new_column=True)
    assert "mine" not in a.obs.columns and USER_COLUMNS_KEY not in a.uns


def test_undo_restores_previous_values_then_removes_the_new_column():
    a = _adata()
    first = label_cells(a, "mine", "A", range(0, 30), new_column=True)
    second = label_cells(a, "mine", "B", range(20, 40), new_column=False)
    # Grouped by value, in category order.
    assert second["previous"] == [
        {"value": UNASSIGNED, "indices": list(range(30, 40))},
        {"value": "A", "indices": list(range(20, 30))},
    ]
    restore(a, "mine", second["previous"], second["created"])
    col = a.obs["mine"]
    assert (col[:30] == "A").all() and (col[30:] == UNASSIGNED).all()
    assert list(col.cat.categories) == [UNASSIGNED, "A"]
    res = restore(a, "mine", first["previous"], first["created"])
    assert res["removed"] and "mine" not in a.obs.columns and user_columns(a) == []


def test_undoing_creation_keeps_a_column_that_still_has_labels():
    a = _adata()
    first = label_cells(a, "mine", "A", [0, 1], new_column=True)
    label_cells(a, "mine", "B", [5], new_column=False)
    res = restore(a, "mine", first["previous"], first["created"])
    assert not res["removed"]
    assert list(a.obs["mine"].cat.categories) == [UNASSIGNED, "B"]


def test_relabelling_every_cell_drops_the_unused_category():
    a = _adata()
    label_cells(a, "mine", "A", range(60), new_column=True)
    assert list(a.obs["mine"].cat.categories) == ["A"]


def test_labelling_drops_the_columns_markers_and_enrichment():
    a = _adata()
    label_cells(a, "mine", "A", range(20), new_column=True)
    sc.tl.rank_genes_groups(a, "mine", method="wilcoxon", key_added="rank_genes_groups__mine")
    a.uns["enrichment__mine__A"] = "[]"
    a.uns["ora_meta__mine__A"] = "{}"
    a.uns["enrichment__leiden__0"] = "[]"
    label_cells(a, "mine", "B", range(20, 30), new_column=False)
    assert "rank_genes_groups__mine" not in a.uns
    assert "enrichment__mine__A" not in a.uns and "ora_meta__mine__A" not in a.uns
    assert "enrichment__leiden__0" in a.uns


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


async def test_label_endpoint_saves_reloads_and_undo_round_trips(client, dataset_manager):
    did = _register(dataset_manager)
    r = await client.post(f"{API}/datasets/{did}/labels", json={
        "column": "my labels", "label": "Doublets?", "indices": [3, 4, 5], "new_column": True,
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["column"] == "my labels" and body["created"] is True

    # Saved to the derived file, never the upload; the new column is listed.
    derived = Path(dataset_manager.data_dir) / "derived" / did / "data.h5ad"
    on_disk = ad.read_h5ad(derived)
    assert list(on_disk.obs["my labels"][3:6]) == ["Doublets?"] * 3
    upload = ad.read_h5ad(Path(dataset_manager.data_dir) / "uploads" / did / "data.h5ad")
    assert "my labels" not in upload.obs.columns
    info = (await client.get(f"{API}/datasets/{did}")).json()
    assert "my labels" in [c["name"] for c in info["obs_columns"]]
    assert info["user_annotation_columns"] == ["my labels"]
    cols = (await client.get(f"{API}/datasets/{did}/labels/columns")).json()
    assert cols["columns"] == ["my labels"]

    r = await client.post(f"{API}/datasets/{did}/labels/undo", json={
        "column": "my labels", "previous": body["previous"], "created": True,
    })
    assert r.status_code == 200, r.text
    assert r.json()["removed"] is True
    assert "my labels" not in ad.read_h5ad(derived).obs.columns
    assert (await client.get(f"{API}/datasets/{did}")).json()["user_annotation_columns"] == []
    steps = [h["step"] for h in provenance.read_provenance(ad.read_h5ad(derived))["history"]]
    assert steps[-2:] == ["label_cells", "undo_label_cells"]


async def test_dataset_info_lists_the_hand_label_columns(client, dataset_manager):
    # A hand-label column whatever its name, and none that has since been deleted.
    a = _adata()
    a.obs["curated"] = pd.Categorical(["x"] * 30 + ["y"] * 30)
    a.uns[USER_COLUMNS_KEY] = json.dumps(["curated", "gone"])
    did = _register(dataset_manager, a)
    info = (await client.get(f"{API}/datasets/{did}")).json()
    assert info["user_annotation_columns"] == ["curated"]
    # Kept in the sidecar, so a dataset a run is working on still says so.
    dataset_manager.mark_busy(did)
    try:
        busy = (await client.get(f"{API}/datasets/{did}")).json()
    finally:
        dataset_manager.mark_idle(did)
    assert busy["user_annotation_columns"] == ["curated"]


async def test_label_endpoint_rejects_bad_requests(client, dataset_manager):
    did = _register(dataset_manager)
    url = f"{API}/datasets/{did}/labels"
    r = await client.post(url, json={"column": "leiden", "label": "x", "indices": [0]})
    assert r.status_code == 400
    r = await client.post(url, json={"column": "leiden", "label": "x", "indices": [0], "new_column": True})
    assert r.status_code == 409
    r = await client.post(url, json={"column": "mine", "label": " ", "indices": [0], "new_column": True})
    assert r.status_code == 400
    r = await client.post(f"{API}/datasets/missing/labels", json={"column": "m", "label": "x", "indices": [0]})
    assert r.status_code == 404
    assert not (Path(dataset_manager.data_dir) / "derived" / did).exists() or not any(
        (Path(dataset_manager.data_dir) / "derived" / did).iterdir()
    )
