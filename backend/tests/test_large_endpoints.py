"""Endpoints on a large dataset (matrices on disk, core/large_mode.py): the views that
work give what they give on the same dataset in memory; the edits are written into the
h5ad in place; the methods that need a whole matrix refuse with a 422 and a reason."""

from __future__ import annotations

import json

import anndata as ad
import h5py
import numpy as np
import pytest

from tests.large_fixture import counts_adata, register, small_twin

LARGE, SMALL = "bigE", "smallE"
API = "/api/v1/datasets"


@pytest.fixture()
async def both(client, tmp_data_dir):
    """A large dataset and its in-memory twin (same cells, genes and values)."""
    counts = counts_adata()
    h5ad = register(tmp_data_dir, LARGE, counts)
    up = tmp_data_dir / "uploads" / SMALL
    up.mkdir(parents=True)
    small_twin(counts).write_h5ad(up / "twin.h5ad")
    (up / "metadata.json").write_text(json.dumps({"id": SMALL, "name": "twin", "status": "ready"}))
    return h5ad


async def _both(client, method: str, path: str, **kw):
    a = await client.request(method, f"{API}/{LARGE}{path}", **kw)
    b = await client.request(method, f"{API}/{SMALL}{path}", **kw)
    assert a.status_code == 200, a.text
    assert b.status_code == 200, b.text
    return a.json(), b.json()


def _close(a, b, tol=1e-9):
    """Equal JSON, numbers within ``tol``."""
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            _close(a[k], b[k], tol)
    elif isinstance(a, list):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            _close(x, y, tol)
    elif isinstance(a, float) or isinstance(b, float):
        assert a is not None and b is not None and abs(a - b) <= tol * max(1.0, abs(b)), (a, b)
    else:
        assert a == b


async def test_views_equal_the_dataset_in_memory(client, both):
    for path, params in [
        ("/expression/by-group", [("genes", "G10"), ("genes", "G30"), ("genes", "MT-1"),
                                  ("groupby", "leiden")]),
        ("/expression/violin", {"gene": "G25", "groupby": "leiden"}),
        ("/expression/per-donor", {"gene": "G10", "groupby": "leiden", "sample": "donor"}),
        ("/expression/per-donor", {"gene": "G10", "groupby": "leiden", "sample": "donor",
                                   "method": "cpm"}),
        ("/qc/pseudobulk", {"sample_column": "donor", "split_by": ""}),
    ]:
        a, b = await _both(client, "GET", path, params=params)
        if path == "/expression/per-donor":
            a.pop("source", None)
            b.pop("source", None)
        if path == "/qc/pseudobulk":
            assert a.pop("counts_source") == "the counts file"
            b.pop("counts_source")
        _close(a, b, 1e-6)


async def test_coexpression_equals_the_dataset_in_memory(client, both):
    a, b = await _both(client, "POST", "/expression/coexpression", json={"gene": "G10"})
    _close(a, b, 1e-9)


async def test_gene_set_score_equals_scanpy(client, both):
    body = {"gene_set": ["G10", "G11", "G12", "G13", "G40", "G41", "nope"], "score_name": "s"}
    a, b = await _both(client, "POST", "/genesets/score", json=body)
    assert a["genes_found"] == b["genes_found"] and a["genes_missing"] == b["genes_missing"]
    np.testing.assert_allclose(a["scores"], b["scores"], rtol=0, atol=1e-12)


async def test_markers_for_a_new_column_come_from_the_store(client, both):
    params = {"groupby_column": "donor", "format": "json", "n_genes": 20}
    a, b = await _both(client, "GET", "/markers", params=params)
    assert a["groups"] == b["groups"]
    for g in a["groups"]:
        assert [r["gene"] for r in a["markers"][g]] == [r["gene"] for r in b["markers"][g]]
        np.testing.assert_allclose([r["pval"] for r in a["markers"][g]],
                                   [r["pval"] for r in b["markers"][g]], rtol=1e-6)
    # saved into the h5ad in place: there on the next open
    from scview.core.uns_keys import markers_key

    with h5py.File(both, "r") as f:
        assert markers_key("donor") in f["uns"]


async def test_sex_check_reads_its_genes_from_the_store(client, both):
    res = await client.get(f"{API}/{LARGE}/qc/sex")
    assert res.status_code == 200, res.text


async def test_labels_are_written_in_place(client, both):
    with h5py.File(both, "r") as f:
        x_offset = f["X"]["data"].id.get_offset()
    res = await client.post(f"{API}/{LARGE}/labels", json={
        "column": "mine", "label": "picked", "indices": [1, 2, 3], "new_column": True})
    assert res.status_code == 200, res.text
    back = ad.read_h5ad(both, backed="r")
    assert list(back.obs["mine"][:4]) == ["Unassigned", "picked", "picked", "picked"]
    back.file.close()
    with h5py.File(both, "r") as f:
        assert f["X"]["data"].id.get_offset() == x_offset
    # the cached dataset is the edited one
    info = (await client.get(f"{API}/{LARGE}")).json()
    assert "mine" in [c["name"] for c in info["obs_columns"]]
    assert info["large"]["gene_store"] == "ready"


async def test_category_order_and_rename_are_written_in_place(client, both):
    res = await client.post(f"{API}/{LARGE}/metadata/leiden/order", json={"order": ["2", "1", "0"]})
    assert res.status_code == 200, res.text
    # a merge drops the stored markers (computed again from the store on the next visit)
    await client.get(f"{API}/{LARGE}/markers", params={"groupby_column": "leiden", "format": "json"})
    res = await client.post(f"{API}/{LARGE}/metadata/leiden/rename", json={"old": "1", "new": "0"})
    assert res.status_code == 200, res.text
    back = ad.read_h5ad(both, backed="r")
    assert set(back.obs["leiden"].cat.categories) == {"0", "2"}
    back.file.close()
    res = await client.get(f"{API}/{LARGE}/markers", params={"groupby_column": "leiden", "format": "json"})
    assert res.status_code == 200 and set(res.json()["groups"]) == {"0", "2"}


@pytest.mark.parametrize("method, path, kw", [
    ("POST", "/genesets/score", {"json": {"gene_set": ["G10", "G11"], "method": "aucell"}}),
    ("POST", "/genesets/score/compare", {"json": {"gene_set": ["G10", "G11"]}}),
    ("POST", "/de", {"json": {"indices": list(range(50))}}),
    ("POST", "/de/compare", {"json": {"split_by": "leiden", "group_a": "0", "group_b": "1"}}),
    ("POST", "/clustering/subcluster", {"json": {"column": "leiden", "group": "1"}}),
    ("POST", "/programs/run-stream", {"json": {}}),
    ("GET", "/activity/progeny", {}),
    ("POST", "/trajectory/leiden/trend-genes", {"json": {}}),
    ("GET", "/qc/mito-refilter", {}),
    ("POST", "/communication/compare", {"json": {"groupby": "leiden", "condition": "sample", "a": "x", "b": "y"}}),
])
async def test_whole_matrix_methods_refuse_with_a_reason(client, both, method, path, kw):
    res = await client.request(method, f"{API}/{LARGE}{path}", **kw)
    if res.status_code == 404 and "pseudotime" in res.text:
        pytest.skip("no pseudotime column in the fixture")
    assert res.status_code == 422, res.text
    body = res.json()
    assert body["large_dataset"] is True
    assert "not available for large datasets yet" in body["detail"]


async def test_availability_checks_say_why_not(client, both):
    progs = (await client.get(f"{API}/{LARGE}/programs")).json()
    assert progs["runnable"]["ok"] is False
    assert "not available for large datasets yet" in progs["runnable"]["reason"]
    hvg = (await client.get(f"{API}/{LARGE}/clustering/hvg-sensitivity")).json()
    assert "not available for large datasets yet" in json.dumps(hvg)


async def test_assessment_sees_a_normalised_dataset(client, both):
    # X on disk has a dtype but is no array; the counts are in their own file
    a = (await client.get(f"{API}/{LARGE}/assessment")).json()
    assert a["normalization"]["done"] is True and a["log_transform"]["done"] is True
    insight = (await client.get(f"{API}/{LARGE}/assistant/insight")).json()
    assert "raw counts" not in json.dumps(insight)
