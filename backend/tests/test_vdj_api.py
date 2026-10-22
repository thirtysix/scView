"""V(D)J endpoints (api/v1/vdj.py): attach, summary, the repertoire views, redefine, remove."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from scview.core import provenance, vdj
from tests.test_hvg_sensitivity import _register
from tests.test_vdj import STATUS, _merged, _repertoire, b_rows, cells_adata, t_rows, write_csv

API = "/api/v1"


async def _attach(client, did, path: Path, **form):
    with open(path, "rb") as fh:
        return await client.post(f"{API}/datasets/{did}/vdj", files={"file": (path.name, fh, "text/csv")},
                                 data={k: str(v) for k, v in form.items()})


async def test_attach_saves_columns_and_serves_the_summary(client, dataset_manager, tmp_path):
    did = _register(dataset_manager, cells_adata(), "vd")
    s = (await client.get(f"{API}/datasets/{did}/vdj")).json()
    assert s["has_vdj"] is False and "filtered_contig_annotations.csv" in s["how_to"]
    assert (await client.get(f"{API}/datasets/{did}/vdj/clonotypes")).status_code == 404

    t = write_csv(tmp_path / "filtered_contig_annotations.csv", t_rows())
    r = await _attach(client, did, t)
    assert r.status_code == 200, r.text
    body = r.json()
    # The T file alone: cell 9 has only B-cell contigs; cell 6's lone TRA is its own clonotype.
    assert body["attached"]["matched"] == 8 and body["attached"]["unmatched_usable"] == 1
    assert body["summary"]["has_vdj"] is True and body["summary"]["n_clonotypes"] == 3
    stored = Path(dataset_manager.data_dir) / "uploads" / did / "vdj" / body["attached"]["stored_as"]
    assert stored.exists() and not os.access(stored, os.W_OK)

    # A B-cell table with the same file name is a second file, not a replacement.
    (tmp_path / "b").mkdir()
    b = write_csv(tmp_path / "b" / "filtered_contig_annotations.csv", b_rows())
    assert (await _attach(client, did, b)).status_code == 200
    again = await _attach(client, did, b)
    assert again.status_code == 409 and "already attached" in again.json()["detail"]

    dataset_manager.evict(did)
    adata = (await dataset_manager.get_or_load_dataset(did)).adata
    assert adata.obs[vdj.STATUS_COL].astype(str).tolist() == STATUS
    assert [h["step"] for h in provenance.read_provenance(adata)["history"]] == ["vdj_attach"] * 2
    ds = (await client.get(f"{API}/datasets/{did}")).json()
    assert vdj.CHAINS_KEY not in ds["available_embeddings"]
    assert {"clonotype", "clone_expansion", "vdj_status"} <= {c["name"] for c in ds["obs_columns"]}
    s = (await client.get(f"{API}/datasets/{did}/vdj")).json()
    assert len(s["files"]) == 2 and s["match"]["cells_without_contigs"] == 1

    rows = (await client.get(f"{API}/datasets/{did}/vdj/clonotypes?n=2")).json()["rows"]
    assert [r["id"] for r in rows] == ["TCR1", "TCR2"] and rows[0]["indices"] == [0, 1, 2]


async def test_redefine_and_remove(client, dataset_manager, tmp_path):
    did = _register(dataset_manager, cells_adata(), "vr")
    assert (await _attach(client, did, write_csv(tmp_path / "all.csv", t_rows() + b_rows()))).status_code == 200
    r = await client.post(f"{API}/datasets/{did}/vdj/clonotypes", json={"same_v_gene": True})
    assert r.status_code == 200 and r.json()["summary"]["same_v_gene"] is True
    r = await client.delete(f"{API}/datasets/{did}/vdj")
    assert r.status_code == 200 and "clonotype" in r.json()["removed_columns"]
    assert r.json()["summary"]["has_vdj"] is False
    dataset_manager.evict(did)
    adata = (await dataset_manager.get_or_load_dataset(did)).adata
    assert "clonotype" not in adata.obs.columns and vdj.CHAINS_KEY not in adata.obsm
    assert (await client.delete(f"{API}/datasets/{did}/vdj")).status_code == 404


async def test_merged_dataset_needs_the_sample(client, dataset_manager, tmp_path):
    did = _register(dataset_manager, _merged(), "vm")
    path = write_csv(tmp_path / "all.csv", t_rows() + b_rows())
    r = await _attach(client, did, path)
    assert r.status_code == 409 and "libraries" in r.json()["detail"]
    assert not list((Path(dataset_manager.data_dir) / "uploads" / did / "vdj").iterdir())
    s = (await client.get(f"{API}/datasets/{did}/vdj")).json()
    names = [c["name"] for c in s["sample_choices"]]
    assert names[0] == "sample" and vdj.SUFFIX_COLUMN in names
    assert (await _attach(client, did, path, sample_column="sample", sample="nope")).status_code == 404
    r = await _attach(client, did, path, sample_column="sample", sample="s1")
    assert r.status_code == 200 and r.json()["summary"]["match"]["cells_not_profiled"] == 4


async def test_repertoire_views(client, dataset_manager, tmp_path):
    a, rows = _repertoire()
    did = _register(dataset_manager, a, "vv")
    assert (await _attach(client, did, write_csv(tmp_path / "r.csv", rows))).status_code == 200
    d = (await client.get(f"{API}/datasets/{did}/vdj/diversity?group_by=group&rarefy=true&seed=1")).json()
    assert d["depth"] == 20 and [g["name"] for g in d["groups"]] == ["A", "B"]
    o = (await client.get(f"{API}/datasets/{did}/vdj/overlap?group_by=group&metric=jaccard")).json()
    assert np.isclose(o["matrix"][0][1], 1 / 46, atol=1e-4)
    u = (await client.get(f"{API}/datasets/{did}/vdj/usage?group_by=group&gene=j&count=cells")).json()
    assert u["genes"] == ["TRBJ1"] and u["share"] == [[1.0], [1.0]]
    t = (await client.get(f"{API}/datasets/{did}/vdj/clonotypes?group_by=group&group=B&n=1")).json()
    assert t["rows"][0]["n_in_group"] == 5
    assert (await client.get(f"{API}/datasets/{did}/vdj/overlap?group_by=nope")).status_code == 404
    assert (await client.get(f"{API}/datasets/{did}/vdj/usage?group_by=group&chain=XYZ")).status_code == 400
