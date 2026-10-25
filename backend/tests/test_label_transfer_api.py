"""Reference mapping endpoints (api/v1/label_transfer.py): info, reference check, streamed run + save."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from scview.core import label_transfer as lt
from scview.core import provenance, replay_script
from tests.test_hvg_sensitivity import _events, _register
from tests.test_label_transfer import _query, _reference

API = "/api/v1"


def _pair(dm, tag: str = "") -> tuple[str, str]:
    ref = _reference()
    ref.obs["n_counts"] = np.asarray(ref.X.sum(axis=1)).ravel()
    q = _query()
    q.uns["scview_active_clustering"] = "cluster"
    return _register(dm, ref, f"ref{tag}"), _register(dm, q, f"qry{tag}")


async def _run(client, did, **body):
    r = await client.post(f"{API}/datasets/{did}/label-transfer/run-stream", json=body)
    assert r.status_code == 200, r.text
    return _events(r.text)


async def test_reference_check_then_run_saves_the_columns(client, dataset_manager):
    ref, did = _pair(dataset_manager)
    info = (await client.get(f"{API}/datasets/{did}/label-transfer")).json()
    assert info["available"] is True and info["saved"] == [] and info["running"] is False
    assert info["methods"] == list(lt.METHODS) and info["k"] == lt.DEFAULT_K

    r = await client.get(f"{API}/datasets/{did}/label-transfer/reference/{ref}")
    assert r.status_code == 200, r.text
    chk = r.json()
    assert chk["ok"] is True and chk["n_cells"] == 450 and chk["has_umap"] is True
    assert chk["label_columns"] == [{"column": "cell_type", "n_labels": 3}]
    assert chk["shared_genes_by_name"] == 900
    assert chk["estimate"]["peak_bytes"] > 0 and chk["estimate"]["seconds"] > 0

    events = await _run(client, did, reference_id=ref, label_column="cell_type", name="Blood ref")
    kinds = [e for e, _ in events]
    assert kinds[0] == "start" and kinds[-2:] == ["result", "complete"], events[-1]
    assert events[0][1]["prefix"] == "blood_ref"
    assert {d["step"] for e, d in events if e == "progress"} >= {"read", "prepared", "genes", "space"}
    res = events[-2][1]
    assert res["columns"] == {"label": "blood_ref_transfer", "score": "blood_ref_transfer_score",
                              "unmapped": "blood_ref_unmapped", "embedding": "X_umap_ref_blood_ref"}
    assert res["cluster_column"] == "cluster"
    top = res["clusters"][0]
    assert top["cluster"] == "3" and top["candidate_absent"] is True
    assert res["reference"]["id"] == ref and res["reference"]["label_column"] == "cell_type"

    # Saved: the dataset reopens with the columns, the embedding, the summary and the step.
    dataset_manager.evict(did)
    ds = (await client.get(f"{API}/datasets/{did}")).json()
    assert "X_umap_ref_blood_ref" in [e if isinstance(e, str) else e.get("name") for e in ds["available_embeddings"]]
    adata = (await dataset_manager.get_or_load_dataset(did)).adata
    truth = adata.obs["truth"].astype(str).to_numpy()
    known = truth != "D"
    assert (adata.obs["blood_ref_transfer"].astype(str).to_numpy()[known] == truth[known]).mean() > 0.97
    assert adata.obs["blood_ref_unmapped"].to_numpy()[~known].mean() > 0.8
    step = provenance.read_provenance(adata)["history"][-1]
    assert step["step"] == "label_transfer" and step["params"]["reference_id"] == ref
    assert step["params"]["reference_file"] == "data.h5ad" and step["params"]["reference_mtime"] > 0
    assert replay_script.describe(adata)[-1]["status"] == replay_script.EXACT
    info = (await client.get(f"{API}/datasets/{did}/label-transfer")).json()
    assert [s["prefix"] for s in info["saved"]] == ["blood_ref"]
    # The reference is untouched.
    src = ad.read_h5ad(Path(dataset_manager.data_dir) / "uploads" / ref / "data.h5ad")
    assert "label_transfer" not in [h["step"] for h in provenance.read_provenance(src)["history"]]


async def test_refusals(client, dataset_manager):
    ref, did = _pair(dataset_manager, "x")
    r = await client.post(f"{API}/datasets/{did}/label-transfer/run-stream",
                          json={"reference_id": did, "label_column": "cluster"})
    assert r.status_code == 400
    r = await client.post(f"{API}/datasets/{did}/label-transfer/run-stream",
                          json={"reference_id": "missing", "label_column": "x"})
    assert r.status_code == 404
    r = await client.post(f"{API}/datasets/{did}/label-transfer/run-stream",
                          json={"reference_id": ref, "label_column": "cell_type", "method": "magic"})
    assert r.status_code == 400
    r = await client.post(f"{API}/datasets/{did}/label-transfer/run-stream",
                          json={"reference_id": ref, "label_column": "cell_type", "k": 1})
    assert r.status_code == 422
    events = await _run(client, did, reference_id=ref, label_column="n_counts")
    assert events[-1][0] == "error" and "numbers" in events[-1][1]["error"]
    assert (await client.get(f"{API}/datasets/{did}/label-transfer/reference/{did}")).status_code == 400
    assert (await client.get(f"{API}/datasets/{did}/label-transfer/reference/missing")).status_code == 404


async def test_too_few_shared_genes_is_explained(client, dataset_manager):
    ref = _reference()
    q = _query()[:, :300].copy()
    rid, did = _register(dataset_manager, ref, "rs"), _register(dataset_manager, q, "qs")
    chk = (await client.get(f"{API}/datasets/{did}/label-transfer/reference/{rid}")).json()
    assert chk["shared_genes_by_name"] == 300 and chk["min_shared_genes"] == lt.MIN_SHARED_GENES
    events = await _run(client, did, reference_id=rid, label_column="cell_type")
    assert events[-1][0] == "error" and "share 300 genes" in events[-1][1]["error"]


async def test_a_dataset_whose_cells_changed_is_not_saved(client, dataset_manager, monkeypatch):
    ref, did = _pair(dataset_manager, "c")
    from scview.core.anndata_adaptor import AnnDataAdaptor

    calls = {"n": 0}
    real = AnnDataAdaptor.cells_fingerprint

    def changing(self):
        calls["n"] += 1
        return real(self) if calls["n"] == 1 else "other cells"

    monkeypatch.setattr(AnnDataAdaptor, "cells_fingerprint", changing)
    events = await _run(client, did, reference_id=ref, label_column="cell_type")
    assert events[-1][0] == "error" and "nothing was saved" in events[-1][1]["error"]
    monkeypatch.undo()
    dataset_manager.evict(did)
    adata = (await dataset_manager.get_or_load_dataset(did)).adata
    assert not any(str(c).endswith("_transfer") for c in adata.obs.columns)


async def test_a_large_dataset_is_refused(client, dataset_manager, monkeypatch):
    from scview.core import large_mode

    ref, did = _pair(dataset_manager, "l")
    monkeypatch.setattr(large_mode, "file_is_large", lambda path: True)
    r = await client.post(f"{API}/datasets/{did}/label-transfer/run-stream",
                          json={"reference_id": ref, "label_column": "cell_type"})
    assert r.status_code == 422 and "Reference mapping" in r.json()["detail"]
    chk = (await client.get(f"{API}/datasets/{did}/label-transfer/reference/{ref}")).json()
    assert chk["ok"] is False and "Reference mapping" in chk["reason"]


def test_preflight_estimate_grows_with_the_cells_and_celltypist():
    from scview.core import preflight

    a = _reference()
    b = ad.concat([a] * 4, index_unique="-")
    h, hb = preflight.header_from_adata(a), preflight.header_from_adata(b)
    small = preflight.label_transfer_need(h, h, method="projection", ref_open=True)
    big = preflight.label_transfer_need(hb, hb, method="projection", ref_open=True)
    assert big > small > preflight.LT_FIXED
    assert preflight.label_transfer_need(h, h, method="celltypist", ref_open=True) > small
    assert preflight.label_transfer_need(h, h, method="projection", ref_open=False) > small
    pd.testing.assert_index_equal(a.var_names, b.var_names)
