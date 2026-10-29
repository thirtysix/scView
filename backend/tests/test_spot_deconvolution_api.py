"""Spot composition endpoints (api/v1/spot_deconvolution.py): info, reference check,
streamed fit + save, the tissue overlay values, the self-check."""

from __future__ import annotations

import numpy as np

from scview.core import provenance, results_store
from scview.core import spot_deconvolution as sd
from tests.test_hvg_sensitivity import _events, _register
from tests.test_spot_deconvolution import TYPES, _profiles, _reference, _spots

API = "/api/v1"


def _pair(dm, tag: str = "") -> tuple[str, str]:
    prof = _profiles()
    ref = _reference(prof)
    ref.obs["n_counts"] = np.asarray(ref.X.sum(axis=1)).ravel()
    spots, _ = _spots(prof, n=100)
    spots.obs["leiden"] = np.where(np.arange(spots.n_obs) < 50, "0", "1")
    spots.uns["scview_active_clustering"] = "leiden"
    return _register(dm, ref, f"sref{tag}"), _register(dm, spots, f"spots{tag}")


async def _post(client, did, path, **body):
    r = await client.post(f"{API}/datasets/{did}/spot-deconvolution/{path}", json=body)
    assert r.status_code == 200, r.text
    return _events(r.text)


async def test_reference_check_then_run_saves_and_serves_the_shares(client, dataset_manager):
    ref, did = _pair(dataset_manager)
    info = (await client.get(f"{API}/datasets/{did}/spot-deconvolution")).json()
    assert info["available"] is True and info["saved"] is None and info["running"] is False
    assert info["methods"] == list(sd.METHODS) and info["min_type_cells"] == sd.MIN_TYPE_CELLS

    chk = (await client.get(f"{API}/datasets/{did}/spot-deconvolution/reference/{ref}")).json()
    assert chk["ok"] is True and chk["n_cells"] == 240 and chk["celltype_column"] == "cell_type"
    assert {t["type"] for t in chk["types"]} == set(TYPES) and chk["n_small"] == 0
    assert chk["shared_genes_by_name"] == 400 and chk["estimate"]["peak_bytes"] > 0
    assert chk["estimate"]["seconds"] > 0 and chk["estimate"]["note"].startswith("about")

    events = await _post(client, did, "run-stream", reference_id=ref, celltype_column="cell_type")
    kinds = [e for e, _ in events]
    assert kinds[0] == "start" and kinds[-2:] == ["result", "complete"], events[-1]
    steps = {d.get("step") for e, d in events if e == "progress"}
    assert steps >= {"read", "prepared", "genes", "platform", "baseline", "sigma", "table", "fit"}
    res = events[-2][1]
    assert res["types"] == sorted(TYPES, key=str.lower) and res["method"] == "rctd"
    assert res["cluster_column"] == "leiden" and set(res["clusters"]["rna"]) == {"0", "1"}
    assert res["n_fitted"] == 100 and res["reference"]["id"] == ref and 0.1 <= res["s_e"] <= 0.5

    # Saved: the dataset reopens with the arrays, the columns, the summary and the step.
    dataset_manager.evict(did)
    adata = (await dataset_manager.get_or_load_dataset(did)).adata
    assert adata.obsm[sd.OBSM_RNA].shape == (100, 4) and sd.OBS_TOP in adata.obs.columns
    step = provenance.read_provenance(adata)["history"][-1]
    assert step["step"] == sd.STEP and step["params"]["reference_id"] == ref
    assert sd.STEP in results_store.ADDITIVE_STEPS
    info = (await client.get(f"{API}/datasets/{did}/spot-deconvolution")).json()
    assert info["saved"]["result_id"] == res["result_id"]
    assert [e["kind"] for e in results_store.entries(did)].count(sd.KIND) == 1

    r = await client.get(f"{API}/datasets/{did}/spot-deconvolution/share", params={"type": "B"})
    assert r.status_code == 200
    vals = r.json()["values"]
    assert len(vals) == 100 and all(0 <= v <= 1 for v in vals)
    k = res["types"].index("B")
    assert np.allclose(vals, adata.obsm[sd.OBSM_RNA][:, k], atol=1e-4)
    r = await client.get(f"{API}/datasets/{did}/spot-deconvolution/share", params={"type": "B", "kind": "cell"})
    assert np.allclose(r.json()["values"], adata.obsm[sd.OBSM_CELLS][:, k], atol=1e-4)
    r = await client.get(f"{API}/datasets/{did}/spot-deconvolution/share", params={"type": "Nope"})
    assert r.status_code == 404
    # The reference is untouched.
    ref_adata = (await dataset_manager.get_or_load_dataset(ref)).adata
    assert sd.STEP not in [h["step"] for h in provenance.read_provenance(ref_adata)["history"]]


async def test_refusals(client, dataset_manager):
    ref, did = _pair(dataset_manager, "x")
    path = f"{API}/datasets/{did}/spot-deconvolution"
    r = await client.post(f"{path}/run-stream", json={"reference_id": did, "celltype_column": "cell_type"})
    assert r.status_code == 400
    r = await client.post(f"{path}/run-stream", json={"reference_id": "missing", "celltype_column": "x"})
    assert r.status_code == 404
    r = await client.post(f"{path}/run-stream", json={"reference_id": ref, "celltype_column": "cell_type",
                                                      "method": "magic"})
    assert r.status_code == 422
    events = await _post(client, did, "run-stream", reference_id=ref, celltype_column="n_counts")
    assert events[-1][0] == "error" and "numbers" in events[-1][1]["error"]
    assert (await client.get(f"{path}/reference/{did}")).status_code == 400
    assert (await client.get(f"{path}/reference/missing")).status_code == 404
    assert (await client.get(f"{path}/share", params={"type": "B"})).status_code == 404
    assert (await client.get(f"{path}/check")).status_code == 404
    # A dataset of cells, not spots: explained, and the run refused.
    prof = _profiles()
    cells = _reference(prof, seed=3)
    cid = _register(dataset_manager, cells, "cellsx")
    info = (await client.get(f"{API}/datasets/{cid}/spot-deconvolution")).json()
    assert info["available"] is False and "needs Visium spots" in info["reason"]
    r = await client.post(f"{API}/datasets/{cid}/spot-deconvolution/run-stream",
                          json={"reference_id": ref, "celltype_column": "cell_type"})
    assert r.status_code == 422 and "needs Visium spots" in r.json()["detail"]


async def test_a_reference_with_small_types_flags_them(client, dataset_manager):
    prof = _profiles()
    ref = _reference(prof)
    ref = ref[np.r_[0:180, 180:190]].copy()  # Stroma: 10 cells
    rid = _register(dataset_manager, ref, "srefsmall")
    spots, _ = _spots(prof, n=60)
    did = _register(dataset_manager, spots, "spotssmall")
    chk = (await client.get(f"{API}/datasets/{did}/spot-deconvolution/reference/{rid}",
                            params={"celltype_column": "cell_type"})).json()
    small = [t for t in chk["types"] if t["small"]]
    assert chk["n_small"] == 1 and small == [{"type": "Stroma", "n_cells": 10, "small": True}]


async def test_self_check_streams_and_is_kept(client, dataset_manager, monkeypatch):
    monkeypatch.setattr(sd, "CHECK_SPOTS", 80)
    monkeypatch.setattr(sd, "CHECK_DEPTHS", (5000,))
    ref, did = _pair(dataset_manager, "c")
    events = await _post(client, did, "check-stream", reference_id=ref, celltype_column="cell_type")
    assert events[-2][0] == "result" and events[-1][0] == "complete", events[-1]
    res = events[-2][1]
    assert {r["method"] for r in res["runs"]} == set(sd.METHODS) and res["n_spots"] == 80
    got = (await client.get(f"{API}/datasets/{did}/spot-deconvolution/check")).json()
    assert got["id"] == res["id"]
    # Nothing saved in the dataset.
    adata = (await dataset_manager.get_or_load_dataset(did)).adata
    assert sd.OBSM_RNA not in adata.obsm


def test_the_guessed_column_is_one_the_fit_accepts():
    import pandas as pd

    from scview.api.v1.spot_deconvolution import _guess_column
    from scview.core import spot_deconvolution as sd

    n_fine = sd.MAX_TYPES + 4  # a lymphoid atlas: 'PrelimCellType' 54 types, 'Subset' 34
    fine = [f"t{i}" for i in range(n_fine) for _ in range(sd.MIN_TYPE_CELLS)]
    coarse = [f"c{i % 4}" for i in range(len(fine))]
    obs = pd.DataFrame({"Donor": [f"d{i % 3}" for i in range(len(fine))], "PrelimCellType": fine, "Subset": coarse})
    cols = [{"column": c, "n_labels": int(obs[c].nunique())} for c in obs.columns]
    assert _guess_column(obs, cols) == "Subset"
    assert _guess_column(obs[["Donor", "PrelimCellType"]], cols[:2]) == "PrelimCellType"  # refused later, named so
    assert _guess_column(obs[["Donor"]], cols[:1]) == "Donor"
