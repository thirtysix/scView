"""Copy-number endpoints (api/v1/cnv.py): info, streamed run + save, heatmap."""

from __future__ import annotations

import base64

import numpy as np

from scview.core import cnv, provenance
from tests.test_cnv import STEP, WINDOW, _tumour
from tests.test_hvg_sensitivity import _events, _register

API = "/api/v1"


async def _run(client, did, **body):
    body = {"reference_column": "cell_type", "reference_groups": ["T", "B"], "window": WINDOW,
            "step": STEP, **body}
    r = await client.post(f"{API}/datasets/{did}/cnv/run", json=body)
    return r


async def test_run_saves_the_profile_and_it_is_served(client, dataset_manager):
    a = _tumour()
    did = _register(dataset_manager, a, "cn")
    info = (await client.get(f"{API}/datasets/{did}/cnv")).json()
    assert info["available"] is False and info["running"] is False
    assert info["runnable"]["ok"] is True and info["runnable"]["genome"] == "human"
    assert info["runnable"]["n_placed"] == a.n_vars  # chrX kept; none of its genes on Y or MT
    assert info["defaults"]["window"] == cnv.DEFAULT_WINDOW and info["estimate"]["need_bytes"] > 0
    assert info["estimate"]["level"] in ("fits", "tight", "wont_fit", "unknown")

    r = await _run(client, did)
    assert r.status_code == 200, r.text
    events = _events(r.text)
    kinds = [e for e, _ in events]
    assert kinds[0] == "start" and kinds[-2:] == ["result", "complete"] and "progress" in kinds
    start = events[0][1]
    assert start["n_cells"] == a.n_obs and start["reference_groups"] == ["T", "B"]
    assert start["n_reference_cells"] == 240 and start["source"] == "log-normalised .raw"
    result = events[-2][1]
    assert result["available"] is True and any(c["high"] for c in result["clusters"])

    # Saved: the dataset reopens with the result; X_cnv is not offered as an embedding.
    dataset_manager.evict(did)
    ds = (await client.get(f"{API}/datasets/{did}")).json()
    assert cnv.OBSM_KEY not in ds["available_embeddings"]
    adata = (await dataset_manager.get_or_load_dataset(did)).adata
    assert adata.obsm[cnv.OBSM_KEY].shape[0] == a.n_obs
    assert {cnv.SCORE_COL, cnv.CLUSTER_COL} <= set(adata.obs.columns)
    assert provenance.read_provenance(adata)["history"][-1]["step"] == "cnv"

    info = (await client.get(f"{API}/datasets/{did}/cnv?groupby=cell_type")).json()
    rows = {g["group"]: g for g in info["by_group"]["groups"]}
    assert rows["Epi"]["ratio"] > 3 and rows["B"]["reference"] is True

    h = (await client.get(f"{API}/datasets/{did}/cnv/heatmap?max_cells=150")).json()
    codes = np.frombuffer(base64.b64decode(h["values"]), dtype=np.int8)
    assert codes.size == h["n_rows"] * h["n_cols"] and h["n_rows"] <= 170
    assert h["column"] == "cell_type"


async def test_bad_requests_and_heatmap_before_a_run(client, dataset_manager):
    did = _register(dataset_manager, _tumour(n_t=40, n_b=40, n_epi=40), "cb")
    r = await _run(client, did, reference_groups=["nope"])
    assert r.status_code == 400 and "Not in 'cell_type'" in r.json()["detail"]
    r = await client.post(f"{API}/datasets/{did}/cnv/run", json={"reference_column": "cell_type",
                                                                  "reference_groups": []})
    assert r.status_code == 422
    r = await client.get(f"{API}/datasets/{did}/cnv/heatmap")
    assert r.status_code == 404 and "run it first" in r.json()["detail"]
    r = await client.get(f"{API}/datasets/missing/cnv")
    assert r.status_code == 404


async def test_a_run_already_streaming_is_refused(client, dataset_manager):
    from scview.api.v1 import cnv as cnv_api

    did = _register(dataset_manager, _tumour(n_t=40, n_b=40, n_epi=40), "cr")
    cnv_api._running.add(did)
    try:
        assert (await client.get(f"{API}/datasets/{did}/cnv")).json()["running"] is True
        r = await _run(client, did)
        assert r.status_code == 409
    finally:
        cnv_api._running.discard(did)
