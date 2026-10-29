"""WNN endpoints (api/v1/wnn.py): plan, run, save, read, weights."""

from __future__ import annotations

import numpy as np

from scview.api.v1 import wnn as wnn_api
from scview.core import provenance, wnn
from tests.test_hvg_sensitivity import _register
from tests.test_wnn import CONTROLS, _cite

API = "/api/v1"


def _events(text: str) -> list[tuple[str, dict]]:
    """The stream's events; keep-alive comments (a first run compiles numba) skipped."""
    import json

    out = []
    for block in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines() if not line.startswith(":"))
        if "event" in lines:
            out.append((lines["event"], json.loads(lines["data"])))
    return out


async def test_run_saves_the_result_and_it_is_served(client, dataset_manager):
    did = _register(dataset_manager, _cite(n_per=40), "wn")
    info = (await client.get(f"{API}/datasets/{did}/wnn?groupby=group")).json()
    assert info["available"] is False and info["running"] is False
    plan = info["runnable"]["plan"]
    assert info["runnable"]["ok"] is True and plan["controls"] == CONTROLS and plan["n_proteins"] == 8
    assert plan["d_rna"] == 10 and plan["d_protein"] == 7 and plan["estimate_seconds"] > 0

    r = await client.post(f"{API}/datasets/{did}/wnn/run-stream", json={"resolution": 0.8})
    assert r.status_code == 200, r.text
    events = _events(r.text)
    kinds = [e for e, _ in events]
    assert kinds[0] == "start" and kinds[-2:] == ["result", "complete"], kinds
    assert [d["stage"] for e, d in events if e == "stage"] == [s for s, _ in wnn.STAGES]
    assert events[0][1]["resolution"] == 0.8 and len(events[0][1]["stages"]) == len(wnn.STAGES)
    result = events[-2][1]
    assert result["available"] is True and result["stale"] is False

    # Saved: the dataset reopens with it; the two new UMAPs are embeddings, the protein PCs not.
    dataset_manager.evict(did)
    ds = (await client.get(f"{API}/datasets/{did}")).json()
    assert {wnn.OBSM_WNN, wnn.OBSM_PROTEIN} <= set(ds["available_embeddings"])
    assert wnn.OBSM_PCA not in ds["available_embeddings"]
    adata = (await dataset_manager.get_or_load_dataset(did)).adata
    assert adata.obsp[wnn.CONN_KEY].shape == (adata.n_obs, adata.n_obs)
    assert "neighbors" in adata.uns and adata.uns["wnn"]["connectivities_key"] == wnn.CONN_KEY
    assert provenance.read_provenance(adata)["history"][-1]["step"] == "wnn"
    for col in wnn.OBS_KEYS:
        assert col in adata.obs.columns

    info = (await client.get(f"{API}/datasets/{did}/wnn?groupby=group")).json()
    assert info["available"] is True and info["stale"] is False
    rows = {r["group"]: r["mean_rna_weight"] for r in info["by_group"]["rows"]}
    assert rows["g2"] > rows["g0"] and rows["g3"] > rows["g1"]
    assert info["embeddings"] == ["X_umap_protein", "X_umap_wnn"]

    w = (await client.get(f"{API}/datasets/{did}/wnn/weights?groupby=group")).json()
    assert len(w["values"]) == adata.n_obs and set(w["violin"]) == {"g0", "g1", "g2", "g3"}
    np.testing.assert_allclose(w["values"], adata.obs[wnn.OBS_RNA].to_numpy(), atol=1e-4)


def _sha(path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


async def test_a_large_dataset_runs_saves_in_place_and_equals_the_dataset_in_memory(
        client, dataset_manager, monkeypatch):
    # The same cells as a large dataset (X on disk, counts and gene store sidecars) and in
    # memory: WNN reads only obsm, so the two runs give the same weights, graph, layouts
    # and clusters (==); the large one is saved into its h5ad in place.
    import anndata as ad
    import h5py
    import pandas as pd
    import scipy.sparse as sp

    from scview.core import large_layout, large_mode
    from tests.large_fixture import register, small_twin

    c = _cite(n_per=40)
    rng = np.random.default_rng(1)
    counts = ad.AnnData(X=sp.csr_matrix(rng.poisson(2.0, (c.n_obs, 40)).astype(np.float32)), obs=c.obs,
                        var=pd.DataFrame(index=[f"G{i}" for i in range(40)]), obsm=dict(c.obsm), uns=c.uns)
    h5ad = register(dataset_manager.data_dir, "wL", counts)
    small = _register(dataset_manager, small_twin(counts), "wS")
    sidecars = large_layout.sidecar_paths(h5ad)
    before = [_sha(p) for p in sidecars]
    with h5py.File(h5ad, "r") as f:
        x_offset = f["X"]["data"].id.get_offset()

    info = (await client.get(f"{API}/datasets/wL/wnn")).json()
    assert info["runnable"]["ok"] is True, info["runnable"]
    assert info["runnable"]["plan"] == (await client.get(f"{API}/datasets/wS/wnn")).json()["runnable"]["plan"]

    def boom(*_a, **_k):
        raise AssertionError("WNN read the expression matrix")

    with monkeypatch.context() as m:  # nothing of X, rows or genes, from here to the save
        for name in ("__getitem__", "_columns", "to_memory", "row_blocks"):
            m.setattr(large_mode.OnDiskMatrix, name, boom)
        r = await client.post(f"{API}/datasets/wL/wnn/run-stream", json={})
    assert r.status_code == 200, r.text
    assert [e for e, _ in _events(r.text)][-2:] == ["result", "complete"], r.text
    r = await client.post(f"{API}/datasets/{small}/wnn/run-stream", json={})
    assert [e for e, _ in _events(r.text)][-2:] == ["result", "complete"], r.text

    # The h5ad was edited in place: X neither rewritten nor moved, the sidecars untouched.
    with h5py.File(h5ad, "r") as f:
        assert f["X"]["data"].id.get_offset() == x_offset
        assert set(wnn.OBSP_KEYS) <= set(f["obsp"]) and set(wnn.OBSM_KEYS) <= set(f["obsm"])
    assert [_sha(p) for p in sidecars] == before

    dataset_manager.evict("wL")
    dataset_manager.evict(small)
    big = await dataset_manager.get_or_load_dataset("wL")
    mem = (await dataset_manager.get_or_load_dataset(small)).adata
    assert big.is_large and big.large_info()["gene_store"] == "ready"
    assert np.array_equal(big.get_expression(["G3", "G17"]),
                          (await dataset_manager.get_or_load_dataset(small)).get_expression(["G3", "G17"],
                                                                                           layer="X"))
    a = big.adata
    for col in wnn.OBS_KEYS:
        assert np.array_equal(a.obs[col].to_numpy(), mem.obs[col].to_numpy()), col
    for key in wnn.OBSM_KEYS:
        assert np.array_equal(np.asarray(a.obsm[key]), np.asarray(mem.obsm[key])), key
    for key in wnn.OBSP_KEYS:
        assert (a.obsp[key] != mem.obsp[key]).nnz == 0, key
    assert a.uns["wnn"]["connectivities_key"] == wnn.CONN_KEY
    assert provenance.read_provenance(a)["history"][-1]["step"] == "wnn"
    info = (await client.get(f"{API}/datasets/wL/wnn?groupby=group")).json()
    assert info["available"] is True and info["stale"] is False and info["by_group"]["rows"]
    ds = (await client.get(f"{API}/datasets/wL")).json()
    assert {wnn.OBSM_WNN, wnn.OBSM_PROTEIN} <= set(ds["available_embeddings"])
    w = (await client.get(f"{API}/datasets/wL/wnn/weights")).json()
    np.testing.assert_allclose(w["values"], mem.obs[wnn.OBS_RNA].to_numpy(), atol=1e-4)


async def test_a_second_run_while_one_works_gets_409(client, dataset_manager):
    did = _register(dataset_manager, _cite(n_per=20), "wb")
    wnn_api._running.add(did)
    try:
        r = await client.post(f"{API}/datasets/{did}/wnn/run-stream", json={})
        assert r.status_code == 409 and "already running" in r.json()["detail"]
        assert (await client.get(f"{API}/datasets/{did}/wnn")).json()["running"] is True
    finally:
        wnn_api._running.discard(did)


async def test_refusals(client, dataset_manager):
    did = _register(dataset_manager, _cite(n_per=20, n_prot=4), "wr")
    info = (await client.get(f"{API}/datasets/{did}/wnn")).json()
    assert info["runnable"]["ok"] is False and "at least 5 proteins" in info["runnable"]["reason"]
    r = await client.post(f"{API}/datasets/{did}/wnn/run-stream", json={})
    assert r.status_code == 400 and "at least 5 proteins" in r.json()["detail"]
    r = await client.get(f"{API}/datasets/{did}/wnn/weights")
    assert r.status_code == 404 and "No WNN result" in r.json()["detail"]
    did2 = _register(dataset_manager, _cite(n_per=20), "wp")
    r = await client.post(f"{API}/datasets/{did2}/wnn/run-stream", json={"protein_pcs": 9})
    assert r.status_code == 400 and "between 2 and 7" in r.json()["detail"]
