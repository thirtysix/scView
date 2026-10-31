"""RNA velocity: getting the counts in (core/velocity_input.py), the method (core/velocity.py)
and its endpoints (api/v1/velocity.py), on synthetic splicing dynamics."""

from __future__ import annotations

import json
import os
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from scview.core import modality_store as ms
from scview.core import preflight, provenance
from scview.core import velocity as vel
from scview.core import velocity_input as vi
from tests.test_hvg_sensitivity import _register
from tests.velocity_fixture import Adaptor, barcode, splicing_counts, write_loom

API = "/api/v1"


def run_core(tmp_path, adata=None, method="stochastic", n_top=200, stop=None):
    adata = adata if adata is not None else splicing_counts()
    if "spliced" in adata.layers:
        vi.adopt_anndata(adata, "ds", tmp_path, file_name="x.h5ad")
    params = vel.Params(method=method, n_top_genes=n_top, n_neighbors=20, n_pcs=10)
    p = vel.prepare(Adaptor(adata), tmp_path, "ds", params)
    out = None
    for ev in vel.iter_run(p, stop):
        if ev["type"] == "result":
            out = ev["out"]
    return adata, p, out


# -- the method on arrays ------------------------------------------------------


def test_quantile_fit_recovers_the_steady_state_ratio_through_the_origin():
    Ms = np.linspace(0, 10, 200)[:, None].astype(np.float32)
    Mu = (0.4 * Ms).astype(np.float32)
    # A transient cloud under the line in the low half must not pull the top-quantile fit.
    Mu[:100] *= 0.2
    gamma, r2, res_std = vel.vel_quantile_fit(Ms, Mu, 95)
    assert gamma[0] == pytest.approx(0.4, rel=1e-5)
    assert res_std[0] > 0
    assert r2[0] < 1


def test_stochastic_gamma_is_the_ols_on_both_moments_with_unit_scales():
    rng = np.random.default_rng(0)
    Ms = rng.uniform(0, 5, (300, 1)).astype(np.float32)
    Mu = (0.5 * Ms).astype(np.float32)
    Mss = (Ms ** 2).astype(np.float32)
    Mus = (0.5 * Mss).astype(np.float32)
    # var_ss = 2 Mss - Ms, cov_us = 2 Mus + Mu: not proportional, so the second moments do count.
    g = vel.vel_stochastic_gamma(Ms, Mu, Mss, Mus, np.ones(1), 95)
    var, cov = 2 * Mss - Ms, 2 * Mus + Mu
    res2 = (cov - float(((var * cov).sum() / (var * var).sum())) * var).std()
    keep = ((Ms / Ms.max() + Mu / Mu.max()) >= np.percentile(Ms / Ms.max() + Mu / Mu.max(), 95)) | (Ms >= np.percentile(Ms, 95))
    x1, y1 = Ms[keep].astype(float), Mu[keep].astype(float)
    x2, y2 = var.astype(float) / res2, cov.astype(float) / res2
    want = ((x1 * y1).sum() + (x2 * y2).sum()) / ((x1 * x1).sum() + (x2 * x2).sum())
    assert g[0] == pytest.approx(want, rel=1e-5)


def test_moment_weights_are_uniform_over_neighbours_and_self():
    C = sp.csr_matrix(np.array([[0, .5, .1], [.5, 0, 0], [.1, 0, 0]], dtype=np.float32))
    W = vel.vel_moment_weights(C).toarray()
    assert W[0].tolist() == pytest.approx([1 / 3] * 3)
    assert W[1].tolist() == pytest.approx([.5, .5, 0])
    # Fewer than the graph holds: each row's strongest, then the same (at most what the sparsest row has).
    W = vel.vel_moment_weights(C, 1, 2).toarray()
    assert W[0].tolist() == pytest.approx([.5, .5, 0])


def test_projection_leaves_the_self_transition_out_of_the_mean():
    """scVelo's documented formula. With scipy >= 1.13 scVelo's own T.setdiag(0) leaves it in (the
    arrow then leans away from the neighbours' centre on most rows); scipy 1.12 removes it."""
    emb = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    row = sp.csr_matrix((np.array([0.5, 0.1, -0.2], dtype=np.float32), (np.zeros(3, dtype=int), np.array([1, 2, 3]))), shape=(1, 4))
    out = vel.vel_project_rows(row, np.array([0]), np.array([0.3]), emb)[0]
    t = np.array([np.expm1(5), np.expm1(1), -np.expm1(2)])
    p = t / (np.abs(t).sum() + np.expm1(3))
    want = (p - p.mean()) @ np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    assert out == pytest.approx(want)


def test_cosine_rows_match_the_definition_and_skip_self():
    X = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 1]], dtype=np.float32)
    V = np.array([[1, 0, 0], [0, 0, 0], [0, 1, 0]], dtype=np.float32)
    indices = np.array([[1], [0], [0]])
    B = vel.vel_neighbour_sets(indices)
    g = vel.vel_cosine_rows(X, V, B, 0, 3, False).toarray()
    assert g[0, 0] == 0 and g[1].tolist() == [0, 0, 0]  # the cell itself; a zero velocity
    d = X[1] - X[0]
    d = d - d.mean()
    assert g[0, 1] == pytest.approx(float(d @ V[0] / (np.linalg.norm(d) * np.linalg.norm(V[0]))))


def test_grid_keeps_only_points_with_mass():
    rng = np.random.default_rng(0)
    X = np.r_[rng.normal(0, 0.1, (200, 2)), rng.normal(10, 0.1, (200, 2))]
    V = np.tile([1.0, 0.0], (400, 1))
    gx, gv = vel.vel_grid(X, V, density=0.5)
    assert 0 < len(gx) < 625
    assert (np.hypot(gx[:, 0] - 5, gx[:, 1] - 5) > 3).all()  # nothing in the empty middle
    assert (gv[:, 1] == 0).all() and (gv[:, 0] > 0).all()


def test_run_finds_direction_on_synthetic_dynamics(tmp_path):
    for method in vel.METHODS:
        adata, p, out = run_core(tmp_path / method, method=method)
        arrows = out["arrows"]["X_umap"]
        assert ((arrows[:, 0] > 0).mean()) > 0.8, method
        assert out["n_velocity_genes"] > 50
        assert out["confidence"].min() >= 0 and np.median(out["confidence"]) > 0.3
        assert 0.05 < out["unspliced_fraction"].mean() < 0.5
        assert "G255" not in out["genes"]  # no shared counts
        for name in (vel.RUN_STORE, vel.FIT_STORE):
            assert vel._next_path(p, name).exists()
        assert not list(ms.store_dir(tmp_path / method, "ds").glob(".velocity-work*"))
        assert {r["group"] for r in out["table"]} == {"early", "late"}


def test_memory_mapped_arrays_give_the_same_run(tmp_path, monkeypatch):
    """Past MEMMAP_BYTES Ms, Mu and the velocity arrays are files beside the store."""
    _a, _p, plain = run_core(tmp_path / "m")
    monkeypatch.setattr(vel, "MEMMAP_BYTES", 1)
    made = []
    real = vel._alloc

    def spy(shape, folder, name):
        arr = real(shape, folder, name)
        made.append(isinstance(arr, np.memmap))
        return arr

    monkeypatch.setattr(vel, "_alloc", spy)
    _a, p, mapped = run_core(tmp_path / "d")
    assert len(made) >= 5 and all(made)
    assert np.array_equal(plain["gamma"], mapped["gamma"]) and np.array_equal(plain["confidence"], mapped["confidence"])
    assert np.array_equal(plain["arrows"]["X_umap"], mapped["arrows"]["X_umap"])
    assert not list(ms.store_dir(tmp_path / "d", "ds").glob(".velocity-work*"))


def test_cancel_leaves_nothing(tmp_path):
    with pytest.raises(vel.VelocityCancelled):
        adata = splicing_counts()
        vi.adopt_anndata(adata, "ds", tmp_path)
        p = vel.prepare(Adaptor(adata), tmp_path, "ds", vel.Params(n_top_genes=200, n_neighbors=20, n_pcs=10))
        calls = {"n": 0}

        def stop():
            calls["n"] += 1
            return calls["n"] > 8

        list(vel.iter_run(p, stop))
    assert sorted(f.name for f in ms.store_dir(tmp_path, "ds").iterdir()) == ["velocity.h5"]


def test_reuses_the_datasets_graph_only_when_it_holds_enough_neighbours(tmp_path):
    import scanpy as sc

    adata = splicing_counts()
    vi.adopt_anndata(adata, "ds", tmp_path)
    tmp = ad.AnnData(adata.X.copy(), obs=adata.obs[[]])
    sc.pp.log1p(tmp)
    sc.pp.pca(tmp, n_comps=12)
    sc.pp.neighbors(tmp, n_neighbors=15, random_state=0)
    adata.obsm["X_pca"] = tmp.obsm["X_pca"]
    adata.obsp["connectivities"], adata.obsp["distances"] = tmp.obsp["connectivities"], tmp.obsp["distances"]
    adata.uns["neighbors"] = dict(tmp.uns["neighbors"])
    used = vel.prepare(Adaptor(adata), tmp_path, "ds", vel.Params(n_neighbors=15, n_pcs=10))
    assert used.connectivities is not None and "stored graph (k = 15)" in used.neighbours_note
    more = vel.prepare(Adaptor(adata), tmp_path, "ds", vel.Params(n_neighbors=30, n_pcs=10))
    assert more.connectivities is None and "k = 15" in more.neighbours_note and "unchanged" in more.neighbours_note


def test_a_graph_with_a_lonely_cell_is_not_reused():
    a = splicing_counts(60, 60)
    c = sp.csr_matrix(np.ones((60, 60), dtype=np.float32))
    d = c.tolil()
    d[3] = 0
    a.obsp["connectivities"], a.obsp["distances"] = c, sp.csr_matrix(d)
    a.uns["neighbors"] = {"params": {"n_neighbors": 30}}
    conn, _d, _k, why = vel._stored_neighbours(a, 30, np.arange(60))
    assert conn is None and "no neighbours" in why
    # A pre-0.7 file holds k as a one-value array.
    a.obsp["distances"] = c
    a.uns["neighbors"] = {"params": {"n_neighbors": np.asarray([30])}}
    assert vel._stored_neighbours(a, 30, np.arange(60))[2] == 30


def test_low_unspliced_and_low_confidence_warn_with_where_the_thresholds_came_from():
    w = vel.warnings_for(0.02, 0.2, 10, 5)
    assert any("10-25%" in x for x in w) and any("0.72 to 0.85" in x for x in w)
    assert any("10 genes" in x for x in w) and any("5 cells" in x for x in w)
    assert vel.warnings_for(0.16, 0.8, 1000, 0) == []


def test_estimate_grows_with_cells_and_says_time():
    small = preflight.estimate_velocity(3000, 3000, 9_000_000, 3_000_000)
    big = preflight.estimate_velocity(30_000, 30_000, 90_000_000, 30_000_000)
    assert big["peak_bytes"] > 5 * small["peak_bytes"] and big["seconds"] > small["seconds"]
    assert "about" in small["note"] and small["seconds_range"][0] < small["seconds"] < small["seconds_range"][1]
    # Past the memory-map size Ms and Mu leave memory: the peak stops growing with them alone.
    huge = preflight.estimate_velocity(1_000_000, 1_000_000, 10**9, 3 * 10**8, method="steady_state")
    assert huge["peak_bytes"] < 2 * 1_000_000 * 2000 * 4


# -- endpoints -----------------------------------------------------------------


def _sse(text: str) -> list[tuple[str, dict]]:
    out = []
    for chunk in text.split("\n\n"):
        lines = [x for x in chunk.split("\n") if x and not x.startswith(":")]
        if len(lines) >= 2 and lines[0].startswith("event:"):
            out.append((lines[0][7:].strip(), json.loads(lines[1][6:])))
    return out


def dataset_with_store(dataset_manager, name="vel", n_cells=300, n_genes=260):
    a = splicing_counts(n_cells, n_genes)
    layers = {k: a.layers[k] for k in vi.LAYERS}
    did = None
    bare = a.copy()
    for k in vi.LAYERS:
        del bare.layers[k]
    did = _register(dataset_manager, bare, name)
    holder = ad.AnnData(a.X.copy(), obs=a.obs[[]].copy(), var=a.var.copy(), layers=layers)
    vi.adopt_anndata(holder, did, dataset_manager.data_dir, file_name="x")
    return did, a


async def _run(client, did, **body):
    r = await client.post(f"{API}/datasets/{did}/velocity/run-stream",
                          json={"n_top_genes": 200, "n_neighbors": 20, **body})
    assert r.status_code == 200, r.text
    return _sse(r.text)


async def test_a_dataset_without_velocity_stays_as_it_is(client, dataset_manager):
    a = ad.AnnData(sp.csr_matrix(np.ones((60, 4), dtype=np.float32)), obs=pd.DataFrame(index=[f"c{i}" for i in range(60)]),
                   var=pd.DataFrame(index=list("abcd")))
    did = _register(dataset_manager, a, "plain")
    s = await client.get(f"{API}/datasets/{did}/velocity")
    assert s.status_code == 200
    body = s.json()
    assert body["attached"] is False and body["run"] is None and body["estimate"] is None
    r = await client.post(f"{API}/datasets/{did}/velocity/run-stream", json={})
    assert r.status_code == 422 and "attach them first" in r.json()["detail"]
    assert (await client.get(f"{API}/datasets/{did}/velocity/grid?embedding=X_umap")).status_code == 404
    assert (await client.get(f"{API}/datasets/{did}/velocity/genes")).status_code == 404
    assert (await client.post(f"{API}/datasets/{did}/velocity/attach", files=[("files", ("x.txt", b"hello"))])).status_code == 422
    assert not list((Path(dataset_manager.data_dir) / "uploads" / did / "velocity").glob("*"))


async def test_run_saves_columns_and_serves_grid_phase_genes(client, dataset_manager):
    did, a = dataset_with_store(dataset_manager)
    events = await _run(client, did)
    kinds = [k for k, _ in events]
    assert kinds[0] == "start" and kinds[-1] == "complete" and "result" in kinds, events[-3:]
    assert events[0][1]["estimate"]["peak_bytes"] > 0
    run = dict(events)["result"]["run"]
    assert run["method"] == "stochastic" and "gene_table" not in run and run["n_velocity_genes"] > 20
    assert "not a fate claim" in run["note"]
    assert run["clusters"][0]["group"] in ("early", "late") and run["group_column"] == "clusters"

    dataset_manager.evict(did)
    adata = (await dataset_manager.get_or_load_dataset(did)).adata
    assert {vel.OBS_CONFIDENCE, vel.OBS_LENGTH, vel.OBS_UNSPLICED} <= set(adata.obs.columns)
    assert "velocity_umap" in adata.obsm and adata.obsm["velocity_umap"].shape == (300, 2)
    assert adata.var[vel.VAR_GENE].sum() == run["n_velocity_genes"]
    assert adata.var.loc["G0", vel.VAR_GAMMA] > 0 or np.isnan(adata.var.loc["G0", vel.VAR_GAMMA])
    assert not any(k in adata.layers for k in vi.LAYERS)
    assert [h["step"] for h in provenance.read_provenance(adata)["history"]][-1] == "velocity"

    g = (await client.get(f"{API}/datasets/{did}/velocity/grid?embedding=X_umap")).json()
    assert g["n"] == len(g["x"]) == len(g["u"]) > 10 and not g["reprojected"] and g["step"] > 0 and g["length_p95"] > 0
    assert np.mean(np.asarray(g["u"]) > 0) > 0.7
    dense = (await client.get(f"{API}/datasets/{did}/velocity/grid?embedding=X_umap&density=1.5")).json()
    assert dense["n"] > g["n"]

    genes = (await client.get(f"{API}/datasets/{did}/velocity/genes")).json()
    assert genes["n_velocity_genes"] == run["n_velocity_genes"] and genes["genes"][0]["velocity_gene"]
    top = genes["genes"][0]["gene"]
    p = (await client.get(f"{API}/datasets/{did}/velocity/phase", params={"gene": top, "color_by": "clusters"})).json()
    assert p["n_points"] == 300 == len(p["spliced"]) == len(p["unspliced"]) and p["gamma"] > 0
    assert p["color"]["kind"] == "category" and set(p["color"]["categories"]) == {"early", "late"}
    assert len(p["color"]["codes"]) == 300
    # The phase is the mean over each cell's neighbours: its spread is under the raw counts'.
    miss = await client.get(f"{API}/datasets/{did}/velocity/phase", params={"gene": "nope"})
    assert miss.status_code == 404 and "fitted" in miss.json()["detail"]
    num = (await client.get(f"{API}/datasets/{did}/velocity/phase", params={"gene": top, "color_by": vel.OBS_CONFIDENCE})).json()
    assert num["color"]["kind"] == "numeric"

    s = (await client.get(f"{API}/datasets/{did}/velocity")).json()
    assert s["attached"] and s["run"]["result_id"] == run["result_id"] and s["stale"] is None
    assert s["source"]["cells_now"] == 300 and s["estimate"]["seconds"] > 0


async def test_steady_state_and_an_embedding_the_run_did_not_see(client, dataset_manager):
    did, a = dataset_with_store(dataset_manager, "vel2")
    events = await _run(client, did, method="steady_state")
    assert dict(events)["result"]["run"]["method"] == "steady_state"
    adaptor = await dataset_manager.get_or_load_dataset(did)
    adaptor.adata.obsm["X_tsne"] = np.c_[a.obsm["X_umap"][:, 1], -a.obsm["X_umap"][:, 0]].astype(np.float32)
    g = (await client.get(f"{API}/datasets/{did}/velocity/grid?embedding=X_tsne")).json()
    assert g["reprojected"] and g["n"] > 10
    assert np.mean(np.asarray(g["v"]) < 0) > 0.7  # the same direction, in this embedding's axes
    # A new layout of the same name counts as another embedding, too.
    adaptor.adata.obsm["X_umap"] = np.c_[-a.obsm["X_umap"][:, 0], a.obsm["X_umap"][:, 1]].astype(np.float32)
    g = (await client.get(f"{API}/datasets/{did}/velocity/grid?embedding=X_umap")).json()
    assert g["reprojected"] and np.mean(np.asarray(g["u"]) < 0) > 0.7
    adaptor.adata.obsm["X_umap_3d"] = np.zeros((300, 3), dtype=np.float32)
    r = await client.get(f"{API}/datasets/{did}/velocity/grid?embedding=X_umap_3d")
    assert r.status_code == 422 and "2D" in r.json()["detail"]
    assert (await client.get(f"{API}/datasets/{did}/velocity/grid?embedding=X_nope")).status_code == 404


async def test_cells_changed_since_the_run_says_run_again(client, dataset_manager):
    did, _a = dataset_with_store(dataset_manager, "vel3")
    await _run(client, did)
    adaptor = await dataset_manager.get_or_load_dataset(did)
    run_fp = vel.saved(adaptor.adata)["fingerprint"]
    assert run_fp == adaptor.cells_fingerprint()
    adaptor.cells_fingerprint = lambda: "others"
    s = (await client.get(f"{API}/datasets/{did}/velocity")).json()
    assert "run velocity again" in s["stale"] and s["source"]["cells_without_now"] == 0
    assert (await client.get(f"{API}/datasets/{did}/velocity/grid?embedding=X_umap")).status_code == 409


async def test_filtered_dataset_runs_on_the_cells_it_has(client, dataset_manager):
    """The store keeps every barcode; cells the dataset no longer has are simply not asked for,
    and a store cell the dataset lacks is reported, not a zero row."""
    a = splicing_counts()
    bare = a[:250].copy()
    for k in vi.LAYERS:
        del bare.layers[k]
    did = _register(dataset_manager, bare, "vel4")
    holder = ad.AnnData(a.X.copy(), obs=a.obs[[]].copy(), var=a.var.copy(), layers={k: a.layers[k] for k in vi.LAYERS})
    vi.adopt_anndata(holder[20:].copy(), did, dataset_manager.data_dir)  # 20 of the dataset's cells have no counts
    events = await _run(client, did)
    run = dict(events)["result"]["run"]
    assert run["n_cells"] == 230 and run["cells_without"] == 20
    assert any("20 cells have no velocity counts" in w for w in run["warnings"])
    dataset_manager.evict(did)
    adata = (await dataset_manager.get_or_load_dataset(did)).adata
    assert adata.obs[vel.OBS_CONFIDENCE].isna().sum() == 20
    assert np.isnan(adata.obsm["velocity_umap"][:20]).all() and not np.isnan(adata.obsm["velocity_umap"][20:]).any()
    g = (await client.get(f"{API}/datasets/{did}/velocity/grid?embedding=X_umap")).json()
    assert g["n"] > 0


async def test_attach_endpoint_loom_then_run_then_attach_again_drops_the_run(client, dataset_manager, tmp_path):
    a = splicing_counts()
    bare = ad.AnnData(a.X.copy(), obs=a.obs.copy(), var=a.var.copy(), obsm={"X_umap": a.obsm["X_umap"]})
    did = _register(dataset_manager, bare, "vel5")
    write_loom(tmp_path / "v.loom", a)
    with open(tmp_path / "v.loom", "rb") as fh:
        r = await client.post(f"{API}/datasets/{did}/velocity/attach", files=[("files", ("v.loom", fh))])
    assert r.status_code == 200, r.text
    src = r.json()["source"]
    assert src["cells_matched"] == 300 and src["genes_by_id"] == 260 and src["kind"] == "loom"
    stored = Path(dataset_manager.data_dir) / "uploads" / did / "velocity" / src["stored_as"][0]
    assert stored.exists() and not os.access(stored, os.W_OK)
    await _run(client, did)
    s = (await client.get(f"{API}/datasets/{did}/velocity")).json()
    assert s["run"] is not None and s["source"]["file"] == "v.loom"

    # A file of other cells: refused, the run and counts stay, the rejected file is not kept.
    other = a.copy()
    other.obs_names = [barcode(i + 1000) + "-1" for i in range(300)]
    write_loom(tmp_path / "o.loom", other)
    with open(tmp_path / "o.loom", "rb") as fh:
        r = await client.post(f"{API}/datasets/{did}/velocity/attach", files=[("files", ("o.loom", fh))])
    assert r.status_code == 422 and "matches only 0" in r.json()["detail"]
    assert len(list(stored.parent.iterdir())) == 1
    assert (await client.get(f"{API}/datasets/{did}/velocity")).json()["run"] is not None

    write_loom(tmp_path / "w.loom", a, sample="again")
    with open(tmp_path / "w.loom", "rb") as fh:
        r = await client.post(f"{API}/datasets/{did}/velocity/attach", files=[("files", ("w.loom", fh))])
    assert r.status_code == 200
    s = (await client.get(f"{API}/datasets/{did}/velocity")).json()
    assert s["run"] is None and s["source"]["file"] == "w.loom"
    dataset_manager.evict(did)
    adata = (await dataset_manager.get_or_load_dataset(did)).adata
    assert vel.OBS_CONFIDENCE not in adata.obs.columns and "velocity_umap" not in adata.obsm
    assert vel.VAR_GAMMA not in adata.var.columns
    assert not ms.store_path(dataset_manager.data_dir, did, vel.RUN_STORE).exists()


async def test_plain_upload_moves_the_layers_and_a_plain_one_changes_nothing(client, dataset_manager, tmp_path):
    a = splicing_counts(60, 70)
    a.write_h5ad(tmp_path / "with.h5ad")
    with open(tmp_path / "with.h5ad", "rb") as fh:
        r = await client.post(f"{API}/datasets/upload", files={"file": ("with.h5ad", fh)})
    assert r.status_code == 201, r.text
    did = r.json()["id"]
    data = Path(dataset_manager.data_dir)
    assert (data / "uploads" / did / "with.h5ad").exists() and (data / "ingested" / did / "with.h5ad").exists()
    adata = (await dataset_manager.get_or_load_dataset(did)).adata
    assert not any(k in adata.layers for k in vi.LAYERS) and adata.n_obs == 60
    assert vi.has_store(data, did)
    s = (await client.get(f"{API}/datasets/{did}/velocity")).json()
    assert s["attached"] and s["source"]["file"] == "with.h5ad" and s["source"]["cells_now"] == 60
    assert ad.read_h5ad(data / "uploads" / did / "with.h5ad", backed="r").layers.keys() >= {"spliced", "unspliced"}

    plain = ad.AnnData(a.X.copy(), obs=a.obs.copy(), var=a.var.copy())
    plain.write_h5ad(tmp_path / "plain.h5ad")
    with open(tmp_path / "plain.h5ad", "rb") as fh:
        r = await client.post(f"{API}/datasets/upload", files={"file": ("plain.h5ad", fh)})
    did2 = r.json()["id"]
    assert not (data / "ingested" / did2).exists() and not vi.has_store(data, did2)
    assert vi.UNS_KEY not in (await dataset_manager.get_or_load_dataset(did2)).adata.uns


def test_delete_dataset_takes_the_store_with_it(dataset_manager):
    did, _a = dataset_with_store(dataset_manager, "vel6", 60, 70)
    assert vi.has_store(dataset_manager.data_dir, did)
    dataset_manager.remove_dataset(did)
    assert not vi.has_store(dataset_manager.data_dir, did)


async def test_large_dataset_runs_from_the_store_and_saves_in_place(client, dataset_manager):
    """X stays on disk: the run reads the store only; the save writes into the h5ad in place."""
    from tests import large_fixture

    a = splicing_counts()
    counts = ad.AnnData(a.X.copy(), obs=a.obs.copy(), var=pd.DataFrame(index=a.var_names))
    counts.var["mt"] = False
    counts.obs["total_counts"] = np.asarray(a.X.sum(axis=1)).ravel()
    counts.obs["n_genes_by_counts"] = np.asarray((a.X > 0).sum(axis=1)).ravel()
    counts.obs["pct_counts_mt"] = 0.0
    counts.obsm["X_umap"] = a.obsm["X_umap"]
    did = "largevel"
    path = large_fixture.register(dataset_manager.data_dir, did, counts)
    assert path
    holder = ad.AnnData(a.X.copy(), obs=a.obs[[]].copy(), var=a.var.copy(), layers={k: a.layers[k] for k in vi.LAYERS})
    vi.adopt_anndata(holder, did, dataset_manager.data_dir)
    adaptor = await dataset_manager.get_or_load_dataset(did)
    assert adaptor.is_large
    x_before = adaptor.adata.X
    events = await _run(client, did, group_by="clusters")
    run = dict(events).get("result", {}).get("run")
    assert run is not None, events[-2:]
    assert run["n_velocity_genes"] > 20
    dataset_manager.evict(did)
    adaptor = await dataset_manager.get_or_load_dataset(did)
    assert adaptor.is_large and vel.OBS_CONFIDENCE in adaptor.adata.obs.columns
    assert vel.VAR_GAMMA in adaptor.adata.var.columns and "velocity_umap" in adaptor.adata.obsm
    del x_before
    g = (await client.get(f"{API}/datasets/{did}/velocity/grid?embedding=X_umap")).json()
    assert g["n"] > 0
    p = (await client.get(f"{API}/datasets/{did}/velocity/genes")).json()
    assert p["n_genes"] == run["n_genes_fit"]
