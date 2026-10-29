"""Spatially variable genes (core/spatial_autocorr.py, api/v1/spatial_autocorr.py).

Expected values: squidpy 1.6.5's defaults on the synthetic datasets in
tests/fixtures/spatial_autocorr (squidpy_moran.py writes them; squidpy is not in the image).
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from scview.core import provenance, replay_script, results_store
from scview.core import spatial_autocorr as sa

API = "/api/v1"
FIX = Path(__file__).parent / "fixtures" / "spatial_autocorr"

_spec = importlib.util.spec_from_file_location("spatial_synthetic", FIX / "synthetic.py")
synthetic = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(synthetic)


@pytest.fixture(scope="module")
def squidpy_ref() -> pd.DataFrame:
    return pd.read_csv(FIX / "squidpy_moran.csv")


@pytest.fixture(scope="module")
def squidpy_degrees() -> pd.DataFrame:
    return pd.read_csv(FIX / "squidpy_degrees.csv")


def _compare(r: sa.Result, ref: pd.DataFrame) -> None:
    t = r.table
    assert list(t["gene"]) == list(ref["gene"])  # the same genes, in squidpy's order (I high first)
    np.testing.assert_allclose(t["I"], ref["I"], rtol=0, atol=1e-6)
    np.testing.assert_allclose(t["var_I"], ref["var_I"], rtol=1e-6)
    # squidpy's p is 1 - cdf(z) (0 beyond z ~ 8.3); here the normal tail: equal to 1e-12.
    np.testing.assert_allclose(t["pval"], ref["pval"], rtol=1e-6, atol=1e-12)
    np.testing.assert_allclose(t["pval_adj"], ref["pval_adj"], rtol=1e-6, atol=1e-12)


@pytest.mark.parametrize("name,make,kind", [("visium", synthetic.visium, "grid"),
                                            ("cells", synthetic.cells, "knn")])
def test_equals_squidpy_defaults(name, make, kind, squidpy_ref, squidpy_degrees):
    adata = make()
    r = sa.run(adata)
    assert r.neighbors == {"kind": kind, "k": 6}
    assert r.libraries == list(adata.obs["library_id"].cat.categories)
    _compare(r, squidpy_ref[squidpy_ref["dataset"] == name])
    # The graph itself: each spot's neighbour count as squidpy's, none across sections.
    deg = squidpy_degrees[squidpy_degrees["dataset"] == name]
    W = sa.spatial_weights(adata.obsm["spatial"], sa.library_labels(adata), kind == "grid")
    assert np.array_equal(np.diff(W.indptr), deg["degree"].to_numpy())
    assert deg["lib_cross"].sum() == 0
    np.testing.assert_allclose(np.asarray(W.sum(axis=1)).ravel()[np.diff(W.indptr) > 0], 1.0)


def test_grid_keeps_sections_apart_and_edge_spots_lose_neighbours():
    adata = synthetic.visium()
    libs = sa.library_labels(adata)
    W = sp.coo_matrix(sa.spatial_weights(adata.obsm["spatial"], libs, True))
    assert (libs[W.row] == libs[W.col]).all()
    # The two sections lie on top of each other: one pooled graph would join them.
    pooled = sp.coo_matrix(sa.spatial_weights(adata.obsm["spatial"], np.zeros(adata.n_obs), True))
    assert (libs[pooled.row] != libs[pooled.col]).any()
    deg = np.diff(sa.spatial_weights(adata.obsm["spatial"], libs, True).indptr)
    assert deg.max() == 6 and (deg < 6).any()
    far = adata.obs_names.get_loc("spot" + str(int(np.flatnonzero(libs == "A")[-1])))
    assert deg[far] == 0  # the lone spot: no neighbour within reach, a zero row


def test_isolated_spot_counts_in_n_and_is_noted():
    r = sa.run(synthetic.visium())
    assert r.n_obs == synthetic.visium().n_obs
    assert any("no neighbour" in n for n in r.notes)


def test_hvg_choice_all_and_list_and_constant_genes():
    adata = synthetic.visium()
    hv = sa.run(adata, genes="hvg")
    assert hv.genes_choice == "hvg" and hv.n_genes == 20
    full = sa.run(adata, genes="all")
    assert full.genes_choice == "all" and full.n_genes == 29 and full.n_constant == 1
    assert "CONST" not in set(full.table["gene"]) and any("same value" in n for n in full.notes)
    # A gene's I does not depend on the others; BH does.
    a = hv.table.set_index("gene")["I"]
    b = full.table.set_index("gene").loc[a.index, "I"]
    np.testing.assert_allclose(a, b, atol=1e-12)
    some = sa.run(adata, genes=["G01", "G02", "NOPE"])
    assert some.genes_choice == "list" and set(some.table["gene"]) == {"G01", "G02"}
    assert any("not in the data" in n for n in some.notes)
    with pytest.raises(sa.SpatialAutocorrError):
        sa.run(adata, genes=["NOPE"])
    with pytest.raises(sa.SpatialAutocorrError):
        sa.run(adata, genes="most")


def test_hvg_without_variable_genes_falls_back_to_all():
    adata = synthetic.visium()
    del adata.var["highly_variable"]
    r = sa.run(adata, genes="hvg")
    assert r.genes_choice == "all" and r.n_genes == 29
    assert any("no variable genes" in n for n in r.notes)


def test_sparse_equals_dense_and_blocks_do_not_matter(monkeypatch):
    adata = synthetic.visium()
    dense = sa.run(adata, genes="all").table
    adata.X = sp.csr_matrix(adata.X.astype(np.float32))
    monkeypatch.setattr(sa, "BLOCK_BYTES", 8 * 30 * 7 * 13)  # 13 spots per row block
    sparse = sa.run(adata, genes="all").table
    assert list(sparse["gene"]) == list(dense["gene"])
    for c in ("I", "pval", "pval_adj"):
        np.testing.assert_allclose(sparse[c], dense[c], rtol=1e-6, atol=1e-9)


def test_one_section_alone():
    adata = synthetic.visium()
    r = sa.run(adata, library_id="B")
    sub = adata[adata.obs["library_id"] == "B"].copy()
    alone = sa.run(sub)
    assert r.libraries == ["B"] and r.n_obs == sub.n_obs
    np.testing.assert_allclose(r.table["I"], alone.table["I"], atol=1e-12)
    with pytest.raises(sa.SpatialAutocorrError):
        sa.run(adata, library_id="Z")


def test_no_spatial_is_refused():
    with pytest.raises(sa.SpatialAutocorrError):
        sa.run(ad.AnnData(np.ones((5, 3))))


def test_duplicate_positions_drop_like_sklearn():
    # Three points at one place: each one's own hit need not come first.
    coords = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [5.0, 0.0]])
    rows, cols = sa.knn_edges(coords, 2, grid=False)
    assert (rows != cols).all() and len(rows) == 10


def test_bh_matches_the_textbook():
    p = np.array([0.01, 0.04, 0.03, 0.2])
    np.testing.assert_allclose(sa.bh_adjust(p), [0.04, 0.16 / 3, 0.16 / 3, 0.2])


def test_estimate_grows_with_spots_and_genes():
    assert sa.estimate_seconds(20_000, 20_000) > sa.LONG_SECONDS > sa.estimate_seconds(4_000, 2_000)


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------


def _register(dm, a: ad.AnnData, did: str) -> str:
    d = Path(dm.data_dir) / "uploads" / did
    d.mkdir(parents=True)
    (d / "metadata.json").write_text(json.dumps({"id": did, "name": did, "status": "ready"}))
    a.write_h5ad(d / "data.h5ad")
    return did


async def test_api_run_then_get_csv_and_saved(client, dataset_manager, squidpy_ref):
    did = _register(dataset_manager, synthetic.visium(), "sp1")
    assert (await client.get(f"{API}/datasets/{did}/spatial/autocorr")).status_code == 404
    got = await client.get(f"{API}/datasets/{did}/spatial/autocorr", params={"allow_empty": True})
    assert got.status_code == 200 and got.json() is None
    r = await client.post(f"{API}/datasets/{did}/spatial/autocorr", json={"genes": "hvg"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["method"] == "moran" and body["neighbors"] == {"kind": "grid", "k": 6}
    assert body["n_genes"] == 20 and body["libraries"] == ["A", "B"] and body["unit"] == "spot"
    rows = body["rows"]
    ref = squidpy_ref[squidpy_ref["dataset"] == "visium"]
    assert [x["gene"] for x in rows] == list(ref["gene"])
    assert set(rows[0]) == {"gene", "I", "expected_I", "var_I", "z", "pval", "pval_adj"}
    np.testing.assert_allclose([x["I"] for x in rows], ref["I"], atol=1e-5)
    assert body["n_significant"] >= 2

    got = (await client.get(f"{API}/datasets/{did}/spatial/autocorr")).json()
    assert got["result_id"] == body["result_id"] and got["rows"] == rows
    csv = await client.get(f"{API}/datasets/{did}/spatial/autocorr/{body['result_id']}.csv")
    assert csv.status_code == 200 and csv.text.startswith("gene,I,")
    assert len(csv.text.strip().splitlines()) == 21

    # Saved with the dataset: squidpy's uns["moranI"] and a History step; GET after a reload.
    results_store.flush()
    dataset_manager.evict(did)
    adata = (await dataset_manager.get_or_load_dataset(did)).adata
    step = provenance.read_provenance(adata)["history"][-1]
    assert step["step"] == "spatial_autocorr" and step["params"]["neighbors"] == "grid"
    assert list(adata.uns["moranI"].index) == list(ref["gene"])
    again = (await client.get(f"{API}/datasets/{did}/spatial/autocorr")).json()
    assert again["result_id"] == body["result_id"]
    ds = (await client.get(f"{API}/datasets/{did}")).json()
    assert "moranI" not in ds["available_embeddings"]

    # A later step (labels, a kept layout) leaves it; other spots (filtered) do not.
    adaptor = await dataset_manager.get_or_load_dataset(did)
    provenance.record_step(adaptor.adata, step="label_cells", tool="test", params={})
    sa._last.clear()
    assert sa.last(adaptor).id == body["result_id"]

    class _Filtered:
        adata = adaptor.adata[:-5].copy()
        h5ad_path = adaptor.h5ad_path

    sa._last.clear()
    assert sa.last(_Filtered()) is None


async def test_api_refuses_without_spatial_and_bad_section(client, dataset_manager):
    did = _register(dataset_manager, ad.AnnData(np.ones((20, 4), dtype=np.float32)), "sp2")
    r = await client.post(f"{API}/datasets/{did}/spatial/autocorr", json={})
    assert r.status_code == 422 and "no spatial coordinates" in r.json()["detail"]
    did = _register(dataset_manager, synthetic.visium(), "sp3")
    r = await client.post(f"{API}/datasets/{did}/spatial/autocorr", json={"library_id": "Z"})
    assert r.status_code == 422 and "section" in r.json()["detail"]
    r = await client.post(f"{API}/datasets/{did}/spatial/autocorr", json={"genes": "most"})
    assert r.status_code == 422
    r = await client.post(f"{API}/datasets/{did}/spatial/autocorr", json={"layer": "nope"})
    assert r.status_code == 404


async def test_api_hvg_fallback_note(client, dataset_manager):
    a = synthetic.cells()
    did = _register(dataset_manager, a, "sp4")
    body = (await client.post(f"{API}/datasets/{did}/spatial/autocorr", json={"genes": "hvg"})).json()
    assert body["genes"] == "all" and body["neighbors"]["kind"] == "knn" and body["unit"] == "cell"
    assert any("no variable genes" in n for n in body["notes"])


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------


def test_replay_code_is_exact_and_runs(tmp_path):
    inp = synthetic.visium()
    out = inp.copy()
    out.X = sp.csr_matrix(out.X)
    inp.X = sp.csr_matrix(inp.X)
    r = sa.run(out, genes="all")
    sa.record(out, r, "all", None, None)
    assert [d["status"] for d in replay_script.describe(out)] == ["exact"]
    src, script, dst = tmp_path / "input.h5ad", tmp_path / "replay.py", tmp_path / "replayed.h5ad"
    inp.write_h5ad(src)
    script.write_text(replay_script.build_script(out, dataset_id="t", dataset_name="test"))
    proc = subprocess.run([sys.executable, str(script), str(src), str(dst)],
                          capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr[-3000:]
    rep = ad.read_h5ad(dst).uns["moranI"]
    assert list(rep.index) == list(r.table["gene"])
    np.testing.assert_allclose(rep["I"], r.table["I"], atol=1e-12)
    np.testing.assert_allclose(rep["pval_norm_fdr_bh"], r.table["pval_adj"], rtol=1e-9, atol=1e-15)


def test_replay_of_an_unreadable_record_is_a_comment():
    snip = replay_script._spatial_autocorr({"genes": "all"}, replay_script._Ctx(adata=None))
    assert snip.status == replay_script.NOT_REPLAYABLE
