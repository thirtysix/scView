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


@pytest.fixture(scope="module")
def squidpy_sections() -> pd.DataFrame:
    return pd.read_csv(FIX / "squidpy_moran_sections.csv")


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
def test_equals_squidpy_defaults(name, make, kind, squidpy_ref, squidpy_degrees, squidpy_sections):
    adata = make()
    r = sa.run(adata)
    assert r.neighbors == {"kind": kind, "k": 6}
    assert r.libraries == list(adata.obs["library_id"].cat.categories)
    assert r.centring == "section"
    # Each section's own I, z and p: squidpy run on that section alone.
    for lib in r.libraries:
        ref = squidpy_sections[(squidpy_sections["dataset"] == name) & (squidpy_sections["library_id"] == lib)]
        mine = r.by_library[r.by_library["library_id"] == lib].set_index("gene").loc[ref["gene"]]
        assert (mine["n_obs"] == ref["n_obs"].to_numpy()).all()
        np.testing.assert_allclose(mine["I"], ref["I"], rtol=0, atol=1e-6)
        np.testing.assert_allclose(mine["pval"], ref["pval"], rtol=1e-6, atol=1e-12)
        np.testing.assert_allclose(mine["pval_adj"], ref["pval_adj"], rtol=1e-6, atol=1e-12)
    # The per-section sums added up and centred on one mean: squidpy's pooled I (sections
    # pooled with one global mean), so the sums themselves are squidpy's.
    libs = sa.library_labels(adata)
    names = [str(x) for x in pd.unique(libs)]
    codes = pd.Index(names).get_indexer(libs)
    W = sa.spatial_weights(adata.obsm["spatial"], libs, kind == "grid")
    idx, _, _ = sa.select_genes(adata, adata.var_names, "hvg")
    sums = sa.section_sums(W, np.asarray(adata.X)[:, idx], codes, len(names))
    total = (*(x.sum(axis=0) for x in sums[:5]), sums[5].all(axis=0))  # no gene differs only between sections
    glob, _ = sa.moran_table(W, total, adata.var_names[idx])
    _compare(sa.Result(id="", dataset_id="", n_obs=0, n_genes=0, neighbors={}, libraries=[], genes_choice="",
                       matrix_source="", table=glob), squidpy_ref[squidpy_ref["dataset"] == name])
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


@pytest.mark.parametrize("name,make,lib", [("visium", synthetic.visium, "A"), ("cells", synthetic.cells, "D")])
def test_a_one_section_dataset_is_unchanged_and_equals_squidpy(name, make, lib, squidpy_sections):
    one = make()
    one = one[one.obs["library_id"] == lib].copy()
    r = sa.run(one)
    assert r.centring == "global" and r.by_library is None and r.sections == []
    # Bitwise what the code before per-section centring gave: one global mean.
    W = sa.spatial_weights(one.obsm["spatial"], sa.library_labels(one), sa.is_grid(one))
    idx, _, _ = sa.select_genes(one, one.var_names, "hvg")
    before, _ = sa.morans_i(W, np.asarray(one.X)[:, idx], one.var_names[idx],
                            sa.spatial_order(one.obsm["spatial"], sa.library_labels(one)))
    pd.testing.assert_frame_equal(r.table, before, check_exact=True)
    ref = squidpy_sections[(squidpy_sections["dataset"] == name) & (squidpy_sections["library_id"] == lib)]
    _compare(r, ref)


def test_pooled_i_is_within_section_and_ignores_a_shift_between_sections():
    """Kuppe 2022, 8 slides: pooled MALAT1 0.61 against at most 0.48 on any one slide. A gene
    that is only higher on one slide (no pattern inside either) is not spatial."""
    adata = synthetic.visium()
    libs = sa.library_labels(adata)
    X = np.asarray(adata.X).copy()
    noise = [i for i, g in enumerate(adata.var_names) if g.startswith("G") and int(g[1:]) % 6 >= 4]
    X[:, noise] += np.where(libs == "B", 3.0, 0.0)[:, None]  # a pure between-section shift
    adata.X = X
    r = sa.run(adata, genes="all")
    pooled = r.table.set_index("gene")["I"]
    per = r.by_library.pivot(index="gene", columns="library_id", values="I")
    names = adata.var_names[noise]
    assert (pooled[names] <= per.loc[names].max(axis=1) + 1e-9).all()
    assert pooled[names].abs().max() < 0.2
    # One global mean (squidpy's pooled I) calls the shift spatial.
    W = sa.spatial_weights(adata.obsm["spatial"], libs, True)
    glob, _ = sa.morans_i(W, X, adata.var_names)
    assert glob.set_index("gene")["I"][names].min() > 0.8
    # A shift does not change a section's own I, nor the pooled I.
    base = sa.run(synthetic.visium(), genes="all")
    np.testing.assert_allclose(pooled[names], base.table.set_index("gene")["I"][names], atol=1e-9)
    assert any("each centred on its own mean" in n for n in r.notes)


def test_centred_moments_match_a_simulation():
    """E[I] and Var[I] under the null for values centred within each section (Cliff and Ord):
    against 4,000 draws of independent normal values with section means far apart."""
    adata = synthetic.visium()
    libs = sa.library_labels(adata)
    codes = pd.Index(["A", "B"]).get_indexer(libs)
    W = sa.spatial_weights(adata.obsm["spatial"], libs, True)
    rng = np.random.default_rng(3)
    M = rng.normal(size=(adata.n_obs, 4000)) + np.where(codes == 1, 50.0, 0.0)[:, None]
    table, _, _ = sa.morans_i_sections(W, M, np.arange(4000).astype(str), codes, ["A", "B"])
    e, v = table["expected_I"].iloc[0], table["var_I"].iloc[0]
    assert abs(table["I"].mean() - e) < 4 * np.sqrt(v / 4000)
    assert abs(table["I"].var() / v - 1) < 0.1
    # One section: squidpy's moments.
    n = adata.n_obs
    s0, s1, s2 = sa.weight_moments(W)
    e1, v1 = sa.centred_moments(W, np.zeros(n, dtype=int), 1)
    assert e1 == pytest.approx(-1 / (n - 1), rel=1e-12)
    assert v1 == pytest.approx((n * n * s1 - n * s2 + 3 * s0 * s0) / ((n - 1) * (n + 1) * s0 * s0) - e1 ** 2,
                               rel=1e-10)


def test_section_sums_blocks_and_sparse_do_not_matter(monkeypatch):
    adata = synthetic.visium()
    libs = sa.library_labels(adata)
    codes = pd.Index(["A", "B"]).get_indexer(libs)
    W = sa.spatial_weights(adata.obsm["spatial"], libs, True)
    X = np.asarray(adata.X)
    whole = sa.section_sums(W, X, codes, 2)
    monkeypatch.setattr(sa, "BLOCK_BYTES", 8 * 30 * 7 * 13)
    blocks = sa.section_sums(W, sp.csr_matrix(X), codes, 2, sa.spatial_order(adata.obsm["spatial"], libs))
    genes = sa.section_sums_by_genes(W, lambda a, b: sp.csr_matrix(X[:, a:b]), X.shape[1], codes, 2, block=7)
    for x, y, z in zip(whole[:5], blocks[:5], genes[:5], strict=True):
        np.testing.assert_allclose(x, y, rtol=1e-10, atol=1e-9)
        np.testing.assert_allclose(x, z, rtol=1e-10, atol=1e-9)
    np.testing.assert_array_equal(whole[5], blocks[5])
    np.testing.assert_array_equal(whole[5], genes[5])


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
    assert set(x["gene"] for x in rows) == set(ref["gene"])
    assert set(rows[0]) == {"gene", "I", "expected_I", "var_I", "z", "pval", "pval_adj", "by_library"}
    assert body["centring"] == "section" and body["library_id"] == "all"
    assert body["sections"] == [{"library_id": "A", "n_obs": 288}, {"library_id": "B", "n_obs": 296}]
    assert set(rows[0]["by_library"]) == {"A", "B"}
    assert set(rows[0]["by_library"]["A"]) == {"I", "z", "pval", "pval_adj"}
    want = sa.run(synthetic.visium()).table
    assert [x["gene"] for x in rows] == list(want["gene"])
    np.testing.assert_allclose([x["I"] for x in rows], want["I"], atol=1e-5)
    assert body["n_significant"] >= 2

    got = (await client.get(f"{API}/datasets/{did}/spatial/autocorr")).json()
    assert got["result_id"] == body["result_id"] and got["rows"] == rows
    csv = await client.get(f"{API}/datasets/{did}/spatial/autocorr/{body['result_id']}.csv")
    assert csv.status_code == 200 and csv.text.startswith("gene,I,")
    assert len(csv.text.strip().splitlines()) == 21
    assert csv.text.splitlines()[0].endswith("I_A,I_B,pval_adj_A,pval_adj_B")

    # Saved with the dataset: squidpy's uns["moranI"] and a History step; GET after a reload.
    results_store.flush()
    dataset_manager.evict(did)
    adata = (await dataset_manager.get_or_load_dataset(did)).adata
    step = provenance.read_provenance(adata)["history"][-1]
    assert step["step"] == "spatial_autocorr" and step["params"]["neighbors"] == "grid"
    assert list(adata.uns["moranI"].index) == list(want["gene"])
    assert step["params"]["centring"] == "section"
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


async def test_api_runs_per_section_coexist_in_the_history(client, dataset_manager):
    """Kuppe 2022: Moran's I one section at a time over 8 sections kept the last 4, and a run on
    section B marked the run on section A superseded. A run is its section and its genes."""
    a = synthetic.visium()
    # 8 sections and the pooled run must all stay: 4 copies of A and B, side by side.
    parts = []
    for i in range(4):
        part = a.copy()
        part.obs_names = [f"{n}_{i}" for n in part.obs_names]
        part.obs["library_id"] = pd.Categorical([f"{x}{i}" for x in part.obs["library_id"]])
        parts.append(part)
    big = ad.concat(parts)
    big.obsm["spatial"] = np.vstack([p.obsm["spatial"] for p in parts])
    big.uns["spatial"] = {lib: {"scalefactors": {"spot_diameter_fullres": 89.0}, "metadata": {}}
                          for lib in big.obs["library_id"].cat.categories}
    big.uns["scview_spatial"] = {"technology": "visium", "unit": "spot"}
    big.var["highly_variable"] = a.var["highly_variable"].to_numpy()
    did = _register(dataset_manager, big, "sphist")
    libs = list(big.obs["library_id"].cat.categories)
    assert len(libs) == 8
    ids = {}
    for lib in [*libs, None]:
        r = await client.post(f"{API}/datasets/{did}/spatial/autocorr", json={"genes": "hvg", "library_id": lib})
        assert r.status_code == 200, r.text
        ids[lib or "all"] = r.json()["result_id"]
        assert r.json()["library_id"] == (lib or "all")
    again = (await client.post(f"{API}/datasets/{did}/spatial/autocorr", json={"genes": "hvg", "library_id": libs[0]})).json()
    results_store.flush()
    listed = (await client.get(f"{API}/datasets/{did}/results")).json()
    entries = listed if isinstance(listed, list) else listed.get("results", listed.get("entries"))
    moran = {e["id"]: e for e in entries if e["kind"] == sa.KIND}
    assert set(ids.values()) <= set(moran)  # nothing dropped
    primaries = {e["id"] for e in moran.values() if e["status"] == "primary"}
    assert primaries == (set(ids.values()) - {ids[libs[0]]}) | {again["result_id"]}
    assert moran[ids[libs[0]]]["status"] == "superseded"  # only by the same section again
    # Each section's last run, and the pooled one, by section.
    for lib in libs[1:]:
        got = (await client.get(f"{API}/datasets/{did}/spatial/autocorr", params={"library_id": lib})).json()
        assert got["result_id"] == ids[lib] and got["libraries"] == [lib] and got["centring"] == "global"
    got = (await client.get(f"{API}/datasets/{did}/spatial/autocorr", params={"library_id": "all"})).json()
    assert got["result_id"] == ids["all"] and len(got["sections"]) == 8
    got = (await client.get(f"{API}/datasets/{did}/spatial/autocorr", params={"library_id": "nope",
                                                                            "allow_empty": True})).json()
    assert got is None


def test_result_key_names_section_and_genes():
    a = synthetic.visium()
    hv, full, one = sa.run(a), sa.run(a, genes="all"), sa.run(a, library_id="B")
    lst, lst2 = sa.run(a, genes=["G01", "G02"]), sa.run(a, genes=["G02", "G01"])
    keys = [sa.result_key(x) for x in (hv, full, one, lst)]
    assert keys[:3] == ["all|hvg", "all|all", "B|hvg"] and keys[3].startswith("all|list:")
    assert sa.result_key(lst2) == keys[3]
    sub = a[a.obs["library_id"] == "A"].copy()
    assert sa.section_of(sa.run(sub)) == "all"  # a one-section dataset: its whole


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


def test_replay_of_a_record_before_section_centring_keeps_one_mean():
    snip = replay_script._spatial_autocorr(
        {"genes": "hvg", "matrix_source": "log-normalised X", "neighbors": "grid", "n_neighs": 6, "library_id": None},
        replay_script._Ctx(adata=None))
    assert not any("centre=" in line for line in snip.code)
    snip = replay_script._spatial_autocorr(
        {"genes": "hvg", "matrix_source": "log-normalised X", "neighbors": "grid", "n_neighs": 6, "library_id": None,
         "centring": "section"}, replay_script._Ctx(adata=None))
    assert any('centre="section"' in line for line in snip.code)


def test_replay_of_an_unreadable_record_is_a_comment():
    snip = replay_script._spatial_autocorr({"genes": "all"}, replay_script._Ctx(adata=None))
    assert snip.status == replay_script.NOT_REPLAYABLE
