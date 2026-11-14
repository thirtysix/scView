"""ATAC endpoints (api/v1/atac.py): state, run, WNN, overlay, differential, export; large datasets."""

from __future__ import annotations

import io
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from scview.api.v1 import atac as atac_api
from scview.core import atac, atac_import, gene_annotation, large_layout, modality_store, provenance, results_store
from tests import multiome_fixture as mf
from tests.test_hvg_sensitivity import _register

API = "/api/v1"


def _events(text: str) -> list[tuple[str, dict]]:
    out = []
    for block in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines() if not line.startswith(":"))
        if "event" in lines:
            out.append((lines["event"], json.loads(lines["data"])))
    return out


def _dataset(n_per: int = 40) -> tuple[ad.AnnData, sp.csr_matrix, pd.DataFrame]:
    g, p, barcodes, table = mf.matrices(n_per=n_per)
    a = ad.AnnData(X=g, obs=pd.DataFrame({"group": pd.Categorical(["A"] * n_per + ["B"] * n_per)}, index=barcodes),
                   var=pd.DataFrame({"gene_ids": [x[1] for x in mf.GENES]}, index=[x[0] for x in mf.GENES]))
    a.obs[atac_import.OBS_TOTAL] = np.asarray(p.sum(axis=1), dtype=np.float32).ravel()
    a.obs[atac_import.OBS_N_PEAKS] = np.diff(p.indptr).astype(np.float32)
    atac_import.write_info(a, {"n_peaks": len(table), "genome": "GRCh38", "nnz": int(p.nnz), "computed": ["peak counts"]})
    rng = np.random.default_rng(0)
    a.obsm["X_pca"] = (np.log1p(g.toarray()) + rng.normal(0, 0.1, g.shape)).astype(np.float32)
    a.uns["neighbors"] = {"connectivities_key": "connectivities", "distances_key": "distances",
                          "params": {"n_neighbors": 15, "method": "umap", "use_rep": "X_pca"}}
    return a, p, table


def _seed(data_dir, did: str, p: sp.csr_matrix, barcodes, table: pd.DataFrame) -> None:
    modality_store.write(atac.store_path(data_dir, did), {atac.MATRIX: p}, barcodes, table["name"].tolist(),
                         atac_import.feature_attrs(table["name"].tolist()), {"genome": "GRCh38"})


@pytest.fixture()
def annotated(dataset_manager, tmp_path):
    gtf = tmp_path / "tiny.gtf"
    gtf.write_text(mf.gtf_text())
    gene_annotation.clear_cache()
    gene_annotation.seed_from(gtf, dataset_manager.data_dir, "GRCh38")
    yield
    gene_annotation.clear_cache()


def _register_multiome(dm, did: str = "at") -> tuple[str, sp.csr_matrix, pd.DataFrame, ad.AnnData]:
    a, p, table = _dataset()
    _register(dm, a, did)
    _seed(dm.data_dir, did, p, list(a.obs_names), table)
    return did, p, table, a


async def test_state_without_atac_is_a_sentence_or_a_404(client, dataset_manager):
    a = ad.AnnData(X=np.ones((5, 3), dtype=np.float32))
    did = _register(dataset_manager, a, "rna")
    r = await client.get(f"{API}/datasets/{did}/atac?allow_empty=true")
    assert r.status_code == 200 and r.json()["available"] is False and "no ATAC peaks" in r.json()["reason"]
    assert (await client.get(f"{API}/datasets/{did}/atac")).status_code == 404
    info = (await client.get(f"{API}/datasets/{did}")).json()
    assert info["atac"] is None
    r = await client.post(f"{API}/datasets/{did}/atac/run-stream", json={})
    assert r.status_code == 404 and "no ATAC peaks" in r.json()["detail"]


async def test_plain_upload_refuses_peaks_and_no_genes(client, dataset_manager, tmp_path):
    a = mf.as_anndata()
    only = a[:, a.var["feature_types"] == "Peaks"].copy()
    path = tmp_path / "only_peaks.h5ad"
    only.write_h5ad(path)
    r = await client.post(f"{API}/datasets/upload", files={"file": ("only_peaks.h5ad", path.read_bytes())})
    assert r.status_code == 422 and "ATAC-only data is not supported" in r.json()["detail"]
    assert not any((Path(dataset_manager.data_dir) / "uploads").iterdir())


async def test_plain_upload_of_a_multiome_h5ad_keeps_its_store(client, dataset_manager, tmp_path):
    path = tmp_path / "m.h5ad"
    mf.as_anndata().write_h5ad(path)
    r = await client.post(f"{API}/datasets/upload", files={"file": ("m.h5ad", path.read_bytes())})
    assert r.status_code == 201, r.text
    did = r.json()["id"]
    info = (await client.get(f"{API}/datasets/{did}")).json()
    assert info["atac"]["n_peaks"] == 72 and info["atac"]["genome"] == "GRCh38" and info["n_genes"] == 4
    state = (await client.get(f"{API}/datasets/{did}/atac")).json()
    assert state["available"] and state["info"]["n_peaks"] == 72 and state["plan"]["n_peaks"] == 72
    assert state["qc_columns"] == [atac_import.OBS_TOTAL, atac_import.OBS_N_PEAKS]


async def test_run_saves_serves_and_keeps_the_rna_graph(client, dataset_manager, annotated):
    did, p, table, a = _register_multiome(dataset_manager)
    state = (await client.get(f"{API}/datasets/{did}/atac?groupby=group")).json()
    assert state["available"] and state["lsi"] is None and state["running"] is None
    assert state["plan"]["estimate_seconds"] > 0 and state["plan"]["estimate_bytes"] > 0
    assert state["wnn_plan"] is None and "Run ATAC analysis first" in state["wnn_reason"]

    r = await client.post(f"{API}/datasets/{did}/atac/run-stream", json={"n_lsi": 10, "dims_to": 8, "resolution": 0.5})
    assert r.status_code == 200, r.text
    events = _events(r.text)
    kinds = [e for e, _ in events]
    assert kinds[0] == "start" and kinds[-2:] == ["result", "complete"], kinds
    assert [d["stage"] for e, d in events if e == "stage"] == [s for s, _ in atac.STAGES]
    result = next(d for e, d in events if e == "result")
    assert result["lsi"]["n_selected"] > 0 and len(result["lsi"]["depth_r"]) == 10
    assert result["activity"]["available"] and result["activity"]["n_genes"] == 4
    assert "peak-based" in result["activity"]["caption"].lower()
    dims = result["lsi"]["dims_used"]
    assert dims[-1] == 8 and dims[0] == (2 if result["lsi"]["dropped_first"] else 1)

    adaptor = await dataset_manager.get_or_load_dataset(did)
    assert adaptor.adata.obsm[atac.OBSM_LSI].shape == (80, 10)
    assert atac.OBSM_UMAP in adaptor.adata.obsm and atac.OBS_CLUSTERS in adaptor.adata.obs.columns
    assert adaptor.adata.uns["neighbors"]["params"]["use_rep"] == "X_pca"  # the RNA graph is untouched
    assert "connectivities" not in adaptor.adata.obsp and atac.CONN_KEY in adaptor.adata.obsp
    steps = [h["step"] for h in provenance.read_provenance(adaptor.adata)["history"]]
    assert "atac_lsi" in steps
    info = (await client.get(f"{API}/datasets/{did}")).json()
    assert atac.OBSM_UMAP in info["available_embeddings"] and atac.OBSM_LSI not in info["available_embeddings"]
    assert any(c["name"] == atac.OBS_CLUSTERS for c in info["obs_columns"])
    assert "LSI" in info["atac"]["computed"] and "gene activity" in info["atac"]["computed"]
    emb = await client.get(f"{API}/datasets/{did}/embeddings/{atac.OBSM_UMAP}")
    assert emb.status_code == 200

    # Overlay, its violin, RNA beside it.
    r = await client.get(f"{API}/datasets/{did}/atac/activity", params={"gene": "GENEA", "groupby": "group"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["values"]) == 80 and set(body["violin"]) == {"A", "B"} and body["r"] > 0.5
    assert np.mean(body["violin"]["A"]) > np.mean(body["violin"]["B"])
    assert len(body["rna"]["rna"]) == 80 and "Peak-based" in body["caption"]
    r = await client.get(f"{API}/datasets/{did}/atac/activity", params={"gene": "NOPE"})
    assert r.status_code == 404 and "no gene activity" in r.json()["detail"]

    # Peaks, paged.
    r = await client.get(f"{API}/datasets/{did}/atac/peaks", params={"limit": 5, "sort": "total"})
    page = r.json()
    assert page["total"] == 72 and len(page["rows"]) == 5
    assert page["rows"][0]["total"] >= page["rows"][-1]["total"]
    assert {"nearest_gene", "distance_to_tss", "peak_class", "selected"} <= set(page["rows"][0])
    r = await client.get(f"{API}/datasets/{did}/atac/peaks", params={"search": "GENEA", "limit": 50})
    assert r.json()["total"] >= 2 and all(x["nearest_gene"] == "GENEA" for x in r.json()["rows"])
    assert (await client.get(f"{API}/datasets/{did}/atac/peaks", params={"sort": "x"})).status_code == 422

    # QC columns.
    qc = (await client.get(f"{API}/datasets/{did}/atac/qc", params={"groupby": "group"})).json()
    assert [c["key"] for c in qc["columns"]] == [atac_import.OBS_TOTAL, atac_import.OBS_N_PEAKS]
    assert qc["group_order"] == ["A", "B"] and len(qc["groups"]) == 80

    # WNN on it.
    state = (await client.get(f"{API}/datasets/{did}/atac?groupby=group")).json()
    assert state["wnn_plan"]["lsi_dims"][0] == 2 and state["wnn_plan"]["lsi_dims"][-1] == 10
    r = await client.post(f"{API}/datasets/{did}/atac/wnn/run-stream", json={})
    events = _events(r.text)
    assert [e for e, _ in events][-2:] == ["result", "complete"], events[-1]
    assert [d["stage"] for e, d in events if e == "stage"] == [s for s, _ in atac.WNN_STAGES]
    adaptor = await dataset_manager.get_or_load_dataset(did)
    w = adaptor.adata.obs[atac.OBS_WNN_WEIGHT].to_numpy()
    assert ((w >= 0) & (w <= 1)).all() and atac.OBSM_WNN in adaptor.adata.obsm
    assert atac.OBS_WNN_CLUSTERS in adaptor.adata.obs.columns
    state = (await client.get(f"{API}/datasets/{did}/atac?groupby=group")).json()
    assert [row["group"] for row in state["wnn"]["by_group"]["rows"]] == ["A", "B"]
    assert state["wnn"]["stale"] is False
    info = (await client.get(f"{API}/datasets/{did}")).json()
    assert atac.OBSM_WNN in info["available_embeddings"]


async def test_differential_stream_kept_by_id_with_csv(client, dataset_manager, annotated):
    did, _p, table, _a = _register_multiome(dataset_manager)
    r = await client.post(f"{API}/datasets/{did}/atac/differential", json={"column": "group"})
    assert r.status_code == 200, r.text
    events = _events(r.text)
    assert [e for e, _ in events][0] == "start" and [e for e, _ in events][-2:] == ["result", "complete"]
    assert any(e == "progress" and d["step"] == "test" for e, d in events)
    result = next(d for e, d in events if e == "result")
    assert result["groups"] == ["A", "B"] and result["n_tested"]["A"] > 0 and result["n_significant"]["A"] >= 6
    row = result["rows"]["A"][0]
    assert {"peak", "nearest_gene", "distance_to_tss", "peak_class", "pct_in", "pct_rest", "log2fc", "pval", "pval_adj"} <= set(row)
    assert "peak_index" not in row
    # Signac's fold change by default, named in the result, its params and the method; the old one on request
    assert result["fold_change"] == "signac" and result["params"]["fold_change"] == "signac"
    assert "Signac's FoldChange for peaks" in result["method"]
    r = await client.post(f"{API}/datasets/{did}/atac/differential", json={"column": "group", "fold_change": "tfidf"})
    old = next(d for e, d in _events(r.text) if e == "result")
    assert old["fold_change"] == "tfidf" and "mean TF-IDF + 1" in old["method"]
    assert old["rows"]["A"][0]["peak"] == row["peak"] and old["rows"]["A"][0]["pval"] == row["pval"]
    assert old["rows"]["A"][0]["log2fc"] != row["log2fc"]
    bad = await client.post(f"{API}/datasets/{did}/atac/differential", json={"column": "group", "fold_change": "v5"})
    assert bad.status_code == 422
    rid = result["id"]
    page = (await client.get(f"{API}/datasets/{did}/atac/differential/{rid}", params={"group": "B", "limit": 3})).json()
    assert list(page["rows"]) == ["B"] and len(page["rows"]["B"]) == 3
    r = await client.get(f"{API}/datasets/{did}/atac/differential/{rid}/csv")
    assert r.status_code == 200 and "attachment" in r.headers["content-disposition"]
    csv = pd.read_csv(io.StringIO(r.text))
    assert {"group", "peak", "nearest_gene", "pval_adj"} <= set(csv.columns) and "peak_index" not in csv.columns
    assert (await client.get(f"{API}/datasets/{did}/atac/differential/nope")).status_code == 404
    bad = await client.post(f"{API}/datasets/{did}/atac/differential", json={"column": "nope"})
    assert bad.status_code == 404 and "not found" in bad.json()["detail"]
    latest = await client.get(f"{API}/datasets/{did}/atac/differential")
    assert latest.status_code == 200


async def test_filtering_cells_keeps_the_store_and_runs_on_what_is_left(client, dataset_manager, annotated):
    did, p, _table, a = _register_multiome(dataset_manager)
    keep = np.arange(10, 70)
    sub = a[keep].copy()
    d = Path(dataset_manager.data_dir) / "uploads" / did / "data.h5ad"
    d.unlink()
    sub.write_h5ad(d)
    store_file = atac.store_path(dataset_manager.data_dir, did)
    before = store_file.stat().st_mtime_ns
    r = await client.post(f"{API}/datasets/{did}/atac/run-stream", json={"n_lsi": 6, "dims_to": 5, "activity": False})
    assert [e for e, _ in _events(r.text)][-1] == "complete"
    assert store_file.stat().st_mtime_ns == before
    adaptor = await dataset_manager.get_or_load_dataset(did)
    assert adaptor.adata.obsm[atac.OBSM_LSI].shape == (60, 6)
    r = await client.get(f"{API}/datasets/{did}/atac/activity", params={"gene": "GENEA"})
    assert r.status_code == 404  # asked to leave it out


async def test_run_says_when_positions_are_missing_and_still_saves_lsi(client, dataset_manager):
    a, p, table = _dataset()
    atac_import.write_info(a, {"n_peaks": len(table), "genome": "hg19", "computed": []})
    did = _register(dataset_manager, a, "no_pos")
    modality_store.write(atac.store_path(dataset_manager.data_dir, did), {atac.MATRIX: p}, list(a.obs_names),
                         table["name"].tolist(), atac_import.feature_attrs(table["name"].tolist()), {"genome": "hg19"})
    r = await client.post(f"{API}/datasets/{did}/atac/run-stream", json={"n_lsi": 6, "dims_to": 5})
    result = next(d for e, d in _events(r.text) if e == "result")
    assert result["lsi"] and result["activity"]["available"] is False and "on hg19 (GRCh37)" in result["activity"]["reason"]


async def test_metrics_attached_afterwards(client, dataset_manager):
    did, _p, _t, a = _register_multiome(dataset_manager)
    frame = pd.DataFrame({"barcode": a.obs_names, "is_cell": 1, "atac_raw_reads": 100, "atac_mitochondrial_reads": 10,
                          "atac_fragments": 50, "atac_TSS_fragments": 25, "atac_peak_region_fragments": 40})
    r = await client.post(f"{API}/datasets/{did}/atac/metrics",
                          files={"file": ("per_barcode_metrics.csv", frame.to_csv(index=False))})
    assert r.status_code == 200, r.text
    assert r.json()["n_matched"] == 80
    state = (await client.get(f"{API}/datasets/{did}/atac")).json()
    assert atac_import.OBS_FRIP in state["qc_columns"] and "per-barcode metrics" in state["info"]["computed"]
    adaptor = await dataset_manager.get_or_load_dataset(did)
    assert np.allclose(adaptor.adata.obs[atac_import.OBS_FRIP], 0.8)
    other = frame.assign(barcode=[f"X{i}" for i in range(80)])
    r = await client.post(f"{API}/datasets/{did}/atac/metrics", files={"file": ("x.csv", other.to_csv(index=False))})
    assert r.status_code == 422 and "None of the 80 cells" in r.json()["detail"]
    r = await client.post(f"{API}/datasets/{did}/atac/metrics", files={"file": ("x.csv", "a,b\n1,2\n")})
    assert r.status_code == 422 and "no column" in r.json()["detail"]


async def test_export_endpoint(client, dataset_manager):
    did, p, _t, _a = _register_multiome(dataset_manager)
    r = await client.get(f"{API}/datasets/{did}/atac/export")
    assert r.status_code == 200 and "_peaks.h5ad" in r.headers["content-disposition"]
    path = Path(dataset_manager.data_dir) / "tmp_export.h5ad"
    path.write_bytes(r.content)
    assert np.array_equal(ad.read_h5ad(path).X.toarray(), p.toarray())


async def test_a_run_reserves_what_it_said_and_lets_go(client, dataset_manager, annotated, monkeypatch):
    did, *_ = _register_multiome(dataset_manager)
    asked: list[tuple[int, str]] = []
    reserved: list[int] = []
    released: list[str] = []
    real_reserve, real_run, real_release = dataset_manager.reserve, dataset_manager.reserve_run, dataset_manager.release_run

    async def reserve(dataset_id, need, extra=0, what="x"):
        asked.append((need, what))
        return await real_reserve(dataset_id, need, extra, what)

    def reserve_run(dataset_id, need, remaining=None):
        reserved.append(need)
        return real_run(dataset_id, need, remaining) or object()

    def release_run(dataset_id):
        released.append(dataset_id)
        real_release(dataset_id)

    monkeypatch.setattr(dataset_manager, "reserve", reserve)
    monkeypatch.setattr(dataset_manager, "reserve_run", reserve_run)
    monkeypatch.setattr(dataset_manager, "release_run", release_run)
    r = await client.post(f"{API}/datasets/{did}/atac/run-stream", json={"n_lsi": 6, "dims_to": 5})
    events = _events(r.text)
    start = next(d for e, d in events if e == "start")
    assert asked[-1][0] == start["estimate_bytes"] and "ATAC analysis" in asked[-1][1]  # after the dataset's own open
    assert reserved == [start["estimate_bytes"]] and released == [did]
    assert did not in atac_api._running and dataset_manager.run_reservations() == {}


async def test_a_run_that_cannot_fit_is_refused_before_it_starts(client, dataset_manager, monkeypatch):
    from scview.core.dataset_manager import DatasetMemoryError

    did, *_ = _register_multiome(dataset_manager)

    await dataset_manager.get_or_load_dataset(did)

    async def reserve(dataset_id, need, extra=0, what="x"):
        raise DatasetMemoryError(dataset_id, need, 1, 2, what)

    monkeypatch.setattr(dataset_manager, "reserve", reserve)
    r = await client.post(f"{API}/datasets/{did}/atac/run-stream", json={})
    assert r.status_code == 507 and did not in atac_api._running
    adaptor = await dataset_manager.get_or_load_dataset(did)
    assert atac.OBSM_LSI not in adaptor.adata.obsm


async def test_a_differential_result_comes_back_from_the_results_store(client, dataset_manager):
    did, *_ = _register_multiome(dataset_manager)
    r = await client.post(f"{API}/datasets/{did}/atac/differential", json={"column": "group"})
    result = next(d for e, d in _events(r.text) if e == "result")
    results_store.flush()
    atac._DA_LAST.clear()
    latest = (await client.get(f"{API}/datasets/{did}/atac/differential")).json()
    assert latest["id"] == result["id"] and latest["n_tested"] == result["n_tested"]
    page = await client.get(f"{API}/datasets/{did}/atac/differential/{result['id']}", params={"group": "A", "limit": 2})
    assert page.status_code == 200 and page.json()["rows"]["A"][0]["peak"] == result["rows"]["A"][0]["peak"]
    # the generic results routes (and the MCP's get_result on them): settings and a table with peak names
    entry = (await client.get(f"{API}/datasets/{did}/results/{result['id']}")).json()
    assert entry["kind"] == "atac_differential" and entry["params"]["column"] == "group"
    t = await client.get(f"{API}/datasets/{did}/results/{result['id']}/table.tsv.gz")
    assert t.status_code == 200, t.text
    table = pd.read_csv(io.BytesIO(t.content), sep="\t", compression="gzip")
    assert {"group", "peak", "z", "log2fc", "pval", "pval_adj"} <= set(table.columns) and "peak_index" not in table.columns
    assert table["peak"].iloc[0] == result["rows"]["A"][0]["peak"]


def test_standard_chromosomes_in_either_naming_and_none_for_another_species():
    from scview.core import genome_build as gb

    got = gb.standard_chromosomes(["chr1", "1", "chrX", "Y", "chr22", "chrM", "MT", "JH584304.1", "KI270728.1",
                                   "chrUn_KI270728v1", "chr1_KI270706v1_random", "GL456211.1", "", None])
    assert got.tolist() == [True] * 5 + [False] * 7 + [True, True]
    assert gb.standard_chromosomes(["chr23", "chr1", "scaffold_9"]).all()  # numbered past 22: not human or mouse
    assert gb.standard_chromosomes(["chrI", "chrII", "chrM"]).all()  # worm / yeast
    assert gb.standard_chromosomes(["scaffold_1", "scaffold_2"]).all()  # no placed name at all
    assert gb.standard_chromosomes([]).size == 0


async def test_differential_leaves_out_peaks_off_the_standard_chromosomes(client, dataset_manager):
    a, p, table = _dataset()
    background = np.flatnonzero(table["kind"].to_numpy() == "background")[:2]
    table = table.copy()
    table.loc[background[0], "name"] = "JH584304.1:100000-100500"  # the mouse kidney's top PT "peak"
    table.loc[background[1], "name"] = "chrM:200-700"
    _register(dataset_manager, a, "nc")
    _seed(dataset_manager.data_dir, "nc", p, list(a.obs_names), table)
    out = {}
    for standard in (True, False):
        r = await client.post(f"{API}/datasets/nc/atac/differential",
                              json={"column": "group", "min_pct": 0, "standard_only": standard})
        out[standard] = next(d for e, d in _events(r.text) if e == "result")
    kept, every = out[True], out[False]
    assert kept["params"]["n_peaks_left_out"] == 2 and every["params"]["n_peaks_left_out"] == 0
    assert kept["n_tested"]["A"] == every["n_tested"]["A"] - 2
    assert "2 peaks on chrM, unplaced scaffolds or alt contigs left out" in kept["method"]
    assert "left out" not in every["method"]
    results_store.flush()
    t = await client.get(f"{API}/datasets/nc/atac/differential/{kept['id']}/csv")
    peaks = set(pd.read_csv(io.StringIO(t.text))["peak"])
    assert not peaks & {"JH584304.1:100000-100500", "chrM:200-700"}
    steps = [h for h in provenance.read_provenance(dataset_manager._datasets["nc"].adata)["history"]
             if h["step"] == "atac_differential"]
    assert steps[0]["params"]["standard_only"] is True and steps[1]["params"]["standard_only"] is False


async def test_one_group_against_another_streams_keeps_and_estimates(client, dataset_manager):
    did, *_ = _register_multiome(dataset_manager, "pair")
    e = (await client.get(f"{API}/datasets/{did}/atac/differential-estimate",
                          params={"column": "group", "test": "lr", "group": "B", "vs": "A"})).json()
    assert e["groups"] == ["A", "B"] and e["n_groups"] == 1 and e["vs"] == "A" and e["n_cells"] == 80
    every = (await client.get(f"{API}/datasets/{did}/atac/differential-estimate", params={"column": "group"})).json()
    assert every["groups"] == ["A", "B"] and every["n_groups"] == 2 and every["vs"] is None
    r = await client.post(f"{API}/datasets/{did}/atac/differential", json={"column": "group", "groups": ["B"], "vs": "A"})
    assert r.status_code == 200, r.text
    events = _events(r.text)
    assert events[0][1]["vs"] == "A" and events[0][1]["groups"] == ["B"]
    res = next(d for e_, d in events if e_ == "result")
    assert res["groups"] == ["B"] and res["vs"] == "A" and res["params"]["n_cells_vs"] == 40
    assert "B vs A (the cells of no other group used)" in res["method"]
    results_store.flush()
    entry = (await client.get(f"{API}/datasets/{did}/results/{res['id']}")).json()
    assert entry["label"] == "Differential accessibility B vs A by group" and entry["params"]["vs"] == "A"
    # each group vs the rest on the same column is a result of its own (a pair does not replace it)
    await client.post(f"{API}/datasets/{did}/atac/differential", json={"column": "group"})
    results_store.flush()
    listed = (await client.get(f"{API}/datasets/{did}/results", params={"kind": "atac_differential"})).json()["results"]
    assert len(listed) == 2 and all(e_["status"] == "primary" for e_ in listed)
    bad = await client.post(f"{API}/datasets/{did}/atac/differential", json={"column": "group", "vs": "A"})
    assert bad.status_code == 422 and "name one group" in bad.json()["detail"]
    same = await client.post(f"{API}/datasets/{did}/atac/differential", json={"column": "group", "groups": ["A"], "vs": "A"})
    assert same.status_code == 422


async def test_one_run_at_a_time(client, dataset_manager):
    did, *_ = _register_multiome(dataset_manager)
    atac_api._running[did] = "wnn"
    try:
        r = await client.post(f"{API}/datasets/{did}/atac/run-stream", json={})
        assert r.status_code == 409 and "already running" in r.json()["detail"]
        assert (await client.get(f"{API}/datasets/{did}/atac")).json()["running"] == "wnn"
    finally:
        atac_api._running.pop(did, None)


async def test_a_large_dataset_runs_from_the_store_without_reading_x(client, dataset_manager, annotated, monkeypatch):
    from scview.core import large_mode
    from tests.large_fixture import register

    did = "lg"
    a, p, table = _dataset(n_per=60)
    counts = ad.AnnData(X=a.X.copy(), obs=a.obs.copy(), var=a.var.copy())
    h5ad = register(dataset_manager.data_dir, did, counts)
    assert large_layout.sidecar_paths(h5ad)
    _seed(dataset_manager.data_dir, did, p, list(a.obs_names), table)
    # The peak info and PCs written in place, as an import and a pipeline would have.
    adaptor = await dataset_manager.get_or_load_dataset(did)
    assert adaptor.is_large
    adaptor.adata.uns[atac_import.UNS_KEY] = a.uns[atac_import.UNS_KEY]
    adaptor.adata.obsm["X_pca"] = a.obsm["X_pca"]
    adaptor.adata.uns["neighbors"] = a.uns["neighbors"]

    def boom(self, *args, **kwargs):  # noqa: ANN001
        raise AssertionError("X was read")

    for name in ("to_memory", "toarray"):
        if hasattr(large_mode.OnDiskMatrix, name):
            monkeypatch.setattr(large_mode.OnDiskMatrix, name, boom)
    r = await client.post(f"{API}/datasets/{did}/atac/run-stream", json={"n_lsi": 8, "dims_to": 6})
    events = _events(r.text)
    assert [e for e, _ in events][-1] == "complete", events[-1]
    adaptor = await dataset_manager.get_or_load_dataset(did)
    assert adaptor.is_large and adaptor.adata.obsm[atac.OBSM_LSI].shape == (120, 8)
    assert atac.OBS_CLUSTERS in adaptor.adata.obs.columns and atac.CONN_KEY in adaptor.adata.obsp
    pca, nb = adaptor.adata.obsm["X_pca"], adaptor.adata.uns["neighbors"]
    dataset_manager.evict(did)  # written into its h5ad in place
    adaptor = await dataset_manager.get_or_load_dataset(did)
    assert adaptor.adata.obsm[atac.OBSM_LSI].shape == (120, 8) and atac.CONN_KEY in adaptor.adata.obsp
    assert atac_import.read_info(adaptor.adata)["lsi"]["n_clusters"] >= 1
    adaptor.adata.obsm["X_pca"], adaptor.adata.uns["neighbors"] = pca, nb
    r = await client.post(f"{API}/datasets/{did}/atac/wnn/run-stream", json={})
    assert [e for e, _ in _events(r.text)][-1] == "complete"
    r = await client.post(f"{API}/datasets/{did}/atac/differential", json={"column": "group"})
    assert [e for e, _ in _events(r.text)][-1] == "complete"
    r = await client.post(f"{API}/datasets/{did}/atac/differential", json={"column": "group", "test": "lr"})
    assert [e for e, _ in _events(r.text)][-1] == "complete"
    adaptor = await dataset_manager.get_or_load_dataset(did)
    assert adaptor.is_large and provenance.read_provenance(adaptor.adata)["history"][-1]["step"] == "atac_differential"
    r = await client.get(f"{API}/datasets/{did}/atac/activity", params={"gene": "GENEA"})
    assert r.status_code == 200 and len(r.json()["values"]) == 120


async def test_differential_lr_says_its_test_keeps_fit_counts_and_a_replay_step(client, dataset_manager, annotated):
    did, *_ = _register_multiome(dataset_manager)
    bad = await client.post(f"{API}/datasets/{did}/atac/differential", json={"column": "group", "test": "t"})
    assert bad.status_code == 422
    r = await client.post(f"{API}/datasets/{did}/atac/differential", json={"column": "group", "test": "lr"})
    assert r.status_code == 200, r.text
    events = _events(r.text)
    start = next(d for e, d in events if e == "start")
    assert start["test"] == "lr" and start["estimate_seconds"] > 0 and start["estimate_bytes"] > 0
    steps = {d["step"] for e, d in events if e == "progress"}
    assert {"detection", "score", "fit"} <= steps
    result = next(d for e, d in events if e == "result")
    assert result["test"] == "lr" and result["params"]["test"] == "lr" and result["params"]["latent"]
    assert "Logistic regression" in result["method"] and "covariate" in result["method"]
    assert sum(result["n_fit"]["A"].values()) >= result["n_tested"]["A"] > 0
    assert result["rows"]["A"][0]["fit"] in ("lrt", "lrt_separated")
    assert did not in atac_api._running
    csv = pd.read_csv(io.StringIO((await client.get(f"{API}/datasets/{did}/atac/differential/{result['id']}/csv")).text))
    assert "fit" in csv.columns
    adaptor = await dataset_manager.get_or_load_dataset(did)
    last = provenance.read_provenance(adaptor.adata)["history"][-1]
    assert last["step"] == "atac_differential" and last["params"]["test"] == "lr" and last["params"]["column"] == "group"
    results_store.flush()
    atac._DA_LAST.clear()  # from the results store: the fit column and its counts come back
    latest = (await client.get(f"{API}/datasets/{did}/atac/differential")).json()
    assert latest["id"] == result["id"] and latest["test"] == "lr" and latest["n_fit"] == result["n_fit"]
    assert latest["rows"]["A"][0]["fit"] == result["rows"]["A"][0]["fit"]


async def test_differential_wilcoxon_says_so_and_records_its_step_too(client, dataset_manager, annotated):
    did, *_ = _register_multiome(dataset_manager)
    r = await client.post(f"{API}/datasets/{did}/atac/differential", json={"column": "group"})
    result = next(d for e, d in _events(r.text) if e == "result")
    assert result["test"] == "wilcoxon" and result["n_fit"] is None and "fit" not in result["rows"]["A"][0]
    adaptor = await dataset_manager.get_or_load_dataset(did)
    assert provenance.read_provenance(adaptor.adata)["history"][-1]["params"]["test"] == "wilcoxon"


async def test_differential_estimate_names_each_test(client, dataset_manager):
    did, *_ = _register_multiome(dataset_manager)
    url = f"{API}/datasets/{did}/atac/differential-estimate"
    w = (await client.get(url, params={"column": "group"})).json()
    lr = (await client.get(url, params={"column": "group", "test": "lr"})).json()
    assert w["test"] == "wilcoxon" and lr["test"] == "lr" and lr["n_groups"] == 2 and lr["n_cells"] == 80
    assert lr["estimate_seconds"] >= w["estimate_seconds"] and lr["estimate_bytes"] > 0
    assert (await client.get(url, params={"column": "nope"})).status_code == 404
    assert (await client.get(url, params={"column": "group", "test": "t"})).status_code == 422


async def test_differential_cells_changed_mid_run_saves_nothing(client, dataset_manager, annotated, monkeypatch):
    from scview.core.anndata_adaptor import AnnDataAdaptor

    did, *_ = _register_multiome(dataset_manager)
    real = atac.iter_differential

    def changed(*a, **kw):
        for item in real(*a, **kw):
            if item["type"] == "result":
                monkeypatch.setattr(AnnDataAdaptor, "cells_fingerprint", lambda _self: "other")
            yield item

    monkeypatch.setattr(atac, "iter_differential", changed)
    r = await client.post(f"{API}/datasets/{did}/atac/differential", json={"column": "group", "test": "lr"})
    last = _events(r.text)[-1]
    assert last[0] == "error" and "cells changed" in last[1]["error"]
    assert did not in atac_api._running
    assert (await client.get(f"{API}/datasets/{did}/atac/differential")).json() is None


async def test_a_run_is_watched_as_one_step_against_its_estimate(monkeypatch):
    """QA round 4: an LSI run estimated at 139 s got a slow note at 61 s, from one stage. The
    whole run is one watched step now, its first note at 1.5 x the estimate."""
    import time

    from scview.core import run_watchdog

    monkeypatch.setattr(run_watchdog, "SAMPLE_SECONDS", 0.01)
    monkeypatch.setattr(run_watchdog, "SLOW_STEP_SECONDS", 0.05)

    class DM:
        def reserve_run(self, dataset_id, need):
            return None

        def release_run(self, dataset_id):
            return None

    def gen(stop):
        for i in range(20):
            time.sleep(0.1)  # each stage longer than the 0.05 s interval
            yield {"type": "stage", "stage": f"s{i}"}
        yield {"type": "result"}

    async def on_result(item, res):
        return {"resources": res}

    text = "".join([e async for e in atac_api._events(DM(), "watched", "run", {"estimate_seconds": 1.0}, 0, gen, on_result,
                                                      "ATAC analysis")])
    events = _events(text)
    kinds = [k for k, _ in events]
    first = kinds.index("watchdog")
    # The fixed interval would have sent one during the first stage; 1.5 x the estimate is 1.5 s in
    assert kinds[:4] == ["start", "stage", "stage", "stage"] and first >= 4
    slow = [d for k, d in events if k == "watchdog"]
    assert slow[0]["kind"] == "slow_step" and slow[0]["step"] == "atac"
    assert slow[0]["expected_s"] == pytest.approx([0.8, 1.2]) and slow[0]["estimate_s"] == 1
    res = dict(events)["result"]["resources"]
    assert res["steps"]["atac"]["elapsed_s"] >= 2.0  # the whole run, not its last stage
    assert kinds[-1] == "complete" and "watched" not in atac_api._running
