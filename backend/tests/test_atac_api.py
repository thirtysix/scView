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
from scview.core import atac, atac_import, gene_annotation, large_layout, modality_store, provenance
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
    assert result["lsi"] and result["activity"]["available"] is False and "not 'hg19'" in result["activity"]["reason"]


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
    r = await client.post(f"{API}/datasets/{did}/atac/wnn/run-stream", json={})
    assert [e for e, _ in _events(r.text)][-1] == "complete"
    r = await client.post(f"{API}/datasets/{did}/atac/differential", json={"column": "group"})
    assert [e for e, _ in _events(r.text)][-1] == "complete"
    r = await client.get(f"{API}/datasets/{did}/atac/activity", params={"gene": "GENEA"})
    assert r.status_code == 200 and len(r.json()["values"]) == 120
