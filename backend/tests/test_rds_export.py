"""Seurat / SingleCellExperiment .rds export: what goes in (core/rds_export.py), the
staging h5ad the R converter reads, and the job endpoints (api/v1/export_rds.py) with a
fake converter (no R here; converter/tests covers the R side)."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import anndata as ad
import h5py
import httpx
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
import scipy.sparse as sp

from scview.core import conversion, rds_export
from tests.large_fixture import counts_adata, register

API = "/api/v1"


def _processed(n: int = 200, g: int = 60) -> ad.AnnData:
    """As scView keeps a processed dataset: X log-normalised, .raw = X, counts layer."""
    rng = np.random.default_rng(3)
    counts = rng.poisson(0.7, (n, g)).astype(np.float32)
    counts[: n // 2, :8] += 4
    a = ad.AnnData(sp.csr_matrix(counts))
    a.obs_names = [f"c{i}" for i in range(n)]
    a.var_names = [f"Gene_{i}" if i < 3 else f"G{i}" for i in range(g)]
    a.layers["counts"] = a.X.copy()
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    a.raw = a
    a.var["highly_variable"] = np.arange(g) < 20
    a.obs["leiden"] = pd.Categorical([str(i % 3) for i in range(n)], categories=["2", "0", "1"])
    a.obs["donor"] = [f"d{i % 4}" for i in range(n)]
    a.obs["is_doublet"] = (np.arange(n) % 10) == 0
    a.obs["n_counts"] = counts.sum(1)
    a.obs["n_genes"] = pd.array((counts > 0).sum(1), dtype="Int64")
    a.uns["leiden_colors"] = ["#ff0000", "#00ff00", "#0000ff"]
    a.uns["scview_active_clustering"] = "leiden"
    a.uns["scview_provenance"] = json.dumps({"schema_version": 1, "history": [
        {"step": "normalize", "params": {}}, {"step": "pca", "params": {"n_comps": 5}}]})
    a.uns["pca"] = {"variance": np.arange(5, 0, -1, dtype=float), "variance_ratio": np.full(5, 0.1)}
    a.uns["rank_genes_groups__leiden"] = {"params": {"groupby": "leiden"}}
    a.obsm["X_pca"] = rng.normal(size=(n, 5)).astype(np.float32)
    a.obsm["X_umap"] = rng.normal(size=(n, 2))
    a.obsm["X_nmf_usage"] = rng.random((n, 4)).astype(np.float32)
    a.obsm["protein_counts"] = pd.DataFrame(rng.poisson(5, (n, 3)).astype(float), index=a.obs_names,
                                            columns=["CD3", "CD4_ab", "CD19"])
    a.obsp["connectivities"] = sp.identity(n, format="csr")
    pcs = np.zeros((g, 5))
    pcs[:20] = rng.normal(size=(20, 5))
    a.varm["PCs"] = pcs
    return a


class _A:
    """An adaptor as plan() reads it: the AnnData, its file, no large mode."""

    def __init__(self, a: ad.AnnData, path: str = "/nonexistent.h5ad") -> None:
        self.adata = a
        self.h5ad_path = path
        self.is_large = False

    def active_clustering_column(self):
        return self.adata.uns.get("scview_active_clustering")


def _register(dm, a: ad.AnnData | None = None, did: str = "rx1", name: str = "My PBMC") -> Path:
    root = Path(dm.data_dir)
    (root / "uploads" / did).mkdir(parents=True)
    (root / "uploads" / did / "metadata.json").write_text(
        json.dumps({"id": did, "name": name, "status": "ready"}))
    (root / "derived" / did).mkdir(parents=True)
    path = root / "derived" / did / "data.h5ad"
    (a if a is not None else _processed()).write_h5ad(path)
    return path


# ---------------------------------------------------------------------------
# What goes in


def test_plan_takes_lognormalised_data_counts_layouts_loadings_and_protein():
    a = _processed()
    p = rds_export.plan(_A(a), "seurat", "My PBMC", "rx1")
    assert p.x.where == "X" and not p.x.is_counts
    assert p.counts.where == "layers['counts']"
    assert list(p.obsm) == ["X_pca", "X_umap"]  # all their columns; per-cell scores are not layouts
    assert p.obsm["X_pca"].shape == (200, 5)
    assert p.pcs is not None and p.pcs.shape == (60, 5)
    assert p.protein["names"] == ["CD3", "CD4_ab", "CD19"] and p.protein["counts"].shape == (200, 3)
    assert p.meta["cluster_column"] == "leiden"
    assert p.meta["variable_genes"] == [f"Gene_{i}" if i < 3 else f"G{i}" for i in range(20)]
    assert p.meta["recipe"] == ["normalize", "pca"]
    assert p.meta["pca_variance"] == [5.0, 4.0, 3.0, 2.0, 1.0]
    assert p.meta["colors"] == {"leiden_colors": ["#ff0000", "#00ff00", "#0000ff"]}
    assert p.dropped["obsm"] == ["X_nmf_usage"]
    assert p.dropped["obsp"] == ["connectivities"]
    assert "rank_genes_groups__leiden" in p.dropped["uns"]
    assert "pca" not in p.dropped["uns"] and "scview_provenance" not in p.dropped["uns"]
    assert "layers" not in p.dropped  # counts is the only layer, and it goes in
    assert any(s.startswith("obsp: connectivities - neighbour graphs") for s in p.meta["left_out"])
    assert p.n_values == a.X.nnz + a.layers["counts"].nnz
    assert p.obs["is_doublet"].dtype == bool  # logical in R, not a "True"/"False" factor
    assert p.obs["n_genes"].dtype == np.float64  # a nullable integer column
    assert list(p.obs["leiden"].cat.categories) == ["2", "0", "1"]  # level order kept


def test_unnormalised_dataset_exports_counts_only():
    a = _processed()
    b = ad.AnnData(a.layers["counts"].copy(), obs=a.obs[["leiden"]].copy(), var=pd.DataFrame(index=a.var_names))
    p = rds_export.plan(_A(b), "seurat")
    assert p.x is None and p.counts is not None and p.counts.where == "X"
    assert any("NormalizeData" in n for n in p.notes)
    assert p.n_values == b.X.nnz  # the one matrix, counted once


def test_scaled_x_uses_the_lognormalised_raw_and_never_scaled_values():
    a = _processed()
    a.X = sp.csr_matrix(sc.pp.scale(a.X.toarray(), copy=True))
    p = rds_export.plan(_A(a), "sce")
    assert p.x.where == ".raw"
    a.raw = None
    p = rds_export.plan(_A(a), "sce")
    assert p.x is None and p.counts.where == "layers['counts']"
    assert any("scaled" in n for n in p.notes)
    del a.layers["counts"]
    with pytest.raises(rds_export.ExportError, match="neither raw counts"):
        rds_export.plan(_A(a), "sce")


def test_cell_ontology_terms_travel_as_cell_metadata():
    import json
    a = _processed()
    a.uns["scview_label_terms"] = json.dumps({"leiden": {"0": "CL:0000236", "1": "unknown"}, "gone": {"x": "CL:1"}})
    p = rds_export.plan(_A(a), "seurat")
    ids = p.obs["leiden_ontology_term_id"]
    assert list(ids[a.obs["leiden"] == "0"].unique()) == ["CL:0000236"]
    assert list(ids[a.obs["leiden"] == "1"].unique()) == ["unknown"]
    assert ids[a.obs["leiden"].to_numpy() == "2"].isna().all()  # no term: NA
    assert "gone_ontology_term_id" not in p.obs  # a column no longer in obs
    assert "scview_label_terms" not in p.dropped.get("uns", [])
    assert any("Cell Ontology" in n for n in p.notes)


def test_unknown_format_is_refused():
    with pytest.raises(rds_export.ExportError, match="Unknown R format"):
        rds_export.plan(_A(_processed()), "loom")


def test_memory_check_refuses_what_the_converter_cannot_hold(monkeypatch):
    from scview.core import memory_budget

    p = rds_export.plan(_A(_processed()), "seurat")
    need = rds_export.estimate_bytes(p)
    assert need == (rds_export.CONVERTER_BASE_BYTES + rds_export.CONVERTER_BYTES_PER_VALUE * p.n_values
                    + (200 * 5 + 200 * 2) * 16 + 200 * 3 * 24)
    monkeypatch.setattr(memory_budget, "read_budget",
                        lambda: memory_budget.MemoryBudget(need - 1, 0, True, "container limit"))
    assert rds_export.memory_check(p)["fits"] is False
    monkeypatch.setattr(memory_budget, "read_budget",
                        lambda: memory_budget.MemoryBudget(None, None, False, "unknown"))
    assert rds_export.memory_check(p)["fits"] is True


def test_staging_file_has_the_layout_the_r_side_reads(tmp_path: Path):
    a = _processed()
    p = rds_export.plan(_A(a), "seurat", "My PBMC", "rx1")
    out = rds_export.write_staging(p, tmp_path / "input.h5ad", {"scview_version": "9.9.9"})
    with h5py.File(out, "r") as f:
        assert {"X", "layers", "obs", "var", "obsm", "varm", "uns"} <= set(f.keys())
        assert f["X"].attrs["encoding-type"] == "csr_matrix"
        x = ad.io.read_elem(f["X"])
        counts = ad.io.read_elem(f["layers/counts"])
        meta = json.loads(ad.io.read_elem(f["uns/scview_rexport"]))
        prot = ad.io.read_elem(f["uns/protein_counts"])
        obs = ad.io.read_elem(f["obs"])
        assert set(f["obsm"].keys()) == {"X_pca", "X_umap"}
        assert f["varm/PCs"].shape == (60, 5)
    np.testing.assert_array_equal(x.toarray(), a.X.toarray())
    np.testing.assert_array_equal(counts.toarray(), a.layers["counts"].toarray())
    np.testing.assert_array_equal(prot, a.obsm["protein_counts"].to_numpy())
    assert meta["scview_version"] == "9.9.9" and meta["dataset_name"] == "My PBMC"
    assert meta["protein_names"] == ["CD3", "CD4_ab", "CD19"]
    assert meta["x_from"] == "X" and meta["counts_from"] == "layers['counts']"
    assert obs["is_doublet"].dtype == bool
    assert not list(tmp_path.glob(".*part"))


def test_staging_without_lognormalised_values_has_no_x(tmp_path: Path):
    a = _processed()
    b = ad.AnnData(a.layers["counts"].copy(), var=pd.DataFrame(index=a.var_names))
    out = rds_export.write_staging(rds_export.plan(_A(b), "sce"), tmp_path / "s.h5ad", {})
    with h5py.File(out, "r") as f:
        assert "X" not in f and "layers/counts" in f


def test_large_dataset_takes_counts_from_its_sidecar(tmp_path: Path):
    from scview.core.anndata_adaptor import AnnDataAdaptor

    counts = counts_adata(n_cells=120, n_genes=50)
    h5ad = register(tmp_path, "bigR", counts)
    adaptor = AnnDataAdaptor(str(h5ad))
    assert adaptor.is_large
    p = rds_export.plan(adaptor, "seurat")
    assert p.counts.where == "the counts file" and p.x.where == "X"
    out = rds_export.write_staging(p, tmp_path / "big_input.h5ad", {})
    with h5py.File(out, "r") as f:
        got = ad.io.read_elem(f["layers/counts"])
    np.testing.assert_array_equal(got.toarray(), counts.X.toarray())


# ---------------------------------------------------------------------------
# The converter client


async def test_trigger_export_sends_the_format_and_returns_the_report():
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen.update(json.loads(req.content))
        assert req.url.path == "/export"
        return httpx.Response(200, json={"status": "ok", "message": "Wrote", "report": {"kept": ["a"]}})

    out = await rds_export.trigger_export("/d/in.h5ad", "/d/out.rds", "sce", "http://conv",
                                          progress_path="/d/p.json", transport=httpx.MockTransport(handler))
    assert out["report"] == {"kept": ["a"]}
    assert seen == {"input_path": "/d/in.h5ad", "output_path": "/d/out.rds", "format": "sce",
                    "progress_path": "/d/p.json"}


@pytest.mark.parametrize("status,body,expect", [
    (500, {"status": "error", "error": "Export failed: boom"}, "Export failed: boom"),
    (404, {"status": "error", "error": "Not found: POST /export"}, "cannot write .rds files yet"),
    (502, None, "answered 502 without saying why"),
])
async def test_trigger_export_errors_are_plain_sentences(status, body, expect):
    def handler(req):
        return httpx.Response(status, json=body) if body is not None else httpx.Response(status, text="<html>")

    with pytest.raises(conversion.ConversionError, match=expect):
        await rds_export.trigger_export("i", "o", "seurat", "http://conv", transport=httpx.MockTransport(handler))


async def test_trigger_export_unreachable_converter_says_it_is_not_running():
    def handler(req):
        raise httpx.ConnectError("refused")

    with pytest.raises(conversion.ConversionError, match="not running"):
        await rds_export.trigger_export("i", "o", "seurat", "http://conv", transport=httpx.MockTransport(handler))


@pytest.mark.parametrize("exc,state", [
    (None, "ok"), (httpx.ConnectError("refused"), "down"), (httpx.ReadTimeout("slow"), "busy"),
])
async def test_converter_state(exc, state):
    def handler(req):
        if exc is not None:
            raise exc
        return httpx.Response(200, json={"status": "ok"})

    assert await rds_export.converter_state("http://conv", transport=httpx.MockTransport(handler)) == state


# ---------------------------------------------------------------------------
# Jobs and cleanup


def test_sweep_removes_old_folders_but_not_running_jobs(tmp_path: Path):
    old = rds_export.new_job(tmp_path, "d1", "seurat", "a.rds")
    running = rds_export.new_job(tmp_path, "d1", "sce", "b.rds")
    stray = rds_export.jobs_root(tmp_path) / "left-by-a-restart"
    stray.mkdir()
    old.status, old.finished = "done", time.time() - 4000
    past = time.time() - 4000
    for d in (old.folder, running.folder, stray):
        os.utime(d, (past, past))
    assert rds_export.sweep(tmp_path) == 2
    assert not old.folder.exists() and not stray.exists() and running.folder.exists()
    assert rds_export.get_job(old.id) is None and rds_export.get_job(running.id) is running
    rds_export.remove(running)


# ---------------------------------------------------------------------------
# Endpoints (fake converter)


@pytest.fixture()
def converter_ok(monkeypatch):
    async def state(url, transport=None):
        return "ok"

    monkeypatch.setattr(rds_export, "converter_state", state)


def _fake_converter(monkeypatch, calls: list, fail: str | None = None):
    async def fake(input_path, output_path, fmt, converter_url, progress_path=None, transport=None):
        with h5py.File(input_path, "r") as f:  # the staging file is complete when R is asked
            calls.append({"fmt": fmt, "keys": sorted(f.keys()),
                          "meta": json.loads(ad.io.read_elem(f["uns/scview_rexport"]))})
        if fail:
            raise conversion.ConversionError(fail)
        Path(output_path).write_bytes(b"RDX3\n" + b"\0" * 100)
        return {"status": "ok", "message": "Wrote a Seurat object (200 cells x 60 genes)",
                "report": {"kept": "only one", "skipped": ["obsp: connectivities"], "renamed_features": 3}}

    monkeypatch.setattr(rds_export, "trigger_export", fake)


async def _finish(job_id: str) -> None:
    job = rds_export.get_job(job_id)
    if job is not None and job.task is not None:
        await job.task


async def test_info_says_what_goes_in_and_whether_it_fits(client, dataset_manager, converter_ok):
    _register(dataset_manager)
    r = await client.get(f"{API}/datasets/rx1/export/rds/info", params={"format": "seurat"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["filename"] == "scview_My_PBMC_seurat.rds"
    assert body["converter"] == "ok" and body["fits"] is True
    assert body["data_from"] == "X" and body["counts_from"] == "layers['counts']"
    assert body["embeddings"] == ["X_pca", "X_umap"] and body["n_proteins"] == 3
    assert body["cluster_column"] == "leiden" and body["running_job"] is None
    assert (await client.get(f"{API}/datasets/rx1/export/rds/info", params={"format": "x"})).status_code == 400
    assert (await client.get(f"{API}/datasets/nope/export/rds/info")).status_code == 404


async def test_export_job_writes_staging_asks_r_and_sends_the_file_once(client, dataset_manager, converter_ok,
                                                                         monkeypatch):
    calls: list = []
    _fake_converter(monkeypatch, calls)
    _register(dataset_manager)
    r = await client.post(f"{API}/datasets/rx1/export/rds", params={"format": "sce"})
    assert r.status_code == 202, r.text
    job_id = r.json()["job"]
    assert r.json()["filename"] == "scview_My_PBMC_sce.rds"
    await _finish(job_id)
    st = (await client.get(f"{API}/datasets/rx1/export/rds/{job_id}/status")).json()
    assert st["status"] == "done", st
    assert st["report"]["kept"] == ["only one"] and st["report"]["renamed_features"] == 3
    assert st["size_bytes"] == 105
    assert calls[0]["fmt"] == "sce" and {"X", "layers", "obs", "obsm"} <= set(calls[0]["keys"])
    assert calls[0]["meta"]["dataset_id"] == "rx1" and calls[0]["meta"]["format"] == "sce"
    job = rds_export.get_job(job_id)
    assert not (job.folder / rds_export.STAGING).exists()  # the staging copy goes when R is done

    f = await client.get(f"{API}/datasets/rx1/export/rds/{job_id}")
    assert f.status_code == 200 and f.content.startswith(b"RDX3")
    assert 'filename="scview_My_PBMC_sce.rds"' in f.headers["content-disposition"]
    assert not job.folder.exists()  # sent once, then removed
    gone = await client.get(f"{API}/datasets/rx1/export/rds/{job_id}")
    assert gone.status_code == 404 and "Start the export again" in gone.json()["detail"]


async def test_converter_failure_ends_the_job_with_its_sentence(client, dataset_manager, converter_ok, monkeypatch):
    calls: list = []
    _fake_converter(monkeypatch, calls, fail="Export failed: the SeuratObject package is not installed")
    _register(dataset_manager)
    job_id = (await client.post(f"{API}/datasets/rx1/export/rds")).json()["job"]
    await _finish(job_id)
    st = (await client.get(f"{API}/datasets/rx1/export/rds/{job_id}/status")).json()
    assert st["status"] == "error" and "SeuratObject" in st["error"]
    r = await client.get(f"{API}/datasets/rx1/export/rds/{job_id}")
    assert r.status_code == 409 and "SeuratObject" in r.json()["detail"]
    job = rds_export.get_job(job_id)
    assert not (job.folder / rds_export.STAGING).exists() and not job.output.exists()
    rds_export.remove(job)


async def test_converter_not_running_is_a_503(client, dataset_manager, monkeypatch):
    async def down(url, transport=None):
        return "down"

    monkeypatch.setattr(rds_export, "converter_state", down)
    _register(dataset_manager)
    r = await client.post(f"{API}/datasets/rx1/export/rds")
    assert r.status_code == 503 and "converter is not running" in r.json()["detail"]
    info = (await client.get(f"{API}/datasets/rx1/export/rds/info")).json()
    assert info["converter"] == "down"


async def test_too_big_for_the_converter_is_a_507(client, dataset_manager, converter_ok, monkeypatch):
    from scview.core import memory_budget

    monkeypatch.setattr(memory_budget, "read_budget",
                        lambda: memory_budget.MemoryBudget(256 * 1024 ** 2, 0, True, "container limit"))
    _register(dataset_manager)
    r = await client.post(f"{API}/datasets/rx1/export/rds")
    assert r.status_code == 507
    assert "GB limit" in r.json()["detail"] and "SCVIEW_MEMORY_GB" in r.json()["detail"]
    info = (await client.get(f"{API}/datasets/rx1/export/rds/info")).json()
    assert info["fits"] is False and "SCVIEW_MEMORY_GB" in info["too_big"]


async def test_a_second_export_of_the_same_format_waits_for_the_first(client, dataset_manager, converter_ok,
                                                                      monkeypatch):
    import asyncio

    release = asyncio.Event()

    async def slow(input_path, output_path, fmt, converter_url, progress_path=None, transport=None):
        Path(progress_path).write_text(json.dumps({"stage": "building", "message": "Building the Seurat object"}))
        await release.wait()
        Path(output_path).write_bytes(b"RDX3")
        return {"status": "ok", "message": "Wrote", "report": {}}

    monkeypatch.setattr(rds_export, "trigger_export", slow)
    _register(dataset_manager)
    job_id = (await client.post(f"{API}/datasets/rx1/export/rds")).json()["job"]
    for _ in range(200):
        if rds_export.get_job(job_id).status == "converting":
            break
        await asyncio.sleep(0.01)
    st = (await client.get(f"{API}/datasets/rx1/export/rds/{job_id}/status")).json()
    assert st["status"] == "converting" and st["message"] == "Building the Seurat object"
    assert (await client.post(f"{API}/datasets/rx1/export/rds")).status_code == 409
    assert (await client.get(f"{API}/datasets/rx1/export/rds/{job_id}")).status_code == 409
    info = (await client.get(f"{API}/datasets/rx1/export/rds/info")).json()
    assert info["running_job"] == job_id
    release.set()
    await _finish(job_id)
    assert (await client.get(f"{API}/datasets/rx1/export/rds/{job_id}")).status_code == 200
