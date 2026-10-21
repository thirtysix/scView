"""Preflight estimates: calibration against measured peaks, monotonicity, the
verdict and its suggestions, the memory and CPU readings, and the endpoints."""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from scview.core import preflight as pf
from scview.core.pipeline import ALL_STEPS, PipelineParams

GiB = 1 << 30
API = "/api/v1"
REPROCESS = ["reset_to_counts", "qc_metrics", "doublet_detection", "filtering", "normalization",
             "log_transform", "highly_variable_genes", "scaling", "pca", "neighbors", "clustering",
             "embeddings", "marker_genes", "enrichment", "cell_cycle", "dissociation_stress"]


# --- calibration: the measured peaks the coefficients were fitted to -----------------------
# Headers as read from the test files; measured = cgroup anon at the peak minus the process
# before loading (scanpy imported), "Reprocess from scratch" steps, 2 CPUs, a fresh process.
def _h(fmt, n, m, mats, obsm=0, obsp=0, varm=0, uns=0, obs_cols=0, var_cols=0, emb=False):
    h = pf.Header(fmt, n, m, [pf.Matrix(w, n, m, sparse, nnz, dt, "int32")
                              for w, sparse, nnz, dt in mats],
                  obsm={"X_pca": [n, 50]} if emb else {}, obsm_bytes=obsm, obsp_bytes=obsp,
                  varm_bytes=varm, uns_bytes=uns, obs_columns=obs_cols, var_columns=var_cols)
    return h


CALIBRATION = {
    # name: (header, measured resident after load, measured run peak, marker columns)
    "pbmc 10x h5": (_h("10x_h5", 1222, 33538, [("X", True, 2498225, "float32")]),
                    0.04, 0.63, []),
    "paul15 (dense)": (_h("h5ad", 2730, 3451, [("X", False, 9421230, "float32"),
                                               ("layers/counts", False, 9421230, "float32"),
                                               ("raw/X", False, 9421230, "float32")],
                          546000, 990600, 1380400, 7497385, 10, 11, emb=True),
                       0.13, 0.62, ["paul15_clusters", "scview_leiden_r0.5"]),
    "Kang (sparse)": (_h("h5ad", 13836, 14053, [("X", False, 194437308, "float64"),
                                                ("layers/counts", True, 9655137, "float64"),
                                                ("raw/X", True, 9655137, "float64")],
                         8301600, 4574264, 5621200, 31893321, 15, 11, emb=True),
                      1.76, 3.00, ["cluster", "scview_leiden_r0.5"]),
    "ovary 10k": (_h("h5ad", 10000, 20082, [("X", True, 12109310, "float32"),
                                            ("layers/counts", False, 200820000, "float32")],
                     uns=10924929, obs_cols=7, var_cols=5),
                  0.91, 6.17, ["cluster", "Cluster"]),
    "ovary 20k": (_h("h5ad", 20000, 20082, [("X", True, 24099492, "float32"),
                                            ("layers/counts", False, 401640000, "float32")],
                     uns=10924929, obs_cols=7, var_cols=5),
                  1.75, 11.73, ["cluster", "Cluster"]),
    # measured through PCA (the steps after it are estimated at 16.5 GiB at most); killed at
    # 18 GiB in doublet detection, measured at 26 GiB
    "ovary 37k": (_h("h5ad", 37104, 20082, [("X", True, 44738263, "float32"),
                                            ("layers/counts", False, 745122528, "float32")],
                     uns=10924929, obs_cols=7, var_cols=5),
                  3.22, 20.86, ["cluster", "Cluster"]),
}


@pytest.mark.parametrize("name", list(CALIBRATION))
def test_estimates_are_within_the_stated_band_of_the_measured_peaks(name):
    h, load_m, peak_m, cols = CALIBRATION[name]
    load = pf.resident_bytes(h) / GiB
    run = pf.estimate_run(h, steps=REPROCESS, params=PipelineParams(marker_columns=cols), warm=False)
    assert abs(run.peak_bytes / GiB / peak_m - 1) <= pf.ERROR_BAND
    assert abs(load - load_m) <= max(pf.ERROR_BAND * load_m, 0.03)


def test_the_dense_counts_layer_puts_the_peak_in_doublet_detection():
    h = CALIBRATION["ovary 20k"][0]
    run = pf.estimate_run(h, steps=REPROCESS, warm=False)
    assert run.peak_step == "doublet_detection"
    # Scrublet copies the dataset: its two dense 20,000 x 20,082 float32 matrices (X after the
    # reset, and the counts layer) are 1.5 GiB each, so the step needs more than 4 of them.
    dense = 20000 * 20082 * 4
    dbl = next(e for e in run.steps if e.step == "doublet_detection")
    assert dbl.transient_bytes > 4 * dense


def test_reset_to_counts_frees_the_dense_x_before_the_later_steps():
    h = CALIBRATION["Kang (sparse)"][0]
    run = pf.estimate_run(h, steps=REPROCESS, warm=False)
    x_dense = 13836 * 14053 * 8
    counts = 9655137 * (8 + 4) + 13837 * 4
    reset, qc = run.steps[0], run.steps[1]
    assert qc.resident_bytes <= reset.resident_bytes - (x_dense - counts)
    # so the run's peak is well under "what it holds now + the largest step"
    biggest = max(e.transient_bytes for e in run.steps)
    assert run.peak_bytes < run.start_bytes + biggest - GiB


# --- monotonic in cells and stored values ----------------------------------------------------


def test_run_estimate_grows_with_cells_and_values():
    peaks = [pf.estimate_run((n, 15000), nnz=n * 1000, steps=REPROCESS, warm=True).peak_bytes
             for n in (1000, 5000, 20000, 80000)]
    assert peaks == sorted(peaks) and len(set(peaks)) == 4
    by_nnz = [pf.estimate_run((10000, 15000), nnz=v, steps=REPROCESS, warm=True).peak_bytes
              for v in (1_000_000, 10_000_000, 50_000_000)]
    assert by_nnz == sorted(by_nnz) and len(set(by_nnz)) == 3
    dense = pf.estimate_run((10000, 15000), dense=True, steps=REPROCESS, warm=True).peak_bytes
    assert dense > by_nnz[-1]


def test_load_estimate_grows_with_cells_and_values():
    def res(n, nnz):
        return pf.resident_bytes(pf.Header("h5ad", n, 20000, [pf.Matrix("X", n, 20000, True, nnz)]))

    assert res(1000, 10**6) < res(10000, 10**6) < res(10000, 10**7)


def test_a_dense_matrix_takes_cells_x_genes_x_itemsize():
    assert pf.Matrix("X", 100, 50, False, 0, "float64").nbytes == 100 * 50 * 8


# --- estimate_step_peak: the pure function for a pre-step hook ------------------------------


def test_step_peak_scaling_holds_two_dense_float32_copies():
    n, m = 5000, 12000
    t = pf.estimate_step_peak("scaling", (n, m), 5_000_000, False)
    assert t >= 2 * n * m * 4  # the dense X and sc.pp.scale's working copy


def test_step_peak_doublets_counts_the_layers_it_copies():
    n, m = 4000, 10000
    base = pf.estimate_step_peak("doublet_detection", (n, m), 0, True)
    assert base >= 4 * n * m * 4  # four float32 copies of a dense X while it picks genes
    more = pf.estimate_step_peak("doublet_detection", (n, m), 0, True, other_bytes=n * 1000)
    assert more - base == n * 1000


def test_step_peak_unknown_step_and_params():
    assert pf.estimate_step_peak("umap", (100, 100), 100, False) == 0
    small = pf.estimate_step_peak("pca", (10000, 15000), 10**6, False,
                                  PipelineParams(n_top_genes=500))
    large = pf.estimate_step_peak("pca", (10000, 15000), 10**6, False,
                                  PipelineParams(n_top_genes=4000))
    assert small < large


def test_step_peak_for_reads_the_live_anndata():
    rng = np.random.default_rng(0)
    a = ad.AnnData(sp.random(300, 200, density=0.1, format="csr", dtype=np.float32, random_state=rng))
    without = pf.step_peak_for(a, "doublet_detection")
    a.layers["counts"] = a.X.copy()
    assert pf.step_peak_for(a, "doublet_detection") > without
    assert pf.step_peak_for(a, "no_such_step") == 0


# --- verdict ------------------------------------------------------------------------------


def _avail(free_gb=5.0, limit_gb=8.0, limited=True):
    return pf.Available(int(limit_gb * GiB), int((limit_gb - free_gb) * GiB), int(free_gb * GiB),
                        limited, "container limit" if limited else "machine")


@pytest.mark.parametrize("need_share, level", [(0.0, "fits"), (0.69, "fits"), (0.70, "tight"),
                                               (1.0, "tight"), (1.01, "wont_fit")])
def test_verdict_thresholds(need_share, level):
    avail = 10 * GiB
    assert pf._level(int(need_share * avail), avail) == level


def test_unknown_memory_and_nothing_free():
    assert pf._level(GiB, None) == "unknown"
    assert pf._level(GiB, 0) == "wont_fit"
    assert pf._level(-GiB, 0) == "fits"


def _verdict(need_gb, cached=pf.Cached(), alternatives=(), avail=None):
    return pf.make_verdict(need=int(need_gb * GiB), peak=int(need_gb * GiB), stage="doublet detection",
                           seconds=(60, 240), avail=avail or _avail(), cached=cached,
                           lead=f"About {need_gb} GB more than now at its peak (doublet detection)",
                           breakdown=[], notes=[], alternatives=list(alternatives))


def test_wont_fit_suggestions_in_order_and_only_those_that_help():
    v = _verdict(7.0, cached=pf.Cached(2, int(1.9 * GiB)),
                 alternatives=[("Skip doublet detection", int(3 * GiB)),
                               ("Use 1,000 variable genes instead of 2,000", int(6.99 * GiB)),
                               ("Subsample the cells: half of them", int(3.5 * GiB))])
    assert v["level"] == "wont_fit"
    s = v["suggestions"]
    assert s[0].startswith("Raise SCVIEW_MEMORY_GB in the .env file in the scView folder from 8.0 to at least")
    # used 3 GB + 7 GB / 0.7 = 13 GB, so the estimate's error still fits
    assert "at least 13 GB" in s[0]
    assert s[1].startswith("Close the 2 other open datasets (about 1.9 GB)")
    assert s[2] == "Skip doublet detection (about 3.0 GB more at its peak)."
    assert s[3] == "Subsample the cells: half of them (about 3.5 GB more at its peak)."
    assert len(s) == 4  # fewer variable genes would not help: left out


def test_the_sentence_and_the_cached_datasets():
    v = _verdict(1.0, cached=pf.Cached(1, int(0.5 * GiB)))
    assert v["level"] == "fits" and v["suggestions"] == []
    assert v["sentence"] == ("About 1.0 GB more than now at its peak (doublet detection), roughly "
                             "1 to 4 min. scView has 5.0 GB free of its 8.0 GB limit. 1 other open "
                             "dataset holds about 0.5 GB.")
    machine = _verdict(1.0, avail=_avail(limited=False))
    assert "of the 8.0 GB this machine gives it (no limit is set)" in machine["sentence"]


def test_closing_others_frees_enough():
    # 4.5 GB needed, 5 free: tight; with 3 GB held by others it would fit under 70%
    v = _verdict(4.5, cached=pf.Cached(2, 3 * GiB))
    assert v["level"] == "tight" and v["cached"]["frees_enough"]
    assert v["suggestions"][1].endswith("that would free enough.")


def test_tiny_amounts_do_not_read_as_zero():
    v = _verdict(1.0, cached=pf.Cached(1, 10 << 20))
    assert v["sentence"].endswith("1 other open dataset holds under 0.1 GB.")


@pytest.mark.parametrize("secs, text", [((2, 8), "a few seconds"), ((20, 80), "roughly 20 s to 1 min"),
                                        ((59.7, 240), "roughly 1 to 4 min"),
                                        ((120, 360), "roughly 2 to 6 min"),
                                        ((1800, 7200), "roughly 30 min to 2.0 h")])
def test_time_text(secs, text):
    assert pf.time_text(secs) == text


# --- memory and CPUs the container has ----------------------------------------------------


def test_file_cache_the_kernel_can_drop_is_not_counted_as_used(tmp_path):
    cg = tmp_path / "cg"
    cg.mkdir()
    (cg / "memory.max").write_text(f"{8 * GiB}\n")
    (cg / "memory.current").write_text(f"{3 * GiB}\n")
    (cg / "memory.stat").write_text(f"anon {GiB}\nfile {2 * GiB}\ninactive_file {GiB}\n")
    a = pf.available_memory(cg, tmp_path / "meminfo")
    assert (a.used_bytes, a.available_bytes, a.limited) == (2 * GiB, 6 * GiB, True)


def test_cpu_quota(tmp_path):
    (tmp_path / "cpu.max").write_text("200000 100000\n")
    assert pf.cpu_quota(tmp_path) == 2.0
    (tmp_path / "cpu.max").write_text("max 100000\n")
    assert pf.cpu_quota(tmp_path) >= 1.0
    # fewer CPUs than the calibration: slower; more: faster, by less than the ratio
    assert pf._cpu_factor(1) > 1 > pf._cpu_factor(8) >= 0.5


# --- endpoints --------------------------------------------------------------------------------


def _adata(n=40, m=30) -> ad.AnnData:
    rng = np.random.default_rng(0)
    a = ad.AnnData(sp.csr_matrix(rng.poisson(0.5, (n, m)).astype(np.float32)))
    a.obs_names = [f"c{i}" for i in range(n)]
    a.var_names = [f"g{j}" for j in range(m)]
    a.obs["cluster"] = pd.Categorical(["a", "b"] * (n // 2))
    return a


def _register(dm, did="pf1") -> str:
    d = Path(dm.data_dir) / "uploads" / did
    d.mkdir(parents=True)
    (d / "metadata.json").write_text(json.dumps({"id": did, "name": did, "status": "ready"}))
    _adata().write_h5ad(d / "data.h5ad")
    return did


async def test_dataset_preflight_unknown_dataset_is_404(client):
    r = await client.get(f"{API}/datasets/nope/preflight", params={"steps": "pca"})
    assert r.status_code == 404


async def test_dataset_preflight_unknown_step_is_422_listing_the_valid_names(client, dataset_manager):
    did = _register(dataset_manager)
    r = await client.get(f"{API}/datasets/{did}/preflight", params={"steps": "qc_metrics,umap"})
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert detail.startswith("Unknown step: umap.")
    assert all(s in detail for s in ALL_STEPS)
    r = await client.get(f"{API}/datasets/{did}/preflight", params={"steps": " , "})
    assert r.status_code == 422 and r.json()["detail"].startswith("No steps given.")


async def test_dataset_preflight_from_the_file_header(client, dataset_manager):
    did = _register(dataset_manager)
    r = await client.get(f"{API}/datasets/{did}/preflight",
                         params={"steps": "pca,qc_metrics,doublet_detection", "n_top_genes": 20})
    assert r.status_code == 200
    body = r.json()
    assert not body["in_memory"]  # read from the header, not loaded
    assert dataset_manager.get_dataset(did) is None
    assert [s["step"] for s in body["steps"]] == ["qc_metrics", "doublet_detection", "pca", "save"]
    v = body["verdict"]
    assert v["level"] in ("fits", "tight", "wont_fit", "unknown")
    assert "GB" in v["sentence"] and v["error_band"] == pf.ERROR_BAND
    assert body["header"]["n_cells"] == 40


async def test_dataset_preflight_uses_the_cached_dataset(client, dataset_manager):
    did = _register(dataset_manager)
    await dataset_manager.load_dataset(did)
    body = (await client.get(f"{API}/datasets/{did}/preflight", params={"steps": "scaling"})).json()
    assert body["in_memory"]
    # what the dataset holds now is not needed again
    assert body["verdict"]["need_bytes"] < body["verdict"]["peak_bytes"]


@pytest.fixture()
async def ingest_client(client, tmp_data_dir):
    from scview.core.ingestion.session import IngestSessionManager
    from scview.dependencies import get_ingest_session_manager
    from scview.main import app

    mgr = IngestSessionManager(str(tmp_data_dir))
    app.dependency_overrides[get_ingest_session_manager] = lambda: mgr
    yield client, mgr
    app.dependency_overrides.pop(get_ingest_session_manager, None)


async def test_ingest_preflight_unknown_session_is_404(ingest_client):
    client, _ = ingest_client
    assert (await client.get(f"{API}/ingest/session/abc123/preflight")).status_code == 404


async def test_ingest_preflight_per_unit_and_merge(ingest_client):
    client, mgr = ingest_client
    sid = mgr.create_session()
    _adata(40, 30).write_h5ad(mgr.files_dir(sid) / "one.h5ad")
    body = (await client.get(f"{API}/ingest/session/{sid}/preflight")).json()
    assert [u["label"] for u in body["units"]] == ["one"] and body["merge"] is None
    v = body["units"][0]["verdict"]
    assert v["sentence"].startswith("40 cells x 30 genes: Loading takes under 0.1 GB at its peak")
    assert body["units"][0]["header"]["matrices"][0]["nnz"] > 0
    _adata(20, 30).write_h5ad(mgr.files_dir(sid) / "two.h5ad")
    body = (await client.get(f"{API}/ingest/session/{sid}/preflight")).json()
    assert len(body["units"]) == 2
    assert body["merge"]["sentence"].startswith("Merging the 2 samples takes ")
    assert "GB at its peak and" in body["merge"]["sentence"]

