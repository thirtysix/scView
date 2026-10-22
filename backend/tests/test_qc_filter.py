"""Cell QC thresholds and their loss preview (core/qc_filter.py, GET /qc/filter-preview).

The preview must be what the filtering step does (same masks, same per-cell
values), per sample and per cluster, with a warning when a threshold empties a
group or takes more than half of it; and the step gains max-genes / min-counts
and judges cells on their counts even when X is no longer counts.
"""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
import scipy.sparse as sp

from scview.core import qc_filter
from scview.core.anndata_adaptor import per_cell_qc
from scview.core.pipeline import PipelineParams, _run_filtering, run_pipeline
from scview.core.qc_filter import Thresholds

API = "/api/v1"


def _fixed(**kw) -> PipelineParams:
    """Params for the fixed cut-offs (MAD per sample is the default)."""
    return PipelineParams(qc_threshold_mode="fixed", **kw)


def _cells(genes, counts, pct=None, has_mt=True) -> dict:
    genes = np.asarray(genes, dtype=float)
    return {"n_genes": genes, "total_counts": np.asarray(counts, dtype=float),
            "pct_mt": None if pct is None else np.asarray(pct, dtype=float), "has_mt": has_mt}


def _dataset(n: int = 800, seed: int = 0, mt: bool = True) -> ad.AnnData:
    """Counts with 4 donors and 4 clusters; cluster '3' has few genes and much mito."""
    rng = np.random.default_rng(seed)
    n_genes = 300
    lam = np.full(n, 1.0)
    clusters = np.repeat(["0", "1", "2", "3"], n // 4)
    lam[clusters == "3"] = 0.25  # small cells: few genes detected
    X = rng.poisson(lam[:, None] * np.ones(n_genes)).astype("float32")
    names = [f"G{i}" for i in range(n_genes)]
    if mt:
        names[:5] = ["MT-CO1", "MT-CO2", "MT-ND1", "MT-ND2", "MT-ATP6"]
        X[clusters == "3", :5] += rng.poisson(20, ((clusters == "3").sum(), 5))
    a = ad.AnnData(sp.csr_matrix(X))
    a.var_names = names
    a.obs_names = [f"c{i}" for i in range(n)]
    a.obs["Donor"] = pd.Categorical(np.tile(["d1", "d2", "d3", "d4"], n // 4))
    a.obs["leiden"] = pd.Categorical(clusters)
    return a


# --- the masks -------------------------------------------------------------------


def test_each_threshold_removes_what_it_says_and_zero_is_off():
    cells = _cells(genes=[100, 250, 900, 5000], counts=[300, 800, 4000, 30000], pct=[2, 30, 5, 10])
    removed = qc_filter.removed_by(cells, Thresholds(min_genes=200, max_genes=4000,
                                                     min_counts=500, max_pct_mt=20))
    assert {k: v.tolist() for k, v in removed.items()} == {
        "min_genes": [True, False, False, False],
        "max_genes": [False, False, False, True],
        "min_counts": [True, False, False, False],
        "max_pct_mt": [False, True, False, False],
    }
    assert qc_filter.keep_mask(removed, 4).tolist() == [False, False, True, False]
    assert qc_filter.removed_by(cells, Thresholds()) == {}


def test_bounds_match_scanpy_and_the_old_mito_rule():
    cells = _cells(genes=[200, 199, 4000, 4001], counts=[500, 499, 1, 1], pct=[20, 19.99, 0, 0])
    removed = qc_filter.removed_by(cells, Thresholds(min_genes=200, max_genes=4000,
                                                     min_counts=500, max_pct_mt=20))
    assert removed["min_genes"].tolist() == [False, True, False, False]  # >= kept, as filter_cells
    assert removed["max_genes"].tolist() == [False, False, False, True]  # <= kept
    assert removed["min_counts"][:2].tolist() == [False, True]
    assert removed["max_pct_mt"][:2].tolist() == [True, False]  # strictly below kept


def test_unmeasurable_mito_and_missing_values_remove_nothing():
    cells = _cells(genes=[np.nan, 50], counts=[10, 10], pct=[0, 0], has_mt=False)
    removed = qc_filter.removed_by(cells, Thresholds(min_genes=100, max_pct_mt=0))
    assert "max_pct_mt" not in removed
    assert removed["min_genes"].tolist() == [False, True]


def test_doublets_are_removed_only_when_asked():
    cells = _cells(genes=[300, 300], counts=[900, 900])
    calls = np.array([True, False])
    assert qc_filter.removed_by(cells, Thresholds(), calls) == {}
    assert qc_filter.removed_by(cells, Thresholds(drop_doublets=True), calls)["doublets"].tolist() == [True, False]


# --- the preview -----------------------------------------------------------------


def test_preview_counts_per_sample_and_cluster_and_warns_on_a_lost_cluster():
    a = _dataset()
    cells = per_cell_qc(a)
    out = qc_filter.preview(a, cells, Thresholds(min_genes=150, max_pct_mt=20))
    assert (out["samples"]["column"], out["samples"]["reason"]) == ("Donor", "donor")
    assert (out["clusters"]["column"], out["clusters"]["reason"]) == ("leiden", "clustering")
    assert sum(g["n_removed"] for g in out["samples"]["groups"]) == out["n_removed"]
    assert sum(g["n_removed"] for g in out["clusters"]["groups"]) == out["n_removed"]
    by_cluster = {g["name"]: g for g in out["clusters"]["groups"]}
    assert by_cluster["3"]["fraction"] == 1.0 and by_cluster["0"]["n_removed"] == 0
    whole = [w for w in out["warnings"] if w["whole"]]
    assert [(w["kind"], w["group"]) for w in whole] == [("cluster", "3")]
    assert whole[0]["message"].startswith("Removes all 200 cells of cluster 3, mostly by ")
    assert out["by_criterion"]["min_genes"]["n_removed"] == 200


def test_more_than_half_warns_and_less_does_not():
    a = _dataset(mt=False)
    genes = per_cell_qc(a)["n_genes"]
    c0 = a.obs["leiden"].to_numpy() == "0"

    def cluster_warnings(q: float) -> dict:
        cut = float(np.quantile(genes[c0], q))
        out = qc_filter.preview(a, per_cell_qc(a), Thresholds(min_genes=cut))
        return {w["group"]: w for w in out["warnings"] if w["kind"] == "cluster"}

    most = cluster_warnings(0.65)  # clusters 0-2 alike: each loses well over half
    assert set(most) == {"0", "1", "2", "3"} and most["3"]["whole"] and not most["0"]["whole"]
    assert most["0"]["fraction"] > 0.5
    assert re.fullmatch(r"Removes \d+% of cluster 0 \(\d+ of 200 cells\), by Min genes\.",
                        most["0"]["message"])
    assert list(cluster_warnings(0.3)) == ["3"]  # a third of 0-2 is no warning; 3 is gone whole


def test_mito_is_left_out_of_the_preview_when_unmeasurable():
    a = _dataset(mt=False)
    out = qc_filter.preview(a, per_cell_qc(a), Thresholds(max_pct_mt=0))
    assert out["thresholds"]["max_pct_mt"] is None and out["has_mt"] is False
    assert out["n_removed"] == 0 and out["by_criterion"] == {}


def test_preview_is_what_the_filtering_step_removes():
    a = _dataset()
    run_pipeline(a, ["qc_metrics"], PipelineParams())  # obs QC columns, as the plots read them
    a.obs["predicted_doublet"] = np.arange(a.n_obs) % 50 == 0
    params = _fixed(min_genes=150, max_genes=290, min_counts=200, max_pct_mt=15,
                    drop_doublets=True, min_cells=0)
    out = qc_filter.preview(a, per_cell_qc(a), Thresholds.from_params(params))
    kept, _ = _run_filtering(a.copy(), params)
    assert kept.n_obs == out["n_kept"] and out["n_removed"] > 0
    assert set(out["by_criterion"]) == {"min_genes", "max_genes", "min_counts", "max_pct_mt", "doublets"}


def test_named_columns_and_a_clustering_equal_to_the_samples():
    a = _dataset()
    out = qc_filter.preview(a, per_cell_qc(a), Thresholds(min_genes=150),
                            sample_column="leiden", cluster_column="leiden")
    assert out["samples"]["column"] == "leiden" and out["clusters"] is None
    with pytest.raises(ValueError, match="not found"):
        qc_filter.preview(a, per_cell_qc(a), Thresholds(), sample_column="nope")


def test_too_many_clusters_are_summarised_by_a_note():
    a = _dataset(n=1000)
    a.obs["leiden"] = pd.Categorical([str(i % 250) for i in range(1000)])
    out = qc_filter.preview(a, per_cell_qc(a), Thresholds(min_genes=150))
    assert out["clusters"]["groups"] == [] and "250 groups" in out["clusters"]["note"]


def test_the_preview_is_fast_at_200k_cells():
    import time

    rng = np.random.default_rng(1)
    n = 200_000
    a = ad.AnnData(np.zeros((n, 1), dtype="float32"))
    a.obs["Donor"] = pd.Categorical(rng.choice([f"d{i}" for i in range(16)], n))
    a.obs["leiden"] = pd.Categorical(rng.integers(0, 40, n).astype(str))
    cells = _cells(genes=rng.integers(50, 6000, n), counts=rng.integers(100, 40000, n),
                   pct=rng.uniform(0, 40, n))
    t = Thresholds(min_genes=200, max_genes=5000, min_counts=500, max_pct_mt=20)
    qc_filter.preview(a, cells, t)  # warm-up
    t0 = time.perf_counter()
    out = qc_filter.preview(a, cells, t)
    took = time.perf_counter() - t0
    assert out["n_cells"] == n and len(out["clusters"]["groups"]) == 40
    assert took < 1.0, f"{took:.3f} s"  # ~0.05 s here; a debounced drag must keep up
    assert len(json.dumps(out)) < 80_000  # ~67 KB: the removal audit adds a 40 x 16 table


# --- the filtering step ----------------------------------------------------------


def test_the_step_takes_max_genes_and_min_counts():
    p = PipelineParams()
    assert (p.max_genes, p.min_counts) == (0, 0)  # off by default
    a = _dataset(mt=False)
    genes = per_cell_qc(a)["n_genes"]
    counts = per_cell_qc(a)["total_counts"]
    out, _ = _run_filtering(a.copy(), _fixed(min_genes=0, min_cells=0, max_genes=250))
    assert out.n_obs == int((genes <= 250).sum())
    out, _ = _run_filtering(a.copy(), _fixed(min_genes=0, min_cells=0, min_counts=280))
    assert out.n_obs == int((counts >= 280).sum())


def test_the_step_judges_cells_on_counts_when_x_is_scaled():
    """Re-running filtering on a processed dataset: X holds z-scores, and counting the
    genes above zero there counts the genes above their mean, not the genes detected
    (a low count of a well-expressed gene is below its mean). The QC histogram shows
    detected genes, so the threshold set on it must cut those."""
    a = _dataset(mt=False)
    rng = np.random.default_rng(3)
    X = a.X.toarray()
    # 50 well-expressed genes, more or less of each per cell (cells differ in size).
    X[:, 10:60] = rng.poisson(rng.uniform(2, 30, (a.n_obs, 1)) * np.ones((1, 50)))
    a.X = sp.csr_matrix(X)
    a.layers["counts"] = a.X.copy()
    sc.pp.normalize_total(a)
    sc.pp.log1p(a)
    sc.pp.scale(a)
    genes = per_cell_qc(a)["n_genes"]
    above_mean = np.asarray((a.X > 0).sum(axis=1)).ravel()
    assert (genes >= 180).sum() > (above_mean >= 180).sum() + 100  # the two disagree here
    out, _ = _run_filtering(a.copy(), _fixed(min_genes=180, min_cells=0))
    assert out.n_obs == int((genes >= 180).sum())
    out, _ = _run_filtering(a, _fixed(min_genes=0, min_cells=0, max_genes=210))
    assert out.n_obs == int((genes <= 210).sum()) > 0


def test_the_step_skips_mito_when_unmeasurable():
    a = _dataset(mt=False)
    a.obs["pct_counts_mt"] = 0.0  # what the QC step writes without MT- genes
    out, _ = _run_filtering(a, _fixed(min_genes=0, min_cells=0, max_pct_mt=0))
    assert out.n_obs == a.n_obs


# --- the endpoint ----------------------------------------------------------------


def _register(dm, a: ad.AnnData, did: str = "qcf") -> str:
    d = Path(dm.data_dir) / "uploads" / did
    d.mkdir(parents=True)
    (d / "metadata.json").write_text(json.dumps({"id": did, "name": did, "status": "ready"}))
    a.write_h5ad(d / "data.h5ad")
    return did


async def test_endpoint_previews_in_a_worker_thread(client, dataset_manager, monkeypatch):
    did = _register(dataset_manager, _dataset())
    threads: list[threading.Thread] = []
    real = qc_filter.preview

    def spy(*args, **kwargs):
        threads.append(threading.current_thread())
        return real(*args, **kwargs)

    monkeypatch.setattr(qc_filter, "preview", spy)
    r = await client.get(f"{API}/datasets/{did}/qc/filter-preview",
                         params={"mode": "fixed", "min_genes": 150, "max_pct_mt": 20,
                                 "max_genes": 0})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["thresholds"]["min_genes"] == 150 and body["n_removed"] >= 200
    assert any(w["whole"] and w["group"] == "3" for w in body["warnings"])
    assert len(threads) == 1 and threads[0] is not threading.main_thread()


async def test_endpoint_defaults_to_the_pipelines_thresholds(client, dataset_manager):
    did = _register(dataset_manager, _dataset())
    body = (await client.get(f"{API}/datasets/{did}/qc/filter-preview")).json()
    p = PipelineParams()
    assert body["thresholds"] == {"mode": "mad", "min_genes": p.min_genes, "max_genes": 0,
                                  "min_counts": 0, "max_pct_mt": p.max_pct_mt,
                                  "drop_doublets": False, "mad_k": 5.0, "mad_k_mt": 3.0,
                                  "mad_upper": False, "mito_mode": "global", "mito_type_column": "",
                                  "mito_type_floor": False}
    assert body["mad"]["column"] == "Donor" and len(body["mad"]["samples"]) == 4
    r = await client.get(f"{API}/datasets/{did}/qc/filter-preview", params={"sample_column": "nope"})
    assert r.status_code == 400
    for bad in ({"mode": "strict"}, {"mad_k": 0}, {"mad_k_mt": -1}):
        r = await client.get(f"{API}/datasets/{did}/qc/filter-preview", params=bad)
        assert r.status_code == 400, bad
    r = await client.get(f"{API}/datasets/ghost/qc/filter-preview")
    assert r.status_code == 404
