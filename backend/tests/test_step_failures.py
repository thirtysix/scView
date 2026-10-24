"""A failed pipeline step skips only the steps that need it (core/step_failures.py).

Before, every later step still ran after a failure (a failed PCA left the graph, clusters
and UMAP to be built from the previous run's PCs) and the run saved as "complete".
"""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from scview.core import chunked, large_run, pipeline, provenance, replay_script, step_failures
from scview.core.pipeline import (
    CELL_CYCLE_G2M_GENES,
    CELL_CYCLE_S_GENES,
    PipelineParams,
    run_pipeline,
    run_pipeline_streamed,
)
from scview.core.uns_keys import markers_key

API = "/api/v1"
PREP = ["normalization", "log_transform", "highly_variable_genes", "scaling", "pca"]
AFTER_PCA = ["neighbors", "clustering", "embeddings", "marker_genes"]


def _counts(n_per: int = 80, seed: int = 0) -> ad.AnnData:
    """Counts in three planted groups, with cell-cycle genes and a batch column."""
    rng = np.random.default_rng(seed)
    names = list(CELL_CYCLE_S_GENES[:12]) + list(CELL_CYCLE_G2M_GENES[:12]) + [f"G{i}" for i in range(120)]
    n = 3 * n_per
    X = rng.poisson(rng.gamma(4.0, 0.5, size=(n, 1)) * rng.gamma(1.0, 1.0, size=len(names)))
    for k in range(3):
        X[k * n_per:(k + 1) * n_per, 30 + 15 * k:45 + 15 * k] += rng.poisson(6.0, (n_per, 15))
    a = ad.AnnData(sp.csr_matrix(X.astype("float32")))
    a.var_names = names
    a.obs_names = [f"C{i}" for i in range(n)]
    a.obs["batch"] = pd.Categorical(["b1", "b2"] * (n // 2))
    return a


def _params(**kw) -> PipelineParams:
    return PipelineParams(n_top_genes=60, n_comps=10, **kw)


def _processed() -> ad.AnnData:
    """A dataset processed once: PCs, graph, clusters, UMAP and markers to go stale."""
    out, res = run_pipeline(_counts(), [*PREP, *AFTER_PCA, "cell_cycle"], _params())
    assert res.status == "complete" and not res.errors
    return out


def _fail(monkeypatch, step: str, message: str = "boom") -> None:
    def broken(*_a, **_k):
        raise RuntimeError(message)

    monkeypatch.setitem(pipeline._STEP_RUNNERS, step, broken)


# --- the map ------------------------------------------------------------------------------


def test_the_map_names_only_real_steps_and_runs_forward():
    order = pipeline.ALL_STEPS
    assert set(step_failures.DEPENDS_ON) == set(order)
    for step, needs in step_failures.DEPENDS_ON.items():
        assert all(order.index(n) < order.index(step) for n in needs), step


def test_what_each_step_needs():
    needs = step_failures.needs
    assert needs("pca") == {"highly_variable_genes", "scaling", "log_transform", "normalization",
                            "filtering", "reset_to_counts"}
    # Scrublet reads the counts, not the QC metrics; filtering works them out when missing
    assert needs("doublet_detection") == {"reset_to_counts"}
    assert "qc_metrics" not in needs("filtering")
    assert "doublet_detection" not in needs("filtering")
    assert "doublet_detection" in needs("filtering", _params(drop_doublets=True))
    # A failed batch correction leaves X_pca, which the graph is then built on
    assert "batch_correction" not in needs("neighbors")
    assert "batch_correction" not in needs("marker_genes")
    # Scaling plans for the genes PCA reads; it does not need them first
    assert "highly_variable_genes" not in needs("scaling")
    # The counts-ranking flavours do not need the normalisation
    assert "normalization" in needs("highly_variable_genes")
    assert "normalization" not in needs("highly_variable_genes", _params(hvg_flavor="seurat_v3"))
    assert "normalization" not in needs("highly_variable_genes",
                                        _params(normalization_method="pearson_residuals"))
    # Scores and annotation by another column do not need the clustering
    assert needs("cell_cycle") == needs("log_transform") | {"log_transform"}
    assert "clustering" in needs("cell_type_annotation")
    assert "clustering" not in needs("cell_type_annotation", _params(annotation_groupby="ct"))


# --- in memory ----------------------------------------------------------------------------


def test_a_failed_pca_skips_what_needs_it_and_the_rest_runs(monkeypatch):
    a = _processed()
    active = a.uns["scview_active_clustering"]
    _fail(monkeypatch, "pca")
    out, res = run_pipeline(a, [*PREP, "batch_correction", *AFTER_PCA, "cell_cycle"],
                            _params(batch_key="batch"))
    assert res.status == "finished_with_errors"
    assert res.errors == {"pca": "boom"}
    assert res.skipped == {s: "pca" for s in ["batch_correction", *AFTER_PCA]}
    assert res.steps_skipped == ["batch_correction", *AFTER_PCA]
    assert res.steps_run == ["normalization", "log_transform", "highly_variable_genes", "scaling",
                             "cell_cycle"]
    # The previous run's PCs, graph, UMAP, markers and active clustering are stale (new
    # variable genes and scaling ran): removed, not left looking current.
    for k in ("X_pca", "X_umap", "X_umap_3d"):
        assert k not in out.obsm, k
    assert "connectivities" not in out.obsp and "neighbors" not in out.uns
    assert "scview_active_clustering" not in out.uns
    assert markers_key(active) not in out.uns and "rank_genes_groups" not in out.uns
    assert active in out.obs.columns  # the labels stay as a column
    assert "S_score" in out.obs.columns  # the cell cycle did not need the PCA
    assert set(res.cleared["obsm"]) >= {"X_pca", "X_umap", "X_umap_3d"}
    # Provenance: the run's outcome after its steps, and a current state without the stale
    prov = provenance.read_provenance(out)
    last = prov["history"][-1]
    assert last["step"] == "run_errors"
    assert last["params"]["failed"] == {"pca": "boom"}
    assert last["params"]["skipped"]["neighbors"] == "pca"
    assert "X_pca" in last["params"]["cleared"]["obsm"]
    cur = prov["current"]
    assert cur["clustering"] is None and cur["embeddings"] == [] and cur["markers_for"] == []
    assert cur["pca"] is False
    assert cur["last_run"] == {"status": "finished_with_errors", "failed": ["pca"],
                               "skipped": ["batch_correction", *AFTER_PCA]}
    assert last["params"]["stale"] == ["pca", "batch_correction", *AFTER_PCA]
    assert provenance.reconcile(out) == []
    # Data Assessment: the removed steps are no longer done; those that ran are
    from scview.core.assessor import assess_preprocessing

    state = assess_preprocessing(out)
    assert not state.pca.done and not state.neighbors.done and not state.embeddings.done
    assert not state.marker_genes.done
    assert state.highly_variable_genes.done and state.cell_cycle.done


def test_a_failed_batch_correction_is_tolerated(monkeypatch):
    a = _counts()
    out, res = run_pipeline(a, [*PREP, "batch_correction", "neighbors", "clustering"],
                            _params(batch_key="no_such_column"))
    assert res.status == "finished_with_errors"
    assert list(res.errors) == ["batch_correction"] and res.skipped == {}
    assert res.steps_run[-2:] == ["neighbors", "clustering"]
    assert "X_pca_harmony" not in out.obsm
    assert out.uns["neighbors"]["params"]["use_rep"] == "X_pca"


def test_a_failed_reclustering_alone_keeps_the_clustering_there_was(monkeypatch):
    """Nothing the clustering needs ran: the old clusters and markers still describe
    the data, so they stay."""
    a = _processed()
    active = a.uns["scview_active_clustering"]
    _fail(monkeypatch, "clustering", "Leiden clustering failed (exit 1)")
    out, res = run_pipeline(a, ["clustering", "marker_genes", "cell_cycle"],
                            _params(clustering_resolution=1.4))
    assert res.status == "finished_with_errors"
    assert res.skipped == {"marker_genes": "clustering"} and res.steps_run == ["cell_cycle"]
    assert res.cleared == {}
    assert out.uns["scview_active_clustering"] == active
    assert markers_key(active) in out.uns


def test_a_failed_filtering_skips_the_whole_analysis_and_saves_what_ran(monkeypatch, tmp_path):
    _fail(monkeypatch, "filtering", "bad per-type column")
    path = tmp_path / "out.h5ad"
    out, res = run_pipeline(_counts(), ["qc_metrics", "doublet_detection", "filtering", *PREP,
                                        "neighbors", "clustering", "cell_cycle"],
                            _params(), output_path=str(path))
    assert res.status == "finished_with_errors"
    assert res.steps_run == ["qc_metrics", "doublet_detection"]
    assert set(res.skipped) == {*PREP, "neighbors", "clustering", "cell_cycle"}
    assert res.output_path == str(path)
    saved = ad.read_h5ad(path)
    assert "doublet_score" in saved.obs.columns and "X_pca" not in saved.obsm


def test_without_a_failure_the_run_is_complete():
    out, res = run_pipeline(_counts(), [*PREP, "neighbors"], _params())
    assert res.status == "complete" and res.skipped == {} and res.cleared == {}
    assert provenance.read_provenance(out)["history"][-1]["step"] == "neighbors"


def test_the_stream_says_which_steps_were_skipped_and_why(monkeypatch):
    a = _processed()
    _fail(monkeypatch, "highly_variable_genes", "loess failed")
    events = list(run_pipeline_streamed(a, [*PREP, "neighbors", "cell_cycle"], _params()))
    kinds = [(e, d.get("step")) for e, d in events if e != "substep"]
    assert ("step_error", "highly_variable_genes") in kinds
    skipped = [d for e, d in events if e == "step_skipped"]
    assert [d["step"] for d in skipped] == ["pca", "neighbors"]
    assert skipped[0] == {"step": "pca", "index": 4, "total": 7, "because": "highly_variable_genes",
                          "reason": "Skipped: it needs highly variable genes, which failed."}
    assert ("step_done", "scaling") in kinds and ("step_done", "cell_cycle") in kinds
    assert not any(e == "step_start" and s in ("pca", "neighbors") for e, s in kinds)
    final = events[-1]
    assert final[0] == "complete" and final[1]["status"] == "finished_with_errors"
    assert final[1]["skipped"] == {"pca": "highly_variable_genes", "neighbors": "highly_variable_genes"}
    assert "highly_variable" in final[1]["cleared"]["var"]  # the old variable genes went


def test_the_replay_removes_what_the_run_removed(monkeypatch, tmp_path):
    import subprocess
    import sys

    inp = _counts()
    a, res = run_pipeline(inp.copy(), [*PREP, *AFTER_PCA], _params())
    assert res.status == "complete"
    _fail(monkeypatch, "pca")
    out, res = run_pipeline(a, [*PREP, *AFTER_PCA], _params(target_sum=5e3))
    assert res.status == "finished_with_errors"
    assert replay_script.describe(out)[-1]["status"] == "exact"
    script = replay_script.build_script(out, dataset_id="t", dataset_name="t")
    assert "#   pca failed: boom" in script and "#   neighbors skipped: it needs pca" in script
    src, path, dst = tmp_path / "in.h5ad", tmp_path / "replay.py", tmp_path / "rep.h5ad"
    inp.write_h5ad(src)
    path.write_text(script)
    proc = subprocess.run([sys.executable, str(path), str(src), str(dst)], capture_output=True,
                          text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr[-3000:]
    rep = ad.read_h5ad(dst)
    assert sorted(rep.obsm) == sorted(out.obsm) == []
    assert "scview_active_clustering" not in rep.uns and "neighbors" not in rep.uns


async def test_the_run_endpoints_report_finished_with_errors(client, dataset_manager):
    d = Path(dataset_manager.data_dir) / "uploads" / "sf1"
    d.mkdir(parents=True)
    (d / "metadata.json").write_text(json.dumps({"id": "sf1", "name": "sf1", "status": "ready"}))
    _counts().write_h5ad(d / "data.h5ad")
    body = {"steps": [*PREP, "batch_correction", "neighbors"],
            "params": {"n_top_genes": 60, "n_comps": 10, "batch_key": "missing"}}
    r = await client.post(f"{API}/datasets/sf1/assessment/run", json=body)
    assert r.status_code == 200
    res = r.json()
    assert res["status"] == "finished_with_errors" and list(res["errors"]) == ["batch_correction"]
    assert res["output_path"]  # saved
    r = await client.post(f"{API}/datasets/sf1/assessment/run-stream",
                          json={"steps": ["pca", "neighbors"], "params": {"n_comps": 500000}})
    text = r.text
    assert "event: step_error" in text and "event: step_skipped" in text
    final = json.loads(text.rsplit("event: complete\ndata: ", 1)[1].split("\n", 1)[0])
    assert final["status"] == "finished_with_errors" and final["skipped"] == {"neighbors": "pca"}


# --- large-dataset mode --------------------------------------------------------------------


@pytest.fixture()
def _blocks(monkeypatch):
    monkeypatch.setattr(chunked, "rows_per_block", lambda *a, **k: 400)


def _large_input(tmp_path: Path) -> Path:
    from tests.test_large_run import _counts as large_counts
    from tests.test_large_run import _write

    return _write(large_counts(n=1100), tmp_path / "in" / "data.h5ad")


LARGE = ["qc_metrics", "filtering", "normalization", "log_transform", "highly_variable_genes",
         "scaling", "pca", "neighbors", "clustering", "cell_cycle"]


def test_large_mode_skips_what_needs_a_failed_step(monkeypatch, tmp_path, _blocks):
    def broken(self):
        raise RuntimeError("PCA did not converge")

    monkeypatch.setattr(large_run.LargeRun, "_pca", broken)
    src = _large_input(tmp_path)
    out = tmp_path / "derived" / "data.h5ad"
    res = large_run.run(src, LARGE, PipelineParams(n_top_genes=200, n_comps=10, chunked="on"),
                        str(out))
    assert res.status == "finished_with_errors" and res.chunked
    assert res.errors == {"pca": "PCA did not converge"}
    assert res.skipped == {"neighbors": "pca", "clustering": "pca"}
    assert res.steps_run[-1] == "cell_cycle"
    saved = ad.read_h5ad(out, backed="r")
    assert "S_score" in saved.obs.columns and "X_pca" not in saved.obsm
    assert provenance.read_provenance(saved)["history"][-1]["step"] == "run_errors"


def test_large_mode_saves_nothing_when_x_could_not_be_log_normalised(monkeypatch, tmp_path, _blocks):
    def broken(self):
        raise RuntimeError("bad target sum")

    monkeypatch.setattr(large_run.LargeRun, "_normalization", broken)
    src = _large_input(tmp_path)
    out = tmp_path / "derived" / "data.h5ad"
    res = large_run.run(src, LARGE, PipelineParams(n_top_genes=200, n_comps=10, chunked="on"),
                        str(out))
    assert res.status == "stopped" and res.output_path is None and not out.exists()
    assert "Nothing was saved: normalization failed" in res.stop_reason
    assert res.skipped["log_transform"] == "normalization"
