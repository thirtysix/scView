"""C006: the replay script and RUN_MANIFEST written from a dataset's provenance."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from scview.core import provenance, replay_script, resolution_sweep, run_manifest
from scview.core.pipeline import PipelineParams, run_pipeline
from scview.core.subcluster import subcluster

API = "/api/v1"


def _counts(n_per: int = 70, n_genes: int = 150, seed: int = 0) -> ad.AnnData:
    """Three populations with their own gene blocks, two samples, MT- genes, a few
    low-count cells for the filter to take."""
    rng = np.random.default_rng(seed)
    parts = []
    for lo in (0, 40, 80):
        X = rng.poisson(0.3, size=(n_per, n_genes)).astype("float32")
        X[:, lo:lo + 40] += rng.poisson(5.0, size=(n_per, 40)).astype("float32")
        parts.append(X)
    X = np.vstack(parts)
    X[:6] = rng.poisson(0.05, size=(6, n_genes))  # nearly empty droplets
    a = ad.AnnData(X)
    a.var_names = [f"MT-{i}" if i >= n_genes - 5 else f"G{i}" for i in range(n_genes)]
    a.obs_names = [f"C{i}" for i in range(X.shape[0])]
    a.obs["sample"] = pd.Categorical(np.where(np.arange(X.shape[0]) % 2, "s1", "s2"))
    return a


def _replay(tmp_path: Path, inp: ad.AnnData, out: ad.AnnData) -> ad.AnnData:
    """Write the input and the script, run the script in a fresh interpreter."""
    src, script, dst = tmp_path / "input.h5ad", tmp_path / "replay.py", tmp_path / "replayed.h5ad"
    inp.write_h5ad(src)
    script.write_text(replay_script.build_script(out, dataset_id="t", dataset_name="test"))
    proc = subprocess.run([sys.executable, str(script), str(src), str(dst)],
                          capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr[-3000:]
    return ad.read_h5ad(dst)


def test_script_reproduces_the_pipeline_and_follow_up_steps(tmp_path):
    inp = _counts()
    a = inp.copy()
    params = PipelineParams(n_top_genes=60, n_comps=15, batch_key="sample", qc_mad_upper=True,
                            qc_sample_column="sample", min_cells=3)
    steps = ["qc_metrics", "filtering", "normalization", "log_transform", "highly_variable_genes",
             "scaling", "pca", "batch_correction", "neighbors", "clustering", "marker_genes"]
    out, res = run_pipeline(a, steps, params)
    assert res.errors == {}
    col = out.uns["scview_active_clustering"]
    subcluster(out, col, "0", 0.5, batch_key="sample")
    resolution_sweep.keep(out, "leiden", 1.5)
    assert {d["status"] for d in replay_script.describe(out)} == {"exact"}

    rep = _replay(tmp_path, inp, out)
    assert out.n_obs < inp.n_obs  # the filter took the empty droplets
    assert list(rep.obs_names) == list(out.obs_names)
    assert list(rep.var_names) == list(out.var_names)
    assert (rep.var["highly_variable"].to_numpy() == out.var["highly_variable"].to_numpy()).all()
    np.testing.assert_allclose(np.abs(rep.obsm["X_pca"]), np.abs(out.obsm["X_pca"]), atol=1e-3)
    for c in (col, f"{col}_sub", resolution_sweep.column_name("leiden", 1.5)):
        assert (rep.obs[c].astype(str) == out.obs[c].astype(str)).all(), c
    key = f"rank_genes_groups__{col}"
    assert list(rep.uns[key]["names"][0]) == list(out.uns[key]["names"][0])


def test_fast_leiden_replays_exactly_with_scanpys_n_iterations(tmp_path):
    """sc.tl.leiden(..., n_iterations=2) gives fast Leiden's labels: the clustering, a
    resolution kept from the sweep and a sub-clustering, which follow its setting."""
    inp = _counts(seed=2)
    params = PipelineParams(n_top_genes=60, n_comps=15, qc_threshold_mode="fixed", min_genes=20,
                            leiden_iterations=2)
    steps = ["qc_metrics", "filtering", "normalization", "log_transform", "highly_variable_genes",
             "scaling", "pca", "neighbors", "clustering"]
    out, res = run_pipeline(inp.copy(), steps, params)
    assert res.errors == {}
    col = out.uns["scview_active_clustering"]
    subcluster(out, col, "0", 0.8)
    resolution_sweep.keep(out, "leiden", 1.5)
    script = replay_script.build_script(out, dataset_id="t", dataset_name="test")
    assert script.count("n_iterations=2") == 3, script  # clustering, sub-cluster, kept resolution
    assert {d["status"] for d in replay_script.describe(out)} == {"exact"}
    m = run_manifest.build_manifest(out, dataset_id="t", dataset_name="test")
    assert next(s for s in m["steps"] if s["step"] == "clustering")["params"]["leiden_iterations"] == 2

    rep = _replay(tmp_path, inp, out)
    for c in (col, f"{col}_sub", resolution_sweep.column_name("leiden", 1.5)):
        assert (rep.obs[c].astype(str) == out.obs[c].astype(str)).all(), c
    assert rep.uns[col]["params"]["n_iterations"] == 2


def test_pearson_residuals_and_louvain_replay(tmp_path):
    inp = _counts(seed=1)
    params = PipelineParams(normalization_method="pearson_residuals", clustering_method="louvain",
                            qc_threshold_mode="fixed", min_genes=20, n_top_genes=60, n_comps=15)
    steps = ["qc_metrics", "filtering", "normalization", "log_transform", "highly_variable_genes",
             "scaling", "pca", "neighbors", "clustering"]
    out, res = run_pipeline(inp.copy(), steps, params)
    assert res.errors == {}
    rep = _replay(tmp_path, inp, out)
    assert rep.n_obs == out.n_obs
    np.testing.assert_allclose(rep.X, out.X, atol=1e-5)  # the residuals themselves
    col = out.uns["scview_active_clustering"]
    assert col.startswith("scview_louvain")
    assert (rep.obs[col].astype(str) == out.obs[col].astype(str)).all()


def _with_history(steps: list[tuple[str, dict]]) -> ad.AnnData:
    a = ad.AnnData(np.zeros((3, 2), dtype="float32"))
    for name, params in steps:
        provenance.record_step(a, step=name, tool="t", params=params)
    return a


# Every step type with the params its recorder writes (and none at all).
_STEPS = [
    ("ingest", {"formats": ["tenx_h5"], "genes_in_rows": False, "n_proteins": 17}),
    ("integrate", {"datasets": ["a", "b"], "join": "inner", "method": "harmony"}),
    ("reset_to_counts", {}),
    ("qc_metrics", {}),
    ("doublet_detection", {"doublet_method": "scrublet", "expected_doublet_rate": 0.06,
                           "expected_rate_from": "sample"}),
    ("doublet_detection", {"doublet_method": "scrublet", "expected_doublet_rate": 0.06,
                           "expected_rate_from": "sample", "neighbors": "exact"}),
    ("filtering", {"qc_threshold_mode": "mad", "qc_mad_k": 5.0, "qc_mad_k_mt": 3.0,
                   "qc_mad_upper": False, "qc_sample_column": "", "min_cells": 3,
                   "max_pct_mt": 20.0, "drop_doublets": True}),
    ("filtering", {"qc_threshold_mode": "fixed", "min_genes": 200, "max_genes": 0,
                   "min_counts": 0, "min_cells": 3, "max_pct_mt": 20.0}),
    ("normalization", {"target_sum": 10000.0}),
    ("log_transform", {}),
    ("highly_variable_genes", {"n_top_genes": 2000, "hvg_flavor": "seurat_v3",
                               "hvg_batch_key": "stim"}),
    ("scaling", {"normalization_method": "pearson_residuals", "theta": 100.0}),
    ("pca", {"n_comps": 50}),
    ("uncorrected_embedding", {"n_neighbors": 15, "umap_min_dist": 0.5}),
    ("batch_correction", {"batch_key": "stim", "batch_method": "harmony"}),
    ("neighbors", {"n_neighbors": 15}),
    ("clustering", {"clustering_method": "louvain", "clustering_resolution": 1.0}),
    ("clustering", {"clustering_method": "leiden", "clustering_resolution": 0.5,
                    "leiden_iterations": 2}),
    ("run_errors", {"status": "finished_with_errors", "failed": {"pca": "boom"},
                    "skipped": {"neighbors": "pca"},
                    "cleared": {"obsm": ["X_pca"], "uns": ["neighbors"], "obs": ["S_score"],
                                "var": ["highly_variable"], "raw": ["raw"]}}),
    ("embeddings", {"umap_min_dist": 0.3}),
    ("marker_genes", {"marker_columns": ["cell_type", "a/b"]}),
    ("enrichment", {"enrichment_columns": ["x"], "enrichment_n_genes": 100}),
    ("cell_cycle", {"species": "human"}),
    ("dissociation_stress", {"genes": ["FOS", "JUN"]}),
    ("dissociation_stress", {}),
    ("cell_type_annotation", {"annotation_method": "celltypist",
                              "celltypist_model": "Immune_All_Low", "annotation_target": "ct"}),
    ("cell_type_annotation", {"annotation_method": "marker_score", "annotation_target": "ms"}),
    ("cell_type_annotation", {"annotation_method": "llm", "annotation_target": "llm_type"}),
    ("subcluster", {"column": "cluster", "group": "3", "resolution": 0.5, "target": "cluster_sub",
                    "batch_key": None}),
    ("keep_resolution", {"clustering_method": "leiden", "clustering_resolution": 0.8,
                         "column": "scview_leiden_r0.8"}),
    ("keep_hvg_setting", {"hvg_flavor": "cell_ranger", "n_top_genes": 1000,
                          "clustering_method": "leiden", "clustering_resolution": 0.5,
                          "n_comps": 30, "n_neighbors": 15, "batch_key": "stim",
                          "column": "hvg_x"}),
    ("pseudotime", {"root_cell": "C1", "column": "scview_dpt_pseudotime"}),
    ("paga", {"groupby": "cluster"}),
    ("annotation_confidence", {"column": "ct", "alpha": 0.1}),
    ("annotation_confidence_unknown", {"column": "ct", "output": "ct_unknown"}),
    ("label_cells", {"column": "mine", "label": "odd", "n_cells": 3, "new_column": True}),
    ("undo_label_cells", {"column": "mine", "n_cells": 3}),
    ("doublet_threshold", {"threshold": 0.3}),
    ("doublet_threshold", {"threshold": "scrublet"}),
    ("annotation_crosscheck", {"columns": ["ct", "cluster"], "reference": "ct",
                               "outputs": ["xc_consensus", "xc_agreement"]}),
    ("merge_clusters", {"column": "cluster", "target": "cluster_merged", "merges": [["1", "2"]],
                        "mapping": {"0": "0", "1": "1+2", "2": "1+2"}, "categories": ["0", "1+2"]}),
]


def test_every_step_type_writes_code_that_compiles():
    a = _with_history(_STEPS + [(name, {}) for name, _ in _STEPS])
    script = replay_script.build_script(a, dataset_id="x", dataset_name="every step")
    compile(script, "replay.py", "exec")
    statuses = [d["status"] for d in replay_script.describe(a)]
    assert len(statuses) == 2 * len(_STEPS)
    assert "unknown" not in statuses[: len(_STEPS)]
    assert set(statuses) <= set(replay_script.STATUSES)
    by_step = dict(zip([s for s, _ in _STEPS], statuses))
    assert by_step["label_cells"] == "not_replayable"
    assert by_step["annotation_crosscheck"] == "not_replayable"
    assert by_step["merge_clusters"] == "exact"
    assert by_step["enrichment"] == "approximate"
    assert by_step["ingest"] == "approximate"  # the antibodies are not moved to obsm
    # helpers are defined once each, before they are used
    assert script.count("def looks_like_counts(") == 1
    assert script.index("def mad_drop(") < script.index("drop = mad_drop(")


def test_unknown_step_gets_a_placeholder():
    a = _with_history([("normalization", {"target_sum": 1e4}),
                       ("future_step", {"k": 1, "nested": {"a": [1, 2]}})])
    script = replay_script.build_script(a)
    compile(script, "replay.py", "exec")
    assert replay_script.describe(a)[1] == {"status": "unknown",
                                            "note": "step type unknown to this version"}
    assert "future_step" in script and "#   k = 1" in script


def test_recorded_values_cannot_inject_code():
    evil = "x\nimport os; os.system('touch /tmp/pwned')\n"
    a = _with_history([
        (evil, {evil: evil}),
        ("clustering", {"clustering_method": "leiden", "clustering_resolution": evil}),
        ("marker_genes", {"marker_columns": [evil, "a'b\"c"]}),
        ("label_cells", {"column": evil, "label": evil}),
        ("annotation_crosscheck", {"columns": [evil], "reference": evil, "outputs": [evil]}),
        ("merge_clusters", {"column": evil, "target": evil, "mapping": {evil: evil}, "categories": [evil]}),
    ])
    provenance.init_source(a, origin=evil, original_filename=evil, fmt=evil)
    script = replay_script.build_script(a, dataset_id=evil, dataset_name=evil)
    compile(script, "replay.py", "exec")
    assert not any(line.lstrip().startswith("import os") for line in script.splitlines())


def test_llm_labels_are_written_in_from_the_record():
    a = _with_history([("cell_type_annotation", {"annotation_method": "llm",
                                                 "annotation_target": "cell_type"})])
    a.uns["cell_type_llm_mapping"] = {"0": "T cells", "1": "B cells"}
    script = replay_script.build_script(a)
    assert replay_script.describe(a)[0]["status"] == "recorded"
    assert "'0': 'T cells'" in script
    # Without the stored answer there is nothing to write in.
    del a.uns["cell_type_llm_mapping"]
    assert replay_script.describe(a)[0]["status"] == "not_replayable"


def test_llm_that_fell_back_to_celltypist_replays_celltypist():
    a = _with_history([("cell_type_annotation", {"annotation_method": "llm",
                                                 "celltypist_model": "Immune_All_Low.pkl",
                                                 "model_species": "human", "gene_overlap": 0.6})])
    script = replay_script.build_script(a)
    assert "celltypist.annotate" in script and "Immune_All_Low.pkl" in script


def test_doublet_threshold_code_matches_the_app():
    from scview.core import doublets

    rng = np.random.default_rng(0)
    a = ad.AnnData(np.zeros((50, 2), dtype="float32"))
    a.obs["doublet_score"] = rng.uniform(0, 0.6, 50)
    a.obs["predicted_doublet"] = a.obs["doublet_score"].to_numpy() > 0.45
    a.uns["scrublet"] = {"threshold": 0.45}
    app, rep = a.copy(), a.copy()
    for t in (0.3, None):  # a threshold, then back to Scrublet's
        doublets.apply_threshold(app, t)
        step = {"threshold": t if t is not None else "scrublet"}
        snippet = replay_script._doublet_threshold(step, replay_script._Ctx(adata=None))
        exec("\n".join(snippet.code), {"adata": rep, "np": np})
        assert (rep.obs["predicted_doublet"].to_numpy(dtype=bool)
                == app.obs["predicted_doublet"].to_numpy(dtype=bool)).all()
        assert set(rep.obs.columns) == set(app.obs.columns)


def test_script_markers_key_matches_the_app():
    from scview.core.uns_keys import markers_key

    ns: dict = {}
    exec(replay_script._HELPERS["markers_key"], ns)
    for col in ("cluster", "Tcm/Naive helper T cells", "a/b/c"):
        assert ns["markers_key"](col) == markers_key(col)


def test_mad_filtering_without_recorded_thresholds_works_them_out():
    a = _with_history([("filtering", {"qc_threshold_mode": "mad", "qc_mad_k": 4.0})])
    script = replay_script.build_script(a)
    assert replay_script.describe(a)[0]["status"] == "approximate"
    assert "mad_thresholds(adata, qc" in script and "k=4.0" in script


def test_mad_thresholds_helper_matches_the_filter():
    """The helper the script uses when thresholds are missing works them out as
    qc_filter does, and mad_drop applies them as the filtering step does."""
    from scview.core import qc_filter
    from scview.core.anndata_adaptor import per_cell_qc

    import scipy.sparse as sp

    a = _counts(n_per=80)
    ns = {"np": np, "pd": pd, "sp": sp}
    for name in ("looks_like_counts", "per_cell_qc", "mad_thresholds", "apply_thresholds"):
        exec(replay_script._HELPERS[name], ns)
    qc = ns["per_cell_qc"](a)
    rec = ns["mad_thresholds"](a, qc, "sample", 5.0, 3.0, True, 20.0)
    drop = ns["mad_drop"](a, qc, rec)
    t = qc_filter.Thresholds(mode="mad", max_pct_mt=20.0, mad_upper=True)
    removed, _ = qc_filter.judge(a, per_cell_qc(a), t, qc_filter.samples_of(a, "sample"))
    assert drop.sum() > 0
    assert (drop == ~qc_filter.keep_mask(removed, a.n_obs)).all()


# ---------------------------------------------------------------------------
# Manifest + endpoints
# ---------------------------------------------------------------------------


def _register(dm, did: str = "rc1", name: str = "Kang (test)") -> Path:
    root = Path(dm.data_dir)
    up = root / "uploads" / did
    up.mkdir(parents=True)
    (up / "metadata.json").write_text(json.dumps({"id": did, "name": name, "status": "ready"}))
    a = _counts()
    upload = up / "kang.h5ad"
    a.write_h5ad(upload)
    provenance.init_source(a, origin="uploaded", original_filename="kang.h5ad", fmt="h5ad")
    out, _ = run_pipeline(a, ["qc_metrics", "normalization", "log_transform"], PipelineParams())
    provenance.record_step(out, step="label_cells", tool="scview.label_selection",
                           params={"column": "mine", "label": "x", "n_cells": 2})
    (root / "derived" / did).mkdir(parents=True)
    out.write_h5ad(root / "derived" / did / "kang.h5ad")
    return upload


def test_input_file_hashes_the_stored_source_not_the_derived_file(tmp_path):
    (tmp_path / "derived" / "d").mkdir(parents=True)
    (tmp_path / "uploads" / "d").mkdir(parents=True)
    derived, upload = tmp_path / "derived" / "d" / "a.h5ad", tmp_path / "uploads" / "d" / "a.h5ad"
    derived.write_bytes(b"processed")
    upload.write_bytes(b"original")
    got = run_manifest.input_file([str(derived), str(upload)])
    assert got["stored_as"] == "uploads"
    assert got["sha256"] == hashlib.sha256(b"original").hexdigest()
    assert run_manifest.input_file([str(derived)]) is None


async def test_recipe_endpoint_says_how_each_step_replays(client, dataset_manager):
    _register(dataset_manager)
    r = await client.get(f"{API}/datasets/rc1/recipe")
    assert r.status_code == 200, r.text
    body = r.json()
    steps = [s["step"] for s in body["recipe"]]
    assert steps == ["qc_metrics", "normalization", "log_transform", "label_cells"]
    first = body["recipe"][0]
    assert first["replay"]["status"] == "exact"
    assert first["timestamp"] and first["scview_version"] and first["tool"]
    assert body["recipe"][-1]["replay"]["status"] == "not_replayable"
    assert body["source"]["original_filename"] == "kang.h5ad"


async def test_script_download(client, dataset_manager):
    upload = _register(dataset_manager)
    r = await client.get(f"{API}/datasets/rc1/recipe/script")
    assert r.status_code == 200, r.text
    assert 'filename="Kang_test_replay.py"' in r.headers["content-disposition"]
    assert r.headers["content-type"].startswith("text/x-python")
    compile(r.text, "replay.py", "exec")
    assert hashlib.sha256(upload.read_bytes()).hexdigest() in r.text
    assert "INPUT = sys.argv[1] if len(sys.argv) > 1 else 'kang.h5ad'" in r.text


async def test_manifest_download(client, dataset_manager):
    upload = _register(dataset_manager)
    r = await client.get(f"{API}/datasets/rc1/recipe/manifest")
    assert r.status_code == 200, r.text
    assert 'filename="Kang_test_RUN_MANIFEST.json"' in r.headers["content-disposition"]
    m = r.json()
    assert m["manifest_version"] == 1 and m["app"]["name"] == "scView"
    assert {"scanpy", "anndata", "numpy"} <= set(m["environment"]["packages"])
    assert m["dataset"] == {**m["dataset"], "id": "rc1", "name": "Kang (test)",
                            "n_genes": 150}
    assert m["input"]["original_filename"] == "kang.h5ad"
    assert m["input"]["stored_file"]["sha256"] == hashlib.sha256(upload.read_bytes()).hexdigest()
    assert [s["step"] for s in m["steps"]][-1] == "label_cells"
    assert m["steps"][0]["replay"]["status"] == "exact"
    assert m["replay_summary"] == {"exact": 3, "not_replayable": 1}


@pytest.mark.parametrize("path", ["recipe/script", "recipe/manifest"])
async def test_unknown_dataset_is_404(client, dataset_manager, path):
    r = await client.get(f"{API}/datasets/missing/{path}")
    assert r.status_code == 404


@pytest.mark.parametrize("processed", [False, True])
def test_doublet_detection_replays_the_apps_scores(tmp_path, processed):
    # The app scores the counts alone (core/scrublet_lean.py); the script copies that code
    # and scores the input the step recorded. A processed X with its counts in a layer was
    # scored on X by the script before (Scrublet on log-normalised values).
    inp = _counts(n_per=100, n_genes=400)  # Scrublet needs over 30 variable genes
    if processed:
        counts = inp.X.copy()
        inp.layers["counts"] = counts
        inp.X = np.log1p(counts / counts.sum(1, keepdims=True).clip(1) * 1e4)
    out, res = run_pipeline(inp.copy(), ["doublet_detection"], PipelineParams())
    assert res.errors == {}
    step = provenance.recipe(out)[-1]["params"]
    assert step.get("input") == ("layers['counts']" if processed else None)
    assert replay_script.describe(out)[-1]["status"] == "exact"
    rep = _replay(tmp_path, inp, out)
    assert np.array_equal(rep.obs["doublet_score"].to_numpy(), out.obs["doublet_score"].to_numpy(),
                          equal_nan=True)
    assert (rep.obs["predicted_doublet"].astype(str) == out.obs["predicted_doublet"].astype(str)).all()
    # The two samples were the channels (no lane column): each one's threshold, as the app's.
    from scview.core import doublets
    assert doublets.channel_record(rep)["rows"] == doublets.channel_record(out)["rows"]
    assert len(doublets.channel_record(out)["rows"]) == 2


def test_umap_3d_setting_reaches_the_step_and_the_replay():
    inp = _counts(seed=3)
    steps = ["qc_metrics", "normalization", "log_transform", "highly_variable_genes", "scaling",
             "pca", "neighbors", "embeddings"]
    params = dict(n_top_genes=60, n_comps=15)
    on, _ = run_pipeline(inp.copy(), steps, PipelineParams(**params))
    assert on.obsm["X_umap_3d"].shape == (on.n_obs, 3)
    assert "n_components=3" in replay_script.build_script(on, dataset_id="t", dataset_name="t")
    # Off: no 3D UMAP, a previous run's taken away, and none in the replay script
    off, res = run_pipeline(on, steps[-1:], PipelineParams(**params, umap_3d="off"))
    assert res.errors == {} and "X_umap_3d" not in off.obsm and "X_umap" in off.obsm
    rec = [h for h in provenance.read_provenance(off)["history"] if h["step"] == "embeddings"]
    assert rec[-1]["params"]["umap_3d"] == "off"
    script = replay_script.build_script(off, dataset_id="t", dataset_name="t")
    assert script.count("n_components=3") == 1  # the first run's step only
    compile(script, "replay.py", "exec")
