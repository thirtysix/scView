"""The pipeline in large-dataset mode (core/large_run.py) against the in-memory pipeline.

Every expected value comes from ``pipeline.run_pipeline`` on the same file read with
``ad.read_h5ad``. The fixture is small but read in several row blocks with a ragged last
one, holds two capture channels, mito genes, the Tirosh cell-cycle genes and the
dissociation-stress genes, so every step has something to do. Exact where the module
says so (QC, doublets, filtering, X, variable genes, markers given the labels, scores);
scaling statistics and PCA to float precision.

Real data: SCVIEW_TESTDATA with kang2018_ifnb_pbmc.h5ad / paul15.h5ad runs the same
comparison there (``test_real_data_equals_in_memory``).
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from scview.core import (
    chunked,
    gene_store,
    large_layout,
    large_run,
    large_scores,
    memory_budget,
    preflight,
    provenance,
    replay_script,
    run_manifest,
    working_copy,
)
from scview.core.anndata_adaptor import AnnDataAdaptor
from scview.core.pipeline import (
    CELL_CYCLE_G2M_GENES,
    CELL_CYCLE_S_GENES,
    PipelineParams,
    check_chunked,
    run_pipeline,
)
from scview.core.run_watchdog import RunControl
from scview.core.stress import STRESS_GENES
from scview.core.uns_keys import markers_key

API = "/api/v1"
GB = 1024**3
ROWS = 700
STEPS = ["reset_to_counts", "qc_metrics", "doublet_detection", "filtering", "normalization",
         "log_transform", "highly_variable_genes", "scaling", "pca", "neighbors", "clustering",
         "embeddings", "marker_genes", "cell_cycle", "dissociation_stress"]
PARAMS = dict(doublet_channel_key="lane", n_top_genes=300, n_comps=20, min_cells=3,
              marker_columns=["group"])
QC_OBS = ["n_genes_by_counts", "total_counts", "total_counts_mt", "pct_counts_mt"]


def _counts(seed: int = 0, n: int = 2600) -> ad.AnnData:
    """NB counts: four groups with their own genes, cycling cells, stressed cells,
    mito-high cells, two lanes, genes never seen and a few rare ones."""
    rng = np.random.default_rng(seed)
    cc = list(CELL_CYCLE_S_GENES[:20]) + list(CELL_CYCLE_G2M_GENES[:20])
    names = ([f"MT-{i}" for i in range(6)] + cc + list(STRESS_GENES)
             + [f"G{i}" for i in range(520)])
    g = len(names)
    group = rng.integers(0, 4, n)
    size = rng.lognormal(0.0, 0.5, n)
    mu = size[:, None] * rng.gamma(0.6, 1.0, g)[None, :] * 2.0
    for k in range(4):
        mu[group == k, 100 + 40 * k: 140 + 40 * k] *= 6.0
    cycling = rng.random(n) < 0.2
    mu[cycling, 6:26] *= 5.0
    mu[rng.random(n) < 0.15, 46:68] *= 6.0
    mu[rng.random(n) < 0.08, :6] *= 10.0
    X = rng.negative_binomial(2, 2.0 / (2.0 + mu)).astype(np.float32)
    X[:, -3:] = 0
    X[:, -8:-3] = 0
    for j in range(g - 8, g - 3):
        X[rng.choice(n, 2, replace=False), j] = 3
    obs = pd.DataFrame({"lane": pd.Categorical(np.where(rng.random(n) < 0.55, "L1", "L2")),
                        "group": pd.Categorical([f"t{k}" for k in group])},
                       index=[f"cell{i}-1" for i in range(n)])
    return ad.AnnData(sp.csr_matrix(X), obs=obs, var=pd.DataFrame(index=names))


@pytest.fixture(autouse=True)
def _small_blocks(monkeypatch):
    """Several row blocks of 700 with a ragged last one."""
    monkeypatch.setattr(chunked, "rows_per_block", lambda *a, **k: ROWS)


def _write(a: ad.AnnData, path: Path, **kw) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    a.write_h5ad(path, compression="gzip", **kw)
    return path


def _both(src: Path, out: Path, steps=STEPS, **params):
    p = {**PARAMS, **params}
    res = large_run.run(src, steps, PipelineParams(**p, chunked="on"), str(out))
    mem, mres = run_pipeline(ad.read_h5ad(src), steps, PipelineParams(**p))
    return ad.read_h5ad(out), res, mem, mres


@pytest.fixture(scope="module")
def processed(tmp_path_factory):
    """(large result, its result, in-memory result, its result, the folder)."""
    import pytest as _pytest

    mp = _pytest.MonkeyPatch()
    mp.setattr(chunked, "rows_per_block", lambda *a, **k: ROWS)
    d = tmp_path_factory.mktemp("large")
    src = _write(_counts(), d / "in" / "data.h5ad")
    try:
        big, res, mem, mres = _both(src, d / "derived" / "data.h5ad")
    finally:
        mp.undo()
    return big, res, mem, mres, d


def _dense(M):
    return M.toarray() if sp.issparse(M) else np.asarray(M)


# ---------------------------------------------------------------------------
# The run against the in-memory pipeline
# ---------------------------------------------------------------------------


def test_every_step_runs_as_in_memory(processed):
    big, res, mem, mres, _ = processed
    assert res.status == "complete" and res.chunked
    assert res.steps_run == mres.steps_run == STEPS
    assert res.errors == mres.errors == {} and res.notes == mres.notes


def test_cells_genes_qc_and_doublets_are_bitwise(processed):
    big, _, mem, _, _ = processed
    assert big.obs_names.equals(mem.obs_names) and big.var_names.equals(mem.var_names)
    assert big.n_obs < 2600 and big.n_vars < 592  # filtering removed cells and genes
    for c in QC_OBS + ["doublet_score", "predicted_doublet"]:
        np.testing.assert_array_equal(big.obs[c].to_numpy(), mem.obs[c].to_numpy())
        assert big.obs[c].dtype == mem.obs[c].dtype, c
    for c in ["mt", "n_cells_by_counts", "mean_counts", "total_counts", "n_cells"]:
        np.testing.assert_array_equal(big.var[c].to_numpy(), mem.var[c].to_numpy())
    assert list(np.asarray(big.uns["scrublet"]["scview_channels"]["name"])) == \
        list(np.asarray(mem.uns["scrublet"]["scview_channels"]["name"]))


def test_x_is_the_log_normalised_matrix_bitwise(processed):
    big, _, mem, _, _ = processed
    assert sp.issparse(big.X) and big.X.dtype == np.float32
    np.testing.assert_array_equal(_dense(big.X), _dense(mem.X))
    assert big.raw is None and not big.layers and big.uns["log1p"] == mem.uns["log1p"]


def test_variable_genes_scaling_and_pca(processed):
    big, _, mem, _, _ = processed
    for c in ("highly_variable", "means", "dispersions", "dispersions_norm"):
        np.testing.assert_array_equal(big.var[c].to_numpy(), mem.var[c].to_numpy())
    assert big.uns["scview_scaling"] == mem.uns["scview_scaling"]
    np.testing.assert_allclose(big.var["mean"], mem.var["mean"], rtol=1e-12, equal_nan=True)
    np.testing.assert_allclose(big.var["std"], mem.var["std"], rtol=1e-12, equal_nan=True)
    xb, xm = big.obsm["X_pca"], mem.obsm["X_pca"]
    assert xb.dtype == xm.dtype == np.float32 and xb.shape == xm.shape
    scale = np.abs(xm).max(axis=0)
    # scanpy's float32 ARPACK against an exact covariance PCA: float32 precision
    assert np.max(np.abs(xb - xm) / scale) < 5e-3
    assert big.varm["PCs"].shape == mem.varm["PCs"].shape
    assert big.uns["pca"]["params"] == mem.uns["pca"]["params"]
    np.testing.assert_allclose(big.uns["pca"]["variance_ratio"], mem.uns["pca"]["variance_ratio"],
                               rtol=1e-3)


def test_graph_clusters_and_umap_are_the_pipeline_s(processed):
    from sklearn.metrics import adjusted_rand_score

    big, _, mem, _, _ = processed
    col = mem.uns["scview_active_clustering"]
    assert big.uns["scview_active_clustering"] == col
    assert adjusted_rand_score(big.obs[col], mem.obs[col]) > 0.9
    for k in ("X_umap", "X_umap_3d"):
        assert big.obsm[k].shape == mem.obsm[k].shape
    assert big.obsp["connectivities"].shape == mem.obsp["connectivities"].shape


def test_markers_given_the_same_labels_are_scanpy_s(processed):
    big, _, mem, _, _ = processed
    key = markers_key("group")  # a fixed column: the same labels in both
    rb, rm = big.uns[key], mem.uns[key]
    for f in ("names", "scores", "pvals", "pvals_adj"):
        for g in rm[f].dtype.names:
            np.testing.assert_array_equal(np.asarray(rb[f][g]), np.asarray(rm[f][g]))
    for g in rm["logfoldchanges"].dtype.names:
        np.testing.assert_allclose(rb["logfoldchanges"][g], rm["logfoldchanges"][g], rtol=1e-5)
    pd.testing.assert_frame_equal(rb["pts"], rm["pts"])
    np.testing.assert_allclose(rb["scview_specificity"]["tau"], rm["scview_specificity"]["tau"],
                               rtol=1e-5)
    assert rb["params"]["use_raw"] is False and rm["params"]["use_raw"] is True  # no .raw


def test_cell_cycle_and_stress_scores_are_bitwise(processed):
    big, _, mem, _, _ = processed
    for c in ("S_score", "G2M_score", "dissociation_stress_score"):
        np.testing.assert_array_equal(big.obs[c].to_numpy(), mem.obs[c].to_numpy())
    assert (big.obs["phase"].astype(str) == mem.obs["phase"].astype(str)).all()


def test_provenance_records_the_same_steps_marked_chunked(processed):
    big, _, mem, _, _ = processed
    hb = provenance.read_provenance(big)["history"]
    hm = provenance.read_provenance(mem)["history"]
    assert [h["step"] for h in hb] == [h["step"] for h in hm]
    assert [h["tool"] for h in hb] == [h["tool"] for h in hm]
    for b, m in zip(hb, hm):
        assert b["params"].pop("chunked") is True
        assert b["params"] == m["params"], b["step"]
    cur = provenance.read_provenance(big)["current"]
    assert cur["large"] is True and cur["anchors"] == ["counts", "lognorm"]


def test_the_result_has_the_large_layout(processed):
    big, res, _, _, d = processed
    out = Path(res.output_path)
    assert large_layout.is_large(big)
    counts, genes = large_layout.counts_path(out, big.uns), large_layout.genes_path(out, big.uns)
    assert counts.exists() and genes.exists()
    assert large_layout.counts_mismatch(counts, big.obs_names, big.var_names) is None
    with h5py.File(counts, "r") as f:
        c = ad.io.read_elem(f["counts"])
    want = ad.read_h5ad(d / "in" / "data.h5ad")[big.obs_names, big.var_names].X
    np.testing.assert_array_equal(_dense(c), _dense(want))
    with gene_store.GeneStore(genes) as st:
        assert st.mismatch(big, source="X", full=True) is None
    # nothing staged is left beside it
    assert sorted(p.name for p in out.parent.iterdir()) == ["data.counts.h5", "data.genes.h5",
                                                            "data.h5ad"]
    assert ad.read_h5ad(out, backed="r").X.shape == big.shape


def test_replay_script_and_manifest_say_the_run_was_chunked(processed):
    big, *_ = processed
    script = replay_script.build_script(big, dataset_id="d", dataset_name="d")
    assert "large-dataset mode" in script
    compile(script, "replay.py", "exec")
    m = run_manifest.build_manifest(big, dataset_id="d", dataset_name="d")
    assert m["dataset"]["large"] is True
    assert all(s["params"].get("chunked") for s in m["steps"])


# ---------------------------------------------------------------------------
# Inputs: counts in .raw or a layer, a large dataset run again
# ---------------------------------------------------------------------------


def test_batch_correction_and_the_uncorrected_layout_run_on_the_light_copy(tmp_path):
    src = _write(_counts(n=1600), tmp_path / "in" / "d.h5ad")
    steps = STEPS[:9] + ["uncorrected_embedding", "batch_correction", "neighbors"]
    big, res, mem, mres = _both(src, tmp_path / "out" / "d.h5ad", steps, batch_key="lane")
    assert res.errors == mres.errors == {} and res.steps_run == steps
    for k in ("X_umap_uncorrected", "X_pca_harmony"):
        assert big.obsm[k].shape == mem.obsm[k].shape
    # Harmony iterates on the PCs, so their float-precision difference grows a little.
    hb, hm = big.obsm["X_pca_harmony"], mem.obsm["X_pca_harmony"]
    assert min(np.corrcoef(hb[:, i], hm[:, i])[0, 1] for i in range(hb.shape[1])) > 0.999
    assert big.uns["neighbors"]["params"]["use_rep"] == "X_pca_harmony"


@pytest.mark.parametrize("where", ["raw", "layer"])
def test_counts_in_raw_or_a_layer(tmp_path, where):
    a = _counts(n=1600)
    counts = a.X.copy()
    lognorm = a.copy()
    import scanpy as sc

    sc.pp.normalize_total(lognorm)
    sc.pp.log1p(lognorm)
    a.X = lognorm.X
    if where == "raw":
        a.raw = ad.AnnData(counts, var=pd.DataFrame(index=a.var_names))
    else:
        a.layers["counts"] = counts
    src = _write(a, tmp_path / "in" / "d.h5ad")
    steps = STEPS[:9] + ["marker_genes"]
    big, res, mem, mres = _both(src, tmp_path / "out" / "d.h5ad", steps)
    assert res.errors == mres.errors == {}
    np.testing.assert_array_equal(_dense(big.X), _dense(mem.X))
    np.testing.assert_array_equal(big.obs["doublet_score"], mem.obs["doublet_score"])
    assert (big.var["highly_variable"] == mem.var["highly_variable"]).all()


def _copy_large(res, tmp_path: Path, folder: str = "derived") -> Path:
    out = tmp_path / folder / "data.h5ad"
    out.parent.mkdir(parents=True, exist_ok=True)
    for p in Path(res.output_path).parent.iterdir():
        shutil.copy2(p, out.parent / p.name)
    return out


def test_rows_stored_out_of_order_give_the_in_memory_scores(tmp_path):
    # 10x's own subsets store each cell's genes out of order; the in-memory scaling step
    # sorts X in place before the scores read it, so X is written sorted too.
    a = _counts(n=1600, seed=3)
    rng = np.random.default_rng(0)
    X = a.X.tocsr()
    for i in range(X.shape[0]):
        a0, b0 = X.indptr[i], X.indptr[i + 1]
        p = rng.permutation(b0 - a0)
        X.indices[a0:b0] = X.indices[a0:b0][p]
        X.data[a0:b0] = X.data[a0:b0][p]
    X.has_sorted_indices = False
    a.X = X
    src = _write(a, tmp_path / "in" / "u.h5ad")
    big, res, mem, mres = _both(src, tmp_path / "out" / "u.h5ad")
    assert res.errors == mres.errors == {}
    assert big.X.has_canonical_format
    for c in ("S_score", "G2M_score", "dissociation_stress_score"):
        np.testing.assert_array_equal(big.obs[c].to_numpy(), mem.obs[c].to_numpy())


def test_a_large_dataset_is_run_again_from_its_files(processed, tmp_path):
    big, res, _, _, _ = processed
    out = _copy_large(res, tmp_path)
    counts, genes = large_layout.sidecar_paths(out)
    before = {p: (p.stat().st_ino, p.stat().st_mtime_ns) for p in (counts, genes)}
    # Steps that change no matrix: X is copied, the sidecars stay as they are.
    steps = ["clustering", "marker_genes", "dissociation_stress"]
    r = large_run.run(out, steps, PipelineParams(**PARAMS, clustering_resolution=1.0,
                                                 chunked="on"), str(out))
    assert r.status == "complete" and r.errors == {}, r.errors
    again = ad.read_h5ad(out)
    assert {p: (p.stat().st_ino, p.stat().st_mtime_ns) for p in (counts, genes)} == before
    np.testing.assert_array_equal(_dense(again.X), _dense(big.X))
    assert "scview_leiden_r1.0" in again.obs and markers_key("scview_leiden_r1.0") in again.uns
    np.testing.assert_array_equal(again.obs["dissociation_stress_score"],
                                  big.obs["dissociation_stress_score"])
    # Reprocessed from scratch: the counts come from the sidecar, as in memory from the
    # same dataset with its counts as the layer (the recorded filtering thresholds too).
    mem = large_run.read_light(out)
    with h5py.File(counts, "r") as f:
        c = ad.io.read_elem(f["counts"])
    mem = ad.AnnData(c.copy(), obs=mem.obs, var=mem.var, obsm=dict(mem.obsm),
                     obsp=dict(mem.obsp), uns=mem.uns, layers={"counts": c})
    mem, _ = run_pipeline(mem, STEPS, PipelineParams(**PARAMS))
    r = large_run.run(out, STEPS, PipelineParams(**PARAMS, chunked="on"), str(out))
    assert r.status == "complete" and r.errors == {}, r.errors
    again = ad.read_h5ad(out)
    assert again.obs_names.equals(mem.obs_names) and again.var_names.equals(mem.var_names)
    np.testing.assert_array_equal(_dense(again.X), _dense(mem.X))
    for c in QC_OBS + ["doublet_score", "S_score", "dissociation_stress_score"]:
        np.testing.assert_array_equal(again.obs[c], mem.obs[c])
        assert again.obs[c].dtype == mem.obs[c].dtype


def test_a_large_dataset_always_runs_in_large_mode(processed, tmp_path, monkeypatch):
    _, res, _, _, _ = processed
    out = _copy_large(res, tmp_path)
    adaptor = AnnDataAdaptor(str(out))
    _budget(monkeypatch, 64 * GB, 1 * GB)  # it would fit in memory
    for mode in ("auto", "off"):
        choice = large_run.choose_mode(_DM(), "d", adaptor, STEPS, PipelineParams(chunked=mode))
        assert choice.chunked and "never read into memory" in choice.reason


def test_the_h5ad_keeps_its_free_space_for_edits_in_place(processed):
    _, res, _, _, _ = processed
    with h5py.File(res.output_path, "r") as f:
        strategy, persist, _ = f.id.get_create_plist().get_file_space_strategy()
    assert persist and strategy == h5py.h5f.FSPACE_STRATEGY_FSM_AGGR


def test_the_header_says_a_large_dataset_opens_backed(processed):
    _, res, _, _, d = processed
    h = preflight.read_header(res.output_path)
    assert h.large and h.backed
    plain = preflight.read_header(d / "in" / "data.h5ad")
    assert not plain.large and not plain.backed


# ---------------------------------------------------------------------------
# Nothing saved unless the run reaches its end
# ---------------------------------------------------------------------------


def _snapshot(folder: Path) -> dict:
    return {p.name: p.read_bytes() for p in folder.iterdir() if p.is_file()}


@pytest.fixture()
def large_copy(processed, tmp_path) -> Path:
    _, res, _, _, _ = processed
    return _copy_large(res, tmp_path)


def test_a_cancelled_run_leaves_the_dataset_as_it_was(large_copy, monkeypatch):
    before = _snapshot(large_copy.parent)
    control = RunControl(before_step=None)
    seen = []
    tick = large_run.LargeRun.tick

    def cancel_in_a_pass(self, done, total):
        seen.append((self._label, done))
        if self._label == "Writing the matrices" and done > 0:
            control.cancel()
        return tick(self, done, total)

    monkeypatch.setattr(large_run.LargeRun, "tick", cancel_in_a_pass)
    r = large_run.run(large_copy, STEPS, PipelineParams(**PARAMS, chunked="on"), str(large_copy),
                      control)
    assert r.status == "cancelled" and "highly variable genes" in r.stop_reason
    assert r.output_path is None and _snapshot(large_copy.parent) == before


def test_a_failed_save_leaves_the_dataset_as_it_was(large_copy, monkeypatch):
    before = _snapshot(large_copy.parent)

    def full(src, dst):
        raise OSError("No space left on device")

    monkeypatch.setattr(large_run.os, "replace", full)  # the first move into place fails
    with pytest.raises(OSError, match="No space"):
        large_run.run(large_copy, STEPS, PipelineParams(**PARAMS, chunked="on"), str(large_copy))
    assert _snapshot(large_copy.parent) == before  # the staging folder is gone too


def test_not_enough_disk_stops_before_any_step(large_copy, monkeypatch):
    before = _snapshot(large_copy.parent)
    monkeypatch.setattr(large_run.shutil, "disk_usage",
                        lambda p: shutil._ntuple_diskusage(10**12, 10**12 - 1000, 1000))
    r = large_run.run(large_copy, STEPS, PipelineParams(**PARAMS, chunked="on"), str(large_copy))
    assert r.status == "stopped" and "Not enough disk" in r.stop_reason and r.steps_run == []
    assert _snapshot(large_copy.parent) == before


def test_the_pre_step_check_uses_this_mode_s_estimate(large_copy, monkeypatch):
    monkeypatch.setattr(preflight, "available_memory", lambda *a, **k: preflight.Available(
        8 * GB, 8 * GB - 1, 1, True, "container limit"))
    r = large_run.run(large_copy, ["qc_metrics"], PipelineParams(chunked="on"), str(large_copy),
                      RunControl())
    assert r.status == "stopped" and "in large-dataset mode it needs" in r.stop_reason


def test_counts_left_in_x_are_refused_before_any_step(tmp_path):
    src = _write(_counts(n=900), tmp_path / "in" / "d.h5ad")
    r = large_run.run(src, ["qc_metrics", "highly_variable_genes"], PipelineParams(chunked="on"),
                      str(tmp_path / "out" / "d.h5ad"))
    assert r.status == "stopped" and "normalization and log transform" in r.stop_reason
    assert not (tmp_path / "out" / "d.h5ad").exists()


def test_streamed_events_include_block_progress(tmp_path):
    src = _write(_counts(n=1600), tmp_path / "in" / "d.h5ad")
    events = list(large_run.run_streamed(src, STEPS[:9], PipelineParams(**PARAMS, chunked="on"),
                                         str(tmp_path / "out" / "d.h5ad")))
    kinds = [e for e, _ in events]
    assert kinds[-1] == "complete" and kinds.count("step_start") == 9
    subs = [d for e, d in events if e == "substep"]
    assert any(d["step"] == "highly_variable_genes" and "Writing" in d["message"] for d in subs)
    assert events[-1][1]["chunked"] is True


# ---------------------------------------------------------------------------
# Scores and CellTypist from row blocks
# ---------------------------------------------------------------------------


def test_scores_equal_scanpy_on_dense_values(tmp_path):
    import scanpy as sc

    a = _counts(n=1500)
    a.X = a.X.toarray()
    sc.pp.normalize_total(a)
    sc.pp.log1p(a)
    path = _write(ad.AnnData(sp.csr_matrix(a.X), obs=a.obs, var=a.var), tmp_path / "x.h5ad")
    store = gene_store.GeneStore(gene_store.build_from_adata(ad.read_h5ad(path), tmp_path / "g.h5"))
    X = chunked.read_lazy(path, "X", chunk_rows=ROWS)
    genes = list(STRESS_GENES)
    for dense in (False, True):
        ref = a.copy() if dense else ad.AnnData(sp.csr_matrix(a.X), obs=a.obs.copy(), var=a.var)
        sc.tl.score_genes(ref, genes, score_name="s", ctrl_size=50, random_state=0)
        got = ad.AnnData(obs=a.obs.copy(), var=a.var)
        means = large_scores.gene_means(store, dense=dense)
        large_scores.score_genes(got, X, means, genes, score_name="s", ctrl_size=50, dense=dense)
        np.testing.assert_array_equal(got.obs["s"].to_numpy(), ref.obs["s"].to_numpy())


def _celltypist_model(var_names, seed=0):
    from celltypist.models import Model
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(seed)
    feats = np.array(list(var_names[::2]) + ["NOT_IN_DATA"])
    Xtr = rng.normal(size=(300, feats.size))
    y = np.array(["A", "B", "C"])[rng.integers(0, 3, 300)]
    scaler = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=200).fit(scaler.transform(Xtr), y)
    clf.features = feats
    return Model(clf, scaler, {"date": "", "details": "test", "url": "", "source": "",
                               "version": "", "number_celltypes": 3})


def test_celltypist_by_blocks_equals_celltypist(tmp_path):
    import celltypist
    import scanpy as sc

    a = _counts(n=1800)
    model = _celltypist_model(a.var_names)
    path = _write(a, tmp_path / "c.h5ad")
    a.obs["clu"] = pd.Categorical(np.asarray(a.obs["group"]))
    inp = a.copy()
    sc.pp.normalize_total(inp, target_sum=1e4)
    sc.pp.log1p(inp)
    want = celltypist.annotate(inp, model=model, majority_voting=True, over_clustering="clu")
    got = large_scores.celltypist_annotate(model, chunked.read_lazy(path, "X", chunk_rows=ROWS),
                                           a.obs, a.var_names, "clu")
    for c in ("predicted_labels", "majority_voting"):
        assert (got.predicted_labels[c].astype(str) == want.predicted_labels[c].astype(str)).all()
    np.testing.assert_array_equal(got.probability_matrix.to_numpy(),
                                  want.probability_matrix.to_numpy(dtype=np.float32))
    with pytest.raises(ValueError, match="clustering"):
        large_scores.celltypist_annotate(model, chunked.read_lazy(path, "X", chunk_rows=ROWS),
                                         a.obs, a.var_names, "")


# ---------------------------------------------------------------------------
# Choosing the mode, and the run endpoints
# ---------------------------------------------------------------------------


def test_chunked_values():
    assert [check_chunked(v) for v in ("auto", "ON", " off ", True, False, None)] == \
        ["auto", "on", "off", "on", "off", "auto"]
    with pytest.raises(ValueError, match="auto, on, off"):
        check_chunked("maybe")


class _DM:
    def cached_ids(self):
        return []


def _budget(monkeypatch, limit, used):
    monkeypatch.setattr(memory_budget, "read_budget", lambda *a, **k: memory_budget.MemoryBudget(
        limit, used, True, "container limit"))


def test_auto_picks_large_mode_when_the_run_would_not_fit(tmp_path, monkeypatch):
    src = _write(_counts(n=900), tmp_path / "d.h5ad")
    adaptor = AnnDataAdaptor(str(src))
    _budget(monkeypatch, 6 * GB, 1 * GB)
    fits = large_run.choose_mode(_DM(), "d", adaptor, STEPS, PipelineParams())
    assert not fits.chunked and "works in memory" in fits.reason
    monkeypatch.setattr(large_run, "in_memory_need", lambda *a: (5 * GB, 4 * GB))
    big = large_run.choose_mode(_DM(), "d", adaptor, STEPS, PipelineParams())
    assert big.chunked and "large-dataset mode" in big.reason and big.need_bytes == 5 * GB
    assert not large_run.choose_mode(_DM(), "d", adaptor, STEPS, PipelineParams(chunked="off")).chunked
    assert large_run.choose_mode(_DM(), "d", adaptor, STEPS, PipelineParams(chunked="on")).chunked


def test_a_backed_dataset_whose_read_in_would_not_fit_goes_to_large_mode(tmp_path, monkeypatch):
    from scview.core import anndata_adaptor

    monkeypatch.setattr(anndata_adaptor, "BACKED_MIN_BYTES", 0)
    src = _write(_counts(n=900), tmp_path / "d.h5ad")
    adaptor = AnnDataAdaptor(str(src))
    assert adaptor.adata.isbacked
    _budget(monkeypatch, 6 * GB, 1 * GB)
    monkeypatch.setattr(preflight, "backed_bytes", lambda h: 50 * GB)
    choice = large_run.choose_mode(_DM(), "d", adaptor, STEPS, PipelineParams())
    assert choice.chunked and choice.need_bytes >= 50 * GB


def _register(data_dir: Path, did: str = "big") -> str:
    d = data_dir / "uploads" / did
    d.mkdir(parents=True, exist_ok=True)
    (d / "metadata.json").write_text(json.dumps({"id": did, "name": did, "status": "ready"}))
    _counts(n=1600).write_h5ad(d / "data.h5ad", compression="gzip")
    return did


@pytest.mark.parametrize("path", ["assessment/run", "assessment/run-stream"])
async def test_auto_runs_in_large_mode_through_the_endpoints_and_reopens(
        client, dataset_manager, monkeypatch, path):
    did = _register(Path(dataset_manager.data_dir))
    monkeypatch.setattr(large_run, "in_memory_need", lambda *a: (10**15, 10**15))
    reads = []
    monkeypatch.setattr(working_copy, "in_memory", lambda a: reads.append(a) or a.to_memory())
    steps = STEPS[:9] + ["neighbors", "clustering", "marker_genes"]
    r = await client.post(f"{API}/datasets/{did}/{path}",
                          json={"steps": steps, "params": {**PARAMS, "marker_columns": []}})
    assert r.status_code == 200, r.text
    if path.endswith("stream"):
        events = [line for line in r.text.splitlines() if line.startswith("data: ")]
        result = json.loads(events[-1][len("data: "):])
    else:
        result = r.json()
    assert result["errors"] == {} and result["steps_run"] == steps
    assert result["chunked"] is True and result["mode"]["chunked"] is True
    assert reads == []  # never read into memory
    adaptor = await dataset_manager.get_or_load_dataset(did)
    assert large_layout.is_large(adaptor.adata) and "X_pca" in adaptor.adata.obsm
    assert adaptor.adata.X.dtype == np.float32


async def test_the_cached_copy_is_let_go_before_a_large_run(client, dataset_manager, monkeypatch):
    import gc
    import weakref

    did = _register(Path(dataset_manager.data_dir))
    old = weakref.ref((await dataset_manager.get_or_load_dataset(did)).adata)
    monkeypatch.setattr(large_run, "in_memory_need", lambda *a: (10**15, 10**15))
    alive = []
    real = large_run.run

    def spy(*a, **k):
        gc.collect()
        alive.append(old() is not None)
        return real(*a, **k)

    monkeypatch.setattr(large_run, "run", spy)
    r = await client.post(f"{API}/datasets/{did}/assessment/run",
                          json={"steps": ["qc_metrics"], "params": {"chunked": "auto"}})
    assert r.status_code == 200, r.text
    assert alive == [False]


def _register_large(processed, data_dir: Path, did: str = "lg") -> Path:
    _, res, _, _, _ = processed
    up = data_dir / "uploads" / did
    up.mkdir(parents=True, exist_ok=True)
    (up / "metadata.json").write_text(json.dumps({"id": did, "name": did, "status": "ready"}))
    out = data_dir / "derived" / did / "data.h5ad"
    out.parent.mkdir(parents=True, exist_ok=True)
    for p in Path(res.output_path).parent.iterdir():
        shutil.copy2(p, out.parent / p.name)
    return out


async def test_paga_and_pseudotime_read_a_large_dataset_without_its_matrices(
        client, dataset_manager, processed, monkeypatch):
    out = _register_large(processed, Path(dataset_manager.data_dir))
    counts, genes = large_layout.sidecar_paths(out)
    sidecars = {p: p.read_bytes() for p in (counts, genes)}
    reads = []
    monkeypatch.setattr(working_copy, "in_memory", lambda a: reads.append(a) or a.to_memory())
    col = "scview_leiden_r0.5"
    r = await client.post(f"{API}/datasets/lg/trajectory/paga", json={"groupby": col})
    assert r.status_code == 200, r.text
    r = await client.post(f"{API}/datasets/lg/trajectory/dpt",
                          json={"root_column": "group", "root_value": "t0"})
    assert r.status_code == 200, r.text
    assert reads == []
    saved = ad.read_h5ad(out)
    assert "scview_dpt_pseudotime" in saved.obs and large_layout.is_large(saved)
    np.testing.assert_array_equal(_dense(saved.X), _dense(processed[0].X))
    assert {p: p.read_bytes() for p in (counts, genes)} == sidecars
    steps = [h["step"] for h in provenance.read_provenance(saved)["history"]]
    assert steps[-2:] == ["paga", "pseudotime"]


async def test_work_that_reads_a_large_dataset_into_memory_is_refused(dataset_manager, processed):
    from fastapi import HTTPException

    from scview.core import processing

    _register_large(processed, Path(dataset_manager.data_dir))
    with pytest.raises(HTTPException) as e:
        await processing.run_exclusive(dataset_manager, "lg", lambda a: a.adata)
    assert e.value.status_code == 422 and "large-dataset layout" in e.value.detail
    # and the dataset is free again
    assert not dataset_manager.is_busy("lg")


async def test_an_unknown_mode_is_refused_before_loading(client, dataset_manager):
    did = _register(Path(dataset_manager.data_dir))
    r = await client.post(f"{API}/datasets/{did}/assessment/run",
                          json={"steps": ["qc_metrics"], "params": {"chunked": "maybe"}})
    assert r.status_code == 422 and "auto, on, off" in r.json()["detail"]
    r = await client.get(f"{API}/datasets/{did}/preflight", params={"steps": "qc_metrics",
                                                                      "chunked": "maybe"})
    assert r.status_code == 422


async def test_preflight_gives_this_mode_s_estimate_and_disk(client, dataset_manager):
    did = _register(Path(dataset_manager.data_dir))
    r = await client.get(f"{API}/datasets/{did}/preflight",
                         params={"steps": ",".join(STEPS), "chunked": "on"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["mode"]["chunked"] is True and body["disk"]["need_bytes"] > 0
    assert [s["step"] for s in body["steps"]] == STEPS + ["save"]
    assert "large-dataset mode" in body["verdict"]["sentence"]
    assert body["in_memory_estimate"]["steps"]
    r = await client.get(f"{API}/datasets/{did}/preflight", params={"steps": ",".join(STEPS)})
    assert r.json()["mode"]["chunked"] is False and "disk" not in r.json()


def test_chunked_estimate_grows_with_blocks_not_cells():
    small = preflight.ChunkedShape(100_000, 28_000, 200_000_000, other=200 * 2**20)
    big = preflight.ChunkedShape(1_300_000, 28_000, 2_600_000_000, other=2 * GB)
    es = preflight.estimate_chunked_run(small, STEPS, PipelineParams())
    eb = preflight.estimate_chunked_run(big, STEPS, PipelineParams())
    # a pass holds a batch of blocks whatever the cells: only what the steps keep grows
    qs = next(e for e in es.steps if e.step == "qc_metrics").transient_bytes
    qb = next(e for e in eb.steps if e.step == "qc_metrics").transient_bytes
    assert qs == qb
    wells = preflight.DoubletPlan([10_000] * 130)  # its 133 GEM wells
    eb = preflight.estimate_chunked_run(big, STEPS, PipelineParams(), doublets=wells)
    h = preflight.Header("h5ad", big.n, big.m, [preflight.Matrix("X", big.n, big.m, True, big.nnz)])
    mem = preflight.estimate_run(h, steps=STEPS, params=PipelineParams(), doublets=wells)
    assert eb.peak_bytes < mem.peak_bytes / 2
    assert preflight.chunked_disk_bytes(big) > 40 * GB


def test_chunked_estimate_matches_the_live_100k_run(monkeypatch):
    """The live run of the UI's steps on the 100k-cell brain (6 GB, 2 CPUs; see preflight's
    calibration): peak 2.19 GB in Leiden, 551 s, 1.65 GB of files."""
    monkeypatch.setattr(chunked, "n_workers", lambda: 2)
    n, m, nnz = 100_000, 27_998, 201_003_839
    h = preflight.Header("h5ad", n, m, [preflight.Matrix("X", n, m, True, nnz)], obs_columns=1,
                         var_columns=1)
    steps = [s for s in STEPS if s != "marker_genes"] + ["marker_genes"]
    est = preflight.estimate_chunked_run(h, steps, PipelineParams(),
                                         doublets=preflight.DoubletPlan([752] * 133))
    assert est.peak_step == "clustering"
    assert abs(est.peak_bytes / (2.19 * GB) - 1) < preflight.ERROR_BAND
    assert est.seconds[0] <= 551 <= est.seconds[1]
    disk = preflight.chunked_disk_bytes(preflight.ChunkedShape.from_header(h))
    assert 1.65 * GB < disk < 4 * GB  # the files, and the gene store's spill while it builds


# ---------------------------------------------------------------------------
# Real data (opt-in)
# ---------------------------------------------------------------------------

TESTDATA = Path(os.environ.get("SCVIEW_TESTDATA", "/nonexistent"))


@pytest.mark.parametrize("name", ["kang2018_ifnb_pbmc.h5ad", "paul15.h5ad"])
def test_real_data_equals_in_memory(name, tmp_path, monkeypatch):
    src = TESTDATA / name
    if not src.exists():
        pytest.skip(f"{name} not in SCVIEW_TESTDATA")
    monkeypatch.setattr(chunked, "rows_per_block", lambda *a, **k: 1500)
    path = tmp_path / "in" / name
    path.parent.mkdir()
    a = ad.read_h5ad(src)
    # Counts as float32 (Kang keeps float64 counts in .raw): in memory X is then float32
    # too, as large-dataset mode stores it, so every value can be compared bit for bit.
    a.X = a.X.astype(np.float32)
    if a.raw is not None:
        a.raw = ad.AnnData(a.raw.X.astype(np.float32), var=a.raw.var, obs=a.obs[[]])
    a.write_h5ad(path, compression="gzip")
    marker_cols = ["cluster"] if "kang" in name else ["paul15_clusters"]
    steps = [s for s in STEPS if s != "doublet_detection"] + ["doublet_detection"]
    big, res, mem, mres = _both(path, tmp_path / "out" / name, steps, doublet_channel_key="",
                                n_top_genes=2000, n_comps=50, marker_columns=marker_cols)
    assert res.errors == mres.errors and res.notes == mres.notes
    assert big.obs_names.equals(mem.obs_names) and big.var_names.equals(mem.var_names)
    for c in QC_OBS + ["doublet_score"]:
        np.testing.assert_array_equal(big.obs[c], mem.obs[c])
    assert (big.var["highly_variable"] == mem.var["highly_variable"]).all()
    np.testing.assert_array_equal(_dense(big.X), _dense(mem.X))
    key = markers_key(marker_cols[0])
    for f in ("names", "scores", "pvals", "pvals_adj"):
        for g in mem.uns[key][f].dtype.names:
            np.testing.assert_array_equal(np.asarray(big.uns[key][f][g]),
                                          np.asarray(mem.uns[key][f][g]))
    for c in ("S_score", "G2M_score", "dissociation_stress_score"):
        if c in mem.obs:
            np.testing.assert_array_equal(big.obs[c], mem.obs[c])
