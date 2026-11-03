"""The file's own UMAP is kept as X_umap_original before scView's UMAP replaces it."""

from __future__ import annotations

import subprocess
import sys

import anndata as ad
import numpy as np
import pandas as pd

from scview.core import replay_script
from scview.core.pipeline import ORIGINAL_UMAP_KEY, PipelineParams, keep_original_umap, run_pipeline

STEPS = ["normalization", "log_transform", "highly_variable_genes", "pca", "neighbors",
         "clustering", "embeddings"]
PARAMS = PipelineParams(n_top_genes=60, n_comps=10, umap_3d="off")


def _counts(seed: int = 0) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    parts = []
    for lo in (0, 40, 80):
        X = rng.poisson(0.3, size=(60, 150)).astype("float32")
        X[:, lo:lo + 40] += rng.poisson(5.0, size=(60, 40)).astype("float32")
        parts.append(X)
    a = ad.AnnData(np.vstack(parts))
    a.var_names = [f"G{i}" for i in range(150)]
    a.obs_names = [f"C{i}" for i in range(a.n_obs)]
    a.obs["sample"] = pd.Categorical(np.where(np.arange(a.n_obs) % 2, "s1", "s2"))
    return a


def _author_umap(a: ad.AnnData) -> np.ndarray:
    xy = np.random.default_rng(7).normal(size=(a.n_obs, 2)).astype(np.float32) * 100
    a.obsm["X_umap"] = xy
    return xy


def test_first_run_keeps_the_files_umap_and_a_second_run_does_not_replace_it():
    a = _counts()
    xy = _author_umap(a)
    out, res = run_pipeline(a, STEPS, PARAMS)
    assert res.errors == {}
    np.testing.assert_array_equal(out.obsm[ORIGINAL_UMAP_KEY], xy)
    assert not np.allclose(out.obsm["X_umap"], xy)  # scView's own now

    again, res = run_pipeline(out, ["embeddings"], PipelineParams(umap_min_dist=0.1, umap_3d="off"))
    assert res.errors == {}
    np.testing.assert_array_equal(again.obsm[ORIGINAL_UMAP_KEY], xy)


def test_reset_to_counts_keeps_it_from_raw():
    """CELLxGENE-style: normalised X, counts in .raw, the author's X_umap. Resetting
    clears every X_* embedding except the kept one."""
    a = _counts()
    xy = _author_umap(a)
    a.raw = a.copy()
    a.X = np.log1p(a.X)
    out, res = run_pipeline(a, ["reset_to_counts", *STEPS], PARAMS)
    assert res.errors == {}
    np.testing.assert_array_equal(out.obsm[ORIGINAL_UMAP_KEY], xy)

    # A later reset (scView's X_umap now) keeps the original and adds nothing
    out2, res = run_pipeline(out, ["reset_to_counts"], PARAMS)
    assert res.errors == {}
    np.testing.assert_array_equal(out2.obsm[ORIGINAL_UMAP_KEY], xy)
    assert "X_umap" not in out2.obsm


def test_no_file_umap_means_nothing_kept_even_on_a_rerun():
    a = _counts()
    out, _ = run_pipeline(a, STEPS, PARAMS)
    assert ORIGINAL_UMAP_KEY not in out.obsm
    again, _ = run_pipeline(out, ["embeddings"], PARAMS)  # X_umap is scView's: not kept
    assert ORIGINAL_UMAP_KEY not in again.obsm
    assert keep_original_umap(again) is False


def test_three_column_umap_keeps_two():
    a = _counts()
    a.obsm["X_umap"] = np.ones((a.n_obs, 3))
    assert keep_original_umap(a) is True
    assert a.obsm[ORIGINAL_UMAP_KEY].shape == (a.n_obs, 2)


def test_replay_keeps_it_once(tmp_path):
    inp = _counts()
    xy = _author_umap(inp)
    inp.raw = inp.copy()
    out, res = run_pipeline(inp.copy(), ["reset_to_counts", *STEPS, "embeddings"], PARAMS)
    assert res.errors == {}
    text = replay_script.build_script(out, dataset_id="t", dataset_name="test")
    assert text.count('adata.obsm["X_umap_original"] =') == 1
    src, script, dst = tmp_path / "in.h5ad", tmp_path / "replay.py", tmp_path / "out.h5ad"
    inp.write_h5ad(src)
    script.write_text(text)
    proc = subprocess.run([sys.executable, str(script), str(src), str(dst)],
                          capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr[-3000:]
    rep = ad.read_h5ad(dst)
    np.testing.assert_array_equal(rep.obsm[ORIGINAL_UMAP_KEY], xy)
    np.testing.assert_allclose(rep.obsm["X_umap"], out.obsm["X_umap"], atol=1e-4)
