"""The file's own PCA is kept as X_pca_original before scView's PCA replaces it (QA round 5:
Pijuan-Sala's 50 author PCs were lost while their UMAP was kept)."""

from __future__ import annotations

import subprocess
import sys

import anndata as ad
import numpy as np

from scview.core import replay_script
from scview.core.pipeline import ORIGINAL_PCA_KEY, PipelineParams, keep_original_pca, run_pipeline
from tests.test_original_umap import PARAMS, STEPS, _counts


def _author_pca(a: ad.AnnData) -> np.ndarray:
    pcs = np.random.default_rng(3).normal(size=(a.n_obs, 50)).astype(np.float32)
    a.obsm["X_pca"] = pcs
    return pcs


def test_first_run_keeps_the_files_pca_and_a_rerun_does_not_replace_it():
    a = _counts()
    pcs = _author_pca(a)
    out, res = run_pipeline(a, STEPS, PARAMS)
    assert res.errors == {}
    np.testing.assert_array_equal(out.obsm[ORIGINAL_PCA_KEY], pcs)
    assert out.obsm["X_pca"].shape[1] == 10  # scView's own now
    again, res = run_pipeline(out, ["pca"], PARAMS)
    assert res.errors == {}
    np.testing.assert_array_equal(again.obsm[ORIGINAL_PCA_KEY], pcs)


def test_reset_to_counts_keeps_it_and_a_scview_pca_is_never_kept():
    a = _counts()
    pcs = _author_pca(a)
    a.raw = a.copy()
    a.X = np.log1p(a.X)
    out, res = run_pipeline(a, ["reset_to_counts"], PARAMS)
    assert res.errors == {} and "X_pca" not in out.obsm
    np.testing.assert_array_equal(out.obsm[ORIGINAL_PCA_KEY], pcs)

    plain, _ = run_pipeline(_counts(), STEPS, PARAMS)
    assert ORIGINAL_PCA_KEY not in plain.obsm and keep_original_pca(plain) is False


def test_replay_keeps_it_once(tmp_path):
    inp = _counts()
    pcs = _author_pca(inp)
    inp.raw = inp.copy()
    out, res = run_pipeline(inp.copy(), ["reset_to_counts", *STEPS, "pca"], PipelineParams(
        n_top_genes=60, n_comps=10, umap_3d="off"))
    assert res.errors == {}
    text = replay_script.build_script(out, dataset_id="t", dataset_name="test")
    assert text.count('adata.obsm["X_pca_original"] =') == 1
    src, script, dst = tmp_path / "in.h5ad", tmp_path / "replay.py", tmp_path / "out.h5ad"
    inp.write_h5ad(src)
    script.write_text(text)
    proc = subprocess.run([sys.executable, str(script), str(src), str(dst)],
                          capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr[-3000:]
    np.testing.assert_array_equal(ad.read_h5ad(dst).obsm[ORIGINAL_PCA_KEY], pcs)
