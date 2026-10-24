"""Regress-out before PCA (core/scaling.py ``regress_out``; C120).

The reference is scanpy's own chain on the same matrix: a dense float32 copy of the
log-normalised variable genes, ``sc.pp.regress_out`` on the chosen obs columns, then
``sc.pp.scale(max_value=10)`` and ``sc.tl.pca``. X is never regressed.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
import scipy.sparse as sp

from scview.core import chunked, large_run, preflight, provenance, replay_script, scaling
from scview.core.pipeline import (
    CELL_CYCLE_G2M_GENES,
    CELL_CYCLE_S_GENES,
    PipelineParams,
    run_pipeline,
)

API = "/api/v1"
PRE = ["qc_metrics", "normalization", "log_transform", "highly_variable_genes"]
PARAMS = dict(n_top_genes=120, n_comps=15)


def _counts(sparse: bool = True, n_per: int = 120, seed: int = 0) -> ad.AnnData:
    """Four groups with their own genes, cycling cells, depth varying per cell, a
    numeric per-cell covariate that drives some genes, and a categorical column."""
    rng = np.random.default_rng(seed)
    cc = list(CELL_CYCLE_S_GENES[:20]) + list(CELL_CYCLE_G2M_GENES[:20])
    names = [f"MT-{i}" for i in range(4)] + cc + [f"G{i}" for i in range(360)]
    n = 4 * n_per
    size = rng.lognormal(0.0, 0.5, n)
    drive = rng.normal(0.0, 1.0, n)
    mu = size[:, None] * rng.gamma(0.6, 1.0, len(names))[None, :] * 2.0
    for k in range(4):
        mu[k * n_per:(k + 1) * n_per, 60 + 50 * k:110 + 50 * k] *= 5.0
    mu[rng.random(n) < 0.25, 4:24] *= 5.0  # S-phase cells
    mu[:, 300:330] *= np.exp(0.8 * drive)[:, None]
    X = rng.poisson(mu).astype(np.float32)
    a = ad.AnnData(sp.csr_matrix(X) if sparse else X)
    a.var_names = names
    a.obs_names = [f"c{i}" for i in range(n)]
    a.obs["drive"] = drive
    a.obs["group"] = pd.Categorical([f"g{k}" for k in np.repeat(np.arange(4), n_per)])
    return a


def _prepared(sparse: bool = True) -> ad.AnnData:
    out, res = run_pipeline(_counts(sparse), PRE, PipelineParams(**PARAMS))
    assert res.errors == {}
    return out


def _scanpy_chain(lognorm: ad.AnnData, keys: list[str]) -> np.ndarray:
    """What PCA should read: scanpy's regress_out then scale, on a copy of the
    variable genes (dense float32)."""
    mask = lognorm.var["highly_variable"].to_numpy(dtype=bool)
    X = lognorm.X[:, mask].astype(np.float32)
    sub = ad.AnnData(X=X.toarray() if sp.issparse(X) else np.ascontiguousarray(X))
    sub.obs = lognorm.obs[keys].copy()
    sc.pp.regress_out(sub, keys)
    sc.pp.scale(sub, max_value=10)
    return np.asarray(sub.X)


@pytest.mark.parametrize("sparse", [True, False])
def test_pca_reads_scanpys_regress_out_then_scale(sparse):
    pre = _prepared(sparse)
    keys = ["total_counts", "pct_counts_mt", "drive"]
    out, res = run_pipeline(pre.copy(), ["scaling", "pca"], PipelineParams(regress_out=keys, **PARAMS))
    assert res.errors == {}
    M, mask = scaling.pca_matrix(out)
    ref = _scanpy_chain(pre, keys)
    assert M.shape == ref.shape
    np.testing.assert_allclose(M, ref, rtol=0, atol=2e-4)
    # The PCA itself: scanpy's PCA of the same chain
    sub = ad.AnnData(ref)
    sc.tl.pca(sub, n_comps=PARAMS["n_comps"], svd_solver="arpack")
    xp, xr = out.obsm["X_pca"], sub.obsm["X_pca"]
    assert np.max(np.abs(np.abs(xp) - np.abs(xr)) / np.abs(xr).max(axis=0)) < 1e-3
    np.testing.assert_allclose(out.uns["pca"]["variance_ratio"], sub.uns["pca"]["variance_ratio"],
                               rtol=1e-4)


def test_regressing_removes_the_covariate_and_leaves_x_alone():
    pre = _prepared()
    plain, _ = run_pipeline(pre.copy(), ["scaling", "pca"], PipelineParams(**PARAMS))
    out, _ = run_pipeline(pre.copy(), ["scaling", "pca"], PipelineParams(regress_out=["drive"], **PARAMS))
    # X and .raw keep the log values, as without regress-out
    assert (out.X != plain.X).nnz == 0 and (out.raw.X != plain.raw.X).nnz == 0
    # The genes the covariate drove no longer correlate with it in what PCA reads
    M, _ = scaling.pca_matrix(out)
    names = out.var_names[out.var["highly_variable"].to_numpy()]
    driven = np.flatnonzero(np.isin(names, [f"G{i}" for i in range(260, 290)]))
    assert driven.size
    d = out.obs["drive"].to_numpy()
    corr = [abs(np.corrcoef(M[:, j], d)[0, 1]) for j in driven]
    assert max(corr) < 1e-4
    P, _ = scaling.pca_matrix(plain)
    assert max(abs(np.corrcoef(P[:, j], d)[0, 1]) for j in driven) > 0.3
    # Recorded: the plan, the coefficients per gene (NaN rows for the other genes), provenance
    assert out.uns[scaling.PLAN_KEY]["regress_out"] == ["drive"]
    coef = out.varm[scaling.REGRESS_KEY]
    hvg = out.var["highly_variable"].to_numpy()
    assert coef.shape == (out.n_vars, 2) and np.isfinite(coef[hvg]).all() and np.isnan(coef[~hvg]).all()
    rec = [h for h in provenance.read_provenance(out)["history"] if h["step"] == "scaling"][-1]
    assert rec["params"]["regress_out"] == ["drive"]
    from scview.core.assessor import assess_preprocessing

    assert "z-scores (after regressing out drive) of the" in assess_preprocessing(out).scaling.details
    # Off by default: nothing stored
    assert scaling.REGRESS_KEY not in plain.varm and "regress_out" not in plain.uns[scaling.PLAN_KEY]


def test_cell_cycle_covariates_are_scored_at_the_step():
    pre = _prepared()
    assert "S_score" not in pre.obs
    keys = ["cc_difference"]
    out, res = run_pipeline(pre.copy(), ["scaling", "pca"], PipelineParams(regress_out=keys, **PARAMS))
    assert res.errors == {}
    for c in ("S_score", "G2M_score", "phase", "cc_difference"):
        assert c in out.obs, c
    np.testing.assert_allclose(out.obs["cc_difference"], out.obs["S_score"] - out.obs["G2M_score"])
    # The scores are the cell-cycle step's
    again, _ = run_pipeline(out.copy(), ["cell_cycle"], PipelineParams(**PARAMS))
    np.testing.assert_array_equal(again.obs["S_score"], out.obs["S_score"])
    lognorm = pre.copy()
    lognorm.obs["cc_difference"] = out.obs["cc_difference"].to_numpy()
    M, _ = scaling.pca_matrix(out)
    np.testing.assert_allclose(M, _scanpy_chain(lognorm, keys), rtol=0, atol=2e-4)
    rec = [h for h in provenance.read_provenance(out)["history"] if h["step"] == "scaling"][-1]
    assert rec["params"]["cell_cycle_scored"] is True


@pytest.mark.parametrize(("keys", "words"), [
    (["no_such"], "there is no column 'no_such'"),
    (["group"], "'group' is not a number per cell"),
])
def test_a_covariate_that_cannot_be_regressed_says_why(keys, words):
    out, res = run_pipeline(_prepared(), ["scaling", "pca"], PipelineParams(regress_out=keys, **PARAMS))
    assert words in res.errors["scaling"]
    assert res.skipped == {"pca": "scaling"}


def test_missing_values_and_pearson_residuals_are_refused():
    pre = _prepared()
    pre.obs.loc[pre.obs_names[3], "drive"] = np.nan
    _, res = run_pipeline(pre, ["scaling"], PipelineParams(regress_out=["drive"], **PARAMS))
    assert "has missing values" in res.errors["scaling"]
    pre = _prepared()
    _, res = run_pipeline(pre, ["scaling"], PipelineParams(
        regress_out=["drive"], normalization_method="pearson_residuals", **PARAMS))
    assert "Pearson residuals" in res.errors["scaling"]


def test_collinear_covariates_give_scanpys_residuals():
    pre = _prepared()
    pre.obs["drive2"] = 2.0 * pre.obs["drive"]
    out, res = run_pipeline(pre.copy(), ["scaling", "pca"],
                            PipelineParams(regress_out=["drive", "drive2"], **PARAMS))
    assert res.errors == {}
    M, _ = scaling.pca_matrix(out)
    # scanpy falls back to a GLM per gene for a singular design: the same residuals
    np.testing.assert_allclose(M, _scanpy_chain(pre, ["drive"]), rtol=0, atol=2e-4)


def test_genes_variable_after_scaling_are_fitted_at_pca():
    pre = _prepared()
    keys = ["drive"]
    planned, _ = run_pipeline(pre.copy(), ["scaling"], PipelineParams(regress_out=keys, **PARAMS))
    redone, res = run_pipeline(planned, ["highly_variable_genes", "pca"],
                               PipelineParams(n_top_genes=200, n_comps=15))
    assert res.errors == {}
    M, _ = scaling.pca_matrix(redone)
    ref_in = pre.copy()
    ref_in.var["highly_variable"] = redone.var["highly_variable"].to_numpy()
    np.testing.assert_allclose(M, _scanpy_chain(ref_in, keys), rtol=0, atol=2e-4)


def test_the_hvg_sensitivity_control_reproduces_a_regressed_dataset():
    from sklearn.metrics import adjusted_rand_score

    from scview.core import hvg_sensitivity as hs

    a, res = run_pipeline(_counts(), [*PRE, "scaling", "pca", "neighbors", "clustering"],
                          PipelineParams(regress_out=["total_counts", "drive"], clustering_resolution=0.5,
                                         **PARAMS))
    assert res.errors == {}
    rec = hs.recorded(a)
    assert rec.regress_out == ["total_counts", "drive"]
    inp = hs._inputs(a, rec)
    mask, labels = hs.run_setting(inp, rec.current, rec)
    assert np.array_equal(hs._pcs(inp, mask, rec.current, rec), a.obsm["X_pca"])
    assert adjusted_rand_score(a.obs[rec.reference], np.asarray(labels)) == 1.0


def test_the_qc_advice_points_at_the_option():
    from scview.core import cell_cycle, qc_drivers

    assert "Scaling step, Regress out" in cell_cycle.FLAG_HELP
    d = qc_drivers.Driver(score="S_score", label="S-phase score", pc=2, r=0.7, r_within=0.6,
                          inside_clusters=True, pc_share=0.1)
    assert "Scaling step, Regress out" in qc_drivers._reason(d)


def test_check_regress_out():
    assert scaling.check_regress_out(None) == [] and scaling.check_regress_out("") == []
    assert scaling.check_regress_out("drive") == ["drive"]
    assert scaling.check_regress_out(["a", " a", "b", ""]) == ["a", "b"]
    with pytest.raises(ValueError):
        scaling.check_regress_out([1, 2])


# -- large-dataset mode ---------------------------------------------------------------


def test_chunked_fit_and_statistics_match_in_memory(monkeypatch):
    import dask.array as da

    pre = _prepared()
    idx = np.flatnonzero(pre.var["highly_variable"].to_numpy())
    C = scaling.covariates(pre.obs, ["total_counts", "drive"])
    X = da.from_array(pre.X, chunks=(100, pre.n_vars), asarray=False)
    B = chunked.regression_coef(X, idx, C)
    np.testing.assert_allclose(B, scaling.regression_coef(pre.X, idx, C), rtol=1e-9, atol=1e-12)
    mean, std = chunked.regressed_stats(X, idx, C, B)
    m2, s2 = scaling.residual_stats(pre.X, idx, C, B)
    np.testing.assert_allclose(mean, m2, atol=1e-9)
    np.testing.assert_allclose(std, s2, rtol=1e-6)


def test_large_mode_regresses_as_in_memory(tmp_path, monkeypatch):
    monkeypatch.setattr(chunked, "rows_per_block", lambda *a, **k: 150)
    src = tmp_path / "in" / "data.h5ad"
    src.parent.mkdir()
    _counts().write_h5ad(src)
    steps = ["qc_metrics", "normalization", "log_transform", "highly_variable_genes", "scaling", "pca"]
    p = dict(regress_out=["total_counts", "cc_difference"], **PARAMS)
    res = large_run.run(src, steps, PipelineParams(**p, chunked="on"), str(tmp_path / "out" / "d.h5ad"))
    assert res.status == "complete", res.errors
    big = ad.read_h5ad(tmp_path / "out" / "d.h5ad")
    mem, mres = run_pipeline(ad.read_h5ad(src), steps, PipelineParams(**p))
    assert mres.errors == {}
    np.testing.assert_allclose(big.obs["cc_difference"], mem.obs["cc_difference"], rtol=1e-6)
    np.testing.assert_allclose(big.varm[scaling.REGRESS_KEY], mem.varm[scaling.REGRESS_KEY],
                               rtol=1e-7, atol=1e-10)
    np.testing.assert_allclose(big.var["std"], mem.var["std"], rtol=1e-6)
    xb, xm = big.obsm["X_pca"], mem.obsm["X_pca"]
    assert np.max(np.abs(np.abs(xb) - np.abs(xm)) / np.abs(xm).max(axis=0)) < 5e-3


# -- replay, preflight, API ------------------------------------------------------------


def test_the_replay_script_regresses_with_scanpy(tmp_path: Path):
    inp = _counts(seed=2)
    steps = [*PRE, "scaling", "pca"]
    out, res = run_pipeline(inp.copy(), steps,
                            PipelineParams(regress_out=["total_counts", "cc_difference"], **PARAMS))
    assert res.errors == {}
    assert {d["status"] for d in replay_script.describe(out)} == {"exact"}
    script = replay_script.build_script(out, dataset_id="t", dataset_name="test")
    assert "sc.pp.regress_out(sub, ['total_counts', 'cc_difference'])" in script
    src, path, dst = tmp_path / "input.h5ad", tmp_path / "replay.py", tmp_path / "replayed.h5ad"
    inp.write_h5ad(src)
    path.write_text(script)
    proc = subprocess.run([sys.executable, str(path), str(src), str(dst)],
                          capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr[-3000:]
    rep = ad.read_h5ad(dst)
    np.testing.assert_allclose(rep.obs["cc_difference"], out.obs["cc_difference"])
    xr, xo = rep.obsm["X_pca"], out.obsm["X_pca"]
    assert np.max(np.abs(np.abs(xr) - np.abs(xo)) / np.abs(xo).max(axis=0)) < 1e-3


def test_preflight_counts_the_regression():
    h = preflight.Header("h5ad", n_cells=100_000, n_genes=20_000, matrices=[
        preflight.Matrix("X", 100_000, 20_000, True, 200_000_000, "float32", "int32")])
    steps = ["highly_variable_genes", "scaling", "pca"]

    def est(**kw):
        run = preflight.estimate_run(h, steps=steps, params=PipelineParams(**kw), cpus=2, warm=True)
        return {e.step: e for e in run.steps}

    by, rby = est(), est(regress_out=["total_counts", "pct_counts_mt"])
    for step in ("scaling", "pca"):
        assert rby[step].seconds[1] > by[step].seconds[1], step
        assert rby[step].transient_bytes > by[step].transient_bytes, step
    # 100k x 2,000 variable genes: the fit adds about 1.8 s (9e-9 s per cell x gene)
    assert 1.0 < (rby["scaling"].seconds[1] - by["scaling"].seconds[1]) / 2 < 3.0
    assert est(regress_out=["cc_difference"])["scaling"].seconds[1] > rby["scaling"].seconds[1]
    big = preflight.estimate_chunked_run(h, steps, PipelineParams(regress_out=["total_counts"]))
    small = preflight.estimate_chunked_run(h, steps, PipelineParams())
    for step in ("scaling", "pca"):
        assert ({e.step: e for e in big.steps}[step].seconds[1]
                > {e.step: e for e in small.steps}[step].seconds[1]), step


@pytest.mark.asyncio
async def test_the_run_endpoint_refuses_a_regress_out_that_is_not_a_list(client, dataset_manager):
    r = await client.post(f"{API}/datasets/nope/assessment/run",
                          json={"steps": ["scaling"], "params": {"regress_out": [1, 2]}})
    assert r.status_code == 422 and "regress_out" in r.json()["detail"]
