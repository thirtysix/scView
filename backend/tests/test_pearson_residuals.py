"""Analytic Pearson residuals as a normalisation option (C091).

Residuals feed HVG selection and PCA only. Everything downstream (markers, DE, cell
cycle, the stress score, expression plots, pseudobulk) reads .raw or the counts,
which must come out identical under both methods. Tested on a sparse and a dense
input: anndata's .raw shares a dense X (the paul15 bug).
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
import scipy.sparse as sp

from scview.api.v1 import normalization as norm_api
from scview.core import normalization, provenance, scaling
from scview.core.assessor import assess_preprocessing, recorded_normalization_method
from scview.core.pipeline import (
    CELL_CYCLE_G2M_GENES,
    CELL_CYCLE_S_GENES,
    PipelineParams,
    looks_like_counts,
    run_pipeline,
)
from scview.core.rerun import descendants, plan_rerun
from scview.core.stress import SCORE_COLUMN, STRESS_GENES
from scview.core.uns_keys import markers_key

API = "/api/v1"
N_PER, N_GROUPS = 80, 3
PREP = ["normalization", "log_transform", "highly_variable_genes", "scaling", "pca"]
DOWNSTREAM = ["marker_genes", "cell_cycle", "dissociation_stress"]


def _counts(dense: bool = False, seed: int = 0) -> ad.AnnData:
    """Counts in three planted groups, with cell-cycle and stress genes present."""
    rng = np.random.default_rng(seed)
    names = (list(CELL_CYCLE_S_GENES[:12]) + list(CELL_CYCLE_G2M_GENES[:12])
             + list(STRESS_GENES[:12]) + [f"G{i}" for i in range(160)])
    n = N_PER * N_GROUPS
    size = rng.gamma(4.0, 0.5, size=(n, 1))  # uneven library sizes
    X = rng.poisson(size * rng.gamma(1.0, 1.0, size=len(names)))
    for k in range(N_GROUPS):
        rows = slice(k * N_PER, (k + 1) * N_PER)
        cols = slice(40 + 15 * k, 55 + 15 * k)
        X[rows, cols] += rng.poisson(6.0, (N_PER, 15))
    X = X.astype("float32")
    a = ad.AnnData(X if dense else sp.csr_matrix(X))
    a.var_names = names
    a.obs_names = [f"C{i}" for i in range(n)]
    a.obs["group"] = pd.Categorical([str(k) for k in range(N_GROUPS) for _ in range(N_PER)])
    a.obs["batch"] = pd.Categorical(["b1", "b2"] * (n // 2))
    return a


def _dense(M) -> np.ndarray:
    return M.toarray() if sp.issparse(M) else np.asarray(M)


def _params(method: str, **kw) -> PipelineParams:
    return PipelineParams(normalization_method=method, n_top_genes=60, n_comps=10,
                          marker_columns=["group"], **kw)


# --- the residuals themselves ------------------------------------------------


@pytest.mark.parametrize("dense", [False, True])
def test_residuals_match_scanpy_in_blocks(dense):
    a = _counts(dense=dense)
    expected = sc.experimental.pp.normalize_pearson_residuals(a, inplace=False)["X"]
    got = normalization.pearson_residuals(a.X, chunk=37)  # several uneven blocks
    assert got.dtype == np.float32 and got.shape == a.shape
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)


def test_residual_variance_matches_scanpys_hvg_ranking():
    a = _counts()
    df = sc.experimental.pp.highly_variable_genes(
        a, flavor="pearson_residuals", n_top_genes=60, inplace=False)
    np.testing.assert_allclose(
        normalization.residual_variance(a.X, chunk=50), df["residual_variances"], rtol=1e-4)


def test_an_all_zero_gene_gets_zero_residuals_not_nan():
    a = _counts()
    X = a.X.toarray()
    X[:, 5] = 0
    R = normalization.pearson_residuals(sp.csr_matrix(X))
    assert np.isfinite(R).all() and not R[:, 5].any()


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError, match="Unknown normalization method"):
        normalization.check_method("sct")
    _, res = run_pipeline(_counts(), ["normalization", "highly_variable_genes", "scaling"],
                          PipelineParams(normalization_method="sct"))
    assert set(res.errors) == {"normalization", "highly_variable_genes", "scaling"}
    assert "Unknown normalization method" in res.errors["normalization"]


# --- the pipeline ------------------------------------------------------------


@pytest.mark.parametrize("dense", [False, True])
def test_pearson_pipeline_feeds_pca_the_residuals_and_keeps_log_values(dense):
    a = _counts(dense=dense)
    counts = _dense(a.X).copy()
    out, res = run_pipeline(a, PREP, _params("pearson_residuals"))
    assert res.errors == {}

    # Variable genes: scanpy's pearson_residuals flavour on the counts.
    ref = ad.AnnData(counts.copy())
    ref.var_names = out.var_names
    df = sc.experimental.pp.highly_variable_genes(
        ref, flavor="pearson_residuals", n_top_genes=60, inplace=False)
    assert out.var["highly_variable"].dtype == bool
    assert list(out.var["highly_variable"]) == list(df["highly_variable"])
    assert "residual_variances" in out.var.columns and "dispersions_norm" not in out.var.columns

    # What PCA reads: the residuals of the variable genes (core/scaling.py; an older
    # scView put every gene's in X). X and .raw: the shifted log; counts kept.
    expected = sc.experimental.pp.normalize_pearson_residuals(ref, inplace=False)["X"]
    hvg = out.var["highly_variable"].to_numpy()
    M, mask = scaling.pca_matrix(out)
    assert np.array_equal(mask, hvg)
    np.testing.assert_allclose(M, expected[:, hvg], rtol=1e-5, atol=1e-5)
    lognorm = np.log1p(counts / counts.sum(axis=1, keepdims=True) * 1e4)
    np.testing.assert_allclose(_dense(out.X), lognorm, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(_dense(out.raw.X), lognorm, rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(_dense(out.layers["counts"]), counts)
    assert out.obsm["X_pca"].shape == (a.n_obs, 10)


def test_pearson_is_recorded_in_provenance_and_the_assessment():
    out, res = run_pipeline(_counts(), PREP, _params("pearson_residuals"))
    assert res.errors == {}
    prov = provenance.read_provenance(out)
    steps = {h["step"]: h for h in prov["history"]}
    assert steps["normalization"]["params"]["normalization_method"] == "pearson_residuals"
    assert steps["highly_variable_genes"]["params"]["hvg_flavor"] == "pearson_residuals"
    assert "pearson_residuals" in steps["highly_variable_genes"]["tool"]
    assert "Pearson residuals" in steps["scaling"]["tool"]
    assert steps["scaling"]["params"]["theta"] == normalization.THETA
    assert prov["current"]["normalization_method"] == "pearson_residuals"
    assert recorded_normalization_method(out) == "pearson_residuals"

    state = assess_preprocessing(out)
    assert "Pearson residual" in state.scaling.details
    assert "Pearson residual" in state.highly_variable_genes.details
    assert state.log_transform.done  # .raw is the shifted log, as under the default


def test_the_default_stays_the_shifted_log():
    assert PipelineParams().normalization_method == "log" == normalization.DEFAULT_METHOD
    out, res = run_pipeline(_counts(), PREP, PipelineParams(n_top_genes=60, n_comps=10))
    assert res.errors == {}
    assert "dispersions_norm" in out.var.columns and "residual_variances" not in out.var.columns
    assert recorded_normalization_method(out) == "log"
    assert "Pearson" not in assess_preprocessing(out).scaling.details
    assert recorded_normalization_method(_counts()) is None  # never processed


@pytest.mark.parametrize("dense", [False, True])
def test_downstream_reads_the_same_values_under_both_methods(dense):
    """Markers, cell cycle and the stress score read .raw: identical either way."""
    runs = {}
    for method in normalization.METHODS:
        out, res = run_pipeline(_counts(dense=dense), PREP + DOWNSTREAM, _params(method))
        assert res.errors == {} and not res.notes, (method, res.errors, res.notes)
        runs[method] = out
    log, pr = runs["log"], runs["pearson_residuals"]
    # PCA input does differ ...
    assert not np.allclose(scaling.pca_matrix(log)[0], scaling.pca_matrix(pr)[0])
    np.testing.assert_array_equal(_dense(log.raw.X), _dense(pr.raw.X))  # ... .raw does not
    np.testing.assert_array_equal(_dense(log.X), _dense(pr.X))  # nor X (both log values)
    np.testing.assert_array_equal(_dense(log.layers["counts"]), _dense(pr.layers["counts"]))

    key = markers_key("group")
    for group in ("0", "1", "2"):
        fc_log = np.asarray(log.uns[key]["logfoldchanges"][group], dtype=float)
        fc_pr = np.asarray(pr.uns[key]["logfoldchanges"][group], dtype=float)
        assert np.isfinite(fc_log).all() and np.isfinite(fc_pr).all()
        np.testing.assert_allclose(fc_pr, fc_log, rtol=1e-6)
        assert list(pr.uns[key]["names"][group]) == list(log.uns[key]["names"][group])
        # The planted markers lead, with a real fold change (not z-scores or residuals).
        top = set(pr.uns[key]["names"][group][:10])
        planted = set(pr.var_names[40 + 15 * int(group): 55 + 15 * int(group)])
        assert len(top & planted) >= 8 and fc_pr[:10].min() > 1
    for col in ("S_score", "G2M_score", SCORE_COLUMN):
        np.testing.assert_allclose(pr.obs[col], log.obs[col], rtol=1e-6)


def test_pearson_with_batch_key_ranks_within_batches():
    out, res = run_pipeline(_counts(), PREP, _params("pearson_residuals", hvg_batch_key="batch"))
    assert res.errors == {}
    assert "highly_variable_nbatches" in out.var.columns
    assert int(out.var["highly_variable"].sum()) == 60


def test_counts_held_only_in_raw_are_found_and_kept():
    a = _counts()
    counts = a.X.toarray()
    a.raw = a.copy()
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)  # an upload: X log-normalised, counts in .raw only
    out, res = run_pipeline(a, PREP[2:], _params("pearson_residuals"))
    assert res.errors == {}
    np.testing.assert_array_equal(_dense(out.layers["counts"]), counts)
    assert not looks_like_counts(out.raw.X)


def test_without_counts_pearson_says_so():
    a = _counts()
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)  # log values only, no counts anywhere
    out, res = run_pipeline(a, PREP[2:], _params("pearson_residuals"))
    assert "need raw counts" in res.errors["highly_variable_genes"]
    assert "need raw counts" in res.errors["scaling"]


def test_pearson_scaling_refuses_counts_in_x():
    # Without normalization + log, .raw would hold counts and markers would read them.
    out, res = run_pipeline(_counts(), ["highly_variable_genes", "scaling"],
                            _params("pearson_residuals"))
    assert "highly_variable_genes" not in res.errors
    assert "run normalization and log transform" in res.errors["scaling"]


def test_reset_to_counts_clears_the_pearson_hvg_columns():
    out, _ = run_pipeline(_counts(), PREP, _params("pearson_residuals"))
    reset, res = run_pipeline(out, ["reset_to_counts"], PipelineParams())
    assert res.errors == {}
    assert not {"residual_variances", "highly_variable_rank", "variances",
                "highly_variable"} & set(reset.var.columns)
    assert looks_like_counts(reset.X)


def test_rerun_treats_a_method_change_as_a_full_reprocess():
    ran = PREP + ["neighbors", "clustering"] + DOWNSTREAM
    for step in ("normalization", "highly_variable_genes", "scaling"):
        assert plan_rerun(ran, step).requires_reprocess
    # The scores read .raw (the log values), so they do not follow the method.
    assert not {"cell_cycle", "dissociation_stress"} & descendants("scaling")
    assert not {"cell_cycle", "dissociation_stress"} & descendants("highly_variable_genes")


# --- mean-variance diagnostic --------------------------------------------------


def test_residual_variance_is_about_one_for_sampling_noise():
    rng = np.random.default_rng(1)
    n = 3000
    size = rng.gamma(4.0, 0.5, size=(n, 1))
    X = rng.poisson(size * np.linspace(0.05, 5, 40))  # pure Poisson: no biology
    X[: n // 2, 0] += rng.poisson(8.0, n // 2)  # one gene that varies for a reason
    mv = normalization.mean_variance(sp.csr_matrix(X.astype("float32")),
                                     [f"G{i}" for i in range(40)])
    rv = mv["var_residuals"]
    assert np.all(np.abs(rv[1:] - 1) < 0.2)
    assert rv[0] > 5
    assert np.all(mv["var_counts"][1:] > 0)


def test_mean_variance_samples_cells_and_drops_undetected_genes():
    a = _counts()
    X = a.X.toarray()
    X[:, 3] = 0
    mv = normalization.mean_variance(sp.csr_matrix(X), a.var_names)
    assert mv["n_cells_used"] == a.n_obs
    assert a.var_names[3] not in mv["genes"] and len(mv["genes"]) == a.n_vars - 1
    assert not mv["keep"][3] and len(mv["var_residuals"]) == a.n_vars - 1
    sampled = normalization.mean_variance(sp.csr_matrix(X), a.var_names, max_cells=100)
    assert sampled["n_cells"] == a.n_obs and sampled["n_cells_used"] == 100
    assert len(sampled["genes"]) == int(sampled["keep"].sum()) == len(sampled["mean"])


def test_object_cache_forgets_a_collected_dataset():
    import gc

    from scview.core.object_cache import ObjectCache

    cache = ObjectCache()
    a, b = _counts(), _counts()
    cache.put(a, {"x": 1})
    cache.put(a, "other", key="k")
    assert cache.get(a) == {"x": 1} and cache.get(a, "k") == "other"
    assert cache.get(b) is None  # an equal but different object misses
    del a
    gc.collect()
    assert len(cache) == 0 and cache.get(b) is None


def _register(dm, a: ad.AnnData, did: str) -> str:
    d = Path(dm.data_dir) / "uploads" / did
    d.mkdir(parents=True)
    (d / "metadata.json").write_text(json.dumps({"id": did, "name": did, "status": "ready"}))
    a.write_h5ad(d / "data.h5ad")
    return did


async def test_mean_variance_endpoint(client, dataset_manager, monkeypatch):
    out, _ = run_pipeline(_counts(), PREP, _params("pearson_residuals"))
    did = _register(dataset_manager, out, "mv")
    threads: list[threading.Thread] = []
    real = norm_api.mean_variance_payload

    def spy(adata):
        threads.append(threading.current_thread())
        return real(adata)

    monkeypatch.setattr(norm_api, "mean_variance_payload", spy)
    r = await client.get(f"{API}/datasets/{did}/qc/mean-variance")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["normalization_method"] == "pearson_residuals"
    assert body["counts_source"] == "layers['counts']"
    n = body["n_genes"]
    assert n == out.n_vars
    for k in ("genes", "mean", "var_counts", "var_log", "var_residuals", "hvg"):
        assert len(body[k]) == n
    assert sum(body["hvg"]) == 60
    assert threads and threads[0] is not threading.main_thread()  # off the event loop
    r2 = await client.get(f"{API}/datasets/{did}/qc/mean-variance")
    assert r2.json() == body and len(threads) == 1  # cached for the loaded dataset


async def test_mean_variance_endpoint_without_counts(client, dataset_manager):
    a = _counts()
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    did = _register(dataset_manager, a, "mvlog")
    r = await client.get(f"{API}/datasets/{did}/qc/mean-variance")
    assert r.status_code == 409 and "need raw counts" in r.json()["detail"]
    assert (await client.get(f"{API}/datasets/ghost/qc/mean-variance")).status_code == 404


async def test_assessment_reports_the_recorded_method(client, dataset_manager):
    out, _ = run_pipeline(_counts(), PREP, _params("pearson_residuals"))
    did = _register(dataset_manager, out, "assess")
    r = await client.get(f"{API}/datasets/{did}/assessment")
    assert r.status_code == 200, r.text
    assert r.json()["normalization_method"] == "pearson_residuals"
