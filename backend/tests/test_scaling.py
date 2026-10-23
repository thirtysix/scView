"""Scaling builds only what PCA reads (core/scaling.py).

The old scaling step made X dense float32 for every gene and z-scored it; PCA then read
the variable genes through a mask. The references below do exactly that, explicitly,
in the test, so a different computation (another layout, other statistics, residuals
with the wrong totals) fails them: the results must be the old ones bit for bit.
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

from scview.core import leiden, normalization, provenance, replay_script, scaling
from scview.core.anndata_adaptor import AnnDataAdaptor
from scview.core.assessor import assess_preprocessing
from scview.core.pipeline import PipelineParams, run_pipeline

PRE = ["normalization", "log_transform", "highly_variable_genes"]
PARAMS = dict(n_top_genes=120, n_comps=20)


def _counts(sparse: bool = True, n_per: int = 150, n_genes: int = 500, seed: int = 0) -> ad.AnnData:
    """Four populations with their own gene blocks, on a noisy background."""
    rng = np.random.default_rng(seed)
    parts = []
    for k in range(4):
        X = rng.poisson(rng.gamma(0.6, 0.5, n_genes), (n_per, n_genes)).astype(np.float32)
        lo = (n_genes - 40) * k // 4
        X[:, lo:lo + 40] += rng.poisson(3.0, (n_per, 40)).astype(np.float32)
        parts.append(X)
    X = np.vstack(parts)
    a = ad.AnnData(sp.csr_matrix(X) if sparse else X)
    a.var_names = [f"G{i}" for i in range(n_genes)]
    a.obs_names = [f"c{i}" for i in range(X.shape[0])]
    return a


def _log_hvg(a: ad.AnnData, method: str = "log") -> ad.AnnData:
    out, res = run_pipeline(a, PRE, PipelineParams(normalization_method=method, **PARAMS))
    assert res.errors == {}
    return out


def _reference(lognorm_hvg: ad.AnnData, n_comps: int = PARAMS["n_comps"]) -> ad.AnnData:
    """The old steps, written out: X dense float32 for EVERY gene, sc.pp.scale(max_value=10),
    then sc.tl.pca with the default mask (the variable genes)."""
    ref = lognorm_hvg.copy()
    X = ref.X.astype(np.float32)
    ref.X = X.toarray() if sp.issparse(X) else X
    sc.pp.scale(ref, max_value=10)
    sc.tl.pca(ref, n_comps=min(n_comps, ref.n_obs - 1, ref.n_vars - 1), svd_solver="arpack")
    return ref


def _pearson_reference(lognorm_hvg: ad.AnnData, n_comps: int = PARAMS["n_comps"]) -> ad.AnnData:
    """The old Pearson step: residuals of every gene in X, then the masked PCA."""
    ref = lognorm_hvg.copy()
    counts, _ = normalization.find_counts(ref)
    ref.X = normalization.pearson_residuals(counts, clip=normalization.default_clip(ref.n_obs))
    sc.tl.pca(ref, n_comps=min(n_comps, ref.n_obs - 1, ref.n_vars - 1), svd_solver="arpack")
    return ref


def _same_pca(new: ad.AnnData, ref: ad.AnnData) -> None:
    assert np.array_equal(new.obsm["X_pca"], ref.obsm["X_pca"])
    assert new.obsm["X_pca"].dtype == ref.obsm["X_pca"].dtype == np.float32
    assert np.array_equal(new.varm["PCs"], ref.varm["PCs"]) and new.varm["PCs"].dtype == ref.varm["PCs"].dtype
    for k in ("variance", "variance_ratio"):
        assert np.array_equal(new.uns["pca"][k], ref.uns["pca"][k]), k
    assert new.uns["pca"]["params"] == ref.uns["pca"]["params"]


def _leiden(a: ad.AnnData) -> np.ndarray:
    g = a.copy()
    sc.pp.neighbors(g, n_neighbors=15, use_rep="X_pca")
    leiden.leiden(g, 0.8, "c")
    return g.obs["c"].astype(str).to_numpy()


# -- the same PCA as before, bit for bit --------------------------------------------


@pytest.mark.parametrize("sparse", [True, False])
def test_pca_on_the_planned_genes_is_the_old_pca_bit_for_bit(sparse):
    pre = _log_hvg(_counts(sparse=sparse))
    ref = _reference(pre)
    new, res = run_pipeline(pre.copy(), ["scaling", "pca"], PipelineParams(**PARAMS))
    assert res.errors == {}
    _same_pca(new, ref)
    assert (np.abs(new.varm["PCs"][~new.var["highly_variable"].to_numpy()]) == 0).all()
    assert np.array_equal(_leiden(new), _leiden(ref))


def test_pearson_residuals_for_pca_are_the_old_ones_bit_for_bit():
    pre = _log_hvg(_counts(), "pearson_residuals")
    ref = _pearson_reference(pre)
    new, res = run_pipeline(pre.copy(), ["scaling", "pca"],
                            PipelineParams(normalization_method="pearson_residuals", **PARAMS))
    assert res.errors == {}
    _same_pca(new, ref)
    assert np.array_equal(_leiden(new), _leiden(ref))


def test_without_variable_genes_every_gene_is_scaled_as_before():
    pre, _ = run_pipeline(_counts(n_genes=150), ["normalization", "log_transform"], PipelineParams())
    ref = _reference(pre)  # no highly_variable column: no mask, X as it is (row-major)
    new, res = run_pipeline(pre.copy(), ["scaling", "pca"], PipelineParams(**PARAMS))
    assert res.errors == {} and scaling.read_plan(new)["genes"] == "all"
    _same_pca(new, ref)


# -- what the dataset holds after the steps --------------------------------------------


def test_x_stays_log_normalised_for_every_gene_and_the_plan_is_recorded():
    a = _counts()
    out, res = run_pipeline(a, ["normalization", "log_transform", "highly_variable_genes", "scaling", "pca"],
                            PipelineParams(**PARAMS))
    assert res.errors == {}
    assert sp.issparse(out.X) and out.X.dtype == np.float32 and out.shape == a.shape
    assert out.layers["counts"].shape == a.shape  # every gene's counts kept
    assert out.raw is not None and out.raw.shape == a.shape
    assert np.array_equal(out.raw.X.toarray(), out.X.toarray())  # .raw = the log values
    assert not scaling.holds_scaled(out.X)
    plan = scaling.read_plan(out)
    assert plan == {"method": "zscore", "genes": "highly_variable", "max_value": 10.0}
    hvg = out.var["highly_variable"].to_numpy()
    assert np.isfinite(out.var["mean"].to_numpy()[hvg]).all()
    assert np.isnan(out.var["std"].to_numpy()[~hvg]).all()
    # What the scaling step recorded says what it did (the replay script reads it).
    rec = [h for h in provenance.read_provenance(out)["history"] if h["step"] == "scaling"][-1]["params"]
    assert rec["scaled_genes"] == "highly_variable" and rec["n_scaled_genes"] == int(hvg.sum())


def test_the_plan_survives_saving_and_pca_alone_after_a_reload(tmp_path):
    pre = _log_hvg(_counts())
    ref = _reference(pre)
    planned, _ = run_pipeline(pre.copy(), ["scaling"], PipelineParams(**PARAMS))
    planned.write_h5ad(tmp_path / "d.h5ad")
    back = ad.read_h5ad(tmp_path / "d.h5ad")
    assert scaling.read_plan(back) == scaling.read_plan(planned)
    new, res = run_pipeline(back, ["pca"], PipelineParams(**PARAMS))
    assert res.errors == {}
    _same_pca(new, ref)


def test_pca_without_a_plan_reads_x_as_it_is():
    # Scaling skipped: PCA on the log values of the variable genes, as before.
    pre = _log_hvg(_counts())
    ref = pre.copy()
    sc.tl.pca(ref, n_comps=PARAMS["n_comps"], svd_solver="arpack")
    new, _ = run_pipeline(pre.copy(), ["pca"], PipelineParams(**PARAMS))
    _same_pca(new, ref)


def test_a_dataset_an_older_scview_scaled_keeps_working():
    # X dense z-scores of every gene, .raw log values, no plan (an older scView's run).
    pre = _log_hvg(_counts())
    old = pre.copy()
    old.raw = old
    X = old.X.astype(np.float32)
    old.X = X.toarray()
    sc.pp.scale(old, max_value=10)
    assert scaling.holds_scaled(old.X) and scaling.read_plan(old) is None
    ref = _reference(pre)
    again, _ = run_pipeline(old.copy(), ["pca"], PipelineParams(**PARAMS))
    _same_pca(again, ref)  # PCA on X as it is: the same as when it was made
    # Scaling again over the z-scores keeps .raw's log values (it used to make .raw scaled).
    rescaled, res = run_pipeline(old.copy(), ["scaling", "pca"], PipelineParams(**PARAMS))
    assert res.errors == {}
    assert not scaling.holds_scaled(rescaled.raw.X)
    assert np.array_equal(rescaled.raw.X.toarray(), pre.X.toarray())


def test_genes_that_became_variable_after_scaling_get_their_statistics_at_pca():
    pre = _log_hvg(_counts())
    planned, _ = run_pipeline(pre.copy(), ["scaling"], PipelineParams(**PARAMS))
    # A different gene set is chosen afterwards (another HVG run, no scaling).
    redone, res = run_pipeline(planned, ["highly_variable_genes", "pca"],
                               PipelineParams(n_top_genes=200, n_comps=20))
    assert res.errors == {}
    ref_in = pre.copy()
    ref_in.var["highly_variable"] = redone.var["highly_variable"].to_numpy()
    _same_pca(redone, _reference(ref_in))


def test_normalising_again_or_a_reset_forgets_the_plan():
    pre = _log_hvg(_counts())
    planned, _ = run_pipeline(pre.copy(), ["scaling"], PipelineParams(**PARAMS))
    assert scaling.read_plan(planned) is not None
    reset, _ = run_pipeline(planned.copy(), ["reset_to_counts"], PipelineParams())
    assert scaling.read_plan(reset) is None
    for step in ("normalization", "log_transform"):
        again, _ = run_pipeline(planned.copy(), [step], PipelineParams())
        assert scaling.read_plan(again) is None, step


# -- what readers say about X -------------------------------------------------------


def test_the_x_layer_says_what_it_holds(tmp_path):
    pre = _log_hvg(_counts())
    new, _ = run_pipeline(pre.copy(), ["scaling", "pca"], PipelineParams(**PARAMS))
    old = pre.copy()
    old.raw = old
    old.X = old.X.toarray()
    sc.pp.scale(old, max_value=10)
    labels = {}
    for name, a in (("new", new), ("old", old), ("upload", _counts())):
        a.write_h5ad(tmp_path / f"{name}.h5ad")
        layers = AnnDataAdaptor(str(tmp_path / f"{name}.h5ad")).available_expression_layers()
        labels[name] = next(x["label"] for x in layers if x["key"] == "X")
    assert labels == {"new": "Log-normalized (X)", "old": "Scaled (X)", "upload": "Main matrix (X)"}


def test_the_assessment_says_pca_reads_the_scaled_variable_genes():
    a = _counts()
    out, _ = run_pipeline(a, ["normalization", "log_transform", "highly_variable_genes", "scaling"],
                          PipelineParams(**PARAMS))
    state = assess_preprocessing(out)
    assert state.scaling.done
    assert state.scaling.details.startswith(
        f"PCA reads z-scores of the {int(out.var['highly_variable'].sum()):,} variable genes; X keeps")


# -- the replay script ----------------------------------------------------------------


def _replay(tmp_path: Path, inp: ad.AnnData, out: ad.AnnData) -> ad.AnnData:
    src, script, dst = tmp_path / "input.h5ad", tmp_path / "replay.py", tmp_path / "replayed.h5ad"
    inp.write_h5ad(src)
    script.write_text(replay_script.build_script(out, dataset_id="t", dataset_name="test"))
    proc = subprocess.run([sys.executable, str(script), str(src), str(dst)],
                          capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr[-3000:]
    return ad.read_h5ad(dst)


@pytest.mark.parametrize("method", ["log", "pearson_residuals"])
def test_the_replay_script_scales_a_copy_and_gets_the_same_pca(tmp_path, method):
    inp = _counts(seed=3)
    steps = ["normalization", "log_transform", "highly_variable_genes", "scaling", "pca"]
    out, res = run_pipeline(inp.copy(), steps, PipelineParams(normalization_method=method, **PARAMS))
    assert res.errors == {}
    assert {d["status"] for d in replay_script.describe(out)} == {"exact"}
    rep = _replay(tmp_path, inp, out)
    assert sp.issparse(rep.X) and not scaling.holds_scaled(rep.X)  # X kept the log values
    _same_pca(rep, out)


def test_an_older_scaling_record_still_replays_into_x():
    a = ad.AnnData(np.zeros((3, 2), dtype="float32"))
    provenance.record_step(a, step="scaling", tool="t", params={"normalization_method": "log"})
    provenance.record_step(a, step="pca", tool="t", params={"n_comps": 50})
    script = replay_script.build_script(a, dataset_id="t", dataset_name="t")
    assert "sc.pp.scale(adata, max_value=10)" in script and "sub = ad.AnnData" not in script
    compile(script, "replay.py", "exec")


@pytest.mark.parametrize("sparse", [True, False])
def test_scaling_never_holds_more_than_a_block_of_dense_genes(monkeypatch, sparse):
    # A sparse X: the statistics come from its stored values, nothing dense at all. A
    # dense X: a block of STAT_BLOCK genes at a time.
    pre = _log_hvg(_counts(sparse=sparse))
    widths = []
    dense = scaling._dense

    def watched(X, idx, order):
        M = dense(X, idx, order)
        widths.append(M.shape[1])
        return M

    monkeypatch.setattr(scaling, "_dense", watched)
    monkeypatch.setattr(scaling, "STAT_BLOCK", 32)
    monkeypatch.setattr(scaling, "SPARSE_CHUNK", 5000)  # several passes' worth of chunks
    planned, _ = run_pipeline(pre.copy(), ["scaling"], PipelineParams(**PARAMS))
    if sparse:
        assert widths == []
    else:
        assert widths and max(widths) <= 32 and sum(widths) == int(pre.var["highly_variable"].sum())
    # Either way the statistics sc.pp.scale works out on every gene densified at once.
    ref = pre.copy()
    ref.X = pre.X.toarray() if sp.issparse(pre.X) else pre.X.copy()
    sc.pp.scale(ref, max_value=10)
    hvg = pre.var["highly_variable"].to_numpy()
    for col in ("mean", "std"):
        assert np.array_equal(planned.var[col].to_numpy()[hvg], ref.var[col].to_numpy()[hvg]), col


def test_pearson_residual_columns_can_be_laid_out_column_major():
    a = _counts(n_genes=80)
    cols = np.array([3, 10, 40, 79])
    full = normalization.pearson_residuals(a.X)
    part = normalization.pearson_residuals(a.X, columns=cols, order="F")
    assert part.flags["F_CONTIGUOUS"] and np.array_equal(part, full[:, cols])


def test_holds_scaled_tells_scaled_values_from_log_values_and_counts():
    a = _counts()
    assert not scaling.holds_scaled(a.X)
    assert not scaling.holds_scaled(np.log1p(a.X.toarray()))
    z = pd.DataFrame(a.X.toarray()).apply(lambda c: (c - c.mean()) / (c.std() or 1)).to_numpy()
    assert scaling.holds_scaled(z)
