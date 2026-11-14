"""Logistic regression with a depth covariate for differential peaks (core/atac_lr.py, atac._iter_lr)."""

from __future__ import annotations

import gzip
import inspect
import subprocess
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from scipy.optimize import minimize
from scipy.stats import chi2

from scview.core import atac, atac_import, atac_lr, gene_annotation, modality_store, provenance, results_store
from tests import multiome_fixture as mf

# R 4.4 glm(binomial) + lmtest::lrtest, Seurat's LRDETest on these vectors (group ~ GENE + lat vs group ~ lat).
Y = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float)
LATENT = np.array([100, 250, 400, 120, 800, 300, 500, 90, 650, 220, 300, 410, 700, 150, 380, 260], dtype=float)
X = np.array([2.1, 0, 3.3, 1.2, 0, 2.8, 0, 0.7, 0, 0, 1.1, 0, 0, 0, 2.0, 0])
R_LRT, R_P, R_DEV_ALT, R_DEV_NULL, R_GENE = 4.479148404148, 0.0343108132402933, 16.568868534256, 21.048016938404, 1.1737987713
X_SEPARATED = np.array([2.1, 0, 3.3, 1.2, 0, 2.8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # open only in the group
R_SEP_LRT, R_SEP_P = 11.0999490700, 0.000863300901822


def _z(latent: np.ndarray = LATENT) -> np.ndarray:
    return (latent - latent.mean()) / latent.std()


def test_the_fit_equals_glm_and_lrtest_with_the_latent_scaled():
    b, dev, ok = atac_lr.logistic_null(Y, _z())
    assert ok and dev == pytest.approx(R_DEV_NULL, abs=1e-8)
    stat, gamma, status = atac_lr.lrt_fit(X[:, None], Y, _z(), b, dev)
    assert status[0] == atac_lr.FIT_OK
    assert stat[0] == pytest.approx(R_LRT, abs=1e-7)
    assert gamma[0] == pytest.approx(R_GENE, abs=1e-6)  # R models the second level, the rest: its sign is the opposite
    z, p = atac_lr.signed_root_p(stat, gamma)
    assert p[0] == pytest.approx(R_P, rel=1e-7) and z[0] > 0
    assert p[0] == pytest.approx(chi2.sf(R_LRT, 1), rel=1e-7)


def test_separation_is_named_and_reports_the_deviances_limit_as_glm_does():
    b, dev, _ = atac_lr.logistic_null(Y, _z())
    stat, gamma, status = atac_lr.lrt_fit(X_SEPARATED[:, None], Y, _z(), b, dev)
    assert status[0] == atac_lr.FIT_SEPARATED and gamma[0] > 5
    assert stat[0] == pytest.approx(R_SEP_LRT, abs=1e-3)
    assert atac_lr.signed_root_p(stat, gamma)[1][0] == pytest.approx(R_SEP_P, rel=1e-3)


def test_a_cell_far_out_on_depth_is_not_separation():
    # PBMC 10k: one cell at 44 sd of depth and a low-depth cluster: a large fitted |eta|, finite estimates.
    rng = np.random.default_rng(1)
    n = 800
    latent = rng.lognormal(9.5, 0.6, n)
    y = (rng.random(n) < 0.3 * (latent < np.median(latent))).astype(float)
    latent[0] = latent.max() * 40
    y[0] = 0.0
    z = _z(latent)
    assert z.max() > 20
    b, dev, ok = atac_lr.logistic_null(y, z)
    assert ok and abs(b[0] + b[1] * z.max()) > 25
    x = np.where(rng.random((n, 3)) < 0.3, rng.exponential(1.0, (n, 3)) + 1.0, 0.0)
    stat, _g, status = atac_lr.lrt_fit(x, y, z, b, dev)
    assert (status == atac_lr.FIT_OK).all() and np.isfinite(stat).all()


def test_a_covariate_that_alone_separates_the_group_leaves_no_null():
    z = np.arange(40, dtype=float)
    y = (z < 10).astype(float)
    assert atac_lr.logistic_null(y, z)[2] is False
    assert atac_lr.logistic_null(1.0 - y, z)[2] is False
    y[3], y[20] = 0.0, 1.0
    assert atac_lr.logistic_null(y, (z - z.mean()) / z.std())[2] is True


def test_the_fits_vector_products_use_no_blas_and_equal_it():
    """OpenBLAS threads are the whole server's: the pool's fits take their vector x block
    products in numpy's own loop (v0.78: PBMC 10k fits 146 s -> 107 s at 2 CPUs)."""
    rng = np.random.default_rng(0)
    v, m = rng.normal(size=500), rng.normal(size=(500, 40))
    np.testing.assert_allclose(atac_lr.vec_mat(v, m), v @ m, rtol=1e-12, atol=1e-12)
    assert atac_lr.vec_mat(v, m[:, 0]) == pytest.approx(v @ m[:, 0])
    assert atac_lr.vec_mat in atac_lr.REPLAY_CODE
    # The pool's code takes no matrix product (gemv / gemm would start BLAS threads).
    for fn in (atac_lr.lrt_fit, atac_lr.deviance_and_mu):
        body = inspect.getsource(fn).split('"""')[-1]
        assert " @ " not in body and "np.dot" not in body and "matmul" not in body, fn.__name__


def test_a_block_of_fits_equals_each_fit_alone_and_a_direct_optimiser():
    rng = np.random.default_rng(3)
    n = 400
    z = rng.normal(size=n)
    y = (rng.random(n) < 0.3 + 0.1 * (z > 0)).astype(float)
    x = np.where(rng.random((n, 12)) < 0.3, rng.exponential(2.0, (n, 12)), 0.0)
    x[:, 3] += 2.0 * y * (rng.random(n) < 0.8)  # one real effect
    b, dev, ok = atac_lr.logistic_null(y, z)
    assert ok
    stat, gamma, status = atac_lr.lrt_fit(x, y, z, b, dev)
    assert (status == atac_lr.FIT_OK).all() and np.argmax(stat) == 3 and gamma[3] > 0
    for j in range(x.shape[1]):
        alone, _g, _s = atac_lr.lrt_fit(x[:, [j]], y, z, b, dev)
        assert alone[0] == pytest.approx(stat[j], abs=1e-9)

    def loss(par: np.ndarray, column: np.ndarray) -> float:
        eta = par[0] + par[1] * z + par[2] * column
        return 2.0 * float(np.logaddexp(0.0, -(2 * y - 1) * eta).sum())

    direct = minimize(loss, np.array([b[0], b[1], 0.0]), args=(x[:, 3],), method="BFGS", options={"gtol": 1e-6})
    assert dev - direct.fun == pytest.approx(stat[3], abs=1e-3)


def test_the_score_products_equal_the_dense_score_test():
    rng = np.random.default_rng(5)
    n, m = 300, 40
    z = rng.normal(size=n)
    y = (rng.random(n) < 0.25).astype(float)
    dense = np.where(rng.random((n, m)) < 0.2, rng.exponential(1.0, (n, m)), 0.0)
    b, _dev, _ok = atac_lr.logistic_null(y, z)
    residual, w, wz, inverse = atac_lr.null_vectors(y, z, b)
    u, a, bb, c = np.zeros((m, 1)), np.zeros((m, 1)), np.zeros((m, 1)), np.zeros((m, 1))
    for lo, hi in ((0, 120), (120, 300)):  # blocks of cells add up
        parts = atac_lr.score_products(sp.csr_matrix(dense[lo:hi]), residual[lo:hi, None], w[lo:hi, None], wz[lo:hi, None])
        for total, part in zip((u, a, bb, c), parts):
            total += part
    stat, _v = atac_lr.score_statistic(u[:, 0], a[:, 0], bb[:, 0], c[:, 0], inverse)
    design = np.column_stack([np.ones(n), z])
    for j in range(m):
        xj = dense[:, j]
        v = (xj * xj * w).sum() - (xj * w) @ design @ np.linalg.inv((design * w[:, None]).T @ design) @ design.T @ (xj * w)
        assert stat[j] == pytest.approx((xj @ residual) ** 2 / v, rel=1e-9)


def test_a_peak_with_no_variance_left_has_no_score_rather_than_a_p_of_one():
    z = _z()
    b, _dev, _ok = atac_lr.logistic_null(Y, z)
    residual, w, wz, inverse = atac_lr.null_vectors(Y, z, b)
    parts = atac_lr.score_products(sp.csr_matrix(np.column_stack([np.zeros(Y.size), X])), residual[:, None], w[:, None], wz[:, None])
    stat, _v = atac_lr.score_statistic(*(q[:, 0] for q in parts), inverse)
    assert np.isnan(stat[0]) and np.isfinite(stat[1])
    assert np.isnan(atac_lr.signed_root_p(stat, parts[0][:, 0])[1][0])


@pytest.fixture()
def world(tmp_path):
    g, p, barcodes, table = mf.matrices(n_per=40)
    a = ad.AnnData(X=g, obs=pd.DataFrame({"group": pd.Categorical(["A"] * 40 + ["B"] * 40)}, index=barcodes))
    path = atac.store_path(tmp_path, "d1")
    modality_store.write(path, {atac.MATRIX: p}, barcodes, table["name"].tolist(),
                         atac_import.feature_attrs(table["name"].tolist()), {"genome": "GRCh38"})
    store = modality_store.ModalityStore(path)
    yield {"dir": tmp_path, "adata": a, "peaks": p, "store": store, "table": table, "rows": store.rows_of(barcodes)}
    store.close()


def _tfidf(p: sp.csr_matrix) -> np.ndarray:
    x = p.toarray().astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.nan_to_num(np.log1p(x / x.sum(axis=1, keepdims=True) * 1e4 * (x.shape[0] / x.sum(axis=0, keepdims=True))))


def _da(world, column: str = "group", **kw) -> dict:
    codes, every, chosen = atac.group_codes(world["adata"], column, kw.pop("groups", None))
    out = None
    for item in atac.iter_differential(world["store"], world["rows"], codes, every, chosen, test="lr", **kw):
        if item["type"] == "result":
            out = item
    return out


def test_lr_finds_each_groups_peaks_on_the_data_slot_with_the_cells_total(world):
    out = _da(world)
    f, t = out["frame"], world["table"]
    top_a = f[(f["group"] == "A") & (f["z"] > 0)].sort_values("pval").head(6)
    assert set(t.iloc[top_a["peak_index"].to_numpy()]["gene"]) == {"GENEA", "GENEB"}
    assert (f["log2fc"][f["z"] > 3] > 0).all()
    assert {"fit", "pval", "pval_adj", "pct_in"} <= set(f.columns)
    assert sum(out["n_fit"]["A"].values()) == t.shape[0]
    # One peak by hand: every labelled cell, x = log1p(TF-IDF), latent = the cell's total over all peaks.
    row = f[(f["group"] == "B") & (f["fit"] == "lrt")].iloc[0]
    x = _tfidf(world["peaks"])[:, int(row["peak_index"])]
    y = (world["adata"].obs["group"] == "B").to_numpy(dtype=float)
    total = np.asarray(world["peaks"].sum(axis=1), dtype=np.float64).ravel()
    b, dev, _ok = atac_lr.logistic_null(y, _z(total))
    stat, _g, _s = atac_lr.lrt_fit(x[:, None], y, _z(total), b, dev)
    assert row["pval"] == pytest.approx(chi2.sf(stat[0], 1), rel=1e-6)


def test_blocks_batches_and_the_mix_of_kinds_of_p(world):
    one = _da(world)
    many = _da(world, batch_values=50)
    pd.testing.assert_frame_equal(one["frame"].reset_index(drop=True), many["frame"].reset_index(drop=True))
    few = _da(world, lrt_per_group=5)
    counts = few["n_fit"]["A"]
    assert counts["lrt"] + counts["lrt_separated"] == 5 and counts["score"] == few["n_tested"]["A"] - 5
    f = few["frame"]
    assert set(f[f["group"] == "A"]["fit"]) <= {"lrt", "lrt_separated", "score"}
    # BH runs across both kinds: the largest adjusted p belongs to the largest p of either.
    a = f[f["group"] == "A"]
    assert a["pval_adj"].iloc[-1] == pytest.approx(a["pval"].iloc[-1])


def test_unlabelled_cells_are_in_no_comparison(world):
    a = world["adata"]
    a.obs["some"] = a.obs["group"].astype(object)
    a.obs.iloc[:10, a.obs.columns.get_loc("some")] = None
    a.obs.iloc[-10:, a.obs.columns.get_loc("some")] = None
    out = _da(world, "some", groups=["A"])
    row = out["frame"][out["frame"]["fit"] == "lrt"].iloc[0]
    keep = a.obs["some"].notna().to_numpy()
    x = _tfidf(world["peaks"])[keep, int(row["peak_index"])]
    y = (a.obs["some"][keep] == "A").to_numpy(dtype=float)
    total = np.asarray(world["peaks"].sum(axis=1), dtype=np.float64).ravel()[keep]
    b, dev, _ok = atac_lr.logistic_null(y, _z(total))
    stat, _g, _s = atac_lr.lrt_fit(x[:, None], y, _z(total), b, dev)
    assert row["pval"] == pytest.approx(chi2.sf(stat[0], 1), rel=1e-6)


def test_fits_that_do_not_settle_keep_their_score_test_and_are_counted(world, monkeypatch):
    real = atac_lr.lrt_fit

    def short(x, y, z, b, dev):
        return real(x, y, z, b, dev, max_iter=1)

    monkeypatch.setattr(atac, "lrt_fit", short)
    out = _da(world)
    counts = out["n_fit"]["A"]
    assert counts["score_no_fit"] > 0
    f = out["frame"]
    hit = f[f["fit"] == "score_no_fit"]
    assert len(hit) == sum(c["score_no_fit"] for c in out["n_fit"].values())
    assert hit["pval"].between(0, 1).all() and (hit["pval"] < 1).any()


def test_a_group_too_small_to_fit_is_counted_not_given_a_p(world):
    a = world["adata"]
    a.obs["tiny"] = pd.Categorical(["one", "one"] + ["big"] * (a.n_obs - 2))
    out = _da(world, "tiny")
    assert out["n_tested"]["one"] == 0 and out["n_significant"]["one"] == 0
    assert out["n_fit"]["one"]["untestable"] > 0 and out["n_tested"]["big"] > 0
    assert (out["frame"]["group"] == "big").all() and out["frame"]["pval"].notna().all()


def test_every_cell_with_one_total_is_refused_in_words(world):
    p = sp.csr_matrix(np.ones((world["rows"].size, 4), dtype=np.float32))
    path = world["dir"] / "same.h5"
    modality_store.write(path, {atac.MATRIX: p}, list(world["adata"].obs_names), [f"chr1:{i}-{i + 1}" for i in range(4)])
    codes, every, chosen = atac.group_codes(world["adata"], "group")
    with modality_store.ModalityStore(path) as store, pytest.raises(atac.AtacError, match="same total"):
        list(atac.iter_differential(store, store.rows_of(world["adata"].obs_names), codes, every, chosen, test="lr"))


def test_stop_ends_the_run_at_the_next_block(world):
    codes, every, chosen = atac.group_codes(world["adata"], "group")
    gen = atac.iter_differential(world["store"], world["rows"], codes, every, chosen, test="lr", stop=lambda: True)
    next(gen)
    with pytest.raises(atac.AtacCancelled):
        next(gen)


def test_wilcoxon_is_what_it_was_when_no_test_is_named(world):
    codes, every, chosen = atac.group_codes(world["adata"], "group")
    out = [i for i in atac.iter_differential(world["store"], world["rows"], codes, every, chosen) if i["type"] == "result"][0]
    assert "fit" not in out["frame"].columns and "n_fit" not in out


@pytest.mark.parametrize("nnz, groups, cells, measured", [
    (104_012_773, 21, 11_898, (240.7, 311.0)),  # PBMC 10k: this laptop 2026-11-06, the b36 agent's run
    (97_980_766, 24, 11_208, (281.0,)),  # PBMC 10k after RNA QC, live through the UI (b36)
    (41_686_530, 21, 4_878, (62.9, 80.0)),  # E18 mouse brain: this laptop, the b36 agent's run
])
def test_the_lr_estimate_is_within_20_percent_of_the_measured_runs(nnz, groups, cells, measured):
    # All 10,000 likelihood ratios per group; the cost per cell of a fit falls with fewer cells.
    est = atac.estimate_da_seconds(nnz, groups, "lr", n_cells=cells)
    for secs in measured:
        assert abs(est / secs - 1) <= 0.2, (est, secs)


def test_estimates_name_the_test_and_leave_wilcoxon_alone():
    assert atac.estimate_da_seconds(104_000_000, 3) == round(5.0 + 104e6 * (3.5e-7 + 3 * 1.4e-8), 0)
    lr = atac.estimate_da_seconds(104_000_000, 24, "lr", n_cells=11_898)
    assert lr > atac.estimate_da_seconds(104_000_000, 24)
    assert atac.estimate_da_seconds(104_000_000, 24, "lr", n_cells=11_898, n_fits=100) < lr
    base =atac.estimate_da_bytes(143_887, 24)
    assert base == atac.estimate_da_bytes(143_887, 24, "wilcoxon", n_cells=11_898)
    assert atac.estimate_da_bytes(143_887, 24, "lr", n_cells=11_898) > 0


def test_a_result_saved_before_the_choice_reads_as_wilcoxon():
    table = pd.DataFrame({"group": ["A"], "peak_index": [0], "peak": ["chr1:1-2"], "pval": [0.1]})
    old = atac.DaResult.__new__(atac.DaResult)
    for key, value in dict(result_id="x", column="c", groups=["A"], table=table, n_tested={"A": 1}, n_significant={"A": 0},
                           params={"min_pct": 0.05}, method="Wilcoxon.", cells="", created_at="").items():
        object.__setattr__(old, key, value)
    payload = atac.da_payload(old)
    assert payload["test"] == "wilcoxon" and payload["n_fit"] is None


def test_results_store_round_trip_keeps_the_fit_counts(tmp_path, monkeypatch):
    table = pd.DataFrame({"group": ["A"], "peak_index": [0], "peak": ["chr1:1-2"], "pval": [0.1], "fit": ["lrt"]})
    res = atac.DaResult("rid", "c", ["A"], table, {"A": 1}, {"A": 0}, {"test": "lr"}, "m", "", "", {"A": {"lrt": 1}})
    enc = results_store._Parts()
    back = results_store._decode(results_store._encode(res, enc), enc.tables, enc.arrays)
    assert atac.da_payload(back)["n_fit"] == {"A": {"lrt": 1}} and atac.da_test(back) == "lr"


def test_copilot_fact_names_the_test_and_its_counts(monkeypatch):
    table = pd.DataFrame({"group": ["A"], "peak_index": [0]})
    res = atac.DaResult("rid", "atac_leiden", ["A"], table, {"A": 500}, {"A": 20}, {"test": "lr"}, "m", "", "",
                        {"A": {"lrt_separated": 3, "score_no_fit": 2, "untestable": 1}})
    monkeypatch.setattr(atac, "latest_da", lambda _a: res)
    text = atac.da_grounding_lines(object())[0]
    assert "logistic regression" in text and "covariate" in text and "20 of 500" in text and "[result:atac]" in text
    assert "3 fits separated" in text and "3 did not converge" in text
    res.params = {}
    res.n_fit = None
    assert "Wilcoxon" in atac.da_grounding_lines(object())[0]
    monkeypatch.setattr(atac, "latest_da", lambda _a: None)
    assert atac.da_grounding_lines(object()) == []


def test_the_router_keeps_the_fact_on_its_words():
    from scview.core.rag import router

    words = {w for words in router._RESULT_FACET_HINTS.values() for w in words}
    assert "differential peak" in words or any("differential peak" in w for w in words)


def test_the_replay_script_reproduces_lr_and_wilcoxon_from_the_input(tmp_path):
    from scview.core import replay_script
    from scview.core.ingestion.session import IngestSessionManager

    gtf = tmp_path / "tiny.gtf"
    gtf.write_text(mf.gtf_text())
    gene_annotation.clear_cache()
    h5 = mf.write_tenx_h5(tmp_path / "pbmc_filtered_feature_bc_matrix.h5")
    m = IngestSessionManager(str(tmp_path / "data"))
    sid = m.create_session()
    (m.files_dir(sid) / h5.name).write_bytes(h5.read_bytes())
    did = m._build_dataset(sid)
    gene_annotation.seed_from(gtf, m.data_dir, "GRCh38")
    a = ad.read_h5ad(next((m.data_dir / "ingested" / did).glob("*.h5ad")))
    # The grouping comes from the input in a replay: a column the run does not make would be absent, so seed it first.
    p = atac.plan(a, m.data_dir, did, n_lsi=8, dims_to=6, resolution=0.7)
    with atac.open_store(a, m.data_dir, did) as store:
        rows = atac.cell_rows(store, a)
        genes = atac.choose_genes(a, p.genome, m.data_dir)
        run = [i for i in atac.iter_run(store, rows, p, data_dir=m.data_dir, dataset_id=did, genes=genes)
               if i["type"] == "result"][0]["result"]
        atac.save_run(a, p, run)
        codes, every, chosen = atac.group_codes(a, atac.OBS_CLUSTERS)
        mine = {}
        for test in ("wilcoxon", "lr"):
            mine[test] = [i for i in atac.iter_differential(store, rows, codes, every, chosen, test=test)
                          if i["type"] == "result"][0]["frame"]
            atac.record_differential(a, {"column": atac.OBS_CLUSTERS, "groups": None, "min_pct": atac.MIN_PCT,
                                         "max_peaks": atac.MAX_PEAKS_PER_GROUP, "test": test,
                                         **({"lrt_peaks": atac.LRT_PER_GROUP} if test == "lr" else {})})
    assert [d["status"] for h, d in zip(atac.provenance_history(a), replay_script.describe(a))
            if h["step"] == "atac_differential"] == ["exact", "exact"]
    src = tmp_path / "input" / h5.name
    src.parent.mkdir()
    src.write_bytes(h5.read_bytes())
    with gzip.open(src.parent / "gencode.v32.basic.annotation.gtf.gz", "wt") as fh:
        fh.write(mf.gtf_text())
    script, dst = tmp_path / "replay.py", tmp_path / "replayed.h5ad"
    script.write_text(replay_script.build_script(a, dataset_id="t", dataset_name="test"))
    proc = subprocess.run([sys.executable, str(script), str(src), str(dst)], capture_output=True, text=True, timeout=600,
                          cwd=tmp_path)
    assert proc.returncode == 0, proc.stderr[-3000:]
    b = ad.read_h5ad(dst)
    with atac.open_store(a, m.data_dir, did) as store:
        names = store.features.astype(str)
    for test, frame in mine.items():
        got = b.uns[f"atac_differential_{test}"]
        assert got["peak"].astype(str).tolist() == names[frame["peak_index"].to_numpy()].tolist()
        np.testing.assert_allclose(got["pval"].to_numpy(dtype=float), frame["pval"].to_numpy(), rtol=1e-9, atol=1e-300)
    assert b.uns["atac_differential_lr"]["fit"].astype(str).tolist() == mine["lr"]["fit"].tolist()
    gene_annotation.clear_cache()


def test_a_step_without_settings_is_not_replayed():
    from scview.core import replay_script

    a = ad.AnnData(X=np.zeros((2, 2), dtype=np.float32))
    provenance.record_step(a, step="atac_differential", tool="t", params={})
    assert replay_script.describe(a)[-1]["status"] == "not_replayable"


def test_a_differential_step_leaves_earlier_results_current():
    assert "atac_differential" in results_store.ADDITIVE_STEPS


def test_module_file_has_no_scview_imports_in_the_replayed_code():
    src = Path(atac_lr.__file__).read_text()
    assert "scview" not in src.split('"""', 2)[2].split("REPLAY_CONSTANTS")[0]


def test_a_small_p_stays_in_the_payload():
    table = pd.DataFrame({"group": ["A", "A"], "peak_index": [0, 1], "peak": ["chr1:1-2", "chr1:3-4"], "pval": [3e-12, 1e-16]})
    res = atac.DaResult("rid", "c", ["A"], table, {"A": 2}, {"A": 2}, {"test": "lr"}, "m", "", "", {"A": {"lrt": 2}})
    assert [r["pval"] for r in atac.da_payload(res)["rows"]["A"]] == [3e-12, 1e-16]
    assert atac.da_payload(res, "A")["rows"]["A"][0]["pval"] == 3e-12
