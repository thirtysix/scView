"""The scDblFinder-style doublet classifier (core/doublet_classifier.py) and the doublet
step's choice of detectors (core/doublets.py ``run_doublets``).

Synthetic counts with known doublets: clusters with their own marker genes, and planted
doublets made by summing two cells of different clusters (what a heterotypic doublet is).
The classifier must rank them above the singlets and call most of them with few false
calls; per channel as Scrublet; the step writes the columns of the detectors it ran.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score

from scview.core import doublet_classifier as dc
from scview.core import doublet_source, doublets, provenance
from scview.core import pipeline as pl


def clustered_counts(n_per: int = 260, k: int = 6, n_genes: int = 900, n_doublets: int = 60,
                     seed: int = 0) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
    """(counts, cluster of each singlet (-1: planted doublet), truth) with the planted
    doublets last: each the sum of two cells of different clusters."""
    rng = np.random.default_rng(seed)
    base = rng.gamma(0.6, 1.0, n_genes)
    profiles = []
    for _ in range(k):
        up = rng.random(n_genes) < 0.08  # each cluster's own markers
        profiles.append(base * np.where(up, rng.uniform(4, 12, n_genes), 1.0))
    labels = np.repeat(np.arange(k), n_per)
    lib = rng.lognormal(7.4, 0.25, labels.size)
    P = np.vstack([profiles[c] / profiles[c].sum() for c in labels])
    X = rng.poisson(P * lib[:, None]).astype(np.float32)
    a = rng.integers(0, labels.size, n_doublets)
    b = rng.integers(0, labels.size, n_doublets)
    for _ in range(50):
        same = labels[a] == labels[b]
        b[same] = rng.integers(0, labels.size, int(same.sum()))
    D = X[a] + X[b]
    counts = sp.csr_matrix(np.vstack([X, D]))
    truth = np.r_[np.zeros(labels.size, bool), np.ones(n_doublets, bool)]
    return counts, np.r_[labels, -np.ones(n_doublets, int)], truth


@pytest.fixture(scope="module")
def planted():
    counts, labels, truth = clustered_counts()
    score, call, info = dc.classify_counts(counts, expected_doublet_rate=truth.mean())
    return counts, truth, score, call, info


def test_planted_doublets_rank_high_and_most_are_called_with_few_false_calls(planted):
    counts, truth, score, call, info = planted
    assert np.isfinite(score).all() and ((score >= 0) & (score <= 1)).all()
    assert roc_auc_score(truth, score) > 0.97
    assert (call & truth).sum() >= 0.8 * truth.sum()  # recall
    assert (call & ~truth).sum() <= 0.02 * (~truth).sum()  # false calls among singlets
    assert call.tolist() == (score > info["threshold"]).tolist()
    assert info["n_artificial"] == counts.shape[0]  # as many as cells (over the floor)
    assert info["n_genes"] <= 900  # fewer genes than N_FEATURES: all seen in 3 cells


def test_same_input_same_scores_and_dense_reads_as_sparse(planted):
    counts, truth, score, call, _ = planted
    again, call2, _ = dc.classify_counts(counts, expected_doublet_rate=truth.mean())
    assert np.array_equal(score, again) and np.array_equal(call, call2)
    dense, _, _ = dc.classify_counts(counts.toarray(), expected_doublet_rate=truth.mean())
    assert np.array_equal(score, dense)


def test_threshold_balances_artificial_doublets_and_the_expected_rate():
    rng = np.random.default_rng(0)
    real = np.r_[rng.uniform(0, 0.2, 950), rng.uniform(0.8, 1.0, 50)]  # 5 % look like doublets
    art = rng.uniform(0.7, 1.0, 1000)
    t = dc.threshold(real, art, rate=0.05)
    assert real[:950].max() <= t < 0.8 and (real > t).sum() == 50
    # Without artificial doublets to separate, the expected rate sets the share called
    flat = np.linspace(0, 1, 1000)
    t = dc.threshold(flat, np.array([1.0]), rate=0.05)
    assert 25 <= (flat > t).sum() <= 75  # 5 % give or take its tolerance (40 % of it)


def test_per_channel_runs_alone_and_small_channels_are_not_scored():
    counts, _, truth = clustered_counts(n_per=150, n_doublets=30, seed=1)
    n = counts.shape[0]
    codes = np.zeros(n, dtype=np.int64)
    codes[-40:] = 1  # a channel of 40 cells: under the floor
    codes[:5] = -1  # no channel
    score, call, runs = dc.classify_by_channel(counts, codes, [0.03, 0.03])
    assert [r["status"] for r in runs] == ["scored", "too_small"]
    assert np.isnan(score[-40:]).all() and not call[-40:].any()
    assert np.isnan(score[:5]).all() and not call[:5].any()
    assert runs[0]["n_cells"] == n - 45 and runs[0]["n_scored"] == n - 45
    assert runs[0]["n_called"] == int(call.sum())
    alone, _, _ = dc.classify_counts(counts[5:n - 40], 0.03)
    assert np.array_equal(score[5:n - 40], alone)  # a channel's scores are its own


def test_cells_with_under_three_genes_are_not_scored():
    counts, _, _ = clustered_counts(n_per=120, n_doublets=20, seed=2)
    X = counts.tolil()
    X[0, :] = 0
    X[0, 0] = 5
    score, call, _ = dc.classify_counts(X.tocsr(), 0.03)
    assert np.isnan(score[0]) and not call[0] and np.isfinite(score[1:]).all()


def test_too_few_cells_is_an_error_the_channel_records_as_failed():
    counts, _, _ = clustered_counts(n_per=20, k=3, n_doublets=5, seed=3)
    with pytest.raises(ValueError):
        dc.classify_counts(counts, 0.03)
    _, _, runs = dc.classify_by_channel(counts, np.zeros(counts.shape[0], dtype=np.int64), [0.03],
                                        min_cells=10)
    assert runs[0]["status"] == "failed" and runs[0]["error"]


# --- the doublet step's detectors ---------------------------------------------------------


def _adata(seed: int = 0) -> tuple[ad.AnnData, np.ndarray]:
    counts, _, truth = clustered_counts(n_per=150, n_doublets=40, seed=seed)
    a = ad.AnnData(counts)
    a.var_names = [f"G{i}" for i in range(a.n_vars)]
    a.obs_names = [f"C{i}" for i in range(a.n_obs)]
    a.obs["channel"] = pd.Categorical(np.where(np.arange(a.n_obs) % 2 == 0, "L1", "L2"))
    return a, truth


def test_both_detectors_write_their_own_columns_per_channel():
    a, truth = _adata()
    found = doublets.run_doublets(a, a.X, "both")
    obs = a.obs
    for col in ("doublet_score", "predicted_doublet", "doublet_score_classifier",
                "predicted_doublet_classifier"):
        assert col in obs.columns
    assert found["doublet_method"] == "both" and found["channel_column"] == "channel"
    rows = found["classifier"]["channels"]
    assert [r["name"] for r in rows] == ["L1", "L2"] and all(r["status"] == "scored" for r in rows)
    assert sum(r["n_called"] for r in rows) == int(obs["predicted_doublet_classifier"].sum())
    assert a.uns[doublets.CLASSIFIER_KEY]["scview_channel_column"] == "channel"
    # auto = both detectors' consensus
    used = doublet_source.resolve(obs)
    assert used.source == "consensus"
    assert used.n == int((obs["predicted_doublet"] & obs["predicted_doublet_classifier"]).sum())
    assert roc_auc_score(truth, obs["doublet_score_classifier"]) > 0.95


def test_one_detector_drops_the_others_results():
    a, _ = _adata(seed=4)
    doublets.run_doublets(a, a.X, "both")
    doublets.run_doublets(a, a.X, "scrublet")
    assert "predicted_doublet_classifier" not in a.obs.columns
    assert doublets.CLASSIFIER_KEY not in a.uns
    assert doublet_source.resolve(a.obs).source == "scrublet"
    found = doublets.run_doublets(a, a.X, "classifier")  # an alias
    assert found["doublet_method"] == "scdblfinder-style"
    assert "predicted_doublet" not in a.obs.columns and "scrublet" not in a.uns
    assert doublet_source.resolve(a.obs).source == "classifier"
    assert found["channel_column"] == "channel" and found["expected_doublet_rate"] is None
    with pytest.raises(doublets.DoubletError):
        doublets.run_doublets(a, a.X, "doubletdetection")


def test_summary_gives_the_classifier_per_channel_and_the_agreement_by_hand():
    a, _ = _adata(seed=5)
    doublets.run_doublets(a, a.X, "both")
    s, c = a.obs["predicted_doublet"].to_numpy(), a.obs["predicted_doublet_classifier"].to_numpy()
    out = doublets.summary(a)
    cls = out["classifier"]
    assert cls["n_called"] == int(c.sum()) and cls["n_scored"] == a.n_obs
    assert sum(cls["hist"]["counts"]) == a.n_obs
    assert [r["name"] for r in cls["channels"]] == ["L1", "L2"]
    assert all(r["threshold"] is not None for r in cls["channels"])
    m = out["methods"]
    assert (m["both"], m["scrublet_only"], m["classifier_only"]) == (
        int((s & c).sum()), int((s & ~c).sum()), int((~s & c).sum()))
    l1 = (a.obs["channel"] == "L1").to_numpy()
    row = m["channels"][0]
    assert row["name"] == "L1" and row["both"] == int((s & c & l1).sum())
    assert row["classifier_only"] == int((~s & c & l1).sum())
    # The classifier alone: no Scrublet histogram, its own block still there
    doublets.run_doublets(a, a.X, "scdblfinder-style")
    alone = doublets.summary(a)
    assert alone["available"] is False and alone["classifier"]["n_called"] == int(
        a.obs["predicted_doublet_classifier"].sum())
    assert alone["source"]["source"] == "classifier"


async def test_the_doublet_card_endpoint_carries_the_classifier_and_a_threshold_leaves_it(
        client, dataset_manager):
    import json
    from pathlib import Path

    a, _ = _adata(seed=8)
    doublets.run_doublets(a, a.X, "both")
    cls = a.obs["predicted_doublet_classifier"].to_numpy().copy()
    d = Path(dataset_manager.data_dir) / "uploads" / "dcls"
    d.mkdir(parents=True)
    (d / "metadata.json").write_text(json.dumps({"id": "dcls", "name": "dcls", "status": "ready"}))
    a.write_h5ad(d / "data.h5ad")
    r = await client.get("/api/v1/datasets/dcls/qc/doublets")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["classifier"]["n_called"] == int(cls.sum())
    assert [row["name"] for row in body["methods"]["channels"]] == ["L1", "L2"]
    assert body["source"]["source"] == "consensus"
    # A threshold on Scrublet's scores moves Scrublet's calls only
    r = await client.post("/api/v1/datasets/dcls/qc/doublets/threshold", json={"threshold": 0.01})
    assert r.status_code == 200, r.text
    after = r.json()
    assert after["classifier"]["n_called"] == int(cls.sum())
    assert after["methods"]["scrublet"] == after["n_called"]


def test_pipeline_step_records_the_method_and_the_classifier_channels():
    a, _ = _adata(seed=6)
    params = pl.PipelineParams(doublet_method="both", doublet_channel_key="channel")
    pl._record_pipeline_step(a, "doublet_detection", params, pl._run_doublet_detection(a, params))
    entry = provenance.read_provenance(a)["history"][-1]
    p = entry["params"]
    assert p["doublet_method"] == "both" and "scDblFinder-style" in entry["tool"]
    assert [c["name"] for c in p["classifier"]["channels"]] == ["L1", "L2"]
    assert [c["name"] for c in p["channels"]] == ["L1", "L2"]  # Scrublet's, as before


def test_large_dataset_mode_scores_each_channel_as_in_memory(tmp_path, monkeypatch):
    """The matrices stay on disk: each channel's rows are read, scored and let go."""
    from scview.core import chunked
    from tests.test_large_run import ROWS, _both, _counts, _write

    monkeypatch.setattr(chunked, "rows_per_block", lambda *a, **k: ROWS)
    src = _write(_counts(n=1600), tmp_path / "in" / "d.h5ad")
    steps = ["qc_metrics", "doublet_detection", "normalization", "log_transform"]
    big, res, mem, mres = _both(src, tmp_path / "out" / "d.h5ad", steps, doublet_method="both")
    assert res.chunked and res.errors == mres.errors == {}
    for c in ("doublet_score_classifier", "predicted_doublet_classifier", "doublet_score"):
        np.testing.assert_array_equal(big.obs[c].to_numpy(), mem.obs[c].to_numpy())
    assert doublets.classifier_record(big)["rows"] == doublets.classifier_record(mem)["rows"]
    assert [r["name"] for r in doublets.classifier_record(big)["rows"]] == ["L1", "L2"]


# --- the preflight estimate ---------------------------------------------------------------

# Peak RSS of classify_counts over the process before it, MiB, 2 CPUs (scview-backend 0.34.1):
# (cells, genes, stored values, measured)
CLASSIFIER_MEASURED = {
    "Kang CTRL lane": (6573, 14053, 4696934, 306),
    "Kang": (13836, 14053, 9655137, 619),
    "hgmm_10k_v3": (11456, 112137, 54817322, 1380),
    "brain 10k": (10000, 27998, 19566113, 764),
    "brain 50k": (50000, 27998, 102138811, 3277),
}


@pytest.mark.parametrize("name", list(CLASSIFIER_MEASURED))
def test_preflight_classifier_memory_within_the_band(name):
    from scview.core import preflight as pf

    n, m, nnz, measured = CLASSIFIER_MEASURED[name]
    params = pl.PipelineParams(doublet_method="scdblfinder-style")
    t = pf.estimate_step_peak("doublet_detection", (n, m), nnz, False, params,
                              doublets=pf.DoubletPlan([n]))
    assert abs((t - pf.DBL_OUT_PER_CELL * n) / pf.MiB / measured - 1) <= pf.ERROR_BAND


def test_preflight_both_detectors_take_the_larger_peak_and_both_times():
    from scview.core import preflight as pf

    n, m, nnz = 13836, 14053, 9655137
    plan = pf.DoubletPlan([6573, 7263], 1450)

    def est(method):
        state = pf._State(n, m, pf.Matrix("X", n, m, True, nnz), {}, None, 0)
        t, _, secs = pf._step_doublet(state, pl.PipelineParams(doublet_method=method), plan)
        return t, secs

    (ts, ss), (tc, sc_), (tb, sb) = est("scrublet"), est("scdblfinder-style"), est("both")
    assert tb == max(ts, tc) and sb == pytest.approx(ss + sc_)
    assert 6.2 / 1.25 <= sc_ - 1.0 <= 6.2 * 1.25  # Kang's two lanes: 3.3 + 3.7 s measured


# --- the replay script ------------------------------------------------------------------


def _barcoded(a: ad.AnnData) -> ad.AnnData:
    """Seurat-style merged 10x barcodes: the lane in the prefix (core/doublets.py)."""
    rng = np.random.default_rng(0)
    codes = ["".join(rng.choice(list("ACGT"), 16)) for _ in range(a.n_obs)]
    a.obs_names = [f"{'A' if i % 2 else 'B'}_{c}-1" for i, c in enumerate(codes)]
    del a.obs["channel"]
    return a


@pytest.mark.parametrize("method", ["both", "scdblfinder-style"])
def test_the_replay_script_scores_what_the_app_scored(tmp_path, method):
    import subprocess
    import sys

    from scview.core import replay_script

    a, _ = _adata(seed=7)
    if method != "both":
        a = _barcoded(a)
    inp = a.copy()
    out, res = pl.run_pipeline(a, ["doublet_detection"], pl.PipelineParams(doublet_method=method))
    assert res.errors == {}
    assert replay_script.describe(out)[-1]["status"] == "exact"
    src, script, dst = tmp_path / "in.h5ad", tmp_path / "replay.py", tmp_path / "out.h5ad"
    inp.write_h5ad(src)
    script.write_text(replay_script.build_script(out, dataset_id="t", dataset_name="t"))
    proc = subprocess.run([sys.executable, str(script), str(src), str(dst)], capture_output=True,
                          text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr[-3000:]
    rep = ad.read_h5ad(dst)
    cols = ["doublet_score_classifier", "predicted_doublet_classifier"]
    cols += ["doublet_score", "predicted_doublet"] if method == "both" else []
    for c in cols:
        assert np.array_equal(rep.obs[c].to_numpy(), out.obs[c].to_numpy(), equal_nan=True), c
    assert doublets.classifier_record(rep)["rows"] == doublets.classifier_record(out)["rows"]
    if method != "both":
        assert "doublet_score" not in rep.obs.columns
        assert (rep.obs["capture_channel"].astype(str) == out.obs["capture_channel"].astype(str)).all()
