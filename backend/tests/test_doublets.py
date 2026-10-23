"""Doublet calls (core/doublets.py, GET /qc/doublets, POST /qc/doublets/threshold).

The expected rate suggested from the cells per channel, the score histogram and
the counts a threshold calls, applying a threshold (Scrublet's call kept and
recoverable, the filtering step dropping the new calls), and the doublet step
taking each channel's suggestion when no rate is given. The channels themselves
(columns, barcodes, the per-channel runs): tests/test_doublet_channels.py.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from scview.core import doublets, provenance
from scview.core import pipeline as pl

API = "/api/v1"


def _sized(sizes: dict[str, int], column: str = "sample") -> ad.AnnData:
    """Obs only: one group per sample with the given number of cells."""
    labels = np.concatenate([[k] * n for k, n in sizes.items()])
    a = ad.AnnData(np.zeros((labels.size, 3), dtype="float32"))
    a.obs_names = [f"c{i}" for i in range(labels.size)]
    a.obs[column] = pd.Categorical(labels, categories=list(sizes))
    return a


def _scored(n: int = 400, seed: int = 0, threshold: float = 0.3) -> ad.AnnData:
    """Two samples, Scrublet-like scores, its threshold in uns and its calls."""
    rng = np.random.default_rng(seed)
    a = _sized({"s1": n // 4, "s2": n - n // 4})
    scores = np.clip(rng.beta(1.2, 12, n), 0, 1)
    scores[:20] = rng.uniform(0.35, 0.8, 20)  # a doublet-like tail, all in s1
    a.obs["doublet_score"] = scores
    a.obs["predicted_doublet"] = scores > threshold
    a.uns["scrublet"] = {"threshold": threshold, "parameters": {"expected_doublet_rate": 0.05}}
    return a


def _counts(n_cells: int = 600, n_genes: int = 400, seed: int = 0) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    a = ad.AnnData(rng.poisson(1.0, size=(n_cells, n_genes)).astype("float32"))
    a.var_names = [f"G{i}" for i in range(n_genes)]
    a.obs_names = [f"C{i}" for i in range(n_cells)]
    a.obs["sample"] = pd.Categorical(np.repeat(["s1", "s2"], n_cells // 2))
    return a


# --- the expected rate -----------------------------------------------------------


def test_rate_is_0_8_percent_per_1000_cells_within_bounds():
    assert doublets.rate_for_cells(1000) == pytest.approx(0.008)
    assert doublets.rate_for_cells(5000) == pytest.approx(0.04)
    assert doublets.rate_for_cells(100) == doublets.MIN_RATE
    assert doublets.rate_for_cells(1_000_000) == doublets.MAX_RATE


def test_without_a_channel_the_samples_are_the_channels_with_their_own_rates():
    s = doublets.suggest_rate(_sized({"small": 2000, "big": 6000}))
    assert (s["column"], s["reason"], s["per_channel"]) == ("sample", "donor", True)
    assert s["note"] == doublets.GUESS_NOTE  # samples pooled in a lane are not channels
    assert [(x["name"], x["n_cells"], x["rate"]) for x in s["channels"]] == [
        ("small", 2000, 0.016), ("big", 6000, 0.048)]
    # the share of all cells expected to be doublets: (2000 x 1.6 % + 6000 x 4.8 %) / 8000
    assert s["rate"] == pytest.approx(0.04)
    assert "rule of thumb" in s["basis"].lower()


def test_without_a_sample_column_all_cells_are_one_channel():
    a = _sized({"x": 8000}, column="label")  # one value, not a sample column
    s = doublets.suggest_rate(a)
    assert s["column"] is None and s["channels"] == [] and s["reason"] == "none"
    assert s["rate"] == pytest.approx(0.064)
    assert "one channel" in s["basis"]


def test_a_named_column_is_used_and_a_missing_one_refused():
    a = _sized({"a": 1000, "b": 1000})
    a.obs["lane"] = pd.Categorical(["L1"] * 2000)
    assert doublets.suggest_rate(a, "lane")["rate"] == pytest.approx(0.016)
    with pytest.raises(doublets.DoubletError, match="not found"):
        doublets.suggest_rate(a, "nope")


# --- the summary -----------------------------------------------------------------


def test_without_scores_only_the_suggestion_is_given():
    out = doublets.summary(_sized({"a": 500, "b": 500}))
    assert out["available"] is False and out["suggestion"]["rate"] == pytest.approx(0.004)


def test_summary_counts_cells_strictly_above_the_threshold_per_sample():
    a = _scored()
    scores = a.obs["doublet_score"].to_numpy()
    out = doublets.summary(a)
    assert out["threshold"] == out["scrublet_threshold"] == pytest.approx(0.3)
    assert out["n_called"] == int((scores > 0.3).sum()) == out["n_called_now"]
    assert sum(out["hist"]["counts"]) == out["n_scored"] == a.n_obs
    by = {s["name"]: s for s in out["channels"]}
    assert by["s1"]["n_called"] + by["s2"]["n_called"] == out["n_called"]
    assert by["s1"]["n_called"] >= 20  # the planted tail
    assert by["s1"]["expected_rate"] == pytest.approx(doublets.MIN_RATE)

    lower = doublets.summary(a, threshold=0.1)
    assert lower["threshold"] == 0.1 and lower["n_called"] == int((scores > 0.1).sum())
    with pytest.raises(doublets.DoubletError):
        doublets.summary(a, threshold=-1)


def test_no_scrublet_threshold_means_no_count_until_one_is_given():
    a = _scored()
    del a.uns["scrublet"]
    out = doublets.summary(a)
    assert out["threshold"] is None and out["n_called"] is None
    # each channel's row counts the calls there are (Scrublet's saved call)
    assert sum(r["n_called"] for r in out["channels"]) == out["n_called_now"]
    assert doublets.summary(a, threshold=0.5)["n_called"] > 0


# --- applying a threshold --------------------------------------------------------


def test_applying_keeps_scrublets_call_and_putting_it_back_restores_it():
    a = _scored()
    original = a.obs["predicted_doublet"].to_numpy().copy()
    scores = a.obs["doublet_score"].to_numpy()

    out = doublets.apply_threshold(a, 0.15)
    assert out["applied"]["threshold"] == 0.15 and out["threshold"] == 0.15
    np.testing.assert_array_equal(a.obs["predicted_doublet"].to_numpy(), scores > 0.15)
    np.testing.assert_array_equal(a.obs["predicted_doublet_scrublet"].to_numpy(), original)

    # A second threshold leaves Scrublet's call as it was, not the first threshold's.
    doublets.apply_threshold(a, 0.5)
    np.testing.assert_array_equal(a.obs["predicted_doublet"].to_numpy(), scores > 0.5)
    np.testing.assert_array_equal(a.obs["predicted_doublet_scrublet"].to_numpy(), original)

    out = doublets.apply_threshold(a, None)
    np.testing.assert_array_equal(a.obs["predicted_doublet"].to_numpy(), original)
    assert "predicted_doublet_scrublet" not in a.obs.columns
    assert out["applied"] is None and doublets.THRESHOLD_KEY not in a.uns
    steps = [h for h in provenance.read_provenance(a)["history"] if h["step"] == "doublet_threshold"]
    assert [h["params"]["threshold"] for h in steps] == [0.15, 0.5, "scrublet"]


def test_a_call_made_from_scores_alone_is_removed_when_put_back():
    a = _scored()
    del a.obs["predicted_doublet"]  # an upload with scores but no calls
    doublets.apply_threshold(a, 0.3)
    assert "predicted_doublet" in a.obs.columns
    assert "predicted_doublet_scrublet" not in a.obs.columns
    doublets.apply_threshold(a, None)
    assert "predicted_doublet" not in a.obs.columns


def test_bad_requests_change_nothing():
    a = _scored()
    before = a.obs.copy()
    for bad in (-0.1, float("nan"), float("inf")):
        with pytest.raises(doublets.DoubletError):
            doublets.apply_threshold(a, bad)
    with pytest.raises(doublets.DoubletError, match="not found"):
        doublets.apply_threshold(a, 0.2, column="nope")
    pd.testing.assert_frame_equal(a.obs, before)
    assert doublets.THRESHOLD_KEY not in a.uns
    with pytest.raises(doublets.DoubletError, match="run doublet detection"):
        doublets.apply_threshold(_sized({"a": 10, "b": 10}), 0.2)


def test_filtering_drops_the_applied_calls():
    a = _scored()
    a.X = np.ones((a.n_obs, 3), dtype="float32")
    out = doublets.apply_threshold(a, 0.2)
    kept, _ = pl._run_filtering(a, pl.PipelineParams(qc_threshold_mode="fixed", min_genes=0,
                                                     min_cells=0, drop_doublets=True))
    assert kept.n_obs == a.n_obs - out["n_called"]


def test_the_applied_threshold_survives_a_save(tmp_path):
    a = _scored()
    doublets.apply_threshold(a, 0.2)
    path = tmp_path / "d.h5ad"
    a.write_h5ad(path)
    b = ad.read_h5ad(path)
    out = doublets.summary(b)
    assert out["applied"]["threshold"] == pytest.approx(0.2) and out["threshold"] == pytest.approx(0.2)
    doublets.apply_threshold(b, None)
    np.testing.assert_array_equal(b.obs["predicted_doublet"].to_numpy(),
                                  a.obs["predicted_doublet_scrublet"].to_numpy())


# --- the doublet step ------------------------------------------------------------


def test_the_step_takes_each_channels_suggestion_unless_a_rate_is_given(monkeypatch):
    from scview.core import scrublet_lean

    seen: list[tuple[int, float]] = []

    def fake_scrublet(counts, expected_doublet_rate, use_approx_neighbors=None, owned=False):
        seen.append((counts.shape[0], expected_doublet_rate))
        return np.zeros(counts.shape[0]), np.zeros(counts.shape[0], dtype=bool), {}

    monkeypatch.setattr(scrublet_lean, "scrublet_counts", fake_scrublet)
    a = _sized({"small": 2000, "big": 6000})
    assert pl.PipelineParams().expected_doublet_rate is None
    found = pl._run_doublet_detection(a, pl.PipelineParams())
    assert sorted(seen) == [(2000, pytest.approx(0.016)), (6000, pytest.approx(0.048))]
    assert found["expected_doublet_rate"] is None  # one per channel
    assert found["expected_rate_from"] == "sample channels (per channel)"
    assert [(c["name"], c["expected_doublet_rate"]) for c in found["channels"]] == [
        ("small", pytest.approx(0.016)), ("big", pytest.approx(0.048))]
    assert (found["channel_column"], found["channel_reason"], found["neighbors"]) == ("sample", "donor", "exact")
    seen.clear()
    found = pl._run_doublet_detection(a, pl.PipelineParams(expected_doublet_rate=0.1))
    assert [r for _, r in seen] == [0.1, 0.1] and found["expected_doublet_rate"] == 0.1
    assert "expected_rate_from" not in found


def test_a_real_run_records_each_channels_rate_and_its_threshold_gives_its_calls():
    a = _counts()
    out, res = pl.run_pipeline(a, ["qc_metrics", "doublet_detection"], pl.PipelineParams())
    assert res.errors == {}
    s = doublets.summary(out)
    # 300 cells per sample (the channels here): the 0.4 % floor of the rule of thumb.
    assert [r["expected_rate"] for r in s["channels"]] == [doublets.MIN_RATE] * 2
    assert s["expected_rate_used"] is None and s["per_channel"] and s["scrublet_threshold"] is None
    scores = out.obs["doublet_score"].to_numpy()
    calls = out.obs["predicted_doublet"].to_numpy()
    for row in s["channels"]:
        idx = (out.obs["sample"] == row["name"]).to_numpy()
        t = row["scrublet_threshold"]
        assert t is not None  # Scrublet's random_state is fixed
        assert np.array_equal(calls[idx], scores[idx] > t)  # its own threshold, its own calls
    assert s["n_called_now"] == int(calls.sum()) == sum(r["n_called"] for r in s["channels"])

    # A threshold set on these scores is dropped by the next run.
    doublets.apply_threshold(out, 0.01)
    pl._run_doublet_detection(out, pl.PipelineParams())
    assert doublets.THRESHOLD_KEY not in out.uns
    assert "predicted_doublet_scrublet" not in out.obs.columns


# --- the endpoints ---------------------------------------------------------------


def _register(dm, a: ad.AnnData, did: str = "dbl") -> str:
    d = Path(dm.data_dir) / "uploads" / did
    d.mkdir(parents=True)
    (d / "metadata.json").write_text(json.dumps({"id": did, "name": did, "status": "ready"}))
    a.write_h5ad(d / "data.h5ad")
    return did


async def test_get_summarises_off_the_loop(client, dataset_manager, monkeypatch):
    did = _register(dataset_manager, _scored())
    threads: list[threading.Thread] = []
    real = doublets.summary

    def spy(*args, **kwargs):
        threads.append(threading.current_thread())
        return real(*args, **kwargs)

    monkeypatch.setattr(doublets, "summary", spy)
    r = await client.get(f"{API}/datasets/{did}/qc/doublets", params={"threshold": 0.2})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["available"] and body["threshold"] == 0.2 and body["suggestion"]["column"] == "sample"
    assert len(threads) == 1 and threads[0] is not threading.main_thread()

    r = await client.get(f"{API}/datasets/{did}/qc/doublets", params={"threshold": -1})
    assert r.status_code == 400
    r = await client.get(f"{API}/datasets/{did}/qc/doublets", params={"column": "nope"})
    assert r.status_code == 400
    r = await client.get(f"{API}/datasets/ghost/qc/doublets")
    assert r.status_code == 404


async def test_post_saves_the_calls_and_puts_scrublets_back(client, dataset_manager):
    a = _scored()
    original = a.obs["predicted_doublet"].to_numpy().copy()
    did = _register(dataset_manager, a)
    r = await client.post(f"{API}/datasets/{did}/qc/doublets/threshold", json={"threshold": 0.12})
    assert r.status_code == 200, r.text
    assert r.json()["applied"]["threshold"] == 0.12
    derived = Path(dataset_manager.data_dir) / "derived" / did / "data.h5ad"
    on_disk = ad.read_h5ad(derived)
    np.testing.assert_array_equal(on_disk.obs["predicted_doublet"].to_numpy(),
                                  on_disk.obs["doublet_score"].to_numpy() > 0.12)
    np.testing.assert_array_equal(on_disk.obs["predicted_doublet_scrublet"].to_numpy(), original)
    # The served copy is the saved one, and the library lists the new column.
    info = await client.get(f"{API}/datasets/{did}")
    assert "predicted_doublet_scrublet" in [c["name"] for c in info.json()["obs_columns"]]

    r = await client.post(f"{API}/datasets/{did}/qc/doublets/threshold", json={"threshold": None})
    assert r.status_code == 200 and r.json()["applied"] is None
    np.testing.assert_array_equal(ad.read_h5ad(derived).obs["predicted_doublet"].to_numpy(), original)

    r = await client.post(f"{API}/datasets/{did}/qc/doublets/threshold", json={"threshold": -2})
    assert r.status_code == 400


async def test_post_without_scores_is_refused(client, dataset_manager):
    did = _register(dataset_manager, _sized({"a": 20, "b": 20}))
    r = await client.post(f"{API}/datasets/{did}/qc/doublets/threshold", json={"threshold": 0.2})
    assert r.status_code == 400 and "doublet detection" in r.json()["detail"]


def test_on_a_processed_dataset_scrublet_scores_the_counts_layer():
    # Run alone on a processed dataset, Scrublet used to get the scaled X
    # (negative, non-integer): it filled the backend's memory and failed on NaNs.
    a = _counts()
    counts = a.X.copy()
    a.layers["counts"] = counts.copy()
    a.X = (a.X - a.X.mean(axis=0)) / (a.X.std(axis=0) + 1)  # a scaled X
    scaled = a.X.copy()
    found = pl._run_doublet_detection(a, pl.PipelineParams())
    assert found["input"] == "layers['counts']"
    assert np.isfinite(a.obs["doublet_score"]).all() and "predicted_doublet" in a.obs
    assert np.array_equal(a.X, scaled) and np.array_equal(a.layers["counts"], counts)


def test_without_counts_anywhere_the_step_is_skipped_with_the_reason():
    a = _counts()
    a.X = a.X / 3.7 - 1.0
    with pytest.raises(pl.StepSkipped, match="needs raw counts"):
        pl._run_doublet_detection(a, pl.PipelineParams())
