"""Stopping a pipeline run between steps (core/run_watchdog.py RunControl): the
pre-step hook ``before_step``.

A run the pre-step check stops keeps (and saves) the steps it did.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd

from scview.core import pipeline, provenance, run_watchdog
from scview.core.run_watchdog import Memory, RunControl, RunWatchdog

STEPS = ["qc_metrics", "normalization", "log_transform", "pca"]


def _quiet() -> RunWatchdog:
    return RunWatchdog(read=lambda: Memory(1 << 30, 8 << 30, True), interval=0.01)


def _counts(n: int = 120, g: int = 40) -> ad.AnnData:
    a = ad.AnnData(np.random.default_rng(0).poisson(1.0, (n, g)).astype("float32"))
    a.var_names = [f"G{i}" for i in range(g)]
    a.obs_names = [f"c{i}" for i in range(n)]
    a.obs["grp"] = pd.Categorical(np.where(np.arange(n) < n // 2, "a", "b"))
    return a


# -- before_step --------------------------------------------------------------


def test_before_step_is_called_with_step_adata_and_params_in_order():
    calls = []

    def hook(step, adata, params):
        calls.append((step, adata.shape, params.n_comps))
        return None

    params = pipeline.PipelineParams(n_comps=7)
    _, result = pipeline.run_pipeline(_counts(), STEPS, params,
                                      control=RunControl(_quiet(), before_step=hook))
    assert [c[0] for c in calls] == STEPS
    assert all(c[1] == (120, 40) and c[2] == 7 for c in calls)
    assert result.status == "complete" and result.stop_reason is None


def test_a_message_stops_the_run_before_that_step_and_keeps_the_steps_done(tmp_path):
    msg = "PCA needs about 9.0 GB and 2.1 GB is free."
    out = tmp_path / "out.h5ad"

    def hook(step, _adata, _params):
        return msg if step == "pca" else None

    _, result = pipeline.run_pipeline(_counts(), STEPS, output_path=str(out),
                                      control=RunControl(_quiet(), before_step=hook))
    assert result.status == "stopped"
    assert result.stop_reason == msg
    assert result.steps_run == ["qc_metrics", "normalization", "log_transform"]
    saved = ad.read_h5ad(out)
    assert "X_pca" not in saved.obsm
    assert [h["step"] for h in provenance.read_provenance(saved)["history"]] == result.steps_run


def test_the_streamed_run_stops_without_starting_the_step(tmp_path):
    def hook(step, _adata, _params):
        return "No room for PCA." if step == "pca" else None

    events = list(pipeline.run_pipeline_streamed(
        _counts(), STEPS, control=RunControl(_quiet(), before_step=hook)))
    starts = [d["step"] for k, d in events if k == "step_start"]
    assert starts == ["qc_metrics", "normalization", "log_transform"]
    kind, data = events[-1]
    assert kind == "complete"
    assert data["status"] == "stopped" and data["stop_reason"] == "No room for PCA."
    assert data["steps_run"] == starts


def test_stopped_before_the_first_step_saves_nothing(tmp_path):
    out = tmp_path / "out.h5ad"
    _, result = pipeline.run_pipeline(_counts(), STEPS, output_path=str(out),
                                      control=RunControl(_quiet(), before_step=lambda *_: "No."))
    assert result.status == "stopped" and result.steps_run == []
    assert not out.exists()


def test_a_failing_check_is_logged_and_the_step_runs(caplog):
    def broken(*_a):
        raise KeyError("estimate")

    _, result = pipeline.run_pipeline(_counts(), ["qc_metrics"],
                                      control=RunControl(_quiet(), before_step=broken))
    assert result.status == "complete" and result.steps_run == ["qc_metrics"]
    assert "pre-step check failed" in caplog.text


def test_the_default_check_is_where_preflight_plugs_in(monkeypatch):
    assert run_watchdog.default_before_step("pca", None, None) is None
    monkeypatch.setattr(run_watchdog, "default_before_step",
                        lambda step, *_: "stop" if step == "normalization" else None)
    _, result = pipeline.run_pipeline(_counts(), STEPS, control=RunControl(_quiet()))
    assert result.status == "stopped" and result.steps_run == ["qc_metrics"]
    # None switches the check off.
    _, result = pipeline.run_pipeline(_counts(), STEPS, control=RunControl(_quiet(), before_step=None))
    assert result.status == "complete"
