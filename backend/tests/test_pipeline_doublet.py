"""Tests for the doublet-detection pipeline step (Scrublet) and its wiring."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pytest

from scview.core.assessor import assess_preprocessing
from scview.core.pipeline import (
    ALL_STEPS,
    PipelineParams,
    _run_doublet_detection,
    _run_filtering,
    run_pipeline,
)
from scview.core.rerun import descendants


def _synthetic(n_cells: int = 500, n_genes: int = 400, seed: int = 0) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    X = rng.poisson(1.0, size=(n_cells, n_genes)).astype("float32")
    a = ad.AnnData(X)
    a.var_names = [f"G{i}" for i in range(n_genes)]
    a.obs_names = [f"C{i}" for i in range(n_cells)]
    return a


# --- Structural wiring (deterministic, no scrublet run) ---------------------

def test_doublet_step_ordered_after_qc_before_filtering():
    assert "doublet_detection" in ALL_STEPS
    assert ALL_STEPS.index("qc_metrics") < ALL_STEPS.index("doublet_detection")
    assert ALL_STEPS.index("doublet_detection") < ALL_STEPS.index("filtering")


def test_doublet_is_descendant_of_qc_metrics():
    assert "doublet_detection" in descendants("qc_metrics")


def test_params_have_doublet_fields():
    p = PipelineParams()
    assert p.doublet_method == "scrublet"
    assert p.expected_doublet_rate == pytest.approx(0.06)
    assert p.drop_doublets is False


# --- Filtering drop_doublets branch (deterministic — flags set by hand) ------

def test_filtering_drops_predicted_doublets_when_enabled():
    a = _synthetic(n_cells=100, n_genes=50)
    flags = np.zeros(100, dtype=bool)
    flags[:10] = True  # mark 10 cells as doublets
    a.obs["predicted_doublet"] = flags
    out = _run_filtering(a, PipelineParams(min_genes=0, min_cells=0, drop_doublets=True))
    assert out.n_obs == 90


def test_filtering_keeps_doublets_when_disabled():
    a = _synthetic(n_cells=100, n_genes=50)
    a.obs["predicted_doublet"] = np.array([True] * 10 + [False] * 90)
    out = _run_filtering(a, PipelineParams(min_genes=0, min_cells=0, drop_doublets=False))
    assert out.n_obs == 100


# --- Real Scrublet run ------------------------------------------------------

def test_doublet_detection_annotates_obs():
    a = _synthetic()
    _run_doublet_detection(a, PipelineParams())
    assert "doublet_score" in a.obs.columns
    assert "predicted_doublet" in a.obs.columns
    assert a.obs["doublet_score"].notna().all()


def test_pipeline_runs_doublet_step_end_to_end():
    a = _synthetic()
    out, res = run_pipeline(
        a, ["qc_metrics", "doublet_detection", "filtering"],
        PipelineParams(min_genes=0, min_cells=0),
    )
    assert res.errors == {}
    assert "doublet_detection" in res.steps_run
    assert "doublet_score" in out.obs.columns
    # assessor recognises the recorded step
    state = assess_preprocessing(out)
    assert state.doublet_detection.done is True
