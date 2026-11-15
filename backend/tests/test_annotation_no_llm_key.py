"""The AI annotation without a language model key: refused, never a silent CellTypist run.

QA round 6 (research/v080-qa/kidney): ``annotation_method llm`` on a stack without a key ran
CellTypist's immune model (podocytes -> "Alveolar macrophages") while the recipe said "llm" and
the run had no note. Now the routes refuse it (422), a run that gets past them skips the step
with the reason, and ``annotation_fallback`` names a method to run instead, recorded as the
method that ran.
"""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from scview.config import Settings
from scview.core import pipeline
from scview.core.pipeline import PipelineParams, run_pipeline

API = "/api/v1"
GENES = ["CD3D", "CD3E", "CD3G", "TRAC", "MS4A1", "CD79A", "CD79B", "CD19"]


@pytest.fixture(autouse=True)
def no_key(monkeypatch):
    monkeypatch.setattr("scview.config.get_settings", lambda: Settings(
        DEEPINFRA_API_KEY="", LLM_API_KEY="", LLM_BASE_URL=""))


def _adata() -> ad.AnnData:
    rng = np.random.default_rng(0)
    X = (rng.random((20, 8)) * 0.1).astype("float32")
    X[:10, :4] += 3.0
    X[10:, 4:] += 3.0
    a = ad.AnnData(X)
    a.var_names = GENES
    a.obs_names = [f"c{i}" for i in range(20)]
    a.obs["cluster"] = pd.Categorical(["0"] * 10 + ["1"] * 10)
    return a


def _records(a) -> list[dict]:
    from scview.core import provenance

    return [h for h in provenance.read_provenance(a)["history"] if h["step"] == "cell_type_annotation"]


def test_plan_refuses_llm_without_a_key_and_names_the_way_out():
    with pytest.raises(pipeline.AnnotationMethodError) as e:
        pipeline.annotation_method_plan(PipelineParams(annotation_method="llm"))
    assert "language model key" in str(e.value) and "annotation_fallback" in str(e.value)
    # other methods need no key
    assert pipeline.annotation_method_plan(PipelineParams(annotation_method="celltypist")) == ("celltypist", None)
    method, note = pipeline.annotation_method_plan(
        PipelineParams(annotation_method="llm", annotation_fallback="marker_score"))
    assert method == "marker_score" and "no language model key" in note and "marker-score" in note
    with pytest.raises(pipeline.AnnotationMethodError, match="not one of"):
        pipeline.annotation_method_plan(PipelineParams(annotation_method="llm", annotation_fallback="bogus"))


def test_plan_runs_llm_with_a_key(monkeypatch):
    monkeypatch.setattr("scview.config.get_settings", lambda: Settings(DEEPINFRA_API_KEY="k", LLM_BASE_URL=""))
    assert pipeline.annotation_method_plan(
        PipelineParams(annotation_method="llm", annotation_fallback="celltypist")) == ("llm", None)


def test_a_run_skips_the_step_with_the_reason_and_writes_nothing():
    a = _adata()
    a, res = run_pipeline(a, ["cell_type_annotation"], PipelineParams(annotation_method="llm"))
    assert res.status == "complete" and not res.errors
    assert "language model key" in res.notes["cell_type_annotation"]
    assert "cell_type_annotation" not in res.steps_run
    assert "cluster_celltypeAnno" not in a.obs
    assert _records(a) == []


def test_a_named_fallback_runs_and_is_recorded_as_the_method_that_ran():
    a = _adata()
    a, res = run_pipeline(a, ["cell_type_annotation"], PipelineParams(
        annotation_method="llm", annotation_fallback="marker_score", annotation_marker_sets="builtin"))
    assert not res.errors, res.errors
    assert set(a.obs["cluster_celltypeAnno"]) == {"T cell", "B cell"}
    [rec] = _records(a)
    p = rec["params"]
    assert p["annotation_method"] == "marker_score" and p["annotation_method_requested"] == "llm"
    assert "marker-score annotation ran instead" in p["annotation_fallback_note"]
    assert res.cautions["cell_type_annotation"] == p["annotation_fallback_note"]


def test_the_replay_script_replays_the_method_that_ran():
    from scview.core import replay_script

    a = _adata()
    a, _ = run_pipeline(a, ["cell_type_annotation"], PipelineParams(
        annotation_method="llm", annotation_fallback="marker_score", annotation_marker_sets="builtin"))
    [rec] = _records(a)
    snip = replay_script._cell_type_annotation(rec["params"], replay_script._Ctx(adata=a))
    code = "\n".join(snip.code)
    assert "MARKER_SETS" in code and "LLM" not in code


def _register(dm, did: str) -> None:
    d = Path(dm.data_dir) / "uploads" / did
    d.mkdir(parents=True, exist_ok=True)
    (d / "metadata.json").write_text(json.dumps({"id": did, "name": did, "status": "ready"}))
    _adata().write_h5ad(d / "data.h5ad")


@pytest.mark.parametrize("path", ["assessment/run", "assessment/run-stream"])
async def test_the_run_routes_refuse_llm_without_a_key(client, dataset_manager, path):
    _register(dataset_manager, "nokey")
    body = {"steps": ["cell_type_annotation"], "params": {"annotation_method": "llm"}}
    r = await client.post(f"{API}/datasets/nokey/{path}", json=body)
    assert r.status_code == 422
    assert "language model key" in r.json()["detail"]


async def test_the_run_route_runs_a_named_fallback(client, dataset_manager):
    _register(dataset_manager, "nokey2")
    body = {"steps": ["cell_type_annotation"],
            "params": {"annotation_method": "llm", "annotation_fallback": "marker_score",
                       "annotation_marker_sets": "builtin"}}
    r = await client.post(f"{API}/datasets/nokey2/assessment/run", json=body)
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["status"] == "complete" and "ran instead" in d["cautions"]["cell_type_annotation"]


def test_preflight_estimates_the_method_that_will_run():
    """The live check: the AI's fallback to CellTypist was estimated as the AI's call (0.1 GB) and the
    47k-nucleus kidney run was OOM-killed scaling CellTypist's genes."""
    from scview.core import preflight

    def peak(**kw):
        est = preflight.estimate_run((40_000, 20_000), nnz=40_000 * 1_500, steps=["cell_type_annotation"],
                                     params=PipelineParams(**kw))
        return next(s.transient_bytes for s in est.steps if s.step == "cell_type_annotation")

    ct = peak(annotation_method="celltypist")
    assert peak(annotation_method="llm", annotation_fallback="celltypist") == ct
    assert peak(annotation_method="llm") < ct / 10
