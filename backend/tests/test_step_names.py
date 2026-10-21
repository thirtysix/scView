"""Unknown pipeline step names (core/step_names.py): 422 with the likely step meant.

Asking for step "umap" (the step is "embeddings") used to return 200 with
``steps_run: []`` and no error.
"""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pytest

from scview.core import pipeline
from scview.core.step_names import step_hint, unknown_steps, unknown_steps_message

API = "/api/v1"


@pytest.mark.parametrize("name, meant", [
    ("umap", "embeddings"), ("tsne", "embeddings"), ("fa", "embeddings"),
    ("leiden", "clustering"), ("louvain", "clustering"),
    ("qc", "qc_metrics"),
    ("normalize", "normalization"), ("normalise", "normalization"),
    ("hvg", "highly_variable_genes"),
    ("harmony", "batch_correction"),
    ("markers", "marker_genes"),
    ("doublets", "doublet_detection"), ("scrublet", "doublet_detection"),
    # Spelling slips (difflib) and the right name in the wrong case or separators
    ("clustring", "clustering"), ("embedding", "embeddings"), ("UMAP", "embeddings"),
    ("Cell-Cycle", "cell_cycle"), ("marker genes", "marker_genes"),
])
def test_hint(name, meant):
    assert step_hint(name) == meant


def test_no_hint_for_nothing_close():
    assert step_hint("make_coffee") is None
    assert "did you mean" not in unknown_steps_message(["make_coffee"])


def test_unknown_steps_keeps_request_order_once_each():
    assert unknown_steps(["pca", "umap", "leiden", "umap", "neighbors"]) == ["umap", "leiden"]
    assert unknown_steps(["pca", "neighbors"]) == []


def test_message_names_each_unknown_step_and_every_valid_one():
    msg = unknown_steps_message(["umap", "foo"])
    assert msg.startswith("Unknown pipeline steps 'umap' (did you mean 'embeddings'?), 'foo'.")
    for step in pipeline.ALL_STEPS:
        assert step in msg
    assert unknown_steps_message(["umap"]).startswith("Unknown pipeline step 'umap'")


def test_run_pipeline_lists_unknown_steps_in_the_result():
    a = ad.AnnData(np.random.default_rng(0).poisson(1.0, (50, 20)).astype("float32"))
    _, result = pipeline.run_pipeline(a, ["qc_metrics", "umap"])
    assert result.steps_run == ["qc_metrics"]
    assert result.unknown_steps == ["umap"]
    events = list(pipeline.run_pipeline_streamed(a.copy(), ["umap"]))
    assert events[-1][0] == "complete"
    assert events[-1][1]["unknown_steps"] == ["umap"]
    assert events[-1][1]["steps_run"] == []


def _register(dm, did: str = "steps") -> str:
    d = Path(dm.data_dir) / "uploads" / did
    d.mkdir(parents=True, exist_ok=True)
    (d / "metadata.json").write_text(json.dumps({"id": did, "name": did, "status": "ready"}))
    a = ad.AnnData(np.random.default_rng(0).poisson(1.0, (60, 30)).astype("float32"))
    a.var_names = [f"G{i}" for i in range(30)]
    a.write_h5ad(d / "data.h5ad")
    return did


@pytest.mark.parametrize("endpoint", ["assessment/run", "assessment/run-stream"])
async def test_run_endpoints_refuse_unknown_steps_with_422(client, dataset_manager, endpoint):
    did = _register(dataset_manager)
    r = await client.post(f"{API}/datasets/{did}/{endpoint}", json={"steps": ["pca", "umap"]})
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert "'umap' (did you mean 'embeddings'?)" in detail
    assert "Valid steps: reset_to_counts, qc_metrics" in detail
    # Nothing ran: the dataset was not even loaded.
    assert dataset_manager.get_dataset(did) is None
    # An empty list is still a 400, and a real step still runs.
    assert (await client.post(f"{API}/datasets/{did}/{endpoint}", json={"steps": []})).status_code == 400


async def test_run_endpoint_still_runs_known_steps(client, dataset_manager):
    did = _register(dataset_manager)
    r = await client.post(f"{API}/datasets/{did}/assessment/run", json={"steps": ["qc_metrics"]})
    assert r.status_code == 200
    body = r.json()
    assert body["steps_run"] == ["qc_metrics"]
    assert body["status"] == "complete"
    assert body["unknown_steps"] == []


async def test_rerun_refuses_an_unknown_step_with_422(client, dataset_manager):
    did = _register(dataset_manager)
    r = await client.post(f"{API}/datasets/{did}/rerun", json={"edited_step": "leiden"})
    assert r.status_code == 422
    assert "did you mean 'clustering'?" in r.json()["detail"]
    assert dataset_manager.get_dataset(did) is None  # refused before loading
    # A real step that cannot be re-run in place keeps its own answer.
    r = await client.post(f"{API}/datasets/{did}/rerun", json={"edited_step": "pca"})
    assert r.status_code in (400, 409)
