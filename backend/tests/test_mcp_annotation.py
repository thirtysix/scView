"""The MCP annotation tools (v0.80, QA round 6): the CellTypist model list, a misspelt model refused
before any job, the annotation cross-check, and the AI annotation refused without a key."""

from __future__ import annotations

import pandas as pd
import pytest

from scview.api.v1 import annotation
from scview.mcp import jobs
from tests.test_mcp import _call, _register, msigdb_dir, test_settings  # noqa: F401 - fixtures

CATALOG = [
    ("Immune_All_Low.pkl", "immune sub-populations combined from 20 tissues of 18 studies", 98),
    ("Human_Lung_Atlas.pkl", "integrated Human Lung Cell Atlas (HLCA) combining multiple datasets", 61),
    ("Healthy_Mouse_Liver.pkl", "cell types from scRNA-seq and snRNA-seq of the healthy murine liver", 17),
    ("Adult_COVID19_PBMC.pkl", "peripheral blood mononuclear cell types from COVID-19 patients", 20),
]


@pytest.fixture(autouse=True)
def catalog(monkeypatch):
    monkeypatch.setattr(annotation, "_model_catalog", lambda: CATALOG)


async def test_celltypist_models_lists_species_tissues_and_cell_types(client):
    r = await _call("celltypist_models", {})
    rows = {row[0]: dict(zip(r["models"]["columns"], row)) for row in r["models"]["rows"]}
    assert r["n_models"] == 4 and r["default"] == "Immune_All_Low.pkl"
    assert rows["Human_Lung_Atlas.pkl"] | {"description": ""} == {
        "model": "Human_Lung_Atlas.pkl", "species": "human", "organs": "lung", "n_types": 61, "description": ""}
    assert rows["Healthy_Mouse_Liver.pkl"]["species"] == "mouse"
    blood = await _call("celltypist_models", {"tissue": "blood"})  # blood and immune are one
    assert [row[0] for row in blood["models"]["rows"]] == ["Immune_All_Low.pkl", "Adult_COVID19_PBMC.pkl"]
    mouse = await _call("celltypist_models", {"species": "mouse"})
    assert [row[0] for row in mouse["models"]["rows"]] == ["Healthy_Mouse_Liver.pkl"]
    assert "names no tissue" in await _call("celltypist_models", {"tissue": "zzz"}, error=True)
    kidney = await _call("celltypist_models", {"tissue": "kidney"})  # QA round 6: none, said in words
    assert kidney["n_models"] == 0 and "No CellTypist model covers kidney" in kidney["note"]
    assert "note" not in blood


async def test_a_misspelt_model_is_refused_before_any_job(client, dataset_manager, monkeypatch):
    """QA round 6: Kidney_list_please.pkl ran 68 s (the dataset loaded) before failing, with no list."""
    did = _register(dataset_manager)
    started = []

    async def stream(*a, **k):
        started.append(a)

    monkeypatch.setattr(jobs.api, "stream", stream)
    text = await _call("annotate_celltypist", {"dataset_id": did, "model": "Human_Lung_Atlass"}, error=True)
    assert "No CellTypist model 'Human_Lung_Atlass'" in text and "Human_Lung_Atlas.pkl" in text
    assert "celltypist_models lists all 4" in text
    text = await _call("run_pipeline", {"dataset_id": did, "steps": ["cell_type_annotation"], "params": {
        "annotation_method": "celltypist", "celltypist_model": "Kidney_list_please.pkl"}}, error=True)
    assert "No CellTypist model 'Kidney_list_please.pkl'" in text
    assert started == []


async def test_a_known_model_runs_with_its_catalog_spelling(client, dataset_manager, monkeypatch):
    did = _register(dataset_manager)
    sent = {}

    async def stream(path, body, on_event, disconnect=None):
        sent.update(body=body)
        await on_event("complete", {"status": "complete", "steps_run": ["cell_type_annotation"]})

    monkeypatch.setattr(jobs.api, "stream", stream)
    r = await _call("annotate_celltypist", {"dataset_id": did, "model": "human_lung_atlas"})
    assert r["job_id"] and sent["body"]["params"]["celltypist_model"] == "Human_Lung_Atlas.pkl"


async def test_an_unreadable_catalog_checks_nothing(client, dataset_manager, monkeypatch):
    def offline():
        raise OSError("no network")

    monkeypatch.setattr(annotation, "_model_catalog", offline)
    monkeypatch.setattr(jobs.api, "stream", lambda *a, **k: _done(*a, **k))
    did = _register(dataset_manager)
    r = await _call("annotate_celltypist", {"dataset_id": did, "model": "Some_New_Model"})
    assert r["job_id"]
    m = await _call("celltypist_models", {})
    assert m["n_models"] == 1 and "could not be read" in m["note"]


async def _done(path, body, on_event, disconnect=None):
    await on_event("complete", {"status": "complete", "steps_run": ["cell_type_annotation"]})


async def test_annotation_crosscheck(client, dataset_manager):
    did = _register(dataset_manager)
    a = dataset_manager  # the registered file: add two label columns through the adaptor
    adaptor = await a.get_or_load_dataset(did)
    obs = adaptor.adata.obs
    obs["author"] = pd.Categorical(obs["cluster"].astype(str).map(lambda c: f"type{c}"))
    obs["other"] = pd.Categorical(obs["cluster"].astype(str).map(lambda c: f"T{c}"))
    r = await _call("annotation_crosscheck", {"dataset_id": did, "columns": ["author", "other", "cluster"]})
    assert r["reference"] == "author" and r["mean_agreement"] == 1.0
    assert r["vs_reference"]["other"]["ari"] == 1.0 and r["vs_reference"]["other"]["mapped_agreement"] == 1.0
    assert r["vs_reference"]["other"]["exact_name_agreement"] == 0.0
    assert r["ari"]["author"]["cluster"] == 1.0
    row = dict(zip(r["labels"]["columns"], r["labels"]["rows"][0]))
    assert row["label"].startswith("type") and "other: T" in row["calls"]
    bad = await _call("annotation_crosscheck", {"dataset_id": did, "columns": ["author", "nope"]}, error=True)
    assert "nope" in bad


async def test_run_pipeline_refuses_the_ai_annotation_without_a_key(client, dataset_manager):
    did = _register(dataset_manager)
    text = await _call("run_pipeline", {"dataset_id": did, "steps": ["cell_type_annotation"],
                                        "params": {"annotation_method": "llm"}}, error=True)
    assert "language model key" in text and "annotation_fallback" in text
