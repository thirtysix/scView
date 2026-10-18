"""CellTypist fit: model species, gene overlap and per-label confidence (C012)."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from scview.core import celltypist_fit as fit
from scview.core import provenance


@pytest.mark.parametrize("name, description, species", [
    ("Immune_All_Low.pkl", "immune sub-populations combined from 20 tissues of 18 studies", "human"),
    ("Adult_Mouse_Gut.pkl", "cell types in the adult mouse gut combined from eight datasets", "mouse"),
    ("Healthy_Mouse_Liver.pkl", "cell types from scRNA-seq and snRNA-seq of the healthy murine liver",
     "mouse"),
    ("Mouse_Dendritic_Subtypes.pkl", "dendritic cell subtypes across murine spleen", "mouse"),
    ("Adult_Mouse_OlfactoryBulb.pkl", "cell types from the olfactory bulb of adult mice", "mouse"),
    ("Developing_Mouse_Brain.pkl", "", "mouse"),
    ("Adult_Pig_Hippocampus.pkl", "cell types from the adult pig hippocampus", "other"),
    ("Adult_RhesusMacaque_Hippocampus.pkl", "hippocampus of adult rhesus monkeys", "other"),
    ("Pan_Fetal_Human.pkl", "stromal and immune populations from the human fetus", "human"),
])
def test_model_species_from_the_catalog(name, description, species):
    assert fit.model_species(name, description) == species


def test_species_warning_only_on_a_known_mismatch():
    assert fit.species_warning("human", "human") == ""
    assert fit.species_warning("human", "unknown") == ""
    assert "mouse dataset" in fit.species_warning("human", "mouse")
    assert "human dataset" in fit.species_warning("mouse", "human")
    assert "another species" in fit.species_warning("other", "human")


def test_gene_overlap_matches_names_exactly():
    o = fit.gene_overlap(["CD3E", "MS4A1", "LYZ", "NKG7"], ["CD3E", "Ms4a1", "LYZ"])
    assert o == {"n_model_genes": 4, "n_shared_genes": 2, "gene_overlap": 0.5}
    assert fit.overlap_caution(0.5) == ""
    assert fit.overlap_caution(None) == ""
    assert "only 30% of the model" in fit.overlap_caution(0.3)
    assert "only 1 of the model's 2,787 genes (0.04%)" in fit.overlap_caution(0.0004, 1, 2787)


def _labelled() -> ad.AnnData:
    """Three labels: T sure (0.9), B unsure (0.3), NK mixed (half 0.8, half 0.2)."""
    labels = ["T"] * 10 + ["B"] * 6 + ["NK"] * 4
    conf = [0.9] * 10 + [0.3] * 6 + [0.8, 0.8, 0.2, 0.2]
    a = ad.AnnData(X=np.zeros((20, 2), dtype=np.float32))
    a.obs["ct"] = pd.Categorical(labels, categories=["T", "B", "NK", "unused"])
    a.obs["ct_confidence"] = np.asarray(conf, dtype=np.float32)
    a.obs["ct_percell"] = pd.Categorical(["T"] * 9 + ["B"] + ["B"] * 3 + ["T"] * 3 + ["NK"] * 4)
    return a


def test_summary_per_label_in_category_order():
    a = _labelled()
    fit.record_run(a, "ct", {"model": "Immune_All_Low.pkl", "model_species": "human",
                             "dataset_species": "human", "gene_overlap": 0.92,
                             "n_model_genes": 100, "n_shared_genes": 92})
    s = fit.summarize(a, "ct")
    assert [r["label"] for r in s["labels"]] == ["T", "B", "NK"]  # no empty "unused"
    t, b, nk = s["labels"]
    assert t["n_cells"] == 10 and t["median_confidence"] == pytest.approx(0.9)
    assert t["pct_below"] == 0 and not t["low_confidence"] and t["pct_agree"] == 90.0
    assert b["low_confidence"] and b["pct_below"] == 100.0 and b["pct_agree"] == 50.0
    assert nk["median_confidence"] == pytest.approx(0.5) and not nk["low_confidence"]
    assert nk["pct_below"] == 50.0
    assert s["n_low_labels"] == 1 and s["model"] == "Immune_All_Low.pkl"
    assert s["median_confidence"] == pytest.approx(0.85)  # 10th and 11th of 20: 0.8, 0.9
    assert s["cautions"] == []
    assert s["confidence_column"] == "ct_confidence" and s["percell_column"] == "ct_percell"


def test_summary_cautions_species_overlap_and_low_median():
    a = _labelled()
    a.obs["ct_confidence"] = np.float32(0.2)
    fit.record_run(a, "ct", {"model": "Immune_All_Low.pkl", "model_species": "human",
                             "dataset_species": "mouse", "gene_overlap": 0.12})
    cautions = fit.summarize(a, "ct")["cautions"]
    assert len(cautions) == 3
    assert "mouse dataset" in cautions[0]
    assert "only 12%" in cautions[1]
    assert "may not match this tissue" in cautions[2]


def test_summary_without_a_confidence_column_is_none_and_a_run_can_be_dropped():
    a = _labelled()
    assert fit.summarize(a, "nope") is None
    del a.obs["ct_confidence"]
    assert fit.summarize(a, "ct") is None
    fit.record_run(a, "ct", {"model": "m.pkl"})
    assert fit.run_columns(a) == ["ct"]
    fit.drop_run(a, "ct")
    assert fit.run_columns(a) == [] and json.loads(a.uns[fit.RUNS_KEY]) == {}


# ---------------------------------------------------------------------------
# The pipeline step
# ---------------------------------------------------------------------------


def _fake_celltypist(monkeypatch, features: list[str], details: str = "immune populations"):
    calls: dict = {}

    def annotate(inp, model, majority_voting, over_clustering=None):
        calls["model"] = model
        votes = np.where(inp.obs["cluster"].astype(str) == "a", "T cells", "B cells")
        labels = pd.DataFrame({"predicted_labels": votes, "majority_voting": votes},
                              index=inp.obs_names)
        prob = pd.DataFrame(np.random.RandomState(1).uniform(size=(inp.n_obs, 2)),
                            index=inp.obs_names, columns=["T cells", "B cells"])
        return types.SimpleNamespace(predicted_labels=labels, probability_matrix=prob)

    loaded = types.SimpleNamespace(features=np.asarray(features), description={"details": details})
    model_cls = types.SimpleNamespace(load=lambda model=None: loaded)
    fake = types.SimpleNamespace(
        annotate=annotate,
        models=types.SimpleNamespace(download_models=lambda **_: None, Model=model_cls),
    )
    monkeypatch.setitem(sys.modules, "celltypist", fake)
    return calls, loaded


def _counts(genes: list[str]) -> ad.AnnData:
    a = ad.AnnData(X=np.random.RandomState(0).poisson(1, (40, len(genes))).astype("float32"))
    a.obs_names = [f"c{i}" for i in range(40)]
    a.var_names = genes
    a.obs["cluster"] = pd.Categorical(["a", "b"] * 20)
    return a


HUMAN = [f"GENE{i}" for i in range(60)] + ["MT-CO1", "MT-ND1"]


def test_step_records_species_and_overlap_in_uns_and_provenance(monkeypatch):
    from scview.core import pipeline as P

    calls, loaded = _fake_celltypist(monkeypatch, HUMAN[:20] + [f"MISSING{i}" for i in range(80)])
    monkeypatch.setattr(P, "_markers_for_column", lambda adata, col: None)
    a = _counts(HUMAN)
    params = P.PipelineParams(annotation_method="celltypist", annotation_target="ct",
                              annotation_groupby="cluster")
    ret = P._run_cell_type_annotation(a, params)
    assert calls["model"] is loaded  # loaded once, for the check and the prediction
    assert ret == {"dataset_species": "human", "model_species": "human",
                   "n_model_genes": 100, "n_shared_genes": 20, "gene_overlap": 0.2}
    s = fit.summarize(a, "ct")
    assert s["model"] == "Immune_All_Low.pkl" and s["gene_overlap"] == 0.2
    assert any("only 20 of the model's 100 genes (20%)" in c for c in s["cautions"])

    P._record_pipeline_step(a, "cell_type_annotation", params, ret)
    rec = provenance.read_provenance(a)["history"][-1]
    assert rec["params"]["gene_overlap"] == 0.2 and rec["params"]["model_species"] == "human"


def test_step_with_no_shared_gene_says_why(monkeypatch):
    from scview.core import pipeline as P

    _fake_celltypist(monkeypatch, HUMAN)  # a human model
    mouse = [g.capitalize() for g in HUMAN[:60]] + ["mt-Co1", "mt-Nd1"]
    a = _counts(mouse)
    with pytest.raises(ValueError, match="None of the 62 genes.*mouse dataset"):
        P._annotate_celltypist(a, P.PipelineParams(celltypist_model="Immune_All_Low"),
                               "cluster", "ct")
    assert "ct" not in a.obs


def test_another_method_forgets_the_celltypist_run(monkeypatch):
    from scview.core import pipeline as P

    _fake_celltypist(monkeypatch, HUMAN)
    monkeypatch.setattr(P, "_markers_for_column", lambda adata, col: None)
    a = _counts(HUMAN)
    P._annotate_celltypist(a, P.PipelineParams(), "cluster", "ct")
    assert fit.run_columns(a) == ["ct"]

    def marker_score(adata, params, groupby, target):
        adata.obs[target] = pd.Categorical(["T cells"] * adata.n_obs)

    monkeypatch.setattr(P, "_annotate_marker_score", marker_score)
    ret = P._run_cell_type_annotation(a, P.PipelineParams(
        annotation_method="marker_score", annotation_groupby="cluster", annotation_target="ct"))
    assert ret is None and fit.run_columns(a) == []


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


def _save(dataset_manager, a: ad.AnnData) -> None:
    d = Path(dataset_manager.data_dir) / "uploads" / "ds"
    d.mkdir(parents=True, exist_ok=True)
    (d / "metadata.json").write_text(json.dumps({"id": "ds", "name": "ds", "status": "ready"}))
    a.write_h5ad(d / "data.h5ad")


async def test_summary_endpoint(client, dataset_manager):
    a = _labelled()
    a.var_names = ["CD3E", "MS4A1"]
    fit.record_run(a, "ct", {"model": "Immune_All_Low.pkl", "model_species": "human",
                             "dataset_species": "human", "gene_overlap": 0.9})
    _save(dataset_manager, a)
    r = await client.get("/api/v1/datasets/ds/annotation/celltypist")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["columns"] == ["ct"] and body["result"] is None
    body = (await client.get("/api/v1/datasets/ds/annotation/celltypist?column=ct")).json()
    assert [row["label"] for row in body["result"]["labels"]] == ["T", "B", "NK"]
    assert body["result"]["n_low_labels"] == 1
    assert (await client.get("/api/v1/datasets/missing/annotation/celltypist")).status_code == 404


async def test_model_catalog_names_each_models_species(client, monkeypatch):
    from scview.api.v1 import annotation

    monkeypatch.setattr(annotation, "_model_catalog", lambda: [
        ("Immune_All_Low.pkl", "immune sub-populations"),
        ("Mouse_Whole_Brain.pkl", "cell types from the whole adult mouse brain"),
    ])
    body = (await client.get("/api/v1/annotation/celltypist-models")).json()
    assert [(m["model"], m["species"]) for m in body["models"]] == [
        ("Immune_All_Low.pkl", "human"), ("Mouse_Whole_Brain.pkl", "mouse")]
