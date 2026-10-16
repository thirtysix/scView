"""Dissociation-stress score and the per-cluster "stress-driven" flag (C053).

A cluster is flagged only when its stress score is an outlier AND its own top
markers are stress genes. A cluster that merely expresses a few of them at
baseline (as monocytes do) scores high but leads with its own markers: no flag.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd

from scview.core import provenance, qc_report
from scview.core.anndata_adaptor import AnnDataAdaptor
from scview.core.assessor import assess_preprocessing
from scview.core.cluster_qc import cluster_qc, grounding_block
from scview.core.pipeline import PipelineParams, run_pipeline, run_pipeline_streamed
from scview.core.rerun import plan_rerun
from scview.core.stress import SCORE_COLUMN, STRESS_GENES
from scview.core.uns_keys import markers_key

API = "/api/v1"
N_PER = 120
N_CLUSTERS = 5
OWN = 30  # marker genes per cluster
STEPS = ["normalization", "log_transform", "marker_genes", "dissociation_stress"]
PARAMS = PipelineParams(marker_columns=["leiden"])


def _dataset(*, stressed: bool = True, monocyte_like: bool = True, mouse: bool = False,
             stress_genes: tuple[str, ...] = STRESS_GENES) -> ad.AnnData:
    """Raw counts. Cluster 4 is stress-driven (every stress gene high); cluster 3
    expresses six stress genes moderately but leads with strong markers of its own."""
    rng = np.random.default_rng(0)
    own = [f"MK{k}X{i}" for k in range(N_CLUSTERS) for i in range(OWN)]
    filler = [f"GENE{i}" for i in range(250)]
    names = list(stress_genes) + own + filler
    n_s = len(stress_genes)
    X = rng.poisson(1.0, (N_PER * N_CLUSTERS, len(names))).astype("float32")
    X[:, :n_s] = rng.poisson(0.5, (X.shape[0], n_s))
    for k in range(N_CLUSTERS):
        rows = slice(k * N_PER, (k + 1) * N_PER)
        cols = slice(n_s + k * OWN, n_s + (k + 1) * OWN)
        X[rows, cols] += rng.poisson(12.0 if k == 3 else 4.0, (N_PER, OWN))
    if monocyte_like:
        X[3 * N_PER: 4 * N_PER, :6] += rng.poisson(3.0, (N_PER, 6))
    if stressed:
        X[4 * N_PER:, :n_s] += rng.poisson(6.0, (N_PER, n_s))
    a = ad.AnnData(X)
    a.var_names = [n[:1] + n[1:].lower() for n in names] if mouse else names
    a.obs_names = [f"C{i}" for i in range(a.n_obs)]
    labels = [str(k) for k in range(N_CLUSTERS) for _ in range(N_PER)]
    a.obs["leiden"] = pd.Categorical(labels, categories=[str(k) for k in range(N_CLUSTERS)])
    return a


def _scored(**kw) -> ad.AnnData:
    out, res = run_pipeline(_dataset(**kw), STEPS, PARAMS)
    assert res.errors == {}, res.errors
    return out


def _rows(report):
    return {c.group: c for c in report.clusters}


# ---------------------------------------------------------------------------
# The score
# ---------------------------------------------------------------------------


def test_step_writes_a_per_cell_score_and_records_what_it_used():
    out = _scored()
    assert SCORE_COLUMN in out.obs.columns
    score = out.obs[SCORE_COLUMN].to_numpy()
    assert np.median(score[4 * N_PER:]) > np.median(score[:3 * N_PER]) + 0.5
    step = [h for h in provenance.read_provenance(out)["history"]
            if h["step"] == "dissociation_stress"][-1]
    assert step["params"]["species"] == "human"
    assert step["params"]["n_genes"] == len(STRESS_GENES)
    assert assess_preprocessing(out).dissociation_stress.done is True


def test_mouse_gene_names_are_scored_too():
    out, res = run_pipeline(_dataset(mouse=True), STEPS, PARAMS)
    assert res.errors == {} and SCORE_COLUMN in out.obs.columns
    step = provenance.read_provenance(out)["history"][-1]
    assert step["params"]["species"] == "mouse"
    assert "Fos" in step["params"]["genes"] and "Hspa1a" in step["params"]["genes"]


def test_too_few_signature_genes_skips_the_step_without_an_error():
    a = _dataset(stress_genes=STRESS_GENES[:5])
    out, res = run_pipeline(a, STEPS, PARAMS)
    assert res.errors == {}
    assert "dissociation_stress" not in res.steps_run
    assert "Only 5 of the 22" in res.notes["dissociation_stress"]
    assert SCORE_COLUMN not in out.obs.columns
    assert "dissociation_stress" not in [h["step"] for h in provenance.read_provenance(out)["history"]]


def test_streamed_run_reports_a_skipped_step_as_done_with_its_reason():
    gen = run_pipeline_streamed(_dataset(stress_genes=STRESS_GENES[:5]), STEPS, PARAMS)
    events = list(gen)
    done = [d for t, d in events if t == "step_done" and d["step"] == "dissociation_stress"]
    assert done and done[0]["skipped"] is True and "Only 5" in done[0]["note"]
    complete = [d for t, d in events if t == "complete"][0]
    assert "dissociation_stress" in complete["notes"] and complete["errors"] == {}


def test_raw_counts_are_refused():
    out, res = run_pipeline(_dataset(), ["dissociation_stress"], PARAMS)
    assert "log-normalised" in res.errors["dissociation_stress"]


def test_score_is_rerunnable_in_place():
    plan = plan_rerun(["normalization", "log_transform", "dissociation_stress"], "dissociation_stress")
    assert plan.rerun_steps == ["dissociation_stress"] and not plan.requires_reprocess


def test_score_is_one_of_the_qc_plots(tmp_path: Path):
    p = tmp_path / "s.h5ad"
    _scored().write_h5ad(p)
    assert SCORE_COLUMN in AnnDataAdaptor(str(p)).qc_distributions()["metrics"]


# ---------------------------------------------------------------------------
# The per-cluster flag
# ---------------------------------------------------------------------------


def test_a_stress_driven_cluster_is_flagged_and_a_monocyte_like_one_is_not():
    report = cluster_qc(_scored(), "leiden")
    rows = _rows(report)
    assert "stress_score" in report.metrics
    assert rows["4"].flags == ["stress_driven"]
    assert len(rows["4"].stress_markers) >= 3
    assert "dissociation stress" in rows["4"].reasons[0]
    assert "stress genes (" in rows["4"].reasons[0]
    # Cluster 3 scores as an outlier too, but its top markers are its own genes.
    typical = report.typical["median_stress_score"]
    assert rows["3"].median_stress_score - typical > 0.15
    assert rows["3"].stress_markers == [] and rows["3"].flags == []
    assert all(rows[g].flags == [] for g in "012")


def test_a_normal_dataset_has_no_stress_flag():
    report = cluster_qc(_scored(stressed=False, monocyte_like=False), "leiden")
    assert all("stress_driven" not in c.flags for c in report.clusters)
    lines, _ = grounding_block(report)
    assert any("dissociation stress" in ln and "No cluster stands out" in ln for ln in lines)


def test_without_markers_an_outlier_is_noted_not_flagged():
    a = _scored()
    del a.uns[markers_key("leiden")]
    a.uns.pop("rank_genes_groups", None)
    report = cluster_qc(a, "leiden")
    assert all("stress_driven" not in c.flags for c in report.clusters)
    # Both score outliers are named: only markers tell cluster 3 and 4 apart.
    assert "Clusters 3, 4 have a high dissociation-stress score" in report.note
    assert "no marker genes" in report.note


def test_qc_report_raises_a_concern_and_a_next_step():
    facts = qc_report.build_facts(SimpleNamespace(adata=_scored()))
    concerns = " ".join(c.detail for c in facts.concerns)
    steps = " ".join(s.detail for s in facts.next_steps)
    assert "Cluster 4 stands out: stress driven." in concerns
    assert "dissociation_stress_score" in steps
    # A stress flag alone doesn't suggest tightening the QC filters.
    assert "tighten the QC filters" not in steps


async def test_cluster_qc_endpoint_returns_the_flag(client, dataset_manager):
    did = "stress"
    d = Path(dataset_manager.data_dir) / "uploads" / did
    d.mkdir(parents=True)
    (d / "metadata.json").write_text(json.dumps({"id": did, "name": did, "status": "ready"}))
    _scored().write_h5ad(d / "data.h5ad")
    r = await client.get(f"{API}/datasets/{did}/cluster-qc", params={"groupby": "leiden"})
    assert r.status_code == 200, r.text
    body = r.json()
    rows = {c["group"]: c for c in body["clusters"]}
    assert rows["4"]["flags"] == ["stress_driven"]
    assert rows["4"]["median_stress_score"] > body["typical"]["median_stress_score"]
    assert "stress_driven" in body["flag_help"]
