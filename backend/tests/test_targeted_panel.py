"""A targeted panel's run: defaults, skips, records, replay and large mode (core/targeted_panel.py)."""

from __future__ import annotations

import json
import subprocess
import sys

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from fastapi.testclient import TestClient

from scview.core import (
    assessor,
    doublets,
    large_run,
    provenance,
    qc_filter,
    replay_script,
    spatial,
    targeted_panel as tp,
)
from scview.core.llm_advisor import get_rule_based_suggestions
from scview.core.ora_background import get_background_genes
from scview.core.pipeline import PipelineParams, _annotation_prompt, run_pipeline
from scview.core.rag.router import heuristic_intent, named_result_facets
from scview.core.species import detect_species

FULL = ["qc_metrics", "doublet_detection", "filtering", "normalization", "log_transform",
        "highly_variable_genes", "scaling", "pca", "neighbors", "clustering", "embeddings",
        "cell_cycle", "dissociation_stress"]


def panel(n_cells: int = 400, n_genes: int = 40, seed: int = 0, recorded: bool = True) -> ad.AnnData:
    """Counts in two groups on a panel with no MT-, RP, cell-cycle or stress genes, some cells
    under 10 counts; coordinates in microns; recorded as a Xenium import does."""
    rng = np.random.default_rng(seed)
    groups = rng.integers(0, 2, n_cells)
    rate = np.tile(rng.gamma(1.0, 1.5, n_genes), (n_cells, 1))
    for g in range(2):
        rate[groups == g, g * 8:(g + 1) * 8] *= 5.0
    counts = rng.poisson(rate).astype(np.float32)
    low = np.arange(12)
    counts[low] = 0
    counts[low, 0] = np.arange(1, 13)  # totals 1..12: under the floor up to 9
    names = [f"GENE{i}" for i in range(n_genes)]
    a = ad.AnnData(X=sp.csr_matrix(counts), var=pd.DataFrame(index=names),
                   obs=pd.DataFrame({"library_id": pd.Categorical(["s1"] * n_cells)},
                                    index=[str(i + 1) for i in range(n_cells)]))
    a.obs["transcript_counts"] = counts.sum(axis=1).astype(int)
    a.obs["control_probe_counts"] = 0
    a.obs["total_counts"] = a.obs["transcript_counts"] + a.obs["control_probe_counts"]
    a.obs["pct_control"] = np.float32(0.0)
    a.obsm["spatial"] = rng.uniform(0, 1000, (n_cells, 2))
    a.uns["spatial"] = {"s1": {"images": {}, "scalefactors": {}, "metadata": {"source": "xenium"}}}
    if recorded:
        a.uns["scview_spatial"] = {"technology": "xenium", "unit": "cell", "targeted_panel": True,
                                   "panel_name": "P", "n_panel_genes": n_genes, "pixel_size": 0.2125,
                                   "organism": "human"}
    return a


def _history(a) -> dict[str, dict]:
    return {h["step"]: h["params"] for h in provenance.read_provenance(a)["history"]}


# ---------------------------------------------------------------------------
# Which datasets are panels
# ---------------------------------------------------------------------------


def test_recorded_flag_or_cells_with_few_genes():
    assert spatial.targeted_panel(panel())
    # CosMx, MERSCOPE, an h5ad from spatialdata-io: cell coordinates and few genes.
    h5ad = panel(recorded=False)
    spatial.adopt(h5ad)
    assert spatial.technology(h5ad)["technology"] == "cells" and spatial.targeted_panel(h5ad)
    # Whole-transcriptome spatial cells, and no coordinates at all, are not.
    wide = ad.AnnData(X=sp.csr_matrix((5, 6001), dtype=np.float32))
    wide.obsm["spatial"] = np.zeros((5, 2))
    assert not spatial.targeted_panel(wide)
    assert not spatial.targeted_panel(ad.AnnData(X=sp.csr_matrix((5, 100), dtype=np.float32)))
    no = panel()
    no.uns["scview_spatial"]["targeted_panel"] = False
    assert not spatial.targeted_panel(no)
    # Variable genes kept in X, the whole transcriptome in .raw: no panel.
    hvg = ad.AnnData(X=sp.csr_matrix((5, 7000), dtype=np.float32))
    hvg.raw = hvg
    hvg = hvg[:, :2000].copy()
    hvg.obsm["spatial"] = np.zeros((5, 2))
    assert not spatial.targeted_panel(hvg)


def test_a_dataset_that_is_no_panel_keeps_every_default():
    a = ad.AnnData(X=sp.csr_matrix((5, 100), dtype=np.float32))
    p = PipelineParams()
    assert tp.for_run(a, p) is p
    assert p.target_sum == 1e4 and p.min_genes == 200 and p.qc_counts_floor == 0 and not p.hvg_all_genes


@pytest.mark.parametrize("explicit,kept", [
    (("target_sum",), "target_sum_median"), (("min_genes",), "min_genes"), (("qc_counts_floor",), "qc_counts_floor"),
    (("n_top_genes",), "hvg_all_genes"), (("hvg_flavor",), "hvg_all_genes"), (("n_comps",), "n_comps"),
])
def test_what_the_user_set_always_wins(explicit, kept):
    assert kept in tp.defaults(40)
    assert kept not in tp.defaults(40, explicit)


def test_fixed_mode_min_counts_of_the_user_replaces_the_floor():
    assert "qc_counts_floor" not in tp.defaults(40, ("min_counts",), "fixed")
    assert tp.defaults(40, ("min_counts",), "mad")["qc_counts_floor"] == tp.COUNTS_FLOOR


def test_large_panel_still_selects_and_pca_stays_below_the_genes():
    assert "hvg_all_genes" not in tp.defaults(5000) and "n_comps" not in tp.defaults(5000)
    assert tp.defaults(2000)["hvg_all_genes"]
    assert tp.defaults(40)["n_comps"] == 20
    assert "n_comps" not in tp.defaults(313)


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


def test_the_default_run_follows_the_panel_path_and_records_it():
    a = panel()
    totals_before = np.asarray(a.X.sum(axis=1)).ravel()
    out, res = run_pipeline(a, FULL, PipelineParams(umap_3d="off"))
    assert res.status == "complete", res
    hist = _history(out)
    # The floor: every cell under 10 counts is gone, whatever the MAD bound.
    assert out.n_obs == a.n_obs - int((totals_before < 10).sum()) or out.n_obs < a.n_obs
    assert np.asarray(out.layers["counts"].sum(axis=1)).min() >= 10
    assert hist["filtering"]["qc_counts_floor"] == 10
    assert hist["filtering"]["qc_thresholds"]["pooled"]["min_counts"] >= 10
    # Normalised to the kept cells' median, recorded as the value.
    kept = np.asarray(out.layers["counts"].sum(axis=1)).ravel()
    assert hist["normalization"]["target_sum"] == pytest.approx(np.median(kept))
    assert "median" in hist["normalization"]["target_sum_rule"]
    assert np.allclose(np.expm1(out.X[:50].toarray() if sp.issparse(out.X) else out.X[:50]).sum(axis=1),
                       np.median(kept), rtol=1e-3)
    # All genes, 20 PCs.
    assert out.var["highly_variable"].all() and hist["highly_variable_genes"]["hvg_flavor"] == "all_genes"
    assert hist["highly_variable_genes"]["n_top_genes"] == out.n_vars
    assert out.obsm["X_pca"].shape[1] == 20
    # Skipped with their sentence, recorded.
    assert res.notes["doublet_detection"] == tp.DOUBLET_SKIP
    assert "under half" in res.notes["cell_cycle"] and "under half" in res.notes["dissociation_stress"]
    assert hist["doublet_detection_skipped"]["reason"] == tp.DOUBLET_SKIP
    assert not {"doublet_score", "S_score", "dissociation_stress_score"} & set(out.obs.columns)
    assert not res.errors if hasattr(res, "errors") else True


def test_explicit_parameters_beat_the_panel_and_history_says_what_ran():
    a = panel()
    p = PipelineParams(target_sum=500.0, n_top_genes=10, qc_counts_floor=0.0, n_comps=5, umap_3d="off",
                       explicit=("target_sum", "n_top_genes", "qc_counts_floor", "n_comps"))
    out, res = run_pipeline(a, FULL, p)
    assert res.status == "complete"
    hist = _history(out)
    assert hist["normalization"]["target_sum"] == 500.0 and "target_sum_rule" not in hist["normalization"]
    assert 10 <= int(out.var["highly_variable"].sum()) <= 12  # ties in scanpy's bins
    assert out.obsm["X_pca"].shape[1] == 5
    assert "qc_counts_floor" not in hist["filtering"]


def test_fixed_mode_has_no_gene_floor_and_a_counts_floor():
    a = panel()
    out, _ = run_pipeline(a, ["qc_metrics", "filtering"], PipelineParams(qc_threshold_mode="fixed"))
    hist = _history(out)["filtering"]
    assert hist["min_genes"] == 0 and hist["min_counts"] == 10
    assert np.asarray(out.X.sum(axis=1)).min() >= 10
    assert out.n_obs > 300  # a gene floor of 200 would have taken all of them


def test_min_cells_still_applies():
    a = panel()
    x = a.X.toarray()
    x[:, 39] = 0
    a.X = sp.csr_matrix(x)
    out, _ = run_pipeline(a, ["qc_metrics", "filtering"])
    assert "GENE39" not in out.var_names


def test_a_panel_with_most_of_the_stress_genes_is_scored():
    a = panel(n_genes=40)
    from scview.core.stress import STRESS_GENES

    a.var_names = list(STRESS_GENES[:12]) + [f"GENE{i}" for i in range(28)]
    assert tp.stress_skip(a) is None
    b = panel(n_genes=40)
    b.var_names = list(STRESS_GENES[:10]) + [f"GENE{i}" for i in range(30)]
    assert "10 of the 22" in tp.stress_skip(b)


def test_a_small_panel_keeps_pca_sane():
    a = panel(n_genes=12, n_cells=300)
    out, res = run_pipeline(a, ["qc_metrics", "filtering", "normalization", "log_transform",
                                "highly_variable_genes", "scaling", "pca"])
    assert res.status == "complete"
    assert out.obsm["X_pca"].shape[1] == 6 < out.n_vars


def test_regressing_out_the_cell_cycle_says_why_it_cannot():
    a = panel()
    p = PipelineParams(regress_out=["S_score", "G2M_score"], explicit=("regress_out",))
    out, res = run_pipeline(a, ["qc_metrics", "filtering", "normalization", "log_transform",
                                "highly_variable_genes", "scaling"], p)
    assert res.status == "finished_with_errors"
    assert "under half" in json.dumps(res.to_dict(), default=str)


def test_ambient_correction_does_not_apply():
    out, res = run_pipeline(panel(), ["ambient_correction"])
    assert res.notes["ambient_correction"] == tp.AMBIENT_SKIP
    assert _history(out)["ambient_correction_skipped"]["reason"] == tp.AMBIENT_SKIP


def test_the_preview_removes_what_the_step_removes():
    a = panel()
    from scview.core.anndata_adaptor import per_cell_qc

    p = tp.for_run(a, PipelineParams())
    t = qc_filter.Thresholds.from_params(p)
    assert t.counts_floor == 10
    preview = qc_filter.preview(a, per_cell_qc(a), t)
    out, _ = run_pipeline(a.copy(), ["filtering"])
    assert preview["n_removed"] == a.n_obs - out.n_obs >= 9


def test_totals_before_qc_metrics_are_the_genes_not_the_controls():
    from scview.core.anndata_adaptor import per_cell_qc

    a = panel()
    a.obs["total_counts"] = a.obs["transcript_counts"] + 5  # Xenium's total holds its controls
    assert np.array_equal(per_cell_qc(a)["total_counts"], a.obs["transcript_counts"].to_numpy(dtype=float))
    out, _ = run_pipeline(a, ["qc_metrics"])
    assert np.array_equal(per_cell_qc(out)["total_counts"], out.obs["transcript_counts"].to_numpy(dtype=float))


# ---------------------------------------------------------------------------
# Replay and large mode
# ---------------------------------------------------------------------------


def test_the_replay_script_makes_the_same_choices(tmp_path):
    a = panel()
    src = tmp_path / "in.h5ad"
    a.write_h5ad(src)
    out, _ = run_pipeline(a, ["qc_metrics", "doublet_detection", "filtering", "normalization", "log_transform",
                              "highly_variable_genes", "scaling", "pca"], PipelineParams(umap_3d="off"))
    script = replay_script.build_script(out)
    target = _history(out)["normalization"]["target_sum"]
    assert f"sc.pp.normalize_total(adata, target_sum={float(target)!r})" in script
    assert 'adata.var["highly_variable"] = True' in script
    assert f"# doublet_detection was skipped: {tp.DOUBLET_SKIP}." in script
    assert "highly_variable_genes(adata, n_top_genes" not in script
    assert '"counts_floor": 10.0' in script or "'counts_floor': 10.0" in script

    # Run it: the same cells, target and PCs.
    ran = tmp_path / "out.h5ad"
    path = tmp_path / "replay.py"
    path.write_text(script)
    proc = subprocess.run([sys.executable, str(path), str(src), str(ran)], capture_output=True, text=True,
                          timeout=300)
    assert proc.returncode == 0, proc.stderr
    back = ad.read_h5ad(ran)
    assert back.n_obs == out.n_obs and back.n_vars == out.n_vars
    assert back.var["highly_variable"].all()
    assert np.allclose(np.abs(back.obsm["X_pca"][:, 0]), np.abs(out.obsm["X_pca"][:, 0]), atol=1e-3)


def test_large_mode_gives_the_in_memory_answer(tmp_path):
    steps = ["qc_metrics", "doublet_detection", "filtering", "normalization", "log_transform",
             "highly_variable_genes", "scaling", "pca", "neighbors", "clustering", "cell_cycle", "dissociation_stress"]
    a = panel(n_cells=600, n_genes=60)
    src = tmp_path / "x.h5ad"
    a.write_h5ad(src)
    mem, mem_res = run_pipeline(ad.read_h5ad(src), steps, PipelineParams())
    res = large_run.run(src, steps, PipelineParams(chunked="on"), str(tmp_path / "out.h5ad"))
    assert res.status == "complete", res
    assert res.notes["doublet_detection"] == tp.DOUBLET_SKIP
    assert res.notes.keys() == mem_res.notes.keys()
    big = ad.read_h5ad(tmp_path / "out.h5ad", backed="r")
    assert big.n_obs == mem.n_obs and big.n_vars == mem.n_vars
    assert big.obs_names.tolist() == mem.obs_names.tolist()
    hist, hist_mem = _history(big), _history(mem)
    assert hist["normalization"]["target_sum"] == hist_mem["normalization"]["target_sum"]
    assert hist["normalization"]["target_sum_rule"] == hist_mem["normalization"]["target_sum_rule"]
    assert hist["highly_variable_genes"]["hvg_flavor"] == "all_genes"
    assert hist["filtering"]["qc_counts_floor"] == 10
    x = big.X[:, :]
    assert np.allclose(x.toarray(), mem.X.toarray(), atol=1e-4)
    # PCs up to sign (the chunked PCA differs at float precision).
    pa, pb = np.asarray(big.obsm["X_pca"]), np.asarray(mem.obsm["X_pca"])
    assert pa.shape == pb.shape
    for i in range(5):
        assert abs(np.corrcoef(pa[:, i], pb[:, i])[0, 1]) > 0.999


# ---------------------------------------------------------------------------
# Assessment, species, advisor, co-pilot
# ---------------------------------------------------------------------------


def test_the_assessment_recommends_the_panel_path():
    a = panel()
    s = assessor.assess_preprocessing(a)
    for step in (s.doublet_detection, s.cell_cycle, s.dissociation_stress, s.ambient_correction):
        assert step.done and step.details.startswith("Not applicable")
    assert doublets.summary(a)["not_applicable"] == tp.DOUBLET_SKIP
    info = tp.assessment(a)
    assert info["n_genes"] == 40 and info["all_genes"]
    assert info["sentences"][0].startswith("Targeted panel of 40 genes")
    assert info["defaults"]["min_genes"] == 0 and info["defaults"]["qc_counts_floor"] == 10
    assert info["defaults"]["target_sum"] == round(np.median(a.obs["total_counts"]))
    assert info["control"]["flag"] is None
    assert tp.assessment(ad.AnnData(X=sp.csr_matrix((5, 100), dtype=np.float32))) is None


def test_a_high_control_share_is_pointed_at():
    a = panel()
    a.obs["pct_control"] = np.where(np.arange(a.n_obs) < 40, 30.0, 0.0)
    flag = tp.assessment(a)["control"]["flag"]
    assert "s1" in flag and "control" in flag


def test_species_comes_from_the_panel_not_the_names():
    a = panel()
    a.var_names = [f"Gene{i}" for i in range(a.n_vars)]  # title case: would read as mouse
    call = detect_species(a)
    assert call.species == "human" and call.method == "recorded" and "experiment.xenium" in call.evidence
    a.uns["scview_spatial"]["organism"] = "zebrafish"
    assert detect_species(a).method != "recorded"


def test_rule_based_suggestions_fit_a_panel():
    state = assessor.assess_preprocessing(panel()).model_dump()
    state["doublet_detection"]["done"] = False  # the rule must drop it by itself too
    res = get_rule_based_suggestions(state, {"n_cells": 400, "n_genes": 40, "targeted_panel": 40})
    steps = {s.step: s for s in res.suggestions}
    assert not {"doublet_detection", "cell_cycle", "dissociation_stress", "highly_variable_genes"} & set(steps)
    assert "target_sum" not in steps["normalization"].suggested_params
    assert "median" in steps["normalization"].reasoning


def test_llm_prompt_names_the_panel_size():
    line = tp.prompt_line(panel())
    assert "targeted panel of 40 genes" in line
    assert line in _annotation_prompt("0: A, B", "breast", line)
    assert tp.prompt_line(ad.AnnData(X=sp.csr_matrix((5, 100), dtype=np.float32))) == ""


def test_grounding_says_the_panel_and_the_router_finds_it():
    lines, detail = spatial.grounding_lines(panel())
    assert "targeted panel of 40 genes" in lines[0] and "[result:tissue]" in lines[0]
    assert "targeted panel of 40 genes" in detail
    assert "qc" in named_result_facets("what is on the gene panel?")
    assert "data" in heuristic_intent("how many genes are in the targeted panel?").sources


def test_no_scary_notes_for_genes_a_panel_does_not_have():
    from scview.core import cluster_qc, mito_by_type, qc_states
    from scview.core.anndata_adaptor import per_cell_qc

    a = panel()
    a.obs["leiden"] = pd.Categorical([str(i % 3) for i in range(a.n_obs)])
    assert "as expected" in mito_by_type.mito_by_type(a, "leiden", per_cell_qc(a, ribo=True)).note
    with pytest.raises(qc_states.QcStatesError, match="as expected"):
        qc_states.compute(a, "leiden", {**per_cell_qc(a), "pct_ribo": None})
    report = cluster_qc.cluster_qc(a, "leiden") if hasattr(cluster_qc, "cluster_qc") else None
    if report is not None:
        assert "as expected" in report.note


def test_the_qc_report_has_no_200_gene_concern():
    from scview.core.qc_report import build_facts

    class A:
        adata = panel()

    f = build_facts(A())
    words = json.dumps([x.model_dump() if hasattr(x, "model_dump") else str(x)
                        for x in [*f.overview, *f.quality, *f.concerns, *f.next_steps]], default=str)
    assert "200 genes" not in words and "Targeted panel of 40 genes" in words


# ---------------------------------------------------------------------------
# Enrichment: the panel is the background, and the answer says so
# ---------------------------------------------------------------------------


def test_the_ora_background_is_the_panel():
    a = panel()
    a.uns["rank_genes_groups"] = {"names": np.rec.fromarrays([np.array(["GENE1", "GENE2"])], names=["0"])}
    a.obs["leiden"] = pd.Categorical(["0"] * a.n_obs)
    assert get_background_genes(a, "leiden", "0") == a.var_names.tolist()


class _Adaptor:
    def __init__(self, adata):
        self.adata = adata
        self.is_large = False

    def gene_names(self):
        return list(self.adata.var_names)


def _client(adata) -> TestClient:
    from fastapi import FastAPI

    from scview.api.v1 import genesets
    from scview.dependencies import get_dataset_manager

    class DM:
        async def get_or_load_dataset(self, _):
            return _Adaptor(adata)

    app = FastAPI()
    app.include_router(genesets.router, prefix="/api/v1")
    app.dependency_overrides[get_dataset_manager] = lambda: DM()
    return TestClient(app)


def test_scoring_carries_the_caution_and_is_not_refused():
    a = panel()
    a.X = a.X.copy()
    a.X.data = np.log1p(a.X.data)
    res = _client(a).post("/api/v1/datasets/x/genesets/score",
                          json={"gene_set": ["GENE1", "GENE2"], "score_name": "s", "method": "aucell"})
    assert res.status_code == 200, res.text
    assert res.json()["panel_caution"] == tp.ENRICHMENT_CAUTION


def test_no_caution_on_other_data():
    assert tp.enrichment_caution(ad.AnnData(X=sp.csr_matrix((5, 100), dtype=np.float32))) is None
    assert tp.enrichment_caution(panel()) == tp.ENRICHMENT_CAUTION


def test_hvg_sensitivity_has_nothing_to_vary_on_all_genes():
    from scview.core import hvg_sensitivity

    out, _ = run_pipeline(panel(), ["qc_metrics", "filtering", "normalization", "log_transform",
                                    "highly_variable_genes"])
    out.obs["leiden"] = pd.Categorical([str(i % 3) for i in range(out.n_obs)])
    with pytest.raises(hvg_sensitivity.SensitivityError, match="keeps all its genes"):
        hvg_sensitivity.recorded(out, "leiden")


def test_the_estimates_leave_the_skipped_doublet_step_out():
    from scview.core import preflight

    a = panel(n_cells=2000, n_genes=300)
    assert preflight.step_peak_for(a, "doublet_detection") == 0
    assert preflight.run_need(a, ["doublet_detection"]) == preflight.run_need(a, [])


def test_a_request_says_what_the_user_set():
    from scview.api.v1.assessment import _run_params

    p = _run_params({"target_sum": 1e4, "not_a_param": 1, "explicit": ["n_comps"]})
    assert p.explicit == ("target_sum",)
    assert _run_params(None).explicit == ()


def test_the_filter_preview_endpoint_applies_the_floor_as_the_run_does():
    from fastapi import FastAPI

    from scview.api.v1 import assessment
    from scview.dependencies import get_dataset_manager

    a = panel()

    class Adaptor(_Adaptor):
        def qc_per_cell(self, ribo=False):
            from scview.core.anndata_adaptor import per_cell_qc

            return per_cell_qc(self.adata, ribo=ribo)

    class DM:
        async def get_or_load_dataset(self, _):
            return Adaptor(a)

    app = FastAPI()
    app.include_router(assessment.router, prefix="/api/v1")
    app.dependency_overrides[get_dataset_manager] = lambda: DM()
    client = TestClient(app)
    url = "/api/v1/datasets/x/qc/filter-preview"
    out, _ = run_pipeline(a.copy(), ["filtering"])
    mad = client.get(url, params={"min_genes": 0, "min_counts": 10, "mode": "mad"}).json()
    assert mad["n_removed"] == a.n_obs - out.n_obs
    none = client.get(url, params={"min_genes": 0, "min_counts": 0, "mode": "fixed", "max_pct_mt": 100}).json()
    assert none["n_removed"] == 0  # the user's own min_counts of 0 in fixed mode wins
    fixed = client.get(url, params={"min_genes": 0, "min_counts": 10, "mode": "fixed", "max_pct_mt": 100}).json()
    assert fixed["n_removed"] == 9
    off = client.get(url, params={"min_genes": 0, "min_counts": 0, "mode": "mad", "counts_floor": 5}).json()
    assert off["n_removed"] < mad["n_removed"] or off["n_removed"] == mad["n_removed"]
