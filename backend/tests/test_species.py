"""Species detection (human / mouse / unknown) and where species changes a result.

Mouse genes are title case (Cd3e) where human ones are upper case (CD3E). Lists
written as human symbols must still find them: the cell-cycle step used to pass
the human spelling to scanpy, which found none of a mouse dataset's genes.
"""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scanpy as sc

from scview.config import Settings
from scview.core import provenance
from scview.core.anndata_adaptor import AnnDataAdaptor, per_cell_qc
from scview.core.assistant import build_grounding_context
from scview.core.llm_advisor import _build_user_message
from scview.core.pipeline import PipelineParams, run_pipeline
from scview.core.species import detect_species, human_genesets_warning, match_genes
from scview.core.uns_keys import markers_key

API = "/api/v1"

HUMAN = ["CD3E", "CD4", "MS4A1", "LYZ", "NKG7", "GAPDH", "ACTB", "MALAT1", "LINC00115",
         "C1orf112", "AC114488.2", "RP11-206L10.2"] + [f"GENE{i}" for i in range(80)]
MOUSE = ["Cd3e", "Cd4", "Ms4a1", "Lyz2", "Nkg7", "Gapdh", "Actb", "Malat1", "Gm12345",
         "1700012B09Rik", "AC114488.2"] + [f"Gene{i}" for i in range(80)]


def _adata(names, var=None) -> ad.AnnData:
    a = ad.AnnData(np.zeros((3, len(names)), dtype="float32"))
    a.var_names = names
    if var:
        for k, v in var.items():
            a.var[k] = v
    return a


# ---------------------------------------------------------------------------
# detect_species
# ---------------------------------------------------------------------------


def test_upper_case_symbols_are_human():
    call = detect_species(_adata(HUMAN + ["MT-CO1", "MT-ND1"]))
    assert call.species == "human" and call.method == "symbol_case"
    assert "upper case" in call.evidence and "'MT-' (2)" in call.evidence


def test_title_case_symbols_are_mouse():
    call = detect_species(_adata(MOUSE + ["mt-Co1"]))
    assert call.species == "mouse" and call.method == "symbol_case"
    assert "title case" in call.evidence and "'mt-' (1)" in call.evidence


def test_ensembl_var_names_decide_before_case():
    ids = [f"ENSMUSG{i:011d}" for i in range(100)]
    call = detect_species(_adata(ids))
    assert (call.species, call.method) == ("mouse", "ensembl")
    assert "ENSMUSG" in call.evidence
    human = detect_species(_adata([f"ENSG{i:011d}.5" for i in range(100)]))
    assert human.species == "human"


def test_ensembl_ids_in_a_gene_ids_column():
    # Symbols in var_names are upper case (misleadingly), but the ids say mouse.
    a = _adata(HUMAN, var={"gene_ids": [f"ENSMUSG{i:011d}" for i in range(len(HUMAN))]})
    call = detect_species(a)
    assert (call.species, call.method) == ("mouse", "ensembl")
    assert "var 'gene_ids'" in call.evidence


def test_other_or_mixed_ensembl_species_are_unknown():
    rat = detect_species(_adata([f"ENSRNOG{i:011d}" for i in range(100)]))
    assert rat.species == "unknown" and "another species" in rat.evidence
    mixed = detect_species(_adata([f"ENSG{i:011d}" for i in range(50)]
                                  + [f"ENSMUSG{i:011d}" for i in range(50)]))
    assert mixed.species == "unknown" and "mix species" in mixed.evidence


def test_lower_case_and_too_few_names_are_unknown():
    fish = detect_species(_adata([f"gene{i}" for i in range(80)]))
    assert fish.species == "unknown" and "lower case" in fish.evidence
    few = detect_species(_adata(["CD3E", "CD4"]))
    assert few.species == "unknown" and "Too few" in few.evidence


def test_mito_prefix_alone_is_enough_and_conflicts_are_unknown():
    assert detect_species(_adata(["Cd3e", "mt-Co1"])).species == "mouse"
    assert detect_species(_adata(["Cd3e", "mt-Co1"])).method == "mito_prefix"
    conflict = detect_species(_adata(MOUSE + ["MT-CO1"]))
    assert conflict.species == "unknown" and "conflicting" in conflict.evidence


def test_raw_var_names_are_used_when_x_holds_fewer_genes():
    a = _adata(MOUSE + ["mt-Co1", "mt-Nd1"])
    a.raw = a
    sub = a[:, :20].copy()  # X: only 20 "variable" genes, no mito
    assert "'mt-' (2)" in detect_species(sub).evidence


# ---------------------------------------------------------------------------
# match_genes
# ---------------------------------------------------------------------------


def test_match_genes_finds_the_datasets_spelling_and_aliases():
    names = ["Fos", "Jun", "Cenpu", "Mcm5"]
    got = match_genes(["FOS", "JUN", "MLF1IP", "EGR1", "MCM5"], names,
                      aliases={"MLF1IP": ("CENPU",)})
    assert got == ["Fos", "Jun", "Cenpu", "Mcm5"]
    # An old and a new name of one gene, both present, count once.
    assert match_genes(["HN1", "JPT1"], ["Jpt1"], aliases={"HN1": ("JPT1",)}) == ["Jpt1"]


def test_genesets_warning_only_for_mouse():
    assert "mouse" in human_genesets_warning(detect_species(_adata(MOUSE)))
    assert human_genesets_warning(detect_species(_adata(HUMAN))) == ""


# ---------------------------------------------------------------------------
# Pipeline: cell cycle and QC on a mouse dataset
# ---------------------------------------------------------------------------

# Mouse spellings of a subset of the Tirosh 2016 genes, with the renamed ones
# (Cenpu, Pimreg, Jpt1) under their current names.
MOUSE_S = ["Mcm5", "Pcna", "Tyms", "Fen1", "Mcm2", "Mcm4", "Rrm1", "Ung", "Gins2", "Mcm6",
           "Cdca7", "Dtl", "Prim1", "Uhrf1", "Cenpu", "Hells", "Rfc2", "Rpa2", "Nasp", "Gmnn"]
MOUSE_G2M = ["Hmgb2", "Cdk1", "Nusap1", "Ube2c", "Birc5", "Tpx2", "Top2a", "Ndc80", "Cks2",
             "Nuf2", "Mki67", "Tmpo", "Cenpf", "Tacc3", "Pimreg", "Smc4", "Ccnb2", "Jpt1",
             "Aurkb", "Bub1"]


def _mouse_cycling(n_per: int = 60) -> ad.AnnData:
    """Log-normalised mouse data: cells 0-59 in S, 60-119 in G2M, the rest resting."""
    rng = np.random.default_rng(0)
    filler = [f"Gene{i}" for i in range(300)]
    names = MOUSE_S + MOUSE_G2M + filler + ["mt-Co1", "mt-Nd1", "Rpl5", "Rps3"]
    X = rng.gamma(1.0, 0.5, (3 * n_per, len(names))).astype("float32")
    s, g = len(MOUSE_S), len(MOUSE_G2M)
    X[:n_per, :s] += 2.0
    X[n_per:2 * n_per, s:s + g] += 2.0
    a = ad.AnnData(X)
    a.var_names = names
    a.obs_names = [f"C{i}" for i in range(a.n_obs)]
    return a


def test_cell_cycle_scores_a_mouse_dataset_and_records_species():
    out, res = run_pipeline(_mouse_cycling(), ["cell_cycle"], PipelineParams())
    assert res.errors == {}, res.errors
    assert {"S_score", "G2M_score", "phase"} <= set(out.obs.columns)
    phase = out.obs["phase"].astype(str).to_numpy()
    assert (phase[:60] == "S").mean() > 0.8
    assert (phase[60:120] == "G2M").mean() > 0.8
    step = provenance.read_provenance(out)["history"][-1]
    assert step["step"] == "cell_cycle"
    assert step["params"]["species"] == "mouse"
    # Renamed genes are found under their current names (CENPU, PIMREG, JPT1).
    assert step["params"]["n_s_genes"] == len(MOUSE_S)
    assert step["params"]["n_g2m_genes"] == len(MOUSE_G2M)


def test_qc_metrics_count_mouse_mito_and_ribo_genes():
    rng = np.random.default_rng(1)
    names = [f"Gene{i}" for i in range(100)] + ["mt-Co1", "mt-Nd1", "Rpl5", "Rps3"]
    a = ad.AnnData(rng.poisson(2.0, (50, len(names))).astype("float32"))
    a.var_names = names
    cells = per_cell_qc(a, ribo=True)
    # 2 of 104 genes at the same rate: about 2% each, never all zero.
    assert cells["has_mt"] and cells["pct_mt"].mean() > 1.0
    assert cells["pct_ribo"] is not None and cells["pct_ribo"].mean() > 1.0
    out, res = run_pipeline(a, ["qc_metrics"], PipelineParams())
    assert res.errors == {} and float(out.obs["pct_counts_mt"].mean()) > 1.0


# ---------------------------------------------------------------------------
# Where species is surfaced: assessment, advisor, co-pilot, enrichment
# ---------------------------------------------------------------------------


def _register(dm, did: str, names: list[str], column: str = "cell_type") -> str:
    d = Path(dm.data_dir) / "uploads" / did
    d.mkdir(parents=True)
    (d / "metadata.json").write_text(json.dumps({"id": did, "name": did, "status": "ready"}))
    rng = np.random.default_rng(0)
    X = rng.poisson(1.0, (90, len(names))).astype("float32")
    X[:30, 0:10] += 8
    X[30:60, 10:20] += 8
    a = ad.AnnData(X)
    a.var_names = names
    a.obs_names = [f"C{i}" for i in range(90)]
    a.obs[column] = pd.Categorical(["T"] * 30 + ["B"] * 30 + ["NK"] * 30)
    sc.tl.rank_genes_groups(a, column, method="wilcoxon", pts=True, key_added=markers_key(column))
    a.write_h5ad(d / "data.h5ad")
    return did


MOUSE_GENES = [f"Gm{i}" for i in range(60)]
HUMAN_GENES = [f"GM{i}" for i in range(60)]


@pytest.fixture()
def msigdb_dir(tmp_path: Path) -> Path:
    d = tmp_path / "msigdb"
    d.mkdir()
    (d / "h.all.v2026.1.Hs.json").write_text(json.dumps({
        "HALLMARK_T_PROGRAM": {"geneSymbols": HUMAN_GENES[0:10]},
        "HALLMARK_B_PROGRAM": {"geneSymbols": HUMAN_GENES[10:20]},
    }))
    return d


@pytest.fixture()
def test_settings(tmp_data_dir: Path, msigdb_dir: Path) -> Settings:
    return Settings(DATA_DIR=str(tmp_data_dir), DEEPINFRA_API_KEY="", RAG_DATABASE_URL="",
                    MSIGDB_DIR=str(msigdb_dir))


async def test_assessment_response_carries_species(client, dataset_manager):
    did = _register(dataset_manager, "mouse", MOUSE_GENES)
    r = await client.get(f"{API}/datasets/{did}/assessment")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["species"]["species"] == "mouse"
    assert body["species"]["evidence"]
    assert body["qc_metrics"]["done"] is False  # the step entries are unchanged


async def test_suggest_puts_species_in_the_dataset_summary(client, dataset_manager, monkeypatch):
    import scview.api.v1.assessment as assessment_api
    from scview.core.llm_advisor import get_rule_based_suggestions

    seen: dict = {}

    def spy(preprocessing_state, dataset_summary=None):
        seen.update(dataset_summary or {})
        return get_rule_based_suggestions(preprocessing_state, dataset_summary)

    monkeypatch.setattr(assessment_api, "get_rule_based_suggestions", spy)
    did = _register(dataset_manager, "mouse", MOUSE_GENES)
    state = (await client.get(f"{API}/datasets/{did}/assessment")).json()
    r = await client.post(f"{API}/datasets/{did}/assessment/suggest",
                          json={"preprocessing_state": state})
    assert r.status_code == 200, r.text
    assert seen["species"] == "mouse"


def test_advisor_prompt_lists_species_once_and_not_as_a_step():
    state = {"qc_metrics": {"done": True, "confidence": "high", "details": "x"},
             "species": {"species": "mouse", "method": "symbol_case", "evidence": "..."}}
    msg = _build_user_message(state, {"n_cells": 10, "n_genes": 5, "species": "mouse"})
    assert "- Species: mouse" in msg
    assert "species: NOT DONE" not in msg and "qc_metrics: DONE" in msg


def test_copilot_dataset_context_names_the_species(tmp_path: Path):
    a = _mouse_cycling()
    p = tmp_path / "m.h5ad"
    a.write_h5ad(p)
    context, _ = build_grounding_context(AnnDataAdaptor(str(p)))
    assert "Species: mouse" in context


async def test_enrichment_on_a_mouse_dataset_warns_about_human_gene_sets(client, dataset_manager):
    did = _register(dataset_manager, "mouse", MOUSE_GENES)
    body = {"column": "cell_type", "group": "T", "n_genes": 10, "collections": ["h.all"]}
    r = await client.post(f"{API}/datasets/{did}/enrichment/compute-local", json=body)
    assert r.status_code == 200, r.text
    out = r.json()
    assert "mouse" in out["warning"] and "ortholog" in out["warning"]
    # Case-insensitive matching still finds the mouse genes in the human set.
    assert out["results"] and out["results"][0]["term"] == "HALLMARK_T_PROGRAM"
    # The cached replay carries the warning too.
    again = (await client.post(f"{API}/datasets/{did}/enrichment/compute-local", json=body)).json()
    assert again["source"] == "precomputed" and "mouse" in again["warning"]


async def test_enrichment_on_a_human_dataset_has_no_warning(client, dataset_manager):
    did = _register(dataset_manager, "human", HUMAN_GENES)
    body = {"column": "cell_type", "group": "T", "n_genes": 10, "collections": ["h.all"]}
    r = await client.post(f"{API}/datasets/{did}/enrichment/compute-local", json=body)
    assert r.status_code == 200, r.text
    assert r.json()["warning"] == ""
