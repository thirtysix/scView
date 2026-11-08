"""Marker qualifiers for shared and mixture labels (core/label_qualifiers.py), and the
annotator's low-quality answer with per-cluster QC notes (pipeline._annotate_llm)."""

from __future__ import annotations

import json
import sys
import types

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from scview.core import label_qualifiers as LQ
from scview.core import pipeline as P
from scview.core.uns_keys import markers_key

GENES = ["CD14", "LYZ", "ISG15", "IFIT3", "CD4", "RPS3", "GZMK", "CD3D", "MS4A1", "LINC00861", "ACTB"]
N = 40  # cells per cluster


def _expr(on: dict[str, float], rng) -> np.ndarray:
    """One cluster's log-like values: each gene in ``on`` detected in that share of cells
    at about 2.5, everything else at 0; ACTB everywhere."""
    x = np.zeros((N, len(GENES)), dtype=np.float32)
    for g, share in {**on, "ACTB": 1.0}.items():
        hit = rng.random(N) < share
        x[hit, GENES.index(g)] = 2.5 + rng.random(hit.sum()).astype(np.float32) * 0.2
    return x


def _data(labels: dict[str, str]) -> ad.AnnData:
    rng = np.random.default_rng(0)
    mono = {"CD14": 0.9, "LYZ": 0.95, "CD4": 0.7}
    blocks = {
        "0": mono,
        # Stimulated monocytes: interferon genes, ribosomal and lncRNA too, and CD4 gone
        "1": {**{k: v for k, v in mono.items() if k != "CD4"}, "ISG15": 0.9, "IFIT3": 0.8,
              "RPS3": 1.0, "LINC00861": 1.0},
        "2": {"CD3D": 0.9, "GZMK": 0.8},
        "3": {"MS4A1": 0.9},
    }
    a = ad.AnnData(X=np.vstack([_expr(blocks[g], rng) for g in blocks]))
    a.var_names = GENES
    a.obs_names = [f"c{i}" for i in range(a.n_obs)]
    a.obs["cluster"] = pd.Categorical(np.repeat(list(blocks), N), categories=list(blocks))
    a.obs["cell_type"] = a.obs["cluster"].astype(str).map(labels).astype("category")
    P._wilcoxon(a, "cluster", key_added=markers_key("cluster"))
    return a


LABELS = {"0": "CD14+ monocytes", "1": "CD14+ monocytes", "2": "T cells / NK cells", "3": "B cells"}


def test_shared_and_mixture_labels_get_the_genes_that_set_them_apart():
    a = _data(LABELS)
    out = LQ.apply(a, "cluster", "cell_type")

    shown = a.obs.groupby("cluster", observed=True)["cell_type_qualified"].first().astype(str).to_dict()
    # Against its sibling: interferon genes up (not the ribosomal or lncRNA gene); a gene
    # down only in a slot left free (two genes at most).
    assert shown["1"] == "CD14+ monocytes (ISG15+ IFIT3+)"
    assert shown["0"] == "CD14+ monocytes (CD4+ ISG15-)"
    # A mixture, against everything else: higher genes only.
    assert shown["2"] == "T cells / NK cells (CD3D+ GZMK+)"
    assert shown["3"] == "B cells"  # one cluster, a clear label: left alone
    assert out == {"qualifiers": 3, "qualifiers_considered": 3, "qualifier_column": "cell_type_qualified"}
    assert set(a.obs["cell_type"].astype(str)) == set(LABELS.values())  # labels untouched

    rec = json.loads(a.uns["cell_type_qualifiers"])
    assert rec["groupby"] == "cluster" and rec["cutoffs"]["max_plus"] == LQ.MAX_PLUS
    assert rec["clusters"]["1"]["versus"] == "same_label" and rec["clusters"]["1"]["reference"] == ["0"]
    assert rec["clusters"]["2"]["versus"] == "all" and rec["clusters"]["2"]["minus"] == []


def test_nothing_standing_out_keeps_the_plain_label():
    a = _data({"0": "Monocytes", "1": "Monocytes", "2": "T cells", "3": "B cells"})
    a.X[:N] = a.X[N:2 * N]  # cluster 0 = cluster 1's cells: nothing tells them apart
    assert LQ.apply(a, "cluster", "cell_type") == {"qualifiers": 0, "qualifiers_considered": 2}
    assert "cell_type_qualified" not in a.obs and "cell_type_qualifiers" not in a.uns


def test_per_cell_labels_and_a_new_run_drop_the_old_column():
    a = _data(LABELS)
    LQ.apply(a, "cluster", "cell_type")
    a.obs["cell_type"] = pd.Categorical(np.where(np.arange(a.n_obs) % 2, "x", "y"))  # per cell
    assert LQ.apply(a, "cluster", "cell_type") == {}
    assert "cell_type_qualified" not in a.obs and "cell_type_qualifiers" not in a.uns


def test_a_merge_by_rename_works_the_qualifiers_out_again():
    a = _data(LABELS)
    LQ.apply(a, "cluster", "cell_type")
    a.obs["cell_type"] = a.obs["cell_type"].astype(str).replace("B cells", "T cells / NK cells").astype("category")
    LQ.follow_rename(a, "cell_type")
    rec = json.loads(a.uns["cell_type_qualifiers"])
    assert rec["clusters"]["3"]["versus"] == "same_label" and rec["clusters"]["3"]["reference"] == ["2"]
    LQ.follow_rename(a, "cell_type", large=True)
    assert "cell_type_qualified" not in a.obs


def test_annotation_step_writes_the_qualified_column(monkeypatch):
    a = _data(LABELS)
    a.uns["scview_active_clustering"] = "cluster"

    def fake_llm(adata, params, groupby, target):
        adata.obs[target] = adata.obs[groupby].astype(str).map(LABELS).astype("category")
        return {"cl_ids": 0}

    monkeypatch.setattr(P, "_annotate_llm", fake_llm)
    monkeypatch.setattr("scview.config.get_settings", lambda: types.SimpleNamespace(
        DEEPINFRA_API_KEY="k", RAG_CHAT_MODEL="m"))
    rec = P._run_cell_type_annotation(a, P.PipelineParams(annotation_groupby="cluster"))
    assert rec["qualifiers"] == 3 and rec["cl_ids"] == 0
    assert "cell_type_qualified" in a.obs


@pytest.mark.parametrize("gene,ok", [("ISG15", True), ("RPS3", False), ("MT-CO1", False),
                                     ("LINC00861", False), ("AC012345.1", False), ("ENSG00000141510", False),
                                     ("Gm12345", False), ("1700001C19Rik", False), ("HLA-DRA", True),
                                     ("MALAT1-AS1", False)])
def test_uninformative_names_are_never_qualifiers(gene, ok):
    assert LQ.informative(gene) is ok


# --- low-quality answer ---------------------------------------------------------


def _qc_data() -> ad.AnnData:
    """5 clusters with stored QC; cluster 4 has few genes and high mito."""
    a = _data({"0": "a", "1": "b", "2": "c", "3": "d"})
    extra = a[:N].copy()
    extra.obs_names = [f"q{i}" for i in range(N)]
    a = ad.concat([a, extra])
    a.obs["cluster"] = pd.Categorical(np.repeat(["0", "1", "2", "3", "4"], N), categories=list("01234"))
    rng = np.random.default_rng(1)
    genes = np.r_[rng.normal(2000, 50, 4 * N), rng.normal(400, 20, N)]
    a.obs["n_genes_by_counts"] = genes
    a.obs["total_counts"] = genes * 3
    a.obs["pct_counts_mt"] = np.r_[rng.normal(3, 0.3, 4 * N), rng.normal(25, 1, N)]
    P._wilcoxon(a, "cluster", key_added=markers_key("cluster"))
    return a


def test_qc_notes_name_damaged_clusters_with_their_numbers():
    notes = P._annotation_qc_notes(_qc_data(), "cluster")
    assert set(notes) == {"4"}
    assert "genes/cell vs" in notes["4"] and "mitochondrial vs" in notes["4"] and "flagged low quality" in notes["4"]


def test_annotator_sees_qc_notes_and_a_low_quality_answer_is_kept_without_an_id(monkeypatch):
    a = _qc_data()
    sent: list[str] = []

    class _Client:
        def __init__(self, **_):
            self.chat = types.SimpleNamespace(completions=self)

        def create(self, messages, **_):
            sent.append(messages[0]["content"])
            reply = ("tissue: blood\n0: Monocytes | CL:0000576\n1: Monocytes | CL:0000576\n"
                     "2: T cells | CL:0000084\n3: B cells | CL:0000236\n4: Low quality cells | CL:0000000")
            msg = types.SimpleNamespace(content=reply)
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg, finish_reason="stop")])

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=_Client))
    monkeypatch.setattr("scview.config.get_settings", lambda: types.SimpleNamespace(
        DEEPINFRA_API_KEY="k", RAG_CHAT_MODEL="m"))
    monkeypatch.setattr(P, "_record_llm_cl_ids", lambda adata, target, ids: {"ids": ids})
    fit = P._annotate_llm(a, P.PipelineParams(), "cluster", "cell_type")

    listing = sent[0].split("Clusters and marker genes:")[1].strip().splitlines()
    assert [ln for ln in listing if "[QC:" in ln] == [ln for ln in listing if ln.startswith("4:")]
    assert "'Low-quality cells | CL:none'" in sent[0]
    assert a.uns["cell_type_llm_mapping"]["4"] == "Low-quality cells"
    assert fit["ids"]["Low-quality cells"] is None  # the model's CL:0000000 is not kept
    assert fit["llm_qc_notes"] == 1


@pytest.mark.parametrize("label", ["Low quality cells", "low-quality cells", "Low-quality", "Likely low quality cells"])
def test_low_quality_spellings(label):
    assert P._LOW_QUALITY.match(label)


def test_low_quality_is_not_a_cell_type_for_the_ontology():
    from scview.core import cell_ontology

    assert cell_ontology._no_type("Low-quality cells")


def test_replay_writes_the_qualified_column_from_the_record():
    from scview.core import replay_script as R

    a = _data(LABELS)
    a.uns["cell_type_llm_mapping"] = dict(LABELS)
    LQ.apply(a, "cluster", "cell_type")
    snip = R._cell_type_annotation({"annotation_method": "llm", "annotation_groupby": "cluster"}, R._Ctx(a))
    code = "\n".join(snip.code)
    assert "cell_type_qualified" in code and "CD14+ monocytes (ISG15+ IFIT3+)" in code
    del a.uns["cell_type_qualifiers"]
    assert "cell_type_qualified" not in "\n".join(R._cell_type_annotation(
        {"annotation_method": "llm", "annotation_groupby": "cluster"}, R._Ctx(a)).code)


def test_the_mixture_rule_is_the_last_naming_rule():
    """Placed before the low-quality text (v0.66.0), the default model named one parent of a
    two-lineage cluster more often (annotation benchmark T4 after-arm)."""
    p = P._annotation_prompt("0: CD3D")
    mixture, low, out = p.index("name the mixture"), p.index("'Low-quality cells | CL:none'"), p.index("Output one line")
    assert low < mixture < out
    assert "name the cell type it is in" not in p
