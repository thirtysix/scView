"""Cell Ontology ids for labels (core/cell_ontology.py): the strict matcher, the LLM
annotator's checked ids (pipeline._annotate_llm), the per-column matches endpoint, the
is_a distance, the pinned download, replay, the co-pilot fact and the MCP tool.
Fixture: real CL stanzas cut from cl-basic.obo 2026-06-08 (cl_pbmc.obo); no network."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import types
from pathlib import Path

import anndata as ad
import httpx
import numpy as np
import pandas as pd
import pytest

from scview.core import cell_ontology as co
from scview.core import marker_db as mdb

API = "/api/v1"
FIX = Path(__file__).parent / "fixtures" / "cell_ontology" / "cl_pbmc.obo"


@pytest.fixture(scope="module")
def index() -> co.Index:
    return co.Index(co.parse(FIX.read_text()), "cl/releases/2026-06-08/cl-basic.owl")


@pytest.fixture()
def cl_dir(tmp_path, monkeypatch) -> Path:
    """DATA_DIR/marker_db with the fixture as the fetched raw obo; nothing fetched."""
    dest = tmp_path / "marker_db"
    (dest / "raw").mkdir(parents=True)
    shutil.copy(FIX, dest / "raw" / mdb.ONTOLOGY_FILE)
    monkeypatch.setattr(mdb, "marker_db_dir", lambda: dest)
    monkeypatch.setattr(mdb, "auto_download_enabled", lambda: False)
    co._cache.clear()
    yield dest
    co._cache.clear()


@pytest.fixture()
def offline(tmp_path, monkeypatch):
    monkeypatch.setattr(mdb, "marker_db_dir", lambda: tmp_path / "none")
    monkeypatch.setattr(mdb, "auto_download_enabled", lambda: False)
    co._cache.clear()
    yield
    co._cache.clear()


# ---------------------------------------------------------------------------
# Strict matching


def test_strict_key_sets_aside_case_plurals_plus_hyphens_and_cells():
    k = co.strict_key
    assert k("CD14+ Monocytes") == k("CD14-positive monocyte") == k("cd14 positive monocytes")
    assert k("NK cells") == k("NK") == k("NK cell") == ("nk",)
    assert k("B-cells") == k("b cell") == ("b",)
    assert k("γδ T cells") == k("gamma-delta T cell")
    assert k("CD16- monocyte") == ("cd16", "negative", "monocyte") != k("CD16 monocyte")
    assert k("Classical monocytes") != k("monocytes classical")  # order kept
    assert k("cells") == ("cell",)


# Kang 2018's 13 author labels: abbreviations get nothing (no abbreviation table here).
KANG = {
    "B": ("CL:0000236", "normalised"), "NK": ("CL:0000623", "normalised"),
    "pDC": ("CL:0000784", "exact_synonym"),
    **{lab: (None, "none") for lab in ("B activated", "CD4 Memory T", "CD4 Naive T", "CD8 T", "CD14 Mono",
                                       "CD16 Mono", "DC", "Eryth", "Mk", "T activated")},
}


@pytest.mark.parametrize("label, expected", list(KANG.items()))
def test_kang_labels(index, label, expected):
    cid, name, how = co.match_label(label, index)
    assert (cid, how) == expected
    assert name == (index.name(cid) if cid else None)


@pytest.mark.parametrize("label, cid, how", [
    ("B cell", "CL:0000236", "exact_name"),
    ("CD14-positive monocyte", "CL:0001054", "exact_name"),
    ("T-cell", "CL:0000084", "exact_synonym"),
    ("NK cell", "CL:0000623", "exact_synonym"),
    ("B cells", "CL:0000236", "normalised"),
    ("CD14+ Monocytes", "CL:0001054", "normalised"),
    ("Classical monocytes", "CL:0000860", "normalised"),
    ("Non-classical monocytes", "CL:0000875", "normalised"),
    ("Plasmacytoid dendritic cells", "CL:0000784", "normalised"),
    ("Megakaryocytes", "CL:0000556", "normalised"),
    ("regulatory t-cells", "CL:0000815", "normalised"),
    # CL's exact synonym of the CD14+CD16+ term; in scRNA-seq it is the non-classical one
    ("CD16+ Monocytes", None, "none"),
    ("CD16+ monocyte", None, "none"),
    ("NK cells / CD8+ T cells", None, "none"),  # a mixture
    ("Unknown", None, "none"),
    ("", None, "none"),
    ("CD8+ T cells", None, "none"),  # CL has only the alpha-beta term: no guess
    ("Monocyte-like", None, "none"),
])
def test_strict_match(index, label, cid, how):
    assert co.match_label(label, index)[0::2] == (cid, how)


AMBIGUOUS = """format-version: 1.2

[Term]
id: CL:0000001
name: alpha cell
synonym: "shared cell" EXACT []

[Term]
id: CL:0000002
name: beta cell
synonym: "shared cell" EXACT []
"""


def test_two_terms_sharing_a_synonym_give_none_unless_the_model_picks_one():
    idx = co.Index(co.parse(AMBIGUOUS))
    assert co.match_label("shared cell", idx)[2] == "none"
    assert co.match_label("Shared cells", idx)[2] == "none"
    r = co.resolve(idx, "shared cell", "CL:0000002")
    assert (r["cl_id"], r["how"], r["source"]) == ("CL:0000002", "exact_synonym", "model")


def test_resolve_keeps_the_models_id_only_when_it_agrees(index):
    r = co.resolve(index, "CD14+ Monocytes", "CL:0001054")
    assert (r["cl_id"], r["how"], r["source"], r["model_cl_id"]) == ("CL:0001054", "normalised", "model", "CL:0001054")
    assert r["url"].endswith("obo_id=CL:0001054") and r["cl_name"] == "CD14-positive monocyte"
    # A real term that is another cell type: the label's own match instead
    r = co.resolve(index, "B cells", "CL:0000623")
    assert (r["cl_id"], r["source"], r["model_cl_id"]) == ("CL:0000236", "label", "CL:0000623")
    # A coarser term than the label names: not kept, and the label has no strict match
    r = co.resolve(index, "CD4+ T cells", "CL:0000084")
    assert (r["cl_id"], r["how"], r["source"]) == (None, "none", "")
    # Not in the ontology, or not an id at all
    assert co.resolve(index, "NK cells", "CL:9999999")["source"] == "label"
    assert co.resolve(index, "NK cells", "nonsense")["model_cl_id"] is None
    assert co.resolve(index, "Unknown", "CL:0000236")["cl_id"] is None
    assert co.resolve(index, "CD16+ monocyte", "CL:0002397")["cl_id"] is None  # misread synonym


# CL's DC stanzas, cut from cl-basic.obo 2026-06-08: "DC1" is an exact synonym of
# conventional DC and "DC2" of plasmacytoid DC (legacy); cDC1 / cDC2 are, by CL's own
# term descriptions, the CD141+ and CD1c+ myeloid DCs.
DC_STANZAS = """
[Term]
id: CL:0000782
name: myeloid dendritic cell
synonym: "mDC" EXACT []
is_a: CL:0000990 {is_inferred="true"} ! conventional dendritic cell

[Term]
id: CL:0000990
name: conventional dendritic cell
synonym: "cDC" EXACT []
synonym: "DC1" EXACT []
synonym: "type 1 DC" EXACT []
is_a: CL:0000451 {is_inferred="true"} ! dendritic cell

[Term]
id: CL:0002394
name: CD141-positive myeloid dendritic cell
is_a: CL:0000782 {is_inferred="true"} ! myeloid dendritic cell
is_a: CL:0000784 ! plasmacytoid dendritic cell

[Term]
id: CL:0002399
name: CD1c-positive myeloid dendritic cell
is_a: CL:0000782 {is_inferred="true"} ! myeloid dendritic cell
"""


@pytest.fixture(scope="module")
def dc_index() -> co.Index:
    return co.Index(co.parse(FIX.read_text() + DC_STANZAS), "cl/releases/2026-06-08/cl-basic.owl")


@pytest.mark.parametrize("label, cid", [
    ("DC2", "CL:0002399"), ("DC2 cells", "CL:0002399"), ("cDC2", "CL:0002399"), ("cDC2s", "CL:0002399"),
    ("DC1", "CL:0002394"), ("cDC1", "CL:0002394"),
])
def test_dc1_dc2_mean_cdc1_cdc2_not_the_cl_synonym(dc_index, label, cid):
    """CellTypist "DC2" was strictly matched to pDC (CL's legacy exact synonym)."""
    assert co.match_label(label, dc_index)[0::2] == (cid, "single_cell")
    # Other names keep their usual match
    assert co.match_label("pDC", dc_index)[0::2] == ("CL:0000784", "exact_synonym")
    assert co.match_label("type 1 DC", dc_index)[0::2] == ("CL:0000990", "exact_synonym")


def test_dc2_is_never_pdc_even_without_the_meant_term(index):
    assert "CL:0002399" not in index.terms  # the PBMC fixture has pDC but no cDC2 term
    assert co.match_label("DC2", index)[0::2] == (None, "none")
    assert co.agrees(index, "DC2", "CL:0000784") is None


def test_the_annotators_dc2_id_is_kept_only_when_it_is_cdc2(dc_index):
    r = co.resolve(dc_index, "DC2", "CL:0000784")  # the model followed CL's synonym
    assert (r["cl_id"], r["how"], r["source"], r["model_cl_id"]) == ("CL:0002399", "single_cell", "label", "CL:0000784")
    r = co.resolve(dc_index, "DC2", "CL:0002399")
    assert (r["cl_id"], r["how"], r["source"]) == ("CL:0002399", "single_cell", "model")
    assert r["how_text"] == co.HOW_STRICT_TEXT["single_cell"]
    assert co.agrees(dc_index, "DC1", "CL:0000990") is None  # CL's DC1 synonym: not this sense
    assert co.agrees(dc_index, "cDC1", "CL:0002394") == "single_cell"


def test_dc2_suggestions_lead_with_cdc2_and_leave_out_pdc(dc_index):
    got = dc_index.suggest("DC2")
    assert [s["id"] for s in got][:1] == ["CL:0002399"] and got[0]["how"] == "atlas"
    assert "CL:0000784" not in [s["id"] for s in got]
    assert [s["id"] for s in dc_index.suggest("pDC")][:1] == ["CL:0000784"]


def test_cl_distance(index):
    assert co.cl_distance("CL:0000236", "CL:0000236", index) == 0
    assert co.cl_distance("CL:0000784", "CL:0000451", index) == 1  # pDC is_a DC
    assert co.cl_distance("CL:0000451", "CL:0000784", index) == 1
    assert co.cl_distance("CL:0002057", "CL:0000576", index) == 2  # via classical monocyte
    assert co.cl_distance("CL:0000236", "CL:0000084", index) == 3  # B cell - lymphocyte - T cell
    assert co.cl_distance("cl_0000236", "CL:0000084", index) == 3
    assert co.cl_distance("CL:0000236", "CL:9999999", index) is None
    assert co.cl_distance("CL:0000999", "CL:0000236", index) is None  # obsolete


def test_match_label_and_distance_load_the_ontology_on_disk(cl_dir):
    assert co.match_label("NK cells") == ("CL:0000623", "natural killer cell", "normalised")
    assert co.cl_distance("CL:0000784", "CL:0000451") == 1


def test_match_label_says_why_without_the_ontology(offline):
    with pytest.raises(co.OntologyUnavailable):
        co.match_label("NK cells")


# ---------------------------------------------------------------------------
# A column's matches, the annotator's record


def _adata(labels: list[str]) -> ad.AnnData:
    a = ad.AnnData(np.zeros((len(labels), 3), dtype="float32"))
    a.obs_names = [f"c{i}" for i in range(len(labels))]
    a.obs["cell_type"] = pd.Categorical(labels)
    return a


def test_map_column_uses_the_annotators_ids_and_matches_the_rest(index):
    a = _adata(["B cells"] * 3 + ["CD4+ T cells"] * 2 + ["NK cells / CD8+ T cells"])
    out = co.map_column(a, "cell_type", index)
    rows = {r["label"]: r for r in out["labels"]}
    assert out["n_labels"] == 3 and out["n_matched"] == 1 and out["annotator"] is False
    assert rows["B cells"]["cl_id"] == "CL:0000236" and rows["B cells"]["n_cells"] == 3
    rec = co.record_annotator_ids(a, "cell_type", {"B cells": "CL:0000236", "CD4+ T cells": "CL:0000624",
                                                   "NK cells / CD8+ T cells": None}, index)
    assert rec["version"] == "cl/releases/2026-06-08/cl-basic.owl"
    assert rec["labels"]["CD4+ T cells"] == {"model_cl_id": "CL:0000624", "cl_id": None, "cl_name": None,
                                             "how": "none", "source": ""}
    assert isinstance(a.uns["cell_type_llm_cl"], str)  # JSON: a label may hold '/'
    out = co.map_column(a, "cell_type", index)
    rows = {r["label"]: r for r in out["labels"]}
    assert out["annotator"] is True and rows["B cells"]["source"] == "model"
    assert rows["CD4+ T cells"]["model_cl_id"] == "CL:0000624" and rows["CD4+ T cells"]["cl_id"] is None
    with pytest.raises(co.TermError):
        co.map_column(a, "nope", index)


def test_the_record_survives_a_save(index, tmp_path):
    a = _adata(["NK cells / CD8+ T cells", "pDC"])
    co.record_annotator_ids(a, "cell_type", {"NK cells / CD8+ T cells": "CL:0000623", "pDC": "CL:0000784"}, index)
    a.write_h5ad(tmp_path / "x.h5ad")
    b = ad.read_h5ad(tmp_path / "x.h5ad")
    assert co.annotator_record(b, "cell_type")["pDC"]["cl_id"] == "CL:0000784"
    assert co.annotator_record(b, "cell_type")["NK cells / CD8+ T cells"]["cl_id"] is None
    assert co.annotator_record(b, "other") == {}


def test_without_an_ontology_the_models_ids_are_kept_unchecked(index):
    a = _adata(["NK cells"])
    rec = co.record_annotator_ids(a, "cell_type", {"NK cells": "cl:0000623"}, None)
    assert rec == {"version": None, "labels": {"NK cells": {
        "model_cl_id": "CL:0000623", "cl_id": None, "cl_name": None, "how": "none", "source": ""}}}
    # checked once an ontology is there
    assert co.map_column(a, "cell_type", index)["labels"][0]["source"] == "model"


def test_suggestions_put_the_annotators_agreeing_id_first(index):
    idx = co.Index(co.parse(AMBIGUOUS))
    a = _adata(["shared cell"])
    co.record_annotator_ids(a, "cell_type", {"shared cell": "CL:0000002"}, idx)
    row = co.suggest(a, "cell_type", idx)["labels"][0]
    assert row["suggestions"][0]["id"] == "CL:0000002" and row["suggestions"][0]["how"] == "annotator"
    assert len({s["id"] for s in row["suggestions"]}) == len(row["suggestions"])


# ---------------------------------------------------------------------------
# The LLM annotator


def test_split_cl_id():
    from scview.core.pipeline import _split_cl_id

    assert _split_cl_id("B cells | CL:0000236") == ("B cells", "CL:0000236")
    assert _split_cl_id("B cells (CL_0000236)") == ("B cells", "CL:0000236")
    assert _split_cl_id("NK cells | CL:none") == ("NK cells", None)
    assert _split_cl_id("CD8+ T cells") == ("CD8+ T cells", None)
    assert _split_cl_id("Cluster CL:12") == ("Cluster CL:12", None)
    # malformed replies seen in the annotation benchmark: nothing of the id stays in the label
    assert _split_cl_id("Microglia | CL:00002077") == ("Microglia", None)
    assert _split_cl_id("CL:none | CL:none") == ("", None)
    assert _split_cl_id("Endothelial cells | CL") == ("Endothelial cells", None)
    # one id per part of a mixture: the label stays clean and no single id is kept
    assert _split_cl_id("NK cells / monocytes | CL:0000623 / CL:0000576") == ("NK cells / monocytes", None)
    assert _split_cl_id("NK cells / monocytes | CL:0000623; CL:0000576") == ("NK cells / monocytes", None)
    assert _split_cl_id("B cells / HSCs | CL:0000236 / CL:none") == ("B cells / HSCs", None)
    assert _split_cl_id("NK cells / monocytes | CL:none") == ("NK cells / monocytes", None)
    assert _split_cl_id("PCL") == ("PCL", None)


def test_malformed_cl_ids_leave_no_text_in_the_labels():
    from scview.core.pipeline import _parse_cluster_labels, _split_cl_id

    raw = "0: Microglia | CL:00002077\n1: CL:none | CL:none\n2: Endothelial cells | CL\n"
    labels = {c: _split_cl_id(lab)[0] for c, lab in _parse_cluster_labels(raw, ["0", "1", "2"]).items()}
    assert labels == {"0": "Microglia", "1": "", "2": "Endothelial cells"}


def test_parse_keeps_the_id_on_the_label_and_the_fallback_is_not_fooled_by_its_colon():
    from scview.core.pipeline import _parse_cluster_labels

    out = _parse_cluster_labels("tissue: blood\na: B cells | CL:0000236\nb: NK cells | CL:none", ["a", "b"])
    assert out == {"a": "B cells | CL:0000236", "b": "NK cells | CL:none"}
    out = _parse_cluster_labels("tissue: blood\nB cells | CL:0000236\nNK cells", ["a", "b"])
    assert out == {"a": "B cells | CL:0000236", "b": "NK cells"}


def _llm_run(monkeypatch, reply: str):
    from scview.core import pipeline as P

    a = ad.AnnData(np.random.default_rng(0).poisson(1, (20, 5)).astype("float32"))
    a.var_names = [f"G{i}" for i in range(5)]
    a.obs["cluster"] = pd.Categorical(["a", "b", "c", "d"] * 5)
    sent: list[str] = []

    class _Client:
        def __init__(self, **_):
            self.chat = types.SimpleNamespace(completions=self)

        def create(self, messages, **_):
            sent.append(messages[0]["content"])
            msg = types.SimpleNamespace(content=reply)
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=_Client))
    monkeypatch.setattr(P, "_top_markers_per_group", lambda adata, g: {k: ["G1", "G2"] for k in "abcd"})
    monkeypatch.setattr("scview.config.get_settings", lambda: types.SimpleNamespace(
        DEEPINFRA_API_KEY="k", RAG_CHAT_MODEL="m", MARKER_DB_AUTO_DOWNLOAD=False))
    fit = P._annotate_llm(a, P.PipelineParams(), "cluster", "cell_type")
    return a, fit, sent


REPLY = ("tissue: blood\na: B cells | CL:0000236\nb: CD4+ T cells | CL:0000624\n"
         "c: natural killer cells | CL:0000236\nd: B cells | CL:0000236")


def test_llm_annotator_asks_for_and_checks_cl_ids(monkeypatch, cl_dir):
    a, fit, sent = _llm_run(monkeypatch, REPLY)
    assert "Cell Ontology id" in sent[0] and "CL:0000236" in sent[0]
    assert a.uns["cell_type_llm_mapping"] == {"a": "B cells", "b": "CD4+ T cells", "c": "Natural killer cells",
                                              "d": "B cells"}
    assert set(a.obs["cell_type"].astype(str)) == {"B cells", "CD4+ T cells", "Natural killer cells"}
    rec = co.annotator_record(a, "cell_type")
    assert set(rec) == {"B cells", "CD4+ T cells", "Natural killer cells"}
    assert rec["B cells"] == {"model_cl_id": "CL:0000236", "cl_id": "CL:0000236", "cl_name": "B cell",
                              "how": "normalised", "source": "model"}
    assert rec["CD4+ T cells"]["cl_id"] is None and rec["CD4+ T cells"]["model_cl_id"] == "CL:0000624"
    # the model's id names another type: re-derived from the label
    assert rec["Natural killer cells"]["cl_id"] == "CL:0000623" and rec["Natural killer cells"]["source"] == "label"
    assert fit == {"cl_ids": 2, "cl_from_model": 1, "cl_version": "cl/releases/2026-06-08/cl-basic.owl"}


def test_llm_annotator_offline_keeps_the_models_ids_unchecked(monkeypatch, offline):
    a, fit, _ = _llm_run(monkeypatch, REPLY)
    assert fit == {}
    rec = co.annotator_record(a, "cell_type")
    assert rec["B cells"]["model_cl_id"] == "CL:0000236" and rec["B cells"]["cl_id"] is None


def test_another_method_drops_an_earlier_llm_runs_ids(monkeypatch, tmp_path):
    from scview.core.pipeline import PipelineParams, _run_cell_type_annotation

    monkeypatch.setattr(mdb, "marker_db_dir", lambda: tmp_path / "marker_db")
    monkeypatch.setattr(mdb, "auto_download_enabled", lambda: False)
    genes = ["CD3D", "CD3E", "CD3G", "TRAC", "MS4A1", "CD79A", "CD79B", "CD19"]
    X = (np.random.default_rng(0).random((20, 8)) * 0.1).astype("float32")
    X[:10, :4] += 3.0
    X[10:, 4:] += 3.0
    a = ad.AnnData(X)
    a.var_names = genes
    a.obs["cluster"] = pd.Categorical(["0"] * 10 + ["1"] * 10)
    a.uns["cell_type_llm_cl"] = json.dumps({"version": None, "labels": {"T cells": {"model_cl_id": "CL:0000084"}}})
    _run_cell_type_annotation(a, PipelineParams(annotation_method="marker_score", annotation_marker_sets="builtin",
                                                annotation_groupby="cluster"))
    assert "cell_type_llm_cl" not in a.uns


# ---------------------------------------------------------------------------
# Replay, co-pilot


def test_replay_writes_the_record_in_with_the_llm_labels():
    from scview.core import replay_script
    from tests.test_replay_script import _with_history

    a = _with_history([("cell_type_annotation", {"annotation_method": "llm", "annotation_target": "cell_type"})])
    a.uns["cell_type_llm_mapping"] = {"0": "T cells", "1": "NK cells / CD8+ T cells"}
    rec = json.dumps({"version": "v", "labels": {"NK cells / CD8+ T cells": {"cl_id": None}}})
    a.uns["cell_type_llm_cl"] = rec
    script = replay_script.build_script(a)
    compile(script, "replay.py", "exec")
    assert "adata.uns['cell_type_llm_cl'] =" in script
    ns: dict = {}
    line_start = script.index("adata.uns['cell_type_llm_cl'] =")
    stmt = script[line_start:].split("\nmarkers_for(", 1)[0]
    exec(stmt.replace("adata.uns['cell_type_llm_cl']", "out"), ns)
    assert ns["out"] == rec
    del a.uns["cell_type_llm_cl"]
    assert "_llm_cl" not in replay_script.build_script(a)


def test_copilot_cell_type_fact_carries_the_kept_ids(index):
    from scview.core.assistant import _annotator_cl_ids

    a = _adata(["B cells", "CD4+ T cells"])
    co.record_annotator_ids(a, "cell_type", {"B cells": "CL:0000236", "CD4+ T cells": "CL:0000624"}, index)
    assert _annotator_cl_ids(a, "cell_type") == {"B cells": " (Cell Ontology CL:0000236, B cell)"}
    assert _annotator_cl_ids(a, "other") == {}


# ---------------------------------------------------------------------------
# API, MCP


def _register(dm, a: ad.AnnData, did: str = "clids") -> str:
    d = Path(dm.data_dir) / "uploads" / did
    d.mkdir(parents=True)
    (d / "metadata.json").write_text(json.dumps({"id": did, "name": did, "status": "ready"}))
    a.write_h5ad(d / "data.h5ad")
    return did


async def test_matches_endpoint(client, dataset_manager, cl_dir, index):
    a = _adata(list(KANG) + ["B cells", "B cells"])
    co.record_annotator_ids(a, "cell_type", {"B cells": "CL:0000236"}, index)
    did = _register(dataset_manager, a)
    r = await client.get(f"{API}/datasets/{did}/annotation/ontology", params={"column": "cell_type"})
    assert r.status_code == 200, r.text
    body = r.json()
    rows = {x["label"]: x for x in body["labels"]}
    assert body["n_labels"] == 14 and body["n_matched"] == 4 and body["annotator"] is True
    assert body["version"] == "cl/releases/2026-06-08/cl-basic.owl"
    assert {k: (v["cl_id"], v["how"]) for k, v in rows.items() if k in KANG} == KANG
    assert rows["B cells"]["source"] == "model" and rows["B cells"]["n_cells"] == 2
    assert rows["pDC"]["url"] == "https://www.ebi.ac.uk/ols4/ontologies/cl/classes?obo_id=CL:0000784"
    assert not (Path(dataset_manager.data_dir) / "derived" / did).exists()  # nothing stored
    r = await client.get(f"{API}/datasets/{did}/annotation/ontology", params={"column": "nope"})
    assert r.status_code == 404


async def test_matches_endpoint_503_offline(client, dataset_manager, offline):
    did = _register(dataset_manager, _adata(["B", "NK"]))
    r = await client.get(f"{API}/datasets/{did}/annotation/ontology", params={"column": "cell_type"})
    assert r.status_code == 503 and "MARKER_DB_AUTO_DOWNLOAD" in r.json()["detail"]


async def test_mcp_tool(client, dataset_manager, cl_dir):
    from tests.test_mcp import _call

    did = _register(dataset_manager, _adata(["B", "NK", "CD14 Mono"]))
    r = await _call("cell_ontology_ids", {"dataset_id": did, "column": "cell_type"})
    assert r["n_matched"] == 2 and r["columns"][:4] == ["label", "cl_id", "cl_name", "how"]
    assert dict((row[0], row[1]) for row in r["labels"]) == {"B": "CL:0000236", "NK": "CL:0000623",
                                                             "CD14 Mono": None}


# ---------------------------------------------------------------------------
# The pinned release


def test_pinned_release_constants():
    assert mdb.ONTOLOGY_URL.endswith(f"/v{mdb.ONTOLOGY_RELEASE}/cl-basic.obo")
    assert len(mdb.ONTOLOGY_SHA256) == 64


def test_another_release_is_refused_and_a_hand_placed_file_is_kept(tmp_path, monkeypatch):
    monkeypatch.setattr(mdb, "ONTOLOGY_URL", "https://cl.test/cl-basic.obo")
    monkeypatch.setattr(mdb, "ONTOLOGY_MIN_TERMS", 5)
    body = FIX.read_bytes()
    client = httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(200, content=body)))
    co._cache.clear()
    with pytest.raises(co.OntologyUnavailable, match="not the release scView expects"):
        co.get_index(tmp_path / "a", auto_download=True, client=client)
    monkeypatch.setattr(mdb, "ONTOLOGY_SHA256", hashlib.sha256(body).hexdigest())
    co.get_index(tmp_path / "b", auto_download=True, client=client, retry=True)
    assert mdb.status(tmp_path / "b")["ontology"]["pinned"] is True
    # a file put in raw/ by hand is used as it is, and said not to be the pinned one
    monkeypatch.setattr(mdb, "ONTOLOGY_SHA256", "0" * 64)
    (tmp_path / "c" / "raw").mkdir(parents=True)
    shutil.copy(FIX, tmp_path / "c" / "raw" / mdb.ONTOLOGY_FILE)
    mdb.ensure(tmp_path / "c", auto_download=False, parts=("ontology",))
    st = mdb.status(tmp_path / "c")["ontology"]
    assert st["downloaded"] and st["pinned"] is False and st["release"] == mdb.ONTOLOGY_RELEASE
    co._cache.clear()


def test_an_unpinned_fetch_from_before_v064_is_replaced_once(tmp_path, monkeypatch):
    """A cl-basic.obo fetched from the unpinned PURL (manifest: served_from a URL, other
    bytes, no "pinned" key) is replaced by the pinned release on first use; while the
    fetch fails, or with downloads off, the old file keeps working."""
    monkeypatch.setattr(mdb, "ONTOLOGY_URL", "https://cl.test/cl-basic.obo")
    monkeypatch.setattr(mdb, "ONTOLOGY_MIN_TERMS", 5)
    body = FIX.read_bytes()
    monkeypatch.setattr(mdb, "ONTOLOGY_SHA256", hashlib.sha256(body).hexdigest())
    old = body.replace(b"data-version: cl/releases/2026-06-08/cl-basic.owl", b"data-version: cl/releases/2025-01-01/cl-basic.owl")
    assert old != body

    def unpinned(dest: Path) -> Path:
        (dest / "raw").mkdir(parents=True)
        (dest / "raw" / mdb.ONTOLOGY_FILE).write_bytes(old)
        mdb.ensure(dest, auto_download=False, parts=("ontology",))  # builds the table from raw/
        m = mdb.read_manifest(dest)
        m["sources"]["ontology"].update(served_from="http://purl.obolibrary.org/obo/cl/cl-basic.obo",
                                        sha256=hashlib.sha256(old).hexdigest())
        m["sources"]["ontology"].pop("pinned", None)
        mdb._write_manifest(dest, m)
        assert mdb.ontology_needs_pinning(dest) and mdb.status(dest)["ontology"]["pinned"] is False
        return dest

    calls: list[str] = []

    def ok(req):
        calls.append(str(req.url))
        return httpx.Response(200, content=body)

    co._cache.clear()
    co._pin_checked.clear()
    a = unpinned(tmp_path / "a")
    idx = co.get_index(a, auto_download=True, client=httpx.Client(transport=httpx.MockTransport(ok)))
    assert calls == ["https://cl.test/cl-basic.obo"]
    assert idx.version == "cl/releases/2026-06-08/cl-basic.owl"
    assert (a / "raw" / mdb.ONTOLOGY_FILE).read_bytes() == body
    assert mdb.status(a)["ontology"]["pinned"] is True and not mdb.ontology_needs_pinning(a)
    co._pin_checked.clear()
    co.get_index(a, auto_download=True, client=httpx.Client(transport=httpx.MockTransport(ok)))
    assert len(calls) == 1  # pinned now: never fetched again

    # the fetch fails: the old file is still used, and is not retried within the process
    co._cache.clear()
    b = unpinned(tmp_path / "b")
    down = []

    def fail(req):
        down.append(1)
        return httpx.Response(503)

    idx = co.get_index(b, auto_download=True, client=httpx.Client(transport=httpx.MockTransport(fail)))
    assert idx.version == "cl/releases/2025-01-01/cl-basic.owl" and (b / "raw" / mdb.ONTOLOGY_FILE).read_bytes() == old
    co.get_index(b, auto_download=True, client=httpx.Client(transport=httpx.MockTransport(fail)))
    assert len(down) == 1 and mdb.ontology_needs_pinning(b)

    # downloads off: nothing fetched, the old file used
    c = unpinned(tmp_path / "c")
    idx = co.get_index(c, auto_download=False, client=httpx.Client(transport=httpx.MockTransport(fail)))
    assert len(down) == 1 and idx.version == "cl/releases/2025-01-01/cl-basic.owl"
    co._cache.clear()
    co._pin_checked.clear()
