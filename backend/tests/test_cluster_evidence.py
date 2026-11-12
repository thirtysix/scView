"""Each cluster's evidence in plain words (core/cluster_evidence.py): the numbers scView's rules
read, the decision they made, and database types as shares of the evidence, never probabilities."""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from scview.core import cell_ontology as co
from scview.core import cluster_evidence as CE
from scview.core import marker_randomness as MR
from scview.core import pipeline as P
from scview.core import provenance
from scview.core.cluster_qc import ClusterQC, ClusterQCReport

API = "/api/v1"
N_TYPES, PER_TYPE, N_GENES = 60, 20, 3000
# type0 is CD8 T cells, type1 CD4 T cells, type2 B cells; the rest have no CL id.
CL = {0: "CL:0000002", 1: "CL:0000003", 2: "CL:0000004"}


class _FakeDB:
    """MarkerDB stand-in: N_TYPES disjoint types of PER_TYPE genes in a G0..G2999 universe."""

    def __init__(self):
        self.types = {f"type{t}": {"genes": [f"G{t * PER_TYPE + i}" for i in range(PER_TYPE)],
                                   "cl": CL.get(t, ""), "sources": []}
                      for t in range(N_TYPES)}
        self.names = sorted(self.types)

    def species(self, species, parts=None):
        return self

    def view(self, job, species):
        return ["fake"]


def _index() -> co.Index:
    """cell > T cell > {CD8 T cell, CD4 T cell}; cell > B cell; 10 other cell types (so T cell
    is not a generic term)."""
    terms = {"CL:0000000": co.Term("CL:0000000", "cell", [], None, []),
             "CL:0000001": co.Term("CL:0000001", "T cell", [], None, ["CL:0000000"]),
             "CL:0000002": co.Term("CL:0000002", "CD8 T cell", [], None, ["CL:0000001"]),
             "CL:0000003": co.Term("CL:0000003", "CD4 T cell", [], None, ["CL:0000001"]),
             "CL:0000004": co.Term("CL:0000004", "B cell", [], None, ["CL:0000000"])}
    for i in range(5, 15):
        terms[f"CL:{i:07d}"] = co.Term(f"CL:{i:07d}", f"other cell {i}", [], None, ["CL:0000000"])
    return co.Index(terms)


def _random_named(rng, n):
    return [f"G{i}" for i in rng.choice(N_TYPES * PER_TYPE, n, replace=False)]


def _markers() -> dict[str, list[str]]:
    rng = np.random.default_rng(1)
    return {
        # as much CD8 (type0) as CD4 (type1) as type3 (no CL id): a T cell, split below it
        "CD8 T": [f"G{i}" for i in [*range(7), *range(20, 27), *range(60, 67)]],
        "junk": _random_named(rng, 25),
        "damaged": [f"G{i}" for i in range(40, 55)],  # B cells, but few genes and high mito
        "mix": [f"G{i}" for i in range(5)] + [f"G{i}" for i in range(40, 45)],
    }


def _adata(column: str = "cell_type", labels=("CD8 T", "junk", "damaged", "mix")) -> ad.AnnData:
    a = ad.AnnData(X=np.zeros((10 * len(labels), N_GENES), dtype=np.float32))
    a.var_names = [f"G{i}" for i in range(N_GENES)]
    a.var["n_cells_by_counts"] = 5 * len(labels)  # every gene detected in half the cells
    a.obs[column] = pd.Categorical([g for g in labels for _ in range(10)])
    return a


def _report(column: str = "cell_type") -> ClusterQCReport:
    rows = [
        ClusterQC(group="CD8 T", n_cells=10, fraction=0.25, median_genes=1400, median_counts=4000, median_pct_mt=5.5),
        ClusterQC(group="junk", n_cells=10, fraction=0.25, median_genes=1370, median_counts=3800, median_pct_mt=5.9),
        ClusterQC(group="damaged", n_cells=10, fraction=0.25, median_genes=1115, median_counts=2500,
                  median_pct_mt=11.2, flags=["low_quality"]),
        ClusterQC(group="mix", n_cells=10, fraction=0.25, median_genes=1600, median_counts=5600, median_pct_mt=6.0,
                  flags=["doublet_like"], doublet_like_of=["CD8 T", "junk"], doublet_like_shared=[6, 4, 9, 12]),
    ]
    return ClusterQCReport(column=column, n_cells=40, n_groups=4,
                           typical={"median_genes": 1370.0, "median_counts": 3900.0, "median_pct_mt": 5.9},
                           clusters=rows)


@pytest.fixture()
def markers(monkeypatch):
    mk = _markers()
    monkeypatch.setattr(CE.marker_selection, "group_markers", lambda adata, column, n: mk)
    return mk


def _by(rep):
    return {c.group: c for c in rep.clusters}


def _text(c, kind):
    return " ".join(ln.text for ln in c.lines if ln.kind == kind)


def test_each_cluster_gets_its_evidence_in_words(markers):
    rep = CE.build(_adata(), "cell_type", _report(), _FakeDB(), _index())
    by = _by(rep)
    assert rep.markers and rep.typing and rep.annotation is None
    # Markers vs random genes: the rule's own numbers, in both directions.
    junk = _text(by["junk"], "markers")
    assert junk.startswith("Markers do not beat 9 of 10 random lists of ")
    assert "better than random" not in junk
    assert "9 of 10 random lists of 25 database genes" in junk and "1 in 10 random lists reach 1e-" in junk
    assert next(ln for ln in by["junk"].lines if ln.kind == "markers").fired
    assert _text(by["CD8 T"], "markers").startswith("Markers match type0 at best p ")
    assert "which beats 9 of 10 random lists of" in _text(by["CD8 T"], "markers")
    # QC against the typical cluster, naming the flag that fired.
    assert _text(by["damaged"], "qc") == ("1,115 genes vs 1,370 typical; 11.2% vs 5.9% mito. Cluster QC flags it "
                                          "low quality (few genes and high mito).")
    assert _text(by["CD8 T"], "qc").endswith("No cluster QC flag.")
    # Doublet-like: the pair the rule picked, its shared markers and RNA vs the larger parent.
    assert _text(by["mix"], "doublet") == (
        "Looks like a mix of CD8 T and junk: 9 of its 12 top markers are theirs (6 of CD8 T's, 4 of junk's), "
        "and its cells carry 1.4x the RNA of CD8 T (5,600 vs 4,000 median counts).")
    # Typing on a named grouping: shares of the evidence, general to specific, never "probability".
    typing = _text(by["CD8 T"], "typing")
    assert typing.startswith("Marker databases: T cell (67% of the evidence) > ")
    assert "CD8 T cell (33%)" in typing and "CD4 T cell (33%)" in typing
    assert "probab" not in json.dumps(rep.model_dump()).lower()
    # The decisions an AI annotation would make (none recorded): the same rules as the annotator.
    assert by["damaged"].label == "Low-quality cells" and not by["damaged"].decided
    # random-like markers: sent to the AI with a note (v0.75.0), never labelled by scView itself
    assert by["junk"].label is None and not by["junk"].decided
    assert _text(by["junk"], "decision") == (
        "AI annotation would ask the AI with a note that its markers do not beat 9 of 10 random gene lists "
        "of the same size against the marker databases: a state the databases lack can look like this.")
    assert by["CD8 T"].label is None and not _text(by["CD8 T"], "decision")


def test_the_evidence_decides_as_the_annotator_does(markers):
    a, report, db = _adata(), _report(), _FakeDB()
    check = MR.check(a, markers, db)
    decided, record = P.decide_clusters(check, report, markers)
    by = _by(CE.build(a, "cell_type", report, db, _index()))
    assert {g: c.label for g, c in by.items() if c.label} == decided
    assert record == {"llm_decided_low_quality": ["damaged"], P.RANDOM_NOTED: ["junk"]}


def test_cluster_numbers_get_no_database_shares(markers, monkeypatch):
    mk = {str(i): v for i, v in enumerate(markers.values())}
    monkeypatch.setattr(CE.marker_selection, "group_markers", lambda adata, column, n: mk)
    report = _report("leiden")
    for i, r in enumerate(report.clusters):
        r.group = str(i)
    report.clusters[3].doublet_like_of = ["0", "1"]
    rep = CE.build(_adata("leiden", ("0", "1", "2", "3")), "leiden", report, _FakeDB(), _index())
    assert not rep.typing and not any(ln.kind == "typing" for c in rep.clusters for ln in c.lines)
    assert any("named groupings only" in n for n in rep.notes)
    assert CE.is_curated(["B cells", "3"]) and not CE.is_curated(["3", "3,1", "12"])


def test_without_ontology_the_types_are_flat_shares(markers):
    by = _by(CE.build(_adata(), "cell_type", _report(), _FakeDB(), None))
    typing = _text(by["CD8 T"], "typing")
    assert typing.startswith("Marker databases") and "type0 (" in typing and "% of the evidence)" in typing


def test_a_recorded_annotation_says_what_scview_did(markers):
    a = _adata()
    provenance.record_step(a, step="cell_type_annotation", tool="scview.cell_type_annotation", params={
        "annotation_method": "llm", "annotation_groupby": "cell_type", "annotation_target": "anno",
        "llm_decided_low_quality": ["damaged"], "llm_random_markers": ["junk"]})
    a.uns["anno_llm_mapping"] = {"CD8 T": "CD8 T cells", "junk": "Unidentified", "damaged": "Low-quality cells",
                                 "mix": "T cells"}
    rep = CE.build(a, "cell_type", _report(), _FakeDB(), _index())
    by = _by(rep)
    assert rep.annotation == "anno"
    assert _text(by["damaged"], "decision") == ("scView labelled this Low-quality cells itself in 'anno', without "
                                                "asking the AI.")
    assert by["damaged"].decided and by["damaged"].label == "Low-quality cells"
    assert _text(by["CD8 T"], "decision") == "The AI named it CD8 T cells in 'anno'."
    # an annotation of v0.71-v0.74: its random-like clusters were labelled Unidentified by scView
    assert by["junk"].label == "Unidentified" and by["junk"].decided
    assert _text(by["junk"], "decision") == ("scView labelled this Unidentified itself in 'anno', without "
                                             "asking the AI.")


def test_a_v075_annotation_says_the_ai_was_told(markers):
    a = _adata()
    provenance.record_step(a, step="cell_type_annotation", tool="scview.cell_type_annotation", params={
        "annotation_method": "llm", "annotation_groupby": "cell_type", "annotation_target": "anno",
        P.RANDOM_NOTED: ["junk"]})
    a.uns["anno_llm_mapping"] = {"CD8 T": "CD8 T cells", "junk": "Injured T cells", "damaged": "B cells",
                                 "mix": "T cells"}
    by = _by(CE.build(a, "cell_type", _report(), _FakeDB(), _index()))
    assert by["junk"].label is None and by["junk"].decided
    assert _text(by["junk"], "decision") == (
        "The AI was told that its markers do not beat 9 of 10 random gene lists of the same size against the "
        "marker databases, and named it Injured T cells in 'anno'.")


def test_no_markers_or_no_databases_say_so(monkeypatch, markers):
    rep = CE.build(_adata(), "cell_type", _report(), None, None)
    assert not rep.typing and any("not downloaded" in n for n in rep.notes)
    assert all(ln.kind != "markers" for c in rep.clusters for ln in c.lines)
    # The QC rule still decides without the databases.
    assert _by(rep)["damaged"].label == "Low-quality cells" and _by(rep)["junk"].label is None
    monkeypatch.setattr(CE.marker_selection, "group_markers", lambda adata, column, n: {})
    rep = CE.build(_adata(), "cell_type", _report(), _FakeDB(), _index())
    assert not rep.markers and "Unified View > Markers" in rep.notes[0]
    assert all(c.label is None for c in rep.clusters)


def test_p_values_read_as_the_app_writes_them():
    assert CE.p_text(2.13) == "1e-2.1" and CE.p_text(0.01) == "1" and CE.p_text(320) == "below 1e-300"


def test_random_check_scores_every_named_list_and_keeps_its_decisions():
    out = MR.check(_adata(), _markers(), _FakeDB())
    assert set(out.flagged) == {"junk"}
    assert out.scores["junk"]["null_cut"] == out.flagged["junk"]["null_cut"]
    assert out.scores["CD8 T"]["best_type"] == "type0" and out.scores["CD8 T"]["best_log10p"] > out.scores["CD8 T"]["null_cut"]
    assert "null_cut" in out.scores["mix"]  # 10 named genes: judged


def test_grounding_lists_decided_and_flagged_clusters_with_their_tags(markers):
    rep = CE.build(_adata(), "cell_type", _report(), _FakeDB(), _index())
    lines, cites = CE.grounding_block(rep)
    text = "\n".join(lines)
    assert "no probabilities" in text
    assert "[result:evidence:cell_type:damaged]" in text and "Cluster QC flags it low quality" in text
    assert "[result:evidence:cell_type:CD8 T]" not in text  # nothing fired: summed up
    assert "The other 1 clusters" in text
    assert dict(cites)["result:evidence:cell_type:junk"] == "junk: evidence"  # flagged, not labelled
    assert "[result:evidence:cell_type:junk]" in text and "do not beat 9 of 10" in text


# ---------------------------------------------------------------------------
# Through the app: endpoint, cache, co-pilot facts, MCP


def _planted_register(dm, did: str = "ev1") -> str:
    from tests.test_cluster_qc import _planted

    d = Path(dm.data_dir) / "uploads" / did
    d.mkdir(parents=True, exist_ok=True)
    (d / "metadata.json").write_text(json.dumps({"id": did, "name": did, "status": "ready"}))
    _planted().write_h5ad(d / "data.h5ad")
    return did


async def test_endpoint_cache_and_bad_columns(client, dataset_manager, monkeypatch):
    did = _planted_register(dataset_manager)
    monkeypatch.setattr(CE, "_on_disk", lambda: (None, None))
    calls = []
    real = CE.build

    def counted(*a, **k):
        calls.append(1)
        return real(*a, **k)

    monkeypatch.setattr(CE, "build", counted)
    r = await client.get(f"{API}/datasets/{did}/clusters/evidence")
    assert r.status_code == 200, r.text
    body = r.json()
    by = {c["group"]: c for c in body["clusters"]}
    assert body["column"] == "leiden" and not body["markers"]
    assert by["5"]["label"] is None  # no markers: the annotator decides nothing
    assert "low quality" in by["5"]["lines"][0]["text"] and by["5"]["lines"][0]["fired"]
    assert (await client.get(f"{API}/datasets/{did}/clusters/evidence")).status_code == 200
    assert len(calls) == 1  # answered from the cache
    r = await client.get(f"{API}/datasets/{did}/clusters/evidence", params={"column": "nope"})
    assert r.status_code == 400
    assert (await client.get(f"{API}/datasets/ghost/clusters/evidence")).status_code == 404


async def test_copilot_answers_why_from_the_evidence(client, dataset_manager, monkeypatch):
    did = _planted_register(dataset_manager, "ev2")
    monkeypatch.setattr(CE, "_on_disk", lambda: (None, None))
    r = await client.post(f"{API}/datasets/{did}/assistant/chat",
                          json={"query": "Why is cluster 5 low quality?"})
    assert r.status_code == 200, r.text
    refs = {s["ref"] for s in r.json()["sources"]}
    assert "result:evidence:leiden:5" in refs


def test_router_keeps_evidence_questions_on_qc():
    from scview.core.rag.router import named_result_facets

    for q in ("Why was cluster 3 labelled Unidentified?", "why is cluster 5 low quality",
              "is cluster 9 doublet-like?"):
        assert "qc" in named_result_facets(q), q


def test_claims_resolve_an_evidence_citation():
    from scview.core.claims import resolve_tag

    refs = ["result:evidence:leiden", "result:evidence:leiden:5", "result:qc:leiden:5"]
    assert resolve_tag("result:evidence:leiden:5", refs) == ["result:evidence:leiden:5"]
    assert resolve_tag("evidence:5", refs) == ["result:evidence:leiden:5"]


async def test_mcp_tool(client, dataset_manager, monkeypatch):
    from tests.test_mcp import _call

    did = _planted_register(dataset_manager, "ev3")
    monkeypatch.setattr(CE, "_on_disk", lambda: (None, None))
    r = await _call("cluster_evidence", {"dataset_id": did})
    assert r["column"] == "leiden" and len(r["clusters"]) == 7
    five = next(c for c in r["clusters"] if c["group"] == "5")
    assert any("low quality" in t for t in five["evidence"])


def test_the_cluster_report_carries_the_groups_evidence(monkeypatch):
    from scview.core import report, report_html
    from tests.test_report import _Adaptor, _sec
    from tests.test_report import _adata as report_adata

    monkeypatch.setattr(CE, "_on_disk", lambda: (None, None))
    rep = report.cluster_report(_Adaptor(report_adata()), "cell_type", "A")
    ev = _sec(rep, "qc")["data"]["evidence"]
    assert ev["status"] == "ok" and any(ln["kind"] == "qc" for ln in ev["lines"])
    page = report_html.render(rep)
    assert "<h3>Evidence</h3>" in page
    assert next(ln["text"] for ln in ev["lines"] if ln["kind"] == "qc").split(".")[0] in page
