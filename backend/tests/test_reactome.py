"""Reactome pathway diagrams (core/reactome.py, api/v1/pathways.py).

Small hand-made layout + graph files in Reactome's real schema (release 98):
a diagram R-HSA-200 drawing a sub-pathway R-HSA-210 (no diagram of its own) as
a shadow, R-HSA-211 below that, an overview R-HSA-100 of sub-pathway boxes only,
and the mouse counterparts. Nothing touches the network: files are placed by
hand, or served by an httpx MockTransport.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import httpx
import numpy as np
import pandas as pd
import pytest
import scanpy as sc

from scview.config import Settings
from scview.core import de_results, reactome
from scview.core.uns_keys import markers_key

API = "/api/v1"

RELATION = "\n".join([
    "R-HSA-100\tR-HSA-200", "R-HSA-100\tR-HSA-300", "R-HSA-200\tR-HSA-210", "R-HSA-210\tR-HSA-211",
    "R-MMU-100\tR-MMU-200", "R-MMU-200\tR-MMU-210",
    "R-BTA-100\tR-BTA-200",  # another species: ignored
]) + "\n"
WITH_DIAGRAMS = "pathwayId\tpathwayName\tisDisease\nR-HSA-200\tChild A\tfalse\nR-HSA-300\tChild B\tfalse\n"


def _seg(x1, y1, x2, y2):
    return {"from": {"x": x1, "y": y1}, "to": {"x": x2, "y": y2}}


def _layout_200(st_id: str = "R-HSA-200") -> dict:
    arrow = {"a": {"x": 96, "y": 46}, "b": {"x": 100, "y": 50}, "c": {"x": 96, "y": 54}, "type": "ARROW",
             "centre": {"x": 96, "y": 54}}
    return {
        "dbId": 200, "stableId": st_id, "displayName": "Child A", "minX": 0, "maxX": 400, "minY": 0,
        "maxY": 300, "disease": False, "notes": [],
        "nodes": [
            {"id": 1, "reactomeId": 1001, "displayName": "STAT1 [cytosol]", "schemaClass": "EntityWithAccessionedSequence",
             "renderableClass": "Protein", "prop": {"x": 10, "y": 20, "width": 60, "height": 20},
             "connectors": [{"edgeId": 50, "type": "INPUT", "segments": [_seg(70, 30, 90, 50)]},
                            {"edgeId": 51, "type": "CATALYST", "segments": [_seg(40, 40, 140, 90)],
                             "endShape": {"c": {"x": 140, "y": 90}, "r": 4, "type": "CIRCLE", "empty": True}}]},
            {"id": 2, "reactomeId": 1002, "displayName": "STAT1:IRF9\x07 <b>complex</b>‮" + "x" * 400,
             "schemaClass": "Complex", "renderableClass": "Complex",
             "prop": {"x": 110, "y": 40, "width": 80, "height": 30},
             "connectors": [{"edgeId": 50, "type": "OUTPUT", "segments": [_seg(100, 50, 110, 50)], "endShape": arrow}]},
            {"id": 3, "reactomeId": 1003, "displayName": "IRF set", "schemaClass": "DefinedSet",
             "renderableClass": "EntitySet", "prop": {"x": 200, "y": 40, "width": 60, "height": 20},
             "connectors": [{"edgeId": 50, "type": "BOGUS", "segments": [_seg(0, 0, 1, 1)]}]},
            {"id": 4, "reactomeId": 1006, "displayName": "ATP", "schemaClass": "SimpleEntity",
             "renderableClass": "Chemical", "prop": {"x": 10, "y": 80, "width": 30, "height": 15}, "connectors": []},
            {"id": 5, "reactomeId": 1007, "displayName": "ruxolitinib", "schemaClass": "ProteinDrug",
             "renderableClass": "ProteinDrug", "prop": {"x": 10, "y": 120, "width": 50, "height": 15},
             "connectors": []},
            {"id": 6, "reactomeId": 1008, "displayName": "SOCS1 gene", "schemaClass": "EntityWithAccessionedSequence",
             "renderableClass": "Gene", "prop": {"x": 300, "y": 200, "width": 60, "height": 40}, "connectors": [],
             "endShape": arrow},
        ],
        "edges": [
            {"id": 50, "reactomeId": 5001, "displayName": "STAT1 binds IRF9", "schemaClass": "Reaction",
             "renderableClass": "Reaction", "reactionType": "association",
             "segments": [_seg(90, 50, 100, 50)], "endShape": arrow,
             "reactionShape": {"a": {"x": 84, "y": 44}, "b": {"x": 96, "y": 56}, "type": "BOX", "empty": True},
             "inputs": [{"id": 1}], "outputs": [{"id": 2}]},
            {"id": 51, "reactomeId": 5002, "displayName": "Other step", "schemaClass": "Reaction",
             "renderableClass": "Reaction", "reactionType": "transition", "segments": [_seg(150, 90, 200, 90)],
             "reactionShape": {"c": {"x": 145, "y": 90}, "r": 5, "r1": 3, "type": "DOUBLE_CIRCLE"},
             "inputs": [], "outputs": []},
        ],
        "compartments": [
            {"id": 80, "displayName": "nucleoplasm", "prop": {"x": 190, "y": 30, "width": 100, "height": 60},
             "textPosition": {"x": 195, "y": 85}},
            {"id": 81, "displayName": "cytosol", "prop": {"x": 0, "y": 0, "width": 400, "height": 300},
             "insets": {"x": 5, "y": 5, "width": 390, "height": 290}, "textPosition": {"x": 10, "y": 290}},
        ],
        "links": [
            {"id": 90, "renderableClass": "EntitySetAndMemberLink", "segments": [_seg(230, 60, 150, 70)]},
            {"id": 91, "renderableClass": "Mystery", "segments": [_seg(0, 0, 1, 1)]},
        ],
        "shadows": [
            {"id": 95, "reactomeId": 2100, "displayName": "Sub", "renderableClass": "Shadow",
             "points": [{"x": 5, "y": 5}, {"x": 200, "y": 5}, {"x": 200, "y": 100}, {"x": 5, "y": 100}]},
        ],
    }


def _graph_200(st_id: str = "R-HSA-200", mouse: bool = False) -> dict:
    sym = (lambda s: s[:1] + s[1:].lower()) if mouse else (lambda s: s)
    return {
        "dbId": 200, "stId": st_id, "speciesName": "Mus musculus" if mouse else "Homo sapiens",
        "nodes": [
            {"dbId": 1001, "stId": "R-HSA-1001", "schemaClass": "EntityWithAccessionedSequence", "diagramIds": [1],
             "geneNames": [sym("STAT1")], "children": []},
            {"dbId": 1002, "stId": "R-HSA-1002", "schemaClass": "Complex", "diagramIds": [2],
             "children": [1001, 1003]},
            # A set listing its complex: a cycle the recursion must not follow forever.
            {"dbId": 1003, "stId": "R-HSA-1003", "schemaClass": "DefinedSet", "diagramIds": [3],
             "children": [1004, 1005, 1002]},
            {"dbId": 1004, "schemaClass": "EntityWithAccessionedSequence",
             "geneNames": [sym("IRF9"), sym("ISGF3G")], "children": []},
            {"dbId": 1005, "schemaClass": "EntityWithAccessionedSequence", "geneNames": [sym("IRF1")],
             "children": []},
            {"dbId": 1006, "schemaClass": "SimpleEntity", "diagramIds": [4]},
            {"dbId": 1007, "schemaClass": "ProteinDrug", "diagramIds": [5]},
            # Found by reactomeId (no diagramIds)
            {"dbId": 1008, "schemaClass": "EntityWithAccessionedSequence", "geneNames": [sym("SOCS1")]},
        ],
        "edges": [],
        "subpathways": [{"dbId": 2100, "stId": st_id.replace("200", "210"), "displayName": "Sub",
                         "events": [5001]}],
    }


def _overview() -> tuple[dict, dict]:
    layout = {"dbId": 100, "stableId": "R-HSA-100", "displayName": "Overview", "minX": 0, "maxX": 100,
              "minY": 0, "maxY": 100, "edges": [], "compartments": [], "links": [], "shadows": [],
              "nodes": [
                  {"id": 7, "reactomeId": 2000, "displayName": "Child A", "renderableClass": "ProcessNode",
                   "prop": {"x": 0, "y": 0, "width": 40, "height": 20}, "connectors": []},
                  {"id": 8, "reactomeId": 3000, "displayName": "Child B", "renderableClass": "ProcessNode",
                   "prop": {"x": 50, "y": 0, "width": 40, "height": 20}, "connectors": []},
              ]}
    graph = {"dbId": 100, "stId": "R-HSA-100", "edges": [], "nodes": [
        {"dbId": 2000, "stId": "R-HSA-200", "schemaClass": "Pathway", "diagramIds": [7]},
        {"dbId": 3000, "stId": "R-HSA-300", "schemaClass": "Pathway", "diagramIds": [8]},
    ]}
    return layout, graph


def _place(data_dir: Path) -> Path:
    root = reactome.release_dir(data_dir)
    (root / "diagram").mkdir(parents=True)
    (root / "ReactomePathwaysRelation.txt").write_text(RELATION)
    (root / "humanPathwaysWithDiagrams.txt").write_text(WITH_DIAGRAMS)
    files = {"R-HSA-200": (_layout_200(), _graph_200()),
             "R-MMU-200": (_layout_200("R-MMU-200"), _graph_200("R-MMU-200", mouse=True)),
             "R-HSA-100": _overview()}
    for sid, (layout, graph) in files.items():
        (root / "diagram" / f"{sid}.json").write_text(json.dumps(layout))
        (root / "diagram" / f"{sid}.graph.json").write_text(json.dumps(graph))
    return root


@pytest.fixture(autouse=True)
def _small_release(monkeypatch):
    monkeypatch.setattr(reactome, "MIN_DIAGRAMS", 1)
    monkeypatch.setattr(reactome, "MIN_RELATIONS", 1)
    reactome.clear_cache()
    de_results.clear()
    yield
    reactome.clear_cache()
    de_results.clear()


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------


def test_the_model_draws_each_class_with_its_genes():
    m = reactome.build_model(_layout_200(), _graph_200())
    by = {n["id"]: n for n in m["nodes"]}
    assert [by[i]["class"] for i in range(1, 7)] == ["Protein", "Complex", "EntitySet", "Chemical", "Protein", "Gene"]
    assert by[1]["genes"] == ["STAT1"]
    # complex -> STAT1 + the set's members, once each; the cyclic child is not followed again
    assert by[2]["genes"] == ["STAT1", "IRF9", "IRF1"]
    assert by[3]["genes"] == ["IRF9", "IRF1"]
    assert "genes" not in by[4]  # a chemical has none
    assert by[5]["drug"] is True and by[5]["genes"] == []
    assert by[6]["genes"] == ["SOCS1"] and by[6]["end"]["type"] == "ARROW"
    assert m["aliases"] == {"IRF9": ["ISGF3G"]}
    assert m["n_genes"] == 4
    assert by[1]["st_id"] == "R-HSA-1001" and (by[1]["x"], by[1]["y"], by[1]["w"], by[1]["h"]) == (10, 20, 60, 20)


def test_labels_are_sanitised_and_capped():
    m = reactome.build_model(_layout_200(), _graph_200())
    label = next(n["label"] for n in m["nodes"] if n["id"] == 2)
    assert "\x07" not in label and "‮" not in label
    assert len(label) <= reactome.MAX_TEXT and label.endswith("…")
    assert label.startswith("STAT1:IRF9 <b>complex</b>")  # text, escaped by the renderer


def test_reactions_connectors_links_and_compartments():
    m = reactome.build_model(_layout_200(), _graph_200())
    r50 = next(r for r in m["reactions"] if r["id"] == 50)
    assert r50["shape"] == {"type": "BOX", "empty": True, "a": [84, 44], "b": [96, 56]}
    assert r50["end"]["type"] == "ARROW" and r50["segments"] == [[90, 50, 100, 50]]
    r51 = next(r for r in m["reactions"] if r["id"] == 51)
    assert r51["shape"]["type"] == "DOUBLE_CIRCLE" and r51["shape"]["r1"] == 3
    types = sorted(c["type"] for c in m["connectors"])
    assert types == ["CATALYST", "INPUT", "OUTPUT"]  # the unknown type is dropped
    assert next(c for c in m["connectors"] if c["type"] == "CATALYST")["end"]["type"] == "CIRCLE"
    assert [link["kind"] for link in m["links"]] == ["EntitySetAndMemberLink"]
    assert [c["label"] for c in m["compartments"]] == ["cytosol", "nucleoplasm"]  # largest first
    assert "inset" in m["compartments"][0] and "inset" not in m["compartments"][1]
    assert m["bounds"] == {"minX": 0, "maxX": 400, "minY": 0, "maxY": 300}
    assert m["attribution"] == "Reactome (CC BY 4.0), release 98"
    assert m["licence_url"] == "https://reactome.org/license"


# ---------------------------------------------------------------------------
# Resolving a pathway to a diagram
# ---------------------------------------------------------------------------


def test_a_pathway_with_its_own_diagram(tmp_path):
    _place(tmp_path)
    m = reactome.get_diagram("R-HSA-200", tmp_path, auto_download=False)
    assert m["st_id"] == "R-HSA-200" and m["highlight"] is None and not m["too_broad"]
    assert m["url"] == "https://reactome.org/PathwayBrowser/#/R-HSA-200"


def test_a_subpathway_is_drawn_in_its_ancestor_with_its_outline(tmp_path):
    _place(tmp_path)
    m = reactome.get_diagram("R-HSA-210", tmp_path, auto_download=False)
    assert m["st_id"] == "R-HSA-200" and m["requested"] == "R-HSA-210"
    h = m["highlight"]
    assert h["st_id"] == "R-HSA-210" and h["exact"] is True
    assert h["outline"] == [[5, 5], [200, 5], [200, 100], [5, 100]]
    assert h["reactions"] == [50]
    assert "no diagram of its own" in m["message"]


def test_a_deeper_pathway_gets_its_nearest_drawn_ancestor_s_outline(tmp_path):
    _place(tmp_path)
    m = reactome.get_diagram("R-HSA-211", tmp_path, auto_download=False)
    assert m["st_id"] == "R-HSA-200"
    assert m["highlight"]["st_id"] == "R-HSA-210" and m["highlight"]["exact"] is False
    assert "within Sub" in m["message"]


def test_an_overview_of_sub_pathways_is_too_broad(tmp_path):
    _place(tmp_path)
    m = reactome.get_diagram("R-HSA-100", tmp_path, auto_download=False)
    assert m["too_broad"] is True
    assert m["suggestions"] == [{"st_id": "R-HSA-200", "name": "Child A"}, {"st_id": "R-HSA-300", "name": "Child B"}]
    assert "overview of 2 sub-pathways" in m["message"]


def test_mouse_pathways_use_the_human_diagram_list_and_mouse_symbols(tmp_path):
    _place(tmp_path)
    m = reactome.get_diagram("R-MMU-210", tmp_path, auto_download=False)
    assert m["st_id"] == "R-MMU-200" and m["species"] == "mouse"
    assert m["highlight"]["st_id"] == "R-MMU-210"
    assert next(n for n in m["nodes"] if n["id"] == 2)["genes"] == ["Stat1", "Irf9", "Irf1"]


def test_unknown_and_malformed_ids(tmp_path):
    _place(tmp_path)
    with pytest.raises(reactome.PathwayNotFound, match="not in Reactome release 98"):
        reactome.get_diagram("R-HSA-999", tmp_path, auto_download=False)
    for bad in ("R-BTA-200", "R-HSA-1/../../x", "WP100", ""):
        with pytest.raises(ValueError):
            reactome.get_diagram(bad, tmp_path, auto_download=False)


def test_without_files_and_fetching_off_it_says_how_to_fetch(tmp_path):
    with pytest.raises(reactome.ReactomeUnavailable, match="REACTOME_AUTO_DOWNLOAD is off"):
        reactome.get_diagram("R-HSA-200", tmp_path, auto_download=False)


def test_a_broken_cached_file_is_removed(tmp_path):
    root = _place(tmp_path)
    (root / "diagram" / "R-HSA-200.json").write_text("{not json")
    with pytest.raises(ValueError, match="could not be read"):
        reactome.get_diagram("R-HSA-200", tmp_path, auto_download=False)
    assert not (root / "diagram" / "R-HSA-200.json").exists()


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------


def _server(requests: list[str], *, relation: str = RELATION):
    files = {
        "ReactomePathwaysRelation.txt": relation.encode(),
        "humanPathwaysWithDiagrams.txt": WITH_DIAGRAMS.encode(),
        "diagram/R-HSA-200.json": json.dumps(_layout_200()).encode(),
        "diagram/R-HSA-200.graph.json": json.dumps(_graph_200()).encode(),
    }

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(str(request.url))
        assert request.method == "GET" and not request.content
        path = request.url.path.removeprefix("/download/98/")
        if path in files:
            return httpx.Response(200, content=files[path])
        return httpx.Response(404, text="not found")

    return httpx.Client(transport=httpx.MockTransport(handler))


@pytest.fixture()
def pinned(monkeypatch):
    monkeypatch.setattr(reactome, "HIERARCHY_FILES", {
        "ReactomePathwaysRelation.txt": hashlib.sha256(RELATION.encode()).hexdigest(),
        "humanPathwaysWithDiagrams.txt": hashlib.sha256(WITH_DIAGRAMS.encode()).hexdigest(),
    })


def test_first_use_fetches_the_hierarchy_and_the_diagram_once(tmp_path, pinned):
    seen: list[str] = []
    with _server(seen) as client:
        m = reactome.get_diagram("R-HSA-210", tmp_path, client=client)
        assert m["st_id"] == "R-HSA-200"
        n = len(seen)
        reactome.clear_cache()
        reactome.get_diagram("R-HSA-210", tmp_path, client=client)
    assert n == 4 and len(seen) == 4  # cached for good
    assert all(u.startswith("https://reactome.org/download/98/") for u in seen)
    assert "https://reactome.org/download/98/diagram/R-HSA-200.json" in seen
    root = reactome.release_dir(tmp_path)
    assert json.loads((root / "manifest.json").read_text())["release"] == 98
    assert not list(root.rglob("*.part"))


def test_a_listed_diagram_missing_from_the_release_says_so(tmp_path, pinned):
    with _server([]) as client:
        with pytest.raises(reactome.PathwayNotFound, match="though the release lists one"):
            reactome.get_diagram("R-HSA-300", tmp_path, client=client)


def test_a_hierarchy_file_that_is_not_the_pinned_one_is_refused(tmp_path, pinned):
    with _server([], relation=RELATION + "R-HSA-1\tR-HSA-2\n") as client:
        with pytest.raises(reactome.ReactomeUnavailable, match="sha256"):
            reactome.get_diagram("R-HSA-200", tmp_path, client=client)
    assert not (reactome.release_dir(tmp_path) / "ReactomePathwaysRelation.txt").exists()


# ---------------------------------------------------------------------------
# Colouring
# ---------------------------------------------------------------------------


def _stored(genes, lfc, padj) -> SimpleNamespace:
    return SimpleNamespace(genes=genes, logfoldchanges=np.array(lfc, dtype=float),
                           pvals_adj=None if padj is None else np.array(padj, dtype=float), label="T: STIM vs CTRL")


def test_values_per_gene_and_node():
    m = reactome.build_model(_layout_200(), _graph_200())
    table = reactome.table_from_result(_stored(["STAT1", "ISGF3G", "SOCS1", "OTHER"],
                                               [2.0, -1.0, float("nan"), 3.0], [0.01, 0.2, 0.5, 0.0]))
    v = reactome.diagram_values(m, table)
    assert v["genes"] == {"STAT1": {"gene": "STAT1", "lfc": 2.0, "padj": 0.01},
                          "IRF9": {"gene": "ISGF3G", "lfc": -1.0, "padj": 0.2}}  # matched by its alias
    assert v["nodes"]["2"] == {"value": 0.5, "n_genes": 3, "n_measured": 2, "n_changed": 1}
    assert v["nodes"]["6"] == {"value": None, "n_genes": 1, "n_measured": 0, "n_changed": 0}  # NaN: not measured
    assert "4" not in v["nodes"] and "5" not in v["nodes"]  # chemical, drug: no genes
    assert (v["n_diagram_genes"], v["n_measured"], v["n_changed"]) == (4, 2, 1)


def test_values_without_adjusted_p_count_no_changes():
    m = reactome.build_model(_layout_200(), _graph_200())
    v = reactome.diagram_values(m, reactome.table_from_result(_stored(["STAT1"], [1.0], None)))
    assert v["nodes"]["1"]["n_changed"] is None and v["n_changed"] is None and v["has_padj"] is False


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------


@pytest.fixture()
def test_settings(tmp_data_dir: Path) -> Settings:
    _place(tmp_data_dir)
    return Settings(DATA_DIR=str(tmp_data_dir), DEEPINFRA_API_KEY="", RAG_DATABASE_URL="",
                    REACTOME_AUTO_DOWNLOAD=False)


def _register(dm, did: str = "rx") -> str:
    d = Path(dm.data_dir) / "uploads" / did
    d.mkdir(parents=True)
    (d / "metadata.json").write_text(json.dumps({"id": did, "name": did, "status": "ready"}))
    genes = ["STAT1", "IRF9", "IRF1", "SOCS1"] + [f"G{i}" for i in range(20)]
    rng = np.random.default_rng(0)
    X = rng.poisson(1.0, (60, len(genes))).astype("float32")
    X[:30, 0] += 6
    a = ad.AnnData(X)
    a.var_names = genes
    a.obs_names = [f"C{i}" for i in range(60)]
    a.obs["cell_type"] = pd.Categorical(["T"] * 30 + ["B"] * 30)
    sc.pp.log1p(a)
    sc.tl.rank_genes_groups(a, "cell_type", method="wilcoxon", key_added=markers_key("cell_type"))
    a.write_h5ad(d / "data.h5ad")
    return did


async def test_diagram_endpoint(client):
    r = await client.get(f"{API}/pathways/reactome/R-HSA-210/diagram")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["st_id"] == "R-HSA-200" and body["highlight"]["reactions"] == [50]
    assert (await client.get(f"{API}/pathways/reactome/R-HSA-999/diagram")).status_code == 404
    assert (await client.get(f"{API}/pathways/reactome/KEGG1/diagram")).status_code == 422


async def test_diagram_endpoint_offline_is_503(client, test_settings):
    test_settings.DATA_DIR = str(Path(test_settings.DATA_DIR) / "empty")
    r = await client.get(f"{API}/pathways/reactome/R-HSA-200/diagram")
    assert r.status_code == 503 and "REACTOME_AUTO_DOWNLOAD" in r.json()["detail"]


async def test_values_from_a_stored_de_result(client, dataset_manager):
    did = _register(dataset_manager)
    rows = [SimpleNamespace(gene=g, logfoldchange=fc, pval=0.01, pval_adj=p, stat=1.0)
            for g, fc, p in (("STAT1", 1.5, 0.001), ("IRF1", -0.5, 0.3))]
    rid = de_results.remember(did, rows, comparison={"mode": "groups", "group_a": "STIM", "group_b": "CTRL"})
    r = await client.get(f"{API}/datasets/{did}/pathways/reactome/R-HSA-210/values", params={"de_result": rid})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["source"] == f"de:{rid}" and "STIM vs CTRL" in body["label"]
    assert body["nodes"]["2"]["value"] == pytest.approx(0.5) and body["nodes"]["2"]["n_changed"] == 1
    assert body["genes"]["STAT1"]["lfc"] == 1.5
    gone = await client.get(f"{API}/datasets/{did}/pathways/reactome/R-HSA-210/values", params={"de_result": "nope"})
    assert gone.status_code == 404 and "run the DE again" in gone.json()["detail"]


async def test_values_from_a_group_s_markers(client, dataset_manager):
    did = _register(dataset_manager)
    r = await client.get(f"{API}/datasets/{did}/pathways/reactome/R-HSA-200/values",
                         params={"column": "cell_type", "group": "T"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["source"] == "markers:cell_type/T" and body["has_padj"] is True
    assert body["genes"]["STAT1"]["lfc"] > 1 and body["n_measured"] == 4
    bad = await client.get(f"{API}/datasets/{did}/pathways/reactome/R-HSA-200/values",
                           params={"column": "cell_type", "group": "NK"})
    assert bad.status_code == 404


async def test_values_need_exactly_one_source(client, dataset_manager):
    did = _register(dataset_manager)
    url = f"{API}/datasets/{did}/pathways/reactome/R-HSA-200/values"
    assert (await client.get(url)).status_code == 422
    assert (await client.get(url, params={"de_result": "x", "column": "cell_type", "group": "T"})).status_code == 422
    assert (await client.get(f"{API}/datasets/{did}/pathways/reactome/bad/values",
                             params={"de_result": "x"})).status_code == 422


def test_the_model_is_not_changed_by_colouring():
    m = reactome.build_model(_layout_200(), _graph_200())
    before = copy.deepcopy(m)
    reactome.diagram_values(m, reactome.table_from_result(_stored(["STAT1"], [1.0], [0.01])))
    assert m == before
