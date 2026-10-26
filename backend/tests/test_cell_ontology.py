"""Cell Ontology terms for labels (core/cell_ontology.py, api/v1/ontology.py): search,
suggestions, the per-column mapping through labelling / undo / rename, the CELLxGENE
export from a confirmed mapping and the co-pilot fact. Fixture: real CL stanzas cut from
cl-basic.obo 2026-06-08 (tests/fixtures/cell_ontology/cl_pbmc.obo); no network."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import anndata as ad
import httpx
import numpy as np
import pandas as pd
import pytest

from scview.core import cell_ontology as co
from scview.core import cellxgene_export as cx
from scview.core import marker_db as mdb
from scview.core.claims import resolve_tag

API = "/api/v1"
FIX = Path(__file__).parent / "fixtures" / "cell_ontology" / "cl_pbmc.obo"
KANG = ["CD14 Mono", "CD4 Naive T", "CD4 Memory T", "B", "CD16 Mono", "T activated", "CD8 T", "NK",
        "DC", "B activated", "Mk", "pDC", "Eryth"]


@pytest.fixture(scope="module")
def index() -> co.Index:
    return co.Index(co.parse(FIX.read_text()))


@pytest.fixture()
def cl_dir(tmp_path, monkeypatch) -> Path:
    """DATA_DIR/marker_db with the fixture as the fetched raw obo; nothing fetched."""
    dest = tmp_path / "marker_db"
    (dest / "raw").mkdir(parents=True)
    shutil.copy(FIX, dest / "raw" / mdb.ONTOLOGY_FILE)
    monkeypatch.setattr(mdb, "marker_db_dir", lambda: dest)
    co._cache.clear()
    yield dest
    co._cache.clear()


def _adata(labels: list[str] | None = None) -> ad.AnnData:
    labels = labels or ["CD14 Mono"] * 4 + ["B"] * 3 + ["NK"] * 3
    rng = np.random.default_rng(0)
    a = ad.AnnData(rng.poisson(1.0, (len(labels), 8)).astype("float32"))
    a.var_names = [f"G{i}" for i in range(8)]
    a.obs_names = [f"c{i}" for i in range(len(labels))]
    a.obs["cell_type"] = pd.Categorical(labels)
    return a


# ---------------------------------------------------------------------------
# Parsing and search


def test_parse_keeps_definitions_synonym_scopes_and_parents_and_drops_obsolete(index):
    t = index.terms["CL:0000815"]
    assert t.name == "regulatory T cell"
    assert t.definition.startswith("A T cell which regulates overall immune responses")
    assert ("Treg", "EXACT") in t.synonyms
    assert not [s for s, sc in t.synonyms if sc == "BROAD"]  # "suppressor T cell" names a wider group
    assert "CL:0000999" not in index.terms  # obsolete
    d = index.describe("CL:0002057")
    assert {p["id"] for p in d["parents"]} >= {"CL:0000860"}
    assert d["url"] == "https://www.ebi.ac.uk/ols4/ontologies/cl/classes?obo_id=CL:0002057"
    assert index.version is None and co.Index(index.terms, "v1").version == "v1"


@pytest.mark.parametrize("q, expected, match", [
    ("CD14 monocyte", "CL:0001054", "words"),
    ("Treg", "CL:0000815", "synonym"),
    ("NK", "CL:0000623", "words"),
    ("B cell", "CL:0000236", "name"),
    ("b-lymphocyte", "CL:0000236", "synonym"),
    ("CD8 T", "CL:0000625", "tokens"),
    ("naive CD4", "CL:0000895", "tokens"),
    ("erythro", "CL:0000764", "prefix"),
    ("pDC", "CL:0000784", "synonym"),
])
def test_search_ranks_the_expected_term_first(index, q, expected, match):
    r = index.search(q)
    assert r[0]["id"] == expected, [x["name"] for x in r]
    assert r[0]["match"] == match
    assert r[0]["definition"]


def test_search_takes_an_id_typed_directly(index):
    assert [t["id"] for t in index.search("CL:0000236")] == ["CL:0000236"]
    assert index.search("cl_0000236")[0]["match"] == "id"
    assert index.search("CL:0123456") == []  # well formed, not a term
    assert index.search("") == [] and index.search("   ") == []
    assert index.search("zzzz") == []


def test_search_reports_the_synonym_it_matched(index):
    hit = index.search("Treg")[0]
    assert hit["matched"] == "Treg" and hit["name"] == "regulatory T cell"
    assert index.search("monocyte")[0]["matched"] is None


def test_search_is_fast_on_an_ontology_of_cl_size(index):
    # ~3,300 terms like cl-basic: names and synonyms from the fixture's words
    words = sorted({w for t in index.terms.values() for w in co._words(t.name)})
    rng = np.random.default_rng(0)
    terms = {}
    for i in range(3400):
        name = " ".join(rng.choice(words, size=int(rng.integers(2, 6))))
        terms[f"CL:{i:07d}"] = co.Term(f"CL:{i:07d}", name, [(name + " cell", "EXACT")], "x", [])
    big = co.Index(terms)
    big.ontology().n_desc("CL:0000000")
    worst = 0.0
    for q in ["T", "CD4 T", "monocyte", "alpha beta", "reg", "natural killer cell"]:
        best = float("inf")
        for _ in range(3):  # best of 3: one call slowed by a busy machine is not the search
            t0 = time.perf_counter()
            big.search(q)
            best = min(best, time.perf_counter() - t0)
        worst = max(worst, best)
    assert worst < 0.05, worst


# ---------------------------------------------------------------------------
# Suggestions


def test_suggestions_for_kangs_labels(index):
    expected = {  # top suggestion
        "CD14 Mono": "CL:0001054", "CD4 Naive T": "CL:0000895", "CD4 Memory T": "CL:0000897",
        "B": "CL:0000236", "T activated": "CL:0000896", "CD8 T": "CL:0000625", "NK": "CL:0000623",
        "DC": "CL:0000451", "B activated": "CL:0000236", "Mk": "CL:0000556", "pDC": "CL:0000784",
        "Eryth": "CL:0000764",
        # Seurat's non-classical monocytes, not CL's synonym on the intermediate term
        "CD16 Mono": "CL:0002396",
    }
    got = {label: index.suggest(label) for label in KANG}
    assert {k: v[0]["id"] for k, v in got.items() if v} == expected
    assert got["CD14 Mono"][0]["how"] == "exact"
    assert got["CD16 Mono"][0]["how"] == "atlas"
    assert "CL:0002397" in [s["id"] for s in got["CD16 Mono"]]  # CL's synonym match stays offered
    found = [t["id"] for t in index.search("CD16 Mono")]
    assert found[0] == "CL:0002396" and "CL:0002397" in found
    assert got["CD4 Naive T"][0]["how"] == "narrower"
    assert got["B activated"][0]["how"] == "subset"  # no activated B cell term: its parent's name
    assert all(len(v) <= co.N_SUGGEST for v in got.values())


def test_suggest_skips_labels_that_name_no_type_and_reads_ids(index):
    assert index.suggest("Unassigned") == [] and index.suggest("unknown") == [] and index.suggest("--") == []
    assert index.suggest("CL:0000623")[0]["id"] == "CL:0000623"


def test_a_trailing_minus_is_not_dropped(index):
    # "CD16- monocyte" is a synonym of the CD16-negative classical monocyte only
    assert index.suggest("CD16- Mono")[0]["id"] == "CL:0002057"
    assert "CL:0002057" not in [s["id"] for s in index.suggest("CD16 Mono")]


# ---------------------------------------------------------------------------
# Mapping


def test_set_read_and_clear_terms():
    a = _adata()
    prev = co.set_terms(a, "cell_type", {"CD14 Mono": "CL:0001054", "B": "cl_0000236"})
    assert prev == {"CD14 Mono": None, "B": None}
    assert json.loads(a.uns[co.TERMS_KEY]) == {"cell_type": {"B": "CL:0000236", "CD14 Mono": "CL:0001054"}}
    s = co.summary(a, "cell_type")
    assert s["n_labels"] == 3 and s["n_mapped"] == 2 and s["missing"] == ["NK"] and not s["complete"]
    assert co.set_terms(a, "cell_type", {"NK": "unknown"}) == {"NK": None}
    assert co.summary(a, "cell_type")["complete"]
    assert co.set_terms(a, "cell_type", {"B": "CL:0000236"}) == {}  # unchanged
    assert co.set_terms(a, "cell_type", {"B": None}) == {"B": "CL:0000236"}
    assert "B" not in co.stored(a, "cell_type")
    assert co.mapped_columns(a) == ["cell_type"]


def test_bad_terms_change_nothing(index):
    a = _adata()
    with pytest.raises(co.TermError, match="Not a Cell Ontology id"):
        co.set_terms(a, "cell_type", {"B": "UBERON:0000178"})
    with pytest.raises(co.TermError, match="not a label"):
        co.set_terms(a, "cell_type", {"B": "CL:0000236", "Platelets": "CL:0000233"})
    with pytest.raises(co.TermError, match="not a current"):
        co.set_terms(a, "cell_type", {"B": "CL:0123456"}, index)
    with pytest.raises(co.TermError) as e:
        co.set_terms(a, "nope", {"B": "CL:0000236"})
    assert e.value.status == 404
    assert co.TERMS_KEY not in a.uns


def test_rename_keeps_the_term_and_merge_takes_the_targets():
    a = _adata()
    co.set_terms(a, "cell_type", {"CD14 Mono": "CL:0001054", "B": "CL:0000236", "NK": "CL:0000623"})
    co.follow_rename(a, "cell_type", "CD14 Mono", "Classical monocytes", merge=False)
    assert co.stored(a, "cell_type") == {"Classical monocytes": "CL:0001054", "B": "CL:0000236",
                                         "NK": "CL:0000623"}
    co.follow_rename(a, "cell_type", "NK", "B", merge=True)
    assert co.stored(a, "cell_type") == {"Classical monocytes": "CL:0001054", "B": "CL:0000236"}
    co.drop_columns(a, ["cell_type"])
    assert co.TERMS_KEY not in a.uns


def test_mapping_survives_an_h5ad_round_trip_with_slashes_in_labels(tmp_path):
    a = _adata(["CD4+/CD8+ T"] * 5 + ["B"] * 5)
    co.set_terms(a, "cell_type", {"CD4+/CD8+ T": "CL:0000084", "B": "CL:0000236"})
    a.write_h5ad(tmp_path / "x.h5ad")
    b = ad.read_h5ad(tmp_path / "x.h5ad")
    assert co.stored(b, "cell_type") == {"CD4+/CD8+ T": "CL:0000084", "B": "CL:0000236"}
    assert list(co.term_ids(b, "cell_type")[:6]) == ["CL:0000084"] * 5 + ["CL:0000236"]


def test_unlabelled_cells_leave_a_mapping_incomplete():
    a = _adata()
    a.obs["cell_type"] = a.obs["cell_type"].cat.add_categories([])
    a.obs.loc["c0", "cell_type"] = np.nan
    co.set_terms(a, "cell_type", {"CD14 Mono": "CL:0001054", "B": "CL:0000236", "NK": "CL:0000623"})
    s = co.summary(a, "cell_type")
    assert s["missing"] == [] and s["n_cells_unlabelled"] == 1 and not s["complete"]
    assert co.term_ids(a, "cell_type") is None


# ---------------------------------------------------------------------------
# Loading: the obo alone, fetched only when allowed


def test_get_index_fetches_only_the_ontology(tmp_path, monkeypatch):
    monkeypatch.setattr(mdb, "ONTOLOGY_URL", "https://cl.test/cl-basic.obo")
    monkeypatch.setattr(mdb, "ONTOLOGY_MIN_TERMS", 5)
    co._cache.clear()
    calls: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(str(req.url))
        if str(req.url) == "https://cl.test/cl-basic.obo":
            return httpx.Response(200, content=FIX.read_bytes())
        return httpx.Response(404)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    idx = co.get_index(tmp_path / "mdb", auto_download=True, client=client)
    assert calls == ["https://cl.test/cl-basic.obo"]  # no marker database
    assert idx.version == "cl/releases/2026-06-08/cl-basic.owl" and "CL:0000623" in idx.terms
    assert idx.terms["CL:0000623"].definition  # parsed from the raw obo, with definitions
    assert co.get_index(tmp_path / "mdb", auto_download=True, client=client) is idx
    co._cache.clear()


def test_get_index_says_why_when_it_cannot(tmp_path, monkeypatch):
    co._cache.clear()
    with pytest.raises(co.OntologyUnavailable, match="MARKER_DB_AUTO_DOWNLOAD is off"):
        co.get_index(tmp_path / "off", auto_download=False)
    monkeypatch.setattr(mdb, "ONTOLOGY_URL", "https://cl.test/cl-basic.obo")
    client = httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(503)))
    with pytest.raises(co.OntologyUnavailable, match="could not be downloaded"):
        co.get_index(tmp_path / "down", auto_download=True, client=client)


# ---------------------------------------------------------------------------
# API


def _register(dm, a: ad.AnnData | None = None, did: str = "onto") -> str:
    d = Path(dm.data_dir) / "uploads" / did
    d.mkdir(parents=True)
    (d / "metadata.json").write_text(json.dumps({"id": did, "name": did, "status": "ready"}))
    (a if a is not None else _adata()).write_h5ad(d / "data.h5ad")
    return did


def _derived(dm, did: str) -> ad.AnnData:
    return ad.read_h5ad(Path(dm.data_dir) / "derived" / did / "data.h5ad")


async def test_search_endpoint(client, cl_dir):
    r = await client.get(f"{API}/ontology/cell-types", params={"q": "CD14 monocyte", "limit": 3})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["terms"][0]["id"] == "CL:0001054" and len(body["terms"]) <= 3
    assert body["version"] == "cl/releases/2026-06-08/cl-basic.owl" and body["n_terms"] == 39
    t0 = time.perf_counter()
    r = await client.get(f"{API}/ontology/cell-types", params={"q": "CL:0000236"})
    assert time.perf_counter() - t0 < 0.5  # index built once (generous bound for a request)
    assert [t["name"] for t in r.json()["terms"]] == ["B cell"]


async def test_search_endpoint_503_when_offline(client, tmp_path, monkeypatch):
    monkeypatch.setattr(mdb, "marker_db_dir", lambda: tmp_path / "empty")
    monkeypatch.setattr(mdb, "auto_download_enabled", lambda: False)
    co._cache.clear()
    r = await client.get(f"{API}/ontology/cell-types", params={"q": "NK"})
    assert r.status_code == 503 and "MARKER_DB_AUTO_DOWNLOAD" in r.json()["detail"]


async def test_set_get_and_suggest_endpoints(client, dataset_manager, cl_dir):
    did = _register(dataset_manager)
    r = await client.get(f"{API}/datasets/{did}/ontology/terms", params={"column": "cell_type"})
    assert r.status_code == 200 and r.json()["n_mapped"] == 0 and r.json()["columns"] == []

    r = await client.post(f"{API}/datasets/{did}/ontology/terms/suggest", json={"column": "cell_type"})
    assert r.status_code == 200, r.text
    sug = {row["label"]: row["suggestions"][0]["id"] for row in r.json()["labels"]}
    assert sug == {"CD14 Mono": "CL:0001054", "B": "CL:0000236", "NK": "CL:0000623"}
    assert co.TERMS_KEY not in ad.read_h5ad(Path(dataset_manager.data_dir) / "uploads" / did / "data.h5ad").uns
    assert not (Path(dataset_manager.data_dir) / "derived" / did).exists()  # suggesting saves nothing

    r = await client.post(f"{API}/datasets/{did}/ontology/terms",
                          json={"column": "cell_type", "terms": {"CD14 Mono": "CL:0001054", "B": "CL:0000236"}})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["previous"] == {"CD14 Mono": None, "B": None} and body["missing"] == ["NK"]
    assert {x["label"]: x["name"] for x in body["labels"]}["B"] == "B cell"
    assert co.stored(_derived(dataset_manager, did), "cell_type") == {"CD14 Mono": "CL:0001054", "B": "CL:0000236"}

    r = await client.post(f"{API}/datasets/{did}/ontology/terms",
                          json={"column": "cell_type", "terms": {"NK": "CL:7777777"}})
    assert r.status_code == 422 and "not a current" in r.json()["detail"]
    r = await client.post(f"{API}/datasets/{did}/ontology/terms", json={"column": "nope", "terms": {"B": None}})
    assert r.status_code == 404

    r = await client.post(f"{API}/datasets/{did}/ontology/terms", json={"column": "cell_type", "clear": True})
    assert r.status_code == 200 and r.json()["n_mapped"] == 0
    assert co.TERMS_KEY not in _derived(dataset_manager, did).uns


async def test_labelling_with_a_term_and_undo(client, dataset_manager, cl_dir):
    did = _register(dataset_manager)
    r = await client.post(f"{API}/datasets/{did}/labels", json={
        "column": "my cells", "label": "Monocytes", "indices": [0, 1, 2, 3], "new_column": True,
        "ontology_id": "CL:0000576"})
    assert r.status_code == 200, r.text
    first = r.json()
    assert first["previous_terms"] == {"Monocytes": None}
    assert co.stored(_derived(dataset_manager, did), "my cells") == {"Monocytes": "CL:0000576"}

    # The same label again with another term: the undo puts the first one back.
    r = await client.post(f"{API}/datasets/{did}/labels", json={
        "column": "my cells", "label": "Monocytes", "indices": [4], "ontology_id": "CL:0001054"})
    second = r.json()
    assert second["previous_terms"] == {"Monocytes": "CL:0000576"}
    r = await client.post(f"{API}/datasets/{did}/labels/undo", json={
        "column": "my cells", "previous": second["previous"], "previous_terms": second["previous_terms"]})
    assert r.status_code == 200, r.text
    assert co.stored(_derived(dataset_manager, did), "my cells") == {"Monocytes": "CL:0000576"}

    # Without a term the label keeps the one it has.
    r = await client.post(f"{API}/datasets/{did}/labels", json={
        "column": "my cells", "label": "Monocytes", "indices": [5]})
    assert r.json()["previous_terms"] == {}
    r = await client.post(f"{API}/datasets/{did}/labels/undo", json={
        "column": "my cells", "previous": r.json()["previous"], "previous_terms": {}})

    # Undoing the labelling that made the column removes its terms with it.
    r = await client.post(f"{API}/datasets/{did}/labels/undo", json={
        "column": "my cells", "previous": first["previous"], "created": True,
        "previous_terms": first["previous_terms"]})
    assert r.json()["removed"] is True
    assert co.TERMS_KEY not in _derived(dataset_manager, did).uns

    r = await client.post(f"{API}/datasets/{did}/labels", json={
        "column": "x", "label": "y", "indices": [0], "new_column": True, "ontology_id": "monocyte"})
    assert r.status_code == 422
    assert "x" not in _derived(dataset_manager, did).obs.columns  # refused before any change


async def test_category_rename_and_merge_carry_the_terms(client, dataset_manager, cl_dir):
    did = _register(dataset_manager)
    await client.post(f"{API}/datasets/{did}/ontology/terms", json={
        "column": "cell_type", "terms": {"CD14 Mono": "CL:0001054", "NK": "CL:0000623"}})
    r = await client.post(f"{API}/datasets/{did}/metadata/cell_type/rename",
                          json={"old": "CD14 Mono", "new": "Classical monocytes"})
    assert r.status_code == 200, r.text
    assert co.stored(_derived(dataset_manager, did), "cell_type") == {
        "Classical monocytes": "CL:0001054", "NK": "CL:0000623"}
    # NK merged into B: B (no term) does not inherit NK's
    r = await client.post(f"{API}/datasets/{did}/metadata/cell_type/rename", json={"old": "NK", "new": "B"})
    assert r.status_code == 200
    body = (await client.get(f"{API}/datasets/{did}/ontology/terms", params={"column": "cell_type"})).json()
    assert {x["label"]: x["id"] for x in body["labels"]} == {"B": None, "Classical monocytes": "CL:0001054"}


# ---------------------------------------------------------------------------
# CELLxGENE export


class _A:
    def __init__(self, a):
        self.adata = a
        self.h5ad_path = "/nonexistent.h5ad"
        self.is_large = False


def test_cellxgene_writes_terms_only_from_a_complete_mapping():
    a = _adata()
    co.set_terms(a, "cell_type", {"CD14 Mono": "CL:0001054", "B": "CL:0000236"})
    p = cx.plan(_A(a), cx.Options(cell_type_terms="cell_type"))
    assert cx.TERM_COLUMN not in p.obs.columns
    item = {i["key"]: i for i in cx.readiness(p)["items"]}[cx.TERM_COLUMN]
    assert item["status"] == "missing" and "no term for 1 of 3 labels: NK" in item["detail"]

    co.set_terms(a, "cell_type", {"NK": "CL:0000623"})
    p = cx.plan(_A(a), cx.Options(cell_type_terms="cell_type"))
    assert list(p.obs[cx.TERM_COLUMN]) == ["CL:0001054"] * 4 + ["CL:0000236"] * 3 + ["CL:0000623"] * 3
    item = {i["key"]: i for i in cx.readiness(p)["items"]}[cx.TERM_COLUMN]
    assert item["status"] == "ok" and "From your Cell Ontology mapping of 'cell_type'" in item["detail"]
    assert any("from your Cell Ontology mapping" in n for n in p.notes)

    # not asked for: nothing is written, whatever the mapping
    assert cx.TERM_COLUMN not in cx.plan(_A(a), cx.Options()).obs.columns
    with pytest.raises(cx.ExportError):
        cx.plan(_A(a), cx.Options(cell_type_terms="nope"))


async def test_cellxgene_info_and_file_from_the_mapping(client, dataset_manager, cl_dir, tmp_path):
    from tests.test_export_formats import _build, _processed
    from tests.test_export_formats import _register as register_export

    a = _processed()
    co.set_terms(a, "cell_type", {"B": "CL:0000236", "T": "CL:0000084"})
    register_export(dataset_manager, a)
    info = (await client.get(f"{API}/datasets/fx1/export/h5ad/info", params={"format": "cellxgene"})).json()
    assert info["ontology_columns"] == [{"column": "cell_type", "complete": True, "n_labels": 2,
                                         "n_mapped": 2, "missing": []}]
    assert info["cell_type_terms"] is None
    info = (await client.get(f"{API}/datasets/fx1/export/h5ad/info",
                             params={"format": "cellxgene", "cell_type_terms": "cell_type"})).json()
    assert info["cell_type_terms"]["written"] is True
    _, content = await _build(client, "fx1", "cellxgene", cell_type_terms="cell_type")
    out = tmp_path / "x.h5ad"
    out.write_bytes(content)
    b = ad.read_h5ad(out)
    assert set(b.obs[cx.TERM_COLUMN].astype(str)) == {"CL:0000236", "CL:0000084"}


# ---------------------------------------------------------------------------
# Co-pilot


def test_copilot_fact_names_the_highlighted_labels_term(index, monkeypatch):
    monkeypatch.setattr(co, "cached", lambda: index)
    a = _adata()
    co.set_terms(a, "cell_type", {"NK": "CL:0000623"})
    lines, detail = co.grounding_lines(a, {"column": "cell_type", "value": "NK"})
    assert "CL:0000623 (natural killer cell)" in lines[-1] and "[result:ontology:cell_type]" in lines[-1]
    assert detail == "cell_type=NK: CL:0000623"
    assert co.grounding_lines(a, {"column": "cell_type", "value": "B"}) == ([], None)
    assert co.grounding_lines(a, None) == ([], None)
    assert resolve_tag("ontology:cell_type", ["result:ontology:cell_type", "result:groups:cell_type"]) == [
        "result:ontology:cell_type"]


def test_copilot_context_carries_the_fact(index, monkeypatch):
    from scview.core import assistant

    monkeypatch.setattr(co, "cached", lambda: index)
    a = _adata()
    co.set_terms(a, "cell_type", {"NK": "CL:0000623"})
    monkeypatch.setattr(assistant, "build_grounding_context", lambda adaptor, facets: ("", []))
    ctx, sources = assistant._assemble(
        _A(a), extra_context="", extra_sources=None, app_context="", app_sources=None,
        view_context={"highlighted": {"column": "cell_type", "value": "NK"}},
        include_data_grounding=True)
    assert "CL:0000623" in ctx
    assert [s.ref for s in sources] == ["result:ontology:cell_type"]
