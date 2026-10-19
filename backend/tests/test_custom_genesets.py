"""Custom gene-set collections: parsing an upload, storing it, and the endpoints
that upload, list, rename and delete one."""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pytest

from scview.core import custom_genesets as cg
from scview.core.custom_genesets import (
    CustomCollectionError,
    CustomGeneSetStore,
    GeneSetFileError,
    coverage,
    is_custom_id,
    parse_gene_set_file,
)

API = "/api/v1"

GMT = (
    "T_PROGRAM\tT cell genes\tcd3e\t CD3D \tCd3g\n"
    "B_PROGRAM\t\tMS4A1\tCD79A\tCD79A\n"
)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def test_gmt_genes_are_trimmed_upper_cased_and_deduplicated():
    p = parse_gene_set_file(GMT.encode(), "mine.gmt")
    assert p.format == "gmt"
    assert p.sets == {"T_PROGRAM": ["CD3E", "CD3D", "CD3G"], "B_PROGRAM": ["MS4A1", "CD79A"]}
    assert p.descriptions == {"T_PROGRAM": "T cell genes"}
    assert any("repeated gene" in w for w in p.warnings)


def test_gmt_skips_blank_and_comment_lines_and_crlf():
    text = "# exported 2026\r\n\r\nS1\tna\tA\tB\r\n"
    assert parse_gene_set_file(text.encode(), "x.gmt").sets == {"S1": ["A", "B"]}


def test_gmt_strips_enrichr_weights_and_a_bom():
    data = "﻿S1\t\tTP53,1.0\tMYC,0.5\n".encode()
    assert parse_gene_set_file(data, "lib.gmt").sets == {"S1": ["TP53", "MYC"]}


def test_gmt_problems_are_all_listed_and_nothing_parsed():
    text = (
        "GOOD\t\tA\tB\n"
        "EMPTY\tno genes here\n"
        "GOOD\t\tC\n"
        "SPACES NOT TABS A B C\n"
        "\t\tA\n"
        "SPACEY\t\tCD3E CD3D\n"
    )
    with pytest.raises(GeneSetFileError) as exc:
        parse_gene_set_file(text.encode(), "bad.gmt")
    err = exc.value
    assert "5 problems" in err.message and "nothing was saved" in err.message
    joined = "\n".join(err.problems)
    assert "line 2: set 'EMPTY' lists no genes" in joined
    assert "line 3: set 'GOOD' appears again (first on line 1)" in joined
    assert "line 4" in joined and "tab" in joined
    assert "line 5: the gene set has no name" in joined
    assert "line 6" in joined and "contains a space" in joined


def test_problem_list_is_capped():
    text = "".join(f"S{i}\tdesc\n" for i in range(50))
    with pytest.raises(GeneSetFileError) as exc:
        parse_gene_set_file(text.encode(), "x.gmt")
    assert len(exc.value.problems) == cg.MAX_PROBLEMS_LISTED + 1
    assert exc.value.problems[-1] == "...and 30 more."


def test_set_and_gene_count_caps(monkeypatch):
    monkeypatch.setattr(cg, "MAX_SETS", 2)
    monkeypatch.setattr(cg, "MAX_GENES_PER_SET", 3)
    text = "A\t\tG1\tG2\tG3\tG4\nB\t\tG1\nC\t\tG1\n"
    with pytest.raises(GeneSetFileError) as exc:
        parse_gene_set_file(text.encode(), "x.gmt")
    joined = "\n".join(exc.value.problems)
    assert "set 'A' (line 1) has 4 genes; the limit is 3" in joined
    assert "more than 2 gene sets" in joined


def test_csv_long_format_with_header():
    text = "gene_set,gene\nT,cd3e\nT,CD3D\nB,ms4a1\nT,CD3E\n"
    p = parse_gene_set_file(text.encode(), "sets.csv")
    assert p.format == "csv"
    assert p.sets == {"T": ["CD3E", "CD3D"], "B": ["MS4A1"]}


def test_tsv_without_header_and_quoted_csv():
    assert parse_gene_set_file(b"T\tCD3E\nT\tCD3D\n", "s.tsv").sets == {"T": ["CD3E", "CD3D"]}
    text = '"My set, v2",CD3E\n"My set, v2",CD3D\n'
    assert parse_gene_set_file(text.encode(), "s.csv").sets == {"My set, v2": ["CD3E", "CD3D"]}


def test_csv_rows_with_the_wrong_column_count_are_reported():
    text = "T,CD3E\nT\nT,CD3D,extra\n"
    with pytest.raises(GeneSetFileError) as exc:
        parse_gene_set_file(text.encode(), "s.csv")
    joined = "\n".join(exc.value.problems)
    assert "line 2: expected two comma-separated columns" in joined
    assert "line 3: expected two comma-separated columns, gene_set and gene; found 3" in joined


def test_txt_is_sniffed_as_gmt_or_table():
    assert parse_gene_set_file(b"S\tdesc\tA\tB\n", "s.txt").format == "gmt"
    assert parse_gene_set_file(b"S\tA\nS\tB\n", "s.txt").format == "tsv"
    assert parse_gene_set_file(b"S,A\nS,B\n", "s.txt").format == "csv"


@pytest.mark.parametrize(
    "data, name, needle",
    [
        (b"S\t\tA\n", "sets.xlsx", "unsupported file type '.xlsx'"),
        (b"S\t\tA\n", "noext", "unsupported file type '(none)'"),
        (b"\x1f\x8b\x08\x00\x00\x00", "sets.gmt", "binary"),
        ("S\t\tA\xe9\n".encode("latin-1"), "sets.gmt", "not UTF-8"),
        (b"\n\n# only comments\n", "sets.gmt", "no gene sets found"),
    ],
)
def test_unusable_files_say_why(data, name, needle):
    with pytest.raises(GeneSetFileError) as exc:
        parse_gene_set_file(data, name)
    assert needle in exc.value.message


def test_file_name_never_leaks_a_path_into_messages():
    with pytest.raises(GeneSetFileError) as exc:
        parse_gene_set_file(b"", "../../etc/sets.gmt")
    assert ".." not in exc.value.message and exc.value.message.startswith("sets.gmt:")


def test_coverage_ignores_case():
    sets = {"A": ["CD3E", "CD3D", "NOPE"], "B": ["XYZ"]}
    assert coverage(sets, ["Cd3e", "Cd3d", "Actb"]) == {
        "n_genes": 4, "n_genes_present": 2, "n_sets_with_genes": 1,
    }


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


def _parsed(text: str = GMT):
    return parse_gene_set_file(text.encode(), "mine.gmt")


def test_store_round_trip_survives_a_new_instance(tmp_path):
    store = CustomGeneSetStore(tmp_path)
    meta = store.create("My sets", _parsed(), source_filename="../mine.gmt")
    assert is_custom_id(meta["id"]) and "." not in meta["id"]
    assert meta["n_sets"] == 2 and meta["n_genes"] == 5
    assert meta["source_filename"] == "mine.gmt"
    assert meta["category"] == "Custom"

    fresh = CustomGeneSetStore(tmp_path)  # e.g. after a restart
    assert [m["name"] for m in fresh.list()] == ["My sets"]
    assert fresh.get_sets(meta["id"])["T_PROGRAM"] == ["CD3E", "CD3D", "CD3G"]
    assert fresh.get_descriptions(meta["id"]) == {"T_PROGRAM": "T cell genes"}
    assert (tmp_path / "genesets" / "custom" / meta["id"] / "meta.json").is_file()


def test_store_rename_and_delete(tmp_path):
    store = CustomGeneSetStore(tmp_path)
    a = store.create("A", _parsed())
    store.create("B", _parsed())
    with pytest.raises(CustomCollectionError, match="already exists"):
        store.rename(a["id"], "  b ")
    assert store.rename(a["id"], "A renamed")["name"] == "A renamed"
    assert store.get_sets(a["id"]) is not None  # contents untouched
    assert store.delete(a["id"]) is True
    assert store.get_meta(a["id"]) is None and store.get_sets(a["id"]) is None
    assert store.delete(a["id"]) is False
    assert [m["name"] for m in store.list()] == ["B"]


def test_store_rejects_duplicate_names_and_too_many(tmp_path, monkeypatch):
    store = CustomGeneSetStore(tmp_path)
    store.create("Same", _parsed())
    with pytest.raises(CustomCollectionError, match="already exists"):
        store.create("same", _parsed())
    monkeypatch.setattr(cg, "MAX_COLLECTIONS", 1)
    with pytest.raises(CustomCollectionError, match="delete one first"):
        store.create("Other", _parsed())


@pytest.mark.parametrize("bad", ["../x", "custom-..", "h.all", "custom-a/b", ""])
def test_store_ignores_ids_it_did_not_make(tmp_path, bad):
    store = CustomGeneSetStore(tmp_path)
    assert store.get_meta(bad) is None
    assert store.get_sets(bad) is None
    assert store.delete(bad) is False


def test_store_skips_half_written_and_broken_collections(tmp_path):
    store = CustomGeneSetStore(tmp_path)
    good = store.create("Good", _parsed())
    root = tmp_path / "genesets" / "custom"
    (root / ".tmp-custom-abc").mkdir()
    (root / "custom-broken").mkdir()
    (root / "custom-broken" / "meta.json").write_text("{not json")
    assert [m["id"] for m in store.list()] == [good["id"]]


def test_store_keeps_only_a_few_collections_in_memory(tmp_path, monkeypatch):
    monkeypatch.setattr(cg, "MAX_CACHED_COLLECTIONS", 2)
    store = CustomGeneSetStore(tmp_path)
    ids = [store.create(f"C{i}", _parsed())["id"] for i in range(3)]
    assert list(store._sets) == ids[1:]
    assert store.get_sets(ids[0])["B_PROGRAM"] == ["MS4A1", "CD79A"]  # read back from disk
    assert list(store._sets) == [ids[2], ids[0]]


def test_store_search_and_exact_lookup(tmp_path):
    store = CustomGeneSetStore(tmp_path)
    meta = store.create("Mine", _parsed())
    hits = store.search("program")
    assert {h["name"] for h in hits} == {"T_PROGRAM", "B_PROGRAM"}
    assert hits[0]["collection"] == meta["id"] and hits[0]["collection_name"] == "Mine"
    assert store.get_gene_set("B_PROGRAM")["genes"] == ["MS4A1", "CD79A"]
    assert store.get_gene_set("b_program") is None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


async def _upload(client, text: str = GMT, filename: str = "mine.gmt", name: str | None = None):
    data = {"name": name} if name is not None else {}
    return await client.post(
        f"{API}/genesets/custom/upload",
        files={"file": (filename, text.encode(), "text/plain")},
        data=data,
    )


async def test_upload_list_rename_delete(client, tmp_data_dir):
    r = await _upload(client, name="  Lab signatures ")
    assert r.status_code == 201, r.text
    meta = r.json()
    assert meta["name"] == "Lab signatures"
    assert meta["n_sets"] == 2 and meta["warnings"]
    cid = meta["id"]

    listed = (await client.get(f"{API}/genesets/custom")).json()["collections"]
    assert [c["id"] for c in listed] == [cid]

    r = await client.patch(f"{API}/genesets/custom/{cid}", json={"name": "Renamed"})
    assert r.status_code == 200 and r.json()["name"] == "Renamed"

    r = await client.delete(f"{API}/genesets/custom/{cid}")
    assert r.status_code == 204
    assert (await client.get(f"{API}/genesets/custom")).json()["collections"] == []
    assert not (tmp_data_dir / "genesets" / "custom" / cid).exists()


async def test_upload_name_defaults_to_the_file_name(client):
    r = await _upload(client, filename="kegg_subset.gmt")
    assert r.json()["name"] == "kegg_subset"


async def test_upload_bad_file_is_400_listing_problems(client, tmp_data_dir):
    r = await _upload(client, text="A\tdesc\nA\t\tX\n")
    assert r.status_code == 400
    detail = r.json()["detail"]
    assert "2 problems" in detail["message"]
    assert any("lists no genes" in p for p in detail["problems"])
    assert (await client.get(f"{API}/genesets/custom")).json()["collections"] == []


async def test_upload_too_large_is_413(client, monkeypatch):
    monkeypatch.setattr("scview.api.v1.custom_genesets.MAX_FILE_BYTES", 10)
    r = await _upload(client)
    assert r.status_code == 413


async def test_upload_duplicate_name_and_bad_rename_are_400(client):
    await _upload(client, name="Mine")
    r = await _upload(client, name="mine")
    assert r.status_code == 400 and "already exists" in r.json()["detail"]
    cid = (await client.get(f"{API}/genesets/custom")).json()["collections"][0]["id"]
    for bad in ["", "  ", "a\nb"]:
        r = await client.patch(f"{API}/genesets/custom/{cid}", json={"name": bad})
        assert r.status_code == 400, bad


@pytest.mark.parametrize("cid", ["nope", "custom-0000", "%2E%2E", "h.all"])
async def test_rename_or_delete_unknown_is_404(client, cid):
    assert (await client.patch(f"{API}/genesets/custom/{cid}", json={"name": "x"})).status_code == 404
    assert (await client.delete(f"{API}/genesets/custom/{cid}")).status_code == 404


def _register(dm, did: str = "ds1", genes: list[str] | None = None) -> str:
    d = Path(dm.data_dir) / "uploads" / did
    d.mkdir(parents=True)
    (d / "metadata.json").write_text(json.dumps({"id": did, "name": did, "status": "ready"}))
    genes = genes or ["Cd3e", "Cd3d", "Ms4a1", "Actb"]
    a = ad.AnnData(np.ones((4, len(genes)), dtype="float32"))
    a.var_names = genes
    a.obs_names = [f"C{i}" for i in range(4)]
    a.write_h5ad(d / "data.h5ad")
    return did


async def test_dataset_listing_reports_genes_present(client, dataset_manager):
    did = _register(dataset_manager)
    cid = (await _upload(client)).json()["id"]
    r = await client.get(f"{API}/datasets/{did}/genesets/custom")
    assert r.status_code == 200, r.text
    (col,) = r.json()["collections"]
    assert col["id"] == cid
    assert col["n_sets"] == 2 and col["n_genes"] == 5
    assert col["n_genes_present"] == 3  # CD3E, CD3D, MS4A1 against a mouse dataset
    assert col["n_sets_with_genes"] == 2

    # The browser's collection list carries them too, as category "Custom",
    # even with no MSigDB configured.
    r = await client.get(f"{API}/datasets/{did}/genesets/collections")
    custom = [c for c in r.json()["collections"] if c["category"] == "Custom"]
    assert [c["id"] for c in custom] == [cid]
    assert custom[0]["available"] is True and custom[0]["n_genes_present"] == 3


async def test_dataset_listing_unknown_dataset_is_404(client):
    assert (await client.get(f"{API}/datasets/ghost/genesets/custom")).status_code == 404


# ---------------------------------------------------------------------------
# Browsing and scoring
# ---------------------------------------------------------------------------


async def test_search_lists_and_finds_custom_sets_without_msigdb(client, dataset_manager):
    did = _register(dataset_manager)
    text = "IFN_RESPONSE_LONG\t\tCD3E\nIFN\t\tCD3D\nOTHER\t\tMS4A1\n"
    cid = (await _upload(client, text=text)).json()["id"]

    r = await client.get(f"{API}/datasets/{did}/genesets/search",
                         params={"q": "", "collection": cid, "limit": 200})
    body = r.json()
    assert [x["name"] for x in body["results"]] == ["IFN_RESPONSE_LONG", "IFN", "OTHER"]
    assert body["results"][0]["collection_name"] == "mine"
    assert "message" not in body

    # Across all collections, an exact name match wins over a longer name
    # containing it, so a result term looked up by name finds itself.
    r = await client.get(f"{API}/datasets/{did}/genesets/search", params={"q": "IFN", "limit": 1})
    assert [x["name"] for x in r.json()["results"]] == ["IFN"]
    r = await client.get(f"{API}/datasets/{did}/genesets/search", params={"q": "ifn"})
    assert {x["name"] for x in r.json()["results"]} == {"IFN_RESPONSE_LONG", "IFN"}


def _register_scoring(dm, did: str = "sc1") -> str:
    d = Path(dm.data_dir) / "uploads" / did
    d.mkdir(parents=True)
    (d / "metadata.json").write_text(json.dumps({"id": did, "name": did, "status": "ready"}))
    X = np.random.default_rng(0).poisson(0.5, size=(60, 80)).astype("float32")
    a = ad.AnnData(X)
    a.var_names = [f"Gene{i}" for i in range(80)]  # mouse-style capitalisation
    a.obs_names = [f"C{i}" for i in range(60)]
    a.write_h5ad(d / "data.h5ad")
    return did


async def test_scoring_matches_upper_case_symbols_to_a_mouse_dataset(client, dataset_manager):
    did = _register_scoring(dataset_manager)
    body = {"gene_set": ["GENE0", "gene1", "Gene2", "GENE0", "NOSUCH"], "score_name": "s"}
    r = await client.post(f"{API}/datasets/{did}/genesets/score", json=body)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["genes_found"] == ["Gene0", "Gene1", "Gene2"]
    assert data["genes_missing"] == ["NOSUCH"]
    assert data["n_cells"] == 60


# ---------------------------------------------------------------------------
# Saving the Gene lists tab's lists as a collection
# ---------------------------------------------------------------------------


def test_lists_are_checked_and_cleaned_like_an_upload():
    p = cg.parse_gene_set_lists({" T cells ": ["cd3e", " CD3D ", "Cd3e"], "B": ["MS4A1"]})
    assert p.format == "lists"
    assert p.sets == {"T cells": ["CD3E", "CD3D"], "B": ["MS4A1"]}
    assert any("repeated gene" in w for w in p.warnings)


def test_bad_lists_are_all_reported():
    with pytest.raises(GeneSetFileError) as exc:
        cg.parse_gene_set_lists(
            {"": ["A"], "Empty": [], "Spacey": ["CD3E CD3D"], "a ": ["X"], "a": ["Y"]}
        )
    joined = "\n".join(exc.value.problems)
    assert "4 problems" in exc.value.message
    assert "no name" in joined
    assert "'Empty' has no genes" in joined
    assert "contains a space" in joined
    assert "'a' appears again" in joined


def test_lists_over_the_gene_limit_are_refused(monkeypatch):
    monkeypatch.setattr(cg, "MAX_GENES_PER_SET", 2)
    with pytest.raises(GeneSetFileError) as exc:
        cg.parse_gene_set_lists({"Big": ["A", "B", "C"]})
    assert "the limit is 2" in exc.value.problems[0]


async def test_save_lists_as_a_collection(client, dataset_manager):
    r = await client.post(
        f"{API}/genesets/custom/from-lists",
        json={"name": " My lists ", "sets": {"T": ["Cd3e", "CD3D"], "B": ["ms4a1"]}},
    )
    assert r.status_code == 201, r.text
    meta = r.json()
    assert meta["name"] == "My lists" and meta["n_sets"] == 2 and meta["n_genes"] == 3
    assert is_custom_id(meta["id"])
    # Matched like any custom collection: the mouse dataset's genes count as present.
    did = _register(dataset_manager)
    listed = (await client.get(f"{API}/datasets/{did}/genesets/custom")).json()["collections"]
    assert [(c["name"], c["n_genes_present"]) for c in listed] == [("My lists", 3)]


async def test_save_lists_errors_are_400_and_save_nothing(client):
    url = f"{API}/genesets/custom/from-lists"
    r = await client.post(url, json={"name": "X", "sets": {"Empty": []}})
    assert r.status_code == 400
    assert any("has no genes" in p for p in r.json()["detail"]["problems"])
    r = await client.post(url, json={"name": "X", "sets": {}})
    assert r.status_code == 400
    assert (await client.post(url, json={"name": "Mine", "sets": {"A": ["X"]}})).status_code == 201
    r = await client.post(url, json={"name": "mine", "sets": {"A": ["X"]}})
    assert r.status_code == 400 and "already exists" in r.json()["detail"]
    r = await client.post(url, json={"name": "Y", "sets": {"A": "X"}})
    assert r.status_code == 422
    assert len((await client.get(f"{API}/genesets/custom")).json()["collections"]) == 1
