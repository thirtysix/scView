"""MSigDB source ids on enrichment rows: each set's exactSource (R-HSA-/R-MMU-
Reactome, WP... WikiPathways, N... KEGG Medicus) and externalDetailsURL, kept by
the loader and carried by ORA and GSEA rows as ``source_id`` / ``source_url``,
so a Reactome row can open its pathway diagram (core/reactome.py)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scview.config import Settings
from scview.core import gsea
from scview.core.geneset_sources import label_records
from scview.core.msigdb_loader import MSigDBLoader
from tests.test_enrichment_endpoints import B_SET, GENES, T_SET, _register

API = "/api/v1"


@pytest.fixture()
def msigdb_dir(tmp_path: Path) -> Path:
    d = tmp_path / "msigdb"
    d.mkdir()
    (d / "c2.cp.reactome.v2026.1.Hs.json").write_text(json.dumps({
        "REACTOME_T_PROGRAM": {"geneSymbols": T_SET, "exactSource": "R-HSA-1059683",
                               "externalDetailsURL": "https://www.reactome.org/content/detail/R-HSA-1059683"},
        "REACTOME_B_PROGRAM": {"geneSymbols": B_SET, "exactSource": "R-HSA-909733",
                               "externalDetailsURL": "javascript:alert(1)"},
    }))
    (d / "c2.cp.kegg_medicus.v2026.1.Hs.json").write_text(json.dumps({
        "KEGG_MEDICUS_T": {"geneSymbols": T_SET, "exactSource": "N00001",
                           "externalDetailsURL": "https://www.kegg.jp/entry/N00001"},
    }))
    (d / "h.all.v2026.1.Hs.json").write_text(json.dumps({"HALLMARK_T": {"geneSymbols": T_SET}}))
    return d


@pytest.fixture()
def test_settings(tmp_data_dir: Path, msigdb_dir: Path) -> Settings:
    return Settings(DATA_DIR=str(tmp_data_dir), DEEPINFRA_API_KEY="", RAG_DATABASE_URL="",
                    MSIGDB_DIR=str(msigdb_dir))


@pytest.fixture(autouse=True)
def _fresh():
    gsea.clear_cache()
    yield
    gsea.clear_cache()


def test_the_loader_keeps_exact_source_and_a_safe_url(msigdb_dir):
    loader = MSigDBLoader(str(msigdb_dir))
    assert loader.get_sources("c2.cp.reactome") == {
        "REACTOME_T_PROGRAM": ("R-HSA-1059683", "https://www.reactome.org/content/detail/R-HSA-1059683"),
        "REACTOME_B_PROGRAM": ("R-HSA-909733", ""),  # not an http(s) URL: dropped
    }
    assert loader.get_sources("h.all") == {}
    assert loader.get_collection_as_dict("c2.cp.reactome")["REACTOME_T_PROGRAM"] == T_SET  # unchanged


async def test_ora_rows_carry_the_source(client, dataset_manager):
    did = _register(dataset_manager)
    r = await client.post(f"{API}/datasets/{did}/enrichment/compute-local", json={
        "column": "cell_type", "group": "T", "n_genes": 10,
        "collections": ["c2.cp.reactome", "c2.cp.kegg_medicus", "h.all"]})
    assert r.status_code == 200, r.text
    rows = {x["term"]: x for x in r.json()["results"]}
    assert rows["REACTOME_T_PROGRAM"]["source_id"] == "R-HSA-1059683"
    assert rows["REACTOME_T_PROGRAM"]["source_url"].startswith("https://www.reactome.org/")
    assert rows["KEGG_MEDICUS_T"]["source_id"] == "N00001"
    assert rows["HALLMARK_T"]["source_id"] == "" and rows["HALLMARK_T"]["source_url"] == ""
    # served again from the dataset's cache: still there
    again = await client.post(f"{API}/datasets/{did}/enrichment/compute-local", json={
        "column": "cell_type", "group": "T", "n_genes": 10,
        "collections": ["c2.cp.reactome", "c2.cp.kegg_medicus", "h.all"]})
    assert again.json()["source"] == "precomputed"
    assert {x["term"]: x["source_id"] for x in again.json()["results"]}["REACTOME_T_PROGRAM"] == "R-HSA-1059683"


async def test_gsea_rows_carry_the_source(client, dataset_manager):
    did = _register(dataset_manager)
    genes = [{"gene": g, "logfoldchange": 3.0 - i * 0.1, "pval": 0.01, "score": 30.0 - i}
             for i, g in enumerate(GENES)]
    r = await client.post(f"{API}/datasets/{did}/enrichment/gsea", json={
        "genes": genes, "collections": ["c2.cp.reactome"], "min_size": 5, "permutations": 100})
    assert r.status_code == 200, r.text
    rows = r.json()["up"]["results"] + r.json()["down"]["results"]
    by = {x["term"]: x["source_id"] for x in rows}
    assert by["REACTOME_T_PROGRAM"] == "R-HSA-1059683" and by["REACTOME_B_PROGRAM"] == "R-HSA-909733"


def test_a_record_cached_before_sources_gets_them_from_the_loader(msigdb_dir):
    loader = MSigDBLoader(str(msigdb_dir))
    old = [{"term": "REACTOME_T_PROGRAM", "collection": "c2.cp.reactome"},
           {"term": "SOMETHING_ELSE", "collection": "c2.cp.reactome"}]
    out = label_records(old, None, loader)
    assert out[0]["source_id"] == "R-HSA-1059683" and out[1]["source_id"] == ""
    assert "source_id" not in old[0]  # copies
    assert "source_id" not in label_records(old, None)[0]  # without a loader: as before
