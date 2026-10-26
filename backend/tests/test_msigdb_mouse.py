"""Mouse MSigDB collections: fetched on first use, the default for a mouse dataset.

Fixture JSON stands in for Broad's files (never the network). A mouse dataset
(title-case symbols) gets MH, M2 and M5 by default with no warning; human
collections on it still work, matched ignoring case, with the warning; a human
dataset sees exactly the collections, order and results it always did.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import anndata as ad
import httpx
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
from scipy.stats import hypergeom

from scview.config import Settings
from scview.core import msigdb_fetch
from scview.core import msigdb_species as ms
from scview.core.geneset_sources import match_query_case, resolve_gene_sets
from scview.core.msigdb_loader import (
    DEFAULT_COLLECTIONS,
    MOUSE_DEFAULT_COLLECTIONS,
    MSIGDB_COLLECTIONS,
    MSigDBLoader,
    collection_species,
    defaults_for,
)
from scview.core.uns_keys import enrichment_key, markers_key

API = "/api/v1"

MOUSE_T = ["Cd3e", "Cd3d", "Lck", "Zap70", "Cd2", "Cd5", "Cd6", "Cd28", "Il7r", "Tcf7"]
MOUSE_B = ["Cd19", "Cd79a", "Cd79b", "Ms4a1", "Pax5", "Ebf1", "Blk", "Bank1", "Cd22", "H2-Ab1"]
MOUSE = MOUSE_T + MOUSE_B + [f"Gm{i}" for i in range(60)]
HUMAN = [g.upper() for g in MOUSE_T] + [g.upper() for g in MOUSE_B[:9]] + ["HLA-DRB1"] + [
    f"G{i}" for i in range(60)
]
# The human B set names HLA-DRB1 where MH has the mouse ortholog H2-Ab1.
H_SETS = {"HALLMARK_T_PROGRAM": {"geneSymbols": HUMAN[0:10]},
          "HALLMARK_B_PROGRAM": {"geneSymbols": HUMAN[10:20]}}
MH_SETS = {"HALLMARK_T_PROGRAM": {"geneSymbols": MOUSE_T},
           "HALLMARK_B_PROGRAM": {"geneSymbols": MOUSE_B}}


def _write(d: Path, cid: str, sets: dict, release: str) -> None:
    (d / f"{cid}.v{release}.json").write_text(json.dumps(sets))


def _mouse_files(d: Path, ids=MOUSE_DEFAULT_COLLECTIONS) -> None:
    for cid in ids:
        sets = MH_SETS if cid == "mh.all" else {f"{cid.upper()}_FILLER": {"geneSymbols": MOUSE[40:46]}}
        _write(d, cid, sets, "2026.2.Mm")


@pytest.fixture()
def msigdb_dir(tmp_path: Path) -> Path:
    d = tmp_path / "msigdb"
    d.mkdir()
    _write(d, "h.all", H_SETS, "2026.1.Hs")
    _write(d, "c8.all", {"C8_FILLER": {"geneSymbols": HUMAN[40:46]}}, "2026.1.Hs")
    return d


@pytest.fixture()
def test_settings(tmp_data_dir: Path, msigdb_dir: Path) -> Settings:
    # Off by default here, so no test can reach the network.
    return Settings(DATA_DIR=str(tmp_data_dir), DEEPINFRA_API_KEY="", RAG_DATABASE_URL="",
                    MSIGDB_DIR=str(msigdb_dir), MSIGDB_AUTO_DOWNLOAD=False)


def _adata(genes: list[str]) -> ad.AnnData:
    rng = np.random.default_rng(0)
    X = rng.poisson(1.0, (90, len(genes))).astype("float32")
    X[:30, 0:10] += 8
    X[30:60, 10:20] += 8
    a = ad.AnnData(X)
    a.var_names = genes
    a.obs_names = [f"C{i}" for i in range(90)]
    a.obs["cell_type"] = pd.Categorical(["T"] * 30 + ["B"] * 30 + ["NK"] * 30)
    sc.tl.rank_genes_groups(a, "cell_type", method="wilcoxon", key_added=markers_key("cell_type"))
    return a


def _register(dm, did: str, genes: list[str]) -> str:
    d = Path(dm.data_dir) / "uploads" / did
    d.mkdir(parents=True)
    (d / "metadata.json").write_text(json.dumps({"id": did, "name": did, "status": "ready"}))
    _adata(genes).write_h5ad(d / "data.h5ad")
    return did


def _lists(genes: list[str], **kw) -> dict:
    return {"up": genes[0:6], "down": genes[10:16], "background": genes, **kw}


def _terms(side: dict) -> dict[str, dict]:
    return {r["term"]: r for r in side["results"]}


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


# ---------------------------------------------------------------------------
# Registry and loader
# ---------------------------------------------------------------------------


def test_every_collection_has_a_species_and_mouse_defaults_mirror_human():
    assert {c["species"] for c in MSIGDB_COLLECTIONS} == {"human"}
    assert collection_species("mh.all") == "mouse" and collection_species("h.all") == "human"
    assert collection_species("custom-abc") is None
    assert defaults_for("mouse")[0] == "mh.all" and defaults_for("human") == DEFAULT_COLLECTIONS
    assert defaults_for("unknown") == DEFAULT_COLLECTIONS
    assert set(MOUSE_DEFAULT_COLLECTIONS) == set(msigdb_fetch.MOUSE_SHA256) & set(MOUSE_DEFAULT_COLLECTIONS)
    assert set(msigdb_fetch.MOUSE_SHA256) == set(msigdb_fetch.MOUSE_IDS)


def test_human_listing_is_unchanged_and_mouse_listing_puts_mouse_first(msigdb_dir: Path):
    _mouse_files(msigdb_dir)
    loader = MSigDBLoader(str(msigdb_dir))
    human = loader.available_collections()
    assert [c["id"] for c in human] == [c["id"] for c in MSIGDB_COLLECTIONS]
    assert [c["id"] for c in loader.available_collections("human")] == [c["id"] for c in human]
    mouse = loader.available_collections("mouse")
    ids = [c["id"] for c in mouse]
    assert ids[0] == "mh.all" and ids.index("h.all") > ids.index("m8.all")
    assert {c["id"]: c["available"] for c in mouse}["mh.all"] is True
    tree = loader.available_collections_hierarchical("mouse")
    assert tree[0]["category"] == "MH" and tree[0]["species"] == "mouse"
    assert any(c["category"] == "H" and c["species"] == "human" for c in tree)
    # A search across all collections finds the mouse Hallmark set first on a mouse dataset.
    assert loader.get_gene_set("HALLMARK_T_PROGRAM", species="mouse")["collection"] == "mh.all"
    assert loader.get_gene_set("HALLMARK_T_PROGRAM")["collection"] == "h.all"


# ---------------------------------------------------------------------------
# Fetch on first use
# ---------------------------------------------------------------------------


def test_fetch_mouse_writes_pinned_files_and_meta(tmp_path: Path, monkeypatch):
    payload = json.dumps(MH_SETS).encode()
    monkeypatch.setattr(msigdb_fetch, "MOUSE_SHA256", {
        cid: hashlib.sha256(payload).hexdigest() for cid in msigdb_fetch.MOUSE_IDS})
    urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        urls.append(str(request.url))
        return httpx.Response(200, content=payload)

    done, errors = msigdb_fetch.fetch_mouse(tmp_path, client=_client(handler))
    assert errors == {} and set(done) == set(msigdb_fetch.MOUSE_IDS)
    assert f"{msigdb_fetch.MOUSE_BASE_URL}/mh.all.v2026.2.Mm.json" in urls
    meta = json.loads((tmp_path / msigdb_fetch.MOUSE_META).read_text())
    assert meta["release"] == "2026.2.Mm" and "CC BY 4.0" in meta["licence"]
    assert "mh.all.v2026.2.Mm.json" in meta["sha256"]
    assert MSigDBLoader(str(tmp_path)).collection_size("mh.all") == 2
    # The human first-start check ignores the mouse files, and vice versa.
    assert len(msigdb_fetch.missing_collections(tmp_path)) == len(MSIGDB_COLLECTIONS)
    assert msigdb_fetch.missing_mouse(tmp_path) == []
    assert msigdb_fetch.mouse_status(tmp_path, True)["state"] == "ready"


def test_a_checksum_mismatch_or_offline_fetch_leaves_nothing(tmp_path: Path):
    def bad(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"TAMPERED": {"geneSymbols": ["X"]}})

    done, errors = msigdb_fetch.fetch_mouse(tmp_path, client=_client(bad))
    assert done == [] and "sha256 mismatch" in errors["mh.all"]
    assert not list(tmp_path.glob("*.json*"))

    def offline(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no network")

    with pytest.raises(msigdb_fetch.MSigDBUnavailable, match="failed .*no network"):
        msigdb_fetch.require(["mh.all"], tmp_path, True, client=_client(offline))
    assert msigdb_fetch.mouse_status(tmp_path, True)["state"] == "failed"
    with pytest.raises(msigdb_fetch.MSigDBUnavailable, match="MSIGDB_AUTO_DOWNLOAD is off"):
        msigdb_fetch.require(["mh.all"], tmp_path, False)
    # Human and custom ids never trigger a fetch.
    msigdb_fetch.require(["h.all", "custom-x"], tmp_path, True, client=_client(offline))


# ---------------------------------------------------------------------------
# Gene sets for the dataset's species
# ---------------------------------------------------------------------------


def test_warning_only_for_another_species_collections():
    assert ms.warning("mouse", ["mh.all", "m5.go.bp"]) == ""
    w = ms.warning("mouse", ["mh.all", "h.all"])
    assert "H is human gene symbols" in w and "mouse MSigDB" in w
    assert ms.warning("mouse", [], human_library=True) != ""
    assert ms.warning("human", DEFAULT_COLLECTIONS) == "" and ms.warning("unknown", ["mh.all"]) == ""
    assert "mouse gene symbols" in ms.warning("human", ["mh.all"])
    assert ms.describe(["mh.all", "m2.cp.reactome", "h.all"], "mouse") == (
        "mouse MSigDB MH, M2 Reactome; human MSigDB H (human symbols, case-matched)")


def test_mixed_selection_is_spelled_the_dataset_way(msigdb_dir: Path):
    """gseapy picks upper- or exact-case matching from the first sets only, so MH
    beside H would lose one of them; both are spelled the mouse way instead."""
    _mouse_files(msigdb_dir, ["mh.all"])
    loader = MSigDBLoader(str(msigdb_dir))
    resolved = resolve_gene_sets(["h.all", "mh.all"], loader, None, species="mouse")
    assert list(resolved.gene_sets)[:2] == ["HALLMARK_T_PROGRAM", "HALLMARK_B_PROGRAM"]
    assert resolved.term_collection["HALLMARK_T_PROGRAM"] == "mh.all"
    assert resolved.term_collection["HALLMARK_T_PROGRAM (h.all)"] == "h.all"
    assert resolved.cross_species == ["h.all"]
    sets = match_query_case(resolved, MOUSE[:6], MOUSE)
    assert sets["HALLMARK_T_PROGRAM (h.all)"] == MOUSE_T
    assert "HLA-DRB1" in sets["HALLMARK_B_PROGRAM (h.all)"]  # no mouse gene: left as it was
    # A human dataset's resolution is untouched by the species argument.
    plain = resolve_gene_sets(["h.all"], loader, None)
    assert resolve_gene_sets(["h.all"], loader, None, species="human").gene_sets == plain.gene_sets


async def test_mouse_dataset_gets_mouse_defaults_and_no_warning(client, dataset_manager, msigdb_dir):
    _mouse_files(msigdb_dir)
    did = _register(dataset_manager, "mm", MOUSE)
    r = await client.post(f"{API}/datasets/{did}/enrichment/gene-lists",
                          json=_lists(MOUSE, down=MOUSE[10:20]))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["collections"] == MOUSE_DEFAULT_COLLECTIONS
    assert body["species"] == "mouse" and body["warning"] == ""
    up, down = _terms(body["up"]), _terms(body["down"])
    assert up["HALLMARK_T_PROGRAM"]["collection"] == "mh.all"
    assert up["HALLMARK_T_PROGRAM"]["species"] == "mouse"
    assert up["HALLMARK_T_PROGRAM"]["genes"] == sorted(MOUSE_T[:6])
    # MH has the mouse gene where H names HLA-DRB1: all ten B genes are covered.
    assert down["HALLMARK_B_PROGRAM"]["overlap_count"] == 10


async def test_human_sets_on_mouse_still_match_and_warn(client, dataset_manager, msigdb_dir):
    _mouse_files(msigdb_dir, ["mh.all"])
    did = _register(dataset_manager, "mm", MOUSE)
    r = await client.post(f"{API}/datasets/{did}/enrichment/gene-lists",
                          json=_lists(MOUSE, collections=["h.all"]))
    body = r.json()
    assert r.status_code == 200 and "H is human gene symbols" in body["warning"]
    t = _terms(body["up"])["HALLMARK_T_PROGRAM"]
    # Same hypergeometric test as before mouse collections existed.
    n, k = len(MOUSE), 10
    assert t["overlap_count"] == 6
    assert t["pvalue"] == pytest.approx(hypergeom.sf(5, n, k, 6))
    assert t["species"] == "human"
    # Mouse and human together: both Hallmark versions are tested.
    r = await client.post(f"{API}/datasets/{did}/enrichment/gene-lists",
                          json=_lists(MOUSE, collections=["mh.all", "h.all"]))
    up = _terms(r.json()["up"])
    assert up["HALLMARK_T_PROGRAM"]["overlap_count"] == 6
    assert up["HALLMARK_T_PROGRAM (h.all)"]["overlap_count"] == 6


async def test_human_dataset_is_unchanged(client, dataset_manager):
    did = _register(dataset_manager, "hs", HUMAN)
    implicit = await client.post(f"{API}/datasets/{did}/enrichment/gene-lists", json=_lists(HUMAN))
    explicit = await client.post(f"{API}/datasets/{did}/enrichment/gene-lists",
                                 json=_lists(HUMAN, collections=DEFAULT_COLLECTIONS))
    a, b = implicit.json(), explicit.json()
    assert a["collections"] == DEFAULT_COLLECTIONS and a["warning"] == "" and a["species"] == "human"
    assert a["up"] == b["up"] and a["down"] == b["down"]
    listed = (await client.get(f"{API}/enrichment/msigdb-collections", params={"dataset_id": did})).json()
    plain = (await client.get(f"{API}/enrichment/msigdb-collections")).json()
    assert [c["id"] for c in listed["collections"]] == [c["id"] for c in plain["collections"]]
    assert listed["hierarchy"] == plain["hierarchy"] and listed["defaults"] == DEFAULT_COLLECTIONS
    assert "mouse_download" not in listed


async def test_mouse_listing_and_offline_503(client, dataset_manager, test_settings, monkeypatch):
    did = _register(dataset_manager, "mm", MOUSE)
    listed = (await client.get(f"{API}/enrichment/msigdb-collections", params={"dataset_id": did})).json()
    assert listed["species"] == "mouse" and listed["defaults"] == MOUSE_DEFAULT_COLLECTIONS
    assert listed["hierarchy"][0]["category"] == "MH"
    assert listed["mouse_download"]["state"] == "off"
    genesets = (await client.get(f"{API}/datasets/{did}/genesets/collections")).json()
    assert genesets["species"] == "mouse" and genesets["collections"][0]["id"] == "mh.all"
    # Not on disk, fetching off: a 503 that says how to get them.
    r = await client.post(f"{API}/datasets/{did}/enrichment/gene-lists", json=_lists(MOUSE))
    assert r.status_code == 503 and "MSIGDB_AUTO_DOWNLOAD is off" in r.json()["detail"]
    # Fetching on but offline: a 503 naming the failure.

    def fail(dest, ids, client=None, *, errors=None, **kw):
        errors.update({cid: "no network" for cid in ids})
        return [], list(ids)

    monkeypatch.setattr(test_settings, "MSIGDB_AUTO_DOWNLOAD", True)
    monkeypatch.setattr(msigdb_fetch, "download_collections", fail)
    monkeypatch.setattr(msigdb_fetch, "start_mouse_download", lambda dest: None)
    r = await client.post(f"{API}/datasets/{did}/enrichment/gene-lists", json=_lists(MOUSE))
    assert r.status_code == 503 and "no network" in r.json()["detail"]


async def test_gsea_on_mouse_uses_mh(client, dataset_manager, msigdb_dir):
    _mouse_files(msigdb_dir)
    did = _register(dataset_manager, "mm", MOUSE)
    genes = [{"gene": g, "score": float(80 - i)} for i, g in enumerate(MOUSE)]
    r = await client.post(f"{API}/datasets/{did}/enrichment/gsea",
                          json={"genes": genes, "collections": ["mh.all", "h.all"], "min_size": 5,
                                "permutations": 100})
    assert r.status_code == 200, r.text
    body = r.json()
    terms = {t["term"]: t for t in body["up"]["results"] + body["down"]["results"]}
    assert terms["HALLMARK_T_PROGRAM"]["species"] == "mouse"
    assert terms["HALLMARK_T_PROGRAM"]["set_size"] == 10
    assert terms["HALLMARK_T_PROGRAM (h.all)"]["set_size"] == 10  # case-matched, not lost
    assert "H is human gene symbols" in body["warning"]


# ---------------------------------------------------------------------------
# Pipeline step, replay script, co-pilot
# ---------------------------------------------------------------------------


def test_pipeline_step_swaps_untouched_defaults_for_mouse(msigdb_dir: Path, tmp_data_dir, monkeypatch):
    from scview.core.pipeline import PipelineParams, _run_enrichment

    _mouse_files(msigdb_dir)
    monkeypatch.setenv("MSIGDB_DIR", str(msigdb_dir))
    monkeypatch.setattr("scview.config.get_settings",
                        lambda: Settings(DATA_DIR=str(tmp_data_dir), MSIGDB_AUTO_DOWNLOAD=False))
    a = _adata(MOUSE)
    ret = _run_enrichment(a, PipelineParams(enrichment_columns=["cell_type"], enrichment_n_genes=10))
    assert ret["enrichment_collections"] == MOUSE_DEFAULT_COLLECTIONS
    assert ret["enrichment_gene_set_species"].startswith("mouse MSigDB MH")
    results = json.loads(a.uns[enrichment_key("cell_type", "T")])
    top = next(r for r in results if r["term"] == "HALLMARK_T_PROGRAM")
    assert top["collection"] == "mh.all" and top["species"] == "mouse"
    fact = ms.enrichment_fact(a)
    assert fact.startswith("Gene sets used: mouse MSigDB MH") and "another species" not in fact


def test_copilot_fact_names_human_sets_on_mouse():
    a = ad.AnnData(np.zeros((2, len(MOUSE)), dtype="float32"))
    a.var_names = MOUSE
    a.uns["ora_meta__cell_type__T"] = json.dumps({"collections": ["h.all"]})
    fact = ms.enrichment_fact(a)
    assert "human MSigDB H (human symbols, case-matched)" in fact and "another species" in fact


def test_replay_script_names_the_gene_set_species():
    from scview.core.replay_script import _Ctx, _enrichment

    snip = _enrichment({"enrichment_columns": ["leiden"], "enrichment_n_genes": 100,
                        "enrichment_collections": ["mh.all", "m5.go.bp"],
                        "enrichment_gene_set_species": "mouse MSigDB MH, M5 GO Biological Process"},
                       _Ctx(adata=None))
    assert "# Gene sets: mouse MSigDB MH, M5 GO Biological Process." in snip.code
    old = _enrichment({"enrichment_columns": ["leiden"], "enrichment_collections": ["h.all"]}, _Ctx(adata=None))
    assert "# Gene sets: human MSigDB H." in old.code


def test_reports_use_mouse_sets_once_on_disk(msigdb_dir: Path, monkeypatch):
    from scview.core.report_cluster import _hallmark
    from scview.core.report_gene import _gene_set_collections

    monkeypatch.setattr("scview.config.get_settings", lambda: Settings(MSIGDB_AUTO_DOWNLOAD=False))
    loader = MSigDBLoader(str(msigdb_dir))
    assert _hallmark(loader, "mouse") == "h.all"  # not fetched yet: the human sets stand in
    assert _gene_set_collections(loader, "mouse") == ["h.all", "c8.all"]
    _mouse_files(msigdb_dir)
    assert _hallmark(loader, "mouse") == "mh.all" and _hallmark(loader, "human") == "h.all"
    assert _gene_set_collections(loader, "mouse") == MOUSE_DEFAULT_COLLECTIONS
    assert _gene_set_collections(loader, "human") == ["h.all", "c8.all"]
