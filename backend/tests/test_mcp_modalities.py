"""The MCP tools for CRISPR screens, cell hashing, CITE-seq protein, 10x Multiome ATAC and
getting files in (scview/mcp/modalities.py, files.py), and the v0.78 friction fixes: called
through the SDK's client in process, as test_mcp.py does."""

from __future__ import annotations

import base64
import json

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from scview.core import protein
from scview.mcp import api, files as mcp_files, tools
from tests.test_atac_api import _register_multiome
from tests.test_atac_fragments_api import _adata as _fragments_adata
from tests.test_atac_fragments_api import _attach, files, seeded  # noqa: F401 — fixtures
from tests.genome_fixture import install
from tests.test_atac_links import _gtf, _register_world, _sequences
from tests.test_atac_motifs_api import annotated, inputs  # noqa: F401 — fixtures
from tests.test_hashing import _cite_hashed, _two_panels
from tests.test_hvg_sensitivity import _register
from tests.test_mcp import PNG, _call, _finish, _png_size, msigdb_dir, test_settings  # noqa: F401 — fixtures
from tests.test_perturbation_api import _with_guides
from tests.test_wnn import _cite

NEW_TOOLS = {"guide_assignment", "mixscape", "get_mixscape", "hashing_demux", "hashing_samples", "atac_status",
             "atac_fragment_qc", "atac_lsi", "wnn", "atac_gene_activity", "atac_differential", "atac_links",
             "atac_motif_enrichment", "atac_chromvar", "upload_instructions", "import_server_path"}


async def _done(r: dict) -> dict:
    """A job tool's reply: the result when it ended within its wait, else the finished job's."""
    if r.get("status") == "running":
        r = await _finish(r["job_id"])
    assert r["status"] == "complete", r
    return r["result"]


async def test_the_new_tools_are_listed():
    from mcp import Client

    from scview.mcp.server import SERVER

    async with Client(SERVER) as c:
        names = {t.name for t in (await c.list_tools()).tools}
    assert NEW_TOOLS <= names and len(names) == len(tools.TOOLS)


# --- CRISPR screens ---------------------------------------------------------------------------------

async def test_guide_assignment_previews_saves_then_mixscape_with_per_guide_tables(client, dataset_manager):
    a, _labels, ko = _with_guides()
    did = _register(dataset_manager, a, "mps")
    prev = await _call("guide_assignment", {"dataset_id": did})
    assert prev["saved"] is False and prev["control"] == "NT" and prev["counts"]["multiple"] == 1
    assert prev["per_target"]["columns"][:2] == ["target", "n_cells"] and "preview" in prev["note"]
    text = await _call("guide_assignment", {"dataset_id": did, "min_ratio": 0.5}, error=True)
    assert "must be 0 or at least 1 (got 0.5)" in text  # the route's own refusal
    saved = await _call("guide_assignment", {"dataset_id": did, "save": True})
    assert saved["saved"] is True and saved["counts"]["multiple"] == 1 and "mixscape" in saved["next"]
    assert saved["n_targets"] == 2 and saved["per_target"]["rows"][0][2] is True  # NT, the control, first
    r = await _done(await _call("mixscape", {"dataset_id": did, "split_by": "rep"}))
    by = {row[0]: dict(zip(r["targets"]["columns"], row)) for row in r["targets"]["rows"]}
    assert by["A"]["n_ko"] == int(ko.sum()) and by["B"]["n_ko"] == 0
    assert r["params"]["de_genes"] == "all" and r["split_by"] == "rep"
    assert "guide" in r["per_guide"]["columns"] and r["per_guide"]["n_total"] >= 3
    assert r["per_replicate"]["n_total"] >= 2
    last = await _call("get_mixscape", {"dataset_id": did})
    assert last["targets"]["rows"] == r["targets"]["rows"] and last["per_guide"] == r["per_guide"]
    listed = await _call("list_results", {"dataset_id": did, "kind": "mixscape"})
    assert listed["n_total"] == 1


def test_the_assignment_reply_reads_the_preview_and_the_saved_call():
    from scview.mcp.modalities import _assignment_reply

    preview = {"control": "NT", "per_target": [], "targets_without_cells": [{"target": "eGFP", "guides": ["eGFPg1"]}]}
    saved = {"control": "NT", "per_target": [], "targets_without_cells": ["eGFP", "PDL1"]}
    assert _assignment_reply(preview, "d", 5, False)["targets_without_cells"] == ["eGFP"]
    assert _assignment_reply(saved, "d", 5, True)["targets_without_cells"] == ["eGFP", "PDL1"]


# --- cell hashing -------------------------------------------------------------------------------------

async def test_hashing_demux_with_the_suggested_tags_then_samples(client, dataset_manager):
    a, _ = protein.split_features(_cite_hashed())
    did = _register(dataset_manager, a, "mhd")
    d = await _call("get_dataset", {"dataset_id": did})
    assert "protein" in d["modalities"]
    r = await _call("hashing_demux", {"dataset_id": did})
    call = r["call"]
    assert call["method"] == "htodemux" and call["tags"] == ["HTO-1", "HTO-2", "HTO-3", "HTO-4"]
    assert sum(call["counts"].values()) == a.n_obs and call["per_call"]["n_total"] >= 4
    assert "hashing" in (await _call("get_dataset", {"dataset_id": did}))["modalities"]
    s = await _call("hashing_samples", {"dataset_id": did, "mapping": {"HTO-1": "P1", "HTO-2": "P2"},
                                        "column": "donor"})
    assert s["samples"]["column"] == "donor" and s["samples"]["counts"]["P1"] > 0


async def test_hashing_refusal_is_the_routes_words_and_split_by_calls_per_lane(client, dataset_manager):
    did = _register(dataset_manager, _two_panels(), "mhp")
    tags = ["T1", "T2", "T3", "T4"]
    text = await _call("hashing_demux", {"dataset_id": did, "tags": tags, "method": "hashedDrops"}, error=True)
    assert "split by 'lane'" in text and "have no counts at all in 50% of the cells" in text  # verbatim
    assert text.endswith("(Here: allow_warnings=true.)")
    r = await _call("hashing_demux", {"dataset_id": did, "tags": tags, "split_by": "lane"})
    assert r["call"]["split_by"] == "lane" and r["call"]["n_uncalled"] == 0
    plain = _register(dataset_manager, ad.AnnData(np.ones((60, 3), dtype=np.float32)), "mhn")
    text = await _call("hashing_demux", {"dataset_id": plain}, error=True)
    assert "no hashtag counts" in text


# --- CITE-seq protein ----------------------------------------------------------------------------------

def _cite_donors() -> ad.AnnData:
    a = _cite(n_per=40)
    n = a.n_obs
    a.obs["donor"] = pd.Categorical([f"D{i % 4}" for i in range(n)])
    a.obs["cond"] = pd.Categorical(np.where(a.obs["group"].isin(["g0", "g2"]), "stim", "ctrl"))
    return a


async def test_protein_de_and_per_donor_values(client, dataset_manager):
    did = _register(dataset_manager, _cite_donors(), "mpr")
    r = await _call("differential_expression", {"dataset_id": did, "column": "group", "group_a": "g0",
                                                "group_b": "g2", "layer": "protein"})
    assert r["layer"] == "protein" and r["n_up"] >= 2
    assert {row[0] for row in r["up"]} >= {"P0", "P1"}  # g0's high proteins
    assert not any("transcriptome" in w for w in r["warnings"])
    res = await _call("get_result", {"dataset_id": did, "result_id": r["result_id"]})
    assert "protein CLR" in res["label"]
    one = await _call("differential_expression", {"dataset_id": did, "column": "group", "group": "g1",
                                                  "layer": "protein_counts"})
    assert one["layer"] == "protein_counts" and {row[0] for row in one["up"]} >= {"P2", "P3"}
    text = await _call("differential_expression", {"dataset_id": did, "column": "group", "group_a": "g0",
                                                   "group_b": "g2", "layer": "protein", "method": "pseudobulk"},
                       error=True)
    assert "cell-level" in text
    # Per donor: a protein, the whole dataset as one group, by condition.
    p = await _call("per_donor_expression", {"dataset_id": did, "gene": "P0", "layer": "protein",
                                             "all_cells": True, "sample": "donor", "condition": "cond"})
    assert p["groups"] == ["All cells"] and p["layer"] == "protein" and "CLR" in p["value_label"]
    assert set(p["values_by_group"]["All cells"]) == {"ctrl", "stim"}
    text = await _call("per_donor_expression", {"dataset_id": did, "gene": "NOPE", "layer": "protein",
                                                "all_cells": True, "sample": "donor"}, error=True)
    assert "not found" in text
    # A dataset without proteins says so.
    rna = _register(dataset_manager, _cite_donors().copy(), "mpr2")
    adaptor = await dataset_manager.get_or_load_dataset(rna)
    for k in (protein.CLR_KEY, protein.COUNTS_KEY):
        adaptor.adata.obsm.pop(k, None)
    text = await _call("differential_expression", {"dataset_id": rna, "column": "group", "group": "g1",
                                                   "layer": "protein"}, error=True)
    assert "no protein" in text


async def test_wnn_on_protein(client, dataset_manager):
    did = _register(dataset_manager, _cite(n_per=40), "mwp")
    r = await _done(await _call("wnn", {"dataset_id": did, "modality": "protein", "resolution": 0.8}))
    assert r["dataset_id"] == did
    text = await _call("wnn", {"dataset_id": did, "modality": "protein", "lsi_first": 2}, error=True)
    assert "atac" in text


# --- ATAC -----------------------------------------------------------------------------------------------

async def test_the_multiome_path_lsi_wnn_activity_differential_and_results(client, dataset_manager, annotated, inputs):  # noqa: F811
    plain = _register(dataset_manager, ad.AnnData(np.ones((5, 3), dtype=np.float32)), "mar")
    s = await _call("atac_status", {"dataset_id": plain})
    assert s["available"] is False and "no ATAC peaks" in s["reason"]
    did, *_ = _register_multiome(dataset_manager)
    s = await _call("atac_status", {"dataset_id": did})
    assert s["available"] and s["plan"]["estimate_seconds"] > 0 and "Run ATAC analysis first" in s["wnn_reason"]
    lsi = await _done(await _call("atac_lsi", {"dataset_id": did, "n_lsi": 10, "dims_to": 8, "resolution": 0.5}))
    assert lsi["lsi"]["n_selected"] > 0 and lsi["activity"]["available"]
    assert "wnn" not in lsi and "info" not in lsi and "LSI" in lsi["computed"]  # the run's own sections
    assert "cells_fingerprint" not in json.dumps(lsi)
    w = await _done(await _call("wnn", {"dataset_id": did, "modality": "atac"}))
    assert w["wnn"]["n_clusters"] >= 1 and "lsi" not in w and "WNN" in w["computed"]
    assert w["dataset_id"] == did
    s = await _call("atac_status", {"dataset_id": did, "groupby": "group"})
    assert s["lsi"]["stale"] is False and s["wnn"] is not None
    act = await _call("atac_gene_activity", {"dataset_id": did, "gene": "GENEA", "groupby": "group"})
    assert act["by_group"]["columns"] == ["group", "n", "mean", "median", "pct_above_0"] and act["kind"]
    assert act["by_group"]["n_total"] == 2
    da = await _done(await _call("atac_differential", {"dataset_id": did, "column": "group", "limit": 2}))
    assert all(len(t["rows"]) <= 2 for t in da["top_peaks"].values())
    assert da["result_id"] and da["counts"]["n_total"] == 2 and set(da["top_peaks"]) == {"A", "B"}
    t = await _call("get_result", {"dataset_id": did, "result_id": da["result_id"], "sort_by": "pval_adj"})
    assert t["kind"] == "atac_differential" and {"peak", "group", "pval_adj"} <= set(t["columns"]) and t["rows"]
    assert da["fold_change"] == "signac"
    pair = await _done(await _call("atac_differential", {"dataset_id": did, "column": "group", "groups": ["A"],
                                                          "vs": "B", "test": "wilcoxon", "fold_change": "tfidf"}))
    assert pair["vs"] == "B" and set(pair["top_peaks"]) == {"A"} and pair["fold_change"] == "tfidf"
    text = await _call("atac_differential", {"dataset_id": did, "column": "group", "vs": "B"}, error=True)
    assert "exactly one group" in text
    # Motifs: the scan runs inside the job, then the enrichment; then chromVAR (scanned already).
    me = await _done(await _call("atac_motif_enrichment", {"dataset_id": did, "da_id": da["result_id"],
                                                            "p_max": 0.5, "min_pct": 0.0, "background": 200}))
    assert me["result_id"] and me["da_id"] == da["result_id"] and set(me["top_motifs"]) == {"A", "B"}
    t = await _call("get_result", {"dataset_id": did, "result_id": me["result_id"]})
    assert t["kind"] == "atac_motif_enrichment" and "motif" in t["columns"] and t["rows"]
    cv = await _done(await _call("atac_chromvar", {"dataset_id": did, "seed": 3, "column": "group"}))
    assert cv["result_id"] and cv["most_variable"]["n_total"] >= 1
    assert cv["by_group"]["columns"] == ["group", "top_motifs", "padj_of_first"] and cv["by_group"]["n_total"] == 2
    from scview.core import results_store

    results_store.flush()
    t = await _call("get_result", {"dataset_id": did, "result_id": cv["result_id"]})
    assert t["kind"] == "atac_chromvar" and "variability" in t["columns"] and t["rows"]


@pytest.fixture()
def links_seeded(dataset_manager, tmp_path, monkeypatch):
    from scview.core import gene_annotation, genome_seq

    gtf = tmp_path / "links.gtf"
    gtf.write_text(_gtf())
    gene_annotation.clear_cache()
    gene_annotation.seed_from(gtf, dataset_manager.data_dir, "GRCh38")
    install(monkeypatch, dataset_manager.data_dir, "GRCh38", _sequences())
    yield dataset_manager
    gene_annotation.clear_cache()
    genome_seq.clear_cache()


async def test_peak_gene_links_and_its_result_table(client, links_seeded):
    did, _a = _register_world(links_seeded)
    r = await _done(await _call("atac_links", {"dataset_id": did, "genes": ["GENEA", "NOTHING"]}))
    assert r["result_id"] and r["counts"]["n_links"] >= 0 and r["genes_not_found"] == ["NOTHING"]
    t = await _call("get_result", {"dataset_id": did, "result_id": r["result_id"]})
    assert t["kind"] == "atac_links" and {"gene", "peak", "padj"} <= set(t["columns"])
    text = await _call("atac_links", {"dataset_id": did, "cells_column": "group"}, error=True)
    assert "cells_value" in text


async def test_fragment_qc_and_the_coverage_plot(client, dataset_manager, files, seeded):  # noqa: F811
    did = _register(dataset_manager, _fragments_adata(), "mfq")
    await _attach(client, did, files)
    s = await _call("atac_status", {"dataset_id": did})
    assert s["available"] is False  # fragments without peaks: no ATAC matrix
    r = await _done(await _call("atac_fragment_qc", {"dataset_id": did}))
    assert r["n_lines"] == 3040 and r["medians"]["tss_enrichment"] > 0
    content = await _call("plot", {"dataset_id": did, "kind": "coverage", "gene": "AAA", "groupby": "cell_type"})
    text = next(b.text for b in content if b.type == "text")
    data = base64.b64decode(next(b for b in content if b.type == "image").data)
    assert data.startswith(PNG) and max(_png_size(data)) <= 1600 and "2 groups of cell_type" in text
    content = await _call("plot", {"dataset_id": did, "kind": "coverage", "region": "chr1:49000-51000",
                                   "groupby": "cell_type", "groups": ["T"]})
    assert "1 groups of cell_type" in next(b.text for b in content if b.type == "text")
    text = await _call("plot", {"dataset_id": did, "kind": "coverage"}, error=True)
    assert "gene or a region" in text
    text = await _call("plot", {"dataset_id": did, "kind": "coverage", "gene": "NOPE"}, error=True)
    assert "no gene" in text


# --- files ------------------------------------------------------------------------------------------------

async def test_upload_instructions_give_the_routes():
    text = await _call("upload_instructions", {})
    text = text[0].text if isinstance(text, list) else str(text)
    assert "/api/v1/ingest/session" in text and "paths=" in text and "/atac/fragments?name=" in text


@pytest.fixture()
def ingest(tmp_data_dir, client):
    from scview.core.ingestion.session import IngestSessionManager
    from scview.dependencies import get_ingest_session_manager
    from scview.main import app

    mgr = IngestSessionManager(str(tmp_data_dir))
    app.dependency_overrides[get_ingest_session_manager] = lambda: mgr
    yield mgr
    app.dependency_overrides.pop(get_ingest_session_manager, None)


async def test_import_server_path_is_off_unless_allowed_and_never_in_public_mode(client, tmp_path, monkeypatch):
    text = await _call("import_server_path", {"path": str(tmp_path / "x.h5ad")}, error=True)
    assert "MCP_IMPORT_DIRS" in text and "upload_instructions" in text
    s = api.settings()
    monkeypatch.setattr(s, "MCP_IMPORT_DIRS", str(tmp_path))
    monkeypatch.setattr(s, "DEPLOYMENT_MODE", "public")
    text = await _call("import_server_path", {"path": str(tmp_path / "x.h5ad")}, error=True)
    assert "public mode" in text


async def test_import_server_path_refuses_paths_outside_the_folders(client, tmp_path, monkeypatch, ingest):
    allowed, outside = tmp_path / "share", tmp_path / "private"
    (allowed / "run").mkdir(parents=True)
    outside.mkdir()
    (outside / "secret.h5ad").write_text("x")
    monkeypatch.setattr(api.settings(), "MCP_IMPORT_DIRS", f" {allowed} , {tmp_path / 'gone'}")
    cases = {
        str(allowed / ".." / "private" / "secret.h5ad"): "outside",
        str(outside / "secret.h5ad"): "outside",
        "share/run/x.h5ad": "not absolute",
        str(allowed / "run" / "missing.h5ad"): "does not exist",
        str(allowed): "inside the import folder",
        "": "absolute path",
    }
    (allowed / "link.h5ad").symlink_to(outside / "secret.h5ad")
    cases[str(allowed / "link.h5ad")] = "outside"
    (allowed / "run" / "ok.txt").write_text("x")
    (allowed / "run" / "escape.h5ad").symlink_to(outside / "secret.h5ad")
    cases[str(allowed / "run")] = "links outside"
    (allowed / "dirlink").symlink_to(outside, target_is_directory=True)
    cases[str(allowed / "dirlink")] = "outside"
    for path, word in cases.items():
        text = await _call("import_server_path", {"path": path}, error=True)
        assert word in text, (path, text)


def test_resolve_import_walks_a_folder_and_skips_what_scview_never_reads(tmp_path):
    root = tmp_path / "share"
    outs = root / "s1" / "outs"
    outs.mkdir(parents=True)
    (outs / "filtered_feature_bc_matrix.h5").write_bytes(b"h5")
    (outs / "possorted_genome_bam.bam").write_bytes(b"bam")
    real, found, skipped = mcp_files.resolve_import(str(root / "s1"), [root.resolve()])
    assert real == (root / "s1").resolve()
    assert [rel for rel, _ in found] == ["s1/outs/filtered_feature_bc_matrix.h5"]
    assert [s["path"] for s in skipped] == ["s1/outs/possorted_genome_bam.bam"]


async def test_import_server_path_copies_and_commits_a_file(client, dataset_manager, tmp_path, monkeypatch, ingest):
    share = tmp_path / "share"
    share.mkdir()
    a = ad.AnnData(X=np.random.default_rng(0).poisson(1.0, (50, 20)).astype(np.float32),
                   obs=pd.DataFrame(index=[f"c{i}" for i in range(50)]),
                   var=pd.DataFrame(index=[f"G{i}" for i in range(20)]))
    src = share / "tiny.h5ad"
    a.write_h5ad(src)
    before = src.read_bytes()
    monkeypatch.setattr(api.settings(), "MCP_IMPORT_DIRS", str(share))
    r = await _call("import_server_path", {"path": str(src), "name": "tiny from the server"})
    assert r["n_files"] == 1 and r["job_id"]
    done = await _finish(r["job_id"])
    assert done["status"] == "complete", done
    did = done["result"]["dataset_id"]
    d = await _call("get_dataset", {"dataset_id": did})
    assert d["n_cells"] == 50 and d["name"] == "tiny from the server"
    assert src.read_bytes() == before and src.exists()  # copied, never moved


# --- friction fixes -----------------------------------------------------------------------------------------

async def test_describe_column_reads_a_category_of_numbers_as_categories(client, dataset_manager):
    a = ad.AnnData(np.ones((340, 3), dtype=np.float32))
    a.obs["leiden"] = pd.Categorical([str(i % 17) for i in range(340)])
    a.obs["depth"] = np.arange(340, dtype=float)
    did = _register(dataset_manager, a, "mdc")
    r = await _call("describe_column", {"dataset_id": did, "column": "leiden"})
    assert r["kind"] == "categories" and r["n_values"] == 17
    assert (await _call("describe_column", {"dataset_id": did, "column": "depth"}))["kind"] == "numeric"
    text = await _call("describe_column", {"dataset_id": did, "column": "leiden", "groupby": ["depth"]},
                       error=True)
    assert "not numeric" in text


async def test_get_recipe_has_measured_time_and_memory(client, dataset_manager):
    from scview.core import provenance

    a = _cite(n_per=20)
    provenance.record_step(a, step="pca", tool="scanpy", params={"n_comps": 5},
                           resources={"elapsed_s": 1.5, "peak_gb": 2.25})
    provenance.record_step(a, step="atac_lsi", tool="scview", params={}, resources={"seconds": 3, "peak_gb": 1.0,
                                                                                    "limit_gb": 6})
    provenance.record_step(a, step="label_cells", tool="scview", params={})
    did = _register(dataset_manager, a, "mrc")
    r = await _call("get_recipe", {"dataset_id": did})
    by = {s["step"]: s for s in r["steps"]}
    assert by["pca"]["measured"] == {"seconds": 1.5, "peak_gb": 2.25}
    assert by["atac_lsi"]["measured"] == {"seconds": 3, "peak_gb": 1.0, "limit_gb": 6}
    assert "measured" not in by["label_cells"]


def test_replay_script_pages_are_large():
    import inspect

    sig = inspect.signature(tools.get_replay_script)
    assert sig.parameters["max_chars"].default == 40_000


async def test_prefs_json_dtype_in_the_metadata_route(client, dataset_manager):
    a = ad.AnnData(np.ones((20, 3), dtype=np.float32))
    a.obs["leiden"] = pd.Categorical([str(i % 3) for i in range(20)])
    did = _register(dataset_manager, a, "mdt")
    r = await client.get(f"/api/v1/datasets/{did}/metadata/leiden", params={"format": "json"})
    assert r.json()["dtype"] == "category" and json.dumps(r.json())


async def test_mixscape_reply_carries_the_protein_check(client, dataset_manager):
    from tests.test_perturbation_protein import _guides_with_protein

    a, _ko = _guides_with_protein()
    did = _register(dataset_manager, a, "mpp")
    await _call("guide_assignment", {"dataset_id": did, "save": True})
    r = await _done(await _call("mixscape", {"dataset_id": did, "replicate_column": "rep3", "lda": False}))
    pr = r["protein"]
    assert pr["proteins"] == ["PDL1", "CD86"] and pr["replicate_column"] == "rep3"
    assert len(pr["flags"]) == 1 and pr["flags"][0].startswith("B: PDL1 -0.")
    first = dict(zip(pr["rows"]["columns"], pr["rows"]["rows"][0]))
    assert first["target"] == "B" and first["flag"] is True and first["unit"] == "replicate"
    assert "protein effect with no RNA KO class" in r["note"]
