"""The MCP spatial tools (scview/mcp/spatial.py), plot on tissue sections, and get_dataset's
spatial fields: called through the SDK's client in process, as test_mcp.py does."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scview.mcp import jobs, plots, spatial as mcp_spatial, tools
from tests.test_hvg_sensitivity import _register
from tests.test_mcp import PNG, _call, _finish, _png_size, msigdb_dir, test_settings  # noqa: F401 — fixtures
from tests.test_spatial_domains import synthetic

SPATIAL_TOOLS = {"spatial_autocorr", "get_spatial_autocorr", "spatial_domains", "nhood_enrichment", "cooccurrence",
                 "get_cooccurrence",
                 "spatial_niches", "spot_composition_reference", "spot_composition", "joint_embedding_plan",
                 "joint_embedding"}


def _two_sections(n_side: int = 12, tech: str = "visium"):
    """Two sections at the same pixel positions (as two slides are), random expression."""
    a = synthetic.tissue(tech, n_side=n_side, sections=2)
    rng = np.random.default_rng(1)
    x = rng.poisson(1.0, size=(a.n_obs, 3)).astype(np.float32)
    x[:, 0] += (a.obs["stripe"].cat.codes.to_numpy() * 3).astype(np.float32)  # G0 follows the stripes
    a.X = np.log1p(x)
    return a


def _text(content) -> str:
    return next(b.text for b in content if b.type == "text")


def _png(content) -> bytes:
    data = base64.b64decode(next(b for b in content if b.type == "image").data)
    assert data.startswith(PNG) and len(data) < 300_000 and max(_png_size(data)) <= 1600
    return data


async def test_the_spatial_tools_are_listed():
    from mcp import Client

    from scview.mcp.server import SERVER

    async with Client(SERVER) as c:
        names = {t.name for t in (await c.list_tools()).tools}
    assert SPATIAL_TOOLS <= names and len(names) == len(tools.TOOLS)


# --- get_dataset and plot ----------------------------------------------------------------------

async def test_get_dataset_lists_spatial_its_sections_and_the_off_tissue_record(client, dataset_manager):
    did = _register(dataset_manager, _two_sections(), "gsp")
    meta = Path(dataset_manager.data_dir) / "uploads" / did / "metadata.json"
    rec = json.loads(meta.read_text())
    rec["off_tissue"] = {"removed": 713, "kept": 4279, "column": "in_tissue"}
    rec["public_source"] = {"about_spatial_note": "Discover lists a spatial assay (Visium), but no positions"}
    meta.write_text(json.dumps(rec))
    d = await _call("get_dataset", {"dataset_id": did})
    assert "spatial" in d["available_embeddings"] and "spatial" in d["modalities"]
    assert d["spatial"]["n_sections"] == 2 and d["spatial"]["sections"] == [["s0", 144], ["s1", 144]]
    assert d["spatial"]["unit"] == "spot" and d["spatial"]["library_column"] == "library_id"
    assert d["off_tissue"] == {"removed": 713, "kept": 4279, "column": "in_tissue"}
    assert "no positions" in d["spatial_note"]


async def test_plot_draws_sections_side_by_side_or_one_alone(client, dataset_manager):
    did = _register(dataset_manager, _two_sections(), "psec")
    both = await _call("plot", {"dataset_id": did, "embedding": "spatial", "color_by": "stripe"})
    _png(both)
    assert "288 spots; 2 of 2 sections (library_id)" in _text(both)
    one = await _call("plot", {"dataset_id": did, "embedding": "spatial", "gene": "G0", "section": "s1"})
    _png(one)
    assert "144 spots" in _text(one) and "section s1, one of 2" in _text(one)  # the other's spots left out
    # A subgroup within the sections drawn: greyed within them, the other section still left out.
    sub = await _call("plot", {"dataset_id": did, "embedding": "spatial", "color_by": "stripe", "section": "s0",
                               "within_column": "stripe", "within_value": "t0"})
    assert "144 spots" in _text(sub) and "drawn with stripe = t0" in _text(sub)
    by_col = await _call("plot", {"dataset_id": did, "embedding": "spatial", "section_column": "stripe",
                                  "section": ["t0", "t1"]})
    assert "2 of 4 sections (stripe)" in _text(by_col)
    err = await _call("plot", {"dataset_id": did, "embedding": "spatial", "section": "s9"}, error=True)
    assert "No section 's9' in library_id; its sections: s0, s1" in err
    err = await _call("plot", {"dataset_id": did, "embedding": "X_tsne"}, error=True)
    assert "spatial" in err


def test_the_sections_figure_caps_its_panels_and_says_so(monkeypatch):
    class Adaptor:
        def __init__(self, obs):
            self.adata = type("A", (), {"obs": obs})()

    libs = pd.Categorical([f"s{i}" for i in range(15) for _ in range(3)], categories=[f"s{i}" for i in range(15)])
    labels, order, note = tools._sections(Adaptor(pd.DataFrame({"library_id": libs})), None, None)
    assert order == [f"s{i}" for i in range(plots.MAX_PANELS)]
    assert note == f"; {plots.MAX_PANELS} of 15 sections (library_id); pass section for the others"
    xy = np.random.default_rng(0).uniform(0, 100, (45, 2))
    png = plots.sections(xy, pd.Series(np.arange(45) % 4).astype(str), labels, order, "t", "c", note)
    assert png.startswith(PNG)
    png = plots.sections(xy, np.arange(45, dtype=float), labels, order[:2], "t", "c", note,
                         focus=np.arange(45) % 2 == 0)
    assert png.startswith(PNG)
    one = pd.DataFrame({"library_id": pd.Categorical(["a"] * 4)})
    assert tools._sections(Adaptor(one), None, None) == (None, [], "")


# --- the runs ------------------------------------------------------------------------------------

async def test_morans_i_runs_and_reads_back(client, dataset_manager):
    did = _register(dataset_manager, _two_sections(), "mor")
    r = await _call("spatial_autocorr", {"dataset_id": did, "genes": ["G0", "G1", "G2"]})
    assert r["status"] == "complete", r
    res = r["result"]
    assert res["genes_ranked"]["rows"][0][0] == "G0" and res["genes_ranked"]["n_total"] == 3
    assert res["n_obs"] == 288 and res["result_id"]
    last = await _call("get_spatial_autocorr", {"dataset_id": did})
    assert last["result_id"] == res["result_id"]
    page = await _call("get_spatial_autocorr", {"dataset_id": did, "result_id": res["result_id"], "limit": 2})
    assert page["n_rows"] == 3 and len(page["rows"]) == 2 and page["next_offset"] == 2
    # Two sections: centred on each one's mean, each section's own I beside the pooled one.
    assert res["centring"] == "section" and res["sections"]["n_total"] == 2
    assert res["by_section"]["columns"] == ["gene", "s0", "s1"] and res["by_section"]["n_total"] == 3
    one = await _call("spatial_autocorr", {"dataset_id": did, "genes": ["G0"], "library_id": "s0"})
    assert one["result"]["n_obs"] == 144 and one["result"]["library_id"] == "s0" and "by_section" not in one["result"]
    assert (await _call("get_spatial_autocorr", {"dataset_id": did, "library_id": "all"}))["result_id"] == res["result_id"]
    assert (await _call("get_spatial_autocorr", {"dataset_id": did, "library_id": "s0"}))["n_obs"] == 144


async def test_a_run_past_the_wait_returns_its_job(client, dataset_manager, monkeypatch):
    did = _register(dataset_manager, _two_sections(), "mwait")
    monkeypatch.setattr(mcp_spatial, "WAIT_SECONDS", 0.0)
    r = await _call("spatial_autocorr", {"dataset_id": did, "genes": ["G0"]})
    assert r["status"] == "running" and r["next"].startswith("job_status")
    done = await _finish(r["job_id"])
    assert done["status"] == "complete" and done["result"]["genes_ranked"]["n_total"] == 1


async def test_domains_enrichment_and_cooccurrence_per_section(client, dataset_manager):
    did = _register(dataset_manager, _two_sections(14), "dom")
    r = await _call("spatial_domains", {"dataset_id": did, "k": 3})
    assert r["status"] == "complete", r
    res = r["result"]
    assert res["k"] == 3 and res["column"] == "spatial_domain"
    sec = res["per_section"]
    assert sec["columns"] == ["section", "domain 0", "domain 1", "domain 2"] and sec["n_total"] == 2
    assert sum(sum(row[1:]) for row in sec["rows"]) == 392
    # The column is taken now: refused with the route's sentence.
    r = await _call("spatial_domains", {"dataset_id": did, "k": 4})
    assert r["status"] == "error" and "Replace" in r["error"]
    r = await _call("spatial_domains", {"dataset_id": did, "suggest_k": True, "k_min": 2, "k_max": 4, "repeats": 2,
                                        "n_hops": 1, "column": "dom_suggested"})
    res = r["result"]
    assert res["suggested"] is True and res["stability_scan"]["ks"] == [2, 3, 4]

    e = (await _call("nhood_enrichment", {"dataset_id": did, "column": "stripe", "n_perms": 50}))["result"]
    assert e["labels"]["n_total"] == 4 and e["self"]["n_total"] == 4
    # Stripes: each stripe keeps to itself and avoids the others.
    assert e["most_apart"]["columns"] == ["a", "b", "z", "links"] and e["most_apart"]["n_total"] >= 3
    assert all(row[2] < 0 for row in e["most_apart"]["rows"]) and all(row[1] > 0 for row in e["self"]["rows"])

    c = (await _call("cooccurrence", {"dataset_id": did, "column": "stripe", "interval": 10}))["result"]
    # The score is symmetric here: each pair of the 4 stripes once.
    assert c["pairs"]["n_total"] == 6 and len(c["pairs"]["columns"]) == 2 + mcp_spatial.CO_DISTANCES + 1
    assert "peak_gb" in c and "resources" not in c and "created_at" not in c
    only = (await _call("cooccurrence", {"dataset_id": did, "column": "stripe", "interval": 10,
                                         "label": "t0"}))["result"]
    assert only["pairs"]["n_total"] == 3 and {row[0] for row in only["pairs"]["rows"]} == {"t0"}
    # Each section scored on its own, then combined; one section's own scores on request.
    assert c["sections_combined"] == "within_sections" and c["sections"]["n_total"] == 2
    s1 = (await _call("cooccurrence", {"dataset_id": did, "column": "stripe", "interval": 10, "section": "s1"}))["result"]
    assert s1["section"] == "s1"
    got = await _call("get_cooccurrence", {"dataset_id": did, "column": "stripe", "section": "s0", "label": "t1"})
    assert got["section"] == "s0" and {row[0] for row in got["pairs"]["rows"]} == {"t1"}
    err = await _call("get_cooccurrence", {"dataset_id": did, "section": "s9"}, error=True)
    assert "s9" in err


async def test_niches_are_refused_on_spots_and_run_on_cells(client, dataset_manager):
    did = _register(dataset_manager, _two_sections(10), "nsp")
    r = await _call("spatial_niches", {"dataset_id": did, "label_column": "stripe"})
    assert r["status"] == "error" and "spot composition" in r["error"].lower()
    did = _register(dataset_manager, _two_sections(10, "xenium"), "ncell")
    r = await _call("spatial_niches", {"dataset_id": did, "label_column": "stripe", "n_niches": 3})
    res = r["result"]
    assert res["n_niches"] == 3 and res["niches"]["n_total"] == 3 and res["per_section"]["n_total"] == 2
    assert "%" in res["niches"]["rows"][0][2]


async def test_spot_composition_reference_check_and_run(client, dataset_manager):
    from tests.test_spot_deconvolution_api import _pair

    ref, did = _pair(dataset_manager, "mcp")
    chk = await _call("spot_composition_reference", {"dataset_id": did, "reference_id": ref})
    assert chk["ok"] is True and chk["celltype_column"] == "cell_type" and chk["types"]["n_total"] == 4
    assert chk["estimate"]["peak_gb"] >= 0 and "cell_type" in chk["label_columns"]
    left = await _call("spot_composition_reference", {"dataset_id": did, "reference_id": ref,
                                                      "celltype_column": "cell_type", "exclude_labels": ["B"]})
    assert left["types"]["n_total"] == 3 and left["excluded_labels"] == ["B"]
    err = await _call("spot_composition_reference", {"dataset_id": did, "reference_id": ref,
                                                     "exclude_labels": ["B"]}, error=True)
    assert "celltype_column" in err
    r = await _call("spot_composition", {"dataset_id": did, "reference_id": ref, "celltype_column": "cell_type"})
    if r["status"] == "running":
        r = await _finish(r["job_id"])
    res = r["result"]
    assert res["n_fitted"] == 100 and res["types"]["n_total"] == 4
    assert res["per_section_shares"]["columns"][:3] == ["section", "n_spots", "n_fitted"]
    assert res["reference"]["id"] == ref
    # Niches on the spots from their cell-type shares.
    r = await _call("spatial_niches", {"dataset_id": did, "source": "spot_composition", "n_niches": 3})
    res = r["result"]
    assert res["source"] == "spot_composition" and res["shares"] == "cells" and res["niches"]["n_total"] == 3
    assert res["reference"]["id"] == ref and "shares_fingerprint" not in res


# --- joint embedding -------------------------------------------------------------------------------

def _slide(seed: int):
    from tests.test_integration import _counts_dataset

    a = _counts_dataset(seed=seed)
    n = a.n_obs
    side = int(np.ceil(np.sqrt(n)))
    a.obsm["spatial"] = np.column_stack([np.arange(n) % side, np.arange(n) // side]).astype(float) * 100
    a.obs["library_id"] = pd.Categorical(["slide"] * n)
    a.uns["spatial"] = {"slide": {"images": {}, "scalefactors": {"spot_diameter_fullres": 20.0},
                                  "metadata": {"source": "spaceranger"}}}
    return a


async def test_joint_embedding_plan_run_and_its_sections(client, dataset_manager, tmp_data_dir):
    from tests.test_integration import _add_dataset

    _add_dataset(tmp_data_dir, "sa", "A", _slide(1))
    _add_dataset(tmp_data_dir, "sb", "B", _slide(2))
    plan = await _call("joint_embedding_plan", {"dataset_ids": ["sa", "sb"], "labels": ["ctrl", "mi"]})
    assert plan["ok"] is True and plan["est_cells"] == 360 and plan["datasets"]["n_total"] == 2
    assert plan["preflight"]["sentence"] and "breakdown" not in plan["preflight"]
    err = await _call("joint_embedding", {"dataset_ids": ["sa", "sb"], "params": {"nope": 1}}, error=True)
    assert "nope" in err
    r = await _call("joint_embedding", {"dataset_ids": ["sa", "sb"], "labels": ["ctrl", "mi"], "name": "slides",
                                        "params": {"n_top_genes": 100, "n_comps": 20}})
    assert r["status"] == "running" and r["est_cells"] == 360
    done = await _finish(r["job_id"])
    assert done["status"] == "complete", done
    new = done["result"]["dataset_id"]
    assert done["dataset_id"] == new and done["result"]["name"] == "slides"
    d = await _call("get_dataset", {"dataset_id": new})
    assert [s[0] for s in d["spatial"]["sections"]] == ["ctrl", "mi"]
    content = await _call("plot", {"dataset_id": new, "embedding": "spatial", "color_by": "cell_type"})
    assert "2 of 2 sections (library_id)" in _text(content)


async def test_a_joint_run_is_refused_before_it_starts_when_the_plan_is(client, dataset_manager, tmp_data_dir):
    from tests.test_integration import _add_dataset

    _add_dataset(tmp_data_dir, "only", "A", _slide(1))
    err = await _call("joint_embedding_plan", {"dataset_ids": ["only", "ghost"]}, error=False)
    assert err["ok"] is False
    text = await _call("joint_embedding", {"dataset_ids": ["only", "ghost"]}, error=True)
    assert "Not started: ghost" in text
    assert not jobs.running("joint_embedding")


# --- the compact form ------------------------------------------------------------------------------

def test_fields_a_newer_route_adds_come_through_compact():
    rows = [{"gene": f"G{i}", "I": 0.5 - i / 100, "z": 3.0, "pval": 1e-3, "pval_adj": 1e-2,
             "by_library": {f"s{j}": {"I": 0.1 * j, "z": 1.0} for j in range(15)}} for i in range(30)]
    r = {"result_id": "x", "n_obs": 10, "rows": rows,
         "sections": [{"library_id": f"s{j}", "n_obs": 5} for j in range(15)],
         "occ_by_section": [[[0.0] * 3] * 60], "centring": "section"}
    out = mcp_spatial._autocorr_reply(r, "d", 5)
    assert out["centring"] == "section" and out["sections"]["n_total"] == 15 and out["sections"]["n_shown"] == 5
    assert out["genes_ranked"]["n_total"] == 30 and len(out["genes_ranked"]["rows"]) == 5
    wide = out["by_section"]
    assert wide["columns"] == ["gene", *[f"s{j}" for j in range(mcp_spatial.MAX_SECTION_COLUMNS)]]
    assert wide["sections_shown"] == f"{mcp_spatial.MAX_SECTION_COLUMNS} of 15"
    assert out["occ_by_section"] == [[[0.0] * 3] * 60]  # one row: small enough to send
    big = mcp_spatial._compact({"m": [[1, 2]] * (mcp_spatial.MAX_MATRIX_ROWS + 1)}, 5)
    assert big["m"]["shape"] == [mcp_spatial.MAX_MATRIX_ROWS + 1, 2]


@pytest.mark.parametrize("name", sorted(SPATIAL_TOOLS))
async def test_every_spatial_tool_argument_is_described(name):
    from mcp import Client

    from scview.mcp.server import SERVER

    async with Client(SERVER) as c:
        t = next(x for x in (await c.list_tools()).tools if x.name == name)
    assert t.input_schema["additionalProperties"] is False
    for arg, prop in t.input_schema["properties"].items():
        assert prop.get("description"), f"{name}.{arg}"
