"""The MCP server at /mcp (scview/mcp/): tools over the /api/v1 routes, jobs, plots, guards.

Tools are called with the SDK's own client, in process (``Client(SERVER)``); they
reach the routes through the whole FastAPI app with the test's dependency
overrides. The guards (MCP_ENABLED, ACCESS_TOKEN, Origin / Host) are checked over
HTTP on the /mcp route itself, the session manager running as in the app's lifespan.
"""

from __future__ import annotations

import asyncio
import base64
import functools
import io
import json
from pathlib import Path

import anndata as ad
import httpx
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from mcp import Client

from scview.config import Settings
from scview.dependencies import get_settings_dep
from scview.main import app
from scview.mcp import compact, jobs, server as mcp_server, tools
from scview.mcp.server import SERVER
from tests.test_public_import import fake  # noqa: F401, F811 — the Discover stand-in (no network)

T_SET = [f"G{i}" for i in range(0, 10)]  # type A's markers
S_SET = [f"G{i}" for i in range(10, 20)]  # up in STIM
GENES = [f"G{i}" for i in range(40)]
PNG = b"\x89PNG\r\n\x1a\n"


@pytest.fixture()
def msigdb_dir(tmp_path: Path) -> Path:
    d = tmp_path / "msigdb"
    d.mkdir()
    (d / "h.all.v2026.1.Hs.json").write_text(json.dumps({
        "HALLMARK_A_PROGRAM": {"geneSymbols": T_SET},
        "HALLMARK_STIM_PROGRAM": {"geneSymbols": S_SET},
    }))
    return d


@pytest.fixture()
def test_settings(tmp_data_dir: Path, msigdb_dir: Path) -> Settings:
    return Settings(DATA_DIR=str(tmp_data_dir), DEEPINFRA_API_KEY="", RAG_DATABASE_URL="",
                    MSIGDB_DIR=str(msigdb_dir))


@functools.lru_cache(maxsize=1)
def _adata() -> ad.AnnData:
    """4 donors x CTRL/STIM x types A/B, 30 cells each: A has G0-G9 high, STIM has
    G10-G19 high in every cell; a PCA, neighbour graph and a UMAP stand-in."""
    import scanpy as sc

    rng = np.random.default_rng(0)
    rows, obs = [], []
    for d in range(4):
        for cond in ("CTRL", "STIM"):
            for typ in ("A", "B"):
                for _ in range(30):
                    rate = np.full(len(GENES), 2.0)
                    if typ == "A":
                        rate[0:10] = 12
                    else:
                        rate[20:30] = 12
                    if cond == "STIM":
                        rate[10:20] = 12
                    rows.append(rng.poisson(rate))
                    obs.append({"donor_id": f"D{d}", "stim": cond, "cluster": typ,
                                "day": 9 if cond == "CTRL" else 11, "is_primary_data": True})
    counts = np.asarray(rows, dtype=np.float32)
    obs = pd.DataFrame(obs, index=[f"c{i}" for i in range(len(rows))])
    for c in ("donor_id", "stim", "cluster"):
        obs[c] = obs[c].astype("category")
    a = ad.AnnData(sp.csr_matrix(np.log1p(counts / counts.sum(1, keepdims=True) * 1e4)), obs=obs)
    a.var_names = GENES
    a.layers["counts"] = sp.csr_matrix(counts)
    sc.pp.pca(a, n_comps=8)
    sc.pp.neighbors(a, n_neighbors=10, use_rep="X_pca")
    a.obsm["X_umap"] = a.obsm["X_pca"][:, :2].copy()
    return a


def _register(dm, did: str = "mcpds") -> str:
    d = Path(dm.data_dir) / "uploads" / did
    d.mkdir(parents=True, exist_ok=True)
    (d / "metadata.json").write_text(json.dumps({"id": did, "name": did, "status": "ready"}))
    _adata().write_h5ad(d / "data.h5ad")
    return did


async def _call(name: str, args: dict, error: bool = False):
    async with Client(SERVER) as c:
        r = await c.call_tool(name, args)
    assert r.is_error == error, r.content
    text = "".join(getattr(b, "text", "") for b in r.content)
    if error:
        return text
    try:
        return json.loads(text)
    except ValueError:
        return r.content


# --- listing and schemas --------------------------------------------------------------------

async def test_tools_listed_with_descriptions_strict_schemas_and_hints():
    async with Client(SERVER) as c:
        listed = (await c.list_tools()).tools
    names = {t.name for t in listed}
    assert len(listed) == len(tools.TOOLS) >= 25
    assert {"list_datasets", "run_pipeline", "differential_expression", "per_donor_expression",
            "composition_test", "enrichment", "plot", "job_status", "cancel_job", "set_labels"} <= names
    for t in listed:
        assert t.description and len(t.description) > 30, t.name
        for arg, prop in t.input_schema.get("properties", {}).items():
            assert prop.get("description"), f"{t.name}.{arg} has no description"
        assert t.input_schema["type"] == "object" and t.input_schema["additionalProperties"] is False, t.name
        assert t.annotations is not None and t.annotations.destructive_hint is False, t.name
    by = {t.name: t for t in listed}
    assert by["list_datasets"].annotations.read_only_hint is True
    for w in ("run_pipeline", "set_labels", "subcluster", "import_public_dataset", "annotate_celltypist"):
        assert by[w].annotations.read_only_hint is False, w
    # run_pipeline documents every pipeline parameter with its default, and refuses others.
    params = by["run_pipeline"].input_schema["properties"]["params"]["anyOf"][0]
    assert params["additionalProperties"] is False
    assert "Default: 2000" in params["properties"]["n_top_genes"]["description"]
    assert "batch_key" in params["properties"] and "explicit" not in params["properties"]


async def test_unknown_arguments_steps_and_pipeline_params_are_refused(client, dataset_manager):
    did = _register(dataset_manager)
    text = await _call("memory_status", {"bogus": 1}, error=True)
    assert "bogus" in text
    text = await _call("run_pipeline", {"dataset_id": did, "steps": ["pca"], "params": {"n_pcs": 30}},
                       error=True)
    assert "'n_pcs'" in text and "n_comps" in text
    text = await _call("run_pipeline", {"dataset_id": did, "steps": ["umap"]}, error=True)
    assert "embeddings" in text
    text = await _call("run_pipeline", {"dataset_id": did, "steps": ["pca"],
                                        "params": {"n_comps": "many"}}, error=True)
    assert "n_comps" in text
    assert not jobs.running()


# --- data tools --------------------------------------------------------------------------------

async def test_data_tools(client, dataset_manager):
    did = _register(dataset_manager)
    r = await _call("list_datasets", {})
    assert [d["id"] for d in r["datasets"]] == [did]
    r = await _call("get_dataset", {"dataset_id": did})
    assert r["n_cells"] == 480 and r["n_genes"] == 40 and "X_umap" in r["available_embeddings"]
    cols = {c["name"]: c for c in r["obs_columns"]}
    assert cols["stim"]["values"] == ["CTRL", "STIM"] and r["large_mode"] is False
    assert cols["is_primary_data"]["values"] == ["True"]  # only the value present
    r = await _call("describe_column", {"dataset_id": did, "column": "donor_id"})
    assert r["kind"] == "categories" and r["counts"][0][1] == 120
    r = await _call("describe_column", {"dataset_id": did, "column": "day"})
    assert r["kind"] == "categories"  # two values: a category, not a distribution
    r = await _call("crosstab", {"dataset_id": did, "row": "donor_id", "col": "stim"})
    assert r["dataset_id"] == did and "D0" in json.dumps(r)
    r = await _call("memory_status", {})
    assert "limit_gb" in r and "available_bytes" in r
    r = await _call("assess_dataset", {"dataset_id": did})
    assert "pca" in r["done"] and "species" in r
    r = await _call("qc_filter_preview", {"dataset_id": did, "mode": "fixed", "min_genes": 30})
    assert r["n_cells"] == 480 and r["n_removed"] > 0
    r = await _call("preflight", {"dataset_id": did, "steps": ["pca", "neighbors"]})
    assert r["verdict"]["level"] in ("fits", "tight", "wont_fit") and r["steps"]["rows"][0][0] == "pca"
    text = await _call("describe_column", {"dataset_id": did, "column": "nope"}, error=True)
    assert "Column 'nope' not found." in text
    text = await _call("get_dataset", {"dataset_id": "missing"}, error=True)
    assert "not found" in text.lower()


async def test_markers_labels_by_filter_and_subcluster(client, dataset_manager):
    did = _register(dataset_manager)
    r = await _call("get_markers", {"dataset_id": did, "column": "cluster", "n_genes": 5})
    assert r["columns"][0] == "gene" and set(r["markers"]) == {"A", "B"}
    assert r["markers"]["A"][0][0] in T_SET
    # A sample column from donor AND day, as for a donor sampled twice.
    flt = {"op": "and", "items": [{"kind": "category", "column": "donor_id", "values": ["D0"]},
                                  {"kind": "range", "column": "day", "min": 11, "max": 11}]}
    r = await _call("set_labels", {"dataset_id": did, "column": "sample", "label": "D0_d11",
                                   "filter": flt, "new_column": True})
    assert r["n_labelled"] == 60 and r["created"] is True
    r = await _call("describe_column", {"dataset_id": did, "column": "sample"})
    assert dict(r["counts"]) == {"Unassigned": 420, "D0_d11": 60}
    text = await _call("set_labels", {"dataset_id": did, "column": "sample", "label": "x", "filter": {
        "op": "and", "items": [{"kind": "category", "column": "donor_id", "values": ["nobody"]}]}}, error=True)
    assert "No cell matches" in text
    r = await _call("subcluster", {"dataset_id": did, "column": "cluster", "group": "A", "resolution": 0.3})
    assert r["column"] == "cluster_sub"
    r = await _call("get_recipe", {"dataset_id": did})
    assert [s["step"] for s in r["steps"]][:2] == ["label_cells", "subcluster"]
    script = await _call("get_replay_script", {"dataset_id": did, "max_chars": 500})
    text = script[0].text if isinstance(script, list) else str(script)
    assert text.startswith("# scView replay script, characters 0-500 of") and "next page: offset=500" in text


# --- statistics --------------------------------------------------------------------------------

async def test_de_per_donor_composition_and_enrichment(client, dataset_manager):
    did = _register(dataset_manager)
    de = await _call("differential_expression", {
        "dataset_id": did, "column": "stim", "group_a": "STIM", "group_b": "CTRL",
        "within_column": "cluster", "within_value": "A", "method": "pseudobulk",
        "replicate_key": "donor_id"})
    assert de["method"] == "pseudobulk" and de["n_samples_a"] == 4 and de["result_id"]
    assert {g[0] for g in de["up"]} >= set(S_SET[:5]) and de["n_up"] >= 8
    cell = await _call("differential_expression", {"dataset_id": did, "column": "cluster", "group": "A",
                                                   "limit": 5})
    assert cell["method"] == "wilcoxon" and cell["n_a"] == 240 and len(cell["up"]) == 5
    assert all(g[0] in T_SET for g in cell["up"])
    flt = {"op": "and", "items": [{"kind": "gene", "gene": "G25", "op": ">", "value": 2.0}]}
    r = await _call("differential_expression", {"dataset_id": did, "filter": flt, "method": "t-test"})
    assert r["method"] == "t-test"
    text = await _call("differential_expression", {"dataset_id": did, "column": "stim", "group_a": "STIM"},
                       error=True)
    assert "group_b" in text

    r = await _call("per_donor_expression", {"dataset_id": did, "gene": "G12", "groupby": "cluster",
                                             "condition": "stim", "sample": "donor_id", "groups": ["A"]})
    assert r["design"]["paired"] is True and set(r["paired_tests"]) == {"A"}
    assert len(r["values_by_group"]["A"]["STIM"]) == 4
    r = await _call("composition_test", {"dataset_id": did, "condition_key": "stim", "group_key": "cluster",
                                         "sample_key": "donor_id"})
    assert r["groups"]["n_total"] == 2 and "fdr" in r["groups"]["columns"]
    r = await _call("expression_by_group", {"dataset_id": did, "genes": ["G0", "G25", "NOPE"],
                                            "groupby": "cluster"})
    assert r["groups"] == ["A", "B"] and r["genes_missing"] == ["NOPE"]
    assert r["mean"][0][0] > r["mean"][1][0]
    r = await _call("score_gene_set", {"dataset_id": did, "genes": T_SET, "groupby": "cluster"})
    assert r["rows"][0]["group"] == "A" and "scores" not in r

    r = await _call("enrichment", {"dataset_id": did, "de_result_id": de["result_id"],
                                   "collections": ["h.all"]})
    assert r["up"]["terms"]["rows"][0][0] == "HALLMARK_STIM_PROGRAM"
    r = await _call("enrichment", {"dataset_id": did, "genes": T_SET[:6], "collections": ["h.all"]})
    assert r["up"]["terms"]["rows"][0][0] == "HALLMARK_A_PROGRAM"
    text = await _call("enrichment", {"dataset_id": did, "column": "cluster", "group": "A",
                                      "collections": ["h.all"]}, error=True)
    assert "get_markers" in text
    await _call("get_markers", {"dataset_id": did, "column": "cluster"})
    r = await _call("enrichment", {"dataset_id": did, "column": "cluster", "group": "A", "n_genes": 10,
                                   "collections": ["h.all"]})
    assert r["terms"]["rows"][0][0] == "HALLMARK_A_PROGRAM"
    r = await _call("enrichment", {"dataset_id": did, "de_result_id": cell["result_id"], "method": "gsea",
                                   "collections": ["h.all"], "min_size": 5})
    assert r["method"] == "gsea" and "up" in r
    text = await _call("enrichment", {"dataset_id": did}, error=True)
    assert "exactly one" in text

    res = await _call("list_results", {"dataset_id": did})
    kinds = {row["kind"] for row in res["rows"]}
    assert {"de", "composition"} <= kinds
    r = await _call("get_result", {"dataset_id": did, "result_id": de["result_id"], "limit": 3,
                                   "sort_by": "pval_adj"})
    assert r["kind"] == "de" and len(r["rows"]) == 3 and r["n_rows"] == de["n_genes_tested"]


async def test_compare_de_results(client, dataset_manager):
    did = _register(dataset_manager)
    scope = {"dataset_id": did, "column": "stim", "group_a": "STIM", "group_b": "CTRL",
             "within_column": "cluster", "within_value": "A", "replicate_key": "donor_id"}
    cell = await _call("differential_expression", scope)
    pb = await _call("differential_expression", {**scope, "method": "pseudobulk"})
    r = await _call("compare_de_results", {"dataset_id": did, "result_a": cell["result_id"],
                                           "result_b": pb["result_id"], "limit": 5})
    assert r["a"]["method"] == "wilcoxon" and r["b"]["method"] == "pseudobulk"
    assert r["a"]["n"] == cell["n_up"] + cell["n_down"] and r["b"]["n"] == pb["n_up"] + pb["n_down"]
    assert r["both"] + r["only_a"]["n"] == r["a"]["n"] and r["both"] + r["only_b"]["n"] == r["b"]["n"]
    assert r["both"] >= 8 and r["sign_agreement"] == 1.0
    assert r["only_a"]["columns"] == ["gene", "lfc_a", "lfc_b"] and len(r["only_a"]["rows"]) <= 5
    text = await _call("compare_de_results", {"dataset_id": did, "result_a": cell["result_id"],
                                              "result_b": "gone"}, error=True)
    assert "DE" in text or "result" in text


async def test_large_dataset_refusal_is_a_tool_error(client, dataset_manager, monkeypatch):
    from scview.core import large_mode

    did = _register(dataset_manager)

    def refuse(adaptor, what, why):
        raise large_mode.LargeDatasetError(what, why)

    monkeypatch.setattr(large_mode, "check", refuse)
    text = await _call("differential_expression", {"dataset_id": did, "column": "cluster", "group": "A"},
                       error=True)
    assert "Refused on a large dataset:" in text and "Differential expression" in text


# --- plots -------------------------------------------------------------------------------------

def _png_size(data: bytes) -> tuple[int, int]:
    from PIL import Image

    with Image.open(io.BytesIO(data)) as im:
        return im.size


@pytest.mark.parametrize("args", [
    {"color_by": "cluster"}, {"gene": "G3"}, {"kind": "violin", "gene": "G3", "groupby": "cluster"},
    {"kind": "dotplot", "genes": ["G0", "G12", "G25"], "groupby": "cluster"},
])
async def test_plot_is_a_png_within_the_caps(client, dataset_manager, args):
    did = _register(dataset_manager)
    content = await _call("plot", {"dataset_id": did, **args})
    image = next(b for b in content if b.type == "image")
    data = base64.b64decode(image.data)
    assert image.mime_type == "image/png" and data.startswith(PNG)
    assert len(data) < 300_000 and max(_png_size(data)) <= 1600


@pytest.mark.parametrize("args", [
    {"color_by": "donor_id"}, {"kind": "violin", "gene": "G12", "groupby": "donor_id"},
    {"kind": "dotplot", "genes": ["G0", "G12"], "groupby": "donor_id"},
])
async def test_plot_within_one_subgroup(client, dataset_manager, args):
    did = _register(dataset_manager)
    content = await _call("plot", {"dataset_id": did, "within_column": "cluster", "within_value": "A", **args})
    text = next(b.text for b in content if b.type == "text")
    assert "cluster = A" in text
    if args.get("kind") != "dotplot":
        assert "240" in text  # type A's cells
    data = base64.b64decode(next(b for b in content if b.type == "image").data)
    assert data.startswith(PNG)
    text = await _call("plot", {"dataset_id": did, "within_column": "cluster", "within_value": "Z",
                                "color_by": "stim"}, error=True)
    assert "No cell has cluster = 'Z'" in text
    text = await _call("plot", {"dataset_id": did, "within_column": "cluster"}, error=True)
    assert "both" in text


async def test_dotplot_within_is_the_subgroups_cells(client, dataset_manager):
    did = _register(dataset_manager)
    d = await tools._dotplot_within(did, ["G0", "G12"], "stim", "", "cluster", "B")
    assert d["groups"] == ["CTRL", "STIM"] and d["genes"] == ["G0", "G12"]
    a = _adata()
    b_ctrl = ((a.obs["cluster"] == "B") & (a.obs["stim"] == "CTRL")).to_numpy()
    x = a[b_ctrl, ["G0", "G12"]].X.toarray()
    assert np.allclose(d["fraction"][0], (x > 0).mean(axis=0), atol=1e-3)
    assert d["mean"][1][1] > d["mean"][0][1]  # G12 up in STIM


def test_labels_skip_tiny_groups_and_move_off_each_other():
    from scview.mcp import plots

    rng = np.random.default_rng(0)
    xy = np.vstack([rng.normal(0, 0.01, (500, 2)), rng.normal(0, 0.01, (500, 2)),
                    rng.normal(5, 0.01, (1, 2)), rng.normal(10, 1, (500, 2))])
    codes = np.repeat([0, 1, 2, 3], [500, 500, 1, 500])
    spots = plots._label_spots(xy, codes, ["IgA PB", "IgG PB", "pDC", "T"])
    names = [n for n, _, _ in spots]
    assert "pDC" not in names and {"IgA PB", "IgG PB", "T"} == set(names)
    (_, _, ya), (_, _, yb) = [s for s in spots if s[0] in ("IgA PB", "IgG PB")]
    assert abs(ya - yb) > 0.2  # the two labels on one spot were moved apart


async def test_plot_samples_a_big_dataset_and_says_so(client, dataset_manager, monkeypatch):
    from scview.mcp import plots

    did = _register(dataset_manager)
    monkeypatch.setattr(plots, "MAX_POINTS", 100)
    content = await _call("plot", {"dataset_id": did, "color_by": "stim"})
    assert "random 100 of 480 cells" in next(b.text for b in content if b.type == "text")
    text = await _call("plot", {"dataset_id": did, "embedding": "X_tsne"}, error=True)
    assert "X_umap" in text


# --- jobs ----------------------------------------------------------------------------------------

async def _finish(job_id: str) -> dict:
    for _ in range(40):
        r = await _call("job_status", {"job_id": job_id, "wait_seconds": 5})
        if r["status"] != "running":
            return r
    raise AssertionError("job did not finish")


async def test_pipeline_job_runs_to_complete(client, dataset_manager):
    did = _register(dataset_manager)
    r = await _call("run_pipeline", {"dataset_id": did, "steps": ["pca", "neighbors", "clustering"],
                                     "params": {"n_comps": 5, "clustering_resolution": 0.3}})
    assert r["status"] == "running" and r["job_id"]
    text = await _call("run_pipeline", {"dataset_id": did, "steps": ["pca"]}, error=True)
    assert "still running" in text
    done = await _finish(r["job_id"])
    assert done["status"] == "complete", done
    assert done["result"]["steps_run"] == ["pca", "neighbors", "clustering"]
    r = await _call("get_recipe", {"dataset_id": did})
    pca = next(s for s in r["steps"] if s["step"] == "pca")
    assert pca["params"]["n_comps"] == 5
    text = await _call("job_status", {"job_id": "nope"}, error=True)
    assert "No job" in text


async def test_cancel_leaves_a_stream_without_a_cancel_route(monkeypatch):
    left = asyncio.Event()

    async def stream(path, body, on_event, disconnect=None):
        await on_event("progress", {"message": "working"})
        await disconnect.wait()
        left.set()

    monkeypatch.setattr(jobs.api, "stream", stream)
    job = jobs.new("milo", "x", "/x", {}, progress_line=tools._line, finish=lambda j: {})
    await asyncio.sleep(0)
    r = await _call("job_status", {"job_id": job.id, "wait_seconds": 0})
    assert r["status"] == "running" and r["progress"] == "progress: working"
    r = await _call("cancel_job", {"job_id": job.id})
    assert r["status"] == "cancelled" and left.is_set()


async def test_cancel_uses_the_runs_cancel_route(client, dataset_manager, monkeypatch):
    did = _register(dataset_manager)
    calls = []

    async def stream(path, body, on_event, disconnect=None):
        await on_event("step_start", {"step": "pca", "index": 0, "total": 1})
        while not calls:
            await asyncio.sleep(0.01)
        await on_event("complete", {"status": "cancelled", "steps_run": []})

    async def post(path, body=None, **q):
        calls.append(path)
        return {"cancelling": True}

    monkeypatch.setattr(jobs.api, "stream", stream)
    monkeypatch.setattr(tools.api, "post", post)
    r = await _call("run_pipeline", {"dataset_id": did, "steps": ["pca"]})
    r = await _call("cancel_job", {"job_id": r["job_id"]})
    assert calls == [f"/datasets/{did}/assessment/cancel"] and r["status"] == "cancelled"


async def test_celltypist_runs_the_annotation_step_as_a_job(client, dataset_manager, monkeypatch):
    did = _register(dataset_manager)
    sent = {}

    async def stream(path, body, on_event, disconnect=None):
        sent.update(path=path, body=body)
        await on_event("complete", {"status": "complete", "steps_run": ["cell_type_annotation"]})

    monkeypatch.setattr(jobs.api, "stream", stream)
    r = await _call("annotate_celltypist", {"dataset_id": did, "model": "Immune_All_High.pkl"})
    done = await _finish(r["job_id"])
    assert done["status"] == "complete" and sent["body"]["steps"] == ["cell_type_annotation"]
    assert sent["body"]["params"]["annotation_method"] == "celltypist"
    assert sent["body"]["params"]["celltypist_model"] == "Immune_All_High.pkl"
    assert "celltypist" in done["result"]  # the route's summary (None: no such run on this data)


async def test_milo_job_on_the_graph(client, dataset_manager):
    did = _register(dataset_manager)
    r = await _call("milo", {"dataset_id": did, "condition_key": "stim", "sample_key": "donor_id",
                             "label_key": "cluster", "prop": 0.2})
    assert r["plan"]["design"] and r["job_id"]
    done = await _finish(r["job_id"])
    assert done["status"] == "complete", done
    res = done["result"]
    assert res["n_nhoods"] > 0 and res["top_neighbourhoods"]["columns"][0] == "label"
    assert "layouts" not in res and len(json.dumps(done)) < compact.MAX_CHARS


async def test_public_search_and_import_job(client, dataset_manager, fake):  # noqa: F811
    from tests.test_public_import import LIVER_B

    r = await _call("search_public_datasets", {"query": "liver", "limit": 5})
    assert r["datasets"][0]["dataset_version_id"] == LIVER_B and "size_mb" in r["datasets"][0]
    r = await _call("import_public_dataset", {"dataset_version_id": LIVER_B})
    done = await _finish(r["job_id"])
    assert done["status"] == "complete", done
    new = done["result"]["dataset_id"]
    assert done["dataset_id"] == new
    r = await _call("list_datasets", {})
    assert new in {d["id"] for d in r["datasets"]}


# --- compact results -----------------------------------------------------------------------------

def test_numbers_keep_four_significant_digits_and_tiny_p_values():
    assert compact.num(1.23456789e-300) == 1.235e-300
    assert compact.num(0.000123456) == 0.0001235
    assert compact.num(float("nan")) is None and compact.num(np.float32(2.5)) == 2.5
    assert compact.num(True) is True and compact.num(7) == 7


def test_long_results_are_cut_to_fit_and_say_so():
    data = {"a": 1, "rows": [{"gene": f"G{i}", "p": 1e-5 * i} for i in range(2000)]}
    text = compact.dumps(data, max_chars=3000)
    out = json.loads(text)
    assert len(text) <= 3200 and out["a"] == 1
    assert out["truncated"]["rows"].endswith("of 2000 shown")


# --- guards over HTTP -----------------------------------------------------------------------------

INIT = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
    "protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "0"}}}
HEADERS = {"accept": "application/json, text/event-stream", "content-type": "application/json"}


async def _post_init(settings: Settings, headers: dict | None = None, host: str = "127.0.0.1:8080"):
    app.dependency_overrides[get_settings_dep] = lambda: settings
    try:
        async with mcp_server.lifespan():
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=f"http://{host}") as c:
                return await c.post("/mcp", json=INIT, headers={**HEADERS, **(headers or {})})
    finally:
        app.dependency_overrides.pop(get_settings_dep, None)


def _rpc(r: httpx.Response) -> dict:
    if r.headers.get("content-type", "").startswith("application/json"):
        return r.json()
    data = [ln[5:].strip() for ln in r.text.splitlines() if ln.startswith("data:")]
    return json.loads(data[-1])


async def test_initialize_over_http_sends_the_instructions(tmp_data_dir):
    r = await _post_init(Settings(DATA_DIR=str(tmp_data_dir)))
    assert r.status_code == 200, r.text
    result = _rpc(r)["result"]
    assert result["serverInfo"]["name"] == "scview"
    text = result["instructions"]
    assert "list_datasets" in text and "pseudobulk" in text and "/api/v1/datasets/upload" in text
    assert 10 <= len(text.splitlines()) <= 20


async def test_disabled_in_public_mode_unless_turned_on(tmp_data_dir):
    r = await _post_init(Settings(DATA_DIR=str(tmp_data_dir), DEPLOYMENT_MODE="public"))
    assert r.status_code == 404
    r = await _post_init(Settings(DATA_DIR=str(tmp_data_dir), DEPLOYMENT_MODE="public", MCP_ENABLED=True))
    assert r.status_code == 200
    r = await _post_init(Settings(DATA_DIR=str(tmp_data_dir), MCP_ENABLED=False))
    assert r.status_code == 404


async def test_access_token_is_required_when_set(tmp_data_dir):
    s = Settings(DATA_DIR=str(tmp_data_dir), ACCESS_TOKEN="s3cret")
    assert (await _post_init(s)).status_code == 401
    assert (await _post_init(s, {"authorization": "Bearer wrong"})).status_code == 401
    assert (await _post_init(s, {"authorization": "Bearer s3cret"})).status_code == 200
    assert (await _post_init(s, {"x-access-token": "s3cret"})).status_code == 200


async def test_hostile_origin_and_host_are_refused(tmp_data_dir):
    s = Settings(DATA_DIR=str(tmp_data_dir))
    r = await _post_init(s, {"origin": "http://evil.example"})
    assert r.status_code == 403 and "MCP_ALLOWED_ORIGINS" in r.json()["detail"]
    assert (await _post_init(s, {"origin": "http://localhost:3000"})).status_code == 200
    assert (await _post_init(s, {"origin": "http://127.0.0.1:8080"})).status_code == 200
    assert (await _post_init(s, host="evil.example")).status_code == 421  # DNS rebinding
    assert (await _post_init(s, host="localhost")).status_code == 200
    lan = Settings(DATA_DIR=str(tmp_data_dir), MCP_ALLOWED_ORIGINS="http://192.168.1.5:8080")
    assert (await _post_init(lan, {"origin": "http://192.168.1.5:8080"}, host="192.168.1.5:8080")).status_code == 200
    assert (await _post_init(lan, {"origin": "http://192.168.1.6:8080"}, host="192.168.1.5:8080")).status_code == 403
