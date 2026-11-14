"""The MCP tools for trajectories and RNA velocity (scview/mcp/trajectory.py), the plot overlays
(velocity arrows, PAGA graph), the phase portrait and a numeric column's violin: called through
the SDK's client in process, as test_mcp.py does."""

from __future__ import annotations

import base64

import numpy as np

from scview.core.trajectory import DPT_COLUMN
from scview.mcp import tools
from tests.test_hvg_sensitivity import _register
from tests.test_mcp import PNG, _call, _finish, msigdb_dir, test_settings  # noqa: F401 — fixtures
from tests.test_trajectory_dpt import N_ISLAND, N_PATH, _path_adata
from tests.test_velocity import API, dataset_with_store

NEW_TOOLS = {"paga", "pseudotime", "pseudotime_genes", "velocity", "get_velocity", "velocity_direction"}


async def _done(r: dict) -> dict:
    if r.get("status") == "running":
        r = await _finish(r["job_id"])
    assert r["status"] == "complete", r
    return r["result"]


def _image(content) -> tuple[bytes, str]:
    image = next(b for b in content if b.type == "image")
    data = base64.b64decode(image.data)
    assert image.mime_type == "image/png" and data.startswith(PNG) and len(data) < 300_000
    return data, next(b.text for b in content if b.type == "text")


async def test_the_new_tools_are_listed():
    from mcp import Client

    from scview.mcp.server import SERVER

    async with Client(SERVER) as c:
        listed = (await c.list_tools()).tools
    names = {t.name for t in listed}
    assert NEW_TOOLS <= names and len(names) == len(tools.TOOLS)
    plot = next(t for t in listed if t.name == "plot").input_schema["properties"]
    assert {"overlay", "value_column"} <= set(plot) and "phase" in str(plot["kind"])


def _path_with_genes():
    """The DPT test's path (early -> mid -> late, an unreachable island) with genes along it."""
    a, t = _path_adata()
    rng = np.random.default_rng(3)
    n = a.n_obs
    tt = np.r_[t, np.full(N_ISLAND, 0.5)]
    cols = {"Rise": 2.5 * tt, "Fall": 2.5 * (1 - tt), **{f"Flat{i}": np.full(n, 0.5) for i in range(20)}}
    X = np.clip(np.column_stack(list(cols.values())) + rng.normal(0, 0.2, (n, len(cols))), 0, None)
    out = type(a)(X.astype("float32"), obs=a.obs.copy(), obsm={"X_pca": a.obsm["X_pca"]})
    out.var_names = list(cols)
    out.obsm["X_umap"] = a.obsm["X_pca"][:, :2].astype("float32")
    return out


async def test_paga_pseudotime_from_the_right_and_a_wrong_root_and_genes(client, dataset_manager):
    did = _register(dataset_manager, _path_with_genes(), "mtraj")
    empty = await _call("paga", {"dataset_id": did})
    assert empty["paga"] is None and "groupby" in empty["note"]
    p = await _call("paga", {"dataset_id": did, "groupby": "stage"})
    assert p["groupby"] == "stage" and p["n_groups"] == 4
    assert {"early", "late"} <= set(p["tree_ends"]) and "mid" not in p["tree_ends"]
    assert p["edges"]["columns"] == ["from", "to", "connectivity", "in_tree"] and "early - mid" in p["tree"]

    right = await _call("pseudotime", {"dataset_id": did, "root_column": "stage", "root_value": "early"})
    assert right["warnings"] == [] and right["root_named_column"] == "scview_dpt_early"
    assert right["n_unreachable"] == N_ISLAND and right["column"] == DPT_COLUMN
    order = [row[0] for row in right["by_group"]["rows"]]
    assert order[:3] == ["early", "mid", "late"] and order[3] == "island"  # no finite value: last
    island = dict(zip(right["by_group"]["columns"], right["by_group"]["rows"][3]))
    assert island["n_missing"] == N_ISLAND and island.get("median") is None
    # Unreachable cells are inf: the metadata route sends them as null (was a 500).
    desc = await _call("describe_column", {"dataset_id": did, "column": "scview_dpt_early"})
    assert desc["kind"] == "numeric" and desc["n_missing"] == N_ISLAND and desc["min"] == 0.0

    route = (await client.post(f"{API}/datasets/{did}/trajectory/dpt",
                               json={"root_column": "stage", "root_value": "mid"})).json()["warnings"]
    wrong = await _call("pseudotime", {"dataset_id": did, "root_column": "stage", "root_value": "mid"})
    assert wrong["warnings"] == route and any("'mid' sits inside the PAGA tree of stage" in w for w in route)

    listed = await _call("pseudotime", {"dataset_id": did})
    assert {"scview_dpt_early", "scview_dpt_mid"} <= set(listed["runs"]) and listed["latest"]["root_value"] == "mid"
    assert listed["runs"]["scview_dpt_mid"]["warnings"] == route

    text = await _call("pseudotime", {"dataset_id": did, "root_column": "stage"}, error=True)
    assert "both root_column and root_value" in text

    g = await _call("pseudotime_genes", {"dataset_id": did, "column": "scview_dpt_early", "n_bins": 10, "limit": 5})
    top = [row[0] for row in g["genes"]["rows"][:2]]
    assert set(top) == {"Rise", "Fall"} and g["n_cells"] == N_PATH
    assert len(g["bins"]["pseudotime"]) == 10 and set(g["trends"]) >= {"Rise", "Fall"}
    branch = await _call("pseudotime_genes", {"dataset_id": did, "column": "scview_dpt_early",
                                              "groups_column": "stage", "groups": ["early", "mid"]})
    assert branch["restrict"] and branch["n_cells"] < N_PATH
    trend = await _call("pseudotime_genes", {"dataset_id": did, "column": "scview_dpt_early",
                                             "genes": ["Rise", "Nope"]})
    rise = trend["mean_expression"]["Rise"]
    assert trend["n_bins"] == 20 and len(rise) == 20 and rise[-1] > rise[0] and trend["not_found"] == ["Nope"]

    # A pseudotime by group, and the PAGA graph over the plot.
    data, caption = _image(await _call("plot", {"dataset_id": did, "kind": "violin",
                                                "value_column": "scview_dpt_early", "groupby": "stage"}))
    assert f"{N_ISLAND:,} without a finite value left out" in caption
    _image(await _call("plot", {"dataset_id": did, "color_by": "stage", "overlay": ["paga"]}))
    _, caption = _image(await _call("plot", {"dataset_id": did, "color_by": "scview_dpt_early", "overlay": ["paga"]}))
    assert "PAGA of stage" in caption
    text = await _call("plot", {"dataset_id": did, "kind": "violin", "value_column": "stage",
                                "groupby": "stage"}, error=True)
    assert "not numeric" in text
    text = await _call("plot", {"dataset_id": did, "kind": "violin", "overlay": ["paga"], "gene": "Rise"}, error=True)
    assert "overlay is drawn on an embedding" in text


async def test_velocity_run_direction_recheck_result_and_plots(client, dataset_manager):
    did, _a = dataset_with_store(dataset_manager, "mvel")
    d = await _call("get_dataset", {"dataset_id": did})
    assert "velocity" in d["modalities"] and "attached" in d["velocity"]
    before = await _call("get_velocity", {"dataset_id": did})
    assert before["attached"] is True and before["run"] is None and before["estimate"]["peak_gb"] >= 0

    r = await _done(await _call("velocity", {"dataset_id": did, "n_top_genes": 200, "n_neighbors": 20,
                                             "order": ["early", "late"], "exclude_genes": ["G0"]}))
    assert r["direction"]["transitions"]["rows"][0][:3] == ["early", "late", "forward"]
    assert not any("backward" in w for w in r["warnings"])
    assert r["params"]["exclude_given"] == ["G0"] and r["excluded"]["n_matched"] == 1
    assert r["groups"]["columns"][0] == "group" and r["result_id"]

    route = (await client.post(f"{API}/datasets/{did}/velocity/direction",
                               json={"order": ["late", "early"]})).json()["warnings"]
    back = await _call("velocity_direction", {"dataset_id": did, "order": ["late", "early"]})
    assert back["warnings"] == route and any("point backward" in w for w in route)
    assert back["direction"]["n_backward"] == 1 and back["direction"]["transitions"]["rows"][0][2] == "backward"
    assert back["direction"]["gene_flags"]["checked"] in (True, False)

    s = await _call("get_velocity", {"dataset_id": did, "genes": True, "limit": 5})
    assert s["run"]["warnings"] == route and s["run"]["direction"]["n_backward"] == 1
    assert s["genes"]["columns"] == ["gene", "velocity_gene", "gamma", "r2"] and len(s["genes"]["rows"]) == 5
    gene = s["genes"]["rows"][0][0]

    _, caption = _image(await _call("plot", {"dataset_id": did, "color_by": "clusters", "overlay": ["velocity"]}))
    assert "velocity arrows on a" in caption
    _, caption = _image(await _call("plot", {"dataset_id": did, "kind": "phase", "gene": gene, "color_by": "clusters"}))
    assert "gamma" in caption and "300 cells" in caption
    text = await _call("plot", {"dataset_id": did, "kind": "phase", "gene": "G0"}, error=True)
    assert "was not among" in text  # left out of the fit
    text = await _call("velocity", {"dataset_id": did, "transitions": [["early"]]}, error=True)
    assert "[from, to]" in text


async def test_without_counts_the_tools_say_how_to_attach_them(client, dataset_manager):
    a = _path_with_genes()
    did = _register(dataset_manager, a, "mnovel")
    d = await _call("get_dataset", {"dataset_id": did})
    assert "velocity" not in d["modalities"] and "upload_instructions" in d["velocity"]
    s = await _call("get_velocity", {"dataset_id": did})
    assert s["attached"] is False and "/velocity/attach" in s["note"]
    text = await _call("upload_instructions", {})
    text = text[0].text if isinstance(text, list) else str(text)
    assert "/velocity/attach" in text and "spliced" in text
    r = await _call("velocity", {"dataset_id": did})
    assert r["status"] == "error" and "attach them first" in r["error"]
    text = await _call("plot", {"dataset_id": did, "overlay": ["velocity"]}, error=True)
    assert "Run velocity" in text
