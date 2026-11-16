"""No run converts a dataset to large-dataset layout unless its caller said yes (v0.81, QA
round 7 bug B1): MCP ``annotate_celltypist`` on a 46,700-cell dataset converted it for good,
unasked, because ``_start_run`` and the run routes allowed the conversion by default and
only ``run_pipeline`` checked. Now the default is no everywhere: every MCP tool that starts
a run takes ``allow_large_mode`` (default false) and checks the preflight first, the run
routes refuse unless the body says ``allow_large_mode: true``, and the refusal says how."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from scview.api.v1 import annotation
from scview.api.v1.assessment import RerunRequest, RunPipelineRequest
from scview.core.run_watchdog import RunControl
from scview.mcp import jobs, tools
from tests.test_mcp import _call, _register, msigdb_dir, test_settings  # noqa: F401 - fixtures
from tests.test_mcp_annotation import CATALOG
from tests.test_mcp_memory import _finish
from tests.test_preflight import _short

SRC = Path(__file__).resolve().parents[1] / "src" / "scview"
GB = 1024**3
RUN_ROUTES = ("/assessment/run", "/rerun")


def _functions(tree: ast.AST):
    return [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]


def _calls(node: ast.AST, name: str) -> list[ast.Call]:
    return [c for c in ast.walk(node) if isinstance(c, ast.Call)
            and (getattr(c.func, "id", None) == name or getattr(c.func, "attr", None) == name)]


def test_start_run_has_no_default_for_the_conversion():
    p = inspect.signature(tools._start_run).parameters["allow_large_mode"]
    assert p.kind is inspect.Parameter.KEYWORD_ONLY and p.default is inspect.Parameter.empty


def test_every_mcp_tool_that_starts_a_run_asks_before_a_conversion():
    """Each caller of ``_start_run`` passes ``allow_large_mode`` from its own argument of
    that name, default False, and checks the preflight (``_refuse_conversion``) first."""
    callers = {}
    for f in sorted((SRC / "mcp").glob("*.py")):
        for fn in _functions(ast.parse(f.read_text())):
            if fn.name == "_start_run":
                continue
            calls = _calls(fn, "_start_run")
            if not calls:
                continue
            callers[fn.name] = f.name
            for c in calls:
                kw = {k.arg: k.value for k in c.keywords}
                assert getattr(kw.get("allow_large_mode"), "id", None) == "allow_large_mode", fn.name
            args = fn.args.args + fn.args.kwonlyargs
            defaults = dict(zip([a.arg for a in fn.args.args][::-1], fn.args.defaults[::-1]))
            defaults.update({a.arg: d for a, d in zip(fn.args.kwonlyargs, fn.args.kw_defaults)})
            assert "allow_large_mode" in [a.arg for a in args], fn.name
            assert getattr(defaults.get("allow_large_mode"), "value", None) is False, fn.name
            assert _calls(fn, "_refuse_conversion"), fn.name
    assert callers == {"run_pipeline": "tools.py", "annotate_celltypist": "tools.py"}


def test_only_start_run_sends_an_mcp_job_to_a_run_route():
    for f in sorted((SRC / "mcp").glob("*.py")):
        for fn in _functions(ast.parse(f.read_text())):
            for c in _calls(fn, "new"):
                text = ast.unparse(c)
                if any(r in text for r in RUN_ROUTES):
                    assert fn.name == "_start_run", (f.name, fn.name)


def test_the_run_routes_and_the_run_control_refuse_by_default():
    assert RunPipelineRequest(steps=["pca"]).allow_large_mode is False
    assert RerunRequest(edited_step="leiden").allow_large_mode is False
    assert RunControl().allow_large is False


def test_every_planned_run_control_in_the_routes_takes_the_callers_answer():
    """A ``RunControl`` with a plan (the runs that choose their mode) is built with
    ``allow_large`` from the request body, never left at a default."""
    found = []
    for f in sorted((SRC / "api").rglob("*.py")):
        for c in _calls(ast.parse(f.read_text()), "RunControl"):
            kw = {k.arg: ast.unparse(k.value) for k in c.keywords}
            if "plan" in kw:
                found.append(f.name)
                assert kw.get("allow_large") == "body.allow_large_mode", (f.name, ast.unparse(c))
    assert len(found) == 3  # /assessment/run, /assessment/run-stream, /rerun


@pytest.fixture()
def catalog(monkeypatch):
    monkeypatch.setattr(annotation, "_model_catalog", lambda: CATALOG)


async def test_celltypist_that_would_convert_is_refused_with_how_to_allow_it(
        client, dataset_manager, monkeypatch, catalog):
    did = _register(dataset_manager)
    bodies = []

    async def stream(path, body, on_event, disconnect=None):
        bodies.append(body)
        await on_event("complete", {"status": "complete", "steps_run": ["cell_type_annotation"]})

    monkeypatch.setattr(jobs.api, "stream", stream)
    _short(monkeypatch, lambda steps: 10 * GB)  # 6 GB limit: the run would convert
    text = await _call("annotate_celltypist", {"dataset_id": did}, error=True)
    assert "Not run:" in text and "converts the dataset for good" in text
    assert "allow_large_mode=true" in text and bodies == []
    r = await _call("annotate_celltypist", {"dataset_id": did, "allow_large_mode": True})
    assert r["job_id"] and bodies[-1]["allow_large_mode"] is True
    await _finish(r["job_id"])
    _short(monkeypatch, lambda steps: GB)  # fits: runs, still without the yes
    r = await _call("annotate_celltypist", {"dataset_id": did})
    assert r["job_id"] and bodies[-1]["allow_large_mode"] is False


async def test_the_run_route_refusal_says_how_to_allow_it(client, dataset_manager, monkeypatch):
    from tests.test_light_run import _register as register_otero

    register_otero(Path(dataset_manager.data_dir))
    body = {"steps": ["pca", "neighbors"], "params": {"chunked": "on"}}  # no allow_large_mode
    r = await client.post("/api/v1/datasets/otero/assessment/run", json=body)
    assert r.status_code == 409
    assert "run it again allowing large-dataset mode" in r.json()["detail"]
    assert "allow_large_mode true" in r.json()["detail"]
