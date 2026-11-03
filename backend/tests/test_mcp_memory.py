"""The MCP tools when memory runs short: DE refused with the route's sentence (batch 39 item A)."""

from __future__ import annotations

from scview.api.v1 import de as de_api
from scview.core import memory_budget
from tests.test_mcp import _call, _register, msigdb_dir, test_settings  # noqa: F401 — fixtures

GB = 1024**3


def _short_of_memory(monkeypatch) -> None:
    def read(*_a, **_k):
        return memory_budget.MemoryBudget(8 * GB, int(7.5 * GB), True, "container limit")

    monkeypatch.setattr(memory_budget, "read_budget", read)
    monkeypatch.setattr(memory_budget, "release_memory", lambda: None)
    monkeypatch.setattr("scview.core.dataset_manager.REFUSAL_WAIT_SECONDS", 1)
    monkeypatch.setattr(de_api, "_test_bytes", lambda *_a: 3 * GB)


async def test_de_that_cannot_fit_is_a_tool_error_with_the_routes_sentence(client, dataset_manager, monkeypatch):
    did = _register(dataset_manager)
    await dataset_manager.get_or_load_dataset(did)
    _short_of_memory(monkeypatch)
    text = await _call("differential_expression", {"dataset_id": did, "column": "stim", "group_a": "STIM",
                                                   "group_b": "CTRL"}, error=True)
    assert f"Differential expression on {did} needs about 3.0 GB at its peak" in text
    assert "0.5 GB free of its 8.0 GB limit" in text


async def _finish(job_id: str) -> dict:
    for _ in range(40):
        r = await _call("job_status", {"job_id": job_id, "wait_seconds": 5})
        if r["status"] != "running":
            return r
    raise AssertionError("job did not finish")


async def test_run_pipeline_refuses_a_conversion_without_the_opt_in(client, dataset_manager):
    from scview.mcp import jobs

    did = _register(dataset_manager)
    text = await _call("run_pipeline", {"dataset_id": did, "steps": ["pca"], "params": {"chunked": "on"}},
                       error=True)
    assert "Not run:" in text and "allow_large_mode=true" in text
    assert "differential expression" in text and ".raw is moved out" in text
    assert not jobs.running()
    r = await _call("preflight", {"dataset_id": did, "steps": ["pca"], "params": {"chunked": "on"}})
    assert r["mode"]["converts"] is True and r["mode"]["turns_off"]


async def test_run_pipeline_passes_the_opt_in_to_the_route(client, dataset_manager, monkeypatch):
    from scview.mcp import jobs as jobs_mod

    did = _register(dataset_manager)
    bodies = []
    real = jobs_mod.new

    def spy(kind, dataset_id, route, body, **kw):
        bodies.append(body)
        return real(kind, dataset_id, route, body, **kw)

    monkeypatch.setattr(jobs_mod, "new", spy)
    r = await _call("run_pipeline", {"dataset_id": did, "steps": ["neighbors", "clustering"]})
    done = await _finish(r["job_id"])
    assert done["status"] == "complete", done
    assert done["result"]["mode"]["light"] is True  # needs no matrix: never converts
    assert bodies[-1]["allow_large_mode"] is False
    r = await _call("run_pipeline", {"dataset_id": did, "steps": ["neighbors"], "allow_large_mode": True})
    await _finish(r["job_id"])
    assert bodies[-1]["allow_large_mode"] is True
