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
