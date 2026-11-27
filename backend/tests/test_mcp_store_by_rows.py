"""MCP ``store_by_rows``, and the run refusal that names it (``mcp/files.py``, ``mcp/tools.py``)."""

from __future__ import annotations

from pathlib import Path

import pytest

from scview.core import csr_store
from scview.mcp import jobs
from tests.test_csr_store import _library_dataset
from tests.test_mcp import _call
from tests.test_store_by_rows import _counts_dataset

pytestmark = pytest.mark.asyncio


async def _finished(job_id: str) -> dict:
    job = jobs.get(job_id)
    await jobs.wait(job, 30)
    return job.summary()


async def test_store_by_rows_says_the_cost_then_rewrites(client, dataset_manager):
    data = Path(dataset_manager.data_dir)
    upload = _library_dataset(data, "old", _counts_dataset("csc"))
    dry = await _call("store_by_rows", {"dataset_id": "old", "dry_run": True})
    assert dry["column_stored"] == dry["blocks_large_mode"] == ["X", "raw/X"] and dry["files"] == 1
    assert dry["enough_disk"] is True and "GB of disk" in dry["text"]
    assert csr_store.csc_in_file(upload) == ["X", "raw/X"] and not (data / "ingested" / "old").exists()

    r = await _call("store_by_rows", {"dataset_id": "old"})
    r = r if r["status"] != "running" else await _finished(r["job_id"])
    assert r["status"] == "complete" and r["result"]["converted"] == ["X", "raw/X"]
    assert csr_store.csc_in_file(dataset_manager._resolve_h5ad("old")) == []
    again = await _call("store_by_rows", {"dataset_id": "old"})
    assert again["files"] == 0 and "nothing to do" in again["note"]


async def test_a_run_that_would_go_to_large_mode_names_the_tool(client, dataset_manager):
    _library_dataset(Path(dataset_manager.data_dir), "old", _counts_dataset("csc"))
    steps = ["qc_metrics", "normalization", "log_transform"]
    text = await _call("run_pipeline", {"dataset_id": "old", "steps": steps, "params": {"chunked": "on"}}, error=True)
    assert "Not run:" in text and "stored by columns (X, raw/X)" in text
    assert "store_by_rows(dataset_id) first (about 5 s and 0." in text and "GB of disk; no value changes)" in text
    assert "allow_large_mode" not in text  # allowing the conversion would not help
    # With the yes to the conversion the run itself is refused, in the same words.
    r = await _call("run_pipeline", {"dataset_id": "old", "steps": steps, "params": {"chunked": "on"},
                                     "allow_large_mode": True})
    done = await _finished(r["job_id"])
    assert done["status"] == "error" and "store the dataset by rows first" in done["error"]
    assert "store_by_rows" in done["error"]
