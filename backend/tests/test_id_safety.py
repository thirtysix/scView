"""Request-supplied ids must never become paths outside their own folder.

`..` reaches the handlers as a path parameter when sent as %2E%2E, and used to
delete the whole data directory.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scview.core.dataset_manager import DatasetManager, is_safe_id
from scview.core.ingestion import IngestSessionManager


def _populate(data_dir: Path) -> Path:
    keep = data_dir / "uploads" / "abc123" / "metadata.json"
    keep.parent.mkdir(parents=True)
    keep.write_text("{}")
    return keep


@pytest.mark.parametrize("bad", ["..", ".", "", "a/b", "../x", "a\\b", "x" * 65])
def test_unsafe_ids_are_rejected(bad):
    assert not is_safe_id(bad)


def test_generated_ids_are_accepted():
    assert is_safe_id("578b70cb-7924-4ee3-90ec-d7fd018b61d8")
    assert is_safe_id("2cca9b93f2044065a3cdbc55d51c0d53")


def test_dataset_manager_ignores_dotdot(tmp_path):
    keep = _populate(tmp_path)
    dm = DatasetManager(data_dir=str(tmp_path))
    assert dm.remove_dataset("..") is False
    assert dm.get_dataset_info("..") is None
    assert dm.h5ad_candidates("..") == []
    assert keep.exists()


def test_session_manager_ignores_dotdot(tmp_path):
    keep = _populate(tmp_path)
    mgr = IngestSessionManager(data_dir=str(tmp_path))
    assert mgr.exists("..") is False
    assert mgr.discard("..") is False
    assert keep.exists()


async def test_delete_dotdot_dataset_is_404_and_deletes_nothing(client, dataset_manager):
    keep = _populate(Path(dataset_manager.data_dir))
    r = await client.delete("/api/v1/datasets/%2E%2E")
    assert r.status_code == 404
    assert keep.exists()
