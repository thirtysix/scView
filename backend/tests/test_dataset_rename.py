"""Renaming a dataset changes its display name only, and rejects unusable names."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scview.core.dataset_manager import DATASET_NAME_MAX, clean_dataset_name

API = "/api/v1"


def _register(dm, did: str = "ds1", name: str = "pbmc_3k") -> Path:
    d = Path(dm.data_dir) / "uploads" / did
    d.mkdir(parents=True)
    meta = d / "metadata.json"
    meta.write_text(json.dumps({"id": did, "name": name, "status": "converting",
                                "created_at": "2026-10-01T09:00:00+00:00"}))
    return meta


async def test_rename_persists_and_shows_in_the_list(client, dataset_manager):
    meta = _register(dataset_manager)
    r = await client.patch(f"{API}/datasets/ds1", json={"name": "  PBMC 3k (healthy donor)  "})
    assert r.status_code == 200, r.text
    assert r.json() == {"id": "ds1", "name": "PBMC 3k (healthy donor)"}
    stored = json.loads(meta.read_text())
    assert stored["name"] == "PBMC 3k (healthy donor)"
    assert stored["created_at"] == "2026-10-01T09:00:00+00:00"  # other fields untouched
    listed = (await client.get(f"{API}/datasets")).json()
    assert [d["name"] for d in listed] == ["PBMC 3k (healthy donor)"]


async def test_rename_unknown_or_unsafe_id_is_404(client, dataset_manager):
    keep = _register(dataset_manager)
    assert (await client.patch(f"{API}/datasets/nope", json={"name": "x"})).status_code == 404
    r = await client.patch(f"{API}/datasets/%2E%2E", json={"name": "x"})
    assert r.status_code == 404
    assert json.loads(keep.read_text())["name"] == "pbmc_3k"


@pytest.mark.parametrize(
    "bad",
    ["", "   ", "a\nb", "tab\there", "nul\x00", "bidi‮exe.h5ad", "zero​width",
     "x" * (DATASET_NAME_MAX + 1)],
)
async def test_rename_rejects_bad_names(client, dataset_manager, bad):
    meta = _register(dataset_manager)
    r = await client.patch(f"{API}/datasets/ds1", json={"name": bad})
    assert r.status_code == 400, r.text
    assert json.loads(meta.read_text())["name"] == "pbmc_3k"


def test_clean_name_keeps_unicode_text():
    assert clean_dataset_name("Hämatopoese – Tag 3 🧬") == "Hämatopoese – Tag 3 🧬"
    assert clean_dataset_name("x" * DATASET_NAME_MAX) == "x" * DATASET_NAME_MAX
    with pytest.raises(ValueError):
        clean_dataset_name(None)
