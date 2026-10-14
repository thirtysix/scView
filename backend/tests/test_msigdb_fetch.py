"""Tests for the first-start MSigDB downloader."""

from __future__ import annotations

import json
from pathlib import Path

import httpx

from scview.core import msigdb_fetch
from scview.core.msigdb_loader import MSIGDB_COLLECTIONS, MSigDBLoader

GENE_SETS = {"HALLMARK_TEST": {"geneSymbols": ["CD3E", "CD4"]}}


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_missing_collections_ignores_version(tmp_path: Path):
    assert len(msigdb_fetch.missing_collections(tmp_path)) == len(MSIGDB_COLLECTIONS)
    (tmp_path / "h.all.v2025.1.Hs.json").write_text("{}")
    missing = msigdb_fetch.missing_collections(tmp_path)
    assert "h.all" not in missing
    assert len(missing) == len(MSIGDB_COLLECTIONS) - 1


def test_download_writes_files_the_loader_reads(tmp_path: Path):
    urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        urls.append(str(request.url))
        return httpx.Response(200, json=GENE_SETS)

    done, failed = msigdb_fetch.download_collections(
        tmp_path, ["h.all", "c8.all"], client=_client(handler)
    )
    assert (done, failed) == (["h.all", "c8.all"], [])
    assert urls[0] == f"{msigdb_fetch.BASE_URL}/h.all.v{msigdb_fetch.MSIGDB_RELEASE}.json"
    loader = MSigDBLoader(str(tmp_path))
    assert loader.collection_size("h.all") == 1
    assert not list(tmp_path.glob("*.part"))


def test_http_error_and_bad_json_leave_no_file(tmp_path: Path):
    def handler(request: httpx.Request) -> httpx.Response:
        if "h.all" in request.url.path:
            return httpx.Response(404)
        return httpx.Response(200, text="<html>captive portal</html>")

    done, failed = msigdb_fetch.download_collections(
        tmp_path, ["h.all", "c8.all"], client=_client(handler)
    )
    assert (done, failed) == ([], ["h.all", "c8.all"])
    assert list(tmp_path.iterdir()) == []


def test_ensure_skips_network_when_complete(tmp_path: Path, monkeypatch):
    for c in MSIGDB_COLLECTIONS:
        (tmp_path / f"{c['id']}.v2026.1.Hs.json").write_text(json.dumps(GENE_SETS))

    def fail(*args, **kwargs):
        raise AssertionError("should not download")

    monkeypatch.setattr(msigdb_fetch, "download_collections", fail)
    msigdb_fetch.ensure_msigdb(tmp_path)


def test_ensure_never_raises_on_unwritable_dir(tmp_path: Path):
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("")
    msigdb_fetch.ensure_msigdb(blocker / "msigdb")
