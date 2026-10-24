"""Upload size caps: MAX_UPLOAD_SIZE_MB in public mode only; private mode takes any size.

A 10x file of a million cells is 4.2 GB, and the old 2 GB default (in the backend for
``/datasets/upload``, and nginx's ``client_max_body_size`` for every route) refused it on
the one machine scView is meant for. The import wizard's ``/ingest/session/{sid}/files``
had no cap of its own, so public mode relied on nginx: it now caps each file itself.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pytest

from scview.config import NO_UPLOAD_CAP, Settings
from scview.core.ingestion import IngestSessionManager
from scview.dependencies import get_ingest_session_manager, get_settings_dep
from scview.main import app

API = "/api/v1"
MB = 1024 * 1024


def test_the_cap_applies_in_public_mode_only(tmp_path):
    assert Settings(DATA_DIR=str(tmp_path), MAX_UPLOAD_SIZE_MB=1).max_upload_bytes == NO_UPLOAD_CAP
    assert Settings(DATA_DIR=str(tmp_path), MAX_UPLOAD_SIZE_MB=1,
                    DEPLOYMENT_MODE="public").max_upload_bytes == MB


@pytest.fixture()
def mode(tmp_data_dir):
    """Run the app with these settings (private or public, a 1 MB cap)."""
    def use(deployment: str):
        s = Settings(DATA_DIR=str(tmp_data_dir), MAX_UPLOAD_SIZE_MB=1, DEPLOYMENT_MODE=deployment,
                     DEEPINFRA_API_KEY="", RAG_DATABASE_URL="")
        app.dependency_overrides[get_settings_dep] = lambda: s
        return s
    yield use
    app.dependency_overrides.pop(get_settings_dep, None)


@pytest.fixture()
async def ingest(tmp_data_dir, client):
    mgr = IngestSessionManager(str(tmp_data_dir))
    app.dependency_overrides[get_ingest_session_manager] = lambda: mgr
    yield client, mgr
    app.dependency_overrides.pop(get_ingest_session_manager, None)


def _two_mb_h5ad(path):
    a = ad.AnnData(X=np.random.default_rng(0).random((1000, 600), dtype=np.float32))
    a.write_h5ad(path)  # uncompressed: 2.4 MB
    return path.read_bytes()


async def test_ingest_files_over_the_public_cap_are_413_and_nothing_is_staged(ingest, mode, tmp_path):
    client, mgr = ingest
    body = _two_mb_h5ad(tmp_path / "big.h5ad")
    mode("public")
    sid = mgr.create_session()
    r = await client.post(f"{API}/ingest/session/{sid}/files",
                          files=[("files", ("small.csv", b"a,b\n1,2\n", "text/csv")),
                                 ("files", ("big.h5ad", body, "application/octet-stream"))])
    assert r.status_code == 413, r.text
    assert "big.h5ad exceeds the maximum upload size of 1 MB" in r.json()["detail"]
    assert mgr.staged_paths(sid) == [] and list(mgr.files_dir(sid).iterdir()) == []


async def test_private_mode_takes_files_over_the_cap(ingest, mode, tmp_path):
    client, mgr = ingest
    body = _two_mb_h5ad(tmp_path / "big.h5ad")
    mode("private")
    sid = mgr.create_session()
    r = await client.post(f"{API}/ingest/session/{sid}/files",
                          files=[("files", ("big.h5ad", body, "application/octet-stream"))])
    assert r.status_code == 200, r.text
    assert [p.name for p in mgr.staged_paths(sid)] == ["big.h5ad"]


async def test_plain_upload_cap_follows_the_mode(client, mode, tmp_path):
    body = _two_mb_h5ad(tmp_path / "big.h5ad")
    mode("public")
    r = await client.post(f"{API}/datasets/upload", files={"file": ("big.h5ad", body)})
    assert r.status_code == 413
    mode("private")
    r = await client.post(f"{API}/datasets/upload", files={"file": ("big.h5ad", body)})
    assert r.status_code == 201, r.text


async def test_merge_plan_endpoint(ingest, tmp_path):
    client, mgr = ingest
    sid = mgr.create_session()
    for name, seed in (("a.h5ad", 1), ("b.h5ad", 2)):
        a = ad.AnnData(X=np.random.default_rng(seed).integers(0, 5, (20, 8)).astype(np.float32))
        a.var_names = [f"G{i}" for i in range(8)]
        a.write_h5ad(mgr.files_dir(sid) / name)
    r = await client.get(f"{API}/ingest/session/{sid}/merge-plan")
    assert r.status_code == 200, r.text
    assert r.json()["samples"] == ["a", "b"] and r.json()["intersection"] == 8
