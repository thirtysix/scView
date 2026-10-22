"""R object conversion: the converter client's errors, and the upload that
converts in the background (status converting -> ready with the report, or
error with the converter's own sentence)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import anndata as ad
import httpx
import numpy as np
import pytest
import scipy.sparse as sp

from scview.core import conversion

UNKNOWN_CLASS = ("This file holds a list object, which scView cannot convert. scView converts "
                 "Seurat, SingleCellExperiment and SummarizedExperiment objects, and classes "
                 "built on them.")

REPORT = {
    "source_class": "SingleCellExperiment", "x": "logcounts", "layers": ["counts"],
    "kept": ["X: the logcounts assay (sparse)"], "skipped": ["altExp ERCC (4 features): not converted"],
}


# ---------------------------------------------------------------------------
# trigger_conversion
# ---------------------------------------------------------------------------

async def test_unknown_class_raises_the_converters_own_sentence():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"status": "error", "error": UNKNOWN_CLASS})

    with pytest.raises(conversion.ConversionError) as exc:
        await conversion.trigger_conversion("/d/x.rds", "/d/x.h5ad", "http://conv:8001",
                                            transport=httpx.MockTransport(handler))
    # not "Server error '500 Internal Server Error' for url ..."
    assert str(exc.value) == UNKNOWN_CLASS


async def test_success_returns_the_report_and_sends_the_progress_path():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"status": "ok", "message": "Converted", "report": REPORT})

    out = await conversion.trigger_conversion("/d/x.rds", "/d/x.h5ad", "http://conv:8001",
                                              progress_path="/d/p.json",
                                              transport=httpx.MockTransport(handler))
    assert out["report"] == REPORT
    assert seen == {"input_path": "/d/x.rds", "output_path": "/d/x.h5ad", "progress_path": "/d/p.json"}


async def test_unreachable_converter_says_so():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    with pytest.raises(conversion.ConversionError, match="could not be reached"):
        await conversion.trigger_conversion("/d/x.rds", "/d/x.h5ad", "http://conv:8001",
                                            transport=httpx.MockTransport(handler))


async def test_an_error_page_without_json_names_the_status():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(502, text="<html>Bad Gateway</html>")

    with pytest.raises(conversion.ConversionError, match="answered 502"):
        await conversion.trigger_conversion("/d/x.rds", "/d/x.h5ad", "http://conv:8001",
                                            transport=httpx.MockTransport(handler))


def test_read_progress_tolerates_missing_and_partial_files(tmp_path: Path):
    p = tmp_path / conversion.PROGRESS_FILE
    assert conversion.read_progress(p) is None
    p.write_text('{"stage": "conv')
    assert conversion.read_progress(p) is None
    p.write_text('{"stage": "converting", "source_class": "SingleCellExperiment"}')
    assert conversion.read_progress(p)["source_class"] == "SingleCellExperiment"


# ---------------------------------------------------------------------------
# upload -> background conversion -> GET
# ---------------------------------------------------------------------------

def _write_small_h5ad(path: str) -> None:
    rng = np.random.default_rng(0)
    X = sp.random(20, 10, density=0.3, format="csr", random_state=rng, dtype=np.float32)
    a = ad.AnnData(X=X)
    a.obs_names = [f"c{i}" for i in range(20)]
    a.var_names = [f"g{i}" for i in range(10)]
    a.obsm["X_umap"] = rng.normal(size=(20, 2))
    a.write_h5ad(path)


async def _finished(did: str) -> None:
    """Wait for the background conversion (it may be over already)."""
    task = conversion.task_for(did)
    if task is not None:
        await task


async def _upload(client, name: str = "obj.rds") -> dict:
    files = {"file": (name, b"RDX3\n not really an rds", "application/octet-stream")}
    resp = await client.post("/api/v1/datasets/upload", files=files)
    assert resp.status_code == 201, resp.text
    return resp.json()


async def test_upload_returns_converting_then_ready_with_the_report(client, monkeypatch):
    release = asyncio.Event()
    calls: list[dict] = []

    async def fake(rds_path, output_path, converter_url, progress_path=None, transport=None):
        calls.append({"rds": rds_path, "out": output_path, "progress": progress_path})
        Path(progress_path).write_text(json.dumps(
            {"stage": "converting", "message": "Converting a SingleCellExperiment (20 cells x 10 features)",
             "source_class": "SingleCellExperiment", "n_cells": 20, "n_genes": 10}))
        await release.wait()
        _write_small_h5ad(output_path)
        return {"status": "ok", "message": "Converted", "report": REPORT}

    monkeypatch.setattr(conversion, "trigger_conversion", fake)
    up = await _upload(client)
    did = up["id"]
    assert up["status"] == "converting"  # the response does not wait for R

    for _ in range(50):  # let the background task reach the converter
        if calls:
            break
        await asyncio.sleep(0.01)
    during = (await client.get(f"/api/v1/datasets/{did}")).json()
    assert during["status"] == "converting"
    assert during["conversion_progress"]["source_class"] == "SingleCellExperiment"
    assert calls[0]["progress"].endswith(f"converted/{did}/{conversion.PROGRESS_FILE}")

    release.set()
    await _finished(did)
    info = (await client.get(f"/api/v1/datasets/{did}")).json()
    assert info["status"] == "ready"
    assert info["conversion"] == REPORT
    assert info["n_cells"] == 20 and info["n_genes"] == 10
    assert "X_umap" in info["available_embeddings"]


async def test_upload_of_an_unknown_class_ends_in_the_converters_error(client, monkeypatch):
    async def refuse(rds_path, output_path, converter_url, progress_path=None, transport=None):
        raise conversion.ConversionError(UNKNOWN_CLASS)

    monkeypatch.setattr(conversion, "trigger_conversion", refuse)
    did = (await _upload(client))["id"]
    await _finished(did)
    info = (await client.get(f"/api/v1/datasets/{did}")).json()
    assert info["status"] == "error"
    assert info["error_message"] == UNKNOWN_CLASS
    assert "conversion" not in info


async def test_a_conversion_cut_off_by_a_restart_becomes_an_error(client, dataset_manager):
    did = "0b9c3f4e-1111-4222-8333-944455556666"
    d = Path(dataset_manager.data_dir) / "uploads" / did
    d.mkdir(parents=True)
    (d / "metadata.json").write_text(json.dumps(
        {"id": did, "name": "obj", "filename": "obj.rds", "status": "converting"}))
    # no task in this process: the backend that started it is gone
    info = (await client.get(f"/api/v1/datasets/{did}")).json()
    assert info["status"] == "error"
    assert info["error_message"] == conversion.INTERRUPTED
