"""A cancelled or refused upload leaves nothing behind: no part file, no empty store folder made
for it, no file kept from a set that was refused (v0.84). The browser's Cancel aborts the XHR,
which the server sees as a disconnect mid-body."""

from __future__ import annotations

import uuid
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from scview.core import atac_fragments as af
from scview.core import cell_outlines, modality_store, raw_upload
from scview.core import transcripts as tr
from scview.core import velocity_input as vi
from scview.main import app
from tests.fragments_fixture import write_fragments
from tests.test_hvg_sensitivity import _register
from tests.test_transcripts import _adata as _spatial_adata

API = "/api/v1"


def _plain(n: int = 20) -> ad.AnnData:
    return ad.AnnData(X=sp.csr_matrix(np.ones((n, 3), dtype=np.float32)),
                      obs=pd.DataFrame(index=[f"BC{i:04d}-1" for i in range(n)]),
                      var=pd.DataFrame(index=["A", "B", "C"]))


async def _post_then_drop(path: str, query: str, content_type: str, sent: list[bytes], declared: int) -> int:
    """POST that sends `sent`, then the client goes away (as an aborted XHR): the status answered."""
    messages = [{"type": "http.request", "body": b, "more_body": True} for b in sent]
    answers: list[dict] = []

    async def receive():
        return messages.pop(0) if messages else {"type": "http.disconnect"}

    async def send(message):
        answers.append(message)

    scope = {
        "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "POST", "scheme": "http",
        "path": path, "raw_path": path.encode(), "query_string": query.encode(), "root_path": "",
        "headers": [(b"host", b"testserver"), (b"content-type", content_type.encode()),
                    (b"content-length", str(declared).encode())],
        "server": ("testserver", 80), "client": ("testclient", 50000),
    }
    await app(scope, receive, send)
    return next(m["status"] for m in answers if m["type"] == "http.response.start")


def _multipart_head(field: str, name: str) -> tuple[str, bytes]:
    b = uuid.uuid4().hex
    head = (f"--{b}\r\nContent-Disposition: form-data; name=\"{field}\"; filename=\"{name}\"\r\n"
            "Content-Type: application/octet-stream\r\n\r\n").encode()
    return f"multipart/form-data; boundary={b}", head


def test_drop_if_empty_keeps_a_folder_with_anything_in_it(tmp_path):
    empty = tmp_path / "a"
    empty.mkdir()
    raw_upload.drop_if_empty(empty)
    assert not empty.exists()
    full = tmp_path / "b"
    full.mkdir()
    (full / "peaks.h5").write_bytes(b"x")
    raw_upload.drop_if_empty(full)
    assert (full / "peaks.h5").exists()
    raw_upload.drop_if_empty(tmp_path / "never")  # no folder: nothing to do


async def test_a_cancelled_fragments_upload_leaves_no_folder(client, dataset_manager, tmp_path):
    did = _register(dataset_manager, _plain(), "ca")
    frag, _idx = write_fragments(tmp_path / "f.tsv.gz", [("chr1", 100 + i, 200 + i, "BC0000-1", 1) for i in range(500)])
    head = frag.read_bytes()[:2048]
    status = await _post_then_drop(f"{API}/datasets/{did}/atac/fragments", "name=atac_fragments.tsv.gz",
                                   "application/octet-stream", [head, b"\0" * 4096], 10**8)
    assert status == 400
    assert not modality_store.store_dir(dataset_manager.data_dir, did).exists()
    assert not af.fragments_path(dataset_manager.data_dir, did).exists()


async def test_a_cancelled_upload_keeps_what_the_folder_held(client, dataset_manager, tmp_path):
    did = _register(dataset_manager, _plain(), "cb")
    folder = modality_store.store_dir(dataset_manager.data_dir, did)
    folder.mkdir(parents=True)
    (folder / "peaks.h5").write_bytes(b"kept")
    status = await _post_then_drop(f"{API}/datasets/{did}/atac/fragments", "name=atac_fragments.tsv.gz.tbi",
                                   "application/octet-stream", [b"\0" * 4096], 10**8)
    assert status == 400
    assert sorted(p.name for p in folder.iterdir()) == ["peaks.h5"]


async def test_a_cancelled_transcripts_upload_leaves_no_folder(client, dataset_manager):
    did = _register(dataset_manager, _spatial_adata(), "cc")
    status = await _post_then_drop(f"{API}/datasets/{did}/spatial/transcripts", "name=transcripts.parquet",
                                   "application/octet-stream", [tr.PARQUET_MAGIC + b"\0" * 4096], 10**7)
    assert status == 400
    assert not tr.store_dir(dataset_manager.data_dir, did).exists()


async def test_a_cancelled_outlines_upload_leaves_no_folder(client, dataset_manager):
    did = _register(dataset_manager, _spatial_adata(), "cd")
    status = await _post_then_drop(f"{API}/datasets/{did}/spatial/outlines", "name=cell_boundaries.parquet",
                                   "application/octet-stream", [b"PAR1" + b"\0" * 4096], 10**7)
    assert status == 400
    assert not cell_outlines.store_dir(dataset_manager.data_dir, did).exists()


async def test_a_refused_outlines_file_leaves_no_folder(client, dataset_manager):
    did = _register(dataset_manager, _spatial_adata(), "ce")
    r = await client.post(f"{API}/datasets/{did}/spatial/outlines", params={"name": "cell_boundaries.parquet"},
                          content=b"not parquet at all")
    assert r.status_code == 422
    assert not cell_outlines.store_dir(dataset_manager.data_dir, did).exists()


async def test_a_cancelled_multipart_upload_never_reaches_the_route(client, dataset_manager):
    did = _register(dataset_manager, _plain(), "cf")
    before = sorted(Path(dataset_manager.data_dir).rglob("*"))
    for path, field, name in [(f"{API}/datasets/{did}/velocity/attach", "files", "sample.loom"),
                              (f"{API}/datasets/{did}/vdj/upload", "file", "contigs.csv"),
                              (f"{API}/datasets/{did}/deconvolution/upload", "file", "bulk.tsv")]:
        ctype, head = _multipart_head(field, name)
        status = await _post_then_drop(path, "", ctype, [head, b"x" * 4096], 10**7)
        assert status == 400, path
    assert sorted(Path(dataset_manager.data_dir).rglob("*")) == before


async def test_a_refused_file_in_a_velocity_set_keeps_none_of_the_set(client, dataset_manager, monkeypatch):
    did = _register(dataset_manager, _plain(), "cg")
    monkeypatch.setattr(vi, "MAX_UPLOAD_BYTES", 1000)
    r = await client.post(f"{API}/datasets/{did}/velocity/attach",
                          files=[("files", ("a.loom", b"x" * 500)), ("files", ("b.loom", b"y" * 5000))])
    assert r.status_code == 413
    folder = Path(dataset_manager.data_dir) / "uploads" / did / "velocity"
    assert not folder.exists()


async def test_an_oversized_contig_table_leaves_no_folder(client, dataset_manager, monkeypatch):
    from scview.core import vdj

    did = _register(dataset_manager, _plain(), "ch")
    monkeypatch.setattr(vdj, "MAX_UPLOAD_BYTES", 1000)
    r = await client.post(f"{API}/datasets/{did}/vdj/upload", files={"file": ("contigs.csv", b"x" * 5000)})
    assert r.status_code == 413
    assert not (Path(dataset_manager.data_dir) / "uploads" / did / "vdj").exists()


# Every route the browser uploads a file to. Large: no route cap under a gigabyte, or the dataset
# is saved after it (the proxy's long timeout); these go through nginx's uncapped location.
SMALL_UPLOADS = ["/api/v1/genesets/custom/upload", "/api/v1/datasets/d1/deconvolution/upload"]
LARGE_UPLOADS = [
    "/api/v1/datasets/upload", "/api/v1/ingest/session/s1/files", "/api/v1/datasets/d1/seq-metrics",
    "/api/v1/datasets/d1/atac/fragments", "/api/v1/datasets/d1/atac/metrics", "/api/v1/datasets/d1/vdj/upload",
    "/api/v1/datasets/d1/labels/import-file", "/api/v1/datasets/d1/velocity/attach",
    "/api/v1/datasets/d1/spatial/outlines", "/api/v1/datasets/d1/spatial/transcripts",
]


def test_every_upload_route_passes_the_json_cap_and_the_large_ones_the_proxy_cap():
    import re

    from scview.main import _UPLOAD_PATH_HINTS

    for path in SMALL_UPLOADS + LARGE_UPLOADS:
        assert any(h in path for h in _UPLOAD_PATH_HINTS), path
    conf = Path(__file__).parents[2] / "frontend" / "nginx.conf"
    if not conf.exists():  # not mounted in a backend-only container
        return
    blocks = re.findall(r"location ~ (\S+) \{([^}]*)\}", conf.read_text())
    uncapped = [pattern for pattern, body in blocks if "client_max_body_size 0;" in body]
    assert len(uncapped) == 1
    for path in LARGE_UPLOADS:
        assert re.match(uncapped[0], path), path
    for path in SMALL_UPLOADS:
        assert not re.match(uncapped[0], path), path
