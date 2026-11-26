"""Every new dataset is stored by rows: the ingest paths and what the estimates count
(``core/csr_store.py``). Expected matrices are scipy's ``tocsr()`` of the source."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import anndata as ad
import numpy as np
import pytest

from scview.core import csr_store, preflight
from scview.core.ingestion import IngestOptions, IngestSessionManager, feature_split
from tests.test_csr_store import _assert_same_dataset, _dataset, _formats, _random_csc, _same

API = "/api/v1"


# ---------------------------------------------------------------------------
# Ingest: every new dataset is stored by rows
# ---------------------------------------------------------------------------


async def _upload(client, path: Path) -> str:
    r = await client.post(f"{API}/datasets/upload", files={"file": (path.name, path.read_bytes(), "x")})
    assert r.status_code == 201, r.text
    return r.json()["id"]


async def test_plain_upload_of_a_csc_file_gets_a_row_stored_working_copy(client, dataset_manager, tmp_data_dir,
                                                                         tmp_path):
    a = _dataset("csc")
    src = tmp_path / "sample.h5ad"
    a.write_h5ad(src, compression="gzip")
    did = await _upload(client, src)
    stored = tmp_data_dir / "uploads" / did / "sample.h5ad"
    assert stored.read_bytes() == src.read_bytes()  # the upload, as sent
    working = tmp_data_dir / "ingested" / did / "sample.h5ad"
    assert dataset_manager.h5ad_candidates(did)[0] == str(working)
    assert [p.name for p in working.parent.iterdir()] == ["sample.h5ad"]
    assert _formats(working) == {"X": "csr_matrix", "layers/counts": "csr_matrix", "layers/dense": "array",
                                 "raw/X": "csr_matrix"}
    meta = json.loads((stored.parent / "metadata.json").read_text())
    assert meta["stored_by_rows"] == {"matrices": ["X", "layers/counts", "raw/X"]}
    loaded = (await dataset_manager.get_or_load_dataset(did)).adata
    _same(loaded.X, a.X.tocsr())
    _assert_same_dataset(ad.read_h5ad(src), loaded)
    assert not dataset_manager._reads  # the memory held for it is let go


@pytest.mark.parametrize("fmt", ["csr", "dense"])
async def test_plain_upload_without_csc_is_opened_as_sent(client, tmp_data_dir, tmp_path, fmt):
    src = tmp_path / "sample.h5ad"
    _dataset(fmt).write_h5ad(src)
    did = await _upload(client, src)
    assert not (tmp_data_dir / "ingested" / did).exists()
    assert "stored_by_rows" not in json.loads((tmp_data_dir / "uploads" / did / "metadata.json").read_text())


async def test_a_failed_rewrite_leaves_the_upload_openable(client, dataset_manager, tmp_data_dir, tmp_path,
                                                           monkeypatch):
    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(csr_store, "convert", boom)
    src = tmp_path / "sample.h5ad"
    _dataset("csc").write_h5ad(src)
    did = await _upload(client, src)
    assert not (tmp_data_dir / "ingested" / did).exists()
    assert (await dataset_manager.get_or_load_dataset(did)).adata.X.format == "csc"
    assert not dataset_manager._reads


def test_feature_split_of_a_csc_file_writes_csr(tmp_path):
    a = _dataset("csc", g=12)
    a.var["feature_types"] = ["Gene Expression"] * 9 + ["Antibody Capture"] * 3
    src, dst = tmp_path / "cite.h5ad", tmp_path / "out.h5ad"
    a.write_h5ad(src)
    feature_split.split_h5ad(src, dst, original_filename="cite.h5ad")
    assert _formats(dst) == {"X": "csr_matrix", "layers/counts": "csr_matrix", "layers/dense": "array",
                             "raw/X": "csr_matrix"}
    got = ad.read_h5ad(dst)
    _same(got.X, a.X[:, :9].tocsr())
    _same(got.raw.X, a.raw.X.tocsr())  # .raw has its own features: kept whole
    np.testing.assert_array_equal(got.obsm["protein_counts"].to_numpy(), a.X[:, 9:].toarray())


def _commit(data_dir: Path, files: list[Path], **opts) -> Path:
    mgr = IngestSessionManager(str(data_dir))
    sid = mgr.create_session()
    for f in files:
        shutil.copyfile(f, mgr.files_dir(sid) / f.name)
    mgr.set_options(sid, IngestOptions(**opts))
    did = mgr._build_dataset(sid)
    return next((data_dir / "ingested" / did).glob("*.h5ad"))


@pytest.mark.parametrize("streamed", ["off", "on"])
def test_ingest_session_stores_a_csc_h5ad_by_rows(tmp_data_dir, tmp_path, streamed):
    a = _dataset("csc")
    src = tmp_path / "sample.h5ad"
    a.write_h5ad(src)
    out = _commit(tmp_data_dir, [src], streamed=streamed)
    assert _formats(out) == {"X": "csr_matrix", "layers/counts": "csr_matrix", "layers/dense": "array",
                             "raw/X": "csr_matrix"}
    got = ad.read_h5ad(out)
    _same(got.X, a.X.tocsr())
    _same(got.layers["counts"], a.layers["counts"].tocsr())
    _same(got.raw.X, a.raw.X.tocsr())
    np.testing.assert_array_equal(got.layers["dense"], a.layers["dense"])
    assert list(got.obs_names) == list(a.obs_names) and list(got.var_names) == list(a.var_names)


def test_ingest_session_stores_mex_by_rows_in_memory(tmp_data_dir, tmp_path):
    d = tmp_path / "mex"
    d.mkdir()
    (d / "matrix.mtx").write_text("%%MatrixMarket matrix coordinate integer general\n3 2 4\n"
                                  "1 1 2\n3 1 1\n2 2 4\n3 2 6\n")
    (d / "barcodes.tsv").write_text("AAACCTGAGAAACCAT-1\nAAACCTGAGAAACCGC-1\n")
    (d / "features.tsv").write_text("".join(f"E{i}\tG{i}\tGene Expression\n" for i in range(3)))
    out = _commit(tmp_data_dir, sorted(d.iterdir()), streamed="off")
    assert _formats(out) == {"X": "csr_matrix"}
    assert ad.read_h5ad(out).X.toarray().tolist() == [[2, 0, 1], [0, 4, 6]]


# ---------------------------------------------------------------------------
# The estimates count it
# ---------------------------------------------------------------------------


def test_the_load_estimate_counts_the_copy_of_a_csc_matrix(tmp_path):
    n, g = 4000, 300
    files = {}
    for fmt in ("csc", "csr"):
        m = _random_csc(n, g, 0.3, np.float32, seed=2)
        a = ad.AnnData(m if fmt == "csc" else m.tocsr())
        a.raw = a
        files[fmt] = tmp_path / f"{fmt}.h5ad"
        a.write_h5ad(files[fmt])
    h_csc, h_csr = (preflight.read_header(files[k], "h5ad") for k in ("csc", "csr"))
    assert [m.csc for m in h_csc.matrices] == [True, True] and not any(m.csc for m in h_csr.matrices)
    csc, csr = preflight.estimate_load(h_csc), preflight.estimate_load(h_csr)
    one = h_csc.x.nbytes
    assert csc.resident_bytes == csr.resident_bytes
    assert csc.peak_bytes == csc.resident_bytes + one  # one matrix's copy, not both
    assert csc.peak_stage == "storing the matrix by rows" and csr.peak_stage == "saving the dataset"
    assert csc.peak_bytes > csr.peak_bytes
    # Only an ingest stores it by rows: opening an existing dataset copies nothing.
    assert preflight.estimate_load(h_csc, ingest=False).peak_bytes == \
        preflight.estimate_load(h_csr, ingest=False).peak_bytes


def test_streamed_needs_count_the_temporary_files_and_the_range(tmp_path, monkeypatch):
    from scview.core.ingestion import streamed
    from scview.core.ingestion.bundling import build_bundle

    for fmt in ("csc", "csr"):
        m = _random_csc(3000, 100, 0.3, np.float32, seed=2)
        ad.AnnData(m if fmt == "csc" else m.tocsr()).write_h5ad(tmp_path / f"{fmt}.h5ad")
    monkeypatch.setattr(csr_store, "RANGE_BYTES", 1024)
    need = {fmt: streamed.disk_need(build_bundle([str(tmp_path / f"{fmt}.h5ad")]).units)
            for fmt in ("csc", "csr")}
    nnz = _random_csc(3000, 100, 0.3, np.float32, seed=2).nnz
    assert need["csc"] - need["csr"] == nnz * 12
    # Memory: the larger of the streamed block and the conversion's range of cells.
    monkeypatch.undo()
    mem = {fmt: streamed.streamed_need(build_bundle([str(tmp_path / f"{fmt}.h5ad")]).units) for fmt in ("csc", "csr")}
    assert mem["csc"] - mem["csr"] == csr_store.working_bytes(n_obs=3000) - streamed.STREAM_WORKING_BYTES > 0
