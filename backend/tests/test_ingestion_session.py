"""Tests for the ingest session manager (stage [5]) and its dataset integration.

Exercises the full pipeline through a real DatasetManager: create session →
stage files → inspect state → commit → dataset registered and resolvable.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.io
import scipy.sparse as sp

from scview.core.dataset_manager import DatasetManager
from scview.core.ingestion import IngestCommitError, IngestOptions, IngestSessionManager


# ---------------------------------------------------------------------------
# Fixtures / builders
# ---------------------------------------------------------------------------


@pytest.fixture()
def data_dir(tmp_path: Path) -> Path:
    for sub in ("uploads", "converted", "cache"):
        (tmp_path / sub).mkdir()
    return tmp_path


@pytest.fixture()
def mgr(data_dir: Path) -> IngestSessionManager:
    return IngestSessionManager(data_dir=str(data_dir))


@pytest.fixture()
def dm(data_dir: Path) -> DatasetManager:
    return DatasetManager(data_dir=str(data_dir))


def _barcode(i: int) -> str:
    bases = "ACGT"
    tail = "".join(bases[(i >> (2 * k)) & 3] for k in range(8))
    return f"AAACCTGA{tail}-1"


def _stage_h5ad(mgr, sid, fname="data.h5ad", n_obs=6, n_var=4, var_prefix="Gene"):
    a = ad.AnnData(X=np.arange(n_obs * n_var, dtype="float32").reshape(n_obs, n_var))
    a.var_names = [f"{var_prefix}{i}" for i in range(n_var)]
    a.write_h5ad(mgr.files_dir(sid) / fname)


def _stage_mex(mgr, sid, prefix="", n_genes=3, n_cells=2):
    d = mgr.files_dir(sid)
    dense = np.arange(1, n_genes * n_cells + 1).reshape(n_genes, n_cells)
    scipy.io.mmwrite(str(d / f"{prefix}matrix.mtx"), sp.csr_matrix(dense))
    (d / f"{prefix}barcodes.tsv").write_text("".join(_barcode(i) + "\n" for i in range(n_cells)))
    (d / f"{prefix}features.tsv").write_text(
        "".join(f"ENSG{i:011d}\tGENE{i}\tGene Expression\n" for i in range(n_genes))
    )


# ---------------------------------------------------------------------------
# Session lifecycle
# ---------------------------------------------------------------------------


def test_create_and_discard(mgr):
    sid = mgr.create_session()
    assert mgr.exists(sid)
    assert mgr.files_dir(sid).is_dir()
    assert mgr.discard(sid) is True
    assert not mgr.exists(sid)


def test_options_roundtrip(mgr):
    sid = mgr.create_session()
    mgr.set_options(sid, IngestOptions(name="My Data", sample_label="donor"))
    opts = mgr.get_options(sid)
    assert opts.name == "My Data"
    assert opts.sample_label == "donor"


def test_state_reports_incomplete_mex(mgr):
    sid = mgr.create_session()
    dense = np.array([[1, 0], [0, 2], [3, 4]])
    scipy.io.mmwrite(str(mgr.files_dir(sid) / "matrix.mtx"), sp.csr_matrix(dense))
    state = mgr.get_state(sid)
    assert state["validation"]["ok"] is False
    assert any(i["code"] == "mex_incomplete" for i in state["validation"]["issues"])


# ---------------------------------------------------------------------------
# Commit → dataset registration
# ---------------------------------------------------------------------------


async def test_commit_single_h5ad(mgr, dm, data_dir):
    sid = mgr.create_session()
    _stage_h5ad(mgr, sid, n_obs=6, n_var=4)
    mgr.set_options(sid, IngestOptions(name="my_h5ad"))

    dataset_id = await mgr.commit(sid, dm)

    # canonical h5ad in ingested/, metadata in uploads/, session gone
    assert (data_dir / "ingested" / dataset_id / "my_h5ad.h5ad").exists()
    assert (data_dir / "uploads" / dataset_id / "metadata.json").exists()
    assert not mgr.exists(sid)

    # registered + resolvable with populated counts
    info = dm.get_dataset_info(dataset_id)
    assert info["name"] == "my_h5ad"
    assert info["n_cells"] == 6
    assert info["n_genes"] == 4
    assert any(d["id"] == dataset_id for d in dm.list_datasets())


async def test_commit_mex(mgr, dm, data_dir):
    sid = mgr.create_session()
    _stage_mex(mgr, sid, n_genes=3, n_cells=2)
    dataset_id = await mgr.commit(sid, dm)
    info = dm.get_dataset_info(dataset_id)
    assert info["n_cells"] == 2
    assert info["n_genes"] == 3
    # originals retained read-only
    assert (data_dir / "uploads" / dataset_id / "matrix.mtx").exists()


async def test_commit_merge_two_samples(mgr, dm):
    sid = mgr.create_session()
    _stage_mex(mgr, sid, prefix="GSM1_", n_genes=3, n_cells=2)
    _stage_mex(mgr, sid, prefix="GSM2_", n_genes=3, n_cells=2)
    mgr.set_options(sid, IngestOptions(name="merged"))

    plan = mgr.merge_plan(sid)
    assert plan is not None and len(plan.samples) == 2

    dataset_id = await mgr.commit(sid, dm)
    info = dm.get_dataset_info(dataset_id)
    assert info["n_cells"] == 4  # 2 + 2
    assert info["n_genes"] == 3
    adaptor = dm.get_dataset(dataset_id)
    assert "sample" in adaptor.adata.obs.columns


async def test_commit_records_provenance(mgr, dm, data_dir):
    from scview.core import provenance

    sid = mgr.create_session()
    _stage_h5ad(mgr, sid, fname="ovary.h5ad")
    dataset_id = await mgr.commit(sid, dm)
    h5 = next((data_dir / "ingested" / dataset_id).glob("*.h5ad"))
    p = provenance.read_provenance(ad.read_h5ad(h5))
    assert p["source"]["origin"] == "ingested"
    assert p["source"]["format"] == "anndata"
    assert p["source"]["original_filename"] == "ovary.h5ad"
    assert any(h["step"] == "ingest" for h in p["history"])


async def test_commit_rejects_incomplete(mgr, dm):
    sid = mgr.create_session()
    dense = np.array([[1, 0], [0, 2], [3, 4]])
    scipy.io.mmwrite(str(mgr.files_dir(sid) / "matrix.mtx"), sp.csr_matrix(dense))
    with pytest.raises(IngestCommitError):
        await mgr.commit(sid, dm)


async def test_commit_rejects_empty(mgr, dm):
    sid = mgr.create_session()
    with pytest.raises(IngestCommitError):
        await mgr.commit(sid, dm)


def _fake_converter(monkeypatch, seen: list, mgr=None, sid=None, fail: str | None = None,
                    n_obs=5):
    """trigger_conversion replaced: writes a small h5ad where the converter would (cells
    named like the .rds's Seurat object), records the call and the session's state."""
    from scview.core import conversion

    async def fake(rds_path, output_path, converter_url, progress_path=None, transport=None):
        Path(progress_path).write_text(json.dumps(
            {"stage": "converting", "source_class": "Seurat", "n_cells": n_obs, "n_genes": 4}))
        state = mgr.get_state(sid)["converting"] if mgr is not None else None
        seen.append({"rds": rds_path, "out": output_path, "url": converter_url, "state": state})
        if fail:
            raise conversion.ConversionError(fail)
        a = ad.AnnData(X=sp.csr_matrix(np.arange(n_obs * 4, dtype="float64").reshape(n_obs, 4)))
        a.obs_names = [f"seurat_{i}" for i in range(n_obs)]
        a.var_names = [f"GENE{i}" for i in range(4)]
        a.obs["seurat_clusters"] = pd.Categorical(["0", "1"] * (n_obs // 2) + ["0"] * (n_obs % 2))
        a.obsm["X_umap"] = np.zeros((n_obs, 2))
        a.write_h5ad(output_path)
        return {"status": "ok", "message": "Converted",
                "report": {"source_class": "Seurat", "detail": "Seurat v5", "kept": [], "skipped": []}}

    monkeypatch.setattr(conversion, "trigger_conversion", fake)


async def test_seurat_among_other_files_is_converted_at_commit(data_dir, dm, monkeypatch):
    from scview.core import provenance

    mgr = IngestSessionManager(data_dir=str(data_dir), converter_url="http://conv:8001")
    sid = mgr.create_session()
    _stage_mex(mgr, sid, prefix="tenx_", n_genes=4, n_cells=3)
    (mgr.files_dir(sid) / "obj.rds").write_bytes(b"\x1f\x8b not really R data")
    state = mgr.get_state(sid)
    codes = [i["code"] for i in state["validation"]["issues"]]
    assert "seurat_converted_at_commit" in codes
    assert state["validation"]["ok"] is True
    assert state["converting"] is None
    assert mgr.merge_plan(sid) is None  # the .rds is read only at commit

    seen: list = []
    _fake_converter(monkeypatch, seen, mgr, sid)
    did = await mgr.commit(sid, dm)
    assert len(seen) == 1 and seen[0]["url"] == "http://conv:8001"
    assert seen[0]["rds"].endswith("files/obj.rds")
    # the state names the R object while it converts
    assert seen[0]["state"]["file"] == "obj.rds" and seen[0]["state"]["of"] == 1
    assert seen[0]["state"]["progress"]["source_class"] == "Seurat"
    info = dm.get_dataset_info(did)
    assert info["n_cells"] == 3 + 5
    assert info["conversion"]["source_class"] == "Seurat"  # shown on Data Assessment
    h5 = next((data_dir / "ingested" / did).glob("*.h5ad"))
    back = ad.read_h5ad(h5)
    assert set(back.obs["sample"]) == {"tenx", "obj"}
    p = provenance.read_provenance(back)
    assert {m["format"] for m in p["source"]["merged_from"]} == {"tenx_mex", "seurat"}
    ingest = next(h for h in p["history"] if h["step"] == "ingest")
    assert ingest["params"]["converted"][0]["file"] == "obj.rds"
    assert ingest["params"]["converted"][0]["source_class"] == "Seurat"
    # the originals keep the .rds; the session (and its converted copy) is gone
    assert (data_dir / "uploads" / did / "obj.rds").is_file()
    assert not (data_dir / "ingest" / sid).exists()


async def test_lone_seurat_in_a_session_is_converted(mgr, dm, data_dir, monkeypatch):
    from scview.core import provenance

    sid = mgr.create_session()
    (mgr.files_dir(sid) / "pbmc.rds").write_bytes(b"RDX3")
    seen: list = []
    _fake_converter(monkeypatch, seen)
    did = await mgr.commit(sid, dm)
    back = ad.read_h5ad(next((data_dir / "ingested" / did).glob("*.h5ad")))
    assert back.n_obs == 5 and "X_umap" in back.obsm
    src = provenance.read_provenance(back)["source"]
    assert src["format"] == "seurat" and src["original_filename"] == "pbmc.rds"


async def test_conversion_failure_is_a_commit_error(mgr, dm, monkeypatch):
    sid = mgr.create_session()
    _stage_h5ad(mgr, sid, "a.h5ad")
    (mgr.files_dir(sid) / "obj.rds").write_bytes(b"RDX3")
    seen: list = []
    _fake_converter(monkeypatch, seen, fail="This file holds a list object, which scView "
                    "cannot convert.")
    with pytest.raises(IngestCommitError, match="obj.rds could not be converted: This file holds"):
        await mgr.commit(sid, dm)
    assert mgr.get_state(sid)["converting"] is None  # cleared; the session stays to retry


async def test_invalid_session_converts_nothing(mgr, dm, monkeypatch):
    sid = mgr.create_session()
    dense = np.array([[1, 0], [0, 2], [3, 4]])
    scipy.io.mmwrite(str(mgr.files_dir(sid) / "matrix.mtx"), sp.csr_matrix(dense))
    (mgr.files_dir(sid) / "obj.rds").write_bytes(b"RDX3")
    seen: list = []
    _fake_converter(monkeypatch, seen)
    with pytest.raises(IngestCommitError):
        await mgr.commit(sid, dm)
    assert seen == []


async def test_unreadable_file_is_a_commit_error(mgr, dm):
    sid = mgr.create_session()
    (mgr.files_dir(sid) / "broken.h5ad").write_bytes(b"not an HDF5 file")
    with pytest.raises(IngestCommitError, match="Could not read"):
        await mgr.commit(sid, dm)
    assert mgr.merge_plan(sid) is None


def test_persist_drops_an_index_name_that_is_also_a_column(mgr, data_dir):
    # ovary.h5ad: obs index named "cell_id" plus a "cell_id" column with other
    # values; anndata refuses to write that, which was a 500 at commit.
    a = ad.AnnData(X=np.ones((3, 2), dtype="float32"))
    a.obs_names = ["c0", "c1", "c2"]
    a.obs["cell_id"] = ["x", "y", "z"]
    a.obs.index.name = "cell_id"
    mgr._persist_dataset("ds1", "clash", a, [])
    back = ad.read_h5ad(data_dir / "ingested" / "ds1" / "clash.h5ad")
    assert list(back.obs_names) == ["c0", "c1", "c2"]
    assert list(back.obs["cell_id"]) == ["x", "y", "z"]


def test_persist_failure_is_a_commit_error_and_leaves_nothing(mgr, data_dir, monkeypatch):
    def boom(self, *args, **kwargs):
        raise ValueError("disk full")

    monkeypatch.setattr(ad.AnnData, "write_h5ad", boom)
    a = ad.AnnData(X=np.ones((2, 2), dtype="float32"))
    with pytest.raises(IngestCommitError, match="could not be saved: disk full"):
        mgr._persist_dataset("ds2", "broken", a, [])
    assert not (data_dir / "uploads" / "ds2").exists()
    assert not (data_dir / "ingested" / "ds2").exists()


# ---------------------------------------------------------------------------
# TTL sweep
# ---------------------------------------------------------------------------


def test_sweep_expired(mgr):
    sid = mgr.create_session()
    # backdate created_at beyond the TTL
    sfile = mgr._session_dir(sid) / "session.json"
    data = json.loads(sfile.read_text())
    data["created_at"] = (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat()
    sfile.write_text(json.dumps(data))

    fresh = mgr.create_session()
    removed = mgr.sweep_expired(ttl_hours=24)
    assert removed == 1
    assert not mgr.exists(sid)
    assert mgr.exists(fresh)
