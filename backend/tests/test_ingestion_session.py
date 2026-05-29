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
