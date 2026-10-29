"""Tests for the ingest session manager (stage [5]) and its dataset integration.

Exercises the full pipeline through a real DatasetManager: create session →
stage files → inspect state → commit → dataset registered and resolvable.
"""

from __future__ import annotations

import asyncio
import json
import os
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
from scview.core.ingestion.session import ConversionBusy


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
                    n_obs=5, gate=None):
    """trigger_conversion replaced: writes a small h5ad where the converter would (cells
    named like the .rds's Seurat object), records the call and the session's state.
    ``gate``: an asyncio.Event the conversion waits for (one still running)."""
    from scview.core import conversion

    async def fake(rds_path, output_path, converter_url, progress_path=None, transport=None):
        Path(progress_path).write_text(json.dumps(
            {"stage": "converting", "source_class": "Seurat", "n_cells": n_obs, "n_genes": 4}))
        state = mgr.get_state(sid)["converting"] if mgr is not None else None
        seen.append({"rds": rds_path, "out": output_path, "url": converter_url, "state": state})
        if gate is not None:
            await gate.wait()
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


# -- "Convert now": R objects converted ahead of commit ----------------------


def _rds_session(mgr, content: bytes = b"RDX3 an R object"):
    """A session with an .h5ad (genes GENE0-3, as the fake conversion has) and an .rds."""
    sid = mgr.create_session()
    _stage_h5ad(mgr, sid, "a.h5ad", n_obs=6, n_var=4, var_prefix="GENE")
    (mgr.files_dir(sid) / "obj.rds").write_bytes(content)
    return sid


def _codes(state) -> list[str]:
    return [i["code"] for i in state["validation"]["issues"]]


async def test_convert_now_puts_the_r_object_in_the_merge_plan(mgr, monkeypatch):
    sid = _rds_session(mgr)
    state = mgr.get_state(sid)
    assert "seurat_converted_at_commit" in _codes(state)
    note = next(i for i in state["validation"]["issues"] if i["code"] == "seurat_converted_at_commit")
    assert "convert it now to include it" in note["suggestion"]
    assert state["converted"] == [] and state["conversion_running"] is False
    assert mgr.merge_plan(sid) is None  # one loadable sample: no plan yet

    seen: list = []
    _fake_converter(monkeypatch, seen, mgr, sid)
    assert mgr.start_conversion(sid) is True
    assert mgr.get_state(sid)["conversion_running"] is True
    await mgr._tasks[sid]
    assert len(seen) == 1 and seen[0]["state"]["file"] == "obj.rds"
    state = mgr.get_state(sid)
    assert state["conversion_running"] is False and state["converting"] is None
    assert state["converted"] == [{"label": "obj", "file": "obj.rds"}]
    assert state["conversion_error"] is None
    assert "seurat_converted_at_commit" not in _codes(state)
    done = next(i for i in state["validation"]["issues"] if i["code"] == "seurat_converted")
    assert done["suggestion"] == "The merge plan includes it."
    # recorded with what identifies the source
    rec = json.loads((mgr._session_dir(sid) / "session.json").read_text())["conversions"]["obj.rds"]
    st = (mgr.files_dir(sid) / "obj.rds").stat()
    assert rec["size"] == st.st_size and rec["mtime_ns"] == st.st_mtime_ns
    plan = mgr.merge_plan(sid)
    assert plan is not None and set(plan.samples) == {"a", "obj"}
    assert plan.intersection == 4 and plan.est_cells == 6 + 5
    assert mgr.start_conversion(sid) is False  # nothing left to convert
    assert len(seen) == 1


async def test_commit_reuses_a_conversion(data_dir, dm, monkeypatch):
    from scview.core import provenance

    mgr = IngestSessionManager(data_dir=str(data_dir), converter_url="http://conv:8001")
    sid = _rds_session(mgr)
    seen: list = []
    _fake_converter(monkeypatch, seen, mgr, sid)
    mgr.start_conversion(sid)
    await mgr._tasks[sid]
    did = await mgr.commit(sid, dm)
    assert len(seen) == 1  # converted once, before the commit
    assert dm.get_dataset_info(did)["n_cells"] == 6 + 5
    back = ad.read_h5ad(next((data_dir / "ingested" / did).glob("*.h5ad")))
    assert set(back.obs["sample"]) == {"a", "obj"}
    ingest = next(h for h in provenance.read_provenance(back)["history"] if h["step"] == "ingest")
    assert ingest["params"]["converted"] == [{"sample": "obj", "file": "obj.rds",
                                              "tool": "scView R converter",
                                              "source_class": "Seurat", "detail": "Seurat v5"}]


async def test_a_changed_r_object_is_converted_again(mgr, dm, monkeypatch):
    sid = _rds_session(mgr)
    seen: list = []
    _fake_converter(monkeypatch, seen, mgr, sid)
    mgr.start_conversion(sid)
    await mgr._tasks[sid]
    assert mgr.merge_plan(sid) is not None
    (mgr.files_dir(sid) / "obj.rds").write_bytes(b"RDX3 another, longer R object")
    state = mgr.get_state(sid)
    assert state["converted"] == [] and "seurat_converted_at_commit" in _codes(state)
    assert mgr.merge_plan(sid) is None
    await mgr.commit(sid, dm)
    assert len(seen) == 2  # the commit converted the new file


async def test_a_removed_or_restaged_r_object_loses_its_conversion(mgr, monkeypatch):
    sid = _rds_session(mgr)
    _fake_converter(monkeypatch, [], mgr, sid)
    mgr.start_conversion(sid)
    await mgr._tasks[sid]
    rds = mgr.files_dir(sid) / "obj.rds"
    rds.unlink()
    assert mgr.get_state(sid)["converted"] == []
    rds.write_bytes(b"RDX3 an R object")  # the same bytes put back: a new file
    mtime = rds.stat().st_mtime_ns
    os.utime(rds, ns=(mtime + 10**9, mtime + 10**9))
    assert mgr.get_state(sid)["converted"] == []


async def test_convert_now_is_refused_while_one_runs(mgr, dm, monkeypatch):
    sid = _rds_session(mgr)
    gate = asyncio.Event()
    seen: list = []
    _fake_converter(monkeypatch, seen, mgr, sid, gate=gate)
    mgr.start_conversion(sid)
    while not seen:
        await asyncio.sleep(0)
    state = mgr.get_state(sid)
    assert state["conversion_running"] is True
    assert state["converting"]["file"] == "obj.rds" and state["converting"]["of"] == 1
    with pytest.raises(ConversionBusy):
        mgr.start_conversion(sid)
    with pytest.raises(ConversionBusy):
        await mgr.commit(sid, dm)
    gate.set()
    await mgr._tasks[sid]
    assert len(seen) == 1 and mgr.get_state(sid)["conversion_running"] is False


async def test_a_conversion_error_lands_in_the_state(mgr, monkeypatch):
    sid = _rds_session(mgr)
    seen: list = []
    _fake_converter(monkeypatch, seen, mgr, sid, fail="This file holds a list object.")
    mgr.start_conversion(sid)
    await mgr._tasks[sid]
    state = mgr.get_state(sid)
    assert state["conversion_error"] == "obj.rds could not be converted: This file holds a list object."
    assert state["conversion_running"] is False and state["converting"] is None
    assert state["converted"] == []
    _fake_converter(monkeypatch, seen, mgr, sid)
    mgr.start_conversion(sid)
    assert mgr.get_state(sid)["conversion_error"] is None  # cleared on a new try
    await mgr._tasks[sid]
    assert mgr.get_state(sid)["converted"] == [{"label": "obj", "file": "obj.rds"}]


async def test_convert_now_needs_an_r_object_and_valid_files(mgr):
    sid = mgr.create_session()
    _stage_h5ad(mgr, sid, "a.h5ad")
    with pytest.raises(IngestCommitError, match="no R object"):
        mgr.start_conversion(sid)
    bad = mgr.create_session()
    scipy.io.mmwrite(str(mgr.files_dir(bad) / "matrix.mtx"), sp.csr_matrix(np.eye(2)))
    (mgr.files_dir(bad) / "obj.rds").write_bytes(b"RDX3")
    with pytest.raises(IngestCommitError):
        mgr.start_conversion(bad)
    assert bad not in mgr._busy


async def test_discarding_stops_a_conversion(mgr, monkeypatch):
    sid = _rds_session(mgr)
    gate = asyncio.Event()
    seen: list = []
    _fake_converter(monkeypatch, seen, mgr, sid, gate=gate)
    mgr.start_conversion(sid)
    task = mgr._tasks[sid]
    while not seen:
        await asyncio.sleep(0)
    mgr.discard(sid)
    with pytest.raises(asyncio.CancelledError):
        await task
    assert sid not in mgr._busy and not mgr.exists(sid)


def test_lone_r_object_note_has_no_merge_suggestion(mgr):
    sid = mgr.create_session()
    (mgr.files_dir(sid) / "obj.rds").write_bytes(b"RDX3")
    note = next(i for i in mgr.get_state(sid)["validation"]["issues"]
                if i["code"] == "seurat_converted_at_commit")
    assert note["suggestion"] == ""


def test_a_stale_converting_record_is_not_shown(mgr):
    # A restart mid-conversion leaves "converting" in session.json: nothing runs any more.
    sid = _rds_session(mgr)
    mgr._set_converting(sid, {"label": "obj", "file": "obj.rds", "index": 1, "of": 1,
                              "progress_path": ""})
    assert mgr.get_state(sid)["converting"] is None


@pytest.fixture()
async def ingest_api(tmp_data_dir, client):
    from scview.dependencies import get_ingest_session_manager
    from scview.main import app

    mgr = IngestSessionManager(str(tmp_data_dir))
    app.dependency_overrides[get_ingest_session_manager] = lambda: mgr
    yield client, mgr
    app.dependency_overrides.pop(get_ingest_session_manager, None)


async def test_convert_endpoint(ingest_api, monkeypatch):
    client, mgr = ingest_api
    api = "/api/v1/ingest/session"
    assert (await client.post(f"{api}/{'0' * 32}/convert")).status_code == 404
    plain = mgr.create_session()
    _stage_h5ad(mgr, plain, "a.h5ad")
    r = await client.post(f"{api}/{plain}/convert")
    assert r.status_code == 400 and "no R object" in r.json()["detail"]

    sid = _rds_session(mgr)
    gate = asyncio.Event()
    seen: list = []
    _fake_converter(monkeypatch, seen, mgr, sid, gate=gate)
    r = await client.post(f"{api}/{sid}/convert")
    assert r.status_code == 202 and r.json()["conversion_running"] is True
    r = await client.post(f"{api}/{sid}/convert")
    assert r.status_code == 409
    assert (await client.post(f"{api}/{sid}/commit")).status_code == 409
    gate.set()
    await mgr._tasks[sid]
    state = (await client.get(f"{api}/{sid}")).json()
    assert state["converted"] == [{"label": "obj", "file": "obj.rds"}]
    plan = (await client.get(f"{api}/{sid}/merge-plan")).json()
    assert set(plan["samples"]) == {"a", "obj"}


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
