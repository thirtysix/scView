"""The offered rewrite of a column-stored dataset already in the library
(``core/store_by_rows.py``, ``api/v1/storage.py``): where it is offered, what it writes, what
it leaves alone, and what happens when the disk is short or the backend is killed.

Expected matrices are the source read back with anndata, never the module's own output.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import anndata as ad
import numpy as np
import pytest

import scview
from scview.core import csr_store, mito_refilter, provenance, replay_script, results_store, store_by_rows
from scview.core.anndata_adaptor import AnnDataAdaptor
from scview.core.dataset_manager import DatasetManager
from tests.test_csr_store import _assert_same_dataset, _dataset, _formats, _library_dataset, _random_csc

API = "/api/v1"
ALL = ["X", "layers/counts", "raw/X"]


def _events(text: str) -> list[tuple[str, dict]]:
    out, event = [], ""
    for line in text.splitlines():
        if line.startswith("event: "):
            event = line[7:]
        elif line.startswith("data: "):
            out.append((event, json.loads(line[6:])))
    return out


def _derived(data_dir: Path, did: str, a: ad.AnnData) -> Path:
    d = data_dir / "derived" / did
    d.mkdir(parents=True)
    a.write_h5ad(d / "old.h5ad")
    return d / "old.h5ad"


def _names(folder: Path) -> list[str]:
    return sorted(p.name for p in folder.iterdir())


# ---------------------------------------------------------------------------
# What it writes
# ---------------------------------------------------------------------------


def test_an_upload_gets_a_row_stored_copy_and_stays_as_sent(tmp_data_dir):
    dm = DatasetManager(data_dir=str(tmp_data_dir))
    upload = _library_dataset(tmp_data_dir, "d1", _dataset("csc"))
    sent = upload.read_bytes()
    assert store_by_rows.blocks_large_mode(upload) == ALL
    seen = []
    out = store_by_rows.run(dm, "d1", lambda done, total: seen.append((done, total)))
    copy = tmp_data_dir / "ingested" / "d1" / "old.h5ad"
    assert out["converted"] == ALL and out["files"] == [str(copy)] and out["active"] == str(copy)
    assert upload.read_bytes() == sent  # the user's file: never changed
    assert _names(copy.parent) == ["old.h5ad"]  # no unfinished file, no temporary folder
    assert not os.access(copy, os.W_OK) or os.geteuid() == 0  # a source: read-only
    assert set(_formats(copy).values()) == {"csr_matrix", "array"}
    _assert_same_dataset(ad.read_h5ad(upload), ad.read_h5ad(copy))
    assert seen[-1][0] == seen[-1][1] == csr_store.work_values(upload) > 0
    # "Reset from original" reads the row-stored copy from now on, not the column-stored upload.
    assert mito_refilter.source_file(dm, "d1") == str(copy)
    assert dm._resolve_h5ad("d1") == str(copy)
    # History shows it, as the last step, with what was converted.
    history = provenance.read_provenance(ad.read_h5ad(copy))["history"]
    assert [h["step"] for h in history] == ["store_by_rows"] and history[0]["params"] == {"matrices": ALL}
    meta = json.loads((tmp_data_dir / "uploads" / "d1" / "metadata.json").read_text())
    assert meta["stored_by_rows"]["matrices"] == ALL and meta["stored_by_rows"]["asked"] is True
    # A second call has nothing to do and writes nothing.
    stamp = copy.stat().st_mtime_ns
    assert store_by_rows.run(dm, "d1")["converted"] == [] and copy.stat().st_mtime_ns == stamp
    assert store_by_rows.plan(dm, "d1")["column_stored"] == []


def test_a_dataset_with_edits_keeps_them_and_both_files_are_row_stored(tmp_data_dir):
    dm = DatasetManager(data_dir=str(tmp_data_dir))
    upload = _library_dataset(tmp_data_dir, "d2", _dataset("csc"))
    edited = _dataset("csc")
    edited.obs["my_label"] = ["a" if i % 2 else "b" for i in range(edited.n_obs)]
    edited.uns["kept_result"] = {"value": np.arange(4)}
    provenance.record_step(edited, step="qc_metrics", tool="test")
    provenance.record_step(edited, step="clustering", tool="test")
    derived = _derived(tmp_data_dir, "d2", edited)
    before = results_store.fingerprint(AnnDataAdaptor(str(derived)))

    out = store_by_rows.run(dm, "d2")
    copy = tmp_data_dir / "ingested" / "d2" / "old.h5ad"
    assert out["files"] == [str(copy), str(derived)] and out["active"] == str(derived)
    assert csr_store.csc_in_file(copy) == [] and csr_store.csc_in_file(derived) == []
    got = ad.read_h5ad(derived)
    _assert_same_dataset(edited, got)
    assert list(got.obs["my_label"]) == list(edited.obs["my_label"])
    np.testing.assert_array_equal(got.uns["kept_result"]["value"], np.arange(4))
    assert [h["step"] for h in provenance.read_provenance(got)["history"]] == [
        "qc_metrics", "clustering", "store_by_rows"]
    # The original's stand-in carries no step of the dataset's history.
    assert provenance.read_provenance(ad.read_h5ad(copy))["history"] == []
    _assert_same_dataset(ad.read_h5ad(upload), ad.read_h5ad(copy))
    assert _names(derived.parent) == ["old.h5ad"] and _names(copy.parent) == ["old.h5ad"]
    # No value changed: a result stored before the rewrite is still current.
    after = results_store.fingerprint(AnnDataAdaptor(str(derived)))
    assert not results_store._is_stale(before, after)


def test_a_row_stored_or_dense_dataset_is_offered_nothing(tmp_data_dir):
    dm = DatasetManager(data_dir=str(tmp_data_dir))
    for did, fmt in (("rows", "csr"), ("dense", "dense")):
        path = _library_dataset(tmp_data_dir, did, _dataset(fmt))
        p = store_by_rows.plan(dm, did)
        assert p["column_stored"] == [] and p["blocks_large_mode"] == [] and p["files"] == 0 and "text" not in p
        assert store_by_rows.run(dm, did)["converted"] == []
        assert not (tmp_data_dir / "ingested" / did).exists() and _names(path.parent) == ["metadata.json", "old.h5ad"]


def test_a_failed_write_leaves_the_dataset_as_it_was(tmp_data_dir, monkeypatch):
    dm = DatasetManager(data_dir=str(tmp_data_dir))
    _library_dataset(tmp_data_dir, "d3", _dataset("csr"))
    derived = _derived(tmp_data_dir, "d3", _dataset("csc", seed=9))
    before = derived.read_bytes()

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(csr_store, "convert", boom)
    with pytest.raises(OSError, match="disk full"):
        store_by_rows.run(dm, "d3")
    assert derived.read_bytes() == before and _names(derived.parent) == ["old.h5ad"]
    monkeypatch.undo()
    assert store_by_rows.run(dm, "d3")["files"] == [str(derived)]  # the upload is row-stored: one file
    _assert_same_dataset(_dataset("csc", seed=9), ad.read_h5ad(derived))


def test_what_a_killed_rewrite_left_is_removed_at_startup_and_before_the_next_one(tmp_data_dir):
    dm = DatasetManager(data_dir=str(tmp_data_dir))
    upload = _library_dataset(tmp_data_dir, "d4", _dataset("csc"))
    derived = _derived(tmp_data_dir, "d4", _dataset("csc"))
    kept = derived.read_bytes()

    def litter():
        for folder in (derived.parent, tmp_data_dir / "ingested" / "d4"):
            folder.mkdir(parents=True, exist_ok=True)
            (folder / f".old.h5ad.0a1b2c3d{store_by_rows.PART}").write_bytes(b"half a file")
            (folder / ".csr-abc123").mkdir()
            (folder / ".csr-abc123" / "0.rows").write_bytes(b"1234")

    litter()
    # The dataset opens as before: the unfinished files are never taken for it.
    assert dm._resolve_h5ad("d4") == str(derived) and derived.read_bytes() == kept
    assert store_by_rows.sweep(tmp_data_dir) == 4
    assert _names(derived.parent) == ["old.h5ad"] and _names(tmp_data_dir / "ingested" / "d4") == []
    assert upload.exists() and derived.read_bytes() == kept
    litter()
    store_by_rows.run(dm, "d4")
    assert _names(derived.parent) == ["old.h5ad"] and _names(tmp_data_dir / "ingested" / "d4") == ["old.h5ad"]


def test_progress_counts_every_stored_value_twice(tmp_path):
    a = _dataset("csc")
    src, dst = tmp_path / "in.h5ad", tmp_path / "out.h5ad"
    a.write_h5ad(src)
    seen = []
    assert csr_store.rewrite_file(src, dst, progress=seen.append) == ALL
    assert sum(seen) == csr_store.work_values(src) == 2 * (a.X.nnz + a.layers["counts"].nnz + a.raw.X.nnz)
    assert [m["where"] for m in csr_store.csc_sizes(src)] == ALL and csr_store.csc_sizes(dst) == []


# ---------------------------------------------------------------------------
# Disk
# ---------------------------------------------------------------------------


def _free(monkeypatch, free: int) -> None:
    usage = shutil.disk_usage

    def fake(path):
        u = usage(path)
        return type(u)(u.total, u.total - free, free)

    monkeypatch.setattr(store_by_rows.shutil, "disk_usage", fake)


def test_the_disk_need_counts_each_new_file_and_the_temporary_files(tmp_data_dir, monkeypatch):
    dm = DatasetManager(data_dir=str(tmp_data_dir))
    upload = _library_dataset(tmp_data_dir, "d5", _dataset("csc"))
    one = store_by_rows.plan(dm, "d5")["disk_bytes"]
    assert one >= upload.stat().st_size  # a second copy, at least
    _derived(tmp_data_dir, "d5", _dataset("csc"))
    assert store_by_rows.plan(dm, "d5")["disk_bytes"] > 1.9 * one  # two files written
    # A matrix over one range of cells is sorted through temporary files: counted.
    monkeypatch.setattr(csr_store, "RANGE_BYTES", 1024)
    assert store_by_rows.plan(dm, "d5")["disk_bytes"] > 2 * one
    _free(monkeypatch, 5 * 2**30)
    with pytest.raises(store_by_rows.NoDiskSpace, match=r"about 6\.0 GB of disk and 5\.0 GB is free"):
        store_by_rows.check_disk(tmp_data_dir, 6 * 2**30)
    store_by_rows.check_disk(tmp_data_dir, 4 * 2**30)
    with pytest.raises(store_by_rows.NoDiskSpace):  # the floor left free
        store_by_rows.check_disk(tmp_data_dir, 5 * 2**30 - 1)


async def test_the_route_refuses_a_short_disk_with_both_numbers_and_writes_nothing(client, dataset_manager, monkeypatch):
    data = Path(dataset_manager.data_dir)
    upload = _library_dataset(data, "d6", _dataset("csc"))
    _free(monkeypatch, 100 * 2**20)
    info = (await client.get(f"{API}/datasets/d6/storage")).json()
    assert info["enough_disk"] is False and info["column_stored"] == ALL
    r = await client.post(f"{API}/datasets/d6/storage/by-rows")
    assert r.status_code == 507
    assert "GB of disk and 0.10 GB is free" in r.json()["detail"] and "second copy" in r.json()["detail"]
    assert not (data / "ingested" / "d6").exists() and csr_store.csc_in_file(upload) == ALL


async def test_an_upload_is_left_column_stored_when_the_disk_is_short(client, dataset_manager, monkeypatch, tmp_path):
    src = tmp_path / "sample.h5ad"
    _dataset("csc").write_h5ad(src)
    _free(monkeypatch, 100 * 2**20)
    with open(src, "rb") as fh:
        r = await client.post(f"{API}/datasets/upload", files={"file": ("sample.h5ad", fh, "application/octet-stream")})
    assert r.status_code == 201
    did = r.json()["id"]
    data = Path(dataset_manager.data_dir)
    assert not (data / "ingested" / did).exists()  # no copy started on a disk it does not fit
    meta = json.loads((data / "uploads" / did / "metadata.json").read_text())
    assert meta["stored_by_rows"]["matrices"] == [] and "GB is free" in meta["stored_by_rows"]["note"]
    assert (await client.get(f"{API}/datasets/{did}/storage")).json()["column_stored"] == ALL
    monkeypatch.undo()  # with room, an upload is stored by rows as before
    with open(src, "rb") as fh:
        r = await client.post(f"{API}/datasets/upload", files={"file": ("sample.h5ad", fh, "application/octet-stream")})
    assert (await client.get(f"{API}/datasets/{r.json()['id']}/storage")).json()["column_stored"] == []


# ---------------------------------------------------------------------------
# The routes, the refusal and the offer
# ---------------------------------------------------------------------------


def _counts_dataset(fmt: str, n: int = 300, g: int = 60) -> ad.AnnData:
    """Counts with groups, as an old upload: X and .raw column-stored (``fmt`` csc)."""
    import pandas as pd

    X = _random_csc(n, g, 0.4, np.float32, seed=3)
    X.data = np.rint(X.data * 5 + 1).astype(np.float32)
    X = {"csc": X.tocsc(), "csr": X.tocsr(), "dense": X.toarray()}[fmt]
    a = ad.AnnData(X, obs=pd.DataFrame({"group": pd.Categorical([f"g{i % 3}" for i in range(n)])},
                                       index=[f"cell{i}" for i in range(n)]),
                   var=pd.DataFrame(index=[f"gene{i}" for i in range(g)]))
    a.raw = a.copy()
    return a


async def test_the_rewrite_streams_its_progress_and_history_shows_the_step(client, dataset_manager):
    data = Path(dataset_manager.data_dir)
    upload = _library_dataset(data, "d7", _counts_dataset("csc"))
    before = await dataset_manager.get_or_load_dataset("d7")
    assert before.adata.X.format == "csc"
    info = (await client.get(f"{API}/datasets/d7/storage")).json()
    assert info["column_stored"] == info["blocks_large_mode"] == ["X", "raw/X"] and info["files"] == 1
    assert info["enough_disk"] and info["seconds"] >= 2 and info["memory_bytes"] > 0 and "GB of disk" in info["text"]

    r = await client.post(f"{API}/datasets/d7/storage/by-rows")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    events = _events(r.text)
    kinds = [e for e, _ in events]
    assert kinds[0] == "start" and kinds[-2:] == ["result", "complete"] and "error" not in kinds
    result = dict(events)["result"]
    assert result["converted"] == ["X", "raw/X"] and result["bytes_after"] > 0
    assert dataset_manager.get_dataset("d7") is None  # the cached column-stored copy is dropped
    after = await dataset_manager.get_or_load_dataset("d7")
    assert after.adata.X.format == "csr" and after.adata.raw.X.format == "csr"
    np.testing.assert_array_equal(ad.read_h5ad(upload).X.toarray(), after.adata.X.toarray())
    recipe = (await client.get(f"{API}/datasets/d7/recipe")).json()["recipe"]
    assert [s["step"] for s in recipe] == ["store_by_rows"] and recipe[0]["replay"]["status"] == "exact"
    # Asked again: nothing to do, said as a result with nothing converted.
    again = dict(_events((await client.post(f"{API}/datasets/d7/storage/by-rows")).text))
    assert again["result"]["converted"] == [] and (await client.get(f"{API}/datasets/d7/storage")).json()["files"] == 0
    assert (await client.get(f"{API}/datasets/nope/storage")).status_code == 404
    assert (await client.post(f"{API}/datasets/nope/storage/by-rows")).status_code == 404


async def test_a_large_mode_run_on_a_column_stored_dataset_names_the_rewrite_then_runs(client, dataset_manager):
    data = Path(dataset_manager.data_dir)
    _library_dataset(data, "d8", _counts_dataset("csc"))
    steps = ["qc_metrics", "normalization", "log_transform"]
    body = {"steps": steps, "params": {"chunked": "on"}, "allow_large_mode": True}
    r = await client.post(f"{API}/datasets/d8/assessment/run", json=body)
    assert r.status_code == 409
    detail = r.json()["detail"]
    assert store_by_rows.REFUSAL_PHRASE in detail and "stored by columns (X, raw/X)" in detail
    assert "store_by_rows" in detail and "/datasets/d8/storage/by-rows" in detail and "Nothing was run" in detail
    # Before the run, the preflight says the same with the rewrite's cost.
    pre = (await client.get(f"{API}/datasets/d8/preflight", params={"steps": "qc_metrics", "chunked": "on"})).json()
    assert pre["mode"]["by_columns"]["blocks_large_mode"] == ["X", "raw/X"] and pre["mode"]["by_columns"]["text"]
    # In memory nothing is blocked: no offer.
    pre = (await client.get(f"{API}/datasets/d8/preflight", params={"steps": "qc_metrics", "chunked": "off"})).json()
    assert "by_columns" not in pre["mode"]

    assert dict(_events((await client.post(f"{API}/datasets/d8/storage/by-rows")).text))["result"]["converted"]
    pre = (await client.get(f"{API}/datasets/d8/preflight", params={"steps": "qc_metrics", "chunked": "on"})).json()
    assert pre["mode"]["converts"] and "by_columns" not in pre["mode"]
    r = await client.post(f"{API}/datasets/d8/assessment/run", json=body)
    assert r.status_code == 200, r.text
    assert r.json()["steps_run"] == steps and r.json().get("status", "complete") == "complete"


@pytest.mark.parametrize("fmt", ["csr", "dense"])
async def test_a_row_stored_or_dense_dataset_gets_no_offer_before_a_large_mode_run(client, dataset_manager, fmt):
    _library_dataset(Path(dataset_manager.data_dir), "d9", _counts_dataset(fmt))
    pre = (await client.get(f"{API}/datasets/d9/preflight", params={"steps": "qc_metrics", "chunked": "on"})).json()
    assert pre["mode"]["converts"] and "by_columns" not in pre["mode"]
    assert (await client.get(f"{API}/datasets/d9/storage")).json()["column_stored"] == []


async def test_the_rewrite_waits_its_turn_and_is_refused_during_a_run(client, dataset_manager):
    data = Path(dataset_manager.data_dir)
    upload = _library_dataset(data, "d10", _counts_dataset("csc"))
    dataset_manager.mark_busy("d10")
    r = await client.post(f"{API}/datasets/d10/storage/by-rows")
    assert r.status_code == 409 and "run is working on this dataset" in r.json()["detail"]
    assert csr_store.csc_in_file(upload) and not (data / "ingested" / "d10").exists()
    dataset_manager.mark_idle("d10")


# ---------------------------------------------------------------------------
# Replay, and where it may be called from
# ---------------------------------------------------------------------------


def test_the_replay_script_runs_the_same_code_with_or_without_the_step():
    def code(a) -> list[str]:
        script = replay_script.build_script(a, dataset_id="t", dataset_name="test")
        return [ln for ln in script.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]

    a = _counts_dataset("csr")
    provenance.record_step(a, step="normalization", tool="scanpy.pp.normalize_total", params={"target_sum": 1e4})
    b = a.copy()
    provenance.record_step(b, step="store_by_rows", tool=store_by_rows.TOOL, params={"matrices": ["X"]})
    provenance.record_step(a, step="log_transform", tool="scanpy.pp.log1p")
    provenance.record_step(b, step="log_transform", tool="scanpy.pp.log1p")
    assert code(a) == code(b)
    assert replay_script.describe(b)[1]["status"] == "exact"  # the step itself: nothing to run


def test_nothing_but_the_asked_for_route_rewrites_a_dataset():
    """Never on open, in the background or in bulk: the one caller is the route the offer posts to."""
    src = Path(scview.__file__).parent
    callers = sorted(p.relative_to(src).as_posix() for p in src.rglob("*.py")
                     if "store_by_rows.run" in p.read_text())
    assert callers == ["api/v1/storage.py"]
    route = (src / "api/v1/storage.py").read_text()
    assert "dataset_edit(dataset_id)" in route and "keepalive(events())" in route
    main = (src / "main.py").read_text()  # the startup only sweeps
    assert "store_by_rows.sweep(" in main and "store_by_rows.run" not in main
