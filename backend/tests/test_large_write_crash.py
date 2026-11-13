"""A large dataset's in-place write is all or nothing (core/large_mode.py ``_Journaled``).

The h5ad is edited where it is (X is too big to copy), and HDF5 has no journal: a process
killed mid-write could leave a file that no longer opens. A child process is killed at each
stage of a write; the next open must find the file exactly as it was (same bytes), and a
write that finishes must leave no journal behind.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import anndata as ad
import h5py
import pytest

from scview.core import large_layout, large_mode
from scview.core.anndata_adaptor import AnnDataAdaptor
from tests.large_fixture import counts_adata, make_large

_CHILD = textwrap.dedent("""
    import os, signal, sys, threading
    import anndata as ad
    from scview.core import large_mode

    path, stage = sys.argv[1], sys.argv[2]
    adata = ad.read_h5ad(path, backed="r")
    adata.obs["edited"] = adata.obs["leiden"].astype(str) + "_new"
    adata.uns["scview_note"] = {"a": 2}

    def hook(at):
        if at == stage:
            os.kill(os.getpid(), signal.SIGKILL)  # as the OOM killer does: no cleanup

    large_mode._HOOK = hook
    large_mode.write_in_place(adata, path, threading.RLock(), obs=True, uns_keys=["scview_note"])
    print("finished")
""")


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_in_child(path: Path, stage: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}
    return subprocess.run([sys.executable, "-c", _CHILD, str(path), stage], env=env,
                          capture_output=True, text=True, timeout=120, cwd=os.getcwd())


@pytest.fixture(scope="module")
def counts():
    return counts_adata()


@pytest.mark.parametrize("stage", ["writing", "written", "journaled", "applying"])
def test_a_write_killed_at_any_stage_leaves_the_file_as_it_was(counts, tmp_path, stage):
    path = make_large(tmp_path / "ds", counts)
    before = _digest(path)
    size = path.stat().st_size
    res = _write_in_child(path, stage)
    assert res.returncode == -9, res.stderr[-2000:]  # killed, not finished
    assert large_mode.journal_path(path).exists()
    assert path.stat().st_size >= size  # what HDF5 wrote past the old end is still there
    # The next open rolls the write back: the same bytes as before.
    assert large_mode.file_is_large(path)
    assert not large_mode.journal_path(path).exists()
    assert _digest(path) == before
    a = AnnDataAdaptor(str(path))
    assert "edited" not in a.adata.obs and "scview_note" not in a.adata.uns
    assert large_layout.is_large(a.adata) and a.adata.X[:2].shape[0] == 2
    a.close()


def test_a_finished_write_leaves_no_journal_and_reads_back(counts, tmp_path):
    path = make_large(tmp_path / "ds", counts)
    with h5py.File(path, "r") as f:
        x_at = f["X"]["data"].id.get_offset()
    res = _write_in_child(path, "never")
    assert res.returncode == 0 and "finished" in res.stdout, res.stderr[-2000:]
    assert not large_mode.journal_path(path).exists()
    back = ad.read_h5ad(path, backed="r")
    assert list(back.obs["edited"])[:2] == [f"{v}_new" for v in list(back.obs["leiden"].astype(str))[:2]]
    assert back.uns["scview_note"]["a"] == 2
    back.file.close()
    with h5py.File(path, "r") as f:
        assert f["X"]["data"].id.get_offset() == x_at  # X not moved


def test_a_write_after_a_crash_rolls_back_first(counts, tmp_path):
    import threading

    path = make_large(tmp_path / "ds", counts)
    assert _write_in_child(path, "journaled").returncode == -9
    adata = ad.read_h5ad(path, backed="r")  # (h5py reads the file as it was: the overwrites never landed)
    adata.uns["scview_other"] = {"b": 1}
    large_mode.write_in_place(adata, path, threading.RLock(), obs=False, uns_keys=["scview_other"])
    assert not large_mode.journal_path(path).exists()
    back = ad.read_h5ad(path, backed="r")
    assert back.uns["scview_other"]["b"] == 1 and "scview_note" not in back.uns and "edited" not in back.obs
    back.file.close()
    adata.file.close()


def test_a_failed_write_is_rolled_back_in_process(counts, tmp_path, monkeypatch):
    import threading

    path = make_large(tmp_path / "ds", counts)
    before = _digest(path)
    adata = ad.read_h5ad(path, backed="r")
    adata.obs["edited"] = "x"

    def boom(stage):
        if stage == "applying":
            raise OSError("disk full")

    monkeypatch.setattr(large_mode, "_HOOK", boom)
    with pytest.raises(OSError, match="disk full"):
        large_mode.write_in_place(adata, path, threading.RLock())
    assert not large_mode.journal_path(path).exists() and _digest(path) == before
    adata.file.close()


def test_recovery_leaves_a_write_in_progress_alone(counts, tmp_path):
    path = make_large(tmp_path / "ds", counts)
    raw = large_mode._Journaled(path)  # holds the lock, intent journal written
    try:
        assert large_mode.journal_path(path).exists()
        assert large_mode.recover(path) is False  # someone is writing: not touched
        assert large_mode.journal_path(path).exists()
    finally:
        raw.close()
    assert not large_mode.journal_path(path).exists()


def test_the_journaled_file_reads_its_own_writes(tmp_path):
    p = tmp_path / "f.bin"
    p.write_bytes(bytes(range(256)) * 40)  # 10,240 bytes: three pages, the last one partial
    raw = large_mode._Journaled(p)
    try:
        raw.seek(4090)
        raw.write(b"ABCDEFGHIJ")  # across a page edge
        raw.seek(10_235)
        raw.write(b"0123456789")  # across the old end
        raw.seek(4088)
        assert raw.read(14) == bytes([248, 249]) + b"ABCDEFGHIJ" + bytes([4, 5])
        raw.seek(10_230)
        assert raw.read(20) == bytes(range(246, 251)) + b"0123456789"
        assert p.read_bytes()[4090:4100] == bytes(range(250, 256)) + bytes(range(4))  # not on disk yet
        raw.commit()
    finally:
        raw.close()
    data = p.read_bytes()
    assert data[4090:4100] == b"ABCDEFGHIJ" and data[10_235:] == b"0123456789" and len(data) == 10_245
