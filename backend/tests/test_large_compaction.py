"""A large dataset's h5ad does not grow with each obs / uns edit (core/large_mode.py).

Edits are written into the file where it is. Each used to leave the table it replaced in
the file (HDF5 gives no space back): 476 -> 820 MB over 14 edits of a 116k-cell dataset.
Now only the changed columns are written, where the old ones were, and what still
accumulates (text) is copied away once it passes ``compaction_threshold``.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import textwrap
import threading
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest

from scview.core import large_layout, large_mode
from scview.core.anndata_adaptor import AnnDataAdaptor
from tests.large_fixture import counts_adata, make_large

N = 20_000


@pytest.fixture(scope="module")
def counts():
    rng = np.random.default_rng(1)
    a = counts_adata(n_cells=N, n_genes=60)
    for i in range(6):
        a.obs[f"label{i}"] = pd.Categorical(rng.choice([f"type{j}" for j in range(12)], N))
        a.obs[f"score{i}"] = rng.normal(size=N)
    return a


def _relabel(adata, column: str, name: str, rng) -> None:
    values = adata.obs[column].astype(str).to_numpy()
    values[rng.choice(len(values), 200, replace=False)] = name
    adata.obs[column] = pd.Categorical(values)


def test_twenty_label_edits_do_not_grow_the_file(counts, tmp_path):
    path = make_large(tmp_path / "ds", counts, gene_store_too=False)
    adata = ad.read_h5ad(path, backed="r")
    lock, rng = threading.RLock(), np.random.default_rng(2)
    start = path.stat().st_size
    with h5py.File(path, "r") as f:
        obs_bytes = sum(d.id.get_storage_size() for d in f["obs"].values() if isinstance(d, h5py.Dataset))
    sizes = []
    for e in range(20):
        _relabel(adata, "label0", f"edit{e}", rng)
        large_mode.write_in_place(adata, path, lock)
        sizes.append(path.stat().st_size)
    # the old code added the whole table each time: 20 x obs_bytes and more
    assert sizes[-1] - start < obs_bytes, (start, sizes, obs_bytes)
    back = ad.read_h5ad(path, backed="r")
    assert list(back.obs["label0"].astype(str)) == list(adata.obs["label0"].astype(str))
    assert list(back.obs.columns) == list(adata.obs.columns) and back.obs_names.equals(adata.obs_names)
    back.file.close()
    adata.file.close()


def test_an_embedding_written_again_takes_the_place_of_the_old_one(counts, tmp_path):
    # obsm / obsp writes of the other large-mode steps (a kept layout, WNN, saved scores)
    path = make_large(tmp_path / "ds", counts, gene_store_too=False)
    adata = ad.read_h5ad(path, backed="r")
    lock, rng = threading.RLock(), np.random.default_rng(4)
    sizes = []
    for _ in range(10):
        adata.obsm["X_mine"] = rng.normal(size=(N, 10)).astype(np.float32)
        large_mode.write_in_place(adata, path, lock, obs=False, obsm_keys=["X_mine"])
        sizes.append(path.stat().st_size)
    one = N * 10 * 4
    assert sizes[-1] - sizes[0] < one // 2, sizes  # was 9 x 0.8 MB: each old array stayed in the file
    back = ad.read_h5ad(path, backed="r")
    assert np.array_equal(back.obsm["X_mine"], adata.obsm["X_mine"])
    back.file.close()
    adata.file.close()


def test_only_the_changed_columns_are_written(counts, tmp_path):
    path = make_large(tmp_path / "ds", counts, gene_store_too=False)
    adata = ad.read_h5ad(path, backed="r")
    with h5py.File(path, "r") as f:
        before = {k: f["obs"][k].id.get_offset() for k in ("_index", "score1", "score2")}
    adata.obs["label1"] = adata.obs["label1"].cat.reorder_categories(
        list(adata.obs["label1"].cat.categories)[::-1])  # a new category order alone
    adata.obs["added"] = np.arange(N)
    del adata.obs["score3"]
    adata.obs["score2"] = adata.obs["score2"].to_numpy() + 1.0
    large_mode.write_in_place(adata, path, threading.RLock())
    with h5py.File(path, "r") as f:
        after = {k: f["obs"][k].id.get_offset() for k in ("_index", "score1")}
        assert "score3" not in f["obs"]
    assert after == {k: before[k] for k in after}  # untouched columns and the cell names stay put
    back = ad.read_h5ad(path, backed="r")
    pd.testing.assert_frame_equal(back.obs, adata.obs)
    assert list(back.obs["label1"].cat.categories) == list(adata.obs["label1"].cat.categories)
    back.file.close()
    adata.file.close()


def test_the_file_is_compacted_once_it_has_grown_past_the_threshold(counts, tmp_path, monkeypatch):
    monkeypatch.setattr(large_mode, "COMPACT_MIN_BYTES", 256 * 1024)
    path = make_large(tmp_path / "ds", counts, gene_store_too=False)
    adaptor = AnnDataAdaptor(str(path))
    adata = adaptor.adata
    x_rows = adata.X[:5].toarray()
    start = path.stat().st_size
    # the size without holes is a little above ``start`` once the entry exists
    limit = start + 2 * large_mode.compaction_threshold(start)
    rng = np.random.default_rng(3)
    names = np.array([f"gene_name_{i}" for i in range(4000)], dtype=object)
    sizes = []
    for e in range(40):  # text is never handed back: each of these leaves its old copy behind
        adata.uns["scview_names"] = {"names": rng.permutation(names)}
        adaptor.write_in_place(obs=False, uns_keys=["scview_names"])
        sizes.append(path.stat().st_size)
    assert sizes[4] - sizes[0] > 400_000  # it does grow, edit by edit (40 of them: 5 MB)
    assert max(sizes) <= limit, (start, limit, sizes)  # and never far past the threshold
    assert any(b < a for a, b in zip(sizes, sizes[1:])), sizes  # it was compacted at least once
    assert not list(path.parent.glob(".*.part")) and not large_mode.journal_path(path).exists()
    assert np.array_equal(adaptor.adata.X[:5].toarray(), x_rows)  # the open dataset reads the new file
    back = ad.read_h5ad(path, backed="r")
    assert list(back.uns["scview_names"]["names"]) == list(adata.uns["scview_names"]["names"])
    assert large_layout.is_large(back)
    back.file.close()
    with h5py.File(path, "r") as f:
        base = int(f.attrs[large_mode.COMPACT_SIZE_ATTR])
    assert large_mode.reclaimable(path) == path.stat().st_size - base
    adaptor.close()


def test_a_compacted_file_keeps_its_free_space_manager(counts, tmp_path):
    # a run's h5ad keeps its free-space manager in the file (large_run.H5AD_CREATE)
    from scview.core import h5_copy, large_run

    made = make_large(tmp_path / "ds", counts, gene_store_too=False)
    path = made.with_name("kept.h5ad")
    h5_copy.compact_copy(made, path, create=large_run.H5AD_CREATE)
    assert large_mode.compact(path, force=True)["compacted"]
    with h5py.File(path, "r") as f:
        strategy, persist, _ = f.id.get_create_plist().get_file_space_strategy()
    assert persist and strategy == h5py.h5f.FSPACE_STRATEGY_FSM_AGGR
    assert large_mode.compact(made, force=True)["compacted"]  # a file made without it stays without
    with h5py.File(made, "r") as f:
        assert not f.id.get_create_plist().get_file_space_strategy()[1]
    back = ad.read_h5ad(path, backed="r")
    assert large_layout.is_large(back) and back.n_obs == N
    back.file.close()


def test_no_compaction_without_the_disk_for_it(counts, tmp_path, monkeypatch):
    path = make_large(tmp_path / "ds", counts, gene_store_too=False)
    adata = ad.read_h5ad(path, backed="r")
    large_mode.write_in_place(adata, path, threading.RLock())  # stamps the size
    adata.file.close()
    before = hashlib.sha256(path.read_bytes()).hexdigest()

    class Usage:
        free = path.stat().st_size - 1

    monkeypatch.setattr(large_mode.shutil, "disk_usage", lambda _p: Usage)
    got = large_mode.compact(path, force=True)
    assert got["compacted"] is False and got["reason"] == "disk"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    assert not list(path.parent.glob(".*.part"))


_CHILD = textwrap.dedent("""
    import os, signal, sys
    from scview.core import large_mode

    def hook(at):
        if at == "compacting":
            os.kill(os.getpid(), signal.SIGKILL)  # as the OOM killer does: no cleanup

    large_mode._HOOK = hook
    large_mode.compact(sys.argv[1], force=True)
    print("finished")
""")


def test_a_compaction_killed_midway_leaves_the_dataset_as_it_was(counts, tmp_path):
    path = make_large(tmp_path / "ds", counts, gene_store_too=False)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}
    res = subprocess.run([sys.executable, "-c", _CHILD, str(path)], env=env, capture_output=True,
                         text=True, timeout=120, cwd=os.getcwd())
    assert res.returncode == -9, res.stderr[-2000:]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before  # the dataset's file: untouched
    stray = large_mode.stray_sidecars(path.parent)
    assert [p.name for p in stray] == [f".{Path(path).name}.compact.part"]  # the next prune removes it
    assert large_mode.compact(path, force=True)["compacted"]  # and so does the next compaction
    assert not list(path.parent.glob(".*.part"))
    a = AnnDataAdaptor(str(path))
    assert a.is_large and a.adata.X[:2].shape[0] == 2
    a.close()


def test_column_and_key_names_with_a_slash_edit_in_place(counts, tmp_path):
    # Seam of two v0.86 changes: columns written one by one (this file) and names holding
    # "/" stored escaped (core/h5_names.py). A column "T/NK" beside "T" must stay ONE key.
    path = make_large(tmp_path / "ds", counts.copy(), gene_store_too=False)
    adata = ad.read_h5ad(path, backed="r")
    lock, rng = threading.RLock(), np.random.default_rng(3)
    adata.obs["T"] = pd.Categorical(rng.choice(["a", "b"], N))
    adata.obs["T/NK"] = pd.Categorical(rng.choice(["c", "d/e"], N))
    adata.uns["T_colors"], adata.uns["T/NK_colors"] = ["#111111", "#222222"], ["#333333", "#444444"]
    large_mode.write_in_place(adata, path, lock, uns_keys=["T_colors", "T/NK_colors"])
    for e in range(3):  # edit the slash column, then drop it, each on top of the stored table
        _relabel(adata, "T/NK", f"x/{e}", rng)
        large_mode.write_in_place(adata, path, lock)
    with h5py.File(path, "r") as f:
        assert isinstance(f["obs"]["T"], h5py.Group) and "NK" not in f["obs"]["T"]  # not nested
        assert "T%2fNK" in f["obs"] and "T%2fNK" in list(f["obs"].attrs["column-order"])
    back = ad.read_h5ad(path, backed="r")
    assert list(back.obs.columns) == list(adata.obs.columns)
    assert list(back.obs["T/NK"].astype(str)) == list(adata.obs["T/NK"].astype(str))
    assert list(back.uns["T/NK_colors"]) == ["#333333", "#444444"] and list(back.uns["T_colors"]) == ["#111111", "#222222"]
    back.file.close()
    del adata.obs["T/NK"]
    large_mode.write_in_place(adata, path, lock)
    back = ad.read_h5ad(path, backed="r")
    assert "T/NK" not in back.obs and "T" in back.obs
    with h5py.File(path, "r") as f:
        assert "T%2fNK" not in f["obs"]
    back.file.close()
    adata.file.close()
