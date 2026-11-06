"""A read of a backed dataset while another request reads it into memory
(``AnnDataAdaptor.to_memory``, ``working_copy.reading``).

A run or an edit reads a backed dataset into memory (``working_copy.in_memory``), and
anndata's ``AnnData.to_memory`` closes the file it read from. A DE request that took the
backed object a moment before was still reading slices of it in a worker thread, and its
next read failed with h5py's "not a dataset" / invalid identifier errors. The read must
finish on the object it started with; the file is closed once the last such read ends.
"""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from scview.core import anndata_adaptor, working_copy
from scview.core.anndata_adaptor import AnnDataAdaptor


def _counts(n: int = 300, g: int = 60) -> ad.AnnData:
    rng = np.random.default_rng(0)
    X = sp.csr_matrix(rng.poisson(1.0, (n, g)).astype(np.float32))
    a = ad.AnnData(X)
    a.var_names = [f"G{i}" for i in range(g)]
    a.obs_names = [f"c{i}" for i in range(n)]
    a.obs["grp"] = pd.Categorical(np.where(np.arange(n) % 2 == 0, "a", "b"))
    a.raw = a.copy()
    return a


@pytest.fixture()
def backed(monkeypatch):
    monkeypatch.setattr(anndata_adaptor, "BACKED_MIN_BYTES", 0)


def _halves(adata) -> tuple[np.ndarray, np.ndarray]:
    first = np.zeros(adata.n_obs, dtype=bool)
    first[: adata.n_obs // 2] = True
    return first, ~first


def _reader(adata, first_done: threading.Event, go_on: threading.Event, out: dict):
    """Take the backed X and .raw's X, read half their rows, wait, then read the other
    half: as a DE test reads the two sides of a comparison from the matrix it took."""
    a, b = _halves(adata)

    def work() -> None:
        try:
            M, R = adata.X, adata.raw.X
            out["a"] = working_copy.read_rows(M, a)
            first_done.set()
            assert go_on.wait(10)
            out["b"] = working_copy.read_rows(M, b)
            out["raw"] = working_copy.read_rows(R, b)
        except Exception as e:  # noqa: BLE001 - checked by the test
            out["error"] = e
            first_done.set()

    return work


def test_a_reader_keeps_the_backed_file_through_a_switch_to_memory(tmp_path, backed):
    path = tmp_path / "d.h5ad"
    expected = _counts()
    expected.write_h5ad(path)
    adaptor = AnnDataAdaptor(str(path))
    adata = adaptor.adata
    assert adata.isbacked
    first_done, go_on, out = threading.Event(), threading.Event(), {}
    work = _reader(adata, first_done, go_on, out)

    def pinned() -> None:
        with adaptor.pinned():
            work()

    reader = threading.Thread(target=pinned)
    reader.start()
    assert first_done.wait(10)
    # The switch never waits for the reader: it returns while the reader is still paused.
    switch = threading.Thread(target=adaptor.to_memory)
    switch.start()
    switch.join(10)
    assert not switch.is_alive()
    assert not adaptor.adata.isbacked
    assert adata.file.is_open  # the reader still uses it
    go_on.set()
    reader.join(10)
    assert "error" not in out, repr(out.get("error"))
    _, b = _halves(expected)
    assert (out["b"] != expected.X[b]).nnz == 0 and (out["raw"] != expected.raw.X[b]).nnz == 0
    assert not adata.file.is_open  # closed once the last reader let go


def test_with_no_reader_the_switch_closes_the_file_at_once(tmp_path, backed):
    path = tmp_path / "d.h5ad"
    _counts().write_h5ad(path)
    adaptor = AnnDataAdaptor(str(path))
    adata = adaptor.adata
    with adaptor.pinned():
        pass
    mem = adaptor.to_memory()
    assert not mem.isbacked and not adata.file.is_open


def test_a_drop_while_a_reader_reads_closes_the_file_after_it(tmp_path, backed):
    # The manager dropping the adaptor (a run evicts its dataset) is the same switch.
    path = tmp_path / "d.h5ad"
    _counts().write_h5ad(path)
    adaptor = AnnDataAdaptor(str(path))
    adata = adaptor.adata
    unpin = adaptor.pin()
    adaptor.close()
    assert adata.file.is_open
    assert working_copy.read_rows(adata.X, _halves(adata)[0]).shape[0] == adata.n_obs // 2
    unpin()
    unpin()  # twice is harmless
    assert not adata.file.is_open


def _register(data_dir: Path, did: str = "x") -> str:
    d = data_dir / "uploads" / did
    d.mkdir(parents=True, exist_ok=True)
    (d / "metadata.json").write_text(json.dumps({"id": did, "name": did, "status": "ready"}))
    _counts().write_h5ad(d / "data.h5ad")
    return did


async def test_a_read_held_by_reading_survives_another_request_reading_the_dataset_in(
        dataset_manager, tmp_data_dir, backed):
    """The DE endpoints' path: ``working_copy.reading`` + ``held.run``, while an edit or
    a run reads the same cached adaptor into memory in another worker thread."""
    did = _register(Path(tmp_data_dir))
    adaptor = await dataset_manager.get_or_load_dataset(did)
    adata = adaptor.adata  # taken before the read, as the DE endpoints take it
    assert adata.isbacked
    first_done, go_on, out = threading.Event(), threading.Event(), {}
    work = _reader(adata, first_done, go_on, out)

    async def de() -> None:
        async with working_copy.reading(dataset_manager, did, 0, "A test") as held:
            await held.run(work)

    async def edit() -> None:
        await asyncio.to_thread(first_done.wait, 10)
        await asyncio.to_thread(working_copy.in_memory, adaptor)
        go_on.set()

    await asyncio.wait_for(asyncio.gather(de(), edit()), 30)
    assert "error" not in out, repr(out.get("error"))
    assert not adaptor.adata.isbacked and not adata.file.is_open
