"""Learned gates on a large dataset (matrices on disk, core/large_mode.py): training reads
the training cells' rows of the counts file, applying one pass over it by row blocks; both
give the dataset in memory's values bitwise (``lt_lognorm`` works cell by cell), so the
gate, the check and the scores equal the in-memory ones, with every read within the cap."""

from __future__ import annotations

import hashlib
import json
import threading

import anndata as ad
import h5py
import numpy as np
import pytest
import scipy.sparse as sp

from scview.core import large_layout, large_mode
from scview.core import learned_gate as lg
from scview.core.label_transfer import lognorm_columns, lognorm_rows, lt_lognorm
from tests.large_fixture import make_large, register, small_twin
from tests.test_learned_gate import _cells
from tests.test_learned_gates_api import _a_ctrl, _events

API = "/api/v1"
LARGE, SMALL = "lgBig", "lgSmall"


def _same(a: sp.csr_matrix, b: sp.csr_matrix) -> None:
    a, b = sp.csr_matrix(a), sp.csr_matrix(b)
    assert a.shape == b.shape and a.dtype == b.dtype
    assert np.array_equal(a.indptr, b.indptr) and np.array_equal(a.indices, b.indices)
    assert np.array_equal(a.data, b.data)  # bitwise: the same float32 operations per value


@pytest.fixture()
def reads(monkeypatch):
    """Every guarded read of an on-disk matrix: the values it brought into memory."""
    seen: list[int] = []
    get = large_mode.OnDiskMatrix.__getitem__

    def spy(self, index):
        out = get(self, index)
        seen.append(int(sp.csr_matrix(out).nnz) if sp.issparse(out) or hasattr(out, "nnz") else 0)
        return out

    monkeypatch.setattr(large_mode.OnDiskMatrix, "__getitem__", spy)
    return seen


def _counts_on_disk(tmp_path, counts: ad.AnnData):
    h5ad = make_large(tmp_path / "big", counts, gene_store_too=False)
    f, M = large_mode.open_matrix(large_layout.sidecar_paths(h5ad)[0], large_layout.COUNTS_KEY,
                                  threading.RLock(), "the counts")
    return f, M


def test_rows_and_columns_equal_lt_lognorm_with_small_reads(tmp_path, monkeypatch, reads):
    counts = _cells()
    f, M = _counts_on_disk(tmp_path, counts)
    try:
        whole = lt_lognorm(counts.X)
        cap = 4_000 * 8  # about 4,000 values a read: a 600-cell sample spans many reads
        monkeypatch.setattr(large_mode, "MAX_READ_BYTES", cap)
        rows = np.sort(np.random.default_rng(1).choice(counts.n_obs, 600, replace=False))
        _same(lognorm_rows(M, rows), whole[rows])
        assert len(reads) > 5 and max(reads) * 8 <= cap
        reads.clear()
        cols = np.array([12, 3, 250, 101, 3 + 1])  # unsorted, as a gate's genes come
        _same(lognorm_columns(M, cols), whole[:, cols])
        assert len(reads) > 5 and max(reads) * 8 <= cap
        with pytest.raises(large_mode.LargeDatasetError):  # the whole matrix stays refused
            M.to_memory()
    finally:
        f.close()


def test_prepare_on_the_rows_alone_equals_the_whole_matrix(tmp_path, monkeypatch):
    counts = _cells()
    f, M = _counts_on_disk(tmp_path, counts)
    monkeypatch.setattr(large_mode, "MAX_READ_BYTES", 20_000 * 8)
    try:
        sel = _a_ctrl()
        whole = lg.prepare(counts, sel, expression=lambda a, rows: (lt_lognorm(a.X), a.var.copy(), "c"))
        asked: list[np.ndarray] = []

        def rows_only(a, rows):
            asked.append(rows)
            return lognorm_rows(M, rows), a.var.copy(), "c"

        part = lg.prepare(counts, sel, expression=rows_only)
    finally:
        f.close()
    assert np.array_equal(asked[0], whole.rows) and asked[0].size < counts.n_obs  # the training cells only
    assert np.array_equal(part.X, whole.X) and part.X.dtype == np.float32
    assert np.array_equal(part.y, whole.y) and np.array_equal(part.rows, whole.rows)
    assert part.genes == whole.genes
    for k in whole.stats:
        assert np.array_equal(part.stats[k], whole.stats[k])
    g1, _ = lg.train(whole, name="A")
    g2, _ = lg.train(part, name="A")
    assert g1.genes == g2.genes and g1.coef == g2.coef and g1.threshold == g2.threshold
    assert g1.cv == g2.cv


@pytest.fixture()
async def both(client, tmp_data_dir):
    """A large dataset and its in-memory twin (same cells, genes and counts)."""
    counts = _cells()
    h5ad = register(tmp_data_dir, LARGE, counts)
    up = tmp_data_dir / "uploads" / SMALL
    up.mkdir(parents=True)
    small_twin(counts).write_h5ad(up / "twin.h5ad")
    (up / "metadata.json").write_text(json.dumps({"id": SMALL, "name": "twin", "status": "ready"}))
    return h5ad


async def _train(client, did: str) -> dict:
    r = await client.post(f"{API}/datasets/{did}/learned-gates/train-stream",
                          json={"name": "A cells", "cells": _a_ctrl()})
    assert r.status_code == 200, r.text
    ev = _events(r.text)
    assert [e for e, _ in ev][-1] == "complete", ev[-1]
    return dict(ev)


def _x_digest(h5ad) -> str:
    h = hashlib.sha256()
    with h5py.File(h5ad, "r") as f:
        for k in ("data", "indices", "indptr"):
            h.update(np.asarray(f["X"][k][:]).tobytes())
    return h.hexdigest()


@pytest.mark.parametrize("small_reads", [False, True])
async def test_train_check_apply_equal_the_dataset_in_memory(client, both, dataset_manager, monkeypatch,
                                                             reads, small_reads):
    cap = 6_000 * 8 if small_reads else large_mode.MAX_READ_BYTES
    monkeypatch.setattr(large_mode, "MAX_READ_BYTES", cap)
    big, small = await _train(client, LARGE), await _train(client, SMALL)
    assert reads and max(reads) * 8 <= cap
    assert len(reads) > (5 if small_reads else 0)
    s_big, s_small = big["start"], small["start"]
    assert s_big.pop("source") == "counts (the counts file), log-normalised here"
    assert s_small.pop("source") == "counts (layers['counts']), log-normalised here"
    assert s_big == s_small
    gb, gs = big["result"]["gate"], small["result"]["gate"]
    for k in ("genes", "coef", "intercept", "mean", "std", "threshold", "cv", "n_train"):
        if k in gs:
            assert gb[k] == gs[k], k  # the same training matrix, so the same fit
    rb, rs = big["result"]["report"], small["result"]["report"]
    rb.pop("expression"), rs.pop("expression")
    assert rb == rs

    gid = gs["id"]
    assert (await client.post(f"{API}/learned-gates/{gid}/save")).status_code == 200
    before = _x_digest(both)
    reads.clear()
    cb = (await client.post(f"{API}/datasets/{LARGE}/learned-gates/{gid}/check")).json()
    cs = (await client.post(f"{API}/datasets/{SMALL}/learned-gates/{gid}/check")).json()
    assert not reads  # a check reads no values
    large = cb.pop("large")
    assert large["seconds"] >= 0 and large["peak_bytes"] > 0 and large["note"].startswith("Large dataset")
    assert cb == cs and cb["ok"]

    r = await client.post(f"{API}/datasets/{LARGE}/learned-gates/{gid}/apply", json={})
    assert r.status_code == 200, r.text
    ab = r.json()
    assert max(reads) * 8 <= cap and len(reads) > (5 if small_reads else 0)
    as_ = (await client.post(f"{API}/datasets/{SMALL}/learned-gates/{gid}/apply", json={})).json()
    assert ab.pop("expression") == "counts (the counts file), log-normalised here"
    as_.pop("expression")
    assert (ab.pop("same_dataset"), as_.pop("same_dataset")) == (False, True)  # the gate was trained on SMALL
    assert ab == as_
    col, score = ab["column"], ab["score_column"]
    obs_small = (await dataset_manager.get_or_load_dataset(SMALL)).adata.obs
    # written into the large h5ad in place, X untouched
    assert _x_digest(both) == before
    on_disk = ad.read_h5ad(both, backed="r")
    try:
        assert np.array_equal(on_disk.obs[score].to_numpy(), obs_small[score].to_numpy())
        assert (on_disk.obs[col].astype(str) == obs_small[col].astype(str).to_numpy()).all()
        assert lg.UNS_KEY in on_disk.uns
    finally:
        on_disk.file.close()
    obs_big = (await dataset_manager.get_or_load_dataset(LARGE)).adata.obs
    assert np.array_equal(obs_big[score].to_numpy(), obs_small[score].to_numpy())


async def test_apply_refusal_comes_before_the_counts_pass(client, both, dataset_manager, reads):
    small = await _train(client, SMALL)
    gid = small["result"]["gate"]["id"]
    assert (await client.post(f"{API}/learned-gates/{gid}/save")).status_code == 200
    adaptor = await dataset_manager.get_or_load_dataset(LARGE)
    adaptor.adata.obs["gate_a_cells"] = "x"  # a column no learned gate wrote
    reads.clear()
    r = await client.post(f"{API}/datasets/{LARGE}/learned-gates/{gid}/apply", json={})
    assert r.status_code == 409 and "already has a column" in r.json()["detail"]
    assert not reads  # refused before any value was read


def test_estimate_scales_with_the_counts(tmp_path):
    counts = _cells()
    f, M = _counts_on_disk(tmp_path, counts)
    try:
        apply, train = lg.large_estimate(M), lg.large_estimate(M, train=True)
    finally:
        f.close()
    nnz = counts.X.nnz
    assert apply["seconds"] == round(nnz * lg.LARGE_SECONDS_PER_VALUE, 1)
    assert apply["peak_bytes"] >= counts.n_obs * lg.MAX_GENES * lg.LARGE_APPLY_BYTES_PER_VALUE
    assert "960 cells" in apply["note"] and "960 cells" in train["note"]
