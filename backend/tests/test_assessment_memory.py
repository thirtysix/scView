"""Data Assessment's requests held in the memory budget (core/assessment_memory.py,
core/assessment_gate.py ``Budget``), and the peaks cut where it was cheap.

At a 3 GB limit, a deep 1.6 GB h5ad (23,578 x 34,209 CSC counts from the R converter) got the
backend killed within 3 s of opening Data Assessment: the ambient RNA card (+1.5 GB, a CSR copy
of the counts), cluster QC (its ambient flags, the same copy) and the rest ran at once, none
of them in the budget. Now each is estimated, reserved and refused with a 507 when it cannot
fit, requests fired together queue behind each other's reservations, and the copies are gone.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
import tracemalloc
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from scview.api.v1 import assessment as assessment_api
from scview.core import (
    ambient, anndata_adaptor, assessment_gate, assessment_memory, memory_budget, normalization,
    sample_pseudobulk, sparse_sums,
)
from scview.core.dataset_manager import DatasetManager, DatasetMemoryError

API = "/api/v1"
GB = 1024**3


def _deep(n: int = 600, g: int = 300, seed: int = 0, fmt: str = "csc") -> ad.AnnData:
    """Counts without QC columns in obs (the per-cell pass runs), MT-/RPL genes, two samples x
    two conditions, three cell types."""
    rng = np.random.default_rng(seed)
    lam = rng.gamma(0.4, 2.0, g) * rng.gamma(4.0, 0.5, (n, 1))
    X = rng.poisson(lam).astype(np.float64)
    names = [f"G{i}" for i in range(g)]
    for j, name in enumerate(["MT-CO1", "MT-ND1", "RPL3", "RPS6", "MALAT1"]):
        names[j] = name
    a = ad.AnnData(sp.csc_matrix(X) if fmt == "csc" else sp.csr_matrix(X) if fmt == "csr" else X)
    a.var_names = names
    a.obs_names = [f"c{i}" for i in range(n)]
    a.obs["sample"] = pd.Categorical(np.where(np.arange(n) % 4 < 2, "DonorA", "DonorB"))
    a.obs["condition"] = pd.Categorical(np.where(np.arange(n) % 2 == 0, "ctrl", "stim"))
    a.obs["leiden"] = pd.Categorical((np.arange(n) % 3).astype(str))
    a.obs["cell_type"] = pd.Categorical(np.array(["LineA", "LineB", "LineC"])[np.arange(n) % 3])
    return a


# --- peaks cut: same answers, no whole copies ----------------------------------------------


def test_csc_per_cell_sums_equal_the_row_blocks():
    a = _deep()
    M = a.X
    masks = [np.asarray(a.var_names.str.startswith("MT-")), np.zeros(a.n_vars, bool),
             np.asarray(a.var_names.str.startswith(("RPL", "RPS")))]
    want = anndata_adaptor._row_sums(sp.csr_matrix(M), masks)
    got = anndata_adaptor._csc_row_sums(M, masks)
    np.testing.assert_allclose(got[0], want[0])
    np.testing.assert_array_equal(got[1], want[1])
    for x, y in zip(got[2], want[2]):
        np.testing.assert_allclose(x, y)
    # Spans smaller than a column: still one pass, the same totals.
    old = anndata_adaptor.ROW_BLOCK_VALUES
    try:
        anndata_adaptor.ROW_BLOCK_VALUES = 7
        again = anndata_adaptor._csc_row_sums(M, masks)
    finally:
        anndata_adaptor.ROW_BLOCK_VALUES = old
    np.testing.assert_allclose(again[0], want[0])
    np.testing.assert_array_equal(again[1], want[1])


@pytest.mark.parametrize("fmt", ["csc", "csr", "dense"])
def test_group_sums_and_detected_genes_in_every_layout(fmt):
    a = _deep(fmt=fmt)
    X = a.X.toarray() if sp.issparse(a.X) else a.X
    codes = np.arange(a.n_obs) % 3
    codes[::7] = -1  # in no group
    want = np.stack([X[codes == k].sum(axis=0) for k in range(3)])
    np.testing.assert_allclose(sparse_sums.group_sums(a.X, codes, 3), want)
    np.testing.assert_array_equal(sparse_sums.detected_per_row(a.X), (X > 0).sum(axis=1))


def test_csc_group_sums_make_no_csr_copy():
    a = _deep(n=4000, g=800)
    codes = np.arange(a.n_obs) % 4
    whole = a.X.data.nbytes + a.X.indices.nbytes
    tracemalloc.start()
    try:
        sparse_sums.group_sums(a.X, codes, 4)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 0.25 * whole, (peak, whole)  # the old CSR conversion: a whole copy


def test_ambient_and_pseudobulk_answer_the_same_on_csc_and_csr():
    csc, csr = _deep(fmt="csc"), _deep(fmt="csr")
    a, b = ambient.diagnose(csc, "leiden"), ambient.diagnose(csr, "leiden")
    assert a["samples"] == b["samples"] and a["markers"] == b["markers"]
    p = sample_pseudobulk.pseudobulk_qc(csc, "sample", "condition")
    q = sample_pseudobulk.pseudobulk_qc(csr, "sample", "condition")
    assert json.dumps(p, sort_keys=True, default=str) == json.dumps(q, sort_keys=True, default=str)


def _mv_reference(X: np.ndarray, rows: np.ndarray, clip: float) -> dict:
    """The diagnostic's numbers worked out whole, densely, as the old code did."""
    C = X[rows].astype(np.float64)
    mean, var = C.mean(axis=0), C.var(axis=0)
    keep = mean > 0
    C = C[:, keep]
    tot = C.sum(axis=1)
    L = np.log1p(C * np.divide(1e4, tot, out=np.zeros_like(tot), where=tot > 0)[:, None])
    genes = C.sum(axis=0)
    mu = np.outer(tot, genes / genes.sum())
    with np.errstate(divide="ignore", invalid="ignore"):
        R = (C - mu) / np.sqrt(mu + mu**2 / normalization.THETA)
    R[~np.isfinite(R)] = 0
    R = np.clip(R, -clip, clip)
    return {"mean": mean[keep], "var_counts": var[keep], "var_log": L.var(axis=0), "var_residuals": R.var(axis=0)}


@pytest.mark.parametrize("fmt", ["csc", "csr", "dense"])
@pytest.mark.parametrize("max_cells", [10_000, 250])
def test_mean_variance_in_blocks_matches_the_whole(fmt, max_cells, monkeypatch):
    monkeypatch.setattr(normalization, "MV_BLOCK_VALUES", 5_000)  # many blocks each way
    monkeypatch.setattr(normalization, "MV_RESIDUAL_VALUES", 9_000)
    a = _deep(fmt=fmt)
    X = a.X.toarray() if sp.issparse(a.X) else a.X
    X[:, 10] = 0  # an undetected gene is left out
    M = sp.csc_matrix(X) if fmt == "csc" else sp.csr_matrix(X) if fmt == "csr" else X
    mv = normalization.mean_variance(M, a.var_names, max_cells=max_cells)
    rows = np.arange(a.n_obs)
    if a.n_obs > max_cells:
        rows = np.sort(np.random.default_rng(0).choice(a.n_obs, max_cells, replace=False))
    ref = _mv_reference(X, rows, normalization.default_clip(rows.size))
    assert mv["n_cells_used"] == rows.size and not mv["keep"][10]
    assert len(mv["genes"]) == int(mv["keep"].sum()) == ref["mean"].size
    for k, v in ref.items():
        np.testing.assert_allclose(mv[k], v, rtol=1e-9, atol=1e-12)


def test_mean_variance_holds_blocks_not_the_sample(monkeypatch):
    monkeypatch.setattr(normalization, "MV_BLOCK_VALUES", 300_000)
    monkeypatch.setattr(normalization, "MV_RESIDUAL_VALUES", 200_000)
    a = _deep(n=6000, g=1500)
    whole = a.X.data.nbytes + a.X.indices.nbytes
    tracemalloc.start()
    try:
        normalization.mean_variance(a.X, a.var_names)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < whole, (peak, whole)  # it held several whole copies (+5.1 GB on the deep file)
    assert peak <= assessment_memory.mean_variance_need(a)


# --- the estimates ----------------------------------------------------------------------------


class _Adaptor:
    def __init__(self, adata):
        self.adata = adata
        self._qc_cells = None

    qc_cached = anndata_adaptor.AnnDataAdaptor.qc_cached
    qc_per_cell = anndata_adaptor.AnnDataAdaptor.qc_per_cell
    is_large = False
    _qc_lock = threading.Lock()


def test_per_cell_need_is_zero_once_cached_or_held_in_obs():
    a = _deep()
    ad_ = _Adaptor(a)
    assert assessment_memory.per_cell_need(ad_) > 0
    ad_.qc_per_cell()
    assert assessment_memory.per_cell_need(ad_) == 0
    assert assessment_memory.per_cell_need(ad_, ribo=True) == 0  # the pass computed the extras
    b = _deep()
    b.obs["nCount_RNA"], b.obs["nFeature_RNA"], b.obs["percent.mt"] = 1.0, 1.0, 1.0
    assert assessment_memory.per_cell_need(_Adaptor(b)) == 0
    assert assessment_memory.per_cell_need(_Adaptor(b), ribo=True) > 0  # % ribo etc. still read


def test_ambient_need_is_zero_once_cached():
    a = _deep()
    assert assessment_memory.ambient_need(a, "leiden") > 0
    ambient.diagnose_cached(a, "leiden")
    assert assessment_memory.ambient_need(a, "leiden") == 0


def _peak(fn) -> int:
    tracemalloc.start()
    try:
        fn()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return peak


def test_estimates_are_at_or_above_what_the_work_allocates():
    a = _deep(n=3000, g=900)
    assert _peak(lambda: _Adaptor(a).qc_per_cell(ribo=True)) <= assessment_memory.per_cell_need(_Adaptor(a), True)
    assert _peak(lambda: ambient.diagnose(a, "leiden")) <= assessment_memory.ambient_need(a, "leiden")
    need = assessment_memory.pseudobulk_need(_Adaptor(a), "", "")
    assert _peak(lambda: sample_pseudobulk.pseudobulk_qc(a, "sample", "condition")) <= need


# --- the budget: queued, refused, never overshooting -----------------------------------------


def _budget(monkeypatch, free_gb: float, limit_gb: float = 8.0) -> None:
    def read(*_a, **_k):
        return memory_budget.MemoryBudget(int(limit_gb * GB), int((limit_gb - free_gb) * GB), True,
                                          "container limit")

    monkeypatch.setattr(memory_budget, "read_budget", read)
    monkeypatch.setattr(memory_budget, "release_memory", lambda: None)
    monkeypatch.setattr("scview.core.dataset_manager.REFUSAL_WAIT_SECONDS", 1)


async def test_two_requests_at_once_cannot_both_take_the_room(tmp_data_dir, monkeypatch):
    """Each fits alone, both together do not: the second waits for the first's reservation to
    end, then runs; they never run (and allocate) at the same time."""
    _budget(monkeypatch, free_gb=1.5)
    dm = DatasetManager(data_dir=str(tmp_data_dir))
    monkeypatch.setattr(dm, "dataset_name", lambda _i: "x name")
    running, most, lock = [], [0], threading.Lock()

    def heavy(name):
        with lock:
            running.append(name)
            most[0] = max(most[0], len(running))
        time.sleep(0.6)
        with lock:
            running.remove(name)
        return name

    budget = assessment_gate.Budget(dm, lambda: GB, "Cluster QC")
    t0 = time.monotonic()
    out = await asyncio.gather(*(assessment_gate.run(f"d{i}", ("k",), heavy, f"r{i}", budget=budget)
                                 for i in range(2)))
    assert sorted(out) == ["r0", "r1"] and most[0] == 1
    assert time.monotonic() - t0 >= 1.1  # one after the other
    assert dm._reads == {}


async def test_shared_requests_share_one_reservation(tmp_data_dir, monkeypatch):
    _budget(monkeypatch, free_gb=1.5)
    dm = DatasetManager(data_dir=str(tmp_data_dir))
    held = []
    real = dm.hold_read

    def spy(ids, need, label=""):
        held.append(need)
        return real(ids, need, label)

    monkeypatch.setattr(dm, "hold_read", spy)
    budget = assessment_gate.Budget(dm, lambda: GB, "QC metrics")
    out = await asyncio.gather(*(assessment_gate.run("d", ("qc",), lambda: (time.sleep(0.3), 7)[1], budget=budget)
                                 for _ in range(3)))
    assert out == [7, 7, 7] and [n for n in held if n] == [GB]


async def test_a_request_that_cannot_fit_is_refused_and_never_runs(tmp_data_dir, monkeypatch):
    _budget(monkeypatch, free_gb=0.5)
    dm = DatasetManager(data_dir=str(tmp_data_dir))
    monkeypatch.setattr(dm, "dataset_name", lambda _i: "x name")

    def never():
        raise AssertionError("it ran")

    with pytest.raises(DatasetMemoryError) as e:
        await assessment_gate.run("d", ("k",), never, budget=assessment_gate.Budget(dm, lambda: 2 * GB, "Cluster QC"))
    text = str(e.value)
    assert text.startswith("Cluster QC on x name needs about 2.0 GB") and "GB limit" in text


async def test_a_queued_request_is_refused_when_the_wait_runs_out(tmp_data_dir, monkeypatch):
    _budget(monkeypatch, free_gb=1.5)
    monkeypatch.setattr(assessment_gate, "QUEUE_SECONDS", 1)
    dm = DatasetManager(data_dir=str(tmp_data_dir))
    monkeypatch.setattr(dm, "dataset_name", lambda _i: "x name")
    release = threading.Event()
    budget = assessment_gate.Budget(dm, lambda: GB, "QC states")
    first = asyncio.ensure_future(assessment_gate.run("a", ("k",), release.wait, 10, budget=budget))
    await asyncio.sleep(0.2)
    with pytest.raises(DatasetMemoryError) as e:
        await assessment_gate.run("b", ("k",), lambda: 1, budget=budget)
    assert "wait for it to finish" in str(e.value)
    release.set()
    assert await first is True


# --- every route through the budget ------------------------------------------------------------


def _register(data_dir: Path, did: str, adata: ad.AnnData) -> None:
    d = data_dir / "uploads" / did
    d.mkdir(parents=True, exist_ok=True)
    (d / "metadata.json").write_text(json.dumps({"id": did, "name": f"{did} name", "status": "ready"}))
    adata.write_h5ad(d / "data.h5ad")


def _query(route: str) -> str:
    return {"/datasets/{dataset_id}/qc/sex": "?column=sample",
            "/datasets/{dataset_id}/qc/libraries": "?sample_column=sample",
            "/datasets/{dataset_id}/qc/pseudobulk": "?sample_column=sample"}.get(route, "")


def test_every_assessment_get_route_is_named():
    """A new GET route in api/v1/assessment.py must be held in the budget (ROUTES) or say how
    it is held (HELD_OTHERWISE)."""
    paths = {r.path for r in assessment_api.router.routes if "GET" in getattr(r, "methods", ())}
    known = set(assessment_memory.ROUTES) | set(assessment_memory.HELD_OTHERWISE)
    assert paths - known == set(), paths - known


async def test_every_route_goes_through_the_budget(client, dataset_manager, tmp_data_dir, monkeypatch):
    _register(Path(tmp_data_dir), "x", _deep())
    await dataset_manager.get_or_load_dataset("x")
    _budget(monkeypatch, free_gb=6.0)
    seen: list[str] = []
    real = assessment_gate._held

    async def spy(dataset_id, budget, fn, *args):
        if budget is not None:
            seen.append(budget.what)
        return await real(dataset_id, budget, fn, *args)

    monkeypatch.setattr(assessment_gate, "_held", spy)
    for route, label in assessment_memory.ROUTES.items():
        seen.clear()
        r = await client.get(API + route.format(dataset_id="x") + _query(route))
        assert r.status_code == 200, (route, r.text)
        assert label in seen, (route, seen)
    assert dataset_manager._reads == {}


async def test_every_route_says_507_when_it_cannot_fit(client, dataset_manager, tmp_data_dir, monkeypatch):
    _register(Path(tmp_data_dir), "x", _deep())
    await dataset_manager.get_or_load_dataset("x")
    _budget(monkeypatch, free_gb=0.5)
    monkeypatch.setattr(assessment_memory, "FIXED", 2 * GB)  # every estimate past what is free
    free_of_work = {"/datasets/{dataset_id}/qc/doublets"}  # obs and uns only: held at 0
    for route, label in assessment_memory.ROUTES.items():
        r = await client.get(API + route.format(dataset_id="x") + _query(route))
        if route in free_of_work:
            assert r.status_code == 200, (route, r.text)
            continue
        assert r.status_code == 507, (route, r.status_code, r.text)
        assert r.json()["detail"].startswith(f"{label} on x name needs about"), (route, r.text)
    assert dataset_manager._reads == {}
