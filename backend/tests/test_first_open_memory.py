"""A first open: one preprocessing assessment per dataset at a time, its samples read in
blocks, and the page's on-demand work (the assessment, the insight, a column's markers) held
in the memory budget, a 507 when it cannot fit.

A 1.6 GB h5ad from convert.R (23,578 x 34,209 CSC counts, deep cell lines) killed a 6 GB
backend on its first open: marker genes for its author labels took +4.5 GB (the shifted
log held three copies of the counts) while the co-pilot's insight assessed it.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from scview.api.v1 import markers as markers_api
from scview.core import assessor, memory_budget, preflight

API = "/api/v1"
GB = 1024**3


def _counts(n: int = 120, g: int = 50, seed: int = 0) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    a = ad.AnnData(sp.csr_matrix(rng.poisson(1.2, (n, g)).astype(np.float32)))
    a.var_names = [f"G{i}" for i in range(g)]
    a.obs_names = [f"c{i}" for i in range(n)]
    a.obs["Celltype"] = pd.Categorical(np.where(np.arange(n) % 3 == 0, "A", "B"))
    return a


# --- one assessment at a time ---------------------------------------------------------


def _slow(monkeypatch, calls: list, pause: float = 0.3):
    real = assessor._assess_now

    def slow(adata):
        calls.append(threading.get_ident())
        time.sleep(pause)
        return real(adata)

    monkeypatch.setattr(assessor, "_assess_now", slow)


def _together(fn, n: int = 3) -> list:
    out: list = [None] * n
    errors: list = []

    def work(i):
        try:
            out[i] = fn()
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    return out + errors


def test_calls_at_once_share_one_assessment(monkeypatch):
    a = _counts()
    calls: list = []
    _slow(monkeypatch, calls)
    states = _together(lambda: assessor.assess_preprocessing(a))
    assert len(calls) == 1
    assert len(states) == 3 and all(isinstance(s, assessor.PreprocessingState) for s in states)
    assert states[0] == states[1] == states[2]
    assert states[0] is not states[1]  # each caller has its own copy
    assert not assessor.assessing(a)


def test_a_later_call_assesses_again(monkeypatch):
    """Nothing is kept: the assessment reads obs values, which change in place."""
    a = _counts()
    calls: list = []
    _slow(monkeypatch, calls, pause=0.0)
    assessor.assess_preprocessing(a)
    a.obs["doublet_id"] = ["Doublet"] * 3 + ["Singlet"] * (a.n_obs - 3)
    d = assessor.assess_preprocessing(a).doublet_detection
    assert len(calls) == 2 and "(3 of 120 cells called doublets)" in d.details


def test_other_datasets_are_not_held_up(monkeypatch):
    a, b = _counts(), _counts(seed=1)
    calls: list = []
    _slow(monkeypatch, calls, pause=0.5)
    pick = iter([a, b])
    lock = threading.Lock()

    def one():
        with lock:
            d = next(pick)
        return assessor.assess_preprocessing(d)

    t0 = time.monotonic()
    out = _together(one, n=2)
    assert len(calls) == 2 and all(isinstance(s, assessor.PreprocessingState) for s in out)
    assert time.monotonic() - t0 < 0.95  # side by side, not one after the other


def test_changed_data_waits_then_assesses_its_own(monkeypatch):
    a = _counts()
    calls: list = []
    _slow(monkeypatch, calls)
    first: list = []
    t = threading.Thread(target=lambda: first.append(assessor.assess_preprocessing(a)))
    t.start()
    time.sleep(0.1)
    a.obs["new_column"] = 1.0  # what the running one read has changed
    second = assessor.assess_preprocessing(a)
    t.join(5)
    assert len(calls) == 2 and first and second is not None


def test_an_error_reaches_every_caller(monkeypatch):
    a = _counts()

    def broken(_adata):
        time.sleep(0.2)
        raise RuntimeError("boom")

    monkeypatch.setattr(assessor, "_assess_now", broken)
    out = _together(lambda: assessor.assess_preprocessing(a))
    assert sum(isinstance(x, RuntimeError) for x in out) == 3
    monkeypatch.undo()
    assert assessor.assess_preprocessing(a).normalization is not None  # not stuck


def test_need_counts_the_sample_blocks_and_nothing_while_one_runs(monkeypatch):
    a = _counts(n=600, g=300)
    per_cell = a.X.nnz / a.n_obs
    want = int(assessor.ASSESS_BLOCK_COPIES * 250 * per_cell * 16 + 200 * 300 * 8 * 3)
    assert assessor.assessment_need(a) == want
    seen: list = []
    real = assessor._assess_now

    def peek(adata):
        seen.append(assessor.assessment_need(adata))
        return real(adata)

    monkeypatch.setattr(assessor, "_assess_now", peek)
    assessor.assess_preprocessing(a)
    assert seen == [0]


# --- the samples, in blocks --------------------------------------------------------------


@pytest.mark.parametrize("form", ["csr", "csc", "dense"])
def test_sample_max_is_nanmax_of_the_dense_sample(form):
    rng = np.random.default_rng(3)
    dense = rng.poisson(0.5, (700, 40)).astype(np.float64)
    dense[3, 4] = np.nan
    M = {"csr": sp.csr_matrix(dense), "csc": sp.csc_matrix(dense), "dense": dense}[form]
    for n in (1, 250, 251, 600):
        assert assessor._sample_max(M, n) == float(np.nanmax(dense[:n]))
    neg = -np.abs(dense[:300]) - 1  # every value stored: no zero in the max
    assert assessor._sample_max(sp.csr_matrix(np.nan_to_num(neg, nan=-1)), 300) == float(
        np.nanmax(np.nan_to_num(neg, nan=-1)))
    with pytest.raises(ValueError):
        assessor._sample_max(M, 0)


def _old_totals(X, n_rows):
    """``_sample_totals``' sparse branch before v0.82 (one float64 copy of the sample)."""
    sub = sp.csr_matrix(X[:n_rows], dtype=np.float64, copy=True)
    d = sub.data
    ok = np.isfinite(d)
    d[~ok] = 0.0
    lo = min(0.0, float(d[ok].min())) if ok.any() else 0.0
    hi = max(0.0, float(d[ok].max())) if ok.any() else 0.0
    if lo < 0:
        return None
    logged = hi <= 20
    if logged:
        np.expm1(d, out=d)
    return logged, np.asarray(sub.sum(axis=1)).ravel()


@pytest.mark.parametrize("kind", ["counts", "log", "negative", "empty"])
def test_sparse_totals_in_blocks_match_the_whole_sample(kind):
    rng = np.random.default_rng(4)
    counts = rng.poisson(2.0, (900, 60)).astype(np.float64)
    M = {"counts": counts, "log": np.log1p(counts), "negative": counts - 1.0,
         "empty": np.zeros_like(counts)}[kind]
    X = sp.csr_matrix(M)
    got, want = assessor._sample_totals(X, 800), _old_totals(X, 800)
    if want is None:
        assert got is None
        return
    assert got[0] == want[0]
    np.testing.assert_allclose(got[1], want[1], rtol=1e-12)


def test_the_assessment_holds_little_beside_a_log_normalised_matrix():
    """The log-transform check read 1,000 cells dense in float64 (0.4 GB at 34k genes)."""
    import tracemalloc

    rng = np.random.default_rng(5)
    n, g = 1200, 6000
    counts = sp.random(n, g, density=0.05, format="csr", random_state=rng, dtype=np.float64)
    counts.data = np.ceil(counts.data * 10)
    a = ad.AnnData(counts.copy())
    a.raw = a.copy()
    tot = np.asarray(counts.sum(axis=1)).ravel()
    a.X = sp.csr_matrix(sp.diags(1e4 / tot) @ counts, dtype=np.float32)
    a.X.data = np.log1p(a.X.data)
    assessor.assess_preprocessing(a)  # imports
    tracemalloc.start()
    try:
        assessor._check_log_transform(a)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < 0.5 * 1000 * g * 8, peak  # was over 1000 x g float64 (and its float32 source)


# --- held in the budget on the routes -------------------------------------------------------


def _register(data_dir: Path, did: str, adata: ad.AnnData) -> None:
    d = data_dir / "uploads" / did
    d.mkdir(parents=True, exist_ok=True)
    (d / "metadata.json").write_text(json.dumps({"id": did, "name": f"{did} name", "status": "ready"}))
    adata.write_h5ad(d / "data.h5ad")


def _budget(monkeypatch, free_gb: float, limit_gb: float = 8.0) -> None:
    def read(*_a, **_k):
        return memory_budget.MemoryBudget(int(limit_gb * GB), int((limit_gb - free_gb) * GB), True,
                                          "container limit")

    monkeypatch.setattr(memory_budget, "read_budget", read)
    monkeypatch.setattr(memory_budget, "release_memory", lambda: None)
    monkeypatch.setattr("scview.core.dataset_manager.REFUSAL_WAIT_SECONDS", 1)


def _spy(monkeypatch, dm) -> list:
    held: list = []
    real = dm.hold_read

    def spy(ids, need, label=""):
        held.append((need, label))
        return real(ids, need, label)

    monkeypatch.setattr(dm, "hold_read", spy)
    return held


async def test_the_assessment_and_insight_hold_their_need(client, dataset_manager, tmp_data_dir, monkeypatch):
    _register(Path(tmp_data_dir), "x", _counts(n=400, g=80))
    await dataset_manager.get_or_load_dataset("x")
    _budget(monkeypatch, free_gb=6.0)
    held = _spy(monkeypatch, dataset_manager)
    need = assessor.assessment_need(dataset_manager.get_dataset("x").adata)
    assert need > 0
    r = await client.get(f"{API}/datasets/x/assessment")
    assert r.status_code == 200, r.text
    r = await client.get(f"{API}/datasets/x/assistant/insight")
    assert r.status_code == 200, r.text
    assert [n for n, label in held if label == "Assessing this dataset"] == [need, need]
    assert dataset_manager._reads == {}


async def test_markers_for_a_column_hold_their_estimate(client, dataset_manager, tmp_data_dir, monkeypatch):
    _register(Path(tmp_data_dir), "x", _counts(n=400, g=80))
    await dataset_manager.get_or_load_dataset("x")
    _budget(monkeypatch, free_gb=6.0)
    held = _spy(monkeypatch, dataset_manager)
    adaptor = dataset_manager.get_dataset("x")
    need = preflight.markers_need(adaptor, 2)
    X = adaptor.adata.X
    # Counts: the shifted log's CSR copy, then the Wilcoxon blocks.
    assert need >= X.nnz * preflight.SHIFTED_LOG_BYTES_PER_VALUE
    r = await client.get(f"{API}/datasets/x/markers?format=json&groupby_column=Celltype")
    assert r.status_code == 200, r.text
    assert any(n == need and label.startswith("Marker genes for 'Celltype'") for n, label in held), held
    assert dataset_manager._reads == {}


async def test_markers_that_cannot_fit_are_refused_before_they_run(client, dataset_manager, tmp_data_dir,
                                                                   monkeypatch):
    _register(Path(tmp_data_dir), "x", _counts(n=400, g=80))
    await dataset_manager.get_or_load_dataset("x")
    _budget(monkeypatch, free_gb=0.5)
    monkeypatch.setattr(preflight, "markers_need", lambda *_a: 2 * GB)

    def never(*_a, **_k):
        raise AssertionError("the markers ran")

    monkeypatch.setattr(markers_api, "_compute_and_save", never)
    r = await client.get(f"{API}/datasets/x/markers?format=json&groupby_column=Celltype")
    assert r.status_code == 507, r.text
    detail = r.json()["detail"]
    assert detail.startswith("Marker genes for 'Celltype' on x name") and "2.0 GB" in detail


def test_markers_need_on_log_values_has_no_copy():
    a = _counts(n=300, g=40)
    a.raw = a.copy()
    a.X = sp.csr_matrix(np.log1p(a.X.toarray()), dtype=np.float32)

    class _A:
        adata = a
        h5ad_path = ""

    on_log = preflight.markers_need(_A, 2)
    a2 = _counts(n=300, g=40)
    _A.adata = a2
    on_counts = preflight.markers_need(_A, 2)
    assert a2.X.nnz == a.X.nnz
    assert on_counts - on_log == a2.X.nnz * preflight.SHIFTED_LOG_BYTES_PER_VALUE + 301 * 8


# --- the preflight's doublet plan ---------------------------------------------------------


def _for_plan(n: int = 800, g: int = 300) -> ad.AnnData:
    rng = np.random.default_rng(6)
    a = ad.AnnData(sp.csr_matrix(rng.poisson(0.8, (n, g)).astype(np.float32)))
    a.var_names = [f"G{i}" for i in range(g)]
    return a


def test_the_doublet_plan_is_counted_once_per_counts_matrix(monkeypatch):
    a = _for_plan()
    calls: list = []
    real = preflight.cost_channels

    def spy(plan, counts, codes):
        calls.append(1)
        return real(plan, counts, codes)

    monkeypatch.setattr(preflight, "cost_channels", spy)
    first = preflight.doublet_plan(a)
    again = preflight.doublet_plan(a)
    assert len(calls) == 1 and first.hvg is not None
    assert (again.channels, again.hvg, again.costed) == (first.channels, first.hvg, first.costed)
    a.X = a.X.copy()  # another matrix: counted again
    preflight.doublet_plan(a)
    assert len(calls) == 2


def test_no_room_no_count(monkeypatch):
    a = _for_plan()
    asked: list = []

    def room(need):
        asked.append(need)
        return None

    plan = preflight.doublet_plan(a, room=room)
    per_cell = a.X.nnz / a.n_obs
    assert asked == [int(800 * per_cell * preflight.DBL_PLAN_BYTES_PER_VALUE)]
    assert plan.channels == [800] and plan.hvg is None  # the header rule then


async def test_the_preflight_holds_room_for_the_count_or_skips_it(client, dataset_manager, tmp_data_dir,
                                                                  monkeypatch):
    _register(Path(tmp_data_dir), "x", _for_plan())
    await dataset_manager.get_or_load_dataset("x")
    held = _spy(monkeypatch, dataset_manager)
    _budget(monkeypatch, free_gb=6.0)
    url = f"{API}/datasets/x/preflight?steps=qc_metrics,doublet_detection"
    assert (await client.get(url)).status_code == 200
    assert [label for _n, label in held] == ["Estimating doublet detection"]
    assert dataset_manager._reads == {}
    preflight._plans.clear()
    held.clear()
    monkeypatch.setattr(preflight, "DBL_PLAN_BYTES_PER_VALUE", 10**7)  # far more than is free
    r = await client.get(url)
    assert r.status_code == 200 and held == []
