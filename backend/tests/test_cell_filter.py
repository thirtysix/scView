"""AND/OR cell filters on genes and metadata (core/cell_filter.py + /cells/filter)."""

from __future__ import annotations

import asyncio
import base64
import json
import threading
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError
from scipy import sparse

from scview.core import cell_filter
from scview.core.anndata_adaptor import AnnDataAdaptor
from scview.core.cell_filter import CellFilter, mask_to_base64

API = "/api/v1"
N = 40


def _adata() -> ad.AnnData:
    """40 cells: G0 rises 0..39 in X (and twice that in the counts layer); stim
    alternates; cell_type in blocks of 10; n_counts with two NaNs; a bool flag."""
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 0.5, (N, 6)).astype("float32")
    X[:, 0] = np.arange(N, dtype="float32")
    a = ad.AnnData(sparse.csr_matrix(X))
    a.var_names = [f"G{i}" for i in range(6)]
    a.obs_names = [f"c{i}" for i in range(N)]
    a.layers["counts"] = sparse.csr_matrix(X * 2)
    a.obs["stim"] = pd.Categorical(np.where(np.arange(N) % 2 == 0, "CTRL", "STIM"))
    ct = np.repeat(["B", "NK", "T", "Mono"], 10).astype(object)
    ct[5] = np.nan
    a.obs["cell_type"] = pd.Categorical(ct, categories=["B", "NK", "T", "Mono"])
    counts = np.arange(N, dtype=float) * 100
    counts[[3, 30]] = np.nan
    a.obs["n_counts"] = counts
    a.obs["flag"] = np.arange(N) < 4
    a.obs["donor"] = np.array([f"d{i % 3}" for i in range(N)], dtype=object)
    return a


def _adaptor(a: ad.AnnData) -> AnnDataAdaptor:
    adaptor = AnnDataAdaptor("unused.h5ad")
    adaptor._adata = a
    return adaptor


def _eval(a: ad.AnnData, spec: dict) -> np.ndarray:
    return cell_filter.evaluate(_adaptor(a), CellFilter.model_validate(spec))


def _decode(mask_b64: str, n: int) -> np.ndarray:
    raw = np.frombuffer(base64.b64decode(mask_b64), dtype=np.uint8)
    return np.unpackbits(raw, bitorder="little")[:n].astype(bool)


def _register(dm, a: ad.AnnData | None = None) -> str:
    d = Path(dm.data_dir) / "uploads" / "cf"
    d.mkdir(parents=True)
    (d / "metadata.json").write_text(json.dumps({"id": "cf", "name": "cf", "status": "ready"}))
    (a if a is not None else _adata()).write_h5ad(d / "data.h5ad")
    return "cf"


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def test_gene_and_category_conditions_combine_with_and():
    a = _adata()
    got = _eval(a, {"op": "and", "items": [
        {"kind": "gene", "gene": "G0", "op": ">", "value": 29.5},
        {"kind": "category", "column": "stim", "values": ["STIM"]},
    ]})
    want = (np.arange(N) > 29.5) & (np.arange(N) % 2 == 1)
    assert np.array_equal(got, want)
    assert got.sum() == 5  # 31, 33, ..., 39


def test_or_with_one_group():
    a = _adata()
    # cell_type in {B} OR (G0 <= 35 AND G0 >= 33): 9 B cells (one is NaN) + 33..35
    got = _eval(a, {"op": "or", "items": [
        {"kind": "category", "column": "cell_type", "values": ["B"]},
        {"kind": "group", "op": "and", "conditions": [
            {"kind": "gene", "gene": "G0", "op": "<=", "value": 35},
            {"kind": "gene", "gene": "G0", "op": ">=", "value": 33},
        ]},
    ]})
    want = np.zeros(N, dtype=bool)
    want[:10] = True
    want[5] = False  # missing cell type matches no value
    want[33:36] = True
    assert np.array_equal(got, want)


def test_range_has_open_ends_and_never_matches_missing_values():
    a = _adata()
    got = _eval(a, {"op": "and", "items": [{"kind": "range", "column": "n_counts", "min": 2500}]})
    want = np.arange(N) * 100 >= 2500
    want[30] = False  # NaN
    assert np.array_equal(got, want)
    got = _eval(a, {"op": "and", "items": [{"kind": "range", "column": "n_counts", "max": 500}]})
    assert np.flatnonzero(got).tolist() == [0, 1, 2, 4, 5]  # 3 is NaN


def test_category_matches_bool_and_text_columns_as_listed():
    a = _adata()
    assert np.flatnonzero(_eval(a, {"op": "and", "items": [
        {"kind": "category", "column": "flag", "values": ["True"]}]})).tolist() == [0, 1, 2, 3]
    got = _eval(a, {"op": "and", "items": [{"kind": "category", "column": "donor", "values": ["d1", "zz"]}]})
    assert np.array_equal(got, np.arange(N) % 3 == 1)
    none = _eval(a, {"op": "and", "items": [{"kind": "category", "column": "stim", "values": ["nope"]}]})
    assert not none.any()


def test_gene_conditions_read_the_filters_layer():
    a = _adata()
    spec = {"op": "and", "items": [{"kind": "gene", "gene": "G0", "op": ">=", "value": 30}]}
    assert _eval(a, spec).sum() == 10  # X: 30..39
    assert _eval(a, {**spec, "layer": "counts"}).sum() == 25  # counts = 2x: 15..39


@pytest.mark.parametrize("spec, status, words", [
    ({"op": "and", "items": [{"kind": "gene", "gene": "NOPE", "op": ">", "value": 0}]}, 404, "NOPE"),
    ({"op": "and", "items": [{"kind": "category", "column": "nope", "values": ["x"]}]}, 404, "nope"),
    ({"op": "and", "items": [{"kind": "range", "column": "stim", "min": 0}]}, 400, "not numeric"),
    ({"op": "and", "items": [{"kind": "range", "column": "flag", "min": 0}]}, 400, "not numeric"),
    ({"op": "and", "layer": "spliced", "items": [{"kind": "gene", "gene": "G0", "op": ">", "value": 0}]},
     404, "spliced"),
])
def test_filters_this_dataset_cannot_answer_raise_with_a_status(spec, status, words):
    with pytest.raises(cell_filter.FilterError) as err:
        _eval(_adata(), spec)
    assert err.value.status == status and words in str(err.value)


@pytest.mark.parametrize("spec", [
    {"op": "and", "items": []},
    {"op": "xor", "items": [{"kind": "category", "column": "stim", "values": ["A"]}]},
    {"op": "and", "items": [{"kind": "category", "column": "stim", "values": []}]},
    {"op": "and", "items": [{"kind": "range", "column": "n_counts"}]},
    {"op": "and", "items": [{"kind": "range", "column": "n_counts", "min": 5, "max": 1}]},
    {"op": "and", "items": [{"kind": "gene", "gene": "G0", "op": "!=", "value": 1}]},
    {"op": "and", "items": [{"kind": "gene", "gene": "G0", "op": ">", "value": "lots"}]},
    {"op": "and", "items": [{"kind": "group", "op": "and", "conditions": [
        {"kind": "group", "op": "or", "conditions": [
            {"kind": "category", "column": "stim", "values": ["A"]}]}]}]},
    {"op": "and", "items": [{"kind": "group", "op": "and", "conditions": [
        {"kind": "category", "column": "stim", "values": ["A"]}] * 20}] * 3},
])
def test_malformed_filters_are_rejected(spec):
    with pytest.raises(ValidationError):
        CellFilter.model_validate(spec)


def test_mask_bits_are_least_significant_first():
    mask = np.zeros(10, dtype=bool)
    mask[[0, 3, 9]] = True
    assert base64.b64decode(mask_to_base64(mask)) == bytes([0b00001001, 0b00000010])
    assert np.array_equal(_decode(mask_to_base64(mask), 10), mask)


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


async def test_endpoint_returns_count_and_mask(client, dataset_manager):
    did = _register(dataset_manager)
    spec = {"op": "and", "items": [
        {"kind": "gene", "gene": "G0", "op": ">", "value": 29.5},
        {"kind": "category", "column": "stim", "values": ["STIM"]},
    ]}
    r = await client.post(f"{API}/datasets/{did}/cells/filter", json={"filter": spec})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n_cells"] == N and body["n_matching"] == 5
    assert np.flatnonzero(_decode(body["mask"], N)).tolist() == [31, 33, 35, 37, 39]

    r = await client.post(f"{API}/datasets/{did}/cells/filter", json={"filter": spec, "mask": False})
    assert r.json() == {"n_cells": N, "n_matching": 5, "mask": None}


async def test_endpoint_errors(client, dataset_manager):
    did = _register(dataset_manager)
    url = f"{API}/datasets/{did}/cells/filter"
    r = await client.post(url, json={"filter": {"op": "and", "items": [
        {"kind": "gene", "gene": "NOPE", "op": ">", "value": 0}]}})
    assert r.status_code == 404 and "NOPE" in r.json()["detail"]
    r = await client.post(url, json={"filter": {"op": "and", "items": []}})
    assert r.status_code == 422
    r = await client.post(f"{API}/datasets/missing/cells/filter", json={"filter": {"op": "and", "items": [
        {"kind": "category", "column": "stim", "values": ["STIM"]}]}})
    assert r.status_code == 404


async def test_evaluation_runs_off_the_event_loop(client, dataset_manager, monkeypatch):
    did = _register(dataset_manager)
    await dataset_manager.load_dataset(did)
    threads: list[threading.Thread] = []
    real = cell_filter.evaluate

    def slow(adaptor, spec):
        threads.append(threading.current_thread())
        time.sleep(0.3)
        return real(adaptor, spec)

    monkeypatch.setattr("scview.api.v1.cells.evaluate", slow)
    gaps: list[float] = []
    done = asyncio.Event()

    async def beat():
        last = time.perf_counter()
        while not done.is_set():
            await asyncio.sleep(0.01)
            now = time.perf_counter()
            gaps.append(now - last - 0.01)
            last = now

    hb = asyncio.create_task(beat())
    try:
        r = await client.post(f"{API}/datasets/{did}/cells/filter", json={"filter": {"op": "and", "items": [
            {"kind": "category", "column": "stim", "values": ["STIM"]}]}})
    finally:
        done.set()
        await hb
    assert r.status_code == 200 and r.json()["n_matching"] == N // 2
    assert threads and threads[0] is not threading.main_thread()
    assert max(gaps, default=0.0) < 0.15
