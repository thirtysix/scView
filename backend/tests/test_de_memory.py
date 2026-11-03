"""Memory of a DE test on a dataset in memory: what it copies, what it says it needs, and
the 507 it gives (also as the MCP tool's error) when that cannot fit.

Liu 2021's 125k innate PBMCs (X and .raw 201M values each, int64 indices), classical
monocytes T0 vs HC: the transcriptome-size check cut .raw to X's genes (a copy of all of
it, +4.5 GB at its peak) and the backend was killed; the test itself takes +0.9 GB.
"""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from scview.api.v1 import de as de_api
from scview.core import memory_budget, normalization, preflight, transcriptome_size

API = "/api/v1"
GB = 1024**3


def _dataset(n: int = 300, g: int = 60, seed: int = 0) -> ad.AnnData:
    """Counts in .raw (same genes, same order), log-normalised X: as CELLxGENE ships them."""
    rng = np.random.default_rng(seed)
    counts = sp.csr_matrix(rng.poisson(1.0, (n, g)).astype(np.float32))
    a = ad.AnnData(counts.copy())
    a.var_names = [f"G{i}" for i in range(g)]
    a.obs_names = [f"c{i}" for i in range(n)]
    a.obs["grp"] = pd.Categorical(np.where(np.arange(n) % 2 == 0, "a", "b"))
    a.obs["donor"] = pd.Categorical([f"d{i % 6}" for i in range(n)])
    a.raw = a.copy()
    a.X = sp.csr_matrix(np.log1p(counts.toarray() / counts.sum(1).A * 1e4), dtype=np.float32)
    return a


def test_read_only_counts_hand_over_raw_without_cutting_it():
    a = _dataset()
    M, source = normalization.find_counts(a, read_only=True)
    assert source == ".raw" and M is a.raw.X
    M, names, _ = normalization.counts_all_genes(a, read_only=True)
    assert M is a.raw.X and list(names) == list(a.var_names)
    # Not read-only: a matrix of its own, as before (callers may change it).
    M, _ = normalization.find_counts(a)
    assert M is not a.raw.X and (M != a.raw.X).nnz == 0
    # Other genes or another order: cut to X's, read-only or not.
    b = a[:, a.var_names[::-1]].copy()
    b.raw = a.raw.to_adata()
    M, _ = normalization.find_counts(b, read_only=True)
    assert M is not b.raw.X and np.allclose(M[:, 0].toarray(), a.raw.X[:, -1].toarray())


def test_the_transcriptome_check_sums_raw_without_copying_it(monkeypatch):
    a = _dataset()

    def no_cut(*_a, **_k):
        raise AssertionError("cut .raw to X's genes")

    monkeypatch.setattr(type(a.raw), "__getitem__", no_cut)
    mask_a = (a.obs["grp"] == "a").to_numpy()
    out = transcriptome_size.check(a, mask_a, ~mask_a)
    assert out is not None and out["source"] == ".raw"
    assert out["median_a"] == float(np.median(np.asarray(a.raw.X.sum(1)).ravel()[mask_a]))


def test_the_in_memory_estimate_counts_the_rows_copied_out():
    a = _dataset(n=1000, g=200)
    M = a.X
    keep = np.zeros(M.shape[0], dtype=bool)
    keep[:400] = True
    rows = M[np.flatnonzero(keep)]
    mm = preflight._kept_rows(M, keep)
    assert mm.nnz == rows.nnz and mm.n_cells == 400 and mm.n_genes == 200
    need = preflight.de_memory_need(M, keep)
    assert need == preflight.de_read_need(mm, 400)
    # Every row compared: the test reads M itself, no copy of the rows.
    every = np.ones(M.shape[0], dtype=bool)
    full = preflight._kept_rows(M, every)
    assert preflight.de_memory_need(M, every) == preflight.de_read_need(full, 1000) - preflight.rows_bytes(full, 1000)
    # Counts are copied once more to be log-normalised for the test; logistic regression copies too.
    assert preflight.de_memory_need(M, keep, counts=True) == need + preflight.rows_bytes(mm, 400)
    assert preflight.de_memory_need(M, keep, logreg=True) > need
    assert preflight.de_memory_need(M, np.zeros(M.shape[0], dtype=bool)) == 0
    # Dense: every element of the kept rows.
    dense = preflight._kept_rows(M.toarray(), keep)
    assert not dense.sparse and dense.nnz == 400 * 200
    # Pseudobulk: the counts rows copied out, with their sums.
    pb = preflight.pseudobulk_memory_need(a.raw.X, keep)
    assert pb == int(preflight.rows_bytes(preflight._kept_rows(a.raw.X, keep), 400) * preflight.PB_ROWS_FACTOR
                     + preflight.DE_FIXED)


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


COMPARE = {"split_by": "grp", "group_a": "a", "group_b": "b"}


async def test_the_route_reserves_the_in_memory_estimate(client, dataset_manager, tmp_data_dir, monkeypatch):
    _register(Path(tmp_data_dir), "x", _dataset())
    await dataset_manager.get_or_load_dataset("x")
    _budget(monkeypatch, free_gb=6.0)
    held = []
    real = dataset_manager.hold_read

    def spy(ids, need, label=""):
        held.append(need)
        return real(ids, need, label)

    monkeypatch.setattr(dataset_manager, "hold_read", spy)
    adata = dataset_manager.get_dataset("x").adata
    r = await client.post(f"{API}/datasets/x/de/compare", json=COMPARE)
    assert r.status_code == 200, r.text
    expected = preflight.de_memory_need(adata.X, np.ones(adata.n_obs, dtype=bool))
    assert expected > 0 and expected in held
    r = await client.post(f"{API}/datasets/x/de/compare", json={**COMPARE, "method": "pseudobulk",
                                                                "replicate_key": "donor"})
    assert r.status_code == 200, r.text
    assert preflight.pseudobulk_memory_need(adata.raw.X, np.ones(adata.n_obs, dtype=bool)) in held
    assert dataset_manager._reads == {}


@pytest.mark.parametrize("method", ["wilcoxon", "pseudobulk"])
async def test_a_test_that_cannot_fit_is_refused_before_it_allocates(client, dataset_manager, tmp_data_dir,
                                                                     monkeypatch, method):
    _register(Path(tmp_data_dir), "x", _dataset())
    await dataset_manager.get_or_load_dataset("x")
    _budget(monkeypatch, free_gb=0.5)
    monkeypatch.setattr(de_api, "_test_bytes", lambda *_a: 2 * GB)
    monkeypatch.setattr(de_api, "_counts_bytes", lambda *_a: 2 * GB)

    def never(*_a, **_k):
        raise AssertionError("the test ran")

    monkeypatch.setattr(de_api, "_prepare", never)
    monkeypatch.setattr("scview.core.pseudobulk.pseudobulk_de", never)
    body = {**COMPARE, "method": method, **({"replicate_key": "donor"} if method == "pseudobulk" else {})}
    r = await client.post(f"{API}/datasets/x/de/compare", json=body)
    assert r.status_code == 507, r.text
    detail = r.json()["detail"]
    assert detail.startswith(("Differential expression on x name", "Pseudobulk DE on x name"))
    assert "2.0 GB" in detail and "0.5 GB free" in detail
