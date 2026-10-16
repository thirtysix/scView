"""QC distributions per sample / batch / cluster (core/qc_groups.py, GET /qc/by-group).

Checks the default column (donor first, as DE and the QC report pick it), that a
group's quantiles reproduce its exact quartiles, that the payload stays bounded
whatever the cell count, and that % mito is left out when unmeasurable.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from scview.core import provenance, qc_groups
from scview.core.anndata_adaptor import per_cell_qc

API = "/api/v1"


def _kang_like(n: int = 640, mt: bool = True, seed: int = 0) -> ad.AnnData:
    """8 donors x CTRL/STIM, a per-condition orig_ident (as Kang's), 4 clusters."""
    rng = np.random.default_rng(seed)
    a = ad.AnnData(rng.poisson(2.0, (n, 50)).astype("float32"))
    names = [f"G{i}" for i in range(50)]
    if mt:
        names[:2] = ["MT-CO1", "MT-ND1"]
    a.var_names = names
    a.obs_names = [f"c{i}" for i in range(n)]
    stim = np.tile(["CTRL", "STIM"], n // 2)
    a.obs["stim"] = pd.Categorical(stim)
    a.obs["orig_ident"] = pd.Categorical(stim)  # repeats the condition
    a.obs["Donor"] = pd.Categorical(np.repeat([f"d{i}" for i in range(8)], n // 8))
    a.obs["leiden"] = pd.Categorical(rng.integers(0, 4, n).astype(str))
    return a


def _obs_only(**cols) -> ad.AnnData:
    n = len(next(iter(cols.values())))
    a = ad.AnnData(np.zeros((n, 3), dtype="float32"))
    a.obs_names = [f"c{i}" for i in range(n)]
    for k, v in cols.items():
        a.obs[k] = pd.Categorical(v)
    return a


# --- the default column --------------------------------------------------------


def test_the_donor_column_beats_a_sample_column_that_repeats_the_condition():
    assert qc_groups.default_grouping(_kang_like()) == ("Donor", "donor")


def test_without_donors_a_batch_then_the_clustering_is_used():
    n = 60
    batch = np.repeat(["b1", "b2", "b3"], n // 3)
    leiden = np.tile(["0", "1"], n // 2)
    assert qc_groups.default_grouping(_obs_only(batch=batch, leiden=leiden)) == ("batch", "batch")
    assert qc_groups.default_grouping(_obs_only(leiden=leiden)) == ("leiden", "clustering")
    assert qc_groups.default_grouping(_obs_only(cell_label=leiden)) == (None, None)


def test_the_batch_correction_key_is_the_batch_used():
    n = 60
    a = _obs_only(batch=np.repeat(["b1", "b2"], n // 2), chemistry=np.tile(["v2", "v3"], n // 2))
    provenance.record_step(a, step="batch_correction", tool="harmony",
                           params={"batch_key": "chemistry"})
    assert qc_groups.default_grouping(a) == ("chemistry", "batch")


def test_a_donor_column_with_too_many_groups_is_passed_over():
    n = 120
    a = _obs_only(sample_id=[f"s{i}" for i in range(n)], leiden=np.tile(["0", "1"], n // 2))
    assert qc_groups.default_grouping(a) == ("leiden", "clustering")


# --- the summaries -------------------------------------------------------------


def test_each_group_carries_its_exact_quartiles_and_range():
    a = _kang_like(n=4000)
    cells = per_cell_qc(a)
    out = qc_groups.qc_by_group(a, "Donor", cells)
    donors = a.obs["Donor"].astype(str).to_numpy()
    for name, key in (("n_genes_by_counts", "n_genes"), ("total_counts", "total_counts"),
                      ("pct_counts_mt", "pct_mt")):
        for g, q in out["metrics"][name].items():
            full = np.asarray(cells[key])[donors == g]
            assert len(q) == qc_groups.N_QUANTILES  # 500 cells -> 201 values
            # Plotly's linear quartiles of 201 points are points 50, 100, 150.
            for i, pct in ((0, 0), (50, 25), (100, 50), (150, 75), (200, 100)):
                assert q[i] == pytest.approx(np.percentile(full, pct), abs=1e-3)


def test_a_small_group_sends_all_its_values():
    a = _kang_like(n=640)
    a.obs["tiny"] = pd.Categorical(np.where(np.arange(640) < 10, "few", "many"))
    out = qc_groups.qc_by_group(a, "tiny", per_cell_qc(a))
    few = out["metrics"]["total_counts"]["few"]
    assert few == sorted(few) and len(few) == 10


def test_the_payload_is_bounded_however_many_cells():
    a = _kang_like(n=40_000)
    a.obs["tiny"] = pd.Categorical(np.where(np.arange(40_000) < 12, "rare", "common"))
    out = qc_groups.qc_by_group(a, "tiny", per_cell_qc(a))
    assert all(len(v) <= qc_groups.N_QUANTILES for m in out["metrics"].values() for v in m.values())
    sc = out["scatter"]
    assert sc["n_shown"] <= qc_groups.N_SCATTER + qc_groups.MIN_SCATTER_PER_GROUP * out["n_groups"]
    assert len(sc["x"]) == len(sc["y"]) == len(sc["group"]) == sc["n_shown"]
    # The 12 rare cells all show, though a proportional sample would hold 1 or 2.
    assert sc["group"].count(out["groups"].index({"name": "rare", "n_cells": 12})) == 12


def test_groups_keep_category_order_and_unlabelled_cells_are_counted():
    a = _kang_like(n=640)
    a.obs["leiden"] = pd.Categorical(
        np.where(np.arange(640) < 40, None, np.tile(["10", "2", "1"], 214)[:640]),
        categories=["1", "2", "10"],
    )
    out = qc_groups.qc_by_group(a, "leiden", per_cell_qc(a))
    assert [g["name"] for g in out["groups"]] == ["1", "2", "10"]
    assert out["n_unlabelled"] == 40
    assert sum(g["n_cells"] for g in out["groups"]) == 600
    assert -1 not in out["scatter"]["group"]


def test_mito_is_left_out_when_unmeasurable():
    a = _kang_like(mt=False)
    out = qc_groups.qc_by_group(a, "Donor", per_cell_qc(a))
    assert out["has_mt"] is False
    assert list(out["metrics"]) == ["n_genes_by_counts", "total_counts"]


def test_too_many_groups_is_refused_with_a_way_out():
    a = _kang_like(n=640)
    a.obs["barcode_group"] = pd.Categorical([f"x{i % 60}" for i in range(640)])
    with pytest.raises(ValueError, match="60 groups.*at most 50.*Pick a column"):
        qc_groups.qc_by_group(a, "barcode_group", per_cell_qc(a))


# --- the endpoint --------------------------------------------------------------


def _register(dm, a: ad.AnnData, did: str = "qcg") -> str:
    d = Path(dm.data_dir) / "uploads" / did
    d.mkdir(parents=True)
    (d / "metadata.json").write_text(json.dumps({"id": did, "name": did, "status": "ready"}))
    a.write_h5ad(d / "data.h5ad")
    return did


async def test_endpoint_defaults_to_the_donor_column(client, dataset_manager):
    did = _register(dataset_manager, _kang_like())
    r = await client.get(f"{API}/datasets/{did}/qc/by-group")
    assert r.status_code == 200, r.text
    body = r.json()
    assert (body["column"], body["reason"]) == ("Donor", "donor")
    assert [g["name"] for g in body["groups"]] == [f"d{i}" for i in range(8)]
    assert set(body["metrics"]) == {"n_genes_by_counts", "total_counts", "pct_counts_mt"}


async def test_endpoint_groups_by_a_named_column(client, dataset_manager):
    did = _register(dataset_manager, _kang_like(mt=False))
    r = await client.get(f"{API}/datasets/{did}/qc/by-group", params={"column": "leiden"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert (body["column"], body["reason"], body["has_mt"]) == ("leiden", "requested", False)
    assert "pct_counts_mt" not in body["metrics"]


async def test_endpoint_without_a_usable_column_says_so(client, dataset_manager):
    a = _kang_like()
    a.obs = a.obs[[]]
    did = _register(dataset_manager, a)
    r = await client.get(f"{API}/datasets/{did}/qc/by-group")
    assert r.status_code == 200, r.text
    assert r.json()["column"] is None and "pick a column" in r.json()["note"]


async def test_endpoint_errors(client, dataset_manager):
    a = _kang_like()
    a.obs["many"] = pd.Categorical([f"x{i % 60}" for i in range(a.n_obs)])
    did = _register(dataset_manager, a)
    r = await client.get(f"{API}/datasets/{did}/qc/by-group", params={"column": "many"})
    assert r.status_code == 400 and "at most 50" in r.json()["detail"]
    r = await client.get(f"{API}/datasets/{did}/qc/by-group", params={"column": "nope"})
    assert r.status_code == 404
    r = await client.get(f"{API}/datasets/ghost/qc/by-group")
    assert r.status_code == 404


async def test_endpoint_computes_in_a_worker_thread(client, dataset_manager, monkeypatch):
    did = _register(dataset_manager, _kang_like())
    threads: list[threading.Thread] = []
    real = qc_groups.qc_by_group

    def spy(*args, **kwargs):
        threads.append(threading.current_thread())
        return real(*args, **kwargs)

    monkeypatch.setattr(qc_groups, "qc_by_group", spy)
    r = await client.get(f"{API}/datasets/{did}/qc/by-group")
    assert r.status_code == 200, r.text
    assert len(threads) == 1 and threads[0] is not threading.main_thread()
