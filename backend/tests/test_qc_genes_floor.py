"""The genes floor under per-sample MAD QC (core/qc_filter.py ``genes_floor``).

A sample whose cells are partly near empty has a wide MAD, so its own genes cut can fall to a
few genes and keep the empty ones (a Kuppe heart slide: 3.7 genes). MAD mode's genes cut is
therefore never below a floor (200 for cells, 100 for spots, none on bins or panels), the
preview and the record say per sample which cut applied and what the floor removed, and a run
recorded before the floor existed is replayed without it.
"""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from scview.core import mito_refilter, provenance, qc_filter, spatial
from scview.core.anndata_adaptor import per_cell_qc
from scview.core.pipeline import PipelineParams, run_pipeline
from scview.core.qc_filter import Thresholds

API = "/api/v1"


def _cells(groups: dict[str, np.ndarray], seed: int = 0) -> ad.AnnData:
    """Cells with these genes per cell per sample (counts = 2.5 x genes, 2% mito)."""
    rng = np.random.default_rng(seed)
    parts = []
    for name, genes in groups.items():
        genes = np.asarray(genes, dtype=float)
        parts.append(pd.DataFrame({"Donor": name, "n_genes_by_counts": genes,
                                   "total_counts": np.round(genes * 2.5),
                                   "pct_counts_mt": np.abs(rng.normal(2.0, 0.5, genes.size))}))
    obs = pd.concat(parts, ignore_index=True)
    obs.index = [f"c{i}" for i in range(len(obs))]
    obs["Donor"] = pd.Categorical(obs["Donor"], categories=list(groups))
    a = ad.AnnData(sp.csr_matrix((len(obs), 3), dtype=np.float32), obs=obs)
    a.var_names = ["MT-CO1", "G1", "G2"]
    return a


def _normal(n: int, seed: int) -> np.ndarray:
    return np.round(np.exp(np.random.default_rng(seed).normal(np.log(1500), 0.3, n)))


def _bimodal(n_empty: int = 90, n_full: int = 110, seed: int = 1) -> np.ndarray:
    """A sample like Kuppe's IZ_P16: many near-empty cells beside normal ones."""
    rng = np.random.default_rng(seed)
    return np.concatenate([np.round(np.exp(rng.normal(np.log(40), 0.6, n_empty))), _normal(n_full, seed)])


def _rows(out: dict) -> dict[str, dict]:
    return {r["name"]: r for r in out["mad"]["samples"]}


def test_the_floor_sets_the_cut_of_a_sample_whose_mad_cut_is_lower():
    a = _cells({"bimodal": _bimodal(), "normal": _normal(200, 2)})
    t = Thresholds(mode="mad", max_pct_mt=20.0, genes_floor=200.0)
    out = qc_filter.preview(a, per_cell_qc(a), t)
    rows = _rows(out)
    bi, ok = rows["bimodal"], rows["normal"]
    assert bi["min_genes_mad"] < 200 and bi["min_genes"] == 200.0 and bi["min_genes_by"] == "floor"
    genes = a.obs["n_genes_by_counts"].to_numpy()[a.obs["Donor"].to_numpy() == "bimodal"]
    assert bi["n_floor"] == int(((genes < 200) & (genes >= bi["min_genes_mad"])).sum()) > 0
    assert ok["min_genes_by"] == "mad" and ok["min_genes"] == ok["min_genes_mad"] > 200 and ok["n_floor"] == 0
    assert out["mad"]["genes_floor"] == 200.0
    # Without the floor the MAD cut keeps most of the near-empty cells.
    off = qc_filter.preview(a, per_cell_qc(a), Thresholds(mode="mad", max_pct_mt=20.0, genes_floor=0.0))
    assert off["n_removed"] < out["n_removed"]
    assert out["n_removed"] - off["n_removed"] == bi["n_floor"]
    assert "min_genes_by" not in _rows(off)["bimodal"] and "genes_floor" not in off["mad"]
    json.dumps(out, allow_nan=False)


def test_the_floor_is_not_applied_where_it_would_remove_most_of_a_sample():
    low = np.round(np.exp(np.random.default_rng(3).normal(np.log(120), 0.2, 200)))  # a low-complexity sample
    a = _cells({"low": low, "normal": _normal(200, 4)})
    out = qc_filter.preview(a, per_cell_qc(a), Thresholds(mode="mad", max_pct_mt=20.0, genes_floor=200.0))
    r = _rows(out)["low"]
    assert r["min_genes_by"] == "mad" and r["min_genes"] == r["min_genes_mad"] < 200
    assert r["floor_skipped"] == int((low < 200).sum()) > 100 and r["n_floor"] == 0


def test_the_default_floor_depends_on_the_data(monkeypatch):
    a = _cells({"d": _normal(150, 5)})
    assert qc_filter.default_genes_floor(a) == 200.0
    assert Thresholds.from_params(PipelineParams()).genes_floor is None  # the data decides
    assert qc_filter.resolve_floor(a, Thresholds.from_params(PipelineParams())).genes_floor == 200.0
    assert qc_filter.resolve_floor(a, Thresholds.from_params(PipelineParams(qc_genes_floor=0))).genes_floor == 0.0
    monkeypatch.setattr(spatial, "has_spatial", lambda adata: True)
    monkeypatch.setattr(spatial, "targeted_panel", lambda adata: False)
    for tech, floor in (({"technology": "visium", "unit": "spot"}, 100.0),
                        ({"technology": "visium_hd", "unit": "cell"}, 100.0),
                        ({"technology": "visium_hd", "unit": "bin"}, 0.0),
                        ({"technology": "cells", "unit": "cell"}, 200.0)):
        monkeypatch.setattr(spatial, "technology", lambda adata, tech=tech: tech)
        assert qc_filter.default_genes_floor(a) == floor, tech
    monkeypatch.setattr(spatial, "targeted_panel", lambda adata: True)
    assert qc_filter.default_genes_floor(a) == 0.0


def test_fixed_mode_does_not_use_the_floor():
    a = _cells({"bimodal": _bimodal()})
    t = Thresholds(mode="fixed", min_genes=10.0, max_pct_mt=20.0, genes_floor=200.0)
    out = qc_filter.preview(a, per_cell_qc(a), t)
    genes = a.obs["n_genes_by_counts"].to_numpy()
    assert out["n_removed"] == int((genes < 10).sum())
    assert out["thresholds"]["min_genes"] == 10.0 and out["thresholds"]["genes_floor"] is None


def test_the_run_records_the_floor_and_the_step_removes_what_the_preview_says():
    a = _cells({"bimodal": _bimodal(), "normal": _normal(200, 2)})
    params = PipelineParams(min_cells=0)
    prev = qc_filter.preview(a, per_cell_qc(a), Thresholds.from_params(params))
    out, res = run_pipeline(a.copy(), ["filtering"], params)
    assert res.errors == {} and out.n_obs == prev["n_kept"]
    p = provenance.read_provenance(out)["history"][-1]["params"]
    assert p["qc_genes_floor"] == 200.0 and p["qc_thresholds"]["genes_floor"] == 200.0
    rows = {r["name"]: r for r in p["qc_thresholds"]["samples"]}
    assert rows["bimodal"]["min_genes_by"] == "floor" and rows["bimodal"]["n_floor"] > 0
    assert "genes never cut below 200" in qc_filter.applied_summary(p["qc_thresholds"])

    # Filtering again with the same settings reuses the record: nothing new is removed.
    again, _ = run_pipeline(out.copy(), ["filtering"], params)
    assert again.n_obs == out.n_obs
    # Another floor reuses the MAD bounds and puts the new floor on top.
    looser = qc_filter.preview(out, per_cell_qc(out), Thresholds(mode="mad", max_pct_mt=20.0, genes_floor=50.0))
    r = _rows(looser)["bimodal"]
    assert looser["mad"]["reused_from"] is not None
    assert r["min_genes_mad"] == rows["bimodal"]["min_genes_mad"] and r["min_genes"] == 50.0


def test_a_run_recorded_before_the_floor_is_replayed_without_it():
    a = _cells({"bimodal": _bimodal()})
    provenance.record_step(a, step="filtering", tool="scview.qc_filter",
                           params={"qc_threshold_mode": "mad", "qc_mad_k": 5.0, "qc_mad_k_mt": 3.0})
    _, params, _ = mito_refilter.plan_params(a)
    assert params.qc_genes_floor == 0.0
    t = Thresholds.from_params(params)
    assert qc_filter.resolve_floor(a, t).genes_floor == 0.0
    # A record with the floor replays it.
    b = _cells({"bimodal": _bimodal()})
    provenance.record_step(b, step="filtering", tool="scview.qc_filter",
                           params={"qc_threshold_mode": "mad", "qc_genes_floor": 200.0})
    assert mito_refilter.plan_params(b)[1].qc_genes_floor == 200.0
    # And the replay script applies a recorded run's own thresholds, floor included.
    out, _ = run_pipeline(_cells({"bimodal": _bimodal()}), ["filtering"], PipelineParams(min_cells=0))
    from scview.core.replay_script import build_script

    script = build_script(out)
    assert '"min_genes_by": "floor"' in script or "'min_genes_by': 'floor'" in script


def _register(dm, a: ad.AnnData, did: str) -> str:
    d = Path(dm.data_dir) / "uploads" / did
    d.mkdir(parents=True)
    (d / "metadata.json").write_text(json.dumps({"id": did, "name": did, "status": "ready"}))
    a.write_h5ad(d / "data.h5ad")
    return did


async def test_the_preview_endpoint_takes_a_genes_floor(client, dataset_manager):
    did = _register(dataset_manager, _cells({"bimodal": _bimodal(), "normal": _normal(200, 2)}), "floor1")
    url = f"{API}/datasets/{did}/qc/filter-preview"
    default = (await client.get(url)).json()
    assert default["thresholds"]["genes_floor"] == 200.0 and default["thresholds"]["min_genes"] is None
    assert _rows(default)["bimodal"]["min_genes_by"] == "floor"
    off = (await client.get(url, params={"genes_floor": 0})).json()
    assert off["thresholds"]["genes_floor"] == 0.0 and off["n_removed"] < default["n_removed"]
    assert (await client.get(url, params={"genes_floor": -5})).status_code == 400


@pytest.mark.parametrize("floor", [None, 0.0])
def test_the_run_params_take_the_floor_from_a_request(floor):
    from scview.api.v1.assessment import _run_params

    params = _run_params({} if floor is None else {"qc_genes_floor": floor})
    assert params.qc_genes_floor == floor
