"""Genes changing along pseudotime (core/pseudotime_genes.py + /trajectory/{column}/trend-genes)."""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from scview.core.pseudotime_genes import (
    MIN_DETECTED,
    TrendError,
    natural_spline_basis,
    trend_genes,
)

API = "/api/v1"
N_MAIN, N_TAIL = 570, 30
TREND_GENES = ["Rise", "Fall", "Hump"]


def test_natural_spline_basis_fits_lines_exactly_and_is_linear_past_the_ends():
    knots = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    x = np.linspace(0, 1, 101)
    S = natural_spline_basis(x, knots)
    assert S.shape == (101, 5)
    beta, *_ = np.linalg.lstsq(S, 2 + 3 * x, rcond=None)
    assert np.allclose(S @ beta, 2 + 3 * x)
    # A smooth curve is fitted closely (4 df beyond the mean).
    y = np.sin(2 * np.pi * x)
    beta, *_ = np.linalg.lstsq(S, y, rcond=None)
    assert np.abs(S @ beta - y).max() < 0.05
    # Natural: every basis function is a straight line beyond the boundary knots.
    for outside in (np.linspace(1, 2, 11), np.linspace(-1, 0, 11)):
        second_diff = np.diff(natural_spline_basis(outside, knots), n=2, axis=0)
        assert np.abs(second_diff).max() < 1e-9


def _trend_adata(seed: int = 0) -> ad.AnnData:
    """Log-normalised expression along a long-tailed pseudotime: most cells in
    [0, 0.3], a small group far out near 0.9 (paul15's dendritic cells from its
    MEPs). Three genes follow the cells' order (rising, falling, a mid peak), one
    marks only the far-out group, and the rest are noise."""
    rng = np.random.default_rng(seed)
    n = N_MAIN + N_TAIL
    pt = np.concatenate([np.sort(rng.uniform(0, 0.3, N_MAIN)), np.sort(rng.uniform(0.85, 0.95, N_TAIL))])
    s = np.arange(n) / (n - 1)  # the cells' order along pseudotime

    def noisy(signal):
        return np.clip(signal + rng.normal(0, 0.3, n), 0, None)

    tail = np.arange(n) >= N_MAIN
    cols = {
        "Rise": noisy(2.5 * s),
        "Fall": noisy(2.5 * (1 - s)),
        "Hump": noisy(2.5 * np.exp(-(((s - 0.5) / 0.12) ** 2))),
        "TailMarker": np.where(tail, noisy(np.full(n, 3.0)), 0.0),
        "Rare": np.where(np.arange(n) % 75 == 0, 3.0, 0.0),  # 8 cells
    }
    for i in range(40):
        cols[f"Flat{i}"] = noisy(np.full(n, 0.5))
    X = np.column_stack(list(cols.values())).astype("float32")
    a = ad.AnnData(X)
    a.var_names = list(cols)
    a.obs_names = [f"c{i}" for i in range(n)]
    a.obs["dpt_pseudotime"] = pt
    a.obs["stage"] = pd.Categorical(np.where(tail, "far", np.where(s < 0.5, "early", "late")))
    return a


def _ranked(result) -> list[str]:
    return [g["gene"] for g in result["genes"]]


def test_ranks_the_genes_that_follow_the_order_above_a_far_out_groups_marker():
    a = _trend_adata()
    res = trend_genes(a, "dpt_pseudotime", n_top=10, n_bins=20)
    ranked = _ranked(res)
    # On raw pseudotime the last spline piece fitted the far-out group alone and
    # its marker came first; along the cells' order each cell counts once.
    assert set(ranked[:3]) == set(TREND_GENES)
    assert ranked.index("TailMarker") >= 3
    genes = {g["gene"]: g for g in res["genes"]}
    assert genes["Rise"]["pattern"] == "rising" and genes["Fall"]["pattern"] == "falling"
    assert genes["Hump"]["pattern"] == "transient"
    assert genes["Fall"]["peak_pseudotime"] < genes["Hump"]["peak_pseudotime"] < genes["Rise"]["peak_pseudotime"]
    assert genes["Rise"]["amplitude"] > 1.5 and genes["Rise"]["pval_adj"] < 1e-10
    assert all(not g.startswith("Flat") for g in ranked[:4])

    # Trends: equal-count bins along pseudotime, one mean + one fitted value per bin.
    bins = res["bins"]
    assert len(bins["pseudotime"]) == 20 and sum(bins["n_cells"]) == N_MAIN + N_TAIL
    assert max(bins["n_cells"]) - min(bins["n_cells"]) <= 1
    assert bins["pseudotime"] == sorted(bins["pseudotime"])
    assert len(genes["Rise"]["mean"]) == len(genes["Rise"]["fitted"]) == 20
    assert genes["Rise"]["mean"][-1] > genes["Rise"]["mean"][0] + 1.5
    # A gene seen in fewer than MIN_DETECTED cells is not tested.
    assert "Rare" not in ranked and res["n_genes_tested"] == a.n_vars - 1
    assert res["min_detected"] == MIN_DETECTED and res["n_cells"] == a.n_obs


def test_restricting_to_a_branch_tests_only_its_cells():
    a = _trend_adata()
    res = trend_genes(a, "dpt_pseudotime", n_top=50, groups_column="stage", groups=["early", "late"])
    assert res["n_cells"] == N_MAIN and res["restrict"] == {"column": "stage", "groups": ["early", "late"]}
    ranked = _ranked(res)
    assert "TailMarker" not in ranked  # never detected in these cells
    assert set(ranked[:3]) == set(TREND_GENES)

    with pytest.raises(TrendError, match="'nope'"):
        trend_genes(a, "dpt_pseudotime", groups_column="stage", groups=["early", "nope"])
    a.obs["tiny"] = pd.Categorical(np.where(np.arange(a.n_obs) < 20, "t", "rest"))
    with pytest.raises(TrendError, match="Only 20 cells .* at least 30"):
        trend_genes(a, "dpt_pseudotime", groups_column="tiny", groups=["t"])
    with pytest.raises(TrendError, match="not found"):
        trend_genes(a, "dpt_pseudotime", groups_column="nope", groups=["early"])


def test_cells_without_pseudotime_are_left_out_and_flat_pseudotime_is_refused():
    a = _trend_adata()
    pt = a.obs["dpt_pseudotime"].to_numpy().copy()
    pt[:5] = np.inf
    pt[5:8] = np.nan
    a.obs["dpt_pseudotime"] = pt
    assert trend_genes(a, "dpt_pseudotime")["n_cells"] == a.n_obs - 8
    a.obs["dpt_pseudotime"] = 0.5
    with pytest.raises(TrendError, match="distinct"):
        trend_genes(a, "dpt_pseudotime")


def test_tests_log_normalised_raw_not_scaled_x_and_sparse_equals_dense():
    a = _trend_adata()
    lognorm = a.X.copy()
    # As after the pipeline: .raw log-normalised, X z-scored.
    a.raw = ad.AnnData(sp.csr_matrix(lognorm), var=a.var.copy(), obs=a.obs[[]].copy())
    a.X = ((lognorm - lognorm.mean(0)) / (lognorm.std(0) + 1e-6)).astype("float32")
    res_sparse = trend_genes(a, "dpt_pseudotime", n_top=10)
    assert res_sparse["expression"] == "log-normalised .raw"
    assert all(v >= 0 for g in res_sparse["genes"] for v in g["mean"])

    dense = ad.AnnData(lognorm, var=a.var.copy(), obs=a.obs.copy())
    res_dense = trend_genes(dense, "dpt_pseudotime", n_top=10)
    assert res_dense["expression"] == "log-normalised X"
    assert _ranked(res_sparse) == _ranked(res_dense)
    for gs, gd in zip(res_sparse["genes"], res_dense["genes"]):
        assert gs["f_stat"] == pytest.approx(gd["f_stat"], rel=1e-6)
        assert gs["mean"] == pytest.approx(gd["mean"], abs=1e-4)


def test_counts_are_log_normalised_first():
    a = _trend_adata()
    counts = np.round(np.expm1(a.X) * 3).astype("float32")
    a.X = sp.csr_matrix(counts)
    res = trend_genes(a, "dpt_pseudotime", n_top=5)
    assert res["expression"].startswith("shifted log of the counts")
    assert set(_ranked(res)[:3]) == set(TREND_GENES)


def _register(dm, adata) -> str:
    d = Path(dm.data_dir) / "uploads" / "ptg"
    d.mkdir(parents=True)
    (d / "metadata.json").write_text(json.dumps({"id": "ptg", "name": "ptg", "status": "ready"}))
    adata.write_h5ad(d / "data.h5ad")
    return "ptg"


async def test_endpoint_ranks_genes_and_changes_nothing(client, dataset_manager):
    a = _trend_adata()
    a.obs["score"] = 1.0
    did = _register(dataset_manager, a)
    url = f"{API}/datasets/{did}/trajectory/dpt_pseudotime/trend-genes"
    r = await client.post(url, json={"n_top": 5, "n_bins": 10})
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(_ranked(body)[:3]) == set(TREND_GENES) and len(body["bins"]["pseudotime"]) == 10

    r = await client.post(url, json={"groups_column": "stage", "groups": ["early", "late"]})
    assert r.status_code == 200 and r.json()["n_cells"] == N_MAIN
    r = await client.post(url, json={"groups_column": "stage", "groups": ["nope"]})
    assert r.status_code == 400 and "nope" in r.json()["detail"]
    assert (await client.post(url, json={"n_top": 0})).status_code == 422
    r = await client.post(f"{API}/datasets/{did}/trajectory/score/trend-genes", json={})
    assert r.status_code == 404
    # A read: no derived file is written.
    assert not (Path(dataset_manager.data_dir) / "derived" / did).exists()
