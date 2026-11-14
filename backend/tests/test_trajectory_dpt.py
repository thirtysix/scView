"""In-app diffusion pseudotime (core/trajectory.py + POST /trajectory/dpt)."""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from scview.core import provenance
from scview.core.trajectory import (
    DPT_COLUMN,
    PAGA_KEY,
    PARAMS_KEY,
    RUNS_KEY,
    TrajectoryError,
    compute_dpt,
    detached_tail,
    root_cell,
)

API = "/api/v1"
N_PATH, N_ISLAND = 300, 25


def test_root_cell_is_the_groups_most_central_cell():
    dcs = np.array([[0.0, 0.0], [1.0, 0.1], [2.0, 0.0], [9.0, 0.0], [5.0, 5.0]])
    group = np.array([True, True, True, True, False])
    # The median of the group is near cell 1/2; the outlier (3) and the
    # non-group cell (4) do not pull it, and eigenvalue 1 is clipped, not inf.
    assert root_cell(dcs, np.array([0.9, 0.5]), group) in (1, 2)
    assert root_cell(dcs, np.array([1.0, 0.5]), group) in (1, 2)
    # DPT's weighting decides which component "central" is measured along:
    # cell 1 is central on DC1 but an outlier on DC2.
    pts = np.array([[0.0, 0.0], [1.0, 10.0], [2.0, 0.0]])
    everyone = np.ones(3, bool)
    assert root_cell(pts, np.array([0.99, 0.01]), everyone) == 1
    assert root_cell(pts, np.array([0.01, 0.99]), everyone) in (0, 2)


def _line_adata(t: np.ndarray, stage: list[str]) -> ad.AnnData:
    """Cells along a curved 1-D path in PCA space at positions ``t``, plus a small
    island far away that no path from the main data reaches."""
    rng = np.random.default_rng(0)
    n = len(t)
    pca = rng.normal(0, 0.15, size=(n + N_ISLAND, 10))
    pca[:n, 0] += 20 * t
    pca[:n, 1] += 4 * np.sin(np.pi * t)
    pca[n:, :2] += 500.0
    a = ad.AnnData(np.zeros((n + N_ISLAND, 3), dtype="float32"))
    a.obs_names = [f"c{i}" for i in range(a.n_obs)]
    a.obs["stage"] = pd.Categorical(stage + ["island"] * N_ISLAND)
    a.obsm["X_pca"] = pca
    a.obs["dpt_pseudotime"] = 0.5  # an imported column that must survive
    return a


def _path_adata() -> tuple[ad.AnnData, np.ndarray]:
    """early -> mid -> late along the path."""
    t = np.sort(np.random.default_rng(1).uniform(0, 1, N_PATH))
    stage = np.where(t < 0.25, "early", np.where(t < 0.75, "mid", "late")).tolist()
    return _line_adata(t, stage), t


def _medians(a) -> dict[str, float]:
    return a.obs.groupby("stage", observed=True)[DPT_COLUMN].median().to_dict()


def test_dpt_orders_the_path_from_the_root_group():
    a, t = _path_adata()
    params = compute_dpt(a, "stage", "early")
    pt = a.obs[DPT_COLUMN].to_numpy()
    outside = t >= 0.25  # beyond the root group, pseudotime follows the path
    assert spearmanr(pt[:N_PATH][outside], t[outside]).statistic > 0.95
    m = _medians(a)
    assert m["early"] < m["mid"] < m["late"]
    assert np.isinf(pt[N_PATH:]).all() and params["n_unreachable"] == N_ISLAND
    root = int(a.obs_names.get_loc(params["root_cell"]))
    assert t[root] < 0.25 and pt[root] == 0.0
    assert params["graph"] == "kNN graph on X_pca (k=15)"
    # Only the new column (+ its params) is added; nothing else is replaced.
    assert (a.obs["dpt_pseudotime"] == 0.5).all()
    assert "X_diffmap" not in a.obsm and "neighbors" not in a.uns
    assert json.loads(a.uns[PARAMS_KEY])["root_value"] == "early"
    assert provenance.read_provenance(a)["history"][-1]["step"] == "pseudotime"

    compute_dpt(a, "stage", "late")  # the other end reverses the order
    m = _medians(a)
    assert m["late"] < m["mid"] < m["early"]


def test_dpt_from_a_branch_point_grows_both_ways():
    # A progenitor group in the middle of the data, with a branch either side.
    t = np.sort(np.random.default_rng(2).uniform(-1, 1, N_PATH))
    stage = np.where(t < -0.2, "left", np.where(t > 0.2, "right", "progenitor")).tolist()
    a = _line_adata(t, stage)
    compute_dpt(a, "stage", "progenitor")
    pt = a.obs[DPT_COLUMN].to_numpy()[:N_PATH]
    assert spearmanr(pt, np.abs(t)).statistic > 0.9
    m = _medians(a)
    assert m["progenitor"] < min(m["left"], m["right"])


def test_dpt_reuses_the_existing_neighbour_graph():
    import scanpy as sc

    a, t = _path_adata()
    sc.pp.neighbors(a, n_neighbors=20, use_rep="X_pca")
    nnz = a.obsp["connectivities"].nnz
    params = compute_dpt(a, "stage", "early")
    assert params["graph"] == "existing neighbour graph (use_rep=X_pca, k=20)"
    assert a.obsp["connectivities"].nnz == nnz and "X_diffmap" not in a.obsm
    assert spearmanr(a.obs[DPT_COLUMN].to_numpy()[:N_PATH], t).statistic > 0.95


def test_dpt_refuses_what_it_cannot_compute():
    a, _ = _path_adata()
    with pytest.raises(TrajectoryError, match="disconnected"):
        compute_dpt(a, "stage", "island")  # 25 cells on their own
    with pytest.raises(TrajectoryError, match="No cells"):
        compute_dpt(a, "stage", "nope")
    del a.obsm["X_pca"]
    with pytest.raises(TrajectoryError, match="PCA"):
        compute_dpt(a, "stage", "early")


def test_detached_tail_finds_a_gap_not_a_long_tail():
    rng = np.random.default_rng(0)
    main = rng.uniform(0, 0.02, 980)
    assert detached_tail(np.r_[main, rng.uniform(0.92, 1.0, 20)]).sum() == 20
    # A long continuous tail (paul15's dendritic cells): no gap wider than what lies below.
    assert detached_tail(np.r_[rng.uniform(0, 0.3, 950), np.linspace(0.3, 1.0, 50)]) is None
    # A gap with more than 5% of the cells past it is a part of the data, not an island.
    assert detached_tail(np.r_[main, rng.uniform(0.92, 1.0, 100)]) is None
    assert detached_tail(np.zeros(100)) is None


def test_dpt_leaves_a_weakly_linked_island_out_of_the_scale(monkeypatch):
    """Pijuan-Sala's erythroid cells after Harmony: 18 weakly linked cells took 0.92-1.0 and
    the other 9,797 were squeezed into 0-0.016. Here the first diffusion is made to give
    that picture for 15 path cells; they are left out and the rest diffused again."""
    from scview.core import trajectory

    a, t = _path_adata()
    island = np.flatnonzero(t > 0.95)[-15:]
    real, calls = trajectory._diffuse, []

    def first_gapped(work, reach, in_group):
        pt, root = real(work, reach, in_group)
        calls.append(int(reach.sum()))
        if len(calls) == 1:
            pt = pt * 0.02
            pt[np.isin(np.flatnonzero(reach), island)] = np.linspace(0.92, 1.0, len(island))
        return pt, root

    monkeypatch.setattr(trajectory, "_diffuse", first_gapped)
    params = compute_dpt(a, "stage", "early")
    pt = a.obs[DPT_COLUMN].to_numpy()
    assert calls == [N_PATH, N_PATH - 15]  # the islands of _line_adata stay unreachable
    assert params["n_detached"] == 15 and params["n_unreachable"] == N_ISLAND
    assert np.isnan(pt[island]).all() and np.isinf(pt[N_PATH:]).all()
    main = np.setdiff1d(np.arange(N_PATH), island)
    assert pt[main].max() == 1.0 and np.median(pt[main]) > 0.2  # the path fills the scale again
    assert spearmanr(pt[main], t[main]).statistic > 0.95
    w = params["warnings"]
    assert len(w) == 1 and w[0].startswith("15 cells hang on the rest of the graph by weak links: from this root "
                                           "they came out at 0.92-1.00 while the other 285 took 0-0.02")


def test_each_root_keeps_its_column_and_a_reversed_root_is_flagged():
    a, t = _path_adata()
    first = compute_dpt(a, "stage", "early")
    assert first["root_named_column"] == "scview_dpt_early" and first["warnings"] == []
    second = compute_dpt(a, "stage", "late")
    assert {"scview_dpt_early", "scview_dpt_late", DPT_COLUMN} <= set(a.obs.columns)
    assert (a.obs[DPT_COLUMN] == a.obs["scview_dpt_late"]).all()
    assert set(json.loads(a.uns[RUNS_KEY])) == {"scview_dpt_early", "scview_dpt_late"}
    # Reversed against the earlier run (and the imported dpt_pseudotime is constant: skipped).
    assert any("runs against 'scview_dpt_early'" in w for w in second["warnings"])
    again = compute_dpt(a, "stage", "late")  # its own earlier column is not "another" pseudotime
    assert sum("runs against" in w for w in again["warnings"]) == 1


def test_a_root_inside_the_paga_tree_is_flagged():
    a, _ = _path_adata()
    a.uns[PAGA_KEY] = json.dumps({"groupby": "stage", "groups": ["early", "mid", "late"],
                                  "tree": [[0, 1], [1, 2]], "edges": [], "sizes": [1, 1, 1]})
    assert compute_dpt(a, "stage", "early")["warnings"] == []
    got = compute_dpt(a, "stage", "mid")["warnings"]
    assert any("'mid' sits inside the PAGA tree of stage (linked to 'early', 'late')" in w for w in got)


def _register(dm, adata) -> tuple[str, Path]:
    d = Path(dm.data_dir) / "uploads" / "traj"
    d.mkdir(parents=True)
    (d / "metadata.json").write_text(json.dumps({"id": "traj", "name": "traj", "status": "ready"}))
    adata.write_h5ad(d / "data.h5ad")
    return "traj", d / "data.h5ad"


async def test_endpoint_saves_the_column_and_the_panel_lists_it(client, dataset_manager):
    a, _ = _path_adata()
    did, original = _register(dataset_manager, a)
    r = await client.post(
        f"{API}/datasets/{did}/trajectory/dpt", json={"root_column": "stage", "root_value": "early"}
    )
    assert r.status_code == 200, r.text
    assert r.json()["column"] == DPT_COLUMN and r.json()["n_unreachable"] == N_ISLAND

    listed = (await client.get(f"{API}/datasets/{did}/trajectory")).json()
    assert DPT_COLUMN in listed["pseudotime_columns"]
    assert listed["computed"]["root_value"] == "early"
    values = (await client.get(f"{API}/datasets/{did}/trajectory/{DPT_COLUMN}")).json()
    assert values["values"][N_PATH:] == [None] * N_ISLAND and values["min"] == 0.0

    # Written to the derived file; the upload is never touched.
    derived = Path(dataset_manager.data_dir) / "derived" / did / "data.h5ad"
    assert DPT_COLUMN in ad.read_h5ad(derived).obs.columns
    assert DPT_COLUMN not in ad.read_h5ad(original).obs.columns
    assert not list(derived.parent.glob("*.partial"))


async def test_endpoint_rejects_bad_requests(client, dataset_manager):
    a, _ = _path_adata()
    del a.obsm["X_pca"]
    did, _ = _register(dataset_manager, a)
    url = f"{API}/datasets/{did}/trajectory/dpt"
    r = await client.post(url, json={"root_column": "nope", "root_value": "early"})
    assert r.status_code == 404
    r = await client.post(url, json={"root_column": "stage", "root_value": "nope"})
    assert r.status_code == 400
    r = await client.post(url, json={"root_column": "stage", "root_value": "early"})
    assert r.status_code == 400 and "PCA" in r.json()["detail"]


async def test_endpoint_runs_with_the_dataset_to_itself(client, dataset_manager, monkeypatch):
    # Like a pipeline run: the dataset is marked busy for the whole thread, so a
    # client that disconnects mid-write can't hand the lock to another edit early.
    from scview.api.v1 import trajectory as traj_api

    a, _ = _path_adata()
    did, _ = _register(dataset_manager, a)
    seen = {}
    real = traj_api.compute_dpt

    def spy(adata, column, value):
        seen["busy"] = dataset_manager.is_busy(did)
        return real(adata, column, value)

    monkeypatch.setattr(traj_api, "compute_dpt", spy)
    r = await client.post(
        f"{API}/datasets/{did}/trajectory/dpt", json={"root_column": "stage", "root_value": "early"}
    )
    assert r.status_code == 200, r.text
    assert seen["busy"] is True
    assert not dataset_manager.is_busy(did)
