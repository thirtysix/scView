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
    PARAMS_KEY,
    RUNS_KEY,
    TrajectoryError,
    compute_dpt,
    compute_paga,
    detached_tail,
    less_specialised_end,
    root_cell,
    root_column_name,
    root_label,
    root_position,
    tree_ends,
    velocity_against_root,
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
    compute_paga(a, "stage")  # early - mid - late (the island has no tree edge)
    first = compute_dpt(a, "stage", "early")
    assert first["warnings"] == []
    assert first["root_checks"]["paga"]["position"] == "end"
    got = compute_dpt(a, "stage", "mid")
    assert any("'mid' sits inside the PAGA tree of stage (linked to 'early', 'late')" in w
               and "opens 2 large branches" in w for w in got["warnings"])
    assert got["root_checks"]["paga"]["position"] == "inside"
    # A record whose groups no longer match is not used.
    a.obs["stage"] = a.obs["stage"].cat.rename_categories({"mid": "middle"})
    stale = compute_dpt(a, "stage", "middle")
    assert not any("PAGA tree" in w for w in stale["warnings"])
    assert "changed since PAGA" in stale["root_checks"]["paga"]["skipped"]


def test_root_position_tells_an_end_a_path_and_a_branch_point():
    # paul15-like: MEP between three large branches and a small one.
    rec = {"groups": ["MEP", "Ery", "GMP", "Mk", "Lymph", "Ery2"], "sizes": [10, 400, 400, 80, 5, 100],
           "tree": [[0, 1], [0, 2], [0, 3], [0, 4], [1, 5]]}
    assert root_position(rec, "MEP") == {"position": "branch point", "branches": [0.503, 0.402, 0.08, 0.005]}
    assert root_position(rec, "Ery")["position"] == "inside"  # one path: MEP's side and Ery2
    assert root_position(rec, "Ery2")["position"] == "end"
    assert root_position(rec, "nope") is None
    rec["sizes"] = [10, 400, 400, 30, 5, 100]  # Mk's branch under 5%: two large branches only
    assert root_position(rec, "MEP")["position"] == "inside"
    assert tree_ends(rec) == ["GMP", "Mk", "Lymph", "Ery2"]


def test_less_specialised_end_needs_both_signals_and_a_clear_margin():
    # Medians measured on scView's processing (pancreas: Ductal true root; erythroid: BP1).
    stats = {"Ductal": {"genes": 2840.5, "top_share": 0.130}, "Alpha": {"genes": 2343.0, "top_share": 0.164},
             "Delta": {"genes": 2257.0, "top_share": 0.202}}
    assert less_specialised_end(stats, "Ductal", list(stats)) is None
    assert less_specialised_end(stats, "Alpha", list(stats)) == "Ductal"
    assert less_specialised_end(stats, "Delta", list(stats)) == "Ductal"
    ery = {"BP1": {"genes": 2951.0, "top_share": 0.128}, "Ery3": {"genes": 2727.0, "top_share": 0.320}}
    assert less_specialised_end(ery, "Ery3", list(ery)) == "BP1"  # genes 1.08x, top share far lower
    near = {"a": {"genes": 1259.5, "top_share": 0.149}, "b": {"genes": 1330.0, "top_share": 0.137}}
    assert less_specialised_end(near, "a", list(near)) is None  # 1.06x and 0.012: no clear margin
    both_ways = {"a": {"genes": 1000.0, "top_share": 0.10}, "b": {"genes": 2000.0, "top_share": 0.15}}
    assert less_specialised_end(both_ways, "a", list(both_ways)) is None  # the signals disagree


def _specialising_path() -> ad.AnnData:
    """The path with raw counts: early cells spread their counts over many genes, late cells
    put most of theirs in a few (a hormone, haemoglobin)."""
    import scipy.sparse as sp

    a, t = _path_adata()
    rng = np.random.default_rng(5)
    n = a.n_obs
    tt = np.r_[t, np.full(N_ISLAND, 0.5)]
    broad = rng.poisson(np.outer(1.0 - 0.8 * tt, np.full(300, 1.0)))
    few = rng.poisson(np.outer(1 + 60 * tt, np.full(5, 1.0)))
    b = ad.AnnData(sp.csr_matrix(np.hstack([few, broad]).astype("float32")), obs=a.obs.copy(),
                   obsm={"X_pca": a.obsm["X_pca"]})
    b.obs_names = a.obs_names
    return b


def test_a_root_at_the_more_specialised_end_is_flagged():
    a = _specialising_path()
    compute_paga(a, "stage")
    right = compute_dpt(a, "stage", "early")
    assert right["warnings"] == []
    spec = right["root_checks"]["specialisation"]
    assert spec["early"]["genes"] > spec["late"]["genes"] and spec["early"]["top_share"] < spec["late"]["top_share"]
    wrong = compute_dpt(a, "stage", "late")
    assert any(w.startswith("'early', another end of the PAGA tree of stage, looks less specialised than 'late'")
               for w in wrong["warnings"])
    # Not in the recipe: the record's warnings say it.
    assert "root_checks" not in provenance.read_provenance(a)["history"][-1]["params"]


def test_an_end_of_another_lineage_is_not_a_candidate_start():
    """Only ends on the root's own lineage are compared (Reck 2025 kidney: from PT S1, medullary
    IC-A off the PT lineage detects more genes but is not where the PT process starts)."""
    from scview.core.trajectory import root_warnings

    a = _specialising_path()
    compute_paga(a, "stage")
    params = compute_dpt(a, "stage", "late")
    assert set(params["root_checks"]["lineage"]["lineage"]) >= {"early", "mid", "late"}
    pt = a.obs[DPT_COLUMN].to_numpy()
    iroot = int(a.obs_names.get_loc(params["root_cell"]))
    found, checks = root_warnings(a, pt, iroot, "x", lineage_of=lambda col, group: {"mid", "late"})
    assert not any("looks less specialised" in w for w in found)
    assert checks["paga"]["ends_off_lineage"] == ["early"]
    found, _ = root_warnings(a, pt, iroot, "x", lineage_of=lambda col, group: None)  # unreadable: all ends
    assert any(w.startswith("'early', another end") for w in found)


def test_specialisation_without_readable_counts_is_skipped_not_fatal(monkeypatch):
    from scview.core import normalization, trajectory

    a = _specialising_path()
    assert set(trajectory.specialisation(a, "stage", ["early", "late"])) == {"early", "late"}

    class Unreadable:
        def __getitem__(self, rows):
            raise OSError("row reads failed")

    monkeypatch.setattr(normalization, "find_counts", lambda adata, read_only=False: (Unreadable(), "x"))
    assert trajectory.specialisation(a, "stage", ["early", "late"]) is None
    compute_paga(a, "stage")
    assert compute_dpt(a, "stage", "late")["root_checks"]["specialisation"] == {"skipped": "no raw counts to read"}


def test_velocity_against_root_reads_the_direction_check():
    def check(*rows):
        return {"checked": True, "n_forward": sum(r[2] == "forward" for r in rows), "embedding": "X_umap",
                "transitions": [{"from": f, "to": t, "status": s, "mean_cos": c, "n_cells": 50}
                                for f, t, s, c in rows]}

    # Pancreas from Ductal: all forward. From Pre-endocrine: its own edge back to Ngn3 high.
    assert not velocity_against_root(check(("Ductal", "Ngn3", "forward", 0.12), ("Ngn3", "Pre", "forward", 0.36)),
                                     "Ductal")["against"]
    pre = velocity_against_root(check(("Pre", "Ngn3", "backward", -0.73), ("Pre", "Beta", "forward", 0.42),
                                      ("Ngn3", "Ductal", "unclear", -0.07)), "Pre")
    assert pre["against"] and pre["root_edges_backward"] == ["Ngn3"]
    # Erythroid from BP1: one backward step late on (MURK genes) does not accuse the root.
    bp1 = check(("BP1", "BP2", "forward", 0.25), ("BP2", "E1", "forward", 0.2), ("E1", "E2", "forward", 0.09),
                ("E2", "E3", "backward", -0.17))
    assert not velocity_against_root(bp1, "BP1")["against"]
    # From Ery3: its own edge looks forward (the reversed step), the rest backward.
    e3 = check(("E3", "E2", "forward", 0.27), ("E2", "E1", "unclear", -0.05), ("E1", "BP2", "backward", -0.08),
               ("BP2", "BP1", "backward", -0.21))
    assert velocity_against_root(e3, "E3")["against"]
    assert velocity_against_root({"checked": False}, "E3") is None and velocity_against_root(None, "E3") is None


def test_the_velocity_check_adds_its_warning_and_never_blocks():
    a, _ = _path_adata()
    compute_paga(a, "stage")
    seen = []

    def backward(column, group):
        seen.append((column, group))
        return {"checked": True, "n_forward": 0, "embedding": "X_umap", "transitions": [
            {"from": group, "to": "mid", "status": "backward", "mean_cos": -0.4, "n_cells": 60}]}

    got = compute_dpt(a, "stage", "late", velocity_check=backward)
    assert seen == [("stage", "late")]
    assert any(w.startswith("The RNA velocity arrows run against this root along the PAGA tree of stage: "
                            "1 transition backward (late -> mid (-0.40)), 0 forward. The arrows flow into 'late' "
                            "from 'mid'.") for w in got["warnings"])
    assert got["root_checks"]["velocity"]["against"] is True

    def broken(column, group):
        raise RuntimeError("no arrows")

    fine = compute_dpt(a, "stage", "early", velocity_check=broken)
    assert not any("velocity" in w for w in fine["warnings"]) and "velocity" not in fine["root_checks"]


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

    def spy(adata, column, value, **kw):
        seen["busy"] = dataset_manager.is_busy(did)
        return real(adata, column, value, **kw)

    monkeypatch.setattr(traj_api, "compute_dpt", spy)
    r = await client.post(
        f"{API}/datasets/{did}/trajectory/dpt", json={"root_column": "stage", "root_value": "early"}
    )
    assert r.status_code == 200, r.text
    assert seen["busy"] is True
    assert not dataset_manager.is_busy(did)


def test_dpt_from_one_cell_keeps_its_own_column_and_record():
    a, t = _path_adata()
    tip = str(a.obs_names[int(np.argmin(t))])  # the first cell along the path
    group = compute_dpt(a, "stage", "early")
    p = compute_dpt(a, root_cell=tip)
    assert p["root_kind"] == "cell" and p["root_cell"] == tip and p["root_rule"] == "the cell given"
    assert p["root_column"] is None and p["root_value"] is None
    assert p["root_named_column"] == root_column_name(root_cell=tip) == f"scview_dpt_cell_{tip}"
    assert group["root_kind"] == "group" and group["root_cell"] != tip  # the group's central cell
    pt = a.obs[DPT_COLUMN].to_numpy()
    assert pt[a.obs_names.get_loc(tip)] == 0.0
    assert spearmanr(pt[:N_PATH], t).statistic > 0.95  # from the very start, every cell in order
    runs = json.loads(a.uns[RUNS_KEY])
    assert set(runs) == {"scview_dpt_early", f"scview_dpt_cell_{tip}"}
    assert runs[f"scview_dpt_cell_{tip}"]["root_kind"] == "cell"
    assert provenance.read_provenance(a)["history"][-1]["params"]["root_cell"] == tip
    assert root_label(p) == f"cell '{tip}'" and root_label(group) == "stage = early"
    # A group named like a cell's column cannot take the cell root's column.
    assert root_column_name("cell x") != root_column_name(root_cell="x")


def test_dpt_root_choice_is_one_group_or_one_cell():
    a, _ = _path_adata()
    for kw in ({}, {"root_column": "stage"}, {"root_column": "stage", "root_value": "early", "root_cell": "c1"},
               {"root_value": "early", "root_cell": "c1"}):
        with pytest.raises(TrajectoryError, match="not both"):
            compute_dpt(a, **kw)
    with pytest.raises(TrajectoryError, match="No cell is named 'nope'"):
        compute_dpt(a, root_cell="nope")
    with pytest.raises(TrajectoryError, match="cell 'c310' sit in a disconnected part"):
        compute_dpt(a, root_cell=f"c{N_PATH + 10}")  # on the island


def test_a_cell_root_inside_the_paga_tree_is_checked_as_its_group():
    a, t = _path_adata()
    compute_paga(a, "stage")
    mid = str(a.obs_names[int(np.argmin(np.abs(t - 0.5)))])
    got = compute_dpt(a, root_cell=mid)["warnings"]
    assert any("'mid' sits inside the PAGA tree of stage" in w for w in got)


def test_a_cell_root_replays_from_that_cell():
    from scview.core import replay_script

    a, t = _path_adata()
    tip = str(a.obs_names[int(np.argmin(t))])
    compute_dpt(a, "stage", "late")
    compute_dpt(a, root_cell=tip)
    script = replay_script.build_script(a)
    compile(script, "replay.py", "exec")
    assert f"get_loc('{tip}')" in script and f"adata.obs['scview_dpt_cell_{tip}']" in script
    assert "# Root: the cell picked by hand." in script
    assert "# Root: the most central cell of stage = late in diffusion space." in script


async def test_endpoint_takes_one_cell_as_the_root(client, dataset_manager):
    a, t = _path_adata()
    did, _ = _register(dataset_manager, a)
    url = f"{API}/datasets/{did}/trajectory/dpt"
    for body in ({}, {"root_column": "stage"}, {"root_column": "stage", "root_value": "early", "root_cell": "c0"},
                 {"root_value": "early"}):
        r = await client.post(url, json=body)
        assert r.status_code == 422, body
        assert "root_cell (one cell's name)" in r.json()["detail"]
    r = await client.post(url, json={"root_cell": "nope"})
    assert r.status_code == 404 and "No cell is named 'nope'" in r.json()["detail"]

    tip = int(np.argmin(t))
    named = (await client.get(f"{API}/datasets/{did}/trajectory/cell-name", params={"index": tip})).json()
    assert named["name"] == f"c{tip}"
    r = await client.get(f"{API}/datasets/{did}/trajectory/cell-name", params={"index": 10_000})
    assert r.status_code == 404
    r = await client.post(url, json={"root_cell": named["name"]})
    assert r.status_code == 200, r.text
    assert r.json()["root_kind"] == "cell" and r.json()["root_named_column"] == f"scview_dpt_cell_c{tip}"
    listed = (await client.get(f"{API}/datasets/{did}/trajectory")).json()
    assert listed["computed"]["root_cell"] == f"c{tip}" and f"scview_dpt_cell_c{tip}" in listed["runs"]
    assert f"scview_dpt_cell_c{tip}" in listed["pseudotime_columns"]


async def test_endpoint_checks_the_root_against_a_saved_velocity_run(client, dataset_manager):
    """The arrows of the velocity fixture point from early to late along the plot's x axis."""
    from scview.core import velocity_input as vi
    from tests.test_hvg_sensitivity import _register as register_named
    from tests.test_velocity import _run as run_velocity
    from tests.velocity_fixture import splicing_counts

    a = splicing_counts(300, 260)
    bare = a.copy()
    for k in vi.LAYERS:
        del bare.layers[k]
    bare.obsm["X_pca"] = np.c_[a.obsm["X_umap"], np.random.default_rng(0).normal(0, 0.3, (a.n_obs, 8))]
    did = register_named(dataset_manager, bare, "dptvel")
    holder = ad.AnnData(a.X.copy(), obs=a.obs[[]].copy(), var=a.var.copy(),
                        layers={k: a.layers[k] for k in vi.LAYERS})
    vi.adopt_anndata(holder, did, dataset_manager.data_dir, file_name="x")
    url = f"{API}/datasets/{did}/trajectory"
    # Before a velocity run: no velocity verdict.
    assert (await client.post(f"{url}/paga", json={"groupby": "clusters"})).status_code == 200
    r = (await client.post(f"{url}/dpt", json={"root_column": "clusters", "root_value": "late"})).json()
    assert "velocity" not in r["root_checks"]

    events = await run_velocity(client, did, group_by="clusters", direction={"order": ["early", "late"]})
    assert [k for k, _ in events][-1] == "complete"
    wrong = (await client.post(f"{url}/dpt", json={"root_column": "clusters", "root_value": "late"})).json()
    assert any(w.startswith("The RNA velocity arrows run against this root along the PAGA tree of clusters: "
                            "1 transition backward (late -> early") for w in wrong["warnings"]), wrong["warnings"]
    assert wrong["root_checks"]["velocity"]["root_edges_backward"] == ["early"]
    right = (await client.post(f"{url}/dpt", json={"root_column": "clusters", "root_value": "early"})).json()
    assert not any("velocity" in w.lower() for w in right["warnings"])
    assert right["root_checks"]["velocity"] == {**right["root_checks"]["velocity"], "against": False, "n_backward": 0}
    # A cell root is checked as its group: the first cell along the line sits in 'early'.
    cell = (await client.post(f"{url}/dpt", json={"root_cell": str(a.obs_names[0])})).json()
    assert cell["root_checks"]["paga"]["root_group"] == "early" and not cell["root_checks"]["velocity"]["against"]
