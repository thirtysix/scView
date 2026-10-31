"""Spatial co-occurrence (core/spatial_cooccurrence.py, core/cooccurrence_kernel.py, routes in
api/v1/spatial_domains.py): the score against squidpy and by hand, sections, units, the sample,
the stream and Cancel, stale results, large datasets, replay, the co-pilot's fact."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from scview.core import assistant, claims, feature_guide, large_layout, large_mode, replay_script
from scview.core import cooccurrence_kernel as kernel
from scview.core import spatial_cooccurrence as cc
from scview.core import spatial_domains as sd
from scview.core.rag import router as rag_router
from tests.large_fixture import register, small_twin
from tests.test_hvg_sensitivity import _register
from tests.test_wnn_api import _events

_FIX = Path(__file__).parent / "fixtures"
_spec = importlib.util.spec_from_file_location("cooccur_synthetic", _FIX / "spatial_domains" / "synthetic.py")
synthetic = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(synthetic)

API = "/api/v1"


@pytest.fixture(scope="module")
def kernels():
    return kernel.compiled_kernels()


def _points(a) -> tuple[np.ndarray, np.ndarray, list[str]]:
    codes, labels = sd._labels_of(a, "stripe")
    return np.asarray(a.obsm["spatial"], dtype=np.float32), codes.astype(np.int32), labels


def _brute(xy: np.ndarray, codes: np.ndarray, n_labels: int, upper: np.ndarray) -> np.ndarray:
    """Ordered pairs at 0 < d <= each upper bound, from the full distance matrix."""
    d = np.sqrt(((xy[:, None].astype(np.float64) - xy[None].astype(np.float64)) ** 2).sum(-1))
    out = np.zeros((n_labels, n_labels, upper.size), dtype=np.int64)
    for t, top in enumerate(upper):
        i, j = np.nonzero((d <= top) & (d > 0))
        np.add.at(out[:, :, t], (codes[i], codes[j]), 1)
    return out


# ---------------------------------------------------------------------------
# The score
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tech", ["cells", "visium"])
def test_equals_squidpy_on_one_section(tech, kernels):
    ref = json.loads((_FIX / "spatial_cooccurrence" / "cooccur_squidpy.json").read_text())[tech]
    a = synthetic.tissue(tech, n_side=20, seed=3)
    xy, codes, labels = _points(a)
    interval = kernel.default_interval(xy, np.zeros(a.n_obs, dtype=np.int32), 20)
    np.testing.assert_array_equal(interval, np.asarray(ref["interval"], dtype=np.float32))
    score, _, _ = kernel.compute(xy, codes, len(labels), np.zeros(a.n_obs, dtype=np.int32), interval, kernels=kernels)
    np.testing.assert_allclose(score, ref["occ"], atol=2e-5)


def test_squidpy_splits_equal_its_mean_of_row_splits(kernels):
    ref = json.loads((_FIX / "spatial_cooccurrence" / "cooccur_squidpy.json").read_text())["cells_splits"]
    a = synthetic.tissue("cells", n_side=20, seed=3)
    xy, codes, labels = _points(a)
    interval = np.asarray(ref["interval"], dtype=np.float32)
    score, count, _ = kernel.split_scores(xy, codes, len(labels), interval, 2, kernels, n_splits=3)
    np.testing.assert_allclose(score, ref["occ"], atol=2e-5)
    # The summed counts are every pair once: the same as one split, while the score is not.
    whole, _, _ = kernel.compute(xy, codes, len(labels), np.zeros(a.n_obs, dtype=np.int32), interval, kernels=kernels)
    pooled = kernel.cooccurrence_scores(count)
    np.testing.assert_allclose(pooled, whole, atol=1e-12)
    assert np.abs(score - whole).max() > 0.01


def test_squidpys_automatic_split_rule():
    assert kernel.n_splits_of(22_897) == 1
    assert kernel.n_splits_of(22_898) == 12  # 22,898 / 12 is the first share under 2,048
    assert kernel.n_splits_of(167_780) == 82
    assert cc.SPLITS_ABOVE_OBS == 22_897


def test_counts_are_cumulative_ordered_pairs_by_brute_force(kernels):
    rng = np.random.default_rng(5)
    xy = rng.uniform(0, 500, (300, 2)).astype(np.float32)
    codes = rng.integers(0, 4, 300).astype(np.int32)
    interval = np.array([3, 20, 41.5, 90, 250, 900], dtype=np.float32)
    score, count, _ = kernel.compute(xy, codes, 4, np.zeros(300, dtype=np.int32), interval, kernels=kernels)
    np.testing.assert_array_equal(count, _brute(xy, codes, 4, interval[1:]))
    assert (np.diff(count, axis=2) >= 0).all()
    np.testing.assert_allclose(score, score.transpose(1, 0, 2), atol=1e-9)
    assert score.shape == (4, 4, 5)  # the first entry is never a bound


def test_score_by_hand_and_a_pair_at_distance_zero():
    # Four points: A at 0 and 1, B at 1 (the same place as an A) and 10.
    xy = np.array([[0, 0], [1, 0], [1, 0], [10, 0]], dtype=np.float32)
    codes = np.array([0, 0, 1, 1], dtype=np.int32)
    edges = np.array([0, 2], dtype=np.float32)
    _, count, _ = kernel.compute(xy, codes, 2, np.zeros(4, dtype=np.int32), edges)
    # Within 2: A0-A1 and A0-B2; A1-B2 sit at distance 0 and are left out.
    assert count[:, :, 0].tolist() == [[2, 1], [1, 0]]
    score = kernel.cooccurrence_scores(count)[:, :, 0]
    # P(B | A) = 1/3, P(B) = 1/4.
    assert score[0, 1] == pytest.approx((1 / 3) / (1 / 4))
    assert score[1, 1] == 0.0  # no B-B pair in reach: 0 as squidpy, never a division by zero


def test_thresholds_out_of_order_come_back_in_the_order_asked(kernels):
    # squidpy's linspace can run downwards when the two lowest-sum cells are far apart.
    xy = np.random.default_rng(1).uniform(0, 100, (80, 2)).astype(np.float32)
    codes = (np.arange(80) % 2).astype(np.int32)
    up = np.array([0, 10, 50], dtype=np.float32)
    down = np.array([0, 50, 10], dtype=np.float32)
    a = kernel.compute(xy, codes, 2, np.zeros(80, dtype=np.int32), up, kernels=kernels)[1]
    b = kernel.compute(xy, codes, 2, np.zeros(80, dtype=np.int32), down, kernels=kernels)[1]
    np.testing.assert_array_equal(a[:, :, ::-1], b)


def test_bins_beyond_the_float32_edge_stay_with_the_edge():
    edges = np.array([1.0, 2.0, 1000.0], dtype=np.float32)
    table, per_unit = kernel.bin_lookup(edges)
    for d in (0.5, 1.0, 1.0001, 2.0, 2.0001, 999.9, 1000.0):
        b = table[int(np.float32(d) * per_unit)]
        while np.float32(d) > edges[b]:
            b += 1
        assert b == np.searchsorted(edges, np.float32(d), side="left")


# ---------------------------------------------------------------------------
# Sections, units, the sample
# ---------------------------------------------------------------------------


def test_sections_are_counted_apart_and_summed(kernels):
    a = synthetic.tissue("cells", n_side=12, seed=2, sections=2)
    xy, codes, labels = _points(a)
    sections = np.unique(sd._libraries(a), return_inverse=True)[1].astype(np.int32)
    interval = np.array([0, 50, 300, 1500], dtype=np.float32)
    _, count, _ = kernel.compute(xy, codes, len(labels), sections, interval, kernels=kernels)
    expected = sum(_brute(xy[sections == s], codes[sections == s], len(labels), interval[1:]) for s in (0, 1))
    np.testing.assert_array_equal(count, expected)
    pooled = _brute(xy, codes, len(labels), interval[1:])
    assert (pooled >= count).all() and pooled.sum() > count.sum()  # what pooling the coordinates would add


def test_several_sections_take_the_widest_thresholds():
    xy = np.random.default_rng(3).uniform(0, 100, (40, 2)).astype(np.float32)
    xy[20:] *= 3
    sections = np.repeat([0, 1], 20).astype(np.int32)
    interval = kernel.default_interval(xy, sections, 5)
    low = [kernel.find_min_max(xy[sections == s])[0] for s in (0, 1)]
    high = [kernel.find_min_max(xy[sections == s])[1] for s in (0, 1)]
    assert interval[0] == min(low) and interval[-1] == max(high)


def test_distance_units_by_technology():
    x = synthetic.tissue("xenium", n_side=8)
    assert cc.distance_units(x)[:3] == ("um", 1.0, None)
    v = synthetic.tissue("visium", n_side=8)  # spot centres 100 pixels apart here: 100 um between them
    assert cc.distance_units(v) == ("um", pytest.approx(1.0), None, "the 100 um between spot centres")
    v.obsm["spatial"] = v.obsm["spatial"] * 1.37
    assert cc.distance_units(v)[1] == pytest.approx(1 / 1.37)
    hd = synthetic.tissue("visium_hd", n_side=8)  # a bin's 8 um is 20 pixels
    assert cc.distance_units(hd)[:3] == ("um", pytest.approx(8 / 20), None)
    hd.uns["spatial"]["s0"]["scalefactors"]["microns_per_pixel"] = 0.2738
    assert cc.distance_units(hd)[1] == pytest.approx(0.2738)
    assert cc.distance_units(synthetic.tissue("cells", n_side=8)) == ("position units", None, None, "")


def test_sections_with_different_pixel_sizes_are_brought_to_one_scale():
    a = synthetic.tissue("visium", n_side=10, sections=2)
    in_s1 = (a.obs["library_id"] == "s1").to_numpy()
    a.obsm["spatial"][in_s1] *= 2  # the same grid at twice the pixels
    unit, per, scales, _ = cc.distance_units(a)
    assert unit == "um" and per == pytest.approx(1.0) and scales["s1"] == pytest.approx(0.5)
    p = cc.plan_cooccurrence(a, "stripe", interval=5)
    inp = cc.prepare_cooccurrence(a, p)
    # Back at one scale, the two sections are the same grid.
    np.testing.assert_allclose(inp.spatial[inp.sections == 1], inp.spatial[inp.sections == 0])
    assert any("pixel sizes differ" in n for n in p.notes)


def test_the_sample_keeps_each_labels_share_in_each_section_and_is_seeded():
    codes = np.r_[np.zeros(9000), np.ones(900), np.full(100, 2)].astype(np.int32)
    sections = (np.arange(10_000) % 2).astype(np.int32)
    rows = kernel.sample_rows(codes, sections, 1000, 7)
    assert abs(rows.size - 1000) <= 6 and (np.diff(rows) > 0).all()
    assert np.bincount(codes[rows]).tolist() == pytest.approx([900, 90, 10], abs=2)
    assert abs((sections[rows] == 0).mean() - 0.5) < 0.01
    np.testing.assert_array_equal(rows, kernel.sample_rows(codes, sections, 1000, 7))
    assert not np.array_equal(rows, kernel.sample_rows(codes, sections, 1000, 8))
    np.testing.assert_array_equal(kernel.sample_rows(codes, sections, 10_000, 7), np.arange(10_000))


def test_a_run_above_the_cap_says_so_and_keeps_the_thresholds_of_all():
    a = synthetic.tissue("cells", n_side=50)
    whole = cc.plan_cooccurrence(a, "stripe", interval=10)
    p = cc.plan_cooccurrence(a, "stripe", interval=10, max_obs=1000)
    assert p.n_used == 1000 and p.thresholds == whole.thresholds
    assert any("a sample of about 1,000 of 2,500" in n and "seed 0" in n for n in p.notes)
    inp = cc.prepare_cooccurrence(a, p)
    assert abs(inp.spatial.shape[0] - 1000) <= 4
    r = cc.run_cooccurrence(inp)
    assert r.count[:, :, -1].sum() == pytest.approx(inp.spatial.shape[0] ** 2 * 0.8, rel=0.3)


# ---------------------------------------------------------------------------
# Plan, refusals, save, stale
# ---------------------------------------------------------------------------


def test_refusals_are_one_sentence():
    a = synthetic.tissue("xenium", n_side=8)
    a.obs["score"] = np.linspace(0, 1, a.n_obs)
    a.obs["one"] = "x"
    for column, words in (("score", "holds numbers"), ("one", "at least 2"), ("nope", "not in this dataset")):
        with pytest.raises(sd.SpatialDomainsError, match=words):
            cc.plan_cooccurrence(a, column)
    with pytest.raises(sd.SpatialDomainsError, match="Unknown way to combine"):
        cc.plan_cooccurrence(a, "stripe", combine="other")
    with pytest.raises(sd.SpatialDomainsError, match="number of distances"):
        cc.plan_cooccurrence(a, "stripe", interval=1)
    with pytest.raises(sd.SpatialDomainsError, match="Distances must be"):
        cc.plan_cooccurrence(a, "stripe", distances=[5.0])
    with pytest.raises(sd.SpatialDomainsError, match="no spatial positions"):
        cc.plan_cooccurrence(ad.AnnData(np.zeros((60, 2), dtype=np.float32), obs=a.obs.iloc[:60].copy()), "stripe")
    two = synthetic.tissue("cells", n_side=8, sections=2)
    with pytest.raises(sd.SpatialDomainsError, match="one section"):
        cc.plan_cooccurrence(two, "stripe", combine=cc.SQUIDPY_SPLITS)
    few = a[:30].copy()
    with pytest.raises(sd.SpatialDomainsError, match="at least 50"):
        cc.plan_cooccurrence(few, "stripe")


def test_observations_without_a_label_are_left_out():
    a = synthetic.tissue("xenium", n_side=12)
    a.obs["stripe"] = a.obs["stripe"].astype(object)
    a.obs.iloc[:10, a.obs.columns.get_loc("stripe")] = np.nan
    a.obs["stripe"] = a.obs["stripe"].astype("category")
    p = cc.plan_cooccurrence(a, "stripe", interval=6)
    assert p.n_labelled == a.n_obs - 10 and any("10 cell(s) without a label" in n for n in p.notes)
    inp = cc.prepare_cooccurrence(a, p)
    assert inp.spatial.shape[0] == a.n_obs - 10 and (inp.codes >= 0).all()
    np.testing.assert_array_equal(inp.spatial, a.obsm["spatial"][10:].astype(np.float32))


def test_saved_where_squidpy_saves_it_and_goes_stale_as_enrichment_does():
    a = synthetic.tissue("xenium", n_side=12)
    inp = cc.prepare_cooccurrence(a, cc.plan_cooccurrence(a, "stripe", interval=8))
    res = cc.run_cooccurrence(inp)
    assert cc.cooccurrence_summary(a) == {"available": False, "columns": []}
    s = cc.save_cooccurrence(a, inp, res, cells_fingerprint="fp")
    stored = a.uns["stripe_co_occurrence"]
    assert stored["occ"].dtype == np.float32 and stored["occ"].shape == (4, 4, 7) and stored["interval"].size == 8
    assert s["available"] and s["stale"] is False and s["columns"] == ["stripe"] and len(s["distances"]) == 7
    assert s["distance_unit"] == "um" and s["distances"][0] == pytest.approx(float(stored["interval"][1]), abs=0.01)
    assert cc.cooccurrence_summary(a, "stripe", "fp")["stale"] is False
    assert "changed since this ran" in cc.cooccurrence_summary(a, "stripe", "other")["stale_reason"]
    b = a.copy()
    b.obs["stripe"] = b.obs["stripe"].cat.rename_categories(lambda x: f"{x}_")
    assert "labels in 'stripe' changed" in cc.cooccurrence_summary(b)["stale_reason"]
    c = a.copy()
    del c.obs["stripe"]
    assert "gone" in cc.cooccurrence_summary(c)["stale_reason"]
    # A second column keeps the first; asking for a column gets that one.
    a.obs["other"] = a.obs["stripe"].astype(str).str[:2].astype("category")
    inp2 = cc.prepare_cooccurrence(a, cc.plan_cooccurrence(a, "other", interval=4))
    s2 = cc.save_cooccurrence(a, inp2, cc.run_cooccurrence(inp2))
    assert s2["column"] == "other" and set(s2["columns"]) == {"stripe", "other"}
    assert cc.cooccurrence_summary(a, "stripe")["column"] == "stripe"


def test_a_slash_in_a_column_does_not_nest_the_h5ad_key():
    assert cc.squidpy_key("cell/type") == "cell_type_co_occurrence"


def test_the_estimate_follows_the_reach_in_x():
    a = synthetic.tissue("cells", n_side=40)
    far = cc.plan_cooccurrence(a, "stripe")
    near = cc.plan_cooccurrence(a, "stripe", distances=cc.even_distances(20.0, 5))
    n = a.n_obs
    assert near.visited_pairs < far.visited_pairs <= n * (n - 1) // 2
    assert near.estimate_seconds < far.estimate_seconds
    # Measured at 2 CPUs: Xenium breast 1.097e10 pairs in 35.6 s, HD 8 um 6.899e10 in 161.8 s;
    # memory: Xenium's child 184 MB at its peak plus the server's copies.
    assert cc.estimate_seconds(10_970_000_000) == pytest.approx(35.6, rel=0.2)
    assert cc.estimate_seconds(68_990_000_000) == pytest.approx(161.8, rel=0.2)
    assert cc.estimate_bytes(166_363, 20, 49) == pytest.approx(215 * 1024**2, rel=0.2)
    p = cc.plan_cooccurrence(a, "stripe", combine=cc.SQUIDPY_SPLITS)
    assert p.n_splits == 1  # under squidpy's size limit there are no splits


def test_cancel_ends_the_child_and_saves_nothing():
    a = synthetic.tissue("cells", n_side=40)
    inp = cc.prepare_cooccurrence(a, cc.plan_cooccurrence(a, "stripe"))
    stops = iter([False, True])  # through the stage, then cancelled before the first line is read
    gen = cc.iter_cooccurrence(inp, lambda: next(stops, True))
    assert next(gen)["type"] == "stage"
    with pytest.raises(sd.SpatialDomainsCancelled):
        next(gen)
    assert "stripe_co_occurrence" not in a.uns


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


async def test_api_runs_saves_and_answers(client, dataset_manager):
    did = _register(dataset_manager, synthetic.tissue("xenium", n_side=20), "co1")
    info = (await client.get(f"{API}/datasets/{did}/spatial/cooccurrence?column=stripe")).json()
    assert info["result"]["available"] is False and info["reason"] is None
    assert info["plan"]["labels"] == ["t0", "t1", "t2", "t3"] and info["plan"]["distance_unit"] == "um"
    assert info["verdict"]["level"] in ("fits", "tight", "wont_fit", "unknown") and "Co-occurrence" in info["verdict"]["sentence"]
    assert info["defaults"]["interval"] == 50 and info["running"] is False

    r = await client.post(f"{API}/datasets/{did}/spatial/cooccurrence/run-stream", json={"cluster_key": "stripe", "interval": 10})
    events = _events(r.text)
    names = [e for e, _ in events]
    assert names[0] == "start" and "stage" in names and "progress" in names and names[-1] == "complete", r.text
    assert events[0][1]["stages"][0]["stage"] == "pairs"
    saved = events[-2][1]
    assert saved["column"] == "stripe" and np.asarray(saved["occ"]).shape == (4, 4, 9) and saved["peak_rss_gb"] > 0

    dataset_manager.evict(did)
    again = (await client.get(f"{API}/datasets/{did}/spatial/cooccurrence?column=stripe")).json()
    assert again["result"]["occ"] == saved["occ"] and again["result"]["stale"] is False
    assert dataset_manager.run_reservations() == {}  # released at the end
    a = (await dataset_manager.get_or_load_dataset(did)).adata
    assert a.uns["stripe_co_occurrence"]["occ"].shape == (4, 4, 9)


async def test_api_max_distance_and_refusals(client, dataset_manager):
    did = _register(dataset_manager, synthetic.tissue("xenium", n_side=14), "co2")
    info = (await client.get(f"{API}/datasets/{did}/spatial/cooccurrence?column=stripe&interval=5&max_distance=100")).json()
    assert info["plan"]["thresholds"] == [0, 20, 40, 60, 80, 100]
    r = await client.post(f"{API}/datasets/{did}/spatial/cooccurrence/run-stream",
                          json={"cluster_key": "stripe", "interval": 5, "max_distance": 100})
    assert _events(r.text)[-2][1]["distances"] == [20, 40, 60, 80, 100]
    url = f"{API}/datasets/{did}/spatial/cooccurrence/run-stream"
    assert (await client.post(url, json={"cluster_key": "nope"})).status_code == 422
    assert (await client.post(url, json={"cluster_key": "stripe", "combine": "x"})).status_code == 422
    assert (await client.post(url, json={"cluster_key": "stripe", "interval": 1})).status_code == 422
    assert (await client.post(f"{API}/datasets/none/spatial/cooccurrence/run-stream",
                              json={"cluster_key": "stripe"})).status_code == 404
    info = (await client.get(f"{API}/datasets/{did}/spatial/cooccurrence?column=nope")).json()
    assert "not in this dataset" in info["reason"] and info["plan"] is None


async def test_a_large_dataset_runs_saves_in_place_and_equals_the_dataset_in_memory(client, dataset_manager, monkeypatch):
    t = synthetic.tissue("xenium", n_side=20)
    rng = np.random.default_rng(1)
    counts = ad.AnnData(X=sp.csr_matrix(rng.poisson(2.0, (t.n_obs, 40)).astype(np.float32)), obs=t.obs,
                        var=pd.DataFrame(index=[f"G{i}" for i in range(40)]), obsm=dict(t.obsm), uns=t.uns)
    h5ad = register(dataset_manager.data_dir, "cL", counts)
    small = _register(dataset_manager, small_twin(counts), "cS")
    sidecars = large_layout.sidecar_paths(h5ad)

    def sha(path) -> str:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    before = [sha(p) for p in sidecars]
    with h5py.File(h5ad, "r") as f:
        x_offset = f["X"]["data"].id.get_offset()

    def boom(*_a, **_k):
        raise AssertionError("A co-occurrence run read the expression matrix")

    with monkeypatch.context() as m:
        for name in ("__getitem__", "toarray", "to_memory"):
            if hasattr(large_mode.OnDiskMatrix, name):
                m.setattr(large_mode.OnDiskMatrix, name, boom)
        big = await dataset_manager.get_or_load_dataset("cL")
        assert big.is_large
        r = await client.post(f"{API}/datasets/cL/spatial/cooccurrence/run-stream", json={"cluster_key": "stripe", "interval": 8})
        large = _events(r.text)[-2][1]
        assert large["available"] is True, r.text

    r = await client.post(f"{API}/datasets/{small}/spatial/cooccurrence/run-stream", json={"cluster_key": "stripe", "interval": 8})
    assert _events(r.text)[-2][1]["occ"] == large["occ"]

    dataset_manager.evict("cL")
    big = await dataset_manager.get_or_load_dataset("cL")
    assert big.is_large
    assert big.adata.uns["stripe_co_occurrence"]["occ"].shape == (4, 4, 7)
    info = (await client.get(f"{API}/datasets/cL/spatial/cooccurrence?column=stripe")).json()
    assert info["result"]["stale"] is False and info["result"]["occ"] == large["occ"]
    assert [sha(p) for p in sidecars] == before
    with h5py.File(h5ad, "r") as f:
        assert f["X"]["data"].id.get_offset() == x_offset


# ---------------------------------------------------------------------------
# Replay, co-pilot, guide
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("settings", [{"interval": 12}, {"distances": [0.0, 40.0, 90.0, 300.0]}, {"interval": 6, "max_obs": 1000}])
def test_replay_gives_the_same_curves(tmp_path, settings):
    inp = synthetic.tissue("xenium", n_side=40 if "max_obs" in settings else 16, sections=2 if "max_obs" not in settings else 1)
    out = inp.copy()
    c = cc.prepare_cooccurrence(out, cc.plan_cooccurrence(out, "stripe", **settings))
    res = cc.run_cooccurrence(c)
    cc.save_cooccurrence(out, c, res)
    assert [x["status"] for x in replay_script.describe(out)] == ["exact"]
    src, script, dst = tmp_path / "input.h5ad", tmp_path / "replay.py", tmp_path / "replayed.h5ad"
    inp.write_h5ad(src)
    script.write_text(replay_script.build_script(out, dataset_id="t", dataset_name="test"))
    proc = subprocess.run([sys.executable, str(script), str(src), str(dst)], capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr[-3000:]
    rep = ad.read_h5ad(dst).uns["stripe_co_occurrence"]
    np.testing.assert_array_equal(rep["occ"], res.occ)
    np.testing.assert_array_equal(rep["count"], res.count)
    np.testing.assert_array_equal(rep["interval"], np.asarray(c.plan.thresholds, dtype=np.float32))


def test_replay_of_an_unreadable_record_is_a_comment():
    ctx = replay_script._Ctx(adata=None)
    assert replay_script._co_occurrence({"column": "x"}, ctx).status == replay_script.NOT_REPLAYABLE


class _Adaptor:
    def __init__(self, adata):
        self.adata = adata


def test_the_fact_names_the_pairs_with_scores_and_resolves_its_family():
    a = synthetic.tissue("xenium", n_side=20)
    ada = _Adaptor(a)
    lines: list[str] = []
    sources: list = []
    assistant._add_cooccurrence(ada, lines, sources)
    assert lines == [] and sources == []
    inp = cc.prepare_cooccurrence(a, cc.plan_cooccurrence(a, "stripe", interval=8))
    res = cc.run_cooccurrence(inp)
    cc.save_cooccurrence(a, inp, res)
    assistant._add_cooccurrence(ada, lines, sources)
    text = "\n".join(lines)
    assert [s.ref for s in sources] == [cc.TAG] and cc.TAG in text
    assert "Co-occurrence of 'stripe'" in text and "within" in text and " um" in text
    assert "t0 with t1 (score" in text  # neighbouring stripes
    t1 = res.occ[0, 1, 0]
    assert f"t0 with t1 (score {t1:.2f})" in text
    assert "does not mean interacting" in text
    for tag in ("result:cooccurrence", "cooccurrence", "co_occurrence", "co-occurrence", "co occurrence", "result:co_occurrence"):
        assert claims.resolve_tag(tag, [cc.TAG, sd.TAG]) == [cc.TAG], tag
    # Stale facts are not given.
    a.obs["stripe"] = a.obs["stripe"].cat.rename_categories(lambda x: f"{x}_")
    assert cc.grounding_lines(a) == ([], "")


def test_the_fact_skips_distances_with_too_few_pairs():
    s = {"labels": ["a", "b", "c"], "occ": np.ones((3, 3, 2)).tolist(), "count": np.zeros((3, 3, 2), dtype=int).tolist()}
    s["occ"][0][1] = [9.0, 1.5]
    s["count"] = (np.full((3, 3, 2), 100) * np.array([0, 1])).tolist()
    high, _, t = cc._top_pairs(s)
    assert t == 1 and high == ["a with b (score 1.50)"]


@pytest.mark.parametrize("question", [
    "which cell types co-occur?", "what is the cooccurrence of these clusters?", "are T cells near each other and B cells?",
    "which cell types are close to each other?", "do they colocalise?", "do these colocalize?", "what is found together here?",
])
def test_router_keeps_the_fact(question):
    assert "groups" in rag_router.named_result_facets(question)


def test_the_full_guide_has_the_path_and_the_compact_guide_stays_as_it_was():
    assert "Unified View > Domains > Co-occurrence" in feature_guide.full_guide()
    assert "Co-occurrence" not in feature_guide.compact_guide()
    assert len(feature_guide.compact_guide()) < 1600
