"""Spatial domains and niches (core/spatial_domains.py, api/v1/spatial_domains.py): the graph,
the features against CellCharter's rule, the mixture, the scan, enrichment against squidpy,
niches, the column rule, the stream, large datasets, replay and the co-pilot's fact."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from collections import deque
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from sklearn.metrics import adjusted_rand_score

from scview.api.v1 import spatial_domains as api
from scview.core import assistant, claims, feature_guide, large_layout, large_mode, provenance, replay_script, spatial
from scview.core import spatial_domains as sd
from scview.core.rag import router as rag_router
from tests.large_fixture import register, small_twin
from tests.test_hvg_sensitivity import _register
from tests.test_wnn_api import _events

_FIX = Path(__file__).parent / "fixtures" / "spatial_domains"
_spec = importlib.util.spec_from_file_location("spatial_domains_synthetic", _FIX / "synthetic.py")
synthetic = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(synthetic)

API = "/api/v1"


def _rings_by_bfs(adj, source, n_hops):
    """Rows at exactly each hop from ``source``, by breadth-first search."""
    adj = sp.csr_matrix(adj)
    depth = {source: 0}
    queue = deque([source])
    while queue:
        u = queue.popleft()
        if depth[u] == n_hops:
            continue
        for v in adj.indices[adj.indptr[u]:adj.indptr[u + 1]]:
            if int(v) not in depth:
                depth[int(v)] = depth[u] + 1
                queue.append(int(v))
    return [[v for v, d in depth.items() if d == h] for h in range(1, n_hops + 1)]


# ---------------------------------------------------------------------------
# Graph and features
# ---------------------------------------------------------------------------


def test_each_technology_gets_its_graph_and_sections_are_never_linked():
    for tech, grid, typical in (("visium", 6, 6), ("visium_hd", 4, 4), ("xenium", None, 6), ("cells", None, 6)):
        a = synthetic.tissue(tech, n_side=20, sections=2)
        assert sd.grid_neighbours(a) == grid, tech
        adj = sd.spatial_graph(a.obsm["spatial"], sd._libraries(a), grid)
        assert abs(adj - adj.T).nnz == 0 and adj.diagonal().sum() == 0
        assert int(np.median(np.diff(adj.indptr))) == typical, tech
        libs = sd._libraries(a)
        rows = np.repeat(np.arange(a.n_obs), np.diff(adj.indptr))
        assert (libs[rows] == libs[adj.indices]).all(), tech


def test_grid_edge_points_keep_only_their_real_neighbours():
    a = synthetic.tissue("visium_hd", n_side=10)
    deg = np.diff(sd.spatial_graph(a.obsm["spatial"], sd._libraries(a), 4).indptr)
    # corners 2, edges 3, inside 4: never the diagonal of a square grid
    assert sorted(np.unique(deg).tolist()) == [2, 3, 4]
    assert (deg == 2).sum() == 4


def test_long_links_above_the_percentile_are_removed():
    rng = np.random.default_rng(0)
    xy = np.vstack([rng.uniform(0, 100, (400, 2)), rng.uniform(1000, 1100, (400, 2))])
    full = sd.delaunay_edges(xy, 100.0)
    cut = sd.delaunay_edges(xy, 99.0)
    long_links = np.linalg.norm(xy[full[0]] - xy[full[1]], axis=1)
    threshold = np.percentile(long_links, 99)
    assert len(cut[0]) == int((long_links <= threshold).sum())
    assert (np.linalg.norm(xy[cut[0]] - xy[cut[1]], axis=1) < 200).all()  # the two islands apart


def test_too_few_or_collinear_points_still_get_neighbours():
    line = np.column_stack([np.arange(10.0), np.zeros(10)])
    rows, cols = sd.delaunay_edges(line)
    assert len(rows) > 0 and set(rows) == set(range(10))
    assert sd.spatial_graph(np.array([[0.0, 0.0], [1.0, 0.0]]), np.array(["a", "a"])).nnz == 2
    assert len(sd.delaunay_edges(np.array([[0.0, 0.0]]))[0]) == 0


def test_features_are_the_means_of_exact_hop_rings():
    a = synthetic.tissue("cells", n_side=12)
    adj = sd.spatial_graph(a.obsm["spatial"], sd._libraries(a))
    X = np.asarray(a.obsm["X_pca"])[:, :4]
    Z = sd.aggregate_neighbors(adj, X, 3)
    assert Z.shape == (a.n_obs, 16) and Z.dtype == np.float32
    np.testing.assert_allclose(Z[:, :4], X, atol=1e-6)
    for source in (0, 7, 50, 143):
        rings = _rings_by_bfs(adj, source, 3)
        for h, ring in enumerate(rings, start=1):
            if ring:
                np.testing.assert_allclose(Z[source, 4 * h:4 * (h + 1)], X[ring].mean(axis=0), atol=1e-4)


def test_an_empty_ring_takes_the_ring_before_it_and_a_lone_row_its_own_vector():
    # Two linked rows and one alone: rings 2 and 3 are empty.
    adj = sp.csr_matrix(np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=np.int8))
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    Z = sd.aggregate_neighbors(adj, X, 3)
    np.testing.assert_allclose(Z[0], [1, 2, 3, 4, 3, 4, 3, 4])
    np.testing.assert_allclose(Z[2], [5, 6] * 4)


def test_no_hops_is_the_plain_vectors():
    a = synthetic.tissue("visium", n_side=10)
    adj = sd.spatial_graph(a.obsm["spatial"], sd._libraries(a), 6)
    np.testing.assert_array_equal(sd.aggregate_neighbors(adj, a.obsm["X_pca"], 0), a.obsm["X_pca"])


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------


def test_neighbours_recover_a_tissue_one_spot_alone_cannot():
    a = synthetic.tissue("visium", n_side=40, noise=3.0, seed=1)
    truth = a.obs["stripe"].astype(str).to_numpy()
    adj = sd.spatial_graph(a.obsm["spatial"], sd._libraries(a), 6)
    X = np.asarray(a.obsm["X_pca"])
    with_hops = np.mean([adjusted_rand_score(truth, sd.fit_domains(sd.aggregate_neighbors(adj, X, 3), 4, s)[0])
                         for s in range(3)])
    alone = np.mean([adjusted_rand_score(truth, sd.fit_domains(sd.aggregate_neighbors(adj, X, 0), 4, s)[0])
                     for s in range(3)])
    assert with_hops > 0.6 and alone < 0.2, (with_hops, alone)


def test_a_fit_is_seeded_and_domains_are_numbered_by_size():
    rng = np.random.default_rng(0)
    Z = np.vstack([rng.normal(0, 1, (300, 3)), rng.normal(8, 1, (100, 3)), rng.normal(-8, 1, (200, 3))])
    a, n_fit = sd.fit_domains(Z.astype(np.float32), 3, 5)
    b, _ = sd.fit_domains(Z.astype(np.float32), 3, 5)
    assert n_fit == 600 and (a == b).all()
    assert np.bincount(a).tolist() == [300, 200, 100]


def test_above_the_cap_it_fits_on_a_sample_and_assigns_all(monkeypatch):
    rng = np.random.default_rng(0)
    Z = np.vstack([rng.normal(0, 1, (800, 3)), rng.normal(9, 1, (800, 3))]).astype(np.float32)
    labels, n_fit = sd.fit_domains(Z, 2, 0, max_fit=500)
    assert n_fit == 500 and labels.shape == (1600,)
    assert adjusted_rand_score(np.repeat([0, 1], 800), labels) == 1.0
    assert (sd.seeded_sample(1000, 100, 3) == sd.seeded_sample(1000, 100, 3)).all()
    assert len(sd.seeded_sample(50, 100, 3)) == 50


def test_singular_covariances_raise_the_regularisation_as_cellcharter_does():
    Z = np.zeros((50, 3), dtype=np.float32)
    Z[25:] = 1.0  # no variance in any cluster, and exactly collinear
    labels, _ = sd.fit_domains(Z, 2, 0)
    assert adjusted_rand_score(np.repeat([0, 1], 25), labels) == 1.0


def test_stability_curve_mean_ari_against_neighbouring_k_and_best_peak():
    rng = np.random.default_rng(0)
    base = rng.integers(0, 3, 300)
    splinter = base.copy()
    splinter[:5] = 3  # k = 4: the same groups and a few strays
    runs = {2: [np.minimum(base, 1)] * 3, 3: [base] * 3, 4: [splinter] * 3,
            5: [rng.integers(0, 5, 300) for _ in range(3)], 6: [rng.integers(0, 6, 300) for _ in range(3)]}
    c = sd.stability_curve(runs, 2, 5)
    assert c["ks"] == [2, 3, 4, 5] and c["suggested"] == 3 and c["peaks"] == [3] and c["has_peak"] is True
    # k = 2 has no k = 1 run to be compared with; a flat or falling curve has no peak.
    assert c["mean"][0] == pytest.approx(adjusted_rand_score(runs[2][0], runs[3][1]), abs=1e-3)
    falling = sd.stability_curve({2: [base] * 2, 3: [base] * 2, 4: [rng.integers(0, 4, 300)] * 2,
                                  5: [rng.integers(0, 5, 300)] * 2, 6: [rng.integers(0, 6, 300)] * 2}, 2, 5)
    assert falling["has_peak"] is False and falling["suggested"] == 2


def test_a_scan_streams_each_fit_and_suggests_the_planted_k():
    a = synthetic.tissue("visium", n_side=24, noise=0.5, seed=2)
    p = sd.plan_domains(a, k=None, n_pcs=6, n_hops=1, k_min=2, k_max=6, repeats=2)
    assert p.estimate_scan_seconds > 0
    events = list(sd.iter_domains(sd.prepare_domains(a, p)))
    scans = [e for e in events if e["type"] == "scan"]
    assert len(scans) == 6 * 2 and scans[-1]["done"] == scans[-1]["total"] == 12  # k = 2..7, twice
    curve = next(e for e in events if e["type"] == "curve")["stability"]
    assert curve["ks"] == [2, 3, 4, 5, 6] and curve["suggested"] in curve["ks"]
    res = events[-1]["result"]
    assert res.k == curve["suggested"] == res.stability["suggested"]
    assert set(res.stage_seconds) == {s for s, _ in sd.DOMAIN_STAGES}


def test_a_stopped_run_stops_between_pieces_of_work():
    a = synthetic.tissue("visium", n_side=12)
    inp = sd.prepare_domains(a, sd.plan_domains(a, k=3))
    gen = sd.iter_domains(inp, stop=lambda: True)
    with pytest.raises(sd.SpatialDomainsCancelled):
        next(gen)


# ---------------------------------------------------------------------------
# Plan, column rule, refusals
# ---------------------------------------------------------------------------


def test_plan_says_what_it_uses_and_what_it_needs():
    a = synthetic.tissue("cells", n_side=20, sections=2)
    p = sd.plan_domains(a, k=7, n_pcs=10, n_hops=3)
    assert p.rep == "X_pca" and p.dims == 40 and p.n_fit == a.n_obs and p.technology == "cells"
    assert any("uncorrected PCA of several sections" in n for n in p.notes)
    assert p.estimate_seconds > 0 and p.estimate_bytes > sd.BASE_BYTES
    a.obsm["X_pca_harmony"] = a.obsm["X_pca"].copy()
    assert sd.plan_domains(a).rep == "X_pca_harmony"
    one = synthetic.tissue("cells", n_side=20)
    one.obsm["X_pca_harmony"] = one.obsm["X_pca"].copy()
    assert sd.plan_domains(one).rep == "X_pca"  # one section: nothing to correct


def test_plan_states_the_sample_above_the_cap(monkeypatch):
    monkeypatch.setattr(sd, "MAX_FIT_OBS", 100)
    a = synthetic.tissue("cells", n_side=20)
    p = sd.plan_domains(a)
    assert p.n_fit == 100 and any("seeded sample of 100" in n for n in p.notes)


def test_more_pcs_than_there_are_uses_what_there_is():
    a = synthetic.tissue("visium", n_side=12)
    p = sd.plan_domains(a, n_pcs=500)
    assert p.n_pcs == 12 and p.n_pcs_available == 12


def test_refusals_are_sentences():
    a = synthetic.tissue("visium", n_side=12)
    no_pca = a.copy()
    del no_pca.obsm["X_pca"]
    with pytest.raises(sd.SpatialDomainsError, match="no PCA"):
        sd.plan_domains(no_pca)
    no_xy = a.copy()
    del no_xy.obsm["spatial"]
    with pytest.raises(sd.SpatialDomainsError, match="no spatial positions"):
        sd.plan_domains(no_xy)
    with pytest.raises(sd.SpatialDomainsError, match="k must be"):
        sd.plan_domains(a, k=1)
    with pytest.raises(sd.SpatialDomainsError, match="usable column name"):
        sd.plan_domains(a, column="bad name!")
    nan = a.copy()
    nan.obsm["spatial"][0, 0] = np.nan
    p = sd.plan_domains(nan)
    with pytest.raises(sd.SpatialDomainsError, match="missing"):
        sd.prepare_domains(nan, p)


def test_a_run_never_overwrites_without_the_users_choice():
    a = synthetic.tissue("visium", n_side=14)
    first = sd.prepare_domains(a, sd.plan_domains(a, k=3))
    sd.save_domains(a, first, sd.run_domains(first))
    before = a.obs["spatial_domain"].copy()
    with pytest.raises(sd.ColumnInUse, match="tick Replace"):
        sd.plan_domains(a, k=4)
    with pytest.raises(sd.ColumnInUse, match="not a domain result"):
        sd.plan_domains(a, k=4, column="stripe", overwrite=True)
    assert sd.free_column(a, sd.DOMAIN_COLUMN, 4) == "spatial_domain_k4"
    assert sd.free_suggested_column(a) == "spatial_domain_suggested"
    other = sd.prepare_domains(a, sd.plan_domains(a, k=4, column="spatial_domain_k4"))
    summary = sd.save_domains(a, other, sd.run_domains(other))
    assert summary["column"] == "spatial_domain_k4" and summary["columns"] == ["spatial_domain", "spatial_domain_k4"]
    assert (a.obs["spatial_domain"] == before).all()
    again = sd.prepare_domains(a, sd.plan_domains(a, k=5, overwrite=True))
    sd.save_domains(a, again, sd.run_domains(again))
    assert a.obs["spatial_domain"].nunique() == 5


def test_saved_result_has_counts_per_section_and_goes_stale_with_the_pcs():
    a = synthetic.tissue("cells", n_side=14, sections=2)
    inp = sd.prepare_domains(a, sd.plan_domains(a, k=3))
    res = sd.run_domains(inp)
    s = sd.save_domains(a, inp, res, cells_fingerprint="cells-1")
    assert isinstance(a.obs["spatial_domain"].dtype, pd.CategoricalDtype)
    assert list(a.obs["spatial_domain"].cat.categories) == ["0", "1", "2"]
    assert s["k"] == 3 and s["stale"] is False and sum(s["sizes"]) == a.n_obs
    assert set(s["per_library"]) == {"s0", "s1"}
    assert [sum(x) for x in zip(*s["per_library"].values())] == s["sizes"]
    assert 0 <= s["coherence"]["shuffled"] <= s["coherence"]["within"] <= 1
    step = provenance.read_provenance(a)["history"][-1]
    assert step["step"] == "spatial_domains" and step["params"]["k"] == 3
    assert sd.domains_summary(a, "cells-2")["stale"] is True  # other cells
    a.obsm["X_pca"] = a.obsm["X_pca"] + 1.0
    assert "PCs changed" in sd.domains_summary(a, "cells-1")["stale_reason"]


def test_a_chosen_k_keeps_the_curve_of_the_same_inputs_only():
    a = synthetic.tissue("visium", n_side=14, noise=0.5)
    p = sd.plan_domains(a, k=None, n_pcs=6, n_hops=1, k_min=2, k_max=4, repeats=2)
    inp = sd.prepare_domains(a, p)
    sd.save_domains(a, inp, sd.run_domains(inp))
    assert sd.domains_summary(a)["scan"]["repeats"] == 2
    q = sd.prepare_domains(a, sd.plan_domains(a, k=3, n_pcs=6, n_hops=1, column="other"))
    assert sd.save_domains(a, q, sd.run_domains(q))["scan"]["repeats"] == 2
    r = sd.prepare_domains(a, sd.plan_domains(a, k=3, n_pcs=6, n_hops=2, column="third"))
    assert sd.save_domains(a, r, sd.run_domains(r))["scan"] is None


def test_estimates_stay_within_a_fifth_of_the_real_slides():
    # Measured at 2 CPUs (batch 32 report): k = 8, 10 PCs, 3 hops, 50,000 fitted.
    xenium = sd.graph_seconds(164_000, 3) + sd.fit_seconds(164_000, 8, 40)
    hd16 = sd.graph_seconds(98_917, 3, 4) + sd.fit_seconds(98_917, 8, 40)
    assert xenium == pytest.approx(11.0, rel=0.2) and hd16 == pytest.approx(8.2, rel=0.2)
    assert sd.scan_seconds(164_000, 40, 2, 15, 5) == pytest.approx(675, rel=0.2)
    assert sd.scan_seconds(3_639, 40, 2, 15, 5) == pytest.approx(53, rel=0.2)
    assert sd.enrichment_seconds(164_000) == pytest.approx(9.7, rel=0.2)
    assert sd.domain_bytes(164_000, 40, 8, 3) / 1024**3 == pytest.approx(0.25, rel=0.2)
    assert sd.domain_bytes(400_000, 40, 8, 3) / 1024**3 == pytest.approx(0.59, rel=0.2)
    assert sd.domain_bytes(399_424, 40, 8, 3, 4) / 1024**3 == pytest.approx(0.40, rel=0.2)


def test_estimates_grow_and_the_sample_caps_the_fit():
    assert sd.fit_seconds(40_000, 7, 40) > sd.fit_seconds(20_000, 7, 40)
    assert sd.scan_seconds(1_000_000, 40, 2, 15, 5) == sd.scan_seconds(sd.MAX_FIT_OBS, 40, 2, 15, 5)
    assert sd.domain_bytes(400_000, 40, 7, 3) > sd.domain_bytes(100_000, 40, 7, 3)


# ---------------------------------------------------------------------------
# Neighbourhood enrichment
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tech,grid", [("cells", None), ("visium", 6)])
def test_enrichment_equals_squidpy(tech, grid):
    ref = json.loads((_FIX / "nhood_squidpy.json").read_text())[tech]
    a = synthetic.tissue(tech, n_side=30, seed=3, sections=2)
    # squidpy's Delaunay graph has no long-link cut; compare on that graph.
    adj = sd.spatial_graph(a.obsm["spatial"], sd._libraries(a), grid, 100.0)
    codes, labels = sd._labels_of(a, "stripe")
    z, count = sd.nhood_enrichment(adj, codes, len(labels), sd._libraries(a), 1000, 0)
    if tech == "visium":  # squidpy's grid is directed; a hexagon's rows are not always symmetric at the edge
        assert np.abs(count - np.asarray(ref["count"])).max() <= 0.01 * np.asarray(ref["count"]).max()
    else:
        np.testing.assert_array_equal(count, ref["count"])
    assert np.corrcoef(z.ravel(), np.asarray(ref["zscore"]).ravel())[0, 1] > 0.995


def test_enrichment_counts_by_hand_and_shuffles_within_each_section():
    adj = sp.csr_matrix(np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=np.int8))
    codes = np.array([0, 1, 0, 0])
    z, count = sd.nhood_enrichment(adj, codes, 2, np.array(["a", "a", "b", "b"]), 50, 0)
    assert count.tolist() == [[2, 1], [1, 0]]
    # Each section's labels only swap within it: the count never varies, so z says nothing.
    assert np.isnan(z).all()
    rng = np.random.default_rng(0)
    groups = [np.array([0, 1, 2]), np.array([3, 4])]
    out = sd.shuffle_within(np.array([0, 0, 1, 5, 6]), groups, rng)
    assert sorted(out[:3].tolist()) == [0, 0, 1] and sorted(out[3:].tolist()) == [5, 6]


def test_enrichment_leaves_out_observations_without_a_label_and_is_seeded():
    a = synthetic.tissue("cells", n_side=14)
    a.obs["stripe"] = a.obs["stripe"].astype(object)
    a.obs.loc[a.obs_names[:10], "stripe"] = np.nan
    a.obs["stripe"] = a.obs["stripe"].astype("category")
    p = sd.plan_enrichment(a, "stripe", n_perms=50)
    assert p.n_labelled == a.n_obs - 10 and any("without a label" in n for n in p.notes)
    r1 = sd.run_enrichment(sd.prepare_enrichment(a, p))
    r2 = sd.run_enrichment(sd.prepare_enrichment(a, p))
    assert np.array_equal(r1.zscore, r2.zscore) and sum(r1.sizes) == a.n_obs - 10
    s = sd.save_enrichment(a, sd.prepare_enrichment(a, p), r1)
    assert s["available"] and s["labels"] == p.labels and len(s["zscore"]) == len(p.labels)
    # Where squidpy keeps it.
    np.testing.assert_array_equal(a.uns["stripe_nhood_enrichment"]["count"], r1.count)
    a.obs["stripe"] = a.obs["stripe"].cat.rename_categories({"t0": "renamed"})
    assert sd.enrichment_summary(a)["stale"] is True


def test_enrichment_refuses_a_numeric_or_one_label_column():
    a = synthetic.tissue("cells", n_side=10)
    a.obs["score"] = np.linspace(0, 1, a.n_obs)
    a.obs["one"] = pd.Categorical(["x"] * a.n_obs)
    with pytest.raises(sd.SpatialDomainsError, match="holds numbers"):
        sd.plan_enrichment(a, "score")
    with pytest.raises(sd.SpatialDomainsError, match="at least 2"):
        sd.plan_enrichment(a, "one")
    with pytest.raises(sd.SpatialDomainsError, match="not in this dataset"):
        sd.plan_enrichment(a, "nope")


# ---------------------------------------------------------------------------
# Composition niches
# ---------------------------------------------------------------------------


def test_window_fractions_by_hand_within_each_section():
    xy = np.array([[0.0, 0], [1, 0], [2, 0], [100, 0], [0, 0], [50, 0]])
    libs = np.array(["a", "a", "a", "a", "b", "b"])
    codes = np.array([0, 0, 1, 1, 1, 1])
    f = sd.window_fractions(xy, codes, 2, libs, 2)
    # Itself and its nearest in its own section (never the other section's point at the same place).
    np.testing.assert_allclose(f[0], [1.0, 0.0])
    np.testing.assert_allclose(f[3], [0.0, 1.0])
    np.testing.assert_allclose(f[4], [0.0, 1.0])
    big = sd.window_fractions(xy, codes, 2, libs, 50)  # a window bigger than the section
    np.testing.assert_allclose(big[4], [0.0, 1.0])
    np.testing.assert_allclose(big[0], [0.5, 0.5])


def test_niches_on_cells_and_their_table():
    a = synthetic.tissue("xenium", n_side=20)
    p = sd.plan_niches(a, "stripe", k_window=20, n_niches=4)
    assert p.n_labelled == a.n_obs and p.estimate_seconds > 0 and p.estimate_bytes > 0
    inp = sd.prepare_niches(a, p)
    res = sd.run_niches(inp)
    s = sd.save_niches(a, inp, res)
    assert list(a.obs["spatial_niche"].cat.categories) == ["0", "1", "2", "3"]
    assert s["available"] and len(s["fractions"]) == 4 and len(s["fractions"][0]) == 4
    assert sum(s["sizes"]) == a.n_obs
    np.testing.assert_allclose(np.sum(s["fractions"], axis=1), 1.0, atol=1e-3)
    # Four stripes, four niches: each niche is mostly one stripe.
    assert min(max(row) for row in s["fractions"]) > 0.5
    assert provenance.read_provenance(a)["history"][-1]["step"] == "spatial_niches"
    with pytest.raises(sd.ColumnInUse):
        sd.plan_niches(a, "stripe")


def test_niches_refuse_spots_and_allow_hd_8um_with_a_note():
    visium = synthetic.tissue("visium", n_side=12)
    with pytest.raises(sd.SpatialDomainsError, match="a Visium spot mixes several cells"):
        sd.plan_niches(visium, "stripe")
    assert sd.niches_allowed(visium)["ok"] is False
    hd8 = synthetic.tissue("visium_hd", n_side=12, bin_um=8.0)
    p = sd.plan_niches(hd8, "stripe", n_niches=3)
    assert any("8 um bins" in n for n in p.notes) and p.unit == "bin"
    hd16 = synthetic.tissue("visium_hd", n_side=12, bin_um=16.0)
    with pytest.raises(sd.SpatialDomainsError, match="these Visium HD bins are 16 um"):
        sd.plan_niches(hd16, "stripe")


def test_niches_leave_unlabelled_rows_nan():
    a = synthetic.tissue("cells", n_side=14)
    a.obs["stripe"] = a.obs["stripe"].astype(object)
    a.obs.loc[a.obs_names[:7], "stripe"] = np.nan
    p = sd.plan_niches(a, "stripe", n_niches=3)
    inp = sd.prepare_niches(a, p)
    res = sd.run_niches(inp)
    sd.save_niches(a, inp, res)
    assert a.obs["spatial_niche"].isna().sum() == 7


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------


async def test_api_runs_domains_saves_and_refuses_a_taken_column(client, dataset_manager):
    did = _register(dataset_manager, synthetic.tissue("visium", n_side=14), "dv")
    info = (await client.get(f"{API}/datasets/{did}/spatial/domains")).json()
    assert info["runnable"]["ok"] is True and info["domains"]["available"] is False and info["running"] is False
    assert info["runnable"]["plan"]["grid"] == 6 and info["free_column"] == "spatial_domain"
    assert info["niches_allowed"]["ok"] is False and "mixes several cells" in info["niches_allowed"]["reason"]
    r = await client.post(f"{API}/datasets/{did}/spatial/domains/run-stream", json={"k": 3})
    assert r.status_code == 200, r.text
    events = _events(r.text)
    kinds = [e for e, _ in events]
    assert kinds[0] == "start" and kinds[-2:] == ["result", "complete"], kinds
    assert [d["stage"] for e, d in events if e == "stage"] == ["graph", "features", "fit", "assign"]
    assert [s["stage"] for s in events[0][1]["stages"]] == ["graph", "features", "fit", "assign"]
    result = events[-2][1]
    assert result["k"] == 3 and result["stale"] is False and "CellCharter" in result["method"]

    # Saved: the dataset reopens with the column, listed as a label column, and the summary.
    dataset_manager.evict(did)
    ds = (await client.get(f"{API}/datasets/{did}")).json()
    col = next(c for c in ds["obs_columns"] if c["name"] == "spatial_domain")
    assert col["dtype"] == "category" and col["n_unique"] == 3
    adata = (await dataset_manager.get_or_load_dataset(did)).adata
    assert provenance.read_provenance(adata)["history"][-1]["step"] == "spatial_domains"
    again = (await client.get(f"{API}/datasets/{did}/spatial/domains")).json()
    assert again["domains"]["available"] is True and again["domains"]["stale"] is False
    assert again["free_column"] == "spatial_domain_k7"

    taken = await client.post(f"{API}/datasets/{did}/spatial/domains/run-stream", json={"k": 4})
    assert taken.status_code == 409 and "tick Replace" in taken.json()["detail"]
    named = await client.post(f"{API}/datasets/{did}/spatial/domains/run-stream",
                              json={"k": 4, "column": "spatial_domain", "overwrite": True})
    assert _events(named.text)[-2][1]["k"] == 4


async def test_api_suggest_streams_the_scan_and_the_curve(client, dataset_manager):
    did = _register(dataset_manager, synthetic.tissue("visium", n_side=16, noise=0.5), "ds")
    r = await client.post(f"{API}/datasets/{did}/spatial/domains/run-stream",
                          json={"k": None, "n_pcs": 6, "n_hops": 1, "k_min": 2, "k_max": 5, "repeats": 2})
    events = _events(r.text)
    assert sum(1 for e, _ in events if e == "scan") == 5 * 2
    curve = next(d for e, d in events if e == "curve")["stability"]
    result = events[-2][1]
    assert result["suggested"] is True and result["k"] == curve["suggested"] and result["scan"]["ks"] == [2, 3, 4, 5]


async def test_api_enrichment_and_niches_on_cells(client, dataset_manager):
    did = _register(dataset_manager, synthetic.tissue("xenium", n_side=14), "dx")
    r = await client.post(f"{API}/datasets/{did}/spatial/nhood-enrichment/run-stream",
                          json={"column": "stripe", "n_perms": 50})
    events = _events(r.text)
    assert any(e == "progress" for e, _ in events)
    assert events[-2][1]["labels"] == ["t0", "t1", "t2", "t3"]
    assert (await client.post(f"{API}/datasets/{did}/spatial/nhood-enrichment/run-stream",
                              json={"column": "nope"})).status_code == 422
    r = await client.post(f"{API}/datasets/{did}/spatial/niches/run-stream",
                          json={"label_column": "stripe", "n_niches": 3})
    assert _events(r.text)[-2][1]["n_niches"] == 3
    dataset_manager.evict(did)
    info = (await client.get(f"{API}/datasets/{did}/spatial/domains?enrichment_column=stripe")).json()
    assert info["enrichment"]["available"] and len(info["enrichment"]["zscore"]) == 4
    assert info["niches"]["available"] and info["niches"]["stale"] is False


async def test_api_refuses_spots_for_niches_and_a_dataset_without_positions(client, dataset_manager):
    did = _register(dataset_manager, synthetic.tissue("visium", n_side=10), "dr")
    r = await client.post(f"{API}/datasets/{did}/spatial/niches/run-stream", json={"label_column": "stripe"})
    assert r.status_code == 422 and "mixes several cells" in r.json()["detail"]
    plain = synthetic.tissue("cells", n_side=10)
    del plain.obsm["spatial"]
    did2 = _register(dataset_manager, plain, "dn")
    info = (await client.get(f"{API}/datasets/{did2}/spatial/domains")).json()
    assert info["runnable"]["ok"] is False and "no spatial positions" in info["runnable"]["reason"]
    r = await client.post(f"{API}/datasets/{did2}/spatial/domains/run-stream", json={"k": 3})
    assert r.status_code == 422


async def test_api_one_run_at_a_time(client, dataset_manager):
    did = _register(dataset_manager, synthetic.tissue("visium", n_side=10), "db")
    api._running.add(did)
    try:
        r = await client.post(f"{API}/datasets/{did}/spatial/domains/run-stream", json={"k": 3})
        assert r.status_code == 409
    finally:
        api._running.discard(did)


async def test_a_memory_refusal_is_said_before_the_run(client, dataset_manager, monkeypatch):
    from scview.core.dataset_manager import DatasetMemoryError

    did = _register(dataset_manager, synthetic.tissue("visium", n_side=10), "dm")

    await dataset_manager.get_or_load_dataset(did)

    async def refuse(dataset_id, need, extra=0, what=""):
        raise DatasetMemoryError(dataset_id, need, 1, 2, what=what)

    monkeypatch.setattr(dataset_manager, "reserve", refuse)
    r = await client.post(f"{API}/datasets/{did}/spatial/domains/run-stream", json={"k": 3})
    assert r.status_code == 507


def _sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


async def test_a_large_dataset_runs_saves_in_place_and_equals_the_dataset_in_memory(client, dataset_manager, monkeypatch):
    t = synthetic.tissue("xenium", n_side=20)
    rng = np.random.default_rng(1)
    counts = ad.AnnData(X=sp.csr_matrix(rng.poisson(2.0, (t.n_obs, 40)).astype(np.float32)), obs=t.obs,
                        var=pd.DataFrame(index=[f"G{i}" for i in range(40)]), obsm=dict(t.obsm), uns=t.uns)
    h5ad = register(dataset_manager.data_dir, "sL", counts)
    small = _register(dataset_manager, small_twin(counts), "sS")
    sidecars = large_layout.sidecar_paths(h5ad)
    before = [_sha(p) for p in sidecars]
    with h5py.File(h5ad, "r") as f:
        x_offset = f["X"]["data"].id.get_offset()

    def boom(*_a, **_k):
        raise AssertionError("A domains run read the expression matrix")

    with monkeypatch.context() as m:
        for name in ("__getitem__", "toarray", "to_memory"):
            if hasattr(large_mode.OnDiskMatrix, name):
                m.setattr(large_mode.OnDiskMatrix, name, boom)
        big = await dataset_manager.get_or_load_dataset("sL")
        assert big.is_large
        info = (await client.get(f"{API}/datasets/sL/spatial/domains")).json()
        assert info["runnable"]["plan"]["technology"] == "xenium"
        r = await client.post(f"{API}/datasets/sL/spatial/domains/run-stream", json={"k": 4})
        assert _events(r.text)[-2][1]["k"] == 4, r.text
        r = await client.post(f"{API}/datasets/sL/spatial/nhood-enrichment/run-stream",
                              json={"column": "stripe", "n_perms": 50})
        assert _events(r.text)[-2][1]["available"] is True, r.text
        r = await client.post(f"{API}/datasets/sL/spatial/niches/run-stream",
                              json={"label_column": "stripe", "n_niches": 3})
        assert _events(r.text)[-2][1]["available"] is True, r.text

    r = await client.post(f"{API}/datasets/{small}/spatial/domains/run-stream", json={"k": 4})
    assert _events(r.text)[-2][1]["k"] == 4
    in_memory = (await dataset_manager.get_or_load_dataset(small)).adata.obs["spatial_domain"].astype(str).to_numpy()

    # Reopened from disk: results there, X never rewritten, sidecars untouched, same labels.
    dataset_manager.evict("sL")
    big = await dataset_manager.get_or_load_dataset("sL")
    assert big.is_large
    np.testing.assert_array_equal(big.adata.obs["spatial_domain"].astype(str).to_numpy(), in_memory)
    assert big.adata.obs["spatial_niche"].nunique() == 3
    info = (await client.get(f"{API}/datasets/sL/spatial/domains?enrichment_column=stripe")).json()
    assert info["domains"]["stale"] is False and info["niches"]["stale"] is False
    assert info["enrichment"]["stale"] is False and len(info["enrichment"]["count"]) == 4
    assert [_sha(p) for p in sidecars] == before
    with h5py.File(h5ad, "r") as f:
        assert f["X"]["data"].id.get_offset() == x_offset
    ds = (await client.get(f"{API}/datasets/sL")).json()
    assert {"spatial_domain", "spatial_niche"} <= {c["name"] for c in ds["obs_columns"]}


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------


def test_replay_gives_the_same_domains_enrichment_and_niches(tmp_path):
    inp = synthetic.tissue("xenium", n_side=14)
    out = inp.copy()
    d = sd.prepare_domains(out, sd.plan_domains(out, k=3, n_pcs=6, seed=2))
    sd.save_domains(out, d, sd.run_domains(d))
    e = sd.prepare_enrichment(out, sd.plan_enrichment(out, "spatial_domain", n_perms=30, seed=1))
    sd.save_enrichment(out, e, sd.run_enrichment(e))
    n = sd.prepare_niches(out, sd.plan_niches(out, "stripe", n_niches=3, seed=4))
    sd.save_niches(out, n, sd.run_niches(n))
    assert [x["status"] for x in replay_script.describe(out)] == ["exact", "exact", "exact"]
    src, script, dst = tmp_path / "input.h5ad", tmp_path / "replay.py", tmp_path / "replayed.h5ad"
    inp.write_h5ad(src)
    script.write_text(replay_script.build_script(out, dataset_id="t", dataset_name="test"))
    proc = subprocess.run([sys.executable, str(script), str(src), str(dst)], capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr[-3000:]
    rep = ad.read_h5ad(dst)
    assert (rep.obs["spatial_domain"].astype(str) == out.obs["spatial_domain"].astype(str)).all()
    assert (rep.obs["spatial_niche"].astype(str) == out.obs["spatial_niche"].astype(str)).all()
    np.testing.assert_allclose(rep.uns["spatial_domain_nhood_enrichment"]["zscore"],
                               out.uns["spatial_domain_nhood_enrichment"]["zscore"], atol=1e-9)


def test_replay_of_an_unreadable_record_is_a_comment():
    ctx = replay_script._Ctx(adata=None)
    for fn in (replay_script._spatial_domains, replay_script._nhood_enrichment, replay_script._spatial_niches):
        assert fn({"column": "x"}, ctx).status == replay_script.NOT_REPLAYABLE


# ---------------------------------------------------------------------------
# Co-pilot
# ---------------------------------------------------------------------------


class _Adaptor:
    def __init__(self, adata):
        self.adata = adata


def test_the_fact_names_k_counts_and_stability_and_resolves_its_family():
    a = synthetic.tissue("visium", n_side=14, noise=0.5)
    ada = _Adaptor(a)
    lines: list[str] = []
    sources: list = []
    assistant._add_spatial_domains(ada, lines, sources)
    assert lines == [] and sources == []
    p = sd.plan_domains(a, k=None, n_pcs=6, n_hops=1, k_min=2, k_max=4, repeats=2)
    inp = sd.prepare_domains(a, p)
    res = sd.run_domains(inp)
    sd.save_domains(a, inp, res)
    e = sd.prepare_enrichment(a, sd.plan_enrichment(a, "spatial_domain", n_perms=30))
    sd.save_enrichment(a, e, sd.run_enrichment(e))
    assistant._add_spatial_domains(ada, lines, sources)
    text = "\n".join(lines)
    assert [s.ref for s in sources] == [sd.TAG] and sd.TAG in text
    assert f"{res.k} spatial domains" in text and f"domain 0: {res.labels.tolist().count(0):,}" in text
    assert "Stability scan" in text and "Neighbourhood enrichment" in text
    assert claims.resolve_tag("result:spatial_domains", [sd.TAG, "result:spatial_genes"]) == [sd.TAG]
    assert claims.resolve_tag("spatial_domains", [sd.TAG, "result:spatial_genes"]) == [sd.TAG]
    assert claims.resolve_tag("niches", [sd.TAG]) == [sd.TAG]
    assert claims.resolve_tag("result:domains:0", [sd.TAG]) == [sd.TAG]
    assert claims.resolve_tag("spatial", ["result:spatial_genes", sd.TAG]) == ["result:spatial_genes"]
    # Stale facts are not given.
    a.obsm["X_pca"] = a.obsm["X_pca"] + 1.0
    assert "spatial domains in" not in "\n".join(sd.grounding_lines(a)[0])


def test_router_keeps_the_fact_and_the_guide_stays_under_its_cap():
    for q in ("what are the spatial domains here?", "which niches did you find?", "how many domains?"):
        assert "groups" in rag_router.named_result_facets(q), q
    guide = feature_guide.compact_guide()
    assert "Domains (niches)" in guide and len(guide) <= 1600, len(guide)


def test_contract_technology_names_still_hold():
    assert spatial.technology(synthetic.tissue("visium_hd", n_side=6))["unit"] == "bin"
    assert spatial.is_spot_data(synthetic.tissue("visium_hd", n_side=6)) is True
    assert spatial.is_spot_data(synthetic.tissue("xenium", n_side=6)) is False
