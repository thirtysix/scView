"""Study planner (core/study_planner.py, api/v1/planner.py).

Expected values come from the generating model written out here (known
spreads, known dispersions), from pydeseq2 (the app's pseudobulk DE), or by
hand; never from the planner's own output.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from scipy import special, stats

from scview.core import composition, pseudobulk
from scview.core import study_planner as planner
from scview.core.claims import resolve_tag

MAIN = threading.main_thread()


# ---------------------------------------------------------------------------
# Fixtures written from the model, independently of the planner's code
# ---------------------------------------------------------------------------


def composition_obs(shares, sd_within, sd_donor, n_donors=8, cells=1000, paired=True,
                    seed=0, fold=None) -> pd.DataFrame:
    """Per-cell labels of a pilot: donor d's log-odds share of type j in a
    sample = logit(share_j) + donor effect + residual; counts binomial."""
    rng = np.random.default_rng(seed)
    shares = np.asarray(shares, dtype=float)
    G = len(shares)
    base = np.log(shares / (1 - shares))
    donor_eff = rng.normal(0, sd_donor, (2 * n_donors, G))
    rows = []
    for d in range(n_donors):
        for c, cond in enumerate(("CTRL", "STIM")):
            donor = f"D{d}" if paired else f"D{d}{cond}"
            eff = donor_eff[d] if paired else donor_eff[d + c * n_donors]
            logit = base + eff + rng.normal(0, sd_within, G)
            if fold is not None and c == 1:
                logit = logit + np.log(fold)
            y = rng.binomial(cells, special.expit(logit))
            for j, k in enumerate(y):
                rows += [(donor, cond, f"t{j}")] * int(k)
    return pd.DataFrame(rows, columns=["Donor", "stim", "cell_type"])


def test_logit_location_gives_the_mean_share():
    rng = np.random.default_rng(1)
    z = rng.standard_normal(400_000)
    for share, sd in ((0.01, 0.8), (0.3, 0.4), (0.7, 1.2)):
        mu = planner.logit_location(np.array([share]), np.array([sd ** 2]))[0]
        assert special.expit(mu + sd * z).mean() == pytest.approx(share, rel=0.01)


@pytest.mark.parametrize("paired", [True, False])
def test_the_spread_estimate_recovers_the_generating_sd(paired):
    """Averaged over pilots of 8 donors, the estimated donor-to-donor variances
    match the ones the pilots were drawn with, counting noise removed."""
    shares = [0.3, 0.2, 0.15, 0.1, 0.05, 0.03]
    sw, sdn = 0.3, 0.5
    within, donor, total = [], [], []
    for seed in range(24):
        obs = composition_obs(shares, sw, sdn, paired=paired, seed=seed)
        pilot = planner.composition_pilot(obs, "cell_type", "stim", "Donor")
        assert pilot.design == ("paired" if paired else "unpaired")
        m = planner.fit_composition(pilot, cal_reps=200)
        within.append(m.sd_within ** 2)
        donor.append(m.sd_donor ** 2)
        total.append(m.sd_within ** 2 + m.sd_donor ** 2)
    if paired:
        assert np.mean(within) == pytest.approx(sw ** 2, rel=0.15)
        assert np.mean(donor) == pytest.approx(sdn ** 2, rel=0.2)
    else:
        assert np.mean(within) == pytest.approx(sw ** 2 + sdn ** 2, rel=0.15)
        assert np.all(np.asarray(donor) == 0)


def test_the_interval_covers_the_true_spread_most_of_the_time():
    shares = [0.3, 0.2, 0.15, 0.1, 0.05, 0.03]
    hits = n = 0
    for seed in range(20):
        obs = composition_obs(shares, 0.35, 0.0, paired=True, seed=100 + seed)
        m = planner.fit_composition(planner.composition_pilot(obs, "cell_type", "stim", "Donor"),
                                    cal_reps=200)
        for s in m.spreads:
            hits += s.sd_low <= 0.35 <= s.sd_high
            n += 1
    assert hits / n >= 0.85  # nominal 95%; the noise subtraction makes it approximate


def _model(shares=(1 / 6,) * 6, sd=0.3):
    obs = composition_obs(list(shares), sd, 0.0, n_donors=6, cells=800, paired=False, seed=3)
    return planner.assumed_composition(
        planner.composition_pilot(obs, "cell_type", "stim", "Donor"), sd)


def test_no_change_is_detected_at_the_nominal_rate():
    """Fold change 1 with equal types (the empirical Bayes prior is right): the
    raw p < alpha rate is alpha and the BH-adjusted one no higher."""
    m = _model()
    grid = planner.CompositionGrid(targets=[0], fold_changes=[1.0], donors=[6], cells=[1000],
                                   design="unpaired", reps=600, seed=5)
    raws, adj = [], []
    for t in range(6):
        p = planner.power_point(m, grid, t, 1.0, 6, 1000)
        raws.append(p.power_raw)
        adj.append(p.power)
    se = np.sqrt(0.05 * 0.95 / (600 * 6))
    assert np.mean(raws) == pytest.approx(0.05, abs=4 * se)
    assert np.mean(adj) <= 0.05 + 4 * se


def test_power_grows_with_donors_and_effect_and_saturates():
    m = _model()
    grid = planner.CompositionGrid(targets=[2], fold_changes=[1.5, 2.0], donors=[3, 6, 12],
                                   cells=[1000], design="unpaired", reps=300, seed=1)
    pw = {(fc, n): planner.power_point(m, grid, 2, fc, n, 1000).power
          for fc in (1.5, 2.0) for n in (3, 6, 12)}
    for fc in (1.5, 2.0):
        assert pw[(fc, 3)] < pw[(fc, 6)] < pw[(fc, 12)]
    for n in (3, 6, 12):
        assert pw[(1.5, n)] <= pw[(2.0, n)] + 0.03
    huge = planner.power_point(m, grid, 2, 4.0, 12, 1000)
    assert huge.power >= 0.98


def test_an_impossible_fold_change_is_left_out():
    m = _model(shares=(0.45, 0.25, 0.15, 0.1, 0.05), sd=0.3)
    grid = planner.CompositionGrid(targets=[0], fold_changes=[3.0], donors=[4], cells=[500],
                                   design="unpaired", reps=20)
    p = planner.power_point(m, grid, 0, 3.0, 4, 500)
    assert p.power is None and not p.pessimistic
    p = planner.power_point(m, grid, 0, 3.0, 4, 500, sd_scale=np.full(5, 0.5))
    assert p.power is None and p.pessimistic  # the dashed curve keeps its own points
    shifted = planner.shifted_shares(np.array([0.4, 0.3, 0.3]), 1, 2.0)
    assert shifted.sum() == pytest.approx(1.0)
    assert shifted[1] == pytest.approx(0.6) and shifted[0] == pytest.approx(0.4 * 0.4 / 0.7)


def test_the_simulation_runs_the_apps_own_test(monkeypatch):
    calls = {"moderated": 0, "transform": 0, "bh": 0}
    real_m, real_t, real_b = (composition.moderated_test, composition.transform_props,
                              composition._bh)

    def spy_m(*a, **k):
        calls["moderated"] += 1
        assert k.get("robust") is True
        return real_m(*a, **k)

    def spy_t(*a, **k):
        calls["transform"] += 1
        return real_t(*a, **k)

    def spy_b(*a, **k):
        calls["bh"] += 1
        return real_b(*a, **k)

    monkeypatch.setattr(composition, "moderated_test", spy_m)
    monkeypatch.setattr(composition, "transform_props", spy_t)
    monkeypatch.setattr(composition, "_bh", spy_b)
    m = _model()
    grid = planner.CompositionGrid(targets=[0, 1], fold_changes=[2.0], donors=[3, 4],
                                   cells=[500], design="unpaired", reps=15, pessimistic=False)
    pts = list(planner.iter_composition_power(m, grid))
    assert len(pts) == planner.composition_points(grid) == 4
    assert calls["moderated"] == calls["bh"] == 4 * 15
    assert calls["transform"] >= 4 * 15


def test_donors_needed_interpolates_between_grid_values():
    P = planner.PowerPoint
    pts = [P("a", 2, 3, 500, 0.4, 0), P("a", 2, 4, 500, 0.7, 0), P("a", 2, 6, 500, 0.85, 0),
           P("a", 2, 8, 500, 0.95, 0)]
    # 0.7 at 4, 0.85 at 6: 0.8 at 4 + 2 * (0.1 / 0.15) = 5.33 -> 6
    assert planner.donors_needed(pts) == 6
    assert planner.donors_needed(pts, 0.4) == 3
    assert planner.donors_needed(pts, 0.99) is None
    assert planner.donors_needed([P("a", 2, 3, 500, None, None)]) is None
    # 0.78 at 8, 0.82 at 12: exactly 10, though the float sum is 10.000000000000007
    assert planner.donors_needed([P("a", 2, 8, 500, 0.78, 0), P("a", 2, 12, 500, 0.82, 0)]) == 10


def test_no_donors_says_so_and_offers_an_entered_spread():
    obs = composition_obs([0.3, 0.2, 0.5], 0.3, 0.3, n_donors=4)
    pilot = planner.composition_pilot(obs, "cell_type", "stim", None)
    assert pilot.design == "none" and not pilot.replicated
    with pytest.raises(planner.NoReplicatesError, match="needs a pilot with replicate donors"):
        planner.fit_composition(pilot)
    # One sample per condition (a sample column that only repeats the condition).
    obs["orig_ident"] = obs["stim"]
    pilot = planner.composition_pilot(obs, "cell_type", "stim", "orig_ident")
    assert pilot.design == "none"
    m = planner.assumed_composition(pilot, planner.ASSUMED_SD)
    assert m.source == "assumed" and np.allclose(m.sd_within, planner.ASSUMED_SD)
    assert not m.can_pair
    req = planner.CompositionRequestModel(group_key="cell_type", condition_key="stim",
                                          sample_key="orig_ident", assumed_sd=0.35,
                                          fold_changes=[2.0], donors=[3, 6], cells=[500], reps=20)
    model, grid = planner.prepare_composition(obs, req)
    events = list(planner.iter_composition_plan(model, grid, 0.8))
    result = events[-1]["result"]
    assert result["source"] == "assumed" and result["design"] == "unpaired"
    assert any("ENTERED" in a for a in result["assumptions"])
    req.assumed_sd = None
    with pytest.raises(planner.NoReplicatesError):
        planner.prepare_composition(obs, req)


def test_a_paired_plan_needs_a_paired_pilot():
    obs = composition_obs([0.3, 0.2, 0.5], 0.3, 0.3, n_donors=4, paired=False)
    req = planner.CompositionRequestModel(group_key="cell_type", condition_key="stim",
                                          sample_key="Donor", design="paired")
    with pytest.raises(planner.PlannerError, match="paired pilot"):
        planner.prepare_composition(obs, req)


def test_one_condition_pilot_plans_an_unpaired_study():
    obs = composition_obs([0.3, 0.2, 0.5], 0.4, 0.0, n_donors=6, paired=False)
    obs = obs[obs["stim"] == "CTRL"]
    pilot = planner.composition_pilot(obs, "cell_type", None, "Donor")
    assert pilot.design == "single" and pilot.n_samples == {"all": 6}
    m = planner.fit_composition(pilot)
    assert not m.can_pair and m.df == 5


# ---------------------------------------------------------------------------
# Pseudobulk DE
# ---------------------------------------------------------------------------


def de_counts(n_donors=4, n_genes=300, paired=True, seed=0, n_de=30, lfc=1.5):
    rng = np.random.default_rng(seed)
    mean = np.exp(rng.normal(4, 1.5, n_genes))
    disp = 0.05 + 2 / mean
    s_cond = np.tile([0, 1], n_donors)
    s_donor = np.repeat(np.arange(n_donors), 2) if paired else np.arange(2 * n_donors)
    donor_eff = np.exp(rng.normal(0, 0.3, (2 * n_donors, n_genes)))
    eff = np.zeros(n_genes)
    eff[:n_de] = lfc * np.log(2) * rng.choice([-1, 1], n_de)
    m = mean * donor_eff[s_donor] * np.exp(eff * s_cond[:, None])
    K = rng.poisson(rng.gamma(1 / disp, m * disp))
    return K, s_cond, s_donor


@pytest.mark.parametrize("paired", [True, False])
def test_the_fast_nb_test_agrees_with_pydeseq2(paired):
    """On genes past the app's count filter (as both the app and the planner test
    them). Below it, a near-empty gene's Cox-Reid optimum can sit at the upper
    bound while pydeseq2's L-BFGS-B stops at its 1e-8 starting value."""
    K, s_cond, s_donor = de_counts(paired=paired, n_genes=600, n_de=60)
    K = K[:, pseudobulk._filter_genes(K, 4)]
    kind = "paired" if paired else "unpaired"
    mine = planner.nb_wald_test(K, s_cond, s_donor, kind)
    meta = pd.DataFrame({"condition": np.where(s_cond == 1, "b", "a"),
                         "donor": [f"d{d}" for d in s_donor]},
                        index=[f"s{i}" for i in range(len(s_cond))])
    df = pd.DataFrame(K, index=meta.index, columns=[f"g{j}" for j in range(K.shape[1])])
    genes, wald = pseudobulk._run_deseq2(df, meta, "~donor + condition" if paired else "~condition")
    ref = pd.DataFrame(genes, columns=["g", "lfc", "p", "padj"]).set_index("g").reindex(df.columns)
    rstat = pd.Series(wald).reindex(df.columns)
    ok = rstat.notna().to_numpy()
    # pydeseq2's contrast is a vs b (reference vs tested): the sign flips.
    assert stats.spearmanr(-mine.stat[ok], rstat[ok]).statistic > 0.995
    sig_m = set(df.columns[mine.padj < 0.05])
    sig_r = set(ref.index[ref["padj"] < 0.05])
    assert len(sig_m & sig_r) / max(len(sig_m | sig_r), 1) > 0.85
    assert np.median(np.abs(-mine.log2fc[ok] - ref["lfc"][ok])) < 0.02


def de_adata(n_donors=8, cells=300, n_genes=400, alpha_bio=0.04, phi_cell=0.6, tau=0.2,
             seed=0) -> ad.AnnData:
    """Cells of type T: sample mean = q * donor effect * Gamma(bio dispersion),
    each cell's count NB around it with dispersion phi_cell; type B a few cells."""
    rng = np.random.default_rng(seed)
    q = np.exp(rng.normal(np.log(0.5), 1.0, n_genes))
    donor = np.exp(rng.normal(-tau ** 2 / 2, tau, (n_donors, n_genes)))
    blocks, obs = [], []
    for d in range(n_donors):
        for cond in ("CTRL", "STIM"):
            lam = q * donor[d] * rng.gamma(1 / alpha_bio, alpha_bio, n_genes)
            cell_mean = rng.gamma(1 / phi_cell, phi_cell, (cells, n_genes)) * lam
            blocks.append(rng.poisson(cell_mean))
            obs += [(f"D{d}", cond, "T")] * cells
            blocks.append(rng.poisson(0.5, (5, n_genes)))
            obs += [(f"D{d}", cond, "B")] * 5
    X = sp.csr_matrix(np.vstack(blocks).astype(np.float32))
    a = ad.AnnData(X=X, obs=pd.DataFrame(obs, columns=["Donor", "stim", "cell_type"],
                                         index=[f"c{i}" for i in range(X.shape[0])]))
    a.var_names = [f"g{j}" for j in range(n_genes)]
    a.layers["counts"] = X.copy()
    return a


def test_de_pilot_splits_the_dispersion_into_donor_and_cell_parts():
    a = de_adata()
    pilot = planner.de_pilot(a, "cell_type", "T", "stim", "Donor")
    assert pilot.design == "paired" and pilot.n_samples == {"CTRL": 8, "STIM": 8}
    assert pilot.dropped_samples == []
    model = planner.fit_de(pilot, n_genes=400)
    assert np.median(model.alpha_cell) == pytest.approx(0.6, rel=0.15)
    assert np.median(model.alpha_bio) == pytest.approx(0.04, rel=0.4)
    assert np.median(model.tau2) == pytest.approx(0.2 ** 2, abs=0.03)
    # Per-cell mean: the generating q (times the mean donor effect, 1).
    gi = [int(g[1:]) for g in model.genes]
    rng = np.random.default_rng(0)
    q = np.exp(rng.normal(np.log(0.5), 1.0, 400))
    assert np.median(model.q / q[gi]) == pytest.approx(1.0, rel=0.1)


def test_de_power_grows_with_donors_and_cells_and_keeps_the_fdr():
    model = planner.fit_de(planner.de_pilot(de_adata(), "cell_type", "T", "stim", "Donor"),
                           n_genes=400)
    grid = planner.DEGrid(log2fc=[1.0], donors=[3, 8], cells=[30, 300], design="unpaired",
                          reps=6, seed=2)
    p = {(n, m): planner.de_power_point(model, grid, 1.0, n, m) for n in (3, 8) for m in (30, 300)}
    assert p[(3, 300)].power < p[(8, 300)].power
    assert p[(8, 30)].power < p[(8, 300)].power
    assert p[(8, 30)].tested_share < p[(8, 300)].tested_share
    assert p[(8, 300)].fdp is not None and p[(8, 300)].fdp < 0.15
    big = planner.de_power_point(model, grid, 3.0, 8, 300)
    assert big.power > 0.9


def test_de_without_replicates_says_so():
    a = de_adata(n_donors=2, n_genes=60)
    a.obs["orig_ident"] = a.obs["stim"]
    pilot = planner.de_pilot(a, "cell_type", "T", "stim", "orig_ident")
    assert pilot.design == "none" and pilot.sums.shape[1] == 0  # no counts read
    with pytest.raises(planner.NoReplicatesError, match="replicate donors"):
        planner.fit_de(pilot)


def test_de_needs_counts():
    a = de_adata(n_donors=3, n_genes=60)
    del a.layers["counts"]
    a.X = sp.csr_matrix(np.log1p(a.X.toarray()))
    with pytest.raises(planner.PlannerError, match="raw counts"):
        planner.de_pilot(a, "cell_type", "T", "stim", "Donor")


def test_sample_sums_match_dense_and_sparse():
    rng = np.random.default_rng(0)
    M = rng.poisson(1.0, (500, 30)).astype(float)
    rows = np.sort(rng.choice(500, 200, replace=False))
    codes = rng.integers(0, 4, 200)
    want1 = np.vstack([M[rows][codes == k].sum(axis=0) for k in range(4)])
    want2 = np.vstack([(M[rows][codes == k] ** 2).sum(axis=0) for k in range(4)])
    old = planner._CHUNK_ELEMENTS
    try:
        planner._CHUNK_ELEMENTS = 64 * 30  # several chunks
        for mat in (M, sp.csr_matrix(M)):
            s1, s2 = planner._sample_sums(mat, rows, codes, 4)
            assert np.allclose(s1, want1) and np.allclose(s2, want2)
    finally:
        planner._CHUNK_ELEMENTS = old


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


def _dataset(root: Path, a: ad.AnnData, did: str = "ds") -> None:
    d = root / "uploads" / did
    d.mkdir(parents=True)
    (d / "metadata.json").write_text(json.dumps({"id": did, "name": did, "status": "ready"}))
    a.write_h5ad(d / "data.h5ad")


def _events(text: str) -> list[tuple[str, dict]]:
    out = []
    for block in text.split("\n\n"):
        lines = [x for x in block.splitlines() if x and not x.startswith(":")]
        if len(lines) == 2 and lines[0].startswith("event: "):
            out.append((lines[0][7:], json.loads(lines[1][6:])))
    return out


def _comp_adata(**kw) -> ad.AnnData:
    obs = composition_obs([0.3, 0.2, 0.15, 0.1, 0.05], 0.3, 0.4, n_donors=4, cells=300, **kw)
    obs.index = [f"c{i}" for i in range(len(obs))]
    a = ad.AnnData(X=sp.csr_matrix((len(obs), 5), dtype=np.float32), obs=obs)
    a.var_names = [f"g{j}" for j in range(5)]
    return a


async def test_pilot_endpoint_detects_the_donor_column(client, dataset_manager):
    a = _comp_adata()
    a.obs["orig_ident"] = a.obs["stim"]
    _dataset(Path(dataset_manager.data_dir), a)
    r = await client.post("/api/v1/datasets/ds/planner/pilot",
                          json={"group_key": "cell_type", "condition_key": "stim"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["sample_key"] == "Donor" and body["sample_key_detected"] is True
    assert body["design"] == "paired" and body["replicated"] is True
    assert body["n_samples"] == {"CTRL": 4, "STIM": 4} and body["message"] is None
    shares = {g["group"]: g["share"] for g in body["groups"]}
    assert body["suggested"][0] == max(shares, key=shares.get)  # the most abundant first
    r = await client.post("/api/v1/datasets/ds/planner/pilot",
                          json={"group_key": "cell_type", "condition_key": "stim",
                                "sample_key": "orig_ident"})
    assert r.status_code == 200 and r.json()["replicated"] is False
    assert "needs a pilot with replicate donors" in r.json()["message"]
    r = await client.post("/api/v1/datasets/ds/planner/pilot", json={"group_key": "nope"})
    assert r.status_code == 400 and "not found" in r.json()["detail"]
    r = await client.post("/api/v1/datasets/missing/planner/pilot", json={"group_key": "x"})
    assert r.status_code == 404


async def test_composition_stream_runs_off_the_loop_and_reaches_the_copilot(
        client, dataset_manager, monkeypatch):
    from scview.core.assistant import build_grounding_context

    _dataset(Path(dataset_manager.data_dir), _comp_adata())
    threads: list[threading.Thread] = []
    real = composition.moderated_test

    def spy(*a, **k):
        threads.append(threading.current_thread())
        return real(*a, **k)

    monkeypatch.setattr(composition, "moderated_test", spy)
    r = await client.post("/api/v1/datasets/ds/planner/composition-stream", json={
        "group_key": "cell_type", "condition_key": "stim", "groups": ["t1"],
        "fold_changes": [2.0], "donors": [3, 6], "cells": [500, 2000], "reps": 20})
    assert r.status_code == 200
    events = _events(r.text)
    kinds = [e for e, _ in events]
    assert kinds[0] == "start" and kinds[-2:] == ["result", "complete"]
    progress = [d for e, d in events if e == "progress"]
    assert len(progress) == events[0][1]["n_points"] == 4 + 2  # grid + pessimistic
    result = events[-2][1]
    assert result["design"] == "paired" and result["sample_key"] == "Donor"
    assert result["sample_key_detected"] is True
    assert {p["donors"] for p in result["points"]} == {3, 6}
    assert len(result["needed"]) == 3  # two cells values + the pessimistic curve
    assert threads and all(t is not MAIN for t in threads)

    adaptor = await dataset_manager.get_or_load_dataset("ds")
    context, sources = build_grounding_context(adaptor, {"groups"})
    assert "Study planner (last run" in context and "- t1 (share" in context
    assert any(s.ref == planner.TAG for s in sources)
    assert resolve_tag("planner", [s.ref for s in sources]) == [planner.TAG]


async def test_composition_stream_rejects_before_streaming(client, dataset_manager):
    a = _comp_adata()
    a.obs["orig_ident"] = a.obs["stim"]
    _dataset(Path(dataset_manager.data_dir), a)
    url = "/api/v1/datasets/ds/planner/composition-stream"
    r = await client.post(url, json={"group_key": "cell_type", "condition_key": "stim",
                                     "sample_key": "orig_ident"})
    assert r.status_code == 400 and "replicate donors" in r.json()["detail"]
    r = await client.post(url, json={"group_key": "cell_type", "condition_key": "stim",
                                     "sample_key": "orig_ident", "assumed_sd": 0.35,
                                     "fold_changes": [2.0], "donors": [3], "cells": [500],
                                     "reps": 10})
    assert r.status_code == 200
    result = next(d for e, d in _events(r.text) if e == "result")
    assert result["source"] == "assumed" and result["assumed_sd"] == 0.35
    r = await client.post(url, json={"group_key": "cell_type", "fold_changes": [50]})
    assert r.status_code == 400
    r = await client.post("/api/v1/datasets/missing/planner/composition-stream",
                          json={"group_key": "cell_type"})
    assert r.status_code == 404


async def test_de_stream(client, dataset_manager):
    _dataset(Path(dataset_manager.data_dir), de_adata(n_donors=4, n_genes=250))
    r = await client.post("/api/v1/datasets/ds/planner/de-stream", json={
        "group_key": "cell_type", "group": "T", "condition_key": "stim",
        "log2fc": [1.0], "donors": [3, 6], "cells": [100], "reps": 3, "n_genes": 200})
    assert r.status_code == 200, r.text
    events = _events(r.text)
    assert [e for e, _ in events][-2:] == ["result", "complete"]
    result = events[-2][1]
    assert result["kind"] == "de" and result["group"] == "T" and result["design"] == "paired"
    assert result["pilot"]["n_simulated"] == 200 and len(result["points"]) == 2
    assert result["pilot"]["counts_source"] == "layers['counts']"
    assert any("DESeq2" in s for s in result["assumptions"])
    r = await client.post("/api/v1/datasets/ds/planner/de-stream",
                          json={"group_key": "cell_type", "group": "nope"})
    assert r.status_code == 400
