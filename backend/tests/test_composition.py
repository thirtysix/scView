"""Replicate-aware composition testing (core/composition.py, api/v1/composition.py).

The reference numbers come from R limma 3.x run on the same tables: per-sample
logit / arcsin-sqrt shares, ``lmFit(y, model.matrix(~cond [+ donor]))``, then
``eBayes(robust=FALSE)``; for three conditions the F-test of the condition
coefficients. That is propeller's computation with robust=FALSE.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from scview.core import composition as comp
from scview.core import provenance
from scview.core.composition import CompositionError, composition_test

MAIN = threading.main_thread()

# Five groups, samples named "<condition>__<donor>" (rows) as limma saw them.
T2_SAMPLES = ["A__d1", "A__d2", "A__d3", "A__d4", "B__d1", "B__d2", "B__d3", "B__d4"]
T2 = [[144, 153, 59, 31, 17], [139, 93, 64, 26, 3], [261, 183, 66, 33, 10],
      [330, 66, 35, 18, 2], [241, 112, 62, 21, 9], [66, 106, 55, 15, 7],
      [232, 137, 68, 31, 16], [213, 73, 44, 16, 10]]
T3_SAMPLES = ["A__d1", "A__d2", "A__d3", "B__d1", "B__d2", "B__d3", "C__d1", "C__d2", "C__d3"]
T3 = [[172, 157, 78, 21, 13], [138, 84, 95, 11, 10], [193, 135, 107, 66, 19],
      [134, 215, 77, 15, 15], [157, 63, 35, 8, 13], [201, 130, 54, 56, 10],
      [82, 242, 102, 19, 5], [240, 93, 74, 39, 34], [135, 89, 22, 42, 11]]

# limma: group -> (statistic, p), and the prior df (None = Inf).
LIMMA_T2 = {
    ("logit", "paired"): (4.91124, {"g4": (1.728276, 0.122624), "g2": (0.896849, 0.39627),
                                    "g3": (-0.660180, 0.527887), "g0": (-0.461590, 0.656802),
                                    "g1": (0.184679, 0.858127)}),
    ("logit", "unpaired"): (12.407, {"g4": (1.697619, 0.106422), "g2": (0.611098, 0.548607),
                                     "g3": (-0.495360, 0.626208), "g0": (-0.350570, 0.729891),
                                     "g1": (0.142604, 0.888152)}),
    ("asin", "paired"): (2.76761, {"g2": (1.158117, 0.292513), "g4": (1.102195, 0.314232),
                                   "g3": (-0.707660, 0.506729), "g0": (-0.464481, 0.659317),
                                   "g1": (0.165629, 0.874101)}),
    ("asin", "unpaired"): (2.30283, {"g4": (1.139474, 0.28631), "g2": (0.700582, 0.502721),
                                     "g3": (-0.677260, 0.516664), "g0": (-0.302765, 0.769516),
                                     "g1": (0.123219, 0.904865)}),
}
LIMMA_T3 = {
    ("logit", "paired"): (None, {"g2": (1.840748, 0.158699), "g3": (1.079269, 0.339844),
                                 "g0": (0.390525, 0.676701), "g1": (0.217482, 0.804542),
                                 "g4": (0.007535, 0.992494)}),
    ("logit", "unpaired"): (None, {"g2": (0.855686, 0.424992), "g3": (0.501706, 0.605497),
                                   "g0": (0.181538, 0.833986), "g1": (0.101098, 0.903844),
                                   "g4": (0.003502, 0.996504)}),
    ("asin", "paired"): (1754.25, {"g2": (2.238498, 0.106922), "g0": (0.709215, 0.492171),
                                   "g3": (0.517439, 0.596136), "g1": (0.506161, 0.602893),
                                   "g4": (0.016409, 0.983725)}),
    ("asin", "unpaired"): (8.07185, {"g2": (1.339195, 0.29349), "g3": (0.259536, 0.775022),
                                     "g0": (0.254117, 0.779085), "g1": (0.176748, 0.839823),
                                     "g4": (0.011935, 0.988146)}),
}


def _obs(samples: list[str], counts, groups: list[str] | None = None,
         cond_col: str = "cond", donor_col: str = "donor") -> pd.DataFrame:
    """One row per cell: ``counts[i][j]`` cells of group j in sample i."""
    groups = groups or [f"g{j}" for j in range(len(counts[0]))]
    rows = []
    for name, row in zip(samples, counts):
        c, d = name.split("__")
        for g, n in zip(groups, row):
            rows += [(c, d, g)] * int(n)
    obs = pd.DataFrame(rows, columns=[cond_col, donor_col, "group"])
    obs.index = [f"c{i}" for i in range(len(obs))]
    return obs.astype("category")


@pytest.mark.parametrize("transform", ["logit", "asin"])
@pytest.mark.parametrize("design", ["paired", "unpaired"])
def test_two_conditions_match_limma(transform, design):
    r = composition_test(_obs(T2_SAMPLES, T2), "cond", "donor", "group",
                         transform=transform, paired=design == "paired")
    df_prior, expected = LIMMA_T2[(transform, design)]
    assert r.tested and r.design == design and r.test == "moderated t"
    assert r.df_residual == (3 if design == "paired" else 6)
    assert r.df_prior == pytest.approx(df_prior, rel=1e-4)
    for g in r.groups:
        stat, p = expected[g.group]
        assert g.stat == pytest.approx(stat, abs=2e-6)
        assert g.pval == pytest.approx(p, rel=1e-4)


@pytest.mark.parametrize("transform", ["logit", "asin"])
@pytest.mark.parametrize("design", ["paired", "unpaired"])
def test_three_conditions_match_limma_f_test(transform, design):
    r = composition_test(_obs(T3_SAMPLES, T3), "cond", "donor", "group",
                         transform=transform, paired=design == "paired")
    df_prior, expected = LIMMA_T3[(transform, design)]
    assert r.tested and r.test == "moderated F" and r.conditions == ["A", "B", "C"]
    if df_prior is None:
        assert r.df_prior is None  # variances pooled completely (limma: Inf)
    else:
        assert r.df_prior == pytest.approx(df_prior, rel=1e-4)
    for g in r.groups:
        stat, p = expected[g.group]
        assert g.stat == pytest.approx(stat, abs=2e-6)
        assert g.pval == pytest.approx(p, rel=1e-4)
        assert g.fold_change is None  # no single ratio for three conditions


def test_reported_shares_fold_change_and_fdr():
    r = composition_test(_obs(T2_SAMPLES, T2), "cond", "donor", "group", reference="B")
    assert r.conditions == ["B", "A"]  # the reference first
    props = np.array(T2) / np.array(T2).sum(axis=1, keepdims=True)
    g0 = next(g for g in r.groups if g.group == "g0")
    assert g0.props["A"] == pytest.approx(props[:4, 0].mean())
    assert g0.props["B"] == pytest.approx(props[4:, 0].mean())
    assert g0.fold_change == pytest.approx(props[:4, 0].mean() / props[4:, 0].mean())
    assert g0.n_cells == {"A": 144 + 139 + 261 + 330, "B": 241 + 66 + 232 + 213}
    # BH over the five groups; the stat flips sign with the reference.
    p = np.array([g.pval for g in r.groups])
    assert [g.fdr for g in r.groups] == pytest.approx(
        list(stats.false_discovery_control(p, method="bh")))
    assert g0.stat == pytest.approx(0.461590, abs=2e-6)
    assert r.n_samples == {"B": 4, "A": 4} and len(r.samples) == 8


# --- planted shift vs a cell-level chi-square ---------------------------------


def _simulate(rng, shift: float = 1.0, n_donors: int = 6, cells: int = 3000) -> pd.DataFrame:
    """Kang-like: every donor in both conditions, donor-to-donor and
    sample-to-sample variation in make-up, and no condition effect except
    ``shift`` times more of the rare group g6 (4%) in STIM, which dilutes every
    other group's share by only ~4% at 2x."""
    base = np.array([0.30, 0.20, 0.15, 0.12, 0.10, 0.07, 0.04, 0.02])
    samples, counts = [], []
    for d in range(n_donors):
        donor = np.exp(rng.normal(0, 0.3, len(base)))
        for cond in ("CTRL", "STIM"):
            p = base * donor * np.exp(rng.normal(0, 0.15, len(base)))
            if cond == "STIM":
                p[6] *= shift
            samples.append(f"{cond}__D{d}")
            counts.append(rng.multinomial(int(cells * rng.uniform(0.7, 1.3)), p / p.sum()))
    return _obs(samples, counts, cond_col="stim", donor_col="Donor")


def test_cells_as_replicates_call_noise_significant_but_donors_do_not():
    """No real shift: sample-to-sample variation alone. A chi-square on the
    pooled cells (cells as replicates) calls many groups significant; the
    replicate-aware test calls (almost) none."""
    rng = np.random.default_rng(11)
    replicate = cell_level = 0
    for _ in range(8):
        r = composition_test(_simulate(rng), "stim", "Donor", "group", reference="CTRL")
        assert r.tested and r.design == "paired" and r.n_samples == {"CTRL": 6, "STIM": 6}
        replicate += r.n_significant
        cell_level += r.n_significant_cell_level
    assert replicate <= 1  # of 64 null groups
    assert cell_level >= 20  # a third or more of them, at the same FDR


def test_a_planted_shift_is_found():
    rng = np.random.default_rng(3)
    others = 0
    for _ in range(5):
        r = composition_test(_simulate(rng, shift=2.0), "stim", "Donor", "group",
                             reference="CTRL")
        g6 = next(g for g in r.groups if g.group == "g6")
        assert g6.fold_change > 1.5 and g6.stat > 0 and g6.fdr < comp.FDR
        others += sum(g.fdr < comp.FDR for g in r.groups if g.group != "g6")
    assert others <= 2  # of 35 groups that only got diluted by ~4%


# --- design detection and guard rails -----------------------------------------


def test_pairing_follows_the_donors():
    kang_like = composition_test(_obs(T2_SAMPLES, T2), "cond", "donor", "group")
    assert kang_like.design == "paired" and kang_like.df_residual == 3
    nested = [s.replace("__d", "__e") if s.startswith("B") else s for s in T2_SAMPLES]
    r = composition_test(_obs(nested, T2), "cond", "donor", "group")
    assert r.design == "unpaired" and not r.notes
    partial = T2_SAMPLES[:7] + ["B__d9"]
    r = composition_test(_obs(partial, T2), "cond", "donor", "group")
    assert r.design == "unpaired" and "independent" in r.notes[0]
    with pytest.raises(CompositionError, match="every 'donor'"):
        composition_test(_obs(partial, T2), "cond", "donor", "group", paired=True)


def test_one_sample_per_condition_gives_proportions_only():
    # A "sample" column that just repeats the condition (Kang's orig_ident).
    obs = _obs(T2_SAMPLES, T2)
    obs["orig_ident"] = obs["cond"]
    r = composition_test(obs, "cond", "orig_ident", "group")
    assert not r.tested and r.n_samples == {"A": 1, "B": 1}
    assert all(g.pval is None and g.fdr is None for g in r.groups)
    assert all(g.props["A"] > 0 for g in r.groups)
    assert "at least 2 per condition" in r.warnings[0]
    r = composition_test(obs, "cond", None, "group")
    assert not r.tested and "No sample or donor column" in r.warnings[0]


def test_small_samples_are_left_out_and_can_leave_too_few():
    counts = [row[:] for row in T2]
    counts[4] = [3, 2, 1, 1, 0]  # B__d1: 7 cells
    r = composition_test(_obs(T2_SAMPLES, counts), "cond", "donor", "group")
    assert r.dropped_samples == ["d1 (B, 7 cells)"]
    assert r.n_samples == {"A": 4, "B": 3} and r.design == "unpaired"
    for row in (5, 6):
        counts[row] = [2, 2, 1, 0, 0]
    r = composition_test(_obs(T2_SAMPLES, counts), "cond", "donor", "group")
    assert not r.tested and r.n_samples["B"] == 1
    counts[7] = [1, 1, 1, 1, 1]
    r = composition_test(_obs(T2_SAMPLES, counts), "cond", "donor", "group")
    assert r.n_samples["B"] == 0 and r.n_cells["B"] == 0
    assert all(g.props["B"] is None and g.fold_change is None for g in r.groups)


def test_conditions_that_are_batches_are_flagged():
    obs = _obs(T2_SAMPLES, T2)
    obs["batch"] = obs["cond"].map({"A": "run1", "B": "run2"}).astype("category")
    r = composition_test(obs, "cond", "donor", "group")
    assert r.tested and any("can't be told apart" in w for w in r.warnings)
    # Crossed with the condition: nothing to say.
    obs["batch"] = obs["donor"].map({"d1": "x", "d2": "x", "d3": "y", "d4": "y"}).astype("category")
    assert composition_test(obs, "cond", "donor", "group").warnings == []


def test_correcting_for_the_condition_is_flagged():
    obs = _obs(T2_SAMPLES, T2)
    r = composition_test(obs, "cond", "donor", "group", batch_key="cond")
    assert any("batch-corrected for 'cond', which is the condition" in w for w in r.warnings)
    r = composition_test(obs, "cond", "donor", "group", batch_key="donor")
    assert r.warnings == []  # donors span both conditions


def test_bad_requests():
    obs = _obs(T2_SAMPLES, T2)
    with pytest.raises(CompositionError, match="different"):
        composition_test(obs, "cond", "cond", "group")
    with pytest.raises(CompositionError, match="not a value"):
        composition_test(obs, "cond", "donor", "group", conditions=["A", "Z"])
    with pytest.raises(CompositionError, match="at least two"):
        composition_test(obs, "cond", "donor", "group", conditions=["A"])
    with pytest.raises(CompositionError, match="reference"):
        composition_test(obs, "cond", "donor", "group", reference="Z")
    with pytest.raises(CompositionError, match="transform"):
        composition_test(obs, "cond", "donor", "group", transform="sqrt")
    obs["many"] = [f"x{i % 150}" for i in range(len(obs))]
    with pytest.raises(CompositionError, match="at most 100"):
        composition_test(obs, "cond", "donor", "many")


def test_missing_values_and_a_subset_of_conditions():
    obs = _obs(T3_SAMPLES, T3)
    obs["group"] = obs["group"].where(np.arange(len(obs)) % 50 != 0)
    r = composition_test(obs, "cond", "donor", "group", conditions=["C", "A"])
    assert r.conditions == ["C", "A"] and r.test == "moderated t"
    assert sum(r.n_cells.values()) == int(obs["group"].notna()[obs["cond"].isin(["A", "C"])].sum())


# --- endpoint -----------------------------------------------------------------


def _dataset(data_dir: Path, obs: pd.DataFrame, did: str = "ds") -> None:
    d = data_dir / "uploads" / did
    d.mkdir(parents=True, exist_ok=True)
    (d / "metadata.json").write_text(json.dumps({"id": did, "name": did, "status": "ready"}))
    a = ad.AnnData(np.zeros((len(obs), 3), dtype="float32"), obs=obs)
    a.var_names = ["G0", "G1", "G2"]
    a.write_h5ad(d / "data.h5ad")


async def test_endpoint_detects_the_donor_column_and_runs_off_the_loop(
        client, dataset_manager, monkeypatch):
    obs = _obs(T2_SAMPLES, T2, cond_col="stim", donor_col="Donor")
    obs["orig_ident"] = obs["stim"]  # sample-like by name, but one per condition
    _dataset(Path(dataset_manager.data_dir), obs)
    threads: list[threading.Thread] = []
    real = comp.composition_test

    def spy(*args, **kwargs):
        threads.append(threading.current_thread())
        return real(*args, **kwargs)

    monkeypatch.setattr(comp, "composition_test", spy)
    resp = await client.post("/api/v1/datasets/ds/composition/test",
                             json={"condition_key": "stim", "group_key": "group"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["sample_key"] == "Donor" and body["sample_key_detected"] is True
    assert body["tested"] and body["design"] == "paired" and body["reference"] == "A"
    assert {g["group"] for g in body["groups"]} == {f"g{i}" for i in range(5)}
    assert body["n_significant_cell_level"] is not None
    assert len(threads) == 1 and threads[0] is not MAIN

    # A named sample column is used as given.
    resp = await client.post("/api/v1/datasets/ds/composition/test",
                             json={"condition_key": "stim", "group_key": "group",
                                   "sample_key": "orig_ident"})
    assert resp.status_code == 200
    assert resp.json()["sample_key_detected"] is False and not resp.json()["tested"]


async def test_endpoint_errors(client, dataset_manager):
    _dataset(Path(dataset_manager.data_dir), _obs(T2_SAMPLES, T2))
    url = "/api/v1/datasets/ds/composition/test"
    r = await client.post(url, json={"condition_key": "nope", "group_key": "group"})
    assert r.status_code == 404
    r = await client.post(url, json={"condition_key": "cond", "group_key": "group",
                                     "conditions": ["A"]})
    assert r.status_code == 400 and "at least two" in r.json()["detail"]
    r = await client.post("/api/v1/datasets/missing/composition/test",
                          json={"condition_key": "cond", "group_key": "group"})
    assert r.status_code == 404


async def test_endpoint_warns_when_clusters_were_corrected_for_the_condition(
        client, dataset_manager):
    obs = _obs(T2_SAMPLES, T2)
    a = ad.AnnData(np.zeros((len(obs), 3), dtype="float32"), obs=obs)
    provenance.record_step(a, step="batch_correction", tool="harmony",
                           params={"batch_key": "cond", "batch_method": "harmony"})
    d = Path(dataset_manager.data_dir) / "uploads" / "ds"
    d.mkdir(parents=True)
    (d / "metadata.json").write_text(json.dumps({"id": "ds", "name": "ds", "status": "ready"}))
    a.var_names = ["G0", "G1", "G2"]
    a.write_h5ad(d / "data.h5ad")
    r = await client.post("/api/v1/datasets/ds/composition/test",
                          json={"condition_key": "cond", "group_key": "group"})
    assert r.status_code == 200
    assert any("batch-corrected for 'cond'" in w for w in r.json()["warnings"])


# --- co-pilot grounding -------------------------------------------------------


async def test_the_last_test_reaches_the_copilot(client, dataset_manager):
    from scview.core.assistant import build_grounding_context

    obs = _obs(T2_SAMPLES, T2, cond_col="stim", donor_col="Donor")
    _dataset(Path(dataset_manager.data_dir), obs)
    adaptor = await dataset_manager.get_or_load_dataset("ds")
    context, sources = build_grounding_context(adaptor, {"groups"})
    # Before any test: only the caution (the dataset has a condition column).
    assert "Never suggest a chi-square or Fisher test" in context
    assert "Composition test (last run" not in context

    r = await client.post("/api/v1/datasets/ds/composition/test",
                          json={"condition_key": "stim", "group_key": "group"})
    assert r.status_code == 200
    context, sources = build_grounding_context(adaptor, {"groups"})
    assert "Composition test (last run; samples as replicates" in context
    assert "B vs A by 'group', samples from 'Donor' (A 4, B 4; paired)" in context
    assert any(s.ref == "result:composition" for s in sources)
    # Not part of a markers-only question.
    context, _ = build_grounding_context(adaptor, {"markers"})
    assert "Composition" not in context


def test_grounding_drops_a_result_whose_groups_changed():
    obs = _obs(T2_SAMPLES, T2)

    class Adaptor:
        pass

    a = Adaptor()
    comp.remember(a, composition_test(obs, "cond", "donor", "group"))
    lines, detail = comp.grounding_lines(a, obs)
    assert detail and "of 5 groups" in detail
    renamed = obs.copy()
    renamed["group"] = renamed["group"].cat.rename_categories({"g0": "T cells"})
    lines, detail = comp.grounding_lines(a, renamed)
    assert detail is None and not any("last run" in x for x in lines)
