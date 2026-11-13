"""The composition test's covariates and partial pairing (Emont 2022: sex had to be
removed with exclude_samples, and ``paired: true`` failed on a design where some mice
gave one depot only)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scview.core import composition as comp
from scview.core.composition import CompositionError, composition_test
from tests.test_mcp import msigdb_dir, test_settings  # noqa: F401 - fixtures


def _obs(mice: int = 6, one_depot: dict | None = None, seed: int = 0) -> pd.DataFrame:
    """Mice x depot (ING, PG) samples, 200 cells each; g0 is commoner in PG, and in
    females; each mouse has its own baseline."""
    rng = np.random.default_rng(seed)
    one_depot = one_depot or {}
    rows = []
    for m in range(mice):
        sex = "F" if m % 2 else "M"
        base = rng.normal(0, 0.3)
        for depot in ("ING", "PG"):
            if one_depot.get(f"m{m}") not in (None, depot):
                continue
            logit = -1.0 + base + (0.8 if depot == "PG" else 0) + (0.6 if sex == "F" else 0)
            share = 1 / (1 + np.exp(-logit))
            n0 = rng.binomial(200, share)
            rows += [{"donor_id": f"m{m}", "sample": f"m{m}_{depot}", "depot": depot, "sex": sex,
                      "group": "g0" if i < n0 else ("g1" if i % 2 else "g2")} for i in range(200)]
    obs = pd.DataFrame(rows)
    obs.index = obs.index.astype(str)
    return obs


def test_a_covariate_enters_the_design_as_limma_would():
    obs = _obs()
    r = composition_test(obs, "depot", "sample", "group", covariates=["sex"])
    assert r.covariates == ["sex"] and r.tested and r.design == "unpaired"
    assert "adjusted for sex" in r.comparison()
    # The same fit by hand: intercept, condition, sex.
    groups = ["g0", "g1", "g2"]
    samples = sorted(obs["sample"].unique(), key=lambda s: ("PG" in s, s))
    counts = np.array([[((obs["sample"] == s) & (obs["group"] == g)).sum() for g in groups] for s in samples])
    meta = obs.drop_duplicates("sample").set_index("sample").loc[samples]
    X = np.column_stack([np.ones(len(samples)), (meta["depot"] == "PG").to_numpy(float),
                         (meta["sex"] == "M").to_numpy(float)])
    fit = comp.moderated_test(comp.transform_props(counts, "logit"), X, [1], robust=True)
    got = {g.group: g for g in r.groups}
    for i, g in enumerate(groups):
        assert got[g].stat == pytest.approx(fit.stat[i], rel=1e-9)
        assert got[g].pval == pytest.approx(fit.pval[i], rel=1e-9)


def test_the_animal_as_a_covariate_blocks_like_pairing_on_it():
    obs = _obs()
    blocked = composition_test(obs, "depot", "sample", "group", covariates=["donor_id"])
    paired = composition_test(obs, "depot", "donor_id", "group", paired=True)
    assert blocked.covariates == ["donor_id"] and paired.design == "paired"
    for a, b in zip(blocked.groups, paired.groups):
        assert a.pval == pytest.approx(b.pval, rel=1e-8)
    # Blocked on the animal: no repeated-donor warning.
    assert not any("not independent" in w for w in blocked.warnings)
    plain = composition_test(obs, "depot", "sample", "group")
    assert any("add 'donor_id' as a covariate" in w for w in plain.warnings)


def test_bad_covariates_are_refused():
    obs = _obs()
    obs["site"] = obs["depot"].map({"ING": "a", "PG": "b"})
    with pytest.raises(CompositionError, match="confounded with the condition"):
        composition_test(obs, "depot", "sample", "group", covariates=["site"])
    obs["per_cell"] = np.arange(len(obs)) % 2
    with pytest.raises(CompositionError, match="varies within a sample"):
        composition_test(obs, "depot", "sample", "group", covariates=["per_cell"])
    with pytest.raises(CompositionError, match="cannot also be a covariate"):
        composition_test(obs, "depot", "sample", "group", covariates=["group"])
    with pytest.raises(CompositionError, match="not found"):
        composition_test(obs, "depot", "sample", "group", covariates=["nope"])


def test_paired_true_uses_the_complete_pairs_and_says_how_many():
    obs = _obs(mice=7, one_depot={"m5": "ING", "m6": "PG"})
    r = composition_test(obs, "depot", "donor_id", "group", paired=True)
    assert r.design == "paired" and r.pairing["n_pairs"] == 5
    assert r.pairing["n_dropped"] == 2 and r.pairing["dropped"] == ["m5", "m6"]
    assert r.n_samples == {"ING": 5, "PG": 5}
    assert any("Paired on the 5 'donor_id' values found in every condition; 2 left out" in n for n in r.notes)
    assert "paired on 5" in r.comparison()
    # Left to itself the design stays unpaired, and the note says paired true would pair 5.
    auto = composition_test(obs, "depot", "donor_id", "group")
    assert auto.design == "unpaired"
    assert any("Paired true pairs the 5 in every condition" in n for n in auto.notes)


def test_pairing_on_a_nested_sample_column_says_what_to_do():
    obs = _obs()
    with pytest.raises(CompositionError) as e:
        composition_test(obs, "depot", "sample", "group", paired=True)
    msg = str(e.value)
    assert "No 'sample' value is in more than one condition" in msg
    assert "'donor_id'" in msg and "as a covariate" in msg


async def test_mcp_and_api_take_covariates(client, dataset_manager):
    from tests.test_mcp import _call, _register

    did = _register(dataset_manager)
    r = await _call("composition_test", {"dataset_id": did, "condition_key": "stim", "group_key": "cluster",
                                         "sample_key": "donor_id", "covariates": ["sex"]})
    # Paired by donor already: sex is constant within each donor, so it adds nothing.
    assert r["design"] == "paired" and r["covariates"] == []
    assert "constant within each paired sample value" in r["covariate_notes"][0]
    text = await _call("composition_test", {"dataset_id": did, "condition_key": "stim", "group_key": "cluster",
                                            "covariates": ["nope"]}, error=True)
    assert "nope" in text


def test_unpaired_without_covariates_is_unchanged():
    obs = _obs()
    r = composition_test(obs, "depot", "sample", "group")
    assert r.covariates == [] and r.covariate_notes == []
    d = r.to_dict()
    assert d["covariates"] == [] and "adjusted" not in d["comparison"]
