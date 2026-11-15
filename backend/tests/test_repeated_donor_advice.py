"""The repeated-donor warning knows the design (Emont 2022, mouse x depot samples): no
warning when the person column is a covariate (blocked), "block on it" when the compared
factor varies within a person (depot inside a mouse), and the factor a person's samples
differ in when the comparison is between people (diet)."""

from __future__ import annotations

import pandas as pd
import pytest

from scview.api.v1.de import differential_expression_compare
from scview.core.composition import composition_test
from scview.core.replicates import repeated_donors
from tests.test_pseudobulk_voom import _DM, _counts
from tests.test_pseudobulk_covariates import _req

MICE = [f"m{i}" for i in range(6)]


def _depots(a):
    """``donor`` is the sample (mouse x depot); A = ING, B = PG."""
    a.obs["donor_id"] = a.obs["donor"].astype(str).str.split("_").str[0].astype("category")
    a.obs["depot"] = a.obs["cond"].map({"A": "ING", "B": "PG"}).astype("category")
    return a


@pytest.mark.asyncio
async def test_within_donor_contrast_says_block_and_a_covariate_silences_it():
    a = _depots(_counts([f"{m}_ING" for m in MICE], [f"{m}_PG" for m in MICE], seed=1))
    res = await differential_expression_compare("x", _req(), _DM(a))
    w = [x for x in res.replicate_warnings if "not independent" in x]
    assert len(w) == 1 and "6 of 6 'donor_id' values" in w[0]
    assert "varies within 'donor_id' (6 of them have samples on both sides)" in w[0]
    assert "add 'donor_id' as a covariate" in w[0] and "person" not in w[0]
    # Blocked on the mouse: the design compares each mouse's depots, nothing to warn about.
    res = await differential_expression_compare("x", _req(covariates=["donor_id"]), _DM(a))
    assert res.design == "~donor_id + condition"
    assert not any("not independent" in x for x in res.replicate_warnings)
    assert res.n_donors_a == 6 and res.n_donors_b == 6


@pytest.mark.asyncio
async def test_between_donor_contrast_names_what_a_donors_samples_differ_in():
    hfd = [f"h{i}_{d}" for i in range(3) for d in ("ING", "PG")]
    chow = [f"c{i}_{d}" for i in range(3) for d in ("ING", "PG")]
    a = _counts(hfd, chow, seed=2)
    a.obs["donor_id"] = a.obs["donor"].astype(str).str.split("_").str[0].astype("category")
    a.obs["depot"] = a.obs["donor"].astype(str).str.split("_").str[1].astype("category")
    res = await differential_expression_compare("x", _req(), _DM(a))
    w = [x for x in res.replicate_warnings if "not independent" in x][0]
    assert "samples differ in 'depot'" in w and "test within one 'depot' value" in w
    assert "use 'donor_id' as the replicate column" in w


def test_composition_gives_the_same_advice():
    rows = []
    for m in range(5):
        for depot in ("ING", "PG"):
            for i in range(40):
                grp = "g0" if (i % (3 if depot == "ING" else 2)) == 0 else "g1"
                rows.append({"mouse_donor": f"m{m}", "sample": f"m{m}_{depot}", "depot": depot,
                             "group": grp})
    obs = pd.DataFrame(rows)
    obs.index = obs.index.astype(str)
    r = composition_test(obs, "depot", "sample", "group", donor_key="mouse_donor")
    assert r.repeated_donors["within"] is True
    assert any("add 'mouse_donor' as a covariate" in w for w in r.warnings)


def test_blocked_returns_no_message():
    obs = pd.DataFrame({"s": ["a1", "a2", "b1", "b2"] * 5, "donor_id": ["a", "a", "b", "b"] * 5,
                        "side": ["x", "y", "x", "y"] * 5})
    sides = [(obs["side"] == "x").to_numpy(), (obs["side"] == "y").to_numpy()]
    rep = repeated_donors(obs, "s", sides, covariates=["donor_id"])
    assert rep["blocked"] is True and rep["message"] is None and rep["within"] is True
    rep = repeated_donors(obs, "s", sides)
    assert rep["blocked"] is False and "block on it" in rep["message"]


@pytest.mark.asyncio
async def test_partly_paired_says_how_to_compare_only_the_donors_on_both_sides():
    # Reck 2025's kidney: UUO donors on both sides (inflammatory vs injured PT), some controls
    # on one side only; disease = UUO holds only donors on both sides.
    part = _counts(["u1", "u2", "u3", "c1"], ["u1", "u2", "u3", "c2", "c3"], seed=10)
    part.obs["disease"] = part.obs["donor"].astype(str).str[0].map({"u": "UUO", "c": "normal"}).astype("category")
    res = await differential_expression_compare("x", _req(pb_method="voom"), _DM(part))
    (w,) = [x for x in res.replicate_warnings if "on both sides" in x]
    assert "Paired: restrict to them with within_column 'donor' and within_value ['u1', 'u2', 'u3'] " \
           "(replicate_key 'donor')" in w
    assert "or with within_column 'disease' = 'UUO' (its 3 donors are all on both sides)" in w
    # Doing what it says: a paired design, and the warning is gone.
    res = await differential_expression_compare(
        "x", _req(pb_method="voom", within_column="disease", within_value="UUO"), _DM(part))
    assert res.design == "~donor + condition" and not any("on both sides" in x for x in res.replicate_warnings)
    res = await differential_expression_compare(
        "x", _req(pb_method="voom", within_column="donor", within_value=["u1", "u2", "u3"],
                  replicate_key="donor"), _DM(part))
    assert res.design == "~donor + condition"


def test_paired_subset_needs_a_value_with_only_shared_donors():
    from scview.core.replicates import paired_subset

    obs = pd.DataFrame({"donor": ["a", "a", "b", "b", "c", "d"], "grp": ["x", "x", "x", "x", "y", "y"],
                        "mixed": ["p", "q", "p", "p", "p", "p"]})
    mask = [True] * 6
    assert paired_subset(obs, "donor", ["a", "b"], mask) == ("grp", "x", 2)
    assert paired_subset(obs, "donor", ["a"], mask) is None  # x also holds b; one donor is not a pair
