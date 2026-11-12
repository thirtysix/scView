"""A technology or batch value on one side only (Wilson 2022: one 3' library = 37% of the
control nuclei, every DKD library 5'): warned on cell-level DE and the composition test,
and a covariate value held by one sample is named (core/confounding.py one_sided_levels)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scview.api.v1.de import differential_expression_compare
from scview.core.composition import composition_test
from scview.core.confounding import one_sided_levels
from tests.test_pseudobulk_covariates import _req
from tests.test_pseudobulk_voom import _DM, _counts

DKD = [f"d{i}" for i in range(4)]
CTRL = [f"c{i}" for i in range(4)]


def _kidney():
    """A = DKD (all 5'), B = control with c3 the one 3' library."""
    a = _counts(DKD, CTRL, seed=4)
    a.obs["assay"] = np.where(a.obs["donor"].astype(str) == "c3", "10x 3' v3", "10x 5' v1")
    a.obs["assay"] = a.obs["assay"].astype("category")
    a.obs["library"] = a.obs["donor"].astype(str).astype("category")  # one library per donor
    return a


def test_one_sided_value_is_found_with_its_one_sample():
    a = _kidney()
    sides = {"DKD": (a.obs["cond"] == "A").to_numpy(), "control": (a.obs["cond"] == "B").to_numpy()}
    hits = one_sided_levels(a.obs, sides, skip=["cond"], sample_key="donor")
    assert len(hits) == 1  # library: one per donor, nested in the sides as usual: no warning
    h = hits[0]
    assert h["column"] == "assay" and h["value"] == "10x 3' v3" and h["side"] == "control"
    assert h["share"] == pytest.approx(0.25) and h["samples"] == ["c3"]
    assert "all from one 'donor' value: c3" in h["message"] and "leave c3 out" in h["message"]
    assert "'assay' as a covariate" in h["message"]
    # Below 10% of its side: not worth a warning.
    small = a[~((a.obs["donor"] == "c3").to_numpy() & (np.arange(a.n_obs) % 30 >= 3))].copy()
    sides = {"DKD": (small.obs["cond"] == "A").to_numpy(), "control": (small.obs["cond"] == "B").to_numpy()}
    assert one_sided_levels(small.obs, sides, skip=["cond"], sample_key="donor") == []


def test_a_technology_that_is_the_condition_is_named():
    a = _kidney()
    a.obs["assay"] = a.obs["cond"].map({"A": "5p", "B": "3p"}).astype("category")
    sides = {"A": (a.obs["cond"] == "A").to_numpy(), "B": (a.obs["cond"] == "B").to_numpy()}
    hits = one_sided_levels(a.obs, sides, skip=["cond"])
    assert {h["value"] for h in hits} == {"5p", "3p"}
    # A batch-named column nested in the sides is the usual design: the replicate guard's job.
    a.obs["batch"] = a.obs["library"]
    assert all(h["column"] != "batch" for h in one_sided_levels(a.obs, sides, skip=["cond"]))


@pytest.mark.asyncio
async def test_cell_level_de_warns_and_pseudobulk_names_a_one_sample_covariate():
    a = _kidney()
    res = await differential_expression_compare("x", _req(method="wilcoxon"), _DM(a))
    assert len(res.technology_warnings) == 1
    assert "'assay' value '10x 3' v3' is only in B" in res.technology_warnings[0]
    res = await differential_expression_compare("x", _req(covariates=["assay"]), _DM(a))
    assert res.covariates == ["assay"] and res.technology_warnings == []
    assert any("'assay' value '10x 3' v3' is held by one sample (c3 (B))" in n
               for n in res.covariate_notes)


def test_composition_warns_on_a_one_sided_technology():
    rows = []
    for d in DKD + CTRL:
        for i in range(60):
            rows.append({"donor": d, "cond": "DKD" if d in DKD else "ctrl",
                         "assay": "3p" if d == "c3" else "5p", "group": "PT" if i % 3 else "TAL"})
    obs = pd.DataFrame(rows)
    obs.index = obs.index.astype(str)
    r = composition_test(obs, "cond", "donor", "group")
    assert any("'assay' value '3p' is only in ctrl" in w and "difference in composition" in w
               for w in r.warnings)
