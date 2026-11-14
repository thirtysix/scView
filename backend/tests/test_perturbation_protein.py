"""The protein check of a Mixscape run (core/perturbation_protein.py): a regulator that
moves a protein with no RNA KO class (Papalexi 2021: CMTM6 lowers PD-L1, all its cells NP)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from scview.core import mixscape, perturbation_protein as pp, protein
from tests.test_mixscape import planted


def _with_protein(flip_rep3: bool = False):
    """The planted screen (A: RNA KO in 70% of its cells; B: no RNA effect) with three
    replicates and two proteins: B lowers P1 by 0.5 in every cell (protein only), A's KO
    cells lower P2 by 0.6. ``flip_rep3``: B raises P1 in replicate r3 instead."""
    a, ko = planted()
    rng = np.random.default_rng(1)
    lab = a.obs["perturbation"].astype(str).to_numpy()
    rep = np.array([f"r{i % 3 + 1}" for i in range(a.n_obs)])
    a.obs["rep3"] = pd.Categorical(rep)
    p1 = rng.normal(2.0, 0.3, a.n_obs)
    p2 = rng.normal(1.5, 0.3, a.n_obs)
    b = lab == "B"
    p1[b] -= 0.5
    if flip_rep3:
        p1[b & (rep == "r3")] += 1.0
    p2[ko] -= 0.6
    a.obsm[protein.CLR_KEY] = pd.DataFrame({"P1": p1, "P2": p2}, index=a.obs_names).astype(np.float32)
    return a, ko


def test_bh_matches_the_definition():
    p = np.array([0.01, 0.04, 0.03, np.nan, 0.5])
    adj = pp._bh(p)
    # sorted 0.01, 0.03, 0.04, 0.5 (m = 4): 0.04, 0.0533, 0.0533, 0.5
    assert np.allclose(adj[[0, 2, 1, 4]], [0.04, 0.04 * 4 / 3, 0.04 * 4 / 3, 0.5]) and np.isnan(adj[3])


def test_a_protein_only_target_is_flagged_and_an_rna_ko_target_is_not():
    a, ko = _with_protein()
    res, _ = mixscape.run(a, mixscape.Params(lda=False, replicate_column="rep3"))
    pr = res.protein
    assert pr is not None and pr["proteins"] == ["P1", "P2"] and pr["replicate_column"] == "rep3"
    by = {(r["target"], r["protein"]): r for r in pr["rows"]}
    b1 = by[("B", "P1")]
    assert b1["ko_fraction"] == 0 and b1["unit"] == "replicate" and b1["n_replicates"] == 3
    assert b1["all"]["diff"] == pytest.approx(-0.5, abs=0.12) and b1["same_sign"] == 3
    assert b1["significant"] and b1["flag"]
    assert [f["target"] + f["protein"] for f in pr["flags"]] == ["BP1"]
    assert pr["flags"][0]["text"].endswith("0% of its cells called KO on RNA.")
    a2 = by[("A", "P2")]  # A's KO cells carry the shift, its NP cells do not
    assert a2["ko"]["diff"] == pytest.approx(-0.6, abs=0.12) and a2["ko"]["padj"] < 1e-6
    assert abs(a2["np"]["diff"]) < 0.15 and a2["significant"] and not a2["flag"]  # A has a KO class
    assert not by[("B", "P2")]["significant"] and not by[("A", "P1")]["significant"]
    assert pr["default_protein"] in ("P1", "P2")
    assert any("CLR per cell over 2 proteins" in n for n in pr["notes"])
    # Plain JSON (no NaN): the SSE event and the saved record.
    json.dumps(res.to_dict(), allow_nan=False)
    out = mixscape.save(a, res, _)
    assert json.loads(a.uns[mixscape.UNS_KEY])["protein"]["flags"][0]["target"] == "B"
    assert out["protein"]["flags"]


def test_replicates_that_disagree_are_not_significant_but_cells_alone_are():
    a, _ = _with_protein(flip_rep3=True)
    res, _c = mixscape.run(a, mixscape.Params(lda=False, replicate_column="rep3"))
    b1 = next(r for r in res.protein["rows"] if (r["target"], r["protein"]) == ("B", "P1"))
    assert b1["unit"] == "replicate" and b1["same_sign"] == 2 and not b1["flag"]
    assert b1["all"]["padj"] < 0.05 and abs(b1["all"]["diff"]) >= pp.MIN_SHIFT  # the cells alone would say yes
    assert any("replicates in 'rep3'" in n for n in res.protein["notes"])
    # Without a replicate column the same cells are judged on cells, and the caption says so.
    res2, _c = mixscape.run(a, mixscape.Params(lda=False))
    b1 = next(r for r in res2.protein["rows"] if (r["target"], r["protein"]) == ("B", "P1"))
    assert b1["unit"] == "cell" and b1["flag"] and "replicates" not in b1
    assert res2.protein["replicate_column"] is None
    assert any("No replicate column" in n for n in res2.protein["notes"])
    # Two replicates: too few for the replicate unit, judged on cells.
    res3, _c = mixscape.run(a, mixscape.Params(lda=False, replicate_column="rep"))
    b1 = next(r for r in res3.protein["rows"] if (r["target"], r["protein"]) == ("B", "P1"))
    assert b1["unit"] == "cell" and b1["n_replicates"] == 2 and b1["p_replicate"] is None and "note" in b1


def test_no_protein_no_check_and_the_copilot_line_names_the_flag():
    a, _ = planted()
    res, _c = mixscape.run(a, mixscape.Params(lda=False))
    assert res.protein is None
    a, _ = _with_protein()
    res, cols = mixscape.run(a, mixscape.Params(lda=False, replicate_column="rep3"))
    mixscape.save(a, res, cols)

    class _Adaptor:
        adata = a

    mixscape._last[_Adaptor] = res  # a class is weak-referenceable: the remembered run
    try:
        lines, _d = mixscape.grounding_lines(_Adaptor)
    finally:
        mixscape._last.pop(_Adaptor, None)
    text = "\n".join(lines)
    assert "protein effect without an RNA KO class" in text and "B P1 -0.5" in text and "judged on replicates" in text


def test_a_constant_protein_and_few_cells_give_no_test():
    lab = np.array(["NT"] * 20 + ["A"] * 10 + ["C"] * 3)
    glob = np.where(lab == "NT", "NT", "NP")
    v = np.ones((lab.size, 1))
    out = pp.check(v, ["K"], lab, glob, "NT", [{"target": "A", "ko_fraction": 0.0},
                                              {"target": "C", "ko_fraction": 0.0}])
    by = {r["target"]: r for r in out["rows"]}
    assert by["A"]["all"]["p"] == 1.0 and by["A"]["all"]["diff"] == 0 and not by["A"]["flag"]
    assert by["A"]["ko"] is None  # no KO cells
    assert by["C"]["all"] is None and not by["C"]["significant"]  # 3 cells: fewer than MIN_CELLS
    json.dumps(out, allow_nan=False)


def _guides_with_protein():
    from tests.test_perturbation_api import _with_guides

    a, labels, ko = _with_guides()
    rng = np.random.default_rng(2)
    p1 = rng.normal(2.0, 0.3, a.n_obs)
    p1[labels == "B"] -= 0.5
    a.obs["rep3"] = pd.Categorical([f"r{i % 3 + 1}" for i in range(a.n_obs)])
    a.obsm[protein.CLR_KEY] = pd.DataFrame({"PDL1": p1, "CD86": rng.normal(1.0, 0.3, a.n_obs)},
                                           index=a.obs_names).astype(np.float32)
    return a, ko


async def test_the_route_streams_and_serves_the_protein_check(client, dataset_manager):
    from scview.core import results_store
    from tests.test_hvg_sensitivity import _events, _register

    a, _ko = _guides_with_protein()
    did = _register(dataset_manager, a, "pp1")
    assert (await client.post(f"/api/v1/datasets/{did}/perturbation/assign", json={})).status_code == 200
    r = await client.post(f"/api/v1/datasets/{did}/perturbation/mixscape/run-stream",
                          json={"split_by": "rep3", "lda": False})
    events = _events(r.text)
    assert any(e == "progress" and d.get("step") == "protein" for e, d in events)
    result = events[-2][1]
    assert [f["target"] + f["protein"] for f in result["protein"]["flags"]] == ["BPDL1"]
    results_store.flush()
    got = (await client.get(f"/api/v1/datasets/{did}/perturbation/mixscape")).json()
    assert got["protein"]["flags"][0]["unit"] == "replicate"
