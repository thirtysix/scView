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


# --- v0.80: the values tested (layer), default by panel size -------------------------------------

def _compositional(n_proteins: int = 4, seed: int = 5):
    """The planted screen with antibody COUNTS on ``n_proteins`` proteins: B lowers the most
    abundant one (P0) to 0.4x in every cell, and nothing else; A's KO cells change no protein.
    On the per-cell CLR of a small panel the fall of P0 lifts the others (compositional)."""
    a, ko = planted()
    rng = np.random.default_rng(seed)
    lab = a.obs["perturbation"].astype(str).to_numpy()
    a.obs["rep3"] = pd.Categorical([f"r{i % 3 + 1}" for i in range(a.n_obs)])
    means = np.r_[400.0, rng.uniform(40, 120, n_proteins - 1)]
    size = rng.lognormal(0, 0.3, a.n_obs)  # droplet capture, the same for every target
    mu = size[:, None] * means[None, :]
    mu[lab == "B", 0] *= 0.4
    counts = rng.poisson(mu).astype(np.float32)
    names = [f"P{j}" for j in range(n_proteins)]
    a.obsm[protein.COUNTS_KEY] = pd.DataFrame(counts, index=a.obs_names, columns=names)
    a.obsm[protein.CLR_KEY] = pd.DataFrame(protein.clr(counts), index=a.obs_names, columns=names)
    return a, ko


def test_small_panel_defaults_to_log_counts_and_clr_shows_the_compositional_rise():
    a, _ko = _compositional(4)
    assert pp.default_layer(a) == pp.COUNTS and pp.resolve_layer(a, None) == pp.COUNTS
    res, cols = mixscape.run(a, mixscape.Params(lda=False, replicate_column="rep3"))
    pr = res.protein
    assert pr["layer"] == pp.COUNTS and pr["layer_choice"] == "auto" and pr["unit_name"] == "log1p counts"
    assert res.params["protein_layer"] == "auto"
    by = {(r["target"], r["protein"]): r for r in pr["rows"]}
    assert by[("B", "P0")]["all"]["diff"] == pytest.approx(np.log(0.4), abs=0.12) and by[("B", "P0")]["flag"]
    assert not any(by[("B", f"P{j}")]["significant"] for j in (1, 2, 3))  # nothing else moved
    assert [f["target"] + f["protein"] for f in pr["flags"]] == ["BP0"] and "(log1p counts)" in pr["flags"][0]["text"]
    assert "fewer than 15: log counts by default" in pr["notes"][0]
    assert pr["reference_shift"] is None  # only CLR has a reference
    # The same run on the per-cell CLR: P0's fall lifts the unchanged proteins past the 0.1 line.
    clr = pp.run(a, *_used(a, res, cols), layer=pp.CLR)
    cb = {(r["target"], r["protein"]): r for r in clr["rows"]}
    lifted = [j for j in (1, 2, 3) if cb[("B", f"P{j}")]["all"]["diff"] >= pp.MIN_SHIFT and cb[("B", f"P{j}")]["significant"]]
    assert lifted, [cb[("B", f"P{j}")]["all"]["diff"] for j in (1, 2, 3)]
    assert clr["reference_shift"]["B"] < -0.1 and abs(clr["reference_shift"]["A"]) < 0.05
    assert any(n.startswith("CLR per cell over 4 proteins") and "compositional" in n for n in clr["notes"])
    assert any("The CLR reference" in n and "B -0." in n for n in clr["notes"])
    json.dumps(clr, allow_nan=False)


def _used(a, res, cols):
    """The arguments ``perturbation_protein.run`` gets from a Mixscape run on ``a``."""
    pl = mixscape.plan(a, mixscape.Params(lda=False, replicate_column="rep3"))
    assert np.array_equal(pl.rows, cols["rows"])
    return (pl.rows, pl.labels, np.asarray(cols[mixscape.GLOBAL]).astype(str), "NT", res.targets, pl.replicates,
            pl.replicate_column)


def test_a_large_panel_keeps_clr_and_a_panel_without_counts_says_why():
    a, _ko = _compositional(16)
    assert pp.default_layer(a) == pp.CLR
    res, _c = mixscape.run(a, mixscape.Params(lda=False, replicate_column="rep3"))
    assert res.protein["layer"] == pp.CLR and res.protein["notes"][0] == "16 proteins: CLR per cell by default."
    assert not any(n.startswith("CLR per cell over") for n in res.protein["notes"])  # not a small panel
    opts = {o["layer"]: o for o in res.protein["layers"]}
    assert opts[pp.CLR]["available"] and not opts[pp.CLR]["compositional"] and opts[pp.COUNTS]["available"]
    # CLR only (an older upload, a foreign object): CLR even on 2 proteins, and the note says why.
    b, _ = _with_protein()
    assert pp.default_layer(b) == pp.CLR
    res, _c = mixscape.run(b, mixscape.Params(lda=False, replicate_column="rep3"))
    assert res.protein["notes"][0] == "2 proteins: CLR per cell, as the dataset keeps no antibody counts."
    opts = {o["layer"]: o for o in res.protein["layers"]}
    assert not opts[pp.COUNTS]["available"] and "no antibody counts" in opts[pp.COUNTS]["reason"]
    with pytest.raises(pp.ProteinLayerError, match="no antibody counts"):
        pp.resolve_layer(b, pp.COUNTS)


def test_layer_choice_is_checked_and_dsb_says_when_it_is_missing_or_compositional():
    from scview.core import dsb

    a, _ko = _compositional(4)
    with pytest.raises(pp.ProteinLayerError, match="Unknown protein layer") as e:
        pp.resolve_layer(a, "protein_tpm")
    assert e.value.status == 422
    with pytest.raises(pp.ProteinLayerError, match="empty droplets"):
        pp.resolve_layer(a, pp.DSB)
    with pytest.raises(mixscape.MixscapeError, match="No DSB values") as e:
        mixscape.plan(a, mixscape.Params(protein_layer=pp.DSB))
    assert e.value.status == 422
    # DSB run with step II on a 4-protein panel: offered, marked compositional, said in the notes.
    vals, record = dsb.compute(a, mode="model_negative")
    a.obsm[dsb.DSB_KEY] = pd.DataFrame(vals, index=a.obs_names, columns=list(a.obsm[protein.COUNTS_KEY].columns))
    a.uns[dsb.UNS_RECORD] = json.dumps(record)
    opt = next(o for o in pp.layers(a) if o["layer"] == pp.DSB)
    assert opt["available"] and opt["compositional"] and "step II" in opt["reason"]
    res, _c = mixscape.run(a, mixscape.Params(lda=False, replicate_column="rep3", protein_layer=pp.DSB))
    assert res.protein["layer"] == pp.DSB and res.protein["layer_choice"] == pp.DSB
    assert any("DSB step II" in n for n in res.protein["notes"])


def test_older_results_read_as_clr():
    assert pp.layer_of(None) == pp.CLR and pp.layer_of({"rows": []}) == pp.CLR
    assert pp.layer_of({"layer": pp.COUNTS}) == pp.COUNTS
    old = {"rows": [], "flags": [{"target": "CMTM6", "protein": "PDL1", "diff": -0.19, "unit": "replicate",
                                  "ko_fraction": 0.0, "text": "x"}]}
    r = mixscape.MixscapeResult(params={}, labels="l", control="NT", split_by=None, n_cells=1, n_control=1,
                                n_targets=1, n_genes=1, gene_source="", expression_source="", targets=[],
                                protein=old)
    assert "vs the controls, CLR, judged on replicates" in mixscape._protein_words(r)
    from scview.mcp.modalities import _protein_reply

    assert _protein_reply(old, 5)["layer"] == pp.CLR


async def test_the_route_shows_the_run_on_another_layer_without_saving(client, dataset_manager):
    from tests.test_hvg_sensitivity import _register
    from tests.test_perturbation_api import _with_guides

    a, labels, _ko = _with_guides()
    rng = np.random.default_rng(4)
    means = np.array([400.0, 80.0, 60.0])
    mu = rng.lognormal(0, 0.3, a.n_obs)[:, None] * means[None, :]
    mu[labels == "B", 0] *= 0.4
    counts = rng.poisson(mu).astype(np.float32)
    a.obsm[protein.COUNTS_KEY] = pd.DataFrame(counts, index=a.obs_names, columns=["PDL1", "CD86", "PDL2"])
    a.obsm[protein.CLR_KEY] = pd.DataFrame(protein.clr(counts), index=a.obs_names, columns=["PDL1", "CD86", "PDL2"])
    a.obs["rep3"] = pd.Categorical([f"r{i % 3 + 1}" for i in range(a.n_obs)])
    did = _register(dataset_manager, a, "pp2")
    url = f"/api/v1/datasets/{did}/perturbation/mixscape"
    assert (await client.get(f"{url}/protein")).status_code == 404  # no run yet
    assert (await client.post(f"/api/v1/datasets/{did}/perturbation/assign", json={})).status_code == 200
    bad = await client.post(f"{url}/run-stream", json={"split_by": "rep3", "lda": False, "protein_layer": "protein_dsb"})
    assert bad.status_code == 422 and "empty droplets" in bad.json()["detail"]
    r = await client.post(f"{url}/run-stream", json={"split_by": "rep3", "lda": False})
    from tests.test_hvg_sensitivity import _events

    result = _events(r.text)[-2][1]
    assert result["protein"]["layer"] == pp.COUNTS and result["params"]["protein_layer"] == "auto"
    assert [f["target"] + f["protein"] for f in result["protein"]["flags"]] == ["BPDL1"]
    v = await client.get(f"{url}/protein", params={"layer": pp.CLR})
    assert v.status_code == 200
    view = v.json()
    assert view["layer"] == pp.CLR and view["layer_choice"] == pp.CLR and view["default_layer"] == pp.COUNTS
    assert view["reference_shift"]["B"] < -0.1
    same = {(x["target"], x["protein"]): x["all"]["n"] for x in view["rows"]}
    assert same == {(x["target"], x["protein"]): x["all"]["n"] for x in result["protein"]["rows"]}  # same cells
    assert (await client.get(f"{url}/protein", params={"layer": "nope"})).status_code == 422
    assert (await client.get(f"{url}/protein", params={"layer": pp.DSB})).status_code == 422
    from scview.core import results_store

    results_store.flush()
    got = (await client.get(url)).json()["protein"]
    assert got["layer"] == pp.COUNTS  # the run keeps its layer
    assert not next(x for x in got["layers"] if x["layer"] == pp.DSB)["available"]
    # DSB computed after the run: offered from then on, on the stored run too.
    from scview.core import dsb

    adata = (await dataset_manager.get_or_load_dataset(did)).adata
    vals, record = dsb.compute(adata, mode="model_negative")
    adata.obsm[dsb.DSB_KEY] = pd.DataFrame(vals, index=adata.obs_names, columns=["PDL1", "CD86", "PDL2"])
    adata.uns[dsb.UNS_RECORD] = json.dumps(record)
    got = (await client.get(url)).json()["protein"]
    assert next(x for x in got["layers"] if x["layer"] == pp.DSB)["available"] and got["layer"] == pp.COUNTS
    over = (await client.get(f"/api/v1/datasets/{did}/perturbation")).json()["mixscape"]["protein"]
    assert next(x for x in over["layers"] if x["layer"] == pp.DSB)["available"]
    assert (await client.get(f"{url}/protein", params={"layer": pp.DSB})).json()["layer"] == pp.DSB
