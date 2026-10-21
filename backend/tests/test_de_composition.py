"""Composition of the two sides of a DE between groups (core/de_composition.py)."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from fastapi import HTTPException

from scview.api.v1.de import (
    DECompareRequest,
    DEResponse,
    differential_expression_compare,
    differential_expression_composition,
)
from scview.core import de_composition as dc


def _obs(groups_a: dict[str, int], groups_b: dict[str, int], column: str = "cell_type") -> pd.DataFrame:
    """Cells of condition A then B, with these many cells per ``column`` group."""
    cond, label = [], []
    for c, groups in (("CTRL", groups_a), ("STIM", groups_b)):
        for g, n in groups.items():
            cond += [c] * n
            label += [g] * n
    return pd.DataFrame({"stim": pd.Categorical(cond), column: pd.Categorical(label)},
                        index=[f"c{i}" for i in range(len(cond))])


def _check(obs: pd.DataFrame, **kw) -> dict:
    a = (obs["stim"] == "CTRL").to_numpy()
    b = (obs["stim"] == "STIM").to_numpy()
    kw.setdefault("default", "cell_type")
    return dc.for_comparison(obs, a, b, "CTRL", "STIM", split_by="stim", **kw)


def test_balanced_composition_does_not_warn():
    obs = _obs({"Mono": 300, "T": 500, "B": 200}, {"Mono": 330, "T": 520, "B": 190})
    res = _check(obs)
    assert res["column"] == "cell_type" and res["auto"] is True
    assert (res["n_a"], res["n_b"]) == (1000, 1040)
    assert res["warn"] is False and res["message"] is None
    # Largest difference first, shares column-normalised within each side.
    top = res["groups"][0]
    assert abs(top["diff_points"]) < dc.WARN_POINTS
    mono = next(g for g in res["groups"] if g["group"] == "Mono")
    assert mono["share_a"] == pytest.approx(0.3) and mono["share_b"] == pytest.approx(330 / 1040, abs=1e-4)
    assert sum(g["share_a"] for g in res["groups"]) == pytest.approx(1.0, abs=1e-3)


def test_a_24_point_shift_warns_with_the_shares_in_a_sentence():
    # Monocytes: 18% of CTRL, 42% of STIM.
    obs = _obs({"Monocytes": 180, "T": 620, "B": 200}, {"Monocytes": 420, "T": 380, "B": 200})
    res = _check(obs)
    assert res["warn"] is True
    top = res["groups"][0]
    assert top["group"] == "Monocytes" and top["diff_points"] == pytest.approx(-24.0)
    assert top["flagged"] is True and top["one_sided"] is False
    msg = res["message"]
    assert "Monocytes is 18% of CTRL cells but 42% of STIM cells (-24 points)" in msg
    assert "1 more group differs" in msg  # T: 62% vs 38%
    assert "Compare > Composition" in msg


def test_a_group_absent_from_one_condition():
    obs = _obs({"Mono": 300, "T": 700}, {"T": 700, "pDC": 300})
    res = _check(obs)
    by = {g["group"]: g for g in res["groups"]}
    assert by["Mono"]["n_b"] == 0 and by["Mono"]["share_b"] == 0.0
    assert by["pDC"]["n_a"] == 0 and by["pDC"]["diff_points"] == pytest.approx(-30.0)
    assert res["warn"] is True
    # A group missing from a side is one-sided: it may be a state the comparison creates.
    assert by["Mono"]["one_sided"] and "nearly absent from" in res["message"]


def test_a_shift_in_a_tiny_group_does_not_warn():
    # 8 vs 2 of 40 cells per side: 15 points, but under MIN_CELLS on both sides.
    obs = _obs({"rare": 8, "T": 16, "B": 16}, {"rare": 2, "T": 19, "B": 19})
    res = _check(obs)
    assert res["groups"][0]["group"] == "rare" and res["groups"][0]["diff_points"] == pytest.approx(15.0)
    assert all(not g["flagged"] for g in res["groups"]) and res["warn"] is False


def test_within_one_group_uses_the_subclustering():
    obs = _obs({"Mono": 400, "T": 600}, {"Mono": 400, "T": 600})
    sub = np.where(obs["cell_type"] == "Mono", "Mono,0", "T")
    stim = (obs["stim"] == "STIM").to_numpy()
    mono = (obs["cell_type"] == "Mono").to_numpy()
    # Within Mono, STIM cells are mostly Mono,1.
    sub[mono & stim & (np.arange(len(obs)) % 4 != 0)] = "Mono,1"
    obs["cell_type_sub"] = pd.Categorical(sub)
    a = (obs["stim"] == "CTRL").to_numpy() & mono
    b = stim & mono
    res = dc.for_comparison(obs, a, b, "CTRL", "STIM", split_by="stim", default="cell_type",
                            within_column="cell_type", within_value="Mono")
    assert res["column"] == "cell_type_sub"
    assert "composition is checked by 'cell_type_sub'" in res["note"]
    assert res["warn"] is True


def test_within_one_group_without_a_finer_column_says_so():
    obs = _obs({"Mono": 400, "T": 600}, {"Mono": 400, "T": 600})
    mono = (obs["cell_type"] == "Mono").to_numpy()
    a = (obs["stim"] == "CTRL").to_numpy() & mono
    b = (obs["stim"] == "STIM").to_numpy() & mono
    res = dc.for_comparison(obs, a, b, "CTRL", "STIM", split_by="stim", default="cell_type",
                            within_column="cell_type", within_value="Mono")
    assert res["column"] is None and res["warn"] is False and res["groups"] == []
    assert "already within one 'cell_type' group (Mono)" in res["note"]


def test_a_qc_state_column_is_a_finer_column():
    obs = _obs({"Mono": 400}, {"Mono": 400})
    obs["qc_state"] = pd.Categorical(np.where(np.arange(800) % 3 == 0, "low-RP", "typical"))
    assert dc.finer_column(obs, "cell_type", np.ones(800, bool), {"stim"}) == "qc_state"


def test_composition_by_the_compared_column_is_degenerate():
    obs = _obs({"0": 300}, {"1": 300}, column="leiden")
    obs["stim"] = obs["leiden"]  # compare two clusters by the clustering itself
    res = dc.for_comparison(obs, (obs["leiden"] == "0").to_numpy(), (obs["leiden"] == "1").to_numpy(),
                            "0", "1", split_by="leiden", default="leiden")
    assert res["warn"] is False and res["groups"] == []
    assert "the compared column itself" in res["note"]


def test_sides_made_of_different_groups_are_nested_not_confounded():
    # Two clusters compared, composition by their subclusters: each lies on one side.
    obs = _obs({"0,a": 100, "0,b": 200}, {"1,a": 150, "1,b": 150}, column="leiden_sub")
    res = _check(obs, default="leiden_sub")
    assert res["nested"] is True and res["warn"] is False and res["message"] is None
    assert "composition is the comparison itself" in res["note"]
    assert not any(g["flagged"] for g in res["groups"])


def test_more_than_50_groups_are_capped_with_a_note():
    groups = {f"g{i}": 10 + i for i in range(60)}
    res = _check(_obs(groups, groups))
    assert res["n_groups"] == 60 and len(res["groups"]) == dc.MAX_GROUPS
    assert "60 'cell_type' groups; the 50 with the largest difference" in res["note"]


def test_unlabelled_cells_are_left_out_with_a_note():
    obs = _obs({"Mono": 300, "T": 700}, {"Mono": 300, "T": 700})
    obs["cell_type"] = obs["cell_type"].astype(object)
    obs.iloc[:50, obs.columns.get_loc("cell_type")] = np.nan
    res = _check(obs)
    assert res["n_missing"] == 50 and res["n_a"] == 950
    assert "50 compared cells have no 'cell_type' label" in res["note"]


def test_no_label_column_at_all():
    obs = pd.DataFrame({"stim": pd.Categorical(["CTRL", "STIM"] * 10)})
    res = dc.for_comparison(obs, (obs["stim"] == "CTRL").to_numpy(), (obs["stim"] == "STIM").to_numpy(),
                            "CTRL", "STIM", split_by="stim", default=None)
    assert res["column"] is None and "No clustering" in res["note"]


def test_numbered_clusters_are_named_with_their_column():
    obs = _obs({"0": 180, "1": 820}, {"0": 420, "1": 580}, column="leiden")
    res = _check(obs, default="leiden")
    assert res["message"].startswith("'leiden' group 0 is 18% of CTRL cells")


def test_default_column_is_the_active_clustering():
    a = ad.AnnData(obs=pd.DataFrame({"cluster": ["x"], "leiden": ["0"]}, index=["c0"]))
    assert dc.default_column(a) == "leiden"
    a.uns["scview_active_clustering"] = "cluster"
    assert dc.default_column(a) == "cluster"


# --------------------------------------------------------------------------- endpoints


class _DM:
    def __init__(self, adata):
        self._a = type("A", (), {"adata": adata})()

    async def get_or_load_dataset(self, _id):
        return self._a


def _shifted(n: int = 200, g: int = 20) -> ad.AnnData:
    """CTRL 20% Mono / STIM 50% Mono, no gene difference planted."""
    rng = np.random.default_rng(3)
    obs = _obs({"Mono": n // 5, "T": n - n // 5}, {"Mono": n // 2, "T": n - n // 2})
    a = ad.AnnData(np.abs(rng.normal(5, 1, size=(len(obs), g))).astype("float32"), obs=obs)
    a.var_names = [f"G{i}" for i in range(g)]
    a.uns["scview_active_clustering"] = "cell_type"
    return a


@pytest.mark.asyncio
async def test_compare_attaches_the_composition():
    body = DECompareRequest(split_by="stim", group_a="STIM", group_b="CTRL")
    res = await differential_expression_compare("x", body, _DM(_shifted()))
    comp = res.composition
    assert comp is not None and comp.column == "cell_type" and comp.warn is True
    assert comp.groups[0].diff_points == pytest.approx(30.0)
    assert comp.label_a == "STIM" and "Mono is 50% of STIM cells but 20% of CTRL cells" in comp.message


@pytest.mark.asyncio
async def test_compare_takes_the_composition_column_and_rejects_an_unknown_one():
    a = _shifted()
    a.obs["other"] = pd.Categorical(["u"] * a.n_obs)
    body = DECompareRequest(split_by="stim", group_a="STIM", group_b="CTRL", composition_by="other")
    res = await differential_expression_compare("x", body, _DM(a))
    assert res.composition.column == "other" and res.composition.auto is False
    assert res.composition.warn is False
    with pytest.raises(HTTPException) as e:
        await differential_expression_compare(
            "x", DECompareRequest(split_by="stim", group_a="STIM", group_b="CTRL",
                                  composition_by="nope"), _DM(a))
    assert e.value.status_code == 404


@pytest.mark.asyncio
async def test_get_composition_matches_the_compare_and_answers_before_a_run():
    a = _shifted()
    got = await differential_expression_composition(
        "x", split_by="stim", group_a="STIM", group_b="CTRL", within_column=None,
        within_value=None, column=None, dm=_DM(a))
    assert got.warn is True and got.groups[0].group == "Mono"
    # Within Mono: no finer column, so a note.
    within = await differential_expression_composition(
        "x", split_by="stim", group_a="STIM", group_b="CTRL", within_column="cell_type",
        within_value="Mono", column=None, dm=_DM(a))
    assert within.column is None and "already within one 'cell_type' group" in within.note
    # A side with no cells: a note, not a 400.
    empty = await differential_expression_composition(
        "x", split_by="stim", group_a="STIM", group_b="nobody", within_column=None,
        within_value=None, column=None, dm=_DM(a))
    assert empty.warn is False and "No nobody cells" in empty.note


@pytest.mark.asyncio
async def test_get_composition_rejects_bad_requests():
    a = _shifted()
    for kw, code in (({"split_by": "nope"}, 404), ({"column": "nope"}, 404), ({"group_b": "STIM"}, 400)):
        args = {"split_by": "stim", "group_a": "STIM", "group_b": "CTRL", "within_column": None,
                "within_value": None, "column": None, **kw}
        with pytest.raises(HTTPException) as e:
            await differential_expression_composition("x", dm=_DM(a), **args)
        assert e.value.status_code == code


@pytest.mark.asyncio
async def test_two_clusters_of_the_clustering_get_a_note_not_a_warning():
    # paul15-like: no condition column; compare two clusters of the active clustering.
    a = _shifted()
    got = await differential_expression_composition(
        "x", split_by="cell_type", group_a="Mono", group_b="T", within_column=None,
        within_value=None, column=None, dm=_DM(a))
    assert got.warn is False and "the compared column itself" in got.note


def test_an_old_de_response_without_composition_still_loads():
    old = {"n_selected": 3, "n_rest": 4, "label": "A", "genes": [], "confounding": None}
    res = DEResponse.model_validate(old)
    assert res.composition is None
    assert "composition" in res.model_dump()
