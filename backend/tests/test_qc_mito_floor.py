"""The per-type % mito floor (qc_filter ``mito_type_floor``, PipelineParams
``qc_mito_type_floor``): per type, never stricter than the global rule.

Fixture values are chosen so every median, MAD and bound is worked out by hand:
- L, a low-baseline type: 30 / 40 / 26 cells at 0.5 / 1.0 / 1.5 % and 4 at 4 %:
  median 1, MAD 0.5, own bound 2.5 (its four 4 % cells go per type);
- H, a high-baseline type: 10 / 12 / 14 % and 4 at 25 %: median 12, MAD 2, bound 18;
- Q, 60 cells at 8 %: no spread, so it keeps the global rule.
Counts and genes are the same in every cell (MAD 0: no bound), so only % mito removes.

Samples (MAD mode): d1 holds 98 L cells (two of the 4 % ones) and Q; d2 holds H and
the other two 4 % L cells.
- d1, 158 cells: median 1.5 (the 79th/80th values), MAD 1.0 -> bound 4.5;
- d2, 102 cells: median 12, MAD 2 -> bound 18, under a 13 % ceiling 13.
"""

from __future__ import annotations

import json
import subprocess
import sys

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from scview.core import mito_by_type, provenance, qc_filter, replay_script, run_manifest
from scview.core.anndata_adaptor import per_cell_qc
from scview.core.pipeline import PipelineParams, _run_filtering, run_pipeline
from scview.core.qc_filter import Thresholds

API = "/api/v1"

L = [0.5] * 30 + [1.0] * 40 + [1.5] * 26 + [4.0] * 4
H = [10.0] * 30 + [12.0] * 40 + [14.0] * 26 + [25.0] * 4
Q = [8.0] * 60
TYPES = ("L", "H", "Q")


def _cells() -> ad.AnnData:
    pct = L + H + Q
    labels = ["L"] * len(L) + ["H"] * len(H) + ["Q"] * len(Q)
    donor = ["d1"] * 96 + ["d1", "d1", "d2", "d2"] + ["d2"] * len(H) + ["d1"] * len(Q)
    n = len(pct)
    obs = pd.DataFrame({"total_counts": 2000.0, "n_genes_by_counts": 800.0, "pct_counts_mt": pct},
                       index=[f"c{i}" for i in range(n)])
    obs["cell_type"] = pd.Categorical(labels, categories=list(TYPES))
    obs["Donor"] = pd.Categorical(donor)
    a = ad.AnnData(sp.csr_matrix((n, 3), dtype=np.float32), obs=obs)
    a.var_names = ["MT-CO1", "G1", "G2"]
    return a


FIXED = PipelineParams(qc_threshold_mode="fixed", min_genes=0, max_pct_mt=13.0, min_cells=0,
                       qc_mito_mode="per_type")
MAD = PipelineParams(qc_threshold_mode="mad", max_pct_mt=13.0, min_cells=0, qc_mito_mode="per_type",
                     qc_sample_column="Donor")


def _with(p: PipelineParams, **kw) -> PipelineParams:
    return PipelineParams(**{**p.__dict__, **kw})


def _by_type(a: ad.AnnData, mask: np.ndarray) -> dict[str, int]:
    labels = a.obs["cell_type"].astype(str).to_numpy()
    return {t: int(mask[labels == t].sum()) for t in TYPES}


def _judge(a, p):
    return qc_filter.judge_types(a, per_cell_qc(a), Thresholds.from_params(p))


def test_the_hand_worked_bounds():
    a = _cells()
    _, bounds, types = _judge(a, MAD)
    rows = {r["name"]: r for r in types.rows}
    assert (rows["L"]["median"], rows["L"]["mad"], rows["L"]["bound"]) == (1.0, 0.5, 2.5)
    assert (rows["H"]["median"], rows["H"]["mad"], rows["H"]["bound"]) == (12.0, 2.0, 18.0)
    assert (rows["Q"]["rule"], rows["Q"]["why"]) == ("global", "no_spread")
    samples = {g["name"]: g for g in bounds.groups}
    assert samples["d1"]["n_cells"] == 158 and samples["d1"]["max_pct_mt"] == pytest.approx(4.5)
    assert samples["d2"]["n_cells"] == 102 and samples["d2"]["max_pct_mt"] == pytest.approx(13.0)


@pytest.mark.parametrize("params", [FIXED, MAD], ids=["fixed", "mad"])
def test_floor_off_is_todays_rule_and_floor_on_never_removes_more_than_global(params):
    a = _cells()
    # Fixed: the global rule is the 13 % ceiling. MAD: d1's 4.5 % for L and Q, d2's 13 % for H.
    glob, _ = qc_filter.judge(a, per_cell_qc(a), Thresholds.from_params(_with(params, qc_mito_mode="global")))
    expect_global = {"L": 0, "H": 30, "Q": 0} if params is FIXED else {"L": 0, "H": 30, "Q": 60}
    assert _by_type(a, glob["max_pct_mt"]) == expect_global
    off, _, t_off = _judge(a, params)
    on, _, t_on = _judge(a, _with(params, qc_mito_type_floor=True))
    # Floor off: as per type always was (L's four 4 % cells go, below the global rule).
    assert _by_type(a, off["max_pct_mt"]) == {"L": 4, "H": 4, "Q": expect_global["Q"]}
    # Floor on: L keeps them (its bound lifted to the global rule's), H keeps its own 18 %.
    assert _by_type(a, on["max_pct_mt"]) == {"L": 0, "H": 4, "Q": expect_global["Q"]}
    for t in TYPES:
        assert _by_type(a, on["max_pct_mt"])[t] <= expect_global[t]
    # cell by cell: removed with the floor = removed by both rules
    assert (on["max_pct_mt"] == (off["max_pct_mt"] & glob["max_pct_mt"])).all()
    # both previews' tables carry both rules, whichever is applied
    for types in (t_off, t_on):
        rows = {r["name"]: r for r in types.to_dict()["types"]}
        assert {k: (r["removed_global"], r["removed_type"], r["removed_floor"]) for k, r in rows.items()} == {
            "L": (0, 4, 0), "H": (30, 4, 4), "Q": (expect_global["Q"],) * 3}
        assert (rows["L"]["floor_uses"], rows["L"]["n_floored"]) == ("global", 100)
        assert (rows["H"]["floor_uses"], rows["H"]["n_floored"]) == ("type", 0)
        assert rows["Q"]["floor_uses"] is None
    assert t_off.to_dict()["floor"] is False and t_on.to_dict()["floor"] is True
    assert {r["name"]: r["removed"] for r in t_on.to_dict()["types"]} == {"L": 0, "H": 4, "Q": expect_global["Q"]}


def test_in_mad_mode_the_global_bound_is_each_cells_own_samples():
    a = _cells()
    _, _, types = _judge(a, _with(MAD, qc_mito_type_floor=True))
    rows = {r["name"]: r for r in types.to_dict()["types"]}
    assert rows["L"]["global_bound"] == [4.5, 13.0]  # d1's bound, and d2's (the ceiling)
    assert rows["H"]["global_bound"] == [13.0, 13.0]
    # a 4 % L cell in d1 (bound 4.5) and one in d2 (13) are both kept with the floor
    labels, pct = a.obs["cell_type"].astype(str).to_numpy(), a.obs["pct_counts_mt"].to_numpy()
    out = (labels == "L") & (pct == 4.0)
    assert set(a.obs["Donor"][out]) == {"d1", "d2"}
    on, _, _ = _judge(a, _with(MAD, qc_mito_type_floor=True))
    assert not on["max_pct_mt"][out].any()


def test_a_type_split_across_samples_can_use_both_bounds():
    """``floor_rows`` on a hand-built type: own bound 8 % in samples whose global
    bounds are 4.5 % (its own is higher) and 13 % (the global one is)."""
    codes = np.array([0, 0, 0, 0, 1, 1])
    types = qc_filter.TypeBounds("t", None, 3.0, codes, [{"name": "M", "bound": 8.0}, {"name": "N", "bound": None}],
                                 own=np.array([True] * 4 + [False] * 2),
                                 hi=np.array([8.0] * 4 + [np.inf] * 2),
                                 global_hi=np.array([4.5, 4.5, 13.0, np.inf, 13.0, 13.0]))
    rows = [dict(r) for r in types.rows]
    qc_filter.floor_rows(types, rows)
    assert (rows[0]["floor_uses"], rows[0]["n_floored"], rows[0]["global_bound"]) == ("mixed", 2, [4.5, 13.0])
    assert (rows[1]["floor_uses"], rows[1]["n_floored"]) == (None, 0)


def test_without_a_global_mito_rule_the_floor_removes_nothing_for_mito():
    a = _cells()
    p = _with(FIXED, max_pct_mt=None, qc_mito_type_floor=True)
    removed, _, types = _judge(a, p)
    assert not removed["max_pct_mt"].any()
    assert {r["name"]: r["floor_uses"] for r in types.to_dict()["types"]}["H"] == "global"  # inf > 18


@pytest.mark.parametrize("params", [FIXED, MAD], ids=["fixed", "mad"])
@pytest.mark.parametrize("floor", [False, True])
def test_the_preview_is_the_filtering_steps_masks(params, floor):
    a = _cells()
    p = _with(params, qc_mito_type_floor=floor)
    out = qc_filter.preview(a, per_cell_qc(a), Thresholds.from_params(p))
    kept, found = _run_filtering(a.copy(), p)
    gone = ~np.isin(a.obs_names, kept.obs_names)
    removed, _, _ = _judge(a, p)
    assert (gone == ~qc_filter.keep_mask(removed, a.n_obs)).all()
    assert out["n_removed"] == int(gone.sum()) and out["thresholds"]["mito_type_floor"] is floor
    lost = _by_type(a, gone)
    assert {r["name"]: r["removed"] for r in out["mito_types"]["types"]} == lost
    assert found["qc_mito_type_floor"] is floor and found["qc_mito_types"]["floor"] is floor
    assert found["qc_mito_types"]["types"] == out["mito_types"]["types"]
    assert lost == ({"L": 0, "H": 4} if floor else {"L": 4, "H": 4}) | {"Q": 0 if params is FIXED else 60}
    json.dumps(out, allow_nan=False)


def test_the_record_and_provenance_say_the_floor_was_on(tmp_path):
    a = _cells()
    out, res = run_pipeline(a, ["filtering"], _with(FIXED, qc_mito_type_floor=True))
    assert res.errors == {}
    p = provenance.read_provenance(out)["history"][-1]["params"]
    assert p["qc_mito_type_floor"] is True and p["qc_mito_mode"] == "per_type"
    path = tmp_path / "f.h5ad"
    out.write_h5ad(path)
    back = ad.read_h5ad(path)
    rec = back.uns[qc_filter.RECORD_KEY]
    assert bool(rec["mito_type_floor"]) is True
    assert qc_filter.recorded_mito_rule(rec)["mito_type_floor"] is True
    assert qc_filter.recorded_audit(back)["mito_type_floor"] is True
    # floor off: the record is as it always was (no floor key)
    off, _ = run_pipeline(_cells(), ["filtering"], FIXED)
    assert "mito_type_floor" not in off.uns[qc_filter.RECORD_KEY]
    assert "mito_type_floor" not in qc_filter.recorded_mito_rule(off.uns[qc_filter.RECORD_KEY])
    assert provenance.read_provenance(off)["history"][-1]["params"]["qc_mito_type_floor"] is False


def test_old_records_mean_the_floor_was_off():
    """A filtering recorded before the floor existed has no ``qc_mito_type_floor``:
    it reads as off everywhere (Thresholds, the card, the replay script)."""
    a = _cells()
    out, _ = run_pipeline(a, ["filtering"], FIXED)
    hist = provenance.read_provenance(out)
    params = hist["history"][-1]["params"]
    params.pop("qc_mito_type_floor")
    for r in params["qc_mito_types"]["types"]:  # an old record's rows had no floor columns either
        for k in ("removed_type", "removed_floor", "floor_uses", "n_floored", "global_bound"):
            r.pop(k, None)
    params["qc_mito_types"].pop("floor")
    out.uns[provenance.UNS_KEY] = json.dumps(hist)
    assert Thresholds.from_params(PipelineParams()).mito_type_floor is False
    card = mito_by_type.mito_by_type(out)
    assert card.filter_type_floor is False and "never stricter" not in card.summary
    assert {t.group: t.applied_floor for t in card.types} == {"L": None, "H": None, "Q": None}
    script = replay_script.build_script(out, dataset_id="t", dataset_name="t")
    assert "GLOBAL_MITO" not in script


def test_an_old_recipe_replays_to_the_same_script_text():
    """Floor off (or absent) writes the per-type lines it always wrote."""
    a = _cells()
    out, _ = run_pipeline(a, ["filtering"], FIXED)
    p = dict(provenance.read_provenance(out)["history"][-1]["params"])
    with_off = replay_script._filtering(p, replay_script._Ctx(out)).code
    p.pop("qc_mito_type_floor")
    absent = replay_script._filtering(p, replay_script._Ctx(out)).code
    assert with_off == absent
    assert any("drop |= type_mito_drop(adata, qc, MITO_TYPES, fixed_drop(" in c for c in absent)
    assert not any("GLOBAL_MITO" in c for c in absent)


@pytest.mark.parametrize("params", [FIXED, MAD], ids=["fixed", "mad"])
def test_the_replay_script_with_the_floor_keeps_the_same_cells(tmp_path, params):
    inp = _cells()
    p = _with(params, qc_mito_type_floor=True)
    out, res = run_pipeline(inp.copy(), ["filtering"], p)
    assert res.errors == {}
    script = replay_script.build_script(out, dataset_id="t", dataset_name="test")
    assert "GLOBAL_MITO = " in script and "& GLOBAL_MITO" in script
    assert replay_script.describe(out)[-1]["status"] == "exact"
    src, path, dst = tmp_path / "input.h5ad", tmp_path / "replay.py", tmp_path / "replayed.h5ad"
    inp.write_h5ad(src)
    path.write_text(script)
    proc = subprocess.run([sys.executable, str(path), str(src), str(dst)], capture_output=True, text=True,
                          timeout=300)
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert list(ad.read_h5ad(dst).obs_names) == list(out.obs_names)
    # the four 4 % L cells are kept: the floor is in the replay, not just in the app
    assert ad.read_h5ad(dst).n_obs == inp.n_obs - 4 - (0 if params is FIXED else 60)
    steps = run_manifest.build_manifest(out, dataset_id="t", dataset_name="test")["steps"]
    assert steps[-1]["params"]["qc_mito_type_floor"] is True


# --- the card (GET /qc/mito-by-type) --------------------------------------------------


def test_the_card_shows_what_the_floor_would_remove():
    """Against the 13 % ceiling (card rule: above it): L's own bound 2.5 % takes 4,
    with the floor (bound max(2.5, 13) = 13) none; H's own 18 % stays 18 %."""
    a = _cells()
    r = mito_by_type.mito_by_type(a, "cell_type", ceiling=13.0)
    rows = {t.group: t for t in r.types}
    assert (rows["L"].removed_bound, rows["L"].floor_bound, rows["L"].removed_floor) == (4, 13.0, 0)
    assert (rows["H"].removed_global, rows["H"].removed_bound, rows["H"].floor_bound,
            rows["H"].removed_floor) == (30, 4, 18.0, 4)
    assert rows["Q"].floor_bound is None and rows["Q"].removed_floor is None
    assert r.removed_floor_total == 4 and r.removed_bound_total == 8


def test_the_card_reads_a_dataset_filtered_with_the_floor():
    a = _cells()
    out, _ = run_pipeline(a, ["filtering"], _with(FIXED, qc_mito_type_floor=True))
    r = mito_by_type.mito_by_type(out)
    assert r.filter_mito_mode == "per_type" and r.filter_type_floor is True
    rows = {t.group: t for t in r.types}
    assert (rows["L"].applied_rule, rows["L"].applied_floor) == ("own", "global")
    assert (rows["H"].applied_rule, rows["H"].applied_floor) == ("own", "type")
    assert "never stricter than the global rule" in r.summary
    assert "1 of 3 types used their own bound" in r.summary  # H; L took the global rule
    assert "Filtered by the global rule" in rows["L"].note
    assert "per-type bounds on 'cell_type', never stricter than the global rule" in mito_by_type.filtering_rule(r)


# --- endpoints --------------------------------------------------------------------------


def _register(dm, a: ad.AnnData, did: str) -> str:
    from pathlib import Path

    d = Path(dm.data_dir) / "uploads" / did
    d.mkdir(parents=True, exist_ok=True)
    (d / "metadata.json").write_text(json.dumps({"id": did, "name": did, "status": "ready"}))
    a.write_h5ad(d / "data.h5ad")
    return did


async def test_the_preview_endpoint_takes_the_floor(client, dataset_manager):
    did = _register(dataset_manager, _cells(), "fl1")
    url = f"{API}/datasets/{did}/qc/filter-preview"
    base = {"mode": "fixed", "min_genes": 0, "max_pct_mt": 13, "mito_mode": "per_type"}
    off = (await client.get(url, params=base)).json()
    on = (await client.get(url, params={**base, "mito_type_floor": "true"})).json()
    assert off["thresholds"]["mito_type_floor"] is False and on["thresholds"]["mito_type_floor"] is True
    assert {r["name"]: r["removed"] for r in off["mito_types"]["types"]} == {"L": 4, "H": 4, "Q": 0}
    assert {r["name"]: r["removed"] for r in on["mito_types"]["types"]} == {"L": 0, "H": 4, "Q": 0}
    assert (off["n_removed"], on["n_removed"]) == (8, 4)
    card = (await client.get(f"{API}/datasets/{did}/qc/mito-by-type", params={"ceiling": 13})).json()
    assert {t["group"]: t["removed_floor"] for t in card["types"]} == {"L": 0, "H": 4, "Q": None}


async def test_the_assessment_reports_the_recorded_floor(client, dataset_manager):
    out, _ = run_pipeline(_cells(), ["filtering"], _with(FIXED, qc_mito_type_floor=True))
    body = (await client.get(f"{API}/datasets/{_register(dataset_manager, out, 'fl2')}/assessment")).json()
    assert (body["qc_mito_mode"], body["qc_mito_type_floor"]) == ("per_type", True)
    out, _ = run_pipeline(_cells(), ["filtering"], FIXED)
    body = (await client.get(f"{API}/datasets/{_register(dataset_manager, out, 'fl3')}/assessment")).json()
    assert (body["qc_mito_mode"], body["qc_mito_type_floor"]) == ("per_type", False)
