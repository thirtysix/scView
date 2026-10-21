"""Per-type % mito bounds in the filtering step (qc_filter ``mito_mode`` "per_type").

Fixture values are chosen so every median, MAD and bound is worked out by hand:
type A (median 3, MAD 1: bound 6), type B (median 12, MAD 2: bound 18), a
20-cell type C at 15 %, a damaged type D (median 70), unlabelled cells at 11 %.
A global 10 % ceiling removes all of B; per type, B loses only its 4 outliers.
"""

from __future__ import annotations

import json
import math

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from scview.core import provenance, qc_filter
from scview.core.anndata_adaptor import per_cell_qc
from scview.core.pipeline import PipelineParams, _run_filtering, run_pipeline
from scview.core.qc_filter import Thresholds

API = "/api/v1"

# type -> (% mito values); A and B: 30 / 40 / 26 at the centre and 4 outliers.
VALUES = {
    "A": [2.0] * 30 + [3.0] * 40 + [4.0] * 26 + [9.0] * 4,  # median 3, MAD 1 -> bound 6
    "B": [10.0] * 30 + [12.0] * 40 + [14.0] * 26 + [25.0] * 4,  # median 12, MAD 2 -> bound 18
    "C": [15.0] * 20,  # too few cells for a bound of its own
    "D": [65.0] * 20 + [75.0] * 20,  # median 70: damaged
}
N_UNLABELLED = 10  # at 11 %: no type, so the global rule


def _cells(seed: int = 0, sample: bool = False) -> ad.AnnData:
    """QC values in obs (as after the QC step); counts and genes alike, so only % mito
    removes anything. ``sample``: two donors, every other cell."""
    pct = [v for vals in VALUES.values() for v in vals] + [11.0] * N_UNLABELLED
    labels = [t for t, vals in VALUES.items() for _ in vals] + [None] * N_UNLABELLED
    n = len(pct)
    obs = pd.DataFrame({"total_counts": 2000.0, "n_genes_by_counts": 800.0, "pct_counts_mt": pct},
                       index=[f"c{i}" for i in range(n)])
    obs["cell_type"] = pd.Categorical(labels, categories=list(VALUES))
    if sample:
        obs["Donor"] = pd.Categorical(np.where(np.arange(n) % 2, "d1", "d2"))
    a = ad.AnnData(sp.csr_matrix((n, 3), dtype=np.float32), obs=obs)
    a.var_names = ["MT-CO1", "G1", "G2"]
    return a


FIXED10 = PipelineParams(qc_threshold_mode="fixed", min_genes=0, max_pct_mt=10.0, min_cells=0)


def _per_type(p: PipelineParams = FIXED10, **kw) -> PipelineParams:
    return PipelineParams(**{**p.__dict__, "qc_mito_mode": "per_type", **kw})


def _by_type(a: ad.AnnData, mask: np.ndarray) -> dict[str, int]:
    labels = a.obs["cell_type"].astype(object).where(a.obs["cell_type"].notna(), "none")
    return {t: int(mask[(labels == t).to_numpy()].sum()) for t in [*VALUES, "none"]}


def test_a_global_ceiling_takes_type_b_and_per_type_bounds_take_its_outliers():
    a = _cells()
    glob, _ = qc_filter.judge(a, per_cell_qc(a), Thresholds.from_params(FIXED10))
    assert _by_type(a, glob["max_pct_mt"]) == {"A": 0, "B": 100, "C": 20, "D": 40, "none": 10}

    removed, _, types = qc_filter.judge_types(a, per_cell_qc(a), Thresholds.from_params(_per_type()))
    assert _by_type(a, removed["max_pct_mt"]) == {"A": 4, "B": 4, "C": 20, "D": 40, "none": 10}
    rows = {r["name"]: r for r in types.to_dict()["types"]}
    assert (rows["A"]["median"], rows["A"]["mad"], rows["A"]["bound"]) == (3.0, 1.0, 6.0)
    assert (rows["B"]["median"], rows["B"]["mad"], rows["B"]["bound"]) == (12.0, 2.0, 18.0)
    assert rows["A"]["rule"] == rows["B"]["rule"] == "own"
    assert (rows["C"]["rule"], rows["C"]["why"], rows["C"]["bound"]) == ("global", "too_few", None)
    assert (rows["D"]["rule"], rows["D"]["why"]) == ("global", "damaged")
    assert "Only 20 cells" in rows["C"]["note"] and "damaged" in rows["D"]["note"]
    # the loss under each rule, per type: what the preview's table shows
    assert {r["name"]: (r["removed_global"], r["removed"]) for r in rows.values()} == {
        "A": (0, 4), "B": (100, 4), "C": (20, 20), "D": (40, 40)}
    d = types.to_dict()
    assert (d["column"], d["reason"], d["k"], d["n_own"], d["n_global"], d["n_unlabelled"]) == (
        "cell_type", "cell type", 3.0, 2, 2, N_UNLABELLED)


def test_the_bound_is_strictly_above_and_the_multiplier_is_the_mito_one():
    a = _cells()
    pct = a.obs["pct_counts_mt"].to_numpy().copy()
    b = np.flatnonzero(a.obs["cell_type"] == "B")
    pct[b[:2]] = [18.0, 18.01]  # at the bound: kept; just above: removed
    a.obs["pct_counts_mt"] = pct
    removed, _, _ = qc_filter.judge_types(a, per_cell_qc(a), Thresholds.from_params(_per_type()))
    assert not removed["max_pct_mt"][b[0]] and removed["max_pct_mt"][b[1]]
    # k = 2 MADs: A's bound 3 + 2 = 5, B's 12 + 4 = 16 (the 25s and the 18.01 go)
    _, _, types = qc_filter.judge_types(a, per_cell_qc(a), Thresholds.from_params(_per_type(qc_mad_k_mt=2.0)))
    assert {r["name"]: r["bound"] for r in types.rows if r["bound"] is not None} == {"A": 5.0, "B": 16.0}


def test_a_type_without_spread_keeps_the_global_rule():
    a = _cells()
    pct = a.obs["pct_counts_mt"].to_numpy().copy()
    pct[(a.obs["cell_type"] == "B").to_numpy()] = 12.0
    a.obs["pct_counts_mt"] = pct
    removed, _, types = qc_filter.judge_types(a, per_cell_qc(a), Thresholds.from_params(_per_type()))
    row = next(r for r in types.rows if r["name"] == "B")
    assert (row["rule"], row["why"], row["mad"]) == ("global", "no_spread", 0.0)
    assert _by_type(a, removed["max_pct_mt"])["B"] == 100  # 12 >= 10: the global ceiling


def test_a_high_bound_is_capped_at_50_and_warned_about():
    a = _cells()
    pct = a.obs["pct_counts_mt"].to_numpy().copy()
    b = (a.obs["cell_type"] == "B").to_numpy()
    vals = np.array(VALUES["B"]) * 3.5  # 35, 42, 49, 87.5: median 42, MAD 7 -> 63
    vals[0] = 50.0  # at the cap: removed whatever its type (median and MAD stay)
    pct[b] = vals
    a.obs["pct_counts_mt"] = pct
    removed, _, types = qc_filter.judge_types(a, per_cell_qc(a), Thresholds.from_params(_per_type()))
    row = next(r for r in types.rows if r["name"] == "B")
    assert row["bound"] == qc_filter.TYPE_BOUND_CAP == 50.0 and row["capped"]
    assert "63.0%: capped at 50%" in row["note"]
    assert _by_type(a, removed["max_pct_mt"])["B"] == 1 + 4  # the 50 and the 87.5s, not the 49s
    assert row["warning"].startswith("B: its own bound, 50.0%")
    assert types.to_dict()["warnings"] == [row["warning"]]
    # a bound between 30 % and the ceiling is not warned about: the ceiling was higher
    _, _, t40 = qc_filter.judge_types(a, per_cell_qc(a),
                                      Thresholds.from_params(_per_type(max_pct_mt=60.0)))
    assert next(r for r in t40.rows if r["name"] == "B")["warning"] is None


def test_mad_mode_per_type_replaces_the_samples_bound_and_the_ceiling():
    """In MAD mode the global rule is each sample's median + 3 MADs capped by the
    ceiling; a type with a bound of its own uses that instead, the others keep it."""
    a = _cells(sample=True)
    mad = PipelineParams(min_cells=0, max_pct_mt=20.0)
    cells = per_cell_qc(a)
    glob, bounds = qc_filter.judge(a, cells, Thresholds.from_params(mad))
    removed, _, types = qc_filter.judge_types(a, cells, Thresholds.from_params(_per_type(mad)))
    pct = cells["pct_mt"]
    # the global rule, worked out independently: per donor median + 3 raw MADs, ceiling 20
    donor = a.obs["Donor"].to_numpy()
    expect = np.zeros(a.n_obs, dtype=bool)
    for d in ("d1", "d2"):
        m = donor == d
        med = np.median(pct[m])
        hi = min(med + 3 * np.median(np.abs(pct[m] - med)), 20.0)
        expect[m] = (pct[m] > hi) | (pct[m] >= 20.0)
    assert (glob["max_pct_mt"] == expect).all()
    labels = a.obs["cell_type"].astype(object).to_numpy()
    own = np.isin(labels, ["A", "B"])
    bound = np.where(labels == "A", 6.0, 18.0)
    assert (removed["max_pct_mt"][own] == (pct[own] > bound[own])).all()
    assert (removed["max_pct_mt"][~own] == expect[~own]).all()
    assert types.removed_global is not None and (types.removed_global == expect).all()


def test_no_mito_genes_leaves_per_type_with_nothing_to_judge():
    a = _cells()
    a.var_names = ["G0", "G1", "G2"]
    a.obs["pct_counts_mt"] = 0.0  # what the QC step writes without MT- genes
    cells = per_cell_qc(a)
    assert cells["has_mt"] is False
    removed, _, types = qc_filter.judge_types(a, cells, Thresholds.from_params(_per_type()))
    assert types is None and "max_pct_mt" not in removed
    out = qc_filter.preview(a, cells, Thresholds.from_params(_per_type()))
    assert out["mito_types"] is None and out["has_mt"] is False
    kept, found = _run_filtering(a.copy(), _per_type())
    assert kept.n_obs == a.n_obs and found["qc_mito_mode"] == "per_type"
    assert "can't be measured" in found["qc_mito_note"]


def test_a_missing_or_absent_type_column_is_refused():
    a = _cells()
    with pytest.raises(ValueError, match="not found"):
        qc_filter.judge_types(a, per_cell_qc(a), Thresholds.from_params(_per_type(qc_mito_type_column="nope")))
    del a.obs["cell_type"]
    with pytest.raises(ValueError, match="need cell types"):
        qc_filter.judge_types(a, per_cell_qc(a), Thresholds.from_params(_per_type()))
    with pytest.raises(ValueError, match="% mito mode"):
        Thresholds(mito_mode="per_cluster").check()
    with pytest.raises(ValueError, match="% mito MAD multiplier"):
        Thresholds(mito_mode="per_type", mad_k_mt=0).check()  # checked in fixed mode too


def test_global_mode_is_unchanged():
    """The default mode judges, records and previews as before: no per-type keys in
    the step's record or provenance, the same masks as the plain rule."""
    a = _cells(sample=True)
    for p in (FIXED10, PipelineParams(min_cells=0)):
        t = Thresholds.from_params(p)
        assert t.mito_mode == "global"
        removed, bounds, types = qc_filter.judge_types(a, per_cell_qc(a), t)
        assert types is None
        plain = qc_filter.removed_by(per_cell_qc(a), t, None, bounds)
        assert list(removed) == list(plain) and all((removed[k] == plain[k]).all() for k in plain)
        kept, found = _run_filtering(a.copy(), p)
        assert not any(k.startswith("qc_mito") for k in found)
        rec = kept.uns[qc_filter.RECORD_KEY]
        assert not any(k.startswith("mito") for k in rec)
        assert [rec["columns"][c]["name"] for c in rec["columns"]] == qc_filter.record_columns(a, "Donor")
    assert PipelineParams().qc_mito_mode == "global" and PipelineParams().qc_mito_type_column == ""


@pytest.mark.parametrize("mode", ["fixed", "mad"])
@pytest.mark.parametrize("mito", ["global", "per_type"])
def test_the_preview_is_what_the_step_removes(mode, mito):
    a = _cells(sample=True)
    p = PipelineParams(qc_threshold_mode=mode, min_genes=0, max_pct_mt=10.0, min_cells=0,
                       qc_mito_mode=mito)
    out = qc_filter.preview(a, per_cell_qc(a), Thresholds.from_params(p))
    kept, found = _run_filtering(a.copy(), p)
    gone = ~np.isin(a.obs_names, kept.obs_names)
    removed, _, _ = qc_filter.judge_types(a, per_cell_qc(a), Thresholds.from_params(p))
    assert (gone == ~qc_filter.keep_mask(removed, a.n_obs)).all()
    assert out["n_removed"] == int(gone.sum()) and out["n_kept"] == kept.n_obs
    assert out["thresholds"]["mito_mode"] == mito
    if mito == "global":  # the global preview carries no per-type table (the card has one)
        assert out["mito_types"] is None and "qc_mito_types" not in found
        return
    table = out["mito_types"]
    assert table["applied"] == "per_type"
    per = {r["name"]: r for r in table["types"]}
    lost = _by_type(a, gone)
    for name in VALUES:  # counts and genes remove nothing here: the loss is the mito rule's
        assert per[name]["removed"] == lost[name]
    assert found["qc_mito_types"]["types"] == table["types"]
    assert per["B"]["removed"] == 4 and per["B"]["removed_global"] > 50
    json.dumps(out, allow_nan=False)


def test_the_record_and_provenance_keep_the_rule_and_round_trip(tmp_path):
    a = _cells()
    out, res = run_pipeline(a, ["filtering"], _per_type())
    assert res.errors == {}
    p = provenance.read_provenance(out)["history"][-1]["params"]
    assert (p["qc_mito_mode"], p["qc_mito_type_column"], p["qc_mad_k_mt"]) == ("per_type", "cell_type", 3.0)
    assert {r["name"]: r["bound"] for r in p["qc_mito_types"]["types"]} == {
        "A": 6.0, "B": 18.0, "C": None, "D": None}
    path = tmp_path / "f.h5ad"
    out.write_h5ad(path)
    back = ad.read_h5ad(path)
    rec = back.uns[qc_filter.RECORD_KEY]
    assert rec["mito_mode"] == "per_type" and rec["mito_type_column"] == "cell_type"
    assert qc_filter.recorded_mito_rule(rec) == {
        "mito_mode": "per_type", "mito_type_column": "cell_type",
        "mito_type_bounds": {"A": 6.0, "B": 18.0, "C": None, "D": None}}
    audit = qc_filter.recorded_audit(back)
    assert audit["mito_mode"] == "per_type" and audit["mito_type_bounds"]["B"] == 18.0
    rows = {r["name"]: r for r in audit["by_type"]}
    assert rows["B"]["n_removed"] == 4 and rows["B"]["n_cells"] == 100
    assert qc_filter.recorded_filtering(back)["qc_mito_mode"] == "per_type"


def test_the_type_column_is_recorded_even_when_scview_made_it():
    a = _cells()
    a.obs["scview_leiden_r0.5"] = a.obs["cell_type"]
    kept, found = _run_filtering(a.copy(), _per_type(qc_mito_type_column="scview_leiden_r0.5"))
    assert found["qc_mito_type_column"] == "scview_leiden_r0.5"
    labels, pct, _ = qc_filter.recorded_removed_mito(kept, "scview_leiden_r0.5")
    assert labels.count("B") == 4 and len(pct) == a.n_obs - kept.n_obs


def test_filtering_again_per_type_reuses_the_bounds():
    """The cells left kept within their type's bound: worked out again from them,
    whose spread is narrower, the bounds would trim them on every re-run."""
    a = _cells()
    once, _ = run_pipeline(a, ["filtering"], _per_type())
    cells = per_cell_qc(once)
    again, _, types = qc_filter.judge_types(once, cells, Thresholds.from_params(_per_type()))
    assert types.reused_from is not None
    assert not again["max_pct_mt"].any()
    assert {r["name"]: r["bound"] for r in types.rows if r["bound"] is not None} == {"A": 6.0, "B": 18.0}
    # another multiplier works them out afresh from the cells left
    _, _, fresh = qc_filter.judge_types(once, cells, Thresholds.from_params(_per_type(qc_mad_k_mt=2.0)))
    assert fresh.reused_from is None
    # the column the last per-type filtering used is the default
    once.obs["other_cell_type"] = once.obs["cell_type"]
    _, _, auto = qc_filter.judge_types(once, cells, Thresholds.from_params(_per_type()))
    assert (auto.column, auto.reason) == ("cell_type", "filtering")


def test_the_preview_with_per_type_bounds_is_fast_at_200k_cells():
    import time

    rng = np.random.default_rng(0)
    n = 200_000
    obs = pd.DataFrame({"total_counts": rng.lognormal(8, 0.4, n), "n_genes_by_counts": rng.lognormal(7, 0.4, n),
                        "pct_counts_mt": rng.gamma(3, 2, n)}, index=[f"c{i}" for i in range(n)])
    obs["cell_type"] = pd.Categorical(rng.integers(0, 30, n).astype(str))
    obs["Donor"] = pd.Categorical(rng.integers(0, 8, n).astype(str))
    a = ad.AnnData(sp.csr_matrix((n, 3), dtype=np.float32), obs=obs)
    a.var_names = ["MT-CO1", "G1", "G2"]
    cells = per_cell_qc(a)
    t = Thresholds.from_params(_per_type(PipelineParams()))
    qc_filter.preview(a, cells, t)  # warm-up
    t0 = time.perf_counter()
    out = qc_filter.preview(a, cells, t)
    assert time.perf_counter() - t0 < 2.0
    assert out["mito_types"]["n_own"] == 30 and math.isfinite(out["fraction_removed"])


def _register(dm, a: ad.AnnData, did: str = "mt1") -> str:
    from pathlib import Path

    d = Path(dm.data_dir) / "uploads" / did
    d.mkdir(parents=True, exist_ok=True)
    (d / "metadata.json").write_text(json.dumps({"id": did, "name": did, "status": "ready"}))
    a.write_h5ad(d / "data.h5ad")
    return did


async def test_the_preview_endpoint_takes_the_mito_mode_and_column(client, dataset_manager):
    a = _cells()
    a.obs["other"] = pd.Categorical(np.where(np.arange(a.n_obs) % 2, "x", "y"))
    did = _register(dataset_manager, a)
    url = f"{API}/datasets/{did}/qc/filter-preview"
    base = {"mode": "fixed", "min_genes": 0, "max_pct_mt": 10}
    body = (await client.get(url, params={**base, "mito_mode": "per_type"})).json()
    assert body["thresholds"]["mito_mode"] == "per_type" and body["mito_types"]["column"] == "cell_type"
    assert {r["name"]: r["removed"] for r in body["mito_types"]["types"]}["B"] == 4
    assert body["n_removed"] == 4 + 4 + 20 + 40 + N_UNLABELLED
    other = (await client.get(url, params={**base, "mito_mode": "per_type", "mito_type_column": "other"})).json()
    assert other["mito_types"]["column"] == "other" and other["mito_types"]["reason"] == "requested"
    glob = (await client.get(url, params=base)).json()
    assert glob["mito_types"] is None and glob["n_removed"] == 170
    for bad in ({"mito_mode": "per_cluster"}, {"mito_mode": "per_type", "mito_type_column": "nope"}):
        r = await client.get(url, params={**base, **bad})
        assert r.status_code == 400, bad
