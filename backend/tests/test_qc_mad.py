"""Per-sample MAD thresholds (core/qc_filter.py MAD mode) and the per-sample QC
table (core/qc_groups.py sample_summary).

MAD mode removes each sample's own outliers: log1p counts and genes below
median - k MADs (and above median + k MADs when the upper bounds are opted in),
% mito above median + k_mt MADs or at the ceiling. The preview and the filtering
step must judge with the same masks, small samples and unlabelled cells take the
pooled bounds, a MAD of 0 sets no bound, and the run records the mode, k and the
thresholds each sample got.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from scview.core import provenance, qc_filter, qc_groups
from scview.core.anndata_adaptor import per_cell_qc
from scview.core.pipeline import PipelineParams, _run_filtering, run_pipeline
from scview.core.qc_filter import Thresholds

API = "/api/v1"
MAD = Thresholds(mode="mad", max_pct_mt=20.0)  # the default: low tails of counts and genes
BOTH = Thresholds(mode="mad", max_pct_mt=20.0, mad_upper=True)  # + the opt-in upper bounds


def _qc(groups: dict[str, int], depth: dict[str, float] | None = None,
        mito: dict[str, float] | None = None, seed: int = 0, mt: bool = True) -> ad.AnnData:
    """Cells whose QC values sit in obs (as after the QC step), per sample: counts
    log-normal around ``depth``, genes ~40 % of counts, % mito around ``mito``."""
    rng = np.random.default_rng(seed)
    depth, mito = depth or {}, mito or {}
    parts = []
    for g, n in groups.items():
        counts = np.round(np.exp(rng.normal(math.log(depth.get(g, 2000.0)), 0.3, n)))
        genes = np.round(counts * np.exp(rng.normal(math.log(0.4), 0.1, n)))
        pct = np.abs(rng.normal(mito.get(g, 3.0), 1.0, n)) if mt else np.zeros(n)
        parts.append(pd.DataFrame({"Donor": g, "total_counts": counts,
                                   "n_genes_by_counts": genes, "pct_counts_mt": pct}))
    obs = pd.concat(parts, ignore_index=True)
    obs.index = [f"c{i}" for i in range(len(obs))]
    obs["Donor"] = pd.Categorical(obs["Donor"], categories=list(groups))
    a = ad.AnnData(sp.csr_matrix((len(obs), 3), dtype=np.float32), obs=obs)
    a.var_names = ["MT-CO1" if mt else "G0", "G1", "G2"]
    return a


def _log_bounds(v: np.ndarray, k: float) -> tuple[float, float]:
    med = np.median(np.log1p(v))
    mad = np.median(np.abs(np.log1p(v) - med))  # scipy's median_abs_deviation, scale 1
    return med - k * mad, med + k * mad


# --- the bounds ------------------------------------------------------------------


def test_bounds_are_median_plus_minus_k_mads_of_log1p_per_sample():
    a = _qc({"d1": 400, "d2": 400}, depth={"d1": 1000, "d2": 8000})
    cells = per_cell_qc(a)
    samples = qc_filter.samples_of(a)
    assert (samples.column, samples.reason) == ("Donor", "donor")
    b = qc_filter.mad_bounds(cells, BOTH, samples)
    d2 = a.obs["Donor"].to_numpy() == "d2"
    lo, hi = _log_bounds(cells["total_counts"][d2], 5)
    g = {row["name"]: row for row in b.groups}["d2"]
    assert g["min_counts"] == pytest.approx(math.expm1(lo), abs=0.05)
    assert g["max_counts"] == pytest.approx(math.expm1(hi), abs=0.05)
    assert not g["pooled"] and g["skipped"] == []
    removed = qc_filter.removed_by(cells, BOTH, None, b)
    logc = np.log1p(cells["total_counts"])
    assert removed["min_counts"][d2].tolist() == (logc[d2] < lo).tolist()
    assert removed["max_counts"][d2].tolist() == (logc[d2] > hi).tolist()
    assert set(removed) == {"min_counts", "max_counts", "min_genes", "max_genes", "max_pct_mt"}


def test_by_default_only_the_low_tail_of_counts_and_genes_is_cut():
    a = _qc({"d1": 400, "d2": 400}, seed=2)
    big = a.obs_names[:30]
    a.obs.loc[big, "total_counts"] *= 20  # far above any sample's median + 5 MADs
    a.obs.loc[big, "n_genes_by_counts"] *= 8
    a.obs.loc["c40", ["total_counts", "n_genes_by_counts"]] = [60, 25]  # far below
    cells = per_cell_qc(a)
    removed, b = qc_filter.judge(a, cells, MAD)
    assert set(removed) == {"min_counts", "min_genes", "max_pct_mt"}  # no max_* criteria
    assert all(g["max_counts"] is None and g["max_genes"] is None for g in b.groups)
    assert np.isinf(b.hi["total_counts"]).all() and np.isinf(b.hi["n_genes"]).all()
    drop = ~qc_filter.keep_mask(removed, a.n_obs)
    assert drop[40] and not drop[np.isin(a.obs_names, big)].any()
    # Opted in, the same big cells go.
    both, _ = qc_filter.judge(a, cells, BOTH)
    assert both["max_counts"][np.isin(a.obs_names, big)].all()


def test_a_cell_is_judged_against_its_own_sample():
    """1,500 counts is ordinary in a shallow sample and far too few in a deep one."""
    a = _qc({"shallow": 300, "deep": 300}, depth={"shallow": 1500, "deep": 20000})
    a.obs.loc["c0", ["total_counts", "n_genes_by_counts"]] = [1500, 600]  # shallow
    a.obs.loc["c300", ["total_counts", "n_genes_by_counts"]] = [1500, 600]  # deep
    removed, _ = qc_filter.judge(a, per_cell_qc(a), MAD)
    drop = ~qc_filter.keep_mask(removed, a.n_obs)
    assert not drop[0] and drop[300]
    # Pooled over both samples, neither would be an outlier: per sample is the point.
    pooled = qc_filter.removed_by(per_cell_qc(a), MAD)
    assert not pooled["min_counts"][300]


def test_small_samples_and_unlabelled_cells_take_the_pooled_bounds():
    a = _qc({"d1": 500, "d2": 500, "tiny": qc_filter.MIN_SAMPLE_CELLS - 1})
    a.obs.loc[["c0", "c1"], "Donor"] = np.nan
    b = qc_filter.mad_bounds(per_cell_qc(a), MAD, qc_filter.samples_of(a, "Donor"))
    rows = {r["name"]: r for r in b.groups}
    assert rows["tiny"]["pooled"] and not rows["d1"]["pooled"]
    for key in ("min_counts", "max_counts", "min_genes", "max_genes", "max_pct_mt"):
        assert rows["tiny"][key] == b.pooled[key]
    assert b.lo["total_counts"][0] == b.lo["total_counts"][-1]  # c0 unlabelled, last cell tiny
    assert b.to_dict()["n_unlabelled"] == 2


def test_a_mad_of_zero_sets_no_bound_but_the_ceiling_still_holds():
    a = _qc({"d1": 300, "nuclei": 300})
    nuc = a.obs["Donor"].to_numpy() == "nuclei"
    pct = np.zeros(nuc.sum())
    pct[:10] = 12.0  # well below the ceiling: kept, there is no MAD bound to break
    pct[10:12] = 30.0  # above the 20 % ceiling: removed
    a.obs.loc[nuc, "pct_counts_mt"] = pct
    removed, b = qc_filter.judge(a, per_cell_qc(a), MAD)
    rows = {r["name"]: r for r in b.groups}
    assert rows["nuclei"]["skipped"] == ["pct_mt"] and rows["nuclei"]["max_pct_mt"] == 20.0
    assert removed["max_pct_mt"][nuc].sum() == 2
    off = Thresholds(mode="mad", max_pct_mt=None)  # no ceiling: nothing judges mito there
    assert qc_filter.judge(a, per_cell_qc(a), off)[0]["max_pct_mt"][nuc].sum() == 0
    assert qc_filter.mad_bounds(per_cell_qc(a), off, qc_filter.samples_of(a)).groups[1][
        "max_pct_mt"] is None


def test_the_ceiling_caps_a_sample_whose_cells_are_mostly_damaged():
    a = _qc({"ok": 400, "bad": 400}, mito={"ok": 3.0, "bad": 25.0})
    removed, b = qc_filter.judge(a, per_cell_qc(a), MAD)
    rows = {r["name"]: r for r in b.groups}
    assert rows["ok"]["max_pct_mt"] < 10  # median + 3 MADs
    assert rows["bad"]["max_pct_mt"] == 20.0  # median + 3 MADs is ~27: the ceiling binds
    bad = a.obs["Donor"].to_numpy() == "bad"
    pct = a.obs["pct_counts_mt"].to_numpy()
    assert removed["max_pct_mt"][bad].tolist() == (pct[bad] >= 20.0).tolist()


def test_no_mito_genes_and_no_sample_column():
    a = _qc({"d1": 400}, mt=False)
    a.obs["pct_counts_mt"] = 0.0
    del a.obs["Donor"]
    removed, b = qc_filter.judge(a, per_cell_qc(a), MAD)
    assert "max_pct_mt" not in removed
    assert b.groups == [] and b.samples.column is None
    assert b.pooled["max_pct_mt"] is None and b.pooled["min_counts"] > 0
    d = b.to_dict()
    assert d["column"] is None and d["ceiling_pct_mt"] is None and d["n_unlabelled"] == 0


def test_missing_values_are_not_removed():
    a = _qc({"d1": 300})
    a.obs.loc["c0", "total_counts"] = np.nan
    removed, _ = qc_filter.judge(a, per_cell_qc(a), BOTH, qc_filter.samples_of(a, "Donor"))
    assert not removed["min_counts"][0] and not removed["max_counts"][0]


def test_bad_modes_multipliers_and_too_many_samples_are_refused():
    a = _qc({"d1": 300})
    for t in (Thresholds(mode="strict"), Thresholds(mode="mad", mad_k=0),
              Thresholds(mode="mad", mad_k_mt=float("nan"))):
        with pytest.raises(ValueError):
            qc_filter.judge(a, per_cell_qc(a), t)
    a.obs["cell"] = pd.Categorical([f"x{i}" for i in range(a.n_obs)])
    with pytest.raises(ValueError, match="at most"):
        qc_filter.judge(a, per_cell_qc(a), MAD, qc_filter.samples_of(a, "cell"))
    with pytest.raises(ValueError, match="not found"):
        qc_filter.samples_of(a, "nope")


def test_mode_defaults():
    assert PipelineParams().qc_threshold_mode == "mad"  # new runs
    t = Thresholds.from_params(PipelineParams())
    assert (t.mode, t.mad_k, t.mad_k_mt, t.max_pct_mt, t.mad_upper) == ("mad", 5.0, 3.0, 20.0, False)
    assert Thresholds.from_params(PipelineParams(qc_mad_upper=True)).mad_upper

    class Old:  # anything without the new fields keeps the fixed cut-offs
        min_genes = 200
    assert Thresholds.from_params(Old()).mode == "fixed"


# --- the preview and the step ----------------------------------------------------


def _tails(seed: int = 5) -> tuple[ad.AnnData, np.ndarray, np.ndarray]:
    """Three samples (one small) with 40 doublet-like cells, 40 small ones and
    doublet calls."""
    a = _qc({"d1": 500, "d2": 400, "d3": 50}, depth={"d2": 6000}, mito={"d3": 8.0})
    rng = np.random.default_rng(seed)
    picks = rng.choice(a.n_obs, 80, replace=False)
    high, low = picks[:40], picks[40:]
    col = a.obs.columns.get_loc("total_counts")
    a.obs.iloc[high, col] *= 12
    a.obs.iloc[low, col] /= 12
    a.obs["predicted_doublet"] = np.arange(a.n_obs) % 97 == 0
    return a, high, low


@pytest.mark.parametrize("upper", [False, True])
def test_the_preview_is_what_the_mad_step_removes(upper):
    a, high, low = _tails()
    params = PipelineParams(min_cells=0, drop_doublets=True, qc_mad_k=4.0, qc_mad_upper=upper)
    out = qc_filter.preview(a, per_cell_qc(a), Thresholds.from_params(params))
    kept, found = _run_filtering(a.copy(), params)
    gone = ~np.isin(a.obs_names, kept.obs_names)
    assert kept.n_obs == out["n_kept"] and out["n_removed"] == int(gone.sum()) > 40
    assert {g["name"]: g["n_removed"] for g in out["samples"]["groups"]} == {
        s["name"]: s["n_removed"] for s in found["qc_thresholds"]["samples"]}
    assert out["mad"]["samples"] == found["qc_thresholds"]["samples"]
    assert out["thresholds"]["mode"] == "mad" and out["thresholds"]["mad_k"] == 4.0
    assert out["thresholds"]["mad_upper"] is upper and out["mad"]["upper"] is upper
    removed, _ = qc_filter.judge(a, per_cell_qc(a), Thresholds.from_params(params))
    assert removed["min_counts"][low].sum() >= 35  # the low tail goes either way
    if upper:
        assert set(out["by_criterion"]) >= {"max_counts", "doublets"}
        assert removed["max_counts"][high].sum() >= 35
    else:  # nothing cuts the high tail of counts (doublet calls, mito or genes still may)
        assert "max_counts" not in out["by_criterion"] and "max_genes" not in out["by_criterion"]
        assert "max_counts" not in removed and not removed["min_counts"][high].any()
    json.dumps(out, allow_nan=False)  # no inf / nan reaches the browser


def test_a_named_sample_column_is_used_by_both():
    a = _qc({"d1": 300, "d2": 300})
    a.obs["lane"] = pd.Categorical(np.where(np.arange(a.n_obs) % 2, "L1", "L2"))
    params = PipelineParams(min_cells=0, qc_sample_column="lane")
    out = qc_filter.preview(a, per_cell_qc(a), Thresholds.from_params(params), sample_column="lane")
    kept, found = _run_filtering(a.copy(), params)
    assert out["mad"]["column"] == found["qc_thresholds"]["column"] == "lane"
    assert out["samples"]["column"] == "lane" and kept.n_obs == out["n_kept"]


def test_the_run_records_the_mode_k_and_each_samples_thresholds():
    a = _qc({"d1": 300, "d2": 300})
    out, res = run_pipeline(a, ["filtering"], PipelineParams(min_cells=0))
    assert res.errors == {}
    entry = provenance.read_provenance(out)["history"][-1]
    p = entry["params"]
    assert p["qc_threshold_mode"] == "mad" and (p["qc_mad_k"], p["qc_mad_k_mt"]) == (5.0, 3.0)
    assert p["qc_mad_upper"] is False
    assert "min_genes" not in p  # not used in MAD mode, so not recorded
    applied = p["qc_thresholds"]
    assert applied["column"] == "Donor" and applied["ceiling_pct_mt"] == 20.0
    assert applied["upper"] is False and applied["samples"][0]["max_counts"] is None
    assert [s["name"] for s in applied["samples"]] == ["d1", "d2"]
    assert sum(s["n_removed"] for s in applied["samples"]) == a.n_obs - out.n_obs
    assert {"min_counts", "max_counts", "min_genes", "max_genes", "max_pct_mt"} <= set(
        applied["samples"][0])

    out2, _ = run_pipeline(_qc({"d1": 300}), ["filtering"],
                           PipelineParams(min_cells=0, qc_threshold_mode="fixed"))
    p = provenance.read_provenance(out2)["history"][-1]["params"]
    assert p["qc_threshold_mode"] == "fixed" and p["min_genes"] == 200
    assert "qc_thresholds" not in p and "qc_mad_k" not in p


def test_the_methods_text_gets_one_line_not_every_samples_bounds():
    from types import SimpleNamespace

    from scview.core.assistant import _methods_skeleton

    a = _qc({"d1": 300, "d2": 300, "tiny": 40})
    out, _ = run_pipeline(a, ["filtering"], PipelineParams(min_cells=0))
    applied = provenance.read_provenance(out)["history"][-1]["params"]["qc_thresholds"]
    line = qc_filter.applied_summary(applied)
    assert line.startswith("outliers below median - 5.0 MADs of log1p counts and genes")
    assert "within each Donor (3 samples, 1 too small and judged on all cells)" in line
    assert "counts per cell at least" in line and "% mito below" in line
    text = _methods_skeleton(SimpleNamespace(adata=out))
    assert "cell thresholds: outliers below median" in text and "'skipped'" not in text
    pooled = dict(applied, column=None, samples=[])
    assert "over all cells (no sample column)" in qc_filter.applied_summary(pooled)

    both, _ = run_pipeline(_qc({"d1": 300}), ["filtering"], PipelineParams(min_cells=0, qc_mad_upper=True))
    line = qc_filter.applied_summary(provenance.read_provenance(both)["history"][-1]["params"]["qc_thresholds"])
    assert line.startswith("outliers beyond median +/- 5.0 MADs") and "counts per cell kept from" in line


def test_a_bad_mode_fails_the_step_not_the_run():
    a = _qc({"d1": 300})
    out, res = run_pipeline(a, ["filtering"], PipelineParams(qc_threshold_mode="loose"))
    assert "filtering" in res.errors and "loose" in res.errors["filtering"]
    assert out.n_obs == a.n_obs


def _heavy_tailed(seed: int = 3) -> ad.AnnData:
    """Two samples with a tail of small, damaged cells: MAD worked out again on what
    the first pass kept would cut deeper each time."""
    a = _qc({"d1": 2000, "d2": 2000}, seed=seed)
    rng = np.random.default_rng(seed)
    low = rng.choice(a.n_obs, 400, replace=False)
    for col in ("total_counts", "n_genes_by_counts"):
        a.obs.iloc[low, a.obs.columns.get_loc(col)] *= rng.uniform(0.15, 0.6, low.size)
    return a


def test_filtering_again_with_the_same_settings_reuses_the_thresholds():
    a = _heavy_tailed()
    once, _ = run_pipeline(a, ["filtering"], PipelineParams(min_cells=0))
    first = provenance.read_provenance(once)["history"][-1]
    assert first["params"]["qc_thresholds"]["reused_from"] is None and once.n_obs < a.n_obs
    prev = qc_filter.preview(once, per_cell_qc(once), MAD)
    assert prev["previous"]["mode"] == "mad" and prev["mad"]["reused_from"] == first["timestamp"]
    assert prev["n_removed"] == 0

    # Worked out afresh from the cells left, the bounds would cut deeper.
    fresh = qc_filter.mad_bounds(per_cell_qc(once), MAD, qc_filter.samples_of(once))
    assert qc_filter.keep_mask(qc_filter.removed_by(per_cell_qc(once), MAD, None, fresh),
                               once.n_obs).sum() < once.n_obs

    twice, _ = run_pipeline(once.copy(), ["filtering"], PipelineParams(min_cells=0))
    assert twice.n_obs == once.n_obs
    second = provenance.read_provenance(twice)["history"][-1]["params"]["qc_thresholds"]
    assert second["reused_from"] == first["timestamp"]
    assert [s["max_genes"] for s in second["samples"]] == [
        s["max_genes"] for s in first["params"]["qc_thresholds"]["samples"]]

    # Other settings: worked out afresh from the cells left.
    stricter = Thresholds(mode="mad", max_pct_mt=20.0, mad_k=3.0)
    out = qc_filter.preview(once, per_cell_qc(once), stricter)
    assert out["mad"]["reused_from"] is None and out["n_removed"] > 0
    blank = _qc({"d1": 300})
    assert qc_filter.preview(blank, per_cell_qc(blank), MAD)["previous"] is None


def test_reuse_needs_the_same_upper_setting_and_old_records_had_it_on():
    a = _heavy_tailed()
    once, _ = run_pipeline(a, ["filtering"], PipelineParams(min_cells=0))
    rec = qc_filter.recorded_filtering(once)
    cells, samples = per_cell_qc(once), qc_filter.samples_of(once)
    assert qc_filter.mad_bounds(cells, MAD, samples, rec).reused_from is not None
    assert qc_filter.mad_bounds(cells, BOTH, samples, rec).reused_from is None
    del rec["qc_thresholds"]["upper"]  # a record from before the upper bounds were opt-in
    assert qc_filter.mad_bounds(cells, MAD, samples, rec).reused_from is None
    assert qc_filter.mad_bounds(cells, BOTH, samples, rec).reused_from is not None


def test_a_malformed_record_is_not_reused():
    a = _qc({"d1": 300, "d2": 300})
    once, _ = run_pipeline(a, ["filtering"], PipelineParams(min_cells=0))
    rec = qc_filter.recorded_filtering(once)
    rec["qc_thresholds"]["samples"][0]["max_genes"] = "lots"
    b = qc_filter.mad_bounds(per_cell_qc(once), MAD, qc_filter.samples_of(once), rec)
    assert b.reused_from is None


def test_a_fixed_run_before_is_not_reused():
    a = _qc({"d1": 300, "d2": 300})
    once, _ = run_pipeline(a, ["filtering"], PipelineParams(min_cells=0, qc_threshold_mode="fixed"))
    out = qc_filter.preview(once, per_cell_qc(once), MAD)
    assert out["previous"]["mode"] == "fixed" and out["mad"]["reused_from"] is None


def test_runs_from_before_mad_mode_read_as_fixed():
    a = _qc({"d1": 300})
    provenance.record_step(a, step="filtering", tool="scview.qc_filter",
                           params={"min_genes": 200, "max_pct_mt": 20.0})
    assert qc_filter.recorded_filtering(a)["qc_threshold_mode"] == "fixed"
    assert qc_filter.recorded_filtering(_qc({"d1": 10})) is None


def test_the_mad_preview_is_fast_at_200k_cells():
    import time

    rng = np.random.default_rng(1)
    n = 200_000
    a = ad.AnnData(np.zeros((n, 1), dtype="float32"))
    a.obs["Donor"] = pd.Categorical(rng.choice([f"d{i}" for i in range(16)], n))
    a.obs["leiden"] = pd.Categorical(rng.integers(0, 40, n).astype(str))
    cells = {"n_genes": rng.integers(50, 6000, n).astype(float),
             "total_counts": rng.integers(100, 40000, n).astype(float),
             "pct_mt": rng.uniform(0, 40, n), "has_mt": True}
    qc_filter.preview(a, cells, MAD)  # warm-up
    t0 = time.perf_counter()
    out = qc_filter.preview(a, cells, MAD)
    took = time.perf_counter() - t0
    assert len(out["mad"]["samples"]) == 16 and len(out["samples"]["groups"]) == 16
    assert took < 1.0, f"{took:.3f} s"
    assert len(json.dumps(out)) < 60_000


# --- the per-sample QC table -----------------------------------------------------


def test_sample_medians_and_range_flags():
    a = _qc({"d1": 500, "d2": 500, "d3": 500, "d4": 500, "poor": 500, "few": 30},
            depth={"poor": 400}, mito={"d4": 12.0})
    cells = per_cell_qc(a)
    names, codes = qc_groups.group_codes(a.obs["Donor"])
    s = qc_groups.sample_summary(cells, names, codes, qc_filter.MIN_SAMPLE_CELLS)
    rows = {r["name"]: r for r in s["rows"]}
    d1 = a.obs["Donor"].to_numpy() == "d1"
    assert rows["d1"]["median_genes"] == pytest.approx(np.median(cells["n_genes"][d1]), abs=0.01)
    assert rows["d1"]["n_cells"] == 500 and rows["d1"]["flags"] == []
    assert {"low_counts", "low_genes"} <= set(rows["poor"]["flags"])
    assert rows["d4"]["flags"] == ["high_mito"]
    assert "few_cells" in rows["few"]["flags"] and rows["few"]["reasons"][0].startswith("Few cells")
    assert s["compared"] and set(s["typical"]) == {"median_genes", "median_counts", "median_pct_mt"}


def test_near_identical_samples_do_not_flag_and_two_are_not_compared():
    a = _qc({f"d{i}": 400 for i in range(5)}, depth={"d0": 2100, "d1": 1900})
    names, codes = qc_groups.group_codes(a.obs["Donor"])
    s = qc_groups.sample_summary(per_cell_qc(a), names, codes, 100)
    assert all(r["flags"] == [] for r in s["rows"])  # 5 % depth differences: under the floors
    b = _qc({"d1": 400, "d2": 400}, depth={"d2": 400})
    names, codes = qc_groups.group_codes(b.obs["Donor"])
    s = qc_groups.sample_summary(per_cell_qc(b), names, codes, 100)
    assert not s["compared"] and all(r["flags"] == [] for r in s["rows"])


def test_the_preview_carries_the_table():
    a = _qc({"d1": 400, "d2": 400, "d3": 400, "low": 400}, depth={"low": 300}, mt=False)
    a.obs["pct_counts_mt"] = 0.0
    out = qc_filter.preview(a, per_cell_qc(a), MAD)
    groups = {g["name"]: g for g in out["samples"]["groups"]}
    assert "median_pct_mt" not in groups["d1"]  # no MT- genes: not measurable
    assert {"low_counts", "low_genes"} <= set(groups["low"]["flags"])
    assert groups["d1"]["n_removed"] == out["mad"]["samples"][0]["n_removed"]
    assert out["samples"]["typical"]["median_genes"] > 0 and out["samples"]["flag_help"]


# --- the endpoints ---------------------------------------------------------------


def _register(dm, a: ad.AnnData, did: str) -> str:
    d = Path(dm.data_dir) / "uploads" / did
    d.mkdir(parents=True)
    (d / "metadata.json").write_text(json.dumps({"id": did, "name": did, "status": "ready"}))
    a.write_h5ad(d / "data.h5ad")
    return did


async def test_preview_endpoint_takes_the_mode_k_and_sample_column(client, dataset_manager):
    a = _qc({"d1": 300, "d2": 300})
    a.obs["lane"] = pd.Categorical(np.where(np.arange(a.n_obs) % 2, "L1", "L2"))
    did = _register(dataset_manager, a, "mad1")
    body = (await client.get(f"{API}/datasets/{did}/qc/filter-preview",
                             params={"mad_k": 3, "sample_column": "lane"})).json()
    assert body["thresholds"]["mad_k"] == 3.0 and body["mad"]["column"] == "lane"
    assert body["thresholds"]["mad_upper"] is False and "max_counts" not in body["by_criterion"]
    upper = (await client.get(f"{API}/datasets/{did}/qc/filter-preview",
                              params={"mad_upper": "true"})).json()
    assert upper["mad"]["upper"] is True and "max_counts" in upper["by_criterion"]
    fixed = (await client.get(f"{API}/datasets/{did}/qc/filter-preview",
                              params={"mode": "fixed", "min_genes": 10})).json()
    assert fixed["mad"] is None and set(fixed["by_criterion"]) == {"min_genes", "max_pct_mt"}


async def test_assessment_reports_the_recorded_mode(client, dataset_manager):
    fresh = _register(dataset_manager, _qc({"d1": 300}), "fresh")
    old = _qc({"d1": 300})
    provenance.record_step(old, step="filtering", tool="scview.qc_filter",
                           params={"min_genes": 200})
    old_id = _register(dataset_manager, old, "old")
    assert (await client.get(f"{API}/datasets/{fresh}/assessment")).json()[
        "qc_threshold_mode"] is None
    assert (await client.get(f"{API}/datasets/{old_id}/assessment")).json()[
        "qc_threshold_mode"] == "fixed"
