"""Who the QC thresholds remove (core/qc_filter.py ``audit``, ``recorded_audit``).

The lab case: a fixed 10 % mito cut-off took 65 % of one cell type and 3 % of the
other, at a rate that followed the drug. The audit cross-tabulates the loss per
cell type x condition, tests removed ~ condition per type (chi-square), flags
both, and the filtering step records what it removed so the audit still works
after the cells are gone (uns record, written to and read from h5ad).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from scipy.stats import chi2_contingency

from scview.core import qc_filter
from scview.core.anndata_adaptor import per_cell_qc
from scview.core.pipeline import PipelineParams, _run_filtering
from scview.core.qc_filter import Thresholds

API = "/api/v1"
MITO = Thresholds(max_pct_mt=10)


def _coculture(condition: bool = True, donors: bool = True, n_type: int = 1000) -> tuple[ad.AnnData, dict]:
    """LNCaP and CAF, half drug half control. % mito 15 (removed at 10 %) for 80 %
    of drug LNCaP, 50 % of control LNCaP and 3 % of each CAF arm; 5 elsewhere."""
    half = n_type // 2
    types, conds, pct = [], [], []
    for t, rates in (("LNCaP", (0.8, 0.5)), ("CAF", (0.03, 0.03))):
        for c, rate in zip(("drug", "control"), rates):
            k = int(round(rate * half))
            types += [t] * half
            conds += [c] * half
            pct += [15.0] * k + [5.0] * (half - k)
    n = len(types)
    a = ad.AnnData(np.zeros((n, 1), dtype="float32"))
    a.obs_names = [f"c{i}" for i in range(n)]
    a.obs["cell_type"] = pd.Categorical(types, categories=["LNCaP", "CAF"])
    if condition:
        a.obs["treatment"] = pd.Categorical(conds)
    if donors:
        a.obs["Donor"] = pd.Categorical(np.tile(["d1", "d2", "d3", "d4"], n // 4))
    a.obs["leiden"] = pd.Categorical(np.where(np.array(types) == "LNCaP", "0", "1"))
    cells = {"n_genes": np.full(n, 2000.0), "total_counts": np.full(n, 6000.0),
             "pct_mt": np.asarray(pct), "has_mt": True}
    return a, cells


def _rows(audit: dict) -> dict:
    return {r["name"]: r for r in audit["by_type"]}


# --- the cross-tab and the test ---------------------------------------------------


def test_a_type_removed_65_vs_3_percent_is_flagged_with_its_numbers():
    a, cells = _coculture()
    out = qc_filter.preview(a, cells, MITO)
    au = out["audit"]
    assert (au["label"]["column"], au["label"]["reason"]) == ("cell_type", "labels")
    assert (au["condition"]["column"], au["condition"]["reason"]) == ("treatment", "condition")
    assert au["condition"]["names"] == ["control", "drug"]
    rows = _rows(au)
    assert (rows["LNCaP"]["n_removed"], rows["LNCaP"]["fraction"]) == (650, 0.65)
    assert (rows["CAF"]["n_removed"], rows["CAF"]["fraction"]) == (30, 0.03)
    assert rows["LNCaP"]["by_criterion"] == {"max_pct_mt": 650}
    assert rows["LNCaP"]["conditions"] == {"n_cells": [500, 500], "n_removed": [250, 400],
                                           "by_criterion": {"max_pct_mt": [250, 400]}}
    loss = [f for f in au["flags"] if f["kind"] == "loss"]
    assert [f["group"] for f in loss] == ["LNCaP"]
    assert loss[0]["message"] == "Removes 65% of LNCaP (650 of 1,000 cells), by Max % mito."
    assert au["overall"]["n_removed"] == out["n_removed"] == 680


def test_a_condition_dependent_rate_is_tested_and_flagged():
    a, cells = _coculture()
    au = qc_filter.preview(a, cells, MITO)["audit"]
    test = _rows(au)["LNCaP"]["test"]
    _, p, dof, _ = chi2_contingency(np.array([[250, 400], [250, 100]]))
    assert test["p"] == pytest.approx(p) and test["dof"] == dof == 1
    assert (test["high"], test["low"], test["max_diff_pp"]) == ("drug", "control", 30.0)
    assert test["compared"] == ["control", "drug"] and not test["low_expected"]
    caf = _rows(au)["CAF"]["test"]
    assert caf["max_diff_pp"] == 0.0 and caf["p"] > 0.5
    cond = [f for f in au["flags"] if f["kind"] == "condition"]
    assert [f["group"] for f in cond] == ["LNCaP", None]  # CAF: same rate; None: over all cells
    assert re.fullmatch(
        r"In LNCaP, Max % mito removes 80% of drug cells \(400 of 500\) but 50% of control cells "
        r"\(250 of 500\) \(p = \de-\d+\): the loss depends on 'treatment'\.", cond[0]["message"])
    assert cond[0]["p"] == pytest.approx(p) and cond[0]["condition"] == "treatment"
    assert au["flags"][0]["kind"] == "condition"  # the condition flags lead
    overall = au["overall"]["test"]
    _, p_all, _, _ = chi2_contingency(np.array([[265, 415], [735, 585]]))
    assert overall["p"] == pytest.approx(p_all) and overall["max_diff_pp"] == 15.0
    assert "Over all cells" in " ".join(f["message"] for f in au["flags"])


def test_a_small_or_insignificant_rate_difference_is_not_flagged():
    a, cells = _coculture(n_type=20_000)
    pct = cells["pct_mt"].copy()
    lncap_ctrl = np.flatnonzero((a.obs["cell_type"] == "LNCaP") & (a.obs["treatment"] == "control"))
    pct[lncap_ctrl[: int(0.75 * lncap_ctrl.size)]] = 15.0  # 75% vs 80%: p tiny, 5 points
    au = qc_filter.preview(a, {**cells, "pct_mt": pct}, MITO)["audit"]
    test = _rows(au)["LNCaP"]["test"]
    assert test["p"] < 1e-6 and test["max_diff_pp"] == 5.0
    assert not [f for f in au["flags"] if f["kind"] == "condition" and f["group"] == "LNCaP"]


def test_without_a_condition_or_samples_the_audit_is_per_type_only():
    a, cells = _coculture(condition=False, donors=False)
    au = qc_filter.preview(a, cells, MITO)["audit"]
    assert au["condition"] is None
    assert all("conditions" not in r and "test" not in r for r in au["by_type"])
    assert "test" not in au["overall"]
    assert [f["kind"] for f in au["flags"]] == ["loss"]


def test_without_a_condition_the_samples_split_the_cells():
    a, cells = _coculture(condition=False)
    au = qc_filter.preview(a, cells, MITO)["audit"]
    assert (au["condition"]["column"], au["condition"]["reason"]) == ("Donor", "sample")
    assert len(_rows(au)["LNCaP"]["conditions"]["n_cells"]) == 4


def test_a_single_condition_has_no_test():
    a, cells = _coculture()
    a.obs["treatment"] = pd.Categorical(["control"] * a.n_obs)
    au = qc_filter.preview(a, cells, MITO, condition_column="treatment")["audit"]
    assert au["condition"]["names"] == ["control"]
    assert all(r["test"] is None for r in au["by_type"]) and au["overall"]["test"] is None
    assert not [f for f in au["flags"] if f["kind"] == "condition"]


def test_unlabelled_cells_count_overall_but_not_per_type():
    a, cells = _coculture()
    labels = a.obs["cell_type"].astype(object).to_numpy()
    labels[:100] = np.nan  # 100 drug LNCaP cells, all of them removed
    a.obs["cell_type"] = pd.Categorical(labels, categories=["LNCaP", "CAF"])
    au = qc_filter.preview(a, cells, MITO)["audit"]
    assert au["n_unlabelled"] == 100
    assert sum(r["n_cells"] for r in au["by_type"]) == a.n_obs - 100
    assert _rows(au)["LNCaP"]["n_removed"] == 650 - 100
    assert au["overall"]["n_removed"] == 680


def test_too_many_label_groups_are_summarised_by_a_note():
    a, cells = _coculture()
    a.obs["cell_type"] = pd.Categorical([f"t{i % 120}" for i in range(a.n_obs)])
    assert qc_filter.label_grouping(a) == ("leiden", "clustering")  # too many to be a default
    au = qc_filter.preview(a, cells, MITO, label_column="cell_type")["audit"]
    assert au["by_type"] == [] and au["flags"] == [f for f in au["flags"] if f["group"] is None]
    assert "'cell_type' has 120 groups; the audit is shown for at most 100." in au["note"]
    assert au["overall"]["n_removed"] == 680
    a.obs["treatment"] = pd.Categorical([f"x{i % 30}" for i in range(a.n_obs)])
    au = qc_filter.preview(a, cells, MITO, label_column="leiden", condition_column="treatment")["audit"]
    assert au["condition"] is None and "'treatment' has 30 groups" in au["note"]


def test_a_group_the_warnings_cover_is_not_flagged_twice():
    a, cells = _coculture()
    out = qc_filter.preview(a, cells, MITO, cluster_column="leiden", label_column="leiden")
    assert [(w["kind"], w["group"]) for w in out["warnings"]] == [("cluster", "0")]
    assert not [f for f in out["audit"]["flags"] if f["kind"] == "loss"]
    # A clustering is named with its column: "leiden 0", not "0".
    cond = [f for f in out["audit"]["flags"] if f["kind"] == "condition" and f["group"] == "0"]
    assert cond[0]["message"].startswith("In leiden 0, Max % mito removes 80% of drug cells")


def test_default_columns():
    a, _ = _coculture()
    assert qc_filter.label_grouping(a) == ("cell_type", "labels")
    del a.obs["cell_type"]
    a.obs["scview_celltype_llm"] = pd.Categorical(["x", "y"] * (a.n_obs // 2))  # remade after filtering
    a.obs["paul15_clusters"] = pd.Categorical(["1Ery", "7MEP"] * (a.n_obs // 2))
    assert qc_filter.label_grouping(a) == ("paul15_clusters", "labels")
    del a.obs["paul15_clusters"]
    a.uns["scview_active_clustering"] = "leiden"
    assert qc_filter.label_grouping(a) == ("leiden", "clustering")
    with pytest.raises(ValueError, match="not found"):
        qc_filter.label_grouping(a, "nope")
    assert qc_filter.condition_grouping(a, None, "leiden", "Donor") == ("treatment", "condition")
    assert qc_filter.condition_grouping(a, None, "treatment", "Donor") == ("Donor", "sample")
    del a.obs["treatment"]
    assert qc_filter.condition_grouping(a, None, "Donor", "Donor") == (None, None)
    assert qc_filter.condition_grouping(a, None, "leiden", None) == (None, None)
    with pytest.raises(ValueError, match="not found"):
        qc_filter.condition_grouping(a, "nope")


def test_nothing_removed_has_rates_but_no_p():
    a, cells = _coculture()
    au = qc_filter.preview(a, cells, Thresholds())["audit"]
    test = _rows(au)["LNCaP"]["test"]
    assert test["p"] is None and test["max_diff_pp"] == 0.0 and au["flags"] == []


# --- the record of what the filtering step removed -------------------------------


def _counts_dataset(n: int = 800, seed: int = 0) -> ad.AnnData:
    """Counts: 4 donors x 2 conditions; type 'B' has much mito in the drug arm."""
    rng = np.random.default_rng(seed)
    n_genes = 300
    X = rng.poisson(1.0, (n, n_genes)).astype("float32")
    names = [f"G{i}" for i in range(n_genes)]
    names[:5] = ["MT-CO1", "MT-CO2", "MT-ND1", "MT-ND2", "MT-ATP6"]
    types = np.repeat(["A", "B"], n // 2)
    cond = np.tile(["ctrl", "drug"], n // 2)
    hot = (types == "B") & (cond == "drug")
    X[hot, :5] += rng.poisson(20, (hot.sum(), 5))
    a = ad.AnnData(sp.csr_matrix(X))
    a.var_names = names
    a.obs_names = [f"c{i}" for i in range(n)]
    a.obs["Donor"] = pd.Categorical(np.repeat(["d1", "d2", "d3", "d4"], n // 4))
    a.obs["stim"] = pd.Categorical(cond)
    a.obs["cell_type"] = pd.Categorical(types)
    a.obs["leiden"] = pd.Categorical(rng.integers(0, 3, n).astype(str))
    a.obs["score"] = rng.normal(size=n)  # numeric: not recorded
    return a


def _fixed(**kw) -> PipelineParams:
    return PipelineParams(qc_threshold_mode="fixed", min_cells=0, **kw)


def _strip(au: dict) -> dict:
    """The audit's numbers, without the record's extras."""
    return {"by_type": [{k: v for k, v in r.items() if k != "removed_median"} for r in au["by_type"]],
            "overall": au["overall"], "label": au["label"], "condition": au["condition"],
            "flags": [f["message"] for f in au["flags"]]}


def test_the_record_round_trips_through_h5ad_and_matches_the_preview(tmp_path):
    a = _counts_dataset()
    params = _fixed(min_genes=0, max_pct_mt=20)
    before = qc_filter.preview(a, per_cell_qc(a), Thresholds.from_params(params))
    assert before["recorded"] is None  # never filtered
    kept, _ = _run_filtering(a.copy(), params)
    assert kept.n_obs == before["n_kept"] < a.n_obs
    rec = kept.uns[qc_filter.RECORD_KEY]
    assert rec["n_removed"] == before["n_removed"] and not rec["truncated"]
    names = [e["name"] for e in rec["columns"].values()]
    assert names[:3] == ["Donor", "stim", "cell_type"] and "leiden" not in names and "score" not in names
    path = tmp_path / "f.h5ad"
    kept.write_h5ad(path)
    back = ad.read_h5ad(path)
    got = qc_filter.recorded_audit(back)
    assert got is not None and not got["stale"]
    assert (got["n_before"], got["n_removed"], got["mode"]) == (a.n_obs, before["n_removed"], "fixed")
    assert _strip(got) == _strip(before["audit"])
    b = _rows(got)["B"]
    assert b["removed_median"]["pct_mt"] > 20 and b["fraction"] > 0.4
    assert any(f["kind"] == "condition" and f["group"] == "B" for f in got["flags"])
    # The preview of the filtered dataset carries it, with the columns asked for.
    again = qc_filter.preview(back, per_cell_qc(back), Thresholds.from_params(params),
                              label_column="cell_type", condition_column="Donor")
    assert again["recorded"]["condition"]["column"] == "Donor"
    assert again["recorded"]["columns"] == names


def test_a_run_that_removes_nothing_still_writes_and_reads(tmp_path):
    a = _counts_dataset()
    kept, _ = _run_filtering(a.copy(), _fixed(min_genes=0, max_pct_mt=None))
    kept.write_h5ad(tmp_path / "f.h5ad")
    got = qc_filter.recorded_audit(ad.read_h5ad(tmp_path / "f.h5ad"))
    assert got["n_removed"] == 0 and got["overall"]["n_cells"] == a.n_obs and got["flags"] == []


def test_columns_not_recorded_fall_back_and_old_or_changed_datasets_are_safe():
    a = _counts_dataset()
    kept, _ = _run_filtering(a.copy(), _fixed(min_genes=0, max_pct_mt=20))
    got = qc_filter.recorded_audit(kept, label_column="leiden")
    assert got["label"]["column"] == "cell_type" and "'leiden' was not recorded" in got["note"]
    del kept.obs["stim"]  # dropped since: the kept cells' values are gone
    got = qc_filter.recorded_audit(kept)
    assert got["condition"]["column"] == "Donor" and "stim" not in got["columns"]
    stale = qc_filter.recorded_audit(kept[:100].copy())
    assert stale["stale"] and "no longer adds up" in stale["note"]
    assert qc_filter.recorded_audit(a) is None  # no record (filtered before it existed)
    a.uns[qc_filter.RECORD_KEY] = {"version": 1, "n_before": "x"}  # malformed
    assert qc_filter.recorded_audit(a) is None


def test_the_record_is_capped(monkeypatch):
    monkeypatch.setattr(qc_filter, "RECORD_MAX_CELLS", 10)
    a = _counts_dataset()
    kept, _ = _run_filtering(a.copy(), _fixed(min_genes=0, max_pct_mt=20))
    rec = kept.uns[qc_filter.RECORD_KEY]
    assert rec["truncated"] and len(rec["obs_names"]) == 10 and rec["n_removed"] > 10
    got = qc_filter.recorded_audit(kept)
    assert got["overall"]["n_removed"] == 10 and "Only the first 10" in got["note"]


# --- the endpoint ----------------------------------------------------------------


def _register(dm, a: ad.AnnData, did: str) -> str:
    d = Path(dm.data_dir) / "uploads" / did
    d.mkdir(parents=True)
    (d / "metadata.json").write_text(json.dumps({"id": did, "name": did, "status": "ready"}))
    a.write_h5ad(d / "data.h5ad")
    return did


async def test_endpoint_audits_the_preview_and_the_record(client, dataset_manager):
    a = _counts_dataset()
    q = {"mode": "fixed", "min_genes": 0, "max_pct_mt": 20}
    did = _register(dataset_manager, a, "aud1")
    body = (await client.get(f"{API}/datasets/{did}/qc/filter-preview", params=q)).json()
    assert body["audit"]["label"]["column"] == "cell_type" and body["recorded"] is None
    assert body["audit"]["condition"]["column"] == "stim"
    r = await client.get(f"{API}/datasets/{did}/qc/filter-preview",
                         params={**q, "label_column": "leiden", "condition_column": "Donor"})
    assert r.status_code == 200
    assert (r.json()["audit"]["label"]["column"], r.json()["audit"]["condition"]["column"]) == ("leiden", "Donor")
    for bad in ({"label_column": "nope"}, {"condition_column": "nope"}):
        r = await client.get(f"{API}/datasets/{did}/qc/filter-preview", params={**q, **bad})
        assert r.status_code == 400, bad
    kept, _ = _run_filtering(a.copy(), _fixed(min_genes=0, max_pct_mt=20))
    did2 = _register(dataset_manager, kept, "aud2")
    body = (await client.get(f"{API}/datasets/{did2}/qc/filter-preview", params=q)).json()
    assert body["recorded"]["n_removed"] == a.n_obs - kept.n_obs
    assert body["recorded"]["label"]["column"] == "cell_type"
    json.dumps(body)  # all plain JSON


# --- the co-pilot's fact ----------------------------------------------------------


def test_the_copilot_fact_states_the_flags_or_says_nothing_was_flagged():
    a, cells = _coculture()
    lines, detail = qc_filter.grounding_lines(a, cells, MITO)
    assert lines[1].startswith("- The default thresholds (fixed cut-offs) would remove 680 of 2,000 cells "
                               "(not applied yet), per 'cell_type' group x 'treatment': In LNCaP, Max % mito")
    assert "Removes 65% of LNCaP (650 of 1,000 cells)" in lines[1]
    assert lines[1].endswith(f"[{qc_filter.GROUNDING_TAG}]") and "3 flag(s)" in detail
    lines, _ = qc_filter.grounding_lines(a, cells, Thresholds())
    assert "No 'cell_type' group loses more than 30%, and no group's removal rate differs" in lines[1]
    b, cells_b = _coculture(condition=False, donors=False)
    lines, _ = qc_filter.grounding_lines(b, cells_b, Thresholds())
    assert "(no condition or sample column to test the removal against)" in lines[1]
    for col in ("cell_type", "leiden"):
        del b.obs[col]
    assert qc_filter.grounding_lines(b, cells_b, MITO) == ([], "")


def test_the_copilot_cites_what_the_filtering_step_removed(tmp_path):
    from scview.core.anndata_adaptor import AnnDataAdaptor
    from scview.core.assistant import build_grounding_context
    from scview.core.claims import resolve_tag

    a = _counts_dataset()
    kept, _ = _run_filtering(a.copy(), _fixed(min_genes=0, max_pct_mt=20))
    p = tmp_path / "f.h5ad"
    kept.write_h5ad(p)
    context, sources = build_grounding_context(AnnDataAdaptor(str(p)), {"qc"})
    fact = next(s for s in sources if s.ref == qc_filter.GROUNDING_TAG)
    removed = a.n_obs - kept.n_obs
    assert f"removed {removed:,} of {a.n_obs:,} cells" in fact.detail
    assert re.search(r"- The filtering step on \d{4}-\d\d-\d\d removed \d+ of 800 cells, "
                     r"per 'cell_type' group x 'stim': ", fact.evidence)
    assert "In B, Max % mito removes 100% of drug cells" in fact.evidence
    assert f"[{qc_filter.GROUNDING_TAG}]" in context
    assert resolve_tag("qc:removal", [s.ref for s in sources]) == [qc_filter.GROUNDING_TAG]
