"""One gene's pseudobulk value per donor in each group (core/donor_expression.py),
behind the Unified View violin's "Per donor" mode."""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from scipy import stats

from scview.core import donor_expression as dx

API = "/api/v1"


def _adata(n_donors: int = 3, cells: int = 12, dense: bool = False, with_counts: bool = True,
           conditions: tuple[str, ...] = ("CTRL", "STIM"), disjoint: bool = False):
    """Donors x conditions x types A/B, ``cells`` each. Gene G0 has donor i's
    counts = i + 1 (+ 2 per condition after the first), every cell's total 100;
    G1 fills the rest. ``disjoint``: different donors in each condition."""
    rows, obs = [], []
    for i in range(n_donors):
        for ci, cond in enumerate(conditions):
            for typ in ("A", "B"):
                for _ in range(cells):
                    g0 = i + 1 + 2 * ci
                    rows.append([g0, 100 - g0])
                    donor = f"D{i}{cond}" if disjoint else f"D{i}"
                    obs.append({"donor_id": donor, "stim": cond, "cell_type": typ})
    counts = np.array(rows, dtype=np.float32)
    lognorm = np.log1p(counts / counts.sum(axis=1, keepdims=True) * 1e4)
    obs = pd.DataFrame(obs)
    obs.index = [f"c{i}" for i in range(len(obs))]
    obs["stim"] = pd.Categorical(obs["stim"], categories=list(reversed(conditions)))  # stored order
    X = lognorm if dense else sp.csr_matrix(lognorm)
    a = ad.AnnData(X, obs=obs)
    a.var_names = ["G0", "G1"]
    if with_counts:
        a.layers["counts"] = sp.csr_matrix(counts)
    return a


def test_mean_of_the_log_normalised_values_per_donor():
    a = _adata()
    r = dx.per_donor_expression(a, "G0", "cell_type", condition_key="stim")
    assert r["sample_key"] == "donor_id" and r["sample_key_detected"]
    assert r["source"] == "log-normalised X"
    assert r["groups"] == ["A", "B"]
    assert r["conditions"] == ["STIM", "CTRL"]  # the stored category order
    assert r["n_donors"] == {"STIM": 3, "CTRL": 3}
    assert len(r["points"]) == 2 * 2 * 3
    p = next(p for p in r["points"] if p["group"] == "A" and p["condition"] == "CTRL"
             and p["sample"] == "D1")
    assert p["n_cells"] == 12
    assert p["value"] == pytest.approx(np.log1p(2 / 100 * 1e4), abs=1e-4)
    assert r["warnings"] == [] and r["message"] is None


def test_log2_cpm_of_the_summed_counts():
    a = _adata()
    r = dx.per_donor_expression(a, "G0", "cell_type", condition_key="stim", method="cpm")
    assert "counts" in r["source"]
    p = next(p for p in r["points"] if p["group"] == "B" and p["condition"] == "STIM"
             and p["sample"] == "D0")
    # 12 cells x 3 counts of 12 x 100: 30,000 CPM
    assert p["value"] == pytest.approx(np.log2(3 / 100 * 1e6 + 1), abs=1e-4)


def test_dense_x_and_the_counts_fallback_give_the_same_means():
    sparse = dx.per_donor_expression(_adata(), "G0", "cell_type", condition_key="stim")
    dense = dx.per_donor_expression(_adata(dense=True), "G0", "cell_type", condition_key="stim")
    assert [p["value"] for p in dense["points"]] == [p["value"] for p in sparse["points"]]
    # X scaled (negatives): the shifted log of the counts instead, the same values.
    scaled = _adata()
    scaled.X = sp.csr_matrix(scaled.X.toarray() - 5)
    r = dx.per_donor_expression(scaled, "G0", "cell_type", condition_key="stim")
    assert r["source"] == "shifted log of the counts in layers['counts']"
    assert [p["value"] for p in r["points"]] == pytest.approx(
        [p["value"] for p in sparse["points"]], abs=1e-4)


def test_fewer_than_three_donors_is_cautioned():
    r = dx.per_donor_expression(_adata(n_donors=2), "G0", "cell_type", condition_key="stim")
    assert r["n_donors"] == {"STIM": 2, "CTRL": 2}
    assert len(r["warnings"]) == 1 and "Fewer than 3 donors" in r["warnings"][0]


def test_small_samples_are_left_out():
    r = dx.per_donor_expression(_adata(cells=5), "G0", "cell_type", condition_key="stim")
    assert r["points"] == [] and r["n_dropped"] == 12
    assert r["n_donors"] == {"STIM": 0, "CTRL": 0}


def test_without_a_condition_one_box_per_group():
    r = dx.per_donor_expression(_adata(), "G0", "cell_type")
    assert r["conditions"] == [""] and r["n_donors"] == {"": 3}
    # each donor's CTRL and STIM cells pooled: one point per donor per group
    assert len(r["points"]) == 6


def test_no_donor_column_is_a_message_not_an_error():
    a = _adata()
    a.obs = a.obs.drop(columns="donor_id")
    r = dx.per_donor_expression(a, "G0", "cell_type", condition_key="stim")
    assert r["sample_key"] is None and r["points"] == []
    assert "need a sample or donor column" in r["message"]


def test_bad_requests():
    a = _adata()
    with pytest.raises(dx.DonorExpressionError, match="not found"):
        dx.per_donor_expression(a, "NOPE", "cell_type")
    with pytest.raises(dx.DonorExpressionError, match="not found"):
        dx.per_donor_expression(a, "G0", "nope")
    with pytest.raises(dx.DonorExpressionError, match="Unknown method"):
        dx.per_donor_expression(a, "G0", "cell_type", method="median")  # type: ignore[arg-type]
    no_counts = _adata(with_counts=False)
    no_counts.X = sp.csr_matrix(no_counts.X.toarray() - 5)
    with pytest.raises(dx.DonorExpressionError, match="normalisation"):
        dx.per_donor_expression(no_counts, "G0", "cell_type")


def test_paired_design_tests_each_group_pair_by_pair():
    r = dx.per_donor_expression(_adata(), "G0", "cell_type", condition_key="stim")
    assert r["design"] == {"paired": True, "n_pairs": 3, "conditions": ["STIM", "CTRL"]}
    assert set(r["paired_tests"]) == {"A", "B"}
    t = r["paired_tests"]["A"]
    # CTRL - STIM in the stored order (STIM first), by hand from the same points.
    pts = {(p["condition"], p["sample"]): p["value"] for p in r["points"] if p["group"] == "A"}
    stim = [pts[("STIM", d)] for d in ("D0", "D1", "D2")]
    ctrl = [pts[("CTRL", d)] for d in ("D0", "D1", "D2")]
    assert t["n_pairs"] == 3
    assert t["mean_diff"] == pytest.approx(np.mean(np.array(ctrl) - np.array(stim)))
    assert t["wilcoxon_p"] == pytest.approx(stats.wilcoxon(ctrl, stim, zero_method="wilcox").pvalue)
    assert t["ttest_p"] == pytest.approx(stats.ttest_rel(ctrl, stim).pvalue)
    assert "3 pairs: read these p-values as descriptive" in t["caution"]
    # A dropped sample leaves that donor out of the pairs, not the design.
    a = _adata(n_donors=5)
    gone = ((a.obs["donor_id"] == "D4") & (a.obs["stim"] == "STIM") & (a.obs["cell_type"] == "A"))
    a = a[~gone.to_numpy()].copy()
    r = dx.per_donor_expression(a, "G0", "cell_type", condition_key="stim")
    assert r["design"]["n_pairs"] == 5
    assert r["paired_tests"]["A"]["n_pairs"] == 4 and r["paired_tests"]["B"]["n_pairs"] == 5
    assert r["paired_tests"]["B"]["caution"] is None


def test_disjoint_donors_are_unpaired_and_unchanged():
    r = dx.per_donor_expression(_adata(disjoint=True), "G0", "cell_type", condition_key="stim")
    assert r["design"] == {"paired": False, "n_pairs": 0, "conditions": ["STIM", "CTRL"]}
    assert r["paired_tests"] == {}
    assert r["n_donors"] == {"STIM": 3, "CTRL": 3} and len(r["points"]) == 12
    # Unsplit: nothing to pair.
    r = dx.per_donor_expression(_adata(), "G0", "cell_type")
    assert r["design"] == {"paired": False, "n_pairs": 0, "conditions": []}


def test_three_timepoints_are_paired_without_a_test():
    a = _adata(conditions=("T0", "T1", "T2"))
    r = dx.per_donor_expression(a, "G0", "cell_type", condition_key="stim")
    assert r["conditions"] == ["T2", "T1", "T0"]
    assert r["design"] == {"paired": True, "n_pairs": 3, "conditions": ["T2", "T1", "T0"]}
    assert r["paired_tests"] == {}
    assert len(r["points"]) == 2 * 3 * 3


def _register(dm, adata, did: str = "ds") -> str:
    d = Path(dm.data_dir) / "uploads" / did
    d.mkdir(parents=True, exist_ok=True)
    (d / "metadata.json").write_text(json.dumps({"id": did, "name": did, "status": "ready"}))
    adata.write_h5ad(d / "data.h5ad")
    return did


async def test_endpoint(client, dataset_manager):
    did = _register(dataset_manager, _adata())
    url = f"{API}/datasets/{did}/expression/per-donor"
    r = await client.get(url, params={"gene": "G0", "groupby": "cell_type", "condition": "stim"})
    assert r.status_code == 200, r.text
    assert len(r.json()["points"]) == 12
    assert r.json()["design"]["paired"] and r.json()["paired_tests"]["A"]["n_pairs"] == 3
    r = await client.get(url, params={"gene": "G0", "groupby": "cell_type", "sample": "donor_id",
                                      "method": "cpm"})
    assert r.status_code == 200, r.text
    assert r.json()["method"] == "cpm" and not r.json()["sample_key_detected"]
    r = await client.get(url, params={"gene": "NOPE", "groupby": "cell_type"})
    assert r.status_code == 404
    r = await client.get(url, params={"gene": "G0", "groupby": "cell_type", "method": "x"})
    assert r.status_code == 400
