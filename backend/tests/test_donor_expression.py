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

from scview.core import donor_expression as dx

API = "/api/v1"


def _adata(n_donors: int = 3, cells: int = 12, dense: bool = False, with_counts: bool = True):
    """Donors x CTRL/STIM x types A/B, ``cells`` each. Gene G0 has donor i's
    counts = i + 1 (+ 2 in STIM), every cell's total 100; G1 fills the rest."""
    rows, obs = [], []
    for i in range(n_donors):
        for cond in ("CTRL", "STIM"):
            for typ in ("A", "B"):
                for _ in range(cells):
                    g0 = i + 1 + (2 if cond == "STIM" else 0)
                    rows.append([g0, 100 - g0])
                    obs.append({"donor_id": f"D{i}", "stim": cond, "cell_type": typ})
    counts = np.array(rows, dtype=np.float32)
    lognorm = np.log1p(counts / counts.sum(axis=1, keepdims=True) * 1e4)
    obs = pd.DataFrame(obs)
    obs.index = [f"c{i}" for i in range(len(obs))]
    obs["stim"] = pd.Categorical(obs["stim"], categories=["STIM", "CTRL"])  # stored order
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
    r = await client.get(url, params={"gene": "G0", "groupby": "cell_type", "sample": "donor_id",
                                      "method": "cpm"})
    assert r.status_code == 200, r.text
    assert r.json()["method"] == "cpm" and not r.json()["sample_key_detected"]
    r = await client.get(url, params={"gene": "NOPE", "groupby": "cell_type"})
    assert r.status_code == 404
    r = await client.get(url, params={"gene": "G0", "groupby": "cell_type", "method": "x"})
    assert r.status_code == 400
