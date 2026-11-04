"""DE between groups warns when the two sides' samples differ in sex (core/sex_check.py
``compare_sides``): sex from the metadata column, else a sex check already run, else no
check. Wilk et al.: every COVID-19 patient male, 2 of 6 controls female, and cell-level
DE put XIST at the top with no warning."""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from scview.core import sex_check

API = "/api/v1"


def _adata(sexes: dict[str, str] | None, with_column: bool = True) -> ad.AnnData:
    """Patients P0-P3 (COVID-19) and controls H0-H3 (normal), 20 cells each in types A/B."""
    rng = np.random.default_rng(0)
    rows, obs = [], []
    for donor, disease in [(f"P{i}", "COVID-19") for i in range(4)] + [(f"H{i}", "normal") for i in range(4)]:
        for typ in ("A", "B"):
            for _ in range(20):
                rate = np.full(20, 3.0)
                if disease == "COVID-19":
                    rate[:5] = 9
                rows.append(rng.poisson(rate))
                rec = {"donor_id": donor, "disease": disease, "cell_type": typ}
                if with_column and sexes:
                    rec["sex"] = sexes[donor]
                obs.append(rec)
    counts = np.asarray(rows, dtype=np.float32)
    obs = pd.DataFrame(obs, index=[f"c{i}" for i in range(len(rows))]).astype("category")
    a = ad.AnnData(sp.csr_matrix(np.log1p(counts / counts.sum(1, keepdims=True) * 1e4)), obs=obs)
    a.var_names = [f"G{i}" for i in range(20)]
    a.layers["counts"] = sp.csr_matrix(counts)
    return a


WILK_LIKE = {**{f"P{i}": "male" for i in range(4)}, "H0": "female", "H1": "female", "H2": "male", "H3": "male"}


def _register(dm, adata, did: str = "sexde") -> str:
    d = Path(dm.data_dir) / "uploads" / did
    d.mkdir(parents=True, exist_ok=True)
    (d / "metadata.json").write_text(json.dumps({"id": did, "name": did, "status": "ready"}))
    adata.write_h5ad(d / "data.h5ad")
    return did


BODY = {"split_by": "disease", "group_a": "COVID-19", "group_b": "normal", "replicate_key": "donor_id"}


@pytest.mark.parametrize("method", ["wilcoxon", "pseudobulk"])
async def test_sides_with_different_sex_make_up_are_flagged(client, dataset_manager, method):
    did = _register(dataset_manager, _adata(WILK_LIKE))
    r = await client.post(f"{API}/datasets/{did}/de/compare", json={**BODY, "method": method,
                                                                    "within_column": "cell_type",
                                                                    "within_value": "A"})
    assert r.status_code == 200, r.text
    sex = r.json()["sex"]
    assert sex["kind"] == "imbalanced" and sex["a"] == {"female": 0, "male": 4}
    assert sex["b"] == {"female": 2, "male": 2}
    assert "COVID-19: 0 female, 4 male; normal: 2 female, 2 male" in sex["message"]
    assert "XIST" in sex["message"] and "metadata column 'sex'" in sex["message"]


async def test_each_side_one_sex_is_confounded_and_balanced_is_quiet(client, dataset_manager):
    did = _register(dataset_manager, _adata({**{f"P{i}": "male" for i in range(4)},
                                             **{f"H{i}": "female" for i in range(4)}}), "conf")
    r = await client.post(f"{API}/datasets/{did}/de/compare", json=BODY)
    assert r.json()["sex"]["kind"] == "confounded" and "no test can separate" in r.json()["sex"]["message"]
    balanced = {d: ("female" if d.endswith(("0", "1")) else "male") for d in WILK_LIKE}
    did = _register(dataset_manager, _adata(balanced), "bal")
    r = await client.post(f"{API}/datasets/{did}/de/compare", json=BODY)
    assert r.json()["sex"] is None


async def test_no_sex_column_no_check_unless_a_sex_check_ran(client, dataset_manager):
    did = _register(dataset_manager, _adata(None, with_column=False), "nosex")
    r = await client.post(f"{API}/datasets/{did}/de/compare", json=BODY)
    assert r.status_code == 200 and r.json()["sex"] is None
    # A stored check (as GET /qc/sex leaves one) is used instead.
    sex_check.remember(did, {"column": "donor_id", "groups": [
        {"name": d, "call": s, "metadata": None} for d, s in WILK_LIKE.items()]})
    try:
        r = await client.post(f"{API}/datasets/{did}/de/compare", json=BODY)
        assert r.json()["sex"]["kind"] == "imbalanced"
        assert "the sex check (from expression)" in r.json()["sex"]["message"]
    finally:
        sex_check._calls.clear()


def test_compare_sides_needs_known_sexes_on_both_sides():
    a = _adata(WILK_LIKE)
    obs = a.obs
    covid = (obs["disease"] == "COVID-19").to_numpy()
    assert sex_check.compare_sides(obs, "donor_id", covid, ~covid, "COVID-19", "normal",
                                   {"P0": "male"}, "x") is None
    r = sex_check.compare_sides(obs, "donor_id", covid, ~covid, "COVID-19", "normal",
                                {d: s for d, s in WILK_LIKE.items()}, "x")
    assert r["kind"] == "imbalanced"


ONE_F_A_SIDE = {d: ("female" if d in ("P0", "H0") else "male") for d in WILK_LIKE}


def _yang_like() -> ad.AnnData:
    """One female donor a side, but H0 gives 4x the cells of P0: donors balance, cells do not."""
    a = _adata(ONE_F_A_SIDE)
    extra = a[(a.obs["donor_id"] == "H0").to_numpy()].copy()
    big = ad.concat([a] + [extra] * 3, index_unique="-")
    big.obs = big.obs.astype("category")
    return big


@pytest.mark.parametrize("method", ["wilcoxon", "t-test"])
async def test_cell_level_tests_warn_when_cells_differ_in_sex(client, dataset_manager, method):
    did = _register(dataset_manager, _yang_like(), f"cells{method}")
    r = await client.post(f"{API}/datasets/{did}/de/compare", json={**BODY, "method": method})
    assert r.status_code == 200, r.text
    sex = r.json()["sex"]
    # 40 of 160 COVID-19 cells (25%) vs 160 of 280 normal cells (57%) from female donors.
    assert sex["kind"] == "cells" and sex["a"]["share_female"] == 0.25
    assert sex["b"]["share_female"] == pytest.approx(160 / 280, abs=1e-4)
    assert "25% of COVID-19's cells and 57% of normal's" in sex["message"]
    assert "15 points" in sex["message"] and sex["threshold"] == sex_check.CELL_SHARE_DIFF


async def test_pseudobulk_keeps_the_donor_rule(client, dataset_manager):
    did = _register(dataset_manager, _yang_like(), "cellspb")
    r = await client.post(f"{API}/datasets/{did}/de/compare", json={**BODY, "method": "pseudobulk"})
    assert r.status_code == 200, r.text
    assert r.json()["sex"] is None  # 1 female of 4 donors a side: balanced by donor


def test_cell_shares_under_the_threshold_are_quiet():
    obs = _adata(ONE_F_A_SIDE).obs
    covid = (obs["disease"] == "COVID-19").to_numpy()
    assert sex_check.compare_cell_shares(obs, "donor_id", covid, ~covid, "C", "N", ONE_F_A_SIDE, "x") is None
    # No cell of a known-sex sample on side A: no judgement.
    assert sex_check.compare_cell_shares(obs, "donor_id", covid, ~covid, "C", "N", {"H0": "female"}, "x") is None


async def test_sex_as_a_pseudobulk_covariate_says_it_is_adjusted(client, dataset_manager):
    did = _register(dataset_manager, _adata(WILK_LIKE), "adj")
    body = {**BODY, "method": "pseudobulk", "within_column": "cell_type", "within_value": "A"}
    plain = (await client.post(f"{API}/datasets/{did}/de/compare", json=body)).json()["sex"]
    assert "adjusted" not in plain and "will show up" in plain["message"]
    r = await client.post(f"{API}/datasets/{did}/de/compare", json={**body, "covariates": ["sex"]})
    assert r.status_code == 200, r.text
    sex = r.json()["sex"]
    assert sex["adjusted"] is True and sex["kind"] == "imbalanced"
    assert sex["message"].startswith("Sex differs between the sides (COVID-19: 0 female, 4 male; normal: 2 female, 2 male)")
    assert "'sex' is a covariate in this design" in sex["message"] and "will show up" not in sex["message"]


def test_unequal_sex_mix_warns_when_neither_side_is_one_sex():
    """Random donor splits of one condition put XIST and Y genes among the hits when the
    sides' sex mix differs (research/pseudobulk/calib); 3 F + 1 M vs 1 F + 3 M says so."""
    obs = _adata(None, with_column=False).obs
    covid = (obs["disease"] == "COVID-19").to_numpy()
    mix = {"P0": "female", "P1": "female", "P2": "female", "P3": "male",
           "H0": "female", "H1": "male", "H2": "male", "H3": "male"}
    r = sex_check.compare_sides(obs, "donor_id", covid, ~covid, "COVID-19", "normal", mix, "x")
    assert r["kind"] == "unequal" and "3 female, 1 male" in r["message"]
    # 2 + 2 vs 1 + 3 differ by 25 points: at the threshold
    two = {**mix, "P2": "male"}
    assert sex_check.compare_sides(obs, "donor_id", covid, ~covid, "C", "N", two, "x")["kind"] == "unequal"
    same = {**mix, "P1": "male", "P2": "male"}  # 1 + 3 a side: balanced
    assert sex_check.compare_sides(obs, "donor_id", covid, ~covid, "C", "N", same, "x") is None


def test_sex_linked_genes_are_named_on_every_de_response():
    from scview.api.v1.de import DEGene, DEResponse

    genes = [DEGene(gene=g, logfoldchange=1.0, pval=0.01, pval_adj=0.04)
             for g in ("IFI27", "XIST", "RPS4Y1", "Xist", "ACTB")]
    r = DEResponse(n_selected=4, n_rest=4, label="x", genes=genes)
    assert r.sex_linked == ["XIST", "RPS4Y1", "Xist"]
    assert DEResponse(n_selected=1, n_rest=1, label="x", genes=genes[:1]).sex_linked == []
    assert sex_check.sex_linked(["UTY", "UTY", "G1"]) == ["UTY"]
