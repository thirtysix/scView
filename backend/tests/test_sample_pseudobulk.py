"""Sample-level pseudobulk QC (core/sample_pseudobulk.py, GET /qc/pseudobulk): the
profiles, what drives each PC, outlier profiles, and the datasets it cannot serve."""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from scview.core import sample_pseudobulk as spb

API = "/api/v1"
N_GENES = 200


def _adata(n_donors: int = 6, cells: int = 30, outlier: str | None = "D4",
           tiny: str | None = "D5", seed: int = 0) -> ad.AnnData:
    """Donors x CTRL/STIM x two cell types, ``cells`` per donor x condition x type.
    STIM raises genes 0-19 fourfold; ``outlier``'s CTRL profile has genes 100-139
    raised tenfold; ``tiny``'s STIM profile has only 3 cells per type."""
    rng = np.random.default_rng(seed)
    base = rng.gamma(2.0, 1.0, N_GENES)
    donor_fx = {f"D{i}": rng.normal(0, 0.15, N_GENES) for i in range(n_donors)}
    rows, obs = [], []
    for i in range(n_donors):
        d = f"D{i}"
        for cond in ("CTRL", "STIM"):
            for typ in ("T", "Mono"):
                lam = base * np.exp(donor_fx[d])
                if cond == "STIM":
                    lam[:20] *= 4
                if d == outlier and cond == "CTRL":
                    lam[100:140] *= 10
                if typ == "Mono":
                    lam[40:60] *= 3
                n = 3 if (d == tiny and cond == "STIM") else cells
                for _ in range(n):
                    rows.append(rng.poisson(lam * rng.uniform(0.7, 1.3)))
                    obs.append({"donor": d, "stim": cond, "orig_ident": cond, "cell_type": typ,
                                "age": 30 + 5 * i, "site": "A" if i % 2 else "B"})
    counts = np.array(rows, dtype=np.float32)
    a = ad.AnnData(sp.csr_matrix(np.log1p(counts / counts.sum(1, keepdims=True) * 1e4)),
                   obs=pd.DataFrame(obs))
    a.layers["counts"] = sp.csr_matrix(counts)
    a.obs_names = [f"c{i}" for i in range(a.n_obs)]
    a.var_names = [f"G{i}" for i in range(N_GENES)]
    return a


def _assoc(r: dict, name: str) -> dict:
    return next(a for a in r["association"] if a["covariate"] == name)


def test_profiles_split_by_the_condition_the_donors_cross():
    r = spb.pseudobulk_qc(_adata())
    assert r["sample_column"] == "donor" and r["split_by"] == "stim"
    assert r["message"] is None and r["counts_source"] == "layers['counts']"
    assert len(r["profiles"]) == 11  # 12 donor x condition, minus the tiny one
    assert r["dropped"] == [{"sample": "D5", "split": "STIM", "n_cells": 6}]
    assert any("fewer than 10 cells" in n for n in r["notes"])
    assert r["n_genes"] == N_GENES or r["n_genes"] <= spb.N_TOP
    assert len(r["explained_variance"]) == spb.N_PCS


def test_the_condition_drives_pc1_and_duplicates_are_merged():
    r = spb.pseudobulk_qc(_adata(outlier=None))
    assert r["drivers"]["PC1"]["covariate"] == "stim"
    stim = _assoc(r, "stim")
    assert stim["same_as"] == ["orig_ident"]  # the same split under another name
    assert stim["pcs"][0]["r2"] > 0.9 and stim["pcs"][0]["p"] < 1e-4
    assert stim["pcs"][0]["p_rank"] is not None
    assert not any(a["covariate"] == "orig_ident" for a in r["association"])
    assert set(r["loadings"]["PC1"]["positive"]) <= {f"G{i}" for i in range(20)}
    # Donor-level covariates: a numeric age (Spearman), a categorical site, the donor.
    assert _assoc(r, "age")["kind"] == "numeric" and _assoc(r, "age")["test"] == "Spearman"
    assert _assoc(r, "site")["kind"] == "categorical"
    assert _assoc(r, "donor")["kind"] == "categorical"
    for name in ("n_cells", "log10 total counts", "median genes per cell"):
        assert _assoc(r, name)["kind"] == "numeric"
    assert not any(a["covariate"] == "cell_type" for a in r["association"])  # varies within


def test_an_outlier_profile_is_flagged_alone():
    r = spb.pseudobulk_qc(_adata())
    assert [o["id"] for o in r["outliers"]] == ["D4 | CTRL"]
    o = r["outliers"][0]
    assert o["distance"] > o["cutoff"] and "SDs from the others" in o["reason"]
    assert "condition's profiles" in r["outlier_rule"]
    flagged = {p["id"] for p in r["profiles"] if p["outlier"]}
    assert flagged == {"D4 | CTRL"}
    clean = spb.pseudobulk_qc(_adata(outlier=None))
    assert clean["outliers"] == []


def test_per_sample_profiles_and_one_cell_type():
    r = spb.pseudobulk_qc(_adata(tiny=None), split_by="")
    assert r["split_by"] is None and len(r["profiles"]) == 6
    assert not any(a["covariate"] == "donor" for a in r["association"])  # one profile each
    mono = spb.pseudobulk_qc(_adata(tiny=None), celltype_column="cell_type", celltype="Mono")
    assert mono["celltype"] == "Mono" and mono["n_cells_used"] == 6 * 2 * 30
    assert {lv["name"] for lv in mono["celltype_levels"]} == {"T", "Mono"}
    with pytest.raises(spb.SampleQcError, match="not a value"):
        spb.pseudobulk_qc(_adata(), celltype_column="cell_type", celltype="B")
    with pytest.raises(spb.SampleQcError, match="not found"):
        spb.pseudobulk_qc(_adata(), sample_column="nope")


def test_the_inferred_sex_is_a_covariate():
    calls = {f"D{i}": ("female" if i < 3 else "male") for i in range(6)}
    calls["D5"] = "ambiguous"
    r = spb.pseudobulk_qc(_adata(), sex_calls=calls)
    sex = next(c for c in r["covariates"] if c["name"] == "inferred sex")
    assert sex["derived"] and "unknown" in sex["values"]
    # One sex only: nothing to compare, so no covariate.
    r = spb.pseudobulk_qc(_adata(), sex_calls={f"D{i}": "female" for i in range(6)})
    assert not any(c["name"] == "inferred sex" for c in r["covariates"])


def test_what_it_cannot_serve_is_said_not_raised():
    a = _adata()
    no_samples = a.copy()
    no_samples.obs = no_samples.obs[["cell_type"]]
    r = spb.pseudobulk_qc(no_samples)
    assert r["sample_column"] is None and "No sample" in r["message"]
    no_counts = a.copy()
    del no_counts.layers["counts"]
    r = spb.pseudobulk_qc(no_counts)
    assert "raw counts" in r["message"] and r["profiles"] == []
    few = spb.pseudobulk_qc(_adata(n_donors=2, tiny=None))
    assert "a PCA needs" not in (few["message"] or "")  # 4 profiles: enough for a PCA
    assert any("Outliers need" in n for n in few["notes"])
    small = spb.pseudobulk_qc(_adata(n_donors=2, tiny=None), celltype_column="cell_type",
                              celltype="T", split_by="")
    assert small["message"] and "a PCA needs" in small["message"]


def test_a_categorical_covariate_with_nearly_a_level_per_profile_is_not_tested():
    cov = {"kind": "categorical", "values": ["a", "b", "c", "c"]}
    tests = spb._association(np.array([[1.0], [2.0], [3.0], [3.5]]), cov)
    assert tests[0]["r2"] is None and tests[0]["p"] is None


def _register(dm, adata, did: str = "ds") -> str:
    d = Path(dm.data_dir) / "uploads" / did
    d.mkdir(parents=True, exist_ok=True)
    (d / "metadata.json").write_text(json.dumps({"id": did, "name": did, "status": "ready"}))
    adata.write_h5ad(d / "data.h5ad")
    return did


async def test_endpoint(client, dataset_manager):
    did = _register(dataset_manager, _adata(outlier=None))
    url = f"{API}/datasets/{did}/qc/pseudobulk"
    r = await client.get(url)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["split_by"] == "stim" and body["drivers"]["PC1"]["covariate"] == "stim"
    r = await client.get(url, params={"split_by": "", "celltype_column": "cell_type",
                                      "celltype": "T"})
    assert r.status_code == 200 and r.json()["split_by"] is None
    assert (await client.get(url, params={"celltype_column": "cell_type",
                                          "celltype": "B"})).status_code == 400
    assert (await client.get(url, params={"sample_column": "nope"})).status_code == 400
