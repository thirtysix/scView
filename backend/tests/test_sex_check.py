"""Per-sample sex from expression (core/sex_check.py, GET /qc/sex): the calls,
metadata mismatches, mixed samples, sex x condition confounding, and the
datasets without the marker genes or a sample column."""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from scview.core import sex_check as sc

API = "/api/v1"
FILLER = [f"GENE{i}" for i in range(60)]


def _adata(donors: dict[str, str], cells: int = 40, conditions=("CTRL", "STIM"),
           y_genes: bool = True, xist: bool = True, mouse: bool = False,
           seed: int = 0) -> ad.AnnData:
    """``cells`` per donor x condition. A donor's sex: "F" expresses XIST in ~70% of
    cells, "M" the Y genes, "FM" (pooled) half its cells one way and half the other."""
    rng = np.random.default_rng(seed)
    f_names = ["Xist"] if mouse else ["XIST"]
    m_names = ["Ddx3y", "Kdm5d"] if mouse else ["RPS4Y1", "DDX3Y"]
    filler = [g.capitalize() for g in FILLER] if mouse else FILLER
    names = filler + (f_names if xist else []) + (m_names if y_genes else [])
    rows, obs = [], []
    for d, sex in donors.items():
        for cond in conditions:
            for j in range(cells):
                base = rng.poisson(3.0, len(filler)).astype(float)
                fem = sex == "F" or (sex == "FM" and j % 2 == 0)
                f = (rng.random(len(f_names)) < (0.7 if fem else 0.0)) * 2.0
                m = (rng.random(len(m_names)) < (0.0 if fem else 0.6)) * 3.0
                rows.append(np.concatenate([base, f if xist else [], m if y_genes else []]))
                obs.append({"donor": d, "stim": cond})
    a = ad.AnnData(sp.csr_matrix(np.array(rows, dtype=np.float32)), obs=pd.DataFrame(obs))
    a.obs_names = [f"c{i}" for i in range(a.n_obs)]
    a.var_names = names
    return a


def test_female_and_male_donors_are_called_with_evidence():
    a = _adata({"D1": "F", "D2": "M", "D3": "F", "D4": "M"})
    r = sc.sex_check(a)
    assert r["column"] == "donor" and r["species"] == "human"
    calls = {g["name"]: (g["call"], g["confidence"]) for g in r["groups"]}
    assert calls == {"D1": ("female", "high"), "D2": ("male", "high"),
                     "D3": ("female", "high"), "D4": ("male", "high")}
    d1 = next(g for g in r["groups"] if g["name"] == "D1")
    assert d1["frac_female"] > 0.5 and d1["frac_male"] == 0 and d1["cpm_female"] > 0
    assert "XIST in" in d1["evidence"] and "80 cells" in d1["evidence"]
    assert r["summary"]["female"] == 2 and r["summary"]["male"] == 2
    # Donors cross the conditions (each has CTRL and STIM): no confounding.
    assert r["confounding"] == [] and r["warnings"] == []


def test_a_pooled_sample_is_mixed_and_warned():
    a = _adata({"D1": "F", "D2": "FM", "D3": "M"})
    r = sc.sex_check(a)
    d2 = next(g for g in r["groups"] if g["name"] == "D2")
    assert d2["call"] == "mixed"
    assert any("'D2' has both female and male markers" in w for w in r["warnings"])


def test_metadata_disagreement_flags_a_possible_swap():
    a = _adata({"D1": "F", "D2": "M", "D3": "F"})
    a.obs["Sex"] = a.obs["donor"].map({"D1": "female", "D2": "F", "D3": "female"})
    r = sc.sex_check(a)
    assert r["metadata_column"] == "Sex"
    d2 = next(g for g in r["groups"] if g["name"] == "D2")
    assert d2["metadata"] == "female" and d2["mismatch"]
    assert any("possible sample swap" in w and "'D2'" in w for w in r["warnings"])
    assert not any(g["mismatch"] for g in r["groups"] if g["name"] != "D2")


def test_sex_confounded_with_condition():
    # Each donor in one condition: CTRL donors female, STIM donors male.
    parts = [_adata({"D1": "F", "D2": "F"}, conditions=("CTRL",), seed=1),
             _adata({"D3": "M", "D4": "M"}, conditions=("STIM",), seed=2)]
    a = ad.concat(parts, index_unique="-")
    r = sc.sex_check(a, "donor")
    hit = r["confounding"][0]
    assert hit["condition"] == "stim" and hit["kind"] == "confounded"
    assert "CTRL female" in hit["message"] and "STIM male" in hit["message"]
    # STIM all male while CTRL has both: imbalanced, not confounded.
    parts = [_adata({"D1": "F", "D2": "M"}, conditions=("CTRL",), seed=1),
             _adata({"D3": "M", "D4": "M"}, conditions=("STIM",), seed=2)]
    r = sc.sex_check(ad.concat(parts, index_unique="-"), "donor")
    hit = r["confounding"][0]
    assert hit["kind"] == "imbalanced" and "'STIM' only male" in hit["message"]
    # All one sex: nothing to confound.
    parts = [_adata({"D1": "F"}, conditions=("CTRL",)), _adata({"D2": "F"}, conditions=("STIM",))]
    assert sc.sex_check(ad.concat(parts, index_unique="-"), "donor")["confounding"] == []


def test_without_y_genes_xist_alone_decides_at_low_confidence():
    a = _adata({"D1": "F", "D2": "M", "D3": "F"}, y_genes=False)
    r = sc.sex_check(a)
    calls = {g["name"]: (g["call"], g["confidence"]) for g in r["groups"]}
    assert calls == {"D1": ("female", "low"), "D2": ("male", "low"), "D3": ("female", "low")}
    assert r["male_genes"] == [] and "RPS4Y1" in r["male_genes_missing"]
    assert any("No Y-chromosome genes" in n for n in r["notes"])


def test_mouse_genes_and_the_case_of_no_markers_or_samples():
    a = _adata({"D1": "F", "D2": "M", "D3": "M"}, mouse=True)
    r = sc.sex_check(a)
    assert r["species"] == "mouse" and r["female_genes"] == ["Xist"]
    assert {g["name"]: g["call"] for g in r["groups"]} == {"D1": "female", "D2": "male",
                                                           "D3": "male"}
    # paul15-like: no sample column, no marker genes.
    bare = _adata({"D1": "F"}, xist=False, y_genes=False)
    del bare.obs["donor"]
    del bare.obs["stim"]
    r = sc.sex_check(bare)
    assert r["column"] is None and [g["name"] for g in r["groups"]] == ["All cells"]
    assert r["groups"][0]["call"] == "undetermined"
    assert any("No sample" in n for n in r["notes"])
    assert any("None of the sex marker genes" in n for n in r["notes"])
    line, concerns = sc.report_lines(r)
    assert "can't be inferred" in line and concerns == []


def test_uniform_low_xist_is_noted_as_one_sex_too_sparse_to_call():
    # Kang-like: XIST in ~6% of every donor's cells, no Y genes.
    a = _adata({f"D{i}": "M" for i in range(4)}, y_genes=False)
    x = a.var_names.get_loc("XIST")
    X = a.X.tolil()
    for i in range(a.n_obs):
        X[i, x] = 1.0 if i % 16 == 0 else 0.0  # 1 in 16 cells (6%) of every donor
    a.X = X.tocsr()
    r = sc.sex_check(a)
    assert {g["call"] for g in r["groups"]} == {"ambiguous"}
    assert any("similar low level in every sample" in n for n in r["notes"])


@pytest.mark.parametrize(("value", "expected"), [
    ("F", "female"), ("Female", "female"), ("PATO:0000384", "male"), ("m", "male"),
    ("unknown", None), (None, None), (float("nan"), None),
])
def test_normalise_sex(value, expected):
    assert sc.normalise_sex(value) == expected


def test_too_few_cells_are_undetermined():
    assert sc.call_sex(5, 0.9, 0.0) == ("undetermined", "none")
    assert sc.call_sex(50, 0.05, 0.05)[0] == "ambiguous"


def test_log_values_without_counts_still_give_detection():
    a = _adata({"D1": "F", "D2": "M", "D3": "F"})
    a.X = sp.csr_matrix(np.log1p(a.X.toarray() / 3.3))  # not counts any more
    r = sc.sex_check(a)
    assert not r["counts"] and r["groups"][0]["cpm_female"] is None
    assert [g["call"] for g in r["groups"]] == ["female", "male", "female"]


def test_report_lines_carry_the_calls_and_warnings():
    a = _adata({"D1": "F", "D2": "FM", "D3": "M"})
    line, concerns = sc.report_lines(sc.sex_check(a))
    assert "1 female, 1 male, 1 mixed" in line and "'donor'" in line
    assert concerns and "pools donors" in concerns[0]


def test_qc_report_states_the_calls_and_the_confounding(tmp_path: Path):
    from scview.core import qc_report
    from scview.core.anndata_adaptor import AnnDataAdaptor

    parts = [_adata({"D1": "F", "D2": "F"}, conditions=("CTRL",), seed=1),
             _adata({"D3": "M", "D4": "M"}, conditions=("STIM",), seed=2)]
    p = tmp_path / "d.h5ad"
    ad.concat(parts, index_unique="-").write_h5ad(p)
    f = qc_report.build_facts(AnnDataAdaptor(str(p)))
    line = next(s.detail for s in f.overview if s.ref == "qc:sex")
    assert "2 female, 2 male" in line
    assert any(c.ref == "qc:sex" and "confounded with 'stim'" in c.detail for c in f.concerns)
    assert any(s.ref == "qc:sex" for s in f.next_steps)


def _register(dm, adata, did: str = "ds") -> str:
    d = Path(dm.data_dir) / "uploads" / did
    d.mkdir(parents=True, exist_ok=True)
    (d / "metadata.json").write_text(json.dumps({"id": did, "name": did, "status": "ready"}))
    adata.write_h5ad(d / "data.h5ad")
    return did


async def test_endpoint(client, dataset_manager):
    did = _register(dataset_manager, _adata({"D1": "F", "D2": "M", "D3": "F"}))
    url = f"{API}/datasets/{did}/qc/sex"
    r = await client.get(url)
    assert r.status_code == 200, r.text
    assert r.json()["column"] == "donor" and len(r.json()["groups"]) == 3
    r = await client.get(url, params={"column": "stim"})
    assert r.status_code == 200 and r.json()["reason"] == "requested"
    assert {g["call"] for g in r.json()["groups"]} == {"mixed"}
    assert (await client.get(url, params={"column": "nope"})).status_code == 404
