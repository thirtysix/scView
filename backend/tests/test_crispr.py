"""Guide assignment for CRISPR screens (core/crispr.py)."""

from __future__ import annotations

import json

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from scview.core import crispr, provenance


def _screen(n_per: dict[str, int] | None = None, seed: int = 0) -> ad.AnnData:
    """Cells with guide counts: each cell's own guide at 20-60 UMIs, background 0-2;
    the last 10 cells carry two guides of different targets at similar counts."""
    rng = np.random.default_rng(seed)
    n_per = n_per or {"CD86g1": 30, "CD86g2": 20, "IFNGR1_1": 40, "NTg1": 25, "NTg2": 25}
    guides = list(n_per)
    rows, own = [], []
    for gi, (g, n) in enumerate(n_per.items()):
        for _ in range(n):
            r = rng.integers(0, 3, len(guides)).astype(float)
            r[gi] = rng.integers(20, 60)
            rows.append(r)
            own.append(g)
    X = np.array(rows)
    X[-10:, 0] = 40
    X[-10:, 2] = 35
    a = ad.AnnData(X=np.ones((len(rows), 5), dtype=np.float32),
                   obs=pd.DataFrame({"own": own}, index=[f"c{i}" for i in range(len(rows))]),
                   var=pd.DataFrame(index=["CD86", "IFNGR1", "ACTB", "GAPDH", "B2M"]))
    crispr.attach_counts(a, X, guides)
    return a


@pytest.mark.parametrize("name,target", [
    ("CD86g1", "CD86"), ("IFNGR1_2", "IFNGR1"), ("IFNGR1g3", "IFNGR1"), ("NTg1", "NT"),
    ("non-targeting_3", "non-targeting"), ("STAT1-1", "STAT1"), ("JAK2_sg2", "JAK2"),
    ("CCNG1", "CCNG1"), ("IFNGR1", "IFNGR1"), ("safe-targeting_0012", "safe-targeting"),
])
def test_strip_guide(name, target):
    assert crispr.strip_guide(name) == target


def test_strip_guide_keeps_a_gene_name_whole():
    # NKX2-1 is a gene: "-1" is part of its name, not a guide suffix.
    assert crispr.strip_guide("NKX2-1") == "NKX2"
    assert crispr.strip_guide("NKX2-1", {"NKX2-1", "ACTB"}) == "NKX2-1"
    assert crispr.strip_guide("NKX2-1_2", {"NKX2-1"}) == "NKX2-1"


def test_control_labels():
    for label in ("NT", "non-targeting", "Non_Targeting", "control", "safe", "safe-targeting", "NTC",
                  "scramble", "intergenic"):
        assert crispr.is_control(label), label
    for label in ("CD86", "NTRK1", "CTRL1", "eGFP"):
        assert not crispr.is_control(label), label
    assert crispr.suggest_control(["CD86", "NT", "STAT1"]) == "NT"


def test_calls_from_counts_threshold_and_fold_rule():
    counts = np.array([[30, 2, 0], [30, 20, 0], [30, 9, 0], [3, 2, 4], [6, 6, 0]], dtype=float)
    g = ["a", "b", "c"]
    assert crispr.calls_from_counts(counts, g, 5, 2) == [["a"], ["a", "b"], ["a"], [], ["a", "b"]]
    # Without the fold rule every guide at >= 5 UMIs counts (pertpy assign_by_threshold).
    assert crispr.calls_from_counts(counts, g, 5, 0) == [["a"], ["a", "b"], ["a", "b"], [], ["a", "b"]]
    assert crispr.calls_from_counts(counts, g, 1, 0)[3] == ["c", "a", "b"]


def test_assign_calls_targets_multiplets_and_controls():
    mapping = {"CD86g1": "CD86", "CD86g2": "CD86", "IFNGR1_1": "IFNGR1", "NTg1": "NT", "NTg2": "NT"}
    calls = [["CD86g1"], ["CD86g1", "CD86g2"], ["CD86g1", "IFNGR1_1"], ["NTg1", "NTg2"], [],
             ["multiple", "multiple_"]]
    gc, pert, n = crispr.assign_calls(calls, mapping, "NT")
    assert list(pert) == ["CD86", "CD86", "multiple", "NT", "none", "multiple"]
    assert list(gc)[:3] == ["CD86g1", "CD86g1|CD86g2", "CD86g1|IFNGR1_1"] and gc[4] == "none"
    assert list(n) == [1, 2, 2, 2, 0, 2]


def test_assign_from_counts_writes_columns_record_and_provenance():
    a = _screen()
    s = crispr.summary(a)
    assert s["has_counts"] and s["suggested"] == {"source": "counts", "column": None}
    p = crispr.preview(a)
    assert p["control"] == "NT" and p["targets"] == ["CD86", "IFNGR1"]
    assert p["counts"] == {"assigned": 130, "control": 40, "multiple": 10, "none": 0}
    out = crispr.assign(a)
    assert out["call"]["counts"]["multiple"] == 10 and out["call"]["n_targets"] == 2
    obs = a.obs
    assert list(obs["perturbation"].cat.categories) == ["NT", "CD86", "IFNGR1", "multiple"]
    own = obs["own"].map(lambda g: crispr.strip_guide(g)).to_numpy()
    single = obs["perturbation"].astype(str).to_numpy() != "multiple"
    assert (obs["perturbation"].astype(str).to_numpy()[single] == own[single]).all()
    assert (obs["guide_call"].astype(str).to_numpy()[:30] == "CD86g1").all()
    assert obs["n_guides"].dtype == np.int32
    rec = json.loads(a.uns[crispr.UNS_KEY])["call"]
    assert rec["source"] == "counts" and rec["params"] == {"min_umi": 5, "min_ratio": 2.0}
    assert provenance.read_provenance(a)["history"][-1]["step"] == "crispr_assign"
    lines, detail = crispr.grounding_lines(a)
    assert "2 targets" in detail and "[result:mixscape]" in lines[0]


def test_mapping_override_and_explicit_control():
    a = _screen()
    p = crispr.preview(a, mapping={"IFNGR1_1": "IFNG-R1"}, control="NT")
    assert p["targets"] == ["CD86", "IFNG-R1"] and p["mapping"]["IFNGR1_1"] == "IFNG-R1"
    crispr.assign(a, mapping={"IFNGR1_1": "IFNG-R1"})
    assert json.loads(a.uns[crispr.UNS_KEY])["call"]["mapping"] == {"IFNGR1_1": "IFNG-R1"}
    with pytest.raises(crispr.CrisprError):
        crispr.preview(a, control="nope")


def test_assign_from_a_column_and_a_clashing_name_is_kept():
    a = _screen()
    del a.obsm[crispr.GUIDE_KEY]
    a.obs["perturbation"] = pd.Categorical(np.where(np.arange(a.n_obs) < 60, "CD86", "non-targeting"))
    a.obs.loc[a.obs_names[:3], "perturbation"] = np.nan
    cands = crispr.candidate_columns(a)
    assert cands[0]["column"] == "perturbation" and cands[0]["control"] == "non-targeting"
    assert crispr.default_source(a) == ("column", "perturbation")
    assert [c["column"] for c in cands] == ["perturbation", "own"]  # own: guide names, NT among them
    assert crispr.available(a)["columns"] == ["perturbation"]  # the dataset info reads named columns only
    crispr.assign(a)
    # The user's own 'perturbation' stays; scView's goes beside it.
    assert "perturbation_scview" in a.obs.columns
    assert (a.obs["perturbation"].astype(str).to_numpy()[3:60] == "CD86").all()
    out = a.obs["perturbation_scview"].astype(str).to_numpy()
    assert (out[:3] == "none").all() and out[60] == "non-targeting"
    # A second assignment rewrites scView's column, not the user's.
    crispr.assign(a, control="non-targeting")
    assert "perturbation_scview_scview" not in a.obs.columns


def test_cellranger_calls_side_file(tmp_path):
    a = _screen()
    f = tmp_path / "protospacer_calls_per_cell.csv"
    f.write_text("cell_barcode,num_features,feature_call,num_umis\n"
                 "c0,1,CD86g1,24\nc1,2,CD86g1|IFNGR1_1,3|10\nc139,1,NTg1,40\nzzz,1,NTg2,9\n")
    assert crispr.looks_like_calls(f.read_text().splitlines()[0])
    assert crispr.attach_calls(a, f, file_name=f.name) == 3
    assert a.obs[crispr.CR_CALL].tolist()[:3] == ["CD86g1", "CD86g1|IFNGR1_1", ""]
    assert crispr.default_source(a)[0] == "cellranger"
    p = crispr.preview(a)
    assert p["source"] == "cellranger" and p["counts"]["none"] == a.n_obs - 3
    assert p["counts"]["multiple"] == 1


def test_merged_cellranger_calls_match_their_sample(tmp_path):
    a = ad.AnnData(X=np.ones((4, 2), dtype=np.float32),
                   obs=pd.DataFrame({"sample": ["s1", "s1", "s2", "s2"]},
                                    index=["AAA-1:s1", "CCC-1:s1", "AAA-1:s2", "GGG-1:s2"]))
    f = tmp_path / "p.csv"
    f.write_text("cell_barcode,num_features,feature_call,num_umis\nAAA-1,1,NTg1,5\n")
    assert crispr.attach_calls(a, f, file_name="p.csv", sample_column="sample", sample="s2") == 1
    assert a.obs[crispr.CR_CALL].tolist() == ["", "", "NTg1", ""]


def test_errors_and_nothing_found():
    a = ad.AnnData(X=np.ones((5, 2), dtype=np.float32))
    assert crispr.available(a) is None
    with pytest.raises(crispr.CrisprError) as e:
        crispr.preview(a)
    assert e.value.status == 422
    with pytest.raises(crispr.CrisprError):
        crispr.preview(a, "counts")
    with pytest.raises(crispr.CrisprError):
        crispr.preview(a, "column", "missing")
    with pytest.raises(crispr.CrisprError):
        crispr.preview(_screen(), "counts", min_umi=0)
