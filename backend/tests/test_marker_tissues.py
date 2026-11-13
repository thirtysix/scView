"""The marker-score annotator keeps to the dataset's tissue (core/marker_tissues.py).

On kidney cortex the annotator named clusters Astrocytes, Syncytiotrophoblast, Colonocyte and
Beta cells (research/v075-qa/wilson2022): the databases' types of every tissue competed. With the
tissue known, types of other organs are left out; immune, blood, vessel and stromal types stay.
Fixtures: tests/test_marker_db.py."""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scview.core import marker_db as mdb
from scview.core import marker_tissues as mt
from tests.test_marker_db import _cells, installed, raw, small_sources  # noqa: F401 (fixtures)


@pytest.mark.parametrize("text,want", [
    ("cortex of kidney", {"kidney"}), ("Renal glomerulus", {"kidney"}), ("Bone marrow", {"immune"}),
    ("Smooth muscle", {"stroma"}), ("Skeletal muscle tissue", {"muscle"}), ("Gall bladder", {"liver"}),
    ("Bladder organ", {"urinary"}), ("Peripheral blood", {"blood"}), ("Cerebral cortex", {"nervous"}),
    ("Dorsolateral prefrontal cortex", {"nervous"}), ("Cortex", set()), ("Undefined", set()), ("", set()),
    ("Immune system", {"immune"}), ("Placenta", {"placenta"}), ("Umbilical cord blood", {"blood"}),
])
def test_organs_reads_database_and_dataset_tissue_names(text, want):
    assert mt.organs(text) == want


def test_allowed_types():
    kidney = {"kidney"}
    assert mt.allowed({}, kidney)  # no tissue named: kept
    assert mt.allowed({"kidney": 1.0}, kidney)
    assert not mt.allowed({"nervous": 0.89, "embryo": 0.11}, kidney)  # astrocytes
    assert not mt.allowed({"nervous": 0.99, "blood": 0.01}, kidney)  # a stray row in blood is not enough
    assert mt.allowed({"immune": 0.6, "blood": 0.2, "lung": 0.2}, kidney)  # an immune type
    assert mt.allowed({"lung": 0.22, "liver": 0.09, "gut": 0.07, "kidney": 0.04, "skin": 0.04,
                       "vessels": 0.09}, kidney)  # described across tissues (epithelial cells)
    assert not mt.allowed({"gut": 0.5, "immune": 0.3, "nervous": 0.2}, kidney)  # colonocyte
    # a lineage every tissue has (its Cell Ontology term), unless half of it is another organ's
    assert mt.allowed({"vessels": 0.38, "nervous": 0.35, "liver": 0.27}, kidney, everywhere=True)
    assert not mt.allowed({"liver": 1.0}, kidney, everywhere=True)  # liver sinusoidal endothelial cells


def test_everywhere_lineage_by_cell_ontology(installed):
    db = mdb.load(installed)
    assert mt.everywhere_lineage("CL:0000738", db.ontology)  # leukocyte itself
    assert not mt.everywhere_lineage("", db.ontology) and not mt.everywhere_lineage(None, None)


def test_dataset_tissue_from_hint_obs_column_or_title():
    a = _cells()
    assert mt.dataset_tissue(a) == (set(), "no tissue known (give one to restrict the database types)")
    a.uns["about_title"] = np.array(["IFN-beta treated and control PBMCs from 8 donors"])  # as Kang's h5ad holds it
    assert mt.dataset_tissue(a)[0] == {"blood"}
    a.uns["about_title"] = "Single-nucleus kidney atlas"
    assert mt.dataset_tissue(a)[0] == {"kidney"}
    a.obs["tissue"] = pd.Categorical(["cortex of kidney"] * a.n_obs)
    o, src = mt.dataset_tissue(a)
    assert o == {"kidney"} and src == "obs 'tissue' (cortex of kidney)"
    assert mt.dataset_tissue(a, "lung")[0] == {"lung"}
    assert mt.dataset_tissue(a, "tumour") == (set(), "'tumour' names no organ known here")


def test_index_built_from_the_raw_files_and_kept(installed):
    index = mt.load(installed)
    assert set(index["panglao"]["human"]["B cells"]) == {"immune"}  # counted per database row
    assert (installed / mt.INDEX_FILE).is_file()
    mt._cache.clear()
    assert mt.load(installed) == index  # read back from the saved index
    with gzip.open(installed / mt.INDEX_FILE, "rt") as f:
        assert '"stamp"' in f.read()


def test_annotator_leaves_out_types_of_another_tissue(installed, monkeypatch):
    db = mdb.load(installed)
    a = _cells()
    a.obs["leiden"] = a.obs["lab"].cat.rename_categories(["0", "1", "2", "3"])
    sdb = db.species("human", db.view("annotate", "human"))
    before, info = mdb.annotate(a, "leiden", "human", db)
    assert before["1"] == "NK cells" and info["tissue"] == {
        "organs": [], "from": "no tissue known (give one to restrict the database types)"}
    # NK cells described in the brain only (made up): left out of a kidney dataset
    index = {part: {"human": {name: {"nervous": 5}}} for part, name in sdb.types["NK cells"]["members"]}
    monkeypatch.setattr(mt, "load", lambda dest: index)
    after, info = mdb.annotate(a, "leiden", "human", db, tissue="kidney")
    assert after["1"] != "NK cells" and after["0"] == before["0"]
    assert info["tissue"]["organs"] == ["kidney"] and info["tissue"]["n_types_left_out"] == 1
    # the same tissue from the data's own column, and none for a brain dataset
    a.obs["tissue"] = "kidney"
    assert mdb.annotate(a, "leiden", "human", db)[0]["1"] != "NK cells"
    assert mdb.annotate(a, "leiden", "human", db, tissue="brain")[0]["1"] == "NK cells"


def test_the_pipeline_passes_the_tissue_hint(installed, monkeypatch):
    from scview.core.pipeline import PipelineParams, _run_cell_type_annotation

    monkeypatch.setattr(mdb, "marker_db_dir", lambda: installed)
    monkeypatch.setattr(mdb, "auto_download_enabled", lambda: False)
    seen = {}
    real = mdb.annotate

    def spy(adata, groupby, species=None, db=None, tissue=""):
        seen["tissue"] = tissue
        return real(adata, groupby, species, db, tissue)

    monkeypatch.setattr(mdb, "annotate", spy)
    a = _cells()
    a.obs["leiden"] = a.obs["lab"].cat.rename_categories(["0", "1", "2", "3"])
    out = _run_cell_type_annotation(a, PipelineParams(annotation_method="marker_score", annotation_target="ms",
                                                      annotation_tissue="kidney"))
    assert seen["tissue"] == "kidney" and out["tissue"]["organs"] == ["kidney"]


def test_no_raw_files_restrict_nothing(tmp_path):
    assert mt.load(Path(tmp_path)) == {}


def test_check_suggestions_keep_to_the_tissue_and_verdicts_do_not(installed, monkeypatch):
    """The marker database check's top-3 suggestions keep to the dataset's tissue (on kidney an
    intercalated-cell cluster was offered Colonocyte); a label's own verdict is unchanged."""
    db = mdb.load(installed)
    a = _cells(extra={"NK cells": ["NKG7", "GNLY", "KLRD1", "NCAM1", "PRF1", "GZMB"]})
    before = mdb.check(a, "lab", "human", db)
    by = {r["label"]: r for r in before["labels"]}
    assert by["T-cells"]["suggestions"][0]["cell_type"] == "NK cells"
    assert before["tissue"] == {"organs": [], "from": "no tissue known (give one to restrict the database types)"}
    # NK cells described in the colon only (made up, as a colonocyte), B and T cells immune
    sdb = db.species("human", db.view("suggest", "human"))
    index: dict = {}
    for t, organ in (("NK cells", "gut"), ("B cells", "immune"), ("T cells", "immune")):
        for part, name in sdb.types[t]["members"]:
            index.setdefault(part, {}).setdefault("human", {})[name] = {organ: 5}
    monkeypatch.setattr(mt, "load", lambda dest: index)
    after = mdb.check(a, "lab", "human", db, tissue="kidney")
    assert after["tissue"]["organs"] == ["kidney"] and after["tissue"]["n_types_left_out"] == 1
    assert after["tissue"]["from"] == "the tissue given ('kidney')"
    by2 = {r["label"]: r for r in after["labels"]}
    assert all(s["cell_type"] != "NK cells" for r in after["labels"] for s in r["suggestions"])
    assert by2["B"]["suggestions"][0]["cell_type"] == "B cells"  # immune types stay
    assert by2["Zorblax"]["suggestions"][0]["cell_type"] == "T cells"
    keys = ("verdict", "db_type", "p", "overlap")
    for r in before["labels"]:  # verdicts on the label's own name: unchanged, the gut type's too
        assert {k: by2[r["label"]][k] for k in keys} == {k: r[k] for k in keys}
    assert by2["NK cells"]["verdict"] == "supported" and by2["NK cells"]["db_type"] == "NK cells"
    # the data's own tissue column, a hint for the gut, and a hint that names no organ
    a.obs["tissue"] = "kidney"
    own = mdb.check(a, "lab", "human", db)
    assert own["tissue"]["from"] == "obs 'tissue' (kidney)" and own["tissue"]["n_types_left_out"] == 1
    gut = mdb.check(a, "lab", "human", db, tissue="colon")
    assert {r["label"]: r for r in gut["labels"]}["T-cells"]["suggestions"][0]["cell_type"] == "NK cells"
    assert gut["tissue"]["n_types_left_out"] == 0
    a.obs = a.obs.drop(columns="tissue")
    none = mdb.check(a, "lab", "human", db, tissue="tumour")
    assert none["tissue"] == {"organs": [], "from": "'tumour' names no organ known here"}
    assert {r["label"]: r for r in none["labels"]}["T-cells"]["suggestions"][0]["cell_type"] == "NK cells"


def test_check_without_raw_files_says_so(installed, monkeypatch):
    monkeypatch.setattr(mt, "load", lambda dest: {})
    res = mdb.check(_cells(), "lab", "human", mdb.load(installed), tissue="kidney")
    assert res["tissue"]["organs"] == ["kidney"] and "no type left out" in res["tissue"]["note"]
    assert "n_types_left_out" not in res["tissue"]
