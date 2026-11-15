"""Marker database rows' Cell Ontology ids, checked against the type's name (v0.80).

CellMarker 3.0 rows carry ids of other cell types: "Mait-like cell", "Bone-specific cell" and
"Pluripotency cell" CL:0000145 (professional antigen presenting cell), kidney segments the generic
epithelial cell (QA round 6, research/v080-qa/kidney). On the full databases 829 of the 1,814 human
types with a row id change (verdict view): 142 take the term their name matches, 687 lose an id
that names another cell type.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd

from scview.core import cell_ontology as co
from scview.core import marker_db as mdb
from scview.core.marker_names import Ontology


def _onto() -> Ontology:
    return Ontology({
        "CL:0000000": ("cell", [], []),
        "CL:0000145": ("professional antigen presenting cell", [], ["CL:0000000"]),
        "CL:0000940": ("mucosal invariant T cell", ["MAIT cell"], ["CL:0000000"]),
        "CL:0000236": ("B cell", [], ["CL:0000000"]),
        "CL:0005009": ("renal principal cell", [], ["CL:0000000"]),
        "CL:0000066": ("epithelial cell", [], ["CL:0000000"]),
        "CL:0000540": ("neuron", [], ["CL:0000000"]),
        "CL:1000309": ("epicardial adipocyte", ["epicardial fat cell"], ["CL:0000000"]),
    })


def test_a_row_id_is_kept_only_when_its_term_names_the_type():
    types = {
        "B cells": ["CL:0000236"],  # the term has the name
        "Mait-like cell": ["CL:0000145"],  # another type's id; the name matches MAIT cell
        "Bone-specific cell": ["CL:0000145"],  # another type's id; the name matches nothing
        "Double-negative B cell": ["CL:0000236"],  # a broader term the name holds
        "Principal cells": ["CL:0005009"],  # the name with its organ
        "Loop of Henle cells": ["CL:0000066"],  # the generic epithelial cell
        "Epicardial cell": ["CL:1000309"],  # "fat" is a cell word, not an organ
        "Cortical neuron": ["CL:9999999"],  # not a term of this release
        "Plain cell": [],  # no row id, and the name matches no term
        "MAIT cells": [],  # no row id: the name's own term, as the matcher reads it
    }
    t = {n: {"cl": ids[0] if ids else "", "cls": list(ids), "genes": []} for n, ids in types.items()}
    counts = mdb.vet_cl_ids(t, _onto())
    got = {n: (v["cl"], v.get("cl_how")) for n, v in t.items()}
    assert got == {
        "B cells": ("CL:0000236", "row"),
        "Mait-like cell": ("CL:0000940", "name"),
        "Bone-specific cell": ("", "dropped"),
        "Double-negative B cell": ("CL:0000236", "row_broader"),
        "Principal cells": ("CL:0005009", "row_organ"),
        "Loop of Henle cells": ("", "dropped"),
        "Epicardial cell": ("", "dropped"),
        "Cortical neuron": ("", "dropped"),
        "Plain cell": ("", None),
        "MAIT cells": ("CL:0000940", "name_no_row_id"),
    }
    assert t["Mait-like cell"]["cl_row"] == "CL:0000145" and "cl_row" not in t["B cells"]
    assert counts == {"row": 1, "name": 1, "dropped": 4, "row_broader": 1, "row_organ": 1, "name_no_row_id": 1}
    # without the ontology nothing is checked
    raw = {"Bone-specific cell": {"cl": "CL:0000145", "cls": ["CL:0000145"], "genes": []}}
    assert not mdb.vet_cl_ids(raw, None) and raw["Bone-specific cell"]["cl"] == "CL:0000145"


def test_another_sources_id_is_tried_before_the_name():
    t = {"B cells": {"cl": "CL:0000145", "cls": ["CL:0000145", "CL:0000236"], "genes": []}}
    mdb.vet_cl_ids(t, _onto())
    assert t["B cells"]["cl"] == "CL:0000236" and t["B cells"]["cl_how"] == "row"


def test_merge_keeps_every_sources_id_and_the_matcher_reads_the_checked_one():
    parts = [("panglao", {"Mait-like cell": {"cl": "CL:0000145", "genes": ["KLRB1", "SLC4A10"]}}),
             ("cellmarker3", {"Mait like cell": {"cl": "", "genes": ["KLRB1"]}})]
    types = mdb.merge(parts)
    sdb = mdb.SpeciesDB("human", types, _onto())
    [name] = sdb.names
    assert sdb.types[name]["cl"] == "CL:0000940"
    # a label naming the antigen presenting cell no longer reaches the MAIT type through the bad id
    assert sdb.matcher.match("professional antigen presenting cell")[0] == []


def _index() -> co.Index:
    T = co.Term
    return co.Index({
        "CL:0000000": T("CL:0000000", "cell", [], None, []),
        "CL:4023041": T("CL:4023041", "L5 extratelencephalic projecting glutamatergic cortical neuron",
                        [("L5 PT", "EXACT")], None, ["CL:0000000"]),
        "CL:0002306": T("CL:0002306", "epithelial cell of proximal tubule", [], None, ["CL:0000000"]),
        "CL:0000786": T("CL:0000786", "plasma cell", [("PC", "EXACT")], None, ["CL:0000000"]),
        "CL:0005009": T("CL:0005009", "renal principal cell", [("principal cell", "EXACT")], None, ["CL:0000000"]),
    })


def test_cl_suggestions_read_kidney_codes_in_a_kidney_dataset():
    """QA round 6: a bare "PT" was offered the L5 PT neuron in a kidney dataset."""
    idx = _index()
    assert [s["id"] for s in idx.suggest("PT")] == ["CL:4023041"]
    assert [s["id"] for s in idx.suggest("PT", kidney=True)][0] == "CL:0002306"
    a = ad.AnnData(np.zeros((4, 1), dtype=np.float32))
    a.obs["lab"] = pd.Categorical(["PT", "PT", "PC", "PC"])
    plain = {r["label"]: [s["id"] for s in r["suggestions"]] for r in co.suggest(a, "lab", idx)["labels"]}
    assert plain["PT"] == ["CL:4023041"] and plain["PC"][0] == "CL:0000786"
    a.obs["tissue"] = pd.Categorical(["kidney"] * 4)
    kid = {r["label"]: [s["id"] for s in r["suggestions"]] for r in co.suggest(a, "lab", idx)["labels"]}
    assert kid["PT"][0] == "CL:0002306" and kid["PC"][0] == "CL:0005009"
