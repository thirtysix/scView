"""Curated vs author cell-type labels (core/label_curation.py): Kuppe 2022's "Myeloid" was
curated as "immature innate lymphoid cell"; the check flags such mappings, cautiously, and
leaves plausible ones (abbreviations, prefixes, finer names) alone."""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from scview.core import label_curation

API = "/api/v1"

PAIRS = [  # (author label, curated label, cells)
    ("Myeloid", "immature innate lymphoid cell", 900),
    ("Fib", "fibroblast of cardiac tissue", 800),
    ("CM", "cardiac muscle cell", 1200),
    ("pDC", "plasmacytoid dendritic cell", 60),
    ("vSMCs", "smooth muscle cell", 300),
    ("Cluster 3", "endothelial cell", 400),  # names share nothing, but the type is not a rare one
    ("T cells", "CD4-positive, alpha-beta T cell", 500),
    ("Monocytes", "B cell", 200),  # lineages disagree
    ("Neuronal", "neural progenitor cell", 15),  # too few cells to flag
]


def _adata(pairs=PAIRS, author="cell_type_original") -> ad.AnnData:
    a_lab, c_lab = [], []
    for a, c, n in pairs:
        a_lab += [a] * n
        c_lab += [c] * n
    a_lab += ["CM"] * 30  # a few cells of an author label curated otherwise
    c_lab += ["fibroblast of cardiac tissue"] * 30
    obs = pd.DataFrame({author: pd.Categorical(a_lab), "cell_type": pd.Categorical(c_lab)},
                       index=[f"c{i}" for i in range(len(a_lab))])
    return ad.AnnData(np.zeros((len(obs), 2), dtype=np.float32), obs=obs)


def test_flags_the_implausible_mappings_only():
    out = label_curation.check(_adata())
    assert (out["curated_column"], out["author_column"]) == ("cell_type", "cell_type_original")
    rows = {r["author"]: r for r in out["rows"]}
    flagged = {a for a, r in rows.items() if r.get("flag")}
    assert flagged == {"Myeloid", "Monocytes"}
    assert "myeloid" in rows["Myeloid"]["flag"] and "immature innate lymphoid cell" in rows["Myeloid"]["flag"]
    assert rows["CM"]["curated"] == "cardiac muscle cell" and rows["CM"]["share"] == pytest.approx(1200 / 1230, abs=1e-3)
    assert rows["CM"]["others"] == [{"curated": "fibroblast of cardiac tissue", "n_cells": 30}]
    assert out["n_flagged"] == 2 and "Nothing was renamed" in out["note"]
    assert [r["author"] for r in out["rows"]][:2] == ["CM", "Myeloid"]  # largest first


@pytest.mark.parametrize("author, curated, expect", [
    ("Fib", "fibroblast", True), ("pDC", "plasmacytoid dendritic cell", True),
    ("vSMCs", "smooth muscle cell", True), ("CM", "cardiac muscle cell", True),
    ("NK", "natural killer cell", True), ("Myeloid", "immature innate lymphoid cell", False),
    ("Adipocyte", "fat cell", False),
])
def test_names_related(author, curated, expect):
    assert label_curation.related(author, curated) is expect


def test_no_pair_no_check():
    a = _adata()
    assert label_curation.check(ad.AnnData(obs=a.obs[["cell_type"]])) is None
    b = _adata(author="author_cell_type")
    assert label_curation.columns(b) == ("cell_type", "author_cell_type")
    assert label_curation.check(b)["n_flagged"] == 2
    clean = _adata(pairs=[("Fib", "fibroblast", 300), ("CM", "cardiac muscle cell", 300)])
    out = label_curation.check(clean)
    assert out["n_flagged"] == 0 and out["note"] is None


def _register(dm, a: ad.AnnData, did: str) -> str:
    d = Path(dm.data_dir) / "uploads" / did
    d.mkdir(parents=True)
    (d / "metadata.json").write_text(json.dumps({"id": did, "name": did, "status": "ready"}))
    a.write_h5ad(d / "data.h5ad")
    return did


async def test_the_assessment_carries_the_check(client, dataset_manager):
    did = _register(dataset_manager, _adata(), "curated")
    body = (await client.get(f"{API}/datasets/{did}/assessment")).json()
    assert body["label_curation"]["n_flagged"] == 2
    plain = _register(dataset_manager, ad.AnnData(np.zeros((30, 2), dtype=np.float32)), "plain")
    assert (await client.get(f"{API}/datasets/{plain}/assessment")).json()["label_curation"] is None
