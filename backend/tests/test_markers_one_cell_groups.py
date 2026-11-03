"""Markers for a column where some groups hold a single cell.

scanpy refuses the whole column ("Could not calculate statistics for groups ... since
they only contain one sample"); seen on author columns in the 1,000-dataset pilot
(8580d236, 43ed6161). Those groups are now left out and named; the others get scanpy's
own result with ``groups=`` the kept ones.
"""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scanpy as sc

from scview.core import pipeline, sparse_markers

API = "/api/v1"


def _adata(n: int = 240, g: int = 60) -> ad.AnnData:
    rng = np.random.default_rng(1)
    X = rng.poisson(0.5, size=(n, g)).astype("float32")
    labels = np.array(["A"] * 100 + ["B"] * 100 + ["C"] * 38 + ["one1", "one2"])
    X[labels == "A", :8] += 4
    X[labels == "B", 8:16] += 4
    a = ad.AnnData(X)
    a.var_names = [f"G{i}" for i in range(g)]
    a.obs_names = [f"c{i}" for i in range(n)]
    a.obs["grp"] = pd.Categorical(labels)
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    return a


def test_testable_groups():
    s = pd.Series(pd.Categorical(["a", "a", "b", "c", "c"], categories=["a", "b", "c", "empty"]))
    groups, skipped = sparse_markers.testable_groups(s)
    assert groups == ["a", "c"]
    assert skipped == ["b", "empty"]
    assert sparse_markers.testable_groups(pd.Series(["x", "x", "y", "y"])) == ("all", [])
    with pytest.raises(ValueError, match="at least 2 cells"):
        sparse_markers.testable_groups(pd.Series(["x", "y"]))


def test_pipeline_markers_leave_out_one_cell_groups():
    a = _adata()
    pipeline._wilcoxon(a, "grp", key_added="m")
    rgg = a.uns["m"]
    assert list(rgg["names"].dtype.names) == ["A", "B", "C"]
    assert rgg["params"]["skipped_groups"] == ["one1", "one2"]

    ref = _adata()
    sc.tl.rank_genes_groups(ref, "grp", method="wilcoxon", pts=True, groups=["A", "B", "C"],
                            key_added="m")
    for field in ("names", "scores", "pvals", "pvals_adj", "logfoldchanges"):
        for grp in ("A", "B", "C"):
            np.testing.assert_array_equal(rgg[field][grp], ref.uns["m"][field][grp])


def test_pipeline_markers_unchanged_without_small_groups():
    a = _adata()
    a = a[a.obs["grp"].isin(["A", "B", "C"])].copy()
    a.obs["grp"] = a.obs["grp"].cat.remove_unused_categories()
    pipeline._wilcoxon(a, "grp", key_added="m")
    assert "skipped_groups" not in a.uns["m"]["params"]


def _register(dm, did: str) -> str:
    d = Path(dm.data_dir) / "uploads" / did
    d.mkdir(parents=True, exist_ok=True)
    (d / "metadata.json").write_text(json.dumps({"id": did, "name": did, "status": "ready"}))
    _adata().write_h5ad(d / "data.h5ad")
    return did


async def test_markers_api_names_skipped_groups(client, dataset_manager):
    did = _register(dataset_manager, "ds1cell")
    r = await client.get(f"{API}/datasets/{did}/markers",
                         params={"groupby_column": "grp", "format": "json", "n_genes": 5})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["groups"] == ["A", "B", "C"]
    assert body["skipped_groups"] == ["one1", "one2"]
    assert "fewer than 2 cells" in body["skipped_reason"]

    # the stored markers keep naming them (Arrow: a count in a header)
    r = await client.get(f"{API}/datasets/{did}/markers",
                         params={"groupby_column": "grp", "format": "arrow"})
    assert r.status_code == 200
    assert r.headers.get("X-Scview-Skipped-Groups") == "2"
