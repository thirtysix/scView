"""Sub-clustering one group on its own graph (core/subcluster.py, C072)."""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import adjusted_rand_score

from scview.core import provenance
from scview.core.subcluster import SubclusterError, subcluster, target_column


def _planted() -> tuple[ad.AnnData, np.ndarray]:
    """Cluster "B" holds two cell types (B1, B2) that differ in their own genes;
    "A" differs from both in others. Counts in X."""
    rng = np.random.default_rng(0)
    sizes = {"A": 200, "B1": 120, "B2": 100}
    truth = np.repeat(list(sizes), list(sizes.values()))
    X = rng.poisson(1.0, (truth.size, 200)).astype("float32")
    X[truth == "A", 80:120] += rng.poisson(6, ((truth == "A").sum(), 40))
    X[truth == "B1", 0:20] += rng.poisson(5, ((truth == "B1").sum(), 20))
    X[truth == "B2", 20:40] += rng.poisson(5, ((truth == "B2").sum(), 20))
    a = ad.AnnData(X)
    a.var_names = [f"G{i}" for i in range(200)]
    a.obs_names = [f"c{i}" for i in range(truth.size)]
    a.obs["cl"] = pd.Categorical(np.where(truth == "A", "A", "B"), categories=["A", "B"])
    return a, truth


def test_a_mixed_cluster_splits_into_its_two_types():
    a, truth = _planted()
    res = subcluster(a, "cl", "B", resolution=0.3)
    assert res["column"] == "cl_sub" and res["n_cells"] == 220
    sub = a.obs["cl_sub"].astype(str)
    b = truth != "A"
    assert res["n_subclusters"] >= 2
    assert adjusted_rand_score(truth[b], sub[b]) > 0.8
    assert (sub[~b] == "A").all()  # the rest of the column is untouched
    assert list(a.obs["cl_sub"].cat.categories)[:2] == ["A", "B.0"]
    assert list(a.obs["cl"].cat.categories) == ["A", "B"]  # the source column too
    assert provenance.read_provenance(a)["history"][-1]["step"] == "subcluster"


def test_splitting_another_group_edits_the_same_sub_column():
    a, _ = _planted()
    subcluster(a, "cl", "B", resolution=0.3)
    first = a.obs["cl_sub"].astype(str).copy()
    subcluster(a, "cl", "A", resolution=0.3)
    after = a.obs["cl_sub"].astype(str)
    b = a.obs["cl"].astype(str) == "B"
    assert (after[b] == first[b]).all()  # B's split survives
    assert after[~b].str.startswith("A.").all()
    assert target_column("cl_sub") == "cl_sub"
    # Splitting a group of the sub column itself works there too.
    subcluster(a, "cl_sub", "B.0", resolution=0.3)
    assert a.obs["cl_sub"].astype(str).str.startswith("B.0.").any()


def test_requests_that_cannot_be_served_say_why():
    a, _ = _planted()
    with pytest.raises(SubclusterError, match="No cells"):
        subcluster(a, "cl", "Z")
    with pytest.raises(SubclusterError, match="not found"):
        subcluster(a, "nope", "A")
    with pytest.raises(SubclusterError, match="Resolution"):
        subcluster(a, "cl", "A", resolution=9)
    a.obs["tiny"] = pd.Categorical(["x"] * 10 + ["y"] * (a.n_obs - 10))
    with pytest.raises(SubclusterError, match="at least"):
        subcluster(a, "tiny", "x")
    assert "cl_sub" not in a.obs.columns and "tiny_sub" not in a.obs.columns


async def test_endpoint_saves_the_sub_column(client, dataset_manager):
    a, _ = _planted()
    d = Path(dataset_manager.data_dir) / "uploads" / "ds"
    d.mkdir(parents=True, exist_ok=True)
    (d / "metadata.json").write_text(json.dumps({"id": "ds", "name": "ds", "status": "ready"}))
    a.write_h5ad(d / "data.h5ad")

    r = await client.post("/api/v1/datasets/ds/clustering/subcluster",
                          json={"column": "cl", "group": "B", "resolution": 0.3})
    assert r.status_code == 200, r.text
    assert r.json()["column"] == "cl_sub"
    saved = ad.read_h5ad(dataset_manager.derived_h5ad_path("ds", "data.h5ad"))
    assert saved.obs["cl_sub"].astype(str).str.startswith("B.").sum() == 220
    info = (await client.get("/api/v1/datasets/ds")).json()
    assert "cl_sub" in [c["name"] for c in info["obs_columns"]]

    bad = await client.post("/api/v1/datasets/ds/clustering/subcluster",
                            json={"column": "cl", "group": "Z"})
    assert bad.status_code == 400
