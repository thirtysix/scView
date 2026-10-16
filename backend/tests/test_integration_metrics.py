"""Integration metrics beyond the mixing score: iLISI, bio-conservation and the
over-correction warning (core/integration.batch_mixing, C070)."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd

from scview.core.integration import batch_mixing


def _joint(corrected: str) -> ad.AnnData:
    """Two datasets x three cell types. Uncorrected: types apart, each dataset
    shifted. Corrected: 'good' removes the shift and keeps the types; 'over'
    collapses everything into one blob."""
    rng = np.random.default_rng(0)
    types = np.repeat(["T", "B", "Mono"], 200)
    datasets = np.tile(np.repeat(["d1", "d2"], 100), 3)
    centres = {"T": 0.0, "B": 8.0, "Mono": 16.0}
    base = np.stack([np.array([centres[t], 0.0]) for t in types]) + rng.normal(0, 1, (600, 2))
    shift = np.where(datasets == "d2", 4.0, 0.0)[:, None] * np.array([[0.0, 1.0]])
    a = ad.AnnData(np.zeros((600, 3), dtype="float32"),
                   obs=pd.DataFrame({"dataset": datasets, "cell_type": types},
                                    index=[f"c{i}" for i in range(600)]))
    a.obsm["X_pca"] = (base + shift).astype("float32")
    a.obsm["X_pca_harmony"] = (base if corrected == "good" else rng.normal(0, 1, (600, 2))).astype("float32")
    a.obs["source_cluster"] = (pd.Series(datasets) + ":" + pd.Series(types)).to_numpy()
    return a


def test_good_correction_mixes_and_keeps_cell_types():
    m = batch_mixing(_joint("good"), k=15)
    assert m["ilisi"]["corrected"] > m["ilisi"]["uncorrected"]
    assert abs(m["ilisi"]["ideal"] - 2.0) < 1e-6
    cons = m["conservation"]
    assert cons["label_column"] == "cell_type" and cons["kind"] == "shared"
    assert cons["corrected"] > 0.95 and cons["uncorrected"] > 0.95
    assert m["overcorrection"] is None


def test_over_correction_is_flagged_with_the_types_that_lost_most():
    m = batch_mixing(_joint("over"), k=15)
    cons = m["conservation"]
    assert cons["uncorrected"] - cons["corrected"] > 0.3
    assert m["overcorrection"] and "merging distinct cell types" in m["overcorrection"]
    assert any(t in m["overcorrection"] for t in ("T", "B", "Mono"))


def test_without_shared_labels_each_datasets_own_clusters_are_used():
    a = _joint("over")
    del a.obs["cell_type"]
    m = batch_mixing(a, k=15)
    assert m["conservation"]["kind"] == "within_dataset"
    assert m["conservation"]["label_column"] == "source_cluster"
    assert m["overcorrection"] and "own dataset's cluster" in m["overcorrection"]
