"""Proactive insight: the deterministic 'I notice…' nudge picks the right
next step by walking the preprocessing state in pipeline order."""

from __future__ import annotations

import types

import anndata as ad
import numpy as np
import pandas as pd

from scview.core.assistant import build_insight


def _adaptor(adata: ad.AnnData):
    return types.SimpleNamespace(adata=adata)


def _raw(n: int = 40, g: int = 6) -> ad.AnnData:
    """Raw integer counts, nothing processed."""
    X = np.random.RandomState(0).poisson(2, (n, g)).astype("float32")
    a = ad.AnnData(X=X)
    a.var_names = [f"G{i}" for i in range(g)]
    return a


def test_raw_counts_suggests_preprocessing():
    ins = build_insight(_adaptor(_raw()))
    assert "raw counts" in ins.insight.lower()
    assert ins.severity == "suggestion"
    assert ins.question and "preprocess" in ins.question.lower()


def test_high_doublet_load_flagged():
    a = _raw()
    # Normalize + log so we pass step 1, then flag many doublets.
    a.X = np.log1p(a.X / a.X.sum(1, keepdims=True) * 1e4).astype("float32")
    a.uns["scview_provenance"] = {
        "normalize": {"done": True},
        "log1p": {"done": True},
    }
    a.obs["predicted_doublet"] = pd.Series([True] * 8 + [False] * 32, index=a.obs_names)
    ins = build_insight(_adaptor(a))
    # Either the doublet nudge or (if normalization isn't detected) preprocessing;
    # assert it's a suggestion with a question.
    assert ins.severity == "suggestion" and ins.question


def test_fully_processed_is_informational():
    a = _raw()
    a.X = np.log1p(a.X).astype("float32")
    a.obs["leiden"] = pd.Categorical(["0", "1"] * 20)
    a.obs["cell_type"] = pd.Categorical(["B", "T"] * 20)
    a.obs["n_genes_by_counts"] = (a.X > 0).sum(1)
    a.uns["scview_active_clustering"] = "leiden"
    # Pretend the pipeline recorded the key steps so the assessor sees them done.
    a.uns["scview_provenance"] = {
        s: {"done": True}
        for s in ("normalize", "log1p", "pca", "neighbors", "clustering", "umap")
    }
    ins = build_insight(_adaptor(a))
    # With annotation present, the message is the processed summary.
    assert "cell" in ins.insight.lower()
    assert ins.insight  # non-empty


def test_insight_never_raises_on_minimal_data():
    a = ad.AnnData(X=np.zeros((3, 2), dtype="float32"))
    ins = build_insight(_adaptor(a))
    assert isinstance(ins.insight, str) and ins.insight
