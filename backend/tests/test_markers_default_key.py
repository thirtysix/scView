"""Markers stored under scanpy's bare rank_genes_groups key are found by column.

A file processed with sc.tl.rank_genes_groups(adata, "louvain") listed its groups
in the Enrichment tab, but computing enrichment for any of them said there were
no marker genes.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

from scview.core.anndata_adaptor import AnnDataAdaptor


def test_markers_under_the_default_key_are_found_by_column(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.poisson(1.0, (90, 20)).astype("float32")
    X[:30, :5] += 8
    a = ad.AnnData(X)
    a.var_names = [f"G{i}" for i in range(20)]
    a.obs["louvain"] = pd.Categorical(["0"] * 30 + ["1"] * 30 + ["2"] * 30)
    sc.tl.rank_genes_groups(a, "louvain", method="wilcoxon")
    path = tmp_path / "x.h5ad"
    a.write_h5ad(path)

    adaptor = AnnDataAdaptor(str(path))
    assert adaptor.has_markers(column="louvain")
    df = adaptor.get_markers(groupby="0", column="louvain", n_genes=5)
    assert df is not None and set(df["group"]) == {"0"} and len(df) == 5
    assert not adaptor.has_markers(column="other")
    assert adaptor.get_markers(column="other") is None
