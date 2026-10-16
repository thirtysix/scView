"""uns cache keys survive "/" in column and group names, and old files still read."""

from __future__ import annotations

import json
import warnings
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

from scview.core import uns_keys as k
from scview.core.anndata_adaptor import AnnDataAdaptor
from scview.core.ora_background import cache_is_current, meta_key, write_meta
from scview.core.pipeline import _finalize_current, drop_enrichment_cache, PipelineParams
from scview.core import provenance

TCM = "Tcm/Naive helper T cells"


def test_names_without_a_slash_keep_their_old_keys():
    assert k.markers_key("cell_type") == "rank_genes_groups__cell_type"
    assert k.enrichment_key("cell_type", "CD4 T") == "enrichment__cell_type__CD4 T"
    assert meta_key("leiden", "3") == "ora_meta__leiden__3"


def test_slash_is_encoded_and_parses_back():
    key = k.enrichment_key("cell/type", TCM)
    assert "/" not in key
    assert k.enrichment_column(key) == "cell/type"
    assert k.markers_column(k.markers_key("cell/type")) == "cell/type"
    assert "/" not in meta_key("ct", TCM)
    assert k.markers_column("enrichment__a__b") is None
    assert k.enrichment_column("rank_genes_groups__a") is None


def test_drop_enrichment_cache_only_drops_that_column():
    a = ad.AnnData(np.zeros((2, 2), dtype="float32"))
    a.uns[k.enrichment_key("cell/type", TCM)] = "[]"
    write_meta(a, "cell/type", TCM, background_n=1, n_genes=1)
    a.uns[k.enrichment_key("cell", "x")] = "[]"
    drop_enrichment_cache(a, "cell/type")
    assert list(a.uns) == [k.enrichment_key("cell", "x")]


def _markers_adata(column: str, labels: list[str]) -> ad.AnnData:
    rng = np.random.default_rng(0)
    X = rng.poisson(1.0, (60, 12)).astype("float32")
    X[:30, :4] += 8
    a = ad.AnnData(X)
    a.var_names = [f"G{i}" for i in range(12)]
    a.obs[column] = pd.Categorical([labels[0]] * 30 + [labels[1]] * 30)
    return a


def test_column_with_a_slash_round_trips(tmp_path):
    a = _markers_adata("cell/type", [TCM, "B"])
    sc.tl.rank_genes_groups(a, "cell/type", method="wilcoxon", key_added=k.markers_key("cell/type"))
    a.uns[k.enrichment_key("cell/type", TCM)] = "[]"
    path = tmp_path / "x.h5ad"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)  # the obs column name itself
        a.write_h5ad(path)

    adaptor = AnnDataAdaptor(str(path))
    assert adaptor.marker_columns() == ["cell/type"]
    assert adaptor.has_markers(column="cell/type")
    assert set(adaptor.get_markers(column="cell/type")["group"]) == {TCM, "B"}
    assert k.enrichment_key("cell/type", TCM) in adaptor.adata.uns

    _finalize_current(adaptor.adata, PipelineParams(), [])
    current = provenance.read_provenance(adaptor.adata)["current"]
    assert current["markers_for"] == ["cell/type"]
    assert current["enrichment_for"] == ["cell/type"]


def test_file_saved_with_raw_slash_keys_is_repaired_on_load(tmp_path):
    # What earlier versions wrote: h5py nested each "/" into a sub-group.
    a = _markers_adata("ct", [TCM, "B"])
    sc.tl.rank_genes_groups(a, "ct", method="wilcoxon", key_added="rank_genes_groups__ct")
    a.uns["rank_genes_groups__a/b"] = a.uns["rank_genes_groups__ct"]
    a.uns[f"enrichment__ct__{TCM}"] = json.dumps([{"term": "X"}])
    write_meta(a, "ct", "B", background_n=12, n_genes=5)  # an unaffected sidecar
    a.uns[f"ora_meta__ct__{TCM}"] = a.uns[meta_key("ct", "B")]
    path = tmp_path / "old.h5ad"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        a.write_h5ad(path)
    assert "enrichment__ct__Tcm" in ad.read_h5ad(path).uns, "the bug this repairs"

    uns = AnnDataAdaptor(str(path)).adata.uns
    assert "enrichment__ct__Tcm" not in uns and "ora_meta__ct__Tcm" not in uns
    assert json.loads(uns[k.enrichment_key("ct", TCM)]) == [{"term": "X"}]
    assert "names" in uns[k.markers_key("a/b")]
    assert "rank_genes_groups__a" not in uns
    assert "names" in uns["rank_genes_groups__ct"], "a normal markers entry is left alone"

    holder = SimpleNamespace(uns=uns)  # cache_is_current only reads .uns
    assert cache_is_current(holder, "ct", TCM, n_genes=5)
    assert cache_is_current(holder, "ct", "B", n_genes=5)


def test_repair_leaves_a_clean_file_untouched():
    uns = {"rank_genes_groups__ct": {"names": [], "params": {}},
           "enrichment__ct__T": "[]", "ora_meta__ct__T": json.dumps({"ora_version": 3}),
           "other": {"nested": {"x": 1}}}
    before = dict(uns)
    assert k.repair_slash_keys(uns) == 0
    assert uns == before
