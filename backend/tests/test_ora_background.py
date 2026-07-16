"""Tests for the experiment-specific ORA background and cache invalidation."""

from __future__ import annotations

import json
import types

import numpy as np
import pytest

from scview.core.ora_background import (
    ORA_VERSION,
    cache_is_current,
    get_background_genes,
    meta_key,
    resolve_markers_key,
    write_meta,
)


def _rgg(groups: list[str], genes: list[str]) -> dict:
    """Minimal rank_genes_groups-shaped uns entry (structured array, as scanpy writes)."""
    names = np.zeros(len(genes), dtype=[(g, "U32") for g in groups])
    for g in groups:
        names[g] = genes
    return {"names": names, "params": {"groupby": "cluster"}}


def _adata(uns: dict, var_names: list[str] | None = None, raw_var: list[str] | None = None):
    raw = types.SimpleNamespace(var_names=raw_var) if raw_var is not None else None
    return types.SimpleNamespace(uns=uns, var_names=var_names or [], raw=raw)


class TestResolveMarkersKey:
    def test_prefers_column_scoped_key(self):
        uns = {"rank_genes_groups__cluster": {}, "rank_genes_groups": {}}
        assert resolve_markers_key(uns, "cluster") == "rank_genes_groups__cluster"

    def test_falls_back_when_groupby_matches(self):
        uns = {"rank_genes_groups": _rgg(["a"], ["G1"])}
        assert resolve_markers_key(uns, "cluster") == "rank_genes_groups"

    def test_no_fallback_when_groupby_differs(self):
        uns = {"rank_genes_groups": _rgg(["a"], ["G1"])}
        assert resolve_markers_key(uns, "leiden") is None

    def test_returns_none_when_absent(self):
        assert resolve_markers_key({}, "cluster") is None


class TestGetBackgroundGenes:
    def test_returns_full_tested_gene_list_not_just_top_markers(self):
        genes = [f"G{i}" for i in range(500)]
        adata = _adata({"rank_genes_groups__cluster": _rgg(["B", "T"], genes)})
        bg = get_background_genes(adata, "cluster", "T")
        assert bg == genes
        assert len(bg) == 500, "background must be every tested gene, not the top-N query"

    def test_unknown_group_falls_back_to_first_field(self):
        genes = ["G1", "G2"]
        adata = _adata({"rank_genes_groups__cluster": _rgg(["B"], genes)})
        assert get_background_genes(adata, "cluster", "nope") == genes

    def test_falls_back_to_raw_var_names_when_no_markers(self):
        # scanpy's rank_genes_groups defaults to use_raw=True when raw exists,
        # so raw.var_names is what was testable.
        adata = _adata({}, var_names=["HVG1"], raw_var=["G1", "G2", "G3"])
        assert get_background_genes(adata, "cluster", "T") == ["G1", "G2", "G3"]

    def test_falls_back_to_var_names_when_no_raw(self):
        adata = _adata({}, var_names=["G1", "G2"])
        assert get_background_genes(adata, "cluster", "T") == ["G1", "G2"]

    def test_returns_none_when_nothing_recoverable(self):
        # Callers must leave gseapy's default alone rather than pass a wrong universe.
        adata = _adata({}, var_names=[])
        assert get_background_genes(adata, "cluster", "T") is None

    def test_coerces_numpy_str_to_python_str(self):
        adata = _adata({"rank_genes_groups__cluster": _rgg(["B"], ["G1"])})
        bg = get_background_genes(adata, "cluster", "B")
        assert all(type(g) is str for g in bg)


class TestCacheInvalidation:
    def test_meta_key_does_not_collide_with_enrichment_prefix(self):
        # assessor/assistant/pipeline scan "enrichment__" and parse the obs column
        # out of the key; a sidecar matching that prefix would be miscounted.
        key = meta_key("cluster", "CD8 T")
        assert not key.startswith("enrichment__")

    def test_legacy_payload_without_sidecar_is_stale(self):
        adata = _adata({"enrichment__cluster__T": "[]"})
        assert cache_is_current(adata, "cluster", "T") is False

    def test_roundtrip_marks_cache_current(self):
        adata = _adata({})
        write_meta(adata, "cluster", "T", background_n=14053, n_genes=100)
        assert cache_is_current(adata, "cluster", "T") is True
        assert json.loads(adata.uns[meta_key("cluster", "T")])["background_n"] == 14053

    def test_version_only_check_ignores_params(self):
        # The UI's "already computed" badge has no request to compare against.
        adata = _adata({})
        write_meta(adata, "cluster", "T", background_n=1, n_genes=100, collections=["h.all"])
        assert cache_is_current(adata, "cluster", "T") is True

    def test_older_version_is_stale(self):
        adata = _adata({meta_key("cluster", "T"): json.dumps({"ora_version": ORA_VERSION - 1})})
        assert cache_is_current(adata, "cluster", "T") is False

    def test_accepts_already_parsed_dict(self):
        adata = _adata({meta_key("cluster", "T"): {"ora_version": ORA_VERSION}})
        assert cache_is_current(adata, "cluster", "T") is True

    @pytest.mark.parametrize("bad", ["not json", "", None, 5])
    def test_corrupt_sidecar_is_stale_not_fatal(self, bad):
        adata = _adata({meta_key("cluster", "T"): bad})
        assert cache_is_current(adata, "cluster", "T") is False

    def test_no_column_is_never_current_and_writes_nothing(self):
        adata = _adata({})
        write_meta(adata, "", "T", background_n=1, n_genes=1)
        assert adata.uns == {}
        assert cache_is_current(adata, "", "T") is False


class TestCacheIsParamsAware:
    """pipeline.py and both endpoints share enrichment__{col}__{group}; a payload
    computed with different parameters must not be served for a later request."""

    def _written(self, **kw):
        adata = _adata({})
        write_meta(adata, "cluster", "T", background_n=100, **kw)
        return adata

    def test_same_params_hit(self):
        a = self._written(n_genes=100, collections=["h.all", "c5.go.bp"])
        assert cache_is_current(a, "cluster", "T", n_genes=100,
                                collections=["h.all", "c5.go.bp"]) is True

    def test_collection_order_does_not_matter(self):
        a = self._written(n_genes=100, collections=["c5.go.bp", "h.all"])
        assert cache_is_current(a, "cluster", "T", n_genes=100,
                                collections=["h.all", "c5.go.bp"]) is True

    def test_different_n_genes_misses(self):
        a = self._written(n_genes=100, collections=["h.all"])
        assert cache_is_current(a, "cluster", "T", n_genes=500,
                                collections=["h.all"]) is False

    def test_different_collections_misses(self):
        a = self._written(n_genes=100, collections=["h.all"])
        assert cache_is_current(a, "cluster", "T", n_genes=100,
                                collections=["c5.go.bp"]) is False

    def test_extra_collection_misses(self):
        a = self._written(n_genes=100, collections=["h.all"])
        assert cache_is_current(a, "cluster", "T", n_genes=100,
                                collections=["h.all", "c5.go.bp"]) is False

    def test_pipeline_payload_not_served_to_endpoint_with_other_collections(self):
        # The concrete collision: pipeline runs its 8 defaults, user then asks for one.
        a = self._written(n_genes=100, collections=[
            "h.all", "c2.cp.kegg_medicus", "c2.cp.reactome", "c2.cp.wikipathways",
            "c5.go.bp", "c5.go.cc", "c5.go.mf", "c8.all",
        ])
        assert cache_is_current(a, "cluster", "T", n_genes=100,
                                collections=["h.all"]) is False

    def test_legacy_sidecar_without_collections_misses_when_collections_requested(self):
        a = _adata({meta_key("cluster", "T"): json.dumps(
            {"ora_version": ORA_VERSION, "n_genes": 100})})
        assert cache_is_current(a, "cluster", "T", n_genes=100,
                                collections=["h.all"]) is False
