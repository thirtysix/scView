"""Tests for the enrichment term-similarity network."""

from __future__ import annotations

import pytest

from scview.core.enrichment_network import (
    DEFAULT_MAX_TERM_SIZE,
    DEFAULT_MIN_SIMILARITY,
    DEFAULT_MIN_TERM_SIZE,
    build_network,
)


def term(name, genes, padj=1e-5, **kw):
    d = {"term": name, "genes": list(genes), "adjusted_pvalue": padj,
         "overlap_count": len(genes), "gene_count": len(genes) * 10, "collection": "c5.go.bp"}
    d.update(kw)
    return d


class TestFiltering:
    def test_empty_input(self):
        n = build_network([])
        assert n["nodes"] == [] and n["clusters"] == [] and n["n_terms"] == 0

    def test_non_significant_terms_excluded(self):
        n = build_network([term("A", ["G1"], padj=0.2)], fdr=0.05)
        assert n["n_terms"] == 0

    def test_terms_without_genes_excluded(self):
        n = build_network([term("A", [])])
        assert n["n_terms"] == 0

    def test_missing_padj_excluded(self):
        n = build_network([{"term": "A", "genes": ["G1"]}])
        assert n["n_terms"] == 0

    def test_truncation_is_reported_not_silent(self):
        terms = [term(f"T{i}", [f"G{i}"], padj=1e-9 * i) for i in range(1, 21)]
        n = build_network(terms, max_terms=5)
        assert n["n_terms"] == 5
        assert n["truncated"] == 15, "a bounded view must report what it dropped"

    def test_truncation_keeps_most_significant(self):
        terms = [term(f"T{i}", [f"G{i}"], padj=10.0 ** -i) for i in range(1, 11)]
        n = build_network(terms, max_terms=3)
        assert {x["term"] for x in n["nodes"]} == {"T10", "T9", "T8"}


class TestSimilarityMetric:
    """The central correction: Jaccard misses containment, combined catches it."""

    # A 5-gene term fully inside a 100-gene term: maximally redundant.
    CHILD = [f"G{i}" for i in range(5)]
    PARENT = [f"G{i}" for i in range(100)]

    def _edges(self, metric, thresh):
        # Size filtering off: these assert metric behaviour, not filtering.
        n = build_network(
            [term("child", self.CHILD), term("parent", self.PARENT)],
            metric=metric, min_similarity=thresh,
            min_term_size=0, max_term_size=10**9,
        )
        return n["edges"]

    def test_jaccard_at_0_1_disconnects_perfect_containment(self):
        # 5/100 = 0.05 < 0.1. This is the blog method's blind spot.
        assert self._edges("jaccard", 0.1) == []

    def test_overlap_coefficient_sees_containment(self):
        # 5/min(5,100) = 1.0
        assert len(self._edges("overlap", 0.5)) == 1

    def test_combined_at_enrichmentmap_default_sees_containment(self):
        # 0.5*0.05 + 0.5*1.0 = 0.525 >= 0.375
        edges = self._edges("combined", DEFAULT_MIN_SIMILARITY)
        assert len(edges) == 1
        assert edges[0]["n_shared"] == 5
        assert edges[0]["jaccard"] == pytest.approx(0.05)

    def test_disjoint_terms_never_connect(self):
        for metric in ("combined", "jaccard", "overlap"):
            n = build_network(
                [term("A", ["G1", "G2"]), term("B", ["G3", "G4"])],
                metric=metric, min_similarity=0.01,
            )
            assert n["edges"] == [], metric

    def test_identical_sets_are_similarity_one(self):
        n = build_network([term("A", ["G1", "G2"]), term("B", ["G1", "G2"])],
                          metric="combined", min_similarity=0.99)
        assert n["edges"][0]["weight"] == pytest.approx(1.0)

    def test_rejects_unknown_metric(self):
        with pytest.raises(ValueError, match="metric must be one of"):
            build_network([term("A", ["G1"])], metric="cosine")

    @pytest.mark.parametrize("bad", [0.0, 1.0, 2.0, -0.5])
    def test_rejects_out_of_range_resolution(self, bad):
        # CPM resolution is a density threshold on weights that never exceed 1;
        # >= 1 silently returns all singletons, so it must be rejected.
        with pytest.raises(ValueError, match="resolution must be between 0 and 1"):
            build_network([term("A", ["G1"])], resolution=bad)


class TestClustering:
    def _two_blocks(self):
        # Two internally-redundant programs sharing nothing across.
        a = [term(f"A{i}", ["G1", "G2", "G3", f"X{i}"]) for i in range(4)]
        b = [term(f"B{i}", ["H1", "H2", "H3", f"Y{i}"]) for i in range(4)]
        return a + b

    def test_separates_disconnected_programs(self):
        n = build_network(self._two_blocks())
        assert n["n_clusters"] == 2
        groups = {}
        for node in n["nodes"]:
            groups.setdefault(node["cluster"], set()).add(node["term"][0])
        assert all(len(v) == 1 for v in groups.values()), "clusters must not mix programs"

    def test_every_node_assigned_and_one_hub_per_cluster(self):
        n = build_network(self._two_blocks())
        assert all("cluster" in node for node in n["nodes"])
        for c in n["clusters"]:
            hubs = [x for x in n["nodes"] if x["cluster"] == c["cluster"] and x["is_hub"]]
            assert len(hubs) == 1
            assert hubs[0]["term"] == c["label"]

    def test_cluster_counts_sum_to_terms(self):
        n = build_network(self._two_blocks())
        assert sum(c["n_terms"] for c in n["clusters"]) == n["n_terms"]

    def test_singletons_still_get_a_cluster(self):
        n = build_network([term("A", ["G1"]), term("B", ["G2"]), term("C", ["G3"])])
        assert n["n_clusters"] == 3
        assert n["edges"] == []

    def test_single_term_does_not_crash_layout(self):
        n = build_network([term("A", ["G1", "G2"])])
        assert n["n_terms"] == 1 and n["n_clusters"] == 1
        assert n["nodes"][0]["is_hub"] is True

    def test_clusters_sorted_by_significance_not_size(self):
        # Size tracks ontology granularity, so ranking by it buries strong small
        # programs under large weak ones.
        big_weak = [term(f"W{i}", ["G1", "G2", "G3", f"X{i}"], padj=1e-4, gene_count=50)
                    for i in range(6)]
        small_strong = [term(f"S{i}", ["H1", "H2", "H3", f"Y{i}"], padj=1e-40, gene_count=50)
                        for i in range(2)]
        n = build_network(big_weak + small_strong)
        assert n["clusters"][0]["label"].startswith("S"), "strongest program must rank first"
        assert n["clusters"][0]["n_terms"] < n["clusters"][1]["n_terms"]

    def test_cluster_members_listed_strongest_first(self):
        terms = [term("weak", ["G1", "G2", "G3"], padj=1e-3, gene_count=50),
                 term("strong", ["G1", "G2", "G3"], padj=1e-30, gene_count=50)]
        n = build_network(terms, min_similarity=0.1)
        assert n["clusters"][0]["terms"] == ["strong", "weak"]


class TestIdenticalTermsNeverSplit:
    """Regression: leidenalg's RBConfigurationVertexPartition penalises communities
    by size, so above resolution ~1 it shattered cliques. On the ovary sample data
    six terms with byte-identical gene sets (similarity exactly 1.0) landed in six
    different clusters, which is the exact opposite of what this feature is for.
    CPM's resolution is a density threshold instead, so a clique survives at any
    setting."""

    def _identical(self, n=6):
        genes = [f"G{i}" for i in range(20)]
        return [term(f"T{i}", genes, padj=10.0 ** -(90 + i), gene_count=100) for i in range(n)]

    @pytest.mark.parametrize("resolution", [0.1, 0.2, 0.4, 0.6, 0.9])
    def test_identical_gene_sets_stay_together_at_every_resolution(self, resolution):
        n = build_network(self._identical(), resolution=resolution)
        clusters = {node["cluster"] for node in n["nodes"]}
        assert len(clusters) == 1, (
            f"terms with identical gene sets were split into {len(clusters)} clusters "
            f"at resolution {resolution}"
        )

    def test_identical_sets_are_one_program(self):
        n = build_network(self._identical())
        assert n["n_clusters"] == 1
        assert n["clusters"][0]["n_terms"] == 6

    def test_two_identical_blocks_give_exactly_two_programs(self):
        a = [term(f"A{i}", [f"G{j}" for j in range(20)], gene_count=100) for i in range(4)]
        b = [term(f"B{i}", [f"H{j}" for j in range(20)], gene_count=100) for i in range(4)]
        n = build_network(a + b)
        assert n["n_clusters"] == 2


class TestDeterminism:
    def test_same_seed_same_result(self):
        terms = [term(f"A{i}", ["G1", "G2", f"X{i}"]) for i in range(6)]
        a = build_network(terms, seed=42)
        b = build_network(terms, seed=42)
        assert [n["cluster"] for n in a["nodes"]] == [n["cluster"] for n in b["nodes"]]
        assert [n["x"] for n in a["nodes"]] == [n["x"] for n in b["nodes"]]

    def test_resolution_is_exposed(self):
        # Cluster count is resolution-dependent; that knob must not be hidden.
        terms = [term(f"A{i}", ["G1", "G2", "G3", f"X{i}"], gene_count=50) for i in range(6)]
        coarse = build_network(terms, resolution=0.1)
        fine = build_network(terms, resolution=0.9)
        assert coarse["n_clusters"] <= fine["n_clusters"]

    def test_params_echoed_for_provenance(self):
        n = build_network([term("A", ["G1"])], seed=7, resolution=0.6, metric="jaccard")
        assert n["params"] == {
            "fdr": 0.05, "metric": "jaccard", "min_similarity": DEFAULT_MIN_SIMILARITY,
            "resolution": 0.6, "seed": 7, "min_term_size": DEFAULT_MIN_TERM_SIZE,
            "max_term_size": DEFAULT_MAX_TERM_SIZE,
        }


class TestTermSizeFilter:
    """GO:BP terms run from 1 to ~2000 genes and scView applies no size filter,
    so the network must bound them or vague parents dominate every cluster."""

    def test_oversized_terms_dropped_and_counted(self):
        terms = [term("huge", [f"G{i}" for i in range(20)], gene_count=1543),
                 term("ok", [f"G{i}" for i in range(20)], gene_count=100)]
        n = build_network(terms)
        assert [x["term"] for x in n["nodes"]] == ["ok"]
        assert n["n_size_filtered"] == 1

    def test_undersized_terms_dropped(self):
        terms = [term("tiny", ["G1", "G2"], gene_count=3),
                 term("ok", ["G1", "G2"], gene_count=100)]
        n = build_network(terms)
        assert [x["term"] for x in n["nodes"]] == ["ok"]

    def test_filter_is_disableable(self):
        terms = [term("huge", ["G1", "G2"], gene_count=1543)]
        n = build_network(terms, min_term_size=0, max_term_size=10**9)
        assert n["n_terms"] == 1 and n["n_size_filtered"] == 0

    def test_missing_gene_count_is_filtered_not_crashed(self):
        n = build_network([{"term": "A", "genes": ["G1"], "adjusted_pvalue": 1e-9}])
        assert n["n_terms"] == 0 and n["n_size_filtered"] == 1


class TestClusterLabelling:
    """Labelling by weighted degree picks the most GENERIC term, because generic
    terms overlap everything. Significance is the better rule."""

    def test_label_is_most_significant_not_best_connected(self):
        # "generic" connects to everything (high degree) but is weakly significant.
        generic = term("generic", ["G1", "G2", "G3", "G4"], padj=1e-3, gene_count=400)
        members = [term(f"M{i}", ["G1", "G2", f"X{i}"], padj=1e-4, gene_count=50)
                   for i in range(4)]
        strong = term("strong", ["G1", "G2", "G3"], padj=1e-12, gene_count=60)
        n = build_network([generic, *members, strong], min_similarity=0.1)
        big = max(n["clusters"], key=lambda c: c["n_terms"])
        assert big["label"] == "strong"

    def test_hub_flag_matches_label(self):
        n = build_network([term("A", ["G1", "G2"], padj=1e-9, gene_count=50),
                           term("B", ["G1", "G2"], padj=1e-3, gene_count=50)],
                          min_similarity=0.1)
        hub = [x for x in n["nodes"] if x["is_hub"]]
        assert len(hub) == 1 and hub[0]["term"] == "A"
