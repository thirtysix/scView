"""Tests for dependency-aware re-run planning."""

from __future__ import annotations

from scview.core.rerun import descendants, plan_rerun


def test_descendants():
    d = descendants("clustering")
    assert "marker_genes" in d and "enrichment" in d
    assert "pca" not in d and "neighbors" not in d  # upstream, not descendants


def test_descendants_of_pca_includes_graph_and_clustering():
    d = descendants("pca")
    assert {"neighbors", "clustering", "marker_genes", "enrichment", "embeddings"} <= d


def test_rerun_clustering_keeps_pca_and_graph():
    ran = ["qc_metrics", "normalization", "log_transform", "pca", "neighbors",
           "clustering", "marker_genes", "enrichment"]
    plan = plan_rerun(ran, "clustering")
    assert plan.requires_reprocess is False
    # re-runs clustering + its downstream that were previously run
    assert plan.rerun_steps == ["clustering", "marker_genes", "enrichment"]
    # PCA and the neighbour graph are NOT recomputed
    assert "pca" in plan.kept_steps and "neighbors" in plan.kept_steps


def test_rerun_only_includes_previously_run_downstream():
    # enrichment was never run → not in the re-run set
    ran = ["pca", "neighbors", "clustering", "marker_genes"]
    plan = plan_rerun(ran, "clustering")
    assert plan.rerun_steps == ["clustering", "marker_genes"]
    assert "enrichment" not in plan.rerun_steps


def test_editing_upstream_requires_reprocess():
    ran = ["normalization", "log_transform", "pca", "neighbors", "clustering"]
    plan = plan_rerun(ran, "normalization")
    assert plan.requires_reprocess is True
    assert "reprocess" in plan.message.lower()


def test_rerun_markers_only():
    ran = ["pca", "neighbors", "clustering", "marker_genes"]
    plan = plan_rerun(ran, "marker_genes")
    assert plan.rerun_steps == ["marker_genes"]
    assert plan.requires_reprocess is False


def test_unknown_step():
    plan = plan_rerun(["clustering"], "bogus")
    assert plan.rerun_steps == []
    assert "unknown" in plan.message.lower()
