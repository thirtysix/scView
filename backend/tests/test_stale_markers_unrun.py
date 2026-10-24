"""Markers of a clustering column written again by a run without the marker step
(core/step_failures.py ``settle_unrun``).

The clustering writes ``scview_leiden_r<res>``; when its graph changed in the run and
the marker step was not part of it (a run that failed before the markers, or one that
left them out), the earlier run's markers and enrichment for that column stayed attached
to the new labels.
"""

from __future__ import annotations

from scview.core import provenance
from scview.core.pipeline import run_pipeline
from scview.core.uns_keys import markers_key
from tests.test_step_failures import AFTER_PCA, PREP, _counts, _fail, _params


def _processed():
    out, res = run_pipeline(_counts(), [*PREP, *AFTER_PCA], _params())
    assert res.status == "complete" and markers_key(out.uns["scview_active_clustering"]) in out.uns
    return out


def test_a_run_that_failed_before_the_markers_drops_them_as_stale(monkeypatch):
    a = _processed()
    col = a.uns["scview_active_clustering"]
    _fail(monkeypatch, "embeddings", "umap broke")
    out, res = run_pipeline(a, [*PREP, "neighbors", "clustering", "embeddings"], _params())
    assert res.status == "finished_with_errors"
    assert out.uns["scview_active_clustering"] == col  # the new labels, same column
    assert markers_key(col) not in out.uns and "rank_genes_groups" not in out.uns
    assert markers_key(col) in res.cleared["uns"]
    rec = provenance.read_provenance(out)["history"][-1]["params"]
    assert "marker_genes" in rec["stale"] and markers_key(col) in rec["cleared"]["uns"]
    from scview.core.assessor import assess_preprocessing

    assert not assess_preprocessing(out).marker_genes.done


def test_a_complete_run_without_the_markers_drops_them_too():
    a = _processed()
    col = a.uns["scview_active_clustering"]
    out, res = run_pipeline(a, ["neighbors", "clustering"], _params(n_neighbors=8))
    assert res.status == "complete"
    assert markers_key(col) not in out.uns and markers_key(col) in res.cleared["uns"]


def test_reclustering_the_same_graph_keeps_them():
    a = _processed()
    col = a.uns["scview_active_clustering"]
    out, res = run_pipeline(a, ["clustering"], _params())
    assert res.cleared == {} and markers_key(col) in out.uns


def test_a_run_with_the_markers_leaves_it_to_the_marker_step():
    a = _processed()
    col = a.uns["scview_active_clustering"]
    out, res = run_pipeline(a, ["neighbors", "clustering", "marker_genes"], _params())
    assert res.status == "complete" and res.cleared == {} and markers_key(col) in out.uns
