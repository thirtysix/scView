"""The marker-score annotator (pipeline._annotate_marker_score) on the marker
databases, its fallback to the built-in sets, and the replay script's record of
database labels. Fixtures: tests/test_marker_db.py."""

from __future__ import annotations

import pytest

from scview.core import marker_db as mdb
from tests.test_marker_db import _cells, installed, raw, small_sources  # noqa: F401 (fixtures)


@pytest.fixture()
def annotator_db(monkeypatch, installed):
    monkeypatch.setattr(mdb, "marker_db_dir", lambda: installed)
    monkeypatch.setattr(mdb, "auto_download_enabled", lambda: False)


def test_annotator_names_clusters_from_the_databases(annotator_db):
    from scview.core.pipeline import PipelineParams, _run_cell_type_annotation

    a = _cells()
    a.obs["leiden"] = a.obs["lab"].cat.rename_categories(["0", "1", "2", "3"])
    out = _run_cell_type_annotation(a, PipelineParams(annotation_method="marker_score", annotation_target="ms"))
    m = a.uns["ms_markerscore_mapping"]
    # Cluster 3's markers (XYZ1, XYZ2, TRDC) are in no type with 5 measured genes
    assert m == {"0": "B cells", "1": "NK cells", "2": "T cells", "3": "Unknown"}
    assert list(a.obs["ms"].iloc[[0, 30, 60]]) == ["B cells", "NK cells", "T cells"]
    assert a.uns["ms_markerscore_source"]["marker_source"] == "database"
    assert out["marker_source"] == "database" and out["databases"] == ["PanglaoDB", "CellMarker 2.0"]


def test_annotator_falls_back_to_the_built_in_sets(monkeypatch, tmp_path):
    from scview.core.pipeline import PipelineParams, _annotate_marker_score

    monkeypatch.setattr(mdb, "marker_db_dir", lambda: tmp_path / "empty")
    monkeypatch.setattr(mdb, "auto_download_enabled", lambda: False)
    a = _cells()
    info = _annotate_marker_score(a, PipelineParams(), "lab", "ms")
    assert info["marker_source"] == "built-in" and a.uns["ms_markerscore_source"] == info
    assert a.uns["ms_markerscore_mapping"]["B"] == "B cell"
    assert not [c for c in a.obs.columns if c.startswith("_ms__")]


def test_replay_writes_database_labels_from_the_record():
    from scview.core.replay_script import RECORDED, _Ctx, _cell_type_annotation

    a = _cells()
    a.uns["ms_markerscore_mapping"] = {"B": "B cells"}
    a.uns["ms_markerscore_source"] = {"marker_source": "database"}
    snip = _cell_type_annotation({"annotation_method": "marker_score", "annotation_target": "ms",
                                  "annotation_groupby": "lab"}, _Ctx(adata=a))
    assert snip.status == RECORDED and any("'B cells'" in line or '"B cells"' in line for line in snip.code)
    assert not any("MARKER_SETS" in line for line in snip.code)
