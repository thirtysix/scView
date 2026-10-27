"""The marker-score annotator (pipeline._annotate_marker_score) on the marker
databases, its fallback to the built-in sets, and the replay script's record of
database labels. Fixtures: tests/test_marker_db.py."""

from __future__ import annotations

import pytest

from scview.core import marker_db as mdb
from tests.test_marker_db import _cells, installed, installed_cm3, raw, small_sources  # noqa: F401 (fixtures)


@pytest.fixture()
def annotator_db(monkeypatch, installed):  # noqa: F811 (the imported fixture)
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
    assert out["marker_source"] == "database" and out["databases"] == ["PanglaoDB", "CellMarker 2.0", "ACT"]


def test_annotator_falls_back_to_the_built_in_sets(monkeypatch, tmp_path):
    from scview.core.pipeline import PipelineParams, _annotate_marker_score

    monkeypatch.setattr(mdb, "marker_db_dir", lambda: tmp_path / "empty")
    monkeypatch.setattr(mdb, "auto_download_enabled", lambda: False)
    a = _cells()
    info = _annotate_marker_score(a, PipelineParams(), "lab", "ms")
    assert info["marker_source"] == "built-in" and a.uns["ms_markerscore_source"] == info
    assert a.uns["ms_markerscore_mapping"]["B"] == "B cell"
    assert not [c for c in a.obs.columns if c.startswith("_ms__")]


def test_built_in_sets_when_chosen_even_with_the_databases_installed(annotator_db):
    from scview.core.pipeline import PipelineParams, _annotate_marker_score

    a = _cells()
    info = _annotate_marker_score(a, PipelineParams(annotation_marker_sets="builtin"), "lab", "ms")
    assert info["marker_source"] == "built-in" and info["chosen"] is True
    assert a.uns["ms_markerscore_mapping"]["B"] == "B cell"


def test_replay_writes_database_labels_from_the_record():
    from scview.core.replay_script import RECORDED, _Ctx, _cell_type_annotation

    a = _cells()
    a.uns["ms_markerscore_mapping"] = {"B": "B cells"}
    a.uns["ms_markerscore_source"] = {"marker_source": "database"}
    snip = _cell_type_annotation({"annotation_method": "marker_score", "annotation_target": "ms",
                                  "annotation_groupby": "lab"}, _Ctx(adata=a))
    assert snip.status == RECORDED and any("'B cells'" in line or '"B cells"' in line for line in snip.code)
    assert not any("MARKER_SETS" in line for line in snip.code)


def test_annotator_uses_panglao_and_curated_cellmarker3(monkeypatch, installed_cm3):  # noqa: F811
    from scview.core.pipeline import PipelineParams, _run_cell_type_annotation

    monkeypatch.setattr(mdb, "marker_db_dir", lambda: installed_cm3)
    monkeypatch.setattr(mdb, "auto_download_enabled", lambda: False)
    a = _cells()
    a.obs["leiden"] = a.obs["lab"].cat.rename_categories(["0", "1", "2", "3"])
    out = _run_cell_type_annotation(a, PipelineParams(annotation_method="marker_score", annotation_target="ms"))
    assert a.uns["ms_markerscore_mapping"] == {"0": "B cells", "1": "NK cells", "2": "T cells", "3": "Unknown"}
    assert out["databases"] == ["PanglaoDB", "CellMarker 3.0 (curated)"] and out["cellmarker"] == "CellMarker 3.0"
    assert a.uns["ms_markerscore_source"]["cellmarker"] == "CellMarker 3.0"


def _immune_cells():
    """Four clusters shaped like Kang's pDC, plasma, CD16 and CD14 monocytes: each
    expresses its own markers, and pDC also JCHAIN / MZB1 (as on Kang)."""
    import anndata as ad
    import numpy as np
    import pandas as pd

    genes = ["LILRA4", "CLEC4C", "IL3RA", "LAMP5", "JCHAIN", "MZB1", "IGHG1", "TNFRSF17", "SDC1",
             "FCGR3A", "MS4A7", "LST1", "CDKN1C", "C1QA", "C1QB", "CD68", "CD14", "LYZ", "S100A8",
             "S100A9", "FCN1"] + [f"G{i}" for i in range(60)]
    own = {"pDC": ["LILRA4", "CLEC4C", "IL3RA", "LAMP5", "JCHAIN", "MZB1"],
           "plasma": ["MZB1", "IGHG1", "TNFRSF17", "SDC1", "JCHAIN"],
           "cd16": ["FCGR3A", "MS4A7", "LST1", "CDKN1C", "C1QA", "C1QB", "CD68", "LYZ"],
           "cd14": ["CD14", "LYZ", "S100A8", "S100A9", "FCN1", "CD68"]}
    rng = np.random.default_rng(0)
    rows, labels = [], []
    for lab, markers in own.items():
        block = rng.poisson(0.2, size=(40, len(genes))).astype(float)
        for g in markers:
            block[:, genes.index(g)] += rng.poisson(6, size=40)
        rows.append(block)
        labels += [lab] * 40
    a = ad.AnnData(np.log1p(np.vstack(rows)), var=pd.DataFrame(index=genes))
    a.obs["lab"] = pd.Categorical(labels)
    return a


def test_built_in_sets_name_pdc_and_cd16_monocytes():
    from scview.core.pipeline import MARKER_SETS_VERSION, PipelineParams, _annotate_marker_score

    a = _immune_cells()
    info = _annotate_marker_score(a, PipelineParams(annotation_marker_sets="builtin"), "lab", "ms")
    assert a.uns["ms_markerscore_mapping"] == {
        "cd14": "Monocyte", "cd16": "CD16+ monocyte", "pDC": "Plasmacytoid dendritic cell",
        "plasma": "Plasma cell"}
    assert info["marker_sets_version"] == MARKER_SETS_VERSION == 2


def test_replay_uses_the_marker_sets_the_run_recorded():
    from scview.core.replay_script import _builtin_marker_sets, _Ctx

    a = _cells()
    old = _builtin_marker_sets({}, _Ctx(adata=a), "ms")
    assert "Plasmacytoid dendritic cell" not in old and "JCHAIN" in old["Plasma cell"]
    a.uns["ms_markerscore_source"] = {"marker_source": "built-in", "marker_sets_version": 2}
    assert "Plasmacytoid dendritic cell" in _builtin_marker_sets({}, _Ctx(adata=a), "ms")
    assert "CD16+ monocyte" in _builtin_marker_sets({"marker_sets_version": 2}, _Ctx(adata=None), "ms")


def test_app_annotation_columns_name_how_each_column_was_made():
    import pandas as pd

    from scview.core.anndata_adaptor import AnnDataAdaptor
    from scview.core.celltypist_fit import record_run

    a = _cells()
    for col in ("ms", "llm_type", "ct"):
        a.obs[col] = pd.Categorical(["x"] * a.n_obs)
    a.uns["ms_markerscore_mapping"] = {"0": "B cell"}
    a.uns["llm_type_llm_mapping"] = {"0": "B cell"}
    a.uns["gone_markerscore_mapping"] = {"0": "B cell"}  # its column was deleted
    record_run(a, "ct", {"model": "Immune_All_Low.pkl"})
    adaptor = AnnDataAdaptor.__new__(AnnDataAdaptor)
    adaptor._adata = a
    assert adaptor.app_annotation_columns() == {"ct": "celltypist", "llm_type": "llm", "ms": "marker_score"}
