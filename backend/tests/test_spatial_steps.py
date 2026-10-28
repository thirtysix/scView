"""Doublet detection on Visium spots (core/spatial.py, core/pipeline.py): skipped with
the reason, recorded in provenance and replayed as a skip; not offered by the
assessment or the doublet card."""

from __future__ import annotations

from scview.core import doublets, provenance, replay_script, spatial
from tests.test_spatial import _squidpy_like


# ---------------------------------------------------------------------------
# Doublet detection does not apply to spots
# ---------------------------------------------------------------------------


def test_doublet_step_is_skipped_on_spots_and_recorded():
    from scview.core.assessor import assess_preprocessing
    from scview.core.pipeline import run_pipeline

    a = _squidpy_like(n=30)
    spatial.adopt(a)
    assert assess_preprocessing(a).doublet_detection.done  # not offered as missing
    out, res = run_pipeline(a, ["doublet_detection"])
    assert res.notes["doublet_detection"] == spatial.DOUBLET_SKIP
    assert "doublet_score" not in out.obs.columns
    (entry,) = [s for s in provenance.read_provenance(out)["history"]
                if s["step"] == "doublet_detection_skipped"]
    assert entry["params"]["reason"] == spatial.DOUBLET_SKIP
    script = replay_script.build_script(out)
    assert f"# doublet_detection was skipped: {spatial.DOUBLET_SKIP}." in script


def test_streamed_run_reports_the_skip():
    from scview.core.pipeline import run_pipeline_streamed

    a = _squidpy_like(n=30)
    spatial.adopt(a)
    events = list(run_pipeline_streamed(a, ["doublet_detection"]))
    (done,) = [e for name, e in events if name == "step_done"]
    assert done["skipped"] and done["note"] == spatial.DOUBLET_SKIP


def test_assessment_and_doublet_card_do_not_recommend_it():
    from scview.core.assessor import assess_preprocessing

    a = _squidpy_like(n=30)
    spatial.adopt(a)
    st = assess_preprocessing(a).doublet_detection
    assert st.details.startswith("Not applicable") and spatial.DOUBLET_SKIP in st.caution
    assert doublets.summary(a)["not_applicable"] == spatial.DOUBLET_SKIP


def test_cell_coordinates_are_not_spots():
    from scview.core.assessor import assess_preprocessing

    a = _squidpy_like(key="X_spatial", lib=None)
    del a.obs["array_row"]
    spatial.adopt(a)
    assert not spatial.is_spot_data(a)
    assert not assess_preprocessing(a).doublet_detection.done
    assert "not_applicable" not in doublets.summary(a)
