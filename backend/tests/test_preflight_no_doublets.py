"""A first run without doublet detection keeps what that step's warm-up stood for
(``preflight.NO_DOUBLET_KEPT_PER_CELL``): its estimate was 0.2-0.5 GB low on larger data
(emont 2022 fat, 20,000 cells: 1.46 GB estimated, 1.71 measured)."""

from __future__ import annotations

import pytest

from scview.core import preflight as pf
from scview.core.pipeline import ALL_STEPS

STEPS = ["reset_to_counts", "qc_metrics", "filtering", "normalization", "log_transform",
         "highly_variable_genes", "scaling", "pca", "neighbors", "clustering", "embeddings",
         "marker_genes", "cell_cycle"]
BIG = {"shape": (20_000, 24_000), "nnz": 50_000_000}


def _both(monkeypatch, **kw):
    """(the estimate now, the estimate without the refit)."""
    now = pf.estimate_run(**kw)
    with monkeypatch.context() as m:
        m.setattr(pf, "NO_DOUBLET_KEPT_PER_CELL", 0)
        old = pf.estimate_run(**kw)
    return now, old


def test_a_cold_run_without_doublets_keeps_it_from_filtering_on(monkeypatch):
    now, old = _both(monkeypatch, **BIG, steps=STEPS, warm=False)
    kept = min(pf.WARMUP["doublet_detection"], pf.NO_DOUBLET_KEPT_PER_CELL * 20_000)
    assert kept == pf.WARMUP["doublet_detection"]  # capped at the step's own warm-up
    after = ALL_STEPS.index("doublet_detection")
    for a, b in zip(now.steps, old.steps):
        if a.step == "save":
            assert a.peak_bytes - b.peak_bytes == kept
            continue
        assert a.peak_bytes - b.peak_bytes == (kept if ALL_STEPS.index(a.step) > after else 0), a.step




def test_paul15_sized_run_keeps_its_cells_share(monkeypatch):
    now, old = _both(monkeypatch, shape=(2_730, 3_451), dense=True, steps=STEPS, warm=False)
    assert now.peak_bytes - old.peak_bytes == pf.NO_DOUBLET_KEPT_PER_CELL * 2_730  # 30 MB


@pytest.mark.parametrize("kw", [
    {"steps": STEPS, "warm": True},  # a warm process holds it already
    {"steps": ["doublet_detection", *STEPS], "warm": False},  # the step's own warm-up counts it
    {"steps": ["reset_to_counts", "qc_metrics"], "warm": False},  # before the step's place
])
def test_nothing_more_otherwise(monkeypatch, kw):
    now, old = _both(monkeypatch, **BIG, **kw)
    assert now.peak_bytes == old.peak_bytes
