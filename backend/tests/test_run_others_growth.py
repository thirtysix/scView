"""A run's pre-step check counts what other runs and reads are still expected to grow by
(``RunControl.others_growth``, ``DatasetManager.outstanding`` without this run): two runs that
each pass alone can overrun together. A run alone sees the numbers it always did.
"""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp

from scview.core import memory_budget, preflight

GB = 1024**3


class _Avail:
    def __init__(self, free: float, used: float = 2.0, limit: float = 8.0) -> None:
        self.limited, self.limit_bytes = True, int(limit * GB)
        self.available_bytes, self.used_bytes = int(free * GB), int(used * GB)


def _check(monkeypatch, need: float, free: float, others: float):
    from scview.core.run_watchdog import RunControl

    monkeypatch.setattr(preflight, "available_memory", lambda: _Avail(free))
    monkeypatch.setattr(preflight, "step_peak_for", lambda *_a: int(need * GB))
    control = RunControl(plan=None)
    control.other_growth = lambda used: int(others * GB) if used == int(2 * GB) else 0
    adata = ad.AnnData(sp.csr_matrix(np.ones((3, 2))))
    return control.halt_before("pca", adata, None)


def test_the_pre_step_check_counts_the_other_runs_growth(monkeypatch):
    assert _check(monkeypatch, 3.0, 6.0, 0.0) is None
    assert _check(monkeypatch, 3.0, 6.0, 3.0) is None  # 0.8 x 3 fits in the 3 left
    status, why = _check(monkeypatch, 3.0, 6.0, 4.0)
    assert status == "stopped"
    assert why.startswith("Stopped before pca: it needs about 3.0 GB more at its peak, and of the 6.0 GB")
    assert "other work still running will take about 4.0 GB" in why and "Run it again" in why
    assert "SCVIEW_MEMORY_GB" not in why


def test_a_step_too_big_by_itself_keeps_its_sentence(monkeypatch):
    status, why = _check(monkeypatch, 20.0, 6.0, 4.0)
    assert status == "stopped" and "Raise SCVIEW_MEMORY_GB" in why and "other work" not in why


def test_a_run_alone_sees_no_others(monkeypatch):
    from scview.core.run_watchdog import RunControl

    assert RunControl().others_growth(GB) == 0
    c = RunControl()
    c.other_growth = lambda used: 1 / 0
    assert c.others_growth(GB) == 0  # the check goes on without them


async def test_two_runs_are_wired_to_each_other_through_the_manager(dataset_manager, tmp_data_dir, monkeypatch):
    """``_make_run_room`` gives the control the manager's count without its own run."""
    from scview.core import processing
    from scview.core.run_watchdog import RunControl

    d = Path(tmp_data_dir) / "uploads" / "x"
    d.mkdir(parents=True)
    (d / "metadata.json").write_text(json.dumps({"id": "x", "name": "x", "status": "ready"}))
    ad.AnnData(sp.csr_matrix(np.ones((5, 3), dtype=np.float32))).write_h5ad(d / "data.h5ad")
    adaptor = await dataset_manager.get_or_load_dataset("x")
    monkeypatch.setattr(memory_budget, "read_budget", lambda *_a, **_k: memory_budget.MemoryBudget(
        20 * GB, 2 * GB, True, "container limit"))
    monkeypatch.setattr(memory_budget, "release_memory", lambda: None)
    dataset_manager.reserve_run("other", 3 * GB)
    dataset_manager.reserve_run("x", 4 * GB)  # this run's own
    control = RunControl()
    await processing._make_run_room(dataset_manager, "x", adaptor, control)
    assert control.others_growth(2 * GB) == 3 * GB


def test_the_large_mode_check_counts_them_too(monkeypatch):
    from scview.core import large_run
    from scview.core.run_watchdog import RunControl

    monkeypatch.setattr(preflight, "available_memory", lambda: _Avail(6.0))
    monkeypatch.setattr(preflight, "chunked_step_peak", lambda *_a, **_k: 3 * GB)
    control = RunControl()
    control.other_growth = lambda used: 4 * GB

    class Run:
        memory_check = large_run.LargeRun.memory_check
        control = None
        _written = True
        light = None
        params = None

        def shape_for_estimate(self):
            return None

        def _doublet_plan(self, step):
            return None

    r = Run()
    r.control = control
    why = r.memory_check("pca")
    assert why.startswith("Stopped before pca: in large-dataset mode it needs about 3.0 GB more at its peak")
    assert "other work still running" in why
    control.other_growth = lambda used: 0
    assert r.memory_check("pca") is None
