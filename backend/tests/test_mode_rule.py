"""One rule for large-dataset mode (v0.81): the run (``large_run.choose_mode``) and its
preflight (``preflight.dataset_preflight``: the pre-run note, the MCP ``run_pipeline``
check, the doublet alternative) both ask ``large_run.mode_rule``.

v0.80 QA (Reck 2025 kidney, 46,957 nuclei) saw the two disagree near the limit. For a
dataset not open yet the preflight costed its load into the need with the load margin,
while the run, which opens the dataset first, finds the load in memory used; and an
anndata < 0.7 file (Kang 2018) had no doublet channels from its file, so its doublet step
was costed as one channel (2.67 GB at the peak against 1.84 GB once open). Kang with
doublet detection: the old preflight kept the run in memory from a 4.5 GB limit, the run
itself from 4.0 GB.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from scview.core import large_run, memory_budget, preflight
from scview.core.pipeline import PipelineParams

GB = 1024**3
STEPS = ["qc_metrics", "normalization", "log_transform", "highly_variable_genes", "scaling", "pca"]


def _dataset(n: int = 400, g: int = 120) -> ad.AnnData:
    rng = np.random.default_rng(0)
    a = ad.AnnData(sp.csr_matrix(rng.poisson(1.0, (n, g)).astype(np.float32)))
    a.var_names = [f"G{i}" for i in range(g)]
    a.obs_names = [f"c{i}" for i in range(n)]
    a.obs["sample"] = pd.Categorical([f"s{i % 2}" for i in range(n)])
    return a


def _register(dm, did: str = "d") -> str:
    d = Path(dm.data_dir) / "uploads" / did
    d.mkdir(parents=True, exist_ok=True)
    (d / "metadata.json").write_text(json.dumps({"id": did, "name": did, "status": "ready"}))
    _dataset().write_h5ad(d / "data.h5ad")
    return did


def _budget(monkeypatch, limit, used) -> None:
    monkeypatch.setattr(memory_budget, "read_budget", lambda *a, **k: memory_budget.MemoryBudget(
        limit, used, True, "container limit"))


def _cached(monkeypatch, nbytes: int) -> None:
    monkeypatch.setattr(preflight, "cached_others",
                        lambda *a, **k: preflight.Cached(1 if nbytes else 0, nbytes))


def _both(dm, did, steps, chunked):
    """(the preflight's mode, the run's mode) for the same request."""
    pre = preflight.dataset_preflight(dm, did, steps, {"chunked": chunked})["mode"]
    run = large_run.choose_mode(dm, did, dm.get_dataset(did), steps, PipelineParams(chunked=chunked))
    return pre, run


# need, used, cached, limit: around the switch of a 6 GB limit (reserve 1.5 GB, margin 1.1)
GRID = list(itertools.product((1 * GB, int(3.2 * GB), 4 * GB, 9 * GB), (int(0.5 * GB), int(1.5 * GB)),
                              (0, 2 * GB), (4 * GB, 6 * GB)))


async def test_the_preflight_and_the_run_give_one_verdict_on_the_same_inputs(dataset_manager, monkeypatch):
    did = _register(dataset_manager)
    await dataset_manager.get_or_load_dataset(did)
    seen = set()
    for (need, used, cached, limit), requested, light, large in itertools.product(
            GRID, ("auto", "on", "off"), (False, True), (False, True)):
        _budget(monkeypatch, limit, used)
        _cached(monkeypatch, cached)
        monkeypatch.setattr(large_run, "in_memory_need", lambda *a, n=need: (n, n // 2))
        monkeypatch.setattr(large_run, "large_info", lambda path, big=large: {"x": 1} if big else None)
        steps = ["neighbors"] if light else STEPS  # neighbours alone need no matrix
        pre, run = _both(dataset_manager, did, steps, requested)
        assert (pre["chunked"], pre["light"], pre["reason"]) == (run.chunked, run.light, run.reason), \
            (need, used, cached, limit, requested, light, large)
        assert pre["need_bytes"] == run.need_bytes and pre["free_bytes"] == run.free_bytes
        seen.add((run.chunked, run.light))
    assert seen == {(True, False), (False, False), (False, True)}  # every outcome was met


async def test_a_dataset_not_open_yet_is_judged_as_the_run_will_find_it(dataset_manager, monkeypatch):
    """Not open: the need once it is open, and what opening it adds counted as used. With
    the load as estimated, the preflight's verdict is the run's (which opens it first)."""
    did = _register(dataset_manager)
    load = int(1.6 * GB)
    for need, used, limit in itertools.product((int(0.5 * GB), int(0.97 * GB), 2 * GB),
                                               (int(0.4 * GB), GB), (4 * GB, int(4.5 * GB), 5 * GB)):
        monkeypatch.setattr(large_run, "opening_need", lambda *a, n=need, **k: (n, n, load))
        monkeypatch.setattr(large_run, "in_memory_need", lambda *a, n=need: (n, n))
        _budget(monkeypatch, limit, used)
        dataset_manager.evict(did)
        pre = preflight.dataset_preflight(dataset_manager, did, STEPS, {})["mode"]
        await dataset_manager.get_or_load_dataset(did)
        _budget(monkeypatch, limit, used + load)  # what opening it added
        run = large_run.choose_mode(dataset_manager, did, dataset_manager.get_dataset(did), STEPS,
                                    PipelineParams())
        assert pre["chunked"] == run.chunked and pre["free_bytes"] == run.free_bytes, (need, used, limit)
        assert pre["reason"].startswith("Once this dataset is open, in memory the run needs")
        assert pre["reason"].removeprefix("Once this dataset is open, i") == run.reason.removeprefix("I")


async def test_kidney_47k_is_not_refused_by_the_preflight_where_the_run_works_in_memory(
        dataset_manager, monkeypatch):
    """Measured on the 47k kidney set (6 libraries, doublet detection): 0.42 GB used before
    opening, 8.35 GB at the peak in all, 7.5 GB more once open, opening adds 0.85 GB
    (measured 0.75-0.88). At a 12.75 GB limit the old preflight rule (the whole peak, load
    included, with the margin) said large-dataset mode, so the MCP refused the run and the
    UI asked; the run itself works in memory. Both now say in memory."""
    from scview.core.dataset_manager import LOAD_MARGIN, working_reserve

    limit, used, peak, run_need, load = (int(12.75 * GB), int(0.42 * GB), int(8.35 * GB),
                                         int(7.5 * GB), int(0.85 * GB))
    old_fits = peak * LOAD_MARGIN + working_reserve(limit) <= limit - used
    run_fits = run_need * LOAD_MARGIN + working_reserve(limit) <= limit - used - load
    assert not old_fits and run_fits  # the disagreement v0.80 shipped

    did = _register(dataset_manager)
    monkeypatch.setattr(large_run, "opening_need", lambda *a, **k: (run_need, run_need, load))
    monkeypatch.setattr(large_run, "in_memory_need", lambda *a: (run_need, run_need))
    _budget(monkeypatch, limit, used)
    pre = preflight.dataset_preflight(dataset_manager, did, STEPS, {})
    assert not pre["mode"]["chunked"] and "converts" not in pre["mode"]
    await dataset_manager.get_or_load_dataset(did)
    _budget(monkeypatch, limit, used + load)
    assert not large_run.choose_mode(dataset_manager, did, dataset_manager.get_dataset(did), STEPS,
                                     PipelineParams()).chunked


async def test_the_need_from_the_file_is_the_need_of_the_open_dataset(dataset_manager):
    """``opening_need`` (header of the file) and ``in_memory_need`` (the open dataset) cost
    the same run the same, so a load estimate is the only thing the preflight guesses."""
    did = _register(dataset_manager)
    path = dataset_manager.h5ad_candidates(did)[0]
    steps = ["qc_metrics", "doublet_detection", "filtering", "normalization", "log_transform",
             "highly_variable_genes", "scaling", "pca", "neighbors"]
    p = PipelineParams(doublet_channel_key="sample")
    need, run, load = large_run.opening_need(path, steps, p, used=GB, limit=64 * GB)
    adaptor = await dataset_manager.get_or_load_dataset(did)
    assert (need, run) == pytest.approx(large_run.in_memory_need(adaptor, steps, p), rel=0.02)
    assert load == preflight.resident_bytes(preflight.read_header(path, "h5ad"))


async def test_the_doublet_alternative_asks_the_same_rule(dataset_manager, monkeypatch):
    did = _register(dataset_manager)
    await dataset_manager.get_or_load_dataset(did)
    _budget(monkeypatch, 6 * GB, GB)
    _cached(monkeypatch, 0)
    steps = ["qc_metrics", "doublet_detection", "normalization"]
    for without, says in ((int(3.5 * GB), False), (int(3.0 * GB), True)):  # 3.85 / 3.3 + 1.5 vs 5 GB
        monkeypatch.setattr(large_run, "in_memory_need",
                            lambda a, s, *_, w=without: (9 * GB if "doublet_detection" in s else w, 0))
        mode = preflight.dataset_preflight(dataset_manager, did, steps, {})["mode"]
        assert mode["chunked"] and ("without_doublets" in mode) is says
        run = large_run.choose_mode(dataset_manager, did, dataset_manager.get_dataset(did),
                                    [s for s in steps if s != "doublet_detection"], PipelineParams())
        assert run.chunked is not says


def test_the_rule_asks_for_the_need_only_when_it_decides_on_it():
    calls = []

    def need():
        calls.append(1)
        return GB, GB

    for requested, light, large in (("on", False, False), ("off", False, False), ("auto", True, False),
                                    ("auto", False, True)):
        large_run.mode_rule(requested, need, used=GB, limit=6 * GB, light=light, large=large)
    assert not calls
    assert not large_run.mode_rule("auto", need, used=None, limit=None).chunked and not calls
    assert not large_run.mode_rule("auto", need, used=GB, limit=6 * GB).chunked and calls
    with pytest.raises(ValueError, match="auto, on, off"):
        large_run.mode_rule("maybe", need, used=GB, limit=6 * GB)


def test_an_old_layout_file_gets_the_doublet_plan_it_has_once_open(tmp_path):
    """anndata < 0.7 (Kang 2018): its channels and Scrublet's variable genes from the file,
    as the open dataset gives them, not one channel of every cell."""
    import h5py

    rng = np.random.default_rng(1)
    n, g = 600, 300
    acgt = np.array(list("ACGT"))
    names = ["".join(rng.choice(acgt, 16)) + f"-{1 + i % 2}" for i in range(n)]
    rec = lambda xs: np.array([(s.encode(),) for s in xs], dtype=[("index", "S24")])  # noqa: E731
    path = tmp_path / "old.h5ad"
    with h5py.File(path, "w") as f:
        f.create_dataset("X", data=rng.poisson(rng.gamma(0.5, 2.0, g), (n, g)).astype(np.float32))
        f.create_dataset("obs", data=rec(names))
        f.create_dataset("var", data=rec([f"g{j}" for j in range(g)]))
    assert large_run.is_legacy_layout(path)
    disk = large_run.doublet_plan_on_disk(path, PipelineParams())
    mem = preflight.doublet_plan(ad.read_h5ad(path), PipelineParams())
    assert disk is not None and disk.hvg is not None
    assert (disk.channels, disk.hvg, disk.costed) == (mem.channels, mem.hvg, mem.costed)
