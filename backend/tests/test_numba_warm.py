"""What numba has compiled, in this process and on disk (core/numba_warm.py)."""

import importlib.machinery
import os
import sys
import types

import numba
import numpy as np
import pytest

from scview.core import numba_warm as nw


def _fake_module(monkeypatch, name, *functions):
    mod = types.ModuleType(name)
    for fn in functions:
        setattr(mod, fn, numba.njit(lambda x: x + 1))
    monkeypatch.setitem(sys.modules, name, mod)
    return mod


def test_compiled_means_numba_has_a_signature_in_this_process(monkeypatch):
    mod = _fake_module(monkeypatch, "umap.umap_", "compute_membership_strengths", "smooth_knn_dist")
    assert not nw.graph_compiled()
    mod.compute_membership_strengths(1.0)
    assert not nw.graph_compiled()  # both functions
    mod.smooth_knn_dist(1.0)
    assert nw.graph_compiled()
    monkeypatch.delitem(sys.modules, "pynndescent.pynndescent_", raising=False)
    assert not nw.search_compiled()  # never imports to find out


def test_real_graph_weights_mark_the_process_warm():
    from umap.umap_ import fuzzy_simplicial_set

    X = np.random.default_rng(0).normal(size=(60, 5)).astype(np.float32)
    fuzzy_simplicial_set(X, 10, np.random.RandomState(0), "euclidean")
    assert nw.graph_compiled() and nw.graph_seconds() == 0.0


def test_compile_left_adds_the_import_only_when_something_is_left(monkeypatch):
    state = {"graph": False, "search": False, "layout": False}
    monkeypatch.setattr(nw, "graph_compiled", lambda: state["graph"])
    monkeypatch.setattr(nw, "search_compiled", lambda: state["search"])
    monkeypatch.setattr(nw, "layout_compiled", lambda: state["layout"])
    monkeypatch.setattr(nw, "import_seconds", lambda: 5.0)
    small, large = nw.SEARCH_MIN_CELLS - 1, nw.SEARCH_MIN_CELLS
    assert nw.graph_seconds() == 5.0 + nw.GRAPH_SECONDS
    assert nw.neighbours_seconds(small) == 5.0 + nw.GRAPH_SECONDS  # exact neighbours: no search
    assert nw.neighbours_seconds(large) == 5.0 + nw.GRAPH_SECONDS + nw.SEARCH_SECONDS
    assert nw.umap_seconds(large) == nw.neighbours_seconds(large) + nw.LAYOUT_SECONDS
    state["graph"] = True  # e.g. BBKNN ran: the search is still to come on a large dataset
    assert nw.neighbours_seconds(small) == 0.0
    assert nw.neighbours_seconds(large) == 5.0 + nw.SEARCH_SECONDS
    state["search"] = state["layout"] = True
    assert nw.neighbours_seconds(large) == nw.umap_seconds(large) == 0.0
    assert nw.neighbours_cold_seconds(large) == (nw.IMPORT_COLD_SECONDS + nw.GRAPH_SECONDS + nw.SEARCH_SECONDS)


def test_import_seconds_reads_the_process_then_the_disk(monkeypatch):
    monkeypatch.setitem(sys.modules, "umap.umap_", types.ModuleType("umap.umap_"))
    assert nw.import_seconds() == 0.0
    monkeypatch.delitem(sys.modules, "umap.umap_")
    monkeypatch.setattr(nw, "on_disk", lambda package="pynndescent": True)
    assert nw.import_seconds() == nw.IMPORT_CACHED_SECONDS
    monkeypatch.setattr(nw, "on_disk", lambda package="pynndescent": False)
    assert nw.import_seconds() == nw.IMPORT_COLD_SECONDS


def test_cache_subpath_is_numbas():
    from numba.core.caching import _CacheLocator

    d = os.path.dirname(numba.__file__)
    assert nw._cache_subpath(d) == _CacheLocator.get_suitable_cache_subpath(os.path.join(d, "x.py"))


@pytest.mark.parametrize("where", ["tree", "numba_cache_dir", "user"])
def test_on_disk_finds_an_index_in_each_place_numba_writes(tmp_path, monkeypatch, where):
    pkg = tmp_path / "site" / "fakepkg"
    pkg.mkdir(parents=True)
    spec = importlib.machinery.ModuleSpec("fakepkg", None, is_package=True)
    spec.submodule_search_locations = [str(pkg)]
    monkeypatch.setattr(nw.importlib.util, "find_spec", lambda name: spec if name == "fakepkg" else None)
    monkeypatch.setenv("NUMBA_CACHE_DIR", str(tmp_path / "nc"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert not nw.on_disk("fakepkg")
    sub = nw._cache_subpath(str(pkg))
    target = {"tree": pkg / "__pycache__", "numba_cache_dir": tmp_path / "nc" / sub,
              "user": tmp_path / "xdg" / "numba" / sub}[where]
    target.mkdir(parents=True)
    (target / "utils.make_heap-130.py311.1.nbc").write_bytes(b"")
    assert not nw.on_disk("fakepkg")  # data without its index
    (target / "utils.make_heap-130.py311.nbi").write_bytes(b"")
    assert nw.on_disk("fakepkg")
    assert not nw.on_disk("no_such_package_here")


def _state(monkeypatch, compiled: bool, import_seconds: float = nw.IMPORT_CACHED_SECONDS):
    for fn in ("graph_compiled", "search_compiled", "layout_compiled"):
        monkeypatch.setattr(nw, fn, lambda: compiled)
    monkeypatch.setattr(nw, "import_seconds", lambda: 0.0 if compiled else import_seconds)


def test_first_run_notes_follow_the_process(monkeypatch):
    from scview.core import hvg_sensitivity as hs
    from scview.core import layout_stability as ls
    from scview.core import merge_check as mc
    from scview.core import wnn

    # Kang after a server restart (2 CPUs): the variable-gene check took 30 s more than after
    # the pipeline, the merge check 30 s, layout stability 28 s.
    _state(monkeypatch, compiled=False)
    for secs in (hs.first_run_seconds(13_836), mc.first_run_seconds(13_836), ls.first_run_seconds(13_836)):
        assert abs(secs - 29.4) <= 0.2 * 29.4
    assert hs.first_run_seconds(3_000) < 10  # exact neighbours: no search to compile
    assert wnn.first_run_seconds(7_798) == wnn.FIRST_RUN_SECONDS
    _state(monkeypatch, compiled=True)
    assert hs.first_run_seconds(13_836) == mc.first_run_seconds(13_836) == ls.first_run_seconds(13_836) == 0
    assert wnn.first_run_seconds(7_798) == 0
    assert wnn.estimate_seconds(7_798) == wnn.estimate_seconds(7_798, warm=True)


def test_pipeline_neighbours_and_velocity_leave_out_a_done_compile(monkeypatch):
    from scview.core import preflight as pf
    from scview.core import velocity as vel

    steps = ["pca", "neighbors"]
    h = (13_836, 14_053)

    def neighbours(warm):
        run = pf.estimate_run(h, nnz=13_836 * 1_000, steps=steps, warm=warm, cpus=2.0)
        return next(e for e in run.steps if e.step == "neighbors").seconds

    _state(monkeypatch, compiled=False, import_seconds=nw.IMPORT_COLD_SECONDS)
    cold = neighbours(False)
    assert neighbours(True) == cold  # nothing compiled yet: the whole compile still counts
    _state(monkeypatch, compiled=True)
    assert neighbours(False) == cold  # warm False pins the cold estimate
    warm = neighbours(True)
    # Kang: 28-31 s first, 1.4 s in a warm server
    assert warm[0] <= 1.4 <= warm[1] and cold[0] <= 29.6 <= cold[1]
    monkeypatch.setattr(vel, "knn_compiled", lambda: False)
    assert pf.estimate_velocity(13_836, 13_836, 10**7, 3 * 10**6)["compile_seconds"] == 0
    _state(monkeypatch, compiled=False)
    assert pf.estimate_velocity(13_836, 13_836, 10**7, 3 * 10**6)["compile_seconds"] == pf.VELOCITY_SECONDS_KNN_COMPILE


def test_the_search_is_compiled_per_distance(monkeypatch):
    """pynndescent compiles nn_descent once per distance function; its signatures name it."""
    mod = types.ModuleType("pynndescent.pynndescent_")

    def alternative_cosine(x):
        return x

    def squared_euclidean(x):
        return x

    class Dispatcher:
        signatures: list = []

    mod.nn_descent = Dispatcher()
    monkeypatch.setitem(sys.modules, "pynndescent.pynndescent_", mod)
    monkeypatch.setattr(nw, "graph_compiled", lambda: True)
    monkeypatch.setattr(nw, "layout_compiled", lambda: True)
    monkeypatch.setattr(nw, "import_seconds", lambda: 0.0)
    assert nw.search_metrics() == set() and nw.search_seconds("cosine") == nw.SEARCH_SECONDS
    assert nw.graph_umap_seconds(100, "euclidean") == 0.0  # exact neighbours, nothing to compile
    assert nw.graph_umap_seconds(100, "cosine") == nw.SEARCH_SECONDS  # scanpy searches cosine at any size
    mod.nn_descent.signatures = [("array(float32, 2d, C)", f"type(CPUDispatcher({alternative_cosine}))")]
    assert nw.search_metrics() == {"cosine"} and nw.graph_umap_seconds(100, "cosine") == 0.0
    assert nw.search_seconds("euclidean") == nw.SEARCH_OTHER_METRIC_SECONDS
    assert nw.graph_umap_seconds(5_000, "euclidean", exact_below=4_097) == nw.SEARCH_OTHER_METRIC_SECONDS
    mod.nn_descent.signatures.append(("array(float32, 2d, C)", f"type(CPUDispatcher({squared_euclidean}))"))
    assert nw.search_metrics() == {"cosine", "euclidean"} and nw.search_seconds("euclidean") == 0.0


def test_real_search_signatures_name_the_distance():
    from pynndescent import NNDescent

    X = np.random.default_rng(0).normal(size=(300, 5)).astype(np.float32)
    NNDescent(X, metric="cosine", n_neighbors=5, random_state=0)
    assert "cosine" in nw.search_metrics()
