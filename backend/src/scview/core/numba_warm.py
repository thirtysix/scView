"""What numba has compiled already, so a "first run" time estimate adds only the compile still to come.

umap-learn and pynndescent compile their code with numba on first use in a process. Every feature
that builds a neighbour graph or a UMAP compiles the same functions (the pipeline's neighbours,
BBKNN's weights, velocity, WNN, the variable-gene check, the merge check, layout stability), so a
flag per feature ("BBKNN has run here") misses the compile another feature already paid: Compare
methods said 25.6 s on Kang and took 5.6 s in a server that had built a graph before. The state
asked here is numba's own: a dispatcher holds its compiled signatures for the life of the process.

pynndescent also keeps some of its code on disk (``cache=True``, in the package's ``__pycache__``, or
under ``NUMBA_CACHE_DIR`` or numba's user cache when the package is read-only). A container restart
keeps that, a recreate does not, and it only shortens the import: umap-learn's own graph and layout
code and pynndescent's search are compiled again in every process.

Measured at 2 CPUs (Kang 13,836 cells, 50 PCs, numba 0.68.1, umap-learn 0.5.12, pynndescent 0.6.0,
2026-11-06): ``import umap.umap_`` 8.9 s with no cache on disk, 5.1 s with it; umap-learn's graph
weights (``smooth_knn_dist``, ``compute_membership_strengths``) 2.6 s; pynndescent's search (scanpy
uses it from 8,192 cells with the euclidean metric, exact neighbours below) about 20 s; the UMAP
layout's epoch about 1 s. A neighbour graph of Kang took 28-31 s first and 1.3 s after.

pynndescent compiles its search once per distance (its dispatcher's signatures name the distance
function): the ATAC graph (cosine) after the pipeline's euclidean search, or WNN's euclidean index
after an ATAC graph, still compiles. Measured at 2 CPUs (2026-11-14): cosine first 11-19 s, the
other distance after it 4-7 s (5k mouse kidney: ATAC graph 20.7 s cold, 1.7 s warm; WNN 21.3 s
with its euclidean search to compile, 14.1 s after).
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import re
import sys

IMPORT_COLD_SECONDS = 8.9  # import umap.umap_ (pynndescent compiles its typed functions on import)
IMPORT_CACHED_SECONDS = 5.1  # the same with pynndescent's cache on disk
GRAPH_SECONDS = 2.6  # umap-learn's graph weights
SEARCH_SECONDS = 20.0  # pynndescent's neighbour search
LAYOUT_SECONDS = 1.0  # the UMAP layout's epoch
SEARCH_MIN_CELLS = 8192  # scanpy searches exactly below this (euclidean); pynndescent from here
SEARCH_OTHER_METRIC_SECONDS = 6.0  # the search for one more distance once it is compiled for another


def _compiled(module: str, name: str) -> bool:
    """Whether ``module.name`` (a numba dispatcher) is compiled in this process. Never imports."""
    mod = sys.modules.get(module)
    return bool(getattr(getattr(mod, name, None), "signatures", None))


def graph_compiled() -> bool:
    """umap-learn's graph weights (every UMAP-style neighbour graph, BBKNN's included)."""
    return _compiled("umap.umap_", "compute_membership_strengths") and _compiled("umap.umap_", "smooth_knn_dist")


def search_compiled() -> bool:
    """pynndescent's approximate neighbour search."""
    return _compiled("pynndescent.pynndescent_", "nn_descent")


def search_metrics() -> set[str]:
    """The distances pynndescent's search is compiled for in this process ("cosine",
    "euclidean", or a function's own name), from its dispatcher's signatures. Never imports."""
    sigs = getattr(getattr(sys.modules.get("pynndescent.pynndescent_"), "nn_descent", None), "signatures", None) or []
    found = set()
    for sig in sigs:
        for name in re.findall(r"function ([\w.<>]+) at", " ".join(str(a) for a in sig)):
            name = name.rsplit(".", 1)[-1]
            found.add(next((m for m in ("cosine", "euclidean") if m in name), name))
    return found


def search_seconds(metric: str) -> float:
    """pynndescent's compile still to come before a search with ``metric``: none when compiled
    for it, ``SEARCH_OTHER_METRIC_SECONDS`` when compiled for another distance only, else
    ``SEARCH_SECONDS``."""
    done = search_metrics()
    if metric in done:
        return 0.0
    return SEARCH_OTHER_METRIC_SECONDS if done else SEARCH_SECONDS


def graph_umap_seconds(n_cells: int, metric: str = "euclidean", exact_below: int = SEARCH_MIN_CELLS) -> float:
    """Compile still to come before a neighbour graph with ``metric`` and its UMAP: scanpy searches
    exactly (no pynndescent) only for euclidean below ``exact_below`` cells; the import when anything
    is left."""
    exact = metric == "euclidean" and int(n_cells) < exact_below
    secs = (0.0 if exact else search_seconds(metric)) + (0.0 if graph_compiled() else GRAPH_SECONDS) \
        + (0.0 if layout_compiled() else LAYOUT_SECONDS)
    return secs + import_seconds() if secs else 0.0


def layout_compiled() -> bool:
    """umap-learn's layout (one epoch of the embedding's optimisation)."""
    return _compiled("umap.layouts", "_nb_optimize_layout_euclidean_single_epoch")


def _cache_subpath(package_dir: str) -> str:
    """numba's directory for a source directory's cache under a cache root (``caching.get_suitable_cache_subpath``)."""
    path = os.path.abspath(package_dir)
    return "_".join([os.path.split(path)[-1], hashlib.sha1(path.encode()).hexdigest()])


def cache_dirs(package_dir: str) -> list[str]:
    """Where numba keeps a package's cache: in the tree, under ``NUMBA_CACHE_DIR``, numba's user cache."""
    sub = _cache_subpath(package_dir)
    user = os.path.join(os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache"), "numba")
    dirs = [os.path.join(package_dir, "__pycache__"), os.path.join(user, sub)]
    if os.environ.get("NUMBA_CACHE_DIR"):
        dirs.insert(0, os.path.join(os.environ["NUMBA_CACHE_DIR"], sub))
    return dirs


def on_disk(package: str = "pynndescent") -> bool:
    """Whether a numba cache index (``*.nbi``) of ``package`` is on disk. Finds it without importing it."""
    try:
        spec = importlib.util.find_spec(package)
    except (ImportError, ValueError):
        return False
    locations = list(spec.submodule_search_locations or []) if spec else []
    for d in (c for loc in locations for c in cache_dirs(loc)):
        try:
            if any(name.endswith(".nbi") for name in os.listdir(d)):
                return True
        except OSError:
            continue
    return False


def import_seconds() -> float:
    """Importing umap-learn (and pynndescent) in this process, if it is still to come."""
    if "umap.umap_" in sys.modules:
        return 0.0
    return IMPORT_CACHED_SECONDS if on_disk() else IMPORT_COLD_SECONDS


def _left(*parts: tuple[bool, float]) -> float:
    """Seconds of the parts not compiled yet, plus the import when any is still to come."""
    secs = sum(seconds for done, seconds in parts if not done)
    return secs + import_seconds() if secs else 0.0


def graph_seconds() -> float:
    """Compile still to come before a UMAP-style graph from given neighbours (BBKNN's)."""
    return _left((graph_compiled(), GRAPH_SECONDS))


def neighbours_seconds(n_cells: int) -> float:
    """Compile still to come before scanpy's neighbour graph of ``n_cells`` (the search from
    ``SEARCH_MIN_CELLS``, then the graph weights)."""
    search = int(n_cells) < SEARCH_MIN_CELLS or search_compiled()
    return _left((graph_compiled(), GRAPH_SECONDS), (search, SEARCH_SECONDS))


def umap_seconds(n_cells: int) -> float:
    """Compile still to come before a neighbour graph and its UMAP layout."""
    search = int(n_cells) < SEARCH_MIN_CELLS or search_compiled()
    return _left((graph_compiled(), GRAPH_SECONDS), (search, SEARCH_SECONDS), (layout_compiled(), LAYOUT_SECONDS))


def neighbours_cold_seconds(n_cells: int) -> float:
    """``neighbours_seconds`` in a fresh process with no cache on disk."""
    return IMPORT_COLD_SECONDS + GRAPH_SECONDS + (SEARCH_SECONDS if int(n_cells) >= SEARCH_MIN_CELLS else 0.0)
