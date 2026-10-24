"""Leiden clustering in a child process, so the server keeps answering.

leidenalg's optimiser is C++ that never releases the GIL: about 1.3 s per
iteration on Kang, and the backend is one process, so every clustering, the
resolution sweep and the variable-gene check froze all other requests
(/health included) for up to ~5 s at a time.

The child runs exactly what ``sc.tl.leiden`` runs by default (the leidenalg
flavour): the same directed graph with edges in the same order, float64
weights, RBConfiguration partition, seed and ``n_iterations=-1``. The clusters
are therefore identical to scanpy's. It imports numpy, igraph and leidenalg
(~1 s), not scanpy (~3.5 s). Run as ``python -m scview.core.leiden`` it is the
child: graph in on stdin, membership out on stdout, both as .npz/.npy bytes.

Fast Leiden (``FAST_ITERATIONS``, opt-in, the pipeline's ``leiden_iterations``) is the
same call stopped after two passes: ``sc.tl.leiden(..., n_iterations=2)``. On the same
saved graph of the brain data (Roihu, resolution 0.5) the full optimisation took 234 s /
1,586 s / 8,738 s at 89k / 222k / 444k cells (about n^2.2), two passes 17 / 44 / 138 /
446 s at 89k / 222k / 444k / 1.21M (about n^1.25), with the same cluster count, a
modularity-type score (RBConfiguration) 0.1-0.35 % lower, and ARI 0.93-0.97 against the
full optimisation's labels, while the full optimisation with another seed agrees with
itself only at ARI 0.46-0.60. Work that re-clusters a dataset (the resolution sweep, the
variable-gene check, the merge check, sub-clustering) uses the setting its clustering was
made with (``recorded_iterations``).
"""

from __future__ import annotations

import contextlib
import io
import subprocess
import sys
from typing import Any

import numpy as np
import pandas as pd

RANDOM_STATE = 0  # sc.tl.leiden's default
N_ITERATIONS = -1  # sc.tl.leiden's default: until a pass improves nothing
FAST_ITERATIONS = 2  # fast Leiden: the same optimisation stopped after two passes
ITERATION_CHOICES = (N_ITERATIONS, FAST_ITERATIONS)


class LeidenError(RuntimeError):
    """The child process failed."""


def check_iterations(value: Any) -> int:
    """The pipeline's ``leiden_iterations`` as -1 (full optimisation) or 2 (fast Leiden);
    a ValueError naming the choices for anything else."""
    n = None
    if not isinstance(value, bool):
        try:
            f = float(value)
            n = int(f) if f.is_integer() else None
        except (TypeError, ValueError, OverflowError):
            n = None
    if n not in ITERATION_CHOICES:
        raise ValueError(f"Unknown Leiden iterations '{value}': choose -1 (full optimisation) or "
                         f"{FAST_ITERATIONS} (fast Leiden).")
    return n


def recorded_iterations(adata, column: str | None = None) -> int:
    """The ``n_iterations`` the clustering ``column`` (default the active clustering) was
    made with: its ``uns[column]["params"]`` (as ``sc.tl.leiden`` and ``leiden`` store
    them), else what provenance recorded for the active clustering, else -1 (scanpy's
    default; also for a column scView did not make, which then follows the active one)."""
    active = adata.uns.get("scview_active_clustering")
    for col in (column, active):
        if not isinstance(col, str):
            continue
        entry = adata.uns.get(col)
        params = entry.get("params") if isinstance(entry, dict) else None
        n = _iterations(params.get("n_iterations")) if isinstance(params, dict) else None
        if n is not None:
            return n
    try:
        from scview.core import provenance

        current = provenance.read_provenance(adata)["current"].get("clustering") or {}
    except Exception:  # noqa: BLE001 — unreadable provenance: scanpy's default
        current = {}
    if not column or column == current.get("column"):
        n = _iterations(current.get("leiden_iterations"))
        if n is not None:
            return n
    return N_ITERATIONS


def _iterations(value: Any) -> int | None:
    """A recorded iteration count: -1, or a positive whole number; None otherwise."""
    try:
        n = int(np.asarray(value).ravel()[0])
    except (TypeError, ValueError, IndexError):
        return None
    return n if n == N_ITERATIONS or n > 0 else None


def membership(adjacency, resolution: float, random_state: int = RANDOM_STATE,
               n_iterations: int = N_ITERATIONS) -> np.ndarray:
    """Community of each cell, numbered by size (largest is 0). ``n_iterations``: -1
    repeats passes until one improves nothing (scanpy's default); a positive number stops
    after that many (fast Leiden's 2, or the merge check's quick count)."""
    # As scanpy's get_igraph_from_adjacency: every stored non-zero, in storage
    # order. The seeded result depends on edge order, so do not canonicalise.
    sources, targets = adjacency.nonzero()
    weights = np.asarray(adjacency[sources, targets]).ravel().astype(np.float64)
    payload = io.BytesIO()
    np.savez(payload, n=adjacency.shape[0], sources=sources, targets=targets, weights=weights,
             resolution=float(resolution), seed=int(random_state), n_iterations=int(n_iterations))
    # A pipeline run's cancel ends the child (core/run_watchdog.py). Imported here:
    # the child imports this module too and should not import more than it needs.
    from scview.core.run_watchdog import current

    control = current()
    proc = subprocess.Popen(
        [sys.executable, "-m", "scview.core.leiden"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    with control.child(proc) if control is not None else contextlib.nullcontext():
        stdout, stderr = proc.communicate(payload.getvalue())
    if proc.returncode != 0:
        if control is not None and control.cancelled:
            raise LeidenError("Leiden clustering was stopped: the run was cancelled.")
        tail = " ".join(stderr.decode(errors="replace").strip().splitlines()[-3:])
        raise LeidenError(f"Leiden clustering failed (exit {proc.returncode}): {tail}")
    return np.load(io.BytesIO(stdout))


def labels(adjacency, resolution: float, random_state: int = RANDOM_STATE,
           n_iterations: int = N_ITERATIONS) -> pd.Categorical:
    """Cluster labels as ``sc.tl.leiden`` stores them: strings, categories in numeric order."""
    groups = membership(adjacency, resolution, random_state, n_iterations)
    return pd.Categorical(values=groups.astype("U"), categories=[str(g) for g in np.unique(groups)])


def leiden(adata, resolution: float, key_added: str, *, random_state: int = RANDOM_STATE,
           n_iterations: int = N_ITERATIONS) -> None:
    """``sc.tl.leiden(adata, resolution=..., key_added=..., random_state=...,
    n_iterations=...)``, same outputs."""
    key = (adata.uns.get("neighbors") or {}).get("connectivities_key", "connectivities")
    if key not in adata.obsp:
        raise ValueError("No neighbour graph: compute neighbours before clustering.")
    adata.obs[key_added] = labels(adata.obsp[key], resolution, random_state, n_iterations)
    adata.uns[key_added] = {"params": {
        "resolution": resolution, "random_state": random_state, "n_iterations": n_iterations,
    }}


def _child() -> None:
    # igraph imports matplotlib.pyplot for its plotting (0.6 of the child's
    # 0.85 s of imports); it treats matplotlib as optional, so pretend it's absent.
    sys.modules["matplotlib"] = None
    import igraph as ig
    import leidenalg

    with np.load(io.BytesIO(sys.stdin.buffer.read())) as d:
        edges = np.column_stack([d["sources"], d["targets"]]).tolist()
        graph = ig.Graph(n=int(d["n"]), edges=edges, directed=True)
        part = leidenalg.find_partition(
            graph, leidenalg.RBConfigurationVertexPartition,
            weights=d["weights"], seed=int(d["seed"]),
            n_iterations=int(d["n_iterations"]) if "n_iterations" in d.files else N_ITERATIONS,
            resolution_parameter=float(d["resolution"]),
        )
    out = io.BytesIO()
    np.save(out, np.asarray(part.membership, dtype=np.int64))
    sys.stdout.buffer.write(out.getvalue())


if __name__ == "__main__":
    _child()
