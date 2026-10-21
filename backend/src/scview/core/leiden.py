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
"""

from __future__ import annotations

import contextlib
import io
import subprocess
import sys

import numpy as np
import pandas as pd

RANDOM_STATE = 0  # sc.tl.leiden's default
N_ITERATIONS = -1  # sc.tl.leiden's default: until a pass improves nothing


class LeidenError(RuntimeError):
    """The child process failed."""


def membership(adjacency, resolution: float, random_state: int = RANDOM_STATE,
               n_iterations: int = N_ITERATIONS) -> np.ndarray:
    """Community of each cell, numbered by size (largest is 0). ``n_iterations``
    other than scanpy's -1 is for a quick look (a cluster count), not for labels
    anyone keeps."""
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


def labels(adjacency, resolution: float, random_state: int = RANDOM_STATE) -> pd.Categorical:
    """Cluster labels as ``sc.tl.leiden`` stores them: strings, categories in numeric order."""
    groups = membership(adjacency, resolution, random_state)
    return pd.Categorical(values=groups.astype("U"), categories=[str(g) for g in np.unique(groups)])


def leiden(adata, resolution: float, key_added: str, *, random_state: int = RANDOM_STATE) -> None:
    """``sc.tl.leiden(adata, resolution=..., key_added=..., random_state=...)``, same outputs."""
    key = (adata.uns.get("neighbors") or {}).get("connectivities_key", "connectivities")
    if key not in adata.obsp:
        raise ValueError("No neighbour graph: compute neighbours before clustering.")
    adata.obs[key_added] = labels(adata.obsp[key], resolution, random_state)
    adata.uns[key_added] = {"params": {
        "resolution": resolution, "random_state": random_state, "n_iterations": N_ITERATIONS,
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
