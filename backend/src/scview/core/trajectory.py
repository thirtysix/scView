"""Diffusion pseudotime computed in-app (scanpy ``diffmap`` + ``dpt``).

The work runs on a scratch AnnData that holds only the neighbour graph, so the
dataset gains exactly one obs column (``DPT_COLUMN``) plus a params record: its
own neighbour graph, any ``X_diffmap`` and any imported ``dpt_pseudotime``
column are left as they were.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components

from scview.core import provenance

logger = logging.getLogger(__name__)

DPT_COLUMN = "scview_dpt_pseudotime"
PARAMS_KEY = "scview_dpt_params"  # uns: JSON record of how DPT_COLUMN was made

N_NEIGHBORS = 15  # only when a graph has to be built (the pipeline's default)
DIFFMAP_COMPS = 15  # scanpy's default
N_DCS = 10  # diffusion components DPT uses (scanpy's default)
MIN_CELLS = 30


class TrajectoryError(ValueError):
    """A pseudotime request that cannot be served (no PCA, bad root group): a 400."""


def graph_source(adata) -> str:
    """Where the neighbour graph will come from; raises when there is none to use."""
    if "neighbors" in adata.uns and {"connectivities", "distances"} <= set(adata.obsp.keys()):
        return "neighbors"
    for rep in ("X_pca_harmony", "X_pca"):
        if rep in adata.obsm:
            return rep
    raise TrajectoryError(
        "Pseudotime needs a neighbour graph or a PCA, and this dataset has neither. "
        "Run the processing pipeline (through PCA) first."
    )


def _scratch_with_graph(adata, source: str) -> ad.AnnData:
    """A cells-only AnnData carrying the dataset's neighbour graph, or one built
    from ``source`` (an obsm representation) with the pipeline's settings."""
    import scanpy as sc

    work = ad.AnnData(obs=pd.DataFrame(index=adata.obs_names.copy()))
    if source == "neighbors":
        nb = dict(adata.uns["neighbors"])
        work.obsp["connectivities"] = adata.obsp["connectivities"].copy()
        work.obsp["distances"] = adata.obsp["distances"].copy()
        nb["connectivities_key"] = "connectivities"
        nb["distances_key"] = "distances"
        work.uns["neighbors"] = nb
    else:
        work.obsm[source] = np.asarray(adata.obsm[source])
        sc.pp.neighbors(work, n_neighbors=N_NEIGHBORS, use_rep=source)
    return work


def root_cell(dcs: np.ndarray, evals: np.ndarray, in_group: np.ndarray) -> int:
    """Index of the root cell: the root group's most central cell in diffusion space.

    ``dcs`` are the non-trivial diffusion components (cells x k) and ``evals``
    their eigenvalues. Distances use DPT's own weighting of each component,
    lambda / (1 - lambda) (Haghverdi et al. 2016), and the root is the group cell
    nearest the group's median there, so the whole group sits at the start and
    pseudotime grows with diffusion distance from it.

    The outermost group cell on DC1 was tried first and rejected: a start state
    often sits at a branch point rather than a tip (paul15's MEP progenitors lie
    between the erythroid and myeloid branches), and there DC1's extreme is a
    cell already part-way down one branch, which put that branch's mature cells
    before the progenitors themselves.
    """
    lam = np.clip(evals, None, 1 - 1e-9)
    y = dcs * (lam / (1 - lam))
    idx = np.flatnonzero(in_group)
    g = y[idx]
    return int(idx[np.argmin(((g - np.median(g, axis=0)) ** 2).sum(axis=1))])


def compute_dpt(adata, root_column: str, root_value: str) -> dict[str, Any]:
    """Diffusion pseudotime from a root group; writes ``DPT_COLUMN`` into ``adata``.

    Cells the root cannot reach through the graph get ``inf``. Returns the
    parameters (also recorded in ``uns`` and the provenance history).
    """
    import scanpy as sc

    if adata.n_obs < MIN_CELLS:
        raise TrajectoryError(f"Pseudotime needs at least {MIN_CELLS} cells.")
    if root_column not in adata.obs.columns:
        raise TrajectoryError(f"Column '{root_column}' not found.")
    col = adata.obs[root_column]
    in_group = (col.astype(str).where(col.notna()) == root_value).to_numpy()
    if not in_group.any():
        raise TrajectoryError(f"No cells have {root_column} = '{root_value}'.")

    source = graph_source(adata)
    work = _scratch_with_graph(adata, source)

    # Diffuse only within the connected part of the graph that holds the root
    # group. Elsewhere no path from the root exists (pseudotime inf, as DPT
    # itself reports), and each extra component would add an indicator
    # component with eigenvalue 1 that swamps the distances used to pick the root.
    _, comp = connected_components(work.obsp["connectivities"], directed=False)
    reach = comp == np.bincount(comp[in_group]).argmax()
    if reach.sum() < MIN_CELLS:
        raise TrajectoryError(
            f"The '{root_value}' cells sit in a disconnected part of the neighbour graph "
            f"with only {int(reach.sum())} cells; pick a root group in the main data."
        )
    sub = work[reach].copy()
    sc.tl.diffmap(sub, n_comps=DIFFMAP_COMPS)
    # Column 0 of X_diffmap is the trivial stationary component; DPT uses the
    # next ones, up to n_dcs, and so does the root choice.
    dcs = slice(1, N_DCS)
    sub_root = root_cell(
        np.asarray(sub.obsm["X_diffmap"][:, dcs]),
        np.asarray(sub.uns["diffmap_evals"][dcs]),
        in_group[reach],
    )
    sub.uns["iroot"] = sub_root
    sc.tl.dpt(sub, n_dcs=N_DCS)
    pt = np.full(adata.n_obs, np.inf)
    pt[reach] = sub.obs["dpt_pseudotime"].to_numpy(dtype=np.float64)
    iroot = int(np.flatnonzero(reach)[sub_root])

    nb_params = (adata.uns.get("neighbors") or {}).get("params", {}) if source == "neighbors" else {}
    params: dict[str, Any] = {
        "column": DPT_COLUMN,
        "root_column": root_column,
        "root_value": root_value,
        "root_cell": str(adata.obs_names[iroot]),
        "root_rule": "most central root-group cell in DPT-weighted diffusion space",
        "graph": (
            f"existing neighbour graph (use_rep={nb_params.get('use_rep', 'default')}, "
            f"k={nb_params.get('n_neighbors', '?')})"
            if source == "neighbors" else f"kNN graph on {source} (k={N_NEIGHBORS})"
        ),
        "diffmap_n_comps": DIFFMAP_COMPS,
        "n_dcs": N_DCS,
        "n_unreachable": int((~np.isfinite(pt)).sum()),
    }
    adata.obs[DPT_COLUMN] = pt
    adata.uns[PARAMS_KEY] = json.dumps(params)
    try:
        provenance.record_step(adata, step="pseudotime", tool="scanpy.tl.dpt", params=params)
    except Exception as e:  # provenance must never block the result
        logger.warning("Could not record pseudotime provenance: %s", e)
    return params


def read_params(adata) -> dict[str, Any] | None:
    """The params record for ``DPT_COLUMN``, or None (absent or malformed)."""
    raw = getattr(adata, "uns", {}).get(PARAMS_KEY)
    try:
        return json.loads(raw) if isinstance(raw, str) else None
    except ValueError:
        return None
