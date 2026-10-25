"""Diffusion pseudotime and the PAGA group graph, computed in-app (scanpy
``diffmap`` + ``dpt``, ``paga``).

The work runs on a scratch AnnData that holds only a copy of the neighbour
graph, so the dataset gains exactly one obs column (``DPT_COLUMN``) plus a
params record, or one PAGA record (``PAGA_KEY``): its own neighbour graph, any
``X_diffmap``, ``uns['paga']`` and any imported ``dpt_pseudotime`` column are
left as they were.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from scview.core import provenance

logger = logging.getLogger(__name__)

DPT_COLUMN = "scview_dpt_pseudotime"
PARAMS_KEY = "scview_dpt_params"  # uns: JSON record of how DPT_COLUMN was made
PAGA_KEY = "scview_paga"  # uns: JSON record of the last PAGA graph (compute_paga)

N_NEIGHBORS = 15  # only when a graph has to be built (the pipeline's default)
DIFFMAP_COMPS = 15  # scanpy's default
N_DCS = 10  # diffusion components DPT uses (scanpy's default)
MIN_CELLS = 30
MAX_PAGA_GROUPS = 100  # as many as a root-group picker lists


class TrajectoryError(ValueError):
    """A pseudotime request that cannot be served (no PCA, bad root group): a 400."""


def graph_source(adata) -> str:
    """Where the neighbour graph will come from; raises when there is none to use."""
    if "neighbors" in adata.uns and {"connectivities", "distances"} <= set(adata.obsp.keys()):
        return "neighbors"
    for rep in ("X_pca_harmony", "X_scanorama", "X_pca"):
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


def _graph_label(adata, source: str) -> str:
    """Which graph a result was computed on, for its params record."""
    if source != "neighbors":
        return f"kNN graph on {source} (k={N_NEIGHBORS})"
    nb_params = (adata.uns.get("neighbors") or {}).get("params", {})
    return (
        f"existing neighbour graph (use_rep={nb_params.get('use_rep', 'default')}, "
        f"k={nb_params.get('n_neighbors', '?')})"
    )


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

    params: dict[str, Any] = {
        "column": DPT_COLUMN,
        "root_column": root_column,
        "root_value": root_value,
        "root_cell": str(adata.obs_names[iroot]),
        "root_rule": "most central root-group cell in DPT-weighted diffusion space",
        "graph": _graph_label(adata, source),
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


def _read_json(adata, key: str) -> dict[str, Any] | None:
    raw = getattr(adata, "uns", {}).get(key)
    try:
        return json.loads(raw) if isinstance(raw, str) else None
    except ValueError:
        return None


def read_params(adata) -> dict[str, Any] | None:
    """The params record for ``DPT_COLUMN``, or None (absent or malformed)."""
    return _read_json(adata, PARAMS_KEY)


def read_paga(adata) -> dict[str, Any] | None:
    """The saved PAGA record (``compute_paga``), or None (absent or malformed)."""
    return _read_json(adata, PAGA_KEY)


def _group_labels(adata, column: str) -> tuple[list[str], pd.Series]:
    """The groups of ``column`` (in its own order when categorical) and each
    cell's label as a string (NaN when it has none)."""
    col = adata.obs[column]
    labels = col.astype(str).where(col.notna())
    present = set(labels.dropna())
    if isinstance(col.dtype, pd.CategoricalDtype):
        order = [str(c) for c in col.cat.categories if str(c) in present]
    else:
        order = sorted(present)
    return order, labels


def compute_paga(adata, groupby: str) -> dict[str, Any]:
    """PAGA connectivity between the groups of ``groupby`` (Wolf et al. 2019,
    scanpy's model v1.2), on a copy of the neighbour graph; saved as ``uns[PAGA_KEY]``.

    A connectivity compares the edges between two groups with what random
    wiring of the same cells would give, capped at 1. Cells without a label are
    left out. Returns the record: the groups in order, their sizes, every edge
    with a non-zero connectivity (``[i, j, w]``, i < j) and the edges of PAGA's
    maximum spanning tree (``[i, j]``).
    """
    import scanpy as sc

    if groupby not in adata.obs.columns:
        raise TrajectoryError(f"Column '{groupby}' not found.")
    if pd.api.types.is_float_dtype(adata.obs[groupby]):
        raise TrajectoryError(f"'{groupby}' holds continuous values, not groups.")
    order, labels = _group_labels(adata, groupby)
    if len(order) < 2:
        raise TrajectoryError(f"PAGA needs at least two groups; '{groupby}' has {len(order)}.")
    if len(order) > MAX_PAGA_GROUPS:
        raise TrajectoryError(
            f"'{groupby}' has {len(order)} groups; PAGA here takes at most {MAX_PAGA_GROUPS}."
        )
    labelled = labels.notna().to_numpy()
    if labelled.sum() < MIN_CELLS:
        raise TrajectoryError(f"PAGA needs at least {MIN_CELLS} labelled cells.")

    source = graph_source(adata)
    # A copy: the stored graph, and any imported uns['paga'] (scanpy writes it), stay put.
    work = _scratch_with_graph(adata, source)
    if not labelled.all():
        work = work[labelled].copy()
    work.obs["group"] = pd.Categorical(labels[labelled].to_numpy(), categories=order)
    sc.tl.paga(work, groups="group")

    conn = sp.triu(sp.csr_matrix(work.uns["paga"]["connectivities"]), k=1).tocoo()
    edges = sorted(
        [int(i), int(j), round(float(w), 4)] for i, j, w in zip(conn.row, conn.col, conn.data)
    )
    tree = sp.coo_matrix(work.uns["paga"]["connectivities_tree"])
    record: dict[str, Any] = {
        "groupby": groupby,
        "groups": order,
        "sizes": np.bincount(work.obs["group"].cat.codes, minlength=len(order)).tolist(),
        "edges": [e for e in edges if e[2] > 0],
        "tree": sorted({(min(int(i), int(j)), max(int(i), int(j))) for i, j in zip(tree.row, tree.col)}),
        "graph": _graph_label(adata, source),
        "model": "v1.2",
        "n_cells": int(labelled.sum()),
    }
    adata.uns[PAGA_KEY] = json.dumps(record)
    record = json.loads(adata.uns[PAGA_KEY])  # tuples -> lists, as read back later
    try:
        provenance.record_step(
            adata, step="paga", tool="scanpy.tl.paga",
            params={"groupby": groupby, "n_groups": len(order), "graph": record["graph"]},
        )
    except Exception as e:  # provenance must never block the result
        logger.warning("Could not record PAGA provenance: %s", e)
    return record


def paga_view(adata, embedding: str | None = None) -> dict[str, Any]:
    """The saved PAGA record, with what the panel draws it with.

    ``positions``: each group's median position on ``embedding`` (x, y, plus z
    for a 3-D embedding, as the scatter draws it; None for a group with no
    cells). ``stale``: why the record no longer matches its grouping (groups or
    sizes changed since), else None. ``root_group``: the group holding the
    pseudotime root cell, where a branch starts. Raises ``KeyError`` for an
    unknown embedding.
    """
    record = read_paga(adata)
    out: dict[str, Any] = {"paga": record, "positions": None, "stale": None, "root_group": None}
    if record is None:
        return out
    groupby, groups = record["groupby"], record["groups"]
    if groupby not in adata.obs.columns:
        out["stale"] = f"The column '{groupby}' is gone; compute PAGA again."
        return out
    order, labels = _group_labels(adata, groupby)
    codes = pd.Index(groups).get_indexer(labels)  # -1: no label, or a new group
    sizes = np.bincount(codes[codes >= 0], minlength=len(groups)).tolist()
    if order != groups or sizes != record["sizes"]:
        out["stale"] = f"'{groupby}' has changed since PAGA was computed; compute it again."
    if embedding:
        if embedding not in adata.obsm:
            raise KeyError(f"Embedding '{embedding}' not found.")
        X = np.asarray(adata.obsm[embedding], dtype=np.float64)
        X = X[:, :3] if X.shape[1] == 3 else X[:, :2]
        positions: list[list[float] | None] = []
        for g in range(len(groups)):
            pts = X[codes == g]
            pts = pts[np.isfinite(pts).all(axis=1)]
            positions.append(np.round(np.median(pts, axis=0), 4).tolist() if len(pts) else None)
        out["positions"] = positions
    root = (read_params(adata) or {}).get("root_cell")
    hit = np.flatnonzero(adata.obs_names == root) if root else []
    if len(hit) and codes[hit[0]] >= 0:
        out["root_group"] = groups[codes[hit[0]]]
    return out
