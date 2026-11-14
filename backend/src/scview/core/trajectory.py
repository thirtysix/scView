"""Diffusion pseudotime and the PAGA group graph, computed in-app (scanpy
``diffmap`` + ``dpt``, ``paga``).

The work runs on a scratch AnnData that holds only a copy of the neighbour
graph, so the dataset gains the pseudotime columns (``DPT_COLUMN``, the latest
run, and one per root, ``root_column_name``) plus params records, or one PAGA
record (``PAGA_KEY``): its own neighbour graph, any ``X_diffmap``,
``uns['paga']`` and any imported ``dpt_pseudotime`` column are left as they were.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from scview.core import provenance, results_store

logger = logging.getLogger(__name__)

DPT_COLUMN = "scview_dpt_pseudotime"  # the latest run
DPT_PREFIX = "scview_dpt_"  # + the root group: each root's run kept (root_column_name)
RUNS_KEY = "scview_dpt_runs"  # uns: JSON {column: params} of each root's column
PARAMS_KEY = "scview_dpt_params"  # uns: JSON record of how DPT_COLUMN was made
PAGA_KEY = "scview_paga"  # uns: JSON record of the last PAGA graph (compute_paga)

N_NEIGHBORS = 15  # only when a graph has to be built (the pipeline's default)
DIFFMAP_COMPS = 15  # scanpy's default
N_DCS = 10  # diffusion components DPT uses (scanpy's default)
MIN_CELLS = 30
MAX_PAGA_GROUPS = 100  # as many as a root-group picker lists
DETACH_MAX_SHARE = 0.05  # a weakly linked tail left out of DPT holds at most this share of the cells
DETACH_ROUNDS = 2  # recomputations after leaving one out (a second island can show then)
REVERSED_RHO = -0.5  # a pseudotime this much against another one: a root warning


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


class _Cells:
    """An AnnData seen as an adaptor by ``results_store.fingerprint`` (the same hash of the
    cells on both sides of a comparison)."""

    def __init__(self, adata):
        self.adata = adata

    def cells_fingerprint(self) -> str:
        return results_store.cells_hash(self.adata)


def graph_fingerprint(adata) -> dict[str, Any] | None:
    """What a PAGA graph or a pseudotime was computed on: the cells and the recipe's steps."""
    return results_store.fingerprint(_Cells(adata))


def graph_stale(adata, record: dict[str, Any] | None) -> str | None:
    """Why a PAGA or pseudotime record no longer matches the dataset's graph, else None:
    a step that changed the cells, the data, the PCs or the graph since (as the results
    list words it), or, for a record from before fingerprints, another graph now."""
    if not record:
        return None
    if record.get("fingerprint"):
        why = results_store.graph_read_stale(_Cells(adata), record["fingerprint"])
        return f"{why[0].upper()}{why[1:]}." if why else None
    try:
        now = _graph_label(adata, graph_source(adata))
    except TrajectoryError:
        return "The neighbour graph it was computed on is gone."
    then = record.get("graph")
    return f"Computed on the {then}; the dataset now has the {now}." if then and then != now else None


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


def detached_tail(pt: np.ndarray, max_share: float = DETACH_MAX_SHARE) -> np.ndarray | None:
    """The cells past an extreme gap at the far end of a pseudotime, or None.

    A few cells joined to the rest by weak links (a near-island of the neighbour graph)
    get a diffusion component of their own with an eigenvalue near 1, and DPT's weighting,
    lambda / (1 - lambda), makes their distance swamp everyone else's. scanpy's ``dpt``
    then scales by the maximum, as here before: on Pijuan-Sala's erythroid cells after
    Harmony, 18 cells took 0.92-1.0 and the other 9,797 were squeezed into 0-0.016.

    A tail counts when it holds at most ``max_share`` of the cells, and the gap before it
    is wider than the whole range of the cells below it and than a quarter of the full
    range. A long but continuous tail (paul15's dendritic cells) has no such gap."""
    pt = np.asarray(pt, dtype=np.float64)
    n = len(pt)
    if n < MIN_CELLS:
        return None
    order = np.argsort(pt, kind="stable")
    s = pt[order]
    span = s[-1] - s[0]
    if not np.isfinite(span) or span <= 0:
        return None
    lo = n - 1 - max(1, int(n * max_share))  # the gap after s[i]: i >= lo keeps >= 1 - share below
    gaps = np.diff(s)[lo:]
    i = lo + int(np.argmax(gaps))
    gap, below = s[i + 1] - s[i], s[i] - s[0]
    if gap <= below or gap <= 0.25 * span:
        return None
    tail = np.zeros(n, dtype=bool)
    tail[order[i + 1:]] = True
    return tail


def _diffuse(work: ad.AnnData, reach: np.ndarray, in_group: np.ndarray) -> tuple[np.ndarray, int]:
    """DPT over the ``reach`` cells from the root group's most central cell: (pseudotime
    of those cells, the root's index among them)."""
    import scanpy as sc

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
    return sub.obs["dpt_pseudotime"].to_numpy(dtype=np.float64), sub_root


def _tree_neighbours(record: dict[str, Any], group: str) -> list[str]:
    groups = record.get("groups") or []
    if group not in groups:
        return []
    g = groups.index(group)
    return [groups[j if i == g else i] for i, j in record.get("tree") or [] if g in (i, j)]


PSEUDOTIME_WORDS = ("pseudotime", "dpt_", "latent_time", "monocle")  # = AnnDataAdaptor.available_pseudotime_columns


def pseudotime_columns(adata) -> list[str]:
    return [str(c) for c in adata.obs.columns
            if any(w in str(c).lower() for w in PSEUDOTIME_WORDS) and pd.api.types.is_numeric_dtype(adata.obs[c])]


def root_column_name(root_value: str) -> str:
    """The obs column a run from this root also writes (one per root, kept beside the
    latest in ``DPT_COLUMN``)."""
    slug = re.sub(r"[^0-9A-Za-z]+", "_", str(root_value)).strip("_")[:40] or "root"
    return f"{DPT_PREFIX}{slug}"


def against_other_pseudotimes(adata, pt: np.ndarray, skip: set[str]) -> tuple[str, float, int] | None:
    """The other pseudotime column this one runs most against: (column, Spearman rho, n
    cells), when rho <= ``REVERSED_RHO``; else None."""
    from scipy.stats import spearmanr

    worst = None
    for c in pseudotime_columns(adata):
        if c in skip:
            continue
        x = pd.to_numeric(adata.obs[c], errors="coerce").to_numpy(dtype=np.float64)
        ok = np.isfinite(pt) & np.isfinite(x)
        if ok.sum() < MIN_CELLS or np.ptp(x[ok]) == 0 or np.ptp(pt[ok]) == 0:
            continue
        rho = float(spearmanr(pt[ok], x[ok]).statistic)
        if rho <= REVERSED_RHO and (worst is None or rho < worst[1]):
            worst = (c, rho, int(ok.sum()))
    return worst


def root_warnings(adata, pt: np.ndarray, root_column: str, root_value: str) -> list[str]:
    """Signs that the root group is not where the process starts, kept simple: it sits
    inside the saved PAGA tree of its own column (pseudotime then runs out several
    ways), or this pseudotime runs against another one the dataset holds (an earlier
    run from another root, an imported ``latent_time``). How many genes cells detect
    (CytoTRACE's signal) was tried and dropped: paul15's MEP root detects the fewest
    (323 median against 1,175 in erythroid cells), so it flagged the right root."""
    out: list[str] = []
    record = read_paga(adata)
    if record and record.get("groupby") == root_column:
        nbrs = _tree_neighbours(record, root_value)
        if len(nbrs) >= 2:
            shown = ", ".join(f"'{n}'" for n in nbrs[:3]) + (" and more" if len(nbrs) > 3 else "")
            out.append(
                f"'{root_value}' sits inside the PAGA tree of {root_column} (linked to {shown}), not at an end, "
                "so pseudotime runs out from it in several directions. Right for a progenitor between lineages; "
                "otherwise the start state at an end of the tree is the usual root."
            )
    other = against_other_pseudotimes(adata, pt, {DPT_COLUMN, root_column_name(root_value)})
    if other is not None:
        col, rho, n = other
        out.append(
            f"This pseudotime runs against '{col}' (Spearman rho {rho:.2f}, n = {n:,} cells): one of the two "
            "starts at the wrong end. Check which group the process begins in."
        )
    return out


def _fmt(v: float) -> str:
    return f"{v:.2g}" if v < 0.1 else f"{v:.2f}"


def compute_dpt(adata, root_column: str, root_value: str) -> dict[str, Any]:
    """Diffusion pseudotime from a root group; writes ``DPT_COLUMN`` into ``adata``.

    Cells the root cannot reach through the graph get ``inf``. Cells past an extreme gap
    (``detached_tail``: a weakly linked island) are left out with NaN and the pseudotime
    is computed again without them, so they do not set its scale. Returns the parameters
    (also recorded in ``uns`` and the provenance history), with ``warnings``.
    """
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
    n_reach = int(reach.sum())
    detached = np.zeros(adata.n_obs, dtype=bool)
    first_gap: tuple[float, float, float] | None = None  # (main max, tail min, tail max), first round
    for round_ in range(DETACH_ROUNDS + 1):
        sub_pt, sub_root = _diffuse(work, reach, in_group)
        tail = None if round_ == DETACH_ROUNDS else detached_tail(sub_pt)
        if tail is None or int(reach.sum() - tail.sum()) < MIN_CELLS or not in_group[reach][~tail].any():
            break
        if first_gap is None:
            first_gap = (float(sub_pt[~tail].max()), float(sub_pt[tail].min()), float(sub_pt[tail].max()))
        idx = np.flatnonzero(reach)[tail]
        detached[idx] = True
        reach[idx] = False
    pt = np.full(adata.n_obs, np.inf)
    pt[detached] = np.nan
    pt[reach] = sub_pt
    iroot = int(np.flatnonzero(reach)[sub_root])

    warnings: list[str] = []
    n_detached = int(detached.sum())
    if n_detached and first_gap is not None:
        main_max, tail_lo, tail_hi = first_gap
        warnings.append(
            f"{n_detached:,} cells hang on the rest of the graph by weak links: from this root they came out at "
            f"{_fmt(tail_lo)}-{_fmt(tail_hi)} while the other {n_reach - n_detached:,} took 0-{_fmt(main_max)}. "
            "They are left out (grey), and the pseudotime was computed again on the main part."
        )
    warnings.extend(root_warnings(adata, pt, root_column, root_value))

    named = root_column_name(root_value)
    params: dict[str, Any] = {
        "column": DPT_COLUMN,
        "root_named_column": named,  # the same values, kept when another root runs
        "root_column": root_column,
        "root_value": root_value,
        "root_cell": str(adata.obs_names[iroot]),
        "root_rule": "most central root-group cell in DPT-weighted diffusion space",
        "graph": _graph_label(adata, source),
        "diffmap_n_comps": DIFFMAP_COMPS,
        "n_dcs": N_DCS,
        "n_unreachable": int(np.isinf(pt).sum()),
        "n_detached": n_detached,
        "warnings": warnings,
    }
    adata.obs[DPT_COLUMN] = pt
    adata.obs[named] = pt
    recorded = dict(params)
    params["fingerprint"] = graph_fingerprint(adata)  # for graph_stale; not in the recipe
    adata.uns[PARAMS_KEY] = json.dumps(params)
    runs = {c: p for c, p in (read_runs(adata) or {}).items() if c in adata.obs.columns}
    runs[named] = params
    adata.uns[RUNS_KEY] = json.dumps(runs)
    try:
        provenance.record_step(adata, step="pseudotime", tool="scanpy.tl.dpt", params=recorded)
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


def public_params(adata, params: dict[str, Any] | None) -> dict[str, Any] | None:
    """A pseudotime's params as the API gives them: ``stale`` (``graph_stale``) instead of
    the fingerprint it is worked out from."""
    if params is None:
        return None
    out = {k: v for k, v in params.items() if k != "fingerprint"}
    out["stale"] = graph_stale(adata, params)
    return out


def read_runs(adata) -> dict[str, Any] | None:
    """Each root's column (``root_column_name``) with its params, or None."""
    return _read_json(adata, RUNS_KEY)


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
        "fingerprint": graph_fingerprint(adata),  # for graph_stale
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
    sizes changed since) or the graph it was computed on (``graph_stale``), else None. ``root_group``: the group holding the
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
    else:
        why = graph_stale(adata, record)
        out["stale"] = f"{why} Compute PAGA again." if why else None
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
