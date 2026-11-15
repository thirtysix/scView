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
CELL_ROOT_PREFIX = "cell_"  # after DPT_PREFIX: a single-cell root's own column
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
# Root checks (``root_checks``), measured on Bastidas-Ponce's pancreas, Pijuan-Sala's erythroid
# cells and paul15 with their authors' labels (true root and every other group as a root):
BRANCH_SHARE = 0.05  # a branch out of the root group holding at least this share of PAGA's cells ...
BRANCH_POINT = 3  # ... this many of them: a branch point between lineages (paul15's 7MEP: 41/40/8%)
TOP_GENES = 20  # "specialised": the share of a cell's counts in its 20 most-counted genes
LESS_SPECIALISED_GENES = 1.1  # another tree end detects at least this many times the genes ...
LESS_SPECIALISED_TOP = 0.02  # ... or its top-20 share is this much lower (both must point its way)
POTENCY_SAMPLE = 2000  # cells per group read for those numbers (seeded)
ROOT_CHOICE = ("Give either root_column and root_value (a start group: its most central cell is the root) "
               "or root_cell (one cell's name), not both.")


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


def root_column_name(root_value: str | None = None, root_cell: str | None = None) -> str:
    """The obs column a run from this root also writes (one per root, kept beside the
    latest in ``DPT_COLUMN``): ``scview_dpt_<group>``, or ``scview_dpt_cell_<cell>`` for
    a single-cell root."""
    if root_cell is not None:
        slug = re.sub(r"[^0-9A-Za-z]+", "_", str(root_cell)).strip("_")[:40] or "root"
        return f"{DPT_PREFIX}{CELL_ROOT_PREFIX}{slug}"
    slug = re.sub(r"[^0-9A-Za-z]+", "_", str(root_value)).strip("_")[:40] or "root"
    if slug.startswith(CELL_ROOT_PREFIX):  # a group called "cell ...": never a cell root's column
        slug = f"group_{slug}"
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


def root_group_in(adata, column: str, iroot: int) -> str | None:
    """The root cell's group in ``column`` (str), or None when it has none."""
    if column not in adata.obs.columns:
        return None
    v = adata.obs[column].iloc[iroot]
    return None if pd.isna(v) else str(v)


def paga_stale(adata, record: dict[str, Any] | None) -> str | None:
    """Why a saved PAGA record no longer fits the dataset (its column gone or changed, or
    its graph replaced), else None."""
    if not record:
        return None
    groupby, groups = record["groupby"], record["groups"]
    if groupby not in adata.obs.columns:
        return f"The column '{groupby}' is gone; compute PAGA again."
    order, labels = _group_labels(adata, groupby)
    codes = pd.Index(groups).get_indexer(labels)  # -1: no label, or a new group
    sizes = np.bincount(codes[codes >= 0], minlength=len(groups)).tolist()
    if order != groups or sizes != record["sizes"]:
        return f"'{groupby}' has changed since PAGA was computed; compute it again."
    why = graph_stale(adata, record)
    return f"{why} Compute PAGA again." if why else None


def _tree_adjacency(record: dict[str, Any]) -> dict[int, set[int]]:
    adj: dict[int, set[int]] = {i: set() for i in range(len(record.get("groups") or []))}
    for i, j in record.get("tree") or []:
        adj[int(i)].add(int(j))
        adj[int(j)].add(int(i))
    return adj


def root_position(record: dict[str, Any], group: str) -> dict[str, Any] | None:
    """Where a group sits in the PAGA tree: "end" (one tree edge), "inside" or "branch point"
    (at least ``BRANCH_POINT`` branches out of it each hold ``BRANCH_SHARE`` of the cells), with
    each branch's share of the cells, largest first; None for a group not in the tree."""
    groups = record.get("groups") or []
    if group not in groups:
        return None
    g = groups.index(group)
    adj = _tree_adjacency(record)
    sizes = [float(x) for x in record.get("sizes") or [1] * len(groups)]
    total = sum(sizes) or 1.0
    shares = []
    for nb in sorted(adj[g]):
        seen, stack = {g, nb}, [nb]
        while stack:
            for y in adj[stack.pop()]:
                if y not in seen:
                    seen.add(y)
                    stack.append(y)
        shares.append(round(sum(sizes[k] for k in seen - {g}) / total, 3))
    shares.sort(reverse=True)
    if len(shares) <= 1:
        position = "end" if shares else "alone"
    elif sum(x >= BRANCH_SHARE for x in shares) >= BRANCH_POINT:
        position = "branch point"
    else:
        position = "inside"
    return {"position": position, "branches": shares}


def tied_links(record: dict[str, Any], group: str) -> list[str]:
    """The close ties (``tree_ties``) that touch ``group``, "kept over instead_of" in words:
    empty when PAGA settles the tree around it."""
    groups = record.get("groups") or []
    if group not in groups:
        return []
    g = groups.index(group)

    def name(e) -> str:
        return f"{groups[e[0]]} - {groups[e[1]]}"

    return [f"{name(t['kept'])} over {name(t['instead_of'])}" for t in record.get("ties") or []
            if t.get("close") and g in (*t["kept"], *t["instead_of"])]


def tree_ends(record: dict[str, Any]) -> list[str]:
    adj = _tree_adjacency(record)
    return [g for i, g in enumerate(record.get("groups") or []) if len(adj[i]) == 1]


def _row_stats(M, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per cell: genes detected, and the share of its counts in its ``TOP_GENES`` genes."""
    block = M[np.sort(rows)]
    if sp.issparse(block):
        block = sp.csr_matrix(block)
        genes = np.diff(block.indptr)
        top = np.zeros(block.shape[0])
        for r in range(block.shape[0]):
            d = block.data[block.indptr[r]:block.indptr[r + 1]]
            total = d.sum()
            if total > 0:
                top[r] = (np.partition(d, -TOP_GENES)[-TOP_GENES:].sum() if d.size > TOP_GENES else total) / total
        return genes.astype(np.float64), top
    block = np.asarray(block, dtype=np.float64)
    genes = (block > 0).sum(axis=1).astype(np.float64)
    total = block.sum(axis=1)
    k = min(TOP_GENES, block.shape[1])
    top_sum = np.partition(block, -k, axis=1)[:, -k:].sum(axis=1)
    return genes, np.divide(top_sum, total, out=np.zeros_like(total), where=total > 0)


def specialisation(adata, column: str, groups: list[str]) -> dict[str, dict[str, Any]] | None:
    """Per group (at least ``MIN_CELLS`` cells): median genes detected per cell and median
    share of a cell's counts in its 20 most-counted genes, on raw counts (at most
    ``POTENCY_SAMPLE`` cells per group, seeded); None without counts."""
    from scview.core.normalization import find_counts

    try:
        M, _ = find_counts(adata, read_only=True)
    except Exception:  # no counts (a light large dataset has no matrix in hand)
        return None
    labels = adata.obs[column]
    labels = labels.astype(str).where(labels.notna()).to_numpy()
    rng = np.random.default_rng(0)
    out: dict[str, dict[str, Any]] = {}
    for g in groups:
        idx = np.flatnonzero(labels == g)
        if len(idx) < MIN_CELLS:
            continue
        if len(idx) > POTENCY_SAMPLE:
            idx = rng.choice(idx, POTENCY_SAMPLE, replace=False)
        try:
            genes, top = _row_stats(M, idx)
        except Exception as e:  # a counts file that cannot be read this way: no check, never a failed run
            logger.warning("Root check: counts not read (%s)", e)
            return None
        out[g] = {"n_cells": int((labels == g).sum()), "genes": float(np.median(genes)),
                  "top_share": round(float(np.median(top)), 4)}
    return out


def less_specialised_end(stats: dict[str, dict[str, Any]], root: str, ends: list[str]) -> str | None:
    """The other tree end that looks least specialised against the root: more genes detected
    AND a lower top-20 share, clearly so on at least one (``LESS_SPECIALISED_*``); else None."""
    r = stats.get(root)
    if r is None:
        return None
    best, best_key = None, None
    for e in ends:
        o = stats.get(e)
        if e == root or o is None or o["genes"] <= r["genes"] or o["top_share"] >= r["top_share"]:
            continue
        if o["genes"] >= LESS_SPECIALISED_GENES * r["genes"] or o["top_share"] <= r["top_share"] - LESS_SPECIALISED_TOP:
            key = (o["genes"] / max(r["genes"], 1.0), r["top_share"] - o["top_share"])
            if best_key is None or key > best_key:
                best, best_key = e, key
    return best


def velocity_against_root(direction: dict[str, Any] | None, root: str) -> dict[str, Any] | None:
    """The velocity check from the root along the PAGA tree, read as a root verdict: arrows run
    against the root when a transition out of the root itself is backward (the arrows flow into
    it) or backward transitions outnumber forward ones. None when unchecked."""
    if not direction or not direction.get("checked"):
        return None
    rows = direction.get("transitions") or []
    back = [r for r in rows if r.get("status") == "backward"]
    own = [r for r in back if r.get("from") == root]
    n_fwd, n_back = int(direction.get("n_forward") or 0), len(back)
    return {"n_forward": n_fwd, "n_backward": n_back, "root_edges_backward": [r["to"] for r in own],
            "against": bool(own) or n_back > n_fwd, "embedding": direction.get("embedding"),
            "backward": [{k: r.get(k) for k in ("from", "to", "mean_cos", "n_cells")} for r in back]}


def root_warnings(adata, pt: np.ndarray, iroot: int, own_column: str,
                  velocity_check=None, lineage_of=None) -> tuple[list[str], dict[str, Any]]:
    """Signs that the root is not where the process starts, and the checks behind them.

    With a current PAGA graph, the root cell's group in PAGA's column (a single-cell root is
    checked as its group is):
    - inside the tree, but not a branch point of ``BRANCH_POINT`` or more large branches:
      pseudotime then runs out both ways along one path. A progenitor between lineages opens
      three or more (paul15's 7MEP); a group part-way along one lineage does not.
    - at an end, while another end looks less specialised (more genes detected, a smaller share
      of the counts in its top 20 genes; CytoTRACE's and StemID's signals): the process may start
      there. Gene counts alone flagged paul15's right MEP root (323 median vs 1,175 in erythroid
      cells), which is inside the tree and not checked this way. Only ends on the root's own
      lineage are compared (``lineage_of(column, group)``: the group names, or None when it
      cannot be read): an end of another lineage is not a candidate start (Reck 2025 kidney
      from PT S1: medullary IC-A, off the PT lineage, detects more genes).
    - ``velocity_check(groupby, group)``: the saved velocity run's direction check from the
      root along the tree (``velocity_against_root``).
    Without PAGA, or always: this pseudotime runs against another one the dataset holds."""
    out: list[str] = []
    checks: dict[str, Any] = {}
    record = read_paga(adata)
    why = paga_stale(adata, record) if record else None
    group = root_group_in(adata, record["groupby"], iroot) if record and not why else None
    if record and why:
        checks["paga"] = {"skipped": why}
    if record and group is not None:
        col = record["groupby"]
        pos = root_position(record, group)
        checks["paga"] = {"column": col, "root_group": group, **(pos or {})}
        tied = tied_links(record, group)
        if tied:
            # PAGA cannot tell the tree around this group from another one: its position is not
            # settled, so it is not judged (paul15's 7MEP: a branch point in scanpy's tie order,
            # inside between the erythroid and myeloid sides when the ratio breaks the ties).
            checks["paga"]["close_ties"] = tied
        if pos and pos["position"] == "inside" and not tied:
            nbrs = _tree_neighbours(record, group)
            shown = ", ".join(f"'{n}'" for n in nbrs[:3]) + (" and more" if len(nbrs) > 3 else "")
            big = sum(x >= BRANCH_SHARE for x in pos["branches"])
            out.append(
                f"'{group}' sits inside the PAGA tree of {col} (linked to {shown}), not at an end, so pseudotime "
                f"runs out from it both ways. It opens {big} large branch{'es' if big != 1 else ''} "
                f"(at least {BRANCH_SHARE:.0%} of the cells each), not the {BRANCH_POINT} or more of a progenitor "
                "between lineages. The usual root is the start state at an end of the tree."
            )
        if pos and pos["position"] == "end":
            ends = tree_ends(record)
            same = lineage_of(col, group) if lineage_of is not None else None
            if same is not None:
                left_out = [e for e in ends if e not in same]
                if left_out:
                    checks["paga"]["ends_off_lineage"] = left_out
                ends = [e for e in ends if e in same]
            stats = specialisation(adata, col, ends)
            if stats is None:
                checks["specialisation"] = {"skipped": "no raw counts to read"}
            else:
                checks["specialisation"] = stats
                other = less_specialised_end(stats, group, ends)
                if other is not None:
                    o, r = stats[other], stats[group]
                    out.append(
                        f"'{other}', another end of the PAGA tree of {col}, looks less specialised than '{group}': "
                        f"its cells detect more genes (median {o['genes']:,.0f} vs {r['genes']:,.0f}) and their "
                        f"{TOP_GENES} most-counted genes take less of the counts ({o['top_share']:.0%} vs "
                        f"{r['top_share']:.0%}). Less specialised cells usually come first. If the process starts "
                        f"there, pick '{other}' as the root."
                    )
        if velocity_check is not None:
            try:
                verdict = velocity_against_root(velocity_check(col, group), group)
            except Exception as e:  # the velocity check must never block the pseudotime
                logger.warning("Velocity root check failed: %s", e)
                verdict = None
            if verdict is not None:
                checks["velocity"] = verdict
                if verdict["against"]:
                    steps = "; ".join(f"{r['from']} -> {r['to']} ({r['mean_cos']:+.2f})"
                                      for r in verdict["backward"][:4])
                    into = (f" The arrows flow into '{group}' from {', '.join(repr(t) for t in verdict['root_edges_backward'])}."
                            if verdict["root_edges_backward"] else "")
                    out.append(
                        f"The RNA velocity arrows run against this root along the PAGA tree of {col}: "
                        f"{verdict['n_backward']} transition{'s' if verdict['n_backward'] != 1 else ''} backward "
                        f"({steps}), {verdict['n_forward']} forward.{into} The root may be an end state. Velocity "
                        "can be wrong too (genes whose transcription steps up, MURK genes): read its own warnings."
                    )
    other = against_other_pseudotimes(adata, pt, {DPT_COLUMN, own_column})
    if other is not None:
        c, rho, n = other
        out.append(
            f"This pseudotime runs against '{c}' (Spearman rho {rho:.2f}, n = {n:,} cells): one of the two "
            "starts at the wrong end. Check which group the process begins in."
        )
    return out, checks


# Lineage check (``lineage_check``): a group whose best route from the root crosses a link
# where under this share of the smaller group's kNN edges reach the other group is not on a
# path from the root. Widest route from the true root, lowest group: paul15 (7MEP) 0.022
# (19Lymph), Bastidas-Ponce pancreas (Ductal) 0.038 (Delta), Pijuan-Sala erythroid (Blood
# progenitors 1) 0.126. Reck 2025 kidney from PT S1: immune, endothelial, stromal, IC and
# glomerular groups 0.002-0.007 (38 of 59 groups), distal nephron 0.024, PT 0.042 and up.
LINEAGE_MIN_SHARE = 0.01
LINEAGE_MIN_CELLS = 0.05  # warn when the groups off the root's lineage hold this share of the cells
LINEAGE_NAMED = 6


def edge_shares(distances, codes: np.ndarray, n_groups: int) -> np.ndarray:
    """Between each pair of groups: their kNN edges (both ways) as a share of the smaller
    group's own edges, ``codes`` -1 for a cell in no group (left out)."""
    keep = codes >= 0
    D = sp.csr_matrix(distances)[keep][:, keep]
    c = codes[keep]
    both, _ = group_links(D, c, n_groups)
    out_edges = np.bincount(c, weights=np.diff(D.indptr), minlength=n_groups)
    smaller = np.minimum.outer(out_edges, out_edges)
    with np.errstate(divide="ignore", invalid="ignore"):
        share = np.where(smaller > 0, both / (2 * smaller), 0.0)
    np.fill_diagonal(share, 0)
    return share


def lineage_check(adata, work: ad.AnnData, pt: np.ndarray, column: str,
                  root: str | None) -> dict[str, Any] | None:
    """The groups of ``column`` that DPT orders (finite pseudotime) but that sit off the root's
    lineage: the widest route to them from the root group (most shared kNN edges at its weakest
    link, ``edge_shares``) falls under ``LINEAGE_MIN_SHARE``. Their pseudotime is distance from
    the root through a weak link, not progress along a process. None when it cannot be read."""
    if root is None or column not in adata.obs.columns or "distances" not in work.obsp:
        return None
    order, labels = _group_labels(adata, column)
    if root not in order or len(order) < 2:
        return None
    finite = np.isfinite(pt)
    codes = np.where(finite, pd.Index(order).get_indexer(labels), -1)
    share = edge_shares(work.obsp["distances"], codes, len(order))
    tree = spanning_tree(len(order), share, share)
    r = order.index(root)
    n = np.bincount(codes[codes >= 0], minlength=len(order))
    widest = {}
    for j in range(len(order)):
        if n[j] and j != r:
            path = _tree_path(tree, r, j)
            widest[j] = min((share[e] for e in path), default=0.0)
    off = [j for j, w in widest.items() if w < LINEAGE_MIN_SHARE]
    on = [order[j] for j in sorted((j for j, w in widest.items() if w >= LINEAGE_MIN_SHARE),
                                   key=lambda j: -widest[j])]  # closest first
    rows = [{"group": order[j], "n_cells": int(n[j]), "widest_share": round(float(widest[j]), 4),
             "median_pseudotime": round(float(np.median(pt[codes == j])), 4)} for j in off]
    rows.sort(key=lambda x: -x["median_pseudotime"])  # the ones DPT puts latest first
    out: dict[str, Any] = {"column": column, "root_group": root, "min_share": LINEAGE_MIN_SHARE,
                           "lineage": [root, *on], "off_lineage": rows}
    n_off, n_all = int(n[off].sum()) if off else 0, int(n.sum())
    if off and n_off >= LINEAGE_MIN_CELLS * n_all:
        named = ", ".join(f"'{x['group']}' ({x['median_pseudotime']:.2g})" for x in rows[:LINEAGE_NAMED])
        more = f" and {len(rows) - LINEAGE_NAMED} more" if len(rows) > LINEAGE_NAMED else ""
        lineage = ", ".join(f"'{g}'" for g in out["lineage"][:LINEAGE_NAMED + 2]) + (
            " and more" if len(out["lineage"]) > LINEAGE_NAMED + 2 else "")
        out["warning"] = (
            f"{len(rows)} groups of {column} ({n_off:,} of {n_all:,} cells, {n_off / n_all:.0%}) are not on a path "
            f"from '{root}': every route to them crosses a link where under {LINEAGE_MIN_SHARE:.0%} of the smaller "
            f"group's neighbours are in the other. DPT still gives them a pseudotime, latest first (median "
            f"shown): {named}{more}. "
            "That value is distance from the root, not progress along a process. Read the pseudotime within the "
            f"root's lineage ({lineage}), or compute it on those groups' cells alone.")
    return out


def _fmt(v: float) -> str:
    return f"{v:.2g}" if v < 0.1 else f"{v:.2f}"


def compute_dpt(adata, root_column: str | None = None, root_value: str | None = None, *,
                root_cell: str | None = None, velocity_check=None) -> dict[str, Any]:
    """Diffusion pseudotime from a root group (``root_column`` = ``root_value``: the group's
    most central cell, ``root_cell``) or from one cell named by ``root_cell``; writes
    ``DPT_COLUMN`` into ``adata``.

    Cells the root cannot reach through the graph get ``inf``. Cells past an extreme gap
    (``detached_tail``: a weakly linked island) are left out with NaN and the pseudotime
    is computed again without them, so they do not set its scale. Returns the parameters
    (also recorded in ``uns`` and the provenance history), with ``warnings`` and the
    ``root_checks`` behind them (``root_warnings``; ``velocity_check(column, group)`` gives
    the saved velocity run's direction check from that group, or None).
    """
    by_cell = root_cell is not None
    if by_cell == (root_column is not None or root_value is not None) or (
            not by_cell and (root_column is None or root_value is None)):
        raise TrajectoryError(ROOT_CHOICE)
    if adata.n_obs < MIN_CELLS:
        raise TrajectoryError(f"Pseudotime needs at least {MIN_CELLS} cells.")
    if by_cell:
        in_group = np.asarray(adata.obs_names == str(root_cell))
        if not in_group.any():
            raise TrajectoryError(f"No cell is named '{root_cell}'.")
        in_group[np.flatnonzero(in_group)[1:]] = False  # a repeated name: its first cell
        what = f"cell '{root_cell}'"
    else:
        if root_column not in adata.obs.columns:
            raise TrajectoryError(f"Column '{root_column}' not found.")
        col = adata.obs[root_column]
        in_group = (col.astype(str).where(col.notna()) == root_value).to_numpy()
        if not in_group.any():
            raise TrajectoryError(f"No cells have {root_column} = '{root_value}'.")
        what = f"'{root_value}' cells"

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
            f"The {what} sit in a disconnected part of the neighbour graph "
            f"with only {int(reach.sum())} cells; pick a root in the main data."
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
    named = root_column_name(root_value, root_cell if by_cell else None)
    lineages: dict[str, dict[str, Any] | None] = {}

    def _lineage(col: str) -> dict[str, Any] | None:
        if col not in lineages:
            lineages[col] = lineage_check(adata, work, pt, col, root_group_in(adata, col, iroot))
        return lineages[col]

    def lineage_of(col: str, group: str) -> set[str] | None:
        found = _lineage(col)
        return set(found["lineage"]) if found is not None else None

    found, checks = root_warnings(adata, pt, iroot, named, velocity_check, lineage_of)
    warnings.extend(found)
    lineage_col = root_column if not by_cell else (checks.get("paga") or {}).get("column")
    if lineage_col:
        lineage = _lineage(lineage_col)
        if lineage is not None:
            checks["lineage"] = lineage
            if lineage.get("warning"):
                warnings.append(lineage.pop("warning"))

    params: dict[str, Any] = {
        "column": DPT_COLUMN,
        "root_named_column": named,  # the same values, kept when another root runs
        "root_kind": "cell" if by_cell else "group",
        "root_column": root_column,
        "root_value": root_value,
        "root_cell": str(adata.obs_names[iroot]),
        "root_rule": ("the cell given" if by_cell
                      else "most central root-group cell in DPT-weighted diffusion space"),
        "graph": _graph_label(adata, source),
        "diffmap_n_comps": DIFFMAP_COMPS,
        "n_dcs": N_DCS,
        "n_unreachable": int(np.isinf(pt).sum()),
        "n_detached": n_detached,
        "warnings": warnings,
        "root_checks": checks,
    }
    adata.obs[DPT_COLUMN] = pt
    adata.obs[named] = pt
    recorded = {k: v for k, v in params.items() if k != "root_checks"}  # the recipe keeps the warnings
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


def root_label(params: dict[str, Any]) -> str:
    """Where a pseudotime run started, in words: "celltype = Ductal" or "cell 'AAAC-1'"."""
    if params.get("root_kind") == "cell":
        return f"cell '{params.get('root_cell')}'"
    return f"{params.get('root_column')} = {params.get('root_value')}"


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
    maximum spanning tree (``[i, j]``). Connectivities saturate at 1, so several can tie:
    the tree breaks a tie by the uncapped ratio (``spanning_tree``); ``ties`` lists each tie
    the tree depends on (``tree_ties``), and ``warnings`` names the close ones.
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
    capped = sp.csr_matrix(work.uns["paga"]["connectivities"]).toarray()
    _, ratio = group_links(work.obsp["distances"], work.obs["group"].cat.codes.to_numpy(), len(order))
    ties: list[dict[str, Any]] = []
    if np.allclose(np.minimum(ratio, 1.0), capped, rtol=0, atol=1e-6):
        tree_edges = spanning_tree(len(order), capped, ratio)
        ties = tree_ties(tree_edges, capped, ratio)
        tie_break = "uncapped ratio"
    else:  # another PAGA than the one group_links mirrors: scanpy's own tree, ties unread
        logger.warning("PAGA connectivities differ from group_links; scanpy's tree kept")
        tree = sp.coo_matrix(work.uns["paga"]["connectivities_tree"])
        tree_edges = sorted({(min(int(i), int(j)), max(int(i), int(j))) for i, j in zip(tree.row, tree.col)})
        tie_break = "scanpy"
    warning = ties_warning(order, ties)
    record: dict[str, Any] = {
        "groupby": groupby,
        "groups": order,
        "sizes": np.bincount(work.obs["group"].cat.codes, minlength=len(order)).tolist(),
        "edges": [e for e in edges if e[2] > 0],
        "tree": tree_edges,
        "tie_break": tie_break,
        "ties": ties,
        "warnings": [warning] if warning else [],
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


def group_links(distances, codes: np.ndarray, n_groups: int) -> tuple[np.ndarray, np.ndarray]:
    """PAGA's group links (scanpy 1.11 ``PAGA._compute_connectivities_v1_2``) as numbers:
    (kNN edges between each pair of groups both ways, ratio to random wiring of the same
    cells), both n_groups x n_groups with zeros on the diagonal. PAGA's connectivity is the
    ratio capped at 1; uncapped, it still tells two saturated links apart. ``distances``:
    the cells' kNN distances (every stored entry is an edge), ``codes``: each cell's group."""
    D = sp.csr_matrix(distances, copy=True)
    D.data = np.ones_like(D.data, dtype=np.float64)
    n = len(codes)
    M = sp.csr_matrix((np.ones(n), (np.arange(n), codes)), shape=(n, n_groups))
    C = (M.T @ D @ M).toarray()
    inner = np.diag(C).copy()
    np.fill_diagonal(C, 0)
    sizes = np.bincount(codes, minlength=n_groups).astype(np.float64)
    out_edges = inner + C.sum(axis=1)  # a group's edges: within it, and out of it
    both = C + C.T
    expected = (out_edges[:, None] * sizes[None, :] + out_edges[None, :] * sizes[:, None]) / max(1, n - 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(both > 0, np.where(expected > 0, both / expected, 1.0), 0.0)
    np.fill_diagonal(ratio, 0)
    return both, ratio


TIE = 1e-9  # two capped connectivities this close are a tie (in practice both saturate at 1)
# A tie the uncapped ratio breaks by less than this factor is still close: worth a warning.
# Reck 2025's kidney nuclei (59 author groups): 78 tied pairs, 10 under 1.5x, among them
# DTL - PT Injured (1.35) over DTL - PT Inflammatory (1.07); most others 2-20x apart.
CLOSE_TIE = 1.5


def spanning_tree(n_groups: int, capped: np.ndarray, ratio: np.ndarray) -> list[tuple[int, int]]:
    """PAGA's maximum spanning tree (forest for a split graph) on the capped connectivity,
    ties broken by the uncapped ratio (then by group order: the same tree every time).
    scanpy's tree breaks a tie by scipy's edge order instead."""
    edges = sorted(((i, j) for i in range(n_groups) for j in range(i + 1, n_groups) if capped[i, j] > 0),
                   key=lambda e: (-capped[e], -ratio[e], e))
    parent = list(range(n_groups))

    def root(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    tree = []
    for i, j in edges:
        a, b = root(i), root(j)
        if a != b:
            parent[a] = b
            tree.append((i, j))
    return sorted(tree)


def _tree_path(tree: list[tuple[int, int]], a: int, b: int) -> list[tuple[int, int]]:
    """The tree edges from group a to group b (empty when they are not linked)."""
    adj: dict[int, list[int]] = {}
    for i, j in tree:
        adj.setdefault(i, []).append(j)
        adj.setdefault(j, []).append(i)
    back = {a: a}
    stack = [a]
    while stack:
        x = stack.pop()
        for y in adj.get(x, []):
            if y not in back:
                back[y] = x
                stack.append(y)
    if b not in back:
        return []
    path = []
    while b != a:
        path.append((min(b, back[b]), max(b, back[b])))
        b = back[b]
    return path


def tree_ties(tree: list[tuple[int, int]], capped: np.ndarray, ratio: np.ndarray) -> list[dict[str, Any]]:
    """The tree edges that tie with an edge left out of the tree: swapping them gives another
    maximum spanning tree on PAGA's capped connectivity. Each: the tree edge kept, the edge it
    beat, their shared connectivity and both uncapped ratios; largest connectivity first."""
    in_tree = set(tree)
    out = []
    n = capped.shape[0]
    for u in range(n):
        for v in range(u + 1, n):
            w = capped[u, v]
            if w <= 0 or (u, v) in in_tree:
                continue
            for e in _tree_path(tree, u, v):
                if abs(capped[e] - w) <= TIE:
                    r_in, r_out = float(ratio[e]), float(ratio[u, v])
                    out.append({"kept": list(e), "instead_of": [u, v], "connectivity": round(float(w), 4),
                                "ratio_kept": round(r_in, 3), "ratio_instead_of": round(r_out, 3),
                                "close": r_in < CLOSE_TIE * r_out})
    out.sort(key=lambda t: (t["ratio_kept"] / max(t["ratio_instead_of"], 1e-12), t["kept"], t["instead_of"]))
    return out


TIES_NAMED = 3  # close ties a warning names


def ties_warning(groups: list[str], ties: list[dict[str, Any]]) -> str | None:
    """Says where the tree depends on a close tie (``CLOSE_TIE``): which edges were kept over
    which, closest first; None when every tie was broken by a clear margin, or there is none."""
    close = [t for t in ties if t["close"]]
    if not close:
        return None
    shown = "; ".join(
        f"'{groups[t['kept'][0]]} - {groups[t['kept'][1]]}' over '{groups[t['instead_of'][0]]} - "
        f"{groups[t['instead_of'][1]]}' ({t['ratio_kept']:.2f} vs {t['ratio_instead_of']:.2f} times "
        f"random wiring)" for t in close[:TIES_NAMED])
    more = f", and {len(close) - TIES_NAMED} more" if len(close) > TIES_NAMED else ""
    return (f"The tree depends on {len(close)} close tie{'s' if len(close) != 1 else ''}: each tied pair "
            "has the same connectivity (PAGA caps it at 1), so another tree fits PAGA as well. "
            f"The edge with more links against random wiring was kept, by under {CLOSE_TIE:g} times: "
            f"{shown}{more}. Read those steps of the tree as uncertain.")


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
    _, labels = _group_labels(adata, groupby)
    codes = pd.Index(groups).get_indexer(labels)  # -1: no label, or a new group
    out["stale"] = paga_stale(adata, record)
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
