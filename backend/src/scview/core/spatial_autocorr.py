"""Spatially variable genes: global Moran's I per gene (C099).

Equal to squidpy 1.6's defaults, ``sq.gr.spatial_neighbors(adata, library_key=
"library_id")`` then ``sq.gr.spatial_autocorr(adata, mode="moran")``, without
importing squidpy (source read: squidpy/gr/_build.py, _ppatterns.py, BSD-3; and
scanpy/metrics/_morans_i.py):

- Neighbours, one graph per ``obs["library_id"]`` (block-diagonal: never across
  sections). ``grid`` (Visium: ``uns["scview_spatial"]["technology"] == "visium"``,
  or, with no such record, squidpy's own rule: ``uns["spatial"]`` exists): each
  spot's 6 nearest spots (Euclidean on ``obsm["spatial"]``), kept when nearer than
  1.3 x the median of those distances in its section, so an edge spot keeps its
  fewer real neighbours. ``knn`` otherwise: the 6 nearest, no cut. Directed, as
  squidpy builds it (j among i's nearest, not necessarily i among j's). A spot
  matches itself out as sklearn's ``kneighbors()`` does (with duplicates at one
  position, the first hit is dropped when the spot itself is not among them).
- Weights row-normalised (``transformation=True``); a spot with no neighbour keeps
  a zero row and still counts in n and the mean, as in squidpy.
- I = n / S0 * sum_ij w_ij z_i z_j / sum_i z_i^2, z = x - mean over all analysed
  spots (sections pooled). Computed from per-gene sums in row blocks, so neither
  the matrix nor a gene block is ever made dense whole.
- p under the normality assumption (``n_perms=None``): E[I] = -1/(n-1), Var[I] =
  (n^2 S1 - n S2 + 3 S0^2) / ((n-1)(n+1) S0^2) - E[I]^2; one-tailed
  (``two_tailed=False``) in the direction of z. squidpy computes ``1 - cdf(z)``,
  which is 0 beyond z ~ 8.3; here the normal tail (``sf``) is used, equal to 1e-12
  absolute and gives the real small p. Benjamini-Hochberg over the genes (fdr_bh).
- Genes: ``highly_variable`` when the dataset has them, else all (squidpy's
  default), or all, or a list. Constant genes (squidpy: I = NaN) are left out and
  counted. Values: the log-normalised matrix (``activity.lognorm_matrix``: .raw
  after scView's pipeline, else X when it holds log values, else the shifted log of
  the counts), or a named layer; squidpy's run on that matrix as X gives the same.

The functions in ``REPLAY_CODE`` use numpy and scipy only (copied into the replay
script with ``REPLAY_CONSTANTS``).
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
import weakref
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats
from scipy.spatial import cKDTree

from scview.core import results_store

logger = logging.getLogger(__name__)

KIND = "spatial_autocorr"
TAG = "result:spatial_genes"
UNS_KEY = "moranI"  # squidpy's key, so an exported h5ad reads in squidpy as its own result
STEP = "spatial_autocorr"
LIBRARY_COLUMN = "library_id"

N_NEIGHS = 6
GRID_SWAY = 1.3  # squidpy: "there's a small amount of sway" in a grid's spacing
BLOCK_BYTES = 64_000_000  # per row block of W @ X (dense values)
GENE_CHOICES = ("hvg", "all")
MAX_LIST_GENES = 50_000
TOP_FACT_GENES = 10
FACT_PADJ = 0.05
# Measured on 2 cores (synthetic Visium grid, spots in random order, 10% non-zero, CSR
# float32, sorted into blocks by spatial_order): 5,184 spots x 2,000 genes 0.1 s, x 20,000
# genes 1.3 s; 20,736 x 20,000 4.8 s.
SECONDS_PER_VALUE = 1.3e-8
LONG_SECONDS = 4.0  # the UI says how long above this

REPLAY_CONSTANTS = ("N_NEIGHS", "GRID_SWAY", "BLOCK_BYTES")


class SpatialAutocorrError(ValueError):
    """A run that cannot start (no spatial data, no such section, no genes): a 422."""


# ---------------------------------------------------------------------------
# The computation (REPLAY_CODE: copied into the replay script)
# ---------------------------------------------------------------------------


def knn_edges(coords, n_neighs, grid):
    """(rows, cols) of each point's ``n_neighs`` nearest other points, as squidpy's
    spatial_neighbors builds them for one section; ``grid``: only those nearer than
    GRID_SWAY x the median of all those distances (Visium's hexagonal grid)."""
    coords = np.asarray(coords, dtype=np.float64)
    n = coords.shape[0]
    k = min(int(n_neighs), n - 1)
    if k < 1:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    dist, idx = cKDTree(coords).query(coords, k=k + 1)
    dist, idx = np.atleast_2d(dist), np.atleast_2d(idx)
    # Drop the point itself (sklearn kneighbors() rule: when it is not among the hits,
    # duplicates at its position, the first hit goes).
    keep = idx != np.arange(n)[:, None]
    no_self = keep.all(axis=1)
    keep[no_self, 0] = False
    dist = dist[keep].reshape(n, k).ravel()
    cols = idx[keep].reshape(n, k).ravel().astype(np.int64)
    rows = np.repeat(np.arange(n, dtype=np.int64), k)
    if grid:
        near = dist < np.median(dist) * GRID_SWAY
        rows, cols = rows[near], cols[near]
    return rows, cols


def spatial_weights(coords, libraries, grid, n_neighs=N_NEIGHS):
    """Row-normalised spatial weights (CSR, n x n): a kNN graph per section (no edge
    between sections), each row summing to 1 (0 for a spot without neighbours)."""
    coords = np.asarray(coords, dtype=np.float64)
    n = coords.shape[0]
    libraries = np.asarray(libraries)
    rows, cols = [], []
    for lib in pd.unique(libraries):
        where = np.flatnonzero(libraries == lib)
        r, c = knn_edges(coords[where], n_neighs, grid)
        rows.append(where[r])
        cols.append(where[c])
    rows = np.concatenate(rows) if rows else np.zeros(0, dtype=np.int64)
    cols = np.concatenate(cols) if cols else np.zeros(0, dtype=np.int64)
    W = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    deg = np.asarray(W.sum(axis=1)).ravel()
    scale = np.divide(1.0, deg, out=np.zeros_like(deg), where=deg > 0)
    return sp.csr_matrix(sp.diags(scale) @ W)


def weight_moments(W):
    """S0, S1, S2 of a weights matrix (pysal / squidpy's _g_moments)."""
    W = sp.csr_matrix(W)
    s0 = float(W.sum())
    t = W + W.T
    s1 = float(t.multiply(t).sum()) / 2.0
    s2 = float(((np.asarray(W.sum(axis=1)).ravel() + np.asarray(W.sum(axis=0)).ravel()) ** 2).sum())
    return s0, s1, s2


def moran_sums(W, M, order=None):
    """Per gene (columns of M, spots x genes, CSR or dense): (x'Wx, sum x, sum x^2,
    x . rowsums(W), x . colsums(W), constant). Row blocks: a block's spots and their
    neighbours are made dense (at most BLOCK_BYTES), never the whole matrix; ``order``
    (spots sorted by position) keeps a block's neighbours few."""
    W = sp.csr_matrix(W)
    n, g = M.shape
    if sp.issparse(M):
        M = sp.csr_matrix(M)
    r = np.asarray(W.sum(axis=1)).ravel()
    c = np.asarray(W.sum(axis=0)).ravel()
    xwx, s, ss, xr, xc = (np.zeros(g) for _ in range(5))
    lo, hi = np.full(g, np.inf), np.full(g, -np.inf)
    step = max(1, int(BLOCK_BYTES // (8 * max(g, 1) * (N_NEIGHS + 1))))
    order = np.arange(n) if order is None else np.asarray(order)
    for a in range(0, n, step):
        rows = order[a:a + step]
        Wb = W[rows]
        near = np.unique(np.concatenate([rows, Wb.indices]))  # sorted
        XR = M[near]
        XR = XR.toarray() if sp.issparse(XR) else np.asarray(XR)
        XR = XR.astype(np.float64, copy=False)
        Wb = sp.csr_matrix((Wb.data, np.searchsorted(near, Wb.indices), Wb.indptr), shape=(len(rows), len(near)))
        Xb = XR[np.searchsorted(near, rows)]
        xwx += (Xb * (Wb @ XR)).sum(axis=0)
        s += Xb.sum(axis=0)
        ss += (Xb * Xb).sum(axis=0)
        lo = np.minimum(lo, Xb.min(axis=0))
        hi = np.maximum(hi, Xb.max(axis=0))
        xr += r[rows] @ Xb
        xc += c[rows] @ Xb
    return xwx, s, ss, xr, xc, lo == hi


def bh_adjust(p):
    """Benjamini-Hochberg adjusted p (statsmodels' fdr_bh)."""
    p = np.asarray(p, dtype=np.float64)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(adj, 1.0)
    return out


def spatial_order(coords, libraries):
    """Spots sorted by section, then row (y), then x: row blocks of neighbours."""
    coords = np.asarray(coords, dtype=np.float64)
    return np.lexsort((coords[:, 0], coords[:, 1], np.asarray(libraries).astype(str)))


def morans_i(W, M, genes, order=None):
    """Moran's I per gene with squidpy's analytic p-values, sorted by I (highest
    first); constant genes are left out. Returns (table, number of constant genes)."""
    n = M.shape[0]
    xwx, s, ss, xr, xc, constant = moran_sums(W, M, order)
    s0, s1, s2 = weight_moments(W)
    mean = s / n
    num = xwx - mean * xr - mean * xc + mean * mean * s0
    den = ss - n * mean * mean
    keep = ~constant
    with np.errstate(divide="ignore", invalid="ignore"):
        score = n / s0 * num[keep] / den[keep]
    expected = -1.0 / (n - 1)
    var = (n * n * s1 - n * s2 + 3 * s0 * s0) / ((n - 1) * (n + 1) * s0 * s0) - expected ** 2
    z = (score - expected) / np.sqrt(var)
    p = np.where(z > 0, stats.norm.sf(z), stats.norm.cdf(z))
    table = pd.DataFrame({
        "gene": np.asarray(genes, dtype=object)[keep], "I": score, "expected_I": expected,
        "var_I": var, "z": z, "pval": p, "pval_adj": bh_adjust(p),
    })
    table = table.sort_values("I", ascending=False, kind="mergesort").reset_index(drop=True)
    return table, int(constant.sum())


REPLAY_CODE = (knn_edges, spatial_weights, weight_moments, moran_sums, bh_adjust, spatial_order, morans_i)


# ---------------------------------------------------------------------------
# On a dataset
# ---------------------------------------------------------------------------


@dataclass
class Result:
    """One Moran's I run (results store kind ``spatial_autocorr``)."""

    id: str
    dataset_id: str
    n_obs: int
    n_genes: int
    neighbors: dict[str, Any]
    libraries: list[str]
    genes_choice: str  # hvg | all | list
    matrix_source: str
    table: pd.DataFrame
    n_constant: int = 0
    notes: list[str] = field(default_factory=list)
    unit: str = "spot"
    elapsed_seconds: float = 0.0
    created_at: float = 0.0


def has_spatial(adata) -> bool:
    try:
        coords = adata.obsm["spatial"]
    except KeyError:
        return False
    return getattr(coords, "ndim", 0) == 2 and coords.shape[1] >= 2


def is_grid(adata) -> bool:
    """Visium's grid (squidpy coord_type "grid"), else generic kNN."""
    info = adata.uns.get("scview_spatial")
    if isinstance(info, dict) and info.get("technology"):
        return str(info["technology"]) == "visium"
    return "spatial" in adata.uns


def unit_of(adata) -> str:
    info = adata.uns.get("scview_spatial")
    if isinstance(info, dict) and info.get("unit") in ("spot", "cell"):
        return str(info["unit"])
    return "spot" if is_grid(adata) else "cell"


def library_labels(adata) -> np.ndarray:
    if LIBRARY_COLUMN in adata.obs.columns:
        s = adata.obs[LIBRARY_COLUMN]
        return s.astype(str).where(s.notna(), "").to_numpy()
    return np.full(adata.n_obs, "", dtype=object)


def values_matrix(adata, layer: str | None):
    """(spots x genes matrix, gene names, where from)."""
    if layer:
        if layer not in adata.layers:
            raise KeyError(f"Layer '{layer}' not found.")
        return adata.layers[layer], adata.var_names, f"layer:{layer}"
    from scview.core.activity import lognorm_matrix

    try:
        return lognorm_matrix(adata)
    except ValueError as e:
        raise SpatialAutocorrError(str(e)) from None


def select_genes(adata, names, choice) -> tuple[np.ndarray, str, list[str]]:
    """(column indices into the matrix, the choice actually made, notes)."""
    names = pd.Index(names)
    notes: list[str] = []
    if isinstance(choice, (list, tuple)):
        wanted = [str(g) for g in choice][:MAX_LIST_GENES]
        idx = names.get_indexer(pd.Index(wanted).unique())
        missing = int((idx < 0).sum())
        idx = idx[idx >= 0]
        if missing:
            notes.append(f"{missing} of the genes asked for are not in the data.")
        if not len(idx):
            raise SpatialAutocorrError("None of these genes are in the data.")
        return np.asarray(idx), "list", notes
    if choice not in GENE_CHOICES:
        raise SpatialAutocorrError(f"Unknown gene choice '{choice}': use hvg, all or a list of genes.")
    if choice == "hvg":
        hv = adata.var.get("highly_variable")
        if hv is not None and bool(np.asarray(hv, dtype=bool).any()):
            flagged = adata.var_names[np.asarray(hv, dtype=bool)]
            idx = names.get_indexer(flagged)
            idx = idx[idx >= 0]
            if len(idx):
                return np.sort(idx), "hvg", notes
        notes.append("The dataset has no variable genes marked, so every gene was tested.")
    return np.arange(len(names)), "all", notes


def estimate_seconds(n_obs: int, n_genes: int) -> float:
    return round(max(0.2, SECONDS_PER_VALUE * n_obs * n_genes), 1)


def run(adata, *, genes: Any = "hvg", library_id: str | None = None, layer: str | None = None,
        dataset_id: str = "") -> Result:
    """Moran's I for the chosen genes on all sections (or one)."""
    if not has_spatial(adata):
        raise SpatialAutocorrError("This dataset has no spatial coordinates (obsm['spatial']).")
    t0 = time.time()
    libs = library_labels(adata)
    rows = np.arange(adata.n_obs)
    if library_id:
        rows = np.flatnonzero(libs == str(library_id))
        if not len(rows):
            raise SpatialAutocorrError(f"No spots in section '{library_id}'.")
    if len(rows) < 3:
        raise SpatialAutocorrError("Moran's I needs at least 3 spots.")
    coords = np.asarray(adata.obsm["spatial"])[rows, :2]
    if not np.isfinite(coords).all():
        raise SpatialAutocorrError("Some spatial coordinates are missing (NaN).")
    M, names, source = values_matrix(adata, layer)
    idx, choice, notes = select_genes(adata, names, genes)
    if len(rows) < adata.n_obs:
        M = M[rows]
    if len(idx) < M.shape[1]:
        M = M[:, idx]
    grid = is_grid(adata)
    W = spatial_weights(coords, libs[rows], grid)
    if W.nnz == 0:
        raise SpatialAutocorrError(f"No {unit_of(adata)} has a neighbour: the positions are too far apart.")
    table, n_constant = morans_i(W, M, np.asarray(names)[idx], spatial_order(coords, libs[rows]))
    if n_constant:
        notes.append(f"{n_constant:,} gene(s) with the same value in every spot were left out.")
    isolated = int((np.diff(W.indptr) == 0).sum())
    unit = unit_of(adata)
    if isolated:
        notes.append(f"{isolated:,} {unit}(s) have no neighbour within reach; they count in the mean only.")
    return Result(
        id=uuid.uuid4().hex[:12], dataset_id=dataset_id, n_obs=int(len(rows)), n_genes=int(len(table)),
        neighbors={"kind": "grid" if grid else "knn", "k": N_NEIGHS},
        libraries=[str(x) for x in pd.unique(libs[rows])], genes_choice=choice, matrix_source=source,
        table=table, n_constant=n_constant, notes=notes, unit=unit,
        elapsed_seconds=round(time.time() - t0, 2), created_at=time.time(),
    )


def uns_frame(r: Result) -> pd.DataFrame:
    """squidpy's uns["moranI"] layout: genes as the index, I, pval_norm, var_norm,
    pval_norm_fdr_bh."""
    t = r.table
    return pd.DataFrame({"I": t["I"].to_numpy(), "pval_norm": t["pval"].to_numpy(),
                         "var_norm": t["var_I"].to_numpy(), "pval_norm_fdr_bh": t["pval_adj"].to_numpy()},
                        index=pd.Index(t["gene"].astype(str).to_numpy()))


def record(adata, r: Result, genes: Any, library_id: str | None, layer: str | None) -> None:
    """The run in provenance (History, replay script) and squidpy's uns["moranI"]."""
    from scview.core import provenance

    adata.uns[UNS_KEY] = uns_frame(r)
    provenance.record_step(
        adata, step=STEP, tool="scview.spatial_autocorr (= squidpy 1.6 spatial_neighbors + spatial_autocorr)",
        params={"genes": genes if isinstance(genes, str) else [str(g) for g in genes],
                "genes_used": r.genes_choice, "library_id": library_id, "layer": layer,
                "matrix_source": r.matrix_source, "neighbors": r.neighbors["kind"], "n_neighs": N_NEIGHS,
                "n_obs": r.n_obs, "n_genes": r.n_genes, "n_constant": r.n_constant},
        resources={"seconds": r.elapsed_seconds},
    )


# ---------------------------------------------------------------------------
# Kept results
# ---------------------------------------------------------------------------

_last: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_lock = threading.Lock()


def remember(adaptor, r: Result) -> None:
    """Keep the run for GET and the co-pilot, in memory and in the results store."""
    try:
        with _lock:
            _last[adaptor] = r
    except TypeError:  # a test double
        pass
    where = f" in section {r.libraries[0]}" if len(r.libraries) == 1 and r.libraries[0] else ""
    results_store.keep(adaptor, KIND, r, result_id=r.id, key="",
                       label=f"Spatially variable genes (Moran's I, {r.n_genes:,} genes){where}")


def last(adaptor) -> Result | None:
    return results_store.remembered(_last, _lock, adaptor, KIND)


def summary_counts(r: Result) -> dict[str, int]:
    t = r.table
    sig = t["pval_adj"] < FACT_PADJ
    return {"n_significant": int((sig & (t["I"] > 0)).sum()), "n_dispersed": int((sig & (t["I"] < 0)).sum())}


def grounding_lines(adaptor) -> tuple[list[str], str | None]:
    """For the co-pilot: the last run's top spatially variable genes."""
    r = last(adaptor)
    if r is None or r.table.empty:
        return [], None
    t = r.table
    c = summary_counts(r)
    top = t[(t["I"] > 0) & (t["pval_adj"] < FACT_PADJ)].head(TOP_FACT_GENES)
    sections = (f"{len(r.libraries)} sections, neighbours within each" if len(r.libraries) > 1
                else "one section")
    detail = (f"Moran's I on {r.n_obs:,} {r.unit}s ({sections}; {r.neighbors['kind']} neighbours, k "
              f"{r.neighbors['k']}), {r.n_genes:,} genes: {c['n_significant']:,} spatially variable "
              f"(I > 0, adjusted p < {FACT_PADJ:g})")
    lines = [f"\n## Spatially variable genes (last run; Moran's I) [{TAG}]", f"- {detail} [{TAG}]."]
    if len(top):
        lines.append("- Highest Moran's I: " + ", ".join(
            f"{row.gene} (I {row.I:.3f}, adjusted p {row.pval_adj:.2g})" for row in top.itertuples()) + ".")
    else:
        lines.append(f"- No gene has I > 0 with adjusted p < {FACT_PADJ:g}.")
    lines.append(f"- Moran's I is 1 when neighbouring {r.unit}s have alike values, about 0 when the pattern "
                 "is random and below 0 when neighbours differ; it does not say which cell types drive it.")
    return lines, detail
