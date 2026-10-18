"""Distances between the groups of a column (clusters, cell types) in the space
the neighbour graph was built in, the view that UMAP and t-SNE plots lack.

UMAP and t-SNE keep each cell's neighbourhood and little else: how far apart
two clusters sit on the plot, and how big each one looks, says little about how
different they are (Chari & Pachter 2023, PLoS Comput Biol 19:e1011288, PMID
37590228). The principal components the graph was built on do keep those
distances, so this module measures them there: each group's centroid (the mean
of its cells' PCs) and the Euclidean distance between every pair of centroids,
with a complete-linkage clustering of the groups (scanpy's dendrogram default)
whose leaf order puts similar groups next to each other in a heatmap, and the
tree itself for a dendrogram beside it. The Pearson correlation between the
centroids (``centroid_correlation``) is given too, in the same space. On Kang
PBMC it keeps the T and NK, the B and the myeloid clusters in blocks, where
average linkage let small far-off groups (Eryth, Mk, pDC) split them.

The space is the neighbour graph's own: Harmony-corrected PCs when the graph
was built on them (a batch-corrected dataset), else ``X_pca``, limited to the
graph's ``n_pcs`` when one was recorded. The response says which.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import pdist, squareform

MAX_GROUPS = 100  # as the Unified View's categorical columns
LINKAGE = "complete"
_PCA_KEYS = ("X_pca_harmony", "X_pca")


class DistanceError(ValueError):
    """The distances can't be computed (no PCs, too few or too many groups)."""


def representation(adata) -> tuple[str, np.ndarray, str]:
    """(obsm key, its PCs, how it was chosen) for the space the neighbour graph
    used. ``how``: ``graph`` (the key recorded with the graph, or ``X_pca`` for
    a graph recorded without one, scanpy's default), ``missing`` (the graph's
    key is not in the file) or ``no_graph``; in the last two, Harmony PCs when
    present, else ``X_pca``."""
    nb = adata.uns.get("neighbors")
    params = (nb.get("params") or {}) if isinstance(nb, dict) else {}
    rep = params.get("use_rep") or ("X_pca" if nb is not None else None)
    if nb is not None and rep in adata.obsm:
        how = "graph"
    else:
        how = "missing" if nb is not None else "no_graph"
        rep = next((k for k in _PCA_KEYS if k in adata.obsm), None)
    if rep is None:
        raise DistanceError("No principal components (X_pca) in this dataset: run the pipeline's PCA step first.")
    X = np.asarray(adata.obsm[rep], dtype=np.float64)
    n_pcs = params.get("n_pcs") if how == "graph" else None
    if isinstance(n_pcs, (int, np.integer)) and 0 < n_pcs < X.shape[1]:
        X = X[:, : int(n_pcs)]
    return rep, X, how


def _groups(labels: pd.Series) -> pd.Categorical:
    """The column as categories in their own order (sorted for a plain column)."""
    if isinstance(labels.dtype, pd.CategoricalDtype):
        return pd.Categorical(labels)
    return pd.Categorical(labels.astype(str).where(labels.notna()))


def centroid_correlation(centroids: np.ndarray) -> np.ndarray:
    """Pearson correlation between group centroids (rows) across the PCs, as
    scanpy's ``tl.dendrogram`` computes it. PCs are centred on the mean cell, so
    this compares how each group differs from the average cell: two groups
    that move the same way from it correlate. Correlations of mean log
    expression (over the variable genes) are high for any two groups, because
    every group shares the overall expression profile: on Kang PBMC by cell type
    they run 0.48 to 0.95 (B vs CD4 naive T 0.76), where the centred PCs give
    -0.77 to 0.69 (B vs CD4 naive T 0.09). A flat centroid correlates 0."""
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.corrcoef(centroids)
    r = np.nan_to_num(np.atleast_2d(r), nan=0.0)
    np.fill_diagonal(r, 1.0)
    return r


def group_distances(adata, column: str) -> dict:
    """Centroid-to-centroid distances between the groups of ``column`` in the
    graph's PC space, and the groups' order under complete-linkage clustering.

    Groups with no cells are left out, and so are cells without a value or with
    non-finite PCs (``n_left_out``). ``distances[i][j]`` pairs ``groups[i]`` and
    ``groups[j]``; ``order`` lists group indices, similar groups adjacent;
    ``tree`` is the clustering itself (scipy linkage rows) for drawing the
    dendrogram, and ``correlation`` the Pearson correlation of the centroids.
    """
    if column not in adata.obs.columns:
        raise DistanceError(f"Column '{column}' not found.")
    rep, X, how = representation(adata)
    cats = _groups(adata.obs[column])
    codes = np.asarray(cats.codes)
    keep = (codes >= 0) & np.isfinite(X).all(axis=1)
    sizes = np.bincount(codes[keep], minlength=len(cats.categories))
    present = np.flatnonzero(sizes)
    if len(present) < 2:
        raise DistanceError(f"'{column}' has {len(present)} group with cells: distances need at least two.")
    if len(present) > MAX_GROUPS:
        raise DistanceError(f"'{column}' has {len(present)} groups; the distance view takes at most {MAX_GROUPS}.")

    # Mean PCs per group: a (groups x cells) indicator times the PCs.
    remap = np.full(len(cats.categories), -1)
    remap[present] = np.arange(len(present))
    rows = remap[codes[keep]]
    member = sp.csr_matrix(
        (np.ones(len(rows)), (rows, np.flatnonzero(keep))), shape=(len(present), adata.n_obs)
    )
    counts = sizes[present].astype(np.float64)
    centroids = np.asarray(member @ X) / counts[:, None]

    condensed = pdist(centroids)
    tree = linkage(condensed, method=LINKAGE, optimal_ordering=True)
    order = leaves_list(tree)
    return {
        "column": column,
        "groups": [str(cats.categories[i]) for i in present],
        "sizes": [int(s) for s in sizes[present]],
        "distances": np.round(squareform(condensed), 4).tolist(),
        "order": [int(i) for i in order],
        # scipy's linkage matrix: row k merges nodes a and b (< n: a group,
        # else row a - n) at height h (a distance), into a node of `size` groups.
        "tree": [[int(a), int(b), round(float(h), 4), int(n)] for a, b, h, n in tree],
        "correlation": np.round(centroid_correlation(centroids), 4).tolist(),
        "correlation_method": "pearson",
        "representation": rep,
        "n_pcs": int(X.shape[1]),
        "rep_source": how,
        "linkage": LINKAGE,
        "metric": "euclidean",
        "n_left_out": int(adata.n_obs - keep.sum()),
    }
