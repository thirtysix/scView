"""Scaling builds only what PCA reads.

PCA reads the variable genes (``sc.tl.pca``'s default ``mask_var="highly_variable"``),
but the scaling step used to make X dense float32 for every gene and z-score it in
place: at 250k cells that was 38 of a run's 46 GB peak, for some 18,000 genes PCA never
read. Now:

- After a run X stays the log-normalised matrix (sparse when it was) for every gene, and
  .raw is set to it as before (markers, DE and the scores read .raw). X and its var keep
  every gene; ``layers['counts']`` too.
- The scaling step records a plan (``uns[PLAN_KEY]``): the method (z-scores of the log
  values, or Pearson residuals of the counts), the genes (the variable genes; every gene
  without them) and its settings (``max_value`` 10; theta and the clip). For z-scores it
  also stores each planned gene's mean and standard deviation in ``var["mean"]`` and
  ``var["std"]`` (the columns ``sc.pp.scale`` wrote; NaN for genes left out), worked out a
  block of genes at a time, so the step itself holds almost nothing.
- The PCA step builds the matrix it reads with ``pca_matrix`` (dense float32, the planned
  genes only, for the length of the step) and stores ``X_pca``, ``uns["pca"]`` and
  ``varm["PCs"]`` (zero rows for genes left out) as ``sc.tl.pca`` with the mask wrote them.
  A chunked build would replace ``pca_matrix`` and nothing else.
- PCA with no plan (scaling skipped, or a dataset an older scView processed, whose X
  already holds the z-scores or residuals of every gene) runs ``sc.tl.pca`` on X as before.

The results are the old ones bit for bit: per gene, the mean and variance of a dense
float32 column do not depend on the other columns (checked on Kang and paul15), the
z-scores are the same element-wise operations, and the matrix is handed to PCA laid out
as anndata's masked view of the old dense X was (column-major). Laid out row-major
instead, arpack's PCs moved by up to 0.017 on Kang.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import scipy.sparse as sp

logger = logging.getLogger(__name__)

PLAN_KEY = "scview_scaling"
ZSCORE = "zscore"
PEARSON = "pearson_residuals"
MAX_VALUE = 10.0  # z-scores are clipped to +-10, as sc.pp.scale(max_value=10)
STAT_BLOCK = 256  # genes per block while the means and standard deviations are worked out
MEAN, STD = "mean", "std"  # var columns, as sc.pp.scale names them


def read_plan(adata) -> dict[str, Any] | None:
    """The recorded plan, or None (scaling not run, or run by an older scView)."""
    plan = adata.uns.get(PLAN_KEY)
    return dict(plan) if isinstance(plan, dict) and plan.get("method") else None


def drop_plan(adata) -> None:
    """Forget the plan: X changed, so its statistics no longer describe it."""
    adata.uns.pop(PLAN_KEY, None)


def gene_mask(adata) -> np.ndarray | None:
    """The genes PCA reads: ``highly_variable`` (sc.tl.pca's default), None for every gene."""
    if "highly_variable" in adata.var.columns:
        return np.asarray(adata.var["highly_variable"].to_numpy(), dtype=bool)
    return None


def holds_scaled(M, rows: int = 2000) -> bool:
    """Whether a matrix holds z-scores or residuals (an older scView's scaled X): values
    below zero in its first rows. Log-normalised values and counts never go below 0."""
    if M is None or not getattr(M, "shape", (0,))[0]:
        return False
    block = M[: min(rows, M.shape[0])]
    if sp.issparse(block):
        return bool(block.nnz) and float(block.data.min()) < 0
    block = np.asarray(block)
    return bool(block.size) and float(block.min()) < 0


def _dense(X, idx: np.ndarray | None, order: str) -> np.ndarray:
    """Columns ``idx`` of X (None: all) as dense float32: the values the old step's
    ``X.astype(np.float32)`` held for them, in ``order``."""
    M = X if idx is None else X[:, idx]
    if sp.issparse(M):
        return M.astype(np.float32, copy=False).toarray(order=order)
    return np.asarray(M, dtype=np.float32, order=order)


def _finish(sums: np.ndarray, squares: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Means and standard deviations from the float64 sums of the values and their squares,
    as ``sc.pp.scale``: the unbiased variance, and 1 for a constant gene."""
    mean = sums / n
    var = squares / n - mean**2
    if n != 1:
        var *= n / (n - 1)
    std = np.sqrt(var)
    std[std == 0] = 1
    return mean, std


def _mean_std(M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per gene, as ``sc.pp.scale`` works them out on a dense matrix: float64 means of the
    values and of their (float32) squares. ``M`` row-major, as the old step's dense X was:
    numpy then adds each column's values up row by row."""
    return _finish(M.sum(axis=0, dtype=np.float64),
                   np.multiply(M, M).sum(axis=0, dtype=np.float64), M.shape[0])


# Stored values per pass of the sparse sums (the int64 column numbers: 32 MB).
SPARSE_CHUNK = 4_000_000


def _sparse_mean_std(X, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """``_mean_std`` of a CSR matrix's columns ``idx`` from its stored values, rows in
    order, into float64 sums (``np.add.at`` adds in order): the additions numpy makes on
    the dense matrix, less the zeros, which add nothing. No dense copy at all. A matrix
    with unsorted or duplicate entries (the 1.3M-neuron subsets) is put in canonical
    order first, in place: the same values, as ``toarray`` sums duplicates too. None for
    a matrix that is not CSR."""
    if getattr(X, "format", None) != "csr":
        return None
    if not X.has_canonical_format:
        X.sum_duplicates()  # sorts each row's entries in place
    n, m = X.shape
    col = np.full(m, -1, dtype=np.int64)
    col[idx] = np.arange(idx.size)
    sums, squares = np.zeros(idx.size), np.zeros(idx.size)
    rows = max(1, SPARSE_CHUNK * n // max(1, X.nnz))
    for r0 in range(0, n, rows):
        a, b = X.indptr[r0], X.indptr[min(n, r0 + rows)]
        j = col[X.indices[a:b]]
        keep = j >= 0
        v = X.data[a:b][keep].astype(np.float32, copy=False)  # as the dense float32 copy holds them
        j = j[keep]
        np.add.at(sums, j, v.astype(np.float64))
        np.add.at(squares, j, np.multiply(v, v).astype(np.float64))
    return _finish(sums, squares, n)


def gene_stats(X, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``_mean_std`` of X's columns ``idx``: from a CSR matrix's stored values in one pass,
    else ``STAT_BLOCK`` genes at a time (each block dense float32, row-major). Either way
    the numbers the old step got from every gene densified at once."""
    found = _sparse_mean_std(X, idx) if sp.issparse(X) else None
    if found is not None:
        return found
    means, stds = [], []
    for j0 in range(0, idx.size, STAT_BLOCK):
        m, s = _mean_std(_dense(X, idx[j0:j0 + STAT_BLOCK], "C"))
        means.append(m)
        stds.append(s)
    if not means:
        return np.zeros(0), np.zeros(0)
    return np.concatenate(means), np.concatenate(stds)


def plan_zscores(adata, max_value: float = MAX_VALUE, stats=None) -> dict[str, Any]:
    """Record z-scores of X (log-normalised) for the genes PCA reads, with each one's
    mean and standard deviation in var; returns what to record in provenance.
    ``stats(idx) -> (mean, std)`` works them out instead of ``gene_stats`` on X (the
    large-dataset run's, by row blocks from disk)."""
    mask = gene_mask(adata)
    idx = np.arange(adata.n_vars) if mask is None else np.flatnonzero(mask)
    mean, std = gene_stats(adata.X, idx) if stats is None else stats(idx)
    for col, values in ((MEAN, mean), (STD, std)):
        full = np.full(adata.n_vars, np.nan)
        full[idx] = values
        adata.var[col] = full
    genes = "all" if mask is None else "highly_variable"
    adata.uns[PLAN_KEY] = {"method": ZSCORE, "genes": genes, "max_value": float(max_value)}
    logger.info("Scaling planned: z-scores of %d genes (%s), max_value=%g", idx.size, genes, max_value)
    return {"scaled_genes": genes, "n_scaled_genes": int(idx.size), "max_value": float(max_value)}


def plan_residuals(adata, theta: float, clip: float, counts_source: str) -> dict[str, Any]:
    """Record Pearson residuals of the counts for the genes PCA reads."""
    mask = gene_mask(adata)
    genes = "all" if mask is None else "highly_variable"
    adata.uns[PLAN_KEY] = {"method": PEARSON, "genes": genes, "theta": float(theta),
                           "clip": float(clip), "counts_source": counts_source}
    n = adata.n_vars if mask is None else int(mask.sum())
    logger.info("Scaling planned: Pearson residuals of %d genes (%s; theta=%g, clip=%.1f)",
                n, genes, theta, clip)
    return {"scaled_genes": genes, "n_scaled_genes": n}


def _zscores_in_place(M: np.ndarray, mean: np.ndarray, std: np.ndarray, max_value: float) -> None:
    """``sc.pp.scale``'s operations on a dense float32 matrix, in place."""
    M -= mean
    np.true_divide(M, std[None, :], out=M)
    np.clip(M, -max_value, max_value, out=M)


def pca_matrix(adata, plan: dict[str, Any] | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    """The matrix PCA reads under the recorded plan, and the genes it covers (a mask;
    None: every gene): dense float32, column-major when masked (as anndata's masked view
    of the old dense X was), held only while PCA runs. Genes that became variable after
    scaling (no mean in var) get their statistics from X now."""
    from scview.core import normalization

    plan = plan or read_plan(adata)
    if plan is None:
        raise ValueError("No scaling plan recorded: run PCA on X as it is.")
    mask = gene_mask(adata)
    idx = None if mask is None else np.flatnonzero(mask)
    order = "C" if mask is None else "F"
    if plan["method"] == PEARSON:
        counts, _ = normalization.find_counts(adata)
        M = normalization.pearson_residuals(counts, theta=float(plan.get("theta", normalization.THETA)),
                                            clip=float(plan["clip"]), columns=idx, order=order)
        return M, mask
    cols = np.arange(adata.n_vars) if idx is None else idx
    mean = _var_column(adata, MEAN, cols)
    std = _var_column(adata, STD, cols)
    missing = np.flatnonzero(~np.isfinite(mean) | ~np.isfinite(std))
    if missing.size:
        logger.info("PCA: %d genes have no scaling statistics (variable since scaling): worked out now",
                    missing.size)
        mean[missing], std[missing] = gene_stats(adata.X, cols[missing])
    M = _dense(adata.X, idx, order)
    _zscores_in_place(M, mean, std, float(plan.get("max_value", MAX_VALUE)))
    return M, mask


def _var_column(adata, name: str, cols: np.ndarray) -> np.ndarray:
    if name not in adata.var.columns:
        return np.full(cols.size, np.nan)
    return np.asarray(adata.var[name].to_numpy(), dtype=np.float64)[cols].copy()


def run_pca(adata, n_comps: int) -> dict[str, Any]:
    """PCA on ``pca_matrix``, stored as ``sc.tl.pca(adata, mask_var="highly_variable")``
    stores it: ``X_pca`` (float32), ``varm["PCs"]`` (float64, zero rows for genes left
    out), ``uns["pca"]`` (params, variance, variance_ratio)."""
    import scanpy as sc

    plan = read_plan(adata)
    M, mask = pca_matrix(adata, plan)
    x_pca, components, ratio, variance = sc.pp.pca(
        M, n_comps=n_comps, svd_solver="arpack", return_info=True)
    del M
    adata.obsm["X_pca"] = x_pca
    if mask is not None:
        pcs = np.zeros((adata.n_vars, n_comps))
        pcs[mask] = components.T
    else:
        pcs = components.T
    adata.varm["PCs"] = pcs
    adata.uns["pca"] = {
        "params": {"zero_center": True, "use_highly_variable": mask is not None,
                   "mask_var": "highly_variable" if mask is not None else None},
        "variance": variance,
        "variance_ratio": ratio,
    }
    return {"pca_input": plan["method"], "pca_genes": int(mask.sum()) if mask is not None else adata.n_vars}
