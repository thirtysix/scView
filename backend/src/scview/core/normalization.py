"""Normalisation options: the shifted log (default) and analytic Pearson residuals.

The shifted log (``normalize_total`` + ``log1p``) stays the default: a benchmark of
transformations found it performs as well as the alternatives (Ahlmann-Eltze &
Huber 2023, PMID 37037999). Analytic Pearson residuals (Lause et al. 2021, PMID
34488842; scanpy's ``experimental.pp``) are the option. Each count is compared with
a negative binomial expectation (cell total x gene share, shared overdispersion
``THETA``), so the residuals are variance-stabilised: genes are ranked by residual
variance to pick the variable genes, and the residuals feed PCA.

That is all they feed. ``.raw`` keeps the log-normalised matrix and
``layers['counts']`` the counts under both methods, so markers, DE, cell cycle, the
stress score, expression plots and pseudobulk read exactly what they read under the
default. The pipeline steps stay the same: with Pearson residuals, the variable-gene
step uses scanpy's ``pearson_residuals`` flavour on the counts and the scaling step
plans residuals instead of z-scores for the matrix PCA reads (core/scaling.py; an
older scView wrote the residuals of every gene to X).
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

METHODS: tuple[str, ...] = ("log", "pearson_residuals")
DEFAULT_METHOD = "log"
THETA = 100.0  # scanpy's default overdispersion for analytic Pearson residuals
_CHUNK = 1000  # genes per block: residuals are dense, so build them a block at a time
DIAGNOSTIC_MAX_CELLS = 20_000  # cells sampled for the mean-variance diagnostic


def check_method(method: str) -> str:
    if method not in METHODS:
        raise ValueError(
            f"Unknown normalization method '{method}'; choose from {', '.join(METHODS)}."
        )
    return method


def find_counts(adata):
    """Raw counts for X's genes, without changing ``adata``: (matrix, where from).

    ``layers['counts']``, then ``.raw`` (when it holds counts for every gene in X),
    then X. Raises ``ValueError`` when none of them holds counts."""
    from scview.core.large_mode import counts_of
    from scview.core.pipeline import looks_like_counts  # pipeline imports this module

    large = counts_of(adata)  # a large dataset's counts sidecar (row reads only)
    if large is not None:
        return large, "the counts file"
    if "counts" in adata.layers and looks_like_counts(adata.layers["counts"]):
        return adata.layers["counts"], "layers['counts']"
    raw = adata.raw
    if (
        raw is not None
        and not len(adata.var_names.difference(raw.var_names))
        and looks_like_counts(raw.X)
    ):
        return raw[:, adata.var_names].X, ".raw"
    if looks_like_counts(adata.X):
        return adata.X, "X"
    raise ValueError(
        "Pearson residuals need raw counts (layers['counts'], .raw or X), and this "
        "dataset has none: use the shifted log instead."
    )


def counts_all_genes(adata):
    """Raw counts for as many genes as the dataset has: (matrix, gene names, where
    from). ``.raw`` can hold more genes than X (an HVG subset upload)."""
    from scview.core.pipeline import looks_like_counts

    raw = adata.raw
    if raw is not None and raw.n_vars > adata.n_vars and looks_like_counts(raw.X):
        return raw.X, raw.var_names, ".raw"
    M, source = find_counts(adata)
    return M, adata.var_names, source


def default_clip(n_cells: int) -> float:
    """scanpy's default clip: residuals beyond +-sqrt(n_cells) are clipped."""
    return float(np.sqrt(n_cells))


def _sums(counts) -> tuple[np.ndarray, np.ndarray, float]:
    cells = np.asarray(counts.sum(axis=1), dtype=np.float64).ravel()
    genes = np.asarray(counts.sum(axis=0), dtype=np.float64).ravel()
    return cells, genes, float(genes.sum())


def _block(counts, j0: int, j1: int, cells, genes, total, theta, clip) -> np.ndarray:
    """Residuals of genes j0:j1 for every cell (float64), as scanpy computes them."""
    C = counts[:, j0:j1]
    C = C.toarray() if sp.issparse(C) else np.asarray(C)
    mu = np.outer(cells, genes[j0:j1] / total) if total > 0 else np.zeros(C.shape)
    with np.errstate(divide="ignore", invalid="ignore"):
        R = (C - mu) / np.sqrt(mu + mu**2 / theta)
    # An all-zero gene or cell has no expectation (0 / 0): no evidence either way.
    R[~np.isfinite(R)] = 0.0
    np.clip(R, -clip, clip, out=R)
    return R


def pearson_residuals(
    counts, theta: float = THETA, clip: float | None = None, chunk: int = _CHUNK,
    columns=None, order: str = "C",
) -> np.ndarray:
    """Analytic Pearson residuals of a cells x genes count matrix, as float32.

    The same values as ``sc.experimental.pp.normalize_pearson_residuals`` (cell and
    gene totals over the whole matrix, residuals clipped at +-sqrt(n_cells) by
    default), but built a block of genes at a time: scanpy holds several dense
    float64 copies of the matrix at once (~4 GB for 14k cells x 14k genes).
    ``columns`` (gene indices): only those genes' residuals, the cell totals
    still taken over every gene, as the pipeline's PCA reads them (core/scaling.py,
    ``order`` "F": laid out as PCA read them from the old residuals of every gene)."""
    n, g = counts.shape
    clip = default_clip(n) if clip is None else float(clip)
    cells, genes, total = _sums(counts)
    if columns is not None:
        columns = np.asarray(columns, dtype=np.int64)
        counts, genes, g = counts[:, columns], genes[columns], columns.size
    out = np.empty((n, g), dtype=np.float32, order=order)
    for j0 in range(0, g, chunk):
        j1 = min(g, j0 + chunk)
        out[:, j0:j1] = _block(counts, j0, j1, cells, genes, total, theta, clip)
    return out


def residual_variance(
    counts, theta: float = THETA, clip: float | None = None, chunk: int = _CHUNK
) -> np.ndarray:
    """Per-gene variance of the Pearson residuals (ddof 0), never the whole matrix."""
    n, g = counts.shape
    clip = default_clip(n) if clip is None else float(clip)
    cells, genes, total = _sums(counts)
    out = np.empty(g, dtype=np.float64)
    for j0 in range(0, g, chunk):
        j1 = min(g, j0 + chunk)
        out[j0:j1] = _block(counts, j0, j1, cells, genes, total, theta, clip).var(axis=0)
    return out


def _mean_var(M) -> tuple[np.ndarray, np.ndarray]:
    """Per-gene mean and variance (ddof 0) of a cells x genes matrix, sparse or dense."""
    n = M.shape[0]
    if sp.issparse(M):
        M = M.tocsc().astype(np.float64)
        mean = np.asarray(M.sum(axis=0)).ravel() / n
        sq = np.asarray(M.multiply(M).sum(axis=0)).ravel() / n
    else:
        M = np.asarray(M, dtype=np.float64)
        mean = M.mean(axis=0)
        sq = (M * M).mean(axis=0)
    return mean, np.maximum(sq - mean**2, 0.0)


def shifted_log(counts, target_sum: float = 1e4):
    """log1p(counts / cell total x target_sum), sparse in, sparse out."""
    totals = np.asarray(counts.sum(axis=1), dtype=np.float64).ravel()
    scale = np.divide(target_sum, totals, out=np.zeros_like(totals), where=totals > 0)
    if sp.issparse(counts):
        out = sp.diags(scale) @ counts.astype(np.float64)
        out = sp.csr_matrix(out)
        out.data = np.log1p(out.data)
        return out
    return np.log1p(np.asarray(counts, dtype=np.float64) * scale[:, None])


def mean_variance(
    counts, gene_names, max_cells: int = DIAGNOSTIC_MAX_CELLS, seed: int = 0
) -> dict:
    """Per-gene mean and variance before and after normalisation, for a diagnostic plot.

    Counts are overdispersed: their variance grows with the mean, faster than a
    Poisson's (variance = mean). The shifted log shrinks that trend but leaves
    lowly expressed genes with little variance; Pearson residuals make the variance
    about 1 for a gene that varies only by sampling, at any mean, so a gene above 1
    varies for a reason. Genes never detected in the sampled cells are left out."""
    n = counts.shape[0]
    rows = np.arange(n)
    if n > max_cells:
        rows = np.sort(np.random.default_rng(seed).choice(n, max_cells, replace=False))
        counts = counts[rows]
    elif hasattr(counts, "row_blocks"):  # a large dataset's counts on disk: its rows, read once
        counts = counts[:]
    if sp.issparse(counts):
        counts = sp.csr_matrix(counts)
    mean, var = _mean_var(counts)
    keep = mean > 0
    counts = counts[:, np.flatnonzero(keep)]
    _, var_log = _mean_var(shifted_log(counts))
    clip = default_clip(counts.shape[0])
    return {
        "n_cells": int(n),
        "n_cells_used": int(counts.shape[0]),
        "theta": THETA,
        "clip": clip,
        "genes": [str(g) for g in np.asarray(gene_names)[keep]],
        "mean": mean[keep],
        "var_counts": var[keep],
        "var_log": var_log,
        "var_residuals": residual_variance(counts, clip=clip),
        "keep": keep,
    }
