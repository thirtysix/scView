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


def find_counts(adata, read_only: bool = False):
    """Raw counts for X's genes, without changing ``adata``: (matrix, where from).

    ``layers['counts']``, then ``.raw`` (when it holds counts for every gene in X),
    then X. Raises ``ValueError`` when none of them holds counts. ``read_only``: the
    caller only reads the matrix, so a ``.raw`` with X's genes in X's order is handed
    over as it is, not cut to them (a copy of all of it: +4.5 GB at its peak on 125k
    PBMCs, which got the backend killed during a DE's transcriptome-size check)."""
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
        on_disk = not sp.issparse(raw.X) and not isinstance(raw.X, np.ndarray)
        if (on_disk or read_only) and raw.var_names.equals(adata.var_names):
            return raw.X, ".raw"  # on disk (a backed dataset): cutting its genes would read it whole
        return raw[:, adata.var_names].X, ".raw"
    if looks_like_counts(adata.X):
        return adata.X, "X"
    raise ValueError(
        "Pearson residuals need raw counts (layers['counts'], .raw or X), and this "
        "dataset has none: use the shifted log instead."
    )


def counts_all_genes(adata, read_only: bool = False):
    """Raw counts for as many genes as the dataset has: (matrix, gene names, where
    from). ``.raw`` can hold more genes than X (an HVG subset upload). ``read_only``:
    as ``find_counts``'s."""
    from scview.core.pipeline import looks_like_counts

    raw = adata.raw
    if raw is not None and raw.n_vars > adata.n_vars and looks_like_counts(raw.X):
        return raw.X, raw.var_names, ".raw"
    M, source = find_counts(adata, read_only)
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


SHIFTED_LOG_ENTRIES = 1 << 22  # stored values scaled at a time (their row scale: 32 MB, never whole)


def shifted_log(counts, target_sum: float = 1e4):
    """log1p(counts / cell total x target_sum), sparse in, sparse (CSR, float64) out.

    One float64 CSR copy of the counts, scaled and logged in place: ``diags @ counts.astype``
    held the cast copy, the product and its CSR at once (+4.5 GB over 1.6 GB of CSC counts,
    23,578 x 34,209, the marker genes a first open computes; the backend died at 6 GB).
    The values are the product's bit for bit (one multiply per value)."""
    totals = np.asarray(counts.sum(axis=1), dtype=np.float64).ravel()
    scale = np.divide(target_sum, totals, out=np.zeros_like(totals), where=totals > 0)
    if sp.issparse(counts):
        if counts.format == "csr":
            out = counts.astype(np.float64, copy=True)
        else:  # tocsr builds new arrays: the copy itself, cast without another
            out = sp.csr_matrix(counts).astype(np.float64, copy=False)
            if out.data is getattr(counts, "data", None):
                out = out.copy()
        out.sum_duplicates()  # as the product did (in place; a no-op on canonical counts)
        ptr, n = out.indptr, out.shape[0]
        r0 = 0
        while r0 < n:  # whole cells, about SHIFTED_LOG_ENTRIES values at a time
            r1 = max(r0 + 1, min(n, int(np.searchsorted(ptr, ptr[r0] + SHIFTED_LOG_ENTRIES, "right")) - 1))
            out.data[ptr[r0]:ptr[r1]] *= np.repeat(scale[r0:r1], np.diff(ptr[r0:r1 + 1]))
            r0 = r1
        np.log1p(out.data, out=out.data)
        if not out.data.all():  # the product kept no zeros (stored ones, cells with no counts)
            out.eliminate_zeros()
        return out
    return np.log1p(np.asarray(counts, dtype=np.float64) * scale[:, None])


# Blocks of the mean-variance diagnostic: stored values of the counts read at a time, and
# cells x genes of the dense residuals built at a time (32 MB as float64).
MV_BLOCK_VALUES = 8_000_000
MV_RESIDUAL_VALUES = 4_000_000


class _Moments:
    """Per-gene sum and sum of squares, added up over blocks of cells."""

    def __init__(self, g: int) -> None:
        self.s = np.zeros(g)
        self.sq = np.zeros(g)

    def add(self, B, cols: slice | None = None) -> None:
        """``B``: cells x genes (sparse or dense); ``cols``: the genes it holds (default all)."""
        at = slice(None) if cols is None else cols
        if sp.issparse(B):
            B = B.astype(np.float64, copy=False)  # squares of float32 counts lose digits
            self.s[at] += np.asarray(B.sum(axis=0)).ravel()
            self.sq[at] += np.asarray(B.multiply(B).sum(axis=0)).ravel()
        else:
            B = np.asarray(B, dtype=np.float64)
            self.s[at] += B.sum(axis=0)
            self.sq[at] += (B * B).sum(axis=0)

    def mean_var(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        mean = self.s / n
        return mean, np.maximum(self.sq / n - mean**2, 0.0)


def _log_of(B, totals: np.ndarray, target_sum: float = 1e4):
    """``shifted_log`` of block ``B`` whose cells have ``totals`` (over every gene)."""
    scale = np.divide(target_sum, totals, out=np.zeros_like(totals), where=totals > 0)
    if sp.issparse(B):
        L = B.astype(np.float64, copy=True).tocoo()
        L.data = np.log1p(L.data * scale[L.row])
        return L
    return np.log1p(np.asarray(B, dtype=np.float64) * scale[:, None])


def _residuals_of(B, totals: np.ndarray, genes: np.ndarray, total: float, clip: float) -> np.ndarray:
    """Pearson residuals of block ``B`` (cells x genes, dense float64), as ``_block``."""
    C = B.toarray() if sp.issparse(B) else np.asarray(B)
    C = np.asarray(C, dtype=np.float64)
    mu = np.outer(totals, genes / total) if total > 0 else np.zeros(C.shape)
    with np.errstate(divide="ignore", invalid="ignore"):
        R = (C - mu) / np.sqrt(mu + mu**2 / THETA)
    R[~np.isfinite(R)] = 0.0
    np.clip(R, -clip, clip, out=R)
    return R


def _row_spans(rows: np.ndarray, per_row: float, budget: int) -> list[np.ndarray]:
    step = max(1, int(budget // max(1.0, per_row)))
    return [rows[a:a + step] for a in range(0, rows.size, step)]


def _take(counts, idx: np.ndarray):
    """Rows ``idx`` (ascending) of ``counts``, in memory, on disk or a large dataset's lazy
    rows: a run of rows as one slice (a view of a dense matrix), else by index."""
    if idx.size and int(idx[-1]) - int(idx[0]) + 1 == idx.size:
        if idx.size == counts.shape[0] and (isinstance(counts, np.ndarray) or sp.issparse(counts)):
            return counts
        return counts[int(idx[0]):int(idx[-1]) + 1]
    return counts[idx]


def mean_variance(
    counts, gene_names, max_cells: int = DIAGNOSTIC_MAX_CELLS, seed: int = 0
) -> dict:
    """Per-gene mean and variance before and after normalisation, for a diagnostic plot.

    Counts are overdispersed: their variance grows with the mean, faster than a
    Poisson's (variance = mean). The shifted log shrinks that trend but leaves
    lowly expressed genes with little variance; Pearson residuals make the variance
    about 1 for a gene that varies only by sampling, at any mean, so a gene above 1
    varies for a reason. Genes never detected in the sampled cells are left out.

    Read a block at a time, never as a copy of the sampled cells: a CSC matrix (a converted
    Seurat file) by spans of genes, anything else by spans of cells, in two passes (totals,
    then the residuals, which need every gene's total). The whole-sample copies (CSR, CSC,
    float64, the kept genes, the shifted log) took +5.1 GB and 86 s on a deep 23,578 x
    34,209 file."""
    n = counts.shape[0]
    rows = np.arange(n)
    if n > max_cells:
        rows = np.sort(np.random.default_rng(seed).choice(n, max_cells, replace=False))
    ns = int(rows.size)
    clip = default_clip(ns)
    if getattr(counts, "format", None) == "csc" and not hasattr(counts, "row_blocks"):
        stats = _mean_variance_by_genes(counts, rows, ns == n, clip)
    else:
        stats = _mean_variance_by_cells(counts, rows, clip)
    mean, var, var_log, var_res = stats
    keep = mean > 0
    return {
        "n_cells": int(n),
        "n_cells_used": ns,
        "theta": THETA,
        "clip": clip,
        "genes": [str(x) for x in np.asarray(gene_names)[keep]],
        "mean": mean[keep],
        "var_counts": var[keep],
        "var_log": var_log[keep],
        "var_residuals": var_res[keep],
        "keep": keep,
    }


def _mean_variance_by_cells(counts, rows: np.ndarray, clip: float):
    n_s, g = rows.size, counts.shape[1]
    nnz = getattr(counts, "nnz", None)
    per_row = (nnz / max(1, counts.shape[0])) if isinstance(nnz, (int, np.integer)) else g
    raw, logged = _Moments(g), _Moments(g)
    totals = np.zeros(n_s)
    at = 0
    for idx in _row_spans(rows, per_row, MV_BLOCK_VALUES):
        B = _take(counts, idx)
        if not sp.issparse(B):
            B = np.asarray(B, dtype=np.float64)
        t = np.asarray(B.sum(axis=1), dtype=np.float64).ravel()
        totals[at:at + t.size] = t
        at += t.size
        raw.add(B)
        logged.add(_log_of(B, t))
    mean, var = raw.mean_var(n_s)
    _, var_log = logged.mean_var(n_s)
    genes, total = raw.s, float(raw.s.sum())
    res = _Moments(g)
    at = 0
    for idx in _row_spans(rows, g, MV_RESIDUAL_VALUES):
        B = _take(counts, idx)
        res.add(_residuals_of(B, totals[at:at + idx.size], genes, total, clip))
        at += idx.size
    _, var_res = res.mean_var(n_s)
    return mean, var, var_log, var_res


def _gene_spans(counts, budget: int) -> list[tuple[int, int]]:
    """Column ranges of about ``budget`` stored values (``anndata_adaptor.column_spans``: an
    on-disk CSC's indptr is never searched value by value)."""
    from scview.core.anndata_adaptor import column_spans

    return column_spans(counts, budget)


def _mean_variance_by_genes(counts, rows: np.ndarray, all_rows: bool, clip: float):
    n_s, g = rows.size, counts.shape[1]

    def block(j0: int, j1: int):
        B = counts[:, j0:j1]
        return B if all_rows else B[rows]

    raw = _Moments(g)
    totals = np.zeros(n_s)
    for j0, j1 in _gene_spans(counts, MV_BLOCK_VALUES):
        B = block(j0, j1)
        totals += np.asarray(B.sum(axis=1), dtype=np.float64).ravel()
        raw.add(B, slice(j0, j1))
    mean, var = raw.mean_var(n_s)
    genes, total = raw.s, float(raw.s.sum())
    logged, res = _Moments(g), _Moments(g)
    width = max(1, MV_RESIDUAL_VALUES // max(1, n_s))
    for j0, j1 in _gene_spans(counts, MV_BLOCK_VALUES):
        B = block(j0, j1)
        logged.add(_log_of(B, totals), slice(j0, j1))
        for k0 in range(j0, j1, width):
            k1 = min(j1, k0 + width)
            res.add(_residuals_of(B[:, k0 - j0:k1 - j0], totals, genes[k0:k1], total, clip),
                    slice(k0, k1))
    _, var_log = logged.mean_var(n_s)
    _, var_res = res.mean_var(n_s)
    return mean, var, var_log, var_res
