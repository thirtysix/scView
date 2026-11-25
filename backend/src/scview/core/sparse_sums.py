"""Sums over a cells x genes matrix that never copy it whole.

A Seurat file converted by the R converter holds its counts as CSC. ``onehot @ M`` with a CSR
one-hot converts M to CSR first: a full copy (+1.5-1.6 GB on a deep 23,578 x 34,209 file, in
the ambient RNA card and the pseudobulk sample QC), and ``(M > 0).sum(axis=1)`` another 0.7 GB.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def csc_on_disk(M) -> bool:
    """A backed dataset's CSC matrix: any rows cut from it read all of it (anndata reads every
    column: 250 rows of a deep 23,578 x 34,209 file, +1.58 GB), columns are cheap."""
    return (M is not None and not sp.issparse(M) and not isinstance(M, np.ndarray)
            and getattr(M, "format", None) == "csc")


def first_columns(M, values: int):
    """The first columns of ``M`` holding about ``values`` stored values (at least one)."""
    from scview.core.anndata_adaptor import column_spans

    j0, j1 = column_spans(M, values)[0]
    return M[:, j0:j1]


def onehot(codes: np.ndarray, k: int) -> sp.csr_matrix:
    """k x cells: 1 where a cell has the code (codes < 0: in no group)."""
    codes = np.asarray(codes)
    ok = codes >= 0
    return sp.csr_matrix((np.ones(int(ok.sum())), (codes[ok], np.flatnonzero(ok))),
                         shape=(k, codes.size))


def group_sums(M, codes: np.ndarray, k: int) -> np.ndarray:
    """k x genes, float64: ``M``'s rows summed by ``codes`` (0..k-1, -1: left out). A CSC
    matrix is summed through its transpose (CSR on the same arrays), a CSR one as it is, a
    dense one by a dense product; one on disk (a backed dataset) a span at a time: columns of
    a CSC, rows otherwise."""
    oh = onehot(codes, k)
    if sp.issparse(M) or isinstance(M, np.ndarray):
        out = (M.T @ oh.T).T if sp.issparse(M) and M.format == "csc" else oh @ M
        return np.asarray(out.toarray() if sp.issparse(out) else out, dtype=np.float64)
    from scview.core.anndata_adaptor import ROW_BLOCK_VALUES, column_spans, stored_values

    n, g = (int(x) for x in M.shape)
    out = np.zeros((k, g))
    if getattr(M, "format", None) == "csc":
        for j0, j1 in column_spans(M, ROW_BLOCK_VALUES):
            B = sp.csc_matrix(M[:, j0:j1])
            out[:, j0:j1] = np.asarray((B.T @ oh.T).T.toarray())
        return out
    step = max(1, int(ROW_BLOCK_VALUES // max(1.0, stored_values(M) / max(1, n))))
    oh = oh.tocsc()
    for a in range(0, n, step):
        b = min(n, a + step)
        B = M[a:b]
        part = oh[:, a:b] @ (B if sp.issparse(B) else np.asarray(B))
        out += np.asarray(part.toarray() if sp.issparse(part) else part)
    return out


def row_totals(M) -> np.ndarray:
    """Each row's sum, float64: in memory as it is, on disk in bounded blocks."""
    if sp.issparse(M) or isinstance(M, np.ndarray):
        return np.asarray(M.sum(axis=1), dtype=np.float64).ravel()
    from scview.core.anndata_adaptor import _row_sums

    return _row_sums(M, [])[0]


def detected_per_row(M) -> np.ndarray:
    """Values above 0 per row, float64, read in bounded blocks (``anndata_adaptor._row_sums``)."""
    from scview.core.anndata_adaptor import _row_sums

    if isinstance(M, np.ndarray):
        return (M > 0).sum(axis=1).astype(np.float64)
    if sp.issparse(M) and M.format not in ("csr", "csc"):
        M = M.tocsr()
    return _row_sums(M, [])[1]
