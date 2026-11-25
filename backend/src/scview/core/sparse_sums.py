"""Sums over a cells x genes matrix that never copy it whole.

A Seurat file converted by the R converter holds its counts as CSC. ``onehot @ M`` with a CSR
one-hot converts M to CSR first: a full copy (+1.5-1.6 GB on a deep 23,578 x 34,209 file, in
the ambient RNA card and the pseudobulk sample QC), and ``(M > 0).sum(axis=1)`` another 0.7 GB.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def onehot(codes: np.ndarray, k: int) -> sp.csr_matrix:
    """k x cells: 1 where a cell has the code (codes < 0: in no group)."""
    codes = np.asarray(codes)
    ok = codes >= 0
    return sp.csr_matrix((np.ones(int(ok.sum())), (codes[ok], np.flatnonzero(ok))),
                         shape=(k, codes.size))


def group_sums(M, codes: np.ndarray, k: int) -> np.ndarray:
    """k x genes, float64: ``M``'s rows summed by ``codes`` (0..k-1, -1: left out). A CSC
    matrix is summed through its transpose (CSR on the same arrays), a CSR one as it is, a
    dense one by a dense product; anything else (an on-disk matrix) is read as CSR."""
    oh = onehot(codes, k)
    if not sp.issparse(M) and not isinstance(M, np.ndarray):
        M = sp.csr_matrix(M)
    if sp.issparse(M) and M.format == "csc":
        out = (M.T @ oh.T).T
    else:
        out = oh @ M
    return np.asarray(out.toarray() if sp.issparse(out) else out, dtype=np.float64)


def detected_per_row(M) -> np.ndarray:
    """Values above 0 per row, float64, read in bounded blocks (``anndata_adaptor._row_sums``)."""
    from scview.core.anndata_adaptor import _row_sums

    if not sp.issparse(M):
        return (np.asarray(M) > 0).sum(axis=1).astype(np.float64)
    if M.format not in ("csr", "csc"):
        M = M.tocsr()
    return _row_sums(M, [])[1]
