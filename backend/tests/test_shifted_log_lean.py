"""normalization.shifted_log: the product's values with one copy of the counts held."""

from __future__ import annotations

import tracemalloc

import numpy as np
import pytest
import scipy.sparse as sp

from scview.core import normalization
from scview.core.normalization import shifted_log


def _product(counts, target_sum: float = 1e4):
    """The implementation before v0.82 (diags @ counts.astype)."""
    totals = np.asarray(counts.sum(axis=1), dtype=np.float64).ravel()
    scale = np.divide(target_sum, totals, out=np.zeros_like(totals), where=totals > 0)
    out = sp.csr_matrix(sp.diags(scale) @ counts.astype(np.float64))
    out.data = np.log1p(out.data)
    return out


def _counts(seed: int = 0, n: int = 300, g: int = 120) -> sp.csr_matrix:
    rng = np.random.default_rng(seed)
    dense = rng.poisson(0.7, size=(n, g)).astype(np.float64)
    dense[5] = 0  # a cell with no counts
    return sp.csr_matrix(dense)


@pytest.mark.parametrize("form", ["csr32", "csc64", "coo_int", "csr_int"])
def test_same_values_and_stored_entries_as_the_product(form):
    X = _counts()
    m = {"csr32": X.astype(np.float32), "csc64": X.tocsc(), "coo_int": X.astype(np.int64).tocoo(),
         "csr_int": X.astype(np.int32)}[form]
    before = m.copy()
    got, want = shifted_log(m), _product(m)
    assert got.format == "csr" and got.dtype == np.float64
    assert np.array_equal(got.toarray(), want.toarray())  # bit for bit
    assert got.nnz == want.nnz
    assert np.array_equal(m.toarray(), before.toarray())  # the counts are left as they were


def test_stored_zeros_and_duplicates_as_the_product():
    X = _counts(1)
    X.data[::7] = 0  # stored zeros
    # Row 0 stores column 1 twice (not canonical): the product summed them.
    dup = sp.csr_matrix((np.array([1.0, 2.0, 3.0, 4.0]), np.array([1, 1, 2, 0]), np.array([0, 3, 3, 4])),
                        shape=(3, 4))
    for m in (X, dup):
        got, want = shifted_log(m), _product(m)
        assert np.array_equal(got.toarray(), want.toarray())
        assert got.nnz == want.nnz


def test_dense_unchanged():
    X = _counts(2).toarray()
    totals = X.sum(axis=1)
    scale = np.divide(1e4, totals, out=np.zeros_like(totals), where=totals > 0)
    assert np.array_equal(shifted_log(X), np.log1p(X * scale[:, None]))


def test_values_scaled_in_blocks(monkeypatch):
    X = _counts(3, n=1000)
    whole = shifted_log(X)
    monkeypatch.setattr(normalization, "SHIFTED_LOG_ENTRIES", 50)
    assert np.array_equal(shifted_log(X).toarray(), whole.toarray())


def test_csc_counts_peak_one_copy(monkeypatch):
    """CSC float64 counts (as convert.R writes them): the product held about three copies
    at once; now the CSR copy and one block's row scale (the real block: 32 MB)."""
    rng = np.random.default_rng(4)
    monkeypatch.setattr(normalization, "SHIFTED_LOG_ENTRIES", 1 << 16)
    X = sp.random(4000, 3000, density=0.08, format="csc", random_state=rng, dtype=np.float64)
    X.data = np.ceil(X.data * 20)
    copy = X.data.nbytes + X.indices.nbytes + X.indptr.nbytes
    tracemalloc.start()
    try:
        out = shifted_log(X)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert out.nnz == X.nnz
    assert peak < 1.5 * copy, (peak, copy)
