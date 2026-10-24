"""Blocks of genes from a matrix in memory (core/matrix_genes.py).

Every block must be exactly the matching columns of scipy's own canonical CSC of the
matrix with explicit zeros dropped (indptr, cell indices and values, in the matrix's
dtype), whatever the layout: CSR with ascending, descending or unordered rows, CSC,
dense in C or Fortran order. The matrix itself must be left as it was.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from scview.core import matrix_genes as mg


def _random(n=300, g=57, density=0.15, dtype=np.float32, seed=0) -> sp.csr_matrix:
    rng = np.random.default_rng(seed)
    m = sp.random(n, g, density=density, format="csr", random_state=rng, dtype=np.float64)
    m.data = np.round(m.data * 8, 1).astype(dtype)  # ties, as log counts have
    m.data[rng.random(m.nnz) < 0.05] *= -1
    m.data[m.indptr[5]:m.indptr[6]] = 0  # an empty row
    m.data[m.indices == 11] = 0  # a gene no cell has
    m.eliminate_zeros()
    m.sort_indices()
    return m


def _with_explicit_zeros(m: sp.csr_matrix) -> sp.csr_matrix:
    m = m.copy()
    m.data[::7] = 0
    m.data[3::11] = -0.0
    return m


def _reorder_rows(m: sp.csr_matrix, how: str, seed: int = 1) -> sp.csr_matrix:
    """The same matrix with each row's stored values in another order."""
    m = m.copy()
    rng = np.random.default_rng(seed)
    for r in range(m.shape[0]):
        a, b = m.indptr[r], m.indptr[r + 1]
        order = np.arange(b - a)[::-1] if how == "descending" else rng.permutation(b - a)
        m.indices[a:b] = m.indices[a:b][order]
        m.data[a:b] = m.data[a:b][order]
    m.has_sorted_indices = False
    return m


def _expected(m) -> sp.csc_matrix:
    e = sp.csc_matrix(m.toarray() if sp.issparse(m) else np.asarray(m))
    e.eliminate_zeros()
    e.sort_indices()
    return e


def _check(m, max_entries, layout):
    before = (m.data.copy(), m.indices.copy(), m.indptr.copy()) if sp.issparse(m) else m.copy()
    src = mg.MatrixGenes(m, [f"G{i}" for i in range(m.shape[1])])
    assert src.layout == layout
    want = _expected(m)
    seen = []
    for g0, g1, block in src.blocks(max_entries):
        seen.append((g0, g1))
        assert block.shape == (m.shape[0], g1 - g0)
        assert block.dtype == m.dtype
        ref = want[:, g0:g1]
        np.testing.assert_array_equal(block.indptr, ref.indptr)
        np.testing.assert_array_equal(block.indices, ref.indices)
        np.testing.assert_array_equal(block.data, ref.data)
        assert block.nnz == ref.nnz
    # contiguous, every gene once
    assert seen[0][0] == 0 and seen[-1][1] == m.shape[1]
    assert all(a[1] == b[0] for a, b in zip(seen[:-1], seen[1:]))
    if sp.issparse(m):  # untouched
        np.testing.assert_array_equal(m.data, before[0])
        np.testing.assert_array_equal(m.indices, before[1])
        np.testing.assert_array_equal(m.indptr, before[2])
    else:
        np.testing.assert_array_equal(m, before)
    return seen


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("max_entries", [1, 97, 10**9])
@pytest.mark.parametrize("how", ["ascending", "descending", "copy"])
def test_csr_blocks_equal_scipy_csc(dtype, max_entries, how):
    m = _with_explicit_zeros(_random(dtype=dtype))
    if how != "ascending":
        m = _reorder_rows(m, how)
    seen = _check(m, max_entries, how)
    if max_entries == 97:
        assert len(seen) > 5  # ragged blocks, not one


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_csc_and_dense_blocks(dtype):
    m = _with_explicit_zeros(_random(dtype=dtype))
    csc = sp.csc_matrix(m)
    csc.indices = csc.indices.copy()
    # a CSC with unsorted cells in a column
    for c in range(csc.shape[1]):
        a, b = csc.indptr[c], csc.indptr[c + 1]
        csc.indices[a:b] = csc.indices[a:b][::-1]
        csc.data[a:b] = csc.data[a:b][::-1]
    csc.has_sorted_indices = False
    _check(csc, 97, "csc")
    dense = m.toarray()
    _check(dense, 1000, "dense")
    _check(np.asfortranarray(dense), 1000, "dense")
    _check(dense, 1, "dense")  # one gene per block


def test_binary_search_matches_searchsorted():
    rng = np.random.default_rng(3)
    m = _random(n=500, g=300, density=0.3)
    lo, hi = m.indptr[:-1].astype(np.int64), m.indptr[1:].astype(np.int64)
    d = _reorder_rows(m, "descending")
    for value in (0, 1, 150, 299, 300, int(rng.integers(300))):
        got = mg._first(m.indices, lo, hi, value, descending=False)
        want = [a + np.searchsorted(m.indices[a:b], value) for a, b in zip(lo, hi)]
        np.testing.assert_array_equal(got, want)
        got = mg._first(d.indices, lo, hi, value, descending=True)
        want = [a + int((d.indices[a:b] >= value).sum()) for a, b in zip(lo, hi)]
        np.testing.assert_array_equal(got, want)


def test_layout_detection():
    m = _random()
    assert mg.MatrixGenes(m, range(m.shape[1])).layout == "ascending"
    assert mg.MatrixGenes(_reorder_rows(m, "descending"), range(m.shape[1])).layout == "descending"
    # descending but for one row: no order
    d = _reorder_rows(m, "descending")
    a = d.indptr[7]
    assert d.indptr[8] - a >= 2
    d.indices[a:a + 2] = d.indices[a:a + 2][::-1].copy()
    assert mg.MatrixGenes(d, range(m.shape[1])).layout == "copy"
    # a duplicate inside a descending row is not "descending"
    dup = _reorder_rows(m, "descending")
    a = dup.indptr[2]
    dup.indices[a + 1] = dup.indices[a]
    assert mg.MatrixGenes(dup, range(m.shape[1])).layout == "copy"


def test_duplicate_entries_are_refused():
    m = _random()
    a = m.indptr[4]
    dup = m.copy()
    dup.indices[a + 1] = dup.indices[a]  # sorted, with a duplicate
    dup.has_sorted_indices = False
    src = mg.MatrixGenes(dup, range(m.shape[1]))
    with pytest.raises(mg.DuplicateEntries):
        list(src.blocks(10**9))
    with pytest.raises(mg.DuplicateEntries):
        list(mg.MatrixGenes(sp.csc_matrix(dup), range(m.shape[1])).blocks(10**9))


def test_unsupported():
    m = _random()
    assert mg.unsupported(m) is None
    assert mg.unsupported(m.toarray()) is None
    assert "int64" in mg.unsupported(m.astype(np.int64))
    assert "float16" in mg.unsupported(m.toarray().astype(np.float16))
    assert "coo" in mg.unsupported(m.tocoo())
    assert "empty" in mg.unsupported(sp.csr_matrix((0, 5), dtype=np.float32))
    with pytest.raises(ValueError):
        mg.MatrixGenes(m.astype(np.int32), range(m.shape[1]))
    with pytest.raises(ValueError, match="gene names"):
        mg.MatrixGenes(m, range(3))
