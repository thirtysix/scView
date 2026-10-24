"""Blocks of genes of a matrix in memory, read the way ``sparse_markers`` reads a gene store.

``sparse_markers`` ranks genes block by block from a source with ``n_cells``,
``n_genes``, ``var_names``, ``nnz`` and ``blocks(max_entries)``: (first gene, end gene,
a CSC block of cells x genes with each gene's cells ascending and no stored zeros).
``GeneStore`` (core/gene_store.py) is that source on disk; ``MatrixGenes`` is the same
over a matrix in memory, in its own dtype (float32 stays float32, float64 float64, so
the sums are the ones scanpy takes):

- CSR (cells x genes) whose rows hold their genes in order, ascending (scipy's
  canonical form) or strictly descending (10x's 2017 1.3M-neuron matrix): a block of
  genes is one segment of every row. The segments' ends are found by a binary search
  over all rows at once (log2 of the longest row steps over n_cells), the block's
  values gathered and put in gene order by scipy's counting sort (cells stay
  ascending). Every value is read once over all blocks; memory is one block (about 40
  bytes per value) and a few arrays of n_cells, never a copy of the matrix.
- CSR in no order (a merge can leave rows so): each value's block of genes is noted
  once (one byte per value), then each block's values are found by one pass over those
  bytes. Converting to CSC instead would hold a copy (8 bytes per value at float32 /
  int32: 1.6 GB for the 201M values of 100k neurons).
- CSC: a slice of columns, each put in canonical form. Dense: a slice of columns, its
  non-zeros.

A stored duplicate (two values for one cell and gene) stops the read with
``DuplicateEntries``: scanpy counts both in ``pts``, so scanpy ranks that matrix. No
layout copies the matrix: the working memory is one block, a few arrays of n_cells, and
for rows in no order one byte per stored value.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp

from scview.core.gene_store import _gene_blocks

# Values per block, as sparse_markers.MAX_ENTRIES (a dense block counts every element);
# a smaller matrix in blocks of an eighth of its values, at least MIN_BLOCK_ENTRIES: a
# block's working set is about 70 bytes per value, and in 4M-value blocks Kang's markers
# (9.6M values) took +0.34 GiB, more than scanpy's +0.22.
BLOCK_ENTRIES = 4_000_000
MIN_BLOCK_ENTRIES = 500_000


def block_entries(nnz: int) -> int:
    """Values per block for a matrix of ``nnz`` stored values."""
    return int(min(BLOCK_ENTRIES, max(MIN_BLOCK_ENTRIES, nnz // 8)))
# Stored values per pass when counting, checking or finding a block (bincount and the
# order check take them as int64: 32 MB a pass).
_CHUNK = 4_000_000
FLOATS = (np.dtype(np.float32), np.dtype(np.float64))


class DuplicateEntries(ValueError):
    """The matrix stores two values for one cell and gene."""


def unsupported(matrix: Any) -> str | None:
    """Why ``MatrixGenes`` cannot read ``matrix`` (scanpy ranks it instead), or None."""
    if sp.issparse(matrix):
        if matrix.format not in ("csr", "csc"):
            return f"a {matrix.format} matrix"
    elif not isinstance(matrix, np.ndarray):
        return f"a {type(matrix).__name__} matrix"
    elif matrix.ndim != 2:
        return "not a 2-D matrix"
    if np.dtype(matrix.dtype) not in FLOATS:
        return f"{matrix.dtype} values"
    if matrix.shape[0] < 1 or matrix.shape[1] < 1:
        return "an empty matrix"
    return None


def _descending(m: sp.csr_matrix) -> bool:
    """Whether every row's column indices fall strictly (a chunk of values at a time)."""
    indptr, indices = m.indptr, m.indices
    n = m.shape[0]
    r0 = 0
    while r0 < n:
        r1 = int(np.searchsorted(indptr, indptr[r0] + _CHUNK, side="right")) - 1
        r1 = min(max(r1, r0 + 1), n)
        p0, p1 = int(indptr[r0]), int(indptr[r1])
        if p1 - p0 > 1:
            d = np.diff(indices[p0:p1].astype(np.int64))
            # the step from one row's last value to the next row's first does not count
            cut = np.asarray(indptr[r0 + 1:r1], dtype=np.int64) - p0 - 1
            cut = cut[(cut >= 0) & (cut < d.size)]
            d[cut] = -1
            if (d >= 0).any():
                return False
        r0 = r1
    return True


class MatrixGenes:
    """A matrix in memory (cells x genes) read by blocks of genes."""

    def __init__(self, matrix: Any, var_names: Sequence[Any]) -> None:
        reason = unsupported(matrix)
        if reason is not None:
            raise ValueError(f"Cannot read genes in blocks from {reason}.")
        self.matrix = matrix
        self.n_cells, self.n_genes = (int(s) for s in matrix.shape)
        self.var_names = pd.Index(var_names)
        if len(self.var_names) != self.n_genes:
            raise ValueError(f"{len(self.var_names)} gene names for {self.n_genes} genes")
        self.dtype = np.dtype(matrix.dtype)
        self.nnz = int(matrix.nnz) if sp.issparse(matrix) else self.n_cells * self.n_genes
        # how blocks are read: dense | csc | ascending | descending | unordered
        if not sp.issparse(matrix):
            self.layout = "dense"
        elif matrix.format == "csc":
            self.layout = "csc"
        elif matrix.has_canonical_format:
            self.layout = "ascending"
        elif _descending(matrix):
            self.layout = "descending"
        else:
            self.layout = "unordered"
        self._cum: np.ndarray | None = None  # stored values up to each gene (CSR)
        # scanpy's mean/variance kernel for a CSC matrix squares in float64, for CSR and
        # dense in the value dtype (sparse_markers squares as scanpy would)
        self.square_dtype = np.dtype(np.float64) if self.layout == "csc" else self.dtype

    @property
    def shape(self) -> tuple[int, int]:
        return self.n_cells, self.n_genes

    def blocks(self, max_entries: int = BLOCK_ENTRIES) -> Iterator[tuple[int, int, sp.csc_matrix]]:
        """(first gene, end gene, CSC block) over contiguous gene ranges of at most
        ``max_entries`` stored values (one gene can exceed it); explicit zeros dropped,
        each gene's cells ascending."""
        if self.layout == "dense":
            yield from self._dense_blocks(max_entries)
        elif self.layout == "csc":
            yield from _csc_blocks(self.matrix, max_entries)
        elif self.layout == "unordered":
            yield from self._unordered_blocks(max_entries)
        else:
            yield from self._csr_blocks(max_entries, descending=self.layout == "descending")

    # -- dense
    def _dense_blocks(self, max_entries: int) -> Iterator[tuple[int, int, sp.csc_matrix]]:
        cols = max(1, max_entries // self.n_cells)
        for g0 in range(0, self.n_genes, cols):
            g1 = min(g0 + cols, self.n_genes)
            # scipy takes the non-zeros row by row, then sorts them into columns stably
            yield g0, g1, sp.csc_matrix(np.asarray(self.matrix[:, g0:g1]))

    # -- CSR
    def _gene_ranges(self, max_entries: int) -> list[tuple[int, int]]:
        """Gene ranges of at most ``max_entries`` stored values (explicit zeros included),
        each gene's values counted a chunk of values at a time (once: a reference test
        reads the blocks once per group)."""
        if self._cum is None:
            idx = self.matrix.indices
            cum = np.zeros(self.n_genes + 1, dtype=np.int64)
            for start in range(0, idx.size, _CHUNK):
                cum[1:] += np.bincount(idx[start:start + _CHUNK], minlength=self.n_genes)
            self._cum = np.cumsum(cum)
        return _gene_blocks(self._cum, max_entries)

    def _csr_blocks(self, max_entries: int, *, descending: bool
                    ) -> Iterator[tuple[int, int, sp.csc_matrix]]:
        m = self.matrix
        indices, data = m.indices, m.data
        row_start = np.asarray(m.indptr[:-1], dtype=np.int64)
        row_end = np.asarray(m.indptr[1:], dtype=np.int64)
        cell_dtype = np.int32 if self.n_cells <= np.iinfo(np.int32).max else np.int64
        cells = np.arange(self.n_cells, dtype=cell_dtype)
        # Each row's values not yet read lie in [start, stop): blocks are taken in gene
        # order, from the front of ascending rows and from the back of descending ones.
        start, stop = row_start, row_end
        for g0, g1 in self._gene_ranges(max_entries):
            last = g1 == self.n_genes
            if descending:
                lo = row_start if last else _first(indices, row_start, stop, g1, descending=True)
                seg = (lo, stop)
                stop = lo
            else:
                hi = row_end if last else _first(indices, start, row_end, g1, descending=False)
                seg = (start, hi)
                start = hi
            yield g0, g1, self._gather(indices, data, cells, *seg, g0, g1)

    def _gather(self, indices, data, cells, lo, hi, g0, g1) -> sp.csc_matrix:
        """The values at positions lo[r] .. hi[r] of every row r as a CSC block."""
        lengths = hi - lo
        k = int(lengths.sum())
        first = np.cumsum(lengths) - lengths
        pos = np.repeat(lo - first, lengths)
        pos += np.arange(k, dtype=np.int64)
        return self._block(data[pos], indices[pos] - g0, np.repeat(cells, lengths), g0, g1)

    def _unordered_blocks(self, max_entries: int) -> Iterator[tuple[int, int, sp.csc_matrix]]:
        m = self.matrix
        indices, data = m.indices, m.data
        indptr = np.asarray(m.indptr, dtype=np.int64)
        ranges = self._gene_ranges(max_entries)
        n = len(ranges)
        code = np.uint8 if n <= 1 << 8 else np.uint16 if n <= 1 << 16 else np.uint32
        of_gene = np.repeat(np.arange(n, dtype=code), [g1 - g0 for g0, g1 in ranges])
        block_of = np.empty(indices.size, dtype=code)
        for s in range(0, indices.size, _CHUNK):
            block_of[s:s + _CHUNK] = of_gene[indices[s:s + _CHUNK]]
        for b, (g0, g1) in enumerate(ranges):
            # positions ascending, so rows ascending
            pos = np.concatenate([np.flatnonzero(block_of[s:s + _CHUNK] == b) + s
                                  for s in range(0, block_of.size, _CHUNK)]
                                 or [np.zeros(0, dtype=np.int64)])
            rows = np.searchsorted(indptr, pos, side="right") - 1
            yield g0, g1, self._block(data[pos], indices[pos] - g0, rows, g0, g1)

    def _block(self, vals, genes, rows, g0, g1) -> sp.csc_matrix:
        """Values with their (block-local) genes and cells, cells ascending, as a CSC
        block without zeros."""
        nonzero = vals != 0
        if not nonzero.all():
            vals, genes, rows = vals[nonzero], genes[nonzero], rows[nonzero]
        if vals.size == 0:
            return sp.csc_matrix((self.n_cells, g1 - g0), dtype=self.dtype)
        # genes x cells, then read as cells x genes: a gene's cells stay ascending
        gm = sp.coo_matrix((vals, (genes, rows)), shape=(g1 - g0, self.n_cells)).tocsr()
        if gm.nnz != vals.size:  # scipy summed two values of one cell and gene
            raise DuplicateEntries("The matrix stores duplicate entries.")
        return sp.csc_matrix((gm.data, gm.indices, gm.indptr), shape=(self.n_cells, g1 - g0))


def _csc_blocks(m: sp.csc_matrix, max_entries: int) -> Iterator[tuple[int, int, sp.csc_matrix]]:
    for g0, g1 in _gene_blocks(np.asarray(m.indptr, dtype=np.int64), max_entries):
        block = sp.csc_matrix(m[:, g0:g1])
        if np.shares_memory(block.data, m.data):  # cleaned below: never the caller's arrays
            block = block.copy()
        if not block.has_canonical_format:
            block.sort_indices()
            # scipy keeps the cached "not canonical" after sorting: ask a fresh wrapper
            fresh = sp.csc_matrix((block.data, block.indices, block.indptr), shape=block.shape)
            if not fresh.has_canonical_format:
                raise DuplicateEntries("The matrix stores duplicate entries.")
            block = fresh
        block.eliminate_zeros()
        yield g0, g1, block


def _first(indices: np.ndarray, lo: np.ndarray, hi: np.ndarray, value: int, *,
           descending: bool) -> np.ndarray:
    """Per row, the first position in [lo, hi) past the column indices on the near side of
    ``value`` (below it in an ascending row, at or above it in a descending one), or hi:
    a binary search over every row at once."""
    lo = lo.copy()
    hi = hi.copy()
    active = np.flatnonzero(lo < hi)
    while active.size:
        a, b = lo[active], hi[active]
        mid = (a + b) >> 1
        near = indices[mid] >= value if descending else indices[mid] < value
        lo[active[near]] = mid[near] + 1
        hi[active[~near]] = mid[~near]
        active = active[lo[active] < hi[active]]
    return lo
