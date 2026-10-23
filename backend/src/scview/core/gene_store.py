"""Gene-major (CSC) copy of a cell-major matrix on disk, for per-gene reads at scale.

A cell-major (CSR) matrix answers "one gene over all cells" by reading every row:
3.4 s in memory and 86 s backed at 1.3M cells, where the same column from a
gene-major copy took 7.8 ms (research/out-of-core, opt2). This module builds that
copy from a matrix that does not fit in memory and reads genes from it.

Build (``build`` / ``GeneStoreWriter``), memory bounded by one staged row chunk
plus one gene block, never the matrix:

1. Rows arrive in order (an h5ad X or layer read by row chunks, a dask array of
   row blocks, an in-memory matrix). Each staged chunk is cleaned (store dtype,
   explicit zeros dropped, canonical indices), transposed to CSC in memory
   (scipy's counting sort keeps rows ascending within a gene), and its cell
   indices and values are appended to two flat spill files. Per chunk only the
   per-gene cumulative counts are kept (n_genes + 1 int64).
2. With every gene's total known, the store's arrays are preallocated, and gene
   blocks of at most ``block_entries`` values are filled one at a time: from each
   chunk's spill segment, the slice of that block's genes (contiguous, since the
   segment is gene-major) is read and placed at its final position. The source
   is read once; each spill byte is written once and read once.

At 100k cells (201M values, read backed from a gzip h5ad, 2 cores) the build took
30 s at 0.80 GB peak RSS (the opt2 prototype: 53 s / 4.3 GB), and one gene over all
cells reads in 1.4 ms. Besides the store (0.77 GB there) a build needs temporary
disk beside it of 8 bytes per stored value (1.6 GB there; ~21 GB at 1.3M cells).

The store is written to a hidden build directory beside ``path`` and moved into
place (``os.replace``) only after it is complete and synced, so a cancelled or
failed build never leaves a file at ``path`` that looks like a store, and an old
store stays readable until the new one replaces it.

Layout (h5): ``data`` (values, float32 by default), ``indices`` (cell index per
value: int32, int64 only beyond 2^31 cells), ``indptr`` (n_genes + 1: int32, or
int64 once the store holds more than 2^31 values, as the full 1.3M-cell brain
does), ``var_names``; attrs ``format``, ``version``, ``shape``, ``nnz`` and the
fingerprint (cells, genes, values) that ``GeneStore.mismatch`` checks.
"""

from __future__ import annotations

import hashlib
import logging
import os
import secrets
import shutil
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

logger = logging.getLogger(__name__)

FORMAT = "scview-gene-store"
VERSION = 1

# Index arrays switch to int64 above this (a module constant so tests can lower it).
_INDEX_LIMIT = int(np.iinfo(np.int32).max)

# Values per staged row chunk (the build's working set is a few times this in bytes).
STAGE_ENTRIES = 16_000_000
# Values per gene block while filling the store, and per block a reader iterates.
BLOCK_ENTRIES = 32_000_000
# Values per HDF5 chunk: one small gene reads one or two chunks. Measured at 100k
# cells (201M values, 2 cores): 2^18 made a gene 4x slower to read than 2^16.
H5_CHUNK = 1 << 16
# Same measurement, file size / build / median read of one gene over all cells:
# lzf 1.56 GB / 29 s / 1.9 ms; lzf+shuffle 0.85 GB / 21 s / 1.8 ms; gzip-4+shuffle
# 0.77 GB / 30 s / 1.4 ms; none 1.61 GB / 12 s / 0.6 ms. gzip also needs no plugin
# outside h5py.
COMPRESSION = "gzip"
COMPRESSION_OPTS = 4
SHUFFLE = True
# Rows whose values are hashed into the fingerprint's cheap probe.
PROBE_ROWS = 64

Progress = Callable[[str, int, int], None]


class GeneStoreError(Exception):
    """A gene store that cannot be used: missing, incomplete or of another format."""


# ---------------------------------------------------------------------------
# fingerprints


def names_digest(names: Sequence[Any]) -> str:
    """Hash of a list of names (each name followed by a NUL, so order and splits count)."""
    h = hashlib.blake2b(digest_size=16)
    values = list(map(str, names))
    for start in range(0, len(values), 100_000):
        part = values[start:start + 100_000]
        h.update(("\x00".join(part) + "\x00").encode("utf-8"))
    h.update(str(len(values)).encode())
    return h.hexdigest()


class _ValuesHash:
    """Hash of a cleaned CSR matrix fed in row order, the same however it is chunked:
    row lengths, column indices and values go to three separate hashes."""

    def __init__(self) -> None:
        self._parts = [hashlib.blake2b(digest_size=16) for _ in range(3)]

    def update(self, m: sp.csr_matrix) -> None:
        self._parts[0].update(np.ascontiguousarray(np.diff(m.indptr), dtype="<i8"))
        self._parts[1].update(np.ascontiguousarray(m.indices, dtype="<i4"))
        self._parts[2].update(np.ascontiguousarray(m.data, dtype=m.data.dtype.newbyteorder("<")))

    def hexdigest(self) -> str:
        h = hashlib.blake2b(digest_size=16)
        for part in self._parts:
            h.update(part.digest())
        return h.hexdigest()


def probe_rows(n_cells: int) -> np.ndarray:
    """The rows whose values the fingerprint's probe hashes (evenly spaced, sorted)."""
    if n_cells <= 0:
        return np.zeros(0, dtype=np.int64)
    k = min(PROBE_ROWS, n_cells)
    return np.unique(np.linspace(0, n_cells - 1, num=k).round().astype(np.int64))


def _clean(block: Any, dtype: np.dtype, *, copy: bool = True) -> sp.csr_matrix:
    """``block`` as a canonical CSR in ``dtype`` without explicit zeros (a copy unless
    ``copy=False``, for a block nobody else holds)."""
    if hasattr(block, "compute") and not sp.issparse(block):  # a dask block
        block = block.compute()
    if sp.issparse(block):
        m = sp.csr_matrix(block, dtype=dtype, copy=copy)
    else:
        m = sp.csr_matrix(np.asarray(block, dtype=dtype))
    m.sum_duplicates()  # sorts indices, merges duplicates
    m.eliminate_zeros()  # including values the dtype cast underflowed to 0
    if m.indices.dtype != np.int32:
        m.indices = m.indices.astype(np.int32)
    return m


# ---------------------------------------------------------------------------
# reading a source matrix by rows


def _is_csc(x: Any) -> bool:
    fmt = getattr(x, "format", None)
    return fmt == "csc"


def _source_indptr(x: Any) -> np.ndarray | None:
    """The row pointer of a CSR source, without reading its values."""
    if sp.issparse(x) and x.format == "csr":
        return np.asarray(x.indptr)
    group = getattr(x, "group", None)
    if getattr(x, "format", None) == "csr" and group is not None and "indptr" in group:
        return group["indptr"][:]
    return None


def _source_nnz(x: Any) -> int | None:
    """Stored values of a sparse source (explicit zeros included), when cheap to know."""
    if sp.issparse(x):
        return int(x.nnz)
    group = getattr(x, "group", None)
    if group is not None and "data" in group:
        return int(group["data"].shape[0])
    return None


def row_chunks(x: Any, *, stage_entries: int = STAGE_ENTRIES) -> Iterator[tuple[int, Any]]:
    """(first row, block) over the rows of ``x`` in order, each block holding about
    ``stage_entries`` values: a scipy matrix, a dense array (in memory or h5py), an
    anndata sparse dataset (``anndata.io.sparse_dataset``, or a backed X), or a dask
    array of row blocks (``anndata.experimental.read_elem_lazy``)."""
    n_cells, n_genes = (int(s) for s in x.shape)
    if hasattr(x, "compute") and hasattr(x, "chunks") and not sp.issparse(x):  # dask
        start = 0
        for rows in x.chunks[0]:
            yield start, x[start:start + rows]
            start += rows
        return
    if sp.issparse(x) and x.format not in ("csr",):
        x = x.tocsr()
    indptr = _source_indptr(x)
    if indptr is not None:
        start = 0
        while start < n_cells:
            # as many rows as fit in stage_entries values, at least one
            limit = int(indptr[start]) + stage_entries
            stop = int(np.searchsorted(indptr, limit, side="right")) - 1
            stop = min(max(stop, start + 1), n_cells)
            yield start, x[start:stop]
            start = stop
        return
    rows = max(1, stage_entries // max(n_genes, 1))
    for start in range(0, n_cells, rows):
        yield start, x[start:min(start + rows, n_cells)]


# ---------------------------------------------------------------------------
# writing


def _sync_dir(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _drop_cache(f: Any, offset: int, length: int) -> None:
    """Tell the kernel a spill range will not be read again (keeps page cache, which
    the memory gauge counts while it is active, from growing with the spill)."""
    if length > 0 and hasattr(os, "posix_fadvise"):
        try:
            os.posix_fadvise(f.fileno(), offset, length, os.POSIX_FADV_DONTNEED)
        except OSError:
            pass


def _gene_blocks(indptr: np.ndarray, max_entries: int) -> list[tuple[int, int]]:
    """Contiguous gene ranges holding at most ``max_entries`` values (at least one gene)."""
    n_genes = len(indptr) - 1
    out = []
    start = 0
    while start < n_genes:
        limit = int(indptr[start]) + max_entries
        stop = int(np.searchsorted(indptr, limit, side="right")) - 1
        stop = min(max(stop, start + 1), n_genes)
        out.append((start, stop))
        start = stop
    return out


def build_dir_prefix(path: str | Path) -> str:
    """Name prefix of the hidden build directories of the store at ``path``."""
    return f".{Path(path).name}.build-"


def remove_stale_builds(path: str | Path) -> int:
    """Delete build directories left beside ``path`` by a process that died mid-build
    (its pid is gone). Returns how many were removed."""
    path = Path(path)
    removed = 0
    if not path.parent.is_dir():
        return 0
    prefix = build_dir_prefix(path)
    for entry in path.parent.iterdir():
        if not entry.name.startswith(prefix):
            continue
        try:
            pid = int(entry.name[len(prefix):].split("-")[0])
        except ValueError:
            pid = -1
        if pid == os.getpid() or _pid_alive(pid):
            continue
        shutil.rmtree(entry, ignore_errors=True)
        removed += 1
    return removed


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def remove(path: str | Path) -> None:
    """Delete the store at ``path`` and any build directory of it."""
    path = Path(path)
    path.unlink(missing_ok=True)
    if path.parent.is_dir():
        prefix = build_dir_prefix(path)
        for entry in path.parent.iterdir():
            if entry.name.startswith(prefix):
                shutil.rmtree(entry, ignore_errors=True)


class GeneStoreWriter:
    """Builds a store from rows pushed in order (``add``), then ``finish``.

    Use as a context manager: an exception inside the block (a cancelled run
    included) aborts the build and removes everything it wrote.

    ``progress(message, done, total)`` is called between pieces of work; it may raise
    to stop the build (the pipeline's substep callback raises ``RunCancelled``).
    """

    def __init__(
        self,
        path: str | Path,
        *,
        obs_names: Sequence[Any],
        var_names: Sequence[Any],
        dtype: Any = np.float32,
        compression: str | None = COMPRESSION,
        compression_opts: Any = COMPRESSION_OPTS,
        shuffle: bool = SHUFFLE,
        source: str = "X",
        source_nnz: int | None = None,
        stage_entries: int = STAGE_ENTRIES,
        block_entries: int = BLOCK_ENTRIES,
        progress: Progress | None = None,
    ) -> None:
        self.path = Path(path)
        self.n_cells = len(obs_names)
        self.var_names = [str(v) for v in var_names]
        self.n_genes = len(self.var_names)
        self.dtype = np.dtype(dtype)
        if self.dtype.kind != "f":
            raise ValueError(f"A gene store holds floating-point values, not {self.dtype}")
        self.compression = compression
        self.compression_opts = compression_opts
        self.shuffle = shuffle
        self.source = source
        self.source_nnz = source_nnz
        self.stage_entries = int(stage_entries)
        self.block_entries = int(block_entries)
        self._progress = progress
        self._cells_digest = names_digest(obs_names)
        self._genes_digest = names_digest(self.var_names)
        self._cell_dtype = np.int64 if self.n_cells > _INDEX_LIMIT else np.int32
        self._hash = _ValuesHash()
        self._probe_rows = probe_rows(self.n_cells)
        self._probe_parts: list[sp.csr_matrix] = []
        self._staged: list[sp.csr_matrix] = []
        self._staged_entries = 0
        self._rows_seen = 0
        self._spill_start = 0  # first row of the staged rows
        self._chunk_cum: list[np.ndarray] = []
        self._chunk_offset: list[int] = []
        self._spilled = 0
        self._done = False
        remove_stale_builds(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._dir = self.path.parent / f"{build_dir_prefix(self.path)}{os.getpid()}-{secrets.token_hex(4)}"
        self._dir.mkdir()
        try:
            self._rows_f = open(self._dir / "rows.bin", "w+b")
            self._vals_f = open(self._dir / "values.bin", "w+b")
        except BaseException:
            shutil.rmtree(self._dir, ignore_errors=True)
            raise

    # -- context manager
    def __enter__(self) -> GeneStoreWriter:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is not None or not self._done:
            self.abort()

    def _tick(self, message: str, done: int, total: int) -> None:
        if self._progress is not None:
            self._progress(message, done, total)

    def abort(self) -> None:
        """Remove everything this build wrote; ``path`` is left as it was."""
        for f in (getattr(self, "_rows_f", None), getattr(self, "_vals_f", None)):
            if f is not None and not f.closed:
                f.close()
        shutil.rmtree(self._dir, ignore_errors=True)
        self._done = True

    # -- phase 1: rows in, spilled gene-major per chunk
    def add(self, block: Any, *, copy: bool = True) -> None:
        """Append the next rows (any 2-D matrix with ``n_genes`` columns). ``copy=False``
        lets the build canonicalise a sparse block in place (one it alone holds)."""
        if self._done:
            raise RuntimeError("This gene store build is finished or aborted")
        m = _clean(block, self.dtype, copy=copy)
        if m.shape[1] != self.n_genes:
            raise ValueError(f"Rows have {m.shape[1]} columns, the store {self.n_genes} genes")
        start = self._rows_seen
        if start + m.shape[0] > self.n_cells:
            raise ValueError(f"More rows than the {self.n_cells} cells the store was made for")
        self._rows_seen += m.shape[0]
        self._hash.update(m)
        lo, hi = np.searchsorted(self._probe_rows, [start, start + m.shape[0]])
        if hi > lo:
            self._probe_parts.append(m[self._probe_rows[lo:hi] - start])
        self._staged.append(m)
        self._staged_entries += m.nnz
        if self._staged_entries >= self.stage_entries:
            self._spill()

    def _spill(self) -> None:
        if not self._staged:
            return
        m = self._staged[0] if len(self._staged) == 1 else sp.vstack(self._staged, format="csr")
        self._staged, self._staged_entries = [], 0
        csc = m.tocsc()  # counting sort: cells stay ascending within each gene
        del m
        offset = self._spilled
        rows = csc.indices.astype(self._cell_dtype)
        rows += self._spill_start
        rows.tofile(self._rows_f)
        np.ascontiguousarray(csc.data, dtype=self.dtype).tofile(self._vals_f)
        self._chunk_cum.append(csc.indptr.astype(np.int64))
        self._chunk_offset.append(offset)
        self._spilled += int(csc.nnz)
        self._spill_start = self._rows_seen
        self._tick("Gene-major copy: reading cells", self._rows_seen, self.n_cells)

    # -- phase 2: fill the store gene block by gene block
    def finish(self) -> Path:
        """Write the store and move it to ``path``; returns ``path``."""
        if self._done:
            raise RuntimeError("This gene store build is finished or aborted")
        try:
            self._spill()
            if self._rows_seen != self.n_cells:
                raise ValueError(f"Got {self._rows_seen} rows for a store of {self.n_cells} cells")
            self._rows_f.flush()
            self._vals_f.flush()
            counts = np.zeros(self.n_genes, dtype=np.int64)
            for cum in self._chunk_cum:
                counts += np.diff(cum)
            indptr = np.zeros(self.n_genes + 1, dtype=np.int64)
            np.cumsum(counts, out=indptr[1:])
            nnz = int(indptr[-1])
            partial = self._dir / "store.h5"
            with h5py.File(partial, "w", rdcc_nbytes=16 << 20) as f:
                self._write(f, indptr, nnz)
            self._rows_f.close()
            self._vals_f.close()
            with open(partial, "rb") as fh:
                os.fsync(fh.fileno())
            os.replace(partial, self.path)
            _sync_dir(self.path.parent)
        except BaseException:
            self.abort()
            raise
        shutil.rmtree(self._dir, ignore_errors=True)
        self._done = True
        return self.path

    def _write(self, f: h5py.File, indptr: np.ndarray, nnz: int) -> None:
        f.attrs["format"] = FORMAT
        f.attrs["version"] = VERSION
        f.attrs["shape"] = np.array([self.n_cells, self.n_genes], dtype=np.int64)
        f.attrs["nnz"] = nnz
        f.attrs["dtype"] = self.dtype.str
        kw: dict[str, Any] = {}
        if nnz > 0:
            kw = {"chunks": (min(H5_CHUNK, nnz),), "compression": self.compression,
                  "compression_opts": self.compression_opts, "shuffle": self.shuffle}
        data = f.create_dataset("data", shape=(nnz,), dtype=self.dtype, **kw)
        index = f.create_dataset("indices", shape=(nnz,), dtype=self._cell_dtype, **kw)
        ptr_dtype = np.int64 if nnz > _INDEX_LIMIT else np.int32
        f.create_dataset("indptr", data=indptr.astype(ptr_dtype))
        f.create_dataset("var_names", data=np.array(self.var_names, dtype=object),
                         dtype=h5py.string_dtype("utf-8"))
        blocks = _gene_blocks(indptr, self.block_entries) if nnz else []
        cell_size = np.dtype(self._cell_dtype).itemsize
        val_size = self.dtype.itemsize
        for i, (g0, g1) in enumerate(blocks):
            p0, p1 = int(indptr[g0]), int(indptr[g1])
            n = p1 - p0
            if n == 0:
                continue
            buf_rows = np.empty(n, dtype=self._cell_dtype)
            buf_vals = np.empty(n, dtype=self.dtype)
            # where each gene's next value goes, within the block
            cursor = indptr[g0:g1] - p0
            for cum, offset in zip(self._chunk_cum, self._chunk_offset):
                a, b = int(cum[g0]), int(cum[g1])
                if b == a:
                    continue
                per_gene = np.diff(cum[g0:g1 + 1])
                self._rows_f.seek((offset + a) * cell_size)
                seg_rows = np.fromfile(self._rows_f, dtype=self._cell_dtype, count=b - a)
                self._vals_f.seek((offset + a) * val_size)
                seg_vals = np.fromfile(self._vals_f, dtype=self.dtype, count=b - a)
                _drop_cache(self._rows_f, (offset + a) * cell_size, (b - a) * cell_size)
                _drop_cache(self._vals_f, (offset + a) * val_size, (b - a) * val_size)
                # the segment is gene-major: gene j's values start at cum[j] - a
                dest = np.repeat(cursor - (cum[g0:g1] - a), per_gene)
                dest += np.arange(b - a, dtype=np.int64)
                buf_rows[dest] = seg_rows
                buf_vals[dest] = seg_vals
                cursor += per_gene
            data[p0:p1] = buf_vals
            index[p0:p1] = buf_rows
            del buf_rows, buf_vals
            self._tick("Gene-major copy: writing genes", g1, self.n_genes)
        # the fingerprint goes in last
        probe = _ValuesHash()
        if self._probe_parts:
            probe.update(sp.vstack(self._probe_parts, format="csr"))
        f.attrs["cells_digest"] = self._cells_digest
        f.attrs["genes_digest"] = self._genes_digest
        f.attrs["values_digest"] = self._hash.hexdigest()
        f.attrs["probe_digest"] = probe.hexdigest()
        f.attrs["source"] = self.source
        f.attrs["source_nnz"] = -1 if self.source_nnz is None else int(self.source_nnz)


def build(
    x: Any,
    path: str | Path,
    *,
    obs_names: Sequence[Any],
    var_names: Sequence[Any],
    dtype: Any = np.float32,
    source: str = "X",
    compression: str | None = COMPRESSION,
    compression_opts: Any = COMPRESSION_OPTS,
    shuffle: bool = SHUFFLE,
    stage_entries: int = STAGE_ENTRIES,
    block_entries: int = BLOCK_ENTRIES,
    progress: Progress | None = None,
) -> Path:
    """Build the gene store of the cell-major matrix ``x`` at ``path`` (see the module
    docstring). ``x`` is read once, by row chunks; ``source`` names what ``x`` is
    ("X", "raw", "layers/<name>") for ``GeneStore.mismatch``."""
    if isinstance(x, h5py.Group):  # an h5ad X / layer group
        import anndata

        x = anndata.io.sparse_dataset(x)
    if len(obs_names) != x.shape[0] or len(var_names) != x.shape[1]:
        raise ValueError(f"Names ({len(obs_names)} cells, {len(var_names)} genes) do not fit "
                         f"the matrix shape {tuple(x.shape)}")
    if _is_csc(x):
        if not sp.issparse(x):
            raise ValueError("A backed CSC matrix cannot be read by rows; load it or "
                             "convert it to CSR first")
        x = x.tocsr()
    with GeneStoreWriter(
        path, obs_names=obs_names, var_names=var_names, dtype=dtype, source=source,
        source_nnz=_source_nnz(x), compression=compression, compression_opts=compression_opts,
        shuffle=shuffle, stage_entries=stage_entries, block_entries=block_entries,
        progress=progress,
    ) as writer:
        for _start, block in row_chunks(x, stage_entries=stage_entries):
            writer.add(block, copy=False)  # a fresh slice of x
        return writer.finish()


def matrix_of(adata: Any, source: str = "X") -> tuple[Any, Sequence[Any], Sequence[Any]]:
    """(matrix, obs_names, var_names) of ``adata`` for a source name ("X", "raw",
    "layers/<name>")."""
    if source == "X":
        return adata.X, adata.obs_names, adata.var_names
    if source == "raw":
        if adata.raw is None:
            raise ValueError("The dataset has no .raw")
        return adata.raw.X, adata.obs_names, adata.raw.var_names
    if source.startswith("layers/"):
        name = source[len("layers/"):]
        return adata.layers[name], adata.obs_names, adata.var_names
    raise ValueError(f"Unknown gene store source {source!r} (X, raw or layers/<name>)")


def build_from_adata(adata: Any, path: str | Path, *, source: str = "X", **kwargs: Any) -> Path:
    """``build`` for ``adata``'s X, .raw or a layer (``source``)."""
    x, obs_names, var_names = matrix_of(adata, source)
    return build(x, path, obs_names=obs_names, var_names=var_names, source=source, **kwargs)


# ---------------------------------------------------------------------------
# reading


class GeneStore:
    """A gene store opened for reading. Thread-safe as h5py is (calls serialise)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise GeneStoreError(f"No gene store at {self.path}")
        try:
            self._f = h5py.File(self.path, "r", rdcc_nbytes=16 << 20, rdcc_nslots=10007)
        except OSError as exc:
            raise GeneStoreError(f"Cannot open the gene store {self.path}: {exc}") from exc
        try:
            attrs = self._f.attrs
            if attrs.get("format") != FORMAT:
                raise GeneStoreError(f"{self.path} is not a gene store")
            if int(attrs.get("version", -1)) != VERSION:
                raise GeneStoreError(f"{self.path} is gene store version {attrs.get('version')}, "
                                     f"this code reads {VERSION}")
            if "values_digest" not in attrs:
                raise GeneStoreError(f"{self.path} is an incomplete gene store")
            self.n_cells, self.n_genes = (int(v) for v in attrs["shape"])
            self.nnz = int(attrs["nnz"])
            self.dtype = np.dtype(str(attrs["dtype"]))
            self.indptr = self._f["indptr"][:].astype(np.int64)
            self._data = self._f["data"]
            self._indices = self._f["indices"]
            self.var_names = pd.Index(self._f["var_names"].asstr()[:])
            self.fingerprint = {
                "n_cells": self.n_cells, "n_genes": self.n_genes, "nnz": self.nnz,
                "cells_digest": str(attrs["cells_digest"]),
                "genes_digest": str(attrs["genes_digest"]),
                "values_digest": str(attrs["values_digest"]),
                "probe_digest": str(attrs["probe_digest"]),
                "source": str(attrs["source"]),
                "source_nnz": int(attrs["source_nnz"]),
                "dtype": self.dtype.str,
            }
        except BaseException:
            self._f.close()
            raise

    @classmethod
    def open(cls, path: str | Path) -> GeneStore:
        return cls(path)

    def close(self) -> None:
        if self._f.id.valid:
            self._f.close()

    def __enter__(self) -> GeneStore:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    @property
    def shape(self) -> tuple[int, int]:
        return self.n_cells, self.n_genes

    @property
    def source(self) -> str:
        return self.fingerprint["source"]

    # -- genes
    def positions(self, genes: Any) -> np.ndarray:
        """Column positions of ``genes`` (names, or integer positions passed through).
        Raises KeyError naming the genes the store does not have."""
        if isinstance(genes, (str, int, np.integer)):
            genes = [genes]
        arr = np.asarray(genes)
        if arr.dtype.kind in "iu":
            pos = arr.astype(np.int64)
            bad = (pos < 0) | (pos >= self.n_genes)
            if bad.any():
                raise KeyError(f"Gene positions out of range: {pos[bad][:5].tolist()}")
            return pos
        pos = self.var_names.get_indexer([str(g) for g in arr])
        if (pos < 0).any():
            missing = [str(g) for g, p in zip(arr, pos) if p < 0]
            raise KeyError(f"Genes not in the gene store: {', '.join(missing[:5])}")
        return pos.astype(np.int64)

    def column(self, gene: Any) -> tuple[np.ndarray, np.ndarray]:
        """(cell indices, values) of one gene's stored values, cells ascending."""
        j = int(self.positions(gene)[0])
        p0, p1 = int(self.indptr[j]), int(self.indptr[j + 1])
        if p1 == p0:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=self.dtype)
        return self._indices[p0:p1], self._data[p0:p1]

    def gene(self, gene: Any, cells: Any = None) -> np.ndarray:
        """One gene's values as a dense vector over all cells, or over ``cells``
        (integer positions or a boolean mask, in the order given)."""
        rows, vals = self.column(gene)
        out = np.zeros(self.n_cells, dtype=self.dtype)
        out[rows] = vals
        return out if cells is None else out[_cell_index(cells, self.n_cells)]

    def genes(self, genes: Any, cells: Any = None) -> sp.csc_matrix:
        """Several genes as a CSC block (cells x genes, in the order given), over all
        cells or ``cells`` (unique integer positions or a boolean mask)."""
        pos = self.positions(genes)
        cols = self._read_columns(pos)
        sizes = np.array([len(r) for r, _ in cols], dtype=np.int64)
        indptr = np.zeros(len(pos) + 1, dtype=np.int64)
        np.cumsum(sizes, out=indptr[1:])
        idx_dtype = np.int32 if max(int(indptr[-1]), self.n_cells) <= _INDEX_LIMIT else np.int64
        rows = (np.concatenate([r for r, _ in cols]).astype(idx_dtype, copy=False)
                if cols else np.zeros(0, idx_dtype))
        vals = (np.concatenate([v for _, v in cols]) if cols else np.zeros(0, self.dtype))
        m = sp.csc_matrix((vals, rows, indptr.astype(idx_dtype)), shape=(self.n_cells, len(pos)))
        if cells is None:
            return m
        return _subset_rows(m, _cell_index(cells, self.n_cells))

    def _read_columns(self, pos: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        """Columns at ``pos``; nearby columns are read in one slice per dataset."""
        if len(pos) == 0:
            return []
        order = np.unique(pos)
        cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        i = 0
        while i < len(order):
            j = i
            # extend while the gap to the next wanted gene is small
            while (j + 1 < len(order)
                   and self.indptr[order[j + 1]] - self.indptr[order[j] + 1] <= H5_CHUNK):
                j += 1
            p0, p1 = int(self.indptr[order[i]]), int(self.indptr[order[j] + 1])
            rows = self._indices[p0:p1] if p1 > p0 else np.zeros(0, np.int64)
            vals = self._data[p0:p1] if p1 > p0 else np.zeros(0, self.dtype)
            for g in order[i:j + 1]:
                a, b = int(self.indptr[g]) - p0, int(self.indptr[g + 1]) - p0
                cache[int(g)] = (rows[a:b], vals[a:b])
            i = j + 1
        return [cache[int(g)] for g in pos]

    def blocks(self, max_entries: int = BLOCK_ENTRIES,
               start: int = 0, stop: int | None = None) -> Iterator[tuple[int, int, sp.csc_matrix]]:
        """(first gene, end gene, CSC block) over contiguous gene ranges of at most
        ``max_entries`` values (one gene can exceed it), one read per dataset each."""
        stop = self.n_genes if stop is None else stop
        ranges = _gene_blocks(self.indptr[start:stop + 1] - self.indptr[start], max_entries)
        for a, b in ranges:
            g0, g1 = start + a, start + b
            p0, p1 = int(self.indptr[g0]), int(self.indptr[g1])
            local = self.indptr[g0:g1 + 1] - p0
            idx_dtype = np.int32 if max(p1 - p0, self.n_cells) <= _INDEX_LIMIT else np.int64
            rows = (self._indices[p0:p1] if p1 > p0 else np.zeros(0, idx_dtype)).astype(idx_dtype, copy=False)
            vals = self._data[p0:p1] if p1 > p0 else np.zeros(0, self.dtype)
            yield g0, g1, sp.csc_matrix((vals, rows, local.astype(idx_dtype)),
                                        shape=(self.n_cells, g1 - g0))

    # -- staleness
    def mismatch(self, data: Any, *, source: str | None = None, full: bool = False) -> str | None:
        """Why this store does not hold the matrix it stands for in ``data``, or None.

        ``data`` is an AnnData (its X, .raw or layer by ``source``, default the source
        the store was built from), a ``Raw``, or a tuple (matrix, obs_names, var_names).
        Checks shape, cell and gene names, the source's stored-value count when cheap
        to read, and the values of the probe rows; ``full`` re-hashes every value
        (one pass over the matrix, bounded memory)."""
        x, obs_names, var_names = _resolve(data, source or self.source)
        fp = self.fingerprint
        if tuple(int(s) for s in x.shape) != (self.n_cells, self.n_genes):
            return (f"the store has {self.n_cells} cells x {self.n_genes} genes, the data "
                    f"{x.shape[0]} x {x.shape[1]}")
        if names_digest(var_names) != fp["genes_digest"]:
            return "the genes differ"
        if names_digest(obs_names) != fp["cells_digest"]:
            return "the cells differ"
        nnz = _source_nnz(x)
        if nnz is not None and fp["source_nnz"] >= 0 and nnz != fp["source_nnz"]:
            return f"the data has {nnz} stored values, the store was built from {fp['source_nnz']}"
        rows = probe_rows(self.n_cells)
        probe = _ValuesHash()
        if len(rows):
            probe.update(_clean(_take_rows(x, rows), self.dtype))
        if probe.hexdigest() != fp["probe_digest"]:
            return "the values differ (probe rows)"
        if full:
            h = _ValuesHash()
            for _start, block in row_chunks(x):
                h.update(_clean(block, self.dtype))
            if h.hexdigest() != fp["values_digest"]:
                return "the values differ"
        return None

    def matches(self, data: Any, *, source: str | None = None, full: bool = False) -> bool:
        return self.mismatch(data, source=source, full=full) is None


def open_store(path: str | Path) -> GeneStore:
    return GeneStore(path)


def _resolve(data: Any, source: str) -> tuple[Any, Sequence[Any], Sequence[Any]]:
    if isinstance(data, tuple):
        return data
    if hasattr(data, "layers") and hasattr(data, "raw"):  # AnnData
        return matrix_of(data, source)
    return data.X, data.obs_names, data.var_names  # Raw


def _take_rows(x: Any, rows: np.ndarray) -> Any:
    if hasattr(x, "compute") and not sp.issparse(x):  # dask
        return x[rows].compute()
    if sp.issparse(x) and x.format != "csr":
        x = x.tocsr()
    return x[rows]


def _cell_index(cells: Any, n_cells: int) -> np.ndarray:
    arr = np.asarray(cells)
    if arr.dtype == bool:
        if arr.shape != (n_cells,):
            raise ValueError(f"A cell mask needs {n_cells} values, got {arr.shape}")
        return np.flatnonzero(arr)
    return arr.astype(np.int64, copy=False)


def _subset_rows(m: sp.csc_matrix, cells: np.ndarray) -> sp.csc_matrix:
    """Rows ``cells`` (unique, any order) of a CSC block, as CSC."""
    n = m.shape[0]
    new = np.full(n, -1, dtype=np.int64)
    new[cells] = np.arange(len(cells))
    if np.count_nonzero(new >= 0) != len(cells):  # repeated cells: scipy indexing
        return sp.csc_matrix(m.tocsr()[cells])
    mapped = new[m.indices]
    keep = mapped >= 0
    column = np.repeat(np.arange(m.shape[1]), np.diff(m.indptr))
    counts = np.bincount(column[keep], minlength=m.shape[1])
    indptr = np.zeros(m.shape[1] + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])
    out = sp.csc_matrix((m.data[keep], mapped[keep], indptr), shape=(len(cells), m.shape[1]))
    out.sort_indices()
    return out
