"""Per-cell matrices that are too big to live in a dataset's h5ad: ATAC peak counts,
spliced and unspliced counts.

A file in obsm or layers is read into memory on every open and rewritten on every save.
10x PBMC multiome has about 100M non-zero peak values, so these matrices sit in their own
file, on disk, and are read by row:

    DATA_DIR/modalities/<dataset id>/<name>.h5

The store is written once and never follows the dataset through a run. It keeps every
barcode it arrived with and is addressed by barcode, so filtering cells needs no rewrite:
ask ``rows_of(adata.obs_names)``, which gives -1 for a cell the store does not hold. The
folder goes with the dataset (``DatasetManager.delete_dataset``).

Layout: ``barcodes``, ``features`` (UTF-8 strings), ``feature_attrs/<name>`` (optional
columns, one value per feature), and one CSR group per matrix (``data``, ``indices``,
``indptr``), cell-major. Attributes in ``attrs`` hold JSON-safe metadata.
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path

import h5py
import numpy as np
import scipy.sparse as sp

SUBDIR = "modalities"
VERSION = 1
MATRICES = "matrices"
_STR = h5py.string_dtype("utf-8")
DEFAULT_BLOCK_VALUES = 20_000_000  # about 240 MB of data + indices in memory per block


class ModalityStoreError(RuntimeError):
    """A store is missing, or does not hold what was asked of it."""


def store_dir(data_dir: str | Path, dataset_id: str) -> Path:
    return Path(data_dir) / SUBDIR / dataset_id


def store_path(data_dir: str | Path, dataset_id: str, name: str) -> Path:
    return store_dir(data_dir, dataset_id) / f"{name}.h5"


def _strings(values: Iterable) -> np.ndarray:
    return np.asarray([str(v) for v in values], dtype=object)


class StoreWriter:
    """Writes a store row block by row block, to ``<name>.h5.part``, renamed on close: a
    run that stops leaves no store that looks finished. Use it as a context manager;
    an exception inside the block removes the part file."""

    def __init__(self, path: str | Path, barcodes: Sequence, features: Sequence,
                 feature_attrs: dict[str, Sequence] | None = None, attrs: dict | None = None) -> None:
        self.path = Path(path)
        self.part = self.path.with_name(f".{self.path.name}.part")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.n_obs = len(barcodes)
        self.n_vars = len(features)
        self._rows: dict[str, int] = {}
        self._f = h5py.File(self.part, "w")
        self._f.attrs["version"] = VERSION
        self._f.attrs["meta"] = json.dumps(attrs or {})
        self._f.create_dataset("barcodes", data=_strings(barcodes), dtype=_STR)
        self._f.create_dataset("features", data=_strings(features), dtype=_STR)
        group = self._f.create_group("feature_attrs")
        for key, values in (feature_attrs or {}).items():
            if len(values) != self.n_vars:
                raise ModalityStoreError(f"feature_attrs[{key!r}] has {len(values)} values for {self.n_vars} features.")
            group.create_dataset(key, data=_strings(values), dtype=_STR)
        self._f.create_group(MATRICES)

    def _matrix(self, name: str) -> h5py.Group:
        groups = self._f[MATRICES]
        if name in groups:
            return groups[name]
        g = groups.create_group(name)
        g.create_dataset("data", shape=(0,), maxshape=(None,), dtype="f4", chunks=(262_144,), compression="lzf")
        g.create_dataset("indices", shape=(0,), maxshape=(None,), dtype="i4", chunks=(262_144,), compression="lzf")
        g.create_dataset("indptr", shape=(self.n_obs + 1,), dtype="i8", fillvalue=0)
        self._rows[name] = 0
        return g

    def append(self, name: str, block: sp.spmatrix | sp.sparray) -> None:
        """Add the next rows of matrix ``name``, in barcode order."""
        csr = sp.csr_matrix(block)
        if csr.shape[1] != self.n_vars:
            raise ModalityStoreError(f"Matrix {name!r}: block has {csr.shape[1]} columns for {self.n_vars} features.")
        csr.sort_indices()
        g = self._matrix(name)
        start_row = self._rows[name]
        if start_row + csr.shape[0] > self.n_obs:
            raise ModalityStoreError(f"Matrix {name!r}: more rows than the {self.n_obs} barcodes.")
        nnz = int(g["data"].shape[0])
        new = nnz + int(csr.nnz)
        g["data"].resize((new,))
        g["indices"].resize((new,))
        g["data"][nnz:new] = csr.data.astype(np.float32)
        g["indices"][nnz:new] = csr.indices.astype(np.int32)
        g["indptr"][start_row + 1: start_row + csr.shape[0] + 1] = nnz + csr.indptr[1:].astype(np.int64)
        self._rows[name] = start_row + csr.shape[0]

    def close(self) -> None:
        for name, rows in self._rows.items():
            if rows != self.n_obs:
                raise ModalityStoreError(f"Matrix {name!r}: {rows} of {self.n_obs} rows written.")
        self._f.close()
        os.replace(self.part, self.path)

    def abort(self) -> None:
        self._f.close()
        self.part.unlink(missing_ok=True)

    def __enter__(self) -> StoreWriter:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.close()
        else:
            self.abort()


def write(path: str | Path, matrices: dict[str, sp.spmatrix | sp.sparray], barcodes: Sequence, features: Sequence,
          feature_attrs: dict[str, Sequence] | None = None, attrs: dict | None = None) -> Path:
    """Write a whole store from matrices already in memory."""
    with StoreWriter(path, barcodes, features, feature_attrs, attrs) as w:
        for name, m in matrices.items():
            w.append(name, m)
    return Path(path)


class ModalityStore:
    """Read side. Safe to share between threads (one lock per open file); opens lazily and
    holds only the barcode and indptr arrays in memory."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise ModalityStoreError(f"{self.path.name} is missing.")
        self._lock = threading.Lock()
        self._f = h5py.File(self.path, "r")
        self.barcodes = np.asarray(self._f["barcodes"].asstr()[:], dtype=object)
        self.features = np.asarray(self._f["features"].asstr()[:], dtype=object)
        self.meta: dict = json.loads(self._f.attrs.get("meta", "{}"))
        self._index: dict[str, int] | None = None
        self._indptr: dict[str, np.ndarray] = {}

    @property
    def matrices(self) -> list[str]:
        return list(self._f[MATRICES].keys())

    @property
    def n_obs(self) -> int:
        return len(self.barcodes)

    @property
    def n_vars(self) -> int:
        return len(self.features)

    def feature_attr(self, name: str) -> np.ndarray:
        group = self._f["feature_attrs"]
        if name not in group:
            raise ModalityStoreError(f"{self.path.name} has no feature column {name!r}.")
        return np.asarray(group[name].asstr()[:], dtype=object)

    def nnz(self, matrix: str) -> int:
        return int(self._group(matrix)["data"].shape[0])

    def rows_of(self, barcodes: Iterable) -> np.ndarray:
        """Row of each barcode, -1 where the store does not hold it."""
        if self._index is None:
            self._index = {str(b): i for i, b in enumerate(self.barcodes)}
        return np.fromiter((self._index.get(str(b), -1) for b in barcodes), dtype=np.int64)

    def _group(self, matrix: str) -> h5py.Group:
        groups = self._f[MATRICES]
        if matrix not in groups:
            raise ModalityStoreError(f"{self.path.name} has no matrix {matrix!r} (has {sorted(groups.keys())}).")
        return groups[matrix]

    def _pointers(self, matrix: str) -> np.ndarray:
        ptr = self._indptr.get(matrix)
        if ptr is None:
            ptr = self._group(matrix)["indptr"][:]
            self._indptr[matrix] = ptr
        return ptr

    def read(self, matrix: str, rows: Sequence[int] | np.ndarray) -> sp.csr_matrix:
        """These rows, in this order, as a CSR. Rows must be >= 0: filter ``rows_of`` first,
        so a cell the store lacks is the caller's decision, never a silent zero row."""
        rows = np.asarray(rows, dtype=np.int64)
        if rows.size and (rows.min() < 0 or rows.max() >= self.n_obs):
            raise ModalityStoreError(f"Row out of range in {self.path.name} (holds {self.n_obs} rows).")
        ptr = self._pointers(matrix)
        order = np.argsort(rows, kind="stable")
        sorted_rows = rows[order]
        lengths = ptr[sorted_rows + 1] - ptr[sorted_rows]
        total = int(lengths.sum())
        data = np.empty(total, dtype=np.float32)
        indices = np.empty(total, dtype=np.int32)
        pos = 0
        i = 0
        with self._lock:
            g = self._group(matrix)
            while i < sorted_rows.size:
                j = i + 1
                # Join rows next to each other (or the same row twice) into one disk read.
                while j < sorted_rows.size and sorted_rows[j] == sorted_rows[j - 1] + 1:
                    j += 1
                start = int(ptr[sorted_rows[i]])
                stop = int(ptr[sorted_rows[j - 1] + 1])
                n = stop - start
                if n:
                    data[pos:pos + n] = g["data"][start:stop]
                    indices[pos:pos + n] = g["indices"][start:stop]
                pos += n
                i = j
        indptr = np.zeros(rows.size + 1, dtype=np.int64)
        indptr[1:] = np.cumsum(lengths)
        out = sp.csr_matrix((data, indices, indptr), shape=(rows.size, self.n_vars))
        if not np.array_equal(order, np.arange(rows.size)):
            out = out[np.argsort(order, kind="stable")]
        return out

    def iter_blocks(self, matrix: str, rows: Sequence[int] | np.ndarray,
                    max_values: int = DEFAULT_BLOCK_VALUES) -> Iterator[tuple[np.ndarray, sp.csr_matrix]]:
        """Walk ``rows`` in order, a block at a time: yields (positions in ``rows``, CSR)."""
        rows = np.asarray(rows, dtype=np.int64)
        ptr = self._pointers(matrix)
        lengths = ptr[rows + 1] - ptr[rows] if rows.size else np.zeros(0, dtype=np.int64)
        start = 0
        while start < rows.size:
            total = 0
            stop = start
            while stop < rows.size and (stop == start or total + lengths[stop] <= max_values):
                total += int(lengths[stop])
                stop += 1
            yield np.arange(start, stop), self.read(matrix, rows[start:stop])
            start = stop

    def close(self) -> None:
        with self._lock:
            self._f.close()

    def __enter__(self) -> ModalityStore:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
