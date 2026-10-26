"""Ingest without reading the matrix into memory (datasets larger than the memory free).

Today's ingest (``loading.load_unit``, ``session._load_and_merge``) reads each file into
an AnnData, merges and writes one gzip h5ad: for the 10x 1.3M-cell mouse brain (2.6e9
stored values) that is ~31 GB before the app can open it. This makes the same h5ad with
the matrix copied a block of cells at a time:

- The cell and gene tables are built as today's readers build them (scanpy 1.11's 10x
  readers' own frames; for an .h5ad, anndata's ``read_elem`` of obs, var, obsm, varm,
  varp, uns) into an AnnData without a matrix, the *skeleton*. Everything today's ingest
  does that never reads the matrix runs on it unchanged: unique names, the feature-type
  split's choices, ``merge_units`` (anndata's concat), the protein panel, provenance,
  sequencing metrics, contig tables, and anndata writes it.
- The matrix goes from the file into the new h5ad's X block by block (``BLOCK_NNZ``
  stored values): a 10x h5's genes x cells CSC arrays (both layouts: ``/matrix`` and
  Cell Ranger 2's ``/<genome>``) are the cells x genes CSR arrays; a MEX matrix.mtx is
  read in chunks of lines (sorted by cell, as Cell Ranger and R write it, or, any order,
  bucketed by cell range on disk first); an .h5ad's X and layers keep their layout
  (``feature_split._subset_matrix``), its .raw and obsp are copied as stored.
- Antibody and CellPlex columns are taken out on the way into small dense tables and
  attached as ``protein.split_features`` attaches them.

Equal to today's ingest of the same files (``tests/test_ingestion_streamed.py`` runs
both): X values, cell and gene order, value dtype; obs and var frames and dtypes; obsm;
uns. Differences, on purpose: a MEX matrix is stored CSR (today's reader leaves it CSC,
which cannot be read by rows); stored indices are int64 only when the value count
needs it (> 2**31, as scipy does in memory; a split's count is bounded by the file's);
a copied .raw is not re-encoded. A merge of several units needs their X alone (an
.h5ad unit with layers is refused with what to do instead).

``choose`` decides: streamed when the preflight estimate of today's in-memory ingest
(``preflight.estimate_load``) does not fit the memory free with the working reserve
(``dataset_manager``'s rule), or when the session's ``streamed`` option says "on".
"""

from __future__ import annotations

import gzip
import logging
import os
import shutil
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd

from scview.core import crispr, hashing, protein
from scview.core.ingestion import feature_split as fs
from scview.core.ingestion.bundling import FileRole, IngestUnit, UnitFormat
from scview.core.ingestion.detection import legacy_tenx_genomes
from scview.core.ingestion.loading import (
    IngestLoadError,
    _path_for_role,
    _single,
    _stage_features,
    _stage_gzipped,
)

logger = logging.getLogger(__name__)

STREAMABLE = (UnitFormat.tenx_h5, UnitFormat.tenx_mex, UnitFormat.anndata)
MODES = ("auto", "on", "off")
BLOCK_NNZ = fs.BLOCK_NNZ  # stored values read and written per block
MEX_CHUNK_LINES = 2_000_000  # matrix.mtx lines parsed per chunk
MEX_BUCKET_ENTRIES = 16_000_000  # entries per on-disk bucket of an unsorted matrix.mtx
# Memory of a streamed ingest beside the cell and gene tables: one block read, its
# converted copies and the remapped indices (~2M values x ~60 B), the parsed MEX chunk.
STREAM_WORKING_BYTES = 384 * 2**20
_I32 = np.iinfo(np.int32).max
_COUNTS_HEAD = 1000  # stored values ``merge._counts_like`` judges
BLOCKS_NOTE = ("Too large to read into memory whole: the matrix is read and saved a block of cells "
               "at a time, giving the same dataset. Saving it writes about as much to disk as the "
               "files uploaded.")


class StreamUnsupported(RuntimeError):
    """A file this cannot ingest in blocks; the message says why."""


# ---------------------------------------------------------------------------
# A matrix in a file, read as blocks of cells
# ---------------------------------------------------------------------------


class Rows:
    """A cells x features matrix in a file, read as CSR row blocks.

    ``blocks()`` yields ``(indptr, indices, data)`` per block of consecutive cells:
    ``indptr`` from 0 (int64, one more than the block's cells), ``indices`` int64
    feature positions, sorted within each cell, ``data`` in ``dtype``."""

    n_obs: int
    n_vars: int
    nnz: int
    dtype: np.dtype

    def blocks(self) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        raise NotImplementedError

    def head(self, n: int = _COUNTS_HEAD) -> np.ndarray:
        """The first ``n`` stored values (what ``merge._counts_like`` reads)."""
        raise NotImplementedError

    def counts_like(self) -> bool:
        vals = self.head()
        return True if len(vals) == 0 else bool(np.all(np.mod(vals, 1) == 0))


class _H5Rows(Rows):
    """CSR arrays in an HDF5 group: a 10x h5 (its genes x cells CSC arrays are the
    cells x genes CSR ones) or an h5ad's X. ``to_float``: int32 values become float32,
    as scanpy's 10x reader makes them."""

    def __init__(self, path: Path, group: str, n_obs: int, n_vars: int, *, to_float: bool):
        self.path, self.group, self.n_obs, self.n_vars = Path(path), group, int(n_obs), int(n_vars)
        with h5py.File(self.path, "r") as f:
            g = f[group]
            self.nnz = int(g["data"].shape[0])
            src = np.dtype(g["data"].dtype)
        self._to_float = to_float and src == np.int32
        self.dtype = np.dtype(np.float32) if self._to_float else src

    def blocks(self):
        with h5py.File(self.path, "r") as f:
            g = f[self.group]
            indptr = g["indptr"][:].astype(np.int64)
            if len(indptr) != self.n_obs + 1:
                raise IngestLoadError(f"{self.path.name}: {len(indptr) - 1} cells in the matrix, "
                                      f"{self.n_obs} barcodes.")
            for start, end in fs._blocks(indptr, BLOCK_NNZ):
                lo, hi = int(indptr[start]), int(indptr[end])
                data = g["data"][lo:hi]
                if self._to_float:
                    data = data.astype(np.float32)
                idx = g["indices"][lo:hi].astype(np.int64, copy=False)
                yield indptr[start:end + 1] - lo, idx, data

    def head(self, n: int = _COUNTS_HEAD) -> np.ndarray:
        with h5py.File(self.path, "r") as f:
            vals = f[self.group]["data"][:n]
        return vals.astype(np.float32) if self._to_float else vals


class _Unsorted(Exception):
    """matrix.mtx entries are not in cell order: read it again by buckets."""


class _MexRows(Rows):
    """A Matrix Market file (genes x cells, 1-based coordinates), read in chunks of lines.

    Values are float32, as ``anndata.read_mtx`` (scanpy's MEX reader) makes them;
    duplicate entries are summed, as its CSR conversion sums them. Files sorted by cell
    (Cell Ranger, R's writeMM) stream in one pass; any other order is bucketed by cell
    range into temporary files under ``workdir`` first (12 bytes per entry)."""

    def __init__(self, path: Path, workdir: Path):
        self.path, self.workdir = Path(path), Path(workdir)
        with _open_text(self.path) as fh:
            self._field, (self.n_vars, self.n_obs, self.nnz), _ = _mtx_header(fh)
        self.dtype = np.dtype(np.float32)
        self.sorted = True  # until a chunk says otherwise

    def blocks(self):
        if self.sorted:
            yield from self._sorted_blocks()
        else:
            yield from self._bucketed_blocks()

    def head(self, n: int = _COUNTS_HEAD) -> np.ndarray:
        for _g, _c, v in self._chunks(lines=n):
            return v[:n]
        return np.zeros(0, dtype=np.float32)

    # -- parsing ------------------------------------------------------------

    def _chunks(self, lines: int = MEX_CHUNK_LINES):
        """(gene, cell, value) arrays per chunk: 0-based int64 positions, float32 values."""
        pattern = self._field == "pattern"
        names = ["g", "c"] if pattern else ["g", "c", "v"]
        dtype = {"g": np.int64, "c": np.int64}
        if not pattern:
            dtype["v"] = np.int64 if self._field == "integer" else np.float64
        with _open_text(self.path) as fh:
            _mtx_header(fh)
            reader = pd.read_csv(fh, sep=r"\s+", header=None, names=names, usecols=range(len(names)),
                                 dtype=dtype, chunksize=lines, engine="c", float_precision="round_trip")
            for chunk in reader:
                g = chunk["g"].to_numpy() - 1
                c = chunk["c"].to_numpy() - 1
                v = (np.ones(len(chunk), dtype=np.float32) if pattern
                     else chunk["v"].to_numpy().astype(np.float32))
                if len(g) and (g.min() < 0 or g.max() >= self.n_vars or c.min() < 0
                               or c.max() >= self.n_obs):
                    raise IngestLoadError(f"{self.path.name} has an entry outside its size line "
                                          f"({self.n_vars} x {self.n_obs}).")
                yield g, c, v

    def _rows(self, g, c, v, c0: int, c1: int):
        """The CSR block of cells [c0, c1) from their entries (any order)."""
        order = np.lexsort((g, c))
        g, c, v = g[order], c[order], v[order]
        if len(g) > 1:
            key = c * self.n_vars + g
            first = np.concatenate([[True], key[1:] != key[:-1]])
            if not first.all():  # duplicates: summed, as the CSR conversion sums them
                starts = np.flatnonzero(first)
                v = np.add.reduceat(v, starts)
                g, c = g[starts], c[starts]
        indptr = np.zeros(c1 - c0 + 1, dtype=np.int64)
        np.cumsum(np.bincount(c - c0, minlength=c1 - c0), out=indptr[1:])
        return indptr, g, v

    def _sorted_blocks(self):
        next_cell = 0  # the first cell not given yet
        carry = (np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.float32))
        for g, c, v in self._chunks():
            if len(c) and ((len(carry[1]) and c[0] < carry[1][0]) or c[0] < next_cell
                           or bool(np.any(c[1:] < c[:-1]))):
                self.sorted = False
                raise _Unsorted
            g, c, v = (np.concatenate([a, b]) for a, b in zip(carry, (g, c, v)))
            if not len(c):
                continue
            last = int(c[-1])  # its entries may go on in the next chunk
            cut = int(np.searchsorted(c, last, side="left"))
            if last > next_cell:
                yield self._rows(g[:cut], c[:cut], v[:cut], next_cell, last)
                next_cell = last
            carry = (g[cut:], c[cut:], v[cut:])
        if next_cell < self.n_obs or len(carry[0]):
            yield self._rows(*carry, next_cell, self.n_obs)

    def _bucketed_blocks(self):
        per = max(1, int(MEX_BUCKET_ENTRIES * self.n_obs // max(1, self.nnz)))
        n_buckets = -(-self.n_obs // per)
        rec = np.dtype([("g", "<i4"), ("c", "<i4"), ("v", "<f4")])
        tmp = Path(tempfile.mkdtemp(prefix=".scview-mtx-", dir=self.workdir))
        try:
            for g, c, v in self._chunks():
                b = c // per
                order = np.argsort(b, kind="stable")
                b = b[order]
                bounds = np.flatnonzero(np.diff(b)) + 1
                for part in np.split(np.arange(len(b)), bounds):
                    if not len(part):
                        continue
                    sel = order[part]
                    out = np.empty(len(sel), dtype=rec)
                    out["g"], out["c"], out["v"] = g[sel], c[sel], v[sel]
                    with open(tmp / f"{int(b[part[0]])}.bin", "ab") as fh:
                        out.tofile(fh)
            for i in range(n_buckets):
                p = tmp / f"{i}.bin"
                r = np.fromfile(p, dtype=rec) if p.exists() else np.zeros(0, dtype=rec)
                p.unlink(missing_ok=True)
                yield self._rows(r["g"].astype(np.int64), r["c"].astype(np.int64), r["v"],
                                 i * per, min(self.n_obs, (i + 1) * per))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


def _open_text(path: Path):
    with open(path, "rb") as fh:
        gz = fh.read(2) == b"\x1f\x8b"
    return gzip.open(path, "rt", encoding="utf-8") if gz else open(path, encoding="utf-8")


def _mtx_header(fh) -> tuple[str, tuple[int, int, int], int]:
    """(field, (rows, cols, entries), lines read) from a Matrix Market file's head."""
    banner = fh.readline()
    parts = banner.lower().split()
    if len(parts) < 5 or parts[0] != "%%matrixmarket" or parts[1] != "matrix":
        raise StreamUnsupported("matrix.mtx has no Matrix Market banner")
    if parts[2] != "coordinate" or parts[4] != "general" or parts[3] not in ("integer", "real",
                                                                             "pattern"):
        raise StreamUnsupported(f"a Matrix Market '{' '.join(parts[2:5])}' matrix")
    n = 1
    for line in fh:
        n += 1
        s = line.strip()
        if not s or s.startswith("%"):
            continue
        rows, cols, nnz = (int(x) for x in s.split()[:3])
        return parts[3], (rows, cols, nnz), n
    raise StreamUnsupported("matrix.mtx has no size line")


# ---------------------------------------------------------------------------
# One unit: its skeleton and its matrix
# ---------------------------------------------------------------------------


@dataclass
class StreamedUnit:
    """One unit as today's ``load_unit`` makes it, without its matrix.

    ``meta``: obs, var (the genes kept), obsm, varm, varp, uns. ``rows``: the matrix,
    cells x the file's features (None for an .h5ad copied as stored). ``keep``: the
    file's feature positions of ``meta.var``, in order. ``take``: the antibody and
    CellPlex columns taken out (positions, names), attached by ``attach_split``."""

    label: str
    meta: ad.AnnData
    rows: Rows | None
    keep: np.ndarray
    n_features: int
    take_ab: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    ab_names: list[str] = field(default_factory=list)
    take_cmo: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    cmo_names: list[str] = field(default_factory=list)
    take_guide: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))  # CRISPR guides
    guide_names: list[str] = field(default_factory=list)
    split: protein.ProteinSplit | None = None
    h5ad: Path | None = None  # an .h5ad: X and layers copied by feature_split, .raw/obsp as stored
    h5ad_extra: list[str] = field(default_factory=list)  # its layers / .raw / obsp present


def open_unit(unit: IngestUnit, workdir: Path) -> StreamedUnit:
    """The unit's skeleton and matrix (reads the cell and gene tables only)."""
    if not unit.complete:
        raise IngestLoadError(
            f"Unit '{unit.label}' is not complete; resolve validation issues before loading.")
    openers = {UnitFormat.tenx_h5: _open_tenx_h5, UnitFormat.tenx_mex: _open_mex,
               UnitFormat.anndata: _open_h5ad}
    opener = openers.get(unit.format)
    if opener is None:
        raise StreamUnsupported(f"{unit.format.value} files are read into memory whole")
    try:
        meta, rows, h5ad, extra = opener(unit, Path(workdir))
    except (IngestLoadError, StreamUnsupported):
        raise
    except Exception as e:  # the same sentence load_unit gives
        logger.exception("Failed to open unit '%s' (%s) in blocks", unit.label, unit.format)
        raise IngestLoadError(f"Could not read this {unit.format.value} dataset: {e}") from e
    if meta.n_obs == 0 or meta.n_vars == 0:
        raise IngestLoadError(
            f"Could not read '{unit.label}' as single-cell data: it was read as "
            f"{unit.format.value} with {meta.n_obs} cells and {meta.n_vars} genes. "
            "Check that it is the file you meant to upload and that it isn't damaged.")
    meta.var_names_make_unique()
    meta.obs_names_make_unique()
    su = StreamedUnit(unit.label, meta, rows, np.arange(meta.n_vars), meta.n_vars, h5ad=h5ad,
                      h5ad_extra=extra)
    _plan_split(su)
    return su


def _collect(dsets: dict, group: h5py.Group, skip=("data", "indices", "indptr")) -> None:
    """scanpy's ``_collect_datasets`` without reading the matrix arrays."""
    for k, v in group.items():
        if isinstance(v, h5py.Dataset):
            if k not in skip:
                dsets[k] = v[()]
        else:
            _collect(dsets, v, skip)


def _open_tenx_h5(unit: IngestUnit, workdir: Path):
    """The frames ``sc.read_10x_h5(gex_only=False)`` makes (scanpy 1.11.5's
    ``_read_v3_10x_h5`` / ``_read_legacy_10x_h5``, the matrix left out)."""
    path = Path(_single(unit))
    with h5py.File(path, "r") as f:
        if "matrix" in f:
            group = "matrix"
            dsets: dict = {}
            _collect(dsets, f["matrix"])
            n_cols, n_rows = (int(x) for x in dsets["shape"])
            obs = {"obs_names": dsets["barcodes"].astype(str)}
            var = {"var_names": dsets["name"].astype(str)}
            if "gene_id" not in dsets:
                var["gene_ids"] = dsets["id"].astype(str)
            else:
                var.update({"gene_ids": dsets["gene_id"].astype(str),
                            "probe_ids": dsets["id"].astype(str)})
            var["feature_types"] = dsets["feature_type"].astype(str)
            if "filtered_barcodes" in f["matrix"]:
                obs["filtered_barcodes"] = dsets["filtered_barcodes"].astype(bool)
            if "features" not in f["matrix"]:
                raise ValueError("10x h5 has no features group")
            var.update(
                (name, dsets[name].astype(bool if item.dtype.kind == "b" else str))
                for name, item in f["matrix"]["features"].items()
                if isinstance(item, h5py.Dataset)
                and name not in ["name", "feature_type", "id", "gene_id", "_all_tag_keys"])
        else:
            genomes = legacy_tenx_genomes(f)
            if len(genomes) != 1:
                raise ValueError(
                    f"{path} contains more than one genome. For legacy 10x h5 files you must "
                    f"specify the genome if more than one is present. Available genomes are: "
                    f"{list(f.keys())}" if len(genomes) > 1 else "no 10x matrix group found")
            group = genomes[0]
            dsets = {}
            _collect(dsets, f[group])
            n_cols, n_rows = (int(x) for x in dsets["shape"])
            obs = {"obs_names": dsets["barcodes"].astype(str)}
            var = {"var_names": dsets["gene_names"].astype(str), "gene_ids": dsets["genes"].astype(str)}
    # With the shape given, anndata keeps the dicts' obs_names / var_names as the index
    # (without it, and without X, it numbers the rows); a length that differs raises.
    meta = ad.AnnData(obs=obs, var=var, shape=(n_rows, n_cols))
    return meta, _H5Rows(path, group, n_rows, n_cols, to_float=True), None, []


def _open_mex(unit: IngestUnit, workdir: Path):
    """The frames ``sc.read_10x_mtx(var_names="gene_symbols", gex_only=False)`` makes
    from the staged barcodes and features (scanpy 1.11.5's ``_read_10x_mtx``)."""
    rows = _MexRows(_path_for_role(unit, FileRole.matrix), workdir)
    with tempfile.TemporaryDirectory(dir=workdir) as tmp:
        d = Path(tmp)
        _stage_gzipped(_path_for_role(unit, FileRole.barcodes), d / "barcodes.tsv.gz")
        _stage_features(_path_for_role(unit, FileRole.features), d / "features.tsv.gz")
        genes = pd.read_csv(d / "features.tsv.gz", header=None, sep="\t")
        barcodes = pd.read_csv(d / "barcodes.tsv.gz", header=None)
    # read_mtx(...).T: an AnnData of the matrix's shape, transposed, then named.
    meta = ad.AnnData(obs=pd.DataFrame(index=pd.RangeIndex(rows.n_vars).astype(str)),
                      var=pd.DataFrame(index=pd.RangeIndex(rows.n_obs).astype(str))).T
    meta.var_names = ad.utils.make_index_unique(pd.Index(genes[1].values))
    meta.var["gene_ids"] = genes[0].values
    meta.var["feature_types"] = genes[2].values
    meta.obs_names = barcodes[0].values
    return meta, rows, None, []


_COPY_ENC = ("csr_matrix", "csc_matrix", "array")


def _enc(node) -> str:
    e = node.attrs.get("encoding-type", "array" if isinstance(node, h5py.Dataset) else "")
    return e.decode() if isinstance(e, bytes) else str(e)


def _open_h5ad(unit: IngestUnit, workdir: Path):
    """obs, var, obsm, varm, varp and uns as ``sc.read_h5ad`` reads them (anndata's
    ``read_elem``); X and the layers stay in the file. A file in a pre-0.7 anndata layout
    (a compound var table, h5sparse X) is read into memory as today."""
    from anndata.io import read_elem

    path = Path(_single(unit))
    with h5py.File(path, "r") as f:
        for key in ("obs", "var"):
            if not isinstance(f.get(key), h5py.Group) or _enc(f[key]) != "dataframe":
                raise StreamUnsupported(f"{key} is stored in an old anndata layout")
        mats = [("X", f["X"])] if "X" in f else []
        mats += [(f"layers/{k}", f["layers"][k]) for k in f.get("layers", {})]
        for where, node in mats:
            if _enc(node) not in _COPY_ENC:
                raise StreamUnsupported(f"{where} is stored as {_enc(node) or 'an old layout'}")
        parts = {k: read_elem(f[k]) for k in ("obs", "var", "obsm", "varm", "varp", "uns") if k in f}
        x = f.get("X")
        rows = None
        if x is not None and _enc(x) == "csr_matrix":
            n_obs, n_vars = (int(s) for s in x.attrs["shape"])
            rows = _H5Rows(path, "X", n_obs, n_vars, to_float=False)
        extra = [k for k in ("layers", "raw", "obsp") if k in f and len(f[k])]
        has_x = x is not None
    meta = ad.AnnData(**parts)
    if not has_x:
        extra.append("no X")
    return meta, rows, path, extra


def _plan_split(su: StreamedUnit) -> None:
    """``protein.split_features``'s choices, made on the skeleton: the genes stay in
    ``meta`` (``keep``), antibodies (up to MAX_PROTEINS) and CellPlex tags (up to
    hashing.MAX_TAGS) are taken out as columns, other types are counted. Without a
    mixed ``feature_types`` an obsm protein matrix of another tool is adopted."""
    meta = su.meta
    types = meta.var.get("feature_types")
    if types is None or not (types.astype(str) == protein.GENE_TYPE).any() or \
            (types.astype(str) == protein.GENE_TYPE).all():
        protein.adopt_obsm(meta)
        return
    types = types.astype(str)
    split = protein.ProteinSplit()
    is_ab = (types == protein.ANTIBODY_TYPE).to_numpy()
    is_cmo = (types == hashing.CMO_TYPE).to_numpy()
    is_guide = (types == crispr.GUIDE_TYPE).to_numpy()
    if not 0 < is_guide.sum() <= crispr.MAX_GUIDES:
        is_guide = np.zeros_like(is_guide)
    other = types[(types != protein.GENE_TYPE) & (types != protein.ANTIBODY_TYPE) & ~is_cmo & ~is_guide]
    split.other_features = {str(k): int(v) for k, v in other.value_counts().items()}
    genes = (types == protein.GENE_TYPE).to_numpy()
    names = meta.var_names
    su.meta = meta[:, genes].copy()
    su.keep = np.flatnonzero(genes)
    if 0 < is_ab.sum() <= protein.MAX_PROTEINS:
        su.take_ab = np.flatnonzero(is_ab)
        su.ab_names = [str(v) for v in names[is_ab]]
    elif split.other_features or is_ab.any():
        su.meta.uns[protein.UNS_KEY] = split.to_dict()
    if 0 < is_cmo.sum() <= hashing.MAX_TAGS:
        su.take_cmo = np.flatnonzero(is_cmo)
        su.cmo_names = [str(v) for v in names[is_cmo]]
    if is_guide.any():
        su.take_guide = np.flatnonzero(is_guide)
        su.guide_names = [str(v) for v in names[is_guide]]
    su.split = split


def counts_like(units: list[StreamedUnit]) -> dict[str, bool]:
    """Whether each unit's matrix holds counts, judged on its first stored values as
    ``merge._counts_like`` judges a loaded X (for ``plan_merge``)."""
    out = {}
    for su in units:
        if su.rows is not None:
            out[su.label] = su.rows.counts_like()
            continue
        vals = np.zeros(0)
        with h5py.File(su.h5ad, "r") as f:
            x = f.get("X")
            if isinstance(x, h5py.Dataset):
                rows = max(1, -(-_COUNTS_HEAD // max(1, x.shape[1])))
                vals = np.asarray(x[:rows]).ravel()[:_COUNTS_HEAD]
            elif x is not None:
                vals = x["data"][:_COUNTS_HEAD]
        out[su.label] = True if len(vals) == 0 else bool(np.all(np.mod(vals, 1) == 0))
    return out


def attach_split(su: StreamedUnit, ab: np.ndarray | None, cmo: np.ndarray | None,
                 guide: np.ndarray | None = None) -> None:
    """The antibody, CellPlex and guide tables, taken out while the matrix was written,
    attached to the unit as ``split_features`` attaches them."""
    if len(su.take_ab) and ab is not None:
        protein.attach(su.meta, ab, su.ab_names, su.split)
    if len(su.take_cmo) and cmo is not None:
        hashing.attach_counts(su.meta, cmo, su.cmo_names, hashing.CMO_TYPE)
    if len(su.take_guide) and guide is not None:
        crispr.attach_counts(su.meta, guide, su.guide_names)


# ---------------------------------------------------------------------------
# Writing X a block at a time
# ---------------------------------------------------------------------------


@dataclass
class _Part:
    unit: StreamedUnit
    colmap: np.ndarray  # file feature position -> output column, -1 dropped


def _index_dtype(nnz: int, n_vars: int) -> np.dtype:
    """As scipy picks it for a CSR in memory: int32 unless the values or columns need int64."""
    return np.dtype(np.int64 if max(nnz, n_vars) > _I32 else np.int32)


def _write_x(group: h5py.Group, parts: list[_Part], n_vars: int) -> list[tuple]:
    """X (cells of every part in order x ``n_vars``) as CSR, with each part's antibody /
    CellPlex columns returned as dense float32 tables ``(ab, cmo, guide)``. A MEX part found out
    of cell order is read again by buckets (the X written so far is dropped)."""
    while True:
        try:
            return _write_x_once(group, parts, n_vars)
        except _Unsorted:
            if "X" in group:
                del group["X"]
            logger.info("matrix.mtx is not sorted by cell: reading it again by cell buckets")


def _write_x_once(group: h5py.Group, parts: list[_Part], n_vars: int) -> list[tuple]:
    n_obs = sum(p.unit.rows.n_obs for p in parts)
    nnz_bound = sum(p.unit.rows.nnz for p in parts)
    dtype = np.result_type(*[p.unit.rows.dtype for p in parts])
    idx_dtype = _index_dtype(nnz_bound, n_vars)
    g = group.create_group("X")
    g.attrs["encoding-type"] = "csr_matrix"
    g.attrs["encoding-version"] = "0.1.0"
    g.attrs["shape"] = np.asarray((n_obs, n_vars), dtype=np.int64)
    chunk = (min(1 << 16, max(1, nnz_bound)),)
    for key, dt in (("data", dtype), ("indices", idx_dtype)):
        g.create_dataset(key, shape=(0,), maxshape=(None,), dtype=dt, chunks=chunk, **fs._GZIP)
    indptr = np.zeros(n_obs + 1, dtype=np.int64)
    row0, out = 0, []
    for p in parts:
        u, rows = p.unit, p.unit.rows
        identity = (p.colmap.size == n_vars and np.array_equal(p.colmap, np.arange(n_vars)))
        monotonic = identity or bool(np.all(np.diff(p.colmap[p.colmap >= 0]) > 0))
        tables = []
        for cols in (u.take_ab, u.take_cmo, u.take_guide):
            if len(cols):
                where = np.full(rows.n_vars, -1, dtype=np.int64)
                where[cols] = np.arange(len(cols))
                tables.append((where, np.zeros((rows.n_obs, len(cols)), dtype=np.float32)))
            else:
                tables.append(None)
        r = 0
        for bptr, idx, data in rows.blocks():
            n = len(bptr) - 1
            counts = np.diff(bptr)
            local = None
            for t in tables:
                if t is None:
                    continue
                if local is None:
                    local = np.repeat(np.arange(n, dtype=np.int64), counts)
                sel = t[0][idx] >= 0
                if sel.any():
                    np.add.at(t[1], (r + local[sel], t[0][idx[sel]]), data[sel])
            if identity:
                new_idx, new_data, new_counts = idx, data, counts
            else:
                mapped = p.colmap[idx]
                k = mapped >= 0
                if local is None:
                    local = np.repeat(np.arange(n, dtype=np.int64), counts)
                rows_k = local[k]
                new_idx, new_data = mapped[k], data[k]
                if not monotonic:  # columns reordered: sort each cell's entries again
                    order = np.lexsort((new_idx, rows_k))
                    new_idx, new_data, rows_k = new_idx[order], new_data[order], rows_k[order]
                new_counts = np.bincount(rows_k, minlength=n)
            fs._append(g["data"], np.asarray(new_data, dtype=dtype))
            fs._append(g["indices"], new_idx.astype(idx_dtype, copy=False))
            indptr[row0 + r + 1:row0 + r + n + 1] = indptr[row0 + r] + np.cumsum(new_counts)
            r += n
        if r != rows.n_obs:
            raise IngestLoadError(f"'{u.label}': the matrix gave {r} cells, its barcodes {rows.n_obs}.")
        row0 += r
        out.append(tuple(None if t is None else t[1] for t in tables))
    g.create_dataset("indptr", data=indptr.astype(idx_dtype), **fs._GZIP)
    return out


def _copy_h5ad(su: StreamedUnit, fo: h5py.File) -> tuple:
    """An .h5ad unit's matrices into ``fo``: X and the layers with the kept genes
    (``feature_split._subset_matrix``, the antibody / tag columns of X taken out), .raw
    and obsp as stored (today's split keeps .raw whole)."""
    keep = np.zeros(su.n_features, dtype=bool)
    keep[su.keep] = True
    masks = []
    for cols in (su.take_ab, su.take_cmo, su.take_guide):
        m = np.zeros(su.n_features, dtype=bool)
        m[cols] = True
        masks.append(m)
    with h5py.File(su.h5ad, "r") as fi:
        tables = [None, None, None]
        if "X" in fi:
            got = fs._subset_matrix(fi["X"], fo, "X", keep, masks)
            tables = [t if m.any() else None for t, m in zip(got, masks)]
        if "layers" in fi:
            g = fs._group_like(fi["layers"], fo, "layers")
            for name in fi["layers"].keys():
                fs._subset_matrix(fi["layers"][name], g, name, keep, [])
        for key in ("raw", "obsp"):
            if key in fi:
                fi.copy(fi[key], fo, name=key)
    return tuple(tables)


# ---------------------------------------------------------------------------
# A session's units into one dataset
# ---------------------------------------------------------------------------


def ingest(units: list[StreamedUnit], part: Path, *, join, sample_label: str, reconciliation):
    """Write the units' matrices into ``part`` (an h5 file) and return the dataset's
    skeleton (merged as ``session._load_and_merge`` merges, when there are several)."""
    from scview.core.ingestion.merge import merge_units

    with h5py.File(part, "w") as fo:
        if len(units) == 1:
            su = units[0]
            if su.h5ad is not None:
                tables = _copy_h5ad(su, fo)
            else:
                colmap = np.full(su.n_features, -1, dtype=np.int64)
                colmap[su.keep] = np.arange(len(su.keep))
                tables = _write_x(fo, [_Part(su, colmap)], su.meta.n_vars)[0]
            attach_split(su, *tables)
            return su.meta
        for su in units:
            if su.rows is None or (su.h5ad is not None and su.h5ad_extra):
                what = ", ".join(su.h5ad_extra) or "a dense or CSC X"
                raise StreamUnsupported(
                    f"'{su.label}' has {what}: a merge of large datasets reads each sample's X "
                    "alone, by rows. Upload the samples one at a time and combine them in Compare "
                    "> Joint embedding, or save each with only a CSR X first.")
        # The merged genes, in order: the merge of the skeletons decides them.
        bare = {su.label: ad.AnnData(obs=su.meta.obs[[]], var=su.meta.var) for su in units}
        var_names = merge_units(bare, join=join, sample_label=sample_label,
                                reconciliation=reconciliation).var_names
        resets = {r.sample: r.via_column for r in (reconciliation.resets if reconciliation and
                                                   reconciliation.needed and reconciliation.feasible
                                                   else [])}
        parts = []
        for su in units:
            names = su.meta.var_names
            if su.label in resets:  # as merge_units re-indexes it
                names = ad.utils.make_index_unique(pd.Index(su.meta.var[resets[su.label]].astype(str)))
            colmap = np.full(su.n_features, -1, dtype=np.int64)
            colmap[su.keep] = var_names.get_indexer(names)
            parts.append(_Part(su, colmap))
        for su, tables in zip(units, _write_x(fo, parts, len(var_names))):
            attach_split(su, *tables)
    metas = {su.label: su.meta for su in units}
    protein.common_panel(list(metas.values()))
    crispr.common_guides(list(metas.values()))
    merged = merge_units(metas, join=join, sample_label=sample_label, reconciliation=reconciliation)
    protein.renormalise(merged)
    if not merged.var_names.equals(var_names):  # pragma: no cover - the same merge twice
        raise IngestLoadError("The merged genes changed between the two merges of the samples.")
    return merged


def finish_h5ad(meta: ad.AnnData, part: Path, dest: Path) -> None:
    """The dataset's h5ad at ``dest``: ``meta`` written by anndata (gzip, strings as
    categoricals, as ``write_h5ad`` writes them) beside the matrices already in ``part``,
    then moved into place."""
    tmp = part.with_name(part.name + ".meta")
    try:
        meta.write_h5ad(tmp, compression="gzip")
        with h5py.File(tmp, "r") as fm, h5py.File(part, "a") as fo:
            for k, v in fm.attrs.items():
                fo.attrs[k] = v
            for key in fm.keys():
                if key not in fo:  # X, layers, .raw, obsp came from the file
                    fm.copy(fm[key], fo, name=key)
        os.replace(part, dest)
    finally:
        tmp.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Streamed or in memory
# ---------------------------------------------------------------------------


@dataclass
class Choice:
    streamed: bool
    reason: str
    need_bytes: int = 0  # today's in-memory ingest, estimated
    free_bytes: int | None = None


def in_memory_need(units: list[IngestUnit], genes_in_rows: bool = True) -> int:
    """Today's ingest at its peak (``preflight.estimate_load``): each unit read whole; a
    merge holds every sample and the merged copy at once (as the ingest preflight says)."""
    from scview.core import preflight

    peaks, residents = [], []
    for u in units:
        h = preflight.read_unit_header(u, genes_in_rows)
        h.backed = False  # the ingest reads an .h5ad whole, whatever its size
        est = preflight.estimate_load(h)
        peaks.append(est.peak_bytes)
        residents.append(est.resident_bytes)
    if len(units) == 1:
        return int(peaks[0])
    return int(max(max(peaks), 2 * sum(residents)))


def choose(units: list[IngestUnit], mode: str = "auto", genes_in_rows: bool = True) -> Choice:
    """Stream when today's in-memory ingest would not fit the memory free with the
    working reserve (``dataset_manager.LOAD_MARGIN`` and ``working_reserve``), or when
    ``mode`` is "on"; "off" never streams. Formats this cannot stream (loom, zarr, text
    tables) are read into memory as before."""
    from scview.core import memory_budget
    from scview.core.dataset_manager import LOAD_MARGIN, working_reserve

    if mode == "off":
        return Choice(False, "the session's streamed option is off")
    streamable = all(u.format in STREAMABLE for u in units)
    if mode == "on":
        if not streamable:
            bad = sorted({u.format.value for u in units if u.format not in STREAMABLE})
            raise StreamUnsupported(f"{', '.join(bad)} files cannot be read in blocks")
        return Choice(True, "the session's streamed option is on")
    if not streamable:
        return Choice(False, "not a format read in blocks")
    try:
        need = in_memory_need(units, genes_in_rows)
    except Exception as e:  # noqa: BLE001 - never block an ingest on the estimate
        logger.warning("No in-memory estimate for the ingest: %s", e)
        return Choice(False, "no estimate")
    b = memory_budget.read_budget()
    if b.limit_bytes is None or b.used_bytes is None:
        return Choice(False, "memory unknown", need)
    free = b.limit_bytes - b.used_bytes
    streamed = need * LOAD_MARGIN + working_reserve(b.limit_bytes) > free
    gb = 1024**3
    reason = (f"read whole it needs about {need / gb:.1f} GB at its peak; {free / gb:.1f} GB of "
              f"{b.limit_bytes / gb:.1f} GB is free")
    return Choice(streamed, reason, need, free)


def streamed_need(units: list[IngestUnit], genes_in_rows: bool = True) -> int:
    """A streamed ingest's peak: the cell and gene tables (and an .h5ad's obsm, uns),
    the row pointers, one block in flight and the antibody / tag tables."""
    from scview.core import preflight

    total = STREAM_WORKING_BYTES + preflight.LOAD_FIXED
    for u in units:
        h = preflight.read_unit_header(u, genes_in_rows)
        n = h.n_cells or 0
        total += (preflight._frames_bytes(h) + h.obsm_bytes + h.uns_bytes + h.varm_bytes
                  + n * (16 + preflight.LOAD_PER_CELL))
    return int(total)


def disk_need(units: list[IngestUnit]) -> int:
    """Disk a streamed ingest writes: about the size of the files read (gzip either way),
    at most their matrices uncompressed."""
    from scview.core import preflight

    total = 0
    for u in units:
        h = preflight.read_unit_header(u)
        mats = sum(m.nbytes for m in h.matrices)
        total += min(int(1.5 * h.file_bytes), mats) if mats else int(1.5 * h.file_bytes)
    return total
