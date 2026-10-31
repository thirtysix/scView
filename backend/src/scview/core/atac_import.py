"""10x Multiome peaks: out of the gene matrix, into a modality store.

A Cell Ranger ARC ``filtered_feature_bc_matrix`` holds genes ("Gene Expression") and peaks
("Peaks") in one matrix, peaks about 82% of its stored values (PBMC 10k: 104M of 127.5M).
Peaks are never read into X. They are written once, a block of cells at a time, to the store
``DATA_DIR/modalities/<id>/atac.h5`` (``core/modality_store.py``; matrix ``counts``; feature
columns ``chrom``, ``start``, ``end``, ``interval``; ``meta``: genome, source), and the genes
take the usual way through ingest.

Sources: a 10x .h5 or MEX folder, an .h5ad with ``var["feature_types"]`` containing Peaks,
an .h5mu (MuData on disk, ``mod/rna`` + ``mod/atac``, read with h5py and anndata's
``read_elem``; no mudata dependency). A ``per_barcode_metrics.csv`` beside the matrix, or
attached later, adds the FRiP, TSS and mitochondrial fractions.

Per cell, in obs: ``atac_total_counts``, ``atac_n_peaks`` (from the matrix),
``atac_frac_peaks``, ``atac_frac_tss``, ``atac_frac_mito`` (from the metrics).
``uns["scview_atac"]`` (a JSON string): n_peaks, genome, source, what was computed.

Not read: ATAC-only data (peaks and no genes, refused with a sentence, never taken as
genes), a chromatin assay inside a Seurat object (said in the conversion report), peaks in a
.loom or .zarr (a note in the history), and a merge of samples whose peak sets differ (one
sentence at upload: called against different peaks, their columns mean different things).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

from scview.core import modality_store

if TYPE_CHECKING:
    from scview.core.ingestion.bundling import IngestUnit
    from scview.core.ingestion.streamed import Rows

logger = logging.getLogger(__name__)

PEAKS_TYPE = "Peaks"
GENE_TYPE = "Gene Expression"
STORE = "atac"
MATRIX = "counts"
UNS_KEY = "scview_atac"
LEFT_OUT_KEY = "scview_atac_left_out"  # a sentence for the ingest history, popped there
OBS_TOTAL = "atac_total_counts"
OBS_N_PEAKS = "atac_n_peaks"
OBS_FRIP = "atac_frac_peaks"
OBS_TSS = "atac_frac_tss"
OBS_MITO = "atac_frac_mito"
QC_COLUMNS = (OBS_TOTAL, OBS_N_PEAKS, OBS_FRIP, OBS_TSS, OBS_MITO)
VERSION = 1
METRIC_COLUMNS = ("barcode", "atac_raw_reads", "atac_mitochondrial_reads", "atac_fragments",
                  "atac_TSS_fragments", "atac_peak_region_fragments")
METRICS_CHUNK_ROWS = 200_000
# Formats whose peaks can be read without holding the whole matrix.
STORE_FORMATS = ("tenx_h5", "tenx_mex", "anndata", "h5mu")
RNA_MODS = ("rna", "RNA", "gex", "GEX")
ATAC_MODS = ("atac", "ATAC", "peaks", "Peaks")
_INTERVAL = re.compile(r"^(.+?)[:_-](\d+)[-_](\d+)$")

ONLY_PEAKS = (
    "{name} holds ATAC peaks and no genes: scView reads ATAC beside RNA (10x Multiome), "
    "ATAC-only data is not supported yet. Upload the file that has both, or the RNA alone."
)
LOOM_OR_ZARR = ("This file has {n:,} Peaks features, left out: scView reads multiome peaks from "
                "a 10x .h5 or MEX folder, an .h5ad or an .h5mu.")
MERGE_DIFFERENT = (
    "These samples were called against different peaks ({names}), so ATAC is left out of the "
    "merged dataset: its RNA is merged as usual. Upload each sample alone to use its ATAC, or "
    "use a Cell Ranger ARC aggr output, which has one peak set."
)
MERGE_SOME = (
    "Only some samples have ATAC peaks ({names}), so ATAC is left out of the merged dataset: its "
    "RNA is merged as usual. Upload the multiome samples alone to use their ATAC."
)
NOT_COUNTS = ("The ATAC values of {name} are not counts and it has no counts layer, so ATAC is left "
              "out: export the raw peak counts to use them.")


class AtacImportError(RuntimeError):
    """The sentence says which file, and what to do."""


class AtacOnly(AtacImportError):
    """Peaks and no genes."""


# ---------------------------------------------------------------------------
# Feature types, without reading a matrix
# ---------------------------------------------------------------------------


def parse_interval(text: str) -> tuple[str, int, int] | None:
    """('chr1', 10109, 10357) from 'chr1:10109-10357' (also 'chr1-10109-10357' and
    'chr1_10109_10357'); coordinates as given (10x: BED, 0-based half open). None when it
    is not an interval."""
    m = _INTERVAL.match(str(text).strip())
    return (m.group(1), int(m.group(2)), int(m.group(3))) if m else None


def _mod(f: h5py.File, names: tuple[str, ...]) -> str | None:
    mods = f.get("mod")
    return next((n for n in names if isinstance(mods, h5py.Group) and n in mods), None)


def _h5_types(g: h5py.Group) -> np.ndarray:
    return g["features"]["feature_type"][:].astype(str)


def tenx_feature_types(path: str | Path) -> np.ndarray | None:
    """``features/feature_type`` of a 10x h5, or None (Cell Ranger 2's layout has none)."""
    with h5py.File(path, "r") as f:
        g = f.get("matrix")
        if not isinstance(g, h5py.Group) or "features" not in g or "feature_type" not in g["features"]:
            return None
        return _h5_types(g)


def _mex_features(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", header=None, dtype=str, keep_default_na=False)


def unit_feature_types(unit: IngestUnit) -> np.ndarray | None:
    """Every feature's type in a unit's file, read from its metadata; None when it says none."""
    from scview.core.ingestion import feature_split
    from scview.core.ingestion.bundling import FileRole, UnitFormat

    try:
        if unit.format == UnitFormat.tenx_h5 and unit.files:
            return tenx_feature_types(unit.files[0].path)
        if unit.format == UnitFormat.anndata and unit.files:
            return feature_split.feature_types(Path(unit.files[0].path))
        if unit.format == UnitFormat.tenx_mex:
            features = next((f for f in unit.files if f.role == FileRole.features), None)
            if features is None:
                return None
            table = _mex_features(features.path)
            return table[2].to_numpy(dtype=str) if table.shape[1] >= 3 else None
    except (OSError, KeyError, ValueError, RuntimeError, pd.errors.ParserError) as e:
        logger.debug("No feature types for %s: %s", unit.label, e)
    return None


def has_peaks(unit: IngestUnit) -> bool:
    """A unit with ATAC peaks a store can be written from."""
    from scview.core.ingestion.bundling import UnitFormat

    if unit.format == UnitFormat.h5mu and unit.files:
        try:
            with h5py.File(unit.files[0].path, "r") as f:
                return _mod(f, ATAC_MODS) is not None
        except OSError:
            return False
    types = unit_feature_types(unit)
    return types is not None and bool((types == PEAKS_TYPE).any())


def only_peaks_sentence(unit: IngestUnit) -> str | None:
    """The refusal for a unit with peaks and no genes, else None."""
    from scview.core.ingestion.bundling import UnitFormat

    name = unit.files[0].name if unit.files else unit.label
    if unit.format == UnitFormat.h5mu and unit.files:
        try:
            with h5py.File(unit.files[0].path, "r") as f:
                if _mod(f, RNA_MODS) is None:
                    return ONLY_PEAKS.format(name=name)
        except OSError:
            return None
        return None
    types = unit_feature_types(unit)
    if types is None or not (types == PEAKS_TYPE).any() or (types == GENE_TYPE).any():
        return None
    return ONLY_PEAKS.format(name=name)


def refuse_only_peaks(types: pd.Series | np.ndarray | None, name: str = "This file") -> None:
    """Raise AtacOnly when a loaded table's feature types are peaks and no genes."""
    if types is None:
        return
    values = np.asarray(types).astype(str)
    if (values == PEAKS_TYPE).any() and not (values == GENE_TYPE).any():
        raise AtacOnly(ONLY_PEAKS.format(name=name))


def merge_sentence(units: list[IngestUnit]) -> str | None:
    """Why a merge of these units has no ATAC, or None (at most one unit, no peaks at all,
    or one peak set)."""
    if len(units) < 2:
        return None
    with_peaks = [u for u in units if has_peaks(u)]
    if not with_peaks:
        return None
    if len(with_peaks) < len(units):
        return MERGE_SOME.format(names=", ".join(u.label for u in with_peaks))
    plans = [plan_unit(u) for u in with_peaks]
    if any(p is None for p in plans):
        return MERGE_SOME.format(names=", ".join(u.label for u in with_peaks))
    first = plans[0].names
    if any(p.names != first for p in plans[1:]):
        return MERGE_DIFFERENT.format(names=", ".join(u.label for u in with_peaks))
    return None


# ---------------------------------------------------------------------------
# A unit's peaks as a matrix in a file
# ---------------------------------------------------------------------------


@dataclass
class UnitPeaks:
    """Where one unit's peak counts are: ``rows`` gives its cells x file features as CSR
    blocks, ``cols`` the peak columns."""

    label: str
    file: str
    fmt: str
    rows: Rows | None  # None with ``note``: left out, and why
    cols: np.ndarray
    names: list[str]
    intervals: list[str]
    barcodes: pd.Index
    genome: str | None
    genome_assumed: bool = False
    note: str | None = None


def infer_genome(gene_ids) -> str | None:
    """The reference from Ensembl gene ids, when the file does not name one."""
    head = [str(g) for g in list(gene_ids)[:2000]]
    if sum(g.startswith("ENSMUSG") for g in head) > len(head) / 2:
        return "mm10"
    if sum(g.startswith("ENSG") for g in head) > len(head) / 2:
        return "GRCh38"
    return None


class _CscRows:
    """An .h5ad X stored by column (or a dense one): peak columns gathered into memory as
    one CSR block, as a by-column file cannot be read by cell. 12 bytes per peak value."""

    def __init__(self, path: Path, node: str, cols: np.ndarray, n_vars: int):
        self.path, self.node, self.cols, self.n_vars = Path(path), node, cols, n_vars

    def blocks(self):
        from scview.core.ingestion import feature_split as fs

        with h5py.File(self.path, "r") as f:
            src = f[self.node]
            if isinstance(src, h5py.Dataset):
                n_obs = src.shape[0]
                step = max(1, fs.BLOCK_VALUES // max(1, src.shape[1]))
                for start in range(0, n_obs, step):
                    block = sp.csr_matrix(np.asarray(src[start:start + step])[:, self.cols])
                    yield self._widen(block)
                return
            n_obs, _n_var = (int(x) for x in src.attrs["shape"])
            indptr = src["indptr"][:].astype(np.int64)
            keep = np.zeros(self.n_vars, dtype=bool)
            keep[self.cols] = True
            data, rows, columns = [], [], []
            for start, end in fs._blocks(indptr, fs.BLOCK_NNZ):
                lo, hi = int(indptr[start]), int(indptr[end])
                c = np.repeat(np.arange(start, end), np.diff(indptr[start:end + 1]))
                k = keep[c]
                if k.any():
                    data.append(src["data"][lo:hi][k].astype(np.float32))
                    rows.append(src["indices"][lo:hi][k].astype(np.int32))
                    columns.append(c[k].astype(np.int32))
        m = sp.csr_matrix(
            (np.concatenate(data) if data else np.zeros(0, np.float32),
             (np.concatenate(rows) if rows else np.zeros(0, np.int32),
              np.concatenate(columns) if columns else np.zeros(0, np.int32))),
            shape=(n_obs, self.n_vars))
        m.sort_indices()
        yield m.indptr.astype(np.int64), m.indices.astype(np.int64), m.data

    def _widen(self, block: sp.csr_matrix):
        return block.indptr.astype(np.int64), self.cols[block.indices].astype(np.int64), block.data


def _h5ad_node(f: h5py.File, base: str) -> tuple[str, bool]:
    """The node of a (mod's) X holding counts: X, else ``layers/counts``. The flag says
    whether what was found holds counts at all (its first stored values are whole)."""

    def head(node) -> np.ndarray:
        if isinstance(node, h5py.Dataset):
            return np.asarray(node[:8]).ravel()[:1000]
        return np.asarray(node["data"][:1000])

    prefix = f"{base}/" if base else ""
    x = f.get(f"{prefix}X")
    if x is not None and np.all(np.mod(head(x), 1) == 0):
        return f"{prefix}X", True
    for layer in ("counts", "raw_counts"):
        node = f.get(f"{prefix}layers/{layer}")
        if node is not None and np.all(np.mod(head(node), 1) == 0):
            return f"{prefix}layers/{layer}", True
    return f"{prefix}X", False


def _node_rows(path: Path, node: str, cols: np.ndarray, n_obs: int, n_vars: int) -> Any:
    """Row blocks of an h5ad matrix node: read as stored when CSR, gathered otherwise."""
    from scview.core.ingestion import streamed

    with h5py.File(path, "r") as f:
        enc = streamed._enc(f[node])
    if enc == "csr_matrix":
        return streamed._H5Rows(path, node, n_obs, n_vars, to_float=True)
    if enc in ("csc_matrix", "array"):
        return _CscRows(path, node, cols, n_vars)
    raise AtacImportError(f"{path.name}: the ATAC matrix is stored as {enc or 'an old layout'}, which "
                          "scView cannot read in blocks. Save it again with a current anndata.")


def _plan_tenx_h5(unit: IngestUnit) -> UnitPeaks | None:
    from scview.core.ingestion import streamed

    path = Path(unit.files[0].path)
    with h5py.File(path, "r") as f:
        g = f.get("matrix")
        if not isinstance(g, h5py.Group) or "features" not in g or "feature_type" not in g["features"]:
            return None
        types = _h5_types(g)
        cols = np.flatnonzero(types == PEAKS_TYPE)
        if not len(cols):
            return None
        n_vars, n_obs = (int(x) for x in g["shape"][()])
        fe = g["features"]
        names = fe["id"][:][cols].astype(str).tolist()
        intervals = fe["interval"][:][cols].astype(str).tolist() if "interval" in fe else list(names)
        genome = fe["genome"][:][cols].astype(str) if "genome" in fe else None
        ids = fe["id"][:][types == GENE_TYPE].astype(str)
        barcodes = pd.Index(g["barcodes"][:].astype(str))
    named = str(genome[0]) if genome is not None and len(genome) else None
    return UnitPeaks(unit.label, unit.files[0].rel_path or path.name, "10x h5",
                     streamed._H5Rows(path, "matrix", n_obs, n_vars, to_float=True), cols, names, intervals,
                     barcodes, named or infer_genome(ids), genome_assumed=not named)


def _plan_mex(unit: IngestUnit, workdir: Path) -> UnitPeaks | None:
    from scview.core.ingestion import streamed
    from scview.core.ingestion.bundling import FileRole

    features = next((f for f in unit.files if f.role == FileRole.features), None)
    barcodes = next((f for f in unit.files if f.role == FileRole.barcodes), None)
    matrix = next((f for f in unit.files if f.role == FileRole.matrix), None)
    if features is None or barcodes is None or matrix is None:
        return None
    table = _mex_features(features.path)
    if table.shape[1] < 3:
        return None
    cols = np.flatnonzero(table[2].to_numpy(dtype=str) == PEAKS_TYPE)
    if not len(cols):
        return None
    names = table[0].to_numpy(dtype=str)[cols].tolist()
    # Cell Ranger ARC's features.tsv: id, name, type, chrom, start, end.
    if table.shape[1] >= 6:
        sub = table.iloc[cols]
        intervals = [f"{c}:{s}-{e}" for c, s, e in zip(sub[3], sub[4], sub[5])]
    else:
        intervals = list(names)
    ids = table[0].to_numpy(dtype=str)[table[2].to_numpy(dtype=str) == GENE_TYPE]
    bars = pd.read_csv(barcodes.path, header=None, dtype=str)[0]
    return UnitPeaks(unit.label, features.rel_path or features.name, "10x MEX",
                     streamed._MexRows(Path(matrix.path), workdir), cols, names, intervals, pd.Index(bars),
                     infer_genome(ids), genome_assumed=True)


def _plan_h5ad(unit: IngestUnit) -> UnitPeaks | None:
    from anndata.io import read_elem

    from scview.core.ingestion import feature_split as fs

    path = Path(unit.files[0].path)
    with h5py.File(path, "r") as f:
        if not isinstance(f.get("var"), h5py.Group) or "feature_types" not in f["var"]:
            return None
        var = read_elem(f["var"])
        types = var["feature_types"].astype(str).to_numpy()
        cols = np.flatnonzero(types == PEAKS_TYPE)
        if not len(cols):
            return None
        node, counts = _h5ad_node(f, "")
        if not counts:
            return UnitPeaks(unit.label, unit.files[0].rel_path or path.name, "h5ad", None, cols, [], [],
                             pd.Index([]), None, note=NOT_COUNTS.format(name=path.name))
        obs_names = fs._obs_names(f["obs"])
    names = var.index.astype(str)[cols].tolist()
    intervals = (var["interval"].astype(str).to_numpy()[cols].tolist() if "interval" in var.columns
                 else list(names))
    genome = var["genome"].astype(str).to_numpy()[cols][0] if "genome" in var.columns else None
    ids = var["gene_ids"].astype(str).to_numpy()[types == GENE_TYPE] if "gene_ids" in var.columns else []
    rows = _node_rows(path, node, cols, len(obs_names), len(var))
    return UnitPeaks(unit.label, unit.files[0].rel_path or path.name, "h5ad", rows, cols, names, intervals,
                     obs_names, genome or infer_genome(ids), genome_assumed=not genome)


def _plan_h5mu(unit: IngestUnit) -> UnitPeaks | None:
    from anndata.io import read_elem

    from scview.core.ingestion import feature_split as fs

    path = Path(unit.files[0].path)
    with h5py.File(path, "r") as f:
        mod = _mod(f, ATAC_MODS)
        if mod is None:
            return None
        base = f"mod/{mod}"
        var = read_elem(f[f"{base}/var"])
        obs_names = fs._obs_names(f[f"{base}/obs"])
        node, counts = _h5ad_node(f, base)
        if not counts:
            return UnitPeaks(unit.label, unit.files[0].rel_path or path.name, "h5mu", None,
                             np.zeros(0, np.int64), [], [], pd.Index([]), None,
                             note=NOT_COUNTS.format(name=path.name))
        rna = _mod(f, RNA_MODS)
        ids: list[str] = []
        if rna is not None:
            rvar = read_elem(f[f"mod/{rna}/var"])
            if "gene_ids" in rvar.columns:
                ids = rvar["gene_ids"].astype(str).tolist()
            else:
                ids = rvar.index.astype(str).tolist()
    cols = np.arange(len(var))
    names = var.index.astype(str).tolist()
    intervals = var["interval"].astype(str).tolist() if "interval" in var.columns else list(names)
    genome = str(var["genome"].iloc[0]) if "genome" in var.columns and len(var) else None
    rows = _node_rows(path, node, cols, len(obs_names), len(var))
    return UnitPeaks(unit.label, unit.files[0].rel_path or path.name, "h5mu", rows, cols, names, intervals,
                     obs_names, genome or infer_genome(ids), genome_assumed=not genome)


def plan_unit(unit: IngestUnit, workdir: str | Path | None = None) -> UnitPeaks | None:
    """Where a unit's peaks are, or None without peaks (or in a format not read for them)."""
    from scview.core.ingestion.bundling import UnitFormat

    if not unit.files:
        return None
    if unit.format == UnitFormat.tenx_h5:
        return _plan_tenx_h5(unit)
    if unit.format == UnitFormat.tenx_mex:
        return _plan_mex(unit, Path(workdir or Path(unit.files[0].path).parent))
    if unit.format == UnitFormat.anndata:
        return _plan_h5ad(unit)
    if unit.format == UnitFormat.h5mu:
        return _plan_h5mu(unit)
    return None


# ---------------------------------------------------------------------------
# Writing the store
# ---------------------------------------------------------------------------


def feature_attrs(intervals: list[str]) -> dict[str, list[str]]:
    """chrom, start, end, interval per peak; a peak that names no interval has blanks (gene
    activity and nearest genes then say so)."""
    parsed = [parse_interval(s) for s in intervals]
    return {"chrom": [p[0] if p else "" for p in parsed], "start": [str(p[1]) if p else "" for p in parsed],
            "end": [str(p[2]) if p else "" for p in parsed], "interval": [str(s) for s in intervals]}


@dataclass
class Written:
    """One store's cells: per-cell totals and peaks, in the store's order."""

    path: Path
    barcodes: pd.Index
    total: np.ndarray
    n_peaks: np.ndarray
    info: dict[str, Any] = field(default_factory=dict)


def write_store(plans: list[UnitPeaks], part: str | Path, *, merged: bool = False) -> Written:
    """Write the units' peak columns a block of cells at a time. ``merged``: barcodes get
    merge_units' ``:label`` suffix. Peak sets must match (``merge_sentence`` said so before)."""
    from scview.core.ingestion import streamed

    first = plans[0]
    names = first.names
    barcodes = []
    for p in plans:
        if p.names != names:
            raise AtacImportError(MERGE_DIFFERENT.format(names=", ".join(q.label for q in plans)))
        idx = ad.utils.make_index_unique(p.barcodes.astype(str))
        barcodes.append(idx + f":{p.label}" if merged else idx)
    all_barcodes = barcodes[0] if len(barcodes) == 1 else barcodes[0].append(barcodes[1:])
    meta = {"genome": first.genome, "genome_assumed": bool(first.genome_assumed),
            "source": ", ".join(p.file for p in plans), "format": first.fmt, "version": VERSION}
    while True:
        totals: list[np.ndarray] = []
        peaks: list[np.ndarray] = []
        try:
            with modality_store.StoreWriter(part, all_barcodes, names, feature_attrs(first.intervals),
                                            meta) as writer:
                nnz = sum(_write_unit(p, writer, totals, peaks) for p in plans)
            break
        except streamed._Unsorted:  # the MEX reader now reads it by cell buckets; the part file is gone
            logger.info("matrix.mtx is not sorted by cell: writing the peaks again by cell buckets")
    return Written(Path(part), all_barcodes, np.concatenate(totals), np.concatenate(peaks), {
        "n_peaks": len(names), "n_cells": len(all_barcodes), "nnz": int(nnz), "genome": first.genome,
        "genome_assumed": bool(first.genome_assumed), "source": meta["source"], "format": first.fmt})


def _write_unit(p: UnitPeaks, writer: modality_store.StoreWriter, totals: list, peaks: list) -> int:
    where = np.full(int(p.rows.n_vars), -1, dtype=np.int64)
    where[p.cols] = np.arange(len(p.cols))
    nnz = cells = 0
    for bptr, idx, data in p.rows.blocks():
        n = len(bptr) - 1
        mapped = where[idx]
        k = mapped >= 0
        local = np.repeat(np.arange(n, dtype=np.int64), np.diff(bptr))[k]
        counts = np.bincount(local, minlength=n)
        ptr = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(counts, out=ptr[1:])
        values = np.asarray(data[k], dtype=np.float32)
        writer.append(MATRIX, sp.csr_matrix((values, mapped[k].astype(np.int32), ptr), shape=(n, len(p.cols))))
        totals.append(np.bincount(local, weights=values, minlength=n).astype(np.float32))
        peaks.append(counts.astype(np.int32))
        nnz += len(values)
        cells += n
    if cells != len(p.barcodes):
        raise AtacImportError(f"{p.file}: the matrix gave {cells:,} cells for {len(p.barcodes):,} barcodes.")
    return nnz


def info_json(info: dict[str, Any]) -> str:
    return json.dumps({"version": VERSION, "computed": ["peak counts"], **info})


def attach_obs(adata, written: Written) -> dict[str, Any]:
    """Per-cell totals into obs and the summary into uns, joined by barcode. Returns the
    summary. A dataset cell the store lacks (none at ingest) gets NaN."""
    frame = pd.DataFrame({OBS_TOTAL: written.total, OBS_N_PEAKS: written.n_peaks}, index=written.barcodes)
    joined = frame.reindex(adata.obs_names.astype(str))
    adata.obs[OBS_TOTAL] = joined[OBS_TOTAL].to_numpy(dtype=np.float32)
    adata.obs[OBS_N_PEAKS] = joined[OBS_N_PEAKS].to_numpy(dtype=np.float32)
    info = dict(written.info)
    info["n_cells_matched"] = int(joined[OBS_TOTAL].notna().sum())
    adata.uns[UNS_KEY] = info_json(info)
    return info


def read_info(adata) -> dict[str, Any] | None:
    """``uns["scview_atac"]``, or None."""
    raw = adata.uns.get(UNS_KEY) if UNS_KEY in adata.uns else None
    if raw is None:
        return None
    try:
        info = json.loads(str(raw))
    except (TypeError, ValueError):
        logger.warning("Ignoring an unreadable %s", UNS_KEY)
        return None
    return info if isinstance(info, dict) else None


def write_info(adata, info: dict[str, Any]) -> None:
    """Replaces the entry (never mutates it): a save sees it as changed."""
    adata.uns[UNS_KEY] = json.dumps(info)


def dataset_state(adata, data_dir: str | Path, dataset_id: str) -> dict[str, Any] | None:
    """What the dataset description says of ATAC (None without): the Accessibility tab shows
    only when its store is there."""
    if UNS_KEY not in adata.uns or not modality_store.store_path(data_dir, dataset_id, STORE).exists():
        return None
    info = read_info(adata) or {}
    return {"n_peaks": info.get("n_peaks"), "genome": info.get("genome"), "computed": info.get("computed") or []}


def provenance_params(info: dict[str, Any]) -> dict[str, Any]:
    keys = ("n_peaks", "nnz", "genome", "genome_assumed", "source", "format", "n_cells")
    return {"atac": {**{k: info.get(k) for k in keys}, "location": f"{modality_store.SUBDIR}/<id>/{STORE}.h5"}}


# ---------------------------------------------------------------------------
# per_barcode_metrics.csv
# ---------------------------------------------------------------------------


def looks_like_barcode_metrics(header: str) -> bool:
    """Header of Cell Ranger ARC's per_barcode_metrics.csv."""
    columns = {c.strip() for c in header.split(",")}
    return {"barcode", "is_cell", "atac_fragments", "atac_peak_region_fragments"} <= columns


def _ratio(top: pd.Series, bottom: pd.Series) -> np.ndarray:
    a, b = top.to_numpy(dtype=np.float64), bottom.to_numpy(dtype=np.float64)
    return np.divide(a, b, out=np.full(a.shape, np.nan), where=b > 0).astype(np.float32)


def read_barcode_metrics(path: str | Path, wanted: set[str]) -> pd.DataFrame:
    """The ``wanted`` barcodes' rows of a per_barcode_metrics.csv, in chunks (the file has
    every barcode of the whitelist, 88 MB for PBMC 10k)."""
    head = pd.read_csv(path, nrows=0)
    missing = [c for c in METRIC_COLUMNS if c not in head.columns]
    if missing:
        raise AtacImportError(f"{Path(path).name} has no column {missing[0]}: it is not Cell Ranger ARC's "
                              "per_barcode_metrics.csv.")
    parts = []
    for chunk in pd.read_csv(path, usecols=list(METRIC_COLUMNS), chunksize=METRICS_CHUNK_ROWS):
        hit = chunk[chunk["barcode"].astype(str).isin(wanted)]
        if len(hit):
            parts.append(hit)
    if not parts:
        return pd.DataFrame(columns=list(METRIC_COLUMNS))
    return pd.concat(parts).set_index("barcode")


def attach_metrics(adata, path: str | Path, *, sample_column: str | None = None, sample: str | None = None,
                   suffix: str = "") -> dict[str, Any]:
    """FRiP, TSS and mitochondrial fractions per cell from a per_barcode_metrics.csv.
    ``sample_column`` / ``sample``: only that sample's cells (a merged dataset, whose cell
    names carry ``suffix``). Raises AtacImportError when no cell is in the file."""
    names = adata.obs_names.astype(str)
    mask = np.ones(adata.n_obs, dtype=bool)
    if sample_column:
        mask = (adata.obs[sample_column].astype(str) == str(sample)).to_numpy()
    cells = names[mask]
    bare = pd.Index([c[:-len(suffix)] if suffix and c.endswith(suffix) else c for c in cells])
    table = read_barcode_metrics(path, set(bare))
    if table.empty:
        raise AtacImportError(f"None of the {len(cells):,} cells is in {Path(path).name}: it is from another "
                              "sample, or the cell names are not 10x barcodes. Nothing was attached.")
    table = table.reindex(bare)
    for column, values in ((OBS_FRIP, _ratio(table["atac_peak_region_fragments"], table["atac_fragments"])),
                           (OBS_TSS, _ratio(table["atac_TSS_fragments"], table["atac_fragments"])),
                           (OBS_MITO, _ratio(table["atac_mitochondrial_reads"], table["atac_raw_reads"]))):
        old = (adata.obs[column].to_numpy(dtype=np.float32, copy=True) if column in adata.obs.columns
               else np.full(adata.n_obs, np.nan, dtype=np.float32))
        old[mask] = values
        adata.obs[column] = old
    info = read_info(adata) or {"version": VERSION, "computed": []}
    computed = list(info.get("computed") or [])
    if "per-barcode metrics" not in computed:
        computed.append("per-barcode metrics")
    info["computed"] = computed
    files = list(info.get("metrics_files") or [])
    if Path(path).name not in files:
        files.append(Path(path).name)
    info["metrics_files"] = files
    write_info(adata, info)
    matched = int(table["atac_fragments"].notna().sum())
    return {"n_cells": len(cells), "n_matched": matched, "file": Path(path).name}
