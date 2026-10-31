"""ATAC analysis from the peak store (``core/atac_import.py``), read a block of cells at a time.

Method: Signac (MIT; Stuart et al. 2021). ``FindTopFeatures``: peaks above the 5th percentile
of total counts. ``RunTFIDF`` method 1: ``log(1 + TF x 10,000 x IDF)`` with ``TF = count / the
cell's total`` and ``IDF = cells / the peak's total``, totals over every peak and cell.
``RunSVD``: truncated SVD (block Lanczos with full re-orthogonalisation in place of irlba:
the matrix is never built, each pass of the store reads it through a fixed-size block) with
embeddings scaled per component to mean 0, sd 1. ``DepthCor``: Pearson r of each component with the cell's total; the first component
is left out when |r| > 0.75 and the choice is recorded. Then scanpy: a neighbour graph of
the components under its own key (``atac``; the RNA graph is never touched), its UMAP
``X_umap_atac``, Leiden through ``core/leiden.py`` (``atac_leiden``).

WNN RNA + ATAC: ``core/wnn.py``'s FindMultiModalNeighbors with LSI as the second modality,
dims 2:n by default, each modality's rows L2-normalised as Seurat does. Own keys:
``wnn_atac``, ``X_umap_wnn_atac``, ``wnn_atac_rna_weight``, ``wnn_atac_leiden``.

Gene activity: peak counts within gene body + 2 kb upstream (strand aware; ``core/
gene_annotation.py``), summed per gene. It is peak-based, not fragment-based like Signac's
GeneActivity: reads in a gene but outside every called peak are not counted. Stored for every
barcode in the store ``atac_gene_activity`` (cells x genes, matrix ``gene_activity``) and by
gene in ``atac_gene_activity_by_gene``, whose rows are genes, so one gene's overlay reads one row.

Differential accessibility: Wilcoxon rank-sum on TF-IDF for peaks detected in >= 5% of the
group or the rest, BH FDR within each group. Signac recommends logistic regression with the
cell's total as latent covariate, about 144k fits for each group; this is the fast default.
Peak context: nearest gene by TSS, distance, class (promoter: within 2 kb of a TSS; gene
body; distal).

Checked on 10x PBMC 10k multiome (11,898 cells, 143,887 peaks) against Signac 1.17.1 and Seurat
5.6.0. FindTopFeatures: the same 136,705 peaks. LSI: |r| = 1.000 for components 1-30, >= 0.968 for
all 50, singular values within 0.1%, depth correlations within 0.006. WNN RNA weights: r = 0.976
(0.979 from Seurat's own PCA and LSI), the same mean 0.461. Gene activity against Signac's fragment-
based GeneActivity on 2,001 genes: median gene r between the two 0.79; median r with RNA 0.061 against
0.063; CD3D, MS4A1, LYZ, NKG7 top the same cell groups. Differential peaks against Signac's logistic
regression (1,000 cells per group, as it takes about 20 min per group even so): 78, 72 and 52 of the top 100
enriched peaks shared on three clusters, log2FC Spearman 0.95-0.99. Wilcoxon has no depth
covariate and calls many peaks at this n (46,005 of 50,000 tested in the largest cluster): read the top
of the table, not the count.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import tempfile
import time
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.special import ndtr

from scview.core import atac_import, modality_store, wnn
from scview.core.modality_store import ModalityStore

logger = logging.getLogger(__name__)

TAG = "result:atac"
STORE = atac_import.STORE
MATRIX = atac_import.MATRIX
ACTIVITY_STORE = "atac_gene_activity"
ACTIVITY_BY_GENE = "atac_gene_activity_by_gene"
ACTIVITY_MATRIX = "gene_activity"
EXPORT_DIR = "export"

OBSM_LSI = "X_lsi"
OBSM_UMAP = "X_umap_atac"
GRAPH_KEY = "atac"  # uns["atac"], obsp["atac_connectivities"] / ["atac_distances"]
CONN_KEY = "atac_connectivities"
DIST_KEY = "atac_distances"
OBS_CLUSTERS = "atac_leiden"
OBS_ACTIVITY_TOTAL = "atac_activity_total"
OBSM_WNN = "X_umap_wnn_atac"
WNN_GRAPH_KEY = "wnn_atac"
WNN_CONN_KEY = "wnn_atac_connectivities"
WNN_DIST_KEY = "wnn_atac_distances"
OBS_WNN_WEIGHT = "wnn_atac_rna_weight"
OBS_WNN_CLUSTERS = "wnn_atac_leiden"
RUN_OBS = (OBS_CLUSTERS,)
WNN_OBS = (OBS_WNN_WEIGHT, OBS_WNN_CLUSTERS)
RUN_OBSM = (OBSM_LSI, OBSM_UMAP)
RUN_OBSP = (CONN_KEY, DIST_KEY)
WNN_OBSM = (OBSM_WNN,)
WNN_OBSP = (WNN_CONN_KEY, WNN_DIST_KEY)

MIN_CELLS = 50
N_LSI = 50  # RunSVD's n
# Block Lanczos: directions per step; steps after the first, 2 passes of the store each. Against
# Signac's irlba on PBMC 10k: 20 x 16 steps matches every component (|r| >= 0.968, singular values
# within 0.04%); 30 x 11, about the same work, only the first 40; subspace iteration with 80
# directions and 8 passes only the first 14.
LANCZOS_BLOCK = 20
LANCZOS_STEPS = 15
DIMS_TO = 30  # the neighbour graph's last LSI component (Signac's vignette: 2:30)
MIN_PERCENTILE = 5.0  # FindTopFeatures' min.cutoff = "q5"
DEPTH_LIMIT = 0.75  # |r| with the cell's total above which the first component is left out
SCALE_FACTOR = 1e4  # RunTFIDF's and the activity display's
K_NN = 20  # Seurat's FindNeighbors k.param
SEED = 0
UPSTREAM = 2_000  # Signac's GeneActivity extend.upstream
PROMOTER = 2_000
MIN_PCT = 0.05
MAX_PEAKS_PER_GROUP = 50_000
KEEP_PER_GROUP = 5_000
MAX_GROUPS = 100
MIN_GROUP_CELLS = 3
BLOCK_VALUES = 4_000_000  # stored values per block: about 48 MB of data + indices
DA_BATCH_VALUES = 20_000_000  # stored values per batch of peaks being ranked: about 0.9 GB at its peak
BUCKET_VALUES = 20_000_000  # per on-disk bucket when turning activity by gene: about 240 MB
RESULT_KIND = "atac_differential"
KEEP_RESULTS = 8
# Not gene-like for "nearest gene": pseudogenes and small RNAs.
_SKIP_GENE_TYPES = ("miRNA", "misc_RNA", "rRNA", "snRNA", "snoRNA", "scaRNA", "sRNA", "scRNA", "ribozyme",
                    "vault_RNA", "TEC", "Mt_rRNA", "Mt_tRNA")

STAGES = (("stats", "Peak totals"), ("svd", "TF-IDF and SVD (LSI)"), ("graph", "Neighbours on LSI"),
          ("umap", "ATAC UMAP"), ("leiden", "ATAC clusters (Leiden)"), ("activity", "Gene activity"))
WNN_STAGES = (("knn_rna", "RNA neighbours"), ("knn_lsi", "ATAC neighbours"), ("bandwidth", "Kernel bandwidths"),
              ("weights", "Modality weights"), ("joint_knn", "Joint neighbours"), ("umap", "WNN UMAP"),
              ("leiden", "WNN clusters (Leiden)"))

NO_ATAC = ("This dataset has no ATAC peaks. Upload a 10x Multiome filtered_feature_bc_matrix "
           "(.h5 or folder), an .h5ad with Peaks features or an .h5mu.")
NO_STORE = ("The ATAC peak store of this dataset is missing, so there is nothing to read. Upload the "
            "multiome file again.")
ACTIVITY_CAPTION = ("Peak-based gene activity: counts in called peaks within the gene body + 2 kb upstream, "
                    "not all fragments there as in Signac's GeneActivity.")


class AtacError(ValueError):
    """A run that cannot start or a request that cannot be answered: the sentence says what to do."""

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


class AtacCancelled(Exception):
    """The client went away; nothing is kept."""


# ---------------------------------------------------------------------------
# The store and the dataset's cells
# ---------------------------------------------------------------------------


def store_path(data_dir: str | Path, dataset_id: str, name: str = STORE) -> Path:
    return modality_store.store_path(data_dir, dataset_id, name)


def has_atac(adata, data_dir: str | Path, dataset_id: str) -> bool:
    return store_path(data_dir, dataset_id).exists() and atac_import.UNS_KEY in adata.uns


def open_store(adata, data_dir: str | Path, dataset_id: str) -> ModalityStore:
    if atac_import.UNS_KEY not in adata.uns and not store_path(data_dir, dataset_id).exists():
        raise AtacError(NO_ATAC, 404)
    if not store_path(data_dir, dataset_id).exists():
        raise AtacError(NO_STORE, 404)
    return ModalityStore(store_path(data_dir, dataset_id))


def cell_rows(store: ModalityStore, adata) -> np.ndarray:
    """The store row of each of the dataset's cells; AtacError when one has none: every cell
    of a dataset arrives with the store, and filtering only ever removes cells."""
    rows = store.rows_of(adata.obs_names)
    missing = int((rows < 0).sum())
    if missing:
        raise AtacError(f"{missing:,} of this dataset's {len(rows):,} cells are not in its ATAC store: their names "
                        "changed after import. Upload the multiome file again.", 409)
    return rows


# ---------------------------------------------------------------------------
# Peak statistics, selection, TF-IDF
# ---------------------------------------------------------------------------


@dataclass
class PeakStats:
    total: np.ndarray  # per peak, over the cells asked for
    n_cells: np.ndarray  # cells with the peak
    cell_total: np.ndarray  # per cell, in the order asked for

    @property
    def nnz(self) -> int:
        return int(self.n_cells.sum())


def peak_stats(store: ModalityStore, rows: np.ndarray, stop: Callable[[], bool] = lambda: False) -> PeakStats:
    total = np.zeros(store.n_vars, dtype=np.float64)
    n_cells = np.zeros(store.n_vars, dtype=np.int64)
    cell_total = np.zeros(rows.size, dtype=np.float64)
    for pos, block in store.iter_blocks(MATRIX, rows, BLOCK_VALUES):
        if stop():
            raise AtacCancelled()
        total += np.bincount(block.indices, weights=block.data, minlength=store.n_vars)
        n_cells += np.bincount(block.indices, minlength=store.n_vars)
        cell_total[pos] = np.asarray(block.sum(axis=1), dtype=np.float64).ravel()
    return PeakStats(total, n_cells, cell_total)


def top_peaks(total, percentile=5.0):
    """Signac's FindTopFeatures(min.cutoff = "q5"): the peaks whose total's empirical
    cumulative share is above 5% (positions, in store order)."""
    share = np.searchsorted(np.sort(total), total, side="right") / total.size
    return np.flatnonzero(share > percentile / 100.0)


class TfIdf:
    """The selected peaks' TF-IDF, cells x peaks, as an operator over the store's blocks."""

    def __init__(self, store: ModalityStore, rows: np.ndarray, stats: PeakStats, selected: np.ndarray,
                 stop: Callable[[], bool] = lambda: False, blocks: int = BLOCK_VALUES) -> None:
        self.store, self.rows, self.selected, self.stop, self.blocks = store, rows, selected, stop, blocks
        self.shape = (rows.size, selected.size)
        self.where = np.full(store.n_vars, -1, dtype=np.int32)
        self.where[selected] = np.arange(selected.size, dtype=np.int32)
        with np.errstate(divide="ignore", invalid="ignore"):
            self.idf = np.where(stats.total > 0, rows.size / stats.total, 0.0)
        self.cell_total = stats.cell_total
        self._spooled: list[tuple[np.ndarray, sp.csr_matrix]] | None = None

    def iter_blocks(self) -> Iterator[tuple[np.ndarray, sp.csr_matrix]]:
        if self._spooled is not None:
            for pos, m in self._spooled:
                if self.stop():
                    raise AtacCancelled()
                yield pos, m
            return
        for pos, block in self.store.iter_blocks(MATRIX, self.rows, self.blocks):
            if self.stop():
                raise AtacCancelled()
            yield pos, self.transform(pos, block)

    def spool(self, folder: str | Path, n_values: int) -> bool:
        """Write the TF-IDF to uncompressed files in ``folder`` and read them mapped from now
        on: the SVD's 33 passes then skip the store's decompression and the TF-IDF arithmetic
        (PBMC 10k: from 269 s to 70 s). File-backed, so nothing extra is held: the system can
        drop the pages. False, and nothing changes, without the disk for it."""
        folder = Path(folder)
        if shutil.disk_usage(folder).free < 2 * n_values * 8:
            return False
        data = np.memmap(folder / "data.f4", dtype=np.float32, mode="w+", shape=(max(1, n_values),))
        indices = np.memmap(folder / "indices.i4", dtype=np.int32, mode="w+", shape=(max(1, n_values),))
        at = 0
        spooled: list[tuple[np.ndarray, sp.csr_matrix]] = []
        for pos, m in self.iter_blocks():
            end = at + m.nnz
            data[at:end] = m.data
            indices[at:end] = m.indices
            spooled.append((pos, sp.csr_matrix((data[at:end], indices[at:end], m.indptr.astype(np.int32)),
                                               shape=m.shape, copy=False)))
            at = end
        self._spooled = spooled
        return True

    def transform(self, pos: np.ndarray, block: sp.csr_matrix) -> sp.csr_matrix:
        mapped = self.where[block.indices]
        keep = mapped >= 0
        per_cell = np.repeat(pos, np.diff(block.indptr))
        tf = block.data / self.cell_total[per_cell]
        values = np.log1p(tf * SCALE_FACTOR * self.idf[block.indices])[keep]
        counts = np.bincount(np.repeat(np.arange(pos.size), np.diff(block.indptr))[keep], minlength=pos.size)
        ptr = np.zeros(pos.size + 1, dtype=np.int64)
        np.cumsum(counts, out=ptr[1:])
        return sp.csr_matrix((values.astype(np.float64), mapped[keep], ptr), shape=(pos.size, self.selected.size))

    def dot(self, right: np.ndarray) -> np.ndarray:
        """A @ right (peaks x l -> cells x l)."""
        out = np.zeros((self.shape[0], right.shape[1]), dtype=np.float64)
        for pos, m in self.iter_blocks():
            out[pos] = m @ right
        return out

    def tdot(self, left: np.ndarray) -> np.ndarray:
        """A.T @ left (cells x l -> peaks x l)."""
        out = np.zeros((self.shape[1], left.shape[1]), dtype=np.float64)
        for pos, m in self.iter_blocks():
            out += m.T @ left[pos]
        return out


def block_lanczos_svd(op, k, block=LANCZOS_BLOCK, steps=LANCZOS_STEPS, seed=SEED, progress=None):
    """(U, s, Vt) of ``op``'s top ``k``: block Lanczos on A A.T with full re-orthogonalisation,
    so the store is read 2 + 2 x (``steps`` + 1) times, ``block`` directions at a time, and
    the projected matrix comes from the recurrence (no wide last pass). The basis is cells x
    block x (steps + 1). Signs set so each component's largest loading is positive."""
    rng = np.random.default_rng(seed)
    n_rows, n_cols = op.shape
    dim = min(n_rows, n_cols)
    block = max(1, min(block, dim // (steps + 1)))  # a small matrix: the basis fits inside it
    steps = min(steps, max(0, dim // block - 1))
    size = block * (steps + 1)
    total = 2 + 2 * (steps + 1)
    done = 0

    def tick():
        nonlocal done
        done += 1
        if progress is not None:
            progress(done, total)

    basis = np.zeros((n_rows, size), dtype=np.float64)
    projected = np.zeros((size, size), dtype=np.float64)
    basis[:, :block], _r = np.linalg.qr(op.dot(rng.standard_normal((n_cols, block))))
    tick()
    for j in range(steps + 1):
        lo, hi = j * block, (j + 1) * block
        z = op.tdot(basis[:, lo:hi])
        tick()
        y = op.dot(z)
        tick()
        q = basis[:, :hi]
        projected[:hi, lo:hi] = q.T @ y
        if j < steps:
            y -= q @ (q.T @ y)
            y -= q @ (q.T @ y)  # twice: the basis stays orthogonal to machine precision
            basis[:, hi:hi + block], _r = np.linalg.qr(y)
    values, vectors = np.linalg.eigh(np.triu(projected) + np.triu(projected, 1).T)
    order = np.argsort(-values)[:min(k, size)]
    s = np.sqrt(np.maximum(values[order], 0.0))
    u = basis @ vectors[:, order]
    del basis, projected
    loadings = op.tdot(u) / np.where(s > 0, s, 1.0)
    tick()
    signs = np.sign(loadings[np.abs(loadings).argmax(axis=0), np.arange(order.size)])
    signs[signs == 0] = 1
    return u * signs, s, (loadings * signs).T


def scaled_embedding(u):
    """RunSVD(scale.embeddings = TRUE): each component to mean 0, sd 1 (n - 1)."""
    sd = u.std(axis=0, ddof=1)
    return np.divide(u - u.mean(axis=0), sd, out=np.zeros_like(u), where=sd > 0)


def depth_correlation(embedding, depth):
    """Signac's DepthCor: Pearson r of each component with the cell's total."""
    d = depth - depth.mean()
    e = embedding - embedding.mean(axis=0)
    denom = np.sqrt((e * e).sum(axis=0) * (d * d).sum())
    return np.divide(e.T @ d, denom, out=np.zeros(e.shape[1]), where=denom > 0)


# ---------------------------------------------------------------------------
# Plan, estimates
# ---------------------------------------------------------------------------

# Measured at 2 CPUs on PBMC 10k (11,898 cells, 137k peaks selected, 104M values). A pass of the
# store reads 8 B a value through lzf (3.3 s); the TF-IDF pass that writes the spool; a product costs
# per value x direction, its mapped reads included.
_READ_SECONDS_PER_VALUE = 3.3e-8
_SPOOL_SECONDS_PER_VALUE = 1.0e-7
_PRODUCT_SECONDS_PER_VALUE_DIM = 0.9e-9
_ACTIVITY_SECONDS_PER_VALUE = 3.0e-7
_DA_SECONDS_PER_VALUE = 3.5e-7
_DA_SECONDS_PER_VALUE_GROUP = 1.4e-8
_RUN_FIRST_SECONDS = 20.0
_BASE_BYTES = 600 * 1024**2  # scanpy, numba, umap-learn and pynndescent loaded
_BLOCK_BYTES = BLOCK_VALUES * 40  # one block read, its TF-IDF copies and transposed product
_KEPT_BYTES_PER_CELL = 2_500  # components, layouts, the graph


def estimate_seconds(n_cells: int, nnz: int, *, n_iterations: int = -1, activity: bool = True,
                     n_lsi: int = N_LSI) -> float:
    """A run's time at 2 CPUs: passes of the store, the graph, a UMAP, Leiden (their estimates
    from ``core/preflight.py``), and the gene activity pass with its turn by gene."""
    from scview.core import preflight

    lanczos = 1 + 2 * (LANCZOS_STEPS + 1)
    n = max(0, int(n_cells))
    svd = (nnz * (_READ_SECONDS_PER_VALUE + _SPOOL_SECONDS_PER_VALUE)  # the totals' pass, the spool
           + nnz * _PRODUCT_SECONDS_PER_VALUE_DIM * (lanczos * LANCZOS_BLOCK + n_lsi))
    return round(_RUN_FIRST_SECONDS + svd + preflight.umap_seconds(n, three_d=False) + 3e-4 * n
                 + preflight.leiden_seconds(n, n_iterations)
                 + (nnz * _ACTIVITY_SECONDS_PER_VALUE if activity else 0.0), 0)


def estimate_bytes(n_cells: int, n_peaks: int, *, n_lsi: int = N_LSI) -> int:
    """What a run holds beyond the loaded dataset at its peak: the libraries, a block, the
    SVD's Lanczos basis (cells) with a step's two sides, and the kept arrays; or the largest of
    a UMAP and the Leiden child."""
    from scview.core import preflight

    n = max(0, int(n_cells))
    size = LANCZOS_BLOCK * (LANCZOS_STEPS + 1)
    svd = 8 * (n * size + size * size + 2 * LANCZOS_BLOCK * (n + n_peaks) + n_lsi * n_peaks) + _BLOCK_BYTES
    scale = max(1.0, n / 88_640)
    umap = n * preflight.UMAP_BYTES_PER_CELL * scale ** preflight.UMAP_MEMORY_EXPONENT
    leiden = 60 * 1024**2 + n * 9_000
    return int(_BASE_BYTES + n * _KEPT_BYTES_PER_CELL + max(svd, umap, leiden, BUCKET_VALUES * 12))


def estimate_da_seconds(nnz: int, n_groups: int = 1) -> float:
    """Differential accessibility at 2 CPUs: detection, then the batches of peaks; each group's
    bookkeeping per value. PBMC 10k, 104M values: 40 s with 3 groups tested, 76 s with 24."""
    return round(5.0 + nnz * (_DA_SECONDS_PER_VALUE + n_groups * _DA_SECONDS_PER_VALUE_GROUP), 0)


def estimate_da_bytes(n_peaks: int, n_groups: int) -> int:
    """Per-group detection (groups x peaks of integers, twice for its shares), z, p and fold
    change, and one batch of peaks being ranked."""
    # A batch at its peak: three columns of 4 B concatenated from their parts, a sort order (8 B),
    # the sorted copies beside the columns they replace.
    return int(_BASE_BYTES + n_peaks * n_groups * (8 + 8 + 8 + 8 + 4 + 8 + 4 + 1) + DA_BATCH_VALUES * 44)


def estimate_disk_bytes(nnz: int, activity: bool = True) -> int:
    """Disk a run needs at its peak: the TF-IDF spool while the SVD runs (8 B a value), or with
    gene activity its two stores, which stay (PBMC 10k: 487 + 479 MB for 104M peak values), and the
    buckets that turn it by gene (12 B per activity value, 0.9 per peak value), which go."""
    return int(nnz * 8 if not activity else nnz * (9.8 + 10.8))


@dataclass
class AtacPlan:
    n_cells: int
    n_peaks: int
    nnz: int
    genome: str | None
    n_lsi: int = N_LSI
    dims_to: int = DIMS_TO
    drop_first: str = "auto"  # auto | yes | no
    percentile: float = MIN_PERCENTILE
    k: int = K_NN
    min_dist: float = wnn.DEFAULT_MIN_DIST
    resolution: float = wnn.DEFAULT_RESOLUTION
    n_iterations: int = -1
    activity: bool = True
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["estimate_seconds"] = estimate_seconds(self.n_cells, self.nnz, n_iterations=self.n_iterations,
                                                 activity=self.activity, n_lsi=self.n_lsi)
        d["estimate_bytes"] = estimate_bytes(self.n_cells, self.n_peaks, n_lsi=self.n_lsi)
        d["estimate_disk_bytes"] = estimate_disk_bytes(self.nnz, self.activity)
        return d


def plan(adata, data_dir: str | Path, dataset_id: str, *, n_lsi: int | None = None, dims_to: int | None = None,
         drop_first: str = "auto", resolution: float | None = None, activity: bool = True) -> AtacPlan:
    from scview.core import leiden

    with open_store(adata, data_dir, dataset_id) as store:
        cell_rows(store, adata)
        n_peaks, nnz, n_store = store.n_vars, store.nnz(MATRIX), store.n_obs
        genome = store.meta.get("genome")
    if adata.n_obs < MIN_CELLS:
        raise AtacError(f"ATAC analysis needs at least {MIN_CELLS} cells; this dataset has {adata.n_obs}.")
    width = min(N_LSI if n_lsi is None else int(n_lsi), adata.n_obs - 1, n_peaks - 1)
    if width < 3:
        raise AtacError("ATAC analysis needs at least 4 peaks and 4 cells.")
    to = min(DIMS_TO if dims_to is None else int(dims_to), width)
    if to < 2:
        raise AtacError("The neighbour graph needs at least the first 2 LSI components.")
    if drop_first not in ("auto", "yes", "no"):
        raise AtacError("Leaving out the first component is auto, yes or no.")
    res = wnn.DEFAULT_RESOLUTION if resolution is None else float(resolution)
    if not 0 < res <= 10:
        raise AtacError("Resolution must be above 0 and at most 10.")
    nnz = int(round(nnz * adata.n_obs / max(1, n_store)))  # the cells left after filtering
    emb = wnn._last_params(adata, "embeddings") or {}
    return AtacPlan(n_cells=int(adata.n_obs), n_peaks=n_peaks, nnz=nnz, genome=genome, n_lsi=width, dims_to=to,
                    drop_first=drop_first, resolution=res,
                    min_dist=float(emb["umap_min_dist"] if emb.get("umap_min_dist") is not None
                                   else wnn.DEFAULT_MIN_DIST),
                    k=min(K_NN, adata.n_obs - 1), n_iterations=leiden.recorded_iterations(adata), activity=activity)


# ---------------------------------------------------------------------------
# The run: LSI, graph, UMAP, Leiden, gene activity
# ---------------------------------------------------------------------------


@dataclass
class LsiResult:
    embedding: np.ndarray  # cells x n_lsi, scaled per component
    sdev: np.ndarray
    variance_ratio: np.ndarray
    depth_r: np.ndarray
    dropped_first: bool
    dims_used: list[int]  # 1-based, as Signac writes them
    n_selected: int
    cutoff_total: float
    n_zero_cells: int
    loadings_peaks: list[list[Any]]  # the first components' largest loadings, for the co-pilot


@dataclass
class AtacResult:
    lsi: LsiResult
    connectivities: sp.csr_matrix
    distances: sp.csr_matrix
    neighbors_uns: dict[str, Any]
    umap: np.ndarray
    clusters: pd.Categorical
    activity: dict[str, Any] | None
    activity_total: np.ndarray | None
    seconds: float
    stage_seconds: dict[str, float]


def _sum_squares(op: TfIdf) -> float:
    return float(sum(float(np.dot(m.data, m.data)) for _pos, m in op.iter_blocks()))


def run_lsi(store: ModalityStore, rows: np.ndarray, p: AtacPlan, stats: PeakStats,
            stop: Callable[[], bool] = lambda: False,
            progress: Callable[[int, int], None] | None = None) -> LsiResult:
    selected = top_peaks(stats.total, p.percentile)
    op = TfIdf(store, rows, stats, selected, stop)
    with tempfile.TemporaryDirectory(prefix=".scview-tfidf-", dir=store.path.parent) as tmp:
        op.spool(tmp, int(stats.n_cells[selected].sum()))
        u, s, vt = block_lanczos_svd(op, p.n_lsi, progress=progress)
        total_ss = _sum_squares(op)
        op._spooled = None  # the maps go before their files
    embedding = scaled_embedding(u)
    r = depth_correlation(embedding, stats.cell_total)
    drop = abs(float(r[0])) > DEPTH_LIMIT if p.drop_first == "auto" else p.drop_first == "yes"
    first = 2 if drop else 1
    names = store.features[selected]
    top: list[list[Any]] = []
    for c in range(min(3, vt.shape[0])):
        order = np.argsort(-np.abs(vt[c]))[:5]
        top.append([str(names[i]) for i in order])
    sorted_total = np.sort(stats.total)
    cutoff = float(sorted_total[int(np.ceil(p.percentile / 100 * sorted_total.size)) - 1]) if sorted_total.size else 0.0
    return LsiResult(embedding=embedding.astype(np.float32), sdev=s / np.sqrt(max(1, selected.size - 1)),
                     variance_ratio=(s * s) / total_ss if total_ss > 0 else np.zeros_like(s), depth_r=r,
                     dropped_first=bool(drop), dims_used=list(range(first, p.dims_to + 1)),
                     n_selected=int(selected.size), cutoff_total=cutoff,
                     n_zero_cells=int((stats.cell_total == 0).sum()), loadings_peaks=top)


def _holder(n: int):
    import anndata as ad

    return ad.AnnData(obs=pd.DataFrame(index=[str(i) for i in range(n)]))


def iter_run(store: ModalityStore, rows: np.ndarray, p: AtacPlan, *, data_dir: str | Path, dataset_id: str,
             genes: GeneChoice | None,
             stop: Callable[[], bool] = lambda: False) -> Iterator[dict[str, Any]]:
    """The run as a generator over its stages (``stage``, ``progress`` events), then
    ``{"type": "result", "result": AtacResult}``. Checked between stages and blocks (AtacCancelled)."""
    import scanpy as sc

    from scview.core import leiden

    t0 = time.monotonic()
    names = [s for s, _ in STAGES if s != "activity" or p.activity]
    labels = dict(STAGES)
    spent: dict[str, float] = {}
    current: list[Any] = [None, t0]

    def begin(name: str) -> dict[str, Any]:
        if stop():
            raise AtacCancelled()
        now = time.monotonic()
        if current[0] is not None:
            spent[current[0]] = round(now - current[1], 2)
        current[0], current[1] = name, now
        return {"type": "stage", "stage": name, "label": labels[name], "index": names.index(name) + 1,
                "total": len(names), "elapsed_s": round(now - t0, 1)}

    yield begin("stats")
    stats = peak_stats(store, rows, stop)
    yield begin("svd")
    lsi = run_lsi(store, rows, p, stats, stop)
    yield begin("graph")
    holder = _holder(rows.size)
    use = "atac_rep"
    holder.obsm[use] = lsi.embedding[:, [d - 1 for d in lsi.dims_used]]
    sc.pp.neighbors(holder, n_neighbors=p.k, use_rep=use, metric="cosine", key_added=GRAPH_KEY, random_state=SEED)
    conn = holder.obsp[CONN_KEY].copy()
    distances = holder.obsp[DIST_KEY].copy()
    params = {k: v for k, v in holder.uns[GRAPH_KEY]["params"].items()}
    neighbors_uns = {"connectivities_key": CONN_KEY, "distances_key": DIST_KEY,
                     "params": {**params, "use_rep": OBSM_LSI, "lsi_dims": lsi.dims_used}}
    yield begin("umap")
    umap = wnn._umap(holder, GRAPH_KEY, OBSM_UMAP, p.min_dist)
    del holder
    yield begin("leiden")
    clusters = leiden.labels(conn, p.resolution, n_iterations=p.n_iterations)
    activity = activity_total = None
    if p.activity:
        yield begin("activity")
        if genes is None:
            activity = {"available": False}
        elif genes.reason:
            activity = {"available": False, "reason": genes.reason}
        else:
            activity, totals = write_activity(store, genes, data_dir, dataset_id, stop)
            activity_total = totals[rows]
    now = time.monotonic()
    spent[current[0]] = round(now - current[1], 2)
    yield {"type": "result", "result": AtacResult(lsi=lsi, connectivities=conn, distances=distances,
                                                  neighbors_uns=neighbors_uns, umap=umap, clusters=clusters,
                                                  activity=activity, activity_total=activity_total,
                                                  seconds=now - t0, stage_seconds=spent)}


def method_text(p: AtacPlan, lsi: LsiResult) -> str:
    dims = f"{lsi.dims_used[0]}:{lsi.dims_used[-1]}"
    why = (f"the first component was left out (r = {lsi.depth_r[0]:.2f} with depth)" if lsi.dropped_first
           else f"the first component was kept (r = {lsi.depth_r[0]:.2f} with depth)")
    return (f"Signac's LSI on the peak matrix: peaks above the {p.percentile:g}th percentile of total counts "
            f"({lsi.n_selected:,} of {p.n_peaks:,}), TF-IDF method 1 (log(1 + TF x 10,000 x IDF)), truncated SVD by "
            f"block Lanczos ({p.n_lsi} components; embeddings scaled per component); {why}. Graph on LSI {dims} (k = {p.k}, cosine), "
            f"UMAP, Leiden at resolution {p.resolution:g}.")


def replay_params(p: AtacPlan, lsi: LsiResult) -> dict[str, Any]:
    from scview.core import gene_annotation

    url = None
    try:
        url = gene_annotation.RELEASES[gene_annotation.canonical_genome(p.genome)].url
    except gene_annotation.AnnotationError:
        pass
    return {"percentile": p.percentile, "n_lsi": p.n_lsi, "lanczos_block": LANCZOS_BLOCK,
            "lanczos_steps": LANCZOS_STEPS, "seed": SEED, "dims": lsi.dims_used, "k": p.k,
            "metric": "cosine", "min_dist": p.min_dist, "resolution": p.resolution,
            "n_iterations": p.n_iterations, "gene_activity": bool(p.activity), "upstream": UPSTREAM,
            "genome": p.genome, "gene_positions_url": url}


def save_run(adata, p: AtacPlan, res: AtacResult, *, resources: dict | None = None,
             cells_fingerprint: str | None = None) -> dict[str, Any]:
    """Store the run in ``adata`` (obs, obsm, obsp, uns, a provenance step). The info entry
    is replaced, never mutated."""
    from scview.core import provenance

    lsi = res.lsi
    adata.obsm[OBSM_LSI] = lsi.embedding
    adata.obsm[OBSM_UMAP] = res.umap
    adata.obsp[CONN_KEY] = res.connectivities
    adata.obsp[DIST_KEY] = res.distances
    adata.uns[GRAPH_KEY] = res.neighbors_uns
    adata.obs[OBS_CLUSTERS] = pd.Categorical(np.asarray(res.clusters).astype(str),
                                             categories=list(res.clusters.categories))
    if res.activity_total is not None:
        adata.obs[OBS_ACTIVITY_TOTAL] = res.activity_total.astype(np.float32)
    info = atac_import.read_info(adata) or {"version": atac_import.VERSION, "computed": []}
    created = datetime.now(timezone.utc).isoformat()
    info["lsi"] = {
        "run_id": f"{created}|{adata.n_obs}", "created_at": created, "plan": p.to_dict(),
        "sdev": [round(float(x), 5) for x in lsi.sdev], "variance_ratio": [round(float(x), 6) for x in lsi.variance_ratio],
        "depth_r": [round(float(x), 4) for x in lsi.depth_r], "depth_limit": DEPTH_LIMIT,
        "dropped_first": lsi.dropped_first, "dims_used": lsi.dims_used, "n_selected": lsi.n_selected,
        "cutoff_total": lsi.cutoff_total, "n_zero_cells": lsi.n_zero_cells, "top_peaks": lsi.loadings_peaks,
        "n_clusters": int(len(res.clusters.categories)), "cells_fingerprint": cells_fingerprint,
        "seconds": round(res.seconds, 1), "stage_seconds": dict(res.stage_seconds), "method": method_text(p, lsi),
    }
    if res.activity is not None:
        info["activity"] = res.activity
    computed = [c for c in (info.get("computed") or []) if c not in ("LSI", "gene activity")]
    computed.append("LSI")
    if res.activity is not None and res.activity.get("available"):
        computed.append("gene activity")
    info["computed"] = computed
    atac_import.write_info(adata, info)
    provenance.record_step(adata, step="atac_lsi", tool="scview.atac (Signac RunTFIDF / RunSVD)",
                           params=replay_params(p, lsi), resources=resources)
    return summary(adata, None, cells_fingerprint)


# ---------------------------------------------------------------------------
# Gene activity
# ---------------------------------------------------------------------------


@dataclass
class GeneChoice:
    """Which genes of the dataset have a position, and each gene's region."""

    names: list[str]  # the dataset's gene names, in table order
    ids: list[str]
    chrom: np.ndarray
    start: np.ndarray  # body + 2 kb upstream, 0-based half open
    end: np.ndarray
    n_dataset_genes: int = 0
    by_id: int = 0
    by_name: int = 0
    genome: str | None = None
    source: str | None = None
    reason: str | None = None  # why there are none (a missing annotation, no peak positions)


def _dataset_genes(adata) -> pd.DataFrame:
    """Gene names and ids: .raw's when it has more than X (a pipeline run narrows X)."""
    var = adata.var
    if adata.raw is not None and adata.raw.n_vars > adata.n_vars:
        var = adata.raw.var
    ids = var["gene_ids"].astype(str).str.replace(r"\.\d+$", "", regex=True).to_numpy() \
        if "gene_ids" in var.columns else np.full(len(var), "", dtype=object)
    return pd.DataFrame({"name": var.index.astype(str), "gene_id": ids})


def gene_regions(table, upstream=2000):
    """(start, end) of body + ``upstream`` on the strand's side, floored at 0."""
    plus = table["strand"].to_numpy() == "+"
    start = table["start"].to_numpy(dtype=np.int64) - plus * upstream
    end = table["end"].to_numpy(dtype=np.int64) + (~plus) * upstream
    return np.maximum(start, 0), end


def choose_genes(adata, genome: str | None, data_dir: str | Path, auto_download: bool = True) -> GeneChoice:
    """The dataset's genes with a position (by gene id, else name). ``reason`` says why none."""
    from scview.core import gene_annotation

    empty = GeneChoice([], [], np.zeros(0, object), np.zeros(0, np.int64), np.zeros(0, np.int64))
    try:
        table = gene_annotation.load(genome, data_dir, auto_download=auto_download)
    except gene_annotation.AnnotationError as e:
        empty.reason = str(e)
        return empty
    genes = _dataset_genes(adata)
    by_id = table.drop_duplicates("gene_id").set_index("gene_id")
    by_name = table.drop_duplicates("gene_name").set_index("gene_name")
    id_hit = genes["gene_id"].isin(by_id.index).to_numpy()
    name_hit = genes["name"].isin(by_name.index).to_numpy()
    use = id_hit | name_hit
    picked = []
    for gene_id, name, hit in zip(genes["gene_id"].to_numpy()[use], genes["name"].to_numpy()[use], id_hit[use]):
        picked.append(by_id.loc[gene_id] if hit else by_name.loc[name])
    if not picked:
        empty.n_dataset_genes = len(genes)
        empty.reason = (f"None of this dataset's {len(genes):,} genes is in {gene_annotation.source_of(genome)}: "
                        "it is another species or reference.")
        return empty
    frame = pd.DataFrame(picked)
    start, end = gene_regions(frame)
    return GeneChoice(names=genes["name"].to_numpy()[use].tolist(), ids=genes["gene_id"].to_numpy()[use].tolist(),
                      chrom=frame["chrom"].to_numpy(dtype=object), start=start, end=end,
                      n_dataset_genes=len(genes), by_id=int(id_hit.sum()), by_name=int((name_hit & ~id_hit).sum()),
                      genome=gene_annotation.canonical_genome(genome), source=gene_annotation.source_of(genome))


def peak_positions(store: ModalityStore) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(chrom, start, end) per peak; start and end -1 where a peak names no interval."""
    chrom = store.feature_attr("chrom")
    start = pd.to_numeric(pd.Series(store.feature_attr("start")), errors="coerce").fillna(-1).to_numpy(dtype=np.int64)
    end = pd.to_numeric(pd.Series(store.feature_attr("end")), errors="coerce").fillna(-1).to_numpy(dtype=np.int64)
    return chrom, start, end


def overlap_matrix(chrom, start, end, genes):
    """Peaks x genes: 1 where a peak overlaps the gene's region (``genes``: its chrom, start,
    end and names)."""
    rows, cols = [], []
    gene_order = pd.Series(genes.chrom).groupby(pd.Series(genes.chrom)).groups
    peak_order = pd.Series(chrom).groupby(pd.Series(chrom)).groups
    for c, gene_idx in gene_order.items():
        idx = peak_order.get(c)
        if idx is None:
            continue
        idx = np.asarray(idx)
        idx = idx[np.argsort(start[idx], kind="stable")]
        s, e = start[idx], end[idx]
        reach = np.maximum.accumulate(e)
        gene_idx = np.asarray(gene_idx)
        hi = np.searchsorted(s, genes.end[gene_idx], side="left")  # peaks starting before the gene ends
        lo = np.searchsorted(reach, genes.start[gene_idx], side="right")  # none before this reaches the gene
        for g, a, b in zip(gene_idx, lo, hi):
            if b > a:
                hit = idx[a:b][e[a:b] > genes.start[g]]
                rows.append(hit)
                cols.append(np.full(hit.size, g, dtype=np.int64))
    if not rows:
        return sp.csr_matrix((chrom.size, len(genes.names)), dtype=np.float32)
    r, c = np.concatenate(rows), np.concatenate(cols)
    return sp.csr_matrix((np.ones(r.size, dtype=np.float32), (r, c)), shape=(chrom.size, len(genes.names)))


def write_activity(store: ModalityStore, genes: GeneChoice, data_dir: str | Path, dataset_id: str,
                   stop: Callable[[], bool] = lambda: False) -> tuple[dict[str, Any], np.ndarray]:
    """Peak counts per gene for every barcode in the store, by cell and by gene, written aside
    and renamed. Returns the summary and each barcode's total over all genes."""
    chrom, start, end = peak_positions(store)
    if not (start >= 0).any():
        return ({"available": False, "reason": "The peaks of this dataset name no positions, so they cannot be "
                                                "matched to genes."}, np.zeros(store.n_obs))
    m = overlap_matrix(chrom, start, end, genes)
    n_genes = len(genes.names)
    rows = np.arange(store.n_obs)
    totals = np.zeros(store.n_obs, dtype=np.float64)
    per_gene = np.zeros(n_genes, dtype=np.int64)
    folder = modality_store.store_dir(data_dir, dataset_id)
    path = folder / f"{ACTIVITY_STORE}.h5"
    meta = {"genome": genes.genome, "source": genes.source, "upstream": UPSTREAM, "kind": "peak-based"}
    with modality_store.StoreWriter(path, store.barcodes, genes.names, {"gene_id": genes.ids}, meta) as w:
        for pos, block in store.iter_blocks(MATRIX, rows, BLOCK_VALUES):
            if stop():
                raise AtacCancelled()
            act = (block @ m).tocsr()
            act.sort_indices()
            w.append(ACTIVITY_MATRIX, act)
            totals[pos] = np.asarray(act.sum(axis=1), dtype=np.float64).ravel()
            per_gene += np.bincount(act.indices, minlength=n_genes)
    _write_by_gene(path, folder / f"{ACTIVITY_BY_GENE}.h5", per_gene, meta, stop)
    n_with = int((per_gene > 0).sum())
    return ({"available": True, "n_genes": n_genes, "n_genes_with_peaks": n_with, "by_id": genes.by_id,
             "by_name": genes.by_name, "n_dataset_genes": genes.n_dataset_genes, "genome": genes.genome,
             "source": genes.source, "upstream": UPSTREAM, "nnz": int(per_gene.sum()), "caption": ACTIVITY_CAPTION}, totals)


def _write_by_gene(by_cell: Path, dest: Path, per_gene: np.ndarray, meta: dict, stop: Callable[[], bool]) -> None:
    """The cells x genes store turned by gene, without holding it: entries bucketed by gene
    range on disk, then each bucket sorted and written (one gene is one row of ``dest``)."""
    n_genes = per_gene.size
    cum = np.cumsum(per_gene)
    edges = [0]
    while edges[-1] < n_genes:
        base = cum[edges[-1] - 1] if edges[-1] else 0
        nxt = int(np.searchsorted(cum, base + BUCKET_VALUES, side="right"))
        edges.append(min(n_genes, max(nxt, edges[-1] + 1)))
    bucket_of = np.searchsorted(np.asarray(edges[1:]), np.arange(n_genes), side="right")
    rec = np.dtype([("g", "<i4"), ("c", "<i4"), ("v", "<f4")])
    with tempfile.TemporaryDirectory(prefix=".scview-activity-", dir=dest.parent) as tmp:
        with ModalityStore(by_cell) as src:
            rows = np.arange(src.n_obs)
            barcodes, names = src.barcodes, src.features
            for pos, block in src.iter_blocks(ACTIVITY_MATRIX, rows, BLOCK_VALUES):
                if stop():
                    raise AtacCancelled()
                cells = np.repeat(pos, np.diff(block.indptr))
                buckets = bucket_of[block.indices]
                order = np.argsort(buckets, kind="stable")
                split = np.flatnonzero(np.diff(buckets[order])) + 1
                for part in np.split(order, split):
                    if not part.size:
                        continue
                    out = np.empty(part.size, dtype=rec)
                    out["g"], out["c"], out["v"] = block.indices[part], cells[part], block.data[part]
                    with open(Path(tmp) / f"{int(buckets[part[0]])}.bin", "ab") as fh:
                        out.tofile(fh)
        with modality_store.StoreWriter(dest, names, barcodes, None, meta) as w:
            for b in range(len(edges) - 1):
                g0, g1 = edges[b], edges[b + 1]
                p = Path(tmp) / f"{b}.bin"
                r = np.fromfile(p, dtype=rec) if p.exists() else np.zeros(0, dtype=rec)
                m = sp.csr_matrix((r["v"], (r["g"] - g0, r["c"])), shape=(g1 - g0, len(barcodes)))
                w.append(ACTIVITY_MATRIX, m)
                p.unlink(missing_ok=True)


def activity_gene(adata, data_dir: str | Path, dataset_id: str, gene: str) -> dict[str, Any]:
    """One gene's activity for the dataset's cells: counts, and log1p per 10k for display."""
    path = store_path(data_dir, dataset_id, ACTIVITY_BY_GENE)
    if not path.exists():
        raise AtacError("No gene activity for this dataset yet: run ATAC analysis on the Accessibility tab.", 404)
    with ModalityStore(path) as by_gene:
        hit = by_gene.rows_of([gene])[0]
        if hit < 0:
            raise AtacError(f"'{gene}' has no gene activity: it is not in the dataset's genes with a known position.", 404)
        row = by_gene.read(ACTIVITY_MATRIX, [int(hit)])
        barcodes = by_gene.features
    index = pd.Index(barcodes)
    where = index.get_indexer(adata.obs_names.astype(str))
    if (where < 0).any():
        raise AtacError("This dataset's cell names changed after the gene activity was made. Run it again.", 409)
    dense = np.zeros(barcodes.size, dtype=np.float32)
    dense[row.indices] = row.data
    counts = dense[where]
    if OBS_ACTIVITY_TOTAL not in adata.obs.columns:
        raise AtacError("The activity totals are missing: run ATAC analysis on the Accessibility tab again.", 409)
    total = adata.obs[OBS_ACTIVITY_TOTAL].to_numpy(dtype=np.float64)
    values = np.log1p(np.divide(counts * SCALE_FACTOR, total, out=np.zeros(counts.shape), where=total > 0))
    return {"gene": gene, "counts": counts, "values": values.astype(np.float32), "caption": ACTIVITY_CAPTION}


REPLAY_CONSTANTS = ("LANCZOS_BLOCK", "LANCZOS_STEPS", "SEED", "SCALE_FACTOR")
REPLAY_CODE = (top_peaks, block_lanczos_svd, scaled_embedding, depth_correlation, gene_regions, overlap_matrix)


# ---------------------------------------------------------------------------
# Peak context
# ---------------------------------------------------------------------------


def annotate_peaks(chrom: np.ndarray, start: np.ndarray, end: np.ndarray, table: pd.DataFrame) -> pd.DataFrame:
    """Per peak: nearest gene by TSS, distance to that TSS (0 when inside the peak; positive
    downstream of the TSS on the gene's strand), class (promoter within 2 kb of a TSS, gene
    body, distal). Pseudogenes and small RNAs are left out of "nearest"."""
    keep = ~table["gene_type"].astype(str).str.contains("pseudogene") & ~table["gene_type"].isin(_SKIP_GENE_TYPES)
    t = table[keep]
    n = chrom.size
    near = np.full(n, "", dtype=object)
    dist = np.full(n, np.nan)
    klass = np.full(n, "distal", dtype=object)
    peak_groups = pd.Series(chrom).groupby(pd.Series(chrom)).groups
    for c, group in t.groupby("chrom", sort=False):
        idx = peak_groups.get(c)
        if idx is None:
            continue
        idx = np.asarray(idx)
        order = np.argsort(group["tss"].to_numpy(), kind="stable")
        tss = group["tss"].to_numpy()[order]
        gene = group["gene_name"].to_numpy(dtype=object)[order]
        strand = group["strand"].to_numpy(dtype=object)[order]
        s, e = start[idx], end[idx]
        right = np.searchsorted(tss, s, side="left")  # first TSS at or after the peak's start
        left = np.clip(right - 1, 0, tss.size - 1)
        right = np.clip(right, 0, tss.size - 1)
        def gap(t_):  # bases from the peak to a TSS; 0 inside
            return np.maximum(0, np.maximum(s - t_, t_ - (e - 1)))
        d_left, d_right = gap(tss[left]), gap(tss[right])
        pick = np.where(d_left <= d_right, left, right)
        d = np.minimum(d_left, d_right).astype(np.float64)
        centre = (s + e) / 2
        sign = np.where(strand[pick] == "-", -1.0, 1.0) * np.sign(centre - tss[pick])
        near[idx] = gene[pick]
        dist[idx] = d * np.where(sign == 0, 1.0, sign)
        bodies = group.sort_values("start")
        starts = bodies["start"].to_numpy()
        reach = np.maximum.accumulate(bodies["end"].to_numpy())
        i = np.searchsorted(starts, e, side="left")
        inside = (i > 0) & (reach[np.clip(i - 1, 0, reach.size - 1)] > s)
        klass[idx[inside]] = "gene body"
        klass[idx[d <= PROMOTER]] = "promoter"
    valid = start >= 0
    near[~valid] = ""
    klass[~valid] = ""
    dist[~valid] = np.nan
    return pd.DataFrame({"nearest_gene": near, "distance_to_tss": dist, "peak_class": klass})


# ---------------------------------------------------------------------------
# Differential accessibility
# ---------------------------------------------------------------------------


def _bh(p: np.ndarray) -> np.ndarray:
    n = p.size
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(ranked, 1.0)
    return out


def wilcoxon_sparse(values: np.ndarray, cells: np.ndarray, codes: np.ndarray, n_per_group: np.ndarray, n_cells: int
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Rank-sum z and two-sided p of one peak for every group vs the rest, from its non-zero
    values: the zeros (one big tie) are counted, never laid out. Normal approximation with the
    tie correction, no continuity correction. ``codes``: each cell's group, -1 for none (the
    cell then is in no comparison: ``n_cells`` counts the labelled cells)."""
    k = n_per_group.size
    labelled = codes[cells] >= 0
    v, c = values[labelled], codes[cells[labelled]]
    n_nz = v.size
    n_zero = n_cells - n_nz
    order = np.argsort(v, kind="stable")
    sv = v[order]
    if n_nz:
        first = np.concatenate([[True], sv[1:] != sv[:-1]])
        starts = np.flatnonzero(first)
        sizes = np.diff(np.append(starts, n_nz))
        avg = starts + (sizes + 1) / 2.0  # average 1-based rank within the non-zeros
        ranks_sorted = np.repeat(avg, sizes) + n_zero
        ties = float(np.sum(sizes.astype(np.float64) ** 3 - sizes))
        nz_sum = np.bincount(c[order], weights=ranks_sorted, minlength=k)
        nz_count = np.bincount(c, minlength=k)
    else:
        ties, nz_sum, nz_count = 0.0, np.zeros(k), np.zeros(k, dtype=np.int64)
    if n_zero > 1:
        ties += float(n_zero) ** 3 - n_zero
    rank_sum = nz_sum + (n_per_group - nz_count) * (n_zero + 1) / 2.0
    n1, n2 = n_per_group.astype(np.float64), (n_cells - n_per_group).astype(np.float64)
    u = rank_sum - n1 * (n1 + 1) / 2.0
    sigma2 = n1 * n2 / 12.0 * ((n_cells + 1) - ties / (n_cells * (n_cells - 1.0)))
    z = np.divide(u - n1 * n2 / 2.0, np.sqrt(np.maximum(sigma2, 0)), out=np.zeros(k), where=sigma2 > 0)
    return z, 2 * ndtr(-np.abs(z))


@dataclass
class DaResult:
    result_id: str
    column: str
    groups: list[str]
    table: pd.DataFrame  # per group, the KEEP_PER_GROUP lowest p
    n_tested: dict[str, int]
    n_significant: dict[str, int]
    params: dict[str, Any]
    method: str
    cells: str
    created_at: str


def group_codes(adata, column: str, groups: list[str] | None = None) -> tuple[np.ndarray, list[str], list[str]]:
    """Each cell's group code (-1 for a missing label), every group, and the ones tested."""
    if column not in adata.obs.columns:
        raise AtacError(f"Column '{column}' not found.", 404)
    s = adata.obs[column]
    labels = s.astype(str).where(s.notna()).to_numpy(dtype=object)
    present = {x for x in labels if x is not None and x == x}
    order = [str(c) for c in s.cat.categories] if hasattr(s, "cat") else sorted(present, key=str)
    every = [g for g in order if g in present]
    if len(every) > MAX_GROUPS:
        raise AtacError(f"'{column}' has {len(every)} groups; at most {MAX_GROUPS} can be tested. Pick a label column.")
    if len(every) < 2:
        raise AtacError(f"'{column}' has fewer than 2 groups.")
    code = {g: i for i, g in enumerate(every)}
    codes = np.fromiter((code.get(x, -1) for x in labels), dtype=np.int32, count=labels.size)
    chosen = every if not groups else [str(g) for g in groups]
    unknown = [g for g in chosen if g not in code]
    if unknown:
        raise AtacError(f"'{column}' has no group '{unknown[0]}'.", 404)
    return codes, every, chosen


def iter_differential(store: ModalityStore, rows: np.ndarray, codes: np.ndarray, every: list[str], chosen: list[str],
                      *, min_pct: float = MIN_PCT, max_peaks: int = MAX_PEAKS_PER_GROUP,
                      stop: Callable[[], bool] = lambda: False, batch_values: int = DA_BATCH_VALUES
                      ) -> Iterator[dict[str, Any]]:
    """Wilcoxon on TF-IDF, each chosen group vs every other labelled cell, as a generator
    (``progress`` events, then ``{"type": "result", "frame": ..., "n_tested": ...}``): one pass
    for detection, then one per batch of peaks (``batch_values`` stored values)."""
    n_vars = store.n_vars
    k = len(every)
    labelled = codes >= 0
    n_labelled = int(labelled.sum())
    n_per_group = np.bincount(codes[labelled], minlength=k).astype(np.int64)
    detect = np.zeros((k, n_vars), dtype=np.int64)
    total = np.zeros(n_vars, dtype=np.float64)
    cell_total = np.zeros(rows.size, dtype=np.float64)
    yield {"type": "progress", "step": "detection", "message": "Counting each peak's detection per group"}
    for pos, block in store.iter_blocks(MATRIX, rows, BLOCK_VALUES):
        if stop():
            raise AtacCancelled()
        per_cell = np.repeat(pos, np.diff(block.indptr))
        c = codes[per_cell]
        ok = c >= 0
        detect += np.bincount(c[ok].astype(np.int64) * n_vars + block.indices[ok], minlength=k * n_vars).reshape(k, n_vars)
        total += np.bincount(block.indices, weights=block.data, minlength=n_vars)
        cell_total[pos] = np.asarray(block.sum(axis=1), dtype=np.float64).ravel()
    rest = detect.sum(axis=0, keepdims=True) - detect
    pct_in = detect / n_per_group[:, None].clip(min=1)
    pct_out = rest / (n_labelled - n_per_group)[:, None].clip(min=1)
    tested = np.zeros((k, n_vars), dtype=bool)
    code_of = {g: i for i, g in enumerate(every)}
    for g in chosen:
        i = code_of[g]
        ok = np.flatnonzero((pct_in[i] >= min_pct) | (pct_out[i] >= min_pct))
        if ok.size > max_peaks:
            ok = ok[np.argsort(-np.maximum(pct_in[i][ok], pct_out[i][ok]), kind="stable")[:max_peaks]]
        tested[i, ok] = True
    union = np.flatnonzero(tested.any(axis=0))
    sizes = detect.sum(axis=0)[union]
    batches = np.split(union, np.flatnonzero(np.diff((np.cumsum(sizes) // max(1, batch_values)))) + 1) if union.size else []
    idf = np.where(total > 0, rows.size / np.where(total > 0, total, 1), 0.0)
    z = np.zeros((k, n_vars), dtype=np.float32)
    p = np.ones((k, n_vars), dtype=np.float64)
    fc = np.zeros((k, n_vars), dtype=np.float32)
    n_groups = n_per_group.astype(np.float64)
    for b, peaks in enumerate(batches):
        yield {"type": "progress", "step": "test", "message": f"Testing peaks, batch {b + 1} of {len(batches)}",
               "batch": b + 1, "batches": len(batches)}
        where = np.full(n_vars, -1, dtype=np.int32)
        where[peaks] = np.arange(peaks.size, dtype=np.int32)
        cols, cells, vals = [], [], []
        for pos, block in store.iter_blocks(MATRIX, rows, BLOCK_VALUES):
            if stop():
                raise AtacCancelled()
            mapped = where[block.indices]
            keep = mapped >= 0
            per_cell = np.repeat(pos, np.diff(block.indptr))[keep]
            tf = block.data[keep] / cell_total[per_cell]
            cols.append(mapped[keep])
            cells.append(per_cell.astype(np.int32))
            vals.append((tf * SCALE_FACTOR * idf[peaks[mapped[keep]]]).astype(np.float32))
        col = np.concatenate(cols) if cols else np.zeros(0, np.int32)
        cel = np.concatenate(cells) if cells else np.zeros(0, np.int32)
        val = np.concatenate(vals) if vals else np.zeros(0, np.float32)
        del cols, cells, vals
        order = np.argsort(col, kind="stable")
        col, cel, val = col[order], cel[order], val[order]
        ptr = np.searchsorted(col, np.arange(peaks.size + 1))
        for j, peak in enumerate(peaks):
            lo, hi = ptr[j], ptr[j + 1]
            v, c = val[lo:hi], cel[lo:hi]
            z_j, p_j = wilcoxon_sparse(v, c, codes, n_per_group, n_labelled)
            z[:, peak], p[:, peak] = z_j, p_j
            # Seurat's FoldChange on the data slot: means of expm1(TF-IDF), which is what val holds.
            ok = codes[c] >= 0
            sums = np.bincount(codes[c[ok]], weights=v[ok], minlength=k)
            mean_in = sums / n_groups.clip(min=1)
            mean_out = (sums.sum() - sums) / (n_labelled - n_per_group).clip(min=1)
            fc[:, peak] = np.log2((mean_in + 1) / (mean_out + 1))
    frames = []
    n_tested: dict[str, int] = {}
    n_significant: dict[str, int] = {}
    for g in chosen:
        i = code_of[g]
        ok = np.flatnonzero(tested[i])
        padj = _bh(p[i, ok])
        n_tested[g] = int(ok.size)
        n_significant[g] = int((padj < 0.05).sum())
        o = np.lexsort((-np.abs(z[i, ok]), p[i, ok]))[:KEEP_PER_GROUP]
        frames.append(pd.DataFrame({
            "group": g, "peak_index": ok[o].astype(np.int32), "z": z[i, ok][o], "log2fc": fc[i, ok][o],
            "pct_in": pct_in[i, ok][o].astype(np.float32), "pct_rest": pct_out[i, ok][o].astype(np.float32),
            "pval": p[i, ok][o], "pval_adj": padj[o]}))
    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    yield {"type": "result", "frame": frame, "n_tested": n_tested, "n_significant": n_significant}


def attach_context(frame: pd.DataFrame, store: ModalityStore, genome: str | None, data_dir: str | Path,
                   auto_download: bool = True) -> tuple[pd.DataFrame, str | None]:
    """Peak name, nearest gene, distance, class on a result table; the note when gene
    positions are missing (the table then has names only)."""
    from scview.core import gene_annotation

    out = frame.copy()
    idx = out["peak_index"].to_numpy()
    out.insert(1, "peak", store.features[idx].astype(str))
    chrom, start, end = peak_positions(store)
    try:
        table = gene_annotation.load(genome, data_dir, auto_download=auto_download)
    except gene_annotation.AnnotationError as e:
        for c in ("nearest_gene", "peak_class"):
            out[c] = ""
        out["distance_to_tss"] = np.nan
        return out, str(e)
    ann = annotate_peaks(chrom[idx], start[idx], end[idx], table)
    for c in ann.columns:
        out[c] = ann[c].to_numpy()
    return out, None


DA_METHOD = ("Wilcoxon rank-sum on TF-IDF, each group vs the rest (no latent covariate), for peaks detected in >= "
             "{min_pct:.0%} of the group or the rest, the {max_peaks:,} most detected at most; BH FDR within each "
             "group; log2 fold change of mean expm1(TF-IDF) + 1.")
_DA_LAST: dict[str, DaResult] = {}


def da_payload(res: DaResult, group: str | None = None, offset: int = 0, limit: int = 100) -> dict[str, Any]:
    """One result for the UI: its counts, and a page of rows of one group (each group's
    first page without one)."""
    t = res.table
    columns = [c for c in t.columns if c != "peak_index"]
    if group:
        rows = t[t["group"] == group].iloc[offset:offset + limit]
        pages = {group: json.loads(rows[columns].to_json(orient="records"))}
    else:
        pages = {g: json.loads(t[t["group"] == g].iloc[:limit][columns].to_json(orient="records")) for g in res.groups}
    return {"id": res.result_id, "column": res.column, "groups": list(res.groups), "n_tested": dict(res.n_tested),
            "n_significant": dict(res.n_significant), "rows": pages, "params": dict(res.params), "method": res.method,
            "kept_per_group": KEEP_PER_GROUP, "created_at": res.created_at}


def remember_da(adaptor, dataset_id: str, res: DaResult) -> None:
    from scview.core import results_store

    _DA_LAST[res.result_id] = res
    while len(_DA_LAST) > KEEP_RESULTS:
        _DA_LAST.pop(next(iter(_DA_LAST)))
    results_store.keep(adaptor, RESULT_KIND, res, result_id=res.result_id, key=res.column,
                       label=f"Differential accessibility by {res.column} ({len(res.groups)} groups)")


def recall_da(dataset_id: str, result_id: str) -> DaResult | None:
    from scview.core import results_store

    hit = _DA_LAST.get(result_id)
    if hit is not None:
        return hit
    return results_store.recall(dataset_id, result_id, RESULT_KIND)


def latest_da(adaptor) -> DaResult | None:
    from scview.core import results_store

    try:
        return results_store.latest(adaptor, RESULT_KIND)
    except Exception:  # noqa: BLE001 - a test double, an unreadable result
        return None


# ---------------------------------------------------------------------------
# WNN RNA + ATAC
# ---------------------------------------------------------------------------


@dataclass
class WnnAtacPlan:
    n_cells: int
    rep: str
    d_rna: int
    lsi_dims: list[int]  # 1-based
    k: int = wnn.K_NN
    knn_range: int = wnn.KNN_RANGE
    min_dist: float = wnn.DEFAULT_MIN_DIST
    resolution: float = wnn.DEFAULT_RESOLUTION
    n_iterations: int = -1
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["estimate_seconds"] = estimate_wnn_seconds(self.n_cells, self.n_iterations)
        d["estimate_bytes"] = wnn.estimate_bytes(self.n_cells, self.d_rna, len(self.lsi_dims))
        return d


def estimate_wnn_seconds(n_cells: int, n_iterations: int = -1) -> float:
    from scview.core import preflight

    n = max(0, int(n_cells))
    # PBMC 10k, 11k cells: 28 s with numba already compiled, 52 s right after a start.
    return round(10.0 + 1.6e-3 * n + preflight.umap_seconds(n, three_d=False)
                 + preflight.leiden_seconds(n, n_iterations), 0)


def plan_wnn(adata, *, lsi_first: int = 2, lsi_last: int | None = None, resolution: float | None = None) -> WnnAtacPlan:
    from scview.core import leiden

    if OBSM_LSI not in adata.obsm:
        raise AtacError("Run ATAC analysis first: WNN needs the LSI components of the Accessibility tab.")
    try:
        rep, d_r, notes = wnn._rna_rep(adata)
    except wnn.WnnError as e:
        raise AtacError(str(e)) from e
    width = int(np.asarray(adata.obsm[OBSM_LSI]).shape[1])
    last = width if lsi_last is None else int(lsi_last)
    if not 1 <= lsi_first < last <= width:
        raise AtacError(f"LSI components must lie between 1 and {width}, at least two of them.")
    res = wnn.DEFAULT_RESOLUTION if resolution is None else float(resolution)
    if not 0 < res <= 10:
        raise AtacError("Resolution must be above 0 and at most 10.")
    emb = wnn._last_params(adata, "embeddings") or {}
    return WnnAtacPlan(n_cells=int(adata.n_obs), rep=rep, d_rna=d_r, lsi_dims=list(range(lsi_first, last + 1)),
                       min_dist=float(emb["umap_min_dist"] if emb.get("umap_min_dist") is not None
                                      else wnn.DEFAULT_MIN_DIST),
                       resolution=res, n_iterations=leiden.recorded_iterations(adata), notes=notes)


@dataclass
class WnnAtacInputs:
    plan: WnnAtacPlan
    rna: np.ndarray
    lsi: np.ndarray
    fingerprint: str


def prepare_wnn(adata, p: WnnAtacPlan) -> WnnAtacInputs:
    rna = np.asarray(adata.obsm[p.rep])[:, :p.d_rna].astype(np.float32, copy=True)
    lsi = np.asarray(adata.obsm[OBSM_LSI])[:, [d - 1 for d in p.lsi_dims]].astype(np.float32, copy=True)
    return WnnAtacInputs(p, rna, lsi, _inputs_fingerprint(rna, lsi))


def _inputs_fingerprint(rna: np.ndarray, lsi: np.ndarray) -> str:
    h = hashlib.sha256(np.ascontiguousarray(rna, dtype=np.float32).tobytes())
    h.update(np.ascontiguousarray(lsi, dtype=np.float32).tobytes())
    return h.hexdigest()[:24]


@dataclass
class WnnAtacResult:
    rna_weight: np.ndarray
    connectivities: sp.csr_matrix
    distances: sp.csr_matrix
    umap: np.ndarray
    clusters: pd.Categorical
    seconds: float
    stage_seconds: dict[str, float]


def iter_wnn(inp: WnnAtacInputs, stop: Callable[[], bool] = lambda: False) -> Iterator[dict[str, Any]]:
    """FindMultiModalNeighbors on the RNA PCs and the LSI components (``core/wnn.py``, its
    second modality here LSI, cosine-normalised like any other), then UMAP and Leiden."""
    from scview.core import leiden

    p = inp.plan
    t0 = time.monotonic()
    names = [s for s, _ in WNN_STAGES]
    labels = dict(WNN_STAGES)
    spent: dict[str, float] = {}
    current: list[Any] = [None, t0]

    def begin(name: str) -> dict[str, Any]:
        if stop():
            raise AtacCancelled()
        now = time.monotonic()
        if current[0] is not None:
            spent[current[0]] = round(now - current[1], 2)
        current[0], current[1] = name, now
        return {"type": "stage", "stage": name, "label": labels[name], "index": names.index(name) + 1,
                "total": len(names), "elapsed_s": round(now - t0, 1)}

    steps = wnn.wnn_steps(inp.rna, inp.lsi, p.k, p.knn_range, wnn.EXACT_MAX_CELLS, wnn.SD_SCALE, second="lsi")
    while True:
        try:
            name = next(steps)
        except StopIteration as done:
            result = done.value
            break
        yield begin(name)
    conn, distances = wnn.wnn_graphs(result["idx"], result["dist"])
    holder = _holder(inp.rna.shape[0])
    holder.obsm[p.rep] = inp.rna
    holder.obsp[WNN_CONN_KEY] = conn.copy()
    holder.obsp[WNN_DIST_KEY] = distances
    holder.uns[WNN_GRAPH_KEY] = _wnn_uns(p)
    yield begin("umap")
    umap = wnn._umap(holder, WNN_GRAPH_KEY, OBSM_WNN, p.min_dist)
    del holder
    yield begin("leiden")
    clusters = leiden.labels(conn, p.resolution, n_iterations=p.n_iterations)
    now = time.monotonic()
    spent[current[0]] = round(now - current[1], 2)
    yield {"type": "result", "result": WnnAtacResult(result["rna_weight"].astype(np.float32), conn, distances, umap,
                                                     clusters, now - t0, spent)}


def _wnn_uns(p: WnnAtacPlan) -> dict[str, Any]:
    return {"connectivities_key": WNN_CONN_KEY, "distances_key": WNN_DIST_KEY,
            "params": {"method": "umap", "n_neighbors": p.k + 1, "use_rep": p.rep, "n_pcs": p.d_rna,
                       "metric": "euclidean", "random_state": SEED, "wnn": True, "lsi_dims": p.lsi_dims}}


def save_wnn(adata, inp: WnnAtacInputs, res: WnnAtacResult, *, resources: dict | None = None,
             cells_fingerprint: str | None = None) -> dict[str, Any]:
    from scview.core import provenance

    p = inp.plan
    created = datetime.now(timezone.utc).isoformat()
    adata.obs[OBS_WNN_WEIGHT] = res.rna_weight
    adata.obs[OBS_WNN_CLUSTERS] = pd.Categorical(np.asarray(res.clusters).astype(str),
                                                 categories=list(res.clusters.categories))
    adata.obsm[OBSM_WNN] = res.umap
    adata.obsp[WNN_CONN_KEY] = res.connectivities
    adata.obsp[WNN_DIST_KEY] = res.distances
    adata.uns[WNN_GRAPH_KEY] = _wnn_uns(p)
    info = atac_import.read_info(adata) or {"version": atac_import.VERSION, "computed": []}
    info["wnn"] = {"run_id": f"{created}|{adata.n_obs}", "created_at": created, "plan": p.to_dict(),
                   "n_clusters": int(len(res.clusters.categories)),
                   "mean_rna_weight": round(float(np.mean(res.rna_weight)), 4), "cells_fingerprint": cells_fingerprint,
                   "inputs_fingerprint": inp.fingerprint, "seconds": round(res.seconds, 1),
                   "stage_seconds": dict(res.stage_seconds),
                   "method": (f"Weighted nearest neighbours (Hao et al. 2021; Seurat FindMultiModalNeighbors, k = {p.k}, "
                              f"{p.knn_range} candidates per modality): RNA {p.rep} (first {p.d_rna} PCs), ATAC LSI "
                              f"{p.lsi_dims[0]}:{p.lsi_dims[-1]}, rows L2-normalised; Leiden at resolution "
                              f"{p.resolution:g}.")}
    computed = [c for c in (info.get("computed") or []) if c != "WNN"]
    info["computed"] = [*computed, "WNN"]
    atac_import.write_info(adata, info)
    provenance.record_step(adata, step="atac_wnn", tool="scview.atac (Seurat FindMultiModalNeighbors algorithm)",
                           params={"rep": p.rep, "n_pcs_rna": p.d_rna, "lsi_dims": p.lsi_dims, "k": p.k,
                                   "knn_range": p.knn_range, "min_dist": p.min_dist, "resolution": p.resolution,
                                   "n_iterations": p.n_iterations, "exact_max_cells": wnn.EXACT_MAX_CELLS},
                           resources=resources)
    return summary(adata, None, cells_fingerprint)


def weight_by_group(adata, column: str) -> dict[str, Any]:
    labels, groups = wnn._groups(adata, column)
    w = adata.obs[OBS_WNN_WEIGHT].to_numpy(dtype=np.float64)
    rows = []
    for g in groups:
        v = w[labels == g]
        rows.append({"group": g, "n_cells": int(v.size), "mean_rna_weight": round(float(v.mean()), 4),
                     "median_rna_weight": round(float(np.median(v)), 4)})
    return {"column": column, "rows": rows}


# ---------------------------------------------------------------------------
# What the UI and the co-pilot read
# ---------------------------------------------------------------------------


def summary(adata, groupby: str | None = None, cells_fingerprint: str | None = None) -> dict[str, Any]:
    info = atac_import.read_info(adata) or {}
    lsi = info.get("lsi")
    w = info.get("wnn")
    out: dict[str, Any] = {"info": {k: v for k, v in info.items() if k not in ("lsi", "wnn", "activity")},
                           "lsi": None, "wnn": None, "activity": info.get("activity")}
    if lsi and OBSM_LSI in adata.obsm:
        out["lsi"] = {**lsi, "stale": _stale(lsi, cells_fingerprint)}
        out["lsi"]["clusters_column"] = OBS_CLUSTERS if OBS_CLUSTERS in adata.obs.columns else None
    # A reset to counts clears the embeddings: the columns left behind are not a result any more.
    if w and OBS_WNN_WEIGHT in adata.obs.columns and OBSM_WNN in adata.obsm:
        out["wnn"] = {**w, "stale": _stale(w, cells_fingerprint)}
        out["wnn"]["clusters_column"] = OBS_WNN_CLUSTERS if OBS_WNN_CLUSTERS in adata.obs.columns else None
        if groupby:
            try:
                out["wnn"]["by_group"] = weight_by_group(adata, groupby)
            except wnn.WnnError as e:
                out["wnn"]["by_group"] = None
                out["wnn"]["by_group_error"] = str(e)
    return out


def _stale(section: dict, fingerprint: str | None) -> bool:
    return fingerprint is not None and section.get("cells_fingerprint") not in (None, fingerprint)


QC_LABELS = {atac_import.OBS_TOTAL: "Peak counts per cell", atac_import.OBS_N_PEAKS: "Peaks detected per cell",
             atac_import.OBS_FRIP: "Fraction of fragments in peaks (FRiP)",
             atac_import.OBS_TSS: "Fraction of fragments at TSS", atac_import.OBS_MITO: "Mitochondrial read fraction"}


def qc_values(adata, groupby: str | None = None) -> dict[str, Any]:
    """Each ATAC QC column per cell (rounded), and each cell's group."""
    columns = []
    for key in atac_import.QC_COLUMNS:
        if key in adata.obs.columns:
            v = adata.obs[key].to_numpy(dtype=np.float64)
            digits = 0 if key in (atac_import.OBS_TOTAL, atac_import.OBS_N_PEAKS) else 4
            columns.append({"key": key, "label": QC_LABELS[key],
                            "values": [None if not np.isfinite(x) else round(float(x), digits) for x in v.tolist()]})
    out: dict[str, Any] = {"columns": columns, "n_cells": int(adata.n_obs), "groups": None, "group_order": None,
                           "column": groupby or None}
    if groupby:
        labels, groups = wnn._groups(adata, groupby)
        out["groups"] = [None if x is None or x != x else str(x) for x in labels]
        out["group_order"] = groups
    return out


def peak_table(store: ModalityStore, rows: np.ndarray, data_dir: str | Path, *, stats: PeakStats | None = None,
               auto_download: bool = True) -> tuple[pd.DataFrame, str | None]:
    """Every peak with its totals over the dataset's cells and its context."""
    from scview.core import gene_annotation

    s = stats if stats is not None else peak_stats(store, rows)
    chrom, start, end = peak_positions(store)
    frame = pd.DataFrame({"peak": store.features.astype(str), "chrom": chrom, "start": start, "end": end,
                          "total": s.total, "n_cells": s.n_cells,
                          "pct_cells": s.n_cells / max(1, rows.size)})
    selected = np.zeros(store.n_vars, dtype=bool)
    selected[top_peaks(s.total)] = True
    frame["selected"] = selected
    note = None
    try:
        table = gene_annotation.load(store.meta.get("genome"), data_dir, auto_download=auto_download)
    except gene_annotation.AnnotationError as e:
        note = str(e)
        frame["nearest_gene"], frame["distance_to_tss"], frame["peak_class"] = "", np.nan, ""
        return frame, note
    ann = annotate_peaks(chrom, start, end, table)
    return pd.concat([frame, ann], axis=1), note


def _stats_cache(data_dir: str | Path, dataset_id: str, cells: str) -> Path:
    return modality_store.store_dir(data_dir, dataset_id) / f"peak_stats_{cells[:16]}.npz"


def cached_stats(store: ModalityStore, rows: np.ndarray, data_dir: str | Path, dataset_id: str, cells: str) -> PeakStats:
    """Peak totals over these cells, kept beside the store (a pass over it takes seconds)."""
    path = _stats_cache(data_dir, dataset_id, cells)
    if path.exists():
        try:
            with np.load(path) as z:
                return PeakStats(z["total"], z["n_cells"], z["cell_total"])
        except (OSError, ValueError, KeyError):
            path.unlink(missing_ok=True)
    s = peak_stats(store, rows)
    part = path.with_name(f".{path.name}.part.npz")
    np.savez(part, total=s.total, n_cells=s.n_cells, cell_total=s.cell_total)
    part.replace(path)
    return s


def export_h5ad(adata, data_dir: str | Path, dataset_id: str, stop: Callable[[], bool] = lambda: False) -> Path:
    """An h5ad of the peak counts of the dataset's cells (cells x peaks; var chrom, start,
    end, the dataset's obs), for Signac and muon users. Written a block at a time and kept
    beside the store until the cells change."""
    import anndata as ad
    import h5py

    from scview.core import results_store
    from scview.core.ingestion import feature_split as fs

    with open_store(adata, data_dir, dataset_id) as store:
        rows = cell_rows(store, adata)
        cells = results_store.cells_hash(adata)
        folder = modality_store.store_dir(data_dir, dataset_id) / EXPORT_DIR
        folder.mkdir(parents=True, exist_ok=True)
        dest = folder / f"peaks_{cells[:16]}.h5ad"
        if dest.exists():
            return dest
        for old in folder.glob("peaks_*.h5ad"):
            old.unlink(missing_ok=True)
        chrom, start, end = peak_positions(store)
        var = pd.DataFrame({"chrom": chrom, "start": start, "end": end, "feature_types": atac_import.PEAKS_TYPE,
                            "genome": store.meta.get("genome") or ""}, index=store.features.astype(str))
        part = folder / f".{dest.name}.part"
        meta_part = folder / f".{dest.name}.meta"
        try:
            with h5py.File(part, "w") as f:
                g = f.create_group("X")
                g.attrs["encoding-type"] = "csr_matrix"
                g.attrs["encoding-version"] = "0.1.0"
                g.attrs["shape"] = np.asarray((rows.size, store.n_vars), dtype=np.int64)
                chunk = (1 << 16,)
                g.create_dataset("data", shape=(0,), maxshape=(None,), dtype="f4", chunks=chunk, **fs._GZIP)
                g.create_dataset("indices", shape=(0,), maxshape=(None,), dtype="i4", chunks=chunk, **fs._GZIP)
                indptr = np.zeros(rows.size + 1, dtype=np.int64)
                for pos, block in store.iter_blocks(MATRIX, rows, BLOCK_VALUES):
                    if stop():
                        raise AtacCancelled()
                    fs._append(g["data"], block.data)
                    fs._append(g["indices"], block.indices)
                    indptr[pos + 1] = np.diff(block.indptr)
                np.cumsum(indptr, out=indptr)
                g.create_dataset("indptr", data=indptr, **fs._GZIP)
            meta = ad.AnnData(obs=adata.obs.copy(), var=var)
            meta.write_h5ad(meta_part, compression="gzip")
            with h5py.File(meta_part, "r") as fm, h5py.File(part, "a") as fo:
                for k, v in fm.attrs.items():
                    fo.attrs[k] = v
                for key in fm.keys():
                    if key not in fo:
                        fm.copy(fm[key], fo, name=key)
            part.replace(dest)
        finally:
            part.unlink(missing_ok=True)
            meta_part.unlink(missing_ok=True)
    return dest


def provenance_history(adata) -> list[dict[str, Any]]:
    from scview.core import provenance

    return provenance.read_provenance(adata).get("history", [])


def grounding_lines(adata, data_dir: str | Path | None = None) -> tuple[list[str], str]:
    """The ATAC results as facts for the co-pilot (nothing without ATAC)."""
    from scview.core.qc_filter import label_grouping

    info = atac_import.read_info(adata)
    if not info:
        return [], ""
    lsi = info.get("lsi") or {}
    detail = f"{info.get('n_peaks', 0):,} peaks on {info.get('genome') or 'an unnamed genome'}"
    lines = ["\n## Chromatin accessibility (10x Multiome ATAC)",
             f"- ATAC import: {detail}; {info.get('nnz', 0):,} non-zero values on disk; computed: "
             f"{', '.join(info.get('computed') or []) or 'nothing yet'} [{TAG}]"]
    for key, label in ((atac_import.OBS_TOTAL, "median peak counts per cell"),
                       (atac_import.OBS_FRIP, "median fraction of fragments in peaks"),
                       (atac_import.OBS_TSS, "median fraction of fragments at TSS")):
        if key in adata.obs.columns:
            lines.append(f"- ATAC {label}: {float(np.nanmedian(adata.obs[key].to_numpy(dtype=np.float64))):,.3g} [{TAG}]")
    if lsi and OBS_CLUSTERS in adata.obs.columns and OBSM_LSI in adata.obsm:
        r = (lsi.get("depth_r") or [0.0])[0]
        kept = "left out" if lsi.get("dropped_first") else "kept"
        dims = lsi.get("dims_used") or [1, 2]
        lines.append(f"- ATAC LSI: {lsi.get('n_selected', 0):,} peaks selected; first component r = {r:.2f} with "
                     f"sequencing depth, {kept}; graph on components {dims[0]}:{dims[-1]}; {lsi.get('n_clusters')} ATAC "
                     f"clusters (obs {OBS_CLUSTERS}) [{TAG}]")
    activity = info.get("activity") or {}
    if activity.get("available"):
        lines.append(f"- ATAC gene activity: {activity.get('n_genes_with_peaks', 0):,} genes have at least one overlapping "
                     f"peak of {activity.get('n_genes', 0):,}; peak-based (body + 2 kb upstream), not fragment-based [{TAG}]")
    w = info.get("wnn") or {}
    if w and OBS_WNN_WEIGHT in adata.obs.columns and OBSM_WNN in adata.obsm:
        column, _why = label_grouping(adata)
        parts: list[str] = []
        if column:
            try:
                rows = sorted(weight_by_group(adata, column)["rows"], key=lambda r: -r["mean_rna_weight"])
                parts = [f"{r['group']} RNA {r['mean_rna_weight']:.2f} (n = {r['n_cells']:,})"
                         for r in rows if r["n_cells"] >= 10]
            except wnn.WnnError:
                parts = []
        lines.append(f"- WNN RNA + ATAC: mean RNA weight {w.get('mean_rna_weight', 0):.2f}; {w.get('n_clusters')} clusters "
                     f"(obs {OBS_WNN_CLUSTERS})" + (f"; mean RNA weight per {column}: " + "; ".join(parts) if parts else "")
                     + f". A cell's weights are relative: high RNA weight means RNA predicted its neighbours better than "
                       f"ATAC, not that its ATAC is poor [{TAG}]")
    return lines, detail


def drop_saved(adata) -> None:
    """Remove the computed results (a merge made them on other cells); the store stays."""
    for key in (*RUN_OBS, *WNN_OBS, OBS_ACTIVITY_TOTAL):
        if key in adata.obs.columns:
            del adata.obs[key]
    for key in (*RUN_OBSM, *WNN_OBSM):
        adata.obsm.pop(key, None)
    for key in (*RUN_OBSP, *WNN_OBSP):
        if key in adata.obsp:
            del adata.obsp[key]
    for key in (GRAPH_KEY, WNN_GRAPH_KEY):
        adata.uns.pop(key, None)


