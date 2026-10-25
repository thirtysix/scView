"""Copy-number inference from expression, inferCNV-style (N23).

Tumour cells often carry large gains and losses of chromosome arms; the genes in
a gained region are expressed a little higher, all together. One gene says
little, but the average over ~100 neighbouring genes does. Comparing that
running average with normal cells of the same sample separates malignant from
normal cells (Tirosh et al. 2016, Science 352:189, PMID 27124452, the inferCNV
method; infercnvpy, BSD-3, is the Python version this follows step by step,
written here from its documented algorithm).

The steps:

1. **Genes.** Each gene of the expression matrix is placed on the genome from the
   shipped Ensembl release 113 tables (``resources/gene_positions_*.tsv.gz``,
   GRCh38 or GRCm39; ``backend/scripts/make_gene_positions.py``): by Ensembl id
   when the dataset has one (var_names or an id column), else by symbol.
   Chromosomes Y and MT are left out by default (Y follows sex, not tumour, and
   MT is not a chromosome arm); genes detected in fewer than ``min_frac`` of the
   cells are dropped (their running mean is mostly noise). Genes are ordered by
   chromosome, then start.
2. **Expression.** Log-normalised values: ``.raw`` (the pipeline keeps them
   there), else X when it holds log values, else log1p of counts per 10,000.
3. **Relative to the reference.** For each reference group (normal cells: immune,
   stromal), the mean of each gene. With one group, a cell's value minus that
   mean; with several, infercnvpy's bounded difference: 0 while the value lies
   between the groups' lowest and highest mean, else the distance to the nearer
   bound, so a gene that only one normal type expresses (HLA, immunoglobulins)
   does not look like a gain. Clipped to +-3.
4. **Smoothing.** Per chromosome, a pyramid-weighted running mean over
   ``window`` genes (weights 1, 2, ..., n/2, ..., 2, 1), only whole windows,
   every ``step``-th window kept. A chromosome with fewer genes than the window
   gets one plain mean.
5. **Centring.** Each cell's median over the windows is subtracted.
6. **Denoising.** Values smaller than ``threshold`` x the standard deviation of
   all values are set to 0. infercnvpy takes that deviation per chunk of 5,000
   cells; here it is over all cells (the same below 5,000 cells).

Outputs: obsm ``X_cnv`` (cells x windows, float32), obs ``cnv_score`` (a cell's
mean absolute value over the windows), obs ``cnv_leiden`` (PCA of X_cnv, 50
components, no centring; 15-neighbour graph; Leiden at resolution 1, all as
infercnvpy's ``tl.pca``/``pp.neighbors``/``tl.leiden``), uns ``scview_cnv``.

What it cannot say: this is inferred from expression, not measured on DNA. It
needs a true normal reference from the same samples; a reference of another
tissue or batch shows as "copy number". Low depth gives noisy profiles, and
focal changes much smaller than a window are invisible.
"""

from __future__ import annotations

import base64
import gzip
import json
import logging
import re
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp

logger = logging.getLogger(__name__)

OBSM_KEY = "X_cnv"
UNS_KEY = "scview_cnv"
SCORE_COL = "cnv_score"
CLUSTER_COL = "cnv_leiden"
TAG = "result:cnv"
VERSION = 1

RESOURCES = Path(__file__).resolve().parents[1] / "resources"
GENOMES = {
    "human": {"file": "gene_positions_human.tsv.gz", "assembly": "GRCh38", "n_autosomes": 22,
              "sha256": "fee78469b283366bd311467d6905ef7eb518559f83212a0fab285e79070c2c46"},
    "mouse": {"file": "gene_positions_mouse.tsv.gz", "assembly": "GRCm39", "n_autosomes": 19,
              "sha256": "1b5015e01bd3992740a9e67d96d55e3f1ee2fd466ab097cd5e6a639777cce5f5"},
}
ENSEMBL_RELEASE = 113
SOURCE_URL = "https://ftp.ensembl.org/pub/release-113/gtf/"

DEFAULT_WINDOW = 100
DEFAULT_STEP = 10
DEFAULT_EXCLUDE = ("chrY", "chrMT")
DEFAULT_MIN_FRAC = 0.01
DEFAULT_THRESHOLD = 1.5
LFC_CLIP = 3.0
MIN_WINDOW, MAX_WINDOW = 11, 501
MAX_STEP = 100
MIN_GENES = 200  # positioned genes needed for a profile worth looking at
MIN_REF_CELLS = 10
N_PCS = 50
N_NEIGHBORS = 15
RESOLUTION = 1.0
SEED = 0
TARGET_SUM = 1e4
CHUNK = 1000  # cells per block: a block is dense (cells x genes; ~18 bytes per value at its peak)
HIGH_RATIO = 2.0  # a group "high" when its mean score is >= this x the reference cells' mean
CHROM_CALL = 0.3  # chromosome mean >= this x the denoising cut: called a gain / loss
MAX_HEATMAP_CELLS = 3000
MAX_HEATMAP_COLS = 500

_ENS_RE = re.compile(r"^(ENS[A-Z]*G\d{6,})(\.\d+)?$")
_ID_COLUMNS = ("gene_ids", "gene_id", "ensembl_id", "ensembl", "gene_ensembl", "feature_id", "id")


class CnvError(Exception):
    """A request the dataset cannot satisfy, in a sentence for the user."""


class CnvCancelled(Exception):
    """The run was stopped (the client went away); nothing is saved."""


# ---------------------------------------------------------------------------
# Gene positions
# ---------------------------------------------------------------------------


@lru_cache(maxsize=2)
def load_positions(genome: str) -> pd.DataFrame:
    """The shipped table for ``genome``: symbol, gene_id, chromosome, start, end,
    in genomic order (chr1..chrN, X, Y, MT)."""
    if genome not in GENOMES:
        raise CnvError(f"No gene positions for '{genome}': human (GRCh38) or mouse (GRCm39).")
    path = RESOURCES / GENOMES[genome]["file"]
    with gzip.open(path, "rt") as f:
        table = pd.read_csv(f, sep="\t", comment="#", dtype={"symbol": str, "gene_id": str,
                                                             "chromosome": str})
    table["symbol"] = table["symbol"].fillna("")
    return table


def chromosome_order(genome: str) -> list[str]:
    n = GENOMES[genome]["n_autosomes"]
    return [f"chr{i}" for i in range(1, n + 1)] + ["chrX", "chrY", "chrMT"]


def pick_genome(adata, genome: str = "auto") -> tuple[str, str]:
    """(genome, why) from the request or the species call (core/species.py)."""
    if genome in GENOMES:
        return genome, f"{genome} chosen"
    if genome != "auto":
        raise CnvError(f"Unknown genome '{genome}': auto, human or mouse.")
    from scview.core.species import detect_species

    call = detect_species(adata)
    if call.species not in GENOMES:
        raise CnvError("The species of this dataset's genes is unclear, so the genes cannot be placed on "
                       "a genome: choose human or mouse. " + call.evidence)
    return call.species, call.evidence


def _strip_version(v: Any) -> str | None:
    m = _ENS_RE.match(str(v))
    return m.group(1) if m else None


def ensembl_ids(adata, names: pd.Index) -> np.ndarray:
    """An Ensembl gene id (no version) per name, or None: from the names themselves,
    else an id column of ``.raw.var`` or ``var`` (by name)."""
    ids = np.array([_strip_version(n) for n in names], dtype=object)
    if np.count_nonzero(ids != None) >= 0.5 * len(names):  # noqa: E711 — elementwise
        return ids
    frames = []
    raw = getattr(adata, "raw", None)
    if raw is not None:
        frames.append(raw.var)
    frames.append(adata.var)
    for frame in frames:
        for col in frame.columns:
            if str(col).lower() not in _ID_COLUMNS or pd.api.types.is_numeric_dtype(frame[col]):
                continue
            col_ids = pd.Series([_strip_version(v) for v in frame[col].astype(str)], index=frame.index)
            if col_ids.notna().sum() < 0.5 * len(frame):
                continue
            fill = col_ids.reindex(names).to_numpy(dtype=object)
            fill = np.array([v if isinstance(v, str) else None for v in fill], dtype=object)
            missing = ids == None  # noqa: E711
            ids[missing] = fill[missing]
    return ids


def place_genes(names: pd.Index, ids: np.ndarray, genome: str,
                exclude: tuple[str, ...] | list[str] = DEFAULT_EXCLUDE) -> tuple[pd.DataFrame, dict[str, int]]:
    """The dataset's genes on the genome: one row per placed gene (``col``: its
    column in the matrix; chromosome, start, end, gene), genomic order, excluded
    chromosomes left out. With counts of how they were matched."""
    table = load_positions(genome)
    by_id = pd.Series(np.arange(len(table)), index=table["gene_id"].to_numpy())
    by_id = by_id[~by_id.index.duplicated()]
    sym = table["symbol"].to_numpy(dtype=object)
    has = sym != ""
    by_sym = pd.Series(np.flatnonzero(has), index=sym[has])
    by_sym = by_sym[~by_sym.index.duplicated()]
    by_upper = pd.Series(by_sym.to_numpy(), index=[s.upper() for s in by_sym.index])
    by_upper = by_upper[~by_upper.index.duplicated()]
    hit_id = pd.Series(ids, dtype=object).map(by_id).to_numpy(dtype=np.float64)
    name_s = pd.Series([str(n) for n in names], dtype=object)
    hit_sym = name_s.map(by_sym).to_numpy(dtype=np.float64)
    hit_up = name_s.str.upper().map(by_upper).to_numpy(dtype=np.float64)
    hit_sym = np.where(np.isnan(hit_sym), hit_up, hit_sym)
    use_id = ~np.isnan(hit_id)
    row = np.where(use_id, hit_id, hit_sym)
    row = np.where(np.isnan(row), -1, row).astype(np.int64)
    n_id = int(use_id.sum())
    n_sym = int(((row >= 0) & ~use_id).sum())
    placed = np.flatnonzero(row >= 0)
    t = table.iloc[row[placed]]
    out = pd.DataFrame({"col": placed, "gene": [str(names[i]) for i in placed],
                        "chromosome": t["chromosome"].to_numpy(), "start": t["start"].to_numpy(),
                        "end": t["end"].to_numpy()})
    order = {c: i for i, c in enumerate(chromosome_order(genome))}
    out["chrom_rank"] = out["chromosome"].map(order)
    n_excluded = int(out["chromosome"].isin(list(exclude)).sum())
    out = out[~out["chromosome"].isin(list(exclude))]
    out = out.sort_values(["chrom_rank", "start", "col"], kind="stable").reset_index(drop=True)
    return out, {"n_genes": len(names), "by_id": n_id, "by_symbol": n_sym,
                 "unplaced": int(len(names) - len(placed)), "excluded": n_excluded}


# ---------------------------------------------------------------------------
# Expression
# ---------------------------------------------------------------------------


@dataclass
class Source:
    M: Any  # cells x genes, any matrix type
    names: pd.Index
    kind: str  # "log" or "counts"
    text: str


def expression_source(adata) -> Source:
    """The matrix to read log-normalised values from (see the module docstring)."""
    from scview.core import gene_programs as gp

    raw = adata.raw
    if raw is not None and gp._is_lognorm(raw.X):
        return Source(raw.X, pd.Index(raw.var_names), "log", "log-normalised .raw")
    if gp._is_lognorm(adata.X):
        return Source(adata.X, pd.Index(adata.var_names), "log", "log-normalised X")
    from scview.core.normalization import counts_all_genes

    try:
        M, names, where = counts_all_genes(adata)
    except ValueError:
        raise CnvError("No log-normalised expression or counts to infer copy number from: run "
                       "normalisation and the log transform first (X holds scaled values).") from None
    return Source(M, pd.Index(names), "counts", f"log1p of counts per {TARGET_SUM:,.0f} from {where}")


def log_block(src: Source, start: int, stop: int, cols: np.ndarray) -> np.ndarray:
    """Rows ``start:stop`` of the log values of columns ``cols``, dense float32."""
    block = src.M[start:stop]
    if src.kind == "counts":
        block = sp.csr_matrix(block, dtype=np.float32) if sp.issparse(block) else np.asarray(block, np.float32)
        totals = np.asarray(block.sum(axis=1), dtype=np.float64).ravel()
        scale = np.divide(TARGET_SUM, totals, out=np.zeros_like(totals), where=totals > 0).astype(np.float32)
        sub = block[:, cols]
        sub = sub.toarray() if sp.issparse(sub) else np.asarray(sub)
        out = sub.astype(np.float32, copy=False) * scale[:, None]
        np.log1p(out, out=out)
        return out
    sub = block[:, cols] if sp.issparse(block) else np.asarray(block)[:, cols]
    return (sub.toarray() if sp.issparse(sub) else np.asarray(sub)).astype(np.float32, copy=False)


def detected_fraction(src: Source, chunk: int = CHUNK) -> np.ndarray:
    """Share of cells with a non-zero value, per gene of the source matrix."""
    M = src.M
    n, m = M.shape
    if sp.issparse(M):
        csr = sp.csr_matrix(M) if not sp.isspmatrix_csr(M) else M
        if csr.nnz and (csr.data == 0).any():
            nz = np.bincount(csr.indices[csr.data != 0], minlength=m)
        else:
            nz = np.bincount(csr.indices, minlength=m)
        return nz / max(n, 1)
    nz = np.zeros(m, dtype=np.int64)
    for s in range(0, n, chunk):
        block = M[s:s + chunk]
        block = block.toarray() if sp.issparse(block) else np.asarray(block)
        nz += (block != 0).sum(axis=0)
    return nz / max(n, 1)


# ---------------------------------------------------------------------------
# The method
# ---------------------------------------------------------------------------


def window_weights(chroms: np.ndarray, window: int, step: int) -> tuple[sp.csc_matrix, pd.DataFrame]:
    """The smoothing as a matrix (genes x kept windows; column j holds window j's
    pyramid weights, summing to 1) and each window's chromosome and first / last
    gene index. ``chroms``: each gene's chromosome, in genomic order."""
    rows, cols, vals = [], [], []
    info = []
    j = 0
    bounds = np.flatnonzero(np.r_[True, chroms[1:] != chroms[:-1], True])
    for a, b in zip(bounds[:-1], bounds[1:]):
        n = b - a
        if window < n:
            r = np.arange(1, window + 1, dtype=np.float64)
            w = np.minimum(r, r[::-1])
            w /= w.sum()
            starts = np.arange(0, n - window + 1, step)
        else:
            w = np.full(n, 1.0 / n)
            starts = np.array([0])
        for s in starts:
            rows.append(np.arange(a + s, a + s + len(w)))
            cols.append(np.full(len(w), j))
            vals.append(w)
            info.append((str(chroms[a]), int(a + s), int(a + s + len(w) - 1)))
            j += 1
    W = sp.csc_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(len(chroms), j))
    return W, pd.DataFrame(info, columns=["chromosome", "first", "last"])


def reference_means(src: Source, cols: np.ndarray, labels: np.ndarray, groups: list[str],
                    chunk: int = CHUNK) -> np.ndarray:
    """Mean log value of each gene in each reference group (groups x genes)."""
    sums = np.zeros((len(groups), len(cols)), dtype=np.float64)
    counts = np.zeros(len(groups), dtype=np.int64)
    gidx = {g: i for i, g in enumerate(groups)}
    code = np.array([gidx.get(v, -1) if isinstance(v, str) else -1 for v in labels])
    for s in range(0, len(code), chunk):
        c = code[s:s + chunk]
        if not (c >= 0).any():
            continue
        keep = np.flatnonzero(c >= 0)
        block = log_block(src, s, s + len(c), cols)[keep]
        for i in np.unique(c[keep]):
            m = c[keep] == i
            sums[i] += block[m].sum(axis=0, dtype=np.float64)
            counts[i] += int(m.sum())
    return (sums / np.maximum(counts, 1)[:, None]).astype(np.float32)


def relative(block: np.ndarray, ref: np.ndarray, clip: float = LFC_CLIP) -> np.ndarray:
    """Step 3: the bounded difference from the reference means, clipped (in place)."""
    if ref.shape[0] == 1:
        block -= ref[0]
    else:
        lo, hi = ref.min(axis=0), ref.max(axis=0)
        above = block > hi
        below = block < lo
        out = np.zeros_like(block)
        np.subtract(block, hi, out=out, where=above)
        np.subtract(block, lo, out=out, where=below)
        block = out
    np.clip(block, -clip, clip, out=block)
    return block


def smooth_center(block: np.ndarray, W: sp.csc_matrix) -> np.ndarray:
    """Steps 4-5: running means (cells x windows), each cell's median subtracted;
    float64, as infercnvpy (values near the denoising cut stay on the same side)."""
    S = np.asarray(sp.csr_matrix(W.T) @ block.T.astype(np.float64)).T
    S -= np.median(S, axis=1, keepdims=True)
    return S


@dataclass
class CnvPlan:
    src: Source
    genes: pd.DataFrame  # placed + kept genes, genomic order
    W: sp.csc_matrix
    windows: pd.DataFrame
    genome: str
    genome_why: str
    ref_column: str
    ref_groups: list[str]
    ref_labels: np.ndarray  # per cell: its reference group or None
    n_ref_cells: int
    window: int
    step: int
    exclude: list[str]
    min_frac: float
    threshold: float | None
    match: dict[str, int]
    n_low: int  # placed genes dropped as detected in too few cells
    notes: list[str] = field(default_factory=list)

    @property
    def n_cells(self) -> int:
        return int(self.src.M.shape[0])


def clean_params(window: int | None, step: int | None, exclude: list[str] | None,
                 min_frac: float | None, threshold: float | None, genome: str) -> tuple:
    window = DEFAULT_WINDOW if window is None else int(window)
    step = DEFAULT_STEP if step is None else int(step)
    if not MIN_WINDOW <= window <= MAX_WINDOW:
        raise CnvError(f"The window must be {MIN_WINDOW} to {MAX_WINDOW} genes.")
    if not 1 <= step <= min(MAX_STEP, window):
        raise CnvError(f"The step must be 1 to {min(MAX_STEP, window)} genes (at most the window).")
    exclude = list(DEFAULT_EXCLUDE if exclude is None else exclude)
    valid = set(chromosome_order(genome))
    bad = [c for c in exclude if c not in valid]
    if bad:
        raise CnvError(f"Unknown chromosome(s) to leave out: {', '.join(bad)} (use chr1.., chrX, chrY, chrMT).")
    min_frac = DEFAULT_MIN_FRAC if min_frac is None else float(min_frac)
    if not 0 <= min_frac <= 0.5:
        raise CnvError("The minimum share of cells a gene is detected in must be 0 to 0.5.")
    if threshold is not None and not 0 <= threshold <= 5:
        raise CnvError("The denoising threshold must be 0 to 5 standard deviations (or off).")
    return window, step, exclude, min_frac, threshold


def plan(adata, ref_column: str, ref_groups: list[str], *, window: int | None = None,
         step: int | None = None, exclude: list[str] | None = None, min_frac: float | None = None,
         threshold: float | None = DEFAULT_THRESHOLD, genome: str = "auto") -> CnvPlan:
    """Check the request and work out the genes and windows (reads the matrix once,
    for the detection rates; nothing is copied)."""
    if adata.n_obs < 50:
        raise CnvError("Copy-number inference needs at least 50 cells.")
    if not ref_column or ref_column not in adata.obs.columns:
        raise CnvError("Choose the column that holds the normal reference cells (e.g. the cell types).")
    groups = [str(g) for g in dict.fromkeys(ref_groups or [])]
    if not groups:
        raise CnvError("Choose at least one reference group of normal cells (e.g. T cells, B cells, "
                       "myeloid or stromal cells).")
    col = adata.obs[ref_column]
    labels = col.astype(str).where(col.notna()).to_numpy(dtype=object)
    present = set(v for v in labels if isinstance(v, str))
    missing = [g for g in groups if g not in present]
    if missing:
        raise CnvError(f"Not in '{ref_column}': {', '.join(missing)}.")
    in_ref = np.array([isinstance(v, str) and v in groups for v in labels])
    sizes = {g: int((labels == g).sum()) for g in groups}
    small = [g for g, n in sizes.items() if n < MIN_REF_CELLS]
    if small:
        raise CnvError(f"Reference groups need at least {MIN_REF_CELLS} cells each: "
                       + ", ".join(f"{g} ({sizes[g]})" for g in small) + ".")
    if in_ref.all():
        raise CnvError("Every cell is in the reference: leave out the groups you want to test.")
    genome, why = pick_genome(adata, genome)
    window, step, exclude, min_frac, threshold = clean_params(window, step, exclude, min_frac, threshold,
                                                              genome)
    src = expression_source(adata)
    genes, match = place_genes(src.names, ensembl_ids(adata, src.names), genome, exclude)
    notes: list[str] = []
    frac = detected_fraction(src)
    low = frac[genes["col"].to_numpy()] < min_frac if len(genes) else np.zeros(0, bool)
    n_low = int(low.sum())
    genes = genes[~low].reset_index(drop=True)
    if len(genes) < MIN_GENES:
        raise CnvError(
            f"Only {len(genes)} genes could be placed on the {GENOMES[genome]['assembly']} genome and are "
            f"detected in at least {min_frac:.0%} of cells; copy-number inference needs {MIN_GENES}. "
            f"({match['unplaced']:,} of {match['n_genes']:,} genes have no position.)")
    if src.M.shape[1] < 3000:
        notes.append(f"The matrix holds only {src.M.shape[1]:,} genes (variable genes?): profiles are "
                     "coarser than with every gene.")
    W, windows = window_weights(genes["chromosome"].to_numpy(dtype=object), window, step)
    return CnvPlan(src=src, genes=genes, W=W, windows=windows, genome=genome, genome_why=why,
                   ref_column=ref_column, ref_groups=groups,
                   ref_labels=np.where(in_ref, labels, None), n_ref_cells=int(in_ref.sum()),
                   window=window, step=step, exclude=exclude, min_frac=min_frac, threshold=threshold,
                   match=match, n_low=n_low, notes=notes)


@dataclass
class CnvResult:
    X: np.ndarray  # cells x windows, float32
    score: np.ndarray
    labels: pd.Categorical
    noise_sd: float
    cut: float | None
    seconds: float


def iter_cnv(p: CnvPlan, stop: Callable[[], bool] = lambda: False,
             chunk: int = CHUNK) -> Iterator[dict[str, Any]]:
    """Run the method; yields ``progress`` dicts, then ``{"type": "result",
    "result": CnvResult}``. ``stop`` is checked between blocks: when it says so,
    CnvCancelled is raised and nothing is kept."""
    t0 = time.monotonic()
    cols = p.genes["col"].to_numpy()
    n = p.n_cells
    yield {"type": "progress", "stage": "reference", "done": 0, "total": n, "elapsed_s": 0.0}
    ref = reference_means(p.src, cols, p.ref_labels, p.ref_groups, chunk)
    if stop():
        raise CnvCancelled()
    X = np.empty((n, p.W.shape[1]), dtype=np.float32)
    total = total_sq = 0.0
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        block = relative(log_block(p.src, s, e, cols), ref)
        S = smooth_center(block, p.W)
        X[s:e] = S
        total += float(S.sum(dtype=np.float64))
        total_sq += float(np.square(S, dtype=np.float64).sum())
        yield {"type": "progress", "stage": "windows", "done": e, "total": n,
               "elapsed_s": round(time.monotonic() - t0, 1)}
        if stop():
            raise CnvCancelled()
    count = X.size
    mean = total / max(count, 1)
    sd = float(np.sqrt(max(total_sq / max(count, 1) - mean * mean, 0.0)))
    cut = None
    if p.threshold:
        cut = p.threshold * sd
        for s in range(0, n, chunk):
            blk = X[s:s + chunk]
            blk[np.abs(blk) < cut] = 0
    score = np.concatenate([np.abs(X[s:s + chunk]).mean(axis=1) for s in range(0, n, chunk)]
                           ).astype(np.float32)
    yield {"type": "progress", "stage": "clusters", "done": n, "total": n,
           "elapsed_s": round(time.monotonic() - t0, 1)}
    if stop():
        raise CnvCancelled()
    labels = cluster(X)
    yield {"type": "result", "result": CnvResult(X=X, score=score, labels=labels, noise_sd=sd, cut=cut,
                                                 seconds=time.monotonic() - t0)}


def run(p: CnvPlan, chunk: int = CHUNK) -> CnvResult:
    """``iter_cnv`` to its end."""
    for item in iter_cnv(p, chunk=chunk):
        if item["type"] == "result":
            return item["result"]
    raise CnvError("The run ended without a result.")  # pragma: no cover


def cluster(X: np.ndarray, n_pcs: int = N_PCS, n_neighbors: int = N_NEIGHBORS,
            resolution: float = RESOLUTION) -> pd.Categorical:
    """CNV clusters as infercnvpy makes them: PCA without centring (arpack), a
    15-neighbour graph, Leiden (in a child process, core/leiden.py)."""
    import anndata as ad
    import scanpy as sc

    from scview.core import leiden

    n, m = X.shape
    if n < 3 or m < 2 or not np.any(X):
        return pd.Categorical(["0"] * n, categories=["0"])
    # Sparse float64, the matrix infercnvpy hands to the same scanpy calls.
    tmp = ad.AnnData(X=sp.csr_matrix(X, dtype=np.float64))
    k = max(1, min(n_pcs, m - 1, n - 1))
    sc.tl.pca(tmp, n_comps=k, svd_solver="arpack", zero_center=False, random_state=SEED)
    sc.pp.neighbors(tmp, n_neighbors=min(n_neighbors, n - 1), use_rep="X_pca", random_state=SEED)
    return leiden.labels(tmp.obsp["connectivities"], resolution, random_state=SEED)


# ---------------------------------------------------------------------------
# Saved results
# ---------------------------------------------------------------------------


def method_text(p: CnvPlan) -> str:
    refs = ", ".join(p.ref_groups)
    thr = (f"values under {p.threshold:g} SD set to 0" if p.threshold else "no denoising")
    return (
        "Copy number inferred from expression, inferCNV-style (Tirosh et al. 2016; as infercnvpy): "
        f"log-normalised expression ({p.src.text}) of {len(p.genes):,} genes placed on "
        f"{GENOMES[p.genome]['assembly']} (Ensembl {ENSEMBL_RELEASE}) and detected in >= {p.min_frac:.0%} "
        f"of cells, relative to the reference ({refs} in '{p.ref_column}'; bounded difference "
        f"between the groups' means), clipped to +-{LFC_CLIP:g}; pyramid-weighted running mean over "
        f"{p.window} genes per chromosome, every {p.step}th window; each cell's median subtracted; "
        f"{thr}. Score: a cell's mean absolute value. Clusters: PCA (no centring), "
        f"{N_NEIGHBORS}-neighbour graph, Leiden at resolution {RESOLUTION:g}. Chromosomes left out: "
        f"{', '.join(p.exclude) or 'none'}."
    )


def replay_params(p: CnvPlan) -> dict[str, Any]:
    """What provenance records and the replay script reads."""
    return {
        "reference_column": p.ref_column, "reference_groups": list(p.ref_groups), "window": p.window,
        "step": p.step, "exclude": list(p.exclude), "min_frac": p.min_frac, "threshold": p.threshold,
        "lfc_clip": LFC_CLIP, "genome": p.genome, "assembly": GENOMES[p.genome]["assembly"],
        "ensembl_release": ENSEMBL_RELEASE, "source": p.src.kind if p.src.kind == "counts" else
        ("raw" if p.src.text.endswith(".raw") else "X"), "target_sum": TARGET_SUM,
        "n_pcs": N_PCS, "n_neighbors": N_NEIGHBORS, "resolution": RESOLUTION, "seed": SEED,
        "genes": p.genes["gene"].tolist(), "chromosomes": p.genes["chromosome"].tolist(),
    }


def _chrom_spans(windows: pd.DataFrame) -> list[dict[str, Any]]:
    out = []
    ch = windows["chromosome"].to_numpy(dtype=object)
    bounds = np.flatnonzero(np.r_[True, ch[1:] != ch[:-1], True])
    for a, b in zip(bounds[:-1], bounds[1:]):
        out.append({"chromosome": str(ch[a]), "start": int(a), "end": int(b)})  # windows a..b-1
    return out


def save(adata, p: CnvPlan, res: CnvResult, *, resources: dict | None = None,
         fingerprint: str | None = None) -> dict[str, Any]:
    """Store the result in ``adata`` (obsm, obs, uns, a provenance step); return
    the summary. Replaces the uns entry, so ``write_derived`` sees it changed."""
    from scview.core import provenance

    created = datetime.now(timezone.utc).isoformat()
    g = p.genes
    first, last = p.windows["first"].to_numpy(), p.windows["last"].to_numpy()
    ref_mask = np.array([v is not None for v in p.ref_labels])
    info = {
        "version": VERSION, "run_id": f"{created}|{adata.n_obs}", "created_at": created,
        "genome": p.genome, "assembly": GENOMES[p.genome]["assembly"], "ensembl_release": ENSEMBL_RELEASE,
        "genome_why": p.genome_why, "reference_column": p.ref_column, "reference_groups": list(p.ref_groups),
        "n_reference_cells": p.n_ref_cells, "n_cells": int(adata.n_obs), "n_genes": len(g),
        "n_windows": int(res.X.shape[1]), "match": p.match, "n_low": p.n_low, "source": p.src.text,
        "notes": list(p.notes), "chromosomes": _chrom_spans(p.windows),
        "params": {"window": p.window, "step": p.step, "exclude": list(p.exclude), "min_frac": p.min_frac,
                   "threshold": p.threshold, "lfc_clip": LFC_CLIP, "n_pcs": N_PCS,
                   "n_neighbors": N_NEIGHBORS, "resolution": RESOLUTION},
        "noise_sd": round(res.noise_sd, 6), "cut": None if res.cut is None else round(res.cut, 6),
        "reference_score": round(float(res.score[ref_mask].mean()), 6) if ref_mask.any() else None,
        "seconds": round(res.seconds, 1), "cells_fingerprint": fingerprint, "method": method_text(p),
    }
    adata.obsm[OBSM_KEY] = res.X.astype(np.float32, copy=False)
    adata.obs[SCORE_COL] = res.score.astype(np.float32)
    adata.obs[CLUSTER_COL] = pd.Categorical(np.asarray(res.labels).astype(str),
                                            categories=[str(c) for c in res.labels.categories])
    adata.uns[UNS_KEY] = {
        "info": json.dumps(info),
        "window_chromosome": p.windows["chromosome"].to_numpy(dtype=object).astype(str),
        "window_start": g["start"].to_numpy()[first].astype(np.int64),
        "window_end": g["end"].to_numpy()[last].astype(np.int64),
    }
    provenance.record_step(adata, step="cnv", tool="scview.cnv", params=replay_params(p),
                           resources=resources)
    return summary(adata, p.ref_column)


def stored(adata) -> dict[str, Any] | None:
    """The saved run's info, or None (none, or it no longer fits the cells)."""
    entry = adata.uns.get(UNS_KEY) if UNS_KEY in adata.uns else None
    if not isinstance(entry, dict) or "info" not in entry:
        return None
    try:
        info = json.loads(str(entry["info"]))
    except (TypeError, ValueError):
        logger.warning("Ignoring an unreadable %s", UNS_KEY)
        return None
    X = adata.obsm[OBSM_KEY] if OBSM_KEY in adata.obsm else None
    if X is None or X.shape[0] != adata.n_obs or X.shape[1] != info.get("n_windows"):
        return None
    if SCORE_COL not in adata.obs.columns:
        return None
    return info


def _group_labels(adata, column: str) -> tuple[np.ndarray, list[str]]:
    if column not in adata.obs.columns:
        raise CnvError(f"Column '{column}' not found.")
    s = adata.obs[column]
    labels = s.astype(str).where(s.notna()).to_numpy(dtype=object)
    present = set(v for v in labels if isinstance(v, str))
    order = [str(c) for c in s.cat.categories] if hasattr(s, "cat") else sorted(present)
    return labels, [g for g in order if g in present]


def _r(v: float, d: int = 4) -> float:
    return round(float(v), d)


def by_group(adata, column: str, info: dict, max_groups: int = 100) -> list[dict[str, Any]]:
    """Per group of ``column``: its cells' score quartiles and Tukey whiskers (the
    box plot), mean, and whether it is a reference group."""
    labels, groups = _group_labels(adata, column)
    if len(groups) > max_groups:
        raise CnvError(f"'{column}' has {len(groups)} groups; the plot takes at most {max_groups}.")
    score = adata.obs[SCORE_COL].to_numpy(dtype=np.float64)
    ref_score = info.get("reference_score") or 0.0
    refs = set(info["reference_groups"]) if column == info["reference_column"] else set()
    out = []
    for g in groups:
        v = score[labels == g]
        v = v[np.isfinite(v)]
        if not len(v):
            continue
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        iqr = q3 - q1
        lo = float(v[v >= q1 - 1.5 * iqr].min())
        hi = float(v[v <= q3 + 1.5 * iqr].max())
        mean = float(v.mean())
        out.append({"group": g, "n": int(len(v)), "mean": _r(mean, 5), "q1": _r(q1, 5), "median": _r(med, 5),
                    "q3": _r(q3, 5), "lo": _r(lo, 5), "hi": _r(hi, 5), "reference": g in refs,
                    "ratio": _r(mean / ref_score, 2) if ref_score > 0 else None})
    return out


def chromosome_means(adata, mask: np.ndarray, info: dict) -> dict[str, float]:
    """Mean CNV value per chromosome over the cells in ``mask``."""
    X = adata.obsm[OBSM_KEY]
    rows = np.asarray(X[mask], dtype=np.float64)
    out = {}
    for c in info["chromosomes"]:
        out[c["chromosome"]] = float(rows[:, c["start"]:c["end"]].mean()) if len(rows) else 0.0
    return out


def calls(means: dict[str, float], info: dict) -> tuple[list[str], list[str]]:
    """Chromosomes called gained / lost in a set of cells: mean >= CHROM_CALL x the
    denoising cut (or 1 SD when denoising was off), largest first."""
    cut = info.get("cut") or info.get("noise_sd") or 0.0
    lim = CHROM_CALL * cut
    if lim <= 0:
        return [], []
    gains = sorted([c for c, v in means.items() if v >= lim], key=lambda c: -means[c])
    losses = sorted([c for c, v in means.items() if v <= -lim], key=lambda c: means[c])
    return gains, losses


def cluster_table(adata, info: dict) -> list[dict[str, Any]]:
    """Per CNV cluster: cells, mean score, ratio to the reference cells, share of
    reference cells, gained / lost chromosomes; highest score first."""
    if CLUSTER_COL not in adata.obs.columns:
        return []
    labels = adata.obs[CLUSTER_COL].astype(str).to_numpy()
    score = adata.obs[SCORE_COL].to_numpy(dtype=np.float64)
    ref_col = info["reference_column"]
    refs = set(info["reference_groups"])
    in_ref = (adata.obs[ref_col].astype(str).isin(refs).to_numpy() if ref_col in adata.obs.columns
              else np.zeros(adata.n_obs, bool))
    ref_score = info.get("reference_score") or 0.0
    out = []
    for c in pd.unique(labels):
        m = labels == c
        mean = float(score[m].mean())
        gains, losses = calls(chromosome_means(adata, m, info), info)
        out.append({"cluster": str(c), "n": int(m.sum()), "mean": _r(mean, 5),
                    "ratio": _r(mean / ref_score, 2) if ref_score > 0 else None,
                    "reference_share": _r(in_ref[m].mean(), 3), "gains": gains[:6], "losses": losses[:6]})
    out.sort(key=lambda r: -r["mean"])
    return out


def is_high(row: dict[str, Any]) -> bool:
    return row.get("ratio") is not None and row["ratio"] >= HIGH_RATIO and row.get("reference_share", 0) < 0.5


def summary(adata, groupby: str | None = None) -> dict[str, Any]:
    """The saved result for the UI: ``{"available": False}`` without one."""
    info = stored(adata)
    if info is None:
        return {"available": False}
    out: dict[str, Any] = {"available": True, **info}
    col = groupby or info["reference_column"]
    try:
        out["by_group"] = {"column": col, "groups": by_group(adata, col, info)}
    except CnvError as exc:
        out["by_group"] = None
        out["by_group_error"] = str(exc)
    out["clusters"] = cluster_table(adata, info)
    for r in out["clusters"]:
        r["high"] = is_high(r)
    return out


def _bins(info: dict, max_cols: int) -> tuple[list[tuple[int, int]], list[dict[str, Any]]]:
    """Display columns: runs of consecutive windows within one chromosome, at most
    ``max_cols`` in all; and each chromosome's first / last display column."""
    n = info["n_windows"]
    f = max(1, int(np.ceil(n / max_cols)))
    bins, spans = [], []
    for c in info["chromosomes"]:
        a0 = len(bins)
        for s in range(c["start"], c["end"], f):
            bins.append((s, min(s + f, c["end"])))
        spans.append({"chromosome": c["chromosome"], "start": a0, "end": len(bins)})
    return bins, spans


def sample_cells(labels: np.ndarray, groups: list[str], max_cells: int, seed: int = SEED) -> dict[str, np.ndarray]:
    """Cells to draw per group: all when they fit, else a share proportional to the
    group's size (at least 20, or all of a smaller group)."""
    rng = np.random.default_rng(seed)
    idx = {g: np.flatnonzero(labels == g) for g in groups}
    total = sum(len(v) for v in idx.values())
    if total <= max_cells:
        return idx
    out = {}
    for g, v in idx.items():
        k = min(len(v), max(20, int(round(max_cells * len(v) / total))))
        out[g] = np.sort(rng.choice(v, size=k, replace=False))
    return out


def heatmap(adata, groupby: str | None = None, max_cells: int = MAX_HEATMAP_CELLS,
            max_cols: int = MAX_HEATMAP_COLS) -> dict[str, Any]:
    """Sampled cells x genome for the heatmap: rows grouped by ``groupby`` (the
    reference column by default), ordered by CNV cluster then score within a
    group; columns are windows averaged in runs within each chromosome. Values are
    int8, base64, ``value = code * scale``."""
    info = stored(adata)
    if info is None:
        raise CnvError("No copy-number result for this dataset yet: run it first.")
    col = groupby or info["reference_column"]
    labels, groups = _group_labels(adata, col)
    if len(groups) > 100:
        raise CnvError(f"'{col}' has {len(groups)} groups; group the heatmap by a column with at most 100.")
    picks = sample_cells(labels, groups, max_cells)
    score = adata.obs[SCORE_COL].to_numpy(dtype=np.float64)
    clus = (adata.obs[CLUSTER_COL].astype(str).to_numpy() if CLUSTER_COL in adata.obs.columns
            else np.zeros(adata.n_obs, dtype=str))
    rows, row_groups = [], []
    for g in groups:
        v = picks.get(g, np.zeros(0, int))
        if not len(v):
            continue
        order = np.lexsort((-score[v], clus[v]))
        rows.append(v[order])
        row_groups.append({"group": g, "n_sampled": int(len(v)), "n_total": int((labels == g).sum()),
                           "reference": col == info["reference_column"] and g in info["reference_groups"]})
    cells = np.concatenate(rows) if rows else np.zeros(0, int)
    X = np.asarray(adata.obsm[OBSM_KEY][cells], dtype=np.float32)
    bins, spans = _bins(info, max_cols)
    D = np.column_stack([X[:, a:b].mean(axis=1) for a, b in bins]) if len(cells) else np.zeros((0, len(bins)))
    nz = np.abs(D[D != 0])
    vmax = float(np.percentile(nz, 99)) if nz.size else float(info.get("cut") or 0.1)
    vmax = max(vmax, 1e-6)
    codes = np.clip(np.round(D / vmax * 127), -127, 127).astype(np.int8)
    return {
        "column": col, "groups": row_groups, "n_rows": int(len(cells)), "n_cols": len(bins),
        "chromosomes": spans, "vmax": _r(vmax, 6), "scale": vmax / 127,
        "values": base64.b64encode(codes.tobytes()).decode("ascii"),
        "clusters": clus[cells].tolist(), "n_cells_total": int(adata.n_obs),
    }


def runnable(adata) -> dict[str, Any]:
    """Whether a run can start: genome, expression source, how many genes have a
    position (reads gene names and a sample of the matrix only)."""
    try:
        genome, why = pick_genome(adata)
        src = expression_source(adata)
        genes, match = place_genes(src.names, ensembl_ids(adata, src.names), genome, DEFAULT_EXCLUDE)
        ok = len(genes) >= MIN_GENES
        return {"ok": ok, "reason": None if ok else
                f"Only {len(genes)} of {match['n_genes']:,} genes have a position on the "
                f"{GENOMES[genome]['assembly']} genome; at least {MIN_GENES} are needed.",
                "genome": genome, "assembly": GENOMES[genome]["assembly"], "genome_why": why,
                "source": src.text, "n_placed": int(len(genes)), "match": match}
    except CnvError as exc:
        return {"ok": False, "reason": str(exc), "genome": None, "assembly": None, "genome_why": None,
                "source": None, "n_placed": None, "match": None}


def estimate_seconds(n_cells: int, n_genes: int, n_windows: int) -> float:
    """Seconds for a run, from maynard2020_3k (3,000 cells, 23,182 genes kept, 2 cores:
    ~17 s, of which ~6 s PCA + graph + Leiden, growing about as n^1.15)."""
    win = n_cells * n_genes * 1.5e-7 + n_cells * n_windows * DEFAULT_WINDOW * 2e-9
    clus = 2.0 + 6e-4 * n_cells ** 1.15
    return round(win + clus, 1)


# ---------------------------------------------------------------------------
# Co-pilot
# ---------------------------------------------------------------------------


def grounding_lines(adata) -> tuple[list[str], str]:
    """The saved result as a fact: reference, high-score groups of the reference
    column, high CNV clusters with their gained / lost chromosomes."""
    info = stored(adata)
    if info is None:
        return [], ""
    refs = "/".join(info["reference_groups"])
    col = info["reference_column"]
    try:
        groups = by_group(adata, col, info)
    except CnvError:
        groups = []
    high_groups = [g for g in groups if not g["reference"] and g["ratio"] is not None
                   and g["ratio"] >= HIGH_RATIO]
    high_groups.sort(key=lambda g: -g["mean"])
    clusters = [c for c in cluster_table(adata, info) if is_high(c)]
    detail = (f"ref: {refs} in {col}; {len(high_groups)} group(s) and {len(clusters)} CNV cluster(s) with "
              f">= {HIGH_RATIO:g}x the reference score")
    parts = [f"- CNV (inferCNV-style from expression, ref: {refs} in '{col}', {info['n_genes']:,} genes, "
             f"window {info['params']['window']}): reference cells' mean score "
             f"{(info.get('reference_score') or 0):.3f}."]
    if high_groups:
        parts.append(" High-score groups: " + "; ".join(
            f"{g['group']} (mean {g['mean']:.3f}, {g['ratio']:.1f}x ref, n = {g['n']:,})"
            for g in high_groups[:5]) + ".")
    else:
        parts.append(f" No {col} group outside the reference reaches {HIGH_RATIO:g}x the reference score.")
    if clusters:
        def one(c: dict) -> str:
            pat = ", ".join([f"{x} gain" for x in c["gains"][:3]] + [f"{x} loss" for x in c["losses"][:3]])
            return f"cluster {c['cluster']} (n = {c['n']:,}, {c['ratio']:.1f}x ref{'; ' + pat if pat else ''})"
        parts.append(" High CNV clusters (cnv_leiden): " + "; ".join(one(c) for c in clusters[:5]) + ".")
    parts.append(" Inferred from expression, not DNA; needs a true normal reference; low depth is noisy "
                 f"[{TAG}]")
    return ["\n## Copy number (inferred)", "".join(parts)], detail
