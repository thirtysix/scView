"""DSB: CITE-seq protein normalised against the background measured in empty droplets.

Antibody counts carry a background from unbound antibody in the droplet, at a level that
differs by antibody, and a per-cell technical part (how well the droplet captured
anything). DSB (Mulè, Martins & Tsang 2022, Nat Commun 13:2099,
doi:10.1038/s41467-022-29356-8; R package ``dsb``, MIT licence) removes both. It is
implemented here from the paper and the package source (dsb 2.0.1,
``DSBNormalizeProtein`` and ``ModelNegativeADTnorm``):

- **Step I, ambient correction**: each protein's counts in empty droplets, ``log(x + 10)``
  (pseudocount 10), give its background mean and SD; a cell's value is
  ``(log(x + 10) - mean) / SD``: how many background SDs above the empty droplets it is.
- **Step II, technical component** (``denoise.counts = TRUE``): for each cell a
  2-component Gaussian mixture over its proteins (mclust ``Mclust(G = 2)``: models "E"
  and "V", the better BIC; initial split at the median, mclust's ``qclass``) gives the
  mean of its lower (non-staining) component. With isotype controls
  (``use.isotype.control = TRUE``) the technical component is the first principal
  component of that mean and the isotypes' step-I values (scaled); without isotypes it is
  the mean alone (dsb's ``use.isotype.control = FALSE``, which dsb advises against when
  isotypes exist: a note says so). It is regressed out of every protein
  (limma ``removeBatchEffect(covariates = ...)``: the centred covariate's slope removed).

**Empty droplets** come from the raw (unfiltered) 10x matrix attached as a side file
(``core/seq_metrics.py``). The raw file is not kept, so when it is read, the droplets
that are not the dataset's cells are binned by log10 RNA and log10 protein UMIs (bins of
0.1, RNA below 10^4, protein below 10^5) and each bin's per-protein sums of
``log(x + 10)`` (and of squares) are kept in ``uns[UNS_BACKGROUND]``. Any window on that
grid then gives the background mean and SD exactly. The default window is the dsb
vignette's for this kind of data: log10 protein UMIs from 1.5 up to 3 and log10 RNA UMIs
from 0.3 (2 UMIs: dsb's ``rna.size > 0``) up to 2.5; both are exposed. The window's
bounds include the lower edge, exclude the upper.

**Without empty droplets** (no raw matrix) the dsb package's own mode for that case,
``ModelNegativeADTnorm``, is used rather than refusing: each protein's ``log(x + 1)``
over the cells gets a 2-component mixture and its lower mean is subtracted, then step II
as above. Many CITE-seq uploads come without their raw matrix, and this is the mode the
dsb authors provide for them; it is labelled everywhere, with dsb's own caution that a
protein every cell expresses can end up centred near 0.

Validation against R dsb 2.0.1 (limma 3.68.5, mclust 6.1.3; rocker/r2u 24.04,
2026-10-26) on 10x pbmc_10k_protein_v3 (7,865 cells, 17 antibodies, 3 isotypes; raw and
filtered feature-barcode h5 from
cf.10xgenomics.com/samples/cell-exp/3.0.0/pbmc_10k_protein_v3/), same cells and the
vignette's window on both sides: the binned background gave the same 69,162 droplets and
background mean / SD to 1e-12; ``DSBNormalizeProtein`` with the 3 isotypes and without
isotypes was matched to 1e-12 in every value (per-cell mixture means to 7e-13), Pearson
1.0 per protein. Against ``ModelNegativeADTnorm`` Pearson per protein >= 0.9999991
(largest difference 0.06): with more than 2,000 cells mclust starts each protein's fit
from a random subset of 2,000 (``mclust.options("subset")``), scView from all cells, so
the two differ where EM settles.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DSB_KEY = "protein_dsb"  # obsm: cells x proteins, DSB values
UNS_BACKGROUND = "scview_dsb_background"  # per sample: the binned empty-droplet sums
UNS_RECORD = "scview_dsb"  # the last run's record (JSON)
TAG = "result:dsb"
STEP = "protein_dsb"
DSB_URL = "https://doi.org/10.1038/s41467-022-29356-8"

PSEUDOCOUNT = 10.0  # DSBNormalizeProtein's default (ModelNegativeADTnorm uses 1)
NEG_PSEUDOCOUNT = 1.0
BIN = 0.1  # log10 UMIs per bin
RNA_BINS = 40  # RNA below 10^4 UMIs
PROT_BINS = 50  # protein below 10^5 UMIs
RNA_WINDOW = (0.3, 2.5)  # dsb vignette: rna.size > 0 & rna.size < 2.5 (log10)
PROT_WINDOW = (1.5, 3.0)  # dsb vignette: prot.size > 1.5 & prot.size < 3
MIN_DROPLETS = 50  # fewer in the window: no background estimate
FEW_DROPLETS = 1000  # a note below this
LOW_SD = 0.05  # dsb prints these proteins: little spread in the background
NNZ_CHUNK = 8_000_000
BC_CHUNK = 1_000_000
MEX_ROWS = 1_000_000

EM_TOL = 1e-5  # mclust emControl()$tol[1]
EM_MAX_ITER = 5000
EM_EPS = 2.220446049250313e-16  # .Machine$double.eps: smaller variance is a failed fit
BLOCK_VALUES = 2_000_000  # values per block in the mixture fits

MODES = ("auto", "empty_droplets", "model_negative")

HELP = {
    "dsb": (
        "DSB: each protein relative to its background in empty droplets (how many background "
        "SDs above them a cell is), with the per-cell technical noise taken out using the isotype "
        "controls. Around 0 is background; clearly positive is staining. Mulè et al. 2022."
    ),
    "clr": (
        "CLR: each protein's count relative to the cell's other proteins (centred log-ratio, "
        "Seurat margin 2). Needs no empty droplets; background is not removed."
    ),
    "model_negative": (
        "No raw matrix with empty droplets is attached, so the background is estimated from the "
        "cells themselves (dsb's ModelNegativeADTnorm): the lower of two groups of each protein's "
        "values. A protein that every cell expresses can then sit near 0."
    ),
    "window": (
        "Empty droplets used as background: droplets that are not cells, with RNA and protein UMI "
        "totals inside these log10 ranges (dsb's vignette: protein 1.5 to 3, RNA up to 2.5)."
    ),
}


class DsbError(ValueError):
    """A request that can't be met, as a sentence for the user."""

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


# ---------------------------------------------------------------------------
# Reading the empty droplets (also copied into replay scripts: REPLAY_CODE)
# ---------------------------------------------------------------------------


def size_bin(total, width=0.1, n_bins=40):
    """Bin of each UMI total by log10 (bins of ``width`` from 0); -1 for 0 UMIs or past
    the last bin. The tiny offset keeps 10, 100 and 1000 in the bin they start."""
    t = np.asarray(total, dtype=np.float64)
    out = np.full(t.shape, -1, dtype=np.int64)
    pos = t >= 1
    b = np.floor(np.log10(t[pos]) / width + 1e-9).astype(np.int64)
    b[b >= n_bins] = -1
    out[pos] = b
    return out


def barcode_core(names):
    """The bare barcode sequence ("AAAC...-1" and "s1:AAAC..." give "AAAC...")."""
    s = pd.Series(np.asarray(names, dtype=str), dtype=object)
    return s.str.extract(r"([ACGTN]{8,})", expand=False).fillna("").to_numpy(dtype=str)


def cell_matcher(cells):
    """``is_cell(barcodes)``: full barcode or bare sequence among the dataset's cells."""
    cells = np.asarray(list(cells), dtype=str)
    full = set(cells.tolist())
    core = set(c for c in barcode_core(cells).tolist() if c)

    def is_cell(barcodes):
        bcs = np.asarray(barcodes, dtype=str)
        if bcs.size == 0:
            return np.zeros(0, dtype=bool)
        in_full = np.fromiter((b in full for b in bcs.tolist()), dtype=bool, count=bcs.size)
        in_core = np.fromiter((c in core for c in barcode_core(bcs).tolist()), dtype=bool,
                              count=bcs.size)
        return in_full | in_core

    return is_cell


def _accumulate(acc, rna, prot, bc_of_entry, slot, vals, idx, is_cell_fn, barcodes_of):
    """Add one block of barcodes to the grid sums: ``rna``/``prot`` the block's UMI
    totals, entries (``vals`` at features ``idx``) of barcode ``bc_of_entry``."""
    n_prot = acc["sum"].shape[1]
    rb = size_bin(rna, BIN_WIDTH, acc["rna_bins"])
    pb = size_bin(prot, BIN_WIDTH, acc["prot_bins"])
    cand = np.flatnonzero((rb >= 0) & (pb >= 0))
    if cand.size == 0:
        return
    cells = is_cell_fn(barcodes_of(cand))
    acc["n_cells_excluded"] += int(cells.sum())
    cand = cand[~cells]
    if cand.size == 0:
        return
    keep = np.zeros(rna.size, dtype=bool)
    keep[cand] = True
    flat_bin = rb * acc["prot_bins"] + pb
    n_grid = acc["rna_bins"] * acc["prot_bins"]
    acc["n"] += np.bincount(flat_bin[cand], minlength=n_grid)
    sel = keep[bc_of_entry] & (slot[idx] >= 0)
    if sel.any():
        v = np.log1p(vals[sel] / PSEUDO)
        where = flat_bin[bc_of_entry[sel]] * n_prot + slot[idx[sel]]
        acc["sum"] += np.bincount(where, weights=v, minlength=n_grid * n_prot).reshape(n_grid, n_prot)
        acc["sumsq"] += np.bincount(where, weights=v * v, minlength=n_grid * n_prot).reshape(n_grid, n_prot)


def _new_grid(n_prot):
    n_grid = RNA_GRID * PROT_GRID
    return {"rna_bins": RNA_GRID, "prot_bins": PROT_GRID, "n": np.zeros(n_grid, dtype=np.int64),
            "sum": np.zeros((n_grid, n_prot)), "sumsq": np.zeros((n_grid, n_prot)),
            "n_cells_excluded": 0}


def _grid_record(acc, features, n_barcodes):
    """The stored background: only the grid's non-empty bins."""
    nz = np.flatnonzero(acc["n"] > 0)
    return {"features": np.asarray(features, dtype=str), "bins": nz.astype(np.int32),
            "n": acc["n"][nz].astype(np.int64), "sum": acc["sum"][nz], "sumsq": acc["sumsq"][nz],
            "grid": np.array([acc["rna_bins"], acc["prot_bins"]], dtype=np.int64),
            "bin_width": float(BIN_WIDTH), "pseudocount": float(PSEUDO),
            "n_cells_excluded": int(acc["n_cells_excluded"]), "n_barcodes": int(n_barcodes)}


def background_from_h5(path, cells, nnz_chunk=8_000_000, bc_chunk=1_000_000):
    """The empty-droplet background of a raw Cell Ranger HDF5 matrix (v3+ layout), read in
    blocks of matrix entries: per (log10 RNA, log10 protein) bin, the droplets that are not
    ``cells`` and each antibody's sums of log(x + pseudocount) and of its square. None
    when the matrix has no Antibody Capture features."""
    is_cell_fn = cell_matcher(cells)
    with h5py.File(path, "r") as f:
        if "matrix" not in f:
            raise ValueError(f"{path} is not a Cell Ranger v3+ HDF5 matrix.")
        m = f["matrix"]
        n_bc = int(m["shape"][1])
        feats = m["features"]
        names = feats["name"][:].astype(str)
        if "feature_type" not in feats:
            return None
        types = feats["feature_type"][:].astype(str)
        ab = np.flatnonzero(types == "Antibody Capture")
        if ab.size == 0:
            return None
        is_gex = (types == "Gene Expression").astype(np.float64)
        slot = np.full(names.size, -1, dtype=np.int64)
        slot[ab] = np.arange(ab.size)
        is_ab = (slot >= 0).astype(np.float64)
        indptr = np.asarray(m["indptr"][:], dtype=np.int64)
        data, indices, bcs = m["data"], m["indices"], m["barcodes"]
        acc = _new_grid(ab.size)
        b0 = 0
        while b0 < n_bc:
            s = int(indptr[b0])
            b1 = int(np.searchsorted(indptr, s + nnz_chunk, side="right")) - 1
            b1 = min(max(b1, b0 + 1), n_bc, b0 + bc_chunk)
            e = int(indptr[b1])
            if e > s:
                vals = data[s:e].astype(np.float64)
                idx = np.asarray(indices[s:e], dtype=np.int64)
                seg = indptr[b0:b1 + 1] - s
                bc_of_entry = np.repeat(np.arange(b1 - b0), np.diff(seg))
                rna = np.bincount(bc_of_entry, weights=vals * is_gex[idx], minlength=b1 - b0)
                prot = np.bincount(bc_of_entry, weights=vals * is_ab[idx], minlength=b1 - b0)
                start = b0
                _accumulate(acc, rna, prot, bc_of_entry, slot, vals, idx, is_cell_fn,
                            lambda rows: bcs[start + rows[0]:start + rows[-1] + 1][rows - rows[0]].astype(str))
            b0 = b1
    return _grid_record(acc, names[ab], n_bc)


def background_from_mex(matrix, barcodes, features, cells, rows=1_000_000):
    """``background_from_h5`` for a raw 10x MEX triplet (matrix.mtx, barcodes.tsv,
    features.tsv; gzipped or not), in two streamed passes over the matrix."""
    import gzip

    def text(p):
        with open(p, "rb") as fh:
            gz = fh.read(2) == b"\x1f\x8b"
        return (gzip.open(p, "rt", encoding="utf-8", errors="replace") if gz
                else open(p, "rt", encoding="utf-8", errors="replace"))

    with text(features) as fh:
        parts = [ln.rstrip("\n").split("\t") for ln in fh if ln.strip()]
    names = np.array([p[1] if len(p) > 1 else p[0] for p in parts], dtype=str)
    types = np.array([p[2] if len(p) > 2 else "Gene Expression" for p in parts], dtype=str)
    ab = np.flatnonzero(types == "Antibody Capture")
    if ab.size == 0:
        return None
    is_gex = (types == "Gene Expression").astype(np.float64)
    slot = np.full(names.size, -1, dtype=np.int64)
    slot[ab] = np.arange(ab.size)
    is_ab = (slot >= 0).astype(np.float64)
    n_header = 0
    dims = None
    with text(matrix) as fh:
        for line in fh:
            n_header += 1
            s = line.strip()
            if s and not s.startswith("%"):
                dims = [int(x) for x in s.split()[:3]]
                break
    if not dims:
        raise ValueError(f"{matrix} has no Matrix Market dimensions line.")
    if dims[0] == names.size:
        feat_col, bc_col, n_bc = 0, 1, dims[1]
    elif dims[1] == names.size:
        feat_col, bc_col, n_bc = 1, 0, dims[0]
    else:
        raise ValueError(f"The raw matrix is {dims[0]} x {dims[1]} but its features file lists "
                         f"{names.size} features.")

    def chunks():
        with text(matrix) as fh:
            yield from pd.read_csv(fh, sep=" ", header=None, skiprows=n_header, usecols=[0, 1, 2],
                                   dtype={0: np.int64, 1: np.int64, 2: np.float64}, chunksize=rows)

    rna = np.zeros(n_bc)
    prot = np.zeros(n_bc)
    for c in chunks():
        f_i = c[feat_col].to_numpy() - 1
        b_i = c[bc_col].to_numpy() - 1
        w = c[2].to_numpy()
        rna += np.bincount(b_i, weights=w * is_gex[f_i], minlength=n_bc)
        prot += np.bincount(b_i, weights=w * is_ab[f_i], minlength=n_bc)
    with text(barcodes) as fh:
        bc_names = np.array([ln.rstrip("\n").split("\t")[0] for ln in fh if ln.strip()], dtype=str)
    if bc_names.size != n_bc:
        raise ValueError(f"The raw barcodes file lists {bc_names.size:,} barcodes; the matrix has "
                         f"{n_bc:,}.")
    acc = _new_grid(ab.size)
    # Which barcodes are background candidates is known from the totals: decide once.
    is_cell_fn = cell_matcher(cells)
    rb = size_bin(rna, BIN_WIDTH, acc["rna_bins"])
    pb = size_bin(prot, BIN_WIDTH, acc["prot_bins"])
    cand = np.flatnonzero((rb >= 0) & (pb >= 0))
    cells_mask = np.zeros(n_bc, dtype=bool)
    cells_mask[cand] = is_cell_fn(bc_names[cand])
    acc["n_cells_excluded"] = int(cells_mask.sum())
    keep = np.zeros(n_bc, dtype=bool)
    keep[cand] = ~cells_mask[cand]
    flat_bin = rb * acc["prot_bins"] + pb
    n_grid = acc["rna_bins"] * acc["prot_bins"]
    n_prot = ab.size
    acc["n"] += np.bincount(flat_bin[keep], minlength=n_grid)
    for c in chunks():
        f_i = c[feat_col].to_numpy() - 1
        b_i = c[bc_col].to_numpy() - 1
        sel = keep[b_i] & (slot[f_i] >= 0)
        if not sel.any():
            continue
        v = np.log1p(c[2].to_numpy()[sel] / PSEUDO)
        where = flat_bin[b_i[sel]] * n_prot + slot[f_i[sel]]
        acc["sum"] += np.bincount(where, weights=v, minlength=n_grid * n_prot).reshape(n_grid, n_prot)
        acc["sumsq"] += np.bincount(where, weights=v * v, minlength=n_grid * n_prot).reshape(n_grid, n_prot)
    return _grid_record(acc, names[ab], n_bc)


def window_stats(bg, rna_window, prot_window):
    """(droplets, background mean, background SD) per antibody of a stored background,
    over the droplets with lo <= log10 UMIs < hi for RNA and protein. Mean and SD are of
    log(x + pseudocount), the SD with n - 1 (R's ``sd``)."""
    width = float(bg["bin_width"])
    n_prot_bins = int(np.asarray(bg["grid"])[1])
    bins = np.asarray(bg["bins"], dtype=np.int64)
    rb, pb = bins // n_prot_bins, bins % n_prot_bins

    def inside(b, lo, hi):
        return (b * width >= lo - 1e-9) & ((b + 1) * width <= hi + 1e-9)

    sel = inside(rb, *rna_window) & inside(pb, *prot_window)
    n = int(np.asarray(bg["n"])[sel].sum())
    n_ab = np.asarray(bg["sum"]).shape[1]
    if n < 2:
        return n, np.full(n_ab, np.nan), np.full(n_ab, np.nan)
    s = np.asarray(bg["sum"], dtype=np.float64)[sel].sum(axis=0)
    ss = np.asarray(bg["sumsq"], dtype=np.float64)[sel].sum(axis=0)
    shift = math.log(float(bg["pseudocount"]))  # the sums are of log(x + pc) - log(pc)
    mean = shift + s / n
    var = np.maximum((ss - s * s / n) / (n - 1), 0.0)
    return n, mean, np.sqrt(var)


# ---------------------------------------------------------------------------
# The DSB maths (also copied into replay scripts)
# ---------------------------------------------------------------------------


def qclass2(x):
    """mclust's ``qclass(x, 2)``: the two-class split at the quantiles its univariate
    mixture starts from (1 = lower, 2 = upper); None for a constant row."""
    x = np.asarray(x, dtype=np.float64)
    if x.size < 2 or x.max() == x.min():
        return None
    eps = np.std(x, ddof=1) * math.sqrt(EM_EPS)
    q = np.array([np.nan])
    n = 2
    while q.size < 3:
        n += 1
        q = np.unique(np.quantile(x, np.linspace(0.0, 1.0, n)))
    if q.size > 3:
        drop = np.argsort(np.diff(q), kind="stable")[:q.size - 3]
        q = np.delete(q, drop)
    q[0] = x.min() - eps
    q[-1] = x.max() + eps
    cl = np.zeros(x.size, dtype=np.int64)
    for i in range(2):
        cl[(x >= q[i]) & (x < q[i + 1])] = i + 1
    return cl


def _em2(X, z1, equal):
    """mclust ``me`` for 2 univariate components on each row of ``X`` from the start
    responsibilities ``z1`` of component 1: model "E" (``equal`` variance) or "V".
    Returns (component means (rows x 2), log-likelihood, fit ok)."""
    m, n = X.shape
    mu = np.full((m, 2), np.nan)
    ll = np.full(m, np.nan)
    ok = np.zeros(m, dtype=bool)
    active = np.arange(m)
    Xa, za = X, z1.copy()
    hold = np.full(m, np.inf)
    log2pi = math.log(2 * math.pi)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for _ in range(EM_MAX_ITER):
            z2 = 1.0 - za
            n1 = za.sum(axis=1)
            n2 = n - n1
            m1 = (za * Xa).sum(axis=1) / n1
            m2 = (z2 * Xa).sum(axis=1) / n2
            d1 = Xa - m1[:, None]
            d2 = Xa - m2[:, None]
            s1 = (za * d1 * d1).sum(axis=1)
            s2 = (z2 * d2 * d2).sum(axis=1)
            if equal:
                v1 = v2 = (s1 + s2) / n
            else:
                v1, v2 = s1 / n1, s2 / n2
            bad = ~(v1 > EM_EPS) | ~(v2 > EM_EPS) | ~np.isfinite(m1) | ~np.isfinite(m2)
            l1 = np.log(n1 / n)[:, None] - 0.5 * (log2pi + np.log(v1))[:, None] - d1 * d1 / (2 * v1)[:, None]
            l2 = np.log(n2 / n)[:, None] - 0.5 * (log2pi + np.log(v2))[:, None] - d2 * d2 / (2 * v2)[:, None]
            lse = np.logaddexp(l1, l2)
            hood = lse.sum(axis=1)
            za = np.exp(l1 - lse)
            err = np.abs(hood - hold) / (1.0 + np.abs(hood))
            hold = hood
            done = bad | ~np.isfinite(hood) | (err <= EM_TOL)
            # Mclust then takes one more M-step from the last E-step when the mixing
            # proportions still move (summary.mclustBIC: sum of squares > sqrt(eps)).
            q1 = za.sum(axis=1)
            again = 2 * ((n1 - q1) / n) ** 2 > math.sqrt(EM_EPS)
            f1 = np.where(again, (za * Xa).sum(axis=1) / q1, m1)
            f2 = np.where(again, ((1.0 - za) * Xa).sum(axis=1) / (n - q1), m2)
            rows = active[done]
            mu[rows, 0], mu[rows, 1] = f1[done], f2[done]
            ll[rows] = hood[done]
            ok[rows] = ~(bad[done] | ~np.isfinite(hood[done]))
            if done.all():
                return mu, ll, ok
            keep = ~done
            active, Xa, za, hold = active[keep], Xa[keep], za[keep], hold[keep]
        mu[active, 0], mu[active, 1] = m1[keep], m2[keep]  # not converged: the last estimate
        ll[active] = hood[keep]
        ok[active] = True
    return mu, ll, ok


def lower_means(X):
    """Each row's mean of mixture component 1 (the lower start class), as
    ``Mclust(x, G = 2)$parameters$mean[1]``: "E" and "V" fitted, the larger BIC kept.
    Returns (means, fit ok); a row whose fits both fail gets its lower-half mean."""
    X = np.asarray(X, dtype=np.float64)
    m, n = X.shape
    out = np.full(m, np.nan)
    fitted = np.zeros(m, dtype=bool)
    if m == 0:
        return out, fitted
    step = max(1, BLOCK_VALUES // max(n, 1))
    for r0 in range(0, m, step):
        B = X[r0:r0 + step]
        med = np.median(B, axis=1)
        lo, hi = B.min(axis=1), B.max(axis=1)
        z1 = (B < med[:, None]).astype(np.float64)
        odd = np.flatnonzero((med == lo) | (med == hi))
        usable = np.ones(B.shape[0], dtype=bool)
        for i in odd:
            cl = qclass2(B[i])
            if cl is None:
                usable[i] = False
            else:
                z1[i] = (cl == 1).astype(np.float64)
        muE, llE, okE = _em2(B, z1, True)
        muV, llV, okV = _em2(B, z1, False)
        bicE = 2 * llE - 4 * math.log(n)
        bicV = 2 * llV - 5 * math.log(n)
        useV = okV & (~okE | (bicV > bicE))
        res = np.where(useV, muV[:, 0], muE[:, 0])
        good = (okE | okV) & usable
        with np.errstate(invalid="ignore"):
            fallback = np.where(z1.sum(axis=1) > 0, (z1 * B).sum(axis=1) / np.maximum(z1.sum(axis=1), 1), B.mean(axis=1))
        out[r0:r0 + step] = np.where(good, res, fallback)
        fitted[r0:r0 + step] = good
    return out, fitted


def step1_empty(counts, mean, sd):
    """Step I: (log(x + 10) - background mean) / background SD; a protein with no spread
    in the background (SD 0) is mean-subtracted only."""
    x = np.log(np.asarray(counts, dtype=np.float64) + PSEUDO)
    sd = np.where(np.asarray(sd) > 0, sd, 1.0)
    return (x - np.asarray(mean)[None, :]) / sd[None, :]


def step1_model_negative(counts):
    """ModelNegativeADTnorm's step I: log(x + 1) minus each protein's lower mixture mean
    over the cells. Returns (values, background per protein, fitted per protein)."""
    x = np.log(np.asarray(counts, dtype=np.float64) + NEG_PSEUDO)
    mu1, fitted = lower_means(x.T)
    mu1 = np.where(fitted, mu1, 0.0)  # dsb: "log transformed without background correction"
    return x - mu1[None, :], mu1, fitted


def technical_component(norm, isotype_idx, background_mean):
    """dsb's per-cell technical component: with isotypes, the first principal component
    of the isotypes' values and the cell's background mean (each scaled); without, the
    background mean."""
    bg = np.asarray(background_mean, dtype=np.float64)
    if len(isotype_idx) == 0:
        return bg
    M = np.column_stack([np.asarray(norm)[:, isotype_idx], bg])
    M = M - M.mean(axis=0)
    sd = M.std(axis=0, ddof=1)
    M = M[:, sd > 0] / sd[sd > 0]
    if M.shape[1] == 0:
        return bg
    w, v = np.linalg.eigh(M.T @ M)
    return M @ v[:, np.argmax(w)]


def regress_out(norm, covariate):
    """limma ``removeBatchEffect(x, covariates = v)`` with cells as columns: the centred
    covariate's least-squares slope (with an intercept) removed from each protein."""
    vc = np.asarray(covariate, dtype=np.float64)
    vc = vc - vc.mean()
    denom = float(vc @ vc)
    if denom <= 0:
        return np.asarray(norm, dtype=np.float64)
    beta = (vc @ np.asarray(norm, dtype=np.float64)) / denom
    return norm - np.outer(vc, beta)


def step2(norm, isotype_idx):
    """Step II on step-I values (cells x proteins): (denoised values, technical component,
    per-cell background mean, cells whose mixture fit failed)."""
    bg_mean, fitted = lower_means(norm)
    tech = technical_component(norm, isotype_idx, bg_mean)
    return regress_out(norm, tech), tech, bg_mean, int((~fitted).sum())


# The constants the functions above read, under the names they use there.
BIN_WIDTH = BIN
RNA_GRID = RNA_BINS
PROT_GRID = PROT_BINS
PSEUDO = PSEUDOCOUNT
NEG_PSEUDO = NEG_PSEUDOCOUNT

# Copied into replay scripts (core/replay_script.py): numpy, pandas and h5py only.
REPLAY_CONSTANTS = ("EM_TOL", "EM_MAX_ITER", "EM_EPS", "BLOCK_VALUES", "BIN_WIDTH", "RNA_GRID",
                    "PROT_GRID", "PSEUDO", "NEG_PSEUDO")
REPLAY_CODE = (size_bin, barcode_core, cell_matcher, _accumulate, _new_grid, _grid_record,
               background_from_h5, background_from_mex, window_stats, qclass2, _em2, lower_means,
               step1_empty, step1_model_negative, technical_component, regress_out, step2)


# ---------------------------------------------------------------------------
# Backgrounds kept in the dataset (read when a raw matrix is attached)
# ---------------------------------------------------------------------------


def background_for_files(h5: Path | None, mex: tuple[Path, Path, Path] | None, cells) -> dict | None:
    """The binned empty-droplet background of a raw matrix; None without one, or when it
    has no antibody features."""
    if h5 is not None:
        return background_from_h5(h5, cells, NNZ_CHUNK, BC_CHUNK)
    if mex is not None:
        return background_from_mex(*mex, cells=cells, rows=MEX_ROWS)
    return None


def _display_names(features) -> list[str]:
    from scview.core import protein

    return protein._unique([protein.display_name(str(f)) for f in features])


def read_backgrounds(adata) -> dict[str, dict]:
    raw = getattr(adata, "uns", {}).get(UNS_BACKGROUND)
    if not isinstance(raw, dict):
        return {}
    return {str(k): v for k, v in raw.items() if isinstance(v, dict) and "bins" in v}


def store_backgrounds(adata, backgrounds: dict[str, dict]) -> None:
    """Add (or replace, by sample) backgrounds; a new dict so a save sees the entry changed."""
    current = dict(read_backgrounds(adata))
    current.update(backgrounds)
    adata.uns[UNS_BACKGROUND] = {k: dict(v) for k, v in current.items()}


def remove_background(adata, sample: str) -> None:
    current = dict(read_backgrounds(adata))
    if current.pop(sample, None) is None:
        return
    if current:
        adata.uns[UNS_BACKGROUND] = current
    else:
        del adata.uns[UNS_BACKGROUND]


def background_summary(bg: dict) -> dict:
    """What the sequencing record says about a background (the arrays live in uns)."""
    n, _, _ = window_stats(bg, RNA_WINDOW, PROT_WINDOW)
    return {"n_antibodies": int(len(bg["features"])), "n_droplets_default_window": n,
            "n_cells_excluded": int(bg["n_cells_excluded"])}


def _sample_rows(adata, sample: str) -> tuple[np.ndarray, str | None]:
    from scview.core import ambient, seq_metrics

    rec = (seq_metrics.read(adata) or {}).get("samples", {}).get(sample)
    return ambient._record_rows(adata, rec)


def _density(bg: dict) -> dict:
    """Droplets per (RNA, protein) bin, for the window plot."""
    n_prot_bins = int(np.asarray(bg["grid"])[1])
    bins = np.asarray(bg["bins"], dtype=np.int64)
    return {"rna_bin": (bins // n_prot_bins).tolist(), "prot_bin": (bins % n_prot_bins).tolist(),
            "n": np.asarray(bg["n"]).astype(int).tolist(), "bin_width": float(bg["bin_width"])}


# ---------------------------------------------------------------------------
# Running it on a dataset
# ---------------------------------------------------------------------------


def _counts(adata) -> pd.DataFrame:
    from scview.core import protein

    m = adata.obsm.get(protein.COUNTS_KEY)
    if not isinstance(m, pd.DataFrame) or m.shape[1] == 0:
        raise DsbError("This dataset has no CITE-seq protein counts.", 404)
    return m


def _windows(rna_window, prot_window) -> tuple[tuple[float, float], tuple[float, float]]:
    out = []
    for name, w, top in (("RNA", rna_window, RNA_BINS * BIN), ("protein", prot_window, PROT_BINS * BIN)):
        lo, hi = (round(float(w[0]), 1), round(float(w[1]), 1))
        if not (0 <= lo < hi <= top + 1e-9):
            raise DsbError(f"The {name} window must run from at least 0 up to at most {top:g} "
                           "(log10 UMIs), its lower end below its upper.")
        out.append((lo, hi))
    return out[0], out[1]


def default_isotypes(names: list[str]) -> list[str]:
    from scview.core import protein

    return [n for n in names if protein.is_control(n)]


def _resolve_mode(mode: str, backgrounds: dict) -> str:
    if mode not in MODES:
        raise DsbError(f"Unknown mode '{mode}': use one of {', '.join(MODES)}.")
    if mode == "auto":
        return "empty_droplets" if backgrounds else "model_negative"
    if mode == "empty_droplets" and not backgrounds:
        raise DsbError("No empty droplets: attach the raw (unfiltered) raw_feature_bc_matrix with its "
                       "antibody features under Sequencing QC, or use the no-empty-droplet mode.")
    return mode


def compute(adata, *, mode: str = "auto", rna_window=RNA_WINDOW, prot_window=PROT_WINDOW,
            isotypes: list[str] | None = None, denoise: bool = True) -> tuple[np.ndarray, dict]:
    """DSB values (cells x proteins, the protein counts' order) and the run's record.
    Changes nothing."""
    counts_df = _counts(adata)
    names = [str(c) for c in counts_df.columns]
    counts = counts_df.to_numpy(dtype=np.float64)
    backgrounds = read_backgrounds(adata)
    mode = _resolve_mode(mode, backgrounds)
    rna_window, prot_window = _windows(rna_window, prot_window)
    iso = default_isotypes(names) if isotypes is None else [str(i) for i in isotypes]
    missing = [i for i in iso if i not in names]
    if missing:
        raise DsbError(f"Not in the protein panel: {', '.join(missing)}.")
    iso_idx = [names.index(i) for i in iso]
    notes: list[str] = []
    proteins: list[dict[str, Any]] = [{"protein": n, "isotype": n in iso} for n in names]
    samples: list[dict[str, Any]] = []
    if mode == "empty_droplets":
        norm = np.full(counts.shape, np.nan)
        covered = np.zeros(adata.n_obs, dtype=bool)
        for sample, bg in backgrounds.items():
            rows, column = _sample_rows(adata, sample)
            if rows.size == 0:
                notes.append(f"Sample '{sample}' has empty droplets but no cells here: left out.")
                continue
            bg_names = _display_names(bg["features"])
            absent = [n for n in names if n not in bg_names]
            if absent:
                raise DsbError(f"The raw matrix of '{sample}' does not have these antibodies of the "
                               f"panel: {', '.join(absent[:10])}. Attach the raw matrix of the same run.")
            n_drops, mean, sd = window_stats(bg, rna_window, prot_window)
            if n_drops < MIN_DROPLETS:
                raise DsbError(f"Only {n_drops} empty droplets of '{sample}' are inside the window "
                               f"(RNA {rna_window[0]}-{rna_window[1]}, protein {prot_window[0]}-"
                               f"{prot_window[1]} log10 UMIs); at least {MIN_DROPLETS} are needed. "
                               "Widen the window.")
            order = [bg_names.index(n) for n in names]
            mean, sd = mean[order], sd[order]
            norm[rows] = step1_empty(counts[rows], mean, sd)
            covered[rows] = True
            samples.append({"sample": sample, "column": column, "n_cells": int(rows.size),
                            "n_droplets": n_drops, "n_cells_excluded": int(bg["n_cells_excluded"]),
                            "raw_files": _raw_files(adata, sample)})
            if n_drops < FEW_DROPLETS:
                notes.append(f"'{sample}': only {n_drops:,} empty droplets in the window; the "
                             "background estimate is rough.")
            for p, mu_, sd_ in zip(proteins, mean, sd):
                p.setdefault("background_mean", {})[sample] = round(float(mu_), 5)
                p.setdefault("background_sd", {})[sample] = round(float(sd_), 5)
            low = [n for n, s in zip(names, sd) if s < LOW_SD]
            if low:
                notes.append(f"'{sample}': little spread in the background (SD < {LOW_SD}) for "
                             f"{', '.join(low[:8])}: check their values.")
        if not covered.any():
            raise DsbError("None of the dataset's cells belong to a sample with empty droplets.")
        if not covered.all():
            raise DsbError(f"{int((~covered).sum()):,} cells belong to no sample with a raw matrix: "
                           "attach the raw matrix of every sample, or use the no-empty-droplet mode.")
    else:
        norm, mu1, fitted = step1_model_negative(counts)
        for p, mu_, ok_ in zip(proteins, mu1, fitted):
            p["empirical_background"] = round(float(mu_), 5)
            if not ok_:
                p["note"] = "no background fit: log values only"
        notes.append("No empty droplets were used: each protein's background is the lower of two "
                     "groups of its values in the cells (dsb ModelNegativeADTnorm). A protein that "
                     "every cell expresses can then sit near 0.")
    n_failed = 0
    if denoise:
        norm, tech, bg_mean, n_failed = step2(norm, iso_idx)
        if not iso_idx:
            notes.append("No isotype controls in the panel: the technical component is each cell's "
                         "background mean alone (dsb use.isotype.control = FALSE).")
        if n_failed:
            notes.append(f"{n_failed:,} cells had no two-group fit (all proteins alike): their "
                         "lower-half mean was used.")
        tech_summary = {"mean": round(float(np.mean(tech)), 4), "sd": round(float(np.std(tech, ddof=1)), 4)
                        if tech.size > 1 else 0.0,
                        "corr_background_mean": _corr(tech, bg_mean)}
    else:
        tech_summary = None
    for j, p in enumerate(proteins):
        col = norm[:, j]
        p["median"] = round(float(np.median(col)), 4)
        p["q99"] = round(float(np.quantile(col, 0.99)), 4)
    record = {
        "version": 1, "mode": mode, "denoise": bool(denoise), "isotypes": iso,
        "pseudocount": PSEUDOCOUNT if mode == "empty_droplets" else NEG_PSEUDOCOUNT,
        "rna_window": list(rna_window) if mode == "empty_droplets" else None,
        "prot_window": list(prot_window) if mode == "empty_droplets" else None,
        "samples": samples, "n_cells": int(adata.n_obs), "n_proteins": len(names),
        "n_droplets": int(sum(s["n_droplets"] for s in samples)), "n_fit_failed": n_failed,
        "technical": tech_summary, "proteins": proteins, "notes": notes,
        "method": _method_text(mode, denoise, iso), "url": DSB_URL,
    }
    return norm.astype(np.float32), record


def _corr(a, b) -> float | None:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.size < 3 or a.std() == 0 or b.std() == 0:
        return None
    return round(float(abs(np.corrcoef(a, b)[0, 1])), 4)


def _raw_files(adata, sample: str) -> list[str]:
    from scview.core import seq_metrics

    rec = (seq_metrics.read(adata) or {}).get("samples", {}).get(sample) or {}
    return [str(f) for f in (rec.get("files") or {}).get("raw") or []]


def _method_text(mode: str, denoise: bool, iso: list[str]) -> str:
    first = ("step I against empty droplets (log(x + 10), background mean and SD)"
             if mode == "empty_droplets" else
             "step I against an empirical background in the cells (ModelNegativeADTnorm, log(x + 1))")
    if not denoise:
        return f"dsb {first}; no step II"
    second = (f"step II with isotypes {', '.join(iso)} and the per-cell background mean" if iso
              else "step II with the per-cell background mean (no isotypes)")
    return f"dsb {first}; {second}"


def run(adata, **kwargs) -> dict:
    """Compute DSB and keep it in the dataset: obsm[DSB_KEY], the record in uns, a
    provenance step. Returns the record."""
    from scview.core import provenance

    values, record = compute(adata, **kwargs)
    cols = [p["protein"] for p in record["proteins"]]
    adata.obsm[DSB_KEY] = pd.DataFrame(values, index=adata.obs_names, columns=cols)
    adata.uns[UNS_RECORD] = json.dumps(record)
    try:
        provenance.record_step(
            adata, step=STEP, tool="scView DSB (dsb 2.0.1 method)",
            params={"mode": record["mode"], "denoise": record["denoise"], "isotypes": record["isotypes"],
                    "rna_window": record["rna_window"], "prot_window": record["prot_window"],
                    "pseudocount": record["pseudocount"],
                    "samples": [{k: s[k] for k in ("sample", "column", "raw_files", "n_droplets")}
                                for s in record["samples"]],
                    "proteins": cols, "features": _features_of(adata, cols)},
        )
    except Exception as e:  # provenance must never block the result
        logger.warning("Could not record %s provenance: %s", STEP, e)
    return record


def _features_of(adata, names: list[str]) -> list[str]:
    """The 10x feature name of each protein (for the replay script reading a 10x file)."""
    from scview.core import protein

    original = (adata.uns.get(protein.UNS_KEY) or {}).get("original") or {}
    try:
        original = dict(original)
    except (TypeError, ValueError):
        original = {}
    return [str(original.get(n, n)) for n in names]


def read_record(adata) -> dict | None:
    raw = getattr(adata, "uns", {}).get(UNS_RECORD)
    if raw is None:
        return None
    try:
        rec = json.loads(raw) if isinstance(raw, (str, bytes)) else dict(raw)
    except (TypeError, ValueError):
        return None
    return rec if isinstance(rec, dict) and DSB_KEY in getattr(adata, "obsm", {}) else None


def status(adata) -> dict:
    """For the Protein tab: which modes can run (and why not), the empty droplets per
    sample with the droplet density for the window plot, the default isotypes and
    windows, and the last run's record."""
    from scview.core import protein

    if not protein.has_protein(adata):
        return {"available": False, "reason": "This dataset has no CITE-seq protein counts.",
                "computed": False}
    m = adata.obsm.get(protein.COUNTS_KEY)
    if not isinstance(m, pd.DataFrame) or m.shape[1] == 0:
        return {"available": False, "computed": False,
                "reason": "The protein values here are not counts (no obsm['protein_counts']): DSB "
                          "needs raw antibody counts."}
    names = [str(c) for c in m.columns]
    samples = []
    for sample, bg in read_backgrounds(adata).items():
        rows, _ = _sample_rows(adata, sample)
        n_drops, _, _ = window_stats(bg, RNA_WINDOW, PROT_WINDOW)
        bg_names = _display_names(bg["features"])
        samples.append({"sample": sample, "n_cells": int(rows.size), "n_droplets_default": n_drops,
                        "n_cells_excluded": int(bg["n_cells_excluded"]),
                        "missing_antibodies": [n for n in names if n not in bg_names][:10],
                        "density": _density(bg)})
    return {
        "available": True, "reason": None,
        "empty_droplets": bool(samples),
        "empty_droplets_reason": None if samples else (
            "No raw matrix with antibody features is attached, so there are no empty droplets. "
            "Attach raw_feature_bc_matrix.h5 under Sequencing QC, or run without them "
            "(dsb's ModelNegativeADTnorm)."),
        "samples": samples, "isotypes": default_isotypes(names), "proteins": names,
        "defaults": {"rna_window": list(RNA_WINDOW), "prot_window": list(PROT_WINDOW),
                     "bin_width": BIN, "rna_max": RNA_BINS * BIN, "prot_max": PROT_BINS * BIN,
                     "min_droplets": MIN_DROPLETS},
        "computed": DSB_KEY in adata.obsm, "record": read_record(adata), "help": HELP,
        "url": DSB_URL,
    }


def drop(adata) -> None:
    """Forget DSB values whose protein panel or cells changed (a merge, hashtags moved out)."""
    adata.obsm.pop(DSB_KEY, None)
    if UNS_RECORD in getattr(adata, "uns", {}):
        del adata.uns[UNS_RECORD]


# ---------------------------------------------------------------------------
# Co-pilot and results
# ---------------------------------------------------------------------------


def grounding_lines(adata) -> tuple[list[str], str | None]:
    """For the co-pilot: how the DSB values were made, once computed."""
    rec = read_record(adata)
    if rec is None:
        return [], None
    if rec["mode"] == "empty_droplets":
        how = (f"background from {rec['n_droplets']:,} empty droplets (log10 RNA "
               f"{rec['rna_window'][0]}-{rec['rna_window'][1]}, protein {rec['prot_window'][0]}-"
               f"{rec['prot_window'][1]})")
    else:
        how = "background estimated from the cells (no empty droplets; ModelNegativeADTnorm)"
    iso = f"isotypes {', '.join(rec['isotypes'])}" if rec["isotypes"] else "no isotype controls"
    detail = f"DSB protein values: {how}; {iso}; step II {'on' if rec['denoise'] else 'off'}"
    top = sorted(rec["proteins"], key=lambda p: -p.get("q99", 0))[:5]
    lines = ["\n## Protein normalisation: DSB (layer 'protein_dsb')",
             f"- {detail} [{TAG}].",
             "- 99th percentile DSB (background SDs above empty droplets): "
             + ", ".join(f"{p['protein']} {p['q99']:.1f}" for p in top)]
    return lines, detail


def results_value(record: dict) -> dict:
    """What History > Results keeps: the per-protein table and the run's settings."""
    rows = []
    for p in record["proteins"]:
        row = {"protein": p["protein"], "isotype": p["isotype"], "median_dsb": p["median"],
               "q99_dsb": p["q99"]}
        for k in ("background_mean", "background_sd"):
            for sample, v in (p.get(k) or {}).items():
                row[f"{k}_{sample}" if len(record["samples"]) > 1 else k] = v
        if "empirical_background" in p:
            row["empirical_background"] = p["empirical_background"]
        rows.append(row)
    return {"proteins": rows, **{k: record[k] for k in ("mode", "denoise", "isotypes", "rna_window",
                                                          "prot_window", "n_droplets", "notes")}}
