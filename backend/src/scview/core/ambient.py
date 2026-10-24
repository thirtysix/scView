"""Ambient ("soup") RNA: how much each sample carries, which markers it may drive, and
an opt-in correction.

Free-floating mRNA from lysed cells is captured in every droplet beside the cell's own,
so the abundant transcripts of one cell type (LYZ in PBMC, haemoglobin, IGKC, insulin)
show up at a low level in every cluster. That inflates DE against those clusters and
makes markers look broadly expressed. The method is SoupX's (Young & Behjati 2020,
GigaScience, doi:10.1093/gigascience/giaa151), implemented here from the paper and the
package documentation (SoupX is GPL-2; no code of it is used):

- **Soup profile** per sample: each gene's share of the UMIs in the empty droplets of the
  raw (unfiltered) 10x matrix, the droplets with more than 0 and fewer than
  ``SOUP_UMI_MAX`` (100) UMIs, SoupX's ``estimateSoup`` default (``soupRange = c(0, 100)``).
  It is read when the raw matrix is attached (``core/seq_metrics.py``, a side file of the
  sample) and kept in ``uns[UNS_SOUP]``: the raw file itself is not kept. Without a raw
  matrix there are no empty droplets; the cells' own summed counts per capture channel
  (``doublets.channels``, the doublet step's rule) stand in, clearly labelled
  approximate, and are used only for the marker flag, never for a contamination fraction.
- **Contamination fraction rho** per sample (SoupX ``autoEstCont``, its defaults): genes
  very specific to one cluster (tf-idf of the share of cells expressing them, SoupX
  ``quickMarkers``: hypergeometric q < 0.01, tf-idf > 1) that are also high in the soup
  (above its 90% quantile), at most 100; for each, the clusters where no cell clearly
  expresses it (Poisson test against the soup at 80% contamination, BH q >= 0.2 in every
  cell, SoupX ``estimateNonExpressingCells``); each (gene, cluster) pair then gives an
  estimate, observed / expected counts, and rho is the mode of the mean of their gamma
  posteriors (prior mode 0.05, sd 0.10), searched over 0.01-0.8 in steps of 0.001; the
  range where the curve is at least half its height is reported with it. Clusters are the
  active clustering (``cluster_qc.default_grouping``); without one the GET says so, and
  the correction step clusters the cells itself (``quick_clusters``).
- **Ambient-driven marker flag** per cluster: a top marker (``marker_selection``, the one
  marker choice) that is among the ``SOUP_TOP_GENES`` (100) genes of the soup and, outside its
  cluster, is found at about what rho x soup puts there (``AMBIENT_RATIO_MIN`` to
  ``AMBIENT_RATIO_MAX``: half to twice). Its signal in the other clusters is then likely
  ambient; far below that the soup adds too little to matter (a clean sample at the 1%
  floor of rho), far above it the gene is expressed there. Without rho (no raw matrix, or
  no estimate) only the soup share is used, from the approximate profile (its top
  ``SOUP_TOP_GENES_APPROX`` genes).
- **Correction** (``correct_counts``, the pipeline's ``ambient_correction`` step, opt-in):
  SoupX's ``adjustCounts`` subtraction for each cell alone (``clusters = FALSE``): rho x
  the cell's UMIs are removed, shared out in proportion to the soup profile over the
  genes the cell has counts for, never below zero (the water-filling SoupX calls
  ``alloc``); then rounded to whole counts per cell (largest remainders, so every value
  stays at or below the original count). The original counts stay in
  ``layers[ORIGINAL_LAYER]``; reset to counts restores them.

SoupX's own default for ``adjustCounts`` works on cluster sums and shares the removed
counts back out to the cells; the per-cell form used here removes the same total from each
cell and needs no clustering at correction time.

Validation against R SoupX 1.6.2 (rocker/r2u, 2026-10-24), same clusters on both sides, on
the 10x raw + filtered matrices of pbmc_1k_v3, heart_1k_v3 and neuron_1k_v3
(cf.10xgenomics.com/samples/cell-exp/3.0.0/<name>/<name>_{raw,filtered}_feature_bc_matrix.h5):
the soup profile (to 1e-16), quickMarkers (all 1,060 / 4,553 / 1,741 genes, cluster and
tf-idf to 5e-15), rho and its range (0.01 at the floor for all three: clean samples) and
adjustCounts(clusters = FALSE) at rho 0.01 / 0.05 / 0.2 (to 5e-11) were identical; with 5%
and 20% soup added to the pbmc_1k_v3 cells both gave 0.053 and 0.202. The simulations in
``tests/test_ambient.py`` recover rho within 5% and match R to the 0.001 step.
"""

from __future__ import annotations

import hashlib
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

from scview.core.object_cache import ObjectCache

logger = logging.getLogger(__name__)

UNS_SOUP = "scview_ambient_soup"  # per sample: genes, ids, counts of the empty droplets
ORIGINAL_LAYER = "counts_uncorrected"  # the counts before the correction
STEP = "ambient_correction"
SOUPX_URL = "https://doi.org/10.1093/gigascience/giaa151"

SOUP_UMI_MAX = 100  # empty droplets: more than 0 and fewer than this many UMIs
NNZ_CHUNK = 8_000_000  # matrix entries read at a time from a raw matrix
MEX_ROWS = 1_000_000  # Matrix Market lines parsed at a time

# autoEstCont / quickMarkers / estimateNonExpressingCells defaults (SoupX 1.6.2)
TFIDF_MIN = 1.0
SOUP_QUANTILE = 0.90
MAX_MARKERS = 100
RHO_RANGE = (0.01, 0.8)
RHO_MAX_FDR = 0.2
PRIOR_RHO = 0.05
PRIOR_RHO_SD = 0.10
MARKER_FDR = 0.01
EXPRESS_CUT = 0.9
RHO_STEP = 0.001
FEW_MARKERS = 10  # SoupX warns under 10 marker genes, and under 10 estimates
RHO_HIGH = 0.2  # a note: high contamination
RHO_IMPLAUSIBLE = 0.5  # SoupX refuses to set more than this without forceAccept

# Marker flag
SOUP_TOP_GENES = 100
# Without a raw matrix the cells' own counts stand in for the soup: only the most abundant
# genes (LYZ, S100A9, FTL in PBMC) are taken as likely to spill.
SOUP_TOP_GENES_APPROX = 30
AMBIENT_RATIO_MIN = 0.5
AMBIENT_RATIO_MAX = 2.0
TOP_SOUP_SHOWN = 20

HELP = {
    "rho": (
        "Contamination fraction (rho): the share of each droplet's UMIs that is ambient RNA "
        "rather than the cell's own, estimated as SoupX does from genes that one cluster "
        "expresses and the others should not. Typical fresh PBMC: 1-5%; above 20% is high."
    ),
    "soup": (
        "The soup: the RNA in empty droplets (more than 0 and fewer than 100 UMIs in the raw, "
        "unfiltered matrix). Its top genes are the ones ambient RNA adds to every cell."
    ),
    "approximate": (
        "No raw matrix is attached, so there are no empty droplets to read the soup from. "
        "The cells' own summed counts stand in for it: good enough to say which markers are "
        "abundant enough to spill, not to estimate a contamination fraction."
    ),
    "flag": (
        "Ambient-looking marker: one of the top genes of the soup, and outside this cluster it "
        "is found at about the level ambient RNA alone explains (half to twice rho x soup). Its "
        "signal in the other clusters is likely ambient, which can inflate DE against them."
    ),
    "flag_approximate": (
        "Possibly ambient: one of the most abundant genes of the whole sample (no raw matrix "
        "to read the soup from), so it is the kind of gene that spills into other clusters."
    ),
    "correction": (
        "Removes rho x each cell's UMIs, shared out over its genes by the soup profile and never "
        "below zero (SoupX's subtraction for each cell), then rounds to whole counts. The rest "
        "of the run uses the corrected counts; the originals are kept and reset to counts "
        "brings them back."
    ),
}


class AmbientError(ValueError):
    """A request or step that can't be met, as a sentence for the user."""


# ---------------------------------------------------------------------------
# Reading the soup from a raw matrix (also copied into replay scripts: REPLAY_CODE)
# ---------------------------------------------------------------------------


def soup_from_h5(path, umi_max=100, nnz_chunk=8_000_000):
    """The soup of a raw Cell Ranger HDF5 matrix (v3+ layout): per Gene Expression
    feature, its UMIs summed over the droplets with more than 0 and fewer than
    ``umi_max`` Gene Expression UMIs, read in blocks of matrix entries. Names are made
    unique as anndata makes a 10x matrix's (``var_names_make_unique``)."""
    with h5py.File(path, "r") as f:
        if "matrix" not in f:
            raise ValueError(f"{path} is not a Cell Ranger v3+ HDF5 matrix.")
        m = f["matrix"]
        n_bc = int(m["shape"][1])
        feats = m["features"]
        names = feats["name"][:].astype(str)
        ids = feats["id"][:].astype(str)
        types = (feats["feature_type"][:].astype(str) if "feature_type" in feats
                 else np.full(names.size, "Gene Expression"))
        gex = np.flatnonzero(types == "Gene Expression")
        slot = np.full(names.size, -1, dtype=np.int64)
        slot[gex] = np.arange(gex.size)
        indptr = np.asarray(m["indptr"][:], dtype=np.int64)
        data, indices = m["data"], m["indices"]
        soup = np.zeros(gex.size, dtype=np.float64)
        n_drops = 0
        n_umis = 0.0
        b0 = 0
        while b0 < n_bc:
            s = int(indptr[b0])
            b1 = int(np.searchsorted(indptr, s + nnz_chunk, side="right")) - 1
            b1 = min(max(b1, b0 + 1), n_bc)
            e = int(indptr[b1])
            if e > s:
                vals = data[s:e].astype(np.float64)
                g = slot[indices[s:e]]
                vals[g < 0] = 0.0
                seg = indptr[b0:b1 + 1] - s
                cs = np.concatenate(([0.0], np.cumsum(vals)))
                tot = cs[seg[1:]] - cs[seg[:-1]]
                keep = (tot > 0) & (tot < umi_max)
                if keep.any():
                    sel = np.repeat(keep, np.diff(seg)) & (g >= 0)
                    soup += np.bincount(g[sel], weights=vals[sel], minlength=gex.size)
                    n_drops += int(keep.sum())
                    n_umis += float(tot[keep].sum())
            b0 = b1
    return _soup_record(names[gex], ids[gex], soup, n_drops, n_umis, umi_max)


def soup_from_mex(matrix, barcodes, features, umi_max=100, rows=1_000_000):
    """``soup_from_h5`` for a raw 10x MEX triplet (matrix.mtx, barcodes.tsv,
    features.tsv; gzipped or not), in two streamed passes over the matrix."""
    import gzip

    def text(p):
        with open(p, "rb") as fh:
            gz = fh.read(2) == b"\x1f\x8b"
        return (gzip.open(p, "rt", encoding="utf-8", errors="replace") if gz
                else open(p, "rt", encoding="utf-8", errors="replace"))

    with text(features) as fh:
        parts = [ln.rstrip("\n").split("\t") for ln in fh if ln.strip()]
    ids = np.array([p[0] for p in parts], dtype=str)
    names = np.array([p[1] if len(p) > 1 else p[0] for p in parts], dtype=str)
    is_gex = np.array([(p[2] if len(p) > 2 else "Gene Expression") == "Gene Expression"
                       for p in parts], dtype=bool)
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
    gex = np.flatnonzero(is_gex)
    slot = np.full(names.size, -1, dtype=np.int64)
    slot[gex] = np.arange(gex.size)

    def chunks():
        with text(matrix) as fh:
            yield from pd.read_csv(fh, sep=" ", header=None, skiprows=n_header, usecols=[0, 1, 2],
                                   dtype={0: np.int64, 1: np.int64, 2: np.float64}, chunksize=rows)

    totals = np.zeros(n_bc, dtype=np.float64)
    for c in chunks():
        g = slot[c[feat_col].to_numpy() - 1]
        w = np.where(g >= 0, c[2].to_numpy(), 0.0)
        totals += np.bincount(c[bc_col].to_numpy() - 1, weights=w, minlength=n_bc)
    keep = (totals > 0) & (totals < umi_max)
    soup = np.zeros(gex.size, dtype=np.float64)
    for c in chunks():
        g = slot[c[feat_col].to_numpy() - 1]
        sel = keep[c[bc_col].to_numpy() - 1] & (g >= 0)
        soup += np.bincount(g[sel], weights=c[2].to_numpy()[sel], minlength=gex.size)
    return _soup_record(names[gex], ids[gex], soup, int(keep.sum()), float(totals[keep].sum()),
                        umi_max)


def _soup_record(names, ids, soup, n_drops, n_umis, umi_max):
    """The stored soup: only genes with UMIs in it (the others are 0)."""
    unique = np.asarray(ad.utils.make_index_unique(pd.Index(names)), dtype=str)
    nz = np.flatnonzero(soup > 0)
    return {"genes": unique[nz], "ids": np.asarray(ids, dtype=str)[nz],
            "counts": np.rint(soup[nz]).astype(np.int64), "n_droplets": int(n_drops),
            "n_umis": int(round(n_umis)), "umi_max": int(umi_max), "n_features": int(len(names))}


def soup_fractions(soup, var_names, var_ids=None):
    """(soup share of each gene in ``var_names``, share of the soup's UMIs those genes
    hold): matched by name, or by feature id when that covers more of the soup (a
    dataset named by Ensembl ids). The shares sum to 1 over the dataset's genes."""
    counts = np.asarray(soup["counts"], dtype=np.float64)
    total = counts.sum()
    if total <= 0:
        return np.zeros(len(var_names)), 0.0
    best = None
    keys = [(soup["genes"], var_names)]
    if var_ids is not None:
        keys.append((soup["ids"], var_ids))
    for soup_keys, var_keys in keys:
        s = pd.Series(counts, index=pd.Index(np.asarray(soup_keys, dtype=str)))
        s = s[~s.index.duplicated()]
        v = s.reindex(pd.Index(np.asarray(var_keys, dtype=str))).fillna(0.0).to_numpy()
        if best is None or v.sum() > best.sum():
            best = v
    covered = float(best.sum() / total)
    return (best / best.sum() if best.sum() > 0 else best), covered


def _alloc_row(target, y, w):
    """How much of ``target`` each of the counts ``y`` gives up, in proportion to the
    weights ``w`` and never more than ``y`` itself (water-filling: an entry that runs
    out passes its remaining share to the others, SoupX's ``alloc``)."""
    ws = w.sum()
    if target <= 0 or ws <= 0:
        return np.zeros_like(y)
    w = w / ws
    if np.all(target * w <= y):
        return target * w
    with np.errstate(divide="ignore", invalid="ignore"):
        order = np.argsort(np.where(w > 0, y / w, np.inf), kind="stable")
    wo, yo = w[order], y[order]
    cw = np.concatenate(([0.0], np.cumsum(wo)[:-1]))
    cy = np.concatenate(([0.0], np.cumsum(yo)[:-1]))
    with np.errstate(divide="ignore", invalid="ignore"):
        k = np.where(wo > 0, yo / wo * (1.0 - cw) + cy, np.inf)
    full = k <= target
    rest = 1.0 - wo[full].sum()
    resid = target - yo[full].sum()
    out = np.where(full, yo, resid * wo / rest if rest > 0 else 0.0)
    back = np.empty_like(out)
    back[order] = out
    return back


def _round_row(v):
    """Whole counts with the row's total rounded: floors, plus one for the largest
    remainders (stable order), so no value goes above its ceiling."""
    fl = np.floor(v + 1e-9)
    need = int(round(float(v.sum()))) - int(fl.sum())
    if need > 0:
        rem = v - fl
        fl[np.argsort(-rem, kind="stable")[:need]] += 1.0
    return fl


def correct_counts(counts, groups, round_to_int=True):
    """Counts with the soup removed: ``groups`` is a list of (row indices, rho, soup
    share per gene); each of those cells loses rho x its UMIs (``_alloc_row`` over its
    genes with counts), rounded to whole counts per cell (``_round_row``) unless
    ``round_to_int`` is false. Rows in no group stay as they are. Returns a CSR matrix
    of ``counts``' dtype (float64 when not rounded)."""
    X = sp.csr_matrix(counts)
    X.sort_indices()
    data = X.data.astype(np.float64)
    indptr, indices = X.indptr, X.indices
    for rows, rho, frac in groups:
        frac = np.asarray(frac, dtype=np.float64)
        for r in np.asarray(rows, dtype=np.int64):
            a, b = indptr[r], indptr[r + 1]
            if b <= a:
                continue
            y = data[a:b]
            removed = _alloc_row(float(rho) * y.sum(), y, frac[indices[a:b]])
            left = np.maximum(y - removed, 0.0)
            data[a:b] = _round_row(left) if round_to_int else left
    dtype = X.dtype if round_to_int else np.float64
    out = sp.csr_matrix((data.astype(dtype), indices.copy(), indptr.copy()), shape=X.shape)
    out.eliminate_zeros()
    return out


# ---------------------------------------------------------------------------
# Statistics: BH, quickMarkers, autoEstCont
# ---------------------------------------------------------------------------


def bh(p) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values along the last axis (R ``p.adjust``)."""
    p = np.asarray(p, dtype=np.float64)
    if p.size == 0:
        return p
    flat = p.reshape(-1, p.shape[-1])
    out = np.empty_like(flat)
    m = flat.shape[1]
    for i, row in enumerate(flat):
        o = np.argsort(row, kind="stable")[::-1]
        ranked = row[o] * m / np.arange(m, 0, -1)
        q = np.minimum.accumulate(ranked)
        res = np.empty(m)
        res[o] = np.minimum(q, 1.0)
        out[i] = res
    return out.reshape(p.shape)


def _onehot(codes: np.ndarray, k: int) -> sp.csr_matrix:
    ok = codes >= 0
    return sp.csr_matrix((np.ones(int(ok.sum())), (codes[ok], np.flatnonzero(ok))),
                         shape=(k, codes.size))


def quick_markers(counts, codes: np.ndarray, k: int, gene_names=None) -> pd.DataFrame:
    """SoupX ``quickMarkers`` (N = all) reduced as ``autoEstCont`` uses it: per gene its
    most specific cluster among those where it passes (hypergeometric BH q <
    ``MARKER_FDR`` on the cells expressing it, more than ``EXPRESS_CUT`` counts), with
    that cluster's tf-idf (share of the cluster's cells expressing it x log(cells /
    cells expressing it)). Columns gene (index), cluster, tfidf; sorted by tf-idf, ties
    by gene name as SoupX's sort leaves them (by index without names). ``codes``: each
    cell's cluster (0..k-1; every cell has one)."""
    from scipy.stats import hypergeom

    X = sp.csr_matrix(counts)
    B = sp.csr_matrix(((X.data > EXPRESS_CUT).astype(np.float64), X.indices, X.indptr),
                      shape=X.shape)
    n_obs = np.asarray((_onehot(codes, k) @ B).todense())  # k x genes
    size = np.bincount(codes, minlength=k).astype(np.float64)
    n = float(codes.size)
    n_tot = n_obs.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        tf = n_obs / size[:, None]
        idf = np.log(n / n_tot)
        score = tf * idf[None, :]
    p = hypergeom.sf(n_obs - 1, n, n_tot[None, :], size[:, None])
    p = np.where(n_tot[None, :] > 0, p, 1.0)
    q = bh(p)
    score = np.where((q < MARKER_FDR) & (n_tot[None, :] > 0), score, -np.inf)
    best = np.argmax(score, axis=0)
    top = score[best, np.arange(score.shape[1])]
    genes = np.flatnonzero(np.isfinite(top))
    frame = pd.DataFrame({"gene": genes, "cluster": best[genes], "tfidf": top[genes]})
    frame["name"] = (np.asarray(gene_names, dtype=str)[genes] if gene_names is not None
                     else genes)
    out = frame.sort_values(["tfidf", "name"], ascending=[False, True], kind="stable",
                            ignore_index=True)
    return out.drop(columns="name")


def _gamma_k_theta(mode: float, sd: float) -> tuple[float, float]:
    """SoupX's gamma prior parameters from its mode and standard deviation."""
    v2 = (sd / mode) ** 2
    k = 1.0 + v2 ** -2 / 2.0 * (1.0 + math.sqrt(1.0 + 4.0 * v2))
    return k, mode / (k - 1.0)


@dataclass
class RhoFit:
    """A sample's contamination fraction and how it was reached."""

    rho: float | None
    low: float | None = None
    high: float | None = None
    status: str = "ok"  # ok | no_clusters | few_clusters | no_markers
    n_markers: int = 0
    n_estimates: int = 0
    genes: list[str] = field(default_factory=list)  # the marker genes used, best first
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"rho": self.rho, "rho_low": self.low, "rho_high": self.high,
                "rho_status": self.status, "n_markers": self.n_markers,
                "n_estimates": self.n_estimates, "rho_genes": self.genes[:10],
                "notes": list(self.notes)}


def estimate_rho(counts, codes: np.ndarray, frac: np.ndarray, gene_names=None) -> RhoFit:
    """SoupX ``autoEstCont`` on one sample: ``counts`` cells x genes (raw counts),
    ``codes`` each cell's cluster (-1: none; those cells are left out), ``frac`` the soup
    share of each gene (``soup_fractions``)."""
    codes = np.asarray(codes, dtype=np.int64)
    keep = codes >= 0
    X = sp.csr_matrix(counts)[np.flatnonzero(keep)] if not keep.all() else sp.csr_matrix(counts)
    _, codes = np.unique(codes[keep], return_inverse=True)
    k = int(codes.max()) + 1 if codes.size else 0
    if k < 2:
        return RhoFit(None, status="few_clusters",
                      notes=["Fewer than 2 clusters among this sample's cells: rho needs genes "
                             "that one cluster expresses and another does not."])
    frac = np.asarray(frac, dtype=np.float64)
    n_umi = np.asarray(X.sum(axis=1)).ravel()
    mk = quick_markers(X, codes, k, gene_names)
    mk = mk[mk["tfidf"] > TFIDF_MIN]
    soup_min = float(np.quantile(frac, SOUP_QUANTILE))
    mk = mk[frac[mk["gene"].to_numpy()] > soup_min].head(MAX_MARKERS)
    if mk.empty:
        return RhoFit(None, status="no_markers",
                      notes=["No gene is both specific to one cluster and high in the soup, so "
                             "rho can't be estimated (a low-complexity sample?)."])
    genes = mk["gene"].to_numpy()
    sub = X[:, genes].toarray()  # cells x markers
    from scipy.stats import gamma, poisson

    # Clusters where no cell clearly expresses the gene, even at 80% contamination.
    expected = frac[genes][None, :] * n_umi[:, None] * RHO_RANGE[1]
    p = poisson.sf(sub - 1, expected)
    q = bh(p.T)  # per gene, over the cells
    oh = _onehot(codes, k)
    min_q = np.full((k, genes.size), np.inf)
    for c in range(k):
        rows = np.flatnonzero(codes == c)
        if rows.size:
            min_q[c] = q[:, rows].min(axis=1)
    usable = min_q >= RHO_MAX_FDR  # k x markers
    obs = np.asarray(oh @ sub)  # k x markers
    exp = frac[genes][None, :] * np.asarray(oh @ n_umi).ravel()[:, None]
    use = usable & (exp > 0)
    fit = RhoFit(None, n_markers=int(genes.size), n_estimates=int(use.sum()),
                 genes=[str(gene_names[g]) if gene_names is not None else str(g) for g in genes])
    if fit.n_markers < FEW_MARKERS:
        fit.notes.append(f"Only {fit.n_markers} marker genes to estimate from (SoupX warns under "
                         f"{FEW_MARKERS}): rho is uncertain.")
    if fit.n_estimates == 0:
        fit.status = "no_markers"
        fit.notes.append("No cluster is free of the marker genes, so rho can't be estimated.")
        return fit
    if fit.n_estimates < FEW_MARKERS:
        fit.notes.append(f"Only {fit.n_estimates} estimates (SoupX warns under {FEW_MARKERS}): "
                         "rho is unstable.")
    kk, theta = _gamma_k_theta(PRIOR_RHO, PRIOR_RHO_SD)
    o, e = obs[use], exp[use]
    probes = np.round(np.arange(0, 1.0 + RHO_STEP / 2, RHO_STEP), 6)
    post = np.array([gamma.pdf(r, kk + o, scale=theta / (1.0 + theta * e)).mean() for r in probes])
    window = (probes >= RHO_RANGE[0]) & (probes <= RHO_RANGE[1])
    w_probes, w_post = probes[window], post[window]
    i = int(np.argmax(w_post))
    half = w_probes[w_post >= w_post[i] / 2.0]
    fit.rho, fit.low, fit.high = float(w_probes[i]), float(half.min()), float(half.max())
    if fit.rho >= RHO_IMPLAUSIBLE:
        fit.notes.append(f"rho {fit.rho:.0%} is implausibly high (SoupX refuses over "
                         f"{RHO_IMPLAUSIBLE:.0%} unless forced): check the clusters and the raw matrix.")
    elif fit.rho >= RHO_HIGH:
        fit.notes.append(f"rho {fit.rho:.0%} is high: a fifth or more of each droplet's RNA is ambient.")
    if fit.rho <= RHO_RANGE[0]:
        fit.notes.append("At the 1% floor SoupX searches from: little ambient RNA.")
    return fit


# ---------------------------------------------------------------------------
# The dataset's samples, counts and clusters
# ---------------------------------------------------------------------------


@dataclass
class Sample:
    """One sample (capture channel) and its soup."""

    name: str
    column: str | None  # obs column naming it; None: every cell
    rows: np.ndarray
    source: str  # "empty_droplets" | "cells" (the approximate profile)
    frac: np.ndarray  # soup share per gene of the counts matrix
    covered: float = 1.0
    record: dict | None = None  # the stored soup (empty droplets)
    raw_files: list[str] = field(default_factory=list)


def read_soups(adata) -> dict[str, dict]:
    """The stored soups by sample (``uns[UNS_SOUP]``); {} when none."""
    raw = getattr(adata, "uns", {}).get(UNS_SOUP)
    if not isinstance(raw, dict):
        return {}
    return {str(k): v for k, v in raw.items() if isinstance(v, dict) and "counts" in v}


def counts_matrix(adata):
    """(raw counts for X's genes, where from): the counts before any correction
    (``layers[ORIGINAL_LAYER]``), else ``normalization.find_counts``. Raises
    AmbientError without counts."""
    from scview.core.normalization import find_counts
    from scview.core.pipeline import looks_like_counts

    if ORIGINAL_LAYER in adata.layers and looks_like_counts(adata.layers[ORIGINAL_LAYER]):
        return adata.layers[ORIGINAL_LAYER], f"layers['{ORIGINAL_LAYER}']"
    try:
        return find_counts(adata)
    except ValueError:
        raise AmbientError("Ambient RNA needs raw counts (layers['counts'], .raw or X), and this "
                           "dataset has none.") from None


def _var_ids(adata):
    for c in ("gene_ids", "gene_id", "ids", "id", "ensembl_id", "gene_ensembl"):
        if c in adata.var.columns:
            return adata.var[c].astype(str).to_numpy()
    return None


def _record_rows(adata, rec: dict | None) -> tuple[np.ndarray, str | None]:
    """The cells of a sequencing record's sample: obs[column] == sample, or every cell."""
    column = (rec or {}).get("column")
    if column and column in adata.obs.columns:
        mask = (adata.obs[column].astype(str) == str(rec.get("sample"))).to_numpy()
        return np.flatnonzero(mask), str(column)
    if column:
        return np.zeros(0, dtype=np.int64), str(column)
    return np.arange(adata.n_obs), None


def samples(adata, counts=None) -> tuple[list[Sample], list[str]]:
    """The samples with a soup from empty droplets, or (none) the capture channels with
    the approximate profile of their cells; and notes."""
    from scview.core import seq_metrics

    if counts is None:
        counts, _ = counts_matrix(adata)
    soups = read_soups(adata)
    records = (seq_metrics.read(adata) or {}).get("samples", {})
    var_ids = _var_ids(adata)
    out: list[Sample] = []
    notes: list[str] = []
    for name, soup in soups.items():
        rows, column = _record_rows(adata, records.get(name))
        if rows.size == 0:
            notes.append(f"Sample '{name}' has a soup but no cells here ('{column}' no longer "
                         "holds it): left out.")
            continue
        frac, covered = soup_fractions(soup, adata.var_names, var_ids)
        if covered < 0.5:
            notes.append(f"Sample '{name}': the dataset's genes hold only {covered:.0%} of the "
                         "soup's UMIs (other gene names?): the estimate is unreliable.")
        raw = (records.get(name) or {}).get("files", {}).get("raw") or []
        out.append(Sample(name, column, rows, "empty_droplets", frac, covered, soup, list(raw)))
    if out:
        covered_rows = np.unique(np.concatenate([s.rows for s in out]))
        if covered_rows.size < adata.n_obs:
            notes.append(f"{adata.n_obs - covered_rows.size:,} cells belong to no sample with a "
                         "raw matrix: they get no estimate (and are not corrected).")
        return out, notes
    from scview.core import doublets

    ch = doublets.channels(adata)
    sums = np.asarray((_onehot(ch.codes, len(ch.names)) @ sp.csr_matrix(counts)).todense())
    for i, label in enumerate(ch.names):
        rows = np.flatnonzero(ch.codes == i)
        if rows.size == 0:
            continue
        tot = sums[i].astype(np.float64)
        frac = tot / tot.sum() if tot.sum() > 0 else tot
        out.append(Sample(str(label), ch.column if ch.column else None, rows, "cells", frac))
    return out, notes


def _take_rows(X, rows: np.ndarray):
    """``X[rows]``, or ``X`` itself when that is every row in order (no copy)."""
    if rows.size == X.shape[0] and np.array_equal(rows, np.arange(X.shape[0])):
        return X
    return X[rows]


def cluster_codes(adata, column: str | None) -> tuple[np.ndarray, list[str]] | None:
    """Each cell's code in ``column`` (-1 for none) and the labels; None without one."""
    if not column or column not in adata.obs.columns:
        return None
    labels = adata.obs[column]
    codes, uniques = pd.factorize(labels.astype(str).where(labels.notna()), sort=False)
    return np.asarray(codes, dtype=np.int64), [str(u) for u in uniques]


def quick_clusters(counts, seed: int = 0) -> np.ndarray:
    """Clusters for the rho estimate when the dataset has none (a run from counts):
    shifted log, 2,000 variable genes, 30 PCs, 15 neighbours, Leiden at resolution 1 with
    two passes (``core/leiden.py``, in a child process). Codes per cell."""
    import scanpy as sc

    from scview.core import leiden

    X = sp.csr_matrix(counts, dtype=np.float32)
    a = ad.AnnData(X=X)
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    n_top = min(2000, max(10, a.n_vars - 1))
    sc.pp.highly_variable_genes(a, n_top_genes=n_top, flavor="seurat")
    n_pcs = int(max(2, min(30, a.n_obs - 1, int(a.var["highly_variable"].sum()) - 1)))
    sc.pp.pca(a, n_comps=n_pcs, mask_var="highly_variable", random_state=seed)
    sc.pp.neighbors(a, n_neighbors=min(15, max(2, a.n_obs - 1)), random_state=seed)
    lab = leiden.labels(a.obsp["connectivities"], 1.0, random_state=seed, n_iterations=2)
    return np.asarray(pd.Categorical(lab).codes, dtype=np.int64)


# ---------------------------------------------------------------------------
# Diagnostic: per-sample rho, soup genes, marker flags
# ---------------------------------------------------------------------------


def _top_genes(frac: np.ndarray, names, n: int) -> list[dict]:
    idx = np.argsort(-frac, kind="stable")[:n]
    return [{"gene": str(names[i]), "share": round(float(frac[i]), 5)} for i in idx if frac[i] > 0]


def fit_samples(adata, column: str | None, counts=None, codes=None) -> tuple[list[Sample], dict[str, RhoFit], list[str]]:
    """Each sample's soup and (from empty droplets, with clusters) its rho."""
    if counts is None:
        counts, _ = counts_matrix(adata)
    found, notes = samples(adata, counts)
    fits: dict[str, RhoFit] = {}
    if codes is None:
        cc = cluster_codes(adata, column)
        codes = None if cc is None else cc[0]
    X = sp.csr_matrix(counts)
    for s in found:
        if s.source != "empty_droplets":
            continue
        if codes is None:
            fits[s.name] = RhoFit(None, status="no_clusters",
                                  notes=["No clustering yet: rho is estimated from genes one "
                                         "cluster expresses and the others don't. Run clustering "
                                         "(the correction step clusters the cells itself)."])
            continue
        fits[s.name] = estimate_rho(_take_rows(X, s.rows), codes[s.rows], s.frac, adata.var_names)
    return found, fits, notes


def marker_flags(adata, column: str, found: list[Sample], fits: dict[str, RhoFit],
                 counts=None) -> tuple[dict[str, list[dict]], bool]:
    """Per group of ``column``: its ambient-looking top markers (see the top of the
    module), each {gene, soup_rank, ratio}; and whether the flags are approximate (no
    rho: soup share alone)."""
    from scview.core.marker_selection import group_markers

    with_rho = [s for s in found if s.name in fits and fits[s.name].rho is not None]
    approximate = not with_rho
    use = with_rho or found
    markers = group_markers(adata, column)
    cc = cluster_codes(adata, column)
    if not markers or cc is None:
        return {}, approximate
    codes, labels = cc
    if counts is None:
        counts, _ = counts_matrix(adata)
    names = pd.Index(adata.var_names.astype(str))
    if not use:
        return {}, approximate
    # The soup's top genes, cells-weighted over the samples used.
    weights = np.array([s.rows.size for s in use], dtype=np.float64)
    mean_frac = sum(w * s.frac for w, s in zip(weights, use)) / weights.sum()
    rank = np.empty(mean_frac.size, dtype=np.int64)
    rank[np.argsort(-mean_frac, kind="stable")] = np.arange(1, mean_frac.size + 1)
    wanted = sorted({names.get_loc(g) for gs in markers.values() for g in gs
                     if g in names and isinstance(names.get_loc(g), (int, np.integer))})
    top = SOUP_TOP_GENES_APPROX if approximate else SOUP_TOP_GENES
    wanted = [g for g in wanted if rank[g] <= top and mean_frac[g] > 0]
    out: dict[str, list[dict]] = {}
    if not wanted:
        return out, approximate
    k = len(labels)
    X = sp.csr_matrix(counts)
    cols = X[:, wanted].toarray()  # cells x wanted
    n_umi = np.asarray(X.sum(axis=1)).ravel()
    obs_in = np.zeros((k, len(wanted)))
    obs_all = np.zeros(len(wanted))
    exp_out = np.zeros((k, len(wanted)))
    if not approximate:
        for s in use:
            sc_codes = codes[s.rows]
            sub, u = cols[s.rows], n_umi[s.rows]
            oh = _onehot(sc_codes, k)
            obs_in += np.asarray(oh @ sub)
            obs_all += sub.sum(axis=0)
            umi_in = np.asarray(oh @ u).ravel()
            exp_out += fits[s.name].rho * s.frac[wanted][None, :] * (u.sum() - umi_in)[:, None]
    pos = {g: j for j, g in enumerate(wanted)}
    for gi, group in enumerate(labels):
        flagged = []
        for gene in markers.get(group, []):
            if gene not in names:
                continue
            g = names.get_loc(gene)
            if not isinstance(g, (int, np.integer)) or g not in pos:
                continue
            j = pos[g]
            if approximate:
                flagged.append({"gene": gene, "soup_rank": int(rank[g]), "ratio": None})
                continue
            out_obs = obs_all[j] - obs_in[gi, j]
            ex = exp_out[gi, j]
            if ex <= 0 or out_obs <= 0:
                continue
            ratio = out_obs / ex
            if AMBIENT_RATIO_MIN <= ratio <= AMBIENT_RATIO_MAX:
                flagged.append({"gene": gene, "soup_rank": int(rank[g]), "ratio": round(float(ratio), 2)})
        if flagged:
            out[group] = flagged
    return out, approximate


def _hb_note(found: list[Sample], names, species: str = "unknown") -> str | None:
    """Haemoglobin among the soup's top genes (``core/haemoglobin.py``'s genes)."""
    from scview.core import haemoglobin

    hb = set(haemoglobin.hb_genes(names, species))
    if not hb:
        return None
    hits: list[str] = []
    for s in found:
        if s.source != "empty_droplets":
            continue
        for row in _top_genes(s.frac, names, TOP_SOUP_SHOWN):
            if row["gene"] in hb and row["gene"] not in hits:
                hits.append(row["gene"])
    if not hits:
        return None
    return (f"Haemoglobin ({', '.join(hits)}) is among the soup's top genes: lysed red cells. The "
            "cluster QC card flags clusters whose haemoglobin is ambient rather than their own.")


def correction_record(adata) -> dict | None:
    """The last ambient correction's recorded params, if the counts were corrected."""
    if ORIGINAL_LAYER not in adata.layers:
        return None
    from scview.core import provenance

    for h in reversed(provenance.read_provenance(adata).get("history", [])):
        if isinstance(h, dict) and h.get("step") == STEP:
            return dict(h.get("params") or {})
    return {}


def diagnose(adata, column: str | None = None) -> dict:
    """The Ambient RNA card's answer: per sample its soup (top genes), rho and notes; the
    ambient-looking markers per group of ``column`` (default: the active clustering);
    whether the counts were corrected."""
    from scview.core import cluster_qc
    from scview.core.species import detect_species

    if column and column not in adata.obs.columns:
        raise AmbientError(f"Column '{column}' not found.")
    column = column or cluster_qc.default_grouping(adata)
    counts, where = counts_matrix(adata)
    found, fits, notes = fit_samples(adata, column, counts)
    names = adata.var_names.astype(str)
    rows = []
    for s in found:
        row: dict[str, Any] = {
            "sample": s.name, "column": s.column, "n_cells": int(s.rows.size), "source": s.source,
            "top_genes": _top_genes(s.frac, names, TOP_SOUP_SHOWN),
            "genes_covered": round(s.covered, 4),
        }
        if s.record is not None:
            row.update(n_droplets=int(s.record.get("n_droplets", 0)),
                       n_umis=int(s.record.get("n_umis", 0)),
                       umi_max=int(s.record.get("umi_max", SOUP_UMI_MAX)), raw_files=s.raw_files)
        fit = fits.get(s.name)
        row.update(fit.to_dict() if fit is not None else RhoFit(None, status="no_soup").to_dict())
        rows.append(row)
    flags: dict[str, list[dict]] = {}
    approximate = True
    if column:
        try:
            flags, approximate = marker_flags(adata, column, found, fits, counts)
        except Exception as e:  # noqa: BLE001 - the flags never fail the card
            logger.warning("Ambient marker flags failed: %s", e)
    try:
        species = detect_species(adata.var_names).species
    except Exception:  # noqa: BLE001
        species = "unknown"
    return {
        "available": any(s.source == "empty_droplets" for s in found),
        "approximate": not any(s.source == "empty_droplets" for s in found),
        "column": column,
        "counts_from": where,
        "samples": rows,
        "markers": {"column": column, "approximate": approximate, "groups": flags,
                    "n_flagged": sum(len(v) for v in flags.values())},
        "hb_note": _hb_note(found, names, species),
        "notes": notes,
        "correction": correction_record(adata),
        "help": HELP,
        "method": ("SoupX (Young & Behjati 2020): soup from droplets with 0 < UMIs < "
                   f"{SOUP_UMI_MAX}; rho by autoEstCont's defaults."),
        "url": SOUPX_URL,
    }


# Results per loaded dataset object (the dataset manager drops the object after every
# run, so a cached answer never outlives its counts), keyed by what edits made in place
# change: the column and its labels, its stored markers, the soups and sequencing records.
_CACHE = ObjectCache()


def _cache_key(adata, column: str | None) -> tuple:
    from scview.core import cluster_qc, seq_metrics
    from scview.core.ora_background import resolve_markers_key

    col = column or cluster_qc.default_grouping(adata) or ""
    labels = ""
    if col and col in adata.obs.columns:
        values = adata.obs[col]
        codes = pd.factorize(values.astype(str))[0]
        labels = hashlib.sha1(np.ascontiguousarray(codes).tobytes()).hexdigest()
    mkey = resolve_markers_key(adata.uns, col) if col else None
    return ("ambient", column or "", col, labels, id(adata.uns.get(mkey)) if mkey else 0,
            id(adata.uns.get(UNS_SOUP)), id(adata.uns.get(seq_metrics.UNS_KEY)),
            id(adata.X), ORIGINAL_LAYER in adata.layers)


def diagnose_cached(adata, column: str | None = None) -> dict:
    """``diagnose``, answered again from memory while nothing it reads has changed."""
    key = _cache_key(adata, column)
    hit = _CACHE.get(adata, key)
    if hit is None:
        hit = diagnose(adata, column)
        _CACHE.put(adata, hit, key)
    return hit


def flagged_markers(adata, column: str) -> dict[str, list[str]]:
    """Per group: the gene names of its ambient-looking markers (cluster QC). {} when
    none can be worked out."""
    try:
        body = diagnose_cached(adata, column)
    except Exception as e:  # noqa: BLE001 - never fails a caller
        logger.debug("ambient flags unavailable: %s", e)
        return {}
    return {g: [r["gene"] for r in rows] for g, rows in body["markers"]["groups"].items()}


# ---------------------------------------------------------------------------
# The pipeline step
# ---------------------------------------------------------------------------


def run_correction(adata, rho: float | None = None) -> dict:
    """The ``ambient_correction`` step: X (raw counts) replaced by the soup-corrected
    counts, the originals kept in ``layers[ORIGINAL_LAYER]`` (a re-run starts from them,
    never corrects twice), ``layers['counts']`` set to the corrected ones when present.
    ``rho``: every sample's (None: each sample's own estimate; clusters from the active
    clustering, else ``quick_clusters``). Returns what to record."""
    from scview.core import cluster_qc
    from scview.core.pipeline import looks_like_counts

    if not read_soups(adata):
        raise AmbientError("No soup profile: attach the raw (unfiltered) matrix of each sample "
                           "(Data Assessment > Sequencing QC) before correcting for ambient RNA.")
    if rho is not None and not (0.0 <= float(rho) < 1.0):
        raise AmbientError(f"rho must be between 0 and 1 (got {rho}).")
    # X must hold counts (the run's other steps go on from it); a re-run corrects the
    # originals again, never the corrected counts.
    if not looks_like_counts(adata.X):
        raise AmbientError("Ambient RNA correction works on the raw counts in X: run it from counts "
                           "(before normalisation; Reprocess from scratch resets to counts).")
    if ORIGINAL_LAYER in adata.layers:
        original = adata.layers[ORIGINAL_LAYER]
        where = f"layers['{ORIGINAL_LAYER}']"
    else:
        original, where = adata.X, "X"
    X = sp.csr_matrix(original)
    found, notes = samples(adata, X)
    found = [s for s in found if s.source == "empty_droplets"]
    column = None
    codes = None
    if rho is None:
        column = cluster_qc.default_grouping(adata)
        cc = cluster_codes(adata, column)
        if cc is None:
            codes = quick_clusters(X)
            column = None
        else:
            codes = cc[0]
    groups = []
    recorded = []
    for s in found:
        if rho is None:
            fit = estimate_rho(_take_rows(X, s.rows), codes[s.rows], s.frac, adata.var_names)
            if fit.rho is None:
                raise AmbientError(f"Sample '{s.name}': {' '.join(fit.notes)} Set rho by hand to "
                                   "correct anyway.")
            value, source, extra = fit.rho, "estimated", {"rho_low": fit.low, "rho_high": fit.high,
                                                          "n_markers": fit.n_markers}
        else:
            value, source, extra = float(rho), "set", {}
        groups.append((s.rows, value, s.frac))
        recorded.append({"sample": s.name, "column": s.column, "n_cells": int(s.rows.size),
                         "rho": value, "rho_source": source, "raw_files": s.raw_files,
                         "genes_covered": round(s.covered, 4), **extra})
    corrected = correct_counts(X, groups)
    per_before = np.asarray(X.sum(axis=1), dtype=np.float64).ravel()  # row sums: no copies
    per_after = np.asarray(corrected.sum(axis=1), dtype=np.float64).ravel()
    before, after = float(per_before.sum()), float(per_after.sum())
    for rec, (rows, _, _) in zip(recorded, groups):
        b = float(per_before[rows].sum())
        rec["removed_share"] = round((b - float(per_after[rows].sum())) / b, 4) if b else 0.0
    if ORIGINAL_LAYER not in adata.layers:
        adata.layers[ORIGINAL_LAYER] = X if X is not adata.X else X.copy()
    adata.X = corrected
    if "counts" in adata.layers:
        adata.layers["counts"] = corrected.copy()
    covered = np.unique(np.concatenate([s.rows for s in found])) if found else np.zeros(0)
    return {
        "ambient_rho": rho,
        "input": where,
        "samples": recorded,
        "clusters": column if rho is None else None,
        "quick_clusters": rho is None and column is None,
        "soup_umi_max": SOUP_UMI_MAX,
        "n_cells_uncorrected": int(adata.n_obs - covered.size),
        "removed_share": round((before - after) / before, 4) if before else 0.0,
        "notes": notes,
    }


def restore_original(adata) -> bool:
    """Reset to counts: the counts before the correction back in X and
    ``layers['counts']`` (the layer removed). False when there were none."""
    if ORIGINAL_LAYER not in adata.layers:
        return False
    original = adata.layers[ORIGINAL_LAYER]
    del adata.layers[ORIGINAL_LAYER]
    adata.layers["counts"] = original
    adata.X = original.copy()
    return True


# ---------------------------------------------------------------------------
# Sequencing QC: soups read when a raw matrix is attached
# ---------------------------------------------------------------------------


def soup_for_files(h5: Path | None, mex: tuple[Path, Path, Path] | None) -> dict | None:
    """The soup of a raw matrix (``soup_from_h5`` / ``soup_from_mex``); None without one."""
    if h5 is not None:
        return soup_from_h5(h5, SOUP_UMI_MAX, NNZ_CHUNK)
    if mex is not None:
        return soup_from_mex(*mex, umi_max=SOUP_UMI_MAX, rows=MEX_ROWS)
    return None


def soup_summary(soup: dict) -> dict:
    """What the sequencing record says about a soup (small: the arrays live in UNS_SOUP)."""
    counts = np.asarray(soup["counts"])
    order = np.argsort(-counts, kind="stable")[:5]
    total = float(counts.sum()) or 1.0
    return {"n_droplets": int(soup["n_droplets"]), "n_umis": int(soup["n_umis"]),
            "umi_max": int(soup["umi_max"]),
            "top_genes": [{"gene": str(soup["genes"][i]), "share": round(float(counts[i]) / total, 5)}
                          for i in order]}


def store_soups(adata, soups: dict[str, dict]) -> None:
    """Add (or replace, by sample) soups; a new dict so a save sees the entry changed."""
    current = dict(read_soups(adata))
    current.update(soups)
    adata.uns[UNS_SOUP] = {k: {kk: (np.asarray(vv) if isinstance(vv, (list, np.ndarray)) else vv)
                               for kk, vv in v.items()} for k, v in current.items()}


def remove_soup(adata, sample: str) -> None:
    current = dict(read_soups(adata))
    if current.pop(sample, None) is None:
        return
    if current:
        adata.uns[UNS_SOUP] = current
    else:
        del adata.uns[UNS_SOUP]


# Copied into replay scripts (core/replay_script.py): numpy, scipy, pandas, h5py and
# anndata only, so the script reads the soup and subtracts it as scView does.
REPLAY_CONSTANTS = ("SOUP_UMI_MAX",)
REPLAY_CODE = (soup_from_h5, soup_from_mex, _soup_record, soup_fractions, _alloc_row,
               _round_row, correct_counts, _var_ids)
