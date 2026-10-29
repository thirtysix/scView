"""Bulk RNA-seq deconvolution with the open dataset as the single-cell reference.

A bulk sample is a sum over its cells, so its counts are a mixture of the cell
types' profiles. Given those profiles from the open dataset (raw counts by cell
type, and by donor when there are donors), this estimates each bulk sample's
cell-type fractions. Written from the papers' equations (clean room: MuSiC is
GPL-3, so no code is taken from it):

- ``weighted``: MuSiC's weighted non-negative least squares (Wang et al. 2019,
  Nat Commun 10:380, eqs 1-10). Per donor j and type k the relative expression
  ``theta_jgk`` (the type's summed counts of gene g over all its counts); the
  profile ``theta_gk`` is their mean over donors, ``sigma2_gk`` their variance
  (a donor without ``MIN_DONOR_CELLS`` cells of a type is left out of that type);
  ``S_k`` the mean counts per cell (the donors' mean). With ``A = theta * S`` and
  the bulk ``y`` (counts over their sum on the shared genes), ``u = nnls(A, y)``,
  then repeatedly ``w_g = 1 / (delta2 + sum_k u_k^2 S_k^2 sigma2_gk)`` (``delta2``
  the mean squared residual) and ``u = nnls(sqrt(w) A, sqrt(w) y)`` until no
  fraction moves by ``TOL`` or ``MAX_ITER`` rounds. Genes that vary between donors
  weigh less. Needs ``MIN_DONORS_WEIGHTED`` donors with cells of a type.
- ``nnls``: plain NNLS on the shared marker genes (``marker_selection.
  group_markers``, ``MARKERS_PER_TYPE`` per type, their union), the usual
  baseline (Avila Cobos 2020 found OLS/NNLS among the best with a good
  signature) and the fallback without donors.

Cell fractions are ``p = u / sum(u)``; RNA fractions ``p_k S_k / sum(p S)`` (the
share of the sample's RNA from each type). Guards of our own: ``sigma2`` is
floored at its 1st percentile, ``w`` capped at its 99th, and mitochondrial,
ribosomal, Y-chromosome and XIST/TSIX genes and genes with no reference counts
are left out (sex and mito/ribo content differ by donor and handling, not by
cell type). Everything stays on the linear scale (log input is refused: the
benchmarks agree linear matters most).

The self-check (``iter_check``) holds out each donor in turn: the reference
without it deconvolves its pseudobulk and ``N_MIXTURES`` mixtures of its cells
(Dirichlet(1) fractions, 2,000-10,000 cells drawn with replacement, seeded), and
the estimates are scored against the known fractions (Pearson r, RMSE, bias).
It is optimistic: the mixtures come from the same experiment as the reference.

``iter_reference_zip`` streams the reference as files other tools read (10x
matrix, CIBERSORTx single-cell reference, signature, pseudobulk, cell sizes).
"""

from __future__ import annotations

import csv
import gzip
import io
import logging
import math
import re
import threading
import time
import uuid
import weakref
import zipfile
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import nnls

from scview.core import results_store
from scview.core.gene_filters import is_ribo_mito

logger = logging.getLogger(__name__)

KIND = "deconvolution"
CHECK_KIND = "deconvolution_check"
TAG = "result:deconvolution"
STEP = "bulk_deconvolution"
METHODS = ("auto", "weighted", "nnls")
METHOD_NAMES = {"weighted": "MuSiC-style weighted NNLS", "nnls": "NNLS on marker genes"}

MIN_DONOR_CELLS = 3  # a donor's cells of a type, to count towards its profile
MIN_TYPE_CELLS = 10  # a type's cells in all, to be in the reference
MIN_DONORS_WEIGHTED = 2
MIN_DONORS_CHECK = 3
MAX_TYPES = 50
MAX_DONORS = 300
MAX_ITER = 100
TOL = 1e-3
W_CAP_QUANTILE = 0.99
SIGMA2_FLOOR_QUANTILE = 0.01
MARKERS_PER_TYPE = 50
# A type estimated at >= ABSORB_MIN of a sample and >= ABSORB_RATIO times its largest share in
# any one reference donor is flagged: on Kang, CTRL bulk against a CTRL+STIM reference came out
# 60-75% Mk (2% of the cells, 3% at most), the fit using the least interferon-responsive type to
# absorb the missing response. Against the pooled share alone, NNLS on a matched CTRL reference
# flagged CD8 T (5% pooled) in 2 samples, though one donor has 12%: donors differ that much.
ABSORB_MIN = 0.2
ABSORB_RATIO = 5.0
ABSORB_DONOR_CELLS = 50
MIN_MARKER_CPM = 10.0  # a fallback marker must reach this in its own type

# The bulk file.
MAX_BULK_BYTES = 100 * 1024 * 1024  # as uploaded (a GTEx tissue .gct.gz is ~35 MB)
MAX_BULK_UNPACKED = 1024 * 1024 * 1024
MAX_BULK_ROWS = 200_000
MAX_BULK_SAMPLES = 1000
MIN_GENE_SHARE = 0.5
LOG_MAX = 50  # a file whose largest value is below this is log-scaled
TPM_SUM = 1e6
KEEP_FILES = 3  # bulk files kept per dataset

# The self-check.
N_MIXTURES = 50
MAX_MIXTURES = 100
MIX_CELLS = (2000, 10000)
SECONDS_PER_FIT = 0.004  # one weighted fit on ~12k genes x 13 types, 2 cores

# The export.
CIBERSORTX_CELLS = 5000
CIBERSORTX_MIN_CELLS = 3

_Y_XIST = {g.upper() for g in (
    "XIST", "TSIX", "RPS4Y1", "RPS4Y2", "DDX3Y", "KDM5D", "UTY", "EIF1AY", "ZFY", "USP9Y", "NLGN4Y",
    "TXLNGY", "PRKY", "TBL1Y", "PCDH11Y", "TMSB4Y", "AMELY", "SRY", "Uty", "Eif2s3y", "Ddx3y", "Kdm5d",
)}
_ENSEMBL = re.compile(r"^ENS[A-Z]*G\d{6,}", re.I)
_VERSION = re.compile(r"\.\d+(_PAR_Y)?$")
_EXCEL_DATE = re.compile(r"^\d{1,2}-(Mar|Sep|Dec|Feb|Apr|Oct|Nov)$|^(Mar|Sep|Dec)-\d{1,2}$", re.I)
_ID_COLUMNS = ("gene_ids", "gene_id", "ensembl_id", "ensembl", "gene_ensembl", "feature_id", "id")
_SYMBOL_HEADERS = ("gene_name", "description", "symbol", "gene_symbol", "gene", "name", "hgnc_symbol",
                   "external_gene_name")


class DeconvolutionError(ValueError):
    """A request this cannot serve (no counts, a bad file, too few donors): a 422."""


def _excluded(name: str) -> bool:
    return is_ribo_mito(name) or name.upper() in _Y_XIST


# ---------------------------------------------------------------------------
# Matrices
# ---------------------------------------------------------------------------


def weighted_sums(M, W: sp.csr_matrix) -> np.ndarray:
    """``W @ M`` (k x genes) for a cells x genes matrix in memory or on disk
    (a large dataset's counts, read by row blocks)."""
    W = sp.csr_matrix(W)
    if hasattr(M, "row_blocks"):
        out = np.zeros((W.shape[0], M.shape[1]), dtype=np.float64)
        Wc = sp.csc_matrix(W)
        for start, block in M.row_blocks():
            part = Wc[:, start:start + block.shape[0]]
            if part.nnz:
                out += np.asarray((sp.csr_matrix(part) @ sp.csr_matrix(block)).toarray(), dtype=np.float64)
        return out
    prod = W @ (M if sp.issparse(M) else np.asarray(M))
    return np.asarray(prod.toarray() if sp.issparse(prod) else prod, dtype=np.float64)


def _indicator(codes: np.ndarray, k: int) -> sp.csr_matrix:
    """k x cells: row i sums the cells with code i (code -1: no row)."""
    ok = codes >= 0
    cols = np.flatnonzero(ok)
    return sp.csr_matrix((np.ones(cols.size), (codes[ok], cols)), shape=(k, codes.size))


# ---------------------------------------------------------------------------
# The reference
# ---------------------------------------------------------------------------


@dataclass
class Reference:
    """Raw counts summed per (donor, type) group of the open dataset."""

    genes: np.ndarray  # every counts gene name
    gene_ids: np.ndarray | None  # Ensembl ids aligned with genes (no version), or None
    usable: np.ndarray  # bool per gene: not excluded, some counts
    types: list[str]
    donors: list[str]  # [] without a donor column
    P: np.ndarray  # groups x genes
    group_type: np.ndarray
    group_donor: np.ndarray  # -1 without donors
    group_cells: np.ndarray
    totals: np.ndarray  # counts per group, all genes
    celltype_column: str
    donor_column: str | None
    donor_reason: str | None
    counts_source: str
    cell_type: np.ndarray  # per cell: type code, -1 not in the reference
    cell_donor: np.ndarray  # per cell: donor code, -1 none
    notes: list[str] = field(default_factory=list)
    cells_filter: dict[str, Any] | None = None  # {"column", "value", "n_cells"}: the reference cells

    @property
    def type_share(self) -> np.ndarray:
        """Each type's share of the reference cells."""
        n = np.bincount(self.group_type, weights=self.group_cells, minlength=len(self.types))
        return n / max(1.0, float(n.sum()))

    @property
    def type_share_max(self) -> np.ndarray:
        """Each type's largest share of one donor's reference cells, over donors with at least
        ABSORB_DONOR_CELLS (a 5-cell donor's 40% says nothing); the pooled share without them."""
        if not self.donors:
            return self.type_share
        n = np.zeros((len(self.donors), len(self.types)))
        np.add.at(n, (self.group_donor, self.group_type), self.group_cells)
        n = n[n.sum(axis=1) >= ABSORB_DONOR_CELLS]
        if not len(n):
            return self.type_share
        return np.maximum(self.type_share, (n / n.sum(axis=1, keepdims=True)).max(axis=0))

    @property
    def n_cells(self) -> int:
        return int(self.group_cells.sum())


def _gene_ids(adata, genes, source: str) -> np.ndarray | None:
    frames = []
    if source == ".raw" and adata.raw is not None:
        frames.append(adata.raw.var)
    frames.append(adata.var)
    for frame in frames:
        for col in _ID_COLUMNS:
            if col not in frame.columns:
                continue
            ids = frame[col].astype(str)
            if not ids.str.match(_ENSEMBL).mean() > 0.5:
                continue
            mapped = pd.Series(ids.to_numpy(), index=frame.index.astype(str))
            mapped = mapped[~mapped.index.duplicated()].reindex([str(g) for g in genes])
            return mapped.fillna("").str.replace(_VERSION, "", regex=True).to_numpy(dtype=object)
    names = pd.Series([str(g) for g in genes])
    if names.str.match(_ENSEMBL).mean() > 0.5:
        return names.str.replace(_VERSION, "", regex=True).to_numpy(dtype=object)
    return None


def default_donor_column(adata) -> tuple[str | None, str | None]:
    """The donor column (``qc_groups.sample_grouping``, a donor, not a batch), or None."""
    from scview.core import qc_groups

    col, why = qc_groups.sample_grouping(adata)
    return (col, why) if why == "donor" else (None, None)


def reference_columns(adata, exclude: tuple[str | None, ...] = ()) -> list[dict[str, Any]]:
    """Condition-like columns (``confounding.condition_columns``) to restrict the reference
    cells by, with their values; a column that only repeats another (Kang's orig_ident is
    stim) is left out."""
    from scview.core.confounding import condition_columns

    obs = adata.obs
    out, seen = [], set()
    for col in condition_columns(obs):
        if col in exclude:
            continue
        s = obs[col].astype(str).where(obs[col].notna())
        key = tuple(pd.factorize(s)[0].tolist())
        if key in seen:
            continue
        seen.add(key)
        vc = s.value_counts()
        out.append({"column": col, "values": [{"value": str(v), "n_cells": int(n)} for v, n in vc.items()]})
    return out


def build_reference(adata, celltype_column: str | None = None, donor_column: str | None = None,
                    cells_column: str | None = None, cells_value: str | None = None) -> Reference:
    """The open dataset's counts summed per donor x cell type.

    ``celltype_column`` None: an author cell-type column, else the clustering;
    ``donor_column`` None detects the donors, "" means none; ``cells_column`` =
    ``cells_value`` keeps only those cells (one condition, to match the bulk samples)."""
    from scview.core.de_composition import default_column
    from scview.core.pseudobulk import PseudobulkError, raw_counts

    obs = adata.obs
    ct = celltype_column or default_column(adata)
    if not ct:
        raise DeconvolutionError("No cell-type or cluster column: annotate or cluster the cells first.")
    if ct not in obs.columns:
        raise DeconvolutionError(f"Column '{ct}' not found.")
    reason: str | None = "requested"
    if donor_column is None:
        donor_column, reason = default_donor_column(adata)
    elif donor_column == "":
        donor_column, reason = None, None
    if donor_column and donor_column not in obs.columns:
        raise DeconvolutionError(f"Column '{donor_column}' not found.")
    if donor_column == ct:
        raise DeconvolutionError("The donor column cannot be the cell-type column.")
    try:
        M, names, source = raw_counts(adata)
    except PseudobulkError:
        raise DeconvolutionError(
            "Deconvolution needs the raw counts (layers['counts'], .raw or X), and this dataset has none: "
            "re-load it from a file that keeps its counts.") from None

    notes: list[str] = []
    labels = obs[ct]
    lab = labels.astype(str).where(labels.notna()).to_numpy(dtype=object)
    cells_filter: dict[str, Any] | None = None
    if cells_column:
        if cells_column not in obs.columns:
            raise DeconvolutionError(f"Column '{cells_column}' not found.")
        if cells_value is None or cells_value == "":
            raise DeconvolutionError(f"Pick which value of '{cells_column}' the reference cells have.")
        keep = (obs[cells_column].astype(str) == str(cells_value)).to_numpy() & obs[cells_column].notna().to_numpy()
        if not keep.any():
            raise DeconvolutionError(f"No cells have {cells_column} = {cells_value}.")
        lab = np.where(keep, lab, None)
        cells_filter = {"column": cells_column, "value": str(cells_value), "n_cells": int(keep.sum())}
    vc = pd.Series(lab).dropna().value_counts()
    small = [str(t) for t, n in vc.items() if n < MIN_TYPE_CELLS]
    types = sorted((str(t) for t, n in vc.items() if n >= MIN_TYPE_CELLS), key=_natural)
    if small:
        notes.append(f"Left out of the reference (fewer than {MIN_TYPE_CELLS} cells): {', '.join(sorted(small)[:10])}"
                     + (" ..." if len(small) > 10 else "") + ".")
    if len(types) < 2:
        raise DeconvolutionError(f"'{ct}' has fewer than 2 groups of {MIN_TYPE_CELLS}+ cells: nothing to deconvolve into.")
    if len(types) > MAX_TYPES:
        raise DeconvolutionError(f"'{ct}' has {len(types)} groups (at most {MAX_TYPES}): pick a cell-type column.")
    tcode = {t: i for i, t in enumerate(types)}
    cell_type = np.array([tcode.get(v, -1) if v is not None else -1 for v in lab], dtype=np.int64)

    donors: list[str] = []
    cell_donor = np.full(adata.n_obs, -1, dtype=np.int64)
    if donor_column:
        d = obs[donor_column]
        codes, uniq = pd.factorize(d.astype(str).where(d.notna()), sort=True)
        if uniq.size > MAX_DONORS:
            raise DeconvolutionError(f"'{donor_column}' has {uniq.size} values (at most {MAX_DONORS}): "
                                     "it looks like a per-cell column, not donors.")
        donors = [str(u) for u in uniq]
        cell_donor = codes.astype(np.int64)
        missing = int(((cell_donor < 0) & (cell_type >= 0)).sum())
        if missing:
            notes.append(f"{missing:,} cells without a {donor_column} value were left out.")
            cell_type = np.where(cell_donor < 0, -1, cell_type)

    nd = max(1, len(donors))
    gcode = np.where(cell_type >= 0, np.maximum(cell_donor, 0) * len(types) + cell_type, -1)
    used = np.unique(gcode[gcode >= 0])
    remap = np.full(nd * len(types), -1, dtype=np.int64)
    remap[used] = np.arange(used.size)
    codes = np.where(gcode >= 0, remap[np.maximum(gcode, 0)], -1)
    P = weighted_sums(M, _indicator(codes, used.size))
    group_cells = np.bincount(codes[codes >= 0], minlength=used.size).astype(np.int64)
    genes = np.asarray([str(g) for g in names], dtype=object)
    usable = (P.sum(axis=0) > 0) & np.fromiter((not _excluded(g) for g in genes), bool, genes.size)
    return Reference(
        genes=genes, gene_ids=_gene_ids(adata, names, source), usable=usable, types=types, donors=donors, P=P,
        group_type=(used % len(types)).astype(np.int64),
        group_donor=(used // len(types)).astype(np.int64) if donors else np.full(used.size, -1, dtype=np.int64),
        group_cells=group_cells, totals=P.sum(axis=1), celltype_column=ct, donor_column=donor_column or None,
        donor_reason=reason if donor_column else None, counts_source=source, cell_type=cell_type,
        cell_donor=cell_donor, notes=notes, cells_filter=cells_filter,
    )


def _natural(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


@dataclass
class Signature:
    """Per type: mean relative expression (usable genes), its variance over donors, cell size."""

    types: list[str]
    type_index: np.ndarray  # into Reference.types
    theta: np.ndarray  # usable genes x types
    sigma2: np.ndarray | None  # usable genes x types, None without 2+ donors
    S: np.ndarray  # mean counts per cell
    n_donors: np.ndarray  # donors with MIN_DONOR_CELLS+ cells, per type
    weighted_ok: bool
    weighted_why: str
    notes: list[str] = field(default_factory=list)


def signature(ref: Reference, exclude_donor: int | None = None) -> Signature:
    """The types' profiles from the groups (without ``exclude_donor``'s)."""
    U = np.flatnonzero(ref.usable)
    notes: list[str] = []
    cols, sig2, sizes, nds, kept = [], [], [], [], []
    for k, name in enumerate(ref.types):
        g = np.flatnonzero((ref.group_type == k) & (ref.group_donor != (-2 if exclude_donor is None else exclude_donor)))
        if ref.donors:
            g = g[ref.group_cells[g] >= MIN_DONOR_CELLS]
        if g.size == 0 or ref.group_cells[g].sum() < (MIN_DONOR_CELLS if ref.donors else MIN_TYPE_CELLS):
            notes.append(f"{name}: too few cells without the held-out donor; left out." if exclude_donor is not None
                         else f"{name}: no donor has {MIN_DONOR_CELLS}+ of its cells; left out.")
            continue
        theta_j = ref.P[np.ix_(g, U)] / ref.totals[g][:, None]
        cols.append(theta_j.mean(axis=0))
        sig2.append(theta_j.var(axis=0, ddof=1) if g.size >= 2 else np.full(U.size, np.nan))
        sizes.append(float((ref.totals[g] / ref.group_cells[g]).mean()))
        nds.append(int(g.size) if ref.donors else 0)
        kept.append(k)
    if len(kept) < 2:
        raise DeconvolutionError("Fewer than 2 cell types have enough cells for a reference.")
    theta = np.column_stack(cols)
    n_donors = np.asarray(nds, dtype=np.int64)
    n_with = int((n_donors >= MIN_DONORS_WEIGHTED).sum())
    sigma2: np.ndarray | None = None
    if not ref.donors:
        ok, why = False, "no donor column, so the variation between donors is unknown"
    elif len(ref.donors) - (exclude_donor is not None) < MIN_DONORS_WEIGHTED:
        ok, why = False, f"fewer than {MIN_DONORS_WEIGHTED} donors"
    elif n_with * 2 < len(kept):
        ok, why = False, (f"only {n_with} of {len(kept)} cell types have {MIN_DONOR_CELLS}+ cells in "
                          f"{MIN_DONORS_WEIGHTED}+ donors")
    else:
        ok = True
        why = f"{len(ref.donors) - (exclude_donor is not None)} donors give each gene's variation between donors"
        sigma2 = np.column_stack(sig2)
        few = np.isnan(sigma2).all(axis=0)
        if few.any():
            # A type in one donor only: its genes' variation is taken as the other types' mean.
            fill = np.nanmean(sigma2[:, ~few], axis=1)
            sigma2[:, few] = fill[:, None]
            names = [ref.types[kept[i]] for i in np.flatnonzero(few)]
            notes.append(f"{', '.join(names)}: in one donor only, so its between-donor variation is the other "
                         "types' mean.")
        pos = sigma2[sigma2 > 0]
        floor = float(np.quantile(pos, SIGMA2_FLOOR_QUANTILE)) if pos.size else 1e-20
        sigma2 = np.maximum(np.nan_to_num(sigma2, nan=floor), floor)
    return Signature(types=[ref.types[k] for k in kept], type_index=np.asarray(kept, dtype=np.int64), theta=theta,
                     sigma2=sigma2, S=np.asarray(sizes), n_donors=n_donors, weighted_ok=ok, weighted_why=why,
                     notes=notes)


def marker_genes(adata, ref: Reference, sig: Signature) -> tuple[np.ndarray, str]:
    """Positions (into the usable genes) of the marker union for ``nnls``, and where
    they came from: the stored markers' shared choice, else the profiles' fold change."""
    from scview.core.marker_selection import group_markers

    U = ref.genes[ref.usable]
    pos = {g: i for i, g in enumerate(U)}
    chosen: set[int] = set()
    try:
        stored = group_markers(adata, ref.celltype_column, n=MARKERS_PER_TYPE)
    except Exception as exc:  # a large dataset's fraction count, an odd stored result
        logger.debug("deconvolution: no stored markers: %s", exc)
        stored = {}
    for t in sig.types:
        chosen.update(pos[g] for g in stored.get(t, []) if g in pos)
    covered = sum(1 for t in sig.types if any(g in pos for g in stored.get(t, [])))
    if covered * 2 >= len(sig.types) and len(chosen) >= len(sig.types):
        return np.asarray(sorted(chosen), dtype=np.int64), (
            f"the stored markers of {ref.celltype_column} (up to {MARKERS_PER_TYPE} per type, the shared marker choice)")
    # No markers stored for the column: genes by fold change of each type's profile.
    cpm = sig.theta * 1e6
    for k in range(cpm.shape[1]):
        rest = np.delete(cpm, k, axis=1).mean(axis=1) if cpm.shape[1] > 1 else np.zeros(cpm.shape[0])
        lfc = np.log2((cpm[:, k] + 1) / (rest + 1))
        cand = np.flatnonzero((cpm[:, k] >= MIN_MARKER_CPM) & (lfc > 0.25))
        chosen.update(cand[np.argsort(-lfc[cand], kind="stable")[:MARKERS_PER_TYPE]].tolist())
    if len(chosen) < len(sig.types):
        chosen = set(range(cpm.shape[0]))
    return np.asarray(sorted(chosen), dtype=np.int64), (
        f"fold change of each type's profile (no stored markers for {ref.celltype_column}; "
        f"up to {MARKERS_PER_TYPE} per type)")


# ---------------------------------------------------------------------------
# The fit
# ---------------------------------------------------------------------------


@dataclass
class Fit:
    p: np.ndarray  # cell fractions (NaN when the fit failed)
    u: np.ndarray
    r2: float
    residual: float  # |y - A u| / |y|
    iterations: int
    converged: bool
    n_genes: int


def _fractions(u: np.ndarray) -> np.ndarray:
    s = float(u.sum())
    return u / s if s > 0 else np.full(u.size, np.nan)


def _score(A: np.ndarray, y: np.ndarray, u: np.ndarray) -> tuple[float, float]:
    r = y - A @ u
    sst = float(((y - y.mean()) ** 2).sum())
    norm = float(np.linalg.norm(y))
    return (1 - float((r ** 2).sum()) / sst if sst > 0 else float("nan"),
            float(np.linalg.norm(r)) / norm if norm > 0 else float("nan"))


def fit_nnls(A: np.ndarray, y: np.ndarray) -> Fit:
    u = nnls(A, y, maxiter=50 * A.shape[1])[0] if A.size else np.zeros(A.shape[1])
    r2, res = _score(A, y, u)
    return Fit(_fractions(u), u, r2, res, 1, True, int(A.shape[0]))


def fit_weighted(A: np.ndarray, y: np.ndarray, S: np.ndarray, sigma2: np.ndarray) -> Fit:
    """MuSiC's iteration (Wang 2019 eqs 7-10) with the weight cap of this module."""
    u = nnls(A, y, maxiter=50 * A.shape[1])[0]
    p = _fractions(u)
    it, converged = 0, False
    S2s = sigma2 * (S ** 2)[None, :]
    for it in range(1, MAX_ITER + 1):  # noqa: B007 - the count is reported
        if not np.isfinite(p).all():
            break
        r = y - A @ u
        delta2 = float((r ** 2).mean())
        w = 1.0 / (delta2 + S2s @ (u ** 2))
        w = np.minimum(w, np.quantile(w, W_CAP_QUANTILE))
        sw = np.sqrt(w)
        u_new = nnls(A * sw[:, None], y * sw, maxiter=50 * A.shape[1])[0]
        p_new = _fractions(u_new)
        done = bool(np.isfinite(p_new).all() and np.max(np.abs(p_new - p)) < TOL)
        u, p = u_new, p_new
        if done:
            converged = True
            break
    r2, res = _score(A, y, u)
    return Fit(p, u, r2, res, it, converged, int(A.shape[0]))


def choose_method(method: str, sig: Signature) -> tuple[str, str]:
    """(method that runs, why)."""
    if method not in METHODS:
        raise DeconvolutionError(f"Unknown method '{method}' (expected one of {', '.join(METHODS)}).")
    if method == "nnls":
        return "nnls", "requested"
    if method == "weighted":
        if not sig.weighted_ok:
            raise DeconvolutionError(f"The weighted method needs {MIN_DONORS_WEIGHTED}+ donors with cells of most "
                                     f"types: {sig.weighted_why}. Pick a donor column, or use NNLS.")
        return "weighted", "requested"
    if sig.weighted_ok:
        return "weighted", f"Auto: weighted, since {sig.weighted_why}."
    return "nnls", f"Auto: NNLS on marker genes, since {sig.weighted_why}."


def deconvolve(Y: np.ndarray, sig: Signature, method: str, markers: np.ndarray | None,
               genes: np.ndarray | None = None) -> list[Fit]:
    """Fit each column of ``Y`` (usable genes x samples, raw scale; ``genes`` the rows
    of the usable genes they are, default all)."""
    rows = np.arange(sig.theta.shape[0]) if genes is None else np.asarray(genes)
    theta = sig.theta[rows]
    A_all = theta * sig.S[None, :]
    if method == "nnls":
        keep = np.flatnonzero(np.isin(rows, markers)) if markers is not None else np.arange(rows.size)
    else:
        keep = np.arange(rows.size)
    out: list[Fit] = []
    for j in range(Y.shape[1]):
        total = float(Y[:, j].sum())
        if total <= 0:
            out.append(Fit(np.full(len(sig.types), np.nan), np.zeros(len(sig.types)), float("nan"), float("nan"),
                           0, False, 0))
            continue
        y = Y[:, j] / total
        if method == "weighted":
            assert sig.sigma2 is not None
            out.append(fit_weighted(A_all, y, sig.S, sig.sigma2[rows]))
        else:
            out.append(fit_nnls(A_all[keep], y[keep]))
    return out


def rna_fractions(p: np.ndarray, S: np.ndarray) -> np.ndarray:
    m = p * S
    s = float(np.nansum(m))
    return m / s if s > 0 else np.full(p.size, np.nan)


# ---------------------------------------------------------------------------
# The bulk file
# ---------------------------------------------------------------------------


@dataclass
class Bulk:
    """A bulk file matched to the reference's genes."""

    positions: np.ndarray  # into Reference.genes
    values: np.ndarray  # matched genes x samples
    samples: list[str]
    n_rows: int  # gene rows in the file (after dropping STAR / HTSeq summary rows)
    n_matched: int
    id_kind: str  # "ensembl" | "symbol"
    input_kind: str  # "counts" | "tpm" | "non-integer"
    transposed: bool
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _unpack(raw: bytes) -> bytes:
    if raw[:2] == b"\x1f\x8b":
        try:
            with gzip.GzipFile(fileobj=io.BytesIO(raw)) as fh:
                data = fh.read(MAX_BULK_UNPACKED + 1)
        except (OSError, EOFError) as exc:
            raise DeconvolutionError(f"The file looks gzipped but could not be read: {exc}") from None
        if len(data) > MAX_BULK_UNPACKED:
            raise DeconvolutionError(f"The unpacked file is over {MAX_BULK_UNPACKED // 1024 ** 2:,} MB.")
        return data
    return raw


def _read_table(raw: bytes) -> pd.DataFrame:
    text = _unpack(raw)
    if b"\x00" in text[:4096]:
        raise DeconvolutionError("This is not a text table: upload a TSV, CSV or GCT file (optionally .gz).")
    lines = text.split(b"\n", 50)
    skip = 0
    if lines and lines[0].strip().startswith(b"#1."):  # GCT: version, then dimensions
        skip = 2
    while skip < len(lines) and lines[skip].startswith(b"#"):
        skip += 1
    if skip >= len(lines):
        raise DeconvolutionError("The file has no table.")
    header = lines[skip]
    sep = "\t" if b"\t" in header else ("," if b"," in header else r"\s+")
    try:
        df = pd.read_csv(io.BytesIO(text), sep=sep, skiprows=skip, header=0, engine="c" if len(sep) == 1
                         else "python", quoting=csv.QUOTE_MINIMAL, low_memory=False)
    except Exception as exc:  # noqa: BLE001 - any parse failure is the file's
        raise DeconvolutionError(f"Could not read the table: {exc}") from None
    if df.shape[1] < 2:
        raise DeconvolutionError("The table needs a gene column and at least one sample column.")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _numeric(col: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(col) and not pd.api.types.is_bool_dtype(col):
        return col.astype(np.float64)
    s = col.astype(str).str.strip()
    s = s.mask(s.str.upper().isin(["", "NA", "NAN", "NULL", "NONE", "-"]))
    return pd.to_numeric(s, errors="coerce")


class _Matcher:
    """Gene identifiers -> positions in the reference's genes."""

    def __init__(self, genes: np.ndarray, ids: np.ndarray | None):
        self.exact = {g: i for i, g in reversed(list(enumerate(genes)))}
        self.upper: dict[str, int] = {}
        for i, g in reversed(list(enumerate(genes))):
            self.upper[str(g).upper()] = i
        self.ids: dict[str, int] = {}
        if ids is not None:
            for i, e in reversed(list(enumerate(ids))):
                if e:
                    self.ids[str(e).upper()] = i

    def match(self, values: list[str]) -> tuple[np.ndarray, str]:
        vals = [v.strip() for v in values]
        ens = sum(1 for v in vals[:2000] if _ENSEMBL.match(v)) > 0.5 * min(len(vals), 2000)
        out = np.full(len(vals), -1, dtype=np.int64)
        if ens:
            for i, v in enumerate(vals):
                if v.upper().endswith("_PAR_Y"):
                    continue  # GENCODE's Y copies of the X PAR genes
                key = _VERSION.sub("", v).upper()
                out[i] = self.ids.get(key, self.exact.get(_VERSION.sub("", v), -1))
            return out, "ensembl"
        for i, v in enumerate(vals):
            j = self.exact.get(v)
            out[i] = j if j is not None else self.upper.get(v.upper(), -1)
        return out, "symbol"


def parse_bulk(raw: bytes, filename: str, ref: Reference) -> Bulk:
    """A bulk expression table (TSV/CSV/GCT, optionally .gz; genes x samples, or
    samples x genes) matched to the reference's genes."""
    if len(raw) > MAX_BULK_BYTES:
        raise DeconvolutionError(f"The file is over {MAX_BULK_BYTES // 1024 ** 2} MB.")
    df = _read_table(raw)
    matcher = _Matcher(ref.genes, ref.gene_ids)
    notes: list[str] = []
    warnings: list[str] = []

    # TCGA / GDC STAR counts: one sample, several quantities as columns.
    low = {c.lower(): c for c in df.columns}
    if "unstranded" in low and ("gene_id" in low or "gene_name" in low):
        keep = [c for c in (low.get("gene_id"), low.get("gene_name")) if c]
        df = df[[*keep, low["unstranded"]]].rename(columns={low["unstranded"]: Path(filename).name.split(".")[0]
                                                            or "sample"})
        notes.append("A GDC STAR counts file: the 'unstranded' counts were used.")

    numeric = {c: _numeric(df[c]) for c in df.columns}
    is_num = {c: bool(numeric[c].notna().mean() >= 0.99) for c in df.columns}
    text_cols = [c for c in df.columns if not is_num[c]]
    transposed = False
    if text_cols:
        id_col = text_cols[0]
        pos, kind = matcher.match(df[id_col].astype(str).tolist())
        # Symbol columns beside an Ensembl one (GCT Description, GDC gene_name) are the fallback.
        if (pos >= 0).mean() < 0.3:
            for c in text_cols[1:]:
                if c.lower() in _SYMBOL_HEADERS:
                    p2, k2 = matcher.match(df[c].astype(str).tolist())
                    if (p2 >= 0).sum() > (pos >= 0).sum():
                        id_col, pos, kind = c, p2, k2
        head_pos, head_kind = matcher.match([c for c in df.columns if c not in text_cols])
        if (head_pos >= 0).sum() > max(10, (pos >= 0).sum()):
            transposed = True
    else:
        head_pos, head_kind = matcher.match(list(df.columns[1:]))
        if (head_pos >= 0).sum() == 0:
            raise DeconvolutionError("No column of gene names or IDs was found (gene symbols or Ensembl IDs).")
        transposed = True
    if transposed:
        # Samples in rows: the first column names them, the other columns are genes.
        first = df.columns[0]
        gene_cols = [c for c in df.columns[1:] if is_num[c]]
        samples = [str(s) for s in df[first].tolist()]
        values = np.column_stack([numeric[c].to_numpy(dtype=np.float64) for c in gene_cols]).T
        ids = gene_cols
        pos, kind = matcher.match(ids)
        notes.append("Samples were in rows; the table was turned so genes are rows.")
    else:
        sample_cols = [c for c in df.columns if is_num[c]]
        if not sample_cols:
            raise DeconvolutionError("No numeric sample columns were found.")
        samples = sample_cols
        ids = df[id_col].astype(str).tolist()
        values = np.column_stack([numeric[c].to_numpy(dtype=np.float64) for c in sample_cols])

    ids = [str(i).strip() for i in ids]
    summary = np.array([i.startswith(("N_", "__")) for i in ids], dtype=bool)
    if summary.any():
        notes.append(f"{int(summary.sum())} STAR / HTSeq summary rows (N_*, __*) were dropped.")
        keep = ~summary
        ids = [i for i, k in zip(ids, keep, strict=True) if k]
        values, pos = values[keep], pos[keep]
    if len(ids) > MAX_BULK_ROWS:
        raise DeconvolutionError(f"{len(ids):,} gene rows (at most {MAX_BULK_ROWS:,}).")
    if values.shape[1] > MAX_BULK_SAMPLES:
        raise DeconvolutionError(f"{values.shape[1]:,} samples (at most {MAX_BULK_SAMPLES:,}).")
    if values.size == 0:
        raise DeconvolutionError("The table has no values.")
    n_na = int(np.isnan(values).sum())
    if n_na:
        raise DeconvolutionError(f"The table has {n_na:,} missing or non-numeric values: fill them (0 for not "
                                 "detected) or remove those rows.")
    if (values < 0).any():
        raise DeconvolutionError("The table has negative values: it looks centred or log-ratio data. Upload raw "
                                 "counts (or TPM) on the linear scale.")
    if float(values.max()) < LOG_MAX:
        raise DeconvolutionError(f"The largest value is {values.max():.3g}: this looks log-scaled. Deconvolution "
                                 "needs linear values: upload the raw counts (or TPM), not log values.")
    dates = sum(1 for i in ids if _EXCEL_DATE.match(i))
    if dates:
        warnings.append(f"{dates} gene names look like dates (Excel turns MARCH1 or SEPT2 into 1-Mar, 2-Sep); "
                        "they were not matched.")

    integer = bool(np.all(np.mod(values, 1) == 0))
    sums = values.sum(axis=0)
    if integer:
        input_kind = "counts"
    elif np.mean(np.abs(sums / TPM_SUM - 1) < 0.1) >= 0.5:
        input_kind = "tpm"
        warnings.append("These look like TPM (non-integer, each sample sums to about a million): the estimates are "
                        "relative, closer to RNA fractions than cell fractions. Raw counts are better.")
    else:
        input_kind = "non-integer"
        notes.append("Non-integer values (e.g. RSEM expected counts) were used as they are.")

    ok = pos >= 0
    agg = pd.DataFrame(values[ok]).groupby(pos[ok], sort=True).sum()
    n_dup = int(ok.sum() - agg.shape[0])
    if n_dup:
        notes.append(f"{n_dup:,} rows named the same gene and were summed.")
    samples = _unique([str(s).strip() or f"sample_{i + 1}" for i, s in enumerate(samples)])
    return Bulk(positions=agg.index.to_numpy(dtype=np.int64), values=agg.to_numpy(dtype=np.float64), samples=samples,
                n_rows=len(ids), n_matched=int(agg.shape[0]), id_kind=kind, input_kind=input_kind,
                transposed=transposed, warnings=warnings, notes=notes)


def _unique(names: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out = []
    for n in names:
        if n in seen:
            seen[n] += 1
            out.append(f"{n}_{seen[n]}")
        else:
            seen[n] = 0
            out.append(n)
    return out


def shared_genes(ref: Reference, bulk: Bulk) -> tuple[np.ndarray, np.ndarray, float]:
    """(rows of the usable genes in the bulk, the bulk's values for them, share of usable genes found)."""
    usable_pos = np.flatnonzero(ref.usable)
    where = np.full(ref.genes.size, -1, dtype=np.int64)
    where[usable_pos] = np.arange(usable_pos.size)
    rows = where[bulk.positions]
    ok = rows >= 0
    share = float(ok.sum()) / max(1, usable_pos.size)
    return rows[ok], bulk.values[ok], share


# ---------------------------------------------------------------------------
# A run
# ---------------------------------------------------------------------------


@dataclass
class Result:
    """One deconvolution (results store kind ``deconvolution``)."""

    id: str
    dataset_id: str
    celltype_column: str
    donor_column: str | None
    donor_reason: str | None
    method: str
    method_requested: str
    method_reason: str
    types: list[str]
    cell_size: list[float]
    n_donors: list[int]  # per type
    n_reference_cells: int
    n_reference_donors: int
    counts_source: str
    filename: str
    sha256: str
    stored_file: str  # name under the dataset's uploads/<id>/deconvolution/
    samples: list[str]
    fractions: pd.DataFrame  # sample, cell_type, cell_fraction, rna_fraction
    fit: pd.DataFrame  # sample, r2, residual, genes, iterations, converged
    n_genes_file: int
    n_genes_matched: int
    gene_share: float
    n_genes_used: int
    marker_source: str | None
    id_kind: str
    input_kind: str
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    created_at: float = 0.0
    cells: str = ""
    reference_cells: dict[str, Any] | None = None  # {"column", "value", "n_cells"}, None: all cells
    type_share: list[float] = field(default_factory=list)  # each type's share of the reference cells
    flags: list[dict[str, Any]] = field(default_factory=list)  # sample, cell_type, fraction, reference_share


def absorber_flags(samples: list[str], types: list[str], P: np.ndarray, share: np.ndarray,
                   share_max: np.ndarray | None = None) -> list[dict[str, Any]]:
    """(sample, type) whose estimated cell fraction is >= ABSORB_MIN and >= ABSORB_RATIO times
    the type's largest share in one reference donor (``share_max``, default the pooled
    ``share``): likely soaking up a sample-reference difference."""
    base = share if share_max is None else share_max
    out = []
    for i, s in enumerate(samples):
        for k, t in enumerate(types):
            p = P[i, k]
            if np.isfinite(p) and p >= ABSORB_MIN and p >= ABSORB_RATIO * base[k]:
                out.append({"sample": s, "cell_type": t, "fraction": float(p), "reference_share": float(share[k])})
    return out


def absorber_warnings(flags: list[dict[str, Any]], n_samples: int, restricted: bool) -> list[str]:
    """One sentence per flagged type."""
    out = []
    by_type: dict[str, list[dict]] = {}
    for f in flags:
        by_type.setdefault(f["cell_type"], []).append(f)
    for t, fs in by_type.items():
        lo, hi = min(f["fraction"] for f in fs), max(f["fraction"] for f in fs)
        span = f"{lo:.0%}" if round(lo, 2) == round(hi, 2) else f"{lo:.0%}-{hi:.0%}"
        where = f"in {fs[0]['sample']}" if len(fs) == 1 else f"in {len(fs)} of {n_samples} samples"
        out.append(f"{t} is {span} {where} but {fs[0]['reference_share']:.0%} of the reference cells: the fit may be "
                   "using it to absorb a difference between the sample and the reference (a condition or tissue "
                   "the reference does not match)."
                   + ("" if restricted else " Try restricting the reference cells to the matching condition."))
    return out


def run(adata, raw: bytes, filename: str, *, celltype_column: str | None = None, donor_column: str | None = None,
        method: str = "auto", dataset_id: str = "", sha256: str = "", stored_file: str = "",
        cells_column: str | None = None, cells_value: str | None = None) -> Result:
    t0 = time.perf_counter()
    if method not in METHODS:
        raise DeconvolutionError(f"Unknown method '{method}' (expected one of {', '.join(METHODS)}).")
    ref = build_reference(adata, celltype_column, donor_column, cells_column, cells_value)
    sig = signature(ref)
    ran, why = choose_method(method, sig)
    bulk = parse_bulk(raw, filename, ref)
    rows, Y, share = shared_genes(ref, bulk)
    if share < MIN_GENE_SHARE:
        raise DeconvolutionError(
            f"Only {share:.0%} of this dataset's genes are in the bulk file (at least {MIN_GENE_SHARE:.0%} are "
            f"needed): check that the genes are symbols or Ensembl IDs of the same species "
            f"({bulk.n_matched:,} of {bulk.n_rows:,} rows matched).")
    markers, marker_source = (marker_genes(adata, ref, sig) if ran == "nnls" else (None, None))
    fits = deconvolve(Y, sig, ran, markers, genes=rows)
    frac_rows, fit_rows = [], []
    for s, f in zip(bulk.samples, fits, strict=True):
        rna = rna_fractions(f.p, sig.S)
        for k, t in enumerate(sig.types):
            frac_rows.append((s, t, float(f.p[k]), float(rna[k])))
        fit_rows.append((s, f.r2, f.residual, f.n_genes, f.iterations, f.converged))
    notes = [*ref.notes, *sig.notes, *bulk.notes]
    failed = [s for s, f in zip(bulk.samples, fits, strict=True) if not np.isfinite(f.p).all()]
    if failed:
        notes.append(f"No fit for {', '.join(failed[:5])}{' ...' if len(failed) > 5 else ''}: no counts on the "
                     "shared genes.")
    if ran == "weighted" and any(not f.converged for f in fits if np.isfinite(f.p).all()):
        notes.append(f"Some samples did not settle within {MAX_ITER} rounds; their last estimate is shown.")
    tshare = ref.type_share[sig.type_index]
    flags = absorber_flags(bulk.samples, sig.types, np.array([f.p for f in fits]), tshare,
                           ref.type_share_max[sig.type_index])
    warnings = [*bulk.warnings, *absorber_warnings(flags, len(bulk.samples), ref.cells_filter is not None)]
    return Result(
        id=uuid.uuid4().hex[:12], dataset_id=dataset_id, celltype_column=ref.celltype_column,
        donor_column=ref.donor_column, donor_reason=ref.donor_reason, method=ran, method_requested=method,
        method_reason=why, types=sig.types, cell_size=[float(x) for x in sig.S],
        n_donors=[int(x) for x in sig.n_donors], n_reference_cells=ref.n_cells, n_reference_donors=len(ref.donors),
        counts_source=ref.counts_source, filename=filename, sha256=sha256, stored_file=stored_file,
        samples=bulk.samples,
        fractions=pd.DataFrame(frac_rows, columns=["sample", "cell_type", "cell_fraction", "rna_fraction"]),
        fit=pd.DataFrame(fit_rows, columns=["sample", "r2", "residual", "genes", "iterations", "converged"]),
        n_genes_file=bulk.n_rows, n_genes_matched=bulk.n_matched, gene_share=share,
        n_genes_used=int(markers.size if markers is not None else rows.size), marker_source=marker_source,
        id_kind=bulk.id_kind, input_kind=bulk.input_kind, warnings=warnings, notes=notes,
        elapsed_seconds=round(time.perf_counter() - t0, 3), created_at=time.time(),
        cells=results_store.cells_hash(adata), reference_cells=ref.cells_filter,
        type_share=[float(x) for x in tshare], flags=flags,
    )


def record(adata, r: Result) -> None:
    """The run in provenance (History); it only adds a result (``ADDITIVE_STEPS``)."""
    from scview.core import provenance

    provenance.record_step(
        adata, step=STEP, tool="scview.deconvolution (MuSiC-style weighted NNLS, Wang 2019; or NNLS on markers)",
        params={"bulk_file": r.filename, "sha256": r.sha256, "n_samples": len(r.samples), "method": r.method,
                "method_requested": r.method_requested, "celltype_column": r.celltype_column,
                "donor_column": r.donor_column, "n_genes_used": r.n_genes_used, "input_kind": r.input_kind,
                "reference_cells": ({"column": r.reference_cells["column"], "value": r.reference_cells["value"]}
                                    if r.reference_cells else None),
                "n_flagged": len(r.flags), "result_id": r.id},
        resources={"seconds": r.elapsed_seconds},
    )


# ---------------------------------------------------------------------------
# The self-check
# ---------------------------------------------------------------------------


@dataclass
class CheckResult:
    """Leave-one-donor-out self-check (results store kind ``deconvolution_check``)."""

    id: str
    dataset_id: str
    celltype_column: str
    donor_column: str
    methods: list[str]
    n_mixtures: int
    seed: int
    donors: list[str]
    types: list[str]
    overall: list[dict[str, Any]]  # per method: r, rmse, bias, n
    per_type: pd.DataFrame  # method, cell_type, r, rmse, bias, n
    points: pd.DataFrame  # method, donor, mixture (-1 = pseudobulk), cell_type, truth, estimate
    notes: list[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    created_at: float = 0.0
    cells: str = ""
    reference_cells: dict[str, Any] | None = None


def check_estimate_seconds(n_donors: int, n_mixtures: int = N_MIXTURES) -> float:
    """Seconds for the self-check: weighted fits (~15 rounds each) dominate."""
    return n_donors * (n_mixtures + 1) * 15 * SECONDS_PER_FIT


def _metrics(truth: np.ndarray, est: np.ndarray) -> dict[str, Any]:
    ok = np.isfinite(est)
    t, e = truth[ok], est[ok]
    r = float(np.corrcoef(t, e)[0, 1]) if t.size >= 3 and t.std() > 0 and e.std() > 0 else None
    return {"r": r, "rmse": float(np.sqrt(np.mean((e - t) ** 2))) if t.size else None,
            "bias": float(np.mean(e - t)) if t.size else None, "n": int(t.size)}


def iter_check(adata, *, celltype_column: str | None = None, donor_column: str | None = None, method: str = "auto",
               n_mixtures: int = N_MIXTURES, seed: int = 0, dataset_id: str = "",
               stop: Callable[[], bool] | None = None, cells_column: str | None = None,
               cells_value: str | None = None) -> Iterator[tuple[str, dict[str, Any] | CheckResult]]:
    """Yields ("progress", {...}) per held-out donor, then ("result", CheckResult)."""
    from scview.core.pseudobulk import raw_counts

    t0 = time.perf_counter()
    if not 1 <= n_mixtures <= MAX_MIXTURES:
        raise DeconvolutionError(f"Mixtures per donor: 1 to {MAX_MIXTURES}.")
    if method not in METHODS:
        raise DeconvolutionError(f"Unknown method '{method}' (expected one of {', '.join(METHODS)}).")
    ref = build_reference(adata, celltype_column, donor_column, cells_column, cells_value)
    if len(ref.donors) < MIN_DONORS_CHECK:
        raise DeconvolutionError(
            f"The self-check holds out one donor at a time, so it needs {MIN_DONORS_CHECK}+ donors "
            f"({len(ref.donors)} {'in ' + ref.donor_column if ref.donor_column else 'found: pick a donor column'}).")
    full = signature(ref)
    if method == "auto":
        methods = ["weighted", "nnls"] if full.weighted_ok else ["nnls"]
    else:
        choose_method(method, full)
        methods = [method]
    markers, _ = marker_genes(adata, ref, full)
    M, _, _ = raw_counts(adata)
    U = np.flatnonzero(ref.usable)
    rng = np.random.default_rng(seed)
    points: list[tuple] = []
    notes: list[str] = []
    yield "progress", {"done": 0, "total": len(ref.donors), "donor": None,
                       "estimate_seconds": round(check_estimate_seconds(len(ref.donors), n_mixtures), 1)}
    for d, dname in enumerate(ref.donors):
        if stop is not None and stop():
            return
        try:
            sig = signature(ref, exclude_donor=d)
        except DeconvolutionError as exc:
            notes.append(f"{dname}: skipped ({exc})")
            continue
        own_types = [k for k in range(len(ref.types))
                     if ((ref.cell_donor == d) & (ref.cell_type == k)).sum() >= MIN_DONOR_CELLS]
        mix_types = [k for k in own_types if k in set(sig.type_index.tolist())]
        if len(mix_types) < 2:
            notes.append(f"{dname}: fewer than 2 of its cell types are in the rest of the reference; skipped.")
            continue
        # The donor's pseudobulk (its cells of the mixed types) and the mixtures of its cells.
        cells_of = {k: np.flatnonzero((ref.cell_donor == d) & (ref.cell_type == k)) for k in mix_types}
        rows_W, truths = [], []
        n_true = np.array([cells_of[k].size for k in mix_types], dtype=np.float64)
        rows_W.append(np.concatenate([cells_of[k] for k in mix_types]))
        truths.append(n_true / n_true.sum())
        for _ in range(n_mixtures):
            p = rng.dirichlet(np.ones(len(mix_types)))
            n = int(rng.integers(MIX_CELLS[0], MIX_CELLS[1] + 1))
            nk = rng.multinomial(n, p)
            rows_W.append(np.concatenate([rng.choice(cells_of[k], size=c, replace=True)
                                          for k, c in zip(mix_types, nk, strict=True) if c]))
            truths.append(nk / n)
        W = sp.csr_matrix((np.ones(sum(r.size for r in rows_W)),
                           (np.repeat(np.arange(len(rows_W)), [r.size for r in rows_W]), np.concatenate(rows_W))),
                          shape=(len(rows_W), adata.n_obs))
        B = weighted_sums(M, W)[:, U].T  # usable genes x mixtures
        type_pos = {int(k): i for i, k in enumerate(sig.type_index)}
        for meth in methods:
            if meth == "weighted" and not sig.weighted_ok:
                notes.append(f"{dname} held out: {sig.weighted_why}, so the weighted method was not checked.")
                continue
            fits = deconvolve(B, sig, meth, markers)
            for i, f in enumerate(fits):
                truth_full = np.zeros(len(sig.types))
                for j, k in enumerate(mix_types):
                    truth_full[type_pos[k]] = truths[i][j]
                for kk, t in enumerate(sig.types):
                    points.append((meth, dname, i - 1, t, float(truth_full[kk]), float(f.p[kk])))
        yield "progress", {"done": d + 1, "total": len(ref.donors), "donor": dname}
    if not points:
        raise DeconvolutionError("No donor could be held out: " + " ".join(notes))
    pts = pd.DataFrame(points, columns=["method", "donor", "mixture", "cell_type", "truth", "estimate"])
    overall, per_type = [], []
    for meth, g in pts.groupby("method", sort=False):
        overall.append({"method": meth, **_metrics(g["truth"].to_numpy(), g["estimate"].to_numpy())})
        for t, gt in g.groupby("cell_type", sort=False):
            per_type.append({"method": meth, "cell_type": t,
                             **_metrics(gt["truth"].to_numpy(), gt["estimate"].to_numpy())})
    notes.append("Mixtures are drawn from this dataset's own cells, so this is optimistic: real bulk samples add "
                 "platform and protocol differences, and a condition or tissue the reference does not match is not "
                 "tested here.")
    yield "result", CheckResult(
        id=uuid.uuid4().hex[:12], dataset_id=dataset_id, celltype_column=ref.celltype_column,
        donor_column=str(ref.donor_column), methods=[m for m in methods if m in set(pts["method"])],
        n_mixtures=n_mixtures, seed=seed, donors=ref.donors, types=full.types, overall=overall,
        per_type=pd.DataFrame(per_type, columns=["method", "cell_type", "r", "rmse", "bias", "n"]), points=pts,
        notes=notes, elapsed_seconds=round(time.perf_counter() - t0, 3), created_at=time.time(),
        cells=results_store.cells_hash(adata), reference_cells=ref.cells_filter,
    )


def check(adata, **kw) -> CheckResult:
    """``iter_check`` to the end."""
    out = None
    for event, data in iter_check(adata, **kw):
        if event == "result":
            out = data
    assert isinstance(out, CheckResult)
    return out


# ---------------------------------------------------------------------------
# The reference export
# ---------------------------------------------------------------------------


class _Sink:
    """A write-only stream the zip writes into; ``take`` hands back what it holds."""

    def __init__(self) -> None:
        self._parts: list[bytes] = []
        self._size = 0

    def write(self, b) -> int:
        b = bytes(b)
        self._parts.append(b)
        self._size += len(b)
        return len(b)

    def flush(self) -> None:
        pass

    def take(self) -> bytes:
        out = b"".join(self._parts)
        self._parts.clear()
        return out


_LABEL_BAD = re.compile(r"[^A-Za-z0-9_+\-]+")


def cibersortx_label(name: str) -> str:
    """A cell-type label CIBERSORTx accepts: no periods, spaces or tabs."""
    return _LABEL_BAD.sub("_", str(name)).strip("_") or "type"


def _stored(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=time.localtime()[:6])
    info.compress_type = zipfile.ZIP_STORED
    return info


def _tsv(df: pd.DataFrame, index: bool = True) -> bytes:
    return df.to_csv(sep="\t", index=index, lineterminator="\n").encode()


README = """scView deconvolution reference
==============================

From dataset {dataset} ({n_cells:,} cells{subset}, {n_types} cell types in '{celltype}', donors in '{donor}').
Raw counts ({source}), linear scale. Cell types: {types}.

matrix.mtx.gz, features.tsv.gz, barcodes.tsv.gz
    Genes x cells raw counts in the 10x Genomics layout: scanpy.read_10x_mtx(folder), Seurat
    Read10X(folder). With cells.tsv this is the reference for MuSiC (SingleCellExperiment with
    colData cellType + sampleID), BayesPrism (cells x genes raw counts + cell types), SCDC or DWLS.
cells.tsv
    cell_id, cell_type, cell_state (the clustering when it differs from the cell type), donor.
cibersortx_reference.txt
    CIBERSORTx "single cell reference matrix": tab-separated, genes in rows, row 1 = each cell's
    cell-type label (periods and spaces replaced), non-log counts; {n_cx:,} cells sampled by type
    (at most {cx_cap:,}, types with {cx_min}+ cells), seed {seed}. Upload it in "Create Signature Matrix".
signature_cpm.tsv
    Genes x cell types: counts per million of each type's pooled counts (a signature for NNLS,
    CIBERSORT-style tools, or a check by eye).
pseudobulk_counts.tsv, pseudobulk_meta.tsv
    Genes x (donor | cell type) summed raw counts, and each column's donor, cell type and cells.
cell_size.tsv
    Mean counts per cell of each type (MuSiC's S_k): divides RNA fractions into cell fractions.
"""


def iter_reference_zip(adata, *, celltype_column: str | None = None, donor_column: str | None = None,
                       dataset_id: str = "", seed: int = 0, block_cells: int = 5000,
                       cells_column: str | None = None, cells_value: str | None = None) -> Iterator[bytes]:
    """The reference as a zip, streamed in pieces (bounded memory for any number of cells)."""
    from scview.core.de_composition import default_column
    from scview.core.pseudobulk import raw_counts

    ref = build_reference(adata, celltype_column, donor_column, cells_column, cells_value)
    M, _, _ = raw_counts(adata)
    sink = _Sink()
    zf = zipfile.ZipFile(sink, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True)
    cells = np.flatnonzero(ref.cell_type >= 0)
    n_genes = ref.genes.size

    def blocks() -> Iterator[tuple[np.ndarray, sp.csr_matrix]]:
        """(cell positions, their counts) in order, ``block_cells`` at a time."""
        if hasattr(M, "row_blocks"):
            for start, block in M.row_blocks():
                stop = start + block.shape[0]
                sel = cells[(cells >= start) & (cells < stop)]
                if sel.size:
                    yield sel, sp.csr_matrix(block)[sel - start]
            return
        for i in range(0, cells.size, block_cells):
            sel = cells[i:i + block_cells]
            yield sel, sp.csr_matrix(M[sel])

    # matrix.mtx.gz: a first pass for the header (nonzeros, integer or real).
    nnz, integer = 0, True
    for _, X in blocks():
        X.eliminate_zeros()
        nnz += X.nnz
        integer = integer and bool(np.all(np.mod(X.data, 1) == 0))
    # Already gzipped (the 10x layout), so stored, not deflated again.
    with zf.open(_stored("matrix.mtx.gz"), "w", force_zip64=True) as raw_fh:
        with gzip.GzipFile(fileobj=raw_fh, mode="wb", compresslevel=4, mtime=0) as gz:
            gz.write((f"%%MatrixMarket matrix coordinate {'integer' if integer else 'real'} general\n"
                      f"{n_genes} {cells.size} {nnz}\n").encode())
            col0 = 0
            for sel, X in blocks():
                X.eliminate_zeros()
                coo = X.tocoo()
                frame = pd.DataFrame({"g": coo.col + 1, "c": coo.row + 1 + col0,
                                      "v": coo.data.astype(np.int64) if integer else coo.data})
                gz.write(frame.to_csv(sep=" ", header=False, index=False, lineterminator="\n").encode())
                col0 += sel.size
                yield sink.take()
    ids = ref.gene_ids if ref.gene_ids is not None else ref.genes
    feats = pd.DataFrame({"id": ids, "symbol": ref.genes, "type": "Gene Expression"})
    zf.writestr(_stored("features.tsv.gz"), gzip.compress(feats.to_csv(sep="\t", header=False, index=False,
                                                               lineterminator="\n").encode(), mtime=0))
    names = adata.obs_names.to_numpy()[cells].astype(str)
    zf.writestr(_stored("barcodes.tsv.gz"), gzip.compress(("\n".join(names) + "\n").encode(), mtime=0))
    state_col = None
    active = default_column(adata) if ref.celltype_column else None
    clustering = adata.uns.get("scview_active_clustering")
    if isinstance(clustering, str) and clustering in adata.obs.columns and clustering != ref.celltype_column:
        state_col = clustering
    elif active and active != ref.celltype_column:
        state_col = active
    types_arr = np.asarray(ref.types, dtype=object)
    cell_meta = pd.DataFrame({
        "cell_id": names, "cell_type": types_arr[ref.cell_type[cells]],
        "cell_state": (adata.obs[state_col].astype(str).to_numpy()[cells] if state_col
                       else types_arr[ref.cell_type[cells]]),
        "donor": (np.asarray(ref.donors, dtype=object)[ref.cell_donor[cells]] if ref.donors else ""),
    })
    zf.writestr("cells.tsv", _tsv(cell_meta, index=False))
    yield sink.take()

    # CIBERSORTx: up to CIBERSORTX_CELLS cells, by type in proportion, each type with enough cells.
    rng = np.random.default_rng(seed)
    per_type = {k: cells[ref.cell_type[cells] == k] for k in range(len(ref.types))}
    per_type = {k: v for k, v in per_type.items() if v.size >= CIBERSORTX_MIN_CELLS}
    total = sum(v.size for v in per_type.values())
    pick: list[np.ndarray] = []
    for k, v in per_type.items():
        n = v.size if total <= CIBERSORTX_CELLS else max(CIBERSORTX_MIN_CELLS, int(round(CIBERSORTX_CELLS * v.size / total)))
        pick.append(np.sort(rng.choice(v, size=min(n, v.size), replace=False)))
    chosen = np.sort(np.concatenate(pick)) if pick else np.array([], dtype=np.int64)
    if chosen.size > CIBERSORTX_CELLS:
        chosen = np.sort(rng.choice(chosen, size=CIBERSORTX_CELLS, replace=False))
    W = sp.csr_matrix((np.ones(chosen.size), (np.arange(chosen.size), chosen)), shape=(chosen.size, adata.n_obs))
    if hasattr(M, "row_blocks"):
        sub = sp.csr_matrix(weighted_sums(M, W))
    else:
        sub = sp.csr_matrix(M[chosen])
    G = sp.csc_matrix(sub.T)  # genes x cells
    expressed = np.flatnonzero(np.asarray((G != 0).sum(axis=1)).ravel())  # genes seen in the sampled cells
    labels = [cibersortx_label(ref.types[ref.cell_type[c]]) for c in chosen]
    with zf.open("cibersortx_reference.txt", "w", force_zip64=True) as fh:
        fh.write(("GeneSymbol\t" + "\t".join(labels) + "\n").encode())
        Gr = sp.csr_matrix(G)
        for i in range(0, expressed.size, 1000):
            rows = expressed[i:i + 1000]
            dense = Gr[rows].toarray()
            frame = pd.DataFrame(dense.astype(np.int64) if integer else dense, index=ref.genes[rows])
            fh.write(frame.to_csv(sep="\t", header=False, lineterminator="\n").encode())
            yield sink.take()

    # Signature, pseudobulk, cell sizes.
    pooled = np.zeros((len(ref.types), n_genes))
    np.add.at(pooled, ref.group_type, ref.P)
    lib = pooled.sum(axis=1, keepdims=True)
    cpm = pd.DataFrame((pooled / np.where(lib > 0, lib, 1) * 1e6).T, index=ref.genes, columns=ref.types)
    zf.writestr("signature_cpm.tsv", cpm.round(3).to_csv(sep="\t", index_label="gene", lineterminator="\n"))
    gid = [f"{ref.donors[d]}|{ref.types[t]}" if ref.donors else ref.types[t]
           for d, t in zip(ref.group_donor, ref.group_type, strict=True)]
    pb = pd.DataFrame(ref.P.T.astype(np.int64) if integer else ref.P.T, index=ref.genes, columns=gid)
    zf.writestr("pseudobulk_counts.tsv", pb.to_csv(sep="\t", index_label="gene", lineterminator="\n"))
    meta = pd.DataFrame({"column": gid, "donor": [ref.donors[d] if ref.donors else "" for d in ref.group_donor],
                         "cell_type": [ref.types[t] for t in ref.group_type], "n_cells": ref.group_cells})
    zf.writestr("pseudobulk_meta.tsv", _tsv(meta, index=False))
    sig = signature(ref)
    size = pd.DataFrame({"cell_type": sig.types, "mean_counts_per_cell": np.round(sig.S, 3),
                         "n_cells": [int(ref.group_cells[ref.group_type == k].sum()) for k in sig.type_index],
                         "n_donors": sig.n_donors})
    zf.writestr("cell_size.tsv", _tsv(size, index=False))
    zf.writestr("README.txt", README.format(
        dataset=dataset_id or "(this dataset)", n_cells=int(cells.size),
        subset=(f" with {ref.cells_filter['column']} = {ref.cells_filter['value']}" if ref.cells_filter else ""), n_types=len(ref.types),
        celltype=ref.celltype_column, donor=ref.donor_column or "(none)", source=ref.counts_source,
        types=", ".join(ref.types), n_cx=int(chosen.size), cx_cap=CIBERSORTX_CELLS, cx_min=CIBERSORTX_MIN_CELLS,
        seed=seed))
    zf.close()
    yield sink.take()


# ---------------------------------------------------------------------------
# Kept results
# ---------------------------------------------------------------------------

_last: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_last_check: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_lock = threading.Lock()


def remember(adaptor, r: Result) -> None:
    try:
        with _lock:
            _last[adaptor] = r
    except TypeError:  # a test double
        pass
    results_store.keep(adaptor, KIND, r, result_id=r.id, key="",
                       label=f"Bulk deconvolution of {r.filename} ({len(r.samples)} samples, {r.method})")


def remember_check(adaptor, c: CheckResult) -> None:
    try:
        with _lock:
            _last_check[adaptor] = c
    except TypeError:
        pass
    results_store.keep(adaptor, CHECK_KIND, c, result_id=c.id, key="",
                       label=f"Deconvolution self-check ({len(c.donors)} donors held out in turn)")


def _last_of(adaptor, cache, kind: str):
    try:
        cells = results_store.cells_hash(adaptor.adata)
        r = results_store.remembered(cache, _lock, adaptor, kind)
        if r is None:
            r = results_store.latest(adaptor, kind, current_only=False)
            if r is not None and r.cells == cells:
                with _lock:
                    cache.setdefault(adaptor, r)
    except Exception:  # a test double, an unreadable result
        return None
    return r if r is not None and r.cells == cells else None


def last(adaptor) -> Result | None:
    """The last run, while the dataset holds the same cells."""
    return _last_of(adaptor, _last, KIND)


def last_check(adaptor) -> CheckResult | None:
    return _last_of(adaptor, _last_check, CHECK_KIND)


def grounding_lines(adaptor) -> tuple[list[str], str | None]:
    """For the co-pilot: the last run's largest cell fraction per bulk sample."""
    r = last(adaptor)
    if r is None or r.fractions.empty:
        return [], None
    detail = (f"Bulk deconvolution of {r.filename}: {len(r.samples)} samples, {METHOD_NAMES[r.method]}, "
              f"reference {r.celltype_column} ({len(r.types)} types, {r.n_reference_cells:,} cells"
              + (f" with {r.reference_cells['column']} = {r.reference_cells['value']}" if r.reference_cells else "")
              + (f", {r.n_reference_donors} donors" if r.n_reference_donors else "") + ")")
    lines = [f"\n## Bulk deconvolution (last run) [{TAG}]", f"- {detail} [{TAG}]."]
    f = r.fractions.dropna(subset=["cell_fraction"])
    tops = []
    for s in r.samples[:20]:
        g = f[f["sample"] == s]
        if len(g):
            top = g.loc[g["cell_fraction"].idxmax()]
            tops.append(f"{s}: {top['cell_type']} {top['cell_fraction']:.0%}")
    if tops:
        lines.append("- Largest estimated cell fraction per sample: " + "; ".join(tops)
                     + (f" (first 20 of {len(r.samples)})" if len(r.samples) > 20 else "") + ".")
    if r.reference_cells:
        lines.append(f"- Reference cells: only those with {r.reference_cells['column']} = "
                     f"{r.reference_cells['value']} ({r.reference_cells['n_cells']:,} cells).")
    flagged = absorber_warnings(r.flags, len(r.samples), r.reference_cells is not None)
    if flagged:
        lines.append("- Doubtful estimates: " + " ".join(flagged))
    if r.input_kind == "tpm":
        lines.append("- The bulk file was TPM, so these are relative values, nearer RNA fractions.")
    lines.append("- Estimates, not counts: a cell type missing from this dataset cannot be found, and the mixing "
                 "assumes the bulk was made like this dataset.")
    return lines, detail


def estimate_seconds_note(n_donors: int) -> str:
    s = check_estimate_seconds(n_donors)
    return f"about {math.ceil(s / 5) * 5:.0f} s" if s < 90 else f"about {math.ceil(s / 60):.0f} min"
