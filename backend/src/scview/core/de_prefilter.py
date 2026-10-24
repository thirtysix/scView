"""Which genes a cell-level DE test runs on: those detected in enough cells.

A gene seen in a handful of cells on both sides has no power to come out
significant, but it still counts against the Benjamini-Hochberg correction, and
its fold change, a ratio of two near-zero means, lands at the edges of the
volcano. So, as Seurat's FindMarkers does, the Wilcoxon test runs only on genes
detected (> 0) in at least ``min_pct`` of the cells of either group. Filtering
lowly expressed genes before testing mattered in a benchmark of 36 DE methods
(Soneson & Robinson 2018, PMID 29481549).

The default is Seurat v5's 0.01 (v4 used 0.1). On Kang's PBMCs (~1,700 counts
a cell), CD14 Mono vs CD4 Naive T, 0.01 leaves out the 6,016 of 14,053 genes
seen in under 1% of the cells of both groups, and with them 1,586 of the 1,708
genes whose log2 fold change, a ratio against a near-zero mean, is beyond 10;
more genes pass the correction than unfiltered (2,305 vs 2,161). 0.1 would test
only 2,207 genes, about 950 fewer passing, and GSEA and TF activity rank every
tested gene of a stored result (core/de_results.py), so they would see a sixth
of the transcriptome. On deeper data 0.1 costs less (paul15: 2,570 of 3,451
tested); it is one click away.

Pseudobulk DE keeps its own filter on summed counts (core/pseudobulk.py).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

DEFAULT_MIN_PCT = 0.01
# Detection histogram resolution: counts[k] holds the genes detected in
# [k, k + 1) / HIST_BINS of a group's cells, so a client can tell exactly how
# many genes any cut-off on this grid (1%, 5%, 10%, 25%) would keep.
HIST_BINS = 10_000


@dataclass
class Prefilter:
    keep: np.ndarray  # bool per gene: tested
    histogram: list[int]  # HIST_BINS + 1 counts (see HIST_BINS)
    min_pct: float | None  # the cut-off applied; None when none could be

    @property
    def n_tested(self) -> int:
        return int(self.keep.sum())

    @property
    def n_skipped(self) -> int:
        return int(self.keep.size - self.keep.sum())


# Stored values per pass when counting a CSR matrix's detections.
_CHUNK = 4_000_000


def _n_detected(M, rows: np.ndarray) -> np.ndarray:
    """Per gene, how many of the ``rows`` cells have a value above zero.

    A CSR matrix is counted a chunk of rows at a time, in place: copying the rest of the
    cells out (and a boolean matrix of them) took +3.0 GiB on 100k neurons in memory,
    enough to get the backend killed at 8 GB."""
    if sp.issparse(M) and M.format == "csr":
        rows = np.asarray(rows, dtype=bool)
        indptr = np.asarray(M.indptr, dtype=np.int64)
        counts = np.zeros(M.shape[1], dtype=np.int64)
        r0, n = 0, M.shape[0]
        while r0 < n:
            r1 = int(np.searchsorted(indptr, indptr[r0] + _CHUNK, side="right")) - 1
            r1 = min(max(r1, r0 + 1), n)
            p0, p1 = indptr[r0], indptr[r1]
            hit = np.repeat(rows[r0:r1], np.diff(indptr[r0:r1 + 1])) & (M.data[p0:p1] > 0)
            counts += np.bincount(M.indices[p0:p1][hit], minlength=M.shape[1])
            r0 = r1
        return counts
    sub = M[rows]
    if sp.issparse(sub):
        return np.asarray((sub > 0).sum(axis=0)).ravel().astype(np.int64)
    return np.count_nonzero(np.asarray(sub) > 0, axis=0).astype(np.int64)


def _has_negatives(M) -> bool:
    data = M.data if sp.issparse(M) else np.asarray(M)
    return bool(data.size) and bool(np.nanmin(data) < 0)


def prefilter(M, in_a: np.ndarray, min_pct: float) -> Prefilter:
    """The genes of ``M`` (cells x genes, log-normalised) detected in at least
    ``min_pct`` of the ``in_a`` cells or of the other cells, and the histogram
    of each gene's larger detection fraction.

    Scaled values (z-scores, residuals) have no zero for "not detected", so they
    are tested whole and no histogram is made."""
    n_genes = M.shape[1]
    if _has_negatives(M):
        return Prefilter(np.ones(n_genes, dtype=bool), [], None)
    in_a = np.asarray(in_a, dtype=bool)
    n_a, n_b = int(in_a.sum()), int((~in_a).sum())
    det_a, det_b = _n_detected(M, in_a), _n_detected(M, ~in_a)
    # The larger of the two fractions, floored onto the histogram grid in
    # integers (no float rounding can move a gene across a grid cut-off).
    k = np.maximum(det_a * HIST_BINS // max(n_a, 1), det_b * HIST_BINS // max(n_b, 1))
    histogram = np.bincount(k, minlength=HIST_BINS + 1).tolist()
    if min_pct <= 0:
        return Prefilter(np.ones(n_genes, dtype=bool), histogram, 0.0)
    keep = (det_a / max(n_a, 1) >= min_pct) | (det_b / max(n_b, 1) >= min_pct)
    return Prefilter(keep, histogram, float(min_pct))
