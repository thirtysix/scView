"""Peak-gene links: a peak whose accessibility tracks a nearby gene's expression.

Method: Signac's ``LinkPeaks`` (MIT; Ma et al. 2020, SHARE-seq) with its defaults. Peak counts
against log-normalised RNA, Pearson, for peaks and genes in more than 10 cells and peaks within
500 kb of the gene's TSS; a gene with fewer than 2 such peaks is skipped. A pair with |r| > 0.05
is compared with 200 peaks from other chromosomes drawn by ``MatchRegionStats``: weights from GC
percent, the peak's total count and its width, each a Gaussian density of the query (bandwidth 1)
over the density of all those peaks, drawn without replacement. z = (r - background mean) /
background sd, p = pnorm(-|z|), kept at p < 0.05. Added here: BH across every tested pair.

Coordinates follow Signac's parse of a 10x name: ``chr1:9790-10675`` is read as 1-based inclusive,
so the bases are [9789, 10675) in 0-based half open, the width 886. The file's BED would say
[9790, 10675), 885 bases; it is Signac's numbers this has to match.

Two things make this hours faster than R, neither changing what a pair's z is drawn from:

- The peak matrix is turned once into a by-peak copy, memory-mapped (8 B a value), so a gene's
  correlations touch only its candidate and background peaks; each peak's sums are computed once.
  File-backed: the system can drop the pages, and only a slice is ever held. It costs 2 passes of
  the store and a few seconds, which a by-cell product would pay on every gene.
- Each peak draws its background once, from a generator seeded by the run's seed and the peak's
  index. A tested peak is always on its gene's chromosome, so the set it is drawn from is the same
  for every gene near it; Signac draws again per pair from the identical distribution. Two genes
  sharing a peak therefore share a background; each pair's z has the distribution it has in Signac,
  and a gene's z does not depend on which other genes are in the run.

A link is a correlation across the cells used. Across all cells it mostly reflects which cell types
open the peak and express the gene; it is a candidate for regulation, never proof.

Checked against Signac 1.17.1 LinkPeaks on the same matrices. 10x PBMC 10k multiome, 617 genes, 2 Signac seeds: GC percent
of all 143,836 peaks within 5e-11, width equal; MatchRegionStats' weights within 3e-10 (relative) on 12 query peaks; the same
5,306 tested pairs, score within 9e-15. z here vs Signac r = 0.991, median |difference| 0.072; Signac's seeds against each other
0.992, 0.070. Kept links overlap (Jaccard) 0.943 and 0.947; Signac's seeds 0.943. Mouse E18, 180 genes: the same 1,479 pairs, score
within 2e-15, z r = 0.984-0.986 (seeds 0.987), kept 0.939-0.944 (seeds 0.942). Signac took 573 s on the 617 genes at 1 CPU, here 39 s at 2;
variable genes 52 s, all 26,200 genes 171 s, peak cgroup anon 0.77 GiB (under a 2 GB limit: 68 s, no OOM). The replay script
from the original upload reproduces score, z and p bitwise on all 31,951 links.
"""

from __future__ import annotations

import hashlib
import json
import logging
import queue
import re
import shutil
import tempfile
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.special import ndtr

from scview.core import atac, gene_annotation, genome_seq, results_store
from scview.core.atac import MATRIX, AtacCancelled, AtacError, _bh
from scview.core.genome_build import standard_chromosomes

logger = logging.getLogger(__name__)

TAG = "result:atac_links"
STEP = "atac_links"
RESULT_KIND = "atac_links"
MAX_LIST = 5_000  # genes in a typed list
MIN_RUN_CELLS = 50

# Signac's LinkPeaks and MatchRegionStats defaults.
DISTANCE = 500_000
MIN_CELLS = 10
SCORE_CUTOFF = 0.05
N_SAMPLE = 200
P_CUTOFF = 0.05
SEED = 0
OUTSIDE = 1e-6  # approx()'s yleft and yright: a candidate beyond the query's +-3 bandwidths
DENSITY_N = 512
LINK_BLOCK_VALUES = 4_000_000  # stored values per block of the store
PRODUCT_VALUES = 4_000_000  # stored values in one slice of peaks: about 64 MB with its copy
RNA_BATCH_BYTES = 64 * 1024**2  # one batch of genes' dense values
MAX_RNA_BATCH = 256
GC_BATCH = 20_000  # peaks whose bases are held at once

MODES = ("variable", "all", "list")

CAPTION = ("A link is a correlation between a peak's counts and a nearby gene's expression across these cells: a "
           "candidate regulatory element, never proof of regulation.")
CAPTION_ALL_CELLS = (" Across all cells it mostly reflects which cell types open the peak and express the gene: "
                     "restrict the cells to one type to ask about variation within it.")
METHOD = ("Signac's LinkPeaks: Pearson r of peak counts with log-normalised RNA, peaks within {distance:,} bp of the TSS, "
          "peaks and genes in more than {min_cells} cells, |r| > {score_cutoff} tested against {n_sample} peaks from other "
          "chromosomes matched on GC percent, count and width (MatchRegionStats' weights, seed {seed}), one-sided p from z, "
          "kept at p < {p_cutoff}; BH across all tested pairs added.")


# ---------------------------------------------------------------------------
# MatchRegionStats' weights
# ---------------------------------------------------------------------------


def r_density(values):
    """R's ``density(values, bw = 1)`` on its 512 points (stats 4.6, gaussian): (x, y). Linear
    binning on a 512-point grid, a circular convolution of twice that length, then interpolation.
    With bandwidth 1 and a range of thousands the grid is far coarser than the kernel, so the result
    is spiky rather than smooth: that is what Signac's weights are made of, and it is kept."""
    n = DENSITY_N
    x = np.asarray(values, dtype=np.float64)
    start, stop = x.min() - 3.0, x.max() + 3.0
    lo, up = start - 4.0, stop + 4.0
    pos = (x - lo) / ((up - lo) / (n - 1))
    ix = np.floor(pos).astype(np.int64)
    fx = pos - ix
    each = 1.0 / x.size
    y = (np.bincount(ix, (1.0 - fx) * each, minlength=2 * n) + np.bincount(ix + 1, fx * each, minlength=2 * n))[:2 * n]
    kords = np.linspace(0.0, (2 * n - 1) / (n - 1) * (up - lo), 2 * n)
    kords[n + 1:] = -kords[n - 1:0:-1]
    kernel = np.exp(-0.5 * kords * kords) / np.sqrt(2.0 * np.pi)
    smooth = np.maximum(0.0, np.fft.ifft(np.fft.fft(y) * np.conj(np.fft.fft(kernel))).real[:n])
    grid = np.linspace(start, stop, n)
    return grid, np.interp(grid, np.linspace(lo, up, n), smooth)


class BackgroundSampler:
    """Draws one peak's background from the peaks on other chromosomes. ``features``: kept peaks x
    (GC percent, count, width); ``others``: positions of the kept peaks to draw from."""

    def __init__(self, features, others):
        self.others = others
        self.values = features[others]
        self.query = features
        n = others.size
        base = np.full(n, OUTSIDE ** features.shape[1], dtype=np.float64)
        self.order, self.sorted = [], []
        for f in range(features.shape[1]):
            v = self.values[:, f]
            if n:
                m = np.interp(v, *r_density(v), left=OUTSIDE, right=OUTSIDE)
                with np.errstate(divide="ignore"):
                    inverse = 1.0 / m
                inverse[~np.isfinite(inverse)] = 0.0  # weights[!is.finite(weights)] <- 0
                base *= inverse
            order = np.argsort(v, kind="stable")
            self.order.append(order)
            self.sorted.append(v[order])
        self.base = base
        # The query's own density is one Gaussian on its +-3 grid: the same shape for every value.
        self.shape_x, self.shape_y = r_density(np.zeros(1))

    def weights(self, peak):
        """MatchRegionStats' product of (query density / all peaks' density) for one kept peak."""
        w = self.base.copy()
        for f, order in enumerate(self.order):
            q = self.query[peak, f]
            a = np.searchsorted(self.sorted[f], q - 3.0, side="left")
            b = np.searchsorted(self.sorted[f], q + 3.0, side="right")
            near = order[a:b]
            w[near] *= np.interp(self.values[near, f] - q, self.shape_x, self.shape_y) / OUTSIDE
        return w

    def draw(self, peak, store_peak, seed=SEED, n_sample=N_SAMPLE):
        """Kept-peak positions of ``n_sample`` draws without replacement, as sample.int(prob=):
        exponential clocks, the smallest ``n_sample`` of E / w. From a generator of its own, so a
        peak's background does not depend on what else ran."""
        w = self.weights(peak)
        k = min(n_sample, int(np.count_nonzero(w)))
        if k == 0:
            return np.zeros(0, dtype=np.int32)
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, int(store_peak)])))
        with np.errstate(divide="ignore"):
            clocks = rng.standard_exponential(w.size) / w
        pick = np.argpartition(clocks, k - 1)[:k] if k < w.size else np.arange(w.size)
        return self.others[np.sort(pick)].astype(np.int32)


# ---------------------------------------------------------------------------
# The peak matrix by peak
# ---------------------------------------------------------------------------


class ByPeak:
    """The kept peaks' counts by peak, in two memory-mapped files, with each peak's sums."""

    def __init__(self, store, rows, known, folder, *, min_cells=MIN_CELLS, stop=lambda: False, say=lambda _d, _t: None):
        n_vars = store.n_vars
        n_cells = rows.size
        sums = np.zeros(n_vars, dtype=np.float64)
        squares = np.zeros(n_vars, dtype=np.float64)
        stored = np.zeros(n_vars, dtype=np.int64)
        detected = np.zeros(n_vars, dtype=np.int64)
        done = 0
        for pos, block in store.iter_blocks(MATRIX, rows, LINK_BLOCK_VALUES):
            if stop():
                raise AtacCancelled()
            values = block.data.astype(np.float64)
            sums += np.bincount(block.indices, weights=values, minlength=n_vars)
            squares += np.bincount(block.indices, weights=values * values, minlength=n_vars)
            stored += np.bincount(block.indices, minlength=n_vars)
            detected += np.bincount(block.indices[values > 0], minlength=n_vars)
            done += pos.size
            say(done, 2 * n_cells)
        self.n_cells = n_cells
        self.kept = np.flatnonzero((detected > min_cells) & known)
        self.sums, self.squares = sums[self.kept], squares[self.kept]
        self.counts = sums[self.kept].copy()  # FindTopFeatures' count: the peak's total over these cells
        where = np.full(n_vars, -1, dtype=np.int64)
        where[self.kept] = np.arange(self.kept.size)
        nnz = int(stored[self.kept].sum())
        kind = np.int32 if nnz < 2**31 else np.int64  # scipy would copy indices of another type than the pointers
        self.folder = Path(folder)
        self._data = np.memmap(self.folder / "data.f4", dtype=np.float32, mode="w+", shape=(max(1, nnz),))
        self._indices = np.memmap(self.folder / "indices.i", dtype=kind, mode="w+", shape=(max(1, nnz),))
        ptr = np.zeros(self.kept.size + 1, dtype=kind)
        ptr[1:] = np.cumsum(stored[self.kept])
        cursor = ptr[:-1].astype(np.int64)
        for pos, block in store.iter_blocks(MATRIX, rows, LINK_BLOCK_VALUES):
            if stop():
                raise AtacCancelled()
            mapped = where[block.indices]
            keep = mapped >= 0
            lengths = np.bincount(np.repeat(np.arange(pos.size), np.diff(block.indptr))[keep], minlength=pos.size)
            local_ptr = np.zeros(pos.size + 1, dtype=np.int64)
            np.cumsum(lengths, out=local_ptr[1:])
            by_peak = sp.csr_matrix((block.data[keep], mapped[keep], local_ptr), shape=(pos.size, self.kept.size)).tocsc()
            per_peak = np.diff(by_peak.indptr)
            dest = np.repeat(cursor, per_peak) + (np.arange(by_peak.nnz) - np.repeat(by_peak.indptr[:-1], per_peak))
            self._data[dest] = by_peak.data
            self._indices[dest] = pos[by_peak.indices]
            cursor += per_peak
            done += pos.size
            say(done, 2 * n_cells)
        self._matrix = sp.csc_matrix((self._data[:nnz], self._indices[:nnz], ptr), shape=(n_cells, self.kept.size), copy=False)
        self._ptr = ptr
        mean = self.sums / max(n_cells, 1)
        with np.errstate(invalid="ignore"):
            self.mean = mean
            self.sd = np.sqrt(np.maximum(self.squares - n_cells * mean * mean, 0.0) / max(n_cells - 1, 1))

    def correlate(self, columns, y, y_mean, y_sd):
        """corSparse: Pearson r of each kept peak in ``columns`` with ``y``, a slice of peaks at a
        time so a gene near thousands of background peaks holds a bounded copy."""
        out = np.zeros(columns.size, dtype=np.float64)
        sizes = np.asarray(self._ptr[columns + 1], dtype=np.int64) - np.asarray(self._ptr[columns], dtype=np.int64)
        parts = np.cumsum(sizes) // PRODUCT_VALUES
        cuts = np.flatnonzero(np.diff(parts)) + 1
        n = self.n_cells
        at = 0
        for chunk in np.split(columns, cuts):
            if chunk.size:
                cross = np.asarray(self._matrix[:, chunk].T @ y, dtype=np.float64)
                cov = (cross - n * self.mean[chunk] * y_mean) / (n - 1)
                out[at:at + chunk.size] = cov / (self.sd[chunk] * y_sd)
                at += chunk.size
        return out

    def close(self) -> None:
        self._matrix = None
        self._data = self._indices = None


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


def signac_bases(start, end):
    """(start, end) 0-based half open of a peak named ``chrom:start-end`` as Signac parses it."""
    return start - 1, end


def peak_widths(start, end):
    return end - start + 1


def peak_gc(two_bit, chrom, start, end, stop=lambda: False):
    """GC percent per peak on Signac's coordinates (NaN on a contig the genome does not name, or for
    a peak with no position): ``genome_seq.gc_percent`` over a batch of peaks at a time."""
    gc = np.full(chrom.size, np.nan, dtype=np.float64)
    valid = np.flatnonzero(start >= 0)
    lo, hi = signac_bases(start, end)
    for at in range(0, valid.size, GC_BATCH):
        if stop():
            raise AtacCancelled()
        idx = valid[at:at + GC_BATCH]
        codes, offsets = genome_seq.sequences(two_bit, chrom[idx], lo[idx], hi[idx])
        gc[idx] = genome_seq.gc_percent(genome_seq.base_counts(codes, offsets))
    return gc


def _selection_hash(adata, mask: np.ndarray | None) -> str:
    names = adata.obs_names if mask is None else adata.obs_names[mask]
    h = hashlib.sha256(f"{len(names)}\n".encode())
    for at in range(0, len(names), 100_000):
        h.update("\n".join(map(str, names[at:at + 100_000])).encode())
    return h.hexdigest()[:16]


def place_genes(names, ids, table):
    """Each named gene's chrom, TSS and strand: by gene id (``ids``, name -> id), else by name, as
    ``atac.choose_genes``; a gene with no position is left out."""
    by_id = table.drop_duplicates("gene_id").set_index("gene_id")
    by_name = table.drop_duplicates("gene_name").set_index("gene_name")
    rows = []
    for name in names:
        hit = None
        gene_id = ids.get(name, "")
        if gene_id in by_id.index:
            hit = by_id.loc[gene_id]
        elif name in by_name.index:
            hit = by_name.loc[name]
        if hit is not None:
            rows.append((name, gene_id, hit["chrom"], int(hit["tss"]), hit["strand"]))
    return pd.DataFrame(rows, columns=["name", "gene_id", "chrom", "tss", "strand"])


def order_genes(table):
    """By chromosome and TSS: a chromosome's background is prepared once, and its genes share it."""
    order = np.lexsort((table["tss"].to_numpy(), table["chrom"].to_numpy(dtype=str)))
    return table.iloc[order].reset_index(drop=True)


def signed_distance(start, end, tss, strand):
    """Bases from a peak to the TSS: 0 inside, + downstream on the gene's strand (as
    ``atac.annotate_peaks``)."""
    gap = np.maximum(0, np.maximum(start - tss, tss - (end - 1)))
    side = np.where(strand == "-", -1.0, 1.0) * np.sign((start + end) / 2 - tss)
    return gap * np.where(side == 0, 1.0, side) + 0.0  # never -0.0


def iter_links(store, rows, cell_mask, read_genes, genes, peak_names, chrom, start, end, gc, folder, *, seed=SEED,
               distance=DISTANCE, min_cells=MIN_CELLS, score_cutoff=SCORE_CUTOFF, n_sample=N_SAMPLE, p_cutoff=P_CUTOFF,
               stop=lambda: False, standard_only=True):
    """The links, as a generator of ``progress`` events and one ``result``. ``rows``: the store rows
    of the cells used; ``cell_mask``: those cells among what ``read_genes(names)`` returns (cells x
    genes, dense), or None for all; ``genes``: name, chrom, tss (0-based), strand; ``peak_names``,
    ``chrom``, ``start``, ``end``, ``gc``: per store peak; ``folder``: where the by-peak copy goes.
    ``standard_only``: peaks on chrM, unplaced scaffolds and alt contigs are no candidates."""
    standard = standard_chromosomes(chrom) if standard_only else np.ones(gc.size, dtype=bool)
    known = np.isfinite(gc)
    n_unknown = int((~known & standard).sum())
    n_nonstandard = int((~standard).sum())
    known &= standard
    yield {"type": "progress", "step": "by_peak", "message": "Turning the peak matrix by peak", "done": 0, "total": 1}
    by = ByPeak(store, rows, known, folder, min_cells=min_cells, stop=stop)
    kept = by.kept
    peak_chrom = chrom[kept]
    features = np.column_stack([gc[kept], by.counts, peak_widths(start[kept], end[kept]).astype(np.float64)])
    n = by.n_cells
    tested = {"gene": [], "peak": [], "score": [], "z": []}
    n_tested_genes = n_few_cells = n_few_peaks = n_pairs_cis = 0
    total = len(genes)
    done = 0
    batch_genes = max(1, min(MAX_RNA_BATCH, RNA_BATCH_BYTES // max(1, 4 * (cell_mask.size if cell_mask is not None else n))))
    names = genes["name"].to_numpy(dtype=object)
    tss = genes["tss"].to_numpy(dtype=np.int64)
    for c, group in genes.groupby("chrom", sort=False):
        members = group.index.to_numpy()
        here = np.flatnonzero(peak_chrom == c)
        if here.size < 2:
            n_few_peaks += members.size  # a gene with fewer than 2 candidates is never tested
            done += members.size
            continue
        here = here[np.argsort(start[kept[here]], kind="stable")]
        starts, ends = start[kept[here]], end[kept[here]]
        reach = np.maximum.accumulate(ends)
        sampler = BackgroundSampler(features, np.flatnonzero(peak_chrom != c))
        drawn: dict[int, np.ndarray] = {}
        for at in range(0, members.size, batch_genes):
            batch = members[at:at + batch_genes]
            block = read_genes(names[batch].tolist())
            if block.shape[1] != batch.size:
                raise AtacError("A gene chosen for the links is no longer in this dataset. Run it again.", 409)
            if cell_mask is not None:
                block = block[cell_mask]
            if block.shape[0] != n:
                raise AtacError("The dataset's cells changed while the links ran (a pipeline run or a filter). Run it again.", 409)
            for j, g in enumerate(batch):
                if stop():
                    raise AtacCancelled()
                y = block[:, j].astype(np.float64)
                if int((y > 0).sum()) <= min_cells:
                    n_few_cells += 1
                    continue
                tss1 = tss[g] + 1  # the first transcribed base, 1-based, as Signac's resize()
                a = int(np.searchsorted(reach, tss1 - distance, side="left"))
                b = int(np.searchsorted(starts, tss1 + distance, side="right"))
                near = here[a:b][ends[a:b] >= tss1 - distance]
                if near.size < 2:
                    n_few_peaks += 1
                    continue
                y_mean = y.mean()
                y_sd = np.sqrt((y @ y - n * y_mean * y_mean) / (n - 1))
                n_tested_genes += 1
                n_pairs_cis += near.size
                r = by.correlate(near, y, y_mean, y_sd)
                hit = np.flatnonzero(np.abs(r) > score_cutoff)
                if hit.size == 0:
                    continue
                backgrounds = []
                for p in near[hit]:
                    bg = drawn.get(int(p))
                    if bg is None:
                        bg = sampler.draw(int(p), int(kept[p]), seed, n_sample)
                        drawn[int(p)] = bg
                    backgrounds.append(bg)
                sizes = np.fromiter((b_.size for b_ in backgrounds), dtype=np.int64, count=len(backgrounds))
                z = np.full(hit.size, np.nan)
                if sizes.sum():
                    union, inverse = np.unique(np.concatenate(backgrounds), return_inverse=True)
                    r_bg = by.correlate(union, y, y_mean, y_sd)[inverse]
                    offsets = np.concatenate([[0], np.cumsum(sizes)[:-1]])
                    ok = sizes >= 2
                    # reduceat on an empty range gives the element itself; only full ones are read.
                    mean = np.add.reduceat(r_bg, np.minimum(offsets, r_bg.size - 1)) / np.maximum(sizes, 1)
                    spread = r_bg - np.repeat(mean, sizes)
                    var = np.add.reduceat(spread * spread, np.minimum(offsets, r_bg.size - 1)) / np.maximum(sizes - 1, 1)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        z[ok] = (r[hit][ok] - mean[ok]) / np.sqrt(var[ok])
                tested["gene"].append(np.full(hit.size, g, dtype=np.int32))
                tested["peak"].append(near[hit].astype(np.int32))
                tested["score"].append(r[hit])
                tested["z"].append(z)
            done += batch.size
            yield {"type": "progress", "step": "links", "done": done, "total": total,
                   "message": f"Genes: {done:,} of {total:,}"}
    gene_at = np.concatenate(tested["gene"]) if tested["gene"] else np.zeros(0, dtype=np.int32)
    peak_at = np.concatenate(tested["peak"]) if tested["peak"] else np.zeros(0, dtype=np.int32)
    score = np.concatenate(tested["score"]) if tested["score"] else np.zeros(0)
    z = np.concatenate(tested["z"]) if tested["z"] else np.zeros(0)
    usable = np.isfinite(z)
    p = np.ones(z.size)
    p[usable] = ndtr(-np.abs(z[usable]))
    padj = _bh(p) if p.size else p
    keep = np.flatnonzero(usable & (p < p_cutoff))
    store_peak = kept[peak_at[keep]]
    gene_rows = genes.iloc[gene_at[keep]] if keep.size else genes.iloc[:0]
    strand = gene_rows["strand"].to_numpy(dtype=object)
    frame = pd.DataFrame({
        "gene": gene_rows["name"].to_numpy(dtype=object), "peak": np.asarray(peak_names, dtype=object)[store_peak],
        "peak_index": store_peak.astype(np.int32), "chrom": chrom[store_peak], "start": start[store_peak], "end": end[store_peak],
        "tss": gene_rows["tss"].to_numpy(dtype=np.int64),
        "distance_to_tss": signed_distance(start[store_peak], end[store_peak], gene_rows["tss"].to_numpy(dtype=np.int64), strand),
        "score": score[keep], "z": z[keep], "p": p[keep], "padj": padj[keep]})
    by.close()
    yield {"type": "result", "frame": frame, "counts": {
        "n_genes_asked": total, "n_genes_tested": n_tested_genes, "n_genes_few_cells": n_few_cells,
        "n_genes_few_peaks": n_few_peaks, "n_pairs_cis": n_pairs_cis, "n_pairs_tested": int(z.size),
        "n_pairs_no_background": int((~usable).sum()), "n_links": int(keep.size), "n_links_padj": int((padj[keep] < 0.05).sum()),
        "n_peaks_kept": int(kept.size), "n_peaks_unknown": n_unknown, "n_peaks_nonstandard": n_nonstandard, "n_cells": int(n)}}


REPLAY_CONSTANTS = ("DISTANCE", "MIN_CELLS", "SCORE_CUTOFF", "N_SAMPLE", "P_CUTOFF", "SEED", "OUTSIDE", "DENSITY_N",
                    "LINK_BLOCK_VALUES", "PRODUCT_VALUES", "RNA_BATCH_BYTES", "MAX_RNA_BATCH", "GC_BATCH")
REPLAY_CODE = (_selection_hash, r_density, BackgroundSampler, ByPeak, signac_bases, peak_widths, peak_gc, place_genes, order_genes, signed_distance,
               iter_links)


# ---------------------------------------------------------------------------
# What a run would use
# ---------------------------------------------------------------------------


@dataclass
class LinkPlan:
    genes: pd.DataFrame  # name, gene_id, chrom, tss, strand
    mode: str
    genome: str | None
    n_cells: int
    n_dataset_cells: int
    cell_mask: np.ndarray | None
    cells_column: str | None
    cells_value: list[str] | None
    selection: str  # hash of the cells used, in order
    n_asked: int
    missing: list[str] = field(default_factory=list)  # listed names with no gene or no position


def _rna_ready(adaptor) -> None:
    """The matrix ``get_expression`` reads must hold log-normalised values, as Signac's data slot."""
    from scview.core.activity import _non_negative
    from scview.core.pipeline import looks_like_counts

    if getattr(adaptor, "is_large", False):  # a large dataset's X is log-normalised by its layout
        return
    adata = adaptor.adata
    m = adata.raw.X if adaptor.default_expression_layer() == "raw" else adata.X
    if looks_like_counts(m) or not _non_negative(m):
        raise AtacError("Peak-gene links read log-normalised RNA, and this dataset has none yet: run the pipeline's "
                        "normalisation and log transform first.", 422)


def variable_genes(adata) -> list[str]:
    var = adata.var
    if "highly_variable" not in var.columns or not var["highly_variable"].astype(bool).any():
        raise AtacError("This dataset has no variable genes yet: run the pipeline's variable genes step, or choose all "
                        "genes or a list.", 422)
    return var.index[var["highly_variable"].astype(bool).to_numpy()].astype(str).tolist()


def gene_positions(adata, names: list[str], genome: str | None, data_dir: str | Path, auto_download: bool = True) -> pd.DataFrame:
    """Each named gene's chrom, TSS and strand: by gene id, else by name, as ``atac.choose_genes``."""
    try:
        table = gene_annotation.load(genome, data_dir, auto_download=auto_download)
    except gene_annotation.AnnotationError as e:
        raise AtacError(str(e), 422) from e
    known = atac._dataset_genes(adata).drop_duplicates("name").set_index("name")["gene_id"]
    return place_genes(names, known.to_dict(), table)


def cell_selection(adata, column: str | None, values: list[str] | None) -> np.ndarray | None:
    """A mask of the cells whose ``column`` is one of ``values``, None without a column."""
    if not column:
        return None
    if column not in adata.obs.columns:
        raise AtacError(f"Column '{column}' not found.", 404)
    if not values:
        raise AtacError(f"Pick which values of '{column}' the cells should have.", 422)
    s = adata.obs[column]
    mask = s.astype(str).isin([str(v) for v in values]).to_numpy() & s.notna().to_numpy()
    if int(mask.sum()) < MIN_RUN_CELLS:
        raise AtacError(f"{int(mask.sum()):,} cells have {column} in the chosen groups; links need at least {MIN_RUN_CELLS}.", 422)
    return mask


def plan_links(adaptor, data_dir: str | Path, *, genome: str | None, mode: str = "variable", gene_list: list[str] | None = None,
               cells_column: str | None = None, cells_value: list[str] | None = None, auto_download: bool = True) -> LinkPlan:
    adata = adaptor.adata
    _rna_ready(adaptor)
    available = adaptor.gene_names()
    missing: list[str] = []
    if mode == "variable":
        names = variable_genes(adata)
    elif mode == "all":
        names = [str(g) for g in available]
    elif mode == "list":
        asked = list(dict.fromkeys(str(g).strip() for g in (gene_list or []) if str(g).strip()))
        if not asked:
            raise AtacError("Type at least one gene for the links.", 422)
        if len(asked) > MAX_LIST:
            raise AtacError(f"That list has {len(asked):,} genes; at most {MAX_LIST:,} can be typed. Choose all genes instead.", 422)
        have = set(map(str, available))
        missing = [g for g in asked if g not in have]
        names = [g for g in asked if g in have]
        if not names:
            raise AtacError(f"None of the {len(asked):,} listed genes is in this dataset (first: '{asked[0]}').", 404)
    else:
        raise AtacError(f"Genes '{mode}': use one of {', '.join(MODES)}.", 422)
    table = gene_positions(adata, names, genome, data_dir, auto_download)
    if table.empty:
        raise AtacError(f"None of these {len(names):,} genes has a position in {gene_annotation.source_of(genome)}: it is another "
                        "species or reference.", 422)
    placed = set(table["name"])
    missing += [g for g in names if g not in placed][:50] if mode == "list" else []
    table = order_genes(table)
    mask = cell_selection(adata, cells_column, cells_value)
    return LinkPlan(table, mode, genome, int(mask.sum()) if mask is not None else int(adata.n_obs), int(adata.n_obs), mask,
                    cells_column or None, [str(v) for v in cells_value] if cells_column and cells_value else None,
                    _selection_hash(adata, mask), len(names), missing)


# Measured at 2 CPUs through a running server (2026-11-14, v0.79). Each cis pair (a peak within DISTANCE of the TSS)
# is correlated, cheaply; each pair past SCORE_CUTOFF ("tested") draws its background from every peak (a cost per peak
# of the dataset) and correlates it (per value of the mean peak, nnz / peaks). Runs (cells, values, peaks; tested genes,
# cis pairs, tested pairs, seconds):
#   PBMC 10k (11,208, 98.0M, 143,887): variable 1,710 / 123,256 / 16,615 / 54 s; all 21,297 / 1,830,919 / 72,773 / 164 s
#   5k mouse kidney (5,116, 24.0M, 73,918): variable 1,763 / 67,376 / 5,250 / 12.3 s; all 19,953 / 1,048,245 / 58,108 / 59 s
#   E18 mouse brain (4,494, 38.4M, 172,193): variable 1,775 / 148,064 / 15,956 / 44 s; all 17,893 / 1,631,427 / 49,747 / 106 s
# With the tested pairs known the model is within 6% of all six. Before the run they are not: the share of the cis
# pairs counted at plan time (``cis_pairs``) that pass the cutoff is the data's (variable 7.1-11.8%, all 2.6-4.8%), so
# the share by mode is about the geometric mean of the three (all-gene runs a little above, so the kidney's stays
# inside 20%): estimates 45, 166, 14, 49, 42 and 145 s (-17% to +37%, the E18 all-gene run over). The v0.78 model (a fixed cost per pair, fit on PBMC) said 44 and 137 s for the kidney.
_FIRST_SECONDS = 2.0
_BY_PEAK_SECONDS_PER_VALUE = 1.5e-7  # the store read twice and the by-peak copy written
_PER_PEAK_SECONDS = 3e-5  # GC, and each chromosome's densities and sorted features
_TESTED_SECONDS_PER_PEAK = 1.03e-8  # a tested pair's background draw, per peak of the dataset
_TESTED_SECONDS_PER_VALUE = 5.55e-7  # its background's correlations, per value of the mean peak
_TESTED_SHARE = {"variable": 0.093, "list": 0.093, "all": 0.036}
DOWNLOAD_SECONDS = 120.0  # about 800 MB
_BASE_BYTES = 600 * 1024**2  # scanpy and friends loaded


def cis_pairs(genes: pd.DataFrame, chrom, start, end, distance: int = DISTANCE) -> int:
    """The (gene, peak) pairs within ``distance`` of each gene's TSS, peaks on standard chromosomes:
    what the run correlates (``iter_links``, before its cell filters). One sort per chromosome."""
    chrom = np.asarray(chrom).astype(str)
    start, end = np.asarray(start, dtype=np.int64), np.asarray(end, dtype=np.int64)
    ok = standard_chromosomes(chrom) & (start >= 0)
    g_chrom = genes["chrom"].to_numpy(dtype=str)
    tss1 = genes["tss"].to_numpy(dtype=np.int64) + 1
    total = 0
    for c in np.unique(g_chrom):
        here = np.flatnonzero(ok & (chrom == c))
        if here.size < 2:
            continue
        order = np.argsort(start[here], kind="stable")
        starts, reach = start[here][order], np.maximum.accumulate(end[here][order])
        t = tss1[g_chrom == c]
        n = np.searchsorted(starts, t + distance, side="right") - np.searchsorted(reach, t - distance, side="left")
        total += int(np.maximum(n, 0)[n >= 2].sum())
    return total


def estimate_seconds(nnz: int, n_peaks: int, n_genes: int, *, mode: str = "variable", cell_share: float = 1.0,
                     n_cis: int | None = None) -> float:
    """``n_cis``: the plan's cis pairs (``cis_pairs``); without it, 70 per gene (the datasets above: 51 to 90).
    ``cell_share``: the share of the dataset's cells used (``nnz`` is already the used cells'). Fewer cells: fewer
    genes pass and fewer peaks, but more of a gene's pairs reach |r| > 0.05 by chance, so this is only rough for chosen
    groups (the card says so)."""
    cis = n_genes * 70 if n_cis is None else n_cis
    tested = cis * _TESTED_SHARE.get(mode, _TESTED_SHARE["list"]) * min(1.0, cell_share) ** 0.1
    per_pair = n_peaks * _TESTED_SECONDS_PER_PEAK + nnz / max(1, n_peaks) * _TESTED_SECONDS_PER_VALUE
    return round(_FIRST_SECONDS + nnz * _BY_PEAK_SECONDS_PER_VALUE + n_peaks * _PER_PEAK_SECONDS + tested * per_pair, 0)


def estimate_bytes(nnz: int, n_peaks: int, n_genes: int, n_cells: int) -> int:
    """What a run holds beyond the loaded dataset at its peak: the libraries, a block of the store with its by-peak
    copy, a batch of genes, a slice of peaks, one chromosome's features and backgrounds. The by-peak copy itself is
    file-backed and droppable, so it is not counted. PBMC 10k: 0.82 GiB against 0.72-0.77 GiB of cgroup anon; mouse
    E18 with its 0.34 GiB of tables in memory: 0.84 GiB against 0.74-0.78."""
    return int(_BASE_BYTES + LINK_BLOCK_VALUES * 24 + RNA_BATCH_BYTES + PRODUCT_VALUES * 16 + n_peaks * 160)


def estimate_disk_bytes(nnz: int) -> int:
    """The by-peak copy, removed when the run ends."""
    return int(nnz * 8)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass
class LinksResult:
    result_id: str
    table: pd.DataFrame  # the kept links
    counts: dict[str, int]
    params: dict[str, Any]
    method: str
    caption: str
    selection: str
    created_at: str
    seconds: float
    missing: list[str] = field(default_factory=list)


_LAST: dict[str, LinksResult] = {}
KEEP_RESULTS = 8


def run_key(params: dict[str, Any]) -> str:
    return f"{params.get('genes')}|{params.get('cells_column') or ''}|{','.join(params.get('cells_value') or [])}"


def run_label(res: LinksResult) -> str:
    p = res.params
    genes = {"variable": "variable genes", "all": "all genes"}.get(str(p.get("genes")), f"{res.counts.get('n_genes_asked', 0):,} listed genes")
    cells = f"{p['cells_column']} = {', '.join(p.get('cells_value') or [])}" if p.get("cells_column") else "all cells"
    return f"Peak-gene links: {genes}, {cells}"


def remember(adaptor, dataset_id: str, res: LinksResult) -> None:
    _LAST[res.result_id] = res
    while len(_LAST) > KEEP_RESULTS:
        _LAST.pop(next(iter(_LAST)))
    results_store.keep(adaptor, RESULT_KIND, res, result_id=res.result_id, key=run_key(res.params), label=run_label(res),
                       params=dict(res.params))


def recall(dataset_id: str, result_id: str) -> LinksResult | None:
    hit = _LAST.get(result_id)
    if hit is not None:
        return hit
    return results_store.recall(dataset_id, result_id, RESULT_KIND)


def is_current(adaptor, res: LinksResult) -> bool:
    """Whether the cells this result names are still the ones it used (a relabel moves them
    without any recorded step)."""
    try:
        p = res.params or {}
        mask = cell_selection(adaptor.adata, p.get("cells_column"), p.get("cells_value"))
        return _selection_hash(adaptor.adata, mask) == res.selection
    except AtacError:
        return False


def kept_runs(adaptor) -> list[dict[str, Any]]:
    """What is kept for this dataset, newest first, with whether each still applies."""
    dataset_id = results_store.dataset_id_of(adaptor)
    if dataset_id is None:
        return []
    return [{"id": e["id"], "label": e.get("label") or "Peak-gene links", "created_at": e.get("created_at"),
             "stale": bool(e.get("stale"))} for e in results_store.entries(dataset_id, RESULT_KIND, adaptor=adaptor)]


def latest(adaptor) -> LinksResult | None:
    """The newest result computed on the cells and data the dataset holds now."""
    dataset_id = results_store.dataset_id_of(adaptor)
    if dataset_id is None:
        return None
    try:
        for e in results_store.entries(dataset_id, RESULT_KIND, adaptor=adaptor):
            if e.get("stale"):
                continue
            got = results_store.load(dataset_id, e["id"])
            if got is not None and is_current(adaptor, got[1]):
                return got[1]
    except Exception:  # noqa: BLE001 - a test double, an unreadable result
        logger.debug("links: no latest result", exc_info=True)
    return None


SORTS = ("gene", "peak", "distance_to_tss", "score", "z", "p", "padj")


def payload(res: LinksResult, *, gene: str = "", search: str = "", sort: str = "p", descending: bool = False,
            offset: int = 0, limit: int = 100, stale: bool = False) -> dict[str, Any]:
    t = res.table
    if gene:
        t = t[t["gene"].astype(str).str.casefold() == gene.casefold()]
    elif search:
        t = t[t["gene"].astype(str).str.contains(search, case=False, regex=False)
              | t["peak"].astype(str).str.contains(search, case=False, regex=False)]
    if sort == "abs_score":
        order = np.argsort(-np.abs(t["score"].to_numpy()) if descending else np.abs(t["score"].to_numpy()), kind="stable")
    elif sort == "abs_distance":
        order = np.argsort(-np.abs(t["distance_to_tss"].to_numpy()) if descending else np.abs(t["distance_to_tss"].to_numpy()), kind="stable")
    else:
        v = t[sort]
        order = v.sort_values(ascending=not descending, kind="stable").index
        order = t.index.get_indexer(order)
    page = t.iloc[np.asarray(order)[offset:offset + limit]]
    columns = ["gene", "peak", "distance_to_tss", "score", "z", "p", "padj"]
    return {"id": res.result_id, "total": int(len(t)), "offset": offset, "limit": limit, "sort": sort, "descending": descending,
            "rows": json.loads(page[columns].to_json(orient="records", double_precision=15)), "counts": dict(res.counts), "params": dict(res.params),
            "method": res.method, "caption": res.caption, "seconds": res.seconds, "created_at": res.created_at,
            "missing": list(res.missing), "stale": stale}


def csv_text(res: LinksResult) -> str:
    return res.table.drop(columns=["peak_index"], errors="ignore").to_csv(index=False)


def locus(res: LinksResult, store, genome: str | None, data_dir: str | Path, gene: str, auto_download: bool = True) -> dict[str, Any]:
    """One gene's links for a drawing: the arcs (peak middle to TSS), the window that holds them,
    the peaks and genes in that window."""
    t = res.table[res.table["gene"].astype(str).str.casefold() == gene.casefold()]
    if t.empty:
        raise AtacError(f"This result has no links for '{gene}'. Search the table for another gene.", 404)
    chrom = str(t["chrom"].iloc[0])
    tss = int(t["tss"].iloc[0])
    middle = (t["start"].to_numpy(dtype=np.float64) + t["end"].to_numpy(dtype=np.float64)) / 2
    lo, hi = min(tss, float(middle.min())), max(tss, float(middle.max()))
    pad = max(5_000.0, 0.06 * (hi - lo))
    start_w, end_w = max(0, int(lo - pad)), int(hi + pad)
    p_chrom, p_start, p_end = atac.peak_positions(store)
    here = (p_chrom == chrom) & (p_end > start_w) & (p_start < end_w)
    peaks = [{"start": int(s), "end": int(e)} for s, e in zip(p_start[here], p_end[here])]
    genes: list[dict[str, Any]] = []
    note = None
    try:
        table = gene_annotation.load(genome, data_dir, auto_download=auto_download)
        in_view = table[(table["chrom"] == chrom) & (table["end"] > start_w) & (table["start"] < end_w)
                        & (table["gene_type"] == "protein_coding")]
        genes = [{"name": str(r.gene_name), "start": int(r.start), "end": int(r.end), "strand": str(r.strand)}
                 for r in in_view.sort_values("start").head(40).itertuples()]
    except gene_annotation.AnnotationError as e:
        note = str(e)
    arcs = [{"peak": str(r.peak), "start": int(r.start), "end": int(r.end), "middle": float(m), "tss": tss, "score": float(r.score),
             "z": float(r.z), "p": float(r.p), "padj": float(r.padj), "distance_to_tss": float(r.distance_to_tss)}
            for r, m in zip(t.itertuples(), middle)]
    return {"gene": str(t["gene"].iloc[0]), "chrom": chrom, "tss": tss, "start": start_w, "end": end_w, "arcs": arcs, "peaks": peaks,
            "genes": genes, "note": note}


# ---------------------------------------------------------------------------
# The run from a dataset
# ---------------------------------------------------------------------------


def _ensure_genome(data_dir, genome, auto_download, stop) -> Iterator[dict[str, Any]]:
    """``genome_seq.ensure`` off this thread, its progress passed on; the last item has the file."""
    events: queue.Queue = queue.Queue()
    outcome: dict[str, Any] = {}

    def fetch() -> None:
        try:
            outcome["genome"] = genome_seq.ensure(data_dir, genome, auto_download=auto_download,
                                                  progress=events.put, stop=stop)
        except BaseException as e:  # noqa: BLE001 - handed to the generator below
            outcome["error"] = e

    worker = threading.Thread(target=fetch, name="links-genome", daemon=True)
    worker.start()
    while worker.is_alive() or not events.empty():
        try:
            e = events.get(timeout=0.25)
        except queue.Empty:
            continue
        yield {"type": "progress", "step": "genome", "message": e.get("message", ""), "done": e.get("done"), "total": e.get("total")}
    error = outcome.get("error")
    if isinstance(error, genome_seq.GenomeCancelled):
        raise AtacCancelled()
    if isinstance(error, genome_seq.GenomeUnavailable):
        raise AtacError(str(error), 503)
    if error is not None:
        raise error
    yield {"type": "genome", "genome": outcome["genome"]}


def iter_run(adaptor, store, p: LinkPlan, *, data_dir: str | Path, auto_download: bool = True,
             stop: Callable[[], bool] = lambda: False) -> Iterator[dict[str, Any]]:
    """From a plan: the genome, GC, the links. The by-peak copy goes in a folder of its own under
    ``DATA_DIR/tmp``, removed however the run ends."""
    began = time.monotonic()
    rows_all = atac.cell_rows(store, adaptor.adata)
    rows = rows_all if p.cell_mask is None else rows_all[p.cell_mask]
    nnz = int(store.nnz(MATRIX) * (p.n_cells / max(1, store.n_obs)))
    tmp = Path(data_dir) / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    need = estimate_disk_bytes(nnz)
    free = shutil.disk_usage(tmp).free
    if free < need:
        raise AtacError(f"Peak-gene links need {need / 1e9:.1f} GB of disk for a while and the data volume has {free / 1e9:.1f} GB free. "
                        "Free some space and try again.", 507)
    two_bit = None
    for item in _ensure_genome(data_dir, p.genome, auto_download, stop):
        if item["type"] == "genome":
            two_bit = item["genome"]
        else:
            yield item
    yield {"type": "progress", "step": "gc", "message": "GC percent of every peak", "done": 0, "total": 1}
    chrom, start, end = atac.peak_positions(store)
    gc = peak_gc(two_bit, chrom, start, end, stop)
    folder = Path(tempfile.mkdtemp(prefix="links-", dir=tmp))
    try:
        read = adaptor.get_expression
        for item in iter_links(store, rows, p.cell_mask, read, p.genes, store.features, chrom, start, end, gc, folder, stop=stop):
            if item["type"] == "result":
                item["seconds"] = round(time.monotonic() - began, 1)
            yield item
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def run_params(p: LinkPlan, gene_list: list[str] | None) -> dict[str, Any]:
    params: dict[str, Any] = {"genes": p.mode, "cells_column": p.cells_column, "cells_value": p.cells_value, "distance": DISTANCE,
                              "min_cells": MIN_CELLS, "score_cutoff": SCORE_CUTOFF, "n_sample": N_SAMPLE, "p_cutoff": P_CUTOFF,
                              "seed": SEED, "genome": p.genome, "selection": p.selection, "standard_only": True,
                              "genome_url": genome_seq.RELEASES[gene_annotation.canonical_genome(p.genome)].url,
                              "gene_positions_url": gene_annotation.RELEASES[gene_annotation.canonical_genome(p.genome)].url}
    if p.mode == "list":
        params["gene_list"] = list(gene_list or [])
    return params


def build_result(p: LinkPlan, item: dict[str, Any], gene_list: list[str] | None) -> LinksResult:
    caption = CAPTION + ("" if p.cells_column else CAPTION_ALL_CELLS)
    return LinksResult(
        result_id=uuid.uuid4().hex[:12], table=item["frame"], counts=item["counts"], params=run_params(p, gene_list),
        method=METHOD.format(distance=DISTANCE, min_cells=MIN_CELLS, score_cutoff=SCORE_CUTOFF, n_sample=N_SAMPLE, seed=SEED,
                             p_cutoff=P_CUTOFF),
        caption=caption, selection=p.selection, created_at=datetime.now(timezone.utc).isoformat(),
        seconds=float(item.get("seconds") or 0.0), missing=p.missing)


def record(adata, res: LinksResult, resources: dict | None = None) -> None:
    """A provenance step (a result only; the replay script computes it again)."""
    from scview.core import provenance

    provenance.record_step(adata, step=STEP, tool="scview.atac_links (Signac LinkPeaks)",
                           params={**res.params, "result_id": res.result_id, "n_links": res.counts.get("n_links"),
                                   "n_pairs_tested": res.counts.get("n_pairs_tested")},
                           resources=resources)


def grounding_lines(adaptor) -> list[str]:
    """The last links on these cells as facts for the co-pilot."""
    res = latest(adaptor)
    if res is None:
        return []
    c = res.counts
    t = res.table
    top = t.reindex(t["z"].abs().sort_values(ascending=False).index).head(5) if len(t) else t
    named = "; ".join(f"{r.gene} with {r.peak} (r = {r.score:.3f}, z = {r.z:.1f})" for r in top.itertuples())
    p = res.params
    cells = f"{p['cells_column']} in {', '.join(p.get('cells_value') or [])}" if p.get("cells_column") else "all cells"
    line = (f"- Peak-gene links ({cells}, {c.get('n_cells', 0):,} cells; Signac's LinkPeaks, a correlation, a candidate and never "
            f"proof of regulation): {c.get('n_links', 0):,} links at p < {P_CUTOFF} from {c.get('n_pairs_tested', 0):,} pairs tested "
            f"on {c.get('n_genes_tested', 0):,} genes; {c.get('n_links_padj', 0):,} at BH FDR 5%")
    if named:
        line += f"; strongest by |z|: {named}"
    return [line + f" [{TAG}]"]


def named_gene_lines(adaptor, query: str, max_genes: int = 3, max_links: int = 8) -> list[tuple[str, str]]:
    """(gene, fact) for each gene the question names, as the top links alone leave out most genes."""
    res = latest(adaptor)
    if res is None or not query:
        return []
    t = res.table
    linked = {str(g).upper(): str(g) for g in t["gene"].unique()} if len(t) else {}
    known = {str(g).upper() for g in adaptor.adata.var_names}
    out: list[tuple[str, str]] = []
    for word in dict.fromkeys(re.findall(r"[A-Za-z][A-Za-z0-9.-]+", query)):
        key = word.rstrip(".-").upper()
        if len(out) >= max_genes:
            break
        if key in linked:
            rows = t[t["gene"] == linked[key]]
            top = rows.reindex(rows["score"].abs().sort_values(ascending=False).index).head(max_links)
            named = "; ".join(f"{r.peak} ({int(r.distance_to_tss):+,d} bp from the TSS, r = {r.score:.3f}, z = {r.z:.1f}, p = {r.p:.2g})" for r in top.itertuples())
            out.append((linked[key], f"- Peak-gene links of {linked[key]}, {len(rows)} kept at p < {P_CUTOFF} (candidates), the {len(top)} largest |r|: {named}"))
        elif key in known and word[0].isupper() and len(key) >= 3 and re.search(r"link|peak|enhancer|regulat", query, re.I):
            what = {"variable": "variable genes only", "all": "all genes", "list": "a chosen list of genes"}.get(res.params.get("genes"), "its genes")
            out.append((key, f"- Peak-gene links of {key}: none kept in this run, which tested {what}; a gene outside that was not tested."))
    return out
