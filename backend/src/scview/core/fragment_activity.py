"""Fragment-based gene activity: Signac's ``GeneActivity`` on the dataset's fragments file.

Signac 1.17.1 (MIT): ``CollapseToLongestTranscript`` (a gene's span), ``biotypes =
"protein_coding"``, ``Extend(upstream = 2000)`` on the gene's strand, genes ``max.width``
500 kb or wider left out, then ``FeatureMatrix``: every fragment line of a cell that
overlaps the region counts 1 (tabix overlap of the 0-based half-open fragment with the
region; a fragment over two genes counts for both, its read-count column is not used). The
vignette then log-normalises with the median of the cells' totals as the scale factor
(``NormalizeData(scale.factor = median(nCount_RNA))``), which ``atac.activity_gene`` applies.

Unlike the peak-based activity (``core/atac.py``), reads in a gene but outside every called
peak count: Cell Ranger called no peak at the Emcn promoter in the 5k mouse kidney, so its
peak-based activity was 0 in every cell while the fragments there are 6x denser in
endothelium than in proximal tubule.

One pass over the file (``fragment_file.iter_text`` + ``parse``, as the QC pass). The file is
sorted by position, so a gene can be written as soon as the fragments have passed its end:
counts are held for the genes still open (at most 500 kb behind), and rows go to the by-gene
store in the order ``row_order`` gives (chromosomes as the tabix index lists them, then by
start).
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterator, Sequence
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp

UPSTREAM = 2_000  # GeneActivity's extend.upstream (extend.downstream 0)
BIOTYPES = ("protein_coding",)
MAX_WIDTH = 500_000  # genes this wide or wider (with the upstream) are left out
FLUSH_VALUES = 2_000_000  # stored values gathered before a write to the store
PROGRESS_EVERY = 0.02

CAPTION = ("Fragment-based gene activity (Signac's GeneActivity): fragments over the gene body + 2 kb upstream, "
           "protein-coding genes under 500 kb, log-normalised with the median cell total as scale factor. "
           "Accessibility, not expression.")


class FragmentActivityError(ValueError):
    pass


class FragmentActivityCancelled(Exception):
    pass


def signac_genes(start, end, gene_type, biotypes=BIOTYPES, max_width=MAX_WIDTH):
    """The genes GeneActivity keeps: its biotypes, and narrower than ``max_width`` with the
    upstream extension (``start``/``end``: the extended region, 0-based half open)."""
    return np.isin(np.asarray(gene_type, dtype=object), list(biotypes)) & ((np.asarray(end) - np.asarray(start)) < max_width)


def row_order(chrom: np.ndarray, start: np.ndarray, chrom_order: Sequence[str]) -> np.ndarray:
    """The genes in the order their counts are complete: the file's chromosomes in its order,
    by start, then genes on chromosomes the file never names."""
    rank = {str(c): i for i, c in enumerate(chrom_order)}
    r = np.fromiter((rank.get(str(c), len(rank)) for c in chrom), dtype=np.int64, count=len(chrom))
    return np.lexsort((np.arange(len(chrom)), np.asarray(start), r))


def region_hits(fs, fe, gs, ge, reach):
    """(fragment, region) pairs where the fragment [fs, fe) overlaps the region [gs, ge); regions
    sorted by start, ``reach`` the running maximum of their ends. Pure numpy (the replay copies it)."""
    hi = np.searchsorted(gs, fe, side="left")  # regions that start before the fragment ends
    lo = np.searchsorted(reach, fs, side="right")  # every region before lo ends at or before the fragment starts
    n = np.maximum(hi - lo, 0)
    total = int(n.sum())
    if not total:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    frag = np.repeat(np.arange(fs.size), n)
    first = np.repeat(np.cumsum(n) - n, n)
    region = np.repeat(lo, n) + (np.arange(total) - first)
    ok = ge[region] > fs[frag]
    return frag[ok], region[ok]


class _Rows:
    """Rows to the writer in order, gathered to ``FLUSH_VALUES`` stored values a write."""

    def __init__(self, write: Callable[[sp.csr_matrix], None], n_cells: int) -> None:
        self.write, self.n_cells = write, n_cells
        self.parts: list[sp.csr_matrix] = []
        self.values = 0
        self.rows = 0
        self.with_counts = 0

    def add(self, m: sp.csr_matrix) -> None:
        if not m.shape[0]:
            return
        self.parts.append(m)
        self.values += m.nnz
        self.rows += m.shape[0]
        self.with_counts += int((np.diff(m.indptr) > 0).sum())
        if self.values >= FLUSH_VALUES:
            self.flush()

    def empty(self, n: int) -> None:
        if n > 0:
            self.add(sp.csr_matrix((n, self.n_cells), dtype=np.float32))

    def flush(self) -> None:
        if self.parts:
            self.write(sp.vstack(self.parts, format="csr") if len(self.parts) > 1 else self.parts[0])
        self.parts, self.values = [], 0


class _Chrom:
    """One chromosome's genes (sorted by start, ``first`` the row of the first in the written
    order): the counts of the genes the fragments have not passed yet."""

    def __init__(self, start: np.ndarray, end: np.ndarray, rows: _Rows) -> None:
        self.gs, self.ge = start.astype(np.int64), end.astype(np.int64)
        self.reach = np.maximum.accumulate(self.ge) if self.ge.size else self.ge
        self.rows = rows
        self.written = 0
        self.gene = np.zeros(0, dtype=np.int64)
        self.cell = np.zeros(0, dtype=np.int64)
        self.last = -1

    def add(self, fs: np.ndarray, fe: np.ndarray, cell: np.ndarray) -> np.ndarray:
        """Count a run of fragments (sorted by start); returns the cell of each counted pair."""
        if not fs.size:
            return np.zeros(0, dtype=np.int64)
        if fs[0] < self.last or (fs.size > 1 and (np.diff(fs) < 0).any()):
            raise FragmentActivityError("The fragments file is not sorted by position, so genes cannot be counted in "
                                        "one pass. Sort it (sort -k1,1 -k2,2n) and index it with tabix again.")
        self.last = int(fs[-1])
        frag, gene = region_hits(fs, fe, self.gs, self.ge, self.reach)
        cells = cell[frag]
        self.gene = np.concatenate([self.gene, gene])
        self.cell = np.concatenate([self.cell, cells])
        # Every later fragment starts at or after fs[-1]: genes ending there are complete.
        open_ = np.flatnonzero(self.ge[self.written:] > self.last)
        self._write(self.written + (int(open_[0]) if open_.size else self.ge.size - self.written))
        return cells

    def _write(self, upto: int) -> None:
        if upto <= self.written:
            return
        done = self.gene < upto
        g, c = self.gene[done] - self.written, self.cell[done]
        m = sp.csr_matrix((np.ones(g.size, dtype=np.float32), (g, c)), shape=(upto - self.written, self.rows.n_cells))
        m.sum_duplicates()
        self.rows.add(m)
        self.gene, self.cell = self.gene[~done], self.cell[~done]
        self.written = upto

    def finish(self) -> None:
        self._write(self.ge.size)


def iter_counts(path, barcodes: Sequence[str], chrom: np.ndarray, start: np.ndarray, end: np.ndarray,
                write: Callable[[sp.csr_matrix], None], *, stop: Callable[[], bool] = lambda: False
                ) -> Iterator[dict[str, Any]]:
    """Fragments per gene region and barcode, one pass. ``chrom``/``start``/``end``: the regions in
    the written order (``row_order``), ``write`` takes their rows (genes x barcodes) in that order.
    Yields ``progress`` every 2% of the file, then ``{"type": "result", "totals", ...}``: each
    barcode's sum over the genes."""
    from scview.core import fragment_file as ff

    t0 = time.monotonic()
    size = os.path.getsize(path)
    codes = ff.Barcodes(barcodes)
    n = len(codes)
    rows = _Rows(write, n)
    chrom = np.asarray(chrom, dtype=object).astype(str)
    bounds: dict[str, tuple[int, int]] = {}
    for i, c in enumerate(chrom.tolist()):
        a, _b = bounds.get(c, (i, i))
        bounds[c] = (a, i + 1)
    for c, (a, b) in bounds.items():
        if b - a != int((chrom[a:b] == c).sum()):
            raise FragmentActivityError("Genes must come grouped by chromosome (row_order).")
    totals = np.zeros(n, dtype=np.int64)
    current: _Chrom | None = None
    current_name = ""
    done_to = 0  # rows written or owned by the current chromosome
    seen: set[str] = set()
    n_lines = n_counted = 0
    last = 0.0
    for done, text in ff.iter_text(path):
        if stop():
            raise FragmentActivityCancelled()
        p = ff.parse(text, codes)
        del text
        n_lines += p.n_lines
        for name, a, b in p.chroms:
            if current is None or name != current_name:
                if current is not None:
                    current.finish()
                if name in seen:
                    raise FragmentActivityError(f"The fragments file returns to {name} after another chromosome: it is not "
                                                "sorted by position.")
                seen.add(name)
                current_name = name
                lo, hi = bounds.get(name, (done_to, done_to))
                if hi > lo:
                    if lo < done_to:
                        raise FragmentActivityError(f"{name} comes earlier in the gene order than in the fragments file.")
                    rows.empty(lo - done_to)  # chromosomes with genes and no fragments, before this one
                    done_to = hi
                current = _Chrom(start[lo:hi], end[lo:hi], rows)
            cells = current.add(p.start[a:b], p.end[a:b], p.barcode[a:b])
            if cells.size:
                totals += np.bincount(cells, minlength=n)
                n_counted += int(cells.size)
        fraction = done / size if size else 1.0
        if fraction - last >= PROGRESS_EVERY:
            last = fraction
            yield {"type": "progress", "fraction": round(fraction, 3),
                   "message": f"Gene activity: {n_lines:,} fragments read ({fraction:.0%} of the file)"}
    if current is not None:
        current.finish()
    rows.empty(len(chrom) - rows.rows)
    rows.flush()
    yield {"type": "result", "totals": totals, "n_lines": n_lines, "n_counted": n_counted,
           "n_genes_with_fragments": rows.with_counts, "seconds": round(time.monotonic() - t0, 1)}


# The same counts in plain pandas, for the replay script (REPLAY_CODE: copied into it).


def fragment_gene_counts(path, cells, chrom, start, end, chunk_lines=2_000_000):
    """Signac's FeatureMatrix of these regions: a cells x regions CSR (every fragment line of a
    listed cell that overlaps a region counts 1 for it), and the chromosomes the file names."""
    index = pd.Index(cells)
    by_chrom = {}
    for c in pd.unique(np.asarray(chrom, dtype=object)):
        idx = np.flatnonzero(np.asarray(chrom, dtype=object) == c)
        idx = idx[np.argsort(np.asarray(start)[idx], kind="stable")]
        ge = np.asarray(end)[idx].astype(np.int64)
        by_chrom[str(c)] = (idx, np.asarray(start)[idx].astype(np.int64), ge, np.maximum.accumulate(ge))
    rows, cols, seen = [], [], set()
    reader = pd.read_csv(path, sep="\t", comment="#", header=None, usecols=[0, 1, 2, 3],
                         names=["chrom", "start", "end", "barcode"], chunksize=chunk_lines,
                         dtype={"chrom": str, "start": np.int64, "end": np.int64, "barcode": str})
    for chunk in reader:
        seen.update(chunk["chrom"].astype(str).unique().tolist())
        who = index.get_indexer(chunk["barcode"].to_numpy())
        keep = who >= 0
        who, chunk = who[keep], chunk[keep]
        for c, at in chunk.groupby("chrom", sort=False).indices.items():
            genes = by_chrom.get(str(c))
            if genes is None:
                continue
            idx, gs, ge, reach = genes
            frag, region = region_hits(chunk["start"].to_numpy()[at], chunk["end"].to_numpy()[at], gs, ge, reach)
            rows.append(who[at][frag])
            cols.append(idx[region])
    r = np.concatenate(rows) if rows else np.zeros(0, dtype=np.int64)
    c = np.concatenate(cols) if cols else np.zeros(0, dtype=np.int64)
    m = sp.csr_matrix((np.ones(r.size, dtype=np.float32), (r, c)), shape=(len(cells), len(chrom)))
    m.sum_duplicates()
    return m, seen


REPLAY_CODE = (signac_genes, region_hits, fragment_gene_counts)
REPLAY_CONSTANTS = ("BIOTYPES", "MAX_WIDTH")

