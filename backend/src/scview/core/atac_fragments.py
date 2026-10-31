"""ATAC fragments: attach the file, per-cell QC, coverage tracks.

The file (``core/fragment_file.py``) and its ``.tbi`` sit in the dataset's modality folder
(``modality_store.store_dir``), so they go with the dataset. This module takes cell
barcodes, a group column and gene positions from the dataset as it is.

QC follows Signac (MIT), checked per cell against Signac 1.17.1 on 10x PBMC:

- TSS enrichment = ``TSSEnrichment(fast=TRUE)`` (``TSSFast``, ``extract_tss_counts``,
  ``SingleFileCutMatrix``): Tn5 cuts (each fragment's start and end) within 500 bp of a
  protein-coding TSS per base, over cuts in the 100 bp at each end of +-1 kb per base. A cell
  with no flank cuts takes the mean of all cells' flanks. Windows are Signac's to the base,
  including the shift ``SingleFileCutMatrix`` applies: a cut counts when its raw value in the
  file falls in [rs + 2, rs + w + 1] of a region starting at 1-based rs (a start only up to
  rs + w - 2, past which tabix does not return the fragment).
- Nucleosome signal = ``NucleosomeSignal`` (src/group.cpp): fragments of 147 < length < 294
  over fragments of length < 147, on the whole file (Signac's ``n=NULL``; its default reads
  only the first ``cells x 5000`` lines). A cell with no nucleosome-free fragment gets NaN
  where Signac has Inf or NaN.

One pass over the file, on a worker thread plus one inflating (``fragment_file.iter_text``).
Results: obs ``atac_tss_enrichment``, ``atac_nucleosome_signal``, ``atac_n_fragments``;
``uns["scview_atac_fragments"]`` (JSON: the length histogram, TSS profile, medians,
parameters); per-cell length counts in 5 bp bins in the modality store
``atac_fragment_lengths``, summed per group when asked (``lengths_by_group``).

Coverage: per group, mean fragment depth per bin over the group's fragments in cells x 1e6
(``depth per million fragments``; Signac's CoveragePlot scales cut counts by each group's total
to the median total instead). Without the QC's fragment counts it is depth per 1,000 cells, and
the answer says so. Genes are gene bodies from ``core/gene_annotation.py`` (no exons).
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from collections import OrderedDict
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp

from scview.core import fragment_file as ff
from scview.core import gene_annotation, modality_store

logger = logging.getLogger(__name__)

FRAGMENTS_FILE = "atac_fragments.tsv.gz"
INDEX_FILE = FRAGMENTS_FILE + ".tbi"
STATE_FILE = "atac_fragments.json"
LENGTH_STORE = "atac_fragment_lengths"
PEAK_STORE = "atac"
UNS_KEY = "scview_atac_fragments"
TAG = "result:fragqc"
STEP = "atac_fragment_qc"
VERSION = 1

OBS_TSS = "atac_tss_enrichment"
OBS_NUCLEOSOME = "atac_nucleosome_signal"
OBS_N_FRAGMENTS = "atac_n_fragments"

MIN_BARCODE_SHARE = 0.05
CHECK_LINES = 1_000_000

# Signac TSSFast: region start relative to the 0-based TSS (1-based rs = tss + 1 + this), width.
CENTER = (-500, 1001)
FLANK_UP = (-1000, 100)
FLANK_DOWN = (901, 100)
PROFILE_FLANK = 1000
PROFILE_BIN = 5
NUCLEOSOME_FREE_MAX = 147  # length < this
MONO_LOW, MONO_HIGH = 147, 294  # MONO_LOW < length < MONO_HIGH
MAX_LENGTH = 1000  # the histogram's last bin holds longer ones
GROUP_LENGTH_BIN = 5
GROUP_LENGTH_BINS = MAX_LENGTH // GROUP_LENGTH_BIN

MAX_WINDOW = 2_000_000
GENE_PAD = 5_000
DEFAULT_BINS = 400
MAX_BINS = 2_000
MAX_GROUPS = 24
PROGRESS_EVERY = 0.02

# Measured on 10x PBMC multiome (2.24 GiB, 236M lines, 11,898 cells), 2 cores: 106 s, 0.5 GiB of growth.
SECONDS_PER_GB = 47.0
BASE_PEAK_BYTES = 450 * 1024**2
BYTES_PER_CELL = GROUP_LENGTH_BINS * 4 + 6 * 8
LOW_TSS = 1.5

NOTE_MISSING_INDEX = ("The tabix index (atac_fragments.tsv.gz.tbi) is still missing: Cell Ranger ARC writes it "
                      "beside the fragments file, so upload that one too.")
NOTE_MISSING_FRAGMENTS = ("The fragments file (atac_fragments.tsv.gz) is still missing: Cell Ranger ARC writes it "
                          "beside this index, so upload that one too.")
NOT_ATTACHED = "No ATAC fragments file is attached to this dataset: attach atac_fragments.tsv.gz and its .tbi first."


class FragmentsError(ValueError):
    """The sentence says what is wrong and what to do. ``status``: the HTTP status."""

    def __init__(self, message: str, status: int = 422) -> None:
        super().__init__(message)
        self.status = status


class FragmentsCancelled(Exception):
    """The client went away: nothing is saved."""


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------


def fragments_path(data_dir: str | Path, dataset_id: str) -> Path:
    return modality_store.store_dir(data_dir, dataset_id) / FRAGMENTS_FILE


def index_path(data_dir: str | Path, dataset_id: str) -> Path:
    return modality_store.store_dir(data_dir, dataset_id) / INDEX_FILE


def _state_path(data_dir: str | Path, dataset_id: str) -> Path:
    return modality_store.store_dir(data_dir, dataset_id) / STATE_FILE


def part_path(data_dir: str | Path, dataset_id: str, role: str) -> Path:
    """Where an upload is written until it is whole."""
    return modality_store.store_dir(data_dir, dataset_id) / f".{role}.part"


def role_of(name: str) -> str | None:
    """'fragments' or 'index' from a file name, None for neither."""
    low = Path(str(name)).name.lower()
    if low.endswith(".tbi"):
        return "index"
    if low.endswith((".tsv.gz", ".bgz", ".tsv.bgz", ".bed.gz")):
        return "fragments"
    return None


def final_path(data_dir: str | Path, dataset_id: str, role: str) -> Path:
    return index_path(data_dir, dataset_id) if role == "index" else fragments_path(data_dir, dataset_id)


def check_head(head: bytes, role: str) -> None:
    """Refuse a fragments upload on its first bytes rather than after 2.4 GB."""
    if role != "fragments" or ff.is_bgzf(head):
        return
    if head[:2] == b"\x1f\x8b":
        raise FragmentsError("That file is plain gzip, not BGZF, so it cannot be read by region: use "
                             "atac_fragments.tsv.gz as Cell Ranger ARC wrote it, or recompress with bgzip and tabix.")
    raise FragmentsError("That file is not a BGZF-compressed fragments file: use atac_fragments.tsv.gz as "
                         "Cell Ranger ARC wrote it.")


def _read_state(data_dir: str | Path, dataset_id: str) -> dict[str, Any]:
    try:
        return json.loads(_state_path(data_dir, dataset_id).read_text())
    except (OSError, ValueError):
        return {}


def _write_state(data_dir: str | Path, dataset_id: str, state: dict[str, Any]) -> None:
    path = _state_path(data_dir, dataset_id)
    part = path.with_name(f".{path.name}.part")
    part.write_text(json.dumps(state))
    os.replace(part, path)


def detach(data_dir: str | Path, dataset_id: str) -> None:
    """Remove the file, its index, its record and the per-cell lengths."""
    for path in (fragments_path(data_dir, dataset_id), index_path(data_dir, dataset_id),
                 _state_path(data_dir, dataset_id), modality_store.store_path(data_dir, dataset_id, LENGTH_STORE)):
        path.unlink(missing_ok=True)


_AUTOSOME = re.compile(r"^(?:chr)?(\d+)$")


def genome_from_names(names: Sequence[str]) -> str | None:
    """GRCh38 or mm10 from the chromosome names: autosomes to 22 or to 19 (hg19 and GRCh38 have
    the same names, so a wrong build shows as a low TSS enrichment instead)."""
    numbers = [int(m.group(1)) for m in map(_AUTOSOME.match, map(str, names)) if m]
    if not numbers:
        return None
    if max(numbers) >= 20:
        return "GRCh38"
    return "mm10" if max(numbers) >= 15 else None


def _names_in_table(names: Sequence[str], genome: str, data_dir: str | Path, auto_download: bool) -> int | None:
    """How many of the file's chromosomes the gene table has; None when the table is not
    at hand (a network problem must not stop an attach)."""
    try:
        table = gene_annotation.load(genome, data_dir, auto_download=auto_download)
    except gene_annotation.AnnotationError as e:
        logger.info("No gene table to check chromosome names against: %s", e)
        return None
    return len(set(map(str, names)) & set(table["chrom"].astype(str)))


def _check_share(path: str | Path, barcodes: Sequence[str]) -> dict[str, Any]:
    """The barcode check; raises under 5%, with the numbers."""
    found = ff.share_of_cells(path, ff.Barcodes(barcodes), CHECK_LINES)
    share = found["matching"] / found["lines"] if found["lines"] else 0.0
    if share < MIN_BARCODE_SHARE:
        raise FragmentsError(
            f"Only {found['matching']:,} of the first {found['lines']:,} fragments ({share:.1%}) belong to "
            f"this dataset's {len(barcodes):,} cells, under the {MIN_BARCODE_SHARE:.0%} needed: "
            "attach the fragments file of this sample.")
    return {"share": round(share, 4), "lines_checked": found["lines"], "cells_seen": found["cells_seen"]}


def verify(data_dir: str | Path, dataset_id: str, barcodes: Sequence[str], file_name: str | None = None,
           species: str | None = None, auto_download: bool = True) -> dict[str, Any]:
    """Check the attached pair and record what was found: barcode share on the first 1M lines
    (refused under 5%, and the files removed), genome from the index's chromosome names.
    In a worker thread: reads about 50 MB of text."""
    path = fragments_path(data_dir, dataset_id)
    idx = index_path(data_dir, dataset_id)
    state = _read_state(data_dir, dataset_id)
    if file_name:
        state["name"] = str(file_name)
    if path.exists():
        with open(path, "rb") as fh:
            head = fh.read(1024)
        try:
            check_head(head, "fragments")
        except FragmentsError:
            detach(data_dir, dataset_id)
            raise
        if state.get("size") != path.stat().st_size or "share" not in state:
            try:
                state.update(_check_share(path, barcodes), size=path.stat().st_size)
            except FragmentsError:
                detach(data_dir, dataset_id)
                raise
    if idx.exists():
        try:
            index = ff.load_index(idx)
        except ff.FragmentFileError as e:
            idx.unlink(missing_ok=True)
            raise FragmentsError(str(e)) from e
        genome = genome_from_names(index.names)
        state.update(genome=genome, chromosomes=index.names)
        expected = {"human": "GRCh38", "mouse": "mm10"}.get(str(species or "").lower())
        if genome and expected and genome != expected:
            detach(data_dir, dataset_id)
            raise FragmentsError(f"The fragments file's chromosomes are {genome}'s but this dataset's genes are "
                                 f"{species}: attach the fragments file of this sample.")
        if genome:
            shared = _names_in_table(index.names, genome, data_dir, auto_download)
            state["chromosomes_in_annotation"] = shared
            if shared == 0:
                detach(data_dir, dataset_id)
                raise FragmentsError(f"None of the fragments file's chromosome names ({', '.join(index.names[:3])}, ...) "
                                     f"are in the {genome} gene table: use the file Cell Ranger ARC wrote.")
    _write_state(data_dir, dataset_id, state)
    return state


def status(data_dir: str | Path, dataset_id: str, n_cells: int | None = None) -> dict[str, Any]:
    """What is attached, what is still missing (a sentence), and the QC's estimate."""
    frag = fragments_path(data_dir, dataset_id)
    idx = index_path(data_dir, dataset_id)
    has_f, has_i = frag.exists(), idx.exists()
    state = _read_state(data_dir, dataset_id) if (has_f or has_i) else {}
    message = None
    if has_f and not has_i:
        message = NOTE_MISSING_INDEX
    elif has_i and not has_f:
        message = NOTE_MISSING_FRAGMENTS
    out: dict[str, Any] = {
        "attached": has_f and has_i, "fragments": has_f, "index": has_i, "missing": message,
        "name": state.get("name"), "size_bytes": frag.stat().st_size if has_f else None,
        "barcode_share": state.get("share"), "lines_checked": state.get("lines_checked"),
        "cells_seen": state.get("cells_seen"), "genome": state.get("genome"),
        "n_chromosomes": len(state.get("chromosomes") or []) or None,
    }
    if has_f and n_cells is not None:
        out["estimate"] = estimate(frag.stat().st_size, n_cells)
    return out


def require_files(data_dir: str | Path, dataset_id: str) -> tuple[Path, Path]:
    frag = fragments_path(data_dir, dataset_id)
    idx = index_path(data_dir, dataset_id)
    if not frag.exists() and not idx.exists():
        raise FragmentsError(NOT_ATTACHED, 404)
    if not idx.exists():
        raise FragmentsError(NOTE_MISSING_INDEX)
    if not frag.exists():
        raise FragmentsError(NOTE_MISSING_FRAGMENTS)
    return frag, idx


def attach_side_files(data_dir: str | Path, dataset_id: str, fragments: str | Path, index: str | Path | None,
                      barcodes: Sequence[str], species: str | None = None, name: str | None = None) -> dict[str, Any]:
    """Files staged with an ingest: checked where they are, then moved in (a rename on one
    filesystem, so 2.4 GB is not copied). A refusal leaves them where they were."""
    with open(fragments, "rb") as fh:
        check_head(fh.read(1024), "fragments")
    found = _check_share(fragments, barcodes)
    folder = modality_store.store_dir(data_dir, dataset_id)
    folder.mkdir(parents=True, exist_ok=True)
    _write_state(data_dir, dataset_id, {**found, "size": os.path.getsize(fragments)})
    _move(fragments, fragments_path(data_dir, dataset_id))
    if index is not None:
        _move(index, index_path(data_dir, dataset_id))
    return verify(data_dir, dataset_id, barcodes, file_name=name or Path(fragments).name, species=species)


def _move(source: str | Path, target: Path) -> None:
    try:
        os.replace(source, target)
    except OSError:  # another filesystem: a copy, then the source removed
        import shutil

        part = target.with_name(f".{target.name}.part")
        shutil.copyfile(source, part)
        os.replace(part, target)
        Path(source).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Estimates
# ---------------------------------------------------------------------------


def need_bytes(n_cells: int) -> int:
    """Peak the QC pass adds to the backend: text blocks + their arrays + per-cell counts."""
    return BASE_PEAK_BYTES + int(n_cells) * BYTES_PER_CELL


def seconds_for(size_bytes: int) -> float:
    return size_bytes / 1024**3 * SECONDS_PER_GB


def estimate(size_bytes: int, n_cells: int) -> dict[str, Any]:
    """For ``PreflightNote``: memory and time, from the file's size."""
    from scview.core.preflight import time_text

    secs = seconds_for(size_bytes)
    low, high = secs * 0.8, secs * 1.5
    return {"peak_bytes": need_bytes(n_cells), "seconds": round(secs, 1),
            "seconds_range": [round(low, 1), round(high, 1)],
            "note": f"One pass over the whole file, {time_text((low, high))} on 2 cores."}


# ---------------------------------------------------------------------------
# The QC pass
# ---------------------------------------------------------------------------


def tss_table(genome: str | None, data_dir: str | Path, auto_download: bool = True) -> pd.DataFrame:
    """Protein-coding TSSs without chrM, as Signac's GetTSSPositions + TSSEnrichment take them."""
    table = gene_annotation.load(genome, data_dir, auto_download=auto_download)
    keep = (table["gene_type"] == "protein_coding") & (~table["chrom"].isin(["chrM", "MT", "Mt"]))
    return table.loc[keep, ["chrom", "tss", "strand"]].reset_index(drop=True)


class _Track:
    """One chromosome's TSSs. Everything within a window of a TSS is held as base-level
    arrays in compressed coordinates: how many center and flank regions each raw value of a
    start and of an end falls in, and the pile of cuts for the profile."""

    def __init__(self, tss: np.ndarray, minus: np.ndarray) -> None:
        self.tss = tss
        self.minus = minus
        lo = tss - PROFILE_FLANK - 8
        hi = tss + PROFILE_FLANK + 8
        order = np.argsort(lo, kind="stable")
        lo, reach = lo[order], np.maximum.accumulate(hi[order])
        new = np.r_[True, lo[1:] > reach[:-1]]
        self.first = lo[new]
        self.stop = reach[np.r_[new[1:], True]]
        self.offset = np.r_[0, np.cumsum(self.stop - self.first)[:-1]]
        total = int((self.stop - self.first).sum())
        self.center_start, self.center_end = self._table(total, [CENTER])
        self.flank_start, self.flank_end = self._table(total, [FLANK_UP, FLANK_DOWN])
        self.pile = np.zeros(total, dtype=np.int64)

    def _compressed(self, pos: np.ndarray) -> np.ndarray:
        i = np.searchsorted(self.first, pos, side="right") - 1
        return self.offset[i] + pos - self.first[i]

    def _table(self, total: int, regions: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
        starts = np.zeros(total + 1, dtype=np.int32)
        ends = np.zeros(total + 1, dtype=np.int32)
        for at, width in regions:
            rs = self.tss + 1 + at  # the region's 1-based start
            first = self._compressed(rs + 2)
            np.add.at(starts, first, 1)
            np.add.at(starts, self._compressed(rs + width - 2) + 1, -1)
            np.add.at(ends, first, 1)
            np.add.at(ends, self._compressed(rs + width + 1) + 1, -1)
        return np.cumsum(starts)[:-1].astype(np.int16), np.cumsum(ends)[:-1].astype(np.int16)

    def _where(self, pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        i = np.searchsorted(self.first, pos, side="right") - 1
        inside = np.flatnonzero((i >= 0) & (pos < self.stop[np.maximum(i, 0)]))
        i = i[inside]
        return inside, self.offset[i] + pos[inside] - self.first[i]

    def add(self, start: np.ndarray, end: np.ndarray, cell: np.ndarray, center: np.ndarray, flank: np.ndarray) -> None:
        n = center.size
        for pos, ctab, ftab in ((start, self.center_start, self.flank_start), (end, self.center_end, self.flank_end)):
            inside, comp = self._where(pos)
            if not inside.size:
                continue
            who = cell[inside]
            center += np.bincount(who, weights=ctab[comp], minlength=n)
            flank += np.bincount(who, weights=ftab[comp], minlength=n)
            self.pile += np.bincount(comp, minlength=self.pile.size)

    def profile(self) -> np.ndarray:
        """Cuts at each offset from -1 kb to +1 kb of these TSSs, read 5' to 3'."""
        out = np.zeros(2 * PROFILE_FLANK + 1, dtype=np.int64)
        steps = np.arange(2 * PROFILE_FLANK + 1)
        for minus in (False, True):
            tss = self.tss[self.minus == minus]
            if not tss.size:
                continue
            comps = self._compressed(tss - PROFILE_FLANK)[:, None] + steps
            part = self.pile[comps].sum(axis=0)
            out += part[::-1] if minus else part
        return out


@dataclass
class QcResult:
    cells: list[str]
    tss_enrichment: np.ndarray
    nucleosome_signal: np.ndarray
    n_fragments: np.ndarray
    per_cell_lengths: np.ndarray  # (cells, GROUP_LENGTH_BINS)
    length_counts: np.ndarray  # lengths 0..MAX_LENGTH; the last holds longer ones
    profile: np.ndarray
    n_tss: int
    n_lines: int
    seconds: float
    chromosomes_with_tss: list[str] = field(default_factory=list)


def score(center: np.ndarray, flank: np.ndarray) -> np.ndarray:
    """Signac TSSFast: center cuts per base over flank cuts per base; a cell without flank cuts
    takes all cells' mean flank."""
    flank_mean = flank / (FLANK_UP[1] + FLANK_DOWN[1])
    if flank_mean.size:
        flank_mean = np.where(flank == 0, flank_mean.mean(), flank_mean)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = center / flank_mean
    return np.nan_to_num(ratio, nan=0.0, posinf=0.0, neginf=0.0) / CENTER[1]


def iter_qc(path: str | Path, cells: Sequence[str], tss: pd.DataFrame, stop: Callable[[], bool] | None = None,
            ) -> Iterator[dict[str, Any]]:
    """The whole file once. Yields ``{"type": "progress", ...}`` every 2% of the file, then
    ``{"type": "result", "result": QcResult}``. ``stop`` is checked per block of text."""
    t0 = time.monotonic()
    size = os.path.getsize(path)
    codes = ff.Barcodes(cells)
    n = len(codes)
    chroms = {str(c): (g["tss"].to_numpy(dtype=np.int64), g["strand"].to_numpy() == "-")
              for c, g in tss.groupby("chrom", sort=False, observed=True)}
    n_fragments = np.zeros(n, dtype=np.int64)
    free = np.zeros(n, dtype=np.int64)
    mono = np.zeros(n, dtype=np.int64)
    center = np.zeros(n, dtype=np.float64)
    flank = np.zeros(n, dtype=np.float64)
    per_cell = np.zeros(n * GROUP_LENGTH_BINS, dtype=np.uint32)
    lengths = np.zeros(MAX_LENGTH + 1, dtype=np.int64)
    profile = np.zeros(2 * PROFILE_FLANK + 1, dtype=np.int64)
    track: _Track | None = None
    current: str | None = None
    seen_tss: list[str] = []
    n_lines = 0
    last = 0.0
    for done, text in ff.iter_text(path):
        if stop is not None and stop():
            raise FragmentsCancelled()
        p = ff.parse(text, codes)
        del text
        n_lines += p.n_lines
        if p.barcode.size:
            length = p.end - p.start
            n_fragments += np.bincount(p.barcode, minlength=n)
            free += np.bincount(p.barcode[length < NUCLEOSOME_FREE_MAX], minlength=n)
            mono += np.bincount(p.barcode[(length > MONO_LOW) & (length < MONO_HIGH)], minlength=n)
            lengths += np.bincount(np.minimum(np.maximum(length, 0), MAX_LENGTH), minlength=MAX_LENGTH + 1)
            np.add.at(per_cell, p.barcode * GROUP_LENGTH_BINS
                      + np.minimum(np.maximum(length, 0) // GROUP_LENGTH_BIN, GROUP_LENGTH_BINS - 1), 1)
            for name, a, b in p.chroms:
                if name != current:
                    if track is not None:
                        profile += track.profile()
                    current = name
                    track = None
                    if name in chroms:
                        track = _Track(*chroms[name])
                        seen_tss.append(name)
                if track is not None:
                    track.add(p.start[a:b], p.end[a:b], p.barcode[a:b], center, flank)
        fraction = done / size if size else 1.0
        if fraction - last >= PROGRESS_EVERY:
            last = fraction
            yield {"type": "progress", "fraction": round(fraction, 3),
                   "message": f"{n_lines:,} fragments read ({fraction:.0%} of the file)"}
    if track is not None:
        profile += track.profile()
    with np.errstate(divide="ignore", invalid="ignore"):
        signal = np.where(free > 0, mono / free, np.nan)
    n_tss = int(sum(chroms[c][0].size for c in seen_tss))
    yield {"type": "result", "result": QcResult(
        [str(c) for c in cells], score(center, flank).astype(np.float32), signal.astype(np.float32),
        n_fragments, per_cell.reshape(n, GROUP_LENGTH_BINS), lengths, profile, n_tss, n_lines,
        time.monotonic() - t0, seen_tss)}


# The same numbers in plain pandas, for the replay script (REPLAY_CODE: copied into it).


def gencode_tss(gtf):
    """Chromosome -> 0-based TSS of each protein-coding gene (GENCODE GTF, path or URL), without
    chrM and the PAR_Y copies: start on +, end - 1 on -."""
    rows = pd.read_csv(gtf, sep="\t", comment="#", header=None, usecols=[0, 2, 3, 4, 6, 8],
                       names=["chrom", "feature", "start", "end", "strand", "attributes"],
                       dtype={"chrom": str, "feature": str, "strand": str, "attributes": str})
    rows = rows[rows["feature"] == "gene"]
    rows = rows[rows["attributes"].str.contains('gene_type "protein_coding"', regex=False)
                & ~rows["attributes"].str.contains('_PAR_Y"', regex=False)
                & ~rows["chrom"].isin(["chrM", "MT", "Mt"])]
    tss = np.where(rows["strand"] == "-", rows["end"].to_numpy() - 1, rows["start"].to_numpy() - 1)
    return {str(c): tss[rows["chrom"].to_numpy() == c] for c in rows["chrom"].unique()}


def windows_holding(pos, tss, at, width, past):
    """How many of Signac's regions (1-based start = tss + 1 + at, ``width`` bases) count each raw
    start or end value: [rs + 2, rs + width + past]. See the module docstring."""
    first = np.sort(tss + 1 + at + 2)
    last = np.sort(tss + 1 + at + width + past)
    return np.searchsorted(first, pos, side="right") - np.searchsorted(last, pos, side="left")


def fragment_qc(path, cells, tss, chunk_lines=2_000_000):
    """Signac's TSS enrichment (fast) and nucleosome signal per cell, one pass over a fragments
    file; returns arrays in the order of ``cells``."""
    index = pd.Index(cells)
    n = len(cells)
    center, flank = np.zeros(n), np.zeros(n)
    free, mono, total = np.zeros(n), np.zeros(n), np.zeros(n, dtype=np.int64)
    reader = pd.read_csv(path, sep="\t", comment="#", header=None, usecols=[0, 1, 2, 3],
                         names=["chrom", "start", "end", "barcode"], chunksize=chunk_lines,
                         dtype={"chrom": str, "start": np.int64, "end": np.int64, "barcode": str})
    for chunk in reader:
        who = index.get_indexer(chunk["barcode"].to_numpy())
        keep = who >= 0
        who, chunk = who[keep], chunk[keep]
        length = (chunk["end"] - chunk["start"]).to_numpy()
        total += np.bincount(who, minlength=n)
        free += np.bincount(who[length < NUCLEOSOME_FREE_MAX], minlength=n)
        mono += np.bincount(who[(length > MONO_LOW) & (length < MONO_HIGH)], minlength=n)
        for chrom, rows in chunk.groupby("chrom", sort=False).indices.items():
            sites = tss.get(chrom)
            if sites is None:
                continue
            cell = who[rows]
            for col, past in (("start", -2), ("end", 1)):
                pos = chunk[col].to_numpy()[rows]
                center += np.bincount(cell, weights=windows_holding(pos, sites, CENTER[0], CENTER[1], past), minlength=n)
                for at, width in (FLANK_UP, FLANK_DOWN):
                    flank += np.bincount(cell, weights=windows_holding(pos, sites, at, width, past), minlength=n)
    flank_mean = flank / (FLANK_UP[1] + FLANK_DOWN[1])
    flank_mean = np.where(flank == 0, flank_mean.mean(), flank_mean)
    with np.errstate(divide="ignore", invalid="ignore"):
        score = np.nan_to_num(center / flank_mean, nan=0.0, posinf=0.0, neginf=0.0) / CENTER[1]
        signal = np.where(free > 0, mono / free, np.nan)
    return {"tss_enrichment": score, "nucleosome_signal": signal, "n_fragments": total}


REPLAY_CONSTANTS = ("CENTER", "FLANK_UP", "FLANK_DOWN", "NUCLEOSOME_FREE_MAX", "MONO_LOW", "MONO_HIGH")
REPLAY_CODE = (gencode_tss, windows_holding, fragment_qc)


def _fold_profile(profile: np.ndarray) -> list[float]:
    """Fold over the flanks (the 100 bp at each end), in 5 bp bins."""
    flanks = np.r_[profile[:100], profile[-100:]]
    base = flanks.mean() if flanks.size and flanks.mean() > 0 else 1.0
    n = profile.size // PROFILE_BIN
    binned = profile[:n * PROFILE_BIN].reshape(n, PROFILE_BIN).mean(axis=1) / base
    return [round(float(v), 3) for v in binned]


def _median(values: np.ndarray) -> float | None:
    v = values[np.isfinite(values)]
    return round(float(np.median(v)), 3) if v.size else None


def write_lengths(res: QcResult, data_dir: str | Path, dataset_id: str) -> None:
    """Per-cell lengths into the modality store, addressed by barcode, so per-group histograms
    still work after a filter. In a worker thread."""
    features = [f"{i * GROUP_LENGTH_BIN}-{(i + 1) * GROUP_LENGTH_BIN}" for i in range(GROUP_LENGTH_BINS)]
    modality_store.write(modality_store.store_path(data_dir, dataset_id, LENGTH_STORE),
                         {"lengths": sp.csr_matrix(res.per_cell_lengths)}, res.cells, features,
                         attrs={"bin": GROUP_LENGTH_BIN})


def save(adata, res: QcResult, *, genome: str | None, source: str | None, file_name: str | None,
         resources: dict | None = None) -> dict[str, Any]:
    """Store the result in ``adata`` (obs, uns JSON, a provenance step). The per-cell lengths
    are written apart (``write_lengths``)."""
    from scview.core import provenance

    if list(map(str, adata.obs_names)) != res.cells:
        raise FragmentsError("The dataset's cells changed while the QC ran (a pipeline run or a filter); nothing "
                             "was saved. Run it again.", 409)
    adata.obs[OBS_TSS] = res.tss_enrichment
    adata.obs[OBS_NUCLEOSOME] = res.nucleosome_signal
    adata.obs[OBS_N_FRAGMENTS] = res.n_fragments.astype(np.int64)
    created = datetime.now(timezone.utc).isoformat()
    with_any = int((res.n_fragments > 0).sum())
    medians = {"tss_enrichment": _median(res.tss_enrichment), "nucleosome_signal": _median(res.nucleosome_signal),
               "n_fragments": _median(res.n_fragments.astype(np.float64))}
    params = {"genome": genome, "annotation": source, "biotype": "protein_coding", "center_bp": CENTER[1],
              "flank_bp": FLANK_UP[1], "flank_reach_bp": abs(FLANK_UP[0]),
              "nucleosome_free_below": NUCLEOSOME_FREE_MAX, "mono_above": MONO_LOW, "mono_below": MONO_HIGH,
              "file": file_name}
    notes = []
    if medians["tss_enrichment"] is not None and medians["tss_enrichment"] < LOW_TSS:
        notes.append(f"The median TSS enrichment is only {medians['tss_enrichment']}: if the file was aligned to "
                     f"another build than {genome}, the TSS positions miss.")
    info = {
        "version": VERSION, "created_at": created, "n_cells": len(res.cells), "n_cells_with_fragments": with_any,
        "n_lines": int(res.n_lines), "n_fragments_in_cells": int(res.n_fragments.sum()), "medians": medians,
        "length_histogram": {"counts": res.length_counts.astype(np.int64).tolist(), "max": MAX_LENGTH,
                             "n": int(res.length_counts.sum())},
        "tss_profile": {"start": -PROFILE_FLANK, "bin": PROFILE_BIN, "fold": _fold_profile(res.profile),
                        "n_tss": res.n_tss, "chromosomes": len(res.chromosomes_with_tss)},
        "params": params, "seconds": round(res.seconds, 1), "notes": notes,
    }
    adata.uns[UNS_KEY] = json.dumps(info)
    provenance.record_step(adata, step=STEP, tool="scview.atac_fragments (= Signac TSSEnrichment fast=TRUE, "
                           "NucleosomeSignal on the whole file)", params=params, resources=resources)
    return info


def saved(adata) -> dict[str, Any] | None:
    """The saved QC without the per-cell values; None when there is none."""
    raw = adata.uns.get(UNS_KEY) if hasattr(adata, "uns") else None
    if raw is None:
        return None
    try:
        info = json.loads(raw if isinstance(raw, str) else str(raw))
    except ValueError:
        return None
    return info if isinstance(info, dict) else None


def _fmt(v: float | None, digits: int = 2) -> str:
    return "n/a" if v is None else f"{v:,.{digits}f}"


def grounding_lines(adata) -> tuple[list[str], str]:
    """Co-pilot facts: the QC medians, tagged [result:fragqc]; ``(lines, detail)``."""
    info = saved(adata)
    if not info:
        return [], ""
    m = info.get("medians", {})
    line = (f"- Median TSS enrichment {_fmt(m.get('tss_enrichment'))}, nucleosome signal "
            f"{_fmt(m.get('nucleosome_signal'))}, fragments per cell {_fmt(m.get('n_fragments'), 0)} across "
            f"{info.get('n_cells', 0):,} cells ({info.get('n_fragments_in_cells', 0):,} of the file's "
            f"{info.get('n_lines', 0):,} fragments are in cells) [{TAG}]")
    lines = [f"\n## ATAC fragment QC (Signac TSSEnrichment fast and NucleosomeSignal definitions, one pass over the "
             f"fragments file) [{TAG}]", line]
    lines.extend(f"- {n} [{TAG}]" for n in info.get("notes", []))
    return lines, f"median TSS enrichment {_fmt(m.get('tss_enrichment'))}"


def values_by_group(adata, group_by: str | None, max_per_group: int = 2000, seed: int = 0) -> dict[str, Any]:
    """Per-cell QC values for the histograms and violins, sampled to ``max_per_group`` per group."""
    if OBS_TSS not in adata.obs.columns:
        raise FragmentsError("Run the fragment QC first.", 404)
    obs = adata.obs
    labels = _labels(obs, group_by) if group_by else np.full(adata.n_obs, "All cells", dtype=object)
    rng = np.random.default_rng(seed)
    groups = []
    for name in _ordered(obs, group_by, labels):
        idx = np.flatnonzero(labels == name)
        n = idx.size
        if n > max_per_group:
            idx = np.sort(rng.choice(idx, max_per_group, replace=False))
        groups.append({"name": str(name), "n": int(n), "n_shown": int(idx.size),
                       "tss_enrichment": _rounded(obs[OBS_TSS].to_numpy()[idx]),
                       "nucleosome_signal": _rounded(obs[OBS_NUCLEOSOME].to_numpy()[idx]),
                       "n_fragments": _rounded(obs[OBS_N_FRAGMENTS].to_numpy()[idx], 0)})
    return {"group_by": group_by, "groups": groups, "max_per_group": max_per_group}


def _rounded(values: np.ndarray, digits: int = 3) -> list[float | None]:
    return [None if not np.isfinite(v) else round(float(v), digits) for v in np.asarray(values, dtype=float)]


def _labels(obs: pd.DataFrame, column: str) -> np.ndarray:
    if column not in obs.columns:
        raise FragmentsError(f"This dataset has no column '{column}': pick another group column.", 404)
    return obs[column].astype(object).where(obs[column].notna(), None).map(
        lambda v: None if v is None else str(v)).to_numpy(dtype=object)


def _ordered(obs: pd.DataFrame, column: str | None, labels: np.ndarray) -> list[str]:
    """Group names in the column's category order, else by name."""
    present = {v for v in labels.tolist() if v is not None}
    if column and isinstance(obs[column].dtype, pd.CategoricalDtype):
        return [str(c) for c in obs[column].cat.categories if str(c) in present]
    return sorted(present)


def lengths_by_group(adata, data_dir: str | Path, dataset_id: str, group_by: str | None) -> dict[str, Any]:
    """Fragment lengths (5 bp bins) per group, summed from the per-cell store on request. A
    cell the store does not hold counts for nothing."""
    path = modality_store.store_path(data_dir, dataset_id, LENGTH_STORE)
    if not path.exists():
        raise FragmentsError("Run the fragment QC first.", 404)
    obs = adata.obs
    labels = _labels(obs, group_by) if group_by else np.full(adata.n_obs, "All cells", dtype=object)
    names = _ordered(obs, group_by, labels) if group_by else ["All cells"]
    if len(names) > MAX_GROUPS:
        raise FragmentsError(f"'{group_by}' has {len(names)} groups, more than the {MAX_GROUPS} a length plot can "
                             "show: pick a column with fewer.")
    with modality_store.ModalityStore(path) as store:
        rows = store.rows_of(map(str, adata.obs_names))
        groups = []
        for name in names:
            where = rows[(labels == name) & (rows >= 0)]
            counts = np.zeros(store.n_vars, dtype=np.float64)
            for _, block in store.iter_blocks("lengths", np.sort(where)):
                counts += np.asarray(block.sum(axis=0)).ravel()
            total = counts.sum()
            groups.append({"name": str(name), "n_cells": int(where.size), "n_fragments": int(total),
                           "share": [round(float(c / total), 6) if total else 0.0 for c in counts]})
        bin_size = int(store.meta.get("bin", GROUP_LENGTH_BIN))
        n_bins = store.n_vars
    return {"group_by": group_by, "bin": bin_size, "x": [i * bin_size + bin_size / 2 for i in range(n_bins)],
            "groups": groups}


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------


_REGION = re.compile(r"^\s*([^:\s]+)\s*[:\-]\s*([\d,]+)\s*[-:]\s*([\d,]+)\s*$")


def parse_region(text: str) -> tuple[str, int, int]:
    """'chr1:1000-5000' (commas fine) to (chrom, start, end); 0-based half open, as the file."""
    m = _REGION.match(str(text))
    if not m:
        raise FragmentsError(f"Could not read the region '{text}': write it like chr1:1000-5000.")
    start, end = int(m.group(2).replace(",", "")), int(m.group(3).replace(",", ""))
    if end <= start:
        raise FragmentsError(f"The region '{text}' ends before it starts: write it like chr1:1000-5000.")
    return m.group(1), start, end


def gene_window(table: pd.DataFrame, gene: str) -> tuple[str, int, int, dict[str, Any]]:
    """A gene's body +- 5 kb, found by name, else id (any case)."""
    name = str(gene).strip()
    hits = table[table["gene_name"] == name]
    if hits.empty:
        low = name.lower()
        hits = table[(table["gene_name"].str.lower() == low) | (table["gene_id"].str.lower() == low)]
    if hits.empty:
        raise FragmentsError(f"The gene table has no gene '{name}': check the symbol, or give a region like "
                             "chr1:1000-5000.", 404)
    row = hits.iloc[0]
    return str(row["chrom"]), max(0, int(row["start"]) - GENE_PAD), int(row["end"]) + GENE_PAD, _gene_row(row)


def _gene_row(row) -> dict[str, Any]:
    return {"name": str(row["gene_name"]), "id": str(row["gene_id"]), "start": int(row["start"]),
            "end": int(row["end"]), "strand": str(row["strand"]), "type": str(row["gene_type"])}


def genes_in(table: pd.DataFrame | None, chrom: str, start: int, end: int, max_genes: int = 60) -> list[dict[str, Any]]:
    if table is None:
        return []
    hit = table[(table["chrom"] == chrom) & (table["start"] < end) & (table["end"] > start)]
    if len(hit) > max_genes:
        coding = hit[hit["gene_type"] == "protein_coding"]
        hit = coding if len(coding) else hit
        hit = hit.head(max_genes)
    return [_gene_row(r) for _, r in hit.sort_values("start").iterrows()]


_peaks: OrderedDict[tuple[str, float], dict[str, tuple[np.ndarray, np.ndarray]]] = OrderedDict()
_PEAK_RE = re.compile(r"^([^:\s]+)[:\-](\d+)-(\d+)$")


def _load_peaks(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    key = (str(path), path.stat().st_mtime)
    cached = _peaks.get(key)
    if cached is not None:
        _peaks.move_to_end(key)
        return cached
    with modality_store.ModalityStore(path) as store:
        names = None
        for column in ("interval", "peak", "id"):
            try:
                names = store.feature_attr(column)
                break
            except modality_store.ModalityStoreError:
                continue
        if names is None:
            names = store.features
    chroms, starts, ends = [], [], []
    for value in names:
        m = _PEAK_RE.match(str(value))
        if m:
            chroms.append(m.group(1))
            starts.append(int(m.group(2)))
            ends.append(int(m.group(3)))
    frame = pd.DataFrame({"chrom": chroms, "start": starts, "end": ends})
    out = {str(c): (g["start"].to_numpy(dtype=np.int64), g["end"].to_numpy(dtype=np.int64))
           for c, g in frame.groupby("chrom", sort=False)}
    _peaks[key] = out
    while len(_peaks) > 3:
        _peaks.popitem(last=False)
    return out


def peaks_in(data_dir: str | Path, dataset_id: str, chrom: str, start: int, end: int) -> list[dict[str, int]]:
    """Peaks in the region from the dataset's ``atac`` store; empty without one."""
    path = modality_store.store_path(data_dir, dataset_id, PEAK_STORE)
    if not path.exists():
        return []
    try:
        table = _load_peaks(path).get(chrom)
    except (OSError, modality_store.ModalityStoreError) as e:
        logger.warning("Could not read the peaks of %s: %s", dataset_id, e)
        return []
    if table is None:
        return []
    keep = (table[0] < end) & (table[1] > start)
    return [{"start": int(a), "end": int(b)} for a, b in zip(table[0][keep], table[1][keep])]


def _depth(starts: np.ndarray, ends: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """Fragments covering each base of [lo, hi)."""
    diff = np.zeros(hi - lo + 1, dtype=np.int64)
    np.add.at(diff, np.clip(starts - lo, 0, hi - lo), 1)
    np.add.at(diff, np.clip(ends - lo, 0, hi - lo), -1)
    return np.cumsum(diff)[:-1]


def coverage(reader: ff.FragmentReader, adata, chrom: str, start: int, end: int, *, group_by: str | None = None,
             groups: Sequence[str] | None = None, bins: int = DEFAULT_BINS) -> dict[str, Any]:
    """Depth per bin per group. Fragments are read once for the window; every base's depth
    is computed and averaged per bin, so a 2 Mb window costs about 16 MB per group."""
    if end - start > MAX_WINDOW:
        raise FragmentsError(f"That window is {(end - start) / 1e6:.2f} Mb; tracks are drawn for up to "
                             f"{MAX_WINDOW / 1e6:.0f} Mb at a time: ask for a narrower region.")
    if not reader.has_chrom(chrom):
        raise FragmentsError(f"The fragments file has no chromosome '{chrom}' (it has "
                             f"{', '.join(reader.chromosomes[:4])}, ...): check the region {chrom}:{start}-{end}.", 404)
    obs = adata.obs
    labels = _labels(obs, group_by) if group_by else np.full(adata.n_obs, "All cells", dtype=object)
    all_names = _ordered(obs, group_by, labels) if group_by else ["All cells"]
    if groups:
        wanted = {str(g) for g in groups}
        unknown = sorted(wanted - set(all_names))
        if unknown:
            raise FragmentsError(f"'{group_by or 'All cells'}' has no group '{unknown[0]}': pick one it has.", 404)
        names = [n for n in all_names if n in wanted]
    else:
        names = all_names
    notes: list[str] = []
    if len(names) > MAX_GROUPS:
        sizes = pd.Series(labels).value_counts()
        keep = set(sizes[sizes.index.isin(names)].head(MAX_GROUPS).index)
        names = [n for n in names if n in keep]
        notes.append(f"Showing the {MAX_GROUPS} largest groups of '{group_by}'.")
    bins = int(min(max(bins, 1), end - start, MAX_BINS))
    edges = np.linspace(start, end, bins + 1).round().astype(np.int64)
    positions = ((edges[:-1] + edges[1:]) / 2).tolist()
    code = {n: i for i, n in enumerate(names)}
    per_cell = np.fromiter((code.get(v, -1) for v in labels), dtype=np.int64, count=len(labels))
    fs, fe, cell, _ = reader.region(chrom, start, end)
    frag_group = per_cell[cell] if cell.size else np.zeros(0, dtype=np.int64)
    totals = (obs[OBS_N_FRAGMENTS].to_numpy(dtype=np.float64) if OBS_N_FRAGMENTS in obs.columns else None)
    tracks = []
    ymax = 0.0
    for name, i in code.items():
        mask = frag_group == i
        members = per_cell == i
        depth = _depth(fs[mask], fe[mask], start, end)
        summed = np.add.reduceat(depth, edges[:-1] - start)
        mean_depth = summed / np.diff(edges)
        n_cells = int(members.sum())
        total = float(totals[members].sum()) if totals is not None else None
        scale = 1e6 / total if total else (1e3 / n_cells if n_cells else 0.0)
        values = mean_depth * scale
        ymax = max(ymax, float(values.max(initial=0.0)))
        tracks.append({"name": name, "n_cells": n_cells, "n_fragments_in_window": int(mask.sum()),
                       "total_fragments": None if total is None else int(total),
                       "values": [round(float(v), 4) for v in values]})
    normal = ("Mean fragment depth per bin / the group's fragments in cells x 1e6 (depth per million fragments)."
              if totals is not None else
              "Mean fragment depth per bin per 1,000 cells: run the fragment QC to scale by each group's fragments.")
    if not fs.size:
        notes.append(f"No fragment of these cells overlaps {chrom}:{start:,}-{end:,}: try a wider region or another "
                     "chromosome.")
    return {"region": {"chrom": chrom, "start": int(start), "end": int(end)}, "bins": bins,
            "bin_size": round((end - start) / bins, 1), "x": positions, "group_by": group_by, "tracks": tracks,
            "ymax": round(ymax, 4), "normalisation": normal, "notes": notes, "n_fragments": int(fs.size),
            "max_window": MAX_WINDOW}
