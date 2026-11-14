"""Motifs in ATAC peaks: which TFs explain the chromatin differences between groups (Signac's
motif vignette). ``core/tf_motifs.py`` scans 1 kb promoters of marker genes; this scans every peak.

**Scan** (``build``, a child process, cached per peak set in ``DATA_DIR/atac_motifs/<genome>``):
the 879 JASPAR 2024 matrices over the peaks' bases from ``core/genome_seq.py``, the result
motifmatchr gives through Signac's ``AddMotifs`` (motifmatchr 1.34.0, whose ``matchMotifs``
defaults are p.cutoff 5e-5, bg "subject", both strands):

- ``pwm``: TFBSTools' ``toPWM`` (log2 of (count + 0.8 x 0.25) / (N + 0.8) over 0.25), then
  motifmatchr's ``convert_pwm``, which SUBTRACTS log2(0.25) - log2(bg). That is motifmatchr's
  sign (a correct background swap would add it); the words it ranks highest differ, so it is
  kept: the point is the same hits.
- ``threshold``: MOODS' ``threshold_from_p``. Weights rounded to 1/2000, an exact distribution
  of the rounded scores under the peaks' base frequencies (as they read, not strand-averaged),
  the lowest rounded score whose tail is at most p. A matrix whose best word alone is more likely
  than p (every 5-mer and most 6-mers) gets a threshold just under its best score, so it hits
  on that word: 40 to 80% of peaks. The reverse strand uses the forward strand's threshold.
- A hit is a window wholly inside a stretch of known bases of one peak whose real-valued score
  reaches the threshold. Peaks on a contig the 2bit file does not name have no bases: no hits,
  a NaN GC, left out of everything here and counted. MOODS leaves out one window of a matrix
  shorter than 7 (flush with the end of a stretch), and so does this.

No rectangular matrix product, unlike ``tf_motifs.scan``, because peaks vary in length and because
one is 30 times slower here. ``scan_block`` concatenates a block of peaks with an unknown base
between them, sorts the positions by their 7-mer, and asks per matrix and strand only for the
7-mers that could still reach the threshold in the matrix's best window; the survivors are scored
in numpy, the columns that cut most first. Exact: the final score is the plain sum over the
columns, in order, so ``scan_plain`` (every window; tests) gives the same bits.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from scview.core import genome_seq, memory_budget, tf_motifs

logger = logging.getLogger(__name__)

SUBDIR = "atac_motifs"
MOTIF_P_CUTOFF = 5e-5  # motifmatchr::matchMotifs p.cutoff
PSEUDOCOUNT = 0.8  # TFBSTools::toPWM pseudocounts
DP_SCALE = 2000.0  # MOODS' PVAL_DP_MULTIPLIER
INDEX_K = 7  # MOODS' window, motifmatchr's w = 7
BLOCK_BASES = 6_000_000  # one block's bases: about 150 MB of index
BUILD_THREADS = 2
FORMAT = 1
_N = genome_seq.UNKNOWN
_EPS = 1e-9  # slack while pruning; the final comparison is exact


class ScanError(RuntimeError):
    """The sentence says what is missing and what to do."""


class ScanCancelled(Exception):
    """A stop request ended the scan; nothing is kept."""


# ---------------------------------------------------------------------------
# Matrices and thresholds
# ---------------------------------------------------------------------------


def pwm(counts, bg):
    """The 4 x L matrix motifmatchr scans for a JASPAR PFM against base frequencies ``bg``."""
    counts = np.asarray(counts, dtype=np.float64)
    bg = np.asarray(bg, dtype=np.float64)
    even = 0.25
    p = (counts + PSEUDOCOUNT * even) / (counts.sum(axis=0) + PSEUDOCOUNT)
    return np.log2(p / even) - (np.log2(even) - np.log2(bg))[:, None]


def motif_threshold(w, bg, p=MOTIF_P_CUTOFF):
    """MOODS' ``threshold_from_p`` for a 4 x L matrix (module docstring)."""
    w = np.asarray(w, dtype=np.float64)
    bg = np.asarray(bg, dtype=np.float64)
    rounded = np.where(w > 0, np.trunc(DP_SCALE * w + 0.5), np.trunc(DP_SCALE * w - 0.5)).astype(np.int64)
    mins = rounded.min(axis=0)
    rel = rounded - mins
    dist = np.ones(1)
    for j in range(rel.shape[1]):
        new = np.zeros(dist.size + int(rel[:, j].max()))
        for b in range(4):
            s = int(rel[b, j])
            new[s:s + dist.size] += dist * bg[b]
        dist = new
    tail = np.cumsum(dist[::-1])  # P(score >= highest - i)
    if tail[0] > p:
        tops = np.sort(w, axis=0)
        delta = np.inf
        for j in range(w.shape[1]):  # the smallest gap between a column's best and second best
            below = tops[:, j][tops[:, j] < tops[-1, j]]
            if below.size:
                delta = min(delta, tops[-1, j] - below[-1])
        return float(w.max(axis=0).sum() - delta / 2)
    over = np.flatnonzero(tail > p)
    if over.size == 0:
        return float(w.min(axis=0).sum() - 1.0)
    r = dist.size - 1 - int(over[0])
    return float((r + int(mins.sum()) + 1) / DP_SCALE)


def reverse_complement(w):
    return np.asarray(w)[::-1, ::-1]


@dataclass(frozen=True)
class Strand:
    """One matrix on one strand, ready for ``scan_block``."""

    motif: int
    w: np.ndarray  # 4 x L
    t: float
    k: int  # the index it asks: 7, or its length when shorter
    at: int  # its window's first column
    cand: np.ndarray  # k-mers that can still reach ``t``
    cand_score: np.ndarray  # their score in the window
    cols: np.ndarray  # the other columns, the ones that cut most first
    rest: np.ndarray  # the best the columns after each of ``cols`` can add


def _window_scores(w, at, k):
    s = w[:, at]
    for j in range(1, k):
        s = np.add.outer(s, w[:, at + j]).ravel()
    return s


def prepare(motif, w, t):
    """The window whose k-mers leave the fewest candidates, and those k-mers."""
    length = w.shape[1]
    k = min(INDEX_K, length)
    best = w.max(axis=0)
    total = best.sum()
    chosen = None
    for at in range(length - k + 1):
        scores = _window_scores(w, at, k)
        other = total - best[at:at + k].sum()
        mask = scores + other >= t - _EPS
        n = int(mask.sum())
        if chosen is None or n < chosen[0]:
            chosen = (n, at, np.flatnonzero(mask).astype(np.int32), scores[mask])
    _, at, cand, cand_score = chosen
    others = np.array([j for j in range(length) if not at <= j < at + k], dtype=np.int64)
    if others.size:
        others = others[np.argsort(-(best[others] - w[:, others].mean(axis=0)), kind="stable")]
    rest = np.concatenate([np.cumsum(best[others][::-1])[::-1][1:], [0.0]]) if others.size else np.zeros(0)
    return Strand(motif, w, t, k, at, cand, cand_score, others, rest)


def prepare_all(counts, bg, p=MOTIF_P_CUTOFF):
    """Both strands of every matrix, and each matrix's threshold."""
    strands, thresholds = [], np.zeros(len(counts))
    for i, c in enumerate(counts):
        w = pwm(c, bg)
        t = motif_threshold(w, bg, p)
        thresholds[i] = t
        strands.append(prepare(i, w, t))
        strands.append(prepare(i, reverse_complement(w), t))
    return strands, thresholds


# ---------------------------------------------------------------------------
# The scan
# ---------------------------------------------------------------------------


def join_regions(codes, offsets):
    """Regions end to end with an unknown base after each: (codes, each region's start)."""
    n = offsets.size - 1
    starts = offsets[:-1] + np.arange(n)
    out = np.full(codes.size + n, _N, dtype=np.uint8)
    mask = np.ones(out.size, dtype=bool)
    if n:
        mask[offsets[1:] + np.arange(n)] = False
    out[mask] = codes
    return out, starts


def _kmer_index(codes, bad, k):
    """Positions whose k bases are all known, sorted by k-mer, and where each k-mer starts."""
    n = codes.size - k + 1
    if n <= 0:
        return np.zeros(0, np.int32), np.zeros(4 ** k + 1, np.int64)
    kmers = np.zeros(n, dtype=np.uint16)
    for j in range(k):
        kmers = (kmers << 2) | (codes[j:j + n] & 3)
    pos = np.flatnonzero(bad[k:k + n] == bad[:n]).astype(np.int32)
    kmers = kmers[pos]
    order = np.argsort(kmers, kind="stable")  # radix sort on 16 bits
    starts = np.zeros(4 ** k + 1, dtype=np.int64)
    starts[1:] = np.cumsum(np.bincount(kmers, minlength=4 ** k))
    return pos[order], starts


def _scan_strand(s, codes, bad, sorted_pos, kstarts):
    """The windows of this strand that reach its threshold: their first positions."""
    first = kstarts[s.cand]
    lens = kstarts[s.cand + 1] - first
    total = int(lens.sum())
    if total == 0:
        return np.zeros(0, np.int64)
    ends = np.cumsum(lens)
    idx = np.repeat(first - (ends - lens), lens) + np.arange(total)
    anchors = sorted_pos[idx].astype(np.int64) - s.at
    score = np.repeat(s.cand_score, lens)
    length = s.w.shape[1]
    ok = (anchors >= 0) & (anchors + length < codes.size)
    anchors, score = anchors[ok], score[ok]
    ok = bad[anchors + length] == bad[anchors]
    if length < INDEX_K:  # MOODS: a short matrix's window may not touch the end of a stretch
        ok &= codes[anchors + length] != _N
    anchors, score = anchors[ok], score[ok]
    for col, rest in zip(s.cols, s.rest):
        if not anchors.size:
            return anchors
        score += s.w[codes[anchors + col], col]
        ok = score + rest >= s.t - _EPS
        anchors, score = anchors[ok], score[ok]
    if not anchors.size:
        return anchors
    final = np.zeros(anchors.size)
    for col in range(length):
        final += s.w[codes[anchors + col], col]
    return anchors[final >= s.t]


def scan_block(codes, starts, strands, n_motifs, *, threads=1, stop=lambda: False):
    """Regions x motifs hits of one block (``join_regions``'s output)."""
    bad = np.zeros(codes.size + 1, dtype=np.int32)
    bad[1:] = np.cumsum(codes == _N)
    hits = np.zeros((starts.size, n_motifs), dtype=bool)
    index = {k: _kmer_index(codes, bad, k) for k in sorted({s.k for s in strands})}

    def work(s):
        if stop():
            raise ScanCancelled
        anchors = _scan_strand(s, codes, bad, *index[s.k])
        if anchors.size:
            hits[np.searchsorted(starts, anchors, side="right") - 1, s.motif] = True

    if threads > 1:
        with ThreadPoolExecutor(threads) as pool:
            list(pool.map(work, strands, chunksize=8))
    else:
        for s in strands:
            work(s)
    return hits


def scan_plain(codes: np.ndarray, starts: np.ndarray, counts: list[np.ndarray], bg: np.ndarray,
               p: float = MOTIF_P_CUTOFF) -> np.ndarray:
    """Every window of every matrix, no index: the definition ``scan_block`` must match."""
    hits = np.zeros((starts.size, len(counts)), dtype=bool)
    bad = np.zeros(codes.size + 1, dtype=np.int32)
    bad[1:] = np.cumsum(codes == _N)
    for i, c in enumerate(counts):
        w = pwm(c, bg)
        t = motif_threshold(w, bg, p)
        length = w.shape[1]
        n = codes.size - length
        if n <= 0:
            continue
        pos = np.flatnonzero(bad[length:length + n] == bad[:n])
        if length < INDEX_K:
            pos = pos[codes[pos + length] != _N]
        for m in (w, reverse_complement(w)):
            score = np.zeros(pos.size)
            for col in range(length):
                score += m[codes[pos + col], col]
            anchors = pos[score >= t]
            hits[np.searchsorted(starts, anchors, side="right") - 1, i] = True
    return hits


# ---------------------------------------------------------------------------
# The peaks x motifs table
# ---------------------------------------------------------------------------


@dataclass
class PeakMotifTable:
    genome: str
    key: str
    motif_ids: np.ndarray
    motif_names: np.ndarray
    lengths: np.ndarray
    thresholds: np.ndarray
    gc_motif: np.ndarray  # information-weighted GC of each motif (tf_motifs.gc_content)
    packed: np.ndarray  # peaks x motifs, bit-packed
    gc: np.ndarray  # each peak's 100 x (G + C) / length, NaN for no known base
    background: np.ndarray
    meta: dict = field(default_factory=dict)

    @property
    def n_peaks(self) -> int:
        return int(self.packed.shape[0])

    @property
    def n_motifs(self) -> int:
        return int(self.motif_ids.size)

    @property
    def known(self) -> np.ndarray:
        return np.isfinite(self.gc)

    @property
    def gc_rich(self) -> np.ndarray:
        return self.gc_motif > tf_motifs.GC_RICH

    def hits(self, peaks: np.ndarray) -> np.ndarray:
        """These peaks x every motif, as bools."""
        return np.unpackbits(self.packed[peaks], axis=1, count=self.n_motifs).astype(bool)

    def counts(self, peaks: np.ndarray, block: int = 16_384) -> np.ndarray:
        """Per motif, how many of these peaks have a hit."""
        out = np.zeros(self.n_motifs, dtype=np.int64)
        for start in range(0, peaks.size, block):
            out += self.hits(peaks[start:start + block]).sum(axis=0)
        return out


def peak_digest(chrom: Iterable, start: np.ndarray, end: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update("\n".join(str(c) for c in chrom).encode())
    h.update(np.asarray(start, dtype=np.int64).tobytes())
    h.update(np.asarray(end, dtype=np.int64).tobytes())
    return h.hexdigest()


def cache_key(genome: str, peaks: str) -> str:
    """Names one scan's result: the peak set, the two input files, these parameters."""
    doc = {"genome": genome_seq.RELEASES[genome].sha256, "jaspar": tf_motifs.JASPAR.sha256, "peaks": peaks,
           "p": MOTIF_P_CUTOFF, "pseudocount": PSEUDOCOUNT, "dp": DP_SCALE, "format": FORMAT}
    return hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest()[:24]


def table_path(data_dir: str | Path, genome: str, key: str) -> Path:
    return Path(data_dir) / SUBDIR / genome / f"hits_{key}.npz"


def save_table(t: PeakMotifTable, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.part.npz")
    np.savez(tmp, genome=np.str_(t.genome), key=np.str_(t.key), motif_ids=t.motif_ids, motif_names=t.motif_names,
             lengths=t.lengths, thresholds=t.thresholds, gc_motif=t.gc_motif, packed=t.packed, gc=t.gc,
             background=t.background, meta=np.str_(json.dumps(t.meta)))
    os.replace(tmp, path)


def read_table_file(path: Path) -> PeakMotifTable:
    with np.load(path, allow_pickle=False) as z:
        return PeakMotifTable(genome=str(z["genome"]), key=str(z["key"]), motif_ids=z["motif_ids"], motif_names=z["motif_names"],
                              lengths=z["lengths"], thresholds=z["thresholds"], gc_motif=z["gc_motif"], packed=z["packed"],
                              gc=z["gc"], background=z["background"], meta=json.loads(str(z["meta"])))


_tables: dict[str, tuple[float, PeakMotifTable]] = {}
_tables_lock = threading.Lock()


def load_table(data_dir: str | Path, genome: str, key: str) -> PeakMotifTable | None:
    """The scan's result (the last two kept in memory while their files are unchanged), or None."""
    path = table_path(data_dir, genome, key)
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return None
    with _tables_lock:
        hit = _tables.get(str(path))
        if hit and hit[0] == mtime:
            return hit[1]
    try:
        table = read_table_file(path)
    except Exception as e:  # noqa: BLE001 - an unreadable file is scanned again
        logger.warning("Unreadable motif table %s: %s", path, e)
        return None
    with _tables_lock:
        while len(_tables) >= 2:
            _tables.pop(next(iter(_tables)))
        _tables[str(path)] = (mtime, table)
    return table


def jaspar_path(data_dir: str | Path) -> Path:
    return tf_motifs.raw_dir(Path(data_dir) / "tf_motifs") / tf_motifs.JASPAR.filename


def read_jaspar(data_dir: str | Path) -> list[tf_motifs.Motif]:
    return tf_motifs.parse_jaspar(jaspar_path(data_dir).read_text())


def _blocks(lengths, block_bases):
    """Runs of peaks of about ``block_bases`` bases each."""
    out = []
    i, n = 0, lengths.size
    cum = np.cumsum(lengths)
    while i < n:
        base = cum[i - 1] if i else 0
        j = int(np.searchsorted(cum, base + block_bases, side="left")) + 1
        out.append((i, min(n, max(j, i + 1))))
        i = out[-1][1]
    return out


def _best_word_only(w: np.ndarray, bg: np.ndarray, p: float = MOTIF_P_CUTOFF) -> bool:
    """Whether the matrix's best word alone is more likely than p (it then hits on that word)."""
    return float(np.prod(bg[w.argmax(axis=0)])) > p


def build(chrom: np.ndarray, start: np.ndarray, end: np.ndarray, motifs: list[tf_motifs.Motif], genome_file: genome_seq.TwoBit,
          genome: str, key: str, *, progress: Callable[[dict], None] = lambda _e: None, stop: Callable[[], bool] = lambda: False,
          threads: int = BUILD_THREADS, block_bases: int = BLOCK_BASES) -> PeakMotifTable:
    """Read the peaks' bases (two passes: the base frequencies come first), scan them."""
    t0 = time.time()
    n = chrom.size
    valid = (start >= 0) & (end > start)
    in_genome = np.fromiter((genome_file.has(str(c)) for c in chrom), dtype=bool, count=n) & valid
    blocks = _blocks(np.where(valid, end - start, 0), block_bases)
    gc = np.full(n, np.nan, dtype=np.float64)
    totals = np.zeros(4, dtype=np.float64)
    n_unknown = 0
    progress({"stage": "read", "done": 0, "total": n, "message": f"Reading the bases of {n:,} peaks"})
    for i, j in blocks:
        if stop():
            raise ScanCancelled
        idx = i + np.flatnonzero(in_genome[i:j])
        codes, offsets = genome_seq.sequences(genome_file, chrom[idx], start[idx], end[idx])
        counts = genome_seq.base_counts(codes, offsets)
        gc[idx] = genome_seq.gc_percent(counts)
        totals += counts[:, :4].sum(axis=0)
        n_unknown += int(counts[:, 4].sum())
    bg = totals / totals.sum() if totals.sum() > 0 else np.full(4, 0.25)
    progress({"stage": "prepare", "message": f"Preparing {len(motifs)} motifs"})
    strands, thresholds = prepare_all([m.counts for m in motifs], bg)
    n_cols = len(motifs)
    packed = np.zeros((n, (n_cols + 7) // 8), dtype=np.uint8)
    for i, j in blocks:
        if stop():
            raise ScanCancelled
        idx = i + np.flatnonzero(in_genome[i:j])
        if idx.size:
            codes, offsets = genome_seq.sequences(genome_file, chrom[idx], start[idx], end[idx])
            joined, starts = join_regions(codes, offsets)
            packed[idx] = np.packbits(scan_block(joined, starts, strands, n_cols, threads=threads, stop=stop), axis=1)
        progress({"stage": "scan", "done": j, "total": n, "seconds": round(time.time() - t0, 1),
                  "message": f"Scanned {j:,} of {n:,} peaks"})
    n_scanned = int(np.isfinite(gc).sum())
    n_hits = int(_BIT_COUNT[packed].sum())
    meta = {"genome": genome, "n_peaks": n, "n_motifs": n_cols, "n_scanned": n_scanned, "n_left_out": n - n_scanned,
            "n_unknown_bases": n_unknown, "p_cutoff": MOTIF_P_CUTOFF, "pseudocount": PSEUDOCOUNT,
            "background": [round(float(b), 5) for b in bg], "n_hits": n_hits,
            "hit_rate": round(n_hits / max(1, n_scanned * n_cols), 4),
            "n_best_word_only": int(sum(_best_word_only(pwm(m.counts, bg), bg) for m in motifs)),
            "seconds": round(time.time() - t0, 1)}
    return PeakMotifTable(
        genome=genome, key=key, motif_ids=np.asarray([m.id for m in motifs], dtype=str),
        motif_names=np.asarray([m.name for m in motifs], dtype=str),
        lengths=np.asarray([m.counts.shape[1] for m in motifs], dtype=np.int32), thresholds=thresholds,
        gc_motif=np.array([tf_motifs.gc_content(m.counts)[0] for m in motifs]), packed=packed, gc=gc, background=bg, meta=meta)


_BIT_COUNT = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(axis=1)

# At 2 cores through the server: PBMC 10k (143,887 peaks, 121.9 Mb) 111 s, 1.01 GB of cgroup anon over
# what the server held; mouse E18 (172,193 peaks, 146.2 Mb) 124 s.
_SCAN_SECONDS_PER_BASE = 0.85e-6
_SCAN_FIRST_SECONDS = 6.0
_SCAN_BASE_BYTES = 150 * 1024**2
_SCAN_BLOCK_BYTES = 820 * 1024**2
_SCAN_BYTES_PER_PEAK = 240  # gc and the packed hits


def estimate_scan_seconds(n_bases: int) -> float:
    return _SCAN_FIRST_SECONDS + _SCAN_SECONDS_PER_BASE * n_bases


def estimate_scan_bytes(n_peaks: int) -> int:
    return _SCAN_BASE_BYTES + _SCAN_BLOCK_BYTES + _SCAN_BYTES_PER_PEAK * n_peaks


# ---------------------------------------------------------------------------
# The child process, and builds that outlive the request that started them
# ---------------------------------------------------------------------------


def thread_env(threads: int) -> dict[str, str]:
    env = dict(os.environ)
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
        env[var] = str(threads)
    return env


def iter_child(cmd: list[str], stop: Callable[[], bool] | None, threads: int, what: str) -> Iterator[dict]:
    """A child process's JSON progress lines, its ``{"type": "done"}`` line last. ``stop`` kills it, and
    so does closing the generator. ``ScanCancelled`` when stopped, ``ScanError`` when it fails."""
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=thread_env(threads), text=True)
    killed = threading.Event()

    def watch() -> None:
        while proc.poll() is None:
            if stop is not None and stop():
                killed.set()
                proc.kill()
                return
            time.sleep(0.2)

    threading.Thread(target=watch, daemon=True).start()
    errors: list[str] = []
    assert proc.stderr is not None and proc.stdout is not None
    stderr = proc.stderr
    drain = threading.Thread(target=lambda: errors.extend(stderr.readlines()[-3:]), daemon=True)
    drain.start()
    final: dict | None = None
    try:
        for line in proc.stdout:
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if event.get("type") == "done":
                final = event
            else:
                yield event
        proc.wait()
        drain.join(timeout=5)
        if killed.is_set():
            raise ScanCancelled
        if proc.returncode != 0 or final is None:
            tail = " ".join(x.strip() for x in errors)
            raise ScanError(f"{what} failed (exit {proc.returncode}): {tail}")
        yield final
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def run_child(cmd: list[str], on_event: Callable[[dict], None] | None, stop: Callable[[], bool] | None,
              threads: int, what: str) -> dict:
    """``iter_child``, its progress lines to ``on_event``; returns its ``done`` line."""
    final: dict = {}
    for event in iter_child(cmd, stop, threads, what):
        if event.get("type") == "done":
            final = event
        elif on_event is not None:
            on_event({k: v for k, v in event.items() if k != "type"})
    return final


def peak_coordinates(store) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(chrom, start, end) per peak of a store; start and end -1 where a peak names no interval."""
    def number(name: str) -> np.ndarray:
        out = np.full(store.n_vars, -1, dtype=np.int64)
        for i, v in enumerate(store.feature_attr(name)):
            try:
                out[i] = int(v)
            except (TypeError, ValueError):
                pass
        return out

    return store.feature_attr("chrom"), number("start"), number("end")


def scan_cmd(store: Path, jaspar: Path, two_bit: Path, out: Path, genome: str, key: str) -> list[str]:
    return [sys.executable, "-m", "scview.core.atac_motifs", "--store", str(store), "--jaspar", str(jaspar),
            "--two-bit", str(two_bit), "--out", str(out), "--genome", genome, "--key", key]


def _child(argv: list[str]) -> None:
    import argparse
    from scview.core.modality_store import ModalityStore

    ap = argparse.ArgumentParser()
    for a in ("--store", "--jaspar", "--two-bit", "--out", "--genome", "--key"):
        ap.add_argument(a, required=True)
    args = ap.parse_args(argv)
    last = [0.0]

    def emit(e: dict) -> None:
        now = time.time()
        if e.get("stage") == "scan" and e.get("done") != e.get("total") and now - last[0] < 1.0:
            return
        last[0] = now
        sys.stdout.write(json.dumps({"type": "progress", **e}) + "\n")
        sys.stdout.flush()

    with ModalityStore(args.store) as store:
        chrom, start, end = peak_coordinates(store)
    motifs = tf_motifs.parse_jaspar(Path(args.jaspar).read_text())
    table = build(chrom, start, end, motifs, genome_seq.TwoBit(args.two_bit), args.genome, args.key, progress=emit)
    table.meta.update(peak_mb=memory_budget.own_peak_mb(), threads=BUILD_THREADS)
    save_table(table, Path(args.out))
    sys.stdout.write(json.dumps({"type": "done", **table.meta}) + "\n")
    sys.stdout.flush()


@dataclass
class Build:
    key: str
    genome: str
    events: list = field(default_factory=list)
    done: bool = False
    error: str | None = None
    cancelled: bool = False
    started: float = field(default_factory=time.time)
    stop: threading.Event = field(default_factory=threading.Event)

    def add(self, event: dict) -> None:
        self.events.append(event)

    @property
    def last(self) -> dict | None:
        return self.events[-1] if self.events else None


_builds: dict[str, Build] = {}
_builds_lock = threading.Lock()


def current_build(key: str) -> Build | None:
    with _builds_lock:
        b = _builds.get(key)
    return b if b is not None and not b.done else None


def ensure_jaspar(data_dir: str | Path, auto_download: bool, say: Callable[[dict], None]) -> Path:
    """The pinned JASPAR file the promoter table uses too. ``ScanError`` saying what to do."""
    dest = Path(data_dir) / "tf_motifs"
    path = jaspar_path(data_dir)
    src = tf_motifs.JASPAR
    if path.is_file():
        if not tf_motifs.file_ok(dest, src):
            raise ScanError(f"{path} is not the pinned JASPAR file (sha256 differs); delete it to fetch it again.")
        return path
    if not auto_download:
        raise ScanError(f"The JASPAR motifs are not downloaded and TF_MOTIFS_AUTO_DOWNLOAD is off. Place {src.filename} "
                        f"(from {src.url}) in {path.parent}, or turn the switch on.")
    say({"stage": "download", "message": f"Downloading {src.name} ({src.size / 1e6:.1f} MB)"})
    try:
        tf_motifs.download(dest, src)
    except Exception as e:  # noqa: BLE001 - one sentence to the client
        raise ScanError(f"Could not download the JASPAR motifs ({e}). Check the network and try again, or place "
                        f"{src.filename} in {path.parent}.") from e
    return path


def start_build(data_dir: str | Path, genome: str, key: str, store_path: Path, *, genome_auto: bool = True,
                jaspar_auto: bool = True, on_end: Callable[[], None] | None = None) -> Build:
    """The running scan of this peak set, or a new one in a background thread (downloads, then
    the child process). It keeps going when the client that asked goes away; a second request
    joins it. ``Build.stop`` cancels it. ``on_end`` runs when it ends, only for a new one."""
    with _builds_lock:
        b = _builds.get(key)
        if b is not None and not b.done:
            return b
        b = Build(key, genome)
        _builds[key] = b

    def _run() -> None:
        try:
            jaspar = ensure_jaspar(data_dir, jaspar_auto, b.add)
            genome_seq.ensure(data_dir, genome, auto_download=genome_auto, progress=b.add, stop=b.stop.is_set)
            final = run_child(scan_cmd(store_path, jaspar, genome_seq.file_path(data_dir, genome),
                                       table_path(data_dir, genome, key), genome, key),
                              b.add, b.stop.is_set, BUILD_THREADS, "The motif scan")
            b.add({"stage": "done", "message": f"Scanned in {final.get('seconds')} s", **{k: final.get(k) for k in ("seconds", "peak_mb")}})
        except (ScanCancelled, genome_seq.GenomeCancelled):
            b.cancelled = True
        except (ScanError, genome_seq.GenomeUnavailable) as e:
            b.error = str(e)
        except Exception as e:  # noqa: BLE001 - reported to the client
            logger.warning("Motif scan %s failed: %s", key, e, exc_info=True)
            b.error = f"The motif scan failed: {e}"
        finally:
            b.done = True
            if on_end is not None:
                on_end()

    threading.Thread(target=_run, name=f"atac-motifs-{key[:8]}", daemon=True).start()
    return b


# ---------------------------------------------------------------------------
# Enrichment in a group's differential peaks (Signac's FindMotifs)
# ---------------------------------------------------------------------------

BACKGROUND_PEAKS = 40_000  # FindMotifs' background
QUERY_P = 0.005  # Signac's motif vignette: p_val < 0.005 & pct.1 > 0.2, from only.pos = TRUE
# Open in more than 5% of the group (FindMarkers' min.pct), not the vignette's 20%: on nuclei at ~15k
# fragments a cell (5k mouse kidney, v0.78) 20% kept 79-189 peaks for the minor types, 57-89% of them
# promoters, and GC-rich KLF/SP motifs topped every type but PT; at 5% the lineage TF is first for PT
# (HNF4A), TAL (ESRRA/B/G), DCT/CNT and IC (TFCP2L1) and podocytes (WT1). PBMC 10k (17k fragments a
# cell) loses nothing: CEBPA, SPIB, POU2F, BCL11B/RUNX stay first and NK's TBX21 moves from 2nd to 1st.
QUERY_PCT = 0.05
PROMOTER_HEAVY = 0.6  # a query with more promoters than this gets a note
QUERY_LOG2FC = 0.0
MIN_QUERY = 10  # FindMotifs warns under this
SEED = 0
RESULT_KIND = "atac_motif_enrichment"
STEP_ENRICH = "atac_motif_enrichment"
TAG = "result:atacmotifs"
CITATION = ("JASPAR 2024 (Rauluseviciute et al. 2024, Nucleic Acids Res, PMID 37962376; CC BY 4.0); "
            "Signac (Stuart et al. 2021, Nat Methods, PMID 34725479); motifmatchr / MOODS")
CAVEATS = ("A motif hit is not binding. Paralogous TFs share near-identical motifs, so the TF named stands for its family; "
           "whether it is expressed in the group separates some of them. GC-rich motifs enrich easily even after matching "
           "the background on GC.")
ENRICH_METHOD = ("Signac's FindMotifs: {n_background:,} background peaks drawn without replacement from the peaks outside the "
                 "query, weighted to match the query's GC content (density ratio, bandwidth 1%); one-sided hypergeometric, "
                 "BH across the {n_motifs} motifs; fold enrichment = % of query peaks with a hit / % of background peaks.")


class EnrichmentError(ValueError):
    """A run that cannot start: the sentence says what to do."""


def _r_density(x, bw=1.0, n=512):
    """R's ``density(x, bw = bw)``, gaussian: linear binning on a grid of 2n, FFT, n points."""
    lo0, hi0 = float(x.min()) - 3 * bw, float(x.max()) + 3 * bw
    lo, up = lo0 - 4 * bw, hi0 + 4 * bw
    delta = (up - lo) / (n - 1)
    pos = (x - lo) / delta
    ix = np.floor(pos).astype(np.int64)
    fx = pos - ix
    y = (np.bincount(ix, weights=1 - fx, minlength=2 * n) + np.bincount(ix + 1, weights=fx, minlength=2 * n))[:2 * n] / x.size
    grid = np.linspace(0, 2 * (up - lo), 2 * n)
    grid[n + 1:] = -grid[n - 1:0:-1]
    kernel = np.exp(-0.5 * (grid / bw) ** 2) / (bw * np.sqrt(2 * np.pi))
    conv = np.maximum(0.0, np.fft.ifft(np.fft.fft(y) * np.conj(np.fft.fft(kernel))).real[:n])
    xout = np.linspace(lo0, hi0, n)
    return xout, np.interp(xout, np.linspace(lo, up, n), conv)


def match_background(gc, query, n, rng):
    """MatchRegionStats on GC: ``n`` peaks outside the query, without replacement, each with
    weight (the query's GC density at its GC) / (the candidates' density there). A race of
    exponentials gives the same law as drawing one at a time (wrswoR, or ``sample.int``)."""
    outside = np.ones(gc.size, dtype=bool)
    outside[query] = False
    candidates = np.flatnonzero(outside & np.isfinite(gc))
    values = gc[candidates]
    n = min(n, candidates.size)
    qx, qy = _r_density(gc[query])
    mx, my = _r_density(values)
    q = np.interp(values, qx, qy, left=1e-6, right=1e-6)
    m = np.interp(values, mx, my, left=1e-6, right=1e-6)
    with np.errstate(divide="ignore", invalid="ignore"):
        weights = q / m
    weights[~np.isfinite(weights)] = 0.0
    keys = np.full(candidates.size, np.inf)
    ok = weights > 0
    keys[ok] = rng.exponential(size=int(ok.sum())) / weights[ok]
    return candidates[np.argpartition(keys, n - 1)[:n]] if 0 < n < candidates.size else candidates[keys < np.inf]


def _motif_bh(p):
    """p.adjust(p, "BH")."""
    n = p.size
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = np.minimum.accumulate((p[order] * n / np.arange(1, n + 1))[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(ranked, 1.0)
    return out


def motif_numbers(observed, in_background, n_query, n_background):
    """FindMotifs' numbers per motif: phyper(observed - 1, K, N - K, n, lower.tail = FALSE)
    with N the background peaks, K those with a hit, n the query peaks; BH across all motifs."""
    from scipy.stats import hypergeom

    p = np.clip(hypergeom.sf(observed - 1, n_background, in_background, n_query), 0.0, 1.0)
    pct_obs = 100.0 * observed / max(1, n_query)
    pct_bg = 100.0 * in_background / max(1, n_background)
    with np.errstate(divide="ignore", invalid="ignore"):
        fold = pct_obs / pct_bg
    return {"observed": observed, "background": in_background, "pct_observed": pct_obs, "pct_background": pct_bg,
            "fold_enrichment": fold, "pval": p, "pval_adj": _motif_bh(p)}


def find_motifs(table: PeakMotifTable, query: np.ndarray, background: np.ndarray) -> dict[str, np.ndarray]:
    return motif_numbers(table.counts(query), table.counts(background), query.size, background.size)


def group_seed(seed, group):
    return [int(seed), int(hashlib.sha256(str(group).encode()).hexdigest()[:8], 16)]


def query_peaks(frame, group: str, p_column: str, p_max: float, min_pct: float, min_log2fc: float) -> np.ndarray:
    """The peaks a group's motif test asks about, from a stored differential result."""
    rows = frame[frame["group"] == group]
    keep = (rows[p_column] < p_max) & (rows["pct_in"] > min_pct) & (rows["log2fc"] > min_log2fc)
    return rows.loc[keep, "peak_index"].to_numpy(dtype=np.int64)


def enrich_groups(table: PeakMotifTable, frame, groups: list[str], *, p_column: str = "pval", p_max: float = QUERY_P,
                  min_pct: float = QUERY_PCT, min_log2fc: float = QUERY_LOG2FC, n_background: int = BACKGROUND_PEAKS,
                  seed: int = SEED, kept_per_group: int | None = None, stop: Callable[[], bool] = lambda: False,
                  standard: np.ndarray | None = None):
    """Each group's query peaks against a GC-matched background: (one long table, per group counts).
    ``standard``: the peaks that may be query or background (the standard chromosomes), None for all."""
    import pandas as pd

    gc = table.gc if standard is None else np.where(standard, table.gc, np.nan)
    known = np.isfinite(gc)
    frames, summary = [], {}
    for g in groups:
        if stop():
            raise ScanCancelled
        peaks = query_peaks(frame, g, p_column, p_max, min_pct, min_log2fc)
        rows_kept = int((frame["group"] == g).sum())
        query = peaks[known[peaks]]
        info: dict[str, Any] = {"n_query": int(query.size), "n_left_out": int(peaks.size - query.size), "n_background": 0, "note": None,
                                "capped": bool(kept_per_group and rows_kept >= kept_per_group
                                               and frame[frame["group"] == g][p_column].max() < p_max)}
        if query.size == 0:
            info["note"] = (f"No peak of {g} passes these thresholds: loosen them, or run differential peaks with more "
                            "cells in the group.")
            summary[g] = info
            continue
        rng = np.random.default_rng(group_seed(seed, g))
        background = match_background(gc, query, n_background, rng)
        if background.size < query.size:
            info["note"] = (f"{g} has {query.size:,} query peaks and only {background.size:,} other peaks to match them "
                            "against: tighten the thresholds.")
            summary[g] = info
            continue
        if "peak_class" in frame.columns:
            classes = frame.loc[frame["group"] == g].set_index("peak_index")["peak_class"]
            share = float((classes.reindex(query).to_numpy() == "promoter").mean())
            info["promoter_share"] = round(share, 3)
            if share > PROMOTER_HEAVY:
                info["note"] = (f"{share:.0%} of {g}'s query peaks are promoters, where GC-rich KLF and SP motifs top the "
                                "list whatever the cell type: lower 'open in over' to take in distal peaks.")
        numbers = find_motifs(table, query, background)
        info["n_background"] = int(background.size)
        info["n_significant"] = int((numbers["pval_adj"] < 0.05).sum())
        if query.size < MIN_QUERY:
            info["note"] = f"{g} has only {query.size} query peaks: enrichment on so few is not recommended."
        order = np.lexsort((-np.nan_to_num(numbers["fold_enrichment"], nan=-np.inf, posinf=np.inf), numbers["pval"]))
        frames.append(pd.DataFrame({
            "group": g, "motif_id": table.motif_ids[order], "motif": table.motif_names[order],
            "observed": numbers["observed"][order].astype(np.int32), "n_query": int(query.size),
            "background": numbers["background"][order].astype(np.int32), "n_background": int(background.size),
            "pct_observed": numbers["pct_observed"][order], "pct_background": numbers["pct_background"][order],
            "fold_enrichment": numbers["fold_enrichment"][order], "pval": numbers["pval"][order],
            "pval_adj": numbers["pval_adj"][order], "gc": table.gc_motif[order].round(3), "gc_rich": table.gc_rich[order]}))
        summary[g] = info
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return out, summary


def tf_statistics(adaptor, column: str, symbols: Iterable[str], batch: int = 200) -> dict[str, dict]:
    """``tf_motifs.marker_stats`` for these TF symbols, reading ``batch`` genes at a time through
    the adaptor, so a large dataset reads its gene store and never a whole matrix."""
    layer = adaptor.default_expression_layer()
    names = [str(g) for g in adaptor._var_names_for_layer(layer)]  # noqa: SLF001
    exact = set(names)
    upper: dict[str, list[str]] = {}
    for g in names:
        upper.setdefault(g.upper(), []).append(g)
    genes: list[str] = []
    for t in dict.fromkeys(symbols):
        if t in exact:
            genes.append(t)
        elif len(upper.get(t.upper(), [])) == 1:
            genes.append(upper[t.upper()][0])
    out: dict[str, dict] = {}
    for start in range(0, len(genes), batch):
        part = genes[start:start + batch]
        out.update(tf_motifs.marker_stats(adaptor.adata, column, part, adaptor.get_expression(part, layer=layer), part))
    return out


def annotate_expression(frame, stats: dict[str, dict]) -> None:
    """Adds the TF members and whether the TF itself is expressed in each row's group (in place)."""
    tfs, genes, pcts, fcs, expressed = [], [], [], [], []
    for group, motif in zip(frame["group"].astype(str), frame["motif"].astype(str)):
        members = tf_motifs.tf_members(motif)
        e = tf_motifs.tf_expressed(members, group, stats)
        tfs.append(", ".join(members))
        genes.append(e["gene"] if e else "")
        pcts.append(e["pct"] if e else np.nan)
        fcs.append(e["logfc"] if e and e["logfc"] is not None else np.nan)
        expressed.append(bool(e["expressed"]) if e else False)
    frame["tfs"] = tfs
    frame["tf_gene"] = genes
    frame["tf_pct"] = np.asarray(pcts, dtype=np.float64)
    frame["tf_log2fc"] = np.asarray(fcs, dtype=np.float64)
    frame["tf_expressed"] = expressed


@dataclass
class MotifEnrichment:
    result_id: str
    da_id: str
    column: str
    groups: list[str]
    table: Any  # one long table: every motif per group, lowest p first
    summary: dict[str, dict]
    params: dict[str, Any]
    method: str
    genome: str
    scan: dict[str, Any]
    created_at: str


_LAST: dict[str, MotifEnrichment] = {}
KEEP_RESULTS = 4


def remember(adaptor, dataset_id: str, res: MotifEnrichment) -> None:
    from scview.core import results_store

    _LAST[f"{dataset_id}:{res.result_id}"] = res
    while len(_LAST) > KEEP_RESULTS:
        _LAST.pop(next(iter(_LAST)))
    results_store.keep(adaptor, RESULT_KIND, res, result_id=res.result_id, key=res.column, params=dict(res.params),
                       label=f"Motifs in differential peaks by {res.column} ({len(res.groups)} groups)")


def recall(dataset_id: str, result_id: str) -> MotifEnrichment | None:
    from scview.core import results_store

    hit = _LAST.get(f"{dataset_id}:{result_id}")
    if hit is not None:
        return hit
    got = results_store.load(dataset_id, result_id)
    return got[1] if got is not None and isinstance(got[1], MotifEnrichment) else None


def latest(adaptor) -> MotifEnrichment | None:
    from scview.core import results_store

    try:
        return results_store.latest(adaptor, RESULT_KIND)
    except Exception:  # noqa: BLE001 - a test double, an unreadable result
        return None


def payload(res: MotifEnrichment, group: str | None = None, offset: int = 0, limit: int = 100) -> dict[str, Any]:
    """One result for the UI: the counts, a page of one group's rows (each group's first page without one)."""
    t = res.table

    def rows(g: str, start: int = 0) -> list[dict]:
        if t is None or not len(t):
            return []
        # double_precision: the default of 10 decimals turns a p of 1e-12 into 0
        return json.loads(t[t["group"] == g].iloc[start:start + limit].drop(columns=["group"]).to_json(orient="records", double_precision=15))

    pages = {group: rows(group, offset)} if group else {g: rows(g) for g in res.groups}
    return {"id": res.result_id, "da_id": res.da_id, "column": res.column, "groups": list(res.groups), "summary": res.summary,
            "rows": pages, "params": res.params, "method": res.method, "genome": res.genome, "scan": res.scan,
            "caveats": CAVEATS, "citation": CITATION, "created_at": res.created_at}


def record(adata, res: MotifEnrichment, resources: dict | None = None) -> None:
    """The run in provenance (History): a result only, the data are unchanged."""
    from scview.core import provenance

    provenance.record_step(adata, step=STEP_ENRICH, tool="scview.atac_motifs (motifmatchr hits, Signac FindMotifs)",
                           params={**res.params, "column": res.column, "genome": res.genome, "result_id": res.result_id,
                                   "table": res.scan.get("key")}, resources=resources)


# ---------------------------------------------------------------------------
# The logo
# ---------------------------------------------------------------------------

_GLYPHS = {
    "A": "M0 100 L39 0 H61 L100 100 H77 L68 76 H32 L23 100 Z M39 57 H61 L50 26 Z",
    "C": "M95 27 A50 52 0 1 0 95 73 L76 62 A28 30 0 1 1 76 38 Z",
    "G": "M95 27 A50 52 0 1 0 98 68 V47 H54 V63 H77 A28 30 0 1 1 76 38 Z",
    "T": "M0 0 H100 V19 H61 V100 H39 V19 H0 Z",
}
_COLOURS = {"A": "#16a34a", "C": "#2563eb", "G": "#d97706", "T": "#dc2626"}


def logo_heights(counts: np.ndarray) -> np.ndarray:
    """4 x L letter heights in bits: each base's share of the column x the column's information."""
    counts = np.asarray(counts, dtype=np.float64)
    p = (counts + 0.01) / (counts.sum(axis=0) + 0.04)
    return p * np.maximum(2.0 + (p * np.log2(p)).sum(axis=0), 0.0)


def consensus(counts: np.ndarray) -> str:
    return "".join("ACGT"[i] for i in np.asarray(counts).argmax(axis=0))


def logo_svg(counts: np.ndarray, title: str, column: int = 18, height: int = 72) -> str:
    """The motif's sequence logo as a standalone SVG: letters drawn as paths, so no font is needed."""
    from xml.sax.saxutils import escape

    h = logo_heights(counts)
    n = h.shape[1]
    left, bottom = 24, 14
    width = left + n * column + 2
    bit = height / 2.0
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height + bottom + 2}" width="{width}" '
             f'height="{height + bottom + 2}" role="img" aria-label="{escape(title, {chr(34): "&quot;"})}">',
             f"<title>{escape(title)}</title>",
             f'<path d="M{left - 2} 1 V{height + 1}" stroke="#94a3b8" fill="none" stroke-width="1"/>']
    for b in (0, 1, 2):
        y = height + 1 - b * bit
        parts.append(f'<text x="{left - 5}" y="{y + 3:.1f}" font-size="8" text-anchor="end" fill="#64748b" '
                     f'font-family="sans-serif">{b}</text>')
    for j in range(n):
        y = float(height + 1)
        for b in np.argsort(h[:, j], kind="stable"):
            size = float(h[b, j]) * bit
            if size < 0.25:
                continue
            y -= size
            letter = "ACGT"[int(b)]
            parts.append(f'<path transform="translate({left + j * column + 1} {y:.2f}) scale({(column - 2) / 100:.3f} {size / 100:.4f})" '
                         f'd="{_GLYPHS[letter]}" fill="{_COLOURS[letter]}" fill-rule="evenodd"/>')
        parts.append(f'<text x="{left + j * column + column / 2:.1f}" y="{height + bottom - 1}" font-size="7" text-anchor="middle" '
                     f'fill="#64748b" font-family="sans-serif">{j + 1}</text>')
    parts.append("</svg>")
    return "".join(parts)


REPLAY_CONSTANTS = ("MOTIF_P_CUTOFF", "PSEUDOCOUNT", "DP_SCALE", "INDEX_K", "BLOCK_BASES", "BUILD_THREADS", "_N", "_EPS")
REPLAY_CODE = (pwm, motif_threshold, reverse_complement, Strand, _window_scores, prepare, prepare_all, join_regions, _kmer_index, _scan_strand,
               scan_block, _blocks, _r_density, match_background, _motif_bh, motif_numbers, group_seed)


# ---------------------------------------------------------------------------
# The co-pilot
# ---------------------------------------------------------------------------


def _row_text(r: dict) -> str:
    gc = ", GC-rich" if r.get("gc_rich") else ""
    tf = ""
    if isinstance(r.get("tf_gene"), str) and r["tf_gene"] and r.get("tf_pct") == r.get("tf_pct"):
        tf = f", {r['tf_gene']} RNA in {100 * float(r['tf_pct']):.0f}% of the group's cells"
    return f"{r['motif']} ({r['motif_id']}) fold {r['fold_enrichment']:.2f}, FDR {r['pval_adj']:.2g}{tf}{gc}"


def grounding_lines(adaptor) -> tuple[list[str], str]:
    """The last enrichment on these cells as facts: the top motifs per group, FDR < 5%."""
    res = latest(adaptor)
    if res is None or res.table is None or not len(res.table):
        return [], ""
    p = res.params
    which = "FDR" if p.get("p_column") == "pval_adj" else "p"
    detail = (f"JASPAR 2024 motifs in differential peaks by '{res.column}' ({res.genome}; {which} < {p.get('p_max')}, open in over "
              f"{100 * float(p.get('min_pct', 0)):.0f}% of the group, up in it; {p.get('background')} GC-matched background peaks)")
    lines = ["\n## Motifs in differential peaks (Signac FindMotifs; candidates, not proven binding)", f"- {detail}; hypergeometric, BH per group [{TAG}]."]
    t = res.table
    for g in res.groups[:14]:
        rows = t[(t["group"] == g) & (t["pval_adj"] < 0.05) & (t["fold_enrichment"] > 1)].head(4)
        n = res.summary.get(g, {}).get("n_query", 0)
        if not len(rows):
            lines.append(f"- Motifs in peaks, {g} ({n} query peaks): none with FDR < 5% [{TAG}]")
            continue
        lines.append(f"- Motifs in peaks, {g} ({n} query peaks): " + "; ".join(_row_text(r) for r in rows.to_dict("records")) + f" [{TAG}]")
    lines.append("- Caveat: a motif hit is not binding; paralogues share motifs, so check the TF's own RNA; GC-rich motifs enrich easily "
                 "even after matching. Say 'candidate'.")
    return lines, detail


def named_motif_lines(adaptor, query: str, max_motifs: int = 2, max_groups: int = 4) -> list[tuple[str, str]]:
    """(TF, fact) for each TF the question names: the groups where its motif enriches most."""
    res = latest(adaptor)
    if res is None or res.table is None or not len(res.table) or not query:
        return []
    t = res.table
    parts = {part.upper() for name in t["motif"].astype(str).unique() for part in name.split("::")}
    out: list[tuple[str, str]] = []
    for word in dict.fromkeys(re.findall(r"[A-Za-z][A-Za-z0-9-]+", query)):
        key = word.upper()
        if key not in parts or len(out) >= max_motifs:
            continue
        names = t["motif"].astype(str).map(lambda n: key in [x.upper() for x in n.split("::")])
        rows = t[names].sort_values("pval").drop_duplicates("group").head(max_groups)
        out.append((key, f"- Motif of {key} in differential peaks, its {len(rows)} best groups of {len(res.groups)}: "
                         + "; ".join(f"{r['group']}: {_row_text(r)}" for r in rows.to_dict("records"))))
    return out


if __name__ == "__main__":
    _child(sys.argv[1:])
