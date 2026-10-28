"""Upstream TF candidates per group, from transcription-factor motifs in promoters (C095).

Which TFs could sit upstream of a group's marker genes: the JASPAR 2024 motifs found
in more of the markers' promoters than in the other measured genes' promoters. A lead
to follow up, never proof of regulation (see ``CAVEATS``).

**Inputs** (fetched once, on first use, into ``DATA_DIR/tf_motifs/raw``, each checked
against a pinned sha256; ``TF_MOTIFS_AUTO_DOWNLOAD=false`` turns fetching off and a
file placed there by hand is used when its sha256 matches):

- JASPAR 2024 CORE vertebrates non-redundant PFMs (879 matrices, CC BY 4.0;
  Rauluseviciute et al. 2024, NAR, PMID 37962376);
- UCSC ``upstream1000.fa.gz`` (1,000 bp upstream of each RefSeq transcript's TSS,
  on the transcript's strand) and ``refGene.txt.gz`` (accession -> gene symbol), for
  hg38, mm10 and mm39. Mouse data use mm10 unless the dataset names GRCm39/mm39.

**One promoter per gene symbol** (``choose_promoters``): of the transcripts whose
accession refGene gives that symbol, the one (1) on a primary chromosome (chr1-22,
X, Y; not an alt, fix or random contig), then (2) protein-coding (NM_) before
non-coding (NR_), then (3) with the fewest unknown bases, then (4) with the lowest
accession number (the longest-standing RefSeq record, usually the reference
transcript), then (5) first in the file. Upstream only: sequence just downstream of
the TSS (5' UTR, first intron) is not scanned.

**Scan** (``build``, run in a child process at 2 threads, ``run_child``): each PFM
becomes a log-odds matrix in bits, ``log2(p / b)`` with ``p = (count + 0.8 b) /
(N + 0.8)`` (a pseudocount of 0.8 shared out by the background, as Nishida et al.
2009 recommend) against the promoters' own base composition ``b`` (A = T, C = G:
both strands together). Weights are rounded to 1/100 bit, and the score threshold of
each motif is the lowest integer score whose exact probability under the
background, from the discretised score distribution (dynamic programming over the
columns, as TFM-Pvalue), is under ``P_THRESHOLD`` (1e-4): the scan sums the same
integers, so the threshold is exact for the scores it compares. Both strands are
scanned (the reverse complement as a second matrix); a promoter has a hit when any
window on either strand reaches the threshold, and a window containing an unknown
base never does. A motif too short or too flat to reach p < 1e-4 at all (a 6-mer's
best word has p ~2.4e-4) is never tested (``unreachable``). One matrix product per
motif length (one-hot windows x stacked matrices), in blocks of promoters: the
genes x motifs hit matrix is kept bit-packed in ``DATA_DIR/tf_motifs/{assembly}``,
keyed on the input sha256s and these parameters.

**GC flag**: a motif whose information-weighted GC content (each column's G + C
weighted by its information content) is over 0.7 is flagged "GC-rich": CpG-island
promoters are GC-rich too, so such motifs (SP/KLF, E2F, NRF1, ZBTB, many ETS) hit
the promoters of broadly expressed genes and come out enriched more easily.

**Test** (``enrich``): per group, the universe N is the dataset's measured genes
(expressed in at least one cell) with a promoter; the query is the group's marker
genes from ``marker_selection.group_markers`` (the shared marker choice) in the
universe (n); K is the universe genes whose promoter has the motif and k the
markers among them. One-sided hypergeometric P(X >= k), the odds ratio (0.5 added
to every cell when one is 0), and Benjamini-Hochberg across every motif with K > 0
in that group; a motif is reported when k >= 3.
"""

from __future__ import annotations

import fcntl
import gzip
import hashlib
import json
import logging
import math
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

import numpy as np

logger = logging.getLogger(__name__)

P_THRESHOLD = 1e-4
PSEUDOCOUNT = 0.8
SCALE = 100  # integer score steps per bit
PROMOTER_LEN = 1000
GC_RICH = 0.7
MIN_OVERLAP = 3
MAX_ROWS = 100  # reported motifs per group, lowest p first
MAX_OVERLAP_GENES = 30
DEFAULT_MARKERS = 100
TOP_ACTIVITY_SHARE = 0.1  # "agrees": the TF is in the top 10% of CollecTRI TFs for the group
MIN_ACTIVITY_RANK = 5
EXPRESSED_PCT = 0.1  # a TF expressed in at least 10% of the group's cells counts as expressed
BUILD_THREADS = 2
PROMOTER_RULE = "primary-chrom > NM > fewest N > lowest accession > file order (v1)"
FORMAT = 1
KIND = "tf_motifs"  # results store
STEP = "tf_motif_enrichment"  # provenance (additive: adds a result only)
TAG = "result:tfmotifs"
MAX_DOWNLOAD_BYTES = 64 * 1024 * 1024  # the largest file is 10.8 MB
_N = 4  # code of an unknown base
_MASK = -1_000_000  # an unknown base in a window: never a hit
_BLOCK_FLOATS = 1 << 24  # floats per block matrix (64 MB)

CITATION = ("JASPAR 2024 (Rauluseviciute et al. 2024, Nucleic Acids Res, PMID 37962376; CC BY 4.0); "
            "promoters: UCSC upstream1000 of RefSeq transcripts (refGene)")
LICENCE = "JASPAR: CC BY 4.0; UCSC/RefSeq: no restrictions"
CAVEATS = (
    "Promoters only (1 kb upstream of the TSS): enhancers and distal sites are not looked at. "
    "GC-rich motifs (SP/KLF, E2F, NRF1, many ETS) hit CpG-island promoters of broadly expressed "
    "genes and come out enriched more easily. Paralogous TFs share near-identical motifs, so the "
    "TF named is one of a family. A motif in a promoter is not binding, and marker genes include "
    "indirect targets: each row is a candidate to check, not a regulator."
)


@dataclass(frozen=True)
class Source:
    name: str
    url: str
    sha256: str
    filename: str
    size: int


@dataclass(frozen=True)
class Assembly:
    name: str
    species: str
    label: str
    upstream: Source
    refgene: Source
    estimate_seconds: int


_UCSC = "https://hgdownload.soe.ucsc.edu/goldenPath"
JASPAR = Source(
    "JASPAR 2024 CORE vertebrates non-redundant (PFM)",
    "https://jaspar.elixir.no/download/data/2024/CORE/JASPAR2024_CORE_vertebrates_non-redundant_pfms_jaspar.txt",
    "2a6c7c24afb0614ed418c9f02b68845adebf38f80f6cebabbcd5de804eaacb59",
    "JASPAR2024_CORE_vertebrates_non-redundant_pfms_jaspar.txt", 283_814,
)


def _ucsc(asm: str, kind: str, sha: str, size: int) -> Source:
    path = "bigZips/upstream1000.fa.gz" if kind == "upstream" else "database/refGene.txt.gz"
    fname = "upstream1000.fa.gz" if kind == "upstream" else "refGene.txt.gz"
    label = "upstream 1 kb promoters" if kind == "upstream" else "RefSeq genes (refGene)"
    return Source(f"UCSC {asm} {label}", f"{_UCSC}/{asm}/{path}", sha, f"{asm}.{fname}", size)


ASSEMBLIES: dict[str, Assembly] = {
    "hg38": Assembly("hg38", "human", "human (hg38)",
                     _ucsc("hg38", "upstream", "c066280171b8170c50357090c5305352ef6680c13197b02e51f98398a2e1c8e7", 10_826_712),
                     _ucsc("hg38", "refgene", "58fbc18d60bcda5fd4635e71b99dffde09d6eb67995c7f9ded3af194eb789756", 8_668_756),
                     estimate_seconds=120),
    "mm10": Assembly("mm10", "mouse", "mouse (mm10)",
                     _ucsc("mm10", "upstream", "98edf5730b3c4de3513bf7a04b050961569d047d5028b2675a16ca36fd3b69ce", 6_948_661),
                     _ucsc("mm10", "refgene", "fab8ea8282ea60cf6d9bd17ac3ef2c91bbf3abd47c3ca7c0054b3196a7276fa1", 4_813_121),
                     estimate_seconds=120),
    "mm39": Assembly("mm39", "mouse", "mouse (mm39)",
                     _ucsc("mm39", "upstream", "d767428085152b59344b440860abcb21390e230d320925dc6865af54cbe2c559", 8_392_619),
                     _ucsc("mm39", "refgene", "7a8fd4392045198e85ef96f00da6427a6b9fa23b61bc6aa52cde1b3c09d61aed", 3_099_688),
                     estimate_seconds=120),
}
DEFAULT_ASSEMBLY = {"human": "hg38", "mouse": "mm10"}


class MotifTableUnavailable(RuntimeError):
    """An input file is missing and could not be fetched (offline, switched off, bad file)."""


# ---------------------------------------------------------------------------
# Parsing


@dataclass
class Motif:
    id: str
    name: str
    counts: np.ndarray  # 4 x L (A, C, G, T)


_ROW = re.compile(r"^([ACGT])\s*\[([^\]]*)\]$")


def parse_jaspar(text: str) -> list[Motif]:
    """JASPAR's PFM format: ``>ID NAME`` then one ``X [ c1 c2 ... ]`` line per base A, C, G, T.
    Raises ValueError on anything else."""
    motifs: list[Motif] = []
    head: tuple[str, str] | None = None
    rows: dict[str, list[float]] = {}

    def flush() -> None:
        if head is None:
            return
        if set(rows) != set("ACGT"):
            raise ValueError(f"motif {head[0]} lacks rows {sorted(set('ACGT') - set(rows))}")
        lengths = {len(v) for v in rows.values()}
        if len(lengths) != 1 or 0 in lengths:
            raise ValueError(f"motif {head[0]} has rows of different lengths")
        counts = np.array([rows[b] for b in "ACGT"], dtype=np.float64)
        if (counts < 0).any() or (counts.sum(axis=0) <= 0).any():
            raise ValueError(f"motif {head[0]} has a negative or empty column")
        motifs.append(Motif(head[0], head[1], counts))

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            flush()
            parts = line[1:].split(None, 1)
            if not parts:
                raise ValueError("a motif header without an id")
            head = (parts[0], parts[1].strip() if len(parts) > 1 else parts[0])
            rows = {}
            continue
        m = _ROW.match(line)
        if m is None or head is None:
            raise ValueError(f"not a JASPAR PFM line: {line[:60]!r}")
        try:
            rows[m.group(1)] = [float(x) for x in m.group(2).split()]
        except ValueError:
            raise ValueError(f"motif {head[0]}: a count is not a number") from None
    flush()
    if not motifs:
        raise ValueError("no motif in the file")
    return motifs


_VARIANT = re.compile(r"\(var\.\d+\)$", re.IGNORECASE)


def tf_members(name: str) -> list[str]:
    """The TF symbols a JASPAR name stands for: a dimer ``MAX::MYC`` is two, a variant
    ``NR2F1(var.2)`` is ``NR2F1``."""
    out = []
    for part in str(name).split("::"):
        p = _VARIANT.sub("", part.strip()).strip()
        if p and p not in out:
            out.append(p)
    return out


_CODES = np.full(256, _N, dtype=np.uint8)
for _i, _b in enumerate("ACGT"):
    _CODES[ord(_b)] = _i
    _CODES[ord(_b.lower())] = _i


def encode(seq: bytes | str) -> np.ndarray:
    """A sequence as base codes: A 0, C 1, G 2, T 3, anything else 4."""
    if isinstance(seq, str):
        seq = seq.encode("ascii", "replace")
    return _CODES[np.frombuffer(seq, dtype=np.uint8)]


def _open_text(path: Path):
    path = Path(path)
    with open(path, "rb") as fh:
        gz = fh.read(2) == b"\x1f\x8b"
    return gzip.open(path, "rt") if gz else open(path, "rt")


def parse_upstream(path: Path) -> Iterator[tuple[str, str, str]]:
    """(accession, chromosome, sequence) per record of a UCSC upstream FASTA, headers like
    ``>NM_000299_up_1000_chr1_201282452_f chr1:201282452-201283451``."""
    acc = chrom = None
    parts: list[str] = []
    with _open_text(path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(">"):
                if acc is not None:
                    yield acc, chrom or "", "".join(parts)
                name = line[1:].split(None, 1)[0]
                acc, _, rest = name.partition("_up_")
                # rest = "1000_chr1_201282452_f"; the chromosome can hold "_" (alt contigs)
                bits = rest.split("_", 1)[1] if "_" in rest else ""
                chrom = bits.rsplit("_", 2)[0] if bits.count("_") >= 2 else bits
                parts = []
            elif line:
                parts.append(line)
    if acc is not None:
        yield acc, chrom or "", "".join(parts)


def parse_refgene(path: Path) -> dict[str, str]:
    """accession -> gene symbol from a UCSC refGene table (name in column 2, name2 in 13)."""
    out: dict[str, str] = {}
    with _open_text(path) as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) < 13:
                continue
            acc, sym = f[1].split(".")[0], f[12].strip()
            if acc and sym:
                out.setdefault(acc, sym)
    if not out:
        raise ValueError("no accession -> symbol rows in the refGene file")
    return out


_PRIMARY = re.compile(r"^chr(\d+|X|Y)$")
_ACC_NUM = re.compile(r"_(\d+)$")


def _promoter_rank(acc: str, chrom: str, n_unknown: int, order: int) -> tuple:
    m = _ACC_NUM.search(acc)
    return (0 if _PRIMARY.match(chrom) else 1, 0 if acc.startswith("NM_") else 1, n_unknown,
            int(m.group(1)) if m else sys.maxsize, order)


def choose_promoters(records: Iterable[tuple[str, str, str]], symbols: dict[str, str],
                     length: int = PROMOTER_LEN) -> tuple[list[str], list[str], np.ndarray]:
    """One promoter per gene symbol (the rule in the module docstring): (symbols, the
    chosen accession of each, n x ``length`` base codes). A shorter record is padded with
    unknown bases at its distal end, a longer one keeps its ``length`` bases next to the TSS."""
    best: dict[str, tuple[tuple, str, np.ndarray]] = {}
    for order, (acc, chrom, seq) in enumerate(records):
        sym = symbols.get(acc)
        if not sym:
            continue
        codes = encode(seq)
        if codes.size >= length:
            codes = codes[-length:]
        else:
            codes = np.concatenate([np.full(length - codes.size, _N, dtype=np.uint8), codes])
        rank = _promoter_rank(acc, chrom, int((codes == _N).sum()), order)
        if sym not in best or rank < best[sym][0]:
            best[sym] = (rank, acc, codes)
    names = sorted(best)
    if not names:
        raise ValueError("no promoter maps to a gene symbol")
    return (names, [best[s][1] for s in names],
            np.stack([best[s][2] for s in names]) if names else np.zeros((0, length), np.uint8))


def background(codes: np.ndarray) -> np.ndarray:
    """Base frequencies of the promoters, both strands together (A = T, C = G)."""
    counts = np.bincount(codes.ravel(), minlength=5)[:4].astype(np.float64)
    if counts.sum() == 0:
        return np.full(4, 0.25)
    at = (counts[0] + counts[3]) / 2
    cg = (counts[1] + counts[2]) / 2
    total = 2 * (at + cg)
    return np.array([at, cg, cg, at]) / total


# ---------------------------------------------------------------------------
# Scoring


def log_odds(counts: np.ndarray, bg: np.ndarray, pseudocount: float = PSEUDOCOUNT) -> np.ndarray:
    """4 x L log-odds in bits: log2(p / b), p = (count + pseudocount b) / (N + pseudocount)."""
    counts = np.asarray(counts, dtype=np.float64)
    bg = np.asarray(bg, dtype=np.float64)
    p = (counts + pseudocount * bg[:, None]) / (counts.sum(axis=0) + pseudocount)
    return np.log2(p / bg[:, None])


def integer_matrix(w: np.ndarray, scale: int = SCALE) -> np.ndarray:
    return np.rint(np.asarray(w) * scale).astype(np.int64)


def score_distribution(wint: np.ndarray, bg: np.ndarray) -> tuple[int, np.ndarray]:
    """(lowest score, P(score = lowest + i) for every i) of a random background word
    under the integer matrix ``wint`` (4 x L)."""
    wint = np.asarray(wint, dtype=np.int64)
    mins = wint.min(axis=0)
    rel = wint - mins
    dist = np.ones(1)
    for j in range(wint.shape[1]):
        width = int(rel[:, j].max())
        new = np.zeros(dist.size + width)
        for b in range(4):
            s = int(rel[b, j])
            new[s:s + dist.size] += dist * bg[b]
        dist = new
    return int(mins.sum()), dist


def threshold(wint: np.ndarray, bg: np.ndarray, p: float = P_THRESHOLD) -> int | None:
    """The lowest integer score t with P(score >= t) < p under the background, or None
    when even the best word is that likely (the motif cannot reach p)."""
    lo, dist = score_distribution(wint, bg)
    sf = np.cumsum(dist[::-1])[::-1]  # P(score >= lo + i), summed from the smallest
    ok = np.flatnonzero(sf < p)
    return None if ok.size == 0 else lo + int(ok[0])


def gc_content(counts: np.ndarray) -> tuple[float, float]:
    """(information-weighted GC, consensus GC) of a PFM."""
    counts = np.asarray(counts, dtype=np.float64)
    p = (counts + 0.01) / (counts.sum(axis=0) + 0.04)
    ic = np.maximum(2.0 + (p * np.log2(p)).sum(axis=0), 0.0)
    gc = p[1] + p[2]
    weighted = float((ic * gc).sum() / ic.sum()) if ic.sum() > 0 else float(gc.mean())
    consensus = float(np.isin(counts.argmax(axis=0), (1, 2)).mean())
    return weighted, consensus


def _stack(wints: list[np.ndarray]) -> np.ndarray:
    """(5L) x 2m float32: each matrix and its reverse complement as columns, rows
    base-major (base * L + position), a fifth "unknown base" row of ``_MASK``."""
    cols = []
    for w in wints:
        rc = w[::-1, ::-1]  # A<->T and C<->G is codes 3 - b; and the positions reversed
        for m in (w, rc):
            cols.append(np.vstack([m, np.full((1, m.shape[1]), _MASK)]).ravel())
    fwd = cols[0::2]
    rev = cols[1::2]
    return np.asarray(fwd + rev, dtype=np.float32).T


def scan(codes: np.ndarray, wints: list[np.ndarray], thresholds: np.ndarray,
         progress: Callable[[int, int], None] | None = None) -> np.ndarray:
    """promoters x motifs: whether any window on either strand reaches the motif's
    threshold. ``codes``: n x length base codes; ``wints``: 4 x L integer matrices."""
    n, length = codes.shape
    m = len(wints)
    hits = np.zeros((n, m), dtype=bool)
    by_len: dict[int, list[int]] = {}
    for i, w in enumerate(wints):
        if w.shape[1] <= length:
            by_len.setdefault(w.shape[1], []).append(i)
    groups = []
    for L, idx in sorted(by_len.items()):
        groups.append((L, np.asarray(idx), _stack([wints[i] for i in idx]),
                       np.asarray(thresholds, dtype=np.float32)[idx]))
    widest = max((5 * L + 2 * len(idx) for L, idx, _, _ in groups), default=1)
    block = max(1, min(n, _BLOCK_FLOATS // (length * widest)))
    eye = np.eye(5, dtype=np.float32)
    for start in range(0, n, block):
        part = codes[start:start + block]
        onehot = eye[part].transpose(0, 2, 1)  # b x 5 x length
        for L, idx, W, thr in groups:
            P = length - L + 1
            win = np.lib.stride_tricks.sliding_window_view(onehot, L, axis=2)  # b x 5 x P x L
            X = np.ascontiguousarray(win.transpose(0, 2, 1, 3)).reshape(-1, 5 * L)
            S = (X @ W).reshape(part.shape[0], P, W.shape[1])
            best = S.max(axis=1)
            k = len(idx)
            hits[start:start + part.shape[0], idx] = (best[:, :k] >= thr) | (best[:, k:] >= thr)
        if progress is not None:
            progress(min(n, start + block), n)
    return hits


# ---------------------------------------------------------------------------
# The gene x motif table


@dataclass
class MotifTable:
    assembly: str
    genes: np.ndarray  # gene symbols with a promoter
    accessions: np.ndarray
    motif_ids: np.ndarray
    motif_names: np.ndarray
    lengths: np.ndarray
    thresholds: np.ndarray  # integer score (1/SCALE bit); -1 = cannot reach P_THRESHOLD
    gc: np.ndarray  # information-weighted GC
    hits: np.ndarray  # genes x motifs bool
    background: np.ndarray
    meta: dict = field(default_factory=dict)

    @property
    def reachable(self) -> np.ndarray:
        return self.thresholds >= 0

    @property
    def gc_rich(self) -> np.ndarray:
        return self.gc > GC_RICH


def build(jaspar: Path, upstream: Path, refgene: Path, *, assembly: str,
          progress: Callable[[dict], None] | None = None) -> MotifTable:
    """The genes x motifs hit table from the three input files."""
    say = progress or (lambda _e: None)
    t0 = time.time()
    say({"stage": "read", "message": "Reading the motifs and promoters"})
    motifs = parse_jaspar(Path(jaspar).read_text())
    names, accs, codes = choose_promoters(parse_upstream(upstream), parse_refgene(refgene))
    bg = background(codes)
    wints, thr = [], []
    for mo in motifs:
        w = integer_matrix(log_odds(mo.counts, bg))
        t = threshold(w, bg)
        wints.append(w)
        thr.append(-1 if t is None else t)
    thr_arr = np.asarray(thr, dtype=np.int64)
    ok = np.flatnonzero(thr_arr >= 0)
    say({"stage": "scan", "message": f"Scanning {len(names):,} promoters for {len(ok):,} motifs",
         "done": 0, "total": len(names)})

    def tick(done: int, total: int) -> None:
        say({"stage": "scan", "message": f"Scanned {done:,} of {total:,} promoters", "done": done,
             "total": total, "seconds": round(time.time() - t0, 1)})

    hits = np.zeros((len(names), len(motifs)), dtype=bool)
    if ok.size:
        hits[:, ok] = scan(codes, [wints[i] for i in ok], thr_arr[ok], tick)
    gc = np.array([gc_content(mo.counts)[0] for mo in motifs])
    return MotifTable(
        assembly=assembly, genes=np.asarray(names, dtype=str), accessions=np.asarray(accs, dtype=str),
        motif_ids=np.asarray([mo.id for mo in motifs], dtype=str),
        motif_names=np.asarray([mo.name for mo in motifs], dtype=str),
        lengths=np.asarray([mo.counts.shape[1] for mo in motifs], dtype=np.int64),
        thresholds=thr_arr, gc=gc, hits=hits, background=bg,
        meta={"assembly": assembly, "n_genes": len(names), "n_motifs": len(motifs),
              "n_unreachable": int((thr_arr < 0).sum()), "seconds": round(time.time() - t0, 1),
              "p_threshold": P_THRESHOLD, "pseudocount": PSEUDOCOUNT, "scale": SCALE,
              "promoter_rule": PROMOTER_RULE, "background": [round(float(b), 4) for b in bg],
              "hit_rate": round(float(hits[:, ok].mean()) if ok.size else 0.0, 4)},
    )


def save_table(table: MotifTable, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part.npz")
    np.savez_compressed(
        tmp, genes=table.genes, accessions=table.accessions, motif_ids=table.motif_ids,
        motif_names=table.motif_names, lengths=table.lengths, thresholds=table.thresholds,
        gc=table.gc, packed=np.packbits(table.hits, axis=1), n_motifs=np.int64(table.hits.shape[1]),
        background=table.background, assembly=np.str_(table.assembly), meta=np.str_(json.dumps(table.meta)),
    )
    tmp.replace(path)


def load_table_file(path: Path) -> MotifTable:
    with np.load(path, allow_pickle=False) as z:
        n_motifs = int(z["n_motifs"])
        return MotifTable(
            assembly=str(z["assembly"]), genes=z["genes"], accessions=z["accessions"],
            motif_ids=z["motif_ids"], motif_names=z["motif_names"], lengths=z["lengths"],
            thresholds=z["thresholds"], gc=z["gc"],
            hits=np.unpackbits(z["packed"], axis=1, count=n_motifs).astype(bool),
            background=z["background"], meta=json.loads(str(z["meta"])),
        )


def cache_key(assembly: str) -> str:
    """Names the table built from the pinned inputs with these parameters."""
    a = ASSEMBLIES[assembly]
    doc = {"jaspar": JASPAR.sha256, "upstream": a.upstream.sha256, "refgene": a.refgene.sha256,
           "p": P_THRESHOLD, "pseudocount": PSEUDOCOUNT, "scale": SCALE, "length": PROMOTER_LEN,
           "rule": PROMOTER_RULE, "format": FORMAT}
    return hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest()[:16]


def table_path(dest: Path, assembly: str) -> Path:
    return Path(dest) / assembly / f"motif_hits_{cache_key(assembly)}.npz"


_tables: dict[str, tuple[float, MotifTable]] = {}
_tables_lock = threading.Lock()


def load_table(dest: Path, assembly: str) -> MotifTable | None:
    """The built table for ``assembly`` (kept in memory while the file is unchanged), or None."""
    path = table_path(dest, assembly)
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return None
    with _tables_lock:
        cached = _tables.get(str(path))
        if cached and cached[0] == mtime:
            return cached[1]
    try:
        table = load_table_file(path)
    except Exception as e:  # noqa: BLE001 - an unreadable cache is rebuilt
        logger.warning("Unreadable motif table %s: %s", path, e)
        return None
    with _tables_lock:
        _tables[str(path)] = (mtime, table)
    return table


def read_meta(dest: Path, assembly: str) -> dict | None:
    t = load_table(dest, assembly)
    return dict(t.meta) if t is not None else None


# ---------------------------------------------------------------------------
# Input files


def raw_dir(dest: Path) -> Path:
    return Path(dest) / "raw"


def sources_for(assembly: str) -> list[Source]:
    a = ASSEMBLIES[assembly]
    return [JASPAR, a.upstream, a.refgene]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_ok(dest: Path, src: Source) -> bool:
    path = raw_dir(dest) / src.filename
    return path.is_file() and _sha256(path) == src.sha256


def download(dest: Path, src: Source, client=None) -> None:
    """Fetch one file and check its sha256; no partial file is left behind."""
    import httpx

    folder = raw_dir(dest)
    folder.mkdir(parents=True, exist_ok=True)
    part = folder / f"{src.filename}.part"
    own = client is None
    client = client or httpx.Client(timeout=httpx.Timeout(120.0, connect=20.0), follow_redirects=True)
    try:
        digest, size = hashlib.sha256(), 0
        with client.stream("GET", src.url) as resp:
            resp.raise_for_status()
            with open(part, "wb") as fh:
                for chunk in resp.iter_bytes():
                    size += len(chunk)
                    if size > MAX_DOWNLOAD_BYTES:
                        raise ValueError(f"download exceeds {MAX_DOWNLOAD_BYTES} bytes")
                    digest.update(chunk)
                    fh.write(chunk)
        if digest.hexdigest() != src.sha256:
            raise ValueError(f"sha256 {digest.hexdigest()[:12]}... is not the pinned {src.sha256[:12]}...")
        part.replace(folder / src.filename)
    finally:
        part.unlink(missing_ok=True)
        if own:
            client.close()


def ensure_files(dest: Path, assembly: str, *, auto_download: bool = True, client=None,
                 progress: Callable[[dict], None] | None = None) -> dict[str, Path]:
    """The three input files of ``assembly``, fetched when missing. Raises
    ``MotifTableUnavailable`` saying what to do."""
    say = progress or (lambda _e: None)
    folder = raw_dir(dest)
    out = {}
    for key, src in zip(("jaspar", "upstream", "refgene"), sources_for(assembly)):
        path = folder / src.filename
        if path.is_file() and _sha256(path) != src.sha256:
            raise MotifTableUnavailable(
                f"{path} is not the pinned file (sha256 differs); delete it to fetch it again.")
        if not path.is_file():
            if not auto_download:
                raise MotifTableUnavailable(
                    f"{src.name} is not downloaded and TF_MOTIFS_AUTO_DOWNLOAD is off. Place "
                    f"{src.filename} (from {src.url}) in {folder}, or turn the switch on.")
            say({"stage": "download", "message": f"Downloading {src.name} ({src.size / 1e6:.1f} MB)"})
            try:
                folder.mkdir(parents=True, exist_ok=True)
                with open(folder / ".download.lock", "w") as lk:
                    fcntl.flock(lk, fcntl.LOCK_EX)
                    if not path.is_file():
                        download(dest, src, client=client)
            except Exception as e:  # noqa: BLE001
                logger.warning("Could not fetch %s: %s", src.url, e)
                host = src.url.split("/")[2]
                raise MotifTableUnavailable(
                    f"{src.name} is not downloaded yet and fetching it failed ({e}). scView fetches "
                    f"it once from {host}; check the internet connection and try again, or place "
                    f"{src.filename} in {folder} by hand.") from e
        out[key] = path
    return out


# ---------------------------------------------------------------------------
# The child process


def run_child(files: dict[str, Path], out: Path, assembly: str,
              on_event: Callable[[dict], None] | None = None,
              stop: Callable[[], bool] | None = None, threads: int = BUILD_THREADS) -> dict:
    """Build the table in a child process (``python -m scview.core.tf_motifs``) limited to
    ``threads`` BLAS threads, so the server keeps answering; progress lines go to
    ``on_event``. Returns the child's final event (seconds, peak memory)."""
    env = dict(os.environ)
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS"):
        env[var] = str(threads)
    cmd = [sys.executable, "-m", "scview.core.tf_motifs", "--assembly", assembly,
           "--jaspar", str(files["jaspar"]), "--upstream", str(files["upstream"]),
           "--refgene", str(files["refgene"]), "--out", str(out)]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, text=True)
    final: dict = {}
    killer = None
    if stop is not None:
        def _watch() -> None:
            while proc.poll() is None:
                if stop():
                    proc.kill()
                    return
                time.sleep(0.2)
        killer = threading.Thread(target=_watch, daemon=True)
        killer.start()
    assert proc.stdout is not None
    for line in proc.stdout:
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if event.get("type") == "done":
            final = event
        elif on_event is not None:
            on_event(event)
    stderr = proc.stderr.read() if proc.stderr is not None else ""
    proc.wait()
    if proc.returncode != 0 or not final:
        if stop is not None and stop():
            raise RuntimeError("The motif scan was stopped.")
        tail = " ".join(stderr.strip().splitlines()[-3:])
        raise RuntimeError(f"The motif scan failed (exit {proc.returncode}): {tail}")
    return final


def _child(argv: list[str]) -> None:
    import argparse
    import resource

    ap = argparse.ArgumentParser()
    for a in ("--assembly", "--jaspar", "--upstream", "--refgene", "--out"):
        ap.add_argument(a, required=True)
    args = ap.parse_args(argv)
    last = [0.0]

    def emit(e: dict) -> None:
        now = time.time()
        if e.get("stage") == "scan" and e.get("done") not in (0, e.get("total")) and now - last[0] < 1.0:
            return  # at most one progress line a second
        last[0] = now
        sys.stdout.write(json.dumps({"type": "progress", **e}) + "\n")
        sys.stdout.flush()

    t0 = time.time()
    table = build(Path(args.jaspar), Path(args.upstream), Path(args.refgene), assembly=args.assembly,
                  progress=emit)
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # kB on Linux
    table.meta.update(peak_mb=round(peak_mb), threads=os.environ.get("OPENBLAS_NUM_THREADS"),
                      built_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                      inputs={s.filename: s.sha256 for s in sources_for(args.assembly)}
                      if args.assembly in ASSEMBLIES else {})
    save_table(table, Path(args.out))
    sys.stdout.write(json.dumps({"type": "done", "seconds": round(time.time() - t0, 1),
                                 "peak_mb": round(peak_mb), **{k: table.meta[k] for k in
                                                               ("n_genes", "n_motifs", "n_unreachable")}}) + "\n")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Builds that outlive the request that started them


@dataclass
class Build:
    assembly: str
    events: list = field(default_factory=list)
    done: bool = False
    error: str | None = None
    started: float = field(default_factory=time.time)

    def add(self, event: dict) -> None:
        self.events.append(event)

    @property
    def last(self) -> dict | None:
        return self.events[-1] if self.events else None


_builds: dict[str, Build] = {}
_builds_lock = threading.Lock()


def current_build(assembly: str) -> Build | None:
    with _builds_lock:
        b = _builds.get(assembly)
    return b if b is not None and not b.done else None


def start_build(dest: Path, assembly: str, *, auto_download: bool = True) -> Build:
    """The running build of ``assembly``, or a new one in a background thread (downloads,
    then the child process). It keeps going when the client that asked goes away."""
    with _builds_lock:
        b = _builds.get(assembly)
        if b is not None and not b.done:
            return b
        b = Build(assembly)
        _builds[assembly] = b

    def _run() -> None:
        try:
            files = ensure_files(dest, assembly, auto_download=auto_download, progress=b.add)
            final = run_child(files, table_path(dest, assembly), assembly, on_event=b.add)
            b.add({"stage": "done", "message": f"Built in {final.get('seconds')}s", **final})
        except Exception as e:  # noqa: BLE001 - reported to the client
            logger.warning("Motif table build for %s failed: %s", assembly, e)
            b.error = str(e)
        finally:
            b.done = True

    threading.Thread(target=_run, name=f"tf-motifs-{assembly}", daemon=True).start()
    return b


# ---------------------------------------------------------------------------
# Which assembly a dataset uses


_MM39 = re.compile(r"grcm39|mm39", re.IGNORECASE)


def assembly_for(adata, species: str) -> str | None:
    """hg38 for human, mm39 when the dataset names GRCm39 / mm39 (uns or a var ``genome``
    column), else mm10 for mouse; None for an unknown species."""
    if species == "human":
        return "hg38"
    if species != "mouse":
        return None
    hints: list[str] = []
    for key in ("genome", "assembly", "reference_genome", "reference", "genome_build"):
        v = adata.uns.get(key) if hasattr(adata, "uns") else None
        if isinstance(v, str):
            hints.append(v)
    for col in ("genome", "assembly"):
        if col in getattr(adata, "var", {}).columns:
            vals = adata.var[col].astype(str).unique()[:5]
            hints.extend(str(v) for v in vals)
    return "mm39" if any(_MM39.search(h) for h in hints) else "mm10"


# ---------------------------------------------------------------------------
# The test


def _index(table_genes: np.ndarray, genes: Iterable[str]) -> dict[str, int]:
    """Dataset gene name -> table row: the exact symbol, else the symbol ignoring case
    when only one table gene has it."""
    exact = {g: i for i, g in enumerate(table_genes.tolist())}
    upper: dict[str, int] = {}
    dup: set[str] = set()
    for g, i in exact.items():
        u = g.upper()
        if u in upper:
            dup.add(u)
        upper[u] = i
    out = {}
    for g in genes:
        if g in exact:
            out[g] = exact[g]
        elif g.upper() in upper and g.upper() not in dup:
            out[g] = upper[g.upper()]
    return out


def odds_ratio(k: np.ndarray, n: int, K: np.ndarray, N: int) -> np.ndarray:
    """Of the 2 x 2 table (marker / not, hit / not), 0.5 added to each cell when one is 0."""
    a = k.astype(np.float64)
    b = n - a
    c = K - a
    d = N - K - b
    zero = (a == 0) | (b == 0) | (c == 0) | (d == 0)
    a, b, c, d = (np.where(zero, x + 0.5, x) for x in (a, b, c, d))
    return (a * d) / (b * c)


def bh(p: np.ndarray) -> np.ndarray:
    from scipy import stats

    p = np.asarray(p, dtype=np.float64)
    return stats.false_discovery_control(p, method="bh") if p.size else p


def enrich(table: MotifTable, universe: Iterable[str], markers: dict[str, list[str]],
           *, min_overlap: int = MIN_OVERLAP, max_rows: int = MAX_ROWS) -> dict[str, Any]:
    """Per group: the motifs with at least ``min_overlap`` markers among their hits,
    lowest p first (up to ``max_rows``), with the group's numbers."""
    from scipy.stats import hypergeom

    universe = list(dict.fromkeys(str(g) for g in universe))
    idx = _index(table.genes, universe)
    # One dataset gene per table row (two names reaching one promoter count once).
    rows: dict[int, str] = {}
    for g in universe:
        if g in idx:
            rows.setdefault(idx[g], g)
    urows = np.fromiter(rows.keys(), dtype=np.int64, count=len(rows))
    names = np.asarray(list(rows.values()), dtype=object)
    N = int(urows.size)
    H = table.hits[urows] if N else np.zeros((0, table.hits.shape[1]), bool)
    K = H.sum(axis=0).astype(np.int64)
    tested = (K > 0) & table.reachable
    pos = {g: i for i, g in enumerate(names.tolist())}
    groups = []
    for group, genes in markers.items():
        q = sorted({pos[g] for g in genes if g in pos})
        n = len(q)
        entry: dict[str, Any] = {"group": str(group), "n_markers": len(genes), "n": n, "rows": [],
                                 "n_tested": int(tested.sum()), "n_significant": 0, "n_reported": 0}
        if n < min_overlap:
            entry["note"] = (f"{n} of its {len(genes)} marker genes have a promoter here; at least "
                             f"{min_overlap} are needed.")
            groups.append(entry)
            continue
        Hq = H[q]
        k = Hq.sum(axis=0).astype(np.int64)
        p = np.ones(k.size)
        p[tested] = hypergeom.sf(k[tested] - 1, N, K[tested], n)
        padj = np.ones(k.size)
        padj[tested] = bh(p[tested])
        orat = odds_ratio(k, n, K, N)
        keep = np.flatnonzero(tested & (k >= min_overlap))
        keep = keep[np.lexsort((-k[keep], p[keep]))]
        entry["n_significant"] = int(((padj < 0.05) & tested & (k >= min_overlap)).sum())
        entry["n_reported"] = int(keep.size)
        for j in keep[:max_rows]:
            overlap = [str(names[q[i]]) for i in np.flatnonzero(Hq[:, j])]
            entry["rows"].append({
                "motif_id": str(table.motif_ids[j]), "motif": str(table.motif_names[j]),
                "tfs": tf_members(str(table.motif_names[j])),
                "k": int(k[j]), "n": n, "K": int(K[j]), "N": N,
                "odds_ratio": round(float(orat[j]), 3), "p": float(p[j]), "padj": float(padj[j]),
                "gc": round(float(table.gc[j]), 3), "gc_rich": bool(table.gc_rich[j]),
                "overlap": overlap[:MAX_OVERLAP_GENES], "n_overlap": len(overlap),
            })
        groups.append(entry)
    return {"N": N, "n_motifs": int(table.hits.shape[1]), "n_motifs_tested": int(tested.sum()),
            "n_unreachable": int((~table.reachable).sum()), "groups": groups}


def tf_expressed(tfs: list[str], group: str, stats: dict[str, dict]) -> dict | None:
    """The member of ``tfs`` (a dimer's two, a single TF's one) most expressed in ``group``:
    gene, share of the group's cells expressing it, log fold change from the group's
    marker stats (None when the stored result does not rank it). None when no member is
    measured. ``stats``: TF symbol upper-cased -> {gene, pct: {group: x}, logfc: {group: x}}."""
    best = None
    for t in tfs:
        s = stats.get(t.upper())
        if s is None:
            continue
        pct = s["pct"].get(group)
        if pct is None:
            continue
        if best is None or pct > best["pct"]:
            lfc = s.get("logfc", {}).get(group)
            best = {"gene": s["gene"], "pct": round(float(pct), 3),
                    "logfc": None if lfc is None or not math.isfinite(lfc) else round(float(lfc), 3),
                    "expressed": bool(pct >= EXPRESSED_PCT)}
    return best


def activity_agrees(tfs: list[str], group: str, activity: dict | None) -> bool | None:
    """Whether CollecTRI activity backs the candidate in ``group``: some member's mean
    activity in the group is above the rest of the cells' and in the group's top 10% of
    TFs. None without an activity result, or when no member is a CollecTRI TF.
    ``activity``: {"rank": {group: {TF upper: rank (1 = highest group-vs-rest)}},
    "diff": {group: {TF upper: difference}}}."""
    if not activity:
        return None
    ranks, diffs = activity["rank"].get(group), activity["diff"].get(group)
    if ranks is None:
        return None
    top = max(MIN_ACTIVITY_RANK, math.ceil(TOP_ACTIVITY_SHARE * max(len(ranks), 1)))
    seen = False
    for t in tfs:
        r = ranks.get(t.upper())
        if r is None:
            continue
        seen = True
        if diffs[t.upper()] > 0 and r <= top:
            return True
    return False if seen else None


def activity_ranks(groups: list[str], sources: list[str], means: np.ndarray,
                   sizes: np.ndarray) -> dict:
    """Group-vs-rest CollecTRI mean activity differences and their rank within each group,
    from the groups x sources means and the cells per group."""
    means = np.nan_to_num(np.asarray(means, dtype=np.float64))
    sizes = np.asarray(sizes, dtype=np.float64)
    total = (means * sizes[:, None]).sum(axis=0)
    n_all = sizes.sum()
    rank: dict[str, dict] = {}
    diff: dict[str, dict] = {}
    up = [s.upper() for s in sources]
    for i, g in enumerate(groups):
        rest_n = n_all - sizes[i]
        if sizes[i] == 0 or rest_n <= 0:
            continue
        d = means[i] - (total - means[i] * sizes[i]) / rest_n
        order = np.argsort(-d, kind="stable")
        r = np.empty(len(d), dtype=np.int64)
        r[order] = np.arange(1, len(d) + 1)
        rank[g] = dict(zip(up, r.tolist()))
        diff[g] = dict(zip(up, d.tolist()))
    return {"rank": rank, "diff": diff}


def marker_stats(adata, column: str, tfs: Iterable[str], M, genes) -> dict[str, dict]:
    """For each TF symbol measured in the log-normalised matrix ``M`` (cells x ``genes``):
    the share of each group's cells expressing it, and its log fold change per group
    from the stored marker result for ``column`` (when that result ranks it). Keyed by
    the symbol upper-cased; a symbol two dataset genes share ignoring case is matched
    exactly only."""
    import pandas as pd
    import scipy.sparse as sp

    from scview.core.ora_background import resolve_markers_key

    names = [str(g) for g in genes]
    exact = {g: i for i, g in enumerate(names)}
    upper: dict[str, list[int]] = {}
    for i, g in enumerate(names):
        upper.setdefault(g.upper(), []).append(i)
    pos: dict[str, int] = {}
    for t in tfs:
        if t in exact:
            pos[t.upper()] = exact[t]
        elif len(upper.get(t.upper(), [])) == 1:
            pos[t.upper()] = upper[t.upper()][0]
    if not pos:
        return {}
    col = adata.obs[column]
    labels = col.astype(str).where(col.notna())
    groups = sorted(set(labels.dropna()))
    codes = pd.Index(groups).get_indexer(labels)
    ok = codes >= 0
    sizes = np.bincount(codes[ok], minlength=len(groups)).astype(np.float64)
    onehot = sp.csr_matrix((np.ones(int(ok.sum())), (codes[ok], np.flatnonzero(ok))),
                           shape=(len(groups), len(codes)))
    keys = list(pos)
    sub = M[:, [pos[k] for k in keys]]
    if sp.issparse(sub):
        sub = sp.csr_matrix(sub)
        nz = sp.csr_matrix(((sub.data != 0).astype(np.float64), sub.indices, sub.indptr), shape=sub.shape)
    else:
        nz = (np.asarray(sub) != 0).astype(np.float64)
    expressing = onehot @ nz
    expressing = expressing.toarray() if sp.issparse(expressing) else np.asarray(expressing)
    with np.errstate(divide="ignore", invalid="ignore"):
        pct = np.where(sizes[:, None] > 0, expressing / sizes[:, None], np.nan)
    lfc: dict[str, dict[str, float]] = {}
    key = resolve_markers_key(adata.uns, column)
    try:
        rgg = adata.uns[key] if key else None
        if rgg is not None and rgg.get("logfoldchanges") is not None:
            for g in rgg["names"].dtype.names:
                gn = np.asarray(rgg["names"][g]).astype(str)
                lf = np.asarray(rgg["logfoldchanges"][g], dtype=np.float64)[: gn.size]
                wanted = {n.upper(): float(v) for n, v in zip(gn, lf) if n.upper() in pos}
                for k, v in wanted.items():
                    lfc.setdefault(k, {})[str(g)] = v
    except (KeyError, TypeError, AttributeError, ValueError):
        lfc = {}
    return {k: {"gene": names[pos[k]],
                "pct": {g: float(pct[gi, j]) for gi, g in enumerate(groups) if np.isfinite(pct[gi, j])},
                "logfc": lfc.get(k, {})}
            for j, k in enumerate(keys)}


def record(adata, result: dict) -> None:
    """The run in provenance (History): a result only, the data are unchanged."""
    from scview.core import provenance

    provenance.record_step(
        adata, step=STEP, tool="scview.tf_motifs (JASPAR 2024 promoter motifs, hypergeometric)",
        params={"column": result["column"], "assembly": result["assembly"],
                "n_markers": result["n_markers_requested"], "p_threshold": P_THRESHOLD,
                "pseudocount": PSEUDOCOUNT, "min_overlap": MIN_OVERLAP, "promoter_bp": PROMOTER_LEN,
                "universe": result["N"], "n_motifs_tested": result["n_motifs_tested"],
                "table": cache_key(result["assembly"]) if result["assembly"] in ASSEMBLIES else None,
                "result_id": result["result_id"]},
        resources={"seconds": result.get("elapsed_seconds")},
    )


def annotate(result: dict, stats: dict[str, dict], activity: dict | None) -> None:
    """Adds ``tf_expressed`` and ``activity_agrees`` to every reported row."""
    for g in result["groups"]:
        for r in g["rows"]:
            r["tf_expressed"] = tf_expressed(r["tfs"], g["group"], stats)
            r["activity_agrees"] = activity_agrees(r["tfs"], g["group"], activity)


# ---------------------------------------------------------------------------
# Kept results and the co-pilot


_last: Any = None  # weakref.WeakKeyDictionary, made on first use
_last_lock = threading.Lock()


def _cache():
    global _last
    if _last is None:
        import weakref

        _last = weakref.WeakKeyDictionary()
    return _last


def remember(adaptor, result: dict) -> None:
    from scview.core import results_store

    try:
        with _last_lock:
            _cache()[adaptor] = result
    except TypeError:  # a test double
        pass
    results_store.keep(adaptor, KIND, result, key=result["column"], result_id=result["result_id"],
                       label=f"Upstream TF candidates (promoter motifs): '{result['column']}'")


def last(adaptor) -> dict | None:
    from scview.core import results_store

    try:
        return results_store.remembered(_cache(), _last_lock, adaptor, KIND)
    except Exception:  # noqa: BLE001 - a test double, an unreadable result
        return None


def _row_text(r: dict) -> str:
    gc = ", GC-rich" if r.get("gc_rich") else ""
    return f"{r['motif']} ({r['motif_id']}) {r['k']}/{r['n']} markers, padj {r['padj']:.2g}{gc}"


def grounding_lines(adaptor, highlighted: dict | None = None) -> tuple[list[str], str | None]:
    """The last run's candidate TFs: for the highlighted group when it is in the run, else
    up to 3 per group (padj < 0.05)."""
    r = last(adaptor)
    if not r:
        return [], None
    detail = (f"JASPAR 2024 motifs in 1 kb promoters ({r['assembly']}), markers of each "
              f"'{r['column']}' group vs {r['N']:,} measured genes with a promoter")
    lines = ["\n## Upstream TF candidates from promoter motifs (candidates, not proven regulators)",
             f"- {detail}; hypergeometric, BH per group [{TAG}]."]
    hl = highlighted if isinstance(highlighted, dict) else {}
    wanted = [g for g in r["groups"] if hl.get("column") == r["column"] and g["group"] == str(hl.get("value"))]
    for g in (wanted or r["groups"][:12]):
        sig = [x for x in g["rows"] if x["padj"] < 0.05][: 5 if wanted else 3]
        if not sig:
            lines.append(f"- TF motif candidates, {g['group']}: none with padj < 0.05")
            continue
        lines.append(f"- TF motif candidates, {g['group']}: " + "; ".join(_row_text(x) for x in sig))
    lines.append("- Caveat: promoter motifs only; GC-rich motifs enrich easily; paralogues share motifs; "
                 "a motif hit is not binding. Say 'candidate'.")
    return lines, detail


if __name__ == "__main__":
    _child(sys.argv[1:])
