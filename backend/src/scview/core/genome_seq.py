"""Genome sequence for peaks: GC content and the bases a motif scan reads.

Peak-gene links match background peaks on GC content, and motif scans read each peak's
bases, so both need the reference sequence. UCSC's ``hg38.2bit``, ``mm10.2bit`` and
``mm39.2bit`` hold the same bases as Bioconductor's BSgenome.Hsapiens.UCSC.hg38 and
BSgenome.Mmusculus.UCSC.mm10 / .mm39, which Signac uses. 700 to 840 MB each, fetched on first use into ``DATA_DIR/genome``, sha256
pinned; ``GENOME_AUTO_DOWNLOAD=false`` turns fetching off and a file placed there by hand is
used when its sha256 matches.

No new dependency: the 2bit format is a small index plus 2 bits per base, read through a
memory map, so only the pages a peak touches are loaded and they count as reclaimable cache.

Coordinates are 0-based half open, as in BED, ``gene_annotation`` and 10x fragments. Base
codes match ``tf_motifs.encode``: A 0, C 1, G 2, T 3, unknown 4. Lower-case repeat masking
is ignored. A contig UCSC names differently (10x ``KI270728.1``, UCSC
``chr1_KI270728v1_random``) is reported as unknown bases, not guessed: peaks there have a
NaN GC and no scan hits, and callers leave them out and say how many.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import mmap
import shutil
import struct
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import httpx
import numpy as np

from scview.core.gene_annotation import canonical_genome

SUBDIR = "genome"
UNKNOWN = 4
TIMEOUT = httpx.Timeout(120.0, connect=20.0)
_SIGNATURE = 0x1A412743
_FREE_MARGIN = 200 * 1024**2  # left free on the volume after a download


@dataclass(frozen=True)
class Release:
    genome: str
    source: str
    url: str
    sha256: str
    size: int
    filename: str


RELEASES = {
    "GRCh38": Release("GRCh38", "UCSC hg38",
                      "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.2bit",
                      "1f67aaa17a77b327738fe750ab37430a85dddf73c7d9189385ad1839259acec0", 835_393_456, "hg38.2bit"),
    "mm10": Release("mm10", "UCSC mm10",
                    "https://hgdownload.soe.ucsc.edu/goldenPath/mm10/bigZips/mm10.2bit",
                    "4de06ca0bfca5417ae3782c09cf2caf7afc97c190172f60dc07b783f1bc6f39d", 714_784_109, "mm10.2bit"),
    # sha256 of the file as fetched 2026-11-13 (dated 2020-07-30); its md5 matches UCSC's md5sum.txt.
    "mm39": Release("mm39", "UCSC mm39",
                    "https://hgdownload.soe.ucsc.edu/goldenPath/mm39/bigZips/mm39.2bit",
                    "40b4c36f83e761a40ba6d1b5357435ae56dd4df4db6cbafd7f5b1a4c85c2e360", 714_181_470, "mm39.2bit"),
}


class GenomeUnavailable(RuntimeError):
    """The sentence says which genome, and what to do."""


class GenomeCancelled(Exception):
    """A stop request ended the download; nothing partial is left."""


# 2bit packs T 00, C 01, A 10, G 11, first base in the high bits.
_FROM_2BIT = np.array([3, 1, 0, 2], dtype=np.uint8)
_BYTE_TO_CODES = np.array(
    [[_FROM_2BIT[(b >> s) & 3] for s in (6, 4, 2, 0)] for b in range(256)], dtype=np.uint8,
)


class TwoBit:
    """Random access to a .2bit file. Thread-safe for reads: nothing here seeks."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._fh = open(self.path, "rb")
        self._mm = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)
        head = self._mm[:16]
        if len(head) < 16:
            raise ValueError(f"{self.path.name} is too short to be a 2bit file.")
        sig_le = struct.unpack("<I", head[:4])[0]
        if sig_le == _SIGNATURE:
            self._order = "<"
        elif struct.unpack(">I", head[:4])[0] == _SIGNATURE:
            self._order = ">"
        else:
            raise ValueError(f"{self.path.name} is not a 2bit file.")
        version, count = struct.unpack(self._order + "II", head[4:12])
        if version != 0:
            raise ValueError(f"{self.path.name} is 2bit version {version}; only version 0 is read.")
        self._offsets: dict[str, int] = {}
        pos = 16
        for _ in range(count):
            n = self._mm[pos]
            name = self._mm[pos + 1:pos + 1 + n].decode("ascii")
            pos += 1 + n
            self._offsets[name] = struct.unpack(self._order + "I", self._mm[pos:pos + 4])[0]
            pos += 4
        self._records: dict[str, tuple[int, np.ndarray, np.ndarray, int]] = {}
        self._lock = threading.Lock()

    @property
    def names(self) -> list[str]:
        return list(self._offsets)

    def _record(self, chrom: str) -> tuple[int, np.ndarray, np.ndarray, int]:
        """(size, unknown-run starts, unknown-run ends, offset of the packed bases)."""
        with self._lock:
            rec = self._records.get(chrom)
        if rec is not None:
            return rec
        u4 = np.dtype(self._order + "u4")
        pos = self._offsets[chrom]
        size, n_blocks = struct.unpack(self._order + "II", self._mm[pos:pos + 8])
        pos += 8
        starts = np.frombuffer(self._mm, u4, n_blocks, pos).astype(np.int64)
        pos += 4 * n_blocks
        ends = starts + np.frombuffer(self._mm, u4, n_blocks, pos).astype(np.int64)
        pos += 4 * n_blocks
        n_mask = struct.unpack(self._order + "I", self._mm[pos:pos + 4])[0]
        pos += 4 + 8 * n_mask + 4  # mask starts and sizes, then a reserved word
        rec = (int(size), starts, ends, pos)
        with self._lock:
            self._records[chrom] = rec
        return rec

    def has(self, chrom: str) -> bool:
        return chrom in self._offsets

    def size(self, chrom: str) -> int:
        return self._record(chrom)[0]

    def fetch(self, chrom: str, start: int, end: int) -> np.ndarray:
        """Base codes of [start, end). Outside the contig, or on a contig this file does
        not have, the bases are unknown (4); the result always has ``end - start`` bases."""
        length = max(int(end) - int(start), 0)
        out = np.full(length, UNKNOWN, dtype=np.uint8)
        if length == 0 or chrom not in self._offsets:
            return out
        size, n_starts, n_ends, base = self._record(chrom)
        lo, hi = max(int(start), 0), min(int(end), size)
        if lo >= hi:
            return out
        first, last = lo // 4, (hi - 1) // 4
        packed = np.frombuffer(self._mm, np.uint8, last - first + 1, base + first)
        codes = _BYTE_TO_CODES[packed].ravel()[lo - first * 4:hi - first * 4]
        i = int(np.searchsorted(n_ends, lo, side="right"))
        while i < n_starts.size and n_starts[i] < hi:
            codes[max(int(n_starts[i]) - lo, 0):max(int(n_ends[i]) - lo, 0)] = UNKNOWN
            i += 1
        out[lo - int(start):hi - int(start)] = codes
        return out

    def close(self) -> None:
        self._mm.close()
        self._fh.close()


def sequences(genome: TwoBit, chroms: Iterable[str], starts: Iterable[int],
              ends: Iterable[int]) -> tuple[np.ndarray, np.ndarray]:
    """Many regions as one array: (codes of every region end to end, offsets of length
    n + 1). Region i is ``codes[offsets[i]:offsets[i + 1]]``."""
    chroms = list(chroms)
    s = np.asarray(list(starts), dtype=np.int64)
    e = np.asarray(list(ends), dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(np.maximum(e - s, 0))]).astype(np.int64)
    codes = np.empty(int(offsets[-1]), dtype=np.uint8)
    for i, chrom in enumerate(chroms):
        codes[offsets[i]:offsets[i + 1]] = genome.fetch(chrom, int(s[i]), int(e[i]))
    return codes, offsets


def base_counts(codes: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """n x 5 counts per region of A, C, G, T, unknown."""
    n = offsets.size - 1
    region = np.repeat(np.arange(n), np.diff(offsets))
    return np.bincount(region * 5 + codes.astype(np.int64), minlength=n * 5).reshape(n, 5)


def gc_percent(counts: np.ndarray) -> np.ndarray:
    """100 x (G + C) / region length, as Signac's RegionStats. NaN for a region with no
    known base (a contig the genome file does not name)."""
    counts = np.asarray(counts, dtype=np.float64)
    length = counts.sum(axis=1)
    known = counts[:, :4].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        gc = 100.0 * (counts[:, 1] + counts[:, 2]) / length
    gc[known == 0] = np.nan
    return gc


def background(counts: np.ndarray) -> np.ndarray:
    """Base frequencies (A, C, G, T) over the regions, as they read, not strand-averaged."""
    total = np.asarray(counts, dtype=np.float64)[:, :4].sum(axis=0)
    return total / total.sum() if total.sum() > 0 else np.full(4, 0.25)


# ---------------------------------------------------------------------------
# The file


def file_path(data_dir: str | Path, genome: str | None) -> Path:
    return Path(data_dir) / SUBDIR / RELEASES[canonical_genome(genome)].filename


def _stamp(path: Path) -> Path:
    return path.with_name(path.name + ".verified")


def _is_verified(path: Path, release: Release) -> bool:
    """Whether the file is the pinned one. Hashing 800 MB takes seconds, so a good result
    is remembered beside the file, keyed on its size and modification time."""
    if not path.is_file():
        return False
    st = path.stat()
    key = {"sha256": release.sha256, "size": st.st_size, "mtime_ns": st.st_mtime_ns}
    stamp = _stamp(path)
    try:
        if json.loads(stamp.read_text()) == key:
            return True
    except (OSError, ValueError):
        pass
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    if h.hexdigest() != release.sha256:
        return False
    stamp.write_text(json.dumps(key))
    return True


def status(data_dir: str | Path, genome: str | None) -> dict:
    """What a card shows before a run: whether the file is here and what a fetch costs.
    Never hashes (safe on the event loop)."""
    release = RELEASES[canonical_genome(genome)]
    path = file_path(data_dir, genome)
    return {"genome": release.genome, "source": release.source, "url": release.url,
            "size": release.size, "downloaded": path.is_file()}


def _download(release: Release, path: Path, say: Callable[[dict], None],
              stop: Callable[[], bool]) -> None:
    part = path.with_name(f".{path.name}.part")
    free = shutil.disk_usage(path.parent).free
    if free < release.size + _FREE_MARGIN:
        raise GenomeUnavailable(
            f"The {release.source} sequence needs {release.size / 1e9:.1f} GB and the data volume "
            f"has {free / 1e9:.1f} GB free. Free some space and try again.")
    digest = hashlib.sha256()
    size = 0
    last = 0
    try:
        with httpx.Client(timeout=TIMEOUT, follow_redirects=True) as client, \
                client.stream("GET", release.url) as resp:
            resp.raise_for_status()
            with open(part, "wb") as fh:
                for chunk in resp.iter_bytes(1 << 20):
                    if stop():
                        raise GenomeCancelled
                    size += len(chunk)
                    if size > release.size:
                        raise GenomeUnavailable(f"The {release.source} download is larger than expected.")
                    digest.update(chunk)
                    fh.write(chunk)
                    if size - last >= 50_000_000:
                        last = size
                        say({"stage": "download", "done": size, "total": release.size,
                             "message": f"Downloading {release.source}: {size / 1e6:,.0f} of "
                                        f"{release.size / 1e6:,.0f} MB"})
        if digest.hexdigest() != release.sha256:
            raise GenomeUnavailable(f"The {release.source} download is not the file scView expects.")
        part.replace(path)
    except httpx.HTTPError as e:
        raise GenomeUnavailable(
            f"Could not download the {release.source} sequence ({e}). Check the network and try again, "
            f"or place {release.filename} (from {release.url}) in {path.parent}.") from e
    finally:
        part.unlink(missing_ok=True)


_open: dict[str, TwoBit] = {}
_open_lock = threading.Lock()


def ensure(data_dir: str | Path, genome: str | None, *, auto_download: bool = True,
           progress: Callable[[dict], None] | None = None,
           stop: Callable[[], bool] = lambda: False) -> TwoBit:
    """The open genome, fetched the first time. Blocks on the download and hashing, so never
    call it on the event loop. Two runs asking at once share one download (file lock)."""
    release = RELEASES[canonical_genome(genome)]
    path = file_path(data_dir, genome)
    say = progress or (lambda _e: None)
    with _open_lock:
        opened = _open.get(str(path))
    if opened is not None and path.is_file():
        return opened
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.parent / f".{release.filename}.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.is_file() and not _is_verified(path, release):
            raise GenomeUnavailable(
                f"{path} is not the pinned {release.source} file (sha256 differs); delete it to fetch it again.")
        if not path.is_file():
            if not auto_download:
                raise GenomeUnavailable(
                    f"The {release.source} sequence is not on this machine and GENOME_AUTO_DOWNLOAD is off. "
                    f"Place {release.filename} (from {release.url}) in {path.parent}, or turn the switch on.")
            say({"stage": "download", "done": 0, "total": release.size,
                 "message": f"Downloading {release.source} ({release.size / 1e6:,.0f} MB, once)"})
            _download(release, path, say, stop)
            _is_verified(path, release)
    two_bit = TwoBit(path)
    with _open_lock:
        return _open.setdefault(str(path), two_bit)


def clear_cache() -> None:
    with _open_lock:
        for g in _open.values():
            g.close()
        _open.clear()
