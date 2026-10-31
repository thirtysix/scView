"""Reading a 10x ATAC fragments file: BGZF + tabix without pysam.

``atac_fragments.tsv.gz`` (Cell Ranger ARC / ATAC) holds one line per fragment,
``chrom start end barcode count``, 0-based half open, sorted by position, in BGZF: a chain
of gzip members of at most 64 KB, each giving its own compressed size in a gzip extra
field. ``.tbi`` beside it says which blocks hold a region (htslib ``tabix.pdf``, MIT:
UCSC binning + a 16 kb linear index; a virtual offset is ``block offset << 16 | offset in
the uncompressed block``). 10x PBMC multiome is 2.4 GB of this, about 9 GB of text.

Parsing is numpy on the raw bytes, not a text reader: tabs and newlines are found once,
barcodes are hashed as 64-bit words and compared with the cells' hashes, and only cell
lines get their numbers parsed. ``parse`` falls back to pandas for a line shape the fast
path does not expect (extra columns, a comment line in the middle).
"""

from __future__ import annotations

import io
import os
import queue
import struct
import threading
import zlib
from collections import OrderedDict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

_READ = 4 * 1024 * 1024
_MAX_BLOCK = 65536
TEXT_BLOCK = 32 * 1024 * 1024  # text parsed at a time in a full pass
_MIN_SHIFT = 14
_DEPTH = 5
_HASH = np.array([0x9E3779B185EBCA87, 0xC2B2AE3D27D4EB4F, 0x165667B19E3779F9, 0x27D4EB2F165667C5], dtype=np.uint64)


class FragmentFileError(ValueError):
    """The sentence says what is wrong with the file and what to do."""


def is_bgzf(head: bytes) -> bool:
    """A BGZF block header: gzip + FEXTRA + the subfield 'BC' of length 2."""
    if len(head) < 18 or head[:4] != b"\x1f\x8b\x08\x04":
        return False
    xlen = struct.unpack_from("<H", head, 10)[0]
    extra = head[12:12 + xlen]
    pos = 0
    while pos + 4 <= len(extra):
        si1, si2, slen = struct.unpack_from("<BBH", extra, pos)
        if si1 == 66 and si2 == 67 and slen == 2:
            return True
        pos += 4 + slen
    return False


def _block_size(buf, pos: int) -> int:
    """Total size of the BGZF block starting at ``pos`` of ``buf``, -1 when ``buf`` does not
    hold its header yet."""
    if len(buf) - pos < 18:
        return -1
    if buf[pos:pos + 4] != b"\x1f\x8b\x08\x04":
        raise FragmentFileError("The fragments file is not BGZF past its first block: it is damaged; "
                                "use the file Cell Ranger ARC wrote.")
    xlen = struct.unpack_from("<H", buf, pos + 10)[0]
    if len(buf) - pos < 12 + xlen:
        return -1
    p = pos + 12
    stop = p + xlen
    while p + 4 <= stop:
        si1, si2, slen = struct.unpack_from("<BBH", buf, p)
        if si1 == 66 and si2 == 67 and slen == 2:
            return struct.unpack_from("<H", buf, p + 4)[0] + 1
        p += 4 + slen
    raise FragmentFileError("A block of the fragments file has no BGZF size field: it is damaged; "
                            "use the file Cell Ranger ARC wrote.")


def _inflate(block) -> bytes:
    xlen = struct.unpack_from("<H", block, 10)[0]
    try:
        return zlib.decompress(bytes(block[12 + xlen:-8]), wbits=-15)
    except zlib.error as e:
        raise FragmentFileError("A block of the fragments file does not decompress: it is damaged; upload it "
                                "again.") from e


def iter_bgzf(path: str | Path) -> Iterator[tuple[int, bytes]]:
    """Every block of a BGZF file in order: (compressed bytes read so far, text)."""
    with open(path, "rb") as fh:
        buf = b""
        done = 0
        while True:
            chunk = fh.read(_READ)
            buf = buf + chunk if buf else chunk
            pos = 0
            while True:
                size = _block_size(buf, pos)
                if size < 0 or len(buf) - pos < size:
                    break
                yield done + pos + size, _inflate(memoryview(buf)[pos:pos + size])
                pos += size
            done += pos
            buf = buf[pos:]
            if not chunk:
                if buf:
                    raise FragmentFileError("The fragments file is cut off inside a block: upload it again.")
                return


def read_all_bgzf(path: str | Path) -> bytes:
    return b"".join(text for _, text in iter_bgzf(path))


def _read_ahead(items: Iterator, size: int = 2) -> Iterator:
    """``items`` produced on a second thread, at most ``size`` ahead: inflating releases the
    GIL, so one pass uses two cores and no more."""
    q: queue.Queue = queue.Queue(maxsize=size)
    stop = threading.Event()
    end = object()

    def put(item) -> bool:
        while not stop.is_set():
            try:
                q.put(item, timeout=0.2)
                return True
            except queue.Full:
                continue
        return False

    def work() -> None:
        try:
            for item in items:
                if not put(item):
                    return
            put(end)
        except BaseException as e:  # noqa: BLE001 - raised on the reading side
            put(e)

    thread = threading.Thread(target=work, name="fragments-inflate", daemon=True)
    thread.start()
    try:
        while True:
            item = q.get()
            if item is end:
                return
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        stop.set()
        thread.join(timeout=5)


def iter_text(path: str | Path, block: int = TEXT_BLOCK) -> Iterator[tuple[int, bytes]]:
    """The file's text in pieces of whole lines of about ``block`` bytes: (compressed bytes
    read so far, text)."""

    def pieces() -> Iterator[tuple[int, bytes]]:
        parts: list[bytes] = []
        size = 0
        done = 0
        for done, text in iter_bgzf(path):
            parts.append(text)
            size += len(text)
            if size >= block:
                joined = b"".join(parts)
                cut = joined.rfind(b"\n") + 1
                if cut:
                    yield done, joined[:cut]
                    parts = [joined[cut:]]
                    size = len(parts[0])
                else:
                    parts = [joined]
        tail = b"".join(parts)
        if tail:
            yield done, tail if tail.endswith(b"\n") else tail + b"\n"

    return _read_ahead(pieces())


# ---------------------------------------------------------------------------
# Barcodes
# ---------------------------------------------------------------------------


def _hash_rows(rows: np.ndarray) -> np.ndarray:
    """64-bit hash of each row of an (n, width) uint8 array. Colliding with a cell barcode
    takes about 1e-8 per file of a few million barcodes."""
    n, width = rows.shape
    words = (width + 7) // 8
    padded = np.zeros((n, words * 8), dtype=np.uint8)
    padded[:, :width] = rows
    w = padded.view(np.uint64).reshape(n, words)
    out = np.full(n, 0xCBF29CE484222325 ^ width, dtype=np.uint64)
    for i in range(words):
        out = (out ^ w[:, i]) * _HASH[i % len(_HASH)]
        out ^= out >> np.uint64(29)
    return out


class Barcodes:
    """Maps a file's barcodes to a position in the dataset's list."""

    def __init__(self, names: Sequence[str]) -> None:
        self.names = [str(b) for b in names]
        hashes = np.zeros(len(self.names), dtype=np.uint64)
        by_width: dict[int, list[int]] = {}
        encoded = [b.encode() for b in self.names]
        for i, b in enumerate(encoded):
            by_width.setdefault(len(b), []).append(i)
        for width, idx in by_width.items():
            if width == 0:
                continue
            rows = np.frombuffer(b"".join(encoded[i] for i in idx), dtype=np.uint8).reshape(len(idx), width)
            hashes[idx] = _hash_rows(rows)
        self.order = np.argsort(hashes, kind="stable")
        self.sorted = hashes[self.order]

    def __len__(self) -> int:
        return len(self.names)

    def lookup(self, hashes: np.ndarray) -> np.ndarray:
        """Position of each hash, -1 where it is no barcode of the list."""
        if not len(self.sorted):
            return np.full(hashes.shape, -1, dtype=np.int64)
        pos = np.searchsorted(self.sorted, hashes)
        pos[pos == len(self.sorted)] = len(self.sorted) - 1
        hit = self.sorted[pos] == hashes
        out = self.order[pos]
        out[~hit] = -1
        return out

    def hash_of(self, name: str) -> np.uint64:
        raw = np.frombuffer(name.encode(), dtype=np.uint8).reshape(1, -1)
        return _hash_rows(raw)[0]


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


@dataclass
class Parsed:
    """Lines of one piece of text. ``chroms``: (name, first, stop) over the kept lines, in
    file order. ``barcode``: position in the Barcodes list, -1 for another barcode."""

    chroms: list[tuple[str, int, int]]
    start: np.ndarray
    end: np.ndarray
    barcode: np.ndarray
    count: np.ndarray | None
    n_lines: int


def _ints(a: np.ndarray, first: np.ndarray, stop: np.ndarray) -> np.ndarray:
    """Decimal numbers at a[first:stop] per line. A field of the same width is parsed
    together: a few widths, up to 9 digits each."""
    width = stop - first
    out = np.zeros(first.size, dtype=np.int64)
    if not first.size:
        return out
    low, high = int(width.min()), int(width.max())
    if low < 1 or high > 18:
        raise FragmentFileError("A line of the fragments file has an empty or oversized number: "
                                "use the file Cell Ranger ARC wrote.")
    if low == high:
        groups = [(low, None)]
    else:
        groups = [(int(w), np.flatnonzero(width == w)) for w in range(low, high + 1)]
    for w, idx in groups:
        if idx is not None and not idx.size:
            continue
        at = first if idx is None else first[idx]
        val = np.zeros(at.size, dtype=np.int64)
        bad = np.zeros(at.size, dtype=bool)
        for j in range(w):
            digit = a[at + j].astype(np.int64) - 48
            bad |= (digit < 0) | (digit > 9)
            val = val * 10 + digit
        if bad.any():
            raise FragmentFileError("A line of the fragments file has a position that is not a number: "
                                    "use the file Cell Ranger ARC wrote.")
        if idx is None:
            out = val
        else:
            out[idx] = val
    return out


def _hash_fields(a: np.ndarray, first: np.ndarray, stop: np.ndarray) -> np.ndarray:
    width = stop - first
    out = np.zeros(first.size, dtype=np.uint64)
    if not first.size:
        return out
    low, high = int(width.min()), int(width.max())
    if low == high:
        if low:
            out = _hash_rows(a[first[:, None] + np.arange(low)])
        return out
    for w in np.unique(width):
        if w:
            idx = np.flatnonzero(width == w)
            out[idx] = _hash_rows(a[first[idx][:, None] + np.arange(int(w))])
    return out


def _segments(buf: bytes, line_first: np.ndarray, tab: np.ndarray) -> list[tuple[str, int, int]]:
    """(chromosome, first line, stop line): the lines of a chromosome are next to each other
    in a sorted file, so each boundary is a binary search."""
    out: list[tuple[str, int, int]] = []
    n = line_first.size
    i = 0
    while i < n:
        name = buf[line_first[i]:tab[i]]
        if buf[line_first[n - 1]:tab[n - 1]] == name:
            stop = n
        else:
            lo, hi = i, n - 1  # lo: has the name, hi: does not
            while hi - lo > 1:
                mid = (lo + hi) // 2
                if buf[line_first[mid]:tab[mid]] == name:
                    lo = mid
                else:
                    hi = mid
            stop = hi
        out.append((name.decode(), i, stop))
        i = stop
    return out


def _parse_slow(buf: bytes, barcodes: Barcodes, cells_only: bool, want_count: bool) -> Parsed:
    """Any tab-separated fragments text: extra columns, comments anywhere."""
    table = pd.read_csv(io.BytesIO(buf), sep="\t", header=None, comment="#", usecols=[0, 1, 2, 3, 4],
                        names=["chrom", "start", "end", "barcode", "count"], engine="c",
                        dtype={"chrom": str, "start": np.int64, "end": np.int64, "barcode": str, "count": np.int64})
    index = {b: i for i, b in enumerate(barcodes.names)}
    bc = table["barcode"].map(index).fillna(-1).to_numpy(dtype=np.int64)
    if cells_only:
        table = table[bc >= 0]
        bc = bc[bc >= 0]
    chrom = table["chrom"].to_numpy()
    edges = np.flatnonzero(chrom[1:] != chrom[:-1]) + 1 if len(chrom) else np.zeros(0, dtype=np.int64)
    first = np.r_[0, edges] if len(chrom) else np.zeros(0, dtype=np.int64)
    stop = np.r_[edges, len(chrom)] if len(chrom) else np.zeros(0, dtype=np.int64)
    return Parsed([(str(chrom[i]), int(i), int(j)) for i, j in zip(first, stop)],
                  table["start"].to_numpy(), table["end"].to_numpy(), bc,
                  table["count"].to_numpy() if want_count else None, int(buf.count(b"\n") - buf.count(b"#")))


def parse(buf: bytes, barcodes: Barcodes, cells_only: bool = True, want_count: bool = False) -> Parsed:
    """Parse whole lines. ``cells_only`` keeps only lines of a listed barcode, and parses the
    numbers of those alone."""
    # Comment lines sit at the top of the file.
    skip = 0
    while buf.startswith(b"#", skip):
        nl = buf.find(b"\n", skip)
        if nl < 0:
            skip = len(buf)
            break
        skip = nl + 1
    if skip:
        buf = buf[skip:]
    a = np.frombuffer(buf, dtype=np.uint8)
    nl = np.flatnonzero(a == 10)
    n = nl.size
    empty = np.zeros(0, dtype=np.int64)
    if not n:
        return Parsed([], empty, empty, empty, empty if want_count else None, 0)
    tabs = np.flatnonzero(a == 9)
    if tabs.size != 4 * n or not (tabs[3::4] < nl).all() or not (tabs[4::4] > nl[:-1]).all():
        return _parse_slow(buf, barcodes, cells_only, want_count)
    first = np.empty(n, dtype=np.int64)
    first[0] = 0
    first[1:] = nl[:-1] + 1
    if (a[first] == 35).any():
        return _parse_slow(buf, barcodes, cells_only, want_count)
    t1, t2, t3, t4 = tabs[0::4], tabs[1::4], tabs[2::4], tabs[3::4]
    bc = barcodes.lookup(_hash_fields(a, t3 + 1, t4))
    segments = _segments(buf, first, t1)
    if cells_only:
        keep = np.flatnonzero(bc >= 0)
        cuts = np.searchsorted(keep, [s for _, s, _ in segments] + [n])
        chroms = [(name, int(cuts[i]), int(cuts[i + 1])) for i, (name, _, _) in enumerate(segments)
                  if cuts[i + 1] > cuts[i]]
        t2, t3, t4, nl_k, bc = t2[keep], t3[keep], t4[keep], nl[keep], bc[keep]
        t1 = t1[keep]
    else:
        chroms = segments
        nl_k = nl
    return Parsed(chroms, _ints(a, t1 + 1, t2), _ints(a, t2 + 1, t3), bc,
                  _ints(a, t4 + 1, nl_k) if want_count else None, n)


# ---------------------------------------------------------------------------
# The tabix index
# ---------------------------------------------------------------------------


@dataclass
class TabixIndex:
    names: list[str]
    bins: list[dict[int, np.ndarray]]  # per reference: bin -> (n, 2) uint64 virtual offsets
    linear: list[np.ndarray]  # per reference: first offset of each 16 kb window
    meta: int  # the comment character, as an int
    skip: int
    zero_based: bool


def parse_tbi(raw: bytes) -> TabixIndex:
    if raw[:4] != b"TBI\x01":
        raise FragmentFileError("That .tbi file is not a tabix index: use the one Cell Ranger ARC wrote "
                                "beside the fragments file.")
    n_ref, fmt, _seq, _beg, _end, meta, skip, l_nm = struct.unpack_from("<8i", raw, 4)
    pos = 36
    names = [s.decode() for s in raw[pos:pos + l_nm].split(b"\0")[:n_ref]]
    pos += l_nm
    bins: list[dict[int, np.ndarray]] = []
    linear: list[np.ndarray] = []
    for _ in range(n_ref):
        n_bin = struct.unpack_from("<i", raw, pos)[0]
        pos += 4
        table: dict[int, np.ndarray] = {}
        for _ in range(n_bin):
            bin_id, n_chunk = struct.unpack_from("<Ii", raw, pos)
            pos += 8
            table[bin_id] = np.frombuffer(raw, dtype="<u8", count=2 * n_chunk, offset=pos).reshape(n_chunk, 2)
            pos += 16 * n_chunk
        bins.append(table)
        n_intv = struct.unpack_from("<i", raw, pos)[0]
        pos += 4
        linear.append(np.frombuffer(raw, dtype="<u8", count=n_intv, offset=pos))
        pos += 8 * n_intv
    return TabixIndex(names, bins, linear, meta, skip, bool(fmt & 0x10000))


def _reg2bins(beg: int, end: int) -> list[int]:
    """Bins that may hold records overlapping [beg, end) (tabix.pdf, min_shift 14, depth 5)."""
    end -= 1
    out = [0]
    shift = _MIN_SHIFT + 3 * _DEPTH
    t = 0
    for level in range(1, _DEPTH + 1):
        t += 1 << ((level - 1) * 3)
        shift -= 3
        out.extend(range(t + (beg >> shift), t + (end >> shift) + 1))
    return out


_index_cache: OrderedDict[tuple[str, float], TabixIndex] = OrderedDict()
_index_lock = threading.Lock()
_INDEX_CACHE = 4


def load_index(path: str | Path) -> TabixIndex:
    """The parsed index, cached per file."""
    key = (str(path), os.path.getmtime(path))
    with _index_lock:
        cached = _index_cache.get(key)
        if cached is not None:
            _index_cache.move_to_end(key)
            return cached
    try:
        raw = read_all_bgzf(path)
    except (FragmentFileError, zlib.error) as e:
        raise FragmentFileError("That .tbi file could not be read as a tabix index: use the one Cell Ranger ARC "
                                "wrote beside the fragments file.") from e
    try:
        index = parse_tbi(raw)
    except struct.error as e:
        raise FragmentFileError("That .tbi file is cut off: upload it again.") from e
    with _index_lock:
        _index_cache[key] = index
        while len(_index_cache) > _INDEX_CACHE:
            _index_cache.popitem(last=False)
    return index


def _chunks_for(index: TabixIndex, tid: int, beg: int, end: int) -> list[tuple[int, int]]:
    """Virtual offset ranges to read for [beg, end), merged."""
    linear = index.linear[tid]
    window = beg >> _MIN_SHIFT
    min_off = 0
    if linear.size:
        min_off = int(linear[min(window, linear.size - 1)]) if window < linear.size else int(linear[-1])
    found: list[tuple[int, int]] = []
    table = index.bins[tid]
    for b in _reg2bins(beg, end):
        chunks = table.get(b)
        if chunks is None:
            continue
        for s, e in chunks.tolist():
            if e > min_off:
                found.append((max(s, min_off), e))
    found.sort()
    merged: list[tuple[int, int]] = []
    for s, e in found:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


class FragmentReader:
    """Random access by region. ``barcodes``: the dataset's cells, in the order the barcode
    index refers to. Safe to share between threads: each call opens the file itself."""

    def __init__(self, path: str | Path, index_path: str | Path, barcodes: Sequence[str] | Barcodes) -> None:
        self.path = Path(path)
        self.index = load_index(index_path)
        self.barcodes = barcodes if isinstance(barcodes, Barcodes) else Barcodes(barcodes)
        self._tid = {name: i for i, name in enumerate(self.index.names)}

    @property
    def chromosomes(self) -> list[str]:
        return list(self.index.names)

    def has_chrom(self, chrom: str) -> bool:
        return chrom in self._tid

    def _read_virtual(self, fh, s: int, e: int) -> bytes:
        """The text between two virtual offsets."""
        coffset, uoffset = s >> 16, s & 0xFFFF
        end_c, end_u = e >> 16, e & 0xFFFF
        parts: list[bytes] = []
        fh.seek(coffset)
        buf = b""
        pos = 0
        at = coffset
        while at <= end_c:
            size = _block_size(buf, pos)
            if size < 0 or len(buf) - pos < size:
                chunk = fh.read(max(_MAX_BLOCK * 4, end_c - at + _MAX_BLOCK))
                if not chunk:
                    break
                buf = buf[pos:] + chunk
                pos = 0
                continue
            text = _inflate(memoryview(buf)[pos:pos + size])
            lo = uoffset if at == coffset else 0
            hi = end_u if at == end_c else len(text)
            parts.append(text[lo:hi])
            pos += size
            at += size
        return b"".join(parts)

    def region(self, chrom: str, start: int, end: int, cells_only: bool = True
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Fragments overlapping [start, end) on ``chrom``: arrays (start, end, barcode index,
        count) in file order. ``cells_only=False`` keeps other barcodes, with index -1."""
        empty = np.zeros(0, dtype=np.int64)
        tid = self._tid.get(chrom)
        if tid is None or end <= start:
            return empty, empty, empty, empty
        start = max(0, int(start))
        out: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        with open(self.path, "rb") as fh:
            for s, e in _chunks_for(self.index, tid, start, int(end)):
                text = self._read_virtual(fh, s, e)
                if not text:
                    continue
                if not text.endswith(b"\n"):
                    text = text[:text.rfind(b"\n") + 1]
                p = parse(text, self.barcodes, cells_only=cells_only, want_count=True)
                for name, a, b in p.chroms:
                    if name != chrom:
                        continue
                    keep = (p.start[a:b] < end) & (p.end[a:b] > start)
                    out.append((p.start[a:b][keep], p.end[a:b][keep], p.barcode[a:b][keep], p.count[a:b][keep]))
        if not out:
            return empty, empty, empty, empty
        return tuple(np.concatenate([o[i] for o in out]) for i in range(4))  # type: ignore[return-value]


def share_of_cells(path: str | Path, barcodes: Barcodes, max_lines: int = 1_000_000) -> dict[str, int]:
    """Of the first ``max_lines`` lines or so: how many, how many belong to a listed barcode,
    how many cells those are. Reads no further than it needs (about 50 MB of text)."""
    lines = 0
    hits = 0
    seen = np.zeros(len(barcodes), dtype=bool)
    tail = b""
    parts: list[bytes] = []
    size = 0
    piece = max(64 * 1024, min(TEXT_BLOCK, max_lines * 8))

    def count(text: bytes) -> None:
        nonlocal lines, hits
        p = parse(text, barcodes)
        lines += p.n_lines
        hits += p.barcode.size
        seen[p.barcode] = True

    for _, text in iter_bgzf(path):
        parts.append(text)
        size += len(text)
        if size >= piece:
            joined = tail + b"".join(parts)
            cut = joined.rfind(b"\n") + 1
            count(joined[:cut])
            tail = joined[cut:]
            parts, size = [], 0
            if lines >= max_lines:
                break
    else:
        joined = tail + b"".join(parts)
        if joined:
            count(joined if joined.endswith(b"\n") else joined + b"\n")
    return {"lines": int(lines), "matching": int(hits), "cells_seen": int(seen.sum())}
