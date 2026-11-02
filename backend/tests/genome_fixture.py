"""A small genome for unit tests: writes a real .2bit file and pins it, so ``genome_seq.ensure``
runs its normal path with no network."""

from __future__ import annotations

import hashlib
import struct
from pathlib import Path

from scview.core import genome_seq

_TO_2BIT = {"T": 0, "C": 1, "A": 2, "G": 3}


def write_2bit(path: str | Path, seqs: dict[str, str]) -> Path:
    """``seqs``: name -> bases, where N is an unknown run and lower case is repeat masked."""
    path = Path(path)
    index_size = sum(1 + len(n) + 4 for n in seqs)
    offset = 16 + index_size
    records = []
    index = b""
    for name, seq in seqs.items():
        upper = seq.upper()
        n_runs, mask_runs = _runs(upper, "N"), _runs_mask(seq)
        rec = struct.pack("<II", len(seq), len(n_runs))
        rec += b"".join(struct.pack("<I", s) for s, _ in n_runs)
        rec += b"".join(struct.pack("<I", e - s) for s, e in n_runs)
        rec += struct.pack("<I", len(mask_runs))
        rec += b"".join(struct.pack("<I", s) for s, _ in mask_runs)
        rec += b"".join(struct.pack("<I", e - s) for s, e in mask_runs)
        rec += struct.pack("<I", 0)
        packed = bytearray((len(seq) + 3) // 4)
        for i, base in enumerate(upper):
            packed[i // 4] |= _TO_2BIT.get(base, 0) << (6 - 2 * (i % 4))
        rec += bytes(packed)
        index += struct.pack("<B", len(name)) + name.encode("ascii") + struct.pack("<I", offset)
        records.append(rec)
        offset += len(rec)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<IIII", 0x1A412743, 0, len(seqs), 0) + index + b"".join(records))
    return path


def _runs(seq: str, base: str) -> list[tuple[int, int]]:
    runs, start = [], None
    for i, b in enumerate(seq):
        if b == base and start is None:
            start = i
        elif b != base and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(seq)))
    return runs


def _runs_mask(seq: str) -> list[tuple[int, int]]:
    runs, start = [], None
    for i, b in enumerate(seq):
        if b.islower() and start is None:
            start = i
        elif not b.islower() and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(seq)))
    return runs


def install(monkeypatch, data_dir: str | Path, genome: str, seqs: dict[str, str]) -> Path:
    """Put ``seqs`` where ``ensure`` looks for ``genome`` and pin their sha256 for this test."""
    release = genome_seq.RELEASES[genome]
    path = write_2bit(Path(data_dir) / genome_seq.SUBDIR / release.filename, seqs)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setitem(genome_seq.RELEASES, genome, genome_seq.Release(
        release.genome, release.source, release.url, digest, path.stat().st_size, release.filename))
    genome_seq.clear_cache()
    return path
