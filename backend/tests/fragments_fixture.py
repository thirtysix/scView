"""A fragments file for tests: real BGZF blocks and a real tabix index, written here.

``gzip`` does not write BGZF, so each block is built by hand (the 'BC' extra field with the
block's size, raw deflate, CRC + length, the EOF block). The index follows htslib's
``tabix.pdf``: a chunk per run of records in one bin, a linear index of the first offset
touching each 16 kb window.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

EOF_BLOCK = bytes.fromhex("1f8b08040000000000ff0600424302001b0003000000000000000000")


def bgzf_block(data: bytes) -> bytes:
    co = zlib.compressobj(6, zlib.DEFLATED, -15)
    body = co.compress(data) + co.flush()
    size = len(body) + 25 + 1  # header 18 + footer 8; BSIZE is total - 1
    head = b"\x1f\x8b\x08\x04\0\0\0\0\0\xff\x06\0BC\x02\0" + struct.pack("<H", size - 1)
    return head + body + struct.pack("<II", zlib.crc32(data), len(data))


def _bin_of(beg: int, end: int) -> int:
    end -= 1
    for shift, offset in ((14, 4681), (17, 585), (20, 73), (23, 9), (26, 1)):
        if beg >> shift == end >> shift:
            return offset + (beg >> shift)
    return 0


def write_fragments(path: str | Path, lines: list[tuple[str, int, int, str, int]], header: str = "# id=test\n",
                    block_size: int = 2000) -> tuple[Path, Path]:
    """Write ``path`` + ``path.tbi`` from sorted lines (chrom, start, end, barcode, count)."""
    path = Path(path)
    blocks = bytearray()
    pending = bytearray(header.encode())
    voffsets: list[tuple[int, int]] = []

    def flush() -> None:
        nonlocal pending, blocks
        if pending:
            blocks += bgzf_block(bytes(pending))
            pending = bytearray()

    for chrom, start, end, barcode, count in lines:
        if len(pending) >= block_size:
            flush()
        begin = (len(blocks) << 16) | len(pending)
        pending += f"{chrom}\t{start}\t{end}\t{barcode}\t{count}\n".encode()
        voffsets.append((begin, (len(blocks) << 16) | len(pending)))
    flush()
    blocks += EOF_BLOCK
    path.write_bytes(bytes(blocks))

    names: list[str] = []
    tables: dict[str, dict[int, list[list[int]]]] = {}
    linear: dict[str, dict[int, int]] = {}
    for (chrom, start, end, _, _), (beg, stop) in zip(lines, voffsets):
        if chrom not in tables:
            names.append(chrom)
            tables[chrom], linear[chrom] = {}, {}
        chunks = tables[chrom].setdefault(_bin_of(start, end), [])
        # Offsets move to the next block at a flush: a stop at a block's end meets the
        # next record's begin only in the same run when they are the same block.
        if chunks and chunks[-1][1] == beg:
            chunks[-1][1] = stop
        else:
            chunks.append([beg, stop])
        for w in range(start >> 14, ((end - 1) >> 14) + 1):
            linear[chrom].setdefault(w, beg)

    nm = b"".join(n.encode() + b"\0" for n in names)
    out = bytearray(b"TBI\x01")
    out += struct.pack("<8i", len(names), 0x10000 | 2, 1, 2, 3, 35, 0, len(nm)) + nm
    for name in names:
        out += struct.pack("<i", len(tables[name]))
        for b, chunks in sorted(tables[name].items()):
            out += struct.pack("<Ii", b, len(chunks))
            for s, e in chunks:
                out += struct.pack("<QQ", s, e)
        windows = linear[name]
        n_intv = max(windows) + 1 if windows else 0
        out += struct.pack("<i", n_intv)
        last = 0
        for w in range(n_intv):
            last = windows.get(w, last)
            out += struct.pack("<Q", last)
    index = path.with_name(path.name + ".tbi")
    index.write_bytes(bgzf_block(bytes(out)) + EOF_BLOCK)
    return path, index
