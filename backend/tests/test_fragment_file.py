"""BGZF + tabix reader (core/fragment_file.py): a region = a full scan of the lines."""

from __future__ import annotations

import gzip

import numpy as np
import pytest

from scview.core import fragment_file as ff
from tests.fragments_fixture import EOF_BLOCK, bgzf_block, write_fragments

CELLS = ["AAAC-1", "AAAG-1", "AAAT-1", "LONGERBARCODE-1"]
OTHER = ["CCCC-1", "GGGG-1", "SHORT"]


def _lines(n: int = 6000, seed: int = 0) -> list[tuple[str, int, int, str, int]]:
    rng = np.random.default_rng(seed)
    out = []
    for chrom, span in (("chr1", 2_000_000), ("chr10", 400_000), ("chrX", 90_000)):
        starts = np.sort(rng.integers(0, span, n // 3))
        lengths = rng.integers(30, 800, starts.size)
        lengths[::97] = 70_000  # some fragments cross windows and land in a higher bin
        names = rng.choice(CELLS + OTHER, starts.size)
        counts = rng.integers(1, 9, starts.size)
        out += [(chrom, int(s), int(s + w), str(b), int(c)) for s, w, b, c in zip(starts, lengths, names, counts)]
    return out


def _scan(lines, chrom, lo, hi):
    index = {b: i for i, b in enumerate(CELLS)}
    keep = [x for x in lines if x[0] == chrom and x[1] < hi and x[2] > lo]
    return ([x[1] for x in keep], [x[2] for x in keep], [index.get(x[3], -1) for x in keep], [x[4] for x in keep])


@pytest.fixture(scope="module")
def pair(tmp_path_factory):
    lines = _lines()
    path, index = write_fragments(tmp_path_factory.mktemp("frag") / "f.tsv.gz", lines)
    return lines, path, index


def test_is_bgzf_tells_it_from_plain_gzip(pair):
    _, path, _ = pair
    assert ff.is_bgzf(path.read_bytes()[:1024])
    assert not ff.is_bgzf(gzip.compress(b"chr1\t1\t2\tA\t1\n"))
    assert not ff.is_bgzf(b"chr1\t1\t2\tA\t1\n" * 10)
    assert ff.is_bgzf(EOF_BLOCK)


def test_reading_every_block_gives_the_text(pair):
    lines, path, _ = pair
    text = ff.read_all_bgzf(path).decode()
    body = [ln for ln in text.splitlines() if not ln.startswith("#")]
    assert len(body) == len(lines) and body[0] == "\t".join(map(str, lines[0]))
    pieces = list(ff.iter_text(path, block=5000))
    assert all(t.endswith(b"\n") for _, t in pieces) and pieces[-1][0] == path.stat().st_size
    assert b"".join(t for _, t in pieces).decode() == text


def test_a_cut_off_file_says_so(pair, tmp_path):
    _, path, _ = pair
    cut = tmp_path / "cut.gz"
    cut.write_bytes(path.read_bytes()[:-1000])
    with pytest.raises(ff.FragmentFileError, match="cut off"):
        ff.read_all_bgzf(cut)


def test_index_parses(pair):
    _, _, index = pair
    tbi = ff.load_index(index)
    assert tbi.names == ["chr1", "chr10", "chrX"] and tbi.meta == 35 and tbi.zero_based
    assert all(len(b) for b in tbi.bins)
    assert ff.load_index(index) is tbi  # cached


def test_a_non_index_is_a_sentence(tmp_path):
    bad = tmp_path / "x.tbi"
    bad.write_bytes(bgzf_block(b"NOPE" + b"\0" * 40) + EOF_BLOCK)
    with pytest.raises(ff.FragmentFileError, match="not a tabix index"):
        ff.load_index(bad)
    plain = tmp_path / "y.tbi"
    plain.write_bytes(b"this is not compressed at all, not even a little bit")
    with pytest.raises(ff.FragmentFileError, match="could not be read"):
        ff.load_index(plain)


def test_region_equals_a_full_scan(pair):
    lines, path, index = pair
    reader = ff.FragmentReader(path, index, CELLS)
    rng = np.random.default_rng(1)
    spans = {"chr1": 2_100_000, "chr10": 500_000, "chrX": 170_000}
    regions = [("chr1", 0, 10), ("chr1", 2_000_000, 2_200_000), ("chr1", 0, 2_100_000), ("chrX", 0, 170_000)]
    for chrom, span in spans.items():
        regions += [(chrom, int(a), int(a + w)) for a, w in zip(rng.integers(0, span, 16), rng.integers(1, 90_000, 16))]
    for chrom, lo, hi in regions:
        got = reader.region(chrom, lo, hi, cells_only=False)
        want = _scan(lines, chrom, lo, hi)
        for g, w in zip(got, want):
            assert g.tolist() == w, (chrom, lo, hi)


def test_cells_only_drops_other_barcodes(pair):
    lines, path, index = pair
    reader = ff.FragmentReader(path, index, CELLS)
    s, e, b, c = reader.region("chr10", 1000, 300_000)
    want = [x for x in zip(*_scan(lines, "chr10", 1000, 300_000)) if x[2] >= 0]
    assert s.tolist() == [x[0] for x in want] and b.tolist() == [x[2] for x in want]
    assert c.tolist() == [x[3] for x in want] and e.tolist() == [x[1] for x in want]
    assert b.size and b.min() >= 0


def test_a_missing_chromosome_or_empty_window_gives_nothing(pair):
    _, path, index = pair
    reader = ff.FragmentReader(path, index, CELLS)
    assert reader.region("chrZ", 0, 100)[0].size == 0
    assert reader.region("chr1", 500, 500)[0].size == 0
    assert reader.region("chrX", 50_000_000, 50_001_000)[0].size == 0
    assert not reader.has_chrom("chrZ") and reader.has_chrom("chr1")


def test_parse_spots_barcodes_of_any_width_and_chromosome_edges():
    text = (b"#comment\nchr1\t10\t20\tAAAC-1\t1\nchr1\t11\t21\tSHORT\t2\n"
            b"chr2\t5\t100\tLONGERBARCODE-1\t3\nchr2\t6\t60\tAAAC-1\t4\nchrX\t1\t2\tNOPE\t5\n")
    p = ff.parse(text, ff.Barcodes(CELLS), cells_only=False, want_count=True)
    assert p.n_lines == 5 and p.chroms == [("chr1", 0, 2), ("chr2", 2, 4), ("chrX", 4, 5)]
    assert p.barcode.tolist() == [0, -1, 3, 0, -1] and p.count.tolist() == [1, 2, 3, 4, 5]
    assert p.start.tolist() == [10, 11, 5, 6, 1] and p.end.tolist() == [20, 21, 100, 60, 2]
    kept = ff.parse(text, ff.Barcodes(CELLS))
    assert kept.n_lines == 5 and kept.chroms == [("chr1", 0, 1), ("chr2", 1, 3)]
    assert kept.barcode.tolist() == [0, 3, 0] and kept.start.tolist() == [10, 5, 6] and kept.count is None


def test_parse_takes_extra_columns_and_comments_in_the_middle():
    extra = b"chr1\t10\t20\tAAAC-1\t1\t+\nchr1\t11\t21\tAAAG-1\t2\t-\n"
    p = ff.parse(extra, ff.Barcodes(CELLS), want_count=True)
    assert p.barcode.tolist() == [0, 1] and p.count.tolist() == [1, 2] and p.chroms == [("chr1", 0, 2)]
    middle = b"chr1\t10\t20\tAAAC-1\t1\n#note\nchr2\t11\t21\tAAAG-1\t2\n"
    p = ff.parse(middle, ff.Barcodes(CELLS))
    assert p.chroms == [("chr1", 0, 1), ("chr2", 1, 2)] and p.end.tolist() == [20, 21]


def test_a_position_that_is_no_number_is_a_sentence():
    with pytest.raises(ff.FragmentFileError, match="not a number"):
        ff.parse(b"chr1\t1x\t20\tAAAC-1\t1\n", ff.Barcodes(CELLS))


def test_share_of_cells_reads_only_the_start(pair):
    lines, path, _ = pair
    all_lines = ff.share_of_cells(path, ff.Barcodes(CELLS), max_lines=10**7)
    assert all_lines["lines"] == len(lines)
    assert all_lines["matching"] == sum(x[3] in CELLS for x in lines) and all_lines["cells_seen"] == 4
    few = ff.share_of_cells(path, ff.Barcodes(CELLS), max_lines=500)
    assert 500 <= few["lines"] < len(lines)


def test_barcodes_lookup_is_exact():
    codes = ff.Barcodes(CELLS)
    assert codes.lookup(np.array([codes.hash_of("AAAG-1"), codes.hash_of("ZZZZ-1")], dtype=np.uint64)).tolist() == [1, -1]
    assert ff.Barcodes([]).lookup(np.array([codes.hash_of("A")], dtype=np.uint64)).tolist() == [-1]
