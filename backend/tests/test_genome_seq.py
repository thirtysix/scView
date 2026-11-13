import numpy as np
import pytest

from tests.genome_fixture import install, write_2bit
from scview.core import genome_seq
from scview.core.genome_seq import TwoBit


def _text(codes: np.ndarray) -> str:
    return "".join("ACGTN"[c] for c in codes)


SEQ = "ACGTACGTTTGGCCAANNNNNNNNacgtGATTACA"


@pytest.fixture
def two_bit(tmp_path):
    g = TwoBit(write_2bit(tmp_path / "t.2bit", {"chr1": SEQ, "chr2": "GGGCCC"}))
    yield g
    g.close()


def test_reads_every_slice_including_unknown_runs_and_masked_bases(two_bit):
    assert two_bit.names == ["chr1", "chr2"]
    assert two_bit.size("chr1") == len(SEQ)
    upper = SEQ.upper()
    for start in range(len(SEQ)):
        for end in range(start, len(SEQ) + 1):
            assert _text(two_bit.fetch("chr1", start, end)) == upper[start:end]


def test_outside_the_contig_or_an_absent_contig_is_unknown(two_bit):
    assert _text(two_bit.fetch("chr2", 4, 9)) == "CCNNN"
    assert _text(two_bit.fetch("chr2", -2, 2)) == "NNGG"
    assert _text(two_bit.fetch("KI270728.1", 0, 3)) == "NNN"


def test_regions_end_to_end_with_gc_and_background(two_bit):
    codes, offsets = genome_seq.sequences(two_bit, ["chr2", "chr1", "nope"], [0, 16, 0], [6, 24, 4])
    assert offsets.tolist() == [0, 6, 14, 18]
    counts = genome_seq.base_counts(codes, offsets)
    assert counts[0].tolist() == [0, 3, 3, 0, 0]
    assert counts[1].tolist() == [0, 0, 0, 0, 8]
    gc = genome_seq.gc_percent(counts)
    assert gc[0] == 100.0
    assert np.isnan(gc[1]) and np.isnan(gc[2])
    assert genome_seq.background(counts).tolist() == [0.0, 0.5, 0.5, 0.0]


def test_gc_counts_unknown_bases_in_the_length_as_signac_does(tmp_path):
    g = TwoBit(write_2bit(tmp_path / "t.2bit", {"chr1": "GGNN"}))
    codes, offsets = genome_seq.sequences(g, ["chr1"], [0], [4])
    assert genome_seq.gc_percent(genome_seq.base_counts(codes, offsets))[0] == 50.0
    g.close()


def test_big_endian_files_read_the_same(tmp_path):
    import struct

    p = write_2bit(tmp_path / "le.2bit", {"chr1": "ACGT"})
    raw = bytearray(p.read_bytes())
    sig, ver, n, res = struct.unpack("<IIII", raw[:16])
    be = struct.pack(">IIII", sig, ver, n, res)
    pos = 16
    name_len = raw[pos]
    off = struct.unpack("<I", raw[pos + 1 + name_len:pos + 5 + name_len])[0]
    be += bytes(raw[pos:pos + 1 + name_len]) + struct.pack(">I", off)
    size, nb, nm, rs = struct.unpack("<IIII", raw[off:off + 16])
    be += struct.pack(">IIII", size, nb, nm, rs) + bytes(raw[off + 16:])
    q = tmp_path / "be.2bit"
    q.write_bytes(be)
    g = TwoBit(q)
    assert _text(g.fetch("chr1", 0, 4)) == "ACGT"
    g.close()


def test_rejects_a_file_that_is_not_2bit(tmp_path):
    p = tmp_path / "x.2bit"
    p.write_bytes(b"\0" * 32)
    with pytest.raises(ValueError, match="not a 2bit"):
        TwoBit(p)


def test_ensure_uses_a_pinned_file_and_never_fetches(tmp_path, monkeypatch):
    install(monkeypatch, tmp_path, "GRCh38", {"chr1": "ACGT"})
    g = genome_seq.ensure(tmp_path, "GRCh38-2020-A", auto_download=False)
    assert _text(g.fetch("chr1", 0, 4)) == "ACGT"
    assert genome_seq.ensure(tmp_path, "hg38", auto_download=False) is g
    genome_seq.clear_cache()


def test_ensure_refuses_a_file_that_is_not_the_pinned_one(tmp_path, monkeypatch):
    path = install(monkeypatch, tmp_path, "GRCh38", {"chr1": "ACGT"})
    path.write_bytes(path.read_bytes() + b"x")
    with pytest.raises(genome_seq.GenomeUnavailable, match="sha256 differs"):
        genome_seq.ensure(tmp_path, "GRCh38")
    genome_seq.clear_cache()


def test_ensure_with_fetching_off_says_what_to_do(tmp_path):
    with pytest.raises(genome_seq.GenomeUnavailable, match="GENOME_AUTO_DOWNLOAD"):
        genome_seq.ensure(tmp_path, "mm10", auto_download=False)
    with pytest.raises(genome_seq.GenomeUnavailable, match=r"UCSC mm39 .*GENOME_AUTO_DOWNLOAD.*Place mm39\.2bit"):
        genome_seq.ensure(tmp_path, "GRCm39-2024-A", auto_download=False)


def test_mm39_uses_its_own_pinned_file(tmp_path, monkeypatch):
    install(monkeypatch, tmp_path, "mm39", {"chr8": "GGCCAATT"})
    g = genome_seq.ensure(tmp_path, "GRCm39-2024-A", auto_download=False)
    assert g.path.name == "mm39.2bit" and _text(g.fetch("chr8", 0, 4)) == "GGCC"
    genome_seq.clear_cache()


def test_mm39_download_checks_free_space_first(tmp_path, monkeypatch):
    import collections
    import shutil

    monkeypatch.setattr(shutil, "disk_usage", lambda _p: collections.namedtuple("u", "total used free")(1, 1, 10**8))
    with pytest.raises(genome_seq.GenomeUnavailable, match=r"UCSC mm39 sequence needs 0\.7 GB .* 0\.1 GB free"):
        genome_seq.ensure(tmp_path, "mm39", auto_download=True)
    assert not (tmp_path / genome_seq.SUBDIR / "mm39.2bit").exists()


def test_status_never_needs_the_file(tmp_path, monkeypatch):
    s = genome_seq.status(tmp_path, "mm10")
    assert s["downloaded"] is False and s["size"] > 0 and s["url"].endswith("mm10.2bit")
    install(monkeypatch, tmp_path, "mm10", {"chr1": "A"})
    assert genome_seq.status(tmp_path, "mm10")["downloaded"] is True
    genome_seq.clear_cache()
