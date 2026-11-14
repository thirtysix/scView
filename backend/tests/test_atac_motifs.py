"""Motifs in peaks (core/atac_motifs.py): matrices and thresholds as motifmatchr, the scan, FindMotifs."""

from __future__ import annotations

import json
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import pandas as pd
import pytest
from scipy.stats import hypergeom

from scview.core import atac_motifs as am
from scview.core import genome_seq, tf_motifs
from tests import genome_fixture

# Real matrices from the pinned JASPAR file, and motifmatchr 1.34.0's get_thresholds for them at the
# base frequencies of the PBMC 10k peaks (rtracklayer on hg38.2bit, 143,836 peaks).
PBMC_BG = np.array([0.26035128752866094, 0.23947415101431174, 0.23979829318562115, 0.26037626827140614])
ARNT = np.array([[4, 19, 0, 0, 0, 0], [16, 0, 20, 0, 0, 0], [0, 1, 0, 20, 0, 20], [0, 0, 0, 0, 20, 0]], dtype=float)  # MA0004.1


def _random_codes(rng, n):
    return rng.integers(0, 4, n).astype(np.uint8)


def _motif_like(rng, length, strength=3.0):
    """A PFM with a clear preferred base in each column."""
    counts = rng.integers(1, 10, (4, length)).astype(float)
    counts[rng.integers(0, 4, length), np.arange(length)] += 10 * strength
    return counts


def test_pwm_is_motifmatchr_conversion_with_its_sign():
    w = am.pwm(ARNT, PBMC_BG)
    p = (ARNT + 0.8 * 0.25) / (ARNT.sum(axis=0) + 0.8)
    expected = np.log2(p / 0.25) - (np.log2(0.25) - np.log2(PBMC_BG))[:, None]
    assert np.allclose(w, expected)
    # the sign: a base with a higher background frequency scores HIGHER than at an even background
    assert (am.pwm(ARNT, PBMC_BG) - am.pwm(ARNT, np.full(4, 0.25)))[0, 0] > 0


def test_threshold_of_a_short_motif_sits_just_under_its_best_word():
    """A 6-mer's best word alone is more likely than 5e-5: MOODS keeps that word (max - min_delta / 2)."""
    w = am.pwm(ARNT, PBMC_BG)
    t = am.motif_threshold(w, PBMC_BG)
    best = w.max(axis=0).sum()
    assert best > t > best - 1
    assert am._best_word_only(w, PBMC_BG)


def test_threshold_is_the_lowest_rounded_score_with_tail_at_most_p():
    rng = np.random.default_rng(3)
    counts = _motif_like(rng, 12)
    bg = np.array([0.3, 0.2, 0.2, 0.3])
    w = am.pwm(counts, bg)
    p = 5e-5
    t = am.motif_threshold(w, bg, p)
    rounded = np.where(w > 0, np.trunc(2000 * w + 0.5), np.trunc(2000 * w - 0.5)).astype(np.int64)
    # the same tail by brute force over words of the rounded matrix is impossible at 4^12; check by sampling
    # the exact distribution: every score at or above t has tail <= p, one below has tail > p
    lo, dist = tf_motifs.score_distribution(rounded, bg)
    sf = np.cumsum(dist[::-1])[::-1]
    i = int(round(t * 2000)) - lo
    assert sf[i] <= p < sf[i - 1]


def test_threshold_uses_the_forward_strand_for_both():
    rng = np.random.default_rng(5)
    counts = _motif_like(rng, 10)
    strands, thresholds = am.prepare_all([counts], PBMC_BG)
    assert len(strands) == 2
    assert strands[0].t == strands[1].t == thresholds[0]
    assert np.array_equal(strands[1].w, strands[0].w[::-1, ::-1])


def _scan_inputs(rng, n_regions=120, with_n=True):
    regions = []
    for _ in range(n_regions):
        codes = _random_codes(rng, int(rng.integers(3, 400)))
        if with_n and rng.random() < 0.1 and codes.size > 20:
            at = int(rng.integers(0, codes.size - 5))
            codes[at:at + int(rng.integers(1, 5))] = 4
        regions.append(codes)
    offsets = np.concatenate([[0], np.cumsum([r.size for r in regions])]).astype(np.int64)
    return am.join_regions(np.concatenate(regions), offsets), regions


def test_join_regions_puts_an_unknown_base_between_regions():
    joined, starts = am.join_regions(np.array([0, 1, 2, 3, 0], dtype=np.uint8), np.array([0, 2, 5]))
    assert joined.tolist() == [0, 1, 4, 2, 3, 0, 4]
    assert starts.tolist() == [0, 3]


@pytest.mark.parametrize("lengths", [(5, 6, 7, 8), (9, 12, 15, 21), (6, 33)])
def test_indexed_scan_gives_the_bits_of_the_plain_scan(lengths):
    rng = np.random.default_rng(sum(lengths))
    counts = [_motif_like(rng, n, strength=float(rng.uniform(0.3, 3))) for n in lengths]
    counts.append(np.full((4, 8), 5.0))  # a flat matrix: every word its best
    (joined, starts), _ = _scan_inputs(rng)
    bg = np.array([0.26, 0.24, 0.24, 0.26])
    strands, _t = am.prepare_all(counts, bg)
    plain = am.scan_plain(joined, starts, counts, bg)
    assert plain.any()
    assert np.array_equal(am.scan_block(joined, starts, strands, len(counts)), plain)
    assert np.array_equal(am.scan_block(joined, starts, strands, len(counts), threads=2), plain)


def test_hits_only_inside_one_stretch_of_known_bases():
    counts = np.zeros((4, 8))
    counts[np.array([0, 1, 2, 3, 0, 1, 2, 3]), np.arange(8)] = 100  # ACGTACGT
    bg = np.full(4, 0.25)
    site = np.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.uint8)
    rc = (3 - site)[::-1]
    regions = {
        "whole": site,
        "reverse": rc,
        "split": np.concatenate([site[:4], [4], site[4:]]).astype(np.uint8),
        "across": site[:4],  # a window may not run into the next region, which holds the rest
        "next": site[4:],
    }
    codes = np.concatenate(list(regions.values())).astype(np.uint8)
    offsets = np.concatenate([[0], np.cumsum([r.size for r in regions.values()])]).astype(np.int64)
    joined, starts = am.join_regions(codes, offsets)
    strands, _ = am.prepare_all([counts], bg)
    hits = am.scan_block(joined, starts, strands, 1)[:, 0]
    assert hits.tolist() == [True, True, False, False, False]


def test_a_motif_shorter_than_the_window_misses_the_last_position_as_moods_does():
    counts = np.zeros((4, 5))
    counts[np.array([0, 1, 2, 3, 0]), np.arange(5)] = 100  # ACGTA: only that word, its best
    bg = np.full(4, 0.25)
    assert am._best_word_only(am.pwm(counts, bg), bg)
    word = np.array([0, 1, 2, 3, 0], dtype=np.uint8)
    # C pads away from the reverse complement's word TACGT
    regions = [np.concatenate([word, [1]]).astype(np.uint8), np.concatenate([[1], word]).astype(np.uint8)]
    joined, starts = am.join_regions(np.concatenate(regions), np.array([0, 6, 12]))
    strands, _ = am.prepare_all([counts], bg)
    assert am.scan_block(joined, starts, strands, 1)[:, 0].tolist() == [True, False]
    assert am.scan_plain(joined, starts, [counts], bg)[:, 0].tolist() == [True, False]


def _small_genome(monkeypatch, tmp_path, rng):
    seqs = {"chr1": "".join("ACGT"[i] for i in rng.integers(0, 4, 40_000))}
    seqs["chr1"] = seqs["chr1"][:5000] + "N" * 200 + seqs["chr1"][5200:]
    genome_fixture.install(monkeypatch, tmp_path, "GRCh38", seqs)
    return genome_seq.TwoBit(genome_seq.file_path(tmp_path, "GRCh38")), seqs


def test_build_leaves_out_peaks_on_unnamed_contigs_and_round_trips(monkeypatch, tmp_path):
    rng = np.random.default_rng(1)
    genome, seqs = _small_genome(monkeypatch, tmp_path, rng)
    chrom = np.array(["chr1", "chr1", "chr1", "KI270728.1", "chr1"], dtype=object)
    start = np.array([100, 4900, 7000, 50, -1])
    end = np.array([900, 5300, 7900, 800, -1])
    motifs = [tf_motifs.Motif("M1", "AAA::BBB", _motif_like(rng, 8)), tf_motifs.Motif("M2", "CCC", _motif_like(rng, 6))]
    events: list[dict] = []
    table = am.build(chrom, start, end, motifs, genome, "GRCh38", "k", progress=events.append, block_bases=500)
    assert table.known.tolist() == [True, True, True, False, False]
    assert table.meta["n_left_out"] == 2 and table.meta["n_unknown_bases"] == 200
    assert not table.hits(np.array([3, 4])).any()
    assert {e["stage"] for e in events} >= {"read", "scan"}
    # the base frequencies are those of the scanned peaks, as they read
    text = seqs["chr1"][100:900] + seqs["chr1"][4900:5300] + seqs["chr1"][7000:7900]
    counts = np.array([text.count(b) for b in "ACGT"], dtype=float)
    assert np.allclose(table.background, counts / counts.sum())
    path = tmp_path / "t.npz"
    am.save_table(table, path)
    back = am.read_table_file(path)
    assert np.array_equal(back.packed, table.packed) and back.meta == table.meta and back.genome == "GRCh38"
    assert np.array_equal(back.gc[:3], table.gc[:3])
    assert table.counts(np.arange(5)).tolist() == table.hits(np.arange(5)).sum(axis=0).tolist()


def test_build_can_be_stopped(monkeypatch, tmp_path):
    rng = np.random.default_rng(2)
    genome, _ = _small_genome(monkeypatch, tmp_path, rng)
    with pytest.raises(am.ScanCancelled):
        am.build(np.array(["chr1"], dtype=object), np.array([0]), np.array([100]), [tf_motifs.Motif("M", "M", _motif_like(rng, 8))],
                 genome, "GRCh38", "k", stop=lambda: True)


def test_cache_key_follows_the_peak_set():
    a = am.peak_digest(["chr1", "chr2"], np.array([1, 2]), np.array([3, 4]))
    assert a == am.peak_digest(["chr1", "chr2"], np.array([1, 2]), np.array([3, 4]))
    assert a != am.peak_digest(["chr1", "chr2"], np.array([1, 2]), np.array([3, 5]))
    assert am.cache_key("GRCh38", a) != am.cache_key("mm10", a)
    assert am.table_path("/d", "mm10", "x") == Path("/d/atac_motifs/mm10/hits_x.npz")


def test_estimates_are_within_20_percent_of_what_was_measured():
    # Through the server at 2 cores: PBMC 10k 111 s and 1.01 GB of anon; mouse E18 124 s.
    assert abs(am.estimate_scan_seconds(121_861_766) / 111 - 1) < 0.2
    assert abs(am.estimate_scan_seconds(146_162_749) / 124 - 1) < 0.2
    assert abs(am.estimate_scan_bytes(143_887) / 1.01e9 - 1) < 0.2


def _child_script(tmp_path: Path, body: str) -> list[str]:
    script = tmp_path / "child.py"
    script.write_text(body)
    return ["python", str(script)]


def test_iter_child_reports_progress_and_a_failure_in_one_sentence(tmp_path):
    ok = _child_script(tmp_path, 'print(\'{"type": "progress", "message": "a"}\', flush=True)\nprint("noise")\nprint(\'{"type": "done", "seconds": 1}\')\n')
    assert [e["type"] for e in am.iter_child(ok, None, 1, "x")] == ["progress", "done"]
    bad = _child_script(tmp_path, 'import sys\nprint("boom", file=sys.stderr)\nsys.exit(3)\n')
    with pytest.raises(am.ScanError, match="x failed .exit 3.: boom"):
        list(am.iter_child(bad, None, 1, "x"))


def test_iter_child_stop_kills_it(tmp_path):
    slow = _child_script(tmp_path, 'import time\nprint(\'{"type": "progress"}\', flush=True)\ntime.sleep(60)\n')
    stopped = {"now": False}
    gen = am.iter_child(slow, lambda: stopped["now"], 1, "x")
    assert next(gen)["type"] == "progress"
    stopped["now"] = True
    with pytest.raises(am.ScanCancelled):
        list(gen)


def test_child_process_scans_and_saves(monkeypatch, tmp_path):
    from scview.core import modality_store
    import scipy.sparse as sp

    rng = np.random.default_rng(4)
    _genome, _seqs = _small_genome(monkeypatch, tmp_path, rng)
    store = tmp_path / "atac.h5"
    names = ["chr1:100-900", "chr1:2000-2600"]
    modality_store.write(store, {"counts": sp.csr_matrix(np.ones((2, 2)))}, ["a", "b"], names,
                         {"chrom": ["chr1", "chr1"], "start": ["100", "2000"], "end": ["900", "2600"]}, {"genome": "GRCh38"})
    jaspar = tmp_path / "j.txt"
    jaspar.write_text(">M1 NAME1\nA [ 9 0 0 0 9 0 0 0 ]\nC [ 0 9 0 0 0 9 0 0 ]\nG [ 0 0 9 0 0 0 9 0 ]\nT [ 0 0 0 9 0 0 0 9 ]\n")
    out = tmp_path / "out" / "hits.npz"
    final = am.run_child(am.scan_cmd(store, jaspar, genome_seq.file_path(tmp_path, "GRCh38"), out, "GRCh38", "kk"), None, None, 1, "scan")
    assert final["n_peaks"] == 2 and final["peak_mb"] > 0
    table = am.read_table_file(out)
    assert table.key == "kk" and table.motif_names.tolist() == ["NAME1"]


# ---------------------------------------------------------------------------
# FindMotifs
# ---------------------------------------------------------------------------


def _table(n_peaks=400, n_motifs=3, seed=0) -> am.PeakMotifTable:
    rng = np.random.default_rng(seed)
    hits = rng.random((n_peaks, n_motifs)) < 0.2
    hits[:40, 0] = True  # motif 0 in the first 40 peaks
    return am.PeakMotifTable(
        genome="GRCh38", key="k", motif_ids=np.array([f"MA{i}" for i in range(n_motifs)]),
        motif_names=np.array(["SPI1", "MAX::MYC", "NOPE"][:n_motifs]), lengths=np.full(n_motifs, 8), thresholds=np.zeros(n_motifs),
        gc_motif=np.array([0.5, 0.8, 0.4])[:n_motifs], packed=np.packbits(hits, axis=1), gc=rng.normal(45, 8, n_peaks), background=np.full(4, 0.25))


def test_r_density_matches_a_plain_gaussian_kde():
    x = np.random.default_rng(0).normal(40, 5, 2000)
    grid, y = am._r_density(x)
    exact = np.exp(-0.5 * (grid[:, None] - x[None, :]) ** 2).sum(axis=1) / (x.size * np.sqrt(2 * np.pi))
    assert grid[0] == pytest.approx(x.min() - 3) and grid[-1] == pytest.approx(x.max() + 3) and grid.size == 512
    assert np.abs(y - exact).max() < 2e-3


def test_match_background_leaves_out_the_query_and_follows_its_gc():
    rng = np.random.default_rng(0)
    gc = np.concatenate([rng.normal(60, 2, 200), rng.normal(40, 5, 8000)])
    query = np.arange(100)
    chosen = am.match_background(gc, query, 60, np.random.default_rng(1))
    assert chosen.size == 60 and np.intersect1d(chosen, query).size == 0
    assert np.unique(chosen).size == 60
    assert abs(gc[chosen].mean() - gc[query].mean()) < 3  # drawn up to the query's 60 from a pool mostly at 40
    assert np.array_equal(chosen, am.match_background(gc, query, 60, np.random.default_rng(1)))


def test_match_background_takes_all_when_asked_for_too_many_and_skips_unknown_gc():
    gc = np.array([np.nan, 40, 41, 42, 43, 44.0])
    chosen = am.match_background(gc, np.array([1]), 40_000, np.random.default_rng(0))
    assert sorted(chosen.tolist()) == [2, 3, 4, 5]


def test_find_motifs_numbers_are_phyper_and_bh():
    t = _table()
    query, background = np.arange(40), np.arange(40, 400)
    out = am.find_motifs(t, query, background)
    hits = t.hits(np.arange(400))
    k, K = hits[:40].sum(axis=0), hits[40:].sum(axis=0)
    assert out["observed"].tolist() == k.tolist() and out["background"].tolist() == K.tolist()
    assert out["pval"][0] == pytest.approx(hypergeom.sf(k[0] - 1, 360, K[0], 40))
    assert out["fold_enrichment"][0] == pytest.approx((k[0] / 40) / (K[0] / 360))
    assert np.allclose(out["pval_adj"], tf_motifs.bh(out["pval"]))


def _da_frame():
    return pd.DataFrame({"group": ["A"] * 40 + ["A"] * 3 + ["B"] * 2, "peak_index": list(range(40)) + [100, 101, 102] + [200, 201],
                         "z": 1.0, "log2fc": [1.0] * 40 + [-1.0, 1.0, 1.0] + [1.0, 1.0], "pct_in": [0.5] * 40 + [0.5, 0.1, 0.5] + [0.5, 0.5],
                         "pct_rest": 0.1, "pval": [1e-5] * 40 + [1e-5, 1e-5, 0.1] + [1e-5, 1e-5], "pval_adj": 1e-3})


def test_query_is_the_vignettes_p_pct_and_positive_change():
    f = _da_frame()
    assert am.query_peaks(f, "A", "pval", 0.005, 0.2, 0.0).tolist() == list(range(40))
    # the threshold is a control, on p or on FDR
    assert 102 in am.query_peaks(f, "A", "pval_adj", 0.05, 0.2, 0.0).tolist()


def test_enrich_groups_finds_the_motif_and_notes_small_queries():
    t = _table()
    frame, summary = am.enrich_groups(t, _da_frame(), ["A", "B"], n_background=300, kept_per_group=5000)
    assert summary["A"]["n_query"] == 40 and summary["A"]["n_background"] == 300 and not summary["A"]["capped"]
    a = frame[frame["group"] == "A"]
    assert a.iloc[0]["motif"] == "SPI1" and a.iloc[0]["pval"] < 1e-5  # lowest p first
    assert a["gc_rich"].tolist()[a["motif"].tolist().index("MAX::MYC")]
    assert "only 2 query peaks" in summary["B"]["note"]
    # seeded per group: the same run gives the same numbers
    again, _ = am.enrich_groups(t, _da_frame(), ["A", "B"], n_background=300)
    assert a["pval"].tolist() == again[again["group"] == "A"]["pval"].tolist()
    other, _ = am.enrich_groups(t, _da_frame(), ["A"], n_background=300, seed=9)
    assert other["background"].tolist() != a["background"].tolist()


def test_enrich_groups_says_when_nothing_passes_and_drops_peaks_without_bases():
    t = _table()
    t.gc[5] = np.nan
    frame, summary = am.enrich_groups(t, _da_frame(), ["A"], n_background=300, p_max=1e-9)
    assert frame.empty and "No peak of A" in summary["A"]["note"]
    _f, s = am.enrich_groups(t, _da_frame(), ["A"], n_background=300)
    assert s["A"]["n_query"] == 39 and s["A"]["n_left_out"] == 1


def test_peaks_off_the_standard_chromosomes_are_neither_query_nor_background():
    t = _table()
    standard = np.ones(t.n_peaks, dtype=bool)
    standard[[3, 7]] = False  # two query peaks on a scaffold
    standard[40:] = False
    standard[40:340] = True  # 300 background candidates left, all drawn
    _f, s = am.enrich_groups(t, _da_frame(), ["A"], n_background=1_000, standard=standard)
    assert s["A"]["n_query"] == 38 and s["A"]["n_left_out"] == 2 and s["A"]["n_background"] == 300
    _f, every = am.enrich_groups(t, _da_frame(), ["A"], n_background=1_000)
    assert every["A"]["n_background"] > 300


def test_a_query_at_the_stored_rows_limit_is_capped():
    t = _table()
    _f, s = am.enrich_groups(t, _da_frame(), ["A"], n_background=300, kept_per_group=43, p_max=0.5)
    assert s["A"]["capped"]


def test_annotate_expression_separates_paralogues():
    frame = pd.DataFrame({"group": ["A", "A", "A"], "motif": ["SPI1", "MAX::MYC", "NOPE"]})
    stats = {"SPI1": {"gene": "SPI1", "pct": {"A": 0.5}, "logfc": {"A": 1.5}},
             "MYC": {"gene": "MYC", "pct": {"A": 0.02}, "logfc": {}}}
    am.annotate_expression(frame, stats)
    assert frame["tf_expressed"].tolist() == [True, False, False]
    assert frame["tfs"].tolist()[1] == "MAX, MYC" and frame["tf_gene"].tolist() == ["SPI1", "MYC", ""]
    assert np.isnan(frame["tf_pct"].iloc[2]) and frame["tf_log2fc"].iloc[0] == 1.5


def test_payload_keeps_small_p_values_and_opens_without_new_fields():
    t = pd.DataFrame({"group": ["A", "A"], "motif_id": ["M1", "M2"], "pval": [1e-12, 3e-300], "pval_adj": [2e-12, 6e-300],
                      "fold_enrichment": [float("inf"), float("nan")]})
    res = am.MotifEnrichment("r", "d", "c", ["A"], t, {"A": {"n_query": 1}}, {}, "m", "GRCh38", {}, "now")
    out = am.payload(res)
    json.dumps(out, allow_nan=False)
    rows = out["rows"]["A"]
    assert rows[0]["pval"] == pytest.approx(1e-12) and rows[1]["pval"] == pytest.approx(3e-300)
    assert rows[0]["fold_enrichment"] is None and rows[1]["fold_enrichment"] is None


def test_logo_is_a_standalone_svg():
    svg = am.logo_svg(ARNT, 'Arnt & "friend" (MA0004.1)')
    root = ElementTree.fromstring(svg)
    assert root.tag.endswith("svg") and "Arnt" in root.attrib["aria-label"]
    assert len([e for e in root if e.tag.endswith("path")]) > 6
    assert am.consensus(ARNT) == "CACGTG"
    h = am.logo_heights(ARNT)
    assert h.shape == ARNT.shape and (h.sum(axis=0) <= 2.0).all() and h[:, 3].sum() > 1.5
