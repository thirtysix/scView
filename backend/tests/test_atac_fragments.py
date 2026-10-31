"""ATAC fragment QC and coverage (core/atac_fragments.py): the Signac definitions on a file
small enough to count by hand."""

from __future__ import annotations

import gzip
import json

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from scview.core import atac_fragments as af
from scview.core import fragment_file as ff
from scview.core import gene_annotation, modality_store
from tests.fragments_fixture import write_fragments

CELLS = ["AAAC-1", "AAAG-1", "AAAT-1"]
TSS = 50_000


def _tss(*rows) -> pd.DataFrame:
    return pd.DataFrame(rows or [("chr1", TSS, "+")], columns=["chrom", "tss", "strand"])


def _qc(lines, tmp_path, cells=None, tss=None, **kw):
    path, _ = write_fragments(tmp_path / "f.tsv.gz", sorted(lines, key=lambda x: (x[0], x[1])))
    out = list(af.iter_qc(path, cells or CELLS, tss if tss is not None else _tss(), **kw))
    assert out[-1]["type"] == "result"
    return out[-1]["result"]


def _raw_window(tss: int, at: int, width: int, which: str) -> tuple[int, int]:
    """The raw values Signac counts for a region: see the module docstring."""
    rs = tss + 1 + at
    return rs + 2, rs + width + (1 if which == "end" else -2)


def test_windows_are_signacs_to_the_base(tmp_path):
    # A fragment placed so one cut sits at each edge of the center window, the other far away.
    lo_s, hi_s = _raw_window(TSS, -500, 1001, "start")
    lo_e, hi_e = _raw_window(TSS, -500, 1001, "end")
    far_end, far_start = TSS + 20_000, TSS - 20_000
    inside = [("chr1", lo_s, far_end, "AAAC-1", 1), ("chr1", hi_s, far_end, "AAAC-1", 1),
              ("chr1", far_start, lo_e, "AAAC-1", 1), ("chr1", far_start, hi_e, "AAAC-1", 1)]
    outside = [("chr1", lo_s - 1, far_end, "AAAG-1", 1), ("chr1", hi_s + 1, far_end, "AAAG-1", 1),
               ("chr1", far_start, lo_e - 1, "AAAG-1", 1), ("chr1", far_start, hi_e + 1, "AAAG-1", 1)]
    # One flank cut for each so the score is the center count / (1 / 200) / 1001.
    flank = _raw_window(TSS, -1000, 100, "start")[0]
    flanks = [("chr1", flank, far_end, "AAAC-1", 1), ("chr1", flank, far_end, "AAAG-1", 1)]
    r = _qc(inside + outside + flanks, tmp_path)
    assert r.tss_enrichment[0] == pytest.approx(4 / (1 / 200) / 1001)
    assert r.tss_enrichment[1] == pytest.approx(0.0)


def test_flank_windows_on_both_sides_each_count(tmp_path):
    far_e = TSS + 30_000
    up_lo, up_hi = _raw_window(TSS, -1000, 100, "start")
    dn_lo, dn_hi = _raw_window(TSS, 901, 100, "start")
    center = _raw_window(TSS, -500, 1001, "start")[0]
    lines = [("chr1", center, far_e, "AAAC-1", 1), ("chr1", center + 1, far_e, "AAAC-1", 1)]
    for pos in (up_lo, up_hi, dn_lo, dn_hi):
        lines.append(("chr1", pos, far_e, "AAAC-1", 1))
    # Just outside each flank: no count.
    for pos in (up_lo - 1, dn_hi + 1):
        lines.append(("chr1", pos, far_e, "AAAC-1", 1))
    r = _qc(lines, tmp_path, cells=["AAAC-1"])
    assert r.tss_enrichment[0] == pytest.approx(2 / (4 / 200) / 1001)


def test_a_cell_without_flank_cuts_takes_the_mean_flank(tmp_path):
    center = TSS + 1
    flank = _raw_window(TSS, -1000, 100, "start")[0]
    far = TSS + 30_000
    lines = [("chr1", flank, far, "AAAC-1", 1)] * 6 + [("chr1", center, far, "AAAG-1", 1)]
    r = _qc(lines, tmp_path)
    mean_flank = (6 / 200 + 0 + 0) / 3  # all cells, with the zeros
    assert r.tss_enrichment[1] == pytest.approx(1 / mean_flank / 1001, rel=1e-5)
    assert r.tss_enrichment[0] == 0.0 and r.tss_enrichment[2] == 0.0  # no center cut, no fragment


def test_overlapping_tss_windows_count_a_cut_for_each(tmp_path):
    far = TSS + 30_000
    flank = _raw_window(TSS, -1000, 100, "start")[0]
    lines = [("chr1", TSS, far, "AAAC-1", 1), ("chr1", flank, far, "AAAC-1", 1)]
    two = _qc(lines, tmp_path, cells=["AAAC-1"], tss=_tss(("chr1", TSS, "+"), ("chr1", TSS + 100, "-")))
    one = _qc(lines, tmp_path, cells=["AAAC-1"])
    assert two.tss_enrichment[0] == pytest.approx(2 * one.tss_enrichment[0])


def test_nucleosome_signal_uses_signacs_bounds(tmp_path):
    lines = [("chr1", 100, 100 + w, "AAAC-1", 1) for w in (146, 146, 147, 148, 293, 294, 400)]
    lines += [("chr1", 200, 348, "AAAG-1", 1)]  # mono only: no nucleosome-free fragment
    r = _qc(lines, tmp_path)
    assert r.nucleosome_signal[0] == pytest.approx(2 / 2)  # 148 and 293; 147 and 294 in neither
    assert np.isnan(r.nucleosome_signal[1]) and np.isnan(r.nucleosome_signal[2])
    assert r.n_fragments.tolist() == [7, 1, 0]


def test_counts_lines_not_reads_and_only_listed_cells(tmp_path):
    lines = [("chr1", 100, 150, "AAAC-1", 9), ("chr1", 100, 150, "OTHER-1", 9), ("chr2", 7, 60, "AAAC-1", 1)]
    r = _qc(lines, tmp_path)
    assert r.n_fragments.tolist() == [2, 0, 0] and r.n_lines == 3
    assert r.length_counts[50] == 1 and r.length_counts[53] == 1 and r.length_counts.sum() == 2


def test_length_histogram_keeps_long_ones_in_its_last_bin(tmp_path):
    r = _qc([("chr1", 100, 100 + w, "AAAC-1", 1) for w in (40, 999, 1000, 1001, 5000)], tmp_path)
    assert r.length_counts[40] == 1 and r.length_counts[999] == 1 and r.length_counts[af.MAX_LENGTH] == 3
    assert r.per_cell_lengths[0, 8] == 1 and r.per_cell_lengths[0, af.GROUP_LENGTH_BINS - 1] == 4


def test_tss_profile_reads_5_prime_to_3_prime_on_both_strands(tmp_path):
    far = TSS + 30_000
    lines = [("chr1", TSS + 200, far, "AAAC-1", 1)]
    plus = _qc(lines, tmp_path, cells=["AAAC-1"]).profile
    assert plus[af.PROFILE_FLANK + 200] == 1 and plus.sum() == 1
    minus = _qc(lines, tmp_path, cells=["AAAC-1"], tss=_tss(("chr1", TSS, "-"))).profile
    assert minus[af.PROFILE_FLANK - 200] == 1 and minus.sum() == 1


def test_stop_ends_the_pass_with_nothing_kept(tmp_path):
    path, _ = write_fragments(tmp_path / "f.tsv.gz", [("chr1", 10, 60, "AAAC-1", 1)])
    with pytest.raises(af.FragmentsCancelled):
        list(af.iter_qc(path, CELLS, _tss(), stop=lambda: True))


def test_progress_comes_before_the_result(tmp_path):
    lines = [("chr1", i, i + 50, "AAAC-1", 1) for i in range(30_000)]
    path, _ = write_fragments(tmp_path / "f.tsv.gz", lines)
    out = list(af.iter_qc(path, CELLS, _tss()))
    assert out[-1]["type"] == "result" and out[0]["type"] == "progress"


# -- saving ------------------------------------------------------------------


def _adata(cells=None) -> ad.AnnData:
    cells = cells or CELLS
    obs = pd.DataFrame({"cell_type": pd.Categorical(["B", "T", "B"][: len(cells)])}, index=cells)
    return ad.AnnData(X=sp.csr_matrix(np.ones((len(cells), 2), dtype=np.float32)), obs=obs,
                      var=pd.DataFrame(index=["GeneA", "GeneB"]))


def _result(tmp_path) -> af.QcResult:
    far = TSS + 30_000
    flank = _raw_window(TSS, -1000, 100, "start")[0]
    lines = [("chr1", TSS, TSS + 100, "AAAC-1", 1), ("chr1", flank, far, "AAAC-1", 1),
             ("chr1", TSS, TSS + 200, "AAAG-1", 1), ("chr1", TSS + 5, TSS + 80, "AAAT-1", 1)]
    return _qc(lines, tmp_path)


def test_save_writes_obs_uns_and_a_step(tmp_path):
    from scview.core import provenance

    adata = _adata()
    info = af.save(adata, _result(tmp_path), genome="GRCh38", source="GENCODE v32 basic", file_name="f.tsv.gz",
                   resources={"seconds": 1.0})
    assert adata.obs[af.OBS_N_FRAGMENTS].tolist() == [2, 1, 1]
    assert adata.obs[af.OBS_TSS].iloc[0] > 0
    assert info["medians"]["n_fragments"] == 1.0 and info["n_cells"] == 3 and info["n_lines"] == 4
    assert info["tss_profile"]["n_tss"] == 1 and len(info["tss_profile"]["fold"]) == 400
    assert info["length_histogram"]["n"] == 4 and info["params"]["genome"] == "GRCh38"
    stored = json.loads(adata.uns[af.UNS_KEY])
    assert stored["medians"] == info["medians"] and af.saved(adata)["version"] == af.VERSION
    assert provenance.read_provenance(adata)["history"][-1]["step"] == af.STEP


def test_saved_after_a_cell_change_is_refused(tmp_path):
    with pytest.raises(af.FragmentsError, match="cells changed"):
        af.save(_adata(CELLS[:2]), _result(tmp_path), genome="GRCh38", source=None, file_name=None)


def test_low_tss_enrichment_says_what_it_may_mean(tmp_path):
    adata = _adata()
    res = _result(tmp_path)
    res.tss_enrichment = np.array([0.5, 0.6, 0.7], dtype=np.float32)
    assert "another build" in af.save(adata, res, genome="GRCh38", source=None, file_name=None)["notes"][0]


def test_grounding_lines_carry_the_medians(tmp_path):
    adata = _adata()
    assert af.grounding_lines(adata) == ([], "")
    af.save(adata, _result(tmp_path), genome="GRCh38", source=None, file_name=None)
    lines, detail = af.grounding_lines(adata)
    assert all(af.TAG in x for x in lines) and "Median TSS enrichment" in lines[1] and "median TSS" in detail


def test_lengths_by_group_survive_a_filter(tmp_path):
    adata = _adata()
    af.write_lengths(_result(tmp_path), tmp_path, "d")
    out = af.lengths_by_group(adata, tmp_path, "d", "cell_type")
    assert [g["name"] for g in out["groups"]] == ["B", "T"] and out["bin"] == af.GROUP_LENGTH_BIN
    b = out["groups"][0]
    assert b["n_cells"] == 2 and b["n_fragments"] == 3 and sum(b["share"]) == pytest.approx(1.0, abs=1e-4)
    subset = adata[[0, 1]].copy()
    assert af.lengths_by_group(subset, tmp_path, "d", "cell_type")["groups"][0]["n_fragments"] == 2
    assert af.lengths_by_group(adata, tmp_path, "d", None)["groups"][0]["n_fragments"] == 4


def test_lengths_before_the_qc_says_run_it(tmp_path):
    with pytest.raises(af.FragmentsError, match="Run the fragment QC"):
        af.lengths_by_group(_adata(), tmp_path, "d", None)


def test_values_by_group_are_capped_honestly(tmp_path):
    adata = _adata()
    af.save(adata, _result(tmp_path), genome="GRCh38", source=None, file_name=None)
    out = af.values_by_group(adata, "cell_type", max_per_group=1)
    b = out["groups"][0]
    assert b["name"] == "B" and b["n"] == 2 and b["n_shown"] == 1 and len(b["tss_enrichment"]) == 1
    with pytest.raises(af.FragmentsError, match="no column"):
        af.values_by_group(adata, "nope")
    with pytest.raises(af.FragmentsError, match="Run the fragment QC"):
        af.values_by_group(_adata(), None)


# -- attaching ----------------------------------------------------------------


def _attached(tmp_path, lines, cells=None, species=None, **kw):
    folder = modality_store.store_dir(tmp_path, "d")
    folder.mkdir(parents=True, exist_ok=True)
    write_fragments(folder / af.FRAGMENTS_FILE, lines)
    return af.verify(tmp_path, "d", cells or CELLS, file_name="atac_fragments.tsv.gz", species=species,
                     auto_download=False, **kw)


def _many(chroms=("chr1", "chr22", "chrX"), barcode="AAAC-1"):
    return [(c, i * 10, i * 10 + 50, barcode, 1) for c in chroms for i in range(200)]


def test_verify_records_the_share_and_genome(tmp_path):
    state = _attached(tmp_path, _many())
    assert state["share"] == 1.0 and state["genome"] == "GRCh38" and state["cells_seen"] == 1
    st = af.status(tmp_path, "d", 3)
    assert st["attached"] and st["missing"] is None and st["estimate"]["peak_bytes"] >= af.BASE_PEAK_BYTES
    assert st["estimate"]["seconds"] >= 0 and st["genome"] == "GRCh38" and st["name"] == "atac_fragments.tsv.gz"


def test_verify_refuses_a_low_share_with_the_numbers_and_removes_the_files(tmp_path):
    lines = _many(barcode="OTHER-1") + [("chrX", 90_000, 90_050, "AAAC-1", 1)]
    with pytest.raises(af.FragmentsError) as e:
        _attached(tmp_path, lines)
    message = str(e.value)
    assert "1 of the first 601 fragments (0.2%)" in message and "3 cells" in message and "5%" in message
    assert not af.fragments_path(tmp_path, "d").exists() and not af.index_path(tmp_path, "d").exists()


def test_verify_refuses_the_other_species(tmp_path):
    with pytest.raises(af.FragmentsError, match="GRCh38's but this dataset's genes are mouse"):
        _attached(tmp_path, _many(), species="mouse")


def test_status_says_what_is_missing_either_way(tmp_path):
    assert af.status(tmp_path, "d")["attached"] is False
    with pytest.raises(af.FragmentsError, match="No ATAC fragments file") as e:
        af.require_files(tmp_path, "d")
    assert e.value.status == 404
    _attached(tmp_path, _many())
    af.index_path(tmp_path, "d").unlink()
    st = af.status(tmp_path, "d")
    assert st["fragments"] and not st["attached"] and "Cell Ranger ARC" in st["missing"]
    with pytest.raises(af.FragmentsError, match="Cell Ranger ARC"):
        af.require_files(tmp_path, "d")
    write_fragments(af.index_path(tmp_path, "d").with_name("x.tsv.gz"), _many())
    af.index_path(tmp_path, "d").with_name("x.tsv.gz.tbi").rename(af.index_path(tmp_path, "d"))
    af.fragments_path(tmp_path, "d").unlink()
    assert "Cell Ranger ARC" in af.status(tmp_path, "d")["missing"]


def test_check_head_tells_gzip_from_other_files():
    with pytest.raises(af.FragmentsError, match="plain gzip"):
        af.check_head(gzip.compress(b"chr1\t1\t2\tA\t1\n"), "fragments")
    with pytest.raises(af.FragmentsError, match="not a BGZF"):
        af.check_head(b"chr1\t1\t2\tA\t1\n" * 50, "fragments")
    af.check_head(b"anything", "index")


def test_role_of_names():
    assert af.role_of("atac_fragments.tsv.gz") == "fragments" and af.role_of("x/y.tsv.gz.tbi") == "index"
    assert af.role_of("matrix.h5") is None


@pytest.mark.parametrize("names,genome", [(["chr1", "chr22", "chrX"], "GRCh38"), (["chr1", "chr19", "chrX"], "mm10"),
                                          (["1", "21"], "GRCh38"), (["scaffold_1"], None), (["chr1", "chr2"], None)])
def test_genome_from_names(names, genome):
    assert af.genome_from_names(names) == genome


def test_side_files_move_in_and_verify(tmp_path):
    source = tmp_path / "staged" / "my_fragments.tsv.gz"
    source.parent.mkdir()
    frag, idx = write_fragments(source, _many())
    state = af.attach_side_files(tmp_path, "d", frag, idx, CELLS)
    assert not frag.exists() and not idx.exists() and af.fragments_path(tmp_path, "d").exists()
    assert state["name"] == "my_fragments.tsv.gz" and state["share"] == 1.0


def test_detach_removes_all(tmp_path):
    _attached(tmp_path, _many())
    af.write_lengths(_result(tmp_path), tmp_path, "d")
    af.detach(tmp_path, "d")
    assert not any(modality_store.store_dir(tmp_path, "d").iterdir())


# -- coverage ----------------------------------------------------------------


def _reader(tmp_path, lines, cells=None):
    path, idx = write_fragments(tmp_path / "c.tsv.gz", sorted(lines, key=lambda x: (x[0], x[1])))
    return ff.FragmentReader(path, idx, cells or CELLS)


def _with_totals(adata, totals):
    adata.obs[af.OBS_N_FRAGMENTS] = totals
    return adata


def test_parse_region_forms():
    assert af.parse_region("chr1:1,000-5,000") == ("chr1", 1000, 5000)
    assert af.parse_region(" chrX:10-20 ") == ("chrX", 10, 20)
    with pytest.raises(af.FragmentsError, match="chr1:1000-5000"):
        af.parse_region("banana")
    with pytest.raises(af.FragmentsError, match="ends before"):
        af.parse_region("chr1:50-10")


def test_coverage_is_mean_depth_per_bin_scaled_per_group(tmp_path):
    # B = cells 0, 2 (1,000 fragments in all); T = cell 1 (500).
    lines = [("chr1", 100, 200, "AAAC-1", 1), ("chr1", 150, 250, "AAAT-1", 1), ("chr1", 100, 200, "AAAG-1", 1),
             ("chr1", 900, 2000, "OTHER", 1)]
    adata = _with_totals(_adata(), [400, 500, 600])
    out = af.coverage(_reader(tmp_path, lines), adata, "chr1", 100, 300, group_by="cell_type", bins=2)
    assert out["x"] == [150.0, 250.0] and out["bin_size"] == 100.0 and out["n_fragments"] == 3
    by = {t["name"]: t for t in out["tracks"]}
    # B: 150 bases of depth 1 + 50 of depth 2 in the first bin = mean 1.5; 50 bases in the second = 0.5.
    assert by["B"]["values"] == pytest.approx([1.5 / 1000 * 1e6, 0.5 / 1000 * 1e6])
    assert by["T"]["values"] == pytest.approx([1.0 / 500 * 1e6, 0.0])
    assert by["B"]["n_cells"] == 2 and by["B"]["total_fragments"] == 1000 and out["ymax"] == pytest.approx(2000.0)
    assert "per million" in out["normalisation"]


def test_coverage_without_the_qc_says_what_it_shows(tmp_path):
    lines = [("chr1", 100, 200, "AAAC-1", 1)]
    out = af.coverage(_reader(tmp_path, lines), _adata(), "chr1", 100, 200, bins=1)
    assert "per 1,000 cells" in out["normalisation"] and out["tracks"][0]["name"] == "All cells"
    assert out["tracks"][0]["values"] == pytest.approx([1.0 / 3 * 1000], rel=1e-3)


def test_coverage_clips_fragments_to_the_window(tmp_path):
    lines = [("chr1", 0, 1000, "AAAC-1", 1)]
    out = af.coverage(_reader(tmp_path, lines), _with_totals(_adata(), [1, 1, 1]), "chr1", 400, 500,
                      group_by="cell_type", groups=["B"], bins=1)
    assert [t["name"] for t in out["tracks"]] == ["B"] and out["tracks"][0]["values"] == pytest.approx([5e5])


def test_coverage_cap_and_misses_are_sentences(tmp_path):
    reader = _reader(tmp_path, [("chr1", 100, 200, "AAAC-1", 1)])
    with pytest.raises(af.FragmentsError, match="up to 2 Mb"):
        af.coverage(reader, _adata(), "chr1", 0, af.MAX_WINDOW + 1)
    with pytest.raises(af.FragmentsError, match="no chromosome 'chrQ'.*chrQ:5-10") as e:
        af.coverage(reader, _adata(), "chrQ", 5, 10)
    assert e.value.status == 404
    with pytest.raises(af.FragmentsError, match="no column"):
        af.coverage(reader, _adata(), "chr1", 0, 500, group_by="nope")
    with pytest.raises(af.FragmentsError, match="no group 'Z'"):
        af.coverage(reader, _adata(), "chr1", 0, 500, group_by="cell_type", groups=["Z"])
    out = af.coverage(reader, _adata(), "chr1", 50_000, 60_000)
    assert out["n_fragments"] == 0 and "No fragment of these cells overlaps chr1:50,000-60,000" in out["notes"][0]


def test_coverage_keeps_the_largest_groups_past_the_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(af, "MAX_GROUPS", 1)
    out = af.coverage(_reader(tmp_path, [("chr1", 100, 200, "AAAC-1", 1)]), _adata(), "chr1", 0, 500,
                      group_by="cell_type", bins=5)
    assert [t["name"] for t in out["tracks"]] == ["B"] and "largest" in out["notes"][0]


def _gene_table(tmp_path) -> pd.DataFrame:
    gtf = tmp_path / "t.gtf"
    gtf.write_text(
        'chr1\tX\tgene\t10001\t20000\t.\t+\t.\tgene_id "ENSG1.1"; gene_type "protein_coding"; gene_name "AAA";\n'
        'chr1\tX\tgene\t30001\t31000\t.\t-\t.\tgene_id "ENSG2.1"; gene_type "lncRNA"; gene_name "BBB";\n'
        'chr2\tX\tgene\t500\t900\t.\t+\t.\tgene_id "ENSG3.1"; gene_type "protein_coding"; gene_name "CCC";\n')
    return gene_annotation.parse_gtf(gtf)


def test_gene_window_is_the_body_plus_5kb_found_in_any_case(tmp_path):
    table = _gene_table(tmp_path)
    assert af.gene_window(table, "AAA")[:3] == ("chr1", 5000, 25000)
    assert af.gene_window(table, "bbb")[3]["strand"] == "-"
    assert af.gene_window(table, "ENSG3")[0:2] == ("chr2", 0)  # floor at the chromosome's start
    with pytest.raises(af.FragmentsError, match="no gene 'ZZZ'"):
        af.gene_window(table, "ZZZ")


def test_genes_in_region_with_strand(tmp_path):
    genes = af.genes_in(_gene_table(tmp_path), "chr1", 19_000, 30_500)
    assert [g["name"] for g in genes] == ["AAA", "BBB"] and genes[1]["strand"] == "-" and genes[0]["start"] == 10000
    assert af.genes_in(None, "chr1", 0, 10) == []


def test_tss_table_keeps_protein_coding_off_chrm(tmp_path):
    gtf = tmp_path / "t.gtf"
    gtf.write_text(
        'chr1\tX\tgene\t101\t200\t.\t+\t.\tgene_id "E1.1"; gene_type "protein_coding"; gene_name "A";\n'
        'chr1\tX\tgene\t301\t400\t.\t-\t.\tgene_id "E2.1"; gene_type "protein_coding"; gene_name "B";\n'
        'chr1\tX\tgene\t501\t600\t.\t+\t.\tgene_id "E3.1"; gene_type "lncRNA"; gene_name "C";\n'
        'chrM\tX\tgene\t1\t50\t.\t+\t.\tgene_id "E4.1"; gene_type "protein_coding"; gene_name "D";\n')
    gene_annotation.clear_cache()
    gene_annotation.seed_from(gtf, tmp_path, "GRCh38")
    try:
        t = af.tss_table("GRCh38", tmp_path, auto_download=False)
    finally:
        gene_annotation.clear_cache()
    assert t["tss"].tolist() == [100, 399] and t["strand"].tolist() == ["+", "-"]


def test_peaks_in_reads_the_atac_store_when_there_is_one(tmp_path):
    assert af.peaks_in(tmp_path, "d", "chr1", 0, 100) == []
    modality_store.write(modality_store.store_path(tmp_path, "d", af.PEAK_STORE),
                         {"counts": sp.csr_matrix(np.ones((3, 3), dtype=np.float32))}, CELLS,
                         ["chr1:100-200", "chr1:300-400", "chr2:100-200"])
    assert af.peaks_in(tmp_path, "d", "chr1", 150, 310) == [{"start": 100, "end": 200}, {"start": 300, "end": 400}]
    assert af.peaks_in(tmp_path, "d", "chr9", 0, 10) == []


def test_peaks_in_prefers_an_interval_column(tmp_path):
    modality_store.write(modality_store.store_path(tmp_path, "e", af.PEAK_STORE),
                         {"counts": sp.csr_matrix(np.ones((3, 1), dtype=np.float32))}, CELLS, ["peak1"],
                         feature_attrs={"interval": ["chr3-10-20"]})
    assert af.peaks_in(tmp_path, "e", "chr3", 0, 100) == [{"start": 10, "end": 20}]
