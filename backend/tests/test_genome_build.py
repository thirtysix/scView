"""hg19 vs GRCh38, mm9 and mm39 vs mm10 on ATAC input (core/genome_build.py): told from a named
reference, a fragments header or coordinates past the assumed build's chromosome ends. hg19 / mm9
are recorded and refused by every genome-dependent step with one sentence; mm39 is a build scView
has (since v0.77). Other species and builds get one sentence too."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from scview.core import atac, atac_import, gene_annotation, genome_build, modality_store, provenance
from scview.core import atac_fragments as af
from tests import multiome_fixture as mf
from tests.fragments_fixture import write_fragments
from tests.test_atac_import import _dataset, _session, _store

HG19_HEADER = ("# id=old\n# pipeline_name=cellranger-atac\n"
               "# reference_path=/refs/refdata-cellranger-atac-hg19-1.2.0\n")
CELLS = ["AAAC-1", "AAAG-1", "AAAT-1"]


@pytest.mark.parametrize("name", ["hg19", "GRCh37", "refdata-cellranger-atac-hg19-1.2.0", "b37", "hs37d5",
                                  "/refs/GRCh37.p13/fasta"])
def test_names_hg19(name):
    assert genome_build.names_hg19(name)


@pytest.mark.parametrize("name", ["GRCh38", "hg38", "mm10", "GRCh38-2020-A", "refdata-cellranger-arc-GRCh38-2020-A-2.0.0",
                                  "", None, "hg190"])
def test_not_hg19(name):
    assert not genome_build.names_hg19(name)


def test_positions_past_grch38_ends_are_hg19():
    # 9q34.3 on hg19: past GRCh38's chr9 (138,394,717), within hg19's (141,213,431).
    assert genome_build.from_positions(["chr1", "chr9"], [1_000, 139_400_500]) == genome_build.HG19
    assert genome_build.from_positions(["19"], [59_000_000]) == genome_build.HG19  # Ensembl-style names


def test_positions_within_both_or_past_both_say_nothing():
    assert genome_build.from_positions(["chr1", "chr9"], [1_000, 138_000_000]) is None
    # GRCh38's chr17 is longer than hg19's: a GRCh38 set past hg19's end.
    assert genome_build.from_positions(["chr17", "chr9"], [83_000_000, 139_400_500]) is None
    # Mouse chr19 (61 Mb) runs past both human ends.
    assert genome_build.from_positions(["chr19", "chr13"], [60_000_000, 114_500_000]) is None
    assert genome_build.from_positions(["chrM", "KI270728.1", "chr9"], [16_000, 1_000, "x"]) is None
    assert genome_build.from_intervals([("chr9", 139_400_000, 139_400_500), None]) == genome_build.HG19


def test_canonical_genome_refuses_hg19_with_the_sentence():
    for name in ("hg19", "GRCh37"):
        with pytest.raises(gene_annotation.AnnotationError) as e:
            gene_annotation.canonical_genome(name)
        assert str(e.value) == genome_build.HG19_REFUSAL
    with pytest.raises(gene_annotation.AnnotationError, match="'xenopus', a reference scView does not know"):
        gene_annotation.canonical_genome("xenopus")
    assert gene_annotation.canonical_genome("GRCh38-2020-A") == "GRCh38"


def test_gene_activity_refuses_hg19(tmp_path):
    choice = atac.choose_genes(mf.as_anndata(), "hg19", tmp_path, auto_download=False)
    assert choice.names == [] and choice.reason == genome_build.HG19_REFUSAL


# -- import ------------------------------------------------------------------


def test_tenx_h5_naming_hg19_is_recorded_and_noted(tmp_path):
    h5 = mf.write_tenx_h5(tmp_path / "pbmc_filtered_feature_bc_matrix.h5", genome="hg19")
    m, sid = _session(tmp_path, h5)
    did = m._build_dataset(sid)
    a = _dataset(m, did)
    with _store(m, did) as s:
        assert s.meta["genome"] == "hg19" and s.meta["genome_note"] == genome_build.HG19_REFUSAL
    info = atac_import.read_info(a)
    assert info["genome"] == "hg19" and info["genome_note"] == genome_build.HG19_REFUSAL
    assert atac_import.dataset_state(a, m.data_dir, did)["genome_note"] == genome_build.HG19_REFUSAL
    history = provenance.read_provenance(a)["history"]
    assert any(genome_build.HG19_REFUSAL in str(h.get("note") or "") for h in history)


def test_tenx_h5_on_grch38_has_no_note(tmp_path):
    h5 = mf.write_tenx_h5(tmp_path / "pbmc_filtered_feature_bc_matrix.h5")
    m, sid = _session(tmp_path, h5)
    did = m._build_dataset(sid)
    info = atac_import.read_info(_dataset(m, did))
    assert info["genome"] == "GRCh38" and "genome_note" not in info


def _plan(intervals, genome="GRCh38", assumed=True):
    return atac_import.UnitPeaks("s", "f", "10x MEX", rows=object(), cols=np.arange(len(intervals)),
                                 names=list(intervals), intervals=list(intervals), barcodes=pd.Index(CELLS),
                                 genome=genome, genome_assumed=assumed)


def test_an_assumed_genome_follows_the_coordinates():
    hg19 = atac_import.check_build(_plan(["chr1:100-600", "chr9:139400000-139400500"]))
    assert hg19.genome == "hg19" and not hg19.genome_assumed and hg19.genome_note == genome_build.HG19_REFUSAL
    unnamed = atac_import.check_build(_plan(["chr9:139400000-139400500"], genome=None))
    assert unnamed.genome == "hg19"
    kept = atac_import.check_build(_plan(["chr1:100-600", "chr9:130000000-130000500"]))
    assert kept.genome == "GRCh38" and kept.genome_assumed and kept.genome_note is None


def test_a_named_genome_is_not_second_guessed_by_coordinates():
    named = atac_import.check_build(_plan(["chr9:139400000-139400500"], assumed=False))
    assert named.genome == "GRCh38" and named.genome_note is None
    mouse = atac_import.check_build(_plan(["chr9:139400000-139400500"], genome="mm10"))
    assert mouse.genome == "mm10"


# -- fragments ---------------------------------------------------------------


def _attach(tmp_path, lines, header="# id=test\n", species=None):
    folder = modality_store.store_dir(tmp_path, "d")
    folder.mkdir(parents=True, exist_ok=True)
    write_fragments(folder / af.FRAGMENTS_FILE, lines, header=header)
    return af.verify(tmp_path, "d", CELLS, file_name="atac_fragments.tsv.gz", species=species, auto_download=False)


def _lines(extra=()):
    out = [(c, i * 10, i * 10 + 50, "AAAC-1", 1) for c in ("chr1", "chr22", "chrX") for i in range(200)]
    return sorted([*out, *extra], key=lambda x: (x[0], x[1]))


def test_fragments_header_naming_hg19(tmp_path):
    folder = tmp_path / "f"
    folder.mkdir()
    path, _ = write_fragments(folder / "x.tsv.gz", _lines(), header=HG19_HEADER)
    header = genome_build.read_header(path)
    assert header["pipeline_name"] == "cellranger-atac" and genome_build.from_header(header) == "hg19"
    assert genome_build.read_header(tmp_path / "missing.tsv.gz") == {}

    state = _attach(tmp_path, _lines(), header=HG19_HEADER, species="human")  # hg19 is human: kept
    assert state["genome"] == "hg19" and state["genome_note"] == genome_build.HG19_REFUSAL
    st = af.status(tmp_path, "d")
    assert st["genome"] == "hg19" and st["genome_note"] == genome_build.HG19_REFUSAL


def test_fragments_index_reaching_past_grch38_ends(tmp_path):
    late = [("chr9", 139_400_000 + i * 10, 139_400_050 + i * 10, "AAAC-1", 1) for i in range(5)]
    state = _attach(tmp_path, _lines(late))
    assert state["genome"] == "hg19"


def test_fragments_on_grch38_stay_grch38(tmp_path):
    arc = "# reference_path=/mnt/refdata-cellranger-arc-GRCh38-2020-A-2.0.0\n"
    early = [("chr9", 138_000_000 + i * 10, 138_000_050 + i * 10, "AAAC-1", 1) for i in range(5)]
    state = _attach(tmp_path, _lines(early), header=arc)
    assert state["genome"] == "GRCh38" and "genome_note" not in state
    with pytest.raises(af.FragmentsError, match="genes are mouse"):
        _attach(tmp_path, _lines(), header=HG19_HEADER, species="mouse")


def test_tss_table_refuses_hg19(tmp_path):
    with pytest.raises(gene_annotation.AnnotationError, match="hg19"):
        af.tss_table("hg19", tmp_path, auto_download=False)


# -- mouse: mm9 vs mm10 ------------------------------------------------------

MM9_HEADER = "# id=old\n# pipeline_name=cellranger-atac\n# reference_path=/refs/refdata-mm9-NCBIM37\n"


@pytest.mark.parametrize("name", ["mm9", "MM9", "NCBIM37", "refdata-mm9-1.0", "/refs/MGSCv37/fasta"])
def test_names_mm9(name):
    assert genome_build.names_mm9(name) and genome_build.old_build(name) == genome_build.MM9
    assert genome_build.refusal(name) == genome_build.MM9_REFUSAL


@pytest.mark.parametrize("name", ["mm10", "GRCm38", "mm39", "GRCm39", "refdata-cellranger-arc-mm10-2020-A-2.0.0",
                                  "mm90", "hg19", "", None])
def test_not_mm9(name):
    assert not genome_build.names_mm9(name)


def test_positions_past_mm10_ends_are_mm9():
    # Distal chr7 on mm9 (H19/Igf2): past mm10's chr7 (145,441,459), within mm9's (152,524,553).
    assert genome_build.from_positions(["chr1", "chr7"], [1_000, 149_800_500]) == genome_build.MM9
    assert genome_build.from_positions(["8"], [130_500_000]) == genome_build.MM9  # Ensembl-style names
    assert genome_build.from_intervals([("chr7", 149_800_000, 149_800_500), None]) == genome_build.MM9


def test_mouse_positions_within_both_or_past_both_say_nothing():
    assert genome_build.from_positions(["chr1", "chr7"], [1_000, 140_000_000]) is None  # a clean mm10 set
    # mm10's chr13 and chrY run longer than mm9's: an mm10 set past mm9's end is not mm9.
    assert genome_build.from_positions(["chr7", "chr13"], [149_800_500, 120_400_000]) is None
    assert genome_build.from_positions(["chr7", "chrY"], [149_800_500, 20_000_000]) is None
    # A human chromosome mouse lacks, or human lengths past both: not mouse.
    assert genome_build.from_positions(["chr7", "chr21"], [149_800_500, 1_000]) is None
    assert genome_build.from_positions(["chr1", "chr7"], [240_000_000, 149_800_500]) is None


def test_canonical_genome_refuses_mm9_with_the_sentence(tmp_path):
    from scview.api.v1 import atac_motifs
    from scview.core import genome_seq

    for name in ("mm9", "NCBIM37"):
        with pytest.raises(gene_annotation.AnnotationError) as e:
            gene_annotation.canonical_genome(name)
        assert str(e.value) == genome_build.MM9_REFUSAL
    with pytest.raises(gene_annotation.AnnotationError, match=r"on mm9 \(NCBIM37\)"):
        genome_seq.file_path(tmp_path, "mm9")  # GC, motifs and links read the 2bit through this
    with pytest.raises(atac.AtacError) as err:
        atac_motifs._genome("mm9")
    assert err.value.status == 422 and str(err.value) == genome_build.MM9_REFUSAL
    assert gene_annotation.canonical_genome("mm10-2020-A") == "mm10"
    choice = atac.choose_genes(mf.as_anndata(), "mm9", tmp_path, auto_download=False)
    assert choice.names == [] and choice.reason == genome_build.MM9_REFUSAL


def test_tenx_h5_naming_mm9_is_recorded_and_noted(tmp_path):
    h5 = mf.write_tenx_h5(tmp_path / "e18_filtered_feature_bc_matrix.h5", genome="mm9")
    m, sid = _session(tmp_path, h5)
    did = m._build_dataset(sid)
    a = _dataset(m, did)
    info = atac_import.read_info(a)
    assert info["genome"] == "mm9" and info["genome_note"] == genome_build.MM9_REFUSAL
    assert atac_import.dataset_state(a, m.data_dir, did)["genome_note"] == genome_build.MM9_REFUSAL
    history = provenance.read_provenance(a)["history"]
    assert any(genome_build.MM9_REFUSAL in str(h.get("note") or "") for h in history)


def test_an_assumed_mm10_follows_the_coordinates():
    mm9 = atac_import.check_build(_plan(["chr1:100-600", "chr7:149800000-149800500"], genome="mm10"))
    assert mm9.genome == "mm9" and not mm9.genome_assumed and mm9.genome_note == genome_build.MM9_REFUSAL
    unnamed = atac_import.check_build(_plan(["chr7:149800000-149800500"], genome=None))
    assert unnamed.genome == "mm9"
    kept = atac_import.check_build(_plan(["chr1:100-600", "chr7:140000000-140000500"], genome="mm10"))
    assert kept.genome == "mm10" and kept.genome_assumed and kept.genome_note is None
    named = atac_import.check_build(_plan(["chr7:149800000-149800500"], genome="mm10", assumed=False))
    assert named.genome == "mm10" and named.genome_note is None
    # An assumed human genome is not turned into mouse by coordinates.
    human = atac_import.check_build(_plan(["chr7:149800000-149800500"]))
    assert human.genome == "GRCh38" and human.genome_note is None


def _mouse_lines(extra=()):
    out = [(c, i * 10, i * 10 + 50, "AAAC-1", 1) for c in ("chr1", "chr19", "chrX") for i in range(200)]
    return sorted([*out, *extra], key=lambda x: (x[0], x[1]))


def test_fragments_header_naming_mm9(tmp_path):
    state = _attach(tmp_path, _mouse_lines(), header=MM9_HEADER, species="mouse")
    assert state["genome"] == "mm9" and state["genome_note"] == genome_build.MM9_REFUSAL
    st = af.status(tmp_path, "d")
    assert st["genome"] == "mm9" and st["genome_note"] == genome_build.MM9_REFUSAL
    with pytest.raises(af.FragmentsError, match="genes are human"):
        _attach(tmp_path, _mouse_lines(), header=MM9_HEADER, species="human")


def test_fragments_index_reaching_past_mm10_ends(tmp_path):
    late = [("chr7", 149_800_000 + i * 10, 149_800_050 + i * 10, "AAAC-1", 1) for i in range(5)]
    state = _attach(tmp_path, _mouse_lines(late))
    assert state["genome"] == "mm9" and state["genome_note"] == genome_build.MM9_REFUSAL


def test_fragments_on_mm10_stay_mm10(tmp_path):
    arc = "# reference_path=/mnt/refdata-cellranger-arc-mm10-2020-A-2.0.0\n"
    early = [("chr7", 140_000_000 + i * 10, 140_000_050 + i * 10, "AAAC-1", 1) for i in range(5)]
    state = _attach(tmp_path, _mouse_lines(early), header=arc)
    assert state["genome"] == "mm10" and "genome_note" not in state
    # An hg19 header on a mouse file says nothing about mouse builds.
    state = _attach(tmp_path, _mouse_lines(early), header=HG19_HEADER)
    assert state["genome"] == "mm10" and "genome_note" not in state


def test_tss_table_refuses_mm9(tmp_path):
    with pytest.raises(gene_annotation.AnnotationError, match="mm9"):
        af.tss_table("mm9", tmp_path, auto_download=False)


# -- mouse: mm39 vs mm10 -----------------------------------------------------

MM39_HEADER = ("# id=new\n# pipeline_name=cellranger-arc\n"
               "# reference_path=/refs/refdata-cellranger-arc-GRCm39-2024-A\n")
# Itgb1's TSS on mm39 chr8 (129,412,134): past mm10's chr8 (129,401,213), within mm39's (130,127,694).
ITGB1_MM39 = 129_412_000


@pytest.mark.parametrize("name", ["mm39", "MM39", "GRCm39", "GRCm39-2024-A", "refdata-cellranger-arc-GRCm39-2024-A",
                                  "/refs/refdata-gex-GRCm39-2024-A/fasta/genome.fa"])
def test_names_mm39(name):
    assert genome_build.names_mm39(name) and genome_build.old_build(name) == genome_build.MM39
    # a build scView has since v0.77: no refusal
    assert genome_build.refusal(name) is None and genome_build.unsupported(name) is None
    assert genome_build.supported(name) == "mm39"


@pytest.mark.parametrize("name", ["mm10", "mm10-2020-A", "GRCm38", "refdata-cellranger-arc-mm10-2020-A-2.0.0",
                                  "mm9", "NCBIM37", "mm390", "GRCm390", "hg19", "GRCh38", "", None])
def test_not_mm39(name):
    assert not genome_build.names_mm39(name)
    assert genome_build.old_build(name) != genome_build.MM39


def test_positions_past_mm10_ends_inside_mm39_are_mm39():
    assert genome_build.from_positions(["chr1", "chr8"], [1_000, ITGB1_MM39]) == genome_build.MM39
    # chr13 and chr4 are longer on mm39 than on mm10 but shorter on mm9: only mm39 holds them.
    assert genome_build.from_positions(["13"], [120_500_000]) == genome_build.MM39
    assert genome_build.from_positions(["chr4"], [156_520_000]) == genome_build.MM39
    assert genome_build.from_intervals([("chr8", ITGB1_MM39, ITGB1_MM39 + 500), None]) == genome_build.MM39


def test_mm39_is_not_mm9_and_mm9_is_not_mm39():
    # mm9's distal chr7 (H19/Igf2) runs past mm39's chr7 (144,995,196) too: still mm9.
    assert genome_build.from_positions(["chr7"], [149_800_500]) == genome_build.MM9
    assert genome_build.from_positions(["chr7", "chr8"], [149_800_500, ITGB1_MM39]) == genome_build.MM9
    # chr8 past mm39's end but inside mm9's: mm9.
    assert genome_build.from_positions(["chr8"], [130_500_000]) == genome_build.MM9
    # A whole mm39 set also reaches chr13 past mm9's end: never mm9.
    assert genome_build.from_positions(["chr8", "chr13"], [ITGB1_MM39, 120_500_000]) == genome_build.MM39
    # Past mm39's end (mm9 chr7) and past mm9's (mm39-only chr13): neither.
    assert genome_build.from_positions(["chr7", "chr13"], [149_800_500, 120_500_000]) is None


def test_mm39_positions_that_say_nothing():
    # mm10's chr7 runs longer than mm39's: an mm10 set past mm39's end, nothing past mm10's.
    assert genome_build.from_positions(["chr7", "chr8"], [145_200_000, 129_000_000]) is None
    # Past mm39's own end on chr8 and past mm9's on chr4: neither other build.
    assert genome_build.from_positions(["chr8", "chr4"], [130_200_000, 156_000_000]) is None
    # A human-only chromosome: not mouse.
    assert genome_build.from_positions(["chr8", "chr21"], [ITGB1_MM39, 1_000]) is None


def test_mm39_is_a_build_scview_has(tmp_path):
    from scview.api.v1 import atac_motifs
    from scview.core import genome_seq

    for name in ("mm39", "GRCm39", "GRCm39-2024-A", "refdata-cellranger-arc-GRCm39-2024-A"):
        assert gene_annotation.canonical_genome(name) == "mm39"
    # GC, motifs and links read the 2bit through this; gene bodies and TSS from GENCODE vM33
    assert genome_seq.file_path(tmp_path, "GRCm39-2024-A").name == "mm39.2bit"
    assert atac_motifs._genome("GRCm39-2024-A") == "mm39"
    assert gene_annotation.source_of("GRCm39") == "GENCODE vM33 basic"
    assert gene_annotation.canonical_genome("GRCm38") == "mm10"
    st = genome_seq.status(tmp_path, "mm39")
    assert st == {"genome": "mm39", "source": "UCSC mm39", "size": 714_181_470, "downloaded": False,
                  "url": "https://hgdownload.soe.ucsc.edu/goldenPath/mm39/bigZips/mm39.2bit"}
    # without the table on disk and fetching off: a reason about the file, not a refusal
    choice = atac.choose_genes(mf.as_anndata(), "mm39", tmp_path, auto_download=False)
    assert choice.names == [] and "ANNOTATION_AUTO_DOWNLOAD" in choice.reason


def test_tenx_h5_naming_grcm39_is_recorded_without_a_note(tmp_path):
    h5 = mf.write_tenx_h5(tmp_path / "e18_filtered_feature_bc_matrix.h5", genome="GRCm39-2024-A", build="mm39")
    m, sid = _session(tmp_path, h5)
    did = m._build_dataset(sid)
    a = _dataset(m, did)
    with _store(m, did) as s:
        assert s.meta["genome"] == "mm39" and "genome_note" not in s.meta
    info = atac_import.read_info(a)
    assert info["genome"] == "mm39" and "genome_note" not in info
    assert "genome_note" not in atac_import.dataset_state(a, m.data_dir, did)


def test_a_dataset_v076_recorded_as_mm39_works_without_reimport(tmp_path):
    """v0.76 refused mm39: it recorded genome "mm39" with the refusal as genome_note in the ATAC
    record, the store and the fragments state. The note is now worked out again from the genome."""
    old = ("This ATAC input is on mm39 (GRCm39), and scView has gene positions and genome sequence for "
           "GRCh38 and mm10 only, so gene activity, ... are off.")
    a = mf.as_anndata()
    atac_import.write_info(a, {"n_peaks": 1, "genome": "mm39", "genome_note": old, "computed": ["peak counts"]})
    store = atac.store_path(tmp_path, "d")
    modality_store.write(store, {atac.MATRIX: sp.csr_matrix(np.ones((a.n_obs, 1), dtype=np.float32))},
                         list(a.obs_names), ["chr8:1-500"], atac_import.feature_attrs(["chr8:1-500"]),
                         {"genome": "mm39", "genome_note": old})
    assert "genome_note" not in atac_import.dataset_state(a, tmp_path, "d")
    assert "genome_note" not in atac.summary(a)["info"]  # what the Accessibility tab shows
    lines, _ = atac.grounding_lines(a)
    assert not any("mm39 (GRCm39)" in line for line in lines)
    af._write_state(tmp_path, "d", {"genome": "mm39", "genome_note": old})
    assert af.status(tmp_path, "d")["genome_note"] is None
    # an hg19 record keeps its refusal, in today's words
    atac_import.write_info(a, {"n_peaks": 1, "genome": "hg19", "genome_note": "old words"})
    assert atac_import.dataset_state(a, tmp_path, "d")["genome_note"] == genome_build.HG19_REFUSAL
    assert atac.summary(a)["info"]["genome_note"] == genome_build.HG19_REFUSAL


def test_an_assumed_mm10_with_peaks_past_its_ends_becomes_mm39():
    itgb1 = f"chr8:{ITGB1_MM39}-{ITGB1_MM39 + 500}"
    mm39 = atac_import.check_build(_plan(["chr1:100-600", itgb1], genome="mm10"))
    assert mm39.genome == "mm39" and not mm39.genome_assumed and mm39.genome_note is None
    assert atac_import.check_build(_plan([itgb1], genome=None)).genome == "mm39"
    named = atac_import.check_build(_plan([itgb1], genome="mm10", assumed=False))
    assert named.genome == "mm10" and named.genome_note is None
    human = atac_import.check_build(_plan([itgb1]))  # an assumed human genome stays human
    assert human.genome == "GRCh38" and human.genome_note is None


def test_fragments_header_naming_grcm39(tmp_path):
    state = _attach(tmp_path, _mouse_lines(), header=MM39_HEADER, species="mouse")
    assert state["genome"] == "mm39" and "genome_note" not in state
    st = af.status(tmp_path, "d")
    assert st["genome"] == "mm39" and st["genome_note"] is None
    with pytest.raises(af.FragmentsError, match="genes are human"):
        _attach(tmp_path, _mouse_lines(), header=MM39_HEADER, species="human")


def test_fragments_index_reaching_past_mm10_ends_on_mm39(tmp_path):
    # In Itgb1's body: the index only shows a 16 kb window's start, so past mm10's end by more than that.
    body = ITGB1_MM39 + 20_000
    late = [("chr8", body + i * 10, body + 50 + i * 10, "AAAC-1", 1) for i in range(5)]
    state = _attach(tmp_path, _mouse_lines(late))
    assert state["genome"] == "mm39" and "genome_note" not in state


def _seed_mm39(tmp_path):
    gtf = tmp_path / "vM33-like.gtf"
    gtf.write_text(mf.gtf_text("mm39"))
    gene_annotation.clear_cache()
    gene_annotation.seed_from(gtf, tmp_path, "GRCm39-2024-A")


def test_tss_table_on_mm39(tmp_path):
    _seed_mm39(tmp_path)
    try:
        t = af.tss_table("mm39", tmp_path, auto_download=False)
    finally:
        gene_annotation.clear_cache()
    assert set(t["chrom"]) == {"chr8", "chr17"}
    assert all(tss > genome_build.MM10_SIZES[c[3:]] for c, tss in zip(t["chrom"], t["tss"]))  # past mm10's ends


def test_gene_activity_on_mm39_peaks_past_mm10_ends(tmp_path):
    """The mm39 fixture: genes and peaks past mm10's chromosome ends, mouse ids; gene activity
    finds every gene's peaks on mm39's positions."""
    _g, p, barcodes, table = mf.matrices(n_per=20, build="mm39")
    _seed_mm39(tmp_path)
    try:
        path = atac.store_path(tmp_path, "d")
        modality_store.write(path, {atac.MATRIX: p}, barcodes, table["name"].tolist(),
                             atac_import.feature_attrs(table["name"].tolist()), {"genome": "mm39"})
        genes = mf.genes_of("mm39")
        a = ad.AnnData(X=sp.csr_matrix(_g), obs=pd.DataFrame(index=barcodes),
                       var=pd.DataFrame({"gene_ids": [x[1] for x in genes]}, index=[x[0] for x in genes]))
        chosen = atac.choose_genes(a, "GRCm39-2024-A", tmp_path, auto_download=False)
        assert chosen.genome == "mm39" and chosen.source == "GENCODE vM33 basic" and len(chosen.names) == 4
        with modality_store.ModalityStore(path) as store:
            summary, _totals = atac.write_activity(store, chosen, tmp_path, "d")
        assert summary["n_genes_with_peaks"] == 4
    finally:
        gene_annotation.clear_cache()


# -- other species and builds --------------------------------------------------

@pytest.mark.parametrize("name,label", [
    ("T2T-CHM13v2.0", "T2T-CHM13"), ("hs1", "T2T-CHM13"), ("chm13", "T2T-CHM13"), ("mRatBN7.2", "rat"),
    ("rn6", "rat"), ("Rnor_6.0", "rat"), ("GRCz11", "zebrafish"), ("danRer11", "zebrafish"), ("dm6", "fly"),
    ("BDGP6.46", "fly"), ("WBcel235", "worm"), ("ce11", "worm"), ("GRCg7b", "chicken"), ("galGal6", "chicken"),
    ("Mmul_10", "macaque"), ("rheMac10", "macaque"), ("Sscrofa11.1", "pig"), ("ARS-UCD1.2", "cow"),
    ("canFam6", "dog"), ("R64-1-1", "yeast"), ("sacCer3", "yeast"), ("TAIR10", "Arabidopsis"),
    ("refdata-cellranger-arc-mRatBN7.2-2024-A", "rat"),
])
def test_other_builds_by_name(name, label):
    assert label in genome_build.other_build(name)
    sentence = genome_build.unsupported(name)
    assert label in sentence and "GRCh38 (human) and mm10 or mm39 (mouse)" in sentence
    assert "RNA analysis" in sentence and "work as usual" in sentence
    with pytest.raises(gene_annotation.AnnotationError) as e:
        gene_annotation.canonical_genome(name)
    assert str(e.value) == sentence


@pytest.mark.parametrize("name", ["GRCh38", "GRCh38-2020-A", "mm10", "GRCm39-2024-A", "hg19", "mm9", "", None,
                                  "pbmc_granulocyte_sorted_10k", "hs10", "dm60", "rn10"])
def test_not_other_builds(name):
    assert genome_build.other_build(name) is None


def test_nothing_named_and_unknown_names_say_so():
    assert "does not say which genome" in genome_build.unsupported(None)
    assert "does not say which genome" in genome_build.unsupported("nan")
    assert "'xenopus', a reference scView does not know" in genome_build.unsupported("xenopus")
    assert genome_build.note(None) is None  # nothing recorded: the steps say why on their own
    assert genome_build.note("other", "stored words") == "stored words"
    assert genome_build.note("GRCh38", "stored words") is None


@pytest.mark.parametrize("names,word", [
    (["chr1", "chr2L", "chr3R", "chrX"], "fly"), (["chr1", "chrZ", "chrW"], "bird"),
    (["chrI", "chrII", "chrIII", "chrX"], "worm or yeast"), (["1", "2", "25"], "chr25"),
])
def test_other_by_chromosome_names(names, word):
    assert word in genome_build.other_from_names(names)


@pytest.mark.parametrize("names", [["chr1", "chr22", "chrX", "chrY", "chrM"], ["1", "19", "X", "MT"],
                                   ["chr1", "KI270728.1", "chrUn_GL000220v1", "chrEBV"], []])
def test_human_and_mouse_names_are_not_other(names):
    assert genome_build.other_from_names(names) is None


def test_other_by_positions_past_every_human_and_mouse_end():
    # T2T-CHM13's chr9 is 150.6 Mb (GRCh38 138.4, hg19 141.2); a rat chr1 is 260 Mb.
    assert "chr9 reaches 150.0 Mb" in genome_build.other_from_positions(["chr9"], [150_000_000])
    assert "T2T-CHM13" in genome_build.other_from_positions(["chr1"], [260_000_000])
    # within some human or mouse build: not other (hg19's chr9, mm10's chrY)
    assert genome_build.other_from_positions(["chr9", "chrY"], [141_000_000, 90_000_000]) is None
    assert genome_build.other_from_positions(["chrM", "KI270728.1"], [16_000, 300_000_000]) is None


def test_import_records_other_species_with_one_note():
    zebrafish = atac_import.check_build(_plan(["chr1:100-600", "chr25:100-600"], genome=None))
    assert zebrafish.genome == genome_build.OTHER and "chr25" in zebrafish.genome_note
    assert "RNA analysis" in zebrafish.genome_note
    chm13 = atac_import.check_build(_plan(["chr1:100-600", "chr9:150000000-150000500"]))  # assumed GRCh38
    assert chm13.genome == genome_build.OTHER and "T2T-CHM13" in chm13.genome_note
    rat = atac_import.check_build(_plan(["chr1:100-600"], genome="mRatBN7.2", assumed=False))
    assert rat.genome == "mRatBN7.2" and "a rat genome" in rat.genome_note
    human = atac_import.check_build(_plan(["chr1:100-600", "chr22:100-600"]))
    assert human.genome == "GRCh38" and human.genome_note is None
    with pytest.raises(gene_annotation.AnnotationError, match="another species or build"):
        gene_annotation.canonical_genome(genome_build.OTHER)


def test_fragments_of_another_species(tmp_path):
    fish = sorted([(c, i * 10, i * 10 + 50, "AAAC-1", 1) for c in ("chr1", "chr25") for i in range(200)],
                  key=lambda x: (x[0], x[1]))
    state = _attach(tmp_path, fish, species="human")  # kept: the note says why the genome steps are off
    assert state["genome"] == genome_build.OTHER and "chr25" in state["genome_note"]
    assert af.status(tmp_path, "d")["genome_note"] == state["genome_note"]
    rat = "# reference_path=/refs/refdata-cellranger-arc-mRatBN7.2-2024-A\n"
    state = _attach(tmp_path, _lines(), header=rat)
    assert state["genome"] == genome_build.OTHER and "a rat genome" in state["genome_note"]
    # Fragments named on a build scView has are not second-guessed by another word in the path.
    t2t_dir = "# reference_path=/t2t_project/refdata-cellranger-arc-GRCh38-2020-A\n"
    assert _attach(tmp_path, _lines(), header=t2t_dir)["genome"] == "GRCh38"
