"""hg19 vs GRCh38 and mm9 vs mm10 on ATAC input (core/genome_build.py): told from a named
reference, a fragments header or coordinates past the newer build's chromosome ends; recorded as
hg19 / mm9 and refused by every genome-dependent step with one sentence."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

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
    with pytest.raises(gene_annotation.AnnotationError, match="not 'xenopus'"):
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
