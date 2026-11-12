"""hg19 vs GRCh38 on ATAC input (core/genome_build.py): told from a named reference, a fragments
header or coordinates past GRCh38's chromosome ends; recorded as hg19 and refused by every
genome-dependent step with one sentence."""

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
