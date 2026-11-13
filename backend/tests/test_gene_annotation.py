"""Gene positions from a GTF: 0-based half-open coordinates, TSS by strand, no network in tests."""

import gzip

import pytest

from scview.core import gene_annotation as ga

GTF = """##description: test
chr1\tHAVANA\tgene\t11869\t14409\t.\t+\t.\tgene_id "ENSG00000223972.5"; gene_type "transcribed"; gene_name "DDX11L1";
chr1\tHAVANA\ttranscript\t11869\t14409\t.\t+\t.\tgene_id "ENSG00000223972.5"; gene_name "DDX11L1";
chr1\tHAVANA\tgene\t14404\t29570\t.\t-\t.\tgene_id "ENSG00000227232.5"; gene_type "pseudo"; gene_name "WASH7P";
chrY\tHAVANA\tgene\t100\t200\t.\t+\t.\tgene_id "ENSG00000182378.14_PAR_Y"; gene_type "protein_coding"; gene_name "PLCXD1";
"""


@pytest.fixture(autouse=True)
def _fresh():
    ga.clear_cache()
    yield
    ga.clear_cache()


def _gtf(tmp_path):
    path = tmp_path / "t.gtf.gz"
    with gzip.open(path, "wt") as f:
        f.write(GTF)
    return path


def test_parse_keeps_genes_only_with_zero_based_coordinates(tmp_path):
    t = ga.parse_gtf(_gtf(tmp_path))
    assert t.gene_name.tolist() == ["DDX11L1", "WASH7P"]
    assert t.gene_id.tolist() == ["ENSG00000223972", "ENSG00000227232"]
    plus, minus = t.iloc[0], t.iloc[1]
    assert (plus.start, plus.end, plus.tss) == (11868, 14409, 11868)
    assert (minus.start, minus.end, minus.tss) == (14403, 29570, 29569)


@pytest.mark.parametrize("name,expected", [
    ("GRCh38", "GRCh38"), ("GRCh38-2020-A", "GRCh38"), ("hg38", "GRCh38"), ("mm10-2020-A", "mm10"), ("GRCm38", "mm10"),
    ("mm39", "mm39"), ("GRCm39", "mm39"), ("GRCm39-2024-A", "mm39"), ("refdata-cellranger-arc-GRCm39-2024-A", "mm39"),
    ("refdata-cellranger-arc-mm10-2020-A-2.0.0", "mm10"), ("refdata-gex-GRCh38-2024-A", "GRCh38"),
])
def test_genome_names(name, expected):
    assert ga.canonical_genome(name) == expected


def test_mm39_release_is_gencode_vm33():
    r = ga.RELEASES["mm39"]  # the annotation 10x built GRCm39-2024-A from
    assert r.source == "GENCODE vM33 basic" and "release_M33/gencode.vM33.basic.annotation.gtf.gz" in r.url
    assert len(r.sha256) == 64


def test_other_genome_says_which():
    with pytest.raises(ga.AnnotationError, match=r"on a zebrafish genome \(danRer11\).*RNA analysis .* work as usual"):
        ga.canonical_genome("danRer11")
    with pytest.raises(ga.AnnotationError, match="'Foo-1', a reference scView does not know.*mm10 or mm39"):
        ga.canonical_genome("Foo-1")
    with pytest.raises(ga.AnnotationError, match=r"on hg19 \(GRCh37\)"):  # core/genome_build.py
        ga.canonical_genome("hg19")


def test_seeded_table_loads_without_network(tmp_path):
    ga.seed_from(_gtf(tmp_path), tmp_path, "GRCh38")
    t = ga.load("GRCh38-2020-A", tmp_path, auto_download=False)
    assert list(t.columns) == ga.COLUMNS
    assert t.gene_name.tolist() == ["DDX11L1", "WASH7P"]
    assert ga.load("hg38", tmp_path, auto_download=False) is t


def test_download_off_says_so(tmp_path):
    with pytest.raises(ga.AnnotationError, match="ANNOTATION_AUTO_DOWNLOAD"):
        ga.load("mm10", tmp_path, auto_download=False)
