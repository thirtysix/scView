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
])
def test_genome_names(name, expected):
    assert ga.canonical_genome(name) == expected


def test_other_genome_says_which():
    with pytest.raises(ga.AnnotationError, match="not 'hg19'"):
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
