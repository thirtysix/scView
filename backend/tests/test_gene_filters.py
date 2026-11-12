"""Tests for the ribosomal/mitochondrial query filter."""

from __future__ import annotations

import pytest

from scview.core.gene_filters import is_ribo_mito, is_unnamed, select_query_genes


class TestIsUnnamed:
    @pytest.mark.parametrize("gene", [
        "ENSG00000289963", "ENSG00000289963.2", "ENSMUSG00000064370",  # Ensembl ids
        "AC012345.1", "AL627309.5", "AP000233.2", "BX284668.6", "Z98884.1",  # clone (GenBank) ids
        "RP11-34P13.7", "CTD-2245F17.3", "XXbac-B461K10.4",  # old GENCODE clone names
    ])
    def test_ids_without_a_name(self, gene):
        assert is_unnamed(gene) is True

    @pytest.mark.parametrize("gene", [
        "CD3E", "LINC01857", "ADAM7-AS1", "MMADHC-DT", "FTH1P20", "HLA-DRB1", "MT-CO1",
        "Gm12345", "1700020I14Rik", "AC2", "RP2", "ENSG", "C14orf180",
    ])
    def test_names_stay(self, gene):
        assert is_unnamed(gene) is False


class TestIsRiboMito:
    @pytest.mark.parametrize("gene", [
        "RPL5", "RPS3", "RPLP0", "RPLP1", "RPSA", "RPL7A", "RPS4Y1", "RPL3L",  # cytoplasmic ribosomal
        "MRPL15", "MRPS12",                                                     # mito-ribosomal
        "MT-CO1", "MT-ND1", "MT-ATP6",                                          # mito-encoded
    ])
    def test_flags_ribosomal_and_mitochondrial(self, gene):
        assert is_ribo_mito(gene) is True

    @pytest.mark.parametrize("gene", [
        "RPS6KA1", "RPS6KB1", "RPS6KC1", "RPS6KL1",  # S6 KINASES, not ribosomal proteins
        "MT1A", "MT2A",                              # metallothioneins (no hyphen)
        "GATA4", "FOXL2", "INHA", "DLK1", "PECAM1",  # ordinary markers
        "MTOR", "MTHFR",                             # start with MT but not MT-
    ])
    def test_does_not_flag_lookalikes(self, gene):
        assert is_ribo_mito(gene) is False

    def test_case_insensitive_for_mouse_symbols(self):
        assert is_ribo_mito("Rpl5") and is_ribo_mito("mt-Co1") and is_ribo_mito("Mrps12")


class TestSelectQueryGenes:
    def test_drops_ribo_mito_then_takes_top_n(self):
        ranked = ["RPL5", "GATA4", "MT-CO1", "FOXL2", "RPS3", "INHA", "DLK1"]
        assert select_query_genes(ranked, 3) == ["GATA4", "FOXL2", "INHA"]

    def test_fills_to_n_from_real_biology_not_padded_by_artifacts(self):
        # A cluster whose top genes are mostly ribosomal must still yield n real genes.
        ranked = ["RPL5", "RPS3", "RPL7", "GATA4", "MT-CO1", "FOXL2", "INHA", "DLK1", "AMH"]
        assert select_query_genes(ranked, 3) == ["GATA4", "FOXL2", "INHA"]

    def test_opt_out_keeps_everything(self):
        ranked = ["RPL5", "GATA4", "MT-CO1"]
        assert select_query_genes(ranked, 3, exclude_ribo_mito=False) == ranked

    def test_returns_at_most_n(self):
        assert len(select_query_genes([f"GENE{i}" for i in range(50)], 10)) == 10

    def test_short_list_after_filter_is_not_padded(self):
        ranked = ["RPL5", "GATA4", "RPS3"]
        assert select_query_genes(ranked, 10) == ["GATA4"]
