"""Tests for the ribosomal/mitochondrial query filter."""

from __future__ import annotations

import pytest

from scview.core.gene_filters import is_ribo_mito, select_query_genes


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
