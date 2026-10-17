"""CITE-seq protein modality (core/protein.py): split, names, controls, CLR."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from scview.core import protein
from scview.core.anndata_adaptor import AnnDataAdaptor


def _mixed(n_cells: int = 6, seed: int = 0) -> ad.AnnData:
    """Four genes, three antibodies (one an isotype control) and a CRISPR guide,
    as a 10x feature-barcoding matrix reads."""
    rng = np.random.default_rng(seed)
    names = ["CD3E", "MS4A1", "LYZ", "GNLY", "CD3_TotalSeqB", "CD19_TotalSeqB",
             "IgG1_control_TotalSeqB", "guide_A"]
    types = ["Gene Expression"] * 4 + ["Antibody Capture"] * 3 + ["CRISPR Guide Capture"]
    X = rng.poisson(5, size=(n_cells, len(names))).astype(np.float32)
    a = ad.AnnData(sp.csr_matrix(X), var=pd.DataFrame({"feature_types": types}, index=names))
    a.obs_names = [f"c{i}" for i in range(n_cells)]
    return a


def test_clr_is_seurats_per_cell_formula():
    x = np.array([[0, 10, 100], [5, 5, 5]], dtype=float)
    # Seurat: log1p(x / exp(sum(log1p(x[x > 0])) / length(x))), per cell.
    expected = np.vstack([np.log1p(r / np.exp(np.log1p(r).sum() / 3)) for r in x])
    assert np.allclose(protein.clr(x), expected, atol=1e-6)
    assert np.allclose(protein.clr(sp.csr_matrix(x)), expected, atol=1e-6)
    # A cell's ratios do not depend on its depth much: the same profile, 10x deeper.
    assert protein.clr(x[1:] * 10)[0].std() == pytest.approx(0, abs=1e-6)


def test_split_takes_antibodies_and_other_features_out_of_the_genes():
    a = _mixed()
    genes, split = protein.split_features(a)
    assert list(genes.var_names) == ["CD3E", "MS4A1", "LYZ", "GNLY"]
    counts = genes.obsm[protein.COUNTS_KEY]
    assert list(counts.columns) == ["CD3", "CD19", "IgG1_control"]
    assert np.array_equal(counts.to_numpy(), a.X[:, 4:7].toarray())
    assert list(genes.obsm[protein.CLR_KEY].index) == list(genes.obs_names)
    assert split.controls == ["IgG1_control"]
    assert split.original["CD3"] == "CD3_TotalSeqB"
    assert split.other_features == {"CRISPR Guide Capture": 1}
    assert genes.uns[protein.UNS_KEY]["n_proteins"] == 3
    assert protein.protein_names(genes) == ["CD3", "CD19", "IgG1_control"]
    assert protein.values(genes, "CD19", normalised=False).tolist() == a.X[:, 5].toarray().ravel().tolist()


def test_genes_only_data_is_left_alone():
    a = _mixed()[:, :4].copy()
    out, split = protein.split_features(a)
    assert out is a and split is None and not protein.has_protein(out)
    plain = ad.AnnData(np.ones((3, 2), dtype=np.float32))
    assert protein.split_features(plain) == (plain, None)


def test_a_totalvi_protein_matrix_is_adopted_if_it_holds_counts():
    a = _mixed()[:, :4].copy()
    a.obsm["protein_expression"] = pd.DataFrame(
        {"CD4_TotalSeqA": [1, 0, 3, 4, 5, 6], "Mouse_IgG2a_isotype": [0, 1, 0, 1, 0, 1]}, index=a.obs_names)
    out, split = protein.split_features(a)
    assert protein.protein_names(out) == ["CD4", "Mouse_IgG2a_isotype"]
    assert split.controls == ["Mouse_IgG2a_isotype"]
    b = _mixed()[:, :4].copy()
    b.obsm["protein_expression"] = pd.DataFrame({"CD4": np.linspace(-1, 1, 6)}, index=b.obs_names)
    assert protein.split_features(b)[1] is None and not protein.has_protein(b)  # normalised: skipped


@pytest.mark.parametrize("name,control", [
    ("IgG1_control", True), ("Mouse IgG2a isotype Ctrl", True), ("Isotype_MOPC-21", True),
    ("rat_IgG2b", True), ("Hamster_IgG", True), ("CD3", False), ("IgD", False), ("IgM", False),
    ("HLA-DR", False), ("CD45RA", False),
])
def test_isotype_controls(name, control):
    assert protein.is_control(name) is control


def test_display_names():
    assert protein.display_name("CD8a_TotalSeqB") == "CD8a"
    assert protein.display_name("CD279-TotalSeq-C") == "CD279"
    assert protein.display_name("CD3.AbSeq") == "CD3"
    assert protein.display_name("CD4") == "CD4"


def test_the_adaptor_reads_proteins_as_layers(tmp_path):
    genes, _ = protein.split_features(_mixed())
    path = tmp_path / "cite.h5ad"
    genes.write_h5ad(path)
    a = AnnDataAdaptor(str(path))
    info = a.protein_info()
    assert info["names"] == ["CD3", "CD19", "IgG1_control"] and info["controls"] == ["IgG1_control"]
    clr = a.get_expression(["CD19", "CD3"], layer="protein")
    assert np.allclose(clr, genes.obsm["protein_clr"][["CD19", "CD3"]].to_numpy())
    counts = a.get_expression(["CD3"], layer="protein_counts")
    assert np.array_equal(counts[:, 0], genes.obsm["protein_counts"]["CD3"].to_numpy())
    # Genes and proteins stay apart; the general layer list does not offer proteins.
    assert a.get_expression(["CD3"]).shape[1] == 0
    assert "protein" not in {lyr["key"] for lyr in a.available_expression_layers()}
    groups = a.adata.obs.assign(g=["a", "b"] * 3)
    a.adata.obs = groups
    by = a.expression_by_group(["CD3", "LYZ"], "g", layer="protein")
    assert by["genes"] == ["CD3"] and by["groups"] == ["a", "b"]
    ad.AnnData(np.ones((2, 2), dtype=np.float32)).write_h5ad(tmp_path / "plain.h5ad")
    assert AnnDataAdaptor(str(tmp_path / "plain.h5ad")).protein_info() is None


def test_a_merge_keeps_the_shared_panel_and_renormalises():
    a, _ = protein.split_features(_mixed(seed=1))
    b, _ = protein.split_features(_mixed(seed=2))
    b.obsm[protein.COUNTS_KEY] = b.obsm[protein.COUNTS_KEY][["CD19", "CD3"]]  # a smaller panel
    assert protein.common_panel([a, b]) == ["CD3", "CD19"]
    assert protein.CLR_KEY not in a.obsm and list(b.obsm[protein.COUNTS_KEY].columns) == ["CD3", "CD19"]
    merged = ad.concat([a, b], index_unique="-")
    protein.renormalise(merged)
    assert list(merged.obsm[protein.CLR_KEY].columns) == ["CD3", "CD19"]
    assert np.allclose(merged.obsm[protein.CLR_KEY].to_numpy(),
                       protein.clr(merged.obsm[protein.COUNTS_KEY].to_numpy()))
    # One dataset without antibodies: the modality goes from all of them.
    c, _ = protein.split_features(_mixed(seed=3))
    plain = _mixed()[:, :4].copy()
    assert protein.common_panel([c, plain]) == [] and not protein.has_protein(c)
