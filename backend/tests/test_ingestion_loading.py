"""Tests for the ingestion loader (stage [4] LOAD).

Builds realistic tiny fixtures, runs them through build_bundle, then loads the
resulting unit into an AnnData and asserts shape, orientation, names and var
annotation. Covers the renamed/gzipped MEX staging path specifically.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pytest
import scipy.io
import scipy.sparse as sp

from scview.core.ingestion import IngestLoadError, build_bundle, load_unit


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _barcode(i: int) -> str:
    bases = "ACGT"
    tail = "".join(bases[(i >> (2 * k)) & 3] for k in range(8))
    return f"AAACCTGA{tail}-1"


def _mex(tmp: Path, prefix: str = "", *, n_genes: int = 3, n_cells: int = 2,
         ncols: int = 3, gz: bool = False) -> list[Path]:
    # matrix is genes x cells (10x convention)
    dense = np.arange(1, n_genes * n_cells + 1).reshape(n_genes, n_cells)
    mtx = tmp / f"{prefix}matrix.mtx"
    scipy.io.mmwrite(str(mtx), sp.csr_matrix(dense))
    bc = tmp / f"{prefix}barcodes.tsv"
    bc.write_text("".join(_barcode(i) + "\n" for i in range(n_cells)))
    ft = tmp / f"{prefix}features.tsv"
    if ncols == 3:
        ft.write_text("".join(f"ENSG{i:011d}\tGENE{i}\tGene Expression\n" for i in range(n_genes)))
    else:  # v2: id + symbol
        ft.write_text("".join(f"ENSG{i:011d}\tGENE{i}\n" for i in range(n_genes)))
    paths = [mtx, bc, ft]
    if gz:
        gzs = []
        for p in paths:
            g = p.with_suffix(p.suffix + ".gz")
            with open(p, "rb") as fi, gzip.open(g, "wb") as fo:
                fo.write(fi.read())
            p.unlink()
            gzs.append(g)
        return gzs
    return paths


def _h5ad(p: Path, n_obs: int = 5, n_var: int = 4) -> Path:
    a = ad.AnnData(X=np.arange(n_obs * n_var, dtype="float32").reshape(n_obs, n_var))
    a.obs_names = [f"cell{i}" for i in range(n_obs)]
    a.var_names = [f"Gene{i}" for i in range(n_var)]
    a.write_h5ad(p)
    return p


def _tenx_h5(p: Path, n_genes: int = 3, n_cells: int = 2) -> Path:
    dense = np.arange(1, n_genes * n_cells + 1).reshape(n_genes, n_cells)
    m = sp.csc_matrix(dense)
    with h5py.File(p, "w") as f:
        g = f.create_group("matrix")
        g.create_dataset("data", data=m.data.astype("int32"))
        g.create_dataset("indices", data=m.indices.astype("int64"))
        g.create_dataset("indptr", data=m.indptr.astype("int64"))
        g.create_dataset("shape", data=np.array([n_genes, n_cells], dtype="int32"))
        g.create_dataset("barcodes", data=np.array([_barcode(i).encode() for i in range(n_cells)]))
        ff = g.create_group("features")
        ff.create_dataset("id", data=np.array([f"ENSG{i}".encode() for i in range(n_genes)]))
        ff.create_dataset("name", data=np.array([f"GENE{i}".encode() for i in range(n_genes)]))
        ff.create_dataset("feature_type", data=np.array([b"Gene Expression"] * n_genes))
    return p


def _only_unit(paths):
    bundle = build_bundle(paths)
    assert len(bundle.units) == 1
    return bundle.units[0]


# ---------------------------------------------------------------------------
# 10x MEX
# ---------------------------------------------------------------------------


def test_load_mex_basic(tmp_path):
    adata = load_unit(_only_unit(_mex(tmp_path, n_genes=3, n_cells=2)))
    assert adata.shape == (2, 3)  # cells × genes
    assert list(adata.var_names) == ["GENE0", "GENE1", "GENE2"]
    assert "gene_ids" in adata.var.columns
    assert adata.obs_names[0].startswith("AAACCTGA")


def test_load_mex_renamed_and_gzipped(tmp_path):
    # GEO-prefixed + gzipped: proves the canonical-name staging works.
    adata = load_unit(_only_unit(_mex(tmp_path, "GSM999_", n_genes=4, n_cells=3, gz=True)))
    assert adata.shape == (3, 4)


def test_load_mex_v2_two_column_features(tmp_path):
    adata = load_unit(_only_unit(_mex(tmp_path, n_genes=3, n_cells=2, ncols=2)))
    assert adata.shape == (2, 3)
    assert list(adata.var_names) == ["GENE0", "GENE1", "GENE2"]


# ---------------------------------------------------------------------------
# Single-file formats
# ---------------------------------------------------------------------------


def test_load_h5ad(tmp_path):
    adata = load_unit(_only_unit([_h5ad(tmp_path / "data.h5ad", n_obs=5, n_var=4)]))
    assert adata.shape == (5, 4)
    assert list(adata.var_names) == ["Gene0", "Gene1", "Gene2", "Gene3"]


def test_load_tenx_h5(tmp_path):
    adata = load_unit(_only_unit([_tenx_h5(tmp_path / "f.h5", n_genes=3, n_cells=2)]))
    assert adata.shape == (2, 3)


def test_load_dense_genes_in_rows_default(tmp_path):
    # genes × cells (genes in rows) — the default orientation.
    p = tmp_path / "expr.csv"
    p.write_text(
        "gene,cellA,cellB,cellC,cellD\n"
        "GeneX,1,2,3,4\n"
        "GeneY,5,6,7,8\n"
        "GeneZ,9,10,11,12\n"
    )
    adata = load_unit(_only_unit([p]))
    assert adata.shape == (4, 3)  # 4 cells × 3 genes after transpose
    assert list(adata.obs_names) == ["cellA", "cellB", "cellC", "cellD"]
    assert list(adata.var_names) == ["GeneX", "GeneY", "GeneZ"]


def test_load_dense_cells_in_rows_option(tmp_path):
    p = tmp_path / "expr.csv"
    p.write_text(
        "cell,GeneA,GeneB,GeneC,GeneD\n"
        "cell0,1,2,3,4\n"
        "cell1,5,6,7,8\n"
        "cell2,9,10,11,12\n"
    )
    adata = load_unit(_only_unit([p]), options={"genes_in_rows": False})
    assert adata.shape == (3, 4)  # cells already in rows
    assert list(adata.var_names) == ["GeneA", "GeneB", "GeneC", "GeneD"]


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_incomplete_unit_raises(tmp_path):
    unit = _only_unit(_mex(tmp_path)[:1])  # lone matrix
    with pytest.raises(IngestLoadError):
        load_unit(unit)


def test_seurat_rds_delegated_error(tmp_path):
    p = tmp_path / "obj.rds"
    p.write_bytes(b"\x1f\x8b\x08\x00fake")
    with pytest.raises(IngestLoadError, match="converter"):
        load_unit(_only_unit([p]))


def test_loom_without_loompy_raises_friendly(tmp_path):
    try:
        import loompy  # noqa: F401
        pytest.skip("loompy installed; friendly-error path not exercised")
    except ImportError:
        pass
    # Build a minimal HDF5 that detection classifies as loom.
    p = tmp_path / "data.loom"
    with h5py.File(p, "w") as f:
        f.create_dataset("matrix", data=np.zeros((3, 2)))
        f.create_group("row_attrs")
        f.create_group("col_attrs")
    with pytest.raises(IngestLoadError, match="loompy"):
        load_unit(_only_unit([p]))
