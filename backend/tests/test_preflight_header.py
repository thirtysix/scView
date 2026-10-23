"""Preflight header readers: shape, storage and sizes of a data file without its matrix."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from scview.core.preflight import header_from_adata, read_header, sniff_format


def _adata(n_obs=40, n_vars=30, dense=False, seed=0) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    counts = rng.poisson(0.3, size=(n_obs, n_vars)).astype(np.float32)
    X = counts if dense else sp.csr_matrix(counts)
    a = ad.AnnData(X=X)
    a.obs_names = [f"c{i}" for i in range(n_obs)]
    a.var_names = [f"g{j}" for j in range(n_vars)]
    a.obs["group"] = pd.Categorical(["a", "b"] * (n_obs // 2))
    return a


def _nnz(a: ad.AnnData) -> int:
    return int(a.X.nnz) if sp.issparse(a.X) else int(np.count_nonzero(a.X))


def test_h5ad_sparse_with_layers_raw_and_obsm(tmp_path):
    a = _adata()
    a.layers["counts"] = a.X.copy()
    a.raw = a.copy()
    a.obsm["X_pca"] = np.zeros((40, 5), dtype=np.float32)
    a.uns["note"] = np.arange(100, dtype=np.int64)
    p = tmp_path / "s.h5ad"
    a.write_h5ad(p)
    h = read_header(p)
    assert (h.format, h.n_cells, h.n_genes) == ("h5ad", 40, 30)
    x = h.x
    assert x.sparse and x.nnz == _nnz(a) and x.dtype == "float32"
    # values + column indices + row pointers, as scipy holds them
    assert x.nbytes == a.X.data.nbytes + a.X.indices.nbytes + a.X.indptr.nbytes
    assert h.matrix("layers/counts").nnz == x.nnz
    assert h.matrix("raw/X").n_genes == 30
    assert h.obsm == {"X_pca": [40, 5]}
    assert h.obsm_bytes == 40 * 5 * 4
    assert h.uns_bytes >= 800
    assert h.obs_columns == 1
    assert not h.backed and not h.estimated and not h.guess


def test_h5ad_dense(tmp_path):
    a = _adata(dense=True)
    p = tmp_path / "d.h5ad"
    a.write_h5ad(p, compression="gzip")
    h = read_header(p)
    assert not h.x.sparse and h.x.nnz == 40 * 30
    assert h.x.nbytes == a.X.nbytes  # 40 x 30 float32


def test_header_from_adata_matches_the_file(tmp_path):
    a = _adata()
    a.layers["counts"] = a.X.copy()
    p = tmp_path / "s.h5ad"
    a.write_h5ad(p)
    file_h, mem_h = read_header(p), header_from_adata(ad.read_h5ad(p))
    assert [m.to_dict() for m in file_h.matrices] == [m.to_dict() for m in mem_h.matrices]


def test_variable_genes_are_counted_and_an_old_style_var_dataset_is_read(tmp_path):
    a = _adata()
    a.var["highly_variable"] = np.arange(a.n_vars) < 7
    p = tmp_path / "hv.h5ad"
    a.write_h5ad(p)
    assert read_header(p).n_hvg == 7 == header_from_adata(a).n_hvg
    # Older h5ad files (Kang's) store var as one compound dataset, not a group of columns:
    # reading the variable genes from it once made the whole header unreadable.
    with h5py.File(p, "a") as f:
        del f["var"]
        f["var"] = np.array([(f"g{j}".encode(), 0.5) for j in range(a.n_vars)],
                            dtype=[("index", "S8"), ("gene.mean", "f8")])
    h = read_header(p)
    assert h.format == "h5ad" and h.x is not None and h.x.nnz == _nnz(a) and h.n_hvg is None


def _write_10x_h5(path: Path, counts: np.ndarray) -> None:
    """A CellRanger v3 .h5: genes x cells CSC in /matrix."""
    csc = sp.csc_matrix(counts.T.astype(np.int32))  # genes x cells
    n_genes, n_cells = csc.shape
    with h5py.File(path, "w") as f:
        g = f.create_group("matrix")
        g["data"], g["indices"], g["indptr"] = csc.data, csc.indices.astype(np.int64), csc.indptr
        g["shape"] = np.array([n_genes, n_cells], dtype=np.int32)
        g["barcodes"] = np.array([f"AAAC{i:04d}-1".encode() for i in range(n_cells)])
        ft = g.create_group("features")
        ft["id"] = np.array([f"ENSG{j:011d}".encode() for j in range(n_genes)])
        ft["name"] = np.array([f"G{j}".encode() for j in range(n_genes)])
        ft["feature_type"] = np.array([b"Gene Expression"] * n_genes)
        ft["genome"] = np.array([b"GRCh38"] * n_genes)


def test_10x_h5(tmp_path):
    counts = np.random.default_rng(1).poisson(0.4, size=(25, 18))
    p = tmp_path / "filtered_feature_bc_matrix.h5"
    _write_10x_h5(p, counts)
    h = read_header(p)
    assert (h.format, h.n_cells, h.n_genes) == ("10x_h5", 25, 18)
    assert h.x.nnz == int(np.count_nonzero(counts)) and h.x.dtype == "float32"


def test_mex_gz(tmp_path):
    p = tmp_path / "matrix.mtx.gz"
    with gzip.open(p, "wt") as fh:
        fh.write("%%MatrixMarket matrix coordinate integer general\n%metadata_json: {}\n")
        fh.write("33538 1222 2498225\n1 1 3\n")
    h = read_header(p)
    assert (h.format, h.n_cells, h.n_genes, h.x.nnz) == ("mex", 1222, 33538, 2498225)


def test_csv_small_file_is_counted_exactly(tmp_path):
    p = tmp_path / "t.csv"
    df = pd.DataFrame(np.arange(12).reshape(4, 3), index=["g1", "g2", "g3", "g4"],
                      columns=["c1", "c2", "c3"])
    df.to_csv(p)
    h = read_header(p)
    assert (h.format, h.n_genes, h.n_cells, h.estimated) == ("table", 4, 3, False)
    assert read_header(p, genes_in_rows=False).n_cells == 4


def test_large_tsv_rows_are_estimated(tmp_path, monkeypatch):
    import scview.core.preflight as pf

    monkeypatch.setattr(pf, "_TABLE_SAMPLE_BYTES", 4096)  # sample a few lines only
    rows = [f"gene{i}\t" + "\t".join(str((i * j) % 7) for j in range(50)) for i in range(2000)]
    p = tmp_path / "t.tsv"
    p.write_text("\t".join(f"c{j}" for j in range(50)) + "\n" + "\n".join(rows) + "\n")
    h = read_header(p)
    assert h.estimated and h.n_cells == 50
    assert abs(h.n_genes - 2000) / 2000 < 0.1
    assert "estimated" in h.note


def test_gzipped_csv_rows_are_estimated_through_the_compression(tmp_path, monkeypatch):
    import scview.core.preflight as pf

    monkeypatch.setattr(pf, "_TABLE_SAMPLE_BYTES", 16384)
    rng = np.random.default_rng(2)
    rows = [f"gene{i}," + ",".join(str(v) for v in rng.poisson(0.5, 80)) for i in range(5000)]
    p = tmp_path / "t.csv.gz"
    with gzip.open(p, "wt") as fh:
        fh.write(",".join(f"c{j}" for j in range(80)) + "\n" + "\n".join(rows) + "\n")
    h = read_header(p)
    assert h.estimated and h.n_cells == 80
    assert abs(h.n_genes - 5000) / 5000 < 0.25


def test_loom(tmp_path):
    rng = np.random.default_rng(3)
    m = rng.poisson(0.2, size=(30, 20)).astype(np.float32)  # genes x cells
    p = tmp_path / "x.loom"
    with h5py.File(p, "w") as f:
        f["matrix"] = m
        f.create_group("layers")["spliced"] = m
        f.create_group("row_attrs")["Gene"] = np.array([f"g{i}".encode() for i in range(30)])
        f.create_group("col_attrs")["CellID"] = np.array([f"c{i}".encode() for i in range(20)])
    h = read_header(p)
    assert (h.format, h.n_cells, h.n_genes) == ("loom", 20, 30)
    assert h.x.nnz == int(np.count_nonzero(m)) and not h.x.nnz_estimated
    assert h.matrix("layers/spliced").nnz == h.x.nnz


def test_loom_nnz_is_sampled_when_large(tmp_path):
    rng = np.random.default_rng(4)
    m = (rng.random((3000, 2000)) < 0.1).astype(np.float32)
    p = tmp_path / "big.loom"
    with h5py.File(p, "w") as f:
        f.create_dataset("matrix", data=m, chunks=(64, 64))
        f.create_group("row_attrs")
        f.create_group("col_attrs")
    h = read_header(p)
    assert h.x.nnz_estimated
    assert abs(h.x.nnz - int(m.sum())) / m.sum() < 0.1


@pytest.mark.parametrize("fmt", [2, 3])
def test_zarr_written_by_anndata(tmp_path, fmt):
    a = _adata()
    a.layers["counts"] = a.X.copy()
    a.obsm["X_umap"] = np.zeros((40, 2), dtype=np.float32)
    p = tmp_path / "s.zarr"
    with ad.settings.override(zarr_write_format=fmt):
        a.write_zarr(p)
    h = read_header(p)
    assert (h.format, h.n_cells, h.n_genes) == ("zarr", 40, 30)
    assert h.x.sparse and h.x.nnz == _nnz(a)
    assert h.matrix("layers/counts").nnz == h.x.nnz
    assert h.obsm["X_umap"] == [40, 2]


def test_zarr_v2_dense_by_hand(tmp_path):
    p = tmp_path / "d.zarr"
    (p / "X").mkdir(parents=True)
    (p / ".zgroup").write_text(json.dumps({"zarr_format": 2}))
    (p / "X" / ".zarray").write_text(json.dumps({"shape": [100, 7], "dtype": "<f8"}))
    h = read_header(p)
    assert (h.n_cells, h.n_genes, h.x.dtype, h.x.sparse) == (100, 7, "float64", False)


def test_rds_is_a_guess(tmp_path):
    p = tmp_path / "obj.rds"
    p.write_bytes(b"\x1f\x8b" + b"\0" * 1000)
    h = read_header(p)
    assert h.format == "rds" and h.guess and h.n_cells is None
    assert "guess" in h.note and h.file_bytes == 1002


def test_damaged_file_is_reported_not_raised(tmp_path):
    p = tmp_path / "bad.h5ad"
    p.write_bytes(b"\x89HDF\r\n\x1a\n" + b"garbage" * 10)
    h = read_header(p, fmt="h5ad")
    assert h.format == "unknown" and "could not be read" in h.note


def test_sniff(tmp_path):
    a = _adata()
    a.write_h5ad(tmp_path / "a.h5ad")
    _write_10x_h5(tmp_path / "b.h5", np.ones((3, 2)))
    (tmp_path / "c.csv").write_text("g,c1,c2\nx,1,2\n")
    assert sniff_format(tmp_path / "a.h5ad") == "h5ad"
    assert sniff_format(tmp_path / "b.h5") == "10x_h5"
    assert sniff_format(tmp_path / "c.csv") == "table"
