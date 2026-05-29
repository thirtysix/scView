"""Tests for the ingestion file-type detection layer.

Each test synthesises a tiny fixture of one format in tmp_path and asserts that
``detect_file`` classifies it correctly — including content-first behaviour for
renamed / gzipped files, which is the whole point of sniffing.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import anndata as ad
import h5py
import numpy as np

from scview.core.ingestion import FileKind, detect_file


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _write_h5ad(p: Path) -> Path:
    ad.AnnData(X=np.zeros((5, 3), dtype="float32")).write_h5ad(p)
    return p


def _write_tenx_h5(p: Path) -> Path:
    with h5py.File(p, "w") as f:
        g = f.create_group("matrix")
        g.create_dataset("data", data=np.array([1, 2, 3]))
        g.create_dataset("indices", data=np.array([0, 1, 2]))
        g.create_dataset("indptr", data=np.array([0, 1, 3]))
        g.create_dataset("shape", data=np.array([3, 2]))
        g.create_dataset("barcodes", data=np.array([b"AAACCTGAGAAACCAT-1", b"AAACCTGAGAAACGGC-1"]))
        feats = g.create_group("features")
        feats.create_dataset("id", data=np.array([b"ENSG00000160791"]))
    return p


def _write_loom(p: Path) -> Path:
    with h5py.File(p, "w") as f:
        f.create_dataset("matrix", data=np.zeros((3, 2)))
        f.create_group("row_attrs")
        f.create_group("col_attrs")
    return p


def _write_mtx(p: Path) -> Path:
    p.write_text(
        "%%MatrixMarket matrix coordinate integer general\n"
        "%\n"
        "3 2 3\n"
        "1 1 4\n2 1 1\n3 2 7\n"
    )
    return p


def _write_barcodes(p: Path) -> Path:
    p.write_text("AAACCTGAGAAACCAT-1\nAAACCTGAGAAACGGC-1\nAAACCTGAGAAAGTGG-1\n")
    return p


def _write_features(p: Path) -> Path:
    p.write_text(
        "ENSG00000243485\tMIR1302-2HG\tGene Expression\n"
        "ENSG00000237613\tFAM138A\tGene Expression\n"
        "ENSG00000186092\tOR4F5\tGene Expression\n"
    )
    return p


def _write_gene_list(p: Path) -> Path:
    p.write_text("CD3D\nCD8A\nIL7R\nCCR7\n")
    return p


def _write_dense_csv(p: Path) -> Path:
    p.write_text(
        "cell,GeneA,GeneB,GeneC,GeneD\n"
        "cell_0,0,1,2,3\n"
        "cell_1,4,5,6,7\n"
    )
    return p


def _gzip(src: Path) -> Path:
    dst = src.with_suffix(src.suffix + ".gz")
    with open(src, "rb") as fin, gzip.open(dst, "wb") as fout:
        fout.write(fin.read())
    return dst


# ---------------------------------------------------------------------------
# Single-file format detection
# ---------------------------------------------------------------------------


def test_detect_h5ad(tmp_path):
    r = detect_file(_write_h5ad(tmp_path / "data.h5ad"))
    assert r.kind == FileKind.anndata_h5ad
    assert r.format_family == "anndata"
    assert r.confidence >= 0.9


def test_detect_tenx_h5(tmp_path):
    r = detect_file(_write_tenx_h5(tmp_path / "filtered_feature_bc_matrix.h5"))
    assert r.kind == FileKind.tenx_h5
    assert r.format_family == "10x"


def test_detect_loom(tmp_path):
    r = detect_file(_write_loom(tmp_path / "data.loom"))
    assert r.kind == FileKind.loom


def test_detect_mtx(tmp_path):
    r = detect_file(_write_mtx(tmp_path / "matrix.mtx"))
    assert r.kind == FileKind.mtx_matrix
    assert r.format_family == "matrix_market"


def test_detect_barcodes(tmp_path):
    r = detect_file(_write_barcodes(tmp_path / "barcodes.tsv"))
    assert r.kind == FileKind.tenx_barcodes


def test_detect_features(tmp_path):
    r = detect_file(_write_features(tmp_path / "features.tsv"))
    assert r.kind == FileKind.tenx_features


def test_detect_gene_list_sidecar(tmp_path):
    r = detect_file(_write_gene_list(tmp_path / "genes.txt"))
    assert r.kind == FileKind.tenx_features


def test_detect_dense_csv(tmp_path):
    r = detect_file(_write_dense_csv(tmp_path / "expression.csv"))
    assert r.kind == FileKind.dense_table
    assert r.details["delimiter"] == "comma"
    assert r.details["n_columns"] == 5


def test_detect_rds_by_extension(tmp_path):
    p = tmp_path / "seurat.rds"
    p.write_bytes(b"\x1f\x8b\x08\x00fake-rds-payload")  # gzip-magic, like a real RDS
    r = detect_file(p)
    assert r.kind == FileKind.seurat_rds


def test_detect_zarr_dir(tmp_path):
    d = tmp_path / "data.zarr"
    d.mkdir()
    (d / ".zgroup").write_text('{"zarr_format": 2}')
    r = detect_file(d)
    assert r.kind == FileKind.zarr_dir


# ---------------------------------------------------------------------------
# Content-first behaviour (the reason we sniff)
# ---------------------------------------------------------------------------


def test_gzipped_barcodes_detected_by_content(tmp_path):
    gz = _gzip(_write_barcodes(tmp_path / "barcodes.tsv"))
    r = detect_file(gz)
    assert r.kind == FileKind.tenx_barcodes
    assert r.details["gzipped"] is True


def test_gzipped_features_detected_by_content(tmp_path):
    gz = _gzip(_write_features(tmp_path / "features.tsv"))
    r = detect_file(gz)
    assert r.kind == FileKind.tenx_features


def test_renamed_geo_barcodes_still_detected(tmp_path):
    # GEO prepends sample ids; content must win over the unhelpful name.
    p = tmp_path / "GSM1234567_sampleA_barcodes.tsv"
    _write_barcodes(p)
    r = detect_file(p)
    assert r.kind == FileKind.tenx_barcodes


def test_name_hint_rescues_unknown(tmp_path):
    # An empty-ish file whose content can't be classified but whose name hints.
    p = tmp_path / "barcodes.tsv"
    p.write_text("\n\n")
    r = detect_file(p)
    assert r.kind == FileKind.tenx_barcodes
    assert r.details.get("name_hint") == "applied"


# ---------------------------------------------------------------------------
# Negative / edge cases
# ---------------------------------------------------------------------------


def test_unknown_binary(tmp_path):
    p = tmp_path / "mystery.bin"
    p.write_bytes(bytes(range(256)))
    r = detect_file(p)
    assert r.kind == FileKind.unknown


def test_missing_path(tmp_path):
    r = detect_file(tmp_path / "does_not_exist.h5ad")
    assert r.kind == FileKind.unknown


def test_empty_text_file(tmp_path):
    p = tmp_path / "empty.csv"
    p.write_text("")
    r = detect_file(p)
    assert r.kind == FileKind.unknown
