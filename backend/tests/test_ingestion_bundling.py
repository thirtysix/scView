"""Tests for the ingestion bundler (stage [2] BUNDLE).

Synthesises tiny real files so detection + role assignment + grouping are
exercised end-to-end, then asserts unit grouping, completeness and merge detection.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import anndata as ad
import h5py
import numpy as np

from scview.core.ingestion import (
    FileRole,
    UnitFormat,
    build_bundle,
)


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _mtx(p: Path) -> Path:
    p.write_text(
        "%%MatrixMarket matrix coordinate integer general\n%\n3 2 3\n1 1 4\n2 1 1\n3 2 7\n"
    )
    return p


def _barcodes(p: Path) -> Path:
    p.write_text("AAACCTGAGAAACCAT-1\nAAACCTGAGAAACGGC-1\nAAACCTGAGAAAGTGG-1\n")
    return p


def _features(p: Path) -> Path:
    p.write_text(
        "ENSG00000243485\tMIR1302-2HG\tGene Expression\n"
        "ENSG00000237613\tFAM138A\tGene Expression\n"
    )
    return p


def _h5ad(p: Path) -> Path:
    ad.AnnData(X=np.zeros((5, 3), dtype="float32")).write_h5ad(p)
    return p


def _dense_csv(p: Path) -> Path:
    p.write_text("cell,GeneA,GeneB,GeneC,GeneD\ncell_0,0,1,2,3\ncell_1,4,5,6,7\n")
    return p


def _tenx_h5(p: Path) -> Path:
    with h5py.File(p, "w") as f:
        g = f.create_group("matrix")
        g.create_dataset("data", data=np.array([1, 2, 3]))
        g.create_dataset("indptr", data=np.array([0, 1, 3]))
    return p


def _mex_triplet(d: Path, prefix: str = "") -> list[Path]:
    return [
        _mtx(d / f"{prefix}matrix.mtx"),
        _barcodes(d / f"{prefix}barcodes.tsv"),
        _features(d / f"{prefix}features.tsv"),
    ]


# ---------------------------------------------------------------------------
# 10x MEX units
# ---------------------------------------------------------------------------


def test_complete_mex_triplet(tmp_path):
    b = build_bundle(_mex_triplet(tmp_path))
    assert len(b.units) == 1
    u = b.units[0]
    assert u.format == UnitFormat.tenx_mex
    assert u.complete is True
    assert u.missing_roles == []
    assert {f.role for f in u.files} == {FileRole.matrix, FileRole.barcodes, FileRole.features}
    assert b.complete is True
    assert b.is_merge is False


def test_lone_matrix_is_incomplete(tmp_path):
    b = build_bundle([_mtx(tmp_path / "matrix.mtx")])
    assert len(b.units) == 1
    u = b.units[0]
    assert u.format == UnitFormat.tenx_mex
    assert u.complete is False
    assert FileRole.barcodes in u.missing_roles
    assert FileRole.features in u.missing_roles
    assert b.complete is False


def test_mex_missing_features(tmp_path):
    b = build_bundle([_mtx(tmp_path / "matrix.mtx"), _barcodes(tmp_path / "barcodes.tsv")])
    u = b.units[0]
    assert u.missing_roles == [FileRole.features]
    assert u.complete is False


def test_geo_prefixed_multisample_grouping(tmp_path):
    paths = _mex_triplet(tmp_path, "GSM4711_") + _mex_triplet(tmp_path, "GSM4712_")
    b = build_bundle(paths)
    assert len(b.units) == 2
    assert all(u.complete for u in b.units)
    assert all(u.format == UnitFormat.tenx_mex for u in b.units)
    assert {u.label for u in b.units} == {"gsm4711", "gsm4712"}
    assert b.is_merge is True
    assert b.complete is True


def test_duplicate_role_flagged(tmp_path):
    # Two matrix files that resolve to the same sample prefix ("s1") and role.
    paths = _mex_triplet(tmp_path, "s1_")
    gz = tmp_path / "s1_matrix.mtx.gz"
    with gzip.open(gz, "wt") as f:
        f.write("%%MatrixMarket matrix coordinate integer general\n%\n3 2 1\n1 1 4\n")
    paths.append(gz)
    b = build_bundle(paths)
    assert len(b.units) == 1
    u = b.units[0]
    assert any("more than one matrix" in i.lower() for i in u.issues)
    assert u.complete is False


# ---------------------------------------------------------------------------
# Single-file formats
# ---------------------------------------------------------------------------


def test_single_h5ad_unit(tmp_path):
    b = build_bundle([_h5ad(tmp_path / "data.h5ad")])
    assert len(b.units) == 1
    assert b.units[0].format == UnitFormat.anndata
    assert b.units[0].complete is True
    assert b.is_merge is False


def test_dense_csv_unit(tmp_path):
    b = build_bundle([_dense_csv(tmp_path / "expr.csv")])
    assert b.units[0].format == UnitFormat.dense_table
    assert b.units[0].complete is True


def test_two_h5ads_is_merge(tmp_path):
    b = build_bundle([_h5ad(tmp_path / "a.h5ad"), _h5ad(tmp_path / "b.h5ad")])
    assert len(b.units) == 2
    assert b.is_merge is True
    assert b.complete is True


def test_mixed_h5_and_mex(tmp_path):
    sub = tmp_path / "sampleB"
    sub.mkdir()
    paths = [_tenx_h5(tmp_path / "sampleA.h5")] + _mex_triplet(sub)
    b = build_bundle(paths)
    assert len(b.units) == 2
    fmts = {u.format for u in b.units}
    assert fmts == {UnitFormat.tenx_h5, UnitFormat.tenx_mex}
    assert b.is_merge is True


# ---------------------------------------------------------------------------
# Unknown handling
# ---------------------------------------------------------------------------


def test_unknown_file_reported_not_loadable(tmp_path):
    junk = tmp_path / "mystery.bin"
    junk.write_bytes(bytes(range(256)))
    b = build_bundle([_h5ad(tmp_path / "data.h5ad"), junk])
    # one loadable unit; the junk file becomes a bundle-level issue
    loadable = [u for u in b.units if u.format != UnitFormat.unknown]
    assert len(loadable) == 1
    assert any("could not be recognised" in i.lower() for i in b.issues)
    assert b.is_merge is False
