"""Tests for the ingestion validator (stage [3] VALIDATE).

Builds bundles via build_bundle (so detection + bundling feed in realistically)
and asserts the structured IngestIssue output: friendly completeness messaging,
10x dimension agreement, transposition detection, and gzip integrity.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import anndata as ad
import numpy as np

from scview.core.ingestion import IssueSeverity, build_bundle, validate_bundle


# ---------------------------------------------------------------------------
# Fixture builders (dimension-controlled)
# ---------------------------------------------------------------------------


def _mtx(p: Path, rows: int, cols: int, nnz: int = 1) -> Path:
    p.write_text(
        f"%%MatrixMarket matrix coordinate integer general\n%\n{rows} {cols} {nnz}\n1 1 4\n"
    )
    return p


def _barcode(i: int) -> str:
    """A valid 16 bp ACGT barcode (no digits) varying by index."""
    bases = "ACGT"
    tail = "".join(bases[(i >> (2 * k)) & 3] for k in range(8))
    return f"AAACCTGA{tail}-1"


def _barcodes(p: Path, n: int) -> Path:
    p.write_text("".join(_barcode(i) + "\n" for i in range(n)))
    return p


def _features(p: Path, n: int) -> Path:
    p.write_text("".join(f"ENSG{i:011d}\tGENE{i}\tGene Expression\n" for i in range(n)))
    return p


def _h5ad(p: Path) -> Path:
    ad.AnnData(X=np.zeros((5, 3), dtype="float32")).write_h5ad(p)
    return p


def _codes(report) -> set[str]:
    return {i.code for i in report.issues}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_complete_matching_mex_is_clean(tmp_path):
    # 10x convention: matrix is features(rows) x barcodes(cols).
    _mtx(tmp_path / "matrix.mtx", rows=2, cols=3)
    _barcodes(tmp_path / "barcodes.tsv", n=3)
    _features(tmp_path / "features.tsv", n=2)
    report = validate_bundle(build_bundle(list(tmp_path.iterdir())))
    assert report.ok is True
    assert not any(i.severity == IssueSeverity.error for i in report.issues)


def test_incomplete_mex_emits_friendly_error(tmp_path):
    _mtx(tmp_path / "matrix.mtx", rows=2, cols=3)
    report = validate_bundle(build_bundle([tmp_path / "matrix.mtx"]))
    assert report.ok is False
    assert "mex_incomplete" in _codes(report)
    issue = next(i for i in report.issues if i.code == "mex_incomplete")
    assert "barcodes" in issue.suggestion.lower()
    assert issue.suggestion  # has actionable next step


def test_dimension_mismatch_is_error(tmp_path):
    _mtx(tmp_path / "matrix.mtx", rows=5, cols=7)  # neither matches 2 genes / 3 cells
    _barcodes(tmp_path / "barcodes.tsv", n=3)
    _features(tmp_path / "features.tsv", n=2)
    report = validate_bundle(build_bundle(list(tmp_path.iterdir())))
    assert report.ok is False
    assert "mtx_dimension_mismatch" in _codes(report)


def test_transposed_matrix_is_warning_not_error(tmp_path):
    # rows==barcodes, cols==features → transposed but recoverable.
    _mtx(tmp_path / "matrix.mtx", rows=3, cols=2)
    _barcodes(tmp_path / "barcodes.tsv", n=3)
    _features(tmp_path / "features.tsv", n=2)
    report = validate_bundle(build_bundle(list(tmp_path.iterdir())))
    assert "mtx_transposed" in _codes(report)
    assert report.ok is True  # warning, not blocking


def test_truncated_gzip_is_error(tmp_path):
    # valid barcodes + features, but a truncated gzip matrix
    _barcodes(tmp_path / "barcodes.tsv", n=3)
    _features(tmp_path / "features.tsv", n=2)
    bad = tmp_path / "matrix.mtx.gz"
    bad.write_bytes(b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x00\x03truncated-garbage")
    report = validate_bundle(build_bundle(list(tmp_path.iterdir())))
    assert report.ok is False
    assert "gzip_truncated" in _codes(report)


def test_gzipped_valid_mex_is_clean(tmp_path):
    # gzip integrity must not false-positive on healthy gzip files.
    _mtx(tmp_path / "m.tmp", rows=2, cols=3)
    with gzip.open(tmp_path / "matrix.mtx.gz", "wt") as f:
        f.write((tmp_path / "m.tmp").read_text())
    (tmp_path / "m.tmp").unlink()
    with gzip.open(tmp_path / "barcodes.tsv.gz", "wt") as f:
        f.write("".join(_barcode(i) + "\n" for i in range(3)))
    with gzip.open(tmp_path / "features.tsv.gz", "wt") as f:
        f.write("".join(f"ENSG{i:011d}\tGENE{i}\tGene Expression\n" for i in range(2)))
    report = validate_bundle(build_bundle(list(tmp_path.iterdir())))
    assert report.ok is True
    assert "gzip_truncated" not in _codes(report)


def test_single_h5ad_has_no_issues(tmp_path):
    report = validate_bundle(build_bundle([_h5ad(tmp_path / "data.h5ad")]))
    assert report.ok is True
    assert report.issues == []
