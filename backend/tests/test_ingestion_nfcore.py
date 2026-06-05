"""nf-core/scrnaseq output ingestion (light MVP).

nf-core/scrnaseq's `mtx_conversions/` emits standard AnnData h5ad files named
`<sample>_{raw,filtered,cellbender_filter}_matrix.h5ad` plus `combined_matrix.h5ad`.
These already load through the existing ingestion engine; we verify detection,
clean labelling, loading, and multi-sample merge detection.
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pytest

from scview.core.ingestion.bundling import _clean_unit_label, build_bundle
from scview.core.ingestion.detection import FileKind, detect_file
from scview.core.ingestion.loading import load_unit


def _write_h5ad(path: Path, n_cells: int = 20, n_genes: int = 10) -> Path:
    a = ad.AnnData(np.random.poisson(1.0, size=(n_cells, n_genes)).astype("float32"))
    a.var_names = [f"G{i}" for i in range(n_genes)]
    a.obs_names = [f"C{i}" for i in range(n_cells)]
    a.write_h5ad(path)
    return path


@pytest.mark.parametrize(
    "stem,expected",
    [
        ("GSM4711_filtered_matrix", "GSM4711"),
        ("SampleA_raw_matrix", "SampleA"),
        ("S1_cellbender_filter_matrix", "S1"),
        ("Donor3_unfiltered_matrix", "Donor3"),
        ("combined_matrix", "combined"),
        ("plain_dataset", "plain_dataset"),  # untouched
        ("matrix", "matrix"),  # too short to strip
    ],
)
def test_clean_unit_label(stem, expected):
    assert _clean_unit_label(stem) == expected


def test_nfcore_h5ad_detects_and_labels(tmp_path):
    p = _write_h5ad(tmp_path / "GSM4711_filtered_matrix.h5ad")
    assert detect_file(p).kind == FileKind.anndata_h5ad
    bundle = build_bundle([p])
    assert len(bundle.units) == 1
    assert bundle.units[0].label == "GSM4711"
    assert not bundle.is_merge
    loaded = load_unit(bundle.units[0], {})
    assert loaded.shape == (20, 10)


def test_nfcore_per_sample_h5ads_flagged_as_merge(tmp_path):
    paths = [
        _write_h5ad(tmp_path / "SampleA_filtered_matrix.h5ad"),
        _write_h5ad(tmp_path / "SampleB_filtered_matrix.h5ad"),
    ]
    bundle = build_bundle(paths)
    assert len(bundle.units) == 2
    assert {u.label for u in bundle.units} == {"SampleA", "SampleB"}
    assert bundle.is_merge is True
