"""A run's result is written beside the derived file and then swapped in.

Writing straight over the derived file meant a backend killed mid-write (out of
memory, which the run watchdog warns about) left a truncated file: the dataset
then could not be opened at all, not even as it was before the run.
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pytest

from scview.core import pipeline


def _counts() -> ad.AnnData:
    a = ad.AnnData(np.random.default_rng(0).poisson(1.0, (60, 20)).astype("float32"))
    a.var_names = [f"G{i}" for i in range(20)]
    return a


def test_a_failed_write_leaves_the_previous_file_whole(tmp_path, monkeypatch):
    out = tmp_path / "derived.h5ad"
    pipeline.run_pipeline(_counts(), ["qc_metrics"], output_path=str(out))
    before = out.read_bytes()

    def dies_half_way(self, filename, **_kw):
        Path(filename).write_bytes(b"\x89HDF half a file")
        raise MemoryError("killed mid-write")

    monkeypatch.setattr(ad.AnnData, "write_h5ad", dies_half_way)
    with pytest.raises(MemoryError):
        pipeline.run_pipeline(_counts(), ["qc_metrics", "normalization"], output_path=str(out))
    assert out.read_bytes() == before
    assert list(tmp_path.iterdir()) == [out]  # no .partial left behind


def test_the_streamed_run_saves_the_same_way(tmp_path):
    out = tmp_path / "derived.h5ad"
    events = list(pipeline.run_pipeline_streamed(_counts(), ["qc_metrics"], output_path=str(out)))
    assert events[-1][1]["output_path"] == str(out)
    assert "total_counts" in ad.read_h5ad(out).obs
    assert list(tmp_path.iterdir()) == [out]
