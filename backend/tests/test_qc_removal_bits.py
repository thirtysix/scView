"""The filtering step's removal record keeps one bit per criterion and cell: uint8 until
v0.62 (8 criteria at most), uint16 since. Old records still read; a 9th criterion is kept."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from scview.core import qc_filter
from scview.core.pipeline import PipelineParams

N = 40


def _adata() -> tuple[ad.AnnData, dict]:
    rng = np.random.default_rng(0)
    a = ad.AnnData(X=np.ones((N, 3), dtype=np.float32),
                   obs=pd.DataFrame({"cell_type": pd.Categorical(["A"] * 20 + ["B"] * 20)},
                                    index=[f"c{i}" for i in range(N)]))
    cells = {"total_counts": rng.uniform(500, 5000, N), "n_genes": rng.uniform(200, 2000, N),
             "pct_mt": rng.uniform(0, 20, N), "has_mt": True}
    return a, cells


def _record(removed: dict[str, np.ndarray]) -> tuple[ad.AnnData, dict]:
    a, cells = _adata()
    drop = np.logical_or.reduce(list(removed.values()))
    t = qc_filter.Thresholds.from_params(PipelineParams(qc_threshold_mode="fixed"))
    rec = qc_filter.removal_record(a, cells, removed, drop, t, None)
    kept = a[~drop].copy()
    kept.uns[qc_filter.RECORD_KEY] = rec
    return kept, rec


def _mask(*idx: int) -> np.ndarray:
    m = np.zeros(N, dtype=bool)
    m[list(idx)] = True
    return m


def test_an_old_uint8_record_reads_as_before():
    removed = {"min_genes": _mask(0, 1, 2), "max_pct_mt": _mask(2, 30), "doublets": _mask(31)}
    kept, rec = _record(removed)
    rec["criterion_bits"] = np.asarray(rec["criterion_bits"]).astype(np.uint8)  # as v0.62 wrote it
    assert np.array_equal(qc_filter.recorded_bits(kept, "max_pct_mt"), [False, False, True, True, False])
    assert np.array_equal(qc_filter.recorded_bits(kept, "doublets"), [False, False, False, False, True])
    assert not qc_filter.recorded_bits(kept, "max_genes").any()
    audit = qc_filter.recorded_audit(kept, label_column="cell_type")
    by = audit["overall"]["by_criterion"]
    assert {c: v for c, v in by.items() if v} == \
        {"min_genes": 3, "max_pct_mt": 2, "doublets": 1}


def test_new_records_are_uint16_and_a_ninth_criterion_survives_h5ad(tmp_path):
    names = [*qc_filter.CRITERIA, "extra_8", "extra_9"]
    assert len(names) == 9
    removed = {c: np.zeros(N, dtype=bool) for c in names}
    removed["min_genes"] = _mask(0)
    removed["extra_9"] = _mask(5, 6)  # bit 8: lost in a uint8
    kept, rec = _record(removed)
    assert np.asarray(rec["criterion_bits"]).dtype == np.uint16
    kept.write_h5ad(tmp_path / "f.h5ad")
    back = ad.read_h5ad(tmp_path / "f.h5ad")
    assert np.asarray(back.uns[qc_filter.RECORD_KEY]["criterion_bits"]).dtype == np.uint16
    assert np.array_equal(qc_filter.recorded_bits(back, "extra_9"), [False, True, True])
    assert np.array_equal(qc_filter.recorded_bits(back, "min_genes"), [True, False, False])
    assert np.array_equal(qc_filter.criterion_bits(back.uns[qc_filter.RECORD_KEY]), [1, 1 << 8, 1 << 8])


def test_more_criteria_than_bits_is_refused():
    removed = {f"c{i}": _mask(i) for i in range(qc_filter.MAX_CRITERIA + 1)}
    with pytest.raises(ValueError, match="at most"):
        _record(removed)
