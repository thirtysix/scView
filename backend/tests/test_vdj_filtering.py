"""Clone sizes recomputed by the filtering step itself (core/vdj.py
``recompute_after_filter`` via ``qc_filter.after_filter``): no stale sizes, no
Recompute prompt, recorded in the filtering's provenance and replayed.

The contig fixture (tests/test_vdj.py) gives TCR1 = cells 0, 1, 2 (size 3),
TCR2 = cells 3, 4 (size 2), BCR1 = cell 9 (size 1); cells 5-8 have no
clonotype. The filter below drops cells 2 and 3 as doublets: TCR1 keeps 0 and 1
(size 2), TCR2 keeps half of itself, cell 4 (size 1: its bin moves from 2-5
cells to 1 cell)."""

from __future__ import annotations

import subprocess
import sys

import anndata as ad
import numpy as np
import pandas as pd

from scview.core import provenance, replay_script, vdj
from scview.core.pipeline import PipelineParams, _run_filtering, run_pipeline
from tests.test_vdj import b_rows, cells_adata, t_rows, write_csv

KEPT = [0, 1, 4, 5, 6, 7, 8, 9]


def _attached(tmp_path):
    path = write_csv(tmp_path / "all_contigs.csv", t_rows() + b_rows())
    a = cells_adata()
    vdj.attach(a, path, file_name=path.name, sha256="abc")
    a.obs["predicted_doublet"] = np.isin(np.arange(a.n_obs), [2, 3])
    return a


def _params() -> PipelineParams:
    return PipelineParams(qc_threshold_mode="fixed", min_genes=0, min_cells=0, drop_doublets=True)


def test_filtering_recomputes_sizes_and_expansion_by_hand(tmp_path):
    a = _attached(tmp_path)
    assert a.obs[vdj.SIZE_COL].tolist()[:5] == [3, 3, 3, 2, 2]
    out, found = _run_filtering(a, _params())
    assert out.obs_names.tolist() == [a.obs_names[i] for i in KEPT]
    size = out.obs[vdj.SIZE_COL].to_numpy()
    assert size[:3].tolist() == [2, 2, 1] and np.isnan(size[3:7]).all() and size[7] == 1
    assert out.obs[vdj.EXPANSION_COL].astype(str).tolist() == (
        ["2-5 cells", "2-5 cells", "1 cell"] + ["no clonotype"] * 4 + ["1 cell"])
    assert out.obs[vdj.CLONOTYPE_COL].astype(object).tolist()[:3] == ["TCR1", "TCR1", "TCR2"]
    # The same as counting the kept cells of each clonotype with pandas.
    clon = out.obs[vdj.CLONOTYPE_COL].astype(object)
    expected = clon.map(clon.value_counts()).astype(float).to_numpy()
    np.testing.assert_array_equal(np.nan_to_num(size, nan=-1), np.nan_to_num(expected, nan=-1))
    assert found["vdj_clonotypes"] == {"same_v_gene": False, "n_clonotypes": 3, "n_cells_in_clonotypes": 4}
    s = vdj.summary(out)
    assert s["stale"] is False and s["stale_note"] is None
    assert s["n_cells_in_clonotypes"] == 4 and s["expansion"]["2-5 cells"] == {"clonotypes": 1, "cells": 2}


def test_the_stored_definition_is_kept(tmp_path):
    a = _attached(tmp_path)
    vdj.set_definition(a, same_v_gene=True)
    out, found = _run_filtering(a, _params())
    assert found["vdj_clonotypes"]["same_v_gene"] is True and vdj.stored_info(out)["same_v_gene"] is True


def test_nothing_removed_or_no_contigs_records_nothing(tmp_path):
    a = _attached(tmp_path)
    a.obs["predicted_doublet"] = False
    _, found = _run_filtering(a, _params())
    assert "vdj_clonotypes" not in found
    plain = cells_adata()
    plain.obs["predicted_doublet"] = np.isin(np.arange(plain.n_obs), [2, 3])
    out, found = _run_filtering(plain, _params())
    assert out.n_obs == 8 and "vdj_clonotypes" not in found


def test_data_filtered_before_the_change_still_shows_the_prompt(tmp_path):
    a = _attached(tmp_path)
    old = a[KEPT].copy()  # cells removed without the step's recompute
    assert vdj.summary(old)["stale"] is True


def test_recorded_in_the_filtering_entry_and_replayed_exactly(tmp_path):
    a = _attached(tmp_path)
    inp = a.copy()
    out, res = run_pipeline(a, ["filtering"], _params())
    assert res.errors == {}
    entry = [h for h in provenance.read_provenance(out)["history"] if h["step"] == "filtering"][-1]
    assert entry["params"]["vdj_clonotypes"]["n_clonotypes"] == 3
    assert [d["status"] for d in replay_script.describe(out)] == ["exact", "exact"]
    script = replay_script.build_script(out, dataset_id="t", dataset_name="test")
    assert "redefine(adata, same_v_gene=False)" in script

    src, path, dst = tmp_path / "input.h5ad", tmp_path / "replay.py", tmp_path / "replayed.h5ad"
    inp.write_h5ad(src)  # already attached: the script starts from no V(D)J columns
    path.write_text(script)
    proc = subprocess.run([sys.executable, str(path), str(src), str(dst)], cwd=tmp_path,
                          capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr[-3000:]
    rep = ad.read_h5ad(dst)
    assert rep.obs_names.tolist() == out.obs_names.tolist()
    for col in vdj.our_columns(out):
        assert rep.obs[col].astype(str).tolist() == out.obs[col].astype(str).tolist(), col
    pd.testing.assert_frame_equal(vdj.read_table(rep), vdj.read_table(out))
