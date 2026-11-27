"""The QC metrics step counts on counts, never on a log-normalised or scaled X.

A public import often carries the authors' processing: X log-normalised, the counts in
``.raw`` or ``layers['counts']``. Run without ``reset_to_counts`` (the UI's "run the missing
steps", the MCP's full run), the step summed X as it was: "total counts" several times the
real ones, a filter preview that removed the wrong cells. Now it reads the counts beside X,
records which matrix it used, and writes nothing when there are no counts at all.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
import scipy.sparse as sp

from scview.core import chunked, large_run, pipeline, provenance, qc_filter, replay_script
from scview.core.pipeline import PipelineParams, run_pipeline

# Looked up, not imported: against a source without the rule these tests fail on what the
# step does, not at collection.
QC_NEEDS_COUNTS = getattr(pipeline, "QC_NEEDS_COUNTS", "QC metrics need raw counts")

API = "/api/v1"
QC_OBS = ["n_genes_by_counts", "total_counts", "total_counts_mt", "pct_counts_mt"]
N_MT = 8


def _counts(n: int = 700, g: int = 300, seed: int = 0) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    depth = rng.lognormal(0.0, 0.6, n)[:, None]
    X = rng.poisson(depth * rng.gamma(0.4, 1.0, g)[None, :] * 3).astype(np.float32)
    names = [f"MT-G{i}" if i < N_MT else f"GENE{i}" for i in range(g)]
    obs = pd.DataFrame({"sample": pd.Categorical(np.where(np.arange(n) % 2, "s1", "s2"))},
                       index=[f"cell{i}" for i in range(n)])
    return ad.AnnData(sp.csr_matrix(X), obs=obs, var=pd.DataFrame(index=names))


def _real(a: ad.AnnData) -> pd.DataFrame:
    """The QC columns of the counts themselves."""
    c = a.copy()
    c.var["mt"] = c.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(c, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)
    return c.obs[QC_OBS]


def _lognorm(where: str, scaled: bool = False, **kw) -> tuple[ad.AnnData, pd.DataFrame]:
    """(X log-normalised with the counts in ``where``: raw | layer | none, the real QC)."""
    a = _counts(**kw)
    real = _real(a)
    counts = a.X.copy()
    if where == "raw":
        a.raw = a.copy()
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    if scaled:
        a.X = a.X.toarray()
        sc.pp.scale(a, max_value=10)
    if where == "layer":
        a.layers["counts"] = counts
    return a, real


def _last(a: ad.AnnData, step: str = "qc_metrics") -> dict:
    return [h for h in provenance.read_provenance(a)["history"] if h["step"] == step][-1]


# --- the step, in memory --------------------------------------------------------------------


@pytest.mark.parametrize("where,name", [("raw", ".raw"), ("layer", "layers['counts']")])
@pytest.mark.parametrize("scaled", [False, True])
def test_qc_metrics_come_from_the_counts_beside_a_processed_x(where, name, scaled):
    a, real = _lognorm(where, scaled=scaled)
    out, res = run_pipeline(a, ["qc_metrics"], PipelineParams())
    assert res.errors == {} and res.steps_run == ["qc_metrics"]
    for c in QC_OBS:
        np.testing.assert_array_equal(out.obs[c].to_numpy(), real[c].to_numpy(), err_msg=c)
    # 5 times the real total on a log-normalised matrix before: whole numbers now.
    assert np.all(np.mod(out.obs["total_counts"].to_numpy(), 1) == 0)
    assert _last(out)["params"]["input"] == name
    assert name in res.cautions["qc_metrics"]
    # per-gene columns: the counts' too
    counts = out.raw.X if where == "raw" else out.layers["counts"]
    np.testing.assert_array_equal(out.var["n_cells_by_counts"].to_numpy(),
                                  np.asarray((counts > 0).sum(axis=0)).ravel())
    np.testing.assert_allclose(out.var["total_counts"].to_numpy(), np.asarray(counts.sum(axis=0)).ravel())
    assert out.var["mt"].sum() == N_MT


def test_counts_in_x_are_counted_as_before_and_record_no_input():
    a = _counts()
    real = _real(a)
    out, res = run_pipeline(a, ["qc_metrics"], PipelineParams())
    for c in QC_OBS:
        np.testing.assert_array_equal(out.obs[c].to_numpy(), real[c].to_numpy(), err_msg=c)
    assert "input" not in _last(out)["params"] and "qc_metrics" not in res.cautions


def test_raw_with_more_genes_than_x_counts_every_gene_of_raw():
    a, real = _lognorm("raw")
    keep = a.var_names[::2]  # X cut to half the genes (4 of the 8 MT- genes left)
    a = a[:, keep].copy()
    out, res = run_pipeline(a, ["qc_metrics"], PipelineParams())
    assert res.errors == {}
    for c in QC_OBS:  # the cell's whole count, as the counts give it
        np.testing.assert_array_equal(out.obs[c].to_numpy(), real[c].to_numpy(), err_msg=c)
    per_gene = np.asarray((out.raw.X > 0).sum(axis=0)).ravel()
    np.testing.assert_array_equal(out.var["n_cells_by_counts"].to_numpy(),
                                  per_gene[out.raw.var_names.get_indexer(keep)])
    assert out.var["mt"].sum() == N_MT // 2


@pytest.mark.parametrize("scaled", [False, True])
def test_without_counts_the_step_is_skipped_and_writes_nothing(scaled):
    a, _ = _lognorm("none", scaled=scaled)
    out, res = run_pipeline(a, ["qc_metrics"], PipelineParams())
    assert res.steps_run == [] and res.errors == {}
    assert res.notes["qc_metrics"] == QC_NEEDS_COUNTS
    assert not set(QC_OBS) & set(out.obs.columns)
    assert "n_cells_by_counts" not in out.var.columns


def test_a_log_normalised_raw_is_not_taken_for_counts():
    a, _ = _lognorm("none")
    a.raw = a.copy()  # scView's own processed files: .raw log-normalised
    sc.pp.scale(a, max_value=10)
    out, res = run_pipeline(a, ["qc_metrics"], PipelineParams())
    assert res.notes["qc_metrics"] == QC_NEEDS_COUNTS and "total_counts" not in out.obs.columns


def test_the_filter_then_removes_what_the_counts_say():
    a, real = _lognorm("raw")
    params = PipelineParams(qc_threshold_mode="fixed", min_genes=0, min_counts=300, max_pct_mt=100, min_cells=0)
    out, res = run_pipeline(a, ["qc_metrics", "filtering"], params)
    assert res.errors == {}
    expect = int((real["total_counts"] >= 300).sum())
    assert 0 < expect < len(real)
    assert out.n_obs == expect


# --- large mode ------------------------------------------------------------------------------


@pytest.mark.parametrize("where,name", [("raw", ".raw"), ("layer", "layers['counts']")])
def test_large_mode_counts_on_the_counts_beside_x(tmp_path, monkeypatch, where, name):
    monkeypatch.setattr(chunked, "rows_per_block", lambda *a, **k: 250)
    a, real = _lognorm(where)
    src = tmp_path / "in" / "d.h5ad"
    src.parent.mkdir()
    a.write_h5ad(src)
    out = tmp_path / "out" / "d.h5ad"
    res = large_run.run(src, ["qc_metrics"], PipelineParams(chunked="on"), str(out))
    assert res.errors == {} and res.steps_run == ["qc_metrics"], (res.errors, res.notes)
    big = ad.read_h5ad(out)
    mem, _ = run_pipeline(ad.read_h5ad(src), ["qc_metrics"], PipelineParams())
    for c in QC_OBS:
        np.testing.assert_array_equal(big.obs[c].to_numpy(), real[c].to_numpy(), err_msg=c)
        np.testing.assert_array_equal(big.obs[c].to_numpy(), mem.obs[c].to_numpy(), err_msg=c)
    for c in ("mt", "n_cells_by_counts", "mean_counts", "pct_dropout_by_counts", "total_counts"):
        np.testing.assert_allclose(big.var[c].to_numpy().astype(float), mem.var[c].to_numpy().astype(float),
                                   rtol=1e-6, err_msg=c)
    assert _last(big)["params"]["input"] == name
    assert name in res.cautions["qc_metrics"]


# --- replay ----------------------------------------------------------------------------------


def _replay(tmp_path: Path, inp: ad.AnnData, out: ad.AnnData) -> ad.AnnData:
    src, script, dst = tmp_path / "input.h5ad", tmp_path / "replay.py", tmp_path / "replayed.h5ad"
    inp.write_h5ad(src)
    script.write_text(replay_script.build_script(out, dataset_id="t", dataset_name="test"))
    proc = subprocess.run([sys.executable, str(script), str(src), str(dst)],
                          capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr[-3000:]
    return ad.read_h5ad(dst)


@pytest.mark.parametrize("where", ["raw", "layer"])
def test_the_replay_script_counts_on_the_recorded_matrix(tmp_path, where):
    inp, real = _lognorm(where)
    out, _ = run_pipeline(inp.copy(), ["qc_metrics"], PipelineParams())
    rep = _replay(tmp_path, inp, out)
    for c in QC_OBS:
        np.testing.assert_array_equal(rep.obs[c].to_numpy(), real[c].to_numpy(), err_msg=c)
    np.testing.assert_array_equal(rep.var["n_cells_by_counts"].to_numpy(), out.var["n_cells_by_counts"].to_numpy())


def _old_run(a: ad.AnnData) -> ad.AnnData:
    """The QC metrics step as it ran before v0.86: on X whatever it held, nothing recorded."""
    a.var["mt"] = a.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(a, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)
    provenance.record_step(a, step="qc_metrics", tool="scanpy.pp.calculate_qc_metrics", params={})
    return a


def test_an_old_recipe_replays_to_its_old_numbers(tmp_path):
    inp, real = _lognorm("raw")
    old = _old_run(inp.copy())
    assert not np.allclose(old.obs["total_counts"], real["total_counts"])
    rep = _replay(tmp_path, inp, old)
    np.testing.assert_allclose(rep.obs["total_counts"].to_numpy(), old.obs["total_counts"].to_numpy(), rtol=1e-6)


# --- datasets already carrying columns from a processed X ----------------------------------


def test_the_note_names_columns_an_old_step_took_from_a_processed_x():
    inp, _ = _lognorm("raw")
    old = _old_run(inp.copy())
    note = qc_filter.not_counts_note(old)
    assert note and "not computed from counts" in note
    # run again: counted on the counts, the note is gone
    out, _ = run_pipeline(old, ["qc_metrics"], PipelineParams())
    assert qc_filter.not_counts_note(out) is None


def test_no_note_for_counts_or_for_columns_scview_did_not_write():
    assert qc_filter.not_counts_note(_old_run(_counts())) is None  # counted on counts
    a, _ = _lognorm("raw")
    a.obs["total_counts"] = np.linspace(0.5, 900.5, a.n_obs)  # the authors' own columns
    a.obs["n_genes_by_counts"] = 10
    assert qc_filter.not_counts_note(a) is None
    assert qc_filter.not_counts_note(_counts()) is None  # no QC columns at all


def _register(dm, a: ad.AnnData, did: str) -> str:
    d = Path(dm.data_dir) / "uploads" / did
    d.mkdir(parents=True, exist_ok=True)
    (d / "metadata.json").write_text(json.dumps({"id": did, "name": did, "status": "ready"}))
    a.write_h5ad(d / "data.h5ad")
    return did


async def test_the_qc_panel_and_the_filter_preview_carry_the_note(client, dataset_manager):
    inp, _ = _lognorm("raw")
    did = _register(dataset_manager, _old_run(inp.copy()), "qcold")
    preview = (await client.get(f"{API}/datasets/{did}/qc/filter-preview")).json()
    panel = (await client.get(f"{API}/datasets/{did}/qc")).json()
    assert "not computed from counts" in preview["not_counts_note"]
    assert panel["not_counts_note"] == preview["not_counts_note"]

    out, _ = run_pipeline(inp.copy(), ["qc_metrics"], PipelineParams())
    did = _register(dataset_manager, out, "qcnew")
    preview = (await client.get(f"{API}/datasets/{did}/qc/filter-preview")).json()
    panel = (await client.get(f"{API}/datasets/{did}/qc")).json()
    assert "not_counts_note" not in preview and "not_counts_note" not in panel
