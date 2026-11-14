"""Reset to counts from the cells as uploaded (core/reset_original.py, ``reset_from_original``).

A dataset filtered once could not get its cells back: reset_to_counts with looser QC kept the
cells the first filtering left (kidney multiome: 4,012 kept, 17 more trimmed). With
``reset_from_original`` the run starts from the stored original upload; a dataset without one is
refused with a sentence, never silently reset from the cells it has.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scview.core import provenance, reset_original
from scview.core.pipeline import PipelineParams
from tests.test_mito_refilter import P1, PASS1, _register

API = "/api/v1"
LOOSE = {"qc_threshold_mode": "fixed", "min_genes": 20, "max_pct_mt": 50.0, "n_top_genes": 100, "n_comps": 15}
STEPS = ["reset_to_counts", "qc_metrics", "filtering", "normalization", "log_transform"]


@pytest.fixture()
def filtered(dataset_manager):
    orig, cur = _register(dataset_manager, "rs1")
    assert cur.n_obs < orig.n_obs  # pass 1's 10 % ceiling removed cells
    return dataset_manager, orig, cur


async def test_a_plain_reset_keeps_the_cells_left(client, filtered):
    dm, orig, cur = filtered
    r = await client.post(f"{API}/datasets/rs1/assessment/run", json={"steps": STEPS, "params": LOOSE})
    assert r.status_code == 200 and r.json()["errors"] == {}
    N = (await dm.get_or_load_dataset("rs1")).adata
    assert N.n_obs == cur.n_obs  # nothing comes back


async def test_reset_from_original_brings_the_removed_cells_back(client, filtered):
    dm, orig, cur = filtered
    r = await client.post(f"{API}/datasets/rs1/assessment/run",
                          json={"steps": STEPS, "params": {**LOOSE, "reset_from_original": True}})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["errors"] == {} and "Started from the 800 cells as uploaded" in body["cautions"]["reset_to_counts"]
    N = (await dm.get_or_load_dataset("rs1")).adata
    assert N.n_obs == orig.n_obs  # a 50 % ceiling keeps every cell of the upload
    hist = provenance.read_provenance(N)["history"]
    assert [h["step"] for h in hist] == PASS1 + STEPS
    reset = hist[len(PASS1)]["params"]
    assert reset["from_original"] and reset["n_original"] == orig.n_obs and reset["n_before"] == cur.n_obs
    assert reset_original.MARK not in N.uns


async def test_the_streamed_run_says_where_it_started(client, filtered):
    dm, orig, cur = filtered
    r = await client.post(f"{API}/datasets/rs1/assessment/run-stream",
                          json={"steps": STEPS, "params": {**LOOSE, "reset_from_original": True}})
    assert r.status_code == 200
    assert "event: reset" in r.text and f'"n_original": {orig.n_obs}' in r.text and "event: complete" in r.text
    assert "event: error" not in r.text
    assert (await dm.get_or_load_dataset("rs1")).adata.n_obs == orig.n_obs
    assert dm.run_reservations() == {}


async def test_atac_qc_columns_come_along_for_the_cells_that_had_them(client, filtered):
    dm, orig, cur = filtered
    adaptor = await dm.get_or_load_dataset("rs1")
    A = reset_original.load(dm, "rs1", _with_tss(adaptor, cur))
    vals = A.obs["atac_tss_enrichment"].to_numpy()
    assert np.isfinite(vals).sum() == cur.n_obs and A.n_obs == orig.n_obs
    mark = reset_original.take_mark(A)
    assert mark["carried_columns"] == ["atac_tss_enrichment"] and "fragment QC runs again" in mark["note"]


def _with_tss(adaptor, cur):
    adaptor.adata.obs["atac_tss_enrichment"] = np.linspace(1, 9, adaptor.adata.n_obs)
    return adaptor


async def test_what_it_refuses(client, dataset_manager):
    _register(dataset_manager, "noorig2")
    (Path(dataset_manager.data_dir) / "uploads" / "noorig2" / "raw.h5ad").unlink()  # derived only
    r = await client.post(f"{API}/datasets/noorig2/assessment/run",
                          json={"steps": STEPS, "params": {"reset_from_original": True}})
    assert r.status_code == 409 and "No original upload is stored" in r.json()["detail"]
    r = await client.post(f"{API}/datasets/noorig2/assessment/run-stream",
                          json={"steps": STEPS, "params": {"reset_from_original": True}})
    assert r.status_code == 409
    _register(dataset_manager, "rs2")
    r = await client.post(f"{API}/datasets/rs2/assessment/run",
                          json={"steps": ["filtering"], "params": {"reset_from_original": True}})
    assert r.status_code == 422 and "reset_to_counts" in r.json()["detail"]


def test_the_run_is_estimated_from_the_original(filtered):
    dm, orig, cur = filtered
    needs = reset_original.step_needs(dm, "rs1", STEPS, PipelineParams(**LOOSE))
    assert set(STEPS) <= set(needs) and all(v >= 0 for v in needs.values())
    assert P1.max_pct_mt == 10.0
