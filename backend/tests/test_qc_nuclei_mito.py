"""The % mito floor under per-sample MAD QC, and the nuclei warning (core/qc_filter.py).

Nuclei hold almost no mitochondria, so % mito piles up near 0 and median + 3 MADs collapses
(10x mouse kidney multiome: a 0.139% bound removed 24% of the nuclei). MAD mode's % mito bound
is therefore never below a floor (2% by default), the preview and the record say per sample
which bound applied, and data that look like nuclei get a warning when % mito alone still
removes more than 5% of a sample.
"""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from scview.core import mito_refilter, provenance, qc_filter
from scview.core.anndata_adaptor import per_cell_qc
from scview.core.pipeline import PipelineParams, run_pipeline
from scview.core.qc_filter import Thresholds

API = "/api/v1"


def _cells(groups: dict[str, tuple[float, float]], n: int = 400, seed: int = 0, tail: float = 0.0,
           **obs_extra) -> ad.AnnData:
    """Cells per sample with % mito ~ |normal(mean, sd)| (a ``tail`` share of them uniform in
    0.3-2%, ambient-heavy nuclei) and plain genes / counts."""
    rng = np.random.default_rng(seed)
    parts = []
    for name, (mean, sd) in groups.items():
        genes = np.round(np.exp(rng.normal(np.log(1500), 0.3, n)))
        pct = np.abs(rng.normal(mean, sd, n))
        k = int(tail * n)
        pct[:k] = rng.uniform(0.3, 2.0, k)
        parts.append(pd.DataFrame({"Donor": name, "n_genes_by_counts": genes,
                                   "total_counts": np.round(genes * 2.5), "pct_counts_mt": pct}))
    obs = pd.concat(parts, ignore_index=True)
    obs.index = [f"c{i}" for i in range(len(obs))]
    obs["Donor"] = pd.Categorical(obs["Donor"], categories=list(groups))
    for k, v in obs_extra.items():
        obs[k] = v
    a = ad.AnnData(sp.csr_matrix((len(obs), 3), dtype=np.float32), obs=obs)
    a.var_names = ["MT-CO1", "G1", "G2"]
    return a


def _rows(out: dict) -> dict[str, dict]:
    return {r["name"]: r for r in out["mad"]["samples"]}


def test_the_floor_lifts_a_collapsed_bound_and_leaves_whole_cells_alone():
    # "nuc": median ~0.05%, a tail of ambient-heavy nuclei up to a few %; "cell": whole cells, median 6%.
    a = _cells({"nuc": (0.05, 0.4), "cell": (6.0, 1.5)})
    cells = per_cell_qc(a)
    off = qc_filter.preview(a, cells, Thresholds(mode="mad", max_pct_mt=20.0, pct_mt_floor=0.0))
    on = qc_filter.preview(a, cells, Thresholds(mode="mad", max_pct_mt=20.0, pct_mt_floor=2.0))
    nuc_off, nuc_on = _rows(off)["nuc"], _rows(on)["nuc"]
    assert nuc_off["max_pct_mt"] < 2.0 and "max_pct_mt_by" not in nuc_off
    assert nuc_on["max_pct_mt"] == 2.0 and nuc_on["max_pct_mt_by"] == "floor"
    assert nuc_on["max_pct_mt_mad"] == nuc_off["max_pct_mt"] and nuc_on["n_pct_mt_floor"] > 0
    assert on["by_criterion"]["max_pct_mt"]["n_removed"] == (off["by_criterion"]["max_pct_mt"]["n_removed"]
                                                             - nuc_on["n_pct_mt_floor"])
    # The whole-cell sample's own bound is far above the floor: unchanged.
    cell = _rows(on)["cell"]
    assert cell["max_pct_mt_by"] == "mad" and cell["max_pct_mt"] == _rows(off)["cell"]["max_pct_mt"]
    assert on["mad"]["pct_mt_floor"] == 2.0 and on["thresholds"]["pct_mt_floor"] == 2.0
    assert "pct_mt_floor" not in off["mad"]


def test_the_floor_never_goes_above_the_ceiling_and_defaults_to_two():
    a = _cells({"nuc": (0.05, 0.4)})
    out = qc_filter.preview(a, per_cell_qc(a), Thresholds(mode="mad", max_pct_mt=1.0, pct_mt_floor=5.0))
    assert out["mad"]["pooled"]["max_pct_mt"] <= 1.0
    assert Thresholds.from_params(PipelineParams()).pct_mt_floor is None
    assert qc_filter.resolve_floor(a, Thresholds.from_params(PipelineParams())).pct_mt_floor == qc_filter.MITO_FLOOR
    assert qc_filter.resolve_floor(a, Thresholds.from_params(PipelineParams(qc_pct_mt_floor=0))).pct_mt_floor == 0.0
    try:
        Thresholds(mode="mad", pct_mt_floor=-1.0).check()
    except ValueError as e:
        assert "floor" in str(e)
    else:
        raise AssertionError("a negative floor must be refused")


def test_nuclei_are_recognised_from_metadata_or_their_mito():
    nuc = _cells({"nuc": (0.05, 0.1)})
    assert "median" in qc_filter.nuclei_reason(nuc, per_cell_qc(nuc))
    cell = _cells({"cell": (6.0, 1.5)})
    assert qc_filter.nuclei_reason(cell, per_cell_qc(cell)) is None
    tagged = _cells({"cell": (6.0, 1.5)}, suspension_type="nucleus")
    assert qc_filter.nuclei_reason(tagged, per_cell_qc(tagged)) == "suspension_type is nucleus"
    multiome = _cells({"cell": (6.0, 1.5)}, atac_total_counts=5000.0)
    assert "Multiome" in qc_filter.nuclei_reason(multiome, per_cell_qc(multiome))


def test_nuclei_warn_when_mito_alone_removes_over_five_percent_of_a_sample():
    # PBMC multiome nuclei hold real mito reads: a bound of ~16% still removes ~5%+.
    a = _cells({"A": (8.0, 3.0), "B": (8.0, 3.0)}, suspension_type="nucleus")
    out = qc_filter.preview(a, per_cell_qc(a), Thresholds(mode="mad", max_pct_mt=10.0))
    w = [x for x in out["warnings"] if x["kind"] == "nuclei"]
    assert len(w) == 1 and w[0] is out["warnings"][0]
    assert "look like nuclei (suspension_type is nucleus)" in w[0]["message"] and "A " in w[0]["message"]
    # Whole cells with the same numbers: no nuclei warning.
    b = _cells({"A": (8.0, 3.0), "B": (8.0, 3.0)})
    out = qc_filter.preview(b, per_cell_qc(b), Thresholds(mode="mad", max_pct_mt=10.0))
    assert not [x for x in out["warnings"] if x["kind"] == "nuclei"]
    # Nuclei the floor protects: no warning.
    c = _cells({"nuc": (0.05, 0.3)})
    out = qc_filter.preview(c, per_cell_qc(c), Thresholds(mode="mad", max_pct_mt=20.0, pct_mt_floor=2.0))
    assert out["by_criterion"]["max_pct_mt"]["n_removed"] < 0.05 * c.n_obs
    assert not [x for x in out["warnings"] if x["kind"] == "nuclei"]


def test_the_run_records_the_floor_and_the_nuclei_note():
    a = _cells({"A": (8.0, 3.0), "B": (0.05, 0.4)}, suspension_type="nucleus")
    params = PipelineParams(min_cells=0, max_pct_mt=10.0)
    prev = qc_filter.preview(a, per_cell_qc(a), Thresholds.from_params(params))
    out, res = run_pipeline(a.copy(), ["filtering"], params)
    assert res.errors == {} and out.n_obs == prev["n_kept"]
    p = provenance.read_provenance(out)["history"][-1]["params"]
    assert p["qc_pct_mt_floor"] == 2.0 and p["qc_thresholds"]["pct_mt_floor"] == 2.0
    assert {r["name"]: r for r in p["qc_thresholds"]["samples"]}["B"]["max_pct_mt_by"] == "floor"
    assert "look like nuclei" in p["qc_nuclei_note"] and res.cautions["filtering"] == p["qc_nuclei_note"]
    assert "% mito never cut below 2%" in qc_filter.applied_summary(p["qc_thresholds"])
    # The same settings again reuse the recorded bounds: nothing new removed.
    again, _ = run_pipeline(out.copy(), ["filtering"], params)
    assert again.n_obs == out.n_obs
    # The replay script applies the recorded bound, floor included.
    from scview.core.replay_script import build_script

    script = build_script(out)
    assert "'max_pct_mt_by': 'floor'" in script or '"max_pct_mt_by": "floor"' in script


def test_a_run_recorded_before_the_floor_is_refiltered_without_it():
    a = _cells({"nuc": (0.05, 0.4)})
    provenance.record_step(a, step="filtering", tool="scview.qc_filter",
                           params={"qc_threshold_mode": "mad", "qc_mad_k": 5.0, "qc_mad_k_mt": 3.0})
    assert mito_refilter.plan_params(a)[1].qc_pct_mt_floor == 0.0
    b = _cells({"nuc": (0.05, 0.4)})
    provenance.record_step(b, step="filtering", tool="scview.qc_filter",
                           params={"qc_threshold_mode": "mad", "qc_pct_mt_floor": 3.0})
    assert mito_refilter.plan_params(b)[1].qc_pct_mt_floor == 3.0


def _register(dm, a: ad.AnnData, did: str) -> str:
    d = Path(dm.data_dir) / "uploads" / did
    d.mkdir(parents=True)
    (d / "metadata.json").write_text(json.dumps({"id": did, "name": did, "status": "ready"}))
    a.write_h5ad(d / "data.h5ad")
    return did


async def test_the_preview_endpoint_takes_a_pct_mt_floor(client, dataset_manager):
    did = _register(dataset_manager, _cells({"nuc": (0.03, 0.03)}, tail=0.15), "mitofloor1")
    url = f"{API}/datasets/{did}/qc/filter-preview"
    default = (await client.get(url)).json()
    assert default["thresholds"]["pct_mt_floor"] == 2.0
    assert default["mad"]["pooled"]["max_pct_mt_by"] == "floor"
    off = (await client.get(url, params={"pct_mt_floor": 0})).json()
    assert off["thresholds"]["pct_mt_floor"] == 0.0 and off["n_removed"] > default["n_removed"]
    assert any(w["kind"] == "nuclei" for w in off["warnings"])
    assert (await client.get(url, params={"pct_mt_floor": -1})).status_code == 400
