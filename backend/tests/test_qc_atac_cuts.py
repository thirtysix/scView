"""Max counts and the 10x Multiome ATAC QC cut-offs in the filtering step (core/qc_filter.py).

Signac's multiome QC (ATAC counts, nucleosome signal, TSS enrichment, an RNA counts ceiling)
could not be applied before v0.78. Each cut-off is off unless set, works in both modes, shows in
the preview, the removal record, the recipe and the replay script; the preview offers Signac's
vignette cuts as a suggestion, never as a default.
"""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from scview.core import atac_fragments, atac_import, provenance, qc_filter
from scview.core.anndata_adaptor import per_cell_qc
from scview.core.pipeline import PipelineParams, run_pipeline
from scview.core.qc_filter import Thresholds

API = "/api/v1"


def _multiome(n: int = 300, seed: int = 0, fragment_qc: bool = True) -> ad.AnnData:
    """Cells with RNA QC plus ATAC counts, and (with ``fragment_qc``) TSS / nucleosome values;
    the first 10 cells have few ATAC counts, the next 5 a nucleosome signal of 3, the next 5 a
    TSS enrichment of 0.5, the next 5 over 30,000 RNA counts."""
    rng = np.random.default_rng(seed)
    genes = np.round(np.exp(rng.normal(np.log(1500), 0.2, n)))
    obs = pd.DataFrame({"Donor": pd.Categorical(np.where(np.arange(n) % 2, "d1", "d2")),
                        "n_genes_by_counts": genes, "total_counts": np.round(genes * 2.5),
                        "pct_counts_mt": np.abs(rng.normal(3.0, 0.5, n)),
                        "atac_total_counts": np.round(np.exp(rng.normal(np.log(15000), 0.3, n)))},
                       index=[f"c{i}" for i in range(n)])
    obs.iloc[:10, obs.columns.get_loc("atac_total_counts")] = 500.0
    obs.iloc[20:25, obs.columns.get_loc("total_counts")] = 40_000.0
    if fragment_qc:
        obs["atac_nucleosome_signal"] = np.abs(rng.normal(0.6, 0.1, n))
        obs["atac_tss_enrichment"] = np.abs(rng.normal(5.0, 0.8, n))
        obs.iloc[10:15, obs.columns.get_loc("atac_nucleosome_signal")] = 3.0
        obs.iloc[15:20, obs.columns.get_loc("atac_tss_enrichment")] = 0.5
        obs.iloc[3, obs.columns.get_loc("atac_tss_enrichment")] = np.nan  # no fragments: kept by it
    a = ad.AnnData(sp.csr_matrix((n, 3), dtype=np.float32), obs=obs)
    a.var_names = ["MT-CO1", "G1", "G2"]
    return a


SIGNAC = dict(max_counts=25_000.0, min_atac_counts=1_800.0, max_atac_counts=100_000.0,
              max_nucleosome_signal=2.0, min_tss_enrichment=1.0)


def test_the_columns_are_the_ones_atac_import_and_fragment_qc_write():
    cols = {c for c, _, _ in qc_filter.ATAC_CRITERIA.values()}
    assert cols == {atac_import.OBS_TOTAL, atac_fragments.OBS_NUCLEOSOME, atac_fragments.OBS_TSS}


@pytest.mark.parametrize("mode", ["fixed", "mad"])
def test_each_cut_removes_its_cells_in_either_mode(mode):
    a = _multiome()
    t = Thresholds(mode=mode, max_pct_mt=20.0, **SIGNAC)
    out = qc_filter.preview(a, per_cell_qc(a), t)
    by = {c: v["n_removed"] for c, v in out["by_criterion"].items()}
    assert by["min_atac_counts"] == 10 and by["max_nucleosome_signal"] == 5
    assert by["min_tss_enrichment"] == 5 and by["max_counts"] == 5 and by["max_atac_counts"] == 0
    assert out["by_criterion"]["max_nucleosome_signal"]["label"] == "Max nucleosome signal"
    assert out["thresholds"]["min_tss_enrichment"] == 1.0 and out["thresholds"]["max_counts"] == 25_000.0
    # Off by default: none of them is in the masks.
    plain = qc_filter.preview(a, per_cell_qc(a), Thresholds(mode=mode, max_pct_mt=20.0))
    assert not set(plain["by_criterion"]) & {"max_counts", *qc_filter.ATAC_CRITERIA}


def test_the_preview_suggests_signacs_cuts_without_applying_them():
    a = _multiome()
    out = qc_filter.preview(a, per_cell_qc(a), Thresholds(mode="mad", max_pct_mt=20.0))
    s = out["atac_qc"]
    assert s["suggested"] == SIGNAC and s["on"] == [] and "Signac" in s["source"]
    assert s["n_removed_by"]["min_atac_counts"] == 10 and s["n_removed"] == 25
    assert "min_atac_counts" not in out["by_criterion"]  # a suggestion only
    # Without fragment QC: only the ATAC counts cut is offered, and the note says how to get the rest.
    b = _multiome(fragment_qc=False)
    s = qc_filter.preview(b, per_cell_qc(b), Thresholds(mode="mad"))["atac_qc"]
    assert set(s["suggested"]) == {"max_counts", "min_atac_counts", "max_atac_counts"}
    assert "fragment QC" in s["missing_note"]
    # RNA-only data: no ATAC block.
    c = _multiome()
    c.obs = c.obs.drop(columns=["atac_total_counts", "atac_nucleosome_signal", "atac_tss_enrichment"])
    assert qc_filter.preview(c, per_cell_qc(c), Thresholds(mode="mad"))["atac_qc"] is None


def test_a_cut_without_its_column_is_refused_with_what_to_run():
    b = _multiome(fragment_qc=False)
    with pytest.raises(ValueError, match="fragment QC"):
        qc_filter.preview(b, per_cell_qc(b), Thresholds(mode="mad", min_tss_enrichment=1.0))
    with pytest.raises(ValueError, match="0 \\(off\\) or more"):
        Thresholds(max_counts=-1.0).check()


def test_the_run_records_the_cuts_in_the_record_recipe_and_replay():
    a = _multiome()
    params = PipelineParams(min_cells=0, max_counts=25_000, qc_min_atac_counts=1_800.0,
                            qc_max_nucleosome_signal=2.0, qc_min_tss_enrichment=1.0)
    prev = qc_filter.preview(a, per_cell_qc(a), Thresholds.from_params(params))
    out, res = run_pipeline(a.copy(), ["filtering"], params)
    assert res.errors == {} and out.n_obs == prev["n_kept"]
    p = provenance.read_provenance(out)["history"][-1]["params"]
    assert p["max_counts"] == 25_000 and p["qc_min_atac_counts"] == 1_800.0
    assert p["qc_max_nucleosome_signal"] == 2.0 and p["qc_min_tss_enrichment"] == 1.0
    assert "qc_max_atac_counts" not in p  # off: not recorded
    rec = out.uns[qc_filter.RECORD_KEY]
    crit = list(rec["criteria"])
    assert {"max_counts", "min_atac_counts", "max_nucleosome_signal", "min_tss_enrichment"} <= set(crit)
    bits = qc_filter.criterion_bits(rec)
    tss_bit = 1 << crit.index("min_tss_enrichment")
    assert int(((bits & tss_bit) > 0).sum()) == 5
    assert set(rec["atac"]) == {"atac_total_counts", "atac_nucleosome_signal", "atac_tss_enrichment"}
    audit = qc_filter.recorded_audit(out)
    assert audit is not None and audit["n_removed"] == prev["n_removed"]

    from scview.core.replay_script import build_script

    script = build_script(out)
    assert "cut_drop(obs_values(adata, 'atac_tss_enrichment'), low=1.0)" in script
    assert "cut_drop(obs_values(adata, 'atac_nucleosome_signal'), high=2.0)" in script
    assert 'cut_drop(qc["total_counts"], high=25000.0)' in script
    # The generated helper works on the same values.
    ns: dict = {"np": np}
    exec(qc_filter_source(script), ns)
    drop = ns["cut_drop"](a.obs["atac_tss_enrichment"].to_numpy(), low=1.0)
    assert int(drop.sum()) == 5 and ns["obs_values"](a, "nope").shape == (a.n_obs,)


def qc_filter_source(script: str) -> str:
    """The cut_drop and obs_values helpers out of a replay script."""
    start = script.index("def cut_drop(")
    end = script.index("\n\n\n", script.index("def obs_values("))
    return script[start:end]


def _register(dm, a: ad.AnnData, did: str) -> str:
    d = Path(dm.data_dir) / "uploads" / did
    d.mkdir(parents=True)
    (d / "metadata.json").write_text(json.dumps({"id": did, "name": did, "status": "ready"}))
    a.write_h5ad(d / "data.h5ad")
    return did


async def test_the_preview_endpoint_takes_the_atac_cuts(client, dataset_manager):
    did = _register(dataset_manager, _multiome(), "atacqc1")
    url = f"{API}/datasets/{did}/qc/filter-preview"
    base = (await client.get(url)).json()
    assert base["atac_qc"]["n_removed"] == 25
    cut = (await client.get(url, params={"min_tss_enrichment": 1, "max_counts": 25000})).json()
    assert cut["by_criterion"]["min_tss_enrichment"]["n_removed"] == 5
    assert cut["atac_qc"]["on"] == ["min_tss_enrichment", "max_counts"]
    c = _multiome(fragment_qc=False)
    did2 = _register(dataset_manager, c, "atacqc2")
    r = await client.get(f"{API}/datasets/{did2}/qc/filter-preview", params={"max_nucleosome_signal": 2})
    assert r.status_code == 400 and "fragment QC" in r.json()["detail"]
