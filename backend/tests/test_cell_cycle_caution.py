"""Cell-cycle phase calls on cells that are not dividing (core/cell_cycle.py).

scanpy calls a cell G1 only when both scores are below 0, so resting cells whose
scores are 0 plus noise come out S or G2M about half the time (Kang 2018 PBMC:
7,827 of 13,836). The phase column stays as scanpy wrote it; the step records,
and the assessment shows, how many cells have both scores near 0.
"""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

from scview.core import provenance
from scview.core.anndata_adaptor import AnnDataAdaptor
from scview.core.assessor import _check_cell_cycle, assess_preprocessing
from scview.core.cell_cycle import NEAR_ZERO_MARGIN, near_zero_phase_calls, phase_caution
from scview.core.pipeline import (
    CELL_CYCLE_G2M_GENES,
    CELL_CYCLE_S_GENES,
    PipelineParams,
    run_pipeline,
)
from scview.core import qc_report

API = "/api/v1"
S_GENES = list(CELL_CYCLE_S_GENES)
G2M_GENES = list(CELL_CYCLE_G2M_GENES)


def _log_normalised(n: int = 300, cycling: bool = False) -> ad.AnnData:
    """Sparse, PBMC-like log-normalised data: most genes undetected in most cells.
    With ``cycling``, the first half of the cells express the S genes and the
    second half the G2M genes."""
    rng = np.random.default_rng(0)
    names = S_GENES + G2M_GENES + [f"GENE{i}" for i in range(400)]
    X = (rng.random((n, len(names))) < 0.03) * rng.gamma(2.0, 0.5, (n, len(names)))
    if cycling:
        s, g = len(S_GENES), len(G2M_GENES)
        X[: n // 2, :s] += 1.5
        X[n // 2 :, s:s + g] += 1.5
    a = ad.AnnData(X.astype("float32"))
    a.var_names = names
    a.obs_names = [f"C{i}" for i in range(n)]
    return a


def _obs(s, g, phase=None) -> pd.DataFrame:
    obs = pd.DataFrame({"S_score": s, "G2M_score": g})
    if phase is not None:
        obs["phase"] = pd.Categorical(phase)
    return obs


def test_counts_cells_with_both_scores_near_zero():
    obs = _obs([0.01, -0.05, 0.09, 0.3, 0.02, np.nan],
               [-0.02, 0.04, -0.2, 0.1, 0.0, 0.0],
               ["S", "G2M", "G1", "S", "G1", "G1"])
    c = near_zero_phase_calls(obs)
    # Cells 0, 1, 4 are inside ±0.1 on both scores; cell 2's G2M and cell 3's S are not.
    assert c == {"n_cells": 5, "n_near_zero": 3, "near_zero_margin": NEAR_ZERO_MARGIN,
                 "n_called_cycling": 3, "n_called_cycling_near_zero": 2}
    assert near_zero_phase_calls(pd.DataFrame({"S_score": [0.0]})) is None


def test_caution_only_when_a_real_share_of_cells_is_near_zero():
    many = {"n_cells": 13836, "n_near_zero": 13162, "near_zero_margin": 0.1,
            "n_called_cycling": 7827, "n_called_cycling_near_zero": 7164}
    text = phase_caution(many)
    assert text.startswith("13,162 of 13,836 cells (95%) have both S and G2M scores within ±0.1 of 0")
    assert "7,164 of them called S or G2M" in text and "not reliable" in text
    # A dividing population (paul15: 5.6% near zero) gets no caution.
    assert phase_caution({"n_cells": 2730, "n_near_zero": 153, "near_zero_margin": 0.1}) is None
    assert phase_caution(None) is None


def test_step_records_the_count_and_leaves_scanpys_phase_alone():
    a = _log_normalised()
    out, res = run_pipeline(a.copy(), ["cell_cycle"], PipelineParams())
    assert res.errors == {}, res.errors
    # The phase column is scanpy's own call on the same genes.
    ref = a.copy()
    sc.tl.score_genes_cell_cycle(ref, s_genes=S_GENES, g2m_genes=G2M_GENES)
    assert list(out.obs["phase"].astype(str)) == list(ref.obs["phase"].astype(str))
    # Resting cells: about half called S or G2M, nearly all on scores near 0.
    params = provenance.read_provenance(out)["history"][-1]["params"]
    assert params["n_cells"] == 300 and params["near_zero_margin"] == NEAR_ZERO_MARGIN
    assert params["n_near_zero"] > 0.9 * 300
    assert params["n_called_cycling"] > 0.3 * 300
    assert params["n_called_cycling_near_zero"] > 0.9 * params["n_called_cycling"]


def test_assessment_shows_the_caution_also_for_a_step_scview_recorded():
    out, _ = run_pipeline(_log_normalised(), ["cell_cycle"], PipelineParams())
    status = assess_preprocessing(out).cell_cycle
    # The recorded-provenance overlay replaces the details, not the caution.
    assert status.details.startswith("Run by scView")
    assert status.caution and "not reliable" in status.caution
    # Really dividing cells: no caution.
    cyc, _ = run_pipeline(_log_normalised(cycling=True), ["cell_cycle"], PipelineParams())
    assert (cyc.obs["phase"].astype(str)[:150] == "S").mean() > 0.8
    assert near_zero_phase_calls(cyc.obs)["n_near_zero"] < 0.1 * cyc.n_obs
    assert assess_preprocessing(cyc).cell_cycle.caution is None
    # Scores made elsewhere (no provenance) are judged the same way.
    assert _check_cell_cycle(ad.AnnData(obs=_obs([0.01] * 10, [0.02] * 10))).caution


def test_qc_report_lists_it_as_a_concern(tmp_path: Path):
    out, _ = run_pipeline(_log_normalised(), ["cell_cycle"], PipelineParams())
    p = tmp_path / "d.h5ad"
    out.write_h5ad(p)
    facts = qc_report.build_facts(AnnDataAdaptor(str(p)))
    concern = next(s for s in facts.concerns if s.ref == "qc:cell_cycle")
    assert concern.detail.startswith("Cell cycle: ") and "not reliable" in concern.detail


async def test_assessment_endpoint_returns_the_caution(client, dataset_manager):
    out, _ = run_pipeline(_log_normalised(), ["cell_cycle"], PipelineParams())
    d = Path(dataset_manager.data_dir) / "uploads" / "cc"
    d.mkdir(parents=True)
    (d / "metadata.json").write_text(json.dumps({"id": "cc", "name": "cc", "status": "ready"}))
    out.write_h5ad(d / "data.h5ad")
    r = await client.get(f"{API}/datasets/cc/assessment")
    assert r.status_code == 200, r.text
    assert "not reliable" in r.json()["cell_cycle"]["caution"]


def test_cell_cycle_scores_log_normalised_x_when_raw_holds_counts():
    # A fresh upload can carry counts in .raw and log-normalised X (Kang). scanpy's
    # use_raw default would score the counts; the step must score X instead.
    import anndata as ad
    import numpy as np
    import scanpy as sc

    from scview.core.pipeline import (
        CELL_CYCLE_G2M_GENES,
        CELL_CYCLE_S_GENES,
        PipelineParams,
        run_pipeline,
    )

    genes = list(dict.fromkeys(CELL_CYCLE_S_GENES + CELL_CYCLE_G2M_GENES)) + [f"G{i}" for i in range(400)]
    rng = np.random.default_rng(0)
    counts = ad.AnnData(rng.poisson(1.5, (300, len(genes))).astype("float32"))
    counts.var_names = genes
    a = counts.copy()
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    expected = a.copy()
    a.raw = counts

    sc.tl.score_genes_cell_cycle(
        expected,
        s_genes=[g for g in CELL_CYCLE_S_GENES if g in genes],
        g2m_genes=[g for g in CELL_CYCLE_G2M_GENES if g in genes],
        use_raw=False,
    )
    out, res = run_pipeline(a, ["cell_cycle"], PipelineParams())
    assert res.errors == {}
    np.testing.assert_allclose(out.obs["S_score"], expected.obs["S_score"], rtol=1e-5)
