"""Cell-type-aware % mito bounds (core/mito_by_type.py): a type with a high
baseline is flagged at a 10 % ceiling, a low one is not; unmeasurable mito, tiny
groups and unlabelled cells; the endpoint and the co-pilot fact line."""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from scview.core import mito_by_type as mbt
from scview.core import provenance
from scview.core.claims import resolve_tag
from scview.core.pipeline import PipelineParams

API = "/api/v1"


def _synthetic(unlabelled: int = 20) -> tuple[ad.AnnData, dict]:
    """LNCaP-like (median ~12 % mito), CAF-like (~3 %), a 10-cell type, unlabelled cells."""
    rng = np.random.default_rng(0)
    mt = np.concatenate([rng.normal(12.1, 1.5, 400), rng.normal(3.0, 1.0, 300),
                         np.full(10, 15.0), rng.normal(5.0, 1.0, unlabelled)]).clip(0)
    labels = ["LNCaP"] * 400 + ["CAF"] * 300 + ["rare"] * 10 + [None] * unlabelled
    a = ad.AnnData(np.zeros((len(mt), 2), dtype="float32"))
    a.obs_names = [f"C{i}" for i in range(a.n_obs)]
    a.obs["cell_type"] = pd.Categorical(labels, categories=["LNCaP", "CAF", "rare"])
    a.obs["leiden"] = pd.Categorical([str(i % 4) for i in range(a.n_obs)])
    a.uns["scview_active_clustering"] = "leiden"
    cells = {"pct_mt": mt, "has_mt": True, "total_counts": np.ones(len(mt)), "n_genes": np.ones(len(mt))}
    return a, cells


def _rows(r):
    return {t.group: t for t in r.types}


def test_high_baseline_type_is_flagged_at_a_10_percent_ceiling():
    a, cells = _synthetic()
    r = mbt.mito_by_type(a, cells=cells, ceiling=10.0)
    assert r.column == "cell_type" and r.reason == "cell type"
    assert r.ceiling == 10.0 and r.ceiling_source == "requested" and r.measurable
    rows = _rows(r)
    assert [t.group for t in r.types] == ["LNCaP", "CAF", "rare"]
    lncap, caf = rows["LNCaP"], rows["CAF"]
    assert lncap.flagged and lncap.median == pytest.approx(12.1, abs=0.3)
    assert lncap.removed_global_frac > 0.8  # most of a ~12 % type sits above 10 %
    assert lncap.bound == pytest.approx(lncap.median + 3 * lncap.mad, abs=1e-2)
    assert lncap.removed_bound_frac < 0.1
    assert "sits below its median" in lncap.reasons[0] and "10.0%" in lncap.reasons[0]
    assert not caf.flagged and caf.removed_global == 0 and caf.reasons == []
    assert set(lncap.quantiles) == {"q05", "q25", "q75", "q95"}
    assert r.n_flagged == 1
    assert r.summary.startswith("A 10.0% ceiling removes") and "of LNCaP (median 12." in r.summary
    assert "its own bound" in r.summary
    assert r.removed_global_total == sum(t.removed_global for t in r.types)


def test_loss_share_flag_without_the_ceiling_below_the_median():
    a, cells = _synthetic()
    lncap = np.asarray(a.obs["cell_type"] == "LNCaP")
    cells["pct_mt"][lncap] = np.linspace(4.0, 14.0, lncap.sum())  # median 9, 40 % above 10
    t = _rows(mbt.mito_by_type(a, cells=cells, ceiling=10.0))["LNCaP"]
    assert t.median < 10 and t.removed_global_frac > mbt.LOSS_FLAG and t.flagged
    assert t.reasons[0].startswith("The 10.0% ceiling removes 40%")


def test_tiny_groups_and_unlabelled_cells():
    a, cells = _synthetic(unlabelled=20)
    r = mbt.mito_by_type(a, cells=cells, ceiling=10.0)
    rare = _rows(r)["rare"]
    # 10 cells at 15 %: every one is above the ceiling, but too few for a bound or a flag.
    assert rare.n_cells == 10 and rare.removed_global == 10 and rare.bound is None
    assert not rare.flagged and "too few" in rare.note
    assert r.n_labelled == 710 and "20 cells have no 'cell_type' label" in r.note
    assert sum(t.n_cells for t in r.types) == 710


def test_no_spread_gives_no_bound():
    a, cells = _synthetic(unlabelled=0)
    cells["pct_mt"][np.asarray(a.obs["cell_type"] == "CAF")] = 2.0
    caf = _rows(mbt.mito_by_type(a, cells=cells))["CAF"]
    assert caf.mad == 0 and caf.bound is None and "MAD 0" in caf.note


def test_unmeasurable_mito_is_said_not_reported_as_zero():
    a, cells = _synthetic()
    cells["has_mt"] = False
    r = mbt.mito_by_type(a, cells=cells)
    assert r.measurable is False and r.types == [] and r.ceiling is None
    assert "can't be measured" in r.note
    lines, detail = mbt.grounding_lines(type("A", (), {"adata": a, "qc_per_cell": lambda self: cells})())
    assert "not measurable" in lines[0] and "[result:mito_by_type]" in lines[0]
    assert detail.endswith("not measurable")


def test_ceiling_recorded_else_default():
    a, cells = _synthetic()
    assert mbt.DEFAULT_CEILING == PipelineParams().max_pct_mt
    r = mbt.mito_by_type(a, cells=cells)
    assert r.ceiling == 20.0 and r.ceiling_source == "default" and not r.already_filtered
    provenance.record_step(a, step="filtering", tool="scview",
                           params={"max_pct_mt": 10.0, "qc_threshold_mode": "mad"})
    r = mbt.mito_by_type(a, cells=cells)
    assert r.ceiling == 10.0 and r.ceiling_source == "recorded" and r.filter_mode == "mad"
    assert r.already_filtered and "MAD mode" in r.note
    with pytest.raises(ValueError):
        mbt.mito_by_type(a, cells=cells, ceiling=-1)


def test_default_column_prefers_author_types_then_clustering():
    a, _ = _synthetic()
    assert mbt.default_column(a) == ("cell_type", "cell type")
    b = a.copy()
    b.obs = b.obs.rename(columns={"cell_type": "paper_clusters"})  # named labels: cell types
    assert mbt.default_column(b) == ("paper_clusters", "cell type")
    c = a.copy()
    del c.obs["cell_type"]
    c.obs["clusters"] = pd.Categorical([str(i % 3) for i in range(c.n_obs)])  # numbered: not types
    assert mbt.default_column(c) == ("leiden", "clustering")
    with pytest.raises(ValueError):
        mbt.mito_by_type(c, "nope", cells=_synthetic()[1])
    del c.obs["leiden"], c.obs["clusters"]
    c.uns.pop("scview_active_clustering")
    r = mbt.mito_by_type(c, cells=_synthetic()[1])
    assert r.column is None and "No cell types" in r.note


def test_grounding_line_and_citation_family():
    a, cells = _synthetic()
    provenance.record_step(a, step="filtering", tool="scview",
                           params={"max_pct_mt": 10.0, "qc_threshold_mode": "fixed"})
    adaptor = type("A", (), {"adata": a, "qc_per_cell": lambda self: cells})()
    lines, detail = mbt.grounding_lines(adaptor)
    assert len(lines) == 1 and "recorded ceiling 10.0%" in lines[0]
    assert "Flagged types: LNCaP (" in lines[0] and lines[0].endswith("[result:mito_by_type]")
    assert detail.startswith("cell_type: 1 of 3 types flagged")
    assert resolve_tag("mito_by_type", ["result:mito_by_type", "result:qc:leiden"]) == ["result:mito_by_type"]


def _register(dm, did: str = "mito1") -> str:
    """Counts with MT- genes: type A ~12 % mito, type B ~3 %."""
    rng = np.random.default_rng(1)
    def block(n, mt_lam):
        return np.hstack([rng.poisson(1.0, (n, 200)), rng.poisson(mt_lam, (n, 10))])
    X = np.vstack([block(200, 2.7), block(200, 0.6)]).astype("float32")
    a = ad.AnnData(X)
    a.var_names = [f"G{i}" for i in range(200)] + [f"MT-{i}" for i in range(10)]
    a.obs_names = [f"C{i}" for i in range(a.n_obs)]
    a.obs["cell_type"] = pd.Categorical(["A"] * 200 + ["B"] * 200)
    d = Path(dm.data_dir) / "uploads" / did
    d.mkdir(parents=True, exist_ok=True)
    (d / "metadata.json").write_text(json.dumps({"id": did, "name": did, "status": "ready"}))
    a.write_h5ad(d / "data.h5ad")
    return did


async def test_mito_by_type_endpoint(client, dataset_manager):
    did = _register(dataset_manager)
    r = await client.get(f"{API}/datasets/{did}/qc/mito-by-type", params={"ceiling": 10})
    assert r.status_code == 200, r.text
    body = r.json()
    rows = {t["group"]: t for t in body["types"]}
    assert body["column"] == "cell_type" and body["measurable"] is True
    assert rows["A"]["flagged"] and not rows["B"]["flagged"]
    assert body["summary"].startswith("A 10.0% ceiling removes")
    default = (await client.get(f"{API}/datasets/{did}/qc/mito-by-type")).json()
    assert default["ceiling"] == 20.0 and default["ceiling_source"] == "default"
    bad = await client.get(f"{API}/datasets/{did}/qc/mito-by-type", params={"column": "nope"})
    assert bad.status_code == 404
    assert (await client.get(f"{API}/datasets/ghost/qc/mito-by-type")).status_code == 404


def test_a_type_of_dying_cells_is_called_damaged_not_given_a_bound():
    """pbmc_1k_v3: a 57-cell cluster at a median 96.5 % mito got its own bound of 104 %."""
    a, cells = _synthetic(unlabelled=0)
    rare = np.asarray(a.obs["cell_type"] == "LNCaP")
    cells["pct_mt"][rare] = np.random.default_rng(2).normal(96.0, 2.0, rare.sum()).clip(0, 100)
    r = mbt.mito_by_type(a, cells=cells)
    t = _rows(r)["LNCaP"]
    assert t.damaged and not t.flagged and t.bound is None and t.removed_bound is None
    assert t.removed_global == 400 and "damaged" in t.note
    assert r.n_flagged == 0 and r.summary.endswith("LNCaP (median 96.0%) looks damaged, not a high-baseline type.")
    assert "no type is flagged" in r.summary


def test_the_unflagged_summary_names_a_type_large_enough_to_judge():
    a, cells = _synthetic(unlabelled=0)
    r = mbt.mito_by_type(a, cells=cells, ceiling=14.0)  # the 10-cell type at 15 % loses all
    assert _rows(r)["rare"].removed_global_frac == 1.0 and not _rows(r)["rare"].flagged
    assert "rare" not in r.summary and "no type is flagged" in r.summary


def test_after_filtering_the_record_adds_the_removed_cells_back():
    """The ceiling already ran: the removed cells come back from the step's record,
    so LNCaP is still flagged (and a type it emptied is still a row)."""
    from scview.core import qc_filter

    a, cells = _synthetic()
    a.obs["cell_type"] = a.obs["cell_type"].cat.add_categories(["gone"])
    a.obs.iloc[700:710, a.obs.columns.get_loc("cell_type")] = "gone"  # the 15 % "rare" cells
    drop = cells["pct_mt"] > 10.0
    t = qc_filter.Thresholds.from_params(PipelineParams(qc_threshold_mode="fixed", min_genes=0,
                                                        max_pct_mt=10.0))
    rec = qc_filter.removal_record(a, cells, {"max_pct_mt": drop}, drop, t, None)
    kept = a[~drop].copy()
    kept.uns[qc_filter.RECORD_KEY] = rec
    kc = {k: (v[~drop] if isinstance(v, np.ndarray) else v) for k, v in cells.items()}

    r = mbt.mito_by_type(kept, cells=kc, ceiling=10.0)
    rows = _rows(r)
    assert r.includes_removed == int(drop[:710].sum())  # the labelled removed cells
    assert rows["LNCaP"].n_cells == 400 and rows["LNCaP"].flagged
    assert rows["gone"].n_cells == 10 and rows["gone"].removed_global == 10
    assert "before that filter" in r.note

    without = mbt.mito_by_type(a[~drop].copy(), cells=kc, ceiling=10.0)  # no record
    assert without.includes_removed == 0 and not _rows(without)["LNCaP"].flagged

    stale = kept[:-1].copy()  # a cell removed since: the record no longer adds up
    sc = {k: (v[:-1] if isinstance(v, np.ndarray) else v) for k, v in kc.items()}
    assert mbt.mito_by_type(stale, cells=sc, ceiling=10.0).includes_removed == 0
