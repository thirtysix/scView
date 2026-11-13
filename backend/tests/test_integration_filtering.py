"""Filtering in the joint run (core/integration.py ``judge_unit`` / ``filter_units``).

The joint pipeline has no filtering step, so a dataset scView never filtered (a raw CELLxGENE
slide) brought its near-empty cells along, and they made clusters of their own. Such datasets
are now filtered first with a single run's QC; datasets filtered before are left alone; the
plan says per dataset what it removes, the run can switch it off, and the joint dataset's
recipe (and replay script) record it.
"""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from scview.core import provenance
from scview.core.dataset_manager import DatasetManager
from scview.core.integration import build_plan, load_units, run_integration_streamed

API = "/api/v1"


def _counts(seed: int, n_full: int = 240, n_empty: int = 0, genes: int = 600) -> ad.AnnData:
    """Cells with a few hundred genes each, plus ``n_empty`` near-empty ones (a handful of counts)."""
    rng = np.random.default_rng(seed)
    full = rng.poisson(rng.uniform(0.3, 1.5, genes), size=(n_full, genes))
    empty = rng.poisson(0.01, size=(n_empty, genes))
    X = sp.csr_matrix(np.vstack([full, empty]).astype(np.float32))
    n = n_full + n_empty
    obs = pd.DataFrame({"cell_type": pd.Categorical(rng.choice(["T", "B"], n))},
                       index=[f"s{seed}_c{i}" for i in range(n)])
    return ad.AnnData(X, obs=obs, var=pd.DataFrame(index=[f"G{i}" for i in range(genes)]))


def _add(data_dir: Path, did: str, a: ad.AnnData) -> None:
    up = data_dir / "uploads" / did
    up.mkdir(parents=True, exist_ok=True)
    (up / "metadata.json").write_text(json.dumps({"id": did, "name": did.upper(), "status": "ready"}))
    a.write_h5ad(up / "data.h5ad")


def _three(tmp_path: Path) -> DatasetManager:
    """raw: never filtered, 60 near-empty cells; done: filtered by scView before (with the same
    kind of cells, which it keeps); clean: never filtered, nothing to remove."""
    _add(tmp_path, "raw", _counts(1, n_empty=60))
    done = _counts(2, n_empty=60)
    provenance.record_step(done, step="filtering", tool="scview.qc_filter",
                           params={"qc_threshold_mode": "mad"})
    _add(tmp_path, "done", done)
    _add(tmp_path, "clean", _counts(3))
    return DatasetManager(str(tmp_path))


def test_the_plan_says_per_dataset_what_the_run_filters(tmp_path):
    dm = _three(tmp_path)
    checks, units = load_units(dm, ["raw", "done", "clean"])
    plan = build_plan(checks, units, max_datasets=32, max_cells=10**6)
    rows = {r.label: r for r in plan.filtering}
    assert rows["RAW"].n_removed >= 60 and not rows["RAW"].filtered_before
    assert rows["RAW"].mode == "mad" and rows["RAW"].genes_floor == 200.0
    assert rows["RAW"].by_criterion.get("min_genes", 0) >= 60
    assert rows["DONE"].filtered_before and rows["DONE"].n_removed == 0
    assert "left as it is" in rows["DONE"].note
    assert rows["CLEAN"].n_removed < 5 and rows["CLEAN"].n_cells == 240


def _run(tmp_path: Path, dm: DatasetManager, did: str, **kw):
    _, units = load_units(dm, ["raw", "done", "clean"])
    removed = units["RAW"].qc.n_removed + units["CLEAN"].qc.n_removed
    events = list(run_integration_streamed(
        units, dataset_id=did, name="joint", data_dir=tmp_path,
        output_path_for=lambda src: dm.derived_h5ad_path(did, src),
        overrides={"n_top_genes": 100, "n_comps": 10}, **kw))
    assert "step_error" not in [e for e, _ in events], [d for e, d in events if e == "step_error"]
    return events, ad.read_h5ad(tmp_path / "derived" / did / "joint.h5ad"), removed


def test_the_run_filters_the_unfiltered_datasets_and_records_it(tmp_path):
    dm = _three(tmp_path)
    events, out, removed = _run(tmp_path, dm, "j1")
    assert ("stage", "filter") in [(e, d.get("stage")) for e, d in events]
    assert out.n_obs == 3 * 240 + 2 * 60 - removed + 0  # done keeps its 60 near-empty cells
    counts = out.obs["dataset"].value_counts()
    assert counts["DONE"] == 300 and counts["RAW"] < 245
    entry = next(h for h in provenance.read_provenance(out)["history"] if h["step"] == "integrate")
    rec = entry["params"]["unit_filtering"]
    assert rec["on"] is True
    rows = {r["label"]: r for r in rec["units"]}
    assert rows["RAW"]["n_removed"] >= 60 and rows["RAW"]["params"]["qc_genes_floor"] == 200.0
    assert "filtered_before" in rows["DONE"]
    # The replay script says what was filtered before the merge.
    from scview.core.replay_script import build_script

    script = build_script(out)
    assert "Datasets scView never filtered were filtered before merging" in script
    assert f"#   RAW: {rows['RAW']['n_removed']:,} of 300 cells removed (mad thresholds)" in script
    assert "#   DONE: filtered before, left as it was" in script


def test_the_run_can_leave_them_unfiltered(tmp_path):
    dm = _three(tmp_path)
    events, out, _ = _run(tmp_path, dm, "j2", filter_units_first=False)
    assert "filter" not in [d.get("stage") for e, d in events if e == "stage"]
    assert out.n_obs == 3 * 240 + 2 * 60
    entry = next(h for h in provenance.read_provenance(out)["history"] if h["step"] == "integrate")
    assert entry["params"]["unit_filtering"]["on"] is False


async def test_the_plan_endpoint_carries_the_filtering(client, dataset_manager):
    _three(Path(dataset_manager.data_dir))
    r = await client.post(f"{API}/integrate/plan", json={"dataset_ids": ["raw", "done", "clean"]})
    assert r.status_code == 200, r.text
    rows = {f["label"]: f for f in r.json()["filtering"]}
    assert rows["RAW"]["n_removed"] >= 60 and rows["DONE"]["filtered_before"] is True
