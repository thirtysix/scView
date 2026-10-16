"""Batch x condition confounding (core/confounding.py) and where it is reported."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd

from scview.api.v1.de import DECompareRequest, differential_expression_compare
from scview.core import provenance
from scview.core.confounding import check, for_comparison, in_design
from scview.core.integration import Unit, confounding_notes


def _obs(batch: list[str], cond: list[str], n: int = 40) -> pd.DataFrame:
    """``n`` cells per (batch, condition) pair listed."""
    rows = [(b, c) for b, c in zip(batch, cond) for _ in range(n)]
    return pd.DataFrame(rows, columns=["batch", "condition"],
                        index=[f"c{i}" for i in range(len(rows))])


def test_one_to_one_is_confounded_and_nested_is_nested():
    hit = check(_obs(["b1", "b2"], ["ctrl", "stim"]), "batch", "condition")
    assert hit is not None and hit.kind == "confounded"
    assert "can't be told apart" in hit.message
    hit = check(_obs(["b1", "b2", "b3", "b4"], ["ctrl", "ctrl", "stim", "stim"]), "batch", "condition")
    assert hit is not None and hit.kind == "nested" and hit.n_batches == 4


def test_crossed_designs_and_a_single_condition_are_fine():
    crossed = _obs(["b1", "b1", "b2", "b2"], ["ctrl", "stim", "ctrl", "stim"])
    assert check(crossed, "batch", "condition") is None
    assert check(_obs(["b1", "b2"], ["ctrl", "ctrl"]), "batch", "condition") is None


def test_a_few_stray_cells_do_not_hide_confounding():
    obs = pd.concat([_obs(["b1", "b2"], ["ctrl", "stim"], n=500),
                     _obs(["b1"], ["stim"], n=3)], ignore_index=True)
    assert check(obs, "batch", "condition").kind == "confounded"


def test_comparison_between_datasets_is_flagged():
    obs = _obs(["CTRL", "STIM"], ["CTRL", "STIM"]).rename(columns={"batch": "dataset", "condition": "stim"})
    mask = np.ones(len(obs), dtype=bool)
    # Split by the dataset itself, or by a condition that lines up with it.
    assert for_comparison(obs, "dataset", mask).batch == "dataset"
    hit = for_comparison(obs, "stim", mask)
    assert hit is not None and (hit.batch, hit.condition) == ("dataset", "stim")
    # Kang-like: donors carry both conditions and there is no batch column.
    kang = _obs(["d1", "d1", "d2", "d2"], ["CTRL", "STIM", "CTRL", "STIM"]).rename(
        columns={"batch": "Donor", "condition": "stim"})
    assert for_comparison(kang, "stim", np.ones(len(kang), dtype=bool)) is None


def _adata(obs: pd.DataFrame) -> ad.AnnData:
    rng = np.random.default_rng(0)
    a = ad.AnnData(rng.poisson(2.0, (len(obs), 30)).astype("float32"), obs=obs.copy())
    a.var_names = [f"G{i}" for i in range(30)]
    return a


def test_a_condition_used_as_the_batch_key_is_reported():
    obs = _obs(["d1", "d1", "d2", "d2"], ["CTRL", "STIM", "CTRL", "STIM"]).rename(
        columns={"batch": "Donor", "condition": "stim"})
    a = _adata(obs)
    provenance.record_step(a, step="batch_correction", tool="harmony",
                           params={"batch_key": "stim", "batch_method": "harmony"})
    hits = in_design(a)
    assert [(h.batch, h.role) for h in hits] == [("stim", "correction")]
    assert "removes the 'stim' difference" in hits[0].message


async def test_de_between_groups_reports_the_confounding():
    obs = _obs(["CTRL", "STIM"], ["CTRL", "STIM"]).rename(columns={"batch": "dataset", "condition": "stim"})
    a = _adata(obs)

    class _DM:
        async def get_or_load_dataset(self, _id):
            return type("A", (), {"adata": a})()

    r = await differential_expression_compare(
        "x", DECompareRequest(split_by="stim", group_a="STIM", group_b="CTRL"), _DM())
    assert r.confounding is not None and r.confounding["batch"] == "dataset"


def test_integration_plan_notes_datasets_that_are_conditions():
    def unit(label: str, cond: str) -> Unit:
        a = _adata(pd.DataFrame({"stim": [cond] * 40}, index=[f"{label}{i}" for i in range(40)]))
        return Unit(dataset_id=label, name=label, label=label, adata=a, counts_source="X")

    notes = confounding_notes({"CTRL": unit("CTRL", "CTRL"), "STIM": unit("STIM", "STIM")})
    assert len(notes) == 1 and "'stim'" in notes[0] and "pseudobulk" in notes[0]
    assert confounding_notes({"A": unit("A", "CTRL"), "B": unit("B", "CTRL")}) == []
