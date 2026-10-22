"""Doublet calls: hashing first when it exists, Scrublet the second opinion
(core/doublet_source.py), and the filtering step's source choice and hashing
negatives (core/qc_filter.py), recorded and replayed.

The fixture's answers are counted by hand: 20 cells, hashing calls cells 0-3
doublets and 4-5 negatives, Scrublet calls cells 2, 3, 4 and 10. So both call
2 and 3; hashing alone 0 and 1; Scrublet alone 4 (a hashing negative) and 10.
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

from scview.core import cluster_qc, doublet_source, doublets, hashing, provenance, qc_filter, replay_script
from scview.core.anndata_adaptor import per_cell_qc
from scview.core.pipeline import PipelineParams, _run_filtering, run_pipeline

API = "/api/v1"
HASH_DOUBLETS = [0, 1, 2, 3]
NEGATIVES = [4, 5]
SCRUBLET = [2, 3, 4, 10]


def _mask(idx, n=20) -> np.ndarray:
    m = np.zeros(n, dtype=bool)
    m[idx] = True
    return m


def _obs(hash_class: bool = True, scrublet: bool = True) -> pd.DataFrame:
    obs = pd.DataFrame(index=[f"c{i}" for i in range(20)])
    if hash_class:
        cls = np.array(["Singlet"] * 20, dtype=object)
        cls[HASH_DOUBLETS] = "Doublet"
        cls[NEGATIVES] = "Negative"
        obs["hash_class"] = pd.Categorical(cls, categories=["Singlet", "Doublet", "Negative"])
    if scrublet:
        obs["predicted_doublet"] = _mask(SCRUBLET)
    return obs


def _adata(hash_class: bool = True, scrublet: bool = True) -> ad.AnnData:
    rng = np.random.default_rng(0)
    a = ad.AnnData(rng.poisson(2.0, size=(20, 30)).astype("float32"), obs=_obs(hash_class, scrublet))
    a.var_names = [f"G{i}" for i in range(30)]
    a.obs["sample"] = pd.Categorical(["s1"] * 10 + ["s2"] * 10)
    return a


def test_hash_class_is_the_column_hashing_writes():
    assert doublet_source.HASH_CLASS == hashing.CLASS


@pytest.mark.parametrize("source, expected, used", [
    ("auto", HASH_DOUBLETS, "hashing"),
    ("hashing", HASH_DOUBLETS, "hashing"),
    ("scrublet", SCRUBLET, "scrublet"),
    ("either", [0, 1, 2, 3, 4, 10], "either"),
    ("legacy", SCRUBLET, "scrublet"),  # before the choice: Scrublet's column first
])
def test_both_calls_each_source(source, expected, used):
    got = doublet_source.resolve(_obs(), source)
    assert got.source == used
    assert np.flatnonzero(got.calls).tolist() == expected


@pytest.mark.parametrize("source, expected", [
    ("auto", HASH_DOUBLETS), ("hashing", HASH_DOUBLETS), ("scrublet", None), ("either", HASH_DOUBLETS),
])
def test_hashing_only(source, expected):
    got = doublet_source.resolve(_obs(scrublet=False), source)
    assert (got is None) if expected is None else np.flatnonzero(got.calls).tolist() == expected


@pytest.mark.parametrize("source, expected", [
    ("auto", SCRUBLET), ("hashing", None), ("scrublet", SCRUBLET), ("either", SCRUBLET),
])
def test_scrublet_only(source, expected):
    got = doublet_source.resolve(_obs(hash_class=False), source)
    assert (got is None) if expected is None else np.flatnonzero(got.calls).tolist() == expected


def test_negatives_are_not_doublets():
    obs = _obs()
    assert not doublet_source.hashing_calls(obs)[NEGATIVES].any()
    assert np.flatnonzero(doublet_source.hash_negatives(obs)).tolist() == NEGATIVES
    assert doublet_source.hash_negatives(_obs(hash_class=False)) is None
    # A cell hashing never called (NaN) is neither.
    obs.loc["c6", "hash_class"] = np.nan
    assert not doublet_source.hashing_calls(obs)[6] and not doublet_source.hash_negatives(obs)[6]


def test_text_typed_calls_after_a_round_trip(tmp_path):
    """Calls stored as text ("True"/"False", "Doublet") read the same as the bools;
    astype(bool) would make every non-empty string True."""
    a = _adata()
    a.obs["predicted_doublet"] = pd.Categorical(np.where(_mask(SCRUBLET), "True", "False"))
    a.obs["hash_class"] = a.obs["hash_class"].astype(str)  # plain text, not categorical
    a.write_h5ad(tmp_path / "x.h5ad")
    obs = ad.read_h5ad(tmp_path / "x.h5ad").obs
    assert np.flatnonzero(doublet_source.scrublet_calls(obs)).tolist() == SCRUBLET
    assert np.flatnonzero(doublet_source.resolve(obs, "either").calls).tolist() == [0, 1, 2, 3, 4, 10]
    assert np.flatnonzero(doublet_source.hash_negatives(obs)).tolist() == NEGATIVES
    obs["predicted_doublet"] = obs["predicted_doublet"].astype(str).str.lower()  # object "true"/"false"
    assert np.flatnonzero(doublet_source.scrublet_calls(obs)).tolist() == SCRUBLET


def test_other_tools_text_columns_still_count():
    obs = pd.DataFrame({"scDblFinder.class": ["singlet", "doublet", "singlet"]})
    got = doublet_source.resolve(obs)
    assert got.source == "scDblFinder.class" and got.calls.tolist() == [False, True, False]
    assert doublet_source.resolve(obs, "scrublet") is None and doublet_source.resolve(obs, "hashing") is None
    with pytest.raises(ValueError):
        doublet_source.resolve(obs, "nope")


def test_describe_and_text_by_hand():
    info = doublet_source.describe(_obs())
    assert info["source"] == "hashing" and info["n_doublets"] == 4 and info["fraction"] == 0.2
    assert info["scrublet"]["n_doublets"] == 4 and info["both"]["n"] == 2
    assert info["hashing"]["n_negatives"] == 2
    assert info["options"] == ["auto", "hashing", "scrublet", "either"]
    assert doublet_source.text(info) == ("4 cells (20.0%) are doublets by cell hashing; Scrublet flags "
                                         "4 (20.0%), 2 (10.0%) by both")
    scr = doublet_source.describe(_obs(hash_class=False))
    assert doublet_source.text(scr) == "4 cells (20.0%) are doublets by Scrublet"
    assert scr["options"] == ["auto", "scrublet"]
    assert doublet_source.describe(pd.DataFrame(index=range(3))) is None


def test_agreement_two_by_two_and_per_sample_by_hand():
    obs = _obs()
    names, codes = ["s1", "s2"], np.array([0] * 10 + [1] * 9 + [-1])  # c19: no sample
    out = doublet_source.agreement(doublet_source.hashing_calls(obs), doublet_source.scrublet_calls(obs),
                                   doublet_source.hash_negatives(obs), names, codes)
    assert (out["both"], out["hashing_only"], out["scrublet_only"], out["neither"]) == (2, 2, 2, 14)
    assert out["negatives"] == 2 and out["negatives_scrublet"] == 1  # c4
    # s1 = c0-c9: hashing 0-3, Scrublet 2, 3, 4; s2 = c10-c18: Scrublet 10.
    assert out["samples"] == [
        {"name": "s1", "n_cells": 10, "hashing": 4, "scrublet": 3, "both": 2},
        {"name": "s2", "n_cells": 9, "hashing": 0, "scrublet": 1, "both": 0},
    ]
    assert out["unassigned"] == {"n_cells": 1, "hashing": 0, "scrublet": 0, "both": 0}


def test_cluster_views_and_scrublet_card_read_their_own_calls():
    a = _adata()
    assert np.flatnonzero(cluster_qc.doublet_calls(a.obs)).tolist() == HASH_DOUBLETS
    a.obs["doublet_score"] = np.where(_mask(SCRUBLET), 0.9, 0.1)
    a.uns["scrublet"] = {"threshold": 0.5}
    s = doublets.summary(a)
    assert s["n_called_now"] == 4  # Scrublet's call, not hashing's
    assert s["source"]["source"] == "hashing"
    ag = s["agreement"]
    assert ag["threshold"] == 0.5 and ag["column"] == "sample" and ag["both"] == 2
    assert [r["hashing"] for r in ag["samples"]] == [4, 0]
    # Applying a threshold keeps Scrublet's own call, never hashing's.
    doublets.apply_threshold(a, 0.05)
    assert np.flatnonzero(a.obs["predicted_doublet_scrublet"]).tolist() == SCRUBLET


# ---------------------------------------------------------------------------
# The filtering step: the source choice and hashing negatives
# ---------------------------------------------------------------------------


def _filtered(**kw):
    params = PipelineParams(qc_threshold_mode="fixed", min_genes=0, min_cells=0, **kw)
    return _run_filtering(_adata(), params)


@pytest.mark.parametrize("kw, dropped", [
    ({"drop_doublets": True}, HASH_DOUBLETS),  # auto: hashing, since it exists
    ({"drop_doublets": True, "qc_doublet_source": "hashing"}, HASH_DOUBLETS),
    ({"drop_doublets": True, "qc_doublet_source": "scrublet"}, SCRUBLET),
    ({"drop_doublets": True, "qc_doublet_source": "either"}, [0, 1, 2, 3, 4, 10]),
    ({"drop_doublets": True, "drop_hash_negatives": True}, [0, 1, 2, 3, 4, 5]),
    ({"drop_hash_negatives": True}, NEGATIVES),
    ({}, []),
])
def test_filtering_drops_each_choice(kw, dropped):
    out, found = _filtered(**kw)
    assert out.obs_names.tolist() == [f"c{i}" for i in range(20) if i not in dropped]
    if kw.get("drop_doublets"):
        assert found["qc_doublet_source"] == kw.get("qc_doublet_source", "auto")
        assert found["qc_doublet_source_used"] == {"auto": "hashing"}.get(
            kw.get("qc_doublet_source", "auto"), kw.get("qc_doublet_source"))
    else:
        assert "qc_doublet_source" not in found
    assert found.get("drop_hash_negatives", False) is bool(kw.get("drop_hash_negatives"))


def test_without_hashing_auto_is_scrublet_and_hashing_drops_nothing():
    a = _adata(hash_class=False)
    params = PipelineParams(qc_threshold_mode="fixed", min_genes=0, min_cells=0, drop_doublets=True)
    out, found = _run_filtering(a.copy(), params)
    assert out.obs_names.tolist() == [f"c{i}" for i in range(20) if i not in SCRUBLET]
    assert found["qc_doublet_source_used"] == "scrublet"
    params.qc_doublet_source, params.drop_hash_negatives = "hashing", True
    out, found = _run_filtering(a.copy(), params)
    assert out.n_obs == 20 and found["qc_doublet_source_used"] is None


def test_preview_says_whose_doublets_and_matches_the_step():
    a = _adata()
    t = qc_filter.Thresholds(drop_doublets=True, doublet_source="either", drop_hash_negatives=True)
    p = qc_filter.preview(a, per_cell_qc(a), t)
    assert p["n_removed"] == 7  # 0-5 and 10
    assert p["by_criterion"]["doublets"]["n_removed"] == 6
    assert p["by_criterion"]["hash_negatives"] == {"label": "Hashing negatives", "n_removed": 2}
    d = p["doublets"]
    assert d["source"] == "either" and d["used"] == {"source": "either", "label": "cell hashing or Scrublet",
                                                     "n": 6}
    assert d["auto"]["source"] == "hashing" and d["both"] == 2 and d["hash_negatives"] == 2
    assert d["n_by_source"] == {"auto": 4, "hashing": 4, "scrublet": 4, "either": 6}
    out, _ = _run_filtering(a.copy(), PipelineParams(qc_threshold_mode="fixed", min_genes=0, min_cells=0,
                                                     drop_doublets=True, qc_doublet_source="either",
                                                     drop_hash_negatives=True))
    assert out.n_obs == 20 - p["n_removed"]
    with pytest.raises(ValueError, match="doublet source"):
        qc_filter.preview(a, per_cell_qc(a), qc_filter.Thresholds(doublet_source="nope"))
    none = _adata(False, False)
    assert qc_filter.preview(none, per_cell_qc(none), t)["doublets"] is None


def test_removal_record_keeps_the_negatives_criterion():
    out, _ = _filtered(drop_doublets=True, drop_hash_negatives=True)
    names = qc_filter.recorded_names(out)
    neg = qc_filter.recorded_bits(out, "hash_negatives")
    assert [n for n, b in zip(names, neg) if b] == ["c4", "c5"]


# ---------------------------------------------------------------------------
# Replay: the recorded source, and old recipes as before
# ---------------------------------------------------------------------------


def _old_doublet_calls(obs):
    """v0.32.0's doublet_calls, verbatim: the oracle for old recipes."""
    if "predicted_doublet" in obs.columns:
        col = obs["predicted_doublet"]
        if pd.api.types.is_bool_dtype(col):
            return col.to_numpy(dtype=bool)
        return col.astype(str).str.lower().isin(("true", "1", "doublet")).to_numpy()
    names = [c for c in ("scDblFinder.class", "scDblFinder_class", "multiplets", "doublet_class",
                         "hash_class") if c in obs.columns]
    names += [c for c in obs.columns if str(c).startswith("DF.classification")]
    for name in names:
        return obs[name].astype(str).str.lower().eq("doublet").to_numpy()
    return None


def _helper_ns() -> dict:
    ns: dict = {"np": np, "pd": pd}
    exec(replay_script._HELPERS["doublet_calls"], ns)
    return ns


def _cases() -> list[pd.DataFrame]:
    both = _obs()
    text = both.copy()
    text["predicted_doublet"] = text["predicted_doublet"].map({True: "True", False: "False"})
    finder = pd.DataFrame({"DF.classification_x": ["Singlet", "Doublet"] * 10}, index=both.index)
    finder_h = finder.assign(hash_class=both["hash_class"].to_numpy())
    named_h = finder_h.assign(**{"scDblFinder.class": ["doublet"] + ["singlet"] * 19})
    return [both, text, _obs(scrublet=False), _obs(hash_class=False), finder, finder_h, named_h,
            pd.DataFrame(index=both.index)]


def test_replay_helper_matches_the_app_for_every_source():
    ns = _helper_ns()
    for obs in _cases():
        for source in (*doublet_source.SOURCES, "legacy"):
            app = doublet_source.resolve(obs, source)
            rep = ns["doublet_calls"](obs, source)
            assert (rep is None) == (app is None), (source, list(obs.columns))
            if app is not None:
                assert (rep == app.calls).all(), (source, list(obs.columns))
        old = _old_doublet_calls(obs)
        legacy = ns["doublet_calls"](obs)  # the default: an old recipe
        assert (legacy is None) == (old is None) and (old is None or (legacy == old).all())
        neg = doublet_source.hash_negatives(obs)
        assert (ns["hash_negatives"](obs) is None) == (neg is None)


def _with_history(steps: list[tuple[str, dict]]) -> ad.AnnData:
    a = ad.AnnData(np.zeros((3, 2), dtype="float32"))
    for name, params in steps:
        provenance.record_step(a, step=name, tool="t", params=params)
    return a


def test_old_recipe_replays_scrublet_and_its_manifest_is_unchanged():
    """A filtering recorded before the choice (no qc_doublet_source) replays the
    old order, Scrublet first, even when the cells now have hashing calls."""
    from scview.core import run_manifest

    old = {"qc_threshold_mode": "fixed", "min_genes": 0, "max_genes": 0, "min_counts": 0,
           "min_cells": 0, "max_pct_mt": 20.0, "drop_doublets": True}
    snippet = replay_script._filtering(old, replay_script._Ctx(adata=None))
    assert "calls = doublet_calls(adata.obs)" in snippet.code and snippet.status == "exact"
    ns = _helper_ns()
    exec("\n".join(c for c in snippet.code if c.startswith("calls =")), {**ns, "adata": _adata()}, ns)
    assert np.flatnonzero(ns["calls"]).tolist() == SCRUBLET
    a = _with_history([("filtering", old)])
    m = run_manifest.build_manifest(a, dataset_id="x", dataset_name="x")
    assert m["steps"][0]["params"] == old and m["steps"][0]["replay"] == {"status": "exact", "note": ""}
    new = replay_script._filtering({**old, "qc_doublet_source": "auto", "qc_doublet_source_used": "hashing"},
                                   replay_script._Ctx(adata=None))
    assert "calls = doublet_calls(adata.obs, 'hashing')" in new.code


def _replay(tmp_path: Path, inp: ad.AnnData, out: ad.AnnData) -> ad.AnnData:
    src, script, dst = tmp_path / "input.h5ad", tmp_path / "replay.py", tmp_path / "replayed.h5ad"
    inp.write_h5ad(src)
    script.write_text(replay_script.build_script(out, dataset_id="t", dataset_name="test"))
    proc = subprocess.run([sys.executable, str(script), str(src), str(dst)], cwd=tmp_path,
                          capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr[-3000:]
    return ad.read_h5ad(dst)


@pytest.mark.parametrize("kw", [
    {"qc_doublet_source": "either", "drop_hash_negatives": True},
    {"qc_doublet_source": "auto"},
    {"qc_doublet_source": "scrublet"},
])
def test_replay_drops_the_same_cells(tmp_path, kw):
    inp = _adata()
    params = PipelineParams(qc_threshold_mode="fixed", min_genes=0, min_cells=0, drop_doublets=True, **kw)
    out, res = run_pipeline(inp.copy(), ["filtering"], params)
    assert res.errors == {} and out.n_obs < 20
    rep = _replay(tmp_path, inp, out)
    assert rep.obs_names.tolist() == out.obs_names.tolist()


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


def _register(dm, a: ad.AnnData, did: str = "dbs") -> str:
    d = Path(dm.data_dir) / "uploads" / did
    d.mkdir(parents=True)
    (d / "metadata.json").write_text(json.dumps({"id": did, "name": did, "status": "ready"}))
    a.write_h5ad(d / "data.h5ad")
    return did


async def test_filter_preview_takes_the_source_and_negatives(client, dataset_manager):
    did = _register(dataset_manager, _adata())
    base = {"mode": "fixed", "min_genes": 0, "drop_doublets": True}
    body = (await client.get(f"{API}/datasets/{did}/qc/filter-preview", params=base)).json()
    assert body["by_criterion"]["doublets"]["n_removed"] == 4 and body["doublets"]["used"]["source"] == "hashing"
    body = (await client.get(f"{API}/datasets/{did}/qc/filter-preview",
                             params={**base, "doublet_source": "scrublet", "drop_hash_negatives": "true"})).json()
    assert body["doublets"]["source"] == "scrublet" and body["n_removed"] == 5  # 2, 3, 4, 10 and 5
    r = await client.get(f"{API}/datasets/{did}/qc/filter-preview", params={**base, "doublet_source": "x"})
    assert r.status_code == 400
