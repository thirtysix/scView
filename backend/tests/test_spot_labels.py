"""Spot composition's reference labels (batch 36): left out, merged, suggested; everywhere they travel."""

from __future__ import annotations

import json
import subprocess
import sys

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from scview.core import provenance, replay_script
from scview.core import spot_deconvolution as sd
from scview.core.rag.router import _RESULT_FACET_HINTS
from tests.large_fixture import make_large
from tests.test_hvg_sensitivity import _events, _register
from tests.test_spot_deconvolution import _fit, _prepare, _profiles, _reference, _spots

API = "/api/v1"
TYPES5 = ["B", "T", "Mono", "Stroma", "Unk_1"]


@pytest.fixture(scope="module")
def data():
    prof = _profiles(n_types=5)
    return _reference(prof, types=TYPES5), _spots(prof, n=60)[0]


def _same(a: sd.Prepared, b: sd.Prepared) -> None:
    assert a.types == b.types and a.genes == b.genes and a.small_types == b.small_types
    for x, y in ((a.mu, b.mu), (a.S, b.S), (a.n_type_cells, b.n_type_cells), (a.keep, b.keep), (a.N, b.N)):
        np.testing.assert_array_equal(x, y)
    assert (a.Y != b.Y).nnz == 0


def _same_fit(a: dict, b: dict) -> None:
    assert a["s_e"] == b["s_e"]
    np.testing.assert_array_equal(a["beta"], b["beta"])
    np.testing.assert_array_equal(a["cells"], b["cells"])
    np.testing.assert_array_equal(a["fit"], b["fit"])


# --- Suggestions ---


@pytest.mark.parametrize("label", [
    "Unknown", "unknown", "UNK", "Unk_1", "Unk2", "LowQ_2", "lowq", "Low quality", "low_quality_3", "low-quality",
    "LowQuality", "Unassigned", "unclassified", "Undefined", "Ambiguous", "Doublet", "doublets", "Debris", "NA", "N/A",
    "none", "None", "Other", "others", "other_3", "Other cells", "Unknown cell type", "Ext_Unk_3", "T_unknown",
])
def test_names_that_look_like_a_catch_all(label):
    assert sd.is_catch_all(label)


@pytest.mark.parametrize("label", [
    "Other_Neuron", "Other T cells", "Junk", "Monk", "Unknot", "Ambig", "Lowly", "Low_Mito", "NA_channel", "Natural killer",
    "Noneuronal", "Mother", "Another", "Doublecortin", "Inh_1", "T cell", "Astro_THAL_med", "Endo", "OPC_1",
])
def test_names_that_do_not(label):
    assert not sd.is_catch_all(label)


def test_suggestions_on_the_real_brain_labels_find_the_three_groups():
    names = ["Astro_AMY", "Ext_L23", "Ext_Unk_1", "Ext_Unk_2", "Ext_Unk_3", "Inh_1", "LowQ_1", "LowQ_2", "Micro", "Nb_1",
             "Oligo_2", "OPC_1", "Unk_1", "Unk_2"]
    assert sd.suggested_exclude(names) == ["Ext_Unk_1", "Ext_Unk_2", "Ext_Unk_3", "LowQ_1", "LowQ_2", "Unk_1", "Unk_2"]


# --- Left out, merged ---


def test_nothing_ticked_is_the_same_run_bit_for_bit(data):
    ref, spots = data
    plain = _prepare(ref, spots)
    empty = _prepare(ref, spots, exclude_labels=[], merge_labels={})
    _same(plain, empty)
    for method in sd.METHODS:
        _same_fit(_fit(plain, method), _fit(empty, method))
    assert plain.info["excluded_labels"] == [] and plain.info["merged_labels"] == {}
    assert plain.info["n_excluded_cells"] == 0


def test_leaving_a_label_out_is_the_reference_without_those_cells(data):
    ref, spots = data
    p = _prepare(ref, spots, exclude_labels=["Unk_1"])
    without = _prepare(ref[ref.obs["cell_type"] != "Unk_1"].copy(), spots)
    _same(p, without)
    assert "Unk_1" not in p.types and p.info["n_excluded_cells"] == 60
    assert p.info["n_reference_cells"] == without.info["n_reference_cells"] == 240
    for method in sd.METHODS:
        _same_fit(_fit(p, method), _fit(without, method))


def test_merging_is_the_column_rewritten_and_a_merged_name_joins_a_label(data):
    ref, spots = data
    merge = {"Mono": "Myeloid", "Stroma": "Myeloid", "Unk_1": "B"}
    p = _prepare(ref, spots, merge_labels=merge)
    rewritten = ref.copy()
    rewritten.obs["cell_type"] = rewritten.obs["cell_type"].astype(str).replace(merge)
    q = _prepare(rewritten, spots)
    _same(p, q)
    assert p.types == ["B", "Myeloid", "T"] and p.n_type_cells.tolist() == [120, 120, 60]
    assert p.info["merged_labels"] == merge and p.info["n_excluded_cells"] == 0
    _same_fit(_fit(p, "nnls"), _fit(q, "nnls"))


def test_merging_counts_before_the_cell_minimum_and_the_type_limit():
    prof = _profiles(n_types=4)
    ref = _reference(prof, cells=20)  # each under MIN_TYPE_CELLS
    spots, _ = _spots(prof, n=20)
    with pytest.raises(sd.SpotDeconvolutionError, match="fewer than 2 types"):
        _prepare(ref, spots)
    p = _prepare(ref, spots, merge_labels={"B": "BT", "T": "BT", "Mono": "MS", "Stroma": "MS"})
    assert p.types == ["BT", "MS"] and p.small_types == []

    many = _reference(_profiles(n_types=sd.MAX_TYPES + 2), types=[f"t_{i}" for i in range(sd.MAX_TYPES + 2)], cells=25)
    spots2, _ = _spots(_profiles(n_types=sd.MAX_TYPES + 2), n=20)
    with pytest.raises(sd.SpotDeconvolutionError, match=rf"{sd.MAX_TYPES + 2} types.*merge or leave out"):
        _prepare(many, spots2)
    assert len(_prepare(many, spots2, merge_labels={"t_0": "t_1", "t_2": "t_3"}).types) == sd.MAX_TYPES
    assert len(_prepare(many, spots2, exclude_labels=["t_0", "t_1"]).types) == sd.MAX_TYPES


def test_excluding_wins_over_merging(data):
    ref, spots = data
    # In both: left out, and the merge is not recorded.
    p = _prepare(ref, spots, exclude_labels=["Unk_1"], merge_labels={"Unk_1": "B"})
    assert p.types == ["B", "Mono", "Stroma", "T"] and p.n_type_cells.tolist() == [60, 60, 60, 60]
    assert p.info["merged_labels"] == {}
    # Merged into a left-out name: goes with it, whatever the name.
    q = _prepare(ref, spots, exclude_labels=["Unk_1"], merge_labels={"Stroma": "Unk_1"})
    assert q.types == ["B", "Mono", "T"] and q.info["n_excluded_cells"] == 120
    # Leaving out a merged name is allowed (a type of the last run may be one).
    r = _prepare(ref, spots, exclude_labels=["Rest"], merge_labels={"Stroma": "Rest", "Unk_1": "Rest"})
    assert r.types == ["B", "Mono", "T"] and r.info["excluded_labels"] == ["Rest"]


def test_label_refusals_are_one_sentence(data):
    ref, spots = data
    for kw, match in (
        ({"exclude_labels": ["Nope"]}, "has no label 'Nope'"),
        ({"merge_labels": {"Nope": "B"}}, "has no label 'Nope'"),
        ({"merge_labels": {"B": "   "}}, "cannot merge"),
        ({"merge_labels": {"B": sd.LOW_COUNTS}}, "cannot merge"),
        ({"merge_labels": {"B": "T", "T": "Mono"}}, "itself merges into 'Mono'"),
        ({"exclude_labels": ["B", "T", "Mono", "Stroma"]}, "fewer than 2 types"),
    ):
        with pytest.raises(sd.SpotDeconvolutionError, match=match) as e:
            _prepare(ref, spots, **kw)
        assert e.value.status == 422
    # A label merged into itself is no merge at all.
    assert _prepare(ref, spots, merge_labels={"B": " B "}).info["merged_labels"] == {}


def test_no_label_cells_stay_out_and_are_not_counted_as_left_out(data):
    ref, spots = data
    r = ref.copy()
    r.obs["cell_type"] = r.obs["cell_type"].astype(object).where(np.arange(r.n_obs) >= 5, None)
    p = _prepare(r, spots, exclude_labels=["Unk_1"])
    assert p.info["n_excluded_cells"] == 60 and p.info["n_reference_cells"] == 235


def test_a_large_reference_takes_the_same_labels(tmp_path, data):
    ref, spots = data
    kw = {"exclude_labels": ["Unk_1"], "merge_labels": {"Mono": "Myeloid", "Stroma": "Myeloid"}}
    with sd.large_reference(str(make_large(tmp_path / "ref", ref))) as (skel, counts):
        large = _prepare(skel, spots, ref_counts=counts, **kw)
    mem = _prepare(ref, spots, **kw)
    assert large.types == mem.types == ["B", "Myeloid", "T"]
    np.testing.assert_allclose(large.mu, mem.mu, rtol=1e-12, atol=0)
    np.testing.assert_array_equal(large.S, mem.S)
    np.testing.assert_allclose(_fit(large, "nnls")["beta"], _fit(mem, "nnls")["beta"], atol=1e-6)


# --- Where they travel ---


def test_summary_step_fact_and_replay_carry_them(tmp_path, data):
    ref, spots = data
    spots = spots.copy()
    p = _prepare(ref, spots, exclude_labels=["Unk_1"], merge_labels={"Mono": "Myeloid", "Stroma": "Myeloid"})
    out = _fit(p)
    s = sd.summarize(p, out, method="rctd")
    assert s["excluded_labels"] == ["Unk_1"] and s["merged_labels"] == {"Mono": "Myeloid", "Stroma": "Myeloid"}
    assert s["n_excluded_cells"] == 60 and s["types"] == ["B", "Myeloid", "T"]
    s["reference"]["path"] = str(tmp_path / "reference.h5ad")
    sd.save(spots, p, out, s)
    params = provenance.read_provenance(spots)["history"][-1]["params"]
    assert params["excluded_labels"] == ["Unk_1"] and params["merged_labels"] == s["merged_labels"]
    assert sd.saved(spots)["excluded_labels"] == ["Unk_1"]

    class Adaptor:
        adata = spots

    lines, _detail = sd.grounding_lines(Adaptor())
    text = "\n".join(lines)
    assert "Left out of the reference before the fit" in text and "Unk_1 (60 cells)" in text
    assert "Myeloid = Mono + Stroma" in text
    assert any("reference label" in h for h in _RESULT_FACET_HINTS["groups"])

    assert replay_script.describe(spots)[-1]["status"] == replay_script.EXACT
    inp = spots.copy()
    for key in (sd.OBSM_RNA, sd.OBSM_CELLS):
        del inp.obsm[key]
    del inp.obs[sd.OBS_TOP], inp.obs[sd.OBS_TOP_SHARE]
    inp.write_h5ad(tmp_path / "input.h5ad")
    ref.write_h5ad(tmp_path / "reference.h5ad")
    script = tmp_path / "replay.py"
    script.write_text(replay_script.build_script(spots, dataset_id="t", dataset_name="test"))
    assert "exclude=SPOT_EXCLUDE, merge=SPOT_MERGE)" in script.read_text()
    r = subprocess.run([sys.executable, str(script), str(tmp_path / "input.h5ad"), str(tmp_path / "out.h5ad")],
                       cwd=tmp_path, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-3000:]
    got = ad.read_h5ad(tmp_path / "out.h5ad")
    assert np.allclose(got.obsm[sd.OBSM_RNA], spots.obsm[sd.OBSM_RNA], atol=1e-6, equal_nan=True)
    assert (got.obs[sd.OBS_TOP].astype(str).to_numpy() == spots.obs[sd.OBS_TOP].astype(str).to_numpy()).all()


def test_a_run_with_no_choices_records_and_replays_as_before(tmp_path, data):
    ref, spots = data
    spots = spots.copy()
    p = _prepare(ref, spots)
    out = _fit(p, "nnls")
    s = sd.summarize(p, out, method="nnls")
    s["reference"]["path"] = str(tmp_path / "reference.h5ad")
    sd.save(spots, p, out, s)
    params = provenance.read_provenance(spots)["history"][-1]["params"]
    assert "excluded_labels" not in params and "merged_labels" not in params
    assert sd.labels_used_line(s) == ""
    script = replay_script.build_script(spots, dataset_id="t", dataset_name="test")
    assert "SPOT_EXCLUDE" not in script and "SPOT_MERGE" not in script

    class Adaptor:
        adata = spots

    assert "Left out of the reference before" not in "\n".join(sd.grounding_lines(Adaptor())[0])


def test_an_old_saved_result_with_no_such_field_still_opens(data):
    ref, spots = data
    spots = spots.copy()
    p = _prepare(ref, spots)
    out = _fit(p, "nnls")
    s = sd.summarize(p, out, method="nnls")
    s["reference"]["path"] = "reference.h5ad"
    sd.save(spots, p, out, s)
    old = json.loads(spots.uns[sd.UNS_KEY])
    for key in ("excluded_labels", "merged_labels", "n_excluded_cells", "over_share"):
        del old[key]
    spots.uns[sd.UNS_KEY] = json.dumps(old)
    assert sd.saved(spots)["types"] == s["types"]

    class Adaptor:
        adata = spots

    assert sd.grounding_lines(Adaptor())[0]
    step = dict(provenance.read_provenance(spots)["history"][-1]["params"])
    snip = replay_script._spot_deconvolution(step, replay_script._Ctx(adata=None))
    assert snip.status == replay_script.EXACT


def test_replay_refuses_a_malformed_record(data):
    ref, spots = data
    spots = spots.copy()
    p = _prepare(ref, spots)
    out = _fit(p, "nnls")
    s = sd.summarize(p, out, method="nnls")
    s["reference"]["path"] = "reference.h5ad"
    sd.save(spots, p, out, s)
    step = dict(provenance.read_provenance(spots)["history"][-1]["params"])
    step["merged_labels"] = "not a mapping"
    assert replay_script._spot_deconvolution(step, replay_script._Ctx(adata=None)).status == replay_script.NOT_REPLAYABLE


def test_the_over_share_list_names_what_the_note_names():
    rna = np.tile([0.3, 0.5, 0.2], (10, 1))
    assert list(sd.over_share_types(["Unk", "Oligo", "Astro"], np.array([20, 500, 480]), rna)) == ["Unk"]
    assert sd.over_share_notes(["Unk", "Oligo", "Astro"], np.array([20, 500, 480]), rna)[0].endswith(
        "leave it out and run again.")
    assert sd.over_share_types(["A"], np.array([5]), np.empty((0, 1))) == {}


def test_the_self_check_uses_the_same_labels(monkeypatch, data):
    monkeypatch.setattr(sd, "CHECK_SPOTS", 40)
    monkeypatch.setattr(sd, "CHECK_DEPTHS", (5000,))
    ref, spots = data
    p = _prepare(ref, spots, check=True, exclude_labels=["Unk_1"], merge_labels={"Mono": "Myeloid", "Stroma": "Myeloid"})
    checked = [i for i in sd.iter_check(p, methods=("nnls",)) if i["type"] == "checked"][0]["out"]
    assert checked["types"] == ["B", "Myeloid", "T"]
    assert checked["excluded_labels"] == ["Unk_1"] and checked["merged_labels"] == {"Mono": "Myeloid", "Stroma": "Myeloid"}


def test_the_estimate_follows_the_type_count():
    a = sd.estimate_seconds(1000, 5000, 40)
    assert sd.estimate_seconds(1000, 5000, 20) < a < sd.estimate_seconds(1000, 5000, 50)


@pytest.mark.parametrize("n_ref, n_spots, n_types, bins, measured", [
    (40_572, 98_896, 7, True, 84.9),  # HD 16 um x the mouse brain reference, merged to 7 types
    (40_572, 98_896, 16, True, 88.9),
    (40_572, 98_896, 26, True, 165.8),
    (40_572, 98_896, 37, True, 424.2),
    (40_572, 2_310, 15, False, 8.7),  # CytAssist FFPE mouse brain spots
    (40_572, 2_310, 37, False, 17.0),
    (73_260, 4_035, 33, False, 31.4),  # V1_Human_Lymph_Node x the lymphoid atlas (Subset)
])
def test_the_estimate_is_within_20_percent_of_the_measured_runs(n_ref, n_spots, n_types, bins, measured):
    # whole runs through the API, 2 CPUs with 2 BLAS threads, 2026-11-06
    est = sd.estimate_seconds(n_ref, n_spots, n_types, bins=bins)
    assert abs(est / measured - 1) <= 0.2


# --- Through the API ---


def _pair(dm, tag: str) -> tuple[str, str, np.ndarray]:
    prof = _profiles(n_types=5)
    ref = _reference(prof, types=TYPES5)
    spots, _ = _spots(prof, n=60)
    spots.obs["leiden"] = np.where(np.arange(spots.n_obs) < 30, "0", "1")
    spots.uns["scview_active_clustering"] = "leiden"
    return _register(dm, ref, f"lref{tag}"), _register(dm, spots, f"lspots{tag}"), prof


async def test_the_reference_report_after_labels(client, dataset_manager):
    rid, did, _ = _pair(dataset_manager, "a")
    url = f"{API}/datasets/{did}/spot-deconvolution/reference/{rid}"
    got = (await client.get(url, params={"celltype_column": "cell_type"})).json()
    assert got["suggested_exclude"] == ["Unk_1"]
    assert got["excluded_labels"] == [] and got["merged_labels"] == {} and got["n_excluded_cells"] == 0
    assert {t["type"]: t["n_cells"] for t in got["types"]} == dict.fromkeys(TYPES5, 60)
    # Nothing ticked by default: the suggestion is only a suggestion.
    assert got["ok"] is True and len(got["types"]) == 5

    r = await client.post(f"{url}/labels", json={"celltype_column": "cell_type", "exclude_labels": ["Unk_1"],
                                                 "merge_labels": {"Mono": "Myeloid", "Stroma": "Myeloid"}})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True and body["n_excluded_cells"] == 60 and body["merged_labels"]["Mono"] == "Myeloid"
    assert {t["type"]: t["n_cells"] for t in body["types"]} == {"B": 60, "Myeloid": 120, "T": 60}
    assert body["suggested_exclude"] == ["Unk_1"]  # of the column, whatever is ticked
    assert body["estimate"]["seconds"] <= got["estimate"]["seconds"]

    r = await client.post(f"{url}/labels", json={"celltype_column": "cell_type", "exclude_labels": ["Nope"]})
    assert r.status_code == 422 and "has no label 'Nope'" in r.json()["detail"]
    r = await client.post(f"{url}/labels", json={"celltype_column": "cell_type",
                                                 "exclude_labels": ["B", "T", "Mono", "Stroma"]})
    assert r.status_code == 200 and r.json()["ok"] is False and "fewer than 2 types" in r.json()["reason"]


async def test_too_many_types_become_a_run_after_merging(client, dataset_manager):
    n = sd.MAX_TYPES + 2
    prof = _profiles(n_types=n)
    ref = _reference(prof, types=[f"t_{i}" for i in range(n)], cells=25)
    spots, _ = _spots(prof, n=20)
    rid, did = _register(dataset_manager, ref, "lmany"), _register(dataset_manager, spots, "lmanys")
    url = f"{API}/datasets/{did}/spot-deconvolution"
    got = (await client.get(f"{url}/reference/{rid}", params={"celltype_column": "cell_type"})).json()
    assert got["ok"] is False and f"{n} types" in got["reason"] and "merge or leave out" in got["reason"]
    assert len(got["types"]) == n  # every label stays listed for the control
    merge = {"t_0": "t_1", "t_2": "t_3"}
    body = (await client.post(f"{url}/reference/{rid}/labels",
                              json={"celltype_column": "cell_type", "merge_labels": merge})).json()
    assert body["ok"] is True and len(body["types"]) == sd.MAX_TYPES
    r = await client.post(f"{url}/run-stream", json={"reference_id": rid, "celltype_column": "cell_type",
                                                     "method": "nnls", "merge_labels": merge})
    events = _events(r.text)
    assert events[-1][0] == "complete", events[-1]
    assert len(events[-2][1]["types"]) == sd.MAX_TYPES and events[-2][1]["merged_labels"] == merge


async def test_a_run_and_a_check_take_them_and_a_bad_name_is_a_422(client, dataset_manager, monkeypatch):
    monkeypatch.setattr(sd, "CHECK_SPOTS", 40)
    monkeypatch.setattr(sd, "CHECK_DEPTHS", (5000,))
    rid, did, _ = _pair(dataset_manager, "b")
    url = f"{API}/datasets/{did}/spot-deconvolution"
    base = {"reference_id": rid, "celltype_column": "cell_type"}
    for path in ("run-stream", "check-stream"):
        r = await client.post(f"{url}/{path}", json={**base, "exclude_labels": ["Nope"]})
        assert r.status_code == 422 and "has no label 'Nope'" in r.json()["detail"]
    r = await client.post(f"{url}/run-stream", json={**base, "exclude_labels": ["x" * 513]})
    assert r.status_code == 422

    edits = {"exclude_labels": ["Unk_1"], "merge_labels": {"Mono": "Myeloid", "Stroma": "Myeloid"}}
    events = _events((await client.post(f"{url}/run-stream", json={**base, "method": "nnls", **edits})).text)
    res = events[-2][1]
    assert events[-1][0] == "complete" and res["types"] == ["B", "Myeloid", "T"] and res["excluded_labels"] == ["Unk_1"]
    assert events[0][1]["estimate"]["seconds"] > 0
    info = (await client.get(url)).json()
    assert info["saved"]["merged_labels"] == edits["merge_labels"] and info["saved"]["n_excluded_cells"] == 60
    shares = (await client.get(f"{url}/share", params={"type": "Myeloid"})).json()["values"]
    assert len(shares) == 60
    assert (await client.get(f"{url}/share", params={"type": "Unk_1"})).status_code == 404

    events = _events((await client.post(f"{url}/check-stream", json={**base, **edits})).text)
    assert events[-1][0] == "complete", events[-1]
    assert events[-2][1]["types"] == ["B", "Myeloid", "T"] and events[-2][1]["excluded_labels"] == ["Unk_1"]
    assert (await client.get(f"{url}/check")).json()["merged_labels"] == edits["merge_labels"]


async def test_an_old_saved_result_opens_through_the_info(client, dataset_manager):
    rid, did, _ = _pair(dataset_manager, "c")
    url = f"{API}/datasets/{did}/spot-deconvolution"
    await client.post(f"{url}/run-stream", json={"reference_id": rid, "celltype_column": "cell_type", "method": "nnls"})
    adata = (await dataset_manager.get_or_load_dataset(did)).adata
    old = json.loads(adata.uns[sd.UNS_KEY])
    for key in ("excluded_labels", "merged_labels", "n_excluded_cells", "over_share"):
        del old[key]
    adata.uns[sd.UNS_KEY] = json.dumps(old)
    got = (await client.get(url)).json()
    assert got["saved"]["types"] == old["types"] and "excluded_labels" not in got["saved"]


def test_labels_series_helper_keeps_missing_and_non_strings():
    labels = pd.Series(["a", None, "b", "c"], index=list("wxyz"), dtype=object)
    out = sd.sd_edit_labels(labels, ["b"], {"c": "a"})
    assert out.tolist() == ["a", None, None, "a"] and list(out.index) == list("wxyz")
