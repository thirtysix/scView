"""Saved program and chromVAR scores (core/saved_scores.py kinds ``program`` and ``chromvar``):
one program's usage or one motif's z, drawn with ``save``, kept as an obs column with the run
it came from; replaced by the same program of the same run or the same motif, flagged when the
dataset's run is a later one, removed on request, saved on a backed dataset, and replayed."""

from __future__ import annotations

import json

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from scview.core import anndata_adaptor, provenance, replay_script, saved_scores, tf_motifs
from scview.core import gene_programs as gp
from tests.test_atac_api import _register_multiome
from tests.test_atac_motifs_api import _scan, annotated, inputs  # noqa: F401 - fixtures
from tests.test_gene_programs import _planted, _saved
from tests.test_hvg_sensitivity import _events, _register

API = "/api/v1"


# --- core ----------------------------------------------------------------------


def _small(n: int = 4) -> ad.AnnData:
    a = ad.AnnData(np.zeros((n, 3), dtype=np.float32))
    a.obs_names = [f"c{i}" for i in range(n)]
    return a


def _fake_programs(a: ad.AnnData, run_id: str, k: int = 2) -> None:
    a.obsm[gp.OBSM_KEY] = np.zeros((a.n_obs, k), dtype=np.float32)
    a.uns[gp.UNS_KEY] = {"info": json.dumps({"k": k, "run_id": run_id})}


def _program(a, run_id: str, index: int, values, k: int = 2) -> str:
    return saved_scores.save(a, kind="program", name=f"P{index} (k {k})", values=values, method="NMF usage share",
                             column_body=f"k{k}_{index}", extra={"run_id": run_id, "program": index, "k": k,
                                                                 "top_genes": ["G1", "G2"]})


def test_a_program_is_named_by_its_run_and_replaced_only_by_the_same_run():
    a = _small()
    col = _program(a, "r1", 2, [0.1, 0.2, 0.3, 0.4])
    assert col == "program_k2_2"
    entry = saved_scores.record(a)[col]
    assert entry["run_id"] == "r1" and entry["program"] == 2 and entry["top_genes"] == ["G1", "G2"]
    assert _program(a, "r1", 2, np.ones(4)) == col  # the same program drawn again
    np.testing.assert_array_equal(a.obs[col], np.ones(4, dtype=np.float32))
    other = _program(a, "r2", 2, np.zeros(4))  # a later run's P2 is another score
    assert other == "program_k2_2_2" and set(saved_scores.record(a)) == {col, other}
    assert _program(a, "r1", 1, np.zeros(4)) == "program_k2_1"


def test_earlier_run_is_flagged_once_the_dataset_has_another_run():
    a = _small()
    _fake_programs(a, "r1")
    col = _program(a, "r1", 1, np.zeros(4))
    runs = saved_scores.current_runs(a)
    flag = saved_scores.obs_flag(saved_scores.record(a)[col], runs)
    assert flag["kind"] == "program" and flag["k"] == 2 and flag["run_id"] == "r1" and flag["earlier_run"] is False
    _fake_programs(a, "r2")
    assert saved_scores.obs_flag(saved_scores.record(a)[col], saved_scores.current_runs(a))["earlier_run"] is True
    del a.obsm[gp.OBSM_KEY]  # no programs at all now: still from an earlier run
    assert saved_scores.obs_flag(saved_scores.record(a)[col], saved_scores.current_runs(a))["earlier_run"] is True
    # Without programs or chromVAR among the saved scores nothing is read.
    b = _small()
    saved_scores.save(b, kind="geneset", name="s", values=np.zeros(4))
    assert saved_scores.current_runs(b) == {}
    assert "earlier_run" not in saved_scores.obs_flag(saved_scores.record(b)["score_s"], {})


def _motif(a, motif_id: str, run_id: str, values) -> str:
    return saved_scores.save(a, kind="chromvar", name="GENEA::NOPE", values=values, method="chromVAR deviation z",
                             column_body=f"GENEA::NOPE_{motif_id}", extra={"motif_id": motif_id, "run_id": run_id,
                                                                            "jaspar": "JASPAR 2024"})


def test_a_motif_keeps_its_nan_and_is_replaced_by_the_same_motif_from_any_run():
    a = _small()
    col = _motif(a, "MA0002.1", "c1", [np.nan, -1.5, 0.0, 2.0])
    assert col == "chromvar_GENEA_NOPE_MA0002.1"
    assert np.isnan(a.obs[col].iloc[0]) and a.obs[col].iloc[1] == pytest.approx(-1.5)
    provenance.record_step(a, step="atac_chromvar", tool="t", params={"run_id": "c1"})
    flag = saved_scores.obs_flag(saved_scores.record(a)[col], saved_scores.current_runs(a))
    assert flag["motif_id"] == "MA0002.1" and flag["jaspar"] == "JASPAR 2024" and flag["earlier_run"] is False
    provenance.record_step(a, step="atac_chromvar", tool="t", params={"run_id": "c2"})
    assert saved_scores.obs_flag(saved_scores.record(a)[col], saved_scores.current_runs(a))["earlier_run"] is True
    assert _motif(a, "MA0002.1", "c2", np.zeros(4)) == col
    assert saved_scores.record(a)[col]["run_id"] == "c2"
    assert _motif(a, "MA0003.1", "c2", np.zeros(4)) == "chromvar_GENEA_NOPE_MA0003.1"


def test_already_saved_needs_the_same_score_method_and_values():
    a = _small()
    vals = [np.nan, -1.5, 0.0, 2.0]
    col = _motif(a, "MA0002.1", "c1", vals)
    same = {"kind": "chromvar", "name": "GENEA::NOPE", "method": "chromVAR deviation z",
            "extra": {"motif_id": "MA0002.1", "run_id": "c1"}}
    assert saved_scores.already_saved(a, values=vals, **same) == col  # NaN where NaN
    assert saved_scores.already_saved(a, values=[np.nan, -1.5, 0.0, 2.5], **same) is None
    assert saved_scores.already_saved(a, values=vals, **{**same, "method": "other"}) is None
    assert saved_scores.already_saved(a, values=vals, **{**same, "extra": {"motif_id": "MA0003.1"}}) is None
    g = saved_scores.save(a, kind="geneset", name="s", values=np.ones(4), method="aucell", genes=["A", "B"])
    assert saved_scores.already_saved(a, kind="geneset", name="s", values=np.ones(4), method="aucell", genes=["A", "B"]) == g
    assert saved_scores.already_saved(a, kind="geneset", name="s", values=np.ones(4), method="aucell", genes=["A"]) is None


def test_a_program_or_motif_without_its_run_is_refused():
    with pytest.raises(saved_scores.SavedScoreError, match="run_id and program"):
        saved_scores.save(_small(), kind="program", name="P1", values=np.zeros(4), extra={"run_id": "r"})
    with pytest.raises(saved_scores.SavedScoreError, match="motif_id"):
        saved_scores.save(_small(), kind="chromvar", name="X", values=np.zeros(4))


# --- replay -------------------------------------------------------------------


def test_replay_reads_a_saved_program_from_the_programs_it_ran():
    a, fit, _inp, out = _saved()
    for raw, index in ((False, 2), (True, 3)):
        provenance.record_step(a, step="save_score", tool="t", params={
            "column": f"program_k3_{index}", "kind": "program", "name": f"P{index} (k 3)",
            "method": "NMF usage" if raw else "NMF usage share", "run_id": out["run_id"], "program": index, "k": 3})
    statuses = [d["status"] for d in replay_script.describe(a)]
    assert statuses[-2:] == [replay_script.EXACT, replay_script.EXACT]
    script = replay_script.build_script(a)
    line = next(x for x in script.splitlines() if x.startswith("adata.obs['program_k3_2']"))
    assert 'adata.obsm["X_nmf_usage"]' in line.replace("'", '"') and "[:, 1]" in line
    assert any(x.startswith("adata.obs['program_k3_3']") and "X_nmf_usage_raw" in x for x in script.splitlines())
    # The line itself gives the saved values on the programs the script computed.
    ns = {"np": np, "adata": a}
    exec(line, ns)  # noqa: S102
    np.testing.assert_allclose(a.obs["program_k3_2"], fit.usage[:, 1], atol=1e-6)


def test_replay_says_a_program_or_motif_is_not_recomputed_without_its_run():
    a = _small()
    for p in ({"column": "program_k2_1", "kind": "program", "name": "P1 (k 2)", "run_id": "r", "program": 1},
              {"column": "chromvar_X_MA1", "kind": "chromvar", "name": "X", "motif_id": "MA1", "run_id": "c"}):
        provenance.record_step(a, step="save_score", tool="t", params=p)
    assert [d["status"] for d in replay_script.describe(a)][-2:] == [replay_script.NOT_REPLAYABLE] * 2
    script = replay_script.build_script(a)
    assert "gene programs it came from are not computed above" in script
    assert "chromVAR run it came from is not computed above" in script


def test_replay_reads_a_saved_motif_from_the_chromvar_it_ran():
    a = _small()
    provenance.record_step(a, step="atac_chromvar", tool="t", params={
        "seed": 0, "iterations": 50, "w": 0.1, "bs": 50, "genome": "GRCh38", "run_id": "c1"})
    provenance.record_step(a, step="save_score", tool="t", params={
        "column": "chromvar_SPI1_MA0080.6", "kind": "chromvar", "name": "SPI1", "motif_id": "MA0080.6", "run_id": "c1"})
    provenance.record_step(a, step="save_score", tool="t", params={  # another run's: not what the script computes
        "column": "chromvar_SPI1_MA0081.1", "kind": "chromvar", "name": "SPI1", "motif_id": "MA0081.1", "run_id": "c0"})
    statuses = [d["status"] for d in replay_script.describe(a)]
    assert statuses[-2:] == [replay_script.EXACT, replay_script.NOT_REPLAYABLE]
    script = replay_script.build_script(a)
    line = next(x for x in script.splitlines() if x.startswith("adata.obs['chromvar_SPI1_MA0080.6']"))
    b = _small(3)
    b.obsm["atac_chromvar_z"] = np.array([[1.0, 2.0], [np.nan, 3.0], [0.5, -4.0]], dtype=np.float32)
    b.uns["atac_chromvar_motifs"] = ["MA0001.1", "MA0080.6"]
    exec(line, {"np": np, "adata": b})  # noqa: S102
    np.testing.assert_array_equal(b.obs["chromvar_SPI1_MA0080.6"], np.array([2.0, 3.0, -4.0], dtype=np.float32))


# --- endpoints: programs ----------------------------------------------------------


def _columns(info: dict) -> dict[str, dict]:
    return {c["name"]: c for c in info["obs_columns"]}


async def _usage(client, did: str, index: int, **params):
    r = await client.get(f"{API}/datasets/{did}/programs/{index}/usage", params=params)
    assert r.status_code == 200, r.text
    return r.json()


async def test_drawing_a_program_with_save_keeps_it_and_a_new_run_marks_it_earlier(client, dataset_manager):
    a, _fit, _inp, out = _saved()
    did = _register(dataset_manager, a, "pgsave")
    body = await _usage(client, did, 2, save="true")
    col = body["saved_column"]
    assert col == "program_k3_2" and body["save_note"] is None
    rec = (await client.get(f"{API}/datasets/{did}/scores")).json()["scores"][col]
    assert rec["kind"] == "program" and rec["run_id"] == out["run_id"] and rec["program"] == 2 and rec["k"] == 3
    assert rec["seeds"] == [0, 1, 2] and rec["method"] == "NMF usage share" and len(rec["top_genes"]) == 10
    assert rec["source"] == "log-normalised .raw"
    flag = _columns((await client.get(f"{API}/datasets/{did}")).json())[col]["saved_score"]
    assert flag["name"] == "P2 (k 3)" and flag["earlier_run"] is False and flag["top_genes"] == rec["top_genes"]
    np.testing.assert_allclose(dataset_manager.get_dataset(did).adata.obs[col], body["values"], atol=1e-5)
    # The same program again (raw usage this time) replaces it; without save nothing changes.
    raw = await _usage(client, did, 2, save="true", raw="true")
    assert raw["saved_column"] == col
    assert (await client.get(f"{API}/datasets/{did}/scores")).json()["scores"][col]["method"] == "NMF usage"
    assert (await _usage(client, did, 1)).get("saved_column") is None
    assert list((await client.get(f"{API}/datasets/{did}/scores")).json()["scores"]) == [col]

    # A new run: the saved column stays, from an earlier run; its P2 is a new column.
    r = await client.post(f"{API}/datasets/{did}/programs/run-stream", json={"k": 3, "seeds": 1, "sweep": []})
    assert _events(r.text)[-1][0] == "complete", r.text[-300:]
    dataset_manager.evict(did)
    cols = _columns((await client.get(f"{API}/datasets/{did}")).json())
    assert cols[col]["saved_score"]["earlier_run"] is True
    again = await _usage(client, did, 2, save="true")
    assert again["saved_column"] == "program_k3_2_2"
    cols = _columns((await client.get(f"{API}/datasets/{did}")).json())
    assert cols["program_k3_2_2"]["saved_score"]["earlier_run"] is False

    r = await client.delete(f"{API}/datasets/{did}/scores/{col}")
    assert r.status_code == 200 and r.json() == {"removed": col, "kind": "program", "name": "P2 (k 3)"}
    dataset_manager.evict(did)
    assert col not in _columns((await client.get(f"{API}/datasets/{did}")).json())


async def test_a_program_saves_on_a_backed_dataset(client, dataset_manager, monkeypatch):
    monkeypatch.setattr(anndata_adaptor, "BACKED_MIN_BYTES", 0)
    a, *_ = _saved()
    did = _register(dataset_manager, a, "pgbacked")
    assert (await dataset_manager.get_or_load_dataset(did)).adata.isbacked
    body = await _usage(client, did, 1, save="true")
    assert body["saved_column"] == "program_k3_1", body["save_note"]
    dataset_manager.evict(did)
    back = await dataset_manager.get_or_load_dataset(did)
    np.testing.assert_allclose(np.asarray(back.adata.obs["program_k3_1"]), body["values"], atol=1e-5)
    assert gp.stored(back.adata) is not None  # the programs stay with it


async def test_no_programs_no_save(client, dataset_manager):
    did = _register(dataset_manager, _planted()[0], "pgnone")
    r = await client.get(f"{API}/datasets/{did}/programs/1/usage", params={"save": "true"})
    assert r.status_code == 404
    assert (await client.get(f"{API}/datasets/{did}/scores")).json()["scores"] == {}


# --- endpoints: chromVAR ----------------------------------------------------------


async def test_drawing_a_motif_with_save_keeps_its_z(client, dataset_manager, annotated, inputs):  # noqa: F811
    did, *_ = _register_multiome(dataset_manager, "cvsave")
    await _scan(client, did)
    r = await client.post(f"{API}/datasets/{did}/atac/motifs/chromvar", json={})
    run_id = next(d for e, d in _events(r.text) if e == "result")["run_id"]
    url = f"{API}/datasets/{did}/atac/motifs/chromvar/motif"
    body = (await client.get(url, params={"motif_id": "MA0002.1", "save": "true"})).json()
    col = body["saved_column"]
    assert col == "chromvar_GENEA_NOPE_MA0002.1" and body["save_note"] is None
    rec = (await client.get(f"{API}/datasets/{did}/scores")).json()["scores"][col]
    assert rec["kind"] == "chromvar" and rec["motif_id"] == "MA0002.1" and rec["run_id"] == run_id
    assert rec["iterations"] == 50 and rec["genome"] == "GRCh38"
    assert rec["jaspar"] == tf_motifs.JASPAR.name and rec["jaspar_sha256"] == tf_motifs.JASPAR.sha256  # the pinned file
    flag = _columns((await client.get(f"{API}/datasets/{did}")).json())[col]["saved_score"]
    assert flag["name"] == "GENEA::NOPE" and flag["earlier_run"] is False
    got = np.asarray(dataset_manager.get_dataset(did).adata.obs[col], dtype=np.float64)
    want = np.array([np.nan if v is None else v for v in body["values"]], dtype=np.float64)
    np.testing.assert_allclose(got, want, atol=1e-3, equal_nan=True)
    assert (got < 0).any() and (got > 0).any()  # signed: the colour scale centres on 0
    # The same motif drawn again: the same column, nothing written or recorded again;
    # without save nothing is written.
    n_steps = len(provenance.read_provenance(dataset_manager.get_dataset(did).adata)["history"])
    again = (await client.get(url, params={"motif_id": "MA0002.1", "save": "true"})).json()
    assert again["saved_column"] == col and again["save_note"] is None
    assert len(provenance.read_provenance(dataset_manager.get_dataset(did).adata)["history"]) == n_steps
    assert (await client.get(url, params={"motif_id": "MA0001.1"})).json().get("saved_column") is None
    assert list((await client.get(f"{API}/datasets/{did}/scores")).json()["scores"]) == [col]
    steps = [h["step"] for h in provenance.read_provenance(dataset_manager.get_dataset(did).adata)["history"]]
    assert steps[-2:] == ["atac_chromvar", "save_score"]
    assert replay_script.describe(dataset_manager.get_dataset(did).adata)[-1]["status"] == replay_script.EXACT
    r = await client.delete(f"{API}/datasets/{did}/scores/{col}")
    assert r.status_code == 200 and r.json()["kind"] == "chromvar"
    assert (await client.get(f"{API}/datasets/{did}/scores")).json()["scores"] == {}


def test_saved_h5ad_keeps_a_motif_column_with_nan(tmp_path):
    a = _small()
    col = _motif(a, "MA0002.1", "c1", [np.nan, 1.0, -1.0, 0.0])
    a.write_h5ad(tmp_path / "s.h5ad")
    b = ad.read_h5ad(tmp_path / "s.h5ad")
    assert np.isnan(b.obs[col].iloc[0]) and saved_scores.record(b)[col]["motif_id"] == "MA0002.1"
    assert isinstance(b.obs, pd.DataFrame)
