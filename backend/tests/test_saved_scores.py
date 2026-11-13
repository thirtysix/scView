"""Saved per-cell scores (core/saved_scores.py, api/v1/saved_scores.py): a scored gene set or
one TF / pathway's activity kept as an obs column with its record; replaced on a re-score,
removed on request, saved in memory, backed and large datasets alike, and replayed."""

from __future__ import annotations

import json

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest
import scanpy as sc

from scview.core import aucell, provenance, replay_script, results_store, saved_scores
from scview.core import anndata_adaptor
from tests.large_fixture import counts_adata, register
from tests.test_activity import _cells, _lognorm, networks, small_networks  # noqa: F401 - fixtures
from tests.test_geneset_score_methods import SET, _adata, _register

API = "/api/v1"
HALLMARK = "HALLMARK_TNFA_SIGNALING_VIA_NFKB"


# --- core ----------------------------------------------------------------------


def _small() -> ad.AnnData:
    a = ad.AnnData(np.zeros((4, 3), dtype=np.float32))
    a.obs_names = [f"c{i}" for i in range(4)]
    a.obs["cell_type"] = pd.Categorical(["a", "b", "a", "b"])
    return a


def test_save_names_the_column_and_records_what_it_holds():
    a = _small()
    col = saved_scores.save(a, kind="geneset", name="IFN response (v2)", values=[0.1, 0.2, -0.3, 0.4],
                            method="aucell", genes=["ISG15", "MX1"], genes_missing=["NOPE"],
                            source="log-normalised .raw", extra={"max_rank": 12})
    assert col == "score_IFN_response_v2"
    assert a.obs[col].dtype == np.float32
    np.testing.assert_allclose(a.obs[col], [0.1, 0.2, -0.3, 0.4], rtol=1e-6)
    assert isinstance(a.uns[saved_scores.RECORD_KEY], str)  # JSON, as the provenance
    entry = saved_scores.record(a)[col]
    assert entry["kind"] == "geneset" and entry["name"] == "IFN response (v2)" and entry["method"] == "aucell"
    assert entry["genes"] == ["ISG15", "MX1"] and entry["genes_missing"] == ["NOPE"]
    assert entry["source"] == "log-normalised .raw" and entry["max_rank"] == 12 and entry["created"]
    step = provenance.read_provenance(a)["history"][-1]
    assert step["step"] == "save_score" and step["params"]["column"] == col
    assert step["params"]["replaced"] is False


def test_the_same_set_scored_again_replaces_its_column_by_any_method():
    a = _small()
    first = saved_scores.save(a, kind="geneset", name=HALLMARK, values=np.ones(4), method="score_genes")
    again = saved_scores.save(a, kind="geneset", name=HALLMARK, values=np.full(4, 2.0), method="aucell")
    assert first == again == f"score_{HALLMARK}"
    assert list(saved_scores.record(a)) == [first]
    assert saved_scores.record(a)[first]["method"] == "aucell"
    assert a.obs[first].tolist() == [2.0] * 4
    assert provenance.read_provenance(a)["history"][-1]["params"]["replaced"] is True


def test_a_column_the_file_made_is_never_overwritten():
    a = _small()
    a.obs["score_IFN"] = [9.0, 9.0, 9.0, 9.0]  # the file's own
    a.obs["activity_STAT1"] = ["x"] * 4
    col = saved_scores.save(a, kind="geneset", name="IFN", values=np.zeros(4), method="score_genes")
    assert col == "score_IFN_2" and a.obs["score_IFN"].tolist() == [9.0] * 4
    act = saved_scores.save(a, kind="activity", name="STAT1", values=np.zeros(4), method="ulm", network="collectri")
    assert act == "activity_STAT1_2"
    # A TF and a pathway of the same name are two scores; the same one again is one.
    other = saved_scores.save(a, kind="activity", name="STAT1", values=np.ones(4), method="ulm", network="progeny")
    assert other == "activity_STAT1_3"
    assert saved_scores.save(a, kind="activity", name="STAT1", values=np.ones(4), network="collectri") == act
    with pytest.raises(saved_scores.SavedScoreError) as e:
        saved_scores.remove(a, "score_IFN")
    assert e.value.status == 404 and "score_IFN" in a.obs


def test_names_are_cleaned_capped_and_kept_unique():
    assert saved_scores.clean_name("geneset", "a/b c:d") == "score_a_b_c_d"
    assert saved_scores.clean_name("geneset", "   ") == "score_unnamed"
    long = "X" * 200
    a = _small()
    c1 = saved_scores.save(a, kind="geneset", name=long, values=np.zeros(4))
    c2 = saved_scores.save(a, kind="geneset", name=long + "Y", values=np.zeros(4))
    assert len(c1) <= saved_scores.NAME_MAX and len(c2) <= saved_scores.NAME_MAX and c1 != c2


def test_remove_drops_the_column_and_cleans_the_record():
    a = _small()
    c1 = saved_scores.save(a, kind="geneset", name="one", values=np.zeros(4))
    c2 = saved_scores.save(a, kind="geneset", name="two", values=np.zeros(4))
    saved_scores.remove(a, c1)
    assert c1 not in a.obs and list(saved_scores.record(a)) == [c2]
    assert provenance.read_provenance(a)["history"][-1]["step"] == "remove_score"
    saved_scores.remove(a, c2)
    assert saved_scores.RECORD_KEY not in a.uns


def test_scores_for_other_cells_are_refused():
    with pytest.raises(saved_scores.SavedScoreError) as e:
        saved_scores.save(_small(), kind="geneset", name="x", values=np.zeros(3))
    assert e.value.status == 409


def test_saving_a_score_does_not_make_results_stale():
    assert {"save_score", "remove_score"} <= results_store.ADDITIVE_STEPS


# --- endpoints ----------------------------------------------------------------


async def _score(client, did: str, **body):
    r = await client.post(f"{API}/datasets/{did}/genesets/score",
                          json={"gene_set": SET, "score_name": HALLMARK, **body})
    assert r.status_code == 200, r.text
    return r.json()


def _columns(info: dict) -> dict[str, dict]:
    return {c["name"]: c for c in info["obs_columns"]}


async def test_scoring_with_save_keeps_the_score_listed_and_on_disk(client, dataset_manager):
    did = _register(dataset_manager, _adata(), "sv")
    body = await _score(client, did, save=True)
    col = body["saved_column"]
    assert col == f"score_{HALLMARK}" and body["save_note"] is None
    info = (await client.get(f"{API}/datasets/{did}")).json()
    flag = _columns(info)[col]["saved_score"]
    assert flag == {"kind": "geneset", "name": HALLMARK, "method": "score_genes", "network": None}
    assert "saved_score" not in _columns(info)["cluster"]
    listed = (await client.get(f"{API}/datasets/{did}/scores")).json()["scores"]
    assert listed[col]["genes"] == body["genes_found"] and listed[col]["genes_missing"] == ["NOSUCHGENE"]
    # The colour the plot draws comes from the column (float32 of the scores).
    adaptor = dataset_manager.get_dataset(did)
    np.testing.assert_allclose(adaptor.adata.obs[col], body["scores"], rtol=1e-6)
    # Saved: a fresh open still has it.
    dataset_manager.evict(did)
    info = (await client.get(f"{API}/datasets/{did}")).json()
    assert _columns(info)[col]["saved_score"]["method"] == "score_genes"


async def test_without_save_nothing_is_written(client, dataset_manager):
    did = _register(dataset_manager, _adata(), "nosave")
    body = await _score(client, did)
    assert body["saved_column"] is None
    info = (await client.get(f"{API}/datasets/{did}")).json()
    assert not any("saved_score" in c for c in info["obs_columns"])


async def test_rescoring_by_another_method_replaces_the_column(client, dataset_manager):
    did = _register(dataset_manager, _adata(), "re")
    first = await _score(client, did, save=True)
    second = await _score(client, did, save=True, method="aucell")
    assert second["saved_column"] == first["saved_column"]
    rec = (await client.get(f"{API}/datasets/{did}/scores")).json()["scores"]
    assert list(rec) == [first["saved_column"]]
    assert rec[first["saved_column"]]["method"] == "aucell" and rec[first["saved_column"]]["max_rank"] >= 2
    adata = dataset_manager.get_dataset(did).adata
    np.testing.assert_allclose(adata.obs[first["saved_column"]], second["scores"], rtol=1e-6)


async def test_delete_removes_a_saved_score_and_never_another_column(client, dataset_manager):
    did = _register(dataset_manager, _adata(), "rm")
    col = (await _score(client, did, save=True))["saved_column"]
    r = await client.delete(f"{API}/datasets/{did}/scores/cluster")
    assert r.status_code == 404
    r = await client.delete(f"{API}/datasets/{did}/scores/{col}")
    assert r.status_code == 200, r.text
    assert r.json() == {"removed": col, "kind": "geneset", "name": HALLMARK}
    dataset_manager.evict(did)
    info = (await client.get(f"{API}/datasets/{did}")).json()
    assert col not in _columns(info) and "cluster" in _columns(info)
    assert (await client.get(f"{API}/datasets/{did}/scores")).json()["scores"] == {}
    steps = [s["step"] for s in provenance.read_provenance(dataset_manager.get_dataset(did).adata)["history"]]
    assert steps[-2:] == ["save_score", "remove_score"]


async def test_a_run_started_or_cells_changed_since_scoring_leave_it_unsaved(client, dataset_manager):
    from scview.api.v1.saved_scores import persist

    did = _register(dataset_manager, _adata(), "busy")
    await client.get(f"{API}/datasets/{did}")
    adaptor = dataset_manager.get_dataset(did)
    cells = adaptor.cells_fingerprint()
    score = {"kind": "geneset", "name": "s", "values": np.zeros(240)}
    dataset_manager.mark_busy(did)
    try:
        column, note = await persist(dataset_manager, did, cells, **score)
    finally:
        dataset_manager.mark_idle(did)
    assert column is None and "being processed" in note
    column, note = await persist(dataset_manager, did, "other cells", **score)
    assert column is None and "cells changed" in note
    assert "score_s" not in adaptor.adata.obs
    column, note = await persist(dataset_manager, did, cells, **score)
    assert column == "score_s" and note is None


async def test_backed_dataset_saves_too(client, dataset_manager, monkeypatch):
    monkeypatch.setattr(anndata_adaptor, "BACKED_MIN_BYTES", 0)
    did = _register(dataset_manager, _adata(), "backed")
    body = await _score(client, did, save=True, method="aucell")
    assert body["saved_column"], body["save_note"]
    dataset_manager.evict(did)
    info = (await client.get(f"{API}/datasets/{did}")).json()
    assert _columns(info)[body["saved_column"]]["saved_score"]["method"] == "aucell"


async def test_large_dataset_writes_the_score_in_place(client, tmp_data_dir):
    h5ad = register(tmp_data_dir, "bigS", counts_adata())
    with h5py.File(h5ad, "r") as f:
        x_offset = f["X"]["data"].id.get_offset()
    r = await client.post(f"{API}/datasets/bigS/genesets/score",
                          json={"gene_set": ["G10", "G11", "G12", "G40"], "score_name": "mine", "save": True})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["saved_column"] == "score_mine", body["save_note"]
    back = ad.read_h5ad(h5ad, backed="r")
    np.testing.assert_allclose(back.obs["score_mine"].to_numpy(), body["scores"], rtol=1e-6)
    assert json.loads(back.uns[saved_scores.RECORD_KEY])["score_mine"]["source"] == "X"
    back.file.close()
    with h5py.File(h5ad, "r") as f:
        assert f["X"]["data"].id.get_offset() == x_offset  # X neither read nor rewritten
    r = await client.delete(f"{API}/datasets/bigS/scores/score_mine")
    assert r.status_code == 200, r.text
    back = ad.read_h5ad(h5ad, backed="r")
    assert "score_mine" not in back.obs
    back.file.close()


async def test_activity_cells_save_one_source(client, dataset_manager, networks):  # noqa: F811
    from tests.test_activity import _register as register_act

    did = register_act(dataset_manager, _lognorm(_cells()), "actsave")
    r = await client.get(f"{API}/datasets/{did}/activity/progeny/cells",
                         params={"source": "JAK-STAT", "save": True})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["saved_column"] == "activity_JAK-STAT", body["save_note"]
    rec = (await client.get(f"{API}/datasets/{did}/scores")).json()["scores"]["activity_JAK-STAT"]
    assert rec["kind"] == "activity" and rec["network"] == "progeny" and rec["method"] == "ulm"
    assert rec["source"] == "log-normalised X" and "PROGENy" in rec["resource"]
    info = (await client.get(f"{API}/datasets/{did}")).json()
    assert _columns(info)["activity_JAK-STAT"]["saved_score"]["network"] == "progeny"
    # Without save (a session drawing it again), nothing changes.
    r = await client.get(f"{API}/datasets/{did}/activity/progeny/cells", params={"source": "JAK-STAT"})
    assert r.json().get("saved_column") is None


# --- replay -------------------------------------------------------------------


def _replayed(adata, steps: list[dict]) -> str:
    for s in steps:
        provenance.record_step(adata, step="save_score", tool="t", params=s)
    return replay_script.build_script(adata)


def test_replay_scores_a_saved_set_with_score_genes_and_aucell():
    a = _adata()
    genes = ["G0", "G1", "G2", "G3", "G4", "G5"]
    rk = aucell.rankings(a)
    expected_au = aucell.score(rk, aucell.match(rk, genes).idx)
    script = _replayed(a.copy(), [
        {"column": "score_s", "kind": "geneset", "name": "s", "method": "score_genes",
         "source": "log-normalised .raw", "genes": genes},
        {"column": "score_a", "kind": "geneset", "name": "a", "method": "aucell",
         "source": rk.source, "genes": genes, "max_rank": rk.max_rank},
        {"column": "score_g", "kind": "geneset", "name": "g", "method": "gsva", "genes": genes},
        {"column": "activity_STAT1", "kind": "activity", "name": "STAT1", "method": "ulm",
         "network": "collectri"},
    ])
    compile(script, "replay.py", "exec")
    assert "aucell_scores(adata.raw.X, adata.raw.var_names, genes, " in script
    assert "sc.tl.score_genes(adata, genes, score_name='score_s', use_raw=True)" in script
    assert "decoupler-py's ulm" in script and "scView's own gsva code" in script
    statuses = [d["status"] for d in replay_script.describe(_with_steps(a.copy()))]
    assert statuses == ["exact", "exact"]
    # The replayed code computes what scView did.
    import scipy.sparse as sp

    ns: dict = {"np": np, "pd": pd, "sc": sc, "ad": ad, "sp": sp}
    exec(replay_script._HELPERS["aucell"], ns)
    got = ns["aucell_scores"](a.raw.X, a.raw.var_names, genes, rk.max_rank)
    np.testing.assert_array_equal(got, expected_au)
    ref = a.copy()
    sc.tl.score_genes(ref, genes, score_name="score_s", use_raw=True)
    from scview.api.v1.genesets import _score_genes

    app, *_ = _score_genes(a, genes, genes, "x")
    np.testing.assert_allclose(ref.obs["score_s"].to_numpy(), app, rtol=1e-6, atol=1e-7)


def _with_steps(a):
    provenance.record_step(a, step="save_score", tool="t", params={
        "column": "score_s", "kind": "geneset", "method": "score_genes", "source": "X", "genes": ["G0"]})
    provenance.record_step(a, step="remove_score", tool="t", params={"column": "score_s"})
    return a


def test_replay_drops_a_removed_score():
    script = replay_script.build_script(_with_steps(_adata()))
    assert "adata.obs.drop(columns=['score_s'], inplace=True, errors=\"ignore\")" in script
    assert "use_raw=False" in script
