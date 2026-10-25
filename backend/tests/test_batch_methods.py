"""Batch-correction methods (core/batch_methods.py): the Scanorama and BBKNN ports, the
pipeline step and neighbours step around them, mixing on a BBKNN graph, the method
comparison (core/batch_compare.py) and its endpoints, replay, preflight and the
large-dataset refusal."""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from scview.core import batch_compare as bc
from scview.core import batch_methods as bm
from scview.core import integration, large_run, preflight, provenance, replay_script
from scview.core.pipeline import (
    PipelineParams,
    _record_pipeline_step,
    _run_batch_correction,
    _run_neighbors,
    _run_pca,
)
from scview.core.step_names import step_hint

API = "/api/v1"
N_PER = 60  # cells per type per batch
TYPES = 3


def _pcs(seed: int = 0, shuffle: bool = True, shift: float = 2.0):
    """Cells of 3 types x 2 batches in 10 dimensions; batch B is shifted. Returns
    (X, batch, type) with the cells shuffled (batches not contiguous)."""
    rng = np.random.default_rng(seed)
    centres = rng.normal(0, 4, size=(TYPES, 10))
    X, batch, kind = [], [], []
    for b, off in (("A", 0.0), ("B", shift)):
        for t in range(TYPES):
            X.append(centres[t] + off + rng.normal(0, 0.5, size=(N_PER, 10)))
            batch += [b] * N_PER
            kind += [f"t{t}"] * N_PER
    X = np.vstack(X).astype(np.float32)
    batch, kind = np.array(batch), np.array(kind)
    if shuffle:
        order = rng.permutation(len(batch))
        X, batch, kind = X[order], batch[order], kind[order]
    return X, batch, kind


def _adata(seed: int = 0) -> ad.AnnData:
    X, batch, kind = _pcs(seed)
    a = ad.AnnData(X=np.zeros((len(batch), 5), np.float32),
                   obs=pd.DataFrame({"batch": pd.Categorical(batch),
                                     "cell_type": pd.Categorical(kind)},
                                    index=[f"c{i}" for i in range(len(batch))]))
    a.obsm["X_pca"] = X
    return a


def _cross_share(Z: np.ndarray, batch: np.ndarray, k: int = 15) -> float:
    from sklearn.neighbors import NearestNeighbors

    idx = NearestNeighbors(n_neighbors=k + 1).fit(Z).kneighbors(Z, return_distance=False)[:, 1:]
    return float((batch[idx] != batch[:, None]).mean())


# --- the methods -----------------------------------------------------------------------


def test_check_method_names_the_choices():
    assert bm.check_method(" BBKNN ") == "bbknn"
    with pytest.raises(ValueError, match="harmony, scanorama, bbknn"):
        bm.check_method("combat")


def test_scanorama_mixes_shifted_batches_and_keeps_the_cell_order():
    X, batch, kind = _pcs(shift=1.0)
    Z = bm.scanorama_correct(X, batch)
    assert Z.shape == X.shape
    assert _cross_share(Z, batch) > _cross_share(X, batch) + 0.15
    # Cell types stay apart.
    assert _cross_share(Z, kind) < 0.05
    # Unsorted batches give what the package gives on them gathered (contiguous, in order
    # of first appearance, as scanpy's wrapper needs), with the user's order kept.
    first = batch[0]
    order = np.concatenate([np.flatnonzero(batch == first), np.flatnonzero(batch != first)])
    Z2 = bm.scanorama_correct(X[order], batch[order])
    np.testing.assert_allclose(Z2, Z[order], atol=1e-6)


def test_scanorama_refuses_a_tiny_batch():
    X, batch, _ = _pcs()
    batch = batch.copy()
    batch[:5] = "tiny"
    with pytest.raises(bm.BatchMethodError, match="more than 20 cells"):
        bm.scanorama_correct(X, batch)


def test_bbknn_takes_the_same_neighbours_from_every_batch():
    X, batch, _ = _pcs()
    dist, conn, params = bm.bbknn_graph(X, batch, neighbors_within_batch=4)
    assert params["n_neighbors"] == 8 and params["bbknn"]["trim"] == 80
    # Each cell's kNN lists: 4 cells of A and 4 of B (itself one of its own batch, at 0).
    for i in range(0, len(batch), 37):
        row = dist[i].indices
        assert (batch[row] == batch[i]).sum() == 3 and (batch[row] != batch[i]).sum() == 4
    assert (conn != conn.T).nnz == 0  # symmetric
    assert conn.diagonal().sum() == 0


def test_bbknn_trim_keeps_each_cells_strongest_links():
    X, batch, _ = _pcs()
    _, conn, _ = bm.bbknn_graph(X, batch, neighbors_within_batch=4, trim=3)
    _, full, _ = bm.bbknn_graph(X, batch, neighbors_within_batch=4, trim=0)
    assert conn.nnz < full.nnz
    assert np.diff(conn.indptr).max() <= np.diff(full.indptr).max()


def test_bbknn_within_gives_about_the_neighbour_count():
    assert bm.bbknn_within(15, 2) == 8
    assert bm.bbknn_within(15, 20) == 1


def test_graph_neighbours_are_the_strongest_links_without_the_cell():
    conn = sp.csr_matrix(np.array([[0, .9, .1, .5], [.9, 0, 0, 0], [.1, 0, 0, .2], [.5, 0, .2, 0]]))
    idx = bm.graph_neighbours(conn, np.array([0, 1]), 2)
    assert idx.tolist() == [[1, 3], [0, 0]]  # cell 1 has one link: repeated


# --- the pipeline steps ------------------------------------------------------------------


@pytest.mark.parametrize("method,key", [("harmony", "X_pca_harmony"), ("scanorama", "X_scanorama")])
def test_embedding_methods_feed_the_neighbour_step(method, key):
    a = _adata()
    p = PipelineParams(batch_key="batch", batch_method=method)
    found = _run_batch_correction(a, p)
    assert found["representation"] == key and key in a.obsm
    other = set(bm.CORRECTED_KEYS) - {key}
    assert not other & set(a.obsm.keys())
    got = _run_neighbors(a, p)
    assert got == {"use_rep": key} and a.uns["neighbors"]["params"]["use_rep"] == key


def test_bbknn_replaces_the_graph_and_the_neighbour_step_keeps_it():
    a = _adata()
    a.obsm["X_pca_harmony"] = a.obsm["X_pca"].copy()  # an older Harmony run
    p = PipelineParams(batch_key="batch", batch_method="bbknn", n_neighbors=10)
    found = _run_batch_correction(a, p)
    assert found["graph"] == "bbknn" and found["neighbors_within_batch"] == 5
    assert "X_pca_harmony" not in a.obsm and bm.is_bbknn_graph(a)
    graph = a.obsp["connectivities"].copy()
    _record_pipeline_step(a, "batch_correction", p, found)
    kept = _run_neighbors(a, p)
    assert kept["kept"] is True
    assert (a.obsp["connectivities"] != graph).nnz == 0
    _record_pipeline_step(a, "neighbors", p, kept)
    hist = provenance.read_provenance(a)["history"]
    assert "BBKNN port" in hist[-2]["tool"] and hist[-2]["params"]["batch_method"] == "bbknn"
    assert hist[-1]["tool"].startswith("BBKNN graph") and hist[-1]["params"]["kept"] is True
    # A new PCA drops the mark: the next neighbour step builds its own graph.
    a.X = np.random.default_rng(0).normal(size=a.X.shape).astype(np.float32)
    _run_pca(a, PipelineParams(n_comps=3))
    assert not bm.is_bbknn_graph(a)
    assert _run_neighbors(a, p) == {"use_rep": "X_pca"}


def test_unknown_batch_key_is_a_clear_error():
    with pytest.raises(ValueError, match="not found"):
        _run_batch_correction(_adata(), PipelineParams(batch_key="nope", batch_method="scanorama"))


# --- mixing on the joint dataset -----------------------------------------------------------


def test_batch_mixing_scores_a_bbknn_graph_from_its_links():
    a = _adata()
    a.obs = a.obs.rename(columns={"batch": "dataset"})
    _run_batch_correction(a, PipelineParams(batch_key="dataset", batch_method="bbknn"))
    m = integration.batch_mixing(a, k=10)
    assert m["spaces"] == {"uncorrected": "X_pca", "corrected": "graph"}
    assert m["k_by_space"] == {"uncorrected": 10, "corrected": 10}
    assert m["corrected"] > m["uncorrected"]
    assert m["conservation"]["label_column"] == "cell_type"


def test_batch_mixing_takes_precomputed_neighbours():
    a = _adata()
    a.obs = a.obs.rename(columns={"batch": "dataset"})
    n = a.n_obs
    m = integration.batch_mixing(
        a, k=5, neighbours={"corrected": lambda q, k: np.tile((q[:, None] + 1) % n, (1, 3))})
    assert m["spaces"]["corrected"] == "neighbours" and m["k_by_space"]["corrected"] == 3


def test_joint_runs_take_every_method():
    assert integration.METHODS == ("harmony", "scanorama", "bbknn", "none")
    assert "batch_correction" in integration.joint_steps("bbknn")
    assert "batch_correction" not in integration.joint_steps("none")
    p = integration.joint_params("scanorama")
    assert (p.batch_key, p.batch_method) == ("dataset", "scanorama")
    assert integration.joint_params("none").batch_key == ""


def test_clonotype_ids_are_prefixed_with_the_dataset_label():
    obs = pd.DataFrame({
        "clonotype": pd.Categorical(["1", "1", "2", "no clonotype", None] + [str(i) for i in range(3, 8)]),
        "cell_type": pd.Categorical(["T"] * 10),
    }, index=[f"c{i}" for i in range(10)])
    out = integration.carried_obs(ad.AnnData(obs=obs), "A")
    assert out["clonotype"].tolist()[:5] == ["A:1", "A:1", "A:2", "no clonotype", None]
    assert out["cell_type"].tolist()[0] == "T"
    # Two datasets' "1" no longer collide.
    other = integration.carried_obs(ad.AnnData(obs=obs), "B")
    assert set(out["clonotype"].dropna()) & set(other["clonotype"].dropna()) == {"no clonotype"}


# --- the comparison --------------------------------------------------------------------------


def test_compare_scores_every_method_and_recommends_by_the_rule():
    events = list(bc.iter_compare(_adata(), "batch", n_neighbors=10))
    kinds = [e for e, _ in events]
    assert kinds[0] == "start" and kinds[-1] == "result"
    assert kinds.count("method") == 4
    start = events[0][1]
    assert start["label_column"] == "cell_type" and start["k"] == 10
    res = events[-1][1]
    rows = {r["method"]: r for r in res["rows"]}
    assert rows["none"]["mixing_ratio"] < 0.2
    for m in ("harmony", "scanorama", "bbknn"):
        assert rows[m]["mixing_ratio"] > rows["none"]["mixing_ratio"], m
    assert res["recommended"] in ("harmony", "scanorama", "bbknn")
    assert "Luecken" in res["background"] and res["notes"][0] == bc.BBKNN_NOTE


def test_compare_stops_between_methods():
    calls = {"n": 0}

    def stop():
        calls["n"] += 1
        return calls["n"] > 2

    events = list(bc.iter_compare(_adata(), "batch", stop=stop))
    assert events[-1][0] == "stopped" and events[-1][1]["done"] == ["none", "harmony"]


def test_compare_needs_pca_and_two_batches():
    a = _adata()
    with pytest.raises(bc.CompareError, match="only one batch"):
        bc.check(a[a.obs["batch"] == "A"].copy(), "batch")
    del a.obsm["X_pca"]
    with pytest.raises(bc.CompareError, match="Run PCA first"):
        bc.check(a, "batch")


def _row(method, ratio, cons, seconds=1.0, error=None):
    return {"method": method, "label": bm.LABELS[method], "mixing_ratio": ratio,
            "conservation": cons, "seconds": seconds, **({"error": error} if error else {})}


def test_recommend_best_conservation_among_methods_that_mix():
    rows = [_row("none", 0.1, 0.90, 0.1), _row("harmony", 0.7, 0.80, 0.5),
            _row("scanorama", 0.6, 0.86, 4), _row("bbknn", 0.3, 0.89, 2)]
    bc._compare_to_none(rows)
    r = bc.recommend(rows, "cell_type", "shared")
    assert r["recommended"] == "scanorama"  # bbknn keeps types better but does not mix
    assert "86%" in r["reason"]
    assert [w.split(":")[0] for w in r["warnings"]] == ["Harmony"]  # 90% -> 80%


def test_recommend_ties_go_to_the_faster_and_none_first():
    rows = [_row("none", 0.1, 0.9, 0.1), _row("harmony", 0.7, 0.85, 0.5),
            _row("scanorama", 0.8, 0.86, 4)]
    r = bc.recommend(rows, "cell_type", "shared")
    assert r["recommended"] == "harmony" and "fastest" in r["reason"]
    mixed = [_row("none", 0.8, 0.9, 0.1), _row("harmony", 0.9, 0.9, 0.5)]
    assert bc.recommend(mixed, "cell_type", "shared")["recommended"] == "none"


def test_recommend_without_labels_or_without_mixing():
    rows = [_row("none", 0.1, None, 0.1), _row("harmony", 0.7, None, 0.5),
            _row("scanorama", 0.8, None, 4)]
    r = bc.recommend(rows, None, "")
    assert r["recommended"] == "harmony" and any("No cell-type" in n for n in r["notes"])
    weak = [_row("none", 0.1, 0.9), _row("harmony", 0.3, 0.9), _row("bbknn", 0.2, 0.9, error=None)]
    r = bc.recommend(weak, "cell_type", "shared")
    assert r["recommended"] == "harmony" and "half of perfect" in r["reason"]


# --- endpoints ------------------------------------------------------------------------------


def _register(dm, a: ad.AnnData, did: str = "bcmp") -> str:
    d = Path(dm.data_dir) / "uploads" / did
    d.mkdir(parents=True)
    (d / "metadata.json").write_text(json.dumps({"id": did, "name": did, "status": "ready"}))
    a.write_h5ad(d / "data.h5ad")
    return did


def _events(text: str) -> list[tuple[str, dict]]:
    out = []
    for block in text.strip().split("\n\n"):
        if block.startswith(":"):
            continue
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        out.append((lines["event"], json.loads(lines["data"])))
    return out


async def test_compare_endpoints(client, dataset_manager):
    did = _register(dataset_manager, _adata())
    r = await client.get(f"{API}/datasets/{did}/batch/compare", params={"batch_key": "batch"})
    assert r.status_code == 200
    info = r.json()
    assert info["available"] and info["label_column"] == "cell_type"
    assert info["estimate_seconds"] > 0 and "Luecken" in info["background"]
    r = await client.get(f"{API}/datasets/{did}/batch/compare", params={"batch_key": "nope"})
    assert r.json()["available"] is False and "not found" in r.json()["reason"]

    r = await client.post(f"{API}/datasets/{did}/batch/compare-stream",
                          json={"batch_key": "batch", "methods": ["none", "harmony"]})
    assert r.status_code == 200
    ev = _events(r.text)
    assert [e for e, _ in ev] == ["start", "method_start", "method", "method_start", "method",
                                  "result", "complete"]
    assert ev[-2][1]["recommended"] == "harmony"
    # Nothing saved: the dataset keeps no corrected PCs.
    adaptor = await dataset_manager.get_or_load_dataset(did)
    assert "X_pca_harmony" not in adaptor.adata.obsm

    r = await client.post(f"{API}/datasets/{did}/batch/compare-stream",
                          json={"batch_key": "batch", "methods": ["combat"]})
    assert r.status_code == 422
    r = await client.post(f"{API}/datasets/{did}/batch/compare-stream", json={"batch_key": "nope"})
    assert r.status_code == 400


async def test_a_run_with_an_unknown_batch_method_is_refused(client, dataset_manager):
    did = _register(dataset_manager, _adata(), "bcmp2")
    r = await client.post(f"{API}/datasets/{did}/assessment/run",
                          json={"steps": ["batch_correction"],
                                "params": {"batch_key": "batch", "batch_method": "combat"}})
    assert r.status_code == 422 and "choose from harmony, scanorama, bbknn" in r.json()["detail"]


# --- large mode, step names, preflight, replay ------------------------------------------------


def test_large_mode_refuses_scanorama_and_bbknn_but_not_harmony():
    for method in ("scanorama", "bbknn"):
        why = large_run.refused(["batch_correction"], PipelineParams(batch_method=method))
        assert why and "not available for large datasets" in why and "Harmony" in why
    assert large_run.refused(["batch_correction"], PipelineParams(batch_method="harmony")) is None
    assert large_run.refused(["pca"], PipelineParams(batch_method="bbknn")) is None


def test_step_hints_for_the_method_names():
    assert step_hint("bbknn") == "batch_correction"
    assert step_hint("Scanorama") == "batch_correction"


def _state(n: int) -> preflight._State:
    return preflight._State(n, 2000, None, {}, None, 0)


def test_preflight_estimates_each_method():
    kang = 13_836
    _, s, harmony = preflight._step_batch(_state(kang), PipelineParams(batch_method="harmony"))
    t, s, scan = preflight._step_batch(_state(kang), PipelineParams(batch_method="scanorama"))
    assert 4.9 * 0.8 <= scan <= 4.9 * 1.2  # measured 4.87 s
    assert 160e6 <= t <= 240e6  # measured Python peak 190 MiB
    _, half_s, half = preflight._step_batch(_state(kang // 2), PipelineParams(batch_method="scanorama"))
    assert 1.93 * 0.8 <= half <= 1.93 * 1.2  # measured 1.93 s
    _, s, bb = preflight._step_batch(_state(kang), PipelineParams(batch_method="bbknn"))
    assert 29.0 * 0.8 <= bb <= 29.0 * 1.2 and s.bbknn_graph  # measured 29.0 s
    assert 0.28 * 0.8 <= harmony <= 0.28 * 1.2  # measured 0.28 s
    # The neighbours step then keeps BBKNN's graph: no time, no memory.
    assert preflight._step_neighbors(s, PipelineParams())[0::2] == (0, 0.0)


def _replay(a: ad.AnnData) -> str:
    return replay_script.build_script(a, dataset_name="t", dataset_id="t")


def test_replay_writes_scanorama_and_bbknn_steps():
    a = _adata()
    for method in ("scanorama", "bbknn"):
        p = PipelineParams(batch_key="batch", batch_method=method)
        _record_pipeline_step(a, "batch_correction", p, _run_batch_correction(a, p))
        _record_pipeline_step(a, "neighbors", p, _run_neighbors(a, p))
    text = _replay(a)
    assert "def scanorama(adata, batch_key):" in text
    assert 'adata.obsm["X_scanorama"] = scanorama(adata, \'batch\')' in text
    assert "approx=False" in text
    assert ("bbknn.bbknn(adata, batch_key='batch', neighbors_within_batch=8, n_pcs=10, "
            "trim=160, computation=\"cKDTree\")") in text
    assert "The BBKNN graph from batch correction is the neighbour graph" in text
    compile(text, "replay.py", "exec")
