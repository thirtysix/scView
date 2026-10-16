"""Clustering resolution sweep (core/resolution_sweep.py + /clustering/sweep, /keep)."""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scanpy as sc

from scview.core import provenance
from scview.core import resolution_sweep as rs
from scview.core.pipeline import PipelineParams, _run_clustering

API = "/api/v1"


def _adata(graph: bool = True) -> ad.AnnData:
    """One diffuse population with a little structure: the cluster count grows
    with resolution (well-separated blobs give the same clusters at every one)."""
    rng = np.random.default_rng(0)
    pca = rng.normal(0, 1, (400, 10)).astype("float32")
    pca[:150, 0] += 3
    a = ad.AnnData(np.zeros((400, 3), dtype="float32"))
    a.obs_names = [f"c{i}" for i in range(400)]
    a.obsm["X_pca"] = pca
    if graph:
        sc.pp.neighbors(a, n_neighbors=15, use_rep="X_pca")
    return a


def test_clean_resolutions_defaults_rounds_dedupes_and_caps():
    assert rs.clean_resolutions(None) == list(rs.DEFAULT_RESOLUTIONS)
    assert rs.clean_resolutions([1.0, 0.304, 0.3, 0.5]) == [0.3, 0.5, 1.0]
    with pytest.raises(rs.SweepError):
        rs.clean_resolutions([0.1 * i for i in range(1, 10)])  # 9 values
    for bad in ([0.0], [-1], [5.5], [float("nan")]):
        with pytest.raises(rs.SweepError):
            rs.clean_resolutions(bad)


def test_column_name_matches_the_pipeline():
    a = _adata()
    _run_clustering(a, PipelineParams(clustering_resolution=1.0))
    assert rs.column_name("leiden", 1) in a.obs.columns
    assert rs.column_name("louvain", 0.5) == "scview_louvain_r0.5"


def test_check_explains_a_missing_graph_and_caps_cells(monkeypatch):
    with pytest.raises(rs.SweepError, match="neighbour graph"):
        rs.check(_adata(graph=False))
    monkeypatch.setattr(rs, "MAX_CELLS", 100)
    with pytest.raises(rs.SweepError, match="at most 100 cells"):
        rs.check(_adata())


def test_dataset_method_reads_provenance_then_the_active_column():
    a = _adata()
    assert rs.dataset_method(a) == "leiden"
    a.uns["scview_active_clustering"] = "scview_louvain_r0.5"
    assert rs.dataset_method(a) == "louvain"
    provenance.set_current(a, clustering={"column": "x", "method": "leiden"})
    assert rs.dataset_method(a) == "leiden"


@pytest.mark.parametrize("method", ["leiden", "louvain"])
def test_cluster_gives_the_pipelines_labels(method):
    a = _adata()
    _run_clustering(a, PipelineParams(clustering_method=method, clustering_resolution=0.8))
    ours = rs.cluster(a, method, 0.8)
    assert list(ours.astype(str)) == list(a.obs[f"scview_{method}_r0.8"].astype(str))


def test_sweep_scores_each_resolution_and_leaves_the_dataset_alone():
    a = _adata()
    obs_before, uns_before = list(a.obs.columns), set(a.uns.keys())
    out = rs.sweep(a, [0.2, 0.6, 1.5], "leiden")
    assert out["method"] == "leiden" and out["n_cells"] == 400
    assert out["representation"] == "X_pca" and out["silhouette_cells"] == 400
    rows = out["results"]
    assert [r["resolution"] for r in rows] == [0.2, 0.6, 1.5]
    assert rows[0]["n_clusters"] < rows[-1]["n_clusters"]
    for r in rows:
        assert sum(r["sizes"]) == 400 and r["sizes"] == sorted(r["sizes"], reverse=True)
        assert r["n_clusters"] == len(r["sizes"])
        assert r["silhouette"] is None or -1 <= r["silhouette"] <= 1
        assert r["column"] == rs.column_name("leiden", r["resolution"]) and not r["exists"]
    assert rows[0]["ari_previous"] is None
    assert all(-1 <= r["ari_previous"] <= 1 for r in rows[1:])
    assert list(a.obs.columns) == obs_before and set(a.uns.keys()) == uns_before


def test_silhouette_uses_a_seeded_subsample(monkeypatch):
    monkeypatch.setattr(rs, "SILHOUETTE_CELLS", 100)
    a = _adata()
    first = rs.sweep(a, [1.0], "leiden")
    again = rs.sweep(a, [1.0], "leiden")
    assert first["silhouette_cells"] == 100
    assert first["results"][0]["silhouette"] == again["results"][0]["silhouette"]


def test_keep_saves_the_column_once_without_changing_the_active_clustering():
    a = _adata()
    a.uns["scview_active_clustering"] = "leiden"
    res = rs.keep(a, "leiden", 0.6)
    assert res == {"column": "scview_leiden_r0.6", "created": True,
                   "n_clusters": a.obs["scview_leiden_r0.6"].nunique()}
    assert a.uns["scview_active_clustering"] == "leiden"
    step = provenance.read_provenance(a)["history"][-1]
    assert step["step"] == "keep_resolution"
    assert step["params"]["clustering_resolution"] == 0.6
    assert rs.keep(a, "leiden", 0.6)["created"] is False
    assert len(provenance.read_provenance(a)["history"]) == 1


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


def _register(dm, a: ad.AnnData) -> str:
    d = Path(dm.data_dir) / "uploads" / "sw"
    d.mkdir(parents=True)
    (d / "metadata.json").write_text(json.dumps({"id": "sw", "name": "sw", "status": "ready"}))
    a.write_h5ad(d / "data.h5ad")
    return "sw"


def _events(text: str) -> list[tuple[str, dict]]:
    out = []
    for block in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        out.append((lines["event"], json.loads(lines["data"])))
    return out


async def test_sweep_endpoint_streams_one_result_per_resolution(client, dataset_manager):
    did = _register(dataset_manager, _adata())
    info = (await client.get(f"{API}/datasets/{did}/clustering/sweep")).json()
    assert info["available"] is True and info["method"] == "leiden"
    assert info["default_resolutions"] == list(rs.DEFAULT_RESOLUTIONS)

    r = await client.post(f"{API}/datasets/{did}/clustering/sweep", json={"resolutions": [0.4, 1.0]})
    assert r.status_code == 200, r.text
    events = _events(r.text)
    assert [e for e, _ in events] == ["start", "result", "result", "complete"]
    assert events[0][1]["resolutions"] == [0.4, 1.0]
    assert [d["resolution"] for e, d in events if e == "result"] == [0.4, 1.0]
    assert not (Path(dataset_manager.data_dir) / "derived" / did).exists()


async def test_sweep_endpoint_explains_a_missing_graph(client, dataset_manager):
    did = _register(dataset_manager, _adata(graph=False))
    info = (await client.get(f"{API}/datasets/{did}/clustering/sweep")).json()
    assert info["available"] is False and "neighbour graph" in info["reason"]
    r = await client.post(f"{API}/datasets/{did}/clustering/sweep", json={})
    assert r.status_code == 400 and "neighbour graph" in r.json()["detail"]
    r = await client.post(f"{API}/datasets/{did}/clustering/keep", json={"resolution": 0.5})
    assert r.status_code == 400


async def test_keep_endpoint_saves_and_reloads(client, dataset_manager):
    did = _register(dataset_manager, _adata())
    r = await client.post(f"{API}/datasets/{did}/clustering/keep", json={"resolution": 0.8})
    assert r.status_code == 200, r.text
    assert r.json()["column"] == "scview_leiden_r0.8" and r.json()["created"] is True
    derived = ad.read_h5ad(Path(dataset_manager.data_dir) / "derived" / did / "data.h5ad")
    assert isinstance(derived.obs["scview_leiden_r0.8"].dtype, pd.CategoricalDtype)
    info = (await client.get(f"{API}/datasets/{did}")).json()
    assert "scview_leiden_r0.8" in [c["name"] for c in info["obs_columns"]]
    # The sweep now marks it as saved, and keeping it again changes nothing.
    r = await client.post(f"{API}/datasets/{did}/clustering/sweep", json={"resolutions": [0.8]})
    assert _events(r.text)[1][1]["exists"] is True
    r = await client.post(f"{API}/datasets/{did}/clustering/keep", json={"resolution": 0.8})
    assert r.json()["created"] is False
    r = await client.post(f"{API}/datasets/{did}/clustering/keep", json={"resolution": 0.8, "method": "x"})
    assert r.status_code == 400
