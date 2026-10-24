"""Marker specificity columns: tau and detection fraction against the next-best group.

Computed once with the markers, kept inside their uns entry and saved with them;
markers saved before the columns existed get them on their first visit.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
import scipy.sparse as sp

from scview.api.v1 import markers as markers_api
from scview.core import marker_specificity as ms
from scview.core import pipeline
from scview.core.uns_keys import markers_key

API = "/api/v1"


def _adata(dense: bool = False) -> ad.AnnData:
    """Three groups of 40 cells. G0 is detected in T only; G8-15 are high in B;
    G1 is detected in every cell."""
    rng = np.random.default_rng(0)
    X = rng.poisson(0.4, size=(120, 40)).astype("float32")
    X[:40, :8] += 6
    X[40:80, 8:16] += 6
    X[40:, 0] = 0
    X[:, 1] = 3
    a = ad.AnnData(X if dense else sp.csr_matrix(X))
    a.var_names = [f"G{i}" for i in range(40)]
    a.obs_names = [f"C{i}" for i in range(120)]
    a.obs["cell_type"] = np.array(["T", "B", "NK"]).repeat(40)
    sc.pp.normalize_total(a)
    sc.pp.log1p(a)
    a.raw = a.copy()
    return a


def _register(dm, a: ad.AnnData, did: str = "sp") -> Path:
    d = Path(dm.data_dir) / "uploads" / did
    d.mkdir(parents=True, exist_ok=True)
    (d / "metadata.json").write_text(json.dumps({"id": did, "name": did, "status": "ready"}))
    path = d / "data.h5ad"
    a.write_h5ad(path)
    return path


def _expected_tau(a: ad.AnnData, gene: str) -> float:
    x = a.raw[:, gene].X
    x = np.asarray(x.toarray() if sp.issparse(x) else x).ravel()
    means = pd.Series(x).groupby(a.obs["cell_type"].to_numpy()).mean().to_numpy()
    return float(np.sum(1 - means / means.max()) / (len(means) - 1))


@pytest.fixture()
def count_specificity(monkeypatch):
    calls = []
    real = ms.compute

    def counting(adata, rgg, column):
        calls.append(column)
        return real(adata, rgg, column)

    monkeypatch.setattr(ms, "compute", counting)
    return calls


async def _markers(client, did="sp", **params):
    r = await client.get(f"{API}/datasets/{did}/markers",
                         params={"format": "json", "groupby_column": "cell_type",
                                 "n_genes": 40, **params})
    assert r.status_code == 200, r.text
    return r.json()


# --- the numbers ------------------------------------------------------------------


def test_tau_bounds():
    means = np.array([[2.0, 1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0, 1.0],
                      [0.0, 1.0, 0.0, 3.0]])
    t = ms.tau(means)
    assert t[0] == pytest.approx(1.0)  # one group only
    assert t[1] == pytest.approx(0.0)  # even
    assert np.isnan(t[2])  # never expressed
    assert t[3] == pytest.approx((1 + 2 / 3) / 2)
    assert np.isnan(ms.tau(np.ones((1, 3)))).all()  # one group: undefined


@pytest.mark.parametrize("dense", [False, True])
def test_group_sums_match_pandas_in_blocks(monkeypatch, dense):
    monkeypatch.setattr(ms, "_CHUNK", 7)  # several blocks, uneven last one
    a = _adata(dense)
    X = a.raw.X
    codes = pd.Categorical(a.obs["cell_type"], categories=["T", "B", "NK"]).codes.astype(np.int64)
    codes[5] = -1  # a cell outside every group is left out
    sums, detected = ms._group_sums(X, codes, 3)
    D = X.toarray() if sp.issparse(X) else np.asarray(X)
    for g in range(3):
        rows = D[codes == g]
        np.testing.assert_allclose(sums[g], rows.sum(axis=0), rtol=1e-5)
        np.testing.assert_allclose(detected[g], (rows > 0).sum(axis=0))


def test_a_dense_matrix_is_read_in_small_blocks(monkeypatch):
    # A dense block is held under _DENSE_ELEMENTS values, not _CHUNK cells, so a
    # 50k x 30k in-memory X is never copied (and upcast) in multi-GB slices.
    monkeypatch.setattr(ms, "_DENSE_ELEMENTS", 100)  # 2 cells of 40 genes
    a = _adata(dense=True)
    seen: list[int] = []

    class Recording:
        shape = a.raw.X.shape

        def __getitem__(self, rows):
            seen.append(rows.stop - rows.start)
            return a.raw.X[rows]

    codes = pd.Categorical(a.obs["cell_type"], categories=["T", "B", "NK"]).codes.astype(np.int64)
    sums, _ = ms._group_sums(Recording(), codes, 3)
    assert max(seen) == 2 and sum(seen) == 120
    np.testing.assert_allclose(sums[0], np.asarray(a.raw.X)[:40].sum(axis=0), rtol=1e-5)


def test_mismatched_groups_give_no_columns():
    a = _adata()
    sc.tl.rank_genes_groups(a, "cell_type", method="wilcoxon", pts=True)
    rgg = a.uns["rank_genes_groups"]
    a.obs["cell_type"] = np.array(["x", "y", "z"]).repeat(40)  # relabelled since
    assert ms.attach(a, rgg, "cell_type") is False
    assert not ms.has_specificity(rgg)
    assert ms.attach(a, rgg, "no_such_column") is False


# --- through the API ---------------------------------------------------------------


@pytest.mark.parametrize("dense", [False, True])
async def test_columns_in_the_markers_table(client, dataset_manager, dense):
    a = _adata(dense)
    _register(dataset_manager, a)
    body = await _markers(client)
    rows = {r["gene"]: r for r in body["markers"]["T"]}

    only_t = rows["G0"]
    assert only_t["tau"] == pytest.approx(1.0, abs=1e-6)
    assert only_t["pct_in"] > 0.9 and only_t["pct_max_other"] == 0
    assert only_t["pct_ratio"] is None  # infinite: no other group detects it (JSON null)

    shared = rows["G1"]
    assert shared["tau"] == pytest.approx(_expected_tau(a, "G1"), rel=1e-4)
    assert shared["tau"] < 0.6
    assert shared["pct_ratio"] == pytest.approx(1.0)

    b_gene = rows["G9"]  # high in B: the best other group for T is B
    assert b_gene["max_other_group"] == "B"
    assert b_gene["tau"] == pytest.approx(_expected_tau(a, "G9"), rel=1e-4)
    pts = dataset_manager.get_dataset("sp").adata.uns[markers_key("cell_type")]["pts"]
    assert b_gene["pct_max_other"] == pytest.approx(pts.loc["G9", ["B", "NK"]].max())
    assert b_gene["pct_ratio"] == pytest.approx(pts.loc["G9", "T"] / pts.loc["G9", "B"])


async def test_computed_once_and_saved_with_the_markers(
    client, dataset_manager, count_specificity
):
    _register(dataset_manager, _adata())
    await _markers(client)
    await _markers(client, groupby="B")
    # tau comes with the ranking itself (core/rank_genes.py, from its group means): no
    # second pass over the matrix
    assert count_specificity == []

    derived = Path(dataset_manager.data_dir) / "derived" / "sp" / "data.h5ad"
    saved = ad.read_h5ad(derived).uns[markers_key("cell_type")]
    assert ms.has_specificity(saved)

    dataset_manager.evict("sp")
    body = await _markers(client)
    assert body["markers"]["T"][0]["tau"] is not None
    assert count_specificity == []


async def test_markers_saved_before_get_them_on_first_visit(
    client, dataset_manager, count_specificity
):
    a = _adata()
    a.obs["cell_type"] = a.obs["cell_type"].astype("category")
    sc.tl.rank_genes_groups(a, "cell_type", method="wilcoxon", pts=True,
                            key_added=markers_key("cell_type"))
    _register(dataset_manager, a)

    body = await _markers(client)
    assert body["markers"]["NK"][0]["tau"] is not None
    assert count_specificity == ["cell_type"]
    await _markers(client)
    assert count_specificity == ["cell_type"]  # kept, not recomputed


async def test_a_files_own_markers_without_pts_get_detection_fractions(client, dataset_manager):
    a = _adata()
    a.obs["cell_type"] = a.obs["cell_type"].astype("category")
    sc.tl.rank_genes_groups(a, "cell_type", method="wilcoxon")  # bare key, no pts
    _register(dataset_manager, a)
    r = await client.get(f"{API}/datasets/sp/markers", params={"format": "json", "n_genes": 40})
    assert r.status_code == 200, r.text
    rows = {m["gene"]: m for m in r.json()["markers"]["T"]}
    assert rows["G0"]["tau"] == pytest.approx(1.0, abs=1e-6)
    assert rows["G0"]["pct_max_other"] == 0
    assert rows["G9"]["max_other_group"] == "B"


async def test_rename_keeps_tau_and_follows_the_group(client, dataset_manager):
    _register(dataset_manager, _adata())
    before = await _markers(client)
    r = await client.post(f"{API}/datasets/sp/metadata/cell_type/rename",
                          json={"old": "B", "new": "B cell"})
    assert r.status_code == 200, r.text
    after = await _markers(client)
    t_before = {m["gene"]: m for m in before["markers"]["T"]}
    t_after = {m["gene"]: m for m in after["markers"]["T"]}
    assert t_after["G9"]["max_other_group"] == "B cell"
    assert t_after["G9"]["tau"] == pytest.approx(t_before["G9"]["tau"])


async def test_in_the_markers_csv_export(client, dataset_manager):
    _register(dataset_manager, _adata())
    r = await client.post(f"{API}/datasets/sp/export", json={
        "format": "csv", "data_type": "markers",
        "params": {"column": "cell_type", "n_genes": 40},
    })
    # The export does not compute markers; it reads stored ones.
    assert r.status_code == 404
    await _markers(client)
    r = await client.post(f"{API}/datasets/sp/export", json={
        "format": "csv", "data_type": "markers",
        "params": {"column": "cell_type", "n_genes": 40},
    })
    assert r.status_code == 200, r.text
    df = pd.read_csv(io.BytesIO(r.content))
    for col in ("tau", "pct_max_other", "max_other_group", "pct_ratio"):
        assert col in df.columns
    g0 = df[(df["group"] == "T") & (df["gene"] == "G0")].iloc[0]
    assert g0["tau"] == pytest.approx(1.0, abs=1e-6)
    assert np.isinf(g0["pct_ratio"])


async def test_export_adds_columns_to_markers_saved_before(client, dataset_manager):
    a = _adata()
    a.obs["cell_type"] = a.obs["cell_type"].astype("category")
    sc.tl.rank_genes_groups(a, "cell_type", method="wilcoxon", pts=True,
                            key_added=markers_key("cell_type"))
    _register(dataset_manager, a)
    r = await client.post(f"{API}/datasets/sp/export", json={
        "format": "xlsx", "data_type": "markers", "params": {"column": "cell_type"},
    })
    assert r.status_code == 200, r.text
    df = pd.read_excel(io.BytesIO(r.content))
    assert {"tau", "pct_ratio"} <= set(df.columns)


# --- the pipeline -----------------------------------------------------------------


def test_pipeline_marker_steps_attach_specificity():
    a = _adata()
    pipeline._markers_for_column(a, "cell_type")
    assert ms.has_specificity(a.uns[markers_key("cell_type")])

    b = _adata()
    b.obs["cell_type"] = b.obs["cell_type"].astype("category")
    pipeline._run_marker_genes(b, pipeline.PipelineParams(marker_columns=["cell_type"]))
    rgg = b.uns[markers_key("cell_type")]
    assert ms.has_specificity(rgg)
    assert rgg[ms.SPECIFICITY_FIELD].loc["G0", "tau"] == pytest.approx(1.0, abs=1e-6)


def test_on_demand_compute_keeps_the_markers_when_specificity_fails(monkeypatch, dataset_manager):
    a = _adata()

    def boom(*_args, **_kwargs):
        raise RuntimeError("no")

    monkeypatch.setattr(ms, "compute", boom)
    monkeypatch.setattr(ms, "tau", boom)  # what the ranking's own frame is made with
    rgg = markers_api._compute_markers(a, "cell_type")
    assert ms.attach(a, rgg, "cell_type") is False
    assert "names" in rgg and not ms.has_specificity(rgg)
