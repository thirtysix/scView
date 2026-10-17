"""Variable-gene sensitivity check (core/hvg_sensitivity.py + /clustering/hvg-sensitivity)."""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

from scview.core import hvg_sensitivity as hs
from scview.core import normalization, provenance
from scview.core.pipeline import PipelineParams, run_pipeline

API = "/api/v1"
STEPS = ["normalization", "log_transform", "highly_variable_genes", "scaling", "pca", "neighbors",
         "clustering"]


def _counts(n: int = 300, g: int = 400, dtype: str = "float64", seed: int = 0) -> ad.AnnData:
    """Three cell types, each with 40 marker genes, plus a random batch."""
    rng = np.random.default_rng(seed)
    types = np.repeat(np.arange(3), n // 3)
    mu = np.tile(rng.gamma(2.0, 0.5, g), (n, 1))
    for t in range(3):
        mu[types == t, 40 * t:40 * t + 40] *= 6
    a = ad.AnnData(sp.csr_matrix(rng.poisson(mu).astype(dtype)))
    a.var_names = [f"G{i}" for i in range(g)]
    a.obs_names = [f"c{i}" for i in range(n)]
    a.obs["type"] = pd.Categorical([f"T{t}" for t in types])
    a.obs["batch"] = pd.Categorical(np.where(rng.random(n) < 0.5, "b1", "b2"))
    return a


def _processed(steps=STEPS, **params) -> ad.AnnData:
    a, res = run_pipeline(_counts(), steps, PipelineParams(
        n_top_genes=200, n_comps=20, clustering_resolution=0.5, **params))
    assert not res.errors
    return a


@pytest.fixture(autouse=True)
def _empty_cache():
    hs._cache.clear()
    yield
    hs._cache.clear()


# --- what the dataset recorded -------------------------------------------------


def test_recorded_reads_the_pipelines_settings():
    a = _processed()
    rec = hs.recorded(a)
    assert rec.reference == "scview_leiden_r0.5"
    assert (rec.method, rec.resolution) == ("leiden", 0.5)
    assert (rec.flavor, rec.n_top_genes, rec.n_comps, rec.n_neighbors) == ("seurat", 200, 20, 15)
    assert rec.batch_key is None and rec.hvg_batch_key is None and rec.notes == []
    assert rec.current == hs.Setting("seurat", 200)


def test_recorded_takes_the_reference_columns_own_method_and_resolution():
    a = _processed()
    a.obs["scview_louvain_r1.2"] = a.obs["scview_leiden_r0.5"]
    rec = hs.recorded(a, "scview_louvain_r1.2")
    assert (rec.method, rec.resolution) == ("louvain", 1.2)
    # A column scView didn't make is re-clustered as the dataset's own clustering was.
    rec = hs.recorded(a, "type")
    assert (rec.method, rec.resolution) == ("leiden", 0.5) and rec.notes


def test_recorded_follows_pearson_residuals_and_harmony():
    a = _processed(steps=[*STEPS[:5], "batch_correction", *STEPS[5:]], batch_key="batch",
                   normalization_method="pearson_residuals")
    rec = hs.recorded(a)
    assert rec.flavor == "pearson_residuals" and rec.batch_key == "batch"


def test_recorded_explains_what_it_cannot_compare():
    a = _counts()
    with pytest.raises(hs.SensitivityError, match="no clustering"):
        hs.recorded(a)
    a.obs["one"] = "x"
    with pytest.raises(hs.SensitivityError, match="1 groups"):
        hs.recorded(a, "one")
    with pytest.raises(hs.SensitivityError, match="not found"):
        hs.recorded(a, "nope")


def test_recorded_falls_back_for_a_dataset_processed_elsewhere():
    a = _processed()
    del a.uns[provenance.UNS_KEY]
    rec = hs.recorded(a)
    # The column's name, the HVG mask, the PCA and scanpy's own records.
    assert (rec.method, rec.resolution, rec.n_top_genes, rec.n_comps) == ("leiden", 0.5, 200, 20)
    assert rec.flavor == "seurat"


# --- settings -------------------------------------------------------------------


def _wide(n_genes: int = 3500) -> ad.AnnData:
    a = ad.AnnData(sp.random(60, n_genes, density=0.05, format="csr", random_state=0,
                             data_rvs=lambda k: np.ones(k)))
    a.var_names = [f"G{i}" for i in range(n_genes)]
    return a


def _rec(**kw) -> hs.Recorded:
    base = dict(reference="c", method="leiden", resolution=0.5, flavor="seurat", n_top_genes=2000,
                hvg_batch_key=None, target_sum=1e4, n_comps=50, n_neighbors=15, batch_key=None)
    return hs.Recorded(**{**base, **kw})


def test_default_settings_are_the_control_two_counts_and_the_other_flavours(monkeypatch):
    monkeypatch.setattr(hs, "seurat_v3_available", lambda: False)
    keys = [s.key for s in hs.default_settings(_wide(), _rec())]
    assert keys == ["seurat_2000", "seurat_1000", "seurat_3000", "cell_ranger_2000",
                    "pearson_residuals_2000"]
    monkeypatch.setattr(hs, "seurat_v3_available", lambda: True)
    assert [s.key for s in hs.default_settings(_wide(), _rec())][-1] == "seurat_v3_2000"
    # From Pearson residuals, the counts are halved and raised in that flavour.
    keys = [s.key for s in hs.default_settings(_wide(), _rec(flavor="pearson_residuals"))]
    assert keys[:3] == ["pearson_residuals_2000", "pearson_residuals_1000", "pearson_residuals_3000"]
    assert "seurat_2000" in keys


def test_without_counts_only_the_log_flavours_run():
    a = _wide()
    a.X = a.X * 0.5  # not counts
    available = hs.flavors_available(a)
    assert available["seurat"] is None and available["cell_ranger"] is None
    assert "counts" in available["pearson_residuals"]
    keys = [s.key for s in hs.default_settings(a, _rec())]
    assert not any(k.startswith(("pearson", "seurat_v3")) for k in keys)


def test_clean_settings_validates_and_dedupes(monkeypatch):
    monkeypatch.setattr(hs, "seurat_v3_available", lambda: False)
    a, rec = _wide(), _rec()
    got = hs.clean_settings(a, rec, [{"flavor": "seurat", "n_top_genes": 500},
                                     {"flavor": "seurat", "n_top_genes": 500}])
    assert got == [hs.Setting("seurat", 500)]
    assert hs.clean_settings(a, rec, None) == hs.default_settings(a, rec)
    bad = [
        ([{"flavor": "magic", "n_top_genes": 500}], "Unknown flavour"),
        ([{"flavor": "seurat", "n_top_genes": 50}], "between"),
        ([{"flavor": "seurat", "n_top_genes": 3600}], "too many"),
        ([{"flavor": "seurat_v3", "n_top_genes": 500}], "scikit-misc"),
        ([{"flavor": "seurat", "n_top_genes": 300 + 100 * i} for i in range(9)], "at most"),
    ]
    for values, message in bad:
        with pytest.raises(hs.SensitivityError, match=message):
            hs.clean_settings(a, rec, values)


def test_estimate_grows_with_cells_settings_and_harmony():
    one = hs.estimate_seconds(10_000, 1, _rec())
    assert hs.estimate_seconds(20_000, 1, _rec()) > one
    assert hs.estimate_seconds(10_000, 5, _rec()) > 4 * one
    assert hs.estimate_seconds(10_000, 1, _rec(batch_key="b")) > one


def test_check_caps_the_cells(monkeypatch):
    a = _processed()
    monkeypatch.setattr(hs, "MAX_CELLS", 100)
    with pytest.raises(hs.SensitivityError, match="at most 100 cells"):
        hs.check(a)


# --- one setting ------------------------------------------------------------------


def test_the_log_matrix_is_the_pipelines_own_in_the_counts_dtype():
    """float32 instead of Kang's float64 counts moved the PCA by 1e-5, and the
    control lost a 24-cell cluster (ARI 0.990 instead of 1.0)."""
    a = _processed()
    assert a.layers["counts"].dtype == np.float64
    inp = hs._inputs(a, hs.recorded(a))
    assert inp.log.dtype == a.raw.X.dtype
    assert (inp.log != a.raw.X).nnz == 0  # bit for bit


def test_the_control_reproduces_the_dataset():
    """Bit for bit: on Kang, a PCA one float32 step off (row-major input) changed
    a Harmony-corrected Leiden partition (ARI 0.85)."""
    a = _processed()
    rec = hs.recorded(a)
    inp = hs._inputs(a, rec)
    mask, labels = hs.run_setting(inp, rec.current, rec)
    assert (mask == a.var["highly_variable"].to_numpy()).all()
    X = hs._pcs(inp, mask, rec.current, rec)
    assert np.array_equal(X, a.obsm["X_pca"])
    assert adjusted_rand_score(a.obs[rec.reference], np.asarray(labels)) == 1.0


def test_the_pearson_control_reproduces_the_dataset():
    a = _processed(normalization_method="pearson_residuals")
    rec = hs.recorded(a)
    inp = hs._inputs(a, rec)
    mask, labels = hs.run_setting(inp, rec.current, rec)
    assert (mask == a.var["highly_variable"].to_numpy()).all()
    assert np.array_equal(hs._pcs(inp, mask, rec.current, rec), a.obsm["X_pca"])
    assert adjusted_rand_score(a.obs[rec.reference], np.asarray(labels)) == 1.0


def test_the_harmony_control_reproduces_the_dataset():
    a = _processed(steps=[*STEPS[:5], "batch_correction", *STEPS[5:]], batch_key="batch")
    rec = hs.recorded(a)
    inp = hs._inputs(a, rec)
    mask, labels = hs.run_setting(inp, rec.current, rec)
    X = hs._harmony(hs._pcs(inp, mask, rec.current, rec), inp, rec)
    assert np.array_equal(X, a.obsm["X_pca_harmony"])
    assert adjusted_rand_score(a.obs[rec.reference], np.asarray(labels)) == 1.0


def test_pearson_residuals_of_some_columns_use_every_genes_totals():
    counts = _counts().X
    idx = np.array([3, 50, 7, 399])
    full = normalization.pearson_residuals(counts)
    assert np.array_equal(normalization.pearson_residuals(counts, columns=idx), full[:, idx])


# --- comparison ---------------------------------------------------------------------


def test_compare_names_what_became_of_each_cluster():
    # A kept; B split in two; C and D merged; E spread over clusters that hold others.
    ref = ["A"] * 20 + ["B"] * 20 + ["C"] * 20 + ["D"] * 10 + ["E"] * 10 + [None] * 4
    new = (["a"] * 20 + ["b1"] * 10 + ["b2"] * 10 + ["cd"] * 30 + ["a"] * 5 + ["b1"] * 5
           + ["a", "b1", "b2", "cd"])
    obs = pd.DataFrame({"r": pd.Categorical(ref)})
    codes, names = hs._reference_codes(ad.AnnData(obs=obs), "r")
    out = hs.compare(codes, names, pd.Categorical(new))
    by = {c["cluster"]: c for c in out["clusters"]}
    assert by["A"]["status"] == "stable" and by["A"]["match"] == "a"
    assert by["A"]["jaccard"] == pytest.approx(20 / 25)
    assert by["B"]["status"] == "splits" and by["B"]["parts"] == 2
    assert by["C"]["status"] == "merges" and by["C"]["contained"] == 1.0
    assert by["C"]["purity"] == pytest.approx(20 / 30, abs=1e-4)
    assert by["D"]["status"] == "merges"
    assert by["E"]["status"] == "dissolves" and by["E"]["n_cells"] == 10
    # Cells without a reference label are left out of every number.
    keep = np.array([r is not None for r in ref])
    r = np.asarray(ref, dtype=object)[keep]
    c = np.asarray(new, dtype=object)[keep]
    assert out["ari"] == pytest.approx(adjusted_rand_score(r, c), abs=1e-4)
    assert out["ami"] == pytest.approx(adjusted_mutual_info_score(r, c), abs=1e-4)
    assert out["n_clusters"] == 4 and sum(out["sizes"]) == len(new)


def test_iter_sensitivity_streams_each_setting_and_leaves_the_dataset_alone():
    a = _processed()
    before = (list(a.obs.columns), sorted(a.uns.keys()), a.X.copy(), a.uns[provenance.UNS_KEY])
    rec = hs.recorded(a)
    settings = [rec.current, hs.Setting("seurat", 300), hs.Setting("pearson_residuals", 200)]
    rows = list(hs.iter_sensitivity(a, settings, rec))
    head, results = rows[0], rows[1:]
    assert head["reference"] == rec.reference and head["n_current_hvgs"] == 200
    assert [s["key"] for s in head["settings"]] == ["seurat_200", "seurat_300", "pearson_residuals_200"]
    assert head["estimate_seconds"] > 0
    assert [r["key"] for r in results] == [s.key for s in settings]
    control = results[0]
    assert control["is_current"] and control["ari"] == 1.0 and control["shared_hvgs"] == 200
    assert {c["status"] for c in control["clusters"]} == {"stable"}
    assert [c["cluster"] for c in control["clusters"]] == list(a.obs[rec.reference].cat.categories)
    assert results[1]["n_hvgs"] == 300 and results[1]["shared_hvgs"] == 200
    assert results[2]["column"] == "scview_leiden_r0.5_hvg_pearson_residuals_200"
    assert not any(r["exists"] for r in results)
    # The three planted types survive every setting.
    for r in results:
        assert {c["status"] for c in r["clusters"]} == {"stable"}, r["key"]
    after = (list(a.obs.columns), sorted(a.uns.keys()), a.X, a.uns[provenance.UNS_KEY])
    assert before[0] == after[0] and before[1] == after[1] and before[3] == after[3]
    assert np.array_equal(before[2], after[2])


def test_a_stop_ends_the_run_before_its_next_substep(monkeypatch):
    a = _processed()
    rec = hs.recorded(a)
    stop = threading.Event()
    clustered = []
    real = hs._cluster
    monkeypatch.setattr(hs, "_cluster", lambda X, r: clustered.append(1) or real(X, r))
    real_pcs = hs._pcs

    def pcs_then_stop(*args):
        out = real_pcs(*args)
        stop.set()
        return out

    monkeypatch.setattr(hs, "_pcs", pcs_then_stop)
    gen = hs.iter_sensitivity(a, [rec.current], rec, stop=stop.is_set)
    next(gen)
    with pytest.raises(hs.Cancelled):
        next(gen)
    assert clustered == []


# --- keep -------------------------------------------------------------------------------


def test_keep_saves_the_labels_the_run_showed(monkeypatch):
    a = _processed()
    rec = hs.recorded(a)
    s = hs.Setting("cell_ranger", 200)
    shown = list(hs.iter_sensitivity(a, [s], rec))[1]
    monkeypatch.setattr(hs, "run_setting", lambda *a, **k: pytest.fail("not from the cache"))
    out = hs.keep(a, s, rec)
    assert out == {"column": shown["column"], "created": True, "n_clusters": shown["n_clusters"]}
    assert a.uns["scview_active_clustering"] == "scview_leiden_r0.5"  # unchanged
    assert provenance.read_provenance(a)["history"][-1]["step"] == "keep_hvg_setting"
    assert hs.keep(a, s, rec)["created"] is False
    # Saving a column changes nothing the clusters depend on: a second keep is cached too.
    assert hs.fingerprint(a, rec) == next(iter(hs._cache))[0]


def test_keep_computes_again_when_the_run_is_gone():
    a = _processed()
    rec = hs.recorded(a)
    s = hs.Setting("seurat", 300)
    shown = list(hs.iter_sensitivity(a, [s], rec))[1]
    cached = hs._cache[(hs.fingerprint(a, rec), s.key)]
    hs._cache.clear()
    hs.keep(a, s, rec)
    assert np.array_equal(np.asarray(a.obs[shown["column"]]), np.asarray(cached))


def test_the_fingerprint_follows_the_counts_not_saved_columns():
    a = _processed()
    rec = hs.recorded(a)
    fp = hs.fingerprint(a, rec)
    a.obs["scview_leiden_r0.5_hvg_seurat_300"] = a.obs["scview_leiden_r0.5"]
    provenance.record_step(a, step="keep_hvg_setting", tool="t", params={})
    assert hs.fingerprint(a, rec) == fp
    a.layers["counts"] = a.layers["counts"] * 2
    assert hs.fingerprint(a, rec) != fp


# --- endpoints -----------------------------------------------------------------------------


def _register(dm, a: ad.AnnData, did: str = "hv") -> str:
    d = Path(dm.data_dir) / "uploads" / did
    d.mkdir(parents=True)
    (d / "metadata.json").write_text(json.dumps({"id": did, "name": did, "status": "ready"}))
    a.write_h5ad(d / "data.h5ad")
    return did


def _events(text: str) -> list[tuple[str, dict]]:
    out = []
    for block in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        out.append((lines["event"], json.loads(lines["data"])))
    return out


async def test_info_endpoint_gives_the_recorded_settings_and_defaults(client, dataset_manager):
    did = _register(dataset_manager, _processed())
    info = (await client.get(f"{API}/datasets/{did}/clustering/hvg-sensitivity")).json()
    assert info["available"] is True and info["reason"] is None
    assert info["recorded"]["reference"] == "scview_leiden_r0.5"
    assert info["current"]["key"] == "seurat_200"
    assert info["default_settings"][0]["key"] == "seurat_200"
    assert {f["flavor"] for f in info["flavors"]} == set(hs.FLAVORS)
    assert info["estimate_seconds"] > 0 and info["max_settings"] == hs.MAX_SETTINGS


async def test_info_endpoint_explains_a_dataset_without_clusters(client, dataset_manager):
    did = _register(dataset_manager, _counts())
    info = (await client.get(f"{API}/datasets/{did}/clustering/hvg-sensitivity")).json()
    assert info["available"] is False and "no clustering" in info["reason"]
    r = await client.post(f"{API}/datasets/{did}/clustering/hvg-sensitivity", json={})
    assert r.status_code == 400 and "no clustering" in r.json()["detail"]


async def test_run_endpoint_streams_one_result_per_setting(client, dataset_manager):
    did = _register(dataset_manager, _processed())
    body = {"settings": [{"flavor": "seurat", "n_top_genes": 200}, {"flavor": "cell_ranger", "n_top_genes": 200}]}
    r = await client.post(f"{API}/datasets/{did}/clustering/hvg-sensitivity", json=body)
    assert r.status_code == 200, r.text
    events = _events(r.text)
    assert [e for e, _ in events] == ["start", "result", "result", "complete"]
    assert [d["key"] for e, d in events if e == "result"] == ["seurat_200", "cell_ranger_200"]
    assert events[1][1]["is_current"] is True
    assert not (Path(dataset_manager.data_dir) / "derived" / did).exists()
    r = await client.post(f"{API}/datasets/{did}/clustering/hvg-sensitivity",
                          json={"settings": [{"flavor": "magic", "n_top_genes": 200}]})
    assert r.status_code == 400 and "Unknown flavour" in r.json()["detail"]


async def test_keep_endpoint_saves_and_reloads(client, dataset_manager):
    did = _register(dataset_manager, _processed())
    setting = {"flavor": "seurat", "n_top_genes": 300}
    r = await client.post(f"{API}/datasets/{did}/clustering/hvg-sensitivity", json={"settings": [setting]})
    shown = _events(r.text)[1][1]
    r = await client.post(f"{API}/datasets/{did}/clustering/hvg-sensitivity/keep", json=setting)
    assert r.status_code == 200, r.text
    assert r.json() == {"column": shown["column"], "created": True, "n_clusters": shown["n_clusters"]}
    derived = ad.read_h5ad(Path(dataset_manager.data_dir) / "derived" / did / "data.h5ad")
    assert isinstance(derived.obs[shown["column"]].dtype, pd.CategoricalDtype)
    info = (await client.get(f"{API}/datasets/{did}")).json()
    assert shown["column"] in [c["name"] for c in info["obs_columns"]]
    r = await client.post(f"{API}/datasets/{did}/clustering/hvg-sensitivity", json={"settings": [setting]})
    assert _events(r.text)[1][1]["exists"] is True
    r = await client.post(f"{API}/datasets/{did}/clustering/hvg-sensitivity/keep", json=setting)
    assert r.json()["created"] is False
    r = await client.post(f"{API}/datasets/{did}/clustering/hvg-sensitivity/keep",
                          json={"flavor": "seurat", "n_top_genes": 5})
    assert r.status_code == 400


async def test_a_client_going_away_stops_the_run_and_releases_it_when_the_thread_ends(
    tmp_path, monkeypatch,
):
    from scview.api.v1 import clustering
    from scview.api.v1.clustering import HvgSensitivityRequest, run_hvg_sensitivity
    from scview.core.dataset_manager import DatasetManager

    dm = DatasetManager(str(tmp_path))
    did = _register(dm, _processed())
    entered, release = threading.Event(), threading.Event()
    clustered: list[int] = []
    real_pcs, real_cluster = hs._pcs, hs._cluster

    def held_pcs(*args):
        entered.set()
        release.wait(10)
        return real_pcs(*args)

    monkeypatch.setattr(hs, "_pcs", held_pcs)
    monkeypatch.setattr(hs, "_cluster", lambda X, r: clustered.append(1) or real_cluster(X, r))
    body = HvgSensitivityRequest(settings=[{"flavor": "seurat", "n_top_genes": 200},
                                           {"flavor": "seurat", "n_top_genes": 300}])
    resp = await run_hvg_sensitivity(did, body, dm)
    chunks: list[str] = []

    async def consume():
        async for chunk in resp.body_iterator:
            chunks.append(chunk)

    task = asyncio.create_task(consume())
    assert await asyncio.to_thread(entered.wait, 10)
    task.cancel()  # the client goes away mid-setting
    with pytest.raises(asyncio.CancelledError):
        await task
    assert did in clustering._hvg_running  # the thread still works on the setting
    release.set()
    for _ in range(500):
        if did not in clustering._hvg_running:
            break
        await asyncio.sleep(0.01)
    assert did not in clustering._hvg_running
    assert clustered == []  # stopped after the PCA, before the graph and the clustering
    assert [c.split("\n")[0] for c in chunks] == ["event: start"]


async def test_the_run_leaves_the_loop_free(client, dataset_manager, monkeypatch):
    from tests.test_event_loop import _worst_stall

    did = _register(dataset_manager, _processed())
    real = hs._pcs

    def slow_pcs(*args):
        threading.Event().wait(0.6)
        return real(*args)

    monkeypatch.setattr(hs, "_pcs", slow_pcs)
    resp, stall = await _worst_stall(client.post(
        f"{API}/datasets/{did}/clustering/hvg-sensitivity",
        json={"settings": [{"flavor": "seurat", "n_top_genes": 200}]},
    ))
    assert "event: complete" in resp.text
    assert stall < 0.3
