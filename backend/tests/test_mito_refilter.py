"""The two-pass % mito flow (core/mito_refilter.py): a fresh matrix without cell
types is filtered by one ceiling, clustered, then re-filtered from the original
upload with per-cluster bounds, the cells pass 1 removed placed by expression.

Fixture: three populations with their own marker genes; X and Y sit at ~3 %
mito, Z (the high-baseline type) at ~12 %, so a fixed 10 % ceiling takes most of
Z, and per-type bounds (Z: ~12 % + 3 MADs) keep it.
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

from scview.core import mito_refilter, provenance, qc_filter, replay_script
from scview.core.pipeline import PipelineParams, run_pipeline

API = "/api/v1"
N_GENES = 150
N_MT = 10
POPS = {"X": (300, 0, 0.76), "Y": (300, 40, 0.76), "Z": (200, 80, 3.34)}  # cells, block start, MT rate
PASS1 = ["qc_metrics", "filtering", "normalization", "log_transform", "highly_variable_genes", "scaling",
         "pca", "neighbors", "clustering"]
P1 = PipelineParams(qc_threshold_mode="fixed", min_genes=20, max_pct_mt=10.0, n_top_genes=100, n_comps=15)


def _counts(seed: int = 0) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    parts, pops = [], []
    for name, (n, lo, mt) in POPS.items():
        X = rng.poisson(0.3, size=(n, N_GENES)).astype("float32")
        X[:, lo:lo + 40] += rng.poisson(5.0, size=(n, 40)).astype("float32")
        parts.append(np.hstack([X, rng.poisson(mt, size=(n, N_MT)).astype("float32")]))
        pops += [name] * n
    a = ad.AnnData(np.vstack(parts))
    a.var_names = [f"G{i}" for i in range(N_GENES)] + [f"MT-{i}" for i in range(N_MT)]
    a.obs_names = [f"C{i}" for i in range(a.n_obs)]
    a.uns["truth"] = np.array(pops, dtype=object)  # not an obs column: the app must not see it
    return a


_TRUTH: list[pd.Series] = []


def _truth(a: ad.AnnData) -> pd.Series:
    """Each cell's population (kept out of obs: the app must not see it)."""
    if not _TRUTH:
        c = _counts()
        _TRUTH.append(pd.Series(c.uns["truth"], index=c.obs_names))
    return _TRUTH[0].loc[a.obs_names]


def _register(dm, did: str = "tp1", pass1: bool = True) -> tuple[ad.AnnData, ad.AnnData | None]:
    """Upload = the counts; derived = pass 1 run on them (as the app stores them)."""
    root = Path(dm.data_dir)
    up = root / "uploads" / did
    up.mkdir(parents=True)
    (up / "metadata.json").write_text(json.dumps({"id": did, "name": did, "status": "ready"}))
    orig = _counts()
    del orig.uns["truth"]
    provenance.init_source(orig, origin="uploaded", original_filename="raw.h5ad", fmt="h5ad")
    orig.write_h5ad(up / "raw.h5ad")
    if not pass1:
        return orig, None
    cur, res = run_pipeline(orig.copy(), PASS1, P1)
    assert res.errors == {}
    (root / "derived" / did).mkdir(parents=True)
    cur.write_h5ad(root / "derived" / did / "raw.h5ad")
    return orig, cur


@pytest.fixture()
def pass1(dataset_manager):
    orig, cur = _register(dataset_manager)
    return dataset_manager, orig, cur


def _majority(cur: ad.AnnData, column: str) -> dict[str, str]:
    truth = _truth(cur)
    return pd.crosstab(cur.obs[column].astype(str).to_numpy(), truth.to_numpy()).idxmax(axis=1).to_dict()


def test_pass_1_takes_most_of_the_high_baseline_population(pass1):
    _, orig, cur = pass1
    lost = _truth(orig)[~orig.obs_names.isin(cur.obs_names)].value_counts()
    assert lost.get("Z", 0) > 140 and lost.get("X", 0) < 10 and lost.get("Y", 0) < 10


def test_removed_cells_are_placed_by_expression(pass1):
    _, orig, cur = pass1
    col = cur.uns["scview_active_clustering"]
    gone = ~orig.obs_names.isin(cur.obs_names)
    labels, r, info = mito_refilter.assign_by_expression(cur, col, orig.X[gone], orig.var_names)
    maj = _majority(cur, col)
    truth = _truth(orig)[gone].to_numpy()
    acc = np.mean([maj.get(lab) == t for lab, t in zip(labels, truth)])
    assert acc > 0.95, acc
    assert info["method"] == "nearest_centroid" and info["n_assigned"] == gone.sum()
    assert info["genes"] >= mito_refilter.MIN_ASSIGN_GENES and info["median_r"] > 0.5
    # MT- genes never place a cell: rows of MT counts alone give no assignment signal
    only_mt = np.zeros((1, orig.n_vars), dtype="float32")
    only_mt[0, -N_MT:] = 50
    lab, score, _ = mito_refilter.assign_by_expression(cur, col, only_mt, orig.var_names)
    assert lab[0] == "" and np.isnan(score[0])


async def test_the_preview_brings_back_the_high_baseline_cells(client, pass1):
    dm, orig, cur = pass1
    body = (await client.get(f"{API}/datasets/tp1/qc/mito-refilter")).json()
    assert body["column"] == cur.uns["scview_active_clustering"]
    assert body["type_column"] == mito_refilter.TYPE_COLUMN
    assert body["n_original"] == orig.n_obs and body["n_current"] == cur.n_obs
    assert body["removed_before"] == orig.n_obs - cur.n_obs
    assert body["typed"]["expression"] == body["removed_before"] and body["typed"]["none"] == 0
    maj = _majority(cur, body["column"])
    z = [t for t in body["types"] if maj.get(t["name"]) == "Z"]
    assert z and all(t["rule"] == "own" and 13 < t["bound"] < 25 for t in z)
    assert sum(t["brought_back"] for t in z) > 120
    assert body["brought_back"] == sum(t["brought_back"] for t in body["types"])
    assert body["n_after"] == orig.n_obs - body["removed_after"]
    assert body["can"].startswith(f"Brings back {body['brought_back']:,} of the {body['removed_before']:,}")
    assert "stay removed" in body["cannot"] and "not carried over" in body["cannot"]
    assert body["steps"] == PASS1 and body["reset_to_counts"] is False


async def test_the_run_keeps_what_the_preview_said_and_records_it(client, pass1, dataset_manager):
    dm, orig, cur = pass1
    preview = (await client.get(f"{API}/datasets/tp1/qc/mito-refilter")).json()
    r = await client.post(f"{API}/datasets/tp1/qc/mito-refilter-stream", json={})
    assert r.status_code == 200
    assert "event: refilter" in r.text and '"stage": "types"' in r.text and "event: complete" in r.text
    assert "event: error" not in r.text and "step_error" not in r.text
    N = (await dataset_manager.get_or_load_dataset("tp1")).adata
    assert N.n_obs == preview["n_after"]
    assert mito_refilter.TYPE_COLUMN in N.obs.columns
    hist = [h["step"] for h in provenance.read_provenance(N)["history"]]
    assert hist == PASS1 + ["mito_refilter"] + PASS1
    entry = provenance.read_provenance(N)["history"][len(PASS1)]
    assert entry["params"]["typed"]["expression"] == preview["removed_before"]
    filt = provenance.read_provenance(N)["history"][len(PASS1) + 2]["params"]
    assert filt["qc_mito_mode"] == "per_type" and filt["qc_mito_type_column"] == mito_refilter.TYPE_COLUMN
    assert filt["qc_threshold_mode"] == "fixed" and filt["max_pct_mt"] == 10.0  # pass 1's own settings
    rec = N.uns[qc_filter.RECORD_KEY]
    assert rec["mito_mode"] == "per_type" and rec["n_before"] == orig.n_obs
    per = {t["name"]: t["removed_after"] for t in preview["types"]}
    after = {t["name"]: t["removed"] for t in filt["qc_mito_types"]["types"]}
    assert {k: v for k, v in after.items() if v} == {k: v for k, v in per.items() if v}
    # the card now reads the per-type rule it was filtered with
    card = (await client.get(f"{API}/datasets/tp1/qc/mito-by-type")).json()
    assert card["column"] == mito_refilter.TYPE_COLUMN and card["filter_mito_mode"] == "per_type"
    assert card["n_flagged"] == 0 and card["summary"].startswith("Filtered with per-type bounds")


async def test_a_global_refilter_from_the_original_gives_pass_1_again(client, pass1, dataset_manager):
    dm, orig, cur = pass1
    r = await client.post(f"{API}/datasets/tp1/qc/mito-refilter-stream", json={"mito_mode": "global"})
    assert "event: complete" in r.text and "event: error" not in r.text
    N = (await dataset_manager.get_or_load_dataset("tp1")).adata
    assert list(N.obs_names) == list(cur.obs_names)


async def test_the_card_adds_back_the_removed_cells_placed_by_expression(client, pass1):
    dm, orig, cur = pass1
    body = (await client.get(f"{API}/datasets/tp1/qc/mito-by-type")).json()
    col = cur.uns["scview_active_clustering"]
    assert body["column"] == col and body["includes_removed"] == orig.n_obs - cur.n_obs
    assert body["assigned_removed"] == body["includes_removed"]
    maj = _majority(cur, col)
    z = [t for t in body["types"] if maj.get(t["group"]) == "Z"]
    assert z and all(t["flagged"] for t in z)
    assert "expression is closest to" in body["note"]
    assert " of cluster " in body["summary"]  # a clustering's numbers are named as clusters
    again = (await client.get(f"{API}/datasets/tp1/qc/mito-by-type")).json()  # served from the cache
    assert again["types"] == body["types"]


async def test_what_it_refuses(client, dataset_manager, tmp_path):
    _register(dataset_manager, "noorig")
    (Path(dataset_manager.data_dir) / "uploads" / "noorig" / "raw.h5ad").unlink()  # derived only
    r = await client.get(f"{API}/datasets/noorig/qc/mito-refilter")
    assert r.status_code == 409 and "original upload" in r.json()["detail"]
    r = await client.post(f"{API}/datasets/noorig/qc/mito-refilter-stream", json={})
    assert r.status_code == 409
    r = await client.get(f"{API}/datasets/noorig/qc/mito-refilter", params={"column": "nope"})
    assert r.status_code == 404
    r = await client.get(f"{API}/datasets/noorig/qc/mito-refilter", params={"mito_mode": "per_cluster"})
    assert r.status_code == 400
    # no MT- genes: % mito can't be measured, so there is nothing to re-filter by
    root = Path(dataset_manager.data_dir) / "uploads" / "nomt"
    root.mkdir(parents=True)
    (root / "metadata.json").write_text(json.dumps({"id": "nomt", "name": "nomt", "status": "ready"}))
    a = _counts()
    del a.uns["truth"]
    a = a[:, :N_GENES].copy()
    a.obs["leiden"] = pd.Categorical(np.arange(a.n_obs) % 3)
    a.write_h5ad(root / "raw.h5ad")
    r = await client.get(f"{API}/datasets/nomt/qc/mito-refilter")
    assert r.status_code == 400 and "can't be measured" in r.json()["detail"]
    assert (await client.get(f"{API}/datasets/ghost/qc/mito-refilter")).status_code == 404


def test_the_replay_script_runs_both_passes_and_keeps_the_same_cells(tmp_path, dataset_manager):
    """Run on the input, the script replays pass 1, starts again from the input with
    the cells typed as the app typed them, and keeps the cells the app kept."""
    from scview.core.anndata_adaptor import AnnDataAdaptor

    orig, cur = _register(dataset_manager, "rp1")
    adaptor = AnnDataAdaptor(str(Path(dataset_manager.data_dir) / "derived" / "rp1" / "raw.h5ad"))
    prep = mito_refilter.prepare(dataset_manager, "rp1", adaptor, None)
    N, res = run_pipeline(prep.adata, prep.steps, prep.params)
    assert res.errors == {}
    script = replay_script.build_script(N, dataset_id="rp1", dataset_name="rp1")
    assert "refilter_types(adata, pass1, " in script and "MITO_TYPES = " in script
    statuses = [d["status"] for d in replay_script.describe(N)]
    assert statuses[len(PASS1)] == "exact"
    src, path, dst = tmp_path / "input.h5ad", tmp_path / "replay.py", tmp_path / "replayed.h5ad"
    orig.write_h5ad(src)
    path.write_text(script)
    proc = subprocess.run([sys.executable, str(path), str(src), str(dst)], capture_output=True, text=True,
                          timeout=600)
    assert proc.returncode == 0, proc.stderr[-3000:]
    out = ad.read_h5ad(dst)
    assert list(out.obs_names) == list(N.obs_names)
    assert (out.obs[mito_refilter.TYPE_COLUMN].astype(str).to_numpy()
            == N.obs[mito_refilter.TYPE_COLUMN].astype(str).to_numpy()).all()


def test_the_replay_helper_types_cells_as_the_app_does():
    orig = _counts()
    cur, _ = run_pipeline(orig.copy(), PASS1, P1)
    col = cur.uns["scview_active_clustering"]
    import scipy.sparse as sp

    ns = {"np": np, "pd": pd, "sp": sp}
    for name in ("looks_like_counts", "refilter_types"):
        exec(replay_script._HELPERS[name], ns)
    typed = ns["refilter_types"](orig, cur, col)
    gone = ~orig.obs_names.isin(cur.obs_names)
    labels, _, _ = mito_refilter.assign_by_expression(cur, col, orig.X[gone], orig.var_names)
    assert list(np.asarray(typed.astype(str))[gone]) == list(labels)
    kept = np.asarray(typed.astype(str))[~gone]
    assert list(kept) == list(cur.obs[col].astype(str))


def test_an_upload_with_log_values_in_x_is_reset_as_pass_1_was(tmp_path, dataset_manager):
    """Sparse, counts in layers['counts'] and log values in X (as ovary_raw): pass 1
    was a Reprocess from Scratch, so the re-filter resets the original too, and the
    replay script does the same."""
    import scanpy as sc
    import scipy.sparse as sp

    from scview.core.anndata_adaptor import AnnDataAdaptor

    root = Path(dataset_manager.data_dir)
    up = root / "uploads" / "lg1"
    up.mkdir(parents=True)
    (up / "metadata.json").write_text(json.dumps({"id": "lg1", "name": "lg1", "status": "ready"}))
    orig = _counts()
    del orig.uns["truth"]
    orig.X = sp.csr_matrix(orig.X)
    orig.layers["counts"] = orig.X.copy()
    sc.pp.normalize_total(orig, target_sum=1e4)
    sc.pp.log1p(orig)
    orig.write_h5ad(up / "raw.h5ad")
    cur, res = run_pipeline(orig.copy(), ["reset_to_counts", *PASS1], P1)
    assert res.errors == {}
    (root / "derived" / "lg1").mkdir(parents=True)
    cur.write_h5ad(root / "derived" / "lg1" / "raw.h5ad")
    adaptor = AnnDataAdaptor(str(root / "derived" / "lg1" / "raw.h5ad"))
    preview = mito_refilter.preview(dataset_manager, "lg1", adaptor, None)
    assert preview["reset_to_counts"] is True and preview["brought_back"] > 120
    prep = mito_refilter.prepare(dataset_manager, "lg1", adaptor, None)
    assert sp.issparse(prep.adata.X) and float(prep.adata.X.max()) == float(orig.layers["counts"].max())
    new, res = run_pipeline(prep.adata, prep.steps, prep.params)
    assert res.errors == {} and new.n_obs == preview["n_after"]
    assert mito_refilter.TYPE_COLUMN in new.obs.columns  # kept through the steps
    script = replay_script.build_script(new, dataset_id="lg1", dataset_name="lg1")
    src, path, dst = tmp_path / "input.h5ad", tmp_path / "replay.py", tmp_path / "replayed.h5ad"
    orig.write_h5ad(src)
    path.write_text(script)
    proc = subprocess.run([sys.executable, str(path), str(src), str(dst)], capture_output=True, text=True,
                          timeout=600)
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert list(ad.read_h5ad(dst).obs_names) == list(new.obs_names)


async def test_the_floor_never_removes_a_cell_pass_1_kept(client, pass1, dataset_manager):
    """With ``floor`` the re-filter is never stricter than the global rule, which here
    is pass 1's own (fixed 10 %): X and Y (~3 %, own bounds below 10 %) lose no cell
    pass 1 kept, and Z (~12 %) still comes back. Without it they lose their tails."""
    dm, orig, cur = pass1
    url = f"{API}/datasets/tp1/qc/mito-refilter"
    off = (await client.get(url)).json()
    on = (await client.get(url, params={"floor": "true"})).json()
    assert off["mito_type_floor"] is False and on["mito_type_floor"] is True
    assert off["newly_removed"] > 0 and on["newly_removed"] == 0
    assert all(t["newly_removed"] == 0 for t in on["types"])
    assert on["brought_back"] == off["brought_back"] > 120  # Z's own bounds are above 10 %
    assert "whichever is higher" in on["can"]
    r = await client.post(f"{API}/datasets/tp1/qc/mito-refilter-stream", json={"floor": True})
    assert "event: complete" in r.text and "event: error" not in r.text
    N = (await dataset_manager.get_or_load_dataset("tp1")).adata
    assert N.n_obs == on["n_after"] and cur.obs_names.isin(N.obs_names).all()
    hist = provenance.read_provenance(N)["history"]
    assert hist[len(PASS1)]["params"]["mito_type_floor"] is True
    filt = hist[len(PASS1) + 2]["params"]
    assert filt["qc_mito_type_floor"] is True and filt["qc_mito_types"]["floor"] is True
    card = (await client.get(f"{API}/datasets/tp1/qc/mito-by-type")).json()
    assert card["filter_type_floor"] is True and "never stricter" in card["summary"]
