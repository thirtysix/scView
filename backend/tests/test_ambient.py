"""Ambient RNA (C066): soup profile, contamination fraction, marker flag, correction step.

Expected values come from the simulation's own truth (cells given a known share of soup
UMIs), from hand-worked examples, or from R SoupX 1.6.2 run on the same inputs in a
throwaway container (rocker/r2u:jammy, r-cran-soupx 1.6.2, 2026-10-24), never from this
module's output. On the 10x pbmc_1k_v3 raw + filtered matrices
(cf.10xgenomics.com/samples/cell-exp/3.0.0/pbmc_1k_v3/, 16 Leiden clusters) the soup,
the 1,060 quickMarkers genes (cluster and tf-idf), rho (0.01, range 0.01-0.028) and
adjustCounts(clusters = FALSE) at rho 0.01 / 0.05 / 0.2 (to 5e-11) were identical; with
5% and 20% soup added to those cells (Multinomial draws from the real soup) R gave 0.053
and 0.202, scView the same (``SOUPX_R`` below records those and the simulation's).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from scview.core import ambient, cluster_qc, preflight, provenance, replay_script, seq_metrics
from scview.core.pipeline import PipelineParams, run_pipeline

API = "/api/v1"


# ---------------------------------------------------------------------------
# A simulation with known contamination
# ---------------------------------------------------------------------------


def simulate(rho: float, seed: int = 0, n_per: int = 100, k: int = 8, n_genes: int = 2000,
             n_mark: int = 30, umi: int = 1000):
    """``k`` cell types with their own 30 marker genes (near absent elsewhere); the soup is
    their mix (type 0 twice as much); each cell holds Multinomial(round(n x (1 - rho)), its
    type) + Multinomial(round(n x rho), soup) UMIs, so rho of every cell is soup."""
    rng = np.random.default_rng(seed)
    base = rng.gamma(0.6, 1.0, size=n_genes)
    lift = rng.uniform(20, 60, size=k * n_mark)
    profiles = []
    for t in range(k):
        p = base.copy()
        p[:k * n_mark] *= 0.005
        blk = slice(t * n_mark, (t + 1) * n_mark)
        p[blk] = base[blk] * lift[blk]
        profiles.append(p / p.sum())
    weights = np.array([2.0] + [1.0] * (k - 1))
    soup = (weights[:, None] * np.array(profiles)).sum(0)
    soup /= soup.sum()
    rows, types = [], []
    for t in range(k):
        n = rng.lognormal(np.log(umi), 0.3, size=n_per).astype(int)
        own = rng.multinomial(np.rint(n * (1 - rho)).astype(int), profiles[t])
        amb = rng.multinomial(np.rint(n * rho).astype(int), soup)
        rows.append(own + amb)
        types += [t] * n_per
    X = sp.csr_matrix(np.vstack(rows).astype(np.float32))
    return X, np.array(types), soup


def _adata(rho: float, seed: int = 0, soup_umis: int = 200_000) -> ad.AnnData:
    """The simulation as a dataset: counts in X, the types as the active clustering, and
    an empty-droplet soup (``soup_umis`` drawn from the soup) stored as a raw matrix gives it."""
    X, types, soup = simulate(rho, seed)
    a = ad.AnnData(X=X)
    a.var_names = [f"G{i}" for i in range(X.shape[1])]
    a.var["gene_ids"] = [f"ENSG{i:05d}" for i in range(X.shape[1])]
    a.obs_names = [f"C{i}" for i in range(X.shape[0])]
    a.obs["leiden"] = pd.Categorical([str(t) for t in types])
    a.uns["scview_active_clustering"] = "leiden"
    counts = np.random.default_rng(seed + 100).multinomial(soup_umis, soup)
    nz = np.flatnonzero(counts)
    ambient.store_soups(a, {"s1": {
        "genes": np.asarray(a.var_names)[nz], "ids": a.var["gene_ids"].to_numpy()[nz],
        "counts": counts[nz].astype(np.int64), "n_droplets": 5000, "n_umis": soup_umis,
        "umi_max": 100, "n_features": X.shape[1]}})
    a.uns[seq_metrics.UNS_KEY] = json.dumps({"version": 1, "samples": {"s1": {
        "sample": "s1", "column": None, "files": {"metrics": None, "raw": ["raw_feature_bc_matrix.h5"]},
        "knee": None, "metrics": None, "notes": []}}})
    return a


# R SoupX 1.6.2 autoEstCont (defaults) on simulate(rho, seed) written out as Matrix Market,
# the soup profile set to the true soup (setSoupProfile), clusters = the types: rho and
# its half-maximum range (rhoEst, rhoFWHM).
SOUPX_R = {
    (0.05, 0): (0.050, 0.031, 0.074),
    (0.05, 1): (0.049, 0.029, 0.073),
    (0.2, 0): (0.191, 0.138, 0.253),
    (0.2, 1): (0.194, 0.135, 0.263),
}


@pytest.mark.parametrize("rho,seed", list(SOUPX_R))
def test_rho_is_recovered_from_a_known_contamination(rho, seed):
    X, types, soup = simulate(rho, seed)
    fit = ambient.estimate_rho(X, types, soup)
    assert fit.status == "ok" and fit.rho is not None
    assert abs(fit.rho - rho) <= 0.2 * rho  # within 20% of the truth
    assert fit.low <= rho <= fit.high
    want, low, high = SOUPX_R[(rho, seed)]
    assert fit.rho == pytest.approx(want, abs=1e-9)
    assert (fit.low, fit.high) == pytest.approx((low, high), abs=1e-9)


def test_rho_needs_clusters_and_markers():
    X, types, soup = simulate(0.05)
    one = ambient.estimate_rho(X, np.zeros(X.shape[0], dtype=int), soup)
    assert one.rho is None and one.status == "few_clusters"
    flat = np.full(X.shape[1], 1.0 / X.shape[1])  # a soup with no high genes: no markers pass
    none = ambient.estimate_rho(X, types, flat)
    assert none.rho is None and none.status == "no_markers"


def test_bh_matches_r_p_adjust():
    # p.adjust(c(0.01, 0.04, 0.03, 0.2, 0.5), "BH") in R
    got = ambient.bh(np.array([0.01, 0.04, 0.03, 0.2, 0.5]))
    assert got == pytest.approx([0.05, 0.0666666667, 0.0666666667, 0.25, 0.5])


def test_gamma_prior_is_soupx_s():
    k, theta = ambient._gamma_k_theta(0.05, 0.10)
    # v2 = 4: k = 1 + 4^-2 / 2 x (1 + sqrt(17)); theta = 0.05 / (k - 1)
    assert k == pytest.approx(1 + (1 / 16) / 2 * (1 + 17 ** 0.5))
    assert theta == pytest.approx(0.05 / (k - 1))


# ---------------------------------------------------------------------------
# Subtraction
# ---------------------------------------------------------------------------


def test_alloc_water_fills_by_hand():
    # Remove 6 from counts [1, 10, 10] with weights [0.5, 0.25, 0.25]: the first can give
    # only 1, the remaining 5 split evenly between the others.
    got = ambient._alloc_row(6.0, np.array([1.0, 10.0, 10.0]), np.array([0.5, 0.25, 0.25]))
    assert got == pytest.approx([1.0, 2.5, 2.5])
    # Everything fits in proportion
    assert ambient._alloc_row(4.0, np.array([5.0, 5.0]), np.array([0.5, 0.5])) == pytest.approx([2, 2])
    # More than the cell has: all of it
    assert ambient._alloc_row(50.0, np.array([3.0, 4.0]), np.array([0.9, 0.1])) == pytest.approx([3, 4])


def test_correction_removes_rho_of_each_cell_never_below_zero():
    X, _, soup = simulate(0.2)
    got = ambient.correct_counts(X, [(np.arange(X.shape[0]), 0.2, soup)], round_to_int=False)
    removed = np.asarray(X.sum(1)).ravel() - np.asarray(got.sum(1)).ravel()
    assert removed == pytest.approx(0.2 * np.asarray(X.sum(1)).ravel(), rel=1e-6)  # float32 sums
    assert got.min() >= 0
    rounded = ambient.correct_counts(X, [(np.arange(X.shape[0]), 0.2, soup)])
    assert np.all(np.mod(rounded.data, 1) == 0) and (rounded - X).max() <= 0
    # per cell, the rounded total is the exact total rounded
    exact = np.asarray(got.sum(1)).ravel()
    assert np.abs(np.asarray(rounded.sum(1)).ravel() - exact).max() <= 0.5 + 1e-9
    # rows in no group are left alone; the soup genes lose the most
    part = ambient.correct_counts(X, [(np.arange(10), 0.2, soup)])
    assert (part[10:] != X[10:]).nnz == 0
    loss = np.asarray((X - rounded).sum(0)).ravel()
    assert np.corrcoef(loss, soup)[0, 1] > 0.9


def test_correction_takes_the_soup_out_of_the_other_cell_types():
    X, types, soup = simulate(0.2)
    fixed = ambient.correct_counts(X, [(np.arange(X.shape[0]), 0.2, soup)])
    others = types != 0
    # Type 0's markers (genes 0-29, the soup's largest part) in the other types' cells:
    # nearly all soup before; the subtraction takes most of it away.
    before = X[others][:, :30].sum()
    after = fixed[others][:, :30].sum()
    clean, _, _ = simulate(0.0)
    assert clean[others][:, :30].sum() < 0.05 * before  # what is left in clean cells
    assert after < 0.35 * before
    # and a type's own markers keep most of their counts
    assert fixed[~others][:, :30].sum() > 0.75 * X[~others][:, :30].sum()


# ---------------------------------------------------------------------------
# Soup from raw matrices
# ---------------------------------------------------------------------------


def _raw(tmp_path: Path):
    from tests.test_seq_metrics import _bc, _raw_counts, _write_h5, _write_mex

    dense = _raw_counts()
    bcs = [_bc(i) for i in range(dense.shape[1])]
    return dense, bcs, _write_h5(tmp_path / "raw.h5", dense, bcs), _write_mex(tmp_path, "s1_raw_", dense, bcs)


def test_soup_from_h5_and_mex_by_hand(tmp_path, monkeypatch):
    dense, _, h5, mex = _raw(tmp_path)
    gex = dense[:30]
    tot = gex.sum(axis=0)
    keep = (tot > 0) & (tot < 100)
    want = gex[:, keep].sum(axis=1)
    got = ambient.soup_from_h5(h5, 100, nnz_chunk=37)  # many blocks
    full = pd.Series(got["counts"], index=got["genes"]).reindex([f"GENE{i}" for i in range(30)]).fillna(0)
    assert full.to_numpy().tolist() == want.tolist()
    assert got["n_droplets"] == int(keep.sum()) and got["n_umis"] == int(tot[keep].sum())
    assert "AB0_TotalSeqB" not in set(got["genes"])  # antibodies are not soup genes
    m = ambient.soup_from_mex(*mex, umi_max=100, rows=50)
    assert list(m["genes"]) == list(got["genes"]) and m["counts"].tolist() == got["counts"].tolist()
    assert m["n_droplets"] == got["n_droplets"]


def test_soup_fractions_match_by_name_or_id():
    soup = {"genes": np.array(["A", "B", "C"]), "ids": np.array(["e1", "e2", "e3"]),
            "counts": np.array([6, 3, 1])}
    frac, covered = ambient.soup_fractions(soup, pd.Index(["B", "A", "Z"]))
    assert frac == pytest.approx([1 / 3, 2 / 3, 0]) and covered == pytest.approx(0.9)
    frac, covered = ambient.soup_fractions(soup, pd.Index(["x", "y", "z"]), np.array(["e3", "e2", "e1"]))
    assert frac == pytest.approx([0.1, 0.3, 0.6]) and covered == pytest.approx(1.0)


def test_attaching_a_raw_matrix_stores_its_soup(tmp_path):
    dense, bcs, h5, _ = _raw(tmp_path)
    a = ad.AnnData(X=sp.csr_matrix(np.ones((150, 30), dtype=np.float32)))
    a.obs_names = bcs[:150]
    a.var_names = [f"GENE{i}" for i in range(30)]
    rec = seq_metrics.attach_files(a, [h5], "s1", None)
    assert rec["soup"]["n_droplets"] > 100 and "_soup" not in rec
    soups = ambient.read_soups(a)
    assert set(soups) == {"s1"} and soups["s1"]["counts"].sum() > 0
    assert "_soup" not in json.loads(a.uns[seq_metrics.UNS_KEY])["samples"]["s1"]
    seq_metrics.remove(a, "s1")
    assert ambient.UNS_SOUP not in a.uns


# ---------------------------------------------------------------------------
# Diagnostic, cluster QC, endpoint
# ---------------------------------------------------------------------------


def test_diagnose_reports_rho_soup_genes_and_flags():
    a = _adata(0.2)
    body = ambient.diagnose(a)
    assert body["available"] and not body["approximate"] and body["column"] == "leiden"
    (s,) = body["samples"]
    assert s["source"] == "empty_droplets" and s["n_cells"] == a.n_obs
    assert abs(s["rho"] - 0.2) <= 0.04 and s["rho_status"] == "ok"
    # the cell types' markers lead the soup (genes 0-239)
    assert all(int(g["gene"][1:]) < 240 for g in s["top_genes"][:10])
    # flags need markers: none stored yet
    assert body["markers"]["groups"] == {}


def _with_markers(a: ad.AnnData) -> ad.AnnData:
    import scanpy as sc

    b = a.copy()
    sc.pp.normalize_total(b, target_sum=1e4)
    sc.pp.log1p(b)
    sc.tl.rank_genes_groups(b, "leiden", method="wilcoxon", pts=True)
    a.uns["rank_genes_groups"] = b.uns["rank_genes_groups"]
    a.raw = b
    return a


def test_marker_flag_marks_soup_genes_seen_elsewhere_at_soup_level():
    a = _with_markers(_adata(0.2))
    body = ambient.diagnose(a)
    groups = body["markers"]["groups"]
    assert not body["markers"]["approximate"]
    # Type 0's markers are the soup's top genes and sit in every other cell at rho x soup.
    flagged0 = {r["gene"] for r in groups.get("0", [])}
    assert flagged0 and all(int(g[1:]) < 30 for g in flagged0)
    assert all(ambient.AMBIENT_RATIO_MIN <= r["ratio"] <= ambient.AMBIENT_RATIO_MAX for r in groups["0"])
    # cluster QC lists them per cluster
    rep = cluster_qc.cluster_qc(a, "leiden")
    row0 = next(c for c in rep.clusters if c.group == "0")
    assert set(row0.ambient_markers) == flagged0 and rep.ambient_markers_help == ambient.HELP["flag"]
    # Clean data: no ambient spill, so nothing outside the clusters to explain.
    clean = _with_markers(_adata(0.0))
    flags = ambient.diagnose(clean)["markers"]["groups"]
    assert sum(len(v) for v in flags.values()) < len(flagged0)


def test_without_a_raw_matrix_the_profile_is_approximate():
    a = _with_markers(_adata(0.2))
    del a.uns[ambient.UNS_SOUP]
    body = ambient.diagnose(a)
    assert body["approximate"] and not body["available"]
    (s,) = body["samples"]
    assert s["source"] == "cells" and s["rho"] is None and s["rho_status"] == "no_soup"
    assert body["markers"]["approximate"]
    for rows in body["markers"]["groups"].values():
        assert all(r["ratio"] is None and r["soup_rank"] <= ambient.SOUP_TOP_GENES_APPROX for r in rows)


def test_the_cached_answer_follows_edits_made_in_place():
    a = _with_markers(_adata(0.2))
    soup = ambient.read_soups(a)["s1"]
    del a.uns[ambient.UNS_SOUP]
    first = ambient.diagnose_cached(a)
    assert not first["available"] and ambient.diagnose_cached(a) is first  # cached
    ambient.store_soups(a, {"s1": soup})  # a raw matrix attached to the open dataset
    second = ambient.diagnose_cached(a)
    assert second["available"]
    a.obs["leiden"] = pd.Categorical(np.where(a.obs["leiden"] == "0", "1", a.obs["leiden"]))  # relabelled
    third = ambient.diagnose_cached(a)
    assert third is not second and "0" not in third["markers"]["groups"]


def test_no_clustering_says_so():
    a = _adata(0.05)
    del a.obs["leiden"]
    del a.uns["scview_active_clustering"]
    (s,) = ambient.diagnose(a)["samples"]
    assert s["rho"] is None and s["rho_status"] == "no_clusters"


def _register(dm, a: ad.AnnData, did: str = "amb") -> str:
    d = Path(dm.data_dir) / "uploads" / did
    d.mkdir(parents=True)
    (d / "metadata.json").write_text(json.dumps({"id": did, "name": "amb", "status": "ready"}))
    a.write_h5ad(d / "data.h5ad")
    return did


async def test_ambient_endpoint(client, dataset_manager):
    did = _register(dataset_manager, _with_markers(_adata(0.2)))
    r = await client.get(f"{API}/datasets/{did}/qc/ambient")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["samples"][0]["rho"] == pytest.approx(0.2, abs=0.04)
    assert body["help"]["rho"] and body["url"] == ambient.SOUPX_URL
    assert body["markers"]["n_flagged"] > 0
    r = await client.get(f"{API}/datasets/{did}/qc/ambient", params={"column": "nope"})
    assert r.status_code == 404
    assert (await client.get(f"{API}/datasets/ghost/qc/ambient")).status_code == 404
    r = await client.get(f"{API}/datasets/{did}/cluster-qc")
    assert r.status_code == 200 and any(c["ambient_markers"] for c in r.json()["clusters"])


async def test_ambient_endpoint_refuses_a_large_dataset(client, tmp_data_dir):
    from tests.large_fixture import counts_adata, register

    register(tmp_data_dir, "bigA", counts_adata())
    r = await client.get(f"{API}/datasets/bigA/qc/ambient")
    assert r.status_code == 422 and "large datasets" in r.json()["detail"]


# ---------------------------------------------------------------------------
# The pipeline step, reset, replay, large mode, preflight
# ---------------------------------------------------------------------------


def test_step_corrects_counts_keeps_originals_and_reset_restores():
    a = _adata(0.2)
    original = a.X.copy()
    out, res = run_pipeline(a, ["ambient_correction", "normalization", "log_transform"], PipelineParams())
    assert res.errors == {}, res.errors
    assert (out.layers[ambient.ORIGINAL_LAYER] != original).nnz == 0
    corrected = out.layers["counts"]
    share = 1 - corrected.sum() / original.sum()
    assert share == pytest.approx(0.2, abs=0.04)
    (h,) = [h for h in provenance.read_provenance(out)["history"] if h["step"] == "ambient_correction"]
    p = h["params"]
    assert p["ambient_rho"] is None and p["samples"][0]["rho_source"] == "estimated"
    assert p["samples"][0]["raw_files"] == ["raw_feature_bc_matrix.h5"]
    assert p["clusters"] == "leiden" and p["quick_clusters"] is False
    # a second correction (X still corrected counts) starts from the originals: never twice
    once, _ = run_pipeline(_adata(0.2), ["ambient_correction"], PipelineParams(ambient_rho=0.2))
    again, res = run_pipeline(once, ["ambient_correction"], PipelineParams(ambient_rho=0.1))
    assert res.errors == {}
    assert 1 - again.X.sum() / original.sum() == pytest.approx(0.1, abs=0.002)
    back, _ = run_pipeline(out, ["reset_to_counts"], PipelineParams())
    assert ambient.ORIGINAL_LAYER not in back.layers
    assert abs(back.X - original).max() == 0 and abs(back.layers["counts"] - original).max() == 0


def test_step_clusters_the_cells_itself_without_a_clustering():
    a = _adata(0.2)
    del a.obs["leiden"]
    del a.uns["scview_active_clustering"]
    out, res = run_pipeline(a, ["ambient_correction"], PipelineParams())
    assert res.errors == {}, res.errors
    (h,) = [h for h in provenance.read_provenance(out)["history"] if h["step"] == "ambient_correction"]
    assert h["params"]["quick_clusters"] is True
    assert h["params"]["samples"][0]["rho"] == pytest.approx(0.2, abs=0.05)


def test_step_without_a_raw_matrix_fails_and_normalization_is_skipped():
    a = _adata(0.05)
    del a.uns[ambient.UNS_SOUP]
    out, res = run_pipeline(a, ["ambient_correction", "normalization"], PipelineParams())
    assert "raw (unfiltered) matrix" in res.errors["ambient_correction"]
    assert "normalization" in res.steps_skipped
    assert ambient.ORIGINAL_LAYER not in out.layers


def test_step_refuses_processed_x():
    a = _adata(0.05)
    a.X = a.X.multiply(0.5).tocsr()  # not whole counts
    with pytest.raises(ambient.AmbientError, match="raw counts in X"):
        ambient.run_correction(a)
    # nor on a corrected, normalised dataset (the originals there would put counts in X)
    out, _ = run_pipeline(_adata(0.05), ["ambient_correction", "normalization", "log_transform"],
                          PipelineParams(ambient_rho=0.05))
    with pytest.raises(ambient.AmbientError, match="raw counts in X"):
        ambient.run_correction(out)
    # and without a soup that is what it says first
    del out.uns[ambient.UNS_SOUP]
    with pytest.raises(ambient.AmbientError, match="raw \\(unfiltered\\) matrix"):
        ambient.run_correction(out)


def test_assessment_reports_the_step():
    from scview.core.assessor import assess_preprocessing

    a = _adata(0.05)
    st = assess_preprocessing(a).ambient_correction
    assert not st.done and "soup profile is attached" in st.details
    out, _ = run_pipeline(a, ["ambient_correction"], PipelineParams(ambient_rho=0.05))
    st = assess_preprocessing(out).ambient_correction
    assert st.done and "5.0%" in st.details


def _write_raw_h5(path: Path, a: ad.AnnData, soup_umis: int = 100_000, seed: int = 3) -> Path:
    """A raw 10x .h5 for ``a``'s genes: its cells (X) plus empty droplets of 1-99 UMIs drawn
    from the soup stored on ``a``."""
    soup = ambient.read_soups(a)["s1"]
    frac, _ = ambient.soup_fractions(soup, a.var_names)
    rng = np.random.default_rng(seed)
    sizes = rng.integers(1, 100, size=soup_umis // 50)
    empties = np.array([rng.multinomial(s, frac) for s in sizes])
    cells = np.asarray(a.X.todense())
    dense = np.vstack([cells, empties]).T  # features x barcodes
    csc = sp.csc_matrix(dense.astype(np.int64))
    with h5py.File(path, "w") as f:
        m = f.create_group("matrix")
        m.create_dataset("data", data=csc.data.astype(np.int32))
        m.create_dataset("indices", data=csc.indices.astype(np.int64))
        m.create_dataset("indptr", data=csc.indptr.astype(np.int64))
        m.create_dataset("shape", data=np.array(dense.shape, dtype=np.int32))
        m.create_dataset("barcodes", data=np.array([f"B{i}" for i in range(dense.shape[1])], dtype="S"))
        ft = m.create_group("features")
        ft.create_dataset("id", data=np.array(a.var["gene_ids"], dtype="S"))
        ft.create_dataset("name", data=np.array(a.var_names, dtype="S"))
        ft.create_dataset("feature_type", data=np.array(["Gene Expression"] * a.n_vars, dtype="S"))
    return path


def test_replay_script_repeats_the_correction(tmp_path):
    a = _adata(0.2)
    raw = _write_raw_h5(tmp_path / "raw_feature_bc_matrix.h5", a)
    soup = ambient.soup_from_h5(raw)
    ambient.store_soups(a, {"s1": soup})  # the soup the attach would have stored
    inp = a.copy()
    out, res = run_pipeline(a, ["ambient_correction", "normalization", "log_transform"], PipelineParams())
    assert res.errors == {}
    src, script, dst = tmp_path / "input.h5ad", tmp_path / "replay.py", tmp_path / "replayed.h5ad"
    inp.write_h5ad(src)
    text = replay_script.build_script(out, dataset_id="t", dataset_name="ambient")
    assert "AMBIENT_SAMPLES" in text and "def correct_counts(" in text
    script.write_text(text)
    proc = subprocess.run([sys.executable, str(script), str(src), str(dst)], capture_output=True,
                          text=True, timeout=600, cwd=tmp_path)
    assert proc.returncode == 0, proc.stderr[-3000:]
    rep = ad.read_h5ad(dst)
    assert abs(rep.layers["counts"] - out.layers["counts"]).max() == 0
    assert abs(rep.layers[ambient.ORIGINAL_LAYER] - out.layers[ambient.ORIGINAL_LAYER]).max() == 0
    statuses = {d["status"] for d in replay_script.describe(out)}
    assert statuses == {"exact"}


def test_replay_without_raw_files_is_not_replayable():
    a = ad.AnnData(np.zeros((3, 2), dtype="float32"))
    provenance.record_step(a, step="ambient_correction", tool="t",
                           params={"ambient_rho": 0.1, "samples": [{"sample": "s", "rho": 0.1}]})
    assert replay_script.describe(a)[0]["status"] == "not_replayable"
    compile(replay_script.build_script(a), "replay.py", "exec")


def test_reset_replay_restores_uncorrected_counts():
    a = ad.AnnData(X=sp.csr_matrix(np.ones((2, 2), dtype=np.float32)))
    a.layers["counts"] = sp.csr_matrix(np.full((2, 2), 2.0, dtype=np.float32))
    a.layers["counts_uncorrected"] = sp.csr_matrix(np.full((2, 2), 3.0, dtype=np.float32))
    code = "\n".join(replay_script._reset_to_counts({}, replay_script._Ctx(adata=a)).code)
    ns = {"adata": a}
    exec(code, ns)
    assert "counts_uncorrected" not in ns["adata"].layers
    assert ns["adata"].X.toarray().tolist() == [[3.0, 3.0], [3.0, 3.0]]


def test_large_dataset_mode_refuses_the_step():
    from scview.core import large_run

    why = large_run.refused(["filtering", "ambient_correction"])
    assert why and why.startswith("Ambient RNA correction is not available for large datasets")
    assert large_run.refused(["filtering", "normalization"]) is None


async def test_large_run_with_the_step_is_a_422(client, tmp_data_dir):
    from tests.large_fixture import counts_adata, register

    register(tmp_data_dir, "bigB", counts_adata())
    r = await client.post(f"{API}/datasets/bigB/assessment/run",
                          json={"steps": ["ambient_correction", "normalization", "log_transform"],
                                "params": {"chunked": "on"}})
    assert r.status_code == 422, r.text
    assert "Ambient RNA correction" in r.json()["detail"]


def test_preflight_estimates_the_step():
    a = _adata(0.2)
    est = preflight.estimate_run(preflight.header_from_adata(a), steps=["ambient_correction"],
                                 params=PipelineParams())
    step = next(s for s in est.steps if s.step == "ambient_correction")
    assert step.transient_bytes > 0 and step.seconds[1] > step.seconds[0] > 0
    # The originals are kept beside X afterwards: the state holds a second copy.
    assert est.end_bytes > est.start_bytes
    assert preflight.step_peak_for(a, "ambient_correction", PipelineParams()) > 0
    set_rho = preflight.estimate_run(preflight.header_from_adata(a), steps=["ambient_correction"],
                                     params=PipelineParams(ambient_rho=0.1))
    assert set_rho.seconds[1] < est.seconds[1]  # no estimate, no quick clustering
