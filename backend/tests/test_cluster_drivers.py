"""C007: what drives a cluster. Haemoglobin (real red cells vs ambient spill-over),
cycling clusters, and technical scores that track the clustering's leading PCs."""

from __future__ import annotations

import json
import types
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from scview.core import qc_drivers, qc_report
from scview.core.anndata_adaptor import AnnDataAdaptor, per_cell_qc
from scview.core.cluster_qc import cluster_qc, grounding_block
from scview.core.haemoglobin import hb_genes
from scview.core.uns_keys import markers_key

API = "/api/v1"
N_PER = 120
N_GROUPS = 6
HUMAN = ("HBB", "HBA1", "HBA2")
FILLER = [f"G{i}" for i in range(200)]
RIBO = [f"RPL{i}" for i in range(5)] + [f"RPS{i}" for i in range(5)]
EXTRA = ["KLF1", "RHD", "ALAS2", "HBEGF"]  # HBEGF: an "HB" name that isn't haemoglobin


def _blood(hb: dict[str, float] | None = None, *, base: float = 0.005, seed: int = 0) -> ad.AnnData:
    """Raw counts for clusters '0'..'5' (~215 counts per cell). ``hb[g]`` is the
    Poisson mean of EACH haemoglobin gene per cell in cluster g (else ``base``)."""
    rng = np.random.default_rng(seed)
    n = N_PER * N_GROUPS
    labels = np.repeat([str(k) for k in range(N_GROUPS)], N_PER)
    lam = np.array([(hb or {}).get(g, base) for g in labels])
    X = np.hstack([
        rng.poisson(1.0, (n, len(FILLER))),
        rng.poisson(1.5, (n, len(RIBO))),
        rng.poisson(lam[:, None], (n, len(HUMAN))),
        rng.poisson(0.2, (n, len(EXTRA))),
    ]).astype("float32")
    a = ad.AnnData(X)
    a.var_names = FILLER + RIBO + list(HUMAN) + EXTRA
    a.obs_names = [f"C{i}" for i in range(n)]
    a.obs["leiden"] = pd.Categorical(labels, categories=[str(k) for k in range(N_GROUPS)])
    a.uns["scview_active_clustering"] = "leiden"
    return a


def _markers(a: ad.AnnData, top: dict[str, list[str]], column: str = "leiden") -> ad.AnnData:
    """Store ranked markers for ``column``: ``top[g]`` first, then filler genes."""
    groups = list(a.obs[column].cat.categories)
    cols = []
    for g in groups:
        lead = top.get(g, [])
        cols.append(np.array(lead + [x for x in FILLER if x not in lead][: 30 - len(lead)], dtype=object))
    a.uns[markers_key(column)] = {"names": np.rec.fromarrays(cols, names=groups)}
    return a


def _rows(report):
    return {c.group: c for c in report.clusters}


def _normalised(a: ad.AnnData) -> ad.AnnData:
    a.layers["counts"] = a.X.copy()
    a.X = np.log1p(a.X / a.X.sum(1, keepdims=True) * 1e4).astype("float32")
    return a


# ---------------------------------------------------------------------------
# Haemoglobin genes and the per-cell share
# ---------------------------------------------------------------------------


def test_haemoglobin_genes_follow_the_species():
    names = ["HBB", "HBA1", "HBEGF", "HBS1L", "HBP1", "CD3E"]
    assert hb_genes(names, "human") == ["HBA1", "HBB"]  # not HBEGF / HBS1L / HBP1
    assert hb_genes(["Hba-a1", "Hbb-bs", "Hbb-b1", "Hbegf"], "mouse") == ["Hba-a1", "Hbb-bs", "Hbb-b1"]
    assert hb_genes(["HBB", "Hbb-bs"], "unknown") == ["HBB", "Hbb-bs"]
    assert hb_genes(["Hbb-bs"], "human") == []


def test_per_cell_haemoglobin_share_from_counts_or_obs():
    a = _blood({"5": 50.0})
    cells = per_cell_qc(a, ribo=True)
    X = np.asarray(a.X)
    want = X[:, [a.var_names.get_loc(g) for g in HUMAN]].sum(1) / X.sum(1) * 100
    assert np.allclose(cells["pct_hb"], want)
    a.obs["pct_counts_hb"] = 7.0  # a QC step already wrote it: used as is
    assert np.allclose(per_cell_qc(a, ribo=True)["pct_hb"], 7.0)
    # Log-normalised values and no counts anywhere: no meaningful share.
    b = _blood()
    b.X = np.log1p(b.X).astype("float32")
    assert per_cell_qc(b, ribo=True)["pct_hb"] is None


# ---------------------------------------------------------------------------
# Erythroid vs ambient
# ---------------------------------------------------------------------------


def test_erythroid_cluster_is_noted_never_called_contamination():
    a = _markers(_blood({"5": 100.0}), {"5": ["HBB", "HBA1", "HBA2", "ALAS2", "G1", "KLF1"]})
    report = cluster_qc(a)
    rows = _rows(report)
    ery = rows["5"]
    assert ery.erythroid and ery.flags == [] and ery.mean_pct_hb > 40
    assert ery.erythroid_markers == ["HBB", "HBA1", "HBA2", "ALAS2", "KLF1"]
    assert "not ambient contamination" in ery.notes[0] and "red-cell genes (HBB" in ery.notes[0]
    assert not report.flagged and report.ambient_hb == ""
    assert "pct_hb" in report.metrics and report.typical["mean_pct_hb"] < 0.05  # non-erythroid
    lines, cites = grounding_block(report)
    text = "\n".join(lines)
    assert "[result:qc:leiden:5]" in text and "Erythroid" in text
    assert "an erythroid cluster's haemoglobin is its own" in text
    assert ("result:qc:leiden:5", "5: erythroid") in cites


def test_haemoglobin_without_erythroid_markers_is_flagged_as_ambient():
    a = _markers(_blood({"5": 2.5}), {})  # ~3% Hb, markers are filler genes
    rows = _rows(cluster_qc(a))
    assert rows["5"].flags == ["hb_ambient"] and not rows["5"].erythroid
    assert "ambient RNA" in rows["5"].reasons[0] and rows["5"].mean_pct_hb > 1
    for g in "01234":
        assert rows[g].flags == [], rows[g].reasons


def test_haemoglobin_genes_leading_the_markers_do_not_make_a_cluster_erythroid():
    # Ambient Hb gathered in one cluster makes HBB/HBA its markers too (Kang's Mk
    # with 3% Hb added, markers recomputed): only other red-cell genes count.
    a = _markers(_blood({"5": 2.5}), {"5": ["HBB", "HBA1", "HBA2", "G7"]})
    row = _rows(cluster_qc(a))["5"]
    assert row.flags == ["hb_ambient"] and not row.erythroid
    assert row.erythroid_markers == ["HBB", "HBA1", "HBA2"]
    assert "no red-cell genes besides haemoglobin" in row.reasons[0]


def test_erythroid_progenitors_are_known_by_their_markers_not_their_level():
    # The same haemoglobin, but KLF1 and RHD lead its markers (paul15's 2Ery-4Ery).
    a = _markers(_blood({"5": 2.5}), {"5": ["KLF1", "RHD", "G0"]})
    row = _rows(cluster_qc(a))["5"]
    assert row.erythroid and row.flags == [] and row.erythroid_markers == ["KLF1", "RHD"]


def test_without_markers_elevated_haemoglobin_is_not_judged():
    report = cluster_qc(_blood({"5": 2.5}))
    assert not report.flagged
    assert "no marker genes to tell erythroid cells from ambient RNA" in report.note
    # A red-cell level is erythroid even without markers.
    row = _rows(cluster_qc(_blood({"5": 100.0})))["5"]
    assert row.erythroid and row.flags == [] and "red-cell level" in row.notes[0]


def test_haemoglobin_across_all_clusters_is_reported_as_ambient():
    a = _markers(_blood({"5": 100.0}, base=1.0), {"5": ["HBB", "HBA1", "HBA2", "ALAS2", "RHD"]})
    report = cluster_qc(a)
    assert not report.flagged  # spread evenly: no one cluster stands out
    assert "spread across the non-erythroid clusters" in report.ambient_hb
    assert _rows(report)["5"].erythroid
    lines, _ = grounding_block(report)
    assert any(ln.startswith("- Ambient haemoglobin:") and "[result:qc:leiden]" in ln for ln in lines)

    from scview.core.assistant import build_insight

    ins = build_insight(types.SimpleNamespace(adata=_normalised(a)))
    assert "spread across" in ins.insight and "ambient" in ins.question


def test_ambient_cluster_feeds_the_insight():
    from scview.core.assistant import build_insight

    a = _normalised(_markers(_blood({"5": 2.5}), {}))
    ins = build_insight(types.SimpleNamespace(adata=a))
    assert "Cluster 5 (leiden) stands out" in ins.insight and "Haemoglobin RNA" in ins.insight


# ---------------------------------------------------------------------------
# Cell cycle
# ---------------------------------------------------------------------------


def _scores(a: ad.AnnData, cycling: dict[str, float], *, seed: int = 1, names=("S_score", "G2M_score")):
    """S/G2M scores near 0 (noise) except a share of each cluster's cells at 0.3."""
    rng = np.random.default_rng(seed)
    n = a.n_obs
    s = rng.normal(-0.02, 0.03, n)
    g = rng.normal(-0.02, 0.03, n)
    labels = a.obs["leiden"].astype(str).to_numpy()
    for grp, share in cycling.items():
        idx = np.flatnonzero(labels == grp)
        hit = idx[: int(round(share * idx.size))]
        s[hit[::2]] = 0.3
        g[hit[1::2]] = 0.3
    a.obs[names[0]], a.obs[names[1]] = s, g
    # scanpy's phase rule on these scores: about half the noise cells get S or G2M.
    a.obs["phase"] = np.where((s < 0) & (g < 0), "G1", np.where(s > g, "S", "G2M"))
    return a


def test_cluster_of_clearly_cycling_cells_is_flagged():
    a = _scores(_blood(), {"3": 0.8, **{g: 0.1 for g in "01245"}})
    report = cluster_qc(a)
    rows = _rows(report)
    assert rows["3"].flags == ["cycling"] and rows["3"].cycling_fraction == pytest.approx(0.8)
    assert "regress out the S and G2M scores" in rows["3"].reasons[0]
    assert all(rows[g].flags == [] for g in "01245")
    assert "cycling_fraction" in report.metrics


def test_phase_calls_resting_on_noise_never_flag_a_cluster():
    # Cluster 4: every cell called G2M, but its scores sit inside the noise band.
    a = _scores(_blood(), {g: 0.1 for g in "01235"})
    labels = a.obs["leiden"].astype(str).to_numpy()
    a.obs.loc[labels == "4", "G2M_score"] = 0.05
    a.obs.loc[labels == "4", "phase"] = "G2M"
    row = _rows(cluster_qc(a))["4"]
    assert row.flags == [] and row.cycling_fraction == 0.0
    # A phase column alone can't be checked against the noise: not used.
    b = _blood()
    b.obs["phase"] = "S"
    assert "cycling_fraction" not in cluster_qc(b).metrics


def test_seurat_score_names_and_every_cluster_dividing():
    # Progenitor-like: every cluster ~70% cycling; only the ones well above flag.
    a = _scores(_blood(), {"0": 0.7, "1": 0.7, "2": 0.85, "3": 0.98, "4": 0.7, "5": 0.65},
                names=("S.Score", "G2M.Score"))
    rows = _rows(cluster_qc(a))
    assert rows["3"].flags == ["cycling"]
    assert rows["2"].flags == []  # 85% vs a typical 70%: not far enough above


# ---------------------------------------------------------------------------
# Technical drivers of the leading PCs
# ---------------------------------------------------------------------------


def _clusters(n_per: int = 200, k: int = 4) -> np.ndarray:
    return np.repeat([f"c{i}" for i in range(k)], n_per)


def test_a_score_moving_along_a_pc_inside_clusters_is_an_axis():
    rng = np.random.default_rng(0)
    labels = _clusters()
    depth = rng.normal(8, 0.5, labels.size)
    X = rng.normal(0, 0.3, (labels.size, 12))
    X[:, 0] = (depth - 8) * 4 + rng.normal(0, 0.3, labels.size)
    out = qc_drivers.check("X_pca", X.astype("float32"), labels, {"log_counts": depth})
    (d,) = out.drivers
    assert d.pc == 1 and d.r > 0.9 and d.inside_clusters and d.r_within > 0.9
    assert "inside clusters too" in d.reason and "Pearson-residual" in d.reason
    assert out.n_pcs == 10 and out.checked == ["Log total counts"] and 0.5 < d.pc_share < 1
    # Unlabelled cells (NaN in the clustering) are left out, not a cluster of their own.
    holes = labels.astype(object)
    holes[::7] = np.nan
    (d2,) = qc_drivers.check("X_pca", X, holes, {"log_counts": depth}).drivers
    assert d2.inside_clusters and abs(d2.r - d.r) < 0.02


def test_a_score_that_only_differs_between_clusters_is_said_to_be_so():
    rng = np.random.default_rng(1)
    labels = _clusters()
    offset = pd.Series(labels).map({"c0": -3.0, "c1": -1.0, "c2": 1.0, "c3": 3.0}).to_numpy()
    X = rng.normal(0, 1.0, (labels.size, 12))
    X[:, 1] = offset + rng.normal(0, 0.5, labels.size)
    ribo = 20 - 3 * offset + rng.normal(0, 1.5, labels.size)  # its own noise
    (d,) = qc_drivers.check("X_pca_harmony", X, labels, {"pct_ribo": ribo}).drivers
    assert d.pc == 2 and d.r < -0.5 and not d.inside_clusters and abs(d.r_within) < 0.2
    assert "mostly between clusters" in d.reason and "A concern only if" in d.reason


def test_weak_constant_and_masked_scores():
    rng = np.random.default_rng(2)
    labels = _clusters()
    X = rng.normal(0, 1.0, (labels.size, 12))
    out = qc_drivers.check("X_pca", X, labels, {
        "log_counts": rng.normal(8, 0.5, labels.size),  # unrelated
        "pct_mt": np.full(labels.size, 3.0),  # constant: not even checked
    })
    assert out.drivers == [] and out.checked == ["Log total counts"]
    # Red cells forming their own PC is biology: haemoglobin is scored without them.
    ery = labels == "c3"
    X[ery, 0] += 6
    hb = np.where(ery, 4.0, 0.0) + rng.normal(0, 0.05, labels.size)
    assert qc_drivers.check("X_pca", X, labels, {"pct_hb": hb}).drivers
    assert not qc_drivers.check("X_pca", X, labels, {"pct_hb": hb}, {"pct_hb": ~ery}).drivers
    assert qc_drivers.check(None, None, labels, {"pct_hb": hb}) == qc_drivers.DriverCheck()


def test_large_data_is_subsampled_with_the_same_answer(monkeypatch):
    rng = np.random.default_rng(3)
    labels = _clusters(500)
    depth = rng.normal(8, 0.5, labels.size)
    X = rng.normal(0, 0.3, (labels.size, 5))
    X[:, 2] = depth + rng.normal(0, 0.3, labels.size)
    full = qc_drivers.check("X_pca", X, labels, {"log_counts": depth}).drivers[0]
    monkeypatch.setattr(qc_drivers, "MAX_CELLS", 400)
    sub = qc_drivers.check("X_pca", X, labels, {"log_counts": depth}).drivers[0]
    assert sub.pc == full.pc == 3 and abs(sub.r - full.r) < 0.05


def _with_pca(a: ad.AnnData, *, depth_axis: bool) -> ad.AnnData:
    rng = np.random.default_rng(4)
    X = rng.normal(0, 1.0, (a.n_obs, 15))
    if depth_axis:
        logc = np.log1p(np.asarray(a.X).sum(1))
        X[:, 0] = (logc - logc.mean()) / logc.std() * 3 + rng.normal(0, 0.2, a.n_obs)
    a.obsm["X_pca"] = X.astype("float32")
    a.uns["neighbors"] = {"params": {"use_rep": "X_pca", "n_neighbors": 15}}
    return a


def test_cluster_qc_reports_drivers_in_the_clustering_space():
    report = cluster_qc(_with_pca(_blood(), depth_axis=True))
    assert report.drivers_space == "X_pca" and report.drivers_n_pcs == 10
    d = report.drivers[0]
    assert d.score == "log_counts" and d.pc == 1 and d.inside_clusters
    assert {"Log total counts", "Ribosomal %", "Haemoglobin %"} <= set(report.drivers_checked)
    lines, _ = grounding_block(report)
    assert any(ln.startswith("- Technical driver (first 10 PCs of X_pca): Log total counts")
               and ln.endswith("[result:qc:leiden]") for ln in lines)

    quiet = cluster_qc(_with_pca(_blood(), depth_axis=False))
    assert quiet.drivers == []
    text = "\n".join(grounding_block(quiet)[0])
    assert "none of log total counts, ribosomal %, haemoglobin % correlates strongly" in text

    bare = cluster_qc(_blood())  # no PCA: nothing to check against
    assert bare.drivers == [] and bare.drivers_n_pcs == 0
    assert "Technical" not in "\n".join(grounding_block(bare)[0])


# ---------------------------------------------------------------------------
# QC report, endpoint, routing
# ---------------------------------------------------------------------------


def _facts(a: ad.AnnData, tmp_path: Path) -> qc_report.QCFacts:
    p = tmp_path / "d.h5ad"
    a.write_h5ad(p)
    return qc_report.build_facts(AnnDataAdaptor(str(p)))


def test_qc_report_names_ambient_haemoglobin_cycling_and_depth(tmp_path: Path):
    a = _markers(_blood({"5": 2.5, "4": 100.0}), {"4": ["HBB", "HBA1", "HBA2"]})
    a = _with_pca(_scores(a, {"3": 0.9, **{g: 0.1 for g in "0124"}}), depth_axis=True)
    f = _facts(_normalised(a), tmp_path)
    quality = {s.ref: s.detail for s in f.quality if s.ref.startswith("result:qc:leiden:")}
    assert "not ambient contamination" in quality["result:qc:leiden:4"]
    assert "ambient RNA" in quality["result:qc:leiden:5"]
    concerns = [s for s in f.concerns if s.ref.startswith("result:qc:leiden")]
    assert any(s.ref == "result:qc:leiden:5" and "haemoglobin without erythroid markers" in s.detail
               for s in concerns)
    assert not any(s.ref in ("result:qc:leiden:3", "result:qc:leiden:4") for s in concerns)
    assert any("Log total counts is an axis" in s.detail for s in concerns)
    steps = " ".join(s.detail for s in f.next_steps)
    assert "Colour the Unified View by phase" in steps and "Pearson residuals" in steps
    assert "haemoglobin gene for cluster 5" in steps
    assert "tighten the QC filters" not in steps  # nothing damaged or doublet-rich


async def test_cluster_qc_endpoint_carries_the_new_fields(client, dataset_manager):
    did = "drv1"
    d = Path(dataset_manager.data_dir) / "uploads" / did
    d.mkdir(parents=True, exist_ok=True)
    (d / "metadata.json").write_text(json.dumps({"id": did, "name": did, "status": "ready"}))
    a = _with_pca(_markers(_blood({"5": 100.0}), {"5": list(HUMAN)}), depth_axis=True)
    a.write_h5ad(d / "data.h5ad")
    r = await client.get(f"{API}/datasets/{did}/cluster-qc")
    assert r.status_code == 200, r.text
    body = r.json()
    ery = next(c for c in body["clusters"] if c["group"] == "5")
    assert ery["erythroid"] is True and ery["mean_pct_hb"] > 40 and ery["notes"]
    assert body["drivers"][0]["score"] == "log_counts" and body["drivers_space"] == "X_pca"
    assert "hb_ambient" in body["flag_help"] and "cycling" in body["flag_help"]
    assert body["typical"]["mean_pct_hb"] < 0.05


def test_driver_questions_route_to_the_qc_facet():
    from scview.core.rag.router import heuristic_intent

    for q in ("Is there ambient haemoglobin RNA in my clusters?",
              "Does the cell cycle drive any cluster?", "Is sequencing depth driving my clusters?"):
        intent = heuristic_intent(q)
        assert "data" in intent.sources and "qc" in intent.data_facets, q
