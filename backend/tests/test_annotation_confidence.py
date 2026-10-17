"""Annotation confidence: split conformal prediction sets per cell (C058)."""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from scview.core import annotation_confidence as conf
from scview.core import provenance


def _blobs(seed: int = 0, n: int = 300) -> ad.AnnData:
    """Four labels in a 10-PC space: A and B far from everything, C and D two
    overlapping halves of one cloud (a classifier tells them apart ~75 % of the
    time). obsm holds the PCA only; X is never read."""
    rng = np.random.default_rng(seed)
    centres = {"A": [8, 0], "B": [0, 8], "C": [0, 0], "D": [1.3, 0]}
    labels = np.repeat(list(centres), n)
    X = rng.normal(size=(labels.size, 10))
    X[:, :2] += np.array([centres[lab] for lab in labels])
    a = ad.AnnData(X=np.zeros((labels.size, 3), dtype=np.float32))
    a.obs_names = [f"c{i}" for i in range(labels.size)]
    a.obsm["X_pca"] = X
    a.obs["cl"] = pd.Categorical(labels, categories=list(centres))
    return a


def _flags(a: ad.AnnData, res: dict) -> np.ndarray:
    return a.obs[res["columns"]["flag"]].astype(str).to_numpy()


@pytest.mark.parametrize("classifier", ["logreg", "knn"])
def test_coverage_holds_on_held_out_cells(classifier):
    a = _blobs()
    res = conf.assess(a, "cl", alpha=0.1, classifier=classifier)
    cov = res["coverage"]
    assert cov["target"] == 0.9
    assert cov["n_held_out"] + cov["n_calibration"] == a.n_obs
    assert cov["within_sampling_error"]
    assert cov["held_out"] >= 0.9 - 2 * cov["sd"]
    assert cov["calibration"] >= 0.9  # by construction
    # Averaged over fresh datasets, within the sampling spread of the mean.
    held = [conf.assess(_blobs(seed), "cl", 0.1, classifier)["coverage"]["held_out"]
            for seed in range(1, 6)]
    assert np.mean(held) >= 0.9 - 2.5 * cov["sd"] / np.sqrt(len(held))


def test_most_shuffled_labels_are_flagged():
    """Plant wrong labels on 10 % of the cells: nearly all of them are rejected,
    mostly as fitting another label; far fewer clean cells are."""
    a = _blobs(n=400)
    rng = np.random.default_rng(7)
    vals = a.obs["cl"].astype(str).to_numpy().copy()
    idx = rng.choice(vals.size, size=vals.size // 10, replace=False)
    new = vals[idx].copy()
    rng.shuffle(new)
    moved = idx[new != vals[idx]]
    vals[idx] = new
    a.obs["cl"] = pd.Categorical(vals)
    # A and B are unmistakable, so a label moved into or out of them is plainly
    # wrong; C and D overlap, so a swap between them may not be.
    truth = np.repeat(list("ABCD"), 400)
    clear = moved[~(np.isin(truth[moved], ["C", "D"]) & np.isin(vals[moved], ["C", "D"]))]

    res = conf.assess(a, "cl", alpha=0.1)
    flag = _flags(a, res)
    rejected = np.isin(flag, conf.REJECTED)
    assert rejected[moved].mean() >= 0.75
    assert rejected[clear].mean() >= 0.95
    assert (flag[clear] == conf.NOT_FIT).mean() >= 0.8  # another label fits them
    clean = np.setdiff1d(np.arange(vals.size), idx)
    assert rejected[clean].mean() <= 0.1
    # A clearly moved cell's competing label is its true one (C or D for either).
    alt = a.obs[res["columns"]["alt"]].astype(str).to_numpy()
    cd = np.isin(alt[clear], ["C", "D"]) & np.isin(truth[clear], ["C", "D"])
    assert ((alt[clear] == truth[clear]) | cd).mean() >= 0.9


def test_overlapping_labels_are_ambiguous_with_each_other():
    """Below the classifier's error rate sets grow: C and D cells become ambiguous
    with each other, A and B stay confident."""
    a = _blobs()
    res = conf.assess(a, "cl", alpha=0.05)
    rows = {r["label"]: r for r in res["labels"]}
    assert rows["A"]["confident"] >= 0.9 and rows["B"]["confident"] >= 0.9
    assert rows["C"]["ambiguous"] >= 0.3 and rows["D"]["ambiguous"] >= 0.3
    assert rows["C"]["confused_with"][0]["label"] == "D"
    assert rows["D"]["confused_with"][0]["label"] == "C"
    pairs = {(p["label"], p["other"]) for p in res["confusions"][:2]}
    assert pairs == {("C", "D"), ("D", "C")}
    # Above it, sets shrink: no one is ambiguous, in-between cells are unsure.
    hi = conf.assess(a, "cl", alpha=0.4)
    assert hi["counts"][conf.AMBIGUOUS] == 0 and hi["counts"][conf.UNSURE] > 0


def test_flag_fit_and_set_size_agree():
    a = _blobs()
    for alpha in (0.05, 0.1, 0.2):
        res = conf.assess(a, "cl", alpha=alpha)
        names = res["columns"]
        flag = _flags(a, res)
        fit = a.obs[names["fit"]].to_numpy()
        size = a.obs[names["set_size"]].to_numpy()
        # The own label is in the set exactly when its conformal p-value is above alpha.
        assert (np.isin(flag, conf.REJECTED) == (fit <= alpha)).all()
        assert ((flag == conf.CONFIDENT) == ((fit > alpha) & (size == 1))).all()
        assert ((flag == conf.AMBIGUOUS) == ((fit > alpha) & (size >= 2))).all()
        assert ((flag == conf.UNSURE) == (size == 0)).all()
        assert ((fit > 0) & (fit <= 1)).all()
        alt = a.obs[names["alt"]]
        assert alt[flag == conf.CONFIDENT].isna().all()
        assert (alt[flag != conf.CONFIDENT].astype(str) != a.obs["cl"][flag != conf.CONFIDENT]
                .astype(str)).all()
        assert sum(res["counts"].values()) == a.n_obs


def test_outputs_registry_and_provenance():
    a = _blobs()
    res = conf.assess(a, "cl")
    names = res["columns"]
    assert names == {"flag": "scview_conf_cl_flag", "fit": "scview_conf_cl_fit",
                     "set_size": "scview_conf_cl_set_size", "alt": "scview_conf_cl_alt"}
    assert list(a.obs[names["flag"]].cat.categories) == conf.FLAGS
    assert a.obs[names["fit"]].dtype == np.float32
    assert res["alpha"] == 0.1 and res["score"] == "lac" and res["classifier"] == "logreg"
    assert res["representation"] == "X_pca" and res["n_pcs"] == 10
    assert res["note"] == conf.NOTE and "not whether the label is right" in res["note"]
    assert res["stale"] is False and res["unknown_column"] is None
    assert [r["label"] for r in res["labels"]] == ["A", "B", "C", "D"]
    step = provenance.read_provenance(a)["history"][-1]
    assert step["step"] == "annotation_confidence"
    assert step["params"]["column"] == "cl" and "LAC" in step["tool"]
    assert json.loads(a.uns[conf.REGISTRY_KEY])["cl"]["alpha"] == 0.1
    assert conf.summary(a, "cl")["coverage"] == res["coverage"]
    assert conf.summary(a, "nope") is None
    # A second check of the column replaces the first's columns, adds none.
    before = list(a.obs.columns)
    again = conf.assess(a, "cl", alpha=0.2)
    assert list(a.obs.columns) == before and again["alpha"] == 0.2
    # Harmony's PCA is preferred, like the neighbour graph.
    a.obsm["X_pca_harmony"] = a.obsm["X_pca"][:, :6]
    assert conf.assess(a, "cl")["representation"] == "X_pca_harmony"


def test_small_labels_and_missing_values_are_not_checked():
    a = _blobs()
    vals = a.obs["cl"].astype(object).to_numpy()
    vals[:5] = "rare"
    vals[5:12] = np.nan
    a.obs["cl"] = pd.Categorical(vals, categories=["A", "B", "C", "D", "rare"])
    res = conf.assess(a, "cl")
    assert res["labels_not_checked"] == [{"label": "rare", "n_cells": 5}]
    assert res["n_checked"] == a.n_obs - 12
    flag = _flags(a, res)
    assert (flag[:12] == conf.NOT_CHECKED).all()
    assert a.obs[res["columns"]["fit"]].iloc[:12].isna().all()
    assert conf.NOT_CHECKED in a.obs[res["columns"]["flag"]].cat.categories


def test_a_large_training_half_is_sampled():
    a = _blobs()
    old = conf.MAX_TRAIN
    conf.MAX_TRAIN = 200
    try:
        res = conf.assess(a, "cl")
    finally:
        conf.MAX_TRAIN = old
    assert res["coverage"]["within_sampling_error"]
    assert res["n_checked"] == a.n_obs  # every cell is still scored


def test_requests_that_cannot_be_served_say_why():
    a = _blobs()
    cols = list(a.obs.columns)
    with pytest.raises(conf.ConfidenceError, match="not found") as e:
        conf.assess(a, "nope")
    assert e.value.status == 404
    with pytest.raises(conf.ConfidenceError, match="Alpha"):
        conf.assess(a, "cl", alpha=0.9)
    with pytest.raises(conf.ConfidenceError, match="classifier"):
        conf.assess(a, "cl", classifier="svm")
    a.obs["one"] = pd.Categorical(["x"] * (a.n_obs - 10) + ["y"] * 10)
    with pytest.raises(conf.ConfidenceError, match="at least two labels"):
        conf.assess(a, "one")
    a.obs["many"] = [f"g{i % 150}" for i in range(a.n_obs)]
    with pytest.raises(conf.ConfidenceError, match="150 different values"):
        conf.assess(a, "many")
    a.obs["scview_conf_x_flag"] = a.obs["cl"]
    with pytest.raises(conf.ConfidenceError, match="output of the confidence check"):
        conf.assess(a, "scview_conf_x_flag")
    no_pca = _blobs()
    del no_pca.obsm["X_pca"]
    with pytest.raises(conf.ConfidenceError, match="needs a PCA"):
        conf.assess(no_pca, "cl")
    assert list(a.obs.columns) == [*cols, "one", "many", "scview_conf_x_flag"]
    assert conf.REGISTRY_KEY not in a.uns


def test_never_overwrites_a_column_it_did_not_write():
    a = _blobs()
    a.obs["scview_conf_cl_fit"] = 1.0  # e.g. from an imported file
    res = conf.assess(a, "cl")
    assert res["columns"]["flag"] == "scview_conf_cl_2_flag"
    assert (a.obs["scview_conf_cl_fit"] == 1.0).all()
    # Two columns whose names clean to the same stem keep separate outputs.
    a.obs["c l"] = a.obs["cl"]
    other = conf.assess(a, "c l")
    assert other["columns"]["flag"] not in res["columns"].values()
    assert conf.assess(a, "cl")["columns"] == res["columns"]  # its own names again


def test_unknown_copy_marks_rejected_cells():
    a = _blobs()
    with pytest.raises(conf.ConfidenceError, match="first") as e:
        conf.make_unknown_column(a, "cl")
    assert e.value.status == 409
    res = conf.assess(a, "cl", alpha=0.2)
    out = conf.make_unknown_column(a, "cl")
    name = out["unknown_column"]
    assert name == "scview_conf_cl_unknown"
    flag = _flags(a, res)
    rejected = np.isin(flag, conf.REJECTED)
    copy = a.obs[name].astype(str).to_numpy()
    assert (copy[rejected] == conf.UNKNOWN).all()
    assert (copy[~rejected] == a.obs["cl"].astype(str).to_numpy()[~rejected]).all()
    assert out["n_unknown"] == int(rejected.sum())
    assert list(a.obs[name].cat.categories) == ["A", "B", "C", "D", conf.UNKNOWN]
    assert conf.UNKNOWN not in a.obs["cl"].cat.categories  # the labels are untouched
    assert conf.summary(a, "cl")["unknown_column"] == name
    assert provenance.read_provenance(a)["history"][-1]["step"] == "annotation_confidence_unknown"
    # A new check rebuilds the copy from its own flags.
    res2 = conf.assess(a, "cl", alpha=0.05)
    rejected2 = np.isin(_flags(a, res2), conf.REJECTED)
    assert ((a.obs[name].astype(str) == conf.UNKNOWN).to_numpy() == rejected2).all()
    assert res2["unknown_column"] == name


def test_a_changed_label_column_makes_the_check_stale():
    a = _blobs()
    conf.assess(a, "cl")
    a.obs["cl"] = a.obs["cl"].cat.rename_categories({"A": "A cells"})
    assert conf.summary(a, "cl")["stale"] is True
    with pytest.raises(conf.ConfidenceError, match="changed since") as e:
        conf.make_unknown_column(a, "cl")
    assert e.value.status == 409
    assert conf.assess(a, "cl")["stale"] is False


# ---------------------------------------------------------------------------
# Scoring by a reference model's probabilities (CellTypist)
# ---------------------------------------------------------------------------


def _reference(a: ad.AnnData, disagree: np.ndarray) -> pd.DataFrame:
    """CellTypist-like one-vs-rest probabilities over the labels plus a class the
    column lacks: high for the cell's own label, except at ``disagree``, where the
    model calls B. Rows don't sum to 1, and some are near 0 throughout."""
    rng = np.random.default_rng(3)
    truth = a.obs["cl"].astype(str).to_numpy()
    classes = ["A", "B", "C", "D", "Plasma cells"]
    P = rng.uniform(0.001, 0.05, size=(a.n_obs, len(classes)))
    for j, c in enumerate(classes[:4]):
        P[truth == c, j] = rng.uniform(0.6, 0.99, size=(truth == c).sum())
    P[disagree] = 0.01
    P[disagree, 1] = 0.95
    P[::97] *= 1e-3  # the model is unsure of everything here
    return pd.DataFrame(P, index=a.obs_names, columns=classes)


def test_reference_probabilities_score_the_cells():
    a = _blobs()
    planted = np.flatnonzero(a.obs["cl"] == "A")[:30]
    conf.store_reference_probabilities(a, "cl", _reference(a, planted), "Immune_All_Low.pkl")
    assert conf.reference_columns(a) == {"cl": "Immune_All_Low.pkl"}
    del a.obsm["X_pca"]  # no PCA needed: nothing is fitted to the dataset
    res = conf.assess(a, "cl", alpha=0.1, classifier="celltypist")
    assert res["representation"] == "CellTypist Immune_All_Low" and res["n_pcs"] == 0
    assert res["note"] == conf.REFERENCE_NOTE and "CellTypist" in res["note"]
    assert res["coverage"]["within_sampling_error"]
    flag = _flags(a, res)
    assert (flag[planted] == conf.NOT_FIT).all()
    assert (a.obs[res["columns"]["alt"]].astype(str).to_numpy()[planted] == "B").all()
    others = np.setdiff1d(np.flatnonzero(a.obs["cl"] == "A"), planted)
    assert (flag[others] == conf.CONFIDENT).mean() >= 0.9
    assert "CellTypist" in provenance.read_provenance(a)["history"][-1]["tool"]


def test_reference_probabilities_that_cannot_be_used_say_why():
    a = _blobs()
    with pytest.raises(conf.ConfidenceError, match="only a column annotated with CellTypist"):
        conf.assess(a, "cl", classifier="celltypist")
    conf.store_reference_probabilities(a, "cl", _reference(a, np.array([], int)), "m.pkl")
    key = json.loads(a.uns[conf.REFERENCE_KEY])["cl"]["obsm"]
    assert a.obsm[key].shape == (a.n_obs, 4) and a.obsm[key].dtype == np.float32
    # Renamed labels no longer match the probabilities.
    a.obs["cl"] = a.obs["cl"].cat.rename_categories({"D": "D cells"})
    assert conf.reference_columns(a) == {}
    with pytest.raises(conf.ConfidenceError, match="changed since CellTypist annotated it"):
        conf.assess(a, "cl", classifier="celltypist")
    assert conf.REGISTRY_KEY not in a.uns
    conf.drop_reference_probabilities(a, "cl")
    assert key not in a.obsm and json.loads(a.uns[conf.REFERENCE_KEY]) == {}
    conf.drop_reference_probabilities(a, "cl")  # nothing left to drop: no error


def test_celltypist_annotation_keeps_its_probabilities_and_others_drop_them(monkeypatch):
    """The pipeline step stores CellTypist's probabilities of the target's labels;
    annotating the same column another way forgets them."""
    import sys
    import types

    from scview.core import pipeline as P

    a = ad.AnnData(X=np.random.RandomState(0).poisson(1, (40, 5)).astype("float32"))
    a.obs_names = [f"c{i}" for i in range(40)]
    a.obs["cluster"] = pd.Categorical(["a", "b"] * 20)
    classes = ["T cells", "B cells", "NK cells"]

    def annotate(inp, model, majority_voting, over_clustering=None):
        votes = np.where(inp.obs["cluster"].astype(str) == "a", "T cells", "B cells")
        labels = pd.DataFrame({"predicted_labels": votes, "majority_voting": votes}, index=inp.obs_names)
        prob = pd.DataFrame(np.random.RandomState(1).uniform(size=(inp.n_obs, 3)),
                            index=inp.obs_names, columns=classes)
        return types.SimpleNamespace(predicted_labels=labels, probability_matrix=prob)

    fake = types.SimpleNamespace(
        annotate=annotate, models=types.SimpleNamespace(download_models=lambda **_: None))
    monkeypatch.setitem(sys.modules, "celltypist", fake)
    P._annotate_celltypist(a, P.PipelineParams(), "cluster", "cell_type")
    assert conf.reference_columns(a) == {"cell_type": "Immune_All_Low.pkl"}
    entry = json.loads(a.uns[conf.REFERENCE_KEY])["cell_type"]
    assert entry["labels"] == sorted(["T cells", "B cells"])  # the column's own labels only
    assert a.obsm[entry["obsm"]].shape == (40, 2)

    def marker_score(adata, params, groupby, target):
        adata.obs[target] = pd.Categorical(["T cells"] * adata.n_obs)

    monkeypatch.setattr(P, "_annotate_marker_score", marker_score)
    monkeypatch.setattr(P, "_markers_for_column", lambda adata, col: None)
    P._run_cell_type_annotation(a, P.PipelineParams(
        annotation_method="marker_score", annotation_groupby="cluster", annotation_target="cell_type"))
    assert conf.reference_columns(a) == {} and entry["obsm"] not in a.obsm


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


def _save(dataset_manager, a: ad.AnnData) -> None:
    d = Path(dataset_manager.data_dir) / "uploads" / "ds"
    d.mkdir(parents=True, exist_ok=True)
    (d / "metadata.json").write_text(json.dumps({"id": "ds", "name": "ds", "status": "ready"}))
    a.write_h5ad(d / "data.h5ad")


async def test_endpoints_run_save_and_report(client, dataset_manager):
    _save(dataset_manager, _blobs())
    info = (await client.get("/api/v1/datasets/ds/annotation/confidence?column=cl")).json()
    assert info["available"] is True and info["representation"] == "X_pca"
    assert info["result"] is None and info["checked_columns"] == []
    assert info["default_alpha"] == 0.1 and "not whether the label is right" in info["note"]
    assert info["reference_columns"] == {} and "celltypist" in info["classifiers"]

    r = await client.post("/api/v1/datasets/ds/annotation/confidence",
                          json={"column": "cl", "alpha": 0.05})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["alpha"] == 0.05 and body["coverage"]["within_sampling_error"]
    saved = ad.read_h5ad(dataset_manager.derived_h5ad_path("ds", "data.h5ad"))
    for name in body["columns"].values():
        assert name in saved.obs.columns
    assert json.loads(saved.uns[conf.REGISTRY_KEY])["cl"]["alpha"] == 0.05
    obs_names = [c["name"] for c in (await client.get("/api/v1/datasets/ds")).json()["obs_columns"]]
    assert body["columns"]["flag"] in obs_names

    info = (await client.get("/api/v1/datasets/ds/annotation/confidence?column=cl")).json()
    assert info["checked_columns"] == ["cl"]
    assert info["result"]["coverage"] == body["coverage"]

    u = await client.post("/api/v1/datasets/ds/annotation/confidence/unknown", json={"column": "cl"})
    assert u.status_code == 200, u.text
    saved = ad.read_h5ad(dataset_manager.derived_h5ad_path("ds", "data.h5ad"))
    assert (saved.obs[u.json()["unknown_column"]] == conf.UNKNOWN).sum() == u.json()["n_unknown"]


async def test_endpoint_errors(client, dataset_manager):
    a = _blobs()
    a.obs["one"] = pd.Categorical(["x"] * a.n_obs)
    _save(dataset_manager, a)
    url = "/api/v1/datasets/ds/annotation/confidence"
    assert (await client.post(url, json={"column": "nope"})).status_code == 404
    assert (await client.post(url, json={"column": "one"})).status_code == 400
    assert (await client.post(url, json={"column": "cl", "alpha": 0})).status_code == 400
    assert (await client.post(f"{url}/unknown", json={"column": "cl"})).status_code == 409
    assert (await client.get("/api/v1/datasets/missing/annotation/confidence")).status_code == 404
    b = _blobs()
    del b.obsm["X_pca"]
    _save(dataset_manager, b)
    dataset_manager.evict("ds")
    info = (await client.get(url)).json()
    assert info["available"] is False and "PCA" in info["reason"]
