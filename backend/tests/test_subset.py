"""A new dataset from some of a dataset's cells (core/subset.py, POST /datasets/{id}/subset)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from scview.core import provenance, replay_script, subset
from scview.core.cell_filter import mask_to_base64

API = "/api/v1"
TYPES = ["T", "B", "Mono"]
N_PER = 60


def _parent(seed: int = 0) -> ad.AnnData:
    """A processed dataset: counts in layers['counts'], log values in X and .raw, scView's
    clusters, a UMAP and a PCA, CITE-seq counts and a WNN result."""
    import scanpy as sc

    rng = np.random.default_rng(seed)
    n_genes = 80
    rows, labels = [], []
    for i, t in enumerate(TYPES):
        mu = np.full(n_genes, 1.0)
        mu[i * 10:(i + 1) * 10] = 9.0
        rows.append(rng.poisson(mu, size=(N_PER, n_genes)))
        labels += [t] * N_PER
    n = len(labels)
    counts = sp.csr_matrix(np.vstack(rows).astype(np.float32))
    obs = pd.DataFrame({
        "cell_type": pd.Categorical(labels, categories=[*TYPES, "NK"]),
        "scview_leiden": pd.Categorical([str(TYPES.index(t)) for t in labels]),
        "scview_mito_type": pd.Categorical(labels),
        "donor": [f"d{i % 4}" for i in range(n)],
        "score": rng.normal(size=n),
    }, index=[f"cell{i}" for i in range(n)])
    var = pd.DataFrame(index=[f"G{i}" for i in range(n_genes)])
    a = ad.AnnData(counts.copy(), obs=obs, var=var)
    a.layers["counts"] = counts
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    a.raw = a
    a.obsm["X_umap"] = rng.normal(size=(n, 2)).astype(np.float32)
    a.obsm["X_pca"] = rng.normal(size=(n, 10)).astype(np.float32)
    a.obsm["protein_counts"] = pd.DataFrame(rng.poisson(5, size=(n, 3)).astype(np.float32),
                                            index=a.obs_names, columns=["CD3", "CD19", "CD14"])
    a.obsm["protein_pca"] = rng.normal(size=(n, 4))
    a.uns["scview_protein"] = json.dumps({"source": "test"})
    a.uns["rank_genes_groups__scview_leiden"] = {"x": 1}
    provenance.init_source(a, origin="upload", original_filename="parent.h5ad", fmt="h5ad")
    provenance.record_step(a, step="clustering", tool="test")
    return a


def _add(data_dir: Path, did: str, name: str, adata: ad.AnnData, subdir: str = "ingested") -> Path:
    up = data_dir / "uploads" / did
    up.mkdir(parents=True, exist_ok=True)
    (up / "metadata.json").write_text(json.dumps({"id": did, "name": name, "status": "ready"}))
    d = data_dir / subdir / did
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{name}.h5ad"
    adata.write_h5ad(path)
    return path


def _new(data_dir: Path, did: str) -> ad.AnnData:
    (path,) = (data_dir / "ingested" / did).glob("*.h5ad")
    return ad.read_h5ad(path)


@pytest.mark.asyncio
async def test_groups_subset_holds_the_cells_counts_and_labels(client, tmp_data_dir):
    parent = _parent()
    path = _add(tmp_data_dir, "p1", "pbmc", parent)
    before = path.stat().st_mtime_ns
    r = await client.post(f"{API}/datasets/p1/subset",
                          json={"groups": {"column": "cell_type", "values": ["Mono", "B"]}})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["n_cells"] == 2 * N_PER and d["n_parent_cells"] == 3 * N_PER and d["name"] == "pbmc - Mono, B"
    assert d["renamed"] == {"scview_leiden": "parent_leiden"} and "run_pipeline" in d["next"]
    assert any("PCA" in x for x in d["not_carried"]) and any("WNN" in x for x in d["not_carried"])
    assert d["counts_source"] == "layers['counts']"

    sub = _new(tmp_data_dir, d["dataset_id"])
    keep = parent.obs["cell_type"].isin(["Mono", "B"]).to_numpy()
    assert list(sub.obs_names) == list(parent.obs_names[keep])
    # counts, not the log values in X
    np.testing.assert_array_equal(sub.X.toarray(), parent.layers["counts"][keep].toarray())
    assert sub.raw is None and not sub.layers
    assert list(sub.obs["cell_type"].cat.categories) == ["B", "Mono"]  # unused dropped
    assert "parent_leiden" in sub.obs and "scview_leiden" not in sub.obs
    assert "scview_mito_type" in sub.obs  # an input of per-type filtering stays as it is
    assert set(sub.obsm) == {"X_umap_parent", "protein_counts"}  # CLR is made again on open
    np.testing.assert_allclose(sub.obsm["X_umap_parent"], parent.obsm["X_umap"][keep])
    assert list(sub.obsm["protein_counts"].index) == list(sub.obs_names)
    assert "rank_genes_groups__scview_leiden" not in sub.uns and "scview_protein" in sub.uns
    assert list(sub.uns[subset.CELLS_KEY]) == list(sub.obs_names)

    rec = provenance.read_provenance(sub)
    assert [h["step"] for h in rec["history"]] == ["subset"]
    step = rec["history"][0]["params"]
    assert step["parent_id"] == "p1" and step["groups"] == {"column": "cell_type", "values": ["Mono", "B"]}
    assert rec["source"]["origin"] == "subset" and rec["source"]["subset_of"]["n_parent_cells"] == 180

    meta = json.loads((tmp_data_dir / "uploads" / d["dataset_id"] / "metadata.json").read_text())
    assert meta["source"] == "subset" and meta["subset_of"]["id"] == "p1" and meta["n_cells"] == 120
    assert meta["subset_of"]["how"] == {"groups": {"column": "cell_type", "values": ["Mono", "B"]}}
    assert path.stat().st_mtime_ns == before  # the parent's file is untouched

    listed = await client.get(f"{API}/datasets")
    assert d["dataset_id"] in {x["id"] for x in listed.json()}
    info = (await client.get(f"{API}/datasets/{d['dataset_id']}")).json()
    assert info["n_cells"] == 120 and "X_umap_parent" in info["available_embeddings"]


@pytest.mark.asyncio
async def test_filter_and_mask_selectors(client, tmp_data_dir):
    parent = _parent()
    _add(tmp_data_dir, "p1", "pbmc", parent)
    flt = {"op": "and", "items": [{"kind": "category", "column": "cell_type", "values": ["T", "B"]},
                                  {"kind": "range", "column": "score", "min": -1.5}]}
    r = await client.post(f"{API}/datasets/p1/subset", json={"filter": flt, "name": "  T and B  "})
    assert r.status_code == 200, r.text
    d = r.json()
    want = parent.obs["cell_type"].isin(["T", "B"]).to_numpy() & (parent.obs["score"] >= -1.5).to_numpy()
    assert d["n_cells"] == int(want.sum()) and d["name"] == "T and B"
    assert d["described"] == "cell_type in {T, B} AND -1.5 <= score"
    assert list(_new(tmp_data_dir, d["dataset_id"]).obs_names) == list(parent.obs_names[want])

    m = await client.post(f"{API}/datasets/p1/cells/filter",
                          json={"filter": {"op": "and", "items": [{"kind": "gene", "gene": "G25", "op": ">", "value": 2.5}]}})
    mask = m.json()["mask"]
    r = await client.post(f"{API}/datasets/p1/subset", json={"mask": mask})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["described"].startswith("a selection of") and d["name"].startswith("pbmc - ")
    bits = subset.decode_mask(mask, parent.n_obs)
    assert list(_new(tmp_data_dir, d["dataset_id"]).obs_names) == list(parent.obs_names[bits])


@pytest.mark.asyncio
@pytest.mark.parametrize("body,status,words", [
    ({"groups": {"column": "nope", "values": ["T"]}}, 404, "Column 'nope' not found"),
    ({"groups": {"column": "cell_type", "values": ["Q"]}}, 404, "no group 'Q'"),
    ({"groups": {"column": "cell_type", "values": ["NK"]}}, 400, "No cells match"),
    ({"filter": {"op": "and", "items": [{"kind": "range", "column": "score", "min": 2.4}]}}, 400, "at least 50"),
    ({"filter": {"op": "and", "items": [{"kind": "range", "column": "score", "min": -99}]}}, 400,
     "the whole dataset"),
    ({"filter": {"op": "and", "items": [{"kind": "gene", "gene": "XYZ", "op": ">", "value": 1}]}}, 404, "XYZ"),
    ({"mask": "AAAA"}, 400, "Select the cells again"),
    ({"mask": "not base64!"}, 400, "base64"),
])
async def test_refusals_say_why(client, tmp_data_dir, body, status, words):
    _add(tmp_data_dir, "p1", "pbmc", _parent())
    r = await client.post(f"{API}/datasets/p1/subset", json=body)
    assert r.status_code == status, r.text
    assert words in r.json()["detail"]
    assert sorted(p.name for p in (tmp_data_dir / "uploads").iterdir()) == ["p1"]  # nothing made


@pytest.mark.asyncio
async def test_request_shape_is_checked(client, tmp_data_dir):
    _add(tmp_data_dir, "p1", "pbmc", _parent())
    url = f"{API}/datasets/p1/subset"
    both = {"groups": {"column": "cell_type", "values": ["T"]}, "mask": "AA=="}
    assert (await client.post(url, json=both)).status_code == 422
    assert (await client.post(url, json={})).status_code == 422
    assert (await client.post(url, json={"groups": {"column": "cell_type", "values": ["T"]},
                                         "negate": True})).status_code == 422
    r = await client.post(url, json={"groups": {"column": "cell_type", "values": ["T"]}, "name": "‮"})
    assert r.status_code == 400 and "invisible" in r.json()["detail"]
    r = await client.post(f"{API}/datasets/missing/subset", json={"groups": {"column": "c", "values": ["T"]}})
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_dry_run_makes_nothing(client, tmp_data_dir):
    _add(tmp_data_dir, "p1", "pbmc", _parent())
    r = await client.post(f"{API}/datasets/p1/subset",
                          json={"groups": {"column": "cell_type", "values": ["T"]}, "dry_run": True})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["dry_run"] and d["dataset_id"] is None and d["n_cells"] == N_PER
    assert any("protein counts (3 antibodies)" in x for x in d["carried"])
    assert sorted(p.name for p in (tmp_data_dir / "uploads").iterdir()) == ["p1"]
    assert not (tmp_data_dir / "ingested" / "x").exists()


@pytest.mark.asyncio
async def test_backed_parent_reads_only_the_chosen_rows(client, tmp_data_dir, monkeypatch):
    from scview.core import anndata_adaptor, working_copy

    monkeypatch.setattr(anndata_adaptor, "BACKED_MIN_BYTES", 0)
    p = _parent()
    a = ad.AnnData(p.layers["counts"], obs=p.obs, var=p.var)  # counts in X, on disk: the backed path
    a.obsm["X_umap"] = p.obsm["X_umap"]
    _add(tmp_data_dir, "p1", "pbmc", a)
    read = []
    real = working_copy.read_rows

    def spy(M, keep):
        read.append(int(keep.sum()))
        return real(M, keep)

    monkeypatch.setattr(working_copy, "read_rows", spy)
    r = await client.post(f"{API}/datasets/p1/subset", json={"groups": {"column": "cell_type", "values": ["B"]}})
    assert r.status_code == 200, r.text
    sub = _new(tmp_data_dir, r.json()["dataset_id"])
    keep = a.obs["cell_type"].eq("B").to_numpy()
    np.testing.assert_array_equal(sub.X.toarray(), a.X[keep].toarray())
    assert read == [N_PER]


@pytest.mark.asyncio
async def test_large_parent_is_refused_with_what_to_do(client, tmp_data_dir, monkeypatch):
    from scview.core import large_mode

    _add(tmp_data_dir, "p1", "pbmc", _parent())
    monkeypatch.setattr(large_mode, "is_large", lambda obj: True)
    r = await client.post(f"{API}/datasets/p1/subset", json={"groups": {"column": "cell_type", "values": ["B"]}})
    assert r.status_code == 422 and r.json()["large_dataset"] is True
    assert "A subset is not available for large datasets yet" in r.json()["detail"]


@pytest.mark.asyncio
async def test_counts_from_the_source_file_follow_the_barcodes(client, tmp_data_dir):
    """A processed parent without counts: read from its upload by barcode, in the parent's
    order (the upload's rows in another order)."""
    import scanpy as sc

    raw = _parent()
    order = np.random.default_rng(3).permutation(raw.n_obs)
    upload = ad.AnnData(raw.layers["counts"][order], obs=pd.DataFrame(index=raw.obs_names[order]),
                        var=pd.DataFrame(index=raw.var_names))
    _add(tmp_data_dir, "p1", "pbmc", upload, subdir="uploads")
    processed = raw.copy()
    del processed.layers["counts"]
    sc.pp.scale(processed)  # X and raw: no counts anywhere in the derived file
    d = tmp_data_dir / "derived" / "p1"
    d.mkdir(parents=True)
    processed.write_h5ad(d / "pbmc.h5ad")
    r = await client.post(f"{API}/datasets/p1/subset", json={"groups": {"column": "cell_type", "values": ["T", "Mono"]}})
    assert r.status_code == 200, r.text
    assert r.json()["counts_source"] == "X of the original upload"
    sub = _new(tmp_data_dir, r.json()["dataset_id"])
    keep = raw.obs["cell_type"].isin(["T", "Mono"]).to_numpy()
    np.testing.assert_array_equal(sub.X.toarray(), raw.layers["counts"][keep].toarray())


def test_counts_before_an_ambient_correction_come_first():
    a = _parent()
    a.layers["counts_uncorrected"] = a.layers["counts"] * 2
    keep = np.zeros(a.n_obs, dtype=bool)
    keep[::3] = True
    c = subset.counts_rows(a, keep, [])
    np.testing.assert_array_equal(c.X.toarray(), (a.layers["counts"] * 2)[keep].toarray())
    assert "ambient" in c.source


def test_spatial_sections_and_input_files_go_with_the_cells(tmp_path):
    from scview.core import atac_import, modality_store, spatial

    a = _parent()
    a.obs[spatial.LIBRARY_COLUMN] = pd.Categorical(["s1"] * 90 + ["s2"] * 90)
    a.obsm[spatial.KEY] = np.arange(a.n_obs * 2, dtype=np.float64).reshape(-1, 2)
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    a.uns[spatial.KEY] = {lib: {"images": {"hires": img, "lowres": img, "fullres": img},
                                "scalefactors": {"tissue_hires_scalef": 0.5}} for lib in ("s1", "s2")}
    a.uns[spatial.INFO_KEY] = json.dumps({"technology": "visium", "unit": "spot"})
    a.uns[atac_import.UNS_KEY] = json.dumps({"n_peaks": 3})
    store = modality_store.store_path(tmp_path, "p1", atac_import.STORE)
    store.parent.mkdir(parents=True)
    store.write_bytes(b"peaks")
    keep = (a.obs[spatial.LIBRARY_COLUMN] == "s2").to_numpy()
    info = subset.contents(a, tmp_path, "p1")
    assert any("ATAC peak" in x for x in info.carried) and any("tissue images" in x for x in info.carried)
    sub, _ = subset.build(a, keep, [], info)
    assert list(sub.uns[spatial.KEY]) == ["s2"]
    assert set(sub.uns[spatial.KEY]["s2"]["images"]) == {"hires", "lowres"}
    np.testing.assert_array_equal(sub.obsm[spatial.KEY], a.obsm[spatial.KEY][keep])
    assert list(sub.obs[spatial.LIBRARY_COLUMN].cat.categories) == ["s2"]
    assert atac_import.UNS_KEY in sub.uns
    sel = subset.Selection(keep, {"mask": True}, "x")
    subset.record(sub, parent={"id": "p1", "name": "p"}, sel=sel, n_parent=a.n_obs, counts_source="X", info=info)
    subset.persist(tmp_path, "new1", "p - s2", sub, parent={"id": "p1", "name": "p"}, sel=sel,
                   n_parent=a.n_obs, info=info)
    assert modality_store.store_path(tmp_path, "new1", atac_import.STORE).read_bytes() == b"peaks"
    back = ad.read_h5ad(next((tmp_path / "ingested" / "new1").glob("*.h5ad")))
    assert back.uns[spatial.KEY]["s2"]["images"]["hires"].shape == (4, 4, 3)


def test_a_failed_write_leaves_nothing(tmp_path, monkeypatch):
    a = _parent()
    keep = a.obs["cell_type"].eq("T").to_numpy()
    info = subset.contents(a, tmp_path, "p1")
    sub, _ = subset.build(a, keep, [], info)
    sel = subset.Selection(keep, {"mask": True}, "x")

    def broken(self, *args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(ad.AnnData, "write_h5ad", broken)
    with pytest.raises(OSError):
        subset.persist(tmp_path, "new1", "n", sub, parent={"id": "p1"}, sel=sel, n_parent=a.n_obs, info=info)
    assert not (tmp_path / "ingested" / "new1").exists() and not (tmp_path / "uploads" / "new1").exists()


def test_a_run_keeps_the_parent_umap_and_columns():
    from scview.core.pipeline import clear_processing

    a = _parent()
    keep = a.obs["cell_type"].eq("T").to_numpy()
    sub, _ = subset.build(a, keep, [], subset.contents(a, None, "p1"))
    sub.obsm["X_pca"] = np.zeros((sub.n_obs, 2))
    sub.obs["scview_leiden"] = "0"
    clear_processing(sub)  # what reset_to_counts clears
    assert "X_umap_parent" in sub.obsm and "X_pca" not in sub.obsm
    assert "parent_leiden" in sub.obs and "scview_leiden" not in sub.obs


def test_obs_names_map_never_collides():
    m = subset.obs_names_map(["scview_leiden", "parent_leiden", "scview_celltypist_x", "scview_mito_type"])
    assert m == {"scview_leiden": "parent_leiden_2", "parent_leiden": "parent_leiden",
                 "scview_celltypist_x": "parent_celltypist_x", "scview_mito_type": "scview_mito_type"}


@pytest.mark.asyncio
async def test_replay_script_makes_the_same_subset_from_the_parent(client, tmp_data_dir, tmp_path):
    parent = _parent()
    parent.obsm["X_umap_original"] = parent.obsm["X_umap"] * 2
    path = _add(tmp_data_dir, "p1", "pbmc", parent)
    r = await client.post(f"{API}/datasets/p1/subset",
                          json={"filter": {"op": "and", "items": [{"kind": "gene", "gene": "G15", "op": ">", "value": 2}]}})
    assert r.status_code == 200, r.text
    sub = _new(tmp_data_dir, r.json()["dataset_id"])
    (step,) = replay_script.describe(sub)
    assert step["status"] == replay_script.EXACT
    script = replay_script.build_script(sub, dataset_id="x", dataset_name="pbmc - sub")
    assert "subset_of_parent(adata, SUBSET_CELLS, columns=" in script and "G15 > 2" in script
    assert "INPUT = sys.argv[1] if len(sys.argv) > 1 else 'scview_pbmc.h5ad'" in script
    assert "/api/v1/datasets/p1/export/h5ad" in script
    f = tmp_path / "replay.py"
    f.write_text(script)
    out = tmp_path / "out.h5ad"
    done = subprocess.run([sys.executable, str(f), str(path), str(out)], capture_output=True, text=True,
                          timeout=300, cwd=tmp_path)
    assert done.returncode == 0, done.stderr
    rep = ad.read_h5ad(out)
    assert list(rep.obs_names) == list(sub.obs_names) and list(rep.obs.columns) == list(sub.obs.columns)
    np.testing.assert_array_equal(rep.X.toarray(), sub.X.toarray())
    assert set(rep.obsm) == set(sub.obsm)
    np.testing.assert_allclose(rep.obsm["X_umap_parent"], sub.obsm["X_umap_parent"])


def test_replay_without_kept_barcodes_says_so():
    a = ad.AnnData(np.zeros((3, 2), dtype=np.float32))
    provenance.record_step(a, step="subset", tool="t", params={"parent_name": "p", "n_cells": 3})
    (step,) = replay_script.describe(a)
    assert step["status"] == replay_script.NOT_REPLAYABLE


def test_estimate_scales_with_the_cells_chosen(tmp_path):
    a = _parent()
    p = tmp_path / "a.h5ad"
    a.write_h5ad(p)
    half, whole = subset.read_need(str(p), 90, 180), subset.read_need(str(p), 180, 180)
    assert subset.FIXED_BYTES < half < whole
    assert subset.read_need(None, 1, 2) == 0


def _processed_parent() -> ad.AnnData:
    """_parent() as a pipeline run leaves it: QC and doublet columns, variable genes and the
    scaling's mean/std in var, filtering and doublet detection recorded, a hand-label column."""
    import scanpy as sc

    from scview.core.user_labels import USER_COLUMNS_KEY

    a = _parent()
    q = ad.AnnData(a.layers["counts"].copy(), obs=pd.DataFrame(index=a.obs_names), var=a.var.copy())
    q.var["mt"] = False
    sc.pp.calculate_qc_metrics(q, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)
    for c in q.obs.columns:
        a.obs[c] = q.obs[c]
    a.obs["doublet_score"] = np.linspace(0, 0.5, a.n_obs)
    a.obs["predicted_doublet"] = a.obs["doublet_score"] > 0.45
    a.obs["author_fine"] = pd.Categorical(np.where(a.obs["cell_type"] == "T", "T naive", "Other"))
    a.uns[USER_COLUMNS_KEY] = json.dumps(["author_fine"])
    for c, v in (("highly_variable", np.arange(a.n_vars) < 20), ("means", 1.0), ("mean", 0.5), ("std", 1.0),
                 ("n_cells_by_counts", 3), ("gene_ids", [f"ENSG{i}" for i in range(a.n_vars)])):
        a.var[c] = v
    provenance.record_step(a, step="doublet_detection", tool="test")
    provenance.record_step(a, step="filtering", tool="test", params={"qc_threshold_mode": "mad"})
    return a


@pytest.mark.asyncio
async def test_a_processed_parent_gives_a_subset_that_assesses_as_unprocessed(client, tmp_data_dir):
    import uuid

    from scview.core import assessor

    _add(tmp_data_dir, "p1", "pbmc", _processed_parent())
    r = await client.post(f"{API}/datasets/p1/subset", json={"groups": {"column": "cell_type", "values": ["T", "B"]}})
    assert r.status_code == 200, r.text
    d = r.json()
    assert str(uuid.UUID(d["dataset_id"])) == d["dataset_id"]  # the dashed form other ids have
    assert "filtering" not in d["steps"] and "doublet_detection" not in d["steps"]
    assert {"qc_metrics", "highly_variable_genes", "pca", "neighbors", "clustering", "embeddings"} <= set(d["steps"])
    assert "parent was filtered by scView" in d["steps_note"] and "low-RNA" in d["steps_note"]
    assert "highly_variable_genes" in d["next"] and "filtering" not in d["next"].split("Left out")[0]
    sub = _new(tmp_data_dir, d["dataset_id"])
    # the parent's QC and doublet values as parent_*, its per-gene results left behind
    assert {"parent_n_genes_by_counts", "parent_total_counts", "parent_doublet_score",
            "parent_predicted_doublet", "parent_leiden", "author_fine"} <= set(sub.obs.columns)
    assert not {"n_genes_by_counts", "total_counts", "doublet_score", "predicted_doublet"} & set(sub.obs.columns)
    assert list(sub.var.columns) == ["gene_ids"]
    state = assessor.assess_preprocessing(sub)
    for step in ("qc_metrics", "highly_variable_genes", "pca", "neighbors", "clustering", "embeddings",
                 "marker_genes"):
        assert getattr(state, step).done is False, step
    assert state.filtering.done and "filtered by scView" in state.filtering.details
    assert state.doublet_detection.done and "parent_doublet_score" in state.doublet_detection.details
    # the step's time and memory, and the parent's own source
    (h,) = provenance.read_provenance(sub)["history"]
    assert h["resources"]["seconds"] >= 0 and h["params"]["parent_source"]["origin"] == "upload"
    assert h["params"]["parent_qc"]["filtered"].startswith("The parent was filtered")
    # the parent's hand labels stay editable
    r = await client.post(f"{API}/datasets/{d['dataset_id']}/labels",
                          json={"column": "author_fine", "label": "T memory", "indices": [0, 1, 2]})
    assert r.status_code == 200, r.text


def test_an_unfiltered_parent_keeps_filtering_in_the_run():
    a = _parent()
    info = subset.contents(a, None, "p1")
    steps, note = subset.default_steps(info.parent_qc)
    assert steps == subset.FULL_RUN and note is None


def test_a_subset_of_a_subset_keeps_one_generation(tmp_path):
    a = _parent()
    keep = a.obs["cell_type"].isin(["T", "B"]).to_numpy()
    info = subset.contents(a, None, "p1")
    s, _ = subset.build(a, keep, [], info)
    subset.record(s, parent={"id": "p1", "name": "p", "columns": list(a.obs.columns)},
                  sel=subset.Selection(keep, {"mask": True}, "x"), n_parent=a.n_obs, counts_source="X", info=info)
    s.obs["scview_leiden"] = pd.Categorical((np.arange(s.n_obs) % 3).astype(str))  # its own clusters
    s.obsm["X_umap"] = np.zeros((s.n_obs, 2))
    info2 = subset.contents(s, None, "s1")
    assert info2.dropped == ["parent_leiden"] and any("grandparent" in x for x in info2.not_carried)
    t, _ = subset.build(s, np.arange(s.n_obs) < 60, [], info2)
    assert "parent_leiden" in t.obs and "parent_leiden_2" not in t.obs
    assert set(t.obs["parent_leiden"].cat.categories) <= {"0", "1", "2"}  # the parent's, not the grandparent's


@pytest.mark.asyncio
async def test_a_filter_labelling_replays_exactly(client, tmp_data_dir, tmp_path):
    a = _parent()
    del a.uns["scview_provenance"]
    path = _add(tmp_data_dir, "p1", "pbmc", a)
    flt = {"op": "or", "layer": "", "items": [{"kind": "category", "column": "cell_type", "values": ["T"]},
                                              {"kind": "range", "column": "score", "min": 1.5}]}
    m = await client.post(f"{API}/datasets/p1/cells/filter", json={"filter": flt})
    idx = np.flatnonzero(subset.decode_mask(m.json()["mask"], a.n_obs)).tolist()
    r = await client.post(f"{API}/datasets/p1/labels", json={"column": "picked", "label": "yes", "indices": idx,
                                                              "new_column": True, "filter": flt})
    assert r.status_code == 200, r.text
    # a filter that does not give these cells is not recorded (the labelling stays not replayable)
    r = await client.post(f"{API}/datasets/p1/labels", json={"column": "picked", "label": "no", "indices": [0],
                                                              "filter": flt})
    assert r.status_code == 200, r.text
    (derived,) = (tmp_data_dir / "derived" / "p1").glob("*.h5ad")
    labelled = ad.read_h5ad(derived)
    steps = replay_script.describe(labelled)
    assert [s["status"] for s in steps] == [replay_script.EXACT, replay_script.NOT_REPLAYABLE]
    script = replay_script.build_script(labelled)
    f = tmp_path / "replay.py"
    f.write_text(script.replace("label_by_filter(adata, 'picked', 'no'", "pass  # (label_by_filter(adata, 'picked', 'no'"))
    out = tmp_path / "out.h5ad"
    done = subprocess.run([sys.executable, str(f), str(path), str(out)], capture_output=True, text=True,
                          timeout=300, cwd=tmp_path)
    assert done.returncode == 0, done.stderr
    rep = ad.read_h5ad(out)
    want = labelled.obs["picked"].astype(str).replace({"no": "yes"})  # the hand label on cell 0 is not replayed
    assert (rep.obs["picked"].astype(str).to_numpy() == want.to_numpy()).all()


def test_replay_names_a_missing_parent_column(tmp_path):
    a = _parent()
    keep = a.obs["cell_type"].eq("T").to_numpy()
    info = subset.contents(a, None, "p1")
    sub, src = subset.build(a, keep, [], info)
    subset.record(sub, parent={"id": "p1", "name": "p", "columns": [*a.obs.columns, "author_fine"]},
                  sel=subset.Selection(keep, {"mask": True}, "x"), n_parent=a.n_obs, counts_source=src, info=info)
    f = tmp_path / "replay.py"
    f.write_text(replay_script.build_script(sub))
    p = tmp_path / "parent.h5ad"
    a.write_h5ad(p)
    done = subprocess.run([sys.executable, str(f), str(p), str(tmp_path / "o.h5ad")], capture_output=True,
                          text=True, timeout=300, cwd=tmp_path)
    assert done.returncode != 0 and "INPUT lacks 1 column(s) the subset carried (author_fine)" in done.stderr


def test_steps_note_names_only_what_was_left_out():
    """Lead check 2026-11-17: doublets left out, filtering kept, but the note warned about filtering."""
    steps, note = subset.default_steps({"filtered": None, "doublets": "Doublet detection ran on the parent"})
    assert "filtering" in steps and "doublet_detection" not in steps
    assert note.startswith("Left out: doublet_detection.") and "Filtering again" not in note
    assert note.endswith("Add the step to do it anyway.")
    steps, note = subset.default_steps({"filtered": "The parent was filtered by scView", "doublets": "ran"})
    assert "filtering" not in steps and "Filtering again" in note and note.endswith("to do them anyway.")
    assert subset.default_steps({"filtered": None, "doublets": None}) == (subset.FULL_RUN, None)
