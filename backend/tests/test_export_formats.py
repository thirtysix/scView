"""CELLxGENE-compatible h5ad, Discover readiness report and Vitessce bundle exports
(core/cellxgene_export.py, core/vitessce_export.py, api/v1/export_formats.py)."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
import scipy.sparse as sp
import zarr

from scview.core import cellxgene_export as cx
from scview.core import vitessce_export as vx
from tests.large_fixture import counts_adata, register

API = "/api/v1"


def _processed(n: int = 300, g: int = 120) -> ad.AnnData:
    """As scView keeps a processed dataset: X log-normalised, .raw = X, counts layer."""
    rng = np.random.default_rng(1)
    counts = rng.poisson(0.6, (n, g)).astype(np.float32)
    counts[: n // 2, :10] += 5
    a = ad.AnnData(sp.csr_matrix(counts))
    a.obs_names = [f"c{i}" for i in range(n)]
    a.obs_names = list(a.obs_names[:-1]) + ["c0"]  # one duplicate cell name
    a.var_names = [f"ENSG{i:011d}" for i in range(g)]
    a.var["gene_symbols"] = [f"GENE{i}" for i in range(g)]
    a.var.iloc[5, 0] = "GENE4"  # duplicate symbol
    a.layers["counts"] = a.X.copy()
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    a.raw = a
    a.var["highly_variable"] = np.arange(g) < 40
    a.obs["leiden"] = pd.Categorical([str(i % 3) for i in range(n)])
    a.obs["cell_type"] = pd.Categorical(["T" if i % 2 else "B" for i in range(n)])
    a.obs["donor"] = [f"d{i % 4}" for i in range(n)]  # text column
    a.obs["is_doublet"] = (np.arange(n) % 10) == 0  # bool
    a.obs["n_counts"] = counts.sum(1)
    a.obs["odd/name"] = pd.Categorical(["x"] * n)
    a.uns["leiden_colors"] = ["#ff0000", "#00ff00", "#0000ff"]
    a.uns["scview_provenance"] = {"x": 1}
    a.uns["rank_genes_groups__leiden"] = {"params": {"groupby": "leiden"}}
    a.obsm["X_umap"] = rng.normal(size=(n, 2)).astype(np.float64)
    a.obsm["X_pca"] = rng.normal(size=(n, 10)).astype(np.float32)
    a.obsm["X_umap_3d"] = rng.normal(size=(n, 3)).astype(np.float32)
    a.obsm["X_nmf_usage"] = rng.random((n, 4)).astype(np.float32)
    a.obsm["module_scores"] = rng.random((n, 3)).astype(np.float32)
    a.obsp["connectivities"] = sp.identity(n, format="csr")
    a.varm["PCs"] = rng.normal(size=(g, 5))
    return a


def _register(dm, a: ad.AnnData | None = None, did: str = "fx1", name: str = "My PBMC") -> Path:
    root = Path(dm.data_dir)
    up = root / "uploads" / did
    up.mkdir(parents=True)
    (up / "metadata.json").write_text(json.dumps({"id": did, "name": name, "status": "ready"}))
    (root / "derived" / did).mkdir(parents=True)
    path = root / "derived" / did / "data.h5ad"
    (a if a is not None else _processed()).write_h5ad(path, compression="gzip")
    return path


def _events(text: str) -> list[tuple[str, dict]]:
    out = []
    for block in text.split("\n\n"):
        lines = [ln for ln in block.splitlines() if ln and not ln.startswith(":")]
        if not lines:
            continue
        ev = lines[0].removeprefix("event: ")
        out.append((ev, json.loads(lines[1].removeprefix("data: "))))
    return out


async def _build(client, did: str, fmt: str, **body) -> tuple[dict, bytes]:
    r = await client.post(f"{API}/datasets/{did}/export/{fmt}/build-stream", json=body)
    assert r.status_code == 200, r.text
    events = _events(r.text)
    assert events[-1][0] == "complete", events
    done = events[-1][1]
    f = await client.get(f"{API}/datasets/{did}/export/files/{done['file']}")
    assert f.status_code == 200
    assert len(f.content) == done["size_bytes"]
    return done, f.content


# ---------------------------------------------------------------------------
# Pieces


def test_clean_obs_makes_labels_categorical_and_names_unique():
    a = _processed()
    obs, notes, left, renamed = cx.clean_obs(a.obs, None)
    assert renamed == 1 and obs.index.is_unique
    assert isinstance(obs["donor"].dtype, pd.CategoricalDtype)
    assert list(obs["is_doublet"].cat.categories) == ["False", "True"]
    assert pd.api.types.is_numeric_dtype(obs["n_counts"])
    assert "odd_name" in obs.columns and any("odd/name" in n for n in notes)
    assert left == []
    with pytest.raises(cx.ExportError):
        cx.clean_obs(a.obs, ["nope"])


def test_symbol_var_puts_symbols_in_the_index_and_keeps_ensembl_ids():
    var, notes, renamed = cx.symbol_var(_processed().var)
    assert var.index[0] == "GENE0" and var["gene_ids"].iloc[0] == "ENSG00000000000"
    assert var.index.is_unique and renamed == 1
    assert "gene_symbols" not in var.columns
    assert any("symbols" in n for n in notes)


def test_symbol_index_left_alone():
    var = pd.DataFrame({"gene_ids": ["ENSG00000000001", "ENSG00000000002"]}, index=["CD3E", "MS4A1"])
    out, _, renamed = cx.symbol_var(var)
    assert list(out.index) == ["CD3E", "MS4A1"] and renamed == 0


def test_embedding_choices_default_to_2d_and_3d_layouts():
    choices = {c["key"]: c for c in cx.embedding_choices(_processed())}
    assert set(choices) == {"X_umap", "X_pca", "X_umap_3d"}  # not usage or protein
    assert choices["X_umap"]["default"] and choices["X_umap_3d"]["default"]
    assert not choices["X_pca"]["default"]
    # Vitessce draws 2D scatters only: the 3D UMAP is offered, not chosen by default
    flat = {c["key"]: c for c in cx.embedding_choices(_processed(), vx.MAX_DIMS)}
    assert flat["X_umap"]["default"] and not flat["X_umap_3d"]["default"]


# ---------------------------------------------------------------------------
# CELLxGENE


async def test_cellxgene_info_lists_contents_size_and_readiness(client, dataset_manager):
    _register(dataset_manager)
    r = await client.get(f"{API}/datasets/fx1/export/h5ad/info", params={"format": "cellxgene"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["filename"] == "scview_My_PBMC_cellxgene.h5ad"
    assert body["exported_embeddings"] == ["X_umap", "X_umap_3d"]
    assert "scview_provenance" in body["dropped"]["uns"]
    assert "leiden_colors" not in body["dropped"]["uns"]
    assert set(body["dropped"]["obsm"]) == {"X_pca", "X_nmf_usage", "module_scores"}
    assert body["dropped"]["layers"] == ["counts"] and body["dropped"]["obsp"] == ["connectivities"]
    assert body["renamed"] == {"cells": 1, "genes": 1}
    assert body["has_counts"] and body["raw_from"] is None
    rd = body["readiness"]
    assert rd["schema_version"] == cx.SCHEMA_VERSION and not rd["ready"]
    by = {i["key"]: i for i in rd["items"]}
    assert by["organism_ontology_term_id"]["status"] == "missing"
    assert by["raw_counts"]["status"] == "missing" and "Include raw counts" in by["raw_counts"]["detail"]
    assert by["reserved_obs"]["status"] == "invalid" and "cell_type" in by["reserved_obs"]["detail"]
    assert "'donor'" in by["donor_id"]["detail"]
    assert by["var_index"]["status"] == "warn"
    # with raw counts chosen
    r = await client.get(f"{API}/datasets/fx1/export/h5ad/info",
                         params={"format": "cellxgene", "raw": "true", "embedding": ["X_umap", "X_pca"]})
    body = r.json()
    assert body["raw_from"] == "layers['counts']"
    assert body["exported_embeddings"] == ["X_umap", "X_pca"]
    assert {i["key"]: i for i in body["readiness"]["items"]}["raw_counts"]["status"] == "ok"


async def test_cellxgene_bad_options(client, dataset_manager):
    _register(dataset_manager)
    r = await client.get(f"{API}/datasets/fx1/export/h5ad/info",
                         params={"format": "cellxgene", "embedding": "X_nmf_usage"})
    assert r.status_code == 400
    r = await client.get(f"{API}/datasets/fx1/export/h5ad/info", params={"format": "loom"})
    assert r.status_code == 404
    r = await client.post(f"{API}/datasets/fx1/export/cellxgene/build-stream", json={"columns": ["nope"]})
    assert r.status_code == 400
    assert (await client.get(f"{API}/datasets/fx1/export/files/..%2Fx.h5ad")).status_code == 404
    assert (await client.get(f"{API}/datasets/fx1/export/files/cellxgene-0123456789abcdef.h5ad")).status_code == 404


async def test_cellxgene_file_opens_with_the_promised_layout(client, dataset_manager, tmp_path):
    src = _register(dataset_manager)
    done, content = await _build(client, "fx1", "cellxgene", include_raw=True,
                                 embeddings=["X_umap", "X_pca"])
    out = tmp_path / "x.h5ad"
    out.write_bytes(content)
    b = ad.read_h5ad(out)
    a = ad.read_h5ad(src)
    assert b.shape == a.shape and b.obs_names.is_unique and b.var_names.is_unique
    assert b.var_names[0] == "GENE0" and b.var["gene_ids"].iloc[0] == "ENSG00000000000"
    assert set(b.obsm) == {"X_umap", "X_pca"}
    assert b.obsm["X_umap"].dtype == np.float32 and b.obsm["X_pca"].shape == (a.n_obs, 2)
    assert isinstance(b.obs["donor"].dtype, pd.CategoricalDtype)
    assert list(b.uns["leiden_colors"]) == list(a.uns["leiden_colors"])
    assert b.uns["default_embedding"] == "X_umap"
    note = b.uns["scview_export"]
    assert note["dataset_id"] == "fx1" and note["format"] == "cellxgene" and note["scview_version"]
    assert not b.layers and not b.obsp and not b.varm
    assert "scview_provenance" not in b.uns
    np.testing.assert_allclose(b.X.toarray(), a.X.toarray(), rtol=1e-6)
    np.testing.assert_array_equal(b.raw.X.toarray(), a.layers["counts"].toarray())
    assert list(b.raw.var_names) == list(b.var_names)
    # never fills in a Discover field
    assert not [c for c in b.obs.columns if c.endswith("_ontology_term_id")]
    # the estimate is in the right range
    info = (await client.get(f"{API}/datasets/fx1/export/h5ad/info",
                             params={"format": "cellxgene", "raw": "true",
                                     "embedding": ["X_umap", "X_pca"]})).json()
    assert 0.5 < info["size_bytes"] / done["size_bytes"] < 2.0
    # the same options again: served from the kept file
    again, _ = await _build(client, "fx1", "cellxgene", include_raw=True, embeddings=["X_umap", "X_pca"])
    assert again["file"] == done["file"]
    files = list((Path(dataset_manager.data_dir) / "cache" / "fx1" / "exports").iterdir())
    assert [f.name for f in files] == [done["file"]]


async def test_cellxgene_from_a_large_dataset_copies_the_files(client, tmp_data_dir, tmp_path):
    counts = counts_adata()
    h5ad = register(tmp_data_dir, "bigX", counts)
    done, content = await _build(client, "bigX", "cellxgene", include_raw=True)
    out = tmp_path / "big.h5ad"
    out.write_bytes(content)
    b = ad.read_h5ad(out)
    with h5py.File(h5ad, "r") as f:
        x = ad.io.read_elem(f["X"])
    np.testing.assert_allclose(b.X.toarray(), x.toarray())
    np.testing.assert_array_equal(b.raw.X.toarray(), counts.X.toarray())
    assert "scview_large" not in b.uns and "X_umap" in b.obsm


def test_readiness_accepts_valid_fields_and_flags_bad_terms():
    a = _processed()
    n = a.n_obs
    a.obs = a.obs.drop(columns=["cell_type"])
    good = {
        "assay_ontology_term_id": "EFO:0009922",
        "tissue_ontology_term_id": "UBERON:0000178",
        "disease_ontology_term_id": "MONDO:0004604 || MONDO:0043004",
        "cell_type_ontology_term_id": "CL:0000236", "development_stage_ontology_term_id": "unknown",
        "sex_ontology_term_id": "PATO:0000383",
        "self_reported_ethnicity_ontology_term_id": "HANCESTRO:0005 || HANCESTRO:0014",
        "suspension_type": "cell", "tissue_type": "tissue", "donor_id": "d1",
    }
    for k, v in good.items():
        a.obs[k] = pd.Categorical([v] * n)
    a.obs["is_primary_data"] = True
    a.var["feature_is_filtered"] = False
    uns = {"organism_ontology_term_id": "NCBITaxon:9606"}  # in uns since schema 6.0.0

    class _A:
        adata = a
        h5ad_path = "/nonexistent.h5ad"
        is_large = False

    def check(**kw):
        p = cx.plan(_A(), cx.Options(include_raw=True))
        return {i["key"]: i for i in cx.readiness(p, **kw)["items"]}

    by = check(uns=uns)
    assert cx.SCHEMA_VERSION == "7.1.0"
    for k in [*good, "organism_ontology_term_id"]:
        assert by[k]["status"] == "ok", by[k]
    assert by["is_primary_data"]["status"] == "ok" and by["raw_counts"]["status"] == "ok"
    assert "reserved_obs" not in by and "deprecated_obs" not in by
    assert check()["organism_ontology_term_id"]["status"] == "missing"
    assert check(uns={"organism_ontology_term_id": "9606"})["organism_ontology_term_id"]["status"] == "invalid"
    # a cell line: Cellosaurus tissue, "na" donor-level fields
    for k, v in {"tissue_type": "cell line", "tissue_ontology_term_id": "CVCL_1P02",
                 "sex_ontology_term_id": "na", "development_stage_ontology_term_id": "na",
                 "self_reported_ethnicity_ontology_term_id": "na", "cell_type_ontology_term_id": "na"}.items():
        a.obs[k] = pd.Categorical([v] * n)
    by = check(uns=uns)
    for k in ("tissue_type", "tissue_ontology_term_id", "sex_ontology_term_id", "cell_type_ontology_term_id"):
        assert by[k]["status"] == "ok", by[k]
    # renamed / old forms are flagged
    a.obs["tissue_type"] = pd.Categorical(["cell culture"] * n)  # "primary cell culture" since 7.0.0
    a.obs["tissue_ontology_term_id"] = pd.Categorical(["blood"] * n)
    a.obs["disease_ontology_term_id"] = pd.Categorical(["PATO:0000001"] * n)
    a.obs["self_reported_ethnicity_ontology_term_id"] = pd.Categorical(["HANCESTRO:0005,HANCESTRO:0014"] * n)
    a.obs["organism_ontology_term_id"] = pd.Categorical(["NCBITaxon:9606"] * n)
    by = check()
    for k in ("tissue_type", "tissue_ontology_term_id", "disease_ontology_term_id",
              "self_reported_ethnicity_ontology_term_id", "deprecated_obs"):
        assert by[k]["status"] == "invalid", by[k]
    assert "moved it to uns" in by["organism_ontology_term_id"]["detail"]
    a.obs["disease_ontology_term_id"] = pd.Categorical(["MONDO:0043004 || MONDO:0004604"] * n)  # unsorted
    assert check()["disease_ontology_term_id"]["status"] == "invalid"


def test_a_stopped_write_leaves_no_file(tmp_path):
    a = _processed()
    path = tmp_path / "d.h5ad"
    a.write_h5ad(path)

    class _A:
        adata = a
        h5ad_path = str(path)
        is_large = False

    p = cx.plan(_A(), cx.Options())
    out = tmp_path / "out.h5ad"
    with pytest.raises(cx.Cancelled):
        cx.write(p, out, {}, stop=lambda: True)
    assert list(tmp_path.iterdir()) == [path]
    pv = vx.plan(_A(), cx.Options())
    with pytest.raises(cx.Cancelled):
        vx.write(pv, tmp_path / "v.zip", "d", "d", {}, stop=lambda: True)
    assert list(tmp_path.iterdir()) == [path]


# ---------------------------------------------------------------------------
# Vitessce


async def test_vitessce_bundle_paths_exist_and_values_match(client, dataset_manager, tmp_path):
    src = _register(dataset_manager)
    info = (await client.get(f"{API}/datasets/fx1/export/h5ad/info",
                             params={"format": "vitessce"})).json()
    assert info["filename"] == "scview_My_PBMC_vitessce.zip"
    assert set(info["obs_columns"]) == {"leiden", "cell_type", "donor", "is_doublet"}
    done, content = await _build(client, "fx1", "vitessce", columns=["leiden", "cell_type"])
    zf = zipfile.ZipFile(io.BytesIO(content))
    names = zf.namelist()
    assert "config.json" in names and "README.txt" in names
    assert "My_PBMC.zarr/.zmetadata" in names
    zf.extractall(tmp_path)
    cfg = json.loads((tmp_path / "config.json").read_text())
    assert cfg["version"] == vx.CONFIG_VERSION
    f = cfg["datasets"][0]["files"][0]
    assert f["fileType"] == "anndata.zarr" and f["url"].startswith("http://127.0.0.1:")
    assert f["url"].endswith("/My_PBMC.zarr")
    components = [v["component"] for v in cfg["layout"]]
    assert components.count("scatterplot") == 1  # X_umap (no X_umap_3d: Vitessce is 2D)
    for c in ("obsSets", "obsSetSizes", "featureList", "heatmap"):
        assert c in components
    store = tmp_path / "My_PBMC.zarr"
    meta = json.loads((store / ".zmetadata").read_text())["metadata"]
    for path in vx.config_paths(cfg):
        assert f"{path}/.zarray" in meta or f"{path}/.zgroup" in meta, path
    # every scatterplot names an embedding type the space defines
    space = cfg["coordinationSpace"]["embeddingType"]
    for v in cfg["layout"]:
        if v["component"] == "scatterplot":
            assert v["coordinationScopes"]["embeddingType"] in space
    g = zarr.open_group(store, mode="r", zarr_format=2)
    a = ad.read_h5ad(src)
    X = g["X"]
    assert X.shape == a.shape and X.chunks[1] == vx.GENE_CHUNK
    np.testing.assert_allclose(X[:, 3], a.X[:, 3].toarray().ravel(), rtol=1e-6)
    b = ad.read_zarr(store)
    assert list(b.obs.columns) == ["leiden", "cell_type"]
    assert b.var_names[0] == "GENE0"
    assert 0 < int(b.var["heatmap_genes"].sum()) <= 40  # from the variable genes
    assert b.uns["scview_export"]["format"] == "vitessce"
    assert "127.0.0.1" in (tmp_path / "README.txt").read_text()
    assert 0.3 < info["size_bytes"] / done["size_bytes"] < 3.0 or done["size_bytes"] < 200_000


async def test_vitessce_variable_genes_only(client, dataset_manager, tmp_path):
    _register(dataset_manager)
    _, content = await _build(client, "fx1", "vitessce", genes="variable")
    zipfile.ZipFile(io.BytesIO(content)).extractall(tmp_path)
    b = ad.read_zarr(tmp_path / "My_PBMC.zarr")
    assert b.n_vars == 40


async def test_vitessce_refusals(client, dataset_manager):
    a = _processed()
    del a.var["highly_variable"]
    _register(dataset_manager, a)
    r = await client.get(f"{API}/datasets/fx1/export/h5ad/info",
                         params={"format": "vitessce", "genes": "variable"})
    assert r.status_code == 400 and "variable" in r.json()["detail"]
    r = await client.get(f"{API}/datasets/fx1/export/h5ad/info",
                         params={"format": "vitessce", "column": "n_counts"})
    assert r.status_code == 400


async def test_vitessce_from_a_large_dataset(client, tmp_data_dir, tmp_path):
    counts = counts_adata()
    h5ad = register(tmp_data_dir, "bigV", counts)
    _, content = await _build(client, "bigV", "vitessce", columns=["leiden"])
    zipfile.ZipFile(io.BytesIO(content)).extractall(tmp_path)
    store = next(tmp_path.glob("*.zarr"))
    X = zarr.open_group(store, mode="r", zarr_format=2)["X"]
    with h5py.File(h5ad, "r") as f:
        x = ad.io.read_elem(f["X"])
    np.testing.assert_allclose(X[:], x.toarray(), rtol=1e-6)
