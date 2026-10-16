"""Downloads: the dataset's own .h5ad, cell metadata, and marker tables."""

from __future__ import annotations

import io
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scanpy as sc

from scview.api.v1.export import file_stem

API = "/api/v1"


def _adata() -> ad.AnnData:
    rng = np.random.default_rng(0)
    X = rng.poisson(1.0, (60, 12)).astype("float32")
    X[:30, :4] += 6
    a = ad.AnnData(X)
    a.obs_names = [f"cell{i}" for i in range(60)]
    a.var_names = [f"G{i}" for i in range(12)]
    a.obs["cluster"] = pd.Categorical(["0"] * 30 + ["1"] * 30)
    a.obs["cell_type"] = pd.Categorical(["T"] * 20 + ["B"] * 20 + ["NK"] * 20)
    for col in ("cluster", "cell_type"):
        sc.tl.rank_genes_groups(a, col, method="wilcoxon", pts=True,
                                key_added=f"rank_genes_groups__{col}")
    return a


def _register(dm, did: str = "ex1", name: str = "PBMC 3k (v2)", derived: bool = True) -> dict[str, Path]:
    root = Path(dm.data_dir)
    up = root / "uploads" / did
    up.mkdir(parents=True)
    (up / "metadata.json").write_text(json.dumps({"id": did, "name": name, "status": "ready"}))
    a = _adata()
    files = {"uploads": up / "orig.h5ad"}
    a.write_h5ad(files["uploads"])
    if derived:
        (root / "derived" / did).mkdir(parents=True)
        files["derived"] = root / "derived" / did / "orig.h5ad"
        a.obs["note"] = "processed"
        a.write_h5ad(files["derived"])
    return files


def test_file_stem_is_filename_safe():
    assert file_stem("PBMC 3k (v2)") == "PBMC_3k_v2"
    assert file_stem("../../etc/passwd") == "etc_passwd"
    assert file_stem("ÄÖ") == "dataset"
    assert len(file_stem("x" * 300)) == 80


async def test_h5ad_info_reports_the_processed_file(client, dataset_manager):
    files = _register(dataset_manager)
    r = await client.get(f"{API}/datasets/ex1/export/h5ad/info")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["filename"] == "scview_PBMC_3k_v2.h5ad"
    assert body["size_bytes"] == files["derived"].stat().st_size
    assert body["source"] == "derived" and body["source_label"] == "processed"


async def test_h5ad_download_streams_the_active_file(client, dataset_manager):
    files = _register(dataset_manager)
    r = await client.get(f"{API}/datasets/ex1/export/h5ad", headers={"Accept-Encoding": "gzip"})
    assert r.status_code == 200
    assert r.content == files["derived"].read_bytes()
    # Not re-gzipped by the middleware, so the browser knows the size up front.
    assert r.headers.get("content-encoding") != "gzip"
    assert int(r.headers["content-length"]) == files["derived"].stat().st_size
    assert 'filename="scview_PBMC_3k_v2.h5ad"' in r.headers["content-disposition"]
    assert r.headers["content-disposition"].startswith("attachment")


async def test_h5ad_download_falls_back_to_the_upload(client, dataset_manager):
    files = _register(dataset_manager, derived=False)
    r = await client.get(f"{API}/datasets/ex1/export/h5ad/info")
    assert r.json()["source_label"] == "as uploaded"
    r = await client.get(f"{API}/datasets/ex1/export/h5ad")
    assert r.content == files["uploads"].read_bytes()


@pytest.mark.parametrize("did", ["%2E%2E", "missing"])
async def test_h5ad_download_unknown_or_unsafe_id_is_404(client, dataset_manager, did):
    _register(dataset_manager)
    assert (await client.get(f"{API}/datasets/{did}/export/h5ad")).status_code == 404
    assert (await client.get(f"{API}/datasets/{did}/export/h5ad/info")).status_code == 404


async def test_h5ad_download_without_a_file_is_404(client, dataset_manager):
    up = Path(dataset_manager.data_dir) / "uploads" / "conv"
    up.mkdir(parents=True)
    (up / "metadata.json").write_text(json.dumps({"id": "conv", "name": "c", "status": "converting"}))
    r = await client.get(f"{API}/datasets/conv/export/h5ad")
    assert r.status_code == 404


async def _export(client, **body):
    return await client.post(f"{API}/datasets/ex1/export", json=body)


async def test_metadata_csv_keeps_cell_ids(client, dataset_manager):
    _register(dataset_manager)
    r = await _export(client, format="csv", data_type="metadata")
    assert r.status_code == 200, r.text
    assert 'filename="scview_PBMC_3k_v2_metadata.csv"' in r.headers["content-disposition"]
    df = pd.read_csv(io.StringIO(r.text), index_col=0)
    assert list(df.index[:2]) == ["cell0", "cell1"]
    assert {"cluster", "cell_type", "note"} <= set(df.columns)


async def test_markers_csv_covers_every_grouping_by_default(client, dataset_manager):
    _register(dataset_manager)
    r = await _export(client, format="csv", data_type="markers")
    assert r.status_code == 200, r.text
    assert 'filename="scview_PBMC_3k_v2_markers.csv"' in r.headers["content-disposition"]
    df = pd.read_csv(io.StringIO(r.text))
    assert df.columns[0] == "grouping"
    assert set(df["grouping"]) == {"cluster", "cell_type"}
    assert set(df.loc[df["grouping"] == "cell_type", "group"]) == {"T", "B", "NK"}


async def test_markers_for_one_column_and_gene_cap(client, dataset_manager):
    _register(dataset_manager)
    r = await _export(client, format="csv", data_type="markers",
                      params={"column": "cell_type", "n_genes": 3})
    assert r.status_code == 200, r.text
    assert "scview_PBMC_3k_v2_markers_cell_type.csv" in r.headers["content-disposition"]
    df = pd.read_csv(io.StringIO(r.text))
    assert set(df["grouping"]) == {"cell_type"}
    assert df.groupby("group").size().max() == 3


async def test_markers_xlsx_has_one_sheet_per_grouping(client, dataset_manager):
    _register(dataset_manager)
    r = await _export(client, format="xlsx", data_type="markers")
    assert r.status_code == 200, r.text
    sheets = pd.read_excel(io.BytesIO(r.content), sheet_name=None)
    assert set(sheets) == {"cluster", "cell_type"}
    assert set(sheets["cluster"]["group"].astype(str)) == {"0", "1"}


async def test_markers_unknown_column_is_404(client, dataset_manager):
    _register(dataset_manager)
    r = await _export(client, format="csv", data_type="markers", params={"column": "nope"})
    assert r.status_code == 404
