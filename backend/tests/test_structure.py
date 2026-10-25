"""AnnData object inspector (core/structure.py, GET /datasets/{id}/structure)."""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

from scview.core import structure as st
from scview.core.anndata_adaptor import AnnDataAdaptor
from tests.large_fixture import counts_adata, register

API = "/api/v1"


def _adata(n: int = 300, genes: int = 50, seed: int = 0) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    counts = sp.csr_matrix(rng.poisson(0.3, (n, genes)).astype(np.float32))
    obs = pd.DataFrame({
        "cluster": pd.Categorical(rng.choice(["a", "b", "c"], n)),
        "score": rng.normal(size=n),
        "flag": rng.random(n) > 0.5,
        "note": [f"cell note {i % 7}" for i in range(n)],
        "empty": np.full(n, np.nan),
    }, index=[f"c{i}" for i in range(n)])
    var = pd.DataFrame({"gene_ids": [f"ENSG{i:05d}" for i in range(genes)]},
                       index=[f"G{i}" for i in range(genes)])
    a = ad.AnnData(X=counts, obs=obs, var=var)
    a.layers["counts"] = counts.copy()
    lognorm = counts.copy()
    lognorm.data = np.log1p(lognorm.data * 3.3)
    a.X = lognorm
    a.obsm["X_umap"] = rng.normal(size=(n, 2)).astype(np.float32)
    a.obsp["connectivities"] = sp.random(n, n, density=0.01, format="csr", random_state=0)
    a.varm["PCs"] = rng.normal(size=(genes, 5))
    a.uns["leiden"] = {"params": {"resolution": 1.0, "n_iterations": -1}}
    a.uns["about_title"] = "A small test"
    a.uns["long_text"] = "x" * 500
    a.uns["big"] = np.arange(100_000)
    a.uns["table"] = pd.DataFrame({"a": [1, 2]})
    a.uns["about_short_title"] = np.array(["Kang"], dtype=object)
    return a


def test_in_memory_matrices_tables_and_uns():
    a = _adata()
    s = st.summarise(_Mem(a))
    assert s["mode"] == "memory" and s["n_obs"] == 300 and s["n_vars"] == 50
    X = s["X"]
    assert X["format"] == "csr" and X["nnz"] == a.X.nnz and X["dtype"] == "float32"
    assert not X["looks_like_counts"] and not X["sample"]["integer"] and X["sample"]["min"] == 0.0
    assert s["layers"]["counts"]["looks_like_counts"]
    assert 0 < X["density"] < 1
    cols = {c["name"]: c for c in s["obs"]["columns"]}
    assert cols["cluster"]["kind"] == "categorical" and cols["cluster"]["n_unique"] == 3
    assert cols["score"]["kind"] == "numeric" and cols["score"]["min"] < 0
    assert cols["flag"]["kind"] == "bool"
    assert cols["note"]["kind"] == "text" and cols["note"]["n_unique"] == 7
    assert cols["empty"]["all_missing"]
    assert {o["key"] for o in s["obsm"]} == {"X_umap"} and s["obsm"][0]["shape"] == [300, 2]
    assert s["obsp"][0]["kind"] == "sparse" and s["varm"][0]["shape"] == [50, 5]
    uns = {n["key"]: n for n in s["uns"]}
    params = uns["leiden"]["children"][0]["children"]
    assert params == [{"key": "n_iterations", "type": "int", "value": -1},
                      {"key": "resolution", "type": "float", "value": 1.0}]
    assert [n["key"] for n in s["uns"]] == sorted(uns)
    assert uns["about_title"]["value"] == "A small test"
    assert "value" not in uns["long_text"] and uns["long_text"]["size"] == 500
    assert uns["big"] == {"key": "big", "type": "array", "shape": [100000], "dtype": "int64"}
    assert uns["table"]["type"] == "DataFrame"
    assert uns["about_short_title"]["value"] == "Kang"
    json.dumps(s)  # serialisable as it is


def test_flags_no_counts_duplicates_and_empty_columns():
    a = _adata()
    del a.layers["counts"]
    a.var_names = ["G1"] + list(a.var_names[1:])  # G1 twice
    s = st.summarise(_Mem(a))
    text = " ".join(f["text"] for f in s["flags"])
    assert "no .raw or layer with counts" in text
    assert "1 duplicate gene names (e.g. G1)" in text
    assert "all missing: empty" in text
    # counts in .raw: no counts flag
    b = _adata()
    del b.layers["counts"]
    b.raw = ad.AnnData(X=_adata().layers["counts"], var=b.var)
    s = st.summarise(_Mem(b))
    assert s["raw"]["looks_like_counts"] and s["raw"]["n_vars"] == 50
    assert not any("no .raw" in f["text"] for f in s["flags"])


def test_scaled_dense_x_without_raw():
    a = _adata()
    a.X = np.random.default_rng(1).normal(size=a.shape)
    s = st.summarise(_Mem(a))
    assert s["X"]["format"] == "dense" and s["X"]["nnz"] is None and s["X"]["sample"]["negative"]
    assert any("negative values (scaled)" in f["text"] for f in s["flags"])


def test_backed_file_is_read_from_its_headers(tmp_path, monkeypatch):
    a = _adata(n=2000)
    a.raw = ad.AnnData(X=a.layers["counts"], var=a.var)
    p = tmp_path / "d.h5ad"
    a.write_h5ad(p)
    monkeypatch.setattr(st, "SAMPLE_VALUES", 400)  # sample slices, not the whole data
    adaptor = AnnDataAdaptor(str(p), backed=True)
    s = st.summarise(adaptor)
    assert s["mode"] == "backed" and adaptor.isbacked
    assert s["X"]["on_disk"] and s["X"]["format"] == "csr" and s["X"]["nnz"] == a.X.nnz
    assert s["X"]["sample"]["n_sampled"] <= 400
    assert s["raw"]["looks_like_counts"] and s["layers"]["counts"]["looks_like_counts"]
    mem = st.summarise(_Mem(a))
    assert s["obs"] == mem["obs"] and s["uns"] == mem["uns"]


def test_dense_on_disk_matrix():
    with h5py.File("/tmp/dense_structure_test.h5", "w", driver="core", backing_store=False) as f:
        f.create_dataset("X", data=np.arange(2000 * 3, dtype=np.float32).reshape(2000, 3))
        m = st.matrix_h5(f["X"])
    assert m["shape"] == [2000, 3] and m["format"] == "dense"
    assert m["sample"]["n_sampled"] <= st.SAMPLE_ROWS * 3 and m["sample"]["integer"]


class _Mem:
    is_large = False
    h5ad_path = "/nonexistent.h5ad"

    def __init__(self, adata):
        self.adata = adata


# --- endpoint -----------------------------------------------------------------------------


async def test_endpoint(client, dataset_manager):
    d = Path(dataset_manager.data_dir) / "uploads" / "ins"
    d.mkdir(parents=True)
    (d / "metadata.json").write_text(json.dumps({"id": "ins", "name": "ins", "status": "ready"}))
    _adata().write_h5ad(d / "data.h5ad")
    r = await client.get(f"{API}/datasets/ins/structure")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n_obs"] == 300 and body["file_bytes"] > 0 and body["layers"]["counts"]["looks_like_counts"]
    assert (await client.get(f"{API}/datasets/none/structure")).status_code == 404


async def test_large_dataset_reads_headers_and_the_counts_sidecar(client, tmp_data_dir):
    register(tmp_data_dir, "bigS", counts_adata())
    r = await client.get(f"{API}/datasets/bigS/structure")
    assert r.status_code == 200, r.text
    s = r.json()
    assert s["mode"] == "large" and s["X"]["on_disk"] and not s["X"]["looks_like_counts"]
    assert s["sidecars"]["counts"]["looks_like_counts"]
    assert s["sidecars"]["counts"]["shape"] == [s["n_obs"], s["n_vars"]]
    assert not any("no .raw" in f["text"] for f in s["flags"])
