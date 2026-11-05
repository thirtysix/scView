"""Gene symbols as gene names for a CELLxGENE .h5ad (Ensembl ids, symbols in
var['feature_name']) uploaded directly (core/gene_symbols.py, POST /datasets/upload).
Seen on Wilk 2020 uploaded as is (2026-11-05): gene search for "XIST" found nothing.
Expected names are written out by hand from the fixture, not from the code."""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from scview.core import gene_symbols, provenance, replay_script

API = "/api/v1"
IDS = [f"ENSG{i:011d}" for i in range(6)]
SYMBOLS = ["XIST", "CD3E", "", "MT-CO1", "CD3E", "RPS4Y1"]
WANT = ["XIST", "CD3E", IDS[2], "MT-CO1", "CD3E-1", "RPS4Y1"]


def _cxg(raw: bool = True) -> ad.AnnData:
    rng = np.random.default_rng(0)
    X = sp.csr_matrix(rng.poisson(2, (30, 6)).astype(np.float32))
    a = ad.AnnData(X=X, obs=pd.DataFrame(index=[f"c{i}" for i in range(30)]),
                   var=pd.DataFrame({"feature_name": SYMBOLS}, index=IDS))
    if raw:
        a.raw = a
    return a


def test_to_symbols_keeps_ids_fills_blanks_and_makes_repeats_unique():
    var, summary = gene_symbols.to_symbols(_cxg().var)
    assert list(var.index) == WANT and list(var["gene_ids"]) == IDS
    assert summary["kept_id"] == 1 and summary["made_unique"] == 1 and summary["n_genes"] == 6
    already = _cxg().var.assign(gene_ids=IDS)
    assert gene_symbols.to_symbols(already) is None  # ids already kept: left alone
    assert gene_symbols.to_symbols(pd.DataFrame(index=["A", "B"])) is None


async def test_upload_of_a_cellxgene_file_gets_symbols_and_keeps_the_original(
        client, dataset_manager, tmp_data_dir, tmp_path):
    src = tmp_path / "wilk_like.h5ad"
    _cxg().write_h5ad(src)
    sent = src.read_bytes()
    r = await client.post(f"{API}/datasets/upload", files={"file": ("wilk_like.h5ad", sent, "x")})
    assert r.status_code == 201, r.text
    did = r.json()["id"]
    assert (tmp_data_dir / "uploads" / did / "wilk_like.h5ad").read_bytes() == sent
    working = tmp_data_dir / "ingested" / did / "wilk_like.h5ad"
    assert dataset_manager.h5ad_candidates(did)[0] == str(working)
    meta = json.loads((tmp_data_dir / "uploads" / did / "metadata.json").read_text())
    assert meta["gene_names"]["var"]["renamed"] == 5 and "raw_var" in meta["gene_names"]
    adata = (await dataset_manager.get_or_load_dataset(did)).adata
    assert list(adata.var_names) == WANT and list(adata.raw.var_names) == WANT
    assert list(adata.var["gene_ids"]) == IDS
    steps = [s for s in provenance.read_provenance(adata)["history"] if s["step"] == "ingest"]
    assert steps and steps[-1]["params"]["gene_symbols"]["var"]["made_unique"] == 1
    hits = (await client.get(f"{API}/datasets/{did}/genes/search", params={"q": "XIST"})).json()
    assert "XIST" in json.dumps(hits)


async def test_upload_with_symbols_already_is_unchanged(client, tmp_data_dir, tmp_path):
    src = tmp_path / "plain.h5ad"
    a = _cxg(raw=False)
    a.var_names = ["A", "B", "C", "D", "E", "F"]
    a.var = a.var.drop(columns=["feature_name"])
    a.write_h5ad(src)
    r = await client.post(f"{API}/datasets/upload", files={"file": ("plain.h5ad", src.read_bytes(), "x")})
    assert r.status_code == 201, r.text
    did = r.json()["id"]
    assert not (tmp_data_dir / "ingested" / did).exists()
    assert "gene_names" not in json.loads((tmp_data_dir / "uploads" / did / "metadata.json").read_text())


def test_replay_repeats_the_rename_and_is_a_no_op_on_the_stored_copy():
    snip = replay_script._ingest({"formats": ["h5ad"], "gene_symbols": {"var": {}}}, None)
    code = "\n".join(snip.code)
    ns = {"adata": _cxg(), "ad": ad, "pd": pd}
    exec(code, ns)  # noqa: S102 - the script's own lines
    assert list(ns["adata"].var_names) == WANT and list(ns["adata"].raw.var_names) == WANT
    again = {"adata": ns["adata"], "ad": ad, "pd": pd}
    exec(code, again)  # noqa: S102
    assert list(again["adata"].var_names) == WANT
    plain = replay_script._ingest({"formats": ["h5ad"]}, None)
    assert "feature_name" not in "\n".join(plain.code)


def test_file_rewrite_of_var_and_raw(tmp_path: Path):
    p = tmp_path / "x.h5ad"
    _cxg().write_h5ad(p)
    assert gene_symbols.file_applies(p)
    summary = gene_symbols.rewrite_file(p)
    assert set(summary) == {"var", "raw_var"} and not gene_symbols.file_applies(p)
    b = ad.read_h5ad(p)
    assert list(b.var_names) == WANT and list(b.raw.var_names) == WANT
