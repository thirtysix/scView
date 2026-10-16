"""ORA with custom gene-set collections, alone or next to MSigDB.

gseapy runs offline against small collections, as in test_enrichment_endpoints:
T cells up-regulate the first ten genes, B cells the next ten.
"""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scanpy as sc

from scview.config import Settings
from scview.core.pipeline import PipelineParams, _run_enrichment
from scview.core.uns_keys import enrichment_key, markers_key

API = "/api/v1"


def _genes(style: str = "human") -> list[str]:
    if style == "mouse":
        return [f"Tg{i}" for i in range(60)]
    genes = [f"G{i}" for i in range(60)]
    genes[9] = "C1orf9"  # a mixed-case human symbol among the T genes
    return genes


@pytest.fixture()
def msigdb_dir(tmp_path: Path) -> Path:
    d = tmp_path / "msigdb"
    d.mkdir()
    g = _genes()
    (d / "h.all.v2026.1.Hs.json").write_text(json.dumps({
        "HALLMARK_T_PROGRAM": {"geneSymbols": g[0:10]},
        "HALLMARK_B_PROGRAM": {"geneSymbols": g[10:20]},
    }))
    return d


@pytest.fixture()
def test_settings(tmp_data_dir: Path) -> Settings:
    """No MSigDB: custom collections must work offline on their own."""
    return Settings(DATA_DIR=str(tmp_data_dir), DEEPINFRA_API_KEY="", RAG_DATABASE_URL="",
                    MSIGDB_DIR="")


def _adata(style: str = "human") -> ad.AnnData:
    genes = _genes(style)
    rng = np.random.default_rng(0)
    X = rng.poisson(1.0, (90, len(genes))).astype("float32")
    X[:30, 0:10] += 8
    X[30:60, 10:20] += 8
    a = ad.AnnData(X)
    a.var_names = genes
    a.obs_names = [f"C{i}" for i in range(90)]
    a.obs["cell_type"] = pd.Categorical(["T"] * 30 + ["B"] * 30 + ["NK"] * 30)
    sc.tl.rank_genes_groups(a, "cell_type", method="wilcoxon", key_added=markers_key("cell_type"))
    return a


def _register(dm, did: str = "enr", style: str = "human") -> str:
    d = Path(dm.data_dir) / "uploads" / did
    d.mkdir(parents=True)
    (d / "metadata.json").write_text(json.dumps({"id": did, "name": did, "status": "ready"}))
    _adata(style).write_h5ad(d / "data.h5ad")
    return did


def _gmt(style: str = "human", extra: str = "") -> str:
    # Written in the "wrong" case on purpose: matching must ignore it.
    g = [x.lower() for x in _genes(style)]
    return (
        "T_SIG\tmy T cells\t" + "\t".join(g[0:10]) + "\n"
        "B_SIG\t\t" + "\t".join(g[10:20]) + "\n" + extra
    )


async def _upload(client, text: str, name: str = "Lab") -> str:
    r = await client.post(
        f"{API}/genesets/custom/upload",
        files={"file": ("lab.gmt", text.encode(), "text/plain")},
        data={"name": name},
    )
    assert r.status_code == 201, r.text
    return r.json()["id"]


def _local(collections: list[str], **kw) -> dict:
    return {"column": "cell_type", "group": "T", "n_genes": 10, "collections": collections, **kw}


async def _compute(client, did: str, body: dict):
    return await client.post(f"{API}/datasets/{did}/enrichment/compute-local", json=body)


async def test_custom_only_enrichment_works_without_msigdb(client, dataset_manager):
    did = _register(dataset_manager)
    cid = await _upload(client, _gmt())
    r = await _compute(client, did, _local([cid]))
    assert r.status_code == 200, r.text
    top = r.json()["results"][0]
    assert top["term"] == "T_SIG"
    assert top["collection"] == cid and top["collection_name"] == "Lab"
    assert top["collection_deleted"] is False
    assert top["adjusted_pvalue"] < 0.01
    # The mixed-case human symbol counts: all ten T genes overlap, in the
    # dataset's own spelling.
    assert top["overlap_count"] == 10
    assert "C1orf9" in top["genes"]


async def test_msigdb_only_without_msigdb_is_still_500(client, dataset_manager):
    did = _register(dataset_manager)
    r = await _compute(client, did, _local(["h.all"]))
    assert r.status_code == 500 and "MSigDB" in r.json()["detail"]


async def test_mouse_dataset_matches_upper_case_custom_sets(client, dataset_manager):
    did = _register(dataset_manager, style="mouse")
    cid = await _upload(client, _gmt("mouse"))
    r = await _compute(client, did, _local([cid]))
    assert r.status_code == 200, r.text
    top = r.json()["results"][0]
    assert top["term"] == "T_SIG" and top["overlap_count"] == 10


async def test_custom_and_msigdb_together(client, dataset_manager, test_settings, msigdb_dir):
    test_settings.MSIGDB_DIR = str(msigdb_dir)
    did = _register(dataset_manager)
    # A custom set reusing an MSigDB name is kept beside it, not merged into it.
    cid = await _upload(client, _gmt(extra="HALLMARK_T_PROGRAM\t\tg0\tg1\tg2\n"))
    r = await _compute(client, did, _local(["h.all", cid]))
    assert r.status_code == 200, r.text
    by_term = {x["term"]: x for x in r.json()["results"]}
    assert by_term["HALLMARK_T_PROGRAM"]["collection"] == "h.all"
    assert by_term["HALLMARK_T_PROGRAM"]["collection_name"] == ""
    assert by_term["HALLMARK_T_PROGRAM"]["overlap_count"] == 10
    assert by_term["HALLMARK_T_PROGRAM (Lab)"]["collection"] == cid
    assert by_term["HALLMARK_T_PROGRAM (Lab)"]["overlap_count"] == 3
    assert by_term["T_SIG"]["collection"] == cid
    assert by_term["T_SIG"]["overlap_count"] == 10


async def test_cached_result_follows_rename_and_delete(client, dataset_manager):
    did = _register(dataset_manager)
    cid = await _upload(client, _gmt())
    first = await _compute(client, did, _local([cid]))
    assert first.json()["source"] == "msigdb_local"

    await client.patch(f"{API}/genesets/custom/{cid}", json={"name": "Renamed"})
    again = (await _compute(client, did, _local([cid]))).json()
    assert again["source"] == "precomputed"
    assert again["results"][0]["collection_name"] == "Renamed"

    assert (await client.delete(f"{API}/genesets/custom/{cid}")).status_code == 204
    # The cached result is still served, and says its collection is gone ...
    gone = (await _compute(client, did, _local([cid]))).json()
    assert gone["source"] == "precomputed"
    assert gone["results"][0]["term"] == "T_SIG"
    assert gone["results"][0]["collection_name"] == "Lab (deleted)"
    assert gone["results"][0]["collection_deleted"] is True
    # ... but computing anything new with it is refused with a reason.
    r = await _compute(client, did, _local([cid], n_genes=20))
    assert r.status_code == 400
    assert "no longer exists" in r.json()["detail"] and cid in r.json()["detail"]


async def test_a_custom_result_is_not_served_for_another_collection(client, dataset_manager):
    did = _register(dataset_manager)
    a = await _upload(client, _gmt(), name="A")
    b = await _upload(client, "OTHER\t\tg0\tg1\tg30\n", name="B")
    await _compute(client, did, _local([a]))
    r = (await _compute(client, did, _local([b]))).json()
    assert r["source"] == "msigdb_local"
    assert {x["collection"] for x in r["results"]} == {b}


async def test_a_collection_sharing_no_genes_gives_no_results(client, dataset_manager):
    did = _register(dataset_manager)
    cid = await _upload(client, "ELSEWHERE\t\tg40\tg41\n")
    r = await _compute(client, did, _local([cid]))
    assert r.status_code == 200, r.text
    assert r.json()["results"] == []


async def test_network_accepts_custom_collections(client, dataset_manager):
    did = _register(dataset_manager)
    cid = await _upload(client, _gmt())
    r = await client.post(f"{API}/datasets/{did}/enrichment/network",
                          json={**_local([cid]), "fdr": 1.0, "min_term_size": 1})
    assert r.status_code == 200, r.text
    assert {n["collection"] for n in r.json()["nodes"]} == {cid}


def test_pipeline_enrichment_step_uses_custom_collections(tmp_data_dir, monkeypatch):
    from scview.core.custom_genesets import CustomGeneSetStore, parse_gene_set_file

    store = CustomGeneSetStore(tmp_data_dir)
    meta = store.create("Lab", parse_gene_set_file(_gmt().encode(), "lab.gmt"))
    monkeypatch.setenv("MSIGDB_DIR", "")
    monkeypatch.setattr(
        "scview.config.get_settings", lambda: Settings(DATA_DIR=str(tmp_data_dir))
    )
    a = _adata()
    _run_enrichment(a, PipelineParams(
        enrichment_columns=["cell_type"], enrichment_collections=[meta["id"]],
        enrichment_n_genes=10,
    ))
    results = json.loads(a.uns[enrichment_key("cell_type", "T")])
    assert results[0]["term"] == "T_SIG"
    assert results[0]["collection"] == meta["id"]
    assert results[0]["overlap_count"] == 10
