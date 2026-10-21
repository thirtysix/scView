"""Gene programs endpoints (api/v1/programs.py): run, save, read, usage, enrichment."""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pytest
from scipy.stats import hypergeom

from scview.config import Settings
from scview.core import gene_programs as gp
from scview.core import provenance
from tests.test_gene_programs import PER_PROG, _planted
from tests.test_hvg_sensitivity import _events, _register

API = "/api/v1"


@pytest.fixture()
def msigdb_dir(tmp_path: Path) -> Path:
    d = tmp_path / "msigdb"
    d.mkdir()
    (d / "h.all.v2026.1.Hs.json").write_text(json.dumps({
        "HALLMARK_FIRST": {"geneSymbols": [f"G{i}" for i in range(PER_PROG)]},
        "HALLMARK_OTHER": {"geneSymbols": [f"G{i}" for i in range(200, 240)]},
    }))
    return d


@pytest.fixture()
def test_settings(tmp_data_dir: Path, msigdb_dir: Path) -> Settings:
    return Settings(DATA_DIR=str(tmp_data_dir), DEEPINFRA_API_KEY="", RAG_DATABASE_URL="",
                    MSIGDB_DIR=str(msigdb_dir))


async def _run(client, did, **body):
    r = await client.post(f"{API}/datasets/{did}/programs/run-stream", json=body)
    assert r.status_code == 200, r.text
    return _events(r.text)


async def test_run_saves_programs_and_they_are_served(client, dataset_manager):
    a, usage, sets = _planted()
    did = _register(dataset_manager, a, "pg")
    info = (await client.get(f"{API}/datasets/{did}/programs")).json()
    assert info["available"] is False and info["runnable"]["ok"] is True
    assert info["runnable"]["n_flagged"] == 175 and info["defaults"]["k"] == gp.DEFAULT_K
    assert info["running"] is False

    events = await _run(client, did, k=3, seeds=3, sweep=[2, 4])
    kinds = [e for e, _ in events]
    assert kinds[0] == "start" and kinds[-2:] == ["result", "complete"]
    assert kinds.count("progress") == 9 and kinds.count("k_result") == 3
    start = events[0][1]
    assert start["n_genes"] == 175 and start["source"] == "log-normalised .raw" and start["ks"] == [2, 4, 3]
    result = events[-2][1]
    assert result["k"] == 3 and [s["k"] for s in result["sweep"]] == [2, 3, 4]
    assert result["resources"]["peak_gb"] is not None or result["resources"]["n_samples"] >= 1

    # Saved: the dataset reopens with the programs, not as an embedding.
    dataset_manager.evict(did)
    ds = (await client.get(f"{API}/datasets/{did}")).json()
    assert gp.OBSM_KEY not in ds["available_embeddings"]
    adata = (await dataset_manager.get_or_load_dataset(did)).adata
    assert adata.obsm[gp.OBSM_KEY].shape == (a.n_obs, 3)
    assert provenance.read_provenance(adata)["history"][-1]["step"] == "gene_programs"

    info = (await client.get(f"{API}/datasets/{did}/programs?groupby=group")).json()
    assert info["available"] is True and len(info["programs"]) == 3
    assert info["by_group"]["groups"] == ["a", "b", "c"] and len(info["by_group"]["mean"]) == 3
    # Each planted program is one saved program, and its group has the highest mean.
    for p in range(3):
        best = max(info["programs"], key=lambda q: len(set(sets[p]) & {g["gene"] for g in q["top_genes"]}))
        j = best["index"] - 1
        row = info["by_group"]["mean"][j]
        assert info["by_group"]["groups"][int(np.argmax(row))] == "abc"[p]

    r = await client.get(f"{API}/datasets/{did}/programs/1/usage?groupby=group")
    assert r.status_code == 200
    u = r.json()
    assert u["name"] == "P1" and len(u["values"]) == a.n_obs and sorted(u["violin"]) == ["a", "b", "c"]
    assert np.allclose(u["values"], adata.obsm[gp.OBSM_KEY][:, 0], atol=1e-4)
    raw = (await client.get(f"{API}/datasets/{did}/programs/1/usage?raw=true")).json()
    assert np.allclose(raw["values"], adata.obsm[gp.OBSM_RAW_KEY][:, 0], rtol=1e-4, atol=1e-4)
    assert (await client.get(f"{API}/datasets/{did}/programs/9/usage")).status_code == 404


async def test_enrichment_tests_top_genes_against_the_used_genes(client, dataset_manager):
    a, _, sets = _planted()
    did = _register(dataset_manager, a, "pe")
    await _run(client, did, k=3, seeds=2)
    info = (await client.get(f"{API}/datasets/{did}/programs")).json()
    first = max(info["programs"], key=lambda q: len(set(sets[0]) & {g["gene"] for g in q["top_genes"]}))
    r = await client.post(f"{API}/datasets/{did}/programs/{first['index']}/enrichment", json={})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["background_n"] == 175 and body["collections"] == ["h.all"]
    hit = next(t for t in body["results"] if t["term"] == "HALLMARK_FIRST")
    # The hypergeometric p by hand: universe = the 175 genes NMF used; the set's
    # members in it: all 25 of G0..G24; query = the program's top genes.
    query = set(body["query"])
    overlap = len(query & set(sets[0]))
    p = hypergeom.sf(overlap - 1, 175, PER_PROG, len(query))
    assert hit["overlap_count"] == overlap and hit["pvalue"] == pytest.approx(p, rel=1e-6)
    # Kept with the run, for the summary and the co-pilot.
    info = (await client.get(f"{API}/datasets/{did}/programs")).json()
    prog = next(q for q in info["programs"] if q["index"] == first["index"])
    assert prog["top_term"]["term"] == "HALLMARK_FIRST"
    from scview.core.assistant import build_grounding_context

    context, sources = build_grounding_context(await dataset_manager.get_or_load_dataset(did), {"markers"})
    assert "Gene programs (consensus NMF)" in context and "HALLMARK_FIRST" in context
    assert any(s.ref == gp.TAG for s in sources)
    from scview.core.claims import resolve_tag

    assert resolve_tag("programs", [gp.TAG, "result:pca"]) == [gp.TAG]
    assert (await client.post(f"{API}/datasets/{did}/programs/7/enrichment", json={})).status_code == 404


async def test_refusals(client, dataset_manager):
    a, _, _ = _planted(layout="scaled")
    did = _register(dataset_manager, a, "ps")
    info = (await client.get(f"{API}/datasets/{did}/programs")).json()
    assert info["runnable"]["ok"] is False and "scaled" in info["runnable"]["reason"]
    r = await client.post(f"{API}/datasets/{did}/programs/run-stream", json={"k": 3})
    assert r.status_code == 400 and "scaled" in r.json()["detail"]
    good = _register(dataset_manager, _planted()[0], "pg2")
    for body in ({"k": 1}, {"seeds": 0}, {"sweep": [99]}, {"n_hvg": 3}):
        r = await client.post(f"{API}/datasets/{good}/programs/run-stream", json=body)
        assert r.status_code in (400, 422), body
    assert (await client.get(f"{API}/datasets/{good}/programs/1/usage")).status_code == 404
    assert (await client.post(f"{API}/datasets/{good}/programs/1/enrichment", json={})).status_code == 404
    assert (await client.get(f"{API}/datasets/nope/programs")).status_code == 404


async def test_a_dataset_whose_cells_changed_is_not_saved(client, dataset_manager, monkeypatch):
    a, _, _ = _planted()
    did = _register(dataset_manager, a, "pc")
    from scview.core.anndata_adaptor import AnnDataAdaptor

    calls = {"n": 0}
    real = AnnDataAdaptor.cells_fingerprint

    def changing(self):
        calls["n"] += 1
        return real(self) if calls["n"] == 1 else "other cells"

    monkeypatch.setattr(AnnDataAdaptor, "cells_fingerprint", changing)
    events = await _run(client, did, k=3, seeds=1)
    assert events[-1][0] == "error" and "nothing was saved" in events[-1][1]["error"]
    monkeypatch.undo()
    dataset_manager.evict(did)
    assert gp.OBSM_KEY not in (await dataset_manager.get_or_load_dataset(did)).adata.obsm


async def test_counts_only_and_dense_dataset_run(client, dataset_manager):
    a, _, _ = _planted(layout="counts", dense=True)
    did = _register(dataset_manager, a, "pd")
    events = await _run(client, did, k=3, seeds=2)
    assert events[0][1]["source"].startswith("shifted log of the counts in X")
    assert events[-1][0] == "complete"
    saved = ad.read_h5ad(next((Path(dataset_manager.data_dir) / "derived" / did).glob("*.h5ad")))
    assert saved.obsm[gp.OBSM_KEY].shape == (a.n_obs, 3)
