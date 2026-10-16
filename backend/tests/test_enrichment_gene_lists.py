"""ORA on explicit gene lists: a DE result's up and down genes, tested genes as background.

``POST /datasets/{id}/enrichment/gene-lists`` runs gseapy offline against a
two-term MSigDB collection in a temp directory (as test_enrichment_endpoints),
so the hypergeometric p-values can be checked against scipy for the universe
the request names.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import anndata as ad
import numpy as np
import pytest
from scipy.stats import hypergeom

from scview.config import Settings

API = "/api/v1"
GENES = [f"G{i}" for i in range(60)] + ["RPL3", "RPS6", "MT-CO1"]
T_SET = GENES[0:10]
B_SET = GENES[10:20]


@pytest.fixture()
def msigdb_dir(tmp_path: Path) -> Path:
    d = tmp_path / "msigdb"
    d.mkdir()
    (d / "h.all.v2026.1.Hs.json").write_text(json.dumps({
        "HALLMARK_T_PROGRAM": {"geneSymbols": T_SET},
        "HALLMARK_B_PROGRAM": {"geneSymbols": B_SET},
    }))
    return d


@pytest.fixture()
def test_settings(tmp_data_dir: Path, msigdb_dir: Path) -> Settings:
    return Settings(DATA_DIR=str(tmp_data_dir), DEEPINFRA_API_KEY="", RAG_DATABASE_URL="",
                    MSIGDB_DIR=str(msigdb_dir))


def _register(dm, did: str = "gl") -> str:
    d = Path(dm.data_dir) / "uploads" / did
    d.mkdir(parents=True)
    (d / "metadata.json").write_text(json.dumps({"id": did, "name": did, "status": "ready"}))
    a = ad.AnnData(np.random.default_rng(0).poisson(1.0, (30, len(GENES))).astype("float32"))
    a.var_names = GENES
    a.obs_names = [f"C{i}" for i in range(30)]
    a.write_h5ad(d / "data.h5ad")
    return did


def _body(**kw) -> dict:
    return {"up": T_SET[:6], "down": B_SET[:6], "background": GENES,
            "collections": ["h.all"], **kw}


def _terms(side: dict) -> dict[str, dict]:
    return {r["term"]: r for r in side["results"]}


async def test_up_and_down_are_enriched_separately(client, dataset_manager):
    did = _register(dataset_manager)
    r = await client.post(f"{API}/datasets/{did}/enrichment/gene-lists", json=_body())
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["background_n"] == len(GENES)
    up, down = _terms(body["up"]), _terms(body["down"])
    # Each list finds its own program and not the other's.
    assert set(up) == {"HALLMARK_T_PROGRAM"}
    assert set(down) == {"HALLMARK_B_PROGRAM"}
    assert up["HALLMARK_T_PROGRAM"]["overlap_count"] == 6
    assert up["HALLMARK_T_PROGRAM"]["collection"] == "h.all"
    assert body["up"]["n_genes_used"] == body["down"]["n_genes_used"] == 6


async def test_the_background_is_the_universe_of_the_test(client, dataset_manager):
    # The same 6 T genes over two universes: p must be the hypergeometric tail
    # for each universe's size, so the request's background is what gseapy used.
    did = _register(dataset_manager)
    for universe in (GENES, GENES[:30]):
        r = await client.post(f"{API}/datasets/{did}/enrichment/gene-lists",
                              json=_body(background=universe, down=[]))
        assert r.status_code == 200, r.text
        assert r.json()["background_n"] == len(universe)
        hit = _terms(r.json()["up"])["HALLMARK_T_PROGRAM"]
        expected = hypergeom.sf(6 - 1, len(universe), len(T_SET), 6)
        assert hit["pvalue"] == pytest.approx(expected, rel=1e-6)


async def test_genes_outside_the_dataset_or_background_are_dropped(client, dataset_manager):
    did = _register(dataset_manager)
    r = await client.post(
        f"{API}/datasets/{did}/enrichment/gene-lists",
        json=_body(
            up=T_SET[:6] + ["NOT_A_GENE", "G50"],  # G50 is measured but not in the background
            background=GENES[:40] + ["NOT_A_GENE", "ALSO_NOT"],
        ),
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["background_n"] == 40
    assert body["up"]["n_genes_requested"] == 8
    assert body["up"]["n_genes_used"] == 6


async def test_ribo_mito_genes_leave_the_query_only_when_asked(client, dataset_manager):
    did = _register(dataset_manager)
    up = T_SET[:6] + ["RPL3", "RPS6", "MT-CO1"]
    on = (await client.post(f"{API}/datasets/{did}/enrichment/gene-lists",
                            json=_body(up=up))).json()
    off = (await client.post(f"{API}/datasets/{did}/enrichment/gene-lists",
                             json=_body(up=up, exclude_ribo_mito=False))).json()
    assert on["up"]["n_excluded_ribo_mito"] == 3 and on["up"]["n_genes_used"] == 6
    assert off["up"]["n_excluded_ribo_mito"] == 0 and off["up"]["n_genes_used"] == 9
    # The background keeps them either way: they were measured.
    assert on["background_n"] == off["background_n"] == len(GENES)


async def test_an_empty_side_has_no_results_and_no_error(client, dataset_manager):
    did = _register(dataset_manager)
    r = await client.post(f"{API}/datasets/{did}/enrichment/gene-lists", json=_body(down=[]))
    assert r.status_code == 200, r.text
    assert r.json()["down"] == {
        "n_genes_requested": 0, "n_genes_used": 0, "n_excluded_ribo_mito": 0, "n_terms": 0,
        "n_significant": 0, "results": [],
    }


async def test_only_the_most_significant_terms_are_returned(client, dataset_manager):
    # Both programs overlap this list; the T program (6 of 10) beats B (1 of 10).
    did = _register(dataset_manager)
    r = await client.post(f"{API}/datasets/{did}/enrichment/gene-lists",
                          json=_body(up=T_SET[:6] + B_SET[:1], max_terms=1))
    assert r.status_code == 200, r.text
    up = r.json()["up"]
    assert up["n_terms"] == 2
    assert [t["term"] for t in up["results"]] == ["HALLMARK_T_PROGRAM"]
    # Counted over every term, not only the ones returned.
    assert up["n_significant"] == 1


async def test_a_background_the_dataset_does_not_measure_is_400(client, dataset_manager):
    did = _register(dataset_manager)
    r = await client.post(f"{API}/datasets/{did}/enrichment/gene-lists",
                          json=_body(background=["X1", "X2"]))
    assert r.status_code == 400
    assert "background" in r.json()["detail"]


async def test_an_empty_background_is_rejected(client, dataset_manager):
    did = _register(dataset_manager)
    r = await client.post(f"{API}/datasets/{did}/enrichment/gene-lists", json=_body(background=[]))
    assert r.status_code == 422


async def test_a_deleted_custom_collection_is_400(client, dataset_manager):
    did = _register(dataset_manager)
    r = await client.post(f"{API}/datasets/{did}/enrichment/gene-lists",
                          json=_body(collections=["custom-0123456789abcdef"]))
    assert r.status_code == 400
    assert "no longer exists" in r.json()["detail"]


async def test_missing_dataset_is_404(client, dataset_manager):
    r = await client.post(f"{API}/datasets/ghost/enrichment/gene-lists", json=_body())
    assert r.status_code == 404


async def test_gene_list_enrichment_runs_in_a_worker_thread(client, dataset_manager, monkeypatch):
    from scview.api.v1 import enrichment

    did = _register(dataset_manager)
    threads: list[threading.Thread] = []
    real = enrichment._ora_records

    def spy(*args, **kwargs):
        threads.append(threading.current_thread())
        return real(*args, **kwargs)

    monkeypatch.setattr(enrichment, "_ora_records", spy)
    r = await client.post(f"{API}/datasets/{did}/enrichment/gene-lists", json=_body())
    assert r.status_code == 200, r.text
    assert len(threads) == 2  # one run per side
    assert all(t is not threading.main_thread() for t in threads)
