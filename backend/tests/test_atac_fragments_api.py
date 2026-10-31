"""ATAC fragments endpoints (api/v1/atac_fragments.py): streamed attach in either order, the QC
run + save, on a dataset in memory and a large one, coverage."""

from __future__ import annotations

import gzip
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from scview.core import atac_fragments as af
from scview.core import gene_annotation, modality_store, provenance, results_store
from tests.fragments_fixture import write_fragments
from tests.test_hvg_sensitivity import _events, _register

API = "/api/v1"
N_CELLS = 60
TSS = 50_000
GTF = ('chr1\tX\tgene\t50001\t60000\t.\t+\t.\tgene_id "E1.1"; gene_type "protein_coding"; gene_name "AAA";\n'
       'chr22\tX\tgene\t9001\t9500\t.\t-\t.\tgene_id "E2.1"; gene_type "protein_coding"; gene_name "BBB";\n')


def _cells() -> list[str]:
    return [f"BC{i:04d}-1" for i in range(N_CELLS)]


def _adata() -> ad.AnnData:
    cells = _cells()
    obs = pd.DataFrame({"cell_type": pd.Categorical(["B" if i % 2 else "T" for i in range(N_CELLS)])}, index=cells)
    rng = np.random.default_rng(0)
    return ad.AnnData(X=sp.csr_matrix(rng.poisson(1.0, (N_CELLS, 5)).astype(np.float32)), obs=obs,
                      var=pd.DataFrame(index=[f"GENE{i}" for i in range(5)]))


def _lines() -> list[tuple[str, int, int, str, int]]:
    rng = np.random.default_rng(1)
    cells = _cells()
    out = []
    for i in range(3000):
        cell = cells[i % N_CELLS]
        start = int(TSS - 1200 + rng.integers(0, 2400))
        out.append(("chr1", start, start + int(rng.integers(40, 300)), cell, 1))
    out += [("chr22", 9000 + i, 9050 + i, cells[i % N_CELLS], 1) for i in range(40)]
    return sorted(out, key=lambda x: (x[0], x[1]))


@pytest.fixture()
def seeded(tmp_data_dir):
    gtf = Path(tmp_data_dir) / "seed.gtf"
    gtf.write_text(GTF)
    gene_annotation.clear_cache()
    gene_annotation.seed_from(gtf, tmp_data_dir, "GRCh38")
    yield
    gene_annotation.clear_cache()


@pytest.fixture()
def files(tmp_path) -> tuple[bytes, bytes]:
    frag, idx = write_fragments(tmp_path / "up.tsv.gz", _lines())
    return frag.read_bytes(), idx.read_bytes()


async def _upload(client, did, name, body, **kw):
    return await client.post(f"{API}/datasets/{did}/atac/fragments", params={"name": name}, content=body, **kw)


async def _attach(client, did, files):
    r = await _upload(client, did, "atac_fragments.tsv.gz", files[0])
    assert r.status_code == 200, r.text
    r = await _upload(client, did, "atac_fragments.tsv.gz.tbi", files[1])
    assert r.status_code == 200, r.text
    return r.json()


async def test_attach_in_either_order_says_what_is_missing(client, dataset_manager, files, seeded):
    did = _register(dataset_manager, _adata(), "fa")
    info = (await client.get(f"{API}/datasets/{did}/atac/fragments")).json()
    assert info["attached"] is False and info["qc"] is None and info["missing"] is None
    r = await _upload(client, did, "atac_fragments.tsv.gz.tbi", files[1])
    assert r.status_code == 200 and "fragments file" in r.json()["missing"] and r.json()["genome"] == "GRCh38"
    r = await _upload(client, did, "atac_fragments.tsv.gz", files[0])
    assert r.status_code == 200, r.text
    st = r.json()
    assert st["attached"] and st["missing"] is None and st["barcode_share"] == 1.0 and st["genome"] == "GRCh38"
    assert st["cells_seen"] == N_CELLS and st["name"] == "atac_fragments.tsv.gz"
    did2 = _register(dataset_manager, _adata(), "fb")
    r = await _upload(client, did2, "atac_fragments.tsv.gz", files[0])
    assert r.status_code == 200 and "Cell Ranger ARC" in r.json()["missing"]
    assert (await client.post(f"{API}/datasets/{did2}/atac/fragments/qc-stream")).status_code == 422
    r = await _upload(client, did2, "x.tbi", files[1])
    assert r.json()["attached"] is True and r.json()["estimate"]["seconds"] >= 0
    folder = modality_store.store_dir(dataset_manager.data_dir, did2)
    assert not list(folder.glob(".*"))  # no part file left


async def test_upload_refusals_are_sentences(client, dataset_manager, files):
    did = _register(dataset_manager, _adata(), "fc")
    r = await _upload(client, did, "matrix.h5", b"x")
    assert r.status_code == 422 and "neither" in r.json()["detail"]
    r = await _upload(client, did, "a.tsv.gz", gzip.compress(b"chr1\t1\t60\tBC0000-1\t1\n" * 100))
    assert r.status_code == 422 and "plain gzip" in r.json()["detail"]
    r = await _upload(client, did, "a.tsv.gz", b"")
    assert r.status_code == 422 and "empty" in r.json()["detail"]
    r = await _upload(client, did, "a.tsv.gz.tbi", b"not an index but long enough")
    assert r.status_code == 422 and "tabix" in r.json()["detail"]
    r = await client.post(f"{API}/datasets/{did}/atac/fragments", params={"name": "a.tsv.gz"},
                          files={"file": ("a.tsv.gz", b"x")})
    assert r.status_code == 415
    r = await _upload(client, "nope", "a.tsv.gz", files[0])
    assert r.status_code == 404
    folder = modality_store.store_dir(dataset_manager.data_dir, did)
    assert not folder.exists() or not list(folder.iterdir())


async def test_upload_of_another_sample_is_refused_with_the_numbers(client, dataset_manager, files):
    a = _adata()
    a.obs_names = [f"ZZ{i:04d}-1" for i in range(N_CELLS)]
    did = _register(dataset_manager, a, "fd")
    r = await _upload(client, did, "a.tsv.gz", files[0])
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert "0 of the first 3,040 fragments (0.0%)" in detail and "60 cells" in detail
    assert not af.fragments_path(dataset_manager.data_dir, did).exists()


async def test_qc_runs_saves_and_reopens(client, dataset_manager, files, seeded):
    did = _register(dataset_manager, _adata(), "fe")
    await _attach(client, did, files)
    r = await client.post(f"{API}/datasets/{did}/atac/fragments/qc-stream")
    assert r.status_code == 200, r.text
    events = _events(r.text)
    kinds = [e for e, _ in events]
    assert kinds[0] == "start" and kinds[-2:] == ["result", "complete"], events[-1]
    assert events[0][1]["estimate"]["peak_bytes"] > 0 and events[0][1]["genome"] == "GRCh38"
    res = events[-2][1]
    assert res["n_lines"] == 3040 and res["n_cells"] == N_CELLS and res["n_fragments_in_cells"] == 3040
    assert res["params"]["annotation"] == gene_annotation.source_of("GRCh38") and res["medians"]["tss_enrichment"] > 0
    assert res["resources"]["peak_gb"] is None or res["resources"]["peak_gb"] >= 0

    dataset_manager.evict(did)
    adata = (await dataset_manager.get_or_load_dataset(did)).adata
    assert adata.obs[af.OBS_N_FRAGMENTS].sum() == 3040 and af.OBS_TSS in adata.obs and af.OBS_NUCLEOSOME in adata.obs
    assert provenance.read_provenance(adata)["history"][-1]["step"] == af.STEP
    assert af.STEP in results_store.ADDITIVE_STEPS
    info =(await client.get(f"{API}/datasets/{did}/atac/fragments")).json()
    assert info["qc"]["medians"] == res["medians"] and info["running"] is False

    values = (await client.get(f"{API}/datasets/{did}/atac/fragments/values",
                               params={"group_by": "cell_type"})).json()
    assert [g["name"] for g in values["groups"]] == ["B", "T"] and values["groups"][0]["n"] == 30
    lengths = (await client.get(f"{API}/datasets/{did}/atac/fragments/lengths",
                                params={"group_by": "cell_type"})).json()
    assert sum(g["n_fragments"] for g in lengths["groups"]) == 3040 and len(lengths["x"]) == af.GROUP_LENGTH_BINS


async def test_qc_without_files_is_a_sentence(client, dataset_manager):
    did = _register(dataset_manager, _adata(), "ff")
    r = await client.post(f"{API}/datasets/{did}/atac/fragments/qc-stream")
    assert r.status_code == 404 and "No ATAC fragments" in r.json()["detail"]
    r = await client.get(f"{API}/datasets/{did}/atac/coverage", params={"region": "chr1:1-100"})
    assert r.status_code == 404 and "No ATAC fragments" in r.json()["detail"]
    r = await client.get(f"{API}/datasets/{did}/atac/fragments/values")
    assert r.status_code == 404 and "Run the fragment QC" in r.json()["detail"]


async def test_qc_on_a_large_dataset_writes_in_place(client, dataset_manager, files, seeded, tmp_path):
    from tests import large_fixture

    base = large_fixture.counts_adata(n_cells=N_CELLS, n_genes=40)
    base.obs_names = _cells()
    did = "flarge"
    large_fixture.register(dataset_manager.data_dir, did, base)
    await _attach(client, did, files)
    r = await client.post(f"{API}/datasets/{did}/atac/fragments/qc-stream")
    events = _events(r.text)
    assert events[-1][0] == "complete", events[-1]
    dataset_manager.evict(did)
    adaptor = await dataset_manager.get_or_load_dataset(did)
    assert adaptor.is_large and adaptor.adata.obs[af.OBS_N_FRAGMENTS].sum() == 3040
    assert af.saved(adaptor.adata)["medians"]["tss_enrichment"] is not None


async def test_coverage_by_gene_and_region(client, dataset_manager, files, seeded):
    did = _register(dataset_manager, _adata(), "fg")
    await _attach(client, did, files)
    url = f"{API}/datasets/{did}/atac/coverage"
    r = await client.get(url, params={"gene": "AAA", "group_by": "cell_type", "bins": 20})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["region"] == {"chrom": "chr1", "start": 45000, "end": 65000} and body["gene"]["name"] == "AAA"
    assert [t["name"] for t in body["tracks"]] == ["B", "T"] and len(body["x"]) == 20
    assert [g["name"] for g in body["genes"]] == ["AAA"] and body["peaks"] == [] and "without exons" in body["genes_note"]
    assert "per 1,000 cells" in body["normalisation"]  # the QC has not run
    r = await client.get(url, params={"region": "chr1:49,000-51,000", "group_by": "cell_type",
                                      "groups": ["T"], "bins": 10})
    assert [t["name"] for t in r.json()["tracks"]] == ["T"] and r.json()["n_fragments"] > 0
    for params, status, word in (({"gene": "NOPE"}, 404, "no gene"), ({"region": "banana"}, 422, "chr1:1000-5000"),
                                 ({"region": "chrZ:1-100"}, 404, "no chromosome"), ({}, 422, "gene or a region"),
                                 ({"region": "chr1:1-3000000"}, 422, "up to 2 Mb"),
                                 ({"region": "chr1:1-100", "group_by": "nope"}, 404, "no column")):
        r = await client.get(url, params=params)
        assert r.status_code == status and word in r.json()["detail"], params


async def test_delete_removes_the_files(client, dataset_manager, files, seeded):
    did = _register(dataset_manager, _adata(), "fh")
    await _attach(client, did, files)
    assert (await client.delete(f"{API}/datasets/{did}/atac/fragments")).json() == {"deleted": True}
    assert (await client.get(f"{API}/datasets/{did}/atac/fragments")).json()["attached"] is False
