"""CELLxGENE Discover import (core/public_import.py, api/v1/public_import.py).

HTTP is mocked throughout (httpx.MockTransport). The listing fixtures are real
curation API responses recorded 2026-10-24 (``GET /curation/v1/datasets``, four
records; ``GET /curation/v1/datasets/0895c838-.../versions``), with the long
per-record lists (cell types, stages, donors, ethnicities) cut to three items.
The h5ad is synthetic but laid out as a Discover schema 7.1.0 file: normalised X,
counts in raw.X, Ensembl index with categorical ``feature_name`` symbols, curated
obs columns after the author's."""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from scview.config import Settings
from scview.core import provenance, public_import as pi, replay_script
from scview.core.assistant import _methods_skeleton, data_citation

API = "/api/v1"
FIX = Path(__file__).parent / "fixtures" / "discover"
LISTING = json.loads((FIX / "datasets_2026-10-24.json").read_text())
VERSIONS = json.loads((FIX / "versions_2026-10-24.json").read_text())
LIVER_B = "0d5aa69f-1ad1-470f-9a1b-4a167da3b545"  # Healthy human liver: B cells, 146 cells
LIVER_B_DATASET = "0895c838-e550-48a3-a777-dbcd35d30272"

IDS = ["ENSG00000000003", "ENSG00000198888", "ENSG00000198763", "ENSG00000000005",
       "ENSG00000111111", "ENSG00000222222", "ENSG00000333333", "ENSG00000444444"]
SYMBOLS = ["TSPAN6", "MT-ND1", "MT-ND2", "TNMD", "DUP", "DUP", "CD19", "ENSG00000444444"]


def discover_h5ad(path: Path, n: int = 40, seed: int = 0) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    counts = rng.poisson(2.0, size=(n, len(IDS))).astype(np.float32)
    norm = np.log1p(counts / counts.sum(1, keepdims=True) * 1e4).astype(np.float32)
    var = pd.DataFrame({"feature_name": pd.Categorical(SYMBOLS),
                        "feature_is_filtered": False,
                        "feature_type": "protein_coding"}, index=IDS)
    obs = pd.DataFrame({
        "donor_id": pd.Categorical(["d1", "d2"] * (n // 2)),
        "cell_type_ontology_term_id": pd.Categorical(["CL:0000236", "CL:0000786"] * (n // 2)),
        "author_cell_type": pd.Categorical(["B", "Plasma"] * (n // 2)),
        "cell_type": pd.Categorical(["B cell", "plasma cell"] * (n // 2)),
        "tissue": pd.Categorical(["liver"] * n),
        "disease": pd.Categorical(["normal"] * n),
    }, index=[f"cell{i}" for i in range(n)])
    raw = ad.AnnData(X=sp.csr_matrix(counts), obs=obs.copy(), var=var.drop(columns="feature_is_filtered"))
    a = ad.AnnData(X=sp.csr_matrix(norm), obs=obs, var=var)
    a.raw = raw
    a.obsm["X_umap"] = rng.normal(size=(n, 2))
    a.uns["citation"] = LISTING_REC()["citation"]
    a.uns["schema_version"] = "7.1.0"
    a.uns["title"] = "Healthy human liver: B cells"
    a.write_h5ad(path)
    return a


def LISTING_REC() -> dict:
    return next(r for r in LISTING if r["dataset_version_id"] == LIVER_B)


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


class FakeDiscover:
    """Routes the two hosts; counts the listing requests."""

    def __init__(self, body: bytes = b"", fail_listing: bool = False, status: int = 200):
        self.body = body
        self.fail_listing = fail_listing
        self.status = status
        self.listing_calls = 0
        self.urls: list[str] = []

    def handler(self, request):
        import httpx

        url = str(request.url)
        self.urls.append(url)
        if url == f"{pi.API_BASE}/datasets":
            self.listing_calls += 1
            if self.fail_listing:
                raise httpx.ConnectError("offline")
            return httpx.Response(200, json=LISTING)
        if url == f"{pi.API_BASE}/datasets/{LIVER_B_DATASET}/versions":
            return httpx.Response(200, json=VERSIONS)
        if url.startswith(pi.FILES_BASE + "/") and url.endswith(".h5ad"):
            return httpx.Response(self.status, content=self.body)
        return httpx.Response(404, json={"detail": "not found"})

    def client(self):
        import httpx

        return httpx.Client(transport=httpx.MockTransport(self.handler))


# ---------------------------------------------------------------------------
# Listing and search
# ---------------------------------------------------------------------------


def test_slim_keeps_what_is_shown_and_recorded():
    s = pi.slim(LISTING_REC())
    assert s["dataset_version_id"] == LIVER_B
    assert s["dataset_id"] == LIVER_B_DATASET
    assert s["cell_count"] == 146 and s["size_bytes"] == 5382428
    assert s["organism"] == ["Homo sapiens"] and s["license"] == "CC BY 4.0"
    assert s["collection_doi"] == "10.1002/hep4.1854"
    assert s["schema_version"] == "7.1.0"
    assert pi.slim({**LISTING_REC(), "tombstone": True}) is None
    assert pi.slim({**LISTING_REC(), "assets": []}) is None


def test_search_filters_sorts_and_counts():
    ds = [pi.slim(r) for r in LISTING]
    everything = pi.search(ds)
    assert everything["total"] == 4
    cells = [d["cell_count"] for d in everything["results"]]
    assert cells == sorted(cells, reverse=True)
    mouse = pi.search(ds, organism="mus musculus")
    assert mouse["total"] == 1 and mouse["results"][0]["organism"] == ["Mus musculus"]
    assert dict(mouse["facets"]["organism"])["Homo sapiens"] == 2  # facets ignore the organism filter
    liver = pi.search(ds, q="liver b cells")
    assert [d["dataset_version_id"] for d in liver["results"]] == [LIVER_B]
    small = pi.search(ds, max_bytes=6_000_000, sort="size")
    assert all(d["size_bytes"] <= 6_000_000 for d in small["results"])
    sizes = [d["size_bytes"] for d in small["results"]]
    assert sizes == sorted(sizes)
    page = pi.search(ds, limit=2, offset=2)
    assert len(page["results"]) == 2 and page["total"] == 4


def test_listing_is_cached_with_a_ttl(tmp_path):
    fake = FakeDiscover()
    cat = pi.DiscoverCatalog(tmp_path, client_factory=fake.client)
    first = cat.listing()
    assert not first.from_cache and len(first.datasets) == 4
    assert (tmp_path / "public_import" / "discover_datasets.json").is_file()
    # A new catalog (a restart) reads the file, not the network.
    again = pi.DiscoverCatalog(tmp_path, client_factory=fake.client).listing()
    assert again.from_cache and fake.listing_calls == 1
    cat.listing(refresh=True)
    assert fake.listing_calls == 2
    expired = pi.DiscoverCatalog(tmp_path, client_factory=fake.client, ttl=0)
    expired.listing()
    assert fake.listing_calls == 3


def test_stale_listing_is_used_when_discover_is_offline(tmp_path):
    pi.DiscoverCatalog(tmp_path, client_factory=FakeDiscover().client).listing()
    offline = FakeDiscover(fail_listing=True)
    got = pi.DiscoverCatalog(tmp_path, client_factory=offline.client, ttl=0).listing()
    assert got.from_cache and len(got.datasets) == 4
    with pytest.raises(pi.PublicImportError) as e:
        pi.DiscoverCatalog(tmp_path / "empty", client_factory=offline.client).listing()
    assert e.value.status == 502


def test_find_an_older_pinned_version_through_its_history(tmp_path):
    fake = FakeDiscover()
    cat = pi.DiscoverCatalog(tmp_path, client_factory=fake.client)
    assert cat.find(LIVER_B)["title"] == "Healthy human liver: B cells"
    older = [v["dataset_version_id"] for v in VERSIONS if v["dataset_version_id"] != LIVER_B][0]
    with pytest.raises(pi.PublicImportError):
        cat.find(older)  # not current, no dataset id: refused
    rec = cat.find(older, dataset_id=LIVER_B_DATASET)
    assert rec["dataset_version_id"] == older


def test_ids_are_validated_and_the_host_is_fixed():
    assert pi.file_url(LIVER_B.upper()) == f"https://datasets.cellxgene.cziscience.com/{LIVER_B}.h5ad"
    for bad in ("../../etc/passwd", "http://evil.example/x", LIVER_B + "/../x", ""):
        with pytest.raises(pi.PublicImportError) as e:
            pi.file_url(bad)
        assert e.value.status == 422


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


def test_download_streams_to_disk_with_progress(tmp_path):
    body = bytes(range(256)) * 9000  # > 2 chunks
    fake = FakeDiscover(body)
    dest = tmp_path / "x.h5ad"
    steps = list(pi.download(LIVER_B, dest, len(body), threading.Event(), client=fake.client()))
    assert dest.read_bytes() == body
    assert steps[-1] == (len(body), len(body)) and len(steps) >= 3
    assert fake.urls == [pi.file_url(LIVER_B)]


def test_download_size_mismatch_and_http_error_leave_nothing(tmp_path):
    dest = tmp_path / "x.h5ad"
    with pytest.raises(pi.PublicImportError):
        list(pi.download(LIVER_B, dest, 999, threading.Event(), client=FakeDiscover(b"abc").client()))
    with pytest.raises(pi.PublicImportError) as e:
        list(pi.download(LIVER_B, dest, 3, threading.Event(), client=FakeDiscover(b"abc", status=403).client()))
    assert e.value.status == 502
    assert list(tmp_path.iterdir()) == []


def test_download_cancel(tmp_path):
    body = b"x" * (3 * pi.CHUNK)
    cancel = threading.Event()
    gen = pi.download(LIVER_B, tmp_path / "x.h5ad", len(body), cancel,
                      client=FakeDiscover(body).client())
    next(gen)
    cancel.set()
    with pytest.raises(pi.ImportCancelled):
        list(gen)
    assert list(tmp_path.iterdir()) == []


def test_disk_check(tmp_path, monkeypatch):
    import shutil
    from collections import namedtuple

    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(shutil, "disk_usage", lambda p: usage(10, 0, 2 * 1024**3))
    pi.check_disk(tmp_path, 100 * 1024**2)
    with pytest.raises(pi.PublicImportError) as e:
        pi.check_disk(tmp_path, 1024**3)
    assert e.value.status == 507


# ---------------------------------------------------------------------------
# The working copy
# ---------------------------------------------------------------------------


def _prepared(tmp_path):
    src = tmp_path / f"{LIVER_B}.h5ad"
    orig = discover_h5ad(src)
    rec = pi.slim(LISTING_REC())
    info = pi.about(rec, _sha(src), src.stat().st_size, "2026-10-24T12:00:00+00:00")
    out = tmp_path / "work.h5ad"
    before = _sha(src)
    summary = pi.prepare(src, out, rec, info)
    assert _sha(src) == before  # the download is never written
    return orig, out, summary, info


def test_prepare_symbols_obs_order_and_matrices(tmp_path):
    orig, out, summary, _ = _prepared(tmp_path)
    a = ad.read_h5ad(out)
    expected = ["TSPAN6", "MT-ND1", "MT-ND2", "TNMD", "DUP", "DUP-1", "CD19", "ENSG00000444444"]
    assert list(a.var_names) == expected
    assert list(a.var["gene_ids"]) == IDS
    assert list(a.raw.var_names) == expected and list(a.raw.var["gene_ids"]) == IDS
    assert list(a.obs.columns[:3]) == ["cell_type", "tissue", "disease"]
    assert summary["author_annotation"] == "cell_type"
    assert summary["var"]["renamed"] == 7
    np.testing.assert_array_equal(a.X.toarray(), orig.X.toarray())
    np.testing.assert_array_equal(a.raw.X.toarray(), orig.raw.X.toarray())
    np.testing.assert_array_equal(a.obsm["X_umap"], orig.obsm["X_umap"])
    assert a.uns["title"] == "Healthy human liver: B cells"  # Discover's own uns kept


def test_prepare_records_source_provenance_and_about(tmp_path):
    _, out, _, info = _prepared(tmp_path)
    a = ad.read_h5ad(out)
    prov = provenance.read_provenance(a)
    assert prov["source"]["origin"] == "cellxgene_discover"
    assert prov["source"]["n_cells"] == 40 and prov["source"]["n_genes"] == 8
    step = prov["history"][-1]
    assert step["step"] == "ingest" and step["tool"] == "scview.public_import"
    p = step["params"]["public_import"]
    assert p["dataset_version_id"] == LIVER_B and p["dataset_id"] == LIVER_B_DATASET
    assert p["collection_id"] == "44531dd9-1388-4416-a117-af0a99de2294"
    assert p["doi"] == "10.1002/hep4.1854" and p["license"] == "CC BY 4.0"
    assert p["sha256"] == info["about_sha256"] and p["schema_version"] == "7.1.0"
    for key in ("about_source", "about_citation", "about_doi", "about_dataset_version_id",
                "about_license", "about_sha256", "about_downloaded_at"):
        assert str(a.uns[key]) == info[key]
    with h5py.File(out) as f:
        assert f["uns/about_source"][()].decode() == "CELLxGENE Discover"


def test_citation_reaches_the_copilot_and_methods(tmp_path):
    _, out, _, _ = _prepared(tmp_path)
    a = ad.read_h5ad(out)
    cite = data_citation(a)
    assert "doi:10.1002/hep4.1854" in cite and "Licence CC BY 4.0." in cite
    assert f"dataset version {LIVER_B} (schema 7.1.0)" in cite

    class _A:
        adata = a

    assert f"Data: {cite}" in _methods_skeleton(_A())
    plain = ad.AnnData(np.zeros((2, 2)))
    assert data_citation(plain) is None


def test_replay_reproduces_the_symbols_from_the_discover_file(tmp_path):
    _, out, _, _ = _prepared(tmp_path)
    stored = ad.read_h5ad(out)
    step = provenance.read_provenance(stored)["history"][-1]
    snip = replay_script._snippet(step, replay_script._Ctx(adata=stored))
    assert snip.status == replay_script.EXACT
    code = "\n".join(snip.code)
    for start, name in ((tmp_path / f"{LIVER_B}.h5ad", "download"), (out, "stored copy")):
        ns = {"ad": ad, "pd": pd, "adata": ad.read_h5ad(start)}
        exec(code, ns)  # noqa: S102 — the generated replay code
        assert list(ns["adata"].var_names) == list(stored.var_names), name
        assert list(ns["adata"].raw.var_names) == list(stored.raw.var_names), name
    script = replay_script.build_script(stored, dataset_id="x", dataset_name="liver")
    compile(script, "replay.py", "exec")
    assert pi.file_url(LIVER_B) in script


def test_memory_verdict(tmp_path, monkeypatch):
    from scview.core import dataset_manager, memory_budget

    src = tmp_path / "a.h5ad"
    discover_h5ad(src)
    gb = 1024**3

    def budget(limit):
        return memory_budget.MemoryBudget(limit, 0, True, "container limit", 0)

    monkeypatch.setattr(dataset_manager, "load_plan",
                        lambda p: dataset_manager.LoadPlan(int(2 * gb), int(1 * gb)))
    monkeypatch.setattr(memory_budget, "read_budget", lambda: budget(8 * gb))
    assert pi.memory_verdict(src)["verdict"] == "fits"
    monkeypatch.setattr(memory_budget, "read_budget", lambda: budget(int(2.5 * gb)))
    assert pi.memory_verdict(src)["verdict"] == "backed"
    monkeypatch.setattr(memory_budget, "read_budget", lambda: budget(1 * gb))
    v = pi.memory_verdict(src)
    assert v["verdict"] == "wont_fit" and "SCVIEW_MEMORY_GB" in v["message"]


def test_run_import_refuses_what_will_not_fit_and_cleans_up(tmp_path):
    src = tmp_path / "src.h5ad"
    discover_h5ad(src)
    body = src.read_bytes()
    rec = {**pi.slim(LISTING_REC()), "size_bytes": len(body)}
    data = tmp_path / "data"
    gen = pi.run_import(rec, data, "ds1", threading.Event(), client=FakeDiscover(body).client(),
                        verdict=lambda p: {"verdict": "wont_fit", "message": "too big"})
    events = []
    with pytest.raises(pi.PublicImportError) as e:
        for ev in gen:
            events.append(ev[0])
    assert e.value.status == 507 and "memory" in events
    assert not (data / "uploads" / "ds1").exists() and not (data / "ingested" / "ds1").exists()


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------


@pytest.fixture()
def fake(monkeypatch, tmp_path):
    from scview.api.v1 import public_import as api

    src = tmp_path / "served.h5ad"
    discover_h5ad(src)
    body = src.read_bytes()
    listing = json.loads(json.dumps(LISTING))
    for r in listing:
        if r["dataset_version_id"] == LIVER_B:
            r["assets"][0]["filesize"] = len(body)
    f = FakeDiscover(body)
    orig = f.handler

    def handler(request):
        import httpx

        if str(request.url) == f"{pi.API_BASE}/datasets":
            f.listing_calls += 1
            return httpx.Response(200, json=listing)
        return orig(request)

    f.handler = handler
    monkeypatch.setattr(pi, "new_client", f.client)
    monkeypatch.setattr(pi, "memory_verdict", lambda p: {"verdict": "fits", "message": "ok"})
    api._catalogs.clear()
    yield f
    api._catalogs.clear()


def _events(text: str) -> list[tuple[str, dict]]:
    out = []
    for block in text.split("\n\n"):
        lines = [ln for ln in block.splitlines() if ln and not ln.startswith(":")]
        ev = next((ln[7:] for ln in lines if ln.startswith("event: ")), None)
        data = next((ln[6:] for ln in lines if ln.startswith("data: ")), None)
        if ev:
            out.append((ev, json.loads(data)))
    return out


@pytest.mark.asyncio
async def test_api_search_detail_and_import(client, fake, dataset_manager):
    r = await client.get(f"{API}/public-import/status")
    assert r.json()["enabled"] is True and r.json()["license"] == "CC BY 4.0"
    r = await client.get(f"{API}/public-import/discover/datasets", params={"q": "liver"})
    assert r.status_code == 200 and r.json()["total"] == 1
    r = await client.get(f"{API}/public-import/discover/datasets/{LIVER_B}")
    assert r.status_code == 200 and r.json()["download_url"] == pi.file_url(LIVER_B)
    assert r.json()["disk"]["ok"] is True
    r = await client.get(f"{API}/public-import/discover/datasets/not-a-uuid")
    assert r.status_code == 422

    r = await client.post(f"{API}/public-import/discover/import-stream",
                          json={"dataset_version_id": LIVER_B})
    assert r.status_code == 200
    events = _events(r.text)
    names = [e for e, _ in events]
    assert names[0] == "job" and names[-1] == "complete", events
    assert "progress" in names and "memory" in names
    did = events[-1][1]["dataset_id"]

    info = (await client.get(f"{API}/datasets/{did}")).json()
    assert info["name"] == "Healthy human liver: B cells"
    src = info["public_source"]
    assert src["about_dataset_version_id"] == LIVER_B and src["about_license"] == "CC BY 4.0"
    assert "doi:10.1002/hep4.1854" in src["citation_text"]
    adaptor = dataset_manager.get_dataset(did)
    assert "MT-ND1" in list(adaptor.adata.var_names)
    assert adaptor.h5ad_path.endswith(f"ingested/{did}/{LIVER_B}.h5ad")
    # The download is kept as sent, beside it.
    assert (Path(dataset_manager.data_dir) / "uploads" / did / f"{LIVER_B}.h5ad").is_file()

    r = await client.post(f"{API}/public-import/jobs/unknown/cancel")
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_api_is_off_in_public_mode_unless_enabled(client, fake, test_settings):
    test_settings.DEPLOYMENT_MODE = "public"
    r = await client.get(f"{API}/public-import/discover/datasets")
    assert r.status_code == 403
    r = await client.post(f"{API}/public-import/discover/import-stream",
                          json={"dataset_version_id": LIVER_B})
    assert r.status_code == 403
    test_settings.PUBLIC_IMPORT_ENABLED = True
    r = await client.get(f"{API}/public-import/discover/datasets")
    assert r.status_code == 200
    assert Settings(PUBLIC_IMPORT_ENABLED=False).public_import_enabled is False
    assert Settings().public_import_enabled is True
