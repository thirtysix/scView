"""Protein interaction subnetwork (C092): STRING download, analysis, gene lists, endpoints.

The STRING files here are synthetic: invented proteins and links written in the
two download formats (space-separated links, tab-separated protein info, both
gzipped), so no test touches the internet. Real-data checks (Kang CD14 Mono
STIM vs CTRL up genes; random sets) are in the C092 report.
"""

from __future__ import annotations

import dataclasses
import gzip
import hashlib
import io
import json
import threading
from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import httpx
import numpy as np
import pandas as pd
import pytest
import scanpy as sc

from scview.core import claims, de_results, gene_programs, ppi

API = "/api/v1"
TAX = 9606
N_BG = 800
MODULE = [f"MOD{k}" for k in range(20)]


def _synthetic() -> tuple[bytes, bytes]:
    """800 background proteins with ~4 random partners each (scores 150-999), a
    20-protein module linked all-to-all at 950, both link directions listed (as
    STRING does), a self link and a link to an unknown id."""
    rng = np.random.default_rng(7)
    names = [f"GENE{k}" for k in range(N_BG)] + MODULE
    ids = [f"{TAX}.ENSP{k:08d}" for k in range(len(names))]
    rows = []
    for a in range(N_BG):
        for b in rng.choice(N_BG, size=2, replace=False):
            if a != b:
                rows.append((ids[a], ids[int(b)], int(rng.integers(150, 1000))))
    for x in range(N_BG, len(names)):
        for y in range(x + 1, len(names)):
            rows.append((ids[x], ids[y], 950))
        rows.append((ids[x], ids[int(rng.integers(N_BG))], 500))
    rows += [(ids[0], ids[0], 999), (ids[1], f"{TAX}.ENSPUNKNOWN", 999)]
    rows += [(b, a, s) for a, b, s in rows]
    links = "protein1 protein2 combined_score\n" + "".join(f"{a} {b} {s}\n" for a, b, s in rows)
    info = "#string_protein_id\tpreferred_name\tprotein_size\tannotation\n" + "".join(
        f"{i}\t{n}\t100\tinvented\n" for i, n in zip(ids, names))
    return gzip.compress(links.encode()), gzip.compress(info.encode())


LINKS, INFO = _synthetic()


@pytest.fixture(autouse=True)
def small_string(monkeypatch):
    """The synthetic files' sha256 and sizes stand in for the real release's."""
    s = ppi.SPECIES["human"]
    monkeypatch.setitem(ppi.SPECIES, "human", dataclasses.replace(
        s, links_sha256=hashlib.sha256(LINKS).hexdigest(), info_sha256=hashlib.sha256(INFO).hexdigest(),
        min_proteins=100, min_edges=100))
    ppi._loaded.clear()


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _serve(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, content=LINKS if "links" in str(request.url) else INFO)


@pytest.fixture(scope="module")
def net() -> ppi.Network:
    s = ppi.SPECIES["human"]
    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(ppi.SPECIES, "human", dataclasses.replace(s, min_proteins=100, min_edges=100))
        return ppi.build_from_bytes("human", LINKS, INFO)


# --- the network ---------------------------------------------------------------


def test_links_are_kept_once_above_400_without_self_or_unknown_links(net):
    raw = pd.read_csv(io.BytesIO(gzip.decompress(LINKS)), sep=" ")
    assert net.n_proteins == N_BG + len(MODULE)
    assert (net.score >= 400).all() and (net.i < net.j).all()
    assert not ((net.i == 0) & (net.j == 0)).any()
    pairs = set(zip(net.i.tolist(), net.j.tolist()))
    assert len(pairs) == net.i.size  # each link once although listed both ways
    kept = raw[(raw.combined_score >= 400) & (raw.protein1 != raw.protein2)
               & ~raw.protein2.str.contains("UNKNOWN") & ~raw.protein1.str.contains("UNKNOWN")]
    want = {tuple(sorted((int(a[-8:]), int(b[-8:])))) for a, b in zip(kept.protein1, kept.protein2)}
    assert pairs == want
    assert net.degree(900)[N_BG] == len(MODULE) - 1  # module links at 950 survive the strictest cut


def test_a_file_for_another_species_or_truncated_is_refused():
    with pytest.raises(ValueError, match="taxon 10090"):
        ppi.build_from_bytes("mouse", LINKS, INFO)
    with pytest.raises(ValueError, match="truncated"):
        ppi.build_from_bytes("human", gzip.compress(b"protein1 protein2 combined_score\n"), INFO)


def test_download_writes_the_network_and_its_provenance(tmp_path):
    ppi.download_network("human", tmp_path, client=_client(_serve))
    meta = ppi.read_meta(tmp_path, "human")
    assert meta["licence"] == "CC BY 4.0" and meta["version"] == "12.0"
    assert meta["links_sha256"] == hashlib.sha256(LINKS).hexdigest()
    assert meta["links_url"].endswith("9606.protein.physical.links.v12.0.txt.gz")
    assert meta["n_proteins"] == N_BG + len(MODULE)
    loaded = ppi.load_network("human", tmp_path, client=_client(lambda r: pytest.fail("fetched again")))
    assert loaded.i.size == meta["n_links"] and loaded.names[N_BG] == "MOD0"
    assert not list(tmp_path.glob("*.part"))


def test_a_wrong_or_failed_download_leaves_nothing_and_says_what_to_do(tmp_path):
    bad = _client(lambda r: httpx.Response(200, text="<html>portal</html>"))
    with pytest.raises(ValueError, match="sha256"):
        ppi.download_network("human", tmp_path, client=bad)
    assert not list(tmp_path.iterdir())
    with pytest.raises(ppi.PPIUnavailable, match="stringdb-downloads.org"):
        ppi.load_network("human", tmp_path, client=_client(lambda r: httpx.Response(503)))
    with pytest.raises(ppi.PPIUnavailable, match="PPI_AUTO_DOWNLOAD is off"):
        ppi.load_network("human", tmp_path, auto_download=False,
                         client=_client(lambda r: pytest.fail("fetched with downloads off")))


# --- analysis ------------------------------------------------------------------


def test_degree_bins_put_unlinked_proteins_alone_and_follow_degree():
    deg = np.array([0, 0, 1, 1, 2, 3, 5, 8, 13, 40])
    bins = ppi.degree_bins(deg, n_bins=4)
    assert (bins[:2] == 0).all() and (bins[2:] > 0).all()
    assert (np.diff(bins) >= 0).all()


def test_the_planted_module_is_far_more_connected_than_chance(net):
    genes = [g.lower() for g in MODULE] + ["MOD0", "NOTAGENE"]  # case, a repeat, an unknown
    r = ppi.analyse(net, genes, score=400, n_draws=300)
    assert r["n_mapped"] == 20 and r["unmapped"] == ["NOTAGENE"] and r["n_given"] == 21
    assert r["n_links"] == 190 and r["expected_links"] < 10
    assert r["p_value"] == pytest.approx(1 / 301)
    assert r["nodes"][0]["name"] == "MOD0" and r["nodes"][0]["gene"] == "mod0"
    assert len(r["hubs"]) == ppi.N_HUBS and r["n_communities"] == 1
    assert all(lk["score"] == 950 for lk in r["links"])
    assert r["attribution"].startswith("STRING v12.0")


def test_random_sets_are_not_called_connected(net):
    rng = np.random.default_rng(3)
    ps = [ppi.analyse(net, list(rng.choice(net.names[:N_BG], 40, replace=False)), n_draws=200,
                      seed=k)["p_value"] for k in range(20)]
    assert np.mean(np.array(ps) < 0.05) <= 0.2
    assert np.median(ps) > 0.2


def test_the_null_is_seeded(net):
    ix = np.arange(N_BG, N_BG + 20)
    a = ppi.null_edge_counts(net, ix, 400, n_draws=50, seed=1)
    assert (a == ppi.null_edge_counts(net, ix, 400, n_draws=50, seed=1)).all()
    assert not (a == ppi.null_edge_counts(net, ix, 400, n_draws=50, seed=2)).all()


def test_two_cliques_joined_by_one_link_are_two_communities_and_isolates_are_not_drawn(net):
    # A synthetic network with two 6-cliques joined by one link and a lone protein.
    names = [f"A{k}" for k in range(6)] + [f"B{k}" for k in range(6)] + ["LONE"]
    pairs = [(x, y) for g in (range(6), range(6, 12)) for x in g for y in g if x < y] + [(0, 6)]
    small = ppi.Network(species="human", names=np.array(names), ids=np.array(names),
                        i=np.array([p[0] for p in pairs], np.int32), j=np.array([p[1] for p in pairs], np.int32),
                        score=np.full(len(pairs), 800, np.int16))
    r = ppi.analyse(small, names, n_draws=20)
    comm = {n["gene"]: n["community"] for n in r["nodes"]}
    assert len({comm[f"A{k}"] for k in range(6)}) == 1 and len({comm[f"B{k}"] for k in range(6)}) == 1
    assert comm["A1"] != comm["B1"] and comm["LONE"] is None
    assert r["n_communities"] == 2 and r["n_isolated"] == 1
    lone = next(n for n in r["nodes"] if n["gene"] == "LONE")
    assert lone["x"] is None and all(n["x"] is not None for n in r["nodes"] if n["gene"] != "LONE")
    assert r["hubs"][:2] == ["A0", "B0"]  # the bridge ends: most partners, most betweenness
    # A stricter score drops every link.
    r9 = ppi.analyse(small, names, score=900, n_draws=20)
    assert r9["n_links"] == 0 and r9["hubs"] == [] and r9["n_communities"] == 0


def test_input_is_capped_and_bad_scores_refused(net):
    r = ppi.analyse(net, list(net.names[:30]), max_genes=10, n_draws=10)
    assert r["truncated"] and r["n_used"] == 10 and r["n_given"] == 30 and r["n_mapped"] == 10
    with pytest.raises(ValueError, match="Score must be"):
        ppi.analyse(net, ["MOD0"], score=500)
    with pytest.raises(ValueError, match="No genes"):
        ppi.analyse(net, ["  "])
    one = ppi.analyse(net, ["MOD0"])
    assert one["p_value"] is None and one["n_links"] == 0


def test_logfc_is_attached_by_gene(net):
    r = ppi.analyse(net, ["MOD0", "MOD1"], logfc={"mod0": 2.5}, n_draws=10)
    assert [n["logfc"] for n in r["nodes"]] == [2.5, None]


# --- gene lists from a dataset ---------------------------------------------------


@dataclass
class G:
    gene: str
    logfoldchange: float
    pval: float
    pval_adj: float


def test_de_lists_take_significant_genes_in_one_direction_smallest_p_first():
    rid = de_results.remember("d1", [
        G("UP1", 2.0, 1e-9, 1e-8), G("UP2", 1.0, 1e-12, 1e-10), G("UPNS", 3.0, 0.5, 0.9),
        G("TINY", 0.1, 1e-20, 1e-19), G("DN1", -1.5, 1e-6, 1e-5),
    ])
    stored = de_results.recall("d1", rid)
    assert ppi.from_de(stored, 10, "up") == (["UP2", "UP1"], {"UP2": 1.0, "UP1": 2.0})
    assert ppi.from_de(stored, 10, "down")[0] == ["DN1"]
    assert ppi.from_de(stored, 2, "both")[0] == ["UP2", "UP1"]


def _markers_adata() -> ad.AnnData:
    rng = np.random.default_rng(0)
    X = rng.poisson(1.0, size=(120, 60)).astype("float32")
    X[:60, :5] += 8
    a = ad.AnnData(X)
    a.var_names = [f"GENE{k}" for k in range(60)]
    a.obs_names = [f"C{k}" for k in range(120)]
    a.obs["cluster"] = pd.Categorical(["c1"] * 60 + ["c2"] * 60)
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    sc.tl.rank_genes_groups(a, "cluster", method="wilcoxon", pts=True)
    return a


def test_sources_resolve_markers_programs_and_refuse_the_rest(monkeypatch):
    a = _markers_adata()
    label, genes, lfc = ppi.resolve_source(a, "d", "markers:cluster/c1", 10, "up")
    assert label == "cluster c1 markers" and set(genes) == {f"GENE{k}" for k in range(5)}
    assert all(lfc[g] > 1 for g in genes)
    with pytest.raises(LookupError, match="not a group"):
        ppi.resolve_source(a, "d", "markers:cluster/c9", 10, "up")
    with pytest.raises(LookupError, match="no longer kept"):
        ppi.resolve_source(a, "d", "de:nope", 10, "up")
    with pytest.raises(ValueError, match="source must be"):
        ppi.resolve_source(a, "d", "pathway:x", 10, "up")
    with pytest.raises(LookupError, match="No gene programs"):
        ppi.resolve_source(a, "d", "program:1", 10, "up")
    info = {"k": 2, "programs": [{"name": "P1", "top_genes": [{"gene": "GENE1"}, {"gene": "GENE2"}]},
                                 {"name": "P2", "top_genes": [{"gene": "GENE9"}]}]}
    monkeypatch.setattr(gene_programs, "stored", lambda adata: info)
    assert ppi.resolve_source(a, "d", "program:2", 10, "up") == ("program P2", ["GENE9"], {})
    with pytest.raises(LookupError, match="no P3"):
        ppi.resolve_source(a, "d", "program:3", 10, "up")


def test_grounding_lines_name_the_last_queries(net):
    class Obj:
        pass

    obj = Obj()
    assert ppi.grounding_lines(obj) == ([], "")
    r = ppi.analyse(net, MODULE, n_draws=100)
    ppi.remember(obj, "cluster 2 up genes", r)
    lines, detail = ppi.grounding_lines(obj)
    assert "cluster 2 up genes: 20 proteins, 190 interactions vs" in lines[1]
    assert "p < 0.01" in lines[1] and "hubs MOD" in lines[1] and lines[1].endswith(f"[{ppi.TAG}]")
    assert detail == "1 protein-interaction query"
    assert claims.resolve_tag("ppi", [ppi.TAG, "result:programs"]) == [ppi.TAG]


# --- endpoints -------------------------------------------------------------------


def _register(dm, a: ad.AnnData, did: str) -> str:
    d = Path(dm.data_dir) / "uploads" / did
    d.mkdir(parents=True)
    (d / "metadata.json").write_text(json.dumps({"id": did, "name": did, "status": "ready"}))
    a.write_h5ad(d / "data.h5ad")
    return did


@pytest.fixture()
def downloaded(tmp_data_dir: Path) -> Path:
    dest = tmp_data_dir / "networks"
    ppi.download_network("human", dest, client=_client(_serve))
    return dest


async def test_status_says_what_is_downloaded(client, downloaded):
    body = (await client.get(f"{API}/ppi/status")).json()
    sp = {s["species"]: s for s in body["species"]}
    assert sp["human"]["downloaded"] and not sp["mouse"]["downloaded"]
    assert body["licence"] == "CC BY 4.0" and body["scores"] == [400, 700, 900] and body["max_genes"] == 500


async def test_network_endpoint_runs_off_the_event_loop(client, downloaded, monkeypatch):
    threads: list[threading.Thread] = []
    real = ppi.analyse

    def spy(*args, **kwargs):
        threads.append(threading.current_thread())
        return real(*args, **kwargs)

    monkeypatch.setattr(ppi, "analyse", spy)
    r = await client.post(f"{API}/ppi/network", json={"species": "human", "genes": MODULE, "score": 700})
    assert r.status_code == 200, r.text
    assert r.json()["n_links"] == 190 and r.json()["score"] == 700
    assert threads and threads[0] is not threading.main_thread()
    bad = await client.post(f"{API}/ppi/network", json={"species": "human", "genes": MODULE, "score": 650})
    assert bad.status_code == 422
    rat = await client.post(f"{API}/ppi/network", json={"species": "rat", "genes": MODULE})
    assert rat.status_code == 422


async def test_offline_is_a_503_with_what_to_do(client, test_settings, monkeypatch):
    monkeypatch.setattr(test_settings, "PPI_AUTO_DOWNLOAD", False)
    r = await client.post(f"{API}/ppi/network", json={"species": "human", "genes": MODULE})
    assert r.status_code == 503 and "python -m scview.core.ppi" in r.json()["detail"]


async def test_dataset_endpoint_resolves_a_de_result_and_feeds_the_copilot(client, dataset_manager, downloaded):
    a = _markers_adata()
    # Rename five genes to module proteins so the list has interactions.
    a.var_names = MODULE[:10] + [f"GENE{k}" for k in range(10, 60)]
    did = _register(dataset_manager, a, "ppi1")
    rid = de_results.remember(did, [G(g, 2.0 - 0.1 * k, 1e-9, 1e-8) for k, g in enumerate(MODULE[:10])],
                              comparison={"mode": "groups", "group_a": "c1", "group_b": "c2"})
    r = await client.get(f"{API}/datasets/{did}/ppi", params={"source": f"de:{rid}", "n": 8})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n_proteins"] == 8 and body["n_links"] == 28 and body["label"].endswith("up genes")
    assert body["nodes"][0]["logfc"] == 2.0 and "upper case" in body["species_evidence"]
    adaptor = await dataset_manager.get_or_load_dataset(did)
    lines, _ = ppi.grounding_lines(adaptor.adata)
    assert "8 proteins, 28 interactions" in lines[1]
    from scview.core.assistant import build_grounding_context

    context, sources = build_grounding_context(adaptor)
    assert "8 proteins, 28 interactions" in context and ppi.TAG in {s.ref for s in sources}

    gone = await client.get(f"{API}/datasets/{did}/ppi", params={"source": "de:missing"})
    assert gone.status_code == 404
    many = await client.get(f"{API}/datasets/{did}/ppi", params={"source": f"de:{rid}", "n": 501})
    assert many.status_code == 422
    down = await client.get(f"{API}/datasets/{did}/ppi", params={"source": f"de:{rid}", "direction": "down"})
    assert down.status_code == 404 and "No genes" in down.json()["detail"]
