"""TF and pathway activity (C073): ULM per cell and on a DE result, the networks.

The fixtures in tests/fixtures/activity are synthetic: invented edges written in
the two download formats (CollecTRI's Zenodo CSV, OmniPath's PROGENy annotation
table), so no test touches the internet or real network data. Agreement with
decoupler 2.0.3 on real data (Kang 2018, both networks) is in the C073 report;
here ULM is checked against its definition, a least-squares fit per cell.
"""

from __future__ import annotations

import dataclasses
import json
import threading
from pathlib import Path

import anndata as ad
import httpx
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
import scipy.sparse as sp
from scipy import stats

from scview.api.v1 import activity as activity_api
from scview.api.v1.de import DECompareRequest, DERequest, differential_expression, differential_expression_compare
from scview.core import activity, activity_nets
from scview.core.pipeline import PipelineParams, run_pipeline

API = "/api/v1"
FIXTURES = Path(__file__).parent / "fixtures" / "activity"
RAW = {
    "collectri": (FIXTURES / "collectri_sample.csv").read_bytes(),
    "progeny": (FIXTURES / "progeny_sample.tsv").read_bytes(),
}
N_GENES = 300


@pytest.fixture(autouse=True)
def small_networks(monkeypatch):
    """The fixtures are far smaller than the real networks the size checks expect."""
    for name, s in list(activity_nets.NETWORKS.items()):
        monkeypatch.setitem(activity_nets.NETWORKS, name,
                            dataclasses.replace(s, min_edges=10, min_sources=3))


def _net(name: str) -> pd.DataFrame:
    return activity_nets.build_from_bytes(name, RAW[name])


def _cells(n_per: int = 100, seed: int = 0, dense: bool = False) -> ad.AnnData:
    """Counts for CTRL and STIM cells: STIM raises STAT1's activated targets (G0-G19)
    and lowers its repressed ones (G20-G24); G295-G299 are never detected."""
    rng = np.random.default_rng(seed)
    n = 2 * n_per
    mu = np.full((n, N_GENES), 2.0)
    mu[n_per:, 0:20] *= 4.0
    mu[n_per:, 20:25] *= 0.25
    X = rng.poisson(mu * rng.gamma(4.0, 0.25, size=(n, 1))).astype("float32")
    X[:, 295:] = 0
    a = ad.AnnData(X if dense else sp.csr_matrix(X))
    a.var_names = [f"G{i}" for i in range(N_GENES)]
    a.obs_names = [f"C{i}" for i in range(n)]
    a.obs["stim"] = pd.Categorical(["CTRL"] * n_per + ["STIM"] * n_per, categories=["STIM", "CTRL"])
    return a


def _lognorm(a: ad.AnnData) -> ad.AnnData:
    a = a.copy()
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    return a


# --- networks ------------------------------------------------------------------


def test_collectri_is_built_as_decoupler_builds_it():
    net = _net("collectri")
    assert list(net.columns) == ["source", "target", "weight"]
    assert not ((net.source == "STAT1") & (net.target == "G299")).any()  # no reference: dropped
    assert not net.duplicated(["source", "target"]).any()
    assert set(net.weight) == {1.0, -1.0}
    stat1 = net[net.source == "STAT1"]
    assert len(stat1) == 25 and (stat1.weight == -1).sum() == 5


def test_progeny_keeps_significant_genes_and_the_best_record():
    net = _net("progeny")
    assert list(net.columns) == ["source", "target", "weight", "padj"]
    assert (net.padj < 0.05).all()
    assert net.source.nunique() == 10
    g0 = net[(net.source == "JAK-STAT") & (net.target == "G0")]
    assert len(g0) == 1 and g0.weight.iloc[0] == -1.5  # p 1e-4 beats the p 0.03 record
    assert len(net[net.source == "JAK-STAT"]) == 30 - 3  # every ninth gene had p 0.2


@pytest.mark.parametrize("bad, why", [
    (b"<html>captive portal</html>", "lacks columns"),
    (b"source,target,weight,references\nSTAT1,G1,2,x\n", r"\+1 or -1"),
    (b"source,target,weight,references\nFOO,G1,1,x\n", "no 'STAT1'"),
])
def test_a_wrong_collectri_file_is_refused(bad, why):
    with pytest.raises(ValueError, match=why):
        activity_nets.build_from_bytes("collectri", bad)


def test_a_truncated_download_is_refused(monkeypatch):
    monkeypatch.setitem(activity_nets.NETWORKS, "progeny",
                        dataclasses.replace(activity_nets.NETWORKS["progeny"], min_edges=10_000))
    with pytest.raises(ValueError, match="looks truncated"):
        activity_nets.build_from_bytes("progeny", RAW["progeny"])


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_download_writes_the_table_and_its_provenance(tmp_path):
    urls = []

    def handler(request):
        urls.append(str(request.url))
        return httpx.Response(200, content=RAW["collectri"])

    net = activity_nets.download_network("collectri", tmp_path, client=_client(handler))
    assert urls == [activity_nets.NETWORKS["collectri"].url]
    meta = activity_nets.read_meta(tmp_path, "collectri")
    assert meta["n_edges"] == len(net) and meta["n_sources"] == net.source.nunique()
    assert meta["download_bytes"] == len(RAW["collectri"]) and len(meta["sha256"]) == 64
    assert sorted(p.name for p in tmp_path.iterdir()) == ["collectri_human.json", "collectri_human.tsv"]
    no_network = _client(lambda request: pytest.fail("fetched a network already on disk"))
    loaded = activity_nets.load_network("collectri", tmp_path, client=no_network)
    pd.testing.assert_frame_equal(loaded, net)


def test_a_failed_download_leaves_nothing_and_says_what_to_do(tmp_path):
    bad = _client(lambda request: httpx.Response(200, text="<html>portal</html>"))
    with pytest.raises(ValueError):
        activity_nets.download_network("progeny", tmp_path, client=bad)
    assert not list(tmp_path.glob("*.tsv")) and not list(tmp_path.glob("*.part"))
    down = _client(lambda request: httpx.Response(503))
    with pytest.raises(activity_nets.NetworkUnavailable, match="fetches it once from omnipathdb.org"):
        activity_nets.load_network("progeny", tmp_path, client=down)
    assert activity_nets.read_meta(tmp_path, "progeny") is None


def test_an_oversized_download_is_cut_off(tmp_path, monkeypatch):
    monkeypatch.setattr(activity_nets, "MAX_DOWNLOAD_BYTES", 100)
    with pytest.raises(ValueError, match="exceeds"):
        activity_nets.download_network(
            "collectri", tmp_path, client=_client(lambda r: httpx.Response(200, content=RAW["collectri"])))
    assert list(tmp_path.iterdir()) == []


# --- ULM -----------------------------------------------------------------------


def _ulm_by_regression(y: np.ndarray, w: np.ndarray) -> float:
    fit = stats.linregress(w, y)
    return fit.slope / fit.stderr


@pytest.mark.parametrize("dense", [False, True])
def test_cell_scores_are_the_t_value_of_a_per_cell_regression(dense):
    a = _lognorm(_cells(n_per=20, dense=dense))
    net = _net("collectri")
    model = activity.CellModel.build(a.X, a.var_names, net)
    assert "SMALLTF" not in model.net.sources  # 3 targets < TMIN
    assert model.net.n_features == 295  # never-detected genes are left out, as decoupler does
    T = model.all()
    measured = np.arange(295)
    X = a.X.toarray() if sp.issparse(a.X) else a.X
    for source in ("STAT1", "TP53"):
        j = model.net.sources.index(source)
        w = model.net.W[:, j].toarray().ravel()[measured]
        for cell in (0, 25, 39):
            expected = _ulm_by_regression(X[cell, measured], w)
            assert T[cell, j] == pytest.approx(expected, rel=1e-4)
        np.testing.assert_allclose(model.source(source), T[:, j], rtol=1e-5)


def test_sparse_and_dense_give_the_same_scores():
    a = _lognorm(_cells())
    net = _net("progeny")
    sparse = activity.CellModel.build(a.X, a.var_names, net).all()
    dense = activity.CellModel.build(a.X.toarray(), a.var_names, net).all()
    np.testing.assert_allclose(sparse, dense, rtol=1e-5, atol=1e-5)


def test_activity_follows_the_planted_change_and_group_means_agree():
    a = _lognorm(_cells())
    model = activity.CellModel.build(a.X, a.var_names, _net("collectri"))
    stat1 = model.source("STAT1")
    assert stat1[100:].mean() > stat1[:100].mean() + 5
    codes = np.r_[np.zeros(100, int), np.ones(100, int)]
    codes[0] = -1  # an unlabelled cell is left out
    means, sizes = model.group_means(codes, 2)
    assert sizes.tolist() == [99, 100]
    j = model.net.sources.index("STAT1")
    assert means[0, j] == pytest.approx(stat1[1:100].mean(), rel=1e-5)
    assert means[1, j] == pytest.approx(stat1[100:].mean(), rel=1e-5)


def test_blocks_cover_every_cell(monkeypatch):
    a = _lognorm(_cells())
    model = activity.CellModel.build(a.X, a.var_names, _net("progeny"))
    whole = model.all()
    monkeypatch.setattr(activity, "_ROWS", 7)
    np.testing.assert_allclose(model.all(), whole)


def test_profile_scores_match_the_regression_and_rank_the_planted_tf():
    rng = np.random.default_rng(3)
    genes = [f"G{i}" for i in range(N_GENES)]
    y = rng.normal(0, 1, N_GENES)
    y[0:20] += 6  # STAT1's activated targets up ...
    y[20:25] -= 6  # ... its repressed ones down
    y[5] = 0.0  # a zero statistic is left out, as decoupler does
    net = _net("collectri")
    res = activity.score_profile(y, genes, net).set_index("source")
    assert res["score"].idxmax() == "STAT1" and res.loc["STAT1", "padj"] < 1e-6
    keep = y != 0
    w = np.zeros(N_GENES)
    stat1 = net[net.source == "STAT1"]
    w[[genes.index(t) for t in stat1.target]] = stat1.weight
    expected = _ulm_by_regression(y[keep], w[keep])
    assert res.loc["STAT1", "score"] == pytest.approx(expected, rel=1e-6)
    p = 2 * stats.t.sf(abs(expected), keep.sum() - 2)
    assert res.loc["STAT1", "pvalue"] == pytest.approx(p, rel=1e-6)
    assert ((res["padj"] >= res["pvalue"]) | np.isclose(res["padj"], res["pvalue"])).all()


def test_a_network_with_no_measured_targets_says_so():
    a = _lognorm(_cells())
    a.var_names = [f"ENSG{i:011d}" for i in range(N_GENES)]
    with pytest.raises(ValueError, match="human gene symbols"):
        activity.CellModel.build(a.X, a.var_names, _net("collectri"))


# --- which matrix is scored ----------------------------------------------------


def test_lognorm_matrix_prefers_log_values_and_never_scores_residuals():
    counts = _cells()
    out, res = run_pipeline(
        counts.copy(), ["normalization", "log_transform", "highly_variable_genes", "scaling"],
        PipelineParams(normalization_method="pearson_residuals", n_top_genes=50),
    )
    assert res.errors == {}
    M, genes, source = activity.lognorm_matrix(out)  # X: residuals; .raw: log values
    assert source == "log-normalised .raw" and M is out.raw.X

    upload = _lognorm(counts)
    upload.raw = counts.copy()  # Kang-style: X log values, counts in .raw
    M, _, source = activity.lognorm_matrix(upload)
    assert source == "log-normalised X" and M is upload.X

    M, _, source = activity.lognorm_matrix(counts.copy())  # counts only
    assert source == "shifted log of the counts in X"
    np.testing.assert_allclose(M.toarray(), _lognorm(counts).X.toarray(), rtol=1e-5, atol=1e-6)

    scaled = _lognorm(counts)
    sc.pp.scale(scaled)  # z-scores and no counts anywhere: nothing to score
    with pytest.raises(ValueError, match="No log-normalised expression"):
        activity.lognorm_matrix(scaled)


# --- endpoints -----------------------------------------------------------------


def _register(dm, a: ad.AnnData, did: str) -> str:
    d = Path(dm.data_dir) / "uploads" / did
    d.mkdir(parents=True)
    (d / "metadata.json").write_text(json.dumps({"id": did, "name": did, "status": "ready"}))
    a.write_h5ad(d / "data.h5ad")
    return did


@pytest.fixture()
def networks(tmp_data_dir: Path) -> Path:
    """Both networks 'downloaded' into the test data volume, from the fixtures."""
    dest = tmp_data_dir / "networks"
    for name in activity_nets.NETWORKS:
        activity_nets.download_network(
            name, dest, client=_client(lambda r, n=name: httpx.Response(200, content=RAW[n])))
    return dest


def test_group_codes_keep_category_order_and_skip_missing_labels():
    cat = pd.Series(pd.Categorical(["b", "a", None, "b"], categories=["b", "a", "unused"]))
    assert activity_api._group_codes(cat) == (["b", "a"], pytest.approx([0, 1, -1, 0]))
    groups, codes = activity_api._group_codes(pd.Series(["y", None, "x"], dtype=object))
    assert groups == ["x", "y"] and codes.tolist() == [1, -1, 0]


async def test_networks_endpoint_says_what_is_downloaded(client, tmp_data_dir):
    r = await client.get(f"{API}/activity/networks")
    assert r.status_code == 200
    assert {n["name"]: n["downloaded"] for n in r.json()["networks"]} == {
        "collectri": False, "progeny": False}
    dest = tmp_data_dir / "networks"
    activity_nets.download_network(
        "progeny", dest, client=_client(lambda req: httpx.Response(200, content=RAW["progeny"])))
    body = {n["name"]: n for n in (await client.get(f"{API}/activity/networks")).json()["networks"]}
    assert body["progeny"]["downloaded"] and body["progeny"]["n_sources"] == 10
    assert body["progeny"]["kind"] == "pathway" and "29295995" in body["progeny"]["citation"]


async def test_summary_scores_cells_and_averages_groups(client, dataset_manager, networks, monkeypatch):
    did = _register(dataset_manager, _lognorm(_cells()), "act")
    threads: list[threading.Thread] = []
    real = activity.CellModel.build

    def spy(*args, **kwargs):
        threads.append(threading.current_thread())
        return real(*args, **kwargs)

    monkeypatch.setattr(activity.CellModel, "build", spy)
    r = await client.get(f"{API}/datasets/{did}/activity/collectri", params={"groupby": "stim"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["groups"] == ["STIM", "CTRL"]  # the column's own category order
    assert body["group_sizes"] == [100, 100]
    assert "SMALLTF" not in body["sources"] and len(body["n_targets"]) == len(body["sources"])
    j = body["sources"].index("STAT1")
    assert body["mean"][0][j] > body["mean"][1][j] + 5
    assert body["matrix_source"] == "log-normalised X" and body["n_genes_measured"] == 295
    assert threads and threads[0] is not threading.main_thread()  # off the event loop

    r = await client.get(f"{API}/datasets/{did}/activity/progeny")
    assert r.status_code == 200 and "groups" not in r.json()
    assert len(threads) == 2  # the CollecTRI model was cached; PROGENy built once


async def test_cells_endpoint_colours_by_one_source(client, dataset_manager, networks):
    a = _lognorm(_cells())
    did = _register(dataset_manager, a, "cells")
    r = await client.get(f"{API}/datasets/{did}/activity/progeny/cells", params={"source": "JAK-STAT"})
    assert r.status_code == 200, r.text
    body = r.json()
    expected = activity.CellModel.build(a.X, a.var_names, _net("progeny")).source("JAK-STAT")
    np.testing.assert_allclose(body["scores"], expected, atol=2e-3)
    assert len(body["scores"]) == a.n_obs and body["n_targets"] > 0
    r = await client.get(f"{API}/datasets/{did}/activity/collectri/cells", params={"source": "SMALLTF"})
    assert r.status_code == 404 and "at least 5 targets" in r.json()["detail"]


async def test_de_endpoint_ranks_sources_on_the_statistic(client, dataset_manager, networks):
    did = _register(dataset_manager, _lognorm(_cells()), "de")
    rng = np.random.default_rng(0)
    y = rng.normal(0, 1, N_GENES)
    y[0:20] += 5
    genes = [f"G{i}" for i in range(N_GENES)]
    r = await client.post(f"{API}/datasets/{did}/activity/collectri/de",
                          json={"genes": genes, "stats": y.tolist()})
    assert r.status_code == 200, r.text
    res = r.json()["results"]
    assert res[0]["source"] in ("STAT1", "IRF1") and res[0]["padj"] < 1e-3
    assert [abs(x["score"]) for x in res] == sorted((abs(x["score"]) for x in res), reverse=True)
    bad = await client.post(f"{API}/datasets/{did}/activity/collectri/de",
                            json={"genes": genes, "stats": [1.0]})
    assert bad.status_code == 400


async def test_mouse_unknown_network_and_offline_are_explained(client, dataset_manager, tmp_data_dir, monkeypatch):
    mouse = _lognorm(_cells())
    mouse.var_names = [f"Gene{i}" for i in range(N_GENES)]  # title case: mouse symbols
    did = _register(dataset_manager, mouse, "mouse")
    r = await client.get(f"{API}/datasets/{did}/activity/collectri")
    assert r.status_code == 422 and "mouse" in r.json()["detail"] and "orthologs" in r.json()["detail"]

    did = _register(dataset_manager, _lognorm(_cells()), "human")
    assert (await client.get(f"{API}/datasets/{did}/activity/dorothea")).status_code == 404
    assert (await client.get(f"{API}/datasets/ghost/activity/collectri")).status_code == 404

    def offline(*args, **kwargs):
        raise httpx.ConnectError("no route to host")

    monkeypatch.setattr(activity_nets, "download_network", offline)
    r = await client.get(f"{API}/datasets/{did}/activity/collectri")
    assert r.status_code == 503 and "check the internet connection" in r.json()["detail"]


# --- the DE statistic the activity is scored on ----------------------------------


class _DM:
    def __init__(self, adata):
        self._a = type("A", (), {"adata": adata})()

    async def get_or_load_dataset(self, _id):
        return self._a


async def test_wilcoxon_de_returns_its_z_scores():
    a = _lognorm(_cells())
    res = await differential_expression("x", DERequest(indices=list(range(100, 200))), _DM(a))
    by = {g.gene: g for g in res.genes}
    assert all(g.stat is not None for g in res.genes)
    assert by["G0"].stat > 5 and by["G20"].stat < -5
    # The z-score ranks like the p-value, but keeps going where p saturates at 0.
    assert stats.norm.sf(abs(by["G0"].stat)) * 2 == pytest.approx(by["G0"].pval, rel=1e-3, abs=1e-300)


async def test_pseudobulk_de_returns_the_wald_statistic():
    rng = np.random.default_rng(1)
    rows, cond, donor = [], [], []
    for c in ("A", "B"):
        for d in range(4):
            mu = np.full((30, 60), 3.0) * rng.lognormal(0, 0.2, size=60)
            if c == "A":
                mu[:, 0] *= 4
            rows.append(rng.poisson(mu))
            cond += [c] * 30
            donor += [f"d{d}"] * 30
    a = ad.AnnData(sp.csr_matrix(np.vstack(rows).astype("float32")))
    a.var_names = [f"G{i}" for i in range(60)]
    a.obs_names = [f"c{i}" for i in range(a.n_obs)]
    a.obs["cond"] = pd.Categorical(cond)
    a.obs["donor"] = pd.Categorical(donor)
    req = DECompareRequest(split_by="cond", group_a="A", group_b="B", method="pseudobulk",
                           replicate_key="donor")
    res = await differential_expression_compare("x", req, _DM(a))
    by = {g.gene: g for g in res.genes}
    assert by["G0"].stat is not None and by["G0"].stat > 5
    assert all(g.stat is not None for g in res.genes)



async def test_de_endpoint_takes_a_stored_result_by_id(client, dataset_manager, networks):
    # A whole transcriptome sent back can pass the 512 KB request cap, so the DE
    # response's id can name the stored result instead; same answer either way.
    from scview.core import de_results

    did = _register(dataset_manager, _lognorm(_cells()), "deid")
    rng = np.random.default_rng(0)
    y = rng.normal(0, 1, N_GENES)
    y[0:20] += 5
    genes = [f"G{i}" for i in range(N_GENES)]

    class G:
        def __init__(self, gene, stat):
            self.gene, self.logfoldchange, self.pval, self.stat = gene, 1.0, 0.01, stat

    rid = de_results.remember(did, [G(g, v) for g, v in zip(genes, y)] + [G("NOSTAT", None)])
    by_id = await client.post(f"{API}/datasets/{did}/activity/collectri/de", json={"result_id": rid})
    by_value = await client.post(f"{API}/datasets/{did}/activity/collectri/de",
                                 json={"genes": genes, "stats": y.tolist()})
    assert by_id.status_code == 200, by_id.text
    assert by_id.json()["results"] == by_value.json()["results"]
    assert by_id.json()["n_genes"] == N_GENES  # the gene without a statistic is left out
    gone = await client.post(f"{API}/datasets/{did}/activity/collectri/de", json={"result_id": "nope"})
    assert gone.status_code == 404 and "run the DE again" in gone.json()["detail"]
