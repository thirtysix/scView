"""Preranked GSEA of a DE result or a marker group (core/gsea.py, POST .../enrichment/gsea).

gseapy runs for real against a small MSigDB collection in a temp directory: one
set planted at the top of the ranking, one at the bottom, one in the middle, and
one too small to test.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
import scipy.sparse as sp

from scview.api.v1.de import DERequest, differential_expression
from scview.config import Settings
from scview.core import de_results, gsea
from scview.core.uns_keys import markers_key

API = "/api/v1"
GENES = [f"G{i}" for i in range(300)] + ["RPL3", "RPS6", "MT-CO1"]
TOP_SET = GENES[0:20]
BOTTOM_SET = GENES[280:300]
MIDDLE_SET = GENES[7:300:15]  # spread evenly over the ranking: no enrichment
SMALL_SET = GENES[0:5]


@pytest.fixture()
def msigdb_dir(tmp_path: Path) -> Path:
    d = tmp_path / "msigdb"
    d.mkdir()
    (d / "h.all.v2026.1.Hs.json").write_text(json.dumps({
        "HALLMARK_TOP": {"geneSymbols": TOP_SET},
        "HALLMARK_BOTTOM": {"geneSymbols": BOTTOM_SET},
        "HALLMARK_MIDDLE": {"geneSymbols": MIDDLE_SET},
        "HALLMARK_SMALL": {"geneSymbols": SMALL_SET},
    }))
    return d


@pytest.fixture()
def test_settings(tmp_data_dir: Path, msigdb_dir: Path) -> Settings:
    return Settings(DATA_DIR=str(tmp_data_dir), DEEPINFRA_API_KEY="", RAG_DATABASE_URL="",
                    MSIGDB_DIR=str(msigdb_dir))


@pytest.fixture(autouse=True)
def _fresh_cache():
    gsea.clear_cache()
    de_results.clear()
    yield
    gsea.clear_cache()
    de_results.clear()


@pytest.fixture()
def prerank_calls(monkeypatch):
    import gseapy

    calls = []
    real = gseapy.prerank

    def counting(*args, **kwargs):
        calls.append(kwargs.get("permutation_num"))
        return real(*args, **kwargs)

    monkeypatch.setattr(gseapy, "prerank", counting)
    return calls


def _register(dm, did: str = "gs") -> str:
    """T cells over-express the top set and under-express the bottom one."""
    d = Path(dm.data_dir) / "uploads" / did
    d.mkdir(parents=True)
    (d / "metadata.json").write_text(json.dumps({"id": did, "name": did, "status": "ready"}))
    rng = np.random.default_rng(0)
    X = rng.poisson(2.0, (90, len(GENES))).astype("float32")
    X[:45, 0:20] += 8
    X[45:, 280:300] += 8
    a = ad.AnnData(X)
    a.var_names = GENES
    a.obs_names = [f"C{i}" for i in range(90)]
    a.obs["cell_type"] = pd.Categorical(np.repeat(["T", "B"], 45))
    sc.pp.normalize_total(a)
    sc.pp.log1p(a)
    sc.tl.rank_genes_groups(a, "cell_type", method="wilcoxon", key_added=markers_key("cell_type"))
    a.write_h5ad(d / "data.h5ad")
    return did


def _de_genes(with_score: bool = True) -> list[dict]:
    """A DE result ranked G0 (most up) .. G299 (most down)."""
    out = []
    n = len(GENES)
    for i, g in enumerate(GENES):
        z = 10 - 20 * i / (n - 1)
        row = {"gene": g, "logfoldchange": z / 5, "pval": float(10 ** -abs(z)), "pval_adj": 0.5}
        if with_score:
            row["score"] = z
        out.append(row)
    return out


def _terms(side: dict) -> dict[str, dict]:
    return {r["term"]: r for r in side["results"]}


# --- ranking --------------------------------------------------------------------


def test_ranking_sorts_drops_and_filters():
    r = gsea.build_ranking(
        [("B", 1.0), ("A", 1.0), ("C", 3.0), ("C", -9.0), ("D", float("nan")), ("RPL3", 2.0),
         ("E", None)],
        "wilcoxon_score", exclude_ribo_mito=True,
    )
    assert r.genes == ["C", "A", "B"]  # highest first, a tie by name, first C kept
    assert r.n_excluded_ribo_mito == 1
    with pytest.raises(gsea.GseaError):
        gsea.build_ranking([("A", 1.0)], "x", exclude_ribo_mito=False)


def test_de_ranking_uses_the_statistic_else_signed_p():
    by_score = gsea.ranking_from_de(_de_genes(), exclude_ribo_mito=False)
    assert by_score.metric == "wilcoxon_score"
    assert by_score.genes[0] == "G0" and by_score.genes[-1] == "MT-CO1"

    rows = [{"gene": "UP", "logfoldchange": 2.0, "pval": 0.0},  # underflow: floored, not inf
            {"gene": "DOWN", "logfoldchange": -1.0, "pval": 1e-5},
            {"gene": "FLAT", "logfoldchange": 0.1, "pval": 1.0}]
    signed = gsea.ranking_from_de(rows, exclude_ribo_mito=False)
    assert signed.metric == "signed_log10_p"
    assert signed.genes == ["UP", "FLAT", "DOWN"]
    assert signed.values[0] == pytest.approx(300.0)
    assert signed.values[-1] == pytest.approx(-5.0)
    # One gene without a score: the whole result ranks by signed p.
    mixed = _de_genes()
    mixed[3].pop("score")
    assert gsea.ranking_from_de(mixed).metric == "signed_log10_p"


def test_permutations_shrink_with_the_number_of_sets():
    assert gsea.permutations_for(50) == 1000
    assert gsea.permutations_for(6835) == gsea.PERMUTATION_BUDGET // 6835
    assert gsea.permutations_for(20_000) == gsea.MIN_PERMUTATIONS
    assert gsea.permutations_for(50, requested=200) == 200


def test_set_size_counts_genes_as_gseapy_matches_them():
    sets = {"S": ["CD14", "LYZ", "FCGR3A"], "T": ["XYZ"]}
    assert gsea.sets_in_range(sets, ["Cd14", "Lyz2", "Fcgr3a", "Actb"], 2, 5) == {"S": sets["S"]}


# --- the run --------------------------------------------------------------------


def test_planted_sets_come_out_with_their_sign_and_leading_edge():
    ranking = gsea.ranking_from_de(_de_genes(), exclude_ribo_mito=True)
    sets = {"TOP": TOP_SET, "BOTTOM": BOTTOM_SET, "MIDDLE": MIDDLE_SET, "SMALL": SMALL_SET}
    run = gsea.run_prerank(ranking, sets, permutations=200)
    assert run.n_sets == 3  # SMALL has fewer than 15 genes
    assert run.permutations == 200
    up = {r["term"]: r for r in run.up}
    down = {r["term"]: r for r in run.down}
    assert up["TOP"]["nes"] > 1.5 and up["TOP"]["fdr"] < 0.05
    assert down["BOTTOM"]["nes"] < -1.5 and down["BOTTOM"]["fdr"] < 0.05
    assert up["TOP"]["set_size"] == 20
    lead = up["TOP"]["leading_edge"]
    assert lead and set(lead) <= set(TOP_SET) and lead[0] == "G0"
    assert run.up[0]["term"] == "TOP"  # most significant first
    mid = up.get("MIDDLE") or down.get("MIDDLE")
    assert mid["fdr"] > 0.05


def test_too_many_sets_or_none_in_range_is_refused(monkeypatch):
    ranking = gsea.ranking_from_de(_de_genes())
    with pytest.raises(gsea.GseaError, match="genes in the ranking"):
        gsea.run_prerank(ranking, {"SMALL": SMALL_SET})
    monkeypatch.setattr(gsea, "MAX_SETS", 2)
    with pytest.raises(gsea.GseaError, match="at most 2"):
        gsea.run_prerank(ranking, {"TOP": TOP_SET, "BOTTOM": BOTTOM_SET, "MIDDLE": MIDDLE_SET})


# --- the endpoint -----------------------------------------------------------------


async def _gsea(client, did="gs", **body):
    r = await client.post(f"{API}/datasets/{did}/enrichment/gsea",
                          json={"collections": ["h.all"], "permutations": 100, **body})
    return r


async def test_de_result(client, dataset_manager):
    did = _register(dataset_manager)
    r = await _gsea(client, did, genes=_de_genes())
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["rank_metric"] == "wilcoxon_score"
    assert "z-score" in body["rank_metric_label"]
    assert body["n_excluded_ribo_mito"] == 3
    assert body["n_genes_ranked"] == 300
    assert body["n_sets_tested"] == 3
    up, down = _terms(body["up"]), _terms(body["down"])
    assert up["HALLMARK_TOP"]["nes"] > 0 and up["HALLMARK_TOP"]["collection"] == "h.all"
    assert down["HALLMARK_BOTTOM"]["nes"] < 0
    assert body["up"]["n_significant"] >= 1 and body["down"]["n_significant"] >= 1


async def test_pseudobulk_result_ranks_by_signed_p(client, dataset_manager):
    did = _register(dataset_manager)
    r = await _gsea(client, did, genes=_de_genes(with_score=False))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["rank_metric"] == "signed_log10_p"
    assert _terms(body["up"])["HALLMARK_TOP"]["fdr"] < 0.05


async def test_marker_group_ranks_by_the_marker_score(client, dataset_manager):
    did = _register(dataset_manager)
    r = await _gsea(client, did, column="cell_type", group="T")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["rank_metric"] == "marker_score"
    assert body["group"] == "T" and body["column"] == "cell_type"
    assert _terms(body["up"])["HALLMARK_TOP"]["nes"] > 0
    assert _terms(body["down"])["HALLMARK_BOTTOM"]["nes"] < 0

    r = await _gsea(client, did, column="cell_type", group="B")
    assert _terms(r.json()["up"])["HALLMARK_BOTTOM"]["nes"] > 0
    r = await _gsea(client, did, column="no_markers", group="T")
    assert r.status_code == 404


async def test_the_same_request_is_served_from_the_cache(client, dataset_manager, prerank_calls):
    did = _register(dataset_manager)
    first = (await _gsea(client, did, genes=_de_genes())).json()
    again = (await _gsea(client, did, genes=_de_genes(), max_terms=1)).json()
    assert prerank_calls == [100]
    assert not first["cached"] and again["cached"]
    assert again["up"]["n_terms"] == first["up"]["n_terms"]
    assert len(again["up"]["results"]) == 1
    # Another ranking, or other parameters, is a new run.
    await _gsea(client, did, genes=_de_genes(with_score=False))
    await _gsea(client, did, genes=_de_genes(), exclude_ribo_mito=False)
    await _gsea(client, did, genes=_de_genes(), seed=7)
    assert len(prerank_calls) == 4


async def test_bad_requests(client, dataset_manager, monkeypatch):
    did = _register(dataset_manager)
    assert (await _gsea(client, did)).status_code == 400  # nothing to rank
    assert (await _gsea(client, did, genes=_de_genes(), group="T")).status_code == 400
    monkeypatch.setattr(gsea, "MAX_SETS", 1)
    r = await _gsea(client, did, genes=_de_genes())
    assert r.status_code == 400 and "Pick fewer collections" in r.json()["detail"]


async def test_runs_off_the_event_loop(client, dataset_manager, monkeypatch):
    import gseapy

    from tests.test_event_loop import _worst_stall

    did = _register(dataset_manager)
    real = gseapy.prerank

    def slow(*args, **kwargs):
        time.sleep(0.5)
        return real(*args, **kwargs)

    monkeypatch.setattr(gseapy, "prerank", slow)
    r, stall = await _worst_stall(_gsea(client, did, genes=_de_genes()))
    assert r.status_code == 200, r.text
    assert stall < 0.25, f"event loop blocked {stall:.2f} s"


async def test_wilcoxon_de_carries_the_statistic():
    a = ad.AnnData(np.abs(np.random.default_rng(0).normal(5, 1, (40, 12))).astype("float32"))
    a.X[:20, :3] += 6
    a.var_names = [f"G{i}" for i in range(12)]

    class _DM:
        async def get_or_load_dataset(self, _id):
            return type("A", (), {"adata": a})()

    res = await differential_expression("x", DERequest(indices=list(range(20))), _DM())
    by = {g.gene: g for g in res.genes}
    assert all(g.score is not None for g in res.genes)
    assert by["G0"].score > 3 and by["G0"].logfoldchange > 0
    # Ranked the way the test ranked them.
    scores = [g.score for g in res.genes]
    assert scores == sorted(scores, reverse=True)


# --- a whole transcriptome: by the DE result's id ------------------------------------


def _register_transcriptome(dm, did: str = "big", n_genes: int = 15_000) -> str:
    """60 cells x 15k genes; the first 30 cells over-express the top set."""
    d = Path(dm.data_dir) / "uploads" / did
    d.mkdir(parents=True)
    (d / "metadata.json").write_text(json.dumps({"id": did, "name": did, "status": "ready"}))
    rng = np.random.default_rng(1)
    X = rng.poisson(0.3, (60, n_genes)).astype("float32")
    X[:30, 0:20] += 5
    X[30:, 280:300] += 5
    a = ad.AnnData(sp.csr_matrix(X))
    a.var_names = [f"G{i}" for i in range(n_genes)]
    a.obs_names = [f"C{i}" for i in range(60)]
    a.write_h5ad(d / "data.h5ad")
    return did


async def test_a_whole_transcriptome_goes_by_the_de_results_id(client, dataset_manager):
    did = _register_transcriptome(dataset_manager)
    r = await client.post(f"{API}/datasets/{did}/de", json={"indices": list(range(30)), "label": "A"})
    assert r.status_code == 200, r.text
    de = r.json()
    assert de["result_id"] and len(de["genes"]) == 15_000

    # Sending every gene back is past the request size cap...
    genes = [{k: g[k] for k in ("gene", "logfoldchange", "pval", "score")} for g in de["genes"]]
    too_big = await _gsea(client, did, genes=genes)
    assert too_big.status_code == 413
    # ... so GSEA takes the result by its id.
    r = await _gsea(client, did, de_result_id=de["result_id"])
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["rank_metric"] == "wilcoxon_score"
    assert body["n_genes_ranked"] == 15_000
    assert _terms(body["up"])["HALLMARK_TOP"]["fdr"] < 0.05
    assert _terms(body["down"])["HALLMARK_BOTTOM"]["nes"] < 0


async def test_an_unknown_or_foreign_result_id_is_404(client, dataset_manager):
    did = _register(dataset_manager)
    other = _register_transcriptome(dataset_manager, "other", n_genes=400)
    r = await client.post(f"{API}/datasets/{other}/de", json={"indices": list(range(30))})
    rid = r.json()["result_id"]
    assert (await _gsea(client, did, de_result_id=rid)).status_code == 404  # another dataset's
    r = await _gsea(client, did, de_result_id="nope")
    assert r.status_code == 404 and "run the DE again" in r.json()["detail"]
    both = await _gsea(client, other, de_result_id=rid, column="cell_type", group="T")
    assert both.status_code == 400


def test_the_store_keeps_the_last_results(monkeypatch):
    monkeypatch.setattr(de_results, "KEEP", 2)

    class G:
        def __init__(self, gene, score=None):
            self.gene, self.logfoldchange, self.pval, self.score = gene, 1.0, 0.01, score

    ids = [de_results.remember("d", [G("A", 2.0), G("B")]) for _ in range(3)]
    assert de_results.recall("d", ids[0]) is None  # dropped for newer ones
    rows = de_results.recall("d", ids[2]).rows()
    assert rows == [{"gene": "A", "logfoldchange": 1.0, "pval": 0.01, "score": 2.0},
                    {"gene": "B", "logfoldchange": 1.0, "pval": 0.01, "score": None}]
    assert de_results.recall("other", ids[2]) is None


async def test_pseudobulk_and_compare_results_carry_an_id(client, dataset_manager):
    did = _register(dataset_manager)
    r = await client.post(f"{API}/datasets/{did}/de/compare",
                          json={"split_by": "cell_type", "group_a": "T", "group_b": "B"})
    assert r.status_code == 200, r.text
    assert de_results.recall(did, r.json()["result_id"]) is not None
