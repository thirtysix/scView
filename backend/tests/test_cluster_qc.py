"""Per-cluster quality flags: a planted bad cluster stands out on robust statistics."""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from scview.core.anndata_adaptor import AnnDataAdaptor
from scview.core.cluster_qc import cluster_qc, for_adaptor, grounding_block

API = "/api/v1"
N_PER = 150


def _counts(rng, n: int, *, genes_lam: float = 1.0, mt_lam: float = 0.3) -> np.ndarray:
    """Cells x (200 genes + 10 MT- + 10 ribosomal) raw counts."""
    return np.hstack([
        rng.poisson(genes_lam, (n, 200)),
        rng.poisson(mt_lam, (n, 10)),
        rng.poisson(1.5, (n, 10)),
    ]).astype("float32")


def _planted(tiny: int = 10) -> ad.AnnData:
    """Clusters 0-3 healthy, 4 doublet-rich, 5 dying (few genes, high mito), 6 tiny."""
    rng = np.random.default_rng(0)
    blocks = [_counts(rng, N_PER) for _ in range(5)]
    blocks.append(_counts(rng, N_PER, genes_lam=0.2, mt_lam=3.0))
    blocks.append(_counts(rng, tiny))
    a = ad.AnnData(np.vstack(blocks))
    a.var_names = (
        [f"G{i}" for i in range(200)] + [f"MT-{i}" for i in range(10)]
        + [f"RPL{i}" for i in range(5)] + [f"RPS{i}" for i in range(5)]
    )
    a.obs_names = [f"C{i}" for i in range(a.n_obs)]
    labels = [str(k) for k in range(6) for _ in range(N_PER)] + ["6"] * tiny
    a.obs["leiden"] = pd.Categorical(labels, categories=[str(k) for k in range(7)])
    dbl = rng.random(a.n_obs) < 0.02
    dbl[4 * N_PER: 5 * N_PER] = rng.random(N_PER) < 0.5
    a.obs["predicted_doublet"] = dbl
    a.uns["scview_active_clustering"] = "leiden"
    return a


def _by_group(report):
    return {c.group: c for c in report.clusters}


def test_planted_clusters_are_flagged_and_healthy_ones_are_not():
    report = cluster_qc(_planted())
    rows = _by_group(report)
    assert report.column == "leiden" and report.n_groups == 7
    assert [c.group for c in report.clusters] == [str(k) for k in range(7)]  # category order
    assert rows["5"].flags == ["low_quality"]
    assert "genes per cell" in rows["5"].reasons[0] and "mitochondrial" in rows["5"].reasons[0]
    assert rows["4"].flags == ["doublet_rich"]
    assert rows["6"].flags == ["small"]
    for g in "0123":
        assert rows[g].flags == [], (g, rows[g].reasons)
    # The numbers behind the flags are on the rows and in the typical values.
    assert rows["5"].median_pct_mt > 20 > report.typical["median_pct_mt"]
    assert rows["5"].median_genes < 0.5 * report.typical["median_genes"]
    assert rows["4"].doublet_fraction > 0.3
    assert {"genes", "counts", "pct_mt", "pct_ribo", "doublet_fraction"} <= set(report.metrics)
    assert report.computed_on_demand is True  # no QC columns in obs


def test_uniform_data_flags_nothing():
    rng = np.random.default_rng(1)
    a = ad.AnnData(_counts(rng, 6 * N_PER))
    a.var_names = [f"G{i}" for i in range(210)] + [f"RPL{i}" for i in range(10)]
    a.obs["cluster"] = pd.Categorical([str(k % 6) for k in range(a.n_obs)])
    report = cluster_qc(a, "cluster")
    assert not report.flagged
    # No MT- genes: % mito can't be measured, so it isn't reported as zero.
    assert "pct_mt" not in report.metrics and report.clusters[0].median_pct_mt is None
    lines, cites = grounding_block(report)
    assert any("No cluster stands out" in ln for ln in lines)
    assert [t for t, _ in cites] == ["result:qc:cluster"]


def test_high_mito_alone_and_obs_metrics_are_used():
    rng = np.random.default_rng(2)
    n = 5 * 100
    a = ad.AnnData(np.zeros((n, 3), dtype="float32"))
    a.obs["cluster"] = pd.Categorical([str(i // 100) for i in range(n)])
    a.obs["n_genes_by_counts"] = rng.normal(1500, 50, n)
    a.obs["total_counts"] = rng.normal(5000, 100, n)
    mt = rng.normal(3, 0.5, n)
    mt[400:] = rng.normal(15, 1, 100)  # cluster 4: high mito but not few genes
    a.obs["pct_counts_mt"] = mt
    report = cluster_qc(a, "cluster")
    rows = _by_group(report)
    assert rows["4"].flags == ["high_mito"]
    assert report.computed_on_demand is False
    assert "doublet_fraction" not in report.metrics


def test_text_doublet_calls_and_score_fallback():
    a = _planted()
    calls = a.obs.pop("predicted_doublet")
    a.obs["scDblFinder.class"] = np.where(calls, "doublet", "singlet")
    assert _by_group(cluster_qc(a))["4"].flags == ["doublet_rich"]
    # Categorical "True"/"False" strings must not all read as True.
    a.obs = a.obs.drop(columns="scDblFinder.class")
    a.obs["predicted_doublet"] = pd.Categorical(np.where(calls, "True", "False"))
    rows = _by_group(cluster_qc(a))
    assert rows["0"].doublet_fraction < 0.1 and rows["4"].flags == ["doublet_rich"]


def test_bad_groupings_raise_and_missing_clustering_is_explained():
    a = _planted()
    with pytest.raises(ValueError):
        cluster_qc(a, "nope")
    a.obs["const"] = "x"
    with pytest.raises(ValueError):
        cluster_qc(a, "const")
    b = ad.AnnData(np.ones((5, 2), dtype="float32"))
    report = cluster_qc(b)
    assert report.column is None and "No clustering" in report.note


def test_two_groups_only_flag_size():
    a = _planted()
    a.obs["half"] = pd.Categorical(np.where(np.arange(a.n_obs) < 5 * N_PER, "a", "b"))
    report = cluster_qc(a, "half")
    assert not report.flagged and "too few" in report.note


def test_grounding_block_cites_flagged_clusters():
    lines, cites = grounding_block(cluster_qc(_planted()))
    text = "\n".join(lines)
    assert "Per-cluster QC" in text and "[result:qc:leiden:5]" in text
    assert [t for t, _ in cites] == [
        "result:qc:leiden", "result:qc:leiden:4", "result:qc:leiden:5", "result:qc:leiden:6"]
    assert "The other 4 clusters" in text


def test_adaptor_caches_per_cell_qc_and_plots_still_work(tmp_path: Path):
    p = tmp_path / "d.h5ad"
    _planted().write_h5ad(p)
    adaptor = AnnDataAdaptor(str(p))
    report = for_adaptor(adaptor)
    assert _by_group(report)["5"].flags == ["low_quality"]
    cells = adaptor.qc_per_cell()
    assert cells is adaptor.qc_per_cell(ribo=True)  # computed once, reused
    assert cells["pct_ribo"] is not None
    qc = adaptor.qc_distributions()
    assert qc["computed_on_demand"] is True and "pct_counts_ribo" not in qc["metrics"]


def _register(dm, did: str = "qc1") -> str:
    d = Path(dm.data_dir) / "uploads" / did
    d.mkdir(parents=True, exist_ok=True)
    (d / "metadata.json").write_text(json.dumps({"id": did, "name": did, "status": "ready"}))
    _planted().write_h5ad(d / "data.h5ad")
    return did


async def test_cluster_qc_endpoint(client, dataset_manager):
    did = _register(dataset_manager)
    r = await client.get(f"{API}/datasets/{did}/cluster-qc")
    assert r.status_code == 200, r.text
    body = r.json()
    flags = {c["group"]: c["flags"] for c in body["clusters"]}
    assert flags["5"] == ["low_quality"] and flags["4"] == ["doublet_rich"]
    assert "low_quality" in body["flag_help"]
    assert (await client.get(f"{API}/datasets/{did}/cluster-qc", params={"groupby": "nope"})).status_code == 400
    assert (await client.get(f"{API}/datasets/ghost/cluster-qc")).status_code == 404


# --- Co-pilot: grounding, routing and the proactive insight --------------------


def _normalised(a: ad.AnnData) -> ad.AnnData:
    """Counts kept in layers['counts'], log-normalised X: past preprocessing step 1."""
    a.layers["counts"] = a.X.copy()
    a.X = np.log1p(a.X / a.X.sum(1, keepdims=True) * 1e4).astype("float32")
    return a


def test_grounding_context_carries_cluster_flags(tmp_path: Path):
    from scview.core.assistant import build_grounding_context

    p = tmp_path / "d.h5ad"
    _planted().write_h5ad(p)
    context, sources = build_grounding_context(AnnDataAdaptor(str(p)), {"qc"})
    assert "Per-cluster QC" in context and "Likely low quality" in context
    refs = {s.ref for s in sources}
    assert {"result:qc:leiden", "result:qc:leiden:5"} <= refs
    # Narrowed to markers only, the QC block stays out of the prompt.
    context, _ = build_grounding_context(AnnDataAdaptor(str(p)), {"markers"})
    assert "Per-cluster QC" not in context


def test_quality_questions_route_to_the_qc_facet():
    from scview.core.rag.router import heuristic_intent

    for q in ("Which clusters look low quality?", "do any clusters have high mito?",
              "are there doublet clusters in my data?"):
        intent = heuristic_intent(q)
        assert "data" in intent.sources and "qc" in intent.data_facets, q


def test_insight_names_the_low_quality_cluster():
    import types

    from scview.core.assistant import build_insight

    a = _normalised(_planted())
    del a.obs["predicted_doublet"]  # else the global doublet nudge comes first
    ins = build_insight(types.SimpleNamespace(adata=a))
    assert "Cluster 5 (leiden) stands out" in ins.insight
    assert ins.question == "Which clusters look low quality?" and ins.severity == "suggestion"


async def test_chat_fallback_answers_quality_question_from_the_numbers(client, dataset_manager):
    did = _register(dataset_manager)
    r = await client.post(
        f"{API}/datasets/{did}/assistant/chat", json={"query": "Which clusters look low quality?"}
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["grounded"] is False  # no key in tests: deterministic facts
    assert "Per-cluster QC" in body["answer"]
    assert "result:qc:leiden:5" in {s["ref"] for s in body["sources"]}
