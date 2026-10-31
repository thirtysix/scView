"""AI co-pilot — grounded chat endpoint + context builder.

No DEEPINFRA_API_KEY is set in tests, so these exercise the deterministic
grounding + templated-fallback path (the substantive new code). The LLM call is
a thin wrapper around the same client pattern as llm_advisor and is not exercised
here.
"""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from scview.core.anndata_adaptor import AnnDataAdaptor
from scview.core import protein
from scview.core.assistant import build_grounding_context

API = "/api/v1"


def _register(dm, did: str = "ds1", *, n: int = 120, g: int = 80) -> str:
    """Register a small two-cluster dataset with a condition column + markers."""
    d = Path(dm.data_dir) / "uploads" / did
    d.mkdir(parents=True, exist_ok=True)
    (d / "metadata.json").write_text(json.dumps({"id": did, "name": did, "status": "ready"}))
    rng = np.random.default_rng(0)
    X = rng.poisson(0.3, size=(n, g)).astype("float32")
    X[: n // 2, :20] += rng.poisson(6.0, size=(n // 2, 20)).astype("float32")
    a = ad.AnnData(X)
    a.var_names = [f"G{i}" for i in range(g)]
    a.obs_names = [f"C{i}" for i in range(n)]
    a.obs["cluster"] = pd.Categorical(["A"] * (n // 2) + ["B"] * (n - n // 2))
    a.obs["condition"] = pd.Categorical((["ctrl", "stim"] * (n // 2 + 1))[:n])
    a.write_h5ad(d / "data.h5ad")
    return did


def _build_h5ad(path: Path, n: int = 120, g: int = 80) -> None:
    rng = np.random.default_rng(1)
    X = rng.poisson(0.3, size=(n, g)).astype("float32")
    a = ad.AnnData(X)
    a.var_names = [f"G{i}" for i in range(g)]
    a.obs_names = [f"C{i}" for i in range(n)]
    a.obs["cluster"] = pd.Categorical(["A"] * (n // 2) + ["B"] * (n - n // 2))
    a.write_h5ad(path)


def test_celltype_fact_cautions_on_a_cluster_that_is_mostly_doublets(tmp_path: Path):
    """Capture 2026-10-28: a joint cluster of 159 cells, 77% predicted doublets, was labelled
    Monocyte (the annotators see markers only; the default run keeps flagged doublets).
    The cell-type fact says so beside such a label, from the ONE doublet source."""
    p = tmp_path / "d.h5ad"
    _build_h5ad(p)
    a = ad.read_h5ad(p)
    a.obs["cell_type"] = a.obs["cluster"].map({"A": "B cell", "B": "Monocyte"}).astype("category")
    a.uns["scview_active_clustering"] = "cluster"
    a.uns["cell_type_llm_mapping"] = {"A": "B cell", "B": "Monocyte"}
    is_b = (a.obs["cluster"] == "B").to_numpy()
    calls = np.zeros(a.n_obs, dtype=bool)
    calls[np.flatnonzero(is_b)[: int(0.75 * is_b.sum())]] = True  # 75% of B, none of A
    a.obs["predicted_doublet"] = calls
    a.obs["doublet_score"] = calls.astype("float32")
    a.write_h5ad(p)
    context, _ = build_grounding_context(AnnDataAdaptor(str(p)))
    assert "- cluster **A** = B cell\n" in context
    assert ("- cluster **B** = Monocyte (caution: 75% of its cells are predicted doublets, so the "
            "label may describe two cell types captured together)") in context
    # no doublet calls: the plain line
    plain = tmp_path / "plain.h5ad"
    _build_h5ad(plain)
    a = ad.read_h5ad(plain)
    a.uns["scview_active_clustering"] = "cluster"
    a.uns["cell_type_llm_mapping"] = {"A": "B cell", "B": "Monocyte"}
    a.obs["cell_type"] = a.obs["cluster"].map({"A": "B cell", "B": "Monocyte"}).astype("category")
    a.write_h5ad(plain)
    context, _ = build_grounding_context(AnnDataAdaptor(str(plain)))
    assert "- cluster **B** = Monocyte\n" in context and "caution:" not in context


def test_grounding_names_the_protein_panel(tmp_path: Path):
    p = tmp_path / "d.h5ad"
    _build_h5ad(p)
    a = ad.read_h5ad(p)
    rng = np.random.default_rng(2)
    protein.attach(a, rng.poisson(9, (a.n_obs, 3)), ["CD3_TotalSeqB", "CD19_TotalSeqB", "IgG1_control_TotalSeqB"],
                   protein.ProteinSplit())
    a.write_h5ad(p)
    context, _ = build_grounding_context(AnnDataAdaptor(str(p)))
    assert "CITE-seq protein panel: 2 antibodies (CD3, CD19); isotype controls IgG1_control" in context
    plain = tmp_path / "plain.h5ad"
    _build_h5ad(plain)
    assert "CITE-seq" not in build_grounding_context(AnnDataAdaptor(str(plain)))[0]


def test_build_grounding_context_includes_facts(tmp_path: Path):
    p = tmp_path / "d.h5ad"
    _build_h5ad(p)
    adaptor = AnnDataAdaptor(str(p))
    context, sources = build_grounding_context(adaptor)

    # dataset summary + preprocessing assessment always present
    assert "## Dataset" in context
    assert "Preprocessing state" in context
    refs = {s.ref for s in sources}
    assert "dataset:summary" in refs
    assert "preprocessing:state" in refs
    # the cluster grouping is surfaced as a citable result fact
    assert any(r.startswith("result:groups:cluster") for r in refs)
    assert "A" in context and "B" in context


def test_marker_facts_list_fold_change_markers_not_score_order(tmp_path: Path):
    """Score order led Kang's CD14 monocytes with FTL/CD63 and its T cells with
    ribosomal genes. The fact lists genes in >= 20% of the group by fold change."""
    import scanpy as sc

    from scview.core.uns_keys import markers_key

    rng = np.random.default_rng(0)
    n = 50
    genes = ["BROAD", "SPEC", "RPL3", "RARE", "FLAT"]
    X = rng.poisson(1.0, (2 * n, len(genes))).astype("float32")
    X[:n, 0] += 2  # every A cell, a little: the top wilcoxon score
    X[: int(0.4 * n), 1] += 30  # 40% of A cells, a lot
    X[:n, 2] += 3  # ribosomal
    X[:, 3] = 0
    X[:3, 3] = 40  # 3 of 50 A cells
    a = ad.AnnData(np.log1p(X))
    a.var_names = genes
    a.obs_names = [f"C{i}" for i in range(2 * n)]
    a.obs["cluster"] = pd.Categorical(["A"] * n + ["B"] * n)
    sc.tl.rank_genes_groups(a, "cluster", method="wilcoxon", pts=True,
                            key_added=markers_key("cluster"))
    assert [str(g) for g in a.uns[markers_key("cluster")]["names"]["A"]][0] != "SPEC"
    p = tmp_path / "m.h5ad"
    a.write_h5ad(p)

    context, sources = build_grounding_context(AnnDataAdaptor(str(p)))
    assert "- **A**: SPEC, BROAD\n" in context
    fact = next(s for s in sources if s.ref == "result:markers:A")
    assert fact.detail == "A top markers: SPEC, BROAD"
    assert fact.evidence == "Top markers by cluster\n- **A**: SPEC, BROAD"


async def test_assistant_chat_grounded_fallback(client, dataset_manager):
    did = _register(dataset_manager)
    r = await client.post(
        f"{API}/datasets/{did}/assistant/chat",
        json={"query": "What cell groupings are in my data?"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    # No API key in tests -> deterministic fallback, but still grounded in facts
    assert body["grounded"] is False
    assert len(body["sources"]) >= 2
    # the answer surfaces real, dataset-specific facts (cluster + condition)
    assert "cluster" in body["answer"]
    assert "condition" in body["answer"]


async def test_assistant_chat_empty_query_400(client, dataset_manager):
    did = _register(dataset_manager)
    r = await client.post(f"{API}/datasets/{did}/assistant/chat", json={"query": "  "})
    assert r.status_code == 400


async def test_assistant_chat_missing_dataset_404(client, dataset_manager):
    r = await client.post(
        f"{API}/datasets/ghost/assistant/chat", json={"query": "hi"}
    )
    assert r.status_code == 404


def test_app_context_grounds_import_questions():
    """Suggested questions about importing must be answerable from app context:
    the feature guide enumerates supported formats and states import is local-file
    only (no URL import), so the LLM doesn't punt or hallucinate a URL flow."""
    from scview.core.assistant import build_app_context

    ctx, sources = build_app_context([{"name": "demo", "n_cells": 100, "n_genes": 50}])
    low = ctx.lower()
    for fmt in (".h5ad", "mtx", ".loom", ".zarr", "csv", ".rds", "nf-core"):
        assert fmt in low, f"supported format {fmt} missing from app context"
    # import is local-file upload only — guards against the URL-import hallucination
    assert "url" in low and "local files only" in low
    assert any(s.ref == "app:features" for s in sources)
