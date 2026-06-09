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
