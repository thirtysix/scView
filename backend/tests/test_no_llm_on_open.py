"""Nothing from a dataset goes to the language model unless the user asked for that call.

QA-8 (2026-11-24): opening any dataset fired ``GET /assistant/insight``, and with a key set
the backend sent the insight sentence (it can name obs columns and values) to the model to
reword it. The requests below are the ones the UI sent when Kang was opened and every panel
visited without clicking anything that asks the model (Playwright log, v0.82 noegress live
check), on a small dataset shaped like it. With a key set, none of them may build a model
client; a route that is asked on purpose (Get AI suggestions) does, so the guard can fail.
"""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import openai
import pandas as pd
import pytest
import scipy.sparse as sp

from scview.api.v1 import assistant as assistant_api
from scview.config import Settings
from scview.dependencies import get_settings_dep
from scview.main import app

API = "/api/v1"

# The UI's requests on open and on each panel, in order ({id}: the dataset).
OPEN_REQUESTS = [
    "/system/memory", "/system/version", "/datasets", "/public-import/status",
    "/datasets/{id}", "/datasets/{id}/assistant/insight",
    "/datasets/{id}/markers?format=json", "/datasets/{id}/markers?format=json&groupby_column=cluster",
    "/datasets/{id}/embeddings/X_tsne?color_by=cluster", "/datasets/{id}/markers/count-split?column=cluster",
    "/datasets/{id}/qc/ambient?column=cluster",
    # Data Assessment
    "/datasets/{id}/structure", "/datasets/{id}/seq-metrics?allow_empty=true", "/datasets/{id}/qc/ambient",
    "/datasets/{id}/hashing", "/datasets/{id}/cluster-qc", "/datasets/{id}/qc", "/datasets/{id}/qc/by-group",
    "/datasets/{id}/qc/mito-by-type", "/datasets/{id}/qc/doublets", "/datasets/{id}/assessment",
    "/datasets/{id}/qc/filter-preview?min_genes=200&max_genes=0&min_counts=0&max_pct_mt=20"
    "&drop_doublets=false&mode=mad&mad_k=5&mad_k_mt=3&mad_upper=false",
    "/datasets/{id}/qc-report", "/assistant/llm-status", "/datasets/{id}/annotation/celltypist",
    "/datasets/{id}/label-transfer", "/datasets/{id}/labels/columns", "/datasets/{id}/annotation/confidence",
    "/datasets/{id}/annotation/crosscheck", "/datasets/{id}/annotation/confidence?column=cluster",
    "/datasets/{id}/annotation/marker-db/status", "/datasets/{id}/qc/sex?column=Donor",
    "/datasets/{id}/qc/libraries?sample_column=Donor",
    "/datasets/{id}/preflight?steps=doublet_detection%2Chighly_variable_genes%2Cscaling%2Cpca%2Cneighbors"
    "%2Ccell_cycle%2Cdissociation_stress",
    # Observations, Gene Sets & Enrichment, Trajectory, Compare, History, AI Co-pilot
    "/datasets/{id}/metadata/summary?groupby=cluster", "/datasets/{id}/metadata/crosstab?row=cluster&col=orig_ident",
    "/enrichment/msigdb-collections?dataset_id={id}", "/datasets/{id}/genesets/collections",
    "/datasets/{id}/enrichment/columns", "/datasets/{id}/enrichment/groups?column=cluster",
    "/datasets/{id}/genesets/custom", "/datasets/{id}/trajectory", "/datasets/{id}/trajectory/paga?embedding=X_tsne",
    "/datasets/{id}/integration/summary", "/datasets/{id}/results", "/datasets/{id}/provenance",
    "/assistant/rag-status",
]


def _kang_like(n: int = 240, g: int = 60) -> ad.AnnData:
    """Raw counts with Kang's obs columns: the insight is a "suggestion", the kind that was reworded."""
    rng = np.random.default_rng(0)
    a = ad.AnnData(sp.csr_matrix(rng.poisson(1.5, (n, g)).astype(np.float32)))
    a.var_names = [f"MT-{i}" for i in range(3)] + [f"G{i}" for i in range(g - 3)]
    a.obs_names = [f"c{i}" for i in range(n)]
    a.obs["cluster"] = pd.Categorical(np.arange(n) % 4).rename_categories(lambda c: f"T{c}")
    a.obs["Donor"] = pd.Categorical(np.where(np.arange(n) % 2 == 0, "D1", "D2"))
    a.obs["orig_ident"] = pd.Categorical(np.where(np.arange(n) % 3 == 0, "CTRL", "STIM"))
    a.obsm["X_tsne"] = rng.normal(size=(n, 2)).astype(np.float32)
    return a


@pytest.fixture()
def built(monkeypatch, client, tmp_data_dir: Path):
    """Every model client the backend builds (openai.AsyncOpenAI / OpenAI), with a key set."""
    calls: list[str] = []

    def record(kind):
        def make(*_a, **_k):
            calls.append(kind)
            raise RuntimeError("a language model client was built")
        return make

    monkeypatch.setattr(openai, "AsyncOpenAI", record("AsyncOpenAI"))
    monkeypatch.setattr(openai, "OpenAI", record("OpenAI"))
    keyed = Settings(DATA_DIR=str(tmp_data_dir), DEEPINFRA_API_KEY="test-key", RAG_DATABASE_URL="",
                     LLM_BASE_URL="", LLM_API_KEY="", LLM_CHAT_MODEL="", LLM_PROVIDER_NAME="")
    assert keyed.llm_api_key == "test-key"
    app.dependency_overrides[get_settings_dep] = lambda: keyed
    monkeypatch.setattr(assistant_api, "_rag_status_cache", {})
    d = tmp_data_dir / "uploads" / "kang"
    d.mkdir(parents=True)
    (d / "metadata.json").write_text(json.dumps({"id": "kang", "name": "kang", "status": "ready"}))
    _kang_like().write_h5ad(d / "data.h5ad")
    return calls


async def test_opening_a_dataset_sends_nothing_to_the_model(client, built):
    failed = []
    for path in OPEN_REQUESTS:
        r = await client.get(API + path.replace("{id}", "kang"))
        if r.status_code >= 500:
            failed.append((path, r.status_code))
    assert failed == []
    insight = (await client.get(f"{API}/datasets/kang/assistant/insight")).json()
    assert insight["severity"] == "suggestion" and "raw counts" in insight["insight"]
    assert built == []


async def test_a_click_that_asks_does_reach_the_model(client, built):
    """The control: the guard sees a model call (Get AI suggestions falls back to rules here)."""
    r = await client.post(f"{API}/datasets/kang/assessment/suggest", json={"preprocessing_state": {}})
    assert r.status_code == 200, r.text
    assert built == ["AsyncOpenAI"]
