"""Narrative QC report: deterministic facts, the fixed-section write-up, and the
per-dataset cache that keeps a reopened tab from billing again. The LLM is a fake."""

from __future__ import annotations

import json
import types
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from scview.core import qc_report
from scview.core.anndata_adaptor import AnnDataAdaptor

from tests.test_cluster_qc import _normalised, _planted

API = "/api/v1"
SECTIONS = ("## Overview", "## Quality", "## Concerns", "## Suggested next steps")


def _kang_like(donors_per_level: int = 4) -> ad.AnnData:
    """The planted clusters plus a CTRL/STIM condition and a donor column."""
    a = _normalised(_planted())
    n = a.n_obs
    a.obs["stim"] = pd.Categorical(np.where(np.arange(n) % 2 == 0, "CTRL", "STIM"))
    a.obs["ind"] = pd.Categorical([f"d{(i // 2) % donors_per_level}" for i in range(n)])
    return a


def _facts(a: ad.AnnData, tmp_path: Path) -> qc_report.QCFacts:
    p = tmp_path / "d.h5ad"
    a.write_h5ad(p)
    return qc_report.build_facts(AnnDataAdaptor(str(p)))


def _refs(items) -> list[str]:
    return [s.ref for s in items]


def test_facts_cover_distributions_design_and_cluster_flags(tmp_path: Path):
    f = _facts(_kang_like(), tmp_path)
    refs = set(_refs(f.sources))
    assert {"qc:size", "qc:steps", "qc:design", "qc:genes", "qc:counts", "qc:mito", "qc:ribo",
            "qc:doublets", "qc:normalization", "qc:filtering", "result:qc:leiden",
            "result:qc:leiden:5"} <= refs
    design = next(s.detail for s in f.overview if s.ref == "qc:design")
    assert "Condition 'stim'" in design and "4 donors" in design and "'ind'" in design
    # The bad clusters become concerns; four donors per group means pseudobulk DE works.
    assert {"result:qc:leiden:4", "result:qc:leiden:5"} <= set(_refs(f.concerns))
    assert any("pseudobulk" in s.detail for s in f.next_steps)
    assert any(s.ref == "qc:normalization" and "Normalised; log-transformed" in s.detail
               for s in f.quality)


def test_wrong_scale_and_missing_replicates_are_concerns(tmp_path: Path):
    a = _kang_like()
    # One donor per condition: CTRL is all d0, STIM all d1.
    a.obs["ind"] = pd.Categorical(np.where(a.obs["stim"] == "CTRL", "d0", "d1"))
    counts = a.layers["counts"]
    a.X = np.log1p(counts / counts.sum(1, keepdims=True) * 1e6).astype("float32")  # CPM
    f = _facts(a, tmp_path)
    norm = next(s for s in f.concerns if s.ref == "qc:normalization")
    assert "1e6" in norm.detail
    assert any(s.ref == "qc:normalization" and "Reprocess" in s.detail for s in f.next_steps)
    assert any(s.ref == "qc:design" and "fewer than 2 donors" in s.detail for s in f.concerns)

    a.obs = a.obs.drop(columns="ind")
    f = _facts(a, tmp_path)
    assert any(s.ref == "qc:design" and "No donor/sample column" in s.detail for s in f.concerns)


def test_raw_counts_without_doublets_suggest_preprocessing(tmp_path: Path):
    a = _planted()
    del a.obs["predicted_doublet"]
    f = _facts(a, tmp_path)
    steps = " ".join(s.detail for s in f.next_steps)
    assert "Normalise and log-transform" in steps and "doublet detection" in steps


def test_deterministic_report_has_the_four_sections_and_tags(tmp_path: Path):
    f = _facts(_kang_like(), tmp_path)
    text = qc_report.deterministic_report(f)
    positions = [text.index(h) for h in SECTIONS]
    assert positions == sorted(positions)
    assert "[qc:mito]" in text and "[result:qc:leiden:5]" in text and "DEEPINFRA_API_KEY" in text


# --- The LLM write-up and its cache ------------------------------------------


class _FakeLLM:
    """Stands in for openai.AsyncOpenAI; streams a canned report, counts calls."""

    calls: list[dict] = []
    text = ["## Overview\n910 cells [qc:size].\n", "## Quality\nFine [qc:mito].\n",
            "## Concerns\n- Cluster 5 [result:qc:leiden:5]\n", "## Suggested next steps\n- Filter."]

    def __init__(self, **_):
        self.chat = types.SimpleNamespace(completions=self)

    async def create(self, **kw):
        _FakeLLM.calls.append(kw)

        async def gen():
            for t in _FakeLLM.text:
                yield types.SimpleNamespace(choices=[types.SimpleNamespace(
                    delta=types.SimpleNamespace(content=t))])

        return gen()


async def _collect(facts, **kw) -> tuple[str, dict, dict]:
    text, first, done = [], {}, {}
    async for ev in qc_report.report_events(facts, **kw):
        if ev["type"] == "facts":
            first = ev
        elif ev["type"] == "delta":
            text.append(ev["text"])
        elif ev["type"] == "done":
            done = ev
    return "".join(text), first, done


async def test_llm_report_is_grounded_cached_and_regenerated(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("openai.AsyncOpenAI", _FakeLLM)
    _FakeLLM.calls = []
    facts = _facts(_kang_like(), tmp_path)
    cache = tmp_path / "cache" / "ds" / "qc_report.json"
    kw = dict(api_key="k", model="deepseek-ai/DeepSeek-V4.1-Flash", cache_file=cache)

    text, first, done = await _collect(facts, **kw)
    assert text.startswith("## Overview") and done["grounded"] and not done["cached"]
    assert first["llm_available"] and first["facts_hash"]
    # One call, facts only: a system prompt and one user message with the tagged facts.
    (call,) = _FakeLLM.calls
    assert len(call["messages"]) == 2 and "[qc:mito]" in call["messages"][1]["content"]
    assert call["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
    assert json.loads(cache.read_text())["facts_hash"] == first["facts_hash"]

    # Reopening replays the cache: no second bill.
    text2, _, done2 = await _collect(facts, **kw)
    assert text2 == text.strip() and done2["cached"] and len(_FakeLLM.calls) == 1
    # Regenerate bills again; so do changed facts.
    await _collect(facts, regenerate=True, **kw)
    assert len(_FakeLLM.calls) == 2
    changed = facts.model_copy(update={"overview": facts.overview[:1]})
    _, f2, done3 = await _collect(changed, **kw)
    assert f2["facts_hash"] != first["facts_hash"] and not done3["cached"]
    assert len(_FakeLLM.calls) == 3


async def test_failed_llm_call_falls_back_and_is_not_cached(tmp_path: Path, monkeypatch):
    class _Broken(_FakeLLM):
        async def create(self, **kw):
            raise RuntimeError("network down")

    monkeypatch.setattr("openai.AsyncOpenAI", _Broken)
    facts = _facts(_kang_like(), tmp_path)
    cache = tmp_path / "qc.json"
    text, _, done = await _collect(facts, api_key="k", model="m", cache_file=cache)
    assert "call failed" in text and "## Concerns" in text and done["grounded"] is False
    assert not cache.exists()


# --- Endpoints ------------------------------------------------------------------


def _register(dm, did: str = "rep1") -> str:
    d = Path(dm.data_dir) / "uploads" / did
    d.mkdir(parents=True, exist_ok=True)
    (d / "metadata.json").write_text(json.dumps({"id": did, "name": did, "status": "ready"}))
    _kang_like().write_h5ad(d / "data.h5ad")
    return did


def _events(body: str) -> list[dict]:
    return [json.loads(b[5:]) for b in body.split("\n\n") if b.startswith("data:")]


async def test_endpoints_without_a_key_show_the_facts(client, dataset_manager):
    did = _register(dataset_manager)
    r = await client.get(f"{API}/datasets/{did}/qc-report")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["report"] == "" and body["llm_available"] is False and body["facts"]

    r = await client.post(f"{API}/datasets/{did}/qc-report", json={})
    body = r.json()
    assert body["grounded"] is False and all(h in body["report"] for h in SECTIONS)

    r = await client.post(f"{API}/datasets/{did}/qc-report-stream", json={})
    evs = _events(r.text)
    assert [e["type"] for e in evs] == ["facts", "delta", "done"]
    assert (await client.get(f"{API}/datasets/ghost/qc-report")).status_code == 404


async def test_stream_endpoint_with_a_key_caches_per_dataset(
    client, dataset_manager, test_settings, monkeypatch
):
    monkeypatch.setattr("openai.AsyncOpenAI", _FakeLLM)
    _FakeLLM.calls = []
    test_settings.DEEPINFRA_API_KEY = "k"
    did = _register(dataset_manager)
    r = await client.post(f"{API}/datasets/{did}/qc-report-stream", json={})
    evs = _events(r.text)
    assert evs[-1]["type"] == "done" and evs[-1]["grounded"] is True
    assert (Path(dataset_manager.data_dir) / "cache" / did / "qc_report.json").exists()
    # The tab's GET now finds it without a call.
    body = (await client.get(f"{API}/datasets/{did}/qc-report")).json()
    assert body["cached"] is True and body["report"].startswith("## Overview")
    assert len(_FakeLLM.calls) == 1
    # Removing the dataset removes its cached report.
    assert dataset_manager.remove_dataset(did)
    assert not (Path(dataset_manager.data_dir) / "cache" / did).exists()
