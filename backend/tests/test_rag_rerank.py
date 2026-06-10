"""Reranking reorders candidates by relevance and degrades gracefully."""

from __future__ import annotations

import pytest

from scview.core.rag import rerank as rr


@pytest.mark.asyncio
async def test_rerank_hits_reorders_by_score(monkeypatch):
    hits = [
        {"content": "irrelevant A"},
        {"content": "the relevant one"},
        {"content": "irrelevant B"},
    ]

    async def fake_scores(query, documents, api_key, model, timeout=20.0):
        # Score the middle doc highest.
        return [0.01, 0.99, 0.02]

    monkeypatch.setattr(rr, "rerank_scores", fake_scores)
    out = await rr.rerank_hits("q", hits, "key", "model", top_k=2)
    # 0.99 > 0.02 > 0.01 → relevant first, then irrelevant B, then A (dropped at k=2).
    assert [h["content"] for h in out] == ["the relevant one", "irrelevant B"]


@pytest.mark.asyncio
async def test_rerank_hits_falls_back_on_none(monkeypatch):
    hits = [{"content": str(i)} for i in range(5)]

    async def no_scores(*a, **k):
        return None  # reranker unavailable

    monkeypatch.setattr(rr, "rerank_scores", no_scores)
    out = await rr.rerank_hits("q", hits, "key", "model", top_k=3)
    assert [h["content"] for h in out] == ["0", "1", "2"]  # original order, truncated


@pytest.mark.asyncio
async def test_rerank_scores_guards_empty_inputs():
    assert await rr.rerank_scores("", ["d"], "k", "m") is None
    assert await rr.rerank_scores("q", [], "k", "m") is None
    assert await rr.rerank_scores("q", ["d"], "", "m") is None
    assert await rr.rerank_scores("q", ["d"], "k", "") is None
