"""Cross-encoder reranking of RAG candidates (DeepInfra reranker).

Hybrid vector+text search casts a wide net; a reranker then scores each candidate
against the *query* directly (a cross-encoder, not independent embeddings), which
is far sharper for ordering. This is an optional sharpening step: any failure
(no model, network error, unexpected response) degrades gracefully to the original
hybrid order, so retrieval never breaks because reranking was unavailable.

DeepInfra inference API: POST /v1/inference/{model} with
``{"queries": [query], "documents": [...]}`` → ``{"scores": [...]}`` aligned to
documents (verified against ``Qwen/Qwen3-Reranker-4B``).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

DEEPINFRA_INFERENCE_URL = "https://api.deepinfra.com/v1/inference"


async def rerank_scores(
    query: str,
    documents: list[str],
    api_key: str,
    model: str,
    timeout: float = 20.0,
) -> list[float] | None:
    """Return a relevance score per document (higher = better), or ``None`` if
    reranking is unavailable/failed so the caller keeps the original order."""
    if not (query and documents and api_key and model):
        return None
    import httpx

    url = f"{DEEPINFRA_INFERENCE_URL}/{model}"
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                url,
                headers={"Authorization": f"Bearer {api_key}"},
                json={"queries": [query], "documents": documents},
            )
        resp.raise_for_status()
        scores = resp.json().get("scores")
        if isinstance(scores, list) and len(scores) == len(documents):
            return [float(s) for s in scores]
        logger.warning("rerank: unexpected response shape; keeping hybrid order")
    except Exception as exc:  # network / auth / parse — all non-fatal
        logger.warning("rerank failed (%s); keeping hybrid order", exc)
    return None


async def rerank_hits(
    query: str,
    hits: list[dict],
    api_key: str,
    model: str,
    top_k: int,
) -> list[dict]:
    """Reorder retrieval ``hits`` (each with a ``content`` field) by reranker
    relevance and return the top ``top_k``. Falls back to ``hits[:top_k]``."""
    if not hits:
        return hits
    docs = [str(h.get("content", "")) for h in hits]
    scores = await rerank_scores(query, docs, api_key, model)
    if scores is None:
        return hits[:top_k]
    order = sorted(range(len(hits)), key=lambda i: scores[i], reverse=True)
    return [hits[i] for i in order[:top_k]]
