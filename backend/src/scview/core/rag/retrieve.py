"""Retrieval orchestration: route → embed → hybrid search → cited context.

Produces the ``extra_context`` block and ``ChatSource`` list that the co-pilot
folds into its grounded answer. Citations are typed by corpus:
  - literature → ``[lit:PMID:########]`` with title/author/year
  - tutorials  → ``[doc:<slug>#<section>]`` with title/section/url
"""

from __future__ import annotations

import logging

from scview.config import Settings
from scview.core.assistant import ChatSource
from scview.core.rag import store
from scview.core.rag.embeddings import embed_query
from scview.core.rag.rerank import rerank_hits

logger = logging.getLogger(__name__)


def _lit_ref(md: dict) -> tuple[str, str]:
    pmid = md.get("pmid") or md.get("doc_id") or "?"
    title = md.get("title", "")
    authors = md.get("authors", "")
    year = md.get("year", "")
    cite = ", ".join(p for p in (authors, str(year)) if p)
    detail = f"{title}" + (f" ({cite})" if cite else "")
    return f"lit:PMID:{pmid}", detail.strip()


def _doc_ref(md: dict) -> tuple[str, str]:
    slug = md.get("slug") or md.get("doc_id") or "doc"
    section = md.get("section", "")
    title = md.get("title", slug)
    url = md.get("url", "")
    ref = f"doc:{slug}" + (f"#{section}" if section else "")
    detail = f"{title}" + (f" — {section}" if section else "") + (f" — {url}" if url else "")
    return ref, detail.strip()


async def retrieve_context(
    query: str, settings: Settings, corpora: list[str]
) -> tuple[str, list[ChatSource]]:
    """Return ``(extra_context, sources)`` for a query against the given corpora,
    or ``("", [])`` if RAG is off or no corpora are requested. The caller (the
    intent classifier) decides ``corpora`` — so retrieval only runs when the
    question actually needs the literature/tutorials knowledge."""
    if not settings.rag_enabled or not corpora:
        return "", []

    # When a reranker is configured, cast a wider net first, then rerank down.
    rerank = bool(settings.RAG_RERANK_MODEL)
    fetch_k = max(settings.RAG_TOP_K, settings.RAG_RERANK_CANDIDATES) if rerank else settings.RAG_TOP_K

    try:
        qvec = await embed_query(query, settings.DEEPINFRA_API_KEY, settings.RAG_EMBED_MODEL)
        if not qvec:
            return "", []
        hits = await store.hybrid_search(
            settings.RAG_DATABASE_URL,
            corpora=corpora,
            query_vec=qvec,
            query_text=query,
            top_k=fetch_k,
            vector_weight=settings.RAG_VECTOR_WEIGHT,
            text_weight=settings.RAG_TEXT_WEIGHT,
        )
    except Exception as exc:
        logger.warning("RAG retrieval failed (%s); answering from in-app facts only", exc)
        return "", []

    if not hits:
        return "", []

    # Sharpen ordering with a cross-encoder reranker (degrades to hybrid order).
    if rerank:
        hits = await rerank_hits(
            query, hits, settings.DEEPINFRA_API_KEY, settings.RAG_RERANK_MODEL, settings.RAG_TOP_K
        )

    lines: list[str] = [
        "# RETRIEVED KNOWLEDGE (cite these tags; literature = research abstracts, "
        "tutorials = methods docs)"
    ]
    sources: list[ChatSource] = []
    for h in hits:
        md = h.get("metadata", {}) or {}
        if h["corpus"] == "literature":
            ref, detail = _lit_ref(md)
            kind = "literature"
        else:
            ref, detail = _doc_ref(md)
            kind = "tutorial"
        snippet = h["content"].strip().replace("\n", " ")
        if len(snippet) > 500:
            snippet = snippet[:500] + "…"
        lines.append(f"\n[{ref}] {detail}\n{snippet}")
        sources.append(ChatSource(kind=kind, ref=ref, detail=detail or snippet[:120]))

    return "\n".join(lines), sources
