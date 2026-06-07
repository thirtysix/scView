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
from scview.core.rag import router, store
from scview.core.rag.embeddings import embed_query

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


async def retrieve_context(query: str, settings: Settings) -> tuple[str, list[ChatSource]]:
    """Return ``(extra_context, sources)`` for a query, or ``("", [])`` if RAG off."""
    if not settings.rag_enabled:
        return "", []

    try:
        decision = await router.route(query, settings.DEEPINFRA_API_KEY, settings.RAG_CHAT_MODEL)
        qvec = await embed_query(query, settings.DEEPINFRA_API_KEY, settings.RAG_EMBED_MODEL)
        if not qvec:
            return "", []
        hits = await store.hybrid_search(
            settings.RAG_DATABASE_URL,
            corpora=decision.corpora,
            query_vec=qvec,
            query_text=query,
            top_k=settings.RAG_TOP_K,
            vector_weight=settings.RAG_VECTOR_WEIGHT,
            text_weight=settings.RAG_TEXT_WEIGHT,
        )
    except Exception as exc:
        logger.warning("RAG retrieval failed (%s); answering from in-app facts only", exc)
        return "", []

    if not hits:
        return "", []

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
