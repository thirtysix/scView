"""AI co-pilot endpoint — grounded chat about a dataset's analysis + results."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from scview.config import Settings
from scview.dependencies import get_dataset_manager, get_settings_dep
from scview.core.dataset_manager import DatasetManager
from scview.core.assistant import ChatMessage, ChatResponse, answer_query
from scview.core.rag.retrieve import retrieve_context

logger = logging.getLogger(__name__)

router = APIRouter()


class ChatRequest(BaseModel):
    query: str
    history: list[ChatMessage] | None = None
    view_context: dict | None = None  # what the user is currently looking at


@router.get("/assistant/rag-status")
async def rag_status(settings: Settings = Depends(get_settings_dep)) -> dict:
    """Whether the literature/tutorials RAG is configured, and per-corpus counts."""
    if not settings.rag_enabled:
        return {"enabled": False, "counts": {}}
    try:
        from scview.core.rag import store

        counts = await store.corpus_counts(settings.RAG_DATABASE_URL)
        return {"enabled": True, "counts": counts}
    except Exception as exc:
        logger.warning("rag-status failed: %s", exc)
        return {"enabled": True, "counts": {}, "error": str(exc)}


@router.post("/datasets/{dataset_id}/assistant/chat", response_model=ChatResponse)
async def assistant_chat(
    dataset_id: str,
    body: ChatRequest,
    dm: DatasetManager = Depends(get_dataset_manager),
    settings: Settings = Depends(get_settings_dep),
) -> ChatResponse:
    """Answer a question grounded in this dataset's analysis state and results.

    Grounds in the in-app facts (preprocessing state, provenance recipe, cluster
    sizes, top markers, enrichment). Falls back to a deterministic factual
    summary when no LLM key is configured. (A methods/literature RAG layer is a
    planned addition — see ``core/assistant.py`` ``extra_context`` hook.)
    """
    if not body.query or not body.query.strip():
        raise HTTPException(status_code=400, detail="Empty query.")

    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")

    try:
        query = body.query.strip()
        # Dual-corpus RAG retrieval (literature + tutorials); empty when RAG is off.
        extra_context, extra_sources = await retrieve_context(query, settings)
        return await answer_query(
            query=query,
            adaptor=adaptor,
            api_key=settings.DEEPINFRA_API_KEY,
            history=body.history,
            extra_context=extra_context,
            extra_sources=extra_sources,
            view_context=body.view_context,
            model=settings.RAG_CHAT_MODEL,
        )
    except Exception as exc:
        logger.error("assistant chat failed for %s: %s", dataset_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Assistant failed: {exc}")
