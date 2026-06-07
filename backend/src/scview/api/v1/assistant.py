"""AI co-pilot endpoint — grounded chat about a dataset's analysis + results."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from scview.config import Settings
from scview.dependencies import get_dataset_manager, get_settings_dep
from scview.core.dataset_manager import DatasetManager
from scview.core.assistant import ChatMessage, ChatResponse, answer_query, build_app_context
from scview.core.rag.retrieve import retrieve_context
from scview.core.rag.router import classify_intent

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
    """Answer a question, routing first so we only spend the resources it needs.

    An intent classifier (using the current tab) picks the minimal knowledge set:
    ``app`` (dataset library + features), ``data`` (the loaded dataset's facts),
    ``tutorials`` / ``literature`` (RAG corpora). RAG retrieval runs *only* when a
    corpus is selected — so "what datasets do we have?" answers from local state
    with no embedding/vector-search cost. Falls back to a deterministic factual
    summary and keyword routing when no LLM key is configured.
    """
    if not body.query or not body.query.strip():
        raise HTTPException(status_code=400, detail="Empty query.")

    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")

    try:
        query = body.query.strip()

        # 1. Classify intent (one cheap call) — what knowledge does this need?
        intent = await classify_intent(
            query, body.view_context, settings.DEEPINFRA_API_KEY, settings.RAG_CHAT_MODEL
        )
        chosen = set(intent.sources)

        # 2. RAG retrieval ONLY for the corpora the classifier picked.
        rag_corpora = [c for c in ("tutorials", "literature") if c in chosen]
        extra_context, extra_sources = await retrieve_context(query, settings, rag_corpora)

        # 3. App context (dataset library + features) only when asked about the app.
        app_context, app_sources = "", []
        if "app" in chosen:
            app_context, app_sources = build_app_context(dm.list_datasets(), body.view_context)

        return await answer_query(
            query=query,
            adaptor=adaptor,
            api_key=settings.DEEPINFRA_API_KEY,
            history=body.history,
            extra_context=extra_context,
            extra_sources=extra_sources,
            app_context=app_context,
            app_sources=app_sources,
            view_context=body.view_context,
            include_data_grounding=("data" in chosen),
            route=sorted(chosen),
            model=settings.RAG_CHAT_MODEL,
        )
    except Exception as exc:
        logger.error("assistant chat failed for %s: %s", dataset_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Assistant failed: {exc}")
