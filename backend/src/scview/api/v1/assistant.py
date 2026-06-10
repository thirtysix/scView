"""AI co-pilot endpoint — grounded chat about a dataset's analysis + results."""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from scview.config import Settings
from scview.dependencies import get_dataset_manager, get_settings_dep
from scview.core.dataset_manager import DatasetManager
from scview.core.assistant import (
    ChatMessage,
    ChatResponse,
    DatasetInsight,
    MethodsResponse,
    answer_query,
    build_app_context,
    build_insight,
    stream_answer,
    write_methods,
)
from scview.core.rag.retrieve import retrieve_context
from scview.core.rag.router import classify_intent

logger = logging.getLogger(__name__)

router = APIRouter()


class ChatRequest(BaseModel):
    query: str
    history: list[ChatMessage] | None = None
    view_context: dict | None = None  # what the user is currently looking at


class MethodsRequest(BaseModel):
    history: list[ChatMessage] | None = None  # the Q&A thread, for emphasis only


class FeedbackRequest(BaseModel):
    rating: str  # "up" | "down"
    question: str | None = None
    answer: str | None = None
    model: str | None = None
    route: list[str] | None = None
    dataset_id: str | None = None


@router.post("/assistant/feedback")
async def assistant_feedback(body: FeedbackRequest) -> dict:
    """Record a per-answer 👍/👎. Logged for now (no DB) — a hook for quality
    monitoring without storing conversation content beyond the log."""
    rating = (body.rating or "").lower()
    if rating not in ("up", "down"):
        raise HTTPException(status_code=400, detail="rating must be 'up' or 'down'.")
    logger.info(
        "assistant feedback: %s | model=%s route=%s dataset=%s | q=%r",
        rating, body.model, body.route, body.dataset_id, (body.question or "")[:160],
    )
    return {"ok": True}


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


@router.get("/datasets/{dataset_id}/assistant/insight", response_model=DatasetInsight)
async def assistant_insight(
    dataset_id: str,
    dm: DatasetManager = Depends(get_dataset_manager),
) -> DatasetInsight:
    """A deterministic one-line 'I notice…' nudge for when a dataset opens.

    No LLM — derived from the preprocessing state + obs structure, so it's cheap
    and reproducible. The optional ``question`` is a click-to-ask follow-up the
    co-pilot can answer (or execute, for action-style questions)."""
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")
    try:
        return build_insight(adaptor)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("insight failed for %s: %s", dataset_id, exc)
        return DatasetInsight(insight="")


@router.post("/datasets/{dataset_id}/assistant/methods", response_model=MethodsResponse)
async def assistant_methods(
    dataset_id: str,
    body: MethodsRequest,
    dm: DatasetManager = Depends(get_dataset_manager),
    settings: Settings = Depends(get_settings_dep),
) -> MethodsResponse:
    """Generate a methods-section write-up from the dataset's provenance recipe.

    Grounded strictly in the recorded steps/tools/params (never invents methods);
    the optional chat history only nudges emphasis. Falls back to a deterministic
    digest with no LLM key."""
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")
    try:
        return await write_methods(
            adaptor, body.history, settings.DEEPINFRA_API_KEY, settings.RAG_CHAT_MODEL
        )
    except Exception as exc:
        logger.error("methods generation failed for %s: %s", dataset_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Methods generation failed: {exc}")


async def _prepare(
    query: str,
    body: ChatRequest,
    dm: DatasetManager,
    settings: Settings,
    *,
    force_app: bool = False,
) -> dict:
    """Classify intent, then assemble only the knowledge it needs — shared by the
    streaming and non-streaming endpoints. ``force_app`` always includes the
    dataset-library/feature context (used when no dataset is loaded, so the
    co-pilot can help a newcomer get started). Returns kwargs for answer_query /
    stream_answer."""
    # 1. Classify intent (one cheap call) — what knowledge does this need?
    intent = await classify_intent(
        query, body.view_context, settings.DEEPINFRA_API_KEY, settings.RAG_CHAT_MODEL
    )
    chosen = set(intent.sources)
    if force_app:
        chosen.add("app")

    # 2. RAG retrieval ONLY for the corpora the classifier picked.
    rag_corpora = [c for c in ("tutorials", "literature") if c in chosen]
    extra_context, extra_sources = await retrieve_context(query, settings, rag_corpora)

    # 3. App context (dataset library + features) when asked about the app.
    app_context, app_sources = "", []
    if "app" in chosen:
        app_context, app_sources = build_app_context(dm.list_datasets(), body.view_context)

    return dict(
        history=body.history,
        extra_context=extra_context,
        extra_sources=extra_sources,
        app_context=app_context,
        app_sources=app_sources,
        view_context=body.view_context,
        include_data_grounding=("data" in chosen),
        data_facets=sorted(getattr(intent, "data_facets", []) or []),
        route=sorted(chosen),
        model=settings.RAG_CHAT_MODEL,
    )


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
    corpus is selected. Falls back to a deterministic factual summary and keyword
    routing when no LLM key is configured. See also the streaming variant below.
    """
    if not body.query or not body.query.strip():
        raise HTTPException(status_code=400, detail="Empty query.")
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")
    try:
        query = body.query.strip()
        kwargs = await _prepare(query, body, dm, settings)
        return await answer_query(query, adaptor, settings.DEEPINFRA_API_KEY, **kwargs)
    except Exception as exc:
        logger.error("assistant chat failed for %s: %s", dataset_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Assistant failed: {exc}")


@router.post("/datasets/{dataset_id}/assistant/chat-stream")
async def assistant_chat_stream(
    dataset_id: str,
    body: ChatRequest,
    dm: DatasetManager = Depends(get_dataset_manager),
    settings: Settings = Depends(get_settings_dep),
) -> StreamingResponse:
    """Streaming variant of the chat endpoint (Server-Sent Events).

    Emits one ``sources`` event (sources + route), then ``delta`` events with the
    answer tokens, then a ``done`` event with follow-up suggestions. Same intent
    routing / grounding as the non-streaming endpoint.
    """
    if not body.query or not body.query.strip():
        raise HTTPException(status_code=400, detail="Empty query.")
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")

    query = body.query.strip()

    async def event_generator():
        try:
            kwargs = await _prepare(query, body, dm, settings)
            async for event in stream_answer(
                query, adaptor, settings.DEEPINFRA_API_KEY, **kwargs
            ):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as exc:
            logger.error("assistant stream failed for %s: %s", dataset_id, exc, exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'error': str(exc)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# --- No-dataset variants: the co-pilot is available before any dataset is loaded
# (to help a newcomer get started — what scView does, how to load data). --------


@router.post("/assistant/chat", response_model=ChatResponse)
async def assistant_chat_app(
    body: ChatRequest,
    dm: DatasetManager = Depends(get_dataset_manager),
    settings: Settings = Depends(get_settings_dep),
) -> ChatResponse:
    """Co-pilot chat with no dataset loaded — grounds in the app (dataset library
    + features) and the methods/literature corpora to help the user get started."""
    if not body.query or not body.query.strip():
        raise HTTPException(status_code=400, detail="Empty query.")
    try:
        query = body.query.strip()
        kwargs = await _prepare(query, body, dm, settings, force_app=True)
        return await answer_query(query, None, settings.DEEPINFRA_API_KEY, **kwargs)
    except Exception as exc:
        logger.error("assistant app chat failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Assistant failed: {exc}")


@router.post("/assistant/chat-stream")
async def assistant_chat_app_stream(
    body: ChatRequest,
    dm: DatasetManager = Depends(get_dataset_manager),
    settings: Settings = Depends(get_settings_dep),
) -> StreamingResponse:
    """Streaming no-dataset co-pilot chat (SSE)."""
    if not body.query or not body.query.strip():
        raise HTTPException(status_code=400, detail="Empty query.")
    query = body.query.strip()

    async def event_generator():
        try:
            kwargs = await _prepare(query, body, dm, settings, force_app=True)
            async for event in stream_answer(query, None, settings.DEEPINFRA_API_KEY, **kwargs):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as exc:
            logger.error("assistant app stream failed: %s", exc, exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'error': str(exc)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
