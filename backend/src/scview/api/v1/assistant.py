"""AI co-pilot endpoint — grounded chat about a dataset's analysis + results."""

from __future__ import annotations

import asyncio
import json
import logging
import time

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from scview.core.sse import keepalive
from scview.config import Settings
from scview.dependencies import get_dataset_manager, get_settings_dep
from scview.core.dataset_manager import DatasetManager
from scview.core import claims as claims_mod
from scview.core.assistant import (
    DEEPINFRA_BASE_URL,
    ChatMessage,
    ChatResponse,
    DatasetInsight,
    MethodsResponse,
    answer_query,
    build_app_context,
    build_insight,
    polish_insight,
    stream_answer,
    write_methods,
)
from scview.core.guards import enforce_query_limits
from scview.core.rag.retrieve import retrieve_context
from scview.core.rag.router import classify_intent, mentions_app_ui

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


class ClaimToJudge(BaseModel):
    id: int
    text: str = Field(max_length=2 * claims_mod.MAX_CLAIM_CHARS)
    evidence: str = Field("", max_length=2 * claims_mod.MAX_EVIDENCE_CHARS)


class CheckClaimsRequest(BaseModel):
    claims: list[ClaimToJudge]


class JudgedClaim(BaseModel):
    id: int
    verdict: str
    reason: str


class CheckClaimsResponse(BaseModel):
    claims: list[JudgedClaim]
    model: str


@router.post("/assistant/check-claims", response_model=CheckClaimsResponse)
async def check_claims(
    body: CheckClaimsRequest, settings: Settings = Depends(get_settings_dep)
) -> CheckClaimsResponse:
    """The ledger's "Check claims" button: one small LLM call judges the claims the
    rules left uncertain, each against only the fact snippet it cited."""
    if not body.claims:
        raise HTTPException(status_code=400, detail="No claims to check.")
    if len(body.claims) > claims_mod.MAX_JUDGE_CLAIMS:
        raise HTTPException(
            status_code=413,
            detail=f"Too many claims (max {claims_mod.MAX_JUDGE_CLAIMS} per check).",
        )
    if not settings.DEEPINFRA_API_KEY:
        raise HTTPException(status_code=503, detail="The language model is not configured.")
    try:
        judged = await claims_mod.judge_claims(
            [c.model_dump() for c in body.claims], settings.DEEPINFRA_API_KEY,
            settings.RAG_CHAT_MODEL, DEEPINFRA_BASE_URL,
        )
    except Exception as exc:
        logger.warning("claims judge failed: %s", exc)
        raise HTTPException(status_code=502, detail="The claims check failed; try again.")
    return CheckClaimsResponse(claims=judged, model=settings.RAG_CHAT_MODEL)


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


# The co-pilot's "Sources" note reads this on every open; a remote (Neon) database
# would be woken each time, so a good answer is kept for a while.
_RAG_STATUS_TTL = 600.0
_rag_status_cache: dict = {}


@router.get("/assistant/rag-status")
async def rag_status(settings: Settings = Depends(get_settings_dep)) -> dict:
    """Whether the literature/tutorials RAG is configured, and what each corpus
    holds: ``corpora[name] = {documents, chunks, last_added}`` (``counts`` keeps
    the chunk counts for older callers). Read-only: one SELECT, nothing written."""
    if not settings.rag_enabled:
        reason = "no_database" if not settings.RAG_DATABASE_URL else "no_api_key"
        return {"enabled": False, "counts": {}, "corpora": {}, "reason": reason}
    hit = _rag_status_cache.get(settings.RAG_DATABASE_URL)
    if hit and time.monotonic() - hit[0] < _RAG_STATUS_TTL:
        return hit[1]
    try:
        from scview.core.rag import store

        corpora = await store.corpus_stats(settings.RAG_DATABASE_URL)
    except Exception as exc:
        # Details go to the log only: a driver error can echo the connection string.
        logger.warning("rag-status failed: %s", exc)
        return {"enabled": True, "counts": {}, "corpora": {}, "error": "database unreachable"}
    out = {
        "enabled": True,
        "counts": {c: s["chunks"] for c, s in corpora.items()},
        "corpora": corpora,
    }
    _rag_status_cache[settings.RAG_DATABASE_URL] = (time.monotonic(), out)
    return out


@router.get("/datasets/{dataset_id}/assistant/insight", response_model=DatasetInsight)
async def assistant_insight(
    dataset_id: str,
    polish: bool = True,
    dm: DatasetManager = Depends(get_dataset_manager),
    settings: Settings = Depends(get_settings_dep),
) -> DatasetInsight:
    """A one-line 'I notice…' nudge for when a dataset opens.

    The nudge is chosen deterministically from the preprocessing state + obs
    structure (cheap, reproducible) — including a QC-anomaly check. When an LLM
    key is configured and ``polish`` is set, an actionable nudge is reworded into
    a friendlier sentence (facts preserved; deterministic text is the fallback).
    The ``question`` is a click-to-ask follow-up the co-pilot answers or executes."""
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")
    try:
        # The per-cluster QC check may read the counts matrix: keep it off the loop.
        insight = await asyncio.to_thread(build_insight, adaptor)
        if polish:
            insight = await polish_insight(insight, settings.DEEPINFRA_API_KEY, settings.RAG_CHAT_MODEL)
        return insight
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
    enforce_query_limits("", body.history, settings)
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
    if force_app or mentions_app_ui(query):
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


async def _warm_cluster_qc(adaptor, kwargs: dict) -> None:
    """Per-cluster QC grounding may need a pass over the counts matrix the first
    time; do that in a thread so the event loop isn't blocked (then it's cached)."""
    facets = kwargs.get("data_facets") or []
    if adaptor is None or not kwargs.get("include_data_grounding"):
        return
    if facets and "qc" not in facets:
        return
    try:
        await asyncio.to_thread(adaptor.qc_per_cell, True)
    except Exception as exc:  # grounding skips QC on its own if this fails
        logger.debug("cluster QC warm-up failed: %s", exc)


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
    enforce_query_limits(body.query, body.history, settings)
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")
    try:
        query = body.query.strip()
        kwargs = await _prepare(query, body, dm, settings)
        await _warm_cluster_qc(adaptor, kwargs)
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
    enforce_query_limits(body.query, body.history, settings)
    adaptor = await dm.get_or_load_dataset(dataset_id)
    if adaptor is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")

    query = body.query.strip()

    async def event_generator():
        try:
            kwargs = await _prepare(query, body, dm, settings)
            await _warm_cluster_qc(adaptor, kwargs)
            async for event in stream_answer(
                query, adaptor, settings.DEEPINFRA_API_KEY, **kwargs
            ):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as exc:
            logger.error("assistant stream failed for %s: %s", dataset_id, exc, exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'error': str(exc)})}\n\n"

    return StreamingResponse(
        keepalive(event_generator()),
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
    enforce_query_limits(body.query, body.history, settings)
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
    enforce_query_limits(body.query, body.history, settings)
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
        keepalive(event_generator()),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
