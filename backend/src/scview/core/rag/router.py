"""Query router — decide which RAG corpora a question should retrieve from.

Two corpora answer different question types:
  - ``tutorials``  — methods / how-to / parameter / interpretation questions.
  - ``literature`` — biological / evidential ("what's known about …") questions.

In-app grounding (the user's own provenance + results) is always added by the
co-pilot regardless of routing, so the router only chooses the *external* corpora
to consult. Uses a cheap LLM classifier with a deterministic keyword fallback.
"""

from __future__ import annotations

import json
import logging

from pydantic import BaseModel

logger = logging.getLogger(__name__)

CORPORA = ("tutorials", "literature")

_METHODS_HINTS = (
    "how do i", "how to", "why ", "should i", "what does", "parameter", "resolution",
    "normalize", "normalisation", "normalization", "log transform", "scale", "scaling",
    "pca", "umap", "t-sne", "tsne", "neighbors", "n_neighbors", "batch correction",
    "harmony", "integrate", "integration", "qc", "quality control", "filter", "threshold",
    "doublet", "highly variable", "hvg", "workflow", "pipeline", "step", "recommend",
    "cluster the", "clustering", "interpret", "read this", "dot plot", "violin", "heatmap",
)
_LITERATURE_HINTS = (
    "marker", "express", "expression of", "gene", "pathway", "disease", "lupus", "cancer",
    "tumor", "immune", "cell type", "what is known", "literature", "study", "studies",
    "paper", "associated with", "role of", "function of", "known to", "implicated",
    "biology", "mechanism", "signature", "interferon", "cytokine", "receptor",
)


class RouteResult(BaseModel):
    corpora: list[str]
    reason: str


def heuristic_route(query: str) -> RouteResult:
    """Keyword-based routing (no LLM). Defaults to both when ambiguous."""
    q = query.lower()
    methods = any(h in q for h in _METHODS_HINTS)
    literature = any(h in q for h in _LITERATURE_HINTS)
    if methods and not literature:
        return RouteResult(corpora=["tutorials"], reason="methods/how-to phrasing")
    if literature and not methods:
        return RouteResult(corpora=["literature"], reason="biological/evidential phrasing")
    # both, or neither → consult both (cheap, and the reranker sorts it out)
    return RouteResult(corpora=["tutorials", "literature"], reason="mixed/ambiguous → both")


_ROUTER_SYSTEM = """\
You route a single-cell RNA-seq user question to knowledge sources. Choose any of:
- "tutorials": methods / how-to / parameter / interpretation questions (e.g. "why \
log-normalize?", "what clustering resolution?", "how do I read a dot plot?").
- "literature": biology / evidence questions answerable from research abstracts \
(e.g. "what marks pDCs?", "is the interferon signature linked to lupus?").
Pick one or both. Respond ONLY as JSON: {"corpora": ["tutorials"|"literature", ...]}."""


async def route(query: str, api_key: str, model: str) -> RouteResult:
    """Classify which corpora to retrieve from. Falls back to heuristics."""
    if not api_key:
        return heuristic_route(query)
    try:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=api_key, base_url="https://api.deepinfra.com/v1/openai")
        resp = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _ROUTER_SYSTEM},
                {"role": "user", "content": query},
            ],
            temperature=0.0,
            max_tokens=64,
        )
        raw = (resp.choices[0].message.content or "").strip()
        if "{" in raw:
            raw = raw[raw.find("{"): raw.rfind("}") + 1]
        data = json.loads(raw)
        corpora = [c for c in data.get("corpora", []) if c in CORPORA]
        if corpora:
            return RouteResult(corpora=corpora, reason="llm-classified")
    except Exception as exc:
        logger.warning("RAG router LLM failed (%s); using heuristic", exc)
    return heuristic_route(query)
