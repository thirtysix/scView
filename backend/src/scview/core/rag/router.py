"""Query intent classifier — decide which knowledge a question needs, so the
co-pilot doesn't waste compute (embeddings + vector search) when it isn't needed.

Four sources (a question may need a minimal subset):
  - ``app``        — scView itself / the dataset library / navigation
                     ("what datasets do we have?", "how do I load data?").
  - ``data``       — the user's CURRENTLY LOADED dataset + its analysis/results
                     ("what cell types are in my data?", "markers of this cluster").
  - ``tutorials``  — single-cell methods / how-to / parameters (RAG corpus).
  - ``literature`` — biology / evidence from research abstracts (RAG corpus).

Only ``tutorials``/``literature`` trigger RAG retrieval; ``app``/``data`` are
answered from local state. Uses a cheap LLM classifier with a keyword fallback,
and takes the current tab as a hint (e.g. on the Data tab, "what do we have?"
leans ``app``).
"""

from __future__ import annotations

import json
import logging

from pydantic import BaseModel

logger = logging.getLogger(__name__)

SOURCES = ("app", "data", "tutorials", "literature")
CORPORA = ("tutorials", "literature")  # the subset that needs RAG retrieval

_APP_HINTS = (
    "what datasets", "datasets do we", "data do we have", "my datasets", "list datasets",
    "available datasets", "load data", "loading data", "upload", "import data", "add data",
    "how do i use", "what can this", "what can scview", "what does scview", "which tab",
    "what tab", "navigate", "get started", "how do i load", "how do i open", "where do i",
)
_DATA_HINTS = (
    "my data", "this dataset", "this cluster", "my cluster", "what cell types", "how many cells",
    "what did i run", "what was run", "my results", "my markers", "my umap", "this gene",
    "these cells", "current dataset", "annotation", "what's in my", "whats in my",
)
_METHODS_HINTS = (
    "how do i", "how to", "why ", "should i", "what does", "parameter", "resolution",
    "normalize", "normalisation", "normalization", "log transform", "scale", "scaling",
    "pca", "umap", "t-sne", "tsne", "neighbors", "n_neighbors", "batch correction",
    "harmony", "integrate", "integration", "qc", "quality control", "filter", "threshold",
    "doublet", "highly variable", "hvg", "workflow", "pipeline", "step", "recommend",
    "cluster the", "clustering", "interpret", "read this", "dot plot", "violin", "heatmap",
)
_LITERATURE_HINTS = (
    "marker", "express", "expression of", "pathway", "disease", "lupus", "cancer",
    "tumor", "immune", "what is known", "literature", "study", "studies",
    "paper", "associated with", "role of", "function of", "known to", "implicated",
    "biology", "mechanism", "signature", "interferon", "cytokine", "receptor",
)


class RouteResult(BaseModel):  # kept for the corpora-level heuristic + tests
    corpora: list[str]
    reason: str


DATA_FACETS = ("identity", "groups", "markers", "enrichment")

_FACET_HINTS = {
    "identity": (
        "article", "paper", "publication", "citation", "doi", "source", "origin",
        "where is this from", "what is this dataset", "what's this dataset", "who made",
    ),
    "markers": ("marker", "gene", "express", "defines", "characteri", "signature", "top genes"),
    "groups": (
        "how many", "composition", "proportion", "count", "cell type", "cell types",
        "cluster", "donor", "condition", "sample", "group", "breakdown", "fraction",
    ),
    "enrichment": ("enrichment", "pathway", "gsea", "go term", "ontology"),
}


class Intent(BaseModel):
    sources: list[str]
    reason: str
    data_facets: list[str] = []  # when 'data': minimal facets needed (empty = all)


def _infer_facets(query: str) -> list[str]:
    """Heuristic: which dataset facets does a 'data' question need? Empty = all."""
    q = query.lower()
    return [f for f, hints in _FACET_HINTS.items() if any(h in q for h in hints)]


def heuristic_route(query: str) -> RouteResult:
    """Keyword routing between the two RAG corpora. Defaults to both when ambiguous."""
    q = query.lower()
    methods = any(h in q for h in _METHODS_HINTS)
    literature = any(h in q for h in _LITERATURE_HINTS)
    if methods and not literature:
        return RouteResult(corpora=["tutorials"], reason="methods/how-to phrasing")
    if literature and not methods:
        return RouteResult(corpora=["literature"], reason="biological/evidential phrasing")
    return RouteResult(corpora=["tutorials", "literature"], reason="mixed/ambiguous → both")


def heuristic_intent(query: str, view_context: dict | None = None) -> Intent:
    """Keyword + tab-hint intent classification (no LLM)."""
    q = query.lower()
    panel = ((view_context or {}).get("panel") or "").lower()
    sources: list[str] = []

    app = any(h in q for h in _APP_HINTS)
    # On the Data tab, generic "what do we have / what's available / list" leans app.
    if (
        panel.startswith("data")
        and "my data" not in q
        and any(k in q for k in ("have", "available", "loaded", "list", "dataset", "data"))
    ):
        app = True
    if app:
        sources.append("app")

    if any(h in q for h in _DATA_HINTS):
        sources.append("data")
    if any(h in q for h in _METHODS_HINTS):
        sources.append("tutorials")
    if any(h in q for h in _LITERATURE_HINTS):
        sources.append("literature")

    if not sources:
        sources = ["data"]  # cheap default: ground in the loaded dataset, no RAG
    # de-dup preserving order
    seen: set[str] = set()
    sources = [s for s in sources if not (s in seen or seen.add(s))]
    facets = _infer_facets(query) if "data" in sources else []
    return Intent(sources=sources, reason="heuristic", data_facets=facets)


_CLASSIFY_SYSTEM = """\
You classify a user's question inside the scView single-cell RNA-seq app to decide \
which knowledge sources to use — picking the MINIMAL set so we don't waste compute. \
Sources:
- "app": about scView itself or the user's dataset library / navigation \
(e.g. "what datasets do we have?", "how do I load data?", "what can this tool do?").
- "data": about the user's CURRENTLY LOADED dataset — its analysis/results AND its \
own identity/source (e.g. "what cell types are in my data?", "markers of this \
cluster", "what steps ran?", "what paper/publication is THIS dataset from?").
- "tutorials": single-cell METHODS / how-to / parameter / interpretation questions \
(e.g. "why log-normalize?", "what clustering resolution?").
- "literature": BIOLOGY / evidence answerable from research abstracts \
(e.g. "what marks pDCs?", "is the interferon signature linked to lupus?").
Only include "tutorials"/"literature" when the question truly needs external \
knowledge — those run expensive retrieval. When you include "data", also return \
"data_facets": the MINIMAL subset of ["identity","groups","markers","enrichment"] \
the question needs (e.g. "what paper is this?" → ["identity"]; "what marks cluster \
3?" → ["markers"]; leave empty only if truly everything is needed). Respond ONLY \
as JSON: {"sources": [...], "data_facets": [...], "reason": "..."}."""


async def classify_intent(
    query: str, view_context: dict | None, api_key: str, model: str
) -> Intent:
    """Classify which knowledge sources a question needs. Falls back to heuristics."""
    if not api_key:
        return heuristic_intent(query, view_context)
    panel = (view_context or {}).get("panel") or "unknown"
    try:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=api_key, base_url="https://api.deepinfra.com/v1/openai")
        resp = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _CLASSIFY_SYSTEM},
                {"role": "user", "content": f'(current tab: "{panel}")\n{query}'},
            ],
            temperature=0.0,
            max_tokens=80,
        )
        raw = (resp.choices[0].message.content or "").strip()
        if "{" in raw:
            raw = raw[raw.find("{"): raw.rfind("}") + 1]
        data = json.loads(raw)
        sources = [s for s in data.get("sources", []) if s in SOURCES]
        if sources:
            facets = [f for f in (data.get("data_facets") or []) if f in DATA_FACETS]
            return Intent(
                sources=sources,
                reason=str(data.get("reason", "llm-classified")),
                data_facets=facets if "data" in sources else [],
            )
    except Exception as exc:
        logger.warning("intent classify LLM failed (%s); using heuristic", exc)
    return heuristic_intent(query, view_context)
