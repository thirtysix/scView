"""Grounded AI co-pilot — chat about *this* dataset's analysis and results.

This is scView's RAG-style assistant, modelled on the WntHub pattern (RAG +
experimental-data context + user query → grounded, cited answer), but here the
"experimental data context" is the user's *own* single-cell analysis. Today it
grounds answers in three in-app sources scView already has:

  1. the dataset's **preprocessing state** (the deterministic assessor),
  2. **what was actually done** (the provenance recipe), and
  3. the dataset's **results** (cluster sizes, top markers per cluster,
     embeddings, enrichment).

A fourth source — a **methods/literature RAG over pgvector** — is a planned
extension (see ``docs/AI_ASSISTANT.md`` §3.6). The context builder exposes a
clean hook (``extra_context``) so that retrieval layer can be added later
without changing the endpoint or the prompt assembly. See
``LITERATURE_RAG_HOOK`` below.

Design principles (mirroring ``llm_advisor.py``):
- **Grounded, not free-floating.** The model is told to answer *only* from the
  provided facts and to say when something isn't in the data.
- **Cited.** Every claim should reference a fact tag (e.g. ``[result:markers:B]``,
  ``[provenance:clustering]``) drawn from the assembled context.
- **Graceful degradation.** With no API key (or on failure) we return a
  deterministic, templated summary of the grounded facts — never a hard error,
  never a hallucination.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel

from scview.core.assessor import assess_preprocessing
from scview.core import provenance

logger = logging.getLogger(__name__)

# Reuse the same provider/model as the preprocessing advisor for consistency.
DEEPINFRA_BASE_URL = "https://api.deepinfra.com/v1/openai"
DEFAULT_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# How much result detail to put in the prompt (keep the context bounded).
_MAX_MARKER_GROUPS = 16
_MAX_MARKER_GENES = 8
_MAX_CATEGORICAL_UNIQUE = 30
_MAX_CATEGORICAL_COLUMNS = 8


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class ChatSource(BaseModel):
    """A grounding fact the answer can cite."""

    kind: str  # "dataset" | "preprocessing" | "provenance" | "result"
    ref: str   # short citable tag, e.g. "result:markers:B" or "provenance:clustering"
    detail: str


class ChatMessage(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[ChatSource]
    grounded: bool          # True if an LLM produced the answer; False = templated fallback
    raw_response: str = ""  # raw LLM text, for transparency
    route: list[str] = []   # which knowledge sources were consulted (app/data/tutorials/literature)


# ---------------------------------------------------------------------------
# Grounding context — assemble facts from the loaded dataset
# ---------------------------------------------------------------------------


def build_grounding_context(adaptor) -> tuple[str, list[ChatSource]]:
    """Assemble a bounded, structured snapshot of the dataset's analysis state.

    Returns ``(context_text, sources)`` where ``context_text`` is the prompt
    block and ``sources`` are the citable facts behind it.
    """
    adata = adaptor.adata
    lines: list[str] = []
    sources: list[ChatSource] = []

    # --- 1. Dataset summary -------------------------------------------------
    n_cells = adaptor.n_cells()
    n_genes = adaptor.n_genes()
    embeddings = [e["name"] for e in adaptor.available_embeddings()]
    lines.append("## Dataset")
    lines.append(f"- Cells: {n_cells:,} | Genes: {n_genes:,}")
    lines.append(f"- Embeddings present: {', '.join(embeddings) if embeddings else 'none'}")
    sources.append(ChatSource(
        kind="dataset", ref="dataset:summary",
        detail=f"{n_cells:,} cells × {n_genes:,} genes; embeddings: "
               f"{', '.join(embeddings) or 'none'}",
    ))

    # --- 2. Preprocessing state (the assessor) ------------------------------
    try:
        state = assess_preprocessing(adata).model_dump()
        done, notdone = [], []
        for step, st in state.items():
            if isinstance(st, dict) and "done" in st:
                (done if st.get("done") else notdone).append(step)
        lines.append("\n## Preprocessing state (deterministic assessment)")
        lines.append(f"- Done: {', '.join(done) if done else 'none'}")
        lines.append(f"- Not done: {', '.join(notdone) if notdone else 'none'}")
        # surface a few high-signal details verbatim
        for step in ("normalization", "clustering", "batch_correction"):
            st = state.get(step)
            if isinstance(st, dict) and st.get("details"):
                lines.append(f"- {step}: {st['details']}")
        sources.append(ChatSource(
            kind="preprocessing", ref="preprocessing:state",
            detail=f"done={done}; not_done={notdone}",
        ))
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("assistant: assessment failed: %s", exc)

    # --- 3. Provenance — what was actually done -----------------------------
    try:
        if provenance.has_provenance(adata):
            steps = provenance.recipe(adata)
            if steps:
                lines.append("\n## Provenance — recorded analysis steps (in order)")
                for s in steps:
                    params = s.get("params") or {}
                    pstr = ", ".join(f"{k}={v}" for k, v in list(params.items())[:6])
                    lines.append(f"- {s['step']}" + (f" ({pstr})" if pstr else ""))
                sources.append(ChatSource(
                    kind="provenance", ref="provenance:recipe",
                    detail=" → ".join(s["step"] for s in steps),
                ))
            issues = provenance.reconcile(adata)
            if issues:
                lines.append("- ⚠ Provenance/data mismatches: " + "; ".join(issues))
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("assistant: provenance read failed: %s", exc)

    # --- 4. Results — categorical groupings + top markers + enrichment ------
    # 4a. Low-cardinality categorical obs columns (cell types, condition, batch)
    cat_cols = []
    for info in adaptor.obs_columns_info():
        nun = info.get("n_unique")
        if nun is not None and 1 < nun <= _MAX_CATEGORICAL_UNIQUE and "values" in info:
            cat_cols.append(info["name"])
    if cat_cols:
        lines.append("\n## Cell groupings (obs columns)")
        for col in cat_cols[:_MAX_CATEGORICAL_COLUMNS]:
            try:
                summary = adaptor.get_obs_summary(col)
                top = sorted(summary.items(), key=lambda kv: -kv[1])
                shown = ", ".join(f"{k} ({v:,})" for k, v in top[:_MAX_CATEGORICAL_UNIQUE])
                lines.append(f"- **{col}**: {shown}")
                sources.append(ChatSource(
                    kind="result", ref=f"result:groups:{col}",
                    detail=f"{col} = {shown}",
                ))
            except Exception as exc:  # pragma: no cover
                logger.debug("assistant: obs summary failed for %s: %s", col, exc)

    # 4b. Top markers per group, for each column that has computed markers
    try:
        marker_cols = adaptor.marker_columns()
    except Exception:  # pragma: no cover
        marker_cols = []
    for mcol in marker_cols[:2]:  # at most the two primary marker columns
        df = adaptor.get_markers(column=mcol, n_genes=_MAX_MARKER_GENES)
        if df is None or df.empty:
            continue
        lines.append(f"\n## Top marker genes per group (column: {mcol})")
        for group, sub in df.groupby("group", sort=False):
            genes = list(sub["gene"])[:_MAX_MARKER_GENES]
            lines.append(f"- **{group}**: {', '.join(genes)}")
            sources.append(ChatSource(
                kind="result", ref=f"result:markers:{group}",
                detail=f"{group} top markers: {', '.join(genes)}",
            ))
        # group cap to keep the prompt bounded
        if df["group"].nunique() > _MAX_MARKER_GROUPS:
            lines.append(f"- (… {df['group'].nunique()} groups total; showing first "
                         f"{_MAX_MARKER_GROUPS})")

    # 4c. Enrichment results, if present in uns
    try:
        enr_keys = [k for k in adata.uns if str(k).startswith("enrichment__")]
        if enr_keys:
            cols = sorted({k.split("__", 1)[1] for k in enr_keys})
            lines.append("\n## Pathway enrichment computed for: " + ", ".join(cols))
            sources.append(ChatSource(
                kind="result", ref="result:enrichment",
                detail="enrichment available for: " + ", ".join(cols),
            ))
    except Exception:  # pragma: no cover
        pass

    return "\n".join(lines), sources


# ---------------------------------------------------------------------------
# App context — the dataset library + what scView can do (no RAG needed)
# ---------------------------------------------------------------------------

_FEATURE_GUIDE = (
    "scView is a browser tool for single-cell RNA-seq. Tabs: Data (import/manage "
    "datasets), Data Assessment (QC plots + AI-guided pipeline: QC → normalize → HVG → "
    "PCA → batch correction → clustering → UMAP → markers → enrichment), Visualizations "
    "(embeddings), Unified View (linked scatter + markers/expression/gene-sets/enrichment "
    "+ violin), Observations, Gene Expression, Gene Sets & Enrichment, Marker Genes, "
    "Trajectory, History (provenance: what was done + edit-&-re-run), and this AI Co-pilot."
)


def build_app_context(
    datasets: list[dict] | None, view_context: dict | None = None
) -> tuple[str, list[ChatSource]]:
    """Assemble app-level facts: the dataset library + a feature/navigation guide.

    Answers questions like "what datasets do we have?" / "how do I load data?"
    without touching the RAG corpora.
    """
    lines = ["## scView app", f"- {_FEATURE_GUIDE}"]
    sources = [ChatSource(kind="app", ref="app:features", detail="scView tabs and capabilities")]

    datasets = datasets or []
    lines.append(f"\n## Dataset library ({len(datasets)} dataset(s) available)")
    if not datasets:
        lines.append("- (no datasets loaded yet — use the Data tab to import one)")
    for d in datasets[:50]:
        name = d.get("name", d.get("id", "?"))
        nc, ng = d.get("n_cells"), d.get("n_genes")
        dims = f"{nc:,}×{ng:,}" if isinstance(nc, int) and isinstance(ng, int) else "unprocessed"
        embs = d.get("available_embeddings") or []
        status = d.get("status", "")
        lines.append(
            f"- **{name}** ({dims}"
            + (f"; embeddings: {', '.join(embs)}" if embs else "")
            + (f"; {status}" if status else "")
            + ")"
        )
    sources.append(ChatSource(
        kind="app", ref="app:library",
        detail=f"{len(datasets)} dataset(s): " + ", ".join(d.get("name", "?") for d in datasets[:20]),
    ))
    return "\n".join(lines), sources


# ---------------------------------------------------------------------------
# Prompt assembly + LLM call
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are scView's single-cell RNA-seq analysis co-pilot. You answer the user's \
questions about THEIR dataset: its analysis, the steps that were run, and the \
results — and about scRNA-seq analysis in general.

Rules:
- Ground every factual claim in the DATA CONTEXT provided below. When you state \
a fact from it, cite the relevant tag in square brackets, e.g. \
"B cells are marked by CD79A and MS4A1 [result:markers:B]" or \
"the data was normalized [preprocessing:state]".
- If the answer is not in the context, say so plainly and, if useful, give \
general scRNA-seq guidance clearly labelled as general (not from their data).
- Be concise and specific. Prefer the user's actual cluster names, gene symbols, \
and counts over generic statements.
- Do NOT invent clusters, genes, counts, or steps that are not in the context. \
Do not give clinical or diagnostic advice.
- If the user says "this cluster", "this gene", or "here", resolve it from the \
"What the user is currently viewing" section when present.
- Format with Markdown: use a bullet or numbered list when enumerating items \
(datasets, clusters, genes, steps), **bold** for key terms, and `code` for gene \
symbols/parameters. Keep answers concise."""


def _build_user_message(query: str, context: str) -> str:
    return (
        "# DATA CONTEXT (facts about the user's dataset — cite these tags)\n"
        f"{context}\n\n"
        "# QUESTION\n"
        f"{query}\n\n"
        "Answer grounded in the DATA CONTEXT, citing the bracketed tags."
    )


def _format_view_context(vc: dict | None) -> str:
    """Format the user's current on-screen view so deictic questions resolve
    ("what is *this* cluster?", "explain *this* gene")."""
    if not vc:
        return ""
    lines = ["## What the user is currently viewing"]
    if vc.get("panel"):
        lines.append(f"- Panel: {vc['panel']}")
    if vc.get("color_by"):
        lines.append(f"- Scatter coloured by: {vc['color_by']}")
    hl = vc.get("highlighted")
    if isinstance(hl, dict) and hl.get("value"):
        lines.append(f"- Highlighted group: {hl.get('column', 'group')} = {hl['value']}")
    if vc.get("overlay"):
        lines.append(f"- Gene/expression overlay: {vc['overlay']}")
    return "\n".join(lines) if len(lines) > 1 else ""


async def answer_query(
    query: str,
    adaptor,
    api_key: str,
    history: list[ChatMessage] | None = None,
    *,
    extra_context: str = "",  # LITERATURE_RAG_HOOK: retrieved doc/abstract chunks
    extra_sources: list[ChatSource] | None = None,  # citations behind extra_context
    app_context: str = "",  # dataset library + feature guide
    app_sources: list[ChatSource] | None = None,
    view_context: dict | None = None,  # what the user is currently looking at
    include_data_grounding: bool = True,  # assemble the loaded dataset's facts
    route: list[str] | None = None,  # which sources the classifier chose
    model: str = DEFAULT_MODEL,
) -> ChatResponse:
    """Answer a question, grounded in whichever sources the intent classifier chose.

    ``include_data_grounding`` controls whether the loaded dataset's facts are
    assembled (skipped for pure app/methods questions). ``app_context`` carries the
    dataset library + feature guide. ``extra_context`` carries retrieved RAG chunks
    (literature/tutorials). ``view_context`` describes the current on-screen view so
    deictic questions resolve. Everything degrades gracefully when unset.
    """
    blocks: list[str] = []
    sources: list[ChatSource] = []
    if include_data_grounding and adaptor is not None:
        data_context, data_sources = build_grounding_context(adaptor)
        blocks.append(data_context)
        sources.extend(data_sources)
    if app_context:
        blocks.insert(0, app_context)
        sources = [*(app_sources or []), *sources]
    if extra_context:
        blocks.insert(0, extra_context)
        sources = [*(extra_sources or []), *sources]
    context = "\n\n".join(b for b in blocks if b)
    view_note = _format_view_context(view_context)
    if view_note:
        context = view_note + "\n\n" + context

    route = route or []
    if not api_key:
        logger.info("assistant: no API key, returning templated fallback")
        return _fallback_answer(query, context, sources, route)

    try:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=api_key, base_url=DEEPINFRA_BASE_URL)
        messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        for m in (history or [])[-6:]:  # keep the last few turns
            if m.role in ("user", "assistant") and m.content:
                messages.append({"role": m.role, "content": m.content})
        messages.append({"role": "user", "content": _build_user_message(query, context)})

        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
            max_tokens=1024,
        )
        answer = (response.choices[0].message.content or "").strip()
        if not answer:
            return _fallback_answer(query, context, sources, route)
        return ChatResponse(
            answer=answer, sources=sources, grounded=True, raw_response=answer, route=route
        )
    except Exception as exc:
        logger.warning("assistant: LLM call failed (%s); using fallback", exc)
        return _fallback_answer(query, context, sources, route)


def _fallback_answer(
    query: str, context: str, sources: list[ChatSource], route: list[str] | None = None
) -> ChatResponse:
    """Deterministic, no-LLM answer: surface the grounded facts honestly."""
    answer = (
        "The AI co-pilot's language model is not configured, so here is a direct "
        "summary of the grounded facts about your dataset. Set `DEEPINFRA_API_KEY` "
        "to enable conversational, cited answers.\n\n"
        f"{context}"
    )
    return ChatResponse(
        answer=answer, sources=sources, grounded=False, raw_response="", route=route or []
    )
