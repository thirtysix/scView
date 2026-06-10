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

import json
import logging
import re

from pydantic import BaseModel, Field

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


class AssistantAction(BaseModel):
    """An allow-listed UI action the co-pilot can request the frontend to perform.

    Safe, reversible view/navigation actions: set_color_by, highlight_cluster,
    open_panel, set_embedding, set_subtab, set_groupby, clear_highlight,
    clear_overlay, show_gene.
    """

    type: str                    # one of the allow-listed action types
    column: str | None = None    # obs column (set_color_by, highlight_cluster, set_groupby)
    value: str | None = None     # group value (highlight_cluster)
    panel: str | None = None     # panel id (open_panel)
    embedding: str | None = None  # obsm key (set_embedding)
    subtab: str | None = None    # Unified View subtab (set_subtab)
    gene: str | None = None      # gene symbol (show_gene)
    label: str = ""              # human-readable confirmation
    # Mutating actions (run a pipeline step) — confirmation-gated:
    requires_confirm: bool = False  # if True, the UI shows a Confirm button (nothing auto-runs)
    step: str | None = None      # pipeline step to run
    params: dict = Field(default_factory=dict)  # PipelineParams overrides for the step
    advisory: str = ""           # overwrite / destructive note for the user
    estimate: str = ""           # rough processing-time estimate


class ChatResponse(BaseModel):
    answer: str
    sources: list[ChatSource]
    grounded: bool          # True if an LLM produced the answer; False = templated fallback
    raw_response: str = ""  # raw LLM text, for transparency
    route: list[str] = []   # which knowledge sources were consulted (app/data/tutorials/literature)
    followups: list[str] = []  # suggested next questions
    actions: list[AssistantAction] = []  # allow-listed UI actions to execute (NL commands)
    model: str | None = None  # which LLM produced the answer (transparency); None = no LLM


# ---------------------------------------------------------------------------
# Grounding context — assemble facts from the loaded dataset
# ---------------------------------------------------------------------------


def _uns_str(adata, key: str, limit: int = 600) -> str:
    """Read a free-text `uns` value that may be a str, list, or ndarray of one."""
    if key not in adata.uns:
        return ""
    val = adata.uns[key]
    try:
        while not isinstance(val, str) and hasattr(val, "__len__") and len(val) > 0:
            val = val[0]
    except Exception:  # pragma: no cover
        return ""
    s = str(val).strip()
    return s[:limit] + "…" if len(s) > limit else s


_ANNO_SUFFIX = "_celltypeAnno"


def _celltype_cluster_map(adata, acol: str) -> dict[str, str]:
    """Map cluster -> cell type for a cell-type annotation column, so the co-pilot
    can answer "what is this cluster?". Prefers the recorded per-cluster mapping;
    otherwise takes the majority cell type per cluster of the source grouping."""
    um = adata.uns.get(f"{acol}_llm_mapping")
    if isinstance(um, dict) and um:
        return {str(k): str(v) for k, v in um.items()}
    if acol.endswith(_ANNO_SUFFIX):
        grouping = acol[: -len(_ANNO_SUFFIX)]
    else:
        grouping = adata.uns.get("scview_active_clustering")
    if not grouping or grouping not in adata.obs.columns or acol not in adata.obs.columns:
        return {}
    try:
        import pandas as pd

        ct = pd.crosstab(adata.obs[grouping], adata.obs[acol])
        return {str(idx): str(ct.loc[idx].idxmax()) for idx in ct.index}
    except Exception:  # pragma: no cover
        return {}


def build_grounding_context(
    adaptor, facets: set[str] | None = None
) -> tuple[str, list[ChatSource]]:
    """Assemble a bounded, structured snapshot of the dataset's analysis state.

    Returns ``(context_text, sources)`` where ``context_text`` is the prompt
    block and ``sources`` are the citable facts behind it.
    """
    adata = adaptor.adata
    lines: list[str] = []
    sources: list[ChatSource] = []

    def want(facet: str) -> bool:
        # facets=None → include everything (back-compat); else only the requested
        # facets. The dataset summary is always included (cheap + always useful).
        return facets is None or facet in facets

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

    # --- 1b. Dataset identity / source (so "what's the paper?" can be answered) ---
    ident: list[str] = []
    prov_src: dict = {}
    if want("identity"):
        has_pub = False
        for key, label in (("about_title", "Title"), ("about_short_title", "Short title"),
                           ("about_readme", "About")):
            val = _uns_str(adata, key)
            if val:
                ident.append(f"- {label}: {val}")
                has_pub = True
        try:
            prov_src = provenance.read_provenance(adata).get("source", {}) or {}
        except Exception:  # pragma: no cover
            prov_src = {}
        for k in ("origin", "original_filename", "format"):
            if prov_src.get(k):
                ident.append(f"- {k.replace('_', ' ').title()}: {prov_src[k]}")
        if not has_pub:
            # No embedded study/citation — say so explicitly so the answer is
            # helpful ("user-uploaded, no publication metadata") not a flat "no info".
            ident.append(
                "- No embedded publication/study metadata: this appears to be a "
                "user-uploaded dataset; its source/citation is not stored in the data "
                "and could be added by the user."
            )
    if ident:
        lines.append("\n## Dataset identity & source "
                     "(use to answer questions about the dataset's paper/origin)")
        lines.extend(ident)
        sources.append(ChatSource(
            kind="dataset", ref="dataset:source",
            detail=(_uns_str(adata, "about_title") or prov_src.get("original_filename") or "source"),
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
    if want("groups"):
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

    # 4a-bis. Cell-type annotations: cluster -> cell type. Compact and high-value for
    # "what is this cluster?", so it's always included (not facet-gated).
    anno_cols = [
        info["name"]
        for info in adaptor.obs_columns_info()
        if (info["name"] == "cell_type" or info["name"].endswith(_ANNO_SUFFIX))
        and 1 < (info.get("n_unique") or 0) <= _MAX_CATEGORICAL_UNIQUE
    ]
    for acol in anno_cols[:3]:
        mapping = _celltype_cluster_map(adata, acol)
        if not mapping:
            continue
        src = acol[: -len(_ANNO_SUFFIX)] if acol.endswith(_ANNO_SUFFIX) else "the clustering"
        lines.append(f"\n## Cell-type annotation '{acol}' — {src} cluster → cell type")
        for cl, ct in list(mapping.items())[:_MAX_CATEGORICAL_UNIQUE]:
            lines.append(f"- cluster **{cl}** = {ct}")
        sources.append(ChatSource(
            kind="result", ref=f"result:celltypes:{acol}",
            detail=f"{acol}: " + "; ".join(f"{k}={v}" for k, v in list(mapping.items())[:12]),
        ))

    # 4b. Top markers per group, for each column that has computed markers
    marker_cols = []
    if want("markers"):
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
        enr_keys = (
            [k for k in adata.uns if str(k).startswith("enrichment__")]
            if want("enrichment")
            else []
        )
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
    "scView is a browser tool for single-cell RNA-seq. "
    "IMPORTING DATA: open the Data tab and drag files onto it, or use the file picker. "
    "Import is by uploading local files only — there is no URL/remote import. "
    "Supported formats: AnnData .h5ad; 10x Genomics MTX (matrix.mtx + barcodes.tsv + "
    "features/genes.tsv, optionally .gz) and 10x HDF5 (.h5); .loom; .zarr; dense CSV/TSV "
    "(optionally .gz); Seurat .rds (converted to .h5ad by the R converter service); and "
    "nf-core/scrnaseq mtx_conversions outputs (*.h5ad). A guided flow bundles multi-file "
    "inputs (e.g. the three 10x MTX files) and can merge several samples into one dataset. "
    "Each imported dataset then appears in the Data-tab library to open or manage. "
    "TABS: Data (import/manage datasets), Data Assessment (QC plots + AI-guided pipeline: "
    "QC → normalize → HVG → PCA → batch correction → clustering → UMAP → markers → "
    "enrichment), Unified View (linked scatter + markers/expression/gene-sets/enrichment + "
    "violin), Observations, Gene Expression, Gene Sets & Enrichment, Marker Genes, "
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
symbols/parameters. Keep answers concise.
- For "what study/paper is this?" when no publication metadata is embedded, don't \
just say "no information" — explain it looks like a user-uploaded dataset whose \
source isn't stored in the data, and that they can add a citation."""


# ---------------------------------------------------------------------------
# Proactive insight (deterministic "I notice…" on dataset open)
# ---------------------------------------------------------------------------


class DatasetInsight(BaseModel):
    """A one-line, next-step nudge surfaced when a dataset opens."""

    insight: str
    question: str | None = None  # a click-to-ask follow-up for the co-pilot
    severity: str = "info"  # "info" | "suggestion"
    polished: bool = False  # True if an LLM rephrased the deterministic text


_CONDITION_HINT_RE = re.compile(
    r"condition|stim|treat|disease|status|genotype|batch|donor|patient|"
    r"sample|orig\.?ident|group|timepoint|sex|tissue",
    re.I,
)

_MT_COL_RE = re.compile(r"pct_counts_mt|percent[._]?mt|pct_counts_mito|percent[._]?mito", re.I)


def _looks_like_celltype_col(name: str) -> bool:
    n = name.lower()
    return "celltype" in n or "cell_type" in n or name.endswith(_ANNO_SUFFIX)


def build_insight(adaptor) -> DatasetInsight:
    """One deterministic "I notice…" line when a dataset opens — what to do next.

    No LLM call: cheap, reproducible, and safe to run on every load. Picks the
    single most useful nudge by walking the preprocessing state in pipeline order
    (un-normalized → doublets → batch split → cluster → annotate → done)."""
    adata = adaptor.adata
    n = int(adata.n_obs)
    try:
        state = assess_preprocessing(adata)
    except Exception:  # pragma: no cover - defensive
        return DatasetInsight(insight=f"{n:,} cells loaded.")

    obs = adata.obs

    # 1. Raw / un-normalized counts → start preprocessing.
    if not state.normalization.done or not state.log_transform.done:
        return DatasetInsight(
            insight=(
                f"This looks like {n:,} cells of raw counts — start with "
                "normalization and a log-transform so the downstream steps are valid."
            ),
            question="How should I preprocess this dataset?",
            severity="suggestion",
        )

    # 2. Heavy doublet load.
    if "predicted_doublet" in obs.columns:
        try:
            frac = float(obs["predicted_doublet"].astype(bool).mean())
            if frac >= 0.08:
                return DatasetInsight(
                    insight=(
                        f"About {frac:.0%} of cells are flagged as predicted doublets "
                        "— consider filtering them before clustering."
                    ),
                    question="Should I filter doublets?",
                    severity="suggestion",
                )
        except Exception:  # pragma: no cover
            pass

    # 2b. QC anomaly: high mitochondrial content (stressed / dying cells).
    for col in obs.columns:
        if not _MT_COL_RE.search(col):
            continue
        try:
            import numpy as np

            med = float(np.nanmedian(obs[col].to_numpy(dtype="float64")))
        except Exception:  # pragma: no cover
            break
        if med >= 15:
            return DatasetInsight(
                insight=(
                    f"Median mitochondrial content is high (~{med:.0f}% per cell) — that "
                    "often marks stressed or dying cells; consider a stricter MT% filter."
                ),
                question="Should I filter high-MT cells?",
                severity="suggestion",
            )
        break

    # 3. Strong split by a condition/batch-like column, not yet integrated.
    if not state.batch_correction.done:
        for col in obs.columns:
            if _looks_like_celltype_col(col) or not _CONDITION_HINT_RE.search(col):
                continue
            try:
                vc = obs[col].astype("category").value_counts()
            except Exception:  # pragma: no cover
                continue
            k = int((vc > 0).sum())
            if 2 <= k <= 8 and vc.iloc[: min(k, 4)].min() >= max(20, 0.02 * n):
                vals = ", ".join(map(str, vc.index[:4]))
                return DatasetInsight(
                    insight=(
                        f"Your cells split by {col} ({vals}) — if that's a batch or "
                        "condition, integration will align shared cell types across groups."
                    ),
                    question="Should I run batch integration?",
                    severity="suggestion",
                )

    celltype_cols = [c for c in obs.columns if _looks_like_celltype_col(c)]
    active = adata.uns.get("scview_active_clustering")

    def _n_clusters() -> int | None:
        if active and active in obs.columns:
            try:
                return int(obs[active].astype("category").nunique())
            except Exception:  # pragma: no cover
                return None
        return None

    # 4. Embeddings/PCA ready but no clustering.
    if not state.clustering.done and (state.pca.done or state.embeddings.done):
        return DatasetInsight(
            insight=(
                "Embeddings are ready but the cells aren't clustered yet — run "
                "Leiden to find cell populations."
            ),
            question="Run clustering",
            severity="suggestion",
        )

    # 5. Clustered but unlabeled.
    if state.clustering.done and not celltype_cols:
        k = _n_clusters()
        head = f"{k} clusters are" if k else "Clusters are"
        return DatasetInsight(
            insight=f"{head} computed but unlabeled — annotate cell types to read them biologically.",
            question="Annotate cell types",
            severity="suggestion",
        )

    # 6. Fully processed.
    bits = [f"{n:,} cells"]
    k = _n_clusters()
    if k:
        bits.append(f"{k} clusters")
    if celltype_cols:
        bits.append("annotated cell types")
    return DatasetInsight(
        insight="This dataset looks fully processed: " + ", ".join(bits) + ".",
        question="What cell types are in my data?",
    )


async def polish_insight(insight: DatasetInsight, api_key: str, model: str = DEFAULT_MODEL) -> DatasetInsight:
    """Optionally rephrase a deterministic nudge into one friendlier sentence,
    preserving its facts and its click-to-ask ``question``. Returns the original
    unchanged with no API key / on failure (the deterministic text is the source
    of truth — the LLM only rewords)."""
    if not api_key or not insight.insight or insight.severity != "suggestion":
        return insight
    try:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=api_key, base_url=DEEPINFRA_BASE_URL)
        resp = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": (
                    "Rephrase this single-cell analysis tip into ONE friendly, concise "
                    "sentence (max 28 words) for a biologist opening their dataset. Keep "
                    "it factual; do NOT add numbers, tools, or claims not in the input. "
                    "Return only the sentence, no quotes."
                )},
                {"role": "user", "content": insight.insight},
            ],
            temperature=0.4,
            max_tokens=80,
        )
        text = (resp.choices[0].message.content or "").strip().strip('"')
        if text:
            return DatasetInsight(
                insight=text, question=insight.question, severity=insight.severity, polished=True
            )
    except Exception as exc:  # pragma: no cover - best-effort
        logger.debug("insight polish failed: %s", exc)
    return insight


# ---------------------------------------------------------------------------
# "Write methods" — provenance recipe → a methods-section paragraph
# ---------------------------------------------------------------------------


class MethodsResponse(BaseModel):
    """A reproducible methods write-up generated from the recorded provenance."""

    methods: str
    grounded: bool  # True if an LLM wrote the prose; False = deterministic template
    model: str | None = None


def _methods_skeleton(adaptor) -> str:
    """A compact, factual recipe digest: dataset size + each recorded step with
    its tool and parameters, in order. This is the *only* thing the methods writer
    is allowed to draw on — so it cannot invent steps that were never run."""
    adata = adaptor.adata
    n, g = int(adata.n_obs), int(adata.n_vars)
    lines = [f"Dataset: {n:,} cells x {g:,} genes."]

    for key in ("about_title", "title", "about_source", "source"):
        val = _uns_str(adata, key, limit=200)
        if val:
            lines.append(f"Source: {val}")
            break

    hist = provenance.read_provenance(adata).get("history", [])
    if not hist:
        lines.append("No processing steps are recorded in the dataset's provenance.")
        return "\n".join(lines)

    lines.append("Recorded analysis steps (in order):")
    for h in hist:
        step = h.get("step", "?")
        tool = h.get("tool", "")
        params = h.get("params", {}) or {}
        pstr = ", ".join(
            f"{k}={v}" for k, v in params.items() if v not in (None, "", [], {})
        )
        line = f"- {step}"
        if tool:
            line += f" (tool: {tool})"
        if pstr:
            line += f" — {pstr}"
        lines.append(line)
    return "\n".join(lines)


def _deterministic_methods(skeleton: str) -> str:
    """No-LLM fallback: present the recipe as a tidy, honest methods digest."""
    return (
        "**Methods (auto-generated from the recorded analysis provenance)**\n\n"
        "The following steps were recorded for this dataset. This is a factual "
        "digest of the provenance recipe; set `DEEPINFRA_API_KEY` for polished "
        "methods-section prose.\n\n"
        f"{skeleton}\n\n"
        "_Verify tool versions and parameters against your run before publication._"
    )


_METHODS_SYSTEM = (
    "You are writing the Methods section of a single-cell RNA-seq study. You are "
    "given the recorded analysis recipe (the exact steps, tools, and parameters "
    "that were run). Write concise, past-tense, third-person scientific prose (1-2 "
    "short paragraphs) describing the workflow. Name the software/tools and cite "
    "specific parameter values from the recipe. Use ONLY what is in the recipe: do "
    "NOT invent tools, parameters, steps, thresholds, or citations, and omit "
    "anything not recorded. Do not make clinical or biological claims. End with a "
    "one-sentence reproducibility note that the full provenance recipe is embedded "
    "in the dataset."
)


async def write_methods(
    adaptor,
    history: list[ChatMessage] | None,
    api_key: str,
    model: str = DEFAULT_MODEL,
) -> MethodsResponse:
    """Generate a methods-section write-up from the dataset's provenance recipe
    (the source of truth), optionally nudged by what the user has been asking
    about. Falls back to a deterministic digest with no API key / on failure."""
    skeleton = _methods_skeleton(adaptor)
    if not api_key:
        return MethodsResponse(methods=_deterministic_methods(skeleton), grounded=False)

    focus = ""
    if history:
        asks = [m.content.strip() for m in history if m.role == "user" and m.content.strip()]
        if asks:
            focus = "\n\nThe user has focused on: " + "; ".join(asks[-3:]) + (
                "\n(Use this only to choose emphasis — never to add unrecorded facts.)"
            )

    try:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=api_key, base_url=DEEPINFRA_BASE_URL)
        resp = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _METHODS_SYSTEM},
                {"role": "user", "content": f"Analysis recipe:\n{skeleton}{focus}\n\nWrite the Methods paragraph."},
            ],
            temperature=0.2,
            max_tokens=700,
        )
        text = (resp.choices[0].message.content or "").strip()
        if text:
            return MethodsResponse(methods=text, grounded=True, model=model)
    except Exception as exc:
        logger.warning("write_methods LLM call failed (%s); using deterministic digest", exc)
    return MethodsResponse(methods=_deterministic_methods(skeleton), grounded=False)


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


def _assemble(
    adaptor,
    *,
    extra_context: str,
    extra_sources: list[ChatSource] | None,
    app_context: str,
    app_sources: list[ChatSource] | None,
    view_context: dict | None,
    include_data_grounding: bool,
    data_facets: list[str] | None = None,
) -> tuple[str, list[ChatSource]]:
    """Build the grounded context string + ordered source list from the chosen
    sources (shared by streaming and non-streaming paths). ``data_facets`` narrows
    the dataset grounding to only the facets a question needs (empty/None = all)."""
    blocks: list[str] = []
    sources: list[ChatSource] = []
    if include_data_grounding and adaptor is not None:
        facets = set(data_facets) if data_facets else None
        data_context, data_sources = build_grounding_context(adaptor, facets)
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
    return context, sources


def _build_messages(query: str, context: str, history: list[ChatMessage] | None) -> list[dict]:
    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in (history or [])[-6:]:  # keep the last few turns
        if m.role in ("user", "assistant") and m.content:
            messages.append({"role": m.role, "content": m.content})
    messages.append({"role": "user", "content": _build_user_message(query, context)})
    return messages


# --- Natural-language UI actions (Phase 1: safe view/navigation commands) -----

_PANELS = {
    "load": "Data", "assessment": "Data Assessment", "unified": "Unified View",
    "observations": "Observations", "expression": "Gene Expression",
    "genesets": "Gene Sets & Enrichment", "markers": "Marker Genes",
    "trajectory": "Trajectory", "provenance": "History", "assistant": "AI Co-pilot",
}
_SUBTABS = {
    "markers": "Markers", "expression": "Expression",
    "genesets": "Gene Sets", "enrichment": "Enrichment",
}
_COMMAND_RE = re.compile(
    r"^\s*(colou?r|show|highlight|select|go to|open|switch|navigate|display|set|view|take me|"
    r"clear|hide|reset|remove|group|jump|focus|overlay|annotate|label|re-?cluster|cluster|run|"
    r"compute|detect|find|score|enrich)\b",
    re.I,
)


def _looks_like_command(query: str) -> bool:
    """Cheap gate: only run action extraction on imperative/command-like messages."""
    return bool(_COMMAND_RE.match(query or ""))


def _estimate(step: str, n_cells: int, method: str | None = None) -> str:
    """Rough processing-time estimate for a confirm-gated step."""
    n = n_cells or 0
    if step == "cell_type_annotation":
        if method == "llm":
            return "~15-45 s (a few model calls)"
        return f"~{max(30, round(n / 250))} s (CellTypist on {n:,} cells)"
    if step == "clustering":
        return f"~{max(15, round(n / 500))} s for {n:,} cells"
    if step == "doublet_detection":
        return f"~{max(15, round(n / 300))} s (Scrublet)"
    if step == "marker_genes":
        return f"~{max(10, round(n / 500))} s for {n:,} cells"
    if step == "enrichment":
        return "~1-3 min (MSigDB enrichment)"
    return ""


def _coerce_actions(
    raw: str, *, columns: list[str], embeddings: list[str], genes_upper: dict[str, str],
    n_cells: int = 0, active_clustering: str = "",
) -> list[AssistantAction]:
    """Parse + strictly validate the model's JSON action array against the allow-list.

    Every field is checked against the real dataset (columns/embeddings/genes/panels);
    anything off the allow-list is dropped, so the model can't drive arbitrary state."""
    m = re.search(r"\[.*\]", raw, re.S)
    if not m:
        return []
    try:
        items = json.loads(m.group(0))
    except Exception:
        return []
    cols, embs = set(columns), set(embeddings)
    out: list[AssistantAction] = []
    for it in items if isinstance(items, list) else []:
        if not isinstance(it, dict):
            continue
        t = str(it.get("type", "")).strip()
        if t == "set_color_by" and it.get("column") in cols:
            c = str(it["column"])
            out.append(AssistantAction(type=t, column=c, label=f"Colored the plot by {c}."))
        elif t == "highlight_cluster" and it.get("column") in cols and it.get("value"):
            c, v = str(it["column"]), str(it["value"])
            out.append(AssistantAction(type=t, column=c, value=v, label=f"Highlighted {v} in {c}."))
        elif t == "open_panel" and str(it.get("panel", "")) in _PANELS:
            p = str(it["panel"])
            out.append(AssistantAction(type=t, panel=p, label=f"Opened {_PANELS[p]}."))
        elif t == "set_embedding" and str(it.get("embedding", "")) in embs:
            e = str(it["embedding"])
            out.append(AssistantAction(type=t, embedding=e,
                                       label=f"Switched the embedding to {e.replace('X_', '').upper()}."))
        elif t == "set_subtab" and str(it.get("subtab", "")) in _SUBTABS:
            s = str(it["subtab"])
            out.append(AssistantAction(type=t, subtab=s, label=f"Opened the {_SUBTABS[s]} tab."))
        elif t == "set_groupby" and it.get("column") in cols:
            c = str(it["column"])
            out.append(AssistantAction(type=t, column=c, label=f"Grouped by {c}."))
        elif t == "clear_highlight":
            out.append(AssistantAction(type=t, label="Cleared the highlight."))
        elif t == "clear_overlay":
            out.append(AssistantAction(type=t, label="Cleared the gene overlay."))
        elif t == "show_gene" and str(it.get("gene", "")).upper() in genes_upper:
            g = genes_upper[str(it["gene"]).upper()]
            out.append(AssistantAction(type=t, gene=g, label=f"Showing {g} expression."))
        elif t == "annotate_cell_types":
            method = str(it.get("method") or "llm").lower()
            if method not in ("llm", "celltypist"):
                method = "llm"
            groupby = str(it.get("groupby") or "")
            if groupby and groupby not in cols:
                groupby = ""
            target = f"{groupby}_celltypeAnno" if groupby else "cell_type"
            params: dict = {"annotation_method": method, "annotation_target": target}
            if groupby:
                params["annotation_groupby"] = groupby
            if it.get("tissue"):
                params["annotation_tissue"] = str(it["tissue"])
            adv = (f"Overwrites the existing '{target}' column."
                   if target in cols else f"Writes a new '{target}' column.")
            out.append(AssistantAction(
                type=t, step="cell_type_annotation", params=params, requires_confirm=True,
                advisory=adv, estimate=_estimate("cell_type_annotation", n_cells, method),
                label=f"Annotate cell types ({'AI / any-tissue' if method == 'llm' else 'CellTypist'})"
                      + (f" on {groupby}" if groupby else "") + ".",
            ))
        elif t == "cluster":
            try:
                res = float(it.get("resolution") or 0.5)
            except Exception:
                res = 0.5
            scol = f"scview_leiden_r{res}"
            adv = (f"Overwrites the existing '{scol}' column."
                   if scol in cols else f"Adds a new clustering column '{scol}'.")
            out.append(AssistantAction(
                type=t, step="clustering",
                params={"clustering_method": "leiden", "clustering_resolution": res},
                requires_confirm=True, advisory=adv, estimate=_estimate("clustering", n_cells),
                label=f"Re-cluster (Leiden, resolution {res}).",
            ))
        elif t == "detect_doublets":
            out.append(AssistantAction(
                type=t, step="doublet_detection", params={}, requires_confirm=True,
                advisory="Adds 'predicted_doublet' and 'doublet_score' columns (cells are flagged, not removed).",
                estimate=_estimate("doublet_detection", n_cells),
                label="Detect doublets (Scrublet).",
            ))
        elif t == "compute_markers":
            gb = str(it.get("groupby") or "")
            if gb not in cols:
                gb = active_clustering if active_clustering in cols else ""
            out.append(AssistantAction(
                type=t, step="marker_genes", params=({"marker_columns": [gb]} if gb else {}),
                requires_confirm=True,
                advisory=f"Computes marker genes for '{gb or 'the active clustering'}'.",
                estimate=_estimate("marker_genes", n_cells),
                label="Compute marker genes" + (f" for {gb}" if gb else "") + ".",
            ))
        elif t == "run_enrichment":
            gb = str(it.get("groupby") or "")
            if gb not in cols:
                gb = active_clustering if active_clustering in cols else ""
            out.append(AssistantAction(
                type=t, step="enrichment", params=({"enrichment_columns": [gb]} if gb else {}),
                requires_confirm=True,
                advisory=f"Computes MSigDB pathway enrichment for '{gb or 'the active clustering'}' "
                         "(needs marker genes).",
                estimate=_estimate("enrichment", n_cells),
                label="Run pathway enrichment" + (f" for {gb}" if gb else "") + ".",
            ))
    return out[:4]


def _deterministic_actions(
    query: str, *, columns: list[str], embeddings: list[str], genes_upper: dict[str, str],
    n_cells: int, active_clustering: str
) -> list[AssistantAction]:
    """Regex fallback for unambiguous single-step commands the LLM sometimes drops
    (e.g. 'detect doublets', 'clear highlight'). Built as dicts and passed through the
    same strict allow-list validation, so this can't introduce off-list actions."""
    q = (query or "").lower()
    dicts: list[dict] = []
    if "doublet" in q:
        dicts.append({"type": "detect_doublets"})
    if "highlight" in q and any(w in q for w in ("clear", "remove", "hide", "reset")):
        dicts.append({"type": "clear_highlight"})
    if "overlay" in q and any(w in q for w in ("clear", "remove", "hide")):
        dicts.append({"type": "clear_overlay"})
    if "annotate" in q:
        dicts.append({"type": "annotate_cell_types"})
    if "enrich" in q:
        dicts.append({"type": "run_enrichment"})
    if "marker" in q:
        dicts.append({"type": "compute_markers"})
    if re.search(r"\b3-?d\b", q):
        dicts.append({"type": "set_embedding", "embedding": "X_umap_3d"})
    if not dicts:
        return []
    return _coerce_actions(
        json.dumps(dicts), columns=columns, embeddings=embeddings, genes_upper=genes_upper,
        n_cells=n_cells, active_clustering=active_clustering,
    )


async def extract_actions(
    query: str, view_context: dict | None, columns: list[str], embeddings: list[str],
    genes_upper: dict[str, str], n_cells: int, active_clustering: str, api_key: str, model: str
) -> list[AssistantAction]:
    """Map a UI command to allow-listed structured actions (one focused LLM call)."""
    panels = ", ".join(f"{k} ({v})" for k, v in _PANELS.items())
    cols = ", ".join(columns[:40]) or "(none)"
    embs = ", ".join(embeddings) or "(none)"
    hl = (view_context or {}).get("highlighted") or {}
    vc = f" Current highlighted group: {hl.get('column')}={hl.get('value')}." if hl else ""
    prompt = (
        "You translate a single scView UI command into structured actions. Use ONLY these types:\n"
        "- set_color_by {type, column}: color the scatter by an obs column.\n"
        "- highlight_cluster {type, column, value}: highlight one group within a column.\n"
        "- set_groupby {type, column}: set the grouping column (for markers/violins).\n"
        "- show_gene {type, gene}: show a gene's expression (gene = a gene symbol).\n"
        "- set_embedding {type, embedding}: switch the embedding (obsm key, e.g. 3D -> X_umap_3d).\n"
        "- set_subtab {type, subtab}: open a Unified View tab (markers|expression|genesets|enrichment).\n"
        "- open_panel {type, panel}: navigate to a panel.\n"
        "- clear_highlight {type} / clear_overlay {type}: clear the current highlight / gene overlay.\n"
        "- annotate_cell_types {type, method?, groupby?, tissue?}: label clusters with cell types "
        "(method: llm = any-tissue AI default, or celltypist).\n"
        "- cluster {type, resolution?}: re-run Leiden clustering at a resolution.\n"
        "- detect_doublets {type}: run doublet detection / find doublets / flag doublet cells.\n"
        "- compute_markers {type, groupby?}: compute marker genes per group.\n"
        "- run_enrichment {type, groupby?}: compute MSigDB pathway enrichment per group.\n"
        f"Valid obs columns: {cols}.\n"
        f"Valid embeddings (obsm key): {embs}.\n"
        f"Valid panels (id): {panels}.\n"
        "Output ONLY a JSON array of action objects, nothing else. If the message is a question "
        "or cannot be mapped to the above, output []." + vc + f"\n\nCommand: {query}"
    )
    try:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=api_key, base_url=DEEPINFRA_BASE_URL)
        resp = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=300,
        )
        actions = _coerce_actions(
            resp.choices[0].message.content or "",
            columns=columns, embeddings=embeddings, genes_upper=genes_upper, n_cells=n_cells,
            active_clustering=active_clustering,
        )
    except Exception as exc:  # pragma: no cover - network/LLM failure
        logger.debug("action extraction failed: %s", exc)
        actions = []
    # Deterministic fallback for unambiguous commands the LLM (or a missing key) dropped.
    if not actions:
        actions = _deterministic_actions(
            query, columns=columns, embeddings=embeddings, genes_upper=genes_upper,
            n_cells=n_cells, active_clustering=active_clustering,
        )
    return actions


async def _maybe_actions(query, adaptor, api_key, view_context, model) -> list[AssistantAction]:
    """Return UI actions if this looks like a command we can map, else []. Uses the LLM
    when a key is set, and falls back to the deterministic matcher (which also runs with
    no key, so basic commands work without the LLM)."""
    if not (adaptor is not None and _looks_like_command(query)):
        return []
    try:
        cols = [c["name"] for c in adaptor.obs_columns_info()]
    except Exception:
        cols = []
    try:
        embs = [e["name"] for e in adaptor.available_embeddings()]
    except Exception:
        embs = []
    try:
        genes_upper = {str(g).upper(): str(g) for g in adaptor.adata.var_names}
    except Exception:
        genes_upper = {}
    try:
        nc = adaptor.n_cells()
    except Exception:
        nc = 0
    try:
        active = str(adaptor.adata.uns.get("scview_active_clustering") or "")
    except Exception:
        active = ""
    if not api_key:
        # No LLM configured: deterministic matcher only (unambiguous commands).
        return _deterministic_actions(
            query, columns=cols, embeddings=embs, genes_upper=genes_upper,
            n_cells=nc, active_clustering=active,
        )
    return await extract_actions(
        query, view_context, cols, embs, genes_upper, nc, active, api_key, model
    )


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
    data_facets: list[str] | None = None,  # narrow the dataset facets included
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
    actions = await _maybe_actions(query, adaptor, api_key, view_context, model)
    if actions:
        return ChatResponse(
            answer=" ".join(a.label for a in actions) or "Done.",
            sources=[], grounded=True, route=["action"], actions=actions, followups=[],
        )

    context, sources = _assemble(
        adaptor, extra_context=extra_context, extra_sources=extra_sources,
        app_context=app_context, app_sources=app_sources, view_context=view_context,
        include_data_grounding=include_data_grounding, data_facets=data_facets,
    )

    route = route or []
    if not api_key:
        logger.info("assistant: no API key, returning templated fallback")
        return _fallback_answer(query, context, sources, route)

    try:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=api_key, base_url=DEEPINFRA_BASE_URL)
        messages = _build_messages(query, context, history)

        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
            max_tokens=1024,
        )
        answer = (response.choices[0].message.content or "").strip()
        if not answer:
            return _fallback_answer(query, context, sources, route)
        followups = await suggest_followups(query, answer, api_key, model)
        return ChatResponse(
            answer=answer, sources=sources, grounded=True, raw_response=answer,
            route=route, followups=followups, model=model,
        )
    except Exception as exc:
        logger.warning("assistant: LLM call failed (%s); using fallback", exc)
        return _fallback_answer(query, context, sources, route)


async def suggest_followups(query: str, answer: str, api_key: str, model: str) -> list[str]:
    """Propose 2-3 short, specific next questions. Empty on no-key/failure."""
    if not api_key:
        return []
    try:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=api_key, base_url=DEEPINFRA_BASE_URL)
        resp = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": (
                    "Given a Q&A about a single-cell RNA-seq dataset in the scView app, "
                    "propose 3 SHORT, specific follow-up questions the user might ask next "
                    "(each < 12 words). Respond ONLY as a JSON array of strings."
                )},
                {"role": "user", "content": f"Q: {query}\nA: {answer}"},
            ],
            temperature=0.5,
            max_tokens=120,
        )
        raw = (resp.choices[0].message.content or "").strip()
        if "[" in raw:
            raw = raw[raw.find("["): raw.rfind("]") + 1]
        import json as _json

        items = _json.loads(raw)
        out = [str(s).strip() for s in items if isinstance(s, str) and s.strip()]
        return out[:3]
    except Exception as exc:  # pragma: no cover - best-effort
        logger.debug("followup generation failed: %s", exc)
        return []


async def stream_answer(
    query: str,
    adaptor,
    api_key: str,
    history: list[ChatMessage] | None = None,
    *,
    extra_context: str = "",
    extra_sources: list[ChatSource] | None = None,
    app_context: str = "",
    app_sources: list[ChatSource] | None = None,
    view_context: dict | None = None,
    include_data_grounding: bool = True,
    data_facets: list[str] | None = None,
    route: list[str] | None = None,
    model: str = DEFAULT_MODEL,
):
    """Async generator yielding chat events for SSE:
    ``{"type":"sources", sources, route, grounded}`` once, then ``{"type":"delta",
    "text"}`` per token, then ``{"type":"done", "followups"}``. Mirrors
    ``answer_query`` but streams the answer."""
    actions = await _maybe_actions(query, adaptor, api_key, view_context, model)
    if actions:
        yield {"type": "sources", "sources": [], "route": ["action"], "grounded": True}
        yield {"type": "delta", "text": " ".join(a.label for a in actions) or "Done."}
        yield {"type": "done", "followups": [], "actions": [a.model_dump() for a in actions]}
        return

    context, sources = _assemble(
        adaptor, extra_context=extra_context, extra_sources=extra_sources,
        app_context=app_context, app_sources=app_sources, view_context=view_context,
        include_data_grounding=include_data_grounding, data_facets=data_facets,
    )
    route = route or []
    yield {
        "type": "sources",
        "sources": [s.model_dump() for s in sources],
        "route": route,
        "grounded": bool(api_key),
    }

    if not api_key:
        fb = _fallback_answer(query, context, sources, route)
        yield {"type": "delta", "text": fb.answer}
        yield {"type": "done", "followups": []}
        return

    parts: list[str] = []
    try:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=api_key, base_url=DEEPINFRA_BASE_URL)
        stream = await client.chat.completions.create(
            model=model,
            messages=_build_messages(query, context, history),
            temperature=0.2,
            max_tokens=1024,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                parts.append(delta)
                yield {"type": "delta", "text": delta}
    except Exception as exc:
        logger.warning("assistant stream failed (%s); using fallback", exc)
        if not parts:
            fb = _fallback_answer(query, context, sources, route)
            yield {"type": "delta", "text": fb.answer}
        yield {"type": "done", "followups": []}
        return

    answer = "".join(parts).strip()
    followups = await suggest_followups(query, answer, api_key, model) if answer else []
    yield {"type": "done", "followups": followups, "model": model}


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
