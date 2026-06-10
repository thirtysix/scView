# scView AI Assistant — Design Spec

*This doc specifies (a) exactly what the AI assistant does **today**, (b) the **design principles**
that keep it trustworthy, and (c) a **roadmap** spec'd concretely enough to build against.*

The AI assistant is scView's headline differentiator for an AI-in-medicine audience: it is the
part of the tool that **interprets the data and advises the analyst** — transparently, reversibly,
and on the record — rather than just displaying results.

---

## 1. As built today

scView's "AI assistant" is **two cooperating layers**: a deterministic **assessor** that
establishes *facts* about the data, and an **LLM advisor** that turns those facts into
*natural-language recommendations* the user can apply with one click. Both feed the **Data
Assessment** panel.

### 1a. The assessor (deterministic) — `backend/src/scview/core/assessor.py`
Inspects the loaded `AnnData` and reports a `PreprocessingState`: for each of ~15 steps, a
`StepStatus { done: bool, confidence: "high"|"medium"|"low", details: str }`.

Steps & how they're detected (concrete, no LLM involved):
- **qc_metrics** — `n_genes_by_counts` / `total_counts` / `pct_counts_mt` (or `percent_mt`) in `obs`.
- **doublet_detection** — `doublet_score` / `predicted_doublet` in `obs` (counts flagged doublets).
- **filtering** — heuristic on min genes/cell (≥200 → likely done; ~0 → not).
- **normalization** — `X` is float **and** (`adata.raw` set or a `counts` layer present); integer `X` → not done (high confidence).
- **log_transform** — sampled `X` max ≤ ~20 and float → done; negatives present → done (low conf, post-scaling).
- **highly_variable_genes** — `highly_variable` column in `var` (+ count selected).
- **scaling** — per-gene mean ≈ 0, std ≈ 1 on a dense sample → done; all-nonnegative → not.
- **pca** — `X_pca` in `obsm` (+ n components, variance ratio if present).
- **batch_correction** — `X_pca_harmony` in `obsm` *(Harmony-specific — a known limitation, see §3.2)*.
- **neighbors** — `connectivities` / `distances` in `obsp`.
- **clustering** — `leiden` / `louvain` / `seurat_clusters` / `cluster` (or any `*clust*`/`*community*`) in `obs` (+ cluster count).
- **embeddings** — `X_umap` / `X_tsne` / any `X_*` with 2–3 dims in `obsm`.
- **marker_genes** — `rank_genes_groups` (single or `rank_genes_groups__{col}`) in `uns`.
- **enrichment** — `enrichment__*` keys in `uns`.
- **cell_cycle** — `S_score` / `G2M_score` (+ optional `phase`) in `obs`.

**Provenance overlay** (`_apply_recorded_provenance`): if scView's own provenance recorded a step,
that step is marked `done=True, confidence="high"` regardless of the heuristic — **recorded truth
beats inference.** This is what ties the assistant to the provenance feature: the assistant trusts
what scView *did*, and only infers what it *didn't observe*.

### 1b. The advisor (LLM) — `backend/src/scview/core/llm_advisor.py`
- **Model:** `meta-llama/Meta-Llama-3.1-8B-Instruct` via **DeepInfra** (`https://api.deepinfra.com/v1/openai`,
  OpenAI-compatible `AsyncOpenAI` client), `temperature=0.3`, `max_tokens=2048`.
- **Input:** a system prompt (role = "expert single-cell bioinformatician"; the 12 nameable steps +
  their params; **dataset-size parameter guidance** — <5k / 5k–50k / >50k cells) and a user message
  built from the **assessor's state + dataset summary** (`n_cells`, `n_genes`, species/organism).
- **Output (strict JSON):**
  ```json
  {
    "suggestions": [
      {
        "step": "filtering",
        "recommended": true,
        "reasoning": "Cells with very low gene counts are present; filter <200 genes and genes in <3 cells.",
        "suggested_params": { "min_genes": 200, "min_cells": 3, "max_pct_mt": 20.0 }
      }
    ]
  }
  ```
  Parsed into `AdvisorResponse { suggestions: LLMSuggestion[], raw_response: str }`
  (`LLMSuggestion { step, recommended, reasoning, suggested_params }`). The parser strips markdown
  fences and falls back to substring extraction; if nothing parses, it supplements with the
  rule-based fallback.
- **Rule-based fallback** (`get_rule_based_suggestions`): same `AdvisorResponse` shape, produced
  from hard-coded scanpy best-practice rules + size-tuned params. Triggered when **no API key** is
  configured **or** the call/parse fails. The assistant therefore **never goes dark** — it always
  returns actionable, well-formed suggestions.

### 1c. The endpoints — `backend/src/scview/api/v1/assessment.py`
- `GET  /datasets/{id}/assessment` → assessor state.
- `POST /datasets/{id}/assessment/suggest` → advisor (LLM if `settings.DEEPINFRA_API_KEY`, else rules).
- `POST /datasets/{id}/assessment/run` and `…/run-stream` (SSE progress) → execute selected steps.

### 1d. The UI loop — `frontend/src/components/panels/DataAssessmentPanel.tsx`
1. Load dataset → fetch assessment → render the step list with status badges (done / medium-conf / not done) and the plain-language `details`.
2. "Get AI Suggestions" → advisor → **amber "Apply" cards** (recommended step + reasoning + suggested params), and per-step inline recommendation boxes.
3. **Apply** fills the params and checks the step — **the user then explicitly clicks "Run."** Nothing auto-executes.
4. Run streams SSE progress; QC plots auto-refresh; provenance records each step; re-running flips a step to an amber "re-run" badge and recomputes only it + downstream.

### 1e. The one-line summary
> **assess (deterministic, reproducible) → advise (LLM, natural language) → user approves → run → record (provenance).**

---

## 2. Design principles *(these are the symposium's trust themes as engineering constraints)*
1. **Deterministic-first.** Every *fact* about the data (the assessor) is computed without the LLM
   and is fully reproducible. The LLM only produces *advice* (free-text reasoning + suggested
   params). Facts and opinions never mix.
2. **The LLM advises; it never silently acts.** No suggestion is executed without explicit user
   approval. There is no "auto-pilot." (Human-in-the-loop by construction.)
3. **Everything is auditable.** Any step the user runs on the assistant's advice is recorded in
   `scview_provenance` (tool, params, effect). The analysis is reconstructable and reviewable —
   the governance/reproducibility property an AI-in-medicine setting demands.
4. **Graceful degradation.** No API key, network failure, or unparseable output → deterministic
   rule-based fallback with the same contract. The tool is useful with the LLM *off*.
5. **Honest uncertainty.** The assessor reports confidence levels and plain reasons; the advisor's
   `raw_response` is retained for transparency. The UI shows *why*, not just *what*.

These principles are the headline for the poster: **transparent, reversible, recorded, degradable
AI guidance** — not an opaque autopilot.

---

## 3. Roadmap — next AI-assistant features (spec'd to build)
Each is specified by **trigger · inputs (reuse existing) · output shape · UI surface · guardrails**,
in priority order. All obey the §2 principles (advise, don't act; record; degrade gracefully).

### 3.1 Narrative QC report *(highest value, lowest risk)*
- **Trigger:** a button on the Data Assessment tab once a dataset is loaded ("Explain this data").
- **Inputs:** the existing assessor `PreprocessingState` + the `GET /datasets/{id}/qc` distributions
  (histograms + summary stats already computed). No new computation.
- **Output:** an LLM-written **plain-language paragraph** — "what this data looks like, what looks
  healthy, what to watch (e.g. a high-mito tail, bimodal counts)" — plus a short bulleted "suggested
  attention" list. Structured as `{ summary: str, watch_items: str[] }`.
- **UI:** a collapsible card above the step list; the `raw_response` is available for transparency.
- **Guardrails:** read-only; advisory; falls back to a templated summary from the deterministic
  stats when the LLM is off. **No clinical claims** — phrasing constrained to data-quality, not biology/diagnosis.

### 3.2 Anomaly & batch-effect flagging *(closes a real gap)*
- **Trigger:** runs as part of assessment; flags surface as advisory cards.
- **Inputs:** **new deterministic detectors** feeding the advisor — e.g. batch mixing (per-batch
  embedding overlap / simple kBET-style or LISI-style score), doublet load (% predicted doublets),
  ambient signal proxy, outlier/low-quality clusters. *Today the assessor only detects
  `X_pca_harmony` and never says "batch effects likely present"* — this fixes that.
- **Output:** `{ flag: str, severity: "info"|"warn", evidence: str, suggested_action: step+params }`,
  then the LLM phrases the recommendation.
- **UI:** severity-colored cards in the assessment panel (consistent with the existing issue cards).
- **Guardrails:** detectors are deterministic and reproducible; the LLM only narrates and proposes;
  the user approves any correction (e.g. enabling Harmony with a chosen `batch_key`).

### 3.3 Parameter optimization
- **Trigger:** an "optimize" affordance on clustering (and optionally neighbors/PCA).
- **Inputs:** cheap metrics over a few candidate settings — e.g. silhouette / modularity across a
  small grid of clustering resolutions, reusing the existing **edit-&-re-run-from-here** machinery
  so only clustering+downstream recompute per candidate.
- **Output:** a ranked table `{ resolution, n_clusters, score }` + an LLM one-liner recommending one,
  with the trade-off explained.
- **UI:** a compact comparison inline on the clustering step; "Apply" wires into the existing re-run.
- **Guardrails:** bounded grid (cost cap); advisory; the chosen run is recorded in provenance.

### 3.4 Natural-language query over the data
- **Trigger:** a query box in the Unified View ("show top DE genes in cluster 3", "recolor by CD8A",
  "which clusters express the interferon signature?").
- **Inputs:** the LLM maps the query to an **allow-listed** action over **existing endpoints**
  (markers / expression / enrichment / gene-set scoring) — never free-form code.
- **Output:** a structured action `{ action: enum, params }` that the frontend executes against the
  current dataset, plus a short natural-language answer.
- **UI:** results render in the existing panels (scatter recolor, markers table, violin).
- **Guardrails (governance-critical):** **closed action vocabulary** only — the model picks from a
  fixed verb list and validated params; it cannot run arbitrary code or pipeline steps. Read-mostly;
  any state-changing action still routes through the approve-then-run path. This is where the
  `llm-cost-abuse-prevention` patterns matter for a hosted deployment.

### 3.5 Explainable provenance narration
- **Trigger:** a "Write methods" button on the History panel.
- **Inputs:** the recorded `scview_provenance` recipe (tools, params, effects).
- **Output:** an LLM-written **methods-style paragraph** ("Cells were filtered (min 200 genes…),
  normalized to 10⁴ counts, log1p-transformed, … clustered with Leiden at resolution 0.5…") suitable
  for a manuscript or report, plus the machine-readable recipe JSON alongside.
- **UI:** a copyable card on the History panel.
- **Guardrails:** generated strictly from the recorded facts (no invented steps); the recipe JSON is
  the source of truth and shown beside the prose so it can be checked.

### 3.6 RAG scientific co-pilot — *the flagship direction* ⭐
A chat assistant that answers the user's questions about **scRNA-seq analysis in general**,
**their particular analysis**, and **their particular results** — every answer **grounded and
cited**, not free-floating LLM text. This is the feature that turns the assistant from a
"next-step coach" into a **trustworthy scientific co-pilot**, and it is the strongest fit for an
AI-in-medicine audience (grounded, auditable, source-cited AI).

**Why it's low-risk to build:** we already have a **production RAG system** in a sibling project
to port from — `105b.Wnt_web/wnt-hub-redesign-git` (WntHub). It combines *RAG + experimental data
context + user query* into grounded, PMID-cited answers, and — crucially — **already uses
DeepInfra**, the same provider scView's advisor uses. We adapt its proven pattern rather than
inventing one.

**The three grounding sources (this is the novel part — scView has all three):**
1. **Methods knowledge (document RAG)** — a vector-indexed corpus of scRNA-seq *method* literature
   and best-practice docs (scanpy/Seurat tutorials, QC-threshold guidance, clustering/annotation
   reviews, benchmark papers). *Answers "why log-normalize? what resolution should I use? how do I
   read this enrichment?"* — like WntHub's PubMed RAG, retargeted to single-cell methods.
2. **The user's particular analysis (provenance)** — the recorded `scview_provenance` recipe
   (steps, params, effects). *Answers "what did I do, in what order, with what settings, and is that
   reasonable?"* — grounded in the actual recorded history (ties directly to §3.5).
3. **The user's particular results (live data)** — structured facts pulled from the loaded
   `AnnData`: cluster compositions, top markers per cluster, enrichment terms, QC distributions,
   cell-type annotations. *Answers "what defines cluster 4? which clusters are proliferating? is
   this signature enriched anywhere?"* — this is scView's analogue of WntHub's "experimental data
   context," except it's the user's own single-cell results.

**Architecture (port from WntHub, adapt to scView's FastAPI + React stack):**
- **Vector store:** PostgreSQL + **pgvector** (HNSW index), hybrid retrieval = vector similarity
  (weight ~0.7) + Postgres full-text `ts_rank` (weight ~0.3), then a **reranker** (DeepInfra
  Qwen3-Reranker). Directly reusable from WntHub's `lib/db/rag-query.js` pattern.
- **Embeddings:** `BAAI/bge-base-en-v1.5` (768-d) via DeepInfra — same as WntHub.
- **Ingestion pipeline** (one-time, for the methods corpus): query-gen → fetch (PubMed / curated
  method docs) → sentence-aware chunk (~225 tokens, 50 overlap) → embed → upsert to pgvector. Port
  WntHub's 3-part `rag/pipeline/` scripts; swap the topic from "Wnt signaling" to "single-cell
  RNA-seq methods."
- **Generation:** DeepInfra chat model (WntHub uses Qwen3-Next-80B for fast / Qwen3-235B for deep;
  scView can start with its existing Llama-3.1-8B and offer a "deep" tier). Prompt = **query
  classification** (general-methods vs my-analysis vs my-results vs hybrid) → assemble context from
  the relevant grounding source(s) → **mandatory inline citations** (PMID for literature; recipe
  step / cluster id / gene for the user's own data) → narrative or structured answer.
- **Async job pattern:** initiate → poll status → return `{ response, sources[] }`, mirroring
  WntHub's `query-initiate` / `query-status` / background-worker split (retrieval + rerank + long
  generation exceeds a single request). In scView this is a clean fit for the existing
  **SSE/notification** infra (`sse-notification-channel` skill) or a job table.
- **Frontend:** a chat panel (adapt `js/components/ai-assistant.js` to React); answers render with a
  **Sources panel** — literature chunks with PMID links + similarity, and, for data-grounded claims,
  links back into the live app (the cited cluster, marker, or provenance step). Clicking a citation
  navigates the Unified View — *the answer and the evidence are the same workspace.*

**Output shape:** `{ answer: str, sources: { kind: "literature"|"provenance"|"result", ref, excerpt, score }[] }`.

**Guardrails (governance-critical — same as §2/§4):** answers are **grounded in retrieved context
and the user's recorded data only**; the model is instructed that **every factual claim carries a
citation** (WntHub already enforces this — "responses without inline citations are unacceptable");
the user's data never leaves their deployment except as prompt context to the configured LLM
endpoint (document this clearly — it's a sensitive-data-governance point); read-only with respect to
the analysis (the co-pilot *explains and suggests*; running anything still routes through the
approve-then-run path of §1). Falls back to literature-only answers when no dataset is loaded.

**Poster framing:** *"Ask scView about your data. Every answer is grounded — in the scRNA-seq
methods literature, in what you actually did (provenance), and in your actual results — and every
claim is cited."* This is the headline AI tease: trustworthy, source-grounded, auditable scientific
AI on the user's own biomedical data.

**Implementation status (branch `feat/ai-copilot`):**

*Phase 1 (2026-06-06) — data-grounded co-pilot, BUILT & LIVE.* Grounding sources **2 (provenance)**
and **3 (results)** + the preprocessing assessment.

*Phase 2 (2026-06-07) — dual-corpus literature RAG, BUILT & LIVE.* Source **1** is now two routed
corpora over pgvector on Neon:
- `core/rag/store.py` (asyncpg + pgvector: `scview_rag_chunks` table, hybrid vector+full-text search),
  `embeddings.py` (DeepInfra `BAAI/bge-base-en-v1.5`, 768-d), `router.py` (LLM classifier + keyword
  fallback → `tutorials` / `literature` / both), `ingest.py` (PubMed E-utilities + tutorial URL/file
  fetch, size-bounded chunking, CLI: `init` / `literature` / `tutorials` / `status`), `retrieve.py`
  (route → embed → hybrid search → typed cited context) wired into `answer_query`'s `extra_context`.
- Endpoint `GET /assistant/rag-status`; config `RAG_DATABASE_URL` + models/weights; offline unit
  tests `tests/test_rag_units.py`.
- **Verified end-to-end on Kang:** methods questions route to `tutorials` and cite `[doc:…]`
  (sc-best-practices), biology questions route to `literature` and cite `[lit:PMID:…]`, in-app facts
  always included; answers are grounded, cited, and honest about corpus coverage. Seeded with a small
  corpus (literature 215 chunks / tutorials 62 chunks) — scale via the ingest CLI.
- **Reuses the WntHub Neon project** (new namespaced table, no collision).
- ⚠ **Ops note:** `asyncpg` + `beautifulsoup4` were added to `pyproject.toml` but pip-installed into
  the running dev container — **rebuild the backend image** (`make build`) to bake them in (same
  pattern as the earlier scikit-image note), else a container recreate loses them.

*Earlier status note (superseded by the above):* the data-grounded slice description below remains
accurate for phase 1. New: `backend/src/scview/core/assistant.py` (`build_grounding_context` + `answer_query`,
DeepInfra chat reusing the `llm_advisor` pattern, deterministic fallback), endpoint
`POST /datasets/{id}/assistant/chat` (`api/v1/assistant.py`), tests `backend/tests/test_assistant.py`
(4 passing), and a frontend **AI Co-pilot** panel (`frontend/src/components/panels/AssistantPanel.tsx`
+ `api/assistant.ts`, wired into the sidebar). Verified end-to-end on the Kang dataset (grounded
answer + 20 cited sources render; fallback path exercised because no `DEEPINFRA_API_KEY` is set in
this environment). **Still TODO — source 1 (literature RAG over pgvector):** `answer_query` exposes an
`extra_context` parameter (the `LITERATURE_RAG_HOOK`) where retrieved+formatted document chunks plug
into the same prompt + sources contract. Standing up Postgres+pgvector, porting WntHub's
`rag-query.js` retrieval + the 3-part ingestion pipeline, and running the methods-corpus ingestion is
**a joint session with the user** (new infra + a long embedding run + external API calls) — not done
unattended.

### 3.7 Co-pilot UX & interaction (brainstorm 2026-06-07)

**Shipped this round:** floating drawer + dedicated panel; view-context awareness;
intent routing (app/data/tutorials/literature) so RAG only runs when needed; dataset
identity/source grounding ("what paper is this from?"); Markdown answers; clickable
citation chips (PubMed links + result→jump-to-cluster); suggested follow-up questions;
token-streaming (SSE); resizable drawer; gentler "Hide" affordance;
**available before a dataset loads** (app-level endpoints help a newcomer get started);
**facet-narrowed data grounding** (identity/groups/markers/enrichment — minimal prompt per question);
**natural-language *actions*** (§3.4 — allow-listed, confirm-gated, with a deterministic fallback);
**"Ask about this" entry points** — a ✦ on each categorical-legend group, the gene
expression overlay, **every marker-table row, and every enrichment-term row** opens the
drawer pre-loaded with a contextual question (groups are highlighted so view-context
resolves too). `ColorLegend` `onAskAbout`/`onAsk` + subtab row buttons →
`viewStore.askCopilot` → `AssistantChat` consumes the queued `pendingAsk`.
**Proactive insight on load** — a deterministic one-line "I notice…" banner when a dataset
opens, picking the most useful next step by walking the preprocessing state in pipeline
order (raw counts → doublet load → condition/batch split → cluster → annotate → done), with
a click-to-ask follow-up. Backend `build_insight` (no LLM, reproducible) +
`GET /datasets/{id}/assistant/insight`; frontend `InsightBanner` (dismissal remembered per
dataset+insight); it includes a **QC-anomaly** branch (high mitochondrial content) and, when
an LLM key is set, **optionally rewords** an actionable nudge into a friendlier sentence
(`polish_insight`; facts + question preserved, deterministic text is the fallback).
**Conversation persistence** — per-dataset threads survive reloads via localStorage
(`scview.chat.v1.<id>`), with a Clear control.
**Cross-encoder reranking** — when `RAG_RERANK_MODEL` is set, retrieval over-fetches
`RAG_RERANK_CANDIDATES` hybrid hits and a DeepInfra reranker (Qwen3-Reranker) reorders them
to `RAG_TOP_K` (`core/rag/rerank.py`; degrades to hybrid order on any failure).
**Write methods** — `POST /datasets/{id}/assistant/methods` turns the provenance recipe
(the source of truth) into a methods-section paragraph (LLM-written, never inventing steps;
deterministic digest fallback); a "Write methods" button appends it to the thread.
**Trust affordances** — answers carry the model name (`ChatResponse.model` + stream `done`)
shown with an "AI-generated — verify" note, plus per-answer 👍/👎 (`POST /assistant/feedback`).

**Still on the roadmap (not yet built):**
- **More "ask about this" surfaces** — QC plots / Observations rows (legend, gene overlay,
  marker rows, and enrichment terms are done).
- **Domain-aware literature** — pull dataset-relevant literature into the corpus; let users add their own papers/docs; show corpus coverage.
- **Token/cost surfacing** for hosted/multi-user deployments.

---

## 4. Guardrails & cost (for any hosted / multi-user deployment)
- The **rule-based fallback already exists** — the assistant is useful and safe with the LLM disabled.
- **Allow-listed actions only** for the NL-query feature (§3.4); no free-form code or unrestricted
  pipeline control from model output.
- A public/multi-tenant deployment needs the **per-user metering, quotas, prompt-injection and
  cost controls** cataloged in the project's `llm-cost-abuse-prevention` / `wire-llm-quota-into-fastapi-app`
  skills — reference those when hosting; **not in scope for the poster build.**
- **No clinical/diagnostic claims** from any LLM surface — phrasing is constrained to data quality
  and analysis methodology, consistent with the §2 principles and the symposium's governance theme.

---

## 5. File reference (as-built)
- `backend/src/scview/core/assessor.py` — deterministic state assessment + provenance overlay.
- `backend/src/scview/core/llm_advisor.py` — DeepInfra/Llama advisor + rule-based fallback (model
  string, system prompt, JSON schema live here).
- `backend/src/scview/api/v1/assessment.py` — assessment / suggest / run / run-stream endpoints.
- `frontend/src/components/panels/DataAssessmentPanel.tsx` — the assess→advise→apply→run UI loop.
- `backend/src/scview/core/pipeline.py` — the 16-step pipeline the advisor recommends into.
- `backend/src/scview/core/provenance.py` — the recorded history the assessor trusts and §3.5 narrates.

*Broader roadmap: `docs/FUTURE.md`.*
