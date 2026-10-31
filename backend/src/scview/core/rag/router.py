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
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass

from pydantic import BaseModel

from scview.core.llm import chat_extra

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
    "outside blood", "other tissues", "elsewhere in the body", "protein atlas", "hpa",
    "copy number", "cnv", "malignant", "protein network", "interaction network",
    "cytokine response",
    "targeted panel", "gene panel", "panel genes", "the panel", "xenium", "control counts", "control probe",
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


# Questions about scView's own screens. The feature list must be in the context for
# these, or the model describes tabs and buttons that don't exist.
_APP_UI_RE = re.compile(
    r"\bscview\b|\b(?:tab|button|panel|menu|heat ?map|dot ?plot|visuali[sz]e|in the app)s?\b"
    r"|\bwhere (?:do|can) i\b",
    re.I,
)


def mentions_app_ui(query: str) -> bool:
    """True when a question is about what scView can show or where to find it."""
    return bool(_APP_UI_RE.search(query or ""))


class RouteResult(BaseModel):  # kept for the corpora-level heuristic + tests
    corpora: list[str]
    reason: str


DATA_FACETS = ("identity", "groups", "markers", "enrichment", "qc")

_FACET_HINTS = {
    "identity": (
        "article", "paper", "publication", "citation", "doi", "source", "origin",
        "where is this from", "what is this dataset", "what's this dataset", "who made",
    ),
    "markers": ("marker", "gene", "express", "defines", "characteri", "signature", "top genes",
                "program", "nmf"),
    "groups": (
        "how many", "composition", "proportion", "count", "cell type", "cell types",
        "cluster", "donor", "condition", "sample", "group", "breakdown", "fraction",
        "ligand", "receptor", "communicat", "signalling", "signaling", "interaction", "sender",
        "receiver", "abundance", "milo", "neighbourhood", "neighborhood",
        "crispr", "guide", "knockout", "knock-out", "knocked out", "perturb", "mixscape",
        "pseudobulk", "pseudo-bulk", "voom", "ruv", "differential", "change",
        "cytokine", "cytosig", "milode",
        "audit", "false positive", "pseudoreplicat", "shuffle",
        "deconvol", "bulk",
    ),
    "enrichment": ("enrichment", "pathway", "gsea", "go term", "ontology"),
    "qc": (
        "quality", "qc", "mito", "doublet", "dying", "damaged", "ribosom", "outlier",
        "stand out", "stands out", "artifact", "artefact",
        "haemoglobin", "hemoglobin", "ambient", "contaminat", "cell cycle", "cell-cycle",
        "cycling", "proliferat", "sequencing depth", "technical",
        "pca", "principal component", "elbow", "loadings", "variance explained",
    ),
}


class Intent(BaseModel):
    sources: list[str]
    reason: str
    data_facets: list[str] = []  # when 'data': minimal facets needed (empty = all)


def _infer_facets(query: str) -> list[str]:
    """Heuristic: which dataset facets does a 'data' question need? Empty = all."""
    q = query.lower()
    return [f for f, hints in _FACET_HINTS.items() if any(h in q for h in hints)]


# Stored results a question can name, and the facet whose facts hold them
# (assistant.build_grounding_context): a classified route that picked other facets
# still gets these, or a question naming the cytokine run is answered without it.
_RESULT_FACET_HINTS = {
    "groups": (
        "cytokine", "cytosig", "milo", "neighbourhood", "neighborhood", "pseudobulk",
        "pseudo-bulk", "voom", "ruv", "crispr", "guide", "knockout", "knock-out", "knocked out",
        "perturb", "mixscape", "ligand", "receptor", "communicat", "composition", "proportion",
        "de audit", "donor shuffle", "deconvol", "bulk", "spot composition", "spots", "spot deconv",
        "visium hd", " bins", "bin composition", "um bin", "µm bin",
        "wnn", "modality", "modalities", "rna weight", "protein weight", "learned gate", "gate",
        "spatial domain", "domains", "niche", "nhood", "stability scan", "neighbourhood enrichment",
        "neighborhood enrichment", "next to each other", "side by side", "adjacent", "colocal", "co-local",
        "velocity", "scvelo", "spliced", "unspliced", "phase portrait", "arrows", "gamma", "latent time",
    ),
    "markers": ("gene program", "programs", "nmf"),
    "qc": ("targeted panel", "gene panel", "panel genes", "the panel", "xenium", "control counts",
           "control probe"),
}


def named_result_facets(query: str) -> list[str]:
    """The facets of the stored results ``query`` names (cytokine response, miloDE,
    pseudobulk, CRISPR, programs, communication, ...)."""
    q = (query or "").lower()
    return [f for f, hints in _RESULT_FACET_HINTS.items() if any(h in q for h in hints)]


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
- "app": about scView itself or the user's dataset library / navigation, including \
how to see or do something in scView (e.g. "what datasets do we have?", "how do I \
load data?", "what can this tool do?", "how do I plot markers in scView?").
- "data": about the user's CURRENTLY LOADED dataset — its analysis/results AND its \
own identity/source (e.g. "what cell types are in my data?", "markers of this \
cluster", "what steps ran?", "what paper/publication is THIS dataset from?"). This \
includes results of analyses run in scView on it: where its markers are expressed in \
other tissues (Human Protein Atlas lookup), cytokine response (CytoSig), protein \
interaction networks (STRING), copy number / malignant cells (inferCNV-style), e.g. \
"are the B cell markers specific outside blood?" → data with ["markers"]. \
Results of comparisons and screens run in scView need ["groups"]: DE between \
conditions (pseudobulk DESeq2 / limma-voom / RUV, neighbourhood DE (miloDE)), \
composition and Milo, and CRISPR screens (guide assignment, Mixscape knockouts), \
e.g. "which knockouts worked?" or "which pseudobulk method did I use?" → data with ["groups"].
- "tutorials": single-cell METHODS / how-to / parameter / interpretation questions \
(e.g. "why log-normalize?", "what clustering resolution?").
- "literature": BIOLOGY / evidence answerable from research abstracts \
(e.g. "what marks pDCs?", "is the interferon signature linked to lupus?").
Only include "tutorials"/"literature" when the question truly needs external \
knowledge — those run expensive retrieval. When you include "data", also return \
"data_facets": the MINIMAL subset of ["identity","groups","markers","enrichment","qc"] \
the question needs (e.g. "what paper is this?" → ["identity"]; "what marks cluster \
3?" → ["markers"]; "which clusters look low quality?" → ["qc"]; leave empty only if \
truly everything is needed). Respond ONLY with one line of JSON, the reason last \
and at most 12 words: {"sources": [...], "data_facets": [...], "reason": "..."}."""


async def classify_intent(
    query: str, view_context: dict | None, api_key: str, model: str,
    *, state: RouteState | None = None,
) -> Intent:
    """Classify which knowledge sources a question needs. Falls back to heuristics.

    With ``state`` (what the app has: RAG, a dataset, stored results, a selection),
    a rule pre-check answers the questions whose route the state decides, without
    the LLM, and ``apply_rules`` corrects the classifier's choice afterwards; the
    classifier's answer is cached per question and tab, so the same question gets
    the same route."""
    if state is not None:
        pre = pre_route(query, view_context, state)
        if pre is not None:
            return pre
    intent = await _classify(query, view_context, api_key, model)
    return apply_rules(intent, query, view_context, state) if state is not None else intent


# The classifier's reply is one short JSON object. At 80 tokens it was cut off
# whenever the model wrote a long "reason" (finish_reason "length", no closing
# brace), and the slice from the first "{" to the last "}" was then empty:
# "Expecting value: line 1 column 1 (char 0)". The reason now comes last and
# short, the budget has room, and the parser reads fenced or cut-off replies.
# No response_format: DeepSeek V4.1-Flash's json_object mode wraps or runs on.
_CLASSIFY_MAX_TOKENS = 200
_FENCE_RE = re.compile(r"^```[a-zA-Z]*\s*|\s*```\s*$")


def parse_intent_reply(raw: str) -> dict | None:
    """The classifier's reply as a dict with ``sources`` (and ``data_facets``,
    ``reason`` when present), tolerating code fences, text around the object and
    a reply cut off after the lists. None when no source list can be read."""
    text = _FENCE_RE.sub("", (raw or "").strip()).strip()
    start = text.find("{")
    if start < 0:
        return None
    end = text.rfind("}")
    if end > start:
        try:
            data = json.loads(text[start:end + 1])
            if isinstance(data, dict):
                return data if data.get("sources") else None
        except json.JSONDecodeError:
            pass
    # Cut off or malformed: read each field on its own.
    out: dict = {}
    body = text[start:]
    for key in ("sources", "data_facets"):
        m = re.search(rf'"{key}"\s*:\s*(\[[^\]]*\]|"[^"]*")', body)
        if m:
            out[key] = re.findall(r'"([^"]*)"', m.group(1))
    m = re.search(r'"reason"\s*:\s*"([^"]*)', body)
    if m:
        out["reason"] = m.group(1)
    return out if out.get("sources") else None


async def _classify(query: str, view_context: dict | None, api_key: str, model: str) -> Intent:
    if not api_key:
        return heuristic_intent(query, view_context)
    vc = view_context or {}
    panel = vc.get("panel") or "unknown"
    subtab = vc.get("subtab") or ""
    key = (" ".join((query or "").lower().split()), str(panel).lower(), str(subtab).lower(), model)
    with _memo_lock:
        hit = _memo.get(key)
        if hit is not None:
            _memo.move_to_end(key)
            return hit.model_copy(deep=True)
    where = f'(current tab: "{panel}"' + (f', subtab: "{subtab}"' if subtab else "") + ")"
    raw = ""
    try:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=api_key, base_url="https://api.deepinfra.com/v1/openai")
        resp = await client.chat.completions.create(
            model=model,
            **chat_extra(model),
            messages=[
                {"role": "system", "content": _CLASSIFY_SYSTEM},
                {"role": "user", "content": f"{where}\n{query}"},
            ],
            temperature=0.0,
            max_tokens=_CLASSIFY_MAX_TOKENS,
        )
        raw = (resp.choices[0].message.content or "").strip()
        data = parse_intent_reply(raw)
        if data is None:
            raise ValueError("no source list in the reply")
        srcs = data.get("sources") or []
        sources = [s for s in ([srcs] if isinstance(srcs, str) else srcs) if s in SOURCES]
        if sources:
            facets = [f for f in (data.get("data_facets") or []) if f in DATA_FACETS]
            intent = Intent(
                sources=sources,
                reason=str(data.get("reason") or "llm-classified"),
                data_facets=facets if "data" in sources else [],
            )
            with _memo_lock:
                _memo[key] = intent.model_copy(deep=True)
                while len(_memo) > _MEMO_KEEP:
                    _memo.popitem(last=False)
            return intent
    except Exception as exc:
        logger.warning("intent classify LLM failed (%s; reply %r); using heuristic", exc, raw[:200])
    return heuristic_intent(query, view_context)


# ---------------------------------------------------------------------------
# Rules the app's state decides (no LLM), and the classifier's cache
# ---------------------------------------------------------------------------

_MEMO_KEEP = 256
_memo: OrderedDict[tuple, Intent] = OrderedDict()
_memo_lock = threading.Lock()


def clear_memo() -> None:
    with _memo_lock:
        _memo.clear()


@dataclass(frozen=True)
class RouteState:
    """What the app has when a question comes in."""

    rag_enabled: bool = False    # the literature/tutorials corpora can be searched
    has_dataset: bool = False    # a dataset is open (its facts can ground the answer)
    has_results: bool = False    # it holds stored results (markers, DE, enrichment, ...)
    has_selection: bool = False  # cells are selected (view_context["selection"])


# A question about the dataset's cell types, clusters or genes.
_CELLS_GENES_RE = re.compile(
    r"\bwhich\s+(?:\w+\s+){0,2}?(?:cell[ -]?types?|clusters?|populations?|groups?|genes?|cells?)\b"
    r"|\b(?:cell[ -]?types?|clusters?|populations?)\b.*\bgenes?\b"
    r"|\bgenes?\b.*\b(?:cell[ -]?types?|clusters?|populations?)\b",
    re.I,
)
# A question that points at the open dataset itself: "my data", "this dataset", "the
# monocyte cluster", "cluster 3", "how many donors", "the QC that was done".
_THIS_DATASET_RE = re.compile(
    r"\b(?:this|my|our|the)\s+(?:data(?:set)?|samples?|donors?|experiment)\b"
    r"|\b(?:the|this|my)\s+(?:[\w+-]+\s+){1,2}(?:clusters?|cell[ -]?types?|populations?)\b"
    r"|\bclusters?\s+\d+\b|\bhow\s+many\s+(?:cells?|donors?|samples?|clusters?|genes?)\b"
    r"|\b(?:was|were)\s+(?:done|run|applied|used|filtered|removed)\b",
    re.I,
)
# ... asking what responds or changes: what the stored comparisons answer.
_RESPONSE_RE = re.compile(
    r"\b(?:respond\w*|responses?|affected|induc\w*|up-?regulat\w*|down-?regulat\w*|"
    r"changes?|changed|differ\w*|react\w*)\b",
    re.I,
)
_RESPONSE_FACETS = ("groups", "markers", "enrichment")


def _has_selection(view_context: dict | None, state: RouteState) -> bool:
    if state.has_selection:
        return True
    sel = (view_context or {}).get("selection")
    return isinstance(sel, dict) and bool(sel.get("n_cells"))


def _refers_to_selection(query: str) -> bool:
    from scview.core.selection_facts import refers_to_selection

    return refers_to_selection(query)


def _literature_wanted(query: str, state: RouteState) -> bool:
    q = query.lower()
    return state.rag_enabled and any(h in q for h in _LITERATURE_HINTS)


# "Write the methods paragraph": the recorded recipe answers it (assistant.write_methods),
# the same as the drawer's "Write methods" button, never the app's feature list.
_METHODS_RE = re.compile(
    r"\b(?:write|draft|generate|give me|produce|create|compose|prepare|make)\b[^.?!]{0,40}?"
    r"\bmethods?\b(?!\s+(?:should|to use|would|could|for de\b|for differential))"
    r"|\bmethods?\s+(?:section|paragraph|text)\b",
    re.I,
)
# "Explain this result", "what am I looking at": about what is on screen.
_ON_SCREEN_RE = re.compile(
    r"\b(?:this|these|the)\s+(?:result|plot|table|comparison|analysis|volcano|figure|output|"
    r"test|enrichment|chart|heat ?map)s?\b|\bon (?:the |my )?screen\b"
    r"|\bwhat (?:am i|i'm|i am) (?:looking at|seeing)\b|\bexplain (?:this|it|that)\b",
    re.I,
)
# "What should I do next?"
_NEXT_RE = re.compile(
    r"\bwhat (?:should|can|could|do) i (?:do|try|run|look at)(?: \w+){0,2} next\b|\bnext steps?\b"
    r"|\bwhat(?:'s| is) next\b|\bwhat now\b|\bwhere (?:do|should) i go from here\b",
    re.I,
)
# The facets that hold the facts behind what a Unified View subtab, Compare tab or
# on-screen result shows (assistant.build_grounding_context).
_SCREEN_FACETS = {
    "de": ("groups",), "markers": ("markers",), "expression": ("markers",),
    "enrichment": ("enrichment", "markers"), "genesets": ("enrichment", "markers"),
    "communication": ("groups",), "programs": ("markers",), "cnv": ("markers",),
    "clonotypes": ("groups",), "perturbation": ("groups",), "activity": ("groups",),
    "protein": ("markers",), "compare": ("groups",), "report": ("groups", "markers", "qc"),
}


def wants_methods(query: str) -> bool:
    """A request for a methods section / paragraph of the analysis."""
    return bool(_METHODS_RE.search(query or ""))


def screen_kind(view_context: dict | None) -> str | None:
    """What the user has on screen: the on-screen result's kind, else the open
    Unified View subtab, else "compare" on the Compare panel."""
    vc = view_context or {}
    scr = vc.get("on_screen")
    if isinstance(scr, dict) and scr.get("kind"):
        return str(scr["kind"])
    if vc.get("subtab"):
        return str(vc["subtab"])
    if str(vc.get("panel") or "").lower() == "compare":
        return "compare"
    return None


def _screen_facets(view_context: dict | None) -> set[str]:
    return set(_SCREEN_FACETS.get(screen_kind(view_context) or "", ()))


def pre_route(query: str, view_context: dict | None, state: RouteState) -> Intent | None:
    """The route of a question the app's state decides, without the classifier:

    - about the selected cells ("these cells", "my lasso") while cells are selected:
      the dataset (with the selection's facts), never the literature alone;
    - which cell types / genes respond or change, on a dataset with stored results:
      the dataset's groups, markers and enrichment facts, plus the literature when
      it can be searched and the question is biological;
    - "explain this result" with a result or subtab on screen: the dataset, with the
      facets behind what is shown;
    - "what should I do next?": the dataset's state (QC, results) and the app's tools.

    None for every other question."""
    if not state.has_dataset:
        return None
    named = set(named_result_facets(query))
    if _NEXT_RE.search(query or ""):
        facets = {"qc", "groups"} | named | _screen_facets(view_context)
        return Intent(sources=["data", "app"], reason="rule: next steps from the dataset's state",
                      data_facets=[f for f in DATA_FACETS if f in facets])
    on_screen = _screen_facets(view_context)
    if on_screen and _ON_SCREEN_RE.search(query or ""):
        facets = on_screen | named
        sources = ["data"] + (["literature"] if _literature_wanted(query, state) else [])
        return Intent(sources=sources, reason=f"rule: the {screen_kind(view_context)} result on screen",
                      data_facets=[f for f in DATA_FACETS if f in facets])
    selection = _has_selection(view_context, state) and _refers_to_selection(query)
    response = state.has_results and bool(_CELLS_GENES_RE.search(query)) and bool(_RESPONSE_RE.search(query))
    if not (selection or response):
        return None
    sources = ["data"] + (["literature"] if _literature_wanted(query, state) else [])
    facets = set(_infer_facets(query))
    if response:
        facets |= set(_RESPONSE_FACETS)
    if selection and not facets:
        facets = {"groups"}
    reason = "rule: about the selected cells" if selection else "rule: what responds, from the stored results"
    return Intent(sources=sources, reason=reason, data_facets=[f for f in DATA_FACETS if f in facets])


def apply_rules(intent: Intent, query: str, view_context: dict | None, state: RouteState) -> Intent:
    """Correct the classifier's route with what the app has:

    - no corpora to search: never the literature or tutorials (an answer from an
      empty retrieval cites nothing); the dataset instead, or the app without one;
    - a question about the dataset's cell types or genes, on a dataset with stored
      results: always the dataset too, even when the literature is chosen;
    - a question about the selected cells: always the dataset;
    - a question pointing at the open dataset ("my data", "the monocyte cluster",
      "cluster 3", "the QC that was done"): always the dataset;
    - a question naming a stored result (cytokine response, miloDE, pseudobulk,
      CRISPR, programs, communication): the dataset too, on a dataset with stored
      results, and the facet that holds it, whatever else the classifier picked."""
    sources = list(intent.sources)
    facets = list(intent.data_facets)
    had_data = "data" in sources
    if not state.rag_enabled:
        sources = [s for s in sources if s not in CORPORA]
    named = named_result_facets(query)
    about_data = state.has_dataset and (
        (state.has_results and (bool(_CELLS_GENES_RE.search(query)) or bool(named)))
        or (_has_selection(view_context, state) and _refers_to_selection(query))
        or bool(_THIS_DATASET_RE.search(query or ""))
    )
    if about_data and "data" not in sources:
        sources.append("data")
    if not sources:
        sources = ["data"] if state.has_dataset else ["app"]
    if "data" in sources and not had_data:
        facets = _infer_facets(query)
    if "data" in sources and facets:  # empty = every facet already
        facets = facets + [f for f in named if f not in facets]
    if "data" not in sources:
        facets = []
    if sources == intent.sources and facets == intent.data_facets:
        return intent
    return Intent(sources=sources, reason=f"{intent.reason} (adjusted to the app's state)", data_facets=facets)
