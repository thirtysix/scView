"""Co-pilot routing reliability (batch 24): the intent classifier's reply parser,
the facets of named stored results, on-screen and next-step questions, methods
requests answered from the recipe, and the on-screen result's facts."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from scview.core import methods_guard, screen_facts
from scview.core.rag import router
from scview.core.rag.router import (
    Intent,
    RouteState,
    apply_rules,
    classify_intent,
    named_result_facets,
    parse_intent_reply,
    pre_route,
    wants_methods,
)

API = "/api/v1"


@pytest.fixture(autouse=True)
def _fresh_memo():
    router.clear_memo()
    yield
    router.clear_memo()


# --- the classifier's reply ----------------------------------------------------

# Recorded DeepSeek-V4.1-Flash replies (2026-10-26). The "methods" one used 78 of
# the old 80-token budget; a longer reason was cut off with no closing brace.
RECORDED_OK = (
    '{"sources": ["data"], "data_facets": ["groups"], "reason": "Question about CRISPR knockout '
    'results (Mixscape) run in scView on the current dataset — a screen/comparison result, '
    'so data with the groups facet."}'
)
RECORDED_CUT = (
    '{"sources": ["app"], "data_facets": [], "reason": "The user wants a methods paragraph for '
    "their paper, which is about scView itself and how to document/describe the analysis "
    "workflow in the app; this is an app-usage/how-to question rather than a question about"
)


def test_the_old_slice_fails_on_a_cut_off_reply():
    raw = RECORDED_CUT
    with pytest.raises(json.JSONDecodeError, match=r"line 1 column 1 \(char 0\)"):
        json.loads(raw[raw.find("{"): raw.rfind("}") + 1])  # the logged error


@pytest.mark.parametrize("raw, sources, facets", [
    (RECORDED_OK, ["data"], ["groups"]),
    (RECORDED_CUT, ["app"], []),
    ('```json\n{"sources": ["data", "tutorials"], "data_facets": ["qc"], "reason": "x"}\n```',
     ["data", "tutorials"], ["qc"]),
    ('Here is the route: {"sources": ["data"], "data_facets": ["markers"], "reason": "y"} Done.',
     ["data"], ["markers"]),
    ('{"sources": ["data", "literature"], "data_facets": ["groups", "mark', ["data", "literature"], None),
    ('{"sources": "data", "data_facets": ["qc"]', ["data"], ["qc"]),
])
def test_reply_parser_reads_fenced_and_cut_off_json(raw, sources, facets):
    data = parse_intent_reply(raw)
    assert data is not None
    got = data["sources"]
    assert ([got] if isinstance(got, str) else got) == sources
    assert data.get("data_facets") == facets


@pytest.mark.parametrize("raw", ["", "data", "I would use the dataset.", '{"reason": "no list"}'])
def test_reply_parser_gives_none_without_a_source_list(raw):
    assert parse_intent_reply(raw) is None


class _FakeLLM:
    reply = RECORDED_CUT
    seen: list[dict] = []

    def __init__(self, *a, **k):
        class _Completions:
            async def create(self, **kw):
                _FakeLLM.seen.append(kw)
                msg = SimpleNamespace(content=_FakeLLM.reply)
                return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

        self.chat = SimpleNamespace(completions=_Completions())


async def test_a_cut_off_reply_is_used_not_the_heuristic(monkeypatch):
    import openai

    monkeypatch.setattr(openai, "AsyncOpenAI", _FakeLLM)
    _FakeLLM.seen = []
    i = await classify_intent("How do I export a figure?", {"panel": "Data"}, "key", "m")
    assert i.sources == ["app"] and i.reason != "heuristic"
    kw = _FakeLLM.seen[0]
    assert kw["max_tokens"] >= 200 and "response_format" not in kw
    # the subtab is part of the question the classifier sees, and of its cache key
    await classify_intent("Explain the plot", {"panel": "Unified View", "subtab": "de"}, "key", "m")
    assert 'subtab: "de"' in _FakeLLM.seen[-1]["messages"][1]["content"]


# --- named results keep their facet ---------------------------------------------


@pytest.mark.parametrize("q, facet", [
    ("What does the CytoSig cytokine run say about B cells?", "groups"),
    ("Summarise the miloDE result", "groups"),
    ("Which pseudobulk genes are strongest?", "groups"),
    ("Which knockouts worked in the CRISPR screen?", "groups"),
    ("What do the gene programs mean?", "markers"),
    ("Which ligand-receptor pairs stand out?", "groups"),
    ("What did the bulk deconvolution find?", "groups"),
])
def test_named_results_add_their_facet_to_a_classified_route(q, facet):
    assert facet in named_result_facets(q)
    picked = Intent(sources=["data"], reason="llm", data_facets=["qc"])
    out = apply_rules(picked, q, None, RouteState(has_dataset=True, has_results=True))
    assert facet in out.data_facets and "qc" in out.data_facets


@pytest.mark.parametrize("q", [
    "Which cytokines are the monocytes responding to?",
    "What does the ligand-receptor analysis show?",
    "Is there cell-cell communication between T cells and monocytes?",
])
def test_named_result_adds_the_dataset_to_a_literature_only_route(q):
    lit_only = Intent(sources=["literature"], reason="llm", data_facets=[])
    st = RouteState(rag_enabled=True, has_dataset=True, has_results=True)
    out = apply_rules(lit_only, q, None, st)
    assert "data" in out.sources and "literature" in out.sources
    assert out.data_facets == ["groups"]
    # no stored results: the classifier's route stands
    bare = RouteState(rag_enabled=True, has_dataset=True)
    assert apply_rules(lit_only, q, None, bare) is lit_only


def test_all_facets_stay_all():
    picked = Intent(sources=["data"], reason="llm", data_facets=[])
    out = apply_rules(picked, "Summarise the cytokine run", None, RouteState(has_dataset=True))
    assert out.data_facets == []


# --- what is on screen, what next, methods --------------------------------------

DE_VC = {"panel": "Unified View", "subtab": "de",
         "on_screen": {"kind": "de", "id": "abc", "label": "CD14 Mono: STIM vs CTRL (Wilcoxon)"}}
ST = RouteState(rag_enabled=False, has_dataset=True, has_results=True)


def test_explain_this_result_routes_to_the_result_on_screen():
    i = pre_route("Explain this result in plain words. What should I check before I believe it?",
                  DE_VC, ST)
    assert i is not None and i.sources == ["data"] and i.data_facets == ["groups"]
    enr = {"panel": "Unified View", "subtab": "enrichment"}
    assert pre_route("What does this table tell me?", enr, ST).data_facets == ["markers", "enrichment"]
    cmp_vc = {"panel": "Compare", "on_screen": {"kind": "compare", "tab": "milode"}}
    assert pre_route("Explain this result", cmp_vc, ST).data_facets == ["groups"]
    # nothing on screen: left to the classifier
    assert pre_route("Explain this result", {"panel": "Data"}, ST) is None


def test_what_next_routes_to_the_dataset_and_the_app():
    i = pre_route("What should I do next with this dataset?", {"panel": "Unified View"}, ST)
    assert i.sources == ["data", "app"] and set(i.data_facets) >= {"qc", "groups"}
    assert pre_route("What should I do next?", None, RouteState()) is None  # no dataset


@pytest.mark.parametrize("q", [
    "Write the methods paragraph for my paper.", "Write the methods for this analysis.",
    "Can you draft a methods section?", "Give me the methods text", "methods paragraph please",
])
def test_methods_requests(q):
    assert wants_methods(q)


@pytest.mark.parametrize("q", [
    "Which DE methods should I use?", "What methods are available for doublets?",
    "Write methods to use for DE?", "How was this clustered?",
])
def test_not_methods_requests(q):
    assert not wants_methods(q)


def _register(dm, did: str) -> str:
    """A small ready dataset in the manager's data dir (as test_assistant.py's)."""
    from pathlib import Path

    import anndata as ad
    import numpy as np
    import pandas as pd

    d = Path(dm.data_dir) / "uploads" / did
    d.mkdir(parents=True, exist_ok=True)
    (d / "metadata.json").write_text(json.dumps({"id": did, "name": did, "status": "ready"}))
    rng = np.random.default_rng(0)
    a = ad.AnnData(rng.poisson(0.3, size=(60, 20)).astype("float32"))
    a.var_names = [f"G{i}" for i in range(20)]
    a.obs_names = [f"C{i}" for i in range(60)]
    a.obs["cluster"] = pd.Categorical(["A"] * 30 + ["B"] * 30)
    a.write_h5ad(d / "data.h5ad")
    return did


async def test_methods_typed_in_chat_take_the_buttons_route(client, dataset_manager):
    did = _register(dataset_manager, "meth1")
    r = await client.post(f"{API}/datasets/{did}/assistant/chat",
                          json={"query": "Write the methods paragraph for my paper."})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["route"] == ["provenance"]
    button = await client.post(f"{API}/datasets/{did}/assistant/methods", json={})
    assert body["answer"] == button.json()["methods"]
    # the stream gives the same text in one delta
    r = await client.post(f"{API}/datasets/{did}/assistant/chat-stream",
                          json={"query": "Write the methods for this analysis."})
    events = [json.loads(ln[6:]) for ln in r.text.splitlines() if ln.startswith("data: ")]
    assert events[0]["route"] == ["provenance"]
    assert "".join(e["text"] for e in events if e["type"] == "delta") == body["answer"]


# --- methods: never a step the recipe lacks -------------------------------------

RECIPE = """Dataset: 13,836 cells x 13,912 genes.
Recorded analysis steps (in order):
- qc_metrics (tool: scanpy.pp.calculate_qc_metrics)
- doublet_detection (tool: scanpy.pp.scrublet)
- normalization (tool: scanpy.pp.normalize_total) — target_sum=10000.0
- pca (tool: scanpy.pp.pca) — n_comps=50
- clustering (tool: scanpy.tl.leiden) — resolution=0.5"""

# The wrong template the co-pilot wrote (2026-10-26 captures), plus a true negation.
WRONG = ("Quality control, normalization, highly variable gene selection, PCA and batch "
         "correction were performed using scView's guided pipeline. Doublets were detected with "
         "Scrublet. Cells were clustered with Leiden at resolution 0.5.\n\nData were integrated "
         "with Harmony before clustering. No batch correction was applied.")


def test_methods_guard_drops_sentences_claiming_unrecorded_steps():
    assert methods_guard.unrecorded_steps(WRONG, RECIPE) == ["batch correction"]
    text, dropped = methods_guard.keep_recorded(WRONG, RECIPE)
    assert dropped == ["batch correction"]
    assert "batch correction were performed" not in text and "Harmony" not in text
    assert "Doublets were detected with Scrublet." in text
    assert "No batch correction was applied." in text  # true, so kept
    # recorded: kept
    with_batch = RECIPE + "\n- batch_correction (tool: harmonypy) — batch_key=donor"
    assert methods_guard.keep_recorded(WRONG, with_batch) == (WRONG, [])


async def test_write_methods_never_claims_an_unrecorded_step(monkeypatch, tmp_path):
    import openai

    from scview.core import assistant

    class _Fake:
        def __init__(self, *a, **k):
            class _C:
                async def create(self, **kw):
                    msg = SimpleNamespace(content=WRONG)
                    return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

            self.chat = SimpleNamespace(completions=_C())

    monkeypatch.setattr(openai, "AsyncOpenAI", _Fake)
    monkeypatch.setattr(assistant, "_methods_skeleton", lambda adaptor: RECIPE)
    res = await assistant.write_methods(object(), None, "key", "m")
    assert res.grounded and "batch" not in res.methods.replace("No batch correction", "")
    assert "Harmony" not in res.methods and "Leiden at resolution 0.5" in res.methods


# --- the result on screen --------------------------------------------------------


def test_de_on_screen_gives_its_numbers(monkeypatch):
    from scview.core import de_results, results_store

    G = SimpleNamespace
    genes = [G(gene="ISG15", logfoldchange=4.8, pval=1e-30, pval_adj=1e-28),
             G(gene="CXCL10", logfoldchange=5.5, pval=1e-20, pval_adj=1e-18),
             G(gene="FOS", logfoldchange=-2.1, pval=1e-6, pval_adj=1e-5),
             G(gene="ACTB", logfoldchange=0.1, pval=0.5, pval_adj=0.9)]
    rid = de_results.remember("dsS", genes, comparison={
        "mode": "groups", "column": "stim", "group_a": "STIM", "group_b": "CTRL",
        "within_column": "cluster", "within_value": "CD14 Mono", "n_a": 2011, "n_b": 2151})
    monkeypatch.setattr(results_store, "dataset_id_of", lambda adaptor: "dsS")
    vc = {"panel": "Unified View", "subtab": "de", "on_screen": {"kind": "de", "id": rid,
                                                                  "label": "CD14 Mono: STIM vs CTRL"}}
    lines, detail = screen_facts.fact_lines(object(), vc)
    text = "\n".join(lines)
    assert "STIM vs CTRL in 'stim', within cluster = CD14 Mono; 2,011 vs 2,151 cells" in text
    assert "2 genes up and 1 down in STIM" in text
    assert "CXCL10 (+5.50), ISG15 (+4.80)" in text and "FOS (-2.10)" in text
    assert detail == "CD14 Mono: STIM vs CTRL (Wilcoxon): 2 up, 1 down"
    view = "\n".join(screen_facts.view_lines(vc))
    assert "Unified View tab open: DE" in view and "CD14 Mono: STIM vs CTRL" in view
    # an unknown id or another dataset: nothing
    assert screen_facts.fact_lines(object(), {"on_screen": {"kind": "de", "id": "nope"}}) == ([], "")


def test_enrichment_on_screen_reads_the_stored_terms():
    import anndata as ad
    import numpy as np

    from scview.core.uns_keys import enrichment_key

    a = ad.AnnData(np.zeros((2, 2), dtype="float32"))
    a.uns[enrichment_key("cluster", "B")] = json.dumps([
        {"term": "HALLMARK_INTERFERON_ALPHA_RESPONSE", "adjusted_pvalue": 1e-9, "overlap_count": 12},
        {"term": "KEGG_RIBOSOME", "adjusted_pvalue": 0.4, "overlap_count": 2},
    ])
    adaptor = SimpleNamespace(adata=a)
    vc = {"subtab": "enrichment", "on_screen": {"kind": "enrichment", "column": "cluster", "group": "B"}}
    lines, _ = screen_facts.fact_lines(adaptor, vc)
    assert "2 gene sets tested, 1 with adjusted p < 0.05" in lines[1]
    assert lines[1].index("INTERFERON") < lines[1].index("RIBOSOME")


def test_screen_tag_resolves_in_the_ledger():
    from scview.core.claims import resolve_tag

    assert resolve_tag("result:screen", ["result:screen", "result:groups:stim"]) == ["result:screen"]
    assert resolve_tag("screen", ["result:screen"]) == ["result:screen"]


@pytest.mark.parametrize("q", [
    "What pathways are enriched in the monocyte cluster?",
    "Summarise the quality control that was done",
    "How many donors are there, and is this a paired design?",
    "What is cluster 3?",
    "Is my data good enough to publish?",
])
def test_a_question_about_the_open_dataset_keeps_the_dataset(q):
    lit_only = Intent(sources=["literature"], reason="llm", data_facets=[])
    out = apply_rules(lit_only, q, None, RouteState(rag_enabled=True, has_dataset=True))
    assert "data" in out.sources and "literature" in out.sources


@pytest.mark.parametrize("q", [
    "What does interferon beta do to monocytes?",
    "How do NK cells kill infected cells?",
])
def test_a_general_biology_question_stays_with_the_literature(q):
    lit_only = Intent(sources=["literature"], reason="llm", data_facets=[])
    assert apply_rules(lit_only, q, None, RouteState(rag_enabled=True, has_dataset=True)) is lit_only
