"""Claims ledger for Write methods: the model's paragraph against the recipe it was given."""

from __future__ import annotations

import json
from types import SimpleNamespace

from scview.core import claims
from scview.core.claims import METHODS_REF, check_answer, check_methods
from tests.test_claims import CONTEXT, SOURCES, VOCAB
from tests.test_copilot_routing import _register

# assistant._methods_skeleton's form (a real dataset's, trimmed)
RECIPE = """Dataset: 1,113 cells x 15,246 genes.
Source: IFN-beta treated and control PBMCs from 8 donors
Recorded analysis steps (in order):
- qc_metrics (tool: scanpy.pp.calculate_qc_metrics)
- doublet_detection (tool: scanpy.pp.scrublet (per capture channel)) — doublet_method=scrublet, expected_doublet_rate=0.06
- filtering (tool: scview.qc_filter + scanpy.pp.filter_genes) — qc_threshold_mode=mad, min_cells=3, max_pct_mt=20.0, \
drop_doublets=False, cell thresholds: outliers below median - 5.0 MADs of log1p counts and genes and above median + \
3.0 MADs of % mito, within each sample (8 samples): counts per cell at least 412-655; genes per cell at least 210-301; \
% mito below 7.9-12.4
- normalization (tool: scanpy.pp.normalize_total) — target_sum=10000.0
- highly_variable_genes (tool: scanpy.pp.highly_variable_genes) — n_top_genes=2000, hvg_flavor=seurat
- pca (tool: scanpy.pp.pca) — n_comps=50
- batch_correction (tool: scview Scanorama port (= scanorama 1.7.4 assemble, exact neighbours)) — batch_key=donor, \
batch_method=scanorama
- neighbors (tool: scanpy.pp.neighbors) — n_neighbors=15, n_comps=50
- clustering (tool: scanpy.tl.leiden/louvain) — clustering_method=leiden, clustering_resolution=0.5
- embeddings (tool: scanpy.tl.umap) — umap_min_dist=0.5
- marker_genes (tool: scanpy.tl.rank_genes_groups) — marker_columns=['scview_leiden_r0.5']"""


def one(text):
    out = check_methods(text, RECIPE)
    assert len(out) == 1, [c.text for c in out]
    return out[0]


def test_the_recorded_values_are_supported():
    c = one("Data from 8 donors (1,113 cells, 15,246 genes) were analysed with scanpy.")
    assert c.verdict == "supported" and "1,113" in c.reason and c.refs == [METHODS_REF]
    c = one("Counts were normalised to 10,000 per cell (10^4), and the 2,000 most highly variable genes "
            "were reduced to 50 principal components.")
    assert c.verdict == "supported", c.reason
    c = one("Leiden clustering (resolution 0.5) used a 15-nearest-neighbour graph built from 50 PCs.")
    assert c.verdict == "supported", c.reason
    c = one("Doublets were called with Scrublet at an expected rate of 6%.")
    assert c.verdict == "supported", c.reason


def test_a_changed_number_is_refuted():
    c = one("Highly variable genes (n = 3,000) were selected.")
    assert c.verdict == "refuted" and "says 3,000 for highly variable genes" in c.reason
    assert "n_top_genes=2000" in c.evidence
    c = one("A neighbourhood graph was built from 30 principal components with 15 neighbours.")
    assert c.verdict == "refuted" and "30 for principal components; the recipe has 50" in c.reason
    c = one("Cells were clustered with Leiden at a resolution of 1.0.")
    assert c.verdict == "refuted" and "the recipe has 0.5" in c.reason
    c = one("Cells with more than 15% mitochondrial reads were removed.")
    assert c.verdict == "refuted" and "15%" in c.reason
    c = one("The final dataset comprised 1,200 cells.")
    assert c.verdict == "refuted" and "1,113" in c.reason


def test_thresholds_written_as_ranges_and_mads():
    c = one("Cells more than 5 MADs below the median of log counts and genes, or more than 3 MADs above the "
            "median mitochondrial fraction, were removed within each sample (7.9-12.4% mito; at least 210 "
            "to 301 genes per cell and 412-655 counts per cell), and genes in fewer than 3 cells were dropped.")
    assert c.verdict == "supported", c.reason
    assert one("Outliers beyond 4 MADs were removed.").verdict == "refuted"


def test_a_changed_tool_or_version_is_refuted():
    c = one("Cells were clustered with the Louvain algorithm.")
    assert c.verdict == "refuted" and "names Louvain; the recipe has Leiden" in c.reason
    c = one("The 2,000 most variable genes were selected with the Seurat v3 flavour.")
    assert c.verdict == "refuted" and "Seurat v3" in c.reason
    c = one("Batch effects were corrected with Harmony.")
    assert c.verdict == "refuted" and "Scanorama" in c.reason
    assert one("Donors were integrated with Scanorama (v1.7.4).").verdict == "supported"
    c = one("Donors were integrated with Scanorama v1.6.")
    assert c.verdict == "refuted" and "the recipe has 1.7.4" in c.reason


def test_what_the_recipe_does_not_say_is_uncertain():
    c = one("Data were analysed with scanpy (v1.10.2).")
    assert c.verdict == "uncertain" and "scanpy 1.10.2" in c.reason
    # the judge reads the whole recipe for it
    assert "Dataset: 1,113 cells" in c.evidence and "n_comps=50" in c.evidence
    c = one("A total of 1,500 doublets were removed.")
    assert c.verdict == "uncertain" and "1,500" in c.reason
    assert one("Trajectories were inferred with PAGA.").verdict == "uncertain"
    # a negation is no tool claim
    assert one("No Harmony correction was applied.").verdict == "uncertain"


def test_prose_is_not_a_claim_and_spans_are_the_written_sentences():
    text = ("Cell-cycle phases were scored. Cells were clustered with Leiden at a resolution of 1.0.\n\n"
            "The full provenance recipe is embedded in the dataset.")
    out = check_methods(text, RECIPE)
    assert [c.verdict for c in out] == ["refuted"]
    assert out[0].span == "Cells were clustered with Leiden at a resolution of 1.0." and out[0].span in text
    assert check_methods(text, "") == []


def test_chat_claims_carry_their_written_sentence():
    answer = "**STIM** has 7,263 cells [result:groups:stim]. CTRL has 6,000 cells [result:groups:stim]."
    out = check_answer(answer, SOURCES, CONTEXT, VOCAB)
    assert [c.span for c in out] == ["**STIM** has 7,263 cells [result:groups:stim].",
                                     "CTRL has 6,000 cells [result:groups:stim]."]
    assert all(c.span in answer for c in out)


# --- write_methods and the routes that return it ------------------------------

WRONG = ("Cells were clustered with the Louvain algorithm at resolution 1.0. "
         "PCA was run with 50 components.")


def _fake_openai(monkeypatch, reply):
    import openai

    class _Fake:
        def __init__(self, *a, **k):
            class _C:
                async def create(self, **kw):
                    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=reply))])

            self.chat = SimpleNamespace(completions=_C())

    monkeypatch.setattr(openai, "AsyncOpenAI", _Fake)


async def test_write_methods_returns_its_ledger(monkeypatch):
    from scview.core import assistant

    _fake_openai(monkeypatch, WRONG)
    monkeypatch.setattr(assistant, "_methods_skeleton", lambda adaptor: RECIPE)
    res = await assistant.write_methods(object(), None, "key", "m")
    assert res.grounded and [c.verdict for c in res.claims] == ["refuted", "supported"]
    # no model: the recipe digest itself, nothing to check
    res = await assistant.write_methods(object(), None, "", "m")
    assert not res.grounded and res.claims == []


async def test_methods_routes_carry_the_claims(client, dataset_manager, monkeypatch):
    from scview.api.v1 import assistant as api
    from scview.core.assistant import MethodsResponse

    ledger = check_methods(WRONG, RECIPE)

    async def fake(*a, **k):
        return MethodsResponse(methods=WRONG, grounded=True, model="m", claims=ledger)

    monkeypatch.setattr(api, "write_methods", fake)
    did = _register(dataset_manager, "methclaims")
    body = (await client.post(f"/api/v1/datasets/{did}/assistant/methods", json={})).json()
    assert [c["verdict"] for c in body["claims"]] == ["refuted", "supported"]
    body = (await client.post(f"/api/v1/datasets/{did}/assistant/chat",
                              json={"query": "Write the methods paragraph."})).json()
    assert [c["verdict"] for c in body["claims"]] == ["refuted", "supported"]
    r = await client.post(f"/api/v1/datasets/{did}/assistant/chat-stream",
                          json={"query": "Write the methods for this analysis."})
    events = [json.loads(ln[6:]) for ln in r.text.splitlines() if ln.startswith("data: ")]
    done = next(e for e in events if e["type"] == "done")
    assert [c["span"] for c in done["claims"]] == [c.span for c in ledger]


def test_the_judge_prompt_still_fits_a_methods_claim():
    # Check claims sends {id, text, evidence}: a methods claim's evidence is the recipe, capped.
    (c,) = check_methods("A total of 1,500 doublets were removed.", RECIPE)
    assert len(c.evidence) <= claims.MAX_EVIDENCE_CHARS + 1
