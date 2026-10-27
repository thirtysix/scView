"""Claims ledger: deterministic checks of co-pilot sentences against cited facts."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from scview.core import claims
from scview.core.assistant import ChatSource
from scview.core.claims import Vocab, check_answer, resolve_tag, split_claims

GENES = frozenset({"CD79A", "MS4A1", "CD74", "FTL", "CD63", "LGALS3", "FCER1G", "GNLY",
                   "NKG7", "CD14", "CD4", "HLA-DPA1", "C15orf48", "ISG15", "TF", "Mpo", "Elane"})
VOCAB = Vocab(
    genes=GENES,
    groups=["B", "CD14 Mono", "NK", "pDC", "Eryth", "STIM", "CTRL", "Monocyte", "B cell",
            "CD4 Naive T", "Neu"],
    numeric_groups=frozenset({"0", "1", "2", "3"}),
)


def _src(ref, evidence, kind="result"):
    return ChatSource(kind=kind, ref=ref, detail=evidence[:60], evidence=evidence)


SOURCES = [
    _src("dataset:summary", "- Cells: 13,836 | Genes: 14,053 | Species: human", "dataset"),
    _src("provenance:recipe", "- pca (n_comps=50)\n- doublet_detection (method=scrublet, "
         "expected_doublet_rate=0.06)", "provenance"),
    _src("result:groups:stim", "- **stim**: STIM (7,263), CTRL (6,573)"),
    _src("result:groups:cluster", "- **cluster**: CD14 Mono (4,225), CD4 Naive T (2,600), "
         "B (998), NK (646), pDC (108), Eryth (60)"),
    _src("result:celltypes:cell_type", "- cluster **B** = B cell\n- cluster **CD14 Mono** = Monocyte"),
    _src("result:qc:cluster", "## Per-cluster QC (deterministic; column 'cluster', 13 groups)\n"
         "- Typical cluster (median across clusters): 610 genes/cell, 1,510 counts/cell, "
         "0% doublets [result:qc:cluster]"),
    _src("result:markers:B", "Top markers by cluster\n- **B**: CD79A, CD74, MS4A1, HLA-DPA1"),
    _src("result:markers:NK", "Top markers by cluster\n- **NK**: GNLY, NKG7"),
    _src("result:markers:CD14 Mono", "Top markers by cluster\n- **CD14 Mono**: FTL, CD63, "
         "C15orf48, LGALS3, FCER1G"),
    _src("result:composition", "- CTRL vs STIM by 'cluster', samples from 'Donor': 0 of 6 groups "
         "at FDR < 0.05 [result:composition].\n- CD14 Mono: 35.0% -> 30.1% (CTRL -> STIM), FDR 0.056"),
    _src("lit:PMID:29227470", "Kang 2018\nMultiplexed droplet sequencing of 14,000 PBMCs "
         "from eight lupus patients, with ISG15 induced by interferon.", "literature"),
]
REFS = [s.ref for s in SOURCES]
CONTEXT = "\n".join(s.evidence for s in SOURCES)


def ledger(answer, sources=SOURCES, vocab=VOCAB):
    return check_answer(answer, sources, CONTEXT, vocab)


def one(answer, **kw):
    out = ledger(answer, **kw)
    assert len(out) == 1, [c.text for c in out]
    return out[0]


# --- citations -------------------------------------------------------------


@pytest.mark.parametrize("tag, want", [
    ("result:markers:B", ["result:markers:B"]),
    ("obs:stim", ["result:groups:stim"]),
    ("dataset", ["dataset:summary"]),
    ("provenance", ["provenance:recipe"]),
    ("result:annotation:cell_type", ["result:celltypes:cell_type"]),
    ("result : qc : cluster", ["result:qc:cluster"]),
    ("composition note", ["result:composition"]),
    ("lit:PMID:29227470", ["lit:PMID:29227470"]),
    ("lit:PMID:1", []),
    ("result:groups:nope", []),
])
def test_resolve_tag(tag, want):
    assert resolve_tag(tag, REFS) == want


def test_resolve_tag_marker_column_means_all_marker_facts():
    got = resolve_tag("result:markers:cluster", REFS)
    assert got == ["result:markers:B", "result:markers:NK", "result:markers:CD14 Mono"]


# --- splitting -------------------------------------------------------------


def test_list_items_inherit_the_lead_in_citations():
    parts = split_claims("Cell counts [obs:cluster]:\n\n- **B**: 998\n- **NK**: 646 [obs:other]\n\n"
                         "Plain closing line.")
    assert parts[0] == ("Cell counts :", ["obs:cluster"], True)
    assert parts[1][1] == ["obs:cluster"] and parts[2][1] == ["obs:other"]
    assert parts[3][1] == []


def test_numbered_list_markers_are_not_sentences():
    parts = split_claims("Steps [provenance]:\n10. **PCA** with n_comps=50\n11. Scaling")
    assert [p[0] for p in parts[1:]] == ["**PCA** with n_comps=50", "Scaling"]


def test_citation_after_the_period_belongs_to_that_sentence():
    parts = split_claims("STIM has 7,263 cells. [obs:stim] CTRL has 6,573 cells [obs:stim].")
    assert [p[1] for p in parts] == [["obs:stim"], ["obs:stim"]]


# --- verdicts --------------------------------------------------------------


def test_supported_numbers_and_genes():
    c = one("STIM has 7,263 cells and CTRL 6,573 [result:groups:stim].")
    assert c.verdict == "supported" and "7,263" in c.reason
    c = one("B cells are marked by `CD79A` and `MS4A1` [result:markers:B].")
    assert c.verdict == "supported"
    assert "CD79A" in c.evidence


def test_wrong_count_for_a_group_is_refuted():
    c = one("STIM has 7,500 cells [result:groups:stim].")
    assert c.verdict == "refuted"
    assert "7,263" in c.reason


def test_a_number_belonging_to_another_group_is_refuted():
    # 108 is on the same line, but it is pDC's count, not Eryth's.
    c = one("Eryth is the smallest cluster at 108 cells [result:groups:cluster].")
    assert c.verdict == "refuted" and "60" in c.reason


@pytest.mark.parametrize("text", [
    "Clusters include CD14 Mono 4,225; CD4 Naive T 2,600; B 998 [result:groups:cluster].",
    "There are 4,225 CD14 Mono, 2,600 CD4 Naive T and 998 B cells [result:groups:cluster].",
    "STIM (7,263 cells) and CTRL (6,573 cells) [result:groups:stim].",
])
def test_lists_of_group_counts_pair_each_number_with_its_group(text):
    assert one(text).verdict == "supported"


@pytest.mark.parametrize("text", [
    "STIM has ~4.8× as many cells as CTRL [result:groups:stim].",
    "STIM has over 9,000 cells [result:groups:stim].",
    "STIM is one of 2 conditions, at about 8,000 cells [result:groups:stim].",
])
def test_ratios_bounds_and_hedges_never_refute(text):
    assert one(text).verdict == "uncertain"


def test_round_and_approximate_numbers_match_at_their_precision():
    assert one("STIM has 7,000 cells [result:groups:stim].").verdict == "supported"
    assert one("STIM has about 7,500 cells [result:groups:stim].").verdict == "supported"
    assert one("STIM has 7,300 cells [result:groups:stim].").verdict == "supported"
    assert one("STIM has 7,250 cells [result:groups:stim].").verdict == "refuted"


def test_population_names_are_not_genes():
    c = one("The CD14 monocytes and CD4+ T cells look distinct [result:groups:cluster].")
    assert c.reason.startswith("No numbers")
    # "Monocyte" is a group name here, so it is masked before the gene pass.
    c = one("The CD14 monocyte markers are distinct [result:groups:cluster].")
    assert c.reason.startswith("No numbers")
    assert one("`CD14` is high in CD14 Mono [result:markers:CD14 Mono].").verdict == "uncertain"


def test_wrong_typical_value_is_refuted_by_its_unit():
    c = one("The typical cluster has 650 genes/cell [result:qc:cluster].")
    assert c.verdict == "refuted"


def test_rounding_and_suffixes_match():
    assert one("A typical cluster has about 1.5k counts/cell [result:qc:cluster].").verdict == "supported"
    assert one("CD14 Mono went 35% to 30% (FDR 0.06) [result:composition].").verdict == "supported"


def test_wrong_fdr_for_a_group_is_refuted():
    c = one("For CD14 Mono the FDR is 0.01 [result:composition].")
    assert c.verdict == "refuted"


def test_share_total_and_count_are_recomputed_from_one_line():
    # 4,225 of 8,637 listed cells = 48.9%
    assert one("CD14 Mono is 49% of these cells [result:groups:cluster].").verdict == "supported"
    assert one("Together they hold 8,637 cells [result:groups:cluster].").verdict == "supported"
    assert one("Together they hold 8,237 cells [result:groups:cluster].").verdict == "uncertain"
    assert one("There are 6 clusters listed [result:groups:cluster].").verdict == "supported"
    # A share is never refuted: the model may have used another denominator.
    assert one("CD14 Mono is 70% of these cells [result:groups:cluster].").verdict == "uncertain"


def test_percent_and_fraction_are_the_same_number():
    c = one("The expected doublet rate was 6% [provenance:recipe].")
    assert c.verdict == "supported"


def test_gene_missing_from_the_cited_group_is_uncertain_not_refuted():
    c = one("NK cells express `CD79A` [result:markers:NK].")
    assert c.verdict == "uncertain" and "CD79A" in c.reason


def test_gene_must_sit_with_the_group_it_is_claimed_for():
    # CD79A is in the facts, but on B's line; the claim gives it to NK.
    c = one("`CD79A` marks NK cells.")
    assert c.verdict == "uncertain"


def test_a_gene_list_belongs_to_the_group_before_it():
    c = one("NK is set apart by `GNLY`, `NKG7`, while B carries `CD79A`, `MS4A1` "
            "[result:markers:NK] [result:markers:B].")
    assert c.verdict == "supported"


def test_right_fact_wrong_citation_is_supported_with_a_note():
    c = one("B has 998 cells [result:markers:B].")
    assert c.verdict == "supported" and "not the one cited" in c.reason
    # The second pass never refutes: the cited fact simply lacks the number.
    assert one("B has 1,234 cells [result:markers:B].").verdict == "uncertain"


def test_negation_and_comparison_cap_at_uncertain():
    c = one("`CD79A` is not a marker of B [result:markers:B].")
    assert c.verdict == "uncertain" and "negation" in c.reason
    c = one("CD14 Mono fell from 30.1% to 35% [result:composition].")
    assert c.verdict == "uncertain" and "comparison" in c.reason


def test_negation_never_hides_a_conflict():
    assert one("STIM does not have 9,000 cells, it has 7,500 [result:groups:stim].").verdict == "refuted"


def test_qualitative_cited_sentence_is_uncertain_and_prose_is_skipped():
    out = ledger("The data looks clean [result:qc:cluster]. You could try a DE test next.")
    assert len(out) == 1
    assert out[0].verdict == "uncertain" and "needs a reading" in out[0].reason


def test_uncited_numbers_are_checked_against_all_facts():
    c = one("The dataset has 13,836 cells.")
    assert c.verdict == "supported" and "No citation" in c.reason
    assert one("The dataset has 12,000 cells.").verdict == "uncertain"


def test_unknown_citation_is_named():
    c = one("There are 13,836 cells [result:cells:all].")
    assert c.refs == [] and "isn't one of the facts" in c.reason
    assert c.verdict == "supported"  # the number itself is in the facts


def test_alias_citations_resolve_before_checking():
    c = one("STIM (7,263 cells) and CTRL (6,573 cells) [obs:stim].")
    assert c.refs == ["result:groups:stim"] and c.verdict == "supported"


def test_literature_claim_checks_atoms_against_the_abstract():
    assert one("`ISG15` is induced by interferon [lit:PMID:29227470].").verdict == "supported"
    c = one("Interferon reshapes monocytes [lit:PMID:29227470].")
    assert c.verdict == "uncertain" and "literature" in c.reason


def test_numeric_cluster_labels_anchor_numbers():
    vocab = Vocab(genes=GENES, groups=[], numeric_groups=frozenset({"0", "3"}))
    srcs = [_src("result:groups:leiden", "- **leiden**: 0 (900), 3 (512)")]
    assert one("Cluster 3 has 512 cells [result:groups:leiden].", sources=srcs, vocab=vocab).verdict == "supported"
    assert one("Cluster 3 has 900 cells [result:groups:leiden].", sources=srcs, vocab=vocab).verdict == "refuted"


def test_dates_versions_gene_digits_and_ordinals_are_not_numbers():
    c = one("Harmony ran on 2026-06-10 with scView 0.18.0 and `C15orf48` led CD14 Mono "
            "[result:markers:CD14 Mono].")
    assert c.verdict == "supported" and c.reason.count(",") == 0


def test_title_case_mouse_symbol_counts_as_a_gene_only_in_code():
    srcs = [_src("result:markers:Neu", "Top markers by cluster\n- **Neu**: Mpo, Elane")]
    assert one("Neu is marked by `Mpo` [result:markers:Neu].", sources=srcs).verdict == "supported"
    # Untagged "Mpo" in prose is not taken as a gene, so nothing hard to check.
    assert one("Neu is marked by Mpo [result:markers:Neu].", sources=srcs).verdict == "uncertain"


def test_stoplisted_symbol_is_not_a_gene_unless_in_code():
    assert one("TF activity was run [provenance:recipe].").reason.startswith("No numbers")


def test_no_vocabulary_still_checks_numbers():
    c = one("There are 13,836 cells [dataset:summary].", vocab=Vocab())
    assert c.verdict == "supported"


CYTO = [_src("result:cytokines", "- T activated: IFN1 z=42.1, IFNG z=25.6\n- DC: IFN1 z=40.8, IFNG z=25.2")]
CYTO_VOCAB = Vocab(genes=GENES | {"IFNG", "ISG15"}, groups=["T activated", "DC"])


def test_a_qualitative_sentence_is_not_supported_by_one_shared_gene():
    # The 2026-10-26 capture: "supported" only because it shares IFNG with the fact.
    c = one("**Interferon response is broad and strong across nearly all cell types** — the top "
            "CytoSig signal in every cluster is **IFN1**, with **IFNG** second [result:cytokines].",
            sources=CYTO, vocab=CYTO_VOCAB)
    assert c.verdict == "uncertain" and "says more than that" in c.reason
    # a number match still supports it; a plain gene list still does too
    assert one("In T activated, IFNG z=25.6 [result:cytokines].", sources=CYTO,
               vocab=CYTO_VOCAB).verdict == "supported"
    assert one("- **`GNLY`**, **`NKG7`** — cytotoxic granule genes [result:markers:NK].").verdict == "supported"


def test_a_count_before_a_listed_group_is_not_that_groups_number():
    # The capture's "refuted" was a misparse: "says ATF2 15; the fact has 0".
    srcs = [_src("result:mixscape", "- Mixscape: KO fraction per target: ATF2: 0 of 995 cells KO (0%), "
                                    "0 DE genes; CAV1: 0 of 792 cells KO (0%), 0 DE genes")]
    vocab = Vocab(genes=GENES, groups=["ATF2", "CAV1", "NT"])
    c = one("**No effect detected — 15 targets** (ATF2, CAV1): no KO cells [result:mixscape].",
            sources=srcs, vocab=vocab)
    assert c.verdict != "refuted", c.reason
    assert one("ATF2 has 995 cells [result:mixscape].", sources=srcs, vocab=vocab).verdict == "supported"
    assert one("ATF2 has 500 cells [result:mixscape].", sources=srcs, vocab=vocab).verdict == "refuted"


def test_evidence_falls_back_to_detail():
    srcs = [ChatSource(kind="app", ref="app:library", detail="4 dataset(s): a, b, c, d")]
    assert one("There are 4 datasets in the library [app:library].", sources=srcs).verdict == "supported"


# --- judge -------------------------------------------------------------------


class _FakeClient:
    def __init__(self, reply):
        self.reply = reply
        self.sent = None
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    async def create(self, **kw):
        self.sent = kw
        if kw.get("stream"):
            return self._stream()
        msg = SimpleNamespace(content=self.reply)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

    async def _stream(self):
        for word in self.reply.split(" "):
            yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=word + " "))])


def _patch_openai(monkeypatch, client):
    import openai

    monkeypatch.setattr(openai, "AsyncOpenAI", lambda **_kw: client)


def test_judge_parses_verdicts_and_keeps_unknown_ids_uncertain(monkeypatch):
    reply = ('Here: [{"id": 0, "verdict": "supported", "reason": "sum of listed counts"},'
             ' {"id": 1, "verdict": "bogus"}, {"id": 9, "verdict": "refuted"}]')
    client = _FakeClient(reply)
    _patch_openai(monkeypatch, client)
    got = asyncio.run(claims.judge_claims(
        [{"id": 0, "text": "Total 8,237", "evidence": "B (1)"},
         {"id": 1, "text": "x", "evidence": ""}],
        "key", "deepseek-ai/DeepSeek-V4.1-Flash", "http://x"))
    assert got[0] == {"id": 0, "verdict": "supported", "reason": "sum of listed counts"}
    assert got[1]["verdict"] == "uncertain"
    assert client.sent["temperature"] == 0
    assert "enable_thinking" in json.dumps(client.sent.get("extra_body"))
    assert "Claim 0" in client.sent["messages"][1]["content"]


KANG_TYPES = ["CD4 Naive T", "CD14 Mono", "CD14 Mono", "B", "CD8 T", "NK", "FCGR3A Mono",
              "CD14 Mono", "DC", "pDC", "Eryth", "Mk", "B Activated"]


def test_the_judge_reads_the_fact_lines_the_claim_names(monkeypatch):
    """Capture 2026-10-26: "Monocyte clusters are 1, 2, 6 and 7" (true) was refuted
    because the judge saw only the first 6 lines of the cited fact (clusters 0-5).
    The rules leave it uncertain; the judge now gets the lines about clusters 1, 2,
    6 and 7, whatever their place in the fact."""
    ev = "\n".join(f"- cluster **{i}** = {t}" for i, t in enumerate(KANG_TYPES))
    src = _src("result:celltypes:cell_type", ev)
    vocab = Vocab(genes=GENES, groups=sorted(set(KANG_TYPES) | {"CTRL", "STIM"}),
                  numeric_groups=frozenset(str(i) for i in range(len(KANG_TYPES))))
    (c,) = check_answer("Monocyte clusters are 1, 2, 6 and 7 [result:celltypes:cell_type].",
                        [src], ev, vocab)
    assert c.verdict == "uncertain" and c.refs == ["result:celltypes:cell_type"]
    assert c.evidence.splitlines() == [
        "- cluster **1** = CD14 Mono", "- cluster **2** = CD14 Mono",
        "- cluster **6** = FCGR3A Mono", "- cluster **7** = CD14 Mono"]
    # the judge's request carries those lines (a claim's evidence is capped in size)
    client = _FakeClient('[{"id": 0, "verdict": "supported", "reason": "listed"}]')
    _patch_openai(monkeypatch, client)
    asyncio.run(claims.judge_claims([c.model_dump()], "k", "m", "u"))
    sent = client.sent["messages"][1]["content"]
    assert "cluster **6** = FCGR3A Mono" in sent and "cluster **7** = CD14 Mono" in sent
    # a claim naming many groups keeps its evidence small
    many = ", ".join(str(i) for i in range(13))
    (c,) = check_answer(f"The clusters are {many} [result:celltypes:cell_type].", [src], ev, vocab)
    assert len(c.evidence) <= claims.MAX_EVIDENCE_CHARS + 1 and len(c.evidence.splitlines()) <= 16


def test_judge_survives_garbage(monkeypatch):
    _patch_openai(monkeypatch, _FakeClient("no json [here"))
    got = asyncio.run(claims.judge_claims([{"id": 3, "text": "t", "evidence": "e"}], "k", "m", "u"))
    assert got == [{"id": 3, "verdict": "uncertain", "reason": "The model gave no verdict."}]


# --- endpoint + chat wiring --------------------------------------------------

API = "/api/v1"


async def test_check_claims_needs_a_model(client):
    r = await client.post(f"{API}/assistant/check-claims",
                          json={"claims": [{"id": 0, "text": "t", "evidence": "e"}]})
    assert r.status_code == 503


async def test_check_claims_caps_input(client):
    many = [{"id": i, "text": "t", "evidence": "e"} for i in range(claims.MAX_JUDGE_CLAIMS + 1)]
    assert (await client.post(f"{API}/assistant/check-claims", json={"claims": many})).status_code == 413
    assert (await client.post(f"{API}/assistant/check-claims", json={"claims": []})).status_code == 400
    long = [{"id": 0, "text": "x" * 5000, "evidence": ""}]
    assert (await client.post(f"{API}/assistant/check-claims", json={"claims": long})).status_code == 422


async def test_check_claims_returns_verdicts(client, test_settings, monkeypatch):
    monkeypatch.setattr(test_settings, "DEEPINFRA_API_KEY", "k")
    _patch_openai(monkeypatch, _FakeClient('[{"id": 4, "verdict": "refuted", "reason": "no"}]'))
    r = await client.post(f"{API}/assistant/check-claims",
                          json={"claims": [{"id": 4, "text": "t", "evidence": "e"}]})
    assert r.status_code == 200, r.text
    assert r.json()["claims"] == [{"id": 4, "verdict": "refuted", "reason": "no"}]


def test_answers_carry_a_ledger_and_sources_hide_evidence(monkeypatch, tmp_path):
    import anndata as ad
    import numpy as np
    import pandas as pd

    from scview.core.anndata_adaptor import AnnDataAdaptor
    from scview.core.assistant import answer_query, stream_answer

    a = ad.AnnData(np.ones((60, 3), dtype="float32"))
    a.var_names = ["CD79A", "MS4A1", "GNLY"]
    a.obs["cluster"] = pd.Categorical(["B"] * 40 + ["NK"] * 20)
    p = tmp_path / "d.h5ad"
    a.write_h5ad(p)
    adaptor = AnnDataAdaptor(str(p))
    _patch_openai(monkeypatch, _FakeClient("B has 40 cells and NK has 30 [result:groups:cluster]."))

    res = asyncio.run(answer_query("how many?", adaptor, "key", model="m"))
    assert [c.verdict for c in res.claims] == ["refuted"]
    dumped = res.model_dump()
    assert all("evidence" not in s for s in dumped["sources"])
    assert dumped["claims"][0]["evidence"].startswith("- **cluster**")

    async def collect():
        return [ev async for ev in stream_answer("how many?", adaptor, "key", model="m")]

    events = asyncio.run(collect())
    assert all("evidence" not in s for s in events[0]["sources"])
    assert events[-1]["type"] == "done" and events[-1]["claims"][0]["verdict"] == "refuted"
