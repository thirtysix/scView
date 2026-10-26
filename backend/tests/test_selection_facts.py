"""Co-pilot facts about a cell selection (core/selection_facts.py), the routing
rules for selection and "which cell types respond" questions (core/rag/router.py),
the claims check of a selection number, and the chat/DE endpoints that carry them."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from scipy.stats import false_discovery_control, mannwhitneyu

from scview.core import de_results, selection_facts as sf
from scview.core.claims import Vocab, check_answer, resolve_tag
from scview.core.rag import router
from scview.core.rag.router import Intent, RouteState, apply_rules, classify_intent, pre_route

API = "/api/v1"
IFN_Q = "Which cell types respond most to interferon, and which genes show it?"


@pytest.fixture(autouse=True)
def _fresh_caches():
    sf.clear_cache()
    router.clear_memo()
    de_results.clear()
    yield
    sf.clear_cache()
    router.clear_memo()
    de_results.clear()


def _adata(n: int = 200, g: int = 60, seed: int = 0) -> ad.AnnData:
    """Log-normalised data: the first 50 cells (the selection) express G0..G4 and
    the ribosomal RPL3 highly; G5 is lower in them."""
    rng = np.random.default_rng(seed)
    counts = rng.poisson(0.4, size=(n, g)).astype(np.float32)
    counts[:50, :5] += rng.poisson(8.0, size=(50, 5))
    counts[:50, 6] += rng.poisson(8.0, size=50)  # RPL3
    counts[50:, 5] += rng.poisson(6.0, size=n - 50)
    totals = counts.sum(1, keepdims=True)
    X = np.log1p(counts / np.maximum(totals, 1) * 1e4)
    a = ad.AnnData(sp.csr_matrix(X.astype(np.float32)))
    names = [f"G{i}" for i in range(g)]
    names[6] = "RPL3"
    a.var_names = names
    a.obs_names = [f"C{i}" for i in range(n)]
    a.obs["cluster"] = pd.Categorical(["A"] * 40 + ["B"] * 60 + ["C"] * 100)
    a.obs["stim"] = pd.Categorical((["CTRL", "STIM"] * n)[:n])
    a.obs["n_genes"] = np.arange(n, dtype=float)
    a.obs["n_counts"] = np.arange(n, dtype=float) * 10
    return a


def _cells(idx, n):
    return {"n_cells": n, "indices": [int(i) for i in idx]}


def _bitmask(idx, n):
    bits = np.zeros(n, dtype=bool)
    bits[list(idx)] = True
    return {"n_cells": n, "bitmask": base64.b64encode(np.packbits(bits, bitorder="little").tobytes()).decode()}


# ---------------------------------------------------------------------------
# Decoding and hashing
# ---------------------------------------------------------------------------


def test_decode_indices_and_bitmask_agree():
    idx = [0, 3, 9, 10, 77, 199]
    a = sf.decode_cells(_cells([9, 0, 3, 3, 10, 77, 199, 500, -1], 200), 200)
    b = sf.decode_cells(_bitmask(idx, 200), 200)
    assert a.tolist() == idx and b.tolist() == idx
    assert sf.selection_hash(a, 200) == sf.selection_hash(b, 200)
    assert sf.selection_hash(a, 200) != sf.selection_hash(a[:-1], 200)


def test_decode_refuses_other_cells_and_bad_payloads():
    with pytest.raises(ValueError, match="made on 150 cells"):
        sf.decode_cells(_cells([1, 2], 150), 200)
    with pytest.raises(ValueError):
        sf.decode_cells({"n_cells": 200, "bitmask": "not base64!"}, 200)
    with pytest.raises(ValueError):
        sf.decode_cells({"n_cells": 200}, 200)
    with pytest.raises(ValueError):
        sf.decode_cells([1, 2, 3], 200)


# ---------------------------------------------------------------------------
# Composition and QC
# ---------------------------------------------------------------------------


def test_composition_shares_and_enrichment_ratio():
    a = _adata()
    mask = np.zeros(a.n_obs, dtype=bool)
    mask[:50] = True  # 40 A + 10 B
    rows = sf.composition(a.obs, mask, "cluster")
    assert [r["category"] for r in rows] == ["A", "B"]
    top = rows[0]
    assert top["n"] == 40
    assert top["pct_selection"] == pytest.approx(80.0)
    assert top["pct_dataset"] == pytest.approx(20.0)
    assert top["ratio"] == pytest.approx(4.0)
    assert rows[1]["pct_selection"] == pytest.approx(20.0) and rows[1]["ratio"] == pytest.approx(20 / 30)


def test_composition_columns_pick_colour_by_then_cluster_and_condition():
    a = _adata()
    a.obs["Donor"] = pd.Categorical(["d1", "d2"] * 100)
    a.obs["barcode_like"] = [f"x{i}" for i in range(a.n_obs)]  # too many values
    assert sf.composition_columns(a.obs, "stim") == ["stim", "cluster", "Donor"]
    assert sf.composition_columns(a.obs, None) == ["cluster", "Donor", "stim"]
    assert "barcode_like" not in sf.composition_columns(a.obs, "barcode_like")


def test_qc_medians_selection_vs_rest():
    qc = {"n_genes": np.arange(10, dtype=float), "pct_mt": None}
    mask = np.zeros(10, dtype=bool)
    mask[:4] = True
    out = sf.qc_medians(qc, mask)
    assert out == {"n_genes": (1.5, 6.5)}


# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------


def test_markers_follow_the_shared_rules_and_scipy_wilcoxon():
    a = _adata()
    idx = np.arange(50)
    m = sf.selection_markers(a, idx)
    up = [g["gene"] for g in m["up"]]
    assert set(up[:5]) == {"G0", "G1", "G2", "G3", "G4"}
    assert "RPL3" not in up  # ribosomal genes are left out
    assert [g["gene"] for g in m["down"]][0] == "G5"
    for g in m["up"] + m["down"]:
        assert g["padj"] < 0.05 and g["pct_selection"] >= 0 and abs(g["log2fc"]) >= 0.25
    # p-values: scipy's asymptotic, tie-corrected rank-sum without continuity correction
    X = a.X.toarray()
    mask = np.zeros(a.n_obs, dtype=bool)
    mask[idx] = True
    j = list(a.var_names).index("G0")
    ref = mannwhitneyu(X[mask, j], X[~mask, j], use_continuity=False, method="asymptotic").pvalue
    col = sp.csc_matrix(a.X[:, [j]])
    z = sf._ranksum_z(col, 0, mask, 50)
    from scipy.stats import norm
    assert 2 * norm.sf(abs(z)) == pytest.approx(ref, rel=1e-9)
    # BH with the untested genes at p = 1 is never below BH over every gene's own p
    pvals = np.array([mannwhitneyu(X[mask, k], X[~mask, k], use_continuity=False,
                                   method="asymptotic").pvalue for k in range(a.n_vars)])
    full = false_discovery_control(np.nan_to_num(pvals, nan=1.0))
    g0 = next(g for g in m["up"] if g["gene"] == "G0")
    assert g0["padj"] >= full[j] - 1e-12


def test_vectorised_rank_sums_equal_the_column_by_column_ones():
    rng = np.random.default_rng(5)
    # tie-heavy: small integers, many zeros, an explicit zero and an all-zero column
    X = rng.poisson(0.7, size=(300, 12)).astype(np.float32)
    X[:, 7] = 0
    M = sp.csr_matrix(X)
    M.data[0] = 0.0  # an explicit zero is a zero
    mask = rng.random(300) < 0.3
    cand = np.array([0, 3, 7, 11, 5])
    many = sf._ranksum_z_many(M, cand, mask, int(mask.sum()))
    col = M[:, cand].tocsc()
    one = [sf._ranksum_z(col, k, mask, int(mask.sum())) for k in range(cand.size)]
    assert np.allclose(many, one, atol=1e-10)
    dense = M.toarray().astype(np.float64)  # scipy computes in the input dtype
    for k, j in enumerate(cand[:2]):
        ref = mannwhitneyu(dense[mask, j], dense[~mask, j], use_continuity=False, method="asymptotic")
        from scipy.stats import norm
        assert 2 * norm.sf(abs(many[k])) == pytest.approx(ref.pvalue, rel=1e-9)


def test_markers_from_counts_are_log_normalised_first():
    a = _adata()
    a.X = sp.csr_matrix(np.expm1(a.X.toarray()).round().astype(np.float32))  # counts-like
    up = [g["gene"] for g in sf.selection_markers(a, np.arange(50))["up"]]
    assert {"G0", "G1"} <= set(up)


def test_facts_are_cached_per_selection(monkeypatch):
    a = _adata()
    calls = []
    real = sf.selection_markers
    monkeypatch.setattr(sf, "selection_markers", lambda *x, **k: calls.append(1) or real(*x, **k))
    f1 = sf.facts(a, _cells(range(50), a.n_obs), dataset_id="ds")
    f2 = sf.facts(a, _bitmask(range(50), a.n_obs), dataset_id="ds")
    assert len(calls) == 1 and f1["hash"] == f2["hash"] and sf.cache_size() == 1
    sf.facts(a, _cells(range(60), a.n_obs), dataset_id="ds")
    assert len(calls) == 2 and sf.cache_size() == 2


def test_grounding_lines_name_numbers_and_stored_de():
    a = _adata()
    sel = {"n_cells": 50, "source": "gate:T cells", "cells": _cells(range(50), a.n_obs)}
    lines, detail = sf.grounding_lines(a, sel, color_by="cluster", dataset_id="ds")
    text = "\n".join(lines)
    assert detail == "50 selected cells (saved gate 'T cells') = 25.0% of 200"
    assert "[result:selection]" in lines[0]
    assert "  - A (cluster): 40 cells, 80.0% of the selection vs 20.0% of the dataset (4.0x)" in text
    assert "QC medians, selection vs rest: genes/cell 24 vs 124; counts/cell 245 vs 1,245" in text
    assert "Genes up in the selection" in text and "G0 (log2FC" in text
    assert "No DE result is stored for exactly these cells" in text
    # a stored DE for exactly these cells is cited
    h = sf.selection_hash(np.arange(50), a.n_obs)

    class Row:
        def __init__(self, gene, p):
            self.gene, self.logfoldchange, self.pval, self.pval_adj, self.stat = gene, 1.0, p, p, 2.0

    de_results.remember("ds", [Row("G0", 0.001), Row("G1", 0.2)],
                        comparison={"mode": "selection", "label": "selection", "n_a": 50, "n_b": 150, "cells": h})
    text = "\n".join(sf.grounding_lines(a, sel, dataset_id="ds")[0])
    assert "A stored DE result compared exactly these cells with the rest: selection (50 cells) vs rest" in text
    assert "1 of 2 genes at adj. p < 0.05" in text


def test_grounding_lines_skip_markers_on_a_large_dataset(monkeypatch):
    from scview.core import large_mode

    a = _adata()

    class Adaptor:
        adata = a
        is_large = True

    monkeypatch.setattr(large_mode, "is_large", lambda obj: True)
    lines, _ = sf.grounding_lines(Adaptor(), {"n_cells": 50, "cells": _cells(range(50), a.n_obs)},
                                  dataset_id="big")
    assert any("not ranked: large dataset" in ln for ln in lines)


def test_grounding_lines_say_when_the_selection_does_not_fit():
    a = _adata()
    lines, detail = sf.grounding_lines(a, {"n_cells": 5, "cells": _cells(range(5), 999)}, dataset_id="x")
    assert detail is None and "could not be used" in lines[1]


def test_kang_sized_facts_are_fast():
    """Kang-sized (14k cells x 14k genes, sparse): well under a second warm."""
    import time

    rng = np.random.default_rng(3)
    n, g = 14_000, 14_000
    X = sp.random(n, g, density=0.06, format="csr", random_state=rng, dtype=np.float32)
    X.data = np.log1p(X.data * 20)
    a = ad.AnnData(X)
    a.var_names = [f"G{i}" for i in range(g)]
    a.obs["cluster"] = pd.Categorical(rng.integers(0, 13, n).astype(str))
    cells = _cells(np.flatnonzero(rng.random(n) < 0.1), n)
    t = time.perf_counter()
    sf.facts(a, cells, dataset_id="kang")
    first = time.perf_counter() - t
    t = time.perf_counter()
    sf.grounding_lines(a, {"n_cells": 1, "cells": cells}, dataset_id="kang")
    assert time.perf_counter() - t < 0.5  # cached
    assert first < 5.0  # loose bound for a 2-core CI container; ~0.5 s measured


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

SEL_VC = {"panel": "Unified View", "selection": {"n_cells": 50, "source": "lasso",
                                                 "cells": {"n_cells": 200, "indices": [1, 2, 3]}}}


def test_selection_questions_route_to_the_dataset_never_literature_alone():
    st = RouteState(rag_enabled=True, has_dataset=True, has_selection=True)
    for q in ("What are these cells?", "How do the selected cells differ from the rest?",
              "what's in my lasso", "Is this selection enriched for interferon genes?"):
        i = pre_route(q, SEL_VC, st)
        assert i is not None and "data" in i.sources, q
        assert i.data_facets, q  # never "all facets"
    assert pre_route("What are these cells?", {"panel": "x"}, RouteState(has_dataset=True)) is None
    assert pre_route("What is a T cell?", SEL_VC, st) is None


def test_ifn_question_routes_to_the_stored_results_deterministically():
    no_rag = RouteState(rag_enabled=False, has_dataset=True, has_results=True)
    i = pre_route(IFN_Q, None, no_rag)
    assert i.sources == ["data"] and i.data_facets == ["groups", "markers", "enrichment"]
    rag = RouteState(rag_enabled=True, has_dataset=True, has_results=True)
    assert pre_route(IFN_Q, None, rag).sources == ["data", "literature"]
    # no stored results: left to the classifier
    assert pre_route(IFN_Q, None, RouteState(has_dataset=True)) is None


def test_rules_drop_corpora_without_rag_and_keep_the_dataset_for_results():
    lit_only = Intent(sources=["literature"], reason="llm", data_facets=[])
    st = RouteState(rag_enabled=False, has_dataset=True, has_results=False)
    out = apply_rules(lit_only, "What is the role of interferon?", None, st)
    assert out.sources == ["data"]
    assert apply_rules(Intent(sources=["tutorials"], reason="x"), "why log?", None,
                       RouteState(rag_enabled=False)).sources == ["app"]
    # RAG on, stored results: a cell-type question keeps the literature and adds the dataset
    st = RouteState(rag_enabled=True, has_dataset=True, has_results=True)
    out = apply_rules(lit_only, "Which cell types are interferon-stimulated?", None, st)
    assert out.sources == ["literature", "data"]
    # nothing to change: the same object back
    same = Intent(sources=["data"], reason="llm", data_facets=["qc"])
    assert apply_rules(same, "which clusters look low quality?", None, st) is same


class _FakeLLM:
    calls = 0
    answer = '{"sources": ["literature"], "data_facets": [], "reason": "biology"}'

    def __init__(self, *a, **k):
        outer = self

        class _Completions:
            async def create(self, **kw):
                _FakeLLM.calls += 1
                msg = type("M", (), {"content": outer.answer})
                return type("R", (), {"choices": [type("C", (), {"message": msg})]})

        self.chat = type("Chat", (), {"completions": _Completions()})()


async def test_classify_intent_rules_and_cache(monkeypatch):
    import openai

    monkeypatch.setattr(openai, "AsyncOpenAI", _FakeLLM)
    _FakeLLM.calls = 0
    st = RouteState(rag_enabled=False, has_dataset=True, has_results=True)
    # the IFN question never reaches the classifier when results are stored
    for _ in range(3):
        i = await classify_intent(IFN_Q, {"panel": "Unified View"}, "key", "m", state=st)
        assert i.sources == ["data"]
    assert _FakeLLM.calls == 0
    # a literature-only pick without RAG becomes the dataset; asked twice, one call
    q = "Is the interferon signature linked to lupus?"
    a1 = await classify_intent(q, {"panel": "Unified View"}, "key", "m", state=st)
    a2 = await classify_intent(q, {"panel": "Unified View"}, "key", "m", state=st)
    assert a1.sources == a2.sources == ["data"] and _FakeLLM.calls == 1
    # without state the classifier's answer stands (back-compat)
    raw = await classify_intent(q, {"panel": "Unified View"}, "key", "m")
    assert raw.sources == ["literature"]


# ---------------------------------------------------------------------------
# Claims
# ---------------------------------------------------------------------------


def test_claims_check_a_selection_number():
    from scview.core.assistant import ChatSource

    a = _adata()
    lines, detail = sf.grounding_lines(a, {"n_cells": 50, "cells": _cells(range(50), a.n_obs)},
                                       color_by="cluster", dataset_id="ds")
    src = ChatSource(kind="result", ref=sf.TAG, detail=detail, evidence="\n".join(lines[1:]))
    refs = [src.ref, "dataset:summary"]
    assert resolve_tag("sel", refs) == [sf.TAG]
    assert resolve_tag("selection:composition", refs) == [sf.TAG]
    vocab = Vocab(genes=frozenset(a.var_names), groups=["A", "B", "C", "CTRL", "STIM"])
    good = check_answer("The selection holds 50 cells, 25.0% of the dataset [result:selection].",
                        [src], "\n".join(lines), vocab)
    assert good[0].verdict == "supported", good[0].reason
    bad = check_answer("80.0% of the selection is A, vs 35.0% of the dataset [sel].",
                       [src], "\n".join(lines), vocab)
    assert bad[0].verdict == "refuted", bad[0].reason
    assert bad[0].verdict != "supported"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


def _register(dm, did: str = "sel1") -> str:
    d = Path(dm.data_dir) / "uploads" / did
    d.mkdir(parents=True, exist_ok=True)
    (d / "metadata.json").write_text(json.dumps({"id": did, "name": did, "status": "ready"}))
    _adata().write_h5ad(d / "data.h5ad")
    return did


async def test_chat_grounds_a_selection_question_in_its_facts(client, dataset_manager):
    did = _register(dataset_manager)
    vc = {"panel": "Unified View", "color_by": "cluster",
          "selection": {"n_cells": 50, "source": "lasso", "cells": _bitmask(range(50), 200)}}
    r = await client.post(f"{API}/datasets/{did}/assistant/chat",
                          json={"query": "What are these cells?", "view_context": vc})
    assert r.status_code == 200, r.text
    body = r.json()
    assert "result:selection" in [s["ref"] for s in body["sources"]]
    assert "data" in body["route"] and "literature" not in body["route"]
    assert "80.0% of the selection" in body["answer"]
    # an unrelated question, selection not fresh: no selection block (small call)
    r = await client.post(f"{API}/datasets/{did}/assistant/chat",
                          json={"query": "What analysis steps were run?", "view_context": vc})
    assert "result:selection" not in [s["ref"] for s in r.json()["sources"]]
    # ... unless the user has just made the selection
    fresh = {**vc, "selection": {**vc["selection"], "fresh": True}}
    r = await client.post(f"{API}/datasets/{did}/assistant/chat",
                          json={"query": "What analysis steps were run?", "view_context": fresh})
    assert "result:selection" in [s["ref"] for s in r.json()["sources"]]


async def test_de_on_a_selection_is_cited_for_the_same_cells(client, dataset_manager):
    did = _register(dataset_manager)
    r = await client.post(f"{API}/datasets/{did}/de", json={"indices": list(range(50))})
    assert r.status_code == 200, r.text
    vc = {"selection": {"n_cells": 50, "source": "lasso", "cells": _cells(range(50), 200)}}
    r = await client.post(f"{API}/datasets/{did}/assistant/chat",
                          json={"query": "How do these cells differ from the rest?", "view_context": vc})
    assert "A stored DE result compared exactly these cells with the rest" in r.json()["answer"]


async def test_run_de_for_the_selection_opens_the_de_tab(client, dataset_manager):
    did = _register(dataset_manager)
    vc = {"selection": {"n_cells": 50, "source": "lasso", "cells": _cells(range(50), 200)}}
    r = await client.post(f"{API}/datasets/{did}/assistant/chat",
                          json={"query": "Run DE for this selection vs the rest", "view_context": vc})
    body = r.json()
    assert body["route"] == ["action"]
    act = body["actions"][0]
    assert act["type"] == "open_de_selection" and act["requires_confirm"] is False
    assert "50 cells" in act["label"] and "Click Run" in act["label"]
