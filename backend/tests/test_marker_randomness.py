"""Marker lists that name no database type better than random genes (core/marker_randomness.py),
and the QC note the annotator sends for them (pipeline._annotate_llm)."""

from __future__ import annotations

import sys
import types

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from scview.config import Settings
from scview.core import marker_randomness as MR
from scview.core import pipeline as P

N_TYPES, PER_TYPE, N_GENES = 60, 20, 3000


class _FakeDB:
    """MarkerDB stand-in: N_TYPES disjoint types of PER_TYPE genes in a G0..G2999 universe."""

    def __init__(self):
        self.types = {f"type{t}": {"genes": [f"G{t * PER_TYPE + i}" for i in range(PER_TYPE)], "cl": None}
                      for t in range(N_TYPES)}
        self.names = sorted(self.types)

    def species(self, species, parts=None):
        return self

    def view(self, job, species):
        return ["fake"]


def _adata(detected: float = 0.5) -> ad.AnnData:
    a = ad.AnnData(X=np.zeros((10, N_GENES), dtype=np.float32))
    a.var_names = [f"G{i}" for i in range(N_GENES)]
    a.var["n_cells_by_counts"] = int(detected * a.n_obs)
    return a


def _random_named(rng, n):
    named = N_TYPES * PER_TYPE  # genes G0..G1199 are in some type
    return [f"G{i}" for i in rng.choice(named, n, replace=False)]


def test_a_type_list_passes_a_random_list_is_flagged_and_unnamed_genes_do_not_count():
    rng = np.random.default_rng(1)
    mk = {
        "type": [f"G{i}" for i in range(15)] + _random_named(rng, 10),  # 15 of type0's genes
        "random": _random_named(rng, 25),
        # 25 genes in no database type (lncRNA/pseudogene-like): not judged, not flagged
        "unnamed": [f"G{i}" for i in range(2000, 2025)],
        "few": _random_named(rng, MR.MIN_NAMED - 1) + [f"G{i}" for i in range(2100, 2116)],
    }
    out = MR.check(_adata(), mk, _FakeDB())
    assert set(out.flagged) == {"random"}
    assert out.flagged["random"]["named"] == 25
    assert out.too_few_named == ["few", "unnamed"]
    assert out.checked == 2
    assert out.skipped is None


def test_null_pool_is_the_detected_named_genes():
    # Nothing detected in >= 10% of cells: an empty pool, so no list is judged (never flagged).
    out = MR.check(_adata(detected=0.0), {"random": _random_named(np.random.default_rng(2), 25)}, _FakeDB())
    assert out.flagged == {} and out.checked == 0


def test_no_databases_on_disk_skips_without_fetching(monkeypatch, tmp_path):
    from scview.core import marker_db as mdb

    monkeypatch.setattr(mdb, "marker_db_dir", lambda: tmp_path)
    monkeypatch.setattr(mdb, "ensure", lambda *a, **k: (_ for _ in ()).throw(AssertionError("fetched")))
    out = MR.check(_adata(), {"a": ["G1"]})
    assert out.skipped == "marker databases not downloaded" and not out.flagged


def test_annotator_notes_a_random_list_and_records_it(monkeypatch):
    rng = np.random.default_rng(0)
    a = ad.AnnData(X=rng.random((20, 4)).astype(np.float32))
    a.var_names = ["CD3D", "CD3E", "MS4A1", "CD79A"]
    a.obs["cluster"] = pd.Categorical(["0"] * 10 + ["1"] * 10)
    monkeypatch.setattr(P, "_top_markers_per_group", lambda adata, g: {"0": ["CD3D", "CD3E"], "1": ["MS4A1"]})
    monkeypatch.setattr(P, "_annotation_qc_notes", lambda adata, g: {"1": "flagged high mito"})
    monkeypatch.setattr(MR, "check", lambda adata, mk: MR.RandomCheck(
        flagged={"0": {}, "1": {}}, checked=2))
    sent: list[str] = []

    class _Client:
        def __init__(self, **_):
            self.chat = types.SimpleNamespace(completions=self)

        def create(self, messages, **_):
            sent.append(messages[0]["content"])
            msg = types.SimpleNamespace(content="tissue: blood\n0: Unknown | CL:none\n1: Low-quality cells | CL:none")
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg, finish_reason="stop")])

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=_Client))
    monkeypatch.setattr("scview.config.get_settings", lambda: Settings(DEEPINFRA_API_KEY="k", RAG_CHAT_MODEL="m"))
    monkeypatch.setattr(P, "_record_llm_cl_ids", lambda adata, target, ids: {})
    fit = P._annotate_llm(a, P.PipelineParams(), "cluster", "cell_type")

    listing = sent[0].split("Clusters and marker genes:")[1].strip().splitlines()
    assert listing[0] == f"0: CD3D, CD3E [QC: {MR.NOTE}]"
    assert listing[1] == f"1: MS4A1 [QC: flagged high mito; {MR.NOTE}]"
    assert fit["llm_random_markers"] == ["0", "1"]
    assert fit["llm_qc_notes"] == 2


def test_a_failing_check_leaves_the_annotation_alone(monkeypatch):
    monkeypatch.setattr(MR, "check", lambda adata, mk: (_ for _ in ()).throw(RuntimeError("boom")))
    qc = {"1": "x"}
    assert P._annotation_random_notes(_adata(), {"1": ["G1"]}, qc) == []
    assert qc == {"1": "x"}


def test_the_prompt_names_the_note_and_its_own_answer_before_the_mixture_rule():
    """A noted cluster gets 'Unidentified', so 'Low-quality cells' stays for damaged cells
    (annotation benchmark arm rm, 2026-11-10); the mixture rule stays the last naming rule."""
    p = P._annotation_prompt("0: CD3D")
    assert MR.NOTE in p
    unid, mixture = p.index("'Unidentified | CL:none'"), p.index("name the mixture")
    assert p.index("'Low-quality cells | CL:none'") < unid < mixture


@pytest.mark.parametrize("label", ["Unidentified", "unidentified cells", " Unidentified cell "])
def test_unidentified_spellings_become_one_label_without_an_id(label):
    assert P._UNIDENTIFIED.match(label)


def test_an_unidentified_answer_keeps_no_cl_id(monkeypatch):
    rng = np.random.default_rng(0)
    a = ad.AnnData(X=rng.random((20, 2)).astype(np.float32))
    a.var_names = ["CD3D", "MS4A1"]
    a.obs["cluster"] = pd.Categorical(["0"] * 10 + ["1"] * 10)
    monkeypatch.setattr(P, "_top_markers_per_group", lambda adata, g: {"0": ["CD3D"], "1": ["MS4A1"]})
    monkeypatch.setattr(P, "_annotation_qc_notes", lambda adata, g: {})
    monkeypatch.setattr(MR, "check", lambda adata, mk: MR.RandomCheck(flagged={"1": {}}, checked=2))

    class _Client:
        def __init__(self, **_):
            self.chat = types.SimpleNamespace(completions=self)

        def create(self, messages, **_):
            msg = types.SimpleNamespace(content="tissue: blood\n0: T cells | CL:0000084\n1: unidentified cells | CL:0000000")
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg, finish_reason="stop")])

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=_Client))
    monkeypatch.setattr("scview.config.get_settings", lambda: Settings(DEEPINFRA_API_KEY="k", RAG_CHAT_MODEL="m"))
    monkeypatch.setattr(P, "_record_llm_cl_ids", lambda adata, target, ids: {"ids": ids})
    fit = P._annotate_llm(a, P.PipelineParams(), "cluster", "cell_type")
    assert a.uns["cell_type_llm_mapping"]["1"] == "Unidentified"
    assert fit["ids"].get("Unidentified") is None
    from scview.core import cell_ontology, label_qualifiers

    assert cell_ontology._no_type("Unidentified")
    assert "unidentified" in label_qualifiers._NO_TYPE
