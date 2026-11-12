"""Marker lists that name no database type better than random genes (core/marker_randomness.py),
and the clusters the annotator decides without the AI (pipeline._annotation_decided)."""

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
    assert out.best["type"] == "type0"  # every list with a named gene, judged or not
    assert "few" in out.best and "unnamed" not in out.best


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


def _annotate(monkeypatch, markers, flags, check, reply):
    """Run _annotate_llm with stand-ins; (fit, prompts sent, adata)."""
    rng = np.random.default_rng(0)
    groups = list(markers)
    a = ad.AnnData(X=rng.random((10 * len(groups), 2)).astype(np.float32))
    a.var_names = ["CD3D", "MS4A1"]
    a.obs["cluster"] = pd.Categorical([g for g in groups for _ in range(10)])
    monkeypatch.setattr(P, "_top_markers_per_group", lambda adata, g: markers)
    monkeypatch.setattr(P, "_annotation_qc_notes",
                        lambda adata, g: {k: "flagged " + ", ".join(v) for k, v in flags.items()})
    from scview.core import cluster_qc

    monkeypatch.setattr(cluster_qc, "cluster_qc", lambda adata, g: types.SimpleNamespace(clusters=[
        types.SimpleNamespace(group=k, flags=flags.get(k, [])) for k in groups]))
    monkeypatch.setattr(MR, "check", check)
    sent: list[str] = []

    class _Client:
        def __init__(self, **_):
            self.chat = types.SimpleNamespace(completions=self)

        def create(self, messages, **_):
            sent.append(messages[0]["content"])
            msg = types.SimpleNamespace(content=reply)
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg, finish_reason="stop")])

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=_Client))
    monkeypatch.setattr("scview.config.get_settings", lambda: Settings(DEEPINFRA_API_KEY="k", RAG_CHAT_MODEL="m"))
    monkeypatch.setattr(P, "_record_llm_cl_ids", lambda adata, target, ids: {"ids": ids})
    fit = P._annotate_llm(a, P.PipelineParams(), "cluster", "cell_type")
    return fit, sent, a


def test_scview_decides_damaged_clusters_and_notes_random_ones_for_the_ai(monkeypatch):
    markers = {"0": ["CD3D"], "1": ["MS4A1"], "2": ["G1"], "3": ["G2"], "4": ["MYH6"]}
    flags = {"2": ["high_mito"], "3": ["low_quality"], "4": ["high_mito"]}
    check = lambda adata, mk: MR.RandomCheck(  # noqa: E731
        flagged={"1": {}, "3": {}}, checked=5, best={"0": "T cells", "1": "B cells", "4": "Cardiomyocytes"})
    fit, sent, a = _annotate(monkeypatch, markers, flags, check,
                             "tissue: heart\n0: T cells | CL:0000084\n4: Cardiomyocytes | CL:0000746")
    listing = sent[0].split("Clusters and marker genes:")[1].strip().splitlines()
    # 1 random: sent with a note (since v0.75.0, no longer "Unidentified"); 2 and 3 damaged (3 is also
    # random: damaged wins); 4 is high mito but best matches cardiomyocytes, so it goes to the AI
    # with its QC note.
    assert listing == ["0: CD3D", f"1: MS4A1 [QC: {MR.NOTE}]", "4: MYH6 [QC: flagged high_mito]"]
    assert MR.NOTE == "markers do not beat 9 of 10 random gene lists of the same size against the marker databases"
    m = a.uns["cell_type_llm_mapping"]
    assert m == {"0": "T cells", "1": "Unknown", "2": "Low-quality cells", "3": "Low-quality cells",
                 "4": "Cardiomyocytes"}  # the reply left 1 out
    assert fit["llm_decided_low_quality"] == ["2", "3"]
    assert fit[P.RANDOM_NOTED] == ["1"] and "llm_random_markers" not in fit
    assert fit["llm_high_mito_type_to_ai"] == ["4"]
    assert fit["ids"].get("Low-quality cells") is None
    assert "Unidentified" not in sent[0] and "Unidentified" not in m.values()


def test_every_cluster_decided_asks_no_model(monkeypatch):
    check = lambda adata, mk: MR.RandomCheck(flagged={"0": {}}, checked=2)  # noqa: E731
    fit, sent, a = _annotate(monkeypatch, {"0": ["G1"], "1": ["G2"]},
                             {"0": ["low_quality"], "1": ["high_mito"]}, check, "")
    assert sent == []
    assert a.uns["cell_type_llm_mapping"] == {"0": "Low-quality cells", "1": "Low-quality cells"}


def test_a_random_like_cluster_alone_still_asks_the_model(monkeypatch):
    check = lambda adata, mk: MR.RandomCheck(flagged={"0": {}}, checked=2)  # noqa: E731
    fit, sent, a = _annotate(monkeypatch, {"0": ["G1"], "1": ["G2"]}, {"1": ["high_mito"]}, check,
                             "tissue: kidney\n0: Injured proximal tubule cells | CL:none")
    assert len(sent) == 1 and f"0: G1 [QC: {MR.NOTE}]" in sent[0]
    assert a.uns["cell_type_llm_mapping"] == {"0": "Injured proximal tubule cells", "1": "Low-quality cells"}


def test_failing_checks_decide_nothing(monkeypatch):
    from scview.core import cluster_qc

    monkeypatch.setattr(MR, "check", lambda adata, mk: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(cluster_qc, "cluster_qc", lambda adata, g: (_ for _ in ()).throw(RuntimeError("boom")))
    assert P._annotation_decided(_adata(), "cluster", {"1": ["G1"]}) == ({}, {})


@pytest.mark.parametrize("name,natural", [
    ("Cardiomyocytes", True), ("Ventricular cardiomyocyte", True), ("Hepatocytes", True),
    ("Proximal tubule cells", True), ("Distal convoluted tubular cell", True), ("Loop of Henle cells", True),
    ("Intercalated cells", True), ("Skeletal muscle cell", True), ("Brown adipocyte", True),
    ("Parietal cells", True), ("Oxyphil cells", True), ("Tubular cell", True),
    ("Peritubular myoid cells", False), ("Pretubular aggregate cell", False), ("Tubulointerstitial cell", False),
    ("Hepatic stellate cells", False), ("Smooth muscle cells", False), ("Parietal epithelial cell", False),
    ("Monocytes", False), ("", False),
])
def test_high_mito_types(name, natural):
    assert bool(P._HIGH_MITO_TYPE.search(name)) is natural


def test_unidentified_is_a_no_type_label():
    from scview.core import cell_ontology, label_qualifiers

    assert cell_ontology._no_type("Unidentified")
    assert "unidentified" in label_qualifiers._NO_TYPE
