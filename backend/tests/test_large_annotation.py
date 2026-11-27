"""The AI annotator's own rules run in large-dataset mode as they do in memory.

In a large run the step works on a copy of the dataset without its matrices. Cluster QC
(the QC notes sent with the markers, and the damaged clusters scView labels itself) and
the label qualifiers read a matrix, failed on that copy, and the failure was only logged:
the run's result had no notes and no ``<target>_qualified`` column, and said nothing.
Here the same dataset is annotated both ways with a stand-in model.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from scview.config import Settings
from scview.core import chunked, large_run, provenance
from scview.core import marker_randomness as MR
from scview.core import pipeline as P
from scview.core.pipeline import PipelineParams, run_pipeline

ROWS = 700
STEPS = ["reset_to_counts", "qc_metrics", "normalization", "log_transform", "marker_genes",
         "cell_type_annotation"]
N_TYPE_GENES = 30
REPLY = "tissue: test tissue\nt0: T cells\nt1: T cells\nt2: B cells\nt3: Monocytes\nt4: Hepatocytes\n"


def _counts(seed: int = 0, n: int = 2400) -> ad.AnnData:
    """Five groups: t0-t2 each with its own 30 genes (one database type each), t3 with 27
    genes spread over as many types (a random-like list), t4 damaged cells (a fifth of the
    counts, ten times the mito)."""
    rng = np.random.default_rng(seed)
    names = [f"MT-{i}" for i in range(6)] + ["MALAT1"] + [f"RPS{i}" for i in range(5)] + [f"G{i}" for i in range(900)]
    g = len(names)
    first = names.index("G0")
    group = rng.integers(0, 5, n)
    size = rng.lognormal(0.0, 0.3, n)
    mu = size[:, None] * rng.gamma(0.6, 1.0, g)[None, :] * 2.0
    for k in range(3):
        mu[group == k, first + N_TYPE_GENES * k: first + N_TYPE_GENES * (k + 1)] *= 8.0
    spread = first + 3 * N_TYPE_GENES + np.arange(27) * N_TYPE_GENES  # one gene of each other type
    mu[np.ix_(group == 3, spread)] *= 8.0
    mu[group == 4] *= 0.2
    mu[np.ix_(group == 4, np.arange(6))] *= 60.0
    mu[np.ix_(group == 4, first + 600 + np.arange(30))] *= 8.0
    X = rng.negative_binomial(2, 2.0 / (2.0 + mu)).astype(np.float32)
    obs = pd.DataFrame({"group": pd.Categorical([f"t{k}" for k in group]),
                        "donor": pd.Categorical([f"d{i % 4}" for i in range(n)])},
                       index=[f"cell{i}-1" for i in range(n)])
    return ad.AnnData(sp.csr_matrix(X), obs=obs, var=pd.DataFrame(index=names))


class _FakeDB:
    """Marker databases stand-in: 30 disjoint types of 30 genes over G0..G899."""

    def __init__(self):
        self.types = {f"type{t:02d}": {"genes": [f"G{t * N_TYPE_GENES + i}" for i in range(N_TYPE_GENES)], "cl": None}
                      for t in range(30)}
        self.names = sorted(self.types)

    def species(self, species, parts=None):
        return self

    def view(self, job, species):
        return ["fake"]


@pytest.fixture()
def model(monkeypatch):
    """A stand-in language model (never a real call); returns the prompts it was sent."""
    sent: list[str] = []

    class _Client:
        def __init__(self, **_):
            self.chat = types.SimpleNamespace(completions=self)

        def create(self, messages, **_):
            sent.append(messages[0]["content"])
            msg = types.SimpleNamespace(content=REPLY)
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg, finish_reason="stop")])

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=_Client))
    monkeypatch.setattr("scview.config.get_settings", lambda: Settings(DEEPINFRA_API_KEY="k", RAG_CHAT_MODEL="m"))
    monkeypatch.setattr(P, "_marker_db_or_none", lambda: _FakeDB())
    monkeypatch.setattr(P, "_record_llm_cl_ids", lambda adata, target, ids: {})
    monkeypatch.setattr(chunked, "rows_per_block", lambda *a, **k: ROWS)
    return sent


def _params(**kw) -> PipelineParams:
    return PipelineParams(min_cells=0, marker_columns=["group"], annotation_method="llm",
                          annotation_groupby="group", annotation_target="ai_label", **kw)


def _step(adata) -> dict:
    return next(h for h in reversed(provenance.read_provenance(adata)["history"])
                if h["step"] == "cell_type_annotation")["params"]


def test_decisions_notes_and_qualified_labels_are_the_same_in_large_mode(tmp_path, model):
    src = tmp_path / "in" / "data.h5ad"
    src.parent.mkdir()
    _counts().write_h5ad(src, compression="gzip")
    mem, mres = run_pipeline(ad.read_h5ad(src), STEPS, _params())
    assert mres.errors == {}
    prompt_mem = model.pop()
    out = tmp_path / "derived" / "data.h5ad"
    res = large_run.run(src, STEPS, _params(chunked="on"), str(out))
    assert res.status == "complete", (res.errors, res.notes)
    prompt_large = model.pop()
    big = ad.read_h5ad(out)

    # In memory the rules do something on this dataset (else the comparison shows nothing):
    rec = _step(mem)
    assert rec["llm_decided_low_quality"] == ["t4"]            # scView's own label, never sent
    assert rec[P.RANDOM_NOTED] == ["t3"]                         # random-like markers: a note to the model
    assert rec["llm_qc_notes"] >= 1 and "[QC: " in prompt_mem
    assert "t4:" not in prompt_mem
    assert rec["qualifiers"] == 2 and "ai_label_qualified" in mem.obs  # two clusters named "T cells"

    assert prompt_large == prompt_mem  # the same markers and the same QC notes went to the model
    big_rec = _step(big)
    for key in ("llm_decided_low_quality", P.RANDOM_NOTED, "llm_qc_notes", "qualifiers",
                "qualifiers_considered", "qualifier_column"):
        assert big_rec.get(key) == rec.get(key), key
    assert list(big.obs["ai_label"].astype(str)) == list(mem.obs["ai_label"].astype(str))
    assert list(big.obs["ai_label_qualified"].astype(str)) == list(mem.obs["ai_label_qualified"].astype(str))
    assert json.loads(big.uns["ai_label_qualifiers"]) == json.loads(mem.uns["ai_label_qualifiers"])
    assert "annotation_checks_skipped" not in big_rec and "cell_type_annotation" not in res.cautions


@pytest.mark.parametrize("large", [False, True])
def test_a_check_that_cannot_run_is_named_in_the_record_and_the_result(tmp_path, model, monkeypatch, large):
    from scview.core import cluster_qc, marker_specificity

    def boom(*a, **k):
        raise RuntimeError("no QC table")

    monkeypatch.setattr(cluster_qc, "cluster_qc", boom)
    monkeypatch.setattr(marker_specificity, "_group_sums", lambda *a, **k: (_ for _ in ()).throw(OSError("read failed")))
    src = tmp_path / "in" / "data.h5ad"
    src.parent.mkdir()
    _counts().write_h5ad(src, compression="gzip")
    if large:
        out = tmp_path / "derived" / "data.h5ad"
        res = large_run.run(src, STEPS, _params(chunked="on"), str(out))
        done = ad.read_h5ad(out)
    else:
        done, res = run_pipeline(ad.read_h5ad(src), STEPS, _params())
    assert res.errors == {}  # the annotation itself still ran
    rec = _step(done)
    assert rec[P.CHECKS_SKIPPED] == {P.CHECK_QC_NOTES: "no QC table", P.CHECK_DAMAGED: "no QC table",
                                     P.CHECK_QUALIFIERS: "read failed"}
    assert "llm_decided_low_quality" not in rec and "t4:" in model[-1]  # the damaged cluster went to the model
    text = res.cautions["cell_type_annotation"]
    assert text.startswith("Not run in this annotation: QC notes for the model (no QC table); "
                           "damaged-cluster check (no QC table); label qualifiers (read failed).")
    assert "look at cluster QC" in text
    assert done.obs["ai_label"].astype(str).nunique() == 4 and "ai_label_qualified" not in done.obs


def test_a_run_with_every_check_says_nothing(tmp_path, model):
    src = tmp_path / "in" / "data.h5ad"
    src.parent.mkdir()
    _counts().write_h5ad(src, compression="gzip")
    done, res = run_pipeline(ad.read_h5ad(src), STEPS, _params())
    assert P.CHECKS_SKIPPED not in _step(done) and "cell_type_annotation" not in res.cautions
