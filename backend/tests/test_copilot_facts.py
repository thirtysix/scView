"""Co-pilot fact lines that made the right number the obvious one (batch 24): the
median cell vs the typical cluster, every cluster a label spans, why a Mixscape
target has no KO cells, and the condition column in the dataset summary."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd

from scview.core import label_clusters, mixscape, provenance
from scview.core.anndata_adaptor import AnnDataAdaptor
from scview.core.assistant import build_grounding_context
from scview.core.claims import Vocab, check_answer
from scview.core.cluster_qc import cluster_qc, grounding_block


def _kang_like(n_per: int = 60) -> ad.AnnData:
    """Mono split over Leiden 1 (CTRL) and 2 (STIM), T in 0, B in 3, with a stim column."""
    rng = np.random.default_rng(0)
    leiden = ["0"] * (2 * n_per) + ["1"] * n_per + ["2"] * n_per + ["3"] * n_per
    label = ["T"] * (2 * n_per) + ["Mono"] * (2 * n_per) + ["B"] * n_per
    stim = (["CTRL", "STIM"] * n_per) + ["CTRL"] * n_per + ["STIM"] * n_per + (["CTRL", "STIM"] * n_per)[:n_per]
    n = len(leiden)
    X = rng.poisson(2.0, (n, 30)).astype("float32")
    X[:, :10] += rng.poisson(3.0, (n, 10)).astype("float32") * (np.arange(n)[:, None] % 3)
    a = ad.AnnData(X)
    a.var_names = [f"G{i}" for i in range(30)]
    a.obs_names = [f"C{i}" for i in range(n)]
    a.obs["scview_leiden_r0.5"] = pd.Categorical(leiden)
    a.obs["cluster"] = pd.Categorical(label)
    a.obs["stim"] = pd.Categorical(stim)
    a.uns["scview_active_clustering"] = "scview_leiden_r0.5"
    return a


def test_qc_fact_gives_the_median_cell_apart_from_the_typical_cluster():
    a = _kang_like()
    report = cluster_qc(a, "scview_leiden_r0.5")
    lines, _ = grounding_block(report)
    text = "\n".join(lines)
    genes = np.median((a.X > 0).sum(axis=1))
    assert f"- Median cell (all {a.n_obs:,} cells, no QC filtering recorded): {genes:,.0f} genes/cell" in text
    assert "Typical cluster (the median of the per-cluster medians, not the median cell)" in text
    provenance.record_step(a, step="filtering", tool="scview.qc_filter", params={})
    lines, _ = grounding_block(cluster_qc(a, "scview_leiden_r0.5"))
    assert "cells, after QC filtering):" in "\n".join(lines)


def test_a_label_lists_every_cluster_it_spans(tmp_path: Path):
    p = tmp_path / "k.h5ad"
    _kang_like().write_h5ad(p)
    adaptor = AnnDataAdaptor(str(p))
    lines, cites = label_clusters.grounding_lines(adaptor)
    text = "\n".join(lines)
    assert "Labels in 'cluster' across the clusters of 'scview_leiden_r0.5'" in text
    assert "1 labels span several clusters" in text
    assert "- **Mono** (120 cells): clusters 1 (60), 2 (60)" in text
    assert "- **B** (60 cells): cluster 3 (60)" in text
    assert [c[0] for c in cites] == ["result:labels:cluster"]  # 'stim' is a condition, not labels
    context, sources = build_grounding_context(adaptor)
    assert "- **Mono** (120 cells): clusters 1 (60), 2 (60)" in context
    # the ledger supports "Mono is clusters 1 and 2" and refutes a wrong size
    vocab = Vocab(genes=frozenset(), groups=["Mono", "T", "B", "CTRL", "STIM"],
                  numeric_groups=frozenset({"0", "1", "2", "3"}))
    ok = check_answer("Mono spans clusters 1 (60 cells) and 2 (60 cells) [result:labels:cluster].",
                      sources, context, vocab)
    assert ok[0].verdict == "supported", ok[0].reason


def test_summary_names_the_condition_column(tmp_path: Path):
    p = tmp_path / "k.h5ad"
    _kang_like().write_h5ad(p)
    context, _ = build_grounding_context(AnnDataAdaptor(str(p)), facets={"markers"})
    assert "- Condition column 'stim': " in context
    assert "mixes these conditions unless compared within one condition" in context


def test_all_np_targets_say_why_not_zero_de_genes(monkeypatch):
    targets = [
        {"target": "IFNGR1", "n_cells": 100, "n_ko": 70, "ko_fraction": 0.7, "n_de_genes": 479},
        {"target": "MYC", "n_cells": 50, "n_ko": 0, "ko_fraction": 0.0, "n_de_genes": 17},
        {"target": "ATF2", "n_cells": 995, "n_ko": 0, "ko_fraction": 0.0, "n_de_genes": 0},
    ]
    r = mixscape.MixscapeResult(
        params={"n_neighbors": 20, "logfc_threshold": 0.25, "pval_cutoff": 0.05, "min_de_genes": 5},
        labels="gene", control="NT", split_by=None, n_cells=1200, n_control=300, n_targets=3,
        n_genes=2000, gene_source="hvg", expression_source="X", targets=targets)
    monkeypatch.setattr(mixscape, "last_result", lambda adaptor: r)
    lines, _ = mixscape.grounding_lines(SimpleNamespace(adata=ad.AnnData(np.zeros((2, 2)))))
    text = "\n".join(lines)
    assert ("MYC (17 DE genes vs the controls, but fewer than 5 cells called KO: not a KO population, "
            "not 'no DE genes')") in text
    assert "ATF2 (0 DE genes vs the controls, fewer than 5: no signature to test)" in text
