"""Doublet-like clusters (core/cluster_qc.py ``doublet_like``, C056).

Synthetic: five cell types with their own marker genes, a cluster of planted doublets
(each the summed counts of a cell of type A and one of type B) and a cluster that sits
between A and C with one cell's RNA (an intermediate state, not doublets). The doublet
cluster is flagged "looks like A + B"; the intermediate one is not (its RNA is a single
cell's), nor any real type.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import anndata as ad

from scview.core import cluster_qc as cq
from scview.core.pipeline import PipelineParams, run_pipeline

TYPES = ["A", "B", "C", "D", "E"]


def _data(seed: int = 0, n_per: int = 220, n_mix: int = 40) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    g = 400
    base = rng.gamma(0.8, 0.6, g)
    profiles = {}
    for k, t in enumerate(TYPES):
        p = base.copy()
        p[40 * k: 40 * k + 40] *= rng.uniform(6, 14, 40)  # each type's own markers
        profiles[t] = p / p.sum()

    def cells(p, n, lib=2500.0):
        return rng.poisson(p * rng.lognormal(np.log(lib), 0.2, n)[:, None]).astype(np.float32)

    blocks = {t: cells(profiles[t], n_per) for t in TYPES}
    a_i, b_i = rng.integers(0, n_per, n_mix), rng.integers(0, n_per, n_mix)
    blocks["AB"] = blocks["A"][a_i] + blocks["B"][b_i]  # doublets: two cells' RNA
    blocks["AC"] = cells((profiles["A"] + profiles["C"]) / 2, n_mix)  # in between, one cell's RNA
    labels = np.concatenate([[t] * len(v) for t, v in blocks.items()])
    a = ad.AnnData(np.vstack(list(blocks.values())))
    a.var_names = [f"G{i}" for i in range(g)]
    a.obs_names = [f"c{i}" for i in range(a.n_obs)]
    a.obs["cluster"] = pd.Categorical(labels, categories=list(blocks))
    return a


@pytest.fixture(scope="module")
def marked():
    a, res = run_pipeline(_data(), ["qc_metrics", "normalization", "log_transform", "marker_genes"],
                          PipelineParams(marker_columns=["cluster"]))
    assert res.errors == {}
    return a


def test_a_cluster_of_doublets_is_named_for_its_two_parents(marked):
    report = cq.cluster_qc(marked, "cluster")
    assert report.doublet_like_checked
    by = {c.group: c for c in report.clusters}
    ab = by["AB"]
    assert "doublet_like" in ab.flags and ab.doublet_like_of == ["A", "B"]
    reason = ab.reasons[ab.flags.index("doublet_like")]
    assert reason.startswith("Looks like doublets of A + B: ")
    assert f"median {ab.median_counts:,.0f} counts" in reason
    assert "doublet_like" in report.flag_help
    # The intermediate state has one cell's RNA, and the real types are their own
    assert "doublet_like" not in by["AC"].flags
    assert not any("doublet_like" in by[t].flags for t in TYPES)


def test_the_copilot_hears_which_two_clusters(marked):
    lines, cites = cq.grounding_block(cq.cluster_qc(marked, "cluster"))
    text = "\n".join(lines)
    assert "**AB**" in text and "Looks like doublets of A + B" in text
    assert any(tag == cq.qc_tag("cluster", "AB") and "doublet_like" in d for tag, d in cites)


def test_the_qc_report_names_it_a_concern_and_a_next_step(marked):
    from types import SimpleNamespace

    from scview.core import qc_report

    facts = qc_report.build_facts(SimpleNamespace(adata=marked))
    assert any("looks like doublets of two other clusters" in c.detail for c in facts.concerns)
    assert any("not doublets of A and B" in s.detail for s in facts.next_steps)


def test_without_markers_nothing_is_judged():
    a, _ = run_pipeline(_data(seed=1), ["qc_metrics", "normalization", "log_transform"], PipelineParams())
    report = cq.cluster_qc(a, "cluster")
    assert not report.doublet_like_checked
    assert not any("doublet_like" in c.flags for c in report.clusters)


def _row(group: str, n: int, counts: float, genes: float) -> cq.ClusterQC:
    return cq.ClusterQC(group=group, n_cells=n, fraction=0.1, median_counts=counts, median_genes=genes)


def test_the_floors_each_say_no():
    m = {"P": set("abcdefghij"), "Q": set("klmnopqrst"), "R": set("uvwxyz0123")}
    m["X"] = set("abcdeklmno")  # half from P, half from Q
    rows = {"P": _row("P", 300, 2000, 900), "Q": _row("Q", 300, 2500, 1000),
            "R": _row("R", 300, 2000, 900), "X": _row("X", 40, 3500, 1300)}
    assert cq.doublet_like_pair(m, rows, "X") == ("P", "Q", 5, 5, 10)
    # under the larger parent's RNA plus half the smaller one's (2,500 + 1,000)
    assert cq.doublet_like_pair(m, {**rows, "X": _row("X", 40, 3400, 1300)}, "X") is None
    # parents of unlike RNA (hgmm_10k_v3: the doublet cluster passes, a subtype of the big one not)
    unlike = {**rows, "P": _row("P", 3000, 40430, 6000), "Q": _row("Q", 1100, 8530, 2500)}
    assert cq.doublet_like_pair(m, {**unlike, "X": _row("X", 357, 46157, 7000)}, "X")[:2] == ("P", "Q")
    assert cq.doublet_like_pair(m, {**unlike, "X": _row("X", 169, 43892, 7000)}, "X") is None
    # a parent flagged low quality / low genes is ambient RNA or damaged cells, not a cell type
    low = {**rows, "Q": rows["Q"].model_copy(update={"flags": ["low_genes"]})}
    assert cq.doublet_like_pair(m, low, "X") is None
    # fewer genes than a parent
    assert cq.doublet_like_pair(m, {**rows, "X": _row("X", 40, 3500, 950)}, "X") is None
    # larger than a parent: a real population
    assert cq.doublet_like_pair(m, {**rows, "X": _row("X", 400, 3500, 1300)}, "X") is None
    # mostly one parent's markers
    one = {**m, "X": set("abcdefghik")}
    assert cq.doublet_like_pair(one, rows, "X") is None
    # too few markers to judge
    few = {**m, "X": set("abckl")}
    assert cq.doublet_like_pair(few, rows, "X") is None
