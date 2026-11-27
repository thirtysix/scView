"""Clusters that are almost all one sample (core/confounding.py single_sample_clusters): named in
the MCP run_pipeline reply after a clustering and as a note in the cluster QC report."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scview.core import confounding


def _obs(clusters: list[str], samples: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"leiden": pd.Categorical(clusters), "sample": pd.Categorical(samples)})


def test_a_cluster_of_one_sample_is_named():
    rng = np.random.default_rng(0)
    samples = [f"s{i}" for i in range(4)]
    mixed = [(c, samples[rng.integers(4)]) for c in ("0", "1", "2") for _ in range(100)]
    lone = [("3", "s2")] * 57 + [("3", "s1")] * 3
    obs = _obs(*zip(*(mixed + lone)))
    found = confounding.single_sample_clusters(obs, "leiden", "sample")
    assert [(r["cluster"], r["sample"], r["n_cells"], r["share"]) for r in found] == [("3", "s2", 60, 0.95)]
    note = confounding.single_sample_note(found, "leiden", "sample", 4)
    assert note.startswith("1 leiden cluster is almost all one sample (of 4 in all): 3 (60 cells, 95% from s2")


@pytest.mark.parametrize("case", ["two_samples", "small", "dominant", "mixed"])
def test_no_note_where_one_sample_clusters_are_expected(case):
    rng = np.random.default_rng(1)
    if case == "two_samples":
        obs = _obs(["0"] * 50 + ["1"] * 50, ["a"] * 50 + ["b"] * 50)
    elif case == "small":  # under MIN_CELLS
        obs = _obs(["0"] * 90 + ["1"] * 9, [f"s{i % 3}" for i in range(90)] + ["s0"] * 9)
    elif case == "dominant":  # the sample is most of the data: it fills clusters by itself
        obs = _obs(["0"] * 200 + ["1"] * 60, ["big"] * 200 + ["big"] * 30 + ["s1", "s2"] * 15)
    else:  # Kang-like: every cell type spread over 8 donors
        obs = _obs([str(i % 6) for i in range(2000)], [f"d{rng.integers(8)}" for _ in range(2000)])
    assert confounding.single_sample_clusters(obs, "leiden", "sample") == []


def test_the_run_reply_carries_the_note():
    import anndata as ad

    from scview.mcp.tools import _single_sample_note

    clusters = ["0"] * 100 + ["1"] * 100 + ["2"] * 40
    samples = [f"s{i % 3}" for i in range(200)] + ["s1"] * 40
    a = ad.AnnData(np.zeros((240, 2), dtype="float32"), obs=_obs(clusters, samples))
    a.obs_names = [f"c{i}" for i in range(240)]

    class Stub:
        adata = a

        def active_clustering_column(self):
            return "leiden"

    note = _single_sample_note(Stub())
    assert note and "2 (40 cells, 100% from s1" in note and "crosstab(leiden x sample)" in note
    assert note.endswith("or correct for sample (Harmony) before clustering.")
    # A dataset already corrected for that column is not told to correct for it.
    from scview.core import provenance

    provenance.record_step(a, step="batch_correction", tool="harmonypy", params={"batch_key": "sample"})
    note = _single_sample_note(Stub())
    assert "2 (40 cells, 100% from s1" in note and "before clustering" not in note
    assert note.endswith("The dataset is already corrected for sample (Harmony), and the correction did not "
                         "merge it: it may be cells only that sample has, or cells of low quality.")


def _counts(obs: pd.DataFrame):
    import anndata as ad

    rng = np.random.default_rng(2)
    a = ad.AnnData(rng.poisson(2.0, (len(obs), 40)).astype("float32"), obs=obs.reset_index(drop=True))
    a.obs_names = [f"c{i}" for i in range(a.n_obs)]
    a.var_names = [f"G{i}" for i in range(a.n_vars)]
    return a


def test_the_cluster_qc_report_carries_the_note_and_flags_nothing():
    """A dataset-level note (n samples, which clusters, share), never a cluster flag; the co-pilot's
    cluster QC facts name it."""
    from scview.core import cluster_qc

    clusters = ["0"] * 100 + ["1"] * 100 + ["2"] * 100 + ["3"] * 40
    samples = [f"s{i % 4}" for i in range(300)] + ["s1"] * 40
    a = _counts(_obs(clusters, samples))
    report = cluster_qc.cluster_qc(a, "leiden")
    one = report.one_sample
    assert one is not None and one.sample_column == "sample" and one.n_samples == 4
    assert [(c["cluster"], c["sample"], c["n_cells"], c["share"]) for c in one.clusters] == [("3", "s1", 40, 1.0)]
    assert one.text.startswith("1 leiden cluster is almost all one sample (of 4 in all): 3 (40 cells, 100% from s1")
    assert all("one_sample" not in c.flags for c in report.clusters)
    lines, _cites = cluster_qc.grounding_block(report)
    assert any(ln.startswith("- Clusters from one sample: 1 leiden cluster") and "[result:qc:leiden]" in ln
               for ln in lines)
    assert report.model_dump()["one_sample"]["clusters"][0]["cluster"] == "3"


@pytest.mark.parametrize("case", ["kang_like", "two_samples", "grouped_by_sample"])
def test_no_note_in_the_report_where_it_does_not_apply(case):
    from scview.core import cluster_qc

    rng = np.random.default_rng(3)
    if case == "kang_like":  # 8 donors spread over every cluster
        obs = _obs([str(i % 5) for i in range(800)], [f"d{rng.integers(8)}" for _ in range(800)])
    elif case == "two_samples":  # under SINGLE_SAMPLE_MIN_SAMPLES
        obs = _obs(["0"] * 60 + ["1"] * 60 + ["2"] * 60, ["a"] * 60 + ["b"] * 120)
    else:  # the grouping is the sample column itself
        obs = _obs([f"s{i % 4}" for i in range(400)], [f"s{i % 4}" for i in range(400)])
    a = _counts(obs)
    report = cluster_qc.cluster_qc(a, "sample" if case == "grouped_by_sample" else "leiden")
    assert report.one_sample is None
    assert not any("one sample" in ln for ln in cluster_qc.grounding_block(report)[0])


def test_the_router_sends_one_sample_questions_to_the_qc_facts():
    from scview.core.rag.router import named_result_facets

    assert "qc" in named_result_facets("Is any cluster from a single sample?")
    assert "qc" in named_result_facets("which clusters are donor-specific")
