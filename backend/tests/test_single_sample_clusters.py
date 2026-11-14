"""Clusters that are almost all one sample (core/confounding.py single_sample_clusters): named in
the MCP run_pipeline reply after a clustering."""

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
