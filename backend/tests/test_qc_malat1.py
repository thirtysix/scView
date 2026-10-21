"""MALAT1 share per cell (per_cell_qc extra), the low_malat1 cluster flag, the
author's intronic share column and the MALAT1 QC histogram."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from scview.core.anndata_adaptor import AnnDataAdaptor, per_cell_qc
from scview.core.cluster_qc import (
    LOW_MALAT1_FRACTION, ClusterQC, _flag_malat1, cluster_qc, grounding_block, intronic_share,
)

N_PER = 120


def _data(*, sparse: bool, gene: str | None = "MALAT1", low_cluster: int | None = 4,
          n_groups: int = 6) -> ad.AnnData:
    """Cells x (200 genes + 5 MT- + the MALAT1 gene); ``low_cluster`` has little MALAT1."""
    rng = np.random.default_rng(0)
    blocks = []
    for k in range(n_groups):
        lam = 0.2 if k == low_cluster else 6.0
        blocks.append(np.hstack([rng.poisson(1.0, (N_PER, 200)), rng.poisson(0.3, (N_PER, 5)),
                                 rng.poisson(lam, (N_PER, 1))]))
    X = np.vstack(blocks).astype("float32")
    a = ad.AnnData(sp.csr_matrix(X) if sparse else X)
    a.var_names = [f"G{i}" for i in range(200)] + [f"MT-{i}" for i in range(5)] + [gene or "G200"]
    a.obs_names = [f"C{i}" for i in range(a.n_obs)]
    a.obs["leiden"] = pd.Categorical([str(k) for k in range(n_groups) for _ in range(N_PER)])
    a.uns["scview_active_clustering"] = "leiden"
    return a


@pytest.mark.parametrize("sparse", [True, False])
def test_per_cell_malat1_share_on_sparse_and_dense(sparse):
    a = _data(sparse=sparse)
    cells = per_cell_qc(a)  # no QC columns in obs: the core pass computes the extras too
    X = a.X.toarray() if sparse else a.X
    expect = X[:, -1] / X.sum(axis=1) * 100
    np.testing.assert_allclose(cells["pct_malat1"], expect, rtol=1e-6)
    assert cells["pct_malat1"][4 * N_PER:5 * N_PER].mean() < 0.5 < cells["pct_malat1"][:N_PER].mean()


def test_mouse_malat1_and_the_extra_pass_when_obs_has_the_core_metrics():
    a = _data(sparse=True, gene="Malat1")
    cells0 = per_cell_qc(a)
    a.obs["total_counts"] = cells0["total_counts"]
    a.obs["n_genes_by_counts"] = cells0["n_genes"]
    a.obs["pct_counts_mt"] = cells0["pct_mt"]
    assert per_cell_qc(a)["pct_malat1"] is None  # no pass needed, none asked for
    np.testing.assert_allclose(per_cell_qc(a, ribo=True)["pct_malat1"], cells0["pct_malat1"])


def test_malat1_absent_log_matrix_and_obs_column():
    assert per_cell_qc(_data(sparse=True, gene=None))["pct_malat1"] is None
    a = _data(sparse=False)
    a.X = np.log1p(a.X)  # not counts: no meaningful %
    a.obs["total_counts"] = 1.0
    a.obs["n_genes_by_counts"] = 1.0
    a.obs["pct_counts_mt"] = 1.0
    assert per_cell_qc(a, ribo=True)["pct_malat1"] is None
    a.obs["percent.malat1"] = 2.5  # an author's column is read as it is
    assert np.all(per_cell_qc(a, ribo=True)["pct_malat1"] == 2.5)


@pytest.mark.parametrize("sparse", [True, False])
def test_low_malat1_cluster_is_flagged_with_the_rule_and_numbers(sparse):
    report = cluster_qc(_data(sparse=sparse))
    rows = {c.group: c for c in report.clusters}
    assert "pct_malat1" in report.metrics
    assert rows["4"].flags == ["low_malat1"]
    d = report.dataset_median_pct_malat1
    assert d is not None and rows["4"].median_pct_malat1 < LOW_MALAT1_FRACTION * d
    reason = rows["4"].reasons[0]
    assert "Low MALAT1" in reason and "a third" in reason and f"{d:.1f}%" in reason
    for g in "01235":
        assert "low_malat1" not in rows[g].flags, rows[g].reasons
    assert "low_malat1" in report.flag_help and "10.1186/s12864-024-11015-5" in report.flag_help["low_malat1"]
    lines, _ = grounding_block(report)
    mal = [ln for ln in lines if ln.startswith("- MALAT1")]
    assert len(mal) == 1 and "4 (" in mal[0] and "[result:qc:leiden]" in mal[0]


def test_no_low_malat1_cluster_says_none_and_absent_gene_says_nothing():
    lines, _ = grounding_block(cluster_qc(_data(sparse=True, low_cluster=None)))
    assert any(ln.startswith("- MALAT1") and ln.split(": ")[-1].startswith("none") for ln in lines)
    report = cluster_qc(_data(sparse=True, gene=None))
    assert "pct_malat1" not in report.metrics and report.dataset_median_pct_malat1 is None
    assert all(c.median_pct_malat1 is None for c in report.clusters)
    assert not any(ln.startswith("- MALAT1") for ln in grounding_block(report)[0])


def test_low_malat1_needs_three_groups_and_skips_erythroid_clusters():
    report = cluster_qc(_data(sparse=True, low_cluster=1, n_groups=2))
    assert all("low_malat1" not in c.flags for c in report.clusters)
    row = ClusterQC(group="ery", n_cells=50, fraction=0.1, median_pct_malat1=0.01, erythroid=True)
    _flag_malat1(row, 2.0, compare=True)
    assert row.flags == []
    row.erythroid = False
    _flag_malat1(row, 2.0, compare=True)
    assert row.flags == ["low_malat1"]


def test_intronic_share_column_is_summarised_not_computed():
    a = _data(sparse=True)
    assert intronic_share(a.obs) is None and cluster_qc(a).intronic_column is None
    frac = np.where(a.obs["leiden"].astype(str) == "4", 0.1, 0.4)
    a.obs["fraction_unspliced"] = frac  # a 0-1 share: given as %
    col, pct = intronic_share(a.obs)
    assert col == "fraction_unspliced" and pct.max() == pytest.approx(40.0)
    report = cluster_qc(a)
    rows = {c.group: c for c in report.clusters}
    assert report.intronic_column == "fraction_unspliced" and "pct_intronic" in report.metrics
    assert rows["4"].median_pct_intronic == pytest.approx(10.0)
    assert rows["0"].median_pct_intronic == pytest.approx(40.0)
    b = a.copy()
    b.obs["pct_intronic"] = "n/a"  # not numbers: ignored, the next one is used
    assert intronic_share(b.obs)[0] == "fraction_unspliced"


def test_qc_histogram_has_malat1_when_the_gene_exists(tmp_path):
    p = tmp_path / "m.h5ad"
    _data(sparse=True).write_h5ad(p)
    qc = AnnDataAdaptor(str(p)).qc_distributions()
    assert "pct_malat1" in qc["metrics"]
    assert qc["metrics"]["pct_malat1"]["median"] > 1.0
    p2 = tmp_path / "n.h5ad"
    _data(sparse=True, gene=None).write_h5ad(p2)
    assert "pct_malat1" not in AnnDataAdaptor(str(p2)).qc_distributions()["metrics"]


def test_barely_captured_malat1_flags_nothing_and_says_why():
    """paul15 (MARS-seq): Malat1 in 0.02 % of the median cell's reads, most clusters at 0."""
    a = _data(sparse=True, gene="Malat1")
    X = a.X.toarray()
    X[:, -1] = 0
    X[::40, -1] = 1  # a read in one cell in 40
    a.X = sp.csr_matrix(X)
    report = cluster_qc(a)
    assert report.dataset_median_pct_malat1 is not None and report.dataset_median_pct_malat1 < 0.5
    assert all("low_malat1" not in c.flags for c in report.clusters)
    assert "barely captured" in report.note
    line = [ln for ln in grounding_block(report)[0] if ln.startswith("- MALAT1")][0]
    assert "barely captured" in line and "no cluster is flagged" in line
