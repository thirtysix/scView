"""A dataset without MT- genes has no measurable % mito.

Kang 2018 has none. After the pipeline's QC step wrote ``pct_counts_mt`` (all
zero, nothing to count) the QC report said "Mitochondrial reads median 0.0% ...
No major concerns", cluster QC used 0 % as a metric, the histogram was one bar
at zero, and the co-pilot answered that mito "looks fine".
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

from scview.core.anndata_adaptor import AnnDataAdaptor, per_cell_qc
from scview.core.cluster_qc import NO_MT_NOTE, cluster_qc, grounding_block
from scview.core.qc_report import build_facts


def _data(mt_genes: bool, n: int = 300, g: int = 60) -> ad.AnnData:
    rng = np.random.default_rng(0)
    a = ad.AnnData(rng.poisson(2.0, (n, g)).astype("float32"))
    names = [f"G{i}" for i in range(g)]
    if mt_genes:
        names[:3] = ["MT-CO1", "MT-ND1", "MT-ATP6"]
    a.var_names = names
    a.obs_names = [f"c{i}" for i in range(n)]
    a.obs["cl"] = pd.Categorical(np.repeat([f"k{i}" for i in range(6)], n // 6))
    # What the pipeline's QC step leaves behind.
    a.var["mt"] = a.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(a, qc_vars=["mt"], inplace=True, percent_top=None)
    return a


def test_a_column_of_zeros_without_mt_genes_is_unmeasured():
    a = _data(mt_genes=False)
    assert "pct_counts_mt" in a.obs and not a.obs["pct_counts_mt"].any()
    assert per_cell_qc(a)["has_mt"] is False


def test_real_mito_and_author_supplied_mito_are_kept():
    assert per_cell_qc(_data(mt_genes=True))["has_mt"] is True
    # Authors often drop MT genes after computing % mito: a non-zero column stays.
    a = _data(mt_genes=False)
    a.obs["pct_counts_mt"] = np.linspace(0.5, 8.0, a.n_obs)
    assert per_cell_qc(a)["has_mt"] is True


def test_qc_plots_report_and_cluster_qc_say_unmeasurable(tmp_path):
    a = _data(mt_genes=False)
    path = tmp_path / "d.h5ad"
    a.write_h5ad(path)
    adaptor = AnnDataAdaptor(str(path))

    qc = adaptor.qc_distributions()
    assert qc["has_mt"] is False
    assert "pct_counts_mt" not in qc["metrics"]
    assert qc["scatter"]["color"] == [] and qc["scatter"]["color_label"] == ""

    facts = build_facts(adaptor)
    mito = [f for f in facts.quality if f.ref == "qc:mito"]
    assert mito and "can't be measured" in mito[0].detail

    report = cluster_qc(adaptor.adata, "cl", cells=adaptor.qc_per_cell)
    assert "pct_mt" not in report.metrics and NO_MT_NOTE in report.note
    lines, _ = grounding_block(report)
    typical = next(line for line in lines if "Typical cluster" in line)
    assert "mito" not in typical  # no "0.0% mito" as if it were measured
    assert "can't be measured" in "\n".join(lines)


def test_measured_mito_still_reaches_the_plots(tmp_path):
    path = tmp_path / "d.h5ad"
    _data(mt_genes=True).write_h5ad(path)
    qc = AnnDataAdaptor(str(path)).qc_distributions()
    assert qc["has_mt"] is True and "pct_counts_mt" in qc["metrics"]
    assert len(qc["scatter"]["color"]) == qc["scatter"]["n_shown"]
